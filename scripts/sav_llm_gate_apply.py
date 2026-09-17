"""Apply P4: LLM as a veto/gate on self-mask drift correction.

Consumes the manifest produced by sav_llm_gate_collect.py (each flagged frame
reviewed and marked CONFIRM or DENY by an external -- here, Claude -- reviewer
looking at the current mask overlay + the reference crop). Uses the same
object_score_logits<0 trigger as the collection pass, so the set of flagged
frames is identical to sav_drift_reprompt_objscore_eval.py's -- only the
correction *policy* differs:

  CONFIRM -> apply the same self-mask reinjection as the always-correct
             baseline (add_new_mask with the last stable mask).
  DENY    -> skip the correction entirely; let the model's own (uncorrected,
             here empty) prediction stand and keep propagating.

This isolates the one variable under test: does gating self-mask correction
on an external confirmation (reducing false-positive corrections) beat always
correcting at the same trigger points (objscore-only baseline, J&F=18.5) or
replacing the correction source with an external box (J&F=10.2)?
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from davis_sam2_eval import prompt_for_object  # noqa: E402
from sav_drift_reprompt_objscore_eval import read_object_score_logit  # noqa: E402
from sav_llm_verify_collect import PILOT  # noqa: E402
from sav_reprompt_eval import load_drift_failure_targets  # noqa: E402
from sav_sam2_eval import (  # noqa: E402
    annotated_frame_indices,
    collect_object_inputs,
    list_frame_names,
)

from sam2.build_sam import build_sam2_video_predictor


def load_manifest_verdicts(manifest_path: Path) -> dict[tuple[str, str, int], str]:
    verdicts = {}
    with manifest_path.open() as f:
        for row in csv.DictReader(f):
            verdict = row["verdict"].strip().upper()
            if verdict not in ("CONFIRM", "DENY"):
                continue
            key = (row["video"], row["object"], int(row["frame_idx"]))
            verdicts[key] = verdict
    return verdicts


def evaluate_object(
    predictor, state, object_id: int, per_frame: dict[int, np.ndarray],
    frame_names: list[str], ann_dir: Path, prompt_policy: str, obj_out_dir: Path,
    video_name: str, verdicts: dict,
):
    sorted_frames = sorted(per_frame)
    first_frame_idx = sorted_frames[0]
    predictor.reset_state(state)

    prompt = prompt_for_object(per_frame[first_frame_idx], prompt_policy)
    _, _, mask_logits = predictor.add_new_points_or_box(
        state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
    )
    obj_idx = predictor._obj_id_to_idx(state, object_id)
    last_stable_mask = mask_logits[0, 0] > 0

    wanted_frames = annotated_frame_indices(ann_dir, object_id, frame_names)
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    if first_frame_idx in wanted_frames:
        arr = last_stable_mask.detach().cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(arr).save(obj_out_dir / f"{frame_names[first_frame_idx]}.png")

    frames_written = 1 if first_frame_idx in wanted_frames else 0
    n_confirmed, n_denied, n_unreviewed = 0, 0, 0
    current_start = first_frame_idx + 1

    while current_start < len(frame_names):
        drift_frame_idx = None
        drift_mask = None
        for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=current_start
        ):
            mask = mask_logits[0, 0] > 0
            if frame_idx in wanted_frames:
                arr = mask.detach().cpu().numpy().astype(np.uint8) * 255
                Image.fromarray(arr).save(obj_out_dir / f"{frame_names[frame_idx]}.png")
                frames_written += 1
            score = read_object_score_logit(state, obj_idx, frame_idx)
            if score < 0.0:
                drift_frame_idx = frame_idx
                drift_mask = mask
                break
            last_stable_mask = mask

        if drift_frame_idx is None:
            break

        key = (video_name, f"{object_id:03d}", drift_frame_idx)
        verdict = verdicts.get(key)
        if verdict == "DENY":
            # Explicitly reviewed and rejected: leave the model's own
            # prediction standing, no reinjection.
            corrected_mask = drift_mask
            n_denied += 1
        else:
            # CONFIRM, or an event past the reviewed manifest (collection was
            # capped at --max-events): fall back to the same self-mask
            # correction as the always-correct baseline, so an unreviewed
            # tail of the video isn't silently starved of any correction --
            # only frames a reviewer actually looked at and rejected are
            # denied.
            _, _, corrected_logits = predictor.add_new_mask(
                state, frame_idx=drift_frame_idx, obj_id=object_id, mask=last_stable_mask
            )
            corrected_mask = corrected_logits[0, 0] > 0
            if verdict == "CONFIRM":
                n_confirmed += 1
            else:
                n_unreviewed += 1

        if drift_frame_idx in wanted_frames:
            arr = corrected_mask.detach().cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(arr).save(obj_out_dir / f"{frame_names[drift_frame_idx]}.png")
        last_stable_mask = corrected_mask
        current_start = drift_frame_idx + 1

    return frames_written, n_confirmed, n_denied, n_unreviewed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--manifest", type=Path, default=Path("results/sav_llm_gate/manifest.csv"))
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument(
        "--all-drift-failures", action="store_true",
        help="Use the full n=30 drift-failure subset instead of the 6-object PILOT.",
    )
    parser.add_argument("--analysis-csv", type=Path, default=Path("results/sav_val_failure_mode_analysis.csv"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    if args.all_drift_failures:
        targets = load_drift_failure_targets(args.analysis_csv)
        pilot = sorted((v, o) for v, objs in targets.items() for o in objs)
    else:
        pilot = PILOT

    verdicts = load_manifest_verdicts(args.manifest)
    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")

    for video_name, object_id in pilot:
        video_dir = args.sav_root / "JPEGImages_24fps" / video_name
        ann_dir = args.sav_root / "Annotations_6fps" / video_name
        frame_names = list_frame_names(video_dir)
        object_inputs = collect_object_inputs(ann_dir, frame_names)
        state = predictor.init_state(str(video_dir))
        obj_out_dir = args.prediction_root / video_name / f"{object_id:03d}"
        frames_written, n_confirmed, n_denied, n_unreviewed = evaluate_object(
            predictor, state, object_id, object_inputs[object_id], frame_names,
            ann_dir, args.prompt_policy, obj_out_dir, video_name, verdicts,
        )
        print(
            f"video={video_name} obj={object_id:03d} frames_written={frames_written} "
            f"confirmed_corrections={n_confirmed} denied={n_denied} unreviewed_fallback={n_unreviewed}",
            flush=True,
        )

    print(f"Done: wrote predictions to {args.prediction_root}")


if __name__ == "__main__":
    main()
