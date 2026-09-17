"""Apply externally-sourced (LLM/human-reviewed) corrective boxes from the manifest
produced by scripts/sav_llm_verify_collect.py, falling back to self-mask reprompt
(same as candidate #1) at any flagged frame the reviewer abstained on.

Uses the same object_score_logits<0 trigger as the collection pass, so the set of
flagged frames is identical -- only the correction source differs. This isolates
the one variable under test: does an externally-sourced box (independent of
SAM2's own internal state, unlike self-mask reinjection) do better than self-
referential correction at the same trigger points?
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
from sav_sam2_eval import (  # noqa: E402
    annotated_frame_indices,
    collect_object_inputs,
    list_frame_names,
)

from sam2.build_sam import build_sam2_video_predictor


def load_manifest_boxes(manifest_path: Path) -> dict[tuple[str, str, int], tuple[float, float, float, float]]:
    boxes = {}
    with manifest_path.open() as f:
        for row in csv.DictReader(f):
            if row["box_x0"].strip() == "":
                continue
            key = (row["video"], row["object"], int(row["frame_idx"]))
            boxes[key] = (
                float(row["box_x0"]), float(row["box_y0"]),
                float(row["box_x1"]), float(row["box_y1"]),
            )
    return boxes


def evaluate_object(
    predictor, state, object_id: int, per_frame: dict[int, np.ndarray],
    frame_names: list[str], ann_dir: Path, prompt_policy: str, obj_out_dir: Path,
    video_name: str, manual_boxes: dict,
):
    sorted_frames = sorted(per_frame)
    first_frame_idx = sorted_frames[0]
    predictor.reset_state(state)
    obj_idx = None

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
    n_external, n_self = 0, 0
    current_start = first_frame_idx + 1

    while current_start < len(frame_names):
        drift_frame_idx = None
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
                break
            last_stable_mask = mask

        if drift_frame_idx is None:
            break

        box_key = (video_name, f"{object_id:03d}", drift_frame_idx)
        if box_key in manual_boxes:
            x0, y0, x1, y1 = manual_boxes[box_key]
            _, _, corrected_logits = predictor.add_new_points_or_box(
                state, frame_idx=drift_frame_idx, obj_id=object_id, box=[x0, y0, x1, y1]
            )
            n_external += 1
        else:
            _, _, corrected_logits = predictor.add_new_mask(
                state, frame_idx=drift_frame_idx, obj_id=object_id, mask=last_stable_mask
            )
            n_self += 1
        corrected_mask = corrected_logits[0, 0] > 0
        if drift_frame_idx in wanted_frames:
            arr = corrected_mask.detach().cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(arr).save(obj_out_dir / f"{frame_names[drift_frame_idx]}.png")
        last_stable_mask = corrected_mask
        current_start = drift_frame_idx + 1

    return frames_written, n_external, n_self


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--manifest", type=Path, default=Path("results/sav_llm_verify/manifest.csv"))
    parser.add_argument("--prediction-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    manual_boxes = load_manifest_boxes(args.manifest)
    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")

    for video_name, object_id in PILOT:
        video_dir = args.sav_root / "JPEGImages_24fps" / video_name
        ann_dir = args.sav_root / "Annotations_6fps" / video_name
        frame_names = list_frame_names(video_dir)
        object_inputs = collect_object_inputs(ann_dir, frame_names)
        state = predictor.init_state(str(video_dir))
        obj_out_dir = args.prediction_root / video_name / f"{object_id:03d}"
        frames_written, n_external, n_self = evaluate_object(
            predictor, state, object_id, object_inputs[object_id], frame_names,
            ann_dir, args.prompt_policy, obj_out_dir, video_name, manual_boxes,
        )
        print(
            f"video={video_name} obj={object_id:03d} frames_written={frames_written} "
            f"external_corrections={n_external} self_corrections={n_self}",
            flush=True,
        )

    print(f"Done: wrote predictions to {args.prediction_root}")


if __name__ == "__main__":
    main()
