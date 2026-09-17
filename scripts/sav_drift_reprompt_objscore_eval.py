"""No-oracle drift-detection using SAM2's own object_score_logits (candidate: separate signal).

Follow-up to scripts/sav_drift_reprompt_eval.py and the threshold sweep in
sam2-sav-failure-analysis.md, which showed that tuning the area-ratio-vs-EMA
trigger (looser or tighter) cannot break past ~24-27 J&F on the 30 drift-failure
objects (vs 84.3 for the oracle ceiling): area is a single, self-referential
signal, so a bad correction corrupts the very quantity used to detect the next
one.

This variant swaps the trigger for object_score_logits, a per-frame confidence
SAM2 already computes internally for occlusion handling (`is_obj_appearing =
object_score_logits > 0`, see sam2/sam2_video_predictor.py). It is decoupled
from mask area/shape, so a bad correction cannot directly bias it the way a bad
mask biases the next area computation.

Access note: propagate_in_video's public generator only yields (frame_idx,
obj_ids, video_res_masks) -- it computes object_score_logits internally but
does not expose it. This script reads it straight out of the same
output_dict_per_obj / cond_frame_outputs|non_cond_frame_outputs structure that
add_new_mask and add_new_points_or_box already read and write (see their
implementations in sam2_video_predictor.py) -- no private methods are called,
only the same state introspected after the public API call.

Caveat: object_score_logits measures "is the tracked object present", not "is
this correct" -- a confident identity switch onto a similar-looking distractor
would not be flagged by this signal. It is a different failure surface from the
area-ratio proxy, not a strictly better detector.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from davis_sam2_eval import prompt_for_object  # noqa: E402
from sav_reprompt_eval import load_drift_failure_targets  # noqa: E402
from sav_sam2_eval import (  # noqa: E402
    annotated_frame_indices,
    collect_object_inputs,
    list_frame_names,
)

from sam2.build_sam import build_sam2_video_predictor


def read_object_score_logit(state, obj_idx: int, frame_idx: int) -> float:
    obj_output_dict = state["output_dict_per_obj"][obj_idx]
    entry = obj_output_dict["cond_frame_outputs"].get(frame_idx)
    if entry is None:
        entry = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
    logits = entry["object_score_logits"]
    return float(logits.reshape(-1)[0].item())


def evaluate_object_with_objscore_reprompt(
    predictor,
    state,
    object_id: int,
    per_frame: dict[int, np.ndarray],
    frame_names: list[str],
    ann_dir: Path,
    prompt_policy: str,
    obj_out_dir: Path,
    score_threshold: float,
):
    sorted_frames = sorted(per_frame)
    first_frame_idx = sorted_frames[0]
    predictor.reset_state(state)
    obj_idx = predictor._obj_id_to_idx(state, object_id)

    prompt = prompt_for_object(per_frame[first_frame_idx], prompt_policy)
    _, _, mask_logits = predictor.add_new_points_or_box(
        state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
    )
    last_stable_mask = (mask_logits[0, 0] > 0)

    wanted_frames = annotated_frame_indices(ann_dir, object_id, frame_names)
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    if first_frame_idx in wanted_frames:
        arr = last_stable_mask.detach().cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(arr).save(obj_out_dir / f"{frame_names[first_frame_idx]}.png")

    frames_written = 1 if first_frame_idx in wanted_frames else 0
    n_corrections = 0
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
            is_drift = score < score_threshold
            if is_drift:
                drift_frame_idx = frame_idx
                break
            last_stable_mask = mask

        if drift_frame_idx is None:
            break  # propagation reached the end of the video without further drift

        _, _, corrected_logits = predictor.add_new_mask(
            state, frame_idx=drift_frame_idx, obj_id=object_id, mask=last_stable_mask
        )
        corrected_mask = corrected_logits[0, 0] > 0
        if drift_frame_idx in wanted_frames:
            arr = corrected_mask.detach().cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(arr).save(obj_out_dir / f"{frame_names[drift_frame_idx]}.png")
        last_stable_mask = corrected_mask
        n_corrections += 1
        current_start = drift_frame_idx + 1

    return frames_written, n_corrections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--analysis-csv", type=Path, default=Path("results/sav_val_failure_mode_analysis.csv"))
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--prediction-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    targets = load_drift_failure_targets(args.analysis_csv)
    total_objects = sum(len(v) for v in targets.values())
    print(f"Loaded {total_objects} drift-failure objects across {len(targets)} videos")

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")

    for i, (video_name, object_ids) in enumerate(sorted(targets.items())):
        video_dir = args.sav_root / "JPEGImages_24fps" / video_name
        ann_dir = args.sav_root / "Annotations_6fps" / video_name
        frame_names = list_frame_names(video_dir)
        object_inputs = collect_object_inputs(ann_dir, frame_names)

        t0 = time.perf_counter()
        state = predictor.init_state(str(video_dir))
        for object_id in sorted(object_ids):
            if object_id not in object_inputs:
                continue
            obj_out_dir = args.prediction_root / video_name / f"{object_id:03d}"
            frames_written, n_corrections = evaluate_object_with_objscore_reprompt(
                predictor, state, object_id, object_inputs[object_id],
                frame_names, ann_dir, args.prompt_policy, obj_out_dir,
                args.score_threshold,
            )
            print(
                f"[{i + 1}/{len(targets)}] video={video_name} obj={object_id:03d} "
                f"frames_written={frames_written} self_corrections={n_corrections}",
                flush=True,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  video total latency_ms={elapsed_ms:.1f}", flush=True)

    print(f"Done: wrote predictions for {total_objects} objects to {args.prediction_root}")


if __name__ == "__main__":
    main()
