"""Diagnostic: does periodic oracle re-prompting recover SA-V propagation-drift failures?

Targets exactly the "propagation-drift failure" objects identified in
sam2-sav-failure-analysis.md (results/sav_val_failure_mode_analysis.csv,
jf < 40 and prompt_frame_iou >= 0.3 for box_centroid's single-prompt run):
objects where the box_centroid prompt was fine on the first frame but the
tracked mask degraded later in the video.

This is an ORACLE ceiling estimate, not a deployable inference-time policy: every
--reprompt-every-th annotated frame gets a *fresh* box_centroid prompt derived from
the ground-truth mask at that frame (same derivation as the initial frame), then
propagation continues from there. It answers one question before building a real
no-oracle drift detector: is the degradation recoverable by occasional correction
at all, or are these objects hard for SAM2 regardless of prompting?
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from davis_sam2_eval import prompt_for_object  # noqa: E402
from sav_sam2_eval import (  # noqa: E402
    annotated_frame_indices,
    collect_object_inputs,
    list_frame_names,
)

from sam2.build_sam import build_sam2_video_predictor


def load_drift_failure_targets(analysis_csv: Path) -> dict[str, set[int]]:
    targets: dict[str, set[int]] = {}
    with analysis_csv.open() as f:
        for row in csv.DictReader(f):
            jf = float(row["jf"])
            iou = row["prompt_frame_iou"]
            if iou == "" or jf >= 40 or float(iou) < 0.3:
                continue
            targets.setdefault(row["video"], set()).add(int(row["object"]))
    return targets


def evaluate_object_with_reprompt(
    predictor,
    state,
    video_name: str,
    object_id: int,
    per_frame: dict[int, np.ndarray],
    frame_names: list[str],
    ann_dir: Path,
    prompt_policy: str,
    reprompt_every: int,
    obj_out_dir: Path,
):
    sorted_frames = sorted(per_frame)
    first_frame_idx = sorted_frames[0]
    predictor.reset_state(state)

    prompt = prompt_for_object(per_frame[first_frame_idx], prompt_policy)
    predictor.add_new_points_or_box(state, frame_idx=first_frame_idx, obj_id=object_id, **prompt)

    reprompt_frames = sorted_frames[reprompt_every::reprompt_every]
    segment_starts = [first_frame_idx] + reprompt_frames
    wanted_frames = annotated_frame_indices(ann_dir, object_id, frame_names)
    obj_out_dir.mkdir(parents=True, exist_ok=True)

    frames_written = 0
    for i, seg_start in enumerate(segment_starts):
        if i > 0:
            seg_prompt = prompt_for_object(per_frame[seg_start], prompt_policy)
            predictor.add_new_points_or_box(state, frame_idx=seg_start, obj_id=object_id, **seg_prompt)
        if i + 1 < len(segment_starts):
            seg_end = segment_starts[i + 1] - 1
            max_frames = seg_end - seg_start
        else:
            max_frames = None
        for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=seg_start, max_frame_num_to_track=max_frames
        ):
            if frame_idx not in wanted_frames:
                continue
            predicted = (mask_logits[0, 0] > 0).detach().cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(predicted).save(obj_out_dir / f"{frame_names[frame_idx]}.png")
            frames_written += 1

    return frames_written, len(reprompt_frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--analysis-csv", type=Path, default=Path("results/sav_val_failure_mode_analysis.csv"))
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--reprompt-every", type=int, default=5, help="Re-prompt every Nth annotated frame")
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
            frames_written, n_reprompts = evaluate_object_with_reprompt(
                predictor, state, video_name, object_id, object_inputs[object_id],
                frame_names, ann_dir, args.prompt_policy, args.reprompt_every, obj_out_dir,
            )
            print(
                f"[{i + 1}/{len(targets)}] video={video_name} obj={object_id:03d} "
                f"frames_written={frames_written} reprompts={n_reprompts}",
                flush=True,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  video total latency_ms={elapsed_ms:.1f}", flush=True)

    print(f"Done: wrote predictions for {total_objects} objects to {args.prediction_root}")


if __name__ == "__main__":
    main()
