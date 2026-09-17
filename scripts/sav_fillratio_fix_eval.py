"""Evaluate a candidate prompt policy on just the low-box_fill_ratio SA-V objects.

P2 from sam2-drift-recovery-results.md: the OTHER (non-drift) SA-V failure mode
from sam2-sav-failure-analysis.md iteration #29 -- objects whose ground-truth
mask is sparse/fragmented relative to its own bounding box (box_fill_ratio<0.2,
n=13/293) collapse even on the prompt frame, because box_centroid's tight box is
built from the full mask's bounding box, which balloons around minor fragments
or an off-mask centroid. Independent of the propagation-drift line of work.

Targets are read directly from results/sav_val_failure_mode_analysis.csv rather
than hardcoded, so this stays in sync with the iteration #29 analysis.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sav_sam2_eval import (  # noqa: E402
    annotated_frame_indices,
    collect_object_inputs,
    evaluate_video,
    list_frame_names,
)

from sam2.build_sam import build_sam2_video_predictor
import torch  # noqa: E402


def load_low_fillratio_targets(analysis_csv: Path, threshold: float) -> dict[str, set[int]]:
    targets: dict[str, set[int]] = {}
    with analysis_csv.open() as f:
        for row in csv.DictReader(f):
            if float(row["box_fill_ratio"]) < threshold:
                targets.setdefault(row["video"], set()).add(int(row["object"]))
    return targets


def video_already_done(video_dir: Path, ann_dir: Path, prediction_root: Path, video_name: str) -> bool:
    """True if every wanted mask frame for this video is already on disk.

    A prior run can be killed mid-video (see 2026-08-31 WSL/host suspend
    crash), leaving a partially-written prediction dir. Re-running
    evaluate_video on such a video just overwrites the existing frames and
    fills in the rest, so we only need to skip videos that are fully done.
    """
    frame_names = list_frame_names(video_dir)
    if not frame_names:
        return True
    object_inputs = collect_object_inputs(ann_dir, frame_names)
    if not object_inputs:
        return True
    output_dir = prediction_root / video_name
    for object_id in object_inputs:
        wanted_frames = annotated_frame_indices(ann_dir, object_id, frame_names)
        obj_out_dir = output_dir / f"{object_id:03d}"
        for frame_idx in wanted_frames:
            if not (obj_out_dir / f"{frame_names[frame_idx]}.png").exists():
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--analysis-csv", type=Path, default=Path("results/sav_val_failure_mode_analysis.csv"))
    parser.add_argument("--fill-ratio-threshold", type=float, default=0.2)
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--prediction-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    targets = load_low_fillratio_targets(args.analysis_csv, args.fill_ratio_threshold)
    total_objects = sum(len(v) for v in targets.values())
    print(f"Loaded {total_objects} low-fill-ratio objects across {len(targets)} videos")

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")

    for i, (video_name, object_ids) in enumerate(sorted(targets.items())):
        video_dir = args.sav_root / "JPEGImages_24fps" / video_name
        ann_dir = args.sav_root / "Annotations_6fps" / video_name
        if video_already_done(video_dir, ann_dir, args.prediction_root, video_name):
            print(f"[{i + 1}/{len(targets)}] video={video_name} already complete, skipping", flush=True)
            continue
        row = evaluate_video(
            predictor, video_dir, ann_dir, args.prompt_policy, args.prediction_root, video_name
        )
        print(f"[{i + 1}/{len(targets)}] video={video_name} objects_in_video={object_ids} row={row}", flush=True)

    print(f"Done: wrote predictions to {args.prediction_root}")


if __name__ == "__main__":
    main()
