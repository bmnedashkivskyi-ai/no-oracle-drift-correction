"""Combined P2 + P5 pipeline on the FULL SA-V dataset (all videos, all objects) --
not a curated failure subset like the individual P2/P5 experiments used.

P2 (largest-connected-component box construction) turns out to need no
fill-ratio gate at all: `davis_sam2_eval.largest_component()` returns the mask
unchanged when it already has a single connected component (`if n <= 1: return
mask`), so `largest_component_box_centroid` is identical to `box_centroid` for
every already-compact object and only differs for fragmented ones. It is
therefore safe to use it as the prompt policy for ALL objects, not just the
box_fill_ratio<0.2 subset P2 was originally scoped to -- this also catches any
fragmented-but-not-low-fill-ratio object the original scoping missed.

P5 (motion/position-prior drift correction, sam2-drift-recovery-results.md
SS10) is applied during propagation for every object via the same
evaluate_object_with_motion_reprompt used by sav_motion_prior_reprompt_eval.py,
imported unchanged -- this script only supplies a different (full-dataset,
not drift-failure-subset) object loop and the largest_component_box_centroid
prompt policy.

This answers a question neither the P2 nor the P5 experiments (each scored on
their own curated subset) could: what is the net effect on the OFFICIAL SA-V
J&F reported for the whole benchmark (72.1 on val, 74.4 on test) once both
fixes are combined into one inference pipeline? Resume-safe like
sav_fillratio_fix_eval.py: skips any object whose predicted masks are already
complete on disk, since a full-dataset run is a multi-hour operation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sav_motion_prior_reprompt_eval import evaluate_object_with_motion_reprompt  # noqa: E402
from sav_sam2_eval import annotated_frame_indices, collect_object_inputs, list_frame_names  # noqa: E402

from sam2.build_sam import build_sam2_video_predictor


def object_already_done(video_dir: Path, ann_dir: Path, object_id: int, prediction_root: Path, video_name: str) -> bool:
    frame_names = list_frame_names(video_dir)
    wanted_frames = annotated_frame_indices(ann_dir, object_id, frame_names)
    if not wanted_frames:
        return True
    obj_out_dir = prediction_root / video_name / f"{object_id:03d}"
    for frame_idx in wanted_frames:
        if not (obj_out_dir / f"{frame_names[frame_idx]}.png").exists():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--video-list", type=Path, default=None, help="default: {sav-root}/{sav-root.name}.txt")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--prompt-policy", default="largest_component_box_centroid")
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--low-ratio", type=float, default=0.35)
    parser.add_argument("--high-ratio", type=float, default=2.5)
    parser.add_argument("--min-area-eps", type=float, default=50.0)
    parser.add_argument("--jump-ratio", type=float, default=8.0)
    parser.add_argument("--max-corrections", type=int, default=60)
    parser.add_argument("--vanish-retry-limit", type=int, default=10)
    parser.add_argument("--require-both-triggers", action="store_true")
    parser.add_argument("--plausibility-ratio", type=float, default=0.0)
    parser.add_argument("--max-correction-rate", type=float, default=0.0)
    parser.add_argument("--correction-window", type=int, default=0)
    parser.add_argument("--early-trigger-skip-frac", type=float, default=0.0)
    parser.add_argument("--rate-warmup-corrections", type=int, default=3)
    parser.add_argument("--soft-nudge-weight", type=float, default=0.0)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--progress-log", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    video_list_path = args.video_list or (args.sav_root / f"{args.sav_root.name}.txt")
    video_names = [line.strip() for line in video_list_path.read_text().splitlines() if line.strip()]
    if args.max_videos:
        video_names = video_names[: args.max_videos]

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")

    log_lines = []
    for i, video_name in enumerate(video_names):
        video_dir = args.sav_root / "JPEGImages_24fps" / video_name
        ann_dir = args.sav_root / "Annotations_6fps" / video_name
        if not video_dir.exists() or not ann_dir.exists():
            continue
        frame_names = list_frame_names(video_dir)
        object_inputs = collect_object_inputs(ann_dir, frame_names)
        if not object_inputs:
            continue

        t0 = time.perf_counter()
        state = predictor.init_state(str(video_dir))
        for object_id, per_frame in sorted(object_inputs.items()):
            if object_already_done(video_dir, ann_dir, object_id, args.prediction_root, video_name):
                line = f"[{i + 1}/{len(video_names)}] video={video_name} obj={object_id:03d} already complete, skipping"
                print(line, flush=True)
                log_lines.append(line)
                continue
            obj_out_dir = args.prediction_root / video_name / f"{object_id:03d}"
            frames_written, n_corrections, n_position_triggers, n_skipped_vanish, n_rejected_corrections, n_skipped_early_trigger = evaluate_object_with_motion_reprompt(
                predictor, state, object_id, per_frame, frame_names, ann_dir,
                args.prompt_policy, obj_out_dir,
                args.ema_alpha, args.low_ratio, args.high_ratio,
                args.min_area_eps, args.jump_ratio, args.max_corrections,
                args.vanish_retry_limit, args.require_both_triggers,
                args.plausibility_ratio, args.max_correction_rate,
                args.correction_window, args.early_trigger_skip_frac,
                args.rate_warmup_corrections, args.soft_nudge_weight,
            )
            line = (
                f"[{i + 1}/{len(video_names)}] video={video_name} obj={object_id:03d} "
                f"frames_written={frames_written} corrections={n_corrections} "
                f"position_triggers={n_position_triggers} skipped_vanish={n_skipped_vanish} "
                f"rejected_corrections={n_rejected_corrections} skipped_early_trigger={n_skipped_early_trigger}"
            )
            print(line, flush=True)
            log_lines.append(line)
            if args.progress_log:
                args.progress_log.write_text("\n".join(log_lines) + "\n")
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  video total latency_ms={elapsed_ms:.1f}", flush=True)

    print(f"Done: wrote predictions for {len(video_names)} videos to {args.prediction_root}")


if __name__ == "__main__":
    main()
