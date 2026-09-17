"""Dynamic memory-gating on the FULL SA-V val population (all videos, all
objects) -- not the curated n=30 known-drift subset used by the Task 2
pilot (sav_memory_gate_eval.py). Mirrors sav_full_pipeline_eval.py's
structure exactly (same all-objects loop via collect_object_inputs, same
object_already_done resumability pattern for a multi-hour run), but wraps
evaluate_object_with_memory_gate (Task 2) instead of
evaluate_object_with_motion_reprompt, and uses plain box_centroid as the
default prompt policy -- not largest_component_box_centroid -- since this
experiment isolates memory-gating's own effect against the plain official
baseline, unlike sav_full_pipeline_eval.py which deliberately combines P2
(largest-component box) and P5 (motion reprompt).

This answers the same full-vs-curated-subset question this project's
paper treats as essential for every mechanism: the n=30 pilot cleared its
pre-committed gate decisively (J&F 39.4 vs baseline 26.1 at
divergence-threshold 0.2, worst per-object regression -1.7pp), but a
curated-subset gain does not by itself establish a full-population gain
(the paper's central finding for the original P5 mechanism). --divergence-
threshold has no default: the pilot-winning value must be passed
explicitly at call time since it is an empirically-chosen result, not a
fixed design constant.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sav_memory_gate_eval import evaluate_object_with_memory_gate  # noqa: E402
from sav_sam2_eval import annotated_frame_indices, collect_object_inputs, list_frame_names  # noqa: E402
from sam2_multimask_capture import install  # noqa: E402
from sam2_memory_gate import install_memory_gate  # noqa: E402

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
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--divergence-threshold", type=float, required=True)
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
    capture = install(predictor)
    gate = install_memory_gate(predictor, capture, args.divergence_threshold)

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
            frames_written, n_checked, n_skipped = evaluate_object_with_memory_gate(
                predictor, state, gate, object_id, per_frame, frame_names,
                ann_dir, args.prompt_policy, obj_out_dir,
            )
            line = (
                f"[{i + 1}/{len(video_names)}] video={video_name} obj={object_id:03d} "
                f"frames_written={frames_written} n_checked={n_checked} n_skipped={n_skipped}"
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
