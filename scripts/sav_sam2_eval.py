"""Evaluate SAM2 video propagation (box_centroid-style prompts) on SA-V val/test.

Unlike DAVIS, SA-V val/test store one binary mask PNG per object per annotated
frame, and objects can be empty (invisible) at some annotated frames or appear
first at a frame other than 0. Each object is therefore prompted and propagated
separately (mirroring sam2/tools/vos_inference.py's
`vos_separate_inference_per_object`), using the same box_centroid-style prompt
policy validated on DAVIS instead of feeding the ground-truth mask directly.
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
from davis_sam2_eval import load_mask, prompt_for_object  # noqa: E402

from sam2.build_sam import build_sam2_video_predictor


def list_frame_names(image_dir: Path) -> list[str]:
    return sorted(p.stem for p in image_dir.glob("*.jpg"))


def collect_object_inputs(ann_dir: Path, frame_names: list[str]) -> dict[int, dict[int, np.ndarray]]:
    """Map object_id -> {frame_idx: mask} for frames with a non-empty mask."""
    name_to_idx = {name: idx for idx, name in enumerate(frame_names)}
    inputs: dict[int, dict[int, np.ndarray]] = {}
    for object_dir in sorted(ann_dir.iterdir()):
        if not object_dir.is_dir():
            continue
        object_id = int(object_dir.name)
        per_frame = {}
        for mask_path in sorted(object_dir.glob("*.png")):
            frame_idx = name_to_idx.get(mask_path.stem)
            if frame_idx is None:
                continue
            mask = load_mask(mask_path)
            if mask.any():
                per_frame[frame_idx] = mask
        if per_frame:
            inputs[object_id] = per_frame
    return inputs


def annotated_frame_indices(ann_dir: Path, object_id: int, frame_names: list[str]) -> set[int]:
    object_dir = ann_dir / f"{object_id:03d}"
    return {
        idx
        for idx, name in enumerate(frame_names)
        if (object_dir / f"{name}.png").exists()
    }


def evaluate_video(
    predictor,
    video_dir: Path,
    ann_dir: Path,
    prompt_policy: str,
    prediction_root: Path,
    video_name: str,
):
    frame_names = list_frame_names(video_dir)
    if not frame_names:
        return None
    object_inputs = collect_object_inputs(ann_dir, frame_names)
    if not object_inputs:
        return None

    t0 = time.perf_counter()
    state = predictor.init_state(str(video_dir))
    output_dir = prediction_root / video_name
    frames_written = 0

    for object_id, per_frame in object_inputs.items():
        predictor.reset_state(state)
        first_frame_idx = min(per_frame)
        prompt = prompt_for_object(per_frame[first_frame_idx], prompt_policy)
        predictor.add_new_points_or_box(
            state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
        )
        wanted_frames = annotated_frame_indices(ann_dir, object_id, frame_names)
        obj_out_dir = output_dir / f"{object_id:03d}"
        obj_out_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, out_obj_ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=first_frame_idx
        ):
            if frame_idx not in wanted_frames:
                continue
            predicted = (mask_logits[0, 0] > 0).detach().cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(predicted).save(obj_out_dir / f"{frame_names[frame_idx]}.png")
            frames_written += 1

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "video": video_name,
        "objects": len(object_inputs),
        "frames_written": frames_written,
        "video_frames": len(frame_names),
        "latency_ms": round(elapsed_ms, 1),
    }


def video_already_done(video_dir: Path, ann_dir: Path, prediction_root: Path, video_name: str) -> bool:
    """True if every wanted mask frame for every object in this video is already
    on disk -- lets a killed/crashed multi-hour full-dataset run resume without
    redoing already-written videos (see sam2-drift-recovery-results.md, the
    2026-08-31 host-sleep crash and later CUDA hangs during this project)."""
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
    parser.add_argument("--video-list", type=Path, default=None, help="default: {sav-root}/sav_val.txt")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--prompt-policy", default="box_centroid")
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
        if video_already_done(video_dir, ann_dir, args.prediction_root, video_name):
            line = f"[{i + 1}/{len(video_names)}] video={video_name} already complete, skipping"
            print(line, flush=True)
            log_lines.append(line)
            continue
        row = evaluate_video(
            predictor, video_dir, ann_dir, args.prompt_policy, args.prediction_root, video_name
        )
        if row is None:
            continue
        line = f"[{i + 1}/{len(video_names)}] " + " | ".join(f"{k}={v}" for k, v in row.items())
        print(line, flush=True)
        log_lines.append(line)
        if args.progress_log:
            args.progress_log.write_text("\n".join(log_lines) + "\n")

    print(f"Done: wrote predictions for {len(video_names)} videos to {args.prediction_root}")


if __name__ == "__main__":
    main()
