"""Evaluate SAM2 video propagation on a DAVIS 2017 subset."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.uint8)


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot create a prompt for an empty object mask")
    return float(xs.mean()), float(ys.mean())


def bounding_box(mask: np.ndarray, expansion: float = 0.0) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot create a box prompt for an empty object mask")
    height, width = mask.shape
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_min = max(0.0, x_min - expansion * box_width)
    x_max = min(width - 1.0, x_max + expansion * box_width)
    y_min = max(0.0, y_min - expansion * box_height)
    y_max = min(height - 1.0, y_max + expansion * box_height)
    return [x_min, y_min, x_max, y_max]


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest connected component of `mask`."""
    from scipy import ndimage

    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return labeled == largest_label


def on_mask_point(mask: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    """Snap `point` onto the nearest foreground pixel of `mask` if it isn't already on it."""
    x, y = int(round(point[0])), int(round(point[1]))
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
        return point
    ys, xs = np.nonzero(mask)
    dists = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
    nearest = np.argmin(dists)
    return float(xs[nearest]), float(ys[nearest])


def prompt_for_object(mask: np.ndarray, policy: str):
    point = centroid(mask)
    box = bounding_box(mask)
    if policy == "largest_component_box_centroid":
        component = largest_component(mask)
        component_point = on_mask_point(component, centroid(component))
        return {"box": bounding_box(component), "points": [component_point], "labels": [1]}
    if policy == "centroid":
        return {"points": [point], "labels": [1]}
    if policy == "box":
        return {"box": box}
    if policy.startswith("box_expand_") and policy.removeprefix("box_expand_").replace(".", "", 1).isdigit():
        expansion = float(policy.removeprefix("box_expand_")) / 100
        return {"box": bounding_box(mask, expansion)}
    if policy in {"two_positive", "three_positive"}:
        ys, xs = np.nonzero(mask)
        quantiles = [0.25, 0.5, 0.75] if policy == "three_positive" else [0.5, 0.75]
        points = [(float(np.quantile(xs, q)), float(np.quantile(ys, q))) for q in quantiles]
        return {"points": points, "labels": [1] * len(points)}
    if policy == "box_centroid":
        return {"box": box, "points": [point], "labels": [1]}
    if policy in {"box_positive_negative", "box_expand_10_negative"}:
        if policy == "box_expand_10_negative":
            box = bounding_box(mask, 0.10)
        height, width = mask.shape
        negative = (max(0.0, box[0] - 20.0), max(0.0, box[1] - 20.0))
        if mask[int(negative[1]), int(negative[0])]:
            negative = (min(width - 1.0, box[2] + 20.0), min(height - 1.0, box[3] + 20.0))
        return {"box": box, "points": [point, negative], "labels": [1, 0]}
    if policy == "box_two_positive":
        ys, xs = np.nonzero(mask)
        points = [point, (float(np.quantile(xs, 0.75)), float(np.quantile(ys, 0.75)))]
        return {"box": box, "points": points, "labels": [1, 1]}
    raise ValueError(f"Unknown prompt policy: {policy}")


def erode(mask: np.ndarray) -> np.ndarray:
    result = np.ones_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(mask, (dy, dx), axis=(0, 1))
            if dy == -1:
                shifted[-1, :] = False
            elif dy == 1:
                shifted[0, :] = False
            if dx == -1:
                shifted[:, -1] = False
            elif dx == 1:
                shifted[:, 0] = False
            result &= shifted
    return result


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(radius):
        result = ~erode(~result)
    return result


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    return mask & ~erode(mask)


def boundary_f_score(pred: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    pred_boundary = boundary(pred)
    target_boundary = boundary(target)
    if not pred_boundary.any() and not target_boundary.any():
        return 1.0
    if not pred_boundary.any() or not target_boundary.any():
        return 0.0
    precision = (pred_boundary & dilate(target_boundary, tolerance)).sum() / pred_boundary.sum()
    recall = (target_boundary & dilate(pred_boundary, tolerance)).sum() / target_boundary.sum()
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate_sequence(
    predictor,
    davis_root: Path,
    sequence: str,
    max_frames: int | None,
    keep_video_on_gpu: bool,
    prompt_policy: str,
    prediction_root: Path | None,
):
    image_dir = davis_root / "JPEGImages" / "480p" / sequence
    annotation_dir = davis_root / "Annotations" / "480p" / sequence
    image_paths = sorted(image_dir.glob("*.jpg"))
    annotation_paths = sorted(annotation_dir.glob("*.png"))
    if not image_paths or len(image_paths) != len(annotation_paths):
        raise RuntimeError(f"Invalid DAVIS sequence: {sequence}")
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
        annotation_paths = annotation_paths[:max_frames]

    first_mask = load_mask(annotation_paths[0])
    object_ids = [int(value) for value in np.unique(first_mask) if value != 0]
    if not object_ids:
        raise RuntimeError(f"No foreground objects in {sequence}")

    t0 = time.perf_counter()
    state = predictor.init_state(
        str(image_dir), offload_video_to_cpu=not keep_video_on_gpu
    )
    for object_id in object_ids:
        prompt = prompt_for_object(first_mask == object_id, prompt_policy)
        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=object_id,
            **prompt,
        )

    per_frame_j = []
    per_frame_f = []
    frames_seen = 0
    for frame_idx, returned_ids, mask_logits in predictor.propagate_in_video(state):
        if frame_idx >= len(annotation_paths):
            break
        ground_truth = load_mask(annotation_paths[frame_idx])
        predicted = mask_logits[:, 0].detach().float().cpu().numpy() > 0
        if prediction_root is not None:
            output_dir = prediction_root / sequence
            output_dir.mkdir(parents=True, exist_ok=True)
            combined = np.zeros_like(ground_truth, dtype=np.uint8)
            for position, object_id in enumerate(returned_ids):
                combined[predicted[position]] = int(object_id)
            Image.fromarray(combined).save(output_dir / f"{frame_idx:05d}.png")
        id_to_position = {int(object_id): position for position, object_id in enumerate(returned_ids)}
        object_j = []
        object_f = []
        for object_id in object_ids:
            gt_object = ground_truth == object_id
            position = id_to_position.get(object_id)
            pred_object = predicted[position] if position is not None else np.zeros_like(gt_object)
            intersection = np.logical_and(pred_object, gt_object).sum()
            union = np.logical_or(pred_object, gt_object).sum()
            object_j.append(float(intersection / union) if union else 1.0)
            object_f.append(boundary_f_score(pred_object, gt_object))
        per_frame_j.append(float(np.mean(object_j)))
        per_frame_f.append(float(np.mean(object_f)))
        frames_seen += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000
    torch.cuda.synchronize()
    return {
        "sequence": sequence,
        "objects": len(object_ids),
        "frames": frames_seen,
        "J": float(np.mean(per_frame_j)),
        "F": float(np.mean(per_frame_f)),
        "J&F": float((np.mean(per_frame_j) + np.mean(per_frame_f)) / 2),
        "latency_ms": elapsed_ms,
        "fps": frames_seen / (elapsed_ms / 1000),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis-root", type=Path, default=Path("data/davis/DAVIS"))
    parser.add_argument("--sam2-root", type=Path, default=Path("sam2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sequences", nargs="+", default=["bike-packing"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--keep-video-on-gpu", action="store_true")
    parser.add_argument(
        "--prompt-policy",
        choices=[
            "centroid", "box", "box_expand_05", "box_expand_10",
            "two_positive", "three_positive", "box_centroid",
            "box_positive_negative", "box_expand_10_negative", "box_two_positive",
        ],
        default="centroid",
    )
    parser.add_argument("--sequence-file", type=Path)
    parser.add_argument("--prediction-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/davis_sam2_baseline.csv"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")
    sequences = args.sequences
    if args.sequence_file:
        sequences = [line.strip() for line in args.sequence_file.read_text().splitlines() if line.strip()]
    rows = []
    for sequence in sequences:
        torch.cuda.reset_peak_memory_stats()
        row = evaluate_sequence(
            predictor,
            args.davis_root,
            sequence,
            args.max_frames,
            args.keep_video_on_gpu,
            args.prompt_policy,
            args.prediction_root,
        )
        rows.append(row)
        print(" | ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
