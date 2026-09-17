"""E1-followup: multi-mask divergence features from an UNCORRECTED pass.

E1 (sam2-drift-recovery-next-directions.md SS8-9, 11) tested six
trajectory-derived features (area deviation, position jump, trigger
timing) against the capstone-vs-P2-only per-object delta and found no
signal (combined R^2=0.026). All six were properties of the SELECTED
mask's history across frames. This script instead captures a
single-frame introspective signal SAM2 already computes internally on
every step -- the divergence between its primary predicted mask and its
own best alternative candidate (DAM4SAM, arxiv 2411.17576) -- which is
never surfaced by track_step (see scripts/sam2_multimask_capture.py) and
was never part of E1's feature set.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from davis_sam2_eval import prompt_for_object  # noqa: E402
from sav_sam2_eval import annotated_frame_indices, collect_object_inputs, list_frame_names  # noqa: E402
from sam2_multimask_capture import install  # noqa: E402

from sam2.build_sam import build_sam2_video_predictor


def mask_bbox(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_union_area_ratio(primary: torch.Tensor, alternative: torch.Tensor) -> float | None:
    primary_bbox = mask_bbox(primary)
    if primary_bbox is None:
        return None
    px0, py0, px1, py1 = primary_bbox
    primary_area = max(px1 - px0, 1) * max(py1 - py0, 1)
    alt_bbox = mask_bbox(alternative)
    if alt_bbox is None:
        return 1.0
    ax0, ay0, ax1, ay1 = alt_bbox
    ux0, uy0 = min(px0, ax0), min(py0, ay0)
    ux1, uy1 = max(px1, ax1), max(py1, ay1)
    union_area = max(ux1 - ux0, 1) * max(uy1 - uy0, 1)
    return primary_area / union_area


def evaluate_object_multimask_dry_run(
    predictor, state, capture, object_id: int, per_frame,
    prompt_policy: str,
) -> dict:
    sorted_frames = sorted(per_frame)
    first_frame_idx = sorted_frames[0]
    predictor.reset_state(state)
    obj_idx = predictor._obj_id_to_idx(state, object_id)

    # This script (like every other sav_*_eval.py script) calls
    # predictor.reset_state(state) immediately before prompting each
    # object, so exactly one object is ever tracked per
    # propagate_in_video call -- the real SAM2 obj_idx for that object
    # must therefore always be 0. Fail loudly if this script is ever
    # reused for multi-object batches, since capture.pop(0, frame_idx)
    # below relies on this invariant (see sam2_multimask_capture.py's
    # module docstring on why key_n, not the real obj_idx, is used).
    assert obj_idx == 0, (
        f"expected single-object propagation (obj_idx == 0), got obj_idx == {obj_idx}"
    )

    prompt = prompt_for_object(per_frame[first_frame_idx], prompt_policy)
    predictor.add_new_points_or_box(
        state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
    )

    n_frames = 0
    divergences: list[float] = []
    iou_gaps: list[float] = []
    first_divergence_rel_frame: int | None = None

    for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
        state, start_frame_idx=first_frame_idx + 1
    ):
        n_frames += 1
        rel_frame = frame_idx - first_frame_idx
        # Pop by capture's call-order key (always 0 for single-object
        # propagation), NOT by the real SAM2 obj_idx -- see the
        # assertion above and sam2_multimask_capture.py's module
        # docstring for why these two are not the same key space.
        try:
            feats = capture.pop(0, frame_idx)
        except KeyError:
            continue
        ious = feats["ious"]
        multimasks = feats["low_res_multimasks"]  # shape [M, H, W]; batch dim already stripped by capture
        sorted_idx = sorted(range(len(ious)), key=lambda i: ious[i], reverse=True)
        best_idx, second_idx = sorted_idx[0], sorted_idx[1]
        primary_mask = torch.nn.functional.interpolate(
            multimasks[best_idx][None, None].float(),
            size=(mask_logits.shape[-2], mask_logits.shape[-1]),
            mode="bilinear", align_corners=False,
        )[0, 0] > 0
        alt_mask = torch.nn.functional.interpolate(
            multimasks[second_idx][None, None].float(),
            size=(mask_logits.shape[-2], mask_logits.shape[-1]),
            mode="bilinear", align_corners=False,
        )[0, 0] > 0
        ratio = bbox_union_area_ratio(primary_mask, alt_mask)
        if ratio is not None:
            divergence = 1.0 - ratio
            divergences.append(divergence)
            iou_gaps.append(ious[best_idx] - ious[second_idx])
            if divergence > 0.3 and first_divergence_rel_frame is None:
                first_divergence_rel_frame = rel_frame

    return {
        "n_frames": n_frames,
        "divergence_mean": statistics.fmean(divergences) if divergences else 0.0,
        "divergence_max": max(divergences) if divergences else 0.0,
        "frac_frames_divergent": (
            sum(1 for d in divergences if d > 0.3) / len(divergences)
        ) if divergences else 0.0,
        "first_divergence_frac": (
            first_divergence_rel_frame / n_frames
        ) if first_divergence_rel_frame is not None and n_frames > 0 else -1.0,
        "iou_gap_mean": statistics.fmean(iou_gaps) if iou_gaps else 0.0,
        "iou_gap_min": min(iou_gaps) if iou_gaps else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--video-list", type=Path, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--prompt-policy", default="largest_component_box_centroid")
    parser.add_argument("--output-csv", type=Path, required=True)
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

    fieldnames = [
        "video", "object_id", "n_frames", "divergence_mean", "divergence_max",
        "frac_frames_divergent", "first_divergence_frac", "iou_gap_mean", "iou_gap_min",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
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
            feats = evaluate_object_multimask_dry_run(
                predictor, state, capture, object_id, per_frame, args.prompt_policy,
            )
            rows.append({"video": video_name, "object_id": f"{object_id:03d}", **feats})
            line = (
                f"[{i + 1}/{len(video_names)}] video={video_name} obj={object_id:03d} "
                f"n_frames={feats['n_frames']} divergence_mean={feats['divergence_mean']:.4f} "
                f"frac_frames_divergent={feats['frac_frames_divergent']:.4f}"
            )
            print(line, flush=True)
            log_lines.append(line)
            if args.progress_log:
                args.progress_log.write_text("\n".join(log_lines) + "\n")
            with args.output_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  video total latency_ms={elapsed_ms:.1f}", flush=True)

    print(f"Done: wrote {len(rows)} object divergence-feature rows to {args.output_csv}")


if __name__ == "__main__":
    main()
