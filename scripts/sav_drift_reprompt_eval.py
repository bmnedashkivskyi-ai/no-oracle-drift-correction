"""No-oracle drift-detection + self-reprompt policy for SAM2 video propagation.

Follow-up to scripts/sav_reprompt_eval.py, which showed that periodic GT-based
re-prompting recovers SA-V propagation-drift failures (J&F 26.1 -> 84.3 on the
30 drift-failure objects from sam2-sav-failure-analysis.md). That was an ORACLE
ceiling estimate. This script asks whether a DEPLOYABLE policy -- one with no
access to ground truth after the first frame -- can capture any of that gain.

Drift signal (no GT): predicted mask area at each raw frame, compared against an
EMA baseline built from accepted (non-drifted) frames. A sudden collapse or
ballooning of area relative to that baseline is treated as a proxy for "the
tracker likely lost or mis-attached the object" -- a self-consistency signal,
not a ground-truth comparison.

Correction (no GT): when drift is flagged at frame t, the model's OWN last
trusted mask (from the last accepted frame) is re-injected at frame t via
add_new_mask -- not the ground-truth mask, not derived from it. This re-anchors
the memory bank on a mask the model already committed to, rather than letting
the drifted prediction poison future memory. Propagation then resumes from t+1.
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


def evaluate_object_with_drift_reprompt(
    predictor,
    state,
    object_id: int,
    per_frame: dict[int, np.ndarray],
    frame_names: list[str],
    ann_dir: Path,
    prompt_policy: str,
    obj_out_dir: Path,
    ema_alpha: float,
    low_ratio: float,
    high_ratio: float,
    min_area_eps: float,
    max_corrections: int,
):
    sorted_frames = sorted(per_frame)
    first_frame_idx = sorted_frames[0]
    predictor.reset_state(state)

    prompt = prompt_for_object(per_frame[first_frame_idx], prompt_policy)
    _, _, mask_logits = predictor.add_new_points_or_box(
        state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
    )
    last_stable_mask = (mask_logits[0, 0] > 0)
    ema_area = float(last_stable_mask.sum().item())

    wanted_frames = annotated_frame_indices(ann_dir, object_id, frame_names)
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    if first_frame_idx in wanted_frames:
        arr = last_stable_mask.detach().cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(arr).save(obj_out_dir / f"{frame_names[first_frame_idx]}.png")

    frames_written = 1 if first_frame_idx in wanted_frames else 0
    n_corrections = 0
    current_start = first_frame_idx + 1

    while current_start < len(frame_names) and n_corrections < max_corrections:
        drift_frame_idx = None
        for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=current_start
        ):
            mask = mask_logits[0, 0] > 0
            area = float(mask.sum().item())

            if frame_idx in wanted_frames:
                arr = mask.detach().cpu().numpy().astype(np.uint8) * 255
                Image.fromarray(arr).save(obj_out_dir / f"{frame_names[frame_idx]}.png")
                frames_written += 1

            is_drift = ema_area >= min_area_eps and (
                area == 0 or area / ema_area < low_ratio or area / ema_area > high_ratio
            )
            if is_drift:
                drift_frame_idx = frame_idx
                break
            last_stable_mask = mask
            ema_area = ema_alpha * area + (1 - ema_alpha) * ema_area

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
        ema_area = float(corrected_mask.sum().item())
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
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--low-ratio", type=float, default=0.3)
    parser.add_argument("--high-ratio", type=float, default=3.0)
    parser.add_argument("--min-area-eps", type=float, default=50.0)
    parser.add_argument("--max-corrections", type=int, default=60)
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
            frames_written, n_corrections = evaluate_object_with_drift_reprompt(
                predictor, state, object_id, object_inputs[object_id],
                frame_names, ann_dir, args.prompt_policy, obj_out_dir,
                args.ema_alpha, args.low_ratio, args.high_ratio,
                args.min_area_eps, args.max_corrections,
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
