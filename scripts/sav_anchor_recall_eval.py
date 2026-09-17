"""Phase B, object-pointer anchor recall -- n=30 pilot
(sam2-memory-gating-phase-b-plan.md Task 2). Applies install_anchor_recall
to PLAIN box_centroid propagation -- no Phase A memory-gating and no P5
reprompt-correction at all -- on the same n=30 known-drift subset used by
every prior pilot in this project (sav_reprompt_eval.py's
load_drift_failure_targets), to isolate anchor-recall's own effect before
any full-val commitment or combination with Phase A (see
sam2-memory-gating-phase-b-design.md SS4 for why the two mechanisms are
tested in isolation first).
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
from sav_sam2_eval import annotated_frame_indices, collect_object_inputs, list_frame_names  # noqa: E402
from sam2_multimask_capture import install  # noqa: E402
from sam2_anchor_recall import install_anchor_recall  # noqa: E402

from sam2.build_sam import build_sam2_video_predictor


def evaluate_object_with_anchor_recall(
    predictor, state, anchor_state, object_id: int, per_frame, frame_names,
    ann_dir: Path, prompt_policy: str, obj_out_dir: Path,
) -> tuple[int, int, int, int]:
    sorted_frames = sorted(per_frame)
    first_frame_idx = sorted_frames[0]
    predictor.reset_state(state)
    anchor_state.reset()
    obj_idx = predictor._obj_id_to_idx(state, object_id)
    assert obj_idx == 0, f"expected single-object propagation, got obj_idx == {obj_idx}"

    prompt = prompt_for_object(per_frame[first_frame_idx], prompt_policy)
    _, _, mask_logits = predictor.add_new_points_or_box(
        state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
    )
    last_mask = mask_logits[0, 0] > 0

    wanted_frames = annotated_frame_indices(ann_dir, object_id, frame_names)
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    if first_frame_idx in wanted_frames:
        arr = last_mask.detach().cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(arr).save(obj_out_dir / f"{frame_names[first_frame_idx]}.png")
        frames_written += 1

    n_frames = 0
    for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
        state, start_frame_idx=first_frame_idx + 1
    ):
        n_frames += 1
        if frame_idx in wanted_frames:
            arr = (mask_logits[0, 0] > 0).detach().cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(arr).save(obj_out_dir / f"{frame_names[frame_idx]}.png")
            frames_written += 1

    return frames_written, anchor_state.n_updates, anchor_state.n_injections, n_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--analysis-csv", type=Path, default=Path("results/sav_val_failure_mode_analysis.csv"))
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--inject-divergence-threshold", type=float, default=0.2)
    parser.add_argument("--prediction-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    targets = load_drift_failure_targets(args.analysis_csv)
    total_objects = sum(len(v) for v in targets.values())
    print(f"Loaded {total_objects} drift-failure objects across {len(targets)} videos")

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")
    capture = install(predictor)
    anchor_state = install_anchor_recall(
        predictor, capture,
        inject_divergence_threshold=args.inject_divergence_threshold,
    )

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
            frames_written, n_updates, n_injections, n_frames = evaluate_object_with_anchor_recall(
                predictor, state, anchor_state, object_id, object_inputs[object_id],
                frame_names, ann_dir, args.prompt_policy, obj_out_dir,
            )
            print(
                f"[{i + 1}/{len(targets)}] video={video_name} obj={object_id:03d} "
                f"frames_written={frames_written} n_updates={n_updates} "
                f"n_injections={n_injections} n_frames={n_frames}",
                flush=True,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  video total latency_ms={elapsed_ms:.1f}", flush=True)

    print(f"Done: wrote predictions to {args.prediction_root}")


if __name__ == "__main__":
    main()
