"""Collect flagged-frame images for LLM-as-external-verifier correction (pilot).

Follow-up to the 5 converged no-oracle drift-reprompt attempts (all self-referential:
single-mask, consensus-K, area-ratio threshold sweep, object_score_logits), which
all landed in J&F [23.8, 27.4] vs the oracle ceiling 84.3
(sam2-sav-failure-analysis.md). The common failure was that the correction source
(the model's own recent mask) is not independent of the detection signal, so
errors reinforce.

This script tests whether an EXTERNALLY-sourced correction -- from a vision-capable
LLM looking at the raw frame, decoupled from SAM2's internal state entirely -- can
break that ceiling. It only collects images; it does not correct anything itself.

For each pilot object: propagate using the object_score_logits<0 trigger (our
cleanest no-oracle signal from iteration #34). At each flagged frame (capped at
--max-events per object), save:
  - a reference crop of the target object from the first prompt frame
  - the full flagged frame (resized to fit --max-dim)
and record frame_idx + scale factor in a manifest CSV. A human or LLM reviewer
then fills in a corrective box per manifest row, which scripts/sav_llm_verify_apply.py
consumes to re-run propagation with real external corrections instead of self-mask
reinjection.

To keep the run itself moving forward while collecting further flagged frames,
propagation after each collected event still self-corrects with the last stable
mask (same as candidate #1/#4) -- this collection pass is diagnostic only, its
own predictions are not scored.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from davis_sam2_eval import prompt_for_object  # noqa: E402
from sav_drift_reprompt_objscore_eval import read_object_score_logit  # noqa: E402
from sav_sam2_eval import (  # noqa: E402
    annotated_frame_indices,
    collect_object_inputs,
    list_frame_names,
)

from sam2.build_sam import build_sam2_video_predictor

PILOT = [
    ("sav_014969", 2),
    ("sav_022396", 0),
    ("sav_024172", 0),
    ("sav_035009", 1),
    ("sav_051039", 1),
    ("sav_010291", 0),
]


def save_reference_crop(video_dir: Path, frame_name: str, gt_mask: np.ndarray, out_path: Path, pad: float = 0.3):
    img = Image.open(video_dir / f"{frame_name}.jpg")
    ys, xs = np.nonzero(gt_mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - bw * pad))
    x1 = min(img.width, int(x1 + bw * pad))
    y0 = max(0, int(y0 - bh * pad))
    y1 = min(img.height, int(y1 + bh * pad))
    img.crop((x0, y0, x1, y1)).save(out_path, quality=90)


def save_flagged_frame(video_dir: Path, frame_name: str, out_path: Path, max_dim: int) -> float:
    img = Image.open(video_dir / f"{frame_name}.jpg")
    scale = 1.0
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    img.save(out_path, quality=90)
    return scale  # resized_px * (1/scale) = original_px


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for video_name, object_id in PILOT:
        video_dir = args.sav_root / "JPEGImages_24fps" / video_name
        ann_dir = args.sav_root / "Annotations_6fps" / video_name
        frame_names = list_frame_names(video_dir)
        object_inputs = collect_object_inputs(ann_dir, frame_names)
        per_frame = object_inputs[object_id]
        sorted_frames = sorted(per_frame)
        first_frame_idx = sorted_frames[0]

        obj_dir = args.out_dir / video_name / f"{object_id:03d}"
        obj_dir.mkdir(parents=True, exist_ok=True)
        save_reference_crop(video_dir, frame_names[first_frame_idx], per_frame[first_frame_idx], obj_dir / "reference.jpg")

        state = predictor.init_state(str(video_dir))
        prompt = prompt_for_object(per_frame[first_frame_idx], args.prompt_policy)
        _, _, mask_logits = predictor.add_new_points_or_box(
            state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
        )
        obj_idx = predictor._obj_id_to_idx(state, object_id)
        last_stable_mask = mask_logits[0, 0] > 0

        current_start = first_frame_idx + 1
        n_events = 0
        while current_start < len(frame_names) and n_events < args.max_events:
            drift_frame_idx = None
            for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=current_start
            ):
                mask = mask_logits[0, 0] > 0
                score = read_object_score_logit(state, obj_idx, frame_idx)
                if score < 0.0:
                    drift_frame_idx = frame_idx
                    break
                last_stable_mask = mask

            if drift_frame_idx is None:
                break

            n_events += 1
            flagged_path = obj_dir / f"event{n_events}_frame{drift_frame_idx}.jpg"
            scale = save_flagged_frame(video_dir, frame_names[drift_frame_idx], flagged_path, args.max_dim)
            manifest_rows.append({
                "video": video_name, "object": f"{object_id:03d}", "frame_idx": drift_frame_idx,
                "event": n_events, "image_path": str(flagged_path), "scale": scale,
                "box_x0": "", "box_y0": "", "box_x1": "", "box_y1": "",
            })
            print(f"{video_name} obj{object_id:03d} event{n_events} frame_idx={drift_frame_idx} -> {flagged_path}")

            # Self-correct with last stable mask just to keep the collection run moving.
            _, _, corrected_logits = predictor.add_new_mask(
                state, frame_idx=drift_frame_idx, obj_id=object_id, mask=last_stable_mask
            )
            last_stable_mask = corrected_logits[0, 0] > 0
            current_start = drift_frame_idx + 1

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video", "object", "frame_idx", "event", "image_path", "scale",
            "box_x0", "box_y0", "box_x1", "box_y1",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nWrote {len(manifest_rows)} flagged events to {manifest_path}")


if __name__ == "__main__":
    main()
