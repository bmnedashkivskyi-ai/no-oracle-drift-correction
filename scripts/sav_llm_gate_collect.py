"""Collect flagged-frame images (with current mask overlay) for P4: LLM as a
veto/gate on self-mask drift correction, not a box-drawer.

P4 from sam2-drift-recovery-results.md SS10: instead of replacing the self-mask
correction source (F5/sav_llm_verify_*, disproven -- external LLM-drawn boxes
underperformed self-mask reinjection, J&F 18.5->10.2 on this same 6-object
pilot), use the LLM only to CONFIRM that a flagged frame is genuine drift
before applying the self-mask correction, attacking the correction-cascade
mechanism (F1-F4) by reducing false-positive corrections rather than by
replacing the correction source.

Same object_score_logits<0 trigger and same 6-object PILOT as
sav_llm_verify_collect.py, so the set of flagged frames is identical and this
isolates the one new variable: gating vs. always-correcting. Unlike the box-
drawing pilot, judging "is this really drift" requires seeing where SAM2's
*current* (pre-correction) mask actually is, not just the raw frame -- so each
flagged frame is saved with the current mask overlaid as a translucent red
region, alongside the same reference crop used before for identity grounding.

Self-corrects with the last stable mask after every flagged event to keep the
collection run moving (diagnostic only, like sav_llm_verify_collect.py) --
this pass's own predictions are not scored.
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
from sav_llm_verify_collect import PILOT, save_reference_crop  # noqa: E402
from sav_reprompt_eval import load_drift_failure_targets  # noqa: E402
from sav_sam2_eval import (  # noqa: E402
    annotated_frame_indices,
    collect_object_inputs,
    list_frame_names,
)

from sam2.build_sam import build_sam2_video_predictor


def save_flagged_frame_with_overlay(
    video_dir: Path, frame_name: str, mask: np.ndarray, out_path: Path, max_dim: int
) -> float:
    img = Image.open(video_dir / f"{frame_name}.jpg").convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    red = np.zeros((*mask.shape, 4), dtype=np.uint8)
    red[mask] = (255, 0, 0, 110)
    overlay = Image.alpha_composite(overlay, Image.fromarray(red, mode="RGBA"))
    composited = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    scale = 1.0
    if max(composited.size) > max_dim:
        scale = max_dim / max(composited.size)
        composited = composited.resize((int(composited.width * scale), int(composited.height * scale)))
    composited.save(out_path, quality=90)
    return scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--all-drift-failures", action="store_true",
        help="Use the full n=30 drift-failure subset (sav_reprompt_eval.load_drift_failure_targets) "
        "instead of the 6-object PILOT.",
    )
    parser.add_argument("--analysis-csv", type=Path, default=Path("results/sav_val_failure_mode_analysis.csv"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    if args.all_drift_failures:
        targets = load_drift_failure_targets(args.analysis_csv)
        pilot = sorted((v, o) for v, objs in targets.items() for o in objs)
    else:
        pilot = PILOT

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for video_name, object_id in pilot:
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
            drift_mask = None
            for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=current_start
            ):
                mask = mask_logits[0, 0] > 0
                score = read_object_score_logit(state, obj_idx, frame_idx)
                if score < 0.0:
                    drift_frame_idx = frame_idx
                    drift_mask = mask
                    break
                last_stable_mask = mask

            if drift_frame_idx is None:
                break

            n_events += 1
            flagged_path = obj_dir / f"event{n_events}_frame{drift_frame_idx}.jpg"
            mask_np = drift_mask.detach().cpu().numpy()
            scale = save_flagged_frame_with_overlay(
                video_dir, frame_names[drift_frame_idx], mask_np, flagged_path, args.max_dim
            )
            manifest_rows.append({
                "video": video_name, "object": f"{object_id:03d}", "frame_idx": drift_frame_idx,
                "event": n_events, "image_path": str(flagged_path), "scale": scale,
                "mask_area_px": int(mask_np.sum()), "verdict": "",
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
            "mask_area_px", "verdict",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nWrote {len(manifest_rows)} flagged events to {manifest_path}")


if __name__ == "__main__":
    main()
