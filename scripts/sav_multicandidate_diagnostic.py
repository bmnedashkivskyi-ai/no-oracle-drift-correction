"""Diagnostic for candidate P1: is SAM2's own greedy per-frame IoU-argmax
candidate choice ever beaten by a temporal-continuity tie-break?

Context: sam2-drift-recovery-results.md P1 originally proposed "use SAM2's
multimask_output + own IoU score as a selector instead of self-mask
reinjection" -- but configs/sam2.1/sam2.1_hiera_t.yaml already has
`multimask_output_for_tracking: true`, so every frame in every prior
experiment (F1-F5, including the box_centroid baseline itself) already
generates 3 candidate masks and auto-picks the highest-self-assessed-IoU one
via torch.argmax(ious) in sam2_base.py's _forward_sam_heads. That mechanism
resolves *prompt* ambiguity (e.g. whole object vs sub-part), not "is my
reference itself wrong" -- so it does not by itself explain why F1-F5 all
plateaued around J&F~25.

The only genuinely untested lever is: capture the 3 raw candidates before the
model's internal argmax collapses them, and ask whether an EXTERNAL selection
criterion -- position/area continuity with the previously accepted mask, which
IS independent of the model's own confidence -- would sometimes pick a
candidate closer to ground truth. This script measures that, cheaply, with a
forward hook on the mask decoder (no library modification), on the standard
box_centroid baseline (no correction applied) across the 30 drift-failure
objects, comparing against GT only at annotated frames.

If continuity-choice rarely disagrees with the model's choice, or is not
better when it does, building the actual override mechanism (P1) is not
worth it and should be deprioritized in favor of P2/P4/P5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from davis_sam2_eval import prompt_for_object  # noqa: E402
from sav_reprompt_eval import load_drift_failure_targets  # noqa: E402
from sav_sam2_eval import (  # noqa: E402
    annotated_frame_indices,
    collect_object_inputs,
    list_frame_names,
    load_mask,
)

from sam2.build_sam import build_sam2_video_predictor


class DecoderCapture:
    def __init__(self, predictor):
        self.last = None
        self.handle = predictor.sam_mask_decoder.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        masks, iou_pred, _sam_tokens, _obj_score = output
        if masks.shape[1] == 3:  # only capture true multimask (3-candidate) calls
            self.last = (masks.detach(), iou_pred.detach())

    def remove(self):
        self.handle.remove()


def candidates_to_video_res(predictor, state, low_res_multimasks: torch.Tensor) -> torch.Tensor:
    image_size = predictor.image_size
    high_res = F.interpolate(
        low_res_multimasks, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    _, video_res = predictor._get_orig_video_res_output(state, high_res)
    return video_res  # [1, 3, video_H, video_W] logits


def mask_centroid_area(mask: torch.Tensor):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(xs) == 0:
        return None, 0
    return (float(xs.float().mean()), float(ys.float().mean())), int(len(xs))


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--analysis-csv", type=Path, default=Path("results/sav_val_failure_mode_analysis.csv"))
    parser.add_argument("--prompt-policy", default="box_centroid")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    targets = load_drift_failure_targets(args.analysis_csv)
    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")
    capture = DecoderCapture(predictor)

    total_frames = 0
    total_annotated = 0
    disagreements_annotated = 0
    continuity_better = 0
    model_better = 0
    tie = 0

    for i, (video_name, object_ids) in enumerate(sorted(targets.items())):
        video_dir = args.sav_root / "JPEGImages_24fps" / video_name
        ann_dir = args.sav_root / "Annotations_6fps" / video_name
        frame_names = list_frame_names(video_dir)
        object_inputs = collect_object_inputs(ann_dir, frame_names)

        for object_id in sorted(object_ids):
            if object_id not in object_inputs:
                continue
            per_frame = object_inputs[object_id]
            sorted_frames = sorted(per_frame)
            first_frame_idx = sorted_frames[0]
            wanted_frames = annotated_frame_indices(ann_dir, object_id, frame_names)

            state = predictor.init_state(str(video_dir))
            prompt = prompt_for_object(per_frame[first_frame_idx], args.prompt_policy)
            _, _, mask_logits = predictor.add_new_points_or_box(
                state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
            )
            prev_centroid, _ = mask_centroid_area(mask_logits[0, 0] > 0)

            for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=first_frame_idx + 1
            ):
                total_frames += 1
                model_mask = mask_logits[0, 0] > 0

                if capture.last is not None:
                    cand_masks_logits, ious = capture.last
                    cand_video_res = candidates_to_video_res(predictor, state, cand_masks_logits)
                    cand_masks = [cand_video_res[0, k] > 0 for k in range(3)]
                    model_idx = int(torch.argmax(ious[0]).item())

                    # Continuity tie-break: candidate whose centroid is closest to the
                    # previously accepted mask's centroid (independent of model confidence).
                    if prev_centroid is not None:
                        dists = []
                        for k in range(3):
                            c, area = mask_centroid_area(cand_masks[k])
                            if c is None:
                                dists.append(float("inf"))
                            else:
                                d = ((c[0] - prev_centroid[0]) ** 2 + (c[1] - prev_centroid[1]) ** 2) ** 0.5
                                dists.append(d)
                        continuity_idx = int(np.argmin(dists))
                    else:
                        continuity_idx = model_idx

                    if frame_idx in wanted_frames:
                        total_annotated += 1
                        gt_path = ann_dir / f"{object_id:03d}" / f"{frame_names[frame_idx]}.png"
                        if gt_path.exists() and continuity_idx != model_idx:
                            disagreements_annotated += 1
                            gt = torch.from_numpy(load_mask(gt_path) > 0).to(model_mask.device)
                            iou_model = iou(cand_masks[model_idx], gt)
                            iou_continuity = iou(cand_masks[continuity_idx], gt)
                            if iou_continuity > iou_model + 1e-6:
                                continuity_better += 1
                            elif iou_model > iou_continuity + 1e-6:
                                model_better += 1
                            else:
                                tie += 1

                new_centroid, _ = mask_centroid_area(model_mask)
                if new_centroid is not None:
                    prev_centroid = new_centroid
                capture.last = None

            print(f"[{i + 1}/{len(targets)}] {video_name} obj={object_id:03d} done", flush=True)

    capture.remove()
    print(f"\ntotal_frames={total_frames} total_annotated={total_annotated}")
    print(f"disagreements_at_annotated_frames={disagreements_annotated}")
    print(f"  continuity_better={continuity_better}  model_better={model_better}  tie={tie}")


if __name__ == "__main__":
    main()
