"""Diagnostic for P3: does an LLM visual selector beat SAM2's own greedy
IoU-argmax candidate choice, or the position-continuity tie-break, when they
disagree?

Direct follow-up to scripts/sav_multicandidate_diagnostic.py (P1), which found
that at the 1637 GT-annotated frames where continuity-based tie-break disagrees
with the model's own choice, continuity is right about as often as the model
(360 vs 347 out of 707 non-tie cases) -- statistically a coin flip. P1's own
conclusion: if a cheap alternative selector doesn't clearly beat this, the
override mechanism isn't worth building. P3 (sam2-drift-recovery-results.md
SS10) proposes swapping the geometric continuity heuristic for a genuinely
different one -- semantic/visual judgment from an LLM -- reusing the exact
same 3-candidate capture (DecoderCapture, unchanged from P1).

This is a MEASUREMENT pass, not a corrected end-to-end pipeline: it samples a
tractable number of disagreement events (capped per object so the manual
review stays bounded), renders all 3 raw candidates as a single color-coded
overlay (red/green/blue) on the frame plus a reference-crop inset, and records
which one an external reviewer (Claude, this session) picks -- WITHOUT ever
showing the reviewer the ground truth. Accuracy against GT is computed
separately, after the manifest is filled in, exactly like the model/continuity
columns already computed by the P1 diagnostic. Only if LLM-choice clearly and
non-trivially beats both existing arms on this sample is building an actual
override mechanism (replaying propagation with LLM-selected candidates)
justified -- P4 showed at scale that a promising small-sample LLM effect can
evaporate or reverse, so this diagnostic step is deliberately cheap before any
such commitment.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from davis_sam2_eval import prompt_for_object  # noqa: E402
from sav_llm_verify_collect import save_reference_crop  # noqa: E402
from sav_reprompt_eval import load_drift_failure_targets  # noqa: E402
from sav_sam2_eval import (  # noqa: E402
    annotated_frame_indices,
    collect_object_inputs,
    list_frame_names,
)

from sam2.build_sam import build_sam2_video_predictor

CANDIDATE_COLORS = [(255, 0, 0, 110), (0, 200, 0, 110), (60, 120, 255, 110)]  # red, green, blue


class DecoderCapture:
    def __init__(self, predictor):
        self.last = None
        self.handle = predictor.sam_mask_decoder.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        masks, iou_pred, _sam_tokens, _obj_score = output
        if masks.shape[1] == 3:
            self.last = (masks.detach(), iou_pred.detach())

    def remove(self):
        self.handle.remove()


def candidates_to_video_res(predictor, state, low_res_multimasks: torch.Tensor) -> torch.Tensor:
    image_size = predictor.image_size
    high_res = F.interpolate(
        low_res_multimasks, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    _, video_res = predictor._get_orig_video_res_output(state, high_res)
    return video_res


def mask_centroid_area(mask: torch.Tensor):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(xs) == 0:
        return None, 0
    return (float(xs.float().mean()), float(ys.float().mean())), int(len(xs))


def save_composite(
    video_dir: Path, frame_name: str, cand_masks_np: list[np.ndarray],
    ref_crop_path: Path, out_path: Path, max_dim: int,
) -> None:
    img = Image.open(video_dir / f"{frame_name}.jpg").convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for mask, color in zip(cand_masks_np, CANDIDATE_COLORS):
        layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
        layer[mask] = color
        overlay = Image.alpha_composite(overlay, Image.fromarray(layer, mode="RGBA"))
    composited = Image.alpha_composite(img, overlay).convert("RGB")

    scale = 1.0
    if max(composited.size) > max_dim:
        scale = max_dim / max(composited.size)
        composited = composited.resize((int(composited.width * scale), int(composited.height * scale)))

    ref = Image.open(ref_crop_path).convert("RGB")
    ref_h = composited.height // 4
    ref_w = int(ref.width * ref_h / ref.height)
    ref = ref.resize((max(ref_w, 1), ref_h))
    canvas = Image.new("RGB", (composited.width + ref.width, max(composited.height, ref.height)), (20, 20, 20))
    canvas.paste(composited, (0, 0))
    canvas.paste(ref, (composited.width, 0))
    canvas.save(out_path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--analysis-csv", type=Path, default=Path("results/sav_val_failure_mode_analysis.csv"))
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--max-events-per-object", type=int, default=2)
    parser.add_argument("--max-dim", type=int, default=900)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    targets = load_drift_failure_targets(args.analysis_csv)
    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")
    capture = DecoderCapture(predictor)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

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

            obj_dir = args.out_dir / video_name / f"{object_id:03d}"
            ref_crop_path = obj_dir / "reference.jpg"
            n_events = 0

            state = predictor.init_state(str(video_dir))
            prompt = prompt_for_object(per_frame[first_frame_idx], args.prompt_policy)
            _, _, mask_logits = predictor.add_new_points_or_box(
                state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
            )
            prev_centroid, _ = mask_centroid_area(mask_logits[0, 0] > 0)

            for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=first_frame_idx + 1
            ):
                if n_events >= args.max_events_per_object:
                    capture.last = None
                    continue

                model_mask = mask_logits[0, 0] > 0

                if capture.last is not None and frame_idx in wanted_frames:
                    cand_masks_logits, ious = capture.last
                    cand_video_res = candidates_to_video_res(predictor, state, cand_masks_logits)
                    cand_masks = [cand_video_res[0, k] > 0 for k in range(3)]
                    model_idx = int(torch.argmax(ious[0]).item())

                    if prev_centroid is not None:
                        dists = []
                        for k in range(3):
                            c, _area = mask_centroid_area(cand_masks[k])
                            dists.append(float("inf") if c is None else
                                         ((c[0] - prev_centroid[0]) ** 2 + (c[1] - prev_centroid[1]) ** 2) ** 0.5)
                        continuity_idx = int(np.argmin(dists))
                    else:
                        continuity_idx = model_idx

                    if continuity_idx != model_idx:
                        if not ref_crop_path.exists():
                            obj_dir.mkdir(parents=True, exist_ok=True)
                            save_reference_crop(
                                video_dir, frame_names[first_frame_idx], per_frame[first_frame_idx], ref_crop_path
                            )
                        n_events += 1
                        cand_masks_np = [m.detach().cpu().numpy() for m in cand_masks]
                        out_path = obj_dir / f"event{n_events}_frame{frame_idx}.jpg"
                        save_composite(video_dir, frame_names[frame_idx], cand_masks_np, ref_crop_path, out_path, args.max_dim)
                        manifest_rows.append({
                            "video": video_name, "object": f"{object_id:03d}", "frame_idx": frame_idx,
                            "event": n_events, "image_path": str(out_path),
                            "model_idx": model_idx, "continuity_idx": continuity_idx,
                            "llm_idx": "",
                        })
                        print(f"{video_name} obj{object_id:03d} event{n_events} frame_idx={frame_idx} -> {out_path}", flush=True)

                new_centroid, _ = mask_centroid_area(model_mask)
                if new_centroid is not None:
                    prev_centroid = new_centroid
                capture.last = None

            print(f"[{i + 1}/{len(targets)}] {video_name} obj={object_id:03d} done ({n_events} events)", flush=True)

    capture.remove()

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video", "object", "frame_idx", "event", "image_path",
            "model_idx", "continuity_idx", "llm_idx",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nWrote {len(manifest_rows)} disagreement events to {manifest_path}")


if __name__ == "__main__":
    main()
