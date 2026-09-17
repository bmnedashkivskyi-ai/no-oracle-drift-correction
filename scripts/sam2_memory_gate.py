"""Dynamic memory-gating, Phase A (sam2-memory-gating-design.md SS4):
skip writing a frame's output into SAM2's non-conditioning memory when its
primary-vs-alternative candidate mask divergence (the same signal used by
the closed multi-mask-divergence pre-selection check, sam2-drift-recovery-
next-directions.md SS13) suggests a distractor is present. This isolates
the "forgetting" half of DAM4SAM's dual-memory mechanism (arxiv 2411.17576)
-- the "recall" half (a separate Distractor-Resolving Memory anchor bank)
is Phase B and out of scope here; see sam2-memory-gating-design.md SS4 for
why that half carries materially higher technical risk (an untrained
positional-encoding slot) and is not attempted in this module.

Mechanism: SAM2's memory-attention window
(sam2/sam2/modeling/sam2_base.py:497-650,
_prepare_memory_conditioned_features) is a fixed recency schedule that
already tolerates a missing slot silently ("if prev is None: continue",
line 571-572 -- this happens natively near the start of every video). This
module exploits exactly that tolerance: it never edits the vendored sam2/
clone, and never changes what the frozen model computes for a frame's own
mask (evaluation output is unaffected) -- it only decides, after the fact,
whether that frame's output is allowed to become a future frame's memory
input, via a monkey-patched propagate_in_video that reimplements the
original's body verbatim (sam2_video_predictor.py's propagate_in_video)
with one inserted gate check.

Confirmed by reading sam2/sam2/sam2_video_predictor.py directly (this
checkout, propagate_in_video at line 546): the draft this module started
from was byte-faithful to the real method's control flow (same start/end
frame_idx computation, same cond/non-cond branch split, same keyword
arguments to `_run_single_frame_inference`, same
`obj_output_dict[storage_key][frame_idx] = current_out` write this module
gates). One real discrepancy was found and fixed: the actual
`propagate_in_video` carries a `@torch.inference_mode()` decorator (line
545), which is NOT cosmetic for a generator method -- torch's context-
manager decorators special-case generator functions
(torch.utils._contextlib._wrap_generator) so that inference_mode is
re-entered around every resumption of the generator (every `next()`/
`send()`), not just once at call time. `build_sam2_video_predictor` only
calls `model.eval()` (see sam2/sam2/build_sam.py), never
`requires_grad_(False)`, so model parameters keep `requires_grad=True`;
every other public entry point on the predictor (`init_state`,
`add_new_points_or_box`, `propagate_in_video_preflight`, etc.) relies on
its own `@torch.inference_mode()` decorator for this reason. A patched
`propagate_in_video` that reused the draft's plain (undecorated) generator
would run `_run_single_frame_inference` and `_get_orig_video_res_output`
(neither of which carries its own inference_mode decorator) outside
inference_mode entirely, building a full autograd graph across the whole
propagation and risking a "cannot be saved for backward" error when a
grad-tracked op touches an inference tensor produced by an earlier,
still-decorated call (e.g. `add_new_points_or_box`'s conditioning-frame
output feeding memory attention here). This module's
`patched_propagate_in_video` is therefore wrapped in
`torch.inference_mode()` before being bound, matching the original
exactly.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from sav_dryrun_multimask_risk_eval import bbox_union_area_ratio  # noqa: E402

from sam2_multimask_capture import MultimaskCapture  # noqa: E402


def per_frame_divergence(feats: dict, pred_masks: torch.Tensor) -> float | None:
    """Same divergence computation as evaluate_object_multimask_dry_run in
    sav_dryrun_multimask_risk_eval.py, extracted for reuse in a live
    per-frame gating decision instead of an aggregate dry-run feature --
    except that function interpolates to mask_logits' shape (full video
    resolution) while this one interpolates to pred_masks' shape (the
    low-res ~256x256 mask from _run_single_frame_inference, making this
    particular interpolate call a no-op here), which does not change the
    resulting divergence value since bbox_union_area_ratio computes a
    scale-invariant area ratio."""
    ious = feats["ious"]
    if len(ious) < 2:
        return None
    multimasks = feats["low_res_multimasks"]  # [M, H, W], batch dim already stripped
    sorted_idx = sorted(range(len(ious)), key=lambda i: ious[i], reverse=True)
    best_idx, second_idx = sorted_idx[0], sorted_idx[1]
    target_size = (pred_masks.shape[-2], pred_masks.shape[-1])
    primary_mask = torch.nn.functional.interpolate(
        multimasks[best_idx][None, None].float(), size=target_size,
        mode="bilinear", align_corners=False,
    )[0, 0] > 0
    alt_mask = torch.nn.functional.interpolate(
        multimasks[second_idx][None, None].float(), size=target_size,
        mode="bilinear", align_corners=False,
    )[0, 0] > 0
    ratio = bbox_union_area_ratio(primary_mask, alt_mask)
    if ratio is None:
        return None
    return 1.0 - ratio


@dataclass
class MemoryGate:
    threshold: float
    n_checked: int = 0
    n_skipped: int = 0

    def reset_counts(self) -> None:
        self.n_checked = 0
        self.n_skipped = 0


def install_memory_gate(
    predictor, capture: MultimaskCapture, divergence_threshold: float
) -> MemoryGate:
    """Monkey-patch predictor.propagate_in_video (instance-bound, not the
    class) to skip storing a non-conditioning frame's output in memory
    when per_frame_divergence exceeds divergence_threshold. Must be called
    AFTER sam2_multimask_capture.install(predictor) -- this function
    consumes the capture it produces, and never re-patches track_step or
    reset_state itself (see the module docstring; those are a separate,
    independent method from propagate_in_video, so the two patches
    compose cleanly)."""
    gate = MemoryGate(threshold=divergence_threshold)
    original_propagate = predictor.propagate_in_video.__func__

    def patched_propagate_in_video(
        self, inference_state, start_frame_idx=None,
        max_frame_num_to_track=None, reverse=False,
    ):
        """Propagate the input points across frames to track in the entire
        video. Reimplements sam2_video_predictor.py's propagate_in_video
        verbatim (see module docstring), with the frame's output write to
        memory gated by per_frame_divergence when a candidate-mask
        divergence signal for this frame is available."""
        self.propagate_in_video_preflight(inference_state)

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video (memory-gated)"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                assert obj_idx == 0, (
                    "install_memory_gate assumes single-object propagation "
                    f"(obj_idx == 0), got obj_idx == {obj_idx} -- capture.pop "
                    "keys would not line up (see sam2_multimask_capture.py's "
                    "module docstring)"
                )
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else:
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,  # run on the slice of a single object
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    # --- gate check (the only behavioral change from the
                    # original propagate_in_video): decide whether this
                    # frame's output is allowed to become a future frame's
                    # memory input. The frozen model's own computation of
                    # pred_masks above is untouched either way -- this only
                    # controls whether current_out gets written into
                    # obj_output_dict[storage_key]. capture.pop(...) reads
                    # the discarded multi-mask candidates that
                    # sam2_multimask_capture's patched track_step stashed
                    # as a side effect of the _run_single_frame_inference
                    # call above (see that module's docstring -- no
                    # ordering hazard, the data is already there).
                    #
                    # NOTE: skipping this write suppresses more than the
                    # maskmem-attention bank narrowly. Per
                    # sam2/sam2/modeling/sam2_base.py:611-620,
                    # _prepare_memory_conditioned_features also reads each
                    # recent frame's obj_ptr out of this same
                    # non_cond_frame_outputs dict (for up to
                    # max_obj_ptrs_in_encoder=16 recent frames), so gating
                    # the write here suppresses BOTH the maskmem-feature
                    # cross-attention channel AND the object-pointer channel
                    # for this frame -- not just "memory" in the narrower
                    # maskmem-bank sense. Relevant to anyone later designing
                    # a more selective gate (e.g. one that suppresses only
                    # one of the two channels).
                    skip_write = False
                    try:
                        feats = capture.pop(obj_idx, frame_idx)
                    except KeyError:
                        feats = None
                    if feats is not None:
                        gate.n_checked += 1
                        divergence = per_frame_divergence(feats, pred_masks)
                        if divergence is not None and divergence > gate.threshold:
                            skip_write = True
                            gate.n_skipped += 1
                    if not skip_write:
                        obj_output_dict[storage_key][frame_idx] = current_out
                    # --- end gate check

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                pred_masks_per_obj[obj_idx] = pred_masks

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks

    # The real propagate_in_video is decorated with @torch.inference_mode()
    # (sam2_video_predictor.py:545). For a generator function this is not
    # cosmetic -- torch's context-manager decorators special-case generator
    # functions (torch.utils._contextlib._wrap_generator) to re-enter
    # inference_mode around every resumption of the generator, not just
    # once at call time. Wrap here, before binding, to match: see the
    # module docstring's "Confirmed by reading" paragraph for why omitting
    # this would build an autograd graph across the whole propagation
    # (model params keep requires_grad=True; only model.eval() is called)
    # and risks an "inference tensor cannot be saved for backward" error
    # when this generator's un-gated calls touch tensors produced by an
    # earlier, still-inference_mode-decorated predictor call.
    patched_propagate_in_video = torch.inference_mode()(patched_propagate_in_video)

    predictor.propagate_in_video = types.MethodType(patched_propagate_in_video, predictor)
    predictor._original_propagate_in_video = original_propagate
    return gate
