"""Phase B, object-pointer anchor recall (sam2-memory-gating-phase-b-
design.md). Tested ALONE, with no Phase A memory-gating active -- see
that design note SS4 for why the two are isolated before any combination.

Mechanism: SAM2's object-pointer conditioning channel
(sam2/sam2/modeling/sam2_base.py, _prepare_memory_conditioned_features)
already includes a temporal positional encoding computed by a continuous
sine function (get_1d_sine_pe), unlike the maskmem channel's small fixed
lookup table -- and the model already evaluates this encoding at
out-of-[0,1]-normalized distances on every video longer than ~16 frames,
via the always-included conditioning-frame pointer (raw, UNCLAMPED
frame_idx - 0 distance, divided by the same small t_diff_max used for
"recent" pointers). This module substitutes a dynamically-tracked
"anchor" pointer -- the most recent confirmed-clean frame's object
pointer -- into the existing farthest recency slot (t_diff =
max_obj_ptrs_in_encoder - 1) when the PREVIOUS frame's divergence signal
suggested a distractor, using the same unclamped-distance sine-PE
mechanism the conditioning frame already exercises. This REPLACES a slot,
never adds one, so total object-pointer count never exceeds the trained
maximum.

Two coordinated monkey-patches on one predictor instance:
  1. propagate_in_video (generator, same @torch.inference_mode() care as
     sam2_memory_gate.py's patch) -- updates AnchorState after each
     frame's decode.
  2. _prepare_memory_conditioned_features (plain method) -- reads
     AnchorState and splices the anchor into the object-pointer list
     before it reaches memory_attention, when the previous frame's
     divergence warrants it.

Confirmed by reading sam2/sam2/modeling/sam2_base.py and
sam2/sam2/sam2_video_predictor.py directly (this checkout, 2026-09-16),
against the draft in task-1-brief.md:

- `_prepare_memory_conditioned_features` (sam2_base.py:497-517 header,
  object-pointer block at 586-650) has exactly the signature the draft
  assumed: `(self, frame_idx, is_init_cond_frame, current_vision_feats,
  current_vision_pos_embeds, feat_sizes, output_dict, num_frames,
  track_in_reverse=False)`. It carries no decorator of its own (plain
  method -- confirmed by grepping the whole file for
  `inference_mode`/`no_grad`: zero matches), so it inherits whatever
  inference-mode context its caller (`_track_step`, itself called from
  `track_step` -> `_run_single_frame_inference`, always reached from
  inside `propagate_in_video`'s or `add_new_points_or_box`'s own
  `@torch.inference_mode()`-decorated frame) is running under. It has no
  training-mode branch that changes this: the one `self.training` check
  in the object-pointer block (`only_obj_ptrs_in_the_past_for_eval`,
  sam2_base.py:591) only filters which conditioning-frame pointers are
  visible by time direction, not whether the method itself runs under
  grad tracking.
- `_track_step`'s call (sam2_base.py:761-770) passes every argument as a
  keyword (`frame_idx=`, `is_init_cond_frame=`, `current_vision_feats=`,
  `current_vision_pos_embeds=`, `feat_sizes=`, `output_dict=`,
  `num_frames=`, `track_in_reverse=`), matching the draft's assumed
  calling convention exactly -- no positional-vs-keyword mismatch risk
  here (unlike the `track_step`/`_track_step` split
  `sam2_multimask_capture.py` had to guard against with
  `inspect.signature(...).bind(...)`).
- The object-pointer construction block's variable names and control
  flow (the `for t_diff in range(1, max_obj_ptrs_in_encoder): ...
  output_dict["non_cond_frame_outputs"].get(t, ...)` loop, `t_diff_max =
  max_obj_ptrs_in_encoder - 1`, `get_1d_sine_pe(obj_pos / t_diff_max, ...)`
  at sam2_base.py:612-634) match the draft's assumptions exactly, so the
  "temporarily substitute non_cond_frame_outputs[farthest_t], call the
  original, restore in finally" splice technique is sound: that loop's
  only interaction with `non_cond_frame_outputs` is the single
  `.get(t, ...)` read this patch redirects.
- Checked for a collision with the OTHER loop that also reads
  `output_dict["non_cond_frame_outputs"]` in this same function -- the
  maskmem spatial-feature loop just above (sam2_base.py:539-568,
  `prev["maskmem_features"]`). That loop only looks back
  `self.num_maskmem - 1` frames (config `num_maskmem: 7`, so at most 6
  frames back, stride 1 -- `memory_temporal_stride_for_eval` is not
  overridden in sam2.1_hiera_t.yaml, so it keeps the class default of 1).
  The farthest object-pointer slot this module targets is `frame_idx -
  (max_obj_ptrs_in_encoder - 1)` = `frame_idx - 15` (`max_obj_ptrs_in_encoder`
  also not overridden in the yaml, class default 16). 15 != any value in
  {1..6}, so the maskmem loop never reads the entry this module
  temporarily replaces with a bare `{"obj_ptr": ...}` dict (which lacks
  the "maskmem_features"/"maskmem_pos_enc" keys that loop would need) --
  no risk of a KeyError from that other loop during the injection window.
- `propagate_in_video` (sam2_video_predictor.py:545-630) matches the
  draft's reimplementation line for line (same preflight call, same
  start/end frame_idx computation, same cond/non-cond branch split, same
  keyword arguments to `_run_single_frame_inference`, same
  `obj_output_dict[storage_key][frame_idx] = current_out` write, same
  final resize/yield), and carries the same `@torch.inference_mode()`
  decorator at line 545 that `sam2_memory_gate.py`'s Task 1 already found
  is load-bearing for a generator method (torch's context-manager
  decorators re-enter around every `next()`/`send()` resumption, not just
  once at call time -- see that module's docstring). This module's
  `patched_propagate_in_video` is wrapped in `torch.inference_mode()`
  before binding, matching the original.

No discrepancy from the draft was found in this pass -- the draft's
assumptions all held against the actual installed files, so the code
below is unchanged from task-1-brief.md's Step 2 except for this
docstring's verification notes.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from sam2_memory_gate import per_frame_divergence  # noqa: E402
from sam2_multimask_capture import MultimaskCapture  # noqa: E402


@dataclass
class AnchorState:
    anchor: tuple[int, torch.Tensor] | None = None
    last_divergence: float | None = None
    n_updates: int = 0
    n_injections: int = 0

    def reset(self) -> None:
        self.anchor = None
        self.last_divergence = None
        self.n_updates = 0
        self.n_injections = 0


def install_anchor_recall(
    predictor,
    capture: MultimaskCapture,
    clean_divergence_threshold: float = 0.1,
    clean_iou_threshold: float = 0.8,
    inject_divergence_threshold: float = 0.2,
) -> AnchorState:
    """Must be called AFTER sam2_multimask_capture.install(predictor).
    Do NOT also call sam2_memory_gate.install_memory_gate on the same
    predictor for this experiment -- Phase B is tested in isolation from
    Phase A (sam2-memory-gating-phase-b-design.md SS4)."""
    state = AnchorState()

    # --- Patch 1: _prepare_memory_conditioned_features (read side) ---
    original_prepare = predictor._prepare_memory_conditioned_features.__func__

    def patched_prepare_memory_conditioned_features(
        self, frame_idx, is_init_cond_frame, current_vision_feats,
        current_vision_pos_embeds, feat_sizes, output_dict, num_frames,
        track_in_reverse=False,
    ):
        should_inject = (
            not is_init_cond_frame
            and state.anchor is not None
            and state.last_divergence is not None
            and state.last_divergence > inject_divergence_threshold
        )
        if not should_inject:
            return original_prepare(
                self, frame_idx, is_init_cond_frame, current_vision_feats,
                current_vision_pos_embeds, feat_sizes, output_dict,
                num_frames, track_in_reverse,
            )

        # Splice the anchor into the farthest recency slot by temporarily
        # replacing that slot's source in output_dict["non_cond_frame_outputs"]
        # for the duration of this call, then restoring it -- this reuses
        # the original function's own lookup logic exactly rather than
        # re-deriving the object-pointer construction block, so any future
        # SAM2 change to that block is inherited automatically.
        #
        # Only ever REPLACE a slot the original code would genuinely have
        # populated (had_farthest is True) -- never fabricate a new one.
        # `farthest_t` lands outside non_cond_frame_outputs whenever it
        # coincides with a conditioning/prompt frame (those live in
        # cond_frame_outputs, not non_cond_frame_outputs) or on short
        # objects whose frame range doesn't reach back that far; in both
        # cases the unmodified code would find no token there at all
        # (`out is None` -> no pointer appended), so injecting would ADD a
        # token instead of REPLACING one, violating the "replace, never
        # add" design choice (sam2-memory-gating-phase-b-design.md SS2)
        # that keeps total object-pointer count at the trained maximum.
        anchor_frame_idx, anchor_obj_ptr = state.anchor
        max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
        t_diff_farthest = max_obj_ptrs_in_encoder - 1
        farthest_t = (
            frame_idx + t_diff_farthest if track_in_reverse
            else frame_idx - t_diff_farthest
        )
        non_cond = output_dict["non_cond_frame_outputs"]
        had_farthest = farthest_t in non_cond
        if not had_farthest:
            return original_prepare(
                self, frame_idx, is_init_cond_frame, current_vision_feats,
                current_vision_pos_embeds, feat_sizes, output_dict,
                num_frames, track_in_reverse,
            )

        original_farthest_entry = non_cond[farthest_t]
        non_cond[farthest_t] = {"obj_ptr": anchor_obj_ptr}
        try:
            result = original_prepare(
                self, frame_idx, is_init_cond_frame, current_vision_feats,
                current_vision_pos_embeds, feat_sizes, output_dict,
                num_frames, track_in_reverse,
            )
        finally:
            non_cond[farthest_t] = original_farthest_entry
        state.n_injections += 1
        return result

    predictor._prepare_memory_conditioned_features = types.MethodType(
        patched_prepare_memory_conditioned_features, predictor
    )
    predictor._original_prepare_memory_conditioned_features = original_prepare

    # --- Patch 2: propagate_in_video (write side, generator) ---
    original_propagate = predictor.propagate_in_video.__func__

    def patched_propagate_in_video(
        self, inference_state, start_frame_idx=None,
        max_frame_num_to_track=None, reverse=False,
    ):
        self.propagate_in_video_preflight(inference_state)
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        if start_frame_idx is None:
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video (anchor-recall)"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                assert obj_idx == 0, (
                    "install_anchor_recall assumes single-object propagation "
                    f"(obj_idx == 0), got obj_idx == {obj_idx}"
                )
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    if self.clear_non_cond_mem_around_input:
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else:
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    obj_output_dict[storage_key][frame_idx] = current_out

                    # Update anchor state from THIS frame's outcome, for use
                    # when preparing the NEXT frame's memory read (causal:
                    # we cannot know this frame's own divergence until after
                    # it's decoded, so injection always lags detection by
                    # one frame -- see design note SS2).
                    try:
                        feats = capture.pop(obj_idx, frame_idx)
                    except KeyError:
                        feats = None
                    divergence = per_frame_divergence(feats, pred_masks) if feats else None
                    primary_iou = max(feats["ious"]) if feats else None
                    if divergence is not None:
                        state.last_divergence = divergence
                        if (
                            divergence <= clean_divergence_threshold
                            and primary_iou is not None
                            and primary_iou >= clean_iou_threshold
                        ):
                            state.anchor = (frame_idx, current_out["obj_ptr"])
                            state.n_updates += 1

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                pred_masks_per_obj[obj_idx] = pred_masks

            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks

    patched_propagate_in_video = torch.inference_mode()(patched_propagate_in_video)
    predictor.propagate_in_video = types.MethodType(patched_propagate_in_video, predictor)
    predictor._original_propagate_in_video = original_propagate

    # --- Patch 3: reset_state (auto-clear AnchorState on reset) ---
    # Mirrors sam2_multimask_capture.py's own fix-round-1 "auto-clear on
    # reset_state()" pattern for MultimaskCapture: this project's universal
    # per-object eval loop (every sav_*_eval.py script, e.g.
    # sav_sam2_eval.py:83) calls predictor.reset_state(state) once per
    # object within a single shared predictor instance. Since
    # install_anchor_recall is called ONCE per predictor (not once per
    # object), state.anchor/state.last_divergence would otherwise persist
    # across objects with nothing to clear them, silently leaking the
    # previous object's anchor object-pointer into the next object's
    # propagation.
    original_reset_state = predictor.reset_state.__func__

    def patched_reset_state(self, inference_state):
        result = original_reset_state(self, inference_state)
        state.reset()
        return result

    predictor.reset_state = types.MethodType(patched_reset_state, predictor)
    predictor._original_reset_state = original_reset_state

    return state
