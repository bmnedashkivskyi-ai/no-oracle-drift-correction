"""Captures SAM2's discarded multi-mask candidates (low_res_multimasks,
ious) during propagate_in_video, without modifying the vendored sam2/
clone.

See track_step in sam2/sam2/modeling/sam2_base.py (track_step at line 814
in the checked-out revision; re-verify line numbers if a different SAM2
checkout is installed). track_step calls self._track_step(...) to get a
`sam_outputs` tuple that still carries all M candidate masks and their M
IoU estimates, then immediately reduces to the single best-IoU mask
before writing `current_out["pred_masks"]` / `current_out["pred_masks_high_res"]`
and returning -- the other M-1 candidates and their IoU estimates are
dropped and never appear in `current_out`.

This module re-derives the discarded candidates by monkey-patching the
*instance* method `predictor.track_step` (not the class -- so other
predictor instances in the same process are unaffected) with a function
that reimplements track_step's body: it calls the same `self._track_step`
the original does, then reproduces byte-for-byte the reduction logic that
track_step itself performs on `sam_outputs` to build `current_out`, while
additionally stashing the full `low_res_multimasks` tensor and the `ious`
vector for every candidate before they would otherwise be discarded.

Confirmed by reading sam2/sam2/modeling/sam2_base.py directly (this
checkout):

- `_track_step(self, frame_idx, is_init_cond_frame, current_vision_feats,
  current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
  output_dict, num_frames, track_in_reverse, prev_sam_mask_logits)` takes
  11 purely positional arguments (no defaults) and returns a 4-tuple
  `(current_out, sam_outputs, high_res_features, pix_feat)`. The last two
  elements are SAM-head input features and conditioned pixel features,
  respectively -- NOT `high_res_masks`/`obj_ptr` as an earlier draft of
  this module assumed; both are discarded here regardless, but are named
  correctly below for clarity.

- `sam_outputs` unpacks as `(low_res_multimasks, high_res_multimasks,
  ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits)` --
  this part of the earlier draft was correct (confirmed against
  `_forward_sam_heads`'s docstring and return statement).

- `_encode_memory_in_output(self, current_vision_feats, feat_sizes,
  point_inputs, run_mem_encoder, high_res_masks, object_score_logits,
  current_out)` -- confirmed parameter order.

- Critically, the *caller* of `track_step`
  (`SAM2VideoPredictor._run_single_frame_inference`, in
  sam2/sam2/sam2_video_predictor.py) invokes it with ALL keyword
  arguments (`self.track_step(frame_idx=..., is_init_cond_frame=..., ...)`),
  not positionally. An earlier draft of this patch assumed positional
  args (`args[0]`, `args[8]`, etc.), which would raise `IndexError` on
  the very first call in real usage, since `args` is empty in practice.
  To be correct under both calling conventions, the patch below binds
  `*args, **kwargs` against `track_step`'s real signature via
  `inspect.signature(...).bind(...)` and reads every value by name from
  the bound arguments, rather than guessing tuple positions.

`propagate_in_video` calls `_run_single_frame_inference` (hence
`track_step`) once per object per frame, iterating `obj_idx` in
increasing order (`for obj_idx in range(batch_size): ...`) within a
single frame's step -- confirmed in sam2_video_predictor.py. Since
`track_step` itself is never told which obj_idx it's running (that
bookkeeping lives one level up, in inference_state), this module
reconstructs obj_idx implicitly: for a given frame_idx, the n-th call
observed is obj_idx == n. This holds for any call path that invokes
`track_step` once per object per frame in ascending obj_idx order,
which includes both `propagate_in_video` and the single-object calls
made by `add_new_points_or_box`/`add_new_mask` on frame 0.

IMPORTANT invariant added in fix round 1 -- auto-clear on reset_state():
this module's `install()` also monkey-patches `predictor.reset_state`
(instance-bound, same technique as `track_step`) so that every call to
`reset_state()` clears `MultimaskCapture`'s internal store after
delegating to the real `reset_state`. This project's established
per-object eval loop (see `scripts/sav_sam2_eval.py` and every other
`sav_*_eval.py` script) calls `predictor.reset_state(state)` once per
object within the same video, and objects in the same video typically
propagate over the same/overlapping frame_idx range. Without an
auto-clear, any caller that does not pop *every* entry it stores before
the next object's `reset_state()` call (e.g. it only pops frames it
cares about -- which is exactly what `sav_sam2_eval.py` does via its
`if frame_idx not in wanted_frames: continue` guard -- or an exception
happens mid-loop before popping) would leave stale entries in the store
that silently collide with the next object's identical `(obj_idx,
frame_idx)` keys, and `capture.pop(0, frame_idx)` for the new object
would silently return the *previous* object's stale data with no error
at all. With the auto-clear, entries not popped before the next
`reset_state()` call are simply discarded (freed), never leaked across
objects. `capture.clear()` is also exposed as a public method for
callers that want to reset the capture without touching predictor state.
"""

from __future__ import annotations

import inspect
import types
from dataclasses import dataclass, field


@dataclass
class MultimaskCapture:
    _store: dict[tuple[int, int], dict] = field(default_factory=dict)

    def pop(self, obj_idx: int, frame_idx: int) -> dict:
        return self._store.pop((obj_idx, frame_idx))

    def clear(self) -> None:
        """Discard all captured-but-unpopped entries.

        Called automatically by the patched `reset_state` installed by
        `install()` below -- see the module docstring's "auto-clear on
        reset_state()" invariant.
        """
        self._store.clear()


def install(predictor) -> MultimaskCapture:
    """Monkey-patch `predictor.track_step` (instance-bound, not the class)
    to additionally capture the discarded multi-mask candidates, and
    monkey-patch `predictor.reset_state` (same technique) so the capture
    auto-clears whenever the predictor's tracking state is reset -- see
    the module docstring's "auto-clear on reset_state()" invariant.

    Returns a `MultimaskCapture` whose `.pop(obj_idx, frame_idx)` yields a
    dict with:
      - "ious": list[float] of length M, SAM2's own IoU estimate for each
        candidate mask.
      - "low_res_multimasks": CPU float tensor of shape [M, H, W], the
        corresponding low-res mask logits for each candidate.
    """
    capture = MultimaskCapture()
    original_track_step = predictor.track_step.__func__
    original_reset_state = predictor.reset_state.__func__
    track_step_sig = inspect.signature(original_track_step)

    def patched_track_step(self, *args, **kwargs):
        bound = track_step_sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments

        current_out, sam_outputs, _high_res_features, _pix_feat = self._track_step(
            a["frame_idx"],
            a["is_init_cond_frame"],
            a["current_vision_feats"],
            a["current_vision_pos_embeds"],
            a["feat_sizes"],
            a["point_inputs"],
            a["mask_inputs"],
            a["output_dict"],
            a["num_frames"],
            a["track_in_reverse"],
            a["prev_sam_mask_logits"],
        )

        (
            low_res_multimasks,
            _high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        # Reproduce track_step's own reduction to the best-mask-only
        # current_out, exactly as sam2_base.py does it.
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            current_out["object_score_logits"] = object_score_logits

        # This project always propagates one object at a time (batch_size=1
        # at every call site -- see the module docstring), so `ious` and
        # `low_res_multimasks` both carry a leading batch dim of size 1.
        # Fail loudly rather than silently capturing only slice 0 of a
        # larger batch if that assumption is ever violated.
        assert ious.shape[0] == 1, (
            f"MultimaskCapture assumes batch_size == 1 (one object tracked "
            f"per track_step call), got ious.shape[0] == {ious.shape[0]}"
        )
        assert low_res_multimasks.shape[0] == 1, (
            f"MultimaskCapture assumes batch_size == 1 (one object tracked "
            f"per track_step call), got low_res_multimasks.shape[0] == "
            f"{low_res_multimasks.shape[0]}"
        )

        frame_idx = a["frame_idx"]
        # obj_idx is implicit: track_step is called once per object per
        # frame, in ascending obj_idx order, inside propagate_in_video's
        # (and add_new_points_or_box's) inner loop -- capture keyed by a
        # monotonic call counter per frame_idx reconstructs it without
        # needing obj_idx threaded through track_step's own signature.
        key_n = sum(1 for k in capture._store if k[1] == frame_idx)
        capture._store[(key_n, frame_idx)] = {
            "ious": ious.detach().float().cpu().tolist()[0],
            "low_res_multimasks": low_res_multimasks.detach().float().cpu()[0],
        }

        self._encode_memory_in_output(
            a["current_vision_feats"],
            a["feat_sizes"],
            a["point_inputs"],
            a["run_mem_encoder"],
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    def patched_reset_state(self, inference_state):
        result = original_reset_state(self, inference_state)
        capture.clear()
        return result

    predictor.track_step = types.MethodType(patched_track_step, predictor)
    predictor.reset_state = types.MethodType(patched_reset_state, predictor)
    predictor._original_track_step = original_track_step
    predictor._original_reset_state = original_reset_state
    return capture


def uninstall(predictor) -> None:
    """Restore `predictor.track_step` and `predictor.reset_state` to the
    originals saved by `install()`, undoing both monkey-patches.

    Raises `AttributeError` if `install()` was never called on this
    predictor instance -- there is nothing to restore in that case.
    """
    predictor.track_step = types.MethodType(
        predictor._original_track_step, predictor
    )
    predictor.reset_state = types.MethodType(
        predictor._original_reset_state, predictor
    )
    del predictor._original_track_step
    del predictor._original_reset_state
