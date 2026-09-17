# Phase B, object-pointer anchor recall (isolated from Phase A) — Implementation Plan

> **For agentic workers:** Execute task-by-task, in order. Tasks 1-3 and 5
> are committed regardless of outcome. Task 4 (full-val run) is
> **conditional on Task 3's pilot gate passing**. Task 6 (documenting a
> combined Phase A + anchor follow-on hypothesis) is only sketched, not
> implemented, and only if Task 4 is genuinely positive — same discipline
> as `sam2-memory-gating-plan.md`.

**Goal:** Test whether a dynamically-updated "identity anchor" object
pointer — substituted into SAM2's existing object-pointer conditioning
channel when a distractor is suspected — improves propagation robustness,
tested **alone** (no Phase A memory-gating active), against the plain
`box_centroid` baseline (val 72.1, n=30 known-drift-subset 26.1).

**Architecture:** Per `sam2-memory-gating-phase-b-design.md`, this uses
SAM2's object-pointer channel (`sam2_base.py:585-648`), not the
maskmem/spatial-feature channel DAM4SAM's own DRM uses — because the
object-pointer channel's positional encoding is a continuous sine
function the model already evaluates at out-of-normal-range distances on
every long video (via the conditioning-frame pointer, confirmed by direct
code reading), unlike the maskmem channel's small fixed lookup table.
Mechanism: maintain one `(frame_idx, obj_ptr)` anchor pair per object,
updated to the most recent "clean" frame (low divergence, high primary-mask
IoU); when the *previous* frame's divergence was high (distractor
suspected), substitute the anchor into the farthest (`t_diff=15`) slot of
the existing 15-slot object-pointer recency window for the *current*
frame's memory read — replacing, not adding, so the total object-pointer
count never exceeds the trained maximum (16).

**Tech Stack:** Python, PyTorch, SAM2.1 Hiera-Tiny (frozen). Official
`sam2/sav_dataset/sav_evaluator.py`. No new dependencies.

**Spec:** `sam2-memory-gating-phase-b-design.md` (read first — it derives
and justifies every architectural choice below from the actual installed
`sam2/sam2/modeling/sam2_base.py`, not from guessing).

## Global Constraints

- **No fine-tuning, no GT at inference.** Pure runtime state substitution
  (which `obj_ptr` occupies a recency slot); no weights touched.
- **Official evaluator only** for every J&F number.
- **Isolate this mechanism's own effect — no Phase A, no P5.** Per the
  design note §4's explicit recommendation: test the anchor alone before
  any combination, for the same attribution reason Phase A was isolated
  from P5. Do NOT call `sam2_memory_gate.install_memory_gate` anywhere in
  this plan's scripts.
- **Reuse, don't duplicate.** Reuse `sam2_multimask_capture.install`/
  `MultimaskCapture` and `sam2_memory_gate.per_frame_divergence` (already
  built and reviewed) for the "is this frame clean / is a distractor
  suspected" signal. Do not reimplement divergence computation a third
  time.
- **Don't touch the vendored `sam2/` clone.**
- **Replace, never add, object-pointer slots.** The injected anchor must
  occupy the existing `t_diff = max_obj_ptrs_in_encoder - 1` slot, not
  extend the list — this is the specific choice that keeps total
  object-pointer count at the trained maximum (see design note §2).
- **Pre-committed decision gates (do not adjust after seeing data),
  mirroring `sam2-memory-gating-plan.md`'s own bars exactly, since this is
  the same class of check on the same benchmark:**
  - *Pilot gate (Task 3 → Task 4):* full-val proceeds only if the best
    swept configuration's n=30 J&F is **≥ 26.1 + 3.0pp = 29.1**, AND no
    single object regresses more than **15pp** versus its own
    `box_centroid`-only score.
  - *Positive-outcome bar (Task 4 → Task 6):* val J&F **≥ 72.1 + 1.0pp =
    73.1** to justify sketching a combined Phase A + anchor follow-on.

---

## File Structure

- **Create:** `scripts/sam2_anchor_recall.py` — the anchor-tracking and
  injection monkey-patch module. Two coordinated patches on one predictor
  instance: (a) wraps `propagate_in_video` to update the stored anchor
  after each frame's decode (reusing the exact byte-faithful
  reimplementation technique from `sam2_memory_gate.py`, since this is
  also a generator method requiring the same `@torch.inference_mode()`
  care); (b) wraps `_prepare_memory_conditioned_features` (a plain method,
  not a generator — simpler patch) to splice the anchor into the
  `t_diff=15` object-pointer slot when the previous frame's divergence was
  high.
- **Create:** `scripts/sav_anchor_recall_eval.py` — n=30 pilot evaluator,
  sibling of `scripts/sav_memory_gate_eval.py` (same
  `load_drift_failure_targets`-based filtering, same PNG-writing
  convention), applying `install_anchor_recall` instead of
  `install_memory_gate`.
- **Create:** `scripts/sav_anchor_recall_full_eval.py` — full-val
  evaluator, sibling of `scripts/sav_memory_gate_full_eval.py`.
- **Modify:** `sam2-evidence-matrix.md`, `sam2-drift-recovery-next-directions.md`
  — document the outcome (Task 5).

## Interfaces

- `sam2_anchor_recall.AnchorState`: dataclass with `anchor: tuple[int, torch.Tensor] | None`
  (frame_idx, obj_ptr), `last_divergence: float | None`, `n_updates: int`,
  `n_injections: int`, and `reset() -> None`.
- `sam2_anchor_recall.install_anchor_recall(predictor, capture: MultimaskCapture, clean_divergence_threshold: float, clean_iou_threshold: float, inject_divergence_threshold: float) -> AnchorState`:
  installs both patches, returns the stats/state object. `clean_*`
  thresholds gate anchor UPDATES (default clean_divergence_threshold=0.1,
  well below Phase A's 0.2 "suspect" cutoff — the anchor should only
  update on frames that are clearly clean, not merely "not flagged
  suspect"; clean_iou_threshold=0.8, matching DAM4SAM's own stability
  bar). `inject_divergence_threshold` gates anchor INJECTION (default
  0.2, reusing Phase A's winning pilot threshold, since it's the same
  underlying signal already calibrated once on this benchmark).
- `sav_anchor_recall_eval.evaluate_object_with_anchor_recall(...) -> tuple[int, int, int, int]`:
  returns `(frames_written, n_updates, n_injections, n_frames)` — one more
  value than Phase A's evaluator (both update and injection counts matter
  here, unlike Phase A's single skip counter).

---

### Task 1: Anchor-tracking and injection monkey-patch module

**Files:**
- Create: `scripts/sam2_anchor_recall.py`

**Interfaces:**
- Consumes: `sam2_multimask_capture.MultimaskCapture`,
  `sam2_memory_gate.per_frame_divergence` (both already built, reviewed).
- Produces: `AnchorState`, `install_anchor_recall` (see above).

- [ ] **Step 1: Re-verify the two target methods against the actual
  installed file before writing a single line of patch code.** This
  project's own history (Task 1 of both prior plans in this series) found
  a real bug each time by skipping this step and trusting a draft. Read,
  directly from `/home/consul/autoresearch/sam2/sam2/modeling/sam2_base.py`:
  - `_prepare_memory_conditioned_features`'s full body (currently around
    lines 497-650, but re-verify — line numbers drift) — specifically the
    object-pointer construction block (`if self.use_obj_ptrs_in_encoder:`
    through `num_obj_ptr_tokens = obj_ptrs.shape[0]`), to confirm the
    exact variable names and control flow this plan's code below assumes.
  - `_track_step`'s exact call to `self._prepare_memory_conditioned_features(...)`
    (currently around line 761) to confirm the keyword arguments this
    plan's patch signature must match.
  - Confirm `_prepare_memory_conditioned_features` is a plain method (no
    `@torch.inference_mode()` decorator of its own) — it should inherit
    the caller's inference-mode context transparently, since it is only
    ever reached through `_track_step` → `track_step` →
    `_run_single_frame_inference`, called from within
    `propagate_in_video`'s own `@torch.inference_mode()` context
    (confirmed in `sam2_memory_gate.py`'s Task 1). If this method (or
    anything it calls) turns out to have its own decoration or
    training-mode branch that changes this assumption, stop and report it
    rather than guessing.

- [ ] **Step 2: Write the module**

```python
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
        anchor_frame_idx, anchor_obj_ptr = state.anchor
        max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
        t_diff_farthest = max_obj_ptrs_in_encoder - 1
        farthest_t = (
            frame_idx + t_diff_farthest if track_in_reverse
            else frame_idx - t_diff_farthest
        )
        non_cond = output_dict["non_cond_frame_outputs"]
        had_farthest = farthest_t in non_cond
        original_farthest_entry = non_cond.get(farthest_t)
        non_cond[farthest_t] = {"obj_ptr": anchor_obj_ptr}
        try:
            result = original_prepare(
                self, frame_idx, is_init_cond_frame, current_vision_feats,
                current_vision_pos_embeds, feat_sizes, output_dict,
                num_frames, track_in_reverse,
            )
        finally:
            if had_farthest:
                non_cond[farthest_t] = original_farthest_entry
            else:
                del non_cond[farthest_t]
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

    return state
```

  **Why the injection patch temporarily mutates and restores
  `non_cond_frame_outputs[farthest_t]` instead of re-deriving the
  object-pointer-list-building loop:** the original function's loop
  (`for t_diff in range(1, max_obj_ptrs_in_encoder): ... output_dict["non_cond_frame_outputs"].get(t, ...)`)
  already does exactly the lookup we want to redirect. Reimplementing
  that whole block (as `sam2_memory_gate.py` had to for `propagate_in_video`,
  because that function has no smaller sub-call to redirect) would risk
  the same class of bug — a subtle mismatch with a future SAM2 change —
  for no benefit here, since this function's read of
  `non_cond_frame_outputs` is a single dict `.get()` we can transparently
  intercept by substituting the dict's contents for the duration of one
  call. This is a narrower, lower-risk patch than `propagate_in_video`'s
  full reimplementation.

- [ ] **Step 3: Sanity-check on GPU with real data**

```
source .venv/bin/activate
python3 -c "
import torch
from sam2.build_sam import build_sam2_video_predictor
import sam2_multimask_capture, sam2_anchor_recall

predictor = build_sam2_video_predictor(
    'configs/sam2.1/sam2.1_hiera_t.yaml',
    'sam2/checkpoints/sam2.1_hiera_tiny.pt', device='cuda',
)
capture = sam2_multimask_capture.install(predictor)
state = sam2_anchor_recall.install_anchor_recall(predictor, capture)

video_state = predictor.init_state('data/sav/sav_val/JPEGImages_24fps/sav_000262')
predictor.add_new_points_or_box(video_state, frame_idx=0, obj_id=0, box=[100, 100, 200, 200])
for frame_idx, obj_ids, masks in predictor.propagate_in_video(video_state, start_frame_idx=1, max_frame_num_to_track=60):
    pass
print(f'n_updates={state.n_updates} n_injections={state.n_injections} anchor={state.anchor is not None}')
"
```
  Run from `scripts/` (or with `sys.path` adjusted). Expected: no
  exceptions; `n_updates > 0` (a 60-frame real object should have at
  least some clean frames); if `n_injections > 0`, manually verify by
  adding a print inside the injection branch (temporarily, for this check
  only) that `farthest_t` is restored correctly afterward — i.e., run the
  same 60 frames a second time in a fresh `init_state` and confirm
  identical output masks both times (determinism check: the
  temporarily-mutated dict must be fully restored, or a second run would
  see corrupted state and diverge).

- [ ] **Step 4: Commit**

```bash
git add scripts/sam2_anchor_recall.py
git commit -m "Add Phase B object-pointer anchor-recall monkey-patch"
```

---

### Task 2: n=30 pilot evaluator

**Files:**
- Create: `scripts/sav_anchor_recall_eval.py`

Mirror `scripts/sav_memory_gate_eval.py` exactly (same imports from
`sav_reprompt_eval.load_drift_failure_targets`, same
`annotated_frame_indices`-gated PNG writing, same CLI arg names), with:
- `install_anchor_recall(predictor, capture)` instead of
  `install_memory_gate(predictor, capture, threshold)` (use the module's
  own defaults for the three thresholds initially — do not expose all
  three as CLI sweep dimensions in the first pass; sweep only
  `--inject-divergence-threshold` over {0.2, 0.3} first, since that's the
  one directly inherited from Phase A's already-validated signal, and the
  `clean_*` thresholds are new and less critical to tune first).
- `evaluate_object_with_anchor_recall(...) -> (frames_written, n_updates, n_injections, n_frames)`
  — one more return value than Phase A's evaluator, per the Interfaces
  section above. Call `state.reset()` (not `gate.reset_counts()`) at the
  same point Phase A calls its reset — right after `predictor.reset_state(state)`.

- [ ] **Step 1: Write the script** (adapt `sav_memory_gate_eval.py`'s
  structure per the differences above — do not paste it verbatim without
  checking the actual current `AnchorState`/`install_anchor_recall`
  signature from Task 1's committed file, since a plan's code block is
  always a draft, not gospel, per this project's own established
  discipline).
- [ ] **Step 2: Run the 2-value sweep** (`--inject-divergence-threshold`
  0.2 and 0.3) on the n=30 subset, score each with the official evaluator
  against `results/sav_drift_subset_gt`.
- [ ] **Step 3: Commit** script + both result CSVs.

---

### Task 3: Apply the pilot gate

Same mechanics as `sam2-memory-gating-plan.md`'s Task 3: identify the
best swept configuration, check both pre-committed bars (≥29.1, worst
regression ≥-15pp) against `results/sav_drift_subset_baseline_pred/results.csv`.
Pass → Task 4. Fail → skip to Task 5, closed at pilot stage.

---

### Task 4 (CONDITIONAL on Task 3 passing): full-val run

Mirror `sam2-memory-gating-plan.md`'s Task 4 exactly: create
`scripts/sav_anchor_recall_full_eval.py` (import
`evaluate_object_with_anchor_recall`, all-objects loop, `box_centroid`
default, no P5 correction anywhere), smoke-test with `--max-videos 2`
first (bounded subagent task), then run the actual 293-object job
**controller-owned, background + silent single-notification monitor** —
per this project's own established finding (three separate incidents
across two prior plans in this series) that dispatched implementer
subagents reliably mishandle a multi-hour foreground wait. Do not
re-attempt having a subagent run this end-to-end; split exactly as
`sam2-memory-gating-plan.md`'s Task 4 was split.

---

### Task 5: Document the outcome (any branch)

Same format as `sam2-memory-gating-plan.md`'s Task 5: add a row to
`sam2-evidence-matrix.md`, append the next numbered section to
`sam2-drift-recovery-next-directions.md` (Реалізація/Результат/Висновок/
Стан серії), explicitly noting this tested the anchor **alone** (Phase A
inactive), and whether Task 6 was triggered.

---

### Task 6 (CONDITIONAL on Task 4 clearing val J&F ≥ 73.1): sketch a combined Phase A + anchor follow-on

Do not implement here — per the design note §4, this is the natural next
hypothesis (recall compensates for forgetting) but deserves its own
scoping note once there's a real anchor-alone result to react to, not a
speculative combination written before either mechanism's individual
effect is known.

## Self-Review

**Spec coverage:** Design note's mechanism (§2) → Task 1. Design note's
"test alone first" recommendation (§4) → Global Constraints + explicit
"do NOT call install_memory_gate" instruction. Pre-committed gates →
Tasks 3/4/6, using the same bars as the sibling plan since it's the same
benchmark and same class of check.

**Placeholder scan:** No TBD/TODO. Task 2/4's "adapt, don't paste
verbatim" instruction is a deliberate discipline note, not a placeholder
— it names exactly what must be re-verified (the real committed Task 1
signatures) before use, mirroring language already validated as
effective in the sibling plan's own Task 2.

**Type/name consistency:** `install_anchor_recall(predictor, capture, clean_divergence_threshold, clean_iou_threshold, inject_divergence_threshold) -> AnchorState`
(Task 1) is the single call site Task 2/4 both use.
`evaluate_object_with_anchor_recall`'s 4-tuple return
(`frames_written, n_updates, n_injections, n_frames`) is named
consistently between Task 2's definition and Task 4's expected unpack.
