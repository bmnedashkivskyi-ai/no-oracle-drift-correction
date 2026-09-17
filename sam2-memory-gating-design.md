# Dynamic memory-gating (DAM4SAM-style) — research & design note

Status: **research/design only** — no code written, no GPU time spent. This
document exists to decide *what* to build before committing to an execution
plan (compare `sam2-multimask-divergence-plan.md`'s own Task/Step breakdown,
which this precedes).

## 1. Why this is a different, harder question than the closed check

`sam2-drift-recovery-next-directions.md` §13 tested whether DAM4SAM's
*introspective signal* (divergence between SAM2's primary and best-alternative
candidate mask) predicts, as a **static per-object feature**, whether an
object needs P5 correction. It doesn't (R²=0.028). That result says nothing
about DAM4SAM's actual mechanism, which never makes a static per-object
decision at all — it makes a **dynamic, per-frame decision about what enters
memory**, continuously, for the whole video. Testing that is a materially
different and larger undertaking: it requires changing what the tracker
*remembers*, not just deciding once whether to run a correction.

## 2. What DAM4SAM actually does (Videnovic et al., 2024)

- **Dual memory.** Recent Appearance Memory (RAM): last 3 frames, sampled
  every 5, temporally encoded. Distractor-Resolving Memory (DRM): anchor
  frames with a confirmed-clean target *and* a visible distractor, no
  temporal encoding (treated as timeless priors).
- **Distractor detection, no GT.** Compare the primary mask's bbox against
  `union(primary bbox, largest-alternative-mask bbox)`. If the ratio drops
  below `θ_anc = 0.7`, a distractor is present. This is the same signal
  already captured non-invasively in this project's `sam2_multimask_capture.py`
  (Task 1 of the closed check) — that infrastructure is reusable here.
- **Update gating.** Only update RAM when tracking is stable (predicted IoU
  > 0.8, mask area within ±20% of recent median) — skip empty/unstable
  frames entirely. Insert into DRM proactively, under the same stability
  condition, the moment a distractor is detected — *before* failure, not
  after.
- **Reported gains** are distractor-specific benchmarks (DiDi +6.9% quality,
  VOT2022 +8.8% EAO), not SA-V, and not this project's frozen Hiera-Tiny
  checkpoint.

## 3. What SAM2's memory mechanism actually looks like (verified against this checkout)

Read directly: `sam2/sam2/modeling/sam2_base.py:497-650`
(`_prepare_memory_conditioned_features`), `sam2/sam2/sam2_video_predictor.py`
(`track_step`, `non_cond_frame_outputs` bookkeeping).

- **The memory window is a fixed recency schedule, not a content-based
  selector.** For a frame at `frame_idx`, the model looks up
  `output_dict["non_cond_frame_outputs"][prev_frame_idx]` at deterministic
  offsets (`frame_idx-1`, then every `memory_temporal_stride_for_eval`-th
  frame going back, `num_maskmem=7` slots total: 1 conditioning frame + 6
  recency frames). There is no scoring, ranking, or content check anywhere
  in this lookup.
- **A missing slot is silently skipped, not backfilled.** `sam2_base.py:571-572`:
  `if prev is None: continue  # skip padding frames`. If a frame's output
  was never written to `non_cond_frame_outputs`, the model just attends to
  fewer memory tokens that step — this already happens naturally near the
  start of a video, so the model is not being pushed outside its training
  distribution by an *occasional* missing slot.
- **Object pointers follow the same fixed-recency pattern** (`sam2_base.py:611-620`,
  up to `max_obj_ptrs_in_encoder=16` most-recent frames), independent of
  the mask memory loop.
- **Every past frame's memory contribution carries a temporal positional
  encoding keyed to its recency slot** (`self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]`,
  `sam2_base.py:581-583`) — a small, fixed, *learned* embedding table of
  exactly `num_maskmem` entries. This table has no entry for "this frame is
  an out-of-schedule anchor from 200 frames ago."

## 4. Two implementations, two risk profiles

### Phase A — minimal: skip writing memory for suspected-distractor frames

Mechanism: reuse `sam2_multimask_capture.py`'s divergence signal; when a
frame's divergence exceeds threshold, **don't call `_encode_memory_in_output`**
for that frame (or discard its result before it's stored), so it's absent
from `non_cond_frame_outputs`. Everything downstream is native SAM2
behavior — the `if prev is None: continue` path this model already
exercises routinely. No new positional-encoding slot, no attention-shape
change, no token the model wasn't designed to receive.

- **What this tests:** whether *shrinking* the memory window around
  suspected distractors (closer to DAM4SAM's RAM stability gate) helps,
  independent of DRM's anchor-recall behavior.
- **Risk:** low. This is instrumentation-level, similar in kind to Task 1
  of the closed check (monkey-patch `track_step`, skip a dict write).
- **Known limitation this can't test:** DAM4SAM's actual gain in the
  DiDi/VOT2022 numbers cited above comes substantially from DRM's
  *recall* of a distractor-resolving anchor, not merely from RAM's
  *forgetting* of unstable frames. Phase A tests only the forgetting half.
- **Side effect beyond the maskmem bank:** skipping the frame's write to
  `non_cond_frame_outputs` also removes it as a source of `obj_ptr`s for
  `_prepare_memory_conditioned_features`'s object-pointer cross-attention
  (`sam2_base.py:611-620`, up to `max_obj_ptrs_in_encoder=16` recent
  frames) — so "skip writing memory" here means suppressing both the
  maskmem-attention channel and the object-pointer channel for that
  frame, not the maskmem bank alone.

### Phase B — full: inject a DRM-style anchor frame into the attention window

Mechanism: maintain a small persistent set of "anchor" frame indices
(confirmed-clean + distractor-visible, per DAM4SAM's insertion rule);
when a distractor is detected, splice that anchor's stored `maskmem_features`
into `to_cat_memory` / `to_cat_memory_pos_embed` in
`_prepare_memory_conditioned_features` *in addition to* the normal recency
window — i.e., give the frozen transformer more memory tokens than
`num_maskmem` was designed to carry.

- **What this tests:** the actual DAM4SAM mechanism, including the part
  Phase A cannot reach.
- **Risk:** materially higher, for a specific, nameable reason: the anchor
  token needs *some* temporal positional encoding, and
  `self.maskmem_tpos_enc` only has `num_maskmem` learned entries, none of
  which mean "arbitrary distance in the past." Candidate mitigations —
  reusing the `t_pos=num_maskmem-1` (furthest) slot's embedding for the
  anchor, or omitting positional encoding for it entirely — are both
  *guesses* about how a frozen, never-retrained-for-this transformer will
  respond to an out-of-schedule token. This is exactly the kind of
  uncertainty this project's own methodology (Section 3.5 of the paper)
  says should be checked cheaply before a full run, not assumed.
- **This is also the point at which "no fine-tuning, frozen model" gets
  closest to its edge.** Weights are still untouched, but the input
  distribution to the frozen attention layers would go outside what any
  training example looked like. That's a legitimate thing to try — SAM2Long
  and DAM4SAM both do variants of it — but it is a qualitatively different
  claim than everything else in this project's series, which has stayed
  strictly within trained input shapes.

## 5. Recommendation

Sequence, don't choose: **Phase A first, as a cheap gate for Phase B**,
mirroring this project's own established discipline (n=30 pilot before
full run; retrospective correlation before live gate; the multi-mask
divergence check itself). Concretely:

1. Phase A on the n=30 known-drift subset (cheap, reuses existing
   instrumentation and prompt policy). If it doesn't at least match D4
   (val 71.3, the current best), Phase B's added engineering risk isn't
   justified by anything Phase A found — DRM's specific contribution
   (anchor recall) would need to be motivated on its own, not inherited
   from a Phase A win that never happened.
2. Only if Phase A clears that bar: design Phase B's positional-encoding
   handling explicitly (pick one candidate mitigation, state the
   hypothesis about why it should work, don't just try it and see), then
   pilot it on the same n=30 subset before any full-val run.

This note stops here — no task breakdown, no scripts, no GPU commitment.
Say the word for a full execution plan (Task/Step breakdown in the style
of `sam2-multimask-divergence-plan.md`) once you've decided whether to
scope this as Phase A only, or Phase A with Phase B conditional.
