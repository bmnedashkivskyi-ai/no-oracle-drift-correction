# Phase B design resolution — DRM-style anchor recall via the object-pointer channel

Status: **research/design only**, resolving the specific blocker
`sam2-memory-gating-design.md` §4 Phase B left open ("the anchor token
needs *some* temporal positional encoding, and `self.maskmem_tpos_enc`
only has `num_maskmem` learned entries... candidate mitigations are both
*guesses*"). This note replaces guessing with a mechanism-grounded choice,
verified against this checkout's actual code and the actual checkpoint
config (`sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml`).

## 1. The finding that resolves the blocker

DAM4SAM's DRM injects an anchor into the **spatial memory-feature**
channel (`to_cat_memory`/`to_cat_memory_pos_embed`, fed by
`maskmem_features` + `self.maskmem_tpos_enc`). That channel's positional
table genuinely has no representation for "arbitrary distance" —
confirmed in the prior design note.

But SAM2 has a **second**, independent memory-conditioning channel: the
**object-pointer** channel (`sam2_base.py:585-648`), gated by
`use_obj_ptrs_in_encoder`. Reading it directly reveals two facts the
prior design note didn't have:

1. **Its positional encoding is a continuous sine function, not a fixed
   lookup table.** `obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)`
   (`sam2_base.py:634`) — `get_1d_sine_pe` accepts any real number, not a
   `num_maskmem`-sized discrete index. There is no analogue of
   `maskmem_tpos_enc`'s hard size limit here.

2. **The model already receives out-of-training-range values on this
   exact code path, on every video longer than ~16 frames, as normal
   operation — not hypothetically, but confirmed by direct code reading.**
   The conditioning (first-prompted) frame's object pointer is added to
   `pos_and_ptrs` with distance `(frame_idx - t) * tpos_sign_mul`
   (`sam2_base.py:596-608`), **unclamped** — `t` here is the conditioning
   frame's own index (typically `0`), so on frame 500 of a video this
   value is `500`. It is then divided by the SAME fixed
   `t_diff_max = max_obj_ptrs_in_encoder - 1 = 15`
   (`sam2_base.py:625, 634` — confirmed `max_obj_ptrs_in_encoder` is not
   overridden in `sam2.1_hiera_t.yaml`, so it takes the class default 16)
   as every other pointer. `500 / 15 ≈ 33.3` — nowhere near the `[0, 1]`
   range the "recent" pointers (`t_diff` 1-15) occupy. Every one of this
   project's own full-val objects (n_frames routinely in the hundreds —
   e.g. `sav_000262` at 528 frames, seen throughout this project's own
   result CSVs) already exercises this out-of-`[0,1]` input on every
   propagated frame, via the conditioning-frame pointer that
   `ptr_cond_outputs` always includes regardless of `only_obj_ptrs_in_the_past_for_eval`
   (`sam2_base.py:590-599` — that flag only filters by *direction*, past
   vs. future, never by *distance*).

**Consequence:** injecting a second, dynamically-chosen anchor frame's
object pointer, using the same unclamped `(frame_idx - anchor_idx)`
distance formula already used for the conditioning frame, exercises a
code path the frozen model already runs on essentially every video in
this evaluation, not a genuinely novel input distribution. This is
qualitatively different from Phase B's original maskmem-channel proposal,
where no comparable "already handles this" argument existed.

## 2. What Phase B now proposes, precisely

**Not** DAM4SAM's literal DRM (which uses the spatial-feature channel).
An **adaptation**: a dynamically-updated anchor **object pointer**,
inserted via the channel already designed to carry an out-of-recency-window
reference (the conditioning-frame pointer already does this for frame 0 —
Phase B generalizes it to a *later, better* anchor frame, chosen while the
video is running, rather than only ever the original prompt frame).

- **Anchor selection (mirrors DAM4SAM's DRM insertion rule, reusing
  already-built infrastructure):** maintain one anchor `(frame_idx,
  obj_ptr)` pair per object. Update it whenever the current frame is
  judged "clean" — reuse this project's existing per-frame divergence
  signal (`sam2_multimask_capture.py`, already built and reviewed) as the
  distractor-visibility check, plus a stability check on primary-mask IoU
  (already available from `capture`'s `ious[best_idx]`, no new
  instrumentation needed) as the "confirmed correct" check. This mirrors
  DAM4SAM's own two-part condition (distractor visible + tracking stable)
  without inventing a new signal.
- **Injection (the load-bearing design choice):** when the PREVIOUS frame's
  divergence signal suggests a distractor, **replace** the
  farthest-in-time slot of the existing 15-slot recency window (`t_diff =
  max_obj_ptrs_in_encoder - 1 = 15`, the least locally-informative of the
  15 "recent" pointers by construction) with the stored anchor's
  `(anchor_frame_idx, obj_ptr)` pair, using the unclamped
  `frame_idx - anchor_frame_idx` distance in the same sine-PE formula the
  conditioning frame already uses. **Replace, don't add** — total
  object-pointer count stays at the trained maximum (16: 1 conditioning +
  15 recency), so this changes *which* content occupies one slot, not
  *how many* tokens the cross-attention receives. This mirrors Phase A's
  own "skip/replace, never add" philosophy for the maskmem channel.
- **No maskmem-channel change at all in this variant.** The spatial
  memory-feature bank behaves exactly as in the unmodified model (or, if
  combined with Phase A, exactly as Phase A left it). This is a
  deliberately narrow first test: does giving the model back an
  *identity* reference (a pointer that says "this is what the target
  looked like when we were sure") help when a distractor is suspected,
  independent of anything touching the spatial feature bank.

**Implementation note (added after Phase B's first test, 2026-09-16):**
the actual first implementation of this mechanism
(`scripts/sam2_anchor_recall.py`; see `sam2-drift-recovery-next-directions.md`
§15) does **not** use the anchor's true unclamped `frame_idx -
anchor_frame_idx` distance described above. It splices the anchor's
object-pointer *content* into the farthest recency slot but leaves that
slot's positional-distance label at its normal fixed value (`t_diff =
max_obj_ptrs_in_encoder - 1 = 15`) — `anchor_frame_idx` is unpacked from
the stored anchor state but never fed into the sine-PE distance
computation. The pilot's negative result (§15) therefore rules out only
this narrower content-substitution variant. Testing the true-distance
variant this section describes — the one §1's positional-encoding
argument is actually about — remains open if this direction is
revisited.

## 3. Honesty about what this is and isn't

This is **not** a replication of DAM4SAM's DRM. It is a
mechanism-adapted analogue that substitutes a channel this frozen
checkpoint can accept without extrapolating beyond an input range it
already exercises, in place of the channel DAM4SAM's own paper actually
uses. Any positive or negative result here speaks to "does an
identity-anchor-recall idea help via SAM2's object-pointer conditioning",
not to "does DAM4SAM's published mechanism transfer to this checkpoint" —
the paper-writing convention already established in this project (see
`sam2-drift-recovery-next-directions.md` §13-14's careful "tested the
introspective signal, not the full mechanism" framing) applies here too,
one level further in.

## 4. Should this be combined with Phase A?

Phase A (skip maskmem writes on suspected distractors, alone) was
net-negative at full-val (72.1→71.3). The mechanistic reading documented
in §14 is that pure forgetting, without a compensating recall mechanism,
discards useful context on objects that didn't need help — exactly the
gap DAM4SAM's own two-part design (RAM forgetting + DRM recall together)
exists to close. This gives a specific, falsifiable hypothesis for
combining them: **recall (this note's object-pointer anchor) might
recover what forgetting (Phase A) alone lost**, since the two act on
different channels (object-pointer identity vs. spatial memory features)
and address complementary failure modes (Phase A doesn't corrupt on
healthy objects if the anchor gives the model something better to lean on
during a false-positive skip).

**Recommendation: test the object-pointer anchor alone first** (no Phase A
memory-gating active), for the same reason Phase A was isolated from P5
in the prior plan — attribute any effect to the new mechanism specifically
before combining it with another intervention. If the anchor-alone signal
is promising, a combined Phase A + anchor variant becomes the natural
follow-on with a concrete hypothesis, not a shot in the dark.

## 5. Risk assessment, updated

- **Positional-encoding risk: substantially reduced**, per §1 — the model
  already runs this code path with unclamped, out-of-normal-range distances
  on every long video via the conditioning-frame pointer.
- **New, smaller risk this note surfaces:** the anchor's object pointer
  comes from a *different frame's* visual content than the conditioning
  frame's, so while the *distance encoding* is architecturally familiar,
  the specific *pointer content* pattern (a mid-video "clean" frame acting
  as a second long-range reference) is not something a training example
  necessarily looked like. This is a real but narrower uncertainty than
  the original Phase B concern — it's about content, not about the
  positional-encoding mechanism breaking.
- **No fine-tuning, frozen weights, no vendored `sam2/` edits** — same
  constraints as every prior direction in this project. The anchor
  mechanism is pure runtime state substitution (which `obj_ptr` occupies
  a given recency slot), implemented via the same monkey-patch technique
  already used for `track_step`/`reset_state`/`propagate_in_video`.

## 6. Next step

This resolves the design blocker. Say the word for a full execution plan
(Task/Step breakdown in the style of `sam2-memory-gating-plan.md`) —
isolated object-pointer-anchor test first, same n=30-pilot-then-full-val
gate discipline as every prior direction in this project.
