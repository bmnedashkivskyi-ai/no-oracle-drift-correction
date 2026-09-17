"""P5: position/motion-continuity prior against multi-instance identity switches.

Every prior correction attempt (F1-F4 self-mask/consensus reference, F5/P3
external or SAM2-native content-based selection, P4 gating) corrects using
mask CONTENT or CONFIDENCE -- appearance-based signals. Both the LLM pilot
(sam2-drift-recovery-results.md p.6) and the P3 candidate-selection diagnostic
(2026-08-31) independently surfaced the same failure category neither content
nor confidence can ever fix: genuine multi-instance ambiguity, where several
objects in frame are visually near-identical (sav_022396's ~15 similar
balloons, sav_035009's several similar meat pieces, sav_052955's two similar
fish, sav_035221's two shoes) -- so a confident, well-formed candidate can
still be locked onto the WRONG instance, and object_score_logits stays high
because the model isn't uncertain, it's just confidently wrong (see the
objscore caveat in sav_drift_reprompt_objscore_eval.py).

P5's lever is different in kind: POSITION, not appearance. Two near-identical
balloons occupy different places; a constant-velocity extrapolation of the
object's own recent trajectory is the one signal that does NOT get confused by
visual similarity between instances, precisely because it never looks at
pixels at all.

Two components, both novel relative to F1-F6/P1-P4:
  1. Detection: in addition to the existing area-collapse signal (area==0 or
     far off the EMA baseline -- catches "lost track" the way F1 always did),
     add a POSITION-jump signal: extrapolate the expected centroid from the
     last two accepted frames' centroids (constant velocity) and flag drift if
     the actual centroid lands far from that prediction, scaled by the
     object's own size. This is the check area/objscore triggers structurally
     cannot perform -- a confident jump onto a same-sized, same-confidence,
     different-location distractor.
  2. Correction: instead of reinjecting old mask CONTENT (which could itself
     already be sitting on the wrong instance, or stale), issue a FRESH
     box prompt centered on the extrapolated position (size taken from the
     last accepted mask's bounding box) via add_new_points_or_box. SAM2
     re-segments whatever is actually at that location, so the correction is
     anchored in space, not carried forward in shape.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from davis_sam2_eval import prompt_for_object  # noqa: E402
from sav_reprompt_eval import load_drift_failure_targets  # noqa: E402
from sav_sam2_eval import (  # noqa: E402
    annotated_frame_indices,
    collect_object_inputs,
    list_frame_names,
)

from sam2.build_sam import build_sam2_video_predictor


def mask_centroid(mask: torch.Tensor):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(xs) == 0:
        return None
    return (float(xs.float().mean()), float(ys.float().mean()))


def mask_bbox_size(mask: torch.Tensor):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(xs) == 0:
        return None
    return (float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))


def evaluate_object_with_motion_reprompt(
    predictor,
    state,
    object_id: int,
    per_frame: dict[int, np.ndarray],
    frame_names: list[str],
    ann_dir: Path,
    prompt_policy: str,
    obj_out_dir: Path,
    ema_alpha: float,
    low_ratio: float,
    high_ratio: float,
    min_area_eps: float,
    jump_ratio: float,
    max_corrections: int,
    vanish_retry_limit: int = 10**9,
    require_both_triggers: bool = False,
    plausibility_ratio: float = 0.0,
    max_correction_rate: float = 0.0,
    correction_window: int = 0,
    early_trigger_skip_frac: float = 0.0,
    rate_warmup_corrections: int = 3,
    soft_nudge_weight: float = 0.0,
):
    sorted_frames = sorted(per_frame)
    first_frame_idx = sorted_frames[0]
    predictor.reset_state(state)

    prompt = prompt_for_object(per_frame[first_frame_idx], prompt_policy)
    _, _, mask_logits = predictor.add_new_points_or_box(
        state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
    )
    last_mask = mask_logits[0, 0] > 0
    frame_h, frame_w = last_mask.shape
    ema_area = float(last_mask.sum().item())
    last_centroid = mask_centroid(last_mask)
    last_size = mask_bbox_size(last_mask) or (frame_w * 0.1, frame_h * 0.1)
    prev_point = (first_frame_idx, last_centroid)  # for velocity estimation
    velocity = (0.0, 0.0)

    wanted_frames = annotated_frame_indices(ann_dir, object_id, frame_names)
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    if first_frame_idx in wanted_frames:
        arr = last_mask.detach().cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(arr).save(obj_out_dir / f"{frame_names[first_frame_idx]}.png")

    frames_written = 1 if first_frame_idx in wanted_frames else 0
    n_corrections = 0
    n_position_triggers = 0
    n_skipped_vanish = 0
    n_rejected_corrections = 0
    consecutive_vanish = 0
    rate_capped = False
    correction_frame_history: list[int] = []
    current_start = first_frame_idx + 1

    # E1 timing variant (sam2-drift-recovery-next-directions.md SS8-9): a
    # retrospective check on already-known capstone per-object deltas found
    # the raw rate/amplitude features (SS8) carry no signal (R^2=0.026), but
    # WHEN the object's first-ever trigger fires does: objects whose first
    # trigger lands in the first 5% of their own frames average -10pp vs the
    # full triggered population's -3.8pp, and the effect fades smoothly as
    # that fraction grows. Unlike the SS8/E1 features, this doesn't need a
    # separate uncorrected pass: the total frame budget for this object is
    # already known before propagation starts, so the gate can be evaluated
    # inline, in the same single corrected pass, using only the FIRST trigger
    # ever seen -- if it fires too early, permanently stop correcting this
    # object (same "give up" fallthrough as vanish_retry_limit exhaustion)
    # and never re-evaluate the timing check again.
    n_frames_total = len(frame_names) - first_frame_idx
    early_trigger_evaluated = False
    early_trigger_permanently_skip = False
    n_skipped_early_trigger = 0

    while current_start < len(frame_names) and n_corrections < max_corrections and not rate_capped:
        drift_frame_idx = None
        drift_reason = None
        for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=current_start
        ):
            mask = mask_logits[0, 0] > 0
            area = float(mask.sum().item())

            if frame_idx in wanted_frames:
                arr = mask.detach().cpu().numpy().astype(np.uint8) * 255
                Image.fromarray(arr).save(obj_out_dir / f"{frame_names[frame_idx]}.png")
                frames_written += 1

            consecutive_vanish = consecutive_vanish + 1 if area == 0 else 0

            is_vanished = area == 0 and ema_area >= min_area_eps
            is_area_drift = ema_area >= min_area_eps and (
                area == 0 or area / ema_area < low_ratio or area / ema_area > high_ratio
            )

            # Computed unconditionally (not short-circuited on is_area_drift)
            # so --require-both-triggers can AND the two signals together;
            # this doesn't change OR-mode behavior, since drift_reason below
            # already prefers "area"/"vanished" over "position" when both
            # would fire, and the OR itself is insensitive to this value's
            # truth when is_area_drift is already True.
            is_position_drift = False
            centroid = mask_centroid(mask)
            if centroid is not None and last_centroid is not None:
                dt = frame_idx - prev_point[0]
                predicted = (
                    last_centroid[0] + velocity[0] * dt,
                    last_centroid[1] + velocity[1] * dt,
                )
                jump = ((centroid[0] - predicted[0]) ** 2 + (centroid[1] - predicted[1]) ** 2) ** 0.5
                scale = max((ema_area ** 0.5), 1.0)
                if jump > jump_ratio * scale:
                    is_position_drift = True

            # A1 (sam2-drift-recovery-next-directions.md SS3): the capstone
            # full-dataset run found the OR-gate between the two triggers has
            # too high a false-positive rate on objects that don't actually
            # have propagation drift (178/293 regressed vs 44 improved).
            # require_both_triggers demands BOTH signals agree (excluding a
            # genuine vanish, which is unambiguous on its own) before firing,
            # trading recall for specificity.
            if require_both_triggers and not is_vanished:
                is_drift_signal = is_area_drift and is_position_drift
            else:
                is_drift_signal = is_area_drift or is_position_drift

            # A vanished object (area==0) has nothing reliable to anchor a
            # fresh box-reprompt to -- forcing one risks confidently
            # latching onto whatever else is at the extrapolated position
            # (background, a distractor). Unlike self-mask reinjection (F1),
            # a wrong box here doesn't just repeat a stale-but-bounded
            # guess, it actively acquires a new, wrong target that then
            # anchors every subsequent correction.
            #
            # A blanket skip-every-vanish-correction policy was tested and
            # found net-negative on val (J&F 42.3->31.4): most vanish-
            # triggered corrections ARE beneficial (quick, correct
            # reacquisition after a brief real occlusion); only the rare
            # case where several *consecutive* attempts all fail to
            # reacquire indicates a genuinely lost object worth giving up
            # on. So: keep retrying (fresh box-reprompt) for the first
            # `vanish_retry_limit` consecutive vanished frames -- same
            # behavior as the original tuned P5 -- and only stop retrying
            # once that budget is exhausted, at which point ema_area is left
            # to decay via the normal update below (do NOT break here --
            # breaking and re-entering propagate_in_video per skipped frame
            # is O(N^2) over a long invisible stretch) until it drops under
            # min_area_eps and the area-drift check stops firing on its own.
            give_up = is_vanished and consecutive_vanish > vanish_retry_limit
            if give_up:
                n_skipped_vanish += 1
            elif is_drift_signal:
                if early_trigger_skip_frac > 0 and not early_trigger_evaluated:
                    early_trigger_evaluated = True
                    rel_frame = frame_idx - first_frame_idx
                    if rel_frame / max(n_frames_total, 1) < early_trigger_skip_frac:
                        early_trigger_permanently_skip = True
                if early_trigger_permanently_skip:
                    n_skipped_early_trigger += 1
                else:
                    drift_frame_idx = frame_idx
                    drift_reason = "area" if is_area_drift else "position"
                    break

            # accepted (or skipped-vanish) frame: update trajectory and EMA area
            if centroid is not None and last_centroid is not None:
                dt = max(frame_idx - prev_point[0], 1)
                velocity = ((centroid[0] - last_centroid[0]) / dt, (centroid[1] - last_centroid[1]) / dt)
                prev_point = (frame_idx, centroid)
            last_centroid = centroid if centroid is not None else last_centroid
            bbox = mask_bbox_size(mask)
            if bbox is not None:
                last_size = bbox
            last_mask = mask
            ema_area = ema_alpha * area + (1 - ema_alpha) * ema_area

        if drift_frame_idx is None:
            break

        if drift_reason == "position":
            n_position_triggers += 1

        dt = drift_frame_idx - prev_point[0]
        predicted_centroid = (
            last_centroid[0] + velocity[0] * dt,
            last_centroid[1] + velocity[1] * dt,
        )

        # F1 (sam2-drift-recovery-next-directions.md SS3-B2/SS12): every prior
        # correction (P5's original, plus A1/B1/A2) either fires or doesn't --
        # once it fires, the reprompt box is a full hard override centered
        # purely on predicted_centroid, ignoring wherever the model's own
        # (uncorrected) mask currently sits. That's fine for a true positive
        # (the model's own position IS the drift), but for a false-positive
        # trigger on an otherwise-healthy object, `centroid` here already
        # holds a perfectly good position -- overriding it entirely is pure
        # downside. Blending the extrapolated position with the model's own
        # current centroid caps how far a single bad correction can move the
        # anchor, at the cost of a slower full snap-back on a genuine drift.
        # 0.0 (default) disables the blend entirely, preserving prior
        # behavior exactly (box centered purely on predicted_centroid).
        if soft_nudge_weight > 0 and centroid is not None:
            reprompt_centroid = (
                (1 - soft_nudge_weight) * predicted_centroid[0] + soft_nudge_weight * centroid[0],
                (1 - soft_nudge_weight) * predicted_centroid[1] + soft_nudge_weight * centroid[1],
            )
        else:
            reprompt_centroid = predicted_centroid

        half_w, half_h = last_size[0] / 2, last_size[1] / 2
        x0 = max(0.0, reprompt_centroid[0] - half_w)
        y0 = max(0.0, reprompt_centroid[1] - half_h)
        x1 = min(float(frame_w - 1), reprompt_centroid[0] + half_w)
        y1 = min(float(frame_h - 1), reprompt_centroid[1] + half_h)
        if x1 <= x0:
            x1 = min(float(frame_w - 1), x0 + 1)
        if y1 <= y0:
            y1 = min(float(frame_h - 1), y0 + 1)

        _, _, corrected_logits = predictor.add_new_points_or_box(
            state, frame_idx=drift_frame_idx, obj_id=object_id, box=[x0, y0, x1, y1]
        )
        corrected_mask = corrected_logits[0, 0] > 0

        # B1 (sam2-drift-recovery-next-directions.md SS3-B): the two test-set
        # catastrophes (sav_004755, sav_017171) and several capstone
        # regressions share a mechanism the vanish_retry_limit fix couldn't
        # touch -- the box-reprompt SUCCEEDS (SAM2 confidently returns a
        # non-empty mask) but on the WRONG thing, and that wrong mask then
        # anchors every later correction. Unlike the *trigger* (A1), which
        # only decides WHETHER to correct, this checks the correction's
        # RESULT: if the new mask's area is wildly inconsistent with the
        # object's own pre-drift size (pre_ema_area), the extrapolated box
        # most likely landed on a distractor or empty background rather than
        # the real object -- reject it and fall back to a conservative
        # "hold position" box (last known good centroid/size, no
        # extrapolation) instead of trusting the aggressive guess.
        pre_ema_area = ema_area
        corrected_area = float(corrected_mask.sum().item())
        if plausibility_ratio > 0 and pre_ema_area >= min_area_eps:
            ratio = corrected_area / pre_ema_area if pre_ema_area > 0 else float("inf")
            implausible = (
                corrected_area == 0
                or ratio > plausibility_ratio
                or ratio < 1.0 / plausibility_ratio
            )
            if implausible:
                n_rejected_corrections += 1
                fb_half_w, fb_half_h = last_size[0] / 2, last_size[1] / 2
                fx0 = max(0.0, last_centroid[0] - fb_half_w)
                fy0 = max(0.0, last_centroid[1] - fb_half_h)
                fx1 = min(float(frame_w - 1), last_centroid[0] + fb_half_w)
                fy1 = min(float(frame_h - 1), last_centroid[1] + fb_half_h)
                if fx1 <= fx0:
                    fx1 = min(float(frame_w - 1), fx0 + 1)
                if fy1 <= fy0:
                    fy1 = min(float(frame_h - 1), fy0 + 1)
                _, _, fallback_logits = predictor.add_new_points_or_box(
                    state, frame_idx=drift_frame_idx, obj_id=object_id, box=[fx0, fy0, fx1, fy1]
                )
                corrected_mask = fallback_logits[0, 0] > 0

        if drift_frame_idx in wanted_frames:
            arr = corrected_mask.detach().cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(arr).save(obj_out_dir / f"{frame_names[drift_frame_idx]}.png")

        last_mask = corrected_mask
        last_centroid = mask_centroid(corrected_mask) or reprompt_centroid
        bbox = mask_bbox_size(corrected_mask)
        if bbox is not None:
            last_size = bbox
        prev_point = (drift_frame_idx, last_centroid)
        velocity = (0.0, 0.0)
        ema_area = float(corrected_mask.sum().item())
        n_corrections += 1
        correction_frame_history.append(drift_frame_idx)
        current_start = drift_frame_idx + 1

        # A2 (sam2-drift-recovery-next-directions.md): a cheap, retroactively
        # validated object-level circuit breaker. Correlating the OR-gate
        # capstone's per-object correction RATE (corrections / frames elapsed
        # so far) against its per-object J&F delta showed a clean monotonic
        # relationship on the full unselected val set: objects with a low
        # correction rate (<0.05) net IMPROVED on average (+3.3pp), while
        # objects with a high rate (>0.3, the trigger firing on >30% of
        # frames -- thrashing, not a one-off drift event) net regressed
        # severely (-11.9pp). A rate that stays low all video long is what a
        # genuine, isolated identity-switch/occlusion recovery looks like; a
        # persistently high rate means the trigger is chronically misfiring
        # on an object with naturally volatile size/position (the false-
        # positive population the capstone finding identified) and every
        # further correction just compounds the damage. Unlike A1 (per-frame
        # trigger AND-gate) and B1 (per-correction result plausibility,
        # closed net-negative), this acts at the OBJECT level: once an
        # object's own correction history shows it's in the bad regime, stop
        # correcting it entirely for the rest of the video and fall through
        # to the uncorrected tail-propagation logic below -- exactly like
        # exhausting max_corrections, just triggered earlier and adaptively
        # per-object instead of by a fixed global count.
        # D3 (sam2-drift-recovery-next-directions.md SS7): the cumulative rate
        # above "remembers" the whole video since the object's first frame, so
        # an object with 2 corrections in the first 20 frames and none since
        # never trips the breaker no matter how quiet the remaining 480 frames
        # are. A sliding window instead measures recent BURSTS of corrections
        # -- reacting to a chronic thrashing episode as soon as it starts,
        # rather than averaging it away over a long prior quiet stretch.
        # correction_window == 0 (default) preserves the exact cumulative-rate
        # behavior above.
        # D2 (sam2-drift-recovery-next-directions.md SS6): the >=3 warm-up
        # below was hardcoded from the start -- a lower warm-up means fewer
        # "free" corrections before the breaker can possibly intervene, at
        # the cost of a noisier rate estimate (fewer samples) when it first
        # fires. rate_warmup_corrections==3 (default) preserves prior
        # behavior exactly.
        if max_correction_rate > 0 and n_corrections >= rate_warmup_corrections:
            if correction_window > 0:
                window_start = drift_frame_idx - correction_window
                count_in_window = sum(1 for f in correction_frame_history if f > window_start)
                effective_elapsed = min(correction_window, max(drift_frame_idx - first_frame_idx, 1))
                rate = count_in_window / effective_elapsed
            else:
                elapsed = max(drift_frame_idx - first_frame_idx, 1)
                rate = n_corrections / elapsed
            if rate > max_correction_rate:
                rate_capped = True

    # If the correction budget was exhausted before reaching the end of the
    # video, keep propagating (uncorrected) so every annotated frame still
    # gets a prediction written -- the evaluator requires a mask for every
    # GT-annotated frame in a predicted video.
    if current_start < len(frame_names):
        for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=current_start
        ):
            if frame_idx in wanted_frames:
                mask = mask_logits[0, 0] > 0
                arr = mask.detach().cpu().numpy().astype(np.uint8) * 255
                Image.fromarray(arr).save(obj_out_dir / f"{frame_names[frame_idx]}.png")
                frames_written += 1

    return frames_written, n_corrections, n_position_triggers, n_skipped_vanish, n_rejected_corrections, n_skipped_early_trigger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--analysis-csv", type=Path, default=Path("results/sav_val_failure_mode_analysis.csv"))
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--ema-alpha", type=float, default=0.2,
                         help="Tuned by coordinate-descent sweep on the n=30 drift subset (2026-09-01/02), "
                         "with jump_ratio/low_ratio/high_ratio held at their own tuned values: peak at 0.2 "
                         "(J&F=42.3), neighbors 0.15->41.5 and 0.25->39.6 both lower.")
    parser.add_argument("--low-ratio", type=float, default=0.35,
                         help="Tuned: plateau 0.35-0.45 (J&F~40), peak at 0.35; F1's original 0.3 gave 39.3 "
                         "here (with jump_ratio=8.0), 0.2 and below collapse to ~34.")
    parser.add_argument("--high-ratio", type=float, default=2.5,
                         help="Tuned: plateau 2.0-2.5 (J&F=41.4 both), declining on both sides; F1's original "
                         "3.0 gave 40.5 here (with low_ratio=0.35, jump_ratio=8.0).")
    parser.add_argument("--min-area-eps", type=float, default=50.0,
                         help="Verified optimal by sweep on the n=30 drift subset (2026-09-03) at the tuned "
                         "jump_ratio/low_ratio/high_ratio/ema_alpha: isolated peak at 50 (J&F=42.3), both "
                         "neighbors lower (40->40.4, 75->40.9). No change from F1's original default.")
    parser.add_argument("--jump-ratio", type=float, default=8.0,
                         help="Flag position drift when centroid jump exceeds jump_ratio * sqrt(EMA area). "
                         "Tuned by sweep on the n=30 drift subset (2026-08-31/09-01): plateau at 8.0-9.0 "
                         "(J&F=39.3), declining on both sides (1.0-5.0: 33.8-37.3, 12.0: 38.4, "
                         "trigger-effectively-off at 1000: 37.6). See sam2-drift-recovery-results.md SS10 P5.")
    parser.add_argument("--max-corrections", type=int, default=60)
    parser.add_argument(
        "--vanish-retry-limit", type=int, default=10,
        help="Keep retrying a fresh box-reprompt for up to this many CONSECUTIVE frames the object reads as "
        "invisible (area==0); once exceeded, stop retrying and let ema_area decay naturally instead. Swept "
        "2026-09-04/05 on val: 0 (blanket skip) is net-negative (J&F 42.3->31.4); 1/2/5 all underperform "
        "unlimited (38.4/39.0/37.5); 10 gives a small genuine gain (42.6, the new default); 20 and unlimited "
        "both give 42.3. On held-out test, vl=10 is neutral (28.9, unchanged from unlimited) -- it does NOT "
        "fix the two catastrophic test regressions (sav_004755/000, sav_017171/000), because both resulted "
        "from a single bad correction attempt succeeding-but-wrong, not a prolonged failed-retry streak; a "
        "retry-count limit structurally cannot catch a first-attempt failure. A correction-plausibility check "
        "(e.g. reject a correction whose resulting mask area is wildly larger than the recent size history) "
        "would target that failure mode instead -- not implemented.",
    )
    parser.add_argument(
        "--require-both-triggers", action="store_true",
        help="A1 (sam2-drift-recovery-next-directions.md): require BOTH the area-ratio and position-jump "
        "signals to agree (excluding a genuine vanish, which is unambiguous alone) before flagging drift, "
        "instead of firing on either alone. Targets the capstone finding that the OR-gate has too high a "
        "false-positive rate on objects without real propagation drift (178/293 regressed vs 44 improved when "
        "applied to the full unselected sav_val). Closed as insufficient on its own (2026-09-07): full-val "
        "J&F=69.0, only +0.3pp over the OR-gate capstone's 68.7 -- area-ratio and position-jump both react to "
        "the same legitimate events (a fast zoom changes both size and centroid), so AND doesn't meaningfully "
        "raise specificity. Kept as an opt-in flag.",
    )
    parser.add_argument(
        "--plausibility-ratio", type=float, default=0.0,
        help="B1 (sam2-drift-recovery-next-directions.md SS3-B): after a box-reprompt correction, reject the "
        "result and fall back to a conservative 'hold position' box (last known centroid/size, no "
        "extrapolation) if the corrected mask's area is more than this many times larger OR smaller than the "
        "pre-drift EMA area. Targets corrections that SUCCEED (SAM2 confidently returns a mask) but on the "
        "WRONG thing -- the mechanism behind both held-out-test catastrophes (sav_004755, sav_017171) and many "
        "capstone regressions, which vanish_retry_limit could not touch since those failed on the first "
        "attempt, not after repeated retries. 0 (default) disables the check entirely, preserving prior "
        "behavior exactly.",
    )
    parser.add_argument(
        "--max-correction-rate", type=float, default=0.0,
        help="A2 (sam2-drift-recovery-next-directions.md): once an object has accumulated >=3 corrections and "
        "its correction rate (corrections / frames elapsed since the object's first frame) exceeds this "
        "value, stop attempting further corrections for the rest of the video (falls through to plain "
        "uncorrected propagation, same as exhausting max_corrections). Retroactively validated on the OR-gate "
        "capstone's full-val per-object data: low-rate objects (<0.05) net improved (+3.3pp mean), high-rate "
        "objects (>0.3) net regressed severely (-11.9pp mean) -- a persistently high rate means the trigger is "
        "chronically misfiring on a naturally volatile object, not catching a one-off drift event. 0 (default) "
        "disables the check entirely, preserving prior behavior exactly.",
    )
    parser.add_argument(
        "--correction-window", type=int, default=0,
        help="D3 (sam2-drift-recovery-next-directions.md SS7): when >0, --max-correction-rate is computed over "
        "only the last N frames (a sliding window) instead of cumulatively since the object's first frame -- "
        "reacts to a recent burst of corrections instead of averaging it away over a long prior quiet stretch. "
        "0 (default) preserves the exact cumulative-rate behavior.",
    )
    parser.add_argument(
        "--early-trigger-skip-frac", type=float, default=0.0,
        help="E1 timing variant (sam2-drift-recovery-next-directions.md SS9): if the object's FIRST-EVER "
        "trigger fires before this fraction of its own total frame count has elapsed, permanently stop "
        "correcting this object for the rest of the video (evaluated once, using only the first trigger seen; "
        "if it fires later than this fraction, correction proceeds normally for the whole video). "
        "Retroactively projected on the OR-gate capstone's full-val per-object data (dry-run, uncorrected pass "
        "-- sav_dryrun_risk_eval.py): objects whose first trigger lands in the first 0.05 fraction of their "
        "frames average -10pp vs the full triggered population's -3.8pp, fading smoothly as the fraction grows. "
        "0 (default) disables the check entirely, preserving prior behavior exactly.",
    )
    parser.add_argument(
        "--rate-warmup-corrections", type=int, default=3,
        help="D2 (sam2-drift-recovery-next-directions.md SS6): number of corrections an object must accumulate "
        "before --max-correction-rate's circuit breaker can first evaluate it (was hardcoded to 3). Lower means "
        "less video where a thrashing object corrects 'for free' before the breaker can act, at the cost of a "
        "noisier rate estimate on fewer samples the first time it checks. 3 (default) preserves prior behavior.",
    )
    parser.add_argument(
        "--soft-nudge-weight", type=float, default=0.0,
        help="F1 (sam2-drift-recovery-next-directions.md SS6/SS12): blend the reprompt box's center between the "
        "velocity-extrapolated predicted_centroid and the model's own raw (uncorrected) centroid at the drift "
        "frame, instead of a full hard override onto predicted_centroid alone. weight=1.0 would ignore the "
        "extrapolation entirely (no correction); weight=0.0 (default) disables the blend, preserving prior "
        "behavior exactly. Caps how far a single false-positive-triggered correction can move the anchor, at "
        "the cost of a slower full snap-back on a genuine drift. Falls back to the pure extrapolated position "
        "when the object is vanished at the drift frame (nothing to blend toward).",
    )
    parser.add_argument("--prediction-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    targets = load_drift_failure_targets(args.analysis_csv)
    total_objects = sum(len(v) for v in targets.values())
    print(f"Loaded {total_objects} drift-failure objects across {len(targets)} videos")

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")

    for i, (video_name, object_ids) in enumerate(sorted(targets.items())):
        video_dir = args.sav_root / "JPEGImages_24fps" / video_name
        ann_dir = args.sav_root / "Annotations_6fps" / video_name
        frame_names = list_frame_names(video_dir)
        object_inputs = collect_object_inputs(ann_dir, frame_names)

        t0 = time.perf_counter()
        state = predictor.init_state(str(video_dir))
        for object_id in sorted(object_ids):
            if object_id not in object_inputs:
                continue
            obj_out_dir = args.prediction_root / video_name / f"{object_id:03d}"
            frames_written, n_corrections, n_position_triggers, n_skipped_vanish, n_rejected_corrections, n_skipped_early_trigger = evaluate_object_with_motion_reprompt(
                predictor, state, object_id, object_inputs[object_id],
                frame_names, ann_dir, args.prompt_policy, obj_out_dir,
                args.ema_alpha, args.low_ratio, args.high_ratio,
                args.min_area_eps, args.jump_ratio, args.max_corrections,
                args.vanish_retry_limit, args.require_both_triggers,
                args.plausibility_ratio, args.max_correction_rate,
                args.correction_window, args.early_trigger_skip_frac,
                args.rate_warmup_corrections, args.soft_nudge_weight,
            )
            print(
                f"[{i + 1}/{len(targets)}] video={video_name} obj={object_id:03d} "
                f"frames_written={frames_written} corrections={n_corrections} "
                f"position_triggers={n_position_triggers} skipped_vanish={n_skipped_vanish} "
                f"rejected_corrections={n_rejected_corrections} skipped_early_trigger={n_skipped_early_trigger}",
                flush=True,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  video total latency_ms={elapsed_ms:.1f}", flush=True)

    print(f"Done: wrote predictions for {total_objects} objects to {args.prediction_root}")


if __name__ == "__main__":
    main()
