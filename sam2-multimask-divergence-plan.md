# Multi-mask divergence (DAM4SAM-style) — Implementation Plan

> **For agentic workers:** Execute task-by-task, in order. Tasks 1-5 are
> committed regardless of outcome (they produce a decision, positive or
> negative). Task 6 is **conditional** — only start it if Task 5's gate
> passes. This mirrors this project's own established discipline (see
> `sam2-drift-recovery-next-directions.md` §3 D3, §8-9 E1): cheap
> retrospective/pilot check before any expensive full-dataset GPU run.

**Goal:** Determine whether a per-frame signal derived from SAM2's own
discarded alternative candidate masks (the multi-hypothesis divergence used
by DAM4SAM, arxiv 2411.17576) predicts which SA-V objects benefit from P5
motion-prior correction — a distinction none of E1's six trajectory-based
features found (combined `R²=0.026`, see `sam2-drift-recovery-next-
directions.md` §8-9,11).

**Architecture:** SAM2's video predictor computes 3 candidate masks + their
IoU estimates every tracking step (`multimask_output_for_tracking: true` is
already active in `configs/sam2.1/sam2.1_hiera_t.yaml`), but
`track_step` in `sam2/sam2/modeling/sam2_base.py` (see lines 814-864)
keeps only the best one and discards `low_res_multimasks`/`ious` before
they reach `current_out` — unlike `object_score_logits`, which the same
function explicitly re-attaches (line ~857). We instrument this
non-invasively from an evaluation script (monkey-patching `track_step` at
call time, no edits inside the vendored `sam2/` clone — it is a
dependency, not vendored, per `README.md` "Setup"), extract a per-frame
`area_ratio(primary_bbox, union(primary_bbox, best_alternative_bbox))`
feature (DAM4SAM's `θ_anc`, disproved-in-that-form iff we merely replicate
it; the actual test is whether it correlates where E1's features did not),
and run the exact same retrospective-correlation-before-live-run check
used for E1, against the same already-computed capstone-vs-P2-only labels.

**Tech Stack:** Python, PyTorch, SAM2.1 Hiera-Tiny (frozen, no fine-tuning
anywhere — `sam2-drift-recovery-charter.md` §1), official
`sam2/sav_dataset/sav_evaluator.py`, no new dependencies.

**Spec:** This project's own research narrative —
`sam2-drift-recovery-next-directions.md` (§8-9, 11 for E1's methodology and
why it failed), `sam2-drift-recovery-charter.md` §1 (project goal:
no-oracle, frozen model), `sam2-evidence-matrix.md` (hypothesis/evidence
table format used for the final writeup). No external spec — this is a
research-direction scouting plan I (Claude) produced from a literature
search on 2026-09-15, see conversation history / `sam2-evidence-matrix.md`
for where the new row belongs.

## Global Constraints

- **No fine-tuning, no GT at inference.** Every feature/gate must be
  computable from a single frozen-model forward pass, matching every prior
  experiment in the series.
- **Official evaluator only.** Any J&F number must come from
  `sam2/sav_dataset/sav_evaluator.py`, never a custom metric.
- **Cheap check before expensive run.** Never launch a full-val (~293
  objects, hours) or full-test correction run until a retrospective
  correlation check (reusing already-computed labels, zero new GPU
  correction calls) has passed a stated numeric bar. This is the exact
  discipline that closed E1 and D3 without wasted GPU time — do not skip
  it here.
- **Don't touch the vendored `sam2/` clone.** All instrumentation lives in
  `scripts/`, applied via monkey-patching at import time, so `git status`
  inside `sam2/` (a separate, gitignored clone) never shows a diff.
- **Follow the existing `sav_dryrun_risk_eval.py` pattern**: one
  uncorrected `propagate_in_video` pass per object, feature columns keyed
  by `(video, object_id)`, CSV output, resumable/streaming write (the
  existing script rewrites the CSV after every object so a killed run
  loses at most one object's work — replicate this).

---

## File Structure

- **Create:** `scripts/sam2_multimask_capture.py` — the monkey-patch
  helper. One responsibility: wrap `SAM2Base.track_step` so the discarded
  `(low_res_multimasks, ious)` from `_track_step`'s `sam_outputs` become
  readable from the caller via a side-channel dict, keyed by
  `(obj_idx, frame_idx)`. No SAM2-specific tracking logic lives here —
  just capture-and-expose.
- **Create:** `scripts/sav_dryrun_multimask_risk_eval.py` — new dry-run
  script, structurally a sibling of `sav_dryrun_risk_eval.py` (same CLI
  args, same one-uncorrected-pass-per-object design), but computing the
  DAM4SAM-style divergence features using `sam2_multimask_capture`
  instead of (or in addition to) the existing area/position features.
  Kept as a separate file rather than extending `sav_dryrun_risk_eval.py`
  in place, because the two scripts have different instrumentation
  dependencies (this one needs the monkey-patch, the original doesn't) —
  matches this project's existing pattern of one script per
  mechanism-variant (`sav_drift_reprompt_eval.py` vs
  `..._objscore_eval.py` vs `..._consensus_eval.py`).
- **Create:** `scripts/sav_risk_feature_correlate.py` — small, reusable
  analysis script: loads a features CSV + two `results.csv` label sources,
  computes `delta = capstone_JF - baseline_JF` per object, reports
  Pearson r per feature and combined linear-regression R². This
  formalizes a check that was so far done ad hoc for E1 (§8, §9, §11 each
  redo it) — worth having as a real, reusable script now that it's the
  third and probably not last use.
- **Modify:** `sam2-evidence-matrix.md` — add one new hypothesis row for
  this direction once Task 5's outcome is known (positive or negative).
- **Modify:** `sam2-drift-recovery-next-directions.md` — append new
  numbered section (§13) documenting the outcome, following the exact
  format of §8-9 (E1).

## Interfaces (so later tasks agree on names)

- `sam2_multimask_capture.install(predictor) -> MultimaskCapture`: patches
  `type(predictor.forward)`-level `track_step` (actually
  `predictor`'s underlying `SAM2Base` — see Task 1 for the exact
  attribute path) and returns a `MultimaskCapture` object.
- `MultimaskCapture.pop(obj_idx: int, frame_idx: int) -> dict`: returns
  `{"ious": list[float], "low_res_multimasks": torch.Tensor[M,H,W]}` for
  that step, or raises `KeyError` if not captured (caller decides
  fallback). Consumed by `sav_dryrun_multimask_risk_eval.py`.
- `evaluate_object_multimask_dry_run(predictor, state, capture, object_id,
  per_frame, prompt_policy, ...) -> dict`: same shape/contract as
  `evaluate_object_dry_run` in `sav_dryrun_risk_eval.py` (returns a flat
  dict of scalar features), so the CSV-writing loop in `main()` is
  copy-adapted, not reinvented.
- `sav_risk_feature_correlate.load_delta_labels(capstone_csv: Path,
  baseline_csv: Path) -> dict[tuple[str,str], float]`: keyed by
  `(video, object_id_str)`, matches the join key already used in
  `sav_dryrun_risk_features_v2.csv` (`video`, `object_id` as zero-padded
  3-digit string).

---

### Task 1: Multi-mask capture instrumentation

**Files:**
- Create: `scripts/sam2_multimask_capture.py`
- Test: manual sanity check (see Step 3) — no pytest harness exists for
  these GPU eval scripts in this project (only
  `davis-evaluation/pytest/test_evaluation.py`, which is a vendored
  dataset-kit test, unrelated); verification here is a printed
  cross-check against the value SAM2 already stores for the *selected*
  mask, which must match exactly.

**Interfaces:**
- Produces: `install(predictor) -> MultimaskCapture` and
  `MultimaskCapture.pop(obj_idx, frame_idx) -> dict` (see above).

- [ ] **Step 1: Locate the exact monkey-patch point**

  `predictor` returned by `build_sam2_video_predictor(...)` is itself a
  `SAM2Base` subclass instance (confirmed:
  `sam2/sam2/modeling/sam2_base.py:814` defines `track_step` directly on
  the class every video predictor inherits). Patch
  `predictor.track_step` (the *bound method on the instance*, via
  `types.MethodType`), not the class — this avoids affecting any other
  predictor instance that might exist in-process.

- [ ] **Step 2: Write the capture wrapper**

```python
"""Captures SAM2's discarded multi-mask candidates (low_res_multimasks,
ious) during propagate_in_video, without modifying the vendored sam2/
clone. See track_step in sam2/sam2/modeling/sam2_base.py:814 -- it keeps
only the best-IoU mask in current_out and drops the other M-1 candidates
plus their IoU estimates before returning. This re-derives them by
wrapping predictor._track_step (the pre-reduction call) instead of
track_step itself, since only _track_step's `sam_outputs` tuple still
carries all M candidates."""

from __future__ import annotations

import types
from dataclasses import dataclass, field


@dataclass
class MultimaskCapture:
    _store: dict[tuple[int, int], dict] = field(default_factory=dict)

    def pop(self, obj_idx: int, frame_idx: int) -> dict:
        return self._store.pop((obj_idx, frame_idx))


def install(predictor) -> MultimaskCapture:
    capture = MultimaskCapture()
    original_track_step = predictor.track_step.__func__

    def patched_track_step(self, *args, **kwargs):
        current_out, sam_outputs, _high_res_masks, _obj_ptr = self._track_step(
            *args, **kwargs
        )
        low_res_multimasks, _high_res_multimasks, ious, low_res_masks, \
            high_res_masks, obj_ptr, object_score_logits = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            current_out["object_score_logits"] = object_score_logits

        frame_idx = args[0]
        # obj_idx is implicit: track_step is called once per object per
        # frame inside propagate_in_video's inner loop -- capture keyed by
        # a monotonic call counter per frame_idx handles that without
        # needing obj_idx threaded through track_step's own signature.
        key_n = sum(1 for k in capture._store if k[1] == frame_idx)
        capture._store[(key_n, frame_idx)] = {
            "ious": ious.detach().float().cpu().tolist()[0],
            "low_res_multimasks": low_res_multimasks.detach().float().cpu(),
        }

        run_mem_encoder = kwargs.get("run_mem_encoder", True)
        if len(args) > 8:
            run_mem_encoder = args[8]
        self._encode_memory_in_output(
            args[2], args[4], args[5], run_mem_encoder,
            high_res_masks, object_score_logits, current_out,
        )
        return current_out

    predictor.track_step = types.MethodType(patched_track_step, predictor)
    predictor._original_track_step = original_track_step  # for uninstall/debug
    return capture
```

  **Why re-derive `track_step` instead of calling the original then
  re-deriving separately:** `_track_step`'s `sam_outputs` tuple is only
  returned from `_track_step`, not from `track_step` — the reduction to
  best-mask-only happens in the 12 lines directly inside `track_step`
  (see `sam2_base.py:834-849`). There's no way to get both the final
  `current_out` *and* the discarded candidates without re-implementing
  those 12 lines. This is why the patch reimplements `track_step`'s body
  instead of wrapping it opaquely — copy it byte-for-byte from
  `sam2_base.py:834-849` (re-check against that exact revision before
  running; SAM2 is a pinned external dependency, but re-verify line
  numbers if `pip install -e .` picked up a different checkout).

- [ ] **Step 3: Sanity-check the capture against SAM2's own stored value**

  Write a throwaway script run once, not committed:

```python
import torch
from sam2.build_sam import build_sam2_video_predictor
from sam2_multimask_capture import install

predictor = build_sam2_video_predictor(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2/checkpoints/sam2.1_hiera_tiny.pt", device="cuda",
)
capture = install(predictor)
state = predictor.init_state("data/sav/sav_val/JPEGImages_24fps/sav_000262")
predictor.add_new_points_or_box(state, frame_idx=0, obj_id=0, box=[100, 100, 200, 200])
for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state, start_frame_idx=1, max_frame_num_to_track=3):
    feats = capture.pop(0, frame_idx)
    best_iou = max(feats["ious"])
    stored_score = predictor._obj_id_to_idx  # sanity anchor only
    print(frame_idx, feats["ious"], "argmax matches best mask:",
          best_iou == feats["ious"][int(torch.tensor(feats["ious"]).argmax())])
```

  Run: `python scripts/sanity_check_capture.py` (adjust box coords to any
  valid annotated object in `sav_000262`; exact coordinates don't matter
  for this check).
  Expected: no `KeyError` from `capture.pop`, `ious` has length 3 (M=3,
  since `multimask_output_for_tracking: true`), and
  `low_res_multimasks.shape[0] == 3`.

- [ ] **Step 4: Commit**

```bash
git add scripts/sam2_multimask_capture.py
git commit -m "Add non-invasive capture of SAM2's discarded multi-mask candidates"
```

---

### Task 2: New dry-run script with divergence features

**Files:**
- Create: `scripts/sav_dryrun_multimask_risk_eval.py`
- Test: `--max-videos 2` run, checked by hand (Step 4) — GPU-bound scripts
  in this project are verified by small-`--max-videos` runs, not unit
  tests (see `sav_dryrun_risk_eval.py`'s own CLI, which has no test file
  either).

**Interfaces:**
- Consumes: `sam2_multimask_capture.install`, `MultimaskCapture.pop` (Task 1).
- Produces: a CSV with columns `video, object_id, n_frames,
  divergence_mean, divergence_max, frac_frames_divergent,
  first_divergence_frac, iou_gap_mean, iou_gap_min` — consumed by Task 4.

- [ ] **Step 1: Implement the per-frame divergence feature**

  DAM4SAM's signal (arxiv 2411.17576): bbox of primary mask vs bbox of
  `union(primary, best-alternative)`; ratio < 0.7 flags a distractor.
  Adapt directly — reuse `mask_centroid`'s sibling bbox helper (none
  exists yet; add one, since only centroid is currently exported from
  `sav_motion_prior_reprompt_eval.py`):

```python
"""E1-followup: multi-mask divergence features from an UNCORRECTED pass.

E1 (sam2-drift-recovery-next-directions.md SS8-9, 11) tested six
trajectory-derived features (area deviation, position jump, trigger
timing) against the capstone-vs-P2-only per-object delta and found no
signal (combined R^2=0.026). All six were properties of the SELECTED
mask's history across frames. This script instead captures a
single-frame introspective signal SAM2 already computes internally on
every step -- the divergence between its primary predicted mask and its
own best alternative candidate (DAM4SAM, arxiv 2411.17576) -- which is
never surfaced by track_step (see scripts/sam2_multimask_capture.py) and
was never part of E1's feature set.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from davis_sam2_eval import prompt_for_object  # noqa: E402
from sav_sam2_eval import annotated_frame_indices, collect_object_inputs, list_frame_names  # noqa: E402
from sam2_multimask_capture import install  # noqa: E402

from sam2.build_sam import build_sam2_video_predictor


def mask_bbox(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_union_area_ratio(primary: torch.Tensor, alternative: torch.Tensor) -> float | None:
    primary_bbox = mask_bbox(primary)
    if primary_bbox is None:
        return None
    px0, py0, px1, py1 = primary_bbox
    primary_area = max(px1 - px0, 1) * max(py1 - py0, 1)
    alt_bbox = mask_bbox(alternative)
    if alt_bbox is None:
        return 1.0
    ax0, ay0, ax1, ay1 = alt_bbox
    ux0, uy0 = min(px0, ax0), min(py0, ay0)
    ux1, uy1 = max(px1, ax1), max(py1, ay1)
    union_area = max(ux1 - ux0, 1) * max(uy1 - uy0, 1)
    return primary_area / union_area


def evaluate_object_multimask_dry_run(
    predictor, state, capture, object_id: int, per_frame,
    prompt_policy: str,
) -> dict:
    sorted_frames = sorted(per_frame)
    first_frame_idx = sorted_frames[0]
    predictor.reset_state(state)
    obj_idx = predictor._obj_id_to_idx(state, object_id)

    prompt = prompt_for_object(per_frame[first_frame_idx], prompt_policy)
    predictor.add_new_points_or_box(
        state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
    )

    n_frames = 0
    divergences: list[float] = []
    iou_gaps: list[float] = []
    first_divergence_rel_frame: int | None = None

    for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
        state, start_frame_idx=first_frame_idx + 1
    ):
        n_frames += 1
        rel_frame = frame_idx - first_frame_idx
        try:
            feats = capture.pop(obj_idx, frame_idx)
        except KeyError:
            continue
        ious = feats["ious"]
        multimasks = feats["low_res_multimasks"][0]  # this object's slice
        sorted_idx = sorted(range(len(ious)), key=lambda i: ious[i], reverse=True)
        best_idx, second_idx = sorted_idx[0], sorted_idx[1]
        primary_mask = torch.nn.functional.interpolate(
            multimasks[best_idx][None, None].float(),
            size=(mask_logits.shape[-2], mask_logits.shape[-1]),
            mode="bilinear", align_corners=False,
        )[0, 0] > 0
        alt_mask = torch.nn.functional.interpolate(
            multimasks[second_idx][None, None].float(),
            size=(mask_logits.shape[-2], mask_logits.shape[-1]),
            mode="bilinear", align_corners=False,
        )[0, 0] > 0
        ratio = bbox_union_area_ratio(primary_mask, alt_mask)
        if ratio is not None:
            divergence = 1.0 - ratio
            divergences.append(divergence)
            iou_gaps.append(ious[best_idx] - ious[second_idx])
            if divergence > 0.3 and first_divergence_rel_frame is None:
                first_divergence_rel_frame = rel_frame

    return {
        "n_frames": n_frames,
        "divergence_mean": statistics.fmean(divergences) if divergences else 0.0,
        "divergence_max": max(divergences) if divergences else 0.0,
        "frac_frames_divergent": (
            sum(1 for d in divergences if d > 0.3) / len(divergences)
        ) if divergences else 0.0,
        "first_divergence_frac": (
            first_divergence_rel_frame / n_frames
        ) if first_divergence_rel_frame is not None and n_frames > 0 else -1.0,
        "iou_gap_mean": statistics.fmean(iou_gaps) if iou_gaps else 0.0,
        "iou_gap_min": min(iou_gaps) if iou_gaps else 0.0,
    }
```

  Note the `obj_idx` key-matching caveat from Task 1 Step 2 (capture keys
  frames by a call-order counter, not real `obj_idx`) — for single-object
  propagation (this project always resets state and propagates one object
  at a time, exactly like `sav_dryrun_risk_eval.py` does), `key_n` is
  always `0`, so `capture.pop(obj_idx, frame_idx)` must be called as
  `capture.pop(0, frame_idx)` regardless of the real SAM2 `obj_idx` — fix
  this by having `evaluate_object_multimask_dry_run` always pop index
  `0`, and assert single-object propagation stays true if this script is
  ever reused for multi-object batches.

- [ ] **Step 2: Fix the obj_idx assumption explicitly**

  Change the `capture.pop(obj_idx, frame_idx)` call in Step 1's code to
  `capture.pop(0, frame_idx)` and add a one-line comment stating why
  (single-object-per-`propagate_in_video`-call, matching every other
  script in this project).

- [ ] **Step 3: Write `main()` by adapting `sav_dryrun_risk_eval.py`'s
  `main()` almost verbatim** — same CLI args
  (`--sav-root`, `--checkpoint`, `--config`, `--video-list`,
  `--max-videos`, `--prompt-policy`, `--output-csv`, `--progress-log`),
  same per-video/per-object loop structure, same streaming CSV rewrite
  after every object. The only differences: call `install(predictor)`
  once after `build_sam2_video_predictor(...)`, call
  `evaluate_object_multimask_dry_run` instead of
  `evaluate_object_dry_run`, and use this script's own `fieldnames` list
  (`video, object_id, n_frames, divergence_mean, divergence_max,
  frac_frames_divergent, first_divergence_frac, iou_gap_mean,
  iou_gap_min`).

- [ ] **Step 4: Small-scale verification run**

  Run:
  ```
  python scripts/sav_dryrun_multimask_risk_eval.py \
    --sav-root data/sav/sav_val --max-videos 2 \
    --output-csv results/sav_val_dryrun_multimask_smoke.csv \
    --progress-log results/sav_val_dryrun_multimask_smoke_progress.log
  ```
  Expected: completes without exceptions, CSV has one row per object in
  the first 2 videos, `n_frames` matches the same objects' `n_frames` in
  `results/sav_val_dryrun_risk_features_v2.csv` exactly (same
  propagation length — cross-check by `video`+`object_id` join), and no
  feature column is uniformly zero (a uniformly-zero column signals the
  capture or interpolation step is broken, not that the signal is
  genuinely absent — that must be diagnosed before proceeding, since a
  silently-broken capture would masquerade as "no signal" in Task 4).

- [ ] **Step 5: Commit**

```bash
git add scripts/sav_dryrun_multimask_risk_eval.py
git commit -m "Add dry-run script for DAM4SAM-style multi-mask divergence features"
```

---

### Task 3: Full val dry-run (no correction, no GT) — the expensive-but-necessary data collection

**Files:**
- None new — runs Task 2's script at full scale.

- [ ] **Step 1: Run the full uncorrected pass on all of `sav_val`**

  Same cost profile as E1's original run (~5 hours, dominated by
  `propagate_in_video`, per `sam2-drift-recovery-next-directions.md` §8):

  ```
  python scripts/sav_dryrun_multimask_risk_eval.py \
    --sav-root data/sav/sav_val \
    --output-csv results/sav_val_dryrun_multimask_features.csv \
    --progress-log results/sav_val_dryrun_multimask_progress.log
  ```

  Expected: 293 rows (one per object, matching
  `sav_val_dryrun_risk_features_v2.csv`'s row count exactly — if it
  doesn't match, some objects failed silently and must be investigated
  before Task 4, not averaged over).

- [ ] **Step 2: Commit the results CSV and log**

```bash
git add results/sav_val_dryrun_multimask_features.csv results/sav_val_dryrun_multimask_progress.log
git commit -m "Run full-val multi-mask divergence dry-run (293 objects)"
```

---

### Task 4: Retrospective correlation check (the actual decision point)

**Files:**
- Create: `scripts/sav_risk_feature_correlate.py`

**Interfaces:**
- Consumes: `results/sav_val_dryrun_multimask_features.csv` (Task 3),
  `results/sav_val_full_pipeline_pred/results.csv` (capstone, already
  exists), `results/sav_val_p2only_pred/results.csv` (baseline, already
  exists).

- [ ] **Step 1: Implement the reusable correlate script**

```python
"""Retrospective feature-vs-outcome correlation check, reusable across
E1-style pre-selection attempts (this is the third time this exact check
is done by hand per sam2-drift-recovery-next-directions.md SS8, SS9, SS11
-- worth having as a real script now)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scipy import stats  # already a transitive dep via SAM2's eval stack
import numpy as np


def load_jf_by_object(csv_path: Path) -> dict[tuple[str, str], float]:
    out = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            seq = row["sequence"].strip()
            if seq in ("", "Global score"):
                continue
            out[(seq, row["obj"].strip())] = float(row["J&F"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--capstone-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--feature-columns", nargs="+", required=True)
    args = parser.parse_args()

    capstone = load_jf_by_object(args.capstone_csv)
    baseline = load_jf_by_object(args.baseline_csv)
    deltas: dict[tuple[str, str], float] = {
        key: capstone[key] - baseline[key]
        for key in capstone if key in baseline
    }

    with args.features_csv.open() as f:
        rows = list(csv.DictReader(f))

    matched_deltas = []
    feature_values: dict[str, list[float]] = {c: [] for c in args.feature_columns}
    for row in rows:
        key = (row["video"], row["object_id"])
        if key not in deltas:
            continue
        matched_deltas.append(deltas[key])
        for col in args.feature_columns:
            feature_values[col].append(float(row[col]))

    print(f"Matched {len(matched_deltas)}/{len(rows)} objects to delta labels")
    y = np.array(matched_deltas)
    X_cols = []
    for col in args.feature_columns:
        x = np.array(feature_values[col])
        r, p = stats.pearsonr(x, y)
        print(f"{col}: pearson r={r:.3f} (p={p:.4f})")
        X_cols.append(x)

    X = np.column_stack(X_cols)
    X_design = np.column_stack([X, np.ones(len(y))])
    coeffs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    print(f"Combined R^2 ({len(args.feature_columns)} features) = {r2:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check against E1's already-known numbers first**

  Run this new script against the *existing* E1 features CSV to confirm
  it reproduces the already-documented result before trusting it on the
  new features:
  ```
  python scripts/sav_risk_feature_correlate.py \
    --features-csv results/sav_val_dryrun_risk_features_v2.csv \
    --capstone-csv results/sav_val_full_pipeline_pred/results.csv \
    --baseline-csv results/sav_val_p2only_pred/results.csv \
    --feature-columns trigger_rate n_triggers area_dev_mean area_dev_max pos_jump_mean pos_jump_max
  ```
  Expected: per-feature Pearson r values matching
  `sam2-drift-recovery-next-directions.md` §8's table
  (`trigger_rate: -0.138`, `n_triggers: -0.080`, `area_dev_mean: -0.058`,
  `area_dev_max: +0.002`, `pos_jump_mean: +0.039`, `pos_jump_max:
  +0.030`) within rounding, and combined `R² ≈ 0.026`. If these don't
  match, the script has a bug (wrong CSV parsing, wrong delta sign, wrong
  join) — fix before using it on the new data, since a buggy correlate
  script is indistinguishable from "no signal" on data you don't already
  know the answer for.

- [ ] **Step 3: Run on the new multi-mask divergence features**

  ```
  python scripts/sav_risk_feature_correlate.py \
    --features-csv results/sav_val_dryrun_multimask_features.csv \
    --capstone-csv results/sav_val_full_pipeline_pred/results.csv \
    --baseline-csv results/sav_val_p2only_pred/results.csv \
    --feature-columns divergence_mean divergence_max frac_frames_divergent first_divergence_frac iou_gap_mean iou_gap_min
  ```

- [ ] **Step 4: Apply the decision gate**

  - **If** any single feature has `|r| ≥ 0.30` **or** combined `R² ≥
    0.10` (roughly 4-10x E1's combined `R²=0.026` — a deliberately high
    bar, since E1's near-zero result showed cheap trajectory features
    carry essentially no signal, and this new feature family should
    clear that floor by a wide margin to justify further GPU spend): the
    gate **passes** → proceed to Task 6.
  - **Otherwise**: the gate **fails** → skip Task 6, go directly to
    Task 5's negative-result writeup.

- [ ] **Step 5: Commit**

```bash
git add scripts/sav_risk_feature_correlate.py
git commit -m "Add reusable feature-vs-outcome correlation script; run on multi-mask divergence features"
```

---

### Task 5: Document the outcome (either branch)

**Files:**
- Modify: `sam2-evidence-matrix.md`
- Modify: `sam2-drift-recovery-next-directions.md`

- [ ] **Step 1: Add a new row to `sam2-evidence-matrix.md`**

  Follow the exact existing row format (hypothesis | setup | result).
  Example for a negative outcome:
  ```
  | SAM2's discarded multi-mask divergence (DAM4SAM-style, primary-vs-alternative bbox area ratio) is a better no-oracle drift-detection signal than area-ratio/position trajectory features | 293 sav_val об'єктів, `scripts/sav_dryrun_multimask_risk_eval.py` + `sav_risk_feature_correlate.py`, delta = capstone(P2+P5 OR-gate) − P2-only | Спростовано: комбінований R²=<value>, найкраща одинична ознака |r|=<value> — не перевищує поріг прийняття (R²≥0.10 або |r|≥0.30), той самий висновок, що й E1 (§8-9) |
  ```
  (Fill in `<value>` from Task 4's actual output — never leave the row
  with placeholder numbers.)

- [ ] **Step 2: Append §13 to `sam2-drift-recovery-next-directions.md`**

  Match the structure of §8 (E1) exactly: **Реалізація** (what was
  built, referencing this plan's Task 1-3), **Результат** (the actual
  correlation numbers from Task 4), **Висновок** (gate passed/failed and
  why), **Стан серії** (update the running "best result" line — unchanged
  at D4 if this closes negatively).

- [ ] **Step 3: Commit**

```bash
git add sam2-evidence-matrix.md sam2-drift-recovery-next-directions.md
git commit -m "Document multi-mask divergence direction outcome (§13)"
```

---

### Task 6 (CONDITIONAL — only if Task 4's gate passed): n=30 pilot gate design

Do not start this task if Task 4 failed the gate. If it passed:

**Files:**
- Create: `scripts/sav_drift_reprompt_divergence_eval.py` (new gate
  variant, sibling of `sav_motion_prior_reprompt_eval.py`)

- [ ] **Step 1: Design a trigger using the validated feature** — reuse
  `sam2_multimask_capture` inside the live-correction loop (not just the
  dry-run), firing a P5-style fresh-box reprompt when
  `frac_frames_divergent` crosses a threshold picked from Task 4's
  bucket analysis (mirror the bucket table style from E1 §8).
- [ ] **Step 2: Run on the same n=30 known-drift subset** used for every
  prior pilot (P5, F1, consensus) — `--all-drift-failures` flag pattern
  already exists across `sav_*_eval.py` scripts, reuse it.
- [ ] **Step 3: Compare against D4 (`71.3`/`72.6`)** — only proceed to a
  full-val run if the n=30 pilot **and** its mechanism story are at least
  as convincing as P5's original n=30 result (`42.3`, `+15.5pp`) was
  before its own full-val run. If the pilot is merely comparable to D4,
  stop — matches this project's own standard for "worth a full run" bar
  used throughout the series.
- [ ] **Step 4: Document + commit**, following the same §-numbering
  convention as Task 5.

---

## Self-Review

**Spec coverage:** Goal (find a feature E1 missed) → Tasks 1-4. Decision
discipline (cheap check before full run) → Task 4's explicit gate.
Documentation convention → Task 5. Contingent follow-through if the
signal is real → Task 6.

**Placeholder scan:** No TBD/TODO; the one deliberately-unfilled `<value>`
in Task 5 Step 1 is explicitly flagged as "fill in from actual output,"
not a placeholder left for someone else to guess.

**Type/name consistency:** `MultimaskCapture.pop(obj_idx, frame_idx)`
(Task 1) is called as `capture.pop(0, frame_idx)` throughout Task 2 per
the single-object-propagation fix in Task 2 Step 2 — consistent.
Feature column names in Task 2's CSV schema match the `--feature-columns`
arguments used in Task 4 Steps 3. `evaluate_object_multimask_dry_run`'s
return dict keys match the CSV `fieldnames` referenced in Task 2 Step 3.
