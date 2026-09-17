# Dynamic memory-gating, Phase A (DAM4SAM-inspired skip-write) — Implementation Plan

> **For agentic workers:** Execute task-by-task, in order. Tasks 1-3 and 5
> are committed regardless of outcome. Task 4 (the expensive full-val run)
> is **conditional on Task 3's pilot gate passing** — do not start it
> otherwise. Task 6 (scoping a fresh Phase B design cycle) is **conditional
> on Task 4 itself being genuinely positive**, not merely "least bad." This
> mirrors the discipline already used twice in this project's history: cheap
> pilot/correlation before any expensive full-dataset commitment
> (`sam2-drift-recovery-next-directions.md` §3 D3, §8-9 E1,
> `sam2-multimask-divergence-plan.md` Task 4).

**Goal:** Test whether refusing to write suspected-distractor frames into
SAM2's memory bank — the "forgetting" half of DAM4SAM's dual-memory
mechanism, isolated from its "recall" half (DRM anchor injection, out of
scope here — see `sam2-memory-gating-design.md` §4 Phase B) — changes
propagation robustness on SA-V, when applied **alone, with no P5
reprompt-correction at all**, against the plain `box_centroid` baseline
(val 72.1, n=30 known-drift-subset 26.1).

**Architecture:** SAM2's memory-attention window is a fixed recency
schedule (`_prepare_memory_conditioned_features`,
`sam2/sam2/modeling/sam2_base.py:497-650`) that silently tolerates a
missing slot (`if prev is None: continue`, line 571-572) — this already
happens near the start of every video, so the model is not being pushed
outside a behavior it exercises natively. We exploit this: monkey-patch
`predictor.propagate_in_video` (same non-invasive, instance-bound
technique as `sam2_multimask_capture.py`'s `track_step`/`reset_state`
patches) to skip the `obj_output_dict[storage_key][frame_idx] = current_out`
write for a non-conditioning frame when that frame's already-available
multi-mask divergence signal (reusing `sam2_multimask_capture.py` and
`sav_dryrun_multimask_risk_eval.py`'s `bbox_union_area_ratio`, both already
built and reviewed) exceeds a threshold. No model weights are touched; no
new token type or positional encoding is introduced — that is specifically
what distinguishes this from the out-of-scope Phase B.

**Tech Stack:** Python, PyTorch, SAM2.1 Hiera-Tiny (frozen), official
`sam2/sav_dataset/sav_evaluator.py`. No new dependencies.

**Spec:** `sam2-memory-gating-design.md` (the design note this plan
executes — read it first for the DAM4SAM background and why Phase A/Phase B
are split this way). Prior art this plan reuses directly:
`scripts/sam2_multimask_capture.py` (Task 1 of
`sam2-multimask-divergence-plan.md`), `scripts/sav_dryrun_multimask_risk_eval.py`
(its Task 2), `scripts/sav_reprompt_eval.py`'s `load_drift_failure_targets`
(the n=30 known-drift-subset loader used by every prior pilot in this
project), `scripts/sav_motion_prior_reprompt_eval.py` /
`scripts/sav_full_pipeline_eval.py` (the established n=30-script /
full-val-script pairing convention this plan's Tasks 2 and 4 follow).

## Global Constraints

- **No fine-tuning, no GT at inference.** The gate decision uses only
  SAM2's own already-computed multi-mask outputs, exactly as in the closed
  multi-mask-divergence check.
- **Official evaluator only** for every J&F number.
- **Isolate this mechanism's own effect.** Apply memory-gating to plain
  `box_centroid` propagation, with **no P5 reprompt-correction**. DAM4SAM's
  own reported gains come from memory management alone, with no
  reprompting step — conflating this with P5 would make it impossible to
  attribute any observed effect to memory-gating specifically.
- **Don't touch the vendored `sam2/` clone.** All instrumentation lives in
  `scripts/`, applied via monkey-patching, exactly like
  `sam2_multimask_capture.py`.
- **Reuse, don't reimplement, the multi-mask capture.** `sam2_memory_gate.py`
  imports and composes `sam2_multimask_capture.install`/`MultimaskCapture`
  and `sav_dryrun_multimask_risk_eval.bbox_union_area_ratio` rather than
  duplicating either — both are already built and independently reviewed.
- **Pre-committed decision gates (do not adjust after seeing data):**
  - *Pilot gate (Task 3 → Task 4):* the full-val run proceeds only if the
    best swept threshold's n=30 J&F is **≥ 26.1 + 3.0pp = 29.1**, AND no
    single object in that pilot regresses by more than **15pp** versus its
    own `box_centroid`-only score (a smaller, explicitly named bar than
    P5's original +15.5pp pilot signal, and an asymmetric-risk cap in the
    same spirit as the language used to approve P5 itself — "no asymmetric
    risk versus baseline"). Otherwise: closed at the pilot stage, same as
    D3 and all three E1 variants.
  - *Positive-outcome bar (Task 4 → Task 6):* Task 6 (scoping Phase B) is
    justified only if the full-val run is **genuinely positive** —
    val J&F **≥ 72.1 + 1.0pp = 73.1** — not merely "closest to baseline."
    D4's own "best of the series" result (71.3, still −0.8pp) does **not**
    clear this bar; Phase B's added engineering risk (an untrained
    positional-encoding token, per the design note) needs a real positive
    signal to justify it, not a smaller negative one.

---

## File Structure

- **Create:** `scripts/sam2_memory_gate.py` — the memory-gating monkey-patch
  module. One responsibility: given an installed `MultimaskCapture`, wrap
  `predictor.propagate_in_video` so it skips writing a frame's output into
  `non_cond_frame_outputs` when that frame's primary-vs-alternative mask
  divergence exceeds a threshold. Exposes a `MemoryGate` stats object so
  callers can see how often the gate actually fired.
- **Create:** `scripts/sav_memory_gate_eval.py` — n=30 pilot evaluator,
  structurally the sibling of `scripts/sav_motion_prior_reprompt_eval.py`
  (same `load_drift_failure_targets`-based filtering, same per-object
  prediction-PNG-writing loop as `scripts/sav_reprompt_eval.py`), but
  applying `install_memory_gate` instead of any reprompt-correction logic.
- **Create:** `scripts/sav_memory_gate_full_eval.py` — full-val evaluator,
  sibling of `scripts/sav_full_pipeline_eval.py`: imports
  `evaluate_object_with_memory_gate` from the n=30 script and supplies an
  all-objects (no drift-subset filter) `main()`, exactly mirroring how
  `sav_full_pipeline_eval.py` imports from
  `sav_motion_prior_reprompt_eval.py`.
- **Modify:** `sam2-evidence-matrix.md`, `sam2-drift-recovery-next-directions.md`
  — document the outcome (Task 5), following the established row/section
  format.

## Interfaces

- `sam2_memory_gate.per_frame_divergence(feats: dict, pred_masks: torch.Tensor) -> float | None`:
  given one `capture.pop(obj_idx, frame_idx)` result and the frame's
  `pred_masks` tensor (for target interpolation size), returns the same
  `1 - bbox_union_area_ratio(...)` divergence used by the closed
  multi-mask-divergence check, or `None` if fewer than 2 candidate masks
  were available.
- `sam2_memory_gate.MemoryGate`: dataclass with `threshold: float`,
  `n_checked: int`, `n_skipped: int`, and `reset_counts() -> None`.
- `sam2_memory_gate.install_memory_gate(predictor, capture: MultimaskCapture, divergence_threshold: float) -> MemoryGate`:
  installs the patch, returns the stats object. Consumed by both eval
  scripts.
- `sav_memory_gate_eval.evaluate_object_with_memory_gate(predictor, state, object_id, per_frame, frame_names, ann_dir, prompt_policy, obj_out_dir) -> tuple[int, int, int]`:
  returns `(frames_written, n_checked, n_skipped)` for one object — same
  return-tuple-of-counts convention as
  `evaluate_object_with_motion_reprompt`. Consumed by
  `sav_memory_gate_full_eval.py`'s `main()`.

---

### Task 1: Memory-gating monkey-patch module

**Files:**
- Create: `scripts/sam2_memory_gate.py`
- Test: manual sanity check (Step 3) — this project has no pytest suite
  for GPU eval scripts; verification is a real small-scale run with
  printed evidence, same as every prior instrumentation task here.

**Interfaces:**
- Consumes: `sam2_multimask_capture.install`, `MultimaskCapture.pop`
  (already built); `sav_dryrun_multimask_risk_eval.bbox_union_area_ratio`
  (already built).
- Produces: `per_frame_divergence`, `MemoryGate`, `install_memory_gate` (see
  above).

- [ ] **Step 1: Write the module**

```python
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
    per-frame gating decision instead of an aggregate dry-run feature."""
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
    consumes the capture it produces."""
    gate = MemoryGate(threshold=divergence_threshold)
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

    predictor.propagate_in_video = types.MethodType(patched_propagate_in_video, predictor)
    predictor._original_propagate_in_video = original_propagate
    return gate
```

  **Why the gate check happens where it does:** `_run_single_frame_inference`
  internally calls `track_step`, which (once `sam2_multimask_capture.install`
  is active) populates `capture`'s store for `(obj_idx, frame_idx)` as a
  side effect, before `_run_single_frame_inference` returns. So by the time
  `patched_propagate_in_video` reaches `capture.pop(...)`, the data is
  already there — no ordering hazard.

  **Note the ordering dependency vs `sam2_multimask_capture`'s own
  `reset_state` patch:** `MultimaskCapture.clear()` fires on every
  `predictor.reset_state(state)` call, which every `sav_*_eval.py` script
  already calls once per object. This means `gate.n_checked`/`n_skipped`
  accumulate **across the whole run** unless the caller calls
  `gate.reset_counts()` per object — Task 2's evaluator does this
  explicitly (see its Step 1) so per-object skip counts are meaningful in
  its output CSV, not a running total.

- [ ] **Step 2: Verify `install()` ordering requirement doesn't silently
  break** — read `sam2_multimask_capture.py`'s `install()` one more time
  and confirm `install_memory_gate` above only reads from `capture`
  (never re-patches `track_step`/`reset_state` itself), so calling
  `sam2_multimask_capture.install(predictor)` then
  `install_memory_gate(predictor, capture, threshold)` composes cleanly:
  the first patch populates `capture`, the second patch only wraps
  `propagate_in_video` (a third, independent method) and reads from the
  same `capture` object. No two patches touch the same method.

- [ ] **Step 3: Sanity-check on GPU with real data**

```
source .venv/bin/activate
python3 -c "
import torch
from sam2.build_sam import build_sam2_video_predictor
import sam2_multimask_capture, sam2_memory_gate

predictor = build_sam2_video_predictor(
    'configs/sam2.1/sam2.1_hiera_t.yaml',
    'sam2/checkpoints/sam2.1_hiera_tiny.pt', device='cuda',
)
capture = sam2_multimask_capture.install(predictor)
gate = sam2_memory_gate.install_memory_gate(predictor, capture, divergence_threshold=0.3)

state = predictor.init_state('data/sav/sav_val/JPEGImages_24fps/sav_000262')
predictor.add_new_points_or_box(state, frame_idx=0, obj_id=0, box=[100, 100, 200, 200])
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state, start_frame_idx=1, max_frame_num_to_track=20):
    pass
print(f'checked={gate.n_checked} skipped={gate.n_skipped}')
obj_idx = predictor._obj_id_to_idx(state, 0)
non_cond = state['output_dict_per_obj'][obj_idx]['non_cond_frame_outputs']
print(f'frames actually stored in memory: {len(non_cond)} (should be <= 20, and < 20 if gate.n_skipped > 0)')
"
```

  Run this from `scripts/` (so the bare-module imports resolve) — script
  path: `cd scripts && python3 -c "..."` or add `sys.path` as the other
  scripts do. Expected: no exceptions; `gate.n_checked` roughly equals
  frames processed; `len(non_cond) == gate.n_checked - gate.n_skipped`
  (every checked-and-not-skipped frame got stored, every skipped one
  didn't) — verify this arithmetic holds exactly, not just approximately,
  since a mismatch would mean the patch's bookkeeping is wrong even if it
  doesn't crash.

- [ ] **Step 4: Commit**

```bash
git add scripts/sam2_memory_gate.py
git commit -m "Add DAM4SAM-inspired memory-gating monkey-patch (Phase A)"
```

---

### Task 2: n=30 pilot evaluator

**Files:**
- Create: `scripts/sav_memory_gate_eval.py`

**Interfaces:**
- Consumes: `sam2_memory_gate.install_memory_gate`, `MemoryGate` (Task 1);
  `sav_reprompt_eval.load_drift_failure_targets` (already exists).
- Produces: `evaluate_object_with_memory_gate(...) -> (frames_written, n_checked, n_skipped)`
  (consumed by Task 4's full-val script).

- [ ] **Step 1: Write the script**

```python
"""Dynamic memory-gating, Phase A -- n=30 pilot (sam2-memory-gating-plan.md
Task 2). Applies install_memory_gate to PLAIN box_centroid propagation --
no P5 reprompt-correction at all -- on the same n=30 known-drift subset
used by every prior pilot in this project (sav_reprompt_eval.py's
load_drift_failure_targets), to isolate memory-gating's own effect before
any full-val commitment (sam2-memory-gating-plan.md's pre-committed pilot
gate: best swept threshold must clear J&F >= 29.1 with no single-object
regression worse than -15pp, or this direction closes here, same as D3 and
all three E1 variants before it).
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
from sav_sam2_eval import annotated_frame_indices, collect_object_inputs, list_frame_names  # noqa: E402
from sam2_multimask_capture import install  # noqa: E402
from sam2_memory_gate import install_memory_gate  # noqa: E402

from sam2.build_sam import build_sam2_video_predictor


def evaluate_object_with_memory_gate(
    predictor, state, gate, object_id: int, per_frame, frame_names,
    ann_dir: Path, prompt_policy: str, obj_out_dir: Path,
) -> tuple[int, int, int]:
    sorted_frames = sorted(per_frame)
    first_frame_idx = sorted_frames[0]
    predictor.reset_state(state)
    gate.reset_counts()
    obj_idx = predictor._obj_id_to_idx(state, object_id)
    assert obj_idx == 0, f"expected single-object propagation, got obj_idx == {obj_idx}"

    prompt = prompt_for_object(per_frame[first_frame_idx], prompt_policy)
    _, _, mask_logits = predictor.add_new_points_or_box(
        state, frame_idx=first_frame_idx, obj_id=object_id, **prompt
    )
    last_mask = mask_logits[0, 0] > 0

    wanted_frames = annotated_frame_indices(ann_dir, object_id, frame_names)
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    if first_frame_idx in wanted_frames:
        arr = last_mask.detach().cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(arr).save(obj_out_dir / f"{frame_names[first_frame_idx]}.png")
        frames_written += 1

    for frame_idx, _out_obj_ids, mask_logits in predictor.propagate_in_video(
        state, start_frame_idx=first_frame_idx + 1
    ):
        if frame_idx in wanted_frames:
            arr = (mask_logits[0, 0] > 0).detach().cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(arr).save(obj_out_dir / f"{frame_names[frame_idx]}.png")
            frames_written += 1

    return frames_written, gate.n_checked, gate.n_skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sav-root", type=Path, default=Path("data/sav/sav_val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--analysis-csv", type=Path, default=Path("results/sav_val_failure_mode_analysis.csv"))
    parser.add_argument("--prompt-policy", default="box_centroid")
    parser.add_argument("--divergence-threshold", type=float, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    targets = load_drift_failure_targets(args.analysis_csv)
    total_objects = sum(len(v) for v in targets.values())
    print(f"Loaded {total_objects} drift-failure objects across {len(targets)} videos")

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")
    capture = install(predictor)
    gate = install_memory_gate(predictor, capture, args.divergence_threshold)

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
            frames_written, n_checked, n_skipped = evaluate_object_with_memory_gate(
                predictor, state, gate, object_id, object_inputs[object_id],
                frame_names, ann_dir, args.prompt_policy, obj_out_dir,
            )
            print(
                f"[{i + 1}/{len(targets)}] video={video_name} obj={object_id:03d} "
                f"frames_written={frames_written} n_checked={n_checked} n_skipped={n_skipped}",
                flush=True,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  video total latency_ms={elapsed_ms:.1f}", flush=True)

    print(f"Done: wrote predictions to {args.prediction_root}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the threshold sweep on the n=30 subset**

  Run four times, one per threshold, each to its own prediction root:
  ```
  for T in 0.2 0.3 0.4 0.5; do
    python scripts/sav_memory_gate_eval.py \
      --divergence-threshold $T \
      --prediction-root results/sav_drift_subset_memgate_${T}_pred
  done
  ```
  Then score each with the official evaluator against the existing n=30
  ground truth (`results/sav_drift_subset_gt`, already present in this
  repo from the original P5/F1 pilots):
  ```
  for T in 0.2 0.3 0.4 0.5; do
    python sam2/sav_dataset/sav_evaluator.py \
      --gt_root results/sav_drift_subset_gt \
      --pred_root results/sav_drift_subset_memgate_${T}_pred
  done
  ```
  Expected: each run completes without exceptions; each evaluator
  invocation prints a global J&F plus a per-object CSV. Record all four
  J&F numbers and each run's per-object worst regression versus the
  n=30 baseline (26.1) — needed for Task 3's gate.

- [ ] **Step 3: Commit**

```bash
git add scripts/sav_memory_gate_eval.py results/sav_drift_subset_memgate_*_pred
git commit -m "Add n=30 pilot evaluator for DAM4SAM-inspired memory-gating; run threshold sweep"
```

  (Check `.gitignore` first — only `results/**/*.csv` is tracked;
  prediction PNGs under `results/sav_drift_subset_memgate_*_pred/` will
  not be added even if you `git add` the directory. That's correct and
  matches every prior pilot's own results directory in this repo — only
  the evaluator's own summary CSV, if it writes one alongside the PNGs,
  gets tracked.)

---

### Task 3: Apply the pilot gate

**Files:** none new — a decision recorded from Task 2's numbers.

- [ ] **Step 1: Identify the best threshold and check both conditions**

  From Task 2 Step 2's four evaluator runs: find the threshold with the
  highest global J&F. Check:
  - Does it reach **≥ 29.1** (the pre-committed 26.1 + 3.0pp bar)?
  - Does its per-object CSV show **no object regressing more than 15pp**
    below that same object's plain-baseline score? The plain per-object
    baseline is already retained at
    `results/sav_drift_subset_baseline_pred/results.csv` (confirmed
    present: global `J&F=26.1`, matching
    `sam2-drift-recovery-results.md`'s "Baseline (`box_centroid`, один
    prompt)" row exactly) — join on `sequence`/`obj` (same
    whitespace-padded format as every other `results.csv` in this
    project; strip before comparing) against the winning threshold's
    output CSV.

- [ ] **Step 2: Record the ruling**

  - **Both conditions met:** gate **passes** → proceed to Task 4.
  - **Either condition fails:** gate **fails** → skip Task 4, go directly
    to Task 5's negative-result writeup, closed at the pilot stage (same
    disposition as D3 and all three E1 variants).

---

### Task 4 (CONDITIONAL on Task 3's gate passing): full-val run

**Files:**
- Create: `scripts/sav_memory_gate_full_eval.py`

**Interfaces:**
- Consumes: `evaluate_object_with_memory_gate` (Task 2).

- [ ] **Step 1: Write the full-val script**, mirroring exactly how
  `scripts/sav_full_pipeline_eval.py` wraps
  `sav_motion_prior_reprompt_eval.evaluate_object_with_motion_reprompt`:
  import `evaluate_object_with_memory_gate` from
  `sav_memory_gate_eval.py`, and supply a `main()` that iterates **every**
  object in `--sav-root` (via `collect_object_inputs`, no
  `load_drift_failure_targets` filter), using the pilot-winning
  `--divergence-threshold` from Task 3 as the (now non-optional, but still
  a flag) parameter.

- [ ] **Step 2: Run on full `sav_val`**

  ```
  python scripts/sav_memory_gate_full_eval.py \
    --divergence-threshold <PILOT_WINNING_THRESHOLD> \
    --prediction-root results/sav_val_memgate_pred
  python sam2/sav_dataset/sav_evaluator.py \
    --gt_root data/sav/sav_val/Annotations_6fps \
    --pred_root results/sav_val_memgate_pred
  ```
  Expected: completes on all 293 objects (~5h wall-clock, per this
  project's established cost profile for a full-val propagation pass);
  produces a global J&F and per-object CSV.

- [ ] **Step 3: Commit**

```bash
git add scripts/sav_memory_gate_full_eval.py
git commit -m "Add full-val memory-gating evaluator; run at pilot-winning threshold"
```

---

### Task 5: Document the outcome (any branch)

**Files:**
- Modify: `sam2-evidence-matrix.md`
- Modify: `sam2-drift-recovery-next-directions.md`

- [ ] **Step 1: Add a row to `sam2-evidence-matrix.md`** in the
  established format, citing the actual numbers obtained (pilot-only if
  Task 3's gate failed; pilot + full-val if Task 4 ran).

- [ ] **Step 2: Append the next numbered section to
  `sam2-drift-recovery-next-directions.md`**, matching §13's structure
  (Реалізація / Результат / Висновок / Стан серії), explicitly stating
  whether Task 6 was triggered and why (or why not).

- [ ] **Step 3: Commit**

```bash
git add sam2-evidence-matrix.md sam2-drift-recovery-next-directions.md
git commit -m "Document memory-gating Phase A outcome"
```

---

### Task 6 (CONDITIONAL on Task 4 clearing the positive-outcome bar, val J&F ≥ 73.1): scope Phase B

Do not start this task otherwise — a merely-less-negative Phase A result
does not justify Phase B's added technical risk (see Global Constraints).

- [ ] **Step 1: Do not implement Phase B here.** Per
  `sam2-memory-gating-design.md` §4, Phase B's positional-encoding
  handling for an out-of-schedule anchor token is an open design question
  requiring its own stated hypothesis, not a "try it and see." Produce a
  new, separate design note (`sam2-memory-gating-phase-b-design.md`)
  proposing one specific positional-encoding mitigation with a stated
  mechanistic reason to expect it to work, informed by whatever Task 4
  actually found (e.g., which threshold, how often the gate fired, what
  kinds of objects benefited) — then stop, for the same kind of
  confirmation this plan itself required before being written.

## Self-Review

**Spec coverage:** Design note's Phase A definition (§4) → Tasks 1-2.
Design note's own recommended sequencing (cheap pilot before full
commitment, §5) → Task 3's gate. Design note's explicit non-goal (Phase B
implementation) → Task 6 stops short of writing any Phase B code, only
scopes a follow-on design note, matching the design note's own final
paragraph ("say the word for a full execution plan... once you've decided
scope").

**Placeholder scan:** `<PILOT_WINNING_THRESHOLD>` in Task 4 Step 2 is
explicitly a value to be filled from Task 3's actual result, not a
placeholder left unresolved — flagged the same way Task 5 Step 1 of
`sam2-multimask-divergence-plan.md` flagged its own fill-in-from-output
value.

**Type/name consistency:** `install_memory_gate(predictor, capture, divergence_threshold) -> MemoryGate`
(Task 1) is called identically in Task 2's `main()` and (by inheritance of
the same function) Task 4's script. `evaluate_object_with_memory_gate`'s
return tuple `(frames_written, n_checked, n_skipped)` (Task 2) matches
what Task 4's full-val `main()` is expected to unpack the same way
`sav_full_pipeline_eval.py` unpacks
`evaluate_object_with_motion_reprompt`'s own multi-value return.
