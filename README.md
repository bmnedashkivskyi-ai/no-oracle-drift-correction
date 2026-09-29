*Language: English · [Українська](README.uk.md)*

# SA-V Drift Recovery: A Negative-Results Study

An empirical study of no-oracle drift-correction strategies for a **frozen** SAM2.1
(Hiera-Tiny) video object segmentation model, evaluated on the official
[SA-V benchmark](https://ai.meta.com/datasets/segment-anything-video/). No model
weights are fine-tuned anywhere in this project — every intervention studied here
is a prompt-time or inference-time policy change.

**Headline finding:** a motion-prior drift-correction mechanism (fresh box
re-prompting from a constant-velocity position extrapolation) recovers large gains
on a ground-truth-preselected subset of genuinely drifting objects (+15.5 J&F on a
known-drift subset), but is *net harmful* when applied to the full, unselected SA-V
val set (72.1 → 68.7 J&F). The bottleneck is not correction precision — it's
**trigger specificity**: the false-positive rate on healthy objects that never
needed correcting. Twelve further mitigation attempts (frame-level gating,
object-level circuit breakers, feature-based pre-selection, and correction
softening) never closed the gap back to the plain uncorrected baseline. The best
result across the whole series (D4) still lands 0.8 (val) and 1.8 (test)
percentage points below simply not correcting at all.

This repository contains the evaluation code, raw per-object results, and the full
experimental narrative behind that finding. See [`paper/`](paper/) for the
manuscript draft and literature review built from this work.

**Paper (English / Українська):** [`paper/paper.pdf`](paper/paper.pdf) ·
[`paper/paper_uk.pdf`](paper/paper_uk.pdf) — full peer-reviewed manuscript, both a
byte-faithful translation of the other. Source: [`paper/draft.md`](paper/draft.md) /
[`paper/draft_uk.md`](paper/draft_uk.md) (Markdown) and
[`paper/paper.tex`](paper/paper.tex) / [`paper/paper_uk.tex`](paper/paper_uk.tex)
(LaTeX, shared [`paper/references.bib`](paper/references.bib)).

## Repository layout

```
scripts/     Evaluation and analysis scripts (this project's own code)
results/     Per-object J&F score tables (CSV) for every experiment variant
paper/       Paper draft + literature review (see paper/research/)
sam2-*.md    Experimental log, methodology, and findings write-ups (chronological)
```

The full per-frame predicted mask PNGs (>1M files, several GB) are not tracked in
this repository — only the aggregate/per-object score CSVs produced by the
official SA-V evaluator. Re-running any script listed below regenerates them
locally.

## Setup

1. **Clone SAM2 separately** (Apache 2.0, Meta AI) — it is a dependency, not
   vendored in this repo:
   ```
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2 && pip install -e .
   # download checkpoints per sam2's own instructions, e.g.:
   # sam2/checkpoints/download_ckpts.sh
   ```
2. **Install this project's own dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Get the SA-V dataset** from Meta AI (registration required; SA-V has its own
   license terms, not covered by this repository's license) and place/symlink it
   so that the scripts' default `--sav-root` (`data/sav/sav_val`) resolves, or
   pass `--sav-root` explicitly.

## Running an evaluation

Every script below scores against the **official** `sav_evaluator.py` (shipped
with the SAM2 repo, `sam2/sav_dataset/sav_evaluator.py`) — none of this project's
own J&F numbers come from a custom metric implementation.

```
python scripts/sav_full_pipeline_eval.py \
  --prediction-root results/my_run_pred \
  --soft-nudge-weight 0.0   # or any of the flags documented below

python sam2/sav_dataset/sav_evaluator.py \
  --gt_root data/sav/sav_val/Annotations_6fps \
  --pred_root results/my_run_pred
```

Key scripts:

| Script | What it does |
|---|---|
| `sav_sam2_eval.py` | Baseline prompting policies (`box_centroid`, `largest_component_box_centroid`), no drift correction |
| `sav_motion_prior_reprompt_eval.py` | The core drift-correction mechanism and every mitigation flag (A1/B1/A2/D1-D4/F1 — see docstring and `--help`), scoped to a curated object subset for fast tuning pilots |
| `sav_full_pipeline_eval.py` | Same mechanism, full unselected SA-V val/test (155/150 videos) — this is where the false-positive cost actually shows up |
| `sav_dryrun_risk_eval.py` | E1: uncorrected dry-run pass collecting per-object risk features for pre-selection experiments |
| `sam2_multimask_capture.py` | Monkey-patches SAM2's `track_step` to capture the discarded multi-mask candidates it computes but never keeps |
| `sav_dryrun_multimask_risk_eval.py` | Uncorrected dry-run pass extracting multi-mask divergence features (the §13 follow-up to E1's pre-selection check) |
| `sav_risk_feature_correlate.py` | Reusable retrospective correlation/decision-gate harness for risk features vs. capstone delta (used by E1 and the multi-mask check) |
| `sam2_memory_gate.py` | Monkey-patches `propagate_in_video` to skip memory writes on suspected-distractor frames (dynamic memory-gating, Phase A) |
| `sav_memory_gate_eval.py` | n=30 pilot threshold sweep for memory-gating on the curated known-drift subset |
| `sav_memory_gate_full_eval.py` | Full SA-V val evaluation of memory-gating (293 objects), mirroring `sav_full_pipeline_eval.py`'s structure |
| `sav_failure_mode_analysis.py` | Identifies which objects are genuine drift failures (used to build the curated subset above) |
| `davis_sam2_eval.py` | Baseline validation on DAVIS (sanity-checking the SAM2 setup before moving to SA-V) |

Every mitigation flag added during this project (`--require-both-triggers`,
`--plausibility-ratio`, `--max-correction-rate`, `--correction-window`,
`--early-trigger-skip-frac`, `--rate-warmup-corrections`, `--soft-nudge-weight`)
defaults to `0`/disabled and reproduces the plain baseline exactly when left
unset — see each flag's `--help` text for the specific experiment and dated
result it corresponds to.

## Results summary

| Configuration | SA-V val J&F | SA-V test J&F |
|---|---:|---:|
| `box_centroid` baseline (no correction) | 72.1 | 74.4 |
| Capstone: drift correction, unselected (OR-gate) | 68.7 | — |
| A1 (AND-gate trigger) | 69.0 | — |
| B1 (result plausibility-check) | 68.2 | — |
| A2 (object-level correction-rate circuit breaker) | 71.1 | 72.2 |
| **D4 (A2 + A1 stacked, best result of the series)** | **71.3** | **72.6** |
| E1 (pre-selection, all 3 feature variants) | no predictive signal (R² ≤ 0.044) | — |
| F1 (soft-nudge correction blend) | 67.9 | — |

Full per-experiment numbers, mechanisms, and dated write-ups are in
[`sam2-drift-recovery-next-directions.md`](sam2-drift-recovery-next-directions.md)
(the main experimental narrative, §1-15 — §14-15 cover two later,
literature-motivated follow-up investigations beyond this series: dynamic
memory-gating, pilot-positive but full-val negative, and object-pointer anchor
recall, whose pilot did not pass its gate) and
[`sam2-drift-recovery-results.md`](sam2-drift-recovery-results.md) (raw iteration
log).

## License

This project's own code (`scripts/`) and documentation are licensed under the
[Apache License 2.0](LICENSE). SAM2 itself is a separate dependency under its own
Apache 2.0 license (Meta AI); the SA-V dataset has its own license terms from
Meta AI and is not redistributed here.
