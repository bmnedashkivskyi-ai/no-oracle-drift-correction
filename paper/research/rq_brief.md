# Research Question Brief

Stage 1 (RESEARCH) deliverable — literature contextualization for the SAM2.1 SA-V
drift-recovery ablation study at `/home/consul/autoresearch`.

## Framing research question

**How does prior literature on (a) promptable video object segmentation with SAM/SAM2,
(b) tracking drift and re-identification in video object segmentation/tracking, (c)
test-time self-correction of frozen models, and (d) false-positive-rate framing in
rare-event/imbalanced detection, explain — and situate — the empirical finding that a
no-oracle drift-correction mechanism (motion-prior fresh-box-reprompt, "P5") produces
large gains on a ground-truth-preselected known-drift subset of SA-V-val objects
(+15.5pp J&F) but is net-harmful when applied unselectively to the full, unselected
object population (val J&F 72.1→68.7, -3.4pp), and that eleven further mitigation
attempts across four families never closed the gap to the plain uncorrected baseline?**

This is *not* a request for new experiments. It is a literature review whose job is to:
1. locate this project's specific mechanisms (P5, A1, B1, A2, D1-D4, E1, F1) within
   existing published techniques,
2. identify what in the existing literature already predicts or explains the observed
   failure pattern (trigger false positives dominating a rare true-positive class),
3. surface concrete, untested techniques from the literature that could plausibly
   target the diagnosed bottleneck (trigger specificity, not correction precision), and
4. honestly flag where this project's approach is not novel vs. where its specific
   contribution (measuring correction precision AND trigger specificity separately, on
   a full unselected population rather than only a curated true-positive subset) is
   thinly covered by prior ablations.

## Sub-questions

1. **SQ1 (mechanism precedent).** Within promptable video segmentation (SAM/SAM2 and
   the broader memory-based VOS family: STM, XMem, Cutie, DEVA) and classical visual
   tracking (template-update trackers, Siamese re-detection), what published
   drift-detection and drift-correction mechanisms already exist, and how do they
   compare — in design, not just outcome — to this project's area-ratio/position-jump
   trigger and fresh-box-reprompt correction (P5), and to its mitigation attempts
   (AND-gating, rate circuit-breaking, plausibility-checking, soft-nudging)?
2. **SQ2 (why net-harm on the full population).** What does the test-time-adaptation /
   self-training literature (confirmation bias, error accumulation, "when does
   self-correction help vs. hurt") and the imbalanced/rare-event detection literature
   (precision collapse under low base rate, PR vs. ROC framing) say about why a
   correction mechanism with strong true-positive performance can still be net-harmful
   when applied to a population where the target condition is rare (~10% of SA-V-val
   objects, per `sam2-drift-recovery-next-directions.md` §2)?
3. **SQ3 (untested remedies).** What concrete, published techniques — not among the
   twelve mechanisms already tried in this project — could plausibly raise trigger
   specificity or reduce the cost of a false-positive correction, and are there
   already-published SAM2-specific mitigations (e.g., SAM2Long, DAM4SAM) that overlap
   or diverge from this project's negative results?

## In scope

- Peer-reviewed or arXiv-preprint work on: SAM/SAM2 architecture and known failure
  modes; VOS memory mechanisms and long-horizon drift; classical tracking template
  drift/re-detection; test-time adaptation and self-training error accumulation;
  precision/recall and false-positive-rate framing for rare-event/imbalanced detection.
- Only sources actually fetched or confirmed via WebSearch/WebFetch with a resolvable
  URL/DOI (see `methodology_note.md`).

## Out of scope

- Fine-tuning or weight-updating approaches to SAM2 (this project studies a frozen
  model by design — `sam2-drift-recovery-results.md` §1).
- General image segmentation literature not connected to video propagation, drift, or
  test-time correction.
- New experiments, new SA-V runs, or re-analysis of this project's existing result
  files beyond what is already recorded in them.
- Exhaustive systematic-review-style coverage (this is a scoped contextualization pass
  for a negative-results ablation paper, not a standalone survey).
