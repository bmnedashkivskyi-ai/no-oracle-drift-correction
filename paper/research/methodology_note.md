# Methodology Note

## What this is

This is a **literature-contextualization pass**, not new primary data collection. The
empirical study it supports (frozen SAM2.1 Hiera-Tiny on SA-V, no-oracle drift-recovery
ablations) is already complete — see `sam2-drift-recovery-charter.md`,
`sam2-drift-recovery-results.md`, `sam2-drift-recovery-next-directions.md` §1-12,
`sam2-full-validation-report.md`, `sam2-evidence-matrix.md`, `sam2-iteration-log.md`,
`sam2-sav-failure-analysis.md`, and `sam2-metric-protocol.md`, all read in full before
this review was written. No new SAM2 runs, no new SA-V evaluation, and no re-analysis
of the project's result CSVs were performed here. This document's sole job is to place
the project's already-fixed findings against the published record.

## Search strategy

**Engine.** All searches used the `WebSearch` tool (general web/arXiv-indexed search,
US region) and the `WebFetch` tool to open and read specific arXiv abstract pages,
GitHub pages, and one PLOS ONE journal page directly. No access to a paywalled
citation database (e.g., Scopus, Web of Science) was available in this environment —
see Limitations in `synthesis.md`.

**Query sequence**, grouped by the four areas named in the task:

1. *SAM/SAM2/SA-V and promptable VOS*: direct searches for the SAM and SAM2 papers by
   name/author, the SA-V dataset, and SAM2 video-segmentation failure-mode analyses
   (occlusion, distractor, drift).
2. *Tracking drift / re-identification / occlusion recovery*: memory-based VOS methods
   by name (STM, XMem, Cutie, DEVA), classical tracker template-update/drift surveys,
   Siamese re-detection and distractor-aware tracking (DaSiamRPN, Siam R-CNN), and
   SAM2-specific distractor/memory follow-ups (DAM4SAM, SAM2Long).
3. *Test-time self-correction / online adaptation of frozen models*: test-time
   adaptation surveys and canonical methods (Tent, CoTTA), pseudo-labeling
   confirmation-bias literature, and LLM self-correction failure literature (as a
   cross-domain analogy to "correction compounding damage").
4. *Trigger-specificity / false-positive suppression under class imbalance*: base-rate
   fallacy / false-positive-paradox framing, precision-recall vs. ROC evaluation under
   imbalance (Saito & Rehmsmeier), and anomaly/changepoint-detection threshold
   calibration under rare-positive-class conditions.

For each candidate source, the abstract page (arXiv `/abs/` page, or in one case the
publisher journal page) was fetched directly via `WebFetch` to confirm the paper is
real, to extract the exact author list and submission/publication date, and to record
the resolvable URL used in `bibliography.md`. No citation was added on the basis of a
search-result snippet alone — every entry in the bibliography was independently opened.

## Inclusion criteria

A source was included only if all of the following held:
1. It resolved to a real, independently fetched URL (arXiv abstract page, publisher
   page, or official project page) — confirmed via `WebFetch`, not inferred from a
   title.
2. It is directly relevant to one of the four scoped areas in `rq_brief.md`, not merely
   SAM2-adjacent by keyword overlap.
3. Its claimed relevance to this project's findings is traceable to a specific,
   verifiable statement in the fetched abstract/content — not to background knowledge
   about the paper.

## Exclusion / non-fabrication policy

Several plausible-sounding candidate topics surfaced in search snippets but were **not**
included because a corresponding real, fetchable paper was not confirmed within the
scope of this pass (e.g., a specific paper on "adaptive per-object drift thresholds
calibrated from early frames" matching this project's un-implemented A3 idea was not
found as a discrete citable work — general per-object/per-track adaptive thresholding
ideas exist in the tracking survey literature already cited, but no single paper was
verified as a precise match, so no citation is forced onto that idea). Where a claim in
`synthesis.md` cannot be backed by a verified source, it is stated as an open
observation, not attributed to a citation.

## Numeric-claim sourcing convention

Every specific number about *this* project (J&F values, percentage-point deltas, object
counts) in `synthesis.md` and `bibliography.md` annotations cites the specific source
document and section, e.g. "`sam2-drift-recovery-next-directions.md` §2" or
"`sam2-drift-recovery-results.md`, capstone section." Every claim about *external*
literature cites the specific paper in `bibliography.md` by its APA in-text key.
