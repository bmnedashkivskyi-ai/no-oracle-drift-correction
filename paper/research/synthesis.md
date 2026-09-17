# Synthesis: Related Work, Contradictions/Confirmations, Candidate Future Work, Limitations

This document is the main Stage-1 deliverable. It maps this project's already-complete
empirical findings (sourced from the eight background documents listed in
`methodology_note.md`) against the 16 externally-verified sources in `bibliography.md`.
All external claims cite a bibliography entry by author-year; all project claims cite
the specific source document and section.

## (a) Related Work Map

### Area 1 — SAM / SAM2 and promptable video object segmentation

This project's entire pipeline sits inside the paradigm SAM (Kirillov et al., 2023)
established and SAM2 (Ravi et al., 2024) extended to video: prompt-to-mask on a frozen
model, no fine-tuning. The project's `box_centroid` and `largest_component_box_centroid`
(P2) prompt policies are direct applications of SAM's point/box-prompt interface
(`sam2-full-validation-report.md`); its evaluation splits (155/293 val, 150/~278 test)
are the official SA-V val/test partitions introduced alongside SAM2 and documented on
the dataset's project page (Meta AI, 2024).

Where this project sits relative to prior art: SAM2's own reported efficiency claim
("3x fewer interactions than prior approaches," Ravi et al., 2024) is precisely the
regime this project stress-tests — not whether SAM2 can be prompted well on a single
frame, but whether a *frozen* SAM2's own propagation can be corrected, without oracle
access, when it drifts. The project's failure-mode decomposition
(`sam2-sav-failure-analysis.md`; `sam2-drift-recovery-results.md` §3) — 83% of SA-V-val
failures are propagation-drift with a correct start, not prompt-frame collapse — is a
project-original empirical finding not directly reported in either SAM/SAM2 paper, but
it is consistent with what Ding et al. (2024, SAM2Long) independently diagnose as SAM2's
memory-buffer "error accumulation": an errored mask written into memory propagates
forward, and on long videos the buffer eventually loses its original anchor, allowing
drift onto a distractor. That two independent analyses (this project's own oracle-vs-
no-oracle ceiling experiment, `sam2-sav-failure-analysis.md` §"Oracle ceiling-тест",
+58.2pp; and SAM2Long's published diagnosis) converge on the same mechanism from
different angles is a genuine confirmation, not merely a shared vocabulary — see (b)
below.

### Area 2 — Tracking drift, re-identification, occlusion recovery, memory-based VOS

The memory-based VOS lineage this project's SAM2 checkpoint descends from — STM (Oh et
al., 2019) → XMem (Cheng & Schwing, 2022) → Cutie (Cheng et al., 2023a) / DEVA (Cheng
et al., 2023b) — each addressed a specific weakness of the previous generation's memory
mechanism: STM established dense feature-space memory matching; XMem added a
three-tier (sensory/working/long-term) memory specifically to survive long videos
without losing the anchor frame; Cutie moved from pixel-level to object-level memory
reading specifically because pixel matching is distractor-fragile. This project's
capstone finding — that a correction mechanism triggered on a frozen SAM2's own
propagation signal is net-harmful when unselectively applied (`sam2-drift-recovery-
results.md`, capstone section: val `J&F` 72.1→68.7) — is a symptom of exactly the
distractor-fragility problem Cutie's architecture targets, but this project cannot draw
on an architectural fix because its frozen-model, inference-time-only design (charter
§1) rules out retraining or architectural substitution.

Classical visual tracking offers a second, independent lineage on the same problem.
DaSiamRPN (Zhu et al., 2018) is the earliest cited precedent for the *specific*
failure this project's P5 targets: "semantic distractors" — visually near-identical
objects a similarity signal cannot discriminate. This project's own multi-instance
cases (`sav_022396`, ~15 nearly identical balloons; `sav_035221`, two visually
identical shoe soles; `sam2-drift-recovery-results.md` §10, P5 section) are textbook
instances of exactly this failure mode, independently rediscovered through this
project's own P3 diagnostic (`sam2-drift-recovery-results.md` §"P3", 2026-08-31) before
P5 was designed. Where this project's P5 differs from DaSiamRPN: DaSiamRPN's
distractor-awareness is *learned* (an incremental appearance-based discriminator);
P5's is *purely positional* (constant-velocity centroid extrapolation), chosen
specifically because appearance is uninformative between visually identical instances
(`sam2-drift-recovery-results.md` §10: "P5 використовує лише ПОЗИЦІЮ, єдиний сигнал,
що не плутається між візуально ідентичними об'єктами"). Siam R-CNN (Voigtlaender et
al., 2020) adds a further relevant precedent — explicit re-detection plus tracking of
distractor objects alongside the target, resolved via dynamic programming over
multiple hypotheses through time — which is structurally different from, and
potentially complementary to, this project's per-trigger, single-hypothesis reprompt.

Two SAM2-specific follow-ups postdate and directly address the mechanism this project
diagnosed independently: DAM4SAM (Videnovic et al., 2024) adds a distractor-aware
memory and an *introspection*-based update — deciding whether to trust and write a
frame's result *before* committing it to memory — which is close in spirit to, but
architecturally distinct from, this project's B1 (post-hoc plausibility check on a
reprompt's output) and A2 (object-level rate circuit-breaker). SAM2Long (Ding et al.,
2024) maintains multiple candidate propagation pathways continuously and prunes by
cumulative score, rather than this project's binary detect-then-hard-reprompt design.
Neither was tried by this project (see Candidate Future Work, (c) below).

**What is genuinely novel about this project's framing relative to this area's prior
art**: none of the cited VOS/tracking papers report measuring *both* (i) correction
precision on a ground-truth-preselected true-positive subset *and* (ii) net effect
of the same mechanism applied unselectively to the full, unfiltered object population,
as two separate, explicitly labeled quantities. This project's own documents state this
directly and are the origin of the framing used here: "Капстоун-прогін... показав, що
специфічність — це і є справжнє вузьке місце, а не точність корекції"
(`sam2-drift-recovery-next-directions.md` §1) — i.e., the paper's own next-directions
document explicitly identifies, as a methodological gap in its *own* prior eleven
attempts, that correction precision and trigger specificity were conflated until the
capstone run. No source found in this review's search pass reports the analogous
measurement (precision on a curated positive subset vs. net effect on an unfiltered
population, explicitly separated and both quantified) for a VOS drift-correction
mechanism; this is consistent with, though not conclusive proof of, the project's own
claim that this gap is "rarely done explicitly in prior ablations."

### Area 3 — Test-time self-correction / online adaptation of frozen models

This project's no-oracle constraint (charter §1: correction "без доступу до ground
truth під час inference") places its entire P5/A1/A2/B1/D1-D4/E1/F1 series inside the
test-time-adaptation (TTA) problem class, even though the project's own documents never
use that term. Tent (Wang et al., 2020) is the canonical example of a frozen model
adapted online using only its own predictions — structurally the same setting. CoTTA
(Wang et al., 2022) is the more directly relevant TTA paper: it diagnoses that
long-horizon self-supervised adaptation on unreliable pseudo-labels causes **error
accumulation**, and mitigates it via weight-averaging and stochastic parameter
restoration — a *dampening*, not *gating*, strategy. This project's own "correction
cascade" mechanism (`sam2-sav-failure-analysis.md`: objects with >15 self-corrections
regress by mean -10.6pp vs. +20.6pp for objects with ≤15 corrections) is an
independently observed instance of the same accumulation dynamic CoTTA was built to
prevent, and this project's A2 (rate circuit-breaker, the single best-performing
mitigation of the twelve tried — `sam2-drift-recovery-next-directions.md` §5, step 4)
is, in effect, a hard-threshold version of CoTTA's soft dampening idea, arrived at
independently and without reference to the TTA literature.

Arazo et al.'s (2020) confirmation-bias analysis of pseudo-labeling gives the general
mechanism name for this project's self-referential-source finding: F1-F4
(`sam2-drift-recovery-results.md` §9) all share "джерело корекції одного роду з
сигналом детекції" (the correction source is the same kind of signal as the detection
signal) — precisely the confirmation-bias structure Arazo et al. describe for
classification pseudo-labels. Huang et al. (2023), though from the LLM-reasoning
literature and not vision, is included as a cross-domain structural analogue: their
finding that self-correction *without an external, reliable oracle* frequently makes
outputs worse rather than better mirrors this project's capstone finding almost
exactly (self-triggered correction, unfiltered, is net-harmful — `sam2-drift-recovery-
results.md`, capstone: -3.4pp val), and both point to the same structural requirement:
self-correction's benefit is bounded not by the quality of the correction step but by
the reliability of the trigger deciding *when* to apply it.

### Area 4 — Trigger-specificity / false-positive suppression under class imbalance

This is the area where a single external source most tightly explains this project's
central quantitative finding. Saito and Rehmsmeier (2015) formalize why a detector's
strong per-instance performance can still collapse to poor net precision under class
imbalance, because false positives on a large majority class overwhelm true positives
on a rare minority class. This project's own numbers instantiate the pattern exactly:
~10% of SA-V-val objects genuinely need drift correction
(`sam2-drift-recovery-next-directions.md` §2), and the capstone run's implied trigger
false-positive rate on the remaining ~90% "healthy" population is documented directly
in the project's own text as "false-positive rate тригера на здоровій популяції ~68%,
драматично вище за прийнятне" (`sam2-drift-recovery-next-directions.md` §2) — a rate
high enough that, even though P5's correction is highly *precise* on true positives
(individual gains up to +90pp J&F, `sam2-drift-recovery-results.md` §10), the net
population-level effect is negative (178 regressions vs. 44 improvements,
`sam2-drift-recovery-results.md`, capstone section). This is precisely the base-rate/
precision-collapse dynamic Saito and Rehmsmeier's paper formalizes, arrived at
independently by this project through direct measurement rather than through
engagement with the imbalanced-classification literature.

## (b) Explicit contradictions / confirmations

**Confirmations** (external literature independently predicts or matches this
project's findings):

1. **Memory-buffer drift onto distractors is a documented SAM2 property, not an
   artifact of this project's pipeline.** Ding et al.'s (2024) SAM2Long diagnosis of
   SAM2's own error-accumulation and buffer-rollover behavior, obtained independently
   and on different benchmarks, confirms the mechanism this project traces specifically
   in its `sav_017171` and `sav_004755` test-set catastrophe analysis
   (`sam2-drift-recovery-results.md`, "Закриття відкритих питань P5").
2. **Self-referential correction sources degrade via confirmation bias / error
   accumulation** — Arazo et al. (2020) and Wang et al. (2022, CoTTA) both formalize
   this for classification/TTA; this project's F1-F4 "correction cascade" finding
   (`sam2-sav-failure-analysis.md`) and its A2 rate-breaker mitigation
   (`sam2-drift-recovery-next-directions.md` §5, step 4) are an independent VOS-domain
   instance and dampening strategy for the same dynamic.
3. **Net harm from a locally-precise correction under class imbalance is a known,
   formalized statistical pattern**, not a project-specific anomaly — Saito and
   Rehmsmeier (2015) directly explains why the capstone run's ~68% healthy-population
   false-positive rate (`next-directions.md` §2) dominates P5's true-positive gains.
4. **Distractor confusion in appearance-only signals is a long-established tracking
   failure mode** — DaSiamRPN (Zhu et al., 2018) named and targeted this in 2018; this
   project's independent rediscovery of the same failure (P3 diagnostic,
   `sam2-drift-recovery-results.md` §"P3") and P5's positional (not appearance-based)
   response is consistent with, and partially reproduces, the tracking literature's
   established distinction between appearance-based and motion-based re-identification
   signals.

**Contradictions / tensions** (where this project's findings sit against a naive
reading of prior literature, or where prior literature's success stories do not
transfer):

1. **AND-gating two independent-seeming signals did not raise specificity as the
   ensemble-detection intuition would predict.** A1 (requiring both area-ratio AND
   position-jump) only recovered +0.3pp over the OR-gate capstone
   (`sam2-drift-recovery-next-directions.md` §5, step 2) because, mechanistically, both
   signals respond to the *same* underlying legitimate events (fast scale change moves
   both area and centroid at once). This is a caution against naively importing
   ensemble/multi-signal-agreement intuitions from anomaly detection without checking
   whether the signals are actually conditionally independent given the legitimate
   (non-drift) event distribution — a nuance not explicitly flagged in the general
   imbalanced-detection sources reviewed here.
2. **A "safer," conservative fallback made things worse, not better** (B1's
   plausibility-check with a non-velocity-extrapolating fallback box performed *worse*
   than no plausibility-check at all — `sam2-drift-recovery-next-directions.md` §5,
   step 3, and `sam2-drift-recovery-results.md` §B1). This runs against a naive reading
   of the TTA-dampening literature (CoTTA, Wang et al., 2022), where restoring toward a
   "safe" prior state is presented as protective; here, the specific choice of "safe"
   state (zero velocity) was itself miscalibrated for exactly the population where
   correction is most needed (fast-moving true-positive objects), producing a new
   self-reinforcing error cycle. The lesson is dampening-strategy-specific, not a
   contradiction of the general dampening principle, but it is a genuine tension worth
   flagging for the paper's Discussion.
3. **Test-set generalization gaps were much larger than any of the reviewed literature
   would predict from a val-tuned mechanism** — P5's val gain of +15.5pp fell to
   +2.0pp on held-out test, and P2's val gain of +0.4pp *reversed* to -0.3pp
   (`sam2-drift-recovery-results.md`, "P5 на held-out sav_test"; "Ізольований P2").
   None of the reviewed TTA or VOS papers report a comparably severe val→test reversal
   for a comparably simple (4-5 hyperparameter) mechanism; this is either a property of
   SA-V's own val/test distributional difference, or of the small true-positive
   samples (n=30 val, n=25 test) used for tuning — the project's own documents flag
   both as open, unresolved possibilities (`sam2-drift-recovery-results.md`, "Висновок"
   under the P5-test-gap section) and this review found no external source that
   resolves which explanation dominates.

## (c) Candidate Future Work (untested — not evaluated by this project)

The following are concrete, published techniques from the literature reviewed above
that target the diagnosed bottleneck (trigger specificity / false-positive
suppression, not correction precision on true positives) and were **not** among this
project's twelve tried mechanisms (P5, A1, B1, A2, D1-D4, E1's three feature variants,
F1). None of these have been implemented or tested by this project; they are literature-
derived suggestions only.

1. **Object-level introspective memory gating (DAM4SAM-style; Videnovic et al., 2024).**
   Rather than gating on trigger signals computed *after* a correction attempt (as B1
   does) or on accumulated correction rate (as A2 does), decide whether to *trust and
   write* a given frame's own propagated mask into memory *before* it can contaminate
   future frames — potentially catching the exact single-bad-frame failure this
   project diagnosed in `sav_004755` (one bad correction became a permanently wrong,
   self-reinforcing anchor; `sam2-drift-recovery-results.md`, "Закриття відкритих
   питань P5").
2. **Multi-path / memory-tree propagation (SAM2Long-style; Ding et al., 2024).**
   Maintain several candidate propagation pathways per object continuously, pruning by
   cumulative confidence, instead of this project's single-hypothesis,
   detect-then-hard-reprompt design — structurally avoids ever committing irreversibly
   to one bad correction, which is the specific failure mode of P5's fresh-box-reprompt
   on `sav_017171` and `sav_004755`.
3. **Learned, appearance-based distractor discrimination combined with P5's positional
   signal (DaSiamRPN-style; Zhu et al., 2018).** P5 uses position specifically because
   appearance cannot separate visually identical instances (`sam2-drift-recovery-
   results.md` §10); a hybrid signal — positional prior gated or weighted by a learned
   distractor-awareness score, as DaSiamRPN uses for classical trackers — was not
   tried, and could plausibly raise specificity on the ~90% healthy population without
   losing sensitivity on genuine multi-instance identity-switch cases.
4. **Object-level re-detection with multi-hypothesis dynamic programming (Siam R-CNN-
   style; Voigtlaender et al., 2020).** Track candidate distractor identities alongside
   the target through time and resolve ambiguity retrospectively via a
   tracklet-association step, rather than this project's per-trigger, forward-only,
   single-decision reprompt.
5. **Object-level memory reading instead of pixel/box-level correction (Cutie-style;
   Cheng et al., 2023a).** This project's corrections operate at the box/mask level;
   Cutie's object-query-based memory reading is a categorically different mechanism
   (top-down object representation vs. bottom-up pixel/box matching) that was not
   accessible to this project's frozen, prompt-only intervention surface, but could in
   principle be combined with a SAM2-based pipeline as an external re-identification
   module.
6. **Explicit precision-recall-based threshold calibration against the known ~10%
   positive-class base rate (Saito & Rehmsmeier, 2015 framing).** This project's
   trigger thresholds (`jump_ratio`, `low_ratio`, `high_ratio`) were tuned by
   maximizing J&F on a GT-preselected true-positive subset (n=30) — i.e., under an
   implicit 100% base rate — never against a precision/recall operating point computed
   on the true ~10% base rate of the full population. A threshold explicitly chosen to
   hold trigger *precision* (not just recall on true positives) above a target level on
   the full, unselected population was not tried, and the project's own E1 attempts
   (dry-run risk features, R²≤0.044, `sam2-drift-recovery-next-directions.md` §8, §11)
   approached but did not frame the problem this way explicitly.
7. **Long-term/tiered memory consolidation as an alternative to reprompting (XMem-
   style; Cheng & Schwing, 2022).** Rather than detecting drift and re-anchoring via a
   fresh box prompt, maintain a consolidated long-term memory bank that is inherently
   more resistant to single-frame corruption — this addresses the root cause (buffer
   rollover losing the true anchor) rather than reacting to its symptom, but requires
   architectural access this project's frozen-checkpoint, inference-only design does
   not use.

## (d) Limitations of this literature review

- **No access to paywalled citation databases.** All searches used `WebSearch`
  (general web/arXiv-indexed) and `WebFetch` against individual pages; no Scopus, Web
  of Science, or ACM Digital Library access was available. Coverage of
  non-arXiv-indexed venues (some journal-only anomaly-detection or classical-tracking
  work) is likely thinner than an institutional-database search would produce.
- **Area 4 (trigger-specificity / false-positive framing) has the thinnest coverage.**
  One strong, directly relevant anchor source (Saito & Rehmsmeier, 2015) was found and
  verified, but this review did not locate a comparably strong, specifically
  *changepoint-detection* (as opposed to static classification) paper matching the
  project's request; general anomaly-detection threshold-calibration material
  surfaced in search but no single additional paper was judged both sufficiently
  on-topic and independently verifiable within this pass's scope, so it was excluded
  per the non-fabrication policy in `methodology_note.md` rather than cited loosely.
- **Recency cutoff and moving target.** Several of the most directly relevant SAM2-
  specific papers (DAM4SAM, SAM2Long) are from late 2024 and were still receiving
  revisions into 2025; this project's own empirical series was conducted without
  reference to either, and it is possible additional SAM2-drift-specific work has
  appeared since this review's search pass (2026-09-14) that was not captured.
- **No independent replication or critical appraisal of cited papers' own claims.**
  This review reports what each paper's abstract and fetched content state; it does
  not independently verify their reported numbers (e.g., SAM2Long's claimed +3.0pp
  average improvement) the way this project verifies its own numbers against official
  evaluators. Claims from external papers should be read as "this paper reports X," not
  as independently confirmed by this review.
- **Scope was set by the task's four named areas**, not by an exhaustive
  citation-graph search (e.g., no systematic backward/forward citation chaining from
  the 16 included sources was performed). A more exhaustive systematic-review-style
  pass would likely surface additional classical tracking and TTA literature beyond
  what is included here.
- **One search area (Area 3) leans on a single cross-domain analogy (Huang et al.,
  2023, LLM self-correction)** that is not vision-domain; it is included because its
  structural finding is unusually close to this project's own, but readers should
  weigh it as an analogy, not a directly transferable empirical result.
