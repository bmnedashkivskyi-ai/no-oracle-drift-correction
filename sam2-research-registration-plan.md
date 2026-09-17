# Research registration work — execution plan

Follows from the research summary in this session's conversation: classic
prospective preregistration does not apply retroactively to this project
(all experiments already run; the series is exploratory/diagnostic, not
confirmatory hypothesis-testing — see ARS's own decision tree,
`deep-research/references/preregistration_guide.md` §1, and its explicit
"Completed-artifact handoff boundary" rule against backdated
preregistration documents). What remains legitimately actionable is an
honest non-preregistration disclosure plus optional archival registration
(Zenodo/OSF/arXiv) of the already-completed work.

Each task below ends with an explicit **Verify** step — check that step's
actual on-disk/repo state before moving to the next task, whether that
check is done by me in the same turn or by you afterward.

## Task 1 — Add a Preregistration Statement to the paper

**Status: DONE (commit `5c4dbb6`; wording revised in `428dac3` after the
Task 2 decision below — see that task's note).**

- [ ] Add a new short section, **"Preregistration Statement"**, to
  `paper/draft.md` and `paper/draft_uk.md`, placed after "Ethics
  Statement" and before "AI-Use Disclosure" (matching the existing
  Data Availability → Ethics → AI-Use → Conflict of Interest → Funding
  ordering of short administrative sections at the end of the paper).
  Content (adapted from the ARS "Disclosure When Not Preregistered"
  template to this project's actual character): state plainly that the
  work was not preregistered, that the series was iterative and
  exploratory by design (each direction motivated by the prior one's
  result, not specified in advance), and that no confirmatory
  statistical hypothesis tests are reported (cross-reference Section 3.5
  / the Limitations section, which already states this).
- [ ] Mirror the same section in `paper/paper.tex` and
  `paper/paper_uk.tex`, in the same document position, respecting each
  file's existing LaTeX conventions (section command, no new
  `\newunicodechar` mappings needed if only plain ASCII/already-mapped
  characters are used — check before adding any new symbol).
- [ ] Rebuild `paper/paper.pdf` (pdflatex + bibtex + pdflatex ×2) and
  `paper/paper_uk.pdf` (xelatex + bibtex + xelatex ×2). Confirm no
  errors, no undefined citations (there shouldn't be any new citations
  in this section).
- [ ] Extract text from both rebuilt PDFs and grep for the new section
  title in both languages to confirm it actually renders, not just
  exists in source.
- [ ] Commit all six touched files (draft.md, draft_uk.md, paper.tex,
  paper_uk.tex, paper.pdf, paper_uk.pdf) together.

**Verify:** `git show --stat <commit>` shows exactly those six files;
`grep -i "preregist" paper/draft.md paper/draft_uk.md` finds the new
section in both; extracted PDF text (`mutool draw -F txt`) contains the
new section heading in both `paper.pdf` and `paper_uk.pdf`.

## Task 2 — Decision gate: make the repository public

**Status: DONE, with a revised approach — user explicitly chose to
squash history for the public copy rather than push it in full (this
plan originally recommended the opposite; recorded here for the
record).** Pushing full history and squashing were in direct tension
with Task 1's original wording, which claimed the exploratory/
confirmatory distinction was "preserved... in the project's full public
iteration history" — that claim would have become false once squashed.
Resolved by rewriting that sentence to rest only on the paper's own
internal chronological framing of results (Sections 3.4–4.5), which
holds regardless of the repository's git history (commit `428dac3`).

Actual execution:
- Repository: **https://github.com/bmnedashkivskyi-ai/no-oracle-drift-correction**
  (created by the user directly via the GitHub UI/API, not by me).
- Local `main` (32 commits, full private history) was left completely
  untouched — no rebase, no reset, no force-push against it.
- Created a separate orphan branch `public-clean` from `main`'s current
  HEAD (`git checkout --orphan public-clean`), containing all 227
  currently-tracked files with **one single root commit** (`da49fc8`,
  "Initial public release..."), and pushed only that branch to the
  `public` remote as `main` (`git push public public-clean:main`).
- Ran a quick pre-push scan for secrets/tokens/API-key patterns and
  personal-identifier leakage across the squashed content — none found
  (two files reference the local `/home/consul/...` dev path; not
  sensitive, left as-is).
- `public-clean` branch remains locally available for future re-squash
  snapshots (each future public sync = fresh orphan commit + force-push,
  not an accumulating history on the public side).

**Verify (done):** `git log --oneline main | wc -l` → 32 (unchanged);
`git log --oneline public/main` → exactly one commit, `da49fc8`;
`git remote -v` shows `public` pointing at the GitHub URL above; the
repository is visible at that URL with a single commit in its history.

## Task 3 (conditional on Task 2) — Archival deposit with a persistent DOI

**Status: unblocked (Task 2 done) — not started.**

- [ ] Connect **https://github.com/bmnedashkivskyi-ai/no-oracle-drift-correction**
  to Zenodo (Zenodo's GitHub integration; requires you to authorize it
  via your own Zenodo/GitHub account — not something I can do on your
  behalf).
- [ ] Cut a tagged release (e.g. `v1.0`) once connected, so Zenodo mints
  a DOI for that snapshot. Note: since the public repo's history will be
  replaced wholesale on each future sync (fresh squash, per Task 2's
  revised approach), a Zenodo release should be cut only after you're
  confident the current squashed snapshot is the one you want archived
  under a permanent DOI — a later re-squash-and-force-push changes the
  repo's default branch content but does not retract an already-minted
  DOI/release archive, so re-squashing after minting a DOI would leave
  the archived release referencing stale content relative to the live
  repo. Decide the sync cadence before minting.
- [ ] Update the paper's Data Availability Statement (all four source
  files again) to cite the real repository URL and the DOI, replacing
  the current generic "accompanying project code repository" phrasing.
  Rebuild both PDFs again.

**Verify:** the Zenodo record exists and resolves; the DOI is a real,
resolvable identifier; the paper's Data Availability Statement quotes
that exact DOI and URL; PDFs rebuilt and re-verified same as Task 1.

## Task 4 (optional, your call, no action until you decide) — venue-specific steps

**Status: not started, no dependency assumed yet — needs you to name a
target venue first, since the two candidate paths differ:**

- **arXiv submission** — gets a permanent timestamped identifier;
  straightforward, no venue-specific formatting constraints beyond
  arXiv's own submission requirements.
- **OSF project registration** — an alternative or complement to Zenodo,
  more common in some fields than in ML; only worth doing if you have a
  specific reason to prefer it over Zenodo (e.g., a funder or
  institutional requirement).

Nothing to verify yet — this task doesn't start until you pick a
direction.
