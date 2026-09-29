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
  currently-tracked files with **one single root commit**, and pushed
  only that branch to the `public` remote as `main`
  (`git push public public-clean:main`).
- **Correction (same day):** the first squash commit (`da49fc8`) still
  carried this repo's standing `Co-Authored-By: Claude Sonnet 5` commit
  trailer. On explicit user instruction to drop it, recreated the orphan
  branch from scratch against the (by-then-updated) `main` and
  re-committed with no trailer (`f329fb1`), then force-pushed to replace
  the public commit (`git push public public-clean:main --force`).
  Verified via `git fetch public && git show --format="%B" -s
  public/main` that the trailer is gone from the live remote commit.
  Scope decision: this only touched the *public* squash commit — local
  `main`'s 32 real commits (many of which carry the same trailer from
  this session's earlier work) were left untouched, since rewriting that
  history would break the exact-commit-hash citations already written
  into several of this project's own planning documents, and local
  history is never pushed anywhere with the trailer visible. Going
  forward in this project, new commits I make will omit the trailer per
  this instruction, overriding the standing attribution default.
- Ran a quick pre-push scan for secrets/tokens/API-key patterns and
  personal-identifier leakage across the squashed content — none found
  (two files reference the local `/home/consul/...` dev path; not
  sensitive, left as-is).
- `public-clean` branch remains locally available for future re-squash
  snapshots (each future public sync = fresh orphan commit + force-push,
  not an accumulating history on the public side).

**Verify (done):** `git log --oneline main | wc -l` → 32 (unchanged);
`git log --oneline public/main` → single commit (`f329fb1` as of the
Co-Authored-By correction below); `git remote -v` shows `public`
pointing at the GitHub URL above; the repository is visible at that URL.

## Task 3 (conditional on Task 2) — Archival deposit with a persistent DOI

**Status: DONE.**

- [x] User connected **https://github.com/bmnedashkivskyi-ai/no-oracle-drift-correction**
  to Zenodo directly (their own account authorization — not something I
  could do).
- [x] Cut `v1.0.0`: created git tag `v1.0.0` on the public squash commit
  (`f329fb1`), pushed it, then created a GitHub Release via
  `gh release create v1.0.0`. Zenodo's webhook picked it up and minted
  **DOI `10.5281/zenodo.22811949`** (concept DOI, which will track future
  versions: `10.5281/zenodo.22811948`) — verified the version DOI
  resolves via Zenodo's public API (`GET
  https://zenodo.org/api/records/22811949`) and its title matches this
  release before using it anywhere.
- [x] Updated the Data Availability Statement in all four paper source
  files (commit `58611da`) to cite the real repo URL and the version DOI
  in place of the old generic "accompanying project code repository"
  phrasing. Rebuilt and verified both PDFs.

**Known, expected chicken-and-egg gap:** the `v1.0.0` release archived on
Zenodo does **not** itself contain the DOI self-citation (impossible for
a first release — you can't cite your own not-yet-minted DOI). The
*public GitHub repo's* `main` branch has since moved ahead of that
archived snapshot (the DOI-citing paper commit above). This is normal
and common for first Zenodo releases; resolve it whenever convenient by
re-squashing `main` → `public-clean` → force-push, then cutting a
`v1.0.1`/`v1.1.0` release so Zenodo archives a self-citing snapshot too
— entirely your call on timing, not urgent.

**Resolved (2026-09-29):** released `v1.0.1`, which contains the
DOI-citing paper plus the Ukrainian-wording, table-layout and Phase B
status fixes. Unlike the v1.0.0 sync, the public commit was created on
top of `public/main` (tree = local `main`, parent = the v1.0.0 commit)
and pushed without `--force`, so the public history is now v1.0.0 →
v1.0.1 and no private history is exposed.

**Verify (done):** DOI resolves and its Zenodo record title matches
("...no-oracle-drift-correction: v1.0.0 — Initial public release");
`grep -o "10.5281/zenodo[.0-9]*"` finds the correct DOI in both rebuilt
PDFs' extracted text; the old generic phrase is gone from both.

## Task 4 (optional, your call) — venue-specific steps

**Status: NOT APPLICABLE — closed by explicit user decision.** Reasoning
given: the project didn't achieve an improvement over baseline — it's a
negative-results study plus two further negative-results follow-up
investigations (Sections 4.5/7) — so neither an arXiv timestamp-priority
claim nor a formal venue submission serves a real purpose here; nothing
about the finding's substance depends on being first to claim it, and
there's no positive result whose priority would need protecting. Both
candidate paths below are explicitly declined, not merely deferred.

Candidate paths considered and declined:
- ~~arXiv submission~~ — a permanent timestamped identifier is only
  valuable for priority protection; not applicable to a negative result.
- ~~OSF project registration~~ — no funder/institutional requirement
  driving it; Zenodo already covers the archival-DOI need (Task 3).

**Verify:** no action taken on either path; this plan is now fully
closed (Tasks 1–4 all resolved — 3 done, 1 explicitly not applicable).
