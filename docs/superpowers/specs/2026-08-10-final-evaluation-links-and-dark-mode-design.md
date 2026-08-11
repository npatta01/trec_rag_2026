# Final Evaluation Links and Dark-Mode Repair Design

**Date:** 2026-08-10

**Status:** approved in conversation

## Objective

Finish the completed-project documentation with three direct analysis links on
the main artifact hub:

1. the existing 119-topic Retrieval quality analysis;
2. a new 119-topic RAGDoll citation-support analysis for accepted run
   `rag26-ss1`;
3. a new 119-topic RAGDoll citation-support analysis for accepted run
   `rag26-ms1-final`.

Also repair the final architecture report so inline code remains readable in
dark mode. Keep all five accepted organizer files immutable, preserve the
project's private-evaluation boundary, and leave public GitHub Pages-style
navigation honest about which reports require tailnet access.

## Verified Starting State

- [`reports/2026-retrieval-nugget-coverage.html`](../../../reports/2026-retrieval-nugget-coverage.html)
  is the existing full 119-topic Retrieval quality report. A private portal
  copy also already exists, but neither the root hub nor the report index links
  to it.
- The older private `trec-rag-two-topic-offline-evaluation.html` page is an
  experiment. It does not evaluate either accepted 119-topic RAG artifact and
  must not be relabeled as a final report.
- The exact accepted RAG artifacts are:
  - `rag26-ss1`, SHA-256
    `a33c60325c198cf90178d5b4c6c1b04209f7d39de278736b6817a8c5666c02a0`;
  - `rag26-ms1-final`, SHA-256
    `72200f7a0e3be19f9c7e8f23d3845f894f5e95e26caff2986ae16f848b4dee00`.
- The accepted single-pass file contains 2,399 answer objects and 3,155
  statement-citation pairs. The accepted multi-stage file contains 4,567
  answer objects and 7,008 statement-citation pairs. A complete RAGDoll
  citation-support evaluation therefore has 10,163 judge tasks before cache
  reuse.
- The shared main checkout currently has 123 validated support-judge cache
  files. Cache identity includes the task, prompt contract, RAGDoll revision,
  provider, model, thinking level, and system prompt, so presence alone does
  not imply that all 123 will match these accepted runs.
- The authenticated `facet-deepseek-b40-v3` selected-evidence handoff remains
  available in the private main checkout with the bundle-recorded manifest
  SHA-256
  `31dc1b3578741339101b8199c0ea027e2d8405e828532aa58f86bf74c0542a0d`.
- Browser-computed styles reproduce the dark-mode defect. After switching the
  final architecture report to `body.quarto-dark`, inline code receives the
  intended lavender text `rgb(222, 181, 242)` but retains Bootstrap's light
  background `rgb(248, 249, 250)`. The report CSS sets only code wrapping, so
  no dark-theme rule currently replaces that inherited light background.

## User-Facing Information Architecture

The main hub will expose three independent analysis cards or links. It will
not add a fourth comparison landing page and will not hide the two RAG reports
behind one generic RAG link.

| Hub label | Target | Access |
|---|---|---|
| Retrieval Quality Analysis | Existing `reports/2026-retrieval-nugget-coverage.html` | Repository-relative; compatible with local browsing and GitHub Pages |
| RAG Analysis: `rag26-ss1` | New authoritative friendly RAGDoll report | Private tailnet portal |
| RAG Analysis: `rag26-ms1-final` | New authoritative friendly RAGDoll report | Private tailnet portal |

The same three-link vocabulary will appear in `reports/index.html`. The final
architecture report will add a concise evaluation section or artifact links
without duplicating the detailed reports. `AGENTS.md` will identify the
Retrieval report, the two private RAG reports, the evaluator skill, and the
accepted input artifacts as key orientation points.

The two private links will be visibly marked **Private / tailnet**. A public
GitHub Pages rendering may contain those links, but it must not imply that an
internet visitor can open them. This task does not enable GitHub Pages, deploy
the repository, or make a private RAGDoll report public.

## RAGDoll Evaluation Contract

### Scope and metric meaning

Each accepted RAG JSONL is evaluated independently over all 119 official
topics. One support task is created per statement-citation pair. RAGDoll's
support judge labels each pair:

- `FS`: the cited selected evidence fully supports the statement;
- `PS`: the evidence provides partial support;
- `NS`: the evidence does not support the statement.

The friendly report may present RAGDoll's citation-support metrics, label
counts, per-topic details, response text, citation positions, and summarized
provenance already allowed by the repository renderer. It must not call this a
general answer-correctness score or an official TREC score.

No 2026 qrels, released gold nuggets, or completed nugget-assignment run are
part of this evaluation. Retrieval effectiveness and nugget coverage inside
the RAG reports must therefore remain explicitly unavailable. Generated
claims must never be substituted for gold nuggets. The separate Retrieval
quality report remains the place to inspect the project's existing 2026
Retrieval coverage analysis.

### Exact accepted-artifact binding

The evaluation must read the checked-in accepted JSONLs without modifying,
reformatting, or overwriting them. Before task construction, it will verify:

1. accepted-file SHA-256, line count, run ID, and topic count against
   `submissions/trec-rag-2026/rag/selected-evidence-sol-v1/metadata.json` and
   `SUBMISSION_LEDGER.md`;
2. the exact 119 official narrative IDs, order, and text;
3. the authenticated selected-evidence handoff manifest hash;
4. each topic's context hash and narrative against that handoff;
5. every cited document belongs to that topic's selected-evidence set;
6. the materialized support task count equals the accepted file's exact
   statement-citation count.

The evaluator currently assumes a live single-pass output directory and one
`generation_identity.json` schema. The accepted single-pass runtime identity
was not preserved with the bundle, while the multi-stage run uses a different
`multistage_generation_identity.json` schema. Implementation will add an
explicit accepted-submission evaluation path to the existing authoritative
`trec_rag.competition_evaluation_report` workflow rather than fabricating a
missing historical runtime receipt.

That path will create a private, deterministic evaluation binding containing
the accepted artifact receipt, bundle-metadata receipt, authenticated handoff
receipt, run ID, and per-topic context receipts. It is a post-run evaluation
identity, not a claim that a missing generation receipt was recovered. When a
preserved source identity is available, as for the multi-stage run, it may be
validated and recorded as additional provenance. The friendly report will
state any missing historical provenance as unavailable rather than inventing
it.

### Judge execution and cache discipline

The first invocation for each run is cache-only and makes zero hosted calls.
It reports the exact task count, compatible cache hits, misses, and report
availability. If misses remain, the already-approved hosted workflow is:

1. print the judge provider (`pi`), model (`openai-codex/gpt-5.5`), thinking
   level (`medium`), exact miss count, and payload shape (one generated
   statement plus its cited selected-evidence text and source metadata);
2. run one validated cache miss with `--run-judge --judge-limit 1`;
3. require one completed label, zero failed/conflicting labels, a reusable
   validated cache entry, and a renderable partial report;
4. resume the full run so the probe is a cache hit and only remaining misses
   call the judge;
5. rerun cache-only and require zero missing, failed, or conflicting judgments
   before calling the report complete.

Because 10,163 tasks are too large for an accidental unbounded process, hosted
execution will retain an explicit bounded worker setting, durable per-task
cache writes, resumability, and failure accounting. Scheduling must not alter
task or cache identities. A failed call remains missing and is retried only by
a later explicit resume; it is never converted into a guessed label.

## Private Artifacts and Portal Publication

For each run, keep the following under an ignored, mode-`0700` evaluation work
directory in the shared checkout or another private path:

- accepted-submission evaluation identity;
- support inputs and task JSONL;
- raw judge events;
- support judgments and assignments;
- evaluation manifest and receipts;
- the first authoritative rendered HTML report.

Raw selected passages, judge prompts/events, task files, judgments, and
manifests must not enter git or the rendered portal. Only the privacy-scanned
friendly HTML produced by `trec_rag.friendly_report` may be copied to:

- `~/codex-rendered/plans/trec-rag-2026-ragdoll-rag26-ss1.html`;
- `~/codex-rendered/plans/trec-rag-2026-ragdoll-rag26-ms1-final.html`.

The private portal index and the derived final-project hub will link those two
pages. Before handoff, each HTTPS URL must return the intended report, remain
behind the existing tailnet-only Tailscale Serve mapping, and pass a privacy
scan for credentials, private filesystem paths, selected passage text, raw
events, and unrelated repository material.

## Dark-Mode Repair

The canonical fix belongs in
`reports/2026-competition-architecture/report.css`; generated HTML will only
change by re-rendering the QMD with Quarto.

Add explicit light and dark inline-code design tokens for foreground,
background, and border. Apply them to inline code with enough specificity to
override Bootstrap while excluding source-code blocks. The dark values must
use a genuinely dark surface with high-contrast lavender text; the light
values must retain the current visual role without relying on Bootstrap's
default gray.

Verification will inspect browser-computed styles after using the report's
actual theme toggle, not merely `prefers-color-scheme`. It will require:

- a dark inline-code background rather than `rgb(248, 249, 250)` or another
  light Bootstrap gray;
- WCAG AA contrast for code text against its background;
- unchanged readable source-code blocks;
- no clipping or overflow at desktop and mobile widths;
- a fresh screenshot confirming that the glossary tokens shown in the user
  report are readable.

## Alternatives Considered

### One comparison page for both RAG runs

Rejected. The user wants the hub to provide one direct link per analysis, not
an additional comparison layer. The reports may be interpreted together, but
no post-hoc combined ranking is required.

### Commit both detailed RAGDoll reports into `reports/`

Rejected. The repository's evaluation workflow keeps detailed RAGDoll reports
and source evaluation material private. Public GitHub Pages must not silently
change that boundary.

### Reuse the old two-topic offline report

Rejected. Its scope and inputs do not match either accepted 119-topic artifact.

### Three direct hub links with private RAG detail pages

Selected. It matches the requested navigation, reuses the verified Retrieval
report, keeps the two accepted RAG strategies distinct, and preserves the
tailnet-only evaluation boundary.

## Documentation Changes

Expected tracked changes are limited to documentation, evaluator support and
tests, report source styling, and re-rendered public HTML:

- root `index.html` and `index.test.js`;
- `reports/index.html` and its tests;
- `reports/2026-competition-architecture.qmd`, `report.css`, rendered HTML,
  and relevant render tests;
- `AGENTS.md` key-artifact orientation;
- the repo-local competition evaluation skill/README if CLI behavior changes;
- accepted-submission evaluation support in `code/trec_rag/` with focused
  tests.

No accepted TSV/JSONL, submission metadata, sealed handoff, raw output,
provider response, cache entry, or judge artifact may be changed or committed.

## Verification and Completion Gate

Completion requires all of the following:

1. Re-run the repository submission preflight and prove all five accepted
   artifact hashes remain unchanged.
2. Run focused tests for accepted-artifact binding, both identity shapes,
   cache-only behavior, probe limits, bounded resume, metric availability, and
   privacy rejection.
3. Produce cache-only receipts for both final accepted runs before any hosted
   call.
4. Complete and validate the one-task hosted probe before the full resume.
5. Finish all 3,155 and 7,008 support tasks with zero missing, failed, or
   conflicting judgments, or report the run as incomplete rather than
   publishing a final score.
6. Re-render each friendly report from its validated manifest and confirm
   report totals reconcile with source tasks and labels.
7. Independently recompute label totals and high-impact macro metrics from the
   private assignments, then compare them with the HTML presentation model.
8. Run the report privacy scanner and inspect the two final HTML pages at
   desktop and mobile widths.
9. Re-render and browser-test the architecture report in light and dark modes,
   including computed inline-code colors and contrast.
10. Run hub/report link tests, internal-link checks, `git diff --check`, and the
    relevant repository test suite.
11. Refresh the private portal, verify all three analysis links over HTTPS,
    and confirm the Serve mapping remains tailnet-only.
12. Confirm `git diff` contains no changes to the five accepted organizer
    artifacts and no private evaluation material.

## Out of Scope

- No new Retrieval, reranking, or answer-generation run.
- No use of RAGDoll results to reorder or alter accepted submissions.
- No official-score claim before organizer judgments are released.
- No public deployment, GitHub Pages enablement, Tailscale Funnel, Codex
  Sites, or sharing-permission change.
- No public copy of detailed RAGDoll reports or private evaluation inputs.
