# TREC RAG Workspace

This completed project contains the final TREC RAG 2026 Retrieval and RAG
submissions, their architecture and validation records, supporting reports, and
the reusable implementation that produced them. All five checked-in organizer
files were uploaded and accepted by Evalbase.

## Final Project Artifacts

- [`index.html`](index.html) - compact entrypoint to the final project.
- [`reports/2026-competition-architecture.html`](reports/2026-competition-architecture.html)
  - image-led walkthrough of the final frozen-source architecture and its two
  submission branches.
- [`reports/2026-competition-architecture.qmd`](reports/2026-competition-architecture.qmd)
  - canonical Quarto source for the generated architecture report.
- [`submissions/trec-rag-2026/SUBMISSION_LEDGER.md`](submissions/trec-rag-2026/SUBMISSION_LEDGER.md)
  - control sheet for the five accepted files, hashes, priorities, and portal
  notes.
- [`submissions/trec-rag-2026/retrieval/cache-first-candidate-core-v1/README.md`](submissions/trec-rag-2026/retrieval/cache-first-candidate-core-v1/README.md)
  - final three-run Retrieval bundle.
- [`submissions/trec-rag-2026/rag/selected-evidence-sol-v1/README.md`](submissions/trec-rag-2026/rag/selected-evidence-sol-v1/README.md)
  - final two-run RAG bundle.
- [Submission validation skill](.agents/skills/validate-trec-rag-2026-submissions/SKILL.md)
  - combined structural and organizer AutoJudge preflight.
- [`reports/index.html`](reports/index.html) - supporting report collection.
- [`code/trec_rag/README.md`](code/trec_rag/README.md) - implementation and
  reproduction reference.

The final system starts from one authenticated source retrieval artifact and
then branches. Cache-first replay produces three variable-depth Retrieval TSVs;
the sealed selected-evidence handoff independently feeds single-pass and
multi-stage RAG generation. The organizer Retrieval TSV is not a Generation
input.

## Supporting Contents

- `reports/index.html` - common entrypoint for the interactive reports.
- `reports/trec-rag-briefing-report.html` - interactive 2026 briefing report
  covering ClimbMix, sample documents, answer nuggets, and solution strategy.
- `reports/trec-rag-2025-writeups/` - standalone interactive report for the
  TREC RAG 2025 team writeups, downloaded PDFs, extracted figures, source
  manifest, and smoke test.
- `reports/notebooks/pyserini_collection_exploration.ipynb` - minimal notebook
  for querying the external Pyserini API.
- `docs/superpowers/` - design and implementation notes produced while building
  the report.
- `configs/` - checked-in experiment configurations.
- `code/` - reusable helpers for remote Pyserini access, topic loading, and the
  config-driven BM25 RAG pipeline.
- `.agents/skills/trec-rag-competition-debug-report/` - repo-local agent skill
  for privately explaining completed competition retrieval and RAG runs.
- `.agents/skills/validate-trec-rag-2026-submissions/` - repo-local validator
  for the exact organizer-facing Retrieval and RAG formats.

## Upstream Inputs

Official TREC RAG inputs are tracked as git submodules:

- `trec-rag-data/` - development and test data used by configs and reports.
- `trec-rag-skills/` - official task, Pyserini, and corpus-creation reference
  material used as source provenance. Repository-specific agent workflows live
  under `.agents/skills/` and do not depend on unpublished submodule revisions.

After cloning, initialize them with:

```bash
git submodule update --init --recursive
```

This repo includes tracked Git hooks under `.githooks/` that run
`git submodule sync --recursive` and `git submodule update --init --recursive`
after checkout and merge. Enable them once per clone:

```bash
git config core.hooksPath .githooks
```

With that setting in place, `git worktree add ...` checks out a worktree and
then initializes or updates its submodules automatically.

## Agent Workflow

This repo is intended to be usable by both Codex and Claude. Shared agent
instructions live in `AGENTS.md`; `CLAUDE.md` is a symlink to that file so the
two tools do not drift.

Before changing generated reports, run the relevant local smoke tests and, when
touching rendered UI, use Playwright screenshots at desktop and mobile widths.
The current report smoke tests are:

```bash
node reports/index.test.js
node reports/trec-rag-briefing-report.test.js
node reports/2025-promising-rag-architecture.test.js
node reports/trec-rag-2025-writeups/interactive-writeup.test.js
```

The Python helper and baseline tests are:

```bash
uv run pytest -q
```
