# TREC RAG Workspace

This repository collects TREC RAG research notes, reports, and reusable
implementation patterns.

## Current Contents

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

## Upstream Inputs

Official TREC RAG inputs are tracked as git submodules:

- `trec-rag-data/` - development and test data used by configs and reports.
- `trec-rag-skills/` - official task, Pyserini, and corpus-creation reference
  material used as source provenance.

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
