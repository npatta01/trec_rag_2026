# TREC RAG Workspace

This repository collects TREC RAG research notes, reports, and reusable
implementation patterns.

## Current Contents

- `reports/trec-rag-briefing-report.html` - earlier standalone briefing report.
- `reports/trec-rag-2025-writeups/` - standalone interactive report for the
  TREC RAG 2025 team writeups, downloaded PDFs, extracted figures, source
  manifest, and smoke test.
- `reports/notebooks/pyserini_collection_exploration.ipynb` - minimal notebook
  for querying the external Pyserini API.
- `docs/superpowers/` - design and implementation notes produced while building
  the report.
- `code/` - placeholder for reusable code that should be shared across reports,
  experiments, and future agents.

## Agent Workflow

This repo is intended to be usable by both Codex and Claude. Shared agent
instructions live in `AGENTS.md`; `CLAUDE.md` is a symlink to that file so the
two tools do not drift.

Before changing generated reports, run the relevant local smoke tests and, when
touching rendered UI, use Playwright screenshots at desktop and mobile widths.
