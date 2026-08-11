# NP Labs · TREC RAG 2026

This repository contains the NP Labs submission for TREC RAG 2026.

## [View the project artifact hub](https://npatta01.github.io/trec_rag_2026/)

The published hub is the best place to start. It presents the final system
architecture, accepted submission records, the 2026 Retrieval quality report,
and both 2026 RAGDoll evaluations in one reader-friendly site.

## Repository references

- [Submission ledger](submissions/trec-rag-2026/SUBMISSION_LEDGER.md) — exact
  accepted files, hashes, priorities, and Evalbase confirmations.
- [Architecture report](reports/2026-competition-architecture.html) — rendered
  system walkthrough; its [Quarto source](reports/2026-competition-architecture.qmd)
  is canonical.
- [Submission validation skill](.agents/skills/validate-trec-rag-2026-submissions/SKILL.md)
  — combined Retrieval format and organizer AutoJudge preflight.
- [Implementation reference](code/trec_rag/README.md) — pipeline behavior,
  commands, reproduction boundaries, and evaluation workflow.

The completed architecture starts from one authenticated source-retrieval
artifact. Cache-first replay produces the Retrieval submissions, while a sealed
selected-evidence handoff independently feeds the single-pass and bounded
multi-stage RAG systems. The organizer-facing Retrieval TSV is not used as
Generation input.

## Developer notes

### Repository layout

- `submissions/trec-rag-2026/` — immutable accepted organizer artifacts and
  their provenance records.
- `reports/` — the public artifact hub and supporting source-backed reports.
- `code/` — reusable evaluation, retrieval, generation, and validation code.
- `configs/` — checked-in experiment and reproduction configurations.
- `docs/superpowers/` — design specifications and implementation plans.
- `.agents/skills/` — repository-local validation and analysis workflows.

### Clone and initialize

Official TREC RAG data and task guidance are tracked as git submodules:

```bash
git submodule update --init --recursive
```

The repository includes hooks that keep submodules synchronized after checkout
and merge. Enable them once per clone:

```bash
git config core.hooksPath .githooks
```

Set up the pinned Python environment with:

```bash
code/tools/setup_env.sh
```

### Verification

Run the complete Python suite:

```bash
.venv/bin/python -m pytest -q
```

Run the public landing and report smoke tests:

```bash
node README.test.js
node index.test.js
node reports/index.test.js
node reports/2026-competition-architecture.test.js
```

Shared instructions for Codex, Claude, and future coding agents live in
[`AGENTS.md`](AGENTS.md).
