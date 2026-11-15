# NP Labs · TREC RAG 2026

NP Labs submission for TREC RAG 2026: bounded retrieval over the ClimbMix
corpus, cited answer generation from one authenticated evidence source, and the
organizer evaluation that followed.

## [View the project site](https://npatta01.github.io/trec_rag_2026/)

[Paper (PDF)](output/pdf/2026-trec-rag-paper.pdf) ·
[Conference talk](https://npatta01.github.io/trec_rag_2026/reports/2026-trec-rag-talk.html) ·
[Approach video](https://npatta01.github.io/trec_rag_2026/reports/2026-trec-rag-approach.mp4) ·
[Architecture walkthrough](https://npatta01.github.io/trec_rag_2026/reports/2026-competition-architecture.html) ·
[Organizer scores](https://npatta01.github.io/trec_rag_2026/reports/2026-organizer-score-explorer.html)

## The problem

TREC RAG 2026 asks systems to answer complex information needs using the
ClimbMix corpus. A useful system has to retrieve trustworthy evidence for every
part of an official narrative, turn that evidence into a clear answer, and keep
each factual claim traceable to an allowed source. Our submission addresses both
sides: variable-depth Retrieval rankings, and cited RAG answers derived from one
authenticated evidence source.

[![The NP Labs TREC RAG 2026 system: bounded source retrieval creates one authenticated source, which branches into three Retrieval submissions and two evidence-grounded RAG submissions.](reports/2026-competition-architecture/01-whole-system.svg)](https://npatta01.github.io/trec_rag_2026/reports/2026-competition-architecture.html)

*The guided walkthrough of this diagram is in the [published artifact hub](https://npatta01.github.io/trec_rag_2026/).*

## Project at a glance

| Verified scope | Count |
| --- | ---: |
| Official narratives | 119 |
| Retrieval runs | 3 |
| RAG runs | 2 |
| Evalbase-accepted organizer files | 5 |

### Reports for reviewers

- [Architecture Report](https://npatta01.github.io/trec_rag_2026/reports/2026-competition-architecture.html)
  — the final system from bounded source retrieval through all five accepted files.
- [Retrieval Quality Report](https://npatta01.github.io/trec_rag_2026/reports/2026-retrieval-nugget-coverage.html)
  — evidence coverage across the 119 official narratives.
- [RAGDoll Evaluation — Single-pass RAG](https://npatta01.github.io/trec_rag_2026/reports/2026-ragdoll-rag26-ss1.html)
  — citation-support analysis for `rag26-ss1`.
- [RAGDoll Evaluation — Multi-stage RAG](https://npatta01.github.io/trec_rag_2026/reports/2026-ragdoll-rag26-ms1-final.html)
  — citation-support analysis for `rag26-ms1-final`.

### Repository references

- [Submission ledger](submissions/trec-rag-2026/SUBMISSION_LEDGER.md) — accepted
  files, hashes, priorities, and Evalbase confirmations.
- [Architecture source](reports/2026-competition-architecture.qmd) — canonical
  Quarto source for the rendered walkthrough.
- [Submission validator](.agents/skills/validate-trec-rag-2026-submissions/SKILL.md)
  — Retrieval format and organizer AutoJudge preflight.
- [Implementation reference](code/trec_rag/README.md) — pipeline behavior,
  commands, and reproduction boundaries.
- [Run archive](docs/run-archive.md) — the Hugging Face backup, the Luna planner
  receipts, and the per-topic trace CSV.

## Developer notes

### Repository layout

- `submissions/trec-rag-2026/` — immutable accepted organizer artifacts, their
  provenance records, and the organizer-released scores.
- `reports/` — the public artifact hub and supporting source-backed reports.
- `code/` — reusable evaluation, retrieval, generation, and validation code.
- `configs/` — checked-in experiment and reproduction configurations.
- `docs/superpowers/` — design specifications and implementation plans.
- `.agents/skills/` — repository-local validation and analysis workflows.

### Clone and initialize

Official TREC RAG data and task guidance are tracked as git submodules. The
repository ships hooks that keep them synchronized after checkout and merge.

```bash
git submodule update --init --recursive
git config core.hooksPath .githooks
code/tools/setup_env.sh
```

### Verification

```bash
.venv/bin/python -m pytest -q
```

```bash
node README.test.js && node index.test.js && node reports/index.test.js
```

Shared instructions for Codex, Claude, and future coding agents live in
[`AGENTS.md`](AGENTS.md).
