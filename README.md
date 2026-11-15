# NP Labs · TREC RAG 2026

Our submission to NIST's TREC RAG 2026 track. Every team answers the same 119
information needs over one fixed corpus, in at most 1,024 words, naming a source
for every statement. Five of our files were accepted by Evalbase: three Retrieval
runs and two Retrieval-Augmented Generation runs.

[![Watch the approach: one request split into eight angles, each becoming two BM25 queries, for sixteen searches in all.](reports/2026-trec-rag-approach-still.jpg)](https://npatta01.github.io/trec_rag_2026/#approach)

**[▶ Watch the approach](https://npatta01.github.io/trec_rag_2026/#approach)** —
87 seconds, with sound: what the track asks, then one real request all the way
through the pipeline.

## [Project site →](https://npatta01.github.io/trec_rag_2026/)

[Paper (PDF)](output/pdf/2026-trec-rag-paper.pdf) ·
[Conference talk](https://npatta01.github.io/trec_rag_2026/reports/2026-trec-rag-talk.html)

## How the system works

[![One official narrative becomes one authenticated frozen source, facet-deepseek-b40-v3, which branches into three Retrieval TSVs and two RAG JSONLs — the five files Evalbase accepted.](reports/2026-competition-architecture/00-system-at-a-glance.svg)](https://npatta01.github.io/trec_rag_2026/reports/2026-competition-architecture.html)

One planning call splits a narrative into up to eight angles, and each angle
becomes one to three BM25 queries over ClimbMix — at most 25 searches per topic.
What comes back is chunked, reranked, de-duplicated, and sealed into a single
authenticated evidence source. Everything after that derives from that one
source: three Retrieval rankings ordered from the same candidate core, and two
sets of cited answers that never re-open the index. The planning is one-shot,
not a search loop.

The [architecture walkthrough](https://npatta01.github.io/trec_rag_2026/reports/2026-competition-architecture.html)
explains the diagram stage by stage, with the exact limits at each step.

## What is in the repo

| Path | What it holds |
| --- | --- |
| [`submissions/trec-rag-2026/`](submissions/trec-rag-2026/SUBMISSION_LEDGER.md) | the five accepted files, their hashes, priorities, and Evalbase confirmations |
| [`code/trec_rag/`](code/trec_rag/README.md) | the retrieval and generation runners, and how to reproduce them |
| [`reports/`](reports/index.html) | the published reports and their canonical sources |
| [`configs/`](configs/) | the checked-in run configurations |
| [`docs/`](docs/) | design notes, agent process records, and the [run archive](docs/run-archive.md) |

## Checks

Official TREC RAG data and task guidance are tracked as git submodules; the
repository ships hooks that keep them synchronized after checkout and merge.

```bash
git submodule update --init --recursive
git config core.hooksPath .githooks
code/tools/setup_env.sh
```

```bash
.venv/bin/python -m pytest -q
```

```bash
git ls-files '*.test.js' | xargs -n1 node
```

Shared instructions for Codex, Claude, and future coding agents live in
[`AGENTS.md`](AGENTS.md).
