# Experiment Index

This file is the human entrypoint for experiment records. Detailed records live
under `reports/experiments/<experiment_id>/`; aggregate CSV indexes live beside
those folders.

## Current Recorded Experiments

| experiment_id | date | split | retriever | hits | ndcg@10 | recall@100 | record |
|---|---|---|---|---:|---:|---:|---|
| `rag25_bm25_full_dev_hits1000_v1` | 2026-07-04 | dev | Pyserini BM25 / ClimbMix | 1000 | 0.4139611685 | 0.1080879457 | `reports/experiments/rag25_bm25_full_dev_hits1000_v1/` |

## Tracking Files

- Run-level index: `reports/experiments/runs.csv`
- Topic score index: `reports/experiments/topic_scores.csv`
- Tracker convention: `reports/experiments/README.md`

The global CSVs are append-friendly indexes. The per-run folders are the source
of truth for each experiment's config, metrics, topic scores, and notes.
