# BM25 Pyserini Baseline Experiment

Run date: 2026-07-04

This experiment records the config-driven BM25 retrieval baseline over the 22
TREC RAG development topics. It uses the remote Pyserini ClimbMix index,
original topic narrative text as the query, passthrough ranking, and projected
development qrels for diagnostics.

## Summary

- Topics: 22 RAG development topics from `rag25-topics-dev.tsv`
- Retriever: remote Pyserini BM25 over `climbmix-400b`
- Query: original topic narrative/prompt text
- Hits per topic: 1000
- Ranking: passthrough BM25 rank with docid dedupe
- Evaluation: projected dev qrels, `ndcg@10` and `recall@100`
- Overall `ndcg@10`: 0.4139611685
- Overall `recall@100`: 0.1080879457

The run was intentionally gentle against the hosted endpoint: cached topics were
reused, uncached requests were issued sequentially, and retries used a longer
timeout with backoff after one topic hit the default 30 second timeout.

## Approach

The baseline treats retrieval as the first reusable stage of the RAG system. For
each development topic, the pipeline uses the original topic narrative as a
single BM25 query against the hosted Pyserini ClimbMix index. The retriever
returns the top `1000` candidates with raw document text, writes the raw
response into a shared cache keyed by query, index, endpoint, and hit count, and
then normalizes those candidates into inspectable JSONL stage records.

No reranking or query decomposition is applied in this experiment. The ranking
stage simply preserves the BM25 order, deduplicates by `docid`, writes a TREC
runfile, selects the top text-bearing evidence for placeholder RAG output, and
evaluates the retrieval order with projected dev qrels. This makes the run a
candidate-pool and retrieval-quality baseline rather than a full answer-quality
system.

## Runtime Config

This is the exact scratch config used for the full 22-topic `hits: 1000` run.
The checked-in baseline config currently keeps `hits: 100`; this experiment
uses `hits: 1000` to inspect a larger candidate pool.

```yaml
experiment:
  id: cache_demo_two_topics_hits1000
  output_dir: tmp/hits1000-full-dev/output

submission:
  team_id: local-baseline

topics:
  path: trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv
  format: tsv

query_understanding:
  variants:
    - name: original
      type: original_topic

retrievers:
  - name: climbmix_bm25
    type: pyserini_remote
    query_variants: [original]
    hits: 1000
    index: climbmix-400b
    cache: true

ranking:
  type: passthrough
  dedupe:
    by: docid
    keep: best_rank
    preserve_provenance: true

evidence:
  type: top_k
  k: 5
  require_text: true
  allow_fewer: true

generation:
  type: placeholder

evaluation:
  kind: dev_projected_qrels
  qrels: trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels
  metrics: [ndcg@10, recall@100]
  relevance_threshold: 2
```

Note: the experiment id was reused from the initial two-topic `hits: 1000`
smoke run so topics `14` and `31` could reuse the raw response cache. A cleaner
future run id would be something like `rag25_bm25_full_dev_hits1000_v1`.

## Outputs

- Runfile:
  `tmp/hits1000-full-dev/output/r_output_trec_rag_2026.tsv`
- Retrieved candidates:
  `tmp/hits1000-full-dev/output/stage_retrieved.jsonl`
- Placeholder RAG output:
  `tmp/hits1000-full-dev/output/rag_output_trec_rag_2026.jsonl`
- Metrics:
  `tmp/hits1000-full-dev/output/retrieval_metrics.json`
- Run-level tracker:
  `reports/experiments/runs.csv`
- Topic score tracker:
  `reports/experiments/topic_scores.csv`
- Raw Pyserini cache:
  `/home/nidhin/projects/trec_rag/outputs/cache_demo_two_topics_hits1000/cache/`

Output sizes and counts:

| artifact | count / size |
|---|---:|
| runfile rows | 22,000 |
| retrieved-stage rows | 22,000 |
| RAG output rows | 22 |
| raw cache files | 22 |
| raw cache size | 665M |
| scratch output dir size | 1.4G |

The raw cache stores the returned Pyserini payload. Candidate document text is
available at `.response.candidates[].doc`.

## Metrics

Aggregate metrics:

| metric | value |
|---|---:|
| ndcg@10 | 0.4139611685 |
| recall@100 | 0.1080879457 |

Per-topic metrics:

The same values are also stored in CSV form at
`reports/experiments/topic_scores.csv` with one row per experiment/topic plus
an `overall` row. Run-level metadata and aggregate metrics are stored in
`reports/experiments/runs.csv`, so future experiment variants can be compared
by topic id without creating a new schema each time.

| topic | ndcg@10 | recall@100 |
|---|---:|---:|
| 14 | 0.3553 | 0.1223 |
| 31 | 0.8478 | 0.1059 |
| 37 | 0.6390 | 0.1052 |
| 58 | 0.5008 | 0.1822 |
| 72 | 0.3237 | 0.1021 |
| 84 | 0.4918 | 0.1141 |
| 144 | 0.0887 | 0.0333 |
| 161 | 0.6663 | 0.1266 |
| 200 | 0.1401 | 0.0587 |
| 213 | 0.1122 | 0.0983 |
| 219 | 0.4934 | 0.1000 |
| 224 | 0.8165 | 0.1348 |
| 225 | 0.2573 | 0.1423 |
| 233 | 0.5438 | 0.0864 |
| 273 | 0.3223 | 0.1515 |
| 300 | 0.3546 | 0.1165 |
| 407 | 0.2786 | 0.1250 |
| 477 | 0.4559 | 0.1010 |
| 499 | 0.6816 | 0.1396 |
| 515 | 0.2299 | 0.0217 |
| 707 | 0.2027 | 0.0887 |
| 897 | 0.3049 | 0.1217 |

Best `ndcg@10` topics:

| topic | ndcg@10 |
|---|---:|
| 31 | 0.8478 |
| 224 | 0.8165 |
| 499 | 0.6816 |
| 161 | 0.6663 |
| 37 | 0.6390 |

Lowest `ndcg@10` topics:

| topic | ndcg@10 |
|---|---:|
| 144 | 0.0887 |
| 213 | 0.1122 |
| 200 | 0.1401 |
| 707 | 0.2027 |
| 515 | 0.2299 |

## Interpretation

The low `ndcg@10` scores do not necessarily mean the collection lacks relevant
documents. Several low-scoring topics have many judged relevant documents, but
plain BM25 does not place the highest-grade documents in the top 10.

Examples:

- Topic `31` has high `ndcg@10` because the top 10 are mostly grade-4 and
  grade-3 documents.
- Topic `144` has low `ndcg@10` because the top 10 are mostly grade-1 or
  grade-0 documents; its first grade-3 document appears at rank 14, and no
  grade-4 document appears in the top 100.
- Topic `200` is hurt by a distracting query facet: the narrative mentions
  "Sodom and Gomorrah", and BM25 retrieves some documents about that phrase
  instead of Holocaust-focused documents.
- Topics such as `707` and `515` are broad, multi-intent health queries, so
  BM25 tends to retrieve generic topical material rather than the best
  answer-supporting documents.

The main takeaway is that `hits: 1000` gives a useful candidate pool, but the
baseline needs reranking and/or query decomposition to improve top-10 ranking
quality. For a clean first submission-style BM25 baseline, `hits: 100` remains a
reasonable default because the current diagnostic metric includes `recall@100`;
`hits: 1000` is most useful once a reranker consumes the larger pool.

## Reproduce

From the baseline worktree:

```bash
PYTHONPATH=code uv run --with pyyaml python -m trec_rag.pipeline \
  --config tmp/hits1000-full-dev/config.yaml
```

Because `cache: true` is enabled, rerunning this config should reuse raw
Pyserini responses already stored under:

```text
/home/nidhin/projects/trec_rag/outputs/cache_demo_two_topics_hits1000/cache/
```
