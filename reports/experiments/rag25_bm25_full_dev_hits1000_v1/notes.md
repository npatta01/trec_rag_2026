# RAG25 BM25 Full Dev Hits 1000 V1

Run date: 2026-07-04

## Summary

This experiment records the config-driven BM25 retrieval baseline over the 22
TREC RAG development topics. It uses the hosted Pyserini ClimbMix index, the
original topic narrative as the query, passthrough ranking, and projected
development qrels for diagnostics.

| metric | value |
|---|---:|
| ndcg@10 | 0.4139611685 |
| recall@100 | 0.1080879457 |

## Approach

The baseline treats retrieval as the first reusable stage of the RAG system. For
each development topic, the pipeline uses the original topic narrative as a
single BM25 query against the hosted Pyserini ClimbMix index. The retriever
returns the top 1000 candidates with raw document text, writes the raw response
into a shared cache keyed by query, index, endpoint, and hit count, and then
normalizes those candidates into inspectable JSONL stage records.

No reranking or query decomposition is applied in this experiment. The ranking
stage preserves BM25 order, deduplicates by `docid`, writes a TREC runfile,
selects the top text-bearing evidence for placeholder RAG output, and evaluates
the retrieval order with projected dev qrels. This is a retrieval and
candidate-pool baseline rather than a full answer-quality system.

## Artifacts

- Manifest: `manifest.yaml`
- Exact runtime config: `config.yaml`
- Metrics JSON: `metrics.json`
- Per-topic scores: `topic_scores.csv`
- Local run output: `tmp/hits1000-full-dev/output`
- Local raw cache:
  `/home/nidhin/projects/trec_rag/outputs/cache_demo_two_topics_hits1000/cache/`

The raw cache stores the returned Pyserini payload. Candidate document text is
available at `.response.candidates[].doc`.

## Observations

`hits: 1000` gives a useful candidate pool, but plain BM25 does not reliably put
the best judged documents in the top 10 for broad, multi-intent narratives.
Topic `31` scores well because its top results are mostly grade-4 and grade-3
documents. Topics such as `144`, `200`, `515`, and `707` score lower because
BM25 retrieves generic or distracting lexical matches near the top.

Next likely improvements are reranking and query decomposition.
