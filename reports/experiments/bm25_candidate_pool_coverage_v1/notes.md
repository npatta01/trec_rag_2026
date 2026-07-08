# BM25 Candidate Pool Coverage

Run date: 2026-07-05

## Summary

The two added retrieval diagnostics separate candidate-pool coverage from
reranking quality:

- `graded_recall@k`: sum of qrel grades found in the retrieved top `k`, divided
  by the total graded qrel mass for the topic.
- `ideal_dcg_coverage@k`: ideal DCG using only retrieved top-`k` candidates,
  divided by the true ideal DCG at `k`.

For reranker ceiling analysis, this report also computes
`oracle_ndcg@10_from_topk`: the best possible nDCG@10 if a perfect reranker could
reorder the BM25 top-`k` candidates.

## Aggregate Results

| depth | binary recall | graded recall | ideal DCG coverage | oracle nDCG@10 from top-k |
|---:|---:|---:|---:|---:|
| 10 | 0.012224 | 0.011415 | 0.493459 | 0.493459 |
| 50 | 0.054849 | 0.053164 | 0.494061 | 0.818719 |
| 100 | 0.108088 | 0.105109 | 0.520526 | 0.917529 |
| 1000 | 0.266354 | 0.258405 | 0.397769 | 0.963148 |

## Completeness Counts

Use this table for the literal question: did BM25 retrieve all judged relevant
documents? Here, relevant means qrel grade `>=2`, matching the experiment
configuration. These counts are only for judged qrel documents; they cannot
prove whether unjudged documents are truly relevant.

The recall values below are pooled counts across all topics. The aggregate
recall values above are mean per-topic recalls, so they differ slightly.

| depth | found relevant | total relevant | missed relevant | pooled recall | found grade mass | total grade mass | pooled graded recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 721 | 12984 | 12263 | 0.055530 | 2331 | 43521 | 0.053560 |
| 100 | 1418 | 12984 | 11566 | 0.109211 | 4610 | 43521 | 0.105926 |
| 1000 | 3493 | 12984 | 9491 | 0.269023 | 11344 | 43521 | 0.260656 |

Full per-topic completeness counts are tracked in `topic_completeness.json`.

## Interpretation

BM25 top-50 has low total relevance coverage, but it already has substantial
nDCG@10 headroom: a perfect reranker over top-50 candidates could average
`0.818719`. BM25 top-1000 raises the oracle nDCG@10 ceiling to `0.963148`, so
retrieval is usually finding enough high-grade documents somewhere in the pool.

That means the current top-50 reranker problem is mostly ordering, not complete
candidate absence. However, the low graded recall still matters for answer
generation and long-tail evidence coverage; alternate queries or facet queries
should improve robustness even if nDCG@10 can already be high from BM25 top-50.

## Weak Topics

At depth 50, the weakest oracle nDCG@10 topics are `144`, `515`, `225`, `213`,
and `707`. These are the topics where top-50 retrieval itself leaves the reranker
with a constrained candidate pool.

At depth 1000, topic `144` remains the main retrieval bottleneck, with oracle
nDCG@10 only `0.584049`. Topic `707` also remains constrained at `0.761121`.

## How To Read This

Use `recall@k` or `graded_recall@k` to answer: did retrieval find the relevant
documents at all?

Example: topic `144` has 421 judged relevant documents at grade `>=2`. BM25
top-1000 found only 47 of them, missing 374. Its first grade-4 document appears
at BM25 rank 257. That is a retrieval bottleneck, not just a reranking problem.

Use `oracle_ndcg@10_from_topk` minus actual BM25 nDCG@10 to answer: were the
relevant documents found but ranked too low?

Example: topic `200` has BM25 nDCG@10 of `0.140084`, but
`oracle_ndcg@10_from_top1000` is `1.000000`. So BM25 did retrieve enough
high-grade documents somewhere in top-1000, but ranked them badly. Its median
retrieved relevant-document rank is 114.5.

Useful reading pattern:

| question | metric or count |
|---|---|
| Did we retrieve all relevant docs? | `found_rel@k / total_rel`, or `recall@k` |
| Did we retrieve most relevance mass? | `graded_recall@k` |
| Could a perfect reranker fix top-10? | `oracle_ndcg@10_from_topk` |
| Are docs present but ranked low? | high oracle nDCG, low actual nDCG, high median relevant rank |
| Is retrieval itself weak? | low oracle nDCG even at large `k` |
