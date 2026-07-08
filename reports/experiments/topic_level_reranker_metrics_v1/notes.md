# Topic-Level Reranker Metrics

Run date: 2026-07-08

## Summary

This report tracks topic-level BM25 versus coverage-aware reranker metrics so aggregate nDCG improvements do not hide regressions.

| method | nDCG@10 | precision@10 | recall@10 | hit rate@10 | relevant count@10 | graded recall@10 | recall@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 top-50 order | 0.413961 | 0.704545 | 0.012224 | 1.000000 | 7.045455 | 0.011415 | 0.054849 |
| Coverage-aware reranker | 0.527544 | 0.781818 | 0.014132 | 0.954545 | 7.818182 | 0.013575 | 0.054849 |
| Delta | +0.113583 | +0.077273 | +0.001908 | -0.045455 | +0.772727 | +0.002160 | +0.000000 |

## Regression Counts

- `nDCG@10` gains: 18 topics; losses: 4 topics; big losses below `-0.1`: 0 topics.
- `precision@10` gains: 11 topics; losses: 4 topics.
- `recall@10` gains: 11 topics; losses: 4 topics.
- `hit_rate@10` losses: 1 topic.

Topic `515` is the clearest remaining failure mode: BM25 has one relevant
document in the top 10, while the reranker has zero. Since both methods have the
same three relevant documents available in the top-50 candidate pool, this is a
reranker ordering regression rather than a retrieval miss.

## How To Read The Metrics

- `nDCG@10` is the main graded ranking metric. Use the delta column to find reranker regressions.
- `precision@10` answers: of the top 10, how many are qrel-relevant?
- `recall@10` answers: how much of the topic's judged relevant set reached the top 10?
- `hit_rate@10` answers: did the top 10 contain at least one relevant document?
- `relevant_count@10` is the literal number of unique relevant docs in the top 10.
- `graded_recall@10` is useful when moving a grade-4 document matters more than moving a grade-2 document.
- `recall@50` is candidate-pool context. It should not change after reranking the same top-50 pool.

## Worst nDCG@10 Topic Deltas

| topic | BM25 nDCG@10 | reranker nDCG@10 | delta | BM25 P@10 | reranker P@10 | BM25 recall@10 | reranker recall@10 | relevant@10 delta | recall@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 515 | 0.229925 | 0.135110 | -0.094816 | 0.100000 | 0.000000 | 0.002415 | 0.000000 | -1 | 0.007246 |
| 224 | 0.816497 | 0.725531 | -0.090967 | 1.000000 | 1.000000 | 0.015314 | 0.015314 | +0 | 0.070444 |
| 897 | 0.304874 | 0.282596 | -0.022278 | 1.000000 | 0.900000 | 0.014837 | 0.013353 | -1 | 0.057864 |
| 31 | 0.847805 | 0.839708 | -0.008098 | 1.000000 | 1.000000 | 0.010811 | 0.010811 | +0 | 0.054054 |
| 161 | 0.666326 | 0.676766 | +0.010441 | 0.800000 | 0.900000 | 0.011645 | 0.013100 | +1 | 0.062591 |
| 499 | 0.681611 | 0.719831 | +0.038220 | 1.000000 | 0.900000 | 0.016420 | 0.014778 | -1 | 0.068966 |
| 14 | 0.355333 | 0.409446 | +0.054113 | 0.700000 | 1.000000 | 0.010072 | 0.014388 | +3 | 0.061871 |
| 225 | 0.257264 | 0.316527 | +0.059263 | 0.900000 | 0.900000 | 0.016216 | 0.016216 | +0 | 0.068468 |

## All Topic Deltas

| topic | nDCG@10 delta | precision@10 delta | recall@10 delta | hit@10 delta | relevant@10 delta | recall@50 | candidate relevant@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 515 | -0.094816 | -0.100000 | -0.002415 | -1 | -1 | 0.007246 | 3 |
| 224 | -0.090967 | +0.000000 | +0.000000 | +0 | +0 | 0.070444 | 46 |
| 897 | -0.022278 | -0.100000 | -0.001484 | +0 | -1 | 0.057864 | 39 |
| 31 | -0.008098 | +0.000000 | +0.000000 | +0 | +0 | 0.054054 | 50 |
| 161 | +0.010441 | +0.100000 | +0.001456 | +0 | +1 | 0.062591 | 43 |
| 499 | +0.038220 | -0.100000 | -0.001642 | +0 | -1 | 0.068966 | 42 |
| 14 | +0.054113 | +0.300000 | +0.004317 | +0 | +3 | 0.061871 | 43 |
| 225 | +0.059263 | +0.000000 | +0.000000 | +0 | +0 | 0.068468 | 38 |
| 273 | +0.076354 | +0.100000 | +0.003030 | +0 | +1 | 0.069697 | 23 |
| 219 | +0.076786 | +0.100000 | +0.001724 | +0 | +1 | 0.055172 | 32 |
| 37 | +0.078957 | +0.200000 | +0.002418 | +0 | +2 | 0.053204 | 44 |
| 300 | +0.097510 | -0.300000 | -0.004724 | +0 | -3 | 0.058268 | 37 |
| 477 | +0.104307 | +0.200000 | +0.003810 | +0 | +2 | 0.059048 | 31 |
| 144 | +0.109668 | +0.300000 | +0.007126 | +0 | +3 | 0.011876 | 5 |
| 707 | +0.145628 | +0.000000 | +0.000000 | +0 | +0 | 0.044369 | 26 |
| 233 | +0.157887 | +0.100000 | +0.001271 | +0 | +1 | 0.050826 | 40 |
| 200 | +0.199446 | +0.600000 | +0.012146 | +0 | +6 | 0.030364 | 15 |
| 407 | +0.220176 | +0.100000 | +0.003378 | +0 | +1 | 0.054054 | 16 |
| 84 | +0.236927 | +0.000000 | +0.000000 | +0 | +0 | 0.065621 | 46 |
| 58 | +0.322406 | +0.000000 | +0.000000 | +0 | +0 | 0.089147 | 46 |
| 72 | +0.351219 | +0.000000 | +0.000000 | +0 | +0 | 0.049945 | 45 |
| 213 | +0.375680 | +0.200000 | +0.011561 | +0 | +2 | 0.063584 | 11 |
