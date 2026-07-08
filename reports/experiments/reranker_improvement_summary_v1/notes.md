# Reranker Improvement Summary

Run date: 2026-07-05

## Technical Summary

The simplest reading is: every tested cross-encoder family improved average
`nDCG@10` over the BM25 top-50 order, but Mixedbread is the only family that
currently gives both a meaningful lift and a tolerable regression profile.

After advisor review, the current recommendation is to carry forward the
**coverage-aware long-doc aggregate**. Pure long-document scoring remains the
simplest ablation, but the coverage-aware version keeps more of the gain while
removing big topic regressions in this dev sample.

```text
score =
  0.5 * long_document_relevance
+ 0.5 * strongest_passage_relevance
+ 0.25 * bounded_coverage_support
```

It reaches `0.527544` `nDCG@10`, which is `+0.113583` over BM25
(`+27.4%` relative lift), with zero topic losses worse than `-0.1`. The
best-average formula reaches `0.559599`, but still has one big regression and
uses topic-level z normalization, so it is better treated as an upper-bound
experiment for now.

## Decision Table

This is the compact decision view. The pure long-doc score is the cleanest
simple baseline; the coverage-aware aggregate is the recommended carry-forward
score because it reduces regression risk.

| candidate | nDCG@10 | delta vs BM25 | losses | big losses | worst topic delta |
|---|---:|---:|---:|---:|---:|
| BM25 candidate order | 0.413961 | +0.000000 | - | - | - |
| Pure long-doc score | 0.515422 | +0.101461 | 6 | 1 | 224: -0.1008 |
| Coverage-aware long-doc aggregate | 0.527544 | +0.113583 | 4 | 0 | 515: -0.0948 |
| Mixedbread prefix cross-encoder | 0.532224 | +0.118263 | 5 | 2 | 224: -0.2222 |
| Best-average z-normalized aggregate | 0.559599 | +0.145638 | 2 | 1 | 224: -0.1241 |

## Why Not Pure Long-Doc

Pure long-document scoring is attractive because it is one score from one model
input:

```text
score = doc_max_32768_buf512_score
```

It already improves BM25 by `+0.101461` `nDCG@10`, so it is a reasonable
simplicity baseline. The reason not to use it as the main recommendation is that
it has a slightly weaker average score and leaves one topic loss below `-0.1`.
The coverage-aware aggregate adds one bounded support term and removes that big
loss in this sample.

## Formula In Plain English

The formula combines whole-document relevance, best-passage relevance, and
bounded evidence coverage. The first two terms ask whether the document is
relevant overall and whether it contains very strong matching evidence. The
final capped support term rewards documents where relevance appears in multiple
distinct spans, without giving an unbounded advantage to long documents.

The implementation formula is:

```text
score =
  0.5 * doc_max_32768_buf512_score
+ 0.5 * top4_weighted_window_score
+ 0.25 * min(relative_span_support_1.0, 6)
```

Plain names map to implementation names as follows:

| plain name | implementation | meaning |
|---|---|---|
| `long_document_relevance` | `doc_max_32768_buf512_score` | Cross-encoder score for the full document when it fits, otherwise the longest safe prefix. |
| `strongest_passage_relevance` | `top4_weighted_window_score` | Weighted summary of the strongest matching windows in the document. |
| `bounded_coverage_support` | `min(relative_span_support_1.0, 6)` | Small capped bonus when multiple distinct spans are close to the best span score. |

## Improvement Over BM25

All rows use the same BM25 top-50 candidate set over 22 development topics.
`losses` counts topics below BM25. `big losses` counts topic deltas below
`-0.1`.

| candidate | nDCG@10 | delta vs BM25 | relative lift | losses | big losses | worst topic delta |
|---|---:|---:|---:|---:|---:|---:|
| BM25 candidate order | 0.413961 | +0.000000 | +0.0% | - | - | - |
| Qwen 4B chunk max | 0.448550 | +0.034589 | +8.4% | 8 | 5 | 37: -0.3791 |
| Qwen 0.6B chunk top3 weighted | 0.458308 | +0.044347 | +10.7% | 8 | 4 | 224: -0.2978 |
| Qwen 0.6B full-doc probability fallback | 0.482041 | +0.068080 | +16.4% | 6 | 5 | 37: -0.2043 |
| Qwen 0.6B hybrid full75/chunk25 margin | 0.489262 | +0.075301 | +18.2% | 6 | 4 | 224: -0.2151 |
| BGE reranker v2 m3 | 0.490388 | +0.076427 | +18.5% | 6 | 3 | 224: -0.1859 |
| MiniLM sequence classifier | 0.499056 | +0.085095 | +20.6% | 6 | 2 | 31: -0.2814 |
| Mixedbread long-context document score | 0.515422 | +0.101461 | +24.5% | 6 | 1 | 224: -0.1008 |
| Mixedbread conservative adjusted formula | 0.527544 | +0.113583 | +27.4% | 4 | 0 | 515: -0.0948 |
| Mixedbread base v2 prefix | 0.532224 | +0.118263 | +28.6% | 5 | 2 | 224: -0.2222 |
| Mixedbread best-average adjusted formula | 0.559599 | +0.145638 | +35.2% | 2 | 1 | 224: -0.1241 |

## What To Carry Forward

The production-facing candidate should be the coverage-aware long-doc aggregate,
not the pure prefix scorer and not the best-average z-normalized formula. It
gives up some average score versus the best-average formula, but it removes
large topic regressions in this dev sample.

The runnable pipeline config for this setup is
`configs/rag25_bm25_mixedbread_rerank_v1.yaml`. It retrieves BM25 top-50
candidates and applies cached coverage-aware Mixedbread reranker scores.

The topic-level companion report tracks the same decision with additional
`precision@10`, `recall@10`, `hit_rate@10`, `relevant_count@10`, and candidate
pool `recall@50` metrics. Its main regression finding is topic `515`: BM25 has
one relevant document in the top 10, while the coverage-aware reranker moves all
three relevant top-50 candidates below rank 10. That is an ordering regression,
not a retrieval-absence issue.

Qwen should not be the default reranker based on these runs. The Qwen runs do
improve average `nDCG@10` over BM25, but the gains are smaller and the topic
regressions are larger. The available 4B chunk run did not outperform the 0.6B
variants. There is no Qwen 8B result artifact in this workspace, so this summary
does not make a claim about 8B.

The follow-up should stay separate: query decomposition, facet coverage, and
alternate-query recall can be evaluated next without changing the current
reranker summary. For this checkpoint, the useful claim is just that the adjusted
Mixedbread document/window aggregate is the best tested score to carry forward.

## Caveats And Guardrails

This is a 22-topic dev-set result, and the formula was selected partly for its
topic-level regression profile. It should be frozen before blind/test use. A
leave-one-topic-out or other held-out sanity check would make the selection more
defensible before treating it as final.

The `0.559599` z-normalized formula is not the production recommendation.
Topic-level z normalization is useful diagnostically, but it adds calibration
complexity and still has one big topic regression.

Do not report only average `nDCG@10`. Keep `losses`, `big losses`, and worst
topic delta beside the average score so regressions remain visible.

## Method Notes

`doc_max_32768_buf512_score` keeps the narrative, reserves 512 tokens for
pair-formatting safety, and then uses as much of the document as fits under the
32,768-token model context. In the BM25 top-50 set, 1,079 of 1,100
narrative/document pairs fit fully and 21 were truncated to the longest safe
prefix.

`top4_weighted_window_score` is:

```text
0.55*s1 + 0.25*s2 + 0.13*s3 + 0.07*s4
```

where `s1 >= s2 >= s3 >= s4` are the top window scores for the document.

`relative_span_support_1.0` counts span-distinct chunks with score at least
`best_chunk_score - 1.0`, capped at 6. A counted chunk must add at least 800
previously uncovered characters, which reduces overlap double-counting.
