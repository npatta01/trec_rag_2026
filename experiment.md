# Experiment history

This is the human decision ledger for merged experiment evidence. Per-experiment
records under [`reports/experiments/`](reports/experiments/) remain the source of
truth; [`runs.csv`](reports/experiments/runs.csv) is their generated machine
index. Work that is not merged is labeled separately and does not appear in the
CSV.

## Current retained configuration

**Retained:** family-balanced RRF over the authenticated all-topic candidate
union: original-narrative BM25 to depth 1,000 plus reviewed facet BM25 streams to
depth 200, all against the remote `climbmix-400b` index. The five
preregistered DUAL alternatives are **Rejected** for promotion because every one
loses known-relevant documents on at least one topic at depth 1,000.

Immediate recommendation: keep a bounded lexical-recovery lane, but put the
main effort into source-diverse RAG evidence selection from a deeper RRF pool.
Evaluate nugget coverage, sentence-level citations, and support before spending
more time on ranking refinements. A full local neural first-stage index over the
roughly 553-million-row, 400-billion-token ClimbMix collection is infeasible at
the present scale and deadline; this is a feasibility conclusion, not a completed
retrieval experiment.

Latest merged evidence: [all-topic decision report](reports/experiments/all_topic_tethered_facet_validation_v1/report.html)
and [experiment record](reports/experiments/all_topic_tethered_facet_validation_v1/README.md).

## Merged experiment chronology

Metrics below are only compared within the same task and cutoff. Reranking
nDCG@10 and candidate-recall results are not a single leaderboard.

| Date | Merged record | Status | Decision evidence |
|---|---|---|---|
| 2026-07-04 | [`rag25_bm25_full_dev_hits1000_v1`](reports/experiments/rag25_bm25_full_dev_hits1000_v1/) | Retained foundation | Original-narrative BM25 established nDCG@10 `0.4140` and Recall@100 `0.1081` over all 22 development topics. |
| 2026-07-05 | [`bm25_candidate_pool_coverage_v1`](reports/experiments/bm25_candidate_pool_coverage_v1/) | Learned | BM25 top 1,000 has high top-10 reranking headroom but captures only about 26% of graded relevance, so ordering and long-tail evidence coverage are separate problems. |
| 2026-07-05 | [`cross_encoder_doc_aggregation_mixedbread_v1`](reports/experiments/cross_encoder_doc_aggregation_mixedbread_v1/) | Retained reference | Mixedbread long-context plus bounded passage support improved mean nDCG@10 without a loss worse than `-0.1`; raw max/top-k aggregation was unsafe. |
| 2026-07-05 | [`reranker_improvement_summary_v1`](reports/experiments/reranker_improvement_summary_v1/) | Reference | Consolidated the conservative top-50 reranker formula; later reproducible raw-logit records are authoritative for this lane. |
| 2026-07-08 | [`cross_encoder_model_comparison_v1`](reports/experiments/cross_encoder_model_comparison_v1/) | Learned | Mixedbread was the strongest tested cross-encoder family for the BM25 top-50 reranking task. |
| 2026-07-08 | [`topic_level_reranker_metrics_v1`](reports/experiments/topic_level_reranker_metrics_v1/) | Superseded | Exposed four topic losses; superseded by the pinned raw-logit comparison because this record depended on ignored local scores. |
| 2026-07-10 | [`bm25_mixedbread_config_comparison_v1`](reports/experiments/bm25_mixedbread_config_comparison_v1/) | Retained reranking evidence | The reproducible raw-logit run improved nDCG@10 from `0.4140` to `0.5310`, with 18 improved and 4 degraded topics; it is a reranking result, not a first-stage recall replacement. |
| 2026-07-10 | [`bm25_mixedbread_topic_regression_postmortem_v1`](reports/experiments/bm25_mixedbread_topic_regression_postmortem_v1/) | Learned | Topics 224 and 515 remained model-signal limited; warm-cache replay required zero new model calls. |
| 2026-07-16 | [`all_topic_tethered_facet_validation_v1`](reports/experiments/all_topic_tethered_facet_validation_v1/) | Retained RRF; Rejected DUAL promotion | Primary DUAL won 20 topics and gained 347 known-relevant documents in aggregate at 1,000, but lost 7 on Topic 31 and 3 on Topic 300. Every alternative failed the zero-loss guard. |

## Tried, learned, rejected, retained

- **Tried:** plain BM25, several cross-encoder models and long-document
  aggregation rules, reviewed facet retrieval, family-balanced RRF, and five
  protected-prefix DUAL ranking arms.
- **Learned:** BM25 usually contains strong top-10 candidates, but its long-tail
  known-relevant capture is limited. Facets add useful candidates, while early
  promotion remains sensitive to stream depth, score calibration, and judgment
  coverage.
- **Rejected:** raw passage-max style aggregation as the default; all five
  all-topic DUAL arms for promotion; and an unbounded full local neural
  first-stage build at current ClimbMix scale.
- **Retained:** remote lexical candidate generation, family-balanced RRF as the
  promotion baseline, a bounded Mixedbread reranker as reference head-ordering
  evidence, and portable sealed experiment records.

## Topic 31/300 postmortem

The merged [Topic 31/300 postmortem](reports/experiments/all_topic_tethered_facet_validation_v1/postmortem.md)
reproduces primary-DUAL changes of `-7` and `-3` known-relevant documents at
depth 1,000. The offline Topic 300 facet-rank-cap-100 replay changes the delta to
`+2` while preserving the full 1,701-document permutation. That replay is
retrospective recovery evidence, not promotion evidence.

The judgment pool is incomplete. Unjudged documents remain **unknown**, not
nonrelevant, so the observed incoming/outgoing counts and facet-tail yields are
judgment-pool dependent.

## Active / unmerged

None of the following is merged, and none is indexed in `runs.csv`:

- **41f9 active worktree, untracked:** the sparse-relevance paired run stopped
  at its frozen gate. Its R1 arm improved head nDCG@10 versus the original-only
  control (`0.3011` vs `0.2262`) but missed the graded-Recall@100 guard
  (`0.0950` vs `0.0958`), so the reranker gate stayed closed.
- **41f9 active worktree, untracked:** the tethered diagnostic is labeled
  `mechanical_fail` and is not production validation.
- **41f9 branch-only:** an xQuAD fusion arm was evaluated and rejected in favor
  of RRF; a deterministic PRF scaffold exists without a merged promotion result.
- **f8e9 active worktree, untracked:** the qrels-free prompt lab completed 27
  streams (4 originals plus 23 facets) and found candidate diversity and lexical
  failure modes. With no qrels, it makes no Recall or nDCG improvement claim.

## Next actions

1. Preregister a bounded lexical recovery that protects the RRF head and tests
   the Topic 300 cap-100 mechanism without treating its retrospective `+2` as a
   held-out win.
2. Start **source-diverse RAG evidence selection** from a deeper retained RRF
   pool. Balance sources and obligations, then measure nugget coverage,
   sentence-level citation correctness, and support.
3. Keep unjudged candidates unknown, report judgment coverage beside ranking
   changes, and keep branch-only evidence out of the merged machine index.
