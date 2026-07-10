# BM25 to Mixedbread Topic Regression Postmortem

## Technical summary

The shipped top-50 reranker changes mean development nDCG@10 from **0.413961** to **0.531008** (**+0.117047**), but regresses **4** of **22** topics. Within the fixed candidate pools, **2** losses are aggregation-sensitive and **2** are model-signal-limited.

The losses beyond -0.10 nDCG@10 are **224 and 515**. Their controlled diagnostics are 224 (model signal limited), 515 (model signal limited). Coverage improves the no-coverage blend for 224. Coverage worsens the no-coverage blend for 515. These are ranking counterfactuals over saved scores, not causal claims.

This version was regenerated from a completed Modal scoring run on `NVIDIA A100-SXM4-80GB` (Modal / CLOUD_PROVIDER_AWS / us-east-1).

Completed warm-cache verification rematerialized scores for **22,000 documents** and **227,156 document windows** from the persistent Modal schema-v2 cache with both models unnecessary and **zero model calls**. The cache remained unchanged and both regenerated artifacts were semantically equal to their canonical artifacts.

The diagnostic labels are controlled descriptions of these saved scores, not causal or held-out generalization claims. The report must be regenerated when the score artifacts or runtime change.

## 2 losses are aggregation-sensitive; 2 are model-signal-limited

For an aggregation-sensitive loss, at least one component/no-coverage counterfactual reaches BM25 while the shipped formula does not. For a model-signal-limited loss, none does. This table isolates formula sensitivity while holding the top-50 candidates and saved raw scores fixed.

| Topic | BM25 | Shipped | Delta | Best counterfactual | Best nDCG | Shipped - no coverage | Diagnostic |
|---:|---:|---:|---:|---|---:|---:|---|
| 224 | 0.816497 | 0.692677 | -0.123820 | passage | 0.643764 | +0.150150 | model signal limited |
| 515 | 0.229925 | 0.118780 | -0.111146 | passage | 0.182457 | -0.005306 | model signal limited |
| 219 | 0.493357 | 0.405354 | -0.088003 | no_coverage | 0.524386 | -0.119032 | aggregation sensitive |
| 31 | 0.847805 | 0.840805 | -0.007000 | passage | 0.894856 | +0.039127 | aggregation sensitive |

The no-coverage counterfactual is the configured document/passage blend with the bounded coverage bonus set to zero. Because nDCG is rank-based, `shipped - no coverage` is a ranking counterfactual, not the arithmetic contribution of the bonus to nDCG.

## Topic 224: saved component signals did not recover BM25

**Short label:** I want to understand why people immigrate or become refugees the challenges.<br>
**Information need:** I want to understand why people immigrate or become refugees, the challenges they face, and how laws and different groups shape immigration policies. Additionally, I'm interested in how various countries and religions view immigrants, and what options migrant workers have to improve their lives.

The shipped ranking changes nDCG@10 from **0.816497** to **0.692677** (**-0.123820**). Precision@10 moves from **1.00** to **0.90**, and the count of grade-2+ documents moves from **10** to **9**.

| Ranking | nDCG@10 | Delta vs BM25 | P@10 | Relevant @10 |
|---|---:|---:|---:|---:|
| bm25 | 0.816497 | +0.000000 | 1.00 | 10 |
| document | 0.597433 | -0.219064 | 0.80 | 8 |
| passage | 0.643764 | -0.172733 | 0.90 | 9 |
| no coverage | 0.542527 | -0.273970 | 0.80 | 8 |
| shipped | 0.692677 | -0.123820 | 0.90 | 9 |

BM25 top-10 qrel grades: `[4, 4, 3, 4, 4, 4, 4, 2, 3, 3]`.<br>
Shipped top-10 qrel grades: `[4, 3, 3, 3, 4, 4, 1, 2, 4, 4]`.

The rows below have the largest negative document-level DCG contribution changes. They show the measured rank/grade mechanism without asserting a semantic cause for the qrel label.

| Docid | Grade | BM25 rank | Shipped rank | Document | Passage | Support | DCG contribution delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| shard_06460_56923 | 4 | 1 | 12 | 5.9375 | 4.6087 | 5 | -15.0000 |
| shard_00796_30325 | 4 | 4 | 25 | 5.5000 | 4.6894 | 2 | -6.4601 |
| shard_03020_73434 | 4 | 5 | 26 | 5.8750 | 3.8106 | 3 | -5.8028 |
| shard_00813_20619 | 4 | 6 | 31 | 4.8750 | 4.7344 | 2 | -5.3431 |
| shard_02343_38729 | 4 | 2 | 9 | 5.5625 | 4.8806 | 6 | -4.9485 |
| shard_05430_31638 | 3 | 3 | 18 | 5.8750 | 4.9900 | 3 | -3.5000 |

Calibration uses this topic's top-50 candidate pool: **50 of 50** rows are explicitly judged (**100.00%**). Spearman measures monotonic association with the full qrel grade; AUC measures separation of grade-2+ from lower grades. Values near zero correlation or 0.5 AUC indicate weak separation inside this topic's candidate pool.

| Component | Spearman vs grade | Relevance AUC | Mean relevant | Mean lower-grade |
|---|---:|---:|---:|---:|
| document raw logit | 0.201 | 0.405 | 5.322 | 5.438 |
| passage weighted raw logit | 0.135 | 0.495 | 4.682 | 4.357 |
| no coverage blend | 0.185 | 0.451 | 5.002 | 4.897 |
| bounded coverage support | 0.019 | 0.356 | 2.848 | 3.500 |
| shipped score | 0.182 | 0.446 | 5.714 | 5.772 |

**Interpretation.** None of the document-only, passage-only, or no-coverage rankings recovers BM25. The strongest saved-score alternative is **passage**, which remains **-0.172733** nDCG from BM25. Coverage improves the no-coverage ranking by **+0.150150** nDCG. Precision@10 changes by **-0.10** and the grade-threshold relevant count changes by **-1**, so both binary inclusion and graded order should be inspected. The largest negative saved DCG movement is `shard_06460_56923` (grade 4), from BM25 rank 1 to shipped rank 12 (-15.0000 DCG). The shipped-score calibration is Spearman **0.182** against qrel grade and AUC **0.446** for the configured relevance threshold; these values describe the observed pool and do not establish a semantic cause.

## Topic 515: saved component signals did not recover BM25

**Short label:** I'm interested in learning why cancer rates are rising particularly among young.<br>
**Information need:** I'm interested in learning why cancer rates are rising, particularly among young people, and whether factors like cell phone use are connected. I’d also like up-to-date information on cancer mortality rates, including annual breast cancer deaths in the UK and the current life expectancy for lung cancer.

The shipped ranking changes nDCG@10 from **0.229925** to **0.118780** (**-0.111146**). Precision@10 moves from **0.10** to **0.10**, and the count of grade-2+ documents moves from **1** to **1**.

| Ranking | nDCG@10 | Delta vs BM25 | P@10 | Relevant @10 |
|---|---:|---:|---:|---:|
| bm25 | 0.229925 | +0.000000 | 0.10 | 1 |
| document | 0.126579 | -0.103346 | 0.00 | 0 |
| passage | 0.182457 | -0.047468 | 0.10 | 1 |
| no coverage | 0.124085 | -0.105840 | 0.00 | 0 |
| shipped | 0.118780 | -0.111146 | 0.10 | 1 |

BM25 top-10 qrel grades: `[1, 2, 1, 1, 1, 1, 1, 0, 1, 1]`.<br>
Shipped top-10 qrel grades: `[0, 0, 1, 2, 1, 1, 0, 0, 1, 0]`.

The rows below have the largest negative document-level DCG contribution changes. They show the measured rank/grade mechanism without asserting a semantic cause for the qrel label.

| Docid | Grade | BM25 rank | Shipped rank | Document | Passage | Support | DCG contribution delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| shard_00906_41399 | 1 | 1 | 18 | 5.8750 | 7.0644 | 5 | -1.0000 |
| shard_05429_39042 | 2 | 2 | 4 | 6.3750 | 7.1081 | 6 | -0.6008 |
| shard_03336_76694 | 1 | 5 | 11 | 5.6875 | 7.1613 | 6 | -0.3869 |
| shard_03429_77666 | 1 | 7 | 47 | 6.0625 | 4.7306 | 2 | -0.3333 |
| shard_04510_23948 | 1 | 9 | 13 | 7.2500 | 7.5625 | 2 | -0.3010 |
| shard_06288_39021 | 1 | 10 | 19 | 6.3125 | 5.9744 | 6 | -0.2891 |

Calibration uses this topic's top-50 candidate pool: **50 of 50** rows are explicitly judged (**100.00%**). Spearman measures monotonic association with the full qrel grade; AUC measures separation of grade-2+ from lower grades. Values near zero correlation or 0.5 AUC indicate weak separation inside this topic's candidate pool.

| Component | Spearman vs grade | Relevance AUC | Mean relevant | Mean lower-grade |
|---|---:|---:|---:|---:|
| document raw logit | -0.053 | 0.436 | 6.167 | 6.257 |
| passage weighted raw logit | -0.019 | 0.574 | 6.553 | 6.394 |
| no coverage blend | -0.029 | 0.539 | 6.360 | 6.325 |
| bounded coverage support | 0.038 | 0.426 | 3.333 | 3.851 |
| shipped score | -0.006 | 0.525 | 7.193 | 7.288 |

**Interpretation.** None of the document-only, passage-only, or no-coverage rankings recovers BM25. The strongest saved-score alternative is **passage**, which remains **-0.047468** nDCG from BM25. Coverage worsens the no-coverage ranking by **-0.005306** nDCG. Binary precision and the grade-threshold relevant count are unchanged, so the measured loss is entirely in graded ordering. The largest negative saved DCG movement is `shard_00906_41399` (grade 1), from BM25 rank 1 to shipped rank 18 (-1.0000 DCG). The shipped-score calibration is Spearman **-0.006** against qrel grade and AUC **0.525** for the configured relevance threshold; these values describe the observed pool and do not establish a semantic cause.

## What was measured

The population is **22** development topics. BM25 retrieves 1,000 documents per topic; the reranker reorders the first **50**. Metrics use the projected development qrels, treat missing judgments as grade 0, and use grade **2+** for precision and relevant-count metrics. nDCG uses the full integer grades.

All counterfactuals reorder exactly the same top-50 documents. Document and passage values are raw logits; the passage component is the saved weighted top-window aggregate. The no-coverage score is:

```text
0.5 * document_raw_logit
+ 0.5 * passage_weighted_raw_logit
```

The shipped score adds:

```text
+ 0.25 * bounded_coverage_support
```

Scores are labeled `mixedbread-ai/mxbai-rerank-base-v2` at revision `3ea9d4dffa7d12a4f366be8e275c349de9fc9865`, backend version `5.6.0`, `bfloat16`, and `raw_logits`. Provider is `Modal / CLOUD_PROVIDER_AWS / us-east-1` and hardware is `NVIDIA A100-SXM4-80GB`. `not_recorded` means the supplied artifacts do not support a stronger runtime claim.

## Modal CUDA versus retained local ROCm logits

The prior top-50 raw-logit artifacts were retained long enough to compare the
same score identities before local cache replacement. Every overlapping row
matched on topic, document, rank, and schema-v2 cache key. For document scores,
**589 of 1,100** values differed; the mean absolute difference was **0.0412**
and the maximum was **0.1875**. For window scores, **7,323 of 12,938** values
differed; the mean absolute difference was **0.0427** and the maximum was
**0.25**.

These differences show that the pinned Modal CUDA and local ROCm executions
were not numerically interchangeable even though they used the same model,
revision, score representation, dtype label, inputs, and cache keys. They do not
isolate hardware from kernel/runtime effects. Exact paths, artifact hashes, and
distribution statistics are in
[modal_cuda_vs_local_rocm_overlap.json](modal_cuda_vs_local_rocm_overlap.json).

## Counterfactual method separates score signal from formula sensitivity

The generator verifies that every shipped top-50 score equals the configured component formula and that each component's `base_rank` matches BM25. It then sorts each topic independently by document, passage, and no-coverage score, breaking ties by BM25 rank. A loss is aggregation-sensitive only if one of those saved-score rankings reaches the BM25 nDCG@10. Otherwise it is model-signal-limited.

This rule deliberately does not call the labels causal. It diagnoses whether the observed regression can be repaired by recombining the existing component scores; it cannot establish why the model assigned those scores or whether the pattern will repeat on hidden topics.

## Limitations and robustness checks

- The analysis covers 22 development topics, so topic-level effects are high variance and unsuitable as precise test-set estimates.
- Projected qrels are the metric source of truth here. Explicit judgments cover 1100 of 1100 top-50 candidates (100.00%); missing judgments receive grade 0. The report does not infer semantic facets or qrel intent beyond those labels.
- Counterfactuals reuse saved logits. They isolate aggregation choices but do not measure the effect of rescoring with a new model, GPU, dtype, library version, or longer candidate depth.
- The no-coverage ranking changes aggregate nDCG by +0.001744 versus shipped and leaves 2 of 2 current large regressions below BM25. This controlled comparison does not by itself support removing or retaining coverage as a general fix.
- Hardware is surfaced from the supplied completed Modal status. This report is current for the promoted Modal artifacts named in the manifest; a later artifact replacement must trigger regeneration.
- The completed warm-cache proof covers this exact score identity and corpus: 22,000 document rows and 227,156 window rows, with zero model calls, an unchanged cache, and semantic equality to canonical artifacts. A model revision, score policy, query/document text, or chunking change requires a new proof.

## Recommended next steps

1. The completed Modal artifacts are reflected here. Diff `metrics.json`, `component_counterfactuals.csv`, and `top10_movements.csv` against any retained pre-Modal snapshot if a cross-runtime effect size is needed.
2. Treat topics 224 and 515 as current score-calibration/model-signal probes. Test BM25-preserving interpolation or a confidence guardrail before tuning aggregation around them.
3. Treat topics 219 and 31 as current aggregation-sensitivity probes. Evaluate formula changes with leave-one-topic-out selection and keep the per-topic regression guardrail, not only mean nDCG.
4. Retain both nDCG@10 and precision/relevant-count diagnostics. Topics 515, 219, and 31 demonstrate that binary precision can stay flat while graded ordering worsens.

## Further questions

- The raw-score overlap quantifies numerical cross-runtime differences, but a retained pre-promotion ranked-output snapshot would still be needed to attribute every metric and rank movement directly to that runtime change.
- Which judged document properties distinguish high-grade documents that BM25 preserves but both reranker components down-rank? That needs an explicit labeled feature study; this report does not infer facets from document prose.
- Can a single frozen blend recover the aggregation-sensitive topics without worsening the 2 signal-limited topics under leave-one-topic-out validation?

## Reproduce

```bash
.venv/bin/python -m trec_rag.topic_regression_postmortem --runtime-status reports/experiments/bm25_mixedbread_topic_regression_postmortem_v1/modal_runtime_status.json --warm-cache-status reports/experiments/bm25_mixedbread_topic_regression_postmortem_v1/modal_warm_cache_verification.json
```

Machine-readable evidence is in [metrics.json](metrics.json), [component_counterfactuals.csv](component_counterfactuals.csv), [component_calibration.csv](component_calibration.csv), [component_grade_summary.csv](component_grade_summary.csv), and [top10_movements.csv](top10_movements.csv). Input paths and SHA-256 digests are pinned in [manifest.yaml](manifest.yaml).
