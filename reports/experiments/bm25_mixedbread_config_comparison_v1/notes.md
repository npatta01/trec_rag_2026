# BM25 vs. Mixedbread Reranker: Modal Raw-Logit Topic Comparison

## Result

Using the promoted Modal A100 raw logits, the configured top-50 Mixedbread
reranker improves mean development nDCG@10 from **0.413961** to **0.531008**:
**+0.117047 absolute** and **+28.27% relative**. It improves **18 of 22**
topics and degrades **4**; topics **224** and **515** cross the predeclared
-0.10 large-regression threshold.

Precision@10 rises from **0.704545** to **0.763636** (+0.059091), equivalent
to 13 additional grade-2+ documents across the 22 top-10 lists. All compared
top-10 and top-50 documents have projected judgments. Set-based metrics at
rank 50 and recall@100 are unchanged by construction because the reranker only
reorders the first 50 documents of the shared BM25 candidate pool.

## Per-topic nDCG@10

Rows are sorted from the largest regression to the largest improvement.

| Topic | Short topic description | BM25 | Reranker | nDCG@10 delta | P@10 delta | relevant delta |
|---:|---|---:|---:|---:|---:|---:|
| 224 | immigration and refugees | 0.816497 | 0.692677 | -0.123820 | -0.10 | -1 |
| 515 | rising cancer rates | 0.229925 | 0.118780 | -0.111146 | +0.00 | +0 |
| 219 | societal impact of technology | 0.493357 | 0.405354 | -0.088003 | +0.00 | +0 |
| 31 | e-waste impacts | 0.847805 | 0.840805 | -0.007000 | +0.00 | +0 |
| 225 | violent video games | 0.257264 | 0.269217 | +0.011953 | -0.20 | -2 |
| 37 | family structures and childhood | 0.638982 | 0.664902 | +0.025920 | +0.10 | +1 |
| 161 | abortion arguments | 0.666326 | 0.696904 | +0.030578 | +0.00 | +0 |
| 897 | alcohol and neighborhood life | 0.304874 | 0.351468 | +0.046593 | +0.00 | +0 |
| 499 | euthanasia legalization | 0.681611 | 0.729292 | +0.047681 | -0.10 | -1 |
| 477 | definitions of race | 0.455915 | 0.522322 | +0.066407 | +0.20 | +2 |
| 273 | African resources and poverty | 0.322328 | 0.419786 | +0.097458 | +0.20 | +2 |
| 144 | financial institutions and banks | 0.088676 | 0.199244 | +0.110568 | +0.30 | +3 |
| 300 | reducing global poverty | 0.354602 | 0.471386 | +0.116784 | -0.10 | -1 |
| 233 | social media and mental health | 0.543836 | 0.716458 | +0.172622 | +0.00 | +0 |
| 407 | housing and rent prices | 0.278636 | 0.453903 | +0.175267 | +0.10 | +1 |
| 14 | sports and societal impact | 0.355333 | 0.557499 | +0.202165 | +0.30 | +3 |
| 707 | health risks of substances | 0.202666 | 0.438771 | +0.236104 | +0.20 | +2 |
| 200 | the Holocaust | 0.140084 | 0.386992 | +0.246909 | +0.30 | +3 |
| 58 | nuclear energy and safety | 0.500808 | 0.774674 | +0.273866 | +0.00 | +0 |
| 84 | vaccine safety | 0.491800 | 0.783126 | +0.291325 | +0.00 | +0 |
| 213 | Korean War | 0.112160 | 0.468324 | +0.356164 | +0.10 | +1 |
| 72 | deforestation | 0.323659 | 0.720292 | +0.396633 | +0.00 | +0 |

The canonical machine-readable views are [topic_metrics.csv](topic_metrics.csv)
and [topic_metric_deltas.csv](topic_metric_deltas.csv).

## Cache and score correctness

The candidate consumed the promoted schema-v2 artifacts containing **22,000
document rows** and **227,156 window rows**. Their SHA-256 digests are
`bebc16f08e91bedd6d29900feb460943e66b1ba0188c89eabeaf6af63993f4a1`
and `e05cde296fd46766b89ccf71a59329571dc3c590cef41ae61bacf41734fa844c`.
The comparison made zero model-inference requests and both configs hit all 22
BM25 retrieval-cache entries.

Modal's CPU-only warm verification rebuilt all 249,156 artifact rows with model
loading forbidden, 22,000 document cache hits, 227,156 window cache hits, and
zero model scores. It proved the persistent cache was unchanged and the rebuilt
artifacts were semantically identical to the canonical Modal artifacts. The
runtime and warm-proof records are retained with the detailed regression report.

On the retained top-50 overlap, Modal CUDA and the prior local ROCm run used the
same schema-v2 cache keys but were not numerically identical: 589 of 1,100
document scores differed (mean absolute difference 0.0412, maximum 0.1875), and
7,323 of 12,938 window scores differed (mean 0.0427, maximum 0.25). The exact
comparison is saved in the regression report's
`modal_cuda_vs_local_rocm_overlap.json`.

## Interpretation limits

- These are 22 development topics evaluated with projected qrels, not a hidden
  test-set claim. Topic deltas are diagnostic rather than precise estimates of
  future performance.
- The qrels fully cover the BM25 top 50 but only a minority of the 22,000
  retrieved documents. Metrics therefore keep the configured rerank depth at
  50 even though raw logits and cache entries now exist for all 1,000 documents
  per topic.
- Cross-runtime score differences measure the complete pinned ROCm-versus-CUDA
  executions; they do not isolate hardware from lower-level kernel behavior.
- The deeper failure analysis, component counterfactuals, calibration, and
  document movements are in the topic regression postmortem.

## Reproduce

```bash
.venv/bin/python -m trec_rag.pipeline_comparison
.venv/bin/python -m trec_rag.topic_regression_postmortem \
  --runtime-status reports/experiments/bm25_mixedbread_topic_regression_postmortem_v1/modal_runtime_status.json \
  --warm-cache-status reports/experiments/bm25_mixedbread_topic_regression_postmortem_v1/modal_warm_cache_verification.json
```
