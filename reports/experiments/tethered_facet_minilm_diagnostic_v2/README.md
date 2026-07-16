# Tethered facet MiniLM diagnostic v2

This create-only report corrects the v1 interpretation after independent
advisor review. It is a **post-qrels diagnostic** over four fixed topics, uses
**no new retrieval**, and is **not production validation**.

The report reads the exact historically anchored qrels projection already
bound by Task 4 to authenticate and recompute bounded diagnostic facts. It
does not read original qrels. Narrative-tethered MiniLM improved judged-
relevant facet yield, but net noise reduction was not established.

## Reproduce

```bash
BASE=outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1
REPORT=reports/experiments/tethered_facet_minilm_diagnostic_v2

.venv/bin/python -m trec_rag.build_tethered_facet_report \
  --task1-receipt "$BASE/scoring/preflight.json" \
  --task2-receipt "$BASE/scoring/scoring_receipt.json" \
  --task3-freeze "$BASE/freeze" \
  --task4-evaluation "$BASE/evaluation" \
  --output "$REPORT"
```

The builder is create-only. Use a new output directory for a replay.
