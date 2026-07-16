# Tethered facet soft-coverage diagnostic v3

This source-bound, post-qrels report evaluates how six ranking arms reorder the
same accepted candidate union across topics 219, 72, 300, and 84. It is a
diagnostic of pooled document recall, macro ranking metrics, and a
qrels-positive facet-attribution proxy. It is not answer-generation, nugget,
faithfulness, or production evaluation.

## Canonical artifacts

- `report.html`: self-contained accessible report.
- `artifact.json`: sanitized canonical report payload and chart-omission note.
- `summary.json`: headline findings, boundaries, source hashes, and zero-call receipt.
- `report_data.sqlite`: inspectable tables for arm metrics, topic deltas,
  overlap decomposition, source identities, and receipts.

## Rebuild

From the repository root:

```bash
.venv/bin/python -m trec_rag.build_tethered_soft_coverage_report \
  --freeze outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/freeze \
  --evaluation outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/evaluation \
  --prior-summary reports/experiments/tethered_facet_minilm_diagnostic_v2/summary.json \
  --output reports/experiments/tethered_facet_minilm_diagnostic_v3
```

The build verifies the approved artifact hashes, bound qrels projection and
accepted union, and zero external-call receipts before writing output. No raw
document text or document identifiers are copied into the report artifacts.
