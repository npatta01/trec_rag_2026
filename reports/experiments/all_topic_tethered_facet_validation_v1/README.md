# All-topic tethered-facet validation v1

Decision: **retain RRF**.

This directory contains the sanitized, reproducible report over the corrected canonical v2 ranking and evaluation. The superseded v1 ranking and evaluation are rejected evidence. This is a retrospective full-development stress test using known-relevant judgments; it is not evidence of generalization, and downstream RAG answer generation is out of scope.

- `summary.json`: decision, exact roots, costs, and scope limits
- `report_data.sqlite`: arm, depth, topic, and facet-bucket rows
- `artifact.json`: hashes and sanitization declaration
- `report.html`: self-contained accessible report

Run verification from the repository root:

```bash
.venv/bin/python -m trec_rag.build_all_topic_tethered_report verify --report reports/experiments/all_topic_tethered_facet_validation_v1
```
