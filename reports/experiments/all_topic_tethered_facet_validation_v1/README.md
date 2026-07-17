# All-topic tethered-facet validation v1

Decision: **retain RRF**.

This directory contains the sanitized, reproducible report over the corrected canonical v2 ranking and evaluation. The superseded v1 ranking and evaluation are rejected evidence. This is a retrospective full-development stress test using known-relevant judgments; it is not evidence of generalization, and downstream RAG answer generation is out of scope.

The report-source trust boundary pins exact planning, retrieval, score-plan, scoring, ranking-v2, and evaluation-v2 roots. To avoid an unnecessary multi-gigabyte semantic replay, verification checks exact bytes of four pinned upstream seal files and the two sealed cost leaves; ranking and evaluation inventories are rehashed in full. It then independently rebuilds and compares JSON, HTML, SQLite bytes, table schemas, and every dataset payload.

- `summary.json`: decision, exact roots, costs, statistics, and scope limits
- `report_data.sqlite`: arm, depth, topic, and facet-bucket rows
- `artifact.json`: hashes and sanitization declaration
- `report.html`: self-contained accessible report

Run verification from the repository root:

```bash
.venv/bin/python -m trec_rag.build_all_topic_tethered_report verify --report reports/experiments/all_topic_tethered_facet_validation_v1
```
