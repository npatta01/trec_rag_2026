# All-topic tethered-facet validation v1

Decision: **retain RRF**.

This directory contains the sanitized report over portable canonical v3 ranking
and evaluation evidence. Superseded v1/v2 evidence remains immutable but is not
admissible for this report. This is a retrospective full-development stress test
using known-relevant judgments; it is not evidence of generalization, and
downstream RAG answer generation is out of scope.

The tracked report is directly viewable without external data. Full source
verification additionally requires a 3 MiB compressed bundle (about 92 MiB
expanded) containing sealed rankings, evaluation tables, and minimal upstream
receipts. It contains candidate document identifiers. It is not stored in Git
or published with the sanitized report.

Expected local bundle:

- `cache/experiments/all_topic_tethered_facet_validation_v1_sources_v3.tar.zst`
- SHA-256: `3b726dcff28f8e67e6d4a5cf330e390c150b0a601091af17ae59b5870bc46e59`

Restore and verify from the repository root on an authorized machine:

```bash
sha256sum cache/experiments/all_topic_tethered_facet_validation_v1_sources_v3.tar.zst
tar --zstd -xf cache/experiments/all_topic_tethered_facet_validation_v1_sources_v3.tar.zst -C .
.venv/bin/python -m trec_rag.build_all_topic_tethered_report verify \
  --report reports/experiments/all_topic_tethered_facet_validation_v1 \
  --root outputs/all_topic_tethered_facet_validation_v1
```

The first command must match the pinned SHA-256 above. Obtain the bundle through
the repository's private artifact-transfer process when it is absent; verification
does not download it automatically.

- `summary.json`: decision, exact roots, costs, statistics, and scope limits
- `report_data.sqlite`: arm, depth, topic, and facet-bucket rows
- `artifact.json`: hashes and sanitization declaration
- `report.html`: self-contained accessible report
