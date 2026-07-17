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
- `postmortem.json`: deterministic sanitized Topic 31/300 boundary analysis and
  offline recovery replay
- `postmortem.md`: approachable rendering of the same postmortem evidence

## Topic 31/300 postmortem

The canonical primary DUAL arm loses 7 known-relevant documents on Topic 31
and 3 on Topic 300 at depth 1,000. Boundary counts in the postmortem keep
unjudged documents explicitly separate from judged-below-2 documents.

The bounded Topic 300 recovery replay keeps the protected RRF top 100, permits
original-stream candidates and facet-only candidates retrieved within a
per-stream rank cap of 100, and returns deferred candidates in canonical RRF
order. It preserves the complete 1,701-document accepted-union permutation and
changes Topic 300 known-relevant capture from −3 to +2 at depth 1,000. This is
retrospective recovery evidence, not a promotion result.

The no-narrative result is separately labeled as a fixed post-hoc diagnostic:
it subtracts the frozen narrative term from the no-redundancy arm's recorded
objectives without replaying greedy state, produces a delta of 0, and is not
promotion-eligible. The replay performs no retrieval, network, download, model
load, inference, hosted, or paid calls.
