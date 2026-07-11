# Deterministic sparse v4 contract artifacts

Status: offline scaffolding only. These files do not authorize model
inference, topic access, qrels access, retrieval, reranking, downloads, hosted
APIs, paid calls, or an agent loop.

This directory contains literal schemas and fixtures for the v4 synthetic
exact-span qualification contract. They are intentionally synthetic and
topic-free. The current bundle has the full 24-case prompt/corpus/gold/request
fixture set, offline ledger/replay schemas, topic-free renderer oracle fixtures,
a static import/open audit fixture, and a model inventory attestation fixture.
It is still not inference-authorizing because no live analyzer/model/runtime
attestation has been collected and no executable runner has passed advisor
review.

`semantic_anchor_case_registry_v1.json` freezes the 24-case universe and category
arithmetic. The corpus, gold-label, and JSONL request fixture files now contain
all 24 cases. `semantic_anchor_request_fixture.case001.json` is retained as the
single-case smoke fixture for byte-level request checks.

Before v4 can be frozen, this bundle must be expanded with the remaining
non-model fixture registries required by
`docs/superpowers/det_sparse_v4_synthetic_design.md`.

The ledger and replay schemas in this directory define artifact shapes only.
They do not run a model, contact a sidecar, replay a response, or validate a
real run directory.
