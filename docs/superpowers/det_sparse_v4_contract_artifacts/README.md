# Deterministic sparse v4 contract artifacts

Status: offline scaffolding only. These files do not authorize model
inference, topic access, qrels access, retrieval, reranking, downloads, hosted
APIs, paid calls, or an agent loop.

This directory contains literal schemas and fixtures for the v4 synthetic
exact-span qualification contract. They are intentionally synthetic and
topic-free. The current bundle has the full 24-case prompt/corpus/gold/request
fixture set, but it is still not inference-authorizing because replay,
attestation, oracle-renderer, and runtime ledgers remain incomplete.

`semantic_anchor_case_registry_v1.json` freezes the 24-case universe and category
arithmetic. The corpus, gold-label, and JSONL request fixture files now contain
all 24 cases. `semantic_anchor_request_fixture.case001.json` is retained as the
single-case smoke fixture for byte-level request checks.

Before v4 can be frozen, this bundle must be expanded with the remaining
non-model fixture registries required by
`docs/superpowers/det_sparse_v4_synthetic_design.md`.
