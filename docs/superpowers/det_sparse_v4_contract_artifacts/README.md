# Deterministic sparse v4 contract artifacts

Status: offline scaffolding only. These files do not authorize model
inference, topic access, qrels access, retrieval, reranking, downloads, hosted
APIs, paid calls, or an agent loop.

This directory contains literal schemas and fixtures for the v4 synthetic
exact-span qualification contract. They are intentionally synthetic and
topic-free. The current bundle is a smoke artifact set, not the full 24-case
qualification corpus.

`semantic_anchor_case_registry_v1.json` freezes the 24-case universe and category
arithmetic. Only `synthetic-case-001` currently has full prompt, request, corpus,
and gold fixtures.

Before v4 can be frozen, this bundle must be expanded to the complete fixture
registry required by `docs/superpowers/det_sparse_v4_synthetic_design.md`.
