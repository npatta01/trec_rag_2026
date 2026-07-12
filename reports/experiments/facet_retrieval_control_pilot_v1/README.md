# Facet retrieval-control pilot v1

This directory freezes a four-stream, qrels-blind candidate-generation control
manifest derived from the exact prior R1 sparse-relevance manifest. Producing
the manifest makes no network call and does not alter the prior pilot.

The pilot covers `200/f07a`, `225/f02`, `225/f04`, and `707/f02`. Each stream
has the cached R1 baseline `B0` and three external reweighted arms: `W0`, `W1`,
and `W2`. The external-attempt ceiling is 12, retrieval depth is 100, and the
minimum interval between request starts is ten seconds.

Admission gates require the byte-pinned R1 source, its exact four baseline
queries, the fixed protected-topic set (`144`, `213`, `224`, `407`, `515`), no
new normalized query vocabulary, exact ordered BM25 settings, and exactly 12
external arms. The manifest generator is create-only; later retrieval work
must preserve these gates and freeze results before opening qrels.

No external retrieval, qrels access, model inference, reranking, or paid call
is performed while building this manifest.
