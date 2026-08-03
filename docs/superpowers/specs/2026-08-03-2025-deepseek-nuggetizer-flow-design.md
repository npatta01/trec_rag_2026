# 2025 DeepSeek Subnarrative, Nuggetizer, and RAGDoll Flow

## Goal

Make the 2025 development-topic workflow safe to run and easy to compare: a
non-agentic BM25 baseline, the existing DeepSeek subnarrative and Nuggetizer
retrieval path, DeepSeek answer generation, and RAGDoll answer-quality
evaluation over the same topics.

This work also fixes the confirmed cross-topic evidence-binding defect in the
merged generation loader.

## Scope and terminology

- “Non-agentic” means the existing original-narrative BM25 baseline in
  `configs/rag25_bm25_full_query_v1.yaml`.
- “DeepSeek subnarrative path” means the existing facet pipeline: DeepSeek
  decomposition, original plus subnarrative retrieval lanes, deterministic
  reranking/coverage selection, and Nuggetizer canonicalization.
- “DeepSeek generation” means the existing fixed-retrieval
  `competition_rag` generator configured with the DeepSeek model; it is not a
  new autonomous answer-generation agent.
- RAGDoll evaluation is a development measurement harness, not an official
  2025 submission path.

## Design

### 1. Topic-scoped evidence resolution

The current generation loader returns one flat `docid -> text` map after
filtering the archive to selected topics. That permits a ranked document ID
from topic A to resolve to text from topic B when the ID is present only in
topic B’s row.

Introduce a topic-aware loader result with the logical shape:

```text
topic_id -> docid -> normalized document text
```

The generation path will pass the topic-to-ranked-docid mapping into the
loader and render each prompt from that topic’s map only. A missing document
for its owning topic remains a hard error. Existing single-topic callers and
RAGDoll support resolution retain a compatible flat-loading boundary where
there is no topic-to-ranked mapping.

The regression test will use two topics with the same document ID but
different text and assert that each generated prompt receives its own text.

### 2. 2025 experiment flow

The workflow uses distinct ignored local experiment namespaces:

```text
2025 topics
    ├── original narrative -> BM25 baseline -> retrieval metrics
    └── DeepSeek subnarratives -> BM25 + reranking -> Nuggetizer claims
                                      -> DeepSeek cited answers
                                                          -> RAGDoll metrics
```

The baseline is replayed from the existing archived BM25 responses when
available. Those archives predate the current provenance sidecars, so the
documented `dev_rag_inputs` replay helper is used rather than synthesizing
sidecars or weakening the strict pipeline cache validator.

The subnarrative path uses the existing competition retrieval components and
configuration shape, with a 2025 development-topic input and a unique output
ID. It must preserve original-topic retrieval as a lane, validate every
DeepSeek decomposition, retain topic/subnarrative provenance, and publish the
run plus full-text document artifact consumed by generation.

The answer-generation configuration uses the DeepSeek model already pinned in
`configs/rag26_competition_rag_deepseek_v1.yaml`, but points at the 2025
development topics and the subnarrative retrieval artifacts. It uses a unique
output ID and never overwrites the baseline or retrieval artifacts.

RAGDoll receives sidecar answer and nugget inputs from `ragdoll_io.py`; support
resolution uses the exact document artifact passed to generation. Evaluation
must report topic-level and aggregate nugget/citation metrics and retain the
provenance needed to distinguish baseline, retrieval, generation, and judge
settings.

### 3. Safety and execution boundaries

- No checked-in full-run configuration is changed to select the 22 development
  topics.
- Local 2025 configurations live under ignored `configs/local/` and use unique
  experiment/output IDs.
- Retrieval, generation, and RAGDoll remain independently runnable stages.
- Hosted retrieval, DeepSeek planning/canonicalization, DeepSeek generation,
  and RAGDoll judge calls require explicit run authorization; offline tests and
  cache-only replay may run without it.
- No raw documents, provider responses, credentials, or generated answer
  artifacts are committed or published.

## Verification requirements

1. A focused failing regression test demonstrates cross-topic misbinding on the
   pre-fix merged code.
2. The topic-scoped loader test passes after the minimal implementation.
3. Existing competition RAG and RAGDoll tests remain green.
4. The full repository suite passes in the prepared environment.
5. Configuration parsing and offline 2025 input construction are verified
   without hosted calls.
6. Any live run is reported separately with topic count, cache hits/misses,
   model calls, output paths, and post-run artifact validation.

## Out of scope

- Replacing the standalone DeepAgent SDK with a new answer-generation agent.
- Changing official 2026 full-run defaults.
- Claiming that a 2025 development score is an official TREC score.
- Running paid hosted calls without a separate explicit authorization.
