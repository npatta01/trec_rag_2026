# gpt-oss-20b structured sparse-query planning smoke test

## Outcome first

The architecture remains sound, but the tested local planner does not.
Facet decomposition and lexical expansion should remain separate, auditable
operations feeding sparse retrieval; `gpt-oss-20b` with the frozen v4 prompt
is not reliable enough to generate those plans.

The replacement run produced:

- three mechanically valid plans: 144, 213, and 224;
- one exact-span validation failure: 407;
- one token-limit JSON truncation: 515;
- zero timeouts and zero retries.

The advisor scored the three valid plans 8/12, 5/12, and 6/12. With zero
accepted plans and a 6.33/12 mean against the precommitted 10/12 threshold,
the model gate failed.

## Why this experiment was run

Dense full-collection retrieval is not currently practical for this corpus.
The tested strategy therefore combines two complementary sparse-retrieval
tools:

- lexical expansion to reduce vocabulary mismatch;
- facet decomposition to stop a long narrative from suppressing smaller
  information needs.

This matches the useful part of the prior-year evidence. MIT Lincoln
Laboratory used sparse retrieval under full-corpus constraints and found that
decomposition alone was not the complete system; parent context and later
reranking mattered. CFDA reported substantial directional gains from BM25
query expansion on prior topics, with a smaller incremental gain from PRF.
The evaluation settings differ, so those results motivate ordering rather
than promise a score here.

Sources:

- [MITLL TREC 2025 RAG paper](https://trec.nist.gov/pubs/trec34/papers/MITLL.rag.pdf)
- [CFDA TREC 2025 RAG paper](https://trec.nist.gov/pubs/trec34/papers/cfdalab.rag.pdf)

## Frozen replacement run

- Run ID: `query_planner_gpt_oss_smoke_v1_replacement_20260711T002032Z`
- Model: `openai/gpt-oss-20b`
- Revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Served name: `gpt-oss-local`
- Prompt: `sparse_query_planner_v4`
- Schema: `query_plan_v1`
- Renderer: `deterministic_sparse_renderer_v2`
- Analyzer: `unicode_content_terms_v1`
- Reasoning: medium
- Output budget: 6,000 tokens
- Temperature: 1.0
- Seed: 0
- Client concurrency: 1
- Per-topic timeout: 1,200 seconds
- Cache: disabled, guaranteeing one live replacement request per topic

The server used vLLM 0.24.0 from
`docker.io/vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`.

## Mechanical results

| Topic | Status | Elapsed | Completion tokens | Detail |
|---|---|---:|---:|---|
| 144 | success | 485.59 s | 5,902 | mechanically valid |
| 213 | success | 409.57 s | 5,009 | mechanically valid |
| 224 | success | 489.84 s | 5,986 | mechanically valid |
| 407 | plan validation error | 479.60 s | 5,865 | nonliteral and constructed spans |
| 515 | invalid JSON | 490.81 s | 6,000 | `finish_reason=length` |

Every raw HTTP response body was committed before validation as base64 plus a
SHA-256 digest. A post-run audit round-tripped all five bodies to their parsed
envelopes, verified all five outcome identities, found no temporary remnants,
and confirmed that the server had drained to zero active requests.

Local ignored artifacts are under
`outputs/query_planner_gpt_oss_smoke_v1/replacement_20260711T002032Z/`.
The original five-way harness incident is preserved separately in
`original_harness_incident.json`; it is not overwritten or treated as a model
result.

## Decision and next gate

Do not run retrieval with these plans and do not hide the failures through
fuzzy span correction. The next planner should use exact token-range IDs,
code-derived core priorities, and stricter expansions, then face the same five
topic rubric plus untouched holdouts. Only after that planner passes should we
retrieve and fuse original, global-expansion, and facet BM25 candidates.
