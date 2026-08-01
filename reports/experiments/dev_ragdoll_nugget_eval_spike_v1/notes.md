# Dev RAGDoll nugget evaluation spike

This is the first answer-quality number produced in this repository. Everything before
it scored rankings against qrels; nothing scored generated text, and the 1,255 released
gold nuggets were referenced by no code at all.

**It validates wiring, not system quality.** Read the comparability section before
quoting any figure.

## What ran

1. `trec_rag.dev_rag_inputs` read the archived Pyserini responses for topics 58 and 213
   and emitted a six-column TREC run plus an organizer-shaped document JSONL.
2. `trec_rag.competition_rag` generated answers with `openai/gpt-5.6-sol`, unchanged from
   the merged competition path.
3. `trec_rag.ragdoll_io` derived a RAGDoll answers file and reshaped the gold nuggets.
4. `ragdoll nuggetizer eval` assigned the fixed gold nuggets and computed metrics.

## Results

| qid | strict_vital | strict_all | vital | all | nuggets | vital | words |
|---|---|---|---|---|---|---|---|
| 58 | 0.5714 | 0.5385 | 0.6857 | 0.6667 | 39 | 35 | 652 |
| 213 | 0.6296 | 0.7000 | 0.7778 | 0.8300 | 50 | 27 | 696 |
| **run** | **0.6005** | **0.6192** | **0.7317** | **0.7483** | 89 | 62 | — |

Label distribution: support 56 (62.9%), partial_support 23 (25.8%), not_support 10
(11.2%), failed 0. Judge cost: 9 calls, $0.1345 provider-reported.

## Integration gates

All passed, and each guards a failure that would otherwise look like a plausible score:

- **Non-empty qid join.** RAGDoll resolves ids from top-level `qid`/`topic_id`/`query_id`
  only and *skips* unmatched answers with a warning. Feeding submission rows directly
  yields empty metrics rather than an error, because our id lives in
  `metadata.narrative_id`. It cannot be moved: the organizer validator requires exactly
  three root keys, so a sidecar answers file is derived instead.
- **Non-degenerate labels.** An empty context short-circuits every nugget to
  `not_support` without any model call, which is indistinguishable from a bad answer.
- **`failed_count` zero.** Parse failures surface as `failed`, not as a low score.

## Reading the not_support cases

The ten misses are specific facts, not paraphrase failures: "Bison Energy includes
nuclear energy in their portfolios", "The Korean War became an election issue in 1952",
"Integration of black and white troops advanced during the Korean War". These look like
genuine coverage gaps from a BM25-only candidate pool. One gold nugget on topic 58,
"fusion energy is the current nuclear energy source", is questionable on its face and is
a reminder that gold nuggets are themselves model-assisted artifacts.

## Comparability

Do not compare these to published TREC numbers or to each other across configurations.

- **Two topics.** The per-topic spread is already 0.057 on strict_vital.
- **Raw BM25 ordering, no reranking.** The development archives predate the retriever
  provenance sidecar (`retrievers.py` raises `unverified cache missing provenance
  sidecar`), so they cannot be replayed through `trec_rag.pipeline`. Synthesizing
  sidecars would defeat that guard, so the archives were read directly instead. A real
  measurement needs a verified retrieval export.
- **Automated assignment scores above NIST manual assignment**, and `strict_vital` is the
  most fragile metric under full automation.
- The judge (`openai-codex/gpt-5.5`) differs from the generator (`openai/gpt-5.6-sol`),
  so no model graded its own writing.

## Next

Scale to all 22 development topics over a verified retrieval run, then add citation
support (`ragdoll support`) and arena battles (`ragdoll arena compare-all`, the primary
2026 metric).
