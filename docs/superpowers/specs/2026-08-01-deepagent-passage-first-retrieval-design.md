# DeepAgent Passage-First Retrieval — Design

**Date:** 2026-08-01

**Status:** proposed, not implemented

## Problem

Researchers see two tools: `search_climbmix` returns metadata for 10 documents,
and `extract_relevant_snippets` returns passages from **one** named document.
Nothing forces a researcher to open a second document, and in practice none
does.

Measured on the last live run: 63 grounded nuggets drawn from **5** documents,
with per-need nugget counts (9, 5, 23, 14, 12) exactly equal to per-document
counts. One document per need, one need per document. Every nugget carries
`single_document` support. Three of eight needs got nothing.

Two separate defects sit behind that:

1. **The first stage is too shallow.** The POC retrieves 10 documents per query.
   This repository's own competition configs retrieve **1000**
   (`rag25_bm25_full_query_v1.yaml`, `rag25_bm25_mixedbread_rerank_v1.yaml`) and
   rerank at `candidate_depth: 50`. The POC is 100× shallower than the pipeline
   it sits beside, and the operator reports queries where no relevant document
   appeared in the top 10 at all.
2. **The agent selects documents, not passages.** Even with deeper retrieval,
   a researcher that opens one document reads 1 of 1000 instead of 1 of 10.
   Depth alone changes nothing.

These are one fix. A deep cheap first stage feeding a cross-encoder that ranks
passages across candidates is the standard two-stage design; the POC currently
has neither half.

## Measurements

Sampled 300 cached ClimbMix documents: median 7,336 characters, p90 31,570,
max 104,157. At the configured 3,500-character chunk with 350 overlap that is a
mean of 4.33 chunks per document.

Local `mxbai-rerank-base-v2` on this ROCm host, warm:

| Rerank depth | Chunks scored | Latency |
| ---: | ---: | ---: |
| 10 documents (today) | ~43 | 1.08s |
| 25 documents | ~108 | 2.69s |
| 50 documents | ~216 | 5.38s |
| 100 documents | ~433 | 10.77s |

Scoring is linear at roughly 25ms per chunk.

**Hosted search is paced at one request per six seconds.** A 50-document rerank
costs 5.38s, so it fits inside a wait the run already takes. Depth 50 is close
to free in wall-clock terms; depth 100 exceeds the window and starts costing
real time.

## Measured recall (22 judged dev topics, umbrela qrels)

BM25 on the untouched narrative, graded against
`rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels`. Threshold matters enormously:
UMBRELA grade 1 means "related but does not answer", and 86% of all judgements
are grade >= 1, so that threshold is uninformative. Grade >= 3 is the set that
actually answers, averaging 193 documents per topic.

| Depth | Recall (grade >= 3) | Recall (grade >= 2) |
| ---: | ---: | ---: |
| 10 | 0.016 | 0.010 |
| 100 | 0.142 | 0.097 |
| 250 | 0.194 | 0.139 |
| 500 | 0.254 | 0.188 |
| 1000 | 0.326 | 0.250 |

Depth 100 to 1000 more than doubles recall of answering documents, the largest
absolute gain on the curve, and efficiency does not decay: the fraction of the
achievable ceiling holds near 31%-33% across the whole range. There is no
saturation point in reach, so **retrieve 1000**.

An earlier reading of this data concluded the opposite, that BM25 saturates by
depth 10. That was measured at grade >= 1 and was an artefact of the permissive
threshold. At grade >= 3 depth 10 captures 1.6% of the answering documents.

The reranker is what selects from the pool, so retrieving 1000 only pays if the
cross-encoder sees the pool. Reranking 50 of 1000 leaves recall at 0.067.

| Rerank depth | Recall (>=3) | Per search | Per topic (~25 searches) |
| ---: | ---: | ---: | ---: |
| 100 | 0.142 | 10.8s | ~4.5 min |
| 250 | 0.194 | 27s | ~11 min |
| 1000 | 0.326 | 108s | ~45 min |

Caveats: 22 topics; LLM-generated judgements; measured on the untouched
narrative, while researchers issue narrower reformulations whose recall is
likely worse. Depth is therefore justified as a floor, not a tuned optimum.

## Design

One combined tool replaces the agent-facing document-selection step:

- retrieve `hits=1000` from BM25 in a single request, which costs no extra
  rate-limit slot and no extra call;
- cross-encoder score the chunks of the pool against the focus query, as deep
  as the run's time budget allows; recall keeps climbing to 1000;
- return a **diversity-constrained top-K passage set**, each row carrying its
  invocation handle, sentences, score, and `document_id`.

Diversity is enforced in code, not requested of the model: a hard per-document
passage cap and a minimum distinct-document target before a second passage from
the same document is admitted. A global top-K without this collapses onto one
document exactly as today.

The document ledger is **retained internally**, not removed: document/focus
pagination, `pages_fetched`, `residual_count`, exhaustion and productivity
states, and `abandon_documents` all keep working, and the `single_document` vs
`multi_document` support label keeps its meaning. The document stops being the
agent's decision surface; it does not stop being provenance.

Citation syntax is unchanged. Handles stay invocation-scoped, and `S3`, `S3.2`,
`S3.2-4` resolve against stored sentence spans exactly as now.

## What not to do

- Do not return per-document aggregates and leave breadth to the model's
  judgement. That gives a better-informed version of the choice that strands it
  on one source.
- Do not implement plain global top-K without deterministic diversification.
- Do not remove document IDs, provenance, or internal exhaustion state.
- Do not reuse the existing per-snippet-call budget accounting unchanged, since
  a global scoring call has materially different cost. Charge it as its own
  retrieval unit.
- Do not raise `hits` without the passage-first surface. Alone it is wasted.
- Do not change citation syntax or reinterpret existing handles.

## Open questions

- Does breadth mean two documents per nugget, per need, or a run-level target?
  `multi_document` is currently per-nugget and asserts observed support only,
  never source independence.
- Should a deepening pass follow the diversified first pass, and what triggers
  it — an unresolved need, a conflict, a thin score margin, weak corroboration?
- Should the global result cache key on the ordered retrieved-document set plus
  every document hash, or on reusable per-chunk scores plus a light aggregation
  record? The per-chunk score cache already makes repeated chunks free across
  queries, which favours the second.
- Recall is argued here from being obviously too shallow, from the operator
  observing queries with no relevant top-10 document, and from the 100× gap with
  this repo's own pipeline. It has not been measured. A recall@k comparison on
  known-relevant documents would settle it.
