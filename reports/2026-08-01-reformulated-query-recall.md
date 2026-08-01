# Recall of Agent-Reformulated Queries at UMBRELA Grade >= 3

**Date:** 2026-08-01

**Status:** measured

Answers the open question in
`docs/superpowers/specs/2026-08-01-deepagent-passage-first-retrieval-design.md`:
that spec's recall curve was measured on untouched narratives, flagged as "the
optimistic end", with the worry that "researchers issue narrower reformulations
whose recall is likely worse". This measures the queries the agent actually
issued.

## Method

151 unique reformulated queries were recovered from the cached Pyserini
responses of prior DeepAgent runs on topic 224 (`cache/retrieval/pyserini_remote`,
narrative hash `baa5af90ea013149`). Topic 224 is judged. Each query, plus the
untouched narrative of all 22 judged dev topics, was retrieved at depth 1000
against `climbmix-400b` and graded against
`rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels`. 173 requests, no model calls.

Topic 224 has 168 documents at grade >= 3 and 1,065 at grade >= 1.

## Harness validation

Before trusting any new number, the harness reproduced the spec's published
curve. It matches to three decimals at every depth:

| Depth | Measured (>= 3) | Spec (>= 3) |
| ---: | ---: | ---: |
| 10 | 0.016 | 0.016 |
| 100 | 0.142 | 0.142 |
| 250 | 0.194 | 0.194 |
| 500 | 0.254 | 0.254 |
| 1000 | 0.326 | 0.326 |

## Result 1: at the POC's current depth, most reformulations return nothing

Per-query recall of answering documents, topic 224, 151 reformulations:

| Depth | Mean | Median | Max | Queries returning zero |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.003 | 0.000 | 0.030 | 98 (65%) |
| 100 | 0.020 | 0.012 | 0.077 | 31 (21%) |
| 250 | 0.040 | 0.030 | 0.143 | 16 (11%) |
| 500 | 0.062 | 0.048 | 0.196 | 8 (5%) |
| 1000 | 0.093 | 0.077 | 0.292 | 3 (2%) |

**At depth 10, 65% of the agent's own queries return no document that answers
the topic.** This confirms the operator's observation, and it is the strongest
single argument for the passage-first change: two thirds of the retrieval work
the agent does is currently wasted before any reranking or reading happens.

## Result 2: it is a depth problem, not a query-quality problem

The same queries at depth 1000 return nothing only 2% of the time. The
reformulations are not bad; they are being truncated.

Pooled union across all 151 reformulations — what a whole run can accumulate:

| Depth | Recall (>= 3) | Recall (>= 1) | Distinct documents |
| ---: | ---: | ---: | ---: |
| 10 | 0.137 | 0.100 | 971 |
| 100 | 0.577 | 0.370 | 8,161 |
| 250 | 0.744 | 0.515 | 18,572 |
| 500 | 0.845 | 0.652 | 34,540 |
| 1000 | **0.952** | 0.760 | 63,851 |

Against the untouched narrative alone on the same topic: 0.024 at depth 10 and
0.345 at depth 1000.

## What this changes

The spec's worry was directionally wrong in the way that matters. Reformulated
queries are worse *per query* than the untouched narrative — but far better *in
aggregate*, because they are diverse. Collectively they reach 0.952 of topic
224's answering documents at depth 1000, against 0.345 for the narrative.

So depth 1000 is justified more strongly than the spec argued, and the spec's
"retrieve 1000" recommendation should be read as well-supported rather than as
a floor pending measurement. Query diversity and retrieval depth are
complementary: neither alone gets close.

The corollary is that the reranker's selectivity, not BM25's reach, is now the
binding constraint. 63,851 distinct documents pass through a run's queries; the
passage stage scores only `rerank_depth` documents per query.

## Caveats, stated plainly

- **One topic.** The per-query and pooled numbers are topic 224 only, because
  that is the only judged topic with cached reformulations. The 22-topic curve
  is untouched narratives only.
- **The queries come from prior runs**, so they reflect the behaviour of the
  agent as it was, not as it will be after the passage-first change.
- **Judgement coverage is thin and unevenly so.** 94.3% of what reformulated
  queries return at depth 1000 is unjudged, against 78.8% for the narrative.
  Unjudged counts as non-relevant, so every number here is a lower bound, and
  it is a looser bound for reformulations than for the narrative. The pooled
  0.952 is therefore conservative, but the per-query comparison against the
  narrative is biased *against* the reformulations by construction: the qrels
  pool was built from narrative-like runs.
- **LLM-generated judgements.** These are umbrela qrels, not human ones.
- Recall of grade >= 3 exceeds recall of grade >= 1 throughout, which is
  expected: strongly on-topic documents are reachable by many queries, whereas
  marginally related ones are not.

## Reproducing

The scripts are scratch, not committed. The measurement is reproducible from
the cached responses: every depth-1000 request is now cached under
`cache/retrieval/pyserini_remote` with the retriever name
`deepagent_recall_probe`, so a re-run costs no hosted calls.
