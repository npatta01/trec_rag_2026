# Facet-aware fusion held-out pilot design

Date: 2026-07-13

## Decision

Run a four-topic, held-out experiment to determine whether explicit facet-aware
set selection preserves relevant documents that facet-local MiniLM promotes but
family-balanced RRF suppresses. The primary proposal is constrained, rank-based
xQuAD. Current family-balanced RRF, quality-gated balanced interleaving, and
unconstrained xQuAD are controls. A TUS-style consensus score is diagnostic only.

This experiment changes fusion, not retrieval architecture. It adds no dense
primary retriever, iterative repair, paid model call, or post-qrels tuning.

## Evidence motivating the design

The completed four-topic MiniLM pilot found 22 relevant documents whose best
facet rank improved from below 20 under BM25 to at most 20 under facet-local
MiniLM. None survived the final family-balanced RRF top 100. The current fusion
assigns the original stream weight 0.5 and divides the facet-family weight 0.5
across all active facet streams. A document uniquely useful to one facet is
therefore structurally weaker than a document present in the original list.

The advisor recommended a set-selection objective whose relevance term can be
supplied by either the original stream or a facet stream. Using original BM25
alone as the global relevance term would repeat the same suppression.

## Held-out topic boundary

Forbidden topics at every boundary: `144`, `213`, `224`, `407`, `515`.

Previously evaluated topics excluded from confirmatory evidence: `200`, `225`,
`707`, `897`.

Eligible untouched topics are:

`14, 31, 37, 58, 72, 84, 161, 219, 233, 273, 300, 477, 499`.

Selection is independent of qrels and prior effectiveness. Sort eligible topics
by `SHA256("rag25_facet_aware_fusion_v1" || topic_id)` and take the first four.
The frozen topics are `233`, `273`, `161`, and `14`, in hash order.

No qrels-derived report, metric, label, or topic selection signal may be opened
before every query, candidate list, facet gate, fused ranking, parameter, and
evaluated topic ID is frozen and hashed.

## Facet plan

Create one deterministic, manually auditable manifest with exactly 24 atomic
facet queries:

- Topic `233`: 3 facets.
- Topic `273`: 7 facets.
- Topic `161`: 7 facets.
- Topic `14`: 7 facets.

Every facet must carry a short subject anchor plus one requested relation,
population, domain, comparison side, or factual obligation. Coverage boundaries
must partition the explicit requests in the narrative. Query terms must be
narrative-derived or have audited bridge-term provenance limited to domain
disambiguation, population binding, relation paraphrasing, or the standard name
of an explicitly requested concept. No query may introduce an answer, cause,
effect, example, date, mechanism, or narrower subtopic.

There is one planning pass and no retrieval-feedback rewrite.

## Candidate generation and local scoring

- Reuse the exact cached original-query top 100 for all four topics.
- Issue exactly 24 new facet top-100 requests through the existing persistent
  one-start-per-three-seconds limiter.
- Use the existing raw-first ledger and exact-identity cache.
- Do not retry a failed immutable attempt without a separately recorded recovery.
- Score facet candidates locally with
  `cross-encoder/ms-marco-MiniLM-L6-v2`, frozen revision
  `c5ee24cb16019beea0893ab7796b1df96625c6b8`, using the existing top-four
  window aggregation and ROCm workflow.
- Candidate pool per topic: original BM25 top 100 plus each accepted facet's
  MiniLM top 50, deduplicated by document ID with complete provenance.

Expected external cost: 24 free endpoint requests and no paid calls. At the
frozen three-second start interval retrieval requires at least 72 seconds.
The prior measured MiniLM throughput projects roughly one minute of local GPU
scoring for a candidate volume of this order; exact window and cache-miss counts
must be reported by preflight before inference.

## Frozen facet-quality gate

A stream is accepted only if all structural query checks pass and its MiniLM
top five satisfy:

- at least 3 contain the frozen subject anchor or approved synonym;
- at least 2 contain both the anchor and intended relation or approved
  paraphrase; and
- fewer than 2 match a frozen wrong-domain pattern.

Essay, dictionary, Scrabble, homework, or template warnings cannot reject a
stream alone. Anchor, relation, domain, and warning lexicons freeze before
retrieval. A rejected facet contributes no forced candidate. If all facets for a
topic are rejected, every fusion arm falls back to the original ranking.

## Rank transforms

Raw BM25 and MiniLM scores never cross query boundaries.

For original rank `r_O(d) <= 100`:

`g(d) = 1 / log2(1 + r_O(d))`; otherwise `g(d) = 0`.

For accepted facet rank `r_f(d) <= 50`:

`p_f(d) = 1 / log2(1 + r_f(d))`; otherwise `p_f(d) = 0`.

Base relevance is:

`Rel(d) = max(g(d), max_f p_f(d))`.

All facet priors are equal: `w_f = 1/m`, where `m` is the number of accepted
facets for that topic.

## Frozen ranking arms

All arms output 100 unique document IDs per topic.

### O: original

Unchanged original-query BM25 ranking.

### RRF: current control

The current topic-local family-balanced RRF: `k=60`, original family weight
0.5, total facet family weight 0.5 divided equally across accepted facets.

### BI: quality-gated balanced interleaving

Odd selection opportunities come from the original stream. Even opportunities
cycle through accepted facets in manifest order. Skip duplicates and exhausted
streams. A facet opportunity consumes its current highest unseen MiniLM-ranked
candidate. Continue until depth 100.

### XQ: rank-normalized xQuAD

Select greedily from the pooled candidates using:

`0.65 * Rel(d) + 0.35 * sum_f[w_f * p_f(d) * product_{s in S}(1-p_f(s))]`.

Ties are resolved by descending `Rel(d)`, best contributing stream rank, then
document ID. Parameters are fixed and may not be tuned on these topics.

### CXQ: constrained xQuAD (primary)

Use the XQ objective plus one coverage deadline per accepted facet. For facet
index `i` in frozen manifest order:

`deadline_i = 10 + ceil(40*i/m)`.

If a facet is not yet represented by its deadline, insert its highest-ranked
unseen MiniLM top-20 candidate. One document may satisfy several facets. Never
force a candidate from a rejected stream. Outside deadline insertions, use XQ.

### TUS-C: consensus diagnostic

Compute a deterministic rank-only analogue of TUS multi-facet consensus. It may
diagnose whether multi-facet documents are useful but is not eligible for
promotion because it can suppress uniquely useful facet evidence.

## Freeze and qrels firewall

Before qrels access, save and hash:

- topic-selection record and protected-topic rejection receipt;
- narrative and facet manifest;
- bridge-term and analyzer provenance;
- retrieval request identities, raw responses, and candidate lists;
- MiniLM tokenizer/model/window/scoring receipts;
- quality-gate lexicons, diagnostics, and accepted/rejected facets;
- definitions and complete rankings for O, RRF, BI, XQ, CXQ, and TUS-C;
- every parameter, tie-break, candidate-pool hash, and topic ID.

Qrels may then be projected once for exactly `233,273,161,14` under a create-only,
path-independent consumption registry. Evaluation code must reject qrels before
all freeze hashes exist and must reject protected, prior-pilot, extra, missing,
or reordered topic IDs.

## Metrics and promotion rule

Primary metrics:

- graded Recall@100;
- novel relevant documents retained at 100 versus RRF;
- fraction of pre-fusion novel relevant documents surviving final fusion.

Guardrails:

- nDCG@10;
- relevant documents at 10;
- judged rate at 10 and 100;
- per-topic deltas for every metric.

Promote CXQ only if all are true:

- aggregate graded Recall@100 improves over RRF;
- at least 25% of pre-fusion novel relevant documents survive;
- novel relevant retention is positive on at least three of four topics;
- aggregate nDCG@10 loses no more than 0.01 versus RRF;
- no topic loses more than 0.10 nDCG@10 versus RRF; and
- CXQ is not worse than XQ on both primary metrics.

If CXQ fails only because XQ is better, promote XQ if XQ improves graded
Recall@100 and passes the same nDCG guardrails. BI is a diagnostic/simple
fallback. RRF remains the result if neither xQuAD arm passes. Do not add Stage A,
new retrieval, recursive repair, or a more expensive reranker in response.

## Deliverables

- frozen design and topic-selection record;
- reviewed 24-facet manifest and exact request budget;
- raw-first retrieval and MiniLM scoring ledgers;
- quality-gate report with representative passages;
- frozen ranking/provenance artifacts for all arms;
- one-time qrels-backed evaluation and mechanical decision;
- rendered private HTML report explaining whether facet-aware fusion preserved
  novel relevant evidence and whether the nDCG guardrails held;
- post-results independent advisor review.

## Acceptance checks

- copied or aliased approvals cannot replay retrieval, inference, or qrels access;
- protected and prior-pilot topics fail at planning, retrieval, scoring, fusion,
  evaluation, cache joins, and reporting;
- score transforms and xQuAD ordering are deterministic under input reordering;
- no raw cross-query score comparison is possible;
- rejected facets cannot receive deadlines;
- every forced CXQ selection records its facet, deadline, source rank, and prior
  coverage state;
- the report reproduces every metric and decision from authenticated artifacts.
