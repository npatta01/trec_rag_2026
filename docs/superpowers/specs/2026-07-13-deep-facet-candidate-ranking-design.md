# Deep facet candidate ranking pilot design

Date: 2026-07-13

## Decision

Run a four-topic, retrieval-only pilot to determine whether deep facet retrieval,
facet-local MiniLM scoring, and a common topic-coherence signal can add useful
candidates without repeating the prior xQuAD failure. This experiment ends at a
ranked candidate list. It does not select generation evidence, allocate an answer
token budget, generate prose, or evaluate citations.

The pilot compares four deterministic rankings over the same frozen candidate
union: current family-balanced RRF, common-query scoring, facet-coverage scoring,
and a dual-score coverage-aware selector. Every arm emits 1,000 unique document
IDs. The complete deduplicated union is also preserved so candidate discovery can
be separated from ranking loss.

## Advisor correction

Full-narrative MiniLM is a soft signal, never an eligibility gate. A document may
strongly answer one narrow request without resembling the complete multi-part
narrative. Facet-local scores are normalized only within their facet; raw MiniLM
or BM25 scores never cross query boundaries.

The selection pipeline is:

```text
original BM25 top 1,000 + accepted facet BM25 top 200
                         |
                facet-local MiniLM
                         |
             deduplicated candidate union
                         |
 common topic coherence + narrative + facet coverage + RRF prior
                         |
              ranked candidates to depth 1,000
```

## Fresh topic boundary

Forbidden topics at every boundary are `144`, `213`, `224`, `407`, and `515`.
Previously qrels-exposed pilot topics are also excluded from confirmatory evidence:
`200`, `225`, `707`, `897`, `233`, `273`, `161`, and `14`.

The remaining development topics are `31, 37, 58, 72, 84, 219, 300, 477, 499`.
Sort them by `SHA256("rag25_deep_facet_candidates_v1" || topic_id)` and select the
first four. The frozen order is `219, 72, 300, 84`.

No qrels file, projection, metric, label, prior topic effectiveness result, or
qrels-derived report may be opened before all queries, candidates, scores,
quality gates, parameters, rankings, and hashes are frozen.

## Frozen query manifest

The original stream is the exact cached narrative query at depth 1,000. Each
facet request uses depth 200. The 25 facet queries below partition explicit
narrative obligations and introduce no candidate answer.

### Topic 219: technology's societal impact

Common topic-coherence query:
`technology societal impacts daily life government business telehealth technical societies device rationing`

Facets:

1. `technology positive effects on society and daily life`
2. `technology negative effects on society and daily life`
3. `technology societal impact on government`
4. `technology impact on business and telehealth`
5. `role of technical societies in technology`
6. `why rationing devices may be needed with technological advancements`

### Topic 72: deforestation

Common topic-coherence query:
`deforestation causes effects environment climate animals humans Amazon rainforest prevention`

Facets:

1. `deforestation environmental impacts`
2. `deforestation climate impacts`
3. `deforestation impacts on animals and biodiversity`
4. `deforestation impacts on humans`
5. `main causes of deforestation`
6. `deforestation effects on the Amazon rainforest`
7. `actions to prevent deforestation`

`biodiversity` is an audited standard concept name for the explicitly requested
animal and environmental effects; it is not a candidate answer.

### Topic 300: preventing global warming

Common topic-coherence query:
`global warming climate change prevention mitigation Antarctica global measures economic costs impacts`

Facets:

1. `specific actions to prevent and reduce global warming and climate change`
2. `climate change actions for Antarctica`
3. `global measures to prevent and reduce climate change`
4. `economic cost of addressing global warming compared with its impacts`
5. `effective strategies to reduce global warming`

### Topic 84: vaccines

Common topic-coherence query:
`vaccine safety hesitancy COVID human vaccine types schedules global health history animal vaccination`

Facets:

1. `human vaccine safety`
2. `causes of public vaccine hesitancy especially COVID-19`
3. `types of human vaccines`
4. `recommended human vaccination schedules`
5. `vaccination impact on global health`
6. `historical public health challenges involving vaccination`
7. `animal vaccination recommendations`

The manifest must additionally freeze each facet's obligation, subject anchors,
relation terms, wrong-domain patterns, analyzer output, manifest order, and query
hash before retrieval.

## Retrieval and cost boundary

- Reuse the four exact cached original-query top-1,000 responses.
- Issue at most 25 new facet requests, each with `hits=200`.
- Use the existing raw-first immutable ledger, exact-identity cache, and persistent
  limiter of one request start per three seconds.
- Minimum uncached request-start time is 75 seconds.
- Do not retry a failed immutable attempt. Record it and continue only if the
  remaining manifest is still valid.
- Expected paid cost is $0. Do not download a model or use hosted inference.
- The already materialized `cross-encoder/ms-marco-MiniLM-L6-v2` revision
  `c5ee24cb16019beea0893ab7796b1df96625c6b8` is the only scorer.

Scoring has two preflighted phases. Before phase 1 inference, freeze exact
document, window, cache-hit, runtime, and memory estimates for all facet-local
scores. Apply the qrels-blind stream gate after phase 1. Before phase 2 inference,
freeze the same estimates for common-query and narrative scores over the accepted
union. Local scoring may proceed only when the model revision and tokenizer
receipt match the prior verified materialization. Stop for approval instead of
running if either phase projects more than ten minutes or requires materializing
any missing model file.

## Candidate construction and score features

Deduplicate repeated occurrences of the same ClimbMix document ID while retaining
every stream, BM25 rank, facet-local MiniLM rank, text hash, and window-score
provenance. Distinct document IDs are never collapsed merely because their text is
identical: exact-text and near duplicates are recorded and softly penalized during
ranking, but remain in the full union.

Score each document with:

- `F_i(d)`: top-four-window MiniLM relevance to facet `i`, converted to a
  within-facet rank percentile. A document absent from facet `i` has zero.
- `G(d)`: top-four-window MiniLM relevance to the frozen common topic-coherence
  query, converted to a topic-wide rank percentile.
- `N(d)`: top-four-window MiniLM relevance to the exact full narrative, converted
  to a topic-wide rank percentile. This is never a gate.
- `R(d)`: family-balanced RRF prior, converted to a topic-wide rank percentile.
- `D(d,S)`: maximum analyzer-token Jaccard similarity to a previously selected
  document, used only as a soft redundancy penalty.

All percentiles use deterministic average ranks for tied scores and are scaled to
`[0,1]`. Raw scores remain in the audit artifact but never enter a cross-query
comparison.

## Frozen quality gate

Apply the existing qrels-blind stream gate after facet-local MiniLM scoring.
A facet is accepted only when its top five contain:

- the subject anchor in at least three documents;
- the anchor plus intended relation in at least two documents; and
- fewer than two wrong-domain matches.

Content-quality warnings cannot reject a stream by themselves. A rejected facet
contributes no facet feature, but its raw response remains in the audit ledger.
No individual document is removed merely for a low common-query or narrative
score.

## Ranking arms

Every arm receives the same full deduplicated union and emits 1,000 unique
document IDs. If an arm's scoring rule exhausts preferred documents, append all
remaining union documents by `R(d)`, then original rank, best facet rank, and
document ID. Nothing is discarded merely because it falls below rank 1,000.

### RRF: current control

Family-balanced rank-only RRF with `k=60`: original family weight `0.5`, total
accepted-facet weight `0.5`, divided equally across accepted facets.

### GLOBAL: common-topic control

Sort by `0.70*G(d) + 0.30*N(d)`, with `R(d)` and document ID as tie-breakers.
This arm measures whether one common semantic query is sufficient.

### FACET: coverage-only control

Greedily maximize equal-weight residual facet coverage:

`FacetGain(d|S) = sum_i (1/m) * F_i(d) * (1 - C_i(S))`

where `C_i(S) = max_{s in S} F_i(s)` and `m` is the number of accepted facets.
Use `R(d)` and document ID as tie-breakers. This deliberately omits common-query
scores to expose the noise cost of facet coverage alone.

### DUAL: primary arm

Greedily maximize:

`U(d|S) = 0.35*G(d) + 0.15*N(d) + 0.15*R(d) + 0.35*FacetGain(d|S) - 0.15*D(d,S)`

No facet has a minimum or maximum quota. Coverage gains diminish after strong
evidence is selected, but every remaining candidate is still eligible and is
eventually appended. Tie-break by `R(d)`, best facet-local percentile, original
rank, best facet rank, then document ID.

## Freeze and one-time evaluation

Before qrels access, create and hash:

- topic-selection and protected-topic rejection receipts;
- narrative, common-query, and 25-facet manifest;
- all raw requests, responses, cache identities, and candidate rows;
- model, tokenizer, window, score-cache, inference, and runtime receipts;
- facet gates and representative qrels-blind diagnostics;
- the complete deduplicated union;
- definitions and complete top-1,000 rankings for RRF, GLOBAL, FACET, and DUAL;
- every normalization, coefficient, tie-break, and evaluated topic ID.

Then project qrels once for exactly `219,72,300,84`. Evaluation code must reject
protected, previously exposed, extra, missing, or reordered topics and must not
permit ranking regeneration after qrels access.

## Metrics and diagnosis

Candidate-discovery metrics:

- complete-union Recall and graded Recall;
- unique grade-2-or-higher documents beyond original BM25 top 1,000;
- per-facet contribution to those novel documents;
- accepted/rejected stream counts and overlap.

Ranking metrics at depths 100, 500, and 1,000:

- Recall and graded Recall;
- novel facet documents retained;
- fraction of union-novel relevant documents retained;
- judged rate;
- nDCG at 10 and 100;
- exact and near-duplicate rates;
- per-topic deltas against RRF.

Diagnosis is mechanical:

- relevant evidence absent from the complete union means retrieval or facet
  planning failed;
- evidence present in the union but missing from an arm means ranking/fusion
  failed;
- high retention with poor nDCG means coverage was over-weighted.

## Promotion and stop rule

Promote DUAL only if all are true:

- the complete union adds at least five grade-2-or-higher documents beyond the
  original top 1,000 across at least two topics;
- DUAL graded Recall@500 is greater than both RRF and GLOBAL;
- DUAL graded Recall@1,000 is not below RRF;
- DUAL retains at least 50% of union-novel relevant facet documents at 1,000;
- aggregate nDCG@10 is within `0.02` of RRF;
- no topic loses more than `0.10` nDCG@10 versus RRF; and
- judged-rate differences are reported and no unjudged document is described as
  irrelevant.

If the union fails the first condition, stop and diagnose retrieval/facet queries.
If the union succeeds but DUAL fails, retain the union and diagnose fusion; do not
reject facet decomposition. GLOBAL and FACET are diagnostic controls and cannot
be promoted from this four-topic pilot.

## Deliverables

- versioned manifest and request preflight;
- raw-first retrieval ledger and exact cache report;
- local MiniLM preflight, benchmark, score cache, and receipt;
- facet gate report with bounded excerpts;
- frozen union and RRF/GLOBAL/FACET/DUAL top-1,000 rankings;
- one-time qrels-backed candidate and ranking evaluation;
- independent post-results advisor review;
- rendered private HTML explaining discovery, filtering, retention, ranking, and
  the mechanical decision.

## Acceptance checks

- protected and previously exposed topics fail at planning, retrieval, scoring,
  cache joins, fusion, evaluation, and reporting;
- qrels cannot load until every ranking hash exists;
- all external requests pass through the persistent limiter and respect the
  separate 25-request ceiling;
- raw BM25 or MiniLM scores never cross query boundaries;
- full-narrative scoring cannot reject a document;
- input reordering cannot change percentiles, rankings, or tie-breaks;
- all arms use identical candidate unions;
- every metric and decision in the HTML reproduces from frozen artifacts;
- no generation, evidence-token budgeting, answer writing, or citation evaluation
  enters this pilot.
