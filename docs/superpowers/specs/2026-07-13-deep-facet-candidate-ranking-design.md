# Deep facet candidate ranking pilot design

Date: 2026-07-13

## Decision

Run a four-topic, retrieval-only pilot to determine whether deep facet retrieval,
facet-local MiniLM scoring, and a common topic-coherence signal can add useful
candidates without repeating the prior xQuAD failure. This experiment ends at a
ranked candidate list. It does not select generation evidence, allocate an answer
token budget, generate prose, or evaluate citations.

The pilot compares four deterministic rankings and one no-redundancy diagnostic
over the same frozen accepted candidate union: current family-balanced RRF,
common-query scoring, facet-local scoring, and a dual-score coverage-aware
selector. Every arm freezes a complete permutation of that union; prefixes at
100, 500, and 1,000 are evaluation views rather than output limits.

## Advisor correction

Full-narrative MiniLM is a soft signal, never an eligibility gate. A document may
strongly answer one narrow request without resembling the complete multi-part
narrative. Facet-local scores are normalized only within their facet; raw MiniLM
or BM25 scores never cross query boundaries.

The selection pipeline is:

```text
original BM25 top 1,000 + all successful facet BM25 top 200
                         |
                facet-local MiniLM
                         |
            qrels-blind stream quality gate
                  /                    \
              U_raw                U_accepted
        discovery audit       ranking candidate union
                                      |
        common topic coherence + narrative + persistent facet
                 relevance + coverage + RRF prior
                                      |
                 complete deterministic permutations
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
facet request uses depth 200. The 24 facet queries below partition explicit
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

1. `effective specific strategies to prevent and reduce global warming and climate change`
2. `climate change actions for Antarctica`
3. `international and government measures to prevent and reduce climate change`
4. `economic cost of addressing global warming compared with its impacts`

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
- Issue exactly 24 new facet requests, each with `hits=200`.
- Use the existing raw-first immutable ledger, exact-identity cache, and persistent
  limiter of one request start per three seconds.
- Minimum uncached request-start time is 72 seconds.
- Do not retry a failed immutable attempt. Any transport, HTTP, schema, identity,
  or short-response failure aborts the pilot before qrels access. A successful
  response must contain exactly 200 unique, text-bearing ClimbMix document IDs.
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

Freeze two document-ID-deduplicated unions while retaining every stream, BM25
rank, facet-local MiniLM rank, text hash, and window-score provenance:

- `U_raw = original@1000 union every successful facet@200 response`;
- `U_accepted = original@1000 union every accepted facet@200 response`.

All ranking arms use `U_accepted`. Evaluation separately measures discovery in
`U_raw`, discovery surviving in `U_accepted`, and qrel-relevant documents lost by
the stream gate. This stage is called **stream gating plus local scoring**: it does
not perform a document-level MiniLM filter. Distinct document IDs are never
collapsed because their text is identical.

MiniLM uses the previously frozen tokenizer and document-window policy: pair
maximum 512 tokens, query maximum 192 tokens, minimum passage budget 256 tokens,
64-token passage overlap, at most 32 deterministically sampled windows per
document, and final non-empty short windows retained. Use the existing top-four
span-distinct aggregation: sort windows by raw logit, accept the first window and
then only spans contributing at least 128 previously uncovered document tokens,
select at most four, apply weights `0.55, 0.25, 0.13, 0.07`, and renormalize the
retained weights when fewer than four windows qualify. Phase-1 preflight must
reproduce these values from the authenticated prior receipt rather than silently
choosing new window settings or aggregation.

For scores sorted from best to worst in a stated population of size `n`, define:

`P(d) = (n - average_rank(d) + 1) / n`

Average rank is deterministic for ties. Thus the best value is `1`, the worst
present value is `1/n`, and an absent facet value is exactly `0`.

Score each document with:

- `F_i(d)`: facet-`i` document score converted with `P` over that facet's 200
  candidates when facet `i` is accepted; rejected facets contribute no feature,
  and a document absent from an accepted facet has zero.
- `G(d)`: common topic-coherence score converted with `P` over `U_accepted`.
- `N(d)`: exact full-narrative score converted with `P` over `U_accepted`; never a
  gate.
- `R(d)`: family-balanced RRF score converted with `P` over `U_accepted`; a
  missing stream rank contributes zero to the underlying RRF sum.
- `L(d) = max_i F_i(d)`: persistent strongest facet-local relevance, or zero when
  no facet is accepted.
- `C_i(S) = max_{s in S} F_i(s)`, with `C_i(empty)=0`.
- `B(d|S) = max_i [F_i(d) * (1 - C_i(S))]`: diminishing strongest uncovered-facet
  bonus, or zero when no facet is accepted; tied facets resolve by manifest order.

Redundancy uses the frozen analyzer's lowercased, stemmed, non-stopword token sets.
Let `J(d,s)` be token-set Jaccard, with `J=0` if either set is empty. Define:

```text
penalty(J) = 0                     when J < 0.80
penalty(J) = (J - 0.80) / 0.20     when J >= 0.80
D(d,S) = 0                         when S is empty
D(d,S) = max_s penalty(J(d,s))     otherwise
```

The same `0.80` threshold defines the reported near-duplicate metric. Exact-text
and near-duplicate documents remain in both unions. Raw scores remain in the audit
artifact but never enter a cross-query comparison.

## Frozen quality gate

Apply the existing qrels-blind stream gate after facet-local MiniLM scoring.
A facet is accepted only when its top five contain:

- the subject anchor in at least three documents;
- the anchor plus intended relation in at least two documents; and
- fewer than two wrong-domain matches.

Content-quality warnings cannot reject a stream by themselves. A rejected facet
contributes no facet feature, but its raw response remains in the audit ledger.
No individual document is removed merely for a low common-query or narrative
score. Anchor, relation, wrong-domain, stopword, and stemming rules freeze in the
manifest before retrieval.

Status values are distinct:

- `failed`: the request or scoring contract failed; abort the pilot;
- `rejected`: a complete scored stream failed the qrels-blind quality gate;
- `accepted`: a complete scored stream passed the gate;
- `unavailable`: reserved for a pre-run missing dependency and never used to
  reinterpret an attempted request.

If a facet has fewer than five scored documents, phase 1 fails. If a topic has zero
accepted facets, that topic's rankings fall back to the exact original permutation
for auditability and the whole pilot is automatically ineligible to advance.

## Ranking arms

Every arm receives exactly `U_accepted` and freezes a complete permutation of all
its document IDs. Prefixes at 100, 500, and 1,000 are evaluated; documents below
1,000 remain ranked and auditable. Because the validated original stream contains
1,000 unique documents, `U_accepted` must contain at least 1,000 IDs.

### RRF: current control

Family-balanced rank-only RRF with `k=60`: original family weight `0.5`, total
accepted-facet weight `0.5`, divided equally across accepted facets. A missing
rank contributes zero. When a topic has no accepted facets, the original family
receives weight `1.0` and the pilot is non-advancing as specified above.

### GLOBAL: common-topic control

Sort by `0.70*G(d) + 0.30*N(d)`, with `R(d)` and document ID as tie-breakers.
This arm measures whether one common semantic query is sufficient.

### FACET: facet-local control

Greedily maximize `0.70*L(d) + 0.30*B(d|S)`. Use `R(d)`, best contributing facet
in manifest order, original rank, best facet rank, and document ID as tie-breakers.
This isolates persistent facet-local relevance plus diminishing coverage without
using common-query scores.

### DUAL: primary arm

Greedily maximize:

`U(d|S) = 0.35*G(d) + 0.15*N(d) + 0.15*R(d) + 0.25*L(d) + 0.10*B(d|S) - 0.15*D(d,S)`

No facet has a minimum or maximum quota. Coverage gains diminish after strong
evidence is selected, but every remaining candidate is still eligible and is
eventually appended. Tie-break by `R(d)`, best facet-local percentile, original
rank, best facet rank, then document ID.

### DUAL-NR: no-redundancy diagnostic

Use the DUAL objective with the `D(d,S)` term fixed to zero. This offline arm is
not eligible to advance; it isolates whether the `0.80` lexical redundancy
threshold suppresses useful same-facet evidence.

For FACET, DUAL, and DUAL-NR, update `C_i(S)` and the maximum redundancy penalty
incrementally after each selection. Precompute token sets and pairwise Jaccard
values. This produces a deterministic `O(|U_accepted|^2 * m)` upper bound rather
than rescanning the selected set inside every candidate comparison.

## Freeze and one-time evaluation

Before qrels access, create and hash:

- topic-selection and protected-topic rejection receipts;
- narrative, common-query, and 24-facet manifest;
- all raw requests, responses, cache identities, and candidate rows;
- model, tokenizer, window, score-cache, inference, and runtime receipts;
- facet gates and representative qrels-blind diagnostics;
- `U_raw`, `U_accepted`, accepted/rejected stream records, and two frozen accepted
  facet-prefix union families at depths 50, 100, and 200: one using each stream's
  BM25 order and one using its facet-local MiniLM order;
- definitions and complete permutations for RRF, GLOBAL, FACET, DUAL, and DUAL-NR;
- every normalization, coefficient, tie-break, and evaluated topic ID.

Write a create-only `SEALED.json` containing every artifact path, byte length, and
SHA-256 value. Evaluation must validate the seal, atomically create a
`QRELS_ACCESSED` sentinel, and only then project qrels once for exactly
`219,72,300,84`. Retrieval, scoring, gating, and ranking commands must refuse to
run whenever that sentinel exists. Evaluation code must reject protected,
previously exposed, extra, missing, or reordered topics.

## Metrics and diagnosis

Candidate-discovery metrics:

- Recall and graded Recall for `U_raw`, `U_accepted`, and both the BM25-ordered and
  MiniLM-ordered accepted facet-prefix unions at depths 50, 100, and 200;
- `NovelRel = {d | qrel(d) >= 2, d not in original@1000, d in U_accepted}`;
- qrel-relevant documents discovered in `U_raw` but lost from `U_accepted` by the
  stream gate;
- per-stream inclusive provenance counts and exclusive-to-stream counts for
  `NovelRel`; inclusive counts are never summed across overlapping streams;
- accepted, rejected, failed, and unavailable stream counts kept separate.

Ranking metrics at depths 100, 500, and 1,000:

- Recall and graded Recall;
- `NovelRel` documents retained;
- fraction of `NovelRel` retained;
- judged rate;
- nDCG at 10 and 100;
- exact and near-duplicate rates;
- per-topic deltas against RRF.

Diagnosis is mechanical:

- qrel-relevant evidence absent from `U_raw` means retrieval or facet planning
  failed;
- evidence present in `U_raw` but absent from `U_accepted` means stream gating
  failed;
- evidence present in `U_accepted` but missing from an arm prefix means ranking or
  fusion failed;
- high retention with poor nDCG means coverage was over-weighted.

Metric definitions freeze before qrels access. Binary relevance is grade `>=2`.
Binary Recall@k divides retrieved relevant IDs by all grade-`>=2` IDs for the
topic. Graded Recall@k uses gain `2^grade - 1` and divides retrieved gain by total
topic gain. nDCG uses the same gain and discount `1/log2(rank+1)`. Unjudged IDs
receive gain zero but are reported separately through judged rate. A metric with
a zero denominator is `null`, never zero, and is excluded from macro means with
the excluded topic count reported.

## Advance and stop rule

This four-topic pilot cannot promote a production method. Advance DUAL to a
larger preregistered validation only if all are true:

- `U_accepted` adds at least five grade-2-or-higher documents beyond the
  original top 1,000 across at least two topics;
- DUAL graded Recall@500 is greater than both RRF and GLOBAL;
- DUAL graded Recall@1,000 is not below RRF;
- DUAL retains at least 50% of `NovelRel` at 1,000;
- aggregate nDCG@10 is within `0.02` of RRF;
- no topic loses more than `0.10` nDCG@10 versus RRF; and
- judged-rate differences are reported and no unjudged document is described as
  irrelevant; and
- in every leave-one-topic-out subset, DUAL macro graded Recall@500 remains above
  both RRF and GLOBAL and its macro nDCG@10 remains within `0.02` of RRF; otherwise
  report the unstable topic and do not advance.

If `U_raw` fails to add qrel-relevant evidence, stop and diagnose retrieval/facet
queries. If `U_raw` succeeds but `U_accepted` loses the evidence, diagnose stream
gating. If `U_accepted` succeeds but DUAL fails, retain the union and diagnose
fusion; do not reject facet decomposition. GLOBAL, FACET, and DUAL-NR are
diagnostic controls and cannot advance from this four-topic pilot.

## Deliverables

- versioned manifest and request preflight;
- raw-first retrieval ledger and exact cache report;
- local MiniLM preflight, benchmark, score cache, and receipt;
- facet gate report with bounded excerpts;
- frozen `U_raw`, `U_accepted`, prefix unions, and complete
  RRF/GLOBAL/FACET/DUAL/DUAL-NR permutations;
- one-time qrels-backed candidate and ranking evaluation;
- independent post-results advisor review;
- rendered private HTML explaining discovery, filtering, retention, ranking, and
  the mechanical decision.

## Acceptance checks

- protected and previously exposed topics fail at planning, retrieval, scoring,
  cache joins, fusion, evaluation, and reporting;
- qrels cannot load until `SEALED.json` validates every artifact hash;
- retrieval, scoring, gating, and ranking refuse to run after the create-only
  `QRELS_ACCESSED` sentinel exists;
- all external requests pass through the persistent limiter and respect the
  exact 24-request ceiling;
- raw BM25 or MiniLM scores never cross query boundaries;
- full-narrative scoring cannot reject a document;
- input reordering cannot change percentiles, rankings, or tie-breaks;
- `U_raw` and `U_accepted` are independently reproducible and gate losses are
  attributable;
- all ranking arms use identical `U_accepted` inputs and freeze full permutations;
- percentile direction, comparison population, tied ranks, absent values, empty
  token sets, zero accepted facets, missing ranks, and tie-breaking facets are
  covered by deterministic tests;
- every metric and decision in the HTML reproduces from frozen artifacts;
- no generation, evidence-token budgeting, answer writing, or citation evaluation
  enters this pilot.
