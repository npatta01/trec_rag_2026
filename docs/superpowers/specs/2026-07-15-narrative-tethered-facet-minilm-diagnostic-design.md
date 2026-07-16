# Narrative-Tethered Facet MiniLM Diagnostic Design

Date: 2026-07-15

## Decision

Run one bounded, post-qrels mechanism diagnostic over the completed deep-facet
pilot. It tests whether MiniLM can rank noisy facet candidates more reliably
when each candidate is scored against the complete narrative plus the one facet
that retrieved it.

The diagnostic compares two matched reranking arms under one new deterministic
two-basket fusion:

- `FACET-2B`: reuse the existing facet-only MiniLM scores;
- `TETHERED-2B`: score the same generating-facet candidate pairs with
  `full narrative + Focus: <generating facet>`.

The current family-balanced RRF ranking remains the baseline. No retrieval,
query repair, adaptive search, model download, training, or hosted inference is
part of this experiment.

This is a diagnostic on topics whose qrels have already been opened. Its result
may explain the observed mechanism and justify a later fresh-topic test, but it
cannot establish production generalization.

## Question and causal contrasts

The experiment answers two separate questions without conflating them:

1. Does the protected two-basket fusion preserve more useful facet discoveries
   than the existing RRF? Compare `FACET-2B` with `RRF`.
2. Does adding the full narrative to each facet-local MiniLM query reduce facet
   noise and improve candidate recall? Compare `TETHERED-2B` with `FACET-2B`.

`TETHERED-2B - FACET-2B` is the primary causal contrast because both arms use
the same candidate pairs, normalization, quotas, ordering rules, and output
union. The only changed mechanism is the MiniLM query.

## Immutable inputs

Use only the authenticated artifacts from
`outputs/rag25_deep_facet_candidates_v1`:

- topics, in frozen order: `219`, `72`, `300`, `84`;
- accepted candidate union: 8,114 unique topic-document rows (`2,182` for
  topic `219`, `2,127` for `72`, `1,712` for `300`, and `2,093` for `84`);
- 24 accepted facet streams;
- 4,800 accepted generating-facet/document pairs;
- existing complete `RRF`, `GLOBAL`, `FACET`, `DUAL`, and `DUAL-NR`
  permutations;
- existing facet-only MiniLM window and document scores;
- existing qrels projection and evaluation artifacts.

The source snapshot hashes, row counts, document IDs, texts, narratives,
facets, facet order, retrieval provenance, and existing scores must match their
prior sealed receipts before any new work begins. Create a new versioned output
directory. Never overwrite the prior experiment.

Hard-reject protected topics `144`, `213`, `224`, `407`, and `515` before every
source join, cache lookup, tokenization, inference, ranking, evaluation, and
reporting boundary. Do not scan or join the global retrieval or reranker cache
by topic wildcard; lookup is permitted only through the frozen four-topic pair
manifest.

## Prohibited operations

This diagnostic performs:

- no retrieval request;
- no network request or paid call;
- no model or tokenizer download;
- no access to the original qrels source and no new projection; only the sealed
  existing four-topic projection may be read after rankings freeze;
- no query-language or BM25 change;
- no query rewrite, obligation proposal, or recursive search;
- no Mixedbread, dense retriever, or newly trained ranker;
- no tuning after metrics are viewed.

If a required local model file, authenticated input, cached text, existing
score, qrels projection, or compatible ROCm runtime is absent, stop and report
the missing dependency.

## Model and query policy

Use the already materialized model and exact prior scoring contract:

- model: `cross-encoder/ms-marco-MiniLM-L6-v2`;
- revision: `c5ee24cb16019beea0893ab7796b1df96625c6b8`;
- local execution: `.venv/bin/python-rocm`;
- backend: Hugging Face Transformers sequence classification;
- inference mode: evaluation with gradients disabled;
- dtype, tokenizer, pair length, passage overlap, deterministic long-document
  window sampling, and top-four span-distinct aggregation: byte-for-byte the
  prior deep-facet MiniLM contract.

For each accepted generating-facet/document pair, serialize the new query as:

```text
<exact full narrative>\n\nFocus: <exact generating facet query>
```

There is no instruction prompt or additional expansion. Freeze the exact UTF-8
query text and SHA-256 hash for every pair. One document retrieved by two facets
is scored twice, once for each generating facet, because those are distinct
information needs.

The prior facet-only score is matched by topic, facet identity, document ID,
text hash, model revision, tokenizer contract, window contract, and aggregation
contract. A mismatch invalidates the pair instead of silently recomputing the
old arm.

## Inference-free preflight and resource ceiling

Before a model forward pass, create and seal:

- the exact four-topic source bindings and protected-topic attestation;
- all 4,800 pair identities and tethered-query hashes;
- tokenizer and model file receipts;
- exact document-window identities;
- exact score-cache hits and misses;
- total, per-topic, and per-facet document and window counts;
- capped-document counts and token-coverage summaries;
- projected runtime and peak-memory evidence from the prior compatible run;
- output paths and create-only behavior.

The new inference may proceed only if:

- there are exactly 4,800 accepted generating-facet/document pairs;
- total tethered windows are at most 25,000;
- the projected runtime is at most ten minutes;
- no model or tokenizer materialization is required;
- the ROCm device and model revision match the receipt;
- no protected or unexpected topic occurs.

Any failed condition stops before inference. Score-cache hits reduce work but do
not change the manifest. Persist successful scores atomically by exact cache
identity so an interrupted local run may resume without repeating completed
pairs. Do not retry a failed model batch automatically.

## Within-facet score normalization

Raw cross-encoder logits are comparable only among documents scored with the
same generating-facet query. They must never be compared across facets.

For each accepted facet containing `n` candidates, sort document scores from
best to worst. Ties receive average rank. Define:

`P_f(d) = (n - average_rank_f(d) + 1) / n`

Thus the best percentile is `1`, the worst present percentile is `1/n`, and an
absent document has no score for that facet. Tie-breaking after the percentile
uses prior facet BM25 rank and then document ID.

The same deterministic percentile calculation is applied separately to:

- the saved facet-only MiniLM document scores for `FACET-2B`;
- the new narrative-tethered MiniLM document scores for `TETHERED-2B`.

If a document belongs to multiple accepted facet streams, retain each
facet-specific percentile in provenance. Selection traverses facet/document
edges globally from highest to lowest percentile while enforcing per-facet
quotas. A multi-facet document is therefore attributed to its highest-scoring
eligible generating facet; ties resolve by frozen facet order, prior facet BM25
rank, and document ID. Once selected, the document is deduplicated.

## Protected two-basket fusion

Every new arm must be a complete, duplicate-free permutation of that topic's
exact slice of the 8,114-row accepted candidate union. Prefixes are evaluation
views, not candidate-output limits.

### Ranks 1--100: protected head

Copy the existing RRF document IDs in exact order. Any difference in the first
100 document IDs, ranks, or count invalidates the arm. Consequently nDCG@100,
Recall@100, and all shallower rank metrics must exactly equal RRF.

### Ranks 101--500: two baskets

Construct two deduplicated baskets after removing the protected head:

1. `RRF basket`: the next 200 unseen documents in existing RRF order.
2. `Facet basket`: 200 documents absent from both the protected head and the
   200-document RRF basket, selected from accepted facets by the arm's
   within-facet MiniLM percentiles.

Facet-basket capacity is allocated equally among the accepted facets for that
topic. The frozen quotas are:

| Topic | Accepted facets | Base quota | Remainder allocation |
|---|---:|---:|---|
| `219` | 7 | 28 | first 4 facets receive one extra slot |
| `72` | 7 | 28 | first 4 facets receive one extra slot |
| `300` | 4 | 50 | none |
| `84` | 6 | 33 | first 2 facets receive one extra slot |

For each topic, sort every eligible facet/document edge by percentile
descending, frozen facet order, prior facet BM25 rank, then document ID. Traverse
that single ordered edge list and accept an edge only when its document is not
in the head, RRF basket, or facet basket and its generating facet has remaining
quota. This simultaneously honors the highest eligible score and the equal
facet budgets. If a facet cannot fill its quota because of duplicates or
exhaustion, place its unused slots in a shortage pool. After the first
traversal, allocate shortage slots one at a time in frozen facet order. For each
slot, select that facet's highest-percentile remaining eligible document; skip
a facet with no eligible document and continue cycling until the facet basket
has 200 documents or every facet is exhausted.

Merge the two disjoint baskets by deterministic alternation, beginning with the
RRF basket at rank 101. If one basket exhausts before contributing 200 unique
documents, fill the remaining ranks through 500 from the other basket and
record the exact shortage. An inability to produce 400 unique documents across
both baskets invalidates the arm.

This design protects 200 strong RRF candidates while guaranteeing bounded,
balanced space for 200 facet-local candidates. It does not let one noisy facet
flood the prefix.

### Ranks after 500: complete candidate union

Append every remaining accepted-union document in the frozen existing `DUAL`
order, skipping documents already selected. The result must contain every and
only accepted-union document ID exactly once.

## Experiment arms

| Arm | Candidate scores | Ranking rule | Purpose |
|---|---|---|---|
| `RRF` | Existing rank-only evidence | Existing complete permutation | Baseline |
| `FACET-2B` | Saved facet-only MiniLM | Protected two-basket fusion | Isolate fusion change |
| `TETHERED-2B` | New narrative + generating-facet MiniLM | Same protected two-basket fusion | Isolate narrative tether |

No coefficient, quota, protected depth, basket size, alternation rule, or query
serialization may change after evaluation. There is no second variant or repair
pass in this diagnostic.

## Evaluation

Reuse only the previously frozen four-topic qrels projection. Validate its hash
against the prior sealed receipt; do not open the original qrels source. A new
evaluation stage may read the existing projection only after all scores,
rankings, and hashes are sealed.

Primary comparison: `TETHERED-2B - FACET-2B`.

Secondary comparisons:

- `FACET-2B - RRF`;
- `TETHERED-2B - RRF`.

Report aggregate and per-topic:

- binary Recall@500 and graded Recall@500;
- binary Recall@1,000 and graded Recall@1,000;
- grade-2-or-higher relevant documents from the frozen 177-document novel set
  retained at 500 and 1,000;
- judged-document rate at 500 and 1,000;
- relevant-document counts entering from the RRF and facet baskets;
- facet contribution counts and relevant yield by facet;
- protected-head identity and metric checks.

Report nDCG@10, nDCG@100, and Recall@100 only as construction invariants. They
must exactly match RRF and are not optimization targets in this diagnostic.

## Frozen diagnostic decision rule

Label `TETHERED-2B` a **mechanical pass** only if every condition holds:

1. its top 100 is exactly identical to RRF;
2. its binary and graded Recall@500 are each at least those of both RRF and
   `FACET-2B`, with a strict improvement on at least one of the two metrics over
   `FACET-2B`;
3. it retains at least 89 of the frozen 177 novel relevant documents by rank
   500;
4. its binary and graded Recall@1,000 are each at least those of both RRF and
   `FACET-2B`;
5. it retains at least 142 of the 177 novel relevant documents by rank 1,000;
6. no topic loses more than `0.02` absolute binary or graded Recall@500 against
   either RRF or `FACET-2B`;
7. its judged-document rates at 500 and 1,000 are no more than `0.05` absolute
   below the corresponding comparison arm.

Label the result **inconclusive** when all score/ranking contracts pass but the
only unmet condition is judged coverage, a basket shortage, or a documented
coverage limitation of the frozen candidate pool. Otherwise label it a
**mechanical fail** and diagnose the failed conditions. These labels do not
authorize production promotion because the topics are qrels-exposed.

## Diagnostics

The result artifact must explain why the metrics moved, not only whether they
moved. Include:

- examples of documents promoted by `FACET-2B` but rejected by
  `TETHERED-2B`, and vice versa;
- for every example, the narrative, generating facet, prior BM25 rank,
  facet-only percentile, tethered percentile, selected passage, document ID,
  qrels grade, and final ranks;
- counts of wrong-domain, generic-process, dictionary/Scrabble, essay-writing,
  pet-health, and other previously observed noise patterns in each facet basket;
- facets whose relevant yield rose, fell, or remained zero;
- relevant documents present below rank 500 and the reason they missed the
  facet quota or RRF basket;
- duplicate pressure and quota shortages;
- exact score-cache hits, misses, windows, runtime, and peak memory.

Pattern flags are descriptive. They cannot change rankings or the decision rule.

## Implementation components

Add focused, reusable code beside the existing deep-facet experiment:

1. **Preflight and scorer**
   - validates all sealed inputs and protected topics;
   - renders exact tethered queries;
   - tokenizes without inference and enforces resource ceilings;
   - reuses the existing MiniLM windowing, aggregation, and score cache;
   - writes create-only raw and document-score artifacts.
2. **Two-basket ranker**
   - computes deterministic within-facet percentiles;
   - applies exact facet quotas, duplicate handling, alternation, and fallback;
   - emits full permutations and document-level provenance.
3. **Evaluator and report builder**
   - validates ranking seals and the existing qrels projection;
   - computes frozen metrics and applies the decision rule mechanically;
   - produces machine-readable results and a standalone accessible HTML report.

Reuse existing schemas and helpers where their contracts match. Do not modify
or reinterpret prior immutable outputs.

## Tests and acceptance

Tests must fail before implementation for the new behavior and then verify:

- protected topics fail at every pipeline boundary;
- only the exact four-topic pair manifest can access the score cache;
- tethered query serialization and hashes are deterministic;
- facet-only and tethered pairs have identical topic/facet/document coverage;
- raw logits never cross facet-query boundaries;
- average-rank percentiles, including ties, match the stated formula;
- quota and remainder allocation total exactly 200 for each topic;
- duplicate skipping and shortage redistribution are deterministic;
- input reordering cannot change a ranking;
- the first 100 rows are byte-equivalent in document order to RRF;
- ranks 101--500 contain exactly 200 RRF-basket and 200 facet-basket documents
  unless a recorded, valid exhaustion fallback applies;
- every arm is a complete duplicate-free permutation of `U_accepted`;
- missing inputs, model drift, more than 25,000 windows, or more than ten
  projected minutes stop before inference;
- evaluation cannot open the original qrels and cannot run before sealing;
- the mechanical decision is reproduced exactly from saved metrics;
- rerunning from identical inputs yields identical hashes except for explicitly
  separated runtime telemetry.

Run targeted unit tests and a fixture-sized end-to-end test before the real
preflight. After the bounded local scoring run, rerun the complete targeted test
suite and validate every output hash and row-count invariant.

The standalone HTML must have no external runtime dependency, render on desktop
and mobile, expose the narrative and facet context for examples, distinguish
verified findings from interpretation, and clearly state that it is a
post-qrels diagnostic with no new retrieval. Verify it with Playwright at both
viewports when available.

## Deliverables

- frozen tethered-query and pair manifest;
- inference-free preflight and resource receipt;
- raw tethered MiniLM window scores and aggregated document scores;
- `FACET-2B` and `TETHERED-2B` full rankings with provenance;
- machine-readable metrics, per-topic deltas, diagnostics, and mechanical
  decision;
- source-backed standalone rendered HTML report;
- a short conclusion stating whether narrative tethering reduced facet noise,
  whether the two-basket fusion recovered novel relevant documents, and what a
  later fresh-topic confirmation would need to test.

## Stop boundary

The experiment ends after the diagnostic report. A mechanical pass may support
a separately designed fresh-topic confirmation. A fail requires root-cause
analysis of score behavior, candidate-pool coverage, and quota effects before
any new scorer, fusion rule, or retrieval pass is proposed. No result from this
qrels-exposed diagnostic automatically triggers further inference or search.
