# Facet Retrieval-Control Pilot Design

## Objective

Determine whether the controls available through the hosted Pyserini REST API
can reduce noise in four problematic facet streams without losing the useful
facet-specific evidence they contribute. This phase improves BM25 candidate
generation only. It does not run a cross-encoder or introduce recursive search.

The hosted service is external and cannot be modified. Its deployed OpenAPI
contract exposes plain query text, hit depth, `k1`, and `b`, but not phrase,
required-term, exclusion, field, filter, or explicit boost syntax. The default
Anserini bag-of-words generator weights repeated query terms by their query term
frequency, so controlled anchor repetition is the only available per-term
weighting mechanism.

## Scope and safety

- Evaluate topics `200`, `225`, and `707` only.
- Hard-reject protected topics `144`, `213`, `224`, `407`, and `515` at planning,
  request construction, cache access, fusion, evaluation, and reporting.
- Preserve all v2.1 artifacts and all earlier immutable retrieval ledgers.
- Reuse the existing cached R1 result for each baseline query.
- Permit at most 12 new external search attempts: three variants for each of
  four streams. Failed attempts count against the ceiling.
- Pace every request through the persistent limiter at one request start per
  ten seconds, burst one, with redirects and automatic retries disabled.
- Preserve every failed attempt. Continuing after a failure requires an
  explicit new run rather than an automatic retry.
- Freeze manifests, request identities, raw responses, candidate lists, and
  rankings before evaluation.
- The earlier pilot already exposed projected development qrels, so this is a
  preregistered follow-up rather than a fully blind experiment. The frozen
  design and rankings prevent further tuning against those qrels.

## Selected streams

The four streams represent distinct observed failure modes while excluding the
less severe duplicate warning stream `200/f05a`.

| Stream | Observed problem | Existing R1 query | Reweighted query |
|---|---|---|---|
| `200/f07a` | Essay and wrong-event drift | `Holocaust enduring effects European Jews` | `Holocaust Holocaust enduring effects European European Jews Jews` |
| `225/f02` | Low-quality video-game pages | `violent video games exposure desensitization violence research` | `violent video games violent video games exposure desensitization violence research` |
| `225/f04` | Generic aggression and domain drift | `aggressive behavior children risk factors psychology` | `aggressive aggressive behavior children children risk factors psychology` |
| `707/f02` | Dog-health drift | `sorbitol human health adverse effects safety` | `sorbitol sorbitol human human health adverse effects safety` |

The reweighted strings introduce no new concepts. They repeat only the existing
subject or population anchors and retain the existing facet relation terms.

## Experiment arms

Each stream has one cached control and three new arms:

| Arm | Query | `k1` | `b` | Purpose |
|---|---|---:|---:|---|
| `B0` | Existing R1 | 0.9 | 0.4 | Cached control |
| `W0` | Reweighted | 0.9 | 0.4 | Isolate anchor repetition |
| `W1` | Reweighted | 0.4 | 0.4 | Reduce the advantage of repeated terms in documents |
| `W2` | Reweighted | 0.4 | 0.0 | Also remove document-length normalization |

Both `k1` and `b` must be sent together for every new W request. Its request and
cache identity must include query text, hit depth, endpoint, index, analyzer
fingerprint, `k1`, and `b`; results from different scoring settings must never
share an identity. B0 reuses the verified legacy cache identity and is bound to
effective defaults `k1=0.9` and `b=0.4` by the control manifest plus its prior
request, response, and candidate hashes.

All arms retrieve depth 100. No query is rewritten after observing its results.

## Inspection and evaluation

Inspect ranks 1–10 for every stream arm using the existing deterministic
coherence, domain-drift, and content-quality diagnostics. Extend reporting to
show both top-five and top-ten counts. Content-quality detection remains a
warning rather than a relevance judgment, but it participates in variant
selection alongside qrels-backed evidence.

For each stream arm, report:

- anchor and anchor-plus-intent coherence at 5 and 10;
- wrong-domain and low-quality counts at 5 and 10;
- overlap with the original-narrative top 100;
- relevant documents at 10 and 100 at grade at least 2;
- graded Recall@100;
- nDCG@10;
- unique grade-at-least-2 documents contributed beyond the original-narrative
  top 100; and
- unique graded gain beyond the original-narrative top 100.

Select one arm per stream mechanically:

1. Reject an arm if it fails anchor coherence or increases both domain drift
   and content-quality noise relative to `B0`.
2. Among remaining arms, maximize unique graded gain beyond the original query.
3. Break ties by graded Recall@100, then nDCG@10, then lower combined drift and
   content-quality count.
4. If still tied, prefer the least changed arm in order `B0`, `W0`, `W1`, `W2`.

This ordering protects facet-specific coverage first and top-rank quality
second. Qrels are used only after every arm and selection rule has frozen.

## Repaired system comparison

Before opening qrels, freeze every topic-level R2 alternative produced by
replacing only the four tested R1 streams while leaving all other R1 streams
unchanged. This produces four alternatives for topic 200, sixteen combinations
for topic 225, four alternatives for topic 707, and the unchanged R1 ranking
for topic 897. Fuse each alternative with the existing family-balanced weighted
RRF (`k=60`, depth 100, original family weight 0.5, facet family weight 0.5
divided across active facets).

The evaluator must not reopen live retrieval ledgers or caches. Version the
control freeze as `facet-control-ranking-freeze-v2` and make it self-contained
for marginal-value metrics with two additional canonical artifacts:

- `candidate_streams.json`, recording exact stream/query/request lineage,
  expected depth, row count, and the per-stream row hash; and
- `candidates.jsonl`, recording only schema version, topic, stream, arm, query
  hash, rank, and document ID in canonical order.

The snapshot contains exactly 20 depth-100 leaf streams (2,000 rows): the four
unchanged original-query streams used by the topic rankings plus B0, W0, W1,
and W2 for each of the four registered facet streams. It omits document text
and retrieval scores. Copy original and B0 rows from the exact hash-verified
in-memory candidates used by the freezer, retaining prior-freeze and
request/cache lineage; W rows retain the verified control-run lineage. Bind
both snapshot files, every per-stream hash, the prior freeze, inspections,
fusion definition, and all 25 ranking hashes transitively into `freeze.json`.
Validate the exact stream set, ranks 1--100, unique document IDs, query hashes,
and protected-topic exclusion before publication. Preserve v1 artifacts
unchanged; evaluation requiring marginal stream metrics must fail clearly on a
v1 freeze rather than migrate it in place.

After evaluation selects one arm per stream, define `R2` as references to the
matching already-frozen topic rankings. Do not perform retrieval, fusion, or
ranking construction after qrels access. Aggregate metrics are calculated from
the four selected frozen topic rankings.

Compare `R2` with frozen `O`, `F0`, and `R1`. Report aggregate and per-topic
nDCG@10, graded Recall@100, Recall@100, precision@10, relevant documents at 10,
and judged rates. Also report the marginal relevant and graded contribution of
each selected facet.

Classify the retrieval repair as successful only if:

- aggregate graded Recall@100 is greater than R1;
- aggregate nDCG@10 is no more than 0.02 below R1;
- no topic loses more than 0.10 nDCG@10 versus R1; and
- the selected arms do not increase combined qrels-free drift and content noise
  across the four streams.

A failure does not prove sparse candidate generation is unusable. It means
these available REST-level controls did not solve the facet-noise problem. A
cross-encoder remains a separately costed follow-up and must compare
original-only candidates with facet-augmented candidates so its incremental
value is measurable.

## Implementation boundaries

Add a focused retrieval-control pilot rather than changing the frozen R0/R1
manifests or earlier experiment outputs:

- a versioned four-stream manifest containing B0/W0/W1/W2 definitions;
- request and cache identities that bind BM25 parameters;
- a raw-first, rate-limited runner with a 12-request ceiling;
- an inspector/evaluator that reproduces arm selection and R2 fusion; and
- a standalone rendered HTML report showing queries, representative results,
  per-stream contributions, system metrics, and the final decision.

Reuse the existing retrieval ledger, persistent rate limiter, exact cache,
inspector, fusion, freeze, and evaluation utilities wherever their contracts
already fit.

## Verification

Tests must verify:

- exact query strings and the four-stream boundary;
- no new vocabulary beyond controlled repetition;
- exact B0/W0/W1/W2 `k1` and `b` settings;
- BM25 parameters appear in request and cache identities;
- both BM25 parameters are sent together;
- the 12-request ceiling and ten-second limiter floor;
- no redirects, retries, or recursive rewrites;
- protected-topic rejection at every boundary;
- deterministic inspection and selection under input reordering;
- a self-contained v2 candidate snapshot with exactly 20 streams and 2,000
  rows, verified before qrels access;
- all 25 topic-level R2 alternatives freeze before qrels access;
- R2 changes only the four selected streams;
- RRF family weights total exactly 1.0; and
- the HTML report reproduces saved metrics and decisions.
