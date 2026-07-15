# Adaptive obligation search v2 design

Date: 2026-07-15

## Decision

Build a separately versioned, bounded discovery-and-search workflow that may
learn one new abstract information need from an existing fixed facet and then
issue one new BM25 request for that accepted need. This is the missing path by
which corpus-derived understanding can add documents that are absent from the
current 8,114-document union.

Discovery v1 remains byte-exact and terminal-only. V2 uses new modules, schemas,
prompts, ledgers, output directories, hashes, and approvals. Nothing may weaken,
reuse as writable state, or overwrite the v1 history.

The primary method is a two-stage local semantic workflow:

1. propose at most one abstract child obligation from one parent/fold reservoir;
2. validate the proposal against independently selected evidence from the
   opposite document fold;
3. freeze at most one child per parent and four per topic;
4. issue one plain-text BM25 request with `hits=1000` per accepted child;
5. score that child's candidates locally with query-specific MiniLM; and
6. add the new candidates to a complete, provenance-preserving adaptive union.

The top 100 and top 1,000 are two diagnostic depths of the same response. V2
never spends a second request merely to change `hits` from 100 to 1,000.

## Verified starting point

The existing qrels-blind baseline contains exactly:

- pilot topics `219`, `72`, `300`, and `84`;
- protected topics `144`, `213`, `224`, `407`, and `515`, rejected at every
  boundary;
- 8,114 accepted topic-document identities;
- four full-narrative queues and 24 fixed O0 facet queues;
- 98,053 MiniLM-scored windows and 96,911 unique query-window pairs;
- complete `NARRATIVE` and `FIXED-O0` rankings containing every document; and
- no available adaptive arm.

The existing rankings have the same document population. Therefore, another
reranking-only discovery pass could improve ordering or passage selection but
could not improve full-pool document recall. New retrieval is required for an
accepted obligation to add genuinely new documents.

The hosted Pyserini client exposes only plain `query` text and integer `hits`.
Repository evidence confirms successful `hits=1000` runs. V2 does not assume
field selection, required-term operators, filters, phrase/slop syntax, boosts,
RM3 controls, or remote BM25 parameter changes.

## Scope

V2 answers this question:

> Can a corpus-derived, cross-validated child obligation retrieve and promote
> relevant documents that the unchanged narrative and fixed O0 queries did not
> surface sufficiently early?

V2 is candidate generation and ordering, not final RAG answer generation.
Answer-fact extraction, evidence cards, and generator context construction stay
out of this design until the candidate mechanism passes.

V2 may add new documents, but it never removes an existing document. The final
adaptive continuation contains the complete baseline union plus every unique
newly retrieved document. Document-count depths are evaluation views, not
eligibility cutoffs.

## Alternatives considered

### A. Deterministic phrase/entity extraction only

This is cheap, reproducible, and useful as a no-LLM control. It is weak at
recognizing paraphrases and abstract information categories, and repeated text
often reflects boilerplate rather than a useful missing need. Retain it as a
diagnostic control, not the primary discovery method.

### B. One semantic proposal pass

One Qwen proposal per parent/fold can recognize abstract needs cheaply, but the
same evidence that suggests a label would also be the only evidence for it.
Exact-span checks catch fabrication, not scope drift or answer facts mislabeled
as obligations. This is insufficient as the promoting method.

### C. Proposal plus opposite-fold semantic validation

This is the selected method. The proposer and validator may use the same pinned
local model, but they see disjoint document folds and different prompts. The
validator cannot inspect the proposing passage. This is evidence independence,
not model independence, and the report must describe it accurately.

## Record model

V2 has only two semantic record types:

- `O0`: a fixed explicit obligation already frozen from the user narrative.
- `O1`: an optional abstract child information need discovered from documents.

V2 does not generate N1 answer nuggets. Mixing facts and search obligations made
v1's schema larger and its semantics harder to validate. Dates, numbers, named
examples, mechanisms, causes, outcomes, and other candidate answers remain
evidence content; they cannot become O1 labels.

Every O1 has exactly one O0 parent and must preserve that parent's topic,
subject, population, domain, and requested relation. It may specialize the
parent but cannot replace, broaden, or contradict it.

## Deterministic evidence units and folds

Rebuild v2 reservoirs directly from the authenticated contract and base MiniLM
scores. Do not consume v1 discovery output as a mutable or authoritative input.

For each of 24 O0 parents and each binary document fold:

1. select the top ten distinct parent-local documents;
2. choose each document's best parent-query MiniLM window;
3. split the exact window into deterministic sentence or list-item units;
4. assign each unit an ID from its topic, parent, fold, document, window, span,
   and text hash; and
5. freeze the ordered units and all source bindings before model execution.

Units contain exact corpus text. The model returns unit IDs rather than copying
long support passages into JSON. This makes support verification exact and keeps
the response small.

Fold assignment remains deterministic from topic and document identity. A
proposal's validation evidence must come from the opposite fold and a distinct
document.

## Proposal stage

There are exactly 48 primary proposal jobs: 24 parents times two folds. Each job
sees only:

- the unchanged narrative;
- one complete parent O0;
- the frozen evidence units for one fold; and
- a compact closed JSON schema.

The response is one object with finite status `SUPPORTED` or `UNSUPPORTED`.
`SUPPORTED` contains exactly one short O1 label, one scope rationale, and one or
two support unit IDs. `UNSUPPORTED` contains a finite reason code and no O1.
Additional keys, prose wrappers, copied passages, confidence scores, or multiple
candidate labels are invalid.

The proposer is pinned to local `Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, deterministic decoding,
temperature `0`, and seed `0`. The prompt forbids outside knowledge and requires
`UNSUPPORTED` when the units do not justify an abstract child need.

Schema, response-token ceiling, primary attempt count, retry ceiling, prompt
hash, model snapshot, tokenizer, input units, expected runtime, and storage must
freeze in an inference-free preflight. The compact unit-ID schema replaces v1's
large multi-record response.

## Validation stage

Deterministic validation runs before another model call. It rejects:

- unknown or mismatched topic, parent, fold, document, window, or unit IDs;
- protected topics;
- support from the proposing fold during validation;
- labels that are empty, copied support passages, or exact/near-exact O0
  duplicates;
- missing source hashes; and
- any schema/status mismatch.

Every surviving proposal becomes one opposite-fold validation job. The job sees
the narrative, parent O0, proposed O1 label, and only the opposite-fold units.
It does not see the proposing units or the proposer's rationale.

The validator returns exactly one finite decision:

- `SUPPORTED` with one or two opposite-fold support unit IDs;
- `NO_EVIDENCE`;
- `OUT_OF_SCOPE`;
- `ANSWER_FACT`;
- `DUPLICATE_O0`; or
- `WRONG_DOMAIN`.

An O1 is eligible only when the proposal and opposite-fold validation each bind
to an exact unit from different documents. No model confidence value is used.

When both folds yield eligible proposals for one parent, choose deterministically
by larger distinct support-document count, shorter normalized label, normalized
label, then proposal ID. Accept at most one O1 per parent. Across a topic, rank
by distinct support-document count, parent manifest order, normalized label, and
proposal ID; accept at most four O1 records per topic.

The no-LLM control emits repeated two-to-five-token content phrases across
distinct documents and folds. It passes through the same scope, duplication,
and acceptance limits but cannot promote the semantic method.

## Model call ledger and retry policy

Proposal and validation are separately approved stages.

Proposal preflight declares:

- 48 primary jobs;
- at most one retry per job, only when the raw completion reaches the frozen
  token ceiling and cannot be parsed as one complete JSON value;
- therefore 48 primary calls and a worst-case ceiling of 96 proposal calls.

After proposals freeze, validation preflight declares the exact number `V` of
surviving jobs, where `0 <= V <= 48`, and a worst-case ceiling of `2V` validation
calls under the same single truncation-retry rule. When `V > 0`, validation
preflight loads exactly one authenticated tokenizer-only runtime to freeze the
prompt-token count for every validation job; it loads no model weights and
records zero inference. When `V = 0`, it loads neither tokenizer nor model.

Validation execution is a separate approval-first production boundary. Its
approval binds the validator role, exact model ID and revision, model-snapshot
manifest, tokenizer identity, prompt counts, schema, code and job hashes, call
ceilings, and create-only ledger destination. The executor replays the complete
proposal and contract chain before constructing the pinned runtime. An approved
zero-job run creates and seals the empty ledger without constructing a model.

No retry is allowed for invalid semantics, schema violations that are not
truncation, unsupported results, runtime/model mismatch, or operator error.
There is no recursive repair prompt.

Every attempt is recorded append-only with stage, job ID, attempt ordinal,
request hash, start/finish state, raw completion hash, token counts, elapsed
time, outcome code, and error class. A started attempt can never disappear from
the ledger. Stage receipts report both planned ceilings and actual calls.

No proposal or validation inference begins until its exact preflight is reviewed
and separately approved.

## Search-query rendering

Each accepted O1 produces one focused plain-text BM25 query by concatenating,
without generic headings or query-language operators:

1. the parent's frozen anchor terms in their recorded order;
2. the complete parent O0 text; and
3. the accepted O1 label.

Remove only exact repeated terms/phrases while preserving first occurrence.
This keeps subject, population, domain, and relation context attached to the
new vocabulary without reintroducing every unrelated clause of the broad
narrative. The full unchanged narrative is retained for subsequent semantic
scoring, not the BM25 request. The exact query text and SHA-256 freeze before
cache inspection or network access.

## Bounded BM25 retrieval

Retrieve once per accepted O1 with `hits=1000`. At most four accepted O1 records
per topic over four topics yields:

- at most 16 primary retrieval requests;
- at most 16,000 raw candidate rows before deduplication; and
- top-100 and top-1,000 diagnostic views from the same responses.

Before retrieval, an exact preflight records accepted O1 count, query strings,
request identities, exact cache hits/misses, maximum external attempts, timeout,
rate-limit state, expected payload/storage, and zero qrels access.

Use the persistent limiter at no more than one request start per three seconds
and a 120-second request timeout. A transport retry policy, if enabled, must be
declared in the retrieval preflight with its exact additional-attempt ceiling;
it may cover only timeout, connection reset, HTTP 429, or HTTP 5xx. HTTP 4xx,
invalid JSON, response-schema failure, and semantic-quality concerns are not
retryable. Every external attempt is appended to the raw-first ledger before a
normalized candidate artifact can be published.

Retrieval, even when fully cached, requires a separate run approval after the
preflight. No query is generated from retrieved O1 results, so the workflow has
exactly one adaptive search generation and is not recursive.

## Candidate normalization and union

Normalize document ID, rank, raw BM25 score, text, text hash, query hash, parent
O0, accepted O1, response hash, and request provenance. Raw BM25 scores order
only one response and never cross queries.

Deduplicate by `(topic_id, document_id)`. Preserve every provenance event when a
document is already in the baseline union or appears under multiple O1 queries.
New documents append to the adaptive candidate population; no baseline document
is removed.

Reports must distinguish:

- baseline documents rediscovered by O1;
- genuinely new document IDs;
- unique contribution at top 100 and top 1,000 per O1; and
- duplicates across accepted O1 queries.

## Query-local MiniLM scoring

For each accepted O1, score the windows of its own top-1,000 BM25 candidates
against a frozen semantic query containing the unchanged narrative, complete
parent O0, and accepted O1 label using
`cross-encoder/ms-marco-MiniLM-L6-v2` revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`.

Preflight first computes exact documents, windows, unique pairs, global-cache
hits, misses, runtime estimate, device, and disk use. Cache identity includes
model revision, tokenizer, query hash, document hash, window span, and
aggregation contract. Any uncached inference requires separate approval.

MiniLM scores order only their O1 queue. Raw or normalized scores never cross
the narrative, O0, or other O1 query boundaries.

## Adaptive continuation

Create three decision arms after all sources and scores freeze:

- `NARRATIVE`: the already sealed full-narrative baseline;
- `FIXED-O0`: the already sealed fixed-facet baseline; and
- `ADAPTIVE-V2`: the complete fixed baseline union plus every new O1 document.

`ADAPTIVE-V2` begins each topic with one distinct narrative document, one from
every O0, and one from every accepted O1. Remaining queues use deterministic
token-deficit scheduling. O1 is bounded: it receives its coverage slot and may
schedule candidates only until its top-1,000 retrieved population is exhausted;
it cannot create another child or recurring query. Every remaining baseline and
new document is retained in a deterministic tail.

If discovery, validation, retrieval, or scoring is incomplete, ADAPTIVE-V2 is
`unavailable` with a finite terminal reason. It is never emitted as an empty
successful ranking. NARRATIVE and FIXED-O0 remain valid independently.

## Evaluation and success criteria

All requests, responses, score shards, continuations, hashes, and a freeze seal
must exist before qrels or review labels can open.

Primary comparisons are ADAPTIVE-V2 versus FIXED-O0 at:

- nDCG@10;
- graded Recall@100;
- graded Recall@1,000;
- judged relevant documents anywhere in the complete adaptive union;
- number of genuinely new judged-relevant document IDs; and
- per-topic regressions.

Supporting diagnostics include judged rate, candidate counts, O1 unique
contribution, source diversity, O0/O1 coverage slots, and direct inspection of
top passages for each accepted O1. The report must state that incomplete qrels
can undercount the value of newly retrieved documents.

Adaptive discovery demonstrates candidate value only if it adds at least one
judged-relevant document absent from FIXED-O0 or produces a positive aggregate
graded Recall@100 or Recall@1,000 delta, while preserving the complete baseline
population. It promotes as the default ordering only when aggregate nDCG@10 is
no more than `0.02` below FIXED-O0 and no topic regresses by more than `0.10`.
If it adds candidate value but misses the ordering rule, retain its new-document
basket as a downstream RAG candidate source while keeping FIXED-O0 as the
primary order.

Failure to add judged-relevant documents does not prove the new documents are
irrelevant when judged rate is low. In that case, report the mechanism as
inconclusive and require blinded assessment rather than tuning on qrels.

## Components and isolation

Implement v2 in focused modules rather than expanding the terminal v1 file:

- `adaptive_obligation_v2_contract.py`: source bindings, exact units, folds,
  schemas, prompts, job manifests, and preflight receipts;
- `adaptive_obligation_v2_propose.py`: proposal execution and append-only call
  ledger;
- `adaptive_obligation_v2_validate.py`: deterministic checks, opposite-fold
  jobs, acceptance, and freeze;
- `adaptive_obligation_v2_retrieve.py`: query freeze, cache audit, rate-limited
  raw-first retrieval, normalization, and union;
- `adaptive_obligation_v2_score.py`: O1 MiniLM preflight, resumable local score
  completion, and verification; and
- `adaptive_obligation_v2_rank.py`: complete adaptive continuation and freeze.

Each public stage supports `preflight` or `verify` without executing the next
stage. Execution commands fail unless the exact approved preflight identity is
supplied. V1 exposes only its existing `verify` and `inspect` commands.

## Implementation boundary approved now

The next implementation plan may authorize only:

- v2 record types, schemas, deterministic unit extraction, fold/reservoir
  construction, prompts, compact renderers, finite status validation, ledgers,
  protected-topic checks, create-only receipts, and tests;
- proposal-stage inference-free preflight with exact 48 primary jobs and retry
  ceiling;
- deterministic retrieval request rendering and an inference/network-free cache
  audit; and
- verification tooling and an updated rendered explanation artifact.

It does not authorize Qwen generation, external BM25 requests, new MiniLM
inference, qrels, review-label access, or final promotion. Each requires the
separate frozen preflight and approval described above.

## Failure and safety invariants

- Protected topics fail before source, cache, model, endpoint, qrels, or output
  access.
- Every stage is create-only and rejects existing, partial, unexpected, or
  symlinked outputs.
- Every artifact and source is bound by path, byte count, row count, SHA-256,
  schema version, and aggregate identity.
- Raw outputs and actual-attempt ledgers precede normalized or accepted records.
- No score crosses a query boundary.
- No O1 becomes answer truth merely because a model proposed it.
- No accepted O1 lacks exact support in distinct documents across folds.
- No stage silently selects another model, endpoint, query depth, retry policy,
  or topic.
- No adaptive result generates another adaptive search.
- V1 remains terminal-only and immutable.
