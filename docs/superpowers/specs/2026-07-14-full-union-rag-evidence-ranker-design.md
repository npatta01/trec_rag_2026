# Full-union adaptive RAG evidence ranker design

Date: 2026-07-14
Revised: 2026-07-14 after two independent advisor reviews

## Decision

Preserve the complete accepted candidate union and build a deterministic,
passage-level evidence ranker. The primary ranker uses explicit scope constraints,
cross-fitted corpus discovery, query-local cross-encoder ordering, nugget novelty,
and token-deficit fairness. It does not use a hand-weighted composite score.

All 8,114 topic-document rows remain eligible. There is no 100, 500, 1,000, or
other document-count eligibility cutoff. Candidate eligibility and generator
context are different concerns: every candidate remains available, while the
selected evidence is evaluated at nested 8,000, 16,000, and 32,000 evidence-token
prefixes per topic.

The four pilot topics contain roughly 46.8 million raw document tokens in
aggregate, or 11.7 million per topic. A generator cannot consume that complete
text directly. Passage extraction and evidence selection are therefore required
even though the candidate population remains complete.

This is a post-qrels mechanism diagnostic on topics `219`, `72`, `300`, and `84`.
It can justify a later fresh-topic validation, not production promotion.
Protected IDs `144`, `213`, `224`, `407`, and `515` remain hard-rejected at every
boundary.

## Why this is a ranker, not a score sort

A cross-encoder supplies a learned local relevance prior for one query-passage
pair. It cannot by itself decide whether a finite evidence set:

- covers every explicit user need;
- adds a new supported fact rather than repeating old evidence;
- preserves broad narrative evidence while adding narrow evidence;
- avoids wrong-domain, duplicate, or single-source flooding;
- allocates context fairly across differently sized information needs; or
- leaves enough context for the final answer.

The evidence ranker makes those set-level decisions. Cross-encoder scores order
passages only inside the queue for the query that produced the score. Scores,
percentiles, BM25 values, and learned logits never cross query boundaries.

Four qrels-exposed topics cannot identify reliable weights for a learned or
hand-tuned listwise ranker. The primary method therefore encodes priorities as
testable constraints and lexicographic decisions. The earlier weighted formula
is retained only as a non-promoting sensitivity arm.

## Scope and success criterion

The experiment asks:

> Can the complete original-plus-facet candidate union produce a compact,
> directly supportable, nonredundant evidence set that covers both the explicit
> narrative and useful in-scope information discovered in the corpus?

Success requires passage-level direct support and distinct nugget coverage, not
merely document relevance or a favorable nDCG cutoff. Document metrics remain
diagnostics.

The deliverable is a RAG-ready evidence packet and a rendered analytical report.
The repository currently implements only placeholder answer generation, so this
experiment does not claim final answer quality. Substantive citation-first
generation follows after the selector passes.

## Obligation and evidence model

Maintain three separate record types:

- `O0`: an explicit obligation derived only from the user narrative. It is
  mandatory, normative, and cannot be removed or demoted by corpus evidence.
- `O1`: an optional corpus-derived sub-obligation. It must specialize exactly one
  `O0` while preserving the original subject, population, domain, and relation.
- `N1`: an atomic evidence nugget or proposition cluster. It is answer content,
  not new user intent.

The hierarchy is:

```text
unchanged narrative (BROAD)
└── explicit obligation O0
    ├── optional derived sub-obligation O1
    │   ├── evidence nugget N1
    │   └── evidence nugget N1
    └── evidence nugget N1
```

A date, number, named example, event, mechanism, cause, outcome, or specific
claim may be retained as an `N1`; it cannot become an `O1` merely because it was
retrieved. Corpus absence never redefines an `O0`.

## Frozen inputs

The experiment consumes these existing immutable artifacts:

- the complete `U_accepted` union from the deep-facet pilot;
- original and accepted-facet retrieval provenance;
- the sealed RRF, GLOBAL, DUAL, cascade, and Mixedbread rankings;
- existing MiniLM facet-local, narrative, and common-query window scores;
- existing Mixedbread residual scores;
- the already opened projected development qrels for later diagnostics.

The input population is exactly 8,114 accepted topic-document rows. Preserve
every document ID, text hash, stream rank, score identity, title, URL/host when
available, and source seal.

No new retrieval request, endpoint call, paid call, dense primary retrieval, or
query rewrite occurs during the selector audit. Qrels, organizer nuggets, and
reference answers are unavailable to obligation discovery and packet creation.

## Passage representation

Reuse the existing deterministic tokenizer and passage-window contract where its
sealed provenance is available:

- pair maximum: 512 tokens;
- query maximum: 192 tokens;
- minimum passage budget: 256 tokens;
- overlap: 64 passage tokens;
- retain the final non-empty short window;
- preserve document ID, character span, token span, and text hash.

The finite evidence packet may include at most three passages from one document.
This is a diversity safeguard, not a length judgment. The all-passages
continuation retains every deferred passage and is not subject to this packet cap.

For this selector experiment, token budgets use the frozen Qwen3 tokenizer over
the exact serialized passage text plus source title and URL/host metadata when
present. System instructions, the user narrative, and future answer tokens are
outside this evidence-only count. Evidence cards are derivative audit artifacts;
the later generation design will re-budget the final serialized context using
the chosen generator's tokenizer.

Document length is processing and context cost only. It is not a positive or
negative relevance feature. Multiple overlapping windows cannot inflate support
or coverage.

When verified title, URL, and host metadata exist, bind them to the passage
identity and expose them to diagnostics. Scoring input may include title and host
context, but missing metadata cannot make a passage ineligible.

## Pass 0: freeze explicit scope

Freeze before corpus inspection:

- `BROAD`: the unchanged complete narrative;
- all 24 accepted explicit obligations `O0`;
- each `O0` query as `unchanged narrative + explicit obligation`;
- subject, population, relation, domain, and wrong-domain rules;
- passage windows and source identities;
- query-specific scoring identities;
- document fold assignment:
  `SHA256(topic_id || document_id) mod 2`.

Scoring a narrative-tethered obligation avoids the naked-facet drift observed in
earlier results. Every `O0` must be collectively complete, individually useful,
and nonredundant with the other explicit obligations. This validation can merge
duplicate `O0` records but cannot introduce external answer content.

The execution order after this freeze is:

1. complete `BROAD` and `O0` score coverage;
2. run the one cross-fitted discovery pass;
3. score accepted `O1` records inside their parent populations;
4. freeze queues and build every selection arm;
5. freeze packets and evidence cards before evaluation.

## Cross-fitted corpus-discovery pass

Fixed narrative obligations are scope axes, not a complete inventory of answer
nuggets. Permit exactly one corpus-inspection pass.

For every `O0` and each document fold:

1. Take the top ten qualified, nonduplicate passages from distinct documents.
2. Balance retrieval provenance when equivalent qualified passages are available.
3. Extract candidate abstract `O1` records and atomic `N1` signatures.
4. Preserve exact supporting spans, document IDs, parent `O0`, and proposer fold.
5. Render each proposed `O1` as a provisional `narrative + O0 + O1` query and
   score only the opposite fold of its parent population.
6. Require a qualified opposite-fold passage with an exact validated support span
   from a distinct document. Repeat A-to-B and B-to-A.

An `O1` is accepted only when it:

- has exactly one `O0` parent;
- preserves the parent's subject, population, domain, and relation;
- specializes rather than replaces the parent;
- has support in at least two nonduplicate documents across the two folds;
- survives wrong-domain and source-duplication checks;
- is distinct from all `O0` and accepted `O1` records; and
- describes an abstract information category rather than a candidate answer.

Rank accepted proposals lexicographically by validating-document count,
independent retrieval-stream count, source diversity, parent-local rank, and
canonical label. Accept at most one `O1` per parent and four per topic. These are
frozen complexity safeguards, not learned relevance weights.

An `N1` may remain a singleton when an exact passage directly supports it.
Singleton nuggets do not become obligations. This preserves rare facts without
allowing one document to redefine the user's request.

Each `N1` contains normalized subject, relation, and object fields plus its exact
support span. Merge exact normalized triples first, then records whose stemmed
content-term sets have token-set Jaccard at least `0.80`. Blinded review separately
adjudicates semantic paraphrases that this lexical rule misses. Report automatic
and adjudicated novelty separately.

The primary semantic proposer is the already materialized
`Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, run locally with temperature `0`,
seed `0`, a frozen JSON schema, and no tools or network. It sees only one parent's
ten passage records from one fold and must return exact support spans. Its output
is a proposal, not a judgment: deterministic scope checks and independent
opposite-fold evidence decide acceptance. A no-LLM repeated-phrase extractor is
the discovery control: it emits contiguous two-to-five-content-token phrases that
occur in at least two nonduplicate documents across the folds, then applies the
same parent and scope checks. If the exact Qwen materialization or preflight
fails, FIXED-O0 still runs and ADAPTIVE is reported `discovery_unavailable`; no
substitute model is selected silently.

No `O1` or `N1` issues a new search request. There is no recursive discovery.

## Score coverage and completion

### Audit first

Build a coverage matrix for every passage-query identity required by the frozen
arms. Reuse a cached score only when model revision, tokenizer, query hash,
document hash, span, and aggregation identity match exactly. Missing scores are
reported as missing and never imputed from another query.

### Local completion

Use the already materialized
`cross-encoder/ms-marco-MiniLM-L6-v2` revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`.

Complete only these score populations:

- `BROAD`: every accepted document's passages against the unchanged narrative;
- `O0`: passages from documents retrieved by that obligation, against
  `narrative + O0`;
- `O1`: passages from the parent `O0` candidate population, against
  `narrative + O0 + O1`.

Do not score the full document-by-all-obligations Cartesian product. Every
document remains eligible through `BROAD`; parent-local populations prevent
generic or long documents from receiving repeated opportunities across unrelated
facets.

Preflight freezes the exact pair count, cache hits, cache misses, runtime estimate,
device, and disk footprint. There is no download, network request, or paid
inference. Scores are written as resumable, create-only shards with full
provenance.

MiniLM is the cheap exhaustive passage scorer. Existing Mixedbread scores remain
a stronger but incomplete diagnostic. A bounded identical-passage Mixedbread
comparison is allowed only if the audit later shows that MiniLM ordering—not
candidate discovery or selection—is the remaining bottleneck.

## Passage qualification

A within-query rank always has a winner, even when every passage is bad. Therefore
rank alone never proves support.

Before a passage can serve an `O0` or `O1`, require:

- subject anchor or audited morphological variant;
- relation, population, and domain coherence;
- no independent wrong-domain failure;
- an exact non-empty support span;
- no selected overlapping or near-duplicate passage; and
- fewer than three already selected packet passages from its document.

If none qualifies, emit an explicit `unsupported` sentinel. Do not substitute a
generic passage to manufacture coverage.

Essay-writing, dictionary, Scrabble/word-game, homework/template, generic-process,
and known wrong-domain patterns remain warnings and audit fields. A source warning
cannot reject a passage by itself and is not converted to a numeric penalty.

For host diversity, prefer an unseen host within the current obligation before a
second passage from a previously selected host whenever a qualified alternative
exists. Missing host metadata receives neither a bonus nor a penalty.

## Selection arms

All arms begin with the same complete candidate union and passage inventory. Each
arm emits one budget-independent continuation. The 8k packet is a complete prefix
of 16k, which is a complete prefix of 32k. After the finite packets, append every
deferred passage deterministically so all 8,114 document candidates remain
represented.

### NARRATIVE: semantic relevance control

Use one `BROAD` queue containing every passage, ordered only by narrative MiniLM
score. Apply the same overlap, near-duplicate, host-diversity, document-cap, and
token-fit rules as the primary arm.

For arm-obligation auditing only, assign a NARRATIVE passage to an `O0` when the
frozen qualification rules say it supports that obligation. This label does not
alter NARRATIVE ordering.

### FIXED-O0: explicit-coverage control

Create `BROAD` plus one queue per explicit `O0`. Order each queue only by its own
query-local cross-encoder score.

Select the best qualified `BROAD` passage, then one qualified passage per `O0` in
manifest order. Mark unsupported obligations explicitly. Fill the remaining
continuation using qualified token-deficit round robin over `BROAD` and `O0`.

### ADAPTIVE: recommended obligation-and-novelty ranker

Create `BROAD`, `O0`, and accepted `O1` queues.

#### Coverage floor

1. Select the best qualified `BROAD` passage.
2. Select one qualified passage per `O0` in manifest order.
3. Only after every `O0` is covered or marked unsupported, select at most one
   dedicated passage per validated `O1`.

An `O1` never receives equal recurring allocation with explicit needs.

#### Novelty-aware token-deficit fill

For `BROAD` plus every non-exhausted `O0`, track selected evidence tokens.
Repeatedly:

1. Choose the obligation with the fewest assigned tokens.
2. Break ties with `BROAD` first, then frozen `O0` manifest order.
3. Within that queue, prefer the highest-ranked qualified passage carrying an
   unseen supported `N1` signature.
4. If no unseen `N1` remains, take the next highest-ranked qualified passage.
5. Prefer an unseen host for the obligation when a qualified alternative exists.
6. Charge the passage's complete token count to its primary obligation.
7. Skip overlap, near-duplicate, token-fit, and document-cap violations.

This is max-min-fair evidence allocation by tokens. Cross-encoder scores determine
relevance inside a need; the ranker determines coverage and novelty across needs.

Each passage has one primary obligation in the promoting arm. A parameter-free
multi-obligation sensitivity may credit several obligations only when the passage
is the top qualified passage in every queue it claims.

### COMPOSITE: non-promoting sensitivity

Retain the previous weighted relevance/coverage/redundancy formula only to show
whether its output differs materially. It cannot promote the system, set a
threshold, or change the primary selector. Bind it exactly to design commit
`1aacea7`:

```text
0.40*marginal_facet + 0.25*narrative + 0.15*coherence
+ 0.10*retrieval_prior + 0.10*strongest_facet
- 0.20*redundancy - 0.05*source_warning - 0.05*token_cost
```

## Redundancy and evidence atomization

Near-duplicate detection uses lowercased, stemmed, non-stopword token sets and
token-set Jaccard:

- below `0.80`: no duplicate finding;
- `0.80` or above: defer the later passage from the finite packet;
- overlapping spans from the same document: defer after the first.

The threshold is a frozen structural parameter, not a relevance weight. Deferred
passages remain in the complete continuation.

After packet selection, convert each selected passage into one or more extractive
evidence cards containing:

- parent `O0` and optional `O1`;
- atomic `N1` label;
- exact quoted support span;
- document ID, title, URL/host when available, and passage offsets;
- whether the card is new support, corroboration, or a competing perspective.

Preserve raw passages before card creation so atomization cannot erase rare
evidence. A card is invalid if its claimed nugget is not supported by its exact
span or if the span does not address the attached obligation. The blind review
evaluates both directions.

## Evaluation

### Qrels-backed document diagnostics

At each token budget report:

- unique cited documents;
- binary and graded qrel gain captured;
- relevant documents introduced only by facets;
- original-query relevant evidence retained;
- judged rate and grades `2`, `3`, and `4` separately;
- evidence gain per 10,000 tokens.

Retain nDCG at document ranks 10, 100, 500, and 1,000 only for continuity and
catastrophic-regression diagnosis.

### Blind passage and card audit

Create 72 explicit arm-obligation audit slots: one slot for each of the 24 `O0`
records from NARRATIVE, FIXED-O0, and ADAPTIVE at 16k tokens. A slot contains the
first packet passage claimed to support that obligation or an explicit
`unsupported` sentinel. Randomize passage-bearing slots deterministically and
hide the arm.

Review labels are:

- direct support;
- topical mention only;
- wrong domain or population;
- usable source quality;
- redundant with selected evidence;
- obligation or obligations supported;
- nugget supported by the exact span;
- distinct new nugget, corroboration, or no new information.

Separately review every accepted `O1` for parent compatibility and every selected
`N1` for direct support. Report inter-reviewer agreement when a second reviewer
is available. The proposer or scorer cannot be the sole judge of its own output.

### Primary evidence metrics

- explicit `O0` supported coverage;
- validated `O1` coverage;
- distinct supported nuggets per 10,000 tokens;
- worst-obligation support;
- direct-support precision;
- exact-span card validity;
- wrong-domain and source-warning rates;
- passage and host redundancy;
- original-evidence retention;
- novel-facet-document retention;
- evidence tokens per supported obligation and nugget.

## Advancement and failure diagnosis

ADAPTIVE may advance to fresh-topic validation only if all are true at 16k or
32k tokens:

- no topic loses explicit-`O0` supported coverage versus FIXED-O0;
- no accepted `O1` is wrong-domain or parent-incompatible;
- direct-support precision is no more than five percentage points below FIXED-O0;
- at least three of four topics gain one distinct valid supported nugget;
- distinct supported nuggets per token improve;
- redundancy does not increase;
- original-query graded evidence retention is at least 90% of FIXED-O0; and
- reviewer agreement is adequate to interpret the difference.

Stop and retain FIXED-O0 when discovery produces no valid `O1`, derived categories
are generic or duplicative, adaptive slots displace explicit support, gains appear
only in document qrels, or reviewer disagreement is too high.

Diagnose failures mechanically:

- **Required `O0` has no supporting candidate passage:** candidate-discovery gap.
- **Support exists but ranks poorly:** passage-scoring gap; compare identical
  passages with a stronger available reranker.
- **Support ranks well but is not selected:** ranker-policy gap.
- **Packet is strong but cards are invalid:** evidence-atomization gap.
- **Cards are strong but the answer is poor:** generation or citation gap.

Only a verified candidate-discovery gap may open a later retrieval-repair design.
That design may issue one narrative-tethered repaired query for an unsupported
`O0`, with a frozen request budget, drift check, and marginal-gain stop rule. It
does not run automatically in this selector experiment.

## Historical grounding and differences

The components are supported by TREC RAG 2025 precedents:

- WING-II used submodular evidence selection, host diversity, and evidence cards.
- TUS decomposed narratives and selected multi-viewpoint keystone documents.
- MITLL retained narrative context when using independent subquestions.
- GenAIus and HLTCOE extracted and organized atomic nuggets.
- GRILL used Retrieve -> Expand -> Refine with document feedback and gap analysis.
- uogTr's explicit decomposition outperformed its Search-R1 iterative agent run.

This design is a disciplined synthesis rather than a claim of an entirely new
algorithm. Its useful differences are complete-union eligibility, sparse-first
candidate generation, narrative-tethered query-local scoring, cross-fitted
one-pass corpus discovery, weight-free token fairness, explicit unsupported
states, and sealed failure diagnosis.

## Output artifacts

Create a versioned output directory containing:

- input bindings, source seals, passage inventory, and metadata audit;
- frozen `BROAD` and `O0` records;
- document-fold assignments;
- proposed, validated, rejected, and frozen `O1`/`N1` records with support spans;
- score-coverage audit and optional local scoring preflight/shards/receipt;
- NARRATIVE, FIXED-O0, ADAPTIVE, and COMPOSITE continuations;
- nested 8k, 16k, and 32k packets;
- extractive evidence cards;
- qrels-backed diagnostics;
- blinded review packet and labels;
- mechanical advancement decision;
- advisor review of the frozen method and findings.

Update the standalone HTML report under
`reports/experiments/deep_facet_candidate_pilot_v1/`. Explain in plain language:

- why no candidate cutoff is used;
- what evidence-token budgets mean;
- the difference between explicit obligations, derived sub-obligations, and
  evidence nuggets;
- how corpus feedback is cross-validated and bounded;
- why cross-encoder scores do not form the final ranker;
- whether selected passages add distinct support;
- whether failures come from retrieval, scoring, selection, atomization, or
  generation;
- what remains before an end-to-end answer can be judged.

Render and verify the report at desktop and mobile viewports. It must remain
self-contained and accessible without an expiring localhost server.

## Safety and reproducibility

- Reject protected IDs before input reads, joins, scoring, evaluation, or report
  generation.
- Do not access a new topic, qrels, organizer nugget, or reference answer.
- Hash every query, document, passage, proposal, score, queue, packet, card, and
  metric.
- Never compare raw or normalized scores across query boundaries.
- Never discard a document from the complete accepted union.
- Use one corpus-discovery pass, no new search, no recursion, and no self-selected
  goal or stopping rule.
- Use no model download, hosted inference, paid call, or external request.
- Preflight and shard any required local inference.
- Preserve unrelated working-tree files unchanged.

## Test requirements

Tests must verify:

- the sealed population contains exactly 8,114 topic-document rows;
- every document remains represented in each complete continuation;
- protected topics fail before source or score access;
- `O0` uses only explicit narrative scope;
- every `O1` has one compatible `O0` parent and cross-fold corroboration;
- specific answer facts cannot become `O1` records;
- singleton `N1` records remain eligible but cannot become obligations;
- discovery executes once and cannot issue retrieval;
- the semantic proposer uses only the frozen local Qwen revision, temperature,
  seed, schema, fold inputs, and exact spans;
- proposer failure cannot select a substitute model or block FIXED-O0;
- automatic and adjudicated nugget novelty remain distinct metrics;
- cached scores match exact model, query, text, span, and metadata hashes;
- each score orders only its own queue;
- every document is eligible through `BROAD`;
- facet and derived scores stay inside their parent candidate population;
- `unsupported` is possible even when a queue is nonempty;
- source warnings cannot reject alone;
- token-deficit ordering is deterministic;
- unseen valid nuggets and hosts precede repeats within an obligation;
- 8k is a prefix of 16k and 16k a prefix of 32k;
- packet document caps do not affect the complete continuation;
- overlap and near-duplicate behavior is deterministic;
- cards retain exact support spans and source IDs;
- COMPOSITE cannot promote or change the primary output;
- advancement decisions reproduce from saved artifacts;
- report claims reproduce from sealed outputs;
- desktop and mobile rendering pass accessibility checks.

## Deferred end-to-end RAG work

The current `code/trec_rag/generation.py` intentionally produces placeholder
output and cannot establish answer quality. After the evidence ranker passes, a
separate design will compare substantive citation-first generation over identical
evidence packets. It will evaluate usefulness, explicit and derived coverage,
claim support, citation validity, and hallucination. Keeping generation separate
prevents a weak generator from obscuring whether candidate discovery and evidence
selection were actually fixed.
