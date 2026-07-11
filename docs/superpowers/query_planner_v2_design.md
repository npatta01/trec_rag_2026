# Query planner v2: exact references, code-owned invariants

## Goal and decision boundary

Keep the useful architecture—global lexical expansion plus facet-specific
sparse retrieval—while removing failure modes that should not depend on a
language model. The model proposes semantic request units, anchor applicability,
facet grouping, and lexical variants. Python owns exact text resolution,
inheritance, priorities, partition checks, budgets, filtering, provenance, and
fallback.

The v1 `gpt-oss-20b` diagnostic is a formal no-go. V2 is a new model-selection
benchmark, not a repair that changes that result.

### v2.1 decoder-compatibility amendment

The first synthetic transport smoke was rejected before inference because
vLLM 0.24 does not implement JSON Schema array `uniqueItems`. The preserved
incident is documented under
`reports/experiments/query_planner_v2_synthetic_smoke_001/`.

V2.1 removes only decoder-side `uniqueItems`; Python's `_v2_ids` validator still
rejects every duplicate reference, so accepted-plan semantics do not change.
Because the prompt embeds the exact schema, the payload version is bumped to
`query_plan_v2_1` and the prompt to `sparse_query_planner_v6`. An offline linter
and pinned-container compiler manifest are mandatory before another HTTP call.
Renderer, tokenizer, analyzer, thresholds, and the smallest-sufficient model
protocol remain frozen.

## 1. Frozen tokenizer and typed ranges

Python tokenizes the exact original narrative into an auditable JSON token
tape. Array position is the token ID. Each token record contains its exact text
and Unicode-code-point start/end offsets.

Tokenizer contract:

- normalization: none before offset calculation;
- offset unit: Python Unicode code points, not UTF-8 bytes;
- lexical token: Unicode letters/numbers with internal straight apostrophes,
  curly apostrophes, or hyphens when followed by another lexical character;
- punctuation: every remaining non-whitespace character is its own token;
- whitespace: not a token, but preserved between token offsets;
- version: `narrative_token_tape_v1`.

The run records the tokenizer version, narrative SHA-256 over UTF-8 bytes,
token count, normalization rule, offset unit, and complete token tape.

The model returns typed ranges only:

```json
{"start_token": 4, "end_token": 6}
```

Start is inclusive and end is exclusive. Python resolves a range from the
first token's character start through the last included token's character end.
It rejects empty, reversed, out-of-bounds, and duplicate ranges. Lists are
sorted; adjacent ranges are merged; genuinely discontinuous ranges remain
separate and are never represented as one quote. The model never recopies
source text.

The user message contains both the unmodified narrative and the JSON token
tape. The narrative is data, not instructions.

## 2. Minimal model output

The constrained output surface is:

```text
schema_version
topic_id

anchors[]
  anchor_id
  range {start_token, end_token}
  kind
  scope: global | coverage
  coverage_refs[]

coverage_items[]
  coverage_id
  source_span_refs[]

facets[]
  facet_id
  coverage_refs[]
  expansion_terms[]

global_expansion.terms[]
```

The schema does not contain model-authored `information_need`, priority,
facet count, complexity class, dependency edges, facet anchor lists, copied
source text, or provenance labels.

Python derives count and complexity, labels every explicit request core,
resolves all audit text, and derives provenance. Any human-readable facet label
is generated in code from resolved ranges plus inherited anchors. It is never
fed to retrieval, repair, validation decisions, or downstream controllers.

## 3. Atomic coverage and exact facet partition

One coverage item represents one predicate/question target, metric,
comparison, association, or constraint-bearing request. Coordinated targets
must be split unless they form an indivisible comparison or relationship.

Mechanical rules:

- each coverage item contains one or two sorted, nonoverlapping exact ranges;
- each facet references one or two coverage items;
- every coverage item is referenced by exactly one facet;
- every facet is nonempty;
- all IDs are unique and all references resolve;
- the number of facets remains adaptive from one through eight.

The two-item facet cap blocks large mega-facets but does not prove coverage
atomicity. Coverage remains model-segmented, mechanically resolved, and
semantically audited. The frozen topic checklist maps every request unit to a
distinct coverage item. Two items may share a facet only when the same passages
and retrieval vocabulary are likely to answer both; blind reviewers penalize
both overgrouping and overfragmentation.

## 4. Scoped anchors and code-derived inheritance

Anchors use exact typed ranges and one of the frozen semantic kinds: entity,
topic, relation, comparison, geography, time, population, metric, constraint,
or modality.

- `scope="global"` requires an empty `coverage_refs` array.
- `scope="coverage"` requires one or more valid coverage references.
- Every facet inherits every validated global anchor plus every coverage-scoped
  anchor attached to its coverage items.
- No model-authored facet anchor list exists.

Python therefore owns resolution and inheritance, while the model still
proposes semantic applicability. Blind review verifies that applicability: the
schema alone cannot prove that credit-union, Dubai, UK, youth, comparison, or
association scope was assigned correctly.

## 5. Conservative lexical expansion

Each model-proposed expansion is:

```text
term
relation: alias | acronym | technical_term | common_variant
anchor_refs[]
```

The broad `neutral_search_term` relation is removed. Python derives whether an
exact term is narrative text; aliases may introduce a lexical form for an
anchored referent but may not introduce a new referent.

Frozen mechanical policy:

- at most three analyzed words per proposed term;
- at most three term objects per facet;
- at most six newly added unique analyzed tokens per facet;
- global new-token allowance `min(8, ceil(0.10 * N))`, where `N` is the
  original narrative's unique analyzed content-token count, plus a separate
  object cap;
- each retained term adds at least one token relative to the base query and
  all previously retained terms;
- reject empty terms, control characters, query operators/field syntax,
  duplicate analyzed forms, and numeric Unicode variants absent from the
  narrative;
- prohibit new entities, dates, facts, examples, mechanisms, causes, effects,
  or candidate answers.

The ledger preserves proposed, retained, and rejected terms plus deterministic
rejection reasons. Dropping an unsafe term protects retrieval but does not make
the planner safe: any prohibited proposal is a planner hard failure. Redundant
proposals are dropped and penalized by the semantic expansion-quality field.
Python never pads a short query automatically; a model-proposed term counts only
when it is independently useful.

## 6. Deterministic rendering, analyzer, and fallback

```text
global query = exact original narrative + retained global expansion terms

facet query = exact coverage ranges in narrative order
            + derived global anchors
            + derived coverage-scoped anchors
            + retained facet expansion terms
```

Components are normalized and deduplicated deterministically. Planner budgets
use the pinned `lucene_default_english_v1` reference analyzer: the inspected
Anserini `DefaultEnglishAnalyzer` chain implemented with Lucene 10.4.0. Its
complete fingerprint is preflighted and stored before any model request.

This is a **planner-budget reference analyzer**, not a claim about the exact
hosted ClimbMix deployment. The hosted service exposes neither its build nor an
analyzer fingerprint, so the run records
`index_id="hosted_climbmix_unknown_revision"`. The 5–25 gate therefore means
tokens under the frozen reference analyzer. It keeps both model arms comparable;
later retrieval evaluation tests the actual rendered strings against the hosted
system and is the evidence for retrieval effectiveness.

Facet queries must contain 5–25 unique analyzed content tokens. The only >25
exception is exactly one resolved source range whose range alone exceeds 25;
anchors or expansions cannot create the exception. The five-token floor stays
frozen for this benchmark so topic 515 does not cause a post-hoc threshold
change.

On an invalid live plan or exhausted diagnostic repair, retain the failure and
fall back deterministically to original-narrative retrieval. Never partially
render an invalid plan.

## 7. Smallest-sufficient model protocol

Model size is not an objective. The first v2 arm is the already-local
`openai/gpt-oss-20b`, because the new schema removes several v1 failures and a
controlled baseline is necessary to tell whether a larger planner adds value.
Its exact revision/runtime remain pinned; reasoning effort and chat-template
settings are frozen after a non-evaluation smoke. The v2.1 tokenizer-only
preflight on the local 8,192-token server measured 2,530–2,672 prompt tokens for
the five diagnostics, so the run ceiling is 5,400 output tokens (120 tokens of worst-case
context margin). The harness rejects any prompt/output request that exceeds the
reported server context and records near-ceiling output as an operational
warning.

Run known diagnostic topics 144, 213, 224, 407, and 515 exactly once. Preserve
exact HTTP bytes and commit immutable first outcomes before validation. At most
one validator-guided repair may be run for diagnosis; repair never contributes
to the model gate.

If the local 20B model achieves 5/5 mechanical and renderer validity, at least
4/5 accepts, mean at least 10/12, no field zero, and no safety hard failure,
freeze it for holdouts and do not download a larger model.

Only if the 20B arm misses that diagnostic threshold, run
`Qwen/Qwen3.5-35B-A3B` official BF16 revision
`59d61f3ce65a6d9863b86d2e96597125219dc754` as a local, sequential, text-only,
direct/non-thinking challenger under the identical v2 schema, renderer, term
policy, known topics, and blinded rubric. Promote the larger model only if it:

- meets the full diagnostic threshold;
- adds no safety failure;
- wins at least three of five paired topic comparisons;
- improves mean score by at least 0.6/12;
- is not more than one point worse on any topic.

For paired scoring, a mechanically valid plan with no safety hard failure beats
an invalid or hard-failed plan. When both are valid and safe, the higher locked
12-point score wins and equal scores tie. Invalid or hard-failed first outcomes
count as 0/12 only for calculation of paired mean improvement. Plans from both
arms are anonymized and shuffled before paired review.

If Qwen3.5 does not show that material gain, stop model escalation and improve
the non-model segmentation/rendering strategy instead; neither model advances.
Qwen3.6 and `gpt-oss-120b` are out of scope for this milestone unless the user
explicitly opens a later challenger study.

The selected model is frozen before holdout selection. A bounded validator
repair is the only agent-like loop at this stage; a retrieval agent remains
deferred until sparse retrieval exists and can be diagnosed from observable
signals.

## 8. Preregistered evaluation

### Known diagnostics

The five v1 topics are for schema debugging and paired model selection only.
They cannot support a fresh generalization claim.

### Five-topic untouched pilot gate

Select five unused topics by a preregistered narrative-only stratified rule:
length, compoundness, domain, and constraint types, with a fixed seed for tie
breaking. Before any output, freeze token-range request-unit checklists, schema,
prompt, tokenizer/analyzer, renderer, model revision/runtime/chat template,
thinking setting, max tokens, sampling, term policy, rubric, reviewer identities,
adjudication, and thresholds.

Two reviewers independently score anonymized first outcomes; disagreements are
adjudicated by the frozen procedure. Any viewed holdout is burned after any
prompt, schema, model, renderer, analyzer, or threshold change.
If the selected model fails this pilot, those topics are burned and Qwen must
not be evaluated on them. Any later challenger requires a fresh pilot set.

Pilot promotion thresholds:

- 5/5 mechanically and renderer valid first outcomes;
- at least 4/5 semantic accepts;
- mean score at least 10/12;
- complete-need coverage = 2 and scope preservation = 2 for every topic;
- no rubric field scored zero;
- no prohibited proposed expansion.

These are pilot thresholds, not a 99% or population-level claim.

### Broader mechanical audit

After the five-topic pilot passes, run a first-output mechanical and term-safety
sweep over all 12 remaining unused development narratives. A future 99%
validity claim requires at least 100 untouched outputs; the 119 live topics can
supply that operational audit.

Report separately:

- planner-envelope validity;
- plan/reference validity;
- renderer validity;
- proposed-term safety;
- retained and rejected expansions;
- finish reason;
- latency and token counts;
- blinded semantic scores.

Only after the pilot and broader development sweep pass do we run BM25 for the
original narrative, global expansion, and facet variants and fuse their
candidate pools.

## 9. What this design does not claim

- It does not make dense full-corpus retrieval affordable.
- It does not make expansion or decomposition sufficient without fusion and
  later reranking.
- It does not use qrels to choose facets, vocabulary, or model versions.
- It does not treat model confidence, output length, or validator success as a
  relevance judgment.
