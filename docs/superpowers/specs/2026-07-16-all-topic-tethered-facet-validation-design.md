# All-Topic Tethered-Facet Validation Design

**Status:** approved for implementation on 2026-07-16

## Objective

Determine whether narrative-tethered, facet-aware candidate ordering is neutral
or better than the existing family-balanced RRF baseline on every one of the 22
development topics. The experiment is a retrospective full-development stress
test. It is not evidence of generalization to unseen topics.

## Authorized scope

The experiment uses exactly these topic IDs:

`14, 31, 37, 58, 72, 84, 144, 161, 200, 213, 219, 224, 225, 233, 273, 300, 407, 477, 499, 515, 707, 897`

The user explicitly authorized the previously protected IDs `144`, `213`,
`224`, `407`, and `515` for this experiment only. Existing protected-topic
guards and historical artifacts remain unchanged. New code must use an explicit
experiment-specific allowlist and must not weaken shared protection constants.

## Why a new uniform experiment is required

The original narrative BM25 top-1,000 is fully cached for all 22 topics. Exact
current facet and MiniLM artifacts exist for only `72`, `84`, `219`, and `300`.
Eight other topics have historical top-100 facet experiments built with
different planners and scoring definitions, while ten topics have no usable
facet plan. Combining those artifacts would confound topic effects with pipeline
differences.

The validation therefore freezes one manifest schema, facet-rendering contract,
retrieval depth, MiniLM query construction, score normalization, ranking logic,
and evaluation contract for all 22 topics.

## Candidate generation

### Original basket

Reuse the exact cached original-narrative top-1,000 candidates. Verify request
identity, query hash, depth, candidate count, and raw-response provenance. No new
original-query request is permitted.

### Facet baskets

Use a single audited planner contract for every topic:

- each facet represents one explicit informational obligation;
- every query carries the topic subject, applicable population/domain, and the
  requested relation;
- bridge terms are allowed only for domain disambiguation, population binding,
  relation paraphrasing, or a standard name for an explicitly requested concept;
- bridge terms record source, purpose, scope rationale, analyzer output, and why
  they are not candidate answers;
- unsupported facts, causes, outcomes, examples, dates, and narrower subtopics
  are rejected;
- naked abstract queries and duplicated coverage obligations are rejected;
- facet order and tie-breaking are deterministic.

Retrieve each accepted facet to depth 200. This matches the four-topic experiment
being replicated. The original narrative remains depth 1,000.

Depth 200 is not assumed sufficient forever. After evaluation, report relevant
and unique-candidate yield by facet-rank buckets `1-50`, `51-100`, `101-150`, and
`151-200`. Material continuing yield in `151-200` supports a separately frozen
top-1,000 facet-depth follow-up. It must not silently alter this confirmatory run.

All new retrieval uses the existing exact-identity cache and persistent limiter:
one request start per three seconds, raw response saved first, and no automatic
retry after an immutable failed attempt. Retrieval cannot start until the facet
manifest and exact request count are frozen.

## MiniLM scoring

Use the same pinned MiniLM model revision, tokenizer, windowing, and aggregation
as the four-topic tethered experiment. For every accepted topic-document pair,
freeze these within-query features:

- full narrative score;
- common/global score used by the existing DUAL objective;
- narrative-plus-originating-facet score for each facet basket that retrieved
  the document.

Raw scores never cross query boundaries. Convert only within-query ranks to the
same deterministic percentile features used by the current implementation.
Tokenizer-only preflight must freeze exact document-query pair and window counts
before model loading. Model inference is local; hosted and paid inference remain
zero.

## Ranking arms

Every arm is a complete permutation of the identical per-topic candidate union.
Freeze all arms before opening qrels:

1. `RRF`: existing family-balanced RRF baseline.
2. `RRF100-STATIC-DUAL`: exact current suggestion; preserve RRF ranks 1-100,
   then append the existing DUAL permutation computed from empty coverage state.
3. `RRF100-STATIC-DUAL-NR`: sensitivity without the lexical-redundancy term.
4. `RRF100-REINIT-DUAL`: seed facet coverage and redundancy state from the
   protected RRF prefix, then greedily rank the residual union.
5. `RRF100-REINIT-DUAL-NR`: the reinitialized no-redundancy sensitivity.
6. `RRF500-REINIT-DUAL`: safety arm preserving RRF ranks 1-500 exactly before
   reinitialized residual ordering.

The current `RRF100-STATIC-DUAL` arm is the primary confirmatory comparison.
The remaining arms are preregistered confirmatory alternatives in this fixed
selection order:

`RRF500-REINIT-DUAL`, `RRF100-REINIT-DUAL`,
`RRF100-REINIT-DUAL-NR`, `RRF100-STATIC-DUAL-NR`.

An alternative may be selected only if the primary fails, it independently
passes every promotion rule, and its paired significance test remains below
`0.05` after Holm correction across the five non-baseline arms. This ladder is
frozen now so an attractive post-qrels arm cannot be promoted opportunistically.

## Evaluation firewall and metrics

Before qrels access, save hashes for:

- the 22-topic authorization receipt;
- narratives, facets, bridge-term provenance, analyzer version, and query hashes;
- retrieval manifests, raw responses, and candidate lists;
- model identity, tokenizer/window plan, score records, and cache identities;
- union membership, normalization rules, DUAL coefficients, protected depths,
  tie-breaks, and every complete ranking.

Only after successful verification may the pinned projected development qrels
be opened at relevance grade `>=2`.

Report per topic and aggregate:

- unique known-relevant document count and binary recall at depths 100, 250,
  500, 1,000, 1,500, and full union;
- graded recall, nDCG, precision, and judged rate at the same finite depths;
- normalized recall AUC through the complete union;
- facet-only known-relevant retention;
- union size and full-union recall ceiling;
- win/tie/loss counts, worst regression, macro delta, pooled delta, paired
  bootstrap confidence interval, and paired sign-flip/randomization result;
- facet yield by the four facet-rank buckets.

Because unjudged documents are treated as nonrelevant, every conclusion must say
`known-relevant` and show judged-rate changes. Full-union recall is identical for
all complete-permutation arms and is a candidate-generation ceiling, not a
ranking win.

## Promotion decision

An arm is eligible only if all conditions hold:

- zero topic losses versus RRF in known-relevant count at depth 1,000;
- positive pooled and macro recall delta at depth 1,000;
- at least eight topic wins at depth 1,000;
- lower bound of the paired 95% topic-bootstrap interval is above zero, or the
  Holm-adjusted exact paired randomization test is below `0.05`;
- no topic regression at depths 250 or 500;
- exact identity with RRF through its protected prefix;
- no judged-rate collapse that invalidates interpretation.

If no arm passes, retain RRF. Correct implementation may still merge even if the
ranking hypothesis fails. Promotion, candidate-generation conclusions, and
final RAG answer quality are separate decisions.

## Deliverables

- versioned all-topic facet prompt/contract and manifest;
- exact retrieval and MiniLM preflight receipts;
- raw-first retrieval ledgers and cached candidate lists;
- authenticated MiniLM score artifacts;
- sealed complete rankings for all six arms;
- qrels-backed all-topic regression metrics and diagnostics;
- a self-contained, mobile-accessible HTML report in the repository;
- a sanitized rendered copy under `/home/npatta01/codex-rendered/plans/` and a
  verified private tailnet-only portal link;
- independent code/method review and an explicit merge/promotion recommendation.
