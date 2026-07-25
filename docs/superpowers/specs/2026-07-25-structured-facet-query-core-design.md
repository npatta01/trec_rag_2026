# Structured Facet Query Core Design

## Objective

Add a small reusable core that accepts a typed, externally supplied answer-aspect
plan for one official TREC topic narrative, validates its mechanical safety, and
renders deterministic facet queries. The core does not generate plans and does
not claim that any model can produce good plans.

## Evidence boundary

The reference branch separates two facts:

- deterministic range resolution, reference validation, rendering, and fallback
  are testable mechanics;
- automatic plan quality is unvalidated. The GPT-OSS V1 smoke accepted zero
  plans, and V2.1 transported schema-valid JSON that failed plan validation.

This change preserves only the first category. It excludes model prompts,
transports, caches, CLIs, raw responses, experiment databases, qrels, organizer
nuggets, and retrieval integration.

## Considered approaches

1. **Pure typed core (selected).** Add frozen records and pure functions beside
   the current query-understanding code. This is small, testable, and cannot
   accidentally promote a generator into runtime use.
2. **Pipeline configuration integration.** Teach the YAML pipeline to load plan
   files. This would require file schemas, provenance policy, configuration, and
   downstream variant semantics before plan quality is validated.
3. **Extract the historical planner.** Split the 3,081-line reference module.
   This risks importing model, transport, prompt, and experiment concerns and
   would not produce a clean mergeable change.

## Public API and records

Create `trec_rag.facet_query_planning` with frozen records:

- `TokenRange(start_token, end_token)`: half-open range over a deterministic
  narrative token tape.
- `Anchor(anchor_id, token_range, kind, scope, coverage_refs)`: an exact
  narrative anchor. Scope is `global` or `coverage`.
- `Expansion(term, relation, anchor_refs)`: an optional lexical-only query term.
- `CoverageItem(coverage_id, source_ranges)`: one answer obligation represented
  by one or two exact narrative ranges.
- `Facet(facet_id, coverage_refs, expansions)`: one independently useful
  retrieval intent.
- `FacetPlan(topic_id, narrative_sha256, anchors, coverage_items, facets)`.
- `FacetPlanningResult(queries, used_fallback, error)`.

`render_facet_queries(topic, plan)` returns a result containing existing
`QueryVariant` records. Successful names are `facet:<facet_id>` with source type
`structured_facet`. An invalid plan returns exactly one `original` variant with
source type `original_topic` and the exact `topic.narrative`.

The API deliberately accepts `Topic` for compatibility but never reads
`Topic.title`. TSV-derived or synthesized titles therefore cannot affect query
text.

## Validation

Validation is fail-closed and precedes all rendering:

- topic ID and SHA-256 must match the official narrative;
- record counts and identifier formats are bounded;
- ranges must be nonempty, in bounds, unique within a coverage item, and
  content-bearing;
- anchor ranges are exact and contain no more than eight content tokens;
- IDs are unique and every reference resolves;
- one to four global anchors are required, including a `topic` or `entity`
  anchor;
- global anchors have no coverage references; coverage-scoped anchors have
  valid nonempty coverage references;
- every coverage item appears in exactly one nonempty facet, and each facet
  references at most two coverage items;
- each facet has at most three optional expansions;
- expansions use an allowed lexical relation, reference only anchors inherited
  by that facet, contain one to three analyzed words, introduce no absent
  numeric run, control character, field syntax, or query operator, and add at
  most six unique content tokens per facet.

There is no partial success. Any validation or rendering error discards the
whole plan and returns the original narrative.

## Deterministic rendering

For each facet in declared order:

1. resolve its coverage ranges and order them by position in the narrative;
2. append global anchors and applicable coverage-scoped anchors in narrative
   order;
3. append nonredundant expansion terms in declared order;
4. de-duplicate identical normalized components without changing the first
   occurrence;
5. join components with one space.

This renders from `Topic.narrative` only. No competition title, qrels, nuggets,
retrieved content, or generated answer text is available to the core.

## Runtime limits and next experiment

The core validates supplied plans in memory and performs no I/O, network calls,
model loading, retrieval, or retries. Limits are 16 coverage items, 8 facets,
4 global anchors, 2 ranges per coverage item, 2 coverage items per facet, 8
content tokens per anchor, 3 expansions per facet, 3 analyzed words per
expansion, and 6 new unique expansion tokens per facet.

Before runtime promotion, a separate experiment must generate plans for a
frozen development set and untouched holdouts, pass every mechanical gate, then
receive blinded human/IR review for coverage, faithfulness, and query utility.
Only accepted plans should proceed to retrieval comparison against the original
narrative baseline.

## Verification

Focused tests cover token ranges, partitioning, scoped anchors, deterministic
rendering, expansion safety and budgets, exact fallback, and title independence.
Existing topic, query-understanding, and pipeline tests guard compatibility.
