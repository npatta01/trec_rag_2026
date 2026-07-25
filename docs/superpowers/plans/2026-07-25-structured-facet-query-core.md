# Structured Facet Query Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable typed validator and deterministic renderer for externally supplied narrative facet plans, with exact original-narrative fallback.

**Architecture:** A new pure `facet_query_planning` module owns frozen contracts, range resolution, mechanical validation, rendering, and fallback. It consumes the existing `Topic` and produces existing `QueryVariant` records without changing the config-driven pipeline or adding any generator.

**Tech Stack:** Python 3.12, frozen dataclasses, standard-library hashing/regex/Unicode handling, pytest.

## Global Constraints

- Render only from the official `Topic.narrative`; never read or synthesize `Topic.title`.
- Do not add automatic plan generation, prompts, model configuration, transport, CLI, qrels/nuggets access, or experiment artifacts.
- Validate the complete plan before returning any facet query; every failure returns exactly one untouched original-narrative query.
- Keep expansion support optional, lexical-only, bounded, and mechanically auditable.
- Do not integrate structured facets into the runtime pipeline in this change.

---

### Task 1: Typed contracts, range resolution, and structural validation

**Files:**
- Create: `code/trec_rag/facet_query_planning.py`
- Create: `code/tests/test_facet_query_planning.py`

**Interfaces:**
- Consumes: `trec_rag.topics.Topic`, `trec_rag.pipeline_models.QueryVariant`
- Produces: `TokenRange`, `Anchor`, `Expansion`, `CoverageItem`, `Facet`, `FacetPlan`, `FacetPlanValidationError`, `FacetPlanningResult`, `tokenize_narrative()`, `validate_facet_plan()`

- [ ] **Step 1: Write failing contract and range tests**

Add tests constructing frozen typed records and asserting that half-open token
ranges preserve exact narrative text. Parameterize negative, empty, reversed,
duplicate, and out-of-bounds ranges and require `FacetPlanValidationError`.

- [ ] **Step 2: Run the focused tests and verify red**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_facet_query_planning.py -q
```

Expected: collection fails because `trec_rag.facet_query_planning` does not exist.

- [ ] **Step 3: Implement the minimal token tape and typed contracts**

Use frozen dataclasses, a deterministic non-whitespace token tape with Unicode
code-point offsets, SHA-256 narrative identity, and exact range resolution.
Define public count caps as module constants.

- [ ] **Step 4: Add failing structural validation tests**

Cover topic/hash mismatch, duplicate IDs, dangling references, invalid global
and coverage anchor scopes, missing topic/entity global anchor, empty facets,
and coverage assigned zero or multiple times.

- [ ] **Step 5: Implement complete structural validation**

Validate all IDs, ranges, scopes, counts, content-bearing anchors/coverage, and
the exact one-facet coverage partition before returning a validated internal
view. Do not perform I/O or parse untyped JSON.

- [ ] **Step 6: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_facet_query_planning.py -q
```

Expected: all Task 1 tests pass.

### Task 2: Expansion safety, deterministic rendering, and fallback

**Files:**
- Modify: `code/trec_rag/facet_query_planning.py`
- Modify: `code/tests/test_facet_query_planning.py`

**Interfaces:**
- Consumes: validated `FacetPlan`
- Produces: `render_facet_queries(topic: Topic, plan: FacetPlan) -> FacetPlanningResult`

- [ ] **Step 1: Write failing scoped-rendering tests**

Use a two-facet housing comparison. Assert narrative-ordered coverage, inherited
global anchors, applicable coverage anchors without cross-facet leakage,
declared-order expansions, deterministic repeated calls, and no title text.

- [ ] **Step 2: Write failing expansion safety tests**

Reject expansions with out-of-scope anchors, unsupported relations, control
characters, `field:value`, Boolean/query operators, absent ASCII or Unicode
numeric runs, more than three analyzed words, more than three expansion objects,
or more than six new unique content tokens.

- [ ] **Step 3: Write failing all-or-nothing fallback tests**

Mutate one late facet to be invalid and assert the result contains no facet
queries, `used_fallback is True`, a retained error string, and exactly:

```python
QueryVariant(
    topic_id=topic.id,
    variant_name="original",
    query_text=topic.narrative,
    source_type="original_topic",
)
```

- [ ] **Step 4: Implement validation, rendering, and fallback**

Derive inherited anchors from scope, retain only nonredundant safe expansions,
de-duplicate identical normalized components, and join them deterministically.
Catch only `FacetPlanValidationError` at the public fallback boundary.

- [ ] **Step 5: Verify red-green behavior and focused suite**

Run the focused suite, temporarily reverse one expected rendered component to
confirm the test fails, restore it, then rerun:

```bash
.venv/bin/python -m pytest code/tests/test_facet_query_planning.py -q
```

Expected: all tests pass after restoration.

### Task 3: Adjacent documentation and compatibility verification

**Files:**
- Modify: `code/trec_rag/README.md`
- Modify: `docs/superpowers/plans/2026-07-25-structured-facet-query-core.md`

**Interfaces:**
- Consumes: final public API and frozen limits
- Produces: concise inputs, outputs, runtime limits, validation command, evidence boundary, and next experimental gate

- [x] **Step 1: Document the reusable core**

Add a `Structured Facet Query Core` section naming the public records and entry
point, official narrative-only behavior, exact fallback, fixed limits, and
focused validation command. Explicitly state that generation quality remains
experimental and no automatic generator is wired into runtime.

- [x] **Step 2: Run focused and compatibility tests**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_facet_query_planning.py \
  code/tests/test_topics.py \
  code/tests/test_pipeline.py \
  -q
```

Expected: all selected tests pass.

- [x] **Step 3: Run the artifact-independent regression suite**

Run:

```bash
.venv/bin/python -m pytest -q \
  --ignore=code/tests/test_all_topic_tethered_rank.py \
  --ignore=code/tests/test_build_all_topic_tethered_report.py
```

Expected: all selected tests pass. Record separately that the full baseline has
19 pre-existing failures caused by the absent ignored ranking/evaluation bundle.

- [x] **Step 4: Review scope, portability, and secrets**

Inspect `git diff --check`, changed paths, tracked file sizes, forbidden
generator/model/prompt/response/SQLite terms, absolute paths, and likely secrets.
Confirm the diff contains only the new core, tests, and concise design/docs.

- [x] **Step 5: Update this plan with verification evidence**

Check completed steps and append exact commands/results. Do not claim automatic
answer-aspect generation is ready.

#### Task 3 verification evidence (2026-07-25)

- Documentation records the typed inputs and results, narrative-only rendering,
  exact all-or-nothing original-query fallback, frozen caps, and the absence of
  runtime generation/integration. Generation and answer-aspect quality remain
  experimental.
- Focused compatibility command completed successfully:

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_facet_query_planning.py \
    code/tests/test_topics.py \
    code/tests/test_pipeline.py \
    -q
  ```

  Result: `102 passed in 0.21s`.
- Artifact-independent regression command completed successfully:

  ```bash
  .venv/bin/python -m pytest -q \
    --ignore=code/tests/test_all_topic_tethered_rank.py \
    --ignore=code/tests/test_build_all_topic_tethered_report.py
  ```

  Result: `304 passed in 29.73s`.
- Full-baseline caveat, recorded separately: the unignored baseline has 19
  pre-existing failures from the absent ranking/evaluation artifact bundle
  covered by the two ignored tests; this facet core does not create or depend
  on those artifacts.
- Scope/portability/secrets review: `git diff --check` was clean; changed paths
  are limited to the new pure core, its tests, this concise README section, and
  this plan. The code has no runtime generator/model/transport integration,
  checked paths are repository-relative, and no likely credential or absolute
  host-path additions were found. The next promotion experiment remains an
  offline held-out retrieval comparison of externally supplied plans against
  the original narrative; no automatic generator is ready for runtime use.
