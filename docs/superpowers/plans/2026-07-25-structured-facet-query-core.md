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
- Produces: `TokenRange`, `Anchor`, `Expansion`, `CoverageItem`, `Facet`, `FacetPlan`, `ValidatedFacetPlan`, `FacetPlanValidationError`, `FacetPlanningResult`, `tokenize_narrative()`, `validate_facet_plan()`

- [x] **Step 1: Write failing contract and range tests**

Add tests constructing frozen typed records and asserting that half-open token
ranges preserve exact narrative text. Parameterize negative, empty, reversed,
duplicate, and out-of-bounds ranges and require `FacetPlanValidationError`.

- [x] **Step 2: Run the focused tests and verify red**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_facet_query_planning.py -q
```

Expected: collection fails because `trec_rag.facet_query_planning` does not exist.

- [x] **Step 3: Implement the minimal token tape and typed contracts**

Use frozen dataclasses, a deterministic non-whitespace token tape with Unicode
code-point offsets, SHA-256 narrative identity, and exact range resolution. The
public caps include 20 total anchors and 128 Unicode code points per expansion
term alongside the narrower structural limits.

- [x] **Step 4: Add failing structural validation tests**

Cover topic/hash mismatch, duplicate IDs, dangling references, invalid global
and coverage anchor scopes, missing topic/entity global anchor, empty facets,
and coverage assigned zero or multiple times.

- [x] **Step 5: Implement complete structural validation**

Validate all IDs, ranges, scopes, counts, content-bearing anchors/coverage, and
the exact one-facet coverage partition before returning a public frozen
`ValidatedFacetPlan` view. Do not perform I/O or parse untyped JSON.

- [x] **Step 6: Run focused tests**

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

- [x] **Step 1: Write failing scoped-rendering tests**

Use a two-facet housing comparison. Assert narrative-ordered coverage, inherited
global anchors, applicable coverage anchors without cross-facet leakage,
declared-order expansions, deterministic repeated calls, and no title text.

- [x] **Step 2: Write failing expansion safety tests**

Reject expansions with out-of-scope anchors, unsupported relations, control
characters, `field:value`, protected Lucene-like syntax including slash,
Boolean/query operators, absent ASCII or Unicode numeric runs, more than 128
Unicode code points, more than three analyzed words, more than three expansion
objects, or more than six new unique content tokens.

- [x] **Step 3: Write failing all-or-nothing fallback tests**

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

- [x] **Step 4: Implement validation, rendering, and fallback**

Derive inherited anchors from scope, retain only nonredundant safe expansions,
de-duplicate identical normalized components, and join them deterministically.
Catch only `FacetPlanValidationError` at the public fallback boundary.

- [x] **Step 5: Verify red-green behavior and focused suite**

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
- Scope/portability/secrets review, using only the tracked implementation diff
  from `daecd27^` through the working tree:

  ```bash
  git diff --check daecd27^
  git diff --name-only daecd27^
  git diff --numstat daecd27^
  git diff --name-only daecd27^ | while IFS= read -r path; do
    printf '%s ' "$path"
    git cat-file -s ":$path"
  done
  git diff --name-only daecd27^ | rg -n -i \
    '(^|/)(generator|model|prompt|raw[-_]?response|sqlite)(/|$)' || true
  git diff --unified=0 daecd27^ -- \
    code/trec_rag/facet_query_planning.py \
    code/tests/test_facet_query_planning.py | rg -n -i -w \
    'generator|model|prompt|raw[-_ ]?response|sqlite' || true
  git diff --unified=0 daecd27^ -- \
    code/trec_rag/facet_query_planning.py \
    code/tests/test_facet_query_planning.py \
    code/trec_rag/README.md \
    docs/superpowers/plans/2026-07-25-structured-facet-query-core.md | rg -n \
    '(^|[^[:alnum:]_])/(home|Users|etc|tmp|var|opt)(/|$)|[A-Za-z]:[\\/]' || true
  git diff --unified=0 daecd27^ -- \
    code/trec_rag/facet_query_planning.py \
    code/tests/test_facet_query_planning.py \
    code/trec_rag/README.md \
    docs/superpowers/plans/2026-07-25-structured-facet-query-core.md | rg -n -i \
    '(api[_-]?key|client[_-]?secret|password|authorization|bearer|access[_-]?token|auth[_-]?token)[[:space:]]*[:=]|-----BEGIN( [A-Z]+)? PRIVATE KEY-----' || true
  ```

  Output: `git diff --check` produced no output. The changed-path and numstat
  commands reported only `code/trec_rag/facet_query_planning.py` (551 added
  lines), `code/tests/test_facet_query_planning.py` (537 added lines),
  `code/trec_rag/README.md` (61 added lines), and this plan (78 added, 5
  removed). The staged tracked sizes were 19,954, 16,899, 17,850, and 10,397
  bytes respectively. The excluded-path, implementation-content,
  host-absolute-path, and likely-secret scans produced no matches. These scans
  do not read `.env` files or other untracked content. The code has no runtime
  generator/model/transport integration. The next promotion experiment remains
  an offline held-out retrieval comparison of externally supplied plans against
  the original narrative; no automatic generator is ready for runtime use.

### Task 4: Final review fix wave

- [x] Flatten all coverage ranges referenced by a facet and globally order them
  by `(start_token, end_token, coverage_id)`, with an interleaved two-item,
  two-range regression.
- [x] Bound plans to 20 total anchors and expansion terms to 128 Unicode code
  points; cover exact acceptance boundaries and exact original-query fallback
  above each boundary.
- [x] Keep Unicode `M*` combining marks only as continuations of active analyzed
  words, covering decomposed Latin, Hindi, and a leading-mark negative case.
- [x] Reject slash with the protected Lucene-like expansion syntax.
- [x] Reject a non-`Topic` with `TypeError` before entering plan fallback.
- [x] Make `ValidatedFacetPlan` a documented public frozen return type.
- [x] Reconcile the Task 1--2 checkboxes and refresh verification evidence.

#### Final-fix verification evidence (2026-07-25)

- TDD baseline:

  ```bash
  .venv/bin/python -m pytest code/tests/test_facet_query_planning.py -q
  ```

  Result before new tests: `54 passed in 0.03s`.
- RED, after adding the final-review regression and boundary tests with no
  production changes:

  ```bash
  .venv/bin/python -m pytest code/tests/test_facet_query_planning.py -q
  ```

  Result: `8 failed, 56 passed in 0.08s`. Every failure matched a requested
  missing behavior; the exact 20-anchor/128-code-point boundary case passed.
- GREEN, after the minimal fixes and coverage-component compatibility repair:

  ```bash
  .venv/bin/python -m pytest code/tests/test_facet_query_planning.py -q
  ```

  Result: `64 passed in 0.03s`.
- Focused compatibility:

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_facet_query_planning.py \
    code/tests/test_topics.py \
    code/tests/test_pipeline.py \
    -q
  ```

  Result: `112 passed in 0.12s`.
- Artifact-independent regression:

  ```bash
  .venv/bin/python -m pytest -q \
    --ignore=code/tests/test_all_topic_tethered_rank.py \
    --ignore=code/tests/test_build_all_topic_tethered_report.py
  ```

  Result: `314 passed in 20.30s`.
- Compile and safety checks:

  ```bash
  .venv/bin/python -m compileall -q \
    code/trec_rag/facet_query_planning.py \
    code/tests/test_facet_query_planning.py
  git diff --check ef34a98
  git diff --check 0f0d051
  ```

  Result: all commands exited zero with no output.
- The final-fix changed-path, excluded-path, implementation-boundary,
  host-absolute-path, likely-secret, private-type, and tracked-binary scans
  found no unexpected files or matches. The tracked final-fix scope is the core,
  its focused tests, the adjacent README, this plan, and the design. No
  generator, transport, retrieval, or runtime-pipeline integration was added.
