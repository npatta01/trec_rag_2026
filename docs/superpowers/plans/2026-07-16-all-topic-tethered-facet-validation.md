# All-Topic Tethered-Facet Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a uniform 22-topic retrospective validation that determines whether protected, narrative-tethered facet ordering introduces any topic-level recall regression versus RRF.

**Architecture:** A new experiment-specific contract authorizes exactly 22 development topics without changing shared protected-topic guards. Reusable planner, retrieval, scoring, ranking, evaluation, and report modules consume small authenticated artifacts; every ranking freezes before the evaluator can open qrels. Candidate retrieval remains top-1,000 for narratives and top-200 per facet, with an explicit depth-saturation diagnostic rather than an unplanned deeper search.

**Tech Stack:** Python 3.12, pytest, existing Pyserini remote retrieval ledger and persistent limiter, `cross-encoder/ms-marco-MiniLM-L6-v2`, ROCm PyTorch, NumPy, standard-library JSON/SQLite/HTML, headless Chrome, and the private Tailscale Serve portal.

## Global Constraints

- Work only in `/home/npatta01/.codex/worktrees/41f9/trec_rag_2026` on `codex/structured-query-planner`.
- Preserve every immutable historical artifact and do not modify shared protected-topic constants.
- New experiment scope is exactly `14,31,37,58,72,84,144,161,200,213,219,224,225,233,273,300,407,477,499,515,707,897`.
- Authorization of `144,213,224,407,515` applies only inside the new all-topic experiment namespace.
- Original narratives retrieve at depth 1,000 from the exact existing cache; facets retrieve at depth 200.
- No qrels access until manifests, retrieval, scores, unions, parameters, and every ranking are sealed and verified.
- Raw BM25 and MiniLM scores never cross query boundaries; only deterministic within-query percentile features may be combined.
- New retrieval uses one request start per three seconds, raw-first persistence, exact-identity caching, and no automatic retry after a failed immutable attempt.
- Tokenizer-only preflight must freeze exact MiniLM work before model load or inference.
- No paid call, hosted inference, download, dense primary retrieval, or answer-generation evaluation.
- Every arm is a complete permutation of the identical per-topic union.
- Report all 22 per-topic outcomes; aggregate gains cannot hide a regression.
- Treat unjudged documents as unknown in prose and use `known-relevant` for qrels-backed counts.
- Keep unrelated untracked sparse-relevance files untouched and out of every commit.
- Render only sanitized derived HTML through the existing tailnet-only portal; never use Funnel or a public listener.

---

### Task 1: Freeze the all-topic experiment contract and facets

**Files:**
- Create: `code/trec_rag/all_topic_facet_contract.py`
- Create: `code/tests/test_all_topic_facet_contract.py`
- Create: `reports/experiments/all_topic_tethered_facet_validation_v1/facet_prompt.md`
- Create at run time: `outputs/all_topic_tethered_facet_validation_v1/planning/`

**Interfaces:**
- Consumes: the 22 development narratives and audited historical facet definitions as reference material.
- Produces: `ALL_TOPIC_IDS`, `validate_authorized_scope(topic_ids)`, `validate_facet_manifest(payload)`, `build_request_plan(payload, original_cache)`, and sealed `manifest.json`, `authorization.json`, `request_plan.json`, and `SEALED.json`.

- [ ] **Step 1: Write failing scope and manifest tests**

```python
def test_scope_is_exactly_the_authorized_22() -> None:
    assert validate_authorized_scope(ALL_TOPIC_IDS) == ALL_TOPIC_IDS
    with pytest.raises(ValueError, match="outside all-topic authorization"):
        validate_authorized_scope((*ALL_TOPIC_IDS, "999"))


def test_shared_protected_constants_are_not_weakened() -> None:
    assert {"144", "213", "224", "407", "515"} <= PROTECTED_TOPIC_IDS


def test_facet_requires_tethered_subject_domain_and_relation() -> None:
    facet = _facet(query="impact", anchor_terms=[], relation_terms=[])
    with pytest.raises(ValueError, match="tether"):
        validate_facet_manifest(_manifest(facet))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q code/tests/test_all_topic_facet_contract.py`

Expected: collection fails because `trec_rag.all_topic_facet_contract` does not exist.

- [ ] **Step 3: Implement the experiment-specific contract**

Define immutable constants and validate exact scope before any source reader runs:

```python
ALL_TOPIC_IDS = (
    "14", "31", "37", "58", "72", "84", "144", "161", "200", "213",
    "219", "224", "225", "233", "273", "300", "407", "477", "499",
    "515", "707", "897",
)
ORIGINAL_DEPTH = 1000
FACET_DEPTH = 200
REQUEST_INTERVAL_SECONDS = 3.0
```

Validate obligation uniqueness, analyzer terms, subject/domain/relation tethering,
bridge-term provenance, deterministic order, query hashes, three-to-nine accepted
facets per topic unless the narrative contains fewer explicit obligations, and
zero qrels fields. Seal the authorization receipt before reading topic sources.

- [ ] **Step 4: Author and validate one 22-topic facet manifest**

Render every facet through the rules in
`reports/experiments/all_topic_tethered_facet_validation_v1/facet_prompt.md`.
Reuse historical wording only when it passes the new validator; otherwise render
the obligation again. Save exact analyzer output and bridge provenance. Run:

```bash
.venv/bin/python -m trec_rag.all_topic_facet_contract freeze \
  --topics trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv \
  --output outputs/all_topic_tethered_facet_validation_v1/planning
.venv/bin/python -m trec_rag.all_topic_facet_contract verify \
  --planning outputs/all_topic_tethered_facet_validation_v1/planning
```

Expected: exactly 22 authorized topics, top-1,000 original cache hits for all 22,
zero original retrieval requests, an exact top-200 facet request count, no qrels
binding, and a verified planning seal.

- [ ] **Step 5: Run focused and compatibility tests**

Run: `.venv/bin/python -m pytest -q code/tests/test_all_topic_facet_contract.py code/tests/test_deep_facet_candidate_gate.py code/tests/test_sparse_relevance_manifest.py`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add code/trec_rag/all_topic_facet_contract.py \
  code/tests/test_all_topic_facet_contract.py \
  reports/experiments/all_topic_tethered_facet_validation_v1/facet_prompt.md
git commit -m "Add all-topic facet validation contract"
```

---

### Task 2: Retrieve and authenticate the uniform candidate union

**Files:**
- Create: `code/trec_rag/all_topic_facet_retrieve.py`
- Create: `code/tests/test_all_topic_facet_retrieve.py`
- Create at run time: `outputs/all_topic_tethered_facet_validation_v1/retrieval/`

**Interfaces:**
- Consumes: Task 1 sealed planning artifacts, complete original top-1,000 cache, exact facet cache, and the existing rate-limited transport.
- Produces: `run_retrieval(planning_dir, output_dir, transport)`, `verify_retrieval(output_dir)`, raw-first ledgers, candidate lists, `accepted_union.jsonl`, and a retrieval seal.

- [ ] **Step 1: Write failing cache, limiter, and union tests**

```python
def test_original_requests_must_be_exact_cache_hits() -> None:
    with pytest.raises(ValueError, match="original cache incomplete"):
        build_retrieval_plan(_planning(), original_cache={})


def test_facet_transport_enforces_interval_and_no_retry() -> None:
    transport = _fake_transport(fail_identity="facet-b")
    result = run_retrieval(_planning(), _output(), transport=transport, clock=_clock())
    assert result["request_start_deltas"] == [3.0, 3.0]
    assert transport.identities.count("facet-b") == 1


def test_union_deduplicates_by_topic_and_document_with_provenance() -> None:
    rows = build_union(_original_rows(), _facet_rows())
    assert len({(r["topic_id"], r["document_id"]) for r in rows}) == len(rows)
    assert rows[0]["stream_provenance"]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q code/tests/test_all_topic_facet_retrieve.py`

Expected: collection fails because the retrieval module does not exist.

- [ ] **Step 3: Implement retrieval by adapting the existing transport**

Reuse `RateLimitedFacetTransport` and the exact-identity cache behavior from
`deep_facet_candidate_run.py`. Reject manifest drift, topic drift, depth drift,
and any attempted original-query network request. Persist each raw response
before parsing candidates. Union rows must retain every stream ID, stream rank,
request identity, query hash, and response hash.

- [ ] **Step 4: Run synthetic and cache-only verification**

Run: `.venv/bin/python -m pytest -q code/tests/test_all_topic_facet_retrieve.py code/tests/test_deep_facet_candidate_run.py`

Expected: all tests pass, including crash-resume and failed-attempt behavior.

- [ ] **Step 5: Execute the sealed retrieval plan**

Run:

```bash
.venv/bin/python -m trec_rag.all_topic_facet_retrieve run \
  --planning outputs/all_topic_tethered_facet_validation_v1/planning \
  --output outputs/all_topic_tethered_facet_validation_v1/retrieval
.venv/bin/python -m trec_rag.all_topic_facet_retrieve verify \
  --retrieval outputs/all_topic_tethered_facet_validation_v1/retrieval
```

Expected: exactly the Task 1 missing request identities are issued, all starts
are at least three seconds apart, original requests remain zero, and every
topic has one authenticated union.

- [ ] **Step 6: Commit Task 2**

```bash
git add code/trec_rag/all_topic_facet_retrieve.py code/tests/test_all_topic_facet_retrieve.py
git commit -m "Add uniform all-topic facet retrieval"
```

---

### Task 3: Preflight and score narrative-tethered MiniLM features

**Files:**
- Create: `code/trec_rag/all_topic_tethered_score.py`
- Create: `code/tests/test_all_topic_tethered_score.py`
- Create at run time: `outputs/all_topic_tethered_facet_validation_v1/scoring/`

**Interfaces:**
- Consumes: Task 2 authenticated unions, Task 1 narratives/facets, pinned MiniLM model identity, and exact score cache.
- Produces: `build_score_plan(...)`, `run_scores(...)`, `verify_scores(...)`, tokenizer window plan, within-query percentile features, and a scoring seal.

- [ ] **Step 1: Write failing query, cache, and normalization tests**

```python
def test_tethered_query_contains_narrative_and_one_facet() -> None:
    query = render_tethered_query("full narrative", "subject relation facet")
    assert "full narrative" in query and "subject relation facet" in query


def test_raw_scores_never_normalize_across_queries() -> None:
    rows = percentile_features(_score_rows())
    assert {(r["topic_id"], r["query_id"]) for r in rows} == {("14", "n"), ("14", "f1")}
    assert _percentiles(rows, "n") == [1.0, 0.0]


def test_preflight_has_no_model_load() -> None:
    plan = build_score_plan(_union(), _manifest(), backend=_tokenizer_only())
    assert plan["external_calls"]["model_load"] == 0
    assert plan["window_count"] > 0
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q code/tests/test_all_topic_tethered_score.py`

Expected: collection fails because the scorer does not exist.

- [ ] **Step 3: Implement scoring as a strict adapter**

Reuse model revision, tokenizer-only backend, window construction, cache keys,
and local runner from `tethered_facet_minilm_score.py`. Generate narrative,
common/global, and narrative-plus-originating-facet score identities. Record exact
pair, window, cache-hit, cache-miss, estimated memory, and runtime counters.

- [ ] **Step 4: Freeze tokenizer-only preflight and verify**

Run:

```bash
.venv/bin/python -m trec_rag.all_topic_tethered_score preflight \
  --retrieval outputs/all_topic_tethered_facet_validation_v1/retrieval \
  --output outputs/all_topic_tethered_facet_validation_v1/scoring
.venv/bin/python -m trec_rag.all_topic_tethered_score verify-preflight \
  --scoring outputs/all_topic_tethered_facet_validation_v1/scoring
```

Expected: exact pair/window/cache-miss counts, zero model loads, and a sealed
score plan. Compare actual counts with the inventory estimate and record the
projected ROCm runtime before inference.

- [ ] **Step 5: Run local scoring and verify authenticated coverage**

Run:

```bash
.venv/bin/python-rocm -m trec_rag.all_topic_tethered_score run \
  --scoring outputs/all_topic_tethered_facet_validation_v1/scoring
.venv/bin/python -m trec_rag.all_topic_tethered_score verify \
  --scoring outputs/all_topic_tethered_facet_validation_v1/scoring
```

Expected: every planned pair has exactly one score, hosted/paid calls are zero,
and within-query percentile features are sealed.

- [ ] **Step 6: Run focused and compatibility tests**

Run: `.venv/bin/python -m pytest -q code/tests/test_all_topic_tethered_score.py code/tests/test_tethered_facet_minilm_score.py code/tests/test_facet_local_minilm_score.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add code/trec_rag/all_topic_tethered_score.py code/tests/test_all_topic_tethered_score.py
git commit -m "Add all-topic tethered MiniLM scoring"
```

---

### Task 4: Freeze static and reinitialized ranking arms

**Files:**
- Create: `code/trec_rag/all_topic_tethered_rank.py`
- Create: `code/tests/test_all_topic_tethered_rank.py`
- Create at run time: `outputs/all_topic_tethered_facet_validation_v1/rankings/`

**Interfaces:**
- Consumes: Tasks 1-3 seals, complete unions, existing RRF features, pinned DUAL coefficients, and tethered percentile features.
- Produces: `build_rankings(topic_input, controls)`, `reinitialized_dual(...)`, `verify_rankings(path)`, six complete per-topic permutations, audit traces, and a ranking seal.

- [ ] **Step 1: Write failing permutation and protected-prefix tests**

```python
def test_every_arm_is_the_same_complete_union() -> None:
    rankings, _ = build_rankings(_topic_input(), _controls())
    expected = set(_topic_input()["docids"])
    assert set(rankings) == set(ARMS)
    assert all(len(order) == len(expected) and set(order) == expected for order in rankings.values())


def test_protected_prefixes_are_exact() -> None:
    rankings, _ = build_rankings(_topic_input(), _controls())
    assert rankings["RRF100-STATIC-DUAL"][:100] == rankings["RRF"][:100]
    assert rankings["RRF500-REINIT-DUAL"][:500] == rankings["RRF"][:500]


def test_reinitialized_state_includes_prefix_coverage() -> None:
    _, audit = build_rankings(_topic_input(), _controls())
    assert audit["RRF100-REINIT-DUAL"]["seed_document_count"] == 100
    assert audit["RRF100-REINIT-DUAL"]["seed_coverage"] == _expected_seed_coverage()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q code/tests/test_all_topic_tethered_rank.py`

Expected: collection fails because the ranker does not exist.

- [ ] **Step 3: Implement all six preregistered arms**

Reuse the existing family-balanced RRF and DUAL objective without tuning. Static
arms call the current empty-state DUAL permutation and splice a prefix. Reinitialized
arms seed facet-coverage and lexical-redundancy state by replaying the protected
RRF prefix, then greedily select residual documents with deterministic document-ID
tie-breaking. Preserve objective components and marginal coverage attribution.

- [ ] **Step 4: Enforce the qrels firewall and create-only seal**

Reject qrels paths/fields recursively, verify every upstream seal, hash every
complete ranking, and write `parameters.json`, `input_bindings.json`,
`rankings.jsonl`, `audit.jsonl`, `summary.json`, and `SEALED.json` with exclusive
creation semantics.

- [ ] **Step 5: Run focused and compatibility tests**

Run: `.venv/bin/python -m pytest -q code/tests/test_all_topic_tethered_rank.py code/tests/test_tethered_facet_soft_coverage.py code/tests/test_deep_facet_candidate_rank.py`

Expected: all tests pass and input reordering cannot change any ranking.

- [ ] **Step 6: Freeze and verify real rankings**

Run:

```bash
.venv/bin/python -m trec_rag.all_topic_tethered_rank freeze \
  --planning outputs/all_topic_tethered_facet_validation_v1/planning \
  --retrieval outputs/all_topic_tethered_facet_validation_v1/retrieval \
  --scoring outputs/all_topic_tethered_facet_validation_v1/scoring \
  --output outputs/all_topic_tethered_facet_validation_v1/rankings
.venv/bin/python -m trec_rag.all_topic_tethered_rank verify \
  --rankings outputs/all_topic_tethered_facet_validation_v1/rankings
```

Expected: six complete permutations for each of 22 topics, exact protected
prefixes, and a verified seal created before qrels access.

- [ ] **Step 7: Commit Task 4**

```bash
git add code/trec_rag/all_topic_tethered_rank.py code/tests/test_all_topic_tethered_rank.py
git commit -m "Add all-topic protected DUAL rankings"
```

---

### Task 5: Evaluate all-topic regressions mechanically

**Files:**
- Create: `code/trec_rag/all_topic_tethered_evaluate.py`
- Create: `code/tests/test_all_topic_tethered_evaluate.py`
- Create at run time: `outputs/all_topic_tethered_facet_validation_v1/evaluation/`

**Interfaces:**
- Consumes: Task 4 verified seal, Task 2 provenance, pinned all-22 projected qrels, and the frozen promotion ladder.
- Produces: `evaluate_all_topics(...)`, `paired_bootstrap(...)`, `paired_sign_flip(...)`, `apply_promotion_rules(...)`, metrics/diagnostics tables, and an evaluation seal.

- [ ] **Step 1: Write failing metric, regression, and firewall tests**

```python
def test_any_topic_loss_blocks_promotion() -> None:
    decision = apply_promotion_rules(_metrics(one_loss=True), _statistics(significant=True))
    assert decision["promoted"] is False
    assert decision["failed_rules"] == ["zero_losses_at_1000"]


def test_aggregate_gain_cannot_hide_topic_loss() -> None:
    result = evaluate_all_topics(_rankings(big_gain_and_one_loss()), _qrels(), _provenance())
    assert result["arms"]["RRF100-STATIC-DUAL"]["pooled_delta_at_1000"] > 0
    assert result["arms"]["RRF100-STATIC-DUAL"]["loss_topic_ids"] == ["31"]


def test_qrels_reader_runs_only_after_ranking_verification() -> None:
    opened = False
    with pytest.raises(ValueError, match="ranking seal"):
        evaluate_frozen(_bad_rankings(), qrels_reader=_record_open(lambda: opened))
    assert opened is False
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q code/tests/test_all_topic_tethered_evaluate.py`

Expected: collection fails because the evaluator does not exist.

- [ ] **Step 3: Implement frozen metrics and paired inference**

Use relevance grade `>=2`, exact depths `(100, 250, 500, 1000, 1500)`, full
union, deterministic bootstrap seed `20260716`, 100,000 topic-bootstrap samples,
and exact sign-flip enumeration over 22 paired deltas when feasible. Apply Holm
correction across the five non-baseline arms in the frozen selection ladder.
Compute known-relevant count, binary/graded recall, nDCG, precision, judged rate,
normalized recall AUC, facet-only retention, full-union ceiling, per-topic deltas,
win/tie/loss, worst regression, and facet-rank-bucket yield.

- [ ] **Step 4: Encode the complete promotion contract**

Require zero losses at 250, 500, and 1,000; positive pooled and macro recall at
1,000; at least eight wins; corrected significance; protected-prefix identity;
and interpretable judged-rate behavior. Try the primary arm first, then the fixed
alternative ladder. If none passes, return `RRF` with every failed rule.

- [ ] **Step 5: Run focused and compatibility tests**

Run: `.venv/bin/python -m pytest -q code/tests/test_all_topic_tethered_evaluate.py code/tests/test_tethered_facet_soft_coverage_evaluate.py code/tests/test_evaluation.py`

Expected: all tests pass.

- [ ] **Step 6: Evaluate once and verify reproducibility**

Run:

```bash
.venv/bin/python -m trec_rag.all_topic_tethered_evaluate evaluate \
  --rankings outputs/all_topic_tethered_facet_validation_v1/rankings \
  --retrieval outputs/all_topic_tethered_facet_validation_v1/retrieval \
  --qrels trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels \
  --output outputs/all_topic_tethered_facet_validation_v1/evaluation
.venv/bin/python -m trec_rag.all_topic_tethered_evaluate verify \
  --evaluation outputs/all_topic_tethered_facet_validation_v1/evaluation
```

Expected: all 22 topics appear in every per-topic table and recomputation matches
saved metrics and the mechanical promotion decision exactly.

- [ ] **Step 7: Commit Task 5**

```bash
git add code/trec_rag/all_topic_tethered_evaluate.py code/tests/test_all_topic_tethered_evaluate.py
git commit -m "Evaluate all-topic facet ranking regressions"
```

---

### Task 6: Build, render, verify, and review the report

**Files:**
- Create: `code/trec_rag/build_all_topic_tethered_report.py`
- Create: `code/tests/test_build_all_topic_tethered_report.py`
- Create: `reports/experiments/all_topic_tethered_facet_validation_v1/README.md`
- Create: `reports/experiments/all_topic_tethered_facet_validation_v1/summary.json`
- Create: `reports/experiments/all_topic_tethered_facet_validation_v1/report_data.sqlite`
- Create: `reports/experiments/all_topic_tethered_facet_validation_v1/artifact.json`
- Create: `reports/experiments/all_topic_tethered_facet_validation_v1/report.html`
- Modify: `/home/npatta01/codex-rendered/index.html`
- Create rendered copy: `/home/npatta01/codex-rendered/plans/trec-2026-all-topic-tethered-facet-validation-v1.html`

**Interfaces:**
- Consumes: verified planning, retrieval, scoring, ranking, and evaluation seals.
- Produces: a self-contained accessible HTML report, machine-readable summary and SQLite data, artifact hashes, private rendered copy, and merge/promotion recommendation.

- [ ] **Step 1: Write failing report-contract tests**

```python
def test_report_names_every_topic_and_every_regression() -> None:
    built = build_report(_sources())
    assert set(built.summary["topic_ids"]) == set(ALL_TOPIC_IDS)
    assert built.summary["decision"]["loss_topic_ids"] == ["31"]
    assert "Topic 31" in built.html


def test_report_states_retrospective_and_known_relevant_limits() -> None:
    html = build_report(_sources()).html
    assert "retrospective full-development stress test" in html
    assert "known-relevant" in html
    assert "not evidence of generalization" in html


def test_html_has_mobile_table_fallback_and_no_external_dependencies() -> None:
    html = build_report(_sources()).html
    assert 'name="viewport"' in html
    assert "overflow-x:auto" in html.replace(" ", "")
    assert "https://cdn" not in html
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest -q code/tests/test_build_all_topic_tethered_report.py`

Expected: collection fails because the report builder does not exist.

- [ ] **Step 3: Implement an answer-first, source-backed report**

Lead with `promote` or `retain RRF`, followed by win/tie/loss counts and worst
topic regression. Include a per-topic responsive table, recall-depth curves,
facet-rank saturation chart, judged-rate caveat, provenance strata, exact costs,
method diagram, and expandable representative diagnostics. Bind every headline
number to the sealed evaluation artifacts and include zero secrets/raw datasets.

- [ ] **Step 4: Build canonical artifacts and reproduce the decision**

Run:

```bash
.venv/bin/python -m trec_rag.build_all_topic_tethered_report \
  --root outputs/all_topic_tethered_facet_validation_v1 \
  --output reports/experiments/all_topic_tethered_facet_validation_v1
.venv/bin/python -m trec_rag.build_all_topic_tethered_report verify \
  --report reports/experiments/all_topic_tethered_facet_validation_v1
```

Expected: summary, SQLite, artifact manifest, and HTML reproduce the sealed
promotion decision and all per-topic metrics.

- [ ] **Step 5: Run tests and browser accessibility checks**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_build_all_topic_tethered_report.py
.venv/bin/python code/tools/check_html_report.py \
  reports/experiments/all_topic_tethered_facet_validation_v1/report.html
```

Use headless Chrome at desktop `1440x1000` and mobile `390x844`. Expected: no
horizontal page overflow, keyboard-reachable disclosures, readable tables,
working anchors, no console errors, and no external runtime request.

- [ ] **Step 6: Publish only the sanitized private rendered copy**

Copy the verified HTML to the authorized rendered directory, add one current
index link, verify the live HTTPS response hash matches the local rendered copy,
and confirm the Serve mapping remains tailnet-only. Do not expose repository
files, raw logs, caches, qrels, or secrets.

- [ ] **Step 7: Run the complete targeted suite and independent review**

Run:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_all_topic_facet_contract.py \
  code/tests/test_all_topic_facet_retrieve.py \
  code/tests/test_all_topic_tethered_score.py \
  code/tests/test_all_topic_tethered_rank.py \
  code/tests/test_all_topic_tethered_evaluate.py \
  code/tests/test_build_all_topic_tethered_report.py
```

Expected: all tests pass. Request independent review of scope isolation, qrels
firewall, ranking correctness, per-topic regression logic, statistical tests,
report reproducibility, and merge versus promotion conclusions. Resolve every
Critical or Important finding and re-run covering tests.

- [ ] **Step 8: Commit Task 6**

```bash
git add code/trec_rag/build_all_topic_tethered_report.py \
  code/tests/test_build_all_topic_tethered_report.py \
  reports/experiments/all_topic_tethered_facet_validation_v1
git commit -m "Report all-topic facet validation"
```

