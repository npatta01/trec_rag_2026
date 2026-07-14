# Facet-Aware Fusion Held-Out Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a qrels-blind four-topic pilot that tests whether facet-local MiniLM plus constrained xQuAD preserves novel relevant documents without materially harming nDCG@10.

**Architecture:** A frozen manifest owns the four topics, 24 source-tethered facets, lexical gate terms, and original-cache identities. A create-only retrieval runner reuses the repository's persistent rate limiter and raw-first ledger, a local scorer ranks each facet's candidates and applies the frozen quality gate, and an offline ranker creates six deterministic arms before evaluation can open the four-topic qrels projection once. A standalone HTML builder renders the authenticated artifacts; a separate advisor reviews the frozen results rather than influencing them.

**Tech Stack:** Python 3.11, pytest, requests + requests-ratelimiter, existing retrieval ledger, PyTorch/Transformers on ROCm, JSON/JSONL artifacts, standalone HTML/CSS/JavaScript.

## Global Constraints

- Reject protected topics `144, 213, 224, 407, 515` at every boundary.
- Exclude prior evaluated topics `200, 225, 707, 897` from confirmatory evidence.
- Use exactly held-out topics `233, 273, 161, 14` in frozen hash order.
- Reuse cached original top 100 and issue at most 24 new facet top-100 requests.
- Start no more than one external request per three seconds through the persistent limiter; no automatic retry of a failed immutable attempt.
- Use `cross-encoder/ms-marco-MiniLM-L6-v2` revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`; no model download is permitted.
- Raw BM25 and MiniLM scores never cross query boundaries; fusion consumes ranks only.
- Freeze all queries, candidates, gates, parameters, and rankings before opening qrels.
- Do not add dense primary retrieval, recursive repair, paid calls, Stage A, or post-qrels tuning.
- Preserve all existing `sparse_relevance_*` untracked files untouched.

---

### Task 1: Frozen manifest and topic firewall

**Files:**
- Create: `code/trec_rag/facet_aware_fusion_manifest.py`
- Create: `code/tests/test_facet_aware_fusion_manifest.py`
- Create: `reports/experiments/facet_aware_fusion_pilot_v1/manifest.json`
- Create: `reports/experiments/facet_aware_fusion_pilot_v1/README.md`

**Interfaces:**
- Produces: `build_manifest(cache_root: Path) -> dict[str, object]`, `load_manifest(path: Path) -> dict[str, object]`, `validate_manifest(payload: Mapping[str, object]) -> None`, and canonical `manifest.json` with ordered topics and facets.
- Manifest facet records contain `topic_id`, `facet_id`, `query`, `obligation`, `anchor_terms`, `relation_terms`, `wrong_domain_patterns`, `bridge_terms`, and `manifest_order`.

- [ ] **Step 1: Write failing firewall, selection, and facet-contract tests**

```python
def test_manifest_has_exact_frozen_boundary_and_24_facets(tmp_path):
    payload = build_manifest(tmp_path)
    assert payload["topic_ids"] == ["233", "273", "161", "14"]
    assert len(payload["facets"]) == 24
    assert {row["topic_id"] for row in payload["facets"]}.isdisjoint(PROTECTED_TOPIC_IDS)

def test_every_facet_is_tethered_and_bridge_terms_are_audited(tmp_path):
    payload = build_manifest(tmp_path)
    for facet in payload["facets"]:
        assert facet["anchor_terms"] and facet["relation_terms"]
        for bridge in facet["bridge_terms"]:
            assert bridge["purpose"] in ALLOWED_BRIDGE_PURPOSES
            assert bridge["rationale"]
```

- [ ] **Step 2: Run tests and confirm import/contract failures**

Run: `.venv/bin/python -m pytest code/tests/test_facet_aware_fusion_manifest.py -q`

Expected: FAIL because `trec_rag.facet_aware_fusion_manifest` does not exist.

- [ ] **Step 3: Implement the exact manifest, canonical JSON validation, hashes, and protected/prior topic rejection**

The 24 facets are the explicit narrative obligations frozen in the design: 3 for topic 233 and 7 each for 273, 161, and 14. The validator must recompute the SHA-256 topic-selection order from `rag25_facet_aware_fusion_v1`, require exact source query text, require existing original cache files with matching query/topic, and refuse unsupported manifest keys or reordered records.

- [ ] **Step 4: Run the targeted tests and create the source-backed manifest**

Run: `.venv/bin/python -m pytest code/tests/test_facet_aware_fusion_manifest.py -q`

Expected: PASS.

Run: `.venv/bin/python -m trec_rag.facet_aware_fusion_manifest create --cache-root /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote --output reports/experiments/facet_aware_fusion_pilot_v1/manifest.json`

Expected: one create-only manifest with `topic_count=4`, `facet_count=24`, and `qrels_opened=false`.

- [ ] **Step 5: Commit manifest code, tests, and report inputs**

```bash
git add code/trec_rag/facet_aware_fusion_manifest.py code/tests/test_facet_aware_fusion_manifest.py reports/experiments/facet_aware_fusion_pilot_v1
git commit -m "Freeze held-out facet-aware fusion manifest"
```

### Task 2: Rate-limited raw-first facet retrieval

**Files:**
- Create: `code/trec_rag/facet_aware_fusion_run.py`
- Create: `code/tests/test_facet_aware_fusion_run.py`
- Generate: `outputs/rag25_facet_aware_fusion_v1/retrieval_v1/`

**Interfaces:**
- Consumes: validated manifest from Task 1, `det_sparse_ledger.RetrievalLedger`, and `remote_client.rate_limited_session`.
- Produces: `build_requests(manifest, endpoint) -> tuple[RetrievalRequest, ...]`, create-only `preflight.json`, ledger/raw response tree, normalized `candidates.jsonl`, and `retrieval_summary.json`.

- [ ] **Step 1: Write failing tests for the exact 24-request allowlist and limiter**

```python
def test_build_requests_is_exact_and_protected_topics_fail(manifest):
    requests = build_requests(manifest, ENDPOINT)
    assert len(requests) == 24
    assert len({row.identity.request_key for row in requests}) == 24
    assert all(row.identity.hits == 100 for row in requests)

def test_transport_uses_persistent_limiter(monkeypatch, manifest):
    monkeypatch.setattr(module, "rate_limited_session", fake_limited_session)
    transport = RateLimitedFacetTransport(config, build_requests(manifest, ENDPOINT))
    assert transport.session is sentinel_session
```

- [ ] **Step 2: Run tests and confirm missing-module failures**

Run: `.venv/bin/python -m pytest code/tests/test_facet_aware_fusion_run.py -q`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement qrels-blind preflight and create-only execution**

Require the endpoint `http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search`, index `climbmix-400b`, `hits=100`, the manifest's analyzer fingerprint, a 24-call global budget, and a seven-call per-topic budget. Bind the session to `/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote/rate-limit.sqlite` with one start per three seconds. Record HTTP failure bytes and never retry the same failed request automatically.

- [ ] **Step 4: Verify tests, run preflight, then execute the one authorized retrieval batch**

Run: `.venv/bin/python -m pytest code/tests/test_facet_aware_fusion_run.py -q`

Expected: PASS.

Run: `.venv/bin/python -m trec_rag.facet_aware_fusion_run preflight --manifest reports/experiments/facet_aware_fusion_pilot_v1/manifest.json --output outputs/rag25_facet_aware_fusion_v1/retrieval_v1`

Expected: `planned_requests=24`, `minimum_start_span_seconds>=69`, `qrels_opened=false`.

Run: `.venv/bin/python -m trec_rag.facet_aware_fusion_run run --manifest reports/experiments/facet_aware_fusion_pilot_v1/manifest.json --output outputs/rag25_facet_aware_fusion_v1/retrieval_v1`

Expected: 24 terminal ledger entries and no protected topic.

- [ ] **Step 5: Commit runner and tests; keep generated retrieval evidence untracked**

```bash
git add code/trec_rag/facet_aware_fusion_run.py code/tests/test_facet_aware_fusion_run.py
git commit -m "Add rate-limited facet-aware retrieval runner"
```

### Task 3: Local MiniLM scoring, quality gate, and rank-only fusion

**Files:**
- Create: `code/trec_rag/facet_aware_fusion_rank.py`
- Create: `code/tests/test_facet_aware_fusion_rank.py`
- Generate: `outputs/rag25_facet_aware_fusion_v1/freeze_v1/`

**Interfaces:**
- Consumes: Task 1 manifest, Task 2 candidates, four exact cached original result files, and the already materialized pinned model snapshot.
- Produces: `rank_normalized(rank: int | None, depth: int) -> float`, `quality_gate(facet, ranked_docs) -> GateDecision`, `build_rankings(...) -> dict[str, dict[str, list[str]]]`, and a create-only freeze containing scores, gates, candidate provenance, O/RRF/BI/XQ/CXQ/TUS-C rankings, parameters, and hashes.

- [ ] **Step 1: Write failing unit tests for rank transforms, gating, interleaving, xQuAD, and CXQ deadlines**

```python
def test_rank_normalization_and_no_raw_score_use():
    assert rank_normalized(1, 50) == 1.0
    assert rank_normalized(None, 50) == 0.0

def test_rejected_facet_is_never_forced():
    rankings, provenance = constrained_xquad(original, [accepted, rejected])
    assert rejected.facet_id not in {row.get("forced_facet") for row in provenance}

def test_input_reordering_does_not_change_rankings(frozen_rows):
    assert build_rankings(frozen_rows) == build_rankings(list(reversed(frozen_rows)))
```

- [ ] **Step 2: Run tests and confirm missing behavior**

Run: `.venv/bin/python -m pytest code/tests/test_facet_aware_fusion_rank.py -q`

Expected: FAIL because the ranking module does not exist.

- [ ] **Step 3: Implement tokenizer preflight, cached-score accounting, ROCm scoring, quality gate, and six ranking arms**

Use the pinned model and top-four span-distinct window aggregation already defined by the prior MiniLM pilot. Deduplicate facet query/document pairs in the global score cache. Gate MiniLM top five using the frozen anchor/relation/wrong-domain patterns. Implement advisor-reviewed `Rel(d)=max(g(d), max_f p_f(d))`, xQuAD lambda `0.35`, deterministic tie breaks, and CXQ deadlines `10 + ceil(40*i/m)`. Record every forced insertion.

- [ ] **Step 4: Run unit tests, preflight exact local work, execute scoring once, and freeze rankings**

Run: `.venv/bin/python -m pytest code/tests/test_facet_aware_fusion_rank.py -q`

Expected: PASS.

Run: `.venv/bin/python-rocm -m trec_rag.facet_aware_fusion_rank preflight --manifest reports/experiments/facet_aware_fusion_pilot_v1/manifest.json --retrieval outputs/rag25_facet_aware_fusion_v1/retrieval_v1 --output outputs/rag25_facet_aware_fusion_v1/freeze_v1`

Expected: exact document/window/cache-hit/cache-miss counts, model revision match, no inference, retrieval, or qrels access.

Run: `.venv/bin/python-rocm -m trec_rag.facet_aware_fusion_rank freeze --manifest reports/experiments/facet_aware_fusion_pilot_v1/manifest.json --retrieval outputs/rag25_facet_aware_fusion_v1/retrieval_v1 --output outputs/rag25_facet_aware_fusion_v1/freeze_v1`

Expected: all six arms contain exactly 100 unique documents per topic and `qrels_opened=false`.

- [ ] **Step 5: Commit scorer/ranker and tests; keep generated model scores and rankings untracked**

```bash
git add code/trec_rag/facet_aware_fusion_rank.py code/tests/test_facet_aware_fusion_rank.py
git commit -m "Add facet-aware ranking and constrained xQuAD"
```

### Task 4: One-time qrels projection and mechanical evaluation

**Files:**
- Create: `code/trec_rag/facet_aware_fusion_evaluate.py`
- Create: `code/tests/test_facet_aware_fusion_evaluate.py`
- Generate: `outputs/rag25_facet_aware_fusion_v1/evaluation_v1/`

**Interfaces:**
- Consumes: authenticated Task 3 freeze and a create-only qrels projection containing exactly `233,273,161,14`.
- Produces: per-topic and aggregate metrics, novel relevant retention, gains/losses, promotion decision, qrels access receipt, and trusted create-only consumption record.

- [ ] **Step 1: Write failing firewall and metric/decision tests**

```python
def test_qrels_cannot_open_before_complete_freeze(tmp_path):
    with pytest.raises(ValueError, match="freeze"):
        evaluate(tmp_path / "partial-freeze", qrels_path, tmp_path / "out")

def test_promotion_rule_is_mechanical():
    decision = decide(metrics_fixture)
    assert decision["promoted_arm"] == "CXQ"
    assert decision["checks"]["novel_retention_fraction"] is True
```

- [ ] **Step 2: Run tests and confirm missing evaluation module**

Run: `.venv/bin/python -m pytest code/tests/test_facet_aware_fusion_evaluate.py -q`

Expected: FAIL because the evaluator does not exist.

- [ ] **Step 3: Implement freeze verification, exact projection, metrics, and promotion rules**

Verify every frozen hash before qrels access. Project only the four exact topics, create the qrels consumption identity before reading the projection, and reject reordered, missing, extra, protected, or prior-pilot topics. Compute graded Recall@100, nDCG@10, relevant@10, judged rates, RRF-relative novel retained documents, and all per-topic deltas. Apply the design decision mechanically without parameter changes.

- [ ] **Step 4: Run tests and perform the one authorized evaluation**

Run: `.venv/bin/python -m pytest code/tests/test_facet_aware_fusion_evaluate.py -q`

Expected: PASS.

Run: `.venv/bin/python -m trec_rag.facet_aware_fusion_evaluate evaluate --freeze outputs/rag25_facet_aware_fusion_v1/freeze_v1 --output outputs/rag25_facet_aware_fusion_v1/evaluation_v1`

Expected: one qrels receipt, metrics for six arms and four exact topics, and a deterministic promoted-or-retained decision.

- [ ] **Step 5: Commit evaluator and tests; keep projected qrels and evaluation evidence untracked**

```bash
git add code/trec_rag/facet_aware_fusion_evaluate.py code/tests/test_facet_aware_fusion_evaluate.py
git commit -m "Evaluate frozen facet-aware fusion pilot"
```

### Task 5: Rendered report, full verification, and independent post-run review

**Files:**
- Create: `code/trec_rag/build_facet_aware_fusion_report.py`
- Create: `code/tests/test_build_facet_aware_fusion_report.py`
- Create: `reports/experiments/facet_aware_fusion_pilot_v1/report.html`
- Create: `reports/experiments/facet_aware_fusion_pilot_v1/summary.json`
- Create: `reports/experiments/facet_aware_fusion_pilot_v1/advisor_review.md`

**Interfaces:**
- Consumes: Tasks 1–4 artifacts and the independent advisor's post-results memo.
- Produces: accessible, responsive standalone report with an answer-first verdict, per-topic metrics, facet acceptance, novel-document flow, representative evidence, limitations, and exact artifact provenance.

- [ ] **Step 1: Write failing report reproduction and accessibility tests**

```python
def test_report_reproduces_decision_and_metrics(tmp_path):
    artifact = build_report(FREEZE, EVALUATION, ADVISOR_REVIEW, tmp_path)
    html = artifact.read_text()
    assert evaluation["decision"]["promoted_arm"] in html
    assert "Novel relevant documents" in html
    assert 'name="viewport"' in html
```

- [ ] **Step 2: Run report tests and confirm missing builder**

Run: `.venv/bin/python -m pytest code/tests/test_build_facet_aware_fusion_report.py -q`

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Ask a fresh advisor to inspect frozen findings before writing conclusions**

Send only the design, manifest, gate diagnostics, rankings, evaluation metrics, and representative judged gains/losses. Ask whether the evidence supports the mechanical decision, whether xQuAD improved recall for the intended reason, whether nDCG damage or qrels sparsity invalidates the conclusion, and what single next experiment is justified. Save the returned review verbatim in `advisor_review.md`; do not alter rankings or metrics.

- [ ] **Step 4: Implement and render the standalone report**

The first screen must state: what ran, external/local cost, RRF vs XQ vs CXQ outcome, count/fraction of novel relevant documents preserved, nDCG guardrail result, advisor verdict, and promote/stop decision. Below it, include mobile-readable metric cards/tables, a document-flow visual, accepted/rejected facet diagnostics, example passages, methodology, and hashes.

- [ ] **Step 5: Run targeted and full verification, then inspect desktop/mobile rendering**

Run: `.venv/bin/python -m pytest code/tests/test_build_facet_aware_fusion_report.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest code/tests/test_facet_aware_fusion_*.py code/tests/test_build_facet_aware_fusion_report.py -q`

Expected: PASS.

Run: `.venv/bin/python -m trec_rag.build_facet_aware_fusion_report --manifest reports/experiments/facet_aware_fusion_pilot_v1/manifest.json --freeze outputs/rag25_facet_aware_fusion_v1/freeze_v1 --evaluation outputs/rag25_facet_aware_fusion_v1/evaluation_v1 --advisor reports/experiments/facet_aware_fusion_pilot_v1/advisor_review.md --output reports/experiments/facet_aware_fusion_pilot_v1/report.html`

Expected: standalone HTML and `summary.json` reproduce the frozen decision.

Serve locally and inspect with Playwright at `1440x1000` and `390x844`. Expected: no horizontal overflow, all controls keyboard reachable, visible focus, readable tables, and identical numeric findings.

- [ ] **Step 6: Commit the report builder, tests, rendered report, summary, and advisor review**

```bash
git add code/trec_rag/build_facet_aware_fusion_report.py code/tests/test_build_facet_aware_fusion_report.py reports/experiments/facet_aware_fusion_pilot_v1
git commit -m "Report facet-aware fusion pilot findings"
```

## Final verification

- [ ] Run `git diff --check HEAD~5..HEAD` and confirm no whitespace errors.
- [ ] Run all new tests and the relevant existing limiter, ledger, MiniLM, and evaluation tests.
- [ ] Verify every report number against the saved JSON artifacts.
- [ ] Verify git status contains no staged/untracked files from the user's existing `sparse_relevance_*` work.
- [ ] Keep the local report server private and return a clickable local report link.
