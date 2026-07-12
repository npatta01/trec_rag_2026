# Facet Retrieval-Control Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a frozen four-stream pilot that tests controlled anchor repetition and supported BM25 parameters, then determines whether those controls reduce facet noise without losing facet-specific evidence.

**Architecture:** Extend the raw-first request identity with backward-compatible optional `k1` and `b` fields. Add a focused manifest, runner, freezer, evaluator, and report builder; freeze all 25 topic-level R2 alternatives before qrels access, then select only among those frozen rankings.

**Tech Stack:** Python 3.12, pytest, requests-ratelimiter, pyrate-limiter, the existing retrieval ledger, Pyserini REST, family-balanced RRF, existing evaluation utilities, and the portable HTML report renderer.

## Global Constraints

- Work in `/home/npatta01/.codex/worktrees/41f9/trec_rag_2026` on `codex/structured-query-planner`.
- Preserve immutable v2.1 artifacts and every earlier retrieval ledger.
- Reject protected topics `144`, `213`, `224`, `407`, and `515` at every boundary.
- Make at most 12 new attempts on topics `200`, `225`, and `707`; failures count against the ceiling.
- Start at most one request every 10 seconds, burst one, redirects disabled, automatic retries disabled.
- Send only query text, `hits=100`, `k1`, and `b`; do not send Lucene/Solr syntax or unsupported controls.
- Reuse cached B0 results and do not run a cross-encoder in this phase.
- Freeze manifests, identities, responses, candidates, inspections, and all 25 R2 topic alternatives before qrels access.
- Keep outputs under `outputs/rag25_facet_retrieval_control_v1/` and the report under `reports/experiments/facet_retrieval_control_pilot_v1/`.
- Run Python with `.venv/bin/python`; stage only files named by each task.

## File map

- Modify `code/trec_rag/det_sparse_ledger.py`, `sparse_relevance_inspector.py`, and their tests.
- Create `code/trec_rag/facet_retrieval_control_manifest.py`, `facet_retrieval_control_run.py`, `facet_retrieval_control_experiment.py`, `facet_retrieval_control_freeze.py`, and `facet_retrieval_control_evaluate.py`.
- Create matching tests, `code/tools/build_facet_retrieval_control_manifest.py`, and `code/trec_rag/build_facet_retrieval_control_report.py`.
- Create durable experiment artifacts under `reports/experiments/facet_retrieval_control_pilot_v1/`.

---

### Task 1: Bind BM25 parameters to request identity

**Files:**
- Modify: `code/trec_rag/det_sparse_ledger.py:39-151`
- Modify: `code/tests/test_det_sparse_ledger.py:20-103`

**Interfaces:**
- Consumes: existing `RetrievalRequest.from_query` callers and v1 artifacts.
- Produces: optional `bm25_k1` and `bm25_b`; old identities retain their original request keys.

- [ ] **Step 1: Write failing identity tests**

```python
def test_explicit_bm25_settings_change_identity_without_changing_legacy_identity():
    legacy = _request(query="weighted query")
    explicit = RetrievalRequest.from_query(
        topic_id=legacy.identity.topic_id, variant_name=legacy.identity.variant_name,
        query_text=legacy.query_text, index_url=legacy.identity.index_url,
        index_id=legacy.identity.index_id, hits=100,
        analyzer_fingerprint_sha256=ANALYZER_SHA, bm25_k1=0.9, bm25_b=0.4,
    )
    assert "bm25_k1" not in legacy.identity.canonical_dict()
    assert explicit.identity.request_key != legacy.identity.request_key

@pytest.mark.parametrize(("k1", "b"), [(0.4, None), (None, 0.4), (-0.1, 0.4), (0.4, 1.1)])
def test_invalid_bm25_pairs_are_rejected(k1, b):
    with pytest.raises(ValueError):
        RetrievalRequest.from_query(
            topic_id="200", variant_name="W1", query_text="query",
            index_url="https://example.test/search", index_id="climbmix-400b",
            hits=100, analyzer_fingerprint_sha256=ANALYZER_SHA,
            bm25_k1=k1, bm25_b=b,
        )
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest code/tests/test_det_sparse_ledger.py -k bm25 -q`

Expected: failure because the BM25 keyword arguments do not exist.

- [ ] **Step 3: Implement backward-compatible fields**

Add `bm25_k1: float | None = None` and `bm25_b: float | None = None` at the end of `RetrievalRequestIdentity`. Require both or neither, finite `k1 >= 0`, and `0 <= b <= 1`; thread both through the factories. Preserve legacy keys by adding them to `canonical_dict()` only when present:

```python
if (self.bm25_k1 is None) != (self.bm25_b is None):
    raise ValueError("bm25_k1 and bm25_b must be provided together")
if self.bm25_k1 is not None:
    value["bm25_k1"] = float(self.bm25_k1)
    value["bm25_b"] = float(self.bm25_b)
```

- [ ] **Step 4: Verify identity and ledger compatibility**

Round-trip an explicit BM25 identity through ledger artifacts while retaining coverage for legacy identities and v1 artifacts.

Run: `.venv/bin/python -m pytest code/tests/test_det_sparse_ledger.py -q`

Expected: all tests pass, including legacy-ledger tests.

- [ ] **Step 5: Commit**

```bash
git add code/trec_rag/det_sparse_ledger.py code/tests/test_det_sparse_ledger.py
git commit -m "Add BM25 parameters to retrieval identity"
```

### Task 2: Freeze the exact control manifest

**Files:**
- Create: `code/trec_rag/facet_retrieval_control_manifest.py`
- Create: `code/tools/build_facet_retrieval_control_manifest.py`
- Create: `code/tests/test_facet_retrieval_control_manifest.py`
- Create: `reports/experiments/facet_retrieval_control_pilot_v1/manifest.json`
- Create: `reports/experiments/facet_retrieval_control_pilot_v1/README.md`

**Interfaces:**
- Consumes: `reports/experiments/sparse_relevance_pilot_v1/r1_manifest.json`.
- Produces: `ControlArm`, `ControlStream`, `ControlManifest`, `build_control_manifest()`, and `load_control_manifest(path)`.

- [ ] **Step 1: Write failing manifest tests**

```python
EXPECTED = {
    ("200", "f07a"): "Holocaust Holocaust enduring effects European European Jews Jews",
    ("225", "f02"): "violent video games violent video games exposure desensitization violence research",
    ("225", "f04"): "aggressive aggressive behavior children children risk factors psychology",
    ("707", "f02"): "sorbitol sorbitol human human health adverse effects safety",
}
def test_manifest_is_exact_and_bounded():
    manifest = build_control_manifest()
    assert {(s.topic_id, s.stream_id): s.reweighted_query for s in manifest.streams} == EXPECTED
    assert sum(a.external for s in manifest.streams for a in s.arms) == 12
    assert manifest.min_interval_seconds == 10.0
    assert all(set(analyze(s.reweighted_query)) <= set(analyze(s.baseline_query)) for s in manifest.streams)
```

Also test exact B0/W0/W1/W2 settings, round-trip JSON, duplicate rejection, and protected-topic rejection.

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest code/tests/test_facet_retrieval_control_manifest.py -q`

Expected: import failure.

- [ ] **Step 3: Implement and generate**

Define immutable arm/stream/manifest dataclasses. Require B0 `(0.9, 0.4, false)`, W0 `(0.9, 0.4, true)`, W1 `(0.4, 0.4, true)`, and W2 `(0.4, 0.0, true)` in that order. Validate exact R1 baseline namespace, no new analyzed vocabulary, exactly four streams, and 12 external arms.

```bash
.venv/bin/python code/tools/build_facet_retrieval_control_manifest.py --r1 reports/experiments/sparse_relevance_pilot_v1/r1_manifest.json --output reports/experiments/facet_retrieval_control_pilot_v1/manifest.json
.venv/bin/python -m pytest code/tests/test_facet_retrieval_control_manifest.py -q
```

Expected: create-only JSON, passing tests, and a README recording scope and gates.

- [ ] **Step 4: Commit**

```bash
git add code/trec_rag/facet_retrieval_control_manifest.py code/tools/build_facet_retrieval_control_manifest.py code/tests/test_facet_retrieval_control_manifest.py reports/experiments/facet_retrieval_control_pilot_v1/manifest.json reports/experiments/facet_retrieval_control_pilot_v1/README.md
git commit -m "Freeze facet retrieval control manifest"
```

### Task 3: Implement the 12-attempt runner

**Files:**
- Create: `code/trec_rag/facet_retrieval_control_run.py`
- Create: `code/tests/test_facet_retrieval_control_run.py`

**Interfaces:**
- Consumes: `ControlManifest`, `RetrievalLedger`, `RemotePyseriniConfig`, and `rate_limited_session`.
- Produces: `RateLimitedControlTransport`, `build_control_requests`, `preflight_control`, `execute_control`, and a CLI.

- [ ] **Step 1: Write failing tests**

```python
def test_requests_are_exactly_twelve_and_parameterized():
    requests = build_control_requests(build_control_manifest(), endpoint=ENDPOINT)
    assert len(requests) == 12
    assert sum(r.identity.topic_id == "225" for r in requests) == 6
    assert all(r.identity.bm25_k1 is not None and r.identity.bm25_b is not None for r in requests)

def test_interval_below_ten_seconds_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="ten seconds"):
        build_control_live_config(ENDPOINT, None, tmp_path / "limit.sqlite", 9.99)

def test_first_failure_stops_without_retry_or_later_calls(tmp_path):
    with pytest.raises(RetrievalLedgerError):
        execute_control(manifest, ledger, failing_transport, ENDPOINT)
    assert transport_calls == 1
    assert ledger.validate_run().failures == 1
```

Also test exact fake-session transmission of `{"query": query, "hits": "100", "k1": "0.4", "b": "0.0"}`, rejection of every protected topic before request construction, cache probing, or transport creation, and rejection of any thirteenth planned or external attempt.

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest code/tests/test_facet_retrieval_control_run.py -q`

Expected: import failure.

- [ ] **Step 3: Implement request construction and preflight**

Build only W0/W1/W2 requests using names such as `facet_control_v1:W0:f07a`, retriever version `pyserini_remote_bm25_controls_v1`, depth 100, and explicit parameters. Reject protected topics before request construction or cache access. Preflight uses `has_verified_cache()` without recording an invocation and rejects more than 12 planned or external attempts.

- [ ] **Step 4: Implement CLI execution**

Implement the control-specific transport in `facet_retrieval_control_run.py`; it sends only query text, `hits=100`, and the explicit identity `k1`/`b`, with redirects and automatic retries disabled. Use one ledger with `max_calls=12` and `max_calls_per_topic=6`. Enforce the 10-second minimum, stop on the first exception, and write `preflight.json` plus `retrieval_summary.json` with manifest/code hashes. A complete run requires zero failures/pending and exactly 12 planned invocations including cache hits.

- [ ] **Step 5: Test and commit**

```bash
.venv/bin/python -m pytest code/tests/test_facet_retrieval_control_run.py code/tests/test_det_sparse_ledger.py -q
git add code/trec_rag/facet_retrieval_control_run.py code/tests/test_facet_retrieval_control_run.py
git commit -m "Add rate-limited facet control runner"
```

Expected: all targeted tests pass.

### Task 4: Extend inspection and freeze 25 alternatives

**Files:**
- Modify: `code/trec_rag/sparse_relevance_inspector.py`
- Modify: `code/tests/test_sparse_relevance_inspector.py`
- Create: `code/trec_rag/facet_retrieval_control_experiment.py`
- Create: `code/trec_rag/facet_retrieval_control_freeze.py`
- Create: `code/tests/test_facet_retrieval_control_experiment.py`

**Interfaces:**
- Consumes: prior O/F0/R1 candidates, new W candidates, the manifest, and `family_balanced_rrf`.
- Produces: top-5/top-10 inspection, `build_topic_alternatives`, and `facet-control-ranking-freeze-v1`.

- [ ] **Step 1: Add failing top-10 inspection test**

```python
def test_inspector_reports_top_ten_without_changing_top_five_decision():
    result = inspect_stream(stream, candidates_with_drift_only_at_6_to_10())
    assert result.domain_drift_top5_count == 0
    assert result.domain_drift_top10_count == 5
    assert result.anchor_top10_count == 10
    assert result.decision == "keep"
```

Run the test, then add `anchor_top10_count`, `anchor_intent_cohit_top10_count`, `domain_drift_top10_count`, and `content_quality_top10_count` without changing top-five rejection behavior.

- [ ] **Step 2: Write failing alternative-matrix tests**

```python
def test_topic_alternative_matrix_is_complete():
    matrix = build_topic_alternatives(r1_arm, control_rows, manifest)
    assert len(matrix) == 25
    assert sum(k.startswith("R2:200:") for k in matrix) == 4
    assert sum(k.startswith("R2:225:") for k in matrix) == 16
    assert sum(k.startswith("R2:707:") for k in matrix) == 4
    assert [k for k in matrix if k.startswith("R2:897:")] == ["R2:897:B0"]
    assert all(len(rows) == 100 for rows in matrix.values())
```

Also assert that only the four registered streams change, shuffled inputs yield identical rows, and protected IDs fail before cache or fusion access.

- [ ] **Step 3: Implement replacement and fusion**

Implement `index_control_streams(baseline_r1, control_rows, manifest)` returning
`dict[tuple[str, str, str], tuple[RetrievedCandidate, ...]]`, and
`build_topic_alternatives(r1_arm, control_rows, manifest)` returning
`dict[str, list[RankedCandidate]]`.

Replace only the registered stream, preserve the facet count, run family-balanced RRF at `k=60` and depth 100, retain only the target topic, and sort every input/combination deterministically.

- [ ] **Step 4: Implement the create-only freezer**

The CLI verifies the prior freeze and ledgers, inspects B0/W0/W1/W2 without qrels, writes 25 ranking files, and binds manifest/prior-freeze/request/response/candidate/inspection/fusion/ranking hashes. Create `freeze.json` with `O_EXCL`, status `frozen_before_qrels`, and no qrels CLI argument.

- [ ] **Step 5: Test and commit**

```bash
.venv/bin/python -m pytest code/tests/test_sparse_relevance_inspector.py code/tests/test_facet_retrieval_control_experiment.py -q
git add code/trec_rag/sparse_relevance_inspector.py code/tests/test_sparse_relevance_inspector.py code/trec_rag/facet_retrieval_control_experiment.py code/trec_rag/facet_retrieval_control_freeze.py code/tests/test_facet_retrieval_control_experiment.py
git commit -m "Freeze facet control ranking alternatives"
```

Expected: passing tests and exactly 25 deterministic topic alternatives.

### Task 5: Evaluate marginal value and select frozen R2

**Files:**
- Modify: `code/trec_rag/facet_retrieval_control_experiment.py`
- Create: `code/trec_rag/facet_retrieval_control_evaluate.py`
- Modify: `code/tests/test_facet_retrieval_control_experiment.py`

**Interfaces:**
- Consumes: verified control freeze, frozen stream candidates, prior O/F0/R1 rankings, and projected qrels.
- Produces: `stream_evaluation.json`, `selection.json`, `evaluation.json`, and `decision.json`.

- [ ] **Step 1: Write failing selection/firewall tests**

```python
def test_selection_prioritizes_unique_graded_gain_and_tie_breaks():
    arms = {
        "B0": arm(gain=3, recall=.10, ndcg=.30, noise=2),
        "W0": arm(gain=5, recall=.09, ndcg=.28, noise=1),
        "W1": arm(gain=5, recall=.11, ndcg=.25, noise=1),
        "W2": arm(gain=5, recall=.11, ndcg=.25, noise=1),
    }
    assert select_stream_arm(arms).arm_id == "W1"

def test_corrupt_freeze_fails_before_qrels_open(monkeypatch):
    monkeypatch.setattr(Path, "open", fail_if_qrels)
    with pytest.raises(ValueError, match="freeze"):
        evaluate_control_freeze(corrupt_freeze, qrels_path)
    assert qrels_was_opened is False
```

Also test rejection when both noise families increase, exact-tie preference B0/W0/W1/W2, protected qrels skipping, and selected ranking references.

- [ ] **Step 2: Implement stream metrics and selection**

Unique relevant contribution is the count of grade-at-least-2 facet documents absent from original top 100. Unique graded gain is the sum of their qrel grades. Exclude coherence failures and arms that increase both drift and content noise versus B0. Maximize `(unique_graded_gain, graded_recall@100, ndcg@10, -top10_noise, -arm_preference)`.

- [ ] **Step 3: Resolve R2 using frozen references only**

```python
selected_rankings = {
    "200": f"R2:200:{selected['200/f07a']}",
    "225": f"R2:225:{selected['225/f02']}-{selected['225/f04']}",
    "707": f"R2:707:{selected['707/f02']}",
    "897": "R2:897:B0",
}
```

Average the four referenced topic metrics; do not create a post-qrels ranking.

- [ ] **Step 4: Implement decision rule**

```python
success = (
    r2_graded_recall > r1_graded_recall
    and r2_ndcg >= r1_ndcg - 0.02
    and min(per_topic_ndcg_deltas) >= -0.10
    and selected_noise <= b0_noise
)
```

Return `retrieval_repair_success` or `retrieval_repair_failed`; never open a reranker gate automatically.

- [ ] **Step 5: Test and commit**

```bash
.venv/bin/python -m pytest code/tests/test_facet_retrieval_control_experiment.py -q
git add code/trec_rag/facet_retrieval_control_experiment.py code/trec_rag/facet_retrieval_control_evaluate.py code/tests/test_facet_retrieval_control_experiment.py
git commit -m "Evaluate facet retrieval control pilot"
```

Expected: deterministic selection and freeze-before-qrels tests pass.

### Task 6: Build the rendered HTML report

**Files:**
- Create: `code/trec_rag/build_facet_retrieval_control_report.py`
- Create: `code/tests/test_build_facet_retrieval_control_report.py`
- Modify: `reports/experiments/facet_retrieval_control_pilot_v1/README.md`
- Generate in Task 7: `reports/experiments/facet_retrieval_control_pilot_v1/artifact.json`
- Generate in Task 7: `reports/experiments/facet_retrieval_control_pilot_v1/report.html`

**Interfaces:**
- Consumes: manifest, inspection, stream evaluation, selection, system evaluation, and decision JSON.
- Produces: canonical portable artifact JSON and self-contained HTML.

- [ ] **Step 1: Write failing artifact test**

```python
def test_report_exposes_queries_noise_marginal_gain_and_decision(fixtures):
    artifact = build_artifact(**fixtures)
    text = json.dumps(artifact)
    for required in (
        "Existing query", "Reweighted query", "k1", "b", "Wrong-domain",
        "Unique relevant beyond original", "Selected arm", "R2",
        "Cross-encoder not run",
    ):
        assert required in text
    assert artifact["snapshot"]["status"] == "ready"
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest code/tests/test_build_facet_retrieval_control_report.py -q`

Expected: import failure.

- [ ] **Step 3: Implement the answer-first artifact**

Include exact arm queries/settings, representative top results, top-5/top-10 drift/content counts, marginal relevant/graded contribution, selected arm rationale, O/F0/R1/R2 metrics, the mechanical decision rule, limitations, and a clear statement that no cross-encoder ran. Add grouped top-10-noise bars, marginal-gain bars, and an O/F0/R1/R2 table. Use only local data and no external runtime dependency.

- [ ] **Step 4: Test and commit**

```bash
.venv/bin/python -m pytest code/tests/test_build_facet_retrieval_control_report.py -q
git add code/trec_rag/build_facet_retrieval_control_report.py code/tests/test_build_facet_retrieval_control_report.py reports/experiments/facet_retrieval_control_pilot_v1/README.md
git commit -m "Add facet control report builder"
```

Expected: artifact tests pass.

### Task 7: Run retrieval, freeze, evaluate, and render

**Files:**
- Create outside git: `outputs/rag25_facet_retrieval_control_v1/run_v1/`
- Create outside git: `outputs/rag25_facet_retrieval_control_v1/freeze_v1/`
- Create outside git: `outputs/rag25_facet_retrieval_control_v1/evaluation_v1/`
- Generate: `reports/experiments/facet_retrieval_control_pilot_v1/artifact.json`
- Generate: `reports/experiments/facet_retrieval_control_pilot_v1/report.html`

**Interfaces:**
- Consumes: committed implementation, existing endpoint credential, prior ledgers/freeze, and qrels after the new freeze.
- Produces: raw-first run, freeze hash, evaluation, decision, and report.

- [ ] **Step 1: Run no-call preflight**

```bash
.venv/bin/python -m trec_rag.facet_retrieval_control_run \
  --manifest reports/experiments/facet_retrieval_control_pilot_v1/manifest.json \
  --endpoint http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search \
  --shared-cache /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/rag25_facet_retrieval_control_v1 \
  --output outputs/rag25_facet_retrieval_control_v1/run_v1 \
  --limiter-state /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote/rate-limit.sqlite \
  --min-interval-seconds 10 --preflight-only
```

Expected: 12 planned requests, zero to 12 verified cache hits, no protected topic, and no more than 12 external attempts. Stop if identities differ.

- [ ] **Step 2: Execute through the limiter**

Run the same command without `--preflight-only`. Expected: at least 10 seconds between external starts and zero failures/pending. If any request fails, preserve the ledger and stop without retry.

- [ ] **Step 3: Freeze all alternatives before qrels**

```bash
.venv/bin/python -m trec_rag.facet_retrieval_control_freeze \
  --manifest reports/experiments/facet_retrieval_control_pilot_v1/manifest.json \
  --base-run outputs/rag25_det_sparse_prompt_lab_v1/rate_limited_continuation_v1 \
  --base-cache /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/rag25_det_sparse_prompt_lab_base_http_restart_v1 \
  --r1-run outputs/rag25_sparse_relevance_paired_v1/run_v2/R1/ledger \
  --r1-cache /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/rag25_sparse_relevance_paired_v1 \
  --control-run outputs/rag25_facet_retrieval_control_v1/run_v1/ledger \
  --control-cache /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/rag25_facet_retrieval_control_v1 \
  --prior-freeze outputs/rag25_sparse_relevance_paired_v1/freeze_v1 \
  --output outputs/rag25_facet_retrieval_control_v1/freeze_v1
```

Expected: `inspection.json`, 25 ranking files, and create-only `freeze.json`; no qrels path is supplied.

- [ ] **Step 4: Evaluate after freeze**

```bash
.venv/bin/python -m trec_rag.facet_retrieval_control_evaluate \
  --freeze-dir outputs/rag25_facet_retrieval_control_v1/freeze_v1 \
  --prior-freeze outputs/rag25_sparse_relevance_paired_v1/freeze_v1 \
  --qrels /home/npatta01/data/competitions/trec_rag_2026/trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels \
  --output outputs/rag25_facet_retrieval_control_v1/evaluation_v1
```

Expected: stream metrics, selection, aggregate evaluation, mechanical decision, and protected qrels rows skipped after topic-ID inspection.

- [ ] **Step 5: Build and package HTML**

```bash
.venv/bin/python -m trec_rag.build_facet_retrieval_control_report \
  --manifest reports/experiments/facet_retrieval_control_pilot_v1/manifest.json \
  --freeze-dir outputs/rag25_facet_retrieval_control_v1/freeze_v1 \
  --evaluation-dir outputs/rag25_facet_retrieval_control_v1/evaluation_v1 \
  --output reports/experiments/facet_retrieval_control_pilot_v1/artifact.json
node /home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs \
  --input reports/experiments/facet_retrieval_control_pilot_v1/artifact.json \
  --output reports/experiments/facet_retrieval_control_pilot_v1/report.html
```

Expected: self-contained HTML with snapshot status `ready`.

- [ ] **Step 6: Render-check and commit durable artifacts**

Serve locally and inspect desktop/mobile with Playwright. Verify no console errors and exact metric reproduction.

```bash
git add reports/experiments/facet_retrieval_control_pilot_v1/artifact.json reports/experiments/facet_retrieval_control_pilot_v1/report.html reports/experiments/facet_retrieval_control_pilot_v1/README.md
git commit -m "Report facet retrieval control findings"
```

### Task 8: Final verification and handoff

**Files:**
- Verify all Task 1-7 files and generated artifacts.

**Interfaces:**
- Consumes: completed implementation and evidence.
- Produces: evidence-backed outcome without cross-encoder claims.

- [ ] **Step 1: Run targeted tests**

```bash
.venv/bin/python -m pytest code/tests/test_det_sparse_ledger.py code/tests/test_sparse_relevance_inspector.py code/tests/test_facet_retrieval_control_manifest.py code/tests/test_facet_retrieval_control_run.py code/tests/test_facet_retrieval_control_experiment.py code/tests/test_build_facet_retrieval_control_report.py -q
```

Expected: all targeted tests pass.

- [ ] **Step 2: Run full suite**

Run: `.venv/bin/python -m pytest code/tests -q`

Expected: all tests pass with only documented pre-existing skips.

- [ ] **Step 3: Verify integrity and scope**

```bash
git diff --check
git status --short
git log --oneline --max-count=12
```

Reproduce freeze/report hashes, verify protected IDs are absent from requests/rankings/report datasets, confirm outputs are unstaged, and preserve unrelated prompt-lab files.

- [ ] **Step 4: Present the result**

Lead with success/failure, selected arm per stream, marginal evidence gained/lost, exact attempt/cache/failure counts, and the rendered HTML link. State that no cross-encoder ran. If successful, propose a separately approved original-only versus facet-augmented cross-encoder comparison; if failed, identify the remaining REST limitation.
