# Tethered Soft-Coverage Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the repository's existing DUAL objective with narrative-tethered facet scores, evaluate complete candidate-ordering proxies, and publish a private rendered v3 report.

**Architecture:** A new ranking adapter authenticates sealed inputs, swaps only the facet-local score maps, and produces complete create-only permutations. A separate evaluator opens the already pinned qrels only after ranking hashes exist. A standalone report builder consumes saved summaries and emits a sanitized, self-contained HTML report plus reproducible data artifacts.

**Tech Stack:** Python 3.12, NumPy, pytest, standard-library JSON/SQLite/HTML, existing TREC RAG ranking and evaluation modules, headless Chrome, and the private Tailscale Serve portal.

## Global Constraints

- Work only in `/home/npatta01/.codex/worktrees/41f9/trec_rag_2026` on `codex/structured-query-planner`.
- Preserve immutable v2.1 and every existing sealed output and report.
- Hard-reject topic IDs `144`, `213`, `224`, `407`, and `515` before source access.
- Pilot topics are exactly `219`, `72`, `300`, and `84`; results are post-qrels diagnostics and cannot promote a production method.
- Perform zero retrieval, network, model load, model inference, hosted inference, paid call, or download.
- Consume exactly the existing 8,114 accepted topic-document rows.
- Raw BM25 and cross-encoder scores never cross query boundaries; reuse within-query rank percentiles.
- Reuse the frozen DUAL coefficients exactly; do not tune weights on these topics.
- Every ranking arm is a complete permutation; rank 100 is a protected-head boundary, not an output cutoff.
- Load qrels only after all ranking artifacts and hashes are frozen.
- Keep unrelated untracked sparse-relevance files untouched and out of every commit.
- Keep canonical source reports in the repository and copy only sanitized derived HTML under `/home/npatta01/codex-rendered/plans/`.
- Serve only through the existing tailnet-only mapping; never use Funnel or a public listener.

---

### Task 1: Freeze tethered soft-coverage permutations

**Files:**
- Create: `code/trec_rag/tethered_facet_soft_coverage.py`
- Create: `code/tests/test_tethered_facet_soft_coverage.py`
- Create at run time: `outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/freeze/`

**Interfaces:**
- Consumes: authenticated accepted-union, phase-2 common/narrative scores, RRF features, tethered facet document scores, and existing NARRATIVE/FIXED-O0 permutations.
- Produces: `build_soft_permutations(topic_input, controls) -> tuple[dict[str, list[str]], dict[str, dict[str, dict[str, object]]]]`, `freeze_soft_rankings(...) -> dict[str, object]`, `verify_soft_freeze(path) -> dict[str, object]`, and create-only ranking artifacts.

- [ ] **Step 1: Write failing unit tests for soft ranking and protected head**

```python
def test_soft_rankings_are_complete_and_protected_head_is_not_a_cutoff() -> None:
    rankings, _audit = build_soft_permutations(_topic_input(), _controls())
    expected = set(_topic_input()["docids"])
    for arm in ("TETHERED-DUAL", "TETHERED-DUAL-NR", "RRF100-TETHERED-DUAL"):
        assert set(rankings[arm]) == expected
        assert len(rankings[arm]) == len(expected)
    assert rankings["RRF100-TETHERED-DUAL"][:100] == _controls()["RRF"][:100]


def test_tethered_scores_replace_only_facet_local_features() -> None:
    rankings, audit = build_soft_permutations(_topic_input(), _controls())
    assert audit["parameters"]["dual"] == {
        "G": 0.35, "N": 0.15, "R": 0.15,
        "L": 0.25, "B": 0.10, "D": -0.15,
    }
    assert audit["parameters"]["facet_score_source"] == "narrative_tethered"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/python -m pytest -q code/tests/test_tethered_facet_soft_coverage.py`

Expected: collection fails because `trec_rag.tethered_facet_soft_coverage` does not exist.

- [ ] **Step 3: Implement the minimal ranking adapter**

Implement these boundaries:

```python
ARMS = (
    "RRF", "NARRATIVE", "FIXED-O0",
    "TETHERED-DUAL", "TETHERED-DUAL-NR",
    "RRF100-TETHERED-DUAL",
)
HEAD_SIZE = 100


def protect_head(head: Sequence[str], tail: Sequence[str], size: int = HEAD_SIZE) -> list[str]:
    protected = list(head[:size])
    seen = set(protected)
    return protected + [document_id for document_id in tail if document_id not in seen]
```

Authenticate the historical sources with their repository verifiers, load the
existing DUAL features, replace only `facets[*].scores` with the tethered score
maps, call the established greedy implementation unchanged, and save:

- `parameters.json` with exact coefficients and zero external-call counters;
- `input_bindings.json` with path-relative hashes and row counts;
- `rankings.jsonl` with arm, topic, rank, document ID, objective components, and coverage attribution;
- `summary.json` with per-topic counts and complete-permutation checks; and
- `SEALED.json` binding all four files before evaluation.

- [ ] **Step 4: Add failing boundary tests**

```python
def test_protected_topic_rejects_before_reader_runs() -> None:
    opened = False
    def reader(_path):
        nonlocal opened
        opened = True
        return []
    with pytest.raises(ValueError, match="protected topic 144"):
        load_topic_rows(["144"], reader=reader)
    assert opened is False


def test_soft_output_is_deterministic_under_input_reordering() -> None:
    forward, _ = build_soft_permutations(_topic_input(), _controls())
    reverse, _ = build_soft_permutations(_topic_input(reversed_rows=True), _controls())
    assert reverse == forward
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q code/tests/test_tethered_facet_soft_coverage.py code/tests/test_deep_facet_candidate_rank.py code/tests/test_tethered_facet_two_basket.py`

Expected: all tests pass.

- [ ] **Step 6: Freeze the real offline rankings**

Run:

```bash
.venv/bin/python -m trec_rag.tethered_facet_soft_coverage freeze \
  --output outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/freeze
.venv/bin/python -m trec_rag.tethered_facet_soft_coverage verify \
  --freeze outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/freeze
```

Expected: six complete 8,114-row arms across the same four topic pools, protected-topic count zero, and every external/model counter zero.

- [ ] **Step 7: Commit Task 1**

```bash
git add code/trec_rag/tethered_facet_soft_coverage.py \
  code/tests/test_tethered_facet_soft_coverage.py
git commit -m "Add tethered soft coverage rankings"
```

---

### Task 2: Evaluate recall-depth and coverage proxies

**Files:**
- Create: `code/trec_rag/tethered_facet_soft_coverage_evaluate.py`
- Create: `code/tests/test_tethered_facet_soft_coverage_evaluate.py`
- Create at run time: `outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/evaluation/`

**Interfaces:**
- Consumes: Task 1 sealed rankings, accepted-union provenance, and pinned qrels projection.
- Produces: `evaluate_proxy(rankings, qrels, provenance, depths) -> dict[str, object]`, `normalized_recall_auc(ranking, relevant) -> float`, `evaluate_frozen_proxy(...) -> dict[str, object]`, and create-only metrics/diagnostics/summary artifacts.

- [ ] **Step 1: Write failing calculation tests**

```python
def test_normalized_recall_auc_uses_the_complete_topic_depth() -> None:
    assert normalized_recall_auc(["r", "x", "r2"], {"r", "r2"}) == pytest.approx(
        (0.5 + 0.5 + 1.0) / 3.0
    )


def test_full_depth_recall_is_identical_for_complete_permutations() -> None:
    result = evaluate_proxy(_rankings(), _qrels(), _provenance(), depths=(1, 2, 4))
    full = {arm: values["aggregate"]["binary_recall_full"] for arm, values in result["arms"].items()}
    assert len(set(full.values())) == 1


def test_proxy_does_not_claim_true_nugget_coverage() -> None:
    result = evaluate_proxy(_rankings(), _qrels(), _provenance(), depths=(1, 4))
    assert result["coverage_proxy"]["is_true_nugget_coverage"] is False
```

- [ ] **Step 2: Run evaluator tests and verify RED**

Run: `.venv/bin/python -m pytest -q code/tests/test_tethered_facet_soft_coverage_evaluate.py`

Expected: collection fails because the evaluator does not exist.

- [ ] **Step 3: Implement metrics and the qrels firewall**

Use relevance grade `>=2`. Compute binary and graded recall, nDCG, relevant
counts, per-topic deltas, facet-only relevant retention, judged rate,
qrels-positive facet attribution, and the discrete normalized recall AUC:

```python
def normalized_recall_auc(ranking: Sequence[str], relevant: set[str]) -> float:
    if not ranking or not relevant:
        return 0.0
    found = 0
    area = 0.0
    for document_id in ranking:
        found += document_id in relevant
        area += found / len(relevant)
    return area / len(ranking)
```

The CLI must call `verify_soft_freeze()` and confirm every ranking hash before it
opens the qrels path. Save `metrics.json`, `diagnostics.json`, `summary.json`, and
`input_bindings.json` with the exact depths `100,250,500,1000,1500,full`.

- [ ] **Step 4: Run focused and compatibility tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_tethered_facet_soft_coverage_evaluate.py \
  code/tests/test_tethered_facet_evaluate.py \
  code/tests/test_deep_facet_candidate_evaluate.py
```

Expected: all tests pass.

- [ ] **Step 5: Evaluate the sealed ranking once**

Run:

```bash
.venv/bin/python -m trec_rag.tethered_facet_soft_coverage_evaluate evaluate \
  --freeze outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/freeze \
  --qrels outputs/rag25_deep_facet_candidates_v1/evaluation_v1/qrels_projection.jsonl \
  --union outputs/rag25_deep_facet_candidates_v1/gate_v1/u_accepted.jsonl \
  --output outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/evaluation
```

Expected: evaluation status complete, all six arms measured, and full-depth recall equal across arms.

- [ ] **Step 6: Independently recompute headline values**

Run a separate read-only Python check from `rankings.jsonl` and
`qrels_projection.jsonl`; compare Recall@500, Recall@1,000, full recall, and AUC
against `metrics.json`. Expected differences are exactly zero within `1e-12`.

- [ ] **Step 7: Commit Task 2**

```bash
git add code/trec_rag/tethered_facet_soft_coverage_evaluate.py \
  code/tests/test_tethered_facet_soft_coverage_evaluate.py
git commit -m "Evaluate tethered soft coverage proxy"
```

---

### Task 3: Build and privately render the v3 analytical report

**Files:**
- Create: `code/trec_rag/build_tethered_soft_coverage_report.py`
- Create: `code/tests/test_build_tethered_soft_coverage_report.py`
- Create: `reports/experiments/tethered_facet_minilm_diagnostic_v3/README.md`
- Create at run time: `reports/experiments/tethered_facet_minilm_diagnostic_v3/report.html`
- Create at run time: `reports/experiments/tethered_facet_minilm_diagnostic_v3/artifact.json`
- Create at run time: `reports/experiments/tethered_facet_minilm_diagnostic_v3/summary.json`
- Create at run time: `reports/experiments/tethered_facet_minilm_diagnostic_v3/report_data.sqlite`

**Interfaces:**
- Consumes: Task 2 verified evaluation, Task 1 freeze, and the v2 report summary.
- Produces: `build_report_payload(...) -> dict[str, object]`, `render_report(payload) -> str`, `write_report(...) -> dict[str, object]`, and a self-contained accessible HTML report.

- [ ] **Step 1: Write failing report tests**

```python
def test_report_separates_proxy_from_final_rag_quality() -> None:
    html = render_report(_payload())
    assert "This is not answer-generation evaluation" in html
    assert "qrels-positive facet exposure is only a proxy" in html


def test_report_has_no_sensitive_paths_or_raw_documents() -> None:
    html = render_report(_payload())
    assert "/home/" not in html
    assert "window_text" not in html
    assert "API_KEY" not in html


def test_report_contains_required_comparisons_and_sources() -> None:
    html = render_report(_payload())
    for value in ("RRF", "NARRATIVE", "FIXED-O0", "TETHERED-DUAL", "RRF100-TETHERED-DUAL"):
        assert value in html
    assert "Sources and reproducibility" in html
```

- [ ] **Step 2: Run report tests and verify RED**

Run: `.venv/bin/python -m pytest -q code/tests/test_build_tethered_soft_coverage_report.py`

Expected: collection fails because the builder does not exist.

- [ ] **Step 3: Implement the answer-first report**

The report must include:

1. the outcome and recommended next action;
2. why 31.1% exhaustive document recall is low but not equivalent to 31.1% answer coverage;
3. original, facet-only, overlap, incremental, and missed-document counts;
4. recall-depth and nDCG comparisons for all arms;
5. per-topic gains/regressions;
6. what the facet proxy measures and cannot measure;
7. strong versus weak facet behavior and remaining candidate-generation gap;
8. the boundary with the separate answer-generation worktree; and
9. source hashes, metric definitions, and zero-call receipts.

Use a semantic HTML table for the nonadditive four-topic overlap decomposition
instead of a chart. Record that chart omission in `artifact.json`: exact overlap
counts are more audit-friendly and a stacked chart would imply false additivity.

- [ ] **Step 4: Run report tests and generate canonical artifacts**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_build_tethered_soft_coverage_report.py
.venv/bin/python -m trec_rag.build_tethered_soft_coverage_report \
  --freeze outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/freeze \
  --evaluation outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_soft_coverage_v1/evaluation \
  --prior-summary reports/experiments/tethered_facet_minilm_diagnostic_v2/summary.json \
  --output reports/experiments/tethered_facet_minilm_diagnostic_v3
```

Expected: four canonical report artifacts, no raw document text, and exact source hashes.

- [ ] **Step 5: Verify desktop/mobile rendering and accessibility**

Open the canonical HTML with headless Chrome at `1440x1000` and `390x844`.
Verify no horizontal overflow, clipped tables, missing headings, broken anchors,
console errors, or inaccessible contrast/focus states.

- [ ] **Step 6: Publish the sanitized derived copy to the private portal**

Copy `report.html` to:

`/home/npatta01/codex-rendered/plans/trec-2026-tethered-soft-coverage-proxy-v3.html`

Update `/home/npatta01/codex-rendered/index.html`, remove the stale v2 portal
entry after the v3 link is verified, and preserve the canonical v2 repository
report. Confirm the live HTTPS bytes match the canonical SHA256 and both
`tailscale serve status` and `tailscale funnel status` show tailnet-only access.

- [ ] **Step 7: Run final verification**

Run all new tests plus the existing tethered, deep-facet ranking, and evaluation
compatibility suites. Run `git diff --check`, scan rendered files for secrets and
remote absolute paths, and verify the output receipts report zero external calls.

- [ ] **Step 8: Commit Task 3**

```bash
git add code/trec_rag/build_tethered_soft_coverage_report.py \
  code/tests/test_build_tethered_soft_coverage_report.py \
  reports/experiments/tethered_facet_minilm_diagnostic_v3
git commit -m "Report tethered soft coverage findings"
```

