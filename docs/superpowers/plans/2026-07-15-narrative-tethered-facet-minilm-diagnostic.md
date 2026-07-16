# Narrative-Tethered Facet MiniLM Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether scoring each cached facet candidate with the full narrative plus its generating facet improves deep candidate recall under a protected, facet-balanced two-basket ranking.

**Architecture:** Add three isolated pipeline modules: an authenticated local-only tethered MiniLM scorer, a deterministic two-basket ranker, and an evaluator/report builder that consumes only the existing projected judgments. Reuse the sealed deep-facet candidate population, windowing policy, global score cache, top-four aggregation, and RRF/DUAL permutations without modifying prior artifacts.

**Tech Stack:** Python 3.11, pytest, Hugging Face Transformers, PyTorch ROCm via `.venv/bin/python-rocm`, existing `GlobalScoreCache`, standard-library JSON/HTML/SQLite helpers, Playwright for rendered verification.

## Global Constraints

- Work only in `/home/npatta01/.codex/worktrees/41f9/trec_rag_2026` on `codex/structured-query-planner`.
- Preserve every existing artifact under `outputs/rag25_deep_facet_candidates_v1`; write only to a new `post_qrels_tethered_facet_minilm_v1` leaf.
- Hard-reject topics `144`, `213`, `224`, `407`, and `515` before source joins, cache access, tokenization, inference, ranking, evaluation, and reporting.
- Accept only frozen topics `219`, `72`, `300`, and `84`, in that order.
- Use exactly 4,800 accepted facet/document pairs from the 24 accepted facet streams and the existing 8,114-row accepted union.
- Perform no retrieval, network request, paid call, model download, query repair, adaptive search, Mixedbread run, dense retrieval, training, original-qrels access, or qrels re-projection.
- Use local `cross-encoder/ms-marco-MiniLM-L6-v2` revision `c5ee24cb16019beea0893ab7796b1df96625c6b8` with the existing tokenizer/window/float32/top-four aggregation contract.
- Serialize every new query exactly as `<full narrative>\n\nFocus: <generating facet query>`.
- Stop before inference unless preflight finds exactly 4,800 pairs, at most 25,000 windows, at most ten projected minutes, the pinned local model, and a working ROCm device.
- Protect exact RRF ranks 1--100; ranks 101--500 must contain 200 RRF-basket and 200 disjoint facet-basket documents per topic unless a recorded exhaustion makes the arm invalid.
- Every new arm must be a complete, duplicate-free permutation of that topic's accepted union.
- Use only the sealed existing four-topic qrels projection after rankings are sealed; never open the original qrels source.
- Treat the result as a post-qrels diagnostic, not a production promotion.
- Do not stage or modify the unrelated untracked `sparse_relevance*` and `prompt_lab*` files already present in the worktree.

## File Structure

- Create `code/trec_rag/tethered_facet_minilm_score.py`: source authentication, query rendering, inference-free preflight, local scoring, score aggregation, and create-only receipts.
- Create `code/tests/test_tethered_facet_minilm_score.py`: query, source, protected-topic, resource-ceiling, cache, and local-run contract tests.
- Create `code/trec_rag/tethered_facet_two_basket.py`: percentile normalization, per-topic quotas, disjoint basket construction, full permutations, seals, and CLI.
- Create `code/tests/test_tethered_facet_two_basket.py`: deterministic ranking, ties, quotas, duplicates, protected head, and permutation tests.
- Create `code/trec_rag/tethered_facet_evaluate.py`: sealed-projection evaluation, novel-retention accounting, diagnostics, and mechanical decision.
- Create `code/tests/test_tethered_facet_evaluate.py`: firewall, metric, decision, and per-topic guard tests.
- Create `code/trec_rag/build_tethered_facet_report.py`: source-bound artifact, SQLite companion, and standalone accessible HTML renderer.
- Create `code/tests/test_build_tethered_facet_report.py`: evidence bindings, narrative/facet examples, caveats, deterministic rendering, and CLI tests.
- Create `reports/experiments/tethered_facet_minilm_diagnostic_v1/README.md`: reproduction commands and interpretation boundary.
- Generate `reports/experiments/tethered_facet_minilm_diagnostic_v1/{artifact.json,summary.json,report_data.sqlite,report.html}` only after the real run verifies.

---

### Task 1: Authenticated tethered-query preflight

**Files:**
- Create: `code/trec_rag/tethered_facet_minilm_score.py`
- Create: `code/tests/test_tethered_facet_minilm_score.py`

**Interfaces:**
- Consumes: sealed `phase1_v1/candidates.jsonl`, `phase1_v1/scoring_receipt.json`, `gate_v1/gates.json`, `gate_v1/summary.json`, the deep-facet manifest, the pinned model materialization receipt, and `GlobalScoreCache`.
- Produces: `render_tethered_query(narrative: str, facet_query: str) -> str`, `build_tethered_candidates(...) -> list[dict[str, object]]`, `create_preflight(...) -> dict[str, object]`, and a CLI `preflight` command.

- [ ] **Step 1: Write failing unit tests for exact query and source coverage**

```python
def test_render_tethered_query_is_exact_and_rejects_blank_parts():
    assert render_tethered_query("Full narrative", "one facet") == (
        "Full narrative\n\nFocus: one facet"
    )
    with pytest.raises(ValueError, match="narrative"):
        render_tethered_query(" ", "one facet")


def test_candidates_cover_only_accepted_facets_and_preserve_identity():
    rows = build_tethered_candidates(manifest(), phase1_rows(), accepted_gates())
    assert len(rows) == 4
    assert {(row["topic_id"], row["facet_id"], row["document_id"]) for row in rows} == {
        ("219", "219-positive", "d1"),
        ("219", "219-positive", "d2"),
        ("84", "84-safety", "d3"),
        ("84", "84-safety", "d4"),
    }
    assert all(row["query"].endswith(f"Focus: {row['facet_query']}") for row in rows)
```

- [ ] **Step 2: Run the focused tests and confirm the missing module failure**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_tethered_facet_minilm_score.py -q
```

Expected: collection fails with `ModuleNotFoundError: trec_rag.tethered_facet_minilm_score`.

- [ ] **Step 3: Implement exact rendering, candidate joins, and protected-topic rejection**

```python
TOPIC_IDS = ("219", "72", "300", "84")
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
PAIR_COUNT = 4_800
WINDOW_CEILING = 25_000
RUNTIME_CEILING_SECONDS = 600.0


def render_tethered_query(narrative: str, facet_query: str) -> str:
    narrative = narrative.strip()
    facet_query = facet_query.strip()
    if not narrative:
        raise ValueError("narrative is required")
    if not facet_query:
        raise ValueError("facet query is required")
    return f"{narrative}\n\nFocus: {facet_query}"


def reject_topic(value: object) -> str:
    topic_id = str(value)
    if topic_id in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic_id} is forbidden")
    if topic_id not in TOPIC_IDS:
        raise ValueError(f"unexpected topic {topic_id}")
    return topic_id
```

`build_tethered_candidates` must validate the manifest and prior phase-1 receipt hashes before reading candidate rows, validate all topic IDs before indexing them, retain only gate rows with `status == "accepted"`, require the exact per-facet 200-row population, bind text/query/model identities, and sort by `(TOPIC_IDS index, manifest_order, prior BM25 rank, document_id)`.

- [ ] **Step 4: Add failing preflight tests for cache order and ceilings**

```python
def test_preflight_is_tokenizer_only_and_freezes_exact_cache_state(tmp_path):
    result = create_preflight(fixture_sources(tmp_path), tmp_path / "out", tokenizer=WordTokenizer(), cache=FakeCache())
    assert result["status"] == "tokenizer_only_preflight_complete"
    assert result["model_constructed"] is False
    assert result["network_access_supported"] is False
    assert result["summary"]["query_document_pair_count"] == 4
    assert (tmp_path / "out" / "candidates.jsonl").exists()
    assert (tmp_path / "out" / "windows.jsonl").exists()


@pytest.mark.parametrize("pairs,windows,seconds", [(4799, 20, 1), (4800, 25001, 1), (4800, 20, 601)])
def test_preflight_stops_on_exact_population_or_resource_drift(pairs, windows, seconds):
    with pytest.raises(ValueError, match="preflight ceiling"):
        enforce_preflight_ceiling(pairs, windows, seconds)
```

- [ ] **Step 5: Implement tokenizer-only preflight and immutable receipt**

Reuse `build_window_plan`, `load_verified_tokenizer`, `load_verified_materialization`, `score_cache_context`, and `GlobalScoreCache`. Write create-only `candidates.jsonl`, `windows.jsonl`, and `preflight.json`; include source SHA-256 bindings, model/tokenizer receipt hashes, exact cache hits/misses, per-topic/per-facet counts, window coverage summaries, `model_constructed: false`, and all ceilings. The CLI surface is:

```python
preflight = subparsers.add_parser("preflight")
preflight.add_argument("--manifest", required=True, type=Path)
preflight.add_argument("--phase1", required=True, type=Path)
preflight.add_argument("--gate", required=True, type=Path)
preflight.add_argument("--output", required=True, type=Path)
preflight.add_argument("--cache-root", type=Path, default=SCORE_CACHE_ROOT)
```

The parser must expose no qrels, retrieval, network, model-download, or hosted-inference argument.

- [ ] **Step 6: Verify Task 1 and commit**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_tethered_facet_minilm_score.py -q
git diff --check
git add code/trec_rag/tethered_facet_minilm_score.py code/tests/test_tethered_facet_minilm_score.py
git commit -m "Add tethered facet MiniLM preflight"
```

Expected: all focused tests pass; only the two Task 1 files are committed.

### Task 2: Bounded local MiniLM scoring

**Files:**
- Modify: `code/trec_rag/tethered_facet_minilm_score.py`
- Modify: `code/tests/test_tethered_facet_minilm_score.py`

**Interfaces:**
- Consumes: Task 1 `preflight.json`, exact window/candidate bytes, the pinned local model receipt, and global score cache.
- Produces: `run_local_scoring(preflight_path: Path, cache_root: Path = SCORE_CACHE_ROOT) -> dict[str, object]`, `aggregate_document_scores(...) -> list[dict[str, object]]`, create-only `scores.jsonl`, `document_scores.jsonl`, and `scoring_receipt.json`, plus CLI `score` and `verify` commands.

- [ ] **Step 1: Write failing cache-resume and aggregation tests**

```python
def test_scoring_only_forwards_exact_cache_misses(tmp_path, monkeypatch):
    runner = FakeRunner(scores={"k2": 2.0})
    receipt = run_local_scoring(preflight_fixture(tmp_path, cached={"k1": 1.0}), runner=runner)
    assert runner.forwarded_keys == ["k2"]
    assert receipt["unique_forward_pair_count"] == 1
    assert receipt["cache_hit_count"] == 1


def test_document_aggregation_matches_existing_top4_contract():
    rows = aggregate_document_scores(window_rows_with_span_overlap())
    assert rows == [{
        "topic_id": "219",
        "facet_id": "219-positive",
        "document_id": "d1",
        "score": pytest.approx(7.35),
        "selected_window_count": 2,
    }]
```

- [ ] **Step 2: Run the two new tests and confirm they fail**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_tethered_facet_minilm_score.py -q -k 'scoring or aggregation'
```

Expected: failure because the scoring and aggregation functions do not exist.

- [ ] **Step 3: Implement local-only scoring and fail-closed verification**

Reuse the batching and float32 serialization from `deep_facet_candidate_score.run_local_scoring`, but require the Task 1 schema and 600-second hard ceiling. Load model/tokenizer only with `local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True`; require `torch.cuda.is_available()` and `torch.version.hip`; use batch size 32 and `torch.inference_mode()`.

Before model construction, re-read cache state and require the exact planned miss keys. Atomically reserve the run. After each successful batch, write through `GlobalScoreCache`; never retry a failed batch. Aggregate with `facet_local_minilm_rank.aggregate_top4`, require exactly 4,800 `(topic, facet, document)` document-score rows, and bind every row to query/text/model/window hashes.

- [ ] **Step 4: Add and pass CLI/firewall tests**

```python
def test_score_cli_has_no_download_network_retrieval_or_qrels_argument():
    parser = build_parser()
    help_text = parser.format_help()
    for forbidden in ("qrels", "endpoint", "download", "retrieval"):
        assert forbidden not in help_text.lower()


def test_protected_topic_fails_before_cache_or_model(monkeypatch):
    monkeypatch.setattr(module, "GlobalScoreCache", ExplodingDependency)
    with pytest.raises(ValueError, match="protected topic 144"):
        verify_preflight(protected_preflight())
```

- [ ] **Step 5: Verify Task 2 and commit**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_tethered_facet_minilm_score.py code/tests/test_deep_facet_candidate_score.py code/tests/test_rerank_score_cache.py -q
git diff --check
git add code/trec_rag/tethered_facet_minilm_score.py code/tests/test_tethered_facet_minilm_score.py
git commit -m "Add bounded tethered facet MiniLM scoring"
```

Expected: all selected scorer/cache tests pass.

### Task 3: Deterministic protected two-basket ranking

**Files:**
- Create: `code/trec_rag/tethered_facet_two_basket.py`
- Create: `code/tests/test_tethered_facet_two_basket.py`

**Interfaces:**
- Consumes: prior sealed RRF and DUAL rankings, accepted-union rows, gate facet order, prior facet-only scores, and Task 2 tethered document scores.
- Produces: `average_rank_percentiles`, `topic_quotas`, `build_facet_basket`, `build_two_basket_permutation`, `freeze_rankings`, and `verify_freeze`; create-only `parameters.json`, `input_bindings.json`, `rankings.jsonl`, `prefixes.json`, `summary.json`, and `SEALED.json`.

- [ ] **Step 1: Write failing percentile and quota tests**

```python
def test_average_rank_percentiles_are_query_local_and_tie_aware():
    assert average_rank_percentiles({"a": 3.0, "b": 3.0, "c": 1.0}) == {
        "a": pytest.approx(5 / 6), "b": pytest.approx(5 / 6), "c": pytest.approx(1 / 3)
    }


@pytest.mark.parametrize("count,expected", [
    (7, [29, 29, 29, 29, 28, 28, 28]),
    (4, [50, 50, 50, 50]),
    (6, [34, 34, 33, 33, 33, 33]),
])
def test_topic_quotas_total_200(count, expected):
    assert topic_quotas(count, 200) == expected
    assert sum(expected) == 200
```

- [ ] **Step 2: Run and confirm the new module is missing**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_tethered_facet_two_basket.py -q
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement percentiles, global edge traversal, and shortage redistribution**

```python
def topic_quotas(facet_count: int, capacity: int = 200) -> list[int]:
    if facet_count <= 0 or capacity <= 0:
        raise ValueError("positive facet count and capacity are required")
    base, remainder = divmod(capacity, facet_count)
    return [base + (index < remainder) for index in range(facet_count)]
```

`build_facet_basket` must exclude the RRF head and RRF basket, sort all `(percentile, facet_order, bm25_rank, document_id)` edges globally, attribute duplicates to the highest-scoring quota-eligible facet, then redistribute shortages one slot per facet in manifest order. Return document IDs plus selected facet, percentile, prior rank, nominal quota, shortage, and duplicate-skip provenance.

- [ ] **Step 4: Write failing protected-head, alternation, and permutation tests**

```python
def test_two_basket_ranking_protects_head_and_is_complete():
    result = build_two_basket_permutation(fixture_topic())
    assert result.document_ids[:100] == fixture_topic().rrf[:100]
    assert result.sources[100:500:2] == ["rrf_basket"] * 200
    assert result.sources[101:500:2] == ["facet_basket"] * 200
    assert len(result.document_ids) == len(set(result.document_ids))
    assert set(result.document_ids) == set(fixture_topic().accepted_union)


def test_ranking_is_invariant_to_input_row_order():
    expected = build_two_basket_permutation(fixture_topic())
    shuffled = build_two_basket_permutation(shuffled_fixture_topic())
    assert shuffled == expected
```

- [ ] **Step 5: Implement both arms, full provenance, create-only seal, and verifier**

Build `FACET-2B` from existing facet-only document scores and `TETHERED-2B` from Task 2 scores. For each topic require exact arm populations, exact first-100 RRF identity, 200 disjoint RRF and facet basket rows from ranks 101--500, then append remaining candidates in existing DUAL order. Store one row per arm/topic/rank with source basket, generating facet, percentile, facet rank, RRF/DUAL ranks, query hash, text hash, and score provenance. `verify_freeze` must recompute every SHA-256 and semantic invariant.

- [ ] **Step 6: Verify Task 3 and commit**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_tethered_facet_two_basket.py code/tests/test_deep_facet_candidate_rank.py -q
git diff --check
git add code/trec_rag/tethered_facet_two_basket.py code/tests/test_tethered_facet_two_basket.py
git commit -m "Add protected two-basket facet ranking"
```

Expected: all selected ranking tests pass.

### Task 4: Projection-only evaluation and mechanical decision

**Files:**
- Create: `code/trec_rag/tethered_facet_evaluate.py`
- Create: `code/tests/test_tethered_facet_evaluate.py`

**Interfaces:**
- Consumes: verified Task 3 freeze, sealed prior `evaluation_v1/qrels_projection.jsonl`, prior evaluation metrics, and frozen novel set derived from relevant accepted-facet candidates absent from original@1000.
- Produces: `evaluate_arm`, `derive_novel_set`, `decide`, and `evaluate`; create-only `metrics.json`, `diagnostics.json`, `decision.json`, `summary.json`, and `input_bindings.json`.

- [ ] **Step 1: Write failing projection firewall and metric tests**

```python
def test_evaluation_accepts_projection_only_after_verified_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_freeze", lambda _: (_ for _ in ()).throw(ValueError("bad seal")))
    monkeypatch.setattr(Path, "read_bytes", ExplodingRead())
    with pytest.raises(ValueError, match="bad seal"):
        evaluate(tmp_path / "freeze", tmp_path / "projection.jsonl", tmp_path / "out")


def test_top100_identity_is_a_hard_invariant():
    with pytest.raises(ValueError, match="top 100"):
        validate_protected_head(rrf=list(range(100)), arm=list(range(99)) + [999])
```

- [ ] **Step 2: Run and confirm the evaluator module is missing**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_tethered_facet_evaluate.py -q
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement metrics, novel-set accounting, and basket diagnostics**

Reuse `deep_facet_candidate_evaluate.evaluate_ranking` and `evaluate_set`. Load only projection rows for exact `TOPIC_IDS`; reject extra, missing, protected, or reordered topics before indexing judgments. Derive the 177-document novel set using the frozen definition and require its exact count. Report binary/graded recall at 500 and 1,000, judged rate, novel retention, per-topic deltas, basket contributions, and relevant yield by facet.

- [ ] **Step 4: Write and pass the complete decision-rule tests**

```python
def test_decision_pass_requires_every_frozen_guard():
    result = decide(passing_evidence())
    assert result["label"] == "mechanical_pass"
    assert all(result["guards"].values())


@pytest.mark.parametrize("guard", [
    "top100_identity", "recall500", "novel500", "recall1000",
    "novel1000", "per_topic_loss", "judged_coverage",
])
def test_each_failed_guard_prevents_pass(guard):
    evidence = passing_evidence()
    evidence = break_guard(evidence, guard)
    assert decide(evidence)["label"] != "mechanical_pass"
```

The implementation must enforce: exact top-100 identity; binary and graded Recall@500 not below RRF or FACET-2B with at least one strict improvement over FACET-2B; at least 89/177 novel documents at 500; binary and graded Recall@1000 not below both controls; at least 142/177 novel documents at 1,000; no per-topic binary or graded Recall@500 loss worse than 0.02; and judged-rate loss no worse than 0.05. Only judged coverage or a documented basket shortage may produce `inconclusive`; all other misses produce `mechanical_fail`.

- [ ] **Step 5: Verify Task 4 and commit**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_tethered_facet_evaluate.py code/tests/test_deep_facet_candidate_evaluate.py -q
git diff --check
git add code/trec_rag/tethered_facet_evaluate.py code/tests/test_tethered_facet_evaluate.py
git commit -m "Add tethered facet diagnostic evaluation"
```

Expected: all selected evaluator tests pass.

### Task 5: Accessible source-backed HTML report

**Files:**
- Create: `code/trec_rag/build_tethered_facet_report.py`
- Create: `code/tests/test_build_tethered_facet_report.py`
- Create: `reports/experiments/tethered_facet_minilm_diagnostic_v1/README.md`

**Interfaces:**
- Consumes: verified Task 1/2 receipts, Task 3 freeze, Task 4 evaluation, and bounded representative rows already stored in diagnostics.
- Produces: deterministic `artifact.json`, `summary.json`, `report_data.sqlite`, and standalone `report.html`; `main(argv)` CLI.

- [ ] **Step 1: Write failing report-content and provenance tests**

```python
def test_report_answers_the_three_user_questions(built):
    html = built.html.lower()
    assert "did narrative tethering reduce facet noise?" in html
    assert "did two-basket fusion recover novel relevant documents?" in html
    assert "what should happen next?" in html


def test_report_exposes_narrative_facet_and_document_evidence(built):
    row = built.artifact["representatives"][0]
    assert row["narrative"]
    assert row["facet_query"]
    assert row["facet_only_percentile"] is not None
    assert row["tethered_percentile"] is not None
    assert row["selected_passage"]
    assert row["qrels_grade"] >= 0
```

- [ ] **Step 2: Run and confirm the report builder is missing**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_build_tethered_facet_report.py -q
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement deterministic artifact, SQLite companion, and HTML**

The report must lead with the mechanical label and metric deltas, show the fixed pipeline visually, compare RRF/FACET-2B/TETHERED-2B at 500 and 1,000, show per-topic/facet contribution tables, and include representative promoted/demoted passages with full narrative and facet visible. It must explicitly say “post-qrels diagnostic,” “no new retrieval,” and “not production validation.” Use semantic headings, table captions, keyboard-focus styles, sufficient contrast, responsive layouts, and no external runtime dependency.

- [ ] **Step 4: Add deterministic/create-only/CLI tests and README commands**

```python
def test_report_is_deterministic_and_create_only(tmp_path, sources):
    first = build_report(sources, tmp_path / "one")
    second = build_report(sources, tmp_path / "two")
    assert first.artifact_bytes == second.artifact_bytes
    assert first.html_bytes == second.html_bytes
    with pytest.raises(FileExistsError):
        build_report(sources, tmp_path / "one")


def test_report_rejects_unbound_or_protected_sources(sources):
    sources.topic_ids.append("144")
    with pytest.raises(ValueError, match="protected topic 144"):
        build_artifact(sources)
```

README reproduction commands must use `.venv/bin/python` for offline stages and `.venv/bin/python-rocm` only for the local scoring stage. Do not include a server or publishing command.

- [ ] **Step 5: Verify Task 5 and commit**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_build_tethered_facet_report.py -q
git diff --check
git add code/trec_rag/build_tethered_facet_report.py code/tests/test_build_tethered_facet_report.py reports/experiments/tethered_facet_minilm_diagnostic_v1/README.md
git commit -m "Add tethered facet diagnostic report"
```

Expected: all selected report tests pass.

### Task 6: Real preflight, bounded scoring, evaluation, and rendered verification

**Files:**
- Generate: `outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1/scoring/*` (preflight and score artifacts share this create-only leaf)
- Generate: `outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1/freeze/*`
- Generate: `outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1/evaluation/*`
- Generate: `reports/experiments/tethered_facet_minilm_diagnostic_v1/{artifact.json,summary.json,report_data.sqlite,report.html}`

**Interfaces:**
- Consumes: the committed Tasks 1--5 and authenticated existing artifacts only.
- Produces: the complete verified diagnostic and rendered local artifact.

- [ ] **Step 1: Run the full inference-free test and source verification suite**

Run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_tethered_facet_minilm_score.py \
  code/tests/test_tethered_facet_two_basket.py \
  code/tests/test_tethered_facet_evaluate.py \
  code/tests/test_build_tethered_facet_report.py \
  code/tests/test_deep_facet_candidate_score.py \
  code/tests/test_deep_facet_candidate_rank.py \
  code/tests/test_deep_facet_candidate_evaluate.py -q
```

Expected: zero failures.

- [ ] **Step 2: Create and inspect the real tokenizer-only preflight**

Run:

```bash
.venv/bin/python -m trec_rag.tethered_facet_minilm_score preflight \
  --manifest reports/experiments/deep_facet_candidate_pilot_v1/manifest.json \
  --phase1 outputs/rag25_deep_facet_candidates_v1/phase1_v1 \
  --gate outputs/rag25_deep_facet_candidates_v1/gate_v1 \
  --output outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1/scoring
```

Expected: exactly 4,800 candidate pairs, no protected topics, at most 25,000 windows, projected runtime at most 600 seconds, and no model construction. If any condition fails, stop and report the receipt; do not score.

- [ ] **Step 3: Run the bounded local ROCm scorer only after preflight passes**

Run:

```bash
.venv/bin/python-rocm -m trec_rag.tethered_facet_minilm_score score \
  --preflight outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1/scoring/preflight.json
```

Expected: exactly 4,800 aggregated document scores, no network or download, elapsed time at most 600 seconds, and a verified create-only scoring receipt. On a failed batch or ceiling, stop without automatic retry.

- [ ] **Step 4: Freeze and verify both two-basket rankings**

Run:

```bash
.venv/bin/python -m trec_rag.tethered_facet_two_basket freeze \
  --deep-root outputs/rag25_deep_facet_candidates_v1 \
  --tethered outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1/scoring \
  --output outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1/freeze
.venv/bin/python -m trec_rag.tethered_facet_two_basket verify \
  --freeze outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1/freeze
```

Expected: both arms have exact RRF top 100, exact per-topic populations, 200+200 valid basket rows at ranks 101--500, and complete duplicate-free permutations.

- [ ] **Step 5: Evaluate from the existing sealed projection and build the report**

Run:

```bash
.venv/bin/python -m trec_rag.tethered_facet_evaluate evaluate \
  --freeze outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1/freeze \
  --prior-evaluation outputs/rag25_deep_facet_candidates_v1/evaluation_v1 \
  --output outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1/evaluation
.venv/bin/python -m trec_rag.build_tethered_facet_report \
  --experiment outputs/rag25_deep_facet_candidates_v1/post_qrels_tethered_facet_minilm_v1 \
  --output reports/experiments/tethered_facet_minilm_diagnostic_v1
```

Expected: a mechanically reproduced pass/inconclusive/fail label and complete local report artifacts.

- [ ] **Step 6: Verify desktop/mobile rendering and artifact integrity**

Use Playwright to open `reports/experiments/tethered_facet_minilm_diagnostic_v1/report.html` at 1440×1000 and 390×844. Check no console errors, horizontal overflow, missing narrative/facet text, inaccessible tables, or broken internal navigation. Verify report source hashes against `artifact.json` and confirm no external requests occur.

- [ ] **Step 7: Run final verification and commit only requested artifacts**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_tethered_facet_minilm_score.py code/tests/test_tethered_facet_two_basket.py code/tests/test_tethered_facet_evaluate.py code/tests/test_build_tethered_facet_report.py -q
git diff --check
git status --short
```

Inspect generated-file sizes and repository policy before staging. The `outputs/`
tree is ignored and remains local; never force-add it. Stage only the generated
report artifacts beside the already committed README. Never stage cache files,
model files, local logs, screenshots, secrets, or unrelated untracked
sparse-relevance work. Commit with:

```bash
git add reports/experiments/tethered_facet_minilm_diagnostic_v1/artifact.json \
  reports/experiments/tethered_facet_minilm_diagnostic_v1/summary.json \
  reports/experiments/tethered_facet_minilm_diagnostic_v1/report_data.sqlite \
  reports/experiments/tethered_facet_minilm_diagnostic_v1/report.html
git commit -m "Run narrative-tethered facet MiniLM diagnostic"
```

Expected: tests pass, the report is reproducible from saved artifacts, and the final commit contains no unrelated files.

## Execution choice

Use **subagent-driven development** because each scoring, ranking, evaluation,
and reporting task has an independent test/review boundary. The primary agent
must verify every diff and test result before advancing, and it alone runs the
real preflight, bounded ROCm inference, final evaluation, and rendered artifact
checks.
