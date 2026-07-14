# Deep Facet Candidate Ranking Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a qrels-blind four-topic pilot that retrieves deep facet candidates, scores them locally and against common topic queries, and tests whether dual-score ranking preserves novel relevant documents at depths 500 and 1,000.

**Architecture:** A frozen manifest owns 25 source-tethered facets, four common queries, topic firewalls, and original-cache identities. A raw-first runner retrieves each facet to depth 200 through the persistent limiter; two authenticated local MiniLM phases score facet streams and then the accepted union. Offline stages freeze raw/accepted unions and five full permutations, seal every byte before qrels, evaluate once, and render a private HTML report with an independent advisor memo.

**Tech Stack:** Python 3.12, pytest, requests + requests-ratelimiter, repository retrieval ledger, Hugging Face Transformers, PyTorch/ROCm, `cross-encoder/ms-marco-MiniLM-L6-v2`, JSON/JSONL, Data Analytics portable HTML packager.

## Global Constraints

- Work in `/home/npatta01/.codex/worktrees/41f9/trec_rag_2026` on `codex/structured-query-planner`.
- Preserve v2.1 artifacts and unrelated untracked `sparse_relevance_*` files.
- Reject protected `144,213,224,407,515` and qrels-exposed `200,225,707,897,233,273,161,14` everywhere.
- Use exactly `219,72,300,84` in that order.
- Reuse original top 1,000; issue exactly 25 facet requests with `hits=200`.
- Use one request start per three seconds through the persistent limiter; never retry an immutable failure.
- Use only local MiniLM revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`; no download or hosted inference.
- Stop if a scoring phase projects above 600 seconds or authenticated model/window receipts differ.
- Narrative MiniLM is never an eligibility gate; raw scores never cross query boundaries.
- Freeze `U_raw`, `U_accepted`, gates, scores, parameters, complete permutations, and `SEALED.json` before qrels.
- Mutating stages refuse to run after `QRELS_ACCESSED` exists.
- No generation, citation work, dense primary retrieval, recursive repair, paid call, or post-qrels tuning.

## File Map

- `code/trec_rag/deep_facet_candidate_manifest.py`: manifest, cache bindings, analyzer terms, topic firewall.
- `code/trec_rag/deep_facet_candidate_run.py`: 25-request preflight and raw-first retrieval.
- `code/trec_rag/deep_facet_candidate_score.py`: two MiniLM preflights, scoring, aggregation, cache receipts.
- `code/trec_rag/deep_facet_candidate_gate.py`: gate, unions, BM25/MiniLM prefix unions.
- `code/trec_rag/deep_facet_candidate_rank.py`: percentiles, five permutations, redundancy, seal.
- `code/trec_rag/deep_facet_candidate_evaluate.py`: one-way qrels access, metrics, advance rule.
- `code/trec_rag/build_deep_facet_candidate_report.py`: source-bound report and HTML.
- Matching tests: `code/tests/test_deep_facet_candidate_*.py` and `code/tests/test_build_deep_facet_candidate_report.py`.
- Durable report: `reports/experiments/deep_facet_candidate_pilot_v1/`.
- Untracked run evidence: `outputs/rag25_deep_facet_candidates_v1/`.

---

### Task 1: Freeze manifest and state boundary

**Files:**
- Create `code/trec_rag/deep_facet_candidate_manifest.py`
- Create `code/tests/test_deep_facet_candidate_manifest.py`
- Create `reports/experiments/deep_facet_candidate_pilot_v1/manifest.json`
- Create `reports/experiments/deep_facet_candidate_pilot_v1/README.md`

**Interfaces:** `build_manifest(cache_root)`, `validate_manifest(payload)`, `load_manifest(path)`, `assert_mutation_allowed(output_root)`.

- [ ] **Step 1: Write failing manifest/firewall tests**

```python
def test_exact_boundary(cache_root):
    value = build_manifest(cache_root)
    assert value["topic_ids"] == ["219", "72", "300", "84"]
    assert len(value["facets"]) == 25
    assert sum(f["topic_id"] == "219" for f in value["facets"]) == 7
    assert all(t["original_hits"] == 1000 for t in value["topics"])

def test_sentinel_refuses_mutation(tmp_path):
    (tmp_path / "QRELS_ACCESSED").write_text("sealed\n")
    with pytest.raises(ValueError, match="qrels already accessed"):
        assert_mutation_allowed(tmp_path)
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest code/tests/test_deep_facet_candidate_manifest.py -q`**

Expected: FAIL because the module is absent.

- [ ] **Step 3: Implement exact spec records and canonical validation**

Copy approved narratives/common queries/facets/analyzer terms. Bind the four original cache files; require `hits=1000`, exactly 1,000 unique text-bearing candidates, exact topic/query, and SHA-256. Canonical JSON uses sorted keys, UTF-8, one trailing newline. Every entrypoint checks the sentinel before reading sources.

- [ ] **Step 4: Verify and create manifest**

```bash
.venv/bin/python -m pytest code/tests/test_deep_facet_candidate_manifest.py -q
.venv/bin/python -m trec_rag.deep_facet_candidate_manifest create \
  --cache-root /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote \
  --output reports/experiments/deep_facet_candidate_pilot_v1/manifest.json
```

Expected: PASS; `topic_count=4`, `facet_count=25`, `original_rows=4000`, `qrels_opened=false`.

- [ ] **Step 5: Commit**

```bash
git add code/trec_rag/deep_facet_candidate_manifest.py code/tests/test_deep_facet_candidate_manifest.py reports/experiments/deep_facet_candidate_pilot_v1
git commit -m "Freeze deep facet candidate manifest"
```

### Task 2: Retrieve 25 facet streams

**Files:** Create `deep_facet_candidate_run.py` and its test; generate `retrieval_v1/`.

**Interfaces:** `build_requests()`, `preflight_retrieval()`, `execute_retrieval()`; produces canonical 5,000-row candidates, summary, ledger, raw responses.

- [ ] **Step 1: Write failing request/response tests**

```python
def test_exact_requests(manifest):
    rows = build_requests(manifest, ENDPOINT)
    assert len(rows) == 25
    assert len({r.identity.request_key for r in rows}) == 25
    assert all(r.identity.hits == 200 for r in rows)

def test_short_response_aborts(fake_transport, manifest, tmp_path):
    fake_transport.candidates = fake_transport.candidates[:199]
    with pytest.raises(ValueError, match="exactly 200 unique"):
        execute_retrieval(manifest, fake_transport, tmp_path)
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest code/tests/test_deep_facet_candidate_run.py -q`**

Expected: FAIL because runner is absent.

- [ ] **Step 3: Implement raw-first preflight/run**

Reuse `facet_aware_fusion_run.py` patterns, endpoint `http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search`, `climbmix-400b`, ledger, exact cache, persistent SQLite limiter. Enforce 25 calls, depth 200, 75-second minimum span, no retry, failure bytes, 200 unique text-bearing docs per response.

- [ ] **Step 4: Verify and execute**

```bash
.venv/bin/python -m pytest code/tests/test_deep_facet_candidate_run.py code/tests/test_deep_facet_candidate_manifest.py -q
.venv/bin/python -m trec_rag.deep_facet_candidate_run preflight --manifest reports/experiments/deep_facet_candidate_pilot_v1/manifest.json --output outputs/rag25_deep_facet_candidates_v1/retrieval_v1
.venv/bin/python -m trec_rag.deep_facet_candidate_run run --manifest reports/experiments/deep_facet_candidate_pilot_v1/manifest.json --output outputs/rag25_deep_facet_candidates_v1/retrieval_v1
```

Expected: tests PASS; 25 successes, 5,000 rows, zero failures/qrels.

- [ ] **Step 5: Commit runner/tests; keep live evidence untracked**

```bash
git add code/trec_rag/deep_facet_candidate_run.py code/tests/test_deep_facet_candidate_run.py
git commit -m "Add deep facet retrieval runner"
```

### Task 3: Facet-local MiniLM and qrels-blind gate

**Files:** Create score/gate modules and tests; generate `phase1_v1/` and `gate_v1/`.

**Interfaces:** `aggregate_top4()`, `build_phase_preflight()`, `run_local_scoring()`, `quality_gate()`, `build_unions()`, `build_prefix_unions()`.

- [ ] **Step 1: Write failing aggregation/two-union tests**

```python
def test_top4_span_distinct():
    rows = scored_windows([9, 8, 7], [(0, 300), (100, 350), (300, 600)])
    assert aggregate_top4(rows) == pytest.approx((.55 * 9 + .25 * 7) / .80)

def test_gate_loss_is_visible(original, accepted, rejected):
    unions = build_unions(original, [accepted, rejected])
    assert rejected.unique_doc in unions.raw_docids
    assert rejected.unique_doc not in unions.accepted_docids
```

- [ ] **Step 2: Run score/gate tests; expect missing modules**

- [ ] **Step 3: Implement phase-1 scoring**

Reuse verified tokenizer/window/cache/ROCm primitives. Bind prior receipt path and SHA `346f3239373924d3ed8dc76076dfb3221c0a29ab8ab24ce71a1e5f98ce180122`. Require token policy `512/192/256`, overlap 64, max 32 windows, 128 new span tokens, top-four weights `.55/.25/.13/.07`. Preflight reports exact windows/cache/runtime/memory and refuses runtime above 600 seconds or downloads.

- [ ] **Step 4: Implement gate/unions**

Rank each 200-row stream by aggregate score; accept with anchor `>=3/5`, anchor+relation `>=2/5`, wrong-domain `<2/5`. Warnings alone cannot reject. Keep status categories separate. Build `U_raw`, `U_accepted`, and both BM25/MiniLM prefix unions at 50/100/200 with provenance.

- [ ] **Step 5: Verify and execute phase 1/gate**

```bash
.venv/bin/python -m pytest code/tests/test_deep_facet_candidate_score.py code/tests/test_deep_facet_candidate_gate.py -q
.venv/bin/python -m trec_rag.deep_facet_candidate_score preflight-phase1 --manifest reports/experiments/deep_facet_candidate_pilot_v1/manifest.json --retrieval outputs/rag25_deep_facet_candidates_v1/retrieval_v1 --output outputs/rag25_deep_facet_candidates_v1/phase1_v1
.venv/bin/python-rocm -m trec_rag.deep_facet_candidate_score score-phase1 --preflight outputs/rag25_deep_facet_candidates_v1/phase1_v1/preflight.json
.venv/bin/python -m trec_rag.deep_facet_candidate_gate freeze --manifest reports/experiments/deep_facet_candidate_pilot_v1/manifest.json --retrieval outputs/rag25_deep_facet_candidates_v1/retrieval_v1 --phase1 outputs/rag25_deep_facet_candidates_v1/phase1_v1 --output outputs/rag25_deep_facet_candidates_v1/gate_v1
```

Expected: tests PASS; runtime <=600 seconds; 25 terminal decisions; two unions/six prefixes; no qrels.

- [ ] **Step 6: Commit score/gate code/tests**

```bash
git add code/trec_rag/deep_facet_candidate_score.py code/tests/test_deep_facet_candidate_score.py code/trec_rag/deep_facet_candidate_gate.py code/tests/test_deep_facet_candidate_gate.py
git commit -m "Add deep facet scoring and stream gate"
```

### Task 4: Common/narrative scoring over `U_accepted`

**Files:** Modify score module/tests; generate `phase2_v1/`.

- [ ] **Step 1: Write failing coverage/non-gating tests**

```python
def test_each_accepted_doc_scored_twice(union, manifest):
    plan = build_phase2_plan(union, manifest)
    assert plan.query_document_pair_count == 2 * unique_topic_docs(union)
    assert all("decision" not in row for row in plan.rows)
```

- [ ] **Step 2: Run score tests; expect missing phase-2 behavior**

- [ ] **Step 3: Implement exact `{topic}:common` and `{topic}:narrative` plan**

Use only `U_accepted` text/hash and the same engine. Require exactly both aggregates per topic/document and no extras. Narrative scores cannot change union membership.

- [ ] **Step 4: Verify and execute phase 2**

```bash
.venv/bin/python -m pytest code/tests/test_deep_facet_candidate_score.py -q
.venv/bin/python -m trec_rag.deep_facet_candidate_score preflight-phase2 --manifest reports/experiments/deep_facet_candidate_pilot_v1/manifest.json --gate outputs/rag25_deep_facet_candidates_v1/gate_v1 --output outputs/rag25_deep_facet_candidates_v1/phase2_v1
.venv/bin/python-rocm -m trec_rag.deep_facet_candidate_score score-phase2 --preflight outputs/rag25_deep_facet_candidates_v1/phase2_v1/preflight.json
```

Expected: runtime <=600 seconds; two-score coverage; zero qrels/retrieval/hosted calls.

- [ ] **Step 5: Commit phase-2 support**

```bash
git add code/trec_rag/deep_facet_candidate_score.py code/tests/test_deep_facet_candidate_score.py
git commit -m "Add accepted union semantic scoring"
```

### Task 5: Five complete permutations and seal

**Files:** Create rank module/test; generate `freeze_v1/`.

**Interfaces:** `rank_percentile()`, `redundancy_penalty()`, `build_permutations()`, `create_seal()`, `verify_seal()`.

- [ ] **Step 1: Write failing percentile/redundancy/permutation/seal tests**

```python
def test_percentile_ties():
    assert rank_percentile({"a": 9, "b": 7, "c": 7}) == {"a": 1, "b": .5, "c": .5}

def test_redundancy_threshold():
    assert redundancy_penalty(.79) == 0
    assert redundancy_penalty(.90) == pytest.approx(.5)

def test_permutations_are_complete_and_order_invariant(inputs):
    assert build_permutations(inputs) == build_permutations(reversed_inputs(inputs))
    assert all(set(v) == inputs.accepted_docids for v in build_permutations(inputs).values())
```

- [ ] **Step 2: Run rank tests; expect missing module**

- [ ] **Step 3: Implement features/arms**

Use `P=(n-average_rank+1)/n`, absent facet zero, `L=max F`, `C=max selected F`, `B=max F*(1-C)`. RRF `k=60`, family weights .5/.5. GLOBAL `.70G+.30N`; FACET `.70L+.30B`; DUAL `.35G+.15N+.15R+.25L+.10B-.15D`; DUAL-NR fixes `D=0`. Implement exact ties, zero-facet fallback/non-advance, complete permutations, incremental updates, Jaccard threshold .80.

- [ ] **Step 4: Implement create-only `SEALED.json`**

Record every pre-qrels path, bytes, hash, schema, rows, topic order, root hash; exclude seal/evaluation. Reject missing/extra/mutated inputs.

- [ ] **Step 5: Verify and freeze**

```bash
.venv/bin/python -m pytest code/tests/test_deep_facet_candidate_rank.py code/tests/test_deep_facet_candidate_gate.py code/tests/test_deep_facet_candidate_score.py -q
.venv/bin/python -m trec_rag.deep_facet_candidate_rank freeze --manifest reports/experiments/deep_facet_candidate_pilot_v1/manifest.json --retrieval outputs/rag25_deep_facet_candidates_v1/retrieval_v1 --phase1 outputs/rag25_deep_facet_candidates_v1/phase1_v1 --gate outputs/rag25_deep_facet_candidates_v1/gate_v1 --phase2 outputs/rag25_deep_facet_candidates_v1/phase2_v1 --output outputs/rag25_deep_facet_candidates_v1/freeze_v1
.venv/bin/python -m trec_rag.deep_facet_candidate_rank verify --freeze outputs/rag25_deep_facet_candidates_v1/freeze_v1
```

Expected: tests PASS; five full permutations/topic; valid prefixes and seal.

- [ ] **Step 6: Commit rank/seal code/tests**

```bash
git add code/trec_rag/deep_facet_candidate_rank.py code/tests/test_deep_facet_candidate_rank.py
git commit -m "Freeze deep facet candidate rankings"
```

### Task 6: One-way qrels evaluation

**Files:** Create evaluator/test; generate `evaluation_v1/`.

**Interfaces:** `create_qrels_sentinel()`, `evaluate_set()`, `evaluate_ranking()`, `decide()`.

- [ ] **Step 1: Write failing firewall/aggregation tests**

```python
def test_unsealed_freeze_cannot_open_qrels(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="seal"):
        evaluate(tmp_path, qrels_loader=lambda: calls.append(1))
    assert calls == []

def test_advance_aggregation(fixture):
    d = decide(fixture)
    assert d["aggregation"]["graded_recall"] == "macro_non_null"
    assert d["aggregation"]["novel_retention"] == "micro_topic_document"
```

- [ ] **Step 2: Run evaluator tests; expect missing module**

- [ ] **Step 3: Implement one-way access/metrics**

Verify seal, atomically create sentinel, then project only exact topics. Use grade >=2, gain `2^grade-1`, discount `1/log2(rank+1)`, unjudged gain zero plus judged rates, null zero denominators, macro arm comparisons, micro NovelRel, inclusive/exclusive attribution, gate loss, prefix metrics, nDCG@10/@100, Recall/graded Recall@100/@500/@1000, leave-one-out.

- [ ] **Step 4: Implement advance diagnosis**

Only `advance_to_larger_validation` exists. Diagnose retrieval when raw adds nothing, gating when raw additions are lost, fusion when accepted additions are omitted; advance only if every frozen guard passes.

- [ ] **Step 5: Verify and evaluate once**

```bash
.venv/bin/python -m pytest code/tests/test_deep_facet_candidate_evaluate.py code/tests/test_deep_facet_candidate_rank.py -q
.venv/bin/python -m trec_rag.deep_facet_candidate_evaluate evaluate --freeze outputs/rag25_deep_facet_candidates_v1/freeze_v1 --output outputs/rag25_deep_facet_candidates_v1/evaluation_v1
```

Expected: tests PASS; one sentinel/receipt; complete metrics and decision.

- [ ] **Step 6: Commit evaluator/tests**

```bash
git add code/trec_rag/deep_facet_candidate_evaluate.py code/tests/test_deep_facet_candidate_evaluate.py
git commit -m "Evaluate deep facet candidate pilot"
```

### Task 7: Advisor review, artifact, and verification

**Files:** Create report builder/test, artifact, summary, HTML, advisor memo under `reports/experiments/deep_facet_candidate_pilot_v1/`.

- [ ] **Step 1: Write failing report/provenance tests**

```python
def test_report_separates_failure_stages(built):
    text = report_markdown(built.artifact)
    assert "U_raw" in text and "U_accepted" in text
    assert built.summary["diagnosis"] in {"retrieval", "gating", "fusion", "advance"}

def test_mutated_sealed_source_rejected(inputs):
    with pytest.raises(ValueError, match="seal"):
        build_report(mutate_source(inputs, "DUAL.jsonl"))
```

- [ ] **Step 2: Run report tests; expect missing builder**

- [ ] **Step 3: Obtain post-results advisor review**

Provide specification, request/runtime receipts, gates, unions, metrics, judged rates, leave-one-out, decision. Require retrieval/gating/fusion verdict and one bounded next experiment. Save verbatim; it cannot change metrics/rankings.

- [ ] **Step 4: Implement/render private HTML**

Include answer-first verdict, funnel, depth sensitivity, gate losses, arm metrics, per-topic deltas, redundancy diagnostic, cost/runtime, judging caveat, advisor memo, hashes.

```bash
.venv/bin/python -m trec_rag.build_deep_facet_candidate_report --artifact reports/experiments/deep_facet_candidate_pilot_v1/artifact.json --summary reports/experiments/deep_facet_candidate_pilot_v1/summary.json --output reports/experiments/deep_facet_candidate_pilot_v1/report.html --tmpdir /home/npatta01/data/competitions/trec_rag_2026/cache/tmp/deep-facet-candidate-report-v1
```

Expected: package/source interaction/1440px/390px verification pass.

- [ ] **Step 5: Complete verification**

```bash
.venv/bin/python -m pytest code/tests/test_deep_facet_candidate_*.py code/tests/test_build_deep_facet_candidate_report.py -q
git diff --check
git status --short
```

Expected: targeted tests pass; no whitespace errors; only unrelated untracked sparse-relevance files remain.

- [ ] **Step 6: Commit report**

```bash
git add code/trec_rag/build_deep_facet_candidate_report.py code/tests/test_build_deep_facet_candidate_report.py reports/experiments/deep_facet_candidate_pilot_v1
git commit -m "Report deep facet candidate pilot findings"
```

## Execution Checkpoints

1. Stop after retrieval if any immutable request failed or returned fewer than 200 unique text-bearing documents.
2. Stop after either scoring preflight only if runtime exceeds 600 seconds, receipt/materialization differs, or uncached-pair ceiling is exceeded.
3. Report the qrels-blind seal hash before evaluation.
4. After qrels, never rerun mutating upstream stages.
5. Final evidence is descriptive and can only advance DUAL to larger validation.
