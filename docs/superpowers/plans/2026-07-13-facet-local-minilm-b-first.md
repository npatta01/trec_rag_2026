# Facet-Local MiniLM B-First Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a four-topic, facet-local MiniLM pilot that measures candidate-pool headroom, promotes relevant documents buried in facet streams, and diagnoses retrieval, reranking, and fusion gaps before any lexical Stage A work.

**Architecture:** Reconstruct the exact frozen R1 candidate population from verified ledgers, create a self-contained 31-stream manifest, and tokenize every query/document pair under a bounded deterministic window policy. Score all windows once into a content-addressed cache, derive facet-only and diagnostic ranking arms offline, freeze rankings and blinded facet-review labels before qrels, then evaluate recall gains and losses plus ranking guardrails.

**Tech Stack:** Python 3.12, pytest, Hugging Face Transformers, PyTorch/ROCm through `.venv/bin/python-rocm`, `cross-encoder/ms-marco-MiniLM-L6-v2`, the existing retrieval ledger and evaluation utilities, family-balanced RRF, and the packaged Data Analytics HTML renderer.

## Global Constraints

- Work in `/home/npatta01/.codex/worktrees/41f9/trec_rag_2026` on `codex/structured-query-planner`.
- Preserve every v2.1 artifact and earlier immutable ledger.
- Use only topics `200`, `225`, `707`, and `897`.
- Reject protected topics `144`, `213`, `224`, `407`, and `515` before every source read, tokenization, cache lookup, inference, fusion, review, evaluation, and report stage.
- Make zero new retrieval calls and zero paid or hosted inference calls.
- Pin model revision `c5ee24cb16019beea0893ab7796b1df96625c6b8` and use only safe Transformers files plus `model.safetensors`.
- Run tokenizer preflight without qrels or model forward passes.
- Cap the benchmark at 256 pairs and require explicit approval before it runs.
- Require a second explicit approval after exact pair-count/runtime estimation and before full inference.
- Freeze every ranking and blinded-review artifact before a separately
  authorized four-topic qrels projection can open. Never pass the all-topic
  qrels file to this pipeline.
- Treat results as descriptive four-topic pilot evidence.
- Do not implement Stage A in this plan.
- Keep generated run artifacts under `outputs/rag25_facet_local_minilm_v1/` and durable report artifacts under `reports/experiments/facet_local_minilm_pilot_v1/`.
- Use `.venv/bin/python` for qrels-free utilities and `.venv/bin/python-rocm` only for approved inference.
- Stage only files named by the active task; preserve unrelated untracked files.

## File map

- `code/trec_rag/facet_local_minilm_manifest.py`: rebuild and authenticate the exact R1 stream population.
- `code/trec_rag/facet_local_minilm_preflight.py`: pin model/tokenizer identity and materialize bounded query-window plans without inference.
- `code/trec_rag/facet_local_minilm_score.py`: benchmark and full scoring with a content-addressed cache and approval receipts.
- `code/trec_rag/facet_local_minilm_rank.py`: aggregate windows, rerank streams, fuse all arms, and create the pre-qrels ranking freeze.
- `code/trec_rag/facet_local_minilm_review.py`: create, validate, adjudicate, and freeze the blinded facet-review pool.
- `code/trec_rag/facet_local_minilm_evaluate.py`: enforce the qrels firewall and compute headroom, gain/loss, recall, and guardrail metrics.
- `code/trec_rag/build_facet_local_minilm_report.py`: build the source-backed analytical artifact.
- Matching tests live under `code/tests/`.

---

### Task 1: Freeze the exact 31-stream source manifest

**Files:**
- Create: `code/trec_rag/facet_local_minilm_manifest.py`
- Create: `code/tests/test_facet_local_minilm_manifest.py`
- Create: `code/tools/build_facet_local_minilm_manifest.py`
- Create: `reports/experiments/facet_local_minilm_pilot_v1/manifest.json`
- Create: `reports/experiments/facet_local_minilm_pilot_v1/README.md`
- Generate, do not commit: `outputs/rag25_facet_local_minilm_v1/source_v1/candidates.jsonl`
- Generate, do not commit: `outputs/rag25_facet_local_minilm_v1/source_v1/source_receipt.json`

**Interfaces:**
- Consumes: the prior R1 freeze, the verified prompt-lab base ledger/cache, the verified R1 ledger/cache, and the 22-stream R1 manifest.
- Produces: `FacetLocalManifest`, `FacetLocalStream`, `load_facet_local_manifest(path)`, `build_facet_local_manifest(...)`, and a self-contained immutable stream/candidate snapshot with a source receipt.

- [ ] **Step 1: Write failing source-boundary tests**

```python
def test_manifest_has_exact_r1_population(frozen_inputs):
    manifest = build_facet_local_manifest(**frozen_inputs)
    assert manifest.topic_ids == ("200", "225", "707", "897")
    assert len(manifest.streams) == 31
    assert sum(stream.expected_rows for stream in manifest.streams) == 3100
    assert manifest.stream_counts == {"200": 10, "225": 8, "707": 4, "897": 9}
    assert sum(stream.family == "original" for stream in manifest.streams) == 4
    assert sum(stream.family == "facet" for stream in manifest.streams) == 27

def test_protected_topic_rejected_before_source_loader(monkeypatch):
    monkeypatch.setattr(module, "PILOT_TOPIC_IDS", ("144",))
    with pytest.raises(ValueError, match="protected topic 144"):
        build_facet_local_manifest(source_loader=lambda: pytest.fail("source read"))
```

Also test exact five retained facet names, exact 22 repaired facets, ranks 1--100, unique document IDs within a stream, full query/text hashes, prior-freeze hash, candidate bytes, deterministic serialization, create-only output, and rejection if a later loader attempts to reopen a source ledger instead of the frozen snapshot.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest code/tests/test_facet_local_minilm_manifest.py -q`

Expected: import failure because the manifest module does not exist.

- [ ] **Step 3: Implement the source loader and manifest**

Use `RetrievalLedger.validate_run()` and `load_verified_result()` directly. Load:

```text
base run:  outputs/rag25_det_sparse_prompt_lab_v1/rate_limited_continuation_v1
base cache: /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/rag25_det_sparse_prompt_lab_base_http_restart_v1
R1 run:    outputs/rag25_sparse_relevance_paired_v1/run_v2/R1/ledger
R1 cache:  /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/rag25_sparse_relevance_paired_v1
prior freeze: outputs/rag25_sparse_relevance_paired_v1/freeze_v1
```

Copy the exact `KEPT_BASE_FACETS` set into the new manifest contract rather than importing a historical implementation module. For each stream, store topic, family, variant, exact query, query hash, source ledger/cache binding, expected depth, and a canonical 100-row candidate hash. Write every candidate's rank, document ID, full text, text hash, and source score to create-only `outputs/rag25_facet_local_minilm_v1/source_v1/candidates.jsonl`. Bind its schema, 3,100-row count, bytes, and SHA-256 in both `source_receipt.json` and the durable manifest. After this step, downstream code accepts the snapshot and receipt only.

- [ ] **Step 4: Generate and verify the durable manifest**

```bash
.venv/bin/python code/tools/build_facet_local_minilm_manifest.py \
  --r1-manifest reports/experiments/sparse_relevance_pilot_v1/r1_manifest.json \
  --prior-freeze outputs/rag25_sparse_relevance_paired_v1/freeze_v1 \
  --base-run outputs/rag25_det_sparse_prompt_lab_v1/rate_limited_continuation_v1 \
  --base-cache /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/rag25_det_sparse_prompt_lab_base_http_restart_v1 \
  --r1-run outputs/rag25_sparse_relevance_paired_v1/run_v2/R1/ledger \
  --r1-cache /home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/rag25_sparse_relevance_paired_v1 \
  --source-output outputs/rag25_facet_local_minilm_v1/source_v1 \
  --manifest-output reports/experiments/facet_local_minilm_pilot_v1/manifest.json
.venv/bin/python -m pytest code/tests/test_facet_local_minilm_manifest.py -q
```

Expected: exactly 31 streams, 3,100 candidate rows, and no protected topics.

- [ ] **Step 5: Commit**

```bash
git add code/trec_rag/facet_local_minilm_manifest.py code/tests/test_facet_local_minilm_manifest.py code/tools/build_facet_local_minilm_manifest.py reports/experiments/facet_local_minilm_pilot_v1/manifest.json reports/experiments/facet_local_minilm_pilot_v1/README.md
git commit -m "Freeze facet-local MiniLM source manifest"
```

### Task 2: Materialize pinned model files and build the tokenizer-only preflight

**Files:**
- Create: `code/trec_rag/facet_local_minilm_preflight.py`
- Create: `code/tests/test_facet_local_minilm_preflight.py`

**Interfaces:**
- Consumes: `FacetLocalManifest`, the immutable source snapshot, a user approval receipt for the allowlisted model download, a verified local pinned tokenizer snapshot, and the shared MiniLM score cache.
- Produces: a safe-model materialization receipt, `WindowPlanRow`, `build_window_plan()`, `verify_window_plan()`, `preflight.json`, and `windows.jsonl`.

- [ ] **Step 1: Write failing token/window-policy tests**

```python
def test_window_policy_is_bounded_and_covers_document_edges(tokenizer, candidate):
    rows = build_window_plan(candidate, tokenizer, query="anchored facet")
    assert 1 <= len(rows) <= 32
    assert rows[0].document_start_token == 0
    assert rows[-1].document_end_token == candidate.document_token_count
    assert all(row.pair_token_count <= 512 for row in rows)

def test_overlong_query_fails_without_truncation(tokenizer, candidate):
    with pytest.raises(ValueError, match="query exceeds 192 tokens"):
        build_window_plan(candidate, tokenizer, query="token " * 193)
```

Also test refusal to materialize without an approval receipt exactly bound to model/revision/allowlist; 64-token overlap; minimum 256-token passage budget; the exact `round_half_up(j * (N - 1) / 31)` 32-window index rule and uniqueness; stable window IDs; query/document/window hashes; exact cache-key derivation; 100,000-miss ceiling; capped-document counts; selected-window token-coverage fractions and per-stream min/median/p95; the deterministic benchmark sample hash; no model construction; no qrels path; protected-topic rejection before tokenizer/cache access; and deterministic output under input reordering.

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest code/tests/test_facet_local_minilm_preflight.py -q`

Expected: import failure.

- [ ] **Step 3: Implement local safe-model resolution**

Pin:

```python
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
ALLOW_PATTERNS = (
    "config.json", "model.safetensors", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.txt",
)
```

The materializer requires a receipt bound to the user's approval and calls
`huggingface_hub.snapshot_download()` with the pinned revision and exact
allowlist. It must not construct the tokenizer or model. Run it once:

```bash
.venv/bin/python -m trec_rag.facet_local_minilm_preflight materialize \
  --model-id cross-encoder/ms-marco-MiniLM-L6-v2 \
  --revision c5ee24cb16019beea0893ab7796b1df96625c6b8 \
  --approval outputs/rag25_facet_local_minilm_v1/approvals/model_download_v1.json \
  --output outputs/rag25_facet_local_minilm_v1/model_v1
```

The create-only receipt records the resolved snapshot path and exact files,
bytes, and SHA-256 hashes. Reject pickle weights and unexpected files. Every
subsequent load uses `local_files_only=True` and verifies this receipt first.

- [ ] **Step 4: Implement and run the qrels-free preflight**

```bash
.venv/bin/python -m trec_rag.facet_local_minilm_preflight \
  --manifest reports/experiments/facet_local_minilm_pilot_v1/manifest.json \
  --source outputs/rag25_facet_local_minilm_v1/source_v1 \
  --model-receipt outputs/rag25_facet_local_minilm_v1/model_v1/materialization.json \
  --score-cache /home/npatta01/data/competitions/trec_rag_2026/cache/reranker/score_cache \
  --output outputs/rag25_facet_local_minilm_v1/preflight_v1
```

Expected: exact total and per-stream window counts, cache hits/misses, maximum lengths, local snapshot bytes/hashes, capped-document counts, token-coverage min/median/p95, a hash-frozen benchmark sample, and zero inference/qrels/retrieval counters. Stop if uncached pairs exceed 100,000.

- [ ] **Step 5: Test and commit**

```bash
.venv/bin/python -m pytest code/tests/test_facet_local_minilm_preflight.py code/tests/test_facet_local_minilm_manifest.py -q
git add code/trec_rag/facet_local_minilm_preflight.py code/tests/test_facet_local_minilm_preflight.py
git commit -m "Add MiniLM tokenizer cost preflight"
```

### Task 3: Implement the gated MiniLM score cache and runner

**Files:**
- Create: `code/trec_rag/facet_local_minilm_score.py`
- Create: `code/tests/test_facet_local_minilm_score.py`

**Interfaces:**
- Consumes: verified `preflight.json`, `windows.jsonl`, pinned model-materialization receipt, benchmark/full approval receipts, and `GlobalScoreCache`.
- Produces: `MiniLMScoreRow`, benchmark telemetry, immutable scoring ledger, and complete raw-logit cache entries.

- [ ] **Step 1: Write failing approval/cache tests**

```python
def test_benchmark_refuses_more_than_256_pairs(preflight, approval):
    with pytest.raises(ValueError, match="256-pair benchmark ceiling"):
        run_benchmark(preflight, approval, pair_limit=257)

def test_full_run_requires_exact_preflight_and_approval_hashes(preflight):
    with pytest.raises(ValueError, match="full inference approval"):
        run_full_scoring(preflight, approval=None)

def test_cache_identity_binds_query_window_and_model(context):
    left = score_cache_key(context, query="facet one", window="same text")
    right = score_cache_key(context, query="facet two", window="same text")
    assert left != right
```

Also test float32/raw-logit metadata, pinned revision, `local_files_only`, batch size 32, evaluation/no-grad mode, no cache overwrite with conflicting scores, resume from exact cache hits, OOM fail-closed behavior, ROCm probe before model load, CPU fallback requiring new approval, protected-topic rejection, no qrels/retrieval/network access, and exact benchmark sampling: for at least 96 misses, 32 longest warm-up pairs plus a hash-frozen 64-pair length-stratified sample timed three times, for 224 total forward pairs.

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest code/tests/test_facet_local_minilm_score.py -q`

Expected: import failure.

- [ ] **Step 3: Implement the runner and cache contract**

Use `AutoTokenizer` and `AutoModelForSequenceClassification` from the pinned local snapshot. Write raw logits to a model/query/window content-addressed cache. The immutable scoring ledger records reservation, cache hit or forward pass, score, elapsed time, device memory, host memory, and output hash for every planned pair.

Benchmark command after explicit approval:

```bash
.venv/bin/python-rocm -m trec_rag.facet_local_minilm_score benchmark \
  --preflight outputs/rag25_facet_local_minilm_v1/preflight_v1/preflight.json \
  --approval outputs/rag25_facet_local_minilm_v1/approvals/benchmark_v1.json \
  --pair-limit 256 \
  --output outputs/rag25_facet_local_minilm_v1/benchmark_v1
```

For at least 96 misses, run one 32-pair longest-length warm-up, then time the
frozen 64-pair sample three times at batch size 32. Use median throughput and
project wall time as `1.25 * uncached_pairs / median_pairs_per_second`. For
1--95 misses, use the frozen small-sample rule from the design and multiplier
`1.50`; zero misses skip the benchmark. Retain peak host/device memory across
warm-up and timed passes, and write `full_inference_request.json` for user
review. Refuse any benchmark whose sample hash differs from preflight.

- [ ] **Step 4: Execute the full scorer only after the second approval**

```bash
.venv/bin/python-rocm -m trec_rag.facet_local_minilm_score run \
  --preflight outputs/rag25_facet_local_minilm_v1/preflight_v1/preflight.json \
  --approval outputs/rag25_facet_local_minilm_v1/approvals/full_inference_v1.json \
  --score-cache /home/npatta01/data/competitions/trec_rag_2026/cache/reranker/score_cache \
  --output outputs/rag25_facet_local_minilm_v1/scoring_v1
```

Expected: all planned pairs are exact cache hits or successful forward passes; zero failed/pending rows; score count and Merkle/root hash match preflight.

- [ ] **Step 5: Test and commit**

```bash
.venv/bin/python -m pytest code/tests/test_facet_local_minilm_score.py -q
git add code/trec_rag/facet_local_minilm_score.py code/tests/test_facet_local_minilm_score.py
git commit -m "Add gated facet-local MiniLM scoring"
```

### Task 4: Aggregate passages, build arms, and freeze rankings

**Files:**
- Create: `code/trec_rag/facet_local_minilm_rank.py`
- Create: `code/tests/test_facet_local_minilm_rank.py`

**Interfaces:**
- Consumes: authenticated source candidates, complete MiniLM scores, and the frozen fusion definition.
- Produces: per-stream top-four and MaxP rankings plus frozen `C0`, `BF100`, `BF50`, `BF20`, `BO100`, `BB100`, and `BF100_MAXP` system rankings.

- [ ] **Step 1: Write failing aggregation and matrix tests**

```python
def test_top4_span_distinct_aggregation_uses_frozen_weights():
    rows = windows_with_scores([9.0, 8.0, 7.0, 6.0], disjoint=True)
    assert aggregate_top4(rows) == pytest.approx(
        0.55 * 9.0 + 0.25 * 8.0 + 0.13 * 7.0 + 0.07 * 6.0
    )

def test_primary_arm_reranks_facets_only(frozen_candidates, scores):
    arms = build_b_arms(frozen_candidates, scores)
    assert original_order(arms["BF100"]) == original_order(arms["C0"])
    assert facet_order(arms["BF100"]) != facet_order(arms["C0"])
```

Also test 128-new-token span eligibility, fewer-than-four weight renormalization, MaxP sensitivity, prior-rank/docid tie-breaks, exact retention depths, 31-stream preservation for BF100, facet-only removal for BF50/BF20, family weights summing to 1.0 for every topic, original weight 0.5, no raw-score comparison across streams, deterministic RRF/provenance, output depth 100, protected-topic rejection, ranking stability under input reordering, and byte/hash identity between C0 and the prior frozen R1 family-RRF baseline.

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest code/tests/test_facet_local_minilm_rank.py -q`

Expected: import failure.

- [ ] **Step 3: Implement stream aggregation and the exact arm matrix**

Use within-stream MiniLM ranks as the only cross-stream input. BF50/BF20 truncate each facet after MiniLM reranking but keep the original depth-100 BM25 stream. BO100 reranks only originals. BB100 reranks both families. BF100_MAXP changes only aggregation. Publish C0 as a byte-for-byte copy of the prior verified `R1__family_rrf.jsonl`. Independently recompute its canonical `(topic_id, rank, document_id)` projection from the 31 frozen streams and require that hash to match the corresponding canonical projection of prior R1.

- [ ] **Step 4: Create and independently verify the pre-qrels ranking freeze**

```bash
.venv/bin/python -m trec_rag.facet_local_minilm_rank \
  --manifest reports/experiments/facet_local_minilm_pilot_v1/manifest.json \
  --preflight outputs/rag25_facet_local_minilm_v1/preflight_v1 \
  --scores outputs/rag25_facet_local_minilm_v1/scoring_v1 \
  --output outputs/rag25_facet_local_minilm_v1/freeze_v1
.venv/bin/python -m trec_rag.facet_local_minilm_rank verify \
  --freeze outputs/rag25_facet_local_minilm_v1/freeze_v1
```

Expected: every stream and seven system rankings are hash-bound; the freeze declares `qrels_opened=false` and contains no qrels path.

- [ ] **Step 5: Test and commit**

```bash
.venv/bin/python -m pytest code/tests/test_facet_local_minilm_rank.py -q
git add code/trec_rag/facet_local_minilm_rank.py code/tests/test_facet_local_minilm_rank.py
git commit -m "Freeze facet-local MiniLM ranking matrix"
```

### Task 5: Build and freeze the blinded facet-relevance review

**Files:**
- Create: `code/trec_rag/facet_local_minilm_review.py`
- Create: `code/tests/test_facet_local_minilm_review.py`

**Interfaces:**
- Consumes: the verified ranking freeze.
- Produces: masked top-two C0/BF50 review pool, reviewer label files, adjudication, unmask map, and a review freeze bound to the ranking freeze.

- [ ] **Step 1: Write failing masking and adjudication tests**

```python
def test_review_packet_masks_system_rank_score_and_docid(freeze):
    packet, secret = build_review_packet(freeze)
    assert len(packet) <= 108
    assert all(set(row) == {"item_id", "topic_id", "facet_query", "passage"} for row in packet)
    assert all("arm" not in row and "docid" not in row for row in packet)
    assert secret

def test_disagreement_requires_third_label(packet, labels_a, labels_b):
    with pytest.raises(ValueError, match="unadjudicated disagreement"):
        freeze_review(packet, labels_a, labels_b, adjudication={})
```

Also test top-two pooling from every one of 27 facets and both systems, within-facet deduplication, the same frozen highest-scoring MiniLM passage for an item regardless of contributing arm, an unmask-map membership for every `(arm, facet, source rank, document ID, passage-selection provenance)` tuple, attribution to both arm denominators for shared items, stable hash-seeded shuffle, exact label vocabulary, two independent reviewers, third-reviewer adjudication, no qrels access, no system unmask before labels freeze, protected-topic rejection, and create-only artifacts. The rubric explicitly asks reviewers to judge the displayed passage, not unseen document content.

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest code/tests/test_facet_local_minilm_review.py -q`

Expected: import failure.

- [ ] **Step 3: Implement packet generation and validation**

```bash
.venv/bin/python -m trec_rag.facet_local_minilm_review create \
  --freeze outputs/rag25_facet_local_minilm_v1/freeze_v1 \
  --output outputs/rag25_facet_local_minilm_v1/review_v1
```

The reviewers receive only `packet.jsonl` and `rubric.md`. After both label files and any adjudication are complete, run:

```bash
.venv/bin/python -m trec_rag.facet_local_minilm_review freeze \
  --review outputs/rag25_facet_local_minilm_v1/review_v1 \
  --labels-a outputs/rag25_facet_local_minilm_v1/review_v1/labels_a.jsonl \
  --labels-b outputs/rag25_facet_local_minilm_v1/review_v1/labels_b.jsonl \
  --adjudication outputs/rag25_facet_local_minilm_v1/review_v1/adjudication.jsonl
```

Expected: an immutable review freeze and system-unmasked aggregate counts; no qrels have opened.

- [ ] **Step 4: Test and commit**

```bash
.venv/bin/python -m pytest code/tests/test_facet_local_minilm_review.py -q
git add code/trec_rag/facet_local_minilm_review.py code/tests/test_facet_local_minilm_review.py
git commit -m "Add blinded facet relevance review"
```

### Task 6: Evaluate headroom, gains, losses, and diagnostic outcomes

**Files:**
- Create: `code/trec_rag/facet_local_minilm_evaluate.py`
- Create: `code/tests/test_facet_local_minilm_evaluate.py`
- Generate, do not commit: `outputs/rag25_facet_local_minilm_v1/evaluation_v1/qrels_access_receipt.json`

**Interfaces:**
- Consumes: verified ranking freeze, verified blinded-review freeze, prior O/R1 rankings, and a separately authorized hash-pinned qrels projection whose sidecar declares exactly topics `200`, `225`, `707`, and `897`.
- Produces: union curves, facet retention, aggregate/per-topic metrics, gained/lost document sets, representative evidence, and one mechanical diagnostic outcome.

- [ ] **Step 1: Write failing firewall and metric tests**

```python
def test_qrels_cannot_open_before_both_freezes_verify(monkeypatch, incomplete_freeze):
    monkeypatch.setattr(module, "read_qrels", lambda path: pytest.fail("qrels opened"))
    with pytest.raises(ValueError, match="freeze is incomplete"):
        evaluate(
            incomplete_freeze,
            qrels_manifest="safe_projection/manifest.json",
            qrels_approval="approvals/qrels_access_v1.json",
        )

def test_gain_loss_accounting_reconciles():
    comparison = compare_docid_sets(baseline={"a", "b"}, candidate={"b", "c"})
    assert comparison.gained == {"c"}
    assert comparison.lost == {"a"}
    assert comparison.net_change == 0
```

Also test raw-union deduplication; paired `O@100 + C0 facets@K` and `O@100 + BF facets@K` curves for `K=20,50,100`; equality of the two raw candidate unions at K=100; separate raw-union, BF per-facet rank/contribution, and final fused top-100 document sets; overall and graded relevance definitions; O@100/R1@100 comparisons; gains plus losses; per-facet relevant retention; exact metric formulas; oracle rankings; judged rates; blinded-review macro aggregation and shared-item attribution; all six ordered diagnostic outcomes; no Stage A output; sidecar rejection before qrels data access when the declared topic set is not exact; projection-hash and freeze-hash binding; refusal of repeat qrels access; deterministic evaluation; and self-hashed outputs.

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest code/tests/test_facet_local_minilm_evaluate.py -q`

Expected: import failure.

- [ ] **Step 3: Implement the evaluator and outcome rules**

Make `BF100` the primary MiniLM arm. BF50/BF20 diagnose retention; BO100/BB100 isolate family effects; BF100_MAXP diagnoses aggregation. Implement the exact ordered decision table in the design, including positive macro Recall@100, non-worse macro graded Recall@100, and the `-0.02` per-topic recall guards for `B_promotes_coverage`. Persist three auditable stages: raw candidate union, BF100 per-facet ranks and RRF contributions before fusion, and final fused top 100. A fusion-block outcome requires at least one relevant document absent from R1 to move from C0 facet rank greater than 20 to BF facet rank at most 20 and then remain absent from final BF100@100. Produce `raw_union.json`, `prefusion.json`, `facet_retention.json`, `systems.json`, `gains_losses.json`, `review_metrics.json`, `representatives.json`, and `decision.json`, each self-hashed and bound to both freezes. The all-topic qrels path is not a supported CLI input.

- [ ] **Step 4: Open qrels once after all freeze hashes verify**

Require the user-authorized safe projection at
`outputs/rag25_facet_local_minilm_v1/authorized_inputs/pilot_qrels_projection_v1/`.
Its sidecar must bind the projected file hash and declare exactly the four
pilot topics. This plan provides no command that reads or derives it from the
all-topic qrels. If it is absent, stop and request explicit direction.

```bash
.venv/bin/python -m trec_rag.facet_local_minilm_evaluate \
  --freeze outputs/rag25_facet_local_minilm_v1/freeze_v1 \
  --review-freeze outputs/rag25_facet_local_minilm_v1/review_v1 \
  --prior-evaluation outputs/rag25_sparse_relevance_paired_v1/evaluation_v1/evaluation.json \
  --qrels-manifest outputs/rag25_facet_local_minilm_v1/authorized_inputs/pilot_qrels_projection_v1/manifest.json \
  --qrels-approval outputs/rag25_facet_local_minilm_v1/approvals/qrels_access_v1.json \
  --output outputs/rag25_facet_local_minilm_v1/evaluation_v1
```

Expected: atomically create `qrels_access_receipt.json`, open the exact
hash-pinned projection once, emit one mutually exclusive diagnostic outcome
with reproducible metric evidence, and perform no post-qrels ranking
construction. A repeat run must use saved evaluation outputs rather than reopen
the projection.

- [ ] **Step 5: Test and commit**

```bash
.venv/bin/python -m pytest code/tests/test_facet_local_minilm_evaluate.py -q
git add code/trec_rag/facet_local_minilm_evaluate.py code/tests/test_facet_local_minilm_evaluate.py
git commit -m "Evaluate facet-local MiniLM pilot"
```

### Task 7: Build, render, verify, and review the final report

**Files:**
- Create: `code/trec_rag/build_facet_local_minilm_report.py`
- Create: `code/tests/test_build_facet_local_minilm_report.py`
- Create: `reports/experiments/facet_local_minilm_pilot_v1/artifact.json`
- Create: `reports/experiments/facet_local_minilm_pilot_v1/report.html`
- Modify: `reports/experiments/facet_local_minilm_pilot_v1/README.md`

**Interfaces:**
- Consumes: authenticated manifest, preflight, scoring, ranking freeze, blinded-review freeze, and evaluation artifacts.
- Produces: a source-backed Data Analytics report artifact and self-contained rendered HTML.

- [ ] **Step 1: Write failing report-integrity tests**

```python
def test_report_leads_with_candidate_headroom_and_diagnostic_outcome(inputs):
    artifact = build_artifact(**inputs)
    text = "\n".join(block.get("body", "") for block in artifact["manifest"]["blocks"])
    assert "raw union" in text.lower()
    assert inputs["decision"]["outcome"] in text
    assert "nDCG@10 is a guardrail" in text

def test_report_cannot_hide_relevant_losses(inputs):
    artifact = build_artifact(**inputs)
    assert artifact["snapshot"]["datasets"]["gain_loss_rows"]
    assert all("gained" in row and "lost" in row and "net_change" in row for row in artifact["snapshot"]["datasets"]["gain_loss_rows"])
```

Also test source hashes, exact 31-stream accounting, model/download/runtime evidence, capped-window coverage diagnostics, one-time qrels-access receipt, blinded-review caveat and shared-item attribution, pilot-only language, candidate-absence versus reranking/fusion diagnosis, the raw/prefusion/final document sets, protected-topic exclusion, representative passage provenance, no raw qrels rows, accessible narrow tables, deterministic artifact output, and create-only HTML generation.

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest code/tests/test_build_facet_local_minilm_report.py -q`

Expected: import failure.

- [ ] **Step 3: Implement and generate the report source**

```bash
.venv/bin/python -m trec_rag.build_facet_local_minilm_report \
  --manifest reports/experiments/facet_local_minilm_pilot_v1/manifest.json \
  --preflight outputs/rag25_facet_local_minilm_v1/preflight_v1 \
  --scoring outputs/rag25_facet_local_minilm_v1/scoring_v1 \
  --freeze outputs/rag25_facet_local_minilm_v1/freeze_v1 \
  --review outputs/rag25_facet_local_minilm_v1/review_v1 \
  --evaluation outputs/rag25_facet_local_minilm_v1/evaluation_v1 \
  --output reports/experiments/facet_local_minilm_pilot_v1/artifact.json
```

- [ ] **Step 4: Render and verify desktop/mobile HTML**

```bash
node /home/npatta01/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs \
  --input reports/experiments/facet_local_minilm_pilot_v1/artifact.json \
  --output reports/experiments/facet_local_minilm_pilot_v1/report.html
```

Expected: validation, packaging, source interaction, 1440 px, and 390 px verification all pass; no publication or sharing.

- [ ] **Step 5: Run full verification and request advisor/code review**

```bash
.venv/bin/python -m pytest -q
git diff --check
```

Ask an IR advisor to verify the experimental interpretation and a code reviewer to verify the source bindings, qrels firewall, and report integrity. Resolve every Critical or Important finding before committing.

- [ ] **Step 6: Commit**

```bash
git add code/trec_rag/build_facet_local_minilm_report.py code/tests/test_build_facet_local_minilm_report.py reports/experiments/facet_local_minilm_pilot_v1/README.md reports/experiments/facet_local_minilm_pilot_v1/artifact.json reports/experiments/facet_local_minilm_pilot_v1/report.html
git commit -m "Report facet-local MiniLM pilot"
```

## Plan self-review checklist

- Exact stream accounting: 31 streams, 27 facets, four originals, 3,100 rows.
- Primary isolation: BF arms rerank facets only; original order is unchanged.
- Model identity: actual cross-encoder, exact revision, safe-file allowlist.
- Cost control: bounded windows/pairs, tokenizer-only preflight, benchmark and full-run approvals.
- Scientific separation: raw union, local reranking, fusion, and final ranking measured separately.
- Baseline identity: C0 bytes and canonical ranking hash match prior frozen R1.
- Relevance honesty: qrels for topic relevance; blinded review for facet relevance.
- No score-calibration leak: raw logits remain within query streams; RRF consumes ranks.
- No qrels leak: rankings and review labels freeze first; only the exact
  four-topic projection can open once under an atomic access receipt.
- Failure diagnosis: candidate absence, local reranker gap, fusion block, ranking-only gain, or coverage promotion.
- No Stage A implementation until BF establishes value.
