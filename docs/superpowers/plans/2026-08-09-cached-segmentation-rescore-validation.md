# Cached Segmentation Rescore Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-score all 22 cached RAG 2025 development topics with the fixed sentence segmenter on one fast dstack GPU, prove that planning/retrieval/passage scoring performed no new work, and compare old versus fixed canonical nuggets against the exact same frozen obligation plans.

**Architecture:** Add one explicit three-value retrieval execution policy and a read-only document-store seam so `cached-upstream-rescore` fails before forbidden upstream work while allowing sentence scoring, similarity, and canonicalization. Add an authenticated A/B validator that reports fragmentation metrics, reuses the 22 completed baseline coverage plans, and makes one candidate-arm judge call per topic. One dstack task restores the portable bundles, runs topic 407 first, then all topics with four workers on one GPU, and publishes a private verified result bundle.

**Tech Stack:** Python 3.12, pytest, spaCy `en_core_web_sm` 3.8.0, PyTorch CUDA, SQLite, dstack 0.20.29, private Hugging Face Buckets, deterministic tar+zstd, OpenRouter `openai/gpt-5.6-sol`.

## Global Constraints

- Work only in `/home/npatta01/data/competitions/trec_rag_2026/.worktrees/fix-sentence-segmentation` on `codex/fix-sentence-segmentation`.
- Topics: `14,31,37,58,72,84,144,161,200,213,219,224,225,233,273,300,407,477,499,515,707,897`.
- Source: private immutable `hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/experiments/nonagentic-rag25-dev-20260806`.
- Destination run ID: `nonagentic-rag25-segmentation-fixed-20260809`; never overwrite a remote prefix.
- Planning, retrieval, document materialization, and passage scoring are fail-closed cache-only/read-only.
- Sentence scoring and similarity may use local GPU work. Canonicalization may make bounded hosted calls.
- Reuse existing baseline `plan.json` bytes and make at most 22 candidate-arm judge calls; make no planner calls.
- Use one on-demand machine, one GPU with at least 48 GB VRAM, 32 GB RAM, and 100 GB disk.
- Topic 407 validates first. The full run uses `execution.topic_workers: 4`.
- Keep caches, outputs, evidence, nuggets, provider responses, and reports private and out of git.
- Ignore Modal completely: no Modal dependency, config key, import, test, or execution path.
- Promotion means push and open a draft PR after validation; never merge automatically.

---

### Task 1: Fail-Closed Document Materialization

**Files:**

- Modify: `code/trec_rag/document_store.py`
- Modify: `code/trec_rag/evidence_store.py`
- Test: `code/tests/test_document_store.py`
- Test: `code/tests/test_evidence_pipeline_contract.py`

**Interfaces:**

- Produces `ReadOnlyDocumentStore(DocumentStore)`, whose `admit_text()` verifies an existing exact object without creating directories, temporary files, or links.
- Adds optional `document_store: DocumentStore | None` to `generate_candidate_artifacts`; default behavior is unchanged.

- [ ] **Step 1: Write failing document-store tests**

```python
def test_read_only_store_admits_only_existing_exact_object(tmp_path: Path) -> None:
    writable = DocumentStore(tmp_path)
    receipt = writable.admit_text("cached document")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    actual = ReadOnlyDocumentStore(tmp_path).admit_text(
        "cached document", expected_sha256=receipt.content_sha256
    )
    assert actual == receipt
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_read_only_store_missing_object_creates_nothing(tmp_path: Path) -> None:
    root = tmp_path / "missing-store"
    with pytest.raises(DocumentStoreIntegrityError, match="unable to read"):
        ReadOnlyDocumentStore(root).admit_text("not cached")
    assert not root.exists()
```

These catch calling the writable admission path or creating a digest directory before verification.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest code/tests/test_document_store.py -k read_only -q
```

Expected: import failure because `ReadOnlyDocumentStore` is absent.

- [ ] **Step 3: Implement the minimal subclass and verify GREEN**

Validate UTF-8 and the optional digest, compute the literal `DocumentReceipt`, call `verify`, compare stored text byte-for-byte, and return the verified receipt. Re-run Step 2.

- [ ] **Step 4: Write the failing candidate-generation injection test**

Use the real evidence fixture. A read-only store containing every source must produce candidate artifacts; deleting one source must raise before any replacement file appears. Assert artifacts and filesystem state, not mock calls.

- [ ] **Step 5: Verify RED, add the seam, and verify GREEN**

```bash
.venv/bin/python -m pytest code/tests/test_evidence_pipeline_contract.py -k read_only_document_store -q
```

Expected RED: unexpected `document_store` argument. Add this parameter without changing defaults:

```python
document_store: DocumentStore | None = None
```

Require an injected value to be a `DocumentStore` and use it for audit, request iteration, and `TopicRecordsBuilder`. Run both Task 1 test files.

- [ ] **Step 6: Commit Task 1**

```bash
git add code/trec_rag/document_store.py code/trec_rag/evidence_store.py \
  code/tests/test_document_store.py code/tests/test_evidence_pipeline_contract.py
git commit -m "Add fail-closed cached document access"
```

---

### Task 2: Explicit Cached-Upstream Rescore Policy

**Files:**

- Modify: `code/trec_rag/topic_dispatch.py`
- Modify: `code/trec_rag/competition_retrieval.py`
- Test: `code/tests/test_topic_dispatch.py`
- Test: `code/tests/test_competition_retrieval_offline_cache.py`
- Test: `code/tests/test_competition_topic_dispatch.py`
- Test: `code/tests/test_competition_retrieval_v2.py`
- Modify compatibility fake: `code/tests/test_facet_pipeline_e2e.py`

**Interfaces:**

- Produces `ExecutionPolicy = Literal["online", "offline-cache-only", "cached-upstream-rescore"]` and canonical `TopicJob.execution_policy`.
- Retains `TopicJob.offline_cache_only` as a derived compatibility property.
- Adds `cached_upstream_rescore: bool = False` to `run_official` and mutually exclusive CLI `--cached-upstream-rescore`.

- [ ] **Step 1: Write failing three-mode dispatch tests**

Use this matrix as literal expected data, publish one real receipt per policy, and verify its filename/body. Also verify cross-mode receipts and unknown policies are rejected:

```python
POLICY_RECEIPTS = (
    ("online", "topic-job-receipt.json"),
    ("offline-cache-only", "topic-job-receipt.offline-cache-only.json"),
    ("cached-upstream-rescore", "topic-job-receipt.cached-upstream-rescore.json"),
)
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest code/tests/test_topic_dispatch.py -k 'policy or receipt' -q
```

Expected: the job still serializes only a Boolean.

- [ ] **Step 3: Implement policy serialization and verify GREEN**

Replace the job field with `execution_policy: str = "online"`, validate three values, derive the compatibility property, advance the receipt schema, use three receipt filenames, and serialize the exact policy. Update internal constructors and run the full dispatch test file.

- [ ] **Step 4: Write failing dependency-policy tests**

For cached rescore, assert the literal flags below and a missing-document failure before sentence scoring or canonicalization:

```text
retriever.cache_only = true
passage scorer.read_only = true
sentence scorer.read_only = false
similarity.cache_only = false
environment loading = true
offline staging = false
```

- [ ] **Step 5: Verify RED**

```bash
.venv/bin/python -m pytest \
  code/tests/test_competition_retrieval_offline_cache.py \
  code/tests/test_competition_topic_dispatch.py \
  -k cached_upstream_rescore -q
```

- [ ] **Step 6: Implement policy plumbing**

Use exactly these predicates:

```python
def _upstream_cache_only(policy: str) -> bool:
    return policy in {"offline-cache-only", "cached-upstream-rescore"}


def _downstream_cache_only(policy: str) -> bool:
    return policy == "offline-cache-only"
```

In cached mode, planning/retrieval/passage scorer are cache-only/read-only; passage search and candidate generation receive `ReadOnlyDocumentStore`; sentence scorer and similarity remain writable; canonicalization stays cache-backed online. Never use `_run_offline_topic_staged` for this mode.

- [ ] **Step 7: Add CLI mutual exclusion and resume isolation**

Reject both cache flags together. Resume validation accepts only a cached-mode receipt and never consumes online/offline receipts.

- [ ] **Step 8: Write failing operation-receipt gates**

Allow downstream sentence/similarity/canonicalization work. For each of `planning`, `retrieval`, and `passage_scores`, independently set `cache_misses`, `network_calls`, `provider_calls`, or `model_batches` to `1` and require cached-mode rejection.

- [ ] **Step 9: Verify RED, implement the shared gate, and verify GREEN**

```bash
.venv/bin/python -m pytest code/tests/test_competition_retrieval_v2.py -k cached_upstream -q
```

Use one validator for topic publish/read and aggregate-manifest publication. Preserve offline mode's all-stage zero-work rule. Then run all Task 2 tests.

- [ ] **Step 10: Commit Task 2**

```bash
git add code/trec_rag/topic_dispatch.py code/trec_rag/competition_retrieval.py \
  code/tests/test_topic_dispatch.py code/tests/test_competition_retrieval_offline_cache.py \
  code/tests/test_competition_topic_dispatch.py code/tests/test_competition_retrieval_v2.py \
  code/tests/test_facet_pipeline_e2e.py
git commit -m "Add cached-upstream rescore mode"
```

---

### Task 3: Authenticated Structural Before/After Metrics

**Files:**

- Create: `code/trec_rag/cached_segmentation_validation.py`
- Create: `code/tests/test_cached_segmentation_validation.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**

- Produces `load_structural_topic(output_root: Path, document_store_root: Path, topic_id: str) -> StructuralTopicMetrics`.
- Produces `compare_structural_runs(*, baseline_output_root: Path, candidate_output_root: Path, document_store_root: Path, baseline_handoff_path: Path, candidate_handoff_path: Path, topic_ids: Sequence[str]) -> StructuralComparison`.
- CLI `structural` writes canonical `structural-comparison.json` and manifest-last `structural-comparison-manifest.json`.

- [ ] **Step 1: Write failing tests with real typed artifacts**

Build small sealed TopicRecords/selection/handoff fixtures with literal candidate texts:

```text
old: Housing Costs | Rents soared. | from Dubai and
fixed: Housing Costs + Rents soared because demand increased. | Prices rose 17%.
```

Assert exact source-unit and candidate counts, median lengths, sub-40 fractions, fragment-proxy counts, first-position fragment status, canonical nugget lengths, and source containment. Source-unit baselines are nonblank `splitlines()` over the authenticated selected-document text; fixed units come from the production sentence/paragraph segmenter over those exact same bytes. These catch reading unauthenticated manifest totals or comparing different document populations.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest code/tests/test_cached_segmentation_validation.py -k structural -q
```

Expected: module import failure.

- [ ] **Step 3: Implement authenticated loaders and metrics**

Open each sealed `TopicRecords` database with its records manifest and shared `DocumentStore`, call `load_candidates()`, validate selections with `load_validated_selection_artifacts`, and load both authenticated generation handoffs. Require identical topic/narrative, decomposition, retrieval, passage, and selected-document source-seal identities before comparison. Candidate IDs may differ.

Use deterministic definitions:

```python
short = len(text.strip()) < 40
fragment = short or text.rstrip()[-1:] not in ".?!"
```

Also report old-line versus fixed-segmentation units over identical selected documents, candidate kinds/splitter identity, selection budgets, candidate/exact-group/selected-cluster counts, canonical state/fallback counts, and evidence-source-group coverage.

- [ ] **Step 4: Add quality-gate tests**

Topic 407 fixed segmentation units must improve over old-line median `11` and sub-40 rate `0.604`. Aggregate candidate sub-40 and selected-fragment rates must not worsen. Reject an upstream identity mismatch, selected-document mismatch, invalid provenance, incomplete selection, or changed narrative.

- [ ] **Step 5: Verify GREEN, document, and commit**

```bash
.venv/bin/python -m pytest code/tests/test_cached_segmentation_validation.py -k structural -q
git add code/trec_rag/cached_segmentation_validation.py \
  code/tests/test_cached_segmentation_validation.py code/trec_rag/README.md
git commit -m "Add segmentation before-after validator"
```

---

### Task 4: Frozen-Plan Paired Nugget Judging

**Files:**

- Modify: `code/trec_rag/retrieval_nugget_coverage.py`
- Modify: `code/trec_rag/cached_segmentation_validation.py`
- Test: `code/tests/test_retrieval_nugget_coverage.py`
- Test: `code/tests/test_cached_segmentation_validation.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**

- Produces `seed_coverage_plan_from_completed_baseline(*, baseline_handoff_manifest_path: Path, baseline_work_dir: Path, candidate_handoff_manifest_path: Path, candidate_work_dir: Path, topic_id: str) -> None`, which validates a complete baseline, requires identical candidate narrative bytes/hash, writes candidate `input.json`, and create-only copies the validated baseline `plan.json`.
- CLI `semantic` loops exact topics, resumes candidate judging with baseline model/prompt identities, and publishes a comparison manifest last.

- [ ] **Step 1: Write the failing plan-reuse test**

Create authenticated old/new handoffs with the same narrative and different claims. Build a complete old coverage bundle with injected fake backends. Seed new state, resume with one judge fake, and assert:

```text
planner calls = 0
candidate judge calls = 1
candidate plan bytes = baseline plan bytes
candidate plan_sha256 = baseline plan_sha256
```

Also reject narrative/model/prompt drift.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_nugget_coverage.py -k seed_completed_baseline -q
```

- [ ] **Step 3: Implement create-only seeding and verify GREEN**

Use `load_completed_coverage_evaluation` and `coverage_input_from_handoff`, compare narrative bytes/hash, validate baseline identity, publish candidate input, and copy only validated `plan.json`. Never copy old judgments/report/manifest.

- [ ] **Step 4: Write failing paired comparison tests**

Use a frozen plan where old is `partial` and fixed is `full`. Assert required-coverage and strict-full deltas, newly covered/regressed obligation IDs, identical plan hashes, hosted-call count, and artifact hashes. Reject missing complete old state, topic mismatch, and narrative mismatch.

- [ ] **Step 5: Verify RED, implement orchestration, and verify GREEN**

```bash
.venv/bin/python -m pytest code/tests/test_cached_segmentation_validation.py -k semantic -q
```

For each topic, validate `retrieval_nugget_coverage_v2/<topic>`, seed fixed state, call `run_coverage_evaluation(mode="resume", allow_hosted_calls=True)`, and reload the completed fixed state. Emit old/new label vectors and macro scores. Gate only when macro required coverage and strict-full rate do not regress; list every per-obligation regression for review.

- [ ] **Step 6: Run both suites and commit**

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_nugget_coverage.py \
  code/tests/test_cached_segmentation_validation.py -q
git add code/trec_rag/retrieval_nugget_coverage.py \
  code/trec_rag/cached_segmentation_validation.py \
  code/tests/test_retrieval_nugget_coverage.py \
  code/tests/test_cached_segmentation_validation.py code/trec_rag/README.md
git commit -m "Compare nuggets against frozen coverage plans"
```

---

### Task 5: Strict Private Result Bundle

**Files:**

- Create: `code/trec_rag/cached_segmentation_result_bundle.py`
- Create: `code/tests/test_cached_segmentation_result_bundle.py`

**Interfaces:**

- CLI `pack-baseline --handoff PATH --coverage-root PATH --destination DIR`.
- CLI `verify-baseline DIR`.
- CLI `pack --run-root PATH --validation-root PATH --destination DIR`.
- CLI `verify DIR`.
- Each command produces exactly `bundle.tar.zst` and `bundle-complete.json`. Baseline markers bind the handoff plus 22 complete coverage directories. Result markers bind archive/member hashes, run ID, Git revision, topics, retrieval export, operation manifest, and both comparisons.

- [ ] **Step 1: Write hostile archive and round-trip tests**

Cover valid baseline and two-topic result fixtures. Reject traversal, absolute paths, symlinks, devices, duplicate/colliding paths, extras, missing manifests, tampered bytes, wrong topic order, changed run/Git identity, and trailing archive bytes. Baseline verification must call `load_completed_coverage_evaluation` for every topic after safe extraction.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest code/tests/test_cached_segmentation_result_bundle.py -q
```

- [ ] **Step 3: Implement deterministic pack/verify**

Reuse safe tar+zstd techniques from `competition_cache_bundle.py` under separate baseline/result schemas and member allowlists. The baseline includes only the authenticated handoff and 22 five-file coverage bundles. The result includes authenticated exports, topic checkpoint/canonical artifacts, operation receipts, and comparisons. Exclude caches, raw provider responses, locks/WAL/SHM, qrels, gold nuggets, model weights, and RAG answers.

- [ ] **Step 4: Verify GREEN and commit**

```bash
.venv/bin/python -m pytest code/tests/test_cached_segmentation_result_bundle.py -q
git add code/trec_rag/cached_segmentation_result_bundle.py \
  code/tests/test_cached_segmentation_result_bundle.py
git commit -m "Bundle cached segmentation validation results"
```

---

### Task 6: One-Machine dstack Workflow

**Files:**

- Create: `code/tools/run_cached_segmentation_validation.sh`
- Create: `code/tools/apply_cached_segmentation_validation.sh`
- Create: `.dstack/rag25-cached-segmentation-validation.yaml`
- Create: `code/tests/test_cached_segmentation_validation_workflow.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**

```text
run_cached_segmentation_validation.sh [--preflight]
  --run-id nonagentic-rag25-segmentation-fixed-20260809
  --source-run-id nonagentic-rag25-dev-20260806
  --baseline-uri hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/artifacts/rag25-segmentation-baseline-20260807
  --config configs/rag25_competition_retrieval_v1.yaml
```

```text
apply_cached_segmentation_validation.sh --preview|--launch --name NAME -- WRAPPER_ARGS
```

- [ ] **Step 1: Write executable workflow tests first**

Run the real wrapper against fake `hf`, `uv`, `git`, `nvidia-smi`, and nested Python. Assert: exact 22 source pairs; empty private destination; verify-all-then-merge; topic 407 cached-rescore canary; two-topic warm probe; final four-worker all-topic run; structural gate before promotion; no planner and at most 22 judge calls; archive-before-marker upload; list/download/byte-compare/reverify; failure propagation; committed-only launch; preview declines and never submits.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest code/tests/test_cached_segmentation_validation_workflow.py -q
```

- [ ] **Step 3: Implement preflight and remote setup**

Follow the existing cache-shard launcher for committed-only transport, secret-safe mapping, pinned interpreter, locked CUDA sync, private-bucket checks, foreground failures, and marker-last round trips. Download 1.30 GiB, verify each `competition_cache_bundle`, then merge all 22 once into fresh `/tmp` cache/output roots. Download the baseline archive/marker pair, byte-verify it, and run `cached_segmentation_result_bundle verify-baseline` before any hosted judge call. Never copy SQLite directly.

- [ ] **Step 4: Implement canary, warm probe, full run, and validation**

Generate ignored configs from tracked `configs/rag25_competition_retrieval_v1.yaml`:

```text
canary 407: one worker
warm probe 14,31: two workers
final 22: four workers
```

All share the restored downstream cache, so canary/probe sentence, similarity, and canonical results become final-run hits. Run topic 407 structural validation before continuing. After full export, run all structural and frozen-plan semantic checks, then pack/upload/round-trip-verify the result.

- [ ] **Step 5: Add task configuration**

Use the pinned `huggingface/trl` image digest and launcher sentinel. Map secret names only. Configure one on-demand performance GPU pool verified against dstack 0.20.29, at least 48 GB VRAM, 32 GB RAM, 100 GB disk, `max_duration: 5h`, hard price cap, RunPod/Vast.ai, and bounded `no-capacity` retry. Exclude A40 unless faster offers are unavailable and separately approved.

- [ ] **Step 6: Verify GREEN and commit**

```bash
.venv/bin/python -m pytest code/tests/test_cached_segmentation_validation_workflow.py -q
bash code/tools/run_cached_segmentation_validation.sh --preflight \
  --run-id nonagentic-rag25-segmentation-fixed-20260809 \
  --source-run-id nonagentic-rag25-dev-20260806 \
  --baseline-uri hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/artifacts/rag25-segmentation-baseline-20260807 \
  --config configs/rag25_competition_retrieval_v1.yaml
git add code/tools/run_cached_segmentation_validation.sh \
  code/tools/apply_cached_segmentation_validation.sh \
  .dstack/rag25-cached-segmentation-validation.yaml \
  code/tests/test_cached_segmentation_validation_workflow.py code/trec_rag/README.md
git commit -m "Run cached segmentation validation on dstack"
```

---

### Task 7: Verification, Review, and Live Gates

**Files:**

- Update: this plan with verification/live evidence
- Private baseline source: `outputs/nonagentic-rag25-dev-all22-replay-p4-20260807`
- Private baseline prefix: `hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/artifacts/rag25-segmentation-baseline-20260807`
- Private result prefix: `hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/experiments/nonagentic-rag25-segmentation-fixed-20260809`

**Interfaces:**

- Produces passing local verification, independent Sol review, unchanged dstack offer preview, one monitored detached task, downloaded/reverified private result, and—only on success—a pushed branch plus draft PR.

- [ ] **Step 1: Run local verification from a clean commit**

```bash
uv lock --check
.venv/bin/python -m pytest \
  code/tests/test_document_store.py \
  code/tests/test_evidence_pipeline_contract.py \
  code/tests/test_topic_dispatch.py \
  code/tests/test_competition_retrieval_offline_cache.py \
  code/tests/test_competition_topic_dispatch.py \
  code/tests/test_competition_retrieval_v2.py \
  code/tests/test_retrieval_nugget_coverage.py \
  code/tests/test_cached_segmentation_validation.py \
  code/tests/test_cached_segmentation_result_bundle.py \
  code/tests/test_cached_segmentation_validation_workflow.py -q
git diff --check origin/master...HEAD
git status --short
```

- [ ] **Step 2: Obtain independent substantive review**

Ask `sol_reviewer` to review `origin/master...HEAD` for cache-safety escapes, document writes, receipt/resume confusion, archive hostility, secrets, concurrency/SQLite hazards, and metric validity. Fix actionable findings test-first and repeat Step 1.

- [ ] **Step 3: Upload the minimal private baseline input**

Package only the old `generation_handoff_manifest.json` and each exact topic's completed `retrieval_nugget_coverage_v2/{input,plan,judgments,report,manifest}.json`. Require an empty private prefix, archive-before-marker upload, remote list/download/byte comparison, and semantic revalidation. Record the archive SHA-256. Exclude passages, qrels, gold nuggets, and provider responses.

- [ ] **Step 4: Run mandatory dstack/HF preflight**

```bash
/home/npatta01/.agents/skills/running-dstack-hf-experiments/scripts/preflight.sh
```

Retain the HF mode and verify the real wrapper's `--preflight` path.

- [ ] **Step 5: Preview without submitting**

```bash
bash code/tools/apply_cached_segmentation_validation.sh \
  --preview --name rag25-segfix-all22-20260809 -- \
  --run-id nonagentic-rag25-segmentation-fixed-20260809 \
  --source-run-id nonagentic-rag25-dev-20260806 \
  --baseline-uri hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/artifacts/rag25-segmentation-baseline-20260807 \
  --config configs/rag25_competition_retrieval_v1.yaml
```

Show the dstack table unchanged. Report GPU/provider/region identities, hourly prices, five-hour maximum exposure, and at least two distinct eligible offers when practical. Stop for user offer approval.

- [ ] **Step 6: Submit exactly once after offer approval**

Run the same launcher with `--launch`, which must call `dstack apply -y -d` once. Check `dstack ps -v`, then monitor logs and GPU memory/utilization without a blocking attach.

- [ ] **Step 7: Enforce canary and full gates**

Before continuing after topic 407, require every non-hit upstream counter to be zero, fixed segmentation-unit median above 11, fixed segmentation-unit sub-40 fraction below 0.604, and all provenance/selection/canonical validators to pass. After all 22, require the same zero-work proof per topic, four-worker completion, authenticated export/handoff, improved candidate/selection fragmentation, and completed semantic comparison.

- [ ] **Step 8: Independently verify downloaded results**

List and download the exact result pair, compare SHA-256 values, and run local bundle verification. Report old/new topic-macro coverage, strict-full rates, label improvements/regressions, fragmentation deltas, elapsed time, peak VRAM, workers, dstack cost, canonicalization calls, and judge calls.

- [ ] **Step 9: Promote only when better or equal**

Require clean branch, `gh auth status`, and reviewed diff. Push `codex/fix-sentence-segmentation` and open a draft PR summarizing the bug, metrics, zero-retrieval proof, nugget comparison, private artifact hashes, and limitations. Do not merge.
