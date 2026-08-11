# Accepted RAG RAGDoll Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate both exact Evalbase-accepted 119-topic RAG JSONLs with the repository's authoritative RAGDoll citation-support workflow and produce two validated, privacy-scanned, tailnet-only reports.

**Architecture:** Add an accepted-artifact input mode beside the existing completed-run config mode. A deterministic post-run binding authenticates the accepted JSONL, bundle metadata, sealed handoff, and per-topic contexts without pretending the missing single-pass runtime receipt was preserved; the evaluator then uses the existing support-task, cache, metric, and friendly-renderer path. Hosted judging remains off by default, starts with one probe, resumes through an identity-stable bounded worker pool, and publishes only after cache-only replay has zero missing, failed, or conflicting judgments.

**Tech Stack:** Python 3.13, pytest/unittest, RAGDoll submodule, repository `JudgeCache`, JSON/JSONL, OpenAI Codex `pi` judge, standalone HTML renderer.

## Global Constraints

- Treat both accepted JSONLs and all three accepted Retrieval TSVs as immutable; their recorded SHA-256 values must not change.
- Evaluate exactly 3,155 citation tasks for `rag26-ss1` and 7,008 for `rag26-ms1-final`, across the exact 119 official topics in handoff order.
- Judge settings are provider `pi`, model `openai-codex/gpt-5.5`, thinking `medium`, temperature `null`, agent binary `pi`, and extension identity `none`.
- No hosted calls occur without `--run-judge`; the first hosted invocation for each run is `--judge-limit 1 --judge-workers 1`.
- Missing qrels, gold nuggets, and nugget assignments stay explicitly unavailable; generated claims are never treated as gold.
- Private work directories are mode `0700`; report files are mode `0600`; raw passages, tasks, events, judgments, assignments, and manifests never enter git or the rendered portal.
- Only HTML written by `trec_rag.friendly_report.write_report` after its dynamic privacy scan may be copied to the existing tailnet-only portal.
- Preserve the missing original single-pass generation receipt as explicitly unavailable provenance; do not label a post-run binding as the historical generation identity.
- Keep the five accepted organizer files byte-identical and leave retrieval, reranking, and answer generation closed.

---

### Task 1: Normalize accepted-run evidence provenance

**Files:**
- Create: `code/trec_rag/accepted_rag_evaluation.py`
- Modify: `code/trec_rag/ragdoll_io.py`
- Modify: `code/tests/test_ragdoll_io.py`
- Create: `code/tests/test_accepted_rag_evaluation.py`

**Interfaces:**
- Produces: `AcceptedRunBinding` with exact artifact, bundle metadata, handoff, run, model, topic-context, and optional source-identity receipts.
- Produces: `build_accepted_run_binding(submission_path: Path, bundle_metadata_path: Path, handoff_manifest_path: Path, *, source_identity_path: Path | None = None) -> AcceptedRunBinding`.
- Produces: `write_accepted_run_binding(binding: AcceptedRunBinding, output_path: Path) -> Path` using canonical sorted JSON and mode `0600`.
- Produces: `load_evidence_binding(path: Path) -> EvidenceBinding` in `ragdoll_io.py`, normalizing existing single-pass, preserved multi-stage, and accepted-run binding schemas.
- Consumes later: `selected_evidence_support_rows(submission_path: Path, handoff_manifest_path: Path, evidence_binding_path: Path) -> list[dict[str, object]]` retains narrative, topic-context, and allowed-document checks for all normalized schemas.

- [ ] **Step 1: Write synthetic binding failures and success cases**

```python
def test_accepted_binding_records_missing_source_identity(tmp_path: Path) -> None:
    paths = build_accepted_fixture(tmp_path, run_id="accepted-single")
    binding = build_accepted_run_binding(paths.submission, paths.bundle_metadata, paths.handoff)
    assert binding.schema_version == "accepted_rag_evaluation_binding_v1"
    assert binding.run_id == "accepted-single"
    assert binding.source_identity_available is False
    assert binding.source_identity_reason == "original generation identity was not preserved"
    assert binding.topic_context_sha256s == paths.topic_context_sha256s

@pytest.mark.parametrize("mutation", ["submission_hash", "run_id", "handoff_hash"])
def test_accepted_binding_rejects_receipt_mismatch(tmp_path: Path, mutation: str) -> None:
    paths = build_accepted_fixture(tmp_path, mutation=mutation)
    with pytest.raises(ValueError, match="accepted.*does not match"):
        build_accepted_run_binding(paths.submission, paths.bundle_metadata, paths.handoff)
```

Add a preserved multi-stage source identity fixture with `submission_run_id` and `topics[*].context_sha256`; assert a mismatch is rejected rather than downgraded to unavailable.

- [ ] **Step 2: Run new tests and verify the interface is absent**

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_accepted_rag_evaluation.py code/tests/test_ragdoll_io.py -q
```

Expected: import/collection failure for `trec_rag.accepted_rag_evaluation` and the new loader.

- [ ] **Step 3: Implement canonical accepted-artifact receipts**

```python
@dataclass(frozen=True)
class AcceptedRunBinding:
    schema_version: str
    run_id: str
    run_desc: str
    team_id: str
    provider: str
    models: tuple[str, ...]
    submission_sha256: str
    bundle_metadata_sha256: str
    handoff_schema_version: str
    handoff_manifest_sha256: str
    topic_context_sha256s: Mapping[str, str]
    source_identity_available: bool
    source_identity_sha256: str | None
    source_identity_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_desc": self.run_desc,
            "team_id": self.team_id,
            "provider": self.provider,
            "models": list(self.models),
            "submission_sha256": self.submission_sha256,
            "bundle_metadata_sha256": self.bundle_metadata_sha256,
            "handoff_schema_version": self.handoff_schema_version,
            "handoff_manifest_sha256": self.handoff_manifest_sha256,
            "topic_context_sha256s": dict(sorted(self.topic_context_sha256s.items())),
            "source_identity_available": self.source_identity_available,
            "source_identity_sha256": self.source_identity_sha256,
            "source_identity_reason": self.source_identity_reason,
        }
```

Match the metadata run entry to the submission suffix and embedded `metadata.run_id`; require byte count, line count, SHA-256, source handoff hash, and topic count to agree. Reject duplicate/reordered topic IDs. Validate an optional multi-stage receipt's submission run ID, handoff hash, and complete topic-context map. Write atomically with parent mode `0700` and file mode `0600`.

- [ ] **Step 4: Normalize schemas without weakening checks**

```python
@dataclass(frozen=True)
class EvidenceBinding:
    kind: str
    run_id: str
    handoff_schema_version: str
    handoff_manifest_sha256: str
    topic_context_sha256s: Mapping[str, str]
    prompt_contract_version: str | None
    source_identity_available: bool
    source_identity_reason: str | None
```

Keep `prompt_contract_version == PROMPT_CONTRACT_VERSION` for historical single-pass identities. Validate schema-specific multi-stage/accepted contract fields without synthesizing a single-pass prompt contract. Keep narrative equality, context equality, and foreign-reference rejection in `selected_evidence_support_rows`.

- [ ] **Step 5: Run tests and commit**

Run the Step 2 command; expected all pass. Then:

```bash
git add code/trec_rag/accepted_rag_evaluation.py code/trec_rag/ragdoll_io.py \
  code/tests/test_accepted_rag_evaluation.py code/tests/test_ragdoll_io.py
git commit -m "feat: bind accepted RAG artifacts for evaluation"
```

---

### Task 2: Add accepted artifacts to the authoritative evaluation CLI

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py`
- Modify: `code/trec_rag/offline_evaluation.py`
- Modify: `code/trec_rag/competition_evaluation_report.py`
- Modify: `code/tests/offline_evaluation_fixture.py`
- Modify: `code/tests/test_offline_evaluation.py`

**Interfaces:**
- Consumes: Task 1 accepted binding functions.
- Produces: `RagArtifactSource` with handoff/output paths, ordered topics, team/run metadata, provider, and display model.
- Produces: optional `rag_artifact_source: RagArtifactSource | None` on `load_debug_report_data` and `build_evaluation_bundle`, preserving current `rag_config_path` mode.
- Produces CLI accepted mode: `--accepted-rag`, `--accepted-bundle-metadata`, `--handoff-manifest`, optional `--source-identity`; mutually exclusive with `--rag-config`.

- [ ] **Step 1: Pin accepted CLI behavior with failing tests**

```python
def test_cli_requires_config_mode_or_complete_accepted_mode(self) -> None:
    with self.assertRaises(SystemExit):
        cli.main(["--retrieval-config", "r.yaml", "--accepted-rag", "run.jsonl"])

def test_accepted_mode_records_post_run_binding(self) -> None:
    fixture = self.accepted_fixture(ONE_TOPIC)
    bundle = build_evaluation_bundle(
        retrieval_config_path=fixture.retrieval_config,
        accepted_submission_path=fixture.rag_output,
        accepted_bundle_metadata_path=fixture.bundle_metadata,
        handoff_manifest_path=fixture.handoff,
        work_dir=fixture.root / "work",
        repository_root=REPOSITORY_ROOT,
        cache_root=fixture.root / "cache",
        judge=None,
        judge_settings=settings(),
    )
    assert bundle.manifest["identities"]["binding_kind"] == "accepted_rag_evaluation_binding_v1"
    assert bundle.manifest["identities"]["source_identity_available"] is False
```

Also reject bytes changed after metadata, wrong team/run description, and a handoff incompatible with the retrieval export.

- [ ] **Step 2: Run accepted-mode tests and verify failure**

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py -k 'accepted or cli_requires_config_mode' -q
```

Expected: failures for missing accepted arguments and `RagArtifactSource`.

- [ ] **Step 3: Separate artifact loading from generation config loading**

```python
@dataclass(frozen=True)
class RagArtifactSource:
    handoff_manifest_path: Path
    output_path: Path
    topic_ids: tuple[str, ...]
    team_id: str
    run_id: str
    run_desc: str
    provider: str
    model: str
```

Refactor `_attach_rag_outputs` to accept this object. Config mode constructs it from `RagGenerationConfig`; accepted mode constructs it from validated binding/handoff. Preserve safe-file snapshots, exact row order, `validate_submission_record`, and retrieval-handoff compatibility.

- [ ] **Step 4: Thread accepted paths through bundle construction**

```python
def build_evaluation_bundle(
    *,
    retrieval_config_path: Path,
    work_dir: Path,
    repository_root: Path,
    cache_root: Path,
    judge_settings: JudgeSettings,
    rag_config_path: Path | None = None,
    accepted_submission_path: Path | None = None,
    accepted_bundle_metadata_path: Path | None = None,
    handoff_manifest_path: Path | None = None,
    source_identity_path: Path | None = None,
    topic_ids: Sequence[str] | None = None,
    qrels_path: Path | None = None,
    gold_nuggets_path: Path | None = None,
    judge: JudgeCallable | None = None,
    judge_limit: int | None = None,
    judge_workers: int = 1,
    created_utc: str | None = None,
) -> EvaluationBundle:
```

Require exactly one input mode. Accepted mode writes `accepted_run_binding.json` inside private work, passes it to support-row construction, and records provenance availability. Config mode retains its existing identity behavior and deterministic rendering.

- [ ] **Step 5: Add CLI validation and portable receipts**

Use an argparse mutually exclusive group for `--rag-config` versus `--accepted-rag`; post-parse require metadata/handoff in accepted mode and permit source identity only there. Portable HTML commands omit absolute private paths; stdout retains the exact invocation.

- [ ] **Step 6: Run regressions and commit**

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py code/tests/test_ragdoll_io.py \
  code/tests/test_accepted_rag_evaluation.py -q
git add code/trec_rag/competition_debug_report.py code/trec_rag/offline_evaluation.py \
  code/trec_rag/competition_evaluation_report.py code/tests/offline_evaluation_fixture.py \
  code/tests/test_offline_evaluation.py
git commit -m "feat: evaluate accepted RAG submissions"
```

Expected: all pass, including byte-identical existing fixture reports.

---

### Task 3: Make hosted judging bounded and resumable

**Files:**
- Modify: `code/trec_rag/offline_evaluation.py`
- Modify: `code/trec_rag/competition_evaluation_report.py`
- Modify: `code/tests/test_offline_evaluation.py`

**Interfaces:**
- Produces: `_resolve_judgments(tasks: Sequence[Mapping[str, Any]], *, cache: JudgeCache, judge: JudgeCallable | None, settings: JudgeSettings, ragdoll: Mapping[str, str], judge_limit: int | None = None, judge_workers: int = 1) -> tuple[list[dict[str, Any]], dict[str, int]]` with deterministic ordering, preselected judge limits, controller-owned cache writes, and bounded hosted calls.
- Produces CLI `--judge-workers N`, positive integer, default `1`, effective only with `--run-judge`.

- [ ] **Step 1: Write concurrency, limit, ordering, and resume tests**

```python
def test_judge_workers_bound_concurrency_and_preserve_order(self) -> None:
    fixture = self.run_fixture(THREE_TOPICS)
    judge = ConcurrencyRecordingJudge(label="FS")
    bundle = self.build(fixture, judge=judge, judge_workers=2)
    self.assertEqual(judge.max_active, 2)
    rows = _rows(bundle.work_dir / "support_judgments.jsonl")
    self.assertEqual([row["task_id"] for row in rows], fixture.expected_task_ids)

def test_probe_limit_schedules_one_miss_with_four_workers(self) -> None:
    fixture = self.run_fixture(THREE_TOPICS)
    judge = ConcurrencyRecordingJudge(label="PS")
    bundle = self.build(fixture, judge=judge, judge_limit=1, judge_workers=4)
    self.assertEqual(bundle.manifest["judge"]["hosted_calls"], 1)
    self.assertEqual(len(judge.calls), 1)
```

Also assert cached tasks never occupy workers, failures are not cached, resume retries only failures, and workers below one are rejected.

- [ ] **Step 2: Run new tests and verify failure**

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py -k 'worker or probe_limit or resume_retries' -q
```

Expected: failure because `_resolve_judgments` lacks `judge_workers`.

- [ ] **Step 3: Implement deterministic bounded execution**

Scan tasks in declared order, resolve cache hits immediately, and select at most `judge_limit` misses before submission. Use `ThreadPoolExecutor(max_workers=judge_workers)` only for selected misses. Collect outcomes by task index, then perform conflict-safe cache writes and assemble judgments in original order on the controller thread. Count a submitted call once even when it fails; never cache exceptions or invalid labels.

- [ ] **Step 4: Expose worker bound and run tests**

Add `--judge-workers`; reject non-default use without `--run-judge`. Record effective workers in private receipt/manifest but not volatile scheduling details in HTML.

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_offline_evaluation.py -q
```

Expected: all pass and deterministic HTML is independent of worker count.

- [ ] **Step 5: Commit bounded judging**

```bash
git add code/trec_rag/offline_evaluation.py \
  code/trec_rag/competition_evaluation_report.py code/tests/test_offline_evaluation.py
git commit -m "feat: bound resumable RAGDoll judging"
```

---

### Task 4: Document and preflight the accepted-run workflow

**Files:**
- Modify: `.agents/skills/trec-rag-competition-debug-report/SKILL.md`
- Modify: `code/trec_rag/README.md`
- Modify: `code/tests/test_offline_evaluation.py`

**Interfaces:**
- Consumes: accepted CLI and worker bound from Tasks 2–3.
- Produces: canonical cache-only → one-task probe → bounded resume → cache-only completion commands.

- [ ] **Step 1: Add a failing documentation contract test**

Assert both docs contain accepted flags, immutable paths, `--judge-limit 1`, `--judge-workers 1`, bounded full resume, final cache-only replay, unavailable qrel/gold metrics, and tailnet-only detail reports.

- [ ] **Step 2: Run the test and verify failure**

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py -k accepted_workflow_documentation -q
```

Expected: missing accepted workflow assertions fail.

- [ ] **Step 3: Update docs and run regressions**

Document the exact repository-relative accepted paths and the exact absolute private main-checkout paths shown in Task 5. State the payload shape: one generated statement, cited selected-evidence text, narrative/source metadata, and no unrelated topic data. Require hit/miss counts before the probe and prohibit calling partial output complete.

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_offline_evaluation.py code/tests/test_accepted_rag_evaluation.py \
  code/tests/test_ragdoll_io.py -q
```

Expected: all pass.

- [ ] **Step 4: Commit workflow documentation**

```bash
git add .agents/skills/trec-rag-competition-debug-report/SKILL.md \
  code/trec_rag/README.md code/tests/test_offline_evaluation.py
git commit -m "docs: preflight accepted RAG evaluations"
```

---

### Task 5: Execute and validate both private 119-topic reports

**Files:**
- Create private/ignored: `/home/npatta01/data/competitions/trec_rag_2026/outputs/final-evaluation/rag26-ss1/`
- Create private/ignored: `/home/npatta01/data/competitions/trec_rag_2026/outputs/final-evaluation/rag26-ms1-final/`
- Reuse private/ignored: `/home/npatta01/data/competitions/trec_rag_2026/cache/ragdoll_support_judge/`
- Create derived/private: `/home/npatta01/codex-rendered/plans/trec-rag-2026-ragdoll-rag26-ss1.html`
- Create derived/private: `/home/npatta01/codex-rendered/plans/trec-rag-2026-ragdoll-rag26-ms1-final.html`

**Interfaces:**
- Consumes: Tasks 1–4, exact accepted JSONLs/metadata, private main retrieval artifact/handoff, and preserved multi-stage identity.
- Produces: two complete manifests with exact task totals, zero missing/failed/conflicting judgments, reconciled metrics, and privacy-scanned friendly HTML.

- [ ] **Step 1: Revalidate immutable inputs and private directories**

Run the five-file submission validator, recompute hashes, and compare with the ledger. Create work directories mode `0700`; store pre-run status/hashes privately.

Expected: all five PASS; RAG hashes equal the Global Constraints.

- [ ] **Step 2: Run cache-only inventory for each accepted run**

For single-pass:

```bash
PYTHONPATH=code .venv/bin/python -m trec_rag.competition_evaluation_report \
  --retrieval-config /home/npatta01/data/competitions/trec_rag_2026/configs/rag26_competition_retrieval_v2.yaml \
  --accepted-rag submissions/trec-rag-2026/rag/selected-evidence-sol-v1/singlepass/rag_output_trec_rag_2026.jsonl \
  --accepted-bundle-metadata submissions/trec-rag-2026/rag/selected-evidence-sol-v1/metadata.json \
  --handoff-manifest /home/npatta01/data/competitions/trec_rag_2026/outputs/facet-deepseek-b40-v3/generation_handoff_manifest.json \
  --work-dir /home/npatta01/data/competitions/trec_rag_2026/outputs/final-evaluation/rag26-ss1 \
  --cache-dir /home/npatta01/data/competitions/trec_rag_2026/cache/ragdoll_support_judge \
  --output /home/npatta01/data/competitions/trec_rag_2026/outputs/final-evaluation/rag26-ss1/evaluation_report.html
```

Expected: 3,155 tasks, zero hosted calls. Repeat with multi-stage accepted/work paths plus:

```text
--source-identity /home/npatta01/.codex/worktrees/rag26-ms1-full-run-1786308971/outputs/rag26-ms1-multistage-final/work/multistage_generation_identity.json
```

Expected: 7,008 tasks, zero hosted calls, source identity available.

- [ ] **Step 3: Record egress boundary and probe one miss per run**

Record provider/model/thinking, exact hits/misses, private raw-event path, workers, and payload shape without printing content. Repeat each cache-only command with:

```text
--run-judge --judge-limit 1 --judge-workers 1
```

Expected per run: one hosted call, one new completed label, zero failures/conflicts, one validated cache write. Stop on any discrepancy.

- [ ] **Step 4: Resume both full evaluations**

Repeat each command with:

```text
--run-judge --judge-workers 4
```

Resume identically after interruption. Expected: all 3,155 and 7,008 tasks complete; failures/conflicts/missing all zero.

- [ ] **Step 5: Require cache-only completion replay**

Rerun both without `--run-judge` into fresh private replay work directories.

Expected: hosted calls zero, cache reuse equals task total, missing/failed/conflicts zero, fully judged true.

- [ ] **Step 6: Reconcile metrics independently**

With a private read-only script, count FS/PS/NS from judgments, group by topic, and recompute RAGDoll metrics from assignments. Compare exact totals and six-decimal macros with manifest and presentation model.

Expected: labels sum to 3,155 and 7,008; 119 topics available; all three views agree.

- [ ] **Step 7: Copy only authoritative friendly HTML and verify HTTPS**

Copy the two `evaluation_report.html` files to the derived portal filenames. Copy no adjacent JSON/JSONL/event/receipt. Require HTTP 200 at:

```text
https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ss1.html
https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-ragdoll-rag26-ms1-final.html
```

Verify exact run/task totals, tailnet-only Serve, privacy scan, unchanged accepted hashes, and clean tracked status.

---

## Plan Completion Checks

- Run `git diff --check`.
- Run `PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_ragdoll_io.py code/tests/test_accepted_rag_evaluation.py code/tests/test_offline_evaluation.py -q`.
- Re-run five-file accepted validation and compare hashes.
- Do not mark a report complete unless cache-only replay has zero missing, failed, and conflicting judgments.
- Hand both verified tailnet URLs to `2026-08-10-final-analysis-navigation-and-dark-mode.md`.
