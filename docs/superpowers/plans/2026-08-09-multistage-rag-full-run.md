# Multi-Stage RAG Full-Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a resumable, identity-bound full-run multi-stage RAG runner that consolidates the frozen per-topic final arm into one organizer-valid JSONL while leaving the single-pass runner unchanged.

**Architecture:** Add one focused orchestration module above the existing `_run_bounded_revision` state machine. It owns full-run identity, bounded topic concurrency, create/resume dispatch, strict final-row reload, all-or-nothing atomic consolidation, CLI dry-run reporting, and nothing inside generation prompts or decisions.

**Tech Stack:** Python 3.12, asyncio, filelock, existing RAG config/handoff/validation helpers, pytest.

## Global Constraints

- Reuse only `outputs/facet-deepseek-b40-v3/generation_handoff_manifest.json`; never rerun retrieval.
- Do not change `competition_rag.py`, frozen prompts, models, audit grouping, splice, Luna screen, retry, or repair behavior.
- Keep the existing maximum of three Sol reservations per topic.
- Keep single-pass and multi-stage run IDs, output directories, work state, and cached rows completely separate.
- Reject `experiment.mode: overwrite`; create and resume are the only multi-stage modes.
- Publish no JSONL unless every selected topic has exactly one locally validated final row in handoff order.
- Make no hosted calls in tests or dry runs.
- Keep local configs and generated artifacts ignored and private.

---

### Task 1: Full-Run Identity and Successful Consolidation

**Files:**
- Create: `code/trec_rag/competition_rag_multistage.py`
- Create: `code/tests/test_competition_rag_multistage.py`

**Interfaces:**
- Consumes: `RagGenerationConfig`, `GenerationHandoff`, selected `GenerationTopic` values, API key, and an injected async topic runner with the `_run_bounded_revision` signature.
- Produces: `_multistage_identity(config, handoff, topics) -> dict[str, Any]`, `_load_final_record(root, config, topic) -> dict[str, Any]`, and `run_multistage_generation(config, handoff, *, api_key, topic_runner=_run_bounded_revision) -> None`.

- [ ] **Step 1: Write the failing successful-run integration test**

Build a literal two-topic authenticated handoff fixture. The fake external topic runner writes a realistic v5 bounded state plus one registered `evaluation/final/submission.jsonl` for each topic. Call `run_multistage_generation` in create mode and assert:

```python
assert invoked == [("rag2026-0", "create"), ("rag2026-1", "create")]
assert published_ids == ["rag2026-0", "rag2026-1"]
assert all(row["metadata"]["run_id"] == f"{config.run_id}-final" for row in rows)
assert json.loads((config.work_dir / "multistage_generation_identity.json").read_text())[
    "handoff_manifest_sha256"
] == handoff.manifest_sha256
```

The production change that makes this test pass is the new run-level orchestrator; wrong topic order, missing identity, early publication, or wrong final run ID must fail it.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_competition_rag_multistage.py::test_create_publishes_all_validated_topics_in_handoff_order -q
```

Expected: collection fails because `trec_rag.competition_rag_multistage` does not exist.

- [ ] **Step 3: Implement the minimal create path**

Implement:

```python
MULTISTAGE_IDENTITY_VERSION = 1

def _multistage_identity(config, handoff, topics):
    return {
        "identity_version": MULTISTAGE_IDENTITY_VERSION,
        "trial_contract_version": TRIAL_CONTRACT_VERSION,
        "handoff_schema_version": handoff.schema_version,
        "handoff_manifest_sha256": handoff.manifest_sha256,
        "submission_run_id": f"{config.run_id}-final",
        "topics": [
            {
                "topic_id": topic.topic_id,
                "context_sha256": topic.context_sha256,
                "bounded_identity": _bounded_identity(config, handoff, topic),
            }
            for topic in topics
        ],
    }
```

Create the work directory and identity before dispatch. Bound concurrent topic execution with
`asyncio.Semaphore(config.concurrency)`. After all topics return, reload their registered final
artifacts, run the existing organizer and exact-hint validators, and publish with
`_atomic_write_text` only after the complete ordered list is in memory.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Task 1 test command and require one pass with no warnings.

- [ ] **Step 5: Commit the independently working create path**

```bash
git add code/trec_rag/competition_rag_multistage.py \
  code/tests/test_competition_rag_multistage.py
git commit -m "feat: orchestrate full multistage RAG runs"
```

---

### Task 2: Resume, Identity Refusal, and All-or-Nothing Failure

**Files:**
- Modify: `code/trec_rag/competition_rag_multistage.py`
- Modify: `code/tests/test_competition_rag_multistage.py`

**Interfaces:**
- Consumes: the Task 1 identity and per-topic roots.
- Produces: `_prepare_multistage_state(config, identity) -> None` and deterministic topic mode selection (`resume` for an existing root, `create` otherwise).

- [ ] **Step 1: Write failing resume and failure tests**

Add two tests:

```python
def test_resume_reuses_started_topic_and_creates_unstarted_topic(...):
    # Precreate the exact run identity and topic 0 final state.
    # Assert dispatch is [(topic 0, "resume"), (topic 1, "create")].

def test_identity_change_or_missing_final_never_publishes(...):
    # Change one identity-bound setting and require ValueError before dispatch.
    # In a separate fresh case, let one fake topic omit its final and assert the
    # organizer JSONL does not exist.
```

The production breaks caught are cross-run row mixing and partial submission publication.

- [ ] **Step 2: Run the two tests and verify RED**

Run:

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_competition_rag_multistage.py -q
```

Expected: the new resume/refusal assertions fail while Task 1 remains green.

- [ ] **Step 3: Implement exact resume/refusal behavior**

On create, refuse an existing output or any work artifact. On resume, require and compare the full
run identity. For each topic, inspect `_bounded_private_root(config, topic)` only to select create
or resume; let `_run_bounded_revision` authenticate the actual state. Gather topic exceptions,
allow siblings to finish, and raise one sanitized summary before consolidation. Reject overwrite
before any deletion or provider call.

- [ ] **Step 4: Run all focused tests and verify GREEN**

Run the Task 2 test command and require all tests to pass.

- [ ] **Step 5: Commit resume safety**

```bash
git add code/trec_rag/competition_rag_multistage.py \
  code/tests/test_competition_rag_multistage.py
git commit -m "fix: make multistage generation safely resumable"
```

---

### Task 3: CLI, Dry Run, Local Configurations, and Runbook

**Files:**
- Modify: `code/trec_rag/competition_rag_multistage.py`
- Modify: `code/tests/test_competition_rag_multistage.py`
- Modify: `code/trec_rag/README.md`
- Create ignored: `configs/local/rag26-rag-singlepass-sol-final-v1.yaml`
- Create ignored: `configs/local/rag26-rag-multistage-sol-final-v1.yaml`
- Create ignored: `configs/local/rag26-rag-multistage-smoke-0-1-2.yaml`

**Interfaces:**
- Consumes: `--config CONFIG [--dry-run]`.
- Produces: a provider-free dry-run budget and a live CLI that loads secrets only after config and handoff authentication.

- [ ] **Step 1: Write the failing dry-run/CLI test**

Invoke `main(["--config", str(config_path), "--dry-run"])` with a literal three-topic handoff and
no API key. Assert output includes exact topic/group/Sol/Luna/concurrency/path values, no output or
work path is created, and the injected provider runner is never called.

- [ ] **Step 2: Run the dry-run test and verify RED**

Run the named dry-run test and require failure because the CLI is absent.

- [ ] **Step 3: Implement CLI and dry-run reporting**

Authenticate config, handoff, and selected topics first. For dry run print:

```text
topics=<count>,groups=<sum>
calls=sol_routine:<2N>,sol_max:<3N>,luna_min:<N+groups>,luna_max:<2N+groups>,provider:0
concurrency=<configured>
handoff=<path>
output=<path>
work=<path>
```

For live execution, load repository environment files, require the configured API key without
printing it, and call `asyncio.run(run_multistage_generation(...))`.

- [ ] **Step 4: Create the three ignored configs and document commands**

Copy the checked-in Sol config shape. Use distinct IDs/output directories, `mode: create`, the
same v3 handoff, and topics `rag2026-0`, `rag2026-1`, `rag2026-2` only in the smoke config. Add
README create/resume/dry-run commands and state explicitly that no cross-strategy generation rows
are reusable.

- [ ] **Step 5: Run focused tests and the real dry run**

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_competition_rag_multistage.py \
  code/tests/test_narrative_blueprint.py \
  code/tests/test_operation_screen.py -q

PYTHONPATH=code .venv/bin/python -m trec_rag.competition_rag_multistage \
  --config configs/local/rag26-rag-multistage-sol-final-v1.yaml --dry-run
```

Require no hosted calls, 119 topics, 691 groups, 238/357 Sol counts, and 810/929 Luna counts.

- [ ] **Step 6: Commit the reusable CLI and runbook**

Stage only tracked code/tests/README; verify ignored configs are absent from the index.

```bash
git add code/trec_rag/competition_rag_multistage.py \
  code/tests/test_competition_rag_multistage.py code/trec_rag/README.md
git commit -m "feat: expose multistage competition RAG runner"
```

---

### Task 4: Release Verification

**Files:**
- Modify: `docs/superpowers/plans/2026-08-09-multistage-rag-full-run.md`

**Interfaces:**
- Consumes: the completed tracked tree and ignored local configs.
- Produces: verified implementation evidence and the exact next live-run decision.

- [ ] **Step 1: Run static and focused verification**

```bash
/home/npatta01/anaconda3/bin/ruff check \
  code/trec_rag/competition_rag_multistage.py \
  code/tests/test_competition_rag_multistage.py
git diff --check
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_competition_rag_multistage.py \
  code/tests/test_narrative_blueprint.py \
  code/tests/test_luna_operation_screen_replay.py \
  code/tests/test_operation_screen.py \
  code/tests/test_generation_handoff.py \
  code/tests/test_competition_rag.py -q
```

- [ ] **Step 2: Run the full suite from an external temp root**

Use a fresh `/var/tmp/trec-rag-pytest.*` directory so path-discovery tests remain outside every
repository marker and clone-heavy tests do not recursively copy their own temp tree:

```bash
TMPDIR=<fresh-var-tmp-dir> PYTHONPATH=code \
  .venv/bin/python -m pytest code/tests -q
```

- [ ] **Step 3: Re-read the design and verify every invariant**

Confirm unchanged single-pass code, unchanged prompt constants, exact run-level identity, no
overwrite path, all-or-nothing publication, handoff ordering, strict final validation, separate
configs, no tracked private artifacts, and zero hosted calls.

- [ ] **Step 4: Record verification and commit**

Update this plan's status/evidence, stage only the plan, and commit:

```bash
git add docs/superpowers/plans/2026-08-09-multistage-rag-full-run.md
git commit -m "docs: record multistage runner verification"
```

- [ ] **Step 5: Handoff live execution separately**

Report the two full config paths, three-topic smoke config, exact call ceilings, fresh output
directories, and expected cache misses. Do not start the smoke or either full hosted run without
explicit live-run authorization.
