# 2026 Agentic Retrieval dstack Full-Run Plan

> Execute this plan with Luna xhigh as the primary orchestrator. Use Luna `worker` agents for independent, non-overlapping implementation tasks and a Sol reviewer only at the final consequential code boundary.

**Goal:** Distribute the frozen 119-topic agentic retrieval run over a rolling dstack GPU pool, durably publish each successful topic, continuously merge verified results into local staging, and produce final organizer artifacts only after all topic seals validate.

**Design:** [Distributed Agentic Retrieval for the 2026 Cohort](../specs/2026-08-07-agentic-dstack-distribution-design.md)

**Current status:** Tasks 0-6 are implemented. The focused worker/bundle/collector/export and offline two-topic round-trip suite passes (131 tests). The dstack/Hugging Face authentication preflight passed on `npatta01-framework`. Live offer preview is next; no machine has been provisioned and no hosted model call has been made.

**Critical path:** reconcile the competition contract -> plan initialization -> agentic topic bundle/import -> dstack worker/launcher -> offline integration test -> live canary -> rolling pool -> final 119-topic export.

## Global invariants

- Retrieval only. Do not start RAG generation.
- Use the canonical 119-topic config as the source of the cohort, but never mutate it for a smoke test.
- Freeze one run plan before any distributed topic starts.
- Use committed-only source transport and the recorded submodule revisions.
- Keep outputs, caches, provider responses, and Hugging Face artifacts private.
- Upload archive first and marker last; use create-only semantics.
- Publish and round-trip-verify each topic before a worker starts its next topic.
- Import into dedicated local staging during the run; build final root exports only at 119/119.
- Never print secret values or raw dstack diagnostic payloads.
- Do not launch a full live run without the explicit launch approval required by repository policy.

## Phase 0: Resolve the competition contract

### Task 0: Reconcile the retrieval-call limit

**Files:**

- Read: `AGENTS.md`
- Read: `trec-rag-skills/skills/trec-rag-2026-track-guidelines/SKILL.md`
- Read: the retrieval-task reference linked by that skill
- Read: `configs/rag26_competition_agentic_retrieval_v1.yaml`
- Read: `docs/superpowers/plans/2026-08-05-agentic-competition-retrieval-runner.md`
- Modify: this plan and the agentic config only if the canonical contract requires it

**Resolution (2026-08-07):** The canonical retrieval-task contract places no fixed maximum on internal candidate generation and explicitly permits query decomposition, fusion of multiple searches, and reranking. The 25-search ceiling is the repository's bounded fixed-planning architecture; the 100-call value is the separate agentic runner's internal safety budget. The agentic config remains unchanged. Organizer validity is enforced at the variable-depth final output boundary.

## Phase 1: Implement the distribution seam

### Task 1: Add plan-only agentic initialization

**Files:**

- Modify: `code/trec_rag/competition_agentic_retrieval.py`
- Modify: `code/trec_rag/agentic_run_state.py`
- Modify: `code/tests/test_competition_agentic_retrieval.py`
- Modify: `code/tests/test_agentic_run_state.py`
- Modify: `code/trec_rag/README.md`

**Step 1: Write failing tests**

Add tests proving an initialization-only invocation:

- selects and freezes the exact configured cohort;
- writes the canonical `AgenticRunPlan` and receipt;
- binds config bytes, narratives, source revision, and submodules;
- performs no provider, Pyserini, reranker, or cache call;
- is idempotent only for identical plan bytes;
- rejects a changed config or cohort under the same experiment directory.
- installs canonical plan bytes atomically on a worker and accepts an existing plan only when byte-identical.

Run:

```bash
.venv/bin/python -m pytest code/tests/test_competition_agentic_retrieval.py -q
```

Expected: new tests fail because no plan-only seam exists.

**Step 2: Add a narrow interface**

Extract or add a function equivalent to:

```python
def initialize_agentic_run(config_path: Path) -> AgenticRunPlanReceipt:
    """Freeze and validate the run plan without executing a topic."""

def deserialize_run_plan(body: bytes) -> AgenticRunPlan:
    """Validate canonical plan bytes and return the authenticated plan."""

def install_run_plan(*, work_dir: Path, body: bytes) -> AgenticRunPlan:
    """Atomically install exact plan bytes, or accept an identical plan."""
```

Expose it as `--initialize-only`. Keep normal create/resume behavior unchanged. Initialization must not require live runtime secrets because it performs no remote calls.

**Step 3: Pass targeted tests and document the command**

Run the targeted test again. Add README guidance stating that distributed workers consume an already-frozen plan and must never independently select a cohort.

**Step 4: Commit checkpoint**

After review and green tests:

```bash
git add code/trec_rag/competition_agentic_retrieval.py code/trec_rag/agentic_run_state.py code/tests/test_competition_agentic_retrieval.py code/trec_rag/README.md
git commit -m "feat: initialize agentic retrieval plans without execution"
```

### Task 1B: Add an assigned-only worker entry point

**Files:**

- Create: `code/trec_rag/competition_agentic_worker.py`
- Modify: `code/trec_rag/competition_agentic_retrieval.py`
- Create: `code/tests/test_competition_agentic_worker.py`

Normal `--resume --topic` is not a safe distributed-worker seam: the current runner still reasons about unresolved siblings and always attempts aggregate export. Extract the existing topic execution loop into a no-export function and make the worker entry point:

- install and authenticate the coordinator's exact plan bytes;
- validate a disjoint assigned subset without changing plan membership;
- execute only the assigned IDs;
- seal successful topics locally;
- never attempt run-level aggregate export;
- return machine-readable per-topic outcomes for the wrapper;
- preserve failed attempts and existing sealed topics.

Write failing tests for assigned-only execution, unresolved sibling isolation, foreign/duplicate assignment rejection, source/config drift, and no-export behavior. Then implement and run:

```bash
.venv/bin/python -m pytest \
  code/tests/test_competition_agentic_worker.py \
  code/tests/test_competition_agentic_retrieval.py \
  code/tests/test_agentic_run_state.py -q
```

### Task 2: Build the authenticated agentic topic bundle

**Files:**

- Create: `code/trec_rag/agentic_retrieval_shard_bundle.py`
- Create: `code/tests/test_agentic_retrieval_shard_bundle.py`
- Modify narrowly if needed: `code/trec_rag/agentic_run_state.py`
- Modify: `code/trec_rag/README.md`

**Step 1: Define the deep interface in tests**

Test public operations equivalent to:

```python
pack_topic(source_run_dir, topic_id, archive_path, marker_path)
verify_topic_bundle(archive_path, marker_path, expected_run_plan_sha256)
import_topic_bundle(archive_path, marker_path, destination_run_dir)
summarize_staging(destination_run_dir)
```

Tests must cover:

- deterministic bytes and canonical manifests;
- exact member allowlist and bounded sizes;
- successful topic-seal validation;
- `retrieval_topic.json`, `generation_topic.json`, and records-receipt closure;
- foreign plan/topic rejection;
- create-only and identical-only publication/import;
- verify-before-lock and idempotent journal recovery;
- rejection of traversal, symlinks, hard links, devices, duplicate paths, prefix collisions, truncated streams, trailing bytes, and digest/size mismatches;
- exclusion of TopicRecords SQLite/cache data, failed attempts, raw responses, traces, lock files, WAL/SHM, qrels, gold data, and weights.

Run:

```bash
.venv/bin/python -m pytest code/tests/test_agentic_retrieval_shard_bundle.py -q
```

Expected: fail before implementation.

**Step 2: Implement bundle schema and verification**

Adapt the proven archive-bounding, canonical-manifest, and transactional-import patterns from `competition_cache_bundle.py`, but keep a separate schema and module. Do not relax fixed-bundle forbidden paths. The minimal bundle is projection-portable for final export; it is not a portable TopicRecords/cache replay bundle.

Make the marker small and independently parseable. It must name the archive SHA-256, byte size, run-plan SHA-256, topic ID, bundle schema, and topic seal digest.

**Step 3: Implement transactional local import**

Import into a temporary run-specific path, create the empty successful-attempt directory required by `load_topic_seal`, fsync durable files, acquire a process lock only after full verification, then install create-only. Record prepare/conflict/complete journal states so an interrupted import can recover safely.

**Step 4: Pass tests and expose a CLI**

Provide `pack`, `verify`, `import`, and `status` commands with machine-readable JSON on stdout and diagnostics on stderr. Run:

```bash
.venv/bin/python -m pytest code/tests/test_agentic_retrieval_shard_bundle.py -q
.venv/bin/python -m trec_rag.agentic_retrieval_shard_bundle --help
```

**Step 5: Commit checkpoint**

Commit only the bundle module, tests, narrow state seam if needed, and its README section.

### Task 3: Add the local collector and append-only wave receipts

**Files:**

- Create: `code/trec_rag/agentic_retrieval_collector.py`
- Create: `code/tests/test_agentic_retrieval_collector.py`
- Reuse: `code/trec_rag/hf_bucket_listing.py`
- Modify: `code/trec_rag/README.md`

**Step 1: Write failing collector tests**

Use a fake HF CLI/listing and synthetic bundles to prove:

- exact path-component matching, including `rag2026-14` versus `rag2026-144`;
- marker-last eligibility;
- unseen bundle download and offline verification;
- idempotent import of identical topics;
- quarantine of malformed/foreign bundles;
- continuation after one topic fails;
- append-only wave receipts with imported, present, rejected, failed, and missing IDs;
- restart from receipts/journal without duplicate mutation;
- no final aggregate export below complete cohort count.
- import/export mutual exclusion so a complete-cohort export cannot race an import.

Run:

```bash
.venv/bin/python -m pytest code/tests/test_agentic_retrieval_collector.py -q
```

**Step 2: Implement one-cycle and watch modes**

The collector accepts the private bucket prefix, local frozen plan, staging root, polling interval, and an optional expected-topic subset. `--once` performs one bounded cycle; `--watch` repeats until complete or interrupted. It never deletes remote or local artifacts.

**Step 3: Add the cohort-complete export gate**

Call the existing agentic aggregate export only when every ordered plan topic has a valid imported topic seal. Below that threshold, produce status and receipts only.

**Step 4: Run tests and commit checkpoint**

Run collector, bundle, agentic runner, export, and run-state tests together before committing.

## Phase 2: Build the dstack execution path

### Task 4: Add the remote agentic shard wrapper

**Files:**

- Create: `code/tools/run_agentic_retrieval_worker.sh`
- Create: `code/tests/test_agentic_retrieval_shard_workflow.py`
- Modify: `code/trec_rag/README.md`

**Step 1: Write shell-contract tests first**

Prove the wrapper:

- validates run ID, task name, exact plan digest, and one-or-more unique topic IDs;
- accepts only topics present in the installed plan;
- verifies transported Git/submodule revisions;
- creates isolated config/output/work/cache paths;
- loads named secrets without printing values;
- runs topics sequentially on one GPU;
- packs, verifies, uploads archive, uploads marker last, lists, downloads, byte-compares, and verifies before advancing;
- uses `--ignore-existing` and has no delete/overwrite path;
- preserves completed sibling topics when a later topic fails;
- emits a safe failure receipt;
- kills and reaps the current process group on signals;
- optionally uploads a cache shard only after all submission-critical topic publications.

Run:

```bash
.venv/bin/python -m pytest code/tests/test_agentic_retrieval_shard_workflow.py -q
```

**Step 2: Implement the wrapper**

Reuse setup, secret-safety, exact HF listing, upload ordering, and round-trip verification patterns from `run_retrieval_cache_shard.sh`. Do not run multiple topics concurrently on one GPU.

**Step 3: Prove failure isolation**

Inject a failure into the second of two fake topics. Verify the first topic remains remotely complete, the second has no success marker, and the task exits nonzero after writing a failure receipt.

### Task 5: Add pinned dstack configuration and launcher

**Files:**

- Create: `.dstack/rag26-agentic-retrieval-worker.yaml`
- Create: `code/tools/apply_agentic_retrieval_worker.sh`
- Modify: `code/tests/test_agentic_retrieval_shard_workflow.py`
- Modify: `code/trec_rag/README.md`

**Step 1: Test configuration/transport invariants**

Require:

- pinned runtime image and dstack `0.20.29`;
- `A40`, `A6000`, or `L40S`, at least 48 GB GPU memory;
- at least 48 GB host RAM and 100 GB disk;
- on-demand marketplace instances;
- bounded `max_price`, zero idle retention, and bounded maximum duration;
- retry only for no-capacity;
- committed-only clean source transport;
- canonical remote/branch ancestry and pinned submodules;
- no `.env` or `.env.local` in transported files;
- private plan/config transport that does not create an ephemeral commit or otherwise change the plan's frozen Git `HEAD`;
- required dstack secret names only;
- preview by default and `--launch` as an explicit mode.

**Step 2: Implement launcher**

Adapt the validation and packaging safeguards from `apply_retrieval_cache_shard.sh`. Accept task name, run ID, plan digest, private artifact prefix, and assigned topic IDs. Materialize the exact committed source snapshot and supply plan/config as private untracked inputs; never create the fixed launcher's ephemeral transport commit because that would change the frozen source revision. Never modify the checked-in config.

Set a six-hour task maximum for the default one-topic task. A separate explicit two-topic fallback may use ten hours. Actual offers and price cap must be reviewed in preview before live launch. Require canary peak disk below 50 GB before retaining a 100 GB request; otherwise change the request to 200 GB before ramping.

**Step 3: Run targeted tests and inspect a preview**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_agentic_retrieval_shard_workflow.py -q
code/tools/apply_agentic_retrieval_worker.sh --help
```

Do not launch. Once the tree is committed and clean, create one temporary one-topic preview and record the offers without exposing diagnostics or secrets.

## Phase 3: Verify the system before spending compute

### Task 6: Add a fully offline two-topic integration test

**Files:**

- Create: `code/tests/test_agentic_distributed_round_trip.py`
- Modify as defects require: only the new agentic distribution modules/scripts

**Step 1: Exercise the whole chain**

With fake providers, fake HF transport, and a two-topic config:

1. initialize a run plan;
2. execute two topics in separate fake workers;
3. pack and publish each topic marker-last;
4. collect topic one while topic two is still pending;
5. import topic two in the next wave;
6. restart the collector and prove idempotence;
7. generate the complete small-cohort aggregate;
8. validate exact topic order, document closure, citations/handoff, and manifest-last receipt.

**Step 2: Inject recovery failures**

Test interruption after archive upload, during local import prepare, and after install before complete journal write. Prove every path either recovers identically or fails closed.

**Step 3: Run targeted and regression suites**

```bash
.venv/bin/python -m pytest \
  code/tests/test_agentic_distributed_round_trip.py \
  code/tests/test_agentic_retrieval_shard_bundle.py \
  code/tests/test_agentic_retrieval_collector.py \
  code/tests/test_agentic_retrieval_shard_workflow.py \
  code/tests/test_competition_agentic_retrieval.py -q
```

Then run the repository's broader relevant test suite and lint/type checks documented in `code/trec_rag/README.md` or project configuration.

### Task 7: Review and readiness audit

Use a Luna worker for a focused requirements audit, then use one Sol reviewer for the final security/correctness review because archive import, immutable publication, and deadline execution are consequential.

The primary agent must resolve findings and re-run tests. Before claiming readiness, verify:

- clean tracked branch with initialized submodules;
- no secrets, output artifacts, raw provider material, caches, or generated archives in Git;
- hostile archive tests pass;
- local offline round trip passes;
- dstack preview works;
- private HF direct-mode preflight passes;
- exact run-plan initialization is reproducible.

## Phase 4: Canary and rolling execution

### Task 8: Freeze the production run and present the launch gate

**Read-only/preparatory actions:**

1. Copy canonical configs to ignored run-specific local files only when an output namespace adjustment is required; do not change topic membership.
2. Run `--initialize-only` to freeze all 119 ordered topics.
3. Verify plan digest, config digest, source revision, submodules, and empty/new artifact namespace.
4. Publish and round-trip-verify the private plan artifact marker-last.
5. Preview current dstack offers for the selected GPU families.
6. Report to the user before launch:
   - 119 topic IDs/count;
   - experiment/run ID and exact private/local paths;
   - plan SHA-256;
   - expected cache reuse/miss status and hosted-call classes;
   - selected offers, hourly/max exposure, active-task limit, task duration;
   - required secrets present by name;
   - verification evidence.

Pause for explicit authorization to provision machines and make full live calls.

### Task 9: Launch the canary and ramp sample

After authorization:

1. Launch one representative unresolved topic as the first canary.
2. After it passes every gate, launch two independent one-topic tasks on different offers/providers when available.
3. Start the local collector in a durable terminal session or supervised foreground process.
4. Monitor safe task status and sanitized logs; do not print secret-bearing diagnostics.
5. Require for each canary:
   - successful manifest-last topic seal;
   - archive/marker upload ordering;
   - remote round-trip verification;
   - local offline import;
   - wave receipt;
   - observed runtime, GPU-memory headroom, API error/throttle rate, and publication time.
6. Retry only an exact unresolved topic with an empty completion prefix.

Do not expand if the canary or either ramp topic cannot round-trip into local staging.

### Task 10: Run the rolling pool

**Scheduler state:** Maintain a canonical local ledger with `pending`, `assigned`, `remote_complete`, `locally_imported`, `failed`, and `retryable` states. Derive completion from authenticated markers and local seals, not task exit codes alone.

**Queue policy:**

1. Assign one deterministic topic per dstack task by default.
2. Start with four active dstack tasks.
3. Refill a task slot immediately after terminal completion; do not wait for a wave boundary.
4. After two clean collector waves, expand to six.
5. Expand to eight only with healthy API latency/throttle rates, GPU memory, and no publication/import backlog.
6. Shrink one level on persistent 429/5xx growth or latency degradation.
7. Use a two-topic sequential queue only as a marketplace-scarcity fallback and only when measured runtime plus sync fits a ten-hour task. Do not initially assign three or four topics to one task.
8. Never run two topic processes concurrently on one GPU.

**Collector policy:**

- collect/import continuously while workers run;
- write one append-only receipt per completed collector cycle;
- display sealed/imported/missing/failing counts and exact topic IDs;
- quarantine foreign or malformed bundles without blocking valid siblings;
- keep the primary shared caches untouched during critical execution.

**Retry policy:**

- diagnose first;
- do not resubmit a dstack task name blindly;
- accept existing immutable completion when it verifies;
- requeue only topic IDs with no valid completion marker and no local seal;
- preserve failure receipts and attempt history;
- cap repeated semantic retries according to the runner's existing policy, distinct from transport retries.

### Task 11: Complete and validate the final export

Only when staging reports 119/119 valid topic seals:

1. Revalidate the run plan and every topic seal in ordered cohort order.
2. Run the existing agentic aggregate export once into the final run root.
3. Read the manifest-last receipt.
4. Verify `generation_handoff_manifest.json`, organizer TREC run, and full-text ZIP.
5. Validate exact 119 topic IDs/narratives, variable-depth TREC rows, six columns, unique `(qid, docid)`, ranking order, document/full-text closure, authenticated handoff, and all organizer constraints from the canonical track skill.
6. Confirm generation has not been run and no generation process has opened the TREC run, ZIP, qrels, gold nuggets, or evaluation scores.
7. Produce a private final readiness receipt with artifact paths and digests. Do not publish or serve outputs.

If the deadline arrives below 119/119, stop launching new work when instructed, preserve the private remote prefixes and local staging ledger, and report the exact authenticated completion set. Do not manufacture a partial final artifact under the canonical completed-run namespace.

## Phase 5: Secondary cache consolidation

After submission-critical topic results are safe:

1. enumerate optional worker cache bundles;
2. download and verify them in an isolated temporary root;
3. use the existing transactional cache merge machinery only for compatible cache records;
4. promote into `cache/retrieval/` and `cache/reranker/` under locks;
5. record imported, identical, conflicting, and rejected entries;
6. never let cache conflicts mutate the authenticated topic results.

## Verification checklist

- [x] Initialization-only tests pass with zero network/provider calls.
- [x] Agentic bundle hostile-input and deterministic-byte tests pass.
- [x] Collector exact-prefix, journal-recovery, and partial-cohort tests pass.
- [x] Shell workflow upload-order, signal, secret, and failure-isolation tests pass.
- [x] Offline two-topic end-to-end round trip passes.
- [ ] Relevant regression suite passes.
- [ ] Luna requirements audit resolved.
- [ ] Sol security/correctness review resolved.
- [ ] dstack/HF preflight passes.
- [ ] Live offer preview and maximum exposure reviewed.
- [ ] User explicitly authorizes live launch.
- [ ] One canary and two independent ramp topics publish and import successfully.
- [ ] Rolling ledger accounts for all 119 topic IDs exactly once.
- [ ] Final export is gated on 119 valid seals and passes organizer validation.

## Immediate next action

Resolve Task 0, then implement Tasks 1-3 as separately reviewed slices using fresh Luna workers with non-overlapping ownership:

- Worker A: plan-only initialization and its tests.
- Worker B: agentic topic-bundle module and its tests.
- Worker C: collector and wave-receipt module/tests, beginning from tests and coordinating only through the specified bundle interface.

The primary agent sequences the slices so each reviewed interface is stable before its consumer begins, integrates, runs the combined tests, then proceeds to the dstack wrapper and launcher. No live compute is started during these implementation tasks.
