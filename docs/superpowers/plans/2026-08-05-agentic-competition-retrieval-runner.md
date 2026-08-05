# Agentic Competition Retrieval Runner Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to
> implement this plan task-by-task. Use `superpowers:test-driven-development`
> for every production behavior and `superpowers:verification-before-completion`
> before the final handoff.

**Goal:** Add a durable, resumable agentic competition-retrieval CLI that uses
the production 20-researcher budget without topic time deadlines, reuses shared
content-addressed caches, and publishes the same sealed generation handoff as
the existing fixed-retrieval pipeline.

**Architecture:** Keep the proven fixed runner and config byte-stable. Add a
separate strict agentic config and runner, backed by authenticated run-plan and
per-topic projection receipts. Each incomplete topic restarts from its official
narrative while exact retrieval, document, reranker, and model artifacts remain
cacheable. Both retrieval modes meet only at the existing typed
`GenerationTopic` / `GenerationHandoff` boundary.

**Tech stack:** Python 3.12, pytest, YAML, SQLite, TopicRecords,
content-addressed document/retrieval/reranker caches, deterministic ZIP/JSON,
OpenRouter DeepSeek, Pyserini REST, Mixedbread ROCm reranking.

## Global constraints

- Work only in the isolated worktree rooted at
  `/tmp/trec-rag-e2e-isolated-worktree-7e9d3b9-20260805` on
  `codex/fix-agentic-facet-finalization`.
- Preserve `configs/rag26_competition_retrieval_v2.yaml` and the existing fixed
  runner's behavior and artifact bytes.
- Never use raw original-query passages as generation fallback evidence.
- A topic with zero live grounded nuggets fails once and waits for a manual
  `--resume`; the runner never retries the whole topic automatically.
- Count-budget exhaustion may publish grounded partial evidence. Provider,
  scoring, evidence-integrity, and zero-grounded failures may not.
- Default create refuses an existing output namespace. Resume never overwrites
  a sealed topic and never changes the run's original topic cohort.
- No all-topic or RAG live run is authorized. The only authorized live
  validation is `rag2026-0` in a fresh local smoke namespace after tests pass.
- Keep outputs, caches, responses, generated claims, and debug material private
  and ignored.

---

### Task 1: Make elapsed-time deadlines optional

**Files:**

- Modify: `code/tests/test_deepagent_budget.py`
- Modify: `code/trec_rag/deepagent_budget.py`

**Contract:** `ResearchBudgetConfig()` has `soft_seconds=None` and
`hard_seconds=None`. Elapsed time remains observable, but cannot stop or refuse
work unless a caller explicitly supplies finite deadlines. Existing explicit
deadline behavior remains supported for bounded tests and experiments.

- [ ] **Step 1: Add the failing default-no-deadline tests.**

  Add one test that advances a fake clock far beyond the old 30/60-minute
  values, admits a survey researcher, and asserts both deadline flags and the
  stop code remain false/`None`. Add validation cases showing either deadline
  may be `None`, while Boolean, negative, non-finite, or `hard < soft` finite
  combinations remain rejected.

- [ ] **Step 2: Verify RED.**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_deepagent_budget.py::test_default_budget_has_no_elapsed_time_stop \
    code/tests/test_deepagent_budget.py::test_optional_deadline_validation \
    -q
  ```

  Expected: the first test refuses on the old hard deadline and the validation
  test fails because `None` is not accepted.

- [ ] **Step 3: Implement the minimal optional-deadline behavior.**

  Change the two config fields to `float | None`, validate only non-`None`
  values, compare finite deadlines only when present, and keep snapshot flags
  false when absent. Do not remove deadline codes or explicit deadline tests.

- [ ] **Step 4: Verify GREEN and regressions.**

  ```bash
  .venv/bin/python -m pytest code/tests/test_deepagent_budget.py -q
  ```

- [ ] **Step 5: Commit.**

  ```bash
  git add code/tests/test_deepagent_budget.py code/trec_rag/deepagent_budget.py
  git commit -m "feat: make agentic time budgets optional"
  ```

### Task 2: Add the strict agentic-only configuration

**Files:**

- Create: `code/tests/test_agentic_retrieval_config.py`
- Create: `code/trec_rag/agentic_retrieval_config.py`
- Create: `configs/rag26_competition_agentic_retrieval_v1.yaml`

**Contract:** `load_agentic_retrieval_config(path)` accepts only
`agentic_retrieval_config_v1` with `retrieval_mode: agentic`, rejects duplicate
or unknown YAML keys, resolves `cache/...` through `repo_cache_root`, and binds
all production model/index/chunking/budget identities. Topic selection uses
only the official narrative TSV and repeated topic IDs.

- [ ] **Step 1: Write strict-schema and path-behavior tests.**

  Cover: canonical config loads all official narratives; fixed config is
  rejected by the agentic loader; agentic config is rejected by the fixed
  loader; unknown/duplicate keys fail; unsafe experiment IDs fail; relative
  cache roots resolve through a simulated shared checkout; repeated topic
  selectors preserve official source order; and all production budget values
  equal the approved 20/3/100/20/8/16/8/30/80/2/3/2 limits with no elapsed
  deadlines.

- [ ] **Step 2: Verify RED.**

  ```bash
  .venv/bin/python -m pytest code/tests/test_agentic_retrieval_config.py -q
  ```

  Expected: import failure because the agentic config module does not exist.

- [ ] **Step 3: Implement typed config records and the loader.**

  Define immutable experiment, retrieval, passage/snippet, model, cache,
  budget, and execution records. Reuse the strict unique-key YAML policy and
  official `Topic` loader without importing the fixed runner. Validate the
  pinned ClimbMix corpus epoch, Mixedbread model/revision, 1,000 source docs,
  100 passages, 3,500/350 chunking, and one topic worker. Expose normalized
  non-secret identity data for receipts.

- [ ] **Step 4: Add the canonical full-run YAML.**

  Use a distinct experiment ID, the 119-topic official TSV, shared cache paths
  (`cache/retrieval/pyserini_remote`, `cache/reranker`,
  `cache/documents/v1`, `cache/models/huggingface`), the pinned DeepSeek model,
  and the approved count budgets. Do not add time-budget keys.

- [ ] **Step 5: Verify GREEN and fixed-config isolation.**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_agentic_retrieval_config.py \
    code/tests/test_facet_pilot_config.py \
    -q
  git diff --exit-code 7e9d3b9 -- configs/rag26_competition_retrieval_v2.yaml
  ```

- [ ] **Step 6: Commit.**

  ```bash
  git add code/tests/test_agentic_retrieval_config.py \
    code/trec_rag/agentic_retrieval_config.py \
    configs/rag26_competition_agentic_retrieval_v1.yaml
  git commit -m "feat: add strict agentic retrieval config"
  ```

### Task 3: Give final synthesis two attempts and grounded-only recovery

**Files:**

- Modify: `code/tests/test_deepagent_research.py`
- Modify: `code/tests/test_deepagent_evidence.py`
- Modify: `code/tests/test_deepagent_retrieval.py`
- Modify: `code/tests/test_topic_records.py`
- Modify: `code/trec_rag/deepagent_research.py`
- Modify: `code/trec_rag/deepagent_evidence.py`
- Modify: `code/trec_rag/deepagent_retrieval.py`
- Modify: `code/trec_rag/topic_records.py`

**Contract:** The coordinator receives at most two directed closeout updates
against the same grounded state. If valid selections still do not exist, the
retriever deterministically selects up to
`MAX_DRAFT_NUGGETS_PER_NEED` live grounded nuggets in each need's recorded
order. Zero live grounded nuggets returns an explicit failed topic; no passage
fallback and no automatic whole-topic retry exists.

- [ ] **Step 1: Add middleware retry regressions.**

  Test that an invalid first closeout leaving `closeout_pending()` true gets one
  second directed `update_retrieval_state` turn, that a valid first closeout
  gets no redundant retry, and that a still-invalid second closeout cannot get
  a third semantic attempt.

- [ ] **Step 2: Verify middleware RED.**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_deepagent_research.py -k "closeout and (retry or synthesis)" \
    -q
  ```

- [ ] **Step 3: Implement the two-attempt middleware counter.**

  Replace the Boolean synthesis grant with an attempt counter bounded by
  `synthesis_reserve_turns`. A second attempt is granted only while live
  evidence still needs selection; research tools stay disabled during both
  synthesis turns.

- [ ] **Step 4: Add deterministic recovery tests.**

  Exercise the real `EvidenceCoverageState`: superseded nuggets are excluded;
  nuggets not associated with a need are excluded; per-need source order and
  the maximum are preserved; existing valid selections are retained; no live
  nuggets returns a zero-grounded result rather than manufacturing rows.

- [ ] **Step 5: Verify recovery RED.**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_deepagent_evidence.py -k "grounded_recovery" -q
  ```

- [ ] **Step 6: Implement recovery as one validated state transition.**

  Add a public recovery method that uses only already admitted, live,
  need-associated nuggets. It records `partial` need status where needed and
  returns whether recovery changed selection; it does not accept passages,
  search results, or external text.

- [ ] **Step 7: Add retriever completion tests.**

  Through the existing real TopicRecords test harness, assert:

  - two invalid synthesis updates plus grounded nuggets produce
    `synthesis_outcome="deterministic_grounded_recovery"` and a complete,
    exportable partial topic;
  - count-budget exhaustion with grounded selected nuggets is complete with
    stopping reason `budget_exhausted`;
  - zero grounded nuggets is incomplete with `zero_grounded_nuggets` and no
    fallback evidence;
  - provider/scoring/evidence-integrity failures remain incomplete.

- [ ] **Step 8: Verify retriever RED, then implement and verify GREEN.**

  Add `synthesis_outcome` to `AgentRetrievalResult`, run recovery before the
  holistic snapshot, and permit authenticated TopicRecords completion to
  record grounded `complete/budget_exhausted`. Keep failure states explicit.

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_deepagent_research.py \
    code/tests/test_deepagent_evidence.py \
    code/tests/test_deepagent_retrieval.py \
    code/tests/test_topic_records.py \
    -q
  ```

- [ ] **Step 9: Commit.**

  ```bash
  git add code/tests/test_deepagent_research.py \
    code/tests/test_deepagent_evidence.py \
    code/tests/test_deepagent_retrieval.py \
    code/tests/test_topic_records.py \
    code/trec_rag/deepagent_research.py \
    code/trec_rag/deepagent_evidence.py \
    code/trec_rag/deepagent_retrieval.py \
    code/trec_rag/topic_records.py
  git commit -m "feat: recover agentic drafts from grounded nuggets"
  ```

### Task 4: Project agentic evidence into Retrieval and Generation records

**Files:**

- Create: `code/tests/test_agentic_generation_export.py`
- Create: `code/trec_rag/agentic_generation_export.py`
- Modify: `code/trec_rag/deepagent_submission.py`
- Modify: `code/tests/test_deepagent_submission.py`

**Contract:** One pure projector consumes an official `Topic`, the final
`EvidenceCoverageReport`, sealed `TopicEvidenceSnapshot`, fused candidates,
immutable search sequence, and document store. It emits variable-depth TREC
rows, full-text candidate records, and one authenticated `GenerationTopic`.

- [ ] **Step 1: Add variable-depth document-order tests.**

  Verify that only documents supporting live non-superseded nuggets appear;
  supporting docs in final RRF order come first; remaining docs use earliest
  `(search ordinal, passage rank, docid)`; duplicates collapse; ranks are
  contiguous; and scores are positive strictly decreasing integers derived as
  `document_count - rank + 1`.

- [ ] **Step 2: Verify ranking RED, then implement GREEN.**

  Add a new agentic ranking function without changing legacy
  `rank_for_submission` semantics.

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_deepagent_submission.py -k "agentic_document_order" -q
  ```

- [ ] **Step 3: Add real projection tests.**

  Build a real document store and typed snapshot with Unicode passages. Assert
  one evidence group per selected need; one selected cluster and claim hint per
  chosen live nugget; exact passage text, document hash, char/byte offsets, and
  final document rank; deterministic group-scoped evidence IDs; and generation
  citation docids are subsets of both TREC rows and full-text candidates.

  Add negative cases for missing/superseded/ungrounded/unassociated selected
  nuggets, mismatched passage/document hashes, missing cited docs, and zero
  selected grounded evidence. Explicitly assert original-query-only passages
  never enter the handoff.

- [ ] **Step 4: Verify projection RED.**

  ```bash
  .venv/bin/python -m pytest code/tests/test_agentic_generation_export.py -q
  ```

- [ ] **Step 5: Implement the pure projector.**

  Use full authenticated `SourcePassage` text and offsets for citations, not
  model-written quotes. Compute a non-circular retrieval-topic receipt from the
  canonical Retrieval/full-text projection and place that digest in
  `TopicSourceReceipts`; the outer per-topic manifest will authenticate both
  this receipt and serialized `GenerationTopic` bytes.

- [ ] **Step 6: Verify GREEN and shared handoff compatibility.**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_agentic_generation_export.py \
    code/tests/test_deepagent_submission.py \
    code/tests/test_generation_handoff.py \
    code/tests/test_generation_handoff_export.py \
    -q
  ```

- [ ] **Step 7: Commit.**

  ```bash
  git add code/tests/test_agentic_generation_export.py \
    code/tests/test_deepagent_submission.py \
    code/trec_rag/agentic_generation_export.py \
    code/trec_rag/deepagent_submission.py
  git commit -m "feat: project agentic grounded evidence"
  ```

### Task 5: Add authenticated run plans and resumable topic seals

**Files:**

- Create: `code/tests/test_agentic_run_state.py`
- Create: `code/trec_rag/agentic_run_state.py`

**Contract:** Create writes `work/run_plan.json` before work. The plan embeds
the exact config bytes (base64) and SHA-256, run ID, ordered original topic
cohort and narrative hashes, official TSV hash, superproject revision, and
submodule revisions, with a self-authenticating plan digest. Per-topic success
publishes create-only projection artifacts followed by a manifest-last receipt;
failed attempts remain private and unsealed.

- [ ] **Step 1: Add run-plan create/resume tests.**

  Cover create-only output, byte-identical readback, changed config bytes/run
  ID/topic narrative/topic source/source revision/submodule revision refusal,
  selector-outside-plan refusal, and resume selectors narrowing execution while
  leaving `planned_topic_ids` unchanged.

- [ ] **Step 2: Verify run-plan RED.**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_agentic_run_state.py -k "run_plan" -q
  ```

- [ ] **Step 3: Implement canonical plan publication and validation.**

  Use private permissions, atomic create-only links, directory fsync, exact
  field validation, and hash-before-use. Never infer resume compatibility from
  a path alone.

- [ ] **Step 4: Add per-topic attempt/seal tests.**

  Verify monotonically numbered attempts retain earlier failures; a sealed
  projection authenticates Retrieval rows, full text, serialized
  `GenerationTopic`, ledger receipt, status/stopping reason/synthesis outcome;
  identical republish is idempotent; conflicting bytes fail; corrupt/missing
  artifacts fail; and only a valid success seal is considered resumable.

- [ ] **Step 5: Verify topic-state RED, then implement GREEN.**

  ```bash
  .venv/bin/python -m pytest code/tests/test_agentic_run_state.py -q
  ```

- [ ] **Step 6: Commit.**

  ```bash
  git add code/tests/test_agentic_run_state.py code/trec_rag/agentic_run_state.py
  git commit -m "feat: seal resumable agentic topic state"
  ```

### Task 6: Publish standard root artifacts manifest-last

**Files:**

- Create: `code/tests/test_agentic_retrieval_export.py`
- Create: `code/trec_rag/agentic_retrieval_export.py`

**Contract:** Root export joins valid topic seals in the original run-plan
order and writes exactly:

- `r_output_trec_rag_2026.tsv`
- `retrieval_with_text.jsonl.zip`
- `generation_handoff_manifest.json`
- `retrieval_export_manifest.json` last

- [ ] **Step 1: Add deterministic multi-topic export tests.**

  Assert topic/source order, variable depth, unique docids per topic, strictly
  decreasing integer scores, deterministic ZIP member metadata/bytes, standard
  handoff schema accepted by `load_generation_handoff`, exact closure between
  handoff citations/TREC/ZIP, artifact sizes/hashes, topic statuses and
  synthesis outcomes in the outer receipt, and byte-identical idempotent
  republish.

- [ ] **Step 2: Add fail-closed tests.**

  Root artifacts remain absent when any planned topic lacks a valid seal or any
  seal/artifact/hash/topic identity changes. Conflicting existing root bytes do
  not get overwritten and the outer manifest is never published early.

- [ ] **Step 3: Verify RED.**

  ```bash
  .venv/bin/python -m pytest code/tests/test_agentic_retrieval_export.py -q
  ```

- [ ] **Step 4: Implement prepare-then-publish export.**

  Validate every topic and all aggregate bytes before acquiring the export
  lock. Serialize TREC and canonical JSON locally; create the fixed-name ZIP
  member with 1980 timestamp and private mode; use
  `prepare_generation_handoff_artifact`; publish the three payloads atomically
  and the outer receipt last.

- [ ] **Step 5: Verify GREEN and RAG-loader compatibility.**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_agentic_retrieval_export.py \
    code/tests/test_generation_handoff.py \
    code/tests/test_competition_rag.py \
    -q
  ```

- [ ] **Step 6: Commit.**

  ```bash
  git add code/tests/test_agentic_retrieval_export.py \
    code/trec_rag/agentic_retrieval_export.py
  git commit -m "feat: publish agentic retrieval artifacts"
  ```

### Task 7: Add the production agentic CLI and targeted resume

**Files:**

- Create: `code/tests/test_competition_agentic_retrieval.py`
- Create: `code/trec_rag/competition_agentic_retrieval.py`

**Contract:**

```bash
.venv/bin/python-rocm -m trec_rag.competition_agentic_retrieval CONFIG
.venv/bin/python-rocm -m trec_rag.competition_agentic_retrieval CONFIG --resume
.venv/bin/python-rocm -m trec_rag.competition_agentic_retrieval \
  CONFIG --resume --topic rag2026-19
```

Create's topic selectors define the immutable run-plan cohort. Resume without
selectors runs all outstanding topics; targeted resume executes only named
outstanding topics. Both skip sealed successes and aggregate over the original
cohort. Any unresolved topic leaves root artifacts absent and prints its IDs
plus a copy-paste resume command.

- [ ] **Step 1: Add orchestration tests using production-shaped local seams.**

  Cover preflight before dependency construction; secret-name presence without
  value leakage; tracked-dirty/submodule/config drift refusal; run plan written
  before the first topic executor call; create namespace refusal; completed
  topic skip; failed topic restart in a new attempt; resume-all; targeted
  resume; selector-outside-plan refusal; 19-complete/20th-failed then targeted
  success publishing all 20; and nonzero failure output containing unresolved
  IDs and the exact command.

- [ ] **Step 2: Verify orchestration RED.**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_competition_agentic_retrieval.py -q
  ```

- [ ] **Step 3: Implement delayed production dependency construction.**

  Load repo env, set the configured shared Hugging Face cache before importing
  the lazy model runtime, build `_build_topic_passage_search`,
  `MixedbreadPassageScorer`, `RelevantSnippetExtractor`, `DocumentStore`,
  `TopicRecordsBuilder`, and `DeepAgentRetriever.from_env` from strict config.
  Use sequential topic execution (`topic_workers: 1`) and retain the retriever's
  within-topic concurrency of three researchers.

- [ ] **Step 4: Implement the topic transaction.**

  Create a fresh attempt, run from the official narrative, publish TopicRecords,
  project grounded evidence, and seal only after every projection validation
  passes. On zero grounded nuggets or another operational failure, write a
  private attempt diagnostic without provider response text and return the
  topic as unresolved. Never call the same topic twice in one invocation.

- [ ] **Step 5: Implement CLI parsing/help and exit behavior.**

  `--resume` is operational and absent from semantic YAML. Repeated `--topic`
  is allowed on create and resume. Create requires at least one selected topic;
  resume uses stored cohort. Print a concise JSON receipt on success and an
  actionable, secret-free error on incomplete runs.

- [ ] **Step 6: Verify GREEN plus fixed-runner regressions.**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_competition_agentic_retrieval.py \
    code/tests/test_competition_retrieval_v2.py \
    code/tests/test_competition_topic_dispatch.py \
    -q
  ```

- [ ] **Step 7: Commit.**

  ```bash
  git add code/tests/test_competition_agentic_retrieval.py \
    code/trec_rag/competition_agentic_retrieval.py
  git commit -m "feat: add resumable agentic retrieval CLI"
  ```

### Task 8: Document create, resume-all, and targeted repair

**Files:**

- Modify: `code/trec_rag/README.md`

**Contract:** An agent reading the README can repair exactly one failed topic
into the same 20-topic artifact namespace without editing config, changing run
ID, rerunning the 19 successes, or losing shared cache reuse.

- [ ] **Step 1: Add the operator section.**

  Document fixed versus agentic configs, one/two-topic local config-copy smoke
  safety, preflight requirements, create, resume-all, targeted resume, original
  cohort semantics, zero-grounded manual retry, exact cache hit identity,
  uncached OpenRouter calls, no overwrite mode, root publication conditions,
  artifact names, and unchanged RAG consumption. Include the 19/20 repair
  example with copy-paste commands.

- [ ] **Step 2: Verify CLI help and examples manually.**

  ```bash
  .venv/bin/python -m trec_rag.competition_agentic_retrieval --help
  ```

- [ ] **Step 3: Commit.**

  ```bash
  git add code/trec_rag/README.md
  git commit -m "docs: explain agentic retrieval resume"
  ```

### Task 9: Run the full offline verification matrix

**Files:**

- Verify all changed production and test files.
- Verify unchanged fixed config and pinned submodules.

- [ ] **Step 1: Run focused contract suites.**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_deepagent_budget.py \
    code/tests/test_deepagent_research.py \
    code/tests/test_deepagent_evidence.py \
    code/tests/test_deepagent_retrieval.py \
    code/tests/test_deepagent_submission.py \
    code/tests/test_topic_records.py \
    code/tests/test_agentic_retrieval_config.py \
    code/tests/test_agentic_generation_export.py \
    code/tests/test_agentic_run_state.py \
    code/tests/test_agentic_retrieval_export.py \
    code/tests/test_competition_agentic_retrieval.py \
    code/tests/test_generation_handoff.py \
    code/tests/test_generation_handoff_export.py \
    code/tests/test_competition_rag.py \
    -q
  ```

- [ ] **Step 2: Run the entire offline suite.**

  ```bash
  .venv/bin/python -m pytest code/tests -q
  ```

- [ ] **Step 3: Verify hygiene and fixed-path stability.**

  ```bash
  git diff --check
  git status --short
  git submodule status --recursive
  git diff --exit-code 7e9d3b9 -- configs/rag26_competition_retrieval_v2.yaml
  ```

- [ ] **Step 4: Commit any test-only corrections through their own RED/GREEN
  cycle, then re-run Steps 1-3.**

### Task 10: Import the validated warm cache and run the authorized smoke

**Private inputs:**

- Retrieval/document/reranker source:
  `/tmp/trec-rag-deepagent-cache-facetfix-46072b3-rag2026-0-20260805`
- Model source:
  `/tmp/trec-rag-deepagent-runtime-facetfix-46072b3-rag2026-0-20260805`
- Target: the main/shared checkout's ignored `cache/` tree.
- Smoke topic: `rag2026-0` only.

- [ ] **Step 1: Audit source and destination before mutation.**

  Count source retrieval transports, score rows by exact cache context,
  content-addressed documents, and model snapshot files. Verify retrieval
  manifests/gzip hashes, SQLite integrity and logical key consistency,
  document SHA filenames/content, and the pinned model revision. Inventory the
  target and reject any same-key/different-byte collision.

- [ ] **Step 2: Import conflict-safely with source untouched.**

  Publish only validated immutable retrieval transport pairs; logically import
  score rows through `GlobalScoreCache.import_scores`; admit document text
  through `DocumentStore.admit_text(expected_sha256=...)`; and create-only copy
  model files. Exclude locks, SHM files, transient attempts, topic state, and
  partial files. Re-run validation against the target and record counts/hashes
  privately.

- [ ] **Step 3: Create ignored local smoke configs.**

  Copy the canonical agentic config to `configs/local/`, give it a fresh
  one-topic experiment ID/output namespace, and keep all semantic retrieval,
  model, and budget identities unchanged. Prepare a matching ignored RAG config
  only for handoff-load validation; do not invoke hosted RAG generation.

- [ ] **Step 4: Preflight and report before external calls.**

  Verify clean tracked tree, exact committed HEAD, pinned submodules, secrets by
  name only, ROCm access, offline pinned model load, one selected topic, fresh
  output path, and shared cache paths. Report expected retrieval/reranker cache
  reuse versus adaptive-query misses, that coordinator/researcher OpenRouter
  calls remain live, and the private output path.

- [ ] **Step 5: Run the one-topic smoke.**

  ```bash
  .venv/bin/python-rocm -m trec_rag.competition_agentic_retrieval \
    configs/local/rag26_agentic_retrieval_rag2026_0_smoke_20260805.yaml \
    --topic rag2026-0
  ```

  Do not auto-retry if it ends with zero grounded nuggets. If interrupted after
  the run plan exists, resume the same namespace with:

  ```bash
  .venv/bin/python-rocm -m trec_rag.competition_agentic_retrieval \
    configs/local/rag26_agentic_retrieval_rag2026_0_smoke_20260805.yaml \
    --resume --topic rag2026-0
  ```

- [ ] **Step 6: Validate the sealed output.**

  Read the outer manifest last; verify artifact hashes, one-topic cohort,
  variable-depth TREC ordering, deterministic full-text ZIP, no raw-passage
  fallback, grounded handoff closure, RAG loader acceptance, cache hit/miss
  counters, and private permissions. Do not run `competition_rag` hosted
  generation.

- [ ] **Step 7: Use `superpowers:verification-before-completion`, then report
  the exact result, cache reuse, artifact paths, and any remaining limitation.**
