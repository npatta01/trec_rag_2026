# RAG 2025 Distributed Retrieval Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish and consolidate the in-flight `rag2026-0` cache canary, then
run the current fixed/non-agentic competition retrieval system over all 22 RAG
2025 development topics, consolidate every verified cache shard locally, prove
zero-work offline replay, and score retrieval against the pinned projected
development qrels.

**Architecture:** Keep expensive retrieval work isolated by topic on dstack and
publish each completed topic as an immutable private Hugging Face Bucket shard.
Run one numeric 2025 topic as a headless canary before enabling two-topic
parallel batches on explicitly constrained 48 GB GPUs. Each topic finalizes and
publishes in its own subshell so a failed sibling cannot discard completed work.
Download and verify every bundle, merge first into an isolated staging cache and
then into the main checkout's shared cache, and evaluate only local
organizer-style TREC run files; qrels never enter remote workers or cache
bundles.

**Tech Stack:** Python 3.12/3.11, pytest, CUDA/ROCm PyTorch, dstack 0.20.29,
private Hugging Face Buckets, deterministic tar+zstd bundles, SQLite portable
score caches, OpenRouter DeepSeek planning/canonicalization, Pyserini REST over
ClimbMix, projected RAG 2025 development qrels.

## Current State

- Worktree:
  `/home/npatta01/data/competitions/trec_rag_2026/.worktrees/codex-distributed-retrieval-cache`
- Branch: `codex/distributed-retrieval-cache`
- Completed 2026 canary: `rag26-cache-dev-r4`, topic `rag2026-0`, run ID
  `nonagentic-two-topic-20260806`. Its immutable private HF shard was published,
  round-tripped, and locally verified as
  `ea9d2799eaa58b8ca50c38cf4d796e58acb90b4d3923f943fc6a89a6243d9622`.
  The exact A40 run is stopped. The isolated staging merge completed as
  `fd8c9bb8cadf76db35fae62888e612020a9fc279566501461cc24a66c1f75459`;
  its offline replay completed with zero misses, network/provider calls, and
  model batches. Main-cache promotion completed as
  `578e96e278759a730e44273119ac1cf69c65326b3a849520e3f64ee52788cfa5`,
  and a second replay against the promoted shared cache produced the same 203
  ranked rows (apart from the intentional experiment-ID run tag), again with
  every work counter at zero.
- The current canary is the final 2026 topic authorized for this sequence. Do
  not start `rag2026-1`.
- The one-topic RAG25 canary `rag25-cache-31` was submitted once from reviewed
  commit `77e1f536ea7c9a8bffc59569bcdda3d48e57bd07` and completed with exit
  status 0 on an on-demand A40 48 GB instance at $0.44/hour. Final cost was
  $0.2182. The immutable topic-`31` prefix contains exactly the archive and
  completion marker; the locally reverified archive SHA-256 is
  `0d9b77f46ed1e7d339df8d42474c4bf746137ca07e2596979f152fe6fdfe98fe`.
  Its isolated staging merge completed as
  `fed38def987a32f044433c812ed92e1d612ae33dc6bb5fa1c503358c36a2d1bc`;
  zero-work ROCm replay and evaluation are next.
- The 2025 topic file contains exactly 22 numeric IDs:
  `14, 31, 37, 58, 72, 84, 144, 161, 200, 213, 219, 224, 225, 233, 273,
  300, 407, 477, 499, 515, 707, 897`.
- The pinned qrels contain 26,341 pooled ClimbMix judgments across all 22
  topics. They are LLM-generated projected development judgments, not
  exhaustive official TREC ground truth. Every report must use the label
  `projected development qrels` and show judged coverage beside relevance
  metrics.
- The pinned qrels SHA-256 is
  `42bf933ae06eb22213312b22e3f2bc39f3dcc2d54e87ebcd8125e9528ddfcc37`.

## Global Constraints

- Work only in the linked worktree until cache promotion. The main checkout is
  the final shared-cache destination, not a code-editing workspace.
- Never print or persist secret values. dstack must map
  `HF_TOKEN=${{ secrets.hf_token }}` directly.
- Do not start another 2026 topic. Finish, verify, consolidate, and stop the
  current 2026 dev environment before provisioning 2025 compute.
- Keep the retrieval algorithm fixed: bounded one-shot planning, at most 25
  query lanes per topic, ClimbMix `climbmix-400b`, Mixedbread revision
  `3ea9d4dffa7d12a4f366be8e275c349de9fc9865`, and MiniLM revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Topic selectors may use the shared safe-ID contract
  `[A-Za-z0-9][A-Za-z0-9_-]{0,127}`. Do not special-case numeric IDs outside
  that contract.
- Each worker uses an isolated cache root. Each HF topic prefix is immutable,
  must be empty before work, and is complete only when it contains exactly
  `bundle.tar.zst` and `bundle-complete.json`.
- Bundles contain no qrels, gold labels, raw provider responses, credentials,
  model weights, lock files, SQLite WAL/SHM files, or RAG answers.
- Merge only while the local shared cache has no other writer. Verify bundles
  before acquiring the destination merge lock. Never delete an incomplete
  merge journal manually; rerun the identical merge command.
- `--score-conflicts keep-existing` may tolerate only finite numerical
  differences across accelerators. Immutable content conflicts still fail.
- A successful offline replay has zero cache misses, network/provider calls,
  and model batches in every receipt. Do not infer success from runtime alone.
- Before each paid launch, preview the exact configuration, preserve the output,
  require multiple eligible offers when practical, keep `max_price: 1.0`, and
  submit exactly once. Bind each approval to the current Git `HEAD` and SHA-256
  hashes of the retrieval config, shard wrapper, launcher, and dstack template;
  re-preview on any identity drift. Never externally retry `dstack apply`.
- Bulk execution is gated on a successful numeric-topic headless canary and its
  local merge, offline replay, and projected-qrels evaluation. It is also gated
  on an independent code review, a successful two-topic concurrency canary,
  measured cost/call/storage projections, and a new explicit user authorization.
- Do not push, publish publicly, expose a service, or delete HF artifacts.

## File Map

- Create: `configs/rag25_competition_retrieval_v1.yaml` — current non-agentic
  retrieval algorithm bound to the 22 RAG 2025 development topics.
- Modify: `code/trec_rag/competition_retrieval.py` — explicitly close both
  production score-cache connections before a topic worker returns.
- Modify: `code/tests/test_competition_topic_dispatch.py` — score-cache
  lifecycle coverage on successful and failed topic execution.
- Modify: `code/tools/run_retrieval_cache_shard.sh` — accept shared safe topic
  IDs first, then repeated selectors for two-topic parallel batches.
- Modify: `.dstack/rag26-retrieval-cache-shard.yaml` — constrain two-topic
  execution to tested 48 GB GPU classes.
- Modify: `code/tests/test_retrieval_cache_shard_workflow.py` — wrapper/config,
  numeric-topic, repeated-topic, and command-contract coverage.
- Create: `code/trec_rag/competition_retrieval_evaluation.py` — strict
  retrieval-only TREC run parser and projected-qrels evaluator.
- Create: `code/tests/test_competition_retrieval_evaluation.py` — parser,
  scope, provenance, metrics, and CLI tests.
- Modify: `code/trec_rag/README.md` — exact 2025 canary, batch, consolidation,
  replay, and evaluation commands.
- Update this plan after each live gate with run names, commit IDs, HF prefixes,
  bundle hashes, merge receipts, cache-operation receipts, metrics, costs, and
  the next action.

---

### Task 1: Close and Consolidate the In-Flight 2026 Canary

**Files:**

- Read: `/tmp/rag2026-0-retrieval.log` on `rag26-cache-dev-r4`
- Modify: `code/trec_rag/competition_cache_bundle.py`
- Test: `code/tests/test_competition_cache_bundle_offline_replay.py`
- Download to ignored:
  `outputs/private-cache-shards/nonagentic-two-topic-20260806/rag2026-0/`
- Merge into ignored staging roots, then the main checkout's `cache/` and
  `outputs/`
- Update: this plan's Current State and verification record

**Interfaces:**

- Consumes: the active remote wrapper and immutable HF prefix
  `hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/experiments/nonagentic-two-topic-20260806/rag2026-0`.
- Produces: one locally verified archive hash, one staging merge receipt, one
  zero-work offline receipt, and one promoted shared-cache merge receipt.

- [x] **Step 1: Monitor without interrupting the wrapper**

  Every 30–60 seconds, check process state, GPU/CPU activity, score-cache row
  growth, bundle file count, and the log tail. Treat a quiet buffered log as
  healthy when I/O, CPU/GPU, cache rows, or output files advance.

  ```bash
  ssh rag26-cache-dev-r4 \
    'ps -o pid,stat,etime,pcpu,pmem,cmd -p "$(pgrep -P "$(cat /tmp/rag2026-0-retrieval.pid)" | head -1)"'
  ssh rag26-cache-dev-r4 \
    'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader'
  ```

- [x] **Step 2: Diagnose and repair the failed-closed package verification**

  Compare the source canonical manifest, the temporary extracted manifest, and
  the exact config identity accepted by `retrieval_export`. Add a regression
  test reproducing the mismatch before changing code or recovery inputs. The
  repair must preserve the completed retrieval and retain all integrity checks;
  do not edit a sealed checkpoint by hand or bypass fresh-replay verification.

  Add a failing test with a config containing at least two official topics but
  a bundle containing one selected topic. Preserve the exact configured topics
  TSV bytes inside the authenticated bundle/replay boundary, select only the
  requested topic through the normal selector, and require the sealed full-set
  digest to remain unchanged. Run the focused offline-replay and bundle suites,
  `py_compile`, and `git diff --check`, then commit only the repair and test.

  After the commit, record the production module's local SHA-256, copy that
  exact verified file into `/workflow/code/trec_rag/` on the existing dev host,
  and require its remote SHA-256 to match. Do not copy configs, caches, outputs,
  or manifests. Run this exact recovery-finalization shell; it deliberately
  begins at pack and never reruns repository setup, model loading, or retrieval:

  ```bash
  ssh rag26-cache-dev-r4 'bash -lic "bash -s"' <<'RECOVERY'
  set -euo pipefail
  cd /workflow
  python=/workflow/.venv/bin/python
  hf=/workflow/.venv/bin/hf
  topic_id=rag2026-0
  config=/workflow/configs/local/nonagentic-two-topic-20260806-rag2026-0.yaml
  bundle_dir=/tmp/trec-rag-cache-shards/nonagentic-two-topic-20260806/rag2026-0/bundle
  remote_prefix=hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/experiments/nonagentic-two-topic-20260806/rag2026-0
  cache_root=/tmp/trec-rag-cache-shards/nonagentic-two-topic-20260806/rag2026-0/cache
  recovery_root=/tmp/trec-rag-cache-shards/nonagentic-two-topic-20260806/rag2026-0/recovery-v1
  log=/tmp/rag2026-0-recovery.log
  export TREC_RAG_CACHE_ROOT="$cache_root"
  export HF_HOME=/tmp/trec-rag-cache-shards/nonagentic-two-topic-20260806/rag2026-0/huggingface
  mkdir -m 700 "$recovery_root"
  exec > >(tee "$log") 2>&1

  "$python" -m trec_rag.competition_cache_bundle pack \
    --config "$config" --topic "$topic_id" --destination "$bundle_dir"
  verify_output=$("$python" -m trec_rag.competition_cache_bundle verify "$bundle_dir")
  printf '%s\n' "$verify_output"
  archive_sha256=${verify_output#archive_sha256=}
  [[ $archive_sha256 =~ ^[0-9a-f]{64}$ ]]

  "$hf" buckets list "$remote_prefix" --recursive --format json \
    >"$recovery_root/preupload.json"
  "$python" -m trec_rag.hf_bucket_listing require-empty \
    "$recovery_root/preupload.json"

  upload_one() {
    local source=$1
    local basename stage roundtrip
    basename=$(basename "$source")
    stage="$recovery_root/upload-$basename"
    roundtrip="$recovery_root/roundtrip-$basename"
    mkdir -m 700 "$stage"
    cp -- "$source" "$stage/$basename"
    "$hf" buckets sync "$stage" "$remote_prefix" --ignore-existing
    "$hf" buckets cp "$remote_prefix/$basename" "$roundtrip"
    cmp -s -- "$source" "$roundtrip"
  }

  upload_one "$bundle_dir/bundle.tar.zst"
  upload_one "$bundle_dir/bundle-complete.json"
  "$hf" buckets list "$remote_prefix" --recursive --format json \
    >"$recovery_root/after.json"
  "$python" -m trec_rag.hf_bucket_listing require-bundle \
    "$recovery_root/after.json"

  mkdir -m 700 "$recovery_root/downloaded-bundle"
  "$hf" buckets cp "$remote_prefix/bundle.tar.zst" \
    "$recovery_root/downloaded-bundle/bundle.tar.zst"
  "$hf" buckets cp "$remote_prefix/bundle-complete.json" \
    "$recovery_root/downloaded-bundle/bundle-complete.json"
  cmp -s -- "$bundle_dir/bundle.tar.zst" \
    "$recovery_root/downloaded-bundle/bundle.tar.zst"
  cmp -s -- "$bundle_dir/bundle-complete.json" \
    "$recovery_root/downloaded-bundle/bundle-complete.json"
  "$python" -m trec_rag.competition_cache_bundle verify \
    "$recovery_root/downloaded-bundle"

  printf '%s\n' \
    'recovery_status=complete' \
    "topic_id=$topic_id" \
    "archive_sha256=$archive_sha256" \
    "remote_prefix=$remote_prefix"
  RECOVERY
  ```

  The recovery shell's zero exit plus its final marker is the separate recovery
  receipt. A partial upload remains failed closed and requires diagnosis; never
  delete or overwrite an immutable remote object.

  Evidence: commit `a14b80b` fixed full official-topic identity; independent
  release review found no blocking issue; commits `928c0af` and its tests cover
  legacy singleton compatibility and prevent source-identity merge leakage.
  The synced production module SHA-256 was
  `c7acea74066c8f00a1f8b4938713ebe7d6d4e423f6b9694fe4ed3f374ae3f0e3`.

- [x] **Step 3: Require recovery publication success**

  The original wrapper remains failed and is not a success signal. Do not
  consolidate until the recovery-finalization shell exits zero and its recovery
  log ends with:

  ```text
  recovery_status=complete
  topic_id=rag2026-0
  archive_sha256=<verified digest>
  remote_prefix=hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/experiments/nonagentic-two-topic-20260806/rag2026-0
  ```

  Independently list the prefix and require exactly the archive and completion
  marker.

  Evidence: recovery v2 exited zero after pack, local verify, immutable
  upload-marker-last, byte round trips, exact listing validation, download, and
  downloaded-bundle replay. Archive SHA-256:
  `ea9d2799eaa58b8ca50c38cf4d796e58acb90b4d3923f943fc6a89a6243d9622`.

- [x] **Step 4: Download and verify locally**

  ```bash
  cd /home/npatta01/data/competitions/trec_rag_2026/.worktrees/codex-distributed-retrieval-cache
  shard_root="$PWD/outputs/private-cache-shards/nonagentic-two-topic-20260806"
  mkdir -p "$shard_root/rag2026-0"
  .venv/bin/hf buckets sync \
    hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/experiments/nonagentic-two-topic-20260806/rag2026-0 \
    "$shard_root/rag2026-0"
  .venv/bin/python -m trec_rag.competition_cache_bundle verify \
    "$shard_root/rag2026-0"
  ```

  Record the verifier's `archive_sha256`.

- [x] **Step 5: Stop the exact paid GPU after local verification**

  As soon as the recovery-finalization shell exits zero and the locally
  downloaded bundle verifies, stop only `rag26-cache-dev-r4` and confirm it is
  no longer active. Do this before staging merge or replay so a local failure
  cannot extend GPU billing. Do not start `rag2026-1`.

  ```bash
  dstack stop rag26-cache-dev-r4 -y
  dstack ps -v
  ```

- [x] **Step 6: Merge into a fresh staging root**

  ```bash
  stage_root="$PWD/outputs/private-cache-staging/nonagentic-two-topic-20260806"
  .venv/bin/python -m trec_rag.competition_cache_bundle merge \
    --cache-root "$stage_root/cache" \
    --outputs-root "$stage_root/outputs" \
    --score-conflicts keep-existing \
    "$shard_root/rag2026-0"
  ```

  Evidence: merge ID
  `fd8c9bb8cadf76db35fae62888e612020a9fc279566501461cc24a66c1f75459`;
  the completion receipt is present under the isolated staging cache.

- [x] **Step 7: Prove zero-work staging replay**

  Create an ignored `configs/local/` copy of the canonical retrieval config
  with experiment ID `nonagentic-rag2026-0-local-replay-20260806`. Keep the real
  inherited `INDEX_URL`, poison the other credentials, select only
  `rag2026-0`, and require an offline-cache-only success receipt.

  ```bash
  OPENROUTER_API_KEY=offline-disabled \
  PYSERINI_API_TOKEN=offline-disabled \
  HF_TOKEN=offline-disabled \
  TREC_RAG_CACHE_ROOT="$stage_root/cache" \
  .venv/bin/python-rocm -m trec_rag.competition_retrieval \
    configs/local/nonagentic-rag2026-0-local-replay-20260806.yaml \
    --offline-cache-only --topic rag2026-0
  ```

  Evidence: replay completed in 257.12 seconds with maximum RSS 9,142,696 KB.
  Cache-hit counts were planning 1, retrieval 9, passage scores 52,448,
  sentence scores 19,863, similarity 8, and canonicalization 8. Every stage
  recorded zero misses, network calls, provider calls, and model batches.

- [x] **Step 8: Promote to the main shared cache**

  ```bash
  main_checkout=/home/npatta01/data/competitions/trec_rag_2026
  .venv/bin/python -m trec_rag.competition_cache_bundle merge \
    --cache-root "$main_checkout/cache" \
    --outputs-root "$main_checkout/outputs" \
    --score-conflicts keep-existing \
    "$shard_root/rag2026-0"
  ```

  Verify the promoted cache with the same offline-only receipt checks.

  Evidence: shared merge ID
  `578e96e278759a730e44273119ac1cf69c65326b3a849520e3f64ee52788cfa5`.
  The promoted-cache replay completed in 257.47 seconds with maximum RSS
  9,143,480 KB and the same per-stage cache hits and zero-work counters. Its
  203 organizer run rows match the staging replay in every field except the
  deliberate experiment-ID run tag.

### Task 2: Close Production Score Caches Before Packaging

**Files:**

- Modify: `code/trec_rag/competition_retrieval.py`
- Test: `code/tests/test_competition_topic_dispatch.py`

**Interfaces:**

- Consumes: the two `GlobalScoreCache` instances owned by
  `MixedbreadPassageScorer` and `MixedbreadSentencePairScorer` inside
  `_run_production_topic_job`.
- Produces: a topic-worker lifecycle guarantee that both SQLite connections
  close on success and failure before `dispatch_topics` returns to bundle
  packaging.

- [x] **Step 1: Add failing lifecycle tests**

  Use fake production scorers whose `score_cache.close()` appends a unique
  marker. Exercise one successful `_run_topic` and one raising `_run_topic`.
  Assert both passage and sentence caches close exactly once in both cases.

  ```python
  class RecordingCache:
      def __init__(self, name: str, closed: list[str]) -> None:
          self.name = name
          self.closed = closed

      def close(self) -> None:
          self.closed.append(self.name)

  assert sorted(closed) == ["passage", "sentence"]
  ```

- [x] **Step 2: Run the focused tests and confirm RED**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_competition_topic_dispatch.py \
    -k 'closes_score_caches' -q
  ```

  Historical RED: neither cache was closed by the production worker before
  commit `189c001`.

- [x] **Step 3: Implement exception-safe cleanup**

  Register each cache's `close` method immediately after its scorer is
  constructed, using `contextlib.ExitStack` around production dependency
  construction and topic execution:

  ```python
  with ExitStack() as score_cache_stack:
      passage_scorer = MixedbreadPassageScorer(...)
      score_cache_stack.callback(passage_scorer.score_cache.close)
      candidate_scorer = MixedbreadSentencePairScorer(...)
      score_cache_stack.callback(candidate_scorer.score_cache.close)
      # Build dependencies, run the topic, validate the outcome, and return.
  ```

  `ExitStack` must attempt both callbacks even if topic execution or one close
  fails. Do not weaken the bundle packer's WAL/SHM rejection.

- [x] **Step 4: Verify lifecycle and bundle regressions, then commit**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_competition_topic_dispatch.py \
    code/tests/test_competition_cache_bundle.py \
    code/tests/test_competition_cache_bundle_offline_replay.py -q
  git diff --check
  git add code/trec_rag/competition_retrieval.py \
    code/tests/test_competition_topic_dispatch.py
  git commit -m "Close topic score caches before shard packaging"
  ```

  Evidence: commit `189c001`; 132 tests passed across topic dispatch, bundle,
  and offline-replay suites; `git diff --check` passed.

- [x] **Step 5: Cover cleanup failures missed by the first patch**

  Add tests proving that `ExitStack` attempts the other callback when one
  `close()` raises, and that the first cache closes if construction of the
  second scorer fails. Require exactly-once cleanup and rerun the Task 2 suites.

  Evidence: commit `33f2d9c`; 136 topic-dispatch, bundle, and offline-replay
  tests passed; `git diff --check` passed.

### Task 3: Bind the Current Retrieval Algorithm to RAG 2025 Topics

**Files:**

- Create: `configs/rag25_competition_retrieval_v1.yaml`
- Modify: `code/tools/run_retrieval_cache_shard.sh`
- Test: `code/tests/test_retrieval_cache_shard_workflow.py`

**Interfaces:**

- Consumes: `FacetPilotConfig`, the existing 2026 fixed retrieval config, and
  `_SAFE_ID = [A-Za-z0-9][A-Za-z0-9_-]{0,127}` from topic dispatch.
- Produces: a tracked `FacetPilotConfig` containing all 22 numeric topics and a
  one-topic shard wrapper accepting `--topic 31` without weakening path safety.

- [x] **Step 1: Add failing numeric-topic and config tests**

  Add assertions equivalent to:

  ```python
  def test_rag25_competition_config_selects_all_22_dev_topics() -> None:
      loaded = load_facet_pilot_config(ROOT / "configs/rag25_competition_retrieval_v1.yaml")
      topics = select_configured_topics(loaded)
      assert [topic.id for topic in topics] == [
          "14", "31", "37", "58", "72", "84", "144", "161", "200",
          "213", "219", "224", "225", "233", "273", "300", "407",
          "477", "499", "515", "707", "897",
      ]

  def test_wrapper_preflight_accepts_numeric_rag25_topic() -> None:
      result = run_wrapper_preflight(
          "--topic", "31", "--run-id", "nonagentic-rag25-dev-20260806",
          "--config", "configs/rag25_competition_retrieval_v1.yaml",
      )
      assert result.returncode == 0
      assert "topic_id=31" in result.stdout
  ```

  Add an automated normalized-YAML parity test against
  `configs/rag26_competition_retrieval_v2.yaml`. Remove only
  `experiment.id` and `topics.path` from both loaded dictionaries, then require
  the remaining structures to be exactly equal. This makes algorithm parity a
  test invariant rather than a manual shell check.

- [x] **Step 2: Run the focused tests and confirm RED**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_retrieval_cache_shard_workflow.py \
    -k 'rag25 or numeric' -q
  ```

  Expected: failure because the config is absent and the wrapper rejects `31`.

- [x] **Step 3: Add the tracked RAG 2025 competition config**

  Copy the exact retrieval, passage, and nugget settings from
  `configs/rag26_competition_retrieval_v2.yaml`, changing only:

  ```yaml
  experiment:
    id: facet-deepseek-rag25-dev-v1

  topics:
    path: trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv

  execution:
    topic_workers: 2
  ```

  Keep `index: climbmix-400b`, the exact corpus epoch, CUDA device identity,
  passage settings, budgets, and model identities byte-for-byte aligned with
  the 2026 config.

- [x] **Step 4: Generalize topic validation at the wrapper boundary**

  Replace the year-specific shell match with the shared safe-ID contract:

  ```bash
  [[ $topic_id =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]] \
    || die "--topic must be a safe configured topic ID"
  ```

  `validate_configured_topic` remains the semantic gate, so a safe but absent
  numeric ID still fails before authentication, model download, or hosted work.

- [x] **Step 5: Verify GREEN and regressions, then commit**

  ```bash
  .venv/bin/python -m pytest code/tests/test_retrieval_cache_shard_workflow.py -q
  git diff --check
  git add configs/rag25_competition_retrieval_v1.yaml \
    code/tools/run_retrieval_cache_shard.sh \
    code/tests/test_retrieval_cache_shard_workflow.py
  git commit -m "Support numeric development topics in cache shards"
  ```

  Evidence: commit `14e128d`; 25 workflow tests passed, the normalized config
  parity mutation was caught at RED and restored, and `git diff --check` passed.

### Task 4: Add Strict Projected-Qrels Evaluation for Competition Runs

**Files:**

- Create: `code/trec_rag/competition_retrieval_evaluation.py`
- Create: `code/tests/test_competition_retrieval_evaluation.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**

- Produces:

  ```python
  def parse_trec_retrieval_run(
      path: Path,
      *,
      expected_topic_ids: Collection[str],
  ) -> list[RankedCandidate]: ...

  def evaluate_competition_retrieval_run(
      run_path: Path,
      qrels_path: Path,
      *,
      topic_ids: Collection[str],
      metric_names: Sequence[str],
      relevance_threshold: int = 2,
  ) -> dict[str, object]: ...
  ```

  The CLI accepts `--run`, `--qrels`, repeated `--topic`, and `--output`.
  Output is canonical JSON containing schema version, explicit
  `projected_development_qrels: true`, exact topic population, run/qrels
  SHA-256 provenance, relevance threshold, aggregate metrics, and per-topic
  metrics.

  Production evaluation accepts only the pinned assessor variant and qrels
  SHA-256. A narrow fixture-only seam may inject a test digest, but the CLI must
  never relabel arbitrary qrels as the pinned diagnostic.

- [x] **Step 1: Write failing parser and evaluator tests**

  Cover a valid six-column TREC run and rejection of malformed columns,
  non-`Q0`, empty IDs, invalid/nonfinite scores, ranks that do not begin at 1
  and remain dense in file order, duplicate ranks, duplicate doc IDs,
  increasing scores, conflicting run tags, missing expected topics, and extra
  topics. Validate qrels column 2 is `0`, grades are integers 0–4, topic scope
  is exactly the 22 configured IDs, topic/doc pairs are unique, and the digest
  equals the pinned SHA-256. Assert exact `ndcg@10`, `precision@10`,
  `recall@100`, `ideal_dcg_coverage@50`, and `judged_count`/`judged_rate` at
  each cutoff 10, 50, and 100 on a tiny fixture.

  ```python
  def test_evaluation_binds_scope_and_projected_qrels_provenance(tmp_path: Path) -> None:
      result = evaluate_competition_retrieval_run(
          run_path,
          qrels_path,
          topic_ids=("31",),
          metric_names=("ndcg@10", "judged_rate@10", "recall@100"),
          relevance_threshold=2,
      )
      assert result["projected_development_qrels"] is True
      assert result["topic_ids"] == ["31"]
      assert result["run"]["sha256"] == sha256(run_path.read_bytes()).hexdigest()
      assert result["qrels"]["sha256"] == sha256(qrels_path.read_bytes()).hexdigest()
  ```

- [x] **Step 2: Run the focused tests and confirm RED**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_competition_retrieval_evaluation.py -q
  ```

- [x] **Step 3: Implement strict parsing and reuse the metric engine**

  Parse each run row into:

  ```python
  RankedCandidate(
      topic_id=topic_id,
      docid=docid,
      rank=rank,
      score=score,
      text="",
      provenance=[{"run_id": run_id}],
  )
  ```

  After structural validation, call `parse_qrels` and `evaluate_ranked` from
  `trec_rag.evaluation`. Write JSON atomically with a trailing newline. Never
  copy qrels into cache roots, bundle roots, or dstack transport artifacts.
  Record the assessor variant, canonical repo-relative path, pinned digest, and
  the label `projected development qrels` in every output.

- [x] **Step 4: Verify the CLI and commit**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_competition_retrieval_evaluation.py \
    code/tests/test_pipeline.py -q
  git diff --check
  git add code/trec_rag/competition_retrieval_evaluation.py \
    code/tests/test_competition_retrieval_evaluation.py \
    code/trec_rag/README.md
  git commit -m "Evaluate competition retrieval on projected dev qrels"
  ```

  Evidence: commit `4e7b836`; 115 evaluator/shared pipeline tests passed,
  `py_compile` passed, and `git diff --check` passed.

- [x] **Step 5: Independent pre-spend review gate**

  Independently review the bundle self-verification repair, lifecycle fix,
  2025 config/wrapper, evaluator, and focused tests. Resolve all correctness
  and safety findings and rerun the affected suites before previewing topic
  `31`. Record reviewed commit IDs in this plan.

  Evidence: the evaluator/bundle/config review resolved all Important findings
  and approved reviewed commit `77e1f53` for the paid one-topic canary. The
  relevant regression population was 298 passing tests before launch.

### Task 5: Run, Consolidate, Replay, and Evaluate One Headless 2025 Canary

**Files:**

- Use: `.dstack/rag26-retrieval-cache-shard.yaml`
- Use: `code/tools/apply_retrieval_cache_shard.sh`
- Download to ignored:
  `outputs/private-cache-shards/nonagentic-rag25-dev-20260806/31/`
- Write ignored evaluation JSON under
  `outputs/private-cache-evaluation/nonagentic-rag25-dev-20260806/`
- Update: this plan's verification record

**Interfaces:**

- Consumes: topic `31`, the tracked RAG 2025 competition config, and the
  projected qrels.
- Produces: one verified numeric-topic HF shard, local zero-work replay, and
  topic-31 projected-qrels metrics with judged coverage.

- [x] **Step 1: Run local and wrapper preflights**

  ```bash
  bash code/tools/run_retrieval_cache_shard.sh --preflight \
    --topic 31 \
    --run-id nonagentic-rag25-dev-20260806 \
    --config configs/rag25_competition_retrieval_v1.yaml
  ```

  Confirm all required secrets are present by name only and confirm the HF
  prefix for topic `31` is empty.

  Evidence: all four dstack secret names were present and the exact private HF
  prefix was empty.

- [x] **Step 2: Preview without submitting**

  ```bash
  code/tools/apply_retrieval_cache_shard.sh --preview \
    --name rag25-cache-31 \
    -- --topic 31 \
    --run-id nonagentic-rag25-dev-20260806 \
    --config configs/rag25_competition_retrieval_v1.yaml
  ```

  Record offers unchanged. Report the one selected topic, empty cache/HF
  prefix, expected planning/Pyserini/model/canonicalization misses, output
  roots, maximum price, and duration before launch.

  Record Git `HEAD` plus SHA-256 hashes for the config, wrapper, launcher, and
  dstack template. Immediately before launch, recompute and compare them; any
  drift invalidates approval and requires a new preview.

  Evidence: seven compliant offers were returned; the selected RunPod A40
  48 GB offer was $0.44/hour. HEAD and all four recorded SHA-256 values matched
  immediately before submission.

- [x] **Step 3: Bind the existing canary authorization to the preview**

  Preserve the actual backend, GPU, price, duration limit, Git `HEAD`, and four
  source hashes. The user's existing instruction to run a 2025 topic authorizes
  exactly this one-topic `31` canary only when that recorded offer remains at or
  below `$1/hour`, uses an allowed GPU, and all hashes remain identical. Record
  that authorization reference after the preview. Any resource change or hash
  drift requires a new preview and explicit decision.

  Evidence: the existing one-topic authorization was bound to topic `31`, the
  $0.44/hour A40 offer, five-hour cap, and reviewed commit `77e1f53`.

- [x] **Step 4: Submit once and monitor headlessly**

  ```bash
  code/tools/apply_retrieval_cache_shard.sh --launch \
    --name rag25-cache-31 \
    -- --topic 31 \
    --run-id nonagentic-rag25-dev-20260806 \
    --config configs/rag25_competition_retrieval_v1.yaml
  ```

  Monitor `dstack ps -v` and ordinary logs. Do not use diagnostic logs that
  serialize environment values. Require the wrapper's remote round-trip
  verification and `shard_status=complete`.

  Evidence: `rag25-cache-31` completed with exit status 0,
  `shard_status=complete`, and cost $0.2182. Remote pack, semantic verify,
  archive-then-marker publication, byte round trips, and downloaded-bundle
  verification all succeeded.

- [ ] **Step 5: Download, verify, stage-merge, and replay**

  Use the Task 1 sequence with run ID `nonagentic-rag25-dev-20260806` and topic
  `31`. The replay config uses a new experiment ID and selects only `31` with
  `.venv/bin/python-rocm ... --offline-cache-only`. Inspect both the topic
  receipt and root manifest and require zero misses/calls/model batches.

  Progress: the exact two-object listing, fresh local download, and semantic
  verification passed with archive SHA-256
  `0d9b77f46ed1e7d339df8d42474c4bf746137ca07e2596979f152fe6fdfe98fe`.
  Fresh staging merge ID:
  `fed38def987a32f044433c812ed92e1d612ae33dc6bb5fa1c503358c36a2d1bc`.
  Replay remains the next gate.

- [ ] **Step 6: Evaluate the local replay**

  ```bash
  .venv/bin/python -m trec_rag.competition_retrieval_evaluation \
    --run outputs/nonagentic-rag25-dev-local-replay-20260806/r_output_trec_rag_2026.tsv \
    --qrels trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels \
    --topic 31 \
    --output outputs/private-cache-evaluation/nonagentic-rag25-dev-20260806/topic-31.json
  ```

  Require a nonzero judged count and report judged rate with every relevance
  cutoff at 10, 50, and 100. Promote the verified bundle into the main shared
  cache only after this gate passes. Record elapsed time, actual dstack cost,
  hosted-call counts, compressed/uncompressed bundle sizes, and peak local disk
  use for the later bulk gate.

### Task 6: Add Two-Topic Parallel Shards for the Remaining Development Set

**Files:**

- Modify: `code/tools/run_retrieval_cache_shard.sh`
- Modify: `code/tests/test_retrieval_cache_shard_workflow.py`
- Modify: `code/trec_rag/README.md`
- Modify: `.dstack/rag26-retrieval-cache-shard.yaml`

**Interfaces:**

- Consumes: repeated `--topic SAFE_ID` selectors.
- Produces: shared repository/environment/model setup once per machine, then
  one isolated one-worker config/cache/output/work root and one independent
  retrieval -> pack -> verify -> upload -> round-trip subshell per topic.

- [x] **Step 1: Add failing repeated-topic contract tests**

  Test two unique topics, rejection of duplicates, rejection of more than two
  topics, isolated per-topic roots/configs, parallel subshells, per-topic
  pack/verify/upload commands, and distinct HF prefixes. Inject one topic
  failure and prove its successful sibling still publishes exactly
  `bundle.tar.zst` and `bundle-complete.json`; the overall wrapper must return
  nonzero. Preserve the existing one-topic behavior.

  ```python
  def test_wrapper_preflight_supports_two_unique_topics() -> None:
      result = run_wrapper_preflight(
          "--topic", "14", "--topic", "37",
          "--run-id", "nonagentic-rag25-dev-20260806",
          "--config", "configs/rag25_competition_retrieval_v1.yaml",
      )
      assert result.returncode == 0
      assert "topic_ids=14,37" in result.stdout
      assert "parallel_topic_processes=2" in result.stdout
  ```

- [x] **Step 2: Run tests and confirm RED**

  ```bash
  .venv/bin/python -m pytest \
    code/tests/test_retrieval_cache_shard_workflow.py \
    -k 'two_unique_topics or duplicate_topic or numeric' -q
  ```

- [x] **Step 3: Implement independent bounded topic subshells**

  Store selectors in `topic_ids=()`, validate every ID with the shared safe
  pattern, reject duplicates, and require one or two topics. Perform checkout,
  environment setup, model prefetch, and immutable-prefix preflight once.
  Generate a separate config and isolated cache/output/work root for each topic,
  forcing `execution.topic_workers: 1`. Launch one subshell per topic in the
  background. Each subshell runs retrieval, pack, local verify, uploads archive
  then marker, downloads, compares, and re-verifies without depending on its
  sibling. Collect both exit codes explicitly; successful siblings remain
  published while any failure makes the overall task nonzero.

- [x] **Step 4: Enforce the two-topic resource envelope**

  Update the dstack task and its rendering tests so batch execution accepts
  only tested 48 GB classes (`A40`, `A6000`, or `L40S`) with at least 48 GB GPU
  memory. Retain `max_price: 1.0` and the existing duration cap. A declined
  preview must contain only compliant offers; never silently fall back to a
  24 GB GPU for two model processes.

- [x] **Step 5: Verify full workflow tests and commit**

  ```bash
  .venv/bin/python -m pytest code/tests/test_retrieval_cache_shard_workflow.py -q
  git diff --check
  git add code/tools/run_retrieval_cache_shard.sh \
    code/tests/test_retrieval_cache_shard_workflow.py \
    code/trec_rag/README.md \
    .dstack/rag26-retrieval-cache-shard.yaml
  git commit -m "Run two retrieval topics per cache shard host"
  ```

  Evidence: commit `19ac67c`; 44 workflow tests passed. Both shell syntax
  checks and `git diff --check` passed.

- [x] **Step 6: Independent pre-concurrency review gate**

  Independently review the parallel wrapper, failure isolation, resource
  contract, and tests. Resolve findings and rerun the workflow suite before any
  paid two-topic launch.

  Evidence: review found the Hugging Face CLI's lexical `14`/`144` prefix
  collision. The final implementation lists the run parent and validates exact
  path components, with initial/preupload/final/concurrent regressions. Re-review
  found no Critical or Important issues; its final documentation-only finding
  was resolved before commit. No two-topic paid run has been authorized or
  launched.

### Task 7: Execute the Remaining 21 Topics in Bounded Headless Batches

**Files:**

- No tracked production changes after Task 6
- Download ignored bundles under
  `outputs/private-cache-shards/nonagentic-rag25-dev-20260806/<topic>/`
- Update: this plan's live-run table

**Interfaces:**

- Consumes the remaining topic IDs after `31`:
  `14, 37, 58, 72, 84, 144, 161, 200, 213, 219, 224, 225, 233, 273, 300,
  407, 477, 499, 515, 707, 897`.
- Produces ten two-topic tasks and one singleton task, each capped at one 48 GB
  GPU, two isolated one-worker topic processes, `$1/hour`, and five hours.

- [ ] **Step 1: Build and validate the exact batch manifest**

  Use these deterministic batches:

  ```text
  14,37
  58,72
  84,144
  161,200
  213,219
  224,225
  233,273
  300,407
  477,499
  515,707
  897
  ```

  Before provisioning, assert the union with completed topic `31` equals the
  exact 22-topic TSV set with no duplicates or omissions.

- [ ] **Step 2: Build the measured cost, call, and capacity gate**

  From topic `31`, record observed elapsed time, dstack price and actual cost,
  planning/Pyserini/model/canonicalization call counts, and compressed and
  uncompressed bundle sizes. Project expected and worst-case compute for the
  remaining manifest (the configuration ceiling is 11 tasks x 5 hours x
  $1/hour = $55, plus hosted calls). Project disk required while downloads,
  aggregate staging, and the main shared cache coexist, then check free space
  with `df`. Report topic list, immutable HF prefixes, expected/worst compute,
  hosted-call estimate, and storage requirement to the user. No
  remaining-topic launch is authorized by this calculation alone.

- [ ] **Step 3: Preview every distinct task configuration**

  Preserve each declined preview. Use safe names such as
  `rag25-cache-14-37`. Do not broaden GPU compatibility, price, providers, or
  spot policy after preview without a new explicit decision. Record and bind
  every preview to Git/config/wrapper/launcher/template hashes; re-preview on
  drift.

- [ ] **Step 4: Obtain offer- and hash-bound concurrency-canary authorization**

  Present the preserved `rag25-cache-14-37` offer, source hashes, observed
  topic-31 measurements, expected cost, and five-hour/$1-hour ceiling. Record
  explicit user authorization for this exact launch after the preview. A market
  or source change invalidates it and requires re-preview and re-authorization.

- [ ] **Step 5: Run exactly one two-topic concurrency canary**

  Launch only `rag25-cache-14-37`. Record host RAM, VRAM, cache growth, both
  process states, cost, and both independent publications. Download, verify,
  stage-merge, and offline-replay topics `14` and `37`. Do not launch a second
  pair until both topics pass and the evidence is recorded.

- [ ] **Step 6: Obtain final measured bulk authorization**

  Incorporate the two-topic canary's actual runtime, price, hosted calls,
  RAM/VRAM, and storage into the remaining-19-topic projection. Re-preview any
  expired or changed offer, present exact run names/topic prefixes/source
  hashes plus expected and worst-case remaining cost, and record explicit user
  authorization after those previews. No later wave may launch under stale
  hashes or an unapproved offer.

- [ ] **Step 7: Launch the remaining manifest in bounded waves**

  Start at most three tasks concurrently. A machine processes at most two
  topics in parallel. Submit each task exactly once and let dstack handle only
  configured native capacity behavior. Record run name, backend, GPU, price,
  topic IDs, submission time, and status.

- [ ] **Step 8: Consolidate each completed wave**

  For every topic prefix: list, download, verify, record archive hash, merge
  into the fresh aggregate staging cache, and run offline replay for the newly
  added topics before starting the next wave. This bounds failure recovery and
  prevents discovering a bad shard after all GPU spend has completed.

- [ ] **Step 9: Handle failures without corrupting successful work**

  A failed task gets root-cause diagnosis before any retry. Reuse already
  completed immutable topic prefixes; retry only topics whose prefixes remain
  empty. Never overwrite or delete a completion marker.

### Task 8: Final Consolidation, Offline Replay, and 22-Topic Evaluation

**Files:**

- Merge all 22 ignored bundle directories
- Write ignored offline replay outputs
- Write ignored evaluation JSON
- Update: this plan with final evidence

**Interfaces:**

- Consumes: exactly 22 locally verified bundles and the pinned projected qrels.
- Produces: a shared local cache, a fresh all-topic zero-work replay, aggregate
  and per-topic development metrics, and an auditable completion record.

- [ ] **Step 1: Verify the complete local bundle inventory**

  Require exactly one directory and one recorded archive hash for every topic
  ID. Re-run `competition_cache_bundle verify` on all 22 directories before
  promotion.

- [ ] **Step 2: Merge all bundles into a fresh aggregate staging root**

  Invoke one merger command with the 22 bundle directories in numeric topic
  order and `--score-conflicts keep-existing`. Require no incomplete journal
  after completion.

- [ ] **Step 3: Run a credential-poisoned 22-topic offline replay**

  Use a new experiment ID, the tracked 2025 competition config, repeated
  selectors for all 22 topics, and
  `.venv/bin/python-rocm ... --offline-cache-only`. Require:

  ```text
  selected_topic_count=22
  cache_misses=0 at every stage
  network_calls=0
  provider_calls=0
  model_batches=0
  ```

  Validate all retrieval outputs and the manifest-last handoff receipt.

- [ ] **Step 4: Evaluate and compare**

  Run `competition_retrieval_evaluation` for all 22 topics. Report aggregate
  and per-topic metrics with judged counts/rates at 10, 50, and 100. Compare
  only if an existing immutable baseline run is located and its bytes and
  SHA-256 are recorded. If no such run exists, treat creating the
  `configs/rag25_bm25_full_query_v1.yaml` baseline as a separate live,
  cache-accounted action requiring its own authorization. Label any comparison
  as projected-development diagnostics, not official TREC results.

- [ ] **Step 5: Promote once and verify instant shared-cache replay**

  With all local cache writers stopped, merge the same verified bundles into:

  ```text
  /home/npatta01/data/competitions/trec_rag_2026/cache
  /home/npatta01/data/competitions/trec_rag_2026/outputs
  ```

  Rerun one representative topic and then all 22 with
  `.venv/bin/python-rocm ... --offline-cache-only` against the shared cache.
  Record wall-clock time and zero-work receipts.

- [ ] **Step 6: Final verification and cleanup**

  Stop only exact run names recorded in this plan's live/batch manifest, verify
  each is inactive, and list any unexpected active run without mutating it.
  Retain private HF bundles and local merge receipts, and report total cost,
  duration, topic coverage, bundle hashes, conflicts, replay timing, and
  metrics. Do not delete remote artifacts.

## Live Run Table

| Phase | Run name | Topics | Status | HF verification | Local merge | Offline replay | Evaluation |
|---|---|---:|---|---|---|---|---|
| 2026 closure | `rag26-cache-dev-r4` | `rag2026-0` | stopped | archive + marker round-trip verified | staging merge complete | pending | n/a |
| 2025 canary | `rag25-cache-31` | `31` | gated | pending | pending | pending | pending |

## Completion Criteria

- No `rag2026-1` or other additional 2026 topic was launched.
- Topic `rag2026-0` exists in the shared local cache and passes zero-work replay.
- All 22 numeric 2025 topics have immutable HF completion markers and locally
  verified archive hashes.
- All 22 merge idempotently into both staging and shared caches.
- A fresh all-topic offline replay performs no external or model work.
- The projected-qrels report binds exact run/qrels hashes, topic population,
  pinned digest and assessor identity, relevance threshold, judged coverage at
  10/50/100, aggregate metrics, and per-topic metrics.
- Every dstack run recorded in this plan is inactive after completion;
  unrelated runs are not stopped.
- Focused tests, full workflow tests, `git diff --check`, and an independent
  consequential-change review pass before the topic-31 spend, before the
  two-topic spend, and before branch handoff.
