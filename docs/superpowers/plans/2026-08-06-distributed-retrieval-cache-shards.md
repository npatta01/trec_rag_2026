# Distributed Retrieval Cache Shards Implementation Plan

> **For Codex:** Use `superpowers:test-driven-development` for every production
> behavior, delegate independent bounded streams when useful, use an independent
> review for the integrated branch, and apply
> `superpowers:verification-before-completion` before every completion claim.

**Goal:** Run fixed/non-agentic competition retrieval topics independently on
multiple dstack machines, archive each topic's complete reusable cache as an
immutable private Hugging Face Bucket shard, merge shards safely into a local
cache, and prove that a new local experiment can replay those topics without
network requests, hosted-model calls, or local model batches.

**Architecture:** Expensive work is cached by its true deterministic input and
model/service identity, never by experiment ID, output directory, worker, or
accelerator. Each worker writes to an isolated cache root, packages only
validated portable artifacts, and publishes an archive before a completion
marker. The local merger verifies the archive, stages all immutable files,
imports score rows transactionally, and records a durable completion receipt.
The existing fixed retrieval runner gains a fail-closed cache-only mode and
per-stage accounting so a zero-work replay is machine-verifiable.

**Tech stack:** Python 3.12, pytest, SQLite, deterministic JSON/JSONL,
`zstandard==0.25.0`, dstack 0.20.x, private Hugging Face Buckets, ROCm/CUDA
PyTorch, Mixedbread reranker, MiniLM similarity, OpenRouter DeepSeek planning
and canonicalization, Pyserini REST retrieval.

## Current status (2026-08-06)

- Isolated linked worktree created from refreshed `origin/master` commit
  `55608f40d34e815fdc5e2a2160e66709dc471a34` at
  `.worktrees/codex-distributed-retrieval-cache`.
- Submodules are initialized at the commits recorded by the superproject.
- The ignored `.env` was copied from the shared checkout without printing it.
- Environment setup verified ROCm GPU access.
- Clean baseline: 2,266 tests passed and 19 skipped.
- Tasks 1–6 are implemented on branch `codex/distributed-retrieval-cache`.
  Cache identities, read-only replay, authenticated deterministic bundles,
  transactional consolidation, the committed-only dstack launcher, and the
  private-HF wrapper all have focused regression coverage.
- The integrated whole-branch review found three important gaps: bundle success
  did not prove a complete offline replay, concurrent score merges could publish
  while rows remained in a WAL, and hostile portable-score members could exceed
  safe parser memory. All three are fixed locally with real-chain, mid-flight
  multiprocessing serialization, and bounded-input regressions. A subsequent
  review also required the verifier to exercise the production staged offline
  runner, verification to precede destination locking, same-thread lock
  re-entry to fail instead of deadlock, and TSV-safe recovered narratives;
  those regressions are now covered and the final follow-up review is running.
- A real one-topic fixture now completes online planning/retrieval/reranking/
  similarity/canonicalization, packs and verifies its shard, merges it, and
  replays from a fresh output namespace with poisoned transports/model loaders.
  Its authenticated operation receipt reports zero misses, network/provider
  calls, and model batches. Removing any required cache namespace, or using an
  uncached planning/canonical fallback, prevents bundle publication.
- Current focused verification includes 159 bundle/offline/retrieval tests and
  21 dstack/HF workflow tests, with clean Ruff formatting/static checks. The
  latest complete run passed 2,478 tests, 19 skips, and 60 subtests. `uv lock
  --check` passed. The optional torch-group verifier made no progress while
  resolving external package metadata and was interrupted after two minutes;
  it previously reached the same transient `repo.radeon.com` availability
  boundary. The full suite will run once more from the committed clean branch.
- Both exact topic wrappers passed credential-free local preflight at the last
  clean commit for run ID `nonagentic-two-topic-20260806`. Commit and verify the
  current review fixes before previewing.
- The private HF bucket is authenticated and both exact target prefixes are
  empty. dstack 0.20.29 now has masked, case-correct `HF_TOKEN`, `INDEX_URL`,
  `PYSERINI_API_TOKEN`, and `OPENROUTER_API_KEY` secrets; the pre-existing
  lowercase `hf_token` alias was left untouched. HF CLI authentication and
  dstack secret injection are configured separately; no token value is copied
  into the tracked repository.
- No dstack task or paid/cache-miss retrieval has been launched.

## Global constraints

- Work only in the linked worktree on branch
  `codex/distributed-retrieval-cache`; leave the main checkout and its untracked
  files untouched.
- Preserve the supported Retrieval → authenticated handoff → Generation
  boundary. This work runs retrieval only; it does not run RAG generation.
- The fixed/non-agentic runner remains bounded one-shot DeepSeek planning; it
  is not the separate open-ended agentic retrieval workflow.
- Cache keys include every input that can change an expensive result and omit
  experiment/output/machine identity. Accelerator and batch-size differences
  may reuse results where the approved numerical tolerance applies.
- `--offline-cache-only` must fail before opening a network client, loading a
  model, writing a cache entry, or mutating a checkpoint when any required
  value is absent.
- Bundle verification rejects links, devices, duplicate/colliding paths,
  traversal, undeclared members, digest/size mismatch, non-canonical JSON, and
  unsafe extraction targets.
- Immutable nonnumeric conflicts always fail. Numerical score/similarity
  conflicts fail by default; `--score-conflicts keep-existing` keeps the
  destination value and records an audit entry.
- A merger is retryable and idempotent. A durable prepare journal exists until
  the merge completion receipt is committed. Supported runners refuse to start
  against a cache root with an incomplete merge. Run consolidation quiescent
  with respect to non-merger score-cache writers; mergers sharing a destination
  are process-serialized and same-thread re-entry fails closed.
- Bundles and remote prefixes contain no credentials, provider raw responses,
  lock files, attempts, SQLite/WAL/SHM files, full corpus dumps, qrels, gold
  nuggets, RAGDoll scores, or generated RAG answers.
- Use the existing private Bucket `Npatta01/trec_mlm_2026` under immutable
  `trec_rag_2026/experiments/<run-id>/<topic-id>/` prefixes. Upload the archive
  first and the completion marker last. Do not implement automatic HF deletion.
- Public model snapshots are downloaded at pinned revisions on each worker;
  model weights are not copied into shard bundles.
- Before any live cache-miss run, show the declined dstack preview unchanged,
  selected topics, expected misses/calls, outputs, eligible offers, and price
  ceiling, then wait for explicit user authorization.
- The only planned live validation is two topics, `rag2026-0` and `rag2026-1`,
  on separate dstack tasks, followed by a credential-free local cache-only
  replay under a new experiment ID. No 119-topic run is authorized.
- Do not push, publish publicly, expose a service, or delete remote artifacts.

---

### Task 1: Add cache-root isolation and fail-closed read primitives

**Owned files:**

- Modify: `code/trec_rag/repo_env.py`
- Modify: `code/trec_rag/retrieval_cache.py`
- Modify: `code/trec_rag/retrievers.py`
- Create/modify focused tests under `code/tests/`

**Contract:** `TREC_RAG_CACHE_ROOT` accepts only an absolute cache root and
overrides the shared-checkout redirect. Retrieval cache-only lookup is a pure
read: it never creates a directory, lock, derivation, or repair file. The
retriever exposes exact transport-call accounting. Cache entry identities and
existing online behavior remain byte-compatible.

- [x] Add failing tests for absolute override validation, pure missing/hit
  lookup, no filesystem mutations, and transport call counts.
- [x] Run the focused tests and preserve the expected RED output in the task
  report.
- [x] Implement the minimum production behavior.
- [x] Run the focused suite and existing cache/retriever regression tests.

### Task 2: Cache deterministic planning, canonicalization, and similarity

**Owned files:**

- Create: `code/trec_rag/planning_cache.py`
- Create: `code/trec_rag/similarity_cache.py`
- Modify: `code/trec_rag/facet_extraction.py`
- Modify: `code/trec_rag/canonical_nuggets.py`
- Modify: `code/trec_rag/evidence_local.py`
- Create/modify focused tests under `code/tests/`

**Contract:** Planning keys hash the exact validated OpenRouter request body,
endpoint, model, prompt/schema revisions, and narrative input. The cache stores
only the validated canonical planning payload plus safe identity metadata, not
raw provider responses. Canonicalization can require a validated cache hit
without instantiating its backend. MiniLM similarity matrices are cached by
model/revision and ordered text hashes, stored as finite `float.hex()` values,
and expose hit/miss/model-batch accounting. Offline misses are side-effect free.

- [x] Add failing behavior tests, including malformed entries, exact identity
  changes, read-only misses, and lazy backend/model construction.
- [x] Verify RED for each new public behavior.
- [x] Implement immutable create-only entries and accounting.
- [x] Run focused plus planning/canonical/evidence regressions.

### Task 3: Make reranker scores read-only and portable

**Owned files:**

- Modify: `code/trec_rag/rerank_score_cache.py`
- Modify: `code/trec_rag/mixedbread_passage_scorer.py`
- Modify sentence-score accounting in `code/trec_rag/evidence_local.py` only
  after coordinating with Task 2 ownership
- Create/modify focused tests under `code/tests/`

**Contract:** `GlobalScoreCache(read_only=True)` opens an existing database
without creating SQLite, WAL, SHM, directories, or schema. `require_many`
returns all exact finite hits or raises one structured missing-entry error.
Portable export emits canonical context metadata and deterministic JSONL rows;
import is one transaction, is idempotent, and supports `strict` or audited
`keep-existing` numerical conflicts. Passage and sentence scorers expose cache
hits, misses, and model batches.

- [x] Write and verify failing tests for read-only hit/miss behavior, portable
  round trips, idempotence, strict rollback, keep-existing audit, and stats.
- [x] Implement the minimal SQLite/API changes without copying database files.
- [x] Run focused and all existing reranker-score regression tests.

### Task 4: Build deterministic, safe cache shard bundles

**Owned files:**

- Create: `code/trec_rag/competition_cache_bundle.py`
- Create: `code/tests/test_competition_cache_bundle.py`
- Create: `code/tests/test_competition_cache_bundle_offline_replay.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Contract:** Provide:

```text
python -m trec_rag.competition_cache_bundle pack \
  --config CONFIG --topic TOPIC --destination ABS_DIR
python -m trec_rag.competition_cache_bundle verify BUNDLE_DIR
python -m trec_rag.competition_cache_bundle merge \
  --cache-root ABS_CACHE --outputs-root ABS_OUTPUTS \
  --score-conflicts strict|keep-existing BUNDLE_DIR...
```

`pack` emits deterministic `bundle.tar.zst` and `bundle-complete.json`. The
archive contains a top-level `cache/` tree, the selected topic checkpoint,
source config, retrieval derivations and complete document-CAS closure,
validated planning/canonical/similarity entries, and portable score JSONL.
`verify` performs bounded streaming validation before extraction. `merge`
stages verified shards, writes a durable journal, create-only installs
immutable files, transactionally imports score rows, writes conflict audits,
and atomically publishes completion. Re-running the same merge converges.

- [x] Write failing pack/verify/merge tests using synthetic tiny artifacts.
- [x] Include attacks for traversal, symlink/hardlink/device, duplicates,
  undeclared members, decompression/size bounds, digest mismatch, and collision.
- [x] Implement deterministic timestamps/order/modes and canonical manifests.
- [x] Make `zstandard==0.25.0` a direct locked dependency.
- [x] Verify exact archive reproducibility and transactional retry behavior.

### Task 5: Integrate cache-only execution and zero-work receipts

**Owned files:**

- Modify: `code/trec_rag/competition_retrieval.py`
- Modify: `code/trec_rag/topic_dispatch.py`
- Modify: `code/trec_rag/evidence_store.py`
- Modify runner/config tests under `code/tests/`
- Modify: `code/trec_rag/README.md`

**Contract:** Add `--offline-cache-only`. Before topic work, the runner refuses
an incomplete merge journal. In cache-only mode it validates every needed
planning, retrieval, canonicalization, passage-score, sentence-score, and
similarity entry without creating clients/models or writing caches/checkpoints.
An online topic writes `<topic>/cache-operation-receipt.json`; the run writes
`cache-operation-manifest.json`. Receipts identify mode and report per stage:
hits, misses, network/provider calls, and model batches. A successful cache-only
receipt has zero misses/calls/model batches and cannot be synthesized from an
older online checkpoint.

- [x] Add failing CLI/integration tests for a complete hit, one miss per stage,
  lazy external factories, immutable checkpoints, incomplete-merge refusal,
  and receipt aggregation.
- [x] Implement explicit dependency construction and accounting.
- [x] Update the ignored two-topic config workflow and cache replay examples.
- [x] Run competition retrieval, topic-dispatch, evidence, and cache regressions.

### Task 6: Add the dstack and private-HF shard workflow

**Owned files:**

- Create: `.dstack/rag26-retrieval-cache-shard.yaml`
- Create: `code/tools/run_retrieval_cache_shard.sh`
- Create: `code/tools/apply_retrieval_cache_shard.sh`
- Create: `code/trec_rag/hf_bucket_listing.py`
- Create/modify shell/config contract tests under `code/tests/`
- Update: `code/trec_rag/README.md`

**Contract:** The task uses image
`huggingface/trl@sha256:4de10fa68e4f4e060cb41885044208e2a44715fc0d52ce83bb071b1fc6d63db1`.
The local launcher requires a clean branch whose upstream is an ancestor,
creates a committed-only clone, and passes that immutable snapshot through
dstack repo transport. The remote wrapper seals any dstack-applied patch in an
ephemeral commit, runs exactly one selected topic in an isolated cache/output
namespace, packs/verifies it, uploads archive first and completion last, then
foreground-verifies the remote listing/download/hash.
It accepts only named dstack secrets `HF_TOKEN`, `INDEX_URL`,
`PYSERINI_API_TOKEN`, and `OPENROUTER_API_KEY`.

Resources: one of `A5000,L4,RTX3090,RTX4090`, at least 24 GB VRAM, 32 GB RAM,
100 GB disk, on-demand, `$1.00/hour` maximum, five-hour maximum duration,
native `no-capacity` retry for 30 minutes, and immediate teardown. Download and
verify the Mixedbread revision
`3ea9d4dffa7d12a4f366be8e275c349de9fc9865` and MiniLM revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` before the live runner.

- [x] Add failing tests that parse the YAML and execute the wrapper's cheap
  preflight path without secrets or provisioning.
- [x] Implement the config/wrapper with foreground exit-code preservation.
- [x] Run the bundled dstack/HF preflight and exact image-command checks.
- [ ] Run and retain both declined previews through the supported
  `code/tools/apply_retrieval_cache_shard.sh --preview` commands in the README;
  preserve the complete output unchanged and do not submit.

### Task 7: Integrate, verify, and independently review

**Owned files:** all branch changes after parallel ownership is reconciled.

- [x] Review every worker diff and report; resolve overlaps deliberately.
- [x] Run focused suites, formatting/static checks used by the repository, and
  the complete pytest suite.
- [x] Run synthetic end-to-end: online fixture build → pack → verify → merge →
  cache-only replay, with credentials removed and filesystems snapshotted.
- [x] Inspect bundles for excluded secrets/raw artifacts and validate exact
  deterministic reproduction.
- [ ] Complete independent follow-up review after fixing all three load-bearing
  whole-branch findings.
- [ ] Re-read this plan and record verification evidence/current next action.

### Task 8: Gated two-machine validation and local replay

**Approval gate:** Stop after declined dstack previews and ask for explicit
authorization. The preview handoff must state two selected topics, exact output
and HF prefixes, expected cache reuse/misses, expected OpenRouter/Pyserini/model
work, eligible offers, price ceiling, maximum duration, and teardown behavior.

After approval only:

- [ ] Submit exactly two detached tasks once, one for `rag2026-0` and one for
  `rag2026-1`; let native bounded no-capacity retry handle marketplace races.
- [ ] Monitor normal logs/status without diagnostic secret-bearing output.
- [ ] Verify each task completion, list/download both immutable HF prefixes,
  and validate SHA-256 plus bundle manifests locally.
- [ ] Merge into a fresh staging cache, then run a new local retrieval
  experiment with `--offline-cache-only`, no external credentials, and the
  same two topic IDs via `.venv/bin/python-rocm`.
- [ ] Require zero misses, zero network/provider calls, zero model batches,
  valid topic checkpoints, handoff, TREC run, full-text ZIP, and operation
  receipts. Compare outputs while allowing small floating-point differences.
- [ ] Promote the verified staging cache into the shared cache using the same
  merger and preserve its completion/conflict receipts.
- [ ] Leave remote deletion out of scope and report the immutable prefixes.
