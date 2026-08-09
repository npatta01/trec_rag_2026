# Cache-First Candidate-Core Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan one task at a time, and `superpowers:test-driven-development` for every behavior change.

**Goal:** Produce three variable-depth retrieval submissions from the latest `facet-deepseek-b40-v3` artifact by scoring only a deterministic candidate core, reusing the existing reranker cache, completing remaining inference remotely in under two hours and under $10, and merging the verified new scores into the local shared cache.

**Architecture:** Derive a candidate core independently for each topic by robust-thresholding every authenticated source lane and taking their union. Seal those decisions and only their documents into a portable input bundle. A single H200-first remote worker scores the two canary topics, verifies a cache-only replay, then scores the other 117 topics without reloading the model. After immutable publication, a local collector verifies every hash and matrix in a fresh cache, transactionally imports non-conflicting scores into the shared cache, and reproduces all three runs cache-only.

**Tech stack:** Python 3.11, pytest, PyTorch/Transformers reranker, SQLite score cache, Bash, dstack, Hugging Face Buckets.

## Global constraints

- Work only in `.worktrees/retrieval-baseline-advisor-runs` on `codex/retrieval-baseline-advisor-runs`.
- Source artifact is `outputs/facet-deepseek-b40-v3`, whose export code commit must equal current `origin/master` (`05fdf35d858bf52bec2843c699a4bfd85d4b8c61`) and whose export manifest SHA-256 is `cb5a81608c48a5c43692ada0c9121cd96b7160589f08f5bbaf4e20612001463a`.
- Use the existing shared cache read-only while building the bundle. Remote inference writes to a new isolated cache.
- Use H200 preferred and H100 fallback with `max_duration: 1h45m`; after two bounded failed attempts costing about $1.08 total, the relaunch remains bounded below $10 total at the configured `$5/hour` ceiling.
- The exact declined dstack preview and price must be shown before launch. The user’s approval authorizes the implementation and the bounded run, but the preview remains the final infrastructure safety check.
- The remote worker must process `rag2026-1` and `rag2026-18` first and stop immediately if their fresh-cache replay is not byte-identical or requires a model batch.
- Never overwrite or delete existing shared cache rows. Any score conflict aborts the merge.
- The final three runs use the same candidate core and therefore the same variable cutoff `k_t`; they differ only in ordering.
- No arbitrary 1,000-document truncation and no padding.

## File structure

- Create `code/trec_rag/retrieval_candidate_core.py`: candidate-core derivation, validation, and serialization.
- Modify `code/trec_rag/retrieval_baseline_runs.py`: candidate-only matrices, authenticated cutoff, and three ordering strategies.
- Modify `code/trec_rag/retrieval_baseline_input_bundle.py`: all-topic sealed candidate bundle and deterministic archive.
- Create `code/trec_rag/retrieval_baseline_collection.py`: remote publication verification, replay, shared-cache merge, and receipt.
- Modify `code/tools/run_retrieval_baseline_worker.sh`: canary gate followed by the remaining topics in one model process.
- Modify `code/tools/apply_retrieval_baseline_worker.sh`: create and launch the all-topic clean snapshot.
- Create `code/tools/collect_retrieval_baseline_worker.sh`: download and invoke local collection safely.
- Modify `.dstack/rag26-retrieval-baseline-worker.yaml`: bounded H200 profile.
- Create `.dstack/rag26-retrieval-baseline-worker-h100.yaml`: bounded H100 fallback profile.
- Modify `code/trec_rag/README.md`: reproducible build, preview, run, collect, and replay commands.
- Create/modify focused tests under `code/tests/` for every module and shell contract above.

---

### Task 1: Implement deterministic candidate-core derivation

**Files:**

- Create: `code/trec_rag/retrieval_candidate_core.py`
- Create: `code/tests/test_retrieval_candidate_core.py`

**Step 1: Write failing tests**

Cover lane parsing, duplicate `(lane, docid)` rejection, non-finite score rejection, unknown-document rejection, exact lane-name validation, even-length medians, MAD and zero-MAD thresholds, union admission, deterministic original-lane fallback, serialization round-trip, and stable SHA-256 provenance.

**Step 2: Run the focused test and confirm RED**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_candidate_core.py -q
```

Expected: import or behavior failures because the module does not exist.

**Step 3: Implement the minimal kernel**

Define immutable records for per-lane statistics and the topic candidate core. Implement:

```python
def derive_candidate_core(
    *,
    topic_id: str,
    lane_scores_path: Path,
    expected_lane_names: tuple[str, ...],
    expected_docids: frozenset[str],
    best_retrieval_ranks: Mapping[str, int],
) -> CandidateCore: ...

def candidate_core_to_dict(core: CandidateCore) -> dict[str, object]: ...
def candidate_core_from_dict(value: Mapping[str, object]) -> CandidateCore: ...
```

For each source lane, compute `median` and `MAD` over its authenticated aggregate document scores. Admit scores at least `median + 2.5 * 1.4826 * MAD`; if MAD is zero, admit scores strictly above the median. Union admissions across the original narrative and all facet-text lanes. If empty, retain exactly the original-lane argmax, breaking ties by global best retrieval rank and then UTF-8 docid bytes. Record per-lane admitted counts, overlaps, multiplicity histogram, pre-fallback count, fallback status, and source hash.

**Step 4: Run tests and confirm GREEN**

Run the focused test until all cases pass.

**Step 5: Commit**

```bash
git add code/trec_rag/retrieval_candidate_core.py code/tests/test_retrieval_candidate_core.py
git commit -m "feat: derive cache-first retrieval candidate cores"
```

---

### Task 2: Make matrices and rankings candidate-core authoritative

**Files:**

- Modify: `code/trec_rag/retrieval_baseline_runs.py`
- Modify: `code/tests/test_retrieval_baseline_runs.py`

**Step 1: Write failing tests**

Add tests proving that only candidate documents are chunked/scored, matrix schema v3 embeds and authenticates the candidate decision, read-back rejects mismatched docs or source hashes, ranking never applies a second document cutoff, every run has exactly `k_t = |C_t|`, and cache-only replay performs zero model batches.

For run 3, test that passage overlap is suppressed within each subnarrative, at most three passages per document/subnarrative are retained before thresholding, strong passages are selected by the same raw median/MAD rule across the complete candidate core, and ranking keys are:

1. number of supported subnarratives;
2. number of strong passages;
3. run-2 combo score;
4. best retrieval rank;
5. UTF-8 docid bytes.

The old global top-100 passage cap and source-lane document restriction must disappear.

**Step 2: Run the focused test and confirm RED**

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_baseline_runs.py -q
```

**Step 3: Implement schema and scoring changes**

- Add `candidate_core` to `TopicMatrix` and bump the matrix header/schema to v3.
- Restrict `score_topic` to candidate docids before chunking or cache lookup.
- Add a multi-topic scoring entrypoint that initializes the scorer once, accepts
  an ordered list of `(topic input, candidate core, output matrix)` jobs, and
  keeps one cache/scorer instance alive across the canary and full phases.
- Require an explicit candidate-core path for every topic in the scoring CLI.
- Validate that matrix documents exactly equal the candidate docids.
- Use all matrix documents as final eligibility in `rank_topic_matrix` and `build_rankings`.
- Preserve run 1 narrative percentiles and run 2 facet/combo percentiles.
- Replace run 3’s global passage cap with per-subnarrative robust strong-passage admission.
- Record per-run ordering components and shared `k_t` in manifests.

**Step 4: Run tests and confirm GREEN**

Run the focused suite, then existing reranker/cache tests that touch the matrix path.

**Step 5: Commit**

```bash
git add code/trec_rag/retrieval_baseline_runs.py code/tests/test_retrieval_baseline_runs.py
git commit -m "feat: rank authenticated candidate-core retrieval runs"
```

---

### Task 3: Seal a deterministic all-topic candidate bundle

**Files:**

- Modify: `code/trec_rag/retrieval_baseline_input_bundle.py`
- Modify: `code/tests/test_retrieval_baseline_input_bundle.py`

**Step 1: Write failing tests**

Test schema v2, 119-topic support, exact official topic-set enforcement, fixed canary order, candidate-core files under an allowed root, copying only candidate documents, exporting only reusable score hits, exact per-topic/global hit-miss counts, path traversal rejection, bundle hash verification, and byte-identical deterministic archives.

**Step 2: Run the focused test and confirm RED**

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_baseline_input_bundle.py -q
```

**Step 3: Implement bundle v2**

- Build and validate one candidate core for every selected topic.
- Require exactly the 119 official topics for the production mode.
- Copy only candidate documents plus required source metadata.
- Seal every candidate decision and include its hash in the root manifest.
- Calculate cache hits/misses over candidate docs × semantic units before archive creation.
- Store `rag2026-1` and `rag2026-18` as authenticated canaries.
- Add deterministic `tar.gz` construction with normalized paths, timestamps, ownership, and ordering.

**Step 4: Run tests and confirm GREEN**

**Step 5: Commit**

```bash
git add code/trec_rag/retrieval_baseline_input_bundle.py code/tests/test_retrieval_baseline_input_bundle.py
git commit -m "feat: seal all-topic candidate-core scoring bundle"
```

---

### Task 4: Implement one-process remote canary and full scoring

**Files:**

- Modify: `code/tools/run_retrieval_baseline_worker.sh`
- Modify: `code/tools/apply_retrieval_baseline_worker.sh`
- Modify: `code/tests/test_retrieval_baseline_worker.py`
- Modify: `.dstack/rag26-retrieval-baseline-worker.yaml`
- Create: `.dstack/rag26-retrieval-baseline-worker-h100.yaml`

**Step 1: Write failing shell-contract tests**

Test that the worker derives all topic IDs from the authenticated bundle, rejects counts other than 119, scores canaries first, performs immediate fresh-cache canary replay, stops on any byte mismatch/model batch/missing row, then processes the remaining 117 without restarting the model. Test immutable output prefixes, complete cache export, SHA256SUMS, manifest-last publication, empty-prefix handling, bounded duration, H200 primary, H100 fallback, and max price settings.

**Step 2: Run tests and confirm RED**

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_baseline_worker.py -q
```

**Step 3: Implement the bounded worker**

- Verify and extract the sealed bundle before importing any portable hits.
- Create a new isolated remote cache.
- Start one scorer process/model and process canaries first.
- Export canary scores to a fresh replay cache and rebuild matrices cache-only.
- Continue the remaining topics only after that gate passes.
- Export the complete isolated score cache and perform a final all-topic fresh replay.
- Publish matrices, three runs, cache archive, receipts, and checksums immutably; write the publication manifest last.
- Make `hf buckets list` empty output a valid empty-prefix result rather than JSON-decoding it.

**Step 4: Configure bounded GPU fallback**

Set both dstack profiles to one hour 45 minutes. Use an H200 marketplace selector first; use the H100 profile only when the H200 preview has no acceptable offer. Keep `max_price` low enough that cumulative attempts remain under $10.

**Step 5: Run tests and confirm GREEN**

**Step 6: Commit**

```bash
git add code/tools/run_retrieval_baseline_worker.sh code/tools/apply_retrieval_baseline_worker.sh code/tests/test_retrieval_baseline_worker.py .dstack/rag26-retrieval-baseline-worker.yaml .dstack/rag26-retrieval-baseline-worker-h100.yaml
git commit -m "feat: run bounded all-topic candidate scoring remotely"
```

---

### Task 5: Verify publication and merge the cache locally

**Files:**

- Create: `code/trec_rag/retrieval_baseline_collection.py`
- Create: `code/tools/collect_retrieval_baseline_worker.sh`
- Create: `code/tests/test_retrieval_baseline_collection.py`

**Step 1: Write failing tests**

Cover missing/extra files, checksum mismatches, manifest-last receipt validation, wrong source or bundle identity, fresh-cache import, byte-identical all-topic matrix replay, detection of model batches, advisory lock acquisition, transactional shared-cache import, conflict abort with no partial writes, idempotent re-import, and merge-receipt contents.

**Step 2: Run the focused test and confirm RED**

```bash
.venv/bin/python -m pytest code/tests/test_retrieval_baseline_collection.py -q
```

**Step 3: Implement collection**

Provide a CLI that accepts the downloaded publication directory, expected source/bundle hashes, fresh replay cache path, shared cache path, and final output path. It must:

1. verify publication checksums and identities;
2. import the portable cache into a new empty replay cache;
3. rebuild all 119 matrices cache-only and compare bytes with remote matrices;
4. acquire an exclusive local merge lock;
5. import scores transactionally into the shared cache, rejecting conflicts and preserving existing rows;
6. write a merge receipt with inserted/existing row counts and before/after hashes;
7. rebuild all matrices and three final runs from the merged shared cache with zero model batches.

The shell wrapper downloads into a fresh explicit directory, never into the shared cache, and invokes the Python collector only after download succeeds.

**Step 4: Run tests and confirm GREEN**

**Step 5: Commit**

```bash
git add code/trec_rag/retrieval_baseline_collection.py code/tools/collect_retrieval_baseline_worker.sh code/tests/test_retrieval_baseline_collection.py
git commit -m "feat: verify and merge remote retrieval score cache"
```

---

### Task 6: Document and verify the complete workflow

**Files:**

- Modify: `code/trec_rag/README.md`
- Modify focused test files as needed for integration coverage.

**Step 1: Add an end-to-end fake-backend test**

Exercise bundle build → canary gate → remaining topics → immutable publication → fresh local replay → transactional cache merge → final run export, using small fixtures and a fake scorer. Assert shared variable depth, no padded topics, and byte-stable outputs.

**Step 2: Document exact operator commands**

Include latest-artifact checks, bundle creation, upload, H200 declined preview, H100 fallback preview, launch, monitoring, download, local collection, final verification, and recovery after interruption. Clearly mark private artifact locations and the bounded-duration/cost limits.

**Step 3: Run focused and broader verification**

```bash
.venv/bin/python -m pytest \
  code/tests/test_retrieval_candidate_core.py \
  code/tests/test_retrieval_baseline_runs.py \
  code/tests/test_retrieval_baseline_input_bundle.py \
  code/tests/test_retrieval_baseline_worker.py \
  code/tests/test_retrieval_baseline_collection.py -q

.venv/bin/python -m pytest code/tests -q
```

Run shell syntax checks for both scripts and load both dstack YAML files through the dstack CLI’s configuration parser.

**Step 4: Commit**

```bash
git add code/trec_rag/README.md code/tests
git commit -m "docs: document cache-first retrieval baseline workflow"
```

---

### Task 7: Independent review and bounded production execution

**Files:**

- Modify only files identified by review findings.
- Generate private ignored artifacts under `outputs/`, `cache/`, and a fresh temporary bundle directory.

**Step 1: Request independent code and design review**

Ask the existing Sol reviewer/advisor to check the candidate-core semantics, matrix authentication, passage-breadth ordering, cache isolation/import safety, remote stop gates, and cost controls. Address every actionable finding and rerun the focused suite.

**Step 2: Build and verify the latest all-topic bundle locally**

Confirm origin/source identities, 119 topics, candidate distribution, exact cache hits/misses, and deterministic archive hashes. Upload the archive to a new private immutable Hugging Face Bucket prefix.

**Step 3: Preview the H200 job**

Run the exact dstack apply command without `--yes`. Show the selected offer, hourly price, maximum job cost, source commit, bundle hash, 119-topic count, canaries, expected hits/misses, and output prefix. If there is no acceptable H200, decline it and preview the H100 profile instead. Do not run both.

**Step 4: Launch and monitor one job**

After the preview safety check, launch the chosen job. Stop it if the canary gate fails, the job cannot finish within one hour 45 minutes, or projected cumulative spend can exceed $10.

**Step 5: Download, verify, and merge locally**

Download the immutable publication, run the collector, inspect the merge receipt, and confirm the shared local cache replay performs zero model batches.

**Step 6: Validate final outputs**

Verify all 119 topics in each of the three organizer-facing runs, variable per-topic depth, identical doc sets across runs, deterministic ordering, no duplicates, no padding, and complete provenance receipts. Report exact runtime, cost, cache hits/misses, inserted rows, output paths, and submission filenames.

**Step 7: Commit any final reviewed fixes**

Stage only intentional source/test/documentation changes. Keep outputs, caches, bundles, logs, provider responses, and secrets untracked.
