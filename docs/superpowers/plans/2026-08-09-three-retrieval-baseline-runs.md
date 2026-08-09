# Three Retrieval Baseline Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce three deterministic, variable-cutoff TREC retrieval runs by rescoring the complete document unions already present in `outputs/facet-deepseek-b40-v3`, without rerunning retrieval.

**Architecture:** A pure local module validates source topics, aggregates a complete passage score matrix, applies one shared eligible-set cutoff, and writes three rankings. GPU workers only fill content-addressed reranker cache misses and publish immutable per-topic matrices; collection and ranking remain local and cache-only.

**Tech Stack:** Python 3.12, pytest, `semantic-text-splitter`, pinned Mixedbread CrossEncoder, SQLite score cache, zstandard archives, dstack 0.20.29, private Hugging Face Buckets.

## Global Constraints

- Source retrieval directory is `/home/npatta01/data/competitions/trec_rag_2026/outputs/facet-deepseek-b40-v3`.
- Do not invoke Pyserini, OpenRouter, qrels, gold nuggets, RAGDoll scores, or generation.
- Use `mixedbread-ai/mxbai-rerank-base-v2` at pinned revision `3ea9d4dffa7d12a4f366be8e275c349de9fc9865`.
- Chunk at 3,500 characters with 350-character overlap and half-open spans.
- Fail closed on missing documents, hashes, score pairs, zero subnarratives, NaN, or infinity.
- All three runs use the same eligible set and topic depth; never pad.
- Multiple BM25 query variants under one subnarrative form one retrieval pool and are scored once against the subnarrative text.
- Remote artifacts and transported source material remain private.

---

### Task 1: Pure scoring and cutoff kernel

**Files:**
- Create: `code/trec_rag/retrieval_baseline_runs.py`
- Create: `code/tests/test_retrieval_baseline_runs.py`

**Interfaces:**
- Produces: `PassageScore`, `DocumentScore`, `TopicScores`, `midrank_percentiles`, `suppress_overlaps`, `weighted_passage_score`, `select_eligible_documents`, and `build_rankings`.
- Consumes: finite raw logits and half-open passage spans; it performs no file or model I/O.

- [ ] **Step 1: Write failing golden tests for percentile and overlap semantics**

  Add literal fixtures proving tied values use midranks, `N=1` maps to `1.0`, exact overlap coefficient `0.5` suppresses the lower-scored span, and top-four weights renormalize.

- [ ] **Step 2: Run the golden tests and verify RED**

  Run: `.venv/bin/python -m pytest -q code/tests/test_retrieval_baseline_runs.py`

  Expected: collection/import failure because `trec_rag.retrieval_baseline_runs` does not exist.

- [ ] **Step 3: Implement finite score validation, overlap suppression, aggregation, and percentiles**

  Implement descending score order with span-offset ties, intersection divided by the shorter span, suppression at `>= 0.5`, weights `(0.55, 0.25, 0.13, 0.07)`, and the literal midrank formula from the design.

- [ ] **Step 4: Run tests and verify GREEN**

  Run: `.venv/bin/python -m pytest -q code/tests/test_retrieval_baseline_runs.py`

- [ ] **Step 5: Add failing tests for cutoff and all three rankings**

  Cover per-unit raw positive MAD, zero MAD with strict-above-median admission, narrative-or-subnarrative union, empty fallback, one-subnarrative combo renormalization, pooled-subnarrative breadth, top-three-per-document before top-100-per-subnarrative, deterministic ties, shared eligible set, and no padding.

- [ ] **Step 6: Implement cutoff and ranking functions and verify GREEN**

  Run: `.venv/bin/python -m pytest -q code/tests/test_retrieval_baseline_runs.py`

- [ ] **Step 7: Commit the independently testable kernel**

  Commit: `feat: add retrieval baseline scoring kernel`

### Task 2: Strict source loader, matrix scorer, and TREC export

**Files:**
- Modify: `code/trec_rag/retrieval_baseline_runs.py`
- Modify: `code/tests/test_retrieval_baseline_runs.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Produces: `load_topic_input(source_dir, topic_id, document_store) -> TopicInput`, `score_topic(input, scorer, chunker) -> TopicMatrix`, `read_topic_matrix(path)`, `write_topic_matrix(path)`, `export_runs(...)`, and CLI subcommands `score-topic`, `rank`, and `verify`.
- Consumes: `decomposition/result.json`, `retrieval/audit.json`, `scoring/selection.json`, content-addressed `cache/documents/v1`, and `MixedbreadPassageScorer`.

- [ ] **Step 1: Write failing loader tests using a complete temporary topic fixture**

  Assert exact union membership, minimum retrieval rank across duplicate occurrences, subnarrative pooling, document SHA-256 verification, natural topic ordering, and rejection of incomplete/malformed source topics.

- [ ] **Step 2: Implement strict loading and verify GREEN**

  Run: `.venv/bin/python -m pytest -q code/tests/test_retrieval_baseline_runs.py -k 'load or source or topic'`

- [ ] **Step 3: Write failing matrix I/O and cache-only replay tests**

  Use a real `SemanticTextChunker` with a deterministic fake scorer. Assert every document × (narrative + subnarratives) pair is represented, raw passage spans survive round-trip, the manifest hashes the canonical JSONL bytes, and a second read-only scorer run performs zero model calls.

- [ ] **Step 4: Implement matrix scoring and atomic canonical artifacts**

  Store no full document text in matrices. Include source hashes, model/chunker identity, exact pair counts, cache hits/misses/model batches, and a matrix SHA-256.

- [ ] **Step 5: Write failing end-to-end export tests**

  Assert three TREC files have identical per-topic doc sets and depths, unique contiguous ranks, scores `k-rank+1`, deterministic byte output, support for final depths above 1,000, and manifests without text/secrets.

- [ ] **Step 6: Implement `rank`, `verify`, CLI parsing, and README commands**

  Run: `.venv/bin/python -m pytest -q code/tests/test_retrieval_baseline_runs.py`

- [ ] **Step 7: Run a local 64-document fixture against `rag2026-0`**

  Use a temporary matrix/output directory and the shared cache in read-only mode. Verify pair accounting and deterministic rerun; do not represent the sample as topic validation.

- [ ] **Step 8: Commit the end-to-end local pipeline**

  Commit: `feat: build retrieval baseline matrices and run files`

### Task 3: Private cloud score worker and collector

**Files:**
- Create: `code/trec_rag/retrieval_baseline_bundle.py`
- Create: `code/tests/test_retrieval_baseline_bundle.py`
- Create: `code/tools/run_retrieval_baseline_worker.sh`
- Create: `code/tools/apply_retrieval_baseline_worker.sh`
- Create: `.dstack/rag26-retrieval-baseline-worker.yaml`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Produces: immutable input bundle builder, safe extractor, per-topic worker receipt, cache-delta archive, matrix archive, local collector, `--preflight` worker mode, declined-preview launcher, and detached launch mode.
- Consumes: a clean committed source snapshot, private `hf://buckets/.../trec_rag_2026/artifacts/...` input URI, disjoint topic assignments, and private experiment output prefix.

- [ ] **Step 1: Write failing bundle security and integrity tests**

  Cover traversal/absolute/symlink member rejection, immutable manifest hashes, exact topic closure, missing document rejection, duplicate-topic collection rejection, and idempotent identical collection.

- [ ] **Step 2: Implement deterministic bundle creation/extraction/collection**

  Bundle only selected source-topic JSON, exact referenced document objects, and a portable passage score-cache snapshot. Require canonical member ordering and SHA-256 verification before use.

- [ ] **Step 3: Write failing shell preflight tests**

  Execute the real wrappers with fake `hf` and `dstack`; assert pinned image/interpreter, clean committed-only snapshot, no secret values in argv/files, foreground workload, failure-preserving sync, unique safe run names, and exact topic arguments.

- [ ] **Step 4: Implement worker and launcher**

  Use the digest-pinned CUDA image, direct `hf`, on-demand `[A40, A6000, L40S]` with at least 48 GB VRAM/RAM, a hard `max_price`, and bounded native `no-capacity` retry. Keep scorer batch size explicit in the worker manifest.

- [ ] **Step 5: Run focused cloud-workflow tests**

  Run: `.venv/bin/python -m pytest -q code/tests/test_retrieval_baseline_bundle.py code/tests/test_retrieval_baseline_runs.py`

- [ ] **Step 6: Commit the cloud workflow**

  Commit: `feat: distribute retrieval baseline scoring`

### Task 4: Independent review and two-topic cloud smoke

**Files:**
- Create privately/ignored: `outputs/retrieval-baseline-smoke-20260809/`

**Interfaces:**
- Consumes: reviewed committed code and topics `rag2026-0`, `rag2026-37`.
- Produces: two verified matrices, three smoke TREC files, collection receipt, cache-only replay receipt, and measured runtime/cost envelope.

- [ ] **Step 1: Request independent Sol code review**

  Require review of source binding, cutoff math, pooled-subnarrative breadth, cache identity, artifact integrity, shell safety, and organizer output validity. Address every substantive finding with tests first.

- [ ] **Step 2: Run dstack/HF capability preflight and worker `--preflight`**

  Record only non-secret capability results and exact nested argv.

- [ ] **Step 3: Build and upload immutable two-topic input bundle**

  Verify local archive digest, empty private destination, remote listing, downloaded digest, and member checksums.

- [ ] **Step 4: Show the declined dstack preview unchanged**

  Present the full offers table and maximum smoke cost; wait for explicit confirmation before submission.

- [ ] **Step 5: Launch exactly once, monitor, collect, and verify**

  Confirm topic isolation, complete pair accounting, pinned identities, hashes, shared eligible set/depth, no padding, deterministic byte ordering, and a cache-only replay with zero model batches.

- [ ] **Step 6: Inspect plausibility without qrels**

  Report per-topic pool size, `k_t`, cutoff statistics, run overlap/order differences, breadth distributions, and several top-document evidence summaries without exposing document text outside private outputs.

### Task 5: Full 119-topic run

**Files:**
- Create privately/ignored: `outputs/retrieval-baseline-full-20260809/`

**Interfaces:**
- Consumes: the approved smoke implementation and measured per-topic runtime/cache misses.
- Produces: 119 verified topic matrices and three organizer-ready run files.

- [ ] **Step 1: Create a disjoint shard plan for all 119 topics**

  Balance shards using smoke throughput and exact estimated pair counts. Hash the plan and reject overlaps/omissions.

- [ ] **Step 2: Build/upload the immutable full input bundle and preview every task shape**

  Verify bundle digest remotely and show full dstack offers plus aggregate maximum spend before launch.

- [ ] **Step 3: Launch each approved shard once and monitor to terminal state**

  Use native bounded capacity retry only. Do not externally resubmit ambiguous or failed runs without diagnosis.

- [ ] **Step 4: Download, hash-check, and collect all result shards**

  Verify exact 119-topic coverage, complete matrix accounting, no conflicting cache values, and idempotent collection.

- [ ] **Step 5: Generate and verify the three final run files**

  Run cache-only replay and the verifier twice; require byte-identical outputs and organizer-compliant variable depths.

- [ ] **Step 6: Run final tests and record verification evidence**

  Run: `.venv/bin/python -m pytest -q code/tests/test_retrieval_baseline_runs.py code/tests/test_retrieval_baseline_bundle.py`

  Also run: `git diff --check origin/master...HEAD` and an independent final Sol review.
