# RAG25 Top-1,000 Reranker Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare pinned Mixedbread and GTE ModernBERT rerankers over the same top-1,000 BM25 documents for all 22 TREC RAG 2025 development narratives using maximum passage score and document-level nDCG.

**Architecture:** Reuse the existing original-narrative pipeline to populate the shared, model-independent Pyserini response cache. Reuse `rerank_score_cache` to create model-separated raw-logit window artifacts and content-addressed SQLite caches, then add one strict evaluator that validates both artifact populations, aggregates maximum passage score per document, and writes private machine-readable metrics.

**Tech Stack:** Python 3.12, pytest, existing TREC RAG pipeline/evaluation modules, Sentence Transformers 5.6.0, ROCm locally, CUDA through dstack, private Hugging Face Bucket artifacts.

## Global Constraints

- Work only in `.worktrees/rag25-reranker-1k` on `codex/rag25-reranker-1k`.
- Use exactly 22 numeric RAG25 development topic IDs and one untouched narrative query per topic.
- Retrieve exactly 1,000 ClimbMix BM25 documents per topic.
- Chunk at 3,500 characters with 350-character overlap and score at maximum length 1,024 in BF16.
- Pin Mixedbread revision `3ea9d4dffa7d12a4f366be8e275c349de9fc9865` and GTE revision `f7481e6055501a30fb19d090657df9ec1f79ab2c`.
- Keep retrieval text, score artifacts, model responses, and result details in ignored/private paths; do not publish them.
- Share retrieval responses across models, but require separate model/revision-bound reranker cache identities.
- Run a cheap top-100/full-22 validation before the expensive top-1,000 GPU workload.

---

### Task 1: Strict max-passage evaluator

**Files:**
- Create: `code/trec_rag/reranker_depth_evaluation.py`
- Create: `code/tests/test_reranker_depth_evaluation.py`

**Interfaces:**
- Consumes: `load_window_artifact(path: Path, expected_depth: int) -> WindowArtifact`
- Produces: `evaluate_max_passage_systems(...) -> dict[str, object]` and a CLI that writes `metrics.json` plus `topic_metrics.csv`.

- [x] **Step 1: Write failing artifact-validation tests**

  Test literal two-topic fixtures for contiguous BM25 ranks, complete chunk indices, one identity per artifact, exact topic population, and rejection of duplicate/conflicting scores.

- [x] **Step 2: Run tests and verify RED**

  Run: `.venv/bin/python -m pytest code/tests/test_reranker_depth_evaluation.py -q`

  Expected: import failure because `trec_rag.reranker_depth_evaluation` does not exist.

- [x] **Step 3: Implement strict artifact loading and max aggregation**

  For each `(topic_id, docid)`, require one BM25 rank and chunks `0..chunk_count-1`; compute `max(score)`; rank descending with `(bm25_rank, docid)` tie breaks.

- [x] **Step 4: Write failing metric and comparison tests**

  Use hand-derived qrels to assert nDCG@10/@20, BM25 baseline, judged rates, top-k overlap, per-topic deltas, wins/losses/ties, and deterministic paired-bootstrap output.

- [x] **Step 5: Implement evaluation and writers, then verify GREEN**

  Reuse `trec_rag.evaluation.evaluate_ranked` for repository-consistent nDCG. Write JSON and CSV atomically enough that a failed evaluation does not masquerade as a complete result.

- [x] **Step 6: Run targeted and adjacent tests**

  Run: `.venv/bin/python -m pytest code/tests/test_reranker_depth_evaluation.py code/tests/test_pipeline.py code/tests/test_rerank_score_cache.py -q`

### Task 2: Experiment configurations and documentation

**Files:**
- Create: `configs/rag25_reranker_1k_mixedbread_v1.yaml`
- Create: `configs/rag25_reranker_1k_gte_modernbert_v1.yaml`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes: the shared top-1,000 Pyserini request cache.
- Produces: separate window artifacts and score-cache paths because model and revision participate in `ScoreCacheContext`.

- [x] **Step 1: Add two configs with identical retrieval/chunking policy and distinct model identities**

  Both configs use the RAG25 topic/qrels files, original narrative, `hits: 1000`, `candidate_depth: 1000`, raw logits, BF16, and unique artifact namespaces.

- [x] **Step 2: Validate configs without model loading**

  Run both `rerank_score_cache --dry-run --score-kind window --limit-per-topic 100` after the retrieval cache exists, and confirm their printed global cache paths differ.

- [x] **Step 3: Document the retrieval, scoring, and evaluation commands**

  State explicitly that retrieval is shared and reranker scores are model/revision separated.

### Task 3: Retrieve and validate all 22 top-1,000 pools

**Files:**
- Generated private: `outputs/rag25_bm25_full_query_v1/`
- Generated shared private: `cache/retrieval/pyserini_remote/`

**Interfaces:**
- Consumes: secure `INDEX_URL` and `PYSERINI_API_TOKEN` from ignored repo environment files.
- Produces: 22 request-keyed raw responses and 22,000 normalized retrieved rows.

- [x] **Step 1: Preflight identities and secrets without printing values**

  Confirm exact topic count/IDs, clean tracked worktree, pinned submodules, output namespace, and presence of required retrieval variables.

- [x] **Step 2: Report expected calls and paths, then run the baseline pipeline once**

  Run: `.venv/bin/python -m trec_rag.pipeline --config configs/rag25_bm25_full_query_1k_20260806.yaml`

- [x] **Step 3: Validate retrieval output**

  Require 22 topics, 1,000 unique contiguous-rank documents per topic, exact narrative query identity, and successful baseline metrics.

### Task 4: Cheap full-population scoring gate

**Files:**
- Generated shared private: `cache/reranker/artifacts/rag25_reranker_1k_comparison_v1/`

**Interfaces:**
- Consumes: top-100 prefix of every retrieved topic.
- Produces: resumable score cache entries and a preliminary paired evaluation.

- [x] **Step 1: Score top 100 for both models locally**

  Use `.venv/bin/python-rocm -m trec_rag.rerank_score_cache`, `--score-kind window`, `--limit-per-topic 100`, and model-specific configs.

- [x] **Step 2: Evaluate the top-100 gate**

  Require 22 complete topics, finite scores, model-separated identities, and no evaluation validation errors.

- [x] **Step 3: Measure actual chunk count and update the full-run estimate**

  Extrapolate only after reporting measured top-100 chunks/model and expected cache reuse.

### Task 5: Private dstack top-1,000 scoring run

**Files:**
- Generated ignored: `tmp/rag25-reranker-1k-input/`
- Generated ignored: `tmp/rag25-reranker-1k.dstack.yml`
- Durable private: `hf://buckets/Npatta01/trec_mlm_2026/trec_rag_2026/experiments/<immutable-run-name>/`

**Interfaces:**
- Consumes: immutable checksummed bundle containing only code/config/topic/qrels and the 22 retrieval responses.
- Produces: both window artifacts, model-separated score-cache SQLite files, logs/receipts, manifest, and checksums.

- [x] **Step 1: Run dstack/HF capability preflight**

  Use the skill preflight; verify private bucket prefix is empty and workload image/tool/interpreter paths are exact.

- [x] **Step 2: Build, inspect, hash, and privately upload the minimal input bundle**

  Reject absolute/traversal members and record archive/member digests. Do not upload `.env`, raw logs, qrels beyond the private experiment, or unrelated repository files.

- [x] **Step 3: Exercise the real wrapper preflight**

  Confirm it validates bundle hashes, nested scorer argv, dependencies, GPU/BF16, all 22 topics, 1,000 candidates each, and distinct cache identities without inference.

- [x] **Step 4: Preview exact dstack offers unchanged**

  Use an A40-class compatible pool, on-demand policy, bounded price/duration, and no-capacity-only retry. Show full declined preview output.

- [x] **Step 5: Submit exactly once and monitor**

  Run detached after the already granted authorization; use normal logs and never diagnostic logs for the secret-bearing task.

- [x] **Step 6: Verify durable output**

  List the private prefix, download it, and verify every substantive file against `SHA256SUMS` and the manifest.

### Task 6: Final evaluation and verification

**Files:**
- Generated private: `outputs/rag25-reranker-1k-comparison-v1/metrics.json`
- Generated private: `outputs/rag25-reranker-1k-comparison-v1/topic_metrics.csv`
- Modify: this plan with final status/evidence.

**Interfaces:**
- Consumes: verified full top-1,000 Mixedbread and GTE window artifacts.
- Produces: macro/per-topic nDCG, judged coverage, overlap, paired uncertainty, runtime, cost, and a model recommendation.

- [x] **Step 1: Stage verified score artifacts privately without overwriting shared model caches**

- [x] **Step 2: Run the evaluator at candidate depths 50, 100, 200, 500, and 1,000**

- [x] **Step 3: Verify metric invariants and compare the historical top-50 Mixedbread anchor**

- [x] **Step 4: Run the complete targeted test suite and inspect git diff/status**

- [x] **Step 5: Record final verification evidence and hand off results**

## Final evidence (2026-08-06)

- Retrieval: 22 RAG25 development topics, 22,000 unique BM25 documents, and
  237,089 scored passage windows per reranker. The retrieval baseline was
  nDCG@10 0.413961 and nDCG@20 0.413107.
- Local top-100 gate: Mixedbread 16m54.71s; GTE ModernBERT 8m53.60s;
  GTE was 1.90x faster on the Radeon host. Aggregate nDCG was effectively
  tied at this gate, but top-10 overlap was only 43.6%.
- A40 CUDA run: Mixedbread 4,272s (71m12s); GTE ModernBERT 2,555s
  (42m35s), both exit code 0. Runtime environment was torch 2.9.1+cu128,
  transformers 5.13.0, sentence-transformers 5.6.0, and BF16 inference.
- dstack: run `rag25-reranker-1k-a40-20260806-v2` completed with cost
  `$0.9032`; including the failed preflight, dstack cost was `$0.9518`.
  The result prefix is private and its downloaded four substantive files pass
  `sha256sum -c SHA256SUMS`.
- Full-depth evaluator output is private under
  `outputs/rag25_reranker_1k_comparison_v1/full_a40/`. It contains metrics,
  per-topic CSV, 22,000 document rows per system, and 237,089 passage rows per
  system. The source score artifacts remain staged under ignored `tmp/` paths.
- Full verification: a fresh `.venv/bin/python -m pytest code/tests -q` run
  passed 2,278 tests, skipped 19, and reported 60 subtests passed in 44.25s.
  The post-change adjacent suite was 158 passed.
