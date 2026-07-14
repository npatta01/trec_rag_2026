# Complete RAG Candidate Mixedbread Diagnostic Plan

> **Post-qrels mechanism diagnostic:** the four topics in this pilot are already exposed. This run can diagnose whether a facet-aware Mixedbread pass fixes the ranking mechanism; it cannot provide fresh confirmatory evidence.

**Goal:** Produce a complete, deterministic ordering of every accepted document for each pilot topic while using Mixedbread only on the bounded disagreement pool that can improve the useful head and middle of the RAG candidate ranking.

**Architecture:** Preserve exact RRF ranks 1–10. Form a residual scoring pool from the union of RRF@500, GLOBAL@500, and DUAL@1000, minus the protected head. Score each residual document against one topic-level structured query containing the unchanged narrative and every accepted facet obligation. Rank the best residual documents at 11–500, then append every remaining accepted document in frozen DUAL order. No document is discarded at 100 or 500.

**Model policy:** Reuse the locally materialized `mixedbread-ai/mxbai-rerank-base-v2` revision `3ea9d4dffa7d12a4f366be8e275c349de9fc9865`, sentence-transformers 5.6.0, raw logits, bfloat16, and the historically strongest `chunk_top4_weighted` formula. The inferior and substantially more expensive 32k-token whole-document branch is not run. No network, retrieval, model download, paid call, or recursive query repair.

---

## Task 1: Freeze the qrels-blind scoring manifest and budget

**Files:**
- Create: `code/trec_rag/deep_facet_candidate_mixedbread.py`
- Create: `code/tests/test_deep_facet_candidate_mixedbread.py`
- Create: `outputs/rag25_deep_facet_candidates_v1/post_qrels_mixedbread_v1/preflight/`

1. Write failing tests for protected-topic rejection before source access, identical facet coverage in the structured query, exact protected RRF head, residual-pool construction, complete candidate population, and deterministic hashes.
2. Implement pure builders and validators.
3. Materialize create-only `manifest.json`, `candidates.jsonl`, and `preflight.json` with exact document, character-window, cache-hit, cache-miss, runtime-policy, and zero-paid-cost counts.
4. Seal all scoring identities before model load.
5. Stop rather than score if local model revision, backend, ROCm, source seals, score policy, or budget differs.

## Task 2: Score the bounded disagreement pool locally

**Files:**
- Modify: `code/trec_rag/deep_facet_candidate_mixedbread.py`
- Create: `outputs/rag25_deep_facet_candidates_v1/post_qrels_mixedbread_v1/scoring/`

1. Write failing tests for local-only model loading, raw-logit scoring, resumable create-only topic shards, exact query/text hashes, and score-count completeness.
2. Reuse the repository semantic chunker and Mixedbread cache identity.
3. Run document and passage scoring on ROCm with temporary files rooted under the shared cache scratch directory.
4. Aggregate the strongest four passage logits with frozen weights `0.55, 0.25, 0.13, 0.07`; do not tune weights on these topics.
5. Save scores and a receipt containing elapsed time, GPU/device, peak memory when available, model revision, and $0 external cost.

## Task 3: Build and seal the complete document rankings

**Files:**
- Modify: `code/trec_rag/deep_facet_candidate_mixedbread.py`
- Create: `outputs/rag25_deep_facet_candidates_v1/post_qrels_mixedbread_v1/freeze/`

1. Write failing tests that ranks 1–10 equal RRF exactly; ranks 11–500 use descending Mixedbread score with a deterministic tie-break; every unselected accepted document follows in DUAL order; ranks are contiguous; no duplicates exist; and the output set equals `U_accepted` exactly.
2. Build all four full rankings (8,114 rows total), retaining source and score provenance.
3. Seal parameters, source bindings, score hashes, rankings, and semantic invariants before evaluation.

## Task 4: Evaluate ranking quality for RAG candidate selection

**Files:**
- Modify: `code/trec_rag/deep_facet_candidate_mixedbread.py`
- Create: `outputs/rag25_deep_facet_candidates_v1/post_qrels_mixedbread_v1/evaluation/`

1. Verify the ranking seal, then read only the existing projected qrels artifact.
2. Compare Mixedbread, RRF, GLOBAL, DUAL, and the prior cascade at depths 10, 100, 500, 1000, and full accepted-union depth.
3. Report nDCG@10, graded recall, binary recall, judged rate, relevant count, and novel-facet-document retention curves.
4. State explicitly that full-depth recall is a population property shared by every complete permutation; the useful result is how quickly relevant and novel facet evidence rises into practical candidate depths.
5. Deep-dive any topic regression or failure rather than treating a threshold miss as unexplained.

## Task 5: Render findings and obtain advisor review

**Files:**
- Modify: `code/trec_rag/build_deep_facet_candidate_report.py`
- Modify: `code/tests/test_build_deep_facet_candidate_report.py`
- Modify: `reports/experiments/deep_facet_candidate_pilot_v1/report.html`
- Create: `reports/experiments/deep_facet_candidate_pilot_v1/mixedbread_advisor_review.md`

1. Add an answer-first section explaining whether the reranker fixed noisy facet candidates, how many novel relevant documents moved into 100/500/1000, and that all documents remain available to downstream RAG.
2. Add a compact complete-ranking diagram and per-topic retention table.
3. Ask the advisor to review the frozen method, metrics, regression diagnosis, and recommendation.
4. Incorporate only evidence-backed corrections without retuning the exposed topics.
5. Run targeted tests, the report verifier at desktop and mobile viewports, seal verification, and git diff/status review before committing only scoped files.

## Acceptance criteria

- No protected topic is read or joined.
- No network, retrieval, model download, or paid call occurs.
- The expensive pool is exactly the frozen residual union, not the full accepted union.
- Each output ranking is a complete permutation of every accepted document.
- RRF ranks 1–10 are byte-for-byte preserved by identity.
- Mixedbread scores are comparable within a topic because all residual documents use the same topic-level narrative-plus-obligations query.
- Every score, ranking, metric, and report claim is reproducible from sealed local artifacts.
- Results are labeled post-qrels diagnostic, not fresh validation.
