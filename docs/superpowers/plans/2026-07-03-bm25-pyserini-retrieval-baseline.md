# BM25 Pyserini Retrieval Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a retrieval-only TREC RAG 2026 baseline that runs title-only BM25 against the merged hosted Pyserini helper layer and writes a valid TREC runfile.

**Architecture:** Keep reusable code under `code/trec_rag/`. Topic parsing is separate from retrieval execution, while runfile formatting and validation live with the baseline module so the first baseline stays small and inspectable.

**Tech Stack:** Python standard library, existing `trec_rag.remote_pyserini` helpers, `pytest`, hosted Pyserini ClimbMix endpoint.

---

## File Structure

- Create `code/trec_rag/topics.py`: loads official JSONL topics and dev TSV topics into a shared `Topic` dataclass.
- Create `code/trec_rag/baselines/__init__.py`: marks baseline modules as importable.
- Create `code/trec_rag/baselines/bm25_retrieval.py`: runs title-only BM25 retrieval, caches raw responses, writes TREC runfiles, and validates retrieval output.
- Create `code/tests/test_bm25_retrieval_baseline.py`: unit tests for topic loading, row ordering, caching, output formatting, and validation.
- Modify `code/trec_rag/README.md`: document the BM25 baseline inputs, output, commands, and validation.
- Modify `README.md`: mention the baseline command in the repository overview.

## Task 1: Topic Loading

- [ ] Write failing tests for official JSONL topics, dev TSV topics, whitespace normalization, and malformed input rejection in `code/tests/test_bm25_retrieval_baseline.py`.
- [ ] Run `PYTHONPATH=code uv run --with pytest pytest code/tests/test_bm25_retrieval_baseline.py -q` and verify the tests fail because `trec_rag.topics` does not exist.
- [ ] Implement `Topic`, `derive_title`, `load_topics`, and `write_topics_jsonl` in `code/trec_rag/topics.py`.
- [ ] Re-run the topic tests and verify they pass.

## Task 2: BM25 Retrieval Runner

- [ ] Add failing tests for `run_bm25_retrieval` that use a fake client returning out-of-order ranks, verify title-only queries, verify sorted TREC rows, and verify raw response cache files.
- [ ] Run the targeted test and verify it fails because `trec_rag.baselines.bm25_retrieval` does not exist.
- [ ] Implement `RetrievalRow`, `candidates_to_run_rows`, `write_retrieval_run`, and `run_bm25_retrieval`.
- [ ] Re-run the runner tests and verify they pass.

## Task 3: Retrieval Validation And CLI

- [ ] Add failing tests for missing topics, duplicate docids within a topic, non-contiguous ranks, and malformed six-column rows.
- [ ] Run the targeted validation tests and verify they fail because validation is not implemented.
- [ ] Implement `validate_retrieval_run`, `build_arg_parser`, and `main` in `code/trec_rag/baselines/bm25_retrieval.py`.
- [ ] Re-run all baseline tests and existing remote Pyserini tests.

## Task 4: Documentation And Smoke Check

- [ ] Document how to run the baseline with `PYTHONPATH=code python -m trec_rag.baselines.bm25_retrieval --topics ... --output ...`.
- [ ] Document validation and the generated output locations.
- [ ] Run the full Python test set with `PYTHONPATH=code uv run --with pytest pytest code/tests -q`.
- [ ] If local Pyserini credentials are available through `.env`, run one-topic smoke retrieval and validate the generated runfile.
