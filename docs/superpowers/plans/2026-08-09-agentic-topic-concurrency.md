# Agentic Topic Concurrency Implementation Plan

> **Required sub-skill:** Use `superpowers:test-driven-development` for each behavior change and `superpowers:verification-before-completion` before claiming completion.

**Goal:** Run independent agentic retrieval topics concurrently with a bounded worker count while preserving the global retrieval throttle, per-topic run-state ownership, and sealed export behavior.

**Architecture:** The parent process authenticates the run and allocates topic attempts. Injected test executors use a thread pool; live production execution uses a spawn-based process pool, with each child constructing its own topic-local retriever, scorer, chunker, and agent. The parent records results and failures and performs the existing final selection/export.

**Global constraints:** Keep the per-topic follow-up lock unchanged. Keep provider throttling global. Do not add Runpod, change budgets/depth/cache identity, or alter organizer outputs in this change.

## Task 1 — Add failing concurrency tests

**Files:** `code/tests/test_agentic_retrieval_config.py`, `code/tests/test_competition_agentic_retrieval.py`

- Add a config test proving `execution.topic_workers: 2` is accepted.
- Add a two-topic test with a barrier and active-worker counter; assert both topic executions overlap.
- Run only these tests and confirm they fail for the current exact-one parser and sequential dispatcher.

## Task 2 — Make the worker count configurable

**Files:** `code/trec_rag/agentic_retrieval_config.py`, `configs/rag26_competition_agentic_retrieval_v1.yaml`, `code/tests/test_agentic_retrieval_config.py`

- Accept a positive integer for `execution.topic_workers` while retaining strict validation for malformed values.
- Set the canonical agentic config to two workers and update its contract test.
- Run the config tests.

## Task 3 — Create a topic-local production worker seam

**Files:** `code/trec_rag/competition_agentic_retrieval.py`, `code/tests/test_competition_agentic_retrieval.py`

- Move live model/retriever/chunker construction into the per-topic execution function.
- Add a module-level production worker wrapper suitable for spawn pickling.
- Keep result sealing and failure artifact writes in the parent process.
- Run existing production-wiring and lifecycle tests.

## Task 4 — Dispatch topics with bounded concurrency

**Files:** `code/trec_rag/competition_agentic_retrieval.py`, `code/tests/test_competition_agentic_retrieval.py`

- Allocate requests in selection order and preserve source-order receipt IDs.
- Keep a serial path for one worker; use a thread pool for injected test executors and a spawn process pool for live production.
- Let sibling futures settle before re-raising unexpected exceptions; convert expected topic operational failures to the existing failure artifacts.
- Run the agentic orchestration and run-state tests.

## Task 5 — Document and verify the boundary

**Files:** `code/trec_rag/README.md`

- Document worker ownership, global quota behavior, and the canonical worker count.
- Run the focused retrieval tests, then `git diff --check` and inspect the final diff.
- Remove the temporary worktree `.venv` symlink, stage only requested files, and commit the implementation.
