# Selected-Evidence Fixed Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce DeepSeek and Sol one-shot RAG output from sealed topic passages with unambiguous, validated citations.

**Architecture:** Retrieval projects each validated TopicRecords v3 selection into a strict per-topic generation record and publishes a sealed root handoff beside the organizer retrieval artifacts. One shared v2 runner renders only those passages, requires raw document-ID citations, validates them against the topic packet, then deterministically serializes organizer citation indexes.

**Tech Stack:** Python 3.12, dataclasses, canonical JSON/SHA-256, SQLite-backed TopicRecords, pytest, asyncio, OpenRouter HTTP.

## Global Constraints

- Use only `generation_handoff_manifest_v1` sourced from `topic_records_v4`.
- DeepSeek and Sol share one fixed one-shot implementation; do not import the paired agentic controller.
- Do not retain a generation path from query TSV, TREC run, ZIP, full documents, or document heads.
- Require raw ClimbMix docids from both models and convert them to organizer indexes only after topic-local validation.
- Permit exactly two semantic attempts and never substitute a model, evidence, citation, or partial answer.
- Preserve incomplete retrieval topics when exact selected evidence exists; fail loudly when no evidence exists.
- Persist sanitized, monotonic raw attempt artifacts and make failed topics resumable.
- Do not make hosted calls during implementation or verification.
- Preserve unrelated working-tree changes and stage only this implementation.

---

### Task 1: Strict selected-evidence handoff

**Files:**
- Create: `code/trec_rag/generation_handoff.py`
- Create: `code/tests/test_generation_handoff.py`

**Interfaces:**
- Produces: `GenerationTopic`, `GenerationHandoff`, `serialize_generation_topic()`, `load_generation_handoff()`, `render_generation_evidence()`, and `select_generation_topics()`.
- Consumes: no retrieval-private paths or full document bodies.

- [x] **Step 1: Add the reviewed contract tests before production code.**

  Cover deterministic bytes and hashes, duplicate keys, unknown fields, cross-group joins,
  exact character/UTF-8 byte geometry, topic ordering, fallback identity, and absence of a
  full-document sentinel in rendered evidence.

- [x] **Step 2: Run the handoff tests and verify RED.**

  Run: `.venv/bin/python -m pytest -q code/tests/test_generation_handoff.py`

  Expected: collection fails because `trec_rag.generation_handoff` does not exist.

- [x] **Step 3: Implement the typed canonical handoff.**

  Keep `HANDOFF_SCHEMA_VERSION = "generation_handoff_manifest_v1"`, set
  `SOURCE_CONTRACT = "topic_records_v4"`, reject noncanonical JSON, derive all hashes, and
  expose only selected passage text plus advisory claim hints.

- [x] **Step 4: Run the handoff tests and verify GREEN.**

  Run: `.venv/bin/python -m pytest -q code/tests/test_generation_handoff.py`

  Expected: all tests pass.

- [x] **Step 5: Commit the isolated contract.**

  Stage only the two files and commit `feat: add selected evidence generation handoff`.

### Task 2: TopicRecords v3 projector and manifest-last export

**Files:**
- Create: `code/trec_rag/generation_handoff_export.py`
- Create: `code/tests/test_generation_handoff_export.py`
- Modify: `code/trec_rag/retrieval_export.py`
- Modify: `code/tests/test_retrieval_export.py`

**Interfaces:**
- Consumes: an already validated `TopicRecords` handle, exact selected clusters, canonical
  results, official topic, document ranks, and source receipts.
- Produces: `ValidatedGenerationSnapshot`, `project_generation_topic()`, paired per-topic
  generation projection receipts, and root `generation_handoff_manifest.json`.

- [x] **Step 1: Add projector tests before the projector.**

  Assert that only selected cluster supports are reconstructed, unused supports survive,
  canonical hints cannot widen evidence, fallback passages are exact, and multibyte nonzero
  offsets survive the projection.

- [x] **Step 2: Run projector tests and verify RED.**

  Run: `.venv/bin/python -m pytest -q code/tests/test_generation_handoff_export.py`

  Expected: collection fails because `trec_rag.generation_handoff_export` does not exist.

- [x] **Step 3: Implement the minimal projector.**

  Implement `project_generation_topic(snapshot: ValidatedGenerationSnapshot) -> GenerationTopic`
  and `prepare_generation_handoff_artifact(...) -> PreparedGenerationHandoffArtifact`; require
  exact candidate/member equality and preserve source spans from authenticated candidates.

- [x] **Step 4: Add retrieval-export tests before wiring publication.**

  Add behavior tests for paired projection identity, incomplete-with-evidence topics, tamper
  rejection, resumed projection rebuild equality, generation handoff artifact receipts, and
  outer-manifest-last publication.

- [x] **Step 5: Run focused export tests and verify RED.**

  Run: `.venv/bin/python -m pytest -q code/tests/test_retrieval_export.py -k 'generation or handoff or projection'`

  Expected: new assertions fail because no generation projection or root handoff is published.

- [x] **Step 6: Wire the current v5/v3 exporter.**

  Extend the current retrieval completion fields and TopicRecords v3 seals rather than copying
  the older v4/v2 exporter. Pair organizer and generation projection receipts, validate both
  before mutation, lock per-topic publication, add the handoff to root artifact receipts, and
  write the root manifest last.

- [x] **Step 7: Run focused and complete export tests and verify GREEN.**

  Run: `.venv/bin/python -m pytest -q code/tests/test_generation_handoff_export.py code/tests/test_retrieval_export.py`

  Expected: all tests pass.

- [x] **Step 8: Commit the producer boundary.**

  Stage only handoff-export files and selected `retrieval_export` hunks, preserving concurrent
  canonical-nugget changes; commit `feat: publish selected evidence generation handoff`.

### Task 3: Shared v2 one-shot runner with raw-docid citations

**Files:**
- Modify: `code/trec_rag/competition_rag.py`
- Modify: `code/tests/test_competition_rag.py`
- Create: `code/trec_rag/organizer_documents.py` only if evaluation readers must move out of the runner.
- Create: `code/tests/test_organizer_documents.py` only with that move.
- Delete: `configs/rag26_competition_rag_deepseek_v1.yaml`
- Delete: `configs/rag26_competition_rag_gpt_sol_v1.yaml`
- Create: `configs/rag26_competition_rag_deepseek_v2.yaml`
- Create: `configs/rag26_competition_rag_gpt_sol_v2.yaml`

**Interfaces:**
- Consumes: one `GenerationHandoff` and an optional topic subset under `experiment.topic_ids`.
- Produces: strict organizer JSONL; provider-visible citations are raw docids and persisted rows
  contain deterministic integer indexes.

- [x] **Step 1: Replace v1-path tests with v2 behavior tests before replacing production.**

  Require one manifest input, reject every legacy source selector, prove both checked-in configs
  use the same runner contract, prove prompt sentinel exclusion, raw-docid conversion, numeric and
  foreign citation rejection, exactly two semantic attempts, monotonic resume attempt names, and
  no final output after exhaustion.

- [x] **Step 2: Run the focused runner tests and verify RED.**

  Run: `.venv/bin/python -m pytest -q code/tests/test_competition_rag.py -k 'handoff or docid or semantic or checked_in_competition_configs'`

  Expected: failures show the v1 runner still accepts old inputs and lacks selected-evidence raw-docid handling.

- [x] **Step 3: Implement the v2 fixed runner.**

  Load and authenticate the full handoff before output mutation; render the selected evidence;
  request string citations in the provider schema; validate references and citations against
  `topic.citation_docids`; map first-use docids to final indexes; persist every raw attempt; and
  keep ThreadPoolExecutor topic concurrency.

- [x] **Step 4: Bind resume identity to the complete contract.**

  Include manifest and context hashes, selected topics, rendered prompts, system prompt, response
  schema, semantic-attempt policy, model/provider/endpoint/reasoning/structured-output/
  temperature/token/timeout/transport settings. Reject older or differing identity versions.

- [x] **Step 5: Run all runner tests and verify GREEN.**

  Run: `.venv/bin/python -m pytest -q code/tests/test_competition_rag.py code/tests/test_organizer_documents.py`

  Expected: all present test files pass without network access.

- [x] **Step 6: Commit the citation fix separately.**

  Stage only the runner, its tests, moved reader if used, and v2 configs; commit
  `fix: validate model citations by document id`.

### Task 4: Integration verification and independent review

**Files:**
- Modify: `code/trec_rag/README.md` only where commands/config names changed.
- Modify: `docs/superpowers/plans/2026-08-04-selected-evidence-fixed-generation.md` to record evidence.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: a reviewed, committed checkpoint ready for a PR and later offline two-topic plumbing run.

- [x] **Step 1: Run the cheapest complete offline suite.**

  Run: `.venv/bin/python -m pytest -q code/tests/test_generation_handoff.py code/tests/test_generation_handoff_export.py code/tests/test_retrieval_export.py code/tests/test_competition_rag.py code/tests/test_organizer_documents.py`

  Expected: all present tests pass and no hosted calls occur.

- [x] **Step 2: Run adjacent retrieval and RAGDoll boundary tests.**

  Run: `.venv/bin/python -m pytest -q code/tests/test_competition_retrieval.py code/tests/test_ragdoll_io.py code/tests/test_competition_debug_report.py`

  Expected: all tests pass; any unrelated pre-existing failure is recorded with exact evidence.

- [x] **Step 3: Audit the staged patch.**

  Run: `git diff --cached --check` and inspect `git diff --cached --stat` plus the complete staged diff.

  Expected: no whitespace errors, secrets, outputs, caches, raw responses, or unrelated dirty files.

- [ ] **Step 4: Request an independent substantive code review.**

  Reviewer focus: source closure, topic isolation, citation semantics, retry multiplication,
  resume identity, publication ordering, and preservation of incomplete-with-evidence topics.

- [ ] **Step 5: Resolve findings test-first and rerun verification.**

  For every accepted finding, add or tighten a failing behavior test before the production fix,
  then repeat Steps 1-3.

- [ ] **Step 6: Commit documentation and hand off the PR-ready checkpoint.**

  Commit `docs: document selected evidence generation flow`, then report commit hashes, test
  counts, review disposition, and the exact next command for a local two-topic offline compile.

## Verification Record

- Plan self-review: every approved design requirement maps to Tasks 1-4.
- Placeholder scan: no deferred implementation placeholders remain.
- Type consistency: the producer and consumer both use `GenerationTopic` and
  `generation_handoff_manifest_v1`; citation conversion consumes `topic.citation_docids`.
- Implementation commits through `d2855da`: strict handoff, v2 runner, vital/okay Nuggetizer
  scoring, paired manifest-last export, fixed retrieval failure severity, sealed-handoff debug
  reports, and topic-aware organizer readers for RAGDoll.
- Offline boundary verification on 2026-08-04: 499 tests passed across handoff, projector,
  retrieval export, fixed RAG, retrieval v2, pipeline E2E, canonical nuggets, debug report,
  organizer documents, and RAGDoll IO. No hosted calls were made.
