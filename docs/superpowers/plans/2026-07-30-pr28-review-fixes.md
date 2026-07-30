# PR 28 Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the eight accepted PR #28 review findings without narrowing the current organizer-valid RAG contract.

**Architecture:** Keep the public CLIs and artifact formats stable. Harden the RAG boundary in `competition_rag.py`, make the debug report parse exactly the bytes it receipts and resolve every organizer-valid citation form, and bound the two document-text caches without restructuring unrelated pipeline code.

**Tech Stack:** Python 3.11, pytest, uv with `--no-sync`, standard-library JSON/URL parsing, self-contained HTML.

## Global Constraints

- Preserve organizer-valid RAG rows: extra metadata, zero to three citations, integer or direct-docid citations, and uncited references remain accepted.
- Do not rerun retrieval or hosted-model generation; the debug report may be regenerated only from existing sealed artifacts.
- Use strict RED-GREEN TDD for every production behavior change and record the expected RED failure.
- Keep `.env`, generated reports, raw responses, caches, and organizer submodule contents out of commits.
- Leave `trec-rag-data` and `trec-rag-skills` pinned and clean.
- Run Python through `uv run --no-sync .venv/bin/python` for this user-selected workflow.

---

### Task 1: Harden RAG prompts, parsing, redaction, and operator guidance

**Files:**
- Modify: `code/trec_rag/competition_rag.py`
- Modify: `code/tests/test_competition_rag.py`

**Interfaces:**
- Consumes: organizer topics accepted by `trec_rag.topics.load_narrative_topics`; existing `RagGenerationConfig` modes.
- Produces: an unambiguously docid-labelled prompt, `load_queries(path) -> list[tuple[str, str]]` with canonical TSV semantics, and persisted provider envelopes that omit strings containing percent-decoded API keys.

- [ ] **Step 1: Add four failing behavioral tests**

Add tests proving that: prompt documents are labelled by docid without bracketed numeric pseudo-citations; `load_queries` accepts a narrative containing an extra tab and a Unicode line separator exactly as one narrative; a parsed JSON provider envelope containing `secret%2Dtoken` never persists that encoded secret; and create/failure errors name `experiment.mode: resume` or `experiment.mode: overwrite` rather than nonexistent CLI flags.

- [ ] **Step 2: Run the four tests and verify RED**

Run:

```bash
uv run --no-sync .venv/bin/python -m pytest -q \
  code/tests/test_competition_rag.py -k 'prompt_labels or canonical_topic_parser or percent_encoded or configuration_mode_guidance'
```

Expected: failures expose `[1]` prompt labels, split-line/parser divergence, retained percent-encoded secret text, and `--resume`/`--overwrite` wording.

- [ ] **Step 3: Implement the minimal RAG fixes**

Change `render_prompt` to render `Reference document docid: <docid>` blocks with no numeric bracket labels and update `USER_PROMPT` accordingly. Implement `load_queries` through `load_narrative_topics`, preserving header rejection and exact topic order. In `_redact_text`, repeatedly percent-decode the candidate string and replace the entire string with `[REDACTED]` when any configured secret appears after decoding. Rewrite the two operator errors to reference `experiment.mode` values.

- [ ] **Step 4: Verify GREEN and module compatibility**

```bash
uv run --no-sync .venv/bin/python -m pytest -q code/tests/test_competition_rag.py
```

- [ ] **Step 5: Commit**

```bash
git add code/trec_rag/competition_rag.py code/tests/test_competition_rag.py
git commit -m "Fix competition RAG review findings"
```

---

### Task 2: Bind report receipts to rendered bytes and repair citation navigation

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py`
- Modify: `code/tests/test_competition_debug_report.py`

**Interfaces:**
- Consumes: the same bounded retrieval/RAG artifact paths and `RagAnswerItemReport.citation_docids`.
- Produces: one authenticated bounded snapshot primitive used for every hash-and-parse artifact, valid numeric reference anchors for both citation forms, and concise nonzero CLI errors.

- [ ] **Step 1: Add three failing behavioral tests**

Add deterministic tests proving that a same-size replacement between hashing and parsing of a noncanonical artifact cannot alter rendered data; direct-docid citation chips target the owning numeric reference-card ID; and `main()` converts a malformed-artifact failure into `SystemExit("error: ValueError: ...")` without printing a traceback.

- [ ] **Step 2: Run the tests and verify RED**

```bash
uv run --no-sync .venv/bin/python -m pytest -q \
  code/tests/test_competition_debug_report.py -k 'noncanonical_snapshot or direct_docid_anchor or cli_reports_concise_error'
```

Expected: forged bytes are rendered, the docid link target is absent, and the raw exception escapes.

- [ ] **Step 3: Implement snapshot-bound parsing and link resolution**

Extract a bounded snapshot context manager that yields the exact bytes used to compute the receipt, and use it wherever `_sha256_file` is currently followed by a second parse open (export manifest, RAG JSONL, topic decomposition/selection/selected documents, passage rankings, audit JSON, and any equivalent pattern found by `rg`). In `_render_final_rag`, build a `docid -> zero-based reference index` map and pass it to `_render_rag_answer_item`; construct every chip anchor from the resolved numeric index. Wrap debug-report CLI execution like `competition_rag.main()`, preserving exit 130 for `KeyboardInterrupt` and returning concise typed errors otherwise.

- [ ] **Step 4: Verify GREEN and the report module**

```bash
uv run --no-sync .venv/bin/python -m pytest -q code/tests/test_competition_debug_report.py
```

- [ ] **Step 5: Commit**

```bash
git add code/trec_rag/competition_debug_report.py code/tests/test_competition_debug_report.py
git commit -m "Bind debug report receipts to rendered bytes"
```

---

### Task 3: Bound document projection cache retention

**Files:**
- Modify: `code/trec_rag/facet_evidence.py`
- Modify: `code/tests/test_evidence_pipeline_contract.py`

**Interfaces:**
- Consumes: repeated calls using one current document string.
- Produces: identical scoring text, boundary tuples, and byte offsets while retaining at most one document per cache.

- [ ] **Step 1: Add a failing cache-retention test**

Call `_scoring_text_and_boundaries` and `_byte_offsets` with three distinct document strings, then assert each `cache_info().currsize == 1` and verify the latest document still produces correct offsets and normalized scoring text.

- [ ] **Step 2: Run the test and verify RED**

```bash
uv run --no-sync .venv/bin/python -m pytest -q \
  code/tests/test_evidence_pipeline_contract.py -k bounded_document_projection_cache
```

Expected: both caches retain three entries under the current `maxsize=128` decorators.

- [ ] **Step 3: Apply the minimal memory bound**

Set both existing `lru_cache` decorators to `maxsize=1`. Do not change projection results or public interfaces.

- [ ] **Step 4: Verify GREEN and evidence contracts**

```bash
uv run --no-sync .venv/bin/python -m pytest -q \
  code/tests/test_evidence_pipeline_contract.py code/tests/test_evidence_local_contract.py
```

- [ ] **Step 5: Commit**

```bash
git add code/trec_rag/facet_evidence.py code/tests/test_evidence_pipeline_contract.py
git commit -m "Bound document projection caches"
```

---

### Task 4: Integrate, review, regenerate, and update PR #28

**Files:**
- Modify: `docs/superpowers/plans/2026-07-30-pr28-review-fixes.md`
- Generated only: existing two-topic private debug report outside git

**Interfaces:**
- Consumes: Tasks 1-3 and the existing sealed two-topic retrieval/RAG artifacts.
- Produces: a review-clean branch, refreshed private HTML, and an updated PR #28 head.

- [ ] **Step 1: Run complete verification**

```bash
uv run --no-sync .venv/bin/python -m pytest -q
git diff --check origin/master...HEAD
```

- [ ] **Step 2: Run the post-run report CLI only**

Regenerate from the existing two-topic configs without invoking retrieval or generation. Verify both topic IDs, organizer input hashes, direct-docid citation targets, desktop/mobile overflow, and source/rendered/live byte identity before replacing the tailnet-only portal copy.

- [ ] **Step 3: Run scoped and whole-fix review gates**

Review each task commit for spec and code quality, then review the complete `d378352..HEAD` fix range. Resolve every Critical/Important finding before continuing.

- [ ] **Step 4: Record evidence and commit the plan**

Record RED failures, GREEN counts, final full-suite count, browser evidence, artifact SHA, unchanged input/submodule states, and review outcome.

- [ ] **Step 5: Push the existing PR branch**

```bash
git push origin codex/competition-retrieval-rag
```

Confirm PR #28 points at the new HEAD and remains targeted to `master` in `npatta01/trec_rag_2026`; do not push either organizer submodule repository.
