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

## Integration Evidence

Recorded on 2026-07-30 from linked worktree
`/tmp/trec-rag-competition-paths` on
`codex/competition-retrieval-rag`. This evidence covers Tasks 1-3 and Task 4
Steps 1, 2, and 4. The controller-owned whole-fix review and PR push remain
pending; this section does not claim either is complete.

### TDD and scoped-review record

- Task 1 initial RED: `4 failed, 95 deselected in 0.13s`, exposing numeric
  prompt labels, divergent topic parsing, encoded-secret persistence, and
  nonexistent CLI flags. Initial GREEN was `4 passed, 95 deselected in 0.11s`;
  the module passed with `99 passed in 0.23s`. Review fix round 1 then reproduced
  whitespace-header and layered-percent-decoding gaps with
  `2 failed, 4 passed, 96 deselected in 0.12s`; GREEN was
  `6 passed, 96 deselected in 0.07s`, followed by
  `102 passed in 0.24s`. Scoped spec and quality review were clean after commit
  `3ff0f35`.
- Task 2 initial RED was `3 failed, 95 deselected in 0.34s`, reproducing a
  hash/parse replacement race, dead direct-DocID navigation, and uncaught CLI
  failures. Initial GREEN was `3 passed, 95 deselected in 0.11s`; the module
  passed with `98 passed in 1.96s`. Review fix round 1 reproduced the unauthenticated
  organizer-run reopen with `1 failed, 98 deselected in 0.24s`; GREEN was
  `1 passed, 98 deselected in 0.10s`, followed by
  `99 passed in 2.04s`. Scoped review was clean after commit `f132504`, with
  four explicitly deferred minor test/cleanup items preserved in the SDD
  ledger.
- Task 3 RED processed three distinct documents and failed because the scoring
  cache retained all three (`maxsize=128, currsize=3`). GREEN was
  `1 passed, 13 deselected`; the two evidence modules passed with
  `18 passed in 0.15s`. Review fix round 1 added unconditional cache teardown;
  focused verification remained `1 passed, 13 deselected in 0.08s` and the two
  modules remained `18 passed in 0.15s`. Scoped review was clean after commit
  `cfa662c`.

### Complete verification and sealed post-run refresh

- Fresh repository verification passed: `676 passed in 29.12s`.
  `git diff --check origin/master...HEAD` exited 0 with no output.
- Only `trec_rag.competition_debug_report` was invoked. Retrieval, reranking,
  decomposition, canonicalization, and RAG generation were not invoked. The
  report-only receipt included `rag2026-0` and `rag2026-1`, reported
  `rag_included: true`, and retained SHA-256
  `17855f3ace3b662b67db4a6ff2bb31c7be857f26ff5be2b0d78644287cef5c71`
  for the organizer retrieval TSV and
  `65d2efae3dd54d0428f9bf49ed9fb2666996453418128d6df2a400b89ea076af`
  for the organizer RAG JSONL. The official topics input remained
  `72dc2fd358d3eeda973397ccd7a8775545b19a6deaefc67709167eee6a9f8a2c`.
- Real Google Chrome 150 checks passed at desktop `1440x1000` and true emulated
  mobile `390x844`: exactly one visible topic and one selected topic control;
  the approved story-first nine-stage order; 38 citation links with zero broken
  targets; topic-2 citation navigation into the revealed owning reference;
  ten directly visible selected-document disclosures plus one closed
  remaining-90 disclosure; four compact summary fields with a computed `10.4px`
  inline gap; and no horizontal page overflow (`1425 == 1425` desktop content
  width and `390 == 390` mobile). Light-mode screenshots were inspected at both
  viewports. A separately rendered organizer-valid direct-DocID fixture linked
  `doc-original` to numeric reference anchor `0` and revealed that target after
  activation.
- Exact overall funnels were
  `709 -> 100 -> 800 -> 2016 -> 320 -> 148 -> 71` for `rag2026-0` and
  `576 -> 100 -> 600 -> 1374 -> 240 -> 90 -> 60` for `rag2026-1`.
  All per-subnarrative rows matched sealed records: topic 0 had eight rows of
  `100` ranked documents, `252` passages, and `40` clusters with nugget counts
  `15, 20, 20, 13, 20, 20, 20, 20`; topic 1 had six rows of `100` ranked
  documents, `229` passages, and `40` clusters with nugget counts
  `10, 10, 10, 20, 20, 20`.
- Three unique values from the two available local `.env` paths had zero
  occurrences in the generated report. Chrome loaded zero external resources,
  and static HTML inspection found no external script, stylesheet, image,
  frame, audio, or video dependency. Three credential-pattern diff hits were
  inspected and were the explicit fake keys in redaction tests, not credentials.
- The source report, mode-`0600` rendered copy, and live HTTPS response are
  byte-identical at SHA-256
  `ec897ac66d6d4c4e66ff27818775d7e0bc71803c8aa2e64930dc48a0abbfbc50`;
  live HTTPS returned 200. Tailscale Serve and Funnel status showed only the
  authorized `tailnet only` mapping at
  `https://npatta01-framework.tail481212.ts.net/`, with no public Funnel.
- `trec-rag-data` remained clean and detached at
  `a6255c10119a2984a874f46172d94045168ab1f3`; `trec-rag-skills` remained clean
  and detached at `f281e88f61252662033c681df8b1ed2d0ceda97e`. The organizer
  TSV/RAG hashes and both submodule states were identical before and after the
  report-only refresh.

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

- [x] **Step 1: Run complete verification**

```bash
uv run --no-sync .venv/bin/python -m pytest -q
git diff --check origin/master...HEAD
```

- [x] **Step 2: Run the post-run report CLI only**

Regenerate from the existing two-topic configs without invoking retrieval or generation. Verify both topic IDs, organizer input hashes, direct-docid citation targets, desktop/mobile overflow, and source/rendered/live byte identity before replacing the tailnet-only portal copy.

- [ ] **Step 3: Run scoped and whole-fix review gates**

Review each task commit for spec and code quality, then review the complete `d378352..HEAD` fix range. Resolve every Critical/Important finding before continuing.

- [x] **Step 4: Record evidence and commit the plan**

Record RED failures, GREEN counts, final full-suite count, browser evidence, artifact SHA, unchanged input/submodule states, and review outcome.

- [ ] **Step 5: Push the existing PR branch**

```bash
git push origin codex/competition-retrieval-rag
```

Confirm PR #28 points at the new HEAD and remains targeted to `master` in `npatta01/trec_rag_2026`; do not push either organizer submodule repository.
