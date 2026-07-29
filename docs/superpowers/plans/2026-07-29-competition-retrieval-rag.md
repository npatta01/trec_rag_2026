# Competition Retrieval and RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide independently runnable competition retrieval and fixed-retrieval RAG generation paths that exchange organizer-compatible files and pass a live two-topic end-to-end run.

**Architecture:** Rename the existing retrieval module and config without changing retrieval behavior. Port PR #24's generation runner onto current `origin/master`, tighten every external contract to the current organizer release, and select bounded generation topics from the canonical TSV through strict config. Retrieval and generation remain separate commands joined only by the TREC run and full-text ZIP.

**Tech Stack:** Python 3.12, uv, pytest, PyYAML, requests/OpenRouter, TREC TSV/JSONL/ZIP artifacts.

## Global Constraints

- Both paths read `trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv` directly as exact headerless `narrative_id<TAB>narrative` rows.
- Retrieval emits `r_output_trec_rag_2026.tsv` with six TREC fields and `retrieval_with_text.jsonl.zip` with `query.qid`, `candidates[].docid`, and `candidates[].doc`.
- Generation emits `rag_output_trec_rag_2026.jsonl` with exactly `metadata`, `references`, and `answer`; metadata has exactly `team_id`, `narrative_id`, `narrative`, `run_id`, and `run_desc`.
- The public modules are `trec_rag.competition_retrieval` and `trec_rag.competition_rag`; no `trec_rag.official_run` alias remains.
- Commands use `code/tools/setup_env.sh` followed by `uv run --no-sync python`.
- No combined orchestrator, ranking change, evaluation/UI work, published outputs, or live provider call in automated tests.
- Implementation follows red-green-refactor; each production behavior is preceded by a focused failing test.
- The organizer source pins are `trec-rag-data@a6255c10119a2984a874f46172d94045168ab1f3` and `trec-rag-skills@f281e88f61252662033c681df8b1ed2d0ceda97e`.

---

### Task 1: Canonical retrieval naming and organizer pins

**Files:**
- Rename: `code/trec_rag/official_run.py` to `code/trec_rag/competition_retrieval.py`
- Rename: `configs/facet_pilot_v1.yaml` to `configs/rag26_competition_retrieval_v1.yaml`
- Modify: `code/tests/test_facet_pipeline_e2e.py`
- Modify: `code/tests/test_retrieval_export.py`
- Modify: `code/tests/test_evidence_pipeline_contract.py`
- Modify: `code/trec_rag/evidence_store.py`
- Modify: `code/trec_rag/retrieval_export.py`
- Modify: `code/trec_rag/README.md`
- Modify: `reports/facet_extraction_findings.md`
- Modify gitlinks: `trec-rag-data`, `trec-rag-skills`

**Interfaces:**
- Consumes: existing `run_official(config, *, topic_ids, topic_subset, external)` behavior.
- Produces: `trec_rag.competition_retrieval.run_official` and CLI `python -m trec_rag.competition_retrieval CONFIG [--topic ID ...]`.

- [ ] **Step 1: Change imports in retrieval tests to the new module name and add a CLI assertion that two repeated `--topic` values reach `run_official` in order.**

  The test must fail with `ModuleNotFoundError: trec_rag.competition_retrieval` before the rename.

- [ ] **Step 2: Run the focused red test.**

  Run: `uv run --no-sync python -m pytest code/tests/test_facet_pipeline_e2e.py code/tests/test_retrieval_export.py code/tests/test_evidence_pipeline_contract.py -q`

- [ ] **Step 3: Rename the module and config, update every tracked import/command/config reference, and preserve `run_official` signatures and retrieval artifact behavior verbatim.**

  Do not add an `official_run.py` forwarding module. Keep `schema_version: facet_pilot_config_v1` and `experiment.id: facet-deepseek-b40-v1` in the renamed YAML.

- [ ] **Step 4: Advance the two submodule gitlinks to the exact organizer commits in Global Constraints.**

- [ ] **Step 5: Run the focused tests and reference scan.**

  Run: `uv run --no-sync python -m pytest code/tests/test_facet_pipeline_e2e.py code/tests/test_retrieval_export.py code/tests/test_evidence_pipeline_contract.py -q`

  Run: `git grep -n 'trec_rag\.official_run\|configs/facet_pilot_v1\.yaml' -- ':!docs/superpowers/research/*' ':!docs/superpowers/specs/*' ':!docs/superpowers/plans/*'`

  Expected: tests pass and grep prints no tracked runtime/documentation references.

- [ ] **Step 6: Commit.**

  Commit message: `Rename the competition retrieval path`

### Task 2: Strict generation inputs and configuration

**Files:**
- Create: `code/trec_rag/competition_rag.py`
- Create: `code/tests/test_competition_rag.py`
- Create: `configs/rag26_competition_rag_gpt_sol_v1.yaml`

**Interfaces:**
- Consumes: canonical topic TSV, retrieval TREC run, query-bundled full-text JSONL/ZIP.
- Produces: `RagGenerationConfig`, `load_rag_generation_config(Path)`, `load_queries(Path)`, `select_queries(queries, topic_ids)`, `load_trec_run(...)`, and `load_documents(...)`.

- [ ] **Step 1: Port only PR #24's input/config tests, then tighten them with literal organizer fixtures.**

  Add failing cases proving: duplicate YAML keys fail; `schema_version` is required and equals `competition_rag_config_v1`; unknown/missing fields fail; topic TSV rows have exactly two tab-separated fields and no header; TREC rows have six fields, literal `Q0`, positive unique ranks starting at 1, unique docids, numeric non-increasing scores, and one stable nonempty run tag; ZIP rows accept the organizer core plus local extension fields; `inputs.topic_ids` selects known unique IDs in TSV order and rejects empty/duplicate/unknown IDs before file joins.

- [ ] **Step 2: Run the focused tests and verify they fail because the module is absent.**

  Run: `uv run --no-sync python -m pytest code/tests/test_competition_rag.py -q`

- [ ] **Step 3: Implement the minimal strict config and loaders.**

  Use a unique-key `yaml.SafeLoader`. Resolve input paths against the active checkout and then the shared checkout. The checked-in config must omit `inputs.topic_ids` so production selects all 119 official topics; set its paths to `outputs/facet-deepseek-b40-v1/r_output_trec_rag_2026.tsv` and `outputs/facet-deepseek-b40-v1/retrieval_with_text.jsonl.zip`.

- [ ] **Step 4: Run the focused tests.**

  Run: `uv run --no-sync python -m pytest code/tests/test_competition_rag.py -q`

- [ ] **Step 5: Commit.**

  Commit message: `Add strict competition RAG inputs`

### Task 3: Generation execution, validation, and recovery

**Files:**
- Modify: `code/trec_rag/competition_rag.py`
- Modify: `code/tests/test_competition_rag.py`

**Interfaces:**
- Consumes: Task 2's selected topics, ranked docids, and document text; an injected `JsonGenerator` or production `OpenRouterJsonGenerator`.
- Produces: `run_generation(config, generator)`, CLI `python -m trec_rag.competition_rag --config CONFIG`, resumable per-topic rows, and atomic final organizer JSONL.

- [ ] **Step 1: Port PR #24's generation and transport tests before production code.**

  Preserve tests for strict provider schema, identical transient retries, no semantic repair calls, official query order, resume, atomic consolidation, and CLI accepting only `--config`. Add failing regressions proving generated objects with any keys other than `references` and `answer` are rejected, and overwrite removes the configured output plus the dedicated work directory before a failed replacement so a later resume cannot reuse old rows.

- [ ] **Step 2: Run the focused red tests.**

  Run: `uv run --no-sync python -m pytest code/tests/test_competition_rag.py -q`

- [ ] **Step 3: Implement the provider client and pipeline from PR #24 with the two reviewed fixes.**

  Validate exact five-field metadata, nonempty unique selected references, answer objects with exactly `text` and `citations`, one to three unique zero-based integer citations per answer object, every reference cited, and at most 1,024 `str.split()` words. Overwrite uses a narrowly scoped recursive removal of only `config.resolved_work_dir`, then removes only `config.output_path`; retrieval inputs are never modified.

- [ ] **Step 4: Run the focused tests.**

  Run: `uv run --no-sync python -m pytest code/tests/test_competition_rag.py -q`

- [ ] **Step 5: Commit.**

  Commit message: `Add fixed-retrieval RAG generation`

### Task 4: Cross-path contract and operator documentation

**Files:**
- Modify: `code/tests/test_competition_rag.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes: the real `export_retrieval_run` artifact layout and Task 3 generation loader.
- Produces: a regression-tested file handoff and documented uv commands for full and two-topic runs.

- [ ] **Step 1: Add a failing integration test that creates retrieval-shaped TSV and ZIP artifacts with the real export schema, selects two topics from an official-shaped TSV, runs generation with an injected deterministic generator, and validates the final JSONL.**

  Assert exact topic order, root keys, five metadata fields, selected docids, citation indexes, and output basename. The production change that this catches is either path drifting to a private manifest/checkpoint schema.

- [ ] **Step 2: Run the integration test and verify the expected failure.**

  Run: `uv run --no-sync python -m pytest code/tests/test_competition_rag.py -q`

- [ ] **Step 3: Make only the minimal cross-path adjustment needed, then document the two independent commands, the three-file generation input, the non-submission status of the ZIP sidecar, `inputs.topic_ids`, setup, resume, and overwrite behavior.**

- [ ] **Step 4: Run generation, retrieval, topic, export, and integrity suites.**

  Run: `uv run --no-sync python -m pytest code/tests/test_competition_rag.py code/tests/test_facet_pipeline_e2e.py code/tests/test_retrieval_export.py code/tests/test_topics.py code/tests/test_run_integrity.py -q`

- [ ] **Step 5: Commit.**

  Commit message: `Document the two competition paths`

### Task 5: Full verification and live two-topic end-to-end run

**Files:**
- Create locally only: an ignored smoke config under the SDD workspace or `/tmp`; do not commit credentials, outputs, or raw provider responses.
- Record evidence: implementer report and final handoff only.

**Interfaces:**
- Consumes: committed clean branch, shared `.env`, canonical organizer TSV, retrieval cache/API, reranker cache/ROCm, OpenRouter.
- Produces: two-topic retrieval TSV + full-text ZIP and two-record final RAG JSONL, retained locally for inspection but excluded from git.

- [ ] **Step 1: Set up the uv environment and run the complete automated suite.**

  Run: `code/tools/setup_env.sh`

  Run: `uv run --no-sync python -m pytest -q`

- [ ] **Step 2: Verify compilation, whitespace, secrets, and branch cleanliness.**

  Run: `uv run --no-sync python -m compileall -q code/trec_rag code/tests`

  Run: `git diff --check`

  Run: `git status --short`

  Inspect changed files for credential values; only environment-variable names may be tracked.

- [ ] **Step 3: Run retrieval for `rag2026-0` and `rag2026-1` from the canonical config.**

  Run: `uv run --no-sync python -m trec_rag.competition_retrieval configs/rag26_competition_retrieval_v1.yaml --topic rag2026-0 --topic rag2026-1`

  Validate that the receipt contains exactly those IDs in official order, the TREC file has six fields on every row, and the ZIP contains text for every selected run docid.

- [ ] **Step 4: Create a local smoke config by copying the checked-in generation config and adding `inputs.topic_ids: [rag2026-0, rag2026-1]` plus a smoke-only output directory, then run live generation.**

  Run: `uv run --no-sync python -m trec_rag.competition_rag --config SMOKE_CONFIG`

  Do not print or store the API key. If an external service fails, preserve the bounded artifacts and report the exact service error without weakening validation.

- [ ] **Step 5: Validate the two-record result through `load_rag_generation_config`, `load_queries`, `select_queries`, and `validate_submission_record`; inspect topic IDs, word counts, references, and citations.**

  Expected: exactly two valid compact JSONL rows in organizer order and no untracked smoke artifact in the repository.

- [ ] **Step 6: Run a broad whole-branch review, address actionable findings, and present the integration options without pushing until authorized.**

