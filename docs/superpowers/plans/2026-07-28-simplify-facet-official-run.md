# Simplified Facet Official Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the draft PR's many stage CLIs and shallow handoff modules with one runnable official-run interface and a smaller set of internal domain modules, without changing retrieval, evidence, cache, checkpoint, fallback, or organizer-export behavior.

**Architecture:** `trec_rag.official_run.run_official()` is the only supported caller interface and CLI. It executes five private stages—planning, retrieval/reranking, extractive evidence, canonicalization, and organizer export—using typed values internally and the existing sealed files only as durable checkpoints. Planning, retrieval, evidence, canonicalization, artifact codecs, and export remain separate deep modules; stage-specific command parsers disappear.

**Tech Stack:** Python 3.12, standard-library JSON/JSONL/SQLite/ZIP/hash/atomic filesystem operations, PyYAML, pytest.

## Global Constraints

- Work only on `codex/facet-extraction-clean` in the existing linked worktree and update draft PR #20; do not create a second PR.
- Preserve the current YAML `retrieval`, `reranking`, and `nuggets` blocks. Output paths and filenames remain conventions beneath `outputs/<experiment.id>/`.
- Preserve every current prompt version, model identity, query rendering rule, retrieval/reranking depth, evidence policy, artifact filename, schema version, byte encoding, hash receipt, cache key, resume rule, and organizer-export byte contract unless a task explicitly says otherwise.
- Runtime inputs remain the official topic ID and untouched narrative only. Titles, qrels, organizer subnarratives, and organizer nuggets remain excluded.
- Any planning transport, parsing, schema, or semantic failure must retain exactly one original-narrative retrieval lane. Any canonical failure must retain deterministic exact extractive evidence.
- Keep the full-text ZIP mandatory and preserve the six conventional export files.
- Keep generated BM25 suggestions as historical/inactive plan metadata for compatibility; retrieval remains exactly the original narrative plus one full-text lane per admitted subnarrative.
- Preserve the two unrelated untracked user files; never stage or modify them.
- Do not add compatibility shims for stage CLIs that have never shipped from master. The draft PR is the migration window.
- Use TDD: the new public interface test must be observed failing before production implementation; consolidation then proceeds as refactoring under that green contract.
- Final feature surface target: 9–11 runtime modules and 6–8 focused test files. Prefer deletion over forwarding wrappers.

---

### Task 1: Introduce the one public official-run interface

**Files:**
- Create: `code/trec_rag/official_run.py`
- Create: `code/tests/test_facet_pipeline_e2e.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Produces `run_official(config: str | Path, *, topic_ids: Sequence[str] | None = None, topic_subset: Path | None = None, external: ExternalAdapters | None = None) -> RunReceipt`.
- Produces `main(argv: Sequence[str] | None = None) -> int` for `python -m trec_rag.official_run CONFIG [--topic ID ... | --topic-subset CSV]`.
- `topic_ids=None` means all official topics. An empty sequence, duplicate/unknown IDs, or mixed ID/CSV selection fails before dependency construction.
- `RunReceipt` exposes experiment ID, selected topic IDs, resumed topic IDs, and the existing `RetrievalExportReceipt`; it exposes no phase paths or scorer knobs.
- `ExternalAdapters` contains only hosted planning, Pyserini retrieval, and hosted canonicalization seams. Local scorer/similarity construction stays private.

- [ ] **Step 1: Write and run the failing public-interface test**

Create one compact no-network test importing `run_official`, using one official narrative and strict fake external/local dependencies. Assert official source ordering, forwarding of configured retrieval/reranking/nugget values, fixed output filenames, and a `RunReceipt`. Run:

```bash
PYTHONPATH=code .venv/bin/python -m pytest code/tests/test_facet_pipeline_e2e.py -q
```

Expected RED: import fails because `trec_rag.official_run` does not exist.

- [ ] **Step 2: Relocate the config-driven branch into `official_run.py`**

Move the existing `facet_pilot.main` config branch without changing call order:

```text
load strict config
  -> select narrative-only topics in official order
  -> reject dirty tracked tree
  -> validate/resume canonical seals
  -> lazily construct shared adapters only for pending topics
  -> run each pending topic
  -> require every selected canonical seal
  -> export once
  -> re-read export receipt and return RunReceipt
```

Keep an internal `_RuntimeDependencies` record and `_production_dependencies()` factory so the public signature does not expose document scorer, candidate scorer, similarity provider, cache-ignore checker, or dirty-checker plumbing. Tests may monkeypatch that private factory rather than extend the public interface.

- [ ] **Step 3: Add the thin CLI and verify GREEN**

The parser accepts one positional config path, repeatable `--topic`, and mutually exclusive `--topic-subset`; it has no `run`, `retrieve`, or `score` subcommands and no output/cache/model/depth flags. Run the Task 1 test and the existing config/pilot/export tests. Expected: PASS.

- [ ] **Step 4: Document only the supported command and commit**

Change README examples to `python -m trec_rag.official_run`. Do not yet delete old modules; Task 1 is a behavior-preserving facade and RED/GREEN checkpoint.

---

### Task 2: Consolidate structured planning and OpenRouter admission

**Files:**
- Modify: `code/trec_rag/facet_extraction.py`
- Create: `code/tests/test_facet_planning_contract.py`
- Delete: `code/trec_rag/backend_reply.py`
- Delete: `code/trec_rag/facet_query_planning.py`
- Delete: `code/trec_rag/facet_extraction_cli.py`
- Delete: `code/tests/test_facet_extraction.py`
- Delete: `code/tests/test_facet_query_planning.py`
- Delete: `code/tests/test_facet_subnarratives.py`
- Delete: `code/tests/test_openrouter_facet_backend.py`
- Delete: `code/tests/test_facet_extraction_cli.py`

**Interfaces:**
- `facet_extraction.py` owns `BackendReply`, `Subnarrative`, `GeneratedQueryPlan`, `FacetPlanningResult`, strict parsing/rendering, hosted request/receipt security, `extract_facets`, and exact fallback.
- Other runtime modules import those names only from `trec_rag.facet_extraction`.

- [ ] **Step 1: Add the consolidated planning contract while current code is green**

Create table-driven public-boundary tests retaining: valid narrative-only request; original plus ordered canonical plan rendering; title independence; duplicate keys/non-finite/wrong types; malformed/schema/semantic failure exact fallback; unsafe endpoint refusal; and literal/Unicode-escaped credential suppression. First import all plan types from `facet_extraction` and run the file. Expected RED: the unified module does not yet export every plan type.

- [ ] **Step 2: Move—not duplicate—the planning types and `BackendReply`**

Move the implementations into `facet_extraction.py`, update imports in retrieval/canonical/workflow code, and keep the existing schema/prompt bytes unchanged. Delete the three obsolete runtime modules only after `rg` confirms no runtime imports remain.

- [ ] **Step 3: Prune private-mechanics tests and verify**

Delete assertions about exact private helper names, semaphore/opener implementation, frozen payload key order, and CLI parser aliases. Run the consolidated planning contract plus retrieval, canonical, and E2E tests. Expected: PASS with exact fallback and security cases retained.

---

### Task 3: Consolidate extractive evidence behind typed stage functions

**Files:**
- Create: `code/trec_rag/facet_evidence.py`
- Create: `code/trec_rag/evidence_store.py`
- Create: `code/trec_rag/evidence_local.py`
- Create: `code/tests/test_evidence_pipeline_contract.py`
- Delete: `code/trec_rag/extractive_candidates.py`
- Delete: `code/trec_rag/extractive_candidate_selection.py`
- Delete: `code/trec_rag/extractive_candidate_scoring.py`
- Delete: `code/trec_rag/extractive_candidate_embedding.py`
- Delete: `code/trec_rag/extractive_candidate_cli.py`
- Delete: `code/trec_rag/extractive_candidate_selection_cli.py`
- Delete: `code/trec_rag/facet_canonical_handoff.py`
- Delete: their seven direct test files plus `code/tests/test_facet_candidate_bridge.py`

**Interfaces:**
- `facet_evidence.py` owns exact source-span projection, candidate records/extraction, exact grouping, clustering/MMR selection, and canonical selection records.
- `evidence_local.py` owns the pinned Mixedbread sentence scorer and MiniLM similarity adapter.
- `evidence_store.py` owns retrieval-checkpoint projection, strict evidence JSONL codecs, temporary SQLite spill, manifest/hash validation, and two typed stage calls:

```python
materialize_candidate_inputs(... ) -> HandoffArtifacts
generate_candidate_artifacts(paths: HandoffArtifacts, *, score_cache_root: Path, device: str, scorer: object | None = None) -> CandidateArtifacts
select_evidence_artifacts(paths: CandidateArtifacts, contexts: Path, *, device: str, similarity: object | None = None, policy: SelectionPolicy | None = None) -> SelectionArtifacts
```

- [ ] **Step 1: Write the unified evidence contract and observe RED**

Import records/functions from `trec_rag.evidence` and cover exact character/byte provenance, adjacent supporting sentences, source hash rejection, deterministic candidate bytes, exact-text deduplication, semantic cluster diversity, budgeted selection, and non-finite scorer/similarity rejection. Expected RED: `trec_rag.evidence` is absent.

- [ ] **Step 2: Move pure evidence and local-model code**

Move existing implementations without changing constants or serialization. Update canonicalization, retrieval export, and workflow imports. Do not leave forwarding modules.

- [ ] **Step 3: Replace argv-to-argv stage calls with typed calls**

Move the useful CLI execution bodies and handoff validation into `evidence_store.py`. Replace constructed argument arrays in the workflow with the three typed functions above. Keep current manifest-last publication and exact artifact bytes.

- [ ] **Step 4: Delete shallow suites and verify**

Retain one re-signed tamper case, one empty/original-only case, and one local-only no-network case. Remove parser aliases, write-size traps, GC/weakref checks, individual manifest-field permutations, and tests of mocks rather than outputs. Run evidence, canonical, export, retrieval, and E2E contracts. Expected: PASS.

---

### Task 4: Internalize canonical stage execution and remove the old pilot

**Files:**
- Modify: `code/trec_rag/canonical_nuggets.py`
- Modify: `code/trec_rag/official_run.py`
- Create: `code/tests/test_canonical_nugget_contract.py`
- Delete: `code/trec_rag/canonical_nugget_cli.py`
- Delete: `code/trec_rag/facet_pilot.py`
- Delete: `code/tests/test_canonical_nugget_cli.py`
- Delete: `code/tests/test_canonical_nuggets.py`
- Delete: `code/tests/test_facet_pilot.py`

**Interfaces:**
- `canonical_nuggets.py` owns canonical request construction/admission plus `run_canonical_stage(...) -> CanonicalArtifacts`; it accepts typed paths/settings and never parses argv.
- `official_run.py` owns private per-topic decomposition/retrieval/evidence/canonical checkpoint orchestration. Its private `_run_topic(topic, config, identity, dependencies)` replaces the current 18-argument public `run_topic`.

- [ ] **Step 1: Write the consolidated canonical contract**

Retain valid evidence-bound claims, configured claim/support limits, no call on empty evidence, one no-retry call for non-empty evidence, malformed/semantic failure exact-extractive fallback, cached reply revalidation, tamper rejection before hosted calls, and credential absence from persisted artifacts.

- [ ] **Step 2: Move the canonical cache/manifest loop into the domain module**

Replace the CLI parser with typed arguments, preserve raw/validated request hashes and cache paths, and call it directly from the official workflow.

- [ ] **Step 3: Move remaining workflow implementation into `official_run.py`**

Move decomposition, retrieval/scoring checkpoint orchestration, canonical orchestration, atomic manifest helpers, and audit projection. Replace the long parameter list with config plus `_RuntimeDependencies`. Delete legacy phase modes and `facet_pilot.py`; no forwarding wrapper remains.

- [ ] **Step 4: Verify public E2E and focused contracts**

Run planning, retrieval, evidence, canonical, export, config, and public E2E tests. The E2E must cover a successful one-topic no-network run/resume and a rejected plan that performs only original-narrative retrieval with zero downstream hosted canonical calls.

---

### Task 5: Prune the final surface, documentation, and tests

**Files:**
- Keep/trim: `code/tests/test_facet_retrieval.py`
- Keep/trim: `code/tests/test_retrieval_export.py`
- Keep/merge: `code/tests/test_facet_pilot_config.py` into the E2E contract when practical
- Modify: `code/trec_rag/README.md`
- Modify: `reports/facet_extraction_findings.md`
- Delete: `docs/superpowers/plans/2026-07-27-organizer-compatible-retrieval-export.md`
- Delete: `docs/superpowers/specs/2026-07-27-organizer-compatible-retrieval-export-design.md`

**Interfaces:**
- Final tests are organized by observable contracts: planning, retrieval, evidence, canonicalization, organizer export, and one public E2E/config suite.
- Historical prompt/output findings remain evidence, clearly separate from the supported runtime command.

- [ ] **Step 1: Remove tests that only freeze private layout**

Use parametrization for malformed boundary classes. Keep high-risk algorithm/security/integrity cases, especially exact fallback, non-finite/duplicate-key rejection, credential reflection, source offsets, deterministic ZIP/TREC bytes, re-signed source-chain tamper, and unsupported evidence rejection. Target 6–8 feature test files rather than one file per internal module.

- [ ] **Step 2: Make adjacent documentation match the one-interface architecture**

Document one command, the five internal stages, config blocks, conventional outputs, resume semantics, experimental model quality, and the next held-out evaluation. Remove stage CLI instructions and token-position detail. Keep reports concise and do not add rendered or generated artifacts.

- [ ] **Step 3: Run fresh verification**

```bash
PYTHONPATH=code .venv/bin/python -m pytest \
  code/tests/test_facet_planning_contract.py \
  code/tests/test_facet_retrieval.py \
  code/tests/test_evidence_pipeline_contract.py \
  code/tests/test_canonical_nugget_contract.py \
  code/tests/test_retrieval_export.py \
  code/tests/test_facet_pipeline_e2e.py -q
.venv/bin/python -m compileall -q code/trec_rag code/tests
git diff --check origin/master
```

Then run the repository suite with the known environment-sensitive `INDEX_URL` case isolated and report both commands exactly. Review the diff for secrets, absolute checkout paths, raw responses, SQLite/HTML/generated outputs, and unrelated changes.

- [ ] **Step 4: Independent review, intentional commit, and same-PR update**

Request a whole-branch spec/code review, resolve all Critical/Important findings, recount runtime/test files and lines, commit only scoped files, push `codex/facet-extraction-clean`, and update draft PR #20. The PR description must distinguish verified mechanics from unvalidated decomposition/canonicalization quality.
