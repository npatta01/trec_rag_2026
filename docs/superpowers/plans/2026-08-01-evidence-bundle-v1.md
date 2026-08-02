# Evidence Bundle v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a validated, arm-neutral evidence bundle and deterministic downstream projections without hosted calls.

**Architecture:** Keep existing sealed stage schemas as inputs, and introduce `trec_rag.evidence_bundle` as the only new cross-stage boundary. The bundle stores normalized lanes, unique documents, pre-dedup retrieval events, derived selections, exact evidence spans, provenance-linked nuggets, and hashed agent-trace references. Adapters project it into organizer retrieval inputs and fixed-bundle RAG context.

**Tech Stack:** Python 3.10+, frozen dataclasses, canonical JSON, JSONL, ZIP, pytest.

## Global Constraints

- Preserve the natural deduplicated document union; do not enforce exactly 100 documents.
- Preserve all lane/document memberships and pre-dedup retrieval events.
- Validate every text hash and every evidence/nugget join.
- Keep existing sealed stage schemas and checkpoint formats unchanged.
- Do not run hosted retrieval, nuggetization, generation, or agentic calls.
- Keep generated bundle data out of git; tests use small in-memory fixtures.

---

### Task 1: Define and validate the canonical bundle

**Files:**
- Create: `code/trec_rag/evidence_bundle.py`
- Test: `code/tests/test_evidence_bundle.py`

**Interfaces:**
- `EvidenceBundle`, `BundleLane`, `BundleDocument`, `RetrievalEvent`, `BundleSelection`, `EvidenceSpan`, `BundleNugget`, and `TraceReference` are frozen records.
- `EvidenceBundle.validate()` checks identifiers, hashes, joins, lane memberships, and natural-union counts.
- `EvidenceBundle.to_dict()` and `EvidenceBundle.from_dict()` provide deterministic JSON-compatible serialization.
- `EvidenceBundle.from_retrieval_rows(...)` compiles neutral retrieval rows while preserving multiple lane memberships.

- [x] **Step 1: Write failing tests** for multi-lane document membership, natural-union counts, exact span validation, nugget support validation, and serialization round-trip.
- [x] **Step 2: Run `pytest code/tests/test_evidence_bundle.py -q` and confirm the missing module/API failures.**
- [x] **Step 3: Implement the frozen records, canonical hashing, compilation, validation, and serialization.**
- [x] **Step 4: Re-run the focused tests and then `pytest code/tests/test_evidence_bundle.py -q`.**

### Task 2: Add deterministic downstream projections

**Files:**
- Modify: `code/trec_rag/evidence_bundle.py`
- Test: `code/tests/test_evidence_bundle.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- `EvidenceBundle.to_trec_run()` emits deterministic six-column TREC rows from an explicitly named selection view.
- `EvidenceBundle.to_document_records()` emits the unique document text sidecar without losing lane membership metadata.
- `EvidenceBundle.to_fixed_rag_context()` emits ordered document/evidence/nugget context records and never performs retrieval.
- `EvidenceBundle.write_fixed_rag_inputs(output_dir, selection_id=...)` writes a TREC run and JSONL/ZIP-compatible document sidecar.

- [x] **Step 1: Add failing projection tests for deterministic ordering, natural-union preservation, and fixed-bundle context output.**
- [x] **Step 2: Run the focused projection tests and confirm they fail before implementation.**
- [x] **Step 3: Implement projections with explicit selection IDs and no implicit top-100 truncation.**
- [x] **Step 4: Run the focused tests and the existing retrieval/RAG contract tests.**

### Task 3: Review and document the boundary

**Files:**
- Modify: `code/trec_rag/README.md`
- Test: `code/tests/test_evidence_bundle.py`

- [x] **Step 1: Add a versioned bundle envelope marker and reject payloads with an incompatible schema version.**
- [x] **Step 2: Document the bundle as the cross-stage contract and describe fixed versus agentic downstream consumption.**
- [x] **Step 3: Add a fixture test showing narrative, subnarrative, and synthetic agentic lanes compile to the same schema.**
- [x] **Step 4: Run the full targeted suite: `pytest code/tests/test_evidence_bundle.py code/tests/test_competition_rag.py code/tests/test_retrieval_export.py -q`.**
- [x] **Step 5: Inspect the diff and report any unimplemented artifact compiler or hosted-run work separately.**

### Task 4: Close whole-branch contract gaps

**Files:**
- Modify: `code/trec_rag/evidence_bundle.py`
- Modify: `code/tests/test_evidence_bundle.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Evidence spans require non-empty lane support; nuggets require non-empty evidence support; nugget subnarratives resolve to known `subnarrative` lanes.
- `BundleSelection` carries explicit per-document audit members with inclusion, output rank, and rejection reason; the natural union remains non-ranked and cannot be projected as a ranked run.
- `write_fixed_rag_package(bundles, output_dir, selection_id=...)` writes one deterministic multi-topic query TSV, TREC run, document JSONL/ZIP, and fixed-context JSONL package.
- Bundle lanes carry query-text hashes; multiline document/query text is valid; v1 deserialization is fail-closed on missing relation keys and Boolean scalar fields.

- [x] **Step 1: Write failing regression tests for mandatory provenance, explicit selection audit members, natural-union projection rejection, multiline text, strict payload keys/types, and multi-topic package output.**
- [x] **Step 2: Run the focused tests and confirm the expected failures.**
- [x] **Step 3: Implement the contract fixes and multi-topic package adapter without hosted calls.**
- [x] **Step 4: Restore the complete model-quality warning and remove the duplicate test dictionary key.**
- [x] **Step 5: Run the full targeted suite and inspect generated package bytes for deterministic output.**
