# Task 3 Report — Evidence Bundle v1

Status: DONE

Implemented the Task 3 boundary review scope for the evidence bundle without
changing any existing stage-local schemas and without running hosted calls.

What changed:
- Added a top-level bundle envelope marker,
  `schema_version: evidence_bundle_v1`, in
  `code/trec_rag/evidence_bundle.py`.
- Added a fail-closed compatibility check so `EvidenceBundle.from_dict(...)`
  rejects incompatible bundle schema versions with
  `ValueError("unsupported evidence bundle schema version")`.
- Kept the internal bundle payload shape unchanged: `topic_id`,
  `natural_document_count`, `lanes`, `documents`, `retrieval_events`,
  `selections`, `evidence`, `nuggets`, and `trace_refs` are still serialized
  exactly as before, now under the same top-level dict with the added marker.
- Added focused fixture coverage in `code/tests/test_evidence_bundle.py`
  proving:
  - the version marker is emitted on round-trip serialization
  - incompatible schema versions are rejected on deserialization
  - narrative, subnarrative, and synthetic agentic lanes all compile through
    `EvidenceBundle.from_retrieval_rows(...)` to the same bundle schema
- Updated `code/trec_rag/README.md` to document the bundle as the cross-stage
  contract, the fail-closed schema marker, and the distinction between fixed
  downstream consumers versus agentic consumers that must publish a new bundle
  revision when they retrieve new evidence

Test-first evidence:
- Added the Task 3 tests before implementation.
- Confirmed the red phase with:
  `.venv/bin/python -m pytest code/tests/test_evidence_bundle.py -q -k 'schema_version or lane_kinds_compile'`
  which failed because the bundle emitted no `schema_version` key and
  `EvidenceBundle.from_dict(...)` accepted the incompatible version payload.
- Re-ran the same focused command after implementation and observed:
  `2 passed, 17 deselected`

Verification:
- `.venv/bin/python -m pytest code/tests/test_evidence_bundle.py -q -k 'schema_version or lane_kinds_compile'`
  → `2 passed, 17 deselected`
- `.venv/bin/python -m pytest code/tests/test_evidence_bundle.py code/tests/test_competition_rag.py code/tests/test_retrieval_export.py -q`
  → `191 passed`
- `git diff --check`
  → clean

Scoped diff review:
- The implementation only changes:
  - `code/trec_rag/evidence_bundle.py`
  - `code/tests/test_evidence_bundle.py`
  - `code/trec_rag/README.md`

Unimplemented work intentionally left out of Task 3:
- No artifact compiler/adapters were added beyond the already existing bundle
  projections from Task 2.
- No hosted retrieval, nuggetization, generation, or agentic execution was
  run or added.
- No existing stage-local checkpoint or schema format was changed.

Concerns:
- The bundle compatibility policy is currently one strict accepted version
  (`evidence_bundle_v1`). If a future migration needs to read older or newer
  envelopes, that compatibility table will need to be added explicitly rather
  than inferred.
