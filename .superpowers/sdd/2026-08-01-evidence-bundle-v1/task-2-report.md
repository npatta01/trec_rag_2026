# Task 2 Report — Evidence Bundle v1

Status: DONE

Implemented deterministic downstream projections from `EvidenceBundle` in
`code/trec_rag/evidence_bundle.py`, added focused projection coverage in
`code/tests/test_evidence_bundle.py`, and documented the new projection surface
in `code/trec_rag/README.md`.

What changed:
- Added `EvidenceBundle.to_trec_run(selection_id=...)` to emit deterministic
  six-column TREC rows from an explicit selection, with ordinal scores and no
  implicit top-100 truncation.
- Added `EvidenceBundle.to_document_records(selection_id=...)` to emit the
  organizer-compatible `query`/`candidates` JSONL core while preserving lane
  membership and document-hash metadata as extension fields.
- Added `EvidenceBundle.to_fixed_rag_context(selection_id=...)` to emit ordered
  per-document fixed-bundle context records containing the selected document,
  exact evidence spans, and supported nuggets without performing retrieval.
- Added `EvidenceBundle.write_fixed_rag_inputs(output_dir, selection_id=...)`
  to publish:
  - `r_output_trec_rag_2026.tsv`
  - `retrieval_with_text.jsonl`
  - deterministic `retrieval_with_text.jsonl.zip`
  - `fixed_rag_context.jsonl`
- Reused deterministic JSONL/ZIP writing so the organizer sidecar stays byte
  stable across repeated writes.

Test-first evidence:
- Added the projection tests first.
- Confirmed the red phase with:
  `pytest code/tests/test_evidence_bundle.py -q -k 'to_trec_run or to_document_records or to_fixed_rag_context or write_fixed_rag_inputs'`
  which failed with missing-method `AttributeError`s before implementation.

Verification:
- `pytest code/tests/test_evidence_bundle.py -q -k 'to_trec_run or to_document_records or to_fixed_rag_context or write_fixed_rag_inputs'`
  → `5 passed, 9 deselected`
- `code/tools/setup_env.sh`
  → created the repo-local `.venv` so the contract suite could run with the
  project dependencies instead of the host interpreter
- `.venv/bin/python -m pytest code/tests/test_evidence_bundle.py code/tests/test_competition_rag.py code/tests/test_retrieval_export.py -q`
  → `186 passed`

Concerns:
- Selection ordering is derived deterministically from the best retained
  retrieval event among the selection's source lanes because `BundleSelection`
  currently stores document membership but not an explicit per-selection rank
  ledger. That matches the new tests and keeps projections stable, but if a
  later task needs richer selection provenance (for example, rejection reasons
  or an explicit chosen-from-lane record), the bundle schema will need to carry
  it directly rather than reconstructing it here.
