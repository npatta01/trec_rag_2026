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
- Review fix: fixed-bundle RAG context now filters evidence to spans whose
  `lane_ids` intersect the selection's `source_lane_ids`, includes nuggets only
  when selected support remains, and projects nugget `evidence_ids` to the
  retained support subset.
- Review fix: `BundleSelection.document_ids` now preserves explicit ordered
  selection provenance. Validation keeps `input_document_ids` as sorted natural
  union IDs, requires output `document_ids` to be unique, and downstream
  projections now follow that preserved order directly.

Test-first evidence:
- Added the original projection tests first.
- Confirmed the initial red phase with:
  `pytest code/tests/test_evidence_bundle.py -q -k 'to_trec_run or to_document_records or to_fixed_rag_context or write_fixed_rag_inputs'`
  which failed with missing-method `AttributeError`s before implementation.
- Added review-fix regressions first and confirmed the red phase with:
  `.venv/bin/python -m pytest code/tests/test_evidence_bundle.py -q -k 'selection_document_order or duplicate_selection_output or to_trec_run_uses_explicit_selection or to_document_records_emits_organizer_core or to_fixed_rag_context_orders_documents or filters_support_to_selected_source_lanes or write_fixed_rag_inputs'`
  which failed against the pre-fix implementation because selection outputs
  were reconstructed from retrieval events and fixed context still projected
  unselected support.

Verification:
- `pytest code/tests/test_evidence_bundle.py -q -k 'to_trec_run or to_document_records or to_fixed_rag_context or write_fixed_rag_inputs'`
  → `5 passed, 9 deselected`
- `code/tools/setup_env.sh`
  → created the repo-local `.venv` so the contract suite could run with the
  project dependencies instead of the host interpreter
- `.venv/bin/python -m pytest code/tests/test_evidence_bundle.py code/tests/test_competition_rag.py code/tests/test_retrieval_export.py -q`
  → `186 passed`
- `.venv/bin/python -m pytest code/tests/test_evidence_bundle.py -q -k 'selection_document_order or duplicate_selection_output or to_trec_run_uses_explicit_selection or to_document_records_emits_organizer_core or to_fixed_rag_context_orders_documents or filters_support_to_selected_source_lanes or write_fixed_rag_inputs'`
  → `7 passed, 10 deselected`
- `.venv/bin/python -m pytest code/tests/test_evidence_bundle.py code/tests/test_competition_rag.py code/tests/test_retrieval_export.py -q`
  → `189 passed`

Concerns:
- Selection output ordering is now explicit and preserved, but richer
  per-selection provenance such as rejection reasons or chosen-from-lane
  metadata is still outside `BundleSelection`. If a later task needs that
  attribution, the schema will need additional fields rather than another
  reconstruction step.
