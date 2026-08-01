# Task 4 Report — Evidence Bundle v1

Status: DONE

Implemented the focused whole-branch review response in
`code/trec_rag/evidence_bundle.py`, `code/tests/test_evidence_bundle.py`, and
`code/trec_rag/README.md` without hosted calls and without changing sealed
stage checkpoint formats.

## Contract changes

- Evidence spans now require non-empty lane support; nuggets require non-empty
  evidence support; optional nugget subnarratives must resolve to a known lane
  whose kind is `subnarrative`.
- Added frozen `BundleSelectionMember` audit records with document ID,
  inclusion state, optional output rank, and optional rejection reason.
  Selection document IDs and counts are derived from members and validated
  against their serialized audit summary.
- The `natural_union` selection is preserved as a sorted, unranked relation.
  It cannot be emitted through TREC, document, or fixed-context projections;
  named ranked selections require contiguous explicit output ranks.
- Bundle lanes now carry and validate `query_text_sha256`.
- Ordinary tabs, line feeds, and carriage returns are accepted in source text;
  unsafe control characters remain rejected.
- v1 decoding requires exact top-level and relation-row key sets, including
  empty evidence/nugget/trace relations, and rejects Boolean values where
  integer or numeric scalars are required.
- Added `write_fixed_rag_package(bundles, output_dir, selection_id=...)` for one
  deterministic multi-topic query TSV, six-column TREC run, document JSONL and
  deterministic ZIP, and query-aware fixed-context JSONL package. The package
  uses the existing retrieval exporter TREC-byte validator.
- Restored the complete model-quality warning in the README and removed the
  duplicate `rank` key from the organizer projection test.

## Test-first evidence

The Task 4 regressions were written before production changes. The first
focused run was:

```text
.venv/bin/python -m pytest code/tests/test_evidence_bundle.py -q
14 failed, 19 passed
```

The failures covered unsupported evidence/nuggets, invalid subnarrative joins,
missing selection members, natural-union ranking, missing query hashes,
multiline rejection, missing v1 relations, accepted Boolean numerics, and the
missing multi-topic package API.

After implementation and fixture migration:

```text
.venv/bin/python -m pytest code/tests/test_evidence_bundle.py -q
33 passed in 0.09s
```

## Verification

```text
.venv/bin/python -m pytest \
  code/tests/test_evidence_bundle.py \
  code/tests/test_competition_rag.py \
  code/tests/test_retrieval_export.py -q
205 passed in 0.90s

.venv/bin/python -m py_compile \
  code/trec_rag/evidence_bundle.py \
  code/tests/test_evidence_bundle.py
exit 0

git diff --check
exit 0
```

A separate local two-topic package run wrote the package twice with reversed
bundle input order, compared every output byte, and loaded the query TSV, TREC
run, and document ZIP through the existing fixed-RAG readers. Results:

- deterministic equality: true
- topic/query order: `224`, `225`
- ranked documents for both topics: `doc-a`, `doc-c`
- fixed-context query IDs: `224`, `224`, `225`, `225`
- ZIP member timestamp: `1980-01-01 00:00:00`; mode: `0100600`
- SHA-256:
  - queries: `df4243cf0a2e92adda352803f4c76806d5290b8a847636d2a5f8ac83a8ebe03a`
  - run: `bcf2135aefec7b10f401b22c4ffd9a3bb16fe35125d015cce80d110b06017340`
  - documents JSONL: `da5c7717a1cc6a4a5d8d5441eaff1626976d9608154066f9ae91fc5206a0371d`
  - documents ZIP: `481e7cff49880d1e402307747fec1dcaf8ed595cc9f4447e77050ab84f91db01`
  - context JSONL: `eb528f8eb7334940409a93d3976bfe6cc5b016f751455d9e23df2d7bba149b3d`

Repository-wide pytest cannot fully collect because this environment lacks
`opentelemetry`, affecting the existing organizer/Phoenix tracing tests. With
the two direct collection failures ignored, the broad run reached `777 passed,
19 skipped` and one remaining failure that imports the same missing tracing
dependency. No Task 4 test failed.

## Remaining concerns

- The whole-branch review's P3 serializer-consolidation judgment remains a
  possible follow-up. This focused task reuses the existing emitted-TREC
  validator but does not move organizer JSONL/ZIP/atomic-write helpers into a
  new shared module, avoiding an unrelated cross-module refactor.
- `ruff` is not installed in the repo environment, so no Ruff run was
  available; Python compilation and whitespace checks passed.
