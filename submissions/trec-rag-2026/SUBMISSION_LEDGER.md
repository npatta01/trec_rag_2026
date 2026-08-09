# TREC RAG 2026 submission ledger

This is the control sheet for organizer-facing submissions. It intentionally
contains no Evalbase username, email address, credentials, narratives, or
document text.

## Retrieval task

| Priority | Submission | Run tag | Status | Upload file | SHA-256 | Rows | Portal answers |
|---:|---|---|---|---|---|---:|---|
| 1 | Narrative + subnarrative | `r26-narr-facet-v1` | Ready to upload | [`combo/r_output_trec_rag_2026.tsv`](retrieval/cache-first-candidate-core-v1/combo/r_output_trec_rag_2026.tsv) | `29bc0c29dd51a752d49c734db456926ef94ad5202102cabd92d7fbf3e9dd15e8` | 4,246 | [`combo/EVALBASE.md`](retrieval/cache-first-candidate-core-v1/combo/EVALBASE.md) |
| 2 | Subnarrative evidence breadth | `r26-facet-breadth-v1` | Ready to upload | [`breadth/r_output_trec_rag_2026.tsv`](retrieval/cache-first-candidate-core-v1/breadth/r_output_trec_rag_2026.tsv) | `f42a794418cf692721adcd53df232ca0a3536d13d9cd137d90eb9821562e199d` | 4,246 | [`breadth/EVALBASE.md`](retrieval/cache-first-candidate-core-v1/breadth/EVALBASE.md) |
| 3 | Narrative only | `r26-narrative-v1` | Ready to upload | [`narrative/r_output_trec_rag_2026.tsv`](retrieval/cache-first-candidate-core-v1/narrative/r_output_trec_rag_2026.tsv) | `80985a42e43333085975c27d88a1da82cc11cba6dd576c3fd506839c016f2feb` | 4,246 | [`narrative/EVALBASE.md`](retrieval/cache-first-candidate-core-v1/narrative/EVALBASE.md) |

The priority order is a recommendation: submit the combo run first, breadth
second, and narrative-only third. The organizer permits up to ten runs per
task and uses the submitted priority to choose runs for manual assessment.

## Architecture and provenance

- [`ARCHITECTURE.md`](retrieval/cache-first-candidate-core-v1/ARCHITECTURE.md)
  explains the shared candidate core, variable-depth cutoff, cache-first
  rescoring, and the three ranking variants.
- [`metadata.json`](retrieval/cache-first-candidate-core-v1/metadata.json) is
  the compact machine-readable provenance and Evalbase answer record.
- [`retrieval-baseline-runs-manifest.json`](retrieval/cache-first-candidate-core-v1/retrieval-baseline-runs-manifest.json)
  authenticates scorer and implementation identities, topic-level cutoff
  statistics, input matrices, and run-file receipts.

## Submission procedure

For each row above:

1. Verify the local upload file against the recorded SHA-256.
2. Upload that exact `r_output_trec_rag_2026.tsv` to the Retrieval task.
3. Copy the dropdown and text-area responses from its `EVALBASE.md`.
4. Confirm the run tag shown by Evalbase exactly matches the run file's sixth
   column.
5. After submission, record the confirmation below in a follow-up commit.

## Evalbase confirmations

| Run tag | Evalbase submission ID | Submitted at (UTC) | Final status | Notes |
|---|---|---|---|---|
| `r26-narr-facet-v1` | — | — | Not yet submitted | — |
| `r26-facet-breadth-v1` | — | — | Not yet submitted | — |
| `r26-narrative-v1` | — | — | Not yet submitted | — |
