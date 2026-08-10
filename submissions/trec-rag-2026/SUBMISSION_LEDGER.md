# TREC RAG 2026 submission ledger

This is the control sheet for organizer-facing submissions. It intentionally
contains no Evalbase username, email address, credentials, narratives, or
document text.

## Retrieval task

| Priority | Submission | Run tag | Status | Upload file | SHA-256 | Rows | Portal answers |
|---:|---|---|---|---|---|---:|---|
| 1 | Narrative + subnarrative | `r26-narr-facet-v1` | Submitted | [`combo/r_output_trec_rag_2026.tsv`](retrieval/cache-first-candidate-core-v1/combo/r_output_trec_rag_2026.tsv) | `29bc0c29dd51a752d49c734db456926ef94ad5202102cabd92d7fbf3e9dd15e8` | 4,246 | [`combo/EVALBASE.md`](retrieval/cache-first-candidate-core-v1/combo/EVALBASE.md) |
| 2 | Subnarrative evidence breadth | `r26-facet-breadth-v1` | Submitted | [`breadth/r_output_trec_rag_2026.tsv`](retrieval/cache-first-candidate-core-v1/breadth/r_output_trec_rag_2026.tsv) | `f42a794418cf692721adcd53df232ca0a3536d13d9cd137d90eb9821562e199d` | 4,246 | [`breadth/EVALBASE.md`](retrieval/cache-first-candidate-core-v1/breadth/EVALBASE.md) |
| 3 | Narrative only | `r26-narrative-v1` | Submitted | [`narrative/r_output_trec_rag_2026.tsv`](retrieval/cache-first-candidate-core-v1/narrative/r_output_trec_rag_2026.tsv) | `80985a42e43333085975c27d88a1da82cc11cba6dd576c3fd506839c016f2feb` | 4,246 | [`narrative/EVALBASE.md`](retrieval/cache-first-candidate-core-v1/narrative/EVALBASE.md) |

The priority order is a recommendation: submit the combo run first, breadth
second, and narrative-only third. The organizer permits up to ten runs per
task and uses the submitted priority to choose runs for manual assessment.

## RAG task

| Priority | Submission | Run ID | Status | Upload file | SHA-256 | Topics | Portal notes |
|---:|---|---|---|---|---|---:|---|
| 1 | Selected-evidence multi-stage Sol + Luna | `rag26-ms1-final` | Submitted | [`multistage/rag_output_trec_rag_2026.jsonl`](rag/selected-evidence-sol-v1/multistage/rag_output_trec_rag_2026.jsonl) | `ff384a62069062709abfe9cbfbcce2ba89c4f63ad857b917101488296e11677a` | 119 | [`multistage/EVALBASE.md`](rag/selected-evidence-sol-v1/multistage/EVALBASE.md) |
| 2 | Selected-evidence single-pass Sol | `rag26-ss1` | Submitted | [`singlepass/rag_output_trec_rag_2026.jsonl`](rag/selected-evidence-sol-v1/singlepass/rag_output_trec_rag_2026.jsonl) | `91f757eb12291c118fe21e0828f0e607bba90460616e3405bd88f62eb67ab808` | 119 | [`singlepass/EVALBASE.md`](rag/selected-evidence-sol-v1/singlepass/EVALBASE.md) |

Submit both RAG runs. The suggested priority puts the bounded multi-stage run
first and the independently generated single-pass run second; no post-hoc gold,
qrel, or RAGDoll score was used to choose that order.
Both JSONLs use the registered Evalbase organization short identifier `NP` as
`metadata.team_id`; this corresponds to the `NP Labs` organization shown by
Evalbase.

## Architecture and provenance

- [`ARCHITECTURE.md`](retrieval/cache-first-candidate-core-v1/ARCHITECTURE.md)
  explains the shared candidate core, variable-depth cutoff, cache-first
  rescoring, and the three ranking variants.
- [`metadata.json`](retrieval/cache-first-candidate-core-v1/metadata.json) is
  the compact machine-readable provenance and Evalbase answer record.
- [`retrieval-baseline-runs-manifest.json`](retrieval/cache-first-candidate-core-v1/retrieval-baseline-runs-manifest.json)
  authenticates scorer and implementation identities, topic-level cutoff
  statistics, input matrices, and run-file receipts.
- [`rag/selected-evidence-sol-v1/README.md`](rag/selected-evidence-sol-v1/README.md)
  explains the two RAG generation strategies and their shared sealed handoff.
- [`rag/selected-evidence-sol-v1/metadata.json`](rag/selected-evidence-sol-v1/metadata.json)
  records the RAG artifact hashes, generation costs, warnings, and privacy
  review without retaining prompts, evidence passages, or provider responses.

## Submission procedure

For each row above:

1. Verify the local upload file against the recorded SHA-256.
2. Upload that exact file to the matching Retrieval or RAG task.
3. Copy the dropdown and text-area responses from its `EVALBASE.md`.
4. Confirm the run tag shown by Evalbase exactly matches the Retrieval file's
   sixth column or the RAG file's `metadata.run_id`.
5. After submission, record the confirmation below in a follow-up commit.

## Evalbase confirmations

Evalbase identifies a submission by its run tag within a task rather than by a
separate numeric ID, so the run-page path is recorded as the submission ID.

| Run tag | Evalbase submission ID | Submitted at (UTC) | Final status | Notes |
|---|---|---|---|---|
| `r26-narr-facet-v1` | `/evalbase/run/trec-2026/trec2026-rag-retrieval/r26-narr-facet-v1` | 2026-08-09 (portal date; time not reported) | validation succeeded | Priority 1 confirmed by portal |
| `r26-facet-breadth-v1` | `/evalbase/run/trec-2026/trec2026-rag-retrieval/r26-facet-breadth-v1` | 2026-08-09 (portal date; time not reported) | validation succeeded | Priority 2 confirmed by portal |
| `r26-narrative-v1` | `/evalbase/run/trec-2026/trec2026-rag-retrieval/r26-narrative-v1` | 2026-08-09 (portal date; time not reported) | validation succeeded | Priority 3 confirmed by portal |
| `rag26-ms1-final` | `/evalbase/run/trec-2026/trec2026-rag-gen/rag26-ms1-final` | 2026-08-10T01:29Z (observed at submission) | validation succeeded | Uploaded `ff384a62…` at commit `f7ecd66`; priority 1 |
| `rag26-ss1` | `/evalbase/run/trec-2026/trec2026-rag-gen/rag26-ss1` | 2026-08-10T01:33Z (observed at submission) | validation succeeded | Uploaded `91f757eb…` at commit `f7ecd66`; priority 2 |
