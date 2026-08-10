# Evalbase notes: single-pass RAG

Use this sheet with the exact sibling file
`rag_output_trec_rag_2026.jsonl`.

## Upload identity

| Field | Response |
|---|---|
| Task | Retrieval-Augmented Generation (`RAG`) |
| Submission file | `singlepass/rag_output_trec_rag_2026.jsonl` |
| SHA-256 | `a33c60325c198cf90178d5b4c6c1b04209f7d39de278736b6817a8c5666c02a0` |
| Runtag | `rag26-ss1` |
| Manual or automatic | `automatic` |
| Submission purpose | `real run` |
| Agents used during development | `Yes` |
| Suggested priority for manual assessment | `2` |

## Screenshot field values

| Evalbase field | Select or enter |
|---|---|
| Pyserini REST API, custom retrieval system/index, or combination | `combination of both` |
| Retrieval pipeline category | `sparse/lexical` |
| Retrieval staging | `multi-stage` |
| Neural networks in retrieval | `Yes` |
| Proprietary models in retrieval | `No` |
| Open-weight models in retrieval | `Yes` |
| Proprietary models in generation | `Yes` |
| Open-weight models in generation | `No` |

## Short description

Copy and paste if the form requests a run description:

> Automatic RAG over 119 TREC RAG 2026 narratives using the same sealed
> selected-evidence handoff as the multi-stage run. Retrieval used a
> combination of the organizer Pyserini REST API over ClimbMix-400b and custom
> local evidence selection and reranking. The retrieval pipeline was
> multi-stage and used the open-weight DeepSeek V4 Flash planner and
> Mixedbread mxbai-rerank-base-v2; it used no proprietary retrieval model.
> Generation used proprietary `openai/gpt-5.6-sol` in a single pass over the
> complete selected-evidence context. Codex agents assisted development and
> monitoring. The final records contain sentence-level ClimbMix citations and
> passed organizer validation.

The authenticated Evalbase form is not visible without Login.gov. Confirm its
current field labels after signing in; do not alter the file or run ID.
