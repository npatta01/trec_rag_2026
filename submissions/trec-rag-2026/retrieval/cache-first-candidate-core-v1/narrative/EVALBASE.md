# Evalbase response: narrative only

Use this sheet with the exact sibling file `r_output_trec_rag_2026.tsv`.

## Upload identity

| Field | Response |
|---|---|
| Submission file | `narrative/r_output_trec_rag_2026.tsv` |
| SHA-256 | `80985a42e43333085975c27d88a1da82cc11cba6dd576c3fd506839c016f2feb` |
| Runtag | `r26-narrative-v1` |
| Manual or automatic | `automatic` |
| Standard retrieval or pool enrichment | `real run` |
| Agents used during development | `Yes` |
| Pyserini REST API, custom system, or both | `combination of both` |
| Retrieval-pipeline category | `sparse/lexical` |
| Single-step, multi-stage, or iterative agentic | `multi-stage` |
| Uses neural networks | `Yes` |
| Uses proprietary models in the retrieval pipeline | `No` |
| Uses open-weight models in the retrieval pipeline | `Yes` |
| Priority for manual assessment | `3` |

## How the k documents were selected

Copy and paste:

> Each narrative was allowed to return its own number of documents.
>
> We examined the original narrative and each subnarrative separately. Within each source lane, the median represented a typical document score and the median absolute deviation (MAD) represented normal score variation. We kept documents whose scores were unusually strong for that lane.
>
> Exact rule: when MAD was greater than zero, we kept a document if its aggregate score was at least median(A_u) + 2.5 x 1.4826 x MAD(A_u). When MAD was zero, its score had to exceed the median.
>
> We then combined the documents kept by any lane. If no lane had kept a document, we would have returned the highest-scoring original-narrative document, with deterministic retrieval-rank and document-ID tie-breaking.
>
> This produced a different k for different narratives: 1 to 121 documents across 119 narratives, with 61 distinct depths. No topic was padded or truncated, and no topic needed the fallback. All three submitted variants use this same document set.

## Short description of this run

Copy and paste:

> Planning: A bounded, one-shot, open-weight DeepSeek V4 Flash planner decomposed each narrative into subnarratives.
>
> Retrieval and scoring: This was an automatic, multi-stage real run. Candidates came from the organizer's Pyserini REST API over ClimbMix-400b. The open-weight Mixedbread mxbai-rerank-base-v2 cross-encoder scored passages. Each authenticated subnarrative text lane was treated as one pooled source.
>
> Ranking: The selected document set was shared with the other variants, but this run ordered documents only by their normalized score against the untouched narrative. Retrieval rank and document ID provided deterministic tie-breaking.
>
> Development disclosure: Codex agents assisted with design, implementation, review, and validation, but did not select documents at runtime. The retrieval pipeline used no proprietary model. Hosted services provided inference for open-weight models.
