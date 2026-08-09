# Evalbase response: subnarrative evidence breadth

Use this sheet with the exact sibling file `r_output_trec_rag_2026.tsv`.

## Upload identity

| Field | Response |
|---|---|
| Submission file | `breadth/r_output_trec_rag_2026.tsv` |
| SHA-256 | `f42a794418cf692721adcd53df232ca0a3536d13d9cd137d90eb9821562e199d` |
| Runtag | `r26-facet-breadth-v1` |
| Manual or automatic | `automatic` |
| Standard retrieval or pool enrichment | `real run` |
| Agents used during development | `Yes` |
| Pyserini REST API, custom system, or both | `combination of both` |
| Retrieval-pipeline category | `sparse/lexical` |
| Single-step, multi-stage, or iterative agentic | `multi-stage` |
| Uses neural networks | `Yes` |
| Uses proprietary models in the retrieval pipeline | `No` |
| Uses open-weight models in the retrieval pipeline | `Yes` |
| Priority for manual assessment | `2` |

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
> Ranking: The final ordering prioritized evidence breadth: first the number of distinct supported subnarratives, then the number of strong overlap-suppressed supporting passages, and then the narrative-plus-subnarrative combo score.
>
> Development disclosure: Codex agents assisted with design, implementation, review, and validation, but did not select documents at runtime. The retrieval pipeline used no proprietary model. Hosted services provided inference for open-weight models.
