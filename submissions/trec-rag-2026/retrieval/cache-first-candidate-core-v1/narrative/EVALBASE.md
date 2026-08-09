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

> For each narrative, we applied a robust threshold independently to the original-narrative source lane and every authenticated subnarrative source lane. For lane u, the median and MAD were computed among documents present in that lane. A document was admitted when its aggregate score was at least median(A_u) + 2.5 x 1.4826 x MAD(A_u); when MAD was zero, its score had to exceed the median. We submitted the union of those admissions. If the union had been empty, we would have retained one original-narrative argmax with deterministic retrieval-rank and document-ID ties. This produced variable k from 1 to 121 across 119 narratives (61 distinct depths), with no padding, truncation, or fallback topics. All three submitted variants share this selected set.

## Short description of this run

Copy and paste:

> Automatic multi-stage real run. A bounded, one-shot open-weight DeepSeek V4 Flash planner decomposed each narrative into subnarratives. Candidates came from the organizer's Pyserini REST API over ClimbMix-400b and were passage-scored with the open-weight Mixedbread mxbai-rerank-base-v2 cross-encoder. Each authenticated subnarrative text lane was treated as one pooled source. The selected candidate set is shared with the other variants, but this run orders documents only by their normalized score against the untouched narrative, followed by deterministic retrieval-rank and document-ID ties. Codex agents assisted development through design, implementation, review, and validation; they did not select documents at runtime. The retrieval pipeline used no proprietary model. Hosted services provided inference for open-weight models.
