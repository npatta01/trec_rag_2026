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

> For each narrative, we applied a robust threshold independently to the original-narrative source lane and every authenticated subnarrative source lane. For lane u, the median and MAD were computed among documents present in that lane. A document was admitted when its aggregate score was at least median(A_u) + 2.5 x 1.4826 x MAD(A_u); when MAD was zero, its score had to exceed the median. We submitted the union of those admissions. If the union had been empty, we would have retained one original-narrative argmax with deterministic retrieval-rank and document-ID ties. This produced variable k from 1 to 121 across 119 narratives (61 distinct depths), with no padding, truncation, or fallback topics. All three submitted variants share this selected set.

## Short description of this run

Copy and paste:

> Automatic multi-stage real run. A bounded, one-shot open-weight DeepSeek V4 Flash planner decomposed each narrative into subnarratives. Candidates came from the organizer's Pyserini REST API over ClimbMix-400b and were passage-scored with the open-weight Mixedbread mxbai-rerank-base-v2 cross-encoder. Each authenticated subnarrative text lane was treated as one pooled source. The final ordering prioritizes evidence breadth: first the number of distinct supported subnarratives, then the number of strong overlap-suppressed supporting passages, then the narrative-plus-subnarrative combo score. Codex agents assisted development through design, implementation, review, and validation; they did not select documents at runtime. The retrieval pipeline used no proprietary model. Hosted services provided inference for open-weight models.
