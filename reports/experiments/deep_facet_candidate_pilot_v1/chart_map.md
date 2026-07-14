# Deep-facet candidate report chart map

This supporting note records the visual contract for the canonical HTML report.
It is not a second report surface.

| Report segment | Analytical question | Family / chart | Fields | Supported claim | Palette and non-color policy |
|---|---|---|---|---|---|
| Candidate discovery | Did facet decomposition add judged relevant documents? | Comparison / grouped bar | topic, candidate_set, relevant_documents | The accepted facet union adds relevant candidates on all four topics. | Relaxed multi-category roots; x labels and legend identify each series without color alone. |
| Early precision | Which frozen ranking best preserves head relevance? | Comparison / bar | arm, ndcg10 | Protecting RRF ranks 1–10 preserves the strongest observed nDCG@10. | Single-root preferred; arm labels carry identity. |
| Novel evidence retention | How quickly does each ranking surface facet-only relevant evidence? | Comparison / grouped bar | arm, depth, retention | Complete rankings differ in how quickly novel evidence reaches ranks 500 and 1,000. | Hard two-root cap for the two depths; legend and depth labels remain available. |
| Historical topic regression | Where did DUAL lose early precision versus RRF? | Comparison / signed bar | topic, ndcg10_delta | The original DUAL tradeoff was not isolated to one topic. | Hard two-root signed policy with zero-line context and signed values; no green/red semantics. |
| Targeted Mixedbread stability | Does the reranker improve graded Recall@500 consistently by topic? | Comparison / signed bar | topic, graded_recall500_delta, candidate_count | Per-topic deltas distinguish a stable repair from an aggregate-only gain; all complete candidates remain retained. | Hard two-root signed policy with zero-line context; topic labels and signed values carry meaning without color. |

All charts are native canonical-artifact charts rendered inside the portable HTML
reader. Each has an adjacent narrative block, a bounded SQLite-backed dataset,
and canonical source metadata. Final QA is the packaged desktop/narrow browser
verification rather than a separate static plot.
