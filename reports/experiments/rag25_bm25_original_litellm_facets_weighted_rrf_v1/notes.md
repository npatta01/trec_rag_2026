# RAG25 BM25 Original + LiteLLM Facets Weighted RRF V1

Run date: 2026-07-05

## Summary

This experiment repairs the regression observed in the first title/facet RRF
run by restoring the full narrative query as the primary BM25 stream and keeping
LLM-generated facets as auxiliary retrieval streams. The run uses local
LiteLLM/vLLM facet generation, hosted Pyserini BM25 over `climbmix-400b`,
weighted RRF fusion, and projected development qrels for diagnostics.

| run | ndcg@10 | recall@100 |
|---|---:|---:|
| original narrative BM25 baseline | 0.4139611685 | 0.1080879457 |
| title + facets unweighted RRF | 0.2928717959 | 0.0419527453 |
| original + facets weighted RRF | 0.4294133095 | 0.1080879457 |

The repaired run fully recovers the `recall@100` loss from the title/facet
experiment and improves `ndcg@10` by 0.0154521409 over the narrative-only
baseline.

## What Changed

The regressed run used a derived `title` stream plus LLM facets. Because the TSV
development topics do not contain real titles, the pipeline derived titles from
the first 12 words of each narrative. Those derived titles were often vague
introductory fragments, and unweighted RRF let many short facet streams compete
too strongly against the original information need.

This experiment removes the derived title stream and restores `original_topic`
as the anchor query. It keeps the same LiteLLM facet generation path, but treats
facets as supporting evidence rather than peer streams. Weighted RRF assigns
`original: 4.0` and `facets: 0.25`, so strong narrative BM25 hits dominate while
documents rediscovered by useful facets can still receive a modest boost.

The implementation also adds general weighted RRF support to the pipeline:
`ranking.stream_weights` can now specify per-query-variant weights, defaulting
to 1.0 for variants not listed.

## Pipeline Shape

For each of the 22 development topics, the pipeline builds one full narrative
BM25 query and up to eight LiteLLM-generated facet queries. Both streams search
the hosted Pyserini ClimbMix index with `hits: 100`, using a small sequential
request delay to avoid hosted-service rate limits. The retrieval stage produced
187 query variants and 18,700 raw retrieved rows. Weighted RRF deduplicated and
fused those rows into 17,587 final runfile rows.

The run still uses placeholder RAG generation. The reported gains are retrieval
and ranking gains, not answer-generation gains.

## Artifacts

- Manifest: `manifest.yaml`
- Exact runtime config: `config.yaml`
- Metrics JSON: `metrics.json`
- Per-topic scores: `topic_scores.csv`
- Local output directory:
  `outputs/rag25_bm25_original_litellm_facets_weighted_rrf_v1`
- Local runfile:
  `outputs/rag25_bm25_original_litellm_facets_weighted_rrf_v1/r_output_trec_rag_2026.tsv`
- Local RAG placeholder output:
  `outputs/rag25_bm25_original_litellm_facets_weighted_rrf_v1/rag_output_trec_rag_2026.jsonl`

## Observations

The result confirms that the previous regression came from query-stream design,
not from the local vLLM/LiteLLM stack. The original narrative query carries most
of the recall, while facets are useful only when constrained as auxiliary
signals. The next tuning axis is the weight pair: `original: 4.0` and
`facets: 0.25` is intentionally conservative and preserves recall, but a small
grid over facet weights may find more nDCG improvement without reintroducing the
top-100 recall loss.
