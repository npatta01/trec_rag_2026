# Mixedbread pointwise vs FIRST listwise benchmark

## Result

Use the current Mixedbread pointwise reranker for exhaustive pruning, then FIRST listwise reranking only on its final top 100. Preserve per-facet quotas before the global listwise stage because nDCG does not measure answer-evidence diversity. If operational constraints permit only one reranker, choose Mixedbread: FIRST cannot recover relevant documents or facets excluded from its input pool.

FIRST increased nDCG at every measured cutoff under all three qrel sets. None of the paired improvements reaches p < 0.05 on only 22 topics, so this is consistent directional evidence rather than a conclusive significance result.

## Direct controlled comparison

| Qrels | Metric | Mixedbread | Mixedbread + FIRST | Delta | Relative | 95% bootstrap CI | p |
|---|---|---|---|---|---|---|---|
| codex_gpt5_5_medium | ndcg@10 | 0.408036 | 0.417329 | +0.009294 | +2.28% | [-0.050229, +0.068773] | 0.76646 |
| codex_gpt5_5_medium | ndcg@20 | 0.386058 | 0.399459 | +0.013401 | +3.47% | [-0.019329, +0.048213] | 0.45125 |
| codex_gpt5_5_medium | ndcg@100 | 0.322006 | 0.324476 | +0.002471 | +0.77% | [-0.008065, +0.013298] | 0.66813 |
| qwen3_5_9b | ndcg@10 | 0.377752 | 0.383857 | +0.006105 | +1.62% | [-0.053832, +0.063811] | 0.84446 |
| qwen3_5_9b | ndcg@20 | 0.380709 | 0.387834 | +0.007125 | +1.87% | [-0.029076, +0.043667] | 0.71095 |
| qwen3_5_9b | ndcg@100 | 0.354552 | 0.356217 | +0.001666 | +0.47% | [-0.008228, +0.011582] | 0.75128 |
| ministral_3_14b | ndcg@10 | 0.509586 | 0.551686 | +0.042100 | +8.26% | [-0.023037, +0.103485] | 0.21572 |
| ministral_3_14b | ndcg@20 | 0.489016 | 0.517989 | +0.028973 | +5.92% | [-0.005856, +0.062770] | 0.11856 |
| ministral_3_14b | ndcg@100 | 0.401385 | 0.408005 | +0.006619 | +1.65% | [-0.003142, +0.015918] | 0.19232 |

## Aggregate metrics

| Qrels | System | ndcg@10 | ndcg@20 | ndcg@100 | recall@100 | judged_rate@10 | judged_rate@100 |
|---|---|---|---|---|---|---|---|
| codex_gpt5_5_medium | bm25 | 0.413961 | 0.413107 | 0.410562 | 0.108088 | 1.000000 | 1.000000 |
| codex_gpt5_5_medium | current_mixedbread_pointwise_1000 | 0.408036 | 0.386058 | 0.322006 | 0.076063 | 0.677273 | 0.545000 |
| codex_gpt5_5_medium | current_mixedbread_first_listwise | 0.417329 | 0.399459 | 0.324476 | 0.076063 | 0.659091 | 0.545000 |
| qwen3_5_9b | bm25 | 0.423757 | 0.438359 | 0.487590 | 0.097166 | 1.000000 | 1.000000 |
| qwen3_5_9b | current_mixedbread_pointwise_1000 | 0.377752 | 0.380709 | 0.354552 | 0.069446 | 0.677273 | 0.545000 |
| qwen3_5_9b | current_mixedbread_first_listwise | 0.383857 | 0.387834 | 0.356217 | 0.069446 | 0.659091 | 0.545000 |
| ministral_3_14b | bm25 | 0.542493 | 0.524015 | 0.523484 | 0.093856 | 1.000000 | 1.000000 |
| ministral_3_14b | current_mixedbread_pointwise_1000 | 0.509586 | 0.489016 | 0.401385 | 0.059032 | 0.677273 | 0.545000 |
| ministral_3_14b | current_mixedbread_first_listwise | 0.551686 | 0.517989 | 0.408005 | 0.059032 | 0.659091 | 0.545000 |

## Paired nDCG comparisons

| Qrels | Metric | Baseline | Candidate | Mean delta | 95% bootstrap CI | p | W/T/L |
|---|---|---|---|---|---|---|---|
| codex_gpt5_5_medium | ndcg@10 | bm25 | current_mixedbread_pointwise_1000 | -0.005926 | [-0.102294, +0.100329] | 0.91756 | 7/0/15 |
| codex_gpt5_5_medium | ndcg@20 | bm25 | current_mixedbread_pointwise_1000 | -0.027049 | [-0.104547, +0.059832] | 0.54329 | 8/0/14 |
| codex_gpt5_5_medium | ndcg@100 | bm25 | current_mixedbread_pointwise_1000 | -0.088556 | [-0.141614, -0.028930] | 0.00812 | 5/0/17 |
| codex_gpt5_5_medium | ndcg@10 | bm25 | current_mixedbread_first_listwise | +0.003368 | [-0.092618, +0.109414] | 0.95478 | 10/0/12 |
| codex_gpt5_5_medium | ndcg@20 | bm25 | current_mixedbread_first_listwise | -0.013648 | [-0.094363, +0.071459] | 0.76010 | 12/0/10 |
| codex_gpt5_5_medium | ndcg@100 | bm25 | current_mixedbread_first_listwise | -0.086086 | [-0.140786, -0.025331] | 0.00914 | 6/0/16 |
| codex_gpt5_5_medium | ndcg@10 | current_mixedbread_pointwise_1000 | current_mixedbread_first_listwise | +0.009294 | [-0.050229, +0.068773] | 0.76646 | 12/0/10 |
| codex_gpt5_5_medium | ndcg@20 | current_mixedbread_pointwise_1000 | current_mixedbread_first_listwise | +0.013401 | [-0.019329, +0.048213] | 0.45125 | 12/0/10 |
| codex_gpt5_5_medium | ndcg@100 | current_mixedbread_pointwise_1000 | current_mixedbread_first_listwise | +0.002471 | [-0.008065, +0.013298] | 0.66813 | 11/0/11 |
| qwen3_5_9b | ndcg@10 | bm25 | current_mixedbread_pointwise_1000 | -0.046005 | [-0.114421, +0.024634] | 0.22156 | 8/0/14 |
| qwen3_5_9b | ndcg@20 | bm25 | current_mixedbread_pointwise_1000 | -0.057650 | [-0.127233, +0.011970] | 0.13246 | 7/0/15 |
| qwen3_5_9b | ndcg@100 | bm25 | current_mixedbread_pointwise_1000 | -0.133039 | [-0.186678, -0.077812] | 0.00006 | 2/0/20 |
| qwen3_5_9b | ndcg@10 | bm25 | current_mixedbread_first_listwise | -0.039899 | [-0.118798, +0.036385] | 0.33639 | 11/0/11 |
| qwen3_5_9b | ndcg@20 | bm25 | current_mixedbread_first_listwise | -0.050525 | [-0.115615, +0.017064] | 0.15760 | 8/0/14 |
| qwen3_5_9b | ndcg@100 | bm25 | current_mixedbread_first_listwise | -0.131373 | [-0.183045, -0.077520] | 0.00014 | 3/0/19 |
| qwen3_5_9b | ndcg@10 | current_mixedbread_pointwise_1000 | current_mixedbread_first_listwise | +0.006105 | [-0.053832, +0.063811] | 0.84446 | 12/0/10 |
| qwen3_5_9b | ndcg@20 | current_mixedbread_pointwise_1000 | current_mixedbread_first_listwise | +0.007125 | [-0.029076, +0.043667] | 0.71095 | 11/0/11 |
| qwen3_5_9b | ndcg@100 | current_mixedbread_pointwise_1000 | current_mixedbread_first_listwise | +0.001666 | [-0.008228, +0.011582] | 0.75128 | 11/0/11 |
| ministral_3_14b | ndcg@10 | bm25 | current_mixedbread_pointwise_1000 | -0.032906 | [-0.110629, +0.055194] | 0.46031 | 8/0/14 |
| ministral_3_14b | ndcg@20 | bm25 | current_mixedbread_pointwise_1000 | -0.034999 | [-0.097815, +0.034452] | 0.33339 | 6/0/16 |
| ministral_3_14b | ndcg@100 | bm25 | current_mixedbread_pointwise_1000 | -0.122098 | [-0.168154, -0.068999] | 0.00032 | 2/0/20 |
| ministral_3_14b | ndcg@10 | bm25 | current_mixedbread_first_listwise | +0.009194 | [-0.085652, +0.109517] | 0.85978 | 12/0/10 |
| ministral_3_14b | ndcg@20 | bm25 | current_mixedbread_first_listwise | -0.006026 | [-0.075222, +0.069664] | 0.88402 | 10/0/12 |
| ministral_3_14b | ndcg@100 | bm25 | current_mixedbread_first_listwise | -0.115479 | [-0.160223, -0.062612] | 0.00066 | 2/0/20 |
| ministral_3_14b | ndcg@10 | current_mixedbread_pointwise_1000 | current_mixedbread_first_listwise | +0.042100 | [-0.023037, +0.103485] | 0.21572 | 14/0/8 |
| ministral_3_14b | ndcg@20 | current_mixedbread_pointwise_1000 | current_mixedbread_first_listwise | +0.028973 | [-0.005856, +0.062770] | 0.11856 | 16/0/6 |
| ministral_3_14b | ndcg@100 | current_mixedbread_pointwise_1000 | current_mixedbread_first_listwise | +0.006619 | [-0.003142, +0.015918] | 0.19232 | 13/0/9 |

## Experimental design

- Population: all 22 released RAG 2025 development narratives.
- Candidate control: every ranking method receives the same original-narrative ClimbMix BM25 top 1,000.
- Current pointwise method: `mixedbread-ai/mxbai-rerank-base-v2` BF16, using the repository's long-document, strongest-passage, and bounded span-support aggregate.
- Controlled listwise treatment: `castorini/first_qwen3_8b` BF16, tail-to-head windows of 20 with stride 10 over the exact Mixedbread top 100.
- Evaluation: each of the three released Umbrela qrels is reported separately. Missing judgments are treated as grade zero, matching the repository evaluator.
- Primary ranking metrics: nDCG@10, nDCG@20, and nDCG@100. Recall@100 and judged rates are diagnostics.
- Statistics: paired topic bootstrap confidence intervals and two-sided paired Monte Carlo sign-flip tests with deterministic seeds.

## Execution

The RTX 5070 Ti local BF16 smoke scored 8 rows in 6.26 seconds. At its measured 2,825 prompt tokens/second, the 93,312,315-token full organizer pointwise pass would take about 9.2 hours. The cost-guarded A100 path was used for the controlled FIRST treatment.

## Interpretation for RAG

Pointwise and listwise ranking solve different parts of the pipeline. Pointwise scoring is independent and scalable, making it suitable for pruning a large union. Listwise scoring compares candidates directly and can improve their final ordering, but it sees only the candidates admitted to its top-100 window. For answer generation, facet or sub-narrative coverage must therefore be protected before global reranking; a higher nDCG score alone cannot guarantee broader nugget coverage.

## Cost guard

The tracked Modal configuration reserves at most `$6.74` under the configured `$10.00` cap. The valid FIRST stage cost estimate was `$0.20`. The raw cumulative receipt reports `$6.77`; that conservative audit value includes the earlier smoke estimate even though it was already covered by the configured prior-spend estimate. The unmodified runtime receipt follows:

```json
{
  "budget_gate_smoke": {
    "approved_for_full_run": false,
    "dtype": "bfloat16",
    "elapsed_seconds": 204.357797949,
    "estimated_cost_usd": 0.16705023835543054,
    "gpu": "A100-80GB",
    "inference_seconds": 61.260653588999986,
    "model": "Qwen/Qwen3-Reranker-8B",
    "model_load_seconds": 138.014428113,
    "model_revision": "77d193c791ed757ca307ee72715aa132723da912",
    "output": "qwen3_reranker_8b_bf16_smoke_scores.jsonl",
    "projected_full_inference_seconds": 6419.076082645986,
    "prompt_tokens": 890529,
    "prompt_tokens_per_second": 14536.720518435737,
    "rows": 256,
    "schema_version": "organizer-pointwise-runtime-v1",
    "stage": "smoke",
    "worst_case_total_usd": 9.841632
  },
  "configured_worst_case_usd": 6.7437644,
  "estimated_total_cost_usd": 6.767343479901254,
  "hard_cap_usd": 10.0,
  "incremental_worst_case_usd": 0.3437643999999999,
  "listwise": {
    "dtype": "bfloat16",
    "elapsed_seconds": 244.974625112,
    "estimated_cost_usd": 0.20025205755155326,
    "gpu": "A100-80GB",
    "inference_seconds": 52.78553636400005,
    "minimum_passage_words_used": 125,
    "model": "castorini/first_qwen3_8b",
    "model_load_seconds": 121.43899808600001,
    "model_revision": "90e4165ef17f68f00df6be91826524fcba7df4b2",
    "output": "first_qwen3_8b_bf16_mixedbread_top100.jsonl",
    "prompt_tokens": 793059,
    "rows": 2200,
    "schema_version": "organizer-listwise-runtime-v1",
    "seed": "mixedbread_top100_seed_canonical.jsonl",
    "topics": 22,
    "windows": 198
  },
  "prior_spend_estimate_usd": 6.4,
  "schema_version": "organizer-cascade-runtime-v1",
  "staging": {
    "elapsed_seconds": 0.9368515529999999,
    "estimated_cost_usd": 4.118399426987999e-05,
    "schema_version": "organizer-model-staging-v1",
    "snapshots": {
      "Qwen/Qwen3-Reranker-8B": {
        "revision": "77d193c791ed757ca307ee72715aa132723da912",
        "snapshot": "/__modal/volumes/vo-8UKVNTNTtYCRlgw02OaI0r/hub/models--Qwen--Qwen3-Reranker-8B/snapshots/77d193c791ed757ca307ee72715aa132723da912"
      },
      "castorini/first_qwen3_8b": {
        "revision": "90e4165ef17f68f00df6be91826524fcba7df4b2",
        "snapshot": "/__modal/volumes/vo-8UKVNTNTtYCRlgw02OaI0r/hub/models--castorini--first_qwen3_8b/snapshots/90e4165ef17f68f00df6be91826524fcba7df4b2"
      }
    },
    "worst_case_total_usd": 9.841632
  }
}
```

## Limitations

- The 2026 organizer test narratives have no released qrels, so nDCG is measured on the released 2025 development topics while isolating the organizer's FIRST listwise stage.
- Umbrela qrels are LLM judgments over a pooled set, not exhaustive corpus judgments. Results are shown separately to expose assessor sensitivity.
- Because FIRST only reorders Mixedbread's exact top 100, recall@100 is necessarily unchanged. The observed gain is an ordering gain, not broader candidate recall.
- An attempted full Qwen-to-FIRST reproduction was quarantined before evaluation because its cache adapter stringified the API's structured query object. A corrected exhaustive rerun was rejected by the cumulative cloud-cost guard, so no contaminated Qwen scores appear in these tables.
- This controlled comparison isolates ranking. The current competition pipeline's facet retrieval and round-robin evidence selection are discussed as downstream design constraints, not silently mixed into the candidate set.
