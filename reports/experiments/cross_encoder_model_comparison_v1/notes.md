# Cross-Encoder Model Comparison

Run date: 2026-07-08

This record promotes the cross-encoder and reranker scratch results out of
`tmp/` into a durable experiment report. All full-dev rows use the same
22-topic projected-qrels evaluation and BM25 candidate-order baseline;
the `candidate_depth` column preserves whether a run reranked top-20 or
top-50 candidates.

## Full-Dev Ranking Results

| model | method | nDCG@10 | delta vs BM25 | losses | big losses | worst topic | source |
|---|---|---:|---:|---:|---:|---|---|
| mixedbread-ai/mxbai-rerank-base-v2 | `chunk_top4_weighted` | 0.5399427110 | 0.1259815425 | 4 | 1 | 224: -0.1858914471 | `tmp/st_chunk_eval_hits50_mixedbread_ai__mxbai_rerank_base_v2_ml1024_cm3500_ov350.json` |
| mixedbread-ai/mxbai-rerank-base-v2 | `chunk_top3_weighted` | 0.5353629309 | 0.1214017624 | 4 | 2 | 224: -0.1858914471 | `tmp/st_chunk_eval_hits50_mixedbread_ai__mxbai_rerank_base_v2_ml1024_cm3500_ov350.json` |
| mixedbread-ai/mxbai-rerank-base-v2 | `st_crossencoder` | 0.5322243661 | 0.1182631976 | 5 | 2 | 224: -0.2222050772 | `tmp/st_crossencoder_eval_hits50_mixedbread_ai__mxbai_rerank_base_v2_ml1024.json` |
| mixedbread-ai/mxbai-rerank-base-v2 | `chunk_mean_top3` | 0.5309067952 | 0.1169456266 | 4 | 1 | 224: -0.2506939209 | `tmp/st_chunk_eval_hits50_mixedbread_ai__mxbai_rerank_base_v2_ml1024_cm3500_ov350.json` |
| mixedbread-ai/mxbai-rerank-base-v2 | `chunk_max` | 0.5239315476 | 0.1099703791 | 7 | 1 | 224: -0.1517053415 | `tmp/st_chunk_eval_hits50_mixedbread_ai__mxbai_rerank_base_v2_ml1024_cm3500_ov350.json` |
| mixedbread-ai/mxbai-rerank-base-v2 | `st_crossencoder_longctx` | 0.5154219200 | 0.1014607515 | 6 | 1 | 224: -0.1008225083 | `tmp/st_crossencoder_longctx_hits50_mixedbread_ai__mxbai_rerank_base_v2_ctx32768_buf512.json` |
| cross-encoder/ms-marco-MiniLM-L6-v2 | `seqcls` | 0.4990558712 | 0.0850947027 | 6 | 2 | 31: -0.2814230470 | `tmp/seqcls_eval_hits50_cross_encoder__ms_marco_MiniLM_L6_v2_ml512.json` |
| BAAI/bge-reranker-v2-m3 | `seqcls` | 0.4903882668 | 0.0764270983 | 6 | 3 | 224: -0.1858914471 | `tmp/seqcls_eval_hits50_BAAI__bge_reranker_v2_m3_ml1024.json` |
| Qwen/Qwen3-Reranker-0.6B | `hybrid_full75_chunk25_margin` | 0.4892623970 | 0.0753012285 | 6 | 4 | 224: -0.2150821739 | `tmp/qwen_full_doc_margin_hits50_Qwen3_Reranker_0p6B_ml32768.json` |
| Qwen/Qwen3-Reranker-0.6B | `full_doc_probability_else_chunk` | 0.4820407811 | 0.0680796126 | 6 | 5 | 37: -0.2042534325 | `tmp/qwen_full_doc_margin_hits50_Qwen3_Reranker_0p6B_ml32768.json` |
| Qwen/Qwen3-Reranker-0.6B | `full_doc_margin_else_chunk` | 0.4785243927 | 0.0645632242 | 6 | 6 | 37: -0.2060520588 | `tmp/qwen_full_doc_margin_hits50_Qwen3_Reranker_0p6B_ml32768.json` |
| Qwen/Qwen3-Reranker-0.6B | `hybrid_full75_chunk25_probability` | 0.4761554472 | 0.0621942787 | 6 | 4 | 37: -0.2207186418 | `tmp/qwen_full_doc_margin_hits50_Qwen3_Reranker_0p6B_ml32768.json` |
| Qwen/Qwen3-Reranker-0.6B | `chunk_top3_margin` | 0.4665202182 | 0.0525590497 | 7 | 5 | 224: -0.2583247297 | `tmp/qwen_full_doc_margin_hits50_Qwen3_Reranker_0p6B_ml32768.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_weighted` | 0.4649839734 | 0.0510228049 | 7 | 5 | 224: -0.2562179815 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_ov0.json` |
| Qwen/Qwen3-Reranker-0.6B | `chunk_top3_probability` | 0.4649839734 | 0.0510228049 | 7 | 5 | 224: -0.2562179815 | `tmp/qwen_full_doc_margin_hits50_Qwen3_Reranker_0p6B_ml32768.json` |
| Qwen/Qwen3-Reranker-0.6B | `top4_weighted` | 0.4615683544 | 0.0476071859 | 7 | 5 | 224: -0.2406569368 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_ov0.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_weighted` | 0.4583081461 | 0.0443469776 | 8 | 4 | 224: -0.2977960584 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B.json` |
| Qwen/Qwen3-Reranker-4B | `qwen_max` | 0.4561669893 | 0.0422058208 | 7 | 4 | 37: -0.3224467216 | `tmp/qwen_aggregation_eval_hits20_Qwen3_Reranker_4B.json` |
| Qwen/Qwen3-Reranker-0.6B | `qwen_max` | 0.4549598679 | 0.0409986994 | 9 | 4 | 224: -0.2455346756 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_ov0.json` |
| Qwen/Qwen3-Reranker-0.6B | `top4_weighted` | 0.4516065334 | 0.0376453649 | 9 | 4 | 224: -0.2619092207 | `tmp/qwen_aggregation_eval_hits20_Qwen3_Reranker_0p6B.json` |
| Qwen/Qwen3-Reranker-4B | `top3_weighted` | 0.4515029845 | 0.0375418160 | 7 | 4 | 37: -0.3549384506 | `tmp/qwen_aggregation_eval_hits20_Qwen3_Reranker_4B.json` |
| Qwen/Qwen3-Reranker-0.6B | `qwen_max` | 0.4513516397 | 0.0373904712 | 9 | 5 | 224: -0.3036821859 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B.json` |
| Qwen/Qwen3-Reranker-0.6B | `qwen_max` | 0.4513516397 | 0.0373904712 | 9 | 5 | 224: -0.3036821859 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap2000.json` |
| Qwen/Qwen3-Reranker-0.6B | `qwen_max` | 0.4513516397 | 0.0373904712 | 9 | 5 | 224: -0.3036821859 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap3000.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_weighted` | 0.4507551383 | 0.0367939698 | 7 | 4 | 224: -0.2619092207 | `tmp/qwen_aggregation_eval_hits20_Qwen3_Reranker_0p6B.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p6` | 0.4504224888 | 0.0364613203 | 8 | 1 | 224: -0.2761208300 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_ov0.json` |
| Qwen/Qwen3-Reranker-0.6B | `top4_weighted` | 0.4498007285 | 0.0358395600 | 10 | 6 | 224: -0.2406569368 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap2000.json` |
| Qwen/Qwen3-Reranker-0.6B | `top4_weighted` | 0.4498007285 | 0.0358395600 | 10 | 6 | 224: -0.2406569368 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap3000.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_weighted` | 0.4490912106 | 0.0351300421 | 9 | 6 | 224: -0.2423513360 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap2000.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_weighted` | 0.4490912106 | 0.0351300421 | 9 | 6 | 224: -0.2423513360 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap3000.json` |
| Qwen/Qwen3-Reranker-4B | `top4_weighted` | 0.4487664382 | 0.0348052697 | 7 | 4 | 37: -0.3500637735 | `tmp/qwen_aggregation_eval_hits20_Qwen3_Reranker_4B.json` |
| Qwen/Qwen3-Reranker-0.6B | `top4_weighted` | 0.4487112473 | 0.0347500788 | 8 | 5 | 224: -0.2910518592 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B.json` |
| Qwen/Qwen3-Reranker-4B | `qwen_max` | 0.4485499929 | 0.0345888244 | 8 | 5 | 37: -0.3790959707 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_4B.json` |
| Qwen/Qwen3-Reranker-4B | `top4_weighted` | 0.4476749358 | 0.0337137673 | 7 | 6 | 37: -0.4014531713 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_4B.json` |
| Qwen/Qwen3-Reranker-0.6B | `max_plus_support` | 0.4464881176 | 0.0325269491 | 8 | 6 | 224: -0.3389591550 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_ov0.json` |
| Qwen/Qwen3-Reranker-0.6B | `qwen_max` | 0.4461574245 | 0.0321962560 | 10 | 3 | 224: -0.2438873458 | `tmp/qwen_aggregation_eval_hits20_Qwen3_Reranker_0p6B.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p85` | 0.4460172630 | 0.0320560945 | 8 | 3 | 224: -0.3303275339 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B.json` |
| Qwen/Qwen3-Reranker-4B | `top3_weighted` | 0.4455752541 | 0.0316140856 | 8 | 5 | 37: -0.4014531713 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_4B.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p75` | 0.4431033322 | 0.0291421637 | 8 | 2 | 224: -0.2901583766 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap2000.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p75` | 0.4431033322 | 0.0291421637 | 8 | 2 | 224: -0.2901583766 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap3000.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p75` | 0.4406133949 | 0.0266522264 | 8 | 3 | 224: -0.3012781119 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_ov0.json` |
| Qwen/Qwen3-Reranker-0.6B | `max_plus_support` | 0.4404049084 | 0.0264437399 | 9 | 4 | 224: -0.2696254181 | `tmp/qwen_aggregation_eval_hits20_Qwen3_Reranker_0p6B.json` |
| Qwen/Qwen3-Reranker-4B | `max_plus_support` | 0.4402009190 | 0.0262397505 | 7 | 4 | 37: -0.3549384506 | `tmp/qwen_aggregation_eval_hits20_Qwen3_Reranker_4B.json` |
| Qwen/Qwen3-Reranker-4B | `max_plus_support` | 0.4385501840 | 0.0245890155 | 9 | 6 | 37: -0.4014531713 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_4B.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p6` | 0.4378573317 | 0.0238961632 | 10 | 1 | 224: -0.3429575590 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p85` | 0.4364271310 | 0.0224659625 | 9 | 4 | 224: -0.2958461356 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_ov0.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p6` | 0.4353391786 | 0.0213780100 | 9 | 1 | 224: -0.3019710531 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap3000.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p6` | 0.4349080257 | 0.0209468572 | 9 | 1 | 224: -0.3114564159 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap2000.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p75` | 0.4332455206 | 0.0192843521 | 8 | 3 | 224: -0.3341193732 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B.json` |
| Qwen/Qwen3-Reranker-0.6B | `max_plus_support` | 0.4308599588 | 0.0168987903 | 10 | 6 | 224: -0.3486926890 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p85` | 0.4299664495 | 0.0160052809 | 9 | 4 | 224: -0.2915535673 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap2000.json` |
| Qwen/Qwen3-Reranker-0.6B | `top3_bm25_fusion_a0p85` | 0.4299664495 | 0.0160052809 | 9 | 4 | 224: -0.2915535673 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap3000.json` |
| Qwen/Qwen3-Reranker-0.6B | `max_plus_support` | 0.4129979314 | -0.0009632371 | 11 | 7 | 224: -0.3389591550 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap2000.json` |
| Qwen/Qwen3-Reranker-0.6B | `max_plus_support` | 0.4128344199 | -0.0011267486 | 11 | 7 | 224: -0.3425564076 | `tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap3000.json` |

## Readout

The best raw model family in these artifacts is Mixedbread. The strongest 
single artifact row is the Mixedbread `chunk_top4_weighted` window aggregate 
at `0.539943` nDCG@10, but it still has one large topic regression. The PR's 
recommended production-facing score is captured separately as the 
`coverage_aware_long_doc_aggregate` config because it trades some average 
score for zero topic losses worse than `-0.1` on this dev sample.

Qwen did improve over BM25 on average, but the full-dev Qwen 0.6B/4B rows 
show materially smaller gains and larger worst-topic losses than Mixedbread. 
The available Qwen 8B evidence is qualitative only: a 12-document probe in 
`tmp/qwen_rerank_probe_results.json`, not a 22-topic full-dev run.

## Prompt And Probe Results

The prompt/probe table is not comparable to full-dev nDCG. It reports mean 
per-topic Spearman correlation between model scores and qrel grades on a 
small hard-example probe.

| model | instruction | docs | topics | mean Spearman | worst topic | source |
|---|---|---:|---:|---:|---|---|
| Qwen/Qwen3-Reranker-0.6B | `official_default` | 12 | 3 | 0.6291419622 | 144: 0.1054092553 | `tmp/qwen_rerank_prompt_probe.json` |
| Qwen/Qwen3-Reranker-0.6B | `custom_full_narrative` | 12 | 3 | 0.5270462767 | 515: 0.3162277660 | `tmp/qwen_rerank_prompt_probe.json` |
| Qwen/Qwen3-Reranker-0.6B | `stricter_full_narrative` | 12 | 3 | 0.5270462767 | 515: 0.3162277660 | `tmp/qwen_rerank_prompt_probe.json` |
| Qwen/Qwen3-Reranker-4B | `official_default` | 12 | 3 | 0.3865006029 | 515: -0.1054092553 | `tmp/qwen_rerank_prompt_probe.json` |
| Qwen/Qwen3-Reranker-4B | `custom_full_narrative` | 12 | 3 | -0.0702728369 | 515: -0.6324555320 | `tmp/qwen_rerank_prompt_probe.json` |
| Qwen/Qwen3-Reranker-4B | `stricter_full_narrative` | 12 | 3 | 0.0185185185 | 515: -0.6324555320 | `tmp/qwen_rerank_prompt_probe.json` |
| Qwen/Qwen3-Reranker-8B | `official_default` | 12 | 3 | -0.1405456738 | 515: -0.9486832981 | `tmp/qwen_rerank_prompt_probe.json` |
| Qwen/Qwen3-Reranker-8B | `custom_full_narrative` | 12 | 3 | -0.5973191136 | 200: -0.9486832981 | `tmp/qwen_rerank_prompt_probe.json` |
| Qwen/Qwen3-Reranker-8B | `stricter_full_narrative` | 12 | 3 | -0.1756820922 | 515: -0.9486832981 | `tmp/qwen_rerank_prompt_probe.json` |
| Qwen/Qwen3-Reranker-0.6B | `probe_default` | 75 | 3 | 0.4015224751 | 515: -0.0114200412 | `tmp/qwen_rerank_probe_results.json` |
| Qwen/Qwen3-Reranker-4B | `probe_default` | 75 | 3 | 0.2390466771 | 515: -0.2056267482 | `tmp/qwen_rerank_probe_results.json` |
| Qwen/Qwen3-Reranker-8B | `probe_default` | 75 | 3 | 0.2082594880 | 515: -0.1797959271 | `tmp/qwen_rerank_probe_results.json` |

## Files

- `system_scores.csv`: one row per full-dev model/method result.
- `topic_system_scores.csv`: one row per topic and full-dev system.
- `prompt_probe_scores.csv`: qualitative prompt/probe alignment summary.
- `metrics.json`: machine-readable copy of the same summary.
