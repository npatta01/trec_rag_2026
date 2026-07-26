# Topic 213 controlled generator benchmark

All three generators received the same frozen 42-claim evidence ledger, exact per-facet
sentence quotas, prompt, response contract, and 900-1,000-word candidate target. Citation
selection and local-Qwen support auditing were identical and model-blind.

## Nugget coverage

| Run | Strict | Partial | Vital strict | Candidate words | Submitted words | Retained claims | Excluded | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original local Qwen, unconstrained | 0.520 | 0.630 | 0.556 | n/a | 1883 | 50 | 2 unsupported | n/a |
| Existing organizer-compliant Qwen | 0.420 | 0.500 | 0.444 | n/a | 407 | 20 | 3 | n/a |
| Local Qwen3-4B | 0.420 | 0.490 | 0.407 | 969 | 792 | 35/42 | 7 | $0.000000 |
| DeepSeek V4 Flash | 0.420 | 0.490 | 0.407 | 969 | 792 | 35/42 | 7 | $0.000582 |
| GPT-5.6 Sol | 0.460 | 0.520 | 0.444 | 960 | 789 | 35/42 | 7 | $0.093456 |

## Recommendation

The selected generator is **GPT-5.6 Sol**. Selection is lexicographic by strict coverage,
then vital strict coverage, partial-credit coverage, unsupported submitted sentences, and cost.

## Interpretation

This experiment isolates the generator over one development topic; it does not establish held-out
generalization. Local Qwen also serves as the blinded support and nugget judge, so evaluator-style bias
remains a limitation even though model identity is never included in judge payloads.
