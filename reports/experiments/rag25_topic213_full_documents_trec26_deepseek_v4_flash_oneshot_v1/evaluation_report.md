# Topic 213: DeepSeek V4 Flash one-shot

DeepSeek V4 Flash received all 42 independently supported full-evidence source claims in one successful candidate-producing OpenRouter request spanning the ten organizer sub-narratives. One earlier OpenRouter request returned HTTP success with empty final content and produced no candidate. Local Qwen, served through LiteLLM, independently audited each generated sentence and evaluated nuggets only after the filtered generation and organizer JSONL were frozen and hashed. No DeepSeek rewrite or repair call was made.

## Submission validation

- Generation model: `deepseek/deepseek-v4-flash` through OpenRouter
- Successful candidate-generation calls: 1
- Total OpenRouter generation API calls: 2 (including 1 HTTP-200/empty-content failure)
- Transport attempts for the successful request: 1
- DeepSeek rewrite or repair calls: 0
- Supported source claims supplied: 42
- Candidate sentences: 28
- Submitted sentences: 22
- Excluded by local-Qwen support audit: 6
- Answer words: 527 / 1024
- Unique ClimbMix references: 23
- Citation coverage: 1.000
- Unsupported submitted sentences: 0

## Nugget coverage

| Run | Strict | Partial credit | Vital strict | Supported nuggets |
|---|---:|---:|---:|---:|
| DeepSeek V4 Flash one-shot | 0.420 | 0.510 | 0.481 | 21 / 50 |
| Seeded local-Qwen consolidation | 0.420 | 0.500 | 0.444 | 21 / 50 |
| DeepSeek delta | +0.000 | +0.010 | +0.037 | +0 |

## Recommendation

DeepSeek is the marginal coverage winner: strict coverage ties, while DeepSeek gains one point of partial-credit coverage and 3.7 points of vital strict coverage. The gain is small and costs 120 additional answer words; DeepSeek also had 6 of 28 candidate sentences excluded. The comparison keeps the 42-claim source ledger, local-Qwen support auditor, nugget evaluator, sentence format, and word ceiling fixed, but it is not a pure generator ablation because the seeded Qwen run allowed one bounded rewrite-and-reaudit pass while this DeepSeek run deliberately allowed none. Cross-task ranking should wait for the other isolated runs.

## Integrity

The organizer nuggets were not read until `response_generation.json` and `rag_output_trec_rag_2026.jsonl` were written and hashed. The manifest pins both frozen hashes, all source inputs, the exact requested and returned model identities, and every published artifact.
