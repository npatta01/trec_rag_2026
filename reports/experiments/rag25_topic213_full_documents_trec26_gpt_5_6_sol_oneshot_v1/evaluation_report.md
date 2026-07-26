# GPT-5.6 Sol one-shot Topic 213 report

## Protocol

A single semantic generation request was sent through OpenRouter to `openai/gpt-5.6-sol` at temperature 0. It received all 42 source claims previously judged fully supported across the ten sub-narratives. Local `Qwen/Qwen3-4B-Instruct-2507` independently audited each generated sentence and evaluated nugget coverage. No generation rewrite or repair call was made.

The candidate was support-audited, filtered, written, and SHA-256 sealed before the 50 organizer nuggets were opened.

## Results

- Strict nugget coverage: `0.360`
- Partial-credit coverage: `0.410`
- Vital strict coverage: `0.333`
- Submitted sentences: `7`
- Submitted words: `245` / 1024
- Unique references: `16`
- Citation coverage: `1.000`
- Unsupported submitted sentences: `0`
- Sentences excluded by Qwen audit: `3`
- Semantic generation requests: `1`
- Successful-request HTTP attempts: `1`
- Rejected pre-completion HTTP requests: `2`
- Total generation-endpoint HTTP requests: `3`

## Format-compliant baseline comparison

| Run | Generator | Sentences | Words | Strict | Partial credit | Vital strict |
|---|---|---:|---:|---:|---:|---:|
| Existing format-compliant baseline | Local Qwen with bounded repair | 20 | 407 | 0.420 | 0.500 | 0.444 |
| This run | GPT-5.6 Sol, one shot | 7 | 245 | 0.360 | 0.410 | 0.333 |

Strict delta: `-0.060`. Partial-credit delta: `-0.090`. The baseline used a bounded rewrite pass while this run intentionally did not, so the table compares final system outcomes rather than generation-only model quality.

GPT-5.6 Sol produced 10 candidates, exactly one per sub-narrative. The independent audit excluded 3, leaving three facets with no submitted sentence; this compression is the main observed source of lost coverage.

## Recommendation

Use the existing format-compliant local-Qwen baseline for the current Topic 213 submission comparison: it has higher strict, partial-credit, and vital coverage while retaining perfect citation coverage. Keep this GPT-5.6 Sol result as the clean one-shot ablation, not the preferred run. Treat the ranking cautiously because Topic 213 is development data and the local-Qwen judge is an experimental evaluator, not the official organizer scorer.
