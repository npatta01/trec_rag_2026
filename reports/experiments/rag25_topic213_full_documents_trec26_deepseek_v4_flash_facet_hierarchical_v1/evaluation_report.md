# Topic 213: DeepSeek V4 Flash facet hierarchy

This controlled run used only the 42 source claims previously judged supported. DeepSeek made one
atomic-fact extraction call for each of the ten sub-narratives, followed by one synthesis call over
only those extraction outputs. Local Qwen independently audited the synthesized sentences and
evaluated organizer nuggets only after all generation artifacts and the official submission were
frozen.

## Organizer-format result

- Submitted sentences: 27
- Answer words: 628 / 1024
- Direct `shard_*` references: 30
- Excluded unsupported sentences: 8
- Citation coverage: 1.000
- Unsupported submitted sentences: 0

## Nugget coverage

| Metric | Qwen one-shot baseline | DeepSeek facet hierarchy | Delta |
|---|---:|---:|---:|
| Strict coverage | 0.420 | 0.440 | +0.020 |
| Partial-credit coverage | 0.500 | 0.490 | -0.010 |
| Vital strict coverage | 0.444 | 0.444 | +0.000 |
| Submitted sentences | 20 | 27 | +7 |
| Answer words | 407 | 628 | +221 |

## Call accounting

- DeepSeek model requested: `deepseek/deepseek-v4-flash`
- Logical generation calls: 11 (10 extraction + 1 synthesis)
- Originating model generations represented: 11
- Network requests in this invocation: 0
- Resumed checkpoint reads in this invocation: 11
- Prompt tokens: 6554
- Completion tokens: 5510
- Total tokens: 12064

## Recommendation

On strict nugget coverage, the hierarchical DeepSeek run is the better of these two controlled format-compliant runs.
The row-level nugget comparison and full sentence-to-fact-to-source-to-passage lineage are retained
for diagnosing which facets gained or lost coverage. This report does not compare the separate
DeepSeek one-shot and GPT one-shot worktrees.
