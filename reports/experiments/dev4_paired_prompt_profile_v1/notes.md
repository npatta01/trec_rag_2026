# Paired prompt-profile comparison on four development topics

Supersedes the two-topic comparison in `dev_ragdoll_nugget_eval_spike_v1`, which reported
atomic answer objects as neutral-to-better on nugget coverage. With four paired topics and a
single judge throughout, **that conclusion does not hold**.

## Result: a real trade-off, consistent in both directions

| | A `default` | B `atomic_claims` | delta |
|---|---|---|---|
| strict_vital | **0.6348** | 0.5698 | −0.065 |
| strict_all | **0.5854** | 0.5375 | −0.048 |
| weighted precision, first citation | 0.475 | **0.671** | +0.196 |
| weighted precision, all judged | 0.415 | **0.622** | +0.207 |
| hard precision | 0.071 | **0.383** | +0.312 |
| support labels | 4 FS / 116 PS / 51 NS | 48 FS / 92 PS / 18 NS | |

All four topics favour A on coverage. All four favour B on support. Neither direction has a
single exception, which is what makes a four-topic result worth acting on despite the size.

## Why coverage drops: length, mostly

| topic | A words | B words | A strict_vital | B strict_vital |
|---|---|---|---|---|
| 58 | 770 | 514 | 0.714 | 0.629 |
| 72 | 653 | 344 | 0.528 | 0.431 |
| 144 | 747 | 756 | 0.576 | 0.515 |
| 200 | 935 | 772 | 0.721 | 0.705 |

Across the eight arm-topic points, word count correlates with `strict_vital` at **r = +0.70**.
The atomic instruction suppresses length, and nugget scoring is pure recall with no length
penalty, so shorter answers simply cover less.

Topic 144 is the informative exception: arm B wrote nine words *more* and still lost 0.061 on
coverage, while posting its best support figure (0.808). So length is the dominant mechanism
but not the only one.

**Neither arm approaches the 1,024-word cap** — A ranges 653 to 935, B 344 to 772. Unused
budget is uncollected coverage in both arms.

## The obvious next arm

Atomic structure plus an explicit instruction to spend the full word budget. If the coverage
loss is mostly length, arm C should keep B's support gain without B's coverage penalty. That
is the experiment to run before deciding what the submission uses.

## Caveats

- Four topics. The direction is consistent but this is not a significance test.
- **Generation is high variance.** The same default config produced 9 answer objects for topic
  58 in one run and 28 in another. Some of any observed delta is run-to-run noise.
- The judge changed from `openai-codex/gpt-5.5` to `openrouter/deepseek/deepseek-v4-flash` for
  cost reasons, so absolute values are not comparable to the earlier record. Within this
  record every cell uses the same judge. The two judges also disagree markedly at the label
  level: on arm A, gpt-5.5 returned 31 Partial and no No-Support, while DeepSeek returned 51
  No-Support out of 175.
- Generation for the remaining 18 topics was stopped on cost grounds at $20.63. Only topics
  that completed under both arms are analysed.
- Automated assignment scores above NIST manual assignment; treat all figures as relative.
