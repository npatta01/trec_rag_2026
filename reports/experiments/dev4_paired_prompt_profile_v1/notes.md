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


## Arm C: atomic claims pushed toward the word cap

| arm | strict_vital | strict_all | wp first-cite | wp all-judged | hard precision | No Support rate |
|---|---|---|---|---|---|---|
| A default | **0.635** | 0.585 | 0.475 | 0.415 | 0.071 | 30% |
| B atomic | 0.570 | 0.537 | **0.671** | **0.622** | 0.383 | **11%** |
| C atomic + budget | 0.621 | **0.596** | 0.633 | 0.542 | **0.438** | 30% |

**No arm dominates.** Arm C confirmed the length hypothesis — lifting arm B from ~600 to ~890
words recovered most of the lost coverage (strict_vital 0.570 to 0.621) and produced the best
strict_all and hard precision of the three.

But it also **tripled the citation error rate back to arm A's level**: 89 No Support out of 296
judgments, against 18 out of 158 for arm B. Weighted precision, which is what the organizers
report, is accordingly *worse* in C than in B on both variants. The likely mechanism is that
instructing the model to fill the budget makes it assert more than the retrieved evidence
supports.

So the three arms are really three positions on one trade-off: A says most and grounds worst,
B grounds best and says least, C says nearly as much as A while grounding better than A but
noticeably worse than B.

**Arm C's numbers predate a prompt change.** They were produced under the original wording
("use most of that budget"). The profile now targets roughly 900 words with a hard 1,024
ceiling, after two arm-A topics exceeded the cap outright. Whether the tighter target keeps
C's coverage without its No Support blowup is untested.

## Generation reliability: about one topic in four fails first time

| run | attempted | failed |
|---|---|---|
| arm A | 22 | 7 |
| arm B | 8 | 2 |
| arm C | 4 | 1 |

Three modes: `uncited references` (4), `exceeds 1,024 words` (2), malformed completion (3).

The uncited-reference failures are **self-imposed**. The task reference states plainly that
uncited references do not hurt the score and that validators must not reject on them; this
generator enforces all-cited only to match the organizer baseline script. The constraint is
also mathematically tight, needing `references <= 3 x objects`: topic 58 succeeded with 91
references and 33 objects, a ceiling of 99 and slack of 8.

Reference counts are themselves out of line on that topic. Organizer baselines carry a median
of 10 and 16 references per topic and never exceed 25; topic 58 produced 83 under arm A and 91
under arm C, close to copying the whole 100-document pool, which the task reference explicitly
advises against.

Deterministically dropping uncited references and renumbering citations would make that
failure mode impossible without changing any claim or its supporting document. At 119 topics a
25% first-attempt failure rate means roughly 30 resumes at about $0.71 each.


## Arm D: one citation per claim, constraints moved out of the prompt

Two changes shipped together, because neither works alone:

1. **Prompt**: dropped "cite every reference"; told the model to cite the single best supporting
   document and add a second or third only when each independently supports the same object.
2. **Pipeline**: `normalize_generated_record` rebuilds `references` from the citations actually
   used, and `trim_to_word_limit` drops trailing objects over the cap. Both organizer
   constraints are now enforced deterministically rather than asked for in the prompt.

| arm | strict_vital | strict_all | wp first | wp all | hard prec | No Support | oracle wp first |
|---|---|---|---|---|---|---|---|
| A default | **0.635** | 0.585 | 0.475 | 0.415 | 0.071 | 30% | 0.537 |
| B atomic | 0.570 | 0.537 | 0.671 | 0.622 | 0.383 | 11% | 0.696 |
| C atomic + budget | 0.621 | **0.596** | 0.633 | 0.542 | 0.438 | 30% | 0.745 |
| D focused citations | 0.588 | 0.569 | **0.692** | **0.692** | **0.481** | **9.6%** | **0.768** |

The model complied exactly: **all 161 answer objects carry exactly one citation**, and reference
sprawl collapsed — topic 58 went from 91 references to 15.

**Why the citation rule mattered.** Measured on arm C, Full Support rate falls off a cliff as
citations per object rise:

| citations on the object | Full Support | No Support |
|---|---|---|
| 1 | 72.5% | 13.7% |
| 2 | 27.9% | 21.3% |
| 3 | 11.1% | 44.4% |

The all-cited rule was the cause. To cover a 91-entry reference list across 33 objects the model
must attach roughly three citations to nearly every one, and the trailing citations frequently
do not support the claim. The rule manufactured the errors it was unrelated to.

**Recommendation: arm D.** It wins every support metric as generated *and* at the oracle repair
ceiling. Because every object carries exactly one citation, its `wp_all` equals `wp_first` by
construction, so it cannot be diluted on the all-judged variant — a structural guarantee the
other arms cannot obtain. It gives up 0.047 strict_vital against arm A in exchange for +0.217
weighted precision and +0.410 hard precision.

**Coverage is still unexplained.** Arm D is the longest arm (954 to 981 words) yet ranks third on
coverage, so length is not the whole mechanism after all — constraining each claim to a single
supporting document appears to narrow what the model will assert. Worth understanding before
treating D as final.

**Reliability.** Zero uncited-reference failures in arm D, against four across earlier runs:
normalization removed that mode entirely. The word cap then became the dominant failure (three
occurrences), which is why the deterministic trim was added. The pattern holds in both cases —
a constraint enforced in the prompt costs failed generations, and the same constraint enforced
in a post-processor costs nothing.


## Document selection: extraction dominates truncation

The generator showed each document's first 1,000 words. With a median document of 3,426 words,
91% are cut and the model sees 17% of the pool, always the opening, which on scraped pages is
frequently navigation furniture.

A first comparison changed selection *and* budget together (975-word truncation against
261-word extraction) and so could not attribute the resulting 0.057 coverage loss. Repeating it
at matched budget separates them:

| documents | $/topic | strict_vital | strict_all | wp first | hard prec | No Support |
|---|---|---|---|---|---|---|
| head-truncated 957w | 0.712 | 0.588 | **0.569** | 0.692 | 0.481 | 9.6% |
| extractive 931w | 0.788 | **0.613** | 0.564 | **0.717** | **0.506** | **7.4%** |
| extractive 261w | **0.295** | 0.531 | 0.511 | **0.823** | **0.692** | **4.8%** |

**At matched budget extraction wins four of five metrics**, each by about 0.025, with
all-coverage level. No single delta clears the four-topic noise bar, but four independent
metrics moving together is stronger evidence than any one of them. The earlier coverage loss
belonged to the budget cut, not the selection method, so truncation has no remaining argument.

**Passage budget is a dial trading coverage against precision.** Dropping 931 to 261 words costs
0.082 strict_vital and buys 0.106 weighted precision. Less but better-selected text makes the
model assert less and ground what it does assert more firmly. Extraction dominates truncation at
both points, so the dial setting is a separate decision from the selection method.

Where to sit depends on how organizers weight nugget coverage against citation support, which is
not observable from the released material.

**Extraction at full budget is not a cost saving** — $0.7882 against $0.7115. Fewer document
words are offset by longer answers, and output tokens dominate the bill. Only shrinking the
budget reduces cost.

**Measurement note.** Citations are judged against the evidence each generator actually read.
That is internally consistent but not what organizers do: they resolve references from the
index, that is full documents. Judging every arm against full document text is the realistic
measurement and remains untested. An earlier version of the 261-word comparison judged it
against head-truncated text its generator never saw, inflating No Support to 32.7% and inverting
the conclusion; that error is corrected above.

The selector is lexical, scoring windows by distinct narrative-term coverage. Gold nuggets often
cover facets the narrative never names, so a cross-encoder should select better; the window
scorer in `facet_retrieval.MixedbreadCoverageScorer` already exists for that upgrade.
