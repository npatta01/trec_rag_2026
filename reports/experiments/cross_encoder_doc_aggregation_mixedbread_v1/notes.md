# Cross-Encoder Document Aggregation: Mixedbread Base V2

Run date: 2026-07-05

## Summary

The regression is not primarily a chunking bug. It is a document-scoring problem:
raw max/top-k cross-encoder scores over-reward documents with one highly
matching passage, while the dev qrels often reward broader practical coverage.

The long-context follow-up is the better current basis for this PR. The scorer
uses `doc_max_32768_buf512_score`: it reads the full document when the
narrative/document pair fits in the model context, and otherwise reads the
longest safe prefix. In the BM25 top-50 candidate set, 1,079 of 1,100 candidates
fit fully and 21 needed truncation.

The best average long-context formula improved BM25 from `0.413961` to
`0.559599` nDCG@10, but it still had one large regression. If the current
priority is avoiding large regressions, the better fixed formula is:

```text
score =
  0.5 * doc_max_32768_buf512_score
  + 0.5 * top4_weighted_window_score
  + 0.25 * min(relative_span_support_1.0, 6)
```

This reached `0.527544` nDCG@10, `+0.113583` over BM25, with zero topic losses
worse than `-0.1`.

## Formula Definitions

Let `s1 >= s2 >= ... >= sn` be the document's window scores for one topic.

`top4_weighted_window_score = 0.55*s1 + 0.25*s2 + 0.13*s3 + 0.07*s4`.

`doc_prefix_1024_score` is the cross-encoder score from the document-level input
with `max_length=1024`. This is prefix/truncated-document scoring, not a true
full-document judgment. The Mixedbread base config reports a 32,768-token
context, so a follow-up full-context run should use a larger `max_length` and
should be labeled separately, for example `doc_full_32768_score`.

`doc_max_32768_buf512_score` is the later long-context score. It keeps the
narrative, reserves 512 tokens for pair-formatting safety, and uses as many
document tokens as fit under the model's 32,768-token context. It is a true
full-document score only for candidates where `doc_fits_context=true`; otherwise
it is the longest safe document-prefix score.

Token-length diagnostic for the 1,100 narrative/document pairs in the BM25 top-50
candidate set:

| cap | pairs over cap |
|---:|---:|
| 1,024 | 1,044 |
| 2,048 | 922 |
| 4,096 | 622 |
| 8,192 | 265 |
| 16,384 | 95 |
| 32,768 | 21 |

The median pair length is 4,596 tokens; p95 is 22,032; max is 163,222. So a
32k document-prefix score would cover most candidates, but not all candidates.
The remaining over-32k documents still need chunk/window fallback.

## Long-Context Follow-Up

I ran a follow-up scorer using `doc_max_32768_buf512`: the scorer reserves 512
tokens for pair-formatting safety, keeps the narrative, then uses as much of the
document as fits under the 32,768-token model context. It records whether the
document fit fully.

For the 1,100 BM25 top-50 candidates, 1,079 fit fully and 21 were truncated to
the longest safe prefix.

| formula | nDCG@10 | delta vs BM25 | losses | big losses | worst delta | topic 224 |
|---|---:|---:|---:|---:|---:|---:|
| pure `doc_max_32768_buf512_score` | 0.515422 | +0.101461 | 6 | 1 | -0.1008 | -0.1008 |
| best average long-context aggregate | 0.559599 | +0.145638 | 2 | 1 | -0.1241 | -0.1241 |
| best no-big long-context aggregate | 0.527544 | +0.113583 | 4 | 0 | -0.0948 | -0.0910 |
| previous no-big formula with long-context score | 0.533212 | +0.119251 | 4 | 1 | -0.1151 | +0.1329 |

The best no-big long-context aggregate is:

```text
score =
  0.5 * doc_max_32768_buf512_score
  + 0.5 * top4_weighted_window_score
  + 0.25 * min(relative_span_support_1.0, 6)
```

The old support setting, `relative_span_support_1.5` with weight `0.5`, now fixes
topic `224` strongly, but creates a big topic `515` loss. The tighter
`relative_span_support_1.0` with weight `0.25` is the better conservative choice
with the long-context document score.

`relative_span_support_X` counts chunks whose score is within `X` of the
document's best chunk score. To reduce overlap double-counting, a counted chunk
must add at least 800 previously uncovered characters. The count is capped in the
formula, usually at 6.

## Prefix-1024 Key Results

| formula | nDCG@10 | delta vs BM25 | losses | big losses | worst delta | topic 224 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.413961 | - | - | - | - | - |
| top4_weighted | 0.539943 | +0.125982 | 4 | 1 | -0.1859 | -0.1859 |
| 0.5 prefix-1024 + 0.5 top4 | 0.541780 | +0.127819 | 4 | 1 | -0.1782 | -0.1782 |
| z-normalized blend + support | 0.549646 | +0.135685 | 4 | 1 | -0.1757 | -0.1757 |
| raw blend + bounded support | 0.521673 | +0.107712 | 6 | 0 | -0.0938 | -0.0246 |
| rank top4 + bounded support | 0.500675 | +0.086714 | 9 | 0 | -0.0442 | -0.0442 |

The prefix-1024 no-big-regression formula changes topic `224` top-10 qrel grades
from raw top4's `[4,2,3,3,3,3,4,4,1,4]` to
`[4,4,4,4,4,2,3,3,3,3]`.

Remaining worst deltas for the prefix-1024 formula are topic `515` (`-0.0938`),
`161` (`-0.0755`), `499` (`-0.0255`), `224` (`-0.0246`), `233` (`-0.0210`),
and `897` (`-0.0165`).

## Advisor Feedback

The advisor agreed this is a standard long-document reranking issue: score
windows or passages, then aggregate to a document score. Their main cautions were:

- Keep this PR facet-free unless facets become necessary.
- Test a small family: top-k means, top4 weighting, bounded relative support,
  peakiness penalty, and optional full/top4 blend.
- Prefer relative support such as `s_i >= s1 - delta`, not absolute score
  thresholds.
- Cap support and avoid raw mass/length bonuses.
- Avoid double-counting overlapping chunks.
- Use leave-one-topic-out selection as a sanity check, and freeze any formula
  before using it on blind data.

The expanded sweep followed those recommendations. Raw length and raw mass
features were not reliable. Bounded relative support was useful when combined
with the full/top4 blend.

## Relation To Published Practice

This is consistent with common long-document reranking practice. TREC Deep
Learning separates passage and document ranking/reranking tasks, and document
ranking often uses passage-level signals. TREC systems have used MaxP and
K-Max-AvgP style passage aggregation for BERT rerankers, with K-Max-AvgP reported
as better than MaxP in one TREC 2021 system paper. The Expando-Mono-Duo line uses
the highest-scoring passage as a document representative for later reranking.
PARADE explicitly studies passage aggregation and reports that broader
information needs benefit from document-level passage aggregation.

Sources:

- TREC Deep Learning guidelines: https://microsoft.github.io/msmarco/TREC-Deep-Learning.html
- CIP at TREC 2021 Deep Learning Track: https://trec.nist.gov/pubs/trec30/papers/CIP-DL.pdf
- Comparing Score Aggregation Approaches for Document Retrieval with BERT: https://cs.uwaterloo.ca/~jimmylin/publications/ZhangXinyu_etal_ECIR2021.pdf
- Expando-Mono-Duo: https://arxiv.org/pdf/2101.05667
- PARADE: https://arxiv.org/abs/2008.09093

## Recommendation

For this PR, use the long-context document score plus bounded relative span
support as the candidate production aggregation formula if the goal is to reduce
large regressions without facets or training:

```text
0.5 * doc_max_32768_buf512_score
+ 0.5 * top4_weighted_window_score
+ 0.25 * min(relative_span_support_1.0, 6)
```

This is a modest average improvement over the prefix-1024 conservative formula
(`0.527544` vs. `0.521673` nDCG@10) and uses the available long context for 98%
of the candidate set.

Do not use raw document length, raw count above absolute score thresholds, or raw
score mass as the main coverage proxy. They can improve topic `224`, but they
create worse regressions elsewhere.

The z-normalized long-context blend plus support is useful as an upper-bound
experiment (`0.559599` nDCG@10), but I would not ship it as the default because
it still leaves a large topic `224` regression and adds topic-level calibration
complexity.
