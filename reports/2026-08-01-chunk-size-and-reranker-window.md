# Chunk Size, Reranker Window, and a Hypothesis That Did Not Survive

**Date:** 2026-08-01

**Status:** measured, 22 judged dev topics

## Summary

Two questions: how large should a chunk be, and how much of it should the
cross-encoder see. They are separate decisions and were being conflated.

- **The 512-token cap was ours, not the model's.** Fixed; this is a correctness
  fix, not an accuracy one.
- **Chunk size barely matters between 1,200 and 3,500 characters.** A one-topic
  result suggesting otherwise did not survive 22 topics.
- **Reranking is worth a lot**: graded nDCG@20 rises from 0.702 to 0.793.
- **A max-statistic explanation for the one-topic result was disconfirmed** by
  two independent tests.

## The reranker window

`_SNIPPET_MAX_LENGTH = 512` was a hardcoded constant in this repository.
`mxbai-rerank-base-v2` is Qwen2-based with `max_position_embeddings=32768`.

The cost was not mainly lost recall. A truncated chunk was still returned to the
agent with citable sentence spans over its whole text, so an agent could cite
sentences the scorer never read, and the relevance score was not a claim about
them.

Controlled position test — one answering sentence planted at varying positions
inside an otherwise irrelevant 3,500-character filler chunk:

| max_length | needle at 0% | at 50% | at 75% | at 98% |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 7.750 | 6.062 | **2.188** | **2.188** |
| 1024 | 6.750 | 5.500 | 5.750 | 6.812 |
| 2048 | 6.750 | 5.500 | 5.750 | 6.812 |

Filler alone scores 2.188 and the needle alone scores 10.188, so at 512 the tail
signal is not merely weakened, it is entirely absent. 1024 recovers it; 2048 is
byte-identical because a 3,500-character chunk reaches only ~838 tokens.

Raised to 1024. Reranking costs about 75% more. Ranking quality did not improve.

## Chunk size, 22 topics

Depth-100 BM25 pool per topic, documents ranked by MaxP, graded nDCG@20:

| config | nDCG@20 |
| --- | ---: |
| BM25 (control) | 0.7021 |
| chunk 3500 | **0.7926** |
| chunk 2000 | 0.7851 |
| chunk 1200 | 0.7820 |

Paired sign test against 3500: **2000 is 11 wins / 11 losses — exactly even**;
1200 is 16 wins / 6 losses, which is marginal at n=22.

The conclusion holds at the more conventional grade >= 2 binarization, where
P@5 is 0.891 / 0.882 / 0.864 for 3500 / 2000 / 1200.

**Keep 3,500.** Not because larger is better — 3500 versus 2000 is a coin flip —
but because it is the cheapest of the equals, producing the fewest chunks to
score.

## The hypothesis that failed

A single-topic run had shown a large, apparently monotone advantage for bigger
chunks (P@5 0.800 at 3500 against 0.200 at 600). The proposed explanation was
**max-statistic bias**: MaxP takes a maximum over a document's chunks, so
smaller chunks give more draws, inflating long documents regardless of
relevance.

Two tests, both negative.

**1. Direct probe.** Spearman correlation between a document's chunk count and
its rank, computed on **irrelevant documents only**, where any correlation is by
definition an artifact. Lottery-ticket inflation predicts a positive coefficient
rising as chunks shrink.

| chunk size | mean rho | median |
| ---: | ---: | ---: |
| 3500 | −0.151 | −0.130 |
| 2000 | −0.137 | −0.113 |
| 1200 | −0.080 | −0.051 |

Negative at every size: among irrelevant documents, *more* chunks correlates
with ranking *worse*.

**2. Chunk-count-matched control.** Each document's chunks were subsampled down
to the count it had at 3,500 characters, three seeds, so only the number of
draws changed. If inflation were real, matching would remove it and push quality
toward the 3500 result.

| config | matched nDCG@20 | unmatched | delta |
| --- | ---: | ---: | ---: |
| 2000 | 0.7880 | 0.7851 | +0.0029 |
| 1200 | 0.7606 | 0.7820 | **−0.0215** |

Matching did nothing at 2000 and made 1200 notably *worse*. Having more chunks
at 1,200 characters is genuinely useful — more of the document actually gets
scored — which is the opposite of a lottery.

The hypothesis is rejected. The original one-topic result was noise: every
difference in it was one or two documents, and its apparent monotonicity was
already broken at P@5, where 2000 scored below 1200.

## Two cautions recorded for whoever reads this next

**The best single-topic configuration was not what it appeared to be.**
`chunk 3500 / cap 512` never evaluated 3,500-character chunks; it scored roughly
the first 2,000 characters of each 3,150-character stride, which is closer to a
strided lead-text sample. Any claim of the form "large context scores better"
cannot rest on that row.

**Raw cross-encoder logits are not comparable across input lengths.** The head
is under no invariance constraint and position encoding is length-sensitive.
Rank-based evaluations such as those above remain valid, because each run is
internally consistent. Absolute-score thresholds and cross-configuration score
comparisons are not.

## Open, not concluded

A chunk-count-penalised MaxP (`max − 0.35·√n`) scored best on binary P@5 in the
first sweep, consistently across all three chunk sizes (+0.055 each). The
coefficient was guessed, the gain has not been reproduced under graded nDCG, and
the mechanism originally proposed for it is now disconfirmed. It is recorded as
an unexplained signal, not adopted. Estimating `E[max | n]` from irrelevant
documents would be the principled version.
