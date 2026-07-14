# Independent advisor memo

## Verdict

Keep `RRF` as the promoted arm. The mechanical decision is correct.

However, the experiment does **not** show that facets or facet-local MiniLM failed. It shows that the tested xQuAD relevance term and its unbounded list allocation were unsafe:

- Facet retrieval found substantial new judged-relevant evidence.
- RRF used facet agreement effectively to improve top-rank ordering.
- XQ/CXQ admitted far too many facet-only documents.
- Incomplete judging materially exaggerates the apparent XQ/CXQ collapse.

The right conclusion is: **retain facets, replace the fusion design.**

## Verified results

| Arm | nDCG@10 | Graded Recall@100 | Relevant@10 | Judged@100 |
|---|---:|---:|---:|---:|
| Original | 0.4720 | 0.11086 | 7.25 | 100.0% |
| RRF | **0.5895** | **0.11098** | **8.00** | 98.25% |
| BI | 0.4264 | 0.07920 | 6.50 | 71.0% |
| XQ/CXQ | 0.2836 | 0.05758 | 4.75 | 51.5% |
| TUS-C | 0.4848 | 0.10750 | 7.75 | 95.0% |

RRF improved nDCG@10 over the original on all four topics, while aggregate Recall@100 was essentially unchanged.

## Why RRF improved nDCG

All 40 RRF top-10 documents were already in the original top-100. RRF therefore improved nDCG primarily by **reranking original candidates**, not by flooding the head with new facet-only documents.

It promoted original documents that also appeared strongly in one or more facet lists. Examples include original ranks 87, 25, 23, and 19 being promoted because facet-local MiniLM supplied corroborating evidence.

This is a useful facet signal: cross-stream agreement identified better full-topic documents.

## Why XQ/CXQ retained 66 novel documents but lost effectiveness

The xQuAD relevance term was:

`Rel(d) = max(original-rank score, best facet-rank score)`

That makes every stream’s rank-1 document globally equivalent:

- original rank 1 → `Rel = 1`
- every facet rank 1 → `Rel = 1`

Thus a document that is merely best for one narrow facet receives the same base relevance as the best original-query document. The “Rel” term was therefore not global topic relevance; it was the best local rank from any stream.

The consequence was severe:

| Arm | Facet-only documents in top 100 across four topics |
|---|---:|
| RRF | 21 / 400 |
| TUS-C | 38 / 400 |
| BI | 195 / 400 |
| XQ/CXQ | **311 / 400** |

XQ replaced 290 RRF documents. Under available judgments, it gained 75 relevant documents but lost 207 relevant RRF documents.

Its high novel retention—66/73—is real, but it was achieved by spending most of the ranking budget on facet streams. Novel retention alone can therefore be maximized while overall recall falls.

The main defect is a **fusion-budget and relevance-calibration problem**, not evidence that facet-local MiniLM cannot identify useful documents.

## Novelty-definition audit

The 73-document denominator is:

> Unique grade-≥2 documents in accepted-facet MiniLM top-20 lists that are absent from RRF@100.

Therefore RRF retaining zero of these 73 is true **by construction**. It is not an independent criticism of RRF.

Nevertheless, the underlying recall finding is strong:

- Accepted facet top-20 lists contained **83 judged-relevant documents absent from the original top 100**.
- RRF recovered some of them, leaving **73 judged-relevant documents absent from both O and RRF**.
- Counts for those 73 were:
  - topic 233: 16
  - topic 273: 7
  - topic 161: 28
  - topic 14: 22
- XQ/CXQ retained 66: all novel documents for three topics and 15/22 for topic 14.

Thus the experiment does prove that facet decomposition plus local MiniLM adds useful recall candidates. The failure is getting those candidates safely into a depth-100 result.

## Judging artifact

The evaluation treats unjudged documents as grade zero, but coverage differs sharply:

- Original: 100% judged at 100
- RRF: 98.25%
- XQ/CXQ: 51.5%

Of the 290 documents XQ added relative to RRF:

- only 103 were judged;
- 187 were unjudged.

All 290 RRF documents it displaced were judged.

Among judged XQ top-100 documents, 159/206 were grade ≥2, versus 291/393 for RRF. This judged-only comparison is not unbiased because missingness is highly non-random, but it shows that the evidence does not support calling all facet additions irrelevant.

Therefore:

- The preregistered metrics correctly block promotion.
- The magnitude of the apparent XQ collapse is partly a judging-coverage artifact.
- We should say “XQ did not demonstrate safe improvement,” not “XQ documents were proven irrelevant.”

## Why XQ and CXQ were identical

The files are byte-identical, and every one of the 400 CXQ selections was recorded as an ordinary `xquad` selection. No deadline insertion fired.

Ordinary XQ had already covered all accepted facets extremely early:

- topic 233: by rank 2
- topic 273: by rank 5
- topic 161: by rank 4
- topic 14: by rank 5

This is not evidence of a bug. It means the CXQ deadlines were inactive and the pilot provides no independent evidence about deadline forcing. It also confirms that base XQ was already over-aggressive about facet coverage.

## Single best next bounded experiment

Test a **protected RRF backbone with globally scored residual facet slots** on fresh held-out topics.

1. Reuse the same BM25 and accepted facet candidates.
2. Score the complete topic-level candidate union once with MiniLM using the **full original narrative**. Because every document receives the same query, these scores are comparable within the topic.
3. Keep existing facet-local MiniLM ranks as the coverage signal.
4. Freeze RRF ranks 1–80 unchanged.
5. For the remaining 20 positions, select from:
   - RRF ranks 81–100; and
   - accepted-facet top-20 candidates absent from the protected head.
6. Rank that residual pool by full-narrative MiniLM score, with at most two new documents per facet and no forced insertion.
7. Keep an RRF tail document whenever its global narrative score exceeds the proposed facet replacement.
8. Evaluate only after ranking freezes, and ensure every added output document is judged—or blind-adjudicate the at-most-80 additions across four topics.

This directly tests the missing capability:

> Can globally relevant facet evidence replace only weak RRF-tail documents?

It preserves nDCG@10 exactly by construction, limits maximum disruption to 20%, adds no retrieval calls, uses the already-cached cheap MiniLM, and avoids comparing facet-local scores across queries.
