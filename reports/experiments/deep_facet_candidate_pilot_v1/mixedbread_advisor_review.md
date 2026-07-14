# Independent protected-head Mixedbread review

## Verdict

The Mixedbread diagnostic is a meaningful mechanism success, but it is not a
safe overall win and the frozen no-promotion decision is correct. Stronger
reranking improved graded gain at 500 and 1,000 while preserving the RRF top
ten and nearly all deep novel evidence. It also reduced binary Recall@500,
substantially damaged nDCG@100, and failed badly on topic 219.

Do not adjust this query, cutoff, or score on these qrels-exposed topics. The
next test should be one preregistered, facet-local Mixedbread design on fresh
topics, with an explicit original-query basket.

## Artifact and implementation audit

Verified from the sealed artifacts and a complete semantic recomputation:

- the residual pool is exactly the union of RRF@500, GLOBAL@500, and DUAL@1,000,
  excluding the protected RRF top ten;
- ranks 1–10 are the exact RRF order;
- ranks 11–500 are the 490 strongest aggregated Mixedbread residual scores,
  with frozen DUAL rank as the tie-break;
- ranks 501 onward are the remaining complete DUAL order;
- all 8,114 topic-document candidates form duplicate-free complete
  permutations, so all 177 known novel relevant candidates remain present;
- preflight, scoring, and ranking construction record no retrieval, network, or
  paid call and no qrels read before the ranking seal; and
- 36,435 windows for 4,469 residual documents were scored locally in 4,458
  seconds at 8.17 windows/second. Peak device and host memory were approximately
  2.74 GB and 3.56 GB.

The sealed ranking root `936b9a96d9b7fc2f984dca63d7e512bcfa8d382caef3e9e56999ee36c2166dc1`
recomputed successfully from its bound source rankings, windows, and scores.

## What worked

The following are verified results:

- nDCG@10 is exactly RRF: `0.4326`;
- graded Recall@500 is `0.2161`, above RRF `0.2101`, GLOBAL `0.1979`, DUAL
  `0.1855`, and the positional cascade `0.1870`;
- graded Recall@1,000 is `0.2869`, above RRF `0.2659` and DUAL `0.2799`;
- 99/177 novel relevant documents survive at 500 and 153/177 at 1,000;
- all 177 remain in the complete ranking; and
- per-topic graded Recall@500 improves over RRF on topics 72 (`+0.0515`), 300
  (`+0.0095`), and 84 (`+0.0029`).

This establishes that a stronger cross-encoder can identify useful high-grade
facet evidence better than the prior fixed fusion rules. It also shows that the
candidate-generation work should be retained.

## Why this is not a safe win

### Binary coverage fell

Aggregate binary Recall@500 fell from RRF `0.1976` to `0.1791`, even though
graded Recall increased. Mixedbread therefore selected fewer grade-2-or-higher
documents overall while favoring higher-grade documents strongly enough to win
the exponential-gain metric.

At 500, the relevant-document counts changed as follows:

- topic 219: 85 to 53;
- topic 72: 199 to 187;
- topic 300: 130 to 131; and
- topic 84: 153 to 148.

This trade is unsuitable if the downstream goal values broad evidence recall,
not only a smaller set of highly graded passages.

### The protected top ten masks ranks 11–100

nDCG@10 is safe only because those ten positions were copied from RRF. At 100:

- nDCG falls from RRF `0.4048` to `0.2727`;
- graded Recall falls from `0.1155` to `0.0675`; and
- judged coverage falls from `97.25%` to `45.25%`.

The new arm does retrieve 23 novel relevant documents at 100, versus zero for
RRF, but it displaces too much established original-query evidence. A production
system that consumes more than ten candidates would still experience this loss.

### Topic 219 remains a decisive failure

At 500, topic 219 loses 40 RRF-relevant documents and gains only eight, for a
net graded-gain loss of 124. Thirty-seven of the 40 lost documents have original
query provenance. Broad grade-4 AI and societal-impact documents are displaced,
while the strongest gained documents disproportionately emphasize the narrower
telehealth facet. At 100 the loss is larger; by 1,000 the net graded-gain gap is
only two, so this is primarily an ordering failure rather than candidate loss.

Mixedbread score separation is also weak for topic 219: median residual scores
are `5.409` for relevant, `5.295` for judged nonrelevant, and `5.253` for
unjudged documents. The relevant/nonrelevant separation is materially larger on
the other three topics.

## Judging-coverage limitation

The RRF comparison remains biased toward RRF because unjudged documents receive
zero gain:

- judged@100: Mixedbread `45.25%`, RRF `97.25%`;
- judged@500: Mixedbread `30.75%`, RRF `36.75%`; and
- judged@1,000: Mixedbread `23.25%`, RRF `23.38%`.

The depth-1,000 improvement is the most defensible because coverage is nearly
identical. At 500, the six-point aggregate gap makes the exact advantage over
RRF uncertain. Mixedbread does beat GLOBAL at 500 with nearly comparable judged
coverage (`30.75%` versus `29.75%`), which supports a real gain over the weaker
common-query semantic control. It does not remove the topic-219 or raw-recall
concerns.

## Did the all-facet query dilute relevance?

**Inference:** probably on topic 219, but the experiment does not prove it.

The topic-219 query concatenates a broad narrative with seven heterogeneous
obligations: daily-life effects, government, business, telehealth, technical
societies, and device rationing. The observed promotion of telehealth-specific
documents, loss of broad societal-impact evidence, and weak score separation
are consistent with a cross-encoder rewarding one conspicuous facet while not
calibrating comprehensive topic relevance.

**Unknown:** there is no matched Mixedbread facet-local arm, so query dilution
cannot be separated from model calibration, projected-qrels coverage, long-text
window aggregation, or topic-specific candidate quality.

The result supports testing narrative-tethered facet-local scoring next. It does
not support an unconstrained global `max-over-facet` ranking: that would repeat
the multiple-opportunity and facet-flooding risks already observed. A max local
score is reasonable only inside a bounded facet basket while an original-query
basket is protected separately.

## One bounded next experiment

Preregister one **two-basket, facet-local Mixedbread** arm on fresh qrels-blind
topics:

1. Preserve RRF ranks 1–100 exactly.
2. For each accepted facet's existing 200 candidates, score each document with
   one query containing the full narrative plus only that generating facet as a
   `Focus:` obligation. Do not concatenate all facets.
3. Convert scores to percentiles within each facet; for a document retrieved by
   multiple facets, retain its maximum generating-facet percentile. Raw scores
   never cross facet-query boundaries.
4. Fill ranks 101–500 from two deterministic baskets:
   - 200 next-unseen RRF documents, preserving the broad original-query basket;
   - 200 facet documents, selected by facet-local percentile with equal
     per-facet slot budgets and manifest-order remainder allocation.
   Interleave the two baskets and skip duplicates.
5. Append every remaining candidate in frozen DUAL order, preserving the full
   candidate union.
6. Freeze exact candidate/window counts, cache coverage, runtime, and memory
   before separately approved inference. Run one arm only and do not tune the
   100/200/200 allocation after qrels access.

This directly tests whether shorter narrative-plus-one-facet queries recover
specific evidence without sacrificing the broad original basket. It bounds the
effect of max-over-facet scoring rather than allowing it to dominate the list.

### Stop rule

Do not run another repair if the fresh evaluation fails any of these:

- nDCG@100 is exactly RRF by construction;
- binary Recall@500 is not below RRF;
- graded Recall@500 exceeds RRF;
- at least 50% of discovered novel relevant documents survive at 500;
- binary and graded Recall@1,000 are not below RRF;
- novel retention@1,000 is at least 80%;
- no topic loses more than `0.02` binary or graded Recall@500 versus RRF; and
- judging coverage is comparable, or the disagreement pool receives blind
  judgments.

The current four topics may inform this frozen design but must not be used to
tune or confirm it. Only fresh topics with rankings sealed before qrels access
can determine whether the method should advance.
