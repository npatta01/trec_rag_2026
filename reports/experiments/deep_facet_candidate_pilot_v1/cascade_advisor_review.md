# Independent cascade advisor review

## Verdict

The cascade is implemented correctly and the mechanical decision to stop is
correct. It proves that positional protection can restore RRF's head while
preserving deep facet coverage, but it does not solve selection at ranks
11–500. The next useful test is a bounded, stronger reranker over the
disagreement pool, not another hand-tuned fusion cutoff.

This remains post-qrels diagnostic evidence. It cannot authorize production or
serve as confirmation on these four topics.

## Implementation and decision audit

The sealed ranking exactly matches the frozen rule for every topic:

- ranks 1–10 are the exact RRF head;
- ranks 11–100 are the first 90 unseen GLOBAL documents;
- ranks 101 onward follow DUAL, skipping prior selections;
- every topic is a complete, duplicate-free permutation of its original
  `U_accepted` candidate set; and
- the freeze records zero retrieval and inference calls and no qrels read while
  constructing the arm.

The metric and decision relationships are internally consistent:

- nDCG@10 exactly equals RRF: `0.4326`;
- graded Recall@500 is `0.1870`, below RRF `0.2101` and GLOBAL `0.1979`;
- graded Recall@1,000 is `0.2804`, above RRF `0.2659` but just below GLOBAL
  `0.2821`;
- the arm retains `155/177` novel relevant documents at 1,000, satisfying the
  frozen `142/177` floor; and
- the per-topic Recall@500 floor fails on topics 219 (`-0.0613` versus RRF) and
  72 (`-0.0209`).

The cascade therefore improved DUAL's head and preserved DUAL's deep novelty,
but its middle still failed the preregistered comparison. The no-advance result
is correct.

## Judging-coverage caveat

Comparison with RRF is not defensible as a clean estimate of shallow relevance:

- judged@100: cascade `47.0%`, RRF `97.25%`;
- judged@500: cascade `28.75%`, RRF `36.75%`; and
- judged@1,000: cascade `22.90%`, RRF `23.38%`.

Because unjudged documents receive zero gain, the large shallow gap favors RRF.
The exact nDCG@10 equality is nevertheless valid because the top ten documents
and order are identical. The depth-1,000 comparison is also materially more
credible because coverage is nearly matched.

GLOBAL is a more informative middle-depth control: judged@500 differs by only
one aggregate percentage point (`28.75%` versus `29.75%`), and the cascade still
loses graded Recall@500 by `0.0109`. Coverage is not perfectly balanced per
topic—especially topic 72—so this is diagnostic rather than conclusive, but the
result does not support advancing the cascade.

## Interpretation

Another fixed splice or coefficient adjustment is unlikely to resolve the main
problem. The cascade can choose *where* each existing ranker controls the list,
but it cannot decide which facet-only documents are globally useful enough to
displace strong original-query evidence. Moving the 101 boundary after seeing
these qrels would be cutoff tuning on exposed topics.

The evidence instead supports targeted stronger relevance estimation:

- RRF remains strongest at the head;
- the facet union contains 177 known novel relevant documents;
- DUAL retains 155 of them at 1,000;
- GLOBAL/DUAL/CASCADE remain weak or uncertain in the middle; and
- matched depth-1,000 coverage shows that deep recall exists, while the
  unresolved task is promoting the right subset earlier.

## One bounded next experiment

Run one **protected-head targeted cross-encoder rerank**:

1. Keep RRF ranks 1–10 unchanged.
2. Form one qrels-blind residual pool per topic from the union of RRF@500,
   GLOBAL@500, and DUAL@1,000, excluding the protected head.
3. Score that pool once with the previously strongest available cross-encoder,
   `mixedbread-ai/mxbai-rerank-base-v2`, using one identical structured query per
   topic containing the original narrative plus the complete accepted-facet
   obligation list. This makes scores comparable within a topic and asks about
   relevance to any requested information need without comparing scores across
   facet queries.
4. Select the best 490 documents for ranks 11–500, then append all remaining
   documents in frozen DUAL order. The complete `U_accepted` permutation remains
   intact, so no discovered candidate is discarded.
5. Before inference, freeze the exact document/window count, cache coverage,
   runtime, memory, and cost. Stop for approval if the model is not already
   materialized or the bounded run exceeds the agreed local-compute budget.
6. Run one arm only, with no weight, cutoff, prompt, or candidate-pool tuning.

These qrels-exposed topics may be used only for a mechanism diagnostic. Any
effectiveness claim requires the same frozen arm on fresh topics, with qrels
opened only after its ranking is sealed.

### Stop rule

Stop without another fusion repair unless the targeted arm:

- preserves RRF nDCG@10 exactly;
- exceeds both RRF and GLOBAL graded Recall@500;
- remains at least RRF on graded Recall@1,000;
- retains at least 80% of novel relevant evidence at 1,000;
- loses no more than `0.02` graded Recall@500 on any topic; and
- has comparable judging coverage, or receives blind judgments for its
  top-500 disagreement documents.

If judged coverage remains materially unequal, label the result inconclusive
rather than treating unjudged candidates as irrelevant.
