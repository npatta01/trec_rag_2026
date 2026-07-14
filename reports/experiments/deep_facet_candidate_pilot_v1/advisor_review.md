### Verdict

`diagnosis = fusion` is justified as a **pipeline-stage diagnosis**: facet retrieval found useful documents, the stream gate preserved nearly all of them, and the failure occurred when ranking the accepted union.

It should not be interpreted as “facet candidates are irrelevant.” The exact failure is an inadequately calibrated early-ranking objective, compounded by differential qrels coverage.

### Evidence

- `U_raw` found 179 grade-≥2 documents beyond original@1000.
- `U_accepted` preserved 177 across all four topics:
  - 219: 19
  - 72: 77
  - 300: 60
  - 84: 21
- The single rejected stream lost only two known-relevant documents.
- DUAL retained 155/177 novel documents at 1000, versus 95/177 for RRF.
- DUAL slightly beat RRF on aggregate graded Recall@1000: `0.2799` versus `0.2659`.
- But DUAL lost at 500: `0.1855` versus RRF `0.2101`, and nDCG@10 fell from `0.4326` to `0.3344`.
- DUAL lost nDCG@10 to RRF on every topic and failed every leave-one-topic-out check.

This is a deep-ranking success but an early-allocation failure.

### Important caveats and red flags

1. **Judging coverage strongly favors RRF near the head.**

   - Judged@100: RRF `97.25%`, DUAL `36.75%`
   - Judged@500: RRF `36.75%`, DUAL `28.15%`
   - Judged@1000 is much closer: RRF `23.38%`, DUAL `22.83%`

   Because unjudged documents receive zero gain, shallow nDCG and recall understate facet-heavy arms. The no-advance decision is valid, but the magnitude of DUAL’s apparent early-precision loss is uncertain.

2. **The DUAL features are not calibrated relevance probabilities.**

   `G`, `N`, and `R` are percentiles over roughly 1,700–2,200 documents; each `F_i` is a percentile over 200. Equal percentile values do not imply equal relevance likelihood. Static coefficients assume comparability that has not been demonstrated.

3. **`L=max_i F_i` has a multiple-opportunity bias.**

   Topics with seven facets give documents more chances to receive a high local score than the four-facet topic. This can over-promote narrow facet matches.

4. **The quality gate is intentionally permissive.**

   One accepted stream had content warnings in four of its top five documents. Thus “accepted” means coherent enough to retain, not high-quality enough for early ranking.

5. **Redundancy is not the main problem.**

   DUAL and DUAL-NR are nearly identical. Changing the Jaccard penalty is unlikely to fix relevance.

### One bounded next experiment

Test one deterministic **RRF–GLOBAL–DUAL cascade**, using the existing sealed union and scores:

1. Ranks 1–10: exact RRF order.
2. Ranks 11–100: GLOBAL order, skipping already selected documents.
3. Ranks 101 onward: DUAL order, skipping duplicates.
4. Append every remaining `U_accepted` document in DUAL order, preserving the complete permutation and all 177 discovered candidates.
5. Add no retrieval, inference, new weights, filtering, or iterative tuning.

Why this arm:

- RRF protects nDCG@10 exactly.
- GLOBAL was stronger than DUAL at early aggregate relevance.
- DUAL supplies deep facet coverage and retained 155/177 novel documents at 1000.
- The arm isolates positional budgeting rather than redesigning retrieval.

This evaluation is diagnostic because the current topics are qrels-exposed. Freeze the cascade before computing its metrics and do not tune its boundaries afterward.

### Stop rule

Stop without another repair if any condition fails:

- nDCG@10 differs from RRF;
- graded Recall@500 does not exceed both RRF and GLOBAL;
- graded Recall@1000 falls below RRF;
- novel retention@1000 is below 80% (`142/177`);
- any topic loses more than `0.02` graded Recall@500 versus RRF; or
- differential judged coverage prevents a defensible comparison.

If the arm passes, it justifies one fresh preregistered validation—not production promotion.
