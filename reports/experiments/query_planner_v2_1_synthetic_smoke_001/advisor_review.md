# Advisor review: query planner v2.1 synthetic smoke 001

Date: 2026-07-11

## Verdict

**NO-GO for the formal known-five diagnostic. Stop and freeze the current
`gpt-oss-20b` v2.1 planner arm.**

The transport, pinned runtime, XGrammar schema, JSON envelope, finish reason,
raw-first ledger, and fallback all behaved as designed. The preregistered
terminal outcome was nevertheless `plan_validation_error`, so it did not clear
the admission rule requiring a completely valid synthetic first emission and
advisor approval before the known-five run.

This is a protocol decision from one admission topic, not a claim that every
20B model is generally incapable. Topics 144, 213, 224, 407, and 515 remain
untouched under v2.1.

## Mechanical findings

1. The emitted global anchors had kinds `comparison`, `geography`, and
   `geography`; none was `entity` or `topic`, despite the explicit prompt rule.
   Python correctly rejected the plan with `global anchors must include an
   entity or topic anchor`.
2. For diagnosis only, the advisor parsed an in-memory copy after changing the
   first anchor kind to `topic`. That exposed a second independent validation
   failure: global expansion term `maintenance` referenced the coverage-scoped
   `maintenance_indicators` anchor. No stored artifact was changed, and this
   was not accepted as a repair or rerun.

Because the first error was not the only mechanical issue, post-hoc prompt,
schema, or validator tuning from this output is especially inappropriate.

## Prohibited continuation

- no retry or validator-guided repair;
- no prompt/schema tuning from the synthetic semantics;
- no known-five v2.1 calls;
- no Qwen/larger-model download or challenger run;
- no retrieval or reranking with the invalid plan.

## Next justified direction

Open a separate, preregistered **non-generative sparse retrieval arm**:

1. retain original-narrative BM25 as a permanent stream;
2. split request clauses/facets deterministically while carrying shared parent
   context into every facet;
3. retrieve each facet independently;
4. fuse original and facet rankings with reciprocal-rank fusion;
5. test lexical expansion separately, preferring conservative dictionary/
   acronym expansion or query-side/corpus-derived pseudo-relevance feedback.

Evaluate four explicit arms:

1. original;
2. original + facets;
3. original + expansion;
4. original + facets + expansion.

This directly answers whether decomposition and expansion are complementary
without conflating them or paying for a larger planner. Only a separate future
protocol could authorize an agent/controller as an exception handler after
these deterministic ablations establish where vocabulary or coverage failures
remain.
