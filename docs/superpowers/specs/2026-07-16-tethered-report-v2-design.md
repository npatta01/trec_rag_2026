# Tethered Facet Diagnostic Report v2 Design

## Goal

Correct the report's interpretation without modifying the existing generated v1
artifact or any upstream Task 1--4 output. A future builder run emits schema v2,
whose JSON, summary, and standalone HTML all state only conclusions supported by
the authenticated frozen evidence.

## Evidence model

The builder derives and stores these v2 fields:

- `answers.narrative_tether_reduced_noise = false` and
  `answers.narrative_tether_noise_conclusion = "not_established"`.
- Aggregate warning-regex counts by arm and pattern, summed from every validated
  Task 4 diagnostic row. The artifact records that these are crude lexical
  warnings and that a zero `wrong_domain` literal match cannot establish absence
  of semantic drift.
- Representative qrels-grade distributions overall and by movement. These
  disclose all displayed rows and cannot be used as an estimate of net noise.
- A novel-evidence statement whose count is reconciled across Task 4 metrics,
  decision, summary, diagnostics, authenticated qrels, and frozen candidate
  identities. The documents were already in the accepted candidate pool and
  absent from the original top 1,000; MiniLM only reordered, promoted, or
  retained them.
- A mechanical-decision explanation derived from authenticated aggregate and
  per-topic metrics: TETHERED versus RRF Recall@500 and delta, topic 84 versus
  RRF at 500, and protected-head nDCG equality.
- A full accepted-union recall ceiling. For each arm, the builder reconstructs
  exact `(topic_id, document_id)` sets from authenticated Task 3 ranking rows,
  rejects duplicate or inconsistent identities, and requires FACET-2B and
  TETHERED-2B to contain the same set. Per topic and micro, the numerator is the
  count of grade-2+ authenticated qrels documents in that union; the denominator
  is every grade-2+ document in the anchored projection.

No production expected counts or metric values are hardcoded.

## Presentation

The standalone HTML renders:

- “Judged-relevant yield improved; net noise reduction not established.”
- An arm-level warning-regex table and the lexical-warning caveat.
- The novel-pool provenance statement.
- The numerical mechanical-fail explanation, including protected top-100 nDCG.
- The full-union ceiling by topic and micro.
- The displayed representative grade distribution and a warning that examples
  are bounded illustrations, not evidence of aggregate noise reduction.
- The recommended next experiment: offline full-union soft-coverage ranking
  using existing scores, without terminal truncation or equal hard quotas;
  curves at 100, 250, 500, 1,000, 1,500, and full union plus AUC, followed by a
  fresh preregistered topic evaluation.

Tables retain captions and scoped headers, remain horizontally scrollable on
small screens, and the page remains standalone with no external runtime.

## Provenance and failure behavior

All new values come from buffers and hashes already authenticated by the report
builder. The builder describes its qrels use as bounded diagnostic recomputation
from the exact anchored projection; it does not read original qrels, retrieve
documents, or run a new evaluation. It fails closed on Task 3 arm-set drift,
duplicate ranking identities, invalid ranks, missing metric inputs, diagnostic
count disagreement, or any claimed novel document absent from the frozen
accepted candidate identity set.

## Testing

Strict regressions cover:

- v2 schema and the explicit not-established noise conclusion;
- exact arm-level warning sums and restamp resistance;
- novel-document membership and accurate no-discovery wording;
- derived mechanical-fail values and protected-head equality;
- full-union per-topic/micro arithmetic, arm-set equality, duplicate rejection,
  and authenticated-projection restamp resistance;
- representative grade distributions and non-generalization wording;
- the mandated next-step design and accessible HTML structure;
- deterministic output, the full tethered suite, and legacy compatibility.

