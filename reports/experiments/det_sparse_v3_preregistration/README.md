# `det_sparse_v3` preregistration and offline admission

Date: 2026-07-11

Status: design frozen at commit `9fef81b`; synthetic-only implementation and
pre-candidate admission review are complete with advisor GO. The one canonical
preflight has not yet run. No remaining candidate narrative, qrels, retrieval
result, graded metric, model, reranker, agent, or paid API has been opened or
called.

## Why v3 exists

V1 and v2 were both retired after zero-cost offline shape review:

- V1 copied all of the first unit, so three two-unit child queries reconstructed
  the original.
- V2 bounded that copy to a strict leading prefix, but two selected children
  retained only request boilerplate. “Global measures” lost climate change, and
  “different views on it” lost abortion.

V3 changes only the failed mechanism. It finds a compact source-backed anchor
anywhere in the first unit using cross-unit recurrence. All v1/v2 plans,
outputs, topics, caches, and tickets remain excluded.

## Frozen identities

- Planner: `det_sparse_v3`.
- Renderer: `det_sparse_recurrent_anchor_renderer_v3`.
- Anchor selector: `det_sparse_cross_unit_recurrent_anchor_v1`.
- Selection: `det_sparse_anchor_critical_quantile_selection_v3`.
- Experiment/output: `rag25_det_sparse_structural4_v3`.
- Candidate IDs: `14, 31, 58, 72, 219, 233, 273, 477, 499`.
- Fresh run/global ticket namespaces: `det_sparse_v3_fresh_run_local` and
  `rag25_det_sparse_structural4_v3`.

The complete advisor-approved executable contract and dark-mode-safe workflow
are in `docs/superpowers/det_sparse_v3_design.md`.

## Scientific guardrails

- Exact token surfaces are aligned to the pinned Lucene analyzer; any mismatch
  falls back.
- A versioned 114-surface conversational inventory excludes only exact aligned
  source occurrences, preserving an audit of analyzer stem collisions.
- An anchor needs at least two jointly supported recurrent terms, at least 2/3
  recurrence precision, and no more than half of the first unit's analyzer
  occurrences.
- Separately supported disjoint subjects cause
  `ambiguous_recurrent_anchor`, even when their recurrence strengths differ.
- Every child gets exact anchor plus exact coverage text and must contribute at
  least two eligible nonconversational terms outside the anchor.
- Critical-first selection forces one raw anchorless case, with a final
  unmasked anchorless witness, before choosing three fixed shape quantiles.
- One failed selected shape archives v3 without repair, replacement, or
  resampling.

## Cost and model boundary

Offline admission uses no generative model and makes no external retrieval or
reranking call. This keeps the referent correction isolated. If v3 fails, a
local-model challenger must be a fresh version restricted to exact source-span
selection; it cannot rescue v3 or reuse burned topics. A larger or hosted model
requires measured improvement over the smallest viable local model.

Any future retrieval proposal remains capped at nine calls per topic and 36
total, with exactly 100 results, one attempt, no retry, and no redirect. The
hosted collection still has no immutable revision, so that gate is closed.

## Pre-candidate gate

Before decoding any of the nine narratives:

1. Complete every synthetic anchor, ambiguity, alignment, criticality, payload,
   selection, provenance, and semantic-replay test frozen in the design.
2. Pass the full repository suite and live pinned-Lucene integration.
3. Obtain independent implementation and IR-advisor approval.
4. Commit a clean source tree and restart/attest the immutable local analyzer.

Only then may exactly one create-only qrels-blind canonical preflight run.
Canonical commit/tree/freeze identities, selected topics, shapes, costs, and the
final scientific verdict will be recorded afterward.

## Pre-candidate implementation verdict

The executable v3 boundary now includes the deterministic planner, strict
configuration, critical-first selector, exact source/runtime provenance, and a
create-only semantic preflight. The preflight writes its reservation and frozen
114-surface analyzer projection before candidate access, retains complete plans
for all nine screened candidates, and requires a terminal completion receipt
written only after post-seal attestation and a fresh full replay. A failed or
interrupted build without that receipt can never later validate.

Iterative advisor review found and corrected material issues before any topic
was opened:

- evidence duplicates and selected candidates are compared by the full exact
  evidence identity rather than SHA equality;
- all eight recurrent-anchor score priorities, alignment layers, and recurrence
  evidence fields now have adversarial tests;
- the reused Lucene attestation helper and frozen design are byte-pinned, with
  a complete import/source closure;
- canonical bytes, paths, numeric types, special filesystem entries, request
  arithmetic, all named plan mutations, and selection mutations fail closed;
- a reproduced post-seal transient-drift bug was fixed with the terminal
  completion protocol, and ordinary Python bool/int/float equality was removed
  from provenance and in-memory-config gates.

Pre-candidate verification on 2026-07-11:

- focused v3 suite: `124 passed`;
- full repository: `494 passed, 9 skipped`;
- explicit pinned-Lucene sidecar integration: `11 passed`;
- lint of all new v3 source and tests: clean;
- live local projection: analyzer fingerprint `f9bbd4e7...d8def4`, projection
  `752ce66f...c68b23`, 75 terms;
- attested local image digest:
  `sha256:1eeacc8c295ed4805f6ffead2417b1936aad296b02ea9e56b457230befc9e98d`;
- external retrieval/model/reranker/qrels/paid calls: `0 / 0 / 0 / false / 0`.

Two independent implementation reviews now give GO for a clean offline source
commit. This is not a scientific GO for the intervention and does not open the
external gate. That decision requires the frozen canonical selection followed
by review of only its four qrels-blind selected shapes.
