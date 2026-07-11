# `det_sparse_v3` preregistration and offline admission

Date: 2026-07-11

Status: archived scientific no-go. The design was frozen at commit `9fef81b`,
the reviewed implementation was committed at `169c093`, and the one canonical
qrels-blind preflight completed with zero eligible topics. No retrieval result,
qrels, graded metric, model, reranker, agent, or paid API was opened or called.

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

Two independent implementation reviews gave GO for the clean offline source
commit. That was not a scientific GO for the intervention and did not open the
external gate. The canonical result below failed before four shapes existed.

## Canonical v3 result

The single allowed canonical run used source commit/tree
`169c0938c7b5f6af73a490c64957ff45c43a49e7` /
`b918add18e556ce377f41ab2ae33bc2d64821a7c`. It completed its post-seal fresh
replay and terminal receipt, then stopped with:

```text
selection status: failure
failure code: fewer_than_four_eligible_topics
eligible / candidates: 0 / 9
selected topics: none
mechanical_valid: false
```

Eight candidates returned `recurrent_anchor_unavailable`; one returned
`token_analyzer_alignment_mismatch` before recurrence screening. The redacted
structural audit (no narrative or query text) was:

| Topic | Units | Recurrent identities | Windows | Admissible | Failure |
|---:|---:|---:|---:|---:|---|
| 14 | 2 | 2 | 123 | 0 | recurrent anchor unavailable |
| 31 | 2 | 2 | 141 | 0 | recurrent anchor unavailable |
| 58 | 2 | 1 | 141 | 0 | recurrent anchor unavailable |
| 72 | 5 | 0 | 57 | 0 | recurrent anchor unavailable |
| 219 | 3 | 1 | 141 | 0 | recurrent anchor unavailable |
| 233 | 2 | 0 | 75 | 0 | recurrent anchor unavailable |
| 273 | 5 | 2 | 105 | 0 | recurrent anchor unavailable |
| 477 | — | — | — | 0 | token/analyzer alignment mismatch |
| 499 | 2 | 0 | 105 | 0 | recurrent anchor unavailable |

Across the eight aligned plans, 888 windows produced zero admissible anchors
and zero recurrent cores. Recurrent cardinality was 711 windows with zero
identities, 176 with one, and only one with two. The two-identity window was not
a near miss: it used six token records and five unique terms, had no joint raw
child support, and simultaneously failed the unique-term, recurrence-precision,
and joint-support gates. In other two-identity plans, the minimum span joining
both identities required seven or eight token records, already outside the
frozen six-record bound. Seventy-five singleton windows avoided the other
recorded rejection reasons, but admitting them would relax both the two-term
referent guarantee and the coupled precision rule, leaving many ambiguous
choices rather than fixing semantic reference.

This is a recurrence-recall/co-location failure, not a score, tie-break, topic
selection, or retrieval-ranking failure. Relaxing to one recurrent term would
weaken referent specificity; dropping joint support would admit split,
incoherent evidence; widening the window after viewing this run would be a
post-hoc repair. V3 is therefore archived without replacement, resampling, or
rule change, and selected-shape review is skipped because no selected shapes
exist. The entire nine-topic v3 universe is consumed for future tuning; a
challenger must use untouched topics.

## Seal and cost evidence

- freeze SHA-256: `6a3b0aa730a94bf9005ffddbb040d1ef7a603619389c4295d0046ea643027c69`;
- terminal receipt SHA-256:
  `70f22ad9af727cc7ae075d83487c838bae31c9f5344b22543af0c48e850d596a`;
- metadata SHA-256: `a91cb98baa3978d053edbb5af0fd7a5cfba00ee2ae39a224bed39e97838cafc4`;
- selection SHA-256: `bb35ef48e8912651a7f2302d331ac3434595be421e5b88a22a472fc4b91f25ae`;
- candidate-screen SHA-256:
  `cfec0a00bdaa636fbc06cd1df7f2c88705034e240300a5dd5f5c0a79fb3a14ec`;
- 18 unique sealed artifacts and 20 exact observed files including the freeze
  and terminal receipt;
- planned/derived requests: `0 / 0`;
- external/model/reranker/qrels/paid calls: `0 / 0 / 0 / false / 0`;
- external gate: `blocked` (`hosted_index_revision_unknown`).

Two post-run reviewers independently confirmed the complete but mechanically
invalid archive. It cannot authorize retrieval.

## Recommended next version

Use a new untouched-topic version that changes only the failed anchor selector:
one pinned, schema-constrained call to the smallest viable local model consumes
the exact full token tape and units and returns only `select|abstain` plus a
consecutive first-unit token range. It may not generate terms, retry, repair, or
run an agent loop. Recurrence remains an audit signal, not a hard prerequisite.
Deterministic child grouping, protected parent, payload rules, sparse admission,
cost firewall, and human shape review remain unchanged. Fix the one alignment
failure generically with whole-unit Lucene token offsets mapped back to exact
token records, not by weakening anchor admission. A larger or hosted model
remains disallowed unless a separately measured challenger materially improves
on the local arm.
