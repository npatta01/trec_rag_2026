# Bounded Splice Revision Results

**Date:** 2026-08-08
**Status:** topic-707 mechanism trial inconclusive; stopped before a third attempt or any
additional topic.

## Decision

The deterministic splice implementation passed its offline contract and integration checks, but
neither live topic-707 attempt produced a draft-to-splice pair. Do not interpret this run as evidence
that bounded splicing improves or harms answer quality.

Stop on topic `707`: the two attempts made four Sol requests in total, including one zero-cost
pre-completion schema rejection. Another attempt would move farther beyond the user's practical
three-Sol-call-per-topic limit. Do not run a post-hoc nugget or citation comparison because there is
no genuine paired revision to compare.

The next decision is whether to use a new fresh topic under the corrected schema, or first run a
cheaper Luna-only mechanical splice probe over private existing artifacts. Either option must be
explicitly labeled; the latter would test transport/schema/application but would not test Sol
revision quality.

## Offline implementation evidence

- The splice schema, validator, deterministic applier, and repair-provenance checks passed 15
  focused tests.
- The combined splice and narrative-blueprint integration set passed 33 tests.
- Ruff, Python compilation, and diff checks passed.
- Two independent Luna reviews found no actionable contract or integration issue.
- A scoped re-review approved the provider-schema compatibility fix.
- The supported competition runner and checked-in full-run configs were not changed.

## Live attempt 1

**Namespace:** `rag26-bounded-splice-707-20260808`

- Scope: topic `707` only.
- Extraction diagnostic: strict full retrieval-nugget coverage `1.0`.
- Authenticated handoff: 4 generated groups, 58 claim hints, 190 selected passages, and 149
  citation-domain documents.
- Completed: one Luna planner, one Sol draft, four Luna group audits.
- Draft: valid, 900 words, 47 answer objects.
- Revision: OpenRouter returned HTTP 400 before a semantic response because the strict provider
  schema contained unsupported top-level `allOf` / `if` / `then` keywords.
- Repair: not called because no parsed splice payload existed.
- Final: the valid draft rebound under the final run identity; no answer-text revision occurred.
- Provider-reported cost: **`$0.195945500`**.

The exact provider diagnostic identified the root cause. Working strict schemas in this repository
leave cross-field semantics to local validation, while the new splice schema attempted to encode
the `keep_draft`/`edit` relationship with provider-side conditional keywords. Commit `e0c368f`
removed only those unsupported keywords. Local validation still enforces that `keep_draft` has no
operations and `edit` has at least one.

## Live attempt 2

**Namespace:** `rag26-bounded-splice-707-r2-20260808`

- Scope and authenticated handoff: identical to attempt 1, under a new experiment identity and
  clean reservation ledger.
- Completed: one Luna planner, one Sol draft, one Sol deterministic draft repair.
- Initial draft: 1,145 words and 65 answer objects, exceeding the organizer's 1,024-word ceiling.
- Repair: returned a valid 823-word answer with 58 answer objects.
- Audits and quality revision: skipped by construction because the draft failure consumed the sole
  repair allowance; spending a splice revision afterward would exceed the three-reservation topic
  ceiling.
- Final: valid repaired draft, not a bounded splice result.
- Provider-reported cost: **`$0.417561260`**.

This attempt did not submit the corrected splice schema to the provider, so live provider
acceptance of that schema remains unverified despite the focused regression test and scoped code
review.

## Combined call and cost ledger

| Attempt | Luna requests | Sol requests | Sol roles | Cost |
|---|---:|---:|---|---:|
| `707` | 5 | 2 | draft; revision schema rejection | `$0.195945500` |
| `707-r2` | 1 | 2 | draft; deterministic draft repair | `$0.417561260` |
| **Total** | **6** | **4** | one zero-cost pre-completion rejection | **`$0.613506760`** |

No RAGDoll/DeepSeek nuggetizer or citation-support calls were made for this experiment.

## What the result establishes

Verified:

- The fail-closed behavior worked in both attempts.
- A provider schema rejection did not trigger an unconstrained repair call.
- An over-limit initial draft used the only deterministic repair reservation and could not flow
  into a fourth semantic call.
- Both attempts sealed structurally valid final output without exposing private artifacts.
- Old state was not overwritten or resumed under a changed schema; attempt 2 used a new namespace.

Not established:

- Whether Sol can reliably emit a valid splice payload under the corrected schema.
- Whether splice operations improve coverage, coherence, caveat preservation, or citation support.
- Whether the provisional six-operation/four-insertion/eight-touched-object budgets are useful.
- Whether deterministic splicing creates acceptable discourse seams.

## Why no qualitative or automatic pair verdict is reported

Attempt 1's final answer differs from its draft only in bound metadata, not answer text. Attempt 2
has no retained valid first draft and no quality revision. A draft-versus-final score table would
therefore be misleading. The organizer evaluation remains multi-objective—pairwise preference,
nugget coverage, weighted citation precision, and weighted citation recall—but none can answer the
splice question without a real pair.

## Privacy and provenance

Aggregate counts, outcomes, hashes, and provider-reported costs were verified from the private
manifest-last receipts and state ledgers under the two ignored output namespaces. This report
contains no narrative, answer, passage, prompt, audit-card, raw provider response, or individual
evaluation judgment.
