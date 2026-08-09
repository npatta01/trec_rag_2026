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
manifest-last receipts and state ledgers under the ignored output namespaces. This report
contains no narrative, answer, passage, prompt, audit-card, raw provider response, or individual
evaluation judgment.

## Deadline attempt: fresh topic 897

The two final-review safety findings were fixed before this run: recovered revision payloads are
bound to the current prompt/schema hashes, and an invalid draft now stops before audits or
revision instead of spending the splice-repair reservation on a full-answer rewrite. The focused
gate passed with 35 tests, Ruff, `py_compile`, `git diff --check`, and a zero-call dry run.

**Namespace:** `rag26-bounded-splice-897-20260808`

- Extraction diagnostic: strict full retrieval-nugget coverage `1.0`.
- Authenticated handoff: 5 groups, 87 claim hints, 226 selected passages, and 162 citation-domain
  documents.
- Hosted generation: 6 Luna calls and 3 Sol calls (draft, revision, splice-only repair).
- Draft: valid, 705 words, 42 answer objects, and 39 references.
- Initial revision: parsed `edit` response with six operations, but local citation validation
  rejected the operation set atomically.
- Repair: retained only two authenticated operations from the initial proposal.
- Final: valid, 753 words, 43 answer objects, and 43 references.
- Generation cost: Luna `$0.007289625`; Sol `$0.476130000`; total `$0.483419625`.

The Luna qualitative review preferred the final. Both arms covered all five top-level narrative
obligations; the final added two relevant omissions without obvious discourse seams. A direct
selected-passage check of the five citation links introduced by those objects found four fully
supported, one partially supported, and zero unsupported. The partial result is a mild temporal
qualifier that should be softened in a future prompt iteration, not grounds for a fourth Sol call
or a manual edit to the sealed experiment.

## Paired coverage result

Both arms were evaluated independently against the same 75 released development nuggets with
DeepSeek V4 Flash at minimal thinking. Each arm joined exactly one topic and reported zero failed
assignments.

| Metric | Draft | Final | Delta |
|---|---:|---:|---:|
| Strict vital | 0.433962 | 0.547170 | **+0.113208** |
| Strict all | 0.413333 | 0.533333 | **+0.120000** |
| Partial-credit vital (raw) | 0.518868 | 0.632075 | +0.113207 |
| Partial-credit all (raw) | 0.500000 | 0.606667 | +0.106667 |
| Words | 705 | 753 | +48 |
| Answer objects | 42 | 43 | +1 |

The cheap judge emitted the non-canonical label `partial` for six draft nuggets. RAGDoll counts
that label as zero rather than the intended half credit, so the raw partial-credit deltas are
overstated. Treating those six labels as `partial_support` gives adjusted draft scores of
0.575472 vital and 0.540000 overall, leaving positive adjusted deltas of **+0.056603** and
**+0.066667**. Strict scores are unaffected by this alias issue. The label-level comparison had
12 improvements, 3 regressions, and 60 unchanged nuggets, but independent judge variance means
the qualitative/evidence checks remain part of the decision rather than optimizing the score
alone.

The paired evaluator made 16 cheap hosted calls and cost `$0.009871000`. Topic 897 therefore cost
`$0.493290625` including generation and coverage evaluation. Across all three live attempts in
this report, the provider-reported total was `$1.106797385`.

## Deadline verdict

**Prefer the topic-897 final and retain the bounded splice design as the best current approach.**
It produced a meaningful, coverage-positive revision within the 1,024-word cap and the three-Sol
ceiling. This is still one topic, not evidence of general reliability. The immediate production
lesson is to keep the whole-narrative draft, use cheap per-group omission audits, and permit one
atomic evidence-bound splice repair; do not add another open-ended rewrite pass.
