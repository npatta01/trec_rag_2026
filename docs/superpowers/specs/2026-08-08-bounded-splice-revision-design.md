# Bounded Splice Revision Experiment Design

## Objective

Test whether one deterministic, bounded edit pass can recover important evidence-backed
omissions without the widespread sentence churn, lost caveats, repetition, and listification
observed in the complete-answer revision trial.

This remains a throwaway development experiment. It does not change the supported competition
runner or justify an all-topic run.

## Decision

Replace the experimental revision stage's complete-answer response with one strict `splice`
response. Every edit refers to the immutable first draft. Local code validates the complete patch
atomically, applies it deterministically, and preserves every untouched draft answer object exactly.

The first fresh topic is `707`. It has strict full retrieval-nugget coverage (`1.0`) in the frozen
extraction evaluator and was not used in the earlier blueprint or revision experiments. Run and
inspect it before preparing either of the remaining two fresh topics.

## User Constraints

- Test one fresh topic first; do not run all topics together.
- Use the complete official narrative as the answer contract. Generated groups are retrieval
  structure, not equal-length answer sections.
- Prioritize important narrative needs under the organizer's 1,024-word ceiling; an answer may be
  intentionally incomplete.
- Do not optimize only for a noisy automatic score. Inspect the paired answers before opening
  post-hoc evaluator results.
- Use Luna for the planner and group audits. Use no more than two routine Sol calls: draft and
  revision. A third Sol call is reserved only for deterministic patch-validation repair.
- Keep gold, qrels, nuggets, evaluator judgments, and scores out of every generation-time input.
- Keep tests focused on the experimental contract and deterministic applier. Do not harden the
  production runner during this experiment.

## Official Evaluation Lens

The submitted answer is evaluated through anonymized pairwise battles, individualized
AutoNuggetizer-style coverage, weighted citation precision, and weighted citation recall. The
experiment therefore treats coherent prioritization, qualifiers, grounding, and citation placement
as co-equal with nugget coverage. A one-run development nugget delta cannot decide the result.

## Revision Response

The Sol revision returns exactly:

```json
{
  "decision": "keep_draft",
  "operations": []
}
```

or:

```json
{
  "decision": "edit",
  "operations": [
    {
      "start_index": 12,
      "delete_count": 2,
      "new_object": {
        "text": "One self-contained replacement sentence.",
        "citations": ["authenticated-climbmix-docid"]
      },
      "audit_card_ids": ["a003"]
    }
  ]
}
```

All fields are required and unexpected fields are rejected. `new_object.citations` contains one to
three unique raw ClimbMix docids from the authenticated handoff citation domain.

## Splice Semantics

Every index refers to the original draft's zero-based `answer` array:

- `delete_count: 0`: insert `new_object` before `start_index`; `start_index == len(answer)` appends.
- `delete_count: 1`: replace one draft object.
- `delete_count: 2` or `3`: replace one short contiguous range with one synthesized object.
- Pure deletion is not supported in this experiment.

Operations are validated against the original draft, must have unique start indexes, and cannot
overlap. An insertion inside or at the start of a replaced range conflicts; an insertion at the
range's exclusive end does not. Valid operations are applied from the highest start index downward.

## Hard Budgets

- At most 6 operations.
- At most 4 insertions (`delete_count == 0`).
- At most 8 original draft objects touched across replacement and merge ranges.
- At most 1,024 whitespace-separated answer words after assembly.
- An empty operation list is valid only with `decision: keep_draft`.
- `decision: edit` requires at least one operation.
- Every operation names at least one unique known merged-audit-card ID.

These are experimental guardrails, not claimed optima. No minimum word count or operation count is
imposed.

## Deterministic Application

The draft remains the valid fallback. The applier deep-copies it, leaves every untargeted answer
object byte-for-byte equivalent, and preserves the relative order of untouched objects. Existing
draft references retain their positions. Authenticated docids newly cited by splice objects are
appended to `references` in first-use order, and new citations are converted to their numeric
reference positions. The final run ID is rebound only after assembly.

The normal organizer record validator and exact-hint citation validator run on the complete
assembled candidate. Local code does not repair prose, drop low-priority edits to fit the word cap,
or silently accept a subset of an invalid patch.

## Prompt Contract

The revision sees the same authenticated narrative, plan, advisory hints, selected passages,
citation domain, indexed validated draft, and deterministically ID-labeled merged audit cards used
by the prior trial. It is told:

- the complete narrative and central explanatory arc outrank exhaustive card inclusion;
- audit cards are candidates, not requirements;
- prefer important evidence-backed omissions;
- prefer replacement or short merge over insertion when prose is generic or repetitive;
- retain causal qualifications, uncertainty, balance, and planner `must` obligations;
- do not create a citation-by-citation inventory;
- return `keep_draft` when no bounded edit would improve the complete answer.

The common writer evidence context is separated from its full-answer output instructions so the
splice stage receives only one response contract.

## Validation Repair and Fallback

Patch validation is atomic. An out-of-range index, overlap, budget violation, unknown audit card,
duplicate or unauthenticated citation, malformed answer object, or invalid assembled submission
rejects the complete patch.

One optional Sol repair call is allowed only when a parsed splice payload fails deterministic local
validation. The repair may correct the wrapper, indexes, range lengths, audit-card references, or
select a subset of the originally proposed operations. Every repaired `new_object`, including its
text and citations, must exactly match one from the original proposal. The repair cannot introduce
new answer prose or react to post-hoc quality scores. If it fails, the system seals the draft as the
final candidate.

A provider schema failure without a parsed payload falls back directly to the draft because there
is no authenticated operation set to repair.

## Persistence and Identity

Increment the bounded trial contract version so old full-rewrite state cannot resume under splice
semantics. Bind the splice prompt and schema hashes into the existing durable state identity and
receipts. Preserve the existing reservation-before-request ledger and the hard ceiling of three Sol
semantic reservations per topic.

## Focused Verification

Before a live call, add only targeted tests that prove:

- the strict response schema and keep/edit relationship;
- insertion, replacement, and adjacent merge behavior;
- exact preservation of untouched answer objects and stable existing reference positions;
- deterministic appending and remapping of new authenticated docids;
- atomic rejection of bad bounds, overlap, budget excess, unknown cards, and invalid citations;
- repaired output cannot introduce new answer objects;
- the trial dry run still reports one planner, one audit per group, two routine Sol reservations,
  and one repair-only reservation.

Run the focused unit tests and existing blueprint/trial tests. Do not add a broad production test
suite for this throwaway mechanism.

## First Live Topic

Create a new ignored local config and output namespace for topic `707`. Before calling models,
report its single-topic scope, expected planner/audit/Sol call counts, cache/input reuse, and output
directory. Run only topic `707` sequentially through draft, audit, splice, validation, and sealing.

Compare the draft and final with identities hidden and order randomized. Judge complete-narrative
responsiveness, prioritization, explanatory flow, qualifications, redundancy, seam coherence, and
grounding before running nugget or citation diagnostics. Record exact hosted calls and cost.

Only after this inspection decide whether the same frozen contract should continue to two more
fresh topics.

## Success Interpretation

For topic `707`, the mechanism is worth continuing when the patch invariants hold and the final is
qualitatively preferred, or tied while recovering meaningful high-priority content, without a clear
grounding, caveat, or central-arc regression. Automatic metrics provide diagnostics; a stable,
repeated regression that corresponds to a real lost point is stronger evidence than a single judge
delta.

If the patch preserves prose but creates incoherent seams, the next candidate is a constrained full
rewrite with mechanically protected objects. If a weak draft cannot be repaired within the splice
budget, test a severity-gated rewrite escape hatch in a later experiment rather than silently
loosening this one.

## Privacy

Narratives, passages, prompts, provider responses, answers, audit cards, and per-nugget judgments
remain in ignored private output directories. Commit only reusable throwaway code, focused tests,
and aggregate privacy-reviewed findings.
