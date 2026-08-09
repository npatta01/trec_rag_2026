# Bounded Splice Citation Hardening Design

## Objective

Retain the approved whole-narrative draft → per-group Luna omission audit → bounded Sol splice
architecture while preventing the compound-answer and evidence-routing failures measured in PR
#53. This is a narrow hardening pass on the throwaway bounded-splice path, not a merge of PR #53
or a change to the supported competition runner.

## Decision

Borrow PR #53's citation-precision safeguards without adopting its DeepSeek priority planner:

1. Every splice `new_object` must be one self-contained sentence intended to express one atomic
   claim.
2. It may cite one or two documents, ordered strongest first. One is the default; a second is
   permitted only when it independently supports the complete object.
3. Every cited docid must come from the selected evidence linked to at least one audit card named
   by that operation, in addition to belonging to the topic-wide authenticated citation domain.

The prompt carries the semantic atomic-claim and complete-object-support rules. Local validation
enforces the mechanically decidable subset: one sentence, at most two citations, known cards,
and exact card-linked document routing. The existing whole-answer organizer and exact-hint
validators remain the final structural gate.

## Why This Change

PR #53's four-topic `priority_aware` arm combined multiple planned claims and citations into
compound answer objects. Relative to its `coverage_aware` arm, hard citation precision fell from
0.8974 to 0.6322 and weighted first-citation precision fell from 0.9291 to 0.7971. Topic 897's
bounded-splice final contains the same risky shape: one inserted two-sentence object with two
distinct claims and separate citations.

The current splice validator also checks citations only against the topic-wide authenticated
domain. Naming an audit card does not currently constrain the operation to that card's evidence,
so an unrelated same-topic document can pass structural validation.

## Interfaces

`validate_splice_payload` and `validate_repaired_splice_payload` will replace the flat
`allowed_audit_card_ids` argument with:

```python
audit_card_docids: Mapping[str, tuple[str, ...]]
```

The mapping is built by the trial orchestrator from each merged card's authenticated
`evidence_aliases` and the topic evidence alias→docid mapping. An operation's allowed citation set
is the union of the mapped docids for its named cards. Unknown cards, empty routing sets, and
citations outside that union reject the complete patch.

The splice response schema changes `new_object.citations.maxItems` from three to two. The draft
writer remains unchanged in this pass; the stricter rule applies only to new or replaced splice
objects.

## Single-Sentence Validation

A splice object must contain exactly one terminal sentence. The validator rejects text with an
internal sentence boundary followed by a new sentence and text without terminal punctuation.
This is intentionally conservative: a false rejection preserves the already-valid draft. Local
code does not claim to prove semantic atomicity; the prompt and post-hoc support judge cover that
remaining judgment.

## Prompt Contract

Both the initial splice prompt and repair prompt state:

- one self-contained sentence and one atomic claim per `new_object`;
- one strongest citation by default, no more than two;
- every citation must independently support the complete object;
- citations must be drawn only from evidence linked to the operation's named audit cards.

The repair remains unable to rewrite prose or citations. It may only retain, reorder, or repair
the wrapper around exact initially proposed objects.

## Evaluation

After provider-free implementation verification, run the existing RAGDoll support judge on the
already frozen topic-897 draft and final artifacts using the same DeepSeek V4 Flash/minimal
settings used by PR #53. Generation remains frozen; no gold, judgments, or evaluation output can
enter a generation prompt.

Report paired weighted first-citation precision, weighted all-citation precision, hard precision,
task counts, failures, and evaluator cost. This measurement diagnoses the existing output; it
does not retroactively validate the hardened contract or authorize another generation run.

## Failure Behavior

All patch validation remains atomic. Any compound object, excess citation, unknown card,
card-unlinked citation, geometry error, word-cap violation, or final organizer failure preserves
the validated draft. No fourth Sol call and no full-answer rewrite are introduced.

## Focused Tests

Use only these public seams:

- `splice_response_schema` and `validate_splice_payload` for the schema, sentence, citation-count,
  and card-linked routing contract;
- `validate_repaired_splice_payload` to prove repairs cannot bypass the same rules;
- the trial's card-routing helper and revision prompt renderer to prove exact alias→docid mapping
  and prompt clauses.

Run the existing bounded-splice and narrative-blueprint focused suites, Ruff, `py_compile`, and
`git diff --check`. Do not add production-runner tests or merge PR #53 in this task.

## Deferred Work

After the deadline, evaluate using PR #53's `coverage_aware` contract for the initial draft on a
shared cohort. Do not adopt the current `priority_aware` planner unless an atomic writer restores
citation precision in a controlled comparison.
