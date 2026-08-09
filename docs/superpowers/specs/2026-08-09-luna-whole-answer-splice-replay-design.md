# Luna Whole-Answer Splice Replay Design

## Objective

Test whether one inexpensive whole-answer Luna pass can preserve the meaningful omission recovery
of the hardened bounded-splice experiment without per-group audits or a second Sol call.

This is a throwaway replay on the already validated drafts for development topics `233` and `499`.
It must not change the production competition runner or regenerate the Sol drafts.

## Decision Being Tested

The target flow is:

```text
authenticated handoff + authenticated planner state + validated Sol draft
  -> one Luna whole-answer splice proposal
  -> deterministic splice validation and application
  -> normalized final candidate or unchanged validated-draft fallback
```

Topic `233` is the negative probe because the full flow proposed one weak replacement. Topic `499`
is the positive probe because the full flow produced five meaningful edits. Both source runs remain
immutable comparison arms.

## Inputs and Authentication

The replay takes:

- one local RAG experiment config with a unique run ID and output directory;
- exactly one topic selected from the authenticated generation handoff;
- the private source root of a completed hardened bounded-revision topic.

Before any hosted call, local code must verify that the source state and registered files are intact,
that the source handoff digest and topic-context digest match the replay's handoff and selected topic,
and that the source contains a completed authenticated planner state and validated draft. It reloads
the blueprint through the existing blueprint-state validator and revalidates the draft under the
replay run identity.

## One-Call Prompt and Output

Luna receives the untouched narrative, authenticated blueprint, every advisory claim hint, every
selected passage, the full citation domain, and the immutable indexed draft. It does not receive
gold nuggets, qrels, evaluator outputs, the old final answer, or old revision operations.

The prompt asks Luna to return `keep_draft` or at most three atomic splice operations. It must:

- improve the full narrative rather than add merely interesting detail;
- prefer insertions when supported word headroom exists;
- use a replacement only when the new sentence is clearly more useful than everything removed;
- preserve causal qualifications, tradeoffs, and nonredundant draft details;
- emit one self-contained sentence and one strongest citation by default, with at most two;
- identify its supporting selected-evidence aliases directly;
- return `keep_draft` when no operation clears the bar.

The replay uses a strict provider schema derived from the existing splice schema. For this isolated
mode, each operation's existing `audit_card_ids` field carries selected-evidence aliases such as
`e001`; the local evidence-alias map binds each alias to exactly one authenticated document ID. This
keeps the existing deterministic validator and applier unchanged while making the experiment's
direct-evidence provenance explicit in the prompt and receipt.

## Validation and Failure Behavior

Local code validates the entire operation set atomically before applying anything:

- at most three operations;
- existing splice geometry and word-budget limits;
- one terminal sentence per new object;
- one or two unique citations;
- every citation inside the authenticated topic domain;
- every citation linked to an evidence alias named by that operation;
- normalized, compact references and full organizer-record validation after assembly.

There is no model repair call. A missing, malformed, unsupported, over-budget, or otherwise invalid
response produces the validated draft under the replay final identity. Transport failure remains
resumable without permitting a second semantic Luna completion for the same replay stage.

## Artifacts and Privacy

Each replay writes private ignored artifacts under its unique config output directory: source
identity, state, prompt/schema-bound receipt, draft/final submissions, generation identities, and a
manifest-last aggregate. Raw prompts, passages, answers, and provider responses remain private.

Only aggregate findings and reusable throwaway code may be committed. Post-hoc nugget and citation
evaluation begins only after both replay final candidates are sealed.

## Evaluation and Decision Rule

Run topics sequentially: `233` first, inspect it, then `499`.

For each topic compare three arms: unchanged draft, current full-flow final, and Luna-replay final.
Use structural validity, words, operation count/type, semantic citation support, nugget coverage,
cost, and a gold-blind qualitative review of completeness, grounding, coherence, and meaningfulness.

Prefer the simplified replay if it avoids or rejects topic `233`'s weak replacement while retaining
meaningful supported improvement on topic `499`, with no material loss of citation support or
coherence. If Luna cannot do both, prefer the simpler planner-plus-one-Sol-draft flow rather than
restore the expensive full revision topology from this two-topic result alone.

## Verification Scope

Keep tests focused on the new experiment seam:

- prompt provenance and direct-evidence alias instructions;
- the three-operation cap and evidence-linked citation validation;
- authenticated source loading and draft fallback;
- one zero-call dry run and the two sequential live replays.

Do not build a broad production test suite for this throwaway mode.
