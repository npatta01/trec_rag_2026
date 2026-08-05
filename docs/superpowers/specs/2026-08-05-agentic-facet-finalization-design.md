# Agentic Facet Finalization Design

## Status

Approved for implementation on 2026-08-05. This design repairs the live
agentic retrieval closeout failure observed at merge commit `7e9d3b9` without
changing researcher or coordinator model interfaces.

## Decision

Keep the existing ownership model:

```text
researcher proposes grounded evidence and useful facet labels
  -> coordinator accepts or rejects semantic state changes
  -> deterministic finalization persists accepted identities before evidence
```

Researchers do not receive `update_retrieval_state`, so they cannot admit a
facet. The coordinator remains the sole semantic authority. Finalization is a
mechanical projection of already validated coordinator state and never calls a
hosted model.

This preserves the repository's topic-first design: research may reveal a
missing facet and the coordinator may add it. Removing facet proposals from
researcher bundles would require prompt, schema, task-assignment, merge, and
test changes while losing same-round adaptive coverage. It is not the simplest
complete repair.

## Failure Being Repaired

The in-memory coverage state accepts facets before nuggets and validates every
facet ID and citation. Durable facet admission currently happens only when an
ID motivates a search or when a trusted researcher-creator mapping exists.
Production coordinator merges occur outside researcher task context, so a
valid evidence-referenced facet can be absent from the durable ledger at
closeout. `TopicRecordsBuilder` then correctly rejects the dangling evidence
reference with `researcher evidence facet is not admitted`.

The durable integrity check remains unchanged.

## Finalization Contract

Before writing any researcher handoff, finalization derives the ordered,
deduplicated set of subnarrative IDs referenced by every live nugget:

- use `nugget.facet_ids` when present;
- otherwise use `nugget.need_ids` for direct-need evidence.

It admits all those identities through the existing `_ensure_ledger_facets`
path. That path obtains canonical text and origin from the validated coverage
report:

- narrative-origin facets become durable `initial` facets;
- snippet-origin facets become durable `research_discovered` facets;
- direct needs use their recorded question.

Researcher creator provenance remains separate from facet existence. A
researcher facet-update row is written only when trusted task context actually
identifies the creator. Finalization must not infer authorship merely because a
researcher saw the same passage.

## Error Handling and Atomicity

- Unknown facets and ungrounded citations continue to be rejected while
  applying coordinator state deltas.
- Unknown passages, identity conflicts, and wrong-run handoffs continue to be
  rejected by `TopicRecordsBuilder`.
- The topic builder remains unpublished until finalization and publication
  succeed; a failed attempt cannot expose a sealed partial topic.
- This patch fixes referential ordering. A future deeper module may combine all
  closeout rows in one SQLite transaction, but that refactor is outside this
  competition-unblock scope.

## Verification

Add a regression through the real SQLite-backed `TopicRecordsBuilder`, not the
permissive recording fake. It must reproduce the production sequence:

1. coordinator records a need and a facet;
2. researcher searches while motivated by the need, so the facet is not
   admitted opportunistically by the query;
3. coordinator merges a grounded nugget referencing that facet outside
   researcher task context;
4. retrieval finalizes successfully;
5. the durable topic snapshot contains the facet, researcher evidence, and
   completion state.

The regression must fail with the original integrity error before production
code changes, then pass after the minimal finalization change. Existing focused
DeepAgent, topic-records, submission, and retrieval-path tests must remain
green.

## Non-Goals

- No researcher or coordinator prompt changes.
- No `EvidenceBundle`, `CandidateNugget`, or `ResearchTaskEnvelope` schema
  changes.
- No SQLite schema or cache identity changes.
- No weakening of durable referential-integrity checks.
- No inferred researcher authorship for coordinator-admitted facets.
