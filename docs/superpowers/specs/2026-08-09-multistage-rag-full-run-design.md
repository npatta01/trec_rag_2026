# Multi-Stage RAG Full-Run Promotion Design

## Decision

Create a separate `trec_rag.competition_rag_multistage` runner around the frozen bounded-revision
state machine. Keep `trec_rag.competition_rag` unchanged for the single-pass submission. Both runs
consume the same authenticated 119-topic `generation_handoff_manifest.json`, but use distinct
experiment IDs, output directories, work directories, identities, and final JSONL files.

Do not generalize the single-pass runner, shell-loop 119 one-topic configs, manually concatenate
rows, mix single-pass fallback rows into the multi-stage run, or change any generation prompt,
model, audit, splice, screen, retry, or repair policy.

## Run Layout

```text
facet-deepseek-b40-v3/generation_handoff_manifest.json
├── competition_rag                 → single-pass work/rows → submission A JSONL
└── competition_rag_multistage      → bounded topic states → submission B JSONL
```

The two organizer files are derived artifacts. Their authoritative resumable state is private and
strategy-specific: `work/rows/` for single pass and `work/bounded_revision/<topic>/` for
multi-stage.

## Multi-Stage Components

### Full-run runner

The new runner loads the existing strict RAG YAML schema and authenticated handoff, selects topics
in manifest order, loads repository environment files, and calls the existing
`_run_bounded_revision` coroutine. `generation.concurrency` bounds the number of topics active at
once; all stages inside a topic remain unchanged.

The runner supports the existing `experiment.mode` values as follows:

- `create`: refuse any existing output or work artifact, write one run identity, then create every
  selected topic state;
- `resume`: require the exact run identity, resume topics with existing state, and create topics
  that had not started before interruption;
- `overwrite`: reject it; this promotion adds no destructive path.

### Run identity

Write `work/multistage_generation_identity.json` before any hosted call. It binds:

- identity and trial-contract versions;
- handoff schema and manifest digest;
- the exact ordered topic IDs and context digests;
- the complete existing `_bounded_identity` projection for every topic;
- the final organizer run ID, derived as `<experiment.id>-final`.

Resume compares the complete recorded object byte-semantically after JSON parsing and refuses any
difference. Per-topic state continues to enforce its own identity and registered-file hash chain.

### Topic execution and failure handling

Run selected topics with bounded concurrency. Each topic chooses `resume` when its bounded state
root exists and `create` otherwise. A failure is recorded by the existing state machine; sibling
topics may finish, but the full runner returns a failure and does not publish a new organizer
JSONL until every selected topic has a validated final.

The existing deterministic fallback remains authoritative: invalid screen output preserves the
validated draft, and exhausted repair does not authorize a single-pass row or any new model call.

### Consolidation

After every topic completes, reload and revalidate each registered final artifact. Require exactly
one JSON object per selected topic, the exact narrative, the final run ID, the configured team and
run description, authenticated citation domain, exact-hint citation constraints, and the 1,024
word limit. Preserve handoff order and atomically replace only the dedicated multi-stage
`rag_output_trec_rag_2026.jsonl`.

Publication is all-or-nothing for the selected topic set. A full official config omits
`experiment.topic_ids`, so completion means exactly all 119 handoff topics.

## Configurations

Create ignored local configs with fresh namespaces:

- `rag26-rag-singlepass-sol-final-v1` for the existing single-pass runner;
- `rag26-rag-multistage-sol-final-v1` for the new runner;
- `rag26-rag-multistage-smoke-0-1-2` for an integration-only three-topic smoke.

All point to `outputs/facet-deepseek-b40-v3/generation_handoff_manifest.json`. The smoke config is
never submitted and does not change the full-run identity.

## Dry Run and Call Budget

`--dry-run` authenticates the config and handoff but writes no state and makes no provider call. It
prints selected topic count, exact group/audit count, routine and maximum Sol reservations, Luna
call range, concurrency, handoff path, output path, and work path.

For the current official handoff, the expected full-run budget is 119 topics, 691 group audits,
238 routine Sol reservations, at most 357 Sol reservations with repair, and 810–929 Luna calls
depending on whether each topic proposes nonempty splice operations.

## Verification

Use focused tests with a fake external topic runner while exercising the real run identity,
create/resume selection, final-row loading, validation, ordering, and atomic publication code.
Required cases:

1. create publishes one ordered row per selected topic and writes the run identity;
2. resume uses existing topic state, creates an unstarted topic, and refuses an identity change;
3. a missing or invalid final prevents publication;
4. dry run makes no provider call or filesystem mutation.

Run focused multi-stage, bounded-revision, operation-screen, generation-handoff, and competition
RAG tests, followed by the full repository suite. Hosted smoke and full runs require a separate
explicit execution step after reporting topic count, call budget, expected cache reuse, and output
directories.

## Privacy and Scope

Keep configs under ignored `configs/local/`; keep state, prompts, selected evidence, provider
responses, answers, and manifests under ignored `outputs/`. Commit only reusable runner code,
focused tests, and documentation. Do not open retrieval outputs beyond the authenticated handoff,
and never read qrels, gold nuggets, RAGDoll scores, the organizer TREC run, or the full-text ZIP
during generation.
