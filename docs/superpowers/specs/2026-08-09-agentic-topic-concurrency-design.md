# Agentic Topic Concurrency Design

## Goal

Allow independent agentic retrieval topics to execute concurrently, bounded by
the existing `execution.topic_workers` setting, while preserving the global
hosted-index rate limit, per-topic authenticated run state, and sealed export
contract.

## Context

The agentic runner currently allocates, executes, and seals topics in a plain
sequential loop. Independent topics do not share semantic evidence state, topic
records, or output roots, so topic-level concurrency is a safe scheduling seam.
Researcher concurrency inside one topic remains a separate concern: passage
searches currently use a per-topic lock around shared budget, evidence, and
search state. This change does not remove that lock.

## Chosen architecture

`execution.topic_workers` becomes a positive integer. The checked-in agentic
configuration uses `2`; local smoke configurations may use `1` or another
operator-selected value according to accelerator memory.

The parent runner continues to authenticate the config, official topic source,
repository revisions, run plan, and selected topic cohort before live work. It
allocates one authenticated attempt per selected topic, then dispatches those
requests through a bounded executor. Results are merged in official input order;
execution completion order never changes run-plan membership or export order.

Production topic requests use a spawn-based process pool. Each child constructs
its own Pyserini adapter, scorer, snippet ranker, document store, and agentic
retriever, so model objects, SQLite connections, and topic records are not
shared across processes. The existing content-addressed cache claims and the
existing shared retrieval rate limiter remain the coordination mechanisms.

The injected test seam uses a bounded thread pool when concurrency is enabled,
because test factories are commonly closures and cannot be serialized into a
spawned process. Serial injected execution remains unchanged.

Data flow:

```text
authenticated parent
  ├─ allocate topic attempt A ─┐
  ├─ allocate topic attempt B ─┼─ bounded topic workers
  └─ allocate topic attempt C ─┘      │
                                      ├─ local topic pipeline
                                      ├─ shared rate-limited retrieval cache
                                      └─ shared claimed score cache
  parent seals returned projections and publishes aggregate export only when
  every planned topic is sealed
```

The hosted Pyserini request dispatcher remains globally rate-limited. More topic
workers can overlap retrieval waits with local scoring, snippet work, and model
calls, but cannot raise the provider's allowed request-start rate.

## Failure and resume behavior

- A `TopicOperationalError` writes the same safe, private `failure.json` as the
  serial runner and leaves that topic unresolved.
- A complete child result is sealed by the parent using the existing
  `seal_topic_success` path.
- An unexpected child exception is collected after sibling workers finish, then
  the invocation fails without publishing aggregate artifacts. Successful topic
  seals remain resumable.
- `executed_topic_ids` remains in selected source order, not completion order.
- Resume and targeted resume keep their current cohort and seal validation rules.

## Non-goals

- No Runpod endpoint or remote scoring implementation.
- No change to researcher budgets, search depth, passage limits, or cache
  identities.
- No removal of the per-topic follow-up lock.
- No attempt to bypass or multiply the hosted-index rate limit.

## Testing strategy

- Validate that the canonical agentic config records two topic workers and that
  invalid non-positive/non-integer values are rejected.
- Add a two-topic injected test whose workers must overlap; it fails under the
  old sequential loop and passes only when the configured bound is honored.
- Preserve existing lifecycle tests for create, failure, resume, targeted repair,
  export ordering, and secret-safe diagnostics.
- Run the agentic config, agentic orchestration, and run-state test modules, then
  the targeted retrieval regression suite.

## Operational guidance

Start with `topic_workers: 2` on a single accelerator and inspect memory and
cache receipts. Increase only when the model/runtime and hosted-index limiter
show useful overlap. Multiple machines require an external/global limiter; each
machine must not independently assume it owns the provider quota.
