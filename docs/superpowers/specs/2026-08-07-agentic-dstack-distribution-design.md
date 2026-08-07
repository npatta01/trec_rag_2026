# Distributed Agentic Retrieval for the 2026 Cohort

Date: 2026-08-07  
Status: Core distribution path implemented and offline-verified; live preview pending  
Scope: retrieval only for the 119-topic 2026 test cohort

## Objective

Finish as much of the authenticated 2026 agentic-retrieval cohort as possible before the competition deadline, while preserving every successful topic immediately and retaining a locally verifiable aggregate even if cloud capacity, a worker, or the coordinator fails.

This design optimizes for durable topic completions rather than a promised wall-clock finish. It does not start dstack compute or hosted model calls. Launch remains a separate, explicit operational gate after implementation, tests, and an offer preview.

## Decisions

1. Freeze one immutable 119-topic `AgenticRunPlan` before distributing work. Every worker and the coordinator must use the same plan bytes and SHA-256 identity.
2. Make one topic one dstack task and run one topic process per GPU. A short sequential queue is a capacity-scarcity fallback, and it must publish and round-trip-verify each completed topic before starting the next.
3. Use a rolling pool rather than synchronized waves. Refill a slot as soon as a task reaches a terminal state.
4. Start conservatively and expand only from measured health: two one-topic canaries, then four active topic tasks, then six, and at most eight while API latency, throttling, GPU memory, and publication remain healthy.
5. Publish an immutable, authenticated result bundle immediately after each successful topic. Upload the completion marker last.
6. Import completed bundles into a dedicated local staging aggregate continuously. Write a durable receipt after each import batch (a “wave”), while cloud workers continue.
7. Never repeatedly rewrite organizer-facing root artifacts. Build them once, after all 119 topic seals validate against the frozen run plan.
8. Treat shared retrieval/reranker cache consolidation as a secondary lane. Topic results are submission-critical; a large cache upload must not delay their publication.
9. Retry only unresolved topics whose immutable remote prefix is empty. Never delete or overwrite a completion marker.
10. Keep all result bundles, caches, provider material, and outputs private.

## Why incremental merge wins

Merging only at the end creates one large failure window: successful remote work is durable, but the local coordinator has no continuously validated recovery point and discovers incompatibilities late. Rebuilding aggregate organizer files after every topic has the opposite problem: unnecessary churn and greater risk of exposing an incomplete run as final.

The chosen split is:

```text
worker topic completion
  -> immutable remote topic bundle + marker
  -> local verified import into run-specific staging
  -> append-only wave receipt
  -> next work continues

all 119 staged topic seals present
  -> one final aggregate export
  -> organizer-facing validation
```

This makes intermediate progress useful without pretending that a partial cohort is a valid final submission.

## Authoritative identities

The coordinator initializes the run exactly once and publishes:

- canonical agentic config bytes and digest;
- ordered 119-topic cohort and narrative hashes;
- source Git revision and submodule revisions;
- agentic run-plan JSON and its SHA-256;
- experiment ID and private artifact prefix.

A worker refuses to run or package a topic unless the plan includes that topic and the local plan digest equals the published digest. An importer refuses a bundle unless all of those identities match its own staging run.

## Topic result bundle

The fixed-retrieval cache bundle is not reused as a semantic format. Its safe archive and transactional-import techniques may be adapted, but agentic state receives its own module and schema.

A successful agentic topic bundle contains only the authenticated closure needed to resume and aggregate that topic:

- bundle schema and canonical manifest;
- `run_plan_sha256` and topic membership proof;
- the successful manifest-last topic seal;
- `retrieval_topic.json`;
- `generation_topic.json`;
- `topic_records_receipt.json`;
- member sizes and SHA-256 digests.

The current aggregate exporter needs only those sealed projection bytes; the TopicRecords receipt is authenticated metadata, not authentication of the underlying SQLite ledger. Therefore the minimal submission-critical bundle excludes the TopicRecords database and cache closure as well as failed attempts, raw provider responses, trace/debug payloads, locks, WAL/SHM files, qrels, gold nuggets, model weights, and unrelated cache entries. TopicRecords/cache portability is a separate, larger secondary bundle if later required for replay.

Archive verification must reject traversal, links, devices, duplicate/colliding paths, unexpected members, oversized streams, digest mismatches, trailing bytes, an unsealed topic, a foreign run plan, or incomplete document closure.

## Publication protocol

Use the existing private Hugging Face Bucket and a run-specific hierarchy:

```text
trec_rag_2026/experiments/<run-id>/
  plan/
    agentic-run-plan.json
    plan-complete.json
  topics/<topic-id>/
    bundle.tar.zst
    bundle-complete.json
  failures/<topic-id>/<task-name>/
    failure-receipt.json
  workers/<task-name>/
    cache-bundle.tar.zst        # optional, secondary lane
    cache-complete.json         # optional, marker last
```

The worker performs, in the foreground:

1. run one topic against the installed plan;
2. validate the local topic seal and semantic closure;
3. pack and locally verify the topic bundle;
4. upload the archive with create-only semantics;
5. upload the completion marker last;
6. list the exact remote prefix;
7. download both objects;
8. compare bytes and re-run semantic verification;
9. only then continue to the next queued topic.

A failure receipt is diagnostic only and can never be imported as a successful topic.

## Local staging and wave receipts

The local coordinator uses a dedicated run-specific staging tree, separate from the primary shared cache. Import is verify-before-lock, create-only or identical-only, transactionally staged, and idempotent after interruption.

Each collector cycle:

1. lists only exact completed topic prefixes;
2. downloads unseen archive/marker pairs;
3. verifies the remote marker and archive offline;
4. imports topics under a process lock;
5. validates the installed topic seal again;
6. writes an append-only wave receipt containing imported, already-present, rejected, failed, and still-missing topic IDs plus all relevant digests.

The collector never touches the final organizer-facing root artifacts until the cohort-complete gate passes.

## Scheduling policy

### Unit of work

- One GPU and one dstack task run one agentic topic by default.
- The rolling scheduler assigns deterministic, disjoint topic IDs from the frozen plan.
- A two-topic sequential queue is allowed only when marketplace capacity is scarce and a measured topic plus publication fits comfortably twice inside the configured task duration.
- Three- or four-topic queues are not used initially. Short tasks reduce stranded work and simplify exact retries.

### Pool growth

The runner itself permits up to three concurrent researchers. Four active topic tasks therefore create roughly twelve concurrent researcher lanes, matching the Pyserini service's conservative concurrency guidance.

- Canary: two active one-topic tasks on different offers when possible.
- Healthy baseline: four active tasks.
- Expansion: six active tasks after two clean completion/import cycles.
- Burst ceiling: eight active tasks only if there is no persistent 429 rate, transport retry growth, hosted-provider degradation, GPU-memory pressure, or publication backlog.
- Backoff: shrink one level when throttling or latency remains elevated across a collector cycle; isolate repeated failures to the exact topic.

This is a control policy, not a runtime promise. The launcher must expose the active-task ceiling so the coordinator can adjust it without changing the run plan.

### GPU task shape

Start from the proven marketplace families (`A40`, `A6000`, `L40S`) with at least 48 GB GPU memory, at least 48 GB host RAM, and 100 GB disk. Use on-demand instances, a pinned image, a pinned dstack version, a bounded hourly price, and a six-hour maximum for a one-topic canary/task. A two-topic sequential fallback requires at least a ten-hour maximum. The exact offer and price cap are chosen from `dstack apply` preview immediately before launch; no assumptions about live availability or cost are embedded here. The canary must report peak disk usage below 50 GB before retaining a 100 GB request; otherwise raise the request to 200 GB.

## Pipelining

Pipelining occurs across independently durable stages:

```text
Pool A: topic execution -> per-topic validation -> HF publication
Pool B: rolling dstack refill and exact-topic retry
Local:  HF collection -> offline verify -> staging import -> wave receipt
Final:  cohort gate -> aggregate export -> organizer validation
Later:  optional cache shard verification and promotion
```

Hugging Face uploads remain foreground operations inside each worker. We gain overlap from other workers and the local collector, not from an untracked background upload that can be killed with the task.

## Failure and recovery rules

- Worker loss before marker upload: prefix is incomplete and not importable; diagnose, then retry the topic under an empty prefix.
- Archive uploaded but marker absent: treat as incomplete; never infer success.
- Marker present but round-trip verification failed: quarantine locally and stop that task; never overwrite remotely.
- Duplicate completion: accept only when remote and local bytes/digests are identical.
- Topic failure: preserve the safe failure receipt and requeue only that topic.
- Collector interruption: recover from its journal and repeat idempotently.
- Coordinator loss: reconstruct staging from the immutable run plan and completed remote topic prefixes.
- API throttling: reduce active tasks; do not increase per-topic researcher concurrency.
- Deadline reached with fewer than 119: retain a fully authenticated partial staging aggregate and completion ledger, but do not label it a valid final organizer export.

## Launch gates

Before any live full-run call, the operator must see and approve:

- clean tracked source revision and pinned submodules;
- exact 119-topic cohort and run-plan digest;
- private HF prefix and local staging/output paths;
- expected cache reuse/misses and hosted-call classes;
- dstack offer table, maximum active tasks, task duration, and maximum price exposure;
- presence (never values) of `INDEX_URL`, `PYSERINI_API_TOKEN`, `OPENROUTER_API_KEY`, and `HF_TOKEN`;
- passing unit, workflow, archive-hostility, offline round-trip, and small-cohort integration tests.

The first live action is two one-topic canaries, not the 119-topic launch. Promotion to the rolling pool requires both remote round trips and local imports to validate.

## Non-goals

- No RAG generation is started by this workflow.
- No dynamic distributed broker or shared mutable database is introduced.
- No public artifact serving or publication occurs.
- No partial organizer-facing aggregate is presented as complete.
- No fixed-retrieval bundle contract is weakened to accommodate agentic state.

## Competition-contract resolution

The canonical track retrieval contract places no fixed maximum on internal candidate generation and explicitly allows query decomposition, fusion of multiple searches, and reranking before variable-depth final selection. The repository's 25-search ceiling belongs to its supported bounded, one-shot fixed-planning architecture. The agentic runner is a separate internal retrieval strategy with a 100-retrieval-call safety budget; it does not alter the organizer-facing format or variable-depth selection rule. The agentic config therefore remains unchanged. Final validity still depends on all organizer-facing retrieval validation rules, not on exhausting the internal call budget.

## Success criteria

- Every successful topic is independently durable, authenticated, and round-trip verified.
- Local staging can be reconstructed from the run plan plus private remote topic prefixes.
- Retries cannot change the frozen cohort or overwrite completed work.
- Collection and imports continue while remaining topics execute.
- When all 119 topics are sealed, one final export passes the existing agentic submission contract and manifest-last verification.
