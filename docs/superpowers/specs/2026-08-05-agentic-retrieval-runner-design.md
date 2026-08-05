# Agentic Competition Retrieval Runner Design

## Status

Approved by the user on 2026-08-05. The implementation will add a durable
agentic competition-retrieval entry point, make its caches reusable across
linked worktrees and repeated runs, remove invocation-level time deadlines,
and publish the same sealed generation-handoff contract consumed by the
existing competition RAG runner.

The first live validation after implementation is explicitly limited to
`rag2026-0`, using a fresh output namespace and the validated warm cache from
the earlier agentic run. It is not authorization for an all-topic run.

## Objective

Support two explicit retrieval workflows without coupling their orchestration:

```text
fixed config   -> existing fixed runner   --+
                                             +-> generation_handoff_manifest_v1
agentic config -> new agentic runner      --+              |
                                                              v
                                              existing competition RAG runner
```

The fixed pipeline remains the stable competition path. The agentic pipeline
gets its own strict configuration and CLI, but both paths meet at the existing
typed `GenerationTopic` / `GenerationHandoff` boundary. Generation therefore
does not need to know which retrieval strategy produced its evidence.

## Architecture Decision

Use separate runners and configurations with a shared artifact seam.

- `configs/rag26_competition_retrieval_v2.yaml` remains the fixed,
  non-agentic configuration. Its strict `facet_pilot_config_v2` schema is its
  mode marker; its bytes and existing checkpoint identities are not changed.
- Add `configs/rag26_competition_agentic_retrieval_v1.yaml` with a strict
  agentic-only schema and an explicit `retrieval_mode: agentic` field.
- Add `trec_rag.competition_agentic_retrieval` as the durable agentic CLI.
  Supplying a fixed config to this CLI or an agentic config to the fixed CLI
  fails before any cache mutation, hosted request, or output publication.
- Reuse the existing topic passage-search, TopicRecords, document-store,
  generation-handoff, and organizer-format records. Do not add an agentic
  branch inside the fixed runner.

A single dispatcher was rejected because it would migrate the proven fixed
config and checkpoint identities for a convenience that is not needed. Folding
agentic control flow into the fixed runner was rejected because the two
orchestrators have different state, failure, and resume semantics.

## Agentic Configuration

The checked-in agentic config is a full-run configuration selecting all 119
official narratives by default. Like the fixed runner, the CLI accepts repeated
`--topic` selectors for smoke runs. The canonical agentic configuration records:

- a safe experiment/run ID and output namespace;
- the official topic TSV;
- `climbmix-400b` and its pinned corpus epoch;
- shared retrieval, reranker, document, and local model cache identities;
- the pinned Mixedbread model, revision, chunking, and scoring settings;
- the OpenRouter coordinator/researcher model;
- topic-level execution concurrency;
- the production research-count budgets;
- two total final-synthesis attempts; and
- `create` or `resume` lifecycle mode.

The production research limits are the configuration actually planned and
validated: 20 researcher invocations, 3 concurrent researchers, 100 total
researcher retrieval calls, 20 tool calls per researcher, 8 searches, 16
snippet pages, 8 passage searches, 30 researcher model turns, 80 coordinator
model turns, and a 2-turn synthesis reserve. The no-yield and no-progress
guards remain enabled.

There is no soft or hard elapsed-time budget. `ResearchBudget` continues to
measure elapsed time for diagnostics, but time cannot refuse a researcher or
stop a topic. Count, concurrency, no-yield, no-progress, retrieval-availability,
and model-turn limits still bound work. Individual HTTP/provider timeouts remain
finite so a dead connection cannot hang forever; they are transport safety
settings, not a topic time budget.

Exhausting the researcher, retrieval-call, or model-turn budget is a normal
bounded-completion condition, not an export failure. The runner records the
exact stopping reason and unresolved need IDs, gives the coordinator its
reserved final synthesis opportunity when possible, and continues to topic
projection. Unresolved needs do not block the Retrieval artifacts or generation
handoff. The handoff contains the grounded draft-selected evidence that exists;
missing or invalid draft selection follows the grounded-nugget retry and
recovery policy defined below.

## Shared Cache Contract

Both runners resolve relative `cache/...` paths through `repo_cache_root`, so
linked worktrees use the main checkout's ignored cache tree automatically:

- `cache/retrieval/pyserini_remote` for authenticated retrieval responses;
- `cache/reranker` for content-addressed Mixedbread scores;
- `cache/documents/v1` for authenticated document receipts and text; and
- an ignored shared local model cache for the pinned Mixedbread snapshot.

Cache identity remains independent of experiment ID and output directory.
Consequently, an exact repeated query against the same index/corpus identity
can reuse its retrieval response, and an exact query/passage/model/chunker
identity can reuse its reranker score. A different adaptive query is a cache
miss and is added normally. Coordinator and researcher OpenRouter responses are
not cached; agent state always starts fresh for an incomplete topic.

Before the live smoke, import the already validated temporary agentic cache
into the shared repository cache. The import validates every entry and permits
only identical content-addressed duplicates. A path/key collision with
different bytes fails without replacing either copy. The source cache remains
untouched.

## Run Lifecycle and Resume

Every experiment owns a distinct output directory. `create` refuses any
existing run namespace. `resume` first authenticates the config bytes, selected
topic identities, source revision, submodule revisions, and every sealed
per-topic receipt.

A valid completed topic is reused without calling the agent or index. An
incomplete or unsealed topic is restarted from its official narrative with a
new temporary ledger; it may reuse shared retrieval, document, reranker, and
model caches. There is no mid-topic semantic resume and no mutation of a sealed
topic ledger. Stale temporary state is retained as private diagnostics rather
than being treated as authoritative or overwritten.

The root organizer artifacts and export manifest are published only after all
selected topics have a valid sealed projection. Repeated publication with
identical authenticated bytes is idempotent. Different bytes in an existing
sealed namespace are an integrity error; there is no agentic overwrite mode.

## Topic Data Flow

For each official topic, the runner:

1. creates a topic-scoped `TopicRecordsBuilder`;
2. performs the untouched original-narrative passage search;
3. runs the coordinator and bounded researchers against the shared passage and
   snippet adapters;
4. commits grounded researcher handoffs and accepted facets;
5. records completion and seals the topic ledger;
6. projects organizer Retrieval rows and a typed `GenerationTopic`; and
7. publishes a sealed per-topic projection receipt.

The official variable-depth Retrieval run contains every unique document that
supports a live, non-superseded accepted nugget. Supporting documents present
in the final agentic RRF list retain that list's order. Remaining supporting
documents follow by earliest `(search ordinal, passage rank, docid)` occurrence
across the immutable search sequence. Scores are derived from final ordinals
and are non-increasing. No unrelated document is added to reach a fixed cutoff.

The generation projection is intentionally narrower. For every need with a
final coordinator-selected `draft_nugget_id`, it creates one evidence group
whose ID is the need ID and whose text is the need's recorded question. Each
selected live nugget becomes a selected cluster and an advisory claim hint.
Its grounded passage citations become `EvidencePassage` rows containing the
exact stored passage text, document ID, document hash, source character and
byte offsets, and final Retrieval document rank. Group-scoped deterministic
evidence IDs allow one grounded source or nugget to support more than one need
without violating handoff ownership rules. A missing, superseded, or
ungrounded draft nugget—or one not associated with the need that selected
it—is an integrity error rather than a silent omission.

Raw original-query passages are never promoted into the generation handoff.
When the coordinator's final synthesis omits a required draft selection or
names a missing, superseded, ungrounded, or need-unassociated nugget, the
invalid synthesis is rejected and the coordinator receives one fresh synthesis
attempt against the same grounded state, with no new retrieval. Thus final
synthesis has at most two semantic attempts.

If both synthesis attempts fail while live grounded nuggets exist, a
deterministic recovery selector keeps, for each need, up to the existing
`MAX_DRAFT_NUGGETS_PER_NEED` live grounded nugget IDs in that need's recorded
order. Projection then uses those nuggets and their authenticated cited
passages, and the topic receipt records `deterministic_grounded_recovery` as the
synthesis outcome. This recovery cannot introduce a nugget, passage, or
document absent from the validated coverage state.

If a completed topic attempt has no live grounded nuggets at all, nothing is
projected and the topic fails immediately. The command exits nonzero, reports
the topic ID and zero-grounded-nugget reason, preserves its private diagnostics,
and leaves shared caches available. A later user-initiated `resume` or new run
restarts that topic from the official narrative and automatically reuses exact
cache hits. The runner does not retry the entire topic on its own, and
publishing raw search passages or fabricated evidence is forbidden.

Every handoff citation document must be a subset of both the topic's Retrieval
rows and its full-text archive. The projection validates that closure before it
can be sealed.

## Root Publication and RAG Consumption

The agentic exporter writes the standard private retrieval artifacts under its
own experiment directory:

- `r_output_trec_rag_2026.tsv`;
- `retrieval_with_text.jsonl.zip`;
- `generation_handoff_manifest.json`; and
- `retrieval_export_manifest.json` as the completion marker.

The outer manifest authenticates the selected topic order, source revisions,
per-topic seals, topic statuses, artifact sizes, and artifact SHA-256 hashes.
It is written last after the TREC rows, full-text archive, and generation
handoff validate together. Topic status preserves budget exhaustion and partial
coverage without making the handoff unusable.

The existing `trec_rag.competition_rag` loader consumes the handoff unchanged.
An agentic RAG config differs only in its experiment/output identity and
`inputs.handoff_manifest` path. Fixed and agentic RAG outputs therefore remain
separate and cannot overwrite one another.

## Failure Semantics

- Configuration, topic selection, dirty-worktree, submodule, secret-presence,
  and output-namespace checks happen before hosted work.
- Cache corruption, conflicting cache import bytes, foreign documents,
  ungrounded citations, dangling facets, source-offset mismatches, and
  cross-artifact document-closure failures are fatal integrity errors.
- A provider or retrieval failure during evidence collection leaves the topic
  unsealed and prevents root publication. `resume` restarts that topic from the
  beginning with warm caches. A final-synthesis provider failure follows the
  synthesis retry and grounded-nugget recovery policy instead.
- Research-count exhaustion and unresolved needs are not provider/retrieval
  failures; they publish the available grounded nuggets or use deterministic
  grounded-nugget recovery after the synthesis retry.
- The runner never fabricates evidence, weakens TopicRecords integrity, drops
  a citation to force validation, reads organizer qrels/gold/RAGDoll outputs,
  or silently falls back to fixed retrieval.
- Outputs, caches, ledgers, provider responses, corpus text, and diagnostics
  remain private and ignored by git.

## Verification

Implementation proceeds test-first. Automated verification must cover:

- strict agentic config parsing, wrong-mode rejection, and exact production
  budget values;
- absence of time-based admission and stopping under an advanced fake clock;
- unchanged count, concurrency, no-yield, and no-progress limits;
- successful handoff publication after researcher-budget exhaustion with both
  partial draft evidence and deterministic grounded-nugget recovery;
- rejection and one fresh retry of an invalid final synthesis;
- immediate, clearly reported failure after a zero-grounded-nugget topic, with
  no automatic topic retry and warm-cache reuse on a later manual run;
- linked-worktree shared cache resolution and conflict-safe cache import;
- exact cache hits for repeated retrieval and reranker identities;
- fresh agent state on incomplete-topic restart;
- reuse only of authenticated completed-topic receipts;
- deterministic agentic `GenerationTopic` projection, recovery behavior, and
  rejection of ungrounded or unranked evidence;
- handoff-document subset closure across the TREC run and full-text ZIP;
- manifest-last publication and refusal to overwrite conflicting output;
- unchanged fixed retrieval and existing RAG behavior; and
- a fake-provider, one-topic end-to-end run through the new CLI.

After the focused and full local suites pass, run one live `rag2026-0` smoke
with the checked-in agentic config, a fresh experiment/output ID, and the shared
warm cache. Before launching, report the exact topic, cache inventory and
expected hit/miss behavior, hosted calls, and output paths. Afterward, validate
the sealed ledger, cache statuses, TREC/ZIP/handoff document closure, artifact
hashes, manifest-last receipt, RAG loadability, git cleanliness, and pinned
submodules. Do not start an all-topic retrieval or RAG run without separate
explicit authorization.

## Non-Goals

- No unified fixed/agentic dispatcher.
- No changes to the fixed retrieval algorithm or its existing config bytes.
- No caching of coordinator or researcher LLM responses.
- No mid-topic agent-state checkpointing.
- No automatic whole-topic retry after a zero-grounded-nugget result.
- No original-query passage fallback in the generation handoff.
- No prompt/schema change that prevents researchers from proposing facets.
- No RAG generation-model, prompt, retry, or citation-policy change.
- No all-topic live run, public publishing, or submission upload.
