# Topic Records and Retrieval Cache v2 Design

## Status and decision

Approved with required changes by the Sol xhigh architecture advisor on
2026-08-03, then checked by a read-only Claude Opus architecture pass and a
Luna consistency audit. Their required identity, publication, migration, and
topic-worker clarifications are incorporated below.

Implementation is deliberately split into three sequential slices:

1. replace repeated canonical candidate JSONL with a content-addressed
   `DocumentStore` and one sealed `records.sqlite3` per topic;
2. after a representative cached 2025 topic passes, complete blocking Task
   5.5 validation and projection handoff;
3. replace the ordinary organizer request cache with immutable normalized
   request entries that reference the same document store.

A broad topic run is outside this design gate until all three slices pass their
focused and end-to-end validation.

## Architecture at a glance

```text
parent scheduler (local, configurable N)
  |
  +-- TopicJob 14 --> child process --> 14/records.sqlite3
  |                       |             14/canonical/retrieval-projection.json
  |                       +-- process-local validation session (never serialized)
  |
  +-- TopicJob 31 --> child process --> 31/records.sqlite3
  |                       |             31/canonical/retrieval-projection.json
  |                       +-- process-local validation session (never serialized)
  |
  +-- ... one disjoint database/output root per topic
  |
  +<-- sealed TopicProjectionReceipt values only
  |
  +-- official source-order concatenation -----> TREC TSV + full-text ZIP
                                                  + global manifest last

shared immutable roots: cache/documents/v1 + cache/retrieval/.../v2
shared mutable reranker state: partitioned parts or lock-protected before N > 1
```

## Problem

The current canonical evidence pipeline repeats source text in every candidate:
candidate text, evidence sentence text, matched paragraph text, neighboring
paragraph text, and passage source text are all serialized even though they are
validated slices of one source document. Twenty-one observed 2025 candidate
files occupy 216.66 GiB; one topic's candidate file is 12.65 GB while its
next-largest artifact is 83.6 MB.

The ordinary organizer request cache separately stores one complete response
per query under a truncated request key and publishes response and metadata as
separate files. A stronger full-key, manifest-last cache already exists in the
deterministic sparse ledger and supplies the publication semantics for slice 3.

## Non-negotiable identities

The implementation preserves three different byte identities:

1. **Transport bytes** are the exact organizer HTTP response bytes.
2. **Evidence bytes** are the exact UTF-8 encoding of the canonical text used
   for source hashes and character/byte offsets. `DocumentStore` never changes
   whitespace or Unicode.
3. **Scoring text** is a derived whitespace-normalized representation with its
   own hash and version.

No digest may be reused across these roles. Transport request identity and local
derivation identity are separate: the retrieval text extractor and normalizer
version is part of the derivation identity and topic-records stage identity,
never the transport request key. A local normalizer bump must rederive from raw
bytes with zero network calls.

## Module seams

### `DocumentStore`

`trec_rag.document_store.DocumentStore` is a deep module with a digest-only
interface:

```python
DocumentStore(root: Path)
store.admit_text(text: str, *, expected_sha256: str | None = None) -> DocumentReceipt
store.read_text(content_sha256: str) -> str
store.verify(content_sha256: str) -> DocumentReceipt
```

Callers cannot resolve a body by `docid` and cannot construct object paths.
Exact UTF-8 bodies are stored create-only under a versioned, two-level SHA-256
layout. Admission writes and fsyncs a unique sibling temporary file, then
publishes without replacement. A concurrent identical object succeeds after
verification; different bytes under the same digest are an integrity error.

Production uses the shared ignored root `cache/documents/v1`. Topic artifacts
contain only hashes, never absolute store paths.

### `TopicRecords`

`trec_rag.topic_records` owns the durable topic database and exposes:

```python
TopicRecordsBuilder(destination, topic_id, document_store)
builder.bind_document(docid, exact_text, expected_sha256)
builder.add_candidate(candidate)
builder.publish(identity) -> PublishedTopicRecords(receipt, validation_session)

TopicRecords.open(database, manifest, expected_topic_id, document_store,
                  validation_session=None) -> TopicRecords
records.validation_session -> ValidatedTopicRecords  # non-serializable
records.selection_pool(context, limit) -> PreclusterPool
records.load_candidates(required_keys) -> Mapping[CandidateKey, ExtractiveCandidate]
records.validate_all_sources() -> None

build_topic_projection(config, topic, records) -> TopicProjectionReceipt
export_retrieval_run(config, topics, projection_receipts, code_commit)
  -> RetrievalExportReceipt
```

The interface hides schema creation, relational writes, deterministic queries,
semantic hashing, source reconstruction, integrity checks, and publication.
Builder writes are batched by document. Selection and per-topic projection use
the same open `TopicRecords` handle and its current-process validation session.
`build_topic_projection` writes a private, sealed per-topic projection plus a
receipt containing topic identity, source seals, byte hash, and row counts.
The files are `canonical/retrieval-projection.json` and
`canonical/retrieval-projection-manifest.json`, with the manifest published
last and included in `canonical/complete.json`.
The global exporter accepts only those receipts, verifies their bytes, and
concatenates topics in the official narrative source-file order supplied by the
validated config; it never lexically sorts IDs and never reopens topic records.
On resume, the topic worker performs one untrusted deep open to mint a new
session before rebuilding or validating its projection. A validation session
never crosses a process boundary and is never serialized.

### `RetrievalCache` (slice 3)

`trec_rag.retrieval_cache.RetrievalCache` owns immutable organizer request
entries:

```python
cache.lookup(transport_identity, derivation_identity, query_text, *, offline=False)
  -> CachedRetrieval | None
cache.commit(transport_identity, derivation_identity, query_text, raw_response)
  -> CachedRetrieval
```

It admits document bodies through `DocumentStore`, stores ordered hit references
without text, retains a losslessly compressed exact raw response, and validates
that reparsing raw bytes reproduces the references.

## Topic database schema v2

Each database contains exactly one topic. All relational tables are strict and
use foreign keys.

```text
topic_identity(
  topic_id TEXT PRIMARY KEY
)

document_binding(
  document_pk INTEGER PRIMARY KEY,
  topic_id TEXT NOT NULL,
  docid TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  character_count INTEGER NOT NULL,
  byte_count INTEGER NOT NULL,
  UNIQUE(topic_id, docid),
  UNIQUE(topic_id, docid, content_sha256),
  FOREIGN KEY(topic_id) REFERENCES topic_identity(topic_id)
)

subnarrative_identity(
  topic_id TEXT NOT NULL,
  subnarrative_id TEXT NOT NULL,
  subnarrative_sha256 TEXT NOT NULL,
  PRIMARY KEY(topic_id, subnarrative_id),
  FOREIGN KEY(topic_id) REFERENCES topic_identity(topic_id)
)

candidate(
  candidate_pk INTEGER PRIMARY KEY,
  topic_id TEXT NOT NULL,
  candidate_nugget_id TEXT NOT NULL,
  document_pk INTEGER NOT NULL,
  subnarrative_id TEXT NOT NULL,
  candidate_kind TEXT NOT NULL,
  nugget_type TEXT NOT NULL,
  start_char INTEGER NOT NULL,
  end_char INTEGER NOT NULL,
  text_sha256 TEXT NOT NULL,
  sentence_score REAL NOT NULL,
  document_subnarrative_rank INTEGER NOT NULL,
  scoring_text_sha256 TEXT NOT NULL,
  subnarrative_sha256 TEXT NOT NULL,
  sentence_splitter_version TEXT NOT NULL,
  UNIQUE(topic_id, candidate_nugget_id),
  UNIQUE(candidate_pk, document_pk),
  FOREIGN KEY(document_pk)
    REFERENCES document_binding(document_pk),
  FOREIGN KEY(topic_id, subnarrative_id)
    REFERENCES subnarrative_identity(topic_id, subnarrative_id)
)

candidate_span(
  candidate_pk INTEGER NOT NULL,
  role TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  start_char INTEGER NOT NULL,
  end_char INTEGER NOT NULL,
  start_byte INTEGER NOT NULL,
  end_byte INTEGER NOT NULL,
  text_sha256 TEXT NOT NULL,
  cross_encoder_score REAL,
  PRIMARY KEY(candidate_pk, role, ordinal),
  FOREIGN KEY(candidate_pk)
    REFERENCES candidate(candidate_pk)
)

candidate_passage_link(
  candidate_pk INTEGER NOT NULL,
  document_pk INTEGER NOT NULL,
  ordinal INTEGER NOT NULL,
  passage_pk INTEGER NOT NULL,
  PRIMARY KEY(candidate_pk, ordinal),
  UNIQUE(candidate_pk, passage_pk),
  FOREIGN KEY(candidate_pk, document_pk)
    REFERENCES candidate(candidate_pk, document_pk),
  FOREIGN KEY(passage_pk, document_pk)
    REFERENCES passage(passage_pk, document_pk)
)

passage(
  passage_pk INTEGER PRIMARY KEY,
  document_pk INTEGER NOT NULL,
  passage_id TEXT NOT NULL,
  lane_id TEXT NOT NULL,
  query_id TEXT NOT NULL,
  scoring_start_char INTEGER NOT NULL,
  scoring_end_char INTEGER NOT NULL,
  source_start_char INTEGER NOT NULL,
  source_end_char INTEGER NOT NULL,
  source_start_byte INTEGER NOT NULL,
  source_end_byte INTEGER NOT NULL,
  source_text_sha256 TEXT NOT NULL,
  scoring_text_sha256 TEXT NOT NULL,
  chunk_text_sha256 TEXT NOT NULL,
  normalization_version TEXT NOT NULL,
  cross_encoder_score REAL NOT NULL,
  cross_encoder_rank INTEGER NOT NULL,
  UNIQUE(document_pk, passage_id),
  UNIQUE(passage_pk, document_pk),
  FOREIGN KEY(document_pk)
    REFERENCES document_binding(document_pk)
)

stage_seal(
  stage TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL,
  semantic_sha256 TEXT NOT NULL,
  identity_json TEXT NOT NULL,
  row_counts_json TEXT NOT NULL
)
```

Candidate, sentence, paragraph, context, and passage text are reconstructed from
the bound document body and stored offsets. Complete passage provenance,
including score and rank, is stored once in `passage`; candidate links contain
only compact integer ownership/ordinal references. Exact
candidate grouping uses `(candidate_kind, text_sha256)` and verifies
reconstructed text before use. This removes large repeated text while
preserving the existing semantic selection order.

## Topic-aware correctness

The only document binding lookup is `(topic_id, docid) -> content_sha256`.
`document_binding` has `PRIMARY KEY(topic_id, docid)`, so conflicting content
inside one topic fails immediately. Different topic databases may bind the same
`docid` to different content hashes. Identical bodies deduplicate safely in the
global content-addressed store.

No global `docid -> body` interface or map is permitted. Candidates and
passages reference one `document_pk`; each candidate-passage link repeats that
compact key and has composite foreign keys to both parents, making a
cross-document link structurally impossible. Surrogate keys are physical only:
semantic projection joins through stable topic/docid/content, candidate nugget,
ordinal, and passage identities and never hashes arbitrary surrogate values.
Span normalization is deferred: v2 keeps the existing validated character/byte
offsets and does not add a second span model. The builder writes in batches
grouped by document, while semantic projection sorts by explicit natural keys
rather than insertion order.

Same-topic docid/body disagreement is a conservative failure. The system must
not silently choose first-writer or last-writer text; the precise policy remains
an advisor decision/test gate before retriever migration if the source cannot
be proven equivalent.

## Semantic seal and SQLite lifecycle

The v2 semantic seal hashes explicit **natural-key projections**, never raw
SQLite rows or surrogate primary keys. Each projection is framed by table name,
column name, and value, then sorted by the listed natural key:

| projection | natural order and hashed values |
|---|---|
| topic | `topic_id` |
| document | `(topic_id, docid)` plus content hash and character/byte counts |
| subnarrative | `(topic_id, subnarrative_id)` plus subnarrative hash |
| candidate | `(topic_id, candidate_nugget_id)` plus the joined document natural identity and all candidate semantic fields |
| span | `(topic_id, candidate_nugget_id, role, ordinal)` plus all offset/hash/score fields |
| passage | `(topic_id, docid, content_sha256, passage_id)` plus the complete passage provenance tuple |
| candidate-passage link | `(topic_id, candidate_nugget_id, ordinal)` plus the joined passage natural identity |

`document_pk`, `candidate_pk`, and `passage_pk` are excluded from both ordering
and bytes. The implementation must issue explicit joined `SELECT` projections;
`SELECT *` over a v2 physical table is forbidden in semantic hashing. Integers
and text have length-delimited encodings; nullable values have explicit tags;
floating-point values use `float.hex()` rather than SQLite file representation
or locale-sensitive decimal formatting. Equivalent insertion order or
surrogate allocation therefore produces the same semantic seal.

Publication order for slice 1 is:

1. admit and verify every document body;
2. build a unique sibling temporary SQLite database;
3. insert candidate records in transactions and insert the scoped stage seal;
4. run `PRAGMA foreign_key_check`, `PRAGMA integrity_check`, source/hash checks,
   and character/byte-offset reconstruction;
5. force a rollback-journal final state, close the database, and reject any
   remaining `-wal`/`-shm` files;
6. atomically publish `records.sqlite3` without replacing a conflicting file;
7. write `canonical/records-manifest.json` atomically;
8. write the outer `canonical/complete.json` checkpoint last.

The records manifest stores topic ID, schema versions, stage identity, semantic
seal, database byte SHA-256/size, row counts, sorted document-hash closure, and
sorted request-key closure. The stage-1 request-key closure may be empty because
retrieval is already sealed upstream; slice 3 populates it. A copied topic
database plus its declared document closure must validate under a different
document-store root.

Equivalent builds must have equal semantic seals and organizer-facing output.
SQLite file-byte equality across versions is not required, although each
published file is byte-hashed for later tamper detection.

Validation is a current-process capability, not a trusted disk marker. A
successful validator returns a non-serializable `ValidatedTopicRecords`
validation session bound to the exact database bytes, exact manifest bytes,
validator version, and complete CAS closure it checked. `TopicRecords.open`
without a session performs full validation; with a matching current-process
session it stream-hashes the exact database/manifest bytes and verifies the CAS
closure before a fast rebind. Downstream selection and per-topic projection
must receive that capability or validate afresh; the global exporter receives
only a sealed projection receipt. A manifest
`source_validation` block may be diagnostic only; it must never authorize
skipping deep validation after restart or rebinding.
The first validator is direct and document-indexed: validate each document's
derived state and all references against the exact body, then validate the
relational projection.

## Organizer request cache v2

The transport request identity is canonical JSON hashed with full SHA-256 and
includes only remote-result inputs:

- exact query SHA-256;
- index ID and endpoint identity;
- operator-selected `corpus_epoch` (or organizer index snapshot ID);
- requested hit depth;
- every configured remote retrieval parameter that can change semantic results.

Topic, variant, and retriever labels are manifest provenance, not key inputs.
The local `DerivationIdentity` contains the versioned field-selection rule,
parser, extractor, and text-normalizer versions and is addressed beneath the
transport entry. A derivation bump reads retained raw bytes and creates a new
derivation with zero HTTP calls. Missing docids, ranks, scores, or non-finite
scores are integrity errors; coercing defaults and ambiguous field fallbacks are
forbidden. The evidence selector admits the exact UTF-8 bytes of one explicitly
versioned organizer field: it does not collapse whitespace, normalize Unicode,
or concatenate arbitrary dictionary values. Scoring normalization remains a
separate derived identity and never changes stored evidence offsets.

Layout:

```text
cache/retrieval/pyserini_remote/v2/<prefix>/<request-sha256>/
  raw.body.gz
  transport-manifest.json
  derived/<derivation-sha256>/
    hits.json
    derivation-manifest.json
```

`raw.body.gz` losslessly retains exact response bytes; its recorded hash and
length describe the uncompressed transport bytes. Gzip uses `mtime=0` and a
fixed compression level. Each derivation stores ordered `rank`, `score`,
`docid`, and `content_sha256` values. Cache validation decompresses and
reparses the raw authority, regenerates normalized references, and verifies
every document object.

The transport manifest has an exact field set binding its schema version,
full request key, canonical transport-identity JSON, exact query hash, raw
uncompressed SHA-256/length, gzip SHA-256/length, and raw filename. A derivation
manifest independently binds its schema version, full derivation key,
canonical derivation-identity JSON, parent request key, parent raw SHA-256,
hits filename/SHA-256/length/count, ordered semantic-hit digest, and sorted
document closure. Unknown or missing fields fail closed. This prevents a valid
derivation from being paired with different raw authority.

A transport writer builds under a unique sibling directory, fsyncs the exact
raw body, publishes `raw.body.gz` create-only with `os.link`, verifies a race
winner byte-for-byte, and uses the same primitive to publish
`transport-manifest.json` last. Each derivation is a separate immutable
sub-entry whose `derivation-manifest.json` is linked last, so a future local
normalizer can add a derivation without mutating transport authority. Readers
ignore a transport or derivation without its valid completion manifest. The
first complete transport writer wins. A transport loser succeeds idempotently
only when the uncompressed raw hash/length and deterministic gzip bytes are
identical; byte-different raw authority is a visible same-epoch conflict even
when it normalizes to the same hits. Beneath one identical raw parent, a
derivation loser succeeds only when its exact hits receipt and ordered semantic
digest agree. A changed `corpus_epoch` produces a new transport key rather than
a conflict.
Directory rename publication, ordinary last-writer-wins, and `INSERT OR
REPLACE` are forbidden for semantic state.

An explicit offline mode makes a cache miss fail before any network client is
reachable. Raw response bytes are retained for local rederivation.

Raw response deletion is not part of v2. Any future pruning requires a separate,
explicitly authorized cache schema and garbage-collection policy.

## Integration and compatibility

There is no generated-artifact backward compatibility:

- new canonical checkpoints require `records.sqlite3` and
  `canonical/records-manifest.json` instead of `canonical/candidates.jsonl` and
  `canonical/candidate-manifest.json`;
- the throwaway Task 5.5 prototype is the only v1-reader boundary and is pinned
  to its isolated prototype branch; production v2 code rejects v1 manifests;
- selection manifests refer to records semantic/file hashes;
- retrieval export opens topic records with the expected topic ID, validates
  source reconstruction through `DocumentStore`, seals a per-topic projection,
  and emits unchanged organizer TREC and full-text ZIP bytes;
- existing generated outputs are neither migrated nor deleted. New experiment
  namespaces regenerate topics from reusable retrieval, reranker, and hosted
  response caches.

In-memory text-bearing stage records may remain during slice 1. Slice 3 changes
the persistent request cache first; replacing every in-memory `text: str` field
is optional unless profiling shows it is a runtime-memory bottleneck.

## Topic-first local execution

The schedulable unit is one complete topic, not one pipeline stage. The parent
submits a serializable `TopicJob` containing config identity, topic ID, source
commit, output namespace, and an importable dependency-factory descriptor
(`module`, `qualname`, canonical scalar kwargs). A spawned child process reloads
configuration, resolves that top-level factory, constructs its own
retriever/scorer/model adapters, owns exactly one topic
directory and SQLite database, and runs retrieval through sealed per-topic
projection before returning only receipts and metrics. Non-serializable model
objects and validation sessions never leave the child.

Local process count is configurable and defaults to one for compatibility; the
same topic job contract is movable to other infrastructure without changing
artifact identities. There is no hard-coded global GPU scoring worker. Each
topic is independently runnable, and the operator chooses local concurrency
according to available GPU memory and cache warmth.

At `topic_workers=1`, the existing in-process live adapter objects and arbitrary
callable dependency factory remain supported. `topic_workers>1` requires the
serializable factory descriptor; a lambda, closure, or live adapter object
fails before workers start with an actionable configuration error. Tests use a
module-level deterministic fake factory resolved through the same descriptor.

Topic writers have disjoint output roots. Shared organizer/document caches are
immutable, and shared reranker writes are partitioned or locked before process
parallelism is enabled. Continuation tickets and leases are topic-scoped. An
equivalent-output race may converge on either physical writer after byte and
semantic verification; a contradictory race fails visibly. The parent verifies
all per-topic projection receipts, restores official narrative source order,
publishes the global retrieval outputs, and writes the global manifest last.

## Error handling

The modules fail closed on:

- malformed or mismatched SHA-256 values;
- invalid UTF-8 document objects;
- topic identity mismatch;
- same-topic `docid` content conflict;
- missing or corrupt content objects;
- invalid character/byte ranges or text hashes;
- foreign-key or SQLite integrity errors;
- a sealed database with WAL/SHM companions;
- partial or contradictory request cache entries;
- semantic cache conflict under one request identity;
- changed stage identity, semantic seal, or database byte receipt;
- use of a serialized validation marker as a validation capability;
- malformed organizer records or an offline cache miss before network access.

No validator repairs a sealed artifact in place.

## Required verification

Unit and integration tests must prove:

- same `docid`, different topic/body resolves correctly; same-topic conflict
  fails;
- all candidate text roles reconstruct exactly, including non-ASCII
  character/byte offsets;
- corrupt/missing blobs, database bytes, seals, references, topics, WAL/SHM,
  and offsets are rejected;
- two independent builds have the same semantic seal and selection/export
  bytes;
- equivalent insertion orders and batched per-document writes have the same
  semantic seal and organizer-facing projection;
- surrogate-key allocation changes do not change the seal, while any natural
  identity or semantic field change does;
- same-topic docid/body disagreement is rejected conservatively, never by
  silently choosing text;
- a cross-document candidate/passage link and duplicate `(document,
  passage_id)` are rejected by v2 constraints before publication;
- a `ValidatedTopicRecords` capability rejects changed database, manifest, or
  CAS bytes and cannot be restored from disk;
- same-output concurrent publications converge and conflicting outputs remain
  visible failures;
- crash injection before each publication boundary never exposes a complete
  manifest over partial state;
- transport and derivation manifests reject altered raw authority, wrong
  parent keys, and mismatched raw/hit pairs across every crash/race boundary;
- a copied database and declared object closure validate from a different root;
- existing focused evidence, retrieval-export, and competition-retrieval tests
  pass under the versioned contract;
- one cached representative 2025 topic completes without network/model cache
  misses, produces equivalent selected evidence and organizer projection, and
  demonstrates a material size reduction before slice 3 begins.
- the storage report distinguishes logical topic closure bytes (database plus
  every unique referenced CAS object), marginal newly admitted CAS bytes, and
  total shared-store bytes; the <20% gate uses logical closure bytes;
- untrusted open is <=30 seconds (target <=15), same-session rebind is <=2
  seconds, and selection, canonical, and organizer outputs are exact; the
  per-topic projection handoff consumes the validated capability.
- two topics can execute in separate spawned processes with disjoint output
  ownership and deterministic global manifest-last concatenation.

## Scope exclusions

- No deletion or conversion of existing 216.66 GiB artifacts.
- No broad 2025 or 2026 run during implementation.
- No changes to the concurrent dirty `competition_rag.py` workstream.
- No distributed shared-filesystem guarantee; v2 is verified for local POSIX
  publication and movable topic shards.
