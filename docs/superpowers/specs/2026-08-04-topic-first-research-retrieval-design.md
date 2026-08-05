# Topic-First Research Retrieval Design

## Status

Agreed target architecture as of 2026-08-04. This document consolidates the
decisions made for topic-parallel retrieval, researcher handoffs, evidence
storage, Nuggetizer canonicalization, and the generation handoff. It is a
design contract, not a claim that every component is already implemented.

## Architecture

```text
local topic dispatcher
  |
  +-- topic A worker --> topic A directory --> records.sqlite3
  +-- topic B worker --> topic B directory --> records.sqlite3
  +-- topic C worker --> topic C directory --> records.sqlite3

Within one topic worker:

topic coordinator
  |
  +-- track facet coverage and prioritize unanswered facets
  +-- launch independent researchers in parallel
  +-- merge validated researcher handoffs for the same run
  +-- add evidence-discovered facets when useful
  +-- stop when covered or the maximum researcher count is reached
  |
  v
holistic topic evidence ledger
  |
  v
Nuggetizer per changed subnarrative
  |
  +-- atomic deduplicated claims
  +-- importance: vital | okay
  +-- exact evidence and document citations
  |
  v
sealed generation handoff
```

Topics are the unit of execution, storage, retry, resume, and optional
infrastructure sharding. There is no singleton GPU scoring worker. A topic owns
one output directory and one SQLite ledger; topic workers may execute locally
in parallel and can later be assigned to other infrastructure without changing
the topic contract.

## Researcher retrieval contract

Each researcher receives an uncovered evidence goal, prioritizing breadth
across unanswered facets before deeper investigation.

```text
focused facet query
  -> organizer retrieval at depth 1,000
  -> deterministic citation-bound passage chunking
  -> Mixedbread ranking of the resulting passages against that query
  -> return the top 100 passages to the researcher
  -> retain every relevant passage needed for the facet
  -> validate passage text, source document, hashes, and citation
  -> publish a researcher handoff into the topic run
```

A query has one primary information need so retrieval and reranking remain
focused. One passage may support several facets, and the ledger can credit it
to each applicable subnarrative without duplicating the source document.

Researchers may return multiple passages from one document or evidence from
several documents. Citations always identify the source document. A researcher
validates its own retrieved passages before handoff; the coordinator does not
repeat expensive retrieval or scoring.

## Facet growth and stopping

- The initial narrative creates the first facet set.
- Research may reveal a missing facet; the coordinator may add it.
- New researchers target uncovered or weak facets before deepening an already
  well-supported facet.
- The topic stops when coverage is sufficient or the maximum researcher count
  is reached.
- There is no separate topic-wide retrieval-K cap. The fixed per-query bounds
  are retrieval depth 1,000 documents and a reranked passage ceiling of 100.
- Budget exhaustion, a deadline, or absent data does not discard useful work.
  The topic persists an explicit incomplete result and generation may consume
  what exists.

An incomplete result is never reported as complete. Individual researcher work
is committed only after its evidence validates. API calls use a 300-second
deadline with bounded retry and no silent fallback to another model or method.

## Ledger and storage

```text
runs/<run_id>/topics/<topic-directory>/
|-- records.sqlite3
|-- researchers/
|   `-- <researcher-id>.<checkpoint-or-handoff>
|-- manifest.json
`-- handoff/
    `-- generation_handoff_manifest.json
```

Opening the topic directory establishes topic identity. Ordinary ledger
methods therefore do not accept a separate topic ID. Every researcher handoff
must belong to the same run ID before it can merge into that topic.

The per-topic database provides one holistic coordinator view containing:

- facet and subnarrative identities and coverage;
- query and researcher provenance;
- retrieved candidates and reranker scores;
- document bindings;
- validated passage offsets and hashes;
- deduplicated evidence links;
- canonical nuggets and their vital/okay labels; and
- complete/incomplete state and stopping reason.

Document content is stored once in the shared content-addressed document store.
The topic database binds `(topic, docid)` to the content hash. Candidates,
passages, and nuggets reference the document plus exact character/byte offsets
instead of repeating full text. Cross-topic `docid -> text` lookup is forbidden.
Passage-handle identity includes the exact citation `docid` as well as content,
span, text, and chunker identity. Duplicate-content documents therefore remain
separately citable; score-cache identity may still deduplicate their identical
query/text model computation.

## Cache contract

```text
cache/documents/v1/<content-sha256>
cache/retrieval/<request-sha256>/...
cache/reranker/<facet-query-sha256>/<passage-identity-sha256>/...
```

- Organizer response bytes and normalized document references are cached by
  complete request identity.
- Mixedbread passage scores are cached by the exact facet query, source
  document content, passage offsets/text identity, model identity, and scoring
  configuration.
- An unchanged retrieval or score is never recomputed.
- Nuggetizer is rerun only for subnarratives whose admitted evidence changed.
- Creator and scorer provenance are retained; the scorer output is `vital` or
  `okay`.

Old cache and artifact schemas are not backward compatible. A version mismatch
rejects them; they may be deleted when convenient.

## Coordinator interface

The coordinator consumes one materialized topic snapshot, rather than issuing
separate lookups into every researcher ledger. Researcher checkpoints are
handoff inputs; after merge, the topic ledger is authoritative.

The generation handoff contains the holistic set of subnarratives, coverage,
canonical nuggets, exact evidence, document citations, completion state, and
stopping reason. Answer generation may run on an incomplete handoff, but any
downstream inability to answer must remain an explicit error or limitation.

## Publication and concurrency

Workers write temporary artifacts and publish atomically. Concurrent writers
for the same deterministic topic output are harmless only when the bytes are
identical. Different bytes for the same topic/run identity are an integrity
conflict and must not be silently mixed.

The global exporter receives sealed topic receipts and concatenates topics in
official source order. A global manifest is published last.

The 1,000-document and 100-passage values are internal search ceilings, not an
organizer submission cutoff.  The final Retrieval run chooses a variable
document depth independently for each narrative and emits only deduplicated
documents backed by validated evidence that the system predicts is useful for
answer generation.  It must never pad a topic to 100 or 1,000 rows.  Passage
text remains private evidence state; organizer run rows contain document IDs
and document-level ranks/scores only.

## Implementation boundary

The hosted Nuggetizer creator/scorer path, `vital`/`okay` persistence, scorer
mode cache identity, composite creator/scorer cache provenance, and private
retrieval projection labels are implemented in the active worktree. The full
topic-first researcher/coordinator orchestration remains an architectural
boundary to verify against whichever retrieval implementation checkpoint is
integrated next.
