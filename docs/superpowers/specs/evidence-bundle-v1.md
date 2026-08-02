# Evidence Bundle v1

## Goal

Create one validated, arm-neutral boundary artifact for fixed narrative
retrieval, subnarrative retrieval, agentic retrieval, direct evidence
extraction, canonical nuggetization, and downstream RAG/AG generation.

The bundle preserves the natural deduplicated document union. It never silently
truncates a lane union to a fixed depth. Any depth or context-budget projection
is recorded as a named derived selection.

## Contract

An `EvidenceBundle` is a per-topic record with these independent relations:

- `lanes`: narrative, subnarrative, or agentic query lanes with stable IDs,
  query text, parent lane, and producer metadata.
- `documents`: each unique corpus document once, with text and a content hash.
- `retrieval_events`: one pre-dedup row per lane/document observation, retaining
  rank, score, retriever identity, and optional search/read event identity.
- `selections`: derived views over the natural union, retaining policy, input
  count, output count, explicit per-document inclusion/output rank, and
  rejection reason when applicable. The natural union is not a ranked view.
- `evidence`: exact sentence/passage/document spans linked to a document and
  one or more lanes, with selector identity and source hashes.
- `nuggets`: direct or canonical claims linked to one or more evidence IDs and
  optional subnarrative IDs.
- `trace_refs`: optional hashed references to external agent traces. The full
  trace is not embedded in the bundle.

All identifiers are stable within a bundle. All text-bearing records carry
SHA-256 hashes. Every evidence span must resolve to its parent document text;
every nugget support must resolve to an evidence record.

## Downstream projections

The bundle package provides deterministic projections for:

1. Organizer fixed retrieval: multi-topic query TSV, six-column TREC run, and
   document text JSONL/ZIP sidecar.
2. Fixed-bundle RAG: ordered per-topic context records containing documents, evidence,
   and nuggets without enabling new retrieval.

Agentic answer generation may consume the fixed-bundle projection. If it is
allowed to retrieve new documents, those observations must be written as a
new bundle revision rather than hidden in the answer trace.

## Compatibility policy

Existing `ExtractiveCandidate`, `SubnarrativeSelection`, and
`CanonicalNuggetResult` records remain valid stage-internal artifacts during the
first implementation. Deterministic compilers/adapters make the bundle the
cross-stage contract. Replacing those internals can be considered after the
compiler passes the two-topic fixture and round-trip tests.

## Non-goals

- No hosted retrieval, nuggetization, answer generation, or agent execution.
- No fixed top-100 requirement.
- No ranking policy hidden inside the bundle model.
- No deletion of existing sealed checkpoint formats.
