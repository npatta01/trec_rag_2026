# Organizer Pi Baselines: Single-Topic Phoenix Trace Design

**Date:** 2026-07-29  
**Status:** Approved with Phoenix Cloud delivery  
**Topic:** `rag2026-1`

## Objective

Run both organizer-provided TREC RAG 2026 Pi baselines for the same official
topic and make their behavior directly inspectable in a hosted Phoenix project.
The comparison covers the Piika agentic BM25 path and the fixed-retrieval
`ragnarok_style_ag.py` path.

## Scope

- Use the organizers' published source revisions and inputs.
- Run only `rag2026-1`.
- Preserve each baseline's native Pi JSONL events and normal output artifacts.
- Export full-content traces to Phoenix Cloud, including the official narrative,
  prompts, search queries, search results, retrieved passage text, opened
  documents, assistant output, citations, validation results, timings, and
  errors.
- Keep Phoenix credentials out of commands, logs, traces, and tracked files.
- Do not publish another service or change file-sharing permissions.

## Chosen Approach

Run the organizer code unchanged and translate its durable Pi event streams
into OpenInference-compatible OpenTelemetry spans after execution. This avoids
maintaining instrumentation forks in both TypeScript and Python and preserves
the exact baseline behavior. Native events remain the audit source; Phoenix is
the derived interactive view.

Direct source instrumentation was rejected because it would create two tracing
patches with different runtimes and could change the code being reproduced. A
Pi executable proxy was rejected because it would lose organizer-level context
and be sensitive to Pi CLI protocol changes.

## Execution Inputs

### Piika agentic BM25

- Source: `nourj98/piika`, branch `codex/trec-rag-prompt`, pinned to the
  organizer-documented revision.
- Input: a generated one-row TSV copied exactly from the official test topics.
- Retrieval: the organizer-documented authenticated Pyserini REST service and
  `climbmix-400b` index.
- Output: one normalized query artifact, raw Pi events, stderr, and a
  submission-shaped JSONL record.

### Fixed-retrieval Pi generation

- Source: the organizer's `ragnarok_style_ag.py`.
- Input: the same one-row topic TSV, the published six-column ranked run, and
  the published document archive.
- Retrieval depth and document truncation follow the published baseline
  configuration: ranked top 100, independently capped at 1,000 words per
  document.
- Output: one normalized answer row, raw Pi events, stderr, and validation
  status.

## Trace Model

Use one Phoenix project, `trec-rag-2026-pi-baselines`, with two root traces that
share a session identifier and the attributes `topic.id=rag2026-1` and
`content.capture=full`.

### Agentic trace

The `piika-agentic` root contains:

1. prompt construction;
2. each assistant turn;
3. each `search` tool call with reason, query, and complete returned hits;
4. each `read_document` call with reason, docid, and full returned content;
5. final cited answer parsing; and
6. organizer-format validation.

### Fixed-retrieval trace

The `ragnarok-fixed` root contains:

1. ranked top-100 evidence preparation;
2. the complete rendered system and user prompts;
3. the Pi generation call and assistant events;
4. final cited answer parsing; and
5. organizer-format validation.

Use OpenInference span kinds where they fit: `CHAIN` for each root,
`RETRIEVER` for searches and fixed evidence preparation, `TOOL` for document
reads, and `LLM` for assistant generation. Preserve source event timestamps
when present; otherwise use event order and explicitly mark reconstructed timing.

## Data Handling

The user explicitly authorized sending full content to hosted Phoenix. The
exporter reads `PHOENIX_API_KEY` and the collector endpoint from the process
environment. It never serializes authentication headers into native artifacts
or span attributes. Before export, a secret scan rejects known credential
values in trace payloads and local deliverables.

## Failure Handling

- If either baseline fails, retain its native events and export an error-status
  trace instead of hiding the partial execution.
- A failure in one path does not erase or invalidate the other path's evidence.
- If Phoenix export fails, retain a local OTLP-compatible trace representation
  and report the failure without rerunning the paid model call.
- Flush and shut down the OpenTelemetry provider before process exit so the
  short-lived exporter does not lose spans.

## Verification

Before the paid calls, verify:

- pinned source revisions and organizer input hashes;
- one and only one topic in each generated TSV;
- the fixed run has 100 unique ranked documents for `rag2026-1` and all source
  text is available;
- Pi authentication, Pyserini authentication, and Phoenix authentication using
  checks that do not print secrets; and
- trace conversion against synthetic Pi event fixtures.

After execution, verify:

- both native event files and both validated answer records exist;
- Phoenix contains exactly two successful or explicitly failed root traces for
  the shared topic/session;
- expected child span categories and full-content fields are present;
- trace inputs and final answers match the native artifacts; and
- no credential value appears in tracked files, local artifacts, or exported
  attributes.

## Deliverables

- Native organizer artifacts for both one-topic runs under ignored local output
  directories.
- A reusable event-to-Phoenix conversion tool under `code/` with tests and
  adjacent usage documentation.
- Hosted Phoenix project and direct trace links when the API exposes stable
  links.
- A concise handoff containing the topic, run outcomes, artifact paths, Phoenix
  links, and verification evidence.
