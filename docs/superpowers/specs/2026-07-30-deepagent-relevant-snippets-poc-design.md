# Deep Agent Relevant-Snippets POC Design

## Objective

Replace blind document-prefix excerpts in the Deep Agent retrieval prototype
with compact search metadata and an on-demand `extract_relevant_snippets` tool.
The POC should let the agent inspect multiple relevant passages from long
documents without copying every retrieved document into the main model context.

## Scope

The POC will:

- preserve the existing narrative-first ClimbMix search and bounded follow-up
  search behavior;
- make original and follow-up search results model-visible as metadata only;
- keep complete retrieved document text in an invocation-local registry;
- allow the agent to request relevance-ranked snippet pages from any document
  retrieved during that invocation;
- make both retrieval tools cache-first without exposing cache decisions,
  keys, paths, or controls to the agent;
- return ten snippets per page by default and allow pagination until all ranked
  snippets are exhausted;
- use the existing semantic chunker and a local Mixedbread cross-encoder by
  default;
- keep a small-LLM ranker as an optional implementation of the same narrow
  ranking interface;
- provide state-backed ephemeral scratch files for notes and automatic spill of
  oversized tool results;
- preserve optional Phoenix tracing without tracing complete documents.

The POC will not add global limits on inspected documents, snippet pages, or
snippet count. It will not integrate with the official pipeline, organizer
export, or sealed experiment artifacts. It will not add a CLI, persistent chat
state, a second corpus retriever, or a production-hardening test matrix.
Scratch state is not a persistent cache and is discarded with the SDK
invocation.

## Confirmed ClimbMix Schema

The cached byte-for-byte ClimbMix responses contain `doc`, `docid`, `rank`, and
`score`. The document is unstructured plain text; no structured title or
headline is available. Search results therefore omit `title` rather than
inventing or deriving one.

## Agent Tools

### `search_climbmix`

The existing tool keeps its query and search-budget behavior but returns only:

```json
{
  "documents": [
    {
      "docid": "shard_06460_56923",
      "rank": 1,
      "score": 12.34,
      "text_length": 48291
    }
  ],
  "remaining_budget": 2
}
```

The mandatory original-narrative search is still performed before the first
model call. Its initial model message contains the untouched narrative plus the
same metadata-only document list. Full text remains in SDK-side result records
and the invocation-local registry. Cache hits and misses are intentionally absent
from the model-visible payload.

### `extract_relevant_snippets`

```python
extract_relevant_snippets(
    document_id: str,
    focus_query: str,
    cursor: str | None = None,
) -> str
```

The tool accepts only a document ID already retrieved during the current
invocation. It returns a JSON page:

```json
{
  "document_id": "shard_06460_56923",
  "focus_query": "refugee challenges and asylum laws",
  "snippets": [
    {
      "chunk_id": "shard_06460_56923:0007",
      "start_char": 8100,
      "end_char": 9310,
      "text": "...",
      "relevance_score": 0.91
    }
  ],
  "next_cursor": "opaque-cursor-or-null"
}
```

`snippets_per_page` is a fixed SDK configuration value with default `10`; the
agent cannot override it in a tool call. Page size is count-based, not token- or
character-based. There is no global page limit. A cursor resumes the stable
ranking at the next offset until `next_cursor` is `null`.

## Snippet Extraction

`deepagent_snippets` owns a small, testable boundary:

- `SnippetExtractionConfig`: page size plus chunk size/overlap;
- `SnippetRanker`: ranks a focus query against immutable `TextChunk` records;
- `LocalMixedbreadSnippetRanker`: default local implementation;
- `SmallLLMSnippetRanker`: optional adapter using a supplied small chat model;
- `RelevantSnippetExtractor`: chunking, ranking, de-duplication, pagination, and
  cursor validation;
- immutable snippet/page records.

The default reuses `SemanticTextChunker` with the repository's established
3,500-character chunk size and 350-character overlap. The local ranker uses the
pinned `mixedbread-ai/mxbai-rerank-base-v2` cross-encoder and shared score cache.
Scores order chunks by descending relevance, then stable chunk ID. Overlapping
or duplicate selections are suppressed while distinct relevant regions from
the same document remain eligible. The POC ranks every non-empty chunk and uses
no minimum-score threshold, so pagination can continue through the complete
stable, de-duplicated ranking.

The optional small-LLM ranker receives bounded batches of chunk IDs and text and
returns an ordering. It never receives a complete document in one prompt. This
adapter remains dependency-injected so the default path makes no additional
hosted inference call.

Cursor contents are opaque to the model and bind the document ID, focus-query
hash, ranker/config identity, document-text hash, and next offset. A different
document, query, ranker, configuration, or document body invalidates the cursor.

## Cache-First Tool Execution

Caching belongs to the tools, not the agent. The model never receives cache
arguments, cache status, cache paths, continuation tickets, or instructions to
reuse a prior result.

`search_climbmix` retains the existing `PyseriniRemoteRetriever` cache-first
behavior and validated provenance. Its only agent-supplied argument, the exact
`query`, participates in the existing request-cache identity alongside the
effective retrieval configuration; there is no model-visible cache argument.
`extract_relevant_snippets` adds a content-addressed result cache under the
repository's shared reranker cache.
Before chunking, local model loading, or optional hosted inference, it checks an
exact tool-result identity containing:

- every tool argument: `document_id`, the exact `focus_query`, and the explicit
  `cursor` value including `null`;
- the retrieved document-text SHA-256;
- `snippets_per_page`, chunk size, overlap, and de-duplication policy;
- ranker backend, model, revision, scoring policy, and implementation version;
- cursor schema and tool-result schema versions.

Changing any argument or effective configuration produces a different cache
identity. The first-page `cursor=null` call is cached like every later page.
Identical small-LLM calls therefore do not repeat hosted inference. Cache files
store the full identity, output hash, and response; malformed, stale, or
identity-mismatched entries are rejected rather than returned. Cache lookup and
write behavior remains invisible to document ranking and agent reasoning.

Chunk and score caches may use narrower internal identities, but the public
tool-result cache always includes every tool argument. No cache content is
stored in Deep Agent scratch state.

## Ephemeral Scratch and Context Spill

The agent uses an explicit Deep Agents 0.7 `StateBackend`, never
`FilesystemBackend`, `LocalShellBackend`, or another host-mounted backend. It
may use only state-backed `ls`, `read_file`, `write_file`, `edit_file`, `delete`,
`glob`, and `grep` alongside the two retrieval tools. `execute`, shell access,
`task`, subagents, skills, memory, and host paths remain unavailable.

Deep Agents' filesystem middleware automatically moves an oversized tool result
to `/large_tool_results/...` in the ephemeral state backend and returns a
preview plus file reference. The agent may inspect that file with paginated
`read_file` calls or write its own temporary notes. This scratch space exists
only for the current SDK invocation and is never used as the persistent result
cache.

## Invocation State and Data Flow

1. Search the untouched narrative through the existing retriever.
2. Store each complete candidate by document ID in a private registry.
3. Send only narrative plus search metadata to the first model call.
4. Let the agent issue follow-up searches; merge their candidates into the same
   registry and return metadata only.
5. Let the agent call `extract_relevant_snippets` for a registered document and
   focus query.
6. Validate the complete cache identity and return an exact cached page before
   chunking, loading a ranker, or making an optional hosted call.
7. On a cache miss, chunk and rank, persist the validated page, and return the
   next ten ranked, non-duplicate snippets plus an optional cursor.
8. Let Deep Agents spill oversized tool results or agent notes into ephemeral
   state-backed scratch files when needed.
9. Preserve the existing deterministic document-level RRF result independently
   of which snippets the agent inspected.

If the same document ID appears in multiple searches, the first non-empty text
must match later text; conflicting document content is an error rather than a
silent registry overwrite.

## Context and Tracing Boundaries

Complete documents never enter agent messages, tool outputs, or Phoenix span
attributes. Search spans retain document IDs, ranks, scores, lengths, cache
state, and latency. A snippet-extraction span records document/chunk IDs,
offsets, scores, backend identity, page offset, and whether another page exists.
Snippet text is traced only when the existing `trace_content` setting allows
content; metadata-only mode redacts it. Scratch paths and scratch-file contents
are not added by the SDK to manual Phoenix span attributes.

## Failure Behavior

- Unknown or not-yet-retrieved document IDs return a tool-visible error.
- Blank focus queries return a tool-visible error.
- Invalid or mismatched cursors return a tool-visible error without revealing
  cursor internals.
- An empty or unchunkable document returns an empty page with
  `next_cursor: null`; the tool never falls back to the first document
  characters.
- Chunker/ranker failures are surfaced without hidden retries.
- Optional small-LLM failures do not silently switch ranking backends.
- Invalid or mismatched cache entries fail with a non-disclosing cache-integrity
  error; the agent is not asked to repair or bypass them.
- Snippet failures do not change completed searches or document-level fusion.

## POC Verification

Keep verification deliberately small:

1. A long-document smoke test places the relevant passage near the end and
   proves the local ranker returns it rather than the document prefix.
2. A pagination smoke test proves multiple relevant snippets continue across
   stable ten-item pages without duplication.
3. An agent-context smoke test proves original and follow-up search payloads
   contain metadata but not complete document text.
4. An authorization smoke test rejects a document ID that was not retrieved in
   the current invocation.
5. A cache-first smoke test proves an exact repeated call does no ranker/hosted
   work and that changing each tool argument changes the result-cache identity.
6. A tool-allowlist smoke test permits ephemeral state read/write tools while
   rejecting shell, task, and host-filesystem access.
7. A lightweight ranker-contract test covers the default adapter boundary and
   the optional small-LLM adapter without a live hosted call.

Run the existing focused Deep Agent, tracing, chunking, and remote-retriever
tests plus the repository suite. A live provider run is optional for this POC;
do not spend another OpenRouter or Pyserini request merely to validate local
snippet selection.

## Deliverables

- `code/trec_rag/deepagent_snippets.py`: snippet records, ranker boundary,
  extraction, pagination, and cursor handling;
- `code/trec_rag/deepagent_retrieval.py`: metadata-only search payloads,
  invocation registry, and the second agent tool;
- `code/trec_rag/deepagent_tracing.py`: bounded snippet span support;
- focused POC tests and README usage notes.
