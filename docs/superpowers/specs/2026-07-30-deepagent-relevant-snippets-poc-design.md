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
- return ten snippets per page by default and allow pagination until all ranked
  snippets are exhausted;
- use the existing semantic chunker and a local Mixedbread cross-encoder by
  default;
- keep a small-LLM ranker as an optional implementation of the same narrow
  ranking interface;
- preserve optional Phoenix tracing without tracing complete documents.

The POC will not add global limits on inspected documents, snippet pages, or
snippet count. It will not integrate with the official pipeline, organizer
export, or sealed experiment artifacts. It will not add a CLI, persistent chat
state, a second corpus retriever, or a production-hardening test matrix.

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
and the invocation-local registry. The retrieval-only middleware exposes exactly
`search_climbmix` and `extract_relevant_snippets`; filesystem, shell, task, and
other Deep Agent tools remain unavailable.

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
stable ranking.

The optional small-LLM ranker receives bounded batches of chunk IDs and text and
returns an ordering. It never receives a complete document in one prompt. This
adapter remains dependency-injected so the default path makes no additional
hosted inference call.

Cursor contents are opaque to the model and bind the document ID, focus-query
hash, ranker/config identity, document-text hash, and next offset. A different
document, query, ranker, configuration, or document body invalidates the cursor.

## Invocation State and Data Flow

1. Search the untouched narrative through the existing retriever.
2. Store each complete candidate by document ID in a private registry.
3. Send only narrative plus search metadata to the first model call.
4. Let the agent issue follow-up searches; merge their candidates into the same
   registry and return metadata only.
5. Let the agent call `extract_relevant_snippets` for a registered document and
   focus query.
6. Chunk once per document/configuration, rank once per document/focus/backend,
   and cache both results for the invocation.
7. Return the next ten ranked, non-duplicate snippets and an optional cursor.
8. Preserve the existing deterministic document-level RRF result independently
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
content; metadata-only mode redacts it.

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
5. A lightweight ranker-contract test covers the default adapter boundary and
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
