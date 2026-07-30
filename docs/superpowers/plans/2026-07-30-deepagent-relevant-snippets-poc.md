# Deep Agent Relevant Snippets POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace blind document-prefix context with cache-first, paginated relevant-snippet extraction while giving the Deep Agent only ephemeral state-backed scratch space.

**Architecture:** Keep full ClimbMix documents in an invocation-local registry and expose metadata-only search results plus `extract_relevant_snippets(document_id, focus_query, cursor=None)`. A focused snippet module owns semantic chunking, ranker adapters, stable de-duplication, opaque cursors, exact tool-result caching, and internal score caching. Deep Agents 0.7 runs with an explicit `StateBackend` and a strict allowlist containing only retrieval tools and safe state filesystem tools.

**Tech Stack:** Python 3.12, Deep Agents 0.7, LangChain/OpenRouter, `semantic-text-splitter`, Sentence Transformers Mixedbread cross-encoder, repository JSON/JSONL caches, OpenTelemetry/OpenInference/Phoenix, pytest.

## Global Constraints

- Search the supplied narrative byte-for-byte before the first model call.
- Search and snippet tool payloads must not expose cache controls, keys, paths, status, or full documents.
- Every public tool argument and every effective extraction/ranker setting participates in the exact tool-result cache identity.
- `snippets_per_page` is a fixed SDK setting with default `10`; there is no global snippet/page/document limit.
- Use 3,500-character semantic chunks with 350-character overlap.
- Default to `mixedbread-ai/mxbai-rerank-base-v2` revision `3ea9d4dffa7d12a4f366be8e275c349de9fc9865`; retain a dependency-injected small-LLM adapter.
- Omit document titles because ClimbMix does not provide them.
- Agent scratch is invocation-local `StateBackend` data, never a persistent cache or host-mounted filesystem.
- Do not add CLI integration, official-pipeline integration, organizer export changes, live-provider tests, or a production-hardening matrix.
- Keep verification POC-sized: focused smoke tests plus the existing repository suite.

---

## File Map

- Create `code/trec_rag/deepagent_snippets.py`: snippet records, ranker contract and adapters, exact page cache, score-cache integration, de-duplication, cursor validation, and default construction.
- Create `code/tests/test_deepagent_snippets.py`: long-document, pagination, cache-identity, cache-first, and ranker-adapter smoke tests.
- Modify `code/trec_rag/deepagent_retrieval.py`: metadata-only search views, invocation document registry, snippet tool, explicit state backend, and safe tool allowlist.
- Modify `code/tests/test_deepagent_retrieval.py`: three-argument agent factory, metadata-only assertions, snippet authorization/integration, and scratch allowlist coverage.
- Modify `code/trec_rag/deepagent_tracing.py`: document-length search evidence and bounded snippet-page spans.
- Modify `code/tests/test_deepagent_tracing.py`: snippet span validation/redaction and revised search metadata assertions.
- Modify `code/trec_rag/README.md`: Python SDK example and concise explanation of snippet pagination, caching ownership, and scratch lifetime.

---

### Task 1: Core snippet extraction, pagination, and exact page cache

**Files:**
- Create: `code/trec_rag/deepagent_snippets.py`
- Create: `code/tests/test_deepagent_snippets.py`

**Interfaces:**
- Consumes: `trec_rag.chunking.TextChunk`, `TextChunker`, `ChunkingConfig`, and `SemanticTextChunker`.
- Produces: `SnippetExtractionConfig`, `ScoredTextChunk`, `RelevantSnippet`, `SnippetPage`, `SnippetExtractionResult`, `SnippetRanker`, `SnippetResultCache`, and `RelevantSnippetExtractor.extract(document_id, document_text, focus_query, cursor=None)`.

- [ ] **Step 1: Add representative extraction and cache smoke tests**

Use an injected deterministic ranker and temporary cache to cover the relevant passage near the end, ten-item pages, multiple snippets from one document, stable no-duplicate continuation, blank-query rejection, invalid-cursor rejection, and exact cache behavior:

```python
class CountingRanker:
    identity = {"backend": "test", "model": "keyword-v1"}

    def __init__(self) -> None:
        self.calls = 0

    def rank(self, focus_query: str, chunks: Sequence[TextChunk]) -> tuple[ScoredTextChunk, ...]:
        self.calls += 1
        words = set(focus_query.casefold().split())
        return tuple(
            ScoredTextChunk(chunk=chunk, relevance_score=float(sum(word in chunk.text.casefold() for word in words)))
            for chunk in chunks
        )


def test_cache_first_pages_use_every_tool_argument(tmp_path: Path) -> None:
    ranker = CountingRanker()
    extractor = RelevantSnippetExtractor(
        ranker=ranker,
        result_cache=SnippetResultCache(tmp_path / "pages"),
        config=SnippetExtractionConfig(snippets_per_page=10),
    )
    first = extractor.extract("doc-a", LONG_DOCUMENT, "target passage", cursor=None)
    repeated = extractor.extract("doc-a", LONG_DOCUMENT, "target passage", cursor=None)
    assert repeated.page == first.page
    assert repeated.cache_status == "hit"
    assert ranker.calls == 1
    assert extractor.extract("doc-a", LONG_DOCUMENT, "different query", None).cache_status == "miss"
    assert extractor.extract("doc-b", LONG_DOCUMENT, "target passage", None).cache_status == "miss"
    assert first.page.next_cursor is not None
    assert extractor.extract("doc-a", LONG_DOCUMENT, "target passage", first.page.next_cursor).cache_status == "miss"
```

- [ ] **Step 2: Run the new test file and confirm the missing-module failure**

Run: `.venv/bin/python -m pytest code/tests/test_deepagent_snippets.py -q`

Expected: collection fails because `trec_rag.deepagent_snippets` does not exist.

- [ ] **Step 3: Implement immutable page records and complete cache identities**

Create the focused contracts and keep model-visible serialization separate from internal cache metadata:

```python
@dataclass(frozen=True)
class SnippetExtractionConfig:
    snippets_per_page: int = 10
    chunk_max_characters: int = 3_500
    chunk_overlap_characters: int = 350
    duplicate_overlap_ratio: float = 0.8


@dataclass(frozen=True)
class ScoredTextChunk:
    chunk: TextChunk
    relevance_score: float


@dataclass(frozen=True)
class RelevantSnippet:
    chunk_id: str
    start_char: int
    end_char: int
    text: str
    relevance_score: float


@dataclass(frozen=True)
class SnippetPage:
    document_id: str
    focus_query: str
    snippets: tuple[RelevantSnippet, ...]
    next_cursor: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "focus_query": self.focus_query,
            "snippets": [asdict(snippet) for snippet in self.snippets],
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True)
class SnippetExtractionResult:
    page: SnippetPage
    cache_status: Literal["hit", "miss"]
    ranker_backend: str
    page_offset: int


class SnippetRanker(Protocol):
    @property
    def identity(self) -> Mapping[str, object]:
        raise NotImplementedError

    def rank(self, focus_query: str, chunks: Sequence[TextChunk]) -> tuple[ScoredTextChunk, ...]:
        raise NotImplementedError
```

Validate positive page/chunk settings, finite scores, nonblank IDs/query, and cursor/config bindings. The exact cache identity must include `document_id`, exact `focus_query`, exact `cursor` including `None`, document SHA-256, page size, chunk settings, overlap policy, complete ranker identity, cursor schema version, result schema version, and implementation version.

- [ ] **Step 4: Implement stable ranking, de-duplication, opaque cursors, and cache-first extraction**

`RelevantSnippetExtractor.extract` must perform operations in this order:

```python
identity = self._result_identity(document_id, document_text, focus_query, cursor)
cached = self._result_cache.get(identity)
if cached is not None:
    cached_page, cached_offset = cached
    return SnippetExtractionResult(
        cached_page, "hit", str(self._ranker.identity["backend"]), cached_offset
    )

offset = self._decode_and_validate_cursor(cursor, identity_without_cursor)
chunks = tuple(self._chunker.split_text(document_text, document_id=document_id))
ranked = sorted(
    self._ranker.rank(focus_query, chunks),
    key=lambda row: (-row.relevance_score, row.chunk.chunk_id),
)
deduplicated = self._deduplicate(ranked)
page = self._page(document_id, focus_query, deduplicated, offset)
self._result_cache.put(identity, page, page_offset=offset)
return SnippetExtractionResult(
    page, "miss", str(self._ranker.identity["backend"]), offset
)
```

Suppress normalized exact duplicates and chunks whose character-span intersection divided by the shorter span is at least `duplicate_overlap_ratio`; preserve distinct relevant regions. Encode only schema version, binding digest, and next offset into URL-safe base64 cursor text. `SnippetResultCache` stores canonical identity, model-visible response, internal page offset, and response SHA-256 at `schema_v1/<digest[:2]>/<digest>.json`, uses `FileLock`, writes via temporary-file replacement, and raises a constant `SnippetCacheIntegrityError` for malformed or mismatched entries. The internal page offset is returned in `SnippetExtractionResult` for tracing but omitted by `SnippetPage.as_dict()`.

- [ ] **Step 5: Run the core snippet smoke tests**

Run: `.venv/bin/python -m pytest code/tests/test_deepagent_snippets.py -q`

Expected: all core extraction, pagination, validation, and exact-cache tests pass without loading a hosted or local model.

- [ ] **Step 6: Commit the core snippet boundary**

```bash
git add code/trec_rag/deepagent_snippets.py code/tests/test_deepagent_snippets.py
git commit -m "Add cache-first relevant snippet extraction"
```

---

### Task 2: Local Mixedbread and optional small-LLM ranker adapters

**Files:**
- Modify: `code/trec_rag/deepagent_snippets.py`
- Modify: `code/tests/test_deepagent_snippets.py`

**Interfaces:**
- Consumes: `SnippetRanker`, `ScoredTextChunk`, `GlobalScoreCache`, `ScoreCacheContext`, and dependency-injected model objects.
- Produces: `LocalMixedbreadSnippetRanker`, `SmallLLMSnippetRanker`, and `create_default_snippet_extractor(root, config=None, device="auto")`.

- [ ] **Step 1: Add lightweight adapter-contract tests**

Use fake cross-encoder and fake chat-model objects. Assert lazy model loading, cached scores skip prediction/invocation, exact ranker identity includes backend/model/revision/scoring/version/batch settings, missing scores alone are inferred, and small-LLM JSON must contain one finite score for each requested chunk ID. Do not make live hosted or Hugging Face calls.

```python
def test_local_ranker_scores_only_cache_misses(tmp_path: Path) -> None:
    model = FakeCrossEncoder(scores=[0.8, 0.2])
    ranker = LocalMixedbreadSnippetRanker(
        score_cache_root=tmp_path,
        model_loader=lambda **_kwargs: model,
        device="cpu",
    )
    assert [row.relevance_score for row in ranker.rank("query", CHUNKS)] == [0.8, 0.2]
    assert [row.relevance_score for row in ranker.rank("query", CHUNKS)] == [0.8, 0.2]
    assert model.predict_calls == 1
```

- [ ] **Step 2: Run only the adapter tests and confirm they fail**

Run: `.venv/bin/python -m pytest code/tests/test_deepagent_snippets.py -q -k 'ranker or adapter'`

Expected: failures identify missing concrete ranker classes.

- [ ] **Step 3: Implement the lazy local cross-encoder adapter**

Use pinned constants and repository score-cache conventions:

```python
DEFAULT_SNIPPET_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
DEFAULT_SNIPPET_MODEL_REVISION = "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"

class LocalMixedbreadSnippetRanker:
    @property
    def identity(self) -> Mapping[str, object]:
        return {
            "backend": "sentence_transformers_cross_encoder",
            "backend_version": self._backend_version,
            "model": self._model_name,
            "model_revision": self._model_revision,
            "score_representation": "raw_logits",
            "max_length": self._max_length,
            "batch_size": self._batch_size,
            "device": self._device,
            "implementation_version": 1,
        }
```

Build a `GlobalScoreCache` under `cache/reranker/score_cache`, retrieve per-query/per-chunk scores before loading the model, lazily instantiate `sentence_transformers.CrossEncoder`, predict only missing `(focus_query, chunk.text)` pairs with identity activation/raw logits, persist scores, then return rows in input-chunk order. Device `auto` resolves to ROCm-visible `cuda` when available, otherwise CPU.

- [ ] **Step 4: Implement the optional bounded small-LLM adapter**

Accept an injected chat model with `invoke`, `batch_size=10`, ranker identity, and `GlobalScoreCache`. Prompt one bounded batch at a time with only `focus_query` plus `{chunk_id, text}` records and require strict JSON:

```json
{"scores":[{"chunk_id":"doc:0000","score":0.87}]}
```

Reject missing, duplicated, unknown, boolean, or non-finite scores. Store validated per-chunk scores before returning the global deterministic ordering. Never fall back to the local backend on failure.

- [ ] **Step 5: Add the default extractor factory and run adapter tests**

`create_default_snippet_extractor` must construct `SemanticTextChunker(ChunkingConfig(max_characters=3500, overlap_characters=350))`, `LocalMixedbreadSnippetRanker`, and `SnippetResultCache(repo_cache_root(root) / "reranker" / "deepagent_snippets")` without loading the model.

Run: `.venv/bin/python -m pytest code/tests/test_deepagent_snippets.py -q`

Expected: all snippet tests pass offline.

- [ ] **Step 6: Commit ranker adapters**

```bash
git add code/trec_rag/deepagent_snippets.py code/tests/test_deepagent_snippets.py
git commit -m "Add snippet ranker adapters"
```

---

### Task 3: Metadata-only Deep Agent tools and ephemeral scratch

**Files:**
- Modify: `code/trec_rag/deepagent_retrieval.py`
- Modify: `code/tests/test_deepagent_retrieval.py`

**Interfaces:**
- Consumes: `RelevantSnippetExtractor`, `SnippetExtractionResult`, `create_default_snippet_extractor`, and Deep Agents `StateBackend`.
- Produces: metadata-only `search_climbmix`, `extract_relevant_snippets(document_id, focus_query, cursor=None)`, a three-argument `AgentFactory`, and safe state-filesystem access.

- [ ] **Step 1: Update focused integration tests first**

Change fake factories to receive both tool callables:

```python
def agent_factory(
    model: str,
    search_tool: Callable[[str], str],
    snippet_tool: Callable[[str, str, str | None], str],
) -> FakeAgent:
    return FakeAgent(
        lambda payload: {
            "messages": [{"role": "assistant", "content": "Coverage is sufficient."}]
        }
    )
```

Add smoke assertions that the first prompt and follow-up payload contain only `docid`, `rank`, `score`, and `text_length`; a unique sentinel from full document text is absent. Call the snippet tool for an original document and a follow-up document, reject unknown IDs and blank focus queries, preserve full documents in `AgentRetrievalResult`, and raise on conflicting nonempty text for one document ID.

- [ ] **Step 2: Run the focused retrieval tests and confirm expected failures**

Run: `.venv/bin/python -m pytest code/tests/test_deepagent_retrieval.py -q`

Expected: failures are limited to the new factory signature, metadata payload, snippet tool, and allowlist expectations.

- [ ] **Step 3: Replace prefix excerpts with metadata and an invocation registry**

Replace `_bounded_candidates` with:

```python
def _candidate_metadata(candidates: Sequence[RetrievedCandidate], *, limit: int) -> list[dict[str, object]]:
    return [
        {
            "docid": candidate.docid,
            "rank": candidate.rank,
            "score": candidate.score,
            "text_length": len(candidate.text),
        }
        for candidate in candidates[:limit]
    ]
```

Within `retrieve`, maintain `documents: dict[str, str]`. Register every completed search result before exposing its metadata. An existing nonempty document body must exactly match later content; otherwise raise an error rather than overwriting it.

Create `extract_relevant_snippets` as a closure with exact arguments. It returns JSON from `result.page.as_dict()`, never internal `cache_status`, ranker identity, keys, or paths. Unknown document IDs, blank focus queries, invalid cursors, and extractor failures return concise tool-visible JSON errors without exposing internals.

- [ ] **Step 4: Give the agent only retrieval tools plus state-backed scratch**

Change `_create_agent` to accept both retrieval tools and pass an explicit backend:

```python
create_deep_agent(
    model=provider_model,
    tools=[search_tool, snippet_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
    middleware=[_RetrievalOnlyMiddleware()],
    backend=StateBackend(),
)
```

Use this exact allowlist:

```python
_ALLOWED_TOOLS = frozenset({
    "search_climbmix", "extract_relevant_snippets",
    "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep",
})
```

Continue setting `parallel_tool_calls=False`. Tests must prove `write_file` and `read_file` survive both model and tool middleware boundaries while `execute`, `task`, and an unrelated tool are denied. Assert the real Deep Agents factory exposes the two retrieval tools and seven safe filesystem tools, uses `StateBackend`, and does not expose `task` or `execute`.

Revise the system prompt to direct targeted follow-up searching, document inspection through the snippet tool, pagination when more evidence is useful, and ephemeral state-file use only for oversized output or temporary notes. Do not discuss caching in the prompt.

- [ ] **Step 5: Thread default extraction through constructor and `from_env`**

Add optional `snippet_extractor: RelevantSnippetExtractor | None` injection. `from_env(root=...)` constructs the default extractor rooted in the shared repository cache. Direct construction may lazily create the same default only if the snippet tool is called, so existing retrieval-only tests do not load a ranker.

- [ ] **Step 6: Run focused Deep Agent tests**

Run: `.venv/bin/python -m pytest code/tests/test_deepagent_snippets.py code/tests/test_deepagent_retrieval.py -q`

Expected: all focused extraction, metadata-context, registry, authorization, middleware, factory, concurrency, and fusion tests pass.

- [ ] **Step 7: Commit the SDK integration**

```bash
git add code/trec_rag/deepagent_retrieval.py code/tests/test_deepagent_retrieval.py
git commit -m "Integrate relevant snippets with Deep Agent retrieval"
```

---

### Task 4: Phoenix snippet spans, documentation, and verification

**Files:**
- Modify: `code/trec_rag/deepagent_tracing.py`
- Modify: `code/trec_rag/deepagent_retrieval.py`
- Modify: `code/tests/test_deepagent_tracing.py`
- Modify: `code/tests/test_deepagent_retrieval.py`
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes: `SnippetExtractionResult.page`, internal cache status, ranker backend, and existing `RetrievalTracing` configuration.
- Produces: `snippet_span(document_id, focus_query)` with bounded `record_page(...)`; search spans containing document lengths rather than excerpts.

- [ ] **Step 1: Add focused trace redaction and validation tests**

Change search evidence from `excerpts` to `text_lengths`. Add one in-memory-exporter test that records a snippet page and asserts:

```python
assert span.attributes["snippet.document_id"] == "doc-a"
assert span.attributes["snippet.chunk_ids"] == ("doc-a:0007",)
assert span.attributes["snippet.start_chars"] == (8100,)
assert span.attributes["snippet.end_chars"] == (9310,)
assert span.attributes["snippet.relevance_scores"] == (0.91,)
assert span.attributes["snippet.cache_status"] == "hit"
assert span.attributes["snippet.ranker_backend"] == "sentence_transformers_cross_encoder"
assert span.attributes["snippet.has_next_page"] is True
```

Parameterize `trace_content=True/False` and assert snippet text is present only in content mode and otherwise equals `REDACTED_CONTENT`. Assert no cache path, cache key, cursor, scratch path, or complete document attribute is emitted.

- [ ] **Step 2: Run tracing and retrieval tests and confirm expected failures**

Run: `.venv/bin/python -m pytest code/tests/test_deepagent_tracing.py code/tests/test_deepagent_retrieval.py -q`

Expected: failures identify the new typed snippet span and revised search record shape.

- [ ] **Step 3: Implement bounded purpose-specific trace facades**

Update `_RetrieverSpan.record_search` to accept `text_lengths: Sequence[int]` and emit `retrieval.document_text_lengths`, never `retrieval.document_excerpts`. Add `_SnippetSpan.record_page` with typed arrays for at most ten chunk IDs, offsets, finite scores, bounded texts, cache status, backend, latency, page offset, and `has_next_page`. `RetrievalTracing.snippet_span` must use span name `deepagent.extract_relevant_snippets` and inherit existing narrative/query content redaction behavior.

Wrap each snippet extraction call in the new span, record only its validated page metadata and internally observed cache status/backend, and preserve current behavior that trace rejection/export failure cannot change retrieval results.

- [ ] **Step 4: Update the experimental SDK README**

Keep the existing direct-narrative example and add concise bullets explaining:

- search results visible to the agent are metadata only;
- the agent can request ten relevance-ranked snippets and follow `next_cursor`;
- both tools own their caches and expose no cache controls to the model;
- multiple snippets from one document are supported;
- oversized results and notes may use invocation-local state scratch that is discarded afterward;
- no title field is emitted because the endpoint does not return titles;
- Phoenix search spans contain document lengths, and snippet text follows `trace_content`.

- [ ] **Step 5: Run cheap source and focused checks**

```bash
git diff --check
rg -n 'excerpt|document_excerpts|FilesystemBackend|LocalShellBackend|cache_(key|path)' code/trec_rag/deepagent_retrieval.py code/trec_rag/deepagent_tracing.py code/trec_rag/README.md
.venv/bin/python -m pytest \
  code/tests/test_deepagent_snippets.py \
  code/tests/test_deepagent_tracing.py \
  code/tests/test_deepagent_retrieval.py \
  code/tests/test_chunking.py \
  code/tests/test_remote_pyserini.py -q
```

Expected: no forbidden model-visible excerpt/cache-path behavior and all focused tests pass.

- [ ] **Step 6: Run the repository suite**

Run: `.venv/bin/python -m pytest -q`

Expected: the full suite passes. Do not perform a live OpenRouter, Pyserini, Phoenix, or model-download run solely for this POC.

- [ ] **Step 7: Commit tracing and documentation**

```bash
git add \
  code/trec_rag/deepagent_tracing.py \
  code/trec_rag/deepagent_retrieval.py \
  code/tests/test_deepagent_tracing.py \
  code/tests/test_deepagent_retrieval.py \
  code/trec_rag/README.md
git commit -m "Trace and document Deep Agent snippet retrieval"
```
