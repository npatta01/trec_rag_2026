from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import pytest

import trec_rag.deepagent_snippets as deepagent_snippets
from trec_rag.chunking import ChunkingConfig, SemanticTextChunker, TextChunk
from trec_rag.deepagent_snippets import (
    DEFAULT_SNIPPET_MODEL,
    DEFAULT_SNIPPET_MODEL_REVISION,
    LocalMixedbreadSnippetRanker,
    RelevantSnippetExtractor,
    ScoredTextChunk,
    SmallLLMSnippetRanker,
    SnippetCacheIntegrityError,
    SnippetExtractionConfig,
    SnippetResultCache,
    create_default_snippet_extractor,
)
from trec_rag.rerank_score_cache import DEFAULT_BACKEND_VERSION


class FixedChunker:
    """A deterministic chunker that keeps extraction tests model-free."""

    def __init__(
        self, chunks: Sequence[TextChunk], *, implementation: str = "fixed-v1"
    ) -> None:
        self._chunks = list(chunks)
        self.identity = {
            "backend": "test_fixed_chunker",
            "implementation": implementation,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
                for chunk in chunks
            ],
        }

    def split_text(self, text: str, *, document_id: str) -> list[TextChunk]:
        assert document_id == "doc-a" or document_id == "doc-b"
        return [
            TextChunk(
                document_id=document_id,
                chunk_id=chunk.chunk_id.replace("doc-a", document_id, 1),
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
            for chunk in self._chunks
        ]


class CountingRanker:
    identity = {"backend": "test", "model": "keyword-v1", "version": 1}

    def __init__(self) -> None:
        self.calls = 0

    def rank(
        self, focus_query: str, chunks: Sequence[TextChunk]
    ) -> tuple[ScoredTextChunk, ...]:
        self.calls += 1
        words = set(focus_query.casefold().split())
        return tuple(
            ScoredTextChunk(
                chunk=chunk,
                relevance_score=float(
                    sum(word in chunk.text.casefold() for word in words)
                ),
            )
            for chunk in chunks
        )


def _chunks(*texts: str) -> tuple[TextChunk, ...]:
    offset = 0
    rows: list[TextChunk] = []
    for index, text in enumerate(texts):
        rows.append(
            TextChunk(
                document_id="doc-a",
                chunk_id=f"doc-a:{index:04d}",
                text=text,
                start_char=offset,
                end_char=offset + len(text),
            )
        )
        offset += len(text) + 1
    return tuple(rows)


LONG_CHUNKS = _chunks(
    *[f"background section {index}" for index in range(10)],
    "target passage near the end with primary evidence",
    "target passage final supporting evidence",
)
LONG_DOCUMENT = "\n".join(chunk.text for chunk in LONG_CHUNKS)


def _extractor(
    tmp_path: Path,
    chunks: Sequence[TextChunk] = LONG_CHUNKS,
    *,
    snippets_per_page: int = 10,
) -> tuple[RelevantSnippetExtractor, CountingRanker]:
    ranker = CountingRanker()
    return (
        RelevantSnippetExtractor(
            ranker=ranker,
            chunker=FixedChunker(chunks),
            result_cache=SnippetResultCache(tmp_path / "pages"),
            config=SnippetExtractionConfig(snippets_per_page=snippets_per_page),
        ),
        ranker,
    )


def test_relevant_passage_near_document_end_leads_first_ten_item_page(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)

    result = extractor.extract("doc-a", LONG_DOCUMENT, "target passage")

    assert len(result.page.snippets) == 10
    assert [snippet.chunk_id for snippet in result.page.snippets[:2]] == [
        "doc-a:0010",
        "doc-a:0011",
    ]
    assert result.page.snippets[0].text == "target passage near the end with primary evidence"
    assert result.page.next_cursor is not None
    assert result.page.as_dict().keys() == {
        "document_id",
        "focus_query",
        "snippets",
        "next_cursor",
    }
    assert "page_offset" not in result.page.as_dict()


def test_complete_short_document_is_one_bounded_relevant_snippet(tmp_path: Path) -> None:
    short_document = "A complete short document with relevant evidence."
    chunks = (
        TextChunk(
            document_id="doc-a",
            chunk_id="doc-a:0000",
            text=short_document,
            start_char=0,
            end_char=len(short_document),
        ),
    )
    extractor, _ranker = _extractor(tmp_path, chunks)

    result = extractor.extract("doc-a", short_document, "relevant evidence")

    assert len(short_document) < 3_500
    assert len(result.page.snippets) == 1
    assert result.page.snippets[0].text == short_document
    assert result.page.snippets[0].start_char == 0
    assert result.page.snippets[0].end_char == len(short_document)
    assert result.page.next_cursor is None


def test_continuation_is_stable_and_has_no_duplicate_snippets(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)

    first = extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    second = extractor.extract("doc-a", LONG_DOCUMENT, "target passage", first.page.next_cursor)

    assert second.page_offset == 10
    assert len(second.page.snippets) == 2
    assert second.page.next_cursor is None
    assert {snippet.chunk_id for snippet in first.page.snippets}.isdisjoint(
        snippet.chunk_id for snippet in second.page.snippets
    )


def test_deduplication_suppresses_normalized_and_overlapping_chunks(tmp_path: Path) -> None:
    document = "target passage overlap target passage elsewhere"
    later_start = document.rindex("target passage")
    chunks = (
        TextChunk("doc-a", "doc-a:0000", document[0:14], 0, 14),
        TextChunk("doc-a", "doc-a:0001", document[0:14], 0, 14),
        TextChunk("doc-a", "doc-a:0002", document[2:17], 2, 17),
        TextChunk(
            "doc-a",
            "doc-a:0003",
            document[later_start:],
            later_start,
            len(document),
        ),
    )
    extractor, _ranker = _extractor(tmp_path, chunks, snippets_per_page=10)

    result = extractor.extract("doc-a", document, "target passage")

    assert [snippet.chunk_id for snippet in result.page.snippets] == [
        "doc-a:0000",
        "doc-a:0003",
    ]


def test_extract_rejects_blank_query_and_invalid_cursor(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)

    with pytest.raises(ValueError, match="focus_query"):
        extractor.extract("doc-a", LONG_DOCUMENT, " \n ")
    with pytest.raises(ValueError, match="cursor"):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage", "not-a-cursor")
    with pytest.raises(ValueError, match="^invalid cursor$"):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage", "snowman-☃")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("snippets_per_page", True),
        ("snippets_per_page", 1.5),
        ("snippets_per_page", "10"),
        ("chunk_max_characters", True),
        ("chunk_max_characters", 1.5),
        ("chunk_max_characters", "3500"),
        ("chunk_overlap_characters", True),
        ("chunk_overlap_characters", 1.5),
        ("chunk_overlap_characters", "350"),
    ],
)
def test_config_rejects_non_integer_page_and_chunk_counts(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        SnippetExtractionConfig(**{field: value})


def test_config_and_ranker_scores_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="snippets_per_page"):
        SnippetExtractionConfig(snippets_per_page=0)
    with pytest.raises(ValueError, match="chunk_overlap_characters"):
        SnippetExtractionConfig(chunk_max_characters=10, chunk_overlap_characters=10)
    with pytest.raises(ValueError, match="duplicate_overlap_ratio"):
        SnippetExtractionConfig(duplicate_overlap_ratio=1.1)

    class NonFiniteRanker(CountingRanker):
        def rank(self, focus_query: str, chunks: Sequence[TextChunk]) -> tuple[ScoredTextChunk, ...]:
            return (ScoredTextChunk(chunks[0], float("nan")),)

    extractor = RelevantSnippetExtractor(
        ranker=NonFiniteRanker(),
        chunker=FixedChunker(LONG_CHUNKS),
        result_cache=SnippetResultCache(tmp_path / "pages"),
    )
    with pytest.raises(ValueError, match="finite"):
        extractor.extract("doc-a", LONG_DOCUMENT, "target")


def test_cache_first_pages_use_every_tool_argument(tmp_path: Path) -> None:
    extractor, ranker = _extractor(tmp_path)

    first = extractor.extract("doc-a", LONG_DOCUMENT, "target passage", cursor=None)
    repeated = extractor.extract("doc-a", LONG_DOCUMENT, "target passage", cursor=None)

    assert repeated.page == first.page
    assert repeated.cache_status == "hit"
    assert ranker.calls == 1
    assert extractor.extract("doc-a", LONG_DOCUMENT, "different query", None).cache_status == "miss"
    assert extractor.extract("doc-b", LONG_DOCUMENT, "target passage", None).cache_status == "miss"
    assert first.page.next_cursor is not None
    assert (
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage", first.page.next_cursor).cache_status
        == "miss"
    )
    assert extractor.extract("doc-a", LONG_DOCUMENT + " revised", "target passage", None).cache_status == "miss"


def test_cache_identity_distinguishes_injected_chunker_implementations(tmp_path: Path) -> None:
    cache = SnippetResultCache(tmp_path / "pages")
    first_ranker = CountingRanker()
    first = RelevantSnippetExtractor(
        ranker=first_ranker,
        chunker=FixedChunker(LONG_CHUNKS, implementation="split-policy-a"),
        result_cache=cache,
    )
    second_ranker = CountingRanker()
    second = RelevantSnippetExtractor(
        ranker=second_ranker,
        chunker=FixedChunker(LONG_CHUNKS, implementation="split-policy-b"),
        result_cache=cache,
    )

    assert first.extract("doc-a", LONG_DOCUMENT, "target passage").cache_status == "miss"
    assert second.extract("doc-a", LONG_DOCUMENT, "target passage").cache_status == "miss"
    assert first_ranker.calls == 1
    assert second_ranker.calls == 1


def test_injected_chunker_requires_a_stable_serializable_identity(tmp_path: Path) -> None:
    class UnidentifiedChunker:
        def split_text(self, text: str, *, document_id: str) -> list[TextChunk]:
            return []

    with pytest.raises(ValueError, match="chunker identity"):
        RelevantSnippetExtractor(
            ranker=CountingRanker(),
            chunker=UnidentifiedChunker(),
            result_cache=SnippetResultCache(tmp_path / "pages"),
        )


def test_semantic_chunker_subclass_requires_its_own_identity(tmp_path: Path) -> None:
    class CustomSemanticChunker(SemanticTextChunker):
        pass

    with pytest.raises(ValueError, match="chunker identity"):
        RelevantSnippetExtractor(
            ranker=CountingRanker(),
            chunker=CustomSemanticChunker(),
            result_cache=SnippetResultCache(tmp_path / "pages"),
        )


def _rewrite_cache_response(cache_file: Path, *, next_cursor: object) -> None:
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["response"]["next_cursor"] = next_cursor
    encoded_response = json.dumps(
        payload["response"],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload["response_sha256"] = sha256(encoded_response).hexdigest()
    cache_file.write_text(json.dumps(payload), encoding="utf-8")


def test_cache_rejects_semantically_invalid_next_cursor(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)
    extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    cache_file = next((tmp_path / "pages").glob("schema_v1/*/*.json"))
    _rewrite_cache_response(cache_file, next_cursor="snowman-☃")

    with pytest.raises(SnippetCacheIntegrityError, match="^invalid snippet cache entry$"):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")


def test_cache_rejects_next_cursor_that_disagrees_with_page_offset(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)
    extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    cache_file = next((tmp_path / "pages").glob("schema_v1/*/*.json"))
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["page_offset"] = 1
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SnippetCacheIntegrityError, match="^invalid snippet cache entry$"):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")


def test_cache_rejects_malformed_entries_without_returning_them(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)
    extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    cache_file = next((tmp_path / "pages").glob("schema_v1/*/*.json"))
    cache_file.write_text(json.dumps({"broken": True}), encoding="utf-8")

    with pytest.raises(SnippetCacheIntegrityError):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")


class FakeCrossEncoder:
    """Offline cross-encoder fake returning one configured score batch per call."""

    def __init__(self, score_batches: Sequence[Sequence[float]]) -> None:
        self._score_batches = [list(batch) for batch in score_batches]
        self.predict_calls = 0
        self.pairs: list[list[tuple[str, str]]] = []
        self.predict_kwargs: list[dict[str, object]] = []

    def predict(self, pairs: Sequence[tuple[str, str]], **kwargs: object) -> list[float]:
        self.predict_calls += 1
        self.pairs.append(list(pairs))
        self.predict_kwargs.append(dict(kwargs))
        return self._score_batches.pop(0)


@dataclass
class FakeChatReply:
    content: str


class FakeChatModel:
    """Offline chat fake that records bounded JSON prompts."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeChatReply:
        self.prompts.append(prompt)
        return FakeChatReply(self._responses.pop(0))


ADAPTER_CHUNKS = _chunks("first evidence", "second evidence")


def test_local_ranker_scores_only_cache_misses_and_loads_lazily(tmp_path: Path) -> None:
    model = FakeCrossEncoder([[0.8, 0.2]])
    loader_calls: list[dict[str, object]] = []

    def load_model(**kwargs: object) -> FakeCrossEncoder:
        loader_calls.append(dict(kwargs))
        return model

    ranker = LocalMixedbreadSnippetRanker(
        score_cache_root=tmp_path,
        model_loader=load_model,
        device="cpu",
    )

    assert loader_calls == []
    assert [row.relevance_score for row in ranker.rank("query", ADAPTER_CHUNKS)] == [0.8, 0.2]
    assert [row.relevance_score for row in ranker.rank("query", ADAPTER_CHUNKS)] == [0.8, 0.2]
    assert model.predict_calls == 1
    assert loader_calls == [
        {
            "model_name": DEFAULT_SNIPPET_MODEL,
            "revision": DEFAULT_SNIPPET_MODEL_REVISION,
            "max_length": 512,
            "device": "cpu",
        }
    ]
    assert model.pairs == [[("query", "first evidence"), ("query", "second evidence")]]
    assert model.predict_kwargs[0] | {"activation_fn": None} == {
        "batch_size": 32,
        "show_progress_bar": False,
        "convert_to_tensor": True,
        "activation_fn": None,
    }
    assert model.predict_kwargs[0]["activation_fn"](3.25) == 3.25


def test_local_ranker_identity_and_partial_cache_miss_are_exact(tmp_path: Path) -> None:
    model = FakeCrossEncoder([[0.8], [0.2]])
    ranker = LocalMixedbreadSnippetRanker(
        score_cache_root=tmp_path,
        model_loader=lambda **_kwargs: model,
        device="cpu",
    )

    assert ranker.identity == {
        "backend": "sentence_transformers_cross_encoder",
        "backend_version": DEFAULT_BACKEND_VERSION,
        "model": DEFAULT_SNIPPET_MODEL,
        "model_revision": DEFAULT_SNIPPET_MODEL_REVISION,
        "score_representation": "raw_logits",
        "max_length": 512,
        "batch_size": 32,
        "device": "cpu",
        "implementation_version": 1,
    }
    assert [row.relevance_score for row in ranker.rank("query", ADAPTER_CHUNKS[:1])] == [0.8]
    assert [row.relevance_score for row in ranker.rank("query", ADAPTER_CHUNKS)] == [0.8, 0.2]
    assert model.pairs == [[("query", "first evidence")], [("query", "second evidence")]]


@pytest.mark.parametrize(
    ("first_settings", "second_settings"),
    [
        ({"batch_size": 1, "device": "cpu"}, {"batch_size": 2, "device": "cpu"}),
        ({"batch_size": 1, "device": "cpu"}, {"batch_size": 1, "device": "cuda"}),
    ],
)
def test_local_score_cache_partitions_effective_adapter_settings(
    tmp_path: Path,
    first_settings: dict[str, object],
    second_settings: dict[str, object],
) -> None:
    first_model = FakeCrossEncoder([[0.8]])
    first = LocalMixedbreadSnippetRanker(
        score_cache_root=tmp_path,
        model_loader=lambda **_kwargs: first_model,
        **first_settings,
    )
    assert first.rank("query", ADAPTER_CHUNKS[:1])[0].relevance_score == 0.8

    second_model = FakeCrossEncoder([[0.2]])
    second = LocalMixedbreadSnippetRanker(
        score_cache_root=tmp_path,
        model_loader=lambda **_kwargs: second_model,
        **second_settings,
    )

    assert second.rank("query", ADAPTER_CHUNKS[:1])[0].relevance_score == 0.2
    assert second_model.predict_calls == 1


def test_local_ranker_score_cache_partitions_resolved_auto_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_device = "cpu"
    monkeypatch.setattr(
        deepagent_snippets,
        "_choose_device",
        lambda _requested: resolved_device,
    )
    first_model = FakeCrossEncoder([[0.8]])
    first = LocalMixedbreadSnippetRanker(
        score_cache_root=tmp_path,
        model_loader=lambda **_kwargs: first_model,
        device="auto",
    )
    assert first.rank("query", ADAPTER_CHUNKS[:1])[0].relevance_score == 0.8

    resolved_device = "cuda"
    second_model = FakeCrossEncoder([[0.2]])
    second = LocalMixedbreadSnippetRanker(
        score_cache_root=tmp_path,
        model_loader=lambda **_kwargs: second_model,
        device="auto",
    )

    assert second.identity["device"] == "cuda"
    assert second.rank("query", ADAPTER_CHUNKS[:1])[0].relevance_score == 0.2
    assert second_model.predict_calls == 1


def test_small_llm_ranker_caches_validated_scores_and_batches_prompts(tmp_path: Path) -> None:
    chat = FakeChatModel(['{"scores":[{"chunk_id":"doc-a:0000","score":0.2},{"chunk_id":"doc-a:0001","score":0.8}]}'])
    ranker = SmallLLMSnippetRanker(
        chat_model=chat,
        score_cache_root=tmp_path,
        model_name="test-small-llm",
        model_revision="test-revision",
        backend_version="test-backend",
        batch_size=2,
    )

    assert ranker.identity == {
        "backend": "small_llm_json",
        "backend_version": "test-backend",
        "model": "test-small-llm",
        "model_revision": "test-revision",
        "score_representation": "json_scalar",
        "batch_size": 2,
        "implementation_version": 1,
    }
    assert [row.chunk.chunk_id for row in ranker.rank("query", ADAPTER_CHUNKS)] == [
        "doc-a:0001",
        "doc-a:0000",
    ]
    assert [row.relevance_score for row in ranker.rank("query", ADAPTER_CHUNKS)] == [0.8, 0.2]
    assert chat.prompts[0].startswith(
        "Score each chunk's relevance to the focus query. Return only strict JSON "
        'matching {"scores":[{"chunk_id":"...","score":0.0}]}.\n'
    )
    assert [json.loads(prompt.rsplit("\n", 1)[1]) for prompt in chat.prompts] == [
        {
            "focus_query": "query",
            "chunks": [
                {"chunk_id": "doc-a:0000", "text": "first evidence"},
                {"chunk_id": "doc-a:0001", "text": "second evidence"},
            ],
        }
    ]


def test_small_llm_ranker_sends_every_uncached_duplicate_text_chunk_id(
    tmp_path: Path,
) -> None:
    duplicate_text_chunks = _chunks("same evidence", "same evidence")
    chat = FakeChatModel(
        [
            '{"scores":['
            '{"chunk_id":"doc-a:0000","score":0.6},'
            '{"chunk_id":"doc-a:0001","score":0.6}'
            "]}"
        ]
    )
    ranker = SmallLLMSnippetRanker(
        chat_model=chat,
        score_cache_root=tmp_path,
        model_name="test-small-llm",
    )

    assert [row.chunk.chunk_id for row in ranker.rank("query", duplicate_text_chunks)] == [
        "doc-a:0000",
        "doc-a:0001",
    ]
    prompt_data = json.loads(chat.prompts[0].rsplit("\n", 1)[1])
    assert [chunk["chunk_id"] for chunk in prompt_data["chunks"]] == [
        "doc-a:0000",
        "doc-a:0001",
    ]


def test_small_llm_ranker_rejects_conflicting_duplicate_text_scores(tmp_path: Path) -> None:
    duplicate_text_chunks = _chunks("same evidence", "same evidence")
    chat = FakeChatModel(
        [
            '{"scores":['
            '{"chunk_id":"doc-a:0000","score":0.6},'
            '{"chunk_id":"doc-a:0001","score":0.4}'
            "]}"
        ]
    )
    ranker = SmallLLMSnippetRanker(
        chat_model=chat,
        score_cache_root=tmp_path,
        model_name="test-small-llm",
    )

    with pytest.raises(ValueError, match="conflicting scores for identical chunk text"):
        ranker.rank("query", duplicate_text_chunks)


def test_small_llm_ranker_does_not_cache_a_cross_batch_duplicate_conflict(
    tmp_path: Path,
) -> None:
    duplicate_text_chunks = _chunks("same evidence", "same evidence")
    conflicting = SmallLLMSnippetRanker(
        chat_model=FakeChatModel(
            [
                '{"scores":[{"chunk_id":"doc-a:0000","score":0.6}]}',
                '{"scores":[{"chunk_id":"doc-a:0001","score":0.4}]}',
            ]
        ),
        score_cache_root=tmp_path,
        model_name="test-small-llm",
        batch_size=1,
    )
    with pytest.raises(ValueError, match="conflicting scores for identical chunk text"):
        conflicting.rank("query", duplicate_text_chunks)

    retry_chat = FakeChatModel(
        [
            '{"scores":[{"chunk_id":"doc-a:0000","score":0.2}]}',
            '{"scores":[{"chunk_id":"doc-a:0001","score":0.2}]}',
        ]
    )
    retry = SmallLLMSnippetRanker(
        chat_model=retry_chat,
        score_cache_root=tmp_path,
        model_name="test-small-llm",
        batch_size=1,
    )

    assert [row.relevance_score for row in retry.rank("query", duplicate_text_chunks)] == [
        0.2,
        0.2,
    ]
    assert len(retry_chat.prompts) == 2


def test_small_llm_ranker_score_cache_partitions_batch_size(tmp_path: Path) -> None:
    first = SmallLLMSnippetRanker(
        chat_model=FakeChatModel(
            ['{"scores":[{"chunk_id":"doc-a:0000","score":0.8}]}']
        ),
        score_cache_root=tmp_path,
        model_name="test-small-llm",
        batch_size=1,
    )
    assert first.rank("query", ADAPTER_CHUNKS[:1])[0].relevance_score == 0.8

    second_chat = FakeChatModel(
        ['{"scores":[{"chunk_id":"doc-a:0000","score":0.2}]}']
    )
    second = SmallLLMSnippetRanker(
        chat_model=second_chat,
        score_cache_root=tmp_path,
        model_name="test-small-llm",
        batch_size=2,
    )

    assert second.rank("query", ADAPTER_CHUNKS[:1])[0].relevance_score == 0.2
    assert len(second_chat.prompts) == 1


@pytest.mark.parametrize(
    "response",
    [
        '{"scores":[{"chunk_id":"doc-a:0000","score":0.8}]}',
        '{"scores":[{"chunk_id":"doc-a:0000","score":0.8},{"chunk_id":"doc-a:0000","score":0.2}]}',
        '{"scores":[{"chunk_id":"doc-a:0000","score":0.8},{"chunk_id":"unknown","score":0.2}]}',
        '{"scores":[{"chunk_id":"doc-a:0000","score":true},{"chunk_id":"doc-a:0001","score":0.2}]}',
        '{"scores":[{"chunk_id":"doc-a:0000","score":NaN},{"chunk_id":"doc-a:0001","score":0.2}]}',
        '{"scores":[],"scores":[{"chunk_id":"doc-a:0000","score":0.8},{"chunk_id":"doc-a:0001","score":0.2}]}',
    ],
)
def test_small_llm_ranker_rejects_invalid_score_sets(tmp_path: Path, response: str) -> None:
    ranker = SmallLLMSnippetRanker(
        chat_model=FakeChatModel([response]),
        score_cache_root=tmp_path,
        model_name="test-small-llm",
    )

    with pytest.raises(ValueError, match="small-LLM score response"):
        ranker.rank("query", ADAPTER_CHUNKS)


def test_default_extractor_constructs_lazy_local_ranker(tmp_path: Path) -> None:
    extractor = create_default_snippet_extractor(tmp_path)

    assert isinstance(extractor._ranker, LocalMixedbreadSnippetRanker)
    assert isinstance(extractor._chunker, SemanticTextChunker)
    assert extractor._chunker.config == ChunkingConfig(max_characters=3500, overlap_characters=350)
    assert extractor._result_cache.root_dir == tmp_path / "cache" / "reranker" / "deepagent_snippets"
