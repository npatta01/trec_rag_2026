from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from trec_rag.chunking import TextChunk
from trec_rag.deepagent_snippets import (
    RelevantSnippetExtractor,
    ScoredTextChunk,
    SnippetCacheIntegrityError,
    SnippetExtractionConfig,
    SnippetResultCache,
)


class FixedChunker:
    """A deterministic chunker that keeps extraction tests model-free."""

    def __init__(self, chunks: Sequence[TextChunk]) -> None:
        self._chunks = list(chunks)

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


def test_cache_rejects_malformed_entries_without_returning_them(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)
    extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    cache_file = next((tmp_path / "pages").glob("schema_v1/*/*.json"))
    cache_file.write_text(json.dumps({"broken": True}), encoding="utf-8")

    with pytest.raises(SnippetCacheIntegrityError):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
