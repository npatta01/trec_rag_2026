from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
from queue import Empty
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Lock
from typing import Any

import pytest
from filelock import FileLock

import trec_rag.deepagent_snippets as deepagent_snippets
from trec_rag.chunking import ChunkingConfig, SemanticTextChunker, TextChunk
from trec_rag.deepagent_snippets import (
    DEFAULT_SNIPPET_MODEL,
    DEFAULT_SNIPPET_MODEL_REVISION,
    InvalidSnippetCursorError,
    LocalMixedbreadSnippetRanker,
    RelevantSnippet,
    RelevantSnippetExtractor,
    ScoredTextChunk,
    SmallLLMSnippetRanker,
    SnippetCacheIntegrityError,
    SnippetExtractionConfig,
    SnippetPage,
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


class _ProcessRecordingRanker(CountingRanker):
    def __init__(self, calls: Any, rendezvous: Any) -> None:
        super().__init__()
        self._calls = calls
        self._rendezvous = rendezvous

    def rank(
        self, focus_query: str, chunks: Sequence[TextChunk]
    ) -> tuple[ScoredTextChunk, ...]:
        with self._calls.get_lock():
            self._calls.value += 1
        try:
            self._rendezvous.wait()
        except BrokenBarrierError:
            pass
        return super().rank(focus_query, chunks)


class _ProcessCrossEncoder:
    def __init__(self, score: float, calls: Any, rendezvous: Any) -> None:
        self._score = score
        self._calls = calls
        self._rendezvous = rendezvous

    def predict(
        self, pairs: Sequence[tuple[str, str]], **_kwargs: object
    ) -> list[float]:
        with self._calls.get_lock():
            self._calls.value += 1
        try:
            self._rendezvous.wait()
        except BrokenBarrierError:
            pass
        return [self._score for _pair in pairs]


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


def _extract_in_process(
    cache_root: str,
    start_barrier: Any,
    rank_barrier: Any,
    rank_calls: Any,
    results: Any,
) -> None:
    extractor = RelevantSnippetExtractor(
        ranker=_ProcessRecordingRanker(rank_calls, rank_barrier),
        chunker=FixedChunker(LONG_CHUNKS),
        result_cache=SnippetResultCache(Path(cache_root)),
    )
    try:
        start_barrier.wait()
        result = extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    except Exception as exc:  # pragma: no cover - reported to the parent process
        results.put(("error", type(exc).__name__, str(exc)))
    else:
        results.put(("ok", result.cache_status))


def _rank_local_in_process(
    cache_root: str,
    score: float,
    start_barrier: Any,
    predict_barrier: Any,
    predict_calls: Any,
    results: Any,
) -> None:
    model = _ProcessCrossEncoder(score, predict_calls, predict_barrier)
    ranker = LocalMixedbreadSnippetRanker(
        score_cache_root=Path(cache_root),
        model_loader=lambda **_kwargs: model,
        device="cpu",
    )
    try:
        start_barrier.wait()
        ranked = ranker.rank("query", ADAPTER_CHUNKS[:1])
    except Exception as exc:  # pragma: no cover - reported to the parent process
        results.put(("error", type(exc).__name__, str(exc)))
    else:
        results.put(("ok", ranked[0].relevance_score))


def _load_cached_score_in_process(
    cache_root: str,
    about_to_construct: Any,
    results: Any,
) -> None:
    about_to_construct.put(True)

    def fail_loader(**_kwargs: object) -> object:
        raise AssertionError("cached score unexpectedly invoked the model loader")

    try:
        ranker = LocalMixedbreadSnippetRanker(
            score_cache_root=Path(cache_root),
            model_loader=fail_loader,
            device="cpu",
        )
        score = ranker.rank("query", ADAPTER_CHUNKS[:1])[0].relevance_score
    except Exception as exc:  # pragma: no cover - reported to the parent process
        results.put(("error", type(exc).__name__, str(exc)))
    else:
        results.put(("ok", score))


def _thread_result_cache_probe(cache_root: str, results: Any) -> None:
    calls: list[str] = []
    calls_lock = Lock()
    rank_barrier = Barrier(2, timeout=0.5)

    class RendezvousRanker(CountingRanker):
        def rank(
            self, focus_query: str, chunks: Sequence[TextChunk]
        ) -> tuple[ScoredTextChunk, ...]:
            with calls_lock:
                calls.append(focus_query)
            try:
                rank_barrier.wait()
            except BrokenBarrierError:
                pass
            return super().rank(focus_query, chunks)

    extractors = tuple(
        RelevantSnippetExtractor(
            ranker=RendezvousRanker(),
            chunker=FixedChunker(LONG_CHUNKS),
            result_cache=SnippetResultCache(Path(cache_root)),
        )
        for _ in range(2)
    )
    start_barrier = Barrier(3, timeout=5)
    executor = ThreadPoolExecutor(max_workers=2)

    def extract(extractor: RelevantSnippetExtractor) -> str:
        start_barrier.wait()
        return extractor.extract(
            "doc-a", LONG_DOCUMENT, "target passage"
        ).cache_status

    try:
        futures = [executor.submit(extract, item) for item in extractors]
        start_barrier.wait()
        statuses = [future.result(timeout=5) for future in futures]
        executor.shutdown(wait=True)
    except Exception as exc:  # pragma: no cover - reported to the parent process
        executor.shutdown(wait=False, cancel_futures=True)
        results.put(("error", type(exc).__name__, str(exc)))
    else:
        results.put(("ok", sorted(statuses), calls))


def _stop_processes(processes: Sequence[Any]) -> None:
    started = [process for process in processes if process.pid is not None]
    for process in started:
        process.join(timeout=0.1)
    for process in started:
        if process.is_alive():
            process.terminate()
    for process in started:
        process.join(timeout=2)
    for process in started:
        if process.is_alive():
            process.kill()
            process.join(timeout=2)


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


def _cache_file(root: Path) -> Path:
    schema_dir = root / f"schema_v{deepagent_snippets.RESULT_SCHEMA_VERSION}"
    return next(schema_dir.glob("*/*.json"))


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
    assert result.page.page_index == 0
    assert result.page.residual_count == 2
    assert result.page.residual_top_score == 0.0
    assert result.page.returned_min_score == 0.0
    assert result.page.pages_estimated == 2
    assert result.page.as_dict().keys() == {
        "document_id",
        "focus_query",
        "snippets",
        "next_cursor",
        "page_index",
        "residual_count",
        "residual_top_score",
        "returned_min_score",
        "pages_estimated",
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
    assert result.page.page_index == 0
    assert result.page.residual_count == 0
    assert result.page.residual_top_score is None
    assert result.page.returned_min_score == 2.0
    assert result.page.pages_estimated == 1


def test_continuation_reports_exhausted_residual_ranking(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)

    first = extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    second = extractor.extract("doc-a", LONG_DOCUMENT, "target passage", first.page.next_cursor)

    assert second.page_offset == 10
    assert len(second.page.snippets) == 2
    assert second.page.next_cursor is None
    assert second.page.page_index == 1
    assert second.page.residual_count == 0
    assert second.page.residual_top_score is None
    assert second.page.returned_min_score == 0.0
    assert second.page.pages_estimated == 2
    assert {snippet.chunk_id for snippet in first.page.snippets}.isdisjoint(
        snippet.chunk_id for snippet in second.page.snippets
    )


def test_empty_page_reports_no_pagination_or_score_signals(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path, ())

    result = extractor.extract("doc-a", "", "target passage")

    assert result.page.page_index == 0
    assert result.page.residual_count == 0
    assert result.page.residual_top_score is None
    assert result.page.returned_min_score is None
    assert result.page.pages_estimated == 0


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
    with pytest.raises(InvalidSnippetCursorError, match="cursor"):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage", "not-a-cursor")
    with pytest.raises(InvalidSnippetCursorError, match="^invalid cursor$"):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage", "snowman-☃")


def test_invalid_cursor_is_classified_before_any_cache_entry(
    tmp_path: Path,
) -> None:
    extractor, _ranker = _extractor(tmp_path)
    cursor = "not-a-cursor"
    identity = extractor._result_identity(
        "doc-a", LONG_DOCUMENT, "target passage", cursor
    )
    cache_file = extractor._result_cache._path(identity)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text('{"broken":true}', encoding="utf-8")

    with pytest.raises(InvalidSnippetCursorError, match="^invalid cursor$"):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage", cursor)


def test_cursor_offset_edits_are_rejected_but_identical_instances_can_resume(
    tmp_path: Path,
) -> None:
    first_extractor, _ranker = _extractor(tmp_path / "first")
    first = first_extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    assert first.page.next_cursor is not None
    encoded = first.page.next_cursor
    decoded = json.loads(
        base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")
    )
    decoded["next_offset"] += 1
    edited = base64.urlsafe_b64encode(
        json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")

    with pytest.raises(InvalidSnippetCursorError, match="^invalid cursor$"):
        first_extractor.extract("doc-a", LONG_DOCUMENT, "target passage", edited)

    second_extractor, _ranker = _extractor(tmp_path / "second")
    resumed = second_extractor.extract(
        "doc-a", LONG_DOCUMENT, "target passage", first.page.next_cursor
    )
    assert resumed.page_offset == 10
    assert "doc-a" not in encoded
    assert "target passage" not in encoded
    assert str(tmp_path) not in encoded


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
            return tuple(
                ScoredTextChunk(chunk, float("nan") if index == 0 else 0.0)
                for index, chunk in enumerate(chunks)
            )

    extractor = RelevantSnippetExtractor(
        ranker=NonFiniteRanker(),
        chunker=FixedChunker(LONG_CHUNKS),
        result_cache=SnippetResultCache(tmp_path / "pages"),
    )
    with pytest.raises(ValueError, match="finite"):
        extractor.extract("doc-a", LONG_DOCUMENT, "target")


def test_live_chunker_rejects_oversized_complete_document_but_allows_bounded_short_one(
    tmp_path: Path,
) -> None:
    maximum = 20
    config = SnippetExtractionConfig(
        chunk_max_characters=maximum,
        chunk_overlap_characters=0,
    )
    long_document = "L" * (maximum + 1)
    long_chunk = TextChunk(
        "doc-a", "doc-a:0000", long_document, 0, len(long_document)
    )
    long_extractor = RelevantSnippetExtractor(
        ranker=CountingRanker(),
        chunker=FixedChunker((long_chunk,)),
        result_cache=SnippetResultCache(tmp_path / "long-pages"),
        config=config,
    )

    with pytest.raises(ValueError, match="chunker returned an invalid text chunk"):
        long_extractor.extract("doc-a", long_document, "focus")

    short_document = "S" * maximum
    short_chunk = TextChunk(
        "doc-a", "doc-a:0000", short_document, 0, len(short_document)
    )
    short_extractor = RelevantSnippetExtractor(
        ranker=CountingRanker(),
        chunker=FixedChunker((short_chunk,)),
        result_cache=SnippetResultCache(tmp_path / "short-pages"),
        config=config,
    )
    short_page = short_extractor.extract("doc-a", short_document, "focus").page

    assert [snippet.text for snippet in short_page.snippets] == [short_document]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "foreign"])
def test_ranker_must_score_every_input_chunk_exactly_once(
    tmp_path: Path, failure: str
) -> None:
    chunks = _chunks("first", "second")
    document = "\n".join(chunk.text for chunk in chunks)

    class InvalidCoverageRanker(CountingRanker):
        def rank(
            self, focus_query: str, rows: Sequence[TextChunk]
        ) -> tuple[ScoredTextChunk, ...]:
            first = ScoredTextChunk(rows[0], 1.0)
            if failure == "missing":
                return (first,)
            if failure == "foreign":
                foreign = TextChunk(
                    "doc-a", "doc-a:9999", "foreign", 0, len("foreign")
                )
                return (first, ScoredTextChunk(foreign, 0.5))
            return (first, first)

    extractor = RelevantSnippetExtractor(
        ranker=InvalidCoverageRanker(),
        chunker=FixedChunker(chunks),
        result_cache=SnippetResultCache(tmp_path / failure),
    )

    with pytest.raises(
        ValueError, match="exactly one score per input chunk|unknown text chunk"
    ):
        extractor.extract("doc-a", document, "focus")


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


def test_cache_hit_validates_stored_chunk_manifest_without_rechunking(
    tmp_path: Path,
) -> None:
    class CountingChunker(FixedChunker):
        def __init__(self, chunks: Sequence[TextChunk]) -> None:
            super().__init__(chunks)
            self.calls = 0

        def split_text(self, text: str, *, document_id: str) -> list[TextChunk]:
            self.calls += 1
            return super().split_text(text, document_id=document_id)

    chunker = CountingChunker(LONG_CHUNKS)
    extractor = RelevantSnippetExtractor(
        ranker=CountingRanker(),
        chunker=chunker,
        result_cache=SnippetResultCache(tmp_path / "pages"),
    )

    assert extractor.extract("doc-a", LONG_DOCUMENT, "target passage").cache_status == "miss"
    assert extractor.extract("doc-a", LONG_DOCUMENT, "target passage").cache_status == "hit"
    assert chunker.calls == 1


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


def _rewrite_checksum_consistent_cache_response(
    cache_file: Path, mutate: Callable[[dict[str, object]], None]
) -> None:
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    mutate(payload["response"])
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
    cache_file = _cache_file(tmp_path / "pages")
    _rewrite_cache_response(cache_file, next_cursor="snowman-☃")

    with pytest.raises(SnippetCacheIntegrityError, match="^invalid snippet cache entry$"):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")


def test_cache_rejects_next_cursor_that_disagrees_with_page_offset(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)
    extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    cache_file = _cache_file(tmp_path / "pages")
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["page_offset"] = 1
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SnippetCacheIntegrityError, match="^invalid snippet cache entry$"):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")


def test_cache_rejects_malformed_entries_without_returning_them(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)
    extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    cache_file = _cache_file(tmp_path / "pages")
    cache_file.write_text(json.dumps({"broken": True}), encoding="utf-8")

    with pytest.raises(SnippetCacheIntegrityError):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("page_index", -1),
        ("residual_count", True),
        ("residual_top_score", float("nan")),
        ("returned_min_score", "0.5"),
        ("pages_estimated", -1),
    ],
)
def test_cache_rejects_invalid_pagination_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    extractor, _ranker = _extractor(tmp_path)
    extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    cache_file = _cache_file(tmp_path / "pages")
    payload = json.loads(cache_file.read_text())
    payload["response"][field] = value
    try:
        response = deepagent_snippets._canonical_json(payload["response"])
    except ValueError:
        response = json.dumps(
            payload["response"],
            allow_nan=True,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    payload["response_sha256"] = sha256(response.encode()).hexdigest()
    cache_file.write_text(json.dumps(payload))

    with pytest.raises(SnippetCacheIntegrityError):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("page_index", 1),
        ("next_cursor", None),
        ("returned_min_score", 1.0),
        ("pages_estimated", 1),
    ],
)
def test_cache_rejects_inconsistent_pagination_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    extractor, _ranker = _extractor(tmp_path)
    extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    cache_file = _cache_file(tmp_path / "pages")
    _rewrite_checksum_consistent_cache_response(
        cache_file, lambda response: response.__setitem__(field, value)
    )

    with pytest.raises(SnippetCacheIntegrityError):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")


@pytest.mark.parametrize(
    "mutation",
    ["page_size", "duplicate", "offset", "text", "document_chunk_identity"],
)
def test_cache_rejects_checksum_consistent_semantically_invalid_pages(
    tmp_path: Path, mutation: str
) -> None:
    extractor, _ranker = _extractor(tmp_path)
    extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    cache_file = _cache_file(tmp_path / "pages")

    def mutate(response: dict[str, object]) -> None:
        snippets = response["snippets"]
        assert isinstance(snippets, list)
        if mutation == "page_size":
            source = LONG_CHUNKS[8]
            snippets.append(
                {
                    "chunk_id": source.chunk_id,
                    "start_char": source.start_char,
                    "end_char": source.end_char,
                    "text": source.text,
                    "relevance_score": 0.0,
                }
            )
            response["next_cursor"] = None
        elif mutation == "duplicate":
            snippets[1] = dict(snippets[0])
        elif mutation == "offset":
            snippets[0]["start_char"] += 1
        elif mutation == "text":
            snippets[0]["text"] += "!"
        else:
            snippets[0]["chunk_id"] = "doc-a:9999"

    _rewrite_checksum_consistent_cache_response(cache_file, mutate)

    with pytest.raises(
        SnippetCacheIntegrityError, match="^invalid snippet cache entry$"
    ):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")


def test_cache_rejects_checksum_consistent_duplicate_source_chunk_identity(
    tmp_path: Path,
) -> None:
    extractor, _ranker = _extractor(tmp_path)
    extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    cache_file = _cache_file(tmp_path / "pages")
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    source_chunks = payload["source_chunks"]
    assert isinstance(source_chunks, list)
    duplicate_id = dict(source_chunks[1])
    duplicate_id["chunk_id"] = source_chunks[0]["chunk_id"]
    source_chunks[1] = duplicate_id
    payload["source_chunks_sha256"] = sha256(
        json.dumps(
            source_chunks,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        SnippetCacheIntegrityError, match="^invalid snippet cache entry$"
    ):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")


def test_cache_rejects_nonprogressing_empty_continuation_page(
    tmp_path: Path,
) -> None:
    cache = SnippetResultCache(tmp_path / "pages")
    extractor = RelevantSnippetExtractor(
        ranker=CountingRanker(),
        chunker=FixedChunker(()),
        result_cache=cache,
    )
    identity = extractor._result_identity("doc-a", "", "focus", None)
    binding_identity = extractor._identity_without_cursor(identity)
    cache.put(
        identity,
        SnippetPage(
            document_id="doc-a",
            focus_query="focus",
            snippets=(),
            next_cursor=extractor._encode_cursor(binding_identity, 0),
            page_index=0,
            residual_count=0,
            residual_top_score=None,
            returned_min_score=None,
            pages_estimated=0,
        ),
        page_offset=0,
    )

    with pytest.raises(
        SnippetCacheIntegrityError, match="^invalid snippet cache entry$"
    ):
        extractor.extract("doc-a", "", "focus")


def test_cache_rejects_complete_long_document_even_with_consistent_checksum(
    tmp_path: Path,
) -> None:
    maximum = 20
    document = "L" * (maximum + 1)
    cache = SnippetResultCache(tmp_path / "pages")
    extractor = RelevantSnippetExtractor(
        ranker=CountingRanker(),
        chunker=FixedChunker(()),
        result_cache=cache,
        config=SnippetExtractionConfig(
            chunk_max_characters=maximum,
            chunk_overlap_characters=0,
        ),
    )
    identity = extractor._result_identity("doc-a", document, "focus", None)
    cache.put(
        identity,
        SnippetPage(
            document_id="doc-a",
            focus_query="focus",
            snippets=(
                RelevantSnippet(
                    chunk_id="doc-a:0000",
                    start_char=0,
                    end_char=len(document),
                    text=document,
                    relevance_score=1.0,
                ),
            ),
            next_cursor=None,
            page_index=0,
            residual_count=0,
            residual_top_score=None,
            returned_min_score=1.0,
            pages_estimated=1,
        ),
        page_offset=0,
    )

    with pytest.raises(
        SnippetCacheIntegrityError, match="^invalid snippet cache entry$"
    ):
        extractor.extract("doc-a", document, "focus")


def test_identical_threaded_result_cache_misses_are_single_flight(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(
        target=_thread_result_cache_probe,
        args=(str(tmp_path / "pages"), results),
    )
    try:
        process.start()
        process.join(timeout=10)
        assert not process.is_alive(), "threaded cache probe deadlocked"
        assert process.exitcode == 0
        assert results.get(timeout=2) == (
            "ok",
            ["hit", "miss"],
            ["target passage"],
        )
    finally:
        _stop_processes((process,))
        results.close()
        results.join_thread()


def test_identical_process_result_cache_misses_are_single_flight(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    start_barrier = context.Barrier(3, timeout=5)
    rank_barrier = context.Barrier(2, timeout=0.5)
    rank_calls = context.Value("i", 0)
    results = context.Queue()
    processes = [
        context.Process(
            target=_extract_in_process,
            args=(
                str(tmp_path / "pages"),
                start_barrier,
                rank_barrier,
                rank_calls,
                results,
            ),
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        start_barrier.wait()
        for process in processes:
            process.join(timeout=10)
        assert not any(process.is_alive() for process in processes), (
            "process cache probe deadlocked"
        )
        assert [process.exitcode for process in processes] == [0, 0]
        outcomes = [results.get(timeout=2) for _ in processes]
        assert sorted(outcomes) == [("ok", "hit"), ("ok", "miss")]
        assert rank_calls.value == 1
    finally:
        start_barrier.abort()
        rank_barrier.abort()
        _stop_processes(processes)
        results.close()
        results.join_thread()


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


def test_local_ranker_lazy_model_initialization_is_thread_safe(tmp_path: Path) -> None:
    model = object()
    loader_calls: list[str] = []
    loader_lock = Lock()
    loader_barrier = Barrier(2, timeout=0.5)

    def load_model(**_kwargs: object) -> object:
        with loader_lock:
            loader_calls.append("load")
        try:
            loader_barrier.wait()
        except BrokenBarrierError:
            pass
        return model

    ranker = LocalMixedbreadSnippetRanker(
        score_cache_root=tmp_path,
        model_loader=load_model,
        device="cpu",
    )
    start_barrier = Barrier(3, timeout=5)

    def get_model() -> object:
        start_barrier.wait()
        return ranker._get_model()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(get_model) for _index in range(2)]
        start_barrier.wait()
        loaded_models = [future.result(timeout=5) for future in futures]

    assert loaded_models == [model, model]
    assert loader_calls == ["load"]


def test_local_score_cache_spawned_process_misses_are_single_flight(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    start_barrier = context.Barrier(3, timeout=5)
    predict_barrier = context.Barrier(2, timeout=0.5)
    predict_calls = context.Value("i", 0)
    results = context.Queue()
    processes = [
        context.Process(
            target=_rank_local_in_process,
            args=(
                str(tmp_path),
                score,
                start_barrier,
                predict_barrier,
                predict_calls,
                results,
            ),
        )
        for score in (0.8, 0.2)
    ]
    try:
        for process in processes:
            process.start()
        start_barrier.wait()
        for process in processes:
            process.join(timeout=10)
        assert not any(process.is_alive() for process in processes), (
            "score-cache process probe deadlocked"
        )
        assert [process.exitcode for process in processes] == [0, 0]
        outcomes = [results.get(timeout=2) for _process in processes]
        assert all(outcome[0] == "ok" for outcome in outcomes), outcomes
        scores = [outcome[1] for outcome in outcomes]
        assert scores[0] == scores[1]
        assert predict_calls.value == 1
        score_files = list(tmp_path.rglob("*.jsonl"))
        assert len(score_files) == 1
        rows = score_files[0].read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1
        json.loads(rows[0])

        reloaded_model = FakeCrossEncoder([])
        reloaded = LocalMixedbreadSnippetRanker(
            score_cache_root=tmp_path,
            model_loader=lambda **_kwargs: reloaded_model,
            device="cpu",
        )
        assert (
            reloaded.rank("query", ADAPTER_CHUNKS[:1])[0].relevance_score
            == scores[0]
        )
        assert reloaded_model.predict_calls == 0
    finally:
        start_barrier.abort()
        predict_barrier.abort()
        _stop_processes(processes)
        results.close()
        results.join_thread()


def test_local_score_cache_constructor_waits_for_locked_complete_jsonl(
    tmp_path: Path,
) -> None:
    seeded_model = FakeCrossEncoder([[0.8]])
    seeded = LocalMixedbreadSnippetRanker(
        score_cache_root=tmp_path,
        model_loader=lambda **_kwargs: seeded_model,
        device="cpu",
    )
    assert seeded.rank("query", ADAPTER_CHUNKS[:1])[0].relevance_score == 0.8
    score_path = seeded._score_cache.path
    original = score_path.read_bytes()
    lock = FileLock(str(score_path.with_suffix(score_path.suffix + ".lock")))
    context = multiprocessing.get_context("spawn")
    about_to_construct = context.Queue()
    results = context.Queue()
    process = context.Process(
        target=_load_cached_score_in_process,
        args=(str(tmp_path), about_to_construct, results),
    )
    try:
        lock.acquire(timeout=2)
        try:
            with score_path.open("ab") as sink:
                sink.write(b'{"partial":')
                sink.flush()
                os.fsync(sink.fileno())
            process.start()
            assert about_to_construct.get(timeout=2) is True
            with pytest.raises(Empty):
                results.get(timeout=0.5)
        finally:
            try:
                with score_path.open("wb") as sink:
                    sink.write(original)
                    sink.flush()
                    os.fsync(sink.fileno())
            finally:
                lock.release()

        process.join(timeout=10)
        assert not process.is_alive(), "score-cache constructor probe deadlocked"
        assert process.exitcode == 0
        assert results.get(timeout=2) == ("ok", 0.8)
    finally:
        _stop_processes((process,))
        about_to_construct.close()
        about_to_construct.join_thread()
        results.close()
        results.join_thread()


def test_small_llm_ranker_caches_validated_scores_and_batches_prompts(tmp_path: Path) -> None:
    chat = FakeChatModel(
        [
            '{"scores":[{"chunk_id":"doc-a:0000","score":0.2}]}',
            '{"scores":[{"chunk_id":"doc-a:0001","score":0.8}]}',
        ]
    )
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
        "implementation_version": 2,
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
            "chunks": [{"chunk_id": "doc-a:0000", "text": "first evidence"}],
        },
        {
            "focus_query": "query",
            "chunks": [{"chunk_id": "doc-a:0001", "text": "second evidence"}],
        },
    ]


def test_small_llm_ranker_sends_every_uncached_duplicate_text_chunk_id(
    tmp_path: Path,
) -> None:
    duplicate_text_chunks = _chunks("same evidence", "same evidence")
    chat = FakeChatModel(
        [
            '{"scores":[{"chunk_id":"doc-a:0000","score":0.6}]}',
            '{"scores":[{"chunk_id":"doc-a:0001","score":0.6}]}',
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
    prompt_data = [json.loads(prompt.rsplit("\n", 1)[1]) for prompt in chat.prompts]
    assert [batch["chunks"][0]["chunk_id"] for batch in prompt_data] == [
        "doc-a:0000",
        "doc-a:0001",
    ]


def test_small_llm_ranker_keeps_same_text_scores_distinct_by_chunk_id(
    tmp_path: Path,
) -> None:
    duplicate_text_chunks = _chunks("same evidence", "same evidence")
    chat = FakeChatModel(
        [
            '{"scores":[{"chunk_id":"doc-a:0000","score":0.6}]}',
            '{"scores":[{"chunk_id":"doc-a:0001","score":0.4}]}',
        ]
    )
    ranker = SmallLLMSnippetRanker(
        chat_model=chat,
        score_cache_root=tmp_path,
        model_name="test-small-llm",
    )

    ranked = ranker.rank("query", duplicate_text_chunks)

    assert [(row.chunk.chunk_id, row.relevance_score) for row in ranked] == [
        ("doc-a:0000", 0.6),
        ("doc-a:0001", 0.4),
    ]


def test_small_llm_ranker_cache_includes_complete_peer_batch_context(
    tmp_path: Path,
) -> None:
    three_chunks = _chunks("first", "second", "third")
    first = SmallLLMSnippetRanker(
        chat_model=FakeChatModel(
            [
                '{"scores":['
                '{"chunk_id":"doc-a:0000","score":0.6},'
                '{"chunk_id":"doc-a:0001","score":0.4}'
                "]}",
                '{"scores":[{"chunk_id":"doc-a:0002","score":0.2}]}',
            ]
        ),
        score_cache_root=tmp_path,
        model_name="test-small-llm",
        batch_size=2,
    )
    first.rank("query", three_chunks)

    retry_chat = FakeChatModel(
        [
            '{"scores":[{"chunk_id":"doc-a:0000","score":0.3}]}',
            '{"scores":[{"chunk_id":"doc-a:0001","score":0.1}]}',
        ]
    )
    retry = SmallLLMSnippetRanker(
        chat_model=retry_chat,
        score_cache_root=tmp_path,
        model_name="test-small-llm",
        batch_size=2,
    )

    assert [row.relevance_score for row in retry.rank("query", three_chunks[:2])] == [
        0.3,
        0.1,
    ]
    assert len(retry_chat.prompts) == 2


def test_small_llm_ranker_cache_includes_chunk_id_for_identical_text(
    tmp_path: Path,
) -> None:
    first_chunk = _chunks("same evidence")[0]
    first = SmallLLMSnippetRanker(
        chat_model=FakeChatModel(
            ['{"scores":[{"chunk_id":"doc-a:0000","score":0.8}]}']
        ),
        score_cache_root=tmp_path,
        model_name="test-small-llm",
    )
    assert first.rank("query", (first_chunk,))[0].relevance_score == 0.8

    other_id = TextChunk(
        document_id="doc-a",
        chunk_id="doc-a:0099",
        text=first_chunk.text,
        start_char=first_chunk.start_char,
        end_char=first_chunk.end_char,
    )
    second_chat = FakeChatModel(
        ['{"scores":[{"chunk_id":"doc-a:0099","score":0.2}]}']
    )
    second = SmallLLMSnippetRanker(
        chat_model=second_chat,
        score_cache_root=tmp_path,
        model_name="test-small-llm",
    )

    assert second.rank("query", (other_id,))[0].relevance_score == 0.2
    assert len(second_chat.prompts) == 1


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
        '{"scores":[]}',
        '{"scores":[{"chunk_id":"doc-a:0000","score":0.8},{"chunk_id":"doc-a:0000","score":0.2}]}',
        '{"scores":[{"chunk_id":"unknown","score":0.2}]}',
        '{"scores":[{"chunk_id":"doc-a:0000","score":true}]}',
        '{"scores":[{"chunk_id":"doc-a:0000","score":NaN}]}',
        '{"scores":[],"scores":[{"chunk_id":"doc-a:0000","score":0.8}]}',
    ],
)
def test_small_llm_ranker_rejects_invalid_score_sets(tmp_path: Path, response: str) -> None:
    ranker = SmallLLMSnippetRanker(
        chat_model=FakeChatModel([response]),
        score_cache_root=tmp_path,
        model_name="test-small-llm",
    )

    with pytest.raises(ValueError, match="small-LLM score response"):
        ranker.rank("query", ADAPTER_CHUNKS[:1])


def test_default_extractor_constructs_lazy_local_ranker(tmp_path: Path) -> None:
    extractor = create_default_snippet_extractor(tmp_path)

    assert isinstance(extractor._ranker, LocalMixedbreadSnippetRanker)
    assert isinstance(extractor._chunker, SemanticTextChunker)
    assert extractor._chunker.config == ChunkingConfig(max_characters=3500, overlap_characters=350)
    assert extractor._result_cache.root_dir == tmp_path / "cache" / "reranker" / "deepagent_snippets"
