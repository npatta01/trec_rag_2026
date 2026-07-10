import hashlib
import json
from dataclasses import replace

import pytest

from trec_rag.chunking import TextChunk
from trec_rag.pipeline_models import RetrievedCandidate
from trec_rag.ranking import coverage_aware_long_doc_rank
from trec_rag.rerank_score_cache import (
    GlobalScoreCache,
    ScoreCacheContext,
    _document_artifact_row,
    _score_document_rows,
    _score_window_rows,
    _seed_document_score_cache,
)
from trec_rag.topics import Topic


class FakeModel:
    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0
        self.pairs: list[tuple[str, str]] = []
        self.predict_kwargs = {}

    def predict(self, pairs, **kwargs):
        self.calls += 1
        self.pairs.extend(pairs)
        self.predict_kwargs = kwargs
        return [self.score for _pair in pairs]


class FakeChunker:
    def split_text(self, text: str, *, document_id: str):
        midpoint = len(text) // 2
        return [
            TextChunk(document_id, f"{document_id}:0000", text[:midpoint], 0, midpoint),
            TextChunk(document_id, f"{document_id}:0001", text[midpoint:], midpoint, len(text)),
        ]


def _candidate() -> RetrievedCandidate:
    return RetrievedCandidate(
        topic_id="31",
        variant_name="original",
        retriever_name="bm25",
        query_text="explain bank failures",
        docid="doc-a",
        rank=1,
        score=12.0,
        text="Silicon Valley Bank failed after a rapid deposit run and bond losses.",
    )


def test_document_scores_reuse_global_content_cache(tmp_path):
    topic = Topic("31", "Banks", "Explain bank failures")
    context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=32768,
        score_kind="doc_max_32768_buf512",
    )
    cache = GlobalScoreCache(tmp_path, context)
    model = FakeModel(7.5)

    rows = _score_document_rows(
        model=model,
        topic=topic,
        candidates=[_candidate()],
        existing_scores={},
        batch_size=8,
        score_cache=cache,
        score_kind=context.score_kind,
    )

    assert model.calls == 1
    assert rows[0]["score"] == 7.5
    assert rows[0]["score_cache_key"]
    assert model.predict_kwargs["activation_fn"](3.0) == 3.0
    assert rows[0]["score_representation"] == "raw_logits"

    warm_cache = GlobalScoreCache(tmp_path, context)
    cold_model = FakeModel(99.0)
    reused_rows = _score_document_rows(
        model=cold_model,
        topic=topic,
        candidates=[_candidate()],
        existing_scores={},
        batch_size=8,
        score_cache=warm_cache,
        score_kind=context.score_kind,
    )

    assert cold_model.calls == 0
    assert reused_rows[0]["score"] == 7.5
    assert reused_rows[0]["score_cache_key"] == rows[0]["score_cache_key"]


def test_seeded_document_artifact_scores_materialize_from_global_cache(tmp_path):
    topic = Topic("31", "Banks", "Explain bank failures")
    context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=32768,
        score_kind="doc_max_32768_buf512",
    )
    cache = GlobalScoreCache(tmp_path, context)
    artifact_row = _document_artifact_row(
        topic=topic,
        candidate=_candidate(),
        score=3.25,
        score_cache=cache,
    )
    seeded = _seed_document_score_cache(
        topic=topic,
        candidates=[_candidate()],
        existing_scores={("31", "doc-a"): [artifact_row]},
        score_cache=cache,
    )

    assert seeded == 1

    model = FakeModel(99.0)
    rows = _score_document_rows(
        model=model,
        topic=topic,
        candidates=[_candidate()],
        existing_scores={},
        batch_size=8,
        score_cache=GlobalScoreCache(tmp_path, context),
        score_kind=context.score_kind,
    )

    assert model.calls == 0
    assert rows[0]["score"] == 3.25


def test_stale_document_artifact_is_not_seeded_under_changed_text(tmp_path):
    topic = Topic("31", "Banks", "Explain bank failures")
    context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=32768,
        score_kind="doc_max_32768_buf512",
    )
    cache = GlobalScoreCache(tmp_path, context)
    original = _candidate()
    stale_row = _document_artifact_row(
        topic=topic,
        candidate=original,
        score=3.25,
        score_cache=cache,
    )
    changed = replace(original, text=original.text + " Updated evidence.")

    seeded = _seed_document_score_cache(
        topic=topic,
        candidates=[changed],
        existing_scores={("31", "doc-a"): [stale_row]},
        score_cache=cache,
    )

    assert seeded == 0
    assert cache.get(query_text=changed.query_text, text=changed.text) is None


def test_window_scores_reuse_global_content_cache(tmp_path):
    topic = Topic("31", "Banks", "Explain bank failures")
    context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=1024,
        score_kind="window",
    )
    cache = GlobalScoreCache(tmp_path, context)
    model = FakeModel(4.0)

    rows = _score_window_rows(
        model=model,
        topic=topic,
        candidates=[_candidate()],
        existing_scores={},
        batch_size=8,
        chunker=FakeChunker(),
        score_cache=cache,
    )

    assert model.calls == 1
    assert [row["score"] for row in rows] == [4.0, 4.0]
    assert all(row["score_cache_key"] for row in rows)

    cold_model = FakeModel(99.0)
    reused_rows = _score_window_rows(
        model=cold_model,
        topic=topic,
        candidates=[_candidate()],
        existing_scores={},
        batch_size=8,
        chunker=FakeChunker(),
        score_cache=GlobalScoreCache(tmp_path, context),
    )

    assert cold_model.calls == 0
    assert [row["score"] for row in reused_rows] == [4.0, 4.0]
    assert [row["score_cache_key"] for row in reused_rows] == [
        row["score_cache_key"] for row in rows
    ]


def test_partial_window_artifact_materializes_only_missing_chunks(tmp_path):
    topic = Topic("31", "Banks", "Explain bank failures")
    context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=1024,
        score_kind="window",
    )
    cache = GlobalScoreCache(tmp_path, context)
    initial_rows = _score_window_rows(
        model=FakeModel(4.0),
        topic=topic,
        candidates=[_candidate()],
        existing_scores={},
        batch_size=8,
        chunker=FakeChunker(),
        score_cache=cache,
    )
    partial = {("31", "doc-a", 0): [initial_rows[0]]}
    cold_model = FakeModel(99.0)

    repaired_rows = _score_window_rows(
        model=cold_model,
        topic=topic,
        candidates=[_candidate()],
        existing_scores=partial,
        batch_size=8,
        chunker=FakeChunker(),
        score_cache=GlobalScoreCache(tmp_path, context),
    )

    assert cold_model.calls == 0
    assert [(row["chunk_index"], row["score"]) for row in repaired_rows] == [(1, 4.0)]


def test_ranking_rejects_incomplete_v2_window_artifact(tmp_path):
    candidate = _candidate()
    query_hash = hashlib.sha256(candidate.query_text.encode()).hexdigest()
    text_hash = hashlib.sha256(candidate.text.encode()).hexdigest()
    document_path = tmp_path / "document.jsonl"
    document_path.write_text(
        json.dumps(
            {
                "artifact_schema_version": 2,
                "topic_id": candidate.topic_id,
                "docid": candidate.docid,
                "score": 2.0,
                "query_sha256": query_hash,
                "text_sha256": text_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    window_path = tmp_path / "windows.jsonl"
    window_path.write_text(
        json.dumps(
            {
                "artifact_schema_version": 2,
                "topic_id": candidate.topic_id,
                "docid": candidate.docid,
                "score": 3.0,
                "chunk_index": 0,
                "chunk_count": 2,
                "chunk_max_characters": 3500,
                "chunk_overlap_characters": 350,
                "start_char": 0,
                "end_char": 10,
                "query_sha256": query_hash,
                "document_text_sha256": text_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete window reranker scores"):
        coverage_aware_long_doc_rank(
            [candidate],
            document_score_path=document_path,
            window_score_path=window_path,
            expected_score_metadata={"artifact_schema_version": 2},
            expected_window_score_metadata={
                "chunk_max_characters": 3500,
                "chunk_overlap_characters": 350,
            },
            long_document_weight=0.5,
            strongest_passage_weight=0.5,
            coverage_bonus_weight=0.25,
            relative_span_delta=1.0,
            support_cap=6,
            min_new_chars=1,
            top_window_weights=(1.0,),
        )


def test_ranking_rejects_v2_artifact_for_changed_document_text(tmp_path):
    candidate = _candidate()
    query_hash = hashlib.sha256(candidate.query_text.encode()).hexdigest()
    stale_text_hash = hashlib.sha256(b"old document text").hexdigest()
    document_path = tmp_path / "document.jsonl"
    document_path.write_text(
        json.dumps(
            {
                "artifact_schema_version": 2,
                "topic_id": candidate.topic_id,
                "docid": candidate.docid,
                "score": 2.0,
                "query_sha256": query_hash,
                "text_sha256": stale_text_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    window_path = tmp_path / "windows.jsonl"
    window_path.write_text(
        json.dumps(
            {
                "artifact_schema_version": 2,
                "topic_id": candidate.topic_id,
                "docid": candidate.docid,
                "score": 3.0,
                "chunk_index": 0,
                "chunk_count": 1,
                "chunk_max_characters": 3500,
                "chunk_overlap_characters": 350,
                "start_char": 0,
                "end_char": 10,
                "query_sha256": query_hash,
                "document_text_sha256": stale_text_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing or stale document reranker score"):
        coverage_aware_long_doc_rank(
            [candidate],
            document_score_path=document_path,
            window_score_path=window_path,
            expected_score_metadata={"artifact_schema_version": 2},
            expected_window_score_metadata={
                "chunk_max_characters": 3500,
                "chunk_overlap_characters": 350,
            },
            long_document_weight=0.5,
            strongest_passage_weight=0.5,
            coverage_bonus_weight=0.25,
            relative_span_delta=1.0,
            support_cap=6,
            min_new_chars=1,
            top_window_weights=(1.0,),
        )


def test_global_cache_rows_do_not_store_raw_query_or_document_text(tmp_path):
    context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=1024,
        score_kind="window",
    )
    cache = GlobalScoreCache(tmp_path, context)
    cache.add_many([("secret query text", "large document text", 1.0)])

    row = json.loads(cache.path.read_text(encoding="utf-8"))
    assert "secret query text" not in row.values()
    assert "large document text" not in row.values()
    assert row["query_sha256"]
    assert row["text_sha256"]


def test_global_cache_rejects_conflicting_duplicate_content_scores(tmp_path):
    cache = GlobalScoreCache(
        tmp_path,
        ScoreCacheContext(
            backend="sentence-transformers-cross-encoder",
            model="mixedbread-ai/mxbai-rerank-base-v2",
            max_length=1024,
            score_kind="window",
        ),
    )

    with pytest.raises(ValueError, match="conflicting score"):
        cache.add_many(
            [
                ("same query", "same text", 1.0),
                ("same query", "same text", 2.0),
            ]
        )


def test_global_cache_key_separates_score_representation_and_model_revision(tmp_path):
    raw = GlobalScoreCache(
        tmp_path,
        ScoreCacheContext(
            backend="sentence-transformers-cross-encoder",
            model="mixedbread-ai/mxbai-rerank-base-v2",
            max_length=1024,
            score_kind="window",
            model_revision="revision-a",
            score_representation="raw_logits",
        ),
    )
    probability = GlobalScoreCache(
        tmp_path,
        ScoreCacheContext(
            backend="sentence-transformers-cross-encoder",
            model="mixedbread-ai/mxbai-rerank-base-v2",
            max_length=1024,
            score_kind="window",
            model_revision="revision-a",
            score_representation="model_default",
        ),
    )
    other_revision = GlobalScoreCache(
        tmp_path,
        ScoreCacheContext(
            backend="sentence-transformers-cross-encoder",
            model="mixedbread-ai/mxbai-rerank-base-v2",
            max_length=1024,
            score_kind="window",
            model_revision="revision-b",
            score_representation="raw_logits",
        ),
    )

    raw_key = raw.cache_key(query_text="query", text="document")
    assert probability.cache_key(query_text="query", text="document") != raw_key
    assert other_revision.cache_key(query_text="query", text="document") != raw_key
    assert raw.path != probability.path
    assert raw.path != other_revision.path
