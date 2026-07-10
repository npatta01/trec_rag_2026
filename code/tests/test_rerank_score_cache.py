import json

from trec_rag.chunking import TextChunk
from trec_rag.pipeline_models import RetrievedCandidate
from trec_rag.rerank_score_cache import (
    GlobalScoreCache,
    ScoreCacheContext,
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

    def predict(self, pairs, **_kwargs):
        self.calls += 1
        self.pairs.extend(pairs)
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
        existing_keys=set(),
        batch_size=8,
        score_cache=cache,
        score_kind=context.score_kind,
    )

    assert model.calls == 1
    assert rows[0]["score"] == 7.5
    assert rows[0]["score_cache_key"]

    warm_cache = GlobalScoreCache(tmp_path, context)
    cold_model = FakeModel(99.0)
    reused_rows = _score_document_rows(
        model=cold_model,
        topic=topic,
        candidates=[_candidate()],
        existing_keys=set(),
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
    seeded = _seed_document_score_cache(
        topic=topic,
        candidates=[_candidate()],
        existing_scores={("31", "doc-a"): 3.25},
        score_cache=cache,
    )

    assert seeded == 1

    model = FakeModel(99.0)
    rows = _score_document_rows(
        model=model,
        topic=topic,
        candidates=[_candidate()],
        existing_keys=set(),
        batch_size=8,
        score_cache=GlobalScoreCache(tmp_path, context),
        score_kind=context.score_kind,
    )

    assert model.calls == 0
    assert rows[0]["score"] == 3.25


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
        existing_docids=set(),
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
        existing_docids=set(),
        batch_size=8,
        chunker=FakeChunker(),
        score_cache=GlobalScoreCache(tmp_path, context),
    )

    assert cold_model.calls == 0
    assert [row["score"] for row in reused_rows] == [4.0, 4.0]
    assert [row["score_cache_key"] for row in reused_rows] == [
        row["score_cache_key"] for row in rows
    ]


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
