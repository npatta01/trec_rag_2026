import hashlib
import json
import multiprocessing
import os
import sqlite3
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

import trec_rag.rerank_score_cache as score_cache_module
from trec_rag.chunking import TextChunk
from trec_rag.document_store import DocumentStore
from trec_rag.pipeline_config import RetrieverConfig
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate
from trec_rag.ranking import coverage_aware_long_doc_rank
from trec_rag.rerank_score_cache import (
    GlobalScoreCache,
    ScoreCacheContext,
    _document_artifact_row,
    _load_cached_candidates,
    _score_document_rows,
    _score_window_rows,
    _seed_document_score_cache,
    _scores_to_list,
)
from trec_rag.retrieval_cache import (
    DerivationIdentity,
    OrganizerTextNormalizer,
    RetrievalCache,
    TransportIdentity,
)
from trec_rag.topics import Topic


def _score_many_worker(root_dir, context, ready, results, pairs, score, marker_name):
    cache = GlobalScoreCache(root_dir, context)

    def compute_batch(batch):
        marker = Path(root_dir) / marker_name
        try:
            marker.open("x", encoding="utf-8").close()
        except FileExistsError as exc:
            raise AssertionError("a live-leased key was computed twice") from exc
        time.sleep(0.25)
        return [score for _pair in batch]

    ready.wait()
    try:
        results.put(("ok", cache.score_many(pairs, compute_batch, batch_size=8, lease_seconds=2)))
    except BaseException as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))


def _fork_connection_worker(cache, results):
    connection = cache.connection
    results.put((os.getpid(), cache.connection_identity, id(connection)))


def _disjoint_score_many_worker(root_dir, context, ready, results, pairs, name):
    cache = GlobalScoreCache(root_dir, context)

    def compute_batch(batch):
        (Path(root_dir) / f"{name}.started").touch()
        other = "b" if name == "a" else "a"
        deadline = time.time() + 5
        while not (Path(root_dir) / f"{other}.started").exists():
            if time.time() > deadline:
                raise AssertionError("disjoint inference did not overlap")
            time.sleep(0.01)
        return [float(len(pair[0])) for pair in batch]

    ready.wait()
    try:
        results.put(cache.score_many(pairs, compute_batch, batch_size=1, lease_seconds=2))
    except BaseException as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))


def _legacy_import_worker(root_dir, context, source_path, ready, results):
    cache = GlobalScoreCache(root_dir, context)
    ready.wait()
    try:
        results.put(cache.import_legacy_jsonl(source_path, legacy_context=context))
    except BaseException as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))


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


def test_load_cached_candidates_reads_authenticated_v2_retrieval_cache(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache" / "retrieval" / "pyserini_remote"
    endpoint = "https://pyserini.test/v1/climbmix-400b/search"
    query = QueryVariant("14", "original", "the exact narrative", "original_topic")
    retriever = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=1000,
        index="climbmix-400b",
        corpus_epoch="test-corpus-epoch",
    )
    normalizer = OrganizerTextNormalizer()
    retrieval_cache = RetrievalCache(
        cache_dir,
        DocumentStore(tmp_path / "cache" / "documents" / "v1"),
        normalizer,
    )
    identity = TransportIdentity.from_query(
        query_text=query.query_text,
        index_id=retriever.index,
        endpoint_identity=endpoint,
        corpus_epoch=retriever.corpus_epoch,
        hits=retriever.hits,
    )
    raw_response = json.dumps(
        {
            "api": "v1",
            "index": "climbmix-400b",
            "query": {"text": query.query_text},
            "candidates": [
                {"rank": 1, "docid": "doc-1", "score": 3.5, "doc": "exact body"}
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    retrieval_cache.commit(
        identity,
        DerivationIdentity.from_normalizer(normalizer),
        query.query_text,
        raw_response,
    )
    monkeypatch.setenv("INDEX_URL", endpoint)

    candidates = _load_cached_candidates(
        query=query,
        retriever=retriever,
        cache_dir=cache_dir,
        index_url=endpoint,
    )

    assert candidates == [
        RetrievedCandidate(
            topic_id="14",
            variant_name="original",
            retriever_name="climbmix_bm25",
            query_text="the exact narrative",
            docid="doc-1",
            rank=1,
            score=3.5,
            text="exact body",
        )
    ]


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

    row = cache.connection.execute(
        "SELECT query_sha256, text_sha256, score FROM scores"
    ).fetchone()
    assert row[0] == hashlib.sha256(b"secret query text").digest()
    assert row[1] == hashlib.sha256(b"large document text").digest()
    assert row[2] == 1.0


def test_global_cache_get_reloads_scores_added_by_another_instance(tmp_path):
    context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=1024,
        score_kind="window",
    )
    first = GlobalScoreCache(tmp_path, context)
    second = GlobalScoreCache(tmp_path, context)

    second.add_many([("shared query", "shared text", 2.5)])

    assert first.get(query_text="shared query", text="shared text") == 2.5


def test_global_cache_concurrent_identical_adds_publish_one_row(tmp_path):
    context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=1024,
        score_kind="window",
    )
    caches = [GlobalScoreCache(tmp_path, context) for _ in range(2)]
    ready = threading.Barrier(2)
    outcomes: list[tuple[str, object]] = []

    def add(cache: GlobalScoreCache) -> int:
        ready.wait()
        try:
            outcomes.append(("ok", cache.add_many([("shared query", "shared text", 2.5)])))
        except BaseException as exc:  # pragma: no cover - keeps thread failures visible below
            outcomes.append(("error", exc))
        return 0

    results = [threading.Thread(target=add, args=(cache,)) for cache in caches]
    for thread in results:
        thread.start()
    for thread in results:
        thread.join()

    assert [status for status, _value in outcomes] == ["ok", "ok"]
    assert sorted(value for _status, value in outcomes) == [0, 1]
    assert caches[0].connection.execute("SELECT count(*) FROM scores").fetchone()[0] == 1


def test_global_cache_rechecks_and_rejects_conflicting_stale_write(tmp_path):
    context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=1024,
        score_kind="window",
    )
    first = GlobalScoreCache(tmp_path, context)
    second = GlobalScoreCache(tmp_path, context)
    first.add_many([("shared query", "shared text", 1.0)])

    with pytest.raises(ValueError, match="conflicting score"):
        second.add_many([("shared query", "shared text", 2.0)])

    assert first.connection.execute("SELECT count(*) FROM scores").fetchone()[0] == 1


def test_global_cache_fails_closed_on_non_sqlite_database(tmp_path):
    context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=1024,
        score_kind="window",
    )
    path = tmp_path.joinpath(*context.path_parts)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a sqlite database")

    with pytest.raises(sqlite3.DatabaseError, match="not a database"):
        GlobalScoreCache(tmp_path, context)


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


def _v2_context(**overrides):
    values = {
        "backend": "sentence-transformers-cross-encoder",
        "model": "mixedbread-ai/mxbai-rerank-base-v2",
        "max_length": 1024,
        "score_kind": "window",
        "model_revision": "revision-a",
        "backend_version": "backend-a",
        "score_representation": "raw_logits",
        "inference_dtype": "float32",
        "input_policy": "exact-pair-v1",
        "effective_max_length": 1024,
        "scoring_contract": "cross-encoder-contract-v1",
        "transformers_version": "4.55.0",
        "torch_version": "2.7.0",
        "device_family": "cpu",
        "fixed_batch_policy": "batch-size-8",
    }
    values.update(overrides)
    return ScoreCacheContext(**values)


def _legacy_jsonl_row(context, query_text, text, score):
    query_sha256 = hashlib.sha256(query_text.encode()).hexdigest()
    text_sha256 = hashlib.sha256(text.encode()).hexdigest()
    payload = {
        "schema_version": 2,
        "backend": context.backend,
        "backend_version": context.backend_version,
        "model": context.model,
        "model_revision": context.model_revision,
        "score_representation": context.score_representation,
        "inference_dtype": context.inference_dtype,
        "input_policy": context.input_policy,
        "max_length": context.max_length,
        "score_kind": context.score_kind,
        "query_sha256": query_sha256,
        "text_sha256": text_sha256,
    }
    cache_key = hashlib.sha256(
        json.dumps(
            {
                **payload,
                "schema_version": 2,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        **payload,
        "cache_key": cache_key,
        "score": score,
    }


def _hash_for_test(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def test_v2_context_separates_paths_and_fails_closed_on_meta_mismatch(tmp_path):
    context = _v2_context()
    cache = GlobalScoreCache(tmp_path, context)
    same = GlobalScoreCache(tmp_path, _v2_context())
    changed = GlobalScoreCache(tmp_path, _v2_context(device_family="cuda"))

    assert cache.path.suffix == ".sqlite3"
    assert "score-cache-v2" in cache.path.as_posix()
    assert cache.path == same.path
    assert cache.path != changed.path
    assert cache.context_sha256 == hashlib.sha256(cache.context_json.encode()).hexdigest()

    with cache.connection:
        cache.connection.execute(
            "UPDATE cache_meta SET value = ? WHERE key = 'context_sha256'",
            ("00" * 32,),
        )
    cache.close()
    with pytest.raises(ValueError, match="context|metadata|schema"):
        GlobalScoreCache(tmp_path, context)


def test_v2_initialization_does_not_sample_database_before_sqlite_lock(
    tmp_path,
    monkeypatch,
):
    context = _v2_context()
    database = tmp_path.joinpath(*context.path_parts)
    real_exists = Path.exists

    def reject_prelock_sample(path):
        if path == database:
            raise AssertionError("database existence was sampled before SQLite locking")
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", reject_prelock_sample)

    cache = GlobalScoreCache(tmp_path, context)

    assert cache.connection.execute("SELECT count(*) FROM cache_meta").fetchone()[0] == 3


def test_v2_lookup_preserves_order_and_deduplicates_keys(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    cache.seed_many(
        [("q1", "t1", 1.25), ("q2", "t2", 2.5)],
        source_path="fixture",
        source_sha256="11" * 32,
    )

    assert cache.lookup_many(
        [("q2", "t2"), ("q1", "t1"), ("q2", "t2"), ("missing", "text")]
    ) == [2.5, 1.25, 2.5, None]


def test_v2_overlapping_processes_compute_each_live_key_once(tmp_path):
    context = _v2_context()
    mp = multiprocessing.get_context("spawn")
    ready = mp.Event()
    results = mp.Queue()
    processes = [
        mp.Process(
            target=_score_many_worker,
            args=(tmp_path, context, ready, results, [("same", "text")], 6.5, "same.marker"),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    outcomes = [results.get(timeout=5) for _ in processes]
    assert sorted(outcomes) == [("ok", [6.5]), ("ok", [6.5])]


def test_v2_disjoint_batches_overlap_inference(tmp_path):
    context = _v2_context()
    mp = multiprocessing.get_context("spawn")
    ready = mp.Event()
    results = mp.Queue()

    processes = [
        mp.Process(
            target=_disjoint_score_many_worker,
            args=(tmp_path, context, ready, results, [("a", "text-a"), ("shared", "text")], "a"),
        ),
        mp.Process(
            target=_disjoint_score_many_worker,
            args=(tmp_path, context, ready, results, [("shared", "text"), ("b", "text-b")], "b"),
        ),
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert sorted(results.get(timeout=5) for _ in processes) == [[1.0, 6.0], [6.0, 1.0]]


def test_v2_heartbeat_prevents_takeover_and_late_conflict_is_hard(tmp_path):
    context = _v2_context()
    first = GlobalScoreCache(tmp_path, context)
    second = GlobalScoreCache(tmp_path, context)
    started = threading.Event()
    release = threading.Event()
    outcome = []

    def slow_compute(batch):
        started.set()
        release.wait(timeout=5)
        return [1.0]

    def run_first():
        try:
            outcome.append(
                first.score_many([("q", "t")], slow_compute, batch_size=1, lease_seconds=0.12)
            )
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert started.wait(timeout=2)
    time.sleep(0.2)
    claim = second.connection.execute(
        "SELECT lease_expires_at FROM claims"
    ).fetchone()
    assert claim is not None and claim[0] > time.time()
    with second.connection:
        second.connection.execute("UPDATE claims SET lease_expires_at = 0")
    assert second.score_many(
        [("q", "t")], lambda batch: [2.0 for _pair in batch], batch_size=1, lease_seconds=1
    ) == [2.0]
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], BaseException)
    assert first.lookup_many([("q", "t")]) == [2.0]


def test_v2_exception_releases_owned_claims(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())

    def fail(_batch):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        cache.score_many([("q", "t")], fail, batch_size=1)
    assert cache.connection.execute("SELECT count(*) FROM claims").fetchone()[0] == 0
    assert cache.score_many(
        [("q", "t")], lambda batch: [3.0 for _pair in batch], batch_size=1
    ) == [3.0]


def test_v2_connections_are_thread_and_post_fork_local(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    parent_identity = cache.connection_identity
    thread_identities = []
    thread = threading.Thread(target=lambda: thread_identities.append(cache.connection_identity))
    thread.start()
    thread.join()
    assert thread_identities and thread_identities[0] != parent_identity

    mp = multiprocessing.get_context("fork")
    results = mp.Queue()
    process = mp.Process(target=_fork_connection_worker, args=(cache, results))
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0
    child_pid, child_identity, _child_connection_id = results.get(timeout=5)
    assert child_pid != os.getpid()
    assert child_identity[0] == child_pid
    assert child_identity != parent_identity


def test_v2_uses_full_durable_sqlite_settings_and_integrity_check(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    connection = cache.connection
    assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 60000
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    table_sql = {
        row[0]: row[1]
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {"cache_meta", "scores", "claims", "imports"} <= table_sql.keys()
    assert all("STRICT" in sql for sql in table_sql.values())


def test_v2_logical_binding_ignores_unrelated_concurrent_inserts(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    cache.seed_many(
        [("q", "t", 1.0)], source_path="one", source_sha256="22" * 32
    )
    requested = [("q", "t")]
    before = cache.logical_binding(requested)
    cache.seed_many(
        [("unrelated", "text", 9.0)], source_path="two", source_sha256="33" * 32
    )
    assert cache.logical_binding(requested) == before


def test_v2_explicit_legacy_jsonl_import_is_streaming_idempotent_and_conflict_safe(tmp_path):
    context = _v2_context()
    cache = GlobalScoreCache(tmp_path, context)
    legacy = tmp_path / "legacy.jsonl"
    rows = [
        _legacy_jsonl_row(context, "q", "t", 1.5),
        _legacy_jsonl_row(context, "q", "t", 1.5),
    ]
    legacy.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert cache.lookup_many([("q", "t")]) == [None]
    first = cache.import_legacy_jsonl(legacy, legacy_context=context)
    second = cache.import_legacy_jsonl(legacy, legacy_context=context)
    assert first["source_row_count"] == 2
    assert first["inserted_count"] == 1
    assert first["authorization_sha256"] == cache._authorization_digest(context).hex()
    assert second == first
    assert cache.lookup_many([("q", "t")]) == [1.5]
    assert cache.connection.execute(
        "SELECT count(*) FROM imports WHERE authorization_sha256 IS NOT NULL"
    ).fetchone()[0] == 1

    conflicting = tmp_path / "conflicting.jsonl"
    conflicting.write_text(
        json.dumps(_legacy_jsonl_row(context, "q", "t", 2.5)) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="conflict"):
        cache.import_legacy_jsonl(conflicting, legacy_context=context)

    malformed = tmp_path / "malformed-key.jsonl"
    malformed.write_text(
        json.dumps({**_legacy_jsonl_row(context, "q2", "t2", 2.5), "cache_key": "00" * 32})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="key/hash"):
        cache.import_legacy_jsonl(malformed, legacy_context=context)


def test_v2_legacy_import_requires_explicit_context_authorization(tmp_path):
    context = _v2_context()
    cache = GlobalScoreCache(tmp_path, context)
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps(_legacy_jsonl_row(context, "q", "t", 1.5)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="authorization|legacy context"):
        cache.import_legacy_jsonl(legacy)


def test_v2_legacy_import_does_not_replace_empty_target_wal_state(tmp_path):
    context = _v2_context()
    cache = GlobalScoreCache(tmp_path, context)
    protected_pair = cache._normalize_pair(("writer query", "writer text"))
    with cache.connection:
        cache.connection.execute(
            "INSERT INTO claims(key_sha256, query_sha256, text_sha256, owner, token, claimed_at, lease_expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                protected_pair.key,
                protected_pair.query_sha256,
                protected_pair.text_sha256,
                "concurrent-writer",
                b"w" * 16,
                time.time(),
                time.time() + 60,
            ),
        )
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps(_legacy_jsonl_row(context, "q", "t", 1.5)) + "\n",
        encoding="utf-8",
    )

    cache.import_legacy_jsonl(legacy, legacy_context=context)

    assert cache.connection.execute("SELECT count(*) FROM claims").fetchone()[0] == 1
    assert cache.lookup_many([("q", "t")]) == [1.5]


def test_v2_legacy_import_rolls_back_rows_when_target_promotion_fails(tmp_path):
    context = _v2_context()
    cache = GlobalScoreCache(tmp_path, context)
    cache.add_many([("existing", "text", 0.5)])
    legacy = tmp_path / "large-legacy.jsonl"
    legacy.write_text(
        "".join(
            json.dumps(_legacy_jsonl_row(context, f"q-{index}", f"t-{index}", float(index)))
            + "\n"
            for index in range(513)
        ),
        encoding="utf-8",
    )
    cache.connection.execute(
        "CREATE TRIGGER fail_legacy_import_promotion "
        "BEFORE INSERT ON scores WHEN NEW.score = 512.0 "
        "BEGIN SELECT RAISE(ABORT, 'promotion failed'); END"
    )

    with pytest.raises(sqlite3.IntegrityError, match="promotion failed"):
        cache.import_legacy_jsonl(legacy, legacy_context=context)

    assert cache.connection.execute("SELECT count(*) FROM scores").fetchone()[0] == 1
    assert cache.connection.execute("SELECT count(*) FROM imports").fetchone()[0] == 0


def test_v2_same_source_legacy_importers_return_the_same_completed_receipt(tmp_path):
    context = _v2_context()
    source = tmp_path / "legacy.jsonl"
    source.write_text(
        "".join(
            json.dumps(_legacy_jsonl_row(context, f"q-{index}", f"t-{index}", float(index)))
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    seed_cache = GlobalScoreCache(tmp_path, context)
    seed_cache.add_many([("existing", "text", 0.5)])
    seed_cache.close()
    mp = multiprocessing.get_context("spawn")
    ready = mp.Event()
    results = mp.Queue()
    processes = [
        mp.Process(
            target=_legacy_import_worker,
            args=(tmp_path, context, source, ready, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    receipts = [results.get(timeout=5) for _ in processes]

    assert receipts[0] == receipts[1]
    assert receipts[0]["inserted_count"] == 2
    assert GlobalScoreCache(tmp_path, context).connection.execute(
        "SELECT count(*) FROM imports"
    ).fetchone()[0] == 1


def test_v2_expired_claim_takeover_rejects_query_text_hash_mismatch(tmp_path):
    context = _v2_context()
    cache = GlobalScoreCache(tmp_path, context)
    pair = cache._normalize_pair(("requested query", "requested text"))
    with cache.connection:
        cache.connection.execute(
            "INSERT INTO claims(key_sha256, query_sha256, text_sha256, owner, token, claimed_at, lease_expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                pair.key,
                _hash_for_test("different query"),
                _hash_for_test("different text"),
                "expired-owner",
                b"e" * 16,
                time.time() - 60,
                time.time() - 1,
            ),
        )

    with pytest.raises(ValueError, match="hash"):
        cache.score_many(
            [pair.raw],
            lambda batch: [1.0 for _pair in batch],
            batch_size=1,
        )


def test_v2_document_and_window_scoring_use_leased_score_many(tmp_path, monkeypatch):
    topic = Topic("31", "Banks", "Explain bank failures")
    document_context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=32768,
        score_kind="doc_max_32768_buf512",
    )
    document_cache = GlobalScoreCache(tmp_path, document_context)
    monkeypatch.setattr(document_cache, "get", lambda **_kwargs: pytest.fail("get hot path"))
    monkeypatch.setattr(
        document_cache,
        "add_many",
        lambda _rows: pytest.fail("add_many hot path"),
    )
    document_rows = _score_document_rows(
        model=FakeModel(7.5),
        topic=topic,
        candidates=[_candidate()],
        existing_scores={},
        batch_size=8,
        score_cache=document_cache,
        score_kind=document_context.score_kind,
    )
    assert document_rows[0]["score"] == 7.5

    window_context = replace(document_context, max_length=1024, score_kind="window")
    window_cache = GlobalScoreCache(tmp_path, window_context)
    monkeypatch.setattr(window_cache, "get", lambda **_kwargs: pytest.fail("get hot path"))
    monkeypatch.setattr(
        window_cache,
        "add_many",
        lambda _rows: pytest.fail("add_many hot path"),
    )
    window_rows = _score_window_rows(
        model=FakeModel(4.0),
        topic=topic,
        candidates=[_candidate()],
        existing_scores={},
        batch_size=8,
        chunker=FakeChunker(),
        score_cache=window_cache,
    )
    assert [row["score"] for row in window_rows] == [4.0, 4.0]


@pytest.mark.parametrize("missing_object", ["claims", "claims_expiry_idx"])
def test_v2_existing_schema_missing_objects_fails_closed(tmp_path, missing_object):
    context = _v2_context()
    cache = GlobalScoreCache(tmp_path, context)
    cache.close()
    with sqlite3.connect(cache.path) as connection:
        connection.execute(f"DROP {'INDEX' if missing_object.endswith('idx') else 'TABLE'} {missing_object}")

    with pytest.raises(ValueError, match="schema"):
        GlobalScoreCache(tmp_path, context)


def test_v2_lookup_many_batches_sql_and_claims_are_bounded(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    cache.add_many([(f"q-{index}", f"t-{index}", float(index)) for index in range(20)])
    statements: list[str] = []
    cache.connection.set_trace_callback(statements.append)
    assert cache.lookup_many([(f"q-{index}", f"t-{index}") for index in range(20)]) == [
        float(index) for index in range(20)
    ]
    select_statements = [statement for statement in statements if "FROM scores" in statement]
    assert len(select_statements) == 1
    assert " IN (" in select_statements[0]

    claim_counts: list[int] = []
    current_claim_count = 0

    def trace_claim_transactions(statement: str) -> None:
        nonlocal current_claim_count
        upper = statement.upper()
        if upper == "BEGIN IMMEDIATE":
            current_claim_count = 0
        elif "INSERT INTO CLAIMS" in upper:
            current_claim_count += 1
        elif upper == "COMMIT":
            claim_counts.append(current_claim_count)

    cache.connection.set_trace_callback(trace_claim_transactions)
    assert cache.score_many(
        [(f"missing-{index}", "text") for index in range(3)],
        lambda batch: [1.0 for _pair in batch],
        batch_size=1,
    ) == [1.0, 1.0, 1.0]
    assert max(claim_counts) <= 1


def test_v2_score_many_waits_with_backoff_not_fixed_20ms_polling(tmp_path, monkeypatch):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    other = GlobalScoreCache(tmp_path, _v2_context())
    pair = cache._normalize_pair(("waiting query", "waiting text"))
    with other.connection:
        other.connection.execute(
            "INSERT INTO claims(key_sha256, query_sha256, text_sha256, owner, token, claimed_at, lease_expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                pair.key,
                pair.query_sha256,
                pair.text_sha256,
                "other-owner",
                b"o" * 16,
                time.time(),
                time.time() + 60,
            ),
        )
    observed_sleeps: list[float] = []
    sleep_started = threading.Event()
    real_sleep = score_cache_module.time.sleep

    def observe_sleep(seconds: float) -> None:
        observed_sleeps.append(seconds)
        sleep_started.set()
        real_sleep(seconds)

    monkeypatch.setattr(score_cache_module.time, "sleep", observe_sleep)
    result: list[object] = []

    def wait_for_score() -> None:
        result.append(
            cache.score_many(
                [pair.raw],
                lambda batch: [2.0 for _pair in batch],
                batch_size=1,
            )
        )

    thread = threading.Thread(target=wait_for_score)
    thread.start()
    assert sleep_started.wait(timeout=2)
    other.add_many([("waiting query", "waiting text", 2.0)])
    thread.join(timeout=5)

    assert result == [[2.0]]
    assert observed_sleeps and min(observed_sleeps) >= 0.05


def test_v2_thread_connections_do_not_run_full_integrity_check(tmp_path, monkeypatch):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    real_connect = sqlite3.connect
    integrity_checks: list[str] = []

    class TracedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.strip().upper() == "PRAGMA INTEGRITY_CHECK":
                integrity_checks.append(sql)
            return super().execute(sql, parameters)

    def traced_connect(*args, **kwargs):
        kwargs["factory"] = TracedConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(score_cache_module.sqlite3, "connect", traced_connect)
    thread = threading.Thread(target=lambda: cache.lookup_many([("missing", "text")]))
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert integrity_checks == []


@pytest.mark.parametrize("score", [True, False, float("nan"), float("inf"), float("-inf")])
def test_v2_api_rejects_bool_and_nonfinite_scores(tmp_path, score):
    cache = GlobalScoreCache(tmp_path, _v2_context())

    with pytest.raises(ValueError, match="finite|bool"):
        cache.add_many([("q", "t", score)])


def test_v2_schema_and_reads_reject_nonfinite_scores(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    pair = cache._normalize_pair(("q", "t"))
    with pytest.raises(sqlite3.IntegrityError):
        cache.connection.execute(
            "INSERT INTO scores(key_sha256, query_sha256, text_sha256, score) VALUES (?, ?, ?, ?)",
            (pair.key, pair.query_sha256, pair.text_sha256, float("inf")),
        )
    cache.add_many([("q", "t", 1.0)])
    cache.connection.execute("PRAGMA ignore_check_constraints = ON")
    cache.connection.execute("UPDATE scores SET score = ? WHERE key_sha256 = ?", (float("inf"), pair.key))
    with pytest.raises(ValueError, match="finite"):
        cache.lookup_many([("q", "t")])


def test_v2_large_cache_operations_do_not_read_or_rewrite_full_jsonl(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    rows = [(f"q-{index}", f"t-{index}", float(index)) for index in range(100)]
    with patch.object(Path, "read_text", side_effect=AssertionError("full text reload")):
        cache.seed_many(rows, source_path="bulk", source_sha256="44" * 32)
        assert cache.lookup_many([(query, text) for query, text, _score in rows]) == [
            score for _query, _text, score in rows
        ]
        cache.seed_many(
            [("q-100", "t-100", 100.0)], source_path="bulk-2", source_sha256="55" * 32
        )
    assert cache.path.suffix == ".sqlite3"


def test_v2_legacy_import_uses_connection_local_file_backed_temp_staging(tmp_path):
    context = _v2_context()
    cache = GlobalScoreCache(tmp_path, context)
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        "".join(
            json.dumps(_legacy_jsonl_row(context, f"q-{index}", f"t-{index}", float(index)))
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )

    cache.import_legacy_jsonl(legacy, legacy_context=context)

    assert cache.connection.execute("PRAGMA temp_store").fetchone()[0] == 1
    assert not list(tmp_path.glob(".*.import*"))


def test_v2_legacy_import_publication_does_not_insert_rows_one_at_a_time(
    tmp_path,
    monkeypatch,
):
    context = _v2_context()
    cache = GlobalScoreCache(tmp_path, context)
    cache.add_many([("q-0", "t-0", 0.0)])
    legacy = tmp_path / "legacy.jsonl"
    rows = [
        _legacy_jsonl_row(context, "q-0", "t-0", 0.0),
        _legacy_jsonl_row(context, "q-1", "t-1", 1.0),
        _legacy_jsonl_row(context, "q-1", "t-1", 1.0),
        _legacy_jsonl_row(context, "q-2", "t-2", 2.0),
    ]
    legacy.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cache,
        "_insert_score",
        lambda *_args, **_kwargs: pytest.fail("row-wise legacy publication"),
    )

    first = cache.import_legacy_jsonl(legacy, legacy_context=context)
    second = cache.import_legacy_jsonl(legacy, legacy_context=context)

    assert first == second
    assert first["source_row_count"] == 4
    assert first["inserted_count"] == 2
    assert cache.lookup_many([("q-0", "t-0"), ("q-1", "t-1"), ("q-2", "t-2")]) == [
        0.0,
        1.0,
        2.0,
    ]


@pytest.mark.parametrize("policy", ["trec_rag_raw_v2", "extractive_sentence_pair_v1"])
def test_v2_legacy_false_policy_cannot_become_effective_same_policy(tmp_path, policy):
    legacy_context = _v2_context(input_policy=policy)
    cache = GlobalScoreCache(tmp_path, _v2_context(input_policy=policy))
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps(_legacy_jsonl_row(legacy_context, "q", "t", 1.5)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input policy|context rebinding"):
        cache.import_legacy_jsonl(legacy, legacy_context=legacy_context)


def test_v2_legacy_false_raw_policy_migrates_only_to_whitespace_and_receipt_preserves_context(
    tmp_path,
):
    legacy_context = _v2_context(input_policy="trec_rag_raw_v2")
    target_context = _v2_context(input_policy="trec_rag_whitespace_v1")
    cache = GlobalScoreCache(tmp_path, target_context)
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps(_legacy_jsonl_row(legacy_context, "q", "t", 1.5)) + "\n",
        encoding="utf-8",
    )

    receipt = cache.import_legacy_jsonl(legacy, legacy_context=legacy_context)

    assert receipt["legacy_context_sha256"] == legacy_context.context_sha256
    assert receipt["target_context_sha256"] == target_context.context_sha256
    assert receipt["declared_input_policy"] == "trec_rag_raw_v2"
    assert receipt["effective_input_policy"] == "trec_rag_whitespace_v1"


def test_v2_legacy_context_rebinding_rejects_non_policy_mismatch(tmp_path):
    legacy_context = _v2_context(model_revision="legacy-revision")
    cache = GlobalScoreCache(tmp_path, _v2_context())
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps(_legacy_jsonl_row(legacy_context, "q", "t", 1.5)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="context rebinding"):
        cache.import_legacy_jsonl(legacy, legacy_context=legacy_context)


def test_v2_format_neutral_score_import_is_atomic_and_reports_duplicates(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    query_hash = hashlib.sha256(b"import query").hexdigest()
    text_hash = hashlib.sha256(b"import text").hexdigest()
    rows = [
        {"query_sha256": query_hash, "text_sha256": text_hash, "score": 1.25},
        {"query_sha256": query_hash, "text_sha256": text_hash, "score": 1.25},
    ]

    receipt = cache.import_scores(
        rows,
        source_path="facet-ledger",
        source_sha256="66" * 32,
        authorization_sha256="77" * 32,
    )

    assert receipt["source_row_count"] == 2
    assert receipt["duplicate_count"] == 1
    assert receipt["inserted_count"] == 1
    assert receipt["already_identical_count"] == 0
    assert receipt["authorization_sha256"] == "77" * 32
    assert cache.lookup_many([("import query", "import text")]) == [1.25]

    repeated = cache.import_scores(
        rows,
        source_path="facet-ledger",
        source_sha256="66" * 32,
        authorization_sha256="77" * 32,
    )
    assert repeated["inserted_count"] == 0
    assert repeated["already_identical_count"] == 1


def test_v2_format_neutral_score_import_rolls_back_conflicts_and_interruption(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    first = {
        "query_sha256": hashlib.sha256(b"first query").hexdigest(),
        "text_sha256": hashlib.sha256(b"first text").hexdigest(),
        "score": 1.0,
    }
    second = {
        "query_sha256": hashlib.sha256(b"second query").hexdigest(),
        "text_sha256": hashlib.sha256(b"second text").hexdigest(),
        "score": 2.0,
    }

    with pytest.raises(ValueError, match="conflicting score"):
        cache.import_scores(
            [first, second, {**first, "score": 3.0}],
            source_path="conflict",
            source_sha256="88" * 32,
            authorization_sha256="99" * 32,
        )
    assert cache.connection.execute("SELECT count(*) FROM scores").fetchone()[0] == 0
    assert cache.connection.execute("SELECT count(*) FROM imports").fetchone()[0] == 0

    def interrupted_rows():
        yield first
        raise KeyboardInterrupt("source interrupted")

    with pytest.raises(KeyboardInterrupt, match="source interrupted"):
        cache.import_scores(
            interrupted_rows(),
            source_path="interrupted",
            source_sha256="aa" * 32,
            authorization_sha256="bb" * 32,
        )
    assert cache.connection.execute("SELECT count(*) FROM scores").fetchone()[0] == 0
    assert cache.connection.execute("SELECT count(*) FROM imports").fetchone()[0] == 0


def test_v2_format_neutral_score_import_is_concurrent_and_idempotent(tmp_path):
    context = _v2_context()
    rows = [
        {
            "query_sha256": hashlib.sha256(b"concurrent query").hexdigest(),
            "text_sha256": hashlib.sha256(b"concurrent text").hexdigest(),
            "score": 4.5,
        }
    ]
    outcomes = []
    barrier = threading.Barrier(2)

    def import_once():
        cache = GlobalScoreCache(tmp_path, context)
        barrier.wait()
        try:
            outcomes.append(
                cache.import_scores(
                    rows,
                    source_path="concurrent",
                    source_sha256="cc" * 32,
                    authorization_sha256="dd" * 32,
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced by assertions below
            outcomes.append(exc)

    threads = [threading.Thread(target=import_once) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(isinstance(item, dict) for item in outcomes)
    assert sorted(item["inserted_count"] for item in outcomes) == [0, 1]
    assert sorted(item["already_identical_count"] for item in outcomes) == [0, 1]


def test_v2_score_many_claims_only_one_bounded_batch_at_a_time(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    started = threading.Event()
    release = threading.Event()
    result: list[list[float]] = []

    def compute(batch):
        assert cache.connection.execute("SELECT count(*) FROM claims").fetchone()[0] <= 2
        started.set()
        release.wait(timeout=5)
        return [1.0 for _pair in batch]

    thread = threading.Thread(
        target=lambda: result.append(
            cache.score_many(
                [(f"q-{index}", f"t-{index}") for index in range(5)],
                compute,
                batch_size=2,
                lease_seconds=5,
            )
        )
    )
    thread.start()
    assert started.wait(timeout=2)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result == [[1.0, 1.0, 1.0, 1.0, 1.0]]


def test_v2_score_many_timestamps_claim_after_write_lock_and_heartbeats_before_compute(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    blocker = GlobalScoreCache(tmp_path, _v2_context())
    blocker.connection.execute("BEGIN IMMEDIATE")
    released_at: list[float] = []
    claimed_at: list[float] = []
    heartbeat_times: list[float] = []
    compute_started = threading.Event()
    release_compute = threading.Event()
    result: list[list[float]] = []

    original_heartbeat = cache._heartbeat

    def observed_heartbeat(owner, token, lease_seconds):
        heartbeat_times.append(time.time())
        return original_heartbeat(owner, token, lease_seconds)

    cache._heartbeat = observed_heartbeat

    def compute(batch):
        assert heartbeat_times
        claimed_at.append(
            cache.connection.execute("SELECT claimed_at FROM claims").fetchone()[0]
        )
        compute_started.set()
        release_compute.wait(timeout=5)
        return [1.0]

    thread = threading.Thread(
        target=lambda: result.append(
            cache.score_many([("q", "t")], compute, batch_size=1, lease_seconds=5)
        )
    )
    thread.start()
    time.sleep(0.1)
    released_at.append(time.time())
    blocker.connection.commit()
    assert compute_started.wait(timeout=2)
    release_compute.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result == [[1.0]]
    assert claimed_at[0] >= released_at[0]


@pytest.mark.parametrize(
    "unsupported_sql",
    [
        "CREATE VIEW unsupported_view AS SELECT 1",
        "CREATE TRIGGER unsupported_trigger AFTER INSERT ON scores BEGIN SELECT 1; END",
    ],
)
def test_v2_existing_schema_rejects_unsupported_objects(tmp_path, unsupported_sql):
    context = _v2_context()
    cache = GlobalScoreCache(tmp_path, context)
    cache.close()
    with sqlite3.connect(cache.path) as connection:
        connection.execute(unsupported_sql)

    with pytest.raises(ValueError, match="schema"):
        GlobalScoreCache(tmp_path, context)


def test_v2_score_api_rejects_scalar_bool(tmp_path):
    with pytest.raises(ValueError, match="bool|finite"):
        _scores_to_list(True)


def test_v2_api_and_schema_accept_the_same_maximum_finite_score(tmp_path):
    cache = GlobalScoreCache(tmp_path, _v2_context())
    maximum = sys.float_info.max

    cache.add_many([("q", "t", maximum)])

    assert cache.lookup_many([("q", "t")]) == [maximum]


def test_v2_score_artifact_paths_do_not_lookup_before_score_many(tmp_path, monkeypatch):
    topic = Topic("31", "Banks", "Explain bank failures")
    context = ScoreCacheContext(
        backend="sentence-transformers-cross-encoder",
        model="mixedbread-ai/mxbai-rerank-base-v2",
        max_length=32768,
        score_kind="doc_max_32768_buf512",
    )
    cache = GlobalScoreCache(tmp_path, context)
    monkeypatch.setattr(
        cache,
        "lookup_many",
        lambda _pairs: pytest.fail("redundant lookup before score_many"),
    )

    rows = _score_document_rows(
        model=FakeModel(7.5),
        topic=topic,
        candidates=[_candidate()],
        existing_scores={},
        batch_size=8,
        score_cache=cache,
        score_kind=context.score_kind,
    )

    assert rows[0]["score"] == 7.5
