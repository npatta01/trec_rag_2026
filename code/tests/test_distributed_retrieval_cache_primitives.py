from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trec_rag.document_store import DocumentStore
from trec_rag.pipeline_config import RetrieverConfig
from trec_rag.pipeline_models import QueryVariant
from trec_rag.remote_client import RemoteSearchResponse
from trec_rag.remote_config import RemotePyseriniConfig
from trec_rag.repo_env import repo_cache_root
from trec_rag.retrieval_cache import (
    DerivationIdentity,
    OrganizerTextNormalizer,
    RetrievalCache,
    RetrievalCacheMiss,
    TransportIdentity,
)
from trec_rag.retrievers import PyseriniRemoteRetriever


ENDPOINT = "https://pyserini.test/v1/climbmix-400b/search"
QUERY_TEXT = "distributed cache-only query"


def _identity(query_text: str = QUERY_TEXT) -> TransportIdentity:
    return TransportIdentity.from_query(
        query_text=query_text,
        index_id="climbmix-400b",
        endpoint_identity=ENDPOINT,
        corpus_epoch="test-epoch",
        hits=2,
    )


def _response(query_text: str = QUERY_TEXT) -> bytes:
    return json.dumps(
        {
            "api": "v1",
            "index": "climbmix-400b",
            "query": {"text": query_text},
            "candidates": [
                {
                    "doc": "portable body",
                    "docid": "doc-portable",
                    "rank": 1,
                    "score": 7.5,
                }
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _cache(cache_root: Path) -> RetrievalCache:
    return RetrievalCache(
        cache_root / "retrieval" / "pyserini_remote",
        DocumentStore(cache_root / "documents" / "v1"),
        OrganizerTextNormalizer(),
    )


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, int], ...]:
    if not root.exists():
        return ()
    rows: list[tuple[str, str, int]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        content = path.read_bytes()
        rows.append(
            (
                path.relative_to(root).as_posix(),
                hashlib.sha256(content).hexdigest(),
                path.stat().st_mtime_ns,
            )
        )
    return tuple(rows)


def test_repo_cache_root_honors_absolute_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    override = tmp_path / "isolated-cache"
    checkout.mkdir()
    monkeypatch.setenv("TREC_RAG_CACHE_ROOT", str(override))

    assert repo_cache_root(checkout) == override


def test_repo_cache_root_rejects_relative_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TREC_RAG_CACHE_ROOT", "relative/cache")

    with pytest.raises(ValueError, match="TREC_RAG_CACHE_ROOT must be absolute"):
        repo_cache_root(tmp_path)


def test_complete_lookup_miss_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache = _cache(cache_root)
    derivation = DerivationIdentity.from_normalizer(cache.normalizer)

    assert cache.lookup_complete(_identity(), derivation, QUERY_TEXT) is None
    assert not cache_root.exists()


def test_complete_lookup_hit_does_not_touch_cache_files(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache = _cache(cache_root)
    derivation = DerivationIdentity.from_normalizer(cache.normalizer)
    cache.commit(_identity(), derivation, QUERY_TEXT, _response())
    before = _tree_snapshot(cache_root)

    cached = cache.lookup_complete(_identity(), derivation, QUERY_TEXT)

    assert cached is not None
    assert [hit.docid for hit in cached.hits] == ["doc-portable"]
    assert _tree_snapshot(cache_root) == before


def _retriever_config() -> RetrieverConfig:
    return RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=2,
        index="climbmix-400b",
        corpus_epoch="test-epoch",
    )


def _query(query_text: str = QUERY_TEXT) -> QueryVariant:
    return QueryVariant("rag2026-0", "original", query_text, "original_topic")


class _RecordingClient:
    config = RemotePyseriniConfig(ENDPOINT, None, 2, ())

    def __init__(self) -> None:
        self.calls = 0

    def search_raw(self, query_text: str, *, raw_sink=None) -> RemoteSearchResponse:
        self.calls += 1
        raw = _response(query_text)
        if raw_sink is not None:
            raw_sink(raw)
        return RemoteSearchResponse(
            raw=raw,
            payload=json.loads(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_cache_only_retriever_hit_never_constructs_client_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _RecordingClient()
    online = PyseriniRemoteRetriever(
        _retriever_config(), cache_dir=tmp_path / "cache", client=client
    )
    assert [row.docid for row in online.retrieve(_query())] == ["doc-portable"]
    assert online.transport_calls == 1
    before = _tree_snapshot(tmp_path / "cache")

    monkeypatch.setenv("INDEX_URL", ENDPOINT)

    def fail_client(*_args, **_kwargs):
        raise AssertionError("cache-only retrieval constructed a hosted client")

    monkeypatch.setattr("trec_rag.retrievers.RemotePyseriniClient", fail_client)
    replay = PyseriniRemoteRetriever(
        _retriever_config(), cache_dir=tmp_path / "cache", cache_only=True
    )

    assert [row.docid for row in replay.retrieve(_query())] == ["doc-portable"]
    assert replay.transport_calls == 0
    assert replay.cache_stats.hits == 1
    assert replay.cache_stats.misses == 0
    assert _tree_snapshot(tmp_path / "cache") == before


def test_cache_only_retriever_miss_counts_and_fails_before_external_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("INDEX_URL", ENDPOINT)

    def fail_client(*_args, **_kwargs):
        raise AssertionError("cache-only retrieval constructed a hosted client")

    monkeypatch.setattr("trec_rag.retrievers.RemotePyseriniClient", fail_client)
    retriever = PyseriniRemoteRetriever(
        _retriever_config(), cache_dir=cache_root, cache_only=True
    )

    with pytest.raises(RetrievalCacheMiss, match="cache-only retrieval miss"):
        retriever.retrieve(_query())

    assert retriever.transport_calls == 0
    assert retriever.cache_stats.hits == 0
    assert retriever.cache_stats.misses == 1
    assert not cache_root.exists()


@pytest.mark.parametrize("ticket_source", ["argument", "environment"])
def test_cache_only_retriever_rejects_continuation_ticket_before_filesystem_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ticket_source: str
) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.delenv("INDEX_URL", raising=False)

    def fail_file_lock(*_args, **_kwargs):
        raise AssertionError("cache-only construction accessed continuation state")

    monkeypatch.setattr("trec_rag.retrievers.FileLock", fail_file_lock)
    kwargs: dict[str, str] = {}
    if ticket_source == "argument":
        kwargs["continuation_ticket"] = "continuation-token"
    else:
        monkeypatch.setenv("PYSERINI_CONTINUATION_TICKET", "continuation-token")

    with pytest.raises(
        ValueError, match="cache-only retrieval cannot use a continuation ticket"
    ):
        PyseriniRemoteRetriever(
            _retriever_config(), cache_dir=cache_root, cache_only=True, **kwargs
        )

    assert not cache_root.exists()
