from __future__ import annotations

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest

import trec_rag.deep_facet_candidate_run as module
from trec_rag.deep_facet_candidate_manifest import EXCLUDED_TOPIC_IDS, TOPIC_IDS
from trec_rag.deep_facet_candidate_run import (
    ENDPOINT,
    FACET_HITS,
    LIMITER_STATE_PATH,
    MIN_INTERVAL_SECONDS,
    RateLimitedFacetTransport,
    build_live_config,
    build_requests,
    create_preflight,
    execute_retrieval,
)
from trec_rag.det_sparse_ledger import RawTransportResponse


def _manifest() -> dict[str, object]:
    counts = {"219": 7, "72": 7, "300": 4, "84": 7}
    facets: list[dict[str, object]] = []
    for topic_id in TOPIC_IDS:
        for topic_order in range(counts[topic_id]):
            query = f"subject {topic_id} relation {topic_order}"
            facets.append(
                {
                    "topic_id": topic_id,
                    "facet_id": f"{topic_id}-f{topic_order}",
                    "query": query,
                    "manifest_order": len(facets),
                }
            )
    return {
        "schema_version": "fixture",
        "experiment_id": "rag25_deep_facet_candidates_v1",
        "topic_ids": list(TOPIC_IDS),
        "facets": facets,
        "analyzer": {"version": "fixture", "rules": "fixture"},
        "retrieval": {
            "endpoint": ENDPOINT,
            "index": "climbmix-400b",
            "facet_hits": 200,
            "max_external_requests": 25,
            "minimum_interval_seconds": 3,
            "retry_count": 0,
        },
        "qrels_opened": False,
    }


@pytest.fixture(autouse=True)
def _accept_synthetic_manifest(monkeypatch):
    monkeypatch.setattr(module, "validate_manifest", lambda value, *, cache_root: None)


def _body(prefix: str, *, count: int = 200) -> bytes:
    return json.dumps(
        {
            "candidates": [
                {
                    "docid": f"{prefix}-{rank:03d}",
                    "rank": rank,
                    "score": 201.0 - rank,
                    "doc": f"synthetic evidence passage {prefix} {rank}",
                }
                for rank in range(1, count + 1)
            ]
        },
        separators=(",", ":"),
    ).encode("utf-8")


class _FakeTransport:
    one_shot_no_retry = True

    def __init__(self, *, short_at: int | None = None) -> None:
        self.calls = []
        self.short_at = short_at

    def __call__(self, request):
        self.calls.append(request)
        count = 199 if len(self.calls) == self.short_at else 200
        return RawTransportResponse(
            status=200,
            headers={"Content-Type": "application/json"},
            body=_body(request.identity.request_key[:8], count=count),
            elapsed_seconds=0.01,
        )


def test_exact_requests_are_unique_depth_200_and_ordered(tmp_path: Path) -> None:
    requests = build_requests(_manifest(), ENDPOINT, cache_root=tmp_path)

    assert len(requests) == 25
    assert len({request.identity.request_key for request in requests}) == 25
    assert all(request.identity.hits == FACET_HITS == 200 for request in requests)
    assert [request.identity.topic_id for request in requests] == [
        topic_id
        for topic_id, count in (("219", 7), ("72", 7), ("300", 4), ("84", 7))
        for _ in range(count)
    ]


@pytest.mark.parametrize("topic_id", sorted(EXCLUDED_TOPIC_IDS, key=int))
def test_excluded_topic_fails_before_request_construction(
    topic_id: str, tmp_path: Path, monkeypatch
) -> None:
    manifest = _manifest()
    manifest["facets"][0]["topic_id"] = topic_id
    constructed = []
    monkeypatch.setattr(
        module.RetrievalRequest,
        "from_query",
        lambda **kwargs: constructed.append(kwargs),
    )

    with pytest.raises(ValueError, match="excluded topic"):
        build_requests(manifest, ENDPOINT, cache_root=tmp_path)

    assert constructed == []


def test_preflight_freezes_limiter_and_does_not_call_transport(tmp_path: Path) -> None:
    output = tmp_path / "run"
    payload = create_preflight(
        _manifest(), output, cache_root=tmp_path / "cache", source_cache_root=tmp_path
    )

    assert payload["planned_requests"] == 25
    assert payload["projected_external_attempts"] == 25
    assert payload["minimum_start_span_seconds"] == 72.0
    assert payload["minimum_batch_window_seconds"] == 75.0
    assert payload["min_interval_seconds"] == MIN_INTERVAL_SECONDS == 3.0
    assert payload["qrels_opened"] is False
    assert len(payload["requests"]) == 25


def test_successful_run_writes_exact_5000_row_canonical_ledger(tmp_path: Path) -> None:
    output = tmp_path / "run"
    cache = tmp_path / "cache"
    create_preflight(_manifest(), output, cache_root=cache, source_cache_root=tmp_path)
    transport = _FakeTransport()

    summary = execute_retrieval(
        _manifest(),
        transport,
        output,
        cache_root=cache,
        source_cache_root=tmp_path,
    )

    assert len(transport.calls) == 25
    assert summary["candidate_rows"] == 5000
    assert summary["external_calls"] == 25
    assert summary["successes"] == 25
    assert summary["failures"] == 0
    rows = (output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 5000
    assert len(list((output / "ledger" / "raw").glob("*.json"))) == 25
    assert len(list((output / "ledger" / "outcomes").glob("*.json"))) == 25


def test_short_response_aborts_and_preserves_failure_bytes(tmp_path: Path) -> None:
    output = tmp_path / "run"
    cache = tmp_path / "cache"
    create_preflight(_manifest(), output, cache_root=cache, source_cache_root=tmp_path)
    transport = _FakeTransport(short_at=2)

    with pytest.raises(ValueError, match="exactly 200 unique text-bearing"):
        execute_retrieval(
            _manifest(),
            transport,
            output,
            cache_root=cache,
            source_cache_root=tmp_path,
        )

    assert len(transport.calls) == 2
    assert len(list((output / "ledger" / "raw").glob("*.json"))) == 2
    outcomes = [
        json.loads(path.read_text())
        for path in sorted((output / "ledger" / "outcomes").glob("*.json"))
    ]
    assert sorted(row["status"] for row in outcomes) == ["failure", "success"]
    assert not (output / "candidates.jsonl").exists()


class _FakeResponse:
    status_code = 503
    headers = {"Content-Type": "application/json", "Retry-After": "300"}
    content = b'{"error":"exact upstream failure"}'
    elapsed = timedelta(milliseconds=10)


class _FakeSession:
    def __init__(self) -> None:
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse()


def test_transport_is_persistent_rate_limited_and_never_retries(tmp_path: Path) -> None:
    requests = build_requests(_manifest(), ENDPOINT, cache_root=tmp_path)
    config = build_live_config(api_token="token")
    assert config.limiter_state_path == LIMITER_STATE_PATH
    assert config.min_interval_seconds == 3.0
    session = _FakeSession()
    transport = RateLimitedFacetTransport(config, requests, session=session)

    result = transport(requests[0])

    assert result.status == 503
    assert result.body == _FakeResponse.content
    assert len(session.calls) == 1
    assert session.calls[0][1]["params"] == {
        "query": requests[0].query_text,
        "hits": "200",
    }
    assert transport.one_shot_no_retry is True


def test_preflight_rejects_mutation_after_qrels_sentinel(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "QRELS_ACCESSED").write_text("sealed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="qrels already accessed"):
        create_preflight(
            _manifest(), output, cache_root=tmp_path / "cache", source_cache_root=tmp_path
        )
