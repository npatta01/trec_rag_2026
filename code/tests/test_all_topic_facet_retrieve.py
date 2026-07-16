from __future__ import annotations

import copy
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

import trec_rag.all_topic_facet_retrieve as module
from trec_rag.all_topic_facet_retrieve import (
    build_retrieval_plan,
    build_union,
    run_retrieval,
    verify_retrieval,
)
from trec_rag.det_sparse_ledger import RawTransportResponse


TOPICS = ("1", "2")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _planning() -> dict[str, object]:
    facets = [
        {
            "request_order": order,
            "topic_id": topic_id,
            "facet_id": facet_id,
            "variant_name": f"all_topic_tethered_facet_validation_v1:{facet_id}",
            "query": query,
            "query_sha256": _sha(query.encode()),
            "depth": 200,
        }
        for order, (topic_id, facet_id, query) in enumerate(
            (
                ("1", "facet-a", "subject one relation a"),
                ("1", "facet-b", "subject one relation b"),
                ("2", "facet-c", "subject two relation c"),
            )
        )
    ]
    return {
        "schema_version": "all-topic-facet-request-plan-v1",
        "experiment_id": "all_topic_tethered_facet_validation_v1",
        "topic_ids": list(TOPICS),
        "original_depth": 1000,
        "facet_depth": 200,
        "request_interval_seconds": 3.0,
        "original_cache_hit_count": 2,
        "original_request_count": 0,
        "facet_request_count": 3,
        "total_external_request_count": 3,
        "originals": [
            {
                "topic_id": topic_id,
                "depth": 1000,
                "candidate_count": 1000,
                "cache_hit": True,
                "request_identity": {
                    "topic_id": topic_id,
                    "variant_name": "original",
                    "hits": 1000,
                    "query_sha256": _sha(f"narrative {topic_id}".encode()),
                },
                "raw_response_provenance": {
                    "cache_content_sha256": _sha(f"original-{topic_id}".encode())
                },
            }
            for topic_id in TOPICS
        ],
        "facet_requests": facets,
        "analyzer_sha256": "a" * 64,
        "planning_root_sha256": "b" * 64,
    }


def _original_cache() -> dict[str, dict[str, object]]:
    return {
        topic_id: {
            "topic_id": topic_id,
            "depth": 1000,
            "candidate_count": 1000,
            "request_identity": {
                "topic_id": topic_id,
                "variant_name": "original",
                "hits": 1000,
                "query_sha256": _sha(f"narrative {topic_id}".encode()),
            },
            "response_sha256": _sha(f"original-{topic_id}".encode()),
            "candidates": [
                {
                    "docid": f"{topic_id}-original-{rank}",
                    "rank": rank,
                    "score": float(1001 - rank),
                    "text": f"original passage {topic_id} {rank}",
                }
                for rank in range(1, 1001)
            ],
        }
        for topic_id in TOPICS
    }


def _body(prefix: str, *, count: int = 200) -> bytes:
    return json.dumps(
        {
            "candidates": [
                {
                    "docid": f"{prefix}-{rank}",
                    "rank": rank,
                    "score": float(201 - rank),
                    "doc": f"facet passage {prefix} {rank}",
                }
                for rank in range(1, count + 1)
            ]
        },
        separators=(",", ":"),
    ).encode()


class _Transport:
    one_shot_no_retry = True

    def __init__(self, *, fail_variant: str | None = None, crash_variant: str | None = None):
        self.fail_variant = fail_variant
        self.crash_variant = crash_variant
        self.identities: list[str] = []
        self.raw_exists_when_called: list[bool] = []

    def __call__(self, request):
        variant = request.identity.variant_name.rsplit(":", 1)[-1]
        self.identities.append(variant)
        if variant == self.crash_variant:
            raise KeyboardInterrupt("synthetic crash")
        if variant == self.fail_variant:
            return RawTransportResponse(
                status=503,
                headers={"Retry-After": "300"},
                body=b'{"error":"synthetic failure"}',
                elapsed_seconds=0.01,
            )
        return RawTransportResponse(
            status=200,
            headers={"Content-Type": "application/json"},
            body=_body(variant),
            elapsed_seconds=0.01,
        )


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


@pytest.fixture(autouse=True)
def _authorized_synthetic_topics(monkeypatch):
    monkeypatch.setattr(module, "ALL_TOPIC_IDS", TOPICS)


def test_original_requests_must_be_exact_cache_hits() -> None:
    with pytest.raises(ValueError, match="original cache incomplete"):
        build_retrieval_plan(_planning(), original_cache={})


def test_plan_rejects_topic_and_depth_drift() -> None:
    topic_drift = _planning()
    topic_drift["topic_ids"] = ["2", "1"]
    with pytest.raises(ValueError, match="topic drift"):
        build_retrieval_plan(topic_drift, _original_cache())

    depth_drift = _planning()
    depth_drift["facet_requests"][0]["depth"] = 199
    with pytest.raises(ValueError, match="depth drift"):
        build_retrieval_plan(depth_drift, _original_cache())


def test_union_deduplicates_by_topic_and_document_with_provenance() -> None:
    original = [
        {
            "topic_id": "1",
            "document_id": "doc-a",
            "text": "passage",
            "stream_id": "original",
            "stream_rank": 1,
            "request_identity": {"variant_name": "original"},
            "query_sha256": "a" * 64,
            "response_sha256": "b" * 64,
        }
    ]
    facets = [
        {
            "topic_id": "1",
            "document_id": "doc-a",
            "text": "passage",
            "stream_id": "facet-a",
            "stream_rank": 2,
            "request_identity": {"variant_name": "facet-a"},
            "query_sha256": "c" * 64,
            "response_sha256": "d" * 64,
        }
    ]

    rows = build_union(original, facets)

    assert len({(row["topic_id"], row["document_id"]) for row in rows}) == len(rows) == 1
    assert [p["stream_id"] for p in rows[0]["stream_provenance"]] == [
        "original",
        "facet-a",
    ]
    assert all(
        {"stream_id", "stream_rank", "request_identity", "query_sha256", "response_sha256"}
        <= set(provenance)
        for provenance in rows[0]["stream_provenance"]
    )


def test_limiter_spaces_attempts_and_failed_identity_is_never_retried(tmp_path: Path) -> None:
    transport = _Transport(fail_variant="facet-b")
    result = run_retrieval(
        _planning(),
        tmp_path / "retrieval",
        transport,
        clock=_Clock(),
        cache_root=tmp_path / "cache",
        original_cache=_original_cache(),
    )

    assert result["complete"] is False
    assert result["request_start_deltas"] == [3.0, 3.0]
    assert transport.identities.count("facet-b") == 1
    raw = list((tmp_path / "retrieval" / "ledger" / "raw").glob("*.json"))
    assert len(raw) == 3

    resumed = _Transport()
    with pytest.raises(ValueError, match="immutable failed attempt"):
        run_retrieval(
            _planning(),
            tmp_path / "retrieval",
            resumed,
            clock=_Clock(),
            cache_root=tmp_path / "cache",
            original_cache=_original_cache(),
        )
    assert resumed.identities == []


def test_crash_resume_reuses_successes_and_refuses_pending_attempt(tmp_path: Path) -> None:
    output = tmp_path / "retrieval"
    crashing = _Transport(crash_variant="facet-b")
    with pytest.raises(KeyboardInterrupt, match="synthetic crash"):
        run_retrieval(
            _planning(),
            output,
            crashing,
            clock=_Clock(),
            cache_root=tmp_path / "cache",
            original_cache=_original_cache(),
        )
    assert crashing.identities == ["facet-a", "facet-b"]

    resumed = _Transport()
    with pytest.raises(ValueError, match="pending attempt"):
        run_retrieval(
            _planning(),
            output,
            resumed,
            clock=_Clock(),
            cache_root=tmp_path / "cache",
            original_cache=_original_cache(),
        )
    assert resumed.identities == []


def test_successful_cache_only_resume_seals_and_verifies_union(tmp_path: Path) -> None:
    output = tmp_path / "retrieval"
    cache_root = tmp_path / "cache"
    first = run_retrieval(
        _planning(),
        output,
        _Transport(),
        clock=_Clock(),
        cache_root=cache_root,
        original_cache=_original_cache(),
    )
    assert first["complete"] is True
    assert first["external_attempts"] == 3
    assert verify_retrieval(output)["verified"] is True

    # A fresh output consumes only authenticated exact cache entries.
    cache_transport = _Transport()
    second_output = tmp_path / "cache-only"
    second = run_retrieval(
        _planning(),
        second_output,
        cache_transport,
        clock=_Clock(),
        cache_root=cache_root,
        original_cache=_original_cache(),
    )
    assert second["complete"] is True
    assert second["external_attempts"] == 0
    assert cache_transport.identities == []

    union_path = second_output / "accepted_union.jsonl"
    union_path.write_bytes(union_path.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="retrieval seal mismatch"):
        verify_retrieval(second_output)


def test_transport_rejects_original_network_request(tmp_path: Path) -> None:
    request = module._build_facet_requests(build_retrieval_plan(_planning(), _original_cache()))[0]
    original = copy.deepcopy(request)
    object.__setattr__(original.identity, "variant_name", "original")
    transport = module.RateLimitedFacetTransport(
        module.build_live_config(api_token=None), [request], session=object()
    )
    with pytest.raises(ValueError, match="original-query network request rejected"):
        transport(original)


def test_live_transport_uses_persistent_limiter_and_makes_one_attempt() -> None:
    class Response:
        status_code = 503
        headers = {"Retry-After": "300"}
        content = b'{"error":"upstream"}'
        elapsed = timedelta(milliseconds=1)

    class Session:
        def __init__(self) -> None:
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

    request = module._build_facet_requests(build_retrieval_plan(_planning(), _original_cache()))[0]
    config = module.build_live_config(api_token="token")
    session = Session()
    transport = module.RateLimitedFacetTransport(config, [request], session=session)

    response = transport(request)

    assert response.status == 503
    assert transport.one_shot_no_retry is True
    assert config.min_interval_seconds == 3.0
    assert config.burst == 1
    assert config.limiter_state_path == module.build_live_config(api_token=None).limiter_state_path
    assert len(session.calls) == 1
    assert session.calls[0][1]["allow_redirects"] is False


def test_verify_rejects_semantic_provenance_drift_even_with_fresh_self_seal(
    tmp_path: Path,
) -> None:
    output = tmp_path / "retrieval"
    run_retrieval(
        _planning(),
        output,
        _Transport(),
        clock=_Clock(),
        cache_root=tmp_path / "cache",
        original_cache=_original_cache(),
    )
    original_path = output / "original_candidates.jsonl"
    original = [json.loads(line) for line in original_path.read_text().splitlines()]
    original[0]["query_sha256"] = "f" * 64
    original_path.write_bytes(module._jsonl_bytes(original))
    facets = [
        json.loads(line)
        for line in (output / "facet_candidates.jsonl").read_text().splitlines()
    ]
    union = build_union(original, facets)
    (output / "accepted_union.jsonl").write_bytes(module._jsonl_bytes(union))
    (output / "RETRIEVAL_SEALED.json").unlink()
    module._create_seal(output)

    with pytest.raises(ValueError, match="original stream provenance"):
        verify_retrieval(output)
