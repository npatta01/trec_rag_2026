from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest
import requests

import trec_rag.all_topic_facet_retrieve as module
from trec_rag.all_topic_facet_retrieve import (
    _run_synthetic_retrieval,
    _verify_synthetic_retrieval,
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
        self.last_start_event: dict[str, object] | None = None
        self._event_count = 0
        self._lock = threading.Lock()

    def __call__(self, request):
        variant = request.identity.variant_name.rsplit(":", 1)[-1]
        with self._lock:
            self.identities.append(variant)
            epoch = 1_700_000_000.0 + self._event_count * 3.0
            self._event_count += 1
            self.last_start_event = {
                "request_key": request.identity.request_key,
                "started_at_epoch": epoch,
                "started_at_utc": f"2023-11-14T22:13:{20 + int(epoch - 1_700_000_000):02d}Z",
            }
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


class _ProcessTransport:
    one_shot_no_retry = True

    def __init__(self, calls, event_counter, *, die: bool = False) -> None:
        self.calls = calls
        self.event_counter = event_counter
        self.die = die
        self.last_start_event = None

    def __call__(self, request):
        with self.calls.get_lock():
            self.calls.value += 1
        if self.die:
            os._exit(17)
        with self.event_counter.get_lock():
            index = self.event_counter.value
            self.event_counter.value += 1
        epoch = 1_700_000_000.0 + index * 3.0
        self.last_start_event = {
            "request_key": request.identity.request_key,
            "started_at_epoch": epoch,
            "started_at_utc": f"2023-11-14T22:13:{20 + index * 3:02d}Z",
        }
        variant = request.identity.variant_name.rsplit(":", 1)[-1]
        return RawTransportResponse(
            status=200,
            headers={"Content-Type": "application/json"},
            body=_body(variant),
            elapsed_seconds=0.01,
        )


def _process_synthetic_run(
    planning, original, cache_root, output, calls, events, start, queue, die=False
):
    start.wait()
    try:
        result = module._run_synthetic_retrieval(
            planning,
            Path(output),
            _ProcessTransport(calls, events, die=die),
            clock=_Clock(),
            cache_root=Path(cache_root),
            original_cache=original,
        )
        queue.put(("ok", result["complete"]))
    except BaseException as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


class _NoNetworkAdapter(requests.adapters.BaseAdapter):
    def send(self, request, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.headers = {"Content-Type": "application/json"}
        response.request = request
        return response

    def close(self) -> None:
        return None


def _process_limiter_grant(state_path, ledger_path, request_key, start, queue):
    config = module.RemotePyseriniConfig(
        index_url=module.ENDPOINT,
        api_token=None,
        hits=200,
        queries=(),
        min_interval_seconds=3.0,
        burst=1,
        limiter_state_path=Path(state_path),
    )
    session = module._grant_timestamp_session(
        config, grant_ledger_path=Path(ledger_path)
    )
    session.mount("http://", _NoNetworkAdapter())
    prepared = requests.Request(
        "GET",
        module.ENDPOINT,
        headers={module._REQUEST_KEY_HEADER: request_key},
    ).prepare()
    start.wait()
    session.send(prepared)
    event = session.consume_grant_event(request_key)
    queue.put(event)


@pytest.fixture(autouse=True)
def _authorized_synthetic_topics(monkeypatch):
    monkeypatch.setattr(module, "ALL_TOPIC_IDS", TOPICS)
    monkeypatch.setattr(module, "EXPECTED_FACET_REQUEST_COUNT", 3)


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


def test_union_accepts_whitespace_only_text_variants_and_preserves_first_text() -> None:
    original = [
        {
            "topic_id": "1",
            "document_id": "doc-a",
            "text": "Heading\n\nFirst line.\nSecond line.",
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
            "text": "Heading First line. Second line.",
            "stream_id": "facet-a",
            "stream_rank": 2,
            "request_identity": {"variant_name": "facet-a"},
            "query_sha256": "c" * 64,
            "response_sha256": "d" * 64,
        }
    ]

    rows = build_union(original, facets)

    assert rows[0]["text"] == original[0]["text"]
    assert [item["text_sha256"] for item in rows[0]["stream_provenance"]] == [
        hashlib.sha256(original[0]["text"].encode()).hexdigest(),
        hashlib.sha256(facets[0]["text"].encode()).hexdigest(),
    ]


def test_union_rejects_non_whitespace_text_conflict() -> None:
    rows = [
        {
            "topic_id": "1",
            "document_id": "doc-a",
            "text": "Materially different passage one.",
            "stream_id": "original",
            "stream_rank": 1,
            "request_identity": {"variant_name": "original"},
            "query_sha256": "a" * 64,
            "response_sha256": "b" * 64,
        },
        {
            "topic_id": "1",
            "document_id": "doc-a",
            "text": "Materially different passage two.",
            "stream_id": "facet-a",
            "stream_rank": 1,
            "request_identity": {"variant_name": "facet-a"},
            "query_sha256": "c" * 64,
            "response_sha256": "d" * 64,
        },
    ]

    with pytest.raises(ValueError, match="conflicting passage text"):
        build_union(rows[:1], rows[1:])


def test_limiter_spaces_attempts_and_failed_identity_is_never_retried(tmp_path: Path) -> None:
    transport = _Transport(fail_variant="facet-b")
    result = _run_synthetic_retrieval(
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
    with pytest.raises(ValueError, match="immutable shared failure attempt"):
        _run_synthetic_retrieval(
            _planning(),
            tmp_path / "different-output-after-failure",
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
        _run_synthetic_retrieval(
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
        _run_synthetic_retrieval(
            _planning(),
            tmp_path / "different-output-after-crash",
            resumed,
            clock=_Clock(),
            cache_root=tmp_path / "cache",
            original_cache=_original_cache(),
        )
    assert resumed.identities == []


def test_successful_cache_only_resume_seals_and_verifies_union(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "retrieval"
    cache_root = tmp_path / "cache"
    first = _run_synthetic_retrieval(
        _planning(),
        output,
        _Transport(),
        clock=_Clock(),
        cache_root=cache_root,
        original_cache=_original_cache(),
    )
    assert first["complete"] is True
    assert first["external_attempts"] == 3
    assert _verify_against_synthetic_planning(monkeypatch, output)["verified"] is True

    # A fresh output consumes only authenticated exact cache entries.
    cache_transport = _Transport()
    second_output = tmp_path / "cache-only"
    second = _run_synthetic_retrieval(
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
        _verify_against_synthetic_planning(monkeypatch, second_output)


def test_same_output_resume_reports_authenticated_lifetime_attempts(tmp_path: Path) -> None:
    output = tmp_path / "retrieval"
    first = _run_synthetic_retrieval(
        _planning(),
        output,
        _Transport(),
        clock=_Clock(),
        cache_root=tmp_path / "cache",
        original_cache=_original_cache(),
    )
    assert first["external_attempts"] == 3
    assert first["cache_hits"] == 0

    for name in (
        "RETRIEVAL_SEALED.json",
        "retrieval_summary.json",
        "original_candidates.jsonl",
        "facet_candidates.jsonl",
        "accepted_union.jsonl",
    ):
        (output / name).unlink()

    resumed = _run_synthetic_retrieval(
        _planning(),
        output,
        _Transport(),
        clock=_Clock(),
        cache_root=tmp_path / "cache",
        original_cache=_original_cache(),
    )

    assert resumed["external_attempts"] == 3
    assert resumed["cache_hits"] == 0


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
            self.last_grant_event = {
                "started_at_epoch": 1_700_000_000.0,
                "started_at_utc": "2023-11-14T22:13:20Z",
            }
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
    monkeypatch, tmp_path: Path,
) -> None:
    output = tmp_path / "retrieval"
    _run_synthetic_retrieval(
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
        _verify_against_synthetic_planning(monkeypatch, output)


def _verify_against_synthetic_planning(
    monkeypatch, output: Path, *, planning: dict[str, object] | None = None
) -> dict[str, object]:
    expected = _planning() if planning is None else planning
    monkeypatch.setattr(
        module,
        "_load_sealed_planning",
        lambda planning_dir, *, approved_original_cache_root: expected,
    )
    monkeypatch.setattr(module, "_load_original_caches", lambda plan, *, approved_root: _original_cache())
    return _verify_synthetic_retrieval(
        output,
        Path("/synthetic/planning"),
        approved_original_cache_root=Path(
            "/home/npatta01/data/competitions/trec_rag_2026/outputs/_retriever_cache/pyserini_remote"
        ),
    )


def test_verify_authenticates_external_planning_root(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "retrieval"
    _run_synthetic_retrieval(
        _planning(),
        output,
        _Transport(),
        clock=_Clock(),
        cache_root=tmp_path / "cache",
        original_cache=_original_cache(),
    )
    drifted = _planning()
    drifted["planning_root_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="sealed planning binding"):
        _verify_against_synthetic_planning(monkeypatch, output, planning=drifted)


def test_verify_rejects_resealed_zero_facet_artifact(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "retrieval"
    _run_synthetic_retrieval(
        _planning(),
        output,
        _Transport(),
        clock=_Clock(),
        cache_root=tmp_path / "cache",
        original_cache=_original_cache(),
    )
    original = [json.loads(line) for line in (output / "original_candidates.jsonl").read_text().splitlines()]
    (output / "facet_candidates.jsonl").write_bytes(b"")
    (output / "accepted_union.jsonl").write_bytes(module._jsonl_bytes(build_union(original, [])))
    summary = json.loads((output / "retrieval_summary.json").read_text())
    summary["facet_request_count"] = 0
    summary["facet_candidate_rows"] = 0
    summary["accepted_union_rows"] = len(original)
    (output / "retrieval_summary.json").write_bytes(module._pretty_bytes(summary))
    (output / "RETRIEVAL_SEALED.json").unlink()
    module._create_seal(output)

    with pytest.raises(ValueError, match="facet count|facet stream"):
        _verify_against_synthetic_planning(monkeypatch, output)


def test_verify_rejects_resealed_attempt_counter_drift(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "retrieval"
    _run_synthetic_retrieval(
        _planning(),
        output,
        _Transport(),
        clock=_Clock(),
        cache_root=tmp_path / "cache",
        original_cache=_original_cache(),
    )
    summary = json.loads((output / "retrieval_summary.json").read_text())
    summary["external_attempts"] = 0
    summary["cache_hits"] = 3
    (output / "retrieval_summary.json").write_bytes(module._pretty_bytes(summary))
    (output / "RETRIEVAL_SEALED.json").unlink()
    module._create_seal(output)

    with pytest.raises(ValueError, match="facet count or candidate counts"):
        _verify_against_synthetic_planning(monkeypatch, output)


def test_verify_reopens_original_cache_and_rejects_forged_rows(
    monkeypatch, tmp_path: Path
) -> None:
    planning = _planning()
    root = tmp_path / "approved-original-cache"
    root.mkdir()
    for sealed in planning["originals"]:
        topic_id = sealed["topic_id"]
        supplied = _original_cache()[topic_id]
        payload = {
            "topic_id": topic_id,
            "variant_name": "original",
            "hits": 1000,
            "index": module.ORIGINAL_INDEX,
            "index_url": module.ORIGINAL_INDEX_URL,
            "response": {
                "candidates": [
                    {
                        "docid": row["docid"],
                        "rank": row["rank"],
                        "score": row["score"],
                        "doc": row["text"],
                    }
                    for row in supplied["candidates"]
                ]
            },
        }
        raw = module._pretty_bytes(payload)
        path = root / f"{topic_id}.json"
        path.write_bytes(raw)
        sealed["cache_path"] = str(path)
        sealed["cache_sha256"] = _sha(raw)
        sealed["raw_response_provenance"]["cache_content_sha256"] = _sha(raw)
    monkeypatch.setattr(module, "APPROVED_ORIGINAL_CACHE_ROOT", root)
    source_cache = module._load_original_caches(planning, approved_root=root)
    output = tmp_path / "retrieval"
    _run_synthetic_retrieval(
        planning,
        output,
        _Transport(),
        clock=_Clock(),
        cache_root=tmp_path / "cache",
        original_cache=source_cache,
    )
    original_path = output / "original_candidates.jsonl"
    original = [json.loads(line) for line in original_path.read_text().splitlines()]
    original[0]["document_id"] = "forged-docid"
    original[0]["text"] = "forged passage"
    original[0]["score"] = 999999.0
    original_path.write_bytes(module._jsonl_bytes(original))
    facets = [json.loads(line) for line in (output / "facet_candidates.jsonl").read_text().splitlines()]
    (output / "accepted_union.jsonl").write_bytes(module._jsonl_bytes(build_union(original, facets)))
    (output / "RETRIEVAL_SEALED.json").unlink()
    module._create_seal(output)

    monkeypatch.setattr(
        module,
        "_load_sealed_planning",
        lambda planning_dir, *, approved_original_cache_root: planning,
    )
    with pytest.raises(ValueError, match="original cache candidate"):
        _verify_synthetic_retrieval(
            output,
            Path("/synthetic/planning"),
            approved_original_cache_root=root,
        )


def test_request_start_events_survive_cache_only_restart(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    first = tmp_path / "first"
    _run_synthetic_retrieval(
        _planning(), first, _Transport(), clock=_Clock(), cache_root=cache_root,
        original_cache=_original_cache(),
    )
    second = tmp_path / "second"
    _run_synthetic_retrieval(
        _planning(), second, _Transport(), clock=_Clock(), cache_root=cache_root,
        original_cache=_original_cache(),
    )

    result = _verify_against_synthetic_planning(monkeypatch, second)
    assert result["request_start_count"] == 3
    assert result["minimum_request_start_delta_seconds"] == 3.0
    metadata = [json.loads(path.read_text()) for path in sorted((second / "ledger" / "metadata").glob("*.json"))]
    assert all(row["limiter_granted_start_utc"].endswith("Z") for row in metadata)
    assert len({row["limiter_granted_request_key"] for row in metadata}) == 3


def test_request_start_verifier_allows_sub_millisecond_wall_clock_jitter() -> None:
    module._validate_request_start_deltas([2.9999120235443115, 3.0001])

    with pytest.raises(ValueError, match="3-second limiter interval"):
        module._validate_request_start_deltas([2.998])


def test_shared_cache_claim_allows_only_one_call_per_identity(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    transport = _Transport()
    barrier = threading.Barrier(2)

    def execute(name: str):
        barrier.wait()
        return _run_synthetic_retrieval(
            _planning(),
            tmp_path / name,
            transport,
            clock=_Clock(),
            cache_root=cache_root,
            original_cache=_original_cache(),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, ("race-a", "race-b")))

    assert all(result["complete"] is True for result in results)
    assert sorted(transport.identities) == ["facet-a", "facet-b", "facet-c"]


def test_public_run_rejects_injected_nonproduction_transport(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="exact RateLimitedFacetTransport"):
        run_retrieval(tmp_path / "planning", tmp_path / "output", _Transport())


def test_production_verify_rejects_synthetic_fake_timestamp_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "synthetic"
    _run_synthetic_retrieval(
        _planning(), output, _Transport(), clock=_Clock(),
        cache_root=tmp_path / "cache", original_cache=_original_cache(),
    )
    monkeypatch.setattr(
        module,
        "_load_sealed_planning",
        lambda planning_dir, *, approved_original_cache_root: _planning(),
    )
    monkeypatch.setattr(
        module,
        "_load_original_caches",
        lambda plan, *, approved_root: _original_cache(),
    )

    with pytest.raises(ValueError, match="production execution binding"):
        verify_retrieval(
            output,
            tmp_path / "planning",
            approved_original_cache_root=module.APPROVED_ORIGINAL_CACHE_ROOT,
        )


def test_separate_processes_deduplicate_each_request_identity(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)
    events = context.Value("i", 0)
    start = context.Event()
    queue = context.Queue()
    processes = [
        context.Process(
            target=_process_synthetic_run,
            args=(
                _planning(),
                _original_cache(),
                str(tmp_path / "cache"),
                str(tmp_path / f"process-{index}"),
                calls,
                events,
                start,
                queue,
            ),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0

    assert [queue.get(timeout=1) for _ in processes] == [("ok", True), ("ok", True)]
    assert calls.value == 3


def test_process_death_leaves_shared_pending_claim_nonretryable(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)
    events = context.Value("i", 0)
    start = context.Event()
    queue = context.Queue()
    process = context.Process(
        target=_process_synthetic_run,
        args=(
            _planning(),
            _original_cache(),
            str(tmp_path / "cache"),
            str(tmp_path / "dead-process"),
            calls,
            events,
            start,
            queue,
            True,
        ),
    )
    process.start()
    start.set()
    process.join(10)
    assert process.exitcode == 17
    assert calls.value == 1

    replacement = _Transport()
    with pytest.raises(ValueError, match="immutable shared pending attempt"):
        _run_synthetic_retrieval(
            _planning(),
            tmp_path / "replacement",
            replacement,
            clock=_Clock(),
            cache_root=tmp_path / "cache",
            original_cache=_original_cache(),
        )
    assert replacement.identities == []


def test_fresh_process_limiter_sessions_share_sqlite_and_grant_ledger(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    start = context.Event()
    queue = context.Queue()
    state = tmp_path / "rate-limit.sqlite"
    ledger = tmp_path / "grants.jsonl"
    request_keys = ("a" * 64, "b" * 64)
    processes = [
        context.Process(
            target=_process_limiter_grant,
            args=(str(state), str(ledger), request_key, start, queue),
        )
        for request_key in request_keys
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0
    events = [queue.get(timeout=1) for _ in processes]
    assert {event["request_key"] for event in events} == set(request_keys)
    chronological = sorted(float(event["started_at_epoch"]) for event in events)
    assert chronological[1] - chronological[0] >= 3.0
    assert module._read_limiter_grant_ledger(ledger) == sorted(
        events, key=lambda event: int(event["previous_event_sha256"] != "0" * 64)
    )
