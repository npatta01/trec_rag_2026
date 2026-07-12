from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from trec_rag.det_sparse_ledger import (
    RawTransportResponse,
    RetrievalLedger,
    RetrievalLedgerError,
    RetrievalRequest,
)
from trec_rag.facet_retrieval_control_manifest import (
    ControlManifestError,
    PROTECTED_TOPIC_IDS,
    build_control_manifest,
)
from trec_rag.facet_retrieval_control_run import (
    RETRIEVER_VERSION,
    RateLimitedControlTransport,
    build_control_live_config,
    build_control_requests,
    execute_control,
    main,
    preflight_control,
)


ENDPOINT = "http://api.example.test/v1/climbmix-400b/search"
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT
    / "reports"
    / "experiments"
    / "facet_retrieval_control_pilot_v1"
    / "manifest.json"
)


def _body(count: int = 100) -> bytes:
    return json.dumps(
        {
            "candidates": [
                {
                    "docid": f"doc-{rank:03d}",
                    "rank": rank,
                    "score": 101.0 - rank,
                    "doc": {"contents": f"synthetic evidence passage {rank}"},
                }
                for rank in range(1, count + 1)
            ]
        },
        separators=(",", ":"),
    ).encode()


def _ledger(tmp_path, *, max_calls: int = 12) -> RetrievalLedger:
    return RetrievalLedger(
        tmp_path / "run",
        max_calls=max_calls,
        max_calls_per_topic=6,
        min_results=50,
        required_text_results=50,
    )


def _protected_manifest(topic_id: str):
    manifest = build_control_manifest()
    first = replace(manifest.streams[0], topic_id=topic_id)
    return replace(manifest, streams=(first, *manifest.streams[1:]))


def _tampered_manifest(kind: str):
    manifest = build_control_manifest()
    first = manifest.streams[0]
    if kind == "topic":
        first = replace(first, topic_id="999")
    elif kind == "query":
        first = replace(first, reweighted_query=f"{first.reweighted_query} tampered")
    elif kind == "variant":
        first = replace(first, stream_id="f07a-tampered")
    elif kind == "bm25":
        arms = list(first.arms)
        arms[1] = replace(arms[1], k1=9.0, b=1.0)
        first = replace(first, arms=tuple(arms))
    else:
        raise AssertionError(f"unknown test mutation: {kind}")
    return replace(manifest, streams=(first, *manifest.streams[1:]))


def _cli_args(tmp_path, *, endpoint: str = ENDPOINT):
    return [
        "--manifest",
        str(MANIFEST_PATH),
        "--endpoint",
        endpoint,
        "--shared-cache",
        str(tmp_path / "cache"),
        "--output",
        str(tmp_path / "output"),
        "--limiter-state",
        str(tmp_path / "limit.sqlite"),
        "--min-interval-seconds",
        "10",
    ]


def test_requests_are_exactly_twelve_and_parameterized():
    requests = build_control_requests(build_control_manifest(), endpoint=ENDPOINT)

    assert len(requests) == 12
    assert sum(request.identity.topic_id == "225" for request in requests) == 6
    assert [request.identity.variant_name for request in requests[:3]] == [
        "facet_control_v1:W0:f07a",
        "facet_control_v1:W1:f07a",
        "facet_control_v1:W2:f07a",
    ]
    assert [
        (request.identity.bm25_k1, request.identity.bm25_b)
        for request in requests[:3]
    ] == [(0.9, 0.4), (0.4, 0.4), (0.4, 0.0)]
    assert all(request.identity.hits == 100 for request in requests)
    assert all(
        request.identity.retriever_version == RETRIEVER_VERSION
        for request in requests
    )
    assert all(request.identity.bm25_k1 is not None for request in requests)
    assert all(request.identity.bm25_b is not None for request in requests)


def test_thirteenth_planned_request_is_rejected_before_construction(monkeypatch):
    manifest = build_control_manifest()
    overfull = replace(manifest, streams=(*manifest.streams, manifest.streams[0]))
    constructed = []

    def hostile_constructor(**kwargs):
        constructed.append(kwargs)
        raise AssertionError("request construction must remain untouched")

    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_run.RetrievalRequest.from_query",
        hostile_constructor,
    )

    with pytest.raises(ValueError, match="12 planned"):
        build_control_requests(overfull, endpoint=ENDPOINT)
    assert constructed == []


@pytest.mark.parametrize("kind", ("topic", "query", "variant", "bm25"))
def test_build_rejects_any_exact_manifest_tampering_before_construction(
    kind,
    monkeypatch,
):
    constructed = []
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_run.RetrievalRequest.from_query",
        lambda **kwargs: constructed.append(kwargs),
    )

    with pytest.raises(ControlManifestError):
        build_control_requests(_tampered_manifest(kind), endpoint=ENDPOINT)
    assert constructed == []


@pytest.mark.parametrize("topic_id", PROTECTED_TOPIC_IDS)
def test_every_protected_topic_is_rejected_before_request_construction(
    topic_id,
    monkeypatch,
):
    constructed = []

    def hostile_constructor(**kwargs):
        constructed.append(kwargs)
        raise AssertionError("request construction must remain untouched")

    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_run.RetrievalRequest.from_query",
        hostile_constructor,
    )

    with pytest.raises(ValueError, match=f"protected topic {topic_id}"):
        build_control_requests(_protected_manifest(topic_id), endpoint=ENDPOINT)
    assert constructed == []


@pytest.mark.parametrize("topic_id", PROTECTED_TOPIC_IDS)
def test_every_protected_topic_is_rejected_before_cache_probe(topic_id):
    class HostileLedger:
        def has_verified_cache(self, _request):
            raise AssertionError("cache must remain untouched")

        def call_count(self):
            raise AssertionError("ledger attempts must remain untouched")

    with pytest.raises(ValueError, match=f"protected topic {topic_id}"):
        preflight_control(_protected_manifest(topic_id), HostileLedger(), ENDPOINT)


@pytest.mark.parametrize("kind", ("topic", "query", "variant", "bm25"))
def test_preflight_rejects_any_exact_manifest_tampering_before_cache_probe(kind):
    class HostileLedger:
        def has_verified_cache(self, _request):
            raise AssertionError("cache must remain untouched")

        def call_count(self):
            raise AssertionError("ledger attempts must remain untouched")

    with pytest.raises(ControlManifestError):
        preflight_control(_tampered_manifest(kind), HostileLedger(), ENDPOINT)


def test_preflight_probes_cache_without_recording_an_invocation(tmp_path):
    ledger = _ledger(tmp_path)

    preflight = preflight_control(build_control_manifest(), ledger, ENDPOINT)

    assert preflight["planned_requests"] == 12
    assert preflight["verified_cache_hits"] == 0
    assert preflight["projected_external_attempts"] == 12
    assert ledger.validate_run().planned_requests == 0


def test_thirteenth_projected_external_attempt_is_rejected():
    class LedgerWithPriorAttempt:
        def __init__(self):
            self.probes = []

        def has_verified_cache(self, request):
            self.probes.append(request)
            return False

        def call_count(self):
            return 1

    ledger = LedgerWithPriorAttempt()

    with pytest.raises(ValueError, match="12 external attempts"):
        preflight_control(build_control_manifest(), ledger, ENDPOINT)
    assert len(ledger.probes) == 12


def test_interval_below_ten_seconds_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="ten seconds"):
        build_control_live_config(
            ENDPOINT,
            None,
            tmp_path / "limit.sqlite",
            9.99,
        )


def test_endpoint_userinfo_is_rejected_before_output_or_session_creation(
    tmp_path,
    monkeypatch,
):
    session_calls = []
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_run.rate_limited_session",
        lambda config: session_calls.append(config),
    )

    with pytest.raises(ValueError, match="userinfo"):
        main(
            _cli_args(
                tmp_path,
                endpoint="https://user:password@api.example.test/search",
            )
        )

    assert session_calls == []
    assert not (tmp_path / "output").exists()


class _FakeResponse:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    content = b'{"candidates":[]}'
    elapsed = timedelta(milliseconds=25)


class _FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse()


def _forge_control_request(request, mutation):
    identity = request.identity
    query_text = request.query_text
    topic_id = identity.topic_id
    bm25_k1 = identity.bm25_k1
    if mutation == "topic":
        topic_id = "999"
    elif mutation == "protected":
        topic_id = PROTECTED_TOPIC_IDS[0]
    elif mutation == "k1":
        bm25_k1 = 9.0
    elif mutation == "query":
        query_text = f"{query_text} tampered"
    else:
        raise AssertionError(f"unknown request mutation: {mutation}")
    return RetrievalRequest.from_query(
        topic_id=topic_id,
        variant_name=identity.variant_name,
        query_text=query_text,
        index_url=identity.index_url,
        index_id=identity.index_id,
        hits=identity.hits,
        analyzer_fingerprint_sha256=identity.analyzer_fingerprint_sha256,
        retriever_version=identity.retriever_version,
        bm25_k1=bm25_k1,
        bm25_b=identity.bm25_b,
    )


@pytest.mark.parametrize("mutation", ("topic", "protected", "k1", "query"))
def test_transport_rejects_forged_twelve_entry_allowlist_before_session_creation(
    tmp_path,
    mutation,
    monkeypatch,
):
    allowed_requests = list(
        build_control_requests(build_control_manifest(), endpoint=ENDPOINT)
    )
    allowed_requests[0] = _forge_control_request(allowed_requests[0], mutation)
    session_builds = []
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_run.rate_limited_session",
        lambda config: session_builds.append(config),
    )

    with pytest.raises(ValueError, match="canonical"):
        RateLimitedControlTransport(
            build_control_live_config(
                ENDPOINT,
                None,
                tmp_path / "limit.sqlite",
                10.0,
            ),
            allowed_requests=tuple(allowed_requests),
        )
    assert session_builds == []


def test_transport_rejects_canonical_mapping_with_extra_duplicate_before_session(
    tmp_path,
    monkeypatch,
):
    canonical = build_control_requests(build_control_manifest(), endpoint=ENDPOINT)
    session_builds = []
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_run.rate_limited_session",
        lambda config: session_builds.append(config),
    )

    with pytest.raises(ValueError, match="canonical"):
        RateLimitedControlTransport(
            build_control_live_config(
                ENDPOINT,
                None,
                tmp_path / "limit.sqlite",
                10.0,
            ),
            allowed_requests=(*canonical, canonical[0]),
        )
    assert session_builds == []


def test_transport_sends_only_exact_explicit_http_parameters(tmp_path):
    session = _FakeSession()
    config = build_control_live_config(
        ENDPOINT,
        "test-token",
        tmp_path / "limit.sqlite",
        10.0,
    )
    allowed_requests = build_control_requests(
        build_control_manifest(),
        endpoint=ENDPOINT,
    )
    transport = RateLimitedControlTransport(
        config,
        allowed_requests=allowed_requests,
        session=session,
    )
    request = allowed_requests[2]

    result = transport(request)

    assert result.status == 200
    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url == ENDPOINT
    assert kwargs["params"] == {
        "query": request.query_text,
        "hits": "100",
        "k1": "0.4",
        "b": "0.0",
    }
    assert kwargs["allow_redirects"] is False
    assert kwargs["headers"] == {
        "Accept": "application/json",
        "Authorization": "Bearer test-token",
    }
    assert transport.one_shot_no_retry is True
    assert transport.redirects_allowed is False


def test_transport_rejects_protocol_valid_nonallowlisted_request_before_session(
    tmp_path,
):
    manifest = build_control_manifest()
    allowed_requests = build_control_requests(manifest, endpoint=ENDPOINT)
    session = _FakeSession()
    transport = RateLimitedControlTransport(
        build_control_live_config(
            ENDPOINT,
            None,
            tmp_path / "limit.sqlite",
            10.0,
        ),
        allowed_requests=allowed_requests,
        session=session,
    )
    request = RetrievalRequest.from_query(
        topic_id="200",
        variant_name="facet_control_v1:W2:generic-but-not-frozen",
        query_text="generic protocol valid query",
        index_url=ENDPOINT,
        index_id="climbmix-400b",
        hits=100,
        analyzer_fingerprint_sha256=manifest.analyzer_fingerprint_sha256,
        retriever_version=RETRIEVER_VERSION,
        bm25_k1=0.4,
        bm25_b=0.0,
    )

    with pytest.raises(ValueError, match="allowlist"):
        transport(request)
    assert session.calls == []


def test_first_failure_stops_without_retry_or_later_calls(tmp_path):
    manifest = build_control_manifest()
    ledger = _ledger(tmp_path)
    transport_calls = []

    def failing_transport(request):
        transport_calls.append(request)
        raise TimeoutError("synthetic timeout")

    with pytest.raises(RetrievalLedgerError, match="synthetic timeout"):
        execute_control(manifest, ledger, failing_transport, ENDPOINT)

    assert len(transport_calls) == 1
    report = ledger.validate_run()
    assert report.failures == 1
    assert report.external_calls == 1
    assert report.pending == 0


def test_successful_execution_is_exactly_twelve_create_only_invocations(tmp_path):
    manifest = build_control_manifest()
    ledger = _ledger(tmp_path)
    calls = []

    def transport(request):
        calls.append(request)
        return RawTransportResponse(200, {"X-Synthetic": "true"}, _body(), 0.01)

    summary = execute_control(manifest, ledger, transport, ENDPOINT)

    assert len(calls) == 12
    assert summary["complete"] is True
    assert summary["planned_requests"] == 12
    assert summary["external_calls"] == 12
    assert summary["failures"] == summary["pending"] == 0
    assert len(summary["manifest_sha256"]) == 64
    assert len(summary["runner_code_sha256"]) == 64
    assert (ledger.run_dir / "preflight.json").is_file()
    assert (ledger.run_dir / "retrieval_summary.json").is_file()

    with pytest.raises(FileExistsError):
        execute_control(manifest, ledger, transport, ENDPOINT)
    assert len(calls) == 12


def test_existing_summary_is_rejected_before_any_invocation(tmp_path):
    ledger = _ledger(tmp_path)
    (ledger.run_dir / "retrieval_summary.json").write_text("{}\n", encoding="utf-8")
    calls = []

    with pytest.raises(FileExistsError, match="retrieval_summary.json"):
        execute_control(
            build_control_manifest(),
            ledger,
            lambda request: calls.append(request),
            ENDPOINT,
        )

    assert calls == []
    assert not (ledger.run_dir / "preflight.json").exists()


def test_cli_rejects_existing_final_before_creating_output_or_session(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    summary_path = output / "retrieval_summary.json"
    real_exists = Path.exists
    session_calls = []

    def synthetic_collision(path):
        if path == summary_path:
            return True
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", synthetic_collision)
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_run.rate_limited_session",
        lambda config: session_calls.append(config),
    )

    with pytest.raises(FileExistsError, match="retrieval_summary.json"):
        main(_cli_args(tmp_path))

    assert session_calls == []
    assert not real_exists(output)


def test_preflight_only_stdout_then_execution_reuses_same_output(
    tmp_path,
    monkeypatch,
    capsys,
):
    transport_builds = []

    def fake_transport_factory(config, *, allowed_requests):
        transport_builds.append((config, allowed_requests))

        def transport(_request):
            return RawTransportResponse(
                200,
                {"X-Synthetic": "true"},
                _body(),
                0.01,
            )

        return transport

    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_run.RateLimitedControlTransport",
        fake_transport_factory,
    )
    args = _cli_args(tmp_path)
    output = tmp_path / "output"

    assert main([*args, "--preflight-only"]) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["planned_requests"] == 12
    assert transport_builds == []
    assert not (output / "preflight.json").exists()
    assert not (output / "retrieval_summary.json").exists()

    assert main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["complete"] is True
    assert len(transport_builds) == 1
    assert len(transport_builds[0][1]) == 12
    assert (output / "preflight.json").is_file()
    assert (output / "retrieval_summary.json").is_file()


@pytest.mark.parametrize("topic_id", PROTECTED_TOPIC_IDS)
def test_cli_rejects_protected_topic_before_transport_creation(
    tmp_path,
    topic_id,
    monkeypatch,
):
    session_calls = []
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_run.load_control_manifest",
        lambda _path: _protected_manifest(topic_id),
    )
    monkeypatch.setattr(
        "trec_rag.facet_retrieval_control_run.rate_limited_session",
        lambda _config: session_calls.append(_config),
    )

    with pytest.raises(ValueError, match=f"protected topic {topic_id}"):
        main(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--endpoint",
                ENDPOINT,
                "--shared-cache",
                str(tmp_path / "cache"),
                "--output",
                str(tmp_path / "output"),
                "--limiter-state",
                str(tmp_path / "limit.sqlite"),
                "--min-interval-seconds",
                "10",
            ]
        )
    assert session_calls == []
    assert not (tmp_path / "output").exists()
