from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import trec_rag.facet_aware_fusion_run as module
from trec_rag.det_sparse_ledger import (
    RawTransportResponse,
    RetrievalLedger,
    RetrievalLedgerError,
    RetrievalRequest,
)
from trec_rag.facet_aware_fusion_manifest import ANALYZER_FINGERPRINT_SHA256
from trec_rag.facet_aware_fusion_run import (
    ENDPOINT,
    INDEX_ID,
    LIMITER_STATE_PATH,
    MAX_EXTERNAL_REQUESTS,
    MIN_INTERVAL_SECONDS,
    RateLimitedFacetTransport,
    build_live_config,
    build_requests,
    create_preflight,
    execute_retrieval,
    main,
    preflight_retrieval,
)


ANALYZER_SHA256 = ANALYZER_FINGERPRINT_SHA256
TOPIC_IDS = ("233", "273", "161", "14")
FORBIDDEN_TOPIC_IDS = ("144", "213", "224", "407", "515", "200", "225", "707", "897")


def _manifest() -> dict[str, object]:
    facets: list[dict[str, object]] = []
    manifest_order = -1
    for topic_id, count in zip(TOPIC_IDS, (3, 7, 7, 7), strict=True):
        for topic_order in range(1, count + 1):
            manifest_order += 1
            facets.append(
                {
                    "topic_id": topic_id,
                    "facet_id": f"{topic_id}-f{topic_order:02d}",
                    "query": f"subject {topic_id} relation {topic_order}",
                    "obligation": f"relation {topic_order}",
                    "anchor_terms": [f"subject {topic_id}"],
                    "relation_terms": [f"relation {topic_order}"],
                    "wrong_domain_patterns": ["wrong domain"],
                    "bridge_terms": [],
                    "manifest_order": manifest_order,
                }
            )
    return {
        "schema_version": "facet-aware-fusion-manifest-v1",
        "experiment_id": "rag25_facet_aware_fusion_v1",
        "selection_salt": "rag25_facet_aware_fusion_v1",
        "eligible_topic_ids": list(TOPIC_IDS),
        "protected_topic_ids": ["144", "213", "224", "407", "515"],
        "prior_pilot_topic_ids": ["200", "225", "707", "897"],
        "topic_ids": list(TOPIC_IDS),
        "topic_selection": [],
        "topics": [],
        "facets": facets,
        "hashes": {"analyzer_fingerprint_sha256": ANALYZER_SHA256},
        "qrels_opened": False,
    }


@pytest.fixture(autouse=True)
def _accept_synthetic_manifest(monkeypatch):
    monkeypatch.setattr(
        module,
        "validate_manifest",
        lambda payload, *, cache_root: None,
    )


def _ledger(tmp_path: Path) -> RetrievalLedger:
    return RetrievalLedger(
        tmp_path / "retrieval",
        max_calls=MAX_EXTERNAL_REQUESTS,
        max_calls_per_topic=7,
        min_results=50,
        required_text_results=50,
    )


def _body(*, prefix: str = "doc", count: int = 100) -> bytes:
    return json.dumps(
        {
            "candidates": [
                {
                    "docid": f"{prefix}-{rank:03d}",
                    "rank": rank,
                    "score": 101.0 - rank,
                    "doc": {"contents": f"synthetic evidence passage {rank}"},
                }
                for rank in range(1, count + 1)
            ]
        },
        separators=(",", ":"),
    ).encode("utf-8")


def test_build_requests_is_exact_and_uses_manifest_analyzer_fingerprint():
    requests = build_requests(_manifest(), ENDPOINT)

    assert len(requests) == 24
    assert len({row.identity.request_key for row in requests}) == 24
    assert [row.identity.topic_id for row in requests].count("233") == 3
    assert all(row.identity.hits == 100 for row in requests)
    assert all(row.identity.index_url == ENDPOINT for row in requests)
    assert all(row.identity.index_id == INDEX_ID for row in requests)
    assert all(
        row.identity.analyzer_fingerprint_sha256 == ANALYZER_SHA256
        for row in requests
    )
    assert [row.query_text for row in requests] == [
        row["query"] for row in _manifest()["facets"]
    ]


def test_build_requests_passes_explicit_cache_root_to_manifest_validation(
    tmp_path,
    monkeypatch,
):
    cache_root = tmp_path / "source-cache"
    validations = []
    monkeypatch.setattr(
        module,
        "validate_manifest",
        lambda payload, *, cache_root: validations.append((payload, cache_root)),
    )

    requests = build_requests(_manifest(), ENDPOINT, cache_root=cache_root)

    assert len(requests) == 24
    assert validations == [(_manifest(), cache_root)]


@pytest.mark.parametrize("topic_id", FORBIDDEN_TOPIC_IDS)
def test_forbidden_topics_fail_before_request_construction(topic_id, monkeypatch):
    manifest = _manifest()
    manifest["facets"][0]["topic_id"] = topic_id
    constructed = []
    monkeypatch.setattr(
        module.RetrievalRequest,
        "from_query",
        lambda **kwargs: constructed.append(kwargs),
    )

    with pytest.raises(ValueError, match=f"topic {topic_id}"):
        build_requests(manifest, ENDPOINT)

    assert constructed == []


def test_build_requests_rejects_wrong_count_before_request_construction(monkeypatch):
    manifest = _manifest()
    manifest["facets"].append(copy.deepcopy(manifest["facets"][-1]))
    constructed = []
    monkeypatch.setattr(
        module.RetrievalRequest,
        "from_query",
        lambda **kwargs: constructed.append(kwargs),
    )

    with pytest.raises(ValueError, match="exactly 24"):
        build_requests(manifest, ENDPOINT)

    assert constructed == []


def test_build_requests_rejects_any_other_endpoint():
    with pytest.raises(ValueError, match="frozen endpoint"):
        build_requests(_manifest(), "https://example.test/search")


def test_live_config_freezes_persistent_three_second_limiter():
    config = build_live_config(api_token="test-token")

    assert config.index_url == ENDPOINT
    assert config.hits == 100
    assert config.min_interval_seconds == MIN_INTERVAL_SECONDS == 3.0
    assert config.burst == 1
    assert config.limiter_state_path == LIMITER_STATE_PATH


def test_transport_uses_persistent_limiter(monkeypatch):
    sentinel_session = object()
    calls = []

    def fake_limited_session(config):
        calls.append(config)
        return sentinel_session

    monkeypatch.setattr(module, "rate_limited_session", fake_limited_session)
    config = build_live_config(api_token=None)
    transport = RateLimitedFacetTransport(
        config,
        build_requests(_manifest(), ENDPOINT),
    )

    assert calls == [config]
    assert transport.session is sentinel_session


def test_transport_rejects_allowlist_with_wrong_analyzer_before_session(monkeypatch):
    requests = list(build_requests(_manifest(), ENDPOINT))
    original = requests[0]
    requests[0] = RetrievalRequest.from_query(
        topic_id=original.identity.topic_id,
        variant_name=original.identity.variant_name,
        query_text=original.query_text,
        index_url=ENDPOINT,
        index_id=INDEX_ID,
        hits=100,
        analyzer_fingerprint_sha256="b" * 64,
    )
    session_builds = []
    monkeypatch.setattr(
        module,
        "rate_limited_session",
        lambda config: session_builds.append(config),
    )

    with pytest.raises(ValueError, match="frozen identity"):
        RateLimitedFacetTransport(build_live_config(api_token=None), requests)

    assert session_builds == []


class _FakeResponse:
    status_code = 503
    headers = {"Content-Type": "application/json", "Retry-After": "30"}
    content = b'{"error":"exact upstream failure bytes"}'
    elapsed = timedelta(milliseconds=25)


class _FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse()


def test_transport_returns_http_failure_bytes_without_retry():
    session = _FakeSession()
    requests = build_requests(_manifest(), ENDPOINT)
    transport = RateLimitedFacetTransport(
        build_live_config(api_token="test-token"),
        requests,
        session=session,
    )

    result = transport(requests[0])

    assert result.status == 503
    assert result.body == _FakeResponse.content
    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url == ENDPOINT
    assert kwargs["params"] == {
        "query": requests[0].query_text,
        "hits": "100",
    }
    assert kwargs["allow_redirects"] is False
    assert kwargs["headers"] == {
        "Accept": "application/json",
        "Authorization": "Bearer test-token",
    }
    assert transport.one_shot_no_retry is True


def test_transport_rejects_nonallowlisted_request_before_http():
    session = _FakeSession()
    requests = build_requests(_manifest(), ENDPOINT)
    transport = RateLimitedFacetTransport(
        build_live_config(api_token=None),
        requests,
        session=session,
    )
    request = RetrievalRequest.from_query(
        topic_id="233",
        variant_name="facet_aware_fusion_v1:not-allowed",
        query_text="subject 233 different relation",
        index_url=ENDPOINT,
        index_id=INDEX_ID,
        hits=100,
        analyzer_fingerprint_sha256=ANALYZER_SHA256,
    )

    with pytest.raises(ValueError, match="allowlist"):
        transport(request)

    assert session.calls == []


def test_preflight_is_qrels_blind_and_does_not_reserve_calls(tmp_path):
    ledger = _ledger(tmp_path)

    payload = preflight_retrieval(_manifest(), ledger, ENDPOINT)

    assert payload["planned_requests"] == 24
    assert payload["minimum_start_span_seconds"] == 69.0
    assert payload["qrels_opened"] is False
    assert payload["projected_external_attempts"] == 24
    assert payload["max_external_attempts"] == 24
    assert payload["max_external_attempts_per_topic"] == 7
    assert ledger.validate_run().planned_requests == 0


@pytest.mark.parametrize(
    ("max_calls", "max_calls_per_topic", "message"),
    ((23, 7, "max_calls=24"), (24, 6, "max_calls_per_topic=7")),
)
def test_preflight_rejects_weaker_or_incompatible_ledger_budget(
    tmp_path,
    max_calls,
    max_calls_per_topic,
    message,
):
    ledger = RetrievalLedger(
        tmp_path / "retrieval",
        max_calls=max_calls,
        max_calls_per_topic=max_calls_per_topic,
        min_results=50,
        required_text_results=50,
    )

    with pytest.raises(ValueError, match=message):
        preflight_retrieval(_manifest(), ledger, ENDPOINT)


def test_preflight_file_is_create_only(tmp_path):
    ledger = _ledger(tmp_path)

    payload = create_preflight(_manifest(), ledger, ENDPOINT)

    stored = json.loads((ledger.run_dir / "preflight.json").read_text())
    assert stored == payload
    with pytest.raises(FileExistsError, match="preflight.json"):
        create_preflight(_manifest(), ledger, ENDPOINT)


def test_http_failure_is_raw_first_and_stops_without_retry(tmp_path):
    manifest = _manifest()
    ledger = _ledger(tmp_path)
    create_preflight(manifest, ledger, ENDPOINT)
    calls = []

    def transport(request):
        calls.append(request)
        return RawTransportResponse(
            503,
            {"Content-Type": "application/json"},
            b'{"error":"preserve me"}',
            0.01,
        )

    with pytest.raises(RetrievalLedgerError, match="HTTP status 503"):
        execute_retrieval(manifest, ledger, transport, ENDPOINT)

    assert len(calls) == 1
    request_key = calls[0].identity.request_key
    assert ledger.raw_path(request_key).read_bytes() == b'{"error":"preserve me"}'
    report = ledger.validate_run()
    assert report.external_calls == 1
    assert report.failures == 1
    assert report.pending == 0
    assert not (ledger.run_dir / "candidates.jsonl").exists()
    assert not (ledger.run_dir / "retrieval_summary.json").exists()


def test_execution_writes_normalized_candidates_and_summary_create_only(tmp_path):
    manifest = _manifest()
    ledger = _ledger(tmp_path)
    create_preflight(manifest, ledger, ENDPOINT)
    calls = []

    def transport(request):
        calls.append(request)
        return RawTransportResponse(
            200,
            {"Content-Type": "application/json"},
            _body(prefix=request.identity.variant_name),
            0.01,
        )

    summary = execute_retrieval(manifest, ledger, transport, ENDPOINT)

    assert len(calls) == 24
    assert summary["complete"] is True
    assert summary["planned_requests"] == 24
    assert summary["external_calls"] == 24
    assert summary["successes"] == 24
    assert summary["failures"] == summary["pending"] == 0
    assert summary["candidate_rows"] == 2400
    rows = [
        json.loads(line)
        for line in (ledger.run_dir / "candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 2400
    assert rows[0]["topic_id"] == "233"
    assert rows[0]["facet_id"] == "233-f01"
    assert rows[0]["rank"] == 1
    assert rows[0]["request_key"] == calls[0].identity.request_key
    assert (ledger.run_dir / "retrieval_summary.json").is_file()

    with pytest.raises(FileExistsError, match="retrieval_summary.json"):
        execute_retrieval(manifest, ledger, transport, ENDPOINT)
    assert len(calls) == 24


def test_run_cli_loads_repo_env_before_token_and_never_logs_it(
    tmp_path,
    monkeypatch,
    capsys,
):
    secret = "synthetic-secret-that-must-not-be-logged"
    repo_root = tmp_path / "repo"
    output = tmp_path / "recovery"
    ledger = SimpleNamespace(run_dir=output)
    events = []
    monkeypatch.delenv("PYSERINI_API_TOKEN", raising=False)
    monkeypatch.setattr(
        module,
        "find_repo_root",
        lambda start: events.append(("find_repo_root", start)) or repo_root,
        raising=False,
    )

    def fake_load_repo_env(root):
        events.append(("load_repo_env", root))
        os.environ["PYSERINI_API_TOKEN"] = secret

    monkeypatch.setattr(
        module,
        "load_repo_env",
        fake_load_repo_env,
        raising=False,
    )
    monkeypatch.setattr(module, "load_manifest", lambda path, *, cache_root: _manifest())
    monkeypatch.setattr(module, "_validated_facets", lambda payload, *, cache_root: ())
    monkeypatch.setattr(module, "_ledger", lambda path, *, cache_root: ledger)
    monkeypatch.setattr(module, "_require_fresh_final_outputs", lambda path: None)
    monkeypatch.setattr(
        module,
        "_read_preflight",
        lambda payload, run_ledger, endpoint, *, cache_root: ({}, output / "preflight.json"),
    )
    requests = build_requests(_manifest(), ENDPOINT)
    monkeypatch.setattr(
        module,
        "build_requests",
        lambda payload, endpoint, *, cache_root: requests,
    )

    def fake_build_live_config(*, api_token):
        events.append(("build_live_config", api_token))
        assert api_token == secret
        return SimpleNamespace(api_token=api_token)

    monkeypatch.setattr(module, "build_live_config", fake_build_live_config)
    monkeypatch.setattr(
        module,
        "RateLimitedFacetTransport",
        lambda config, allowed: SimpleNamespace(config=config, allowed=allowed),
    )
    monkeypatch.setattr(
        module,
        "execute_retrieval",
        lambda *args, **kwargs: {"complete": True, "planned_requests": 24},
    )

    assert main(
        [
            "run",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output",
            str(output),
            "--cache-root",
            str(tmp_path / "cache"),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert [event[0] for event in events] == [
        "find_repo_root",
        "load_repo_env",
        "build_live_config",
    ]
    assert secret not in captured.out
    assert secret not in captured.err


def _failed_401_run(tmp_path, manifest):
    failed = RetrievalLedger(
        tmp_path / "failed",
        max_calls=MAX_EXTERNAL_REQUESTS,
        max_calls_per_topic=7,
        min_results=50,
        required_text_results=50,
    )
    create_preflight(manifest, failed, ENDPOINT)

    def unauthorized(_request):
        return RawTransportResponse(
            401,
            {"Content-Type": "application/json"},
            b'{"error":"synthetic unauthorized"}',
            0.01,
        )

    with pytest.raises(RetrievalLedgerError, match="HTTP status 401"):
        execute_retrieval(manifest, failed, unauthorized, ENDPOINT)
    return failed


def test_recovery_preflight_binds_401_and_old_run_cannot_replay(tmp_path):
    manifest = _manifest()
    failed = _failed_401_run(tmp_path, manifest)
    recovery = RetrievalLedger(
        tmp_path / "recovery",
        max_calls=MAX_EXTERNAL_REQUESTS,
        max_calls_per_topic=7,
        min_results=50,
        required_text_results=50,
    )

    receipt = module.create_recovery_preflight(
        manifest,
        failed,
        recovery,
        ENDPOINT,
    )

    failed_report = failed.validate_run()
    failed_request = build_requests(manifest, ENDPOINT)[0]
    assert failed_report.external_calls == failed_report.failures == 1
    assert receipt["failed_request_key"] == failed_request.identity.request_key
    assert receipt["failed_response_sha256"] == hashlib.sha256(
        failed.raw_path(failed_request.identity.request_key).read_bytes()
    ).hexdigest()
    assert receipt["failed_http_status"] == 401
    assert receipt["root_cause"] == "repo_env_not_loaded_before_token_read_v1"
    assert receipt["authorized_batches"] == 1
    assert receipt["failed_run_replay_allowed"] is False
    assert (recovery.run_dir / "recovery_receipt.json").is_file()
    assert (recovery.run_dir / "preflight.json").is_file()

    old_calls = []
    with pytest.raises(ValueError, match="24-call global budget|stored preflight"):
        execute_retrieval(
            manifest,
            failed,
            lambda request: old_calls.append(request),
            ENDPOINT,
        )
    assert old_calls == []

    with pytest.raises(FileExistsError, match="recovery_receipt.json"):
        module.create_recovery_preflight(
            manifest,
            failed,
            recovery,
            ENDPOINT,
        )


def test_recovery_receipt_tampering_stops_before_transport(tmp_path):
    manifest = _manifest()
    failed = _failed_401_run(tmp_path, manifest)
    recovery = RetrievalLedger(
        tmp_path / "recovery",
        max_calls=MAX_EXTERNAL_REQUESTS,
        max_calls_per_topic=7,
        min_results=50,
        required_text_results=50,
    )
    module.create_recovery_preflight(manifest, failed, recovery, ENDPOINT)
    receipt_path = recovery.run_dir / "recovery_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["failed_response_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    calls = []

    with pytest.raises(ValueError, match="recovery receipt"):
        execute_retrieval(
            manifest,
            recovery,
            lambda request: calls.append(request),
            ENDPOINT,
        )

    assert calls == []


def test_recovery_preflight_cli_requires_separate_failed_output(tmp_path):
    args = module._parser().parse_args(
        [
            "recover-preflight",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--failed-output",
            str(tmp_path / "failed"),
            "--output",
            str(tmp_path / "recovery"),
            "--cache-root",
            str(tmp_path / "cache"),
        ]
    )

    assert args.command == "recover-preflight"
    assert args.failed_output == tmp_path / "failed"
    assert args.output == tmp_path / "recovery"
