import json
from pathlib import Path

from trec_rag.det_sparse_ledger import RawTransportResponse, RetrievalLedger
from trec_rag.sparse_relevance_manifest import (
    load_repair_manifest,
    validate_manifest_pair,
)
from trec_rag.sparse_relevance_run import (
    RateLimitedLedgerTransport,
    build_live_config,
    build_requests,
    execute_renderer,
    preflight_pair,
)
from trec_rag.sparse_relevance_experiment import load_verified_ledger_candidates
from trec_rag.det_sparse_ledger import RetrievalRequest


REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = REPO_ROOT / "reports" / "experiments" / "sparse_relevance_pilot_v1"
ENDPOINT = "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"


def _body():
    return json.dumps(
        {
            "candidates": [
                {
                    "rank": rank,
                    "docid": f"doc-{rank}",
                    "score": 101 - rank,
                    "doc": {"contents": f"Document text {rank}"},
                }
                for rank in range(1, 101)
            ]
        }
    ).encode()


def test_tracked_pair_builds_22_plain_text_requests_per_renderer():
    pair = validate_manifest_pair(
        load_repair_manifest(PILOT_DIR / "r0_manifest.json"),
        load_repair_manifest(PILOT_DIR / "r1_manifest.json"),
    )

    r0 = build_requests(pair.r0, endpoint=ENDPOINT)
    r1 = build_requests(pair.r1, endpoint=ENDPOINT)

    assert len(r0) == len(r1) == 22
    assert {row.identity.topic_id for row in r0} == {"200", "225", "707", "897"}
    assert all(row.identity.hits == 100 for row in (*r0, *r1))
    assert all(row.identity.index_url == ENDPOINT for row in (*r0, *r1))
    assert all(row.identity.variant_name.startswith("sparse_relevance_v1:R0:") for row in r0)
    assert all(row.identity.variant_name.startswith("sparse_relevance_v1:R1:") for row in r1)


def test_pair_preflight_counts_verified_cache_without_reservations(tmp_path):
    pair = validate_manifest_pair(
        load_repair_manifest(PILOT_DIR / "r0_manifest.json"),
        load_repair_manifest(PILOT_DIR / "r1_manifest.json"),
    )

    report = preflight_pair(
        pair,
        endpoint=ENDPOINT,
        shared_cache_dir=tmp_path / "cache",
        probe_dir=tmp_path / "probe",
    )

    assert report == {
        "R0": {"planned": 22, "verified_cache_hits": 0, "external_calls": 22},
        "R1": {"planned": 22, "verified_cache_hits": 0, "external_calls": 22},
        "total": {"planned": 44, "verified_cache_hits": 0, "external_calls": 44},
    }
    assert not list((tmp_path / "probe" / "R0" / "attempts").glob("*.json"))


def test_execute_renderer_uses_one_separate_22_call_ledger(tmp_path):
    manifest = load_repair_manifest(PILOT_DIR / "r0_manifest.json")
    ledger = RetrievalLedger(
        tmp_path / "run",
        shared_cache_dir=tmp_path / "cache",
        max_calls=22,
        max_calls_per_topic=9,
    )
    calls = []

    def transport(request):
        calls.append(request.identity.request_key)
        return RawTransportResponse(200, {"Content-Type": "application/json"}, _body(), 0.01)

    results = execute_renderer(
        manifest,
        ledger=ledger,
        transport=transport,
        endpoint=ENDPOINT,
    )

    assert len(results) == len(calls) == 22
    assert all(not result.cache_hit for result in results)
    validation = ledger.validate_run()
    assert validation.successes == validation.external_calls == 22
    assert validation.failures == validation.pending == 0
    assert validation.per_topic_external_calls == {"200": 9, "225": 5, "707": 1, "897": 7}


def test_continuation_config_uses_larger_persistent_rate_window(tmp_path):
    config = build_live_config(
        endpoint=ENDPOINT,
        api_token="token",
        limiter_state=tmp_path / "rate.sqlite",
        min_interval_seconds=10.0,
    )

    assert config.min_interval_seconds == 10.0
    assert config.burst == 1
    assert config.limiter_state_path == tmp_path / "rate.sqlite"


def test_rate_limited_transport_sends_explicit_bm25_settings(monkeypatch, tmp_path):
    calls = []

    class FakeResponse:
        status_code = 200
        headers = {"Content-Type": "application/json"}
        content = _body()

    class FakeSession:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

    monkeypatch.setattr(
        "trec_rag.sparse_relevance_run.rate_limited_session",
        lambda _config: FakeSession(),
    )
    config = build_live_config(
        endpoint=ENDPOINT,
        api_token="token",
        limiter_state=tmp_path / "rate.sqlite",
        min_interval_seconds=10.0,
    )
    transport = RateLimitedLedgerTransport(config)
    query = "weighted query"
    request = RetrievalRequest.from_query(
        topic_id="200",
        variant_name="test:weighted",
        query_text=query,
        index_url=ENDPOINT,
        index_id="climbmix-400b",
        hits=100,
        analyzer_fingerprint_sha256="a" * 64,
        bm25_k1=0.4,
        bm25_b=0.0,
    )

    transport(request)

    assert calls[0][1]["params"] == {
        "query": query,
        "hits": "100",
        "k1": "0.4",
        "b": "0.0",
    }


def test_load_verified_ledger_candidates_reads_cache_and_external_rows(tmp_path):
    shared = tmp_path / "cache"
    run = tmp_path / "run"
    request = RetrievalRequest.from_query(
        topic_id="200",
        variant_name="test:original",
        query_text="Holocaust history",
        index_url=ENDPOINT,
        index_id="climbmix-400b",
        hits=100,
        analyzer_fingerprint_sha256="a" * 64,
    )
    ledger = RetrievalLedger(
        run,
        shared_cache_dir=shared,
        max_calls=1,
        max_calls_per_topic=1,
    )
    ledger.retrieve(
        request,
        lambda _request: RawTransportResponse(200, {}, _body(), 0.01),
    )

    rows = load_verified_ledger_candidates(
        run_dir=run,
        shared_cache_dir=shared,
        max_calls=1,
        max_calls_per_topic=1,
    )

    assert len(rows) == 100
    assert {row.variant_name for row in rows} == {"test:original"}
    assert rows[0].docid == "doc-1"
