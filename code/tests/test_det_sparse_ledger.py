from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from trec_rag import det_sparse_ledger
from trec_rag.det_sparse_ledger import (
    CallBudgetExceeded,
    LedgerIntegrityError,
    RawTransportResponse,
    ReplayRefused,
    ResponseValidationError,
    RetrievalLedger,
    RetrievalLedgerError,
    RetrievalRequest,
    RetrievalRequestIdentity,
    canonical_request_key,
    normalize_response_candidates,
)


ANALYZER_SHA = "a" * 64


def _request(
    number: int = 1,
    *,
    query: str | None = None,
    hits: int = 100,
) -> RetrievalRequest:
    return RetrievalRequest.from_query(
        topic_id=f"topic-{number}",
        variant_name=f"facet-{number}",
        query_text=query or f"exact query {number}",
        index_url="https://index.example/v1/clueweb/search",
        index_id="clueweb22-b",
        hits=hits,
        analyzer_fingerprint_sha256=ANALYZER_SHA,
    )


def _body(count: int = 50, *, empty_text_at: int | None = None) -> bytes:
    rows = []
    for rank in range(1, count + 1):
        text = "" if rank == empty_text_at else f"Document text {rank}"
        rows.append(
            {
                "rank": rank,
                "docid": f"doc-{rank}",
                "score": 100.0 - rank,
                "doc": {"contents": text},
            }
        )
    return json.dumps({"candidates": rows}, separators=(",", ":")).encode()


def _transport(body: bytes | None = None, *, calls: list[str] | None = None):
    def call(request: RetrievalRequest) -> RawTransportResponse:
        if calls is not None:
            calls.append(request.identity.request_key)
        return RawTransportResponse(
            status=200,
            headers={"Content-Type": "application/json", "X-Test": "yes"},
            body=body if body is not None else _body(),
            elapsed_seconds=0.25,
        )

    return call


def test_request_identity_hashes_exact_query_and_all_retrieval_fields():
    request = _request(query="  Exact query, including spaces.  ")
    expected_query_sha = hashlib.sha256(request.query_text.encode()).hexdigest()

    assert request.identity.query_sha256 == expected_query_sha
    assert canonical_request_key(request.identity) == request.identity.request_key
    assert len(request.identity.request_key) == 64
    assert _request(query="Exact query, including spaces.").identity.request_key != (
        request.identity.request_key
    )
    assert _request(query=request.query_text, hits=101).identity.request_key != (
        request.identity.request_key
    )
    assert RetrievalRequest.from_query(
        topic_id=request.identity.topic_id,
        variant_name=request.identity.variant_name,
        query_text=request.query_text,
        index_url=request.identity.index_url,
        index_id=request.identity.index_id,
        hits=request.identity.hits,
        analyzer_fingerprint_sha256=ANALYZER_SHA,
        retriever_version="future-retriever-v2",
    ).identity.request_key != request.identity.request_key

    mismatched = RetrievalRequestIdentity.from_query(
        topic_id="topic-1",
        variant_name="facet-1",
        query_text="different",
        index_url=request.identity.index_url,
        index_id=request.identity.index_id,
        hits=request.identity.hits,
        analyzer_fingerprint_sha256=ANALYZER_SHA,
    )
    with pytest.raises(ValueError, match="does not match"):
        RetrievalRequest(identity=mismatched, query_text=request.query_text)


def test_blank_query_and_impossible_hits_are_rejected_before_reservation(tmp_path: Path):
    with pytest.raises(ValueError, match="query_text"):
        _request(query="   ")

    request = _request(hits=1)
    ledger = RetrievalLedger(tmp_path / "run")
    calls: list[str] = []
    with pytest.raises(ResponseValidationError, match="before reservation"):
        ledger.retrieve(request, _transport(_body(1), calls=calls))
    assert calls == []
    assert ledger.call_count() == 0


def test_success_reserves_before_transport_and_persists_raw_before_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request = _request()
    ledger = RetrievalLedger(
        tmp_path / "run",
        shared_cache_dir=tmp_path / "cache",
    )
    events: list[str] = []
    original_decode = det_sparse_ledger._decode_and_normalize

    def observing_decode(raw: bytes, **kwargs):
        events.append("decode")
        assert ledger.raw_path(request.identity.request_key).read_bytes() == raw
        metadata = json.loads(
            ledger.raw_metadata_path(request.identity.request_key).read_text()
        )
        assert metadata["capture_level"] == (
            "exact_http_body_before_utf8_decode_or_json_parse"
        )
        return original_decode(raw, **kwargs)

    monkeypatch.setattr(det_sparse_ledger, "_decode_and_normalize", observing_decode)

    def transport(current: RetrievalRequest) -> RawTransportResponse:
        events.append("transport")
        assert ledger.reservation_path(current.identity.request_key).is_file()
        assert ledger.call_count() == 1
        return RawTransportResponse(200, {"X-Test": "yes"}, _body(), 0.1)

    result = ledger.retrieve(request, transport)

    assert events == ["transport", "decode"]
    reservation = json.loads(
        ledger.reservation_path(request.identity.request_key).read_text()
    )
    assert reservation["query_text"] == request.query_text
    assert reservation["identity"]["query_sha256"] == hashlib.sha256(
        request.query_text.encode()
    ).hexdigest()
    assert result.cache_hit is False
    assert result.external_calls == 1
    assert len(result.candidates) == 50
    assert result.response_sha256 == hashlib.sha256(_body()).hexdigest()
    assert ledger.validate_run().successes == 1
    assert ledger.load_verified_result(request) == result


def test_exact_verified_shared_cache_hit_costs_zero_and_makes_no_reservation(
    tmp_path: Path,
):
    request = _request()
    shared = tmp_path / "shared"
    first = RetrievalLedger(tmp_path / "run-1", shared_cache_dir=shared)
    live = first.retrieve(request, _transport())
    second = RetrievalLedger(tmp_path / "run-2", shared_cache_dir=shared)

    def forbidden(_request: RetrievalRequest) -> RawTransportResponse:
        raise AssertionError("a cache hit must not enter the transport")

    cached = second.retrieve(request, forbidden)

    assert cached.cache_hit is True
    assert cached.external_calls == 0
    assert cached.candidates == live.candidates
    assert cached.response_sha256 == live.response_sha256
    assert cached.candidates_sha256 == live.candidates_sha256
    assert second.call_count() == 0
    evidence = json.loads(second.cache_hit_path(request.identity.request_key).read_text())
    assert evidence["query_text"] == request.query_text
    assert evidence["external_calls"] == 0
    validation = second.validate_run()
    assert validation.cache_hits == 1
    assert validation.external_calls == 0
    assert validation.planned_requests == 1
    assert second.load_verified_result(request) == cached


def test_same_key_concurrent_cache_miss_makes_only_one_billed_call(tmp_path: Path):
    request = _request()
    shared = tmp_path / "shared"
    first = RetrievalLedger(tmp_path / "run-1", shared_cache_dir=shared)
    second = RetrievalLedger(tmp_path / "run-2", shared_cache_dir=shared)
    transport_entered = threading.Event()
    release_transport = threading.Event()
    second_lock_requested = threading.Event()
    calls = 0
    original_second_lock = second._shared_cache_lock

    def observed_second_lock(request_key: str):
        second_lock_requested.set()
        return original_second_lock(request_key)

    second._shared_cache_lock = observed_second_lock  # type: ignore[method-assign]

    def transport(_request: RetrievalRequest) -> RawTransportResponse:
        nonlocal calls
        calls += 1
        transport_entered.set()
        assert release_transport.wait(timeout=5)
        return RawTransportResponse(200, {}, _body(), 0.1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first.retrieve, request, transport)
        assert transport_entered.wait(timeout=5)
        second_future = executor.submit(second.retrieve, request, transport)
        assert second_lock_requested.wait(timeout=5)
        release_transport.set()
        results = (first_future.result(timeout=5), second_future.result(timeout=5))

    assert calls == 1
    assert sorted(result.cache_hit for result in results) == [False, True]
    assert first.call_count() == 1
    assert second.call_count() == 0
    assert second.validate_run().cache_hits == 1


def test_same_cache_hit_cannot_be_counted_twice_in_one_run(tmp_path: Path):
    request = _request()
    shared = tmp_path / "shared"
    RetrievalLedger(tmp_path / "seed", shared_cache_dir=shared).retrieve(
        request, _transport()
    )
    ledger = RetrievalLedger(tmp_path / "run", shared_cache_dir=shared)
    ledger.retrieve(request, _transport())

    with pytest.raises(ReplayRefused, match="recorded cache-hit"):
        ledger.retrieve(request, _transport())
    assert ledger.validate_run().planned_requests == 1


def test_cache_hit_external_call_count_is_integrity_checked(tmp_path: Path):
    request = _request()
    shared = tmp_path / "shared"
    RetrievalLedger(tmp_path / "seed", shared_cache_dir=shared).retrieve(
        request, _transport()
    )
    ledger = RetrievalLedger(tmp_path / "run", shared_cache_dir=shared)
    ledger.retrieve(request, _transport())
    evidence_path = ledger.cache_hit_path(request.identity.request_key)
    evidence = json.loads(evidence_path.read_text())
    evidence["external_calls"] = 99
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(LedgerIntegrityError, match="not zero"):
        ledger.validate_run()


def test_cache_identity_must_match_even_if_stored_under_requested_key(tmp_path: Path):
    request = _request()
    shared = tmp_path / "shared"
    RetrievalLedger(tmp_path / "run-1", shared_cache_dir=shared).retrieve(
        request, _transport()
    )
    ledger = RetrievalLedger(tmp_path / "run-2", shared_cache_dir=shared)
    _, _, manifest_path = ledger._cache_paths(request.identity.request_key)
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["index_id"] = "different-index"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(LedgerIntegrityError, match="identity"):
        ledger.retrieve(request, _transport())
    assert ledger.call_count() == 0


@pytest.mark.parametrize("artifact", ["raw", "candidates", "manifest"])
def test_partial_shared_cache_fails_closed_without_external_call(
    tmp_path: Path,
    artifact: str,
):
    request = _request()
    shared = tmp_path / "shared"
    first = RetrievalLedger(tmp_path / "run-1", shared_cache_dir=shared)
    first.retrieve(request, _transport())
    second = RetrievalLedger(tmp_path / "run-2", shared_cache_dir=shared)
    paths = dict(
        zip(("raw", "candidates", "manifest"), second._cache_paths(request.identity.request_key))
    )
    paths[artifact].unlink()
    calls: list[str] = []

    with pytest.raises(LedgerIntegrityError, match="partial"):
        second.retrieve(request, _transport(calls=calls))
    assert calls == []
    assert second.call_count() == 0


@pytest.mark.parametrize("artifact", ["raw", "candidates"])
def test_tampered_shared_cache_hashes_fail_closed(tmp_path: Path, artifact: str):
    request = _request()
    shared = tmp_path / "shared"
    first = RetrievalLedger(tmp_path / "run-1", shared_cache_dir=shared)
    first.retrieve(request, _transport())
    second = RetrievalLedger(tmp_path / "run-2", shared_cache_dir=shared)
    raw_path, candidate_path, _ = second._cache_paths(request.identity.request_key)
    target = raw_path if artifact == "raw" else candidate_path
    target.write_bytes(target.read_bytes() + b"tamper")
    calls: list[str] = []

    with pytest.raises(LedgerIntegrityError):
        second.retrieve(request, _transport(calls=calls))
    assert calls == []
    assert second.call_count() == 0


def test_coordinated_candidate_and_manifest_tamper_still_disagrees_with_raw(
    tmp_path: Path,
):
    request = _request()
    shared = tmp_path / "shared"
    first = RetrievalLedger(tmp_path / "run-1", shared_cache_dir=shared)
    first.retrieve(request, _transport())
    second = RetrievalLedger(tmp_path / "run-2", shared_cache_dir=shared)
    _, candidate_path, manifest_path = second._cache_paths(request.identity.request_key)
    candidates = json.loads(candidate_path.read_text())
    candidates[0]["text"] = "coordinated tamper"
    tampered = json.dumps(
        candidates, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    candidate_path.write_bytes(tampered)
    manifest = json.loads(manifest_path.read_text())
    manifest["candidates_sha256"] = hashlib.sha256(tampered).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(LedgerIntegrityError, match="do not reconstruct"):
        second.retrieve(request, _transport())
    assert second.call_count() == 0


def test_valid_cache_cannot_mask_missing_local_success_artifact(tmp_path: Path):
    request = _request()
    shared = tmp_path / "shared"
    ledger = RetrievalLedger(tmp_path / "run", shared_cache_dir=shared)
    ledger.retrieve(request, _transport())
    ledger.raw_path(request.identity.request_key).unlink()
    calls: list[str] = []

    with pytest.raises(LedgerIntegrityError, match="incomplete"):
        ledger.retrieve(request, _transport(calls=calls))
    assert calls == []


def test_coordinated_local_candidate_and_outcome_tamper_disagrees_with_raw(
    tmp_path: Path,
):
    request = _request()
    ledger = RetrievalLedger(tmp_path / "run")
    ledger.retrieve(request, _transport())
    candidate_path = ledger.candidates_path(request.identity.request_key)
    candidates = json.loads(candidate_path.read_text())
    candidates[0]["text"] = "coordinated local tamper"
    tampered = json.dumps(
        candidates, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    candidate_path.write_bytes(tampered)
    outcome_path = ledger.outcome_path(request.identity.request_key)
    outcome = json.loads(outcome_path.read_text())
    outcome["candidates_sha256"] = hashlib.sha256(tampered).hexdigest()
    outcome_path.write_text(json.dumps(outcome))

    with pytest.raises(LedgerIntegrityError, match="do not reconstruct"):
        ledger.validate_run()


def test_transport_failure_is_one_counted_attempt_and_replay_is_refused(tmp_path: Path):
    request = _request()
    ledger = RetrievalLedger(tmp_path / "run", min_results=1)
    calls = 0

    def failing(_request: RetrievalRequest) -> RawTransportResponse:
        nonlocal calls
        calls += 1
        raise TimeoutError("timed out")

    with pytest.raises(RetrievalLedgerError, match="timed out"):
        ledger.retrieve(request, failing)
    assert calls == 1
    assert ledger.call_count() == 1
    assert ledger.validate_run().failures == 1

    with pytest.raises(ReplayRefused, match="already failed"):
        ledger.retrieve(request, failing)
    assert calls == 1


def test_failure_outcome_requires_boolean_raw_state_and_no_candidates(tmp_path: Path):
    request = _request()
    ledger = RetrievalLedger(tmp_path / "run")

    def failing(_request: RetrievalRequest) -> RawTransportResponse:
        raise TimeoutError("timed out")

    with pytest.raises(RetrievalLedgerError):
        ledger.retrieve(request, failing)
    outcome_path = ledger.outcome_path(request.identity.request_key)
    outcome = json.loads(outcome_path.read_text())
    outcome["raw_present"] = "false"
    outcome_path.write_text(json.dumps(outcome))
    with pytest.raises(LedgerIntegrityError, match="must be boolean"):
        ledger.validate_run()

    outcome["raw_present"] = False
    outcome_path.write_text(json.dumps(outcome))
    assert ledger.validate_run().failures == 1
    ledger.candidates_path(request.identity.request_key).write_text("[]")
    with pytest.raises(LedgerIntegrityError, match="must not contain"):
        ledger.validate_run()


def test_tampered_exact_query_in_reservation_fails_hash_validation(tmp_path: Path):
    request = _request()
    ledger = RetrievalLedger(tmp_path / "run", min_results=1)

    def crash(_request: RetrievalRequest) -> RawTransportResponse:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ledger.retrieve(request, crash)
    path = ledger.reservation_path(request.identity.request_key)
    reservation = json.loads(path.read_text())
    reservation["query_text"] = "different query"
    path.write_text(json.dumps(reservation))

    with pytest.raises(LedgerIntegrityError, match="query text/hash mismatch"):
        ledger.call_count()


def test_crashed_pending_attempt_counts_and_refuses_replay(tmp_path: Path):
    request = _request()
    ledger = RetrievalLedger(tmp_path / "run", min_results=1)

    def crash(_request: RetrievalRequest) -> RawTransportResponse:
        raise KeyboardInterrupt("simulated process death")

    with pytest.raises(KeyboardInterrupt):
        ledger.retrieve(request, crash)
    assert ledger.call_count() == 1
    assert ledger.validate_run().pending == 1

    with pytest.raises(ReplayRefused, match="pending/crashed"):
        ledger.retrieve(request, _transport(_body(1)))


def test_crashed_attempt_consumes_the_final_call_slot(tmp_path: Path):
    ledger = RetrievalLedger(
        tmp_path / "run", max_calls=1, min_results=1
    )

    def crash(_request: RetrievalRequest) -> RawTransportResponse:
        raise KeyboardInterrupt("simulated process death")

    with pytest.raises(KeyboardInterrupt):
        ledger.retrieve(_request(1), crash)
    calls: list[str] = []
    with pytest.raises(CallBudgetExceeded, match="1"):
        ledger.retrieve(_request(2), _transport(_body(1), calls=calls))
    assert calls == []


def test_raw_only_crash_is_preserved_but_fails_closed(tmp_path: Path, monkeypatch):
    request = _request()
    exact = b"not decoded before disk"
    ledger = RetrievalLedger(tmp_path / "run", min_results=1)

    def crash_after_raw(_raw: bytes, **_kwargs):
        raise KeyboardInterrupt("crash after raw commit")

    monkeypatch.setattr(det_sparse_ledger, "_decode_and_normalize", crash_after_raw)
    with pytest.raises(KeyboardInterrupt):
        ledger.retrieve(request, _transport(exact))

    assert ledger.raw_path(request.identity.request_key).read_bytes() == exact
    with pytest.raises(LedgerIntegrityError, match="raw-only"):
        ledger.validate_run()
    with pytest.raises(ReplayRefused, match="pending/crashed"):
        ledger.retrieve(request, _transport(_body(1)))


def test_invalid_utf8_is_raw_first_failure_and_is_never_retried(tmp_path: Path):
    request = _request()
    exact = b"\xff\xfe\x00"
    ledger = RetrievalLedger(tmp_path / "run", min_results=1)

    with pytest.raises(ResponseValidationError, match="UTF-8"):
        ledger.retrieve(request, _transport(exact))

    assert ledger.raw_path(request.identity.request_key).read_bytes() == exact
    outcome = json.loads(ledger.outcome_path(request.identity.request_key).read_text())
    assert outcome["status"] == "failure"
    assert outcome["raw_present"] is True
    assert outcome["response_sha256"] == hashlib.sha256(exact).hexdigest()
    with pytest.raises(ReplayRefused):
        ledger.retrieve(request, _transport(_body(1)))


def test_non_success_http_status_is_preserved_raw_and_never_parsed(tmp_path: Path):
    request = _request()
    exact = b"gateway unavailable"
    ledger = RetrievalLedger(tmp_path / "run", min_results=1)

    def transport(_request: RetrievalRequest) -> RawTransportResponse:
        return RawTransportResponse(503, {"Retry-After": "1"}, exact, 0.2)

    with pytest.raises(ResponseValidationError, match="503"):
        ledger.retrieve(request, transport)
    assert ledger.raw_path(request.identity.request_key).read_bytes() == exact
    metadata = json.loads(
        ledger.raw_metadata_path(request.identity.request_key).read_text()
    )
    assert metadata["http_status"] == 503
    assert not ledger.candidates_path(request.identity.request_key).exists()


def test_hard_call_ceiling_stops_before_call_37_and_survives_reopen(tmp_path: Path):
    run_dir = tmp_path / "run"
    ledger = RetrievalLedger(run_dir, min_results=1)
    calls: list[str] = []
    transport = _transport(_body(1), calls=calls)

    for number in range(1, 37):
        ledger.retrieve(_request(number), transport)

    with pytest.raises(CallBudgetExceeded, match="36"):
        ledger.retrieve(_request(37), transport)
    assert len(calls) == 36
    assert ledger.call_count() == 36
    assert not ledger.reservation_path(_request(37).identity.request_key).exists()

    reopened = RetrievalLedger(run_dir, min_results=1)
    with pytest.raises(CallBudgetExceeded):
        reopened.retrieve(_request(38), transport)
    assert len(calls) == 36


def test_per_topic_ceiling_stops_before_tenth_call(tmp_path: Path):
    ledger = RetrievalLedger(tmp_path / "run", min_results=1)
    transport = _transport(_body(1))
    requests = [
        RetrievalRequest.from_query(
            topic_id="same-topic",
            variant_name=f"variant-{number}",
            query_text=f"query {number}",
            index_url="https://index.example/search",
            index_id="index",
            hits=100,
            analyzer_fingerprint_sha256=ANALYZER_SHA,
        )
        for number in range(1, 11)
    ]

    for request in requests[:9]:
        ledger.retrieve(request, transport)
    with pytest.raises(CallBudgetExceeded, match="per-topic.*9"):
        ledger.retrieve(requests[9], transport)

    report = ledger.validate_run()
    assert report.per_topic_external_calls == {"same-topic": 9}


def test_run_policy_is_frozen_so_ceiling_cannot_be_raised_afterward(tmp_path: Path):
    run_dir = tmp_path / "run"
    RetrievalLedger(run_dir)

    with pytest.raises(LedgerIntegrityError, match="frozen"):
        RetrievalLedger(run_dir, max_calls=35)


def test_absolute_call_ceiling_cannot_be_configured_above_36(tmp_path: Path):
    with pytest.raises(ValueError, match="hard ceiling 36"):
        RetrievalLedger(tmp_path / "run", max_calls=37)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "at least 50"),
        (lambda rows: rows[0].update(docid=""), "docid"),
        (lambda rows: rows[1].update(rank=1), "one-based position"),
        (lambda rows: rows[2].update(docid="doc-1"), "duplicate candidate docid"),
        (lambda rows: rows[49]["doc"].update(contents=""), "empty text"),
    ],
)
def test_candidate_admission_rejects_short_or_malformed_top_50(mutation, message):
    response = json.loads(_body())
    mutation(response["candidates"])

    with pytest.raises(ResponseValidationError, match=message):
        normalize_response_candidates(response)


def test_empty_text_is_allowed_only_below_required_top_50():
    response = json.loads(_body(51, empty_text_at=51))

    candidates = normalize_response_candidates(response)

    assert len(candidates) == 51
    assert candidates[50].text == ""


def test_response_depth_above_100_is_rejected_after_raw_capture(tmp_path: Path):
    request = _request()
    ledger = RetrievalLedger(tmp_path / "run")

    with pytest.raises(ResponseValidationError, match="at most 100"):
        ledger.retrieve(request, _transport(_body(101)))

    assert ledger.raw_path(request.identity.request_key).read_bytes() == _body(101)


def test_reordered_declared_ranks_are_rejected():
    response = json.loads(_body(51, empty_text_at=51))
    response["candidates"][0]["rank"] = 51
    response["candidates"][50]["rank"] = 1

    with pytest.raises(ResponseValidationError, match="one-based position"):
        normalize_response_candidates(response)


def test_docid_and_rank_metadata_do_not_masquerade_as_document_text():
    response = json.loads(_body())
    response["candidates"][0] = {"docid": "metadata-only", "rank": 1, "score": 1.0}

    with pytest.raises(ResponseValidationError, match="empty text"):
        normalize_response_candidates(response)


def test_url_metadata_does_not_masquerade_as_document_text():
    response = json.loads(_body())
    response["candidates"][0] = {
        "docid": "metadata-only",
        "rank": 1,
        "score": 1.0,
        "url": "https://example.test/document",
    }

    with pytest.raises(ResponseValidationError, match="empty text"):
        normalize_response_candidates(response)


def test_validate_run_rejects_orphan_candidate_without_reservation(tmp_path: Path):
    ledger = RetrievalLedger(tmp_path / "run")
    key = "b" * 64
    ledger.candidates_path(key).write_text("[]")

    with pytest.raises(LedgerIntegrityError, match="no matching.*reservation"):
        ledger.validate_run()


def test_offline_rebuild_verifies_raw_hash_and_candidate_hashes(tmp_path: Path):
    request = _request()
    ledger = RetrievalLedger(tmp_path / "run", shared_cache_dir=tmp_path / "cache")
    result = ledger.retrieve(request, _transport())

    assert ledger.rebuild_candidates_from_raw(request.identity.request_key) == (
        result.candidates
    )
    ledger.raw_path(request.identity.request_key).write_bytes(b"tampered")
    with pytest.raises(LedgerIntegrityError, match="length mismatch|hash mismatch"):
        ledger.rebuild_candidates_from_raw(request.identity.request_key)
    with pytest.raises(LedgerIntegrityError, match="length mismatch|hash mismatch"):
        ledger.validate_run()


def test_success_outcome_raw_byte_count_is_integrity_checked(tmp_path: Path):
    request = _request()
    ledger = RetrievalLedger(tmp_path / "run")
    ledger.retrieve(request, _transport())
    outcome_path = ledger.outcome_path(request.identity.request_key)
    outcome = json.loads(outcome_path.read_text())
    outcome["raw_body_bytes"] += 1
    outcome_path.write_text(json.dumps(outcome))

    with pytest.raises(LedgerIntegrityError, match="raw byte length"):
        ledger.validate_run()


def test_run_evidence_directories_are_precreated_before_any_call(tmp_path: Path):
    ledger = RetrievalLedger(tmp_path / "run")
    assert all(
        directory.is_dir()
        for directory in (
            ledger.attempts_dir,
            ledger.raw_dir,
            ledger.candidates_dir,
            ledger.outcomes_dir,
            ledger.cache_hits_dir,
        )
    )
