from __future__ import annotations

import copy
import hashlib
import inspect
import json
from datetime import timedelta
from pathlib import Path

import pytest

import trec_rag.adaptive_obligation_v2_retrieve as module
import trec_rag.adaptive_obligation_v2_validate as validate_module
from trec_rag.adaptive_obligation_v2_retrieve import (
    INDEX_ID,
    MAX_ACCEPTED_O1_PER_TOPIC,
    MAX_RETRIEVAL_REQUESTS,
    REQUEST_START_INTERVAL_SECONDS,
    RETRIEVAL_HITS,
    TIMEOUT_SECONDS,
    TRANSPORT_RETRY_COUNT,
    RateLimitedO1Transport,
    audit_retrieval_cache,
    build_retrieval_jobs,
    execute_retrieval,
    freeze_retrieval_inputs,
    load_authenticated_retrieval_inputs,
    render_o1_bm25_query,
    verify_retrieval_preflight,
)
from trec_rag.adaptive_obligation_v2_contract import canonical_sha256
from trec_rag.adaptive_obligation_v2_ledger import AppendOnlyAttemptLedger
from trec_rag.adaptive_obligation_v2_validate import (
    VALIDATION_SCHEMA,
    build_validation_jobs,
    finalize_validation_inventory,
    publish_validation_preflight,
    run_validation_job_with_retry,
)
from trec_rag.det_sparse_ledger import RawTransportResponse


PILOT_TOPICS = ("219", "72", "300", "84")
ENDPOINT = "https://example.test/v1/climbmix-400b/search"


def _accepted_row(number: int = 0, *, topic_id: str = "219") -> dict[str, object]:
    return {
        "schema_version": "adaptive-obligation-v2-accepted-o1-v1",
        "proposal_id": f"proposal-{topic_id}-{number}",
        "topic_id": topic_id,
        "parent_id": f"parent-{topic_id}-{number}",
        "parent_manifest_order": number,
        "anchor_terms": [f"subject-{topic_id}", f"population-{number}"],
        "parent_text": f"Effects on population {number}.",
        "label": f"New obligation {number}",
        "accepted": True,
        "decision": "SUPPORTED",
        "support_document_ids": [f"source-{number}", f"validation-{number}"],
    }


def _accepted_o1_rows(count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number in range(count):
        rows.append(
            _accepted_row(number // len(PILOT_TOPICS), topic_id=PILOT_TOPICS[number % 4])
        )
    return rows


def _response_body(count: int = RETRIEVAL_HITS) -> bytes:
    return json.dumps(
        {
            "candidates": [
                {
                    "rank": rank,
                    "docid": f"doc-{rank}",
                    "score": float(RETRIEVAL_HITS - rank),
                    "doc": {"contents": f"Document body {rank}."},
                }
                for rank in range(1, count + 1)
            ]
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_approval(preflight: Path, approval: Path, **changes: object) -> dict[str, object]:
    receipt = json.loads((preflight / "receipt.json").read_bytes())
    value: dict[str, object] = {
        "schema_version": "adaptive-obligation-v2-retrieval-approval-v1",
        "stage": "retrieval",
        "preflight_sha256": hashlib.sha256(
            (preflight / "receipt.json").read_bytes()
        ).hexdigest(),
        "planned_request_count": receipt["planned_request_count"],
        "primary_external_attempts": receipt["primary_external_attempts"],
        "maximum_external_attempts": receipt["maximum_external_attempts"],
        "transport_retry_count": 0,
        "cache_root": receipt["cache_root"],
        "output_dir": receipt["output_dir"],
        "approved": True,
    }
    value.update(changes)
    approval.write_bytes(_pretty(value))
    return value


def _preflight(
    tmp_path: Path,
    rows: list[dict[str, object]] | None = None,
    *,
    cache_root: Path | None = None,
) -> tuple[Path, Path, Path, dict[str, object]]:
    cache = cache_root or tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "preflight"
    receipt = module._audit_retrieval_cache_rows(
        rows or [_accepted_row()],
        endpoint=ENDPOINT,
        cache_root=cache,
        retrieval_output_dir=tmp_path / "run",
        output_dir=output,
    )
    approval = tmp_path / "approval.json"
    _write_approval(output, approval)
    return output, approval, cache, receipt


def test_bm25_query_is_focused_and_deduplicated() -> None:
    query = render_o1_bm25_query(
        anchor_terms=["deforestation", "Amazon rainforest"],
        parent_text="Effects of deforestation on the Amazon rainforest.",
        o1_label="Indigenous community displacement",
        narrative="This broad narrative must not be copied into BM25.",
    )
    assert query == (
        "deforestation Amazon rainforest Effects of deforestation on the "
        "Amazon rainforest. Indigenous community displacement"
    )
    assert "broad narrative" not in query


def test_query_removes_only_exact_whole_parts_after_first_occurrence() -> None:
    query = render_o1_bm25_query(
        anchor_terms=["  Human aggression ", "human   aggression", "causes"],
        parent_text="Causes of human aggression in adults",
        o1_label="social learning mechanisms",
        narrative="ignored",
    )
    assert query == (
        "Human aggression causes Causes of human aggression in adults "
        "social learning mechanisms"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"anchor_terms": []},
        {"anchor_terms": ["valid", 7]},
        {"parent_text": "  "},
        {"label": ""},
    ],
)
def test_query_source_fields_must_be_nonempty_text(change: dict[str, object]) -> None:
    row = _accepted_row()
    row.update(change)
    with pytest.raises((TypeError, ValueError)):
        module._build_retrieval_jobs_from_inputs([row], endpoint=ENDPOINT)


def test_jobs_use_one_hits_1000_request_per_accepted_o1() -> None:
    jobs = module._build_retrieval_jobs_from_inputs(
        _accepted_o1_rows(16), endpoint=ENDPOINT
    )
    assert len(jobs) == MAX_RETRIEVAL_REQUESTS == 16
    assert all(job["hits"] == RETRIEVAL_HITS == 1000 for job in jobs)
    assert all(job["timeout_seconds"] == TIMEOUT_SECONDS == 120 for job in jobs)
    assert all(job["transport_retry_count"] == TRANSPORT_RETRY_COUNT == 0 for job in jobs)
    assert all(
        job["rate_limiter"]["minimum_request_start_interval_seconds"]
        == REQUEST_START_INTERVAL_SECONDS
        == 3
        for job in jobs
    )


def test_request_identity_binds_every_retrieval_field_and_full_accepted_row() -> None:
    baseline_row = _accepted_row()
    baseline = module._build_retrieval_jobs_from_inputs(
        [baseline_row], endpoint=ENDPOINT
    )[0]
    assert baseline["accepted_o1_sha256"] == canonical_sha256(baseline_row)
    assert baseline["request_key"] == canonical_sha256(baseline["request_identity"])
    assert baseline["request_identity"] == {
        "topic_id": "219",
        "parent_id": "parent-219-0",
        "accepted_o1_id": "proposal-219-0",
        "accepted_o1_sha256": canonical_sha256(baseline_row),
        "query_text": baseline["query_text"],
        "query_sha256": baseline["query_sha256"],
        "endpoint": ENDPOINT,
        "index_id": INDEX_ID,
        "retriever_version": module.RETRIEVER_VERSION,
        "hits": 1000,
        "timeout_seconds": 120,
        "transport_retry_count": 0,
        "rate_limiter": module.RATE_LIMITER_IDENTITY,
    }

    changed_row = copy.deepcopy(baseline_row)
    changed_row["support_document_ids"].append("third-source")  # type: ignore[union-attr]
    changed = module._build_retrieval_jobs_from_inputs(
        [changed_row], endpoint=ENDPOINT
    )[0]
    assert changed["query_text"] == baseline["query_text"]
    assert changed["request_key"] != baseline["request_key"]

    changed_endpoint = module._build_retrieval_jobs_from_inputs(
        [baseline_row], endpoint="https://other.example/search"
    )[0]
    changed_index = module._build_retrieval_jobs_from_inputs(
        [baseline_row], endpoint=ENDPOINT, index_id="other-index"
    )[0]
    assert changed_endpoint["request_key"] != baseline["request_key"]
    assert changed_index["request_key"] != baseline["request_key"]


def test_jobs_are_deterministic_under_input_reordering() -> None:
    rows = _accepted_o1_rows(12)
    assert module._build_retrieval_jobs_from_inputs(
        rows, endpoint=ENDPOINT
    ) == module._build_retrieval_jobs_from_inputs(
        list(reversed(rows)), endpoint=ENDPOINT
    )


def test_duplicate_accepted_ids_fail_but_equal_rendered_queries_keep_both_streams() -> None:
    first = _accepted_row(0)
    duplicate_id = _accepted_row(1)
    duplicate_id["proposal_id"] = first["proposal_id"]
    with pytest.raises(ValueError, match="accepted O1 ID"):
        module._build_retrieval_jobs_from_inputs(
            [first, duplicate_id], endpoint=ENDPOINT
        )

    duplicate_query = copy.deepcopy(first)
    duplicate_query["proposal_id"] = "different-proposal"
    duplicate_query["parent_id"] = "different-parent"
    jobs = module._build_retrieval_jobs_from_inputs(
        [first, duplicate_query], endpoint=ENDPOINT
    )
    assert len(jobs) == 2
    assert jobs[0]["query_sha256"] == jobs[1]["query_sha256"]
    assert jobs[0]["request_key"] != jobs[1]["request_key"]


def test_job_identity_and_cache_audit_nested_schemas_are_closed() -> None:
    job = module._build_retrieval_jobs_from_inputs(
        [_accepted_row()], endpoint=ENDPOINT
    )[0]
    job["unexpected_transport_control"] = "retry forever"
    with pytest.raises(ValueError, match="job fields"):
        module._verify_job(job)

    job = module._build_retrieval_jobs_from_inputs(
        [_accepted_row()], endpoint=ENDPOINT
    )[0]
    identity = dict(job["request_identity"])
    identity["unexpected"] = True
    job["request_identity"] = identity
    job["request_key"] = canonical_sha256(identity)
    job["job_id"] = job["request_key"]
    with pytest.raises(ValueError, match="identity fields"):
        module._verify_job(job)

    job = module._build_retrieval_jobs_from_inputs(
        [_accepted_row()], endpoint=ENDPOINT
    )[0]
    with pytest.raises(ValueError, match="cache probe fields"):
        module._cache_status(
            {
                "hit": True,
                "raw_bytes": 1,
                "raw_sha256": "a" * 64,
                "candidate_count": 1000,
                "unexpected": 1,
            },
            job,
        )


def test_total_and_per_topic_budgets_fail_before_cache() -> None:
    touched: list[object] = []
    with pytest.raises(ValueError, match="16"):
        module._audit_retrieval_cache_rows(
            _accepted_o1_rows(17),
            endpoint=ENDPOINT,
            retrieval_output_dir=Path("/var/tmp/task5-unused"),
            cache_loader=lambda *_args: touched.append("cache"),
        )
    assert touched == []

    with pytest.raises(ValueError, match="four accepted O1"):
        module._audit_retrieval_cache_rows(
            [_accepted_row(number, topic_id="219") for number in range(5)],
            endpoint=ENDPOINT,
            retrieval_output_dir=Path("/var/tmp/task5-unused"),
            cache_loader=lambda *_args: touched.append("cache"),
        )
    assert touched == []


@pytest.mark.parametrize("topic_id", ["144", "213", "224", "407", "515"])
def test_protected_topics_fail_before_cache_or_endpoint_access(topic_id: str) -> None:
    touched: list[str] = []

    class Endpoint:
        def __str__(self) -> str:
            touched.append("endpoint")
            return ENDPOINT

    with pytest.raises(ValueError, match="protected topic"):
        module._audit_retrieval_cache_rows(
            [_accepted_row(topic_id=topic_id)],
            endpoint=Endpoint(),  # type: ignore[arg-type]
            retrieval_output_dir=Path("/var/tmp/task5-unused"),
            cache_loader=lambda *_args: touched.append("cache"),
        )
    assert touched == []


def test_cache_only_audit_reports_exact_hits_misses_rows_and_size() -> None:
    rows = _accepted_o1_rows(4)
    jobs = module._build_retrieval_jobs_from_inputs(rows, endpoint=ENDPOINT)
    hit_keys = {jobs[0]["request_key"], jobs[2]["request_key"]}
    touched: list[str] = []

    def loader(job: dict[str, object]) -> dict[str, object] | None:
        touched.append(str(job["request_key"]))
        if job["request_key"] in hit_keys:
            return {
                "hit": True,
                "raw_bytes": 1234,
                "raw_sha256": "a" * 64,
                "candidate_count": 1000,
            }
        return None

    receipt = module._audit_retrieval_cache_rows(
        rows,
        endpoint=ENDPOINT,
        retrieval_output_dir=Path("/var/tmp/task5-unused"),
        cache_loader=loader,
    )
    assert touched == [str(job["request_key"]) for job in jobs]
    assert receipt["verified_cache_hits"] == 2
    assert receipt["verified_cache_misses"] == 2
    assert receipt["expected_raw_rows"] == 4000
    assert receipt["observed_cached_raw_bytes"] == 2468
    assert receipt["primary_external_attempts"] == 2
    assert receipt["maximum_external_attempts"] == 2
    assert receipt["estimated_new_raw_bytes"] > 0
    assert receipt["network_call_count"] == 0
    assert receipt["qrels_opened"] is False


def test_preflight_binds_cache_and_final_output_destinations(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    final_output = tmp_path / "retrieval-output"
    receipt = module._audit_retrieval_cache_rows(
        [_accepted_row()],
        endpoint=ENDPOINT,
        cache_root=cache,
        retrieval_output_dir=final_output,
    )
    assert receipt["cache_root"] == str(cache.absolute())
    assert receipt["output_dir"] == str(final_output.absolute())


def test_public_executor_has_no_transport_or_destination_override() -> None:
    parameters = inspect.signature(execute_retrieval).parameters
    assert "transport_factory" not in parameters
    assert "output_dir" not in parameters
    assert "cache_root" not in parameters
    assert "session" not in inspect.signature(RateLimitedO1Transport).parameters


def test_cache_probe_rejects_malformed_status_without_transport() -> None:
    with pytest.raises(ValueError, match="cache probe"):
        module._audit_retrieval_cache_rows(
            [_accepted_row()],
            endpoint=ENDPOINT,
            retrieval_output_dir=Path("/var/tmp/task5-unused"),
            cache_loader=lambda _job: {"hit": True, "raw_bytes": -1},
        )


def test_preflight_is_create_only_hash_bound_and_verifiable(tmp_path: Path) -> None:
    output = tmp_path / "preflight"
    (tmp_path / "cache").mkdir()
    receipt = module._audit_retrieval_cache_rows(
        [_accepted_row()],
        endpoint=ENDPOINT,
        cache_root=tmp_path / "cache",
        retrieval_output_dir=tmp_path / "run",
        output_dir=output,
    )
    assert verify_retrieval_preflight(output) == receipt
    with pytest.raises(FileExistsError):
        module._audit_retrieval_cache_rows(
            [_accepted_row()],
            endpoint=ENDPOINT,
            cache_root=tmp_path / "cache",
            retrieval_output_dir=tmp_path / "run",
            output_dir=output,
        )

    mutated = json.loads((output / "receipt.json").read_bytes())
    mutated["hits"] = 100
    (output / "receipt.json").write_bytes(_pretty(mutated))
    with pytest.raises(ValueError):
        verify_retrieval_preflight(output)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"topic_ids": ["219"]}, "topic_ids"),
        ({"unexpected": "not schema metadata"}, "fields"),
    ],
)
def test_preflight_receipt_has_an_exact_closed_schema(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    output, _approval, _cache, _receipt = _preflight(tmp_path)
    mutated = json.loads((output / "receipt.json").read_bytes())
    mutated.update(change)
    (output / "receipt.json").write_bytes(_pretty(mutated))
    with pytest.raises(ValueError, match=message):
        verify_retrieval_preflight(output)


def test_preflight_publish_rejects_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        module._audit_retrieval_cache_rows(
            [_accepted_row()],
            endpoint=ENDPOINT,
            cache_root=tmp_path / "cache",
            retrieval_output_dir=tmp_path / "run",
            output_dir=linked / "preflight",
        )
    assert not (real / "preflight").exists()


def test_preflight_rejects_symlink_root_and_unexpected_children(tmp_path: Path) -> None:
    real, _approval, _cache, _receipt = _preflight(tmp_path)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="missing or unsafe"):
        verify_retrieval_preflight(linked)

    (real / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unsafe"):
        verify_retrieval_preflight(real)


def test_execute_requires_separate_retrieval_approval_before_every_other_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched: list[str] = []
    monkeypatch.setattr(
        module,
        "_capture_retrieval_preflight",
        lambda *_args, **_kwargs: touched.append("preflight"),
    )
    monkeypatch.setattr(
        module,
        "_SecureCacheStore",
        lambda *_args, **_kwargs: touched.append("cache"),
    )
    monkeypatch.setattr(
        module,
        "_claim_output_root",
        lambda *_args, **_kwargs: touched.append("output"),
    )
    monkeypatch.setattr(
        module,
        "_build_approved_transport",
        lambda *_args, **_kwargs: touched.append("transport"),
    )
    with pytest.raises(PermissionError, match="retrieval approval required"):
        execute_retrieval(
            preflight_dir=tmp_path / "preflight",
            approval_path=tmp_path / "missing.json",
        )
    assert touched == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"approved": 1}, "retrieval approval required"),
        ({"approved": False}, "retrieval approval required"),
        ({"stage": "proposal"}, "retrieval approval required"),
        ({"transport_retry_count": True}, "retrieval approval required"),
        ({"maximum_external_attempts": 99}, "retrieval approval required"),
        ({"preflight_sha256": "0" * 64}, "retrieval approval required"),
        ({"cache_root": "/var/tmp/unapproved-cache"}, "retrieval approval required"),
        ({"output_dir": "/var/tmp/unapproved-output"}, "retrieval approval required"),
    ],
)
def test_approval_types_and_exact_bindings_fail_closed(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    preflight, approval, _cache, _receipt = _preflight(tmp_path)
    _write_approval(preflight, approval, **change)
    touched: list[str] = []
    with pytest.raises(PermissionError, match=message):
        execute_retrieval(
            preflight_dir=preflight,
            approval_path=approval,
        )
    assert touched == []
    assert not (tmp_path / "run").exists()


def test_symlinked_approval_fails_before_preflight_cache_session_or_output(
    tmp_path: Path,
) -> None:
    preflight, approval, _cache, _receipt = _preflight(tmp_path)
    real = tmp_path / "real-approval.json"
    approval.rename(real)
    approval.symlink_to(real)
    touched: list[str] = []
    with pytest.raises(PermissionError, match="retrieval approval required"):
        execute_retrieval(
            preflight_dir=preflight,
            approval_path=approval,
        )
    assert touched == []
    assert not (tmp_path / "run").exists()


def test_executor_reserves_before_transport_and_raw_before_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight, approval, cache, _receipt = _preflight(tmp_path)
    output = tmp_path / "run"
    events: list[str] = []
    original_normalize = module._normalize_response

    def observed_normalize(raw: bytes) -> tuple[dict[str, object], ...]:
        events.append("parse")
        raw_files = list((output / "raw").glob("*.body"))
        assert len(raw_files) == 1
        assert raw_files[0].read_bytes() == raw
        return original_normalize(raw)

    monkeypatch.setattr(module, "_normalize_response", observed_normalize)

    class Transport:
        one_shot_no_retry = True
        request_start_interval_seconds = 3
        timeout_seconds = 120

        def __call__(self, job: dict[str, object]) -> RawTransportResponse:
            events.append("transport")
            assert (output / "attempts" / f"{job['request_key']}.json").is_file()
            return RawTransportResponse(200, {"X-Test": "yes"}, _response_body(), 0.1)

    monkeypatch.setattr(
        module, "_build_approved_transport", lambda *_args, **_kwargs: Transport()
    )

    result = execute_retrieval(
        preflight_dir=preflight,
        approval_path=approval,
    )
    assert events == ["transport", "parse"]
    assert result["external_attempts"] == 1
    assert result["candidate_rows"] == 1000
    candidate_files = list((output / "candidates").glob("*.json"))
    assert len(candidate_files) == 1
    first_candidate = json.loads(candidate_files[0].read_bytes())[0]
    assert first_candidate["retrieval_input"] == _accepted_row()
    assert first_candidate["raw_response_sha256"] == hashlib.sha256(
        _response_body()
    ).hexdigest()
    assert first_candidate["text_sha256"] == hashlib.sha256(
        first_candidate["text"].encode()
    ).hexdigest()
    assert first_candidate["query_sha256"] == hashlib.sha256(
        first_candidate["query_text"].encode()
    ).hexdigest()
    assert (output / "summary.json").is_file()


def test_bound_cache_and_output_swaps_cannot_redirect_nested_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight, approval, cache, _receipt = _preflight(tmp_path)
    output = tmp_path / "run"
    held_cache = tmp_path / "held-cache"
    held_output = tmp_path / "held-output"
    outside_cache = tmp_path / "outside-cache"
    outside_output = tmp_path / "outside-output"
    outside_cache.mkdir()
    outside_output.mkdir()

    class Transport:
        one_shot_no_retry = True
        request_start_interval_seconds = 3
        timeout_seconds = 120

        def __call__(self, _job: dict[str, object]) -> RawTransportResponse:
            cache.rename(held_cache)
            cache.symlink_to(outside_cache, target_is_directory=True)
            output.rename(held_output)
            output.symlink_to(outside_output, target_is_directory=True)
            return RawTransportResponse(200, {}, _response_body(), 0.1)

    monkeypatch.setattr(
        module, "_build_approved_transport", lambda *_args, **_kwargs: Transport()
    )
    result = execute_retrieval(
        preflight_dir=preflight,
        approval_path=approval,
    )
    assert result["candidate_rows"] == 1000
    assert not list(outside_cache.iterdir())
    assert not list(outside_output.iterdir())
    assert list(held_cache.rglob("*.manifest.json"))
    assert (held_output / "summary.json").is_file()


def test_fully_cached_execution_still_requires_approval_but_never_builds_transport(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    job = module._build_retrieval_jobs_from_inputs(
        [_accepted_row()], endpoint=ENDPOINT
    )[0]
    raw = _response_body()
    cache.mkdir()
    cache_store = module._SecureCacheStore(cache)
    try:
        cache_store.store(job, raw, module._normalize_response(raw))
    finally:
        cache_store.close()
    preflight, approval, _cache, receipt = _preflight(
        tmp_path, cache_root=cache
    )
    assert receipt["verified_cache_hits"] == 1
    touched: list[str] = []
    result = execute_retrieval(
        preflight_dir=preflight,
        approval_path=approval,
    )
    assert touched == []
    assert result["cache_hits"] == 1
    assert result["external_attempts"] == 0
    assert result["candidate_rows"] == 1000


def test_malformed_raw_response_is_preserved_before_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight, approval, cache, _receipt = _preflight(tmp_path)
    output = tmp_path / "run"

    class Transport:
        one_shot_no_retry = True
        request_start_interval_seconds = 3
        timeout_seconds = 120

        def __call__(self, _job: dict[str, object]) -> RawTransportResponse:
            return RawTransportResponse(200, {}, b"not-json", 0.1)

    monkeypatch.setattr(
        module, "_build_approved_transport", lambda *_args, **_kwargs: Transport()
    )

    with pytest.raises(ValueError, match="UTF-8 JSON"):
        execute_retrieval(
            preflight_dir=preflight,
            approval_path=approval,
        )
    assert [path.read_bytes() for path in (output / "raw").glob("*.body")] == [
        b"not-json"
    ]
    outcomes = [json.loads(path.read_bytes()) for path in (output / "outcomes").glob("*.json")]
    assert outcomes[0]["status"] == "failure"
    assert outcomes[0]["failure_type"] == "response_validation_error"
    assert not list(cache.rglob("*.manifest.json"))


@pytest.mark.parametrize("wire_key", ["candidates", "hits", "results"])
def test_response_normalization_matches_tracked_wire_keys_and_never_uses_docid_as_text(
    wire_key: str,
) -> None:
    payload = json.loads(_response_body())
    raw = json.dumps({wire_key: payload["candidates"]}).encode()
    rows = module._normalize_response(raw)
    assert len(rows) == 1000
    assert rows[0]["text_sha256"] == hashlib.sha256(
        str(rows[0]["text"]).encode()
    ).hexdigest()

    payload["candidates"][0] = {
        "rank": 1,
        "docid": "this identifier is not document content",
        "score": 1.0,
    }
    with pytest.raises(ValueError, match="text-bearing"):
        module._normalize_response(json.dumps(payload).encode())


def test_executor_rejects_transport_with_retry_or_wrong_limiter_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight, approval, cache, _receipt = _preflight(tmp_path)

    class BadTransport:
        one_shot_no_retry = False
        request_start_interval_seconds = 0
        timeout_seconds = 1

    monkeypatch.setattr(
        module, "_build_approved_transport", lambda *_args, **_kwargs: BadTransport()
    )

    with pytest.raises(ValueError, match="one-shot no-retry"):
        execute_retrieval(
            preflight_dir=preflight,
            approval_path=approval,
        )
    assert not (tmp_path / "run").exists()


def test_existing_output_is_create_only_and_transport_is_not_entered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight, approval, cache, _receipt = _preflight(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    touched: list[str] = []

    class Transport:
        one_shot_no_retry = True
        request_start_interval_seconds = 3
        timeout_seconds = 120

        def __call__(self, _job: dict[str, object]) -> RawTransportResponse:
            touched.append("transport-call")
            raise AssertionError

    monkeypatch.setattr(
        module, "_build_approved_transport", lambda *_args, **_kwargs: Transport()
    )

    with pytest.raises(FileExistsError):
        execute_retrieval(
            preflight_dir=preflight,
            approval_path=approval,
        )
    assert touched == []


def test_rate_limited_transport_uses_tracked_session_and_frozen_request_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = module._build_retrieval_jobs_from_inputs(
        [_accepted_row()], endpoint=ENDPOINT
    )[0]

    class Response:
        status_code = 503
        headers = {"Retry-After": "300"}
        content = b'{"error":"upstream"}'
        elapsed = timedelta(milliseconds=20)

    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def get(self, url: str, **kwargs: object) -> Response:
            self.calls.append((url, kwargs))
            return Response()

    session = Session()
    monkeypatch.setattr(module, "rate_limited_session", lambda _config: session)
    transport = RateLimitedO1Transport([job], api_token="secret")
    response = transport(job)
    assert response.status == 503
    assert response.body == Response.content
    assert transport.one_shot_no_retry is True
    assert transport.request_start_interval_seconds == 3
    assert transport.timeout_seconds == 120
    assert session.calls == [
        (
            ENDPOINT,
            {
                "params": {"query": job["query_text"], "hits": "1000"},
                "headers": {
                    "Accept": "application/json",
                    "Authorization": "Bearer secret",
                },
                "timeout": 120,
                "allow_redirects": False,
            },
        )
    ]


def test_real_task4_acceptance_freezes_authenticated_retrieval_inputs(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    contract_dir = (
        repo
        / "outputs/rag25_deep_facet_candidates_v1/"
        "adaptive_obligation_search_v2/contract"
    )
    contract = validate_module._load_verified_contract(contract_dir)
    parent = next(
        row for row in contract["parents"] if row["parent_id"] == "219-positive"
    )
    source = next(
        row
        for row in contract["units"]
        if row["parent_id"] == parent["parent_id"] and row["fold"] == 0
    )
    proposal = {
        "proposal_id": "actual-task4-proposal",
        "status": "SUPPORTED",
        "reason_code": "SUPPORTED",
        "topic_id": parent["topic_id"],
        "parent_id": parent["parent_id"],
        "proposal_fold": 0,
        "parent_manifest_order": parent["manifest_order"],
        "label": "access barriers for disadvantaged communities",
        "scope_rationale": "fixture rationale excluded from validator",
        "support_unit_ids": [source["unit_id"]],
    }
    job = build_validation_jobs([proposal], contract)[0]
    preflight = _validation_preflight_for_job(
        job,
        source_contract_receipt_sha256=hashlib.sha256(
            (contract_dir / "receipt.json").read_bytes()
        ).hexdigest(),
    )
    validation_preflight = tmp_path / "validation-preflight"
    publish_validation_preflight(preflight, validation_preflight)
    preflight_sha256 = hashlib.sha256(
        (validation_preflight / "receipt.json").read_bytes()
    ).hexdigest()
    anchor = validate_module._build_validation_run_anchor(
        [job], preflight_sha256=preflight_sha256, approval_sha256="b" * 64
    )
    validation_ledger = tmp_path / "validation-ledger"
    ledger = AppendOnlyAttemptLedger(
        validation_ledger, expected_anchor=anchor, create_only=True
    )
    opposite_id = str(job["input_unit_ids"][0])  # type: ignore[index]

    class Model:
        def generate(self, _messages, _schema, *, max_new_tokens):
            assert max_new_tokens == 256
            return (
                json.dumps(
                    {
                        "decision": "SUPPORTED",
                        "support_unit_ids": [opposite_id],
                    }
                ).encode(),
                20,
            )

    assert run_validation_job_with_retry(job, ledger=ledger, model=Model())[
        "accepted"
    ] is True
    ledger.seal_completion()
    validation_output = tmp_path / "validation-output"
    finalize_validation_inventory(
        preflight_dir=validation_preflight,
        ledger_dir=validation_ledger,
        output_dir=validation_output,
    )

    retrieval_inputs = tmp_path / "retrieval-inputs"
    receipt = freeze_retrieval_inputs(
        validation_output_dir=validation_output,
        validation_preflight_dir=validation_preflight,
        validation_ledger_dir=validation_ledger,
        contract_dir=contract_dir,
        output_dir=retrieval_inputs,
    )
    authenticated = load_authenticated_retrieval_inputs(retrieval_inputs)
    assert receipt == authenticated["receipt"]
    public_jobs = build_retrieval_jobs(retrieval_inputs, endpoint=ENDPOINT)
    assert len(public_jobs) == 1
    row = authenticated["inputs"][0]
    assert row["anchor_terms"] == ["technology"]
    assert row["parent_text"] == parent["text"]
    assert row["parent_text_sha256"] == parent["text_sha256"]
    assert row["parent_query"] == parent["query"]
    assert row["parent_query_sha256"] == parent["query_sha256"]
    assert row["narrative"] == str(parent["query"]).split(
        "\n\nExplicit obligation:\n", 1
    )[0]
    assert row["narrative_sha256"] == hashlib.sha256(
        str(row["narrative"]).encode()
    ).hexdigest()
    assert row["o1_label"] == proposal["label"]
    assert row["validation_support_unit_ids"] == [opposite_id]
    assert row["proposal_document_ids"] == [source["document_id"]]
    assert row["source_manifest"]["sha256"] == (
        "14954020d5edae579d31a1f2f68e6a98b21b151de139aed664615884a2013eb4"
    )


def _validation_preflight_for_job(
    job: dict[str, object], *, source_contract_receipt_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": validate_module.SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(validate_module.PILOT_TOPIC_IDS),
        "job_count": 1,
        "primary_call_count": 1,
        "retry_call_ceiling": 1,
        "worst_case_call_ceiling": 2,
        "primary_max_new_tokens": validate_module.PRIMARY_MAX_NEW_TOKENS,
        "retry_max_new_tokens": validate_module.RETRY_MAX_NEW_TOKENS,
        "schema_sha256": canonical_sha256(VALIDATION_SCHEMA),
        "jobs_sha256": canonical_sha256([job]),
        "jobs": [job],
        "source_contract_receipt_sha256": source_contract_receipt_sha256,
        "proposal_receipt": {
            "schema_version": validate_module.PROPOSAL_RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "sha256": "1" * 64,
            "proposal_count": 1,
            "proposals_sha256": "2" * 64,
            "proposal_preflight_receipt_sha256": "3" * 64,
            "source_contract_receipt_sha256": source_contract_receipt_sha256,
            "run_anchor_sha256": "4" * 64,
            "completion_sha256": "5" * 64,
        },
        "expected_runtime": {
            "phase": "inference_free_preflight",
            "proposal_calls_executed": 0,
            "validation_calls_executed": 0,
        },
        "model_construction_allowed": False,
        "generation_allowed": False,
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "model_load_count": 0,
        "tokenizer_load_count": 0,
        "inference_count": 0,
        "external_cost_usd": 0.0,
    }
