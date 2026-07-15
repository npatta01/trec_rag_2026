from __future__ import annotations

import copy
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

import trec_rag.adaptive_obligation_v2_retrieve as module
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
    render_o1_bm25_query,
    verify_retrieval_preflight,
)
from trec_rag.adaptive_obligation_v2_contract import canonical_sha256
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
    output = tmp_path / "preflight"
    receipt = audit_retrieval_cache(
        rows or [_accepted_row()],
        endpoint=ENDPOINT,
        cache_root=cache,
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
        build_retrieval_jobs([row], endpoint=ENDPOINT)


def test_jobs_use_one_hits_1000_request_per_accepted_o1() -> None:
    jobs = build_retrieval_jobs(_accepted_o1_rows(16), endpoint=ENDPOINT)
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
    baseline = build_retrieval_jobs([baseline_row], endpoint=ENDPOINT)[0]
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
    changed = build_retrieval_jobs([changed_row], endpoint=ENDPOINT)[0]
    assert changed["query_text"] == baseline["query_text"]
    assert changed["request_key"] != baseline["request_key"]

    changed_endpoint = build_retrieval_jobs(
        [baseline_row], endpoint="https://other.example/search"
    )[0]
    changed_index = build_retrieval_jobs(
        [baseline_row], endpoint=ENDPOINT, index_id="other-index"
    )[0]
    assert changed_endpoint["request_key"] != baseline["request_key"]
    assert changed_index["request_key"] != baseline["request_key"]


def test_jobs_are_deterministic_under_input_reordering() -> None:
    rows = _accepted_o1_rows(12)
    assert build_retrieval_jobs(rows, endpoint=ENDPOINT) == build_retrieval_jobs(
        list(reversed(rows)), endpoint=ENDPOINT
    )


def test_duplicate_accepted_ids_and_query_hashes_fail() -> None:
    first = _accepted_row(0)
    duplicate_id = _accepted_row(1)
    duplicate_id["proposal_id"] = first["proposal_id"]
    with pytest.raises(ValueError, match="accepted O1 ID"):
        build_retrieval_jobs([first, duplicate_id], endpoint=ENDPOINT)

    duplicate_query = copy.deepcopy(first)
    duplicate_query["proposal_id"] = "different-proposal"
    duplicate_query["parent_id"] = "different-parent"
    with pytest.raises(ValueError, match="query hash"):
        build_retrieval_jobs([first, duplicate_query], endpoint=ENDPOINT)


def test_total_and_per_topic_budgets_fail_before_cache() -> None:
    touched: list[object] = []
    with pytest.raises(ValueError, match="16"):
        audit_retrieval_cache(
            _accepted_o1_rows(17),
            endpoint=ENDPOINT,
            cache_loader=lambda *_args: touched.append("cache"),
        )
    assert touched == []

    with pytest.raises(ValueError, match="four accepted O1"):
        audit_retrieval_cache(
            [_accepted_row(number, topic_id="219") for number in range(5)],
            endpoint=ENDPOINT,
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
        audit_retrieval_cache(
            [_accepted_row(topic_id=topic_id)],
            endpoint=Endpoint(),  # type: ignore[arg-type]
            cache_loader=lambda *_args: touched.append("cache"),
        )
    assert touched == []


def test_cache_only_audit_reports_exact_hits_misses_rows_and_size() -> None:
    rows = _accepted_o1_rows(4)
    jobs = build_retrieval_jobs(rows, endpoint=ENDPOINT)
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

    receipt = audit_retrieval_cache(
        rows, endpoint=ENDPOINT, cache_loader=loader
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


def test_cache_probe_rejects_malformed_status_without_transport() -> None:
    with pytest.raises(ValueError, match="cache probe"):
        audit_retrieval_cache(
            [_accepted_row()],
            endpoint=ENDPOINT,
            cache_loader=lambda _job: {"hit": True, "raw_bytes": -1},
        )


def test_preflight_is_create_only_hash_bound_and_verifiable(tmp_path: Path) -> None:
    output = tmp_path / "preflight"
    receipt = audit_retrieval_cache(
        [_accepted_row()],
        endpoint=ENDPOINT,
        cache_root=tmp_path / "cache",
        output_dir=output,
    )
    assert verify_retrieval_preflight(output) == receipt
    with pytest.raises(FileExistsError):
        audit_retrieval_cache(
            [_accepted_row()],
            endpoint=ENDPOINT,
            cache_root=tmp_path / "cache",
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
        audit_retrieval_cache(
            [_accepted_row()],
            endpoint=ENDPOINT,
            cache_root=tmp_path / "cache",
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
        "_load_verified_cache",
        lambda *_args, **_kwargs: touched.append("cache"),
    )
    monkeypatch.setattr(
        module,
        "_claim_output_root",
        lambda *_args, **_kwargs: touched.append("output"),
    )
    with pytest.raises(PermissionError, match="retrieval approval required"):
        execute_retrieval(
            preflight_dir=tmp_path / "preflight",
            approval_path=tmp_path / "missing.json",
            transport_factory=lambda *_args: touched.append("transport"),
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
            output_dir=tmp_path / "run",
            transport_factory=lambda *_args: touched.append("transport"),
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
            output_dir=tmp_path / "run",
            transport_factory=lambda *_args: touched.append("transport"),
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

    result = execute_retrieval(
        preflight_dir=preflight,
        approval_path=approval,
        output_dir=output,
        cache_root=cache,
        transport_factory=lambda _jobs: Transport(),
    )
    assert events == ["transport", "parse"]
    assert result["external_attempts"] == 1
    assert result["candidate_rows"] == 1000
    assert len(list((output / "candidates").glob("*.json"))) == 1
    assert (output / "summary.json").is_file()


def test_fully_cached_execution_still_requires_approval_but_never_builds_transport(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    job = build_retrieval_jobs([_accepted_row()], endpoint=ENDPOINT)[0]
    raw = _response_body()
    module._store_cache(cache, job, raw, module._normalize_response(raw))
    preflight, approval, _cache, receipt = _preflight(
        tmp_path, cache_root=cache
    )
    assert receipt["verified_cache_hits"] == 1
    touched: list[str] = []
    result = execute_retrieval(
        preflight_dir=preflight,
        approval_path=approval,
        output_dir=tmp_path / "run",
        cache_root=cache,
        transport_factory=lambda _jobs: touched.append("transport"),  # type: ignore[arg-type,return-value]
    )
    assert touched == []
    assert result["cache_hits"] == 1
    assert result["external_attempts"] == 0
    assert result["candidate_rows"] == 1000


def test_malformed_raw_response_is_preserved_before_terminal_failure(
    tmp_path: Path,
) -> None:
    preflight, approval, cache, _receipt = _preflight(tmp_path)
    output = tmp_path / "run"

    class Transport:
        one_shot_no_retry = True
        request_start_interval_seconds = 3
        timeout_seconds = 120

        def __call__(self, _job: dict[str, object]) -> RawTransportResponse:
            return RawTransportResponse(200, {}, b"not-json", 0.1)

    with pytest.raises(ValueError, match="UTF-8 JSON"):
        execute_retrieval(
            preflight_dir=preflight,
            approval_path=approval,
            output_dir=output,
            cache_root=cache,
            transport_factory=lambda _jobs: Transport(),
        )
    assert [path.read_bytes() for path in (output / "raw").glob("*.body")] == [
        b"not-json"
    ]
    outcomes = [json.loads(path.read_bytes()) for path in (output / "outcomes").glob("*.json")]
    assert outcomes[0]["status"] == "failure"
    assert outcomes[0]["failure_type"] == "response_validation_error"
    assert not list(cache.rglob("*.manifest.json"))


def test_executor_rejects_transport_with_retry_or_wrong_limiter_before_output(
    tmp_path: Path,
) -> None:
    preflight, approval, cache, _receipt = _preflight(tmp_path)

    class BadTransport:
        one_shot_no_retry = False
        request_start_interval_seconds = 0
        timeout_seconds = 1

    with pytest.raises(ValueError, match="one-shot no-retry"):
        execute_retrieval(
            preflight_dir=preflight,
            approval_path=approval,
            output_dir=tmp_path / "run",
            cache_root=cache,
            transport_factory=lambda _jobs: BadTransport(),
        )
    assert not (tmp_path / "run").exists()


def test_existing_output_is_create_only_and_transport_is_not_entered(
    tmp_path: Path,
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

    with pytest.raises(FileExistsError):
        execute_retrieval(
            preflight_dir=preflight,
            approval_path=approval,
            output_dir=output,
            cache_root=cache,
            transport_factory=lambda _jobs: Transport(),
        )
    assert touched == []


def test_rate_limited_transport_uses_tracked_session_and_frozen_request_shape() -> None:
    job = build_retrieval_jobs([_accepted_row()], endpoint=ENDPOINT)[0]

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
    transport = RateLimitedO1Transport([job], api_token="secret", session=session)
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
