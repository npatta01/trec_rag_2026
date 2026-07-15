from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import os
from datetime import timedelta
from pathlib import Path

import pytest

import trec_rag.adaptive_obligation_v2_propose as propose_module
import trec_rag.adaptive_obligation_v2_retrieve as module
import trec_rag.adaptive_obligation_v2_validate as validate_module
from trec_rag.adaptive_obligation_v2_retrieve import (
    INDEX_ID,
    MAX_RETRIEVAL_REQUESTS,
    REQUEST_START_INTERVAL_SECONDS,
    RETRIEVAL_HITS,
    TIMEOUT_SECONDS,
    TRANSPORT_RETRY_COUNT,
    RateLimitedO1Transport,
    audit_retrieval_cache,
    execute_retrieval,
    freeze_retrieval_inputs,
    load_authenticated_retrieval_inputs,
    render_o1_bm25_query,
    verify_retrieval_preflight,
)
from trec_rag.adaptive_obligation_v2_contract import canonical_sha256
from trec_rag.adaptive_obligation_v2_ledger import AppendOnlyAttemptLedger
from trec_rag.det_sparse_ledger import RawTransportResponse


PILOT_TOPICS = ("219", "72", "300", "84")
ENDPOINT = "https://example.test/v1/climbmix-400b/search"


@pytest.fixture(scope="module")
def real_v2_validation_handoff(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    """Build a no-inference authenticated proposal→validation fixture."""

    repo = Path(__file__).resolve().parents[2]
    source_root = (
        repo
        / "outputs/rag25_deep_facet_candidates_v1/adaptive_obligation_search_v2"
    )
    contract_dir = source_root / "contract"
    proposal_preflight_dir = source_root / "proposal_preflight"
    root = tmp_path_factory.mktemp("real-v2-validation-handoff")

    proposal_preflight_source = (proposal_preflight_dir / "receipt.json").read_bytes()
    proposal_preflight_sha256 = hashlib.sha256(
        proposal_preflight_source
    ).hexdigest()
    captured_proposal_preflight = (
        propose_module._capture_static_preflight_for_task4(
            proposal_preflight_dir,
            expected_receipt_sha256=proposal_preflight_sha256,
        )
    )
    proposal_approval = {
        "schema_version": "adaptive-obligation-v2-proposal-approval-v1",
        "stage": "proposal",
        "preflight_sha256": proposal_preflight_sha256,
        "model": propose_module.MODEL_ID,
        "model_revision": propose_module.MODEL_REVISION,
        "primary_call_count": 48,
        "retry_call_ceiling": 48,
        "approved": True,
    }
    proposal_approval_path = root / "proposal-approval.json"
    proposal_approval_source = _pretty(proposal_approval)
    proposal_approval_path.write_bytes(proposal_approval_source)
    proposal_approval_sha256 = hashlib.sha256(proposal_approval_source).hexdigest()
    proposal_anchor = propose_module._build_run_anchor(
        captured_proposal_preflight.jobs,
        preflight_sha256=proposal_preflight_sha256,
        approval_sha256=proposal_approval_sha256,
    )
    proposal_ledger_dir = root / "proposal-ledger"
    proposal_ledger = AppendOnlyAttemptLedger(
        proposal_ledger_dir, expected_anchor=proposal_anchor, create_only=True
    )
    selected_job = next(
        job
        for job in captured_proposal_preflight.jobs
        if job["parent_id"] == "219-positive" and job["fold"] == 0
    )
    for job in captured_proposal_preflight.jobs:
        value = (
            {
                "status": "SUPPORTED",
                "reason_code": "SUPPORTED",
                "o1": {
                    "label": "access barriers for disadvantaged communities",
                    "scope_rationale": "Independent sources identify a reusable need.",
                    "support_unit_ids": [job["input_unit_ids"][0]],
                },
            }
            if job["job_id"] == selected_job["job_id"]
            else {
                "status": "UNSUPPORTED",
                "reason_code": "NO_ABSTRACT_CHILD",
                "o1": None,
            }
        )
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        propose_module.run_job_with_retry(
            job,
            generate=lambda _ceiling, result=raw: (result, 20),
            ledger=proposal_ledger,
        )
    proposal_ledger.seal_completion()
    _rows, _receipt, proposal_contents = propose_module._proposal_inventory_material(
        captured_proposal_preflight,
        approval_sha256=proposal_approval_sha256,
        sealed=proposal_ledger.read_sealed_results(),
    )
    proposal_inventory_dir = root / "proposal-inventory"
    propose_module._publish_proposal_inventory(
        proposal_inventory_dir, proposal_contents
    )

    class ValidationTokenizer:
        def apply_chat_template(
            self,
            messages: object,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> list[int]:
            assert isinstance(messages, list)
            assert tokenize is True
            assert add_generation_prompt is True
            return list(range(max(1, len(json.dumps(messages).encode()) // 7)))

    validation_preflight = (
        validate_module._build_validation_preflight_with_tokenizer(
            contract_dir,
            proposal_inventory_dir=proposal_inventory_dir,
            proposal_preflight_dir=proposal_preflight_dir,
            proposal_ledger_dir=proposal_ledger_dir,
            tokenizer=ValidationTokenizer(),
        )
    )
    validation_preflight_dir = root / "validation-preflight"
    validate_module.publish_validation_preflight(
        validation_preflight, validation_preflight_dir
    )
    validation_preflight_source = (
        validation_preflight_dir / "receipt.json"
    ).read_bytes()
    validation_preflight_sha256 = hashlib.sha256(
        validation_preflight_source
    ).hexdigest()
    validation_ledger_dir = root / "validation-ledger"
    validation_approval = {
        "schema_version": validate_module.VALIDATION_APPROVAL_SCHEMA_VERSION,
        "stage": "validation",
        "validator_role": validation_preflight["validator_role"],
        "preflight_sha256": validation_preflight_sha256,
        "model": validation_preflight["model"],
        "model_revision": validation_preflight["model_revision"],
        "model_snapshot_manifest_sha256": validation_preflight[
            "model_snapshot_manifest_sha256"
        ],
        "tokenizer_identity_sha256": validation_preflight[
            "tokenizer_identity_sha256"
        ],
        "job_count": validation_preflight["job_count"],
        "primary_call_count": validation_preflight["primary_call_count"],
        "retry_call_ceiling": validation_preflight["retry_call_ceiling"],
        "worst_case_call_ceiling": validation_preflight[
            "worst_case_call_ceiling"
        ],
        "ledger_dir": str(validation_ledger_dir.resolve()),
        "approved": True,
    }
    validation_approval_path = root / "validation-approval.json"
    validation_approval_source = _pretty(validation_approval)
    validation_approval_path.write_bytes(validation_approval_source)
    validation_approval_sha256 = hashlib.sha256(
        validation_approval_source
    ).hexdigest()
    validation_jobs = validation_preflight["jobs"]
    assert isinstance(validation_jobs, list) and validation_jobs
    validation_anchor = validate_module._build_validation_run_anchor(
        validation_jobs,
        preflight_sha256=validation_preflight_sha256,
        approval_sha256=validation_approval_sha256,
        provenance=validate_module._validation_anchor_provenance(
            validation_preflight
        ),
    )
    validation_ledger = AppendOnlyAttemptLedger(
        validation_ledger_dir, expected_anchor=validation_anchor, create_only=True
    )

    class ValidationModel:
        def __init__(self) -> None:
            self.unit_ids = {
                canonical_sha256(job["messages"]): str(job["input_unit_ids"][0])
                for job in validation_jobs
            }

        def generate(self, messages, _schema, *, max_new_tokens):
            assert max_new_tokens == validate_module.PRIMARY_MAX_NEW_TOKENS
            return (
                json.dumps(
                    {
                        "decision": "SUPPORTED",
                        "support_unit_ids": [
                            self.unit_ids[canonical_sha256(messages)]
                        ],
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode(),
                20,
            )

    validation_model = ValidationModel()
    for job in validation_jobs:
        validate_module.run_validation_job_with_retry(
            job,
            ledger=validation_ledger,
            model=validation_model,
        )
    validation_ledger.seal_completion()
    del (
        captured_proposal_preflight,
        proposal_anchor,
        proposal_ledger,
        selected_job,
        proposal_contents,
        validation_ledger,
        validation_model,
        job,
    )
    gc.collect()
    validation_output_dir = root / "validation-output"
    validate_module.finalize_validation_inventory(
        preflight_dir=validation_preflight_dir,
        approval_path=validation_approval_path,
        ledger_dir=validation_ledger_dir,
        output_dir=validation_output_dir,
        proposal_inventory_dir=proposal_inventory_dir,
        proposal_preflight_dir=proposal_preflight_dir,
        proposal_ledger_dir=proposal_ledger_dir,
        source_contract_dir=contract_dir,
    )
    return {
        "contract_dir": contract_dir,
        "proposal_preflight_dir": proposal_preflight_dir,
        "proposal_approval_path": proposal_approval_path,
        "proposal_ledger_dir": proposal_ledger_dir,
        "proposal_inventory_dir": proposal_inventory_dir,
        "validation_preflight_dir": validation_preflight_dir,
        "validation_approval_path": validation_approval_path,
        "validation_ledger_dir": validation_ledger_dir,
        "validation_output_dir": validation_output_dir,
        "validation_jobs": validation_jobs,
    }


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
    )
    module._publish_preflight(output, receipt)
    approval = tmp_path / "approval.json"
    _write_approval(output, approval)
    return output, approval, cache, receipt


def _execute_fixture(
    preflight: Path, approval: Path, *, api_token: str | None = None
) -> dict[str, object]:
    approval_value, approval_sha256 = module._capture_retrieval_approval(approval)
    receipt, receipt_source = module._capture_retrieval_preflight(
        preflight, expected_sha256=str(approval_value["preflight_sha256"])
    )
    module._verify_approval_against_preflight(approval_value, receipt)
    return module._execute_verified_retrieval(
        receipt=receipt,
        receipt_source=receipt_source,
        approval_sha256=approval_sha256,
        api_token=api_token,
    )


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
    assert "output_dir" not in inspect.signature(
        module._audit_retrieval_cache_rows
    ).parameters


def test_forged_free_row_preflight_cannot_verify_or_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    receipt = module._audit_retrieval_cache_rows(
        [_accepted_row()],
        endpoint=ENDPOINT,
        cache_root=cache,
        retrieval_output_dir=tmp_path / "run",
        cache_loader=lambda _job: None,
    )
    preflight = tmp_path / "forged-preflight"
    module._publish_preflight(preflight, receipt)
    with pytest.raises(ValueError, match="retrieval input"):
        verify_retrieval_preflight(preflight)

    approval = tmp_path / "approval.json"
    _write_approval(preflight, approval)
    touched: list[str] = []
    monkeypatch.setattr(
        module,
        "_SecureCacheStore",
        lambda *_args, **_kwargs: touched.append("cache"),
    )
    monkeypatch.setattr(
        module,
        "_build_approved_transport",
        lambda *_args, **_kwargs: touched.append("transport"),
    )
    monkeypatch.setattr(
        module,
        "_claim_output_root",
        lambda *_args, **_kwargs: touched.append("output"),
    )
    with pytest.raises(ValueError, match="retrieval input"):
        execute_retrieval(preflight_dir=preflight, approval_path=approval)
    assert touched == []
    assert not (tmp_path / "run").exists()


def test_cache_probe_rejects_malformed_status_without_transport() -> None:
    with pytest.raises(ValueError, match="cache probe"):
        module._audit_retrieval_cache_rows(
            [_accepted_row()],
            endpoint=ENDPOINT,
            retrieval_output_dir=Path("/var/tmp/task5-unused"),
            cache_loader=lambda _job: {"hit": True, "raw_bytes": -1},
        )


def test_preflight_is_create_only_hash_bound_and_verifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "preflight"
    (tmp_path / "cache").mkdir()
    receipt = module._audit_retrieval_cache_rows(
        [_accepted_row()],
        endpoint=ENDPOINT,
        cache_root=tmp_path / "cache",
        retrieval_output_dir=tmp_path / "run",
    )
    module._publish_preflight(output, receipt)
    assert module._parse_preflight_source(
        module._capture_preflight_source(output)
    ) == receipt
    with pytest.raises(FileExistsError):
        module._publish_preflight(output, receipt)

    mutated = json.loads((output / "receipt.json").read_bytes())
    mutated["hits"] = 100
    (output / "receipt.json").write_bytes(_pretty(mutated))
    with pytest.raises(ValueError):
        module._parse_preflight_source(module._capture_preflight_source(output))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"topic_ids": ["219"]}, "topic_ids"),
        ({"unexpected": "not schema metadata"}, "fields"),
    ],
)
def test_preflight_receipt_has_an_exact_closed_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
    message: str,
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
        receipt = module._audit_retrieval_cache_rows(
            [_accepted_row()],
            endpoint=ENDPOINT,
            cache_root=tmp_path / "cache",
            retrieval_output_dir=tmp_path / "run",
            cache_loader=lambda _job: None,
        )
        module._publish_preflight(linked / "preflight", receipt)
    assert not (real / "preflight").exists()


def test_preflight_rejects_symlink_root_and_unexpected_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
    message: str,
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    result = _execute_fixture(preflight, approval)
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
    result = _execute_fixture(preflight, approval)
    assert result["candidate_rows"] == 1000
    assert not list(outside_cache.iterdir())
    assert not list(outside_output.iterdir())
    assert list(held_cache.rglob("*.manifest.json"))
    assert (held_output / "summary.json").is_file()


def test_fully_cached_execution_still_requires_approval_but_never_builds_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    result = _execute_fixture(preflight, approval)
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
        _execute_fixture(preflight, approval)
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


def test_response_normalization_rejects_nested_docid_only_documents() -> None:
    candidates = [
        {
            "rank": rank,
            "docid": f"d{rank - 1}",
            "score": 1.0,
            "doc": {"docid": f"nested-{rank - 1}"},
        }
        for rank in range(1, RETRIEVAL_HITS + 1)
    ]

    with pytest.raises(ValueError, match="text-bearing"):
        module._normalize_response(json.dumps({"candidates": candidates}).encode())


def test_response_normalization_uses_only_explicit_nested_content() -> None:
    candidates = [
        {
            "rank": rank,
            "docid": f"d{rank - 1}",
            "score": 1.0,
            "doc": {
                "docid": f"nested-{rank - 1}",
                "url": f"https://example.test/{rank - 1}",
                "title": f"Metadata title {rank - 1}",
                "score": 99.0,
                "text": f"Explicit document content {rank - 1}.",
            },
        }
        for rank in range(1, RETRIEVAL_HITS + 1)
    ]

    rows = module._normalize_response(
        json.dumps({"candidates": candidates}).encode()
    )

    assert rows[0]["text"] == "Explicit document content 0."
    assert "nested-0" not in str(rows[0]["text"])
    assert "Metadata title" not in str(rows[0]["text"])
    assert "example.test" not in str(rows[0]["text"])


def test_response_normalization_rejects_arbitrary_nested_metadata() -> None:
    candidates = [
        {
            "rank": rank,
            "docid": f"d{rank - 1}",
            "score": 1.0,
            "doc": {
                "metadata": ["not content", f"tag-{rank - 1}", 42],
                "url": f"https://example.test/{rank - 1}",
                "quality_score": 0.9,
            },
        }
        for rank in range(1, RETRIEVAL_HITS + 1)
    ]

    with pytest.raises(ValueError, match="text-bearing"):
        module._normalize_response(json.dumps({"results": candidates}).encode())


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
        _execute_fixture(preflight, approval)
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
        _execute_fixture(preflight, approval)
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
    tmp_path: Path, real_v2_validation_handoff: dict[str, object]
) -> None:
    paths = real_v2_validation_handoff
    contract_dir = Path(paths["contract_dir"])
    validation = validate_module.load_authenticated_validation_inventory(
        output_dir=Path(paths["validation_output_dir"]),
        preflight_dir=Path(paths["validation_preflight_dir"]),
        approval_path=Path(paths["validation_approval_path"]),
        ledger_dir=Path(paths["validation_ledger_dir"]),
        proposal_inventory_dir=Path(paths["proposal_inventory_dir"]),
        proposal_preflight_dir=Path(paths["proposal_preflight_dir"]),
        proposal_ledger_dir=Path(paths["proposal_ledger_dir"]),
        source_contract_dir=contract_dir,
    )
    validation_jobs = paths["validation_jobs"]
    assert isinstance(validation_jobs, list) and len(validation_jobs) == 1
    assert validation["receipt"]["approval_sha256"] == hashlib.sha256(
        Path(paths["validation_approval_path"]).read_bytes()
    ).hexdigest()
    assert validation["accepted"][0]["proposal_id"] == validation_jobs[0][
        "proposal_id"
    ]
    assert validation["accepted"][0]["validation_support_unit_ids"] == [
        validation_jobs[0]["input_unit_ids"][0]
    ]
    contract = validate_module._load_verified_contract(contract_dir)
    parent = next(
        row for row in contract["parents"] if row["parent_id"] == "219-positive"
    )
    retrieval_inputs = tmp_path / "retrieval-inputs"
    receipt = freeze_retrieval_inputs(
        validation_output_dir=Path(paths["validation_output_dir"]),
        validation_preflight_dir=Path(paths["validation_preflight_dir"]),
        validation_approval_path=Path(paths["validation_approval_path"]),
        validation_ledger_dir=Path(paths["validation_ledger_dir"]),
        proposal_inventory_dir=Path(paths["proposal_inventory_dir"]),
        proposal_preflight_dir=Path(paths["proposal_preflight_dir"]),
        proposal_ledger_dir=Path(paths["proposal_ledger_dir"]),
        contract_dir=contract_dir,
        output_dir=retrieval_inputs,
    )
    authenticated = load_authenticated_retrieval_inputs(retrieval_inputs)
    assert receipt == authenticated["receipt"]
    (retrieval_inputs / "unexpected.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        load_authenticated_retrieval_inputs(retrieval_inputs)
    (retrieval_inputs / "unexpected.txt").unlink()
    cache = tmp_path / "public-cache"
    cache.mkdir()
    retrieval_preflight = tmp_path / "retrieval-preflight"
    public_receipt = audit_retrieval_cache(
        retrieval_inputs,
        endpoint=ENDPOINT,
        cache_root=cache,
        retrieval_output_dir=tmp_path / "retrieval-output",
        output_dir=retrieval_preflight,
    )
    assert public_receipt["planned_request_count"] == 1
    assert verify_retrieval_preflight(retrieval_preflight) == public_receipt
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
    assert row["o1_label"] == "access barriers for disadvantaged communities"
    assert len(row["validation_support_unit_ids"]) == 1
    assert len(row["proposal_document_ids"]) == 1
    assert row["source_manifest"]["sha256"] == (
        "14954020d5edae579d31a1f2f68e6a98b21b151de139aed664615884a2013eb4"
    )

    tampered_output = tmp_path / "tampered-validation"
    tampered_output.mkdir()
    for name in ("validated.jsonl", "accepted.jsonl", "receipt.json"):
        (tampered_output / name).write_bytes(
            (Path(paths["validation_output_dir"]) / name).read_bytes()
        )
    accepted_path = tampered_output / "accepted.jsonl"
    accepted_path.write_bytes(
        accepted_path.read_bytes().replace(b"access", b"denied")
    )
    with pytest.raises(ValueError, match="sealed ledger replay"):
        validate_module.load_authenticated_validation_inventory(
            output_dir=tampered_output,
            preflight_dir=Path(paths["validation_preflight_dir"]),
            approval_path=Path(paths["validation_approval_path"]),
            ledger_dir=Path(paths["validation_ledger_dir"]),
            proposal_inventory_dir=Path(paths["proposal_inventory_dir"]),
            proposal_preflight_dir=Path(paths["proposal_preflight_dir"]),
            proposal_ledger_dir=Path(paths["proposal_ledger_dir"]),
            source_contract_dir=contract_dir,
        )


def test_validation_finalizer_rejects_forged_upstream_hashes_before_ledger(
    tmp_path: Path, real_v2_validation_handoff: dict[str, object]
) -> None:
    paths = real_v2_validation_handoff
    forged = json.loads(
        (Path(paths["validation_preflight_dir"]) / "receipt.json").read_bytes()
    )
    forged["source_contract_receipt_sha256"] = "f" * 64
    proposal_receipt = forged["proposal_receipt"]
    assert isinstance(proposal_receipt, dict)
    proposal_receipt["source_contract_receipt_sha256"] = "f" * 64
    proposal_receipt["sha256"] = "e" * 64
    proposal_receipt["proposals_sha256"] = "d" * 64
    forged_preflight = tmp_path / "forged-validation-preflight"
    validate_module.publish_validation_preflight(forged, forged_preflight)
    forged_source = (forged_preflight / "receipt.json").read_bytes()
    approval = {
        "schema_version": validate_module.VALIDATION_APPROVAL_SCHEMA_VERSION,
        "stage": "validation",
        "validator_role": forged["validator_role"],
        "preflight_sha256": hashlib.sha256(forged_source).hexdigest(),
        "model": forged["model"],
        "model_revision": forged["model_revision"],
        "model_snapshot_manifest_sha256": forged[
            "model_snapshot_manifest_sha256"
        ],
        "tokenizer_identity_sha256": forged["tokenizer_identity_sha256"],
        "job_count": forged["job_count"],
        "primary_call_count": forged["primary_call_count"],
        "retry_call_ceiling": forged["retry_call_ceiling"],
        "worst_case_call_ceiling": forged["worst_case_call_ceiling"],
        "ledger_dir": str((tmp_path / "missing-ledger").resolve()),
        "approved": True,
    }
    approval_path = tmp_path / "forged-validation-approval.json"
    approval_path.write_bytes(validate_module._pretty_bytes(approval))
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="authenticated upstream replay"):
        validate_module.finalize_validation_inventory(
            preflight_dir=forged_preflight,
            approval_path=approval_path,
            ledger_dir=tmp_path / "missing-ledger",
            output_dir=output,
            proposal_inventory_dir=Path(paths["proposal_inventory_dir"]),
            proposal_preflight_dir=Path(paths["proposal_preflight_dir"]),
            proposal_ledger_dir=Path(paths["proposal_ledger_dir"]),
            source_contract_dir=Path(paths["contract_dir"]),
        )
    assert not output.exists()


@pytest.mark.parametrize("publisher", ["retrieval-inputs", "preflight"])
def test_task5_publish_race_is_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: str
) -> None:
    destination = tmp_path / publisher
    real = getattr(module, "_rename_noreplace_at", None)

    def race(parent_fd: int, staging_name: str, destination_name: str) -> None:
        os.mkdir(destination_name, mode=0o700, dir_fd=parent_fd)
        assert real is not None
        real(parent_fd, staging_name, destination_name)

    monkeypatch.setattr(module, "_rename_noreplace_at", race, raising=False)
    with pytest.raises(FileExistsError, match="create-only"):
        if publisher == "retrieval-inputs":
            module._publish_retrieval_inputs(
                destination,
                {"inputs.jsonl": b"", "receipt.json": b"{}\n"},
            )
        else:
            module._publish_preflight(destination, {"status": "fixture"})
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
