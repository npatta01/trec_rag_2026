from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag import det_sparse_v4_contract as v4
from trec_rag.query_schema_compat import require_vllm_xgrammar_compatible


def test_v4_denied_topic_sets_are_frozen_pairwise_disjoint_and_complete():
    v4.validate_denied_topic_sets()

    assert v4.KNOWN_FIVE_TOPIC_IDS == ("144", "213", "224", "407", "515")
    assert v4.V1_TOPIC_IDS == ("200", "225", "707", "897")
    assert v4.V2_TOPIC_IDS == ("37", "84", "161", "300")
    assert v4.V3_TOPIC_IDS == (
        "14",
        "31",
        "58",
        "72",
        "219",
        "233",
        "273",
        "477",
        "499",
    )
    assert len(v4.DENIED_TOPIC_IDS) == 22
    assert v4.DENIED_TOPIC_IDS == tuple(sorted(v4.DENIED_TOPIC_IDS, key=int))


def test_canonical_json_bytes_are_compact_sorted_utf8_and_reject_nan():
    payload = {"z": "café", "a": [2, 1]}
    encoded = v4.canonical_json_bytes(payload)

    assert encoded == b'{"a":[2,1],"z":"caf\xc3\xa9"}'
    assert v4.sha256_bytes(encoded) == v4.sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    )
    with pytest.raises(ValueError):
        v4.canonical_json_bytes({"bad": float("nan")})


def test_offset_request_hashes_decoded_text_bytes_not_json_envelope():
    request, body, body_sha256 = v4.build_offset_request("Café 🚀")

    assert request == {
        "schema_version": "lucene_whole_unit_offsets_request_v1",
        "text": "Café 🚀",
    }
    assert body == (
        b'{"schema_version":"lucene_whole_unit_offsets_request_v1",'
        b'"text":"Caf\xc3\xa9 \xf0\x9f\x9a\x80"}'
    )
    assert body_sha256 == v4.sha256_bytes(body)
    assert v4.text_sha256("Café 🚀") != body_sha256


def _offset_fingerprint():
    return {
        "legacy_term_chain_fingerprint_sha256": "a" * 64,
        "offset_contract_version": "lucene_whole_unit_offsets_v1",
        "offset_server_class_sha256": "b" * 64,
        "offset_lucene_jar_sha256": "c" * 64,
        "offset_runtime_image_digest": "sha256:" + "d" * 64,
    }


def _offset_response(text: str = "Café rocket🚀"):
    return {
        "schema_version": "lucene_whole_unit_offsets_response_v1",
        "text_sha256": v4.text_sha256(text),
        "offset_unit": "unicode_code_points",
        "fingerprint": _offset_fingerprint(),
        "occurrences": [
            {
                "ordinal": 0,
                "term": "café",
                "start_codepoint": 0,
                "end_codepoint": 4,
                "position_increment": 1,
            },
            {
                "ordinal": 1,
                "term": "rocket",
                "start_codepoint": 5,
                "end_codepoint": 11,
                "position_increment": 1,
            },
            {
                "ordinal": 2,
                "term": "🚀",
                "start_codepoint": 11,
                "end_codepoint": 12,
                "position_increment": 1,
            },
        ],
    }


def test_offset_response_shape_accepts_strict_codepoint_offsets_and_fingerprint():
    text = "Café rocket🚀"
    response = _offset_response(text)

    v4.validate_offset_response_shape(
        response,
        text=text,
        expected_fingerprint=_offset_fingerprint(),
    )
    empty = _offset_response("")
    empty["occurrences"] = []
    v4.validate_offset_response_shape(empty, text="")


def test_offset_response_rejects_hash_schema_unit_and_fingerprint_drift():
    text = "Café rocket🚀"
    response = _offset_response(text)

    bad = dict(response, text_sha256="0" * 64)
    with pytest.raises(ValueError, match="text_sha256"):
        v4.validate_offset_response_shape(bad, text=text)

    bad = dict(response, schema_version="future")
    with pytest.raises(ValueError, match="schema_version"):
        v4.validate_offset_response_shape(bad, text=text)

    bad = dict(response, offset_unit="utf16")
    with pytest.raises(ValueError, match="offset_unit"):
        v4.validate_offset_response_shape(bad, text=text)

    bad_fingerprint = dict(_offset_fingerprint(), offset_lucene_jar_sha256="C" * 64)
    bad = dict(response, fingerprint=bad_fingerprint)
    with pytest.raises(ValueError, match="lowercase sha256"):
        v4.validate_offset_response_shape(bad, text=text)

    with pytest.raises(ValueError, match="fingerprint drift"):
        v4.validate_offset_response_shape(
            response,
            text=text,
            expected_fingerprint=dict(
                _offset_fingerprint(),
                offset_server_class_sha256="e" * 64,
            ),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda occurrence: occurrence.update({"ordinal": 9}), "ordinal gap"),
        (lambda occurrence: occurrence.update({"term": ""}), "term"),
        (lambda occurrence: occurrence.update({"start_codepoint": True}), "integer"),
        (lambda occurrence: occurrence.update({"start_codepoint": -1}), "nonempty"),
        (lambda occurrence: occurrence.update({"end_codepoint": 99}), "text length"),
        (lambda occurrence: occurrence.update({"end_codepoint": occurrence["start_codepoint"]}), "nonempty"),
        (lambda occurrence: occurrence.update({"position_increment": -1}), "position_increment"),
    ],
)
def test_offset_response_rejects_invalid_occurrence_rows(mutation, message):
    text = "Café rocket🚀"
    response = _offset_response(text)
    response["occurrences"] = [dict(row) for row in response["occurrences"]]
    mutation(response["occurrences"][1])

    with pytest.raises(ValueError, match=message):
        v4.validate_offset_response_shape(response, text=text)


def test_offset_response_rejects_nonmonotone_occurrence_spans_and_extra_fields():
    text = "Café rocket🚀"
    response = _offset_response(text)
    response["occurrences"] = [dict(row) for row in response["occurrences"]]
    response["occurrences"][2]["start_codepoint"] = 4
    response["occurrences"][2]["end_codepoint"] = 12
    with pytest.raises(ValueError, match="starts are nonmonotone"):
        v4.validate_offset_response_shape(response, text=text)

    response = _offset_response(text)
    response["occurrences"] = [dict(row) for row in response["occurrences"]]
    response["occurrences"][0]["end_codepoint"] = 10
    response["occurrences"][1]["start_codepoint"] = 5
    response["occurrences"][1]["end_codepoint"] = 6
    with pytest.raises(ValueError, match="ends are nonmonotone"):
        v4.validate_offset_response_shape(response, text=text)

    response = _offset_response(text)
    response["extra"] = True
    with pytest.raises(ValueError, match="extra"):
        v4.validate_offset_response_shape(response, text=text)

    response = _offset_response(text)
    response["occurrences"] = [dict(row) for row in response["occurrences"]]
    response["occurrences"][0]["extra"] = True
    with pytest.raises(ValueError, match="extra"):
        v4.validate_offset_response_shape(response, text=text)


def test_model_response_schema_is_xgrammar_friendly_and_case_bounded():
    schema = v4.model_response_schema("case-001", u1_token_count=7)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "schema_version",
        "case_id",
        "decision",
        "start_token",
        "end_token",
    ]
    assert schema["properties"]["schema_version"] == {
        "const": "semantic_anchor_response_v1",
        "type": "string",
    }
    assert schema["properties"]["case_id"] == {"const": "case-001", "type": "string"}
    assert schema["properties"]["decision"] == {
        "enum": ["select", "abstain"],
        "type": "string",
    }
    assert schema["properties"]["start_token"] == {
        "type": "integer",
        "minimum": -1,
        "maximum": 7,
    }
    assert "oneOf" not in json.dumps(schema)
    assert "default" not in json.dumps(schema)


def test_model_response_shape_enforces_sentinel_bounds_and_no_extra_fields():
    v4.validate_model_response_shape(
        {
            "schema_version": "semantic_anchor_response_v1",
            "case_id": "case-001",
            "decision": "select",
            "start_token": 2,
            "end_token": 4,
        },
        case_id="case-001",
        u1_token_count=7,
    )
    v4.validate_model_response_shape(
        {
            "schema_version": "semantic_anchor_response_v1",
            "case_id": "case-001",
            "decision": "abstain",
            "start_token": -1,
            "end_token": -1,
        },
        case_id="case-001",
        u1_token_count=7,
    )

    with pytest.raises(ValueError, match="extra"):
        v4.validate_model_response_shape(
            {
                "schema_version": "semantic_anchor_response_v1",
                "case_id": "case-001",
                "decision": "select",
                "start_token": 2,
                "end_token": 4,
                "confidence": 0.9,
            },
            case_id="case-001",
            u1_token_count=7,
        )
    with pytest.raises(ValueError, match="abstain requires"):
        v4.validate_model_response_shape(
            {
                "schema_version": "semantic_anchor_response_v1",
                "case_id": "case-001",
                "decision": "abstain",
                "start_token": 0,
                "end_token": 0,
            },
            case_id="case-001",
            u1_token_count=7,
        )
    with pytest.raises(ValueError, match="integer"):
        v4.validate_model_response_shape(
            {
                "schema_version": "semantic_anchor_response_v1",
                "case_id": "case-001",
                "decision": "select",
                "start_token": True,
                "end_token": 2,
            },
            case_id="case-001",
            u1_token_count=7,
        )


def test_decision_boundary_grid_is_72_cells_with_distinct_boundaries():
    grid = v4.decision_boundary_grid(u1_token_count=7)

    assert len(grid) == 72
    assert len({case.start_token for case in grid}) == 6
    assert len({case.end_token for case in grid}) == 6
    valid = {(case.decision, case.start_token, case.end_token) for case in grid if case.is_valid}
    assert ("abstain", -1, -1) in valid
    assert ("select", 0, 1) in valid
    assert ("select", 6, 7) in valid
    assert ("select", -1, -1) not in valid
    assert ("abstain", 0, 1) not in valid

    with pytest.raises(ValueError, match="at least 3"):
        v4.decision_boundary_grid(u1_token_count=2)


def test_static_import_audit_rejects_topic_retrieval_and_legacy_planner_imports(tmp_path: Path):
    good = tmp_path / "good.py"
    good.write_text("import json\nfrom pathlib import Path\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from trec_rag.topics import load_topics\n"
        "import trec_rag.query_planner as legacy\n"
        "from trec_rag.remote_pyserini import SearchClient\n",
        encoding="utf-8",
    )

    assert v4.audit_imports([good]) == ()
    issues = v4.audit_imports([bad])
    assert [issue.module for issue in issues] == [
        "trec_rag.topics",
        "trec_rag.query_planner",
        "trec_rag.remote_pyserini",
    ]

    sneaky = tmp_path / "sneaky.py"
    sneaky.write_text(
        "from trec_rag import topics\n"
        "from trec_rag import query_planner as planner\n",
        encoding="utf-8",
    )

    assert [issue.module for issue in v4.audit_imports([sneaky])] == [
        "trec_rag.topics",
        "trec_rag.query_planner",
    ]


def test_denied_path_fragment_audit_catches_real_data_and_external_artifacts():
    text = (
        "trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv\n"
        "cache/retrieval/run.jsonl\n"
        "research-rubrics/example.qrels\n"
    )

    assert v4.audit_denied_path_fragments(text) == (
        "rag25-topics-dev.tsv",
        "research-rubrics",
        "qrels",
        "cache/retrieval",
    )


def _receipt(**overrides):
    receipt = {
        "schema_version": "semantic_anchor_terminal_receipt_v1",
        "terminal_state": "preflight_no_go",
        "attempted_calls": 0,
        "completed_calls": 0,
        "raw_committed_calls": 0,
        "gold_opened": False,
        "artifact_sha256": {"manifest": "a" * 64},
    }
    receipt.update(overrides)
    return receipt


def _case_order():
    return tuple(f"synthetic-case-{index:03d}" for index in range(1, 25))


def _reservation(case_id="synthetic-case-001", request_sha256="1" * 64):
    return {
        "schema_version": "semantic_anchor_reservation_v1",
        "run_id": "run-001",
        "case_id": case_id,
        "request_sha256": request_sha256,
        "create_only": True,
    }


def _dispatch(case_id="synthetic-case-001", request_sha256="1" * 64):
    return {
        "schema_version": "semantic_anchor_dispatch_record_v1",
        "case_id": case_id,
        "request_bytes_sha256": request_sha256,
        "loopback_only": True,
        "dispatch_counted": True,
    }


def _raw_response(case_id="synthetic-case-001", body_sha256="2" * 64):
    return {
        "schema_version": "semantic_anchor_raw_response_body_v1",
        "case_id": case_id,
        "http_status": 200,
        "finish_reason": "stop",
        "served_model": "gpt-oss-local",
        "body_size_bytes": 42,
        "body_sha256": body_sha256,
    }


def _transport_failure(case_id="synthetic-case-001", request_sha256="1" * 64):
    return {
        "schema_version": "semantic_anchor_transport_failure_v1",
        "case_id": case_id,
        "request_bytes_sha256": request_sha256,
        "exception_class": "TimeoutError",
        "exception_message": "synthetic timeout",
    }


def _case_receipt(
    case_id="synthetic-case-001",
    request_sha256="1" * 64,
    raw_response_sha256="2" * 64,
):
    return {
        "schema_version": "semantic_anchor_case_receipt_v1",
        "case_id": case_id,
        "request_sha256": request_sha256,
        "machine_status": "mechanical_pass",
        "raw_response_sha256": raw_response_sha256,
    }


def _manifest(artifact_sha256=None):
    return {
        "schema_version": "semantic_anchor_run_manifest_v1",
        "case_count": 24,
        "artifact_sha256": artifact_sha256 or {"manifest": "a" * 64},
        "terminal_receipt_path": "terminal_receipt.json",
    }


def test_terminal_receipt_counter_and_gold_rules_are_fail_closed():
    v4.validate_terminal_receipt(_receipt())
    v4.validate_terminal_receipt(
        _receipt(
            terminal_state="completed_synthetic_go",
            attempted_calls=24,
            completed_calls=24,
            raw_committed_calls=24,
            gold_opened=True,
        )
    )

    with pytest.raises(ValueError, match="gold must remain unopened"):
        v4.validate_terminal_receipt(_receipt(terminal_state="prefix_integrity_no_go", gold_opened=True))
    with pytest.raises(ValueError, match="completed states require 24"):
        v4.validate_terminal_receipt(
            _receipt(
                terminal_state="completed_qualification_no_go",
                attempted_calls=23,
                completed_calls=23,
                raw_committed_calls=23,
                gold_opened=True,
            )
        )
    with pytest.raises(ValueError, match="first_case_no_go"):
        v4.validate_terminal_receipt(_receipt(terminal_state="first_case_no_go", attempted_calls=0))
    with pytest.raises(ValueError, match="raw_committed_calls"):
        v4.validate_terminal_receipt(
            _receipt(
                terminal_state="transport_no_body_no_go",
                attempted_calls=1,
                completed_calls=0,
                raw_committed_calls=1,
            )
        )


def test_v4_ledger_prefix_accepts_raw_first_single_case_prefix():
    artifact_sha256 = {"manifest": "a" * 64}

    v4.validate_ledger_prefix(
        case_order=_case_order(),
        reservations=[_reservation()],
        dispatches=[_dispatch()],
        raw_responses=[_raw_response()],
        transport_failures=[],
        case_receipts=[_case_receipt()],
        terminal_receipt=_receipt(
            terminal_state="interrupted_incomplete",
            attempted_calls=1,
            completed_calls=1,
            raw_committed_calls=1,
            artifact_sha256=artifact_sha256,
        ),
        run_manifest=_manifest(artifact_sha256),
    )


def test_v4_ledger_prefix_accepts_transport_failure_without_raw_response():
    v4.validate_ledger_prefix(
        case_order=_case_order(),
        reservations=[_reservation()],
        dispatches=[_dispatch()],
        raw_responses=[],
        transport_failures=[_transport_failure()],
        case_receipts=[],
        terminal_receipt=_receipt(
            terminal_state="transport_no_body_no_go",
            attempted_calls=1,
            completed_calls=0,
            raw_committed_calls=0,
        ),
    )


def test_v4_ledger_prefix_rejects_missing_reservation_and_hash_drift():
    with pytest.raises(ValueError, match="dispatch lacks reservation"):
        v4.validate_ledger_prefix(
            case_order=_case_order(),
            reservations=[],
            dispatches=[_dispatch()],
            raw_responses=[],
            transport_failures=[],
            case_receipts=[],
            terminal_receipt=_receipt(
                terminal_state="interrupted_incomplete",
                attempted_calls=1,
                completed_calls=0,
                raw_committed_calls=0,
            ),
        )

    with pytest.raises(ValueError, match="dispatch request hash"):
        v4.validate_ledger_prefix(
            case_order=_case_order(),
            reservations=[_reservation(request_sha256="1" * 64)],
            dispatches=[_dispatch(request_sha256="9" * 64)],
            raw_responses=[],
            transport_failures=[],
            case_receipts=[],
            terminal_receipt=_receipt(
                terminal_state="interrupted_incomplete",
                attempted_calls=1,
                completed_calls=0,
                raw_committed_calls=0,
            ),
        )


def test_v4_ledger_prefix_rejects_terminal_counter_and_manifest_drift():
    with pytest.raises(ValueError, match="attempted_calls"):
        v4.validate_ledger_prefix(
            case_order=_case_order(),
            reservations=[_reservation()],
            dispatches=[_dispatch()],
            raw_responses=[],
            transport_failures=[],
            case_receipts=[],
            terminal_receipt=_receipt(
                terminal_state="interrupted_incomplete",
                attempted_calls=0,
                completed_calls=0,
                raw_committed_calls=0,
            ),
        )

    with pytest.raises(ValueError, match="artifact hashes differ"):
        v4.validate_ledger_prefix(
            case_order=_case_order(),
            reservations=[_reservation()],
            dispatches=[_dispatch()],
            raw_responses=[_raw_response()],
            transport_failures=[],
            case_receipts=[_case_receipt()],
            terminal_receipt=_receipt(
                terminal_state="interrupted_incomplete",
                attempted_calls=1,
                completed_calls=1,
                raw_committed_calls=1,
                artifact_sha256={"manifest": "a" * 64},
            ),
            run_manifest=_manifest({"manifest": "b" * 64}),
        )


def test_v4_ledger_prefix_accepts_completed_24_case_terminal_state():
    case_order = _case_order()
    reservations = []
    dispatches = []
    raw_responses = []
    receipts = []
    for index, case_id in enumerate(case_order, start=1):
        request_sha256 = f"{index:064x}"[-64:]
        body_sha256 = f"{index + 100:064x}"[-64:]
        reservations.append(_reservation(case_id, request_sha256))
        dispatches.append(_dispatch(case_id, request_sha256))
        raw_responses.append(_raw_response(case_id, body_sha256))
        receipts.append(_case_receipt(case_id, request_sha256, body_sha256))

    v4.validate_ledger_prefix(
        case_order=case_order,
        reservations=reservations,
        dispatches=dispatches,
        raw_responses=raw_responses,
        transport_failures=[],
        case_receipts=receipts,
        terminal_receipt=_receipt(
            terminal_state="completed_synthetic_go",
            attempted_calls=24,
            completed_calls=24,
            raw_committed_calls=24,
            gold_opened=True,
        ),
    )


def _gold_select(case_id="synthetic-case-001"):
    return {
        "schema_version": "semantic_anchor_gold_label_v1",
        "case_id": case_id,
        "decision": "select",
        "acceptable_ranges": [{"start_token": 0, "end_token": 2}],
        "wrong_referent_ranges": [{"start_token": 2, "end_token": 4}],
    }


def _gold_abstain(case_id="synthetic-case-019"):
    return {
        "schema_version": "semantic_anchor_gold_label_v1",
        "case_id": case_id,
        "decision": "abstain",
        "acceptable_ranges": [],
        "wrong_referent_ranges": [],
        "abstain_reason": "coequal_disjoint_subjects",
    }


def _model_response(
    case_id="synthetic-case-001",
    decision="select",
    start_token=0,
    end_token=2,
):
    return {
        "schema_version": "semantic_anchor_response_v1",
        "case_id": case_id,
        "decision": decision,
        "start_token": start_token,
        "end_token": end_token,
    }


def test_gold_label_linter_rejects_leaky_or_ambiguous_ranges():
    v4.validate_gold_label_case(_gold_select(), case_id="synthetic-case-001")
    v4.validate_gold_label_case(_gold_abstain(), case_id="synthetic-case-019")

    bad = _gold_select()
    bad["acceptable_ranges"] = [{"start_token": 4, "end_token": 6}]
    with pytest.raises(ValueError, match="inside U1"):
        v4.validate_gold_label_case(bad, case_id="synthetic-case-001")

    bad = _gold_select()
    bad["wrong_referent_ranges"] = [{"start_token": 0, "end_token": 2}]
    with pytest.raises(ValueError, match="overlap"):
        v4.validate_gold_label_case(bad, case_id="synthetic-case-001")

    bad = _gold_abstain()
    bad["acceptable_ranges"] = [{"start_token": 0, "end_token": 1}]
    with pytest.raises(ValueError, match="abstain gold"):
        v4.validate_gold_label_case(bad, case_id="synthetic-case-019")


def test_scorer_classifies_select_abstain_and_mechanical_failure():
    assert (
        v4.classify_model_response(
            _model_response(),
            gold_case=_gold_select(),
            case_id="synthetic-case-001",
        )
        == "correct_select"
    )
    assert (
        v4.classify_model_response(
            _model_response(start_token=2, end_token=4),
            gold_case=_gold_select(),
            case_id="synthetic-case-001",
        )
        == "wrong_referent"
    )
    assert (
        v4.classify_model_response(
            _model_response(decision="abstain", start_token=-1, end_token=-1),
            gold_case=_gold_select(),
            case_id="synthetic-case-001",
        )
        == "wrong_abstain"
    )
    assert (
        v4.classify_model_response(
            _model_response(
                case_id="synthetic-case-019",
                decision="abstain",
                start_token=-1,
                end_token=-1,
            ),
            gold_case=_gold_abstain(),
            case_id="synthetic-case-019",
        )
        == "safe_abstain"
    )
    assert (
        v4.classify_model_response(
            _model_response(start_token=True, end_token=2),
            gold_case=_gold_select(),
            case_id="synthetic-case-001",
        )
        == "mechanical_failure"
    )


def test_scorer_receipt_is_case_ordered_and_built_from_gold_bundle():
    case_order = _case_order()
    gold = {"schema_version": "semantic_anchor_gold_labels_v1", "cases": []}
    responses = {}
    for index, case_id in enumerate(case_order, start=1):
        if index <= 18:
            gold["cases"].append(_gold_select(case_id))
            responses[case_id] = _model_response(case_id=case_id)
        else:
            gold["cases"].append(_gold_abstain(case_id))
            responses[case_id] = _model_response(
                case_id=case_id,
                decision="abstain",
                start_token=-1,
                end_token=-1,
            )

    receipt = v4.build_scorer_receipt(
        responses,
        gold=gold,
        case_order=case_order,
    )

    v4.validate_scorer_receipt(receipt, case_order=case_order)
    assert receipt["case_results"][0] == {
        "case_id": "synthetic-case-001",
        "classification": "correct_select",
    }
    assert receipt["case_results"][-1] == {
        "case_id": "synthetic-case-024",
        "classification": "safe_abstain",
    }

    bad = dict(receipt)
    bad["case_results"] = list(reversed(receipt["case_results"]))
    with pytest.raises(ValueError, match="case order"):
        v4.validate_scorer_receipt(bad, case_order=case_order)


def test_model_inventory_requires_three_loaded_shards_and_denied_original_weight():
    inventory = {
        "schema_version": "semantic_anchor_model_inventory_attestation_v1",
        "repository": "openai/gpt-oss-20b",
        "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
        "quantization_method": "mxfp4",
        "safetensors_index_total_size": 13_761_264_768,
        "snapshot_path": "/models/openai/gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee",
        "loaded_shards": [
            "model-00001.safetensors",
            "model-00002.safetensors",
            "model-00003.safetensors",
        ],
        "loaded_files": [
            "config.json",
            "tokenizer.json",
            "model-00001.safetensors",
            "model-00002.safetensors",
            "model-00003.safetensors",
        ],
        "unloaded_files": ["README.md", "original/model.safetensors"],
        "denied_files": ["original/model.safetensors"],
        "file_sha256": {
            "config.json": "a" * 64,
            "tokenizer.json": "b" * 64,
            "model-00001.safetensors": "c" * 64,
            "model-00002.safetensors": "d" * 64,
            "model-00003.safetensors": "e" * 64,
            "README.md": "f" * 64,
            "original/model.safetensors": "9" * 64,
        },
    }

    v4.validate_model_inventory(inventory)
    bad = dict(inventory)
    bad["loaded_shards"] = ["model-00001.safetensors", "model-00002.safetensors"]
    with pytest.raises(ValueError, match="exactly three"):
        v4.validate_model_inventory(bad)
    bad = dict(inventory)
    bad["denied_files"] = []
    with pytest.raises(ValueError, match="original/model.safetensors"):
        v4.validate_model_inventory(bad)
    bad = dict(inventory)
    bad["loaded_files"] = [*inventory["loaded_files"], "original/model.safetensors"]
    with pytest.raises(ValueError, match="must not be loaded"):
        v4.validate_model_inventory(bad)
    bad = dict(inventory)
    bad["file_sha256"] = dict(inventory["file_sha256"], **{"config.json": "A" * 64})
    with pytest.raises(ValueError, match="lowercase sha256"):
        v4.validate_model_inventory(bad)


def test_runner_visible_artifacts_are_physically_separated_from_gold_artifacts():
    runner = v4.runner_visible_artifacts()
    scorer = v4.scorer_only_artifacts()

    v4.validate_runner_gold_separation(runner, scorer)
    assert "semantic_anchor_gold_labels_v1.json" not in runner
    assert "semantic_anchor_gold_labels_v1.json" in scorer

    with pytest.raises(ValueError, match="overlap"):
        v4.validate_runner_gold_separation(
            ("semantic_anchor_gold_labels_v1.json",),
            ("semantic_anchor_gold_labels_v1.json",),
        )
    with pytest.raises(ValueError, match="gold"):
        v4.validate_runner_gold_separation(("hidden_gold_copy.json",), scorer)


def test_committed_artifact_bundle_is_24_case_offline_set_and_internally_consistent():
    hashes = v4.validate_artifact_bundle()

    assert "semantic_anchor_artifact_manifest_v1.json" not in hashes
    assert "semantic_anchor_case_registry_v1.json" in hashes
    assert "semantic_anchor_request_fixtures_v1.jsonl" in hashes
    assert "semantic_anchor_request_fixture.case001.json" in hashes
    assert "semantic_anchor_gold_labels_v1.json" in hashes
    assert len(hashes["semantic_anchor_request_fixture.case001.json"]) == 64


def test_case_registry_freezes_24_case_shape_and_category_arithmetic():
    registry = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_case_registry_v1.json"
    )

    case_ids = v4.validate_case_registry(registry)
    assert len(case_ids) == 24
    assert case_ids[:3] == (
        "synthetic-case-001",
        "synthetic-case-002",
        "synthetic-case-003",
    )
    cases = registry["cases"]
    selects = [case for case in cases if case["decision"] == "select"]
    abstains = [case for case in cases if case["decision"] == "abstain"]
    assert len(selects) == 18
    assert len(abstains) == 6
    assert {
        (case["u1_anchor_position"], case["anchor_class"], case["child_reference_style"])
        for case in selects
    } == {
        (position, anchor_class, reference_style)
        for position in v4.SELECT_POSITIONS
        for anchor_class in v4.SELECT_ANCHOR_CLASSES
        for reference_style in v4.SELECT_CHILD_REFERENCE_STYLES
    }
    assert tuple(case["abstain_reason"] for case in abstains) == v4.ABSTAIN_REASONS


def test_case_registry_rejects_missing_cartesian_cell():
    registry = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_case_registry_v1.json"
    )
    mutated = dict(registry)
    mutated["cases"] = [dict(case) for case in registry["cases"]]
    mutated["cases"][0]["child_reference_style"] = "ellipsis_generic"

    with pytest.raises(ValueError, match="Cartesian"):
        v4.validate_case_registry(mutated)


def test_replay_mutation_registry_freezes_expected_failure_surface():
    registry = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_replay_mutation_registry_v1.json"
    )

    mutation_ids = v4.validate_replay_mutation_registry(registry)
    assert mutation_ids == v4.REPLAY_MUTATION_IDS
    assert len(mutation_ids) == 8
    assert len({m["expected_failure_code"] for m in registry["mutations"]}) == 8


def test_replay_mutation_registry_rejects_reordered_ids():
    registry = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_replay_mutation_registry_v1.json"
    )
    mutated = dict(registry)
    mutated["mutations"] = [dict(mutation) for mutation in registry["mutations"]]
    mutated["mutations"][0], mutated["mutations"][1] = (
        mutated["mutations"][1],
        mutated["mutations"][0],
    )

    with pytest.raises(ValueError, match="mutation IDs"):
        v4.validate_replay_mutation_registry(mutated)


def test_renderer_oracle_fixture_is_topic_free_and_ordered():
    oracle = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_renderer_oracle_v1.json"
    )

    oracle_ids = v4.validate_renderer_oracle(oracle)
    assert oracle_ids == v4.RENDERER_ORACLE_IDS
    assert len(oracle_ids) == 3


def test_renderer_oracle_rejects_denied_path_fragment():
    oracle = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_renderer_oracle_v1.json"
    )
    mutated = dict(oracle)
    mutated["oracle_cases"] = [dict(case) for case in oracle["oracle_cases"]]
    mutated["oracle_cases"][0]["expected"] = dict(mutated["oracle_cases"][0]["expected"])
    mutated["oracle_cases"][0]["expected"]["facet_queries"] = [
        "rag25-topics-dev.tsv",
        "Aurora Bridge Who signs its log",
    ]

    with pytest.raises(ValueError, match="denied path"):
        v4.validate_renderer_oracle(mutated)


def test_model_inventory_attestation_fixture_matches_local_small_model_contract():
    inventory = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_model_inventory_attestation_v1.json"
    )

    v4.validate_model_inventory(inventory)
    assert inventory["repository"] == "openai/gpt-oss-20b"
    assert inventory["revision"] == "6cee5e81ee83917806bbde320786a8fb61efebee"
    assert inventory["loaded_shards"] == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]


def test_import_open_audit_fixture_freezes_denied_surfaces():
    audit = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_import_open_audit_v1.json"
    )

    v4.validate_import_open_audit_fixture(audit)
    assert tuple(audit["denied_imports"]) == tuple(sorted(v4.DENIED_IMPORTS))
    assert tuple(audit["denied_path_fragments"]) == v4.DENIED_PATH_FRAGMENTS


def test_case001_schema_fixture_matches_generated_schema():
    schema = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_response_v1.case001.schema.json"
    )

    assert schema == v4.expected_case001_response_schema()
    require_vllm_xgrammar_compatible(schema)


def test_all_v4_request_fixture_schemas_are_vllm_xgrammar_compatible():
    request_records = v4.load_jsonl_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_request_fixtures_v1.jsonl"
    )

    assert len(request_records) == 24
    for raw_record in request_records:
        record = raw_record
        assert isinstance(record, dict)
        request = record["request"]
        assert isinstance(request, dict)
        response_format = request["response_format"]
        assert isinstance(response_format, dict)
        json_schema = response_format["json_schema"]
        assert isinstance(json_schema, dict)
        schema = json_schema["schema"]
        assert isinstance(schema, dict)
        require_vllm_xgrammar_compatible(schema)


def test_full_fixture_files_cover_all_registry_cases_in_order():
    registry = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_case_registry_v1.json"
    )
    case_ids = v4.validate_case_registry(registry)
    corpus = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_synthetic_corpus_v1.json"
    )
    gold = v4.load_json_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_gold_labels_v1.json"
    )
    requests = v4.load_jsonl_no_duplicates(
        v4.ARTIFACT_DIR / "semantic_anchor_request_fixtures_v1.jsonl"
    )

    assert tuple(case["case_id"] for case in corpus["cases"]) == case_ids
    assert tuple(case["case_id"] for case in gold["cases"]) == case_ids
    assert tuple(record["case_id"] for record in requests) == case_ids
    assert len(corpus["cases"]) == len(gold["cases"]) == len(requests) == 24


def test_manifest_requires_ledger_and_replay_artifacts():
    hashes = v4.validate_artifact_bundle()

    for filename in v4.LEDGER_SCHEMA_FILES:
        assert filename in hashes
    assert "semantic_anchor_replay_mutation_registry_v1.json" in hashes
    assert "semantic_anchor_reviewer_receipt_v1.schema.json" in hashes
    assert "semantic_anchor_renderer_oracle_v1.json" in hashes
    assert "semantic_anchor_model_inventory_attestation_v1.json" in hashes
    assert "semantic_anchor_import_open_audit_v1.json" in hashes


def test_json_loader_rejects_duplicate_keys(tmp_path: Path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        v4.load_json_no_duplicates(path)
