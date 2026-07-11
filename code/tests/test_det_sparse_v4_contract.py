from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag import det_sparse_v4_contract as v4


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


def test_model_inventory_requires_three_loaded_shards_and_denied_original_weight():
    inventory = {
        "repository": "openai/gpt-oss-20b",
        "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
        "quantization_method": "mxfp4",
        "safetensors_index_total_size": 13_761_264_768,
        "snapshot_path": "/models/openai/gpt-oss-20b/snapshots/rev",
        "loaded_shards": ["model-00001.safetensors", "model-00002.safetensors", "model-00003.safetensors"],
        "loaded_files": ["config.json", "tokenizer.json"],
        "unloaded_files": ["README.md"],
        "denied_files": ["original/model.safetensors"],
        "file_sha256": {"config.json": "a" * 64},
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
