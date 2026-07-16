from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag.tethered_facet_soft_coverage import (
    _orders_from_rows,
    _replace_facet_scores,
    build_soft_permutations,
    freeze_soft_rankings,
    load_topic_rows,
    verify_soft_freeze,
)


def _topic_input(*, reversed_rows: bool = False) -> dict[str, object]:
    docids = [f"d{index:03d}" for index in range(120)]
    rows = list(reversed(docids)) if reversed_rows else docids
    return {
        "topic_id": "219",
        "docids": rows,
        "texts": {document_id: f"document {document_id}" for document_id in rows},
        "original_rank": {
            document_id: index + 1 for index, document_id in enumerate(docids)
        },
        "facets": [
            {
                "facet_id": "positive",
                "manifest_order": 0,
                "scores": {
                    document_id: float(120 - int(document_id[1:]))
                    for document_id in rows
                },
                "bm25_rank": {
                    document_id: index + 1
                    for index, document_id in enumerate(docids)
                },
            },
            {
                "facet_id": "negative",
                "manifest_order": 1,
                "scores": {
                    document_id: float(int(document_id[1:]))
                    for document_id in rows
                },
                "bm25_rank": {
                    document_id: 120 - index
                    for index, document_id in enumerate(docids)
                },
            },
        ],
        "common_scores": {
            document_id: float(120 - index)
            for index, document_id in enumerate(docids)
        },
        "narrative_scores": {
            document_id: float(index) for index, document_id in enumerate(docids)
        },
    }


def _controls() -> dict[str, list[str]]:
    docids = [f"d{index:03d}" for index in range(120)]
    return {
        "RRF": docids,
        "NARRATIVE": list(reversed(docids)),
        "FIXED-O0": docids[::2] + docids[1::2],
    }


def test_soft_rankings_are_complete_and_protected_head_is_not_a_cutoff() -> None:
    rankings, _audit = build_soft_permutations(_topic_input(), _controls())
    expected = set(_topic_input()["docids"])
    for arm in ("TETHERED-DUAL", "TETHERED-DUAL-NR", "RRF100-TETHERED-DUAL"):
        assert set(rankings[arm]) == expected
        assert len(rankings[arm]) == len(expected)
    assert rankings["RRF100-TETHERED-DUAL"][:100] == _controls()["RRF"][:100]


def test_tethered_scores_replace_only_facet_local_features() -> None:
    _rankings, audit = build_soft_permutations(_topic_input(), _controls())
    assert audit["parameters"]["dual"] == {
        "G": 0.35,
        "N": 0.15,
        "R": 0.15,
        "L": 0.25,
        "B": 0.10,
        "D": -0.15,
    }
    assert audit["parameters"]["facet_score_source"] == "narrative_tethered"


def test_protected_topic_rejects_before_reader_runs() -> None:
    opened = False

    def reader(_path: Path) -> list[dict[str, object]]:
        nonlocal opened
        opened = True
        return []

    with pytest.raises(ValueError, match="protected topic 144"):
        load_topic_rows(["144"], reader=reader)
    assert opened is False


def test_soft_output_is_deterministic_under_input_reordering() -> None:
    forward, _ = build_soft_permutations(_topic_input(), _controls())
    reverse, _ = build_soft_permutations(
        _topic_input(reversed_rows=True), _controls()
    )
    assert reverse == forward


def test_freeze_is_create_only_complete_and_sealed(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"topic_id":"219"}\n', encoding="utf-8")
    topic_inputs = {
        topic_id: {**_topic_input(), "topic_id": topic_id}
        for topic_id in ("219", "72", "300", "84")
    }
    controls = {topic_id: _controls() for topic_id in topic_inputs}
    output = tmp_path / "freeze"

    summary = freeze_soft_rankings(
        topic_inputs=topic_inputs,
        controls=controls,
        input_paths={"fixture": source},
        output=output,
    )

    assert summary["ranking_row_count"] == 4 * 6 * 120
    assert summary["protected_topic_count"] == 0
    assert all(summary[counter] == 0 for counter in (
        "network_call_count",
        "retrieval_call_count",
        "model_load_count",
        "inference_count",
        "hosted_inference_call_count",
        "paid_call_count",
    ))
    assert set(path.name for path in output.iterdir()) == {
        "parameters.json",
        "input_bindings.json",
        "rankings.jsonl",
        "summary.json",
        "SEALED.json",
    }
    assert verify_soft_freeze(output)["ranking_row_count"] == 4 * 6 * 120
    binding = json.loads((output / "input_bindings.json").read_text())
    assert binding["inputs"]["fixture"]["path"] == "source.jsonl"
    assert binding["inputs"]["fixture"]["rows"] == 1
    with pytest.raises(FileExistsError, match="create-only"):
        freeze_soft_rankings(
            topic_inputs=topic_inputs,
            controls=controls,
            input_paths={"fixture": source},
            output=output,
        )


def test_verify_rejects_mutated_ranking(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    topic_inputs = {
        topic_id: {**_topic_input(), "topic_id": topic_id}
        for topic_id in ("219", "72", "300", "84")
    }
    output = tmp_path / "freeze"
    freeze_soft_rankings(
        topic_inputs=topic_inputs,
        controls={topic_id: _controls() for topic_id in topic_inputs},
        input_paths={"fixture": source},
        output=output,
    )
    with (output / "rankings.jsonl").open("ab") as sink:
        sink.write(b"{}\n")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_soft_freeze(output)


def test_adapter_replaces_only_facet_score_maps() -> None:
    topic = _topic_input()
    replacement_rows = [
        {
            "topic_id": "219",
            "facet_id": facet["facet_id"],
            "document_id": document_id,
            "score": -float(int(document_id[1:])),
        }
        for facet in topic["facets"]
        for document_id in facet["scores"]
    ]

    replaced = _replace_facet_scores({"219": topic}, replacement_rows)["219"]

    assert replaced["common_scores"] == topic["common_scores"]
    assert replaced["narrative_scores"] == topic["narrative_scores"]
    assert replaced["texts"] == topic["texts"]
    assert replaced["original_rank"] == topic["original_rank"]
    for before, after in zip(topic["facets"], replaced["facets"], strict=True):
        assert after["bm25_rank"] == before["bm25_rank"]
        assert after["manifest_order"] == before["manifest_order"]
        assert after["scores"] != before["scores"]


def test_single_arm_control_file_uses_authenticated_filename_identity() -> None:
    rows = [
        {"topic_id": topic_id, "topic_rank": 1, "document_id": f"{topic_id}-d"}
        for topic_id in ("219", "72", "300", "84")
    ]
    orders = _orders_from_rows(
        rows, ["NARRATIVE"], rank_field="topic_rank", source_arm="NARRATIVE"
    )
    assert orders["219"]["NARRATIVE"] == ["219-d"]
