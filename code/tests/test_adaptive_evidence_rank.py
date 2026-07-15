from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

import trec_rag.adaptive_evidence_rank as rank_module
from trec_rag.adaptive_evidence_rank import (
    best_window_per_document,
    build_fixed_o0_continuation,
    build_narrative_continuation,
    build_rankings,
    load_authenticated_ranking_data,
    verify_baseline_rankings,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _document(topic_id: str, document_id: str, union_order: int) -> dict[str, object]:
    text = f"full text for {topic_id}/{document_id}"
    return {
        "topic_id": topic_id,
        "document_id": document_id,
        "union_order": union_order,
        "text": text,
        "text_sha256": _sha(text),
    }


def _score(
    topic_id: str,
    obligation_id: str,
    document_id: str,
    score: float,
    *,
    window_id: str | None = None,
    start: int = 0,
    passage_tokens: int = 4,
) -> dict[str, object]:
    window_id = window_id or f"{document_id}-{obligation_id}-window"
    window_text = f"exact passage {window_id}"
    query = f"query for {obligation_id}"
    return {
        "topic_id": topic_id,
        "variant": obligation_id,
        "document_id": document_id,
        "document_sha256": _sha(f"full text for {topic_id}/{document_id}"),
        "window_id": window_id,
        "window_text": window_text,
        "window_sha256": _sha(window_text),
        "query_sha256": _sha(query),
        "preflight_sha256": "a" * 64,
        "preflight_windows_sha256": "b" * 64,
        "document_start_token": start,
        "document_end_token": start + passage_tokens,
        "score": score,
    }


def _fixture() -> dict[str, object]:
    documents = [_document("219", document_id, order) for order, document_id in enumerate(("d1", "d2", "d3"), 1)]
    obligations = [
        {
            "topic_id": "219",
            "obligation_id": "219:broad",
            "kind": "broad",
            "manifest_order": -1,
        },
        {
            "topic_id": "219",
            "obligation_id": "219-positive",
            "kind": "o0",
            "manifest_order": 0,
        },
        {
            "topic_id": "219",
            "obligation_id": "219-negative",
            "kind": "o0",
            "manifest_order": 1,
        },
    ]
    scores = [
        _score("219", "219:broad", "d1", 2.0, window_id="d1-weaker-broad", start=0),
        _score("219", "219:broad", "d1", 2.5, window_id="d1-best-broad", start=4),
        _score("219", "219:broad", "d2", 3.0, window_id="d2-best-broad", passage_tokens=9),
        _score("219", "219:broad", "d3", 1.0, window_id="d3-best-broad"),
        _score("219", "219-positive", "d1", 1000.0, passage_tokens=2),
        _score("219", "219-positive", "d2", 999.0),
        _score("219", "219-negative", "d2", -1000.0),
        _score("219", "219-negative", "d3", -1001.0, passage_tokens=3),
    ]
    return {
        "topic_ids": ["219"],
        "obligations": obligations,
        "documents": documents,
        "score_rows": scores,
        "source_bindings": {
            "mode": "provided_data",
            "contract_summary_sha256": "c" * 64,
            "base_score_receipt_sha256": "d" * 64,
            "terminal_discovery_receipt_sha256": "e" * 64,
        },
        "discovery_receipt": {
            "status": "discovery_unavailable",
            "proposed_o1_count": 0,
            "accepted_o1_count": 0,
            "proposed_n1_count": 0,
            "accepted_n1_count": 0,
        },
    }


def _extended_fixture() -> dict[str, object]:
    data = _fixture()
    data["documents"] = [
        _document("219", document_id, order)
        for order, document_id in enumerate(("d1", "d2", "d3", "d4", "d5", "d6"), 1)
    ]
    data["score_rows"] = [
        _score("219", "219:broad", document_id, 10.0 - order, passage_tokens=9)
        for order, document_id in enumerate(("d1", "d2", "d3", "d4", "d5", "d6"), 1)
    ] + [
        _score("219", "219-positive", "d1", 2.0, passage_tokens=2),
        _score("219", "219-positive", "d4", 1.0, passage_tokens=2),
        _score("219", "219-negative", "d3", 20000.0, passage_tokens=3),
        _score("219", "219-negative", "d5", -20000.0, passage_tokens=3),
    ]
    # Make d2 the broad coverage-floor winner while leaving query-local ordering intact.
    data["score_rows"][0]["score"] = 8.0  # type: ignore[index]
    data["score_rows"][1]["score"] = 9.0  # type: ignore[index]
    return data


def _all_document_ids(data: dict[str, object] | None = None) -> set[str]:
    value = data or _fixture()
    return {str(row["document_id"]) for row in value["documents"]}  # type: ignore[index]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_narrative_uses_each_documents_best_broad_window() -> None:
    rows = build_narrative_continuation(_fixture())

    assert [row["document_id"] for row in rows] == ["d2", "d1", "d3"]
    assert rows[0]["window_id"] == "d2-best-broad"
    assert all(row["primary_obligation"] == "219:broad" for row in rows)
    assert rows[0]["window_text"] == "exact passage d2-best-broad"


def test_fixed_o0_starts_with_broad_then_each_o0_in_manifest_order() -> None:
    rows = build_fixed_o0_continuation(_fixture())

    assert [row["primary_obligation"] for row in rows[:3]] == [
        "219:broad",
        "219-positive",
        "219-negative",
    ]
    assert [row["document_id"] for row in rows] == ["d2", "d1", "d3"]


def test_fixed_o0_uses_query_local_scores_and_never_compares_raw_scores() -> None:
    rows = build_fixed_o0_continuation(_fixture())

    assert rows[1]["selection_reason"] == "coverage_floor"
    assert rows[1]["score_scope"] == "219-positive"
    assert rows[2]["score_scope"] == "219-negative"
    assert all(row["score_scope"] == row["primary_obligation"] for row in rows[:3])


def test_both_complete_continuations_preserve_every_document_once() -> None:
    narrative = build_narrative_continuation(_fixture())
    fixed = build_fixed_o0_continuation(_fixture())

    assert [row["document_id"] for row in narrative].count("d1") == 1
    assert {row["document_id"] for row in narrative} == _all_document_ids()
    assert {row["document_id"] for row in fixed} == _all_document_ids()
    assert len({row["document_id"] for row in fixed}) == len(fixed)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_best_window_rejects_nonfinite_scores(value: float) -> None:
    rows = [_score("219", "219:broad", "d1", value)]

    with pytest.raises(ValueError, match="nonfinite"):
        best_window_per_document(rows)


def test_best_window_ties_are_deterministic_under_input_reordering() -> None:
    later = _score(
        "219", "219:broad", "d1", 1.0, window_id="a-later", start=4
    )
    lexically_later = _score(
        "219", "219:broad", "d1", 1.0, window_id="z-earliest", start=0
    )
    winner = _score(
        "219", "219:broad", "d1", 1.0, window_id="a-earliest", start=0
    )

    assert best_window_per_document([later, lexically_later, winner]) == [winner]
    assert best_window_per_document([winner, later, lexically_later]) == [winner]


def test_narrative_document_ties_use_union_order_then_document_id() -> None:
    data = _fixture()
    for row in data["score_rows"]:  # type: ignore[index]
        if row["variant"] == "219:broad":
            row["score"] = 1.0
    reordered = {**data, "score_rows": list(reversed(data["score_rows"]))}  # type: ignore[arg-type]

    expected = ["d1", "d2", "d3"]
    assert [row["document_id"] for row in build_narrative_continuation(data)] == expected
    assert [row["document_id"] for row in build_narrative_continuation(reordered)] == expected


def test_fixed_queue_schedule_is_invariant_to_cross_query_score_scales() -> None:
    data = _extended_fixture()
    shifted = {**data, "score_rows": [dict(row) for row in data["score_rows"]]}  # type: ignore[index]
    offsets = {"219:broad": -1_000_000.0, "219-positive": 9_000_000.0, "219-negative": -8_000_000.0}
    for row in shifted["score_rows"]:  # type: ignore[index]
        row["score"] = float(row["score"]) + offsets[str(row["variant"])]

    baseline = build_fixed_o0_continuation(data)
    changed = build_fixed_o0_continuation(shifted)
    assert [(row["document_id"], row["primary_obligation"]) for row in baseline] == [
        (row["document_id"], row["primary_obligation"]) for row in changed
    ]


def test_fixed_exhausts_facet_queues_then_uses_broad_tail() -> None:
    rows = build_fixed_o0_continuation(_extended_fixture())

    assert [row["document_id"] for row in rows] == ["d2", "d1", "d3", "d4", "d5", "d6"]
    assert [row["selection_reason"] for row in rows] == [
        "coverage_floor",
        "coverage_floor",
        "coverage_floor",
        "token_deficit_round_robin",
        "token_deficit_round_robin",
        "broad_tail",
    ]
    assert rows[-1]["primary_obligation"] == "219:broad"


def test_output_rows_bind_scored_queues_without_claiming_semantic_support() -> None:
    row = build_fixed_o0_continuation(_fixture())[0]

    assert row["scored_obligations"] == [
        "219:broad",
        "219-positive",
        "219-negative",
    ]
    assert "support" not in row
    for name in (
        "document_sha256",
        "window_sha256",
        "query_sha256",
        "preflight_sha256",
        "preflight_windows_sha256",
    ):
        assert len(str(row[name])) == 64


def test_topic_ranks_reset_and_cover_each_topic_exactly() -> None:
    first = _fixture()
    second = _fixture()
    for collection in (second["obligations"], second["documents"], second["score_rows"]):  # type: ignore[assignment]
        for row in collection:  # type: ignore[union-attr]
            row["topic_id"] = "72"
            if "obligation_id" in row:
                row["obligation_id"] = str(row["obligation_id"]).replace("219", "72", 1)
            if "variant" in row:
                row["variant"] = str(row["variant"]).replace("219", "72", 1)
            if "document_sha256" in row:
                row["document_sha256"] = _sha(f"full text for 72/{row['document_id']}")
        for row in second["documents"]:  # type: ignore[index]
            row["text"] = f"full text for 72/{row['document_id']}"
            row["text_sha256"] = _sha(str(row["text"]))
    combined = {
        **first,
        "topic_ids": ["219", "72"],
        "obligations": [*first["obligations"], *second["obligations"]],  # type: ignore[misc]
        "documents": [*first["documents"], *second["documents"]],  # type: ignore[misc]
        "score_rows": [*first["score_rows"], *second["score_rows"]],  # type: ignore[misc]
    }

    rows = build_narrative_continuation(combined)
    assert [(row["topic_id"], row["topic_rank"]) for row in rows] == [
        ("219", 1), ("219", 2), ("219", 3), ("72", 1), ("72", 2), ("72", 3)
    ]


def test_adaptive_is_unavailable_not_an_empty_ranking(tmp_path: Path) -> None:
    output = tmp_path / "rankings"
    build_rankings(_fixture(), output_dir=output)
    availability = json.loads((output / "availability.json").read_text())

    assert availability["arms"]["NARRATIVE"]["status"] == "available"
    assert availability["arms"]["FIXED-O0"]["status"] == "available"
    assert availability["arms"]["ADAPTIVE"]["status"] == "unavailable"
    assert availability["arms"]["COMPOSITE"]["status"] == "unavailable"
    assert availability["arms"]["ADAPTIVE"]["accepted_o1_count"] == 0
    assert availability["arms"]["ADAPTIVE"]["accepted_n1_count"] == 0
    assert not (output / "adaptive.jsonl").exists()
    assert not (output / "composite.jsonl").exists()


def test_build_writes_complete_rankings_and_zero_safety_receipt(tmp_path: Path) -> None:
    output = tmp_path / "rankings"
    receipt = build_rankings(_fixture(), output_dir=output)

    assert len(_read_jsonl(output / "narrative.jsonl")) == 3
    assert len(_read_jsonl(output / "fixed_o0.jsonl")) == 3
    assert receipt["rankings"]["NARRATIVE"]["rows"] == 3
    assert receipt["rankings"]["FIXED-O0"]["rows"] == 3
    assert receipt["topic_ids"] == ["219"]
    assert receipt["protected_topic_count"] == 0
    assert receipt["qrels_opened"] is False
    for name in (
        "network_call_count",
        "retrieval_call_count",
        "hosted_inference_call_count",
        "paid_call_count",
        "model_load_count",
        "inference_count",
    ):
        assert receipt[name] == 0


def test_verifier_detects_ranking_and_receipt_hash_tampering(tmp_path: Path) -> None:
    data = _fixture()
    output = tmp_path / "rankings"
    build_rankings(data, output_dir=output)
    narrative = output / "narrative.jsonl"
    source = narrative.read_bytes()
    narrative.write_bytes(source + b"\n")
    with pytest.raises(ValueError, match="hash"):
        verify_baseline_rankings(output, data=data)

    narrative.write_bytes(source)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["rankings"]["NARRATIVE"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="hash"):
        verify_baseline_rankings(output, data=data)


@pytest.mark.parametrize("mode", ["unexpected", "partial", "symlink"])
def test_verifier_rejects_unexpected_partial_and_symlink_artifacts(
    tmp_path: Path, mode: str
) -> None:
    output = tmp_path / "rankings"
    build_rankings(_fixture(), output_dir=output)
    if mode == "unexpected":
        (output / "unexpected.json").write_text("{}\n", encoding="utf-8")
    elif mode == "partial":
        (output / "fixed_o0.jsonl").unlink()
    else:
        target = tmp_path / "outside.jsonl"
        target.write_text("{}\n", encoding="utf-8")
        (output / "fixed_o0.jsonl").unlink()
        (output / "fixed_o0.jsonl").symlink_to(target)

    with pytest.raises(ValueError, match="inventory"):
        verify_baseline_rankings(output, data=_fixture())


def test_build_is_create_only_even_for_partial_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "rankings"
    output.mkdir()
    (output / "narrative.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(FileExistsError, match="create-only"):
        build_rankings(object(), output_dir=output)


def test_protected_contract_fails_before_scores_or_discovery_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched: list[str] = []

    def protected(_path: Path) -> object:
        touched.append("contract")
        raise ValueError("protected topic 144 is forbidden")

    monkeypatch.setattr(rank_module, "load_score_contract", protected)
    monkeypatch.setattr(
        rank_module,
        "verify_local_scoring",
        lambda path: touched.append("scores"),
    )
    monkeypatch.setattr(
        rank_module,
        "verify_discovery_terminal",
        lambda path: touched.append("discovery"),
    )

    with pytest.raises(ValueError, match="protected topic 144"):
        load_authenticated_ranking_data(
            contract_dir=tmp_path / "contract",
            scores_dir=tmp_path / "scores",
            discovery_dir=tmp_path / "discovery",
        )

    assert touched == ["contract"]
