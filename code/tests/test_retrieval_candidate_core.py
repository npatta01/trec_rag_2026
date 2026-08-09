from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from trec_rag.retrieval_candidate_core import (
    CandidateCore,
    candidate_core_from_dict,
    candidate_core_to_dict,
    derive_candidate_core,
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> bytes:
    body = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    path.write_bytes(body)
    return body


def _row(lane_name: str, docid: str, score: float) -> dict[str, object]:
    return {
        "topic_id": "rag2026-7",
        "lane_name": lane_name,
        "docid": docid,
        "aggregate_score": score,
    }


def _derive(path: Path, docids: set[str]) -> CandidateCore:
    return derive_candidate_core(
        topic_id="rag2026-7",
        lane_scores_path=path,
        expected_lane_names=("original", "facet:s1:text"),
        expected_docids=frozenset(docids),
        best_retrieval_ranks={docid: index + 1 for index, docid in enumerate(sorted(docids))},
    )


def test_positive_mad_threshold_and_union_are_computed_per_source_lane(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lane_scores.jsonl"
    rows: list[dict[str, object]] = []
    for lane, scores in (
        ("original", [0.0, 1.0, 2.0, 3.0, 20.0]),
        ("facet:s1:text", [30.0, 3.0, 2.0, 1.0, 0.0]),
    ):
        rows.extend(_row(lane, f"d{index}", score) for index, score in enumerate(scores))
    body = _write_rows(path, rows)

    core = _derive(path, {f"d{index}" for index in range(5)})

    assert core.candidate_docids == ("d0", "d4")
    assert core.pre_fallback_count == 2
    assert core.fallback_used is False
    assert core.admission_multiplicity_histogram == {"1": 2}
    assert core.lane_scores_sha256 == sha256(body).hexdigest()
    assert [(row.lane_name, row.observed_count, row.admitted_count) for row in core.lanes] == [
        ("original", 5, 1),
        ("facet:s1:text", 5, 1),
    ]
    assert core.lanes[0].median == pytest.approx(2.0)
    assert core.lanes[0].mad == pytest.approx(1.0)
    assert core.lanes[0].threshold == pytest.approx(2.0 + 2.5 * 1.4826)
    assert core.lanes[0].comparison == "greater_than_or_equal"


def test_zero_mad_lane_admits_only_scores_strictly_above_median(tmp_path: Path) -> None:
    path = tmp_path / "lane_scores.jsonl"
    rows = [
        *[_row("original", docid, score) for docid, score in zip("abcd", [0.0, 0.0, 0.0, 5.0])],
        *[_row("facet:s1:text", docid, 1.0) for docid in "abcd"],
    ]
    _write_rows(path, rows)

    core = _derive(path, set("abcd"))

    assert core.candidate_docids == ("d",)
    assert core.lanes[0].comparison == "strictly_greater_than"
    assert core.lanes[0].admitted_count == 1
    assert core.lanes[1].admitted_count == 0


def test_empty_union_falls_back_to_original_argmax_then_rank_then_utf8_docid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lane_scores.jsonl"
    _write_rows(
        path,
        [
            _row("original", "z", 4.0),
            _row("original", "ä", 4.0),
            _row("original", "a", 4.0),
            _row("facet:s1:text", "z", 2.0),
            _row("facet:s1:text", "ä", 2.0),
            _row("facet:s1:text", "a", 2.0),
        ],
    )

    core = derive_candidate_core(
        topic_id="rag2026-7",
        lane_scores_path=path,
        expected_lane_names=("original", "facet:s1:text"),
        expected_docids=frozenset({"z", "ä", "a"}),
        best_retrieval_ranks={"z": 2, "ä": 1, "a": 1},
    )

    assert core.candidate_docids == ("a",)
    assert core.pre_fallback_count == 0
    assert core.fallback_used is True
    assert core.admission_multiplicity_histogram == {}


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), True])
def test_nonfinite_or_boolean_aggregate_score_is_rejected(
    tmp_path: Path, score: object
) -> None:
    path = tmp_path / "lane_scores.jsonl"
    _write_rows(
        path,
        [
            _row("original", "a", score),  # type: ignore[arg-type]
            _row("facet:s1:text", "a", 1.0),
        ],
    )

    with pytest.raises((TypeError, ValueError), match="aggregate_score"):
        _derive(path, {"a"})


def test_duplicate_lane_document_pair_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lane_scores.jsonl"
    _write_rows(
        path,
        [
            _row("original", "a", 1.0),
            _row("original", "a", 2.0),
            _row("facet:s1:text", "a", 1.0),
        ],
    )

    with pytest.raises(ValueError, match="duplicate lane/document"):
        _derive(path, {"a"})


def test_duplicate_json_object_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lane_scores.jsonl"
    path.write_text(
        '{"topic_id":"rag2026-7","lane_name":"original","docid":"a",'
        '"aggregate_score":1,"aggregate_score":2}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _derive(path, {"a"})


def test_exact_authenticated_lane_set_is_required(tmp_path: Path) -> None:
    path = tmp_path / "lane_scores.jsonl"
    _write_rows(path, [_row("original", "a", 1.0), _row("facet:wrong:text", "a", 1.0)])

    with pytest.raises(ValueError, match="exact authenticated lane set"):
        _derive(path, {"a"})


def test_unknown_document_and_wrong_topic_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lane_scores.jsonl"
    rows = [_row("original", "outside", 1.0), _row("facet:s1:text", "a", 1.0)]
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="outside the expected union"):
        _derive(path, {"a"})

    rows[0]["docid"] = "a"
    rows[0]["topic_id"] = "rag2026-8"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="topic identity"):
        _derive(path, {"a"})


def test_rank_map_must_exactly_cover_expected_documents(tmp_path: Path) -> None:
    path = tmp_path / "lane_scores.jsonl"
    _write_rows(path, [_row("original", "a", 1.0), _row("facet:s1:text", "a", 1.0)])

    with pytest.raises(ValueError, match="retrieval-rank map"):
        derive_candidate_core(
            topic_id="rag2026-7",
            lane_scores_path=path,
            expected_lane_names=("original", "facet:s1:text"),
            expected_docids=frozenset({"a"}),
            best_retrieval_ranks={},
        )


def test_candidate_core_serialization_is_canonical_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "lane_scores.jsonl"
    _write_rows(
        path,
        [
            _row("original", "a", 0.0),
            _row("original", "b", 5.0),
            _row("facet:s1:text", "a", 1.0),
            _row("facet:s1:text", "b", 1.0),
        ],
    )
    core = _derive(path, {"a", "b"})

    encoded = candidate_core_to_dict(core)

    assert candidate_core_from_dict(encoded) == core
    assert list(encoded) == [
        "schema_version",
        "topic_id",
        "lane_scores_sha256",
        "candidate_docids",
        "pre_fallback_count",
        "fallback_used",
        "admission_multiplicity_histogram",
        "lanes",
    ]
    with pytest.raises(ValueError, match="schema"):
        candidate_core_from_dict({**encoded, "schema_version": "wrong"})
    with pytest.raises(ValueError, match="candidate docids"):
        candidate_core_from_dict({**encoded, "candidate_docids": ["b", "a"]})
    with pytest.raises(ValueError, match="lane-score SHA-256"):
        candidate_core_from_dict({**encoded, "lane_scores_sha256": "bad"})


def test_candidate_core_records_cross_lane_admission_multiplicity(tmp_path: Path) -> None:
    path = tmp_path / "lane_scores.jsonl"
    rows: list[dict[str, object]] = []
    for lane in ("original", "facet:s1:text"):
        rows.extend(
            _row(lane, docid, score)
            for docid, score in zip(("a", "b", "c", "d"), (0.0, 0.0, 0.0, 9.0))
        )
    _write_rows(path, rows)

    core = _derive(path, {"a", "b", "c", "d"})

    assert core.candidate_docids == ("d",)
    assert core.admission_multiplicity_histogram == {"2": 1}
    assert core.lanes[0].admitted_docids_sha256 == core.lanes[1].admitted_docids_sha256


def test_candidate_core_dataclass_rejects_tampered_fallback_state(tmp_path: Path) -> None:
    path = tmp_path / "lane_scores.jsonl"
    _write_rows(path, [_row("original", "a", 1.0), _row("facet:s1:text", "a", 1.0)])
    core = _derive(path, {"a"})

    with pytest.raises(ValueError, match="fallback"):
        replace(core, fallback_used=False)
