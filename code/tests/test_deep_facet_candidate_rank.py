from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag.deep_facet_candidate_rank import (
    _stream_kind,
    build_permutations,
    create_seal,
    rank_percentile,
    redundancy_penalty,
    verify_seal,
)


def _inputs() -> dict[str, object]:
    return {
        "topic_id": "219",
        "docids": ["a", "b", "c", "d"],
        "texts": {
            "a": "technology benefits society",
            "b": "technology harms society",
            "c": "technology government policy",
            "d": "unrelated text",
        },
        "original_rank": {"a": 1, "d": 2},
        "facets": [
            {
                "facet_id": "positive",
                "manifest_order": 0,
                "scores": {"a": 4.0, "b": 2.0, "c": 1.0},
                "bm25_rank": {"a": 1, "b": 2, "c": 3},
            },
            {
                "facet_id": "government",
                "manifest_order": 1,
                "scores": {"c": 5.0, "a": 1.0, "b": 0.0},
                "bm25_rank": {"c": 1, "a": 2, "b": 3},
            },
        ],
        "common_scores": {"a": 3.0, "b": 1.0, "c": 2.0, "d": 0.0},
        "narrative_scores": {"a": 4.0, "b": 2.0, "c": 3.0, "d": 1.0},
    }


def test_percentile_ties() -> None:
    assert rank_percentile({"a": 9, "b": 7, "c": 7}) == {
        "a": 1.0,
        "b": 0.5,
        "c": 0.5,
    }


def test_redundancy_threshold() -> None:
    assert redundancy_penalty(0.79) == 0.0
    assert redundancy_penalty(0.80) == 0.0
    assert redundancy_penalty(0.90) == pytest.approx(0.5)
    assert redundancy_penalty(1.0) == 1.0


def test_permutations_are_complete_and_order_invariant() -> None:
    inputs = _inputs()
    reversed_inputs = dict(inputs)
    reversed_inputs["docids"] = list(reversed(inputs["docids"]))
    reversed_inputs["texts"] = dict(reversed(list(inputs["texts"].items())))
    reversed_inputs["common_scores"] = dict(
        reversed(list(inputs["common_scores"].items()))
    )
    first = build_permutations(inputs)
    second = build_permutations(reversed_inputs)
    assert first == second
    assert set(first) == {"RRF", "GLOBAL", "FACET", "DUAL", "DUAL-NR"}
    assert all(set(rows) == set(inputs["docids"]) for rows in first.values())
    assert all(len(rows) == len(inputs["docids"]) for rows in first.values())


def test_zero_facet_fallback_is_original_complete_permutation() -> None:
    inputs = _inputs()
    inputs["facets"] = []
    inputs["original_rank"] = {"b": 1, "a": 2, "d": 3, "c": 4}
    assert all(
        rows == ["b", "a", "d", "c"]
        for rows in build_permutations(inputs).values()
    )


def test_legacy_flattened_stream_rows_recover_minilm_identity() -> None:
    # gate_v1 predates the explicit stream_family field: its candidate's
    # ``family=facet`` overwrote the outer bm25/minilm label.
    assert _stream_kind({"family": "facet", "retrieval_score": 4.0}) == "bm25"
    assert _stream_kind(
        {"family": "facet", "score": 2.0, "retrieval_rank": 17}
    ) == "minilm"
    assert _stream_kind({"family": "facet", "stream_family": "minilm"}) == "minilm"


def test_seal_rejects_mutation_and_extra_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    freeze = tmp_path / "freeze"
    source.mkdir()
    freeze.mkdir()
    (source / "a.json").write_text('{"a":1}\n', encoding="utf-8")
    (freeze / "rankings.jsonl").write_text('{"rank":1}\n', encoding="utf-8")
    seal = create_seal(
        manifest_path=source / "a.json",
        source_dirs=[source],
        freeze_dir=freeze,
        topic_ids=["219", "72", "300", "84"],
    )
    assert seal["qrels_opened"] is False
    assert verify_seal(freeze)["root_sha256"] == seal["root_sha256"]

    (source / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        verify_seal(freeze)


def test_seal_is_create_only(tmp_path: Path) -> None:
    source = tmp_path / "manifest.json"
    freeze = tmp_path / "freeze"
    source.write_text("{}\n", encoding="utf-8")
    freeze.mkdir()
    (freeze / "rankings.jsonl").write_text("{}\n", encoding="utf-8")
    create_seal(
        manifest_path=source,
        source_dirs=[],
        freeze_dir=freeze,
        topic_ids=["219", "72", "300", "84"],
    )
    with pytest.raises(FileExistsError):
        create_seal(
            manifest_path=source,
            source_dirs=[],
            freeze_dir=freeze,
            topic_ids=["219", "72", "300", "84"],
        )
