from __future__ import annotations

import json
from pathlib import Path

import pytest

import trec_rag.all_topic_tethered_rank as module
from trec_rag.all_topic_tethered_rank import ARMS, build_rankings


def _topic_input() -> dict[str, object]:
    docids = [f"d{i:03d}" for i in range(1, 601)]
    scores = {docid: float(601 - index) for index, docid in enumerate(docids, 1)}
    return {
        "topic_id": "14",
        "docids": docids,
        "texts": {docid: f"document {index} token {index % 17}" for index, docid in enumerate(docids)},
        "original_rank": {docid: index for index, docid in enumerate(docids, 1)},
        "facets": [
            {
                "facet_id": "14-a",
                "manifest_order": 0,
                "scores": scores,
                "bm25_rank": {docid: index for index, docid in enumerate(reversed(docids), 1)},
            },
            {
                "facet_id": "14-b",
                "manifest_order": 1,
                "scores": {docid: float(index) for index, docid in enumerate(docids, 1)},
                "bm25_rank": {docid: index for index, docid in enumerate(docids, 1)},
            },
        ],
        "common_scores": scores,
        "narrative_scores": {docid: float(index % 31) for index, docid in enumerate(docids, 1)},
    }


def _controls(topic: dict[str, object] | None = None) -> dict[str, object]:
    topic = topic or _topic_input()
    return {"RRF": module._rrf_control(topic)}


def _expected_seed_coverage() -> dict[str, float]:
    features = module._dual_features(_topic_input())
    prefix = _controls()["RRF"][:100]
    facets = features["facets"]
    return {
        facet["facet_id"]: max(features["F"][index].get(docid, 0.0) for docid in prefix)
        for index, facet in enumerate(facets)
    }


def test_every_arm_is_the_same_complete_union() -> None:
    topic = _topic_input()
    rankings, _ = build_rankings(topic, _controls(topic))
    expected = set(topic["docids"])
    assert set(rankings) == set(ARMS)
    assert all(len(order) == len(expected) and set(order) == expected for order in rankings.values())


def test_protected_prefixes_are_exact() -> None:
    rankings, _ = build_rankings(_topic_input(), _controls())
    assert rankings["RRF100-STATIC-DUAL"][:100] == rankings["RRF"][:100]
    assert rankings["RRF500-REINIT-DUAL"][:500] == rankings["RRF"][:500]


def test_reinitialized_state_includes_prefix_coverage() -> None:
    _, audit = build_rankings(_topic_input(), _controls())
    assert audit["RRF100-REINIT-DUAL"]["seed_document_count"] == 100
    assert audit["RRF100-REINIT-DUAL"]["seed_coverage"] == _expected_seed_coverage()


def test_input_reordering_cannot_change_rankings() -> None:
    topic = _topic_input()
    controls = _controls(topic)
    expected, _ = build_rankings(topic, controls)
    reordered = dict(topic)
    reordered["docids"] = list(reversed(topic["docids"]))
    reordered["texts"] = dict(reversed(list(topic["texts"].items())))
    reordered["common_scores"] = dict(reversed(list(topic["common_scores"].items())))
    actual, _ = build_rankings(reordered, controls)
    assert actual == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"qrels": "/secret/qrels.txt"},
        {"nested": [{"qrels_path": "/secret/qrels.txt"}]},
        {"nested": {"QRELS_FILE": "anything"}},
    ],
)
def test_qrels_fields_are_rejected_recursively(payload: dict[str, object]) -> None:
    controls = {**_controls(), **payload}
    with pytest.raises(ValueError, match="qrels"):
        build_rankings(_topic_input(), controls)


def test_verify_rejects_mutated_ranking_bytes(tmp_path: Path) -> None:
    output = tmp_path / "rankings"
    module._freeze_loaded_topics(
        {"14": _topic_input()},
        {"14": _controls()},
        output,
        input_bindings={"test": True},
        expected_topic_ids=("14",),
    )
    assert module.verify_rankings(output)["verified"] is True
    with (output / "rankings.jsonl").open("ab") as sink:
        sink.write(b"{}\n")
    with pytest.raises(ValueError, match="seal|mutated|hash"):
        module.verify_rankings(output)


def test_freeze_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "rankings"
    kwargs = dict(
        topic_inputs={"14": _topic_input()},
        controls={"14": _controls()},
        output=output,
        input_bindings={"test": True},
        expected_topic_ids=("14",),
    )
    module._freeze_loaded_topics(**kwargs)
    with pytest.raises(FileExistsError, match="create-only"):
        module._freeze_loaded_topics(**kwargs)


def test_cli_has_no_qrels_network_or_model_options() -> None:
    parser = module._parser()
    choices = next(action for action in parser._actions if action.dest == "command").choices
    option_names = {
        option
        for child in choices.values()
        for action in child._actions
        for option in action.option_strings
    }
    assert not {"--qrels", "--network", "--model"} & option_names
