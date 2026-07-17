from __future__ import annotations

import json
import shutil
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
    features = module._features_for_authorized_topic(_topic_input())
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


@pytest.mark.parametrize(
    "value",
    [
        "/secret/qrels.txt",
        Path("/secret/qrels.txt"),
        {"source_path": "/secret/TREC_QRELS.tsv"},
    ],
)
def test_qrels_values_are_rejected_recursively(value: object) -> None:
    with pytest.raises(ValueError, match="qrels"):
        module._reject_qrels(value)


def test_cli_rejects_qrels_like_production_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="qrels"):
        module.main(["verify", "--rankings", str(tmp_path / "qrels-freeze")])


def test_verify_rejects_one_topic_resealed_counterfeit(tmp_path: Path) -> None:
    output = tmp_path / "rankings"
    module._freeze_loaded_topics(
        {"14": _topic_input()},
        {"14": _controls()},
        output,
        input_bindings={"test": True},
        expected_topic_ids=("14",),
    )
    with pytest.raises(ValueError, match="canonical|topic|scope|root|contract"):
        module.verify_rankings(output)


def test_feature_adapter_uses_authorized_identity_without_legacy_spoof(monkeypatch: pytest.MonkeyPatch) -> None:
    from trec_rag.deep_facet_candidate_rank import _features

    with pytest.raises(ValueError, match="excluded topic 14"):
        _features(_topic_input())
    seen: list[str] = []
    original = module._features_pure

    def observed(value: dict[str, object]):
        seen.append(str(value["topic_id"]))
        return original(value)

    monkeypatch.setattr(module, "_features_pure", observed)
    features = module._features_for_authorized_topic(_topic_input())
    assert features["topic_id"] == "14"
    assert seen == ["14"]


def test_plain_and_reinitialized_dual_share_one_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = module._dual_objective

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_dual_objective", counted)
    build_rankings(_topic_input(), _controls())
    assert calls == 2 * len(_topic_input()["docids"])


def test_production_bindings_reject_resealed_upstream_root_counterfeit() -> None:
    bindings = module._expected_input_bindings(
        Path("outputs/all_topic_tethered_facet_validation_v1/rankings")
    )
    bindings["retrieval_root_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding|retrieval"):
        module._validate_production_bindings(bindings, Path("outputs/all_topic_tethered_facet_validation_v1/rankings"))


def test_parameters_are_derived_from_local_pins() -> None:
    assert module._parameters() == json.loads(
        Path("outputs/all_topic_tethered_facet_validation_v1/rankings/parameters.json").read_text()
    )


@pytest.mark.parametrize("field", ["network_calls", "model_loads", "inference_calls"])
def test_summary_counter_counterfeit_is_rejected(field: str) -> None:
    summary = json.loads(
        Path("outputs/all_topic_tethered_facet_validation_v1/rankings/summary.json").read_text()
    )
    summary[field] = 1
    with pytest.raises(ValueError, match="counter|safety"):
        module._validate_summary_contract(summary)


def test_audit_counterfeit_missing_objective_is_rejected() -> None:
    row = {
        "schema_version": module.SCHEMA_VERSION,
        "topic_id": "14",
        "arm": "DUAL",
        "kind": "selection",
        "document_id": "doc",
        "coverage_bonus": 0.0,
        "coverage_facet": None,
        "redundancy_penalty": 0.0,
    }
    with pytest.raises(ValueError, match="audit|objective"):
        module._validate_selection_audit(row, {"doc"})


def test_resealed_canonical_counterfeits_are_rejected(tmp_path: Path) -> None:
    source = Path("outputs/all_topic_tethered_facet_validation_v1/rankings")
    counterfeit = tmp_path / "rankings"
    shutil.copytree(source, counterfeit)

    def reseal(name: str, content: bytes) -> None:
        (counterfeit / name).write_bytes(content)
        seal = json.loads((counterfeit / "SEALED.json").read_text())
        seal["files"][name] = {"bytes": len(content), "sha256": module._sha256(content)}
        seal["root_sha256"] = module._sha256(module._compact_bytes(seal["files"]))
        (counterfeit / "SEALED.json").write_bytes(module._pretty_bytes(seal))

    attacks = [
        ("topics", lambda source: source.replace(b'"14"', b'"15"', 1)),
        ("upstream roots", lambda source: source.replace(module.RETRIEVAL_ROOT_SHA256.encode(), b"0" * 64, 1)),
        ("parameters", lambda source: source.replace(b'"G": 0.35', b'"G": 0.36', 1)),
        ("rankings", lambda source: source.replace(b'"document_id":"', b'"document_id":"forged-', 1)),
        ("audit", lambda source: source.replace(b'"objective":', b'"objective":"forged","old_objective":', 1)),
        ("counters", lambda source: source.replace(b'"network_calls": 0', b'"network_calls": 1', 1)),
    ]
    targets = {
        "topics": "parameters.json",
        "upstream roots": "input_bindings.json",
        "parameters": "parameters.json",
        "rankings": "rankings.jsonl",
        "audit": "audit.jsonl",
        "counters": "summary.json",
    }
    baseline_seal = (counterfeit / "SEALED.json").read_bytes()
    for label, attack in attacks:
        name = targets[label]
        baseline = (source / name).read_bytes()
        changed = attack(baseline)
        assert changed != baseline, label
        reseal(name, changed)
        with pytest.raises(ValueError, match="canonical|contract|root|binding|parameter|ranking|audit|counter"):
            module.verify_rankings(counterfeit)
        (counterfeit / name).write_bytes(baseline)
        (counterfeit / "SEALED.json").write_bytes(baseline_seal)


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
