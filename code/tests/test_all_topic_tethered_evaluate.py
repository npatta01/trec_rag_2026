from __future__ import annotations

import json
from pathlib import Path

import pytest

import trec_rag.all_topic_tethered_evaluate as module
from trec_rag.all_topic_facet_contract import ALL_TOPIC_IDS
from trec_rag.all_topic_tethered_rank import ARMS
from trec_rag.all_topic_tethered_evaluate import (
    SELECTION_LADDER,
    _write_evaluation,
    _statistics,
    _load_qrels,
    apply_promotion_rules,
    evaluate_all_topics,
    evaluate_frozen,
    holm_adjust,
    paired_bootstrap,
    paired_sign_flip,
    verify_evaluation,
)


def _fixture(one_loss: bool = False):
    rankings, qrels, provenance = {}, {}, {}
    for index, topic in enumerate(ALL_TOPIC_IDS):
        docs = [f"{topic}-a", f"{topic}-b", f"{topic}-c", f"{topic}-d"]
        baseline = docs[:]
        better = [docs[1], docs[0], docs[2], docs[3]]
        if one_loss and topic == "31":
            better = [docs[2], docs[0], docs[1], docs[3]]
        rankings[topic] = {arm: (baseline[:] if arm == "RRF" else better[:]) for arm in ARMS}
        qrels[topic] = {docs[0]: 2, docs[1]: 3, docs[2]: 0}
        provenance[topic] = {
            docs[0]: [{"stream_id": "original", "stream_rank": 1}],
            docs[1]: [{"stream_id": f"{topic}-facet", "stream_rank": 25}],
            docs[2]: [{"stream_id": f"{topic}-facet", "stream_rank": 75}],
            docs[3]: [{"stream_id": f"{topic}-facet", "stream_rank": 175}],
        }
    return rankings, qrels, provenance


def _passing_metrics(one_loss: bool = False) -> dict[str, object]:
    return {
        "loss_topic_ids_at_250": [],
        "loss_topic_ids_at_500": [],
        "loss_topic_ids": ["31"] if one_loss else [],
        "pooled_delta_at_1000": 0.01,
        "macro_delta_at_1000": 0.01,
        "macro_judged_rate_delta_at_1000": 0.0,
        "wins_at_1000": 8,
        "protected_prefix_identical": True,
    }


def _passing_statistics(significant: bool = True) -> dict[str, object]:
    return {
        "bootstrap_ci_95": [0.001 if significant else -0.001, 0.02],
        "holm_adjusted_p": 0.04 if significant else 0.2,
    }


def test_any_topic_loss_blocks_promotion() -> None:
    decision = apply_promotion_rules(_passing_metrics(one_loss=True), _passing_statistics())
    assert decision["promoted"] is False
    assert decision["failed_rules"] == ["zero_losses_at_1000"]


def test_aggregate_gain_cannot_hide_topic_loss() -> None:
    rankings, qrels, provenance = {}, {}, {}
    for topic in ALL_TOPIC_IDS:
        docs = [f"{topic}-d{index}" for index in range(1002)]
        baseline = docs[:]
        candidate = docs[:]
        candidate[999], candidate[1000] = candidate[1000], candidate[999]
        relevant = {docs[1000]: 2}
        if topic == "31":
            relevant = {docs[999]: 2}
        rankings[topic] = {arm: (baseline[:] if arm == "RRF" else candidate[:]) for arm in ARMS}
        qrels[topic] = relevant
        provenance[topic] = {document: [{"stream_id": "original", "stream_rank": rank}] for rank, document in enumerate(docs, 1)}
    result = evaluate_all_topics(rankings, qrels, provenance, depths=(1, 2, 3))
    arm = result["arms"]["RRF100-STATIC-DUAL"]
    assert arm["pooled_delta_at_1000"] > 0
    assert arm["loss_topic_ids"] == ["31"]


def test_metrics_include_required_recall_quality_and_provenance_fields() -> None:
    rankings, qrels, provenance = _fixture()
    result = evaluate_all_topics(rankings, qrels, provenance, depths=(1, 2, 3))
    topic = result["arms"]["RRF100-STATIC-DUAL"]["per_topic"]["14"]
    assert topic["known_relevant_total"] == 2
    assert topic["known_relevant_count@1"] == 1
    assert topic["binary_recall@1"] == pytest.approx(0.5)
    assert topic["graded_recall@1"] == pytest.approx(7 / 10)
    assert topic["precision@1"] == 1.0
    assert topic["judged_rate@1"] == 1.0
    assert topic["facet_only_known_relevant_retained@1"] == 1
    assert topic["full_union_recall_ceiling"] == 1.0
    assert set(result["facet_rank_bucket_yield"]["14"]) == {"1-50", "51-100", "101-150", "151-200"}


def test_paired_inference_is_deterministic_and_exact() -> None:
    deltas = [1.0] * 22
    first = paired_bootstrap(deltas, samples=100_000, seed=20260716)
    assert first == paired_bootstrap(deltas, samples=100_000, seed=20260716)
    assert first["ci_95"] == [1.0, 1.0]
    sign = paired_sign_flip(deltas)
    assert sign["enumerations"] == 2**22
    assert sign["p_value"] == pytest.approx(1 / 2**22)
    assert holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5}) == {
        "a": 0.03, "b": 0.04, "c": 0.5
    }


def test_qrels_reader_runs_only_after_ranking_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rankings = tmp_path / "rankings"
    retrieval = tmp_path / "retrieval"
    rankings.mkdir(); retrieval.mkdir()
    qrels = tmp_path / "qrels"
    qrels.write_text("must not open", encoding="utf-8")
    opened = False

    def reader(path: Path) -> bytes:
        nonlocal opened
        if path == qrels:
            opened = True
        return path.read_bytes()

    def reject(_path: Path) -> dict[str, object]:
        raise ValueError("ranking seal differs")

    monkeypatch.setattr("trec_rag.all_topic_tethered_evaluate.verify_rankings", reject)
    with pytest.raises(ValueError, match="ranking seal"):
        module._recompute_from_paths(rankings, retrieval, qrels, qrels_reader=reader)
    assert opened is False


def test_promotion_accepts_bootstrap_or_holm_significance_and_all_guards() -> None:
    assert apply_promotion_rules(_passing_metrics(), _passing_statistics())["promoted"] is True
    stats = _passing_statistics(); stats["bootstrap_ci_95"] = [-0.01, 0.02]
    assert apply_promotion_rules(_passing_metrics(), stats)["promoted"] is True
    stats["holm_adjusted_p"] = 0.05
    assert apply_promotion_rules(_passing_metrics(), stats)["failed_rules"] == ["corrected_significance"]


def test_load_qrels_accepts_standard_four_column_trec_format() -> None:
    content = "".join(f"{topic} 0 {topic}-doc 2\n" for topic in ALL_TOPIC_IDS).encode()
    parsed = _load_qrels(content)
    assert parsed["14"] == {"14-doc": 2}
    assert list(parsed) == list(ALL_TOPIC_IDS)


def test_selection_ladder_is_exact_preregistered_family_and_order() -> None:
    assert SELECTION_LADDER == (
        "RRF100-STATIC-DUAL",
        "RRF500-REINIT-DUAL",
        "RRF100-REINIT-DUAL",
        "RRF100-REINIT-DUAL-NR",
        "RRF100-STATIC-DUAL-NR",
    )
    assert set(SELECTION_LADDER) == set(ARMS) - {"RRF"}


def test_holm_family_contains_every_nonbaseline_arm_once() -> None:
    rankings, qrels, provenance = _fixture()
    result = evaluate_all_topics(rankings, qrels, provenance, depths=(1, 2, 3))
    statistics = _statistics(result["arms"])
    assert list(statistics) == list(SELECTION_LADDER)
    assert all("holm_adjusted_p" in statistics[arm] for arm in SELECTION_LADDER)


def test_judged_rate_regression_blocks_promotion() -> None:
    metrics = _passing_metrics()
    metrics["macro_judged_rate_delta_at_1000"] = -1.0
    decision = apply_promotion_rules(metrics, _passing_statistics())
    assert decision["promoted"] is False
    assert decision["rules"]["no_macro_judged_rate_regression_at_1000"] is False


def test_aggregate_tables_cover_every_required_metric() -> None:
    rankings, qrels, provenance = _fixture()
    arm = evaluate_all_topics(rankings, qrels, provenance, depths=(1, 2, 3))["arms"]["RRF100-STATIC-DUAL"]
    assert set(arm["aggregate"]) == {"pooled", "macro"}
    for table in arm["aggregate"].values():
        assert {
            "binary_recall@1", "graded_recall@1", "ndcg@1", "precision@1",
            "judged_rate@1", "normalized_recall_auc",
            "facet_only_known_relevant_retention@1", "full_union_recall_ceiling",
        } <= set(table)


def test_per_topic_deltas_and_worst_regression_preserve_identity() -> None:
    rankings, qrels, provenance = _fixture(one_loss=True)
    arm = evaluate_all_topics(rankings, qrels, provenance, depths=(1, 2, 3))["arms"]["RRF100-STATIC-DUAL"]
    delta = arm["per_topic_deltas"]["31"]
    assert "known_relevant_count@1" in delta
    assert "graded_recall@1" in delta
    worst = arm["worst_regression_at_1"]
    assert worst == {
        "topic_id": "31",
        "known_relevant_count_delta": -1,
        "binary_recall_delta": pytest.approx(-0.5),
    }
    assert arm["wins_at_1"] + arm["ties_at_1"] + arm["losses_at_1"] == 22


def _fake_bindings() -> dict[str, object]:
    return {
        "schema_version": module.SCHEMA_VERSION,
        "ranking_verified_before_qrels_open": True,
        "ranking_root_sha256": module.CANONICAL_RANKING_ROOT_SHA256,
        "ranking_path": module.PINNED_RANKINGS_PATH,
        "qrels": {"path": module.PINNED_QRELS_SUFFIX, "bytes": 1, "sha256": module.PINNED_QRELS_SHA256},
        "accepted_union": {"path": module.PINNED_UNION_PATH, "bytes": 1, "sha256": "a" * 64},
        "upstream_roots": dict(module.PINNED_UPSTREAM_ROOTS),
    }


def _reseal(path: Path) -> str:
    names = ("metrics.json", "diagnostics.json", "input_bindings.json", "summary.json")
    summary = json.loads((path / "summary.json").read_text())
    summary["artifacts"] = {
        name: module._binding((path / name).read_bytes())
        for name in names if name != "summary.json"
    }
    (path / "summary.json").write_bytes(module._pretty(summary))
    files = {name: module._binding((path / name).read_bytes()) for name in names}
    material = {"schema_version": module.SEAL_SCHEMA_VERSION, "status": "sealed_evaluation", "files": files}
    root = module._sha256(module._compact(material))
    (path / "SEALED.json").write_bytes(module._pretty({**material, "root_sha256": root}))
    return root


@pytest.mark.parametrize("attack", ["metric", "qrels", "ranking"])
def test_verify_rejects_resealed_forged_derived_or_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    rankings, qrels, provenance = _fixture()
    expected = evaluate_all_topics(rankings, qrels, provenance, depths=(1, 2, 3))
    bindings = _fake_bindings()
    output = tmp_path / "evaluation"
    _write_evaluation(expected, bindings, output)
    monkeypatch.setattr(module, "_recompute_canonical", lambda: (expected, bindings))
    if attack == "metric":
        value = json.loads((output / "metrics.json").read_text())
        value["arms"]["RRF"]["per_topic"]["14"]["binary_recall@1"] = 0.123
        (output / "metrics.json").write_bytes(module._pretty(value))
    else:
        value = json.loads((output / "input_bindings.json").read_text())
        if attack == "qrels":
            value["qrels"]["sha256"] = "0" * 64
        else:
            value["ranking_root_sha256"] = "0" * 64
        (output / "input_bindings.json").write_bytes(module._pretty(value))
    forged_root = _reseal(output)
    monkeypatch.setattr(module, "CANONICAL_EVALUATION_ROOT_SHA256", forged_root)
    with pytest.raises(ValueError, match="recomputed|binding|metric"):
        verify_evaluation(output)


def test_verify_rejects_noncanonical_evaluation_root_before_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rankings, qrels, provenance = _fixture()
    expected = evaluate_all_topics(rankings, qrels, provenance, depths=(1, 2, 3))
    output = tmp_path / "evaluation"
    _write_evaluation(expected, _fake_bindings(), output)
    monkeypatch.setattr(module, "CANONICAL_EVALUATION_ROOT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="canonical evaluation root"):
        verify_evaluation(output)


def test_canonical_v3_evaluation_root_is_pinned_after_successful_run() -> None:
    assert module.SCHEMA_VERSION.endswith("v3")
    assert module.PINNED_RANKINGS_PATH.endswith("rankings_v3")
    assert len(module.CANONICAL_EVALUATION_ROOT_SHA256) == 64
