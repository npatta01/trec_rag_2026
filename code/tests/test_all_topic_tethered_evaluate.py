from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_rag.all_topic_facet_contract import ALL_TOPIC_IDS
from trec_rag.all_topic_tethered_rank import ARMS
from trec_rag.all_topic_tethered_evaluate import (
    _load_qrels,
    apply_promotion_rules,
    evaluate_all_topics,
    evaluate_frozen,
    holm_adjust,
    paired_bootstrap,
    paired_sign_flip,
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
        "wins_at_1000": 8,
        "protected_prefix_identical": True,
        "judged_rate_interpretable": True,
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
        evaluate_frozen(rankings, retrieval, qrels, tmp_path / "out", qrels_reader=reader)
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
