from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

import trec_rag.facet_local_minilm_evaluate as module
from trec_rag.facet_local_minilm_evaluate import (
    PILOT_TOPIC_IDS,
    add_self_hash,
    aggregate_review_metrics,
    build_facet_retention,
    build_union_curves,
    compare_docid_sets,
    compute_prefusion_evidence,
    evaluate,
    evaluate_ranking,
    select_diagnostic_outcome,
    validate_self_hash,
)


EXACT_RANKING_ARMS = (
    "R1_LEGACY",
    "C0_TOPIC_LOCAL",
    "BF100_TOPIC_LOCAL",
    "BF50_TOPIC_LOCAL",
    "BF20_TOPIC_LOCAL",
    "BO100_TOPIC_LOCAL",
    "BB100_TOPIC_LOCAL",
    "BF100_MAXP_TOPIC_LOCAL",
    "BF100_LEGACY_FUSION",
)


class _UnreadableMapping(dict):
    def __getitem__(self, key):
        raise AssertionError(f"mapping read before topic firewall: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"mapping read before topic firewall: {key}")

    def items(self):
        raise AssertionError("mapping read before topic firewall")

    def values(self):
        raise AssertionError("mapping read before topic firewall")


def _qrels():
    return {
        "200": {"a": 3, "b": 2, "c": 1, "z": 0},
        "225": {"p": 2},
        "707": {"q": 2},
        "897": {"r": 2},
    }


def _neutral_deltas(value=0.0):
    return {
        topic_id: {
            "recall@100": value,
            "graded_recall@100": value,
            "ndcg@10": value,
        }
        for topic_id in PILOT_TOPIC_IDS
    }


def _decision_kwargs():
    return {
        "headroom": 2,
        "pre_fusion_promoted_novel": 1,
        "fusion_blocked_novel": 0,
        "final_novel": 1,
        "final_novel_vs_legacy_r1": 1,
        "net_relevant_change_vs_control": 1,
        "macro_deltas_vs_control": {
            "recall@100": 0.04,
            "graded_recall@100": 0.01,
            "ndcg@10": 0.0,
        },
        "per_topic_deltas_vs_control": _neutral_deltas(),
        "review_deltas": {"direct_answer_rate": 0.1, "wrong_domain_rate": 0.0},
        "macro_deltas_vs_legacy_r1": {
            "recall@100": 0.01,
            "graded_recall@100": 0.0,
            "ndcg@10": -0.01,
        },
        "per_topic_deltas_vs_legacy_r1": _neutral_deltas(),
    }


@pytest.mark.parametrize("failing_verifier", ["ranking", "review"])
def test_qrels_cannot_open_before_each_freeze_verifies(
    monkeypatch, tmp_path, failing_verifier
):
    def fail_ranking(path):
        raise ValueError("ranking freeze is incomplete")

    def fail_review(review, freeze):
        raise ValueError("review freeze is incomplete")

    monkeypatch.setattr(
        module,
        "verify_ranking_freeze",
        fail_ranking
        if failing_verifier == "ranking"
        else lambda path: {"freeze_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        module,
        "verify_blinded_review_freeze",
        fail_review
        if failing_verifier == "review"
        else lambda review, freeze: {"review_freeze_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        module, "load_frozen_inputs", lambda *args, **kwargs: pytest.fail("inputs loaded")
    )
    monkeypatch.setattr(
        module, "read_qrels", lambda *args, **kwargs: pytest.fail("qrels opened")
    )

    with pytest.raises(ValueError, match=f"{failing_verifier} freeze is incomplete"):
        evaluate(
            tmp_path / "freeze",
            review_freeze=tmp_path / "review",
            prior_evaluation=tmp_path / "prior.json",
            qrels_manifest=tmp_path / "safe_projection/manifest.json",
            qrels_approval=tmp_path / "approvals/qrels_access_v1.json",
            output=tmp_path / "evaluation",
        )


def test_gain_loss_accounting_reconciles():
    comparison = compare_docid_sets(baseline={"a", "b"}, candidate={"b", "c"})

    assert comparison.gained == {"c"}
    assert comparison.lost == {"a"}
    assert comparison.net_change == 0
    assert comparison.candidate_count == (
        comparison.baseline_count + len(comparison.gained) - len(comparison.lost)
    )


def test_union_curves_deduplicate_and_pair_control_with_bf_at_all_depths():
    original = {"200": ["a", "o"]}
    control = {
        ("200", "f1"): ["a", "b", "c", "d"],
        ("200", "f2"): ["b", "e", "f", "g"],
    }
    bf = {
        ("200", "f1"): ["c", "b", "d", "a"],
        ("200", "f2"): ["f", "b", "g", "e"],
    }

    curves = build_union_curves(
        original,
        control,
        bf,
        {"200": {docid: 2 for docid in "abcdefg"}},
        depths=(2, 3, 4),
    )

    assert set(curves) == {"C0_TOPIC_LOCAL", "BF100_TOPIC_LOCAL"}
    assert curves["C0_TOPIC_LOCAL"][2]["per_topic"]["200"]["docids"] == [
        "a",
        "b",
        "e",
        "o",
    ]
    assert curves["BF100_TOPIC_LOCAL"][2]["per_topic"]["200"]["docids"] == [
        "a",
        "b",
        "c",
        "f",
        "o",
    ]
    assert curves["C0_TOPIC_LOCAL"][4]["per_topic"]["200"]["docids"] == curves[
        "BF100_TOPIC_LOCAL"
    ][4]["per_topic"]["200"]["docids"]


def test_union_curves_reject_different_raw_candidate_sets_at_depth_100():
    with pytest.raises(ValueError, match="same raw candidates"):
        build_union_curves(
            {"200": ["o"]},
            {("200", "f1"): ["a"]},
            {("200", "f1"): ["b"]},
            {"200": {"a": 2, "b": 2}},
            depths=(20, 50, 100),
        )


@pytest.mark.parametrize("bad_topic", ["144", "999"])
@pytest.mark.parametrize("boundary", ["union", "retention", "prefusion"])
def test_facet_boundaries_reject_out_of_scope_topics_before_qrels_access(
    bad_topic, boundary
):
    qrels = {bad_topic: _UnreadableMapping()}
    with pytest.raises(ValueError, match="topic boundary"):
        if boundary == "union":
            build_union_curves(
                {bad_topic: ["o"]},
                {(bad_topic, "f1"): ["a"]},
                {(bad_topic, "f1"): ["a"]},
                qrels,
                depths=(20, 50, 100),
            )
        elif boundary == "retention":
            build_facet_retention(
                {"C0_TOPIC_LOCAL": {(bad_topic, "f1"): ["a"]}},
                qrels,
                baselines={"O": {bad_topic: set()}},
            )
        else:
            compute_prefusion_evidence(
                control_facets={(bad_topic, "f1"): ["a"]},
                bf_facets={(bad_topic, "f1"): ["a"]},
                stream_weights={(bad_topic, "f1"): 0.5},
                control_final={bad_topic: ["a"]},
                bf_final={bad_topic: ["a"]},
                qrels=qrels,
            )


def test_overall_and_graded_relevance_formulas_are_distinct():
    metrics = evaluate_ranking(["a", "c", "z", "missing"], _qrels()["200"])

    # Overall relevance uses the frozen binary threshold grade >= 2.
    assert metrics["relevant_count@10"] == 1
    assert metrics["precision@10"] == pytest.approx(0.1)
    assert metrics["recall@100"] == pytest.approx(0.5)
    # Graded recall retains every positive grade in numerator and denominator.
    assert metrics["graded_recall@100"] == pytest.approx(4 / 6)
    assert metrics["judged_rate@10"] == pytest.approx(0.3)
    assert metrics["judged_rate@100"] == pytest.approx(0.03)


def test_ndcg_idcg_uses_all_qrels_when_high_grade_document_is_omitted():
    metrics = evaluate_ranking(["b", "x"], _qrels()["200"])
    actual_dcg = (2**2 - 1) / math.log2(2)
    ideal_dcg = (
        (2**3 - 1) / math.log2(2)
        + (2**2 - 1) / math.log2(3)
        + (2**1 - 1) / math.log2(4)
    )

    assert metrics["ndcg@10"] == pytest.approx(actual_dcg / ideal_dcg)
    assert metrics["oracle_ndcg@10_from_top50"] == pytest.approx(
        actual_dcg / ideal_dcg
    )
    assert metrics["oracle_ndcg@10_from_top100"] == pytest.approx(
        actual_dcg / ideal_dcg
    )


def test_facet_retention_reports_depths_and_unique_relevant_contributions():
    result = build_facet_retention(
        {
            "C0_TOPIC_LOCAL": {("200", "f1"): ["a", "b", "x"]},
            "BF100_TOPIC_LOCAL": {("200", "f1"): ["b", "a", "x"]},
        },
        _qrels(),
        baselines={
            "O": {"200": {"a"}},
            "C0_TOPIC_LOCAL": {"200": {"a", "x"}},
            "R1_LEGACY": {"200": {"x"}},
        },
        depths=(1, 2, 3),
    )

    bf = result["BF100_TOPIC_LOCAL"]["200/f1"]
    assert bf["relevant_retained"] == {"1": 1, "2": 2, "3": 2}
    assert bf["unique_relevant_beyond"]["O"] == ["b"]
    assert bf["unique_relevant_beyond"]["C0_TOPIC_LOCAL"] == ["b"]
    assert bf["unique_relevant_beyond"]["R1_LEGACY"] == ["a", "b"]


def test_prefusion_evidence_keeps_raw_union_facet_rank_and_final_sets_separate():
    result = compute_prefusion_evidence(
        control_facets={("200", "f1"): ["x", "a", "b"]},
        bf_facets={("200", "f1"): ["b", "x", "a"]},
        stream_weights={("200", "f1"): 0.25},
        control_final={"200": ["x", "a"]},
        bf_final={"200": ["x", "a"]},
        qrels={"200": {"b": 2}},
        promotion_depth=1,
    )

    assert result["raw_union_docids"]["200"] == ["a", "b", "x"]
    assert len(result["facet_rows"]) == 3
    assert result["facet_rows"][0]["bf_rank"] == 1
    assert result["facet_rows"][0]["control_rank"] == 3
    assert result["facet_rows"][0]["qrel_grade"] == 2
    assert result["facet_rows"][0]["is_relevant"] is True
    assert result["facet_rows"][0]["rrf_contribution"] == pytest.approx(0.25 / 61)
    assert result["final_docids"]["200"] == ["a", "x"]
    assert result["pre_fusion_promoted_novel_docids"] == ["200/b"]
    assert result["fusion_blocked_novel_docids"] == ["200/b"]


def test_review_macro_is_over_facets_and_shared_items_count_in_both_arms():
    rows = [
        {
            "item_id": "shared",
            "label": {
                "relevance": "direct_answer",
                "wrong_domain": False,
                "low_quality": False,
            },
            "memberships": [
                {"arm": "C0_TOPIC_LOCAL", "facet": {"topic_id": "200", "variant_name": "f1"}},
                {"arm": "BF50_TOPIC_LOCAL", "facet": {"topic_id": "200", "variant_name": "f1"}},
            ],
        },
        {
            "item_id": "control-only",
            "label": {
                "relevance": "not_facet_relevant",
                "wrong_domain": True,
                "low_quality": True,
            },
            "memberships": [
                {"arm": "C0_TOPIC_LOCAL", "facet": {"topic_id": "200", "variant_name": "f2"}}
            ],
        },
        {
            "item_id": "bf-only",
            "label": {
                "relevance": "partial_or_related",
                "wrong_domain": False,
                "low_quality": False,
            },
            "memberships": [
                {"arm": "BF50_TOPIC_LOCAL", "facet": {"topic_id": "200", "variant_name": "f2"}}
            ],
        },
    ]

    review = aggregate_review_metrics(rows)

    assert review["counts_by_arm"]["C0_TOPIC_LOCAL"]["denominator"] == 2
    assert review["counts_by_arm"]["BF50_TOPIC_LOCAL"]["denominator"] == 2
    # Macro rates are first calculated within f1/f2, then averaged.
    assert review["macro_by_arm"]["C0_TOPIC_LOCAL"]["direct_answer_rate"] == 0.5
    assert review["macro_by_arm"]["BF50_TOPIC_LOCAL"]["direct_answer_rate"] == 0.5
    assert review["macro_by_arm"]["C0_TOPIC_LOCAL"]["wrong_domain_rate"] == 0.5
    assert review["macro_by_arm"]["BF50_TOPIC_LOCAL"]["wrong_domain_rate"] == 0.0


@pytest.mark.parametrize("bad_topic", ["144", "999"])
def test_review_memberships_reject_out_of_scope_topics_before_label_access(bad_topic):
    rows = [
        {
            "label": _UnreadableMapping(),
            "memberships": [
                {
                    "arm": "C0_TOPIC_LOCAL",
                    "facet": {"topic_id": bad_topic, "variant_name": "f1"},
                }
            ],
        }
    ]

    with pytest.raises(ValueError, match="topic boundary"):
        aggregate_review_metrics(rows)


def test_candidate_generation_gap_is_first_and_never_opens_stage_a():
    kwargs = _decision_kwargs()
    kwargs["headroom"] = 0

    result = select_diagnostic_outcome(**kwargs)

    assert result["outcome"] == "candidate_generation_gap"
    assert result["stage_a_permitted"] is False
    assert result["stage_a_executed"] is False


def test_promotion_requires_every_corrected_and_legacy_guard():
    result = select_diagnostic_outcome(**_decision_kwargs())

    assert result["outcome"] == "B_promotes_coverage"
    assert result["stage_a_permitted"] is True
    assert result["stage_a_executed"] is False
    assert result["failed_promotion_guards"] == []


@pytest.mark.parametrize(
    ("mutation", "guard"),
    [
        (lambda values: values.update(final_novel=0), "final_novel"),
        (lambda values: values.update(net_relevant_change_vs_control=0), "positive_net_relevant_change_vs_control"),
        (lambda values: values["macro_deltas_vs_control"].update({"recall@100": 0.0}), "positive_macro_recall_vs_control"),
        (lambda values: values["macro_deltas_vs_control"].update({"graded_recall@100": -0.001}), "nonnegative_macro_graded_recall_vs_control"),
        (lambda values: values["per_topic_deltas_vs_control"]["200"].update({"recall@100": -0.021}), "per_topic_recall_floor_vs_control"),
        (lambda values: values["per_topic_deltas_vs_control"]["200"].update({"graded_recall@100": -0.021}), "per_topic_graded_recall_floor_vs_control"),
        (lambda values: values["review_deltas"].update({"direct_answer_rate": 0.0}), "positive_direct_answer_rate_delta"),
        (lambda values: values["review_deltas"].update({"wrong_domain_rate": 0.001}), "nonpositive_wrong_domain_rate_delta"),
        (lambda values: values["macro_deltas_vs_control"].update({"ndcg@10": -0.021}), "macro_ndcg_floor_vs_control"),
        (lambda values: values["per_topic_deltas_vs_control"]["200"].update({"ndcg@10": -0.101}), "per_topic_ndcg_floor_vs_control"),
        (lambda values: values["macro_deltas_vs_legacy_r1"].update({"recall@100": -0.001}), "nonnegative_macro_recall_vs_legacy_r1"),
        (lambda values: values["macro_deltas_vs_legacy_r1"].update({"graded_recall@100": -0.001}), "nonnegative_macro_graded_recall_vs_legacy_r1"),
        (lambda values: values["per_topic_deltas_vs_legacy_r1"]["200"].update({"recall@100": -0.021}), "per_topic_recall_floor_vs_legacy_r1"),
        (lambda values: values["per_topic_deltas_vs_legacy_r1"]["200"].update({"graded_recall@100": -0.021}), "per_topic_graded_recall_floor_vs_legacy_r1"),
        (lambda values: values["macro_deltas_vs_legacy_r1"].update({"ndcg@10": -0.021}), "macro_ndcg_floor_vs_legacy_r1"),
        (lambda values: values["per_topic_deltas_vs_legacy_r1"]["200"].update({"ndcg@10": -0.101}), "per_topic_ndcg_floor_vs_legacy_r1"),
    ],
)
def test_any_failed_promotion_guard_preserves_gain_without_stage_a(mutation, guard):
    kwargs = _decision_kwargs()
    mutation(kwargs)

    result = select_diagnostic_outcome(**kwargs)

    assert result["outcome"] == "B_coverage_gain_with_regression"
    assert guard in result["failed_promotion_guards"]
    assert result["stage_a_permitted"] is False


def test_fusion_block_requires_promoted_relevant_doc_to_remain_out_of_final():
    kwargs = _decision_kwargs()
    kwargs.update(
        final_novel=0,
        net_relevant_change_vs_control=0,
        fusion_blocked_novel=1,
        macro_deltas_vs_control={
            "recall@100": 0.0,
            "graded_recall@100": 0.0,
            "ndcg@10": 0.0,
        },
    )

    result = select_diagnostic_outcome(**kwargs)

    assert result["outcome"] == "B_filters_but_fusion_blocks"
    assert result["stage_a_permitted"] is False

    kwargs["fusion_blocked_novel"] = 0
    assert select_diagnostic_outcome(**kwargs)["outcome"] == "B_query_window_or_model_gap"


def test_ranking_only_gain_requires_no_coverage_gain():
    kwargs = _decision_kwargs()
    kwargs.update(
        final_novel=0,
        net_relevant_change_vs_control=0,
        pre_fusion_promoted_novel=0,
        macro_deltas_vs_control={
            "recall@100": 0.0,
            "graded_recall@100": 0.0,
            "ndcg@10": 0.01,
        },
        review_deltas={"direct_answer_rate": 0.0, "wrong_domain_rate": 0.0},
    )

    assert select_diagnostic_outcome(**kwargs)["outcome"] == "ranking_only_gain"


def test_fallback_is_query_window_or_model_gap():
    kwargs = _decision_kwargs()
    kwargs.update(
        final_novel=0,
        net_relevant_change_vs_control=0,
        pre_fusion_promoted_novel=0,
        macro_deltas_vs_control={
            "recall@100": 0.0,
            "graded_recall@100": 0.0,
            "ndcg@10": 0.0,
        },
        review_deltas={"direct_answer_rate": 0.0, "wrong_domain_rate": 0.0},
    )

    assert select_diagnostic_outcome(**kwargs)["outcome"] == "B_query_window_or_model_gap"


def test_sidecar_topic_rejection_occurs_before_qrels_data_access(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "pilot-qrels-projection-v1",
                "status": "authorized_projection",
                "topic_ids": ["200", "225", "707"],
                "projection_path": "projection.txt",
                "projection_sha256": "a" * 64,
            }
        )
    )
    monkeypatch.setattr(module, "verify_ranking_freeze", lambda path: {"freeze_sha256": "b" * 64})
    monkeypatch.setattr(module, "verify_blinded_review_freeze", lambda review, freeze: {"review_freeze_sha256": "c" * 64})
    monkeypatch.setattr(
        module,
        "load_frozen_inputs",
        lambda *args, **kwargs: _synthetic_frozen_evaluation_inputs(),
    )
    monkeypatch.setattr(module, "read_qrels", lambda path: pytest.fail("qrels opened"))

    with pytest.raises(ValueError, match="exactly topics"):
        evaluate(
            tmp_path / "freeze",
            review_freeze=tmp_path / "review",
            prior_evaluation=tmp_path / "prior.json",
            qrels_manifest=manifest,
            qrels_approval=tmp_path / "approval.json",
            output=tmp_path / "evaluation",
        )


def test_self_hash_is_deterministic_and_rejects_tampering():
    first = add_self_hash({"schema_version": "fixture-v1", "values": [3, 2, 1]})
    second = add_self_hash({"values": [3, 2, 1], "schema_version": "fixture-v1"})

    assert first == second
    assert validate_self_hash(first) is True
    broken = {**first, "values": [1, 2, 3]}
    with pytest.raises(ValueError, match="self-hash"):
        validate_self_hash(broken)


def test_authorization_binds_projection_and_both_freezes(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": "pilot-qrels-projection-v1",
        "status": "authorized_projection",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "projection_path": "projection.txt",
        "projection_sha256": "a" * 64,
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    approval_path = tmp_path / "approval.json"
    approval = {
        "schema_version": "pilot-qrels-access-approval-v1",
        "status": "approved",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "qrels_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "qrels_projection_sha256": "a" * 64,
        "ranking_freeze_sha256": "b" * 64,
        "review_freeze_sha256": "c" * 64,
    }
    approval_path.write_text(json.dumps(approval))

    validated = module.validate_qrels_authorization(
        manifest_path,
        approval_path,
        ranking_freeze_sha256="b" * 64,
        review_freeze_sha256="c" * 64,
    )
    assert validated["projection_sha256"] == "a" * 64

    approval["ranking_freeze_sha256"] = "d" * 64
    approval_path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="ranking freeze hash"):
        module.validate_qrels_authorization(
            manifest_path,
            approval_path,
            ranking_freeze_sha256="b" * 64,
            review_freeze_sha256="c" * 64,
        )


def test_repeat_access_is_refused_before_qrels_reopens(monkeypatch, tmp_path):
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "qrels_access_receipt.json").write_text("{}")
    monkeypatch.setattr(module, "read_qrels", lambda path: pytest.fail("qrels reopened"))

    with pytest.raises(FileExistsError, match="repeat qrels access"):
        evaluate(
            tmp_path / "freeze",
            review_freeze=tmp_path / "review",
            prior_evaluation=tmp_path / "prior.json",
            qrels_manifest=tmp_path / "manifest.json",
            qrels_approval=tmp_path / "approval.json",
            output=output,
        )


def test_cli_has_no_all_topic_qrels_argument():
    options = {action.option_strings[0] for action in module.build_argument_parser()._actions if action.option_strings}

    assert options == {
        "-h",
        "--freeze",
        "--review-freeze",
        "--prior-evaluation",
        "--qrels-manifest",
        "--qrels-approval",
        "--output",
    }
    assert "--qrels" not in options


def _synthetic_frozen_evaluation_inputs():
    original = {}
    control_facets = {}
    bf_facets = {}
    rankings = {arm: {} for arm in EXACT_RANKING_ARMS}
    ranking_rows = {arm: {} for arm in rankings}
    details = {}
    review_rows = []
    for topic_id in PILOT_TOPIC_IDS:
        original_docid = f"o-{topic_id}"
        control_docid = f"c-{topic_id}"
        variant = f"facet:{topic_id}:01"
        facet_key = (topic_id, variant)
        original[topic_id] = [original_docid]
        if topic_id == "200":
            novel_docid = "novel-200"
            decoys = [f"decoy-{index}" for index in range(1, 20)]
            control_facets[facet_key] = [control_docid, *decoys, novel_docid]
            bf_facets[facet_key] = [novel_docid, control_docid, *decoys]
            rankings["BF100_TOPIC_LOCAL"][topic_id] = [
                original_docid,
                novel_docid,
                control_docid,
            ]
        else:
            control_facets[facet_key] = [control_docid]
            bf_facets[facet_key] = [control_docid]
            rankings["BF100_TOPIC_LOCAL"][topic_id] = [
                original_docid,
                control_docid,
            ]
        rankings["C0_TOPIC_LOCAL"][topic_id] = [original_docid, control_docid]
        rankings["R1_LEGACY"][topic_id] = [original_docid, control_docid]
        for arm in EXACT_RANKING_ARMS:
            if topic_id not in rankings[arm]:
                rankings[arm][topic_id] = list(
                    rankings["C0_TOPIC_LOCAL"][topic_id]
                )
        for arm in rankings:
            ranking_rows[arm][topic_id] = [
                {
                    "docid": docid,
                    "rank": rank,
                    "text": f"passage for {docid}",
                    "topic_id": topic_id,
                }
                for rank, docid in enumerate(rankings[arm][topic_id], start=1)
            ]
            for row in ranking_rows[arm][topic_id]:
                details[(topic_id, row["docid"])] = row
        review_rows.extend(
            [
                {
                    "item_id": f"control-{topic_id}",
                    "label": {
                        "relevance": "not_facet_relevant",
                        "wrong_domain": False,
                        "low_quality": False,
                    },
                    "memberships": [
                        {
                            "arm": "C0_TOPIC_LOCAL",
                            "facet": {
                                "topic_id": topic_id,
                                "variant_name": variant,
                            },
                        }
                    ],
                },
                {
                    "item_id": f"bf-{topic_id}",
                    "label": {
                        "relevance": "direct_answer",
                        "wrong_domain": False,
                        "low_quality": False,
                    },
                    "memberships": [
                        {
                            "arm": "BF50_TOPIC_LOCAL",
                            "facet": {
                                "topic_id": topic_id,
                                "variant_name": variant,
                            },
                        }
                    ],
                },
            ]
        )
    return {
        "bf_facets": bf_facets,
        "control_facets": control_facets,
        "document_details": details,
        "original": original,
        "prior_evaluation": {"schema_version": "prior-fixture-v1"},
        "prior_evaluation_sha256": "d" * 64,
        "ranking_rows": ranking_rows,
        "rankings": rankings,
        "review_rows": review_rows,
        "stream_weights": {key: 0.5 for key in control_facets},
    }


def _synthetic_qrels():
    return {
        "200": {"o-200": 3, "c-200": 2, "novel-200": 2},
        "225": {"o-225": 3, "c-225": 2},
        "707": {"o-707": 3, "c-707": 2},
        "897": {"o-897": 3, "c-897": 2},
    }


def _artifact_bindings():
    return {
        "ranking_freeze_sha256": "b" * 64,
        "review_freeze_sha256": "c" * 64,
        "prior_evaluation_sha256": "d" * 64,
        "qrels_manifest_sha256": "e" * 64,
        "qrels_projection_sha256": "f" * 64,
    }


def _compact_json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _pretty_json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows):
    return b"".join(_compact_json_bytes(row) for row in rows)


def _write_synthetic_verified_freezes(tmp_path):
    frozen_inputs = _synthetic_frozen_evaluation_inputs()
    freeze = tmp_path / "freeze"
    review = tmp_path / "review"
    freeze.mkdir()
    review.mkdir()
    artifacts = {}

    def write_artifact(relative, content, *, rows=None):
        path = freeze / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        record = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        if rows is not None:
            record["rows"] = rows
        artifacts[relative] = record
        return record

    ranking_records = {}
    for arm in EXACT_RANKING_ARMS:
        rows = [
            row
            for topic_id in PILOT_TOPIC_IDS
            for row in frozen_inputs["ranking_rows"][arm][topic_id]
        ]
        relative = f"rankings/{arm}.jsonl"
        record = write_artifact(relative, _jsonl_bytes(rows), rows=len(rows))
        ranking_records[arm] = {
            "file_sha256": record["sha256"],
            "path": relative,
            "rows": len(rows),
        }

    stream_records = []
    retriever = "synthetic-retriever"
    for topic_id in PILOT_TOPIC_IDS:
        original_variant = f"original:{topic_id}"
        original_rows = [
            {
                "aggregation": "bm25",
                "document_id": docid,
                "family": "original",
                "passage": f"passage for {docid}",
                "rank": rank,
                "topic_id": topic_id,
                "variant": original_variant,
            }
            for rank, docid in enumerate(frozen_inputs["original"][topic_id], start=1)
        ]
        original_path = f"streams/bm25/{topic_id}-original.jsonl"
        original_record = write_artifact(
            original_path, _jsonl_bytes(original_rows), rows=len(original_rows)
        )
        stream_records.append(
            {
                "aggregation": "bm25",
                "family": "original",
                "file_sha256": original_record["sha256"],
                "path": original_path,
                "retriever_name": retriever,
                "rows": len(original_rows),
                "topic_id": topic_id,
                "variant_name": original_variant,
            }
        )

    for (topic_id, variant), control_docids in frozen_inputs[
        "control_facets"
    ].items():
        for aggregation, docids in (
            ("bm25", control_docids),
            ("top4", frozen_inputs["bf_facets"][(topic_id, variant)]),
        ):
            rows = [
                {
                    "aggregation": aggregation,
                    "document_id": docid,
                    "family": "facet",
                    "passage": f"passage for {docid}",
                    "rank": rank,
                    "topic_id": topic_id,
                    "variant": variant,
                }
                for rank, docid in enumerate(docids, start=1)
            ]
            relative = f"streams/{aggregation}/{topic_id}-{aggregation}.jsonl"
            record = write_artifact(relative, _jsonl_bytes(rows), rows=len(rows))
            stream_records.append(
                {
                    "aggregation": aggregation,
                    "family": "facet",
                    "file_sha256": record["sha256"],
                    "path": relative,
                    "retriever_name": retriever,
                    "rows": len(rows),
                    "topic_id": topic_id,
                    "variant_name": variant,
                }
            )

    weights = {
        "schema_version": "family-rrf-topic-local-v2",
        "weights": [
            {
                "topic_id": topic_id,
                "variant_name": variant,
                "weight": weight,
            }
            for (topic_id, variant), weight in frozen_inputs["stream_weights"].items()
        ],
    }
    weight_path = "fusion_weights_v2.json"
    write_artifact(weight_path, _pretty_json_bytes(weights))
    freeze_root = {
        "artifacts": artifacts,
        "fusion_tables": {"family_rrf_topic_local_v2": weight_path},
        "qrels_opened": False,
        "rankings": ranking_records,
        "schema_version": "facet-local-minilm-ranking-freeze-v1",
        "status": "frozen_before_qrels",
        "streams": stream_records,
        "topic_ids": list(PILOT_TOPIC_IDS),
    }
    freeze_root["freeze_sha256"] = hashlib.sha256(
        _compact_json_bytes(freeze_root)
    ).hexdigest()
    (freeze / "freeze.json").write_bytes(_pretty_json_bytes(freeze_root))

    review_rows = frozen_inputs["review_rows"]
    unmasked = _jsonl_bytes(review_rows)
    (review / "unmasked_items.jsonl").write_bytes(unmasked)
    review_root = {
        "bindings": {
            "ranking_freeze_sha256": freeze_root["freeze_sha256"],
            "unmasked_items_sha256": hashlib.sha256(unmasked).hexdigest(),
        },
        "qrels_opened": False,
        "schema_version": "facet-local-minilm-review-freeze-v2",
        "status": "review_frozen_before_qrels",
    }
    review_root["review_freeze_sha256"] = hashlib.sha256(
        _pretty_json_bytes(review_root)
    ).hexdigest()
    (review / "review_freeze.json").write_bytes(_pretty_json_bytes(review_root))

    prior = tmp_path / "prior.json"
    prior.write_bytes(_pretty_json_bytes({"schema_version": "prior-fixture-v1"}))
    return {
        "freeze": freeze,
        "freeze_sha256": freeze_root["freeze_sha256"],
        "prior": prior,
        "review": review,
        "review_freeze_sha256": review_root["review_freeze_sha256"],
        "stream_to_mutate": freeze / stream_records[-1]["path"],
    }


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_artifact_builder_requires_exact_preregistered_ranking_arms(mutation):
    frozen_inputs = _synthetic_frozen_evaluation_inputs()
    rankings = dict(frozen_inputs["rankings"])
    if mutation == "missing":
        rankings.pop("BF20_TOPIC_LOCAL")
    else:
        rankings["UNREGISTERED_ARM"] = rankings["C0_TOPIC_LOCAL"]
    frozen_inputs = {**frozen_inputs, "rankings": rankings}

    with pytest.raises(ValueError, match="exact preregistered system set"):
        module.build_evaluation_artifacts(
            frozen_inputs,
            _synthetic_qrels(),
            bindings=_artifact_bindings(),
        )


@pytest.mark.parametrize("boundary", ["original", "ranking", "review"])
@pytest.mark.parametrize("bad_topic", ["144", "999"])
def test_artifact_builder_rejects_out_of_scope_topics_before_qrels_values(
    boundary, bad_topic
):
    frozen_inputs = _synthetic_frozen_evaluation_inputs()
    if boundary == "original":
        frozen_inputs["original"] = {
            **frozen_inputs["original"],
            bad_topic: ["forbidden"],
        }
    elif boundary == "ranking":
        rankings = dict(frozen_inputs["rankings"])
        rankings["C0_TOPIC_LOCAL"] = {
            **rankings["C0_TOPIC_LOCAL"],
            bad_topic: ["forbidden"],
        }
        frozen_inputs["rankings"] = rankings
    else:
        frozen_inputs["review_rows"] = [
            *frozen_inputs["review_rows"],
            {
                "label": _UnreadableMapping(),
                "memberships": [
                    {
                        "arm": "C0_TOPIC_LOCAL",
                        "facet": {
                            "topic_id": bad_topic,
                            "variant_name": "forbidden",
                        },
                    }
                ],
            },
        ]
    unreadable_qrels = {
        topic_id: _UnreadableMapping() for topic_id in PILOT_TOPIC_IDS
    }

    with pytest.raises(ValueError, match="topic boundary"):
        module.build_evaluation_artifacts(
            frozen_inputs,
            unreadable_qrels,
            bindings=_artifact_bindings(),
        )


def _write_synthetic_qrels_authorization(
    tmp_path,
    *,
    ranking_freeze_sha256="b" * 64,
    review_freeze_sha256="c" * 64,
):
    projection = tmp_path / "projection.txt"
    projection.write_text(
        "\n".join(
            [
                "200 0 o-200 3",
                "200 0 c-200 2",
                "200 0 novel-200 2",
                "225 0 o-225 3",
                "225 0 c-225 2",
                "707 0 o-707 3",
                "707 0 c-707 2",
                "897 0 o-897 3",
                "897 0 c-897 2",
            ]
        )
        + "\n"
    )
    projection_sha256 = hashlib.sha256(projection.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "pilot-qrels-projection-v1",
        "status": "authorized_projection",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "projection_path": projection.name,
        "projection_sha256": projection_sha256,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "schema_version": "pilot-qrels-access-approval-v1",
                "status": "approved",
                "topic_ids": list(PILOT_TOPIC_IDS),
                "qrels_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "qrels_projection_sha256": projection_sha256,
                "ranking_freeze_sha256": ranking_freeze_sha256,
                "review_freeze_sha256": review_freeze_sha256,
            }
        )
    )
    return manifest_path, approval_path


@pytest.mark.parametrize("bad_topic", ["144", "999"])
def test_evaluate_rejects_frozen_input_topics_before_qrels_reader(
    monkeypatch, tmp_path, bad_topic
):
    frozen_inputs = _synthetic_frozen_evaluation_inputs()
    frozen_inputs["original"] = {
        **frozen_inputs["original"],
        bad_topic: ["forbidden"],
    }
    monkeypatch.setattr(
        module, "verify_ranking_freeze", lambda path: {"freeze_sha256": "b" * 64}
    )
    monkeypatch.setattr(
        module,
        "verify_blinded_review_freeze",
        lambda review, freeze: {"review_freeze_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        module, "load_frozen_inputs", lambda *args, **kwargs: frozen_inputs
    )
    monkeypatch.setattr(
        module, "read_qrels", lambda *args, **kwargs: pytest.fail("qrels opened")
    )
    output = tmp_path / "evaluation"

    with pytest.raises(ValueError, match="topic boundary"):
        evaluate(
            tmp_path / "freeze",
            review_freeze=tmp_path / "review",
            prior_evaluation=tmp_path / "prior.json",
            qrels_manifest=tmp_path / "manifest.json",
            qrels_approval=tmp_path / "approval.json",
            output=output,
        )
    assert not (output / "qrels_access_receipt.json").exists()


def test_evaluate_publishes_eight_self_hashed_bound_artifacts(
    monkeypatch, tmp_path
):
    frozen = _write_synthetic_verified_freezes(tmp_path)
    manifest, approval = _write_synthetic_qrels_authorization(
        tmp_path,
        ranking_freeze_sha256=frozen["freeze_sha256"],
        review_freeze_sha256=frozen["review_freeze_sha256"],
    )
    monkeypatch.setattr(
        module,
        "verify_ranking_freeze",
        lambda path: {"freeze_sha256": frozen["freeze_sha256"]},
    )
    monkeypatch.setattr(
        module,
        "verify_blinded_review_freeze",
        lambda review, freeze: {
            "review_freeze_sha256": frozen["review_freeze_sha256"]
        },
    )
    output = tmp_path / "evaluation"
    projection = tmp_path / "projection.txt"
    original_read_bytes = Path.read_bytes
    projection_reads = []

    def guarded_read_bytes(path):
        if path == projection:
            assert (output / "qrels_access_receipt.json").exists()
            projection_reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    result = evaluate(
        frozen["freeze"],
        review_freeze=frozen["review"],
        prior_evaluation=frozen["prior"],
        qrels_manifest=manifest,
        qrels_approval=approval,
        output=output,
    )

    artifact_names = {
        "raw_union.json",
        "prefusion.json",
        "facet_retention.json",
        "systems.json",
        "gains_losses.json",
        "review_metrics.json",
        "representatives.json",
        "decision.json",
    }
    assert {path.name for path in output.iterdir()} == {
        *artifact_names,
        "qrels_access_receipt.json",
    }
    assert projection_reads == [projection]
    assert set(result["artifacts"]) == artifact_names
    for name in artifact_names:
        payload = json.loads((output / name).read_text())
        assert validate_self_hash(payload) is True
        assert payload["bindings"]["ranking_freeze_sha256"] == frozen[
            "freeze_sha256"
        ]
        assert payload["bindings"]["review_freeze_sha256"] == frozen[
            "review_freeze_sha256"
        ]
    assert result["artifacts"]["decision.json"]["decision"]["outcome"] == (
        "B_promotes_coverage"
    )
    assert set(result["artifacts"]["systems.json"]["systems"]) == {
        "O",
        *EXACT_RANKING_ARMS,
    }
    assert result["artifacts"]["decision.json"]["decision"][
        "stage_a_executed"
    ] is False
    assert result["artifacts"]["prefusion.json"][
        "pre_fusion_promoted_novel_docids"
    ] == ["200/novel-200"]
    topic_curve = result["artifacts"]["raw_union.json"]["curves"][
        "C0_TOPIC_LOCAL"
    ]["100"]["per_topic"]["200"]
    assert topic_curve["comparisons"]["O"]["gained"] == [
        "c-200",
        "novel-200",
    ]
    assert topic_curve["comparisons"]["C0_TOPIC_LOCAL"]["gained"] == [
        "novel-200"
    ]
    aggregate_curve = result["artifacts"]["raw_union.json"]["curves"][
        "C0_TOPIC_LOCAL"
    ]["100"]["aggregate"]
    assert aggregate_curve["unique_candidate_documents"] == 28
    assert aggregate_curve["relevant_documents"] == 9
    loaded_again = module.load_frozen_inputs(
        frozen["freeze"],
        frozen["review"],
        frozen["prior"],
        ranking_freeze_sha256=frozen["freeze_sha256"],
        review_freeze_sha256=frozen["review_freeze_sha256"],
    )
    assert result["artifacts"] == module.build_evaluation_artifacts(
        loaded_again,
        result["qrels"],
        bindings=result["artifacts"]["decision.json"]["bindings"],
    )


def test_mutation_after_freeze_verification_is_rejected_before_qrels(
    monkeypatch, tmp_path
):
    frozen = _write_synthetic_verified_freezes(tmp_path)

    def verify_then_mutate(path):
        frozen["stream_to_mutate"].write_bytes(b'{"tampered":true}\n')
        return {"freeze_sha256": frozen["freeze_sha256"]}

    monkeypatch.setattr(module, "verify_ranking_freeze", verify_then_mutate)
    monkeypatch.setattr(
        module,
        "verify_blinded_review_freeze",
        lambda review, freeze: {
            "review_freeze_sha256": frozen["review_freeze_sha256"]
        },
    )
    monkeypatch.setattr(
        module, "read_qrels", lambda *args, **kwargs: pytest.fail("qrels opened")
    )
    output = tmp_path / "evaluation"

    with pytest.raises(ValueError, match="authenticated artifact hash"):
        evaluate(
            frozen["freeze"],
            review_freeze=frozen["review"],
            prior_evaluation=frozen["prior"],
            qrels_manifest=tmp_path / "manifest.json",
            qrels_approval=tmp_path / "approval.json",
            output=output,
        )
    assert not (output / "qrels_access_receipt.json").exists()


@pytest.mark.parametrize("bad_topic", ["144", "999"])
def test_declared_stream_topics_fail_before_any_bound_source_read(
    monkeypatch, tmp_path, bad_topic
):
    frozen = _write_synthetic_verified_freezes(tmp_path)
    root_path = frozen["freeze"] / "freeze.json"
    root = json.loads(root_path.read_text())
    root["streams"][0]["topic_id"] = bad_topic
    root.pop("freeze_sha256")
    root["freeze_sha256"] = hashlib.sha256(_compact_json_bytes(root)).hexdigest()
    root_path.write_bytes(_pretty_json_bytes(root))
    monkeypatch.setattr(
        module,
        "_read_authenticated_bytes",
        lambda *args, **kwargs: pytest.fail("declared source read"),
        raising=False,
    )

    with pytest.raises(ValueError, match="topic boundary"):
        module.load_frozen_inputs(
            frozen["freeze"],
            frozen["review"],
            frozen["prior"],
            ranking_freeze_sha256=root["freeze_sha256"],
            review_freeze_sha256=frozen["review_freeze_sha256"],
        )


def test_existing_evaluation_leaf_blocks_before_qrels_access(monkeypatch, tmp_path):
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "systems.json").write_text("{}")
    monkeypatch.setattr(module, "read_qrels", lambda path: pytest.fail("qrels opened"))

    with pytest.raises(FileExistsError, match="evaluation output"):
        evaluate(
            tmp_path / "freeze",
            review_freeze=tmp_path / "review",
            prior_evaluation=tmp_path / "prior.json",
            qrels_manifest=tmp_path / "manifest.json",
            qrels_approval=tmp_path / "approval.json",
            output=output,
        )
