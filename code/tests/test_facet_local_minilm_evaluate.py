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


def test_qrels_cannot_open_before_both_freezes_verify(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "read_qrels", lambda path: pytest.fail("qrels opened"))

    with pytest.raises(ValueError, match="freeze is incomplete"):
        evaluate(
            tmp_path / "incomplete-freeze",
            qrels_manifest=tmp_path / "safe_projection/manifest.json",
            qrels_approval=tmp_path / "approvals/qrels_access_v1.json",
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


def test_ndcg_and_oracles_use_exact_dcg_formulas():
    metrics = evaluate_ranking(["b", "x", "a"], _qrels()["200"])
    actual_dcg = (2**2 - 1) / math.log2(2) + (2**3 - 1) / math.log2(4)
    ideal_dcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)

    assert metrics["ndcg@10"] == pytest.approx(actual_dcg / ideal_dcg)
    oracle_dcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    full_ideal_dcg = oracle_dcg + (2**1 - 1) / math.log2(4)
    assert metrics["oracle_ndcg@10_from_top50"] == pytest.approx(
        oracle_dcg / full_ideal_dcg
    )
    assert metrics["oracle_ndcg@10_from_top100"] == pytest.approx(
        oracle_dcg / full_ideal_dcg
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
    monkeypatch.setattr(module, "load_frozen_inputs", lambda *args: {})
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
    rankings = {
        "R1_LEGACY": {},
        "C0_TOPIC_LOCAL": {},
        "BF100_TOPIC_LOCAL": {},
    }
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


def _write_synthetic_qrels_authorization(tmp_path):
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
                "ranking_freeze_sha256": "b" * 64,
                "review_freeze_sha256": "c" * 64,
            }
        )
    )
    return manifest_path, approval_path


def test_evaluate_publishes_eight_self_hashed_bound_artifacts(
    monkeypatch, tmp_path
):
    frozen_inputs = _synthetic_frozen_evaluation_inputs()
    manifest, approval = _write_synthetic_qrels_authorization(tmp_path)
    monkeypatch.setattr(
        module, "verify_ranking_freeze", lambda path: {"freeze_sha256": "b" * 64}
    )
    monkeypatch.setattr(
        module,
        "verify_blinded_review_freeze",
        lambda review, freeze: {"review_freeze_sha256": "c" * 64},
    )
    monkeypatch.setattr(module, "load_frozen_inputs", lambda *args: frozen_inputs)
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
        tmp_path / "freeze",
        review_freeze=tmp_path / "review",
        prior_evaluation=tmp_path / "prior.json",
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
        assert payload["bindings"]["ranking_freeze_sha256"] == "b" * 64
        assert payload["bindings"]["review_freeze_sha256"] == "c" * 64
    assert result["artifacts"]["decision.json"]["decision"]["outcome"] == (
        "B_promotes_coverage"
    )
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
    assert result["artifacts"] == module.build_evaluation_artifacts(
        frozen_inputs,
        result["qrels"],
        bindings=result["artifacts"]["decision.json"]["bindings"],
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
