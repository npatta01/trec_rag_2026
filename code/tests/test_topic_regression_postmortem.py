import copy

import pytest

from trec_rag.topic_regression_postmortem import (
    EXPECTED_WARM_DOCUMENT_ROWS_BY_TOPIC,
    EXPECTED_WARM_WINDOW_ROWS_BY_TOPIC,
    _runtime_summary,
    _validate_focus_regressions,
    build_component_records,
    build_counterfactual_rankings,
    classify_regression,
    validate_runtime_status_for_candidate,
    validate_warm_cache_status,
)


def _stage_row(docid, rank, score, *, component=None):
    provenance = [{"source_rank": rank, "source_score": 1.0}]
    if component is not None:
        provenance.append(
            {
                "ranker": "coverage_aware_long_doc_aggregate",
                "candidate_depth": 2,
                **component,
            }
        )
    return {
        "topic_id": "1",
        "docid": docid,
        "rank": rank,
        "score": score,
        "text": f"Text for {docid}",
        "provenance": provenance,
    }


FORMULA = {
    "long_document_weight": 0.5,
    "strongest_passage_weight": 0.5,
    "coverage_bonus_weight": 0.25,
}


def test_component_counterfactuals_validate_formula_and_use_bm25_tie_break():
    baseline = [_stage_row("a", 1, 10.0), _stage_row("b", 2, 9.0)]
    candidate = [
        _stage_row(
            "b",
            1,
            3.5,
            component={
                "base_rank": 2,
                "long_document_relevance": 4.0,
                "strongest_passage_relevance": 2.0,
                "bounded_coverage_support": 2,
            },
        ),
        _stage_row(
            "a",
            2,
            3.25,
            component={
                "base_rank": 1,
                "long_document_relevance": 2.0,
                "strongest_passage_relevance": 4.0,
                "bounded_coverage_support": 1,
            },
        ),
    ]

    records = build_component_records(
        baseline,
        candidate,
        candidate_depth=2,
        formula=FORMULA,
    )
    rankings = build_counterfactual_rankings(records)

    assert [row.docid for row in rankings["document"]] == ["b", "a"]
    assert [row.docid for row in rankings["passage"]] == ["a", "b"]
    assert [row.docid for row in rankings["no_coverage"]] == ["a", "b"]
    assert [row.score for row in rankings["no_coverage"]] == [3.0, 3.0]


def test_component_counterfactuals_reject_formula_mismatch():
    baseline = [_stage_row("a", 1, 10.0), _stage_row("b", 2, 9.0)]
    component = {
        "base_rank": 1,
        "long_document_relevance": 2.0,
        "strongest_passage_relevance": 4.0,
        "bounded_coverage_support": 1,
    }
    candidate = [
        _stage_row("a", 1, 99.0, component=component),
        _stage_row(
            "b",
            2,
            3.5,
            component={
                "base_rank": 2,
                "long_document_relevance": 4.0,
                "strongest_passage_relevance": 2.0,
                "bounded_coverage_support": 2,
            },
        ),
    ]

    with pytest.raises(ValueError, match="does not match reconstructed formula"):
        build_component_records(
            baseline,
            candidate,
            candidate_depth=2,
            formula=FORMULA,
        )


def test_regression_classification_distinguishes_signal_and_aggregation():
    aggregation = classify_regression(
        baseline_ndcg=0.8,
        shipped_ndcg=0.7,
        counterfactual_ndcgs={
            "document": 0.75,
            "passage": 0.85,
            "no_coverage": 0.78,
        },
    )
    signal = classify_regression(
        baseline_ndcg=0.8,
        shipped_ndcg=0.7,
        counterfactual_ndcgs={
            "document": 0.75,
            "passage": 0.76,
            "no_coverage": 0.74,
        },
    )

    assert aggregation["classification"] == "aggregation_sensitive"
    assert aggregation["best_counterfactual"] == "passage"
    assert signal["classification"] == "model_signal_limited"


def test_runtime_summary_recognizes_completed_modal_status_keys():
    summary = _runtime_summary(
        {},
        {},
        {
            "state": "completed",
            "gpu": "NVIDIA A100-SXM4-80GB",
            "modal_cloud_provider": "AWS",
            "modal_region": "us-east-1",
        },
    )

    assert summary["is_completed_modal"] is True
    assert summary["hardware"] == "NVIDIA A100-SXM4-80GB"
    assert summary["provider"] == "Modal / AWS / us-east-1"
    assert summary["modal_region"] == "us-east-1"


def _warm_cache_status():
    return {
        "state": "completed",
        "verification_id": "warm-1",
        "app_name": "warm-verifier",
        "volume_name": "scores",
        "score_cache_unchanged": True,
        "semantic_equal_to_canonical": True,
        "model_required": {"document": False, "window": False},
        "model_scores": {"document": 0, "window": 0},
        "source_status_sha256": "e" * 64,
        "canonical_document_sha256": "c" * 64,
        "canonical_window_sha256": "d" * 64,
        "regenerated_document": {
            "rows": 22_000,
            "rows_by_topic": EXPECTED_WARM_DOCUMENT_ROWS_BY_TOPIC,
            "all_rows_matched_modal_cache": True,
            "sha256": "a" * 64,
        },
        "regenerated_window": {
            "rows": 227_156,
            "rows_by_topic": EXPECTED_WARM_WINDOW_ROWS_BY_TOPIC,
            "all_rows_matched_modal_cache": True,
            "sha256": "b" * 64,
        },
    }


def _set_nested(payload, path, value):
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_warm_cache_status_requires_complete_zero_model_rematerialization():
    validated = validate_warm_cache_status(
        _warm_cache_status(),
        expected_artifact_sha256={"document": "c" * 64, "window": "d" * 64},
        expected_runtime_status_sha256="e" * 64,
        expected_volume_name="scores",
    )

    assert validated["state"] == "completed"
    assert validated["zero_model_calls"] is True
    assert validated["document"]["rows"] == 22_000
    assert validated["window"]["rows"] == 227_156
    assert validated["semantic_equal_to_canonical"] is True
    assert validated["document"]["canonical_sha256"] == "c" * 64


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("state",), "running", "state='completed'"),
        (("score_cache_unchanged",), False, "score_cache_unchanged"),
        (("semantic_equal_to_canonical",), False, "semantic_equal_to_canonical"),
        (("model_required", "document"), True, "models as unnecessary"),
        (("model_scores", "window"), 1, "zero document and window"),
        (("regenerated_document", "rows"), 21_999, "row count must be 22000"),
        (
            ("regenerated_window", "all_rows_matched_modal_cache"),
            False,
            "must match the Modal schema-v2 cache",
        ),
        (
            ("regenerated_document", "rows_by_topic", "14"),
            999,
            "must exactly match the 22-topic RAG25 population",
        ),
    ],
)
def test_warm_cache_status_rejects_incomplete_proof(path, value, match):
    status = copy.deepcopy(_warm_cache_status())
    _set_nested(status, path, value)

    with pytest.raises(ValueError, match=match):
        validate_warm_cache_status(status)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (
            ("canonical_document_sha256",),
            "f" * 64,
            "canonical document SHA-256 does not match",
        ),
        (
            ("source_status_sha256",),
            "f" * 64,
            "does not match the supplied runtime status",
        ),
        (
            ("volume_name",),
            "wrong-volume",
            "does not match the completed scoring runtime",
        ),
    ],
)
def test_warm_cache_status_rejects_unrelated_artifact_or_runtime_proof(
    path, value, match
):
    status = copy.deepcopy(_warm_cache_status())
    _set_nested(status, path, value)

    with pytest.raises(ValueError, match=match):
        validate_warm_cache_status(
            status,
            expected_artifact_sha256={
                "document": "c" * 64,
                "window": "d" * 64,
            },
            expected_runtime_status_sha256="e" * 64,
            expected_volume_name="scores",
        )


def test_runtime_status_must_match_candidate_artifact_digests_and_full_counts():
    payload = {
        "state": "completed",
        "volume_name": "scores",
        "document_rows": 22_000,
        "window_rows": 227_156,
        "document_sha256": "c" * 64,
        "window_sha256": "d" * 64,
    }

    validated = validate_runtime_status_for_candidate(
        payload,
        expected_artifact_sha256={"document": "c" * 64, "window": "d" * 64},
    )

    assert validated["artifact_sha256"] == {
        "document": "c" * 64,
        "window": "d" * 64,
    }

    payload["window_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="window artifact digest does not match"):
        validate_runtime_status_for_candidate(
            payload,
            expected_artifact_sha256={
                "document": "c" * 64,
                "window": "d" * 64,
            },
        )


def test_legacy_runtime_volume_can_be_bound_by_exact_warm_proof():
    payload = {
        "state": "completed",
        "document_rows": 22_000,
        "window_rows": 227_156,
        "document_sha256": "c" * 64,
        "window_sha256": "d" * 64,
    }

    validated = validate_runtime_status_for_candidate(
        payload,
        expected_artifact_sha256={"document": "c" * 64, "window": "d" * 64},
        expected_volume_name="legacy-score-volume",
    )

    assert validated["volume_name"] == "legacy-score-volume"

    payload["volume_name"] = "different-volume"
    with pytest.raises(ValueError, match="does not match the bound warm-cache proof"):
        validate_runtime_status_for_candidate(
            payload,
            expected_artifact_sha256={
                "document": "c" * 64,
                "window": "d" * 64,
            },
            expected_volume_name="legacy-score-volume",
        )


def test_changed_outcome_rejects_focus_topic_that_is_no_longer_a_regression():
    regressions = [
        {
            "topic_id": "224",
            "diagnostic": {"classification": "model_signal_limited"},
        }
    ]

    with pytest.raises(ValueError, match="not regressions.*515"):
        _validate_focus_regressions(["224", "515"], regressions)
