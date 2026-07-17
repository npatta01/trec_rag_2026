from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trec_rag.topic_failure_postmortem import (
    _attribution,
    _verify_root_seal,
    analyze_boundary_changes,
    compute_capture_diagnostic,
    eligible_documents_for_facet_cap,
    main,
    render_postmortem,
    replay_recovery_arms,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
POSTMORTEM_PATH = (
    REPO_ROOT
    / "reports/experiments/all_topic_tethered_facet_validation_v1/postmortem.json"
)


def test_capture_diagnostic_derives_binary_and_graded_totals_from_qrels() -> None:
    result = compute_capture_diagnostic(
        {
            "31": ["a", "b", "c"],
            "300": ["x", "y", "z"],
        },
        {
            "31": {"a": 4, "b": 2, "c": 1, "outside": 3},
            "300": {"x": 3, "y": 0, "outside": 2},
        },
        depth=2,
    )

    assert result == {
        "arm": "RRF",
        "depth": 2,
        "metric_source": "postmortem diagnostic",
        "known_relevant_count": 3,
        "known_relevant_total": 5,
        "binary_recall": pytest.approx(0.6),
        "graded_gain": 25,
        "graded_gain_total": 35,
        "graded_recall": pytest.approx(25 / 35),
        "judged_count": 4,
        "retrieved_count": 4,
        "judged_rate": pytest.approx(1.0),
    }


def test_boundary_attribution_separates_retrieval_streams_from_dual_coverage() -> None:
    result = _attribution(
        {"relevant", "below", "unknown"},
        {
            "relevant": [
                {"stream_id": "original", "stream_rank": 400},
                {"stream_id": "300-strategies", "stream_rank": 120},
            ],
            "below": [
                {"stream_id": "300-global-measures", "stream_rank": 50},
            ],
            "unknown": [{"stream_id": "original", "stream_rank": 600}],
        },
        {
            "relevant": "300-strategies",
            "below": "300-strategies",
            "unknown": "300-strategies",
        },
        {"relevant": 2, "below": 1},
        {
            "300-strategies": {
                "facet_label": "strategies",
                "query_formulation": "effective strategies prevent reduce",
            },
            "300-global-measures": {
                "facet_label": "global measures",
                "query_formulation": "global measures address climate change",
            },
        },
    )

    assert result["facet_retrieval_stream_memberships"] == {
        "300-global-measures": {
            "facet_label": "global measures",
            "query_formulation": "global measures address climate change",
            "total": 1,
            "known_relevant": 0,
            "judged_below_2": 1,
            "unjudged": 0,
            "rank_buckets": {"1-50": 1},
        },
        "300-strategies": {
            "facet_label": "strategies",
            "query_formulation": "effective strategies prevent reduce",
            "total": 1,
            "known_relevant": 1,
            "judged_below_2": 0,
            "unjudged": 0,
            "rank_buckets": {"101-150": 1},
        },
    }
    assert result["dual_selection_coverage_facets"] == {"300-strategies": 3}
    assert "primary_selection_coverage_facet" not in result


def test_tracked_postmortem_contains_authenticated_depth_20_capture() -> None:
    postmortem = json.loads(POSTMORTEM_PATH.read_text())
    capture = postmortem["rrf_capture_diagnostic_at_20"]

    assert capture["known_relevant_count"] == 316
    assert capture["known_relevant_total"] == 12984
    assert capture["binary_recall"] == pytest.approx(316 / 12984)
    assert capture["graded_gain"] == 2492
    assert capture["graded_gain_total"] == 84560
    assert capture["graded_recall"] == pytest.approx(2492 / 84560)
    assert capture["metric_source"] == "postmortem diagnostic"
    assert postmortem["provenance"]["qrels_sha256"] == (
        "42bf933ae06eb22213312b22e3f2bc39f3dcc2d54e87ebcd8125e9528ddfcc37"
    )


def test_tracked_topic_300_attribution_names_actual_stream_formulations() -> None:
    postmortem = json.loads(POSTMORTEM_PATH.read_text())
    attribution = postmortem["topics"]["300"]["primary_boundary_attribution"]
    incoming = attribution["incoming"]
    streams = incoming["facet_retrieval_stream_memberships"]

    assert {stream_id: row["total"] for stream_id, row in streams.items()} == {
        "300-antarctica": 35,
        "300-economic-costs": 32,
        "300-global-measures": 51,
        "300-strategies": 54,
    }
    assert streams["300-strategies"] == {
        "facet_label": "strategies",
        "query_formulation": "global warming climate change effective strategies prevent reduce",
        "total": 54,
        "known_relevant": 1,
        "judged_below_2": 0,
        "unjudged": 53,
        "rank_buckets": {"101-150": 27, "151-200": 27},
    }
    assert incoming["dual_selection_coverage_facets"]["300-strategies"] == 276
    assert "primary_selection_coverage_facet" not in incoming


def test_boundary_analysis_keeps_unjudged_separate() -> None:
    result = analyze_boundary_changes(
        baseline=["a", "b", "c", "d"],
        candidate=["a", "x", "y", "d"],
        qrels={"a": 4, "b": 2, "c": 1, "x": 2},
        depth=3,
    )
    assert result["outgoing"] == {
        "total": 2,
        "known_relevant": 1,
        "judged_below_2": 1,
        "unjudged": 0,
    }
    assert result["incoming"] == {
        "total": 2,
        "known_relevant": 1,
        "judged_below_2": 0,
        "unjudged": 1,
    }


def test_rank_cap_keeps_original_candidates_and_rejects_deep_facet_only() -> None:
    provenance = {
        "original": {"original_rank": 900, "facet_ranks": []},
        "shallow": {"original_rank": None, "facet_ranks": [87, 130]},
        "deep": {"original_rank": None, "facet_ranks": [101, 140]},
    }
    assert eligible_documents_for_facet_cap(provenance, 100) == {"original", "shallow"}


def test_replay_defers_deep_facet_only_documents_in_canonical_rrf_order() -> None:
    documents = ["a", "b", "original", "shallow", "deep"]
    topic_input = {
        "topic_id": "300",
        "docids": documents,
        "texts": {document: f"text for {document}" for document in documents},
        "original_rank": {"a": 1, "b": 2, "original": 900},
        "facets": [
            {
                "facet_id": "300-test",
                "manifest_order": 0,
                "scores": {"shallow": 1.0, "deep": 0.5},
                "bm25_rank": {"shallow": 87, "deep": 101},
            }
        ],
        "common_scores": {document: float(index) for index, document in enumerate(documents)},
        "narrative_scores": {
            document: float(len(documents) - index)
            for index, document in enumerate(documents)
        },
        "rankings": {
            "RRF": ["a", "b", "original", "deep", "shallow"],
            "RRF100-STATIC-DUAL": ["a", "b", "deep", "shallow", "original"],
            "RRF100-STATIC-DUAL-NR": ["a", "b", "deep", "original", "shallow"],
        },
        "fixed_objectives": {
            "RRF100-STATIC-DUAL-NR": {"deep": 0.9, "original": 0.7, "shallow": 0.5}
        },
        "protected_prefix_depth": 2,
    }

    result = replay_recovery_arms(topic_input, {"a": 4, "deep": 2, "shallow": 1})

    rankings = result["rankings"]
    assert rankings["facet_rank_cap_100"] == ["a", "b", "shallow", "original", "deep"]
    assert all(len(order) == 5 and set(order) == set(documents) for order in rankings.values())
    assert result["no_narrative_score"]["status"] == "post-hoc diagnostic"
    assert result["no_narrative_score"]["promotion_eligible"] is False
    assert result["no_narrative_score"]["base_arm"] == "RRF100-STATIC-DUAL-NR"
    assert result["no_narrative_score"]["method"] == "fixed objective minus 0.15*N; no greedy replay"


def test_markdown_labels_unknown_judgments_and_diagnostic_only_ablation() -> None:
    boundary = {
        "known_relevant_delta": -3,
        "outgoing": {"total": 4, "known_relevant": 3, "judged_below_2": 0, "unjudged": 1},
        "incoming": {"total": 4, "known_relevant": 0, "judged_below_2": 1, "unjudged": 3},
    }
    analysis = {
        "schema_version": "topic-failure-postmortem-v1",
        "provenance": {
            "ranking_root_sha256": "a" * 64,
            "retrieval_root_sha256": "b" * 64,
            "scoring_root_sha256": "c" * 64,
            "qrels_sha256": "d" * 64,
        },
        "rrf_capture_diagnostic_at_20": {
            "known_relevant_count": 3,
            "known_relevant_total": 100,
            "binary_recall": 0.03,
            "graded_gain": 21,
            "graded_gain_total": 500,
            "graded_recall": 0.042,
        },
        "topics": {
            "31": {"primary_at_1000": boundary},
            "300": {
                "primary_at_1000": boundary,
                "primary_boundary_attribution": {
                    "outgoing": {
                        "facet_retrieval_stream_memberships": {},
                        "dual_selection_coverage_facets": {"300-strategies": 4},
                    },
                    "incoming": {
                        "facet_retrieval_stream_memberships": {},
                        "dual_selection_coverage_facets": {"300-strategies": 4},
                    },
                },
                "facet_bucket_yield": {"1-50": 0.3, "51-100": 0.2, "101-150": 0.1, "151-200": 0.05},
                "recovery_replay": {
                    "protected_prefix_depth": 100,
                    "permutation_checks": {"facet_rank_cap_100": True},
                    "arms": {
                        "facet_rank_cap_100": {"known_relevant_delta": 1},
                        "no_narrative_score": {
                            "known_relevant_delta": 0,
                            "status": "post-hoc diagnostic",
                            "promotion_eligible": False,
                            "base_arm": "RRF100-STATIC-DUAL-NR",
                            "method": "fixed objective minus 0.15*N; no greedy replay",
                            "redundancy_state": "fixed zero inherited from diagnostic base arm",
                        },
                    },
                },
            },
        },
    }

    markdown = render_postmortem(analysis)

    assert "unjudged (unknown)" in markdown
    assert "Judgment-pool dependent" in markdown
    assert "post-hoc diagnostic" in markdown
    assert "not promotion-eligible" in markdown
    assert "fixed objective minus 0.15*N; no greedy replay" in markdown
    assert "Protected RRF prefix: 100" in markdown
    assert "3 / 100 known-relevant" in markdown
    assert "Facet retrieval-stream attribution (not DUAL selection coverage)" in markdown
    assert "DUAL selection-coverage attribution (not retrieval-stream provenance)" in markdown


def test_cli_requires_every_explicit_source_and_output_path() -> None:
    with pytest.raises(SystemExit):
        main(["--source-root", "/tmp/source", "--qrels", "/tmp/qrels"])


def test_planning_root_seal_uses_its_newline_terminated_canonical_form(tmp_path) -> None:
    files = {"manifest.json": {"bytes": 2, "sha256": "a" * 64}}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")) + "\n"
    expected_root = hashlib.sha256(canonical.encode()).hexdigest()
    seal_path = tmp_path / "SEALED.json"
    content = json.dumps({"files": files, "root_sha256": expected_root}).encode()
    seal_path.write_bytes(content)

    result = _verify_root_seal(
        seal_path,
        expected_root,
        "planning seal",
        authenticated_binding={
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        canonical_trailing_newline=True,
    )

    assert result["root_sha256"] == expected_root


@pytest.mark.parametrize("forgery", ["metadata", "extra", "missing"])
def test_authenticated_seal_binding_rejects_forged_contract_keys(
    tmp_path, forgery: str
) -> None:
    files = {"artifact.json": {"bytes": 2, "sha256": "a" * 64}}
    root = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    canonical = {
        "schema_version": "canonical-seal-v1",
        "experiment_id": "canonical-experiment",
        "files": files,
        "root_sha256": root,
    }
    authenticated = json.dumps(canonical, sort_keys=True).encode()
    binding = {
        "bytes": len(authenticated),
        "sha256": hashlib.sha256(authenticated).hexdigest(),
    }
    forged = dict(canonical)
    if forgery == "metadata":
        forged["schema_version"] = "forged-seal"
    elif forgery == "extra":
        forged["untrusted_extra"] = True
    else:
        del forged["experiment_id"]
    seal_path = tmp_path / "SEALED.json"
    seal_path.write_text(json.dumps(forged, sort_keys=True))

    with pytest.raises(ValueError, match="authenticated.*binding"):
        _verify_root_seal(
            seal_path,
            root,
            "upstream seal",
            authenticated_binding=binding,
        )
