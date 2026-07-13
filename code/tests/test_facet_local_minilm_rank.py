from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from trec_rag.facet_local_minilm_rank import (
    ARM_NAMES,
    aggregate_maxp,
    aggregate_top4,
    build_arm_streams,
    build_control_diff,
    build_stream_rankings,
    build_weight_tables,
    canonical_ranking_sha256,
    fuse_streams,
    generate_freeze,
    load_authenticated_inputs,
    reconstruct_legacy_ranking,
    rerank_stream,
    verify_freeze,
)


ROOT = Path(__file__).resolve().parents[2]


def test_top4_span_distinct_aggregation_uses_frozen_weights():
    rows = [
        {
            "score": score,
            "document_start_token": index * 256,
            "document_end_token": index * 256 + 256,
            "window_id": f"w{index}",
        }
        for index, score in enumerate((9.0, 8.0, 7.0, 6.0))
    ]

    assert aggregate_top4(rows) == pytest.approx(
        0.55 * 9.0 + 0.25 * 8.0 + 0.13 * 7.0 + 0.07 * 6.0
    )


def test_top4_requires_128_new_tokens_and_renormalizes_retained_weights():
    rows = [
        {
            "score": 9.0,
            "document_start_token": 0,
            "document_end_token": 256,
            "window_id": "best",
        },
        {
            "score": 8.0,
            "document_start_token": 128,
            "document_end_token": 384,
            "window_id": "exactly-128-new",
        },
        {
            "score": 7.0,
            "document_start_token": 257,
            "document_end_token": 384,
            "window_id": "only-one-new",
        },
    ]

    assert aggregate_top4(rows) == pytest.approx((0.55 * 9.0 + 0.25 * 8.0) / 0.8)


def test_maxp_is_an_independent_aggregation_sensitivity():
    rows = [
        {
            "score": score,
            "document_start_token": index * 256,
            "document_end_token": index * 256 + 256,
            "window_id": f"w{index}",
        }
        for index, score in enumerate((4.0, 9.0, 7.0))
    ]

    assert aggregate_maxp(rows) == 9.0
    assert aggregate_maxp(rows) != aggregate_top4(rows)


def test_stream_reranking_ties_by_prior_rank_then_docid_and_is_order_stable():
    candidates = [
        {
            "topic_id": "200",
            "variant": "facet:f01",
            "family": "facet",
            "document_id": docid,
            "rank": prior_rank,
            "query": "facet query",
            "source_score": 100.0 - prior_rank,
            "text": f"text for {docid}",
        }
        for docid, prior_rank in (("doc-z", 1), ("doc-b", 2), ("doc-a", 2))
    ]
    windows = [
        {
            "topic_id": "200",
            "variant": "facet:f01",
            "document_id": row["document_id"],
            "score": 5.0,
            "document_start_token": 0,
            "document_end_token": 256,
            "window_id": f"window-{row['document_id']}",
            "window_text": f"passage for {row['document_id']}",
        }
        for row in candidates
    ]

    expected = ["doc-z", "doc-a", "doc-b"]
    first = rerank_stream(candidates, windows, aggregation="top4")
    second = rerank_stream(
        list(reversed(candidates)), list(reversed(windows)), aggregation="top4"
    )

    assert [row["document_id"] for row in first] == expected
    assert [row["rank"] for row in first] == [1, 2, 3]
    assert first == second


def test_topic_local_weights_are_qualified_exact_and_order_invariant():
    retriever = "pyserini_remote_raw_first_v1"
    streams = [
        {"topic_id": "200", "variant": "original", "family": "original"},
        {"topic_id": "200", "variant": "facet:shared", "family": "facet"},
        {"topic_id": "225", "variant": "original", "family": "original"},
        {"topic_id": "225", "variant": "facet:shared", "family": "facet"},
        {"topic_id": "225", "variant": "facet:other", "family": "facet"},
    ]

    legacy, corrected = build_weight_tables(streams, retriever_name=retriever)
    reordered = build_weight_tables(
        list(reversed(streams)), retriever_name=retriever
    )

    assert (legacy, corrected) == reordered
    assert corrected[("200", "original", retriever)] == Fraction(1, 2)
    assert corrected[("200", "facet:shared", retriever)] == Fraction(1, 2)
    assert corrected[("225", "original", retriever)] == Fraction(1, 2)
    assert corrected[("225", "facet:shared", retriever)] == Fraction(1, 4)
    assert corrected[("225", "facet:other", retriever)] == Fraction(1, 4)
    assert {
        topic_id: sum(
            weight for (topic, _variant, _retriever), weight in corrected.items()
            if topic == topic_id
        )
        for topic_id in ("200", "225")
    } == {"200": Fraction(1, 1), "225": Fraction(1, 1)}
    # The historical global-key table retained the last topic's shared weight.
    assert legacy[("facet:shared", retriever)] == Fraction(1, 4)


def test_exact_arm_matrix_preserves_streams_and_truncates_facets_only():
    retriever = "pyserini_remote_raw_first_v1"
    facet_counts = {"200": 9, "225": 7, "707": 3, "897": 8}
    streams = {}
    for topic_id, facet_count in facet_counts.items():
        identities = [
            (topic_id, "prompt_lab_v1:original", retriever, "original"),
            *(
                (topic_id, f"facet:{index:02d}", retriever, "facet")
                for index in range(facet_count)
            ),
        ]
        for topic, variant, retriever_name, family in identities:
            bm25 = tuple(
                {"document_id": f"{topic}-{variant}-{rank:03d}", "rank": rank}
                for rank in range(1, 101)
            )
            top4 = tuple(reversed(bm25))
            maxp = tuple(bm25[1:] + bm25[:1])
            streams[(topic, variant, retriever_name)] = {
                "family": family,
                "bm25": bm25,
                "top4": top4,
                "maxp": maxp,
            }

    matrix = build_arm_streams(streams)

    assert set(ARM_NAMES) == {
        "R1_LEGACY",
        "C0_TOPIC_LOCAL",
        "BF100_TOPIC_LOCAL",
        "BF50_TOPIC_LOCAL",
        "BF20_TOPIC_LOCAL",
        "BO100_TOPIC_LOCAL",
        "BB100_TOPIC_LOCAL",
        "BF100_MAXP_TOPIC_LOCAL",
        "BF100_LEGACY_FUSION",
    }
    assert set(matrix) == set(ARM_NAMES) - {"R1_LEGACY"}
    assert len(matrix["BF100_TOPIC_LOCAL"]) == 31
    for identity, stream in streams.items():
        family = stream["family"]
        assert len(matrix["BF100_TOPIC_LOCAL"][identity]) == 100
        assert len(matrix["BF50_TOPIC_LOCAL"][identity]) == (
            100 if family == "original" else 50
        )
        assert len(matrix["BF20_TOPIC_LOCAL"][identity]) == (
            100 if family == "original" else 20
        )
        if family == "original":
            assert matrix["BF100_TOPIC_LOCAL"][identity] == stream["bm25"]
            assert matrix["BO100_TOPIC_LOCAL"][identity] == stream["top4"]
        else:
            assert matrix["BF100_TOPIC_LOCAL"][identity] == stream["top4"]
            assert matrix["BO100_TOPIC_LOCAL"][identity] == stream["bm25"]
            assert matrix["BF100_MAXP_TOPIC_LOCAL"][identity] == stream["maxp"]
        assert matrix["BB100_TOPIC_LOCAL"][identity] == stream["top4"]
    assert matrix["BF100_LEGACY_FUSION"] == matrix["BF100_TOPIC_LOCAL"]


def test_rrf_is_deterministic_depth_100_and_never_compares_cross_stream_scores():
    retriever = "pyserini_remote_raw_first_v1"
    original = ("200", "original", retriever)
    facet = ("200", "facet:f01", retriever)

    def rows(identity, score_offset):
        topic_id, variant, _ = identity
        return tuple(
            {
                "topic_id": topic_id,
                "variant": variant,
                "family": "original" if variant == "original" else "facet",
                "document_id": f"doc-{rank:03d}",
                "rank": rank,
                "prior_rank": 101 - rank,
                "score": score_offset + rank,
                "source_score": score_offset - rank,
                "query": f"query for {variant}",
                "text": f"text chosen from {variant} at {rank}",
                "aggregation": "bm25" if variant == "original" else "top4",
                "passage": f"passage {variant} {rank}",
                "selected_windows": [],
            }
            for rank in range(1, 101)
        )

    streams = {original: rows(original, 10_000), facet: rows(facet, -10_000)}
    weights = {original: Fraction(1, 2), facet: Fraction(1, 2)}
    first = fuse_streams(streams, weights)
    reordered = fuse_streams(
        {facet: tuple(reversed(streams[facet])), original: tuple(reversed(streams[original]))},
        weights,
    )
    score_changed = {
        identity: tuple(
            {**row, "score": -float(row["score"]), "source_score": 1e30}
            for row in stream
        )
        for identity, stream in streams.items()
    }
    changed = fuse_streams(score_changed, weights)

    assert len(first) == 100
    assert [row["rank"] for row in first] == list(range(1, 101))
    assert first == reordered
    assert [(row["docid"], row["rank"], row["score"], row["text"]) for row in first] == [
        (row["docid"], row["rank"], row["score"], row["text"]) for row in changed
    ]
    assert [entry["variant_name"] for entry in first[0]["provenance"]] == [
        "facet:f01",
        "original",
    ]
    assert sum(entry["rrf_contribution"] for entry in first[0]["provenance"]) == pytest.approx(
        first[0]["score"]
    )


def test_rrf_rejects_protected_topics_before_fusion():
    identity = ("144", "original", "retriever")
    with pytest.raises(ValueError, match="protected topic 144"):
        fuse_streams(
            {
                identity: (
                    {
                        "topic_id": "144",
                        "variant": "original",
                        "document_id": "forbidden",
                        "rank": 1,
                        "score": 1.0,
                    },
                )
            },
            {identity: Fraction(1, 1)},
        )


def test_actual_legacy_ranking_reconstructs_canonical_hash_and_projection():
    manifest = json.loads(
        (ROOT / "reports/experiments/facet_local_minilm_pilot_v1/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_path = (
        ROOT
        / "outputs/rag25_facet_local_minilm_v1/source_v1"
        / manifest["candidate_file"]
    )
    candidates = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
    ]
    legacy_path = (
        ROOT
        / "outputs/rag25_sparse_relevance_paired_v1/freeze_v1/rankings/R1__family_rrf.jsonl"
    )
    historical = [json.loads(line) for line in legacy_path.read_text(encoding="utf-8").splitlines()]
    historical_freeze = json.loads(
        (
            ROOT / "outputs/rag25_sparse_relevance_paired_v1/freeze_v1/freeze.json"
        ).read_text(encoding="utf-8")
    )
    legacy_weights, _corrected = build_weight_tables(
        manifest["streams"], retriever_name="pyserini_remote_raw_first_v1"
    )

    reconstructed = reconstruct_legacy_ranking(candidates, legacy_weights)

    assert hashlib.sha256(legacy_path.read_bytes()).hexdigest() == (
        "1edd1542777ae45c819905b697fcec0242de5421b34e7531fbf0178395837bed"
    )
    assert canonical_ranking_sha256(reconstructed) == historical_freeze["rankings"][
        "R1:family_rrf"
    ]["sha256"]
    assert [
        (row["topic_id"], row["rank"], row["docid"]) for row in reconstructed
    ] == [(row["topic_id"], row["rank"], row["docid"]) for row in historical]


def test_control_diff_is_qrels_free_and_freezes_rank_and_set_changes():
    legacy = [
        {"topic_id": "200", "rank": 1, "docid": "shared"},
        {"topic_id": "200", "rank": 2, "docid": "legacy-only"},
    ]
    corrected = [
        {"topic_id": "200", "rank": 1, "docid": "corrected-only"},
        {"topic_id": "200", "rank": 2, "docid": "shared"},
    ]

    audit = build_control_diff(legacy, corrected)

    assert audit["qrels_opened"] is False
    assert audit["topics"]["200"]["legacy_only"] == ["legacy-only"]
    assert audit["topics"]["200"]["corrected_only"] == ["corrected-only"]
    assert audit["topics"]["200"]["rank_changes"] == [
        {
            "corrected_rank": 2,
            "document_id": "shared",
            "legacy_rank": 1,
            "rank_delta_corrected_minus_legacy": 1,
        }
    ]
    assert "qrels_path" not in json.dumps(audit, sort_keys=True)


def test_real_inputs_are_fully_authenticated_without_qrels_or_new_work():
    inputs = load_authenticated_inputs(
        manifest_path=ROOT
        / "reports/experiments/facet_local_minilm_pilot_v1/manifest.json",
        preflight_dir=ROOT / "outputs/rag25_facet_local_minilm_v1/preflight_v2",
        scores_dir=ROOT / "outputs/rag25_facet_local_minilm_v1/full_scoring_v1",
    )

    assert len(inputs.candidates) == 3100
    assert len(inputs.windows) == 14720
    assert len(inputs.scores) == 14720
    assert inputs.bindings["scoring_receipt_sha256"] == (
        "d8d3f86d16bd25d0f92c682f3707273b08279489636481b18f694e9dd673df00"
    )
    assert inputs.bindings["qrels_opened"] is False
    assert inputs.bindings["retrieval_call_count"] == 0
    assert inputs.bindings["inference_call_count"] == 0


def test_real_stream_rankings_are_complete_and_feed_the_frozen_arm_matrix():
    inputs = load_authenticated_inputs(
        manifest_path=ROOT
        / "reports/experiments/facet_local_minilm_pilot_v1/manifest.json",
        preflight_dir=ROOT / "outputs/rag25_facet_local_minilm_v1/preflight_v2",
        scores_dir=ROOT / "outputs/rag25_facet_local_minilm_v1/full_scoring_v1",
    )

    streams = build_stream_rankings(inputs)
    matrix = build_arm_streams(streams)

    assert len(streams) == 31
    assert sum(record["family"] == "original" for record in streams.values()) == 4
    assert all(
        len(record[aggregation]) == 100
        for record in streams.values()
        for aggregation in ("bm25", "top4", "maxp")
    )
    assert all(
        [row["rank"] for row in record[aggregation]] == list(range(1, 101))
        for record in streams.values()
        for aggregation in ("bm25", "top4", "maxp")
    )
    assert any(
        [row["document_id"] for row in record["top4"]]
        != [row["document_id"] for row in record["bm25"]]
        for record in streams.values()
        if record["family"] == "facet"
    )
    assert all(
        matrix["BF100_TOPIC_LOCAL"][identity] == record["bm25"]
        for identity, record in streams.items()
        if record["family"] == "original"
    )


def test_real_freeze_is_create_only_self_contained_and_independently_replayable(
    tmp_path,
):
    output = tmp_path / "freeze_v1"
    payload = generate_freeze(
        manifest_path=ROOT
        / "reports/experiments/facet_local_minilm_pilot_v1/manifest.json",
        preflight_dir=ROOT / "outputs/rag25_facet_local_minilm_v1/preflight_v2",
        scores_dir=ROOT / "outputs/rag25_facet_local_minilm_v1/full_scoring_v1",
        output_dir=output,
    )

    assert payload["status"] == "frozen_before_qrels"
    assert payload["qrels_opened"] is False
    assert set(payload["rankings"]) == set(ARM_NAMES)
    assert all(record["rows"] == 400 for record in payload["rankings"].values())
    assert payload["rankings"]["R1_LEGACY"]["file_sha256"] == (
        "1edd1542777ae45c819905b697fcec0242de5421b34e7531fbf0178395837bed"
    )
    assert len(payload["streams"]) == 93
    assert payload["fusion_tables"] == {
        "family_rrf_global_key_v1": "fusion_weights_v1.json",
        "family_rrf_topic_local_v2": "fusion_weights_v2.json",
    }
    assert payload["control_diff_audit"] == "legacy_corrected_diff.json"
    assert "qrels_path" not in json.dumps(payload, sort_keys=True)

    verified = verify_freeze(output)
    assert verified["status"] == "verified"
    assert verified["ranking_count"] == 9
    assert verified["stream_ranking_count"] == 93
    assert verified["qrels_opened"] is False
    with pytest.raises(FileExistsError, match="create-only"):
        generate_freeze(
            manifest_path=ROOT
            / "reports/experiments/facet_local_minilm_pilot_v1/manifest.json",
            preflight_dir=ROOT / "outputs/rag25_facet_local_minilm_v1/preflight_v2",
            scores_dir=ROOT / "outputs/rag25_facet_local_minilm_v1/full_scoring_v1",
            output_dir=output,
        )


def test_verify_cli_dispatches_from_process_arguments(monkeypatch, tmp_path, capsys):
    import trec_rag.facet_local_minilm_rank as module

    freeze = tmp_path / "freeze"
    monkeypatch.setattr(sys, "argv", ["facet_local_minilm_rank", "verify", "--freeze", str(freeze)])
    monkeypatch.setattr(
        module,
        "verify_freeze",
        lambda path: {
            "status": "verified",
            "freeze": str(path),
        },
    )

    assert module.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
