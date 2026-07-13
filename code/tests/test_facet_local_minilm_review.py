from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from trec_rag.facet_local_minilm_review import (
    CONTROL_ARM,
    LABEL_VOCABULARY,
    MINILM_ARM,
    RUBRIC_TEXT,
    build_review_packet,
    create_review,
    freeze_labels,
    freeze_review,
    freeze_review_directory,
    unmask_review,
    verify_review_create,
    verify_review_freeze,
    write_review_create,
)


ROOT = Path(__file__).resolve().parents[2]
DURABLE_FREEZE = ROOT / "outputs/rag25_facet_local_minilm_v1/freeze_v1"
PILOT_TOPICS = ("200", "225", "707", "897")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stream_row(topic_id, variant, query, docid, rank, passage, score):
    window_text = f"frozen MiniLM passage: {passage}"
    return {
        "aggregation": "top4",
        "document_id": docid,
        "family": "facet",
        "passage": window_text,
        "prior_rank": rank,
        "query": query,
        "query_sha256": _sha256_text(query),
        "rank": rank,
        "score": score,
        "selected_windows": [
            {
                "document_end_token": 256,
                "document_start_token": 0,
                "score": score + 0.25,
                "window_id": _sha256_text(f"{topic_id}:{variant}:{docid}"),
                "window_text": window_text,
            }
        ],
        "topic_id": topic_id,
        "variant": variant,
    }


@pytest.fixture
def synthetic_freeze():
    facet_counts = {"200": 9, "225": 7, "707": 3, "897": 8}
    streams = []
    ordinal = 0
    for topic_id, facet_count in facet_counts.items():
        for facet_number in range(1, facet_count + 1):
            ordinal += 1
            variant = f"facet:{topic_id}:{facet_number:02d}"
            query = f"facet query {ordinal}"
            shared = f"shared-{ordinal}"
            lexical = f"lexical-{ordinal}"
            minilm = f"minilm-{ordinal}"
            tail = f"tail-{ordinal}"
            top4 = [
                _stream_row(topic_id, variant, query, shared, 1, "shared", 9.0),
                _stream_row(topic_id, variant, query, minilm, 2, "minilm", 8.0),
                _stream_row(topic_id, variant, query, lexical, 3, "lexical", 7.0),
                _stream_row(topic_id, variant, query, tail, 4, "tail", 6.0),
            ]
            bm25 = [
                {
                    **top4[0],
                    "aggregation": "bm25",
                    "passage": "BM25 passage must never be displayed",
                    "rank": 1,
                    "score": 30.0,
                    "selected_windows": [],
                },
                {
                    **top4[2],
                    "aggregation": "bm25",
                    "passage": "different BM25 passage must never be displayed",
                    "rank": 2,
                    "score": 29.0,
                    "selected_windows": [],
                },
                {
                    **top4[1],
                    "aggregation": "bm25",
                    "rank": 3,
                    "score": 28.0,
                    "selected_windows": [],
                },
                {
                    **top4[3],
                    "aggregation": "bm25",
                    "rank": 4,
                    "score": 27.0,
                    "selected_windows": [],
                },
            ]
            streams.append(
                {
                    "bm25": bm25,
                    "bm25_path": f"streams/bm25/{topic_id}-{ordinal}.jsonl",
                    "bm25_sha256": _sha256_text(f"bm25-{ordinal}"),
                    "family": "facet",
                    "retriever_name": "fixture-retriever",
                    "top4": top4,
                    "top4_path": f"streams/top4/{topic_id}-{ordinal}.jsonl",
                    "top4_sha256": _sha256_text(f"top4-{ordinal}"),
                    "topic_id": topic_id,
                    "variant_name": variant,
                }
            )
    return {
        "experiment_manifest_sha256": "b" * 64,
        "facet_streams": streams,
        "ranking_freeze_sha256": "a" * 64,
        "topic_ids": list(PILOT_TOPICS),
    }


@pytest.fixture
def packet_and_secret(synthetic_freeze):
    return build_review_packet(synthetic_freeze)


def _labels(packet, reviewer_id, *, overrides=None, flagged=()):
    overrides = overrides or {}
    flagged = set(flagged)
    return [
        {
            "item_id": row["item_id"],
            "low_quality": row["item_id"] in flagged,
            "relevance": overrides.get(row["item_id"], "direct_answer"),
            "reviewer_id": reviewer_id,
            "wrong_domain": False,
        }
        for row in packet
    ]


def _jsonl_bytes(rows):
    return b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
        for row in rows
    )


def test_review_packet_masks_system_rank_score_and_docid(packet_and_secret):
    packet, secret = packet_and_secret

    assert len(packet) == 81
    assert all(
        set(row) == {"item_id", "topic_id", "facet_query", "passage"}
        for row in packet
    )
    assert all(
        not ({"arm", "rank", "score", "docid", "document_id"} & set(row))
        for row in packet
    )
    assert secret["items"]


def test_top_two_pool_covers_all_27_facets_and_both_corrected_arms(packet_and_secret):
    _packet, secret = packet_and_secret
    memberships = [
        membership
        for item in secret["items"]
        for membership in item["memberships"]
    ]

    facets = {
        (membership["facet"]["topic_id"], membership["facet"]["variant_name"])
        for membership in memberships
    }
    assert len(facets) == 27
    assert {membership["arm"] for membership in memberships} == {
        CONTROL_ARM,
        MINILM_ARM,
    }
    assert len(memberships) == 108
    for facet in facets:
        for arm in (CONTROL_ARM, MINILM_ARM):
            ranks = {
                membership["source_rank"]
                for membership in memberships
                if (
                    membership["facet"]["topic_id"],
                    membership["facet"]["variant_name"],
                )
                == facet
                and membership["arm"] == arm
            }
            assert ranks == {1, 2}


def test_pool_deduplicates_within_facet_and_attributes_shared_items_twice(
    packet_and_secret,
):
    _packet, secret = packet_and_secret
    for item in secret["items"]:
        keys = {
            (
                membership["arm"],
                membership["facet"]["topic_id"],
                membership["facet"]["variant_name"],
            )
            for membership in item["memberships"]
        }
        assert len(keys) == len(item["memberships"])

    shared = [item for item in secret["items"] if len(item["memberships"]) == 2]
    assert len(shared) == 27
    assert all(
        {membership["arm"] for membership in item["memberships"]}
        == {CONTROL_ARM, MINILM_ARM}
        for item in shared
    )


def test_every_item_displays_the_same_highest_scoring_frozen_minilm_window(
    packet_and_secret,
):
    packet, secret = packet_and_secret
    packet_by_id = {row["item_id"]: row for row in packet}

    for item in secret["items"]:
        provenances = [
            membership["passage_selection_provenance"]
            for membership in item["memberships"]
        ]
        assert all(provenance == provenances[0] for provenance in provenances)
        assert provenances[0]["selection"] == "highest_raw_logit_window"
        assert packet_by_id[item["item_id"]]["passage"] == item["passage"]
        assert packet_by_id[item["item_id"]]["passage"] != (
            "BM25 passage must never be displayed"
        )
        assert _sha256_text(item["passage"]) == provenances[0]["passage_sha256"]


def test_secret_map_binds_each_membership_and_passage_provenance(packet_and_secret):
    _packet, secret = packet_and_secret

    for item in secret["items"]:
        for membership in item["memberships"]:
            assert set(membership) == {
                "arm",
                "document_id",
                "facet",
                "passage_selection_provenance",
                "source_rank",
            }
            provenance = membership["passage_selection_provenance"]
            assert {
                "aggregation",
                "document_end_token",
                "document_start_token",
                "minilm_source_rank",
                "passage_sha256",
                "query_sha256",
                "raw_score",
                "selection",
                "stream_path",
                "stream_sha256",
                "window_id",
            } == set(provenance)
            assert membership["document_id"] == item["document_id"]


def test_manifest_hash_shuffle_is_stable_under_input_reordering(synthetic_freeze):
    first_packet, first_secret = build_review_packet(synthetic_freeze)
    reordered = {
        **synthetic_freeze,
        "facet_streams": [
            {
                **stream,
                "bm25": list(reversed(stream["bm25"])),
                "top4": list(reversed(stream["top4"])),
            }
            for stream in reversed(synthetic_freeze["facet_streams"])
        ],
    }
    second_packet, second_secret = build_review_packet(reordered)

    assert first_packet == second_packet
    assert first_secret == second_secret
    assert first_packet != sorted(first_packet, key=lambda row: row["item_id"])


def test_packet_rejects_protected_topics_before_stream_join(synthetic_freeze):
    malicious = {
        **synthetic_freeze,
        "facet_streams": [{"topic_id": "144"}],
    }

    with pytest.raises(ValueError, match="protected topic 144"):
        build_review_packet(malicious)


def test_exact_label_vocabulary_and_schema_are_required(packet_and_secret):
    packet, _secret = packet_and_secret
    labels_a = _labels(packet, "reviewer-a")
    labels_b = _labels(packet, "reviewer-b")
    labels_a[0]["relevance"] = "relevant"

    assert LABEL_VOCABULARY == {
        "direct_answer",
        "partial_or_related",
        "not_facet_relevant",
    }
    with pytest.raises(ValueError, match="label vocabulary"):
        freeze_labels(packet, labels_a, labels_b, [])


def test_two_independent_reviewers_are_required(packet_and_secret):
    packet, _secret = packet_and_secret
    labels_a = _labels(packet, "same-reviewer")
    labels_b = _labels(packet, "same-reviewer")

    with pytest.raises(ValueError, match="independent reviewers"):
        freeze_labels(packet, labels_a, labels_b, [])


def test_disagreement_requires_third_label(packet_and_secret):
    packet, _secret = packet_and_secret
    item_id = packet[0]["item_id"]
    labels_a = _labels(packet, "reviewer-a")
    labels_b = _labels(
        packet,
        "reviewer-b",
        overrides={item_id: "partial_or_related"},
    )

    with pytest.raises(ValueError, match="unadjudicated disagreement"):
        freeze_review(packet, labels_a, labels_b, adjudication={})


def test_adjudicator_must_be_independent_and_cover_only_disagreements(
    packet_and_secret,
):
    packet, _secret = packet_and_secret
    item_id = packet[0]["item_id"]
    labels_a = _labels(packet, "reviewer-a")
    labels_b = _labels(
        packet,
        "reviewer-b",
        overrides={item_id: "partial_or_related"},
    )
    adjudication = [
        {
            "item_id": item_id,
            "relevance": "not_facet_relevant",
            "reviewer_id": "reviewer-a",
        }
    ]

    with pytest.raises(ValueError, match="third reviewer"):
        freeze_labels(packet, labels_a, labels_b, adjudication)


def test_labels_freeze_before_unmask_and_contains_no_system_information(
    packet_and_secret,
):
    packet, secret = packet_and_secret
    item_id = packet[0]["item_id"]
    labels_a = _labels(packet, "reviewer-a")
    labels_b = _labels(
        packet,
        "reviewer-b",
        overrides={item_id: "partial_or_related"},
    )
    adjudication = [
        {
            "item_id": item_id,
            "relevance": "not_facet_relevant",
            "reviewer_id": "reviewer-c",
        }
    ]

    label_freeze = freeze_labels(packet, labels_a, labels_b, adjudication)
    encoded = json.dumps(label_freeze, sort_keys=True)

    assert label_freeze["status"] == "labels_and_adjudication_frozen"
    assert CONTROL_ARM not in encoded
    assert MINILM_ARM not in encoded
    assert not any(item["document_id"] in encoded for item in secret["items"])


def test_unmask_refuses_anything_except_a_verified_label_freeze(packet_and_secret):
    _packet, secret = packet_and_secret

    with pytest.raises(ValueError, match="labels must be frozen"):
        unmask_review({"status": "awaiting_labels"}, secret)


def test_unmask_rejects_forged_status_and_items(packet_and_secret):
    packet, secret = packet_and_secret
    forged = {
        "status": "labels_and_adjudication_frozen",
        "schema_version": "facet-local-minilm-label-freeze-v1",
        "items": [
            {
                "item_id": row["item_id"],
                "low_quality": False,
                "relevance": "direct_answer",
                "wrong_domain": False,
            }
            for row in packet
        ],
        "label_freeze_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="label freeze"):
        unmask_review(forged, secret)


def test_label_freeze_binds_both_review_sources_and_adjudication(packet_and_secret):
    packet, _secret = packet_and_secret
    labels_a = _labels(packet, "reviewer-a")
    labels_b = _labels(packet, "reviewer-b")
    frozen = freeze_labels(packet, labels_a, labels_b, [])

    assert set(frozen["source_hashes"]) == {
        "adjudication_sha256",
        "labels_a_sha256",
        "labels_b_sha256",
    }
    assert all(len(value) == 64 for value in frozen["source_hashes"].values())


def test_label_freeze_rejects_protected_packet_before_label_index(packet_and_secret):
    packet, _secret = packet_and_secret
    malicious = [{**packet[0], "topic_id": "144"}, *packet[1:]]
    with pytest.raises(ValueError, match="protected topic 144"):
        freeze_labels(malicious, [], [], [])


def test_frozen_labels_unmask_shared_items_into_both_arm_denominators(
    packet_and_secret,
):
    packet, secret = packet_and_secret
    first_item = packet[0]["item_id"]
    labels_a = _labels(packet, "reviewer-a", flagged={first_item})
    labels_b = _labels(
        packet,
        "reviewer-b",
        overrides={first_item: "partial_or_related"},
    )
    adjudication = [
        {
            "item_id": first_item,
            "relevance": "partial_or_related",
            "reviewer_id": "reviewer-c",
        }
    ]

    frozen = freeze_review(
        packet,
        labels_a,
        labels_b,
        adjudication,
        secret=secret,
    )
    by_arm = {row["arm"]: row for row in frozen["aggregate_counts"]["by_arm"]}

    assert frozen["status"] == "review_frozen_before_qrels"
    assert by_arm[CONTROL_ARM]["denominator"] == 54
    assert by_arm[MINILM_ARM]["denominator"] == 54
    assert all(
        row["denominator"] == 2
        for row in frozen["aggregate_counts"]["by_arm_topic_facet"]
    )
    assert len(frozen["unmasked_items"]) == len(packet)


def test_rubric_judges_only_the_displayed_passage():
    lowered = RUBRIC_TEXT.casefold()

    assert "displayed passage" in lowered
    assert "unseen document" in lowered
    assert all(label in RUBRIC_TEXT for label in LABEL_VOCABULARY)


def test_create_stage_is_create_only_private_and_has_no_unmasked_public_output(
    tmp_path,
    packet_and_secret,
):
    packet, secret = packet_and_secret
    review = tmp_path / "review_v1"

    receipt = write_review_create(review, packet, secret)

    assert receipt["status"] == "awaiting_two_independent_reviews"
    assert {path.name for path in review.iterdir()} == {
        "create_receipt.json",
        "packet.jsonl",
        "rubric.md",
        "secret_map.json",
    }
    assert (review / "packet.jsonl").read_bytes() == _jsonl_bytes(packet)
    assert (review / "secret_map.json").stat().st_mode & 0o777 == 0o600
    public = (review / "packet.jsonl").read_text() + (review / "rubric.md").read_text()
    assert CONTROL_ARM not in public
    assert MINILM_ARM not in public
    assert not any(item["document_id"] in public for item in secret["items"])
    with pytest.raises(FileExistsError, match="create-only"):
        write_review_create(review, packet, secret)


def test_create_stage_verifier_detects_packet_tampering(tmp_path, packet_and_secret):
    packet, secret = packet_and_secret
    review = tmp_path / "review_v1"
    write_review_create(review, packet, secret)
    assert verify_review_create(review)["status"] == "verified_create_stage"
    with (review / "packet.jsonl").open("ab") as handle:
        handle.write(b"{}\n")

    with pytest.raises(ValueError, match="packet hash"):
        verify_review_create(review)


def test_create_verifier_rejects_semantic_secret_tampering_with_rehashed_receipt(
    tmp_path, packet_and_secret
):
    packet, secret = packet_and_secret
    review = tmp_path / "review_v1"
    write_review_create(review, packet, secret)
    secret_path = review / "secret_map.json"
    receipt_path = review / "create_receipt.json"
    tampered = json.loads(secret_path.read_text())
    tampered["items"][0]["topic_id"] = "144"
    secret_source = (json.dumps(tampered, indent=2, sort_keys=True) + "\n").encode()
    secret_path.write_bytes(secret_source)
    receipt = json.loads(receipt_path.read_text())
    receipt["secret_map_sha256"] = hashlib.sha256(secret_source).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="protected topic 144"):
        verify_review_create(review)


def test_fixture_freeze_directory_persists_bound_label_and_unmask_artifacts(
    tmp_path,
    packet_and_secret,
):
    packet, secret = packet_and_secret
    review = tmp_path / "review_v1"
    write_review_create(review, packet, secret)
    item_id = packet[0]["item_id"]
    labels_a = _labels(packet, "reviewer-a")
    labels_b = _labels(
        packet,
        "reviewer-b",
        overrides={item_id: "partial_or_related"},
    )
    adjudication = [
        {
            "item_id": item_id,
            "relevance": "partial_or_related",
            "reviewer_id": "reviewer-c",
        }
    ]
    labels_a_path = review / "labels_a.jsonl"
    labels_b_path = review / "labels_b.jsonl"
    adjudication_path = review / "adjudication.jsonl"
    labels_a_path.write_bytes(_jsonl_bytes(labels_a))
    labels_b_path.write_bytes(_jsonl_bytes(labels_b))
    adjudication_path.write_bytes(_jsonl_bytes(adjudication))

    result = freeze_review_directory(
        review,
        labels_a_path=labels_a_path,
        labels_b_path=labels_b_path,
        adjudication_path=adjudication_path,
    )

    assert result["status"] == "review_frozen_before_qrels"
    assert (review / "label_freeze.json").exists()
    assert (review / "review_freeze.json").exists()
    assert (review / "unmasked_items.jsonl").exists()
    assert verify_review_freeze(review)["status"] == "verified_review_freeze"
    stored = json.loads((review / "review_freeze.json").read_text())
    assert len(stored["review_freeze_sha256"]) == 64
    assert stored["bindings"]["ranking_freeze_sha256"] == secret[
        "ranking_freeze_sha256"
    ]


def test_real_ranking_freeze_builds_exact_create_stage_without_external_work(tmp_path):
    before = hashlib.sha256((DURABLE_FREEZE / "freeze.json").read_bytes()).hexdigest()
    review = tmp_path / "real_review_v1"

    receipt = create_review(DURABLE_FREEZE, review)
    verified = verify_review_create(review, ranking_freeze_dir=DURABLE_FREEZE)
    after = hashlib.sha256((DURABLE_FREEZE / "freeze.json").read_bytes()).hexdigest()

    assert receipt["ranking_freeze_sha256"] == (
        "f54401b3288eabade7efa75bc0c62f2181b1da44190a3b61c02e1e0fb670c6ed"
    )
    assert receipt["facet_count"] == 27
    assert receipt["membership_count"] == 108
    assert 54 <= receipt["item_count"] <= 108
    assert verified["packet_sha256"] == receipt["packet_sha256"]
    assert before == after
    assert receipt["qrels_opened"] is False
    assert receipt["retrieval_call_count"] == 0
    assert receipt["network_call_count"] == 0
    assert receipt["inference_call_count"] == 0


def test_create_cli_surface_has_no_qrels_retrieval_network_or_inference_option():
    source = (ROOT / "code/trec_rag/facet_local_minilm_review.py").read_text()

    assert "--qrels" not in source
    assert "--retrieval" not in source
    assert "--network" not in source
    assert "--inference" not in source
    assert "requests" not in source
    assert "urllib" not in source
