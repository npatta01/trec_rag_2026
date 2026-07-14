from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from trec_rag.deep_facet_candidate_score import (
    MAX_PHASE_SECONDS,
    aggregate_top4,
    build_phase_preflight,
    phase1_candidates,
    phase2_candidates,
    verify_source_receipt,
)


class _Tokenizer:
    def encode(self, text, *, add_special_tokens=False, truncation=False):
        return list(range(len(text.split())))

    def decode(
        self,
        tokens,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        return " ".join(f"t{token}" for token in tokens)

    def num_special_tokens_to_add(self, *, pair):
        return 3


def _manifest() -> dict[str, object]:
    return {
        "topic_ids": ["219"],
        "facets": [
            {
                "topic_id": "219",
                "facet_id": "219-positive",
                "query": "technology positive effects",
                "manifest_order": 0,
            }
        ],
        "qrels_opened": False,
    }


def _retrieval_rows() -> list[dict[str, object]]:
    return [
        {
            "topic_id": "219",
            "facet_id": "219-positive",
            "manifest_order": 0,
            "query_sha256": hashlib.sha256(
                b"technology positive effects"
            ).hexdigest(),
            "rank": rank,
            "docid": f"d{rank}",
            "score": 10.0 - rank,
            "text": "one two three four five",
        }
        for rank in range(1, 4)
    ]


def test_top4_uses_span_distinct_windows_and_renormalizes() -> None:
    rows = [
        {"score": 9.0, "document_start_token": 0, "document_end_token": 300, "window_id": "a"},
        {"score": 8.0, "document_start_token": 100, "document_end_token": 350, "window_id": "b"},
        {"score": 7.0, "document_start_token": 300, "document_end_token": 600, "window_id": "c"},
    ]
    assert aggregate_top4(rows) == pytest.approx((0.55 * 9 + 0.25 * 7) / 0.80)


def test_source_receipt_requires_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    path.write_bytes(b"exact receipt\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_source_receipt(path, digest) == digest
    with pytest.raises(ValueError, match="authenticated source receipt"):
        verify_source_receipt(path, "0" * 64)


def test_phase1_candidates_bind_facet_queries_and_text_hashes() -> None:
    candidates = phase1_candidates(_manifest(), _retrieval_rows())

    assert len(candidates) == 3
    assert all(row["query"] == "technology positive effects" for row in candidates)
    assert all(row["family"] == "facet" for row in candidates)
    assert candidates[0]["text_sha256"] == hashlib.sha256(
        b"one two three four five"
    ).hexdigest()


def test_preflight_projects_runtime_and_refuses_over_ten_minutes() -> None:
    candidates = phase1_candidates(_manifest(), _retrieval_rows())
    plan = build_phase_preflight(
        candidates,
        _Tokenizer(),
        cache_lookup=lambda query, text: None,
        pairs_per_second=100.0,
        fixed_seconds=1.0,
    )
    assert plan["summary"]["document_count"] == 3
    assert plan["summary"]["window_count"] == 3
    assert plan["summary"]["unique_uncached_pair_count"] == 1
    assert plan["projected_runtime_seconds"] < MAX_PHASE_SECONDS

    with pytest.raises(ValueError, match="600 seconds"):
        build_phase_preflight(
            candidates,
            _Tokenizer(),
            cache_lookup=lambda query, text: None,
            pairs_per_second=0.001,
            fixed_seconds=1.0,
        )


def test_protected_or_qrels_exposed_topics_are_rejected() -> None:
    manifest = _manifest()
    manifest["facets"][0]["topic_id"] = "144"
    rows = _retrieval_rows()
    rows[0]["topic_id"] = "144"
    with pytest.raises(ValueError, match="excluded topic"):
        phase1_candidates(manifest, rows)


def test_phase2_scores_every_accepted_document_with_common_and_narrative_queries() -> None:
    manifest = {
        "topic_ids": ["219"],
        "topics": [
            {
                "topic_id": "219",
                "query": "the exact full narrative",
                "common_query": "technology societal impacts",
            }
        ],
        "qrels_opened": False,
    }
    accepted = [
        {
            "topic_id": "219",
            "document_id": "d1",
            "text": "one document",
            "text_sha256": hashlib.sha256(b"one document").hexdigest(),
            "union_order": 1,
            "provenance": [{"family": "original", "rank": 1}],
        },
        {
            "topic_id": "219",
            "document_id": "d2",
            "text": "two document",
            "text_sha256": hashlib.sha256(b"two document").hexdigest(),
            "union_order": 2,
            "provenance": [{"family": "facet", "facet_id": "219-positive"}],
        },
    ]

    candidates = phase2_candidates(manifest, accepted)

    assert len(candidates) == 4
    assert {(row["document_id"], row["family"]) for row in candidates} == {
        ("d1", "common"),
        ("d1", "narrative"),
        ("d2", "common"),
        ("d2", "narrative"),
    }
    assert all("decision" not in row and "accepted" not in row for row in candidates)
