from __future__ import annotations

import hashlib

import pytest

from trec_rag.all_topic_tethered_score import (
    build_score_plan,
    percentile_features,
    render_tethered_query,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _TokenizerOnly:
    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, text, *, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False and truncation is False
        self.encoded.append(text)
        return text.split()

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return " ".join(token_ids)

    def num_special_tokens_to_add(self, *, pair):
        assert pair is True
        return 3


class _EmptyCache:
    scores: dict[str, float] = {}

    def get(self, *, query_text: str, text: str):
        return None


def _manifest() -> dict[str, object]:
    narrative = "full narrative"
    query = "subject relation facet"
    return {
        "topic_ids": ["14"],
        "topics": [
            {
                "topic_id": "14",
                "manifest_order": 0,
                "narrative": narrative,
                "narrative_sha256": _sha(narrative),
            }
        ],
        "facets": [
            {
                "topic_id": "14",
                "facet_id": "f1",
                "manifest_order": 0,
                "query": query,
                "query_sha256": _sha(query),
                "analyzer_terms": ["subject", "relation", "facet"],
            }
        ],
    }


def _union() -> list[dict[str, object]]:
    text = "document passage"
    return [
        {
            "topic_id": "14",
            "document_id": "d1",
            "text": text,
            "stream_provenance": [
                {
                    "stream_id": "original",
                    "stream_rank": 1,
                    "text_sha256": _sha(text),
                },
                {
                    "stream_id": "f1",
                    "stream_rank": 2,
                    "text_sha256": _sha(text),
                },
            ],
        }
    ]


def _score_rows() -> list[dict[str, object]]:
    return [
        {"topic_id": "14", "query_id": "n", "document_id": "a", "score": 9.0},
        {"topic_id": "14", "query_id": "n", "document_id": "b", "score": 1.0},
        {"topic_id": "14", "query_id": "f1", "document_id": "a", "score": 2.0},
        {"topic_id": "14", "query_id": "f1", "document_id": "b", "score": 2.0},
    ]


def _percentiles(rows: list[dict[str, object]], query_id: str) -> list[float]:
    return [
        float(row["percentile"])
        for row in rows
        if row["query_id"] == query_id
    ]


def test_tethered_query_contains_narrative_and_one_facet() -> None:
    query = render_tethered_query("full narrative", "subject relation facet")
    assert "full narrative" in query and "subject relation facet" in query


def test_raw_scores_never_normalize_across_queries() -> None:
    rows = percentile_features(_score_rows())
    assert {(r["topic_id"], r["query_id"]) for r in rows} == {
        ("14", "n"),
        ("14", "f1"),
    }
    assert _percentiles(rows, "n") == [1.0, 0.0]
    assert _percentiles(rows, "f1") == [0.5, 0.5]


def test_preflight_has_no_model_load() -> None:
    plan = build_score_plan(
        _union(), _manifest(), backend=_TokenizerOnly(), cache=_EmptyCache()
    )
    assert plan["external_calls"] == {
        "model_load": 0,
        "model_inference": 0,
        "network": 0,
        "hosted": 0,
        "paid": 0,
        "qrels": 0,
    }
    assert plan["pair_count"] == 3
    assert plan["window_count"] == 3
    assert plan["cache_hit_count"] == 0
    assert plan["cache_miss_count"] == 3


def test_preflight_reuses_one_facet_pair_even_when_original_also_retrieved_it() -> None:
    plan = build_score_plan(
        _union(), _manifest(), backend=_TokenizerOnly(), cache=_EmptyCache()
    )
    assert [row["query_id"] for row in plan["pairs"]] == ["n", "g", "f1"]


def test_percentiles_reject_duplicate_query_document_identity() -> None:
    rows = _score_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        percentile_features(rows)


def test_explicitly_authorized_historical_topic_uses_same_window_policy() -> None:
    manifest = _manifest()
    manifest["topic_ids"] = ["144"]
    manifest["topics"][0]["topic_id"] = "144"  # type: ignore[index]
    manifest["facets"][0]["topic_id"] = "144"  # type: ignore[index]
    union = _union()
    union[0]["topic_id"] = "144"
    plan = build_score_plan(
        union, manifest, backend=_TokenizerOnly(), cache=_EmptyCache()
    )
    assert plan["pair_count"] == 3
    assert {row["topic_id"] for row in plan["windows"]} == {"144"}


def test_topic_outside_experiment_allowlist_is_rejected() -> None:
    manifest = _manifest()
    manifest["topic_ids"] = ["999"]
    manifest["topics"][0]["topic_id"] = "999"  # type: ignore[index]
    manifest["facets"][0]["topic_id"] = "999"  # type: ignore[index]
    union = _union()
    union[0]["topic_id"] = "999"
    with pytest.raises(ValueError, match="allowlist"):
        build_score_plan(
            union, manifest, backend=_TokenizerOnly(), cache=_EmptyCache()
        )


def test_authenticated_union_accepts_normalized_duplicate_text_provenance() -> None:
    union = _union()
    union[0]["stream_provenance"][1]["text_sha256"] = _sha("document  passage")  # type: ignore[index]
    plan = build_score_plan(
        union, _manifest(), backend=_TokenizerOnly(), cache=_EmptyCache()
    )
    assert plan["pair_count"] == 3


def test_repeated_query_identities_tokenize_the_document_only_once() -> None:
    backend = _TokenizerOnly()
    build_score_plan(_union(), _manifest(), backend=backend, cache=_EmptyCache())
    assert backend.encoded.count("document passage") == 1
