from __future__ import annotations

from trec_rag.deep_facet_candidate_gate import (
    build_prefix_unions,
    build_unions,
    quality_gate,
)


def _facet() -> dict[str, object]:
    return {
        "facet_id": "72-climate",
        "query": "deforestation climate impacts",
        "anchor_terms": ["deforestation"],
        "relation_terms": ["climate", "impact"],
        "wrong_domain_patterns": [r"weather forecast"],
    }


def _doc(docid: str, rank: int, text: str) -> dict[str, object]:
    return {"document_id": docid, "docid": docid, "rank": rank, "text": text}


def test_gate_accepts_coherent_top_five_despite_content_warning() -> None:
    docs = [
        _doc(f"d{rank}", rank, "deforestation climate impact essay")
        for rank in range(1, 6)
    ]
    decision = quality_gate(_facet(), docs)
    assert decision.accepted is True
    assert decision.content_warning_top5_count == 5
    assert decision.failed_checks == ()


def test_gate_rejects_anchor_failure_or_domain_drift() -> None:
    anchor_failure = [
        _doc("d1", 1, "climate impact"),
        _doc("d2", 2, "climate impact"),
        _doc("d3", 3, "deforestation climate impact"),
        _doc("d4", 4, "unrelated material"),
        _doc("d5", 5, "unrelated material"),
    ]
    assert quality_gate(_facet(), anchor_failure).accepted is False

    drift = [
        _doc(f"d{rank}", rank, "deforestation climate impact weather forecast")
        for rank in range(1, 6)
    ]
    decision = quality_gate(_facet(), drift)
    assert decision.accepted is False
    assert decision.failed_checks == ("wrong_domain",)


def test_gate_loss_remains_visible_in_raw_union() -> None:
    original = {
        "219": [_doc("original", 1, "original text")],
    }
    streams = [
        {
            "topic_id": "219",
            "facet_id": "accepted",
            "accepted": True,
            "bm25": [_doc("accepted-only", 1, "accepted text")],
            "minilm": [_doc("accepted-only", 1, "accepted text")],
        },
        {
            "topic_id": "219",
            "facet_id": "rejected",
            "accepted": False,
            "bm25": [_doc("rejected-only", 1, "rejected text")],
            "minilm": [_doc("rejected-only", 1, "rejected text")],
        },
    ]
    unions = build_unions(original, streams)
    assert "rejected-only" in unions["219"]["raw_docids"]
    assert "rejected-only" not in unions["219"]["accepted_docids"]
    assert "accepted-only" in unions["219"]["accepted_docids"]


def test_prefix_unions_freeze_bm25_and_minilm_at_50_100_200() -> None:
    original = {"219": [_doc("original", 1, "original text")]}
    bm25 = [_doc(f"b{rank}", rank, "text") for rank in range(1, 201)]
    minilm = [_doc(f"m{rank}", rank, "text") for rank in range(1, 201)]
    streams = [
        {
            "topic_id": "219",
            "facet_id": "accepted",
            "accepted": True,
            "bm25": bm25,
            "minilm": minilm,
        }
    ]
    prefixes = build_prefix_unions(original, streams)
    assert len(prefixes["219"]["bm25@50"]) == 51
    assert len(prefixes["219"]["bm25@100"]) == 101
    assert len(prefixes["219"]["bm25@200"]) == 201
    assert len(prefixes["219"]["minilm@50"]) == 51
    assert prefixes["219"]["minilm@200"][-1] == "m200"
