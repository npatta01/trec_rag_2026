from __future__ import annotations

import pytest

from trec_rag.artifact_semantics import compare_artifact_rows


def _document_row(docid: str, rank: int, score: float) -> dict[str, object]:
    return {
        "topic_id": "14",
        "docid": docid,
        "rank": rank,
        "score": score,
        # These two documents intentionally share content and therefore a
        # content-addressed score-cache entry.
        "score_cache_key": "shared-content-key",
        "query_sha256": "query-hash",
        "text_sha256": "shared-text-hash",
    }


def test_semantic_comparison_ignores_duplicate_content_group_row_order() -> None:
    canonical = [
        _document_row("doc-a", 1, 2.5),
        _document_row("doc-b", 2, 2.5),
        _document_row("doc-c", 3, -1.0),
    ]
    regenerated = [canonical[2], canonical[1], canonical[0]]

    comparison = compare_artifact_rows(canonical, regenerated, window=False)

    assert comparison["semantic_equal"] is True
    assert comparison["rows"] == 3
    assert comparison["row_order_ignored"] is True
    assert len(str(comparison["canonicalized_sha256"])) == 64


def test_semantic_comparison_rejects_changed_row() -> None:
    canonical = [_document_row("doc-a", 1, 2.5), _document_row("doc-b", 2, 2.5)]
    regenerated = [dict(canonical[1]), dict(canonical[0])]
    regenerated[0]["score"] = 2.0

    with pytest.raises(ValueError, match="changed_fields=\\['score'\\]"):
        compare_artifact_rows(canonical, regenerated, window=False)
