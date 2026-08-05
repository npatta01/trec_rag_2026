from __future__ import annotations

import hashlib

from trec_rag.deepagent_passages import group_by_document
from trec_rag.topic_passage_search import SourcePassage


def _passage(docid: str, rank: int, passage_rank: int) -> SourcePassage:
    text = f"text from {docid}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return SourcePassage(
        f"p-{docid}",
        docid,
        digest,
        rank,
        float(100 - rank),
        0,
        len(text),
        0,
        len(text.encode()),
        digest,
        text,
        float(100 - passage_rank),
        passage_rank,
        f"cache-{docid}",
        digest,
        {"backend": "test-chunker"},
    )


def test_grouping_is_pure_and_preserves_global_order_inside_each_document() -> None:
    passages = (
        _passage("doc-a", 1, 1),
        _passage("doc-b", 2, 2),
        _passage("doc-a", 1, 3),
    )

    grouped = group_by_document(passages)

    assert [docid for docid, _ in grouped] == ["doc-a", "doc-b"]
    assert [row.rank for row in grouped[0][1]] == [1, 3]
    assert [row.rank for row in grouped[1][1]] == [2]
    assert [row for _, rows in grouped for row in rows] == [
        passages[0], passages[2], passages[1]
    ]
