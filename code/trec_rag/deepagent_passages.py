"""Small helpers for exposing shared source-bound passages to researchers.

Passage retrieval, scoring, ordering, and the top-100 policy belong to
``topic_passage_search``.  This module only groups already ordered source rows
for the handle layer; grouping never selects, reranks, or caps a passage.
"""

from __future__ import annotations

from collections.abc import Sequence

from trec_rag.topic_passage_search import SourcePassage


def group_by_document(
    passages: Sequence[SourcePassage],
) -> tuple[tuple[str, tuple[SourcePassage, ...]], ...]:
    """Group rows without changing their order within the source result.

    The caller must continue iterating the original passage sequence when it
    needs global passage order.  This helper is only an index for document
    metadata and does not impose a document or passage limit.
    """
    if not isinstance(passages, Sequence) or isinstance(passages, (str, bytes)):
        raise TypeError("passages must be a sequence of SourcePassage rows")
    if any(not isinstance(passage, SourcePassage) for passage in passages):
        raise TypeError("passages must contain SourcePassage rows")

    order: list[str] = []
    grouped: dict[str, list[SourcePassage]] = {}
    for passage in passages:
        rows = grouped.get(passage.docid)
        if rows is None:
            rows = []
            grouped[passage.docid] = rows
            order.append(passage.docid)
        rows.append(passage)
    return tuple((docid, tuple(grouped[docid])) for docid in order)
