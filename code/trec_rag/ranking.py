"""Ranking and deduplication strategies for retrieved candidates."""

from __future__ import annotations

from collections import defaultdict

from trec_rag.pipeline_models import RankedCandidate, RetrievedCandidate


def _provenance(candidate: RetrievedCandidate) -> dict[str, object]:
    return {
        "variant_name": candidate.variant_name,
        "retriever_name": candidate.retriever_name,
        "query_text": candidate.query_text,
        "source_rank": candidate.rank,
        "source_score": candidate.score,
    }


def passthrough_rank(candidates: list[RetrievedCandidate]) -> list[RankedCandidate]:
    streams = {(candidate.variant_name, candidate.retriever_name) for candidate in candidates}
    if len(streams) > 1:
        raise ValueError("passthrough ranking supports exactly one retrieval stream")

    grouped: dict[tuple[str, str], list[RetrievedCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.topic_id, candidate.docid)].append(candidate)

    winners: list[tuple[RetrievedCandidate, list[RetrievedCandidate]]] = []
    for group in grouped.values():
        ordered_group = sorted(group, key=lambda row: (row.rank, -row.score))
        winners.append((ordered_group[0], ordered_group))

    ranked: list[RankedCandidate] = []
    final_ranks_by_topic: dict[str, int] = defaultdict(int)
    for winner, provenance_candidates in sorted(
        winners,
        key=lambda item: (item[0].topic_id, item[0].rank, -item[0].score),
    ):
        final_ranks_by_topic[winner.topic_id] += 1
        ranked.append(
            RankedCandidate(
                topic_id=winner.topic_id,
                docid=winner.docid,
                rank=final_ranks_by_topic[winner.topic_id],
                score=winner.score,
                text=winner.text,
                provenance=[_provenance(candidate) for candidate in provenance_candidates],
            )
        )
    return ranked
