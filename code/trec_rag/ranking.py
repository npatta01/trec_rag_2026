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


def rrf_rank(
    candidates: list[RetrievedCandidate],
    *,
    k: int = 60,
    stream_weights: dict[str, float] | None = None,
) -> list[RankedCandidate]:
    if k < 1:
        raise ValueError("rrf k must be at least 1")
    stream_weights = stream_weights or {}

    stream_best: dict[tuple[str, str, str, str, str], RetrievedCandidate] = {}
    for candidate in candidates:
        stream_key = (
            candidate.topic_id,
            candidate.variant_name,
            candidate.retriever_name,
            candidate.query_text,
            candidate.docid,
        )
        current = stream_best.get(stream_key)
        if current is None or (candidate.rank, -candidate.score) < (current.rank, -current.score):
            stream_best[stream_key] = candidate

    grouped: dict[tuple[str, str], list[RetrievedCandidate]] = defaultdict(list)
    for candidate in stream_best.values():
        grouped[(candidate.topic_id, candidate.docid)].append(candidate)

    fused_rows: list[tuple[str, str, float, int, RetrievedCandidate, list[RetrievedCandidate]]] = []
    for (topic_id, docid), group in grouped.items():
        ordered_group = sorted(
            group,
            key=lambda row: (row.rank, -row.score, row.variant_name, row.query_text, row.retriever_name),
        )
        fused_score = sum(
            stream_weights.get(row.variant_name, 1.0) / (k + row.rank)
            for row in ordered_group
        )
        fused_rows.append((topic_id, docid, fused_score, ordered_group[0].rank, ordered_group[0], ordered_group))

    ranked: list[RankedCandidate] = []
    final_ranks_by_topic: dict[str, int] = defaultdict(int)
    for topic_id, docid, fused_score, _best_rank, best_candidate, provenance_candidates in sorted(
        fused_rows,
        key=lambda item: (item[0], -item[2], item[3], item[1]),
    ):
        final_ranks_by_topic[topic_id] += 1
        provenance = [
            _provenance(candidate)
            for candidate in sorted(
                provenance_candidates,
                key=lambda row: (row.variant_name, row.query_text, row.retriever_name, row.rank),
            )
        ]
        ranked.append(
            RankedCandidate(
                topic_id=topic_id,
                docid=docid,
                rank=final_ranks_by_topic[topic_id],
                score=fused_score,
                text=best_candidate.text,
                provenance=provenance,
            )
        )
    return ranked
