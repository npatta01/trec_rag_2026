"""Ranking and deduplication strategies for retrieved candidates."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class WindowScore:
    score: float
    start_char: int
    end_char: int
    chunk_index: int


def _load_document_scores(path: Path) -> dict[tuple[str, str], float]:
    scores = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scores[(str(row["topic_id"]), str(row["docid"]))] = float(row["score"])
    return scores


def _load_window_scores(path: Path) -> dict[tuple[str, str], list[WindowScore]]:
    scores: dict[tuple[str, str], list[WindowScore]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scores[(str(row["topic_id"]), str(row["docid"]))].append(
            WindowScore(
                score=float(row["score"]),
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
                chunk_index=int(row["chunk_index"]),
            )
        )
    return scores


def _weighted(values: list[float], weights: tuple[float, ...]) -> float:
    usable_values = values[: len(weights)]
    usable_weights = weights[: len(usable_values)]
    return sum(value * weight for value, weight in zip(usable_values, usable_weights)) / sum(usable_weights)


def _interval_new_chars(interval: tuple[int, int], selected: list[tuple[int, int]]) -> int:
    start, end = interval
    if end <= start:
        return 0
    covered: list[tuple[int, int]] = []
    for selected_start, selected_end in selected:
        overlap_start = max(start, selected_start)
        overlap_end = min(end, selected_end)
        if overlap_end > overlap_start:
            covered.append((overlap_start, overlap_end))
    if not covered:
        return end - start
    covered.sort()
    merged: list[tuple[int, int]] = []
    for covered_start, covered_end in covered:
        if not merged or covered_start > merged[-1][1]:
            merged.append((covered_start, covered_end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], covered_end))
    return (end - start) - sum(covered_end - covered_start for covered_start, covered_end in merged)


def _relative_span_support(
    windows: list[WindowScore],
    *,
    delta: float,
    min_new_chars: int,
) -> int:
    threshold = windows[0].score - delta
    selected: list[tuple[int, int]] = []
    for window in windows:
        if window.score < threshold:
            continue
        interval = (window.start_char, window.end_char)
        if _interval_new_chars(interval, selected) >= min_new_chars:
            selected.append(interval)
    return len(selected)


def coverage_aware_long_doc_rank(
    candidates: list[RetrievedCandidate],
    *,
    document_score_path: Path,
    window_score_path: Path,
    long_document_weight: float,
    strongest_passage_weight: float,
    coverage_bonus_weight: float,
    relative_span_delta: float,
    support_cap: int,
    min_new_chars: int,
    top_window_weights: tuple[float, ...],
) -> list[RankedCandidate]:
    base_ranked = passthrough_rank(candidates)
    document_scores = _load_document_scores(document_score_path)
    window_scores = _load_window_scores(window_score_path)

    scored: list[tuple[RankedCandidate, float, dict[str, float | int]]] = []
    for row in base_ranked:
        key = (row.topic_id, row.docid)
        if key not in document_scores:
            raise ValueError(f"missing document reranker score for topic={row.topic_id} docid={row.docid}")
        windows = sorted(window_scores.get(key, []), key=lambda window: (-window.score, window.chunk_index))
        if not windows:
            raise ValueError(f"missing window reranker scores for topic={row.topic_id} docid={row.docid}")
        window_values = [window.score for window in windows]
        strongest_passage = _weighted(window_values, top_window_weights)
        support = min(
            _relative_span_support(
                windows,
                delta=relative_span_delta,
                min_new_chars=min_new_chars,
            ),
            support_cap,
        )
        document_score = document_scores[key]
        score = (
            long_document_weight * document_score
            + strongest_passage_weight * strongest_passage
            + coverage_bonus_weight * support
        )
        scored.append(
            (
                row,
                score,
                {
                    "long_document_relevance": document_score,
                    "strongest_passage_relevance": strongest_passage,
                    "bounded_coverage_support": support,
                },
            )
        )

    reranked: list[RankedCandidate] = []
    ranks_by_topic: dict[str, int] = defaultdict(int)
    for row, score, components in sorted(scored, key=lambda item: (item[0].topic_id, -item[1], item[0].rank)):
        ranks_by_topic[row.topic_id] += 1
        reranked.append(
            RankedCandidate(
                topic_id=row.topic_id,
                docid=row.docid,
                rank=ranks_by_topic[row.topic_id],
                score=score,
                text=row.text,
                provenance=[
                    *row.provenance,
                    {
                        "ranker": "coverage_aware_long_doc_aggregate",
                        "base_rank": row.rank,
                        **components,
                    },
                ],
            )
        )
    return reranked
