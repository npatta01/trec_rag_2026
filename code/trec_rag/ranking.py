"""Ranking and deduplication strategies for retrieved candidates."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker
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
    chunk_count: int | None = None
    query_sha256: str | None = None
    document_text_sha256: str | None = None
    text_sha256: str | None = None


@dataclass(frozen=True)
class DocumentScore:
    score: float
    query_sha256: str | None = None
    text_sha256: str | None = None


def _validate_score_metadata(
    row: dict[str, object],
    expected: dict[str, object],
    *,
    path: Path,
    line_number: int,
) -> None:
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise ValueError(
                f"{path}:{line_number}: {field} must be {expected_value!r}; "
                f"found {row.get(field)!r}"
            )


def _load_document_scores(
    path: Path,
    *,
    expected_metadata: dict[str, object],
) -> dict[tuple[str, str], list[DocumentScore]]:
    scores: dict[tuple[str, str], list[DocumentScore]] = defaultdict(list)
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        _validate_score_metadata(row, expected_metadata, path=path, line_number=line_number)
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"{path}:{line_number}: document score must be finite")
        scores[(str(row["topic_id"]), str(row["docid"]))].append(
            DocumentScore(
                score=score,
                query_sha256=(str(row["query_sha256"]) if "query_sha256" in row else None),
                text_sha256=(str(row["text_sha256"]) if "text_sha256" in row else None),
            )
        )
    return dict(scores)


def _load_window_scores(
    path: Path,
    *,
    expected_metadata: dict[str, object],
) -> dict[tuple[str, str], dict[int, list[WindowScore]]]:
    scores: dict[tuple[str, str], dict[int, list[WindowScore]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        _validate_score_metadata(row, expected_metadata, path=path, line_number=line_number)
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"{path}:{line_number}: window score must be finite")
        chunk_index = int(row["chunk_index"])
        scores[(str(row["topic_id"]), str(row["docid"]))][chunk_index].append(
            WindowScore(
                score=score,
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
                chunk_index=chunk_index,
                chunk_count=(int(row["chunk_count"]) if "chunk_count" in row else None),
                query_sha256=(str(row["query_sha256"]) if "query_sha256" in row else None),
                document_text_sha256=(
                    str(row["document_text_sha256"])
                    if "document_text_sha256" in row
                    else None
                ),
                text_sha256=(str(row["text_sha256"]) if "text_sha256" in row else None),
            )
        )
    return {key: dict(value) for key, value in scores.items()}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _select_document_score(
    scores: list[DocumentScore],
    *,
    topic_id: str,
    docid: str,
    query_text: str,
    document_text: str,
    require_identity: bool,
) -> float:
    if require_identity:
        query_hash = _sha256_text(query_text)
        text_hash = _sha256_text(document_text)
        scores = [
            score
            for score in scores
            if score.query_sha256 == query_hash and score.text_sha256 == text_hash
        ]
    if not scores:
        raise ValueError(
            f"missing or stale document reranker score for topic={topic_id} docid={docid}"
        )
    values = {score.score for score in scores}
    if len(values) != 1:
        raise ValueError(
            f"conflicting duplicate document scores for topic={topic_id} docid={docid}"
        )
    return values.pop()


def _select_window_scores(
    scores_by_index: dict[int, list[WindowScore]],
    *,
    topic_id: str,
    docid: str,
    query_text: str,
    document_text: str,
    require_identity: bool,
) -> list[WindowScore]:
    query_hash = _sha256_text(query_text)
    document_hash = _sha256_text(document_text)
    selected: dict[int, WindowScore] = {}
    for chunk_index, candidates in scores_by_index.items():
        if require_identity:
            candidates = [
                score
                for score in candidates
                if score.query_sha256 == query_hash
                and score.document_text_sha256 == document_hash
            ]
        if not candidates:
            continue
        identities = {
            (score.score, score.start_char, score.end_char, score.chunk_count)
            for score in candidates
        }
        if len(identities) != 1:
            raise ValueError(
                f"conflicting duplicate window scores for topic={topic_id} "
                f"docid={docid} chunk={chunk_index}"
            )
        selected[chunk_index] = candidates[0]
    if not selected:
        raise ValueError(
            f"missing or stale window reranker scores for topic={topic_id} docid={docid}"
        )
    if require_identity:
        chunk_counts = {score.chunk_count for score in selected.values()}
        if len(chunk_counts) != 1 or None in chunk_counts:
            raise ValueError(
                f"inconsistent window chunk counts for topic={topic_id} docid={docid}"
            )
        chunk_count = int(chunk_counts.pop())
        if chunk_count <= 0 or set(selected) != set(range(chunk_count)):
            raise ValueError(
                f"incomplete window reranker scores for topic={topic_id} docid={docid}; "
                f"found {len(selected)} of {chunk_count} chunks"
            )
    return list(selected.values())


def _weighted(values: list[float], weights: tuple[float, ...]) -> float:
    usable_values = values[: len(weights)]
    usable_weights = weights[: len(usable_values)]
    denominator = sum(usable_weights)
    if denominator <= 0:
        raise ValueError("top-window weights must have a positive used prefix")
    return sum(value * weight for value, weight in zip(usable_values, usable_weights)) / denominator


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
    candidate_depth: int | None = None,
    expected_score_metadata: dict[str, object] | None = None,
    expected_document_score_metadata: dict[str, object] | None = None,
    expected_window_score_metadata: dict[str, object] | None = None,
    long_document_weight: float,
    strongest_passage_weight: float,
    coverage_bonus_weight: float,
    relative_span_delta: float,
    support_cap: int,
    min_new_chars: int,
    top_window_weights: tuple[float, ...],
) -> list[RankedCandidate]:
    base_ranked = passthrough_rank(candidates)
    expected_score_metadata = expected_score_metadata or {}
    expected_document_score_metadata = {
        **expected_score_metadata,
        **(expected_document_score_metadata or {}),
    }
    expected_window_score_metadata = {
        **expected_score_metadata,
        **(expected_window_score_metadata or {}),
    }
    document_scores = _load_document_scores(
        document_score_path,
        expected_metadata=expected_document_score_metadata,
    )
    window_scores = _load_window_scores(
        window_score_path,
        expected_metadata=expected_window_score_metadata,
    )
    document_schema_version = int(
        expected_document_score_metadata.get("artifact_schema_version", 0)
    )
    window_schema_version = int(
        expected_window_score_metadata.get("artifact_schema_version", 0)
    )
    if document_schema_version != window_schema_version:
        raise ValueError("document and window artifact schema versions must match")
    require_artifact_identity = document_schema_version >= 2
    artifact_chunker = None
    if require_artifact_identity:
        try:
            artifact_chunker = SemanticTextChunker(
                ChunkingConfig(
                    max_characters=int(
                        expected_window_score_metadata["chunk_max_characters"]
                    ),
                    overlap_characters=int(
                        expected_window_score_metadata["chunk_overlap_characters"]
                    ),
                )
            )
        except KeyError as exc:
            raise ValueError("schema-v2 window metadata must pin the chunking policy") from exc
    query_texts_by_topic: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        query_texts_by_topic[candidate.topic_id].add(candidate.query_text)
    for topic_id, query_texts in query_texts_by_topic.items():
        if len(query_texts) != 1:
            raise ValueError(f"multiple reranker query texts for topic={topic_id}")

    scored: list[tuple[RankedCandidate, float, dict[str, float | int]]] = []
    for row in base_ranked:
        if candidate_depth is not None and row.rank > candidate_depth:
            continue
        key = (row.topic_id, row.docid)
        query_text = next(iter(query_texts_by_topic[row.topic_id]))
        document_score = _select_document_score(
            document_scores.get(key, []),
            topic_id=row.topic_id,
            docid=row.docid,
            query_text=query_text,
            document_text=row.text,
            require_identity=require_artifact_identity,
        )
        selected_windows = _select_window_scores(
            window_scores.get(key, {}),
            topic_id=row.topic_id,
            docid=row.docid,
            query_text=query_text,
            document_text=row.text,
            require_identity=require_artifact_identity,
        )
        if artifact_chunker is not None:
            expected_chunks = artifact_chunker.split_text(row.text, document_id=row.docid)
            if len(expected_chunks) != len(selected_windows):
                raise ValueError(
                    f"window chunk policy mismatch for topic={row.topic_id} docid={row.docid}"
                )
            by_index = {window.chunk_index: window for window in selected_windows}
            for chunk_index, chunk in enumerate(expected_chunks):
                window = by_index[chunk_index]
                if (
                    window.start_char != chunk.start_char
                    or window.end_char != chunk.end_char
                    or window.text_sha256 != _sha256_text(chunk.text)
                ):
                    raise ValueError(
                        f"stale window content for topic={row.topic_id} "
                        f"docid={row.docid} chunk={chunk_index}"
                    )
        windows = sorted(
            selected_windows,
            key=lambda window: (-window.score, window.chunk_index),
        )
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
        score = (
            long_document_weight * document_score
            + strongest_passage_weight * strongest_passage
            + coverage_bonus_weight * support
        )
        if not math.isfinite(score):
            raise ValueError(
                f"non-finite reranker score for topic={row.topic_id} docid={row.docid}"
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

    reranked_by_topic: dict[str, list[RankedCandidate]] = defaultdict(list)
    for row, score, components in sorted(scored, key=lambda item: (item[0].topic_id, -item[1], item[0].rank)):
        topic_rows = reranked_by_topic[row.topic_id]
        topic_rows.append(
            RankedCandidate(
                topic_id=row.topic_id,
                docid=row.docid,
                rank=len(topic_rows) + 1,
                score=score,
                text=row.text,
                provenance=[
                    *row.provenance,
                    {
                        "ranker": "coverage_aware_long_doc_aggregate",
                        "base_rank": row.rank,
                        "candidate_depth": candidate_depth,
                        **components,
                    },
                ],
            )
        )

    if candidate_depth is None:
        return [
            row
            for topic_id in sorted(reranked_by_topic)
            for row in reranked_by_topic[topic_id]
        ]

    base_by_topic: dict[str, list[RankedCandidate]] = defaultdict(list)
    for row in base_ranked:
        base_by_topic[row.topic_id].append(row)

    reranked: list[RankedCandidate] = []
    for topic_id in sorted(base_by_topic):
        topic_rows = reranked_by_topic[topic_id]
        reranked.extend(topic_rows)
        previous_score = topic_rows[-1].score if topic_rows else 0.0
        for base_row in sorted(base_by_topic[topic_id], key=lambda row: row.rank):
            if base_row.rank <= candidate_depth:
                continue
            previous_score = math.nextafter(previous_score, -math.inf)
            reranked.append(
                RankedCandidate(
                    topic_id=base_row.topic_id,
                    docid=base_row.docid,
                    rank=len(topic_rows) + 1,
                    score=previous_score,
                    text=base_row.text,
                    provenance=[
                        *base_row.provenance,
                        {
                            "ranker": "bm25_tail_after_rerank",
                            "base_rank": base_row.rank,
                            "candidate_depth": candidate_depth,
                        },
                    ],
                )
            )
            topic_rows.append(reranked[-1])
    return reranked
