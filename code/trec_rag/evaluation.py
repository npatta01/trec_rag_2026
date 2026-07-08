"""Retrieval metrics for development qrels."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from trec_rag.pipeline_models import RankedCandidate


Qrels = dict[str, dict[str, int]]


def parse_qrels(path: Path) -> Qrels:
    qrels: Qrels = defaultdict(dict)
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) != 4:
            raise ValueError(f"line {line_number}: expected four qrels columns")
        topic_id, _zero, docid, grade_text = parts
        try:
            qrels[topic_id][docid] = int(grade_text)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: relevance grade must be an integer") from exc
    return dict(qrels)


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def _ndcg_at(ranked_docids: list[str], topic_qrels: dict[str, int], k: int) -> float:
    grades = [topic_qrels.get(docid, 0) for docid in ranked_docids[:k]]
    ideal = sorted(topic_qrels.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(grades) / ideal_dcg


def _recall_at(
    ranked_docids: list[str],
    topic_qrels: dict[str, int],
    k: int,
    relevance_threshold: int,
) -> float:
    relevant = _relevant_docids(topic_qrels, relevance_threshold)
    if not relevant:
        return 0.0
    return _relevant_count_at(ranked_docids, relevant, k) / len(relevant)


def _relevant_docids(topic_qrels: dict[str, int], relevance_threshold: int) -> set[str]:
    return {docid for docid, grade in topic_qrels.items() if grade >= relevance_threshold}


def _relevant_count_at(ranked_docids: list[str], relevant: set[str], k: int) -> int:
    return len(set(ranked_docids[:k]) & relevant)


def _precision_at(
    ranked_docids: list[str],
    topic_qrels: dict[str, int],
    k: int,
    relevance_threshold: int,
) -> float:
    if k <= 0:
        return 0.0
    relevant = _relevant_docids(topic_qrels, relevance_threshold)
    return _relevant_count_at(ranked_docids, relevant, k) / k


def _hit_rate_at(
    ranked_docids: list[str],
    topic_qrels: dict[str, int],
    k: int,
    relevance_threshold: int,
) -> float:
    relevant = _relevant_docids(topic_qrels, relevance_threshold)
    return 1.0 if _relevant_count_at(ranked_docids, relevant, k) > 0 else 0.0


def _graded_recall_at(ranked_docids: list[str], topic_qrels: dict[str, int], k: int) -> float:
    total_grade = sum(max(grade, 0) for grade in topic_qrels.values())
    if total_grade == 0:
        return 0.0
    retrieved_grade = sum(max(topic_qrels.get(docid, 0), 0) for docid in ranked_docids[:k])
    return retrieved_grade / total_grade


def _ideal_dcg_coverage_at(ranked_docids: list[str], topic_qrels: dict[str, int], k: int) -> float:
    true_ideal = sorted(topic_qrels.values(), reverse=True)[:k]
    true_ideal_dcg = _dcg(true_ideal)
    if true_ideal_dcg == 0:
        return 0.0
    retrieved_ideal = sorted((topic_qrels.get(docid, 0) for docid in ranked_docids[:k]), reverse=True)[:k]
    return _dcg(retrieved_ideal) / true_ideal_dcg


def evaluate_ranked(
    ranked: list[RankedCandidate],
    qrels: Qrels,
    *,
    metric_names: list[str] | tuple[str, ...],
    relevance_threshold: int,
) -> dict[str, object]:
    by_topic: dict[str, list[RankedCandidate]] = defaultdict(list)
    for candidate in ranked:
        by_topic[candidate.topic_id].append(candidate)

    per_topic: dict[str, dict[str, float]] = {}
    for topic_id in sorted(by_topic):
        topic_rows = sorted(by_topic[topic_id], key=lambda row: row.rank)
        docids = [row.docid for row in topic_rows]
        topic_qrels = qrels.get(topic_id, {})
        per_topic[topic_id] = {}
        for metric in metric_names:
            name, cutoff_text = metric.split("@", 1)
            cutoff = int(cutoff_text)
            if name == "ndcg":
                per_topic[topic_id][metric] = _ndcg_at(docids, topic_qrels, cutoff)
            elif name == "recall":
                per_topic[topic_id][metric] = _recall_at(
                    docids,
                    topic_qrels,
                    cutoff,
                    relevance_threshold,
                )
            elif name == "precision":
                per_topic[topic_id][metric] = _precision_at(
                    docids,
                    topic_qrels,
                    cutoff,
                    relevance_threshold,
                )
            elif name == "hit_rate":
                per_topic[topic_id][metric] = _hit_rate_at(
                    docids,
                    topic_qrels,
                    cutoff,
                    relevance_threshold,
                )
            elif name == "relevant_count":
                relevant = _relevant_docids(topic_qrels, relevance_threshold)
                per_topic[topic_id][metric] = _relevant_count_at(docids, relevant, cutoff)
            elif name == "graded_recall":
                per_topic[topic_id][metric] = _graded_recall_at(docids, topic_qrels, cutoff)
            elif name == "ideal_dcg_coverage":
                per_topic[topic_id][metric] = _ideal_dcg_coverage_at(docids, topic_qrels, cutoff)
            else:
                raise ValueError(f"unknown metric: {metric}")

    aggregate = {
        metric: (
            sum(topic_metrics[metric] for topic_metrics in per_topic.values()) / len(per_topic)
            if per_topic
            else 0.0
        )
        for metric in metric_names
    }
    return {"metrics": aggregate, "per_topic": per_topic}
