"""Generate a reproducible BM25-to-reranker topic regression postmortem.

The generator consumes saved pipeline stage outputs rather than calling a
retriever or reranker.  It reconstructs document-only, passage-only, and
no-coverage counterfactual rankings from the component values embedded in the
reranked stage provenance, validates that those components reproduce the
shipped score formula, and joins every ranking to qrels.

Outputs are a technical ``notes.md`` report plus machine-readable JSON/CSV/YAML
records.  Re-running this module after replacing score artifacts (for example,
with Modal CUDA outputs) refreshes all conclusions while preserving input
hashes and runtime labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from trec_rag.evaluation import evaluate_ranked, parse_qrels
from trec_rag.pipeline_models import RankedCandidate, jsonable
from trec_rag.rerank_cache_promotion import (
    RAG25_DEV_TOPIC_IDS,
    RAG25_WINDOW_ROWS_PER_TOPIC,
)
from trec_rag.topics import Topic, load_topics


DEFAULT_OUTPUT_DIR = Path(
    "reports/experiments/bm25_mixedbread_topic_regression_postmortem_v1"
)
DEFAULT_METRICS = (
    "ndcg@10",
    "precision@10",
    "relevant_count@10",
    "graded_recall@10",
    "judged_count@10",
    "judged_rate@10",
)
COUNTERFACTUAL_SYSTEMS = ("document", "passage", "no_coverage")
ALL_SYSTEMS = ("bm25", *COUNTERFACTUAL_SYSTEMS, "shipped")
EXPECTED_WARM_DOCUMENT_ROWS = 22_000
EXPECTED_WARM_WINDOW_ROWS = 227_156
EXPECTED_WARM_DOCUMENT_ROWS_BY_TOPIC = {
    topic_id: 1_000 for topic_id in RAG25_DEV_TOPIC_IDS
}
EXPECTED_WARM_WINDOW_ROWS_BY_TOPIC = dict(RAG25_WINDOW_ROWS_PER_TOPIC)
LARGE_REGRESSION_THRESHOLD = -0.1
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class ComponentRecord:
    topic_id: str
    docid: str
    base_rank: int
    document: float
    passage: float
    coverage_support: int
    no_coverage: float
    shipped: float
    text: str
    provenance: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        for field in ("topic_id", "docid", "rank", "score", "text", "provenance"):
            if field not in row:
                raise ValueError(f"{path}:{line_number}: missing {field!r}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no stage rows found")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _source_record(path: Path) -> dict[str, Any]:
    return {
        "path": _portable_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(topic_id))
    except ValueError:
        return (1, topic_id)


def _stage_by_topic(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        topic_id = str(row["topic_id"])
        docid = str(row["docid"])
        key = (topic_id, docid)
        if key in seen:
            raise ValueError(f"duplicate stage row for topic={topic_id} docid={docid}")
        seen.add(key)
        grouped[topic_id].append(row)
    for topic_id, topic_rows in grouped.items():
        topic_rows.sort(key=lambda row: int(row["rank"]))
        ranks = [int(row["rank"]) for row in topic_rows]
        if ranks != list(range(1, len(topic_rows) + 1)):
            raise ValueError(f"topic {topic_id}: stage ranks are not contiguous")
    return dict(grouped)


def _reranker_provenance(row: dict[str, Any]) -> dict[str, Any] | None:
    provenance = row.get("provenance")
    if not isinstance(provenance, list):
        raise ValueError("stage row provenance must be a list")
    matches = [
        item
        for item in provenance
        if isinstance(item, dict)
        and item.get("ranker") == "coverage_aware_long_doc_aggregate"
    ]
    if len(matches) > 1:
        raise ValueError(
            f"topic={row['topic_id']} docid={row['docid']}: duplicate reranker provenance"
        )
    return matches[0] if matches else None


def _formula(candidate_config: dict[str, Any]) -> dict[str, Any]:
    try:
        reranker = candidate_config["ranking"]["reranker"]
        formula = reranker["formula"]
    except (KeyError, TypeError) as exc:
        raise ValueError("candidate config is missing ranking.reranker.formula") from exc
    required = (
        "long_document_weight",
        "strongest_passage_weight",
        "coverage_bonus_weight",
    )
    if not isinstance(formula, dict) or any(field not in formula for field in required):
        raise ValueError("candidate formula is missing a required component weight")
    return {
        "long_document_weight": float(formula["long_document_weight"]),
        "strongest_passage_weight": float(formula["strongest_passage_weight"]),
        "coverage_bonus_weight": float(formula["coverage_bonus_weight"]),
        "relative_span_delta": float(formula.get("relative_span_delta", 0.0)),
        "support_cap": int(formula.get("support_cap", 0)),
        "min_new_chars": int(formula.get("min_new_chars", 0)),
        "top_window_weights": [float(value) for value in formula.get("top_window_weights", [])],
    }


def _candidate_depth(candidate_config: dict[str, Any]) -> int:
    try:
        value = int(candidate_config["ranking"]["reranker"]["candidate_depth"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate config is missing a valid reranker candidate_depth") from exc
    if value < 10:
        raise ValueError("candidate depth must be at least the report cutoff of 10")
    return value


def build_component_records(
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    *,
    candidate_depth: int,
    formula: dict[str, Any],
    score_tolerance: float = 1e-9,
) -> list[ComponentRecord]:
    """Validate stage comparability and extract per-document score components."""

    baseline_by_topic = _stage_by_topic(baseline_rows)
    candidate_by_topic = _stage_by_topic(candidate_rows)
    if set(baseline_by_topic) != set(candidate_by_topic):
        raise ValueError("baseline and candidate topic sets differ")

    records: list[ComponentRecord] = []
    long_weight = float(formula["long_document_weight"])
    passage_weight = float(formula["strongest_passage_weight"])
    coverage_weight = float(formula["coverage_bonus_weight"])
    for topic_id in sorted(baseline_by_topic, key=_topic_sort_key):
        baseline_top = {
            str(row["docid"]): row
            for row in baseline_by_topic[topic_id]
            if int(row["rank"]) <= candidate_depth
        }
        candidate_top = [
            row
            for row in candidate_by_topic[topic_id]
            if _reranker_provenance(row) is not None
        ]
        if len(baseline_top) != candidate_depth or len(candidate_top) != candidate_depth:
            raise ValueError(
                f"topic {topic_id}: expected {candidate_depth} component-scored documents; "
                f"baseline={len(baseline_top)} candidate={len(candidate_top)}"
            )
        if {str(row["docid"]) for row in candidate_top} != set(baseline_top):
            raise ValueError(f"topic {topic_id}: reranked and BM25 candidate pools differ")

        seen_base_ranks: set[int] = set()
        for row in candidate_top:
            component = _reranker_provenance(row)
            assert component is not None
            try:
                base_rank = int(component["base_rank"])
                document = float(component["long_document_relevance"])
                passage = float(component["strongest_passage_relevance"])
                coverage = int(component["bounded_coverage_support"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"topic={topic_id} docid={row['docid']}: incomplete reranker components"
                ) from exc
            if base_rank in seen_base_ranks:
                raise ValueError(f"topic {topic_id}: duplicate component base rank {base_rank}")
            seen_base_ranks.add(base_rank)
            baseline_row = baseline_top[str(row["docid"])]
            if int(baseline_row["rank"]) != base_rank:
                raise ValueError(
                    f"topic={topic_id} docid={row['docid']}: component base rank does not "
                    "match BM25"
                )
            no_coverage = long_weight * document + passage_weight * passage
            reconstructed = no_coverage + coverage_weight * coverage
            shipped = float(row["score"])
            if not math.isclose(
                reconstructed,
                shipped,
                rel_tol=score_tolerance,
                abs_tol=score_tolerance,
            ):
                raise ValueError(
                    f"topic={topic_id} docid={row['docid']}: shipped score {shipped} "
                    f"does not match reconstructed formula {reconstructed}"
                )
            provenance = row["provenance"]
            assert isinstance(provenance, list)
            records.append(
                ComponentRecord(
                    topic_id=topic_id,
                    docid=str(row["docid"]),
                    base_rank=base_rank,
                    document=document,
                    passage=passage,
                    coverage_support=coverage,
                    no_coverage=no_coverage,
                    shipped=shipped,
                    text=str(row["text"]),
                    provenance=provenance,
                )
            )
    return records


def _ranked_from_stage(rows: Sequence[dict[str, Any]]) -> list[RankedCandidate]:
    return [
        RankedCandidate(
            topic_id=str(row["topic_id"]),
            docid=str(row["docid"]),
            rank=int(row["rank"]),
            score=float(row["score"]),
            text=str(row["text"]),
            provenance=list(row["provenance"]),
        )
        for row in rows
    ]


def build_counterfactual_rankings(
    records: Sequence[ComponentRecord],
) -> dict[str, list[RankedCandidate]]:
    """Rank the fixed candidate pools by each saved component score."""

    by_topic: dict[str, list[ComponentRecord]] = defaultdict(list)
    for record in records:
        by_topic[record.topic_id].append(record)
    rankings: dict[str, list[RankedCandidate]] = {
        system: [] for system in COUNTERFACTUAL_SYSTEMS
    }
    for system in COUNTERFACTUAL_SYSTEMS:
        for topic_id in sorted(by_topic, key=_topic_sort_key):
            rows = sorted(
                by_topic[topic_id],
                key=lambda row: (-float(getattr(row, system)), row.base_rank),
            )
            for rank, row in enumerate(rows, 1):
                rankings[system].append(
                    RankedCandidate(
                        topic_id=topic_id,
                        docid=row.docid,
                        rank=rank,
                        score=float(getattr(row, system)),
                        text=row.text,
                        provenance=row.provenance,
                    )
                )
    return rankings


def _rank_lookup(rows: Sequence[RankedCandidate]) -> dict[tuple[str, str], int]:
    return {(row.topic_id, row.docid): row.rank for row in rows}


def _score_lookup(records: Sequence[ComponentRecord]) -> dict[tuple[str, str], ComponentRecord]:
    return {(row.topic_id, row.docid): row for row in records}


def _rankdata(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_mean = statistics.mean(left_ranks)
    right_mean = statistics.mean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_ranks)
        * sum((value - right_mean) ** 2 for value in right_ranks)
    )
    return numerator / denominator if denominator else None


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _calibration_rows(
    records: Sequence[ComponentRecord],
    qrels: dict[str, dict[str, int]],
    *,
    degraded_topics: set[str],
    focus_topics: Sequence[str],
    relevance_threshold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scopes: dict[str, list[ComponentRecord]] = {
        "all_topics": list(records),
        "degraded_topics": [row for row in records if row.topic_id in degraded_topics],
    }
    for topic_id in sorted(degraded_topics | set(focus_topics), key=_topic_sort_key):
        scopes[f"topic_{topic_id}"] = [row for row in records if row.topic_id == topic_id]

    component_fields = {
        "document_raw_logit": "document",
        "passage_weighted_raw_logit": "passage",
        "no_coverage_blend": "no_coverage",
        "bounded_coverage_support": "coverage_support",
        "shipped_score": "shipped",
    }
    summary_rows: list[dict[str, Any]] = []
    grade_rows: list[dict[str, Any]] = []
    for scope, scope_rows in scopes.items():
        grades = [qrels.get(row.topic_id, {}).get(row.docid, 0) for row in scope_rows]
        judged = [row.docid in qrels.get(row.topic_id, {}) for row in scope_rows]
        labels = [grade >= relevance_threshold for grade in grades]
        for component, field in component_fields.items():
            scores = [float(getattr(row, field)) for row in scope_rows]
            relevant_scores = [score for score, label in zip(scores, labels) if label]
            nonrelevant_scores = [score for score, label in zip(scores, labels) if not label]
            summary_rows.append(
                {
                    "scope": scope,
                    "component": component,
                    "candidate_count": len(scope_rows),
                    "judged_count": sum(judged),
                    "spearman_vs_grade": _spearman(scores, grades),
                    "auc_relevance": _auc(scores, labels),
                    "relevance_threshold": relevance_threshold,
                    "mean_relevant_score": (
                        statistics.mean(relevant_scores) if relevant_scores else None
                    ),
                    "mean_nonrelevant_score": (
                        statistics.mean(nonrelevant_scores) if nonrelevant_scores else None
                    ),
                }
            )
            for grade in sorted(set(grades)):
                values = [score for score, row_grade in zip(scores, grades) if row_grade == grade]
                grade_rows.append(
                    {
                        "scope": scope,
                        "component": component,
                        "grade": grade,
                        "count": len(values),
                        "mean_score": statistics.mean(values),
                        "median_score": statistics.median(values),
                        "min_score": min(values),
                        "max_score": max(values),
                    }
                )
    return summary_rows, grade_rows


def classify_regression(
    *,
    baseline_ndcg: float,
    shipped_ndcg: float,
    counterfactual_ndcgs: dict[str, float],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Describe whether a loss is recoverable from the saved component signals.

    ``aggregation_sensitive`` means at least one document, passage, or
    no-coverage ranking reaches the BM25 score while the shipped ranking does
    not. ``model_signal_limited`` means none does.  These labels are controlled
    counterfactual descriptions, not claims about general or external causes.
    """

    if shipped_ndcg >= baseline_ndcg - tolerance:
        raise ValueError("classification requires a shipped regression")
    if set(counterfactual_ndcgs) != set(COUNTERFACTUAL_SYSTEMS):
        raise ValueError("classification requires all counterfactual systems")
    best_system, best_value = max(
        counterfactual_ndcgs.items(),
        key=lambda item: (item[1], -COUNTERFACTUAL_SYSTEMS.index(item[0])),
    )
    classification = (
        "aggregation_sensitive"
        if best_value >= baseline_ndcg - tolerance
        else "model_signal_limited"
    )
    return {
        "classification": classification,
        "best_counterfactual": best_system,
        "best_counterfactual_ndcg_at_10": best_value,
        "best_counterfactual_delta_vs_bm25": best_value - baseline_ndcg,
        "shipped_delta_vs_best_counterfactual": shipped_ndcg - best_value,
    }


def _dcg_contribution(grade: int, rank: int | None, cutoff: int = 10) -> float:
    if rank is None or rank > cutoff:
        return 0.0
    return (2**grade - 1) / math.log2(rank + 1)


def _movement_rows(
    topics: Sequence[Topic],
    baseline_ranked: Sequence[RankedCandidate],
    shipped_ranked: Sequence[RankedCandidate],
    counterfactual_ranked: dict[str, list[RankedCandidate]],
    records: Sequence[ComponentRecord],
    qrels: dict[str, dict[str, int]],
    *,
    focus_topics: Sequence[str],
    relevance_threshold: int,
    cutoff: int = 10,
) -> list[dict[str, Any]]:
    titles = {topic.id: topic.title for topic in topics}
    baseline_lookup = _rank_lookup(baseline_ranked)
    shipped_lookup = _rank_lookup(shipped_ranked)
    counterfactual_lookups = {
        system: _rank_lookup(rows) for system, rows in counterfactual_ranked.items()
    }
    record_lookup = _score_lookup(records)
    rows: list[dict[str, Any]] = []
    for topic_id in focus_topics:
        topic_docids = {
            docid
            for (row_topic, docid), rank in baseline_lookup.items()
            if row_topic == topic_id and rank <= cutoff
        } | {
            docid
            for (row_topic, docid), rank in shipped_lookup.items()
            if row_topic == topic_id and rank <= cutoff
        }
        for docid in topic_docids:
            key = (topic_id, docid)
            record = record_lookup[key]
            baseline_rank = baseline_lookup[key]
            shipped_rank = shipped_lookup[key]
            grade = qrels.get(topic_id, {}).get(docid, 0)
            if baseline_rank <= cutoff and shipped_rank <= cutoff:
                movement = "stayed_top10"
            elif baseline_rank <= cutoff:
                movement = "exited_top10"
            else:
                movement = "entered_top10"
            baseline_dcg = _dcg_contribution(grade, baseline_rank, cutoff)
            shipped_dcg = _dcg_contribution(grade, shipped_rank, cutoff)
            rows.append(
                {
                    "topic_id": topic_id,
                    "topic_title": titles.get(topic_id, ""),
                    "docid": docid,
                    "grade": grade,
                    "is_relevant": grade >= relevance_threshold,
                    "movement": movement,
                    "bm25_rank": baseline_rank,
                    "shipped_rank": shipped_rank,
                    "promotion": baseline_rank - shipped_rank,
                    "document_rank": counterfactual_lookups["document"][key],
                    "passage_rank": counterfactual_lookups["passage"][key],
                    "no_coverage_rank": counterfactual_lookups["no_coverage"][key],
                    "document_raw_logit": record.document,
                    "passage_weighted_raw_logit": record.passage,
                    "bounded_coverage_support": record.coverage_support,
                    "no_coverage_score": record.no_coverage,
                    "shipped_score": record.shipped,
                    "bm25_dcg_contribution": baseline_dcg,
                    "shipped_dcg_contribution": shipped_dcg,
                    "dcg_contribution_delta": shipped_dcg - baseline_dcg,
                }
            )
    rows.sort(
        key=lambda row: (
            _topic_sort_key(str(row["topic_id"])),
            min(int(row["bm25_rank"]), int(row["shipped_rank"])),
            str(row["docid"]),
        )
    )
    return rows


def _recursive_scalar(payload: Any, keys: set[str]) -> str | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in keys and isinstance(value, (str, int, float, bool)):
                return str(value)
        for value in payload.values():
            result = _recursive_scalar(value, keys)
            if result is not None:
                return result
    elif isinstance(payload, list):
        for value in payload:
            result = _recursive_scalar(value, keys)
            if result is not None:
                return result
    return None


def _validated_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _candidate_artifact_sha256(
    candidate_run_metadata: dict[str, Any],
) -> dict[str, str]:
    try:
        reranker_cache = candidate_run_metadata["cache"]["reranker"]
        document_sha256 = reranker_cache["document_scores"]["sha256"]
        window_sha256 = reranker_cache["window_scores"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "candidate run metadata is missing reranker artifact SHA-256 provenance"
        ) from exc
    return {
        "document": _validated_sha256(
            document_sha256,
            label="candidate document artifact sha256",
        ),
        "window": _validated_sha256(
            window_sha256,
            label="candidate window artifact sha256",
        ),
    }


def _runtime_artifact_digests(
    payload: dict[str, Any],
    *,
    kind: str,
) -> dict[str, str]:
    found: dict[str, str] = {}
    validation = payload.get(f"{kind}_validation")
    if validation is not None:
        if not isinstance(validation, dict):
            raise ValueError(f"runtime status {kind}_validation must be an object")
        if "sha256" in validation:
            found[f"{kind}_validation.sha256"] = _validated_sha256(
                validation["sha256"],
                label=f"runtime status {kind}_validation.sha256",
            )
    for field in (f"{kind}_sha256", f"{kind}_artifact_sha256"):
        if field in payload:
            found[field] = _validated_sha256(
                payload[field],
                label=f"runtime status {field}",
            )
    if not found:
        raise ValueError(
            f"runtime status must include a canonical {kind} artifact SHA-256"
        )
    return found


def validate_runtime_status_for_candidate(
    payload: dict[str, Any],
    *,
    expected_artifact_sha256: dict[str, str],
    expected_volume_name: str | None = None,
) -> dict[str, Any]:
    """Bind a completed Modal status to the artifacts consumed by the pipeline."""

    if payload.get("state") != "completed":
        raise ValueError("runtime status must have state='completed'")
    runtime_volume_name = payload.get("volume_name")
    if runtime_volume_name is not None and (
        not isinstance(runtime_volume_name, str) or not runtime_volume_name.strip()
    ):
        raise ValueError("runtime status volume_name must be a non-empty string")
    if expected_volume_name is not None and (
        not isinstance(expected_volume_name, str) or not expected_volume_name.strip()
    ):
        raise ValueError("expected volume_name must be a non-empty string")
    if (
        runtime_volume_name is not None
        and expected_volume_name is not None
        and runtime_volume_name != expected_volume_name
    ):
        raise ValueError(
            "runtime status volume_name does not match the bound warm-cache proof"
        )
    volume_name = runtime_volume_name or expected_volume_name
    if volume_name is None:
        raise ValueError(
            "runtime status must report volume_name, or a warm-cache proof bound "
            "to this exact runtime status must supply it"
        )
    expected_counts = {
        "document": EXPECTED_WARM_DOCUMENT_ROWS,
        "window": EXPECTED_WARM_WINDOW_ROWS,
    }
    validated_digests: dict[str, str] = {}
    for kind, expected_count in expected_counts.items():
        count_field = f"{kind}_rows"
        if payload.get(count_field) != expected_count:
            raise ValueError(
                f"runtime status {count_field} must be {expected_count}; "
                f"found {payload.get(count_field)!r}"
            )
        expected_digest = _validated_sha256(
            expected_artifact_sha256.get(kind),
            label=f"expected {kind} artifact sha256",
        )
        observed_digests = _runtime_artifact_digests(payload, kind=kind)
        mismatches = {
            field: digest
            for field, digest in observed_digests.items()
            if digest != expected_digest
        }
        if mismatches:
            raise ValueError(
                f"runtime status {kind} artifact digest does not match the artifact "
                f"consumed by the candidate pipeline: {mismatches}"
            )
        validated_digests[kind] = expected_digest

        validation = payload.get(f"{kind}_validation")
        if isinstance(validation, dict) and "rows_by_topic" in validation:
            expected_rows_by_topic = (
                EXPECTED_WARM_DOCUMENT_ROWS_BY_TOPIC
                if kind == "document"
                else EXPECTED_WARM_WINDOW_ROWS_BY_TOPIC
            )
            if validation["rows_by_topic"] != expected_rows_by_topic:
                raise ValueError(
                    f"runtime status {kind}_validation.rows_by_topic does not "
                    "match the exact RAG25 topic population"
                )
    return {
        "state": "completed",
        "volume_name": volume_name,
        "artifact_sha256": validated_digests,
    }


def _runtime_summary(
    artifact_manifest: dict[str, Any],
    candidate_run_metadata: dict[str, Any],
    runtime_status: dict[str, Any] | None,
) -> dict[str, Any]:
    reranker_manifest = artifact_manifest.get("reranker")
    reranker_manifest = reranker_manifest if isinstance(reranker_manifest, dict) else {}
    cache = candidate_run_metadata.get("cache")
    cache = cache if isinstance(cache, dict) else {}
    reranker_cache = cache.get("reranker")
    reranker_cache = reranker_cache if isinstance(reranker_cache, dict) else {}
    score_metadata = reranker_cache.get("score_metadata")
    score_metadata = score_metadata if isinstance(score_metadata, dict) else {}
    runtime_payloads: list[Any] = [
        runtime_status or {},
        artifact_manifest.get("runtime") or {},
        candidate_run_metadata.get("runtime") or {},
    ]
    hardware = next(
        (
            value
            for payload in runtime_payloads
            if (
                value := _recursive_scalar(
                    payload,
                    {
                        "gpu",
                        "gpu_name",
                        "device_name",
                        "accelerator",
                        "hardware",
                        "gpu_model",
                    },
                )
            )
            is not None
        ),
        None,
    )
    provider = next(
        (
            value
            for payload in runtime_payloads
            if (
                value := _recursive_scalar(
                    payload,
                    {"provider", "platform", "cloud", "modal_cloud_provider"},
                )
            )
            is not None
        ),
        None,
    )
    modal_cloud_provider = _recursive_scalar(
        runtime_status or {}, {"modal_cloud_provider"}
    )
    modal_region = _recursive_scalar(runtime_status or {}, {"modal_region"})
    runtime_state = _recursive_scalar(runtime_status or {}, {"state"})
    is_completed_modal = bool(
        runtime_status
        and runtime_state == "completed"
        and (modal_cloud_provider or modal_region)
    )
    if is_completed_modal:
        provider_parts = ["Modal"]
        if modal_cloud_provider:
            provider_parts.append(modal_cloud_provider)
        if modal_region:
            provider_parts.append(modal_region)
        provider = " / ".join(provider_parts)
    return {
        "provider": provider or "not_recorded",
        "hardware": hardware or "not_recorded",
        "modal_cloud_provider": modal_cloud_provider,
        "modal_region": modal_region,
        "state": runtime_state or "not_recorded",
        "is_completed_modal": is_completed_modal,
        "model": score_metadata.get("model") or reranker_manifest.get("model"),
        "model_revision": score_metadata.get("model_revision")
        or reranker_manifest.get("model_revision"),
        "backend": reranker_manifest.get("backend"),
        "backend_version": score_metadata.get("backend_version")
        or reranker_manifest.get("backend_version"),
        "inference_dtype": score_metadata.get("inference_dtype")
        or reranker_manifest.get("inference_dtype"),
        "score_representation": score_metadata.get("score_representation")
        or reranker_manifest.get("score_representation"),
        "input_policy": score_metadata.get("input_policy")
        or reranker_manifest.get("input_policy"),
        "artifact_schema_version": score_metadata.get("artifact_schema_version")
        or reranker_manifest.get("artifact_schema_version"),
        "runtime_status": runtime_status,
    }


def validate_warm_cache_status(
    payload: dict[str, Any],
    *,
    expected_artifact_sha256: dict[str, str] | None = None,
    expected_runtime_status_sha256: str | None = None,
    expected_volume_name: str | None = None,
) -> dict[str, Any]:
    """Validate and bind the full-cache, zero-model-call rematerialization proof."""

    if payload.get("state") != "completed":
        raise ValueError("warm-cache status must have state='completed'")
    if payload.get("score_cache_unchanged") is not True:
        raise ValueError("warm-cache status must prove score_cache_unchanged=true")
    if payload.get("semantic_equal_to_canonical") is not True:
        raise ValueError(
            "warm-cache status must prove semantic_equal_to_canonical=true"
        )
    model_required = payload.get("model_required")
    if model_required != {"document": False, "window": False}:
        raise ValueError(
            "warm-cache status must report both document and window models as unnecessary"
        )
    model_scores = payload.get("model_scores")
    if model_scores != {"document": 0, "window": 0}:
        raise ValueError(
            "warm-cache status must report zero document and window model scores"
        )
    for identity_field in ("app_name", "volume_name"):
        value = payload.get(identity_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"warm-cache status must report a non-empty {identity_field}"
            )
    if (
        expected_volume_name is not None
        and payload["volume_name"] != expected_volume_name
    ):
        raise ValueError(
            "warm-cache volume_name does not match the completed scoring runtime"
        )

    validated_source_status_sha256: str | None = None
    if expected_runtime_status_sha256 is not None:
        expected_runtime_status_sha256 = _validated_sha256(
            expected_runtime_status_sha256,
            label="expected runtime status sha256",
        )
        source_status_sha256 = _validated_sha256(
            payload.get("source_status_sha256"),
            label="warm-cache source_status_sha256",
        )
        if source_status_sha256 != expected_runtime_status_sha256:
            raise ValueError(
                "warm-cache source_status_sha256 does not match the supplied runtime status"
            )
        validated_source_status_sha256 = source_status_sha256

    validated_artifacts: dict[str, dict[str, Any]] = {}
    expected_rows_and_topics = (
        (
            "document",
            EXPECTED_WARM_DOCUMENT_ROWS,
            EXPECTED_WARM_DOCUMENT_ROWS_BY_TOPIC,
        ),
        (
            "window",
            EXPECTED_WARM_WINDOW_ROWS,
            EXPECTED_WARM_WINDOW_ROWS_BY_TOPIC,
        ),
    )
    for kind, expected_rows, expected_rows_by_topic in expected_rows_and_topics:
        artifact = payload.get(f"regenerated_{kind}")
        if not isinstance(artifact, dict):
            raise ValueError(f"warm-cache status is missing regenerated_{kind}")
        if artifact.get("rows") != expected_rows:
            raise ValueError(
                f"warm-cache {kind} row count must be {expected_rows}; "
                f"found {artifact.get('rows')!r}"
            )
        if artifact.get("all_rows_matched_modal_cache") is not True:
            raise ValueError(
                f"warm-cache {kind} artifact must match the Modal schema-v2 cache"
            )
        rows_by_topic = artifact.get("rows_by_topic")
        if not isinstance(rows_by_topic, dict):
            raise ValueError(
                f"warm-cache {kind} artifact rows_by_topic must be an object"
            )
        if rows_by_topic != expected_rows_by_topic:
            raise ValueError(
                f"warm-cache {kind} rows_by_topic must exactly match the "
                "22-topic RAG25 population"
            )
        regenerated_sha256 = _validated_sha256(
            artifact.get("sha256"),
            label=f"warm-cache regenerated {kind} sha256",
        )
        canonical_sha256 = _validated_sha256(
            payload.get(f"canonical_{kind}_sha256"),
            label=f"warm-cache canonical {kind} sha256",
        )
        if expected_artifact_sha256 is not None:
            expected_sha256 = _validated_sha256(
                expected_artifact_sha256.get(kind),
                label=f"expected {kind} artifact sha256",
            )
            if canonical_sha256 != expected_sha256:
                raise ValueError(
                    f"warm-cache canonical {kind} SHA-256 does not match the "
                    "artifact consumed by the candidate pipeline"
                )
        validated_artifacts[kind] = {
            "rows": expected_rows,
            "topic_count": len(rows_by_topic),
            "all_rows_matched_modal_cache": True,
            "regenerated_sha256": regenerated_sha256,
            "canonical_sha256": canonical_sha256,
        }

    return {
        "state": "completed",
        "verification_id": payload.get("verification_id"),
        "app_name": payload.get("app_name"),
        "volume_name": payload.get("volume_name"),
        "source_status_sha256": validated_source_status_sha256,
        "score_cache_unchanged": True,
        "semantic_equal_to_canonical": True,
        "model_required": model_required,
        "model_scores": model_scores,
        "zero_model_calls": True,
        "document": validated_artifacts["document"],
        "window": validated_artifacts["window"],
    }


def _fmt(value: float | None, digits: int = 6, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _metric_value(
    evaluations: dict[str, dict[str, Any]], system: str, topic_id: str, metric: str
) -> float:
    return float(evaluations[system]["per_topic"][topic_id][metric])


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _topic_id_list(topic_ids: Iterable[str]) -> str:
    values = list(topic_ids)
    if not values:
        return "none"
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _validate_focus_regressions(
    focus_topics: Sequence[str],
    regressions: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not focus_topics:
        raise ValueError("at least one focus topic is required")
    if len(set(focus_topics)) != len(focus_topics):
        raise ValueError("focus topics must be unique")
    regressions_by_topic = {
        str(regression["topic_id"]): regression for regression in regressions
    }
    non_regressions = [
        topic_id for topic_id in focus_topics if topic_id not in regressions_by_topic
    ]
    if non_regressions:
        raise ValueError(
            "requested focus topics are not regressions in the current artifacts: "
            f"{', '.join(non_regressions)}; select focus topics from "
            f"{', '.join(regressions_by_topic) or 'none'}"
        )
    return regressions_by_topic


def _report_markdown(
    payload: dict[str, Any],
    *,
    topics: Sequence[Topic],
    movement_rows: Sequence[dict[str, Any]],
    calibration_rows: Sequence[dict[str, Any]],
) -> str:
    evaluations = payload["evaluations"]
    regressions = payload["regressions"]
    runtime = payload["runtime"]
    formula = payload["formula"]
    focus_topics = payload["scope"]["focus_topics"]
    titles = {topic.id: topic.title for topic in topics}
    narratives = {topic.id: topic.narrative for topic in topics}
    aggregate_bm25 = float(evaluations["bm25"]["metrics"]["ndcg@10"])
    aggregate_shipped = float(evaluations["shipped"]["metrics"]["ndcg@10"])
    aggregation_count = sum(
        row["diagnostic"]["classification"] == "aggregation_sensitive"
        for row in regressions
    )
    signal_count = sum(
        row["diagnostic"]["classification"] == "model_signal_limited"
        for row in regressions
    )
    aggregation_phrase = "loss is" if aggregation_count == 1 else "losses are"
    signal_phrase = "is" if signal_count == 1 else "are"
    candidate_depth = int(payload["scope"]["candidate_depth"])
    regressions_by_topic = _validate_focus_regressions(focus_topics, regressions)
    large_regressions = [
        row
        for row in regressions
        if float(row["delta_ndcg_at_10"]) < LARGE_REGRESSION_THRESHOLD
    ]
    large_topic_ids = [str(row["topic_id"]) for row in large_regressions]
    large_diagnostics = ", ".join(
        f"{row['topic_id']} ({row['diagnostic']['classification'].replace('_', ' ')})"
        for row in large_regressions
    )
    coverage_helped_ids = [
        str(row["topic_id"])
        for row in large_regressions
        if float(row["coverage_counterfactual_effect"]) > 1e-12
    ]
    coverage_hurt_ids = [
        str(row["topic_id"])
        for row in large_regressions
        if float(row["coverage_counterfactual_effect"]) < -1e-12
    ]
    aggregation_topic_ids = [
        str(row["topic_id"])
        for row in regressions
        if row["diagnostic"]["classification"] == "aggregation_sensitive"
    ]
    signal_topic_ids = [
        str(row["topic_id"])
        for row in regressions
        if row["diagnostic"]["classification"] == "model_signal_limited"
    ]
    unchanged_precision_topic_ids = [
        str(row["topic_id"])
        for row in regressions
        if math.isclose(
            float(row["bm25_precision_at_10"]),
            float(row["shipped_precision_at_10"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    aggregate_no_coverage = float(evaluations["no_coverage"]["metrics"]["ndcg@10"])
    no_coverage_aggregate_delta = aggregate_no_coverage - aggregate_shipped
    no_coverage_unrecovered_large = sum(
        _metric_value(evaluations, "no_coverage", str(row["topic_id"]), "ndcg@10")
        < float(row["bm25_ndcg_at_10"]) - 1e-12
        for row in large_regressions
    )
    if large_regressions:
        large_regression_summary = (
            f"The losses beyond {LARGE_REGRESSION_THRESHOLD:.2f} nDCG@10 are "
            f"**{_topic_id_list(large_topic_ids)}**. Their controlled diagnostics are "
            f"{large_diagnostics}. "
        )
        if coverage_helped_ids:
            large_regression_summary += (
                "Coverage improves the no-coverage blend for "
                f"{_topic_id_list(coverage_helped_ids)}. "
            )
        if coverage_hurt_ids:
            large_regression_summary += (
                "Coverage worsens the no-coverage blend for "
                f"{_topic_id_list(coverage_hurt_ids)}. "
            )
        large_regression_summary += (
            "These are ranking counterfactuals over saved scores, not causal claims."
        )
    else:
        large_regression_summary = (
            f"No topic crosses the predeclared {LARGE_REGRESSION_THRESHOLD:.2f} "
            "large-regression threshold in these artifacts."
        )
    is_completed_modal = bool(runtime["is_completed_modal"])
    warm_cache = payload["warm_cache"]
    judged_count = int(payload["scope"]["judged_candidate_count"])
    candidate_count = int(payload["scope"]["scored_candidate_count"])
    judged_rate = float(payload["scope"]["judged_candidate_rate"])
    modal_summary = (
        f"This version was regenerated from a completed Modal scoring run on "
        f"`{runtime['hardware']}` ({runtime['provider']})."
        if is_completed_modal
        else (
            "No completed Modal runtime status was supplied, so this version remains "
            "the pre-promotion reference and must be regenerated after promotion."
        )
    )
    warm_cache_summary = (
        "Completed warm-cache verification rematerialized scores for **22,000 "
        "documents** and **227,156 document windows** from the persistent Modal "
        "schema-v2 cache with both models unnecessary and **zero model calls**. "
        "The cache remained unchanged and both regenerated artifacts were "
        "semantically equal to their canonical artifacts."
        if warm_cache is not None
        else (
            "No completed warm-cache verification status was supplied, so this "
            "version makes no full-corpus cache-hit claim."
        )
    )

    lines = [
        "# BM25 to Mixedbread Topic Regression Postmortem",
        "",
        "## Technical summary",
        "",
        f"The shipped top-{candidate_depth} reranker changes mean development nDCG@10 from "
        f"**{aggregate_bm25:.6f}** to **{aggregate_shipped:.6f}** "
        f"(**{aggregate_shipped - aggregate_bm25:+.6f}**), but regresses "
        f"**{len(regressions)}** of **{payload['scope']['topic_count']}** topics. "
        f"Within the fixed candidate pools, **{aggregation_count}** "
        f"{aggregation_phrase} aggregation-sensitive and **{signal_count}** "
        f"{signal_phrase} model-signal-limited.",
        "",
        large_regression_summary,
        "",
        modal_summary,
        "",
        warm_cache_summary,
        "",
        "The diagnostic labels are controlled descriptions of these saved scores, "
        "not causal or held-out generalization claims. The report must be "
        "regenerated when the score artifacts or runtime change.",
        "",
        f"## {aggregation_count} {aggregation_phrase} aggregation-sensitive; "
        f"{signal_count} {signal_phrase} model-signal-limited",
        "",
        "For an aggregation-sensitive loss, at least one component/no-coverage "
        "counterfactual reaches BM25 while the shipped formula does not. For a "
        "model-signal-limited loss, none does. This table isolates formula "
        f"sensitivity while holding the top-{candidate_depth} candidates and saved "
        "raw scores fixed.",
        "",
        "| Topic | BM25 | Shipped | Delta | Best counterfactual | Best nDCG | "
        "Shipped - no coverage | Diagnostic |",
        "|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in regressions:
        diagnostic = row["diagnostic"]
        lines.append(
            f"| {row['topic_id']} | {row['bm25_ndcg_at_10']:.6f} | "
            f"{row['shipped_ndcg_at_10']:.6f} | {row['delta_ndcg_at_10']:+.6f} | "
            f"{diagnostic['best_counterfactual']} | "
            f"{diagnostic['best_counterfactual_ndcg_at_10']:.6f} | "
            f"{row['coverage_counterfactual_effect']:+.6f} | "
            f"{diagnostic['classification'].replace('_', ' ')} |"
        )
    lines.extend(
        [
            "",
            "The no-coverage counterfactual is the configured document/passage "
            "blend with the bounded coverage bonus set to zero. Because nDCG is "
            "rank-based, `shipped - no coverage` is a ranking counterfactual, not "
            "the arithmetic contribution of the bonus to nDCG.",
            "",
        ]
    )

    calibration_by_scope_component = {
        (str(row["scope"]), str(row["component"])): row for row in calibration_rows
    }
    movements_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in movement_rows:
        movements_by_topic[str(row["topic_id"])].append(row)

    for topic_id in focus_topics:
        regression = regressions_by_topic[topic_id]
        topic_calibration = calibration_by_scope_component[
            (f"topic_{topic_id}", "shipped_score")
        ]
        topic_candidate_count = int(topic_calibration["candidate_count"])
        topic_judged_count = int(topic_calibration["judged_count"])
        topic_judged_rate = (
            topic_judged_count / topic_candidate_count if topic_candidate_count else 0.0
        )
        classification = str(regression["diagnostic"]["classification"])
        section_title = (
            f"Topic {topic_id}: saved component signals did not recover BM25"
            if classification == "model_signal_limited"
            else f"Topic {topic_id}: a saved-score counterfactual recovered BM25"
        )
        lines.extend(
            [
                f"## {section_title}",
                "",
                f"**Short label:** {titles.get(topic_id, '')}.<br>",
                f"**Information need:** {narratives.get(topic_id, '')}",
                "",
                f"The shipped ranking changes nDCG@10 from "
                f"**{regression['bm25_ndcg_at_10']:.6f}** to "
                f"**{regression['shipped_ndcg_at_10']:.6f}** "
                f"(**{regression['delta_ndcg_at_10']:+.6f}**). Precision@10 "
                f"moves from **{regression['bm25_precision_at_10']:.2f}** to "
                f"**{regression['shipped_precision_at_10']:.2f}**, and the count "
                f"of grade-{payload['scope']['relevance_threshold']}+ documents "
                f"moves from **{int(regression['bm25_relevant_count_at_10'])}** "
                f"to **{int(regression['shipped_relevant_count_at_10'])}**.",
                "",
                "| Ranking | nDCG@10 | Delta vs BM25 | P@10 | Relevant @10 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for system in ALL_SYSTEMS:
            ndcg = _metric_value(evaluations, system, topic_id, "ndcg@10")
            precision = _metric_value(evaluations, system, topic_id, "precision@10")
            relevant_count = _metric_value(
                evaluations, system, topic_id, "relevant_count@10"
            )
            lines.append(
                f"| {system.replace('_', ' ')} | {ndcg:.6f} | "
                f"{ndcg - regression['bm25_ndcg_at_10']:+.6f} | "
                f"{precision:.2f} | {int(relevant_count)} |"
            )

        baseline_grades = [
            int(row["grade"])
            for row in sorted(movements_by_topic[topic_id], key=lambda row: row["bm25_rank"])
            if int(row["bm25_rank"]) <= 10
        ]
        shipped_grades = [
            int(row["grade"])
            for row in sorted(
                movements_by_topic[topic_id], key=lambda row: row["shipped_rank"]
            )
            if int(row["shipped_rank"]) <= 10
        ]
        lines.extend(
            [
                "",
                f"BM25 top-10 qrel grades: `{baseline_grades}`.<br>",
                f"Shipped top-10 qrel grades: `{shipped_grades}`.",
                "",
                "The rows below have the largest negative document-level DCG "
                "contribution changes. They show the measured rank/grade mechanism "
                "without asserting a semantic cause for the qrel label.",
                "",
                "| Docid | Grade | BM25 rank | Shipped rank | Document | Passage | "
                "Support | DCG contribution delta |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        worst_movements = sorted(
            movements_by_topic[topic_id],
            key=lambda row: (float(row["dcg_contribution_delta"]), str(row["docid"])),
        )[:6]
        for row in worst_movements:
            lines.append(
                f"| {row['docid']} | {row['grade']} | {row['bm25_rank']} | "
                f"{row['shipped_rank']} | {row['document_raw_logit']:.4f} | "
                f"{row['passage_weighted_raw_logit']:.4f} | "
                f"{row['bounded_coverage_support']} | "
                f"{row['dcg_contribution_delta']:+.4f} |"
            )

        lines.extend(
            [
                "",
                f"Calibration uses this topic's top-{candidate_depth} candidate pool: "
                f"**{topic_judged_count} of {topic_candidate_count}** rows are "
                f"explicitly judged (**{topic_judged_rate:.2%}**). "
                "Spearman measures monotonic association with the full qrel grade; "
                "AUC measures separation of grade-"
                f"{payload['scope']['relevance_threshold']}+ from lower grades. "
                "Values near zero correlation or 0.5 AUC indicate weak separation "
                "inside this topic's candidate pool.",
                "",
                "| Component | Spearman vs grade | Relevance AUC | Mean relevant | "
                "Mean lower-grade |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for component in (
            "document_raw_logit",
            "passage_weighted_raw_logit",
            "no_coverage_blend",
            "bounded_coverage_support",
            "shipped_score",
        ):
            row = calibration_by_scope_component[(f"topic_{topic_id}", component)]
            lines.append(
                f"| {component.replace('_', ' ')} | "
                f"{_fmt(row['spearman_vs_grade'], 3)} | "
                f"{_fmt(row['auc_relevance'], 3)} | "
                f"{_fmt(row['mean_relevant_score'], 3)} | "
                f"{_fmt(row['mean_nonrelevant_score'], 3)} |"
            )

        diagnostic = regression["diagnostic"]
        best_counterfactual = str(diagnostic["best_counterfactual"])
        best_delta = float(diagnostic["best_counterfactual_delta_vs_bm25"])
        if classification == "model_signal_limited":
            diagnostic_sentence = (
                "None of the document-only, passage-only, or no-coverage rankings "
                "recovers BM25. The strongest saved-score alternative is "
                f"**{best_counterfactual.replace('_', ' ')}**, which remains "
                f"**{best_delta:+.6f}** nDCG from BM25."
            )
        else:
            diagnostic_sentence = (
                f"The **{best_counterfactual.replace('_', ' ')}** counterfactual "
                "recovers the BM25 result and finishes "
                f"**{best_delta:+.6f}** nDCG from BM25, while the shipped formula "
                "still regresses. This isolates aggregation sensitivity in the saved "
                "scores."
            )
        coverage_effect = float(regression["coverage_counterfactual_effect"])
        if coverage_effect > 1e-12:
            coverage_sentence = (
                "Coverage improves the no-coverage ranking by "
                f"**{coverage_effect:+.6f}** nDCG."
            )
        elif coverage_effect < -1e-12:
            coverage_sentence = (
                "Coverage worsens the no-coverage ranking by "
                f"**{coverage_effect:+.6f}** nDCG."
            )
        else:
            coverage_sentence = (
                "Coverage leaves the no-coverage ranking unchanged at nDCG@10."
            )
        precision_delta = float(regression["shipped_precision_at_10"]) - float(
            regression["bm25_precision_at_10"]
        )
        relevant_delta = int(regression["shipped_relevant_count_at_10"]) - int(
            regression["bm25_relevant_count_at_10"]
        )
        if abs(precision_delta) <= 1e-12 and relevant_delta == 0:
            binary_sentence = (
                "Binary precision and the grade-threshold relevant count are unchanged, "
                "so the measured loss is entirely in graded ordering."
            )
        else:
            binary_sentence = (
                f"Precision@10 changes by **{precision_delta:+.2f}** and the "
                f"grade-threshold relevant count changes by **{relevant_delta:+d}**, "
                "so both binary inclusion and graded order should be inspected."
            )
        movement_sentence = ""
        if worst_movements:
            worst = worst_movements[0]
            movement_sentence = (
                f" The largest negative saved DCG movement is `{worst['docid']}` "
                f"(grade {worst['grade']}), from BM25 rank {worst['bm25_rank']} to "
                f"shipped rank {worst['shipped_rank']} "
                f"({float(worst['dcg_contribution_delta']):+.4f} DCG)."
            )
        lines.extend(
            [
                "",
                f"**Interpretation.** {diagnostic_sentence} {coverage_sentence} "
                f"{binary_sentence}{movement_sentence} The shipped-score calibration "
                f"is Spearman **{_fmt(topic_calibration['spearman_vs_grade'], 3)}** "
                f"against qrel grade and AUC **{_fmt(topic_calibration['auc_relevance'], 3)}** "
                "for the configured relevance threshold; these values describe the "
                "observed pool and do not establish a semantic cause.",
                "",
            ]
        )

    hardware_text = str(runtime["hardware"])
    provider_text = str(runtime["provider"])
    lines.extend(
        [
            "## What was measured",
            "",
            f"The population is **{payload['scope']['topic_count']}** development "
            "topics. BM25 retrieves 1,000 documents per topic; the reranker reorders "
            f"the first **{payload['scope']['candidate_depth']}**. Metrics use the "
            "projected development qrels, treat missing judgments as grade 0, and "
            f"use grade **{payload['scope']['relevance_threshold']}+** for precision "
            "and relevant-count metrics. nDCG uses the full integer grades.",
            "",
            f"All counterfactuals reorder exactly the same top-{candidate_depth} "
            "documents. Document "
            "and passage values are raw logits; the passage component is the saved "
            "weighted top-window aggregate. The no-coverage score is:",
            "",
            "```text",
            f"{formula['long_document_weight']} * document_raw_logit",
            f"+ {formula['strongest_passage_weight']} * passage_weighted_raw_logit",
            "```",
            "",
            "The shipped score adds:",
            "",
            "```text",
            f"+ {formula['coverage_bonus_weight']} * bounded_coverage_support",
            "```",
            "",
            f"Scores are labeled `{runtime['model']}` at revision "
            f"`{runtime['model_revision']}`, backend version "
            f"`{runtime['backend_version']}`, `{runtime['inference_dtype']}`, and "
            f"`{runtime['score_representation']}`. Provider is `{provider_text}` and "
            f"hardware is `{hardware_text}`. `not_recorded` means the supplied "
            "artifacts do not support a stronger runtime claim.",
            "",
            "## Counterfactual method separates score signal from formula sensitivity",
            "",
            f"The generator verifies that every shipped top-{candidate_depth} score "
            "equals the "
            "configured component formula and that each component's `base_rank` "
            "matches BM25. It then sorts each topic independently by document, "
            "passage, and no-coverage score, breaking ties by BM25 rank. A loss is "
            "aggregation-sensitive only if one of those saved-score rankings reaches "
            "the BM25 nDCG@10. Otherwise it is model-signal-limited.",
            "",
            "This rule deliberately does not call the labels causal. It diagnoses "
            "whether the observed regression can be repaired by recombining the "
            "existing component scores; it cannot establish why the model assigned "
            "those scores or whether the pattern will repeat on hidden topics.",
            "",
            "## Limitations and robustness checks",
            "",
            "- The analysis covers 22 development topics, so topic-level effects are "
            "high variance and unsuitable as precise test-set estimates.",
            f"- Projected qrels are the metric source of truth here. Explicit "
            f"judgments cover {judged_count} of {candidate_count} top-{candidate_depth} "
            "candidates "
            f"({judged_rate:.2%}); missing judgments receive grade 0. The report does "
            "not infer semantic facets or qrel intent beyond those labels.",
            "- Counterfactuals reuse saved logits. They isolate aggregation choices "
            "but do not measure the effect of rescoring with a new model, GPU, dtype, "
            "library version, or longer candidate depth.",
            f"- The no-coverage ranking changes aggregate nDCG by "
            f"{no_coverage_aggregate_delta:+.6f} versus shipped and leaves "
            f"{no_coverage_unrecovered_large} of {len(large_regressions)} current "
            "large regressions below BM25. This controlled comparison does not by "
            "itself support removing or retaining coverage as a general fix.",
            (
                "- Hardware is surfaced from the supplied completed Modal status. "
                "This report is current for the promoted Modal artifacts named in the "
                "manifest; a later artifact replacement must trigger regeneration."
                if is_completed_modal
                else "- Hardware is surfaced from supplied runtime metadata and otherwise "
                "marked not recorded. Modal CUDA promotion must trigger regeneration "
                "before this document is treated as current."
            ),
            (
                "- The completed warm-cache proof covers this exact score identity "
                "and corpus: 22,000 document rows and 227,156 window rows, with zero "
                "model calls, an unchanged cache, and semantic equality to canonical "
                "artifacts. A model revision, score policy, query/document text, or "
                "chunking change requires a new proof."
                if warm_cache is not None
                else "- No validated warm-cache status was supplied. Full-corpus "
                "cache reuse remains unverified in this version of the report."
            ),
            "",
            "## Recommended next steps",
            "",
            (
                "1. The completed Modal artifacts are reflected here. Diff "
                "`metrics.json`, `component_counterfactuals.csv`, and "
                "`top10_movements.csv` against any retained pre-Modal snapshot if a "
                "cross-runtime effect size is needed."
                if is_completed_modal
                else "1. After Modal scoring is promoted, rerun this generator and diff "
                "`metrics.json`, `component_counterfactuals.csv`, and "
                "`top10_movements.csv` before updating any conclusion."
            ),
            (
                f"2. Treat topics {_topic_id_list(signal_topic_ids)} as current "
                "score-calibration/model-signal probes. Test BM25-preserving "
                "interpolation or a confidence guardrail before tuning aggregation "
                "around them."
                if signal_topic_ids
                else "2. No current regression is model-signal-limited under the saved "
                "counterfactual rule; retain that check as a guardrail on future runs."
            ),
            (
                f"3. Treat topics {_topic_id_list(aggregation_topic_ids)} as current "
                "aggregation-sensitivity probes. Evaluate formula changes with "
                "leave-one-topic-out selection and keep the per-topic regression "
                "guardrail, not only mean nDCG."
                if aggregation_topic_ids
                else "3. No current regression is aggregation-sensitive under the saved "
                "counterfactual rule; retain component ablations on future runs."
            ),
            (
                "4. Retain both nDCG@10 and precision/relevant-count diagnostics. "
                f"Topics {_topic_id_list(unchanged_precision_topic_ids)} demonstrate "
                "that binary precision can stay flat while graded ordering worsens."
                if unchanged_precision_topic_ids
                else "4. Retain both nDCG@10 and precision/relevant-count diagnostics; "
                "they measure different failure modes even though all current losses "
                "also move binary precision."
            ),
            "",
            "## Further questions",
            "",
            (
                "- The current top-10 tables are computed from the promoted Modal "
                "CUDA logits. Quantifying how much CUDA itself changed ordering still "
                "requires a retained pre-promotion ranked-output snapshot; runtime "
                "attribution is not inferred from the post-promotion files alone."
                if is_completed_modal
                else "- Do Modal CUDA logits materially reorder the top 10 on focus "
                f"topics {_topic_id_list(focus_topics)} relative to the currently "
                "pinned artifacts?"
            ),
            "- Which judged document properties distinguish high-grade documents "
            "that BM25 preserves but both reranker components down-rank? That needs "
            "an explicit labeled feature study; this report does not infer facets "
            "from document prose.",
            "- Can a single frozen blend recover the aggregation-sensitive topics "
            f"without worsening the {signal_count} signal-limited topics under "
            "leave-one-topic-out validation?",
            "",
            "## Reproduce",
            "",
            "```bash",
            payload["reproduce_command"],
            "```",
            "",
            "Machine-readable evidence is in "
            "[metrics.json](metrics.json), "
            "[component_counterfactuals.csv](component_counterfactuals.csv), "
            "[component_calibration.csv](component_calibration.csv), "
            "[component_grade_summary.csv](component_grade_summary.csv), and "
            "[top10_movements.csv](top10_movements.csv). Input paths and SHA-256 "
            "digests are pinned in [manifest.yaml](manifest.yaml).",
            "",
        ]
    )
    return "\n".join(lines)


def generate_postmortem(
    *,
    baseline_ranked_path: Path,
    candidate_ranked_path: Path,
    qrels_path: Path,
    topics_path: Path,
    candidate_config_path: Path,
    candidate_run_metadata_path: Path,
    artifact_manifest_path: Path,
    output_dir: Path,
    runtime_status_path: Path | None = None,
    warm_cache_status_path: Path | None = None,
    focus_topics: Sequence[str] = ("224", "515"),
    relevance_threshold: int = 2,
) -> dict[str, Any]:
    baseline_rows = _read_jsonl(baseline_ranked_path)
    candidate_rows = _read_jsonl(candidate_ranked_path)
    candidate_config = _read_yaml(candidate_config_path)
    candidate_run_metadata = _read_json(candidate_run_metadata_path)
    artifact_manifest = _read_yaml(artifact_manifest_path)
    runtime_status = _read_json(runtime_status_path) if runtime_status_path else None
    warm_cache_status = (
        _read_json(warm_cache_status_path) if warm_cache_status_path else None
    )
    if warm_cache_status is not None and runtime_status_path is None:
        raise ValueError(
            "--warm-cache-status requires --runtime-status so the proof can be "
            "bound to the exact completed scoring status"
        )
    candidate_artifact_sha256 = _candidate_artifact_sha256(candidate_run_metadata)
    runtime_binding = (
        validate_runtime_status_for_candidate(
            runtime_status,
            expected_artifact_sha256=candidate_artifact_sha256,
            expected_volume_name=(
                str(warm_cache_status.get("volume_name"))
                if warm_cache_status is not None
                and isinstance(warm_cache_status.get("volume_name"), str)
                else None
            ),
        )
        if runtime_status is not None
        else None
    )
    warm_cache = (
        validate_warm_cache_status(
            warm_cache_status,
            expected_artifact_sha256=candidate_artifact_sha256,
            expected_runtime_status_sha256=(
                _sha256(runtime_status_path) if runtime_status_path is not None else None
            ),
            expected_volume_name=(
                str(runtime_binding.get("volume_name"))
                if runtime_binding is not None
                and isinstance(runtime_binding.get("volume_name"), str)
                else None
            ),
        )
        if warm_cache_status is not None
        else None
    )
    topics = load_topics(topics_path, topic_format="tsv")
    topic_ids = [topic.id for topic in topics]
    missing_focus = set(focus_topics) - set(topic_ids)
    if missing_focus:
        raise ValueError(f"focus topics not present: {sorted(missing_focus)}")
    qrels = parse_qrels(qrels_path)
    candidate_depth = _candidate_depth(candidate_config)
    formula = _formula(candidate_config)
    component_records = build_component_records(
        baseline_rows,
        candidate_rows,
        candidate_depth=candidate_depth,
        formula=formula,
    )
    baseline_ranked = _ranked_from_stage(baseline_rows)
    shipped_ranked = _ranked_from_stage(candidate_rows)
    counterfactual_ranked = build_counterfactual_rankings(component_records)
    rankings = {
        "bm25": baseline_ranked,
        **counterfactual_ranked,
        "shipped": shipped_ranked,
    }
    evaluations = {
        system: evaluate_ranked(
            rows,
            qrels,
            metric_names=DEFAULT_METRICS,
            relevance_threshold=relevance_threshold,
            topic_ids=topic_ids,
        )
        for system, rows in rankings.items()
    }

    regressions: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    topics_by_id = {topic.id: topic for topic in topics}
    for topic_id in sorted(topic_ids, key=_topic_sort_key):
        baseline_ndcg = _metric_value(evaluations, "bm25", topic_id, "ndcg@10")
        shipped_ndcg = _metric_value(evaluations, "shipped", topic_id, "ndcg@10")
        for system in ALL_SYSTEMS:
            ndcg = _metric_value(evaluations, system, topic_id, "ndcg@10")
            counterfactual_rows.append(
                {
                    "topic_id": topic_id,
                    "topic_title": topics_by_id[topic_id].title,
                    "system": system,
                    "ndcg_at_10": ndcg,
                    "delta_vs_bm25": ndcg - baseline_ndcg,
                    "precision_at_10": _metric_value(
                        evaluations, system, topic_id, "precision@10"
                    ),
                    "relevant_count_at_10": _metric_value(
                        evaluations, system, topic_id, "relevant_count@10"
                    ),
                    "graded_recall_at_10": _metric_value(
                        evaluations, system, topic_id, "graded_recall@10"
                    ),
                }
            )
        if shipped_ndcg < baseline_ndcg - 1e-12:
            counterfactual_values = {
                system: _metric_value(evaluations, system, topic_id, "ndcg@10")
                for system in COUNTERFACTUAL_SYSTEMS
            }
            diagnostic = classify_regression(
                baseline_ndcg=baseline_ndcg,
                shipped_ndcg=shipped_ndcg,
                counterfactual_ndcgs=counterfactual_values,
            )
            regressions.append(
                {
                    "topic_id": topic_id,
                    "topic_title": topics_by_id[topic_id].title,
                    "bm25_ndcg_at_10": baseline_ndcg,
                    "shipped_ndcg_at_10": shipped_ndcg,
                    "delta_ndcg_at_10": shipped_ndcg - baseline_ndcg,
                    "bm25_precision_at_10": _metric_value(
                        evaluations, "bm25", topic_id, "precision@10"
                    ),
                    "shipped_precision_at_10": _metric_value(
                        evaluations, "shipped", topic_id, "precision@10"
                    ),
                    "bm25_relevant_count_at_10": _metric_value(
                        evaluations, "bm25", topic_id, "relevant_count@10"
                    ),
                    "shipped_relevant_count_at_10": _metric_value(
                        evaluations, "shipped", topic_id, "relevant_count@10"
                    ),
                    "coverage_counterfactual_effect": shipped_ndcg
                    - counterfactual_values["no_coverage"],
                    "counterfactual_ndcg_at_10": counterfactual_values,
                    "diagnostic": diagnostic,
                }
            )
    regressions.sort(key=lambda row: (row["delta_ndcg_at_10"], _topic_sort_key(row["topic_id"])))
    _validate_focus_regressions(focus_topics, regressions)
    degraded_topics = {str(row["topic_id"]) for row in regressions}
    calibration_rows, grade_rows = _calibration_rows(
        component_records,
        qrels,
        degraded_topics=degraded_topics,
        focus_topics=focus_topics,
        relevance_threshold=relevance_threshold,
    )
    movement_rows = _movement_rows(
        topics,
        baseline_ranked,
        shipped_ranked,
        counterfactual_ranked,
        component_records,
        qrels,
        focus_topics=focus_topics,
        relevance_threshold=relevance_threshold,
    )
    runtime = _runtime_summary(artifact_manifest, candidate_run_metadata, runtime_status)
    runtime["artifact_binding"] = runtime_binding
    judged_candidate_count = sum(
        row.docid in qrels.get(row.topic_id, {}) for row in component_records
    )
    scored_candidate_count = len(component_records)

    source_paths = {
        "baseline_ranked": baseline_ranked_path,
        "candidate_ranked": candidate_ranked_path,
        "qrels": qrels_path,
        "topics": topics_path,
        "candidate_config": candidate_config_path,
        "candidate_run_metadata": candidate_run_metadata_path,
        "artifact_manifest": artifact_manifest_path,
    }
    if runtime_status_path:
        source_paths["runtime_status"] = runtime_status_path
    if warm_cache_status_path:
        source_paths["warm_cache_status"] = warm_cache_status_path
    source_records = {name: _source_record(path) for name, path in source_paths.items()}
    source_records["generator"] = _source_record(Path(__file__))
    generated_at = datetime.now(timezone.utc).isoformat()
    reproduce_command = ".venv/bin/python -m trec_rag.topic_regression_postmortem"
    if runtime_status_path is not None:
        reproduce_command += f" --runtime-status {_portable_path(runtime_status_path)}"
    if warm_cache_status_path is not None:
        reproduce_command += (
            f" --warm-cache-status {_portable_path(warm_cache_status_path)}"
        )
    payload: dict[str, Any] = {
        "generated_at": generated_at,
        "reproduce_command": reproduce_command,
        "scope": {
            "split": "development",
            "topic_count": len(topic_ids),
            "candidate_depth": candidate_depth,
            "metric_cutoff": 10,
            "relevance_threshold": relevance_threshold,
            "focus_topics": list(focus_topics),
            "qrels_policy": "Documents absent from qrels receive grade 0.",
            "scored_candidate_count": scored_candidate_count,
            "judged_candidate_count": judged_candidate_count,
            "judged_candidate_rate": (
                judged_candidate_count / scored_candidate_count
                if scored_candidate_count
                else 0.0
            ),
        },
        "runtime": runtime,
        "warm_cache": warm_cache,
        "formula": formula,
        "sources": source_records,
        "evaluations": evaluations,
        "regressions": regressions,
        "calibration": {
            "definition": (
                "Spearman against integer qrel grade and pairwise AUC for grade >= "
                f"{relevance_threshold}, within the fixed top-{candidate_depth} pools."
            ),
            "rows": calibration_rows,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "component_counterfactuals.csv", counterfactual_rows)
    _write_csv(output_dir / "component_calibration.csv", calibration_rows)
    _write_csv(output_dir / "component_grade_summary.csv", grade_rows)
    _write_csv(output_dir / "top10_movements.csv", movement_rows)

    manifest = {
        "experiment": {
            "id": "bm25_mixedbread_topic_regression_postmortem_v1",
            "generated_at": generated_at,
            "description": (
                "Reproducible topic-level regression diagnosis with saved-score "
                "component counterfactuals."
            ),
        },
        "scope": payload["scope"],
        "formula": formula,
        "runtime": runtime,
        "warm_cache": warm_cache,
        "sources": source_records,
        "outputs": {
            "report": "notes.md",
            "metrics": "metrics.json",
            "counterfactuals": "component_counterfactuals.csv",
            "calibration": "component_calibration.csv",
            "grade_summary": "component_grade_summary.csv",
            "top10_movements": "top10_movements.csv",
        },
        "reproduce": {
            "command": reproduce_command,
            "modal_refresh_rule": (
                "After replacing score artifacts, rerun both pipeline configs and this "
                "generator; do not carry forward conclusions from prior artifacts."
            ),
        },
        "diagnostic_rule": {
            "aggregation_sensitive": (
                "At least one document, passage, or no-coverage saved-score ranking "
                "reaches BM25 nDCG@10 while shipped does not."
            ),
            "model_signal_limited": (
                "None of those saved-score rankings reaches BM25 nDCG@10."
            ),
            "claim_boundary": (
                "Labels describe controlled counterfactuals in this run and are not "
                "causal or test-set generalization claims."
            ),
        },
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(jsonable(manifest), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_dir / "notes.md").write_text(
        _report_markdown(
            payload,
            topics=topics,
            movement_rows=movement_rows,
            calibration_rows=calibration_rows,
        ),
        encoding="utf-8",
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a saved-stage BM25/reranker topic regression postmortem."
    )
    parser.add_argument(
        "--baseline-ranked",
        type=Path,
        default=Path("outputs/rag25_bm25_full_query_v1/stage_ranked.jsonl"),
    )
    parser.add_argument(
        "--candidate-ranked",
        type=Path,
        default=Path("outputs/rag25_bm25_mixedbread_rerank_v1/stage_ranked.jsonl"),
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path(
            "trec-rag-data/trec-rag-2026/development-data/"
            "rag25-dev-umbrela-qrels/"
            "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
        ),
    )
    parser.add_argument(
        "--topics",
        type=Path,
        default=Path(
            "trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"
        ),
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path("configs/rag25_bm25_mixedbread_rerank_v1.yaml"),
    )
    parser.add_argument(
        "--candidate-run-metadata",
        type=Path,
        default=Path("outputs/rag25_bm25_mixedbread_rerank_v1/run_metadata.json"),
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=Path("reports/experiments/bm25_mixedbread_config_comparison_v1/manifest.yaml"),
    )
    parser.add_argument(
        "--runtime-status",
        type=Path,
        help="Optional local/Modal runtime-status JSON used to label provider and hardware.",
    )
    parser.add_argument(
        "--warm-cache-status",
        type=Path,
        help=(
            "Optional completed Modal warm-cache verification JSON; invalid or "
            "incomplete proofs are rejected."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--focus-topics", nargs="+", default=["224", "515"])
    parser.add_argument("--relevance-threshold", type=int, default=2)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = generate_postmortem(
        baseline_ranked_path=args.baseline_ranked,
        candidate_ranked_path=args.candidate_ranked,
        qrels_path=args.qrels,
        topics_path=args.topics,
        candidate_config_path=args.candidate_config,
        candidate_run_metadata_path=args.candidate_run_metadata,
        artifact_manifest_path=args.artifact_manifest,
        runtime_status_path=args.runtime_status,
        warm_cache_status_path=args.warm_cache_status,
        output_dir=args.output_dir,
        focus_topics=args.focus_topics,
        relevance_threshold=args.relevance_threshold,
    )
    aggregate = payload["evaluations"]
    baseline = float(aggregate["bm25"]["metrics"]["ndcg@10"])
    shipped = float(aggregate["shipped"]["metrics"]["ndcg@10"])
    print(f"Wrote regression postmortem to {args.output_dir}")
    print(f"nDCG@10: {baseline:.6f} -> {shipped:.6f} ({shipped - baseline:+.6f})")
    print(f"degraded topics: {len(payload['regressions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
