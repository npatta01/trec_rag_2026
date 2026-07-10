"""Run and compare two pipeline configs at aggregate and topic grain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from trec_rag.evaluation import evaluate_ranked, parse_qrels
from trec_rag.pipeline import run_pipeline
from trec_rag.pipeline_config import PipelineConfig, load_pipeline_config
from trec_rag.pipeline_models import RetrievedCandidate, jsonable
from trec_rag.repo_env import shared_checkout_root
from trec_rag.topics import Topic, load_topics


DEFAULT_METRICS = (
    "ndcg@10",
    "judged_count@10",
    "judged_rate@10",
    "precision@10",
    "recall@10",
    "hit_rate@10",
    "relevant_count@10",
    "graded_recall@10",
    "judged_count@50",
    "judged_rate@50",
    "precision@50",
    "recall@50",
    "hit_rate@50",
    "relevant_count@50",
    "graded_recall@50",
    "ideal_dcg_coverage@50",
    "recall@100",
)


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(topic_id))
    except ValueError:
        return (1, topic_id)


def _metric_slug(metric: str) -> str:
    return metric.replace("@", "_at_").replace("-", "_")


def _status(delta: float, *, tolerance: float) -> str:
    if delta > tolerance:
        return "improved"
    if delta < -tolerance:
        return "degraded"
    return "tied"


def _require_metric_payload(payload: dict[str, object], label: str) -> None:
    if not isinstance(payload.get("metrics"), dict):
        raise ValueError(f"{label}: missing aggregate metrics")
    if not isinstance(payload.get("per_topic"), dict):
        raise ValueError(f"{label}: missing per-topic metrics")


def build_metric_comparison(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    topics: Iterable[Topic],
    metric_names: Iterable[str],
    primary_metric: str,
    baseline_id: str,
    candidate_id: str,
    big_regression_threshold: float = 0.1,
    tie_tolerance: float = 1e-12,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build an apples-to-apples comparison from two evaluation payloads."""

    _require_metric_payload(baseline, baseline_id)
    _require_metric_payload(candidate, candidate_id)
    metric_names = tuple(dict.fromkeys(metric_names))
    if not metric_names:
        raise ValueError("at least one comparison metric is required")
    if primary_metric not in metric_names:
        raise ValueError("primary metric must be included in metric_names")
    if big_regression_threshold <= 0:
        raise ValueError("big_regression_threshold must be greater than zero")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must not be negative")

    baseline_aggregate = baseline["metrics"]
    candidate_aggregate = candidate["metrics"]
    baseline_topics = baseline["per_topic"]
    candidate_topics = candidate["per_topic"]
    assert isinstance(baseline_aggregate, dict)
    assert isinstance(candidate_aggregate, dict)
    assert isinstance(baseline_topics, dict)
    assert isinstance(candidate_topics, dict)

    topic_ids = set(str(topic_id) for topic_id in baseline_topics)
    candidate_topic_ids = set(str(topic_id) for topic_id in candidate_topics)
    if topic_ids != candidate_topic_ids:
        missing = sorted(topic_ids - candidate_topic_ids, key=_topic_sort_key)
        extra = sorted(candidate_topic_ids - topic_ids, key=_topic_sort_key)
        raise ValueError(f"topic sets differ; candidate missing={missing}, extra={extra}")

    topics_by_id = {topic.id: topic for topic in topics}
    aggregate: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        if metric not in baseline_aggregate or metric not in candidate_aggregate:
            raise ValueError(f"metric {metric!r} is missing from one or both aggregate payloads")
        baseline_value = float(baseline_aggregate[metric])
        candidate_value = float(candidate_aggregate[metric])
        aggregate[metric] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": candidate_value - baseline_value,
        }

    topic_rows: list[dict[str, object]] = []
    primary_deltas: list[float] = []
    for topic_id in sorted(topic_ids, key=_topic_sort_key):
        baseline_topic = baseline_topics[topic_id]
        candidate_topic = candidate_topics[topic_id]
        if not isinstance(baseline_topic, dict) or not isinstance(candidate_topic, dict):
            raise ValueError(f"topic {topic_id}: metric payload must be a mapping")
        metric_rows: dict[str, dict[str, float | str]] = {}
        for metric in metric_names:
            if metric not in baseline_topic or metric not in candidate_topic:
                raise ValueError(f"topic {topic_id}: metric {metric!r} is missing")
            baseline_value = float(baseline_topic[metric])
            candidate_value = float(candidate_topic[metric])
            delta = candidate_value - baseline_value
            metric_rows[metric] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
                "status": _status(delta, tolerance=tie_tolerance),
            }
        primary_delta = float(metric_rows[primary_metric]["delta"])
        primary_deltas.append(primary_delta)
        topic = topics_by_id.get(topic_id)
        topic_rows.append(
            {
                "topic_id": topic_id,
                "topic_title": topic.title if topic else "",
                "primary_status": _status(primary_delta, tolerance=tie_tolerance),
                "is_big_regression": primary_delta < -big_regression_threshold,
                "metrics": metric_rows,
            }
        )

    counts = {
        status: sum(row["primary_status"] == status for row in topic_rows)
        for status in ("improved", "degraded", "tied")
    }
    counts["big_regressions"] = sum(bool(row["is_big_regression"]) for row in topic_rows)
    primary_summary: dict[str, object] = {
        "metric": primary_metric,
        "baseline": aggregate[primary_metric]["baseline"],
        "candidate": aggregate[primary_metric]["candidate"],
        "delta": aggregate[primary_metric]["delta"],
        "topic_counts": counts,
        "big_regression_threshold": -big_regression_threshold,
    }
    if primary_deltas:
        primary_summary["topic_delta_distribution"] = {
            "min": min(primary_deltas),
            "median": statistics.median(primary_deltas),
            "max": max(primary_deltas),
        }

    return {
        "comparison": {
            "baseline_id": baseline_id,
            "candidate_id": candidate_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        },
        "topic_count": len(topic_rows),
        "aggregate": aggregate,
        "primary_metric_summary": primary_summary,
        "topics": topic_rows,
    }


def write_comparison_outputs(comparison: dict[str, object], output_dir: Path) -> None:
    """Write JSON plus wide and long topic-level CSV views."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(jsonable(comparison), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    aggregate = comparison["aggregate"]
    topic_rows = comparison["topics"]
    assert isinstance(aggregate, dict)
    assert isinstance(topic_rows, list)
    metric_names = tuple(aggregate)

    wide_fields = ["topic_id", "topic_title", "primary_status", "is_big_regression"]
    for metric in metric_names:
        slug = _metric_slug(metric)
        wide_fields.extend((f"baseline_{slug}", f"candidate_{slug}", f"delta_{slug}"))
    with (output_dir / "topic_metrics.csv").open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=wide_fields, lineterminator="\n")
        writer.writeheader()
        for topic_row in topic_rows:
            metrics = topic_row["metrics"]
            assert isinstance(metrics, dict)
            row: dict[str, object] = {
                "topic_id": topic_row["topic_id"],
                "topic_title": topic_row["topic_title"],
                "primary_status": topic_row["primary_status"],
                "is_big_regression": topic_row["is_big_regression"],
            }
            for metric in metric_names:
                metric_row = metrics[metric]
                assert isinstance(metric_row, dict)
                slug = _metric_slug(metric)
                row[f"baseline_{slug}"] = metric_row["baseline"]
                row[f"candidate_{slug}"] = metric_row["candidate"]
                row[f"delta_{slug}"] = metric_row["delta"]
            writer.writerow(row)

    long_fields = [
        "topic_id",
        "topic_title",
        "metric",
        "baseline",
        "candidate",
        "delta",
        "status",
        "is_primary_metric",
        "is_big_regression",
    ]
    primary_metric = comparison["primary_metric_summary"]["metric"]
    with (output_dir / "topic_metric_deltas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as sink:
        writer = csv.DictWriter(sink, fieldnames=long_fields, lineterminator="\n")
        writer.writeheader()
        for topic_row in topic_rows:
            metrics = topic_row["metrics"]
            assert isinstance(metrics, dict)
            for metric in metric_names:
                metric_row = metrics[metric]
                assert isinstance(metric_row, dict)
                writer.writerow(
                    {
                        "topic_id": topic_row["topic_id"],
                        "topic_title": topic_row["topic_title"],
                        "metric": metric,
                        "baseline": metric_row["baseline"],
                        "candidate": metric_row["candidate"],
                        "delta": metric_row["delta"],
                        "status": metric_row["status"],
                        "is_primary_metric": metric == primary_metric,
                        "is_big_regression": (
                            topic_row["is_big_regression"] if metric == primary_metric else False
                        ),
                    }
                )


def _candidate_pool(
    rows: Iterable[RetrievedCandidate],
) -> dict[str, list[tuple[str, str, int, str, float, str, str]]]:
    pool: dict[str, list[tuple[str, str, int, str, float, str, str]]] = {}
    for row in rows:
        pool.setdefault(row.topic_id, []).append(
            (
                row.variant_name,
                row.retriever_name,
                row.rank,
                row.docid,
                row.score,
                row.query_text,
                hashlib.sha256(row.text.encode("utf-8")).hexdigest(),
            )
        )
    for topic_rows in pool.values():
        topic_rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return pool


def _validate_candidate_pools(
    baseline: Iterable[RetrievedCandidate], candidate: Iterable[RetrievedCandidate]
) -> None:
    baseline_pool = _candidate_pool(baseline)
    candidate_pool = _candidate_pool(candidate)
    if baseline_pool == candidate_pool:
        return
    topic_ids = sorted(set(baseline_pool) | set(candidate_pool), key=_topic_sort_key)
    differences = []
    for topic_id in topic_ids:
        baseline_rows = baseline_pool.get(topic_id, [])
        candidate_rows = candidate_pool.get(topic_id, [])
        if baseline_rows != candidate_rows:
            differences.append(
                f"{topic_id}: baseline={len(baseline_rows)} candidate={len(candidate_rows)}"
            )
    raise ValueError("candidate pools differ; " + "; ".join(differences[:10]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path, *roots: Path | None) -> str:
    resolved = path.resolve()
    for root in roots:
        if root is None:
            continue
        try:
            return str(resolved.relative_to(root.resolve()))
        except ValueError:
            continue
    return str(path)


def _config_metadata(config: PipelineConfig, path: Path) -> dict[str, object]:
    reranker = config.ranking.reranker
    return {
        "config": _portable_path(path, config.root_dir, shared_checkout_root(config.root_dir)),
        "config_sha256": _sha256(path),
        "experiment_id": config.run_id,
        "retrieval_hits": config.retrievers[0].hits,
        "ranking": config.ranking.type,
        "candidate_depth": reranker.candidate_depth if reranker else None,
        "model": reranker.model if reranker else None,
        "model_revision": reranker.model_revision if reranker else None,
        "backend_version": reranker.backend_version if reranker else None,
        "score_representation": reranker.score_representation if reranker else None,
        "inference_dtype": reranker.inference_dtype if reranker else None,
        "input_policy": reranker.input_policy if reranker else None,
        "artifact_schema_version": reranker.artifact_schema_version if reranker else None,
        "document_max_length": reranker.document_max_length if reranker else None,
        "document_pair_buffer_tokens": (
            reranker.document_pair_buffer_tokens if reranker else None
        ),
        "window_max_length": reranker.window_max_length if reranker else None,
        "chunk_max_characters": reranker.chunk_max_characters if reranker else None,
        "chunk_overlap_characters": (
            reranker.chunk_overlap_characters if reranker else None
        ),
    }


def compare_pipeline_configs(
    baseline_config_path: Path,
    candidate_config_path: Path,
    *,
    output_dir: Path,
    metric_names: Iterable[str] = DEFAULT_METRICS,
    primary_metric: str = "ndcg@10",
    big_regression_threshold: float = 0.1,
) -> dict[str, object]:
    """Run two configs, validate comparability, and persist per-topic deltas."""

    baseline_config_path = baseline_config_path.resolve()
    candidate_config_path = candidate_config_path.resolve()
    baseline_config = load_pipeline_config(baseline_config_path)
    candidate_config = load_pipeline_config(candidate_config_path)
    if baseline_config.evaluation is None or candidate_config.evaluation is None:
        raise ValueError("both configs must define evaluation")
    if baseline_config.evaluation.kind != candidate_config.evaluation.kind:
        raise ValueError("configs must use the same evaluation kind")
    if baseline_config.evaluation.qrels != candidate_config.evaluation.qrels:
        raise ValueError("configs must use the same qrels")
    if (
        baseline_config.evaluation.relevance_threshold
        != candidate_config.evaluation.relevance_threshold
    ):
        raise ValueError("configs must use the same relevance threshold")

    baseline_result = run_pipeline(baseline_config_path)
    candidate_result = run_pipeline(candidate_config_path)
    _validate_candidate_pools(baseline_result.retrieved, candidate_result.retrieved)

    topics = load_topics(
        baseline_config.topics.path,
        topic_format=baseline_config.topics.format,
    )
    candidate_topics = load_topics(
        candidate_config.topics.path,
        topic_format=candidate_config.topics.format,
    )
    if {topic.id for topic in topics} != {topic.id for topic in candidate_topics}:
        raise ValueError("configs must use the same topic IDs")

    topic_ids = [topic.id for topic in topics]
    qrels = parse_qrels(baseline_config.evaluation.qrels)
    metric_names = tuple(metric_names)
    baseline_metrics = evaluate_ranked(
        baseline_result.ranked,
        qrels,
        metric_names=metric_names,
        relevance_threshold=baseline_config.evaluation.relevance_threshold,
        topic_ids=topic_ids,
    )
    candidate_metrics = evaluate_ranked(
        candidate_result.ranked,
        qrels,
        metric_names=metric_names,
        relevance_threshold=candidate_config.evaluation.relevance_threshold,
        topic_ids=topic_ids,
    )

    metadata: dict[str, object] = {
        "baseline": _config_metadata(baseline_config, baseline_config_path),
        "candidate": _config_metadata(candidate_config, candidate_config_path),
        "evaluation_kind": baseline_config.evaluation.kind,
        "qrels": _portable_path(
            baseline_config.evaluation.qrels,
            baseline_config.root_dir,
            shared_checkout_root(baseline_config.root_dir),
        ),
        "qrels_sha256": _sha256(baseline_config.evaluation.qrels),
        "topics": _portable_path(
            baseline_config.topics.path,
            baseline_config.root_dir,
            shared_checkout_root(baseline_config.root_dir),
        ),
        "topics_sha256": _sha256(baseline_config.topics.path),
        "relevance_threshold": baseline_config.evaluation.relevance_threshold,
        "candidate_pool_match": True,
        "unjudged_policy": "Documents absent from qrels receive grade 0.",
        "cache": {
            "baseline": baseline_result.run_metadata["cache"],
            "candidate": candidate_result.run_metadata["cache"],
        },
    }
    comparison = build_metric_comparison(
        baseline_metrics,
        candidate_metrics,
        topics=topics,
        metric_names=metric_names,
        primary_metric=primary_metric,
        baseline_id=baseline_config.run_id,
        candidate_id=candidate_config.run_id,
        big_regression_threshold=big_regression_threshold,
        metadata=metadata,
    )
    write_comparison_outputs(comparison, output_dir)
    return comparison


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run two pipeline configs and write aggregate plus per-topic metric deltas."
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/rag25_bm25_full_query_v1.yaml"),
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path("configs/rag25_bm25_mixedbread_rerank_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/bm25_mixedbread_config_comparison_v1"),
    )
    parser.add_argument("--primary-metric", default="ndcg@10")
    parser.add_argument("--big-regression-threshold", type=float, default=0.1)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric names evaluated identically for both ranked lists.",
    )
    return parser


def _cache_hits(comparison: dict[str, object], side: str) -> tuple[int, int]:
    comparison_metadata = comparison["comparison"]
    assert isinstance(comparison_metadata, dict)
    cache = comparison_metadata["cache"]
    assert isinstance(cache, dict)
    side_cache = cache[side]
    assert isinstance(side_cache, dict)
    retrieval = side_cache["retrieval"]
    assert isinstance(retrieval, list)
    return (
        sum(int(row.get("hits", 0)) for row in retrieval if isinstance(row, dict)),
        sum(int(row.get("requests", 0)) for row in retrieval if isinstance(row, dict)),
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    comparison = compare_pipeline_configs(
        args.baseline_config,
        args.candidate_config,
        output_dir=args.output_dir,
        metric_names=args.metrics,
        primary_metric=args.primary_metric,
        big_regression_threshold=args.big_regression_threshold,
    )
    primary = comparison["primary_metric_summary"]
    assert isinstance(primary, dict)
    counts = primary["topic_counts"]
    assert isinstance(counts, dict)
    baseline_hits, baseline_requests = _cache_hits(comparison, "baseline")
    candidate_hits, candidate_requests = _cache_hits(comparison, "candidate")
    print(f"Wrote comparison outputs to {args.output_dir}")
    print(
        f"{primary['metric']}: {float(primary['baseline']):.6f} -> "
        f"{float(primary['candidate']):.6f} ({float(primary['delta']):+.6f})"
    )
    print(
        "topics: "
        f"{counts['improved']} improved, {counts['degraded']} degraded, {counts['tied']} tied"
    )
    print(
        "retrieval cache: "
        f"baseline {baseline_hits}/{baseline_requests} hits; "
        f"candidate {candidate_hits}/{candidate_requests} hits"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
