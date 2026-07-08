"""Experiment record indexing helpers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


RUN_FIELDS = [
    "experiment_id",
    "runtime_id",
    "run_date",
    "split",
    "topics",
    "topic_count",
    "retriever",
    "index",
    "query_source",
    "hits",
    "ranking",
    "evaluation_qrels",
    "record_dir",
    "output_dir",
    "cache_dir",
    "runfile_rows",
    "retrieved_rows",
    "rag_rows",
    "ndcg_at_10",
    "recall_at_100",
    "notes",
]

TOPIC_SCORE_FIELDS = [
    "experiment_id",
    "runtime_id",
    "run_date",
    "split",
    "hits",
    "topic_id",
    "ndcg_at_10",
    "recall_at_100",
]


def _basename(path_text: str) -> str:
    return Path(path_text).name


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _read_manifest(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: manifest must be a mapping")
    return raw


def _run_row(experiments_dir: Path, record_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    experiment = manifest.get("experiment") or {}
    config = manifest.get("config") or {}
    data = manifest.get("data") or {}
    scope = manifest.get("scope") or {}
    artifacts = manifest.get("local_artifacts") or {}
    counts = manifest.get("counts") or {}
    metrics = manifest.get("metrics") or {}
    candidate_depths = scope.get("candidate_depths") or []
    hits = config.get("hits") or scope.get("candidate_depth")
    if hits is None and candidate_depths:
        hits = ",".join(str(depth) for depth in candidate_depths)
    return {
        "experiment_id": _text(experiment.get("id")),
        "runtime_id": _text(experiment.get("runtime_id")),
        "run_date": _text(experiment.get("run_date")),
        "split": _text(experiment.get("split")),
        "topics": _basename(_text(data.get("topics"))),
        "topic_count": _text(data.get("topic_count") or scope.get("topic_count")),
        "retriever": _text(config.get("retriever")),
        "index": _text(config.get("index")),
        "query_source": _text(config.get("query_source")),
        "hits": _text(hits),
        "ranking": _text(config.get("ranking")),
        "evaluation_qrels": _basename(_text(data.get("qrels"))),
        "record_dir": str(record_dir.relative_to(experiments_dir)),
        "output_dir": _text(artifacts.get("output_dir")),
        "cache_dir": _text(artifacts.get("cache_dir")),
        "runfile_rows": _text(counts.get("runfile_rows")),
        "retrieved_rows": _text(counts.get("retrieved_rows")),
        "rag_rows": _text(counts.get("rag_rows")),
        "ndcg_at_10": _text(metrics.get("ndcg_at_10")),
        "recall_at_100": _text(metrics.get("recall_at_100")),
        "notes": _text(manifest.get("notes")),
    }


def _read_topic_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    missing = set(TOPIC_SCORE_FIELDS) - set(rows[0].keys() if rows else TOPIC_SCORE_FIELDS)
    if missing:
        raise ValueError(f"{path}: missing topic score fields: {', '.join(sorted(missing))}")
    return [{field: row.get(field, "") for field in TOPIC_SCORE_FIELDS} for row in rows]


def build_experiment_indexes(experiments_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    run_rows: list[dict[str, str]] = []
    topic_score_rows: list[dict[str, str]] = []
    for manifest_path in sorted(experiments_dir.glob("*/manifest.yaml")):
        record_dir = manifest_path.parent
        manifest = _read_manifest(manifest_path)
        run_rows.append(_run_row(experiments_dir, record_dir, manifest))
        topic_scores_path = record_dir / "topic_scores.csv"
        if topic_scores_path.exists():
            topic_score_rows.extend(_read_topic_scores(topic_scores_path))
    return run_rows, topic_score_rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_indexes(experiments_dir: Path) -> None:
    run_rows, topic_score_rows = build_experiment_indexes(experiments_dir)
    _write_csv(experiments_dir / "runs.csv", RUN_FIELDS, run_rows)
    _write_csv(experiments_dir / "topic_scores.csv", TOPIC_SCORE_FIELDS, topic_score_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate experiment tracker indexes.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("reports/experiments"),
        help="Directory containing per-experiment record folders.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    write_experiment_indexes(args.experiments_dir)
    print(f"Wrote indexes under {args.experiments_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
