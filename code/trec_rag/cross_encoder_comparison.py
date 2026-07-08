"""Build durable cross-encoder comparison records from local eval artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SYSTEM_SCORE_FIELDS = [
    "system_id",
    "model",
    "method",
    "candidate_depth",
    "input_limit",
    "document_count",
    "chunk_count",
    "fit_count",
    "truncated_count",
    "ndcg_at_10",
    "delta_vs_bm25",
    "relative_lift_vs_bm25",
    "topic_losses",
    "big_topic_losses",
    "worst_topic",
    "worst_delta",
    "source_artifact",
    "notes",
]

TOPIC_SYSTEM_SCORE_FIELDS = [
    "system_id",
    "model",
    "method",
    "topic_id",
    "bm25_ndcg_at_10",
    "system_ndcg_at_10",
    "delta_vs_bm25",
    "source_artifact",
]

PROMPT_PROBE_FIELDS = [
    "model",
    "instruction",
    "topic_count",
    "document_count",
    "mean_spearman",
    "worst_topic",
    "worst_spearman",
    "source_artifact",
]

DEFAULT_FULL_DEV_ARTIFACTS = [
    Path("tmp/qwen_aggregation_eval_hits20_Qwen3_Reranker_0p6B.json"),
    Path("tmp/qwen_aggregation_eval_hits20_Qwen3_Reranker_4B.json"),
    Path("tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B.json"),
    Path("tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_ov0.json"),
    Path("tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap2000.json"),
    Path("tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_0p6B_gap3000.json"),
    Path("tmp/qwen_aggregation_eval_hits50_Qwen3_Reranker_4B.json"),
    Path("tmp/qwen_full_doc_margin_hits50_Qwen3_Reranker_0p6B_ml32768.json"),
    Path("tmp/seqcls_eval_hits50_BAAI__bge_reranker_v2_m3_ml1024.json"),
    Path("tmp/seqcls_eval_hits50_cross_encoder__ms_marco_MiniLM_L6_v2_ml512.json"),
    Path("tmp/st_crossencoder_eval_hits50_mixedbread_ai__mxbai_rerank_base_v2_ml1024.json"),
    Path("tmp/st_crossencoder_longctx_hits50_mixedbread_ai__mxbai_rerank_base_v2_ctx32768_buf512.json"),
    Path("tmp/st_chunk_eval_hits50_mixedbread_ai__mxbai_rerank_base_v2_ml1024_cm3500_ov350.json"),
]

DEFAULT_PROMPT_PROBE_ARTIFACTS = [
    Path("tmp/qwen_rerank_prompt_probe.json"),
    Path("tmp/qwen_rerank_probe_results.json"),
]

DEFAULT_OUTPUT_DIR = Path("reports/experiments/cross_encoder_model_comparison_v1")


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return raw


def _fmt(value: float) -> str:
    return f"{value:.10f}"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug or "unknown"


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    if topic_id.isdigit():
        return (0, int(topic_id))
    return (1, topic_id)


def _baseline(data: dict[str, Any]) -> dict[str, Any]:
    metrics = data.get("metrics") or {}
    baseline = metrics.get("bm25")
    if not isinstance(baseline, dict):
        raise ValueError("artifact is missing metrics.bm25")
    return baseline


def _ndcg(system: dict[str, Any]) -> float:
    return float((system.get("metrics") or {})["ndcg@10"])


def _input_limit(data: dict[str, Any]) -> str:
    return _text(data.get("max_length") or data.get("context_length") or data.get("max_characters"))


def _fit_count(data: dict[str, Any]) -> str:
    return _text(data.get("fit_count") or data.get("full_doc_fit_count"))


def _truncated_count(data: dict[str, Any]) -> str:
    return _text(data.get("truncated_count") or data.get("full_doc_too_long_count"))


def _per_topic_deltas(
    baseline: dict[str, Any], system: dict[str, Any]
) -> list[tuple[str, float, float, float]]:
    baseline_topics = baseline.get("per_topic") or {}
    system_topics = system.get("per_topic") or {}
    deltas: list[tuple[str, float, float, float]] = []
    for topic_id in sorted(set(baseline_topics) & set(system_topics), key=_topic_sort_key):
        bm25_ndcg = float(baseline_topics[topic_id]["ndcg@10"])
        system_ndcg = float(system_topics[topic_id]["ndcg@10"])
        deltas.append((topic_id, bm25_ndcg, system_ndcg, system_ndcg - bm25_ndcg))
    return deltas


def _summary(deltas: list[tuple[str, float, float, float]]) -> dict[str, Any]:
    if not deltas:
        return {
            "topic_losses": 0,
            "big_topic_losses": 0,
            "worst_topic": "",
            "worst_delta": 0.0,
        }
    worst_topic, _, _, worst_delta = min(deltas, key=lambda row: (row[3], _topic_sort_key(row[0])))
    return {
        "topic_losses": sum(1 for _, _, _, delta in deltas if delta < -1e-12),
        "big_topic_losses": sum(1 for _, _, _, delta in deltas if delta < -0.1),
        "worst_topic": worst_topic,
        "worst_delta": worst_delta,
    }


def _notes(data: dict[str, Any]) -> str:
    instruction = data.get("instruction")
    method = data.get("method")
    notes = []
    if instruction:
        notes.append(f"instruction={instruction}")
    if method:
        notes.append(f"runtime={method}")
    return "; ".join(notes)


def _artifact_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], float]:
    data = _read_json(path)
    baseline = _baseline(data)
    bm25_ndcg = _ndcg(baseline)
    system_rows: list[dict[str, str]] = []
    topic_rows: list[dict[str, str]] = []

    for method, system in (data.get("metrics") or {}).items():
        if method == "bm25":
            continue
        ndcg = _ndcg(system)
        delta = ndcg - bm25_ndcg
        relative_lift = delta / bm25_ndcg if bm25_ndcg else 0.0
        model = _text(data.get("model"))
        system_id = f"{_slug(model)}__{_slug(method)}__{_slug(path.stem)}"
        deltas = _per_topic_deltas(baseline, system)
        summary = _summary(deltas)
        system_rows.append(
            {
                "system_id": system_id,
                "model": model,
                "method": method,
                "candidate_depth": _text(data.get("hits")),
                "input_limit": _input_limit(data),
                "document_count": _text(data.get("document_count")),
                "chunk_count": _text(data.get("chunk_count")),
                "fit_count": _fit_count(data),
                "truncated_count": _truncated_count(data),
                "ndcg_at_10": _fmt(ndcg),
                "delta_vs_bm25": _fmt(delta),
                "relative_lift_vs_bm25": _fmt(relative_lift),
                "topic_losses": _text(summary["topic_losses"]),
                "big_topic_losses": _text(summary["big_topic_losses"]),
                "worst_topic": _text(summary["worst_topic"]),
                "worst_delta": _fmt(float(summary["worst_delta"])),
                "source_artifact": str(path),
                "notes": _notes(data),
            }
        )
        for topic_id, bm25_topic_ndcg, system_topic_ndcg, topic_delta in deltas:
            topic_rows.append(
                {
                    "system_id": system_id,
                    "model": model,
                    "method": method,
                    "topic_id": topic_id,
                    "bm25_ndcg_at_10": _fmt(bm25_topic_ndcg),
                    "system_ndcg_at_10": _fmt(system_topic_ndcg),
                    "delta_vs_bm25": _fmt(topic_delta),
                    "source_artifact": str(path),
                }
            )

    return system_rows, topic_rows, bm25_ndcg


def collect_full_dev_rows(paths: Iterable[Path]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Collect one system row per artifact method and one topic row per system/topic."""
    system_rows: list[dict[str, str]] = []
    topic_rows: list[dict[str, str]] = []
    for path in paths:
        rows, topics, _ = _artifact_rows(path)
        system_rows.extend(rows)
        topic_rows.extend(topics)
    system_rows.sort(key=lambda row: float(row["ndcg_at_10"]), reverse=True)
    topic_rows.sort(key=lambda row: (row["system_id"], _topic_sort_key(row["topic_id"])))
    return system_rows, topic_rows


def _rankdata(values: list[float]) -> list[float]:
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


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_var * right_var)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _spearman(grades: list[float], scores: list[float]) -> float:
    return _pearson(_rankdata(grades), _rankdata(scores))


def collect_prompt_probe_rows(path: Path) -> list[dict[str, str]]:
    """Summarize Qwen prompt/probe rows by rank correlation with qrel grades."""
    if not path.exists():
        return []
    data = _read_json(path)
    rows: list[dict[str, str]] = []
    for model_result in data.get("results") or []:
        model = _text(model_result.get("model"))
        instruction_sets = model_result.get("instructions")
        if not isinstance(instruction_sets, dict):
            instruction_sets = {"probe_default": model_result.get("chunk_rows") or []}
        for instruction, scored_docs in instruction_sets.items():
            by_topic: dict[str, list[tuple[float, float]]] = {}
            for row in scored_docs or []:
                topic_id = _text(row.get("topic_id"))
                if not topic_id:
                    continue
                by_topic.setdefault(topic_id, []).append(
                    (float(row.get("qrel_grade", 0)), float(row.get("score", 0.0)))
                )
            topic_scores = []
            for topic_id, pairs in sorted(by_topic.items(), key=lambda item: _topic_sort_key(item[0])):
                grades = [grade for grade, _ in pairs]
                scores = [score for _, score in pairs]
                topic_scores.append((topic_id, _spearman(grades, scores)))
            if topic_scores:
                worst_topic, worst_score = min(
                    topic_scores, key=lambda item: (item[1], _topic_sort_key(item[0]))
                )
                mean_score = sum(score for _, score in topic_scores) / len(topic_scores)
            else:
                worst_topic, worst_score, mean_score = "", 0.0, 0.0
            rows.append(
                {
                    "model": model,
                    "instruction": _text(instruction),
                    "topic_count": _text(len(topic_scores)),
                    "document_count": _text(sum(len(pairs) for pairs in by_topic.values())),
                    "mean_spearman": _fmt(mean_score),
                    "worst_topic": worst_topic,
                    "worst_spearman": _fmt(worst_score),
                    "source_artifact": str(path),
                }
            )
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _baseline_ndcg(paths: Iterable[Path]) -> float:
    for path in paths:
        if path.exists():
            _, _, bm25_ndcg = _artifact_rows(path)
            return bm25_ndcg
    return 0.0


def _float_row(row: dict[str, str]) -> dict[str, Any]:
    numeric_fields = {
        "candidate_depth",
        "input_limit",
        "document_count",
        "chunk_count",
        "fit_count",
        "truncated_count",
        "topic_count",
        "ndcg_at_10",
        "delta_vs_bm25",
        "relative_lift_vs_bm25",
        "topic_losses",
        "big_topic_losses",
        "worst_delta",
        "mean_spearman",
        "worst_spearman",
    }
    converted: dict[str, Any] = {}
    for key, value in row.items():
        if key in numeric_fields and value != "":
            if "." in value or key.endswith("delta") or key in {
                "ndcg_at_10",
                "delta_vs_bm25",
                "relative_lift_vs_bm25",
                "worst_delta",
            }:
                converted[key] = float(value)
            else:
                converted[key] = int(value)
        else:
            converted[key] = value
    return converted


def _write_metrics(
    path: Path,
    baseline_ndcg: float,
    system_rows: list[dict[str, str]],
    topic_rows: list[dict[str, str]],
    prompt_rows: list[dict[str, str]],
) -> None:
    best = _float_row(system_rows[0]) if system_rows else {}
    topic_count = len({row["topic_id"] for row in topic_rows})
    candidate_depths = sorted(
        {int(row["candidate_depth"]) for row in system_rows if row["candidate_depth"]}
    )
    metrics = {
        "kind": "dev_projected_qrels_cross_encoder_model_comparison",
        "topic_count": topic_count,
        "candidate_depths": candidate_depths,
        "baseline": {
            "name": "BM25 candidate order",
            "ndcg@10": baseline_ndcg,
        },
        "definitions": {
            "topic_loss": "A topic-level nDCG@10 delta below 0 versus BM25.",
            "big_topic_loss": "A topic-level nDCG@10 delta below -0.1 versus BM25.",
            "prompt_probe_mean_spearman": (
                "Mean per-topic Spearman correlation between probe scores and qrel grades."
            ),
        },
        "system_count": len(system_rows),
        "topic_system_row_count": len(topic_rows),
        "best_full_dev_system": {
            "system_id": best.get("system_id", ""),
            "model": best.get("model", ""),
            "method": best.get("method", ""),
            "ndcg@10": best.get("ndcg_at_10", 0.0),
            "delta_vs_bm25": best.get("delta_vs_bm25", 0.0),
            "topic_losses": best.get("topic_losses", 0),
            "big_topic_losses": best.get("big_topic_losses", 0),
            "worst_topic": best.get("worst_topic", ""),
            "worst_delta": best.get("worst_delta", 0.0),
        },
        "full_dev_results": [_float_row(row) for row in system_rows],
        "prompt_probe_results": [_float_row(row) for row in prompt_rows],
        "not_full_dev": [
            {
                "model": "Qwen/Qwen3-Reranker-8B",
                "status": "qualitative_probe_only",
                "source_artifact": "tmp/qwen_rerank_probe_results.json",
                "note": "No 22-topic full-dev Qwen 8B artifact is present; only the 12-document probe was run.",
            }
        ],
    }
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _write_notes(
    path: Path,
    system_rows: list[dict[str, str]],
    prompt_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# Cross-Encoder Model Comparison",
        "",
        "Run date: 2026-07-08",
        "",
        "This record promotes the cross-encoder and reranker scratch results out of",
        "`tmp/` into a durable experiment report. All full-dev rows use the same",
        "22-topic projected-qrels evaluation and BM25 candidate-order baseline;",
        "the `candidate_depth` column preserves whether a run reranked top-20 or",
        "top-50 candidates.",
        "",
        "## Full-Dev Ranking Results",
        "",
        "| model | method | nDCG@10 | delta vs BM25 | losses | big losses | worst topic | source |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in system_rows:
        lines.append(
            "| {model} | `{method}` | {ndcg} | {delta} | {losses} | {big_losses} | "
            "{worst_topic}: {worst_delta} | `{source}` |".format(
                model=row["model"],
                method=row["method"],
                ndcg=row["ndcg_at_10"],
                delta=row["delta_vs_bm25"],
                losses=row["topic_losses"],
                big_losses=row["big_topic_losses"],
                worst_topic=row["worst_topic"],
                worst_delta=row["worst_delta"],
                source=row["source_artifact"],
            )
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "The best raw model family in these artifacts is Mixedbread. The strongest ",
            "single artifact row is the Mixedbread `chunk_top4_weighted` window aggregate ",
            "at `0.539943` nDCG@10, but it still has one large topic regression. The PR's ",
            "recommended production-facing score is captured separately as the ",
            "`coverage_aware_long_doc_aggregate` config because it trades some average ",
            "score for zero topic losses worse than `-0.1` on this dev sample.",
            "",
            "Qwen did improve over BM25 on average, but the full-dev Qwen 0.6B/4B rows ",
            "show materially smaller gains and larger worst-topic losses than Mixedbread. ",
            "The available Qwen 8B evidence is qualitative only: a 12-document probe in ",
            "`tmp/qwen_rerank_probe_results.json`, not a 22-topic full-dev run.",
            "",
            "## Prompt And Probe Results",
            "",
            "The prompt/probe table is not comparable to full-dev nDCG. It reports mean ",
            "per-topic Spearman correlation between model scores and qrel grades on a ",
            "small hard-example probe.",
            "",
            "| model | instruction | docs | topics | mean Spearman | worst topic | source |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in prompt_rows:
        lines.append(
            "| {model} | `{instruction}` | {docs} | {topics} | {mean} | {worst}: {worst_score} | `{source}` |".format(
                model=row["model"],
                instruction=row["instruction"],
                docs=row["document_count"],
                topics=row["topic_count"],
                mean=row["mean_spearman"],
                worst=row["worst_topic"],
                worst_score=row["worst_spearman"],
                source=row["source_artifact"],
            )
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `system_scores.csv`: one row per full-dev model/method result.",
            "- `topic_system_scores.csv`: one row per topic and full-dev system.",
            "- `prompt_probe_scores.csv`: qualitative prompt/probe alignment summary.",
            "- `metrics.json`: machine-readable copy of the same summary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(
    path: Path,
    full_dev_artifacts: list[Path],
    prompt_probe_artifacts: list[Path],
) -> None:
    source_lines = "\n".join(f"    - {artifact}" for artifact in full_dev_artifacts)
    prompt_lines = "\n".join(f"    - {artifact}" for artifact in prompt_probe_artifacts)
    manifest = f"""experiment:
  id: cross_encoder_model_comparison_v1
  run_date: 2026-07-08
  split: dev
  description: Durable capture of full-dev cross-encoder model comparisons and Qwen prompt probes.

scope:
  topic_count: 22
  candidate_depths:
    - 20
    - 50
  baseline: BM25 candidate order
  metric: ndcg@10
  evaluation_kind: dev_projected_qrels
  relevance_threshold: 2

tracked_record_files:
  notes: notes.md
  metrics: metrics.json
  system_scores: system_scores.csv
  topic_system_scores: topic_system_scores.csv
  prompt_probe_scores: prompt_probe_scores.csv

source_artifacts:
  full_dev:
{source_lines}
  prompt_probes:
{prompt_lines}

metrics:
  see: metrics.json
"""
    path.write_text(manifest, encoding="utf-8")


def write_comparison_report(
    full_dev_artifacts: Iterable[Path],
    output_dir: Path,
    prompt_probe_artifacts: Iterable[Path] | None = None,
) -> None:
    full_dev_artifact_list = [path for path in full_dev_artifacts if path.exists()]
    prompt_artifact_list = [path for path in (prompt_probe_artifacts or []) if path.exists()]
    system_rows, topic_rows = collect_full_dev_rows(full_dev_artifact_list)
    prompt_rows: list[dict[str, str]] = []
    for artifact in prompt_artifact_list:
        prompt_rows.extend(collect_prompt_probe_rows(artifact))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "system_scores.csv", SYSTEM_SCORE_FIELDS, system_rows)
    _write_csv(output_dir / "topic_system_scores.csv", TOPIC_SYSTEM_SCORE_FIELDS, topic_rows)
    _write_csv(output_dir / "prompt_probe_scores.csv", PROMPT_PROBE_FIELDS, prompt_rows)
    _write_metrics(
        output_dir / "metrics.json",
        _baseline_ndcg(full_dev_artifact_list),
        system_rows,
        topic_rows,
        prompt_rows,
    )
    _write_notes(output_dir / "notes.md", system_rows, prompt_rows)
    _write_manifest(output_dir / "manifest.yaml", full_dev_artifact_list, prompt_artifact_list)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write cross-encoder model comparison report.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--full-dev-artifact", type=Path, action="append", dest="full_dev_artifacts")
    parser.add_argument("--prompt-probe-artifact", type=Path, action="append", dest="prompt_probe_artifacts")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    write_comparison_report(
        args.full_dev_artifacts or DEFAULT_FULL_DEV_ARTIFACTS,
        args.output_dir,
        args.prompt_probe_artifacts or DEFAULT_PROMPT_PROBE_ARTIFACTS,
    )
    print(f"Wrote cross-encoder comparison report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
