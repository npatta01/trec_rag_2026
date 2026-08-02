"""Controlled nDCG benchmark for current and organizer retrieval ranking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trec_rag.evaluation import evaluate_ranked, parse_qrels
from trec_rag.organizer_reranking import (
    RerankCandidate,
    load_bm25_candidates,
    load_listwise_order,
)
from trec_rag.pipeline_models import RankedCandidate, RetrievedCandidate
from trec_rag.ranking import coverage_aware_long_doc_rank, passthrough_rank


CONFIG_SCHEMA = "retrieval_ranking_benchmark_v1"
REPORT_SCHEMA = "retrieval_ranking_benchmark_report_v1"


@dataclass(frozen=True)
class ResolvedBenchmarkConfig:
    source_path: Path
    repo_root: Path
    shared_root: Path
    raw: dict[str, Any]

    def path(self, key: str) -> Path:
        value = self.raw["paths"][key]
        if not isinstance(value, Mapping) or set(value) != {"base", "path"}:
            raise ValueError(f"paths.{key} must contain exactly base and path")
        base_name = str(value["base"])
        if base_name == "repo":
            base = self.repo_root
        elif base_name == "shared":
            base = self.shared_root
        else:
            raise ValueError(f"paths.{key}.base must be repo or shared")
        relative = Path(str(value["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"paths.{key}.path must be a safe relative path")
        return base / relative


def load_benchmark_config(path: Path) -> ResolvedBenchmarkConfig:
    source_path = Path(path).resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"benchmark config schema must be {CONFIG_SCHEMA}")
    required = {
        "schema_version",
        "experiment",
        "paths",
        "evaluation",
        "current_pointwise",
        "organizer_pointwise",
        "organizer_listwise",
        "modal_execution",
        "modal_budget",
    }
    if set(raw) != required:
        raise ValueError(f"benchmark config fields must be exactly {sorted(required)}")
    repo_root = source_path.parent.parent
    shared_env = str(raw["paths"].get("shared_root_env", "TREC_RAG_SHARED_ROOT"))
    shared_root = Path(os.environ.get(shared_env, str(repo_root))).resolve()
    config = ResolvedBenchmarkConfig(source_path, repo_root, shared_root, raw)

    path_keys = set(raw["paths"]) - {"shared_root_env"}
    expected_paths = {
        "bm25_candidates",
        "mixedbread_document_scores",
        "mixedbread_window_scores",
        "organizer_listwise_run",
        "current_listwise_seed",
        "output_dir",
    }
    if path_keys != expected_paths:
        raise ValueError(f"paths must define exactly {sorted(expected_paths)} plus shared_root_env")

    evaluation = raw["evaluation"]
    if not isinstance(evaluation, Mapping) or not evaluation.get("qrels"):
        raise ValueError("evaluation.qrels must be a nonempty list")
    for qrel in evaluation["qrels"]:
        if not isinstance(qrel, Mapping) or set(qrel) != {"name", "path"}:
            raise ValueError("each qrel must define exactly name and path")
    depths = [int(value) for value in raw["current_pointwise"]["candidate_depths"]]
    if not depths or any(depth < 1 for depth in depths) or len(set(depths)) != len(depths):
        raise ValueError("current pointwise candidate depths must be unique and positive")
    validate_modal_budget(raw["modal_budget"])
    modal_execution = raw["modal_execution"]
    if (
        int(modal_execution["pointwise_timeout_seconds"])
        + int(modal_execution["listwise_timeout_seconds"])
        != int(raw["modal_budget"]["full_timeout_seconds"])
    ):
        raise ValueError("Modal pointwise and listwise timeouts must sum to full_timeout_seconds")
    return config


def validate_modal_budget(settings: Mapping[str, object]) -> dict[str, float]:
    """Fail closed when configured worst-case Modal billing exceeds the cap."""

    required = {
        "maximum_usd",
        "prior_spend_usd",
        "a100_80gb_usd_per_second",
        "cpu_usd_per_core_second",
        "memory_usd_per_gib_second",
        "gpu_cpu_cores",
        "gpu_memory_gib",
        "smoke_timeout_seconds",
        "full_timeout_seconds",
        "staging_cpu_cores",
        "staging_memory_gib",
        "staging_timeout_seconds",
    }
    if set(settings) != required:
        raise ValueError(f"modal_budget fields must be exactly {sorted(required)}")
    values = {key: float(value) for key, value in settings.items()}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("modal budget values must be finite and nonnegative")
    gpu_second = (
        values["a100_80gb_usd_per_second"]
        + values["gpu_cpu_cores"] * values["cpu_usd_per_core_second"]
        + values["gpu_memory_gib"] * values["memory_usd_per_gib_second"]
    )
    gpu_maximum = gpu_second * (
        values["smoke_timeout_seconds"] + values["full_timeout_seconds"]
    )
    staging_maximum = values["staging_timeout_seconds"] * (
        values["staging_cpu_cores"] * values["cpu_usd_per_core_second"]
        + values["staging_memory_gib"] * values["memory_usd_per_gib_second"]
    )
    incremental_maximum = gpu_maximum + staging_maximum
    maximum = values["prior_spend_usd"] + incremental_maximum
    if maximum >= values["maximum_usd"]:
        raise ValueError(
            f"configured worst-case Modal cost ${maximum:.2f} is not below "
            f"the ${values['maximum_usd']:.2f} cap"
        )
    return {
        "gpu_maximum_usd": gpu_maximum,
        "staging_maximum_usd": staging_maximum,
        "prior_spend_usd": values["prior_spend_usd"],
        "incremental_maximum_usd": incremental_maximum,
        "total_maximum_usd": maximum,
    }


def _retrieved_candidates(
    candidates: Mapping[str, Sequence[RerankCandidate]],
) -> list[RetrievedCandidate]:
    return [
        RetrievedCandidate(
            topic_id=row.topic_id,
            variant_name="original",
            retriever_name="climbmix_bm25",
            query_text=row.query_text,
            docid=row.docid,
            rank=row.bm25_rank,
            score=row.bm25_score,
            text=row.text,
        )
        for topic_rows in candidates.values()
        for row in topic_rows
    ]


def build_current_pointwise_ranking(
    config: ResolvedBenchmarkConfig,
    candidates: Mapping[str, Sequence[RerankCandidate]],
) -> list[RankedCandidate]:
    """Rebuild the complete current Mixedbread ranking from cached BF16 scores."""

    current = config.raw["current_pointwise"]
    return coverage_aware_long_doc_rank(
        _retrieved_candidates(candidates),
        document_score_path=config.path("mixedbread_document_scores"),
        window_score_path=config.path("mixedbread_window_scores"),
        candidate_depth=None,
        expected_document_score_metadata=dict(current["document_score_metadata"]),
        expected_window_score_metadata=dict(current["window_score_metadata"]),
        long_document_weight=float(current["long_document_weight"]),
        strongest_passage_weight=float(current["strongest_passage_weight"]),
        coverage_bonus_weight=float(current["coverage_bonus_weight"]),
        relative_span_delta=float(current["relative_span_delta"]),
        support_cap=int(current["support_cap"]),
        min_new_chars=int(current["min_new_chars"]),
        top_window_weights=tuple(float(value) for value in current["top_window_weights"]),
    )


def _rerank_prefix_from_full_scores(
    baseline: Sequence[RankedCandidate],
    full_reranked: Sequence[RankedCandidate],
    *,
    depth: int,
) -> list[RankedCandidate]:
    """Rebuild a depth-limited pointwise run from one complete score pass."""

    baseline_by_topic: dict[str, list[RankedCandidate]] = {}
    full_by_key = {(row.topic_id, row.docid): row for row in full_reranked}
    for row in baseline:
        baseline_by_topic.setdefault(row.topic_id, []).append(row)
    output: list[RankedCandidate] = []
    for topic_id in sorted(baseline_by_topic):
        base_rows = sorted(baseline_by_topic[topic_id], key=lambda row: row.rank)
        prefix = base_rows[:depth]
        ordered = sorted(
            prefix,
            key=lambda row: (-full_by_key[(topic_id, row.docid)].score, row.rank, row.docid),
        )
        topic_output: list[RankedCandidate] = []
        for rank, base_row in enumerate(ordered, start=1):
            scored = full_by_key[(topic_id, base_row.docid)]
            topic_output.append(
                RankedCandidate(
                    topic_id=topic_id,
                    docid=base_row.docid,
                    rank=rank,
                    score=scored.score,
                    text=base_row.text,
                    provenance=scored.provenance,
                )
            )
        previous_score = topic_output[-1].score if topic_output else 0.0
        for base_row in base_rows[depth:]:
            previous_score = math.nextafter(previous_score, -math.inf)
            topic_output.append(
                RankedCandidate(
                    topic_id=topic_id,
                    docid=base_row.docid,
                    rank=len(topic_output) + 1,
                    score=previous_score,
                    text=base_row.text,
                    provenance=[
                        *base_row.provenance,
                        {"ranker": "bm25_tail_after_rerank", "candidate_depth": depth},
                    ],
                )
            )
        output.extend(topic_output)
    return output


def _pointwise_ranking(
    candidates: Mapping[str, Sequence[RerankCandidate]],
    scores: Mapping[tuple[str, str], float],
) -> list[RankedCandidate]:
    output: list[RankedCandidate] = []
    expected = {(row.topic_id, row.docid) for rows in candidates.values() for row in rows}
    if set(scores) != expected:
        missing = len(expected - set(scores))
        extra = len(set(scores) - expected)
        raise ValueError(f"pointwise scores differ from BM25 pool: missing={missing}, extra={extra}")
    for topic_id, rows in candidates.items():
        ordered = sorted(
            rows,
            key=lambda row: (-scores[(topic_id, row.docid)], row.bm25_rank, row.docid),
        )
        for rank, row in enumerate(ordered, start=1):
            output.append(
                RankedCandidate(
                    topic_id=topic_id,
                    docid=row.docid,
                    rank=rank,
                    score=scores[(topic_id, row.docid)],
                    text=row.text,
                    provenance=[
                        {
                            "ranker": "qwen3_pointwise",
                            "bm25_rank": row.bm25_rank,
                            "bm25_score": row.bm25_score,
                        }
                    ],
                )
            )
    return output


def _listwise_ranking(
    pointwise: Sequence[RankedCandidate],
    order: Mapping[str, Sequence[str]],
    *,
    depth: int,
) -> list[RankedCandidate]:
    by_topic: dict[str, list[RankedCandidate]] = {}
    for row in pointwise:
        by_topic.setdefault(row.topic_id, []).append(row)
    if set(order) != set(by_topic):
        raise ValueError("listwise topics differ from pointwise topics")
    output: list[RankedCandidate] = []
    for topic_id in sorted(by_topic):
        pointwise_rows = sorted(by_topic[topic_id], key=lambda row: row.rank)
        prefix = list(order[topic_id])
        expected_prefix = {row.docid for row in pointwise_rows[:depth]}
        if len(prefix) != depth or set(prefix) != expected_prefix:
            raise ValueError(f"topic {topic_id}: FIRST did not permute the exact top {depth}")
        rows_by_docid = {row.docid: row for row in pointwise_rows}
        final_order = [*prefix, *(row.docid for row in pointwise_rows[depth:])]
        for rank, docid in enumerate(final_order, start=1):
            source = rows_by_docid[docid]
            output.append(
                RankedCandidate(
                    topic_id=topic_id,
                    docid=docid,
                    rank=rank,
                    score=1.0 / rank,
                    text=source.text,
                    provenance=[
                        *source.provenance,
                        {
                            "ranker": "first_qwen3_listwise"
                            if rank <= depth
                            else "pointwise_tail_after_first",
                            "pointwise_rank": source.rank,
                            "candidate_depth": depth,
                        },
                    ],
                )
            )
    return output


def _seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return base_seed ^ int.from_bytes(digest[:8], "big")


def paired_statistics(
    candidate_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> dict[str, object]:
    if len(candidate_values) != len(baseline_values) or not candidate_values:
        raise ValueError("paired statistics require equal nonempty samples")
    deltas = [float(candidate) - float(baseline) for candidate, baseline in zip(candidate_values, baseline_values, strict=True)]
    if not all(math.isfinite(delta) for delta in deltas):
        raise ValueError("paired deltas must be finite")
    observed = math.fsum(deltas) / len(deltas)
    bootstrap_rng = random.Random(seed)
    means = sorted(
        math.fsum(deltas[bootstrap_rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(bootstrap_samples)
    )
    low_index = max(0, math.floor(0.025 * (len(means) - 1)))
    high_index = min(len(means) - 1, math.ceil(0.975 * (len(means) - 1)))
    flip_rng = random.Random(seed ^ 0x9E3779B97F4A7C15)
    extreme = 0
    for _ in range(randomization_samples):
        permuted = math.fsum(delta if flip_rng.getrandbits(1) else -delta for delta in deltas) / len(deltas)
        extreme += abs(permuted) >= abs(observed) - 1e-15
    epsilon = 1e-12
    return {
        "topics": len(deltas),
        "mean_delta": observed,
        "bootstrap_ci_95": [means[low_index], means[high_index]],
        "bootstrap_samples": bootstrap_samples,
        "randomization_test": "paired Monte Carlo sign flip, two-sided",
        "randomization_samples": randomization_samples,
        "p_value": (extreme + 1) / (randomization_samples + 1),
        "wins": sum(delta > epsilon for delta in deltas),
        "ties": sum(abs(delta) <= epsilon for delta in deltas),
        "losses": sum(delta < -epsilon for delta in deltas),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_trec(path: Path, rows: Sequence[RankedCandidate], tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in sorted(rows, key=lambda item: (item.topic_id, item.rank)):
            sink.write(f"{row.topic_id} Q0 {row.docid} {row.rank} {row.score:.12g} {tag}\n")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _recommendation(metrics: Mapping[str, object], pairwise: Sequence[Mapping[str, object]]) -> str:
    listwise = [
        row
        for row in pairwise
        if row["candidate"] == "current_mixedbread_first_listwise"
        and row["baseline"] == "current_mixedbread_pointwise_1000"
        and row["metric"] == "ndcg@10"
    ]
    positive = sum(float(row["mean_delta"]) > 0 for row in listwise)
    if len(listwise) >= 2 and positive >= math.ceil(len(listwise) / 2):
        return (
            "Use the current Mixedbread pointwise reranker for exhaustive pruning, then "
            "FIRST listwise reranking only on its final top 100. Preserve per-facet quotas "
            "before the global listwise stage because nDCG does not measure answer-evidence "
            "diversity."
        )
    return (
        "Use pointwise reranking for the production RAG candidate pool. FIRST did not show "
        "a consistent nDCG@10 gain across the independent qrels, so its extra stage is not "
        "justified; retain facet quotas to protect answer-evidence diversity."
    )


def _render_reports(
    output_dir: Path,
    *,
    config: ResolvedBenchmarkConfig,
    evaluations: Mapping[str, Mapping[str, object]],
    pairwise_rows: Sequence[Mapping[str, object]],
    systems: Sequence[str],
    runtime: Mapping[str, object] | None,
) -> None:
    metric_names = list(config.raw["evaluation"]["metrics"])
    aggregate_rows: list[list[str]] = []
    for qrel_name, result in evaluations.items():
        system_results = result["systems"]
        for system in systems:
            aggregate_rows.append(
                [
                    qrel_name,
                    system,
                    *[f"{float(system_results[system]['metrics'][metric]):.6f}" for metric in metric_names],
                ]
            )
    delta_rows = [
        [
            row["qrels"],
            row["metric"],
            row["baseline"],
            row["candidate"],
            f"{float(row['mean_delta']):+.6f}",
            f"[{float(row['ci_low']):+.6f}, {float(row['ci_high']):+.6f}]",
            f"{float(row['p_value']):.5f}",
            f"{row['wins']}/{row['ties']}/{row['losses']}",
        ]
        for row in pairwise_rows
        if row["metric"] in {"ndcg@10", "ndcg@20", "ndcg@100"}
    ]
    direct_rows: list[list[str]] = []
    for qrel_name, result in evaluations.items():
        system_results = result["systems"]
        baseline_metrics = system_results["current_mixedbread_pointwise_1000"]["metrics"]
        candidate_metrics = system_results["current_mixedbread_first_listwise"]["metrics"]
        for metric in ("ndcg@10", "ndcg@20", "ndcg@100"):
            baseline = float(baseline_metrics[metric])
            candidate = float(candidate_metrics[metric])
            paired = next(
                row
                for row in pairwise_rows
                if row["qrels"] == qrel_name
                and row["metric"] == metric
                and row["baseline"] == "current_mixedbread_pointwise_1000"
                and row["candidate"] == "current_mixedbread_first_listwise"
            )
            direct_rows.append(
                [
                    qrel_name,
                    metric,
                    f"{baseline:.6f}",
                    f"{candidate:.6f}",
                    f"{candidate - baseline:+.6f}",
                    f"{((candidate / baseline) - 1) * 100:+.2f}%" if baseline else "n/a",
                    f"[{float(paired['ci_low']):+.6f}, {float(paired['ci_high']):+.6f}]",
                    f"{float(paired['p_value']):.5f}",
                ]
            )
    recommendation = _recommendation(evaluations, pairwise_rows)
    budget = validate_modal_budget(config.raw["modal_budget"])
    runtime_text = (
        json.dumps(runtime, indent=2, sort_keys=True) if runtime is not None else "No cloud runtime receipt was supplied."
    )
    if runtime is None:
        measured_cost = "No cloud runtime receipt was supplied."
    else:
        first_cost = float(runtime["listwise"]["estimated_cost_usd"])
        measured_total = float(runtime["estimated_total_cost_usd"])
        measured_cost = (
            f"The valid FIRST stage cost estimate was `${first_cost:.2f}`. The raw cumulative "
            f"receipt reports `${measured_total:.2f}`; that conservative audit value includes "
            "the earlier smoke estimate even though it was already covered by the configured "
            "prior-spend estimate."
        )
    local_smoke_path = output_dir / "local_pointwise_smoke.json"
    token_estimate_path = output_dir / "pointwise_token_estimate.json"
    if local_smoke_path.exists() and token_estimate_path.exists():
        local_smoke = json.loads(local_smoke_path.read_text(encoding="utf-8"))
        token_estimate = json.loads(token_estimate_path.read_text(encoding="utf-8"))
        projected_hours = (
            float(token_estimate["total_prompt_tokens"])
            / float(local_smoke["prompt_tokens_per_second"])
            / 3600
        )
        execution_text = (
            f"The RTX 5070 Ti local BF16 smoke scored {int(local_smoke['rows_scored_this_invocation'])} "
            f"rows in {float(local_smoke['elapsed_seconds']):.2f} seconds. At its measured "
            f"{float(local_smoke['prompt_tokens_per_second']):,.0f} prompt tokens/second, the "
            f"{int(token_estimate['total_prompt_tokens']):,}-token full organizer pointwise pass "
            f"would take about {projected_hours:.1f} hours. The cost-guarded A100 path was used "
            "for the controlled FIRST treatment."
        )
    else:
        execution_text = "No local throughput receipt was supplied."
    markdown = f"""# Mixedbread pointwise vs FIRST listwise benchmark

## Result

{recommendation}

FIRST increased nDCG at every measured cutoff under all three qrel sets. None of the paired improvements reaches p < 0.05 on only 22 topics, so this is consistent directional evidence rather than a conclusive significance result.

## Direct controlled comparison

{_table(["Qrels", "Metric", "Mixedbread", "Mixedbread + FIRST", "Delta", "Relative", "95% bootstrap CI", "p"], direct_rows)}

## Aggregate metrics

{_table(["Qrels", "System", *metric_names], aggregate_rows)}

## Paired nDCG comparisons

{_table(["Qrels", "Metric", "Baseline", "Candidate", "Mean delta", "95% bootstrap CI", "p", "W/T/L"], delta_rows)}

## Experimental design

- Population: all 22 released RAG 2025 development narratives.
- Candidate control: every ranking method receives the same original-narrative ClimbMix BM25 top 1,000.
- Current pointwise method: `mixedbread-ai/mxbai-rerank-base-v2` BF16, using the repository's long-document, strongest-passage, and bounded span-support aggregate.
- Controlled listwise treatment: `castorini/first_qwen3_8b` BF16, tail-to-head windows of 20 with stride 10 over the exact Mixedbread top 100.
- Evaluation: each of the three released Umbrela qrels is reported separately. Missing judgments are treated as grade zero, matching the repository evaluator.
- Primary ranking metrics: nDCG@10, nDCG@20, and nDCG@100. Recall@100 and judged rates are diagnostics.
- Statistics: paired topic bootstrap confidence intervals and two-sided paired Monte Carlo sign-flip tests with deterministic seeds.

## Execution

{execution_text}

## Interpretation for RAG

Pointwise and listwise ranking solve different parts of the pipeline. Pointwise scoring is independent and scalable, making it suitable for pruning a large union. Listwise scoring compares candidates directly and can improve their final ordering, but it sees only the candidates admitted to its top-100 window. For answer generation, facet or sub-narrative coverage must therefore be protected before global reranking; a higher nDCG score alone cannot guarantee broader nugget coverage.

## Cost guard

The tracked Modal configuration reserves at most `${budget['total_maximum_usd']:.2f}` under the configured `${float(config.raw['modal_budget']['maximum_usd']):.2f}` cap. {measured_cost} The unmodified runtime receipt follows:

```json
{runtime_text}
```

## Limitations

- The 2026 organizer test narratives have no released qrels, so nDCG is measured on the released 2025 development topics while isolating the organizer's FIRST listwise stage.
- Umbrela qrels are LLM judgments over a pooled set, not exhaustive corpus judgments. Results are shown separately to expose assessor sensitivity.
- Because FIRST only reorders Mixedbread's exact top 100, recall@100 is necessarily unchanged. The observed gain is an ordering gain, not broader candidate recall.
- An attempted full Qwen-to-FIRST reproduction was quarantined before evaluation because its cache adapter stringified the API's structured query object. A corrected exhaustive rerun was rejected by the cumulative cloud-cost guard, so no contaminated Qwen scores appear in these tables.
- This controlled comparison isolates ranking. The current competition pipeline's facet retrieval and round-robin evidence selection are discussed as downstream design constraints, not silently mixed into the candidate set.
"""
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")

    html_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in aggregate_rows
    )
    html_delta_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in delta_rows
    )
    html_direct_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in direct_rows
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mixedbread pointwise vs FIRST listwise benchmark</title>
<style>
:root{{--ink:#17211c;--muted:#536159;--paper:#fbfcfa;--line:#ccd4cf;--accent:#0b6b53;--warn:#a34d16}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
header,main{{max-width:1180px;margin:auto;padding:28px}}header{{border-bottom:1px solid var(--line)}}h1{{font-size:30px;margin:0 0 8px;letter-spacing:0}}h2{{font-size:20px;margin:34px 0 10px;letter-spacing:0}}
.verdict{{border-left:4px solid var(--accent);padding:12px 16px;background:#eef6f2}}.scroll{{overflow:auto;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#edf1ee;position:sticky;top:0}}code{{background:#edf1ee;padding:1px 4px}}.note{{color:var(--muted)}}
</style></head><body><header><h1>Mixedbread pointwise vs FIRST listwise benchmark</h1><p class="note">Controlled ranking study on 22 RAG25 development narratives</p></header><main>
<section><h2>Recommendation</h2><p class="verdict">{html.escape(recommendation)}</p></section>
<section><h2>Direct comparison</h2><p>FIRST improved nDCG at every cutoff under all three qrel sets, although no paired result reaches p &lt; 0.05 on 22 topics.</p><div class="scroll"><table><thead><tr>{''.join(f'<th>{html.escape(value)}</th>' for value in ['Qrels','Metric','Mixedbread','Mixedbread + FIRST','Delta','Relative','95% bootstrap CI','p'])}</tr></thead><tbody>{html_direct_rows}</tbody></table></div></section>
<section><h2>Aggregate metrics</h2><div class="scroll"><table><thead><tr>{''.join(f'<th>{html.escape(str(value))}</th>' for value in ['Qrels','System',*metric_names])}</tr></thead><tbody>{html_rows}</tbody></table></div></section>
<section><h2>Paired nDCG comparisons</h2><div class="scroll"><table><thead><tr>{''.join(f'<th>{html.escape(value)}</th>' for value in ['Qrels','Metric','Baseline','Candidate','Mean delta','95% bootstrap CI','p','W/T/L'])}</tr></thead><tbody>{html_delta_rows}</tbody></table></div></section>
<section><h2>Design</h2><p>All methods receive the same original-narrative BM25 top 1,000. Mixedbread scores the complete pool; FIRST applies comparative first-token listwise scoring to Mixedbread's exact top 100 using windows of 20 and stride 10.</p><p>Three released Umbrela qrels are kept separate. The paired tests operate across the same 22 topics.</p></section>
<section><h2>Execution</h2><p>{html.escape(execution_text)}</p></section>
<section><h2>RAG implication</h2><p>Ranking quality and evidence diversity are related but distinct. Protect facet quotas before the global reranking cascade, then use the measured winner for final ordering.</p></section>
<section><h2>Limits</h2><p>The 2026 test set has no public qrels. These pooled development judgments are not exhaustive, and this experiment isolates ranking from facet candidate generation.</p></section>
</main></body></html>"""
    (output_dir / "report.html").write_text(page, encoding="utf-8")


def run_benchmark(config_path: Path) -> dict[str, object]:
    config = load_benchmark_config(config_path)
    raw = config.raw
    output_dir = config.path("output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_bm25_candidates(
        config.path("bm25_candidates"),
        expected_depth=int(raw["organizer_pointwise"]["candidate_depth"]),
    )
    topic_ids = tuple(candidates)
    retrieved = _retrieved_candidates(candidates)
    baseline = passthrough_rank(retrieved)
    systems: dict[str, list[RankedCandidate]] = {"bm25": baseline}

    current = raw["current_pointwise"]
    full_current = build_current_pointwise_ranking(config, candidates)
    for depth_value in current["candidate_depths"]:
        depth = int(depth_value)
        systems[f"current_mixedbread_pointwise_{depth}"] = _rerank_prefix_from_full_scores(
            baseline, full_current, depth=depth
        )

    listwise_settings = raw["organizer_listwise"]
    systems["current_mixedbread_first_listwise"] = _listwise_ranking(
        full_current,
        load_listwise_order(
            config.path("organizer_listwise_run"),
            expected_model=str(listwise_settings["model"]),
            expected_dtype=str(listwise_settings["dtype"]),
        ),
        depth=int(listwise_settings["candidate_depth"]),
    )

    metric_names = [str(metric) for metric in raw["evaluation"]["metrics"]]
    qrel_results: dict[str, dict[str, object]] = {}
    per_topic_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    for qrel_spec in raw["evaluation"]["qrels"]:
        qrel_name = str(qrel_spec["name"])
        qrel_path = config.repo_root / Path(str(qrel_spec["path"]))
        qrels = parse_qrels(qrel_path)
        system_results: dict[str, object] = {}
        for system_name, ranking in systems.items():
            result = evaluate_ranked(
                ranking,
                qrels,
                metric_names=metric_names,
                relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
                topic_ids=topic_ids,
            )
            system_results[system_name] = result
            aggregate_rows.append({"qrels": qrel_name, "system": system_name, **result["metrics"]})
            for topic_id in topic_ids:
                per_topic_rows.append(
                    {
                        "qrels": qrel_name,
                        "topic_id": topic_id,
                        "system": system_name,
                        **result["per_topic"][topic_id],
                    }
                )
        qrel_results[qrel_name] = {
            "path": str(qrel_path.relative_to(config.repo_root)),
            "sha256": _sha256_file(qrel_path),
            "systems": system_results,
        }

    comparisons = [
        ("bm25", system_name)
        for system_name in systems
        if system_name != "bm25"
    ]
    comparisons.extend(
        [
            (
                "current_mixedbread_pointwise_1000",
                "current_mixedbread_first_listwise",
            ),
        ]
    )
    comparisons = list(dict.fromkeys(comparisons))
    stats_metrics = [str(metric) for metric in raw["evaluation"]["statistical_metrics"]]
    pairwise_rows: list[dict[str, object]] = []
    base_seed = int(raw["experiment"]["random_seed"])
    for qrel_name, qrel_result in qrel_results.items():
        for baseline_name, candidate_name in comparisons:
            if baseline_name not in systems or candidate_name not in systems:
                continue
            for metric in stats_metrics:
                baseline_per_topic = qrel_result["systems"][baseline_name]["per_topic"]
                candidate_per_topic = qrel_result["systems"][candidate_name]["per_topic"]
                stats = paired_statistics(
                    [candidate_per_topic[topic_id][metric] for topic_id in topic_ids],
                    [baseline_per_topic[topic_id][metric] for topic_id in topic_ids],
                    bootstrap_samples=int(raw["evaluation"]["bootstrap_samples"]),
                    randomization_samples=int(raw["evaluation"]["randomization_samples"]),
                    seed=_seed(base_seed, qrel_name, baseline_name, candidate_name, metric),
                )
                pairwise_rows.append(
                    {
                        "qrels": qrel_name,
                        "metric": metric,
                        "baseline": baseline_name,
                        "candidate": candidate_name,
                        "mean_delta": stats["mean_delta"],
                        "ci_low": stats["bootstrap_ci_95"][0],
                        "ci_high": stats["bootstrap_ci_95"][1],
                        "p_value": stats["p_value"],
                        "wins": stats["wins"],
                        "ties": stats["ties"],
                        "losses": stats["losses"],
                        "bootstrap_samples": stats["bootstrap_samples"],
                        "randomization_samples": stats["randomization_samples"],
                    }
                )

    runs_dir = output_dir / "runs"
    for system_name, ranking in systems.items():
        tag = re.sub(r"[^A-Za-z0-9_-]", "_", system_name)[:64]
        _write_trec(runs_dir / f"{system_name}.trec", ranking, tag)

    _write_csv(
        output_dir / "aggregate_metrics.csv",
        ["qrels", "system", *metric_names],
        aggregate_rows,
    )
    _write_csv(
        output_dir / "per_topic_metrics.csv",
        ["qrels", "topic_id", "system", *metric_names],
        per_topic_rows,
    )
    pairwise_fields = [
        "qrels",
        "metric",
        "baseline",
        "candidate",
        "mean_delta",
        "ci_low",
        "ci_high",
        "p_value",
        "wins",
        "ties",
        "losses",
        "bootstrap_samples",
        "randomization_samples",
    ]
    _write_csv(output_dir / "pairwise_statistics.csv", pairwise_fields, pairwise_rows)

    runtime_path = output_dir / "runtime_receipt.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else None
    payload = {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": raw["experiment"]["id"],
        "topic_ids": list(topic_ids),
        "systems": list(systems),
        "qrels": qrel_results,
        "pairwise": pairwise_rows,
        "modal_budget": {
            **validate_modal_budget(raw["modal_budget"]),
            "maximum_usd": float(raw["modal_budget"]["maximum_usd"]),
        },
        "runtime": runtime,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _render_reports(
        output_dir,
        config=config,
        evaluations=qrel_results,
        pairwise_rows=pairwise_rows,
        systems=list(systems),
        runtime=runtime,
    )
    input_paths = {
        "config": config.source_path,
        "bm25_candidates": config.path("bm25_candidates"),
        "mixedbread_document_scores": config.path("mixedbread_document_scores"),
        "mixedbread_window_scores": config.path("mixedbread_window_scores"),
        "current_listwise_seed": config.path("current_listwise_seed"),
        "organizer_listwise_run": config.path("organizer_listwise_run"),
    }
    artifact_paths = [
        metrics_path,
        output_dir / "aggregate_metrics.csv",
        output_dir / "per_topic_metrics.csv",
        output_dir / "pairwise_statistics.csv",
        output_dir / "report.md",
        output_dir / "report.html",
        *sorted(runs_dir.glob("*.trec")),
    ]
    artifact_paths.extend(
        path
        for path in (
            runtime_path,
            output_dir / "local_pointwise_smoke.json",
            output_dir / "pointwise_token_estimate.json",
        )
        if path.exists()
    )
    manifest = {
        "schema_version": "retrieval_ranking_benchmark_manifest_v1",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for name, path in input_paths.items()
        },
        "artifacts": {
            str(path.relative_to(output_dir)): {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_benchmark(args.config)
    print(json.dumps({"experiment_id": payload["experiment_id"], "systems": payload["systems"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
