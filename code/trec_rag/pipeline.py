"""Config-driven TREC RAG experiment pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from trec_rag.evaluation import evaluate_ranked, parse_qrels
from trec_rag.evidence import select_top_k_evidence
from trec_rag.generation import generate_placeholder_rag
from trec_rag.pipeline_config import RetrieverConfig, load_pipeline_config
from trec_rag.pipeline_models import (
    EvidenceRecord,
    QueryVariant,
    RankedCandidate,
    RetrievedCandidate,
    jsonable,
    write_jsonl,
)
from trec_rag.query_understanding import build_query_variants
from trec_rag.ranking import passthrough_rank
from trec_rag.repo_env import load_repo_env, shared_checkout_root
from trec_rag.retrievers import Retriever, pyserini_factory
from trec_rag.topics import Topic, load_topics


RetrieverFactory = Callable[[RetrieverConfig, Path], Retriever]


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    cache_dir: Path
    queries: list[QueryVariant]
    retrieved: list[RetrievedCandidate]
    ranked: list[RankedCandidate]
    evidence: list[EvidenceRecord]
    rag_records: list[dict[str, object]]
    metrics: dict[str, object]


def _write_trec_run(ranked: list[RankedCandidate], path: Path, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for row in sorted(ranked, key=lambda candidate: (candidate.topic_id, candidate.rank)):
            sink.write(f"{row.topic_id} Q0 {row.docid} {row.rank} {row.score} {run_id}\n")


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pipeline_cache_dir(root_dir: Path, experiment_id: str) -> Path:
    cache_root = shared_checkout_root(root_dir) or root_dir
    return cache_root / "outputs" / experiment_id / "cache"


def run_pipeline(
    config_path: Path,
    *,
    retriever_factories: dict[str, RetrieverFactory] | None = None,
) -> PipelineResult:
    config = load_pipeline_config(config_path)
    load_repo_env(config.root_dir)
    retriever_factories = {"pyserini_remote": pyserini_factory, **(retriever_factories or {})}

    output_dir = config.output_dir
    cache_dir = pipeline_cache_dir(config.root_dir, config.run_id)
    topics = load_topics(config.topics.path)
    topics_by_id = {topic.id: topic for topic in topics}

    variant_configs = [
        {"name": variant.name, "type": variant.type} for variant in config.query_variants
    ]
    queries = [
        query
        for topic in topics
        for query in build_query_variants(topic, variant_configs=variant_configs)
    ]
    write_jsonl(queries, output_dir / "stage_queries.jsonl")

    queries_by_variant: dict[str, list[QueryVariant]] = {}
    for query in queries:
        queries_by_variant.setdefault(query.variant_name, []).append(query)

    retrieved: list[RetrievedCandidate] = []
    for retriever_config in config.retrievers:
        if retriever_config.type not in retriever_factories:
            raise ValueError(f"unknown retriever type: {retriever_config.type}")
        retriever = retriever_factories[retriever_config.type](
            retriever_config,
            cache_dir,
        )
        for variant_name in retriever_config.query_variants:
            for query in queries_by_variant[variant_name]:
                retrieved.extend(retriever.retrieve(query))
    write_jsonl(retrieved, output_dir / "stage_retrieved.jsonl")

    if config.ranking.type != "passthrough":
        raise ValueError(f"unknown ranking type: {config.ranking.type}")
    ranked = passthrough_rank(retrieved)
    write_jsonl(ranked, output_dir / "stage_ranked.jsonl")
    _write_trec_run(ranked, output_dir / "r_output_trec_rag_2026.tsv", config.run_id)

    evidence: list[EvidenceRecord] = []
    rag_records: list[dict[str, object]] = []
    ranked_by_topic: dict[str, list[RankedCandidate]] = {}
    for row in ranked:
        ranked_by_topic.setdefault(row.topic_id, []).append(row)
    for topic in topics:
        topic_evidence = select_top_k_evidence(
            ranked_by_topic.get(topic.id, []),
            k=config.evidence.k,
            require_text=config.evidence.require_text,
            allow_fewer=config.evidence.allow_fewer,
        )
        evidence.extend(topic_evidence)
        rag_records.append(
            generate_placeholder_rag(
                topics_by_id[topic.id],
                topic_evidence,
                team_id=config.submission.team_id,
                run_id=config.run_id,
            )
        )
    write_jsonl(evidence, output_dir / "stage_evidence.jsonl")
    write_jsonl(rag_records, output_dir / "rag_output_trec_rag_2026.jsonl")

    metrics: dict[str, object] = {}
    if config.evaluation:
        metrics = evaluate_ranked(
            ranked,
            parse_qrels(config.evaluation.qrels),
            metric_names=config.evaluation.metrics,
            relevance_threshold=config.evaluation.relevance_threshold,
        )
        metrics["kind"] = config.evaluation.kind
    _write_json(metrics, output_dir / "retrieval_metrics.json")

    return PipelineResult(
        output_dir=output_dir,
        cache_dir=cache_dir,
        queries=queries,
        retrieved=retrieved,
        ranked=ranked,
        evidence=evidence,
        rag_records=rag_records,
        metrics=metrics,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a config-driven TREC RAG pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_pipeline(args.config)
    print(f"Wrote pipeline outputs to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
