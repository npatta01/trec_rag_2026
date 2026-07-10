"""Config-driven TREC RAG experiment pipeline."""

from __future__ import annotations

import argparse
import hashlib
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
from trec_rag.ranking import coverage_aware_long_doc_rank, passthrough_rank
from trec_rag.repo_env import load_repo_env, repo_cache_root, shared_checkout_root
from trec_rag.retrievers import Retriever, pyserini_factory
from trec_rag.topics import load_topics


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
    run_metadata: dict[str, object]


def _write_trec_run(ranked: list[RankedCandidate], path: Path, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for row in sorted(ranked, key=lambda candidate: (candidate.topic_id, candidate.rank)):
            sink.write(f"{row.topic_id} Q0 {row.docid} {row.rank} {row.score} {run_id}\n")


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _portable_repo_path(root_dir: Path, path: Path) -> str:
    roots = [root_dir]
    shared_root = shared_checkout_root(root_dir)
    if shared_root:
        roots.append(shared_root)
    resolved = path.resolve()
    for root in roots:
        try:
            return str(resolved.relative_to(root.resolve()))
        except ValueError:
            continue
    return str(path)


def _file_provenance(root_dir: Path, path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": _portable_repo_path(root_dir, path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def pipeline_cache_dir(root_dir: Path, _experiment_id: str) -> Path:
    return repo_cache_root(root_dir) / "retrieval" / "pyserini_remote"


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
    topics = load_topics(config.topics.path, topic_format=config.topics.format)
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
    retrieval_cache: list[dict[str, object]] = []
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
        cache_summary = getattr(retriever, "cache_summary", None)
        retrieval_cache.append(
            {
                "retriever": retriever_config.name,
                "type": retriever_config.type,
                **(
                    cache_summary()
                    if callable(cache_summary)
                    else {"enabled": retriever_config.cache, "status": "not_reported"}
                ),
            }
        )
    write_jsonl(retrieved, output_dir / "stage_retrieved.jsonl")

    if config.ranking.type == "passthrough":
        ranked = passthrough_rank(retrieved)
    elif config.ranking.type == "coverage_aware_long_doc_aggregate":
        if config.ranking.reranker is None:
            raise ValueError("coverage-aware ranking requires ranking.reranker")
        formula = config.ranking.reranker.formula
        metadata_enabled = (
            config.ranking.reranker.artifact_schema_version is not None
            or config.ranking.reranker.score_representation is not None
        )
        expected_score_metadata = (
            {
                key: value
                for key, value in {
                    "model": config.ranking.reranker.model,
                    "model_revision": config.ranking.reranker.model_revision,
                    "backend_version": config.ranking.reranker.backend_version,
                    "score_representation": config.ranking.reranker.score_representation,
                    "inference_dtype": config.ranking.reranker.inference_dtype,
                    "input_policy": config.ranking.reranker.input_policy,
                    "artifact_schema_version": (
                        config.ranking.reranker.artifact_schema_version
                    ),
                }.items()
                if value is not None
            }
            if metadata_enabled
            else {}
        )
        document_score_metadata: dict[str, object] = {}
        if config.ranking.reranker.document_max_length is not None:
            requested_max_length = config.ranking.reranker.document_max_length
            pair_buffer_tokens = (
                config.ranking.reranker.document_pair_buffer_tokens or 0
            )
            document_score_metadata = {
                "requested_max_length": requested_max_length,
                "max_length": requested_max_length - pair_buffer_tokens,
                "pair_buffer_tokens": pair_buffer_tokens,
                "score_kind": (
                    f"doc_max_{requested_max_length}_buf{pair_buffer_tokens}"
                ),
            }
        window_score_metadata: dict[str, object] = {}
        if config.ranking.reranker.window_max_length is not None:
            window_score_metadata = {
                "requested_max_length": config.ranking.reranker.window_max_length,
                "max_length": config.ranking.reranker.window_max_length,
                "pair_buffer_tokens": 0,
                "score_kind": "window",
            }
            if config.ranking.reranker.chunk_max_characters is not None:
                window_score_metadata["chunk_max_characters"] = (
                    config.ranking.reranker.chunk_max_characters
                )
            if config.ranking.reranker.chunk_overlap_characters is not None:
                window_score_metadata["chunk_overlap_characters"] = (
                    config.ranking.reranker.chunk_overlap_characters
                )
        ranked = coverage_aware_long_doc_rank(
            retrieved,
            document_score_path=config.ranking.reranker.document_score_path,
            window_score_path=config.ranking.reranker.window_score_path,
            candidate_depth=config.ranking.reranker.candidate_depth,
            expected_score_metadata=expected_score_metadata,
            expected_document_score_metadata=document_score_metadata,
            expected_window_score_metadata=window_score_metadata,
            long_document_weight=formula.long_document_weight,
            strongest_passage_weight=formula.strongest_passage_weight,
            coverage_bonus_weight=formula.coverage_bonus_weight,
            relative_span_delta=formula.relative_span_delta,
            support_cap=formula.support_cap,
            min_new_chars=formula.min_new_chars,
            top_window_weights=formula.top_window_weights,
        )
    else:
        raise ValueError(f"unknown ranking type: {config.ranking.type}")
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
            topic_ids=topics_by_id,
        )
        metrics["kind"] = config.evaluation.kind
    _write_json(metrics, output_dir / "retrieval_metrics.json")

    reranker_cache: dict[str, object] | None = None
    if config.ranking.reranker:
        reranker_cache = {
            "score_source": config.ranking.reranker.score_source,
            "score_metadata": expected_score_metadata,
            "document_score_policy": document_score_metadata,
            "window_score_policy": window_score_metadata,
            "document_scores": _file_provenance(
                config.root_dir,
                config.ranking.reranker.document_score_path,
            ),
            "window_scores": _file_provenance(
                config.root_dir,
                config.ranking.reranker.window_score_path,
            ),
            "model_inference_requests": 0,
        }
    run_metadata: dict[str, object] = {
        "experiment_id": config.run_id,
        "topic_count": len(topics),
        "retrieved_candidate_count": len(retrieved),
        "ranked_candidate_count": len(ranked),
        "ranking": {
            "type": config.ranking.type,
            "candidate_depth": (
                config.ranking.reranker.candidate_depth if config.ranking.reranker else None
            ),
        },
        "cache": {
            "retrieval_root": _portable_repo_path(config.root_dir, cache_dir),
            "retrieval": retrieval_cache,
            "reranker": reranker_cache,
        },
    }
    _write_json(run_metadata, output_dir / "run_metadata.json")

    return PipelineResult(
        output_dir=output_dir,
        cache_dir=cache_dir,
        queries=queries,
        retrieved=retrieved,
        ranked=ranked,
        evidence=evidence,
        rag_records=rag_records,
        metrics=metrics,
        run_metadata=run_metadata,
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
