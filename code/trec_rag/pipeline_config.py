"""YAML configuration for TREC RAG experiment pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trec_rag.repo_env import find_repo_root, shared_checkout_root


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    output_dir: Path


@dataclass(frozen=True)
class SubmissionConfig:
    team_id: str


@dataclass(frozen=True)
class TopicsConfig:
    path: Path
    format: str


@dataclass(frozen=True)
class QueryVariantConfig:
    name: str
    type: str


@dataclass(frozen=True)
class RetrieverConfig:
    name: str
    type: str
    query_variants: tuple[str, ...]
    hits: int
    index: str | None = None


@dataclass(frozen=True)
class RankingConfig:
    type: str
    dedupe_by: str = "docid"
    dedupe_keep: str = "best_rank"
    preserve_provenance: bool = True


@dataclass(frozen=True)
class EvidenceConfig:
    type: str
    k: int
    require_text: bool = True
    allow_fewer: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    type: str


@dataclass(frozen=True)
class EvaluationConfig:
    kind: str
    qrels: Path
    metrics: tuple[str, ...]
    relevance_threshold: int = 2


@dataclass(frozen=True)
class PipelineConfig:
    root_dir: Path
    experiment: ExperimentConfig
    submission: SubmissionConfig
    topics: TopicsConfig
    query_variants: tuple[QueryVariantConfig, ...]
    retrievers: tuple[RetrieverConfig, ...]
    ranking: RankingConfig
    evidence: EvidenceConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig | None

    @property
    def output_dir(self) -> Path:
        return self.experiment.output_dir

    @property
    def run_id(self) -> str:
        return self.experiment.id


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_text(mapping: dict[str, Any], key: str, owner: str) -> str:
    value = mapping.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{owner}.{key} is required")
    return str(value).strip()


def _resolve_output_path(root_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root_dir / path


def _resolve_input_path(root_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    active_path = root_dir / path
    if active_path.exists():
        return active_path
    shared_root = shared_checkout_root(root_dir)
    if shared_root:
        shared_path = shared_root / path
        if shared_path.exists():
            return shared_path
    return active_path


def _check_unique(names: list[str], label: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate {label}: {name}")
        seen.add(name)


def load_pipeline_config(path: Path) -> PipelineConfig:
    config_path = path.resolve()
    root_dir = find_repo_root(config_path.parent)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = _require_mapping(raw, "config")

    experiment_raw = _require_mapping(config.get("experiment"), "experiment")
    experiment_id = _require_text(experiment_raw, "id", "experiment")
    output_dir = _resolve_output_path(
        root_dir,
        experiment_raw.get("output_dir") or Path("outputs") / experiment_id,
    )

    submission_raw = _require_mapping(config.get("submission"), "submission")
    if "run_id" in submission_raw:
        raise ValueError("submission.run_id is not supported; use experiment.id")
    submission = SubmissionConfig(team_id=_require_text(submission_raw, "team_id", "submission"))

    topics_raw = _require_mapping(config.get("topics"), "topics")
    topics = TopicsConfig(
        path=_resolve_input_path(root_dir, _require_text(topics_raw, "path", "topics")),
        format=_require_text(topics_raw, "format", "topics"),
    )

    query_raw = _require_mapping(config.get("query_understanding"), "query_understanding")
    variant_rows = query_raw.get("variants")
    if not isinstance(variant_rows, list) or not variant_rows:
        raise ValueError("query_understanding.variants must be a non-empty list")
    query_variants = tuple(
        QueryVariantConfig(
            name=_require_text(_require_mapping(row, "query variant"), "name", "query variant"),
            type=_require_text(_require_mapping(row, "query variant"), "type", "query variant"),
        )
        for row in variant_rows
    )
    _check_unique([variant.name for variant in query_variants], "query variant")
    for variant in query_variants:
        if variant.type != "original_topic":
            raise ValueError(f"unknown query variant type: {variant.type}")
    variant_names = {variant.name for variant in query_variants}

    retriever_rows = config.get("retrievers")
    if not isinstance(retriever_rows, list) or not retriever_rows:
        raise ValueError("retrievers must be a non-empty list")
    retrievers: list[RetrieverConfig] = []
    for row in retriever_rows:
        row = _require_mapping(row, "retriever")
        query_variant_names = tuple(str(name) for name in row.get("query_variants", ()))
        if not query_variant_names:
            raise ValueError("retriever.query_variants must be a non-empty list")
        unknown = sorted(set(query_variant_names) - variant_names)
        if unknown:
            raise ValueError(f"unknown query variant reference: {', '.join(unknown)}")
        retrievers.append(
            RetrieverConfig(
                name=_require_text(row, "name", "retriever"),
                type=_require_text(row, "type", "retriever"),
                query_variants=query_variant_names,
                hits=int(row.get("hits") or 100),
                index=str(row["index"]).strip() if row.get("index") else None,
            )
        )
    _check_unique([retriever.name for retriever in retrievers], "retriever")

    ranking_raw = _require_mapping(config.get("ranking"), "ranking")
    if _require_text(ranking_raw, "type", "ranking") != "passthrough":
        raise ValueError(f"unknown ranking type: {ranking_raw.get('type')}")
    dedupe_raw = ranking_raw.get("dedupe") or {}
    if not isinstance(dedupe_raw, dict):
        raise ValueError("ranking.dedupe must be a mapping")
    dedupe_by = str(dedupe_raw.get("by") or "docid")
    dedupe_keep = str(dedupe_raw.get("keep") or "best_rank")
    if dedupe_by != "docid" or dedupe_keep != "best_rank":
        raise ValueError("passthrough dedupe only supports by: docid and keep: best_rank")
    ranking = RankingConfig(
        type="passthrough",
        dedupe_by=dedupe_by,
        dedupe_keep=dedupe_keep,
        preserve_provenance=bool(dedupe_raw.get("preserve_provenance", True)),
    )

    evidence_raw = _require_mapping(config.get("evidence"), "evidence")
    if _require_text(evidence_raw, "type", "evidence") != "top_k":
        raise ValueError(f"unknown evidence type: {evidence_raw.get('type')}")
    evidence = EvidenceConfig(
        type="top_k",
        k=int(evidence_raw.get("k") or 5),
        require_text=bool(evidence_raw.get("require_text", True)),
        allow_fewer=bool(evidence_raw.get("allow_fewer", True)),
    )

    generation_raw = _require_mapping(config.get("generation"), "generation")
    if _require_text(generation_raw, "type", "generation") != "placeholder":
        raise ValueError(f"unknown generation type: {generation_raw.get('type')}")
    generation = GenerationConfig(type="placeholder")

    evaluation = None
    if config.get("evaluation") is not None:
        evaluation_raw = _require_mapping(config.get("evaluation"), "evaluation")
        evaluation = EvaluationConfig(
            kind=str(evaluation_raw.get("kind") or "dev_projected_qrels"),
            qrels=_resolve_input_path(root_dir, _require_text(evaluation_raw, "qrels", "evaluation")),
            metrics=tuple(str(metric) for metric in evaluation_raw.get("metrics") or ()),
            relevance_threshold=int(evaluation_raw.get("relevance_threshold") or 2),
        )

    return PipelineConfig(
        root_dir=root_dir,
        experiment=ExperimentConfig(id=experiment_id, output_dir=output_dir),
        submission=submission,
        topics=topics,
        query_variants=query_variants,
        retrievers=tuple(retrievers),
        ranking=ranking,
        evidence=evidence,
        generation=generation,
        evaluation=evaluation,
    )
