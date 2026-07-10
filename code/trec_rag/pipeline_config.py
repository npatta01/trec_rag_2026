"""YAML configuration for TREC RAG experiment pipelines."""

from __future__ import annotations

import math
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
    cache: bool = True


@dataclass(frozen=True)
class RankingConfig:
    type: str
    dedupe_by: str = "docid"
    dedupe_keep: str = "best_rank"
    preserve_provenance: bool = True
    reranker: "CoverageAwareRerankerConfig | None" = None


@dataclass(frozen=True)
class CoverageAwareFormulaConfig:
    long_document_weight: float = 0.5
    strongest_passage_weight: float = 0.5
    coverage_bonus_weight: float = 0.25
    relative_span_delta: float = 1.0
    support_cap: int = 6
    min_new_chars: int = 800
    top_window_weights: tuple[float, ...] = (0.55, 0.25, 0.13, 0.07)


@dataclass(frozen=True)
class CoverageAwareRerankerConfig:
    model: str
    score_source: str
    document_score_path: Path
    window_score_path: Path
    formula: CoverageAwareFormulaConfig
    candidate_depth: int | None = None
    model_revision: str | None = None
    backend_version: str | None = None
    score_representation: str | None = None
    inference_dtype: str | None = None
    input_policy: str | None = None
    artifact_schema_version: int | None = None
    document_max_length: int | None = None
    document_pair_buffer_tokens: int | None = None
    window_max_length: int | None = None
    chunk_max_characters: int | None = None
    chunk_overlap_characters: int | None = None


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


def _optional_bool(mapping: dict[str, Any], key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"{key} must be a boolean")


def _optional_text(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{key} must not be empty")
    return text


def _optional_nonnegative_int(
    mapping: dict[str, Any],
    key: str,
    *,
    positive: bool,
) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"ranking.reranker.{key} must be a {qualifier} integer")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"ranking.reranker.{key} must be a {qualifier} integer")
    return value


def _coverage_aware_reranker_config(
    ranking_raw: dict[str, Any],
    root_dir: Path,
) -> CoverageAwareRerankerConfig:
    reranker_raw = _require_mapping(ranking_raw.get("reranker"), "ranking.reranker")
    score_source = _require_text(reranker_raw, "score_source", "ranking.reranker")
    if score_source != "cached_artifacts":
        raise ValueError("ranking.reranker.score_source must be cached_artifacts")
    formula_raw = reranker_raw.get("formula") or {}
    if not isinstance(formula_raw, dict):
        raise ValueError("ranking.reranker.formula must be a mapping")
    weights = formula_raw.get("top_window_weights", (0.55, 0.25, 0.13, 0.07))
    if not isinstance(weights, list | tuple) or not weights:
        raise ValueError("ranking.reranker.formula.top_window_weights must be a non-empty list")
    parsed_weights = tuple(float(weight) for weight in weights)
    if not all(math.isfinite(weight) and weight >= 0 for weight in parsed_weights):
        raise ValueError(
            "ranking.reranker.formula.top_window_weights must be finite and non-negative"
        )
    if parsed_weights[0] <= 0:
        raise ValueError(
            "ranking.reranker.formula.top_window_weights must start with a positive weight"
        )
    candidate_depth_raw = reranker_raw.get("candidate_depth")
    if candidate_depth_raw is not None and (
        isinstance(candidate_depth_raw, bool) or not isinstance(candidate_depth_raw, int)
    ):
        raise ValueError("ranking.reranker.candidate_depth must be a positive integer")
    candidate_depth = candidate_depth_raw
    if candidate_depth is not None and candidate_depth <= 0:
        raise ValueError("ranking.reranker.candidate_depth must be greater than zero")
    score_representation = _optional_text(reranker_raw, "score_representation")
    if score_representation not in {None, "raw_logits", "model_default"}:
        raise ValueError(
            "ranking.reranker.score_representation must be raw_logits or model_default"
        )
    artifact_schema_version = _optional_nonnegative_int(
        reranker_raw,
        "artifact_schema_version",
        positive=True,
    )
    document_max_length = _optional_nonnegative_int(
        reranker_raw,
        "document_max_length",
        positive=True,
    )
    document_pair_buffer_tokens = _optional_nonnegative_int(
        reranker_raw,
        "document_pair_buffer_tokens",
        positive=False,
    )
    window_max_length = _optional_nonnegative_int(
        reranker_raw,
        "window_max_length",
        positive=True,
    )
    chunk_max_characters = _optional_nonnegative_int(
        reranker_raw,
        "chunk_max_characters",
        positive=True,
    )
    chunk_overlap_characters = _optional_nonnegative_int(
        reranker_raw,
        "chunk_overlap_characters",
        positive=False,
    )
    if (
        document_max_length is not None
        and document_pair_buffer_tokens is not None
        and document_pair_buffer_tokens >= document_max_length
    ):
        raise ValueError(
            "ranking.reranker.document_pair_buffer_tokens must be smaller than "
            "document_max_length"
        )
    model_revision = _optional_text(reranker_raw, "model_revision")
    backend_version = _optional_text(reranker_raw, "backend_version")
    inference_dtype = _optional_text(reranker_raw, "inference_dtype")
    input_policy = _optional_text(reranker_raw, "input_policy")
    if artifact_schema_version is not None and artifact_schema_version >= 2:
        required_v2 = {
            "model_revision": model_revision,
            "backend_version": backend_version,
            "score_representation": score_representation,
            "inference_dtype": inference_dtype,
            "input_policy": input_policy,
            "document_max_length": document_max_length,
            "document_pair_buffer_tokens": document_pair_buffer_tokens,
            "window_max_length": window_max_length,
            "chunk_max_characters": chunk_max_characters,
            "chunk_overlap_characters": chunk_overlap_characters,
        }
        missing_v2 = [field for field, value in required_v2.items() if value is None]
        if missing_v2:
            raise ValueError(
                "artifact schema v2 requires pinned reranker fields: "
                + ", ".join(missing_v2)
            )
    long_document_weight = float(formula_raw.get("long_document_weight", 0.5))
    strongest_passage_weight = float(formula_raw.get("strongest_passage_weight", 0.5))
    coverage_bonus_weight = float(formula_raw.get("coverage_bonus_weight", 0.25))
    relative_span_delta = float(formula_raw.get("relative_span_delta", 1.0))
    if not all(
        math.isfinite(value)
        for value in (
            long_document_weight,
            strongest_passage_weight,
            coverage_bonus_weight,
            relative_span_delta,
        )
    ):
        raise ValueError("ranking.reranker.formula numeric values must be finite")
    if any(
        value < 0
        for value in (
            long_document_weight,
            strongest_passage_weight,
            coverage_bonus_weight,
        )
    ):
        raise ValueError("ranking.reranker.formula weights must be non-negative")
    if relative_span_delta < 0:
        raise ValueError("ranking.reranker.formula.relative_span_delta must not be negative")
    support_cap = formula_raw.get("support_cap", 6)
    min_new_chars = formula_raw.get("min_new_chars", 800)
    for field, value in (("support_cap", support_cap), ("min_new_chars", min_new_chars)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"ranking.reranker.formula.{field} must be a positive integer"
            )
    return CoverageAwareRerankerConfig(
        model=_require_text(reranker_raw, "model", "ranking.reranker"),
        score_source=score_source,
        document_score_path=_resolve_input_path(
            root_dir,
            _require_text(reranker_raw, "document_score_path", "ranking.reranker"),
        ),
        window_score_path=_resolve_input_path(
            root_dir,
            _require_text(reranker_raw, "window_score_path", "ranking.reranker"),
        ),
        candidate_depth=candidate_depth,
        model_revision=model_revision,
        backend_version=backend_version,
        score_representation=score_representation,
        inference_dtype=inference_dtype,
        input_policy=input_policy,
        artifact_schema_version=artifact_schema_version,
        document_max_length=document_max_length,
        document_pair_buffer_tokens=document_pair_buffer_tokens,
        window_max_length=window_max_length,
        chunk_max_characters=chunk_max_characters,
        chunk_overlap_characters=chunk_overlap_characters,
        formula=CoverageAwareFormulaConfig(
            long_document_weight=long_document_weight,
            strongest_passage_weight=strongest_passage_weight,
            coverage_bonus_weight=coverage_bonus_weight,
            relative_span_delta=relative_span_delta,
            support_cap=support_cap,
            min_new_chars=min_new_chars,
            top_window_weights=parsed_weights,
        ),
    )


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
    topic_format = _require_text(topics_raw, "format", "topics").lower()
    if topic_format not in {"jsonl", "tsv"}:
        raise ValueError("topics.format must be one of: jsonl, tsv")
    topics = TopicsConfig(
        path=_resolve_input_path(root_dir, _require_text(topics_raw, "path", "topics")),
        format=topic_format,
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
                cache=_optional_bool(row, "cache", True),
            )
        )
    _check_unique([retriever.name for retriever in retrievers], "retriever")

    ranking_raw = _require_mapping(config.get("ranking"), "ranking")
    ranking_type = _require_text(ranking_raw, "type", "ranking")
    if ranking_type not in {"passthrough", "coverage_aware_long_doc_aggregate"}:
        raise ValueError(f"unknown ranking type: {ranking_raw.get('type')}")
    dedupe_raw = ranking_raw.get("dedupe") or {}
    if not isinstance(dedupe_raw, dict):
        raise ValueError("ranking.dedupe must be a mapping")
    dedupe_by = str(dedupe_raw.get("by") or "docid")
    dedupe_keep = str(dedupe_raw.get("keep") or "best_rank")
    if dedupe_by != "docid" or dedupe_keep != "best_rank":
        raise ValueError("passthrough dedupe only supports by: docid and keep: best_rank")
    ranking = RankingConfig(
        type=ranking_type,
        dedupe_by=dedupe_by,
        dedupe_keep=dedupe_keep,
        preserve_provenance=bool(dedupe_raw.get("preserve_provenance", True)),
        reranker=(
            _coverage_aware_reranker_config(ranking_raw, root_dir)
            if ranking_type == "coverage_aware_long_doc_aggregate"
            else None
        ),
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
