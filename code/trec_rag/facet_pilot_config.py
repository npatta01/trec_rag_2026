"""Strict typed configuration and topic selection for the facet pilot."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from trec_rag.repo_env import find_repo_root, repo_cache_root, shared_checkout_root
from trec_rag.topics import Topic, load_narrative_topics


_SCHEMA_VERSION = "facet_pilot_config_v1"
_QUERY_SOURCES = ("original", "subnarrative")
_RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
_SELECTION_POLICY = "round_robin_subnarrative_coverage"
_SAFE_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError("YAML mapping keys must be hashable") from exc
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ExperimentSettings:
    id: str


@dataclass(frozen=True)
class RetrievalSettings:
    index: str
    cache_dir: Path
    query_sources: tuple[str, ...]
    candidate_depth_per_query: int


@dataclass(frozen=True)
class RerankingSettings:
    model: str
    score_cache_dir: Path
    device: str
    rerank_depth_per_query: int
    candidate_pool_depth: int
    selection_policy: str


@dataclass(frozen=True)
class NuggetSettings:
    evidence_budget_per_subnarrative: int
    maximum_claims_per_subnarrative: int
    maximum_supporting_documents_per_claim: int


@dataclass(frozen=True)
class FacetPilotConfig:
    root_dir: Path
    experiment: ExperimentSettings
    topics_path: Path
    retrieval: RetrievalSettings
    reranking: RerankingSettings
    nuggets: NuggetSettings

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "outputs" / self.experiment.id

    @property
    def run_id(self) -> str:
        return self.experiment.id

    def resolved_payload(self, selected_topics: Sequence[Topic]) -> dict[str, object]:
        """Return the normalized, non-secret configuration recorded with an export."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "experiment": {"id": self.experiment.id},
            "topics": {
                "path": _portable_path(self.root_dir, self.topics_path),
                "sha256": _sha256_file(self.topics_path),
                "selected_topic_ids": [topic.id for topic in selected_topics],
            },
            "retrieval": {
                "index": self.retrieval.index,
                "cache_dir": _portable_path(self.root_dir, self.retrieval.cache_dir),
                "query_sources": list(self.retrieval.query_sources),
                "candidate_depth_per_query": self.retrieval.candidate_depth_per_query,
            },
            "reranking": {
                "model": self.reranking.model,
                "score_cache_dir": _portable_path(self.root_dir, self.reranking.score_cache_dir),
                "device": self.reranking.device,
                "rerank_depth_per_query": self.reranking.rerank_depth_per_query,
                "candidate_pool_depth": self.reranking.candidate_pool_depth,
                "selection_policy": self.reranking.selection_policy,
            },
            "nuggets": {
                "evidence_budget_per_subnarrative": self.nuggets.evidence_budget_per_subnarrative,
                "maximum_claims_per_subnarrative": self.nuggets.maximum_claims_per_subnarrative,
                "maximum_supporting_documents_per_claim": self.nuggets.maximum_supporting_documents_per_claim,
            },
        }

def load_facet_pilot_config(path: Path) -> FacetPilotConfig:
    config_path = Path(path).resolve()
    root_dir = find_repo_root(config_path.parent)
    raw = _strict_mapping(_load_yaml(config_path), "config")
    _reject_unknown(raw, {"schema_version", "experiment", "topics", "retrieval", "reranking", "nuggets"}, "config")
    if _require_text(raw, "schema_version", "config") != _SCHEMA_VERSION:
        raise ValueError(f"config.schema_version must be {_SCHEMA_VERSION}")

    experiment_raw = _strict_mapping(raw.get("experiment"), "experiment")
    _reject_unknown(experiment_raw, {"id"}, "experiment")
    experiment_id = _require_text(experiment_raw, "id", "experiment")
    if not _SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError("experiment.id must be a safe identifier")

    topics_raw = _strict_mapping(raw.get("topics"), "topics")
    _reject_unknown(topics_raw, {"path"}, "topics")
    topics_path = _resolve_repo_path(root_dir, _require_text(topics_raw, "path", "topics"))

    retrieval_raw = _strict_mapping(raw.get("retrieval"), "retrieval")
    _reject_unknown(retrieval_raw, {"index", "cache_dir", "query_sources", "candidate_depth_per_query"}, "retrieval")
    query_sources = _required_text_tuple(retrieval_raw, "query_sources", "retrieval")
    if query_sources != _QUERY_SOURCES:
        raise ValueError(f"retrieval.query_sources must be {list(_QUERY_SOURCES)}")
    candidate_depth = _positive_int(retrieval_raw, "candidate_depth_per_query", "retrieval")
    retrieval = RetrievalSettings(
        index=_require_text(retrieval_raw, "index", "retrieval"),
        cache_dir=_resolve_cache_path(root_dir, _require_text(retrieval_raw, "cache_dir", "retrieval")),
        query_sources=query_sources,
        candidate_depth_per_query=candidate_depth,
    )

    reranking_raw = _strict_mapping(raw.get("reranking"), "reranking")
    _reject_unknown(reranking_raw, {"model", "score_cache_dir", "device", "rerank_depth_per_query", "candidate_pool_depth", "selection_policy"}, "reranking")
    if _require_text(reranking_raw, "model", "reranking") != _RERANKER_MODEL:
        raise ValueError(f"reranking.model must be {_RERANKER_MODEL}")
    if _require_text(reranking_raw, "selection_policy", "reranking") != _SELECTION_POLICY:
        raise ValueError(f"reranking.selection_policy must be {_SELECTION_POLICY}")
    rerank_depth = _positive_int(reranking_raw, "rerank_depth_per_query", "reranking")
    candidate_pool_depth = _positive_int(reranking_raw, "candidate_pool_depth", "reranking")
    if rerank_depth > candidate_depth:
        raise ValueError("reranking.rerank_depth_per_query must not exceed retrieval.candidate_depth_per_query")
    if candidate_pool_depth > rerank_depth:
        raise ValueError("reranking.candidate_pool_depth must not exceed reranking.rerank_depth_per_query")
    reranking = RerankingSettings(
        model=_RERANKER_MODEL,
        score_cache_dir=_resolve_cache_path(root_dir, _require_text(reranking_raw, "score_cache_dir", "reranking")),
        device=_require_text(reranking_raw, "device", "reranking"),
        rerank_depth_per_query=rerank_depth,
        candidate_pool_depth=candidate_pool_depth,
        selection_policy=_SELECTION_POLICY,
    )

    nuggets_raw = _strict_mapping(raw.get("nuggets"), "nuggets")
    _reject_unknown(nuggets_raw, {"evidence_budget_per_subnarrative", "maximum_claims_per_subnarrative", "maximum_supporting_documents_per_claim"}, "nuggets")
    nuggets = NuggetSettings(
        evidence_budget_per_subnarrative=_bounded_positive_int(nuggets_raw, "evidence_budget_per_subnarrative", "nuggets", maximum=40),
        maximum_claims_per_subnarrative=_bounded_positive_int(nuggets_raw, "maximum_claims_per_subnarrative", "nuggets", maximum=20),
        maximum_supporting_documents_per_claim=_bounded_positive_int(nuggets_raw, "maximum_supporting_documents_per_claim", "nuggets", maximum=3),
    )
    return FacetPilotConfig(root_dir, ExperimentSettings(experiment_id), topics_path, retrieval, reranking, nuggets)


def select_configured_topics(
    config: FacetPilotConfig,
    *,
    topic_ids: Sequence[str] = (),
    subset_csv: Path | None = None,
) -> tuple[Topic, ...]:
    """Select configured topics in the official source-file order."""
    if topic_ids and subset_csv is not None:
        raise ValueError("topic_ids and subset_csv are mutually exclusive")
    official_topics = tuple(load_narrative_topics(config.topics_path))
    if topic_ids:
        requested_ids = tuple(topic_ids)
    elif subset_csv is not None:
        requested_ids = _read_subset_ids(subset_csv)
        if not requested_ids:
            raise ValueError("subset CSV must contain at least one topic_id")
    else:
        requested_ids = ()
    if not requested_ids:
        return official_topics
    _validate_topic_ids(requested_ids, {topic.id for topic in official_topics})
    wanted = set(requested_ids)
    return tuple(topic for topic in official_topics if topic.id in wanted)


def _load_yaml(path: Path) -> object:
    try:
        return yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML config: {path}") from exc


def _strict_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{name} has unknown field(s): {', '.join(sorted(map(str, unknown)))}")


def _require_text(mapping: dict[str, Any], key: str, owner: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}.{key} must be non-empty text")
    return value.strip()


def _positive_int(mapping: dict[str, Any], key: str, owner: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{owner}.{key} must be a positive integer")
    return value


def _bounded_positive_int(mapping: dict[str, Any], key: str, owner: str, *, maximum: int) -> int:
    value = _positive_int(mapping, key, owner)
    if value > maximum:
        raise ValueError(f"{owner}.{key} must not exceed {maximum}")
    return value


def _required_text_tuple(mapping: dict[str, Any], key: str, owner: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{owner}.{key} must be a list of non-empty text")
    return tuple(item.strip() for item in value)


def _resolve_repo_path(root_dir: Path, value: str) -> Path:
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


def _resolve_cache_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    cache_root = repo_cache_root(root_dir).resolve()
    if path.is_absolute():
        resolved_path = path.resolve()
    else:
        relative_path = path.relative_to("cache") if path.parts[:1] == ("cache",) else path
        resolved_path = (cache_root / relative_path).resolve()
    try:
        resolved_path.relative_to(cache_root)
    except ValueError as exc:
        raise ValueError("cache paths must remain beneath the repository cache") from exc
    return resolved_path


def _read_subset_ids(path: Path | None) -> tuple[str, ...]:
    assert path is not None
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or "topic_id" not in reader.fieldnames:
                raise ValueError("subset CSV requires a topic_id header")
            return tuple("" if row.get("topic_id") is None else row["topic_id"].strip() for row in reader)
    except csv.Error as exc:
        raise ValueError("invalid subset CSV") from exc


def _validate_topic_ids(topic_ids: Sequence[str], available: set[str]) -> None:
    seen: set[str] = set()
    for topic_id in topic_ids:
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise ValueError("topic IDs must be non-empty text")
        normalized = topic_id.strip()
        if normalized in seen:
            raise ValueError(f"duplicate topic ID: {normalized}")
        if normalized not in available:
            raise ValueError(f"unknown topic ID: {normalized}")
        seen.add(normalized)


def _portable_path(root_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        shared_root = shared_checkout_root(root_dir)
        if shared_root:
            try:
                return str(path.relative_to(shared_root))
            except ValueError:
                pass
        return str(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
