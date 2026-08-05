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


_SCHEMA_VERSION = "facet_pilot_config_v2"
_QUERY_SOURCES = ("original", "subnarrative")
_RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
_DOCUMENTS_PER_QUERY = 1_000
_PASSAGES_PER_QUERY = 100
_CHUNK_MAX_CHARACTERS = 3_500
_CHUNK_OVERLAP_CHARACTERS = 350
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
    documents_per_query: int
    corpus_epoch: str | None = None


@dataclass(frozen=True)
class PassageSettings:
    model: str
    score_cache_dir: Path
    device: str
    passages_per_query: int
    chunk_max_characters: int
    chunk_overlap_characters: int


@dataclass(frozen=True)
class NuggetSettings:
    evidence_budget_per_subnarrative: int
    maximum_claims_per_subnarrative: int
    maximum_supporting_documents_per_claim: int


@dataclass(frozen=True)
class ExecutionSettings:
    topic_workers: int


@dataclass(frozen=True)
class FacetPilotConfig:
    root_dir: Path
    experiment: ExperimentSettings
    topics_path: Path
    retrieval: RetrievalSettings
    passage: PassageSettings
    nuggets: NuggetSettings
    execution: ExecutionSettings = ExecutionSettings(topic_workers=1)

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
                "documents_per_query": self.retrieval.documents_per_query,
                "corpus_epoch": self.retrieval.corpus_epoch,
            },
            "passage": {
                "model": self.passage.model,
                "score_cache_dir": _portable_path(self.root_dir, self.passage.score_cache_dir),
                "device": self.passage.device,
                "passages_per_query": self.passage.passages_per_query,
                "chunk_max_characters": self.passage.chunk_max_characters,
                "chunk_overlap_characters": self.passage.chunk_overlap_characters,
            },
            "nuggets": {
                "evidence_budget_per_subnarrative": self.nuggets.evidence_budget_per_subnarrative,
                "maximum_claims_per_subnarrative": self.nuggets.maximum_claims_per_subnarrative,
                "maximum_supporting_documents_per_claim": self.nuggets.maximum_supporting_documents_per_claim,
            },
            "execution": {"topic_workers": self.execution.topic_workers},
        }

def load_facet_pilot_config(
    path: Path,
    *,
    source_bytes: bytes | None = None,
) -> FacetPilotConfig:
    config_path = Path(path).resolve()
    root_dir = find_repo_root(config_path.parent)
    raw = _strict_mapping(_load_yaml(config_path, source_bytes=source_bytes), "config")
    if _require_text(raw, "schema_version", "config") != _SCHEMA_VERSION:
        raise ValueError(f"config.schema_version must be {_SCHEMA_VERSION}")
    _reject_unknown(
        raw,
        {
            "schema_version",
            "experiment",
            "topics",
            "retrieval",
            "passage",
            "nuggets",
            "execution",
        },
        "config",
    )

    experiment_raw = _strict_mapping(raw.get("experiment"), "experiment")
    _reject_unknown(experiment_raw, {"id"}, "experiment")
    experiment_id = _require_text(experiment_raw, "id", "experiment")
    if not _SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError("experiment.id must be a safe identifier")

    topics_raw = _strict_mapping(raw.get("topics"), "topics")
    _reject_unknown(topics_raw, {"path"}, "topics")
    topics_path = _resolve_repo_path(root_dir, _require_text(topics_raw, "path", "topics"))

    retrieval_raw = _strict_mapping(raw.get("retrieval"), "retrieval")
    _reject_unknown(
        retrieval_raw,
        {"index", "cache_dir", "query_sources", "documents_per_query", "corpus_epoch"},
        "retrieval",
    )
    query_sources = _required_text_tuple(retrieval_raw, "query_sources", "retrieval")
    if query_sources != _QUERY_SOURCES:
        raise ValueError(f"retrieval.query_sources must be {list(_QUERY_SOURCES)}")
    documents_per_query = _positive_int(retrieval_raw, "documents_per_query", "retrieval")
    if documents_per_query != _DOCUMENTS_PER_QUERY:
        raise ValueError(f"retrieval.documents_per_query must be {_DOCUMENTS_PER_QUERY}")
    retrieval = RetrievalSettings(
        index=_require_text(retrieval_raw, "index", "retrieval"),
        cache_dir=_resolve_cache_path(root_dir, _require_text(retrieval_raw, "cache_dir", "retrieval")),
        query_sources=query_sources,
        documents_per_query=documents_per_query,
        corpus_epoch=_optional_text(retrieval_raw, "corpus_epoch"),
    )

    passage_raw = _strict_mapping(raw.get("passage"), "passage")
    _reject_unknown(
        passage_raw,
        {"model", "score_cache_dir", "device", "passages_per_query", "chunk_max_characters", "chunk_overlap_characters"},
        "passage",
    )
    if _require_text(passage_raw, "model", "passage") != _RERANKER_MODEL:
        raise ValueError(f"passage.model must be {_RERANKER_MODEL}")
    passages_per_query = _positive_int(passage_raw, "passages_per_query", "passage")
    if passages_per_query != _PASSAGES_PER_QUERY:
        raise ValueError(f"passage.passages_per_query must be {_PASSAGES_PER_QUERY}")
    chunk_max_characters = _positive_int(passage_raw, "chunk_max_characters", "passage")
    if chunk_max_characters != _CHUNK_MAX_CHARACTERS:
        raise ValueError(f"passage.chunk_max_characters must be {_CHUNK_MAX_CHARACTERS}")
    chunk_overlap_characters = _positive_int(passage_raw, "chunk_overlap_characters", "passage")
    if chunk_overlap_characters != _CHUNK_OVERLAP_CHARACTERS:
        raise ValueError(f"passage.chunk_overlap_characters must be {_CHUNK_OVERLAP_CHARACTERS}")
    passage = PassageSettings(
        model=_RERANKER_MODEL,
        score_cache_dir=_resolve_cache_path(root_dir, _require_text(passage_raw, "score_cache_dir", "passage")),
        device=_require_text(passage_raw, "device", "passage"),
        passages_per_query=passages_per_query,
        chunk_max_characters=chunk_max_characters,
        chunk_overlap_characters=chunk_overlap_characters,
    )

    nuggets_raw = _strict_mapping(raw.get("nuggets"), "nuggets")
    _reject_unknown(nuggets_raw, {"evidence_budget_per_subnarrative", "maximum_claims_per_subnarrative", "maximum_supporting_documents_per_claim"}, "nuggets")
    nuggets = NuggetSettings(
        evidence_budget_per_subnarrative=_bounded_positive_int(nuggets_raw, "evidence_budget_per_subnarrative", "nuggets", maximum=120),
        maximum_claims_per_subnarrative=_bounded_positive_int(nuggets_raw, "maximum_claims_per_subnarrative", "nuggets", maximum=20),
        maximum_supporting_documents_per_claim=_bounded_positive_int(nuggets_raw, "maximum_supporting_documents_per_claim", "nuggets", maximum=3),
    )
    # Older local smoke configs in this worktree remain safely serial unless
    # they opt into process dispatch. The checked-in production config pins 2.
    execution_value = raw.get("execution", {"topic_workers": 1})
    execution_raw = _strict_mapping(execution_value, "execution")
    _reject_unknown(execution_raw, {"topic_workers"}, "execution")
    execution = ExecutionSettings(
        topic_workers=_positive_int(execution_raw, "topic_workers", "execution")
    )
    return FacetPilotConfig(
        root_dir,
        ExperimentSettings(experiment_id),
        topics_path,
        retrieval,
        passage,
        nuggets,
        execution,
    )


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
    requested_ids = _validate_topic_ids(
        requested_ids,
        {topic.id for topic in official_topics},
    )
    wanted = set(requested_ids)
    return tuple(topic for topic in official_topics if topic.id in wanted)


def _load_yaml(path: Path, *, source_bytes: bytes | None = None) -> object:
    if source_bytes is None:
        source_text = path.read_text(encoding="utf-8")
    else:
        if not isinstance(source_bytes, bytes) or not source_bytes:
            raise ValueError("config source_bytes must be non-empty bytes")
        try:
            source_text = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-8 YAML config: {path}") from exc
    try:
        return yaml.load(
            source_text,
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


def _optional_text(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text when provided")
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


def _validate_topic_ids(
    topic_ids: Sequence[str],
    available: set[str],
) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized_ids: list[str] = []
    for topic_id in topic_ids:
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise ValueError("topic IDs must be non-empty text")
        normalized = topic_id.strip()
        if normalized in seen:
            raise ValueError(f"duplicate topic ID: {normalized}")
        if normalized not in available:
            raise ValueError(f"unknown topic ID: {normalized}")
        seen.add(normalized)
        normalized_ids.append(normalized)
    return tuple(normalized_ids)


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
