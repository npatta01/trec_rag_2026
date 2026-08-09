"""Strict configuration for the competition agentic retrieval runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Sequence

import yaml

from trec_rag.deepagent_budget import ResearchBudgetConfig
from trec_rag.deepagent_retrieval import DEFAULT_MODEL
from trec_rag.mixedbread_passage_scorer import (
    MIXEDBREAD_MODEL,
    MIXEDBREAD_REVISION,
)
from trec_rag.repo_env import (
    find_repo_root,
    repo_cache_root,
    shared_checkout_root,
)
from trec_rag.topics import Topic, load_narrative_topics


SCHEMA_VERSION = "agentic_retrieval_config_v1"
RETRIEVAL_MODE = "agentic"
INDEX_ID = "climbmix-400b"
CORPUS_EPOCH = (
    "climbmix-400b-operator-archive-facet-2025-v2-sha256-"
    "2ccad901eb908ac14747bafd2069f7c8aa96ecaaa8cc30457ff47c73ce3bf81f"
)
DOCUMENTS_PER_QUERY = 1_000
HITS_PER_SEARCH = 10
PASSAGES_PER_QUERY = 100
CHUNK_MAX_CHARACTERS = 3_500
CHUNK_OVERLAP_CHARACTERS = 350
SNIPPETS_PER_PAGE = 10
FUSED_RESULT_LIMIT = 20
TOPIC_WORKERS = 2
OFFICIAL_TOPICS_RELATIVE_PATH = Path(
    "trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv"
)
OFFICIAL_TOPIC_IDS = tuple(f"rag2026-{index}" for index in range(119))
_BUDGET_LIMITS = {
    "max_researcher_invocations": 20,
    "max_concurrent": 3,
    "max_retrieval_calls": 100,
    "max_tools_per_researcher": 20,
    "max_searches_per_researcher": 8,
    "max_snippets_per_researcher": 16,
    "max_passage_searches_per_researcher": 8,
    "max_models_per_researcher": 30,
    "max_main_models": 80,
    "synthesis_reserve_turns": 2,
    "no_yield_calls": 3,
    "no_progress_rounds": 2,
}

_SAFE_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SAFE_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]*\Z")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every depth."""


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
class AgenticExperimentSettings:
    id: str


@dataclass(frozen=True)
class AgenticExecutionSettings:
    topic_workers: int


@dataclass(frozen=True)
class AgenticCacheSettings:
    document_store_dir: Path
    model_cache_dir: Path


@dataclass(frozen=True)
class AgenticRetrievalSettings:
    index: str
    cache_dir: Path
    documents_per_query: int
    hits_per_search: int
    corpus_epoch: str


@dataclass(frozen=True)
class AgenticPassageScoringSettings:
    backend: str
    endpoint_id_env: str | None = None
    api_key_env: str | None = None
    request_batch_size: int | None = None
    timeout_seconds: int | None = None
    max_retries: int | None = None


@dataclass(frozen=True)
class AgenticPassageSettings:
    model: str
    revision: str
    score_cache_dir: Path
    device: str
    passages_per_query: int
    chunk_max_characters: int
    chunk_overlap_characters: int
    scoring: AgenticPassageScoringSettings


@dataclass(frozen=True)
class AgenticSnippetSettings:
    result_cache_dir: Path
    snippets_per_page: int


@dataclass(frozen=True)
class AgenticModelSettings:
    coordinator_and_researcher: str


@dataclass(frozen=True)
class AgenticAgentSettings:
    fused_result_limit: int


@dataclass(frozen=True)
class AgenticRetrievalConfig:
    root_dir: Path
    retrieval_mode: str
    experiment: AgenticExperimentSettings
    topics_path: Path
    execution: AgenticExecutionSettings
    caches: AgenticCacheSettings
    retrieval: AgenticRetrievalSettings
    passage: AgenticPassageSettings
    snippets: AgenticSnippetSettings
    models: AgenticModelSettings
    agent: AgenticAgentSettings
    budget: ResearchBudgetConfig

    @property
    def run_id(self) -> str:
        return self.experiment.id

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "outputs" / self.run_id

    def resolved_payload(self, selected_topics: Sequence[Topic]) -> dict[str, object]:
        topics = tuple(selected_topics)
        if any(not isinstance(topic, Topic) for topic in topics):
            raise TypeError("selected_topics must contain Topic values")
        scoring: dict[str, object] = {"backend": self.passage.scoring.backend}
        if self.passage.scoring.backend == "runpod_flash":
            scoring.update(
                {
                    "endpoint_id_env": self.passage.scoring.endpoint_id_env,
                    "api_key_env": self.passage.scoring.api_key_env,
                    "request_batch_size": self.passage.scoring.request_batch_size,
                    "timeout_seconds": self.passage.scoring.timeout_seconds,
                    "max_retries": self.passage.scoring.max_retries,
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "retrieval_mode": self.retrieval_mode,
            "experiment": {"id": self.run_id},
            "topics": {
                "path": _portable_repo_path(self.root_dir, self.topics_path),
                "sha256": _sha256_file(self.topics_path),
                "selected_topic_ids": [topic.id for topic in topics],
            },
            "execution": {"topic_workers": self.execution.topic_workers},
            "caches": {
                "document_store_dir": _portable_cache_path(
                    self.root_dir, self.caches.document_store_dir
                ),
                "model_cache_dir": _portable_cache_path(
                    self.root_dir, self.caches.model_cache_dir
                ),
            },
            "retrieval": {
                "index": self.retrieval.index,
                "cache_dir": _portable_cache_path(
                    self.root_dir, self.retrieval.cache_dir
                ),
                "documents_per_query": self.retrieval.documents_per_query,
                "hits_per_search": self.retrieval.hits_per_search,
                "corpus_epoch": self.retrieval.corpus_epoch,
            },
            "passage": {
                "model": self.passage.model,
                "revision": self.passage.revision,
                "score_cache_dir": _portable_cache_path(
                    self.root_dir, self.passage.score_cache_dir
                ),
                "device": self.passage.device,
                "passages_per_query": self.passage.passages_per_query,
                "chunk_max_characters": self.passage.chunk_max_characters,
                "chunk_overlap_characters": self.passage.chunk_overlap_characters,
                "scoring": scoring,
            },
            "snippets": {
                "result_cache_dir": _portable_cache_path(
                    self.root_dir, self.snippets.result_cache_dir
                ),
                "snippets_per_page": self.snippets.snippets_per_page,
            },
            "models": {
                "coordinator_and_researcher": (
                    self.models.coordinator_and_researcher
                )
            },
            "agent": {"fused_result_limit": self.agent.fused_result_limit},
            "budget": asdict(self.budget),
        }


def load_agentic_retrieval_config(
    path: str | Path,
    *,
    source_bytes: bytes | None = None,
) -> AgenticRetrievalConfig:
    config_path = Path(path).resolve()
    root_dir = find_repo_root(config_path.parent)
    raw = _strict_mapping(
        _load_yaml(config_path, source_bytes=source_bytes),
        "config",
    )
    _require_constant(raw, "schema_version", "config", SCHEMA_VERSION)
    retrieval_mode = _require_constant(
        raw, "retrieval_mode", "config", RETRIEVAL_MODE
    )
    _reject_unknown(
        raw,
        {
            "schema_version",
            "retrieval_mode",
            "experiment",
            "topics",
            "execution",
            "caches",
            "retrieval",
            "passage",
            "snippets",
            "models",
            "agent",
            "budget",
        },
        "config",
    )

    experiment_raw = _strict_mapping(raw.get("experiment"), "experiment")
    _reject_unknown(experiment_raw, {"id"}, "experiment")
    experiment_id = _require_text(experiment_raw, "id", "experiment")
    if _SAFE_EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ValueError("experiment.id must be a safe identifier")

    topics_raw = _strict_mapping(raw.get("topics"), "topics")
    _reject_unknown(topics_raw, {"path"}, "topics")
    topics_path = _resolve_repo_path(
        root_dir, _require_text(topics_raw, "path", "topics")
    )
    _validate_official_topics_path(root_dir, topics_path)

    execution_raw = _strict_mapping(raw.get("execution"), "execution")
    _reject_unknown(execution_raw, {"topic_workers"}, "execution")
    topic_workers = _require_positive_int(
        execution_raw, "topic_workers", "execution"
    )

    caches_raw = _strict_mapping(raw.get("caches"), "caches")
    _reject_unknown(
        caches_raw,
        {"document_store_dir", "model_cache_dir"},
        "caches",
    )
    caches = AgenticCacheSettings(
        document_store_dir=_resolve_cache_path(
            root_dir,
            _require_text(caches_raw, "document_store_dir", "caches"),
        ),
        model_cache_dir=_resolve_cache_path(
            root_dir,
            _require_text(caches_raw, "model_cache_dir", "caches"),
        ),
    )

    retrieval_raw = _strict_mapping(raw.get("retrieval"), "retrieval")
    _reject_unknown(
        retrieval_raw,
        {
            "index",
            "cache_dir",
            "documents_per_query",
            "hits_per_search",
            "corpus_epoch",
        },
        "retrieval",
    )
    retrieval = AgenticRetrievalSettings(
        index=_require_constant(
            retrieval_raw, "index", "retrieval", INDEX_ID
        ),
        cache_dir=_resolve_cache_path(
            root_dir,
            _require_text(retrieval_raw, "cache_dir", "retrieval"),
        ),
        documents_per_query=_require_exact_int(
            retrieval_raw,
            "documents_per_query",
            "retrieval",
            DOCUMENTS_PER_QUERY,
        ),
        hits_per_search=_require_exact_int(
            retrieval_raw,
            "hits_per_search",
            "retrieval",
            HITS_PER_SEARCH,
        ),
        corpus_epoch=_require_constant(
            retrieval_raw,
            "corpus_epoch",
            "retrieval",
            CORPUS_EPOCH,
        ),
    )

    passage_raw = _strict_mapping(raw.get("passage"), "passage")
    _reject_unknown(
        passage_raw,
        {
            "model",
            "revision",
            "score_cache_dir",
            "device",
            "passages_per_query",
            "chunk_max_characters",
            "chunk_overlap_characters",
            "scoring",
        },
        "passage",
    )
    scoring_value = passage_raw.get("scoring")
    if scoring_value is None:
        scoring = AgenticPassageScoringSettings(backend="local")
    else:
        scoring_raw = _strict_mapping(scoring_value, "passage.scoring")
        backend = _require_text(scoring_raw, "backend", "passage.scoring")
        if backend == "local":
            remote_only = set(scoring_raw) - {"backend"}
            if remote_only:
                names = ", ".join(sorted(remote_only))
                raise ValueError(
                    "passage.scoring remote-only field(s) require runpod_flash: "
                    + names
                )
            scoring = AgenticPassageScoringSettings(backend="local")
        elif backend == "runpod_flash":
            _reject_unknown(
                scoring_raw,
                {
                    "backend",
                    "endpoint_id_env",
                    "api_key_env",
                    "request_batch_size",
                    "timeout_seconds",
                    "max_retries",
                },
                "passage.scoring",
            )
            scoring = AgenticPassageScoringSettings(
                backend=backend,
                endpoint_id_env=_require_environment_name(
                    scoring_raw, "endpoint_id_env", "passage.scoring"
                ),
                api_key_env=_require_environment_name(
                    scoring_raw, "api_key_env", "passage.scoring"
                ),
                request_batch_size=_require_bounded_int(
                    scoring_raw,
                    "request_batch_size",
                    "passage.scoring",
                    minimum=1,
                    maximum=256,
                ),
                timeout_seconds=_require_bounded_int(
                    scoring_raw,
                    "timeout_seconds",
                    "passage.scoring",
                    minimum=5,
                    maximum=3_600,
                ),
                max_retries=_require_bounded_int(
                    scoring_raw,
                    "max_retries",
                    "passage.scoring",
                    minimum=0,
                    maximum=5,
                ),
            )
        else:
            raise ValueError("passage.scoring.backend must be local or runpod_flash")
    passage = AgenticPassageSettings(
        model=_require_constant(
            passage_raw, "model", "passage", MIXEDBREAD_MODEL
        ),
        revision=_require_constant(
            passage_raw, "revision", "passage", MIXEDBREAD_REVISION
        ),
        score_cache_dir=_resolve_cache_path(
            root_dir,
            _require_text(passage_raw, "score_cache_dir", "passage"),
        ),
        device=_require_constant(passage_raw, "device", "passage", "auto"),
        passages_per_query=_require_exact_int(
            passage_raw,
            "passages_per_query",
            "passage",
            PASSAGES_PER_QUERY,
        ),
        chunk_max_characters=_require_exact_int(
            passage_raw,
            "chunk_max_characters",
            "passage",
            CHUNK_MAX_CHARACTERS,
        ),
        chunk_overlap_characters=_require_exact_int(
            passage_raw,
            "chunk_overlap_characters",
            "passage",
            CHUNK_OVERLAP_CHARACTERS,
        ),
        scoring=scoring,
    )

    snippets_raw = _strict_mapping(raw.get("snippets"), "snippets")
    _reject_unknown(
        snippets_raw,
        {"result_cache_dir", "snippets_per_page"},
        "snippets",
    )
    snippets = AgenticSnippetSettings(
        result_cache_dir=_resolve_cache_path(
            root_dir,
            _require_text(snippets_raw, "result_cache_dir", "snippets"),
        ),
        snippets_per_page=_require_exact_int(
            snippets_raw,
            "snippets_per_page",
            "snippets",
            SNIPPETS_PER_PAGE,
        ),
    )

    models_raw = _strict_mapping(raw.get("models"), "models")
    _reject_unknown(models_raw, {"coordinator_and_researcher"}, "models")
    models = AgenticModelSettings(
        coordinator_and_researcher=_require_constant(
            models_raw,
            "coordinator_and_researcher",
            "models",
            DEFAULT_MODEL,
        )
    )

    agent_raw = _strict_mapping(raw.get("agent"), "agent")
    _reject_unknown(agent_raw, {"fused_result_limit"}, "agent")
    agent = AgenticAgentSettings(
        fused_result_limit=_require_exact_int(
            agent_raw,
            "fused_result_limit",
            "agent",
            FUSED_RESULT_LIMIT,
        )
    )

    budget_raw = _strict_mapping(raw.get("budget"), "budget")
    _reject_unknown(budget_raw, set(_BUDGET_LIMITS), "budget")
    budget_values = {
        name: _require_exact_int(
            budget_raw,
            name,
            "budget",
            expected,
        )
        for name, expected in _BUDGET_LIMITS.items()
    }
    budget = ResearchBudgetConfig(**budget_values)

    return AgenticRetrievalConfig(
        root_dir=root_dir,
        retrieval_mode=retrieval_mode,
        experiment=AgenticExperimentSettings(experiment_id),
        topics_path=topics_path,
        execution=AgenticExecutionSettings(topic_workers),
        caches=caches,
        retrieval=retrieval,
        passage=passage,
        snippets=snippets,
        models=models,
        agent=agent,
        budget=budget,
    )


def select_agentic_topics(
    config: AgenticRetrievalConfig,
    *,
    topic_ids: Sequence[str] = (),
) -> tuple[Topic, ...]:
    if not isinstance(config, AgenticRetrievalConfig):
        raise TypeError("config must be an AgenticRetrievalConfig")
    if isinstance(topic_ids, str):
        raise TypeError("topic_ids must be a sequence of topic IDs")
    official_topics = load_narrative_topics(config.topics_path)
    if tuple(topic.id for topic in official_topics) != OFFICIAL_TOPIC_IDS:
        raise ValueError(
            "official topic cohort must contain exactly rag2026-0 through "
            "rag2026-118 in source order"
        )
    requested = _validate_topic_ids(
        tuple(topic_ids),
        {topic.id for topic in official_topics},
    )
    if not requested:
        return official_topics
    wanted = set(requested)
    return tuple(topic for topic in official_topics if topic.id in wanted)


def _load_yaml(path: Path, *, source_bytes: bytes | None) -> object:
    if source_bytes is None:
        try:
            source_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-8 YAML config: {path}") from exc
    else:
        if not isinstance(source_bytes, bytes) or not source_bytes:
            raise ValueError("config source_bytes must be non-empty bytes")
        try:
            source_text = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-8 YAML config: {path}") from exc
    try:
        return yaml.load(source_text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML config: {path}") from exc


def _strict_mapping(value: object, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{owner} must be a string-keyed mapping")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], owner: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{owner} has unknown field(s): {names}")


def _require_text(mapping: dict[str, Any], key: str, owner: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}.{key} must be non-empty text")
    return value.strip()


def _require_constant(
    mapping: dict[str, Any],
    key: str,
    owner: str,
    expected: str,
) -> str:
    value = _require_text(mapping, key, owner)
    if value != expected:
        raise ValueError(f"{owner}.{key} must be {expected}")
    return value


def _require_exact_int(
    mapping: dict[str, Any],
    key: str,
    owner: str,
    expected: int,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{owner}.{key} must be {expected}")
    return value


def _require_positive_int(
    mapping: dict[str, Any],
    key: str,
    owner: str,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{owner}.{key} must be a positive integer")
    return value


def _require_bounded_int(
    mapping: dict[str, Any],
    key: str,
    owner: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = mapping.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(
            f"{owner}.{key} must be an integer between {minimum} and {maximum}"
        )
    return value


def _require_environment_name(
    mapping: dict[str, Any],
    key: str,
    owner: str,
) -> str:
    value = _require_text(mapping, key, owner)
    if _SAFE_ENVIRONMENT_NAME.fullmatch(value) is None:
        raise ValueError(f"{owner}.{key} must be an uppercase environment variable name")
    return value


def _resolve_repo_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    active = (root_dir / path).resolve()
    if active.exists():
        return active
    shared_root = shared_checkout_root(root_dir)
    if shared_root is not None:
        shared = (shared_root / path).resolve()
        if shared.exists():
            return shared
    return active


def _validate_official_topics_path(root_dir: Path, topics_path: Path) -> None:
    allowed = {(root_dir / OFFICIAL_TOPICS_RELATIVE_PATH).resolve()}
    shared_root = shared_checkout_root(root_dir)
    if shared_root is not None:
        allowed.add((shared_root / OFFICIAL_TOPICS_RELATIVE_PATH).resolve())
    if topics_path.resolve() not in allowed:
        raise ValueError(
            "topics.path must be the official topic source path "
            f"{OFFICIAL_TOPICS_RELATIVE_PATH}"
        )


def _resolve_cache_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    cache_root = repo_cache_root(root_dir).resolve()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        relative = path.relative_to("cache") if path.parts[:1] == ("cache",) else path
        resolved = (cache_root / relative).resolve()
    try:
        resolved.relative_to(cache_root)
    except ValueError as exc:
        raise ValueError("cache paths must remain beneath the repository cache") from exc
    return resolved


def _validate_topic_ids(
    topic_ids: Sequence[str],
    available: set[str],
) -> tuple[str, ...]:
    result: list[str] = []
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
        result.append(normalized)
    return tuple(result)


def _portable_repo_path(root_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        shared_root = shared_checkout_root(root_dir)
        if shared_root is not None:
            try:
                return str(path.relative_to(shared_root))
            except ValueError:
                pass
        return str(path)


def _portable_cache_path(root_dir: Path, path: Path) -> str:
    cache_root = repo_cache_root(root_dir).resolve()
    return str(Path("cache") / path.resolve().relative_to(cache_root))


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


__all__ = [
    "AgenticRetrievalConfig",
    "SCHEMA_VERSION",
    "load_agentic_retrieval_config",
    "select_agentic_topics",
]
