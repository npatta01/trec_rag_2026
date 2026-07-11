"""Frozen configuration contract for the deterministic sparse pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from trec_rag.repo_env import find_repo_root, shared_checkout_root


SCHEMA_VERSION = "det_sparse_experiment_v1"
SPLITTER_VERSION = "det_sparse_exact_span_splitter_v1"
TOKENIZER_VERSION = "narrative_token_tape_v1"
PRF_VERSION = "det_sparse_prf_v1"
RRF_VERSION = "weighted_rrf_v1"
RETRIEVAL_LEDGER_VERSION = "det_sparse_retrieval_ledger_v1"

PILOT_TOPIC_IDS = ("200", "225", "707", "897")
LOCKED_PLANNER_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
ARM_NAMES = ("O", "F", "E", "FE")
MAX_EXPANDED_NON_PARENT_FACETS = 3
EVALUATION_METRICS = (
    "recall@100",
    "graded_recall@100",
    "ideal_dcg_coverage@100",
    "recall@50",
    "ndcg@10",
)
EXPERIMENT_ID = "rag25_det_sparse_difficult4_v1"
CONFIG_PATH = "configs/det_sparse_v1.yaml"
OUTPUT_PATH = f"outputs/{EXPERIMENT_ID}"
TOPICS_PATH = (
    "trec-rag-data/trec-rag-2026/development-data/topics/"
    "rag25-topics-dev.tsv"
)
QRELS_PATH = (
    "trec-rag-data/trec-rag-2026/development-data/"
    "rag25-dev-umbrela-qrels/"
    "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
)
ANALYZER_URL = "http://127.0.0.1:18081"
RETRIEVAL_ENDPOINT_URL = (
    "https://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
)
ANALYZER_FINGERPRINT_SHA256 = (
    "f9bbd4e7af26c532105f6dd7e49ce15fa11afd1f0fe7d387847ce41ff7d8def4"
)


@dataclass(frozen=True)
class AnalyzerConfig:
    url: str
    expected_fingerprint_sha256: str


@dataclass(frozen=True)
class RetrievalConfig:
    type: str
    endpoint_env: str
    endpoint_url: str
    index: str
    index_revision: str
    hits: int
    min_results: int
    cache_policy: str


@dataclass(frozen=True)
class CostConfig:
    topic_count: int
    max_facets_per_topic: int
    max_unique_requests_per_topic: int
    max_external_requests: int
    model_calls: int
    reranker_calls: int


@dataclass(frozen=True)
class EvaluationConfig:
    qrels: Path
    relevance_threshold: int
    metrics: tuple[str, ...]
    firewall: str


@dataclass(frozen=True)
class DetSparseConfig:
    root_dir: Path
    config_path: Path
    schema_version: str
    experiment_id: str
    output_dir: Path
    topics_path: Path
    topics_format: str
    topic_ids: tuple[str, ...]
    selection_kind: str
    splitter_version: str
    tokenizer_version: str
    max_facets: int
    min_unique_terms: int
    analyzer: AnalyzerConfig
    retrieval: RetrievalConfig
    retrieval_ledger_version: str
    prf_version: str
    foreground_ranks: tuple[int, int]
    background_ranks: tuple[int, int]
    max_expansion_terms: int
    expand_parent_facet: bool
    max_expanded_facets: int
    rrf_version: str
    rrf_k: int
    arms: tuple[str, ...]
    cost: CostConfig
    evaluation: EvaluationConfig


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_exact_keys(
    mapping: dict[str, Any],
    expected: set[str],
    owner: str,
) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if unknown:
            detail.append("unknown=" + ",".join(unknown))
        raise ValueError(f"{owner} keys differ from frozen schema ({'; '.join(detail)})")


def _text(mapping: dict[str, Any], key: str, owner: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}.{key} must be non-empty text")
    return value.strip()


def _integer(mapping: dict[str, Any], key: str, owner: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{owner}.{key} must be an integer")
    return value


def _boolean(mapping: dict[str, Any], key: str, owner: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{owner}.{key} must be a boolean")
    return value


def _resolve_input(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    active = root / path
    if active.exists():
        return active
    shared = shared_checkout_root(root)
    if shared is not None and (shared / path).exists():
        return shared / path
    return active


def _resolve_output(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_exact(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} is frozen at {expected!r}, got {actual!r}")


def _rank_pair(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must contain exactly two integer ranks")
    return value[0], value[1]


def load_det_sparse_config(path: Path) -> DetSparseConfig:
    """Load and fail closed on any drift from the advisor-frozen pilot."""

    config_path = path.resolve()
    root = find_repo_root(config_path.parent)
    _require_exact(
        config_path,
        (root / CONFIG_PATH).resolve(),
        "config path",
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = _mapping(raw, "config")
    _require_exact_keys(
        config,
        {
            "schema_version",
            "experiment",
            "topics",
            "query_planning",
            "analyzer",
            "retrieval",
            "retrieval_ledger",
            "expansion",
            "fusion",
            "cost",
            "evaluation",
        },
        "config",
    )

    schema_version = _text(config, "schema_version", "config")
    _require_exact(schema_version, SCHEMA_VERSION, "schema_version")

    experiment = _mapping(config.get("experiment"), "experiment")
    _require_exact_keys(experiment, {"id", "output_dir"}, "experiment")
    experiment_id = _text(experiment, "id", "experiment")
    _require_exact(experiment_id, EXPERIMENT_ID, "experiment.id")
    output_path_text = _text(experiment, "output_dir", "experiment")
    _require_exact(output_path_text, OUTPUT_PATH, "experiment.output_dir")
    output_dir = _resolve_output(root, output_path_text)

    topics = _mapping(config.get("topics"), "topics")
    _require_exact_keys(
        topics,
        {"path", "format", "ids", "selection_kind"},
        "topics",
    )
    topic_format = _text(topics, "format", "topics").lower()
    _require_exact(topic_format, "tsv", "topics.format")
    topics_path_text = _text(topics, "path", "topics")
    _require_exact(topics_path_text, TOPICS_PATH, "topics.path")
    raw_topic_ids = topics.get("ids")
    if not isinstance(raw_topic_ids, list) or not all(
        isinstance(topic_id, str) and topic_id for topic_id in raw_topic_ids
    ):
        raise ValueError("topics.ids must be a non-empty string list")
    topic_ids = tuple(raw_topic_ids)
    _require_exact(topic_ids, PILOT_TOPIC_IDS, "topics.ids")
    if LOCKED_PLANNER_TOPIC_IDS.intersection(topic_ids):
        raise ValueError("topics.ids contains a locked planner topic")
    selection_kind = _text(topics, "selection_kind", "topics")
    _require_exact(
        selection_kind,
        "qrel_conditioned_difficult_topic_diagnostic",
        "topics.selection_kind",
    )

    planning = _mapping(config.get("query_planning"), "query_planning")
    _require_exact_keys(
        planning,
        {"splitter_version", "tokenizer_version", "max_facets", "min_unique_terms"},
        "query_planning",
    )
    splitter_version = _text(planning, "splitter_version", "query_planning")
    tokenizer_version = _text(planning, "tokenizer_version", "query_planning")
    max_facets = _integer(planning, "max_facets", "query_planning")
    min_unique_terms = _integer(planning, "min_unique_terms", "query_planning")
    _require_exact(splitter_version, SPLITTER_VERSION, "query_planning.splitter_version")
    _require_exact(tokenizer_version, TOKENIZER_VERSION, "query_planning.tokenizer_version")
    _require_exact(max_facets, 4, "query_planning.max_facets")
    _require_exact(min_unique_terms, 3, "query_planning.min_unique_terms")

    analyzer_raw = _mapping(config.get("analyzer"), "analyzer")
    _require_exact_keys(
        analyzer_raw,
        {"url", "expected_fingerprint_sha256"},
        "analyzer",
    )
    analyzer = AnalyzerConfig(
        url=_text(analyzer_raw, "url", "analyzer"),
        expected_fingerprint_sha256=_text(
            analyzer_raw,
            "expected_fingerprint_sha256",
            "analyzer",
        ),
    )
    _require_exact(analyzer.url, ANALYZER_URL, "analyzer.url")
    parsed_analyzer = urlsplit(analyzer.url)
    if (
        parsed_analyzer.scheme != "http"
        or parsed_analyzer.hostname != "127.0.0.1"
        or parsed_analyzer.port != 18081
        or parsed_analyzer.username is not None
        or parsed_analyzer.password is not None
        or parsed_analyzer.path not in {"", "/"}
        or parsed_analyzer.query
        or parsed_analyzer.fragment
    ):
        raise ValueError("analyzer.url must be the frozen loopback analyzer")
    _require_exact(
        analyzer.expected_fingerprint_sha256,
        ANALYZER_FINGERPRINT_SHA256,
        "analyzer.expected_fingerprint_sha256",
    )

    retrieval_raw = _mapping(config.get("retrieval"), "retrieval")
    _require_exact_keys(
        retrieval_raw,
        {
            "type",
            "endpoint_env",
            "endpoint_url",
            "index",
            "index_revision",
            "hits",
            "min_results",
            "cache_policy",
        },
        "retrieval",
    )
    retrieval = RetrievalConfig(
        type=_text(retrieval_raw, "type", "retrieval"),
        endpoint_env=_text(retrieval_raw, "endpoint_env", "retrieval"),
        endpoint_url=_text(retrieval_raw, "endpoint_url", "retrieval"),
        index=_text(retrieval_raw, "index", "retrieval"),
        index_revision=_text(retrieval_raw, "index_revision", "retrieval"),
        hits=_integer(retrieval_raw, "hits", "retrieval"),
        min_results=_integer(retrieval_raw, "min_results", "retrieval"),
        cache_policy=_text(retrieval_raw, "cache_policy", "retrieval"),
    )
    _require_exact(retrieval.type, "pyserini_remote_raw_first_v1", "retrieval.type")
    _require_exact(retrieval.endpoint_env, "INDEX_URL", "retrieval.endpoint_env")
    _require_exact(
        retrieval.endpoint_url,
        RETRIEVAL_ENDPOINT_URL,
        "retrieval.endpoint_url",
    )
    _require_exact(retrieval.index, "climbmix-400b", "retrieval.index")
    _require_exact(
        retrieval.index_revision,
        "hosted_climbmix_unknown_revision",
        "retrieval.index_revision",
    )
    _require_exact(retrieval.hits, 100, "retrieval.hits")
    _require_exact(retrieval.min_results, 50, "retrieval.min_results")
    _require_exact(
        retrieval.cache_policy,
        "fresh_run_local",
        "retrieval.cache_policy",
    )

    ledger = _mapping(config.get("retrieval_ledger"), "retrieval_ledger")
    _require_exact_keys(ledger, {"version"}, "retrieval_ledger")
    retrieval_ledger_version = _text(ledger, "version", "retrieval_ledger")
    _require_exact(
        retrieval_ledger_version,
        RETRIEVAL_LEDGER_VERSION,
        "retrieval_ledger.version",
    )

    prf = _mapping(config.get("expansion"), "expansion")
    _require_exact_keys(
        prf,
        {
            "version",
            "foreground_ranks",
            "background_ranks",
            "max_terms",
            "expand_parent_facet",
            "max_expanded_facets",
        },
        "expansion",
    )
    prf_version = _text(prf, "version", "expansion")
    foreground_ranks = _rank_pair(prf.get("foreground_ranks"), "expansion.foreground_ranks")
    background_ranks = _rank_pair(prf.get("background_ranks"), "expansion.background_ranks")
    max_expansion_terms = _integer(prf, "max_terms", "expansion")
    expand_parent_facet = _boolean(prf, "expand_parent_facet", "expansion")
    max_expanded_facets = _integer(
        prf,
        "max_expanded_facets",
        "expansion",
    )
    _require_exact(prf_version, PRF_VERSION, "expansion.version")
    _require_exact(foreground_ranks, (1, 5), "expansion.foreground_ranks")
    _require_exact(background_ranks, (6, 50), "expansion.background_ranks")
    _require_exact(max_expansion_terms, 2, "expansion.max_terms")
    _require_exact(expand_parent_facet, False, "expansion.expand_parent_facet")
    _require_exact(
        max_expanded_facets,
        MAX_EXPANDED_NON_PARENT_FACETS,
        "expansion.max_expanded_facets",
    )

    fusion = _mapping(config.get("fusion"), "fusion")
    _require_exact_keys(fusion, {"version", "k", "arms"}, "fusion")
    rrf_version = _text(fusion, "version", "fusion")
    rrf_k = _integer(fusion, "k", "fusion")
    raw_arms = fusion.get("arms")
    if not isinstance(raw_arms, list) or not all(isinstance(arm, str) for arm in raw_arms):
        raise ValueError("fusion.arms must be a string list")
    arms = tuple(raw_arms)
    _require_exact(rrf_version, RRF_VERSION, "fusion.version")
    _require_exact(rrf_k, 60, "fusion.k")
    _require_exact(arms, ARM_NAMES, "fusion.arms")

    cost_raw = _mapping(config.get("cost"), "cost")
    _require_exact_keys(
        cost_raw,
        {
            "topic_count",
            "max_facets_per_topic",
            "max_unique_requests_per_topic",
            "max_external_requests",
            "model_calls",
            "reranker_calls",
        },
        "cost",
    )
    cost = CostConfig(
        topic_count=_integer(cost_raw, "topic_count", "cost"),
        max_facets_per_topic=_integer(cost_raw, "max_facets_per_topic", "cost"),
        max_unique_requests_per_topic=_integer(
            cost_raw,
            "max_unique_requests_per_topic",
            "cost",
        ),
        max_external_requests=_integer(cost_raw, "max_external_requests", "cost"),
        model_calls=_integer(cost_raw, "model_calls", "cost"),
        reranker_calls=_integer(cost_raw, "reranker_calls", "cost"),
    )
    for name, actual, expected in (
        ("cost.topic_count", cost.topic_count, 4),
        ("cost.max_facets_per_topic", cost.max_facets_per_topic, 4),
        ("cost.max_unique_requests_per_topic", cost.max_unique_requests_per_topic, 9),
        ("cost.max_external_requests", cost.max_external_requests, 36),
        ("cost.model_calls", cost.model_calls, 0),
        ("cost.reranker_calls", cost.reranker_calls, 0),
    ):
        _require_exact(actual, expected, name)
    if cost.topic_count * cost.max_unique_requests_per_topic != cost.max_external_requests:
        raise ValueError("cost ceiling does not equal topic_count times per-topic ceiling")

    evaluation_raw = _mapping(config.get("evaluation"), "evaluation")
    _require_exact_keys(
        evaluation_raw,
        {"qrels", "relevance_threshold", "metrics", "firewall"},
        "evaluation",
    )
    qrels_path_text = _text(evaluation_raw, "qrels", "evaluation")
    _require_exact(qrels_path_text, QRELS_PATH, "evaluation.qrels")
    raw_metrics = evaluation_raw.get("metrics")
    if not isinstance(raw_metrics, list) or not all(
        isinstance(metric, str) for metric in raw_metrics
    ):
        raise ValueError("evaluation.metrics must be a string list")
    evaluation = EvaluationConfig(
        qrels=_resolve_input(root, qrels_path_text),
        relevance_threshold=_integer(
            evaluation_raw,
            "relevance_threshold",
            "evaluation",
        ),
        metrics=tuple(raw_metrics),
        firewall=_text(evaluation_raw, "firewall", "evaluation"),
    )
    _require_exact(evaluation.relevance_threshold, 2, "evaluation.relevance_threshold")
    _require_exact(evaluation.metrics, EVALUATION_METRICS, "evaluation.metrics")
    _require_exact(
        evaluation.firewall,
        "frozen_manifest_required",
        "evaluation.firewall",
    )

    return DetSparseConfig(
        root_dir=root,
        config_path=config_path,
        schema_version=schema_version,
        experiment_id=experiment_id,
        output_dir=output_dir,
        topics_path=_resolve_input(root, topics_path_text),
        topics_format=topic_format,
        topic_ids=topic_ids,
        selection_kind=selection_kind,
        splitter_version=splitter_version,
        tokenizer_version=tokenizer_version,
        max_facets=max_facets,
        min_unique_terms=min_unique_terms,
        analyzer=analyzer,
        retrieval=retrieval,
        retrieval_ledger_version=retrieval_ledger_version,
        prf_version=prf_version,
        foreground_ranks=foreground_ranks,
        background_ranks=background_ranks,
        max_expansion_terms=max_expansion_terms,
        expand_parent_facet=expand_parent_facet,
        max_expanded_facets=max_expanded_facets,
        rrf_version=rrf_version,
        rrf_k=rrf_k,
        arms=arms,
        cost=cost,
        evaluation=evaluation,
    )
