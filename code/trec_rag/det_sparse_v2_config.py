"""Fail-closed frozen configuration contract for ``det_sparse_v2``.

The v2 experiment selects four topics from a source-frozen candidate pool by a
deterministic structural procedure.  It deliberately does not encode qrel-
conditioned topic choices.  Loading succeeds only for the tracked config at its
canonical path and only when every policy field matches the advisor-approved
protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from trec_rag.repo_env import find_repo_root, shared_checkout_root


SCHEMA_VERSION = "det_sparse_v2"
PLANNER_VERSION = "det_sparse_v2"
RENDERER_VERSION = "det_sparse_bounded_parent_renderer_v2"
SPLITTER_VERSION = "det_sparse_exact_span_splitter_v1"
TOKENIZER_VERSION = "narrative_token_tape_v1"
SELECTION_VERSION = "det_sparse_structural_selection_v2"
SELECTION_SEED = "det_sparse_v2_structural_selection_20260711"
SELECTION_STRATA = ("A", "B", "C", "D")
PILOT_TOPIC_COUNT = 4

CANDIDATE_TOPIC_IDS = (
    "14",
    "31",
    "37",
    "58",
    "72",
    "84",
    "161",
    "219",
    "233",
    "273",
    "300",
    "477",
    "499",
)
EXCLUDED_TOPIC_IDS = (
    "144",
    "200",
    "213",
    "224",
    "225",
    "407",
    "515",
    "707",
    "897",
)
ARM_NAMES = ("O", "F", "E", "FE")
EVALUATION_METRICS = (
    "recall@100",
    "graded_recall@100",
    "ideal_dcg_coverage@100",
    "recall@50",
    "ndcg@10",
)

EXPERIMENT_ID = "rag25_det_sparse_structural4_v2"
CONFIG_PATH = "configs/det_sparse_v2.yaml"
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
RETRIEVAL_INDEX = "climbmix-400b"
RETRIEVAL_INDEX_REVISION = "hosted_climbmix_unknown_revision"
ANALYZER_FINGERPRINT_SHA256 = (
    "f9bbd4e7af26c532105f6dd7e49ce15fa11afd1f0fe7d387847ce41ff7d8def4"
)

BOUNDED_CONTEXT_ALLOCATION = "floor_half_u1_unique_terms"
PRF_FORMULA_VERSION = "det_sparse_prf_v1"
PRF_ARTIFACT_VERSION = "det_sparse_prf_artifact_v2"
RRF_VERSION = "weighted_rrf_v1"
RETRIEVAL_LEDGER_VERSION = "det_sparse_retrieval_ledger_v1"
RETRIEVAL_RUN_NAMESPACE = "det_sparse_v2_fresh_run_local"
GLOBAL_TICKET_NAMESPACE = EXPERIMENT_ID
EXTERNAL_GATE_STATUS = "blocked"
EXTERNAL_GATE_REASON = "hosted_index_revision_unknown"


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    if not isinstance(node, MappingNode):
        raise ConstructorError(
            None,
            None,
            "expected a mapping node",
            node.start_mark,
        )
    # Resolve YAML merge keys before checking so an explicit key cannot silently
    # override the same policy supplied through a merge mapping.
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicated = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicated:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
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
    required_results: int
    max_attempts: int
    retry_policy: str
    redirect_policy: str
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
class DetSparseV2Config:
    root_dir: Path
    config_path: Path
    schema_version: str
    experiment_id: str
    output_dir: Path
    topics_path: Path
    topics_format: str
    candidate_topic_ids: tuple[str, ...]
    excluded_topic_ids: tuple[str, ...]
    selection_version: str
    selection_seed: str
    selection_strata: tuple[str, ...]
    selected_topic_count: int
    planner_version: str
    renderer_version: str
    splitter_version: str
    tokenizer_version: str
    max_facets: int
    min_facet_terms: int
    parent_min_unique_terms: int
    bounded_context_min_unique_terms: int
    bounded_context_max_unique_terms: int
    bounded_context_allocation: str
    analyzer: AnalyzerConfig
    retrieval: RetrievalConfig
    retrieval_ledger_version: str
    retrieval_run_namespace: str
    global_ticket_namespace: str
    prf_formula_version: str
    prf_artifact_version: str
    foreground_ranks: tuple[int, int]
    background_ranks: tuple[int, int]
    max_expansion_terms: int
    expand_parent_facet: bool
    max_expanded_facets: int
    rrf_version: str
    rrf_k: int
    arms: tuple[str, ...]
    cost: CostConfig
    external_gate_status: str
    external_gate_reason: str
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
    if actual == expected:
        return
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
    if value != value.strip():
        raise ValueError(
            f"{owner}.{key} must not contain leading or trailing whitespace"
        )
    return value


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


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a non-empty string list")
    return tuple(value)


def _rank_pair(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must contain exactly two integer ranks")
    return value[0], value[1]


def _require_exact(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} is frozen at {expected!r}, got {actual!r}")


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


def _validate_loopback_analyzer(analyzer: AnalyzerConfig) -> None:
    parsed = urlsplit(analyzer.url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 18081
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("analyzer.url must be the frozen loopback analyzer")


def _validate_retrieval_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.castorini.uwaterloo.ca"
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/v1/climbmix-400b/search"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("retrieval.endpoint_url must be the frozen HTTPS endpoint")


def load_det_sparse_v2_config(path: Path) -> DetSparseV2Config:
    """Load v2 only when every identity and policy field is exactly frozen."""

    config_path = path.resolve()
    root = find_repo_root(config_path.parent)
    _require_exact(config_path, (root / CONFIG_PATH).resolve(), "config path")

    raw = yaml.load(
        config_path.read_text(encoding="utf-8"),
        Loader=_UniqueKeySafeLoader,
    ) or {}
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
            "external_gate",
            "evaluation",
        },
        "config",
    )

    schema_version = _text(config, "schema_version", "config")
    _require_exact(schema_version, SCHEMA_VERSION, "schema_version")

    experiment = _mapping(config.get("experiment"), "experiment")
    _require_exact_keys(experiment, {"id", "output_dir"}, "experiment")
    experiment_id = _text(experiment, "id", "experiment")
    output_path_text = _text(experiment, "output_dir", "experiment")
    _require_exact(experiment_id, EXPERIMENT_ID, "experiment.id")
    _require_exact(output_path_text, OUTPUT_PATH, "experiment.output_dir")

    topics = _mapping(config.get("topics"), "topics")
    _require_exact_keys(
        topics,
        {"path", "format", "candidate_ids", "excluded_ids", "selection"},
        "topics",
    )
    topics_path_text = _text(topics, "path", "topics")
    topics_format = _text(topics, "format", "topics")
    candidate_ids = _string_tuple(topics.get("candidate_ids"), "topics.candidate_ids")
    excluded_ids = _string_tuple(topics.get("excluded_ids"), "topics.excluded_ids")
    _require_exact(topics_path_text, TOPICS_PATH, "topics.path")
    _require_exact(topics_format, "tsv", "topics.format")
    _require_exact(candidate_ids, CANDIDATE_TOPIC_IDS, "topics.candidate_ids")
    _require_exact(excluded_ids, EXCLUDED_TOPIC_IDS, "topics.excluded_ids")
    if tuple(sorted(candidate_ids, key=int)) != candidate_ids or len(set(candidate_ids)) != len(
        candidate_ids
    ):
        raise ValueError("topics.candidate_ids must be unique and numeric-sorted")
    if tuple(sorted(excluded_ids, key=int)) != excluded_ids or len(set(excluded_ids)) != len(
        excluded_ids
    ):
        raise ValueError("topics.excluded_ids must be unique and numeric-sorted")
    if set(candidate_ids).intersection(excluded_ids):
        raise ValueError("topics candidate and exclusion sets must be disjoint")

    selection = _mapping(topics.get("selection"), "topics.selection")
    _require_exact_keys(
        selection,
        {"version", "seed", "strata", "selected_count"},
        "topics.selection",
    )
    selection_version = _text(selection, "version", "topics.selection")
    selection_seed = _text(selection, "seed", "topics.selection")
    selection_strata = _string_tuple(
        selection.get("strata"),
        "topics.selection.strata",
    )
    selected_topic_count = _integer(selection, "selected_count", "topics.selection")
    _require_exact(selection_version, SELECTION_VERSION, "topics.selection.version")
    _require_exact(selection_seed, SELECTION_SEED, "topics.selection.seed")
    _require_exact(selection_strata, SELECTION_STRATA, "topics.selection.strata")
    _require_exact(selected_topic_count, PILOT_TOPIC_COUNT, "topics.selection.selected_count")

    planning = _mapping(config.get("query_planning"), "query_planning")
    _require_exact_keys(
        planning,
        {
            "planner_version",
            "renderer_version",
            "splitter_version",
            "tokenizer_version",
            "max_facets",
            "min_facet_terms",
            "parent_min_unique_terms",
            "bounded_context",
        },
        "query_planning",
    )
    planner_version = _text(planning, "planner_version", "query_planning")
    renderer_version = _text(planning, "renderer_version", "query_planning")
    splitter_version = _text(planning, "splitter_version", "query_planning")
    tokenizer_version = _text(planning, "tokenizer_version", "query_planning")
    max_facets = _integer(planning, "max_facets", "query_planning")
    min_facet_terms = _integer(planning, "min_facet_terms", "query_planning")
    parent_min_unique_terms = _integer(
        planning,
        "parent_min_unique_terms",
        "query_planning",
    )
    _require_exact(planner_version, PLANNER_VERSION, "query_planning.planner_version")
    _require_exact(renderer_version, RENDERER_VERSION, "query_planning.renderer_version")
    _require_exact(splitter_version, SPLITTER_VERSION, "query_planning.splitter_version")
    _require_exact(tokenizer_version, TOKENIZER_VERSION, "query_planning.tokenizer_version")
    _require_exact(max_facets, 4, "query_planning.max_facets")
    _require_exact(min_facet_terms, 3, "query_planning.min_facet_terms")
    _require_exact(
        parent_min_unique_terms,
        4,
        "query_planning.parent_min_unique_terms",
    )

    bounded_context = _mapping(
        planning.get("bounded_context"),
        "query_planning.bounded_context",
    )
    _require_exact_keys(
        bounded_context,
        {"min_unique_terms", "max_unique_terms", "allocation"},
        "query_planning.bounded_context",
    )
    bounded_min = _integer(
        bounded_context,
        "min_unique_terms",
        "query_planning.bounded_context",
    )
    bounded_max = _integer(
        bounded_context,
        "max_unique_terms",
        "query_planning.bounded_context",
    )
    bounded_allocation = _text(
        bounded_context,
        "allocation",
        "query_planning.bounded_context",
    )
    _require_exact(bounded_min, 2, "query_planning.bounded_context.min_unique_terms")
    _require_exact(bounded_max, 6, "query_planning.bounded_context.max_unique_terms")
    _require_exact(
        bounded_allocation,
        BOUNDED_CONTEXT_ALLOCATION,
        "query_planning.bounded_context.allocation",
    )

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
    _validate_loopback_analyzer(analyzer)
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
            "required_results",
            "max_attempts",
            "retry_policy",
            "redirect_policy",
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
        required_results=_integer(retrieval_raw, "required_results", "retrieval"),
        max_attempts=_integer(retrieval_raw, "max_attempts", "retrieval"),
        retry_policy=_text(retrieval_raw, "retry_policy", "retrieval"),
        redirect_policy=_text(retrieval_raw, "redirect_policy", "retrieval"),
        cache_policy=_text(retrieval_raw, "cache_policy", "retrieval"),
    )
    for name, actual, expected in (
        ("retrieval.type", retrieval.type, "pyserini_remote_raw_first_v1"),
        ("retrieval.endpoint_env", retrieval.endpoint_env, "INDEX_URL"),
        ("retrieval.endpoint_url", retrieval.endpoint_url, RETRIEVAL_ENDPOINT_URL),
        ("retrieval.index", retrieval.index, RETRIEVAL_INDEX),
        ("retrieval.index_revision", retrieval.index_revision, RETRIEVAL_INDEX_REVISION),
        ("retrieval.hits", retrieval.hits, 100),
        ("retrieval.required_results", retrieval.required_results, 100),
        ("retrieval.max_attempts", retrieval.max_attempts, 1),
        ("retrieval.retry_policy", retrieval.retry_policy, "none"),
        ("retrieval.redirect_policy", retrieval.redirect_policy, "none"),
        ("retrieval.cache_policy", retrieval.cache_policy, "fresh_run_local"),
    ):
        _require_exact(actual, expected, name)
    _validate_retrieval_url(retrieval.endpoint_url)

    ledger = _mapping(config.get("retrieval_ledger"), "retrieval_ledger")
    _require_exact_keys(
        ledger,
        {"version", "run_namespace", "global_ticket_namespace"},
        "retrieval_ledger",
    )
    retrieval_ledger_version = _text(ledger, "version", "retrieval_ledger")
    retrieval_run_namespace = _text(
        ledger,
        "run_namespace",
        "retrieval_ledger",
    )
    global_ticket_namespace = _text(
        ledger,
        "global_ticket_namespace",
        "retrieval_ledger",
    )
    _require_exact(
        retrieval_ledger_version,
        RETRIEVAL_LEDGER_VERSION,
        "retrieval_ledger.version",
    )
    _require_exact(
        retrieval_run_namespace,
        RETRIEVAL_RUN_NAMESPACE,
        "retrieval_ledger.run_namespace",
    )
    _require_exact(
        global_ticket_namespace,
        GLOBAL_TICKET_NAMESPACE,
        "retrieval_ledger.global_ticket_namespace",
    )

    expansion = _mapping(config.get("expansion"), "expansion")
    _require_exact_keys(
        expansion,
        {
            "formula_version",
            "artifact_version",
            "foreground_ranks",
            "background_ranks",
            "max_terms",
            "expand_parent_facet",
            "max_expanded_facets",
        },
        "expansion",
    )
    prf_formula_version = _text(expansion, "formula_version", "expansion")
    prf_artifact_version = _text(expansion, "artifact_version", "expansion")
    foreground_ranks = _rank_pair(
        expansion.get("foreground_ranks"),
        "expansion.foreground_ranks",
    )
    background_ranks = _rank_pair(
        expansion.get("background_ranks"),
        "expansion.background_ranks",
    )
    max_expansion_terms = _integer(expansion, "max_terms", "expansion")
    expand_parent_facet = _boolean(expansion, "expand_parent_facet", "expansion")
    max_expanded_facets = _integer(
        expansion,
        "max_expanded_facets",
        "expansion",
    )
    for name, actual, expected in (
        ("expansion.formula_version", prf_formula_version, PRF_FORMULA_VERSION),
        ("expansion.artifact_version", prf_artifact_version, PRF_ARTIFACT_VERSION),
        ("expansion.foreground_ranks", foreground_ranks, (1, 5)),
        ("expansion.background_ranks", background_ranks, (6, 50)),
        ("expansion.max_terms", max_expansion_terms, 2),
        ("expansion.expand_parent_facet", expand_parent_facet, False),
        ("expansion.max_expanded_facets", max_expanded_facets, 3),
    ):
        _require_exact(actual, expected, name)

    fusion = _mapping(config.get("fusion"), "fusion")
    _require_exact_keys(fusion, {"version", "k", "arms"}, "fusion")
    rrf_version = _text(fusion, "version", "fusion")
    rrf_k = _integer(fusion, "k", "fusion")
    arms = _string_tuple(fusion.get("arms"), "fusion.arms")
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
        ("cost.topic_count", cost.topic_count, PILOT_TOPIC_COUNT),
        ("cost.max_facets_per_topic", cost.max_facets_per_topic, 4),
        ("cost.max_unique_requests_per_topic", cost.max_unique_requests_per_topic, 9),
        ("cost.max_external_requests", cost.max_external_requests, 36),
        ("cost.model_calls", cost.model_calls, 0),
        ("cost.reranker_calls", cost.reranker_calls, 0),
    ):
        _require_exact(actual, expected, name)
    if cost.topic_count * cost.max_unique_requests_per_topic != cost.max_external_requests:
        raise ValueError("cost ceiling does not equal topic_count times per-topic ceiling")

    external_gate = _mapping(config.get("external_gate"), "external_gate")
    _require_exact_keys(external_gate, {"status", "reason"}, "external_gate")
    external_gate_status = _text(external_gate, "status", "external_gate")
    external_gate_reason = _text(external_gate, "reason", "external_gate")
    _require_exact(external_gate_status, EXTERNAL_GATE_STATUS, "external_gate.status")
    _require_exact(external_gate_reason, EXTERNAL_GATE_REASON, "external_gate.reason")
    if retrieval.index_revision != RETRIEVAL_INDEX_REVISION:
        raise ValueError("external gate requires the frozen unknown index revision")

    evaluation_raw = _mapping(config.get("evaluation"), "evaluation")
    _require_exact_keys(
        evaluation_raw,
        {"qrels", "relevance_threshold", "metrics", "firewall"},
        "evaluation",
    )
    qrels_path_text = _text(evaluation_raw, "qrels", "evaluation")
    metrics = _string_tuple(evaluation_raw.get("metrics"), "evaluation.metrics")
    relevance_threshold = _integer(
        evaluation_raw,
        "relevance_threshold",
        "evaluation",
    )
    firewall = _text(evaluation_raw, "firewall", "evaluation")
    _require_exact(qrels_path_text, QRELS_PATH, "evaluation.qrels")
    _require_exact(relevance_threshold, 2, "evaluation.relevance_threshold")
    _require_exact(metrics, EVALUATION_METRICS, "evaluation.metrics")
    _require_exact(firewall, "frozen_manifest_required", "evaluation.firewall")
    evaluation = EvaluationConfig(
        qrels=_resolve_input(root, qrels_path_text),
        relevance_threshold=relevance_threshold,
        metrics=metrics,
        firewall=firewall,
    )

    return DetSparseV2Config(
        root_dir=root,
        config_path=config_path,
        schema_version=schema_version,
        experiment_id=experiment_id,
        output_dir=_resolve_output(root, output_path_text),
        topics_path=_resolve_input(root, topics_path_text),
        topics_format=topics_format,
        candidate_topic_ids=candidate_ids,
        excluded_topic_ids=excluded_ids,
        selection_version=selection_version,
        selection_seed=selection_seed,
        selection_strata=selection_strata,
        selected_topic_count=selected_topic_count,
        planner_version=planner_version,
        renderer_version=renderer_version,
        splitter_version=splitter_version,
        tokenizer_version=tokenizer_version,
        max_facets=max_facets,
        min_facet_terms=min_facet_terms,
        parent_min_unique_terms=parent_min_unique_terms,
        bounded_context_min_unique_terms=bounded_min,
        bounded_context_max_unique_terms=bounded_max,
        bounded_context_allocation=bounded_allocation,
        analyzer=analyzer,
        retrieval=retrieval,
        retrieval_ledger_version=retrieval_ledger_version,
        retrieval_run_namespace=retrieval_run_namespace,
        global_ticket_namespace=global_ticket_namespace,
        prf_formula_version=prf_formula_version,
        prf_artifact_version=prf_artifact_version,
        foreground_ranks=foreground_ranks,
        background_ranks=background_ranks,
        max_expansion_terms=max_expansion_terms,
        expand_parent_facet=expand_parent_facet,
        max_expanded_facets=max_expanded_facets,
        rrf_version=rrf_version,
        rrf_k=rrf_k,
        arms=arms,
        cost=cost,
        external_gate_status=external_gate_status,
        external_gate_reason=external_gate_reason,
        evaluation=evaluation,
    )
