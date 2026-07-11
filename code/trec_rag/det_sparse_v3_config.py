"""Fail-closed configuration contract for deterministic sparse retrieval v3.

Loading this module or its tracked YAML does not inspect topics, qrels, prior
outputs, or retrieval state.  The loader accepts only the canonical v3 path and
requires every policy value and value type to equal the committed contract.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from trec_rag.query_analyzer import QueryAnalyzer
from trec_rag.repo_env import find_repo_root


SCHEMA_VERSION = "det_sparse_v3"
PLANNER_VERSION = "det_sparse_v3"
RENDERER_VERSION = "det_sparse_recurrent_anchor_renderer_v3"
SPLITTER_VERSION = "det_sparse_exact_span_splitter_v1"
TOKENIZER_VERSION = "narrative_token_tape_v1"
ANCHOR_SELECTOR_VERSION = "det_sparse_cross_unit_recurrent_anchor_v1"
SELECTION_VERSION = "det_sparse_anchor_critical_quantile_selection_v3"
SELECTION_SEED = "det_sparse_v3_recurrent_anchor_selection_20260711"
PILOT_TOPIC_COUNT = 4
QUANTILE_BIN_COUNT = 3

CANDIDATE_TOPIC_IDS = (
    "14",
    "31",
    "58",
    "72",
    "219",
    "233",
    "273",
    "477",
    "499",
)
EXCLUDED_TOPIC_IDS = (
    "37",
    "84",
    "144",
    "161",
    "200",
    "213",
    "224",
    "225",
    "300",
    "407",
    "515",
    "707",
    "897",
)

CONVERSATIONAL_SURFACE_VERSION = "det_sparse_conversational_surfaces_v1"
CONVERSATIONAL_NORMALIZATION_VERSION = "unicode_nfkc_casefold_v1"
CONVERSATIONAL_SURFACES = (
    "a",
    "about",
    "also",
    "an",
    "and",
    "answer",
    "are",
    "ask",
    "asking",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "curious",
    "deeper",
    "describe",
    "description",
    "detailed",
    "did",
    "discuss",
    "discussion",
    "do",
    "does",
    "explain",
    "explanation",
    "finally",
    "for",
    "from",
    "gain",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "hope",
    "hoping",
    "how",
    "i",
    "i'd",
    "i’d",
    "i'm",
    "i’m",
    "identify",
    "in",
    "information",
    "interest",
    "interested",
    "is",
    "it",
    "its",
    "know",
    "knowing",
    "learn",
    "learning",
    "like",
    "liked",
    "list",
    "look",
    "looking",
    "may",
    "me",
    "might",
    "must",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "overview",
    "please",
    "provide",
    "question",
    "report",
    "say",
    "shall",
    "she",
    "should",
    "tell",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "these",
    "they",
    "this",
    "those",
    "to",
    "understand",
    "understanding",
    "want",
    "wanted",
    "wants",
    "was",
    "we",
    "we're",
    "we’re",
    "were",
    "what",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
)
CONVERSATIONAL_SURFACE_COUNT = 114
CONVERSATIONAL_SOURCE_SHA256 = (
    "b317e77660c066025c43359d52f04d6da9370c41b744dec1e1fbde6802cc23c8"
)
CONVERSATIONAL_NORMALIZED_SHA256 = (
    "d89564c9eab834c466078a1dca10adf1c33b5ed4c92dd7bed08f4fdefac6697c"
)
CONVERSATIONAL_PROJECTION_ENCODING = (
    "compact_json_source_normalized_analyzer_tokens_projection_v1"
)
CONVERSATIONAL_PROJECTION_SHA256 = (
    "752ce66fae9cbfc2038191d8beb068d84a8c1f70928b472dfa3ceaff63c68b23"
)
CONVERSATIONAL_PROJECTED_TERM_COUNT = 75
CONVERSATIONAL_PROJECTED_TERMS_SHA256 = (
    "4dc62d6d589b79e2ef80adad89ca8830d570f9ded37066e83dbb53d2237aac19"
)

CANDIDATE_EVIDENCE_VERSION = "det_sparse_candidate_evidence_sha256_v1"
CANDIDATE_EVIDENCE_ENCODING = "compact_sorted_key_utf8_json_no_newline"
CANDIDATE_EVIDENCE_KEYS = (
    "end",
    "narrative_sha256",
    "start",
    "text_sha256",
    "token_ids",
)

EXPERIMENT_ID = "rag25_det_sparse_structural4_v3"
CONFIG_PATH = "configs/det_sparse_v3.yaml"
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
ANALYZER_FINGERPRINT_SHA256 = (
    "f9bbd4e7af26c532105f6dd7e49ce15fa11afd1f0fe7d387847ce41ff7d8def4"
)
RETRIEVAL_ENDPOINT_URL = (
    "https://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
)
RETRIEVAL_INDEX = "climbmix-400b"
RETRIEVAL_INDEX_REVISION = "hosted_climbmix_unknown_revision"
RETRIEVAL_LEDGER_VERSION = "det_sparse_retrieval_ledger_v1"
RETRIEVAL_RUN_NAMESPACE = "det_sparse_v3_fresh_run_local"
GLOBAL_TICKET_NAMESPACE = EXPERIMENT_ID
PRF_FORMULA_VERSION = "det_sparse_prf_v1"
PRF_ARTIFACT_VERSION = "det_sparse_prf_artifact_v3"
RRF_VERSION = "weighted_rrf_v1"
ARM_NAMES = ("O", "F", "E", "FE")
EXTERNAL_GATE_STATUS = "blocked"
EXTERNAL_GATE_REASON = "hosted_index_revision_unknown"
EVALUATION_METRICS = (
    "recall@100",
    "graded_recall@100",
    "ideal_dcg_coverage@100",
    "recall@50",
    "ndcg@10",
)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_conversational_surface(surface: str) -> str:
    """Apply the sole v3 surface-inventory normalization."""

    if not isinstance(surface, str):
        raise TypeError("surface must be text")
    return unicodedata.normalize("NFKC", surface).casefold()


NORMALIZED_CONVERSATIONAL_SURFACES = tuple(
    sorted(
        {normalize_conversational_surface(item) for item in CONVERSATIONAL_SURFACES},
        key=lambda item: item.encode("utf-8"),
    )
)

if len(CONVERSATIONAL_SURFACES) != CONVERSATIONAL_SURFACE_COUNT:
    raise RuntimeError("v3 conversational source inventory count drifted")
if len(set(CONVERSATIONAL_SURFACES)) != CONVERSATIONAL_SURFACE_COUNT:
    raise RuntimeError("v3 conversational source inventory contains duplicates")
if len(NORMALIZED_CONVERSATIONAL_SURFACES) != CONVERSATIONAL_SURFACE_COUNT:
    raise RuntimeError("v3 normalized conversational inventory count drifted")
if _canonical_json_sha256(list(CONVERSATIONAL_SURFACES)) != CONVERSATIONAL_SOURCE_SHA256:
    raise RuntimeError("v3 conversational source inventory hash drifted")
if (
    _canonical_json_sha256(list(NORMALIZED_CONVERSATIONAL_SURFACES))
    != CONVERSATIONAL_NORMALIZED_SHA256
):
    raise RuntimeError("v3 normalized conversational inventory hash drifted")


@dataclass(frozen=True)
class ConversationalProjectionRecord:
    source_surface: str
    normalized_surface: str
    analyzer_tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "analyzer_tokens": list(self.analyzer_tokens),
            "normalized_surface": self.normalized_surface,
            "source_surface": self.source_surface,
        }


@dataclass(frozen=True)
class ConversationalProjection:
    encoding: str
    records: tuple[ConversationalProjectionRecord, ...]
    sha256: str
    projected_terms: tuple[str, ...]
    projected_terms_sha256: str
    analyzer_fingerprint_sha256: str


def build_conversational_projection(
    query_analyzer: QueryAnalyzer,
) -> ConversationalProjection:
    """Compute and verify the frozen surface projection before topic access."""

    fingerprint = query_analyzer.fingerprint
    fingerprint_sha256 = _canonical_json_sha256(fingerprint.to_dict())
    if fingerprint_sha256 != ANALYZER_FINGERPRINT_SHA256:
        raise ValueError("conversational projection analyzer fingerprint drifted")
    records: list[ConversationalProjectionRecord] = []
    projected_terms: set[str] = set()
    for surface in CONVERSATIONAL_SURFACES:
        analyzed = query_analyzer.analyze(surface)
        if analyzed.fingerprint != fingerprint:
            raise ValueError("conversational projection analyzer fingerprint changed")
        record = ConversationalProjectionRecord(
            source_surface=surface,
            normalized_surface=normalize_conversational_surface(surface),
            analyzer_tokens=tuple(analyzed.tokens),
        )
        records.append(record)
        projected_terms.update(record.analyzer_tokens)
    ordered_terms = tuple(
        sorted(projected_terms, key=lambda item: item.encode("utf-8"))
    )
    projection_sha256 = _canonical_json_sha256(
        [record.to_dict() for record in records]
    )
    projected_terms_sha256 = _canonical_json_sha256(list(ordered_terms))
    if projection_sha256 != CONVERSATIONAL_PROJECTION_SHA256:
        raise ValueError("conversational surface-to-analyzer projection drifted")
    if len(ordered_terms) != CONVERSATIONAL_PROJECTED_TERM_COUNT:
        raise ValueError("conversational projected term count drifted")
    if projected_terms_sha256 != CONVERSATIONAL_PROJECTED_TERMS_SHA256:
        raise ValueError("conversational projected term set drifted")
    return ConversationalProjection(
        encoding=CONVERSATIONAL_PROJECTION_ENCODING,
        records=tuple(records),
        sha256=projection_sha256,
        projected_terms=ordered_terms,
        projected_terms_sha256=projected_terms_sha256,
        analyzer_fingerprint_sha256=fingerprint_sha256,
    )


@dataclass(frozen=True)
class SelectionConfig:
    version: str
    seed: str
    selected_count: int
    critical_label: str
    critical_intersection: str
    digest: str
    remaining_sort: tuple[str, ...]
    binning: str
    bin_count: int
    merge_masked_policy: str


@dataclass(frozen=True)
class CandidateWindowConfig:
    max_token_records: int
    min_unique_terms: int
    max_unique_terms: int
    min_recurrent_terms: int
    recurrence_precision_numerator: int
    recurrence_precision_denominator: int
    min_joint_child_df: int
    max_analyzed_occurrences: int
    parent_occurrence_divisor: int
    maximal_core_only: bool
    disjoint_maximal_core_policy: str
    tie_break: tuple[str, ...]


@dataclass(frozen=True)
class CandidateEvidenceConfig:
    version: str
    hash: str
    encoding: str
    keys: tuple[str, ...]
    offset_unit: str
    token_id_semantics: str


@dataclass(frozen=True)
class CriticalityConfig:
    selection_intersection: str
    audit_intersection: str
    critical_label: str
    min_child_payload_terms: int
    merge_masked_policy: str


@dataclass(frozen=True)
class ConversationalSurfaceConfig:
    version: str
    normalization: str
    hash_encoding: str
    normalized_order: str
    surfaces: tuple[str, ...]
    source_count: int
    source_sha256: str
    normalized_surfaces: tuple[str, ...]
    normalized_count: int
    normalized_sha256: str
    projection_encoding: str
    expected_projection_sha256: str
    projected_terms_order: str
    projected_term_count: int
    projected_terms_sha256: str
    projection_policy: str


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
class DetSparseV3Config:
    root_dir: Path
    config_path: Path
    schema_version: str
    experiment_id: str
    output_dir: Path
    topics_path: Path
    topics_format: str
    candidate_topic_ids: tuple[str, ...]
    excluded_topic_ids: tuple[str, ...]
    selection: SelectionConfig
    planner_version: str
    renderer_version: str
    splitter_version: str
    tokenizer_version: str
    anchor_selector_version: str
    max_facets: int
    min_facet_terms: int
    min_units: int
    parent_min_unique_terms: int
    candidate_window: CandidateWindowConfig
    candidate_evidence: CandidateEvidenceConfig
    criticality: CriticalityConfig
    conversational_surfaces: ConversationalSurfaceConfig
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

    @property
    def selection_version(self) -> str:
        return self.selection.version

    @property
    def selection_seed(self) -> str:
        return self.selection.seed

    @property
    def selected_topic_count(self) -> int:
        return self.selection.selected_count


class _StrictLoader(yaml.SafeLoader):
    """Safe YAML loader rejecting duplicate keys and aliases."""

    def compose_node(self, parent: object, index: object) -> yaml.Node:
        if self.check_event(AliasEvent):
            event = self.get_event()
            raise ConstructorError(
                "while composing strict v3 YAML",
                event.start_mark,
                "YAML aliases are forbidden",
                event.start_mark,
            )
        return super().compose_node(parent, index)


def _construct_unique_mapping(
    loader: _StrictLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    if not isinstance(node, MappingNode):
        raise ConstructorError(None, None, "expected a mapping node", node.start_mark)
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _expected_config() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": {"id": EXPERIMENT_ID, "output_dir": OUTPUT_PATH},
        "topics": {
            "path": TOPICS_PATH,
            "format": "tsv",
            "candidate_ids": list(CANDIDATE_TOPIC_IDS),
            "excluded_ids": list(EXCLUDED_TOPIC_IDS),
            "selection": {
                "version": SELECTION_VERSION,
                "seed": SELECTION_SEED,
                "selected_count": PILOT_TOPIC_COUNT,
                "critical_label": "anchorless",
                "critical_intersection": "full_aligned_analyzer_terms",
                "digest": "sha256_seed_nul_topic_nul_narrative_sha_v1",
                "remaining_sort": [
                    "capped_unit_count",
                    "original_unique_term_count",
                    "facet_count",
                    "anchor_core_term_count",
                    "numeric_topic_id",
                ],
                "binning": "half_open_floor_thirds_v1",
                "bin_count": QUANTILE_BIN_COUNT,
                "merge_masked_policy": "stop_no_replacement",
            },
        },
        "query_planning": {
            "planner_version": PLANNER_VERSION,
            "renderer_version": RENDERER_VERSION,
            "splitter_version": SPLITTER_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "anchor_selector_version": ANCHOR_SELECTOR_VERSION,
            "max_facets": 4,
            "min_facet_terms": 3,
            "min_units": 2,
            "parent_min_unique_terms": 4,
            "candidate_window": {
                "max_token_records": 6,
                "min_unique_terms": 2,
                "max_unique_terms": 4,
                "min_recurrent_terms": 2,
                "recurrence_precision_numerator": 2,
                "recurrence_precision_denominator": 3,
                "min_joint_child_df": 1,
                "max_analyzed_occurrences": 6,
                "parent_occurrence_divisor": 2,
                "maximal_core_only": True,
                "disjoint_maximal_core_policy": "original_only_ambiguous",
                "tie_break": [
                    "max_core_score",
                    "later_start",
                    "shorter_unicode_codepoint_span",
                    "smaller_unsigned_utf8",
                ],
            },
            "candidate_evidence": {
                "version": CANDIDATE_EVIDENCE_VERSION,
                "hash": "sha256",
                "encoding": CANDIDATE_EVIDENCE_ENCODING,
                "keys": list(CANDIDATE_EVIDENCE_KEYS),
                "offset_unit": "unicode_code_points",
                "token_id_semantics": "ordered_zero_based_tape_indices",
            },
            "criticality": {
                "selection_intersection": "full_aligned_analyzer_terms",
                "audit_intersection": "eligible_nonconversational_terms",
                "critical_label": "anchorless",
                "min_child_payload_terms": 2,
                "merge_masked_policy": "stop_no_replacement",
            },
            "conversational_surfaces": {
                "version": CONVERSATIONAL_SURFACE_VERSION,
                "normalization": CONVERSATIONAL_NORMALIZATION_VERSION,
                "hash_encoding": "compact_json_utf8_sha256_v1",
                "normalized_order": "unsigned_utf8_lexicographic",
                "source_count": CONVERSATIONAL_SURFACE_COUNT,
                "source_sha256": CONVERSATIONAL_SOURCE_SHA256,
                "normalized_count": CONVERSATIONAL_SURFACE_COUNT,
                "normalized_sha256": CONVERSATIONAL_NORMALIZED_SHA256,
                "projection_encoding": CONVERSATIONAL_PROJECTION_ENCODING,
                "expected_projection_sha256": CONVERSATIONAL_PROJECTION_SHA256,
                "projected_terms_order": "unsigned_utf8_lexicographic",
                "projected_term_count": CONVERSATIONAL_PROJECTED_TERM_COUNT,
                "projected_terms_sha256": CONVERSATIONAL_PROJECTED_TERMS_SHA256,
                "projection_policy": "verify_attested_analyzer_before_candidates",
                "surfaces": list(CONVERSATIONAL_SURFACES),
            },
        },
        "analyzer": {
            "url": ANALYZER_URL,
            "expected_fingerprint_sha256": ANALYZER_FINGERPRINT_SHA256,
        },
        "retrieval": {
            "type": "pyserini_remote_raw_first_v1",
            "endpoint_env": "INDEX_URL",
            "endpoint_url": RETRIEVAL_ENDPOINT_URL,
            "index": RETRIEVAL_INDEX,
            "index_revision": RETRIEVAL_INDEX_REVISION,
            "hits": 100,
            "required_results": 100,
            "max_attempts": 1,
            "retry_policy": "none",
            "redirect_policy": "none",
            "cache_policy": "fresh_run_local",
        },
        "retrieval_ledger": {
            "version": RETRIEVAL_LEDGER_VERSION,
            "run_namespace": RETRIEVAL_RUN_NAMESPACE,
            "global_ticket_namespace": GLOBAL_TICKET_NAMESPACE,
        },
        "expansion": {
            "formula_version": PRF_FORMULA_VERSION,
            "artifact_version": PRF_ARTIFACT_VERSION,
            "foreground_ranks": [1, 5],
            "background_ranks": [6, 50],
            "max_terms": 2,
            "expand_parent_facet": False,
            "max_expanded_facets": 3,
        },
        "fusion": {"version": RRF_VERSION, "k": 60, "arms": list(ARM_NAMES)},
        "cost": {
            "topic_count": PILOT_TOPIC_COUNT,
            "max_facets_per_topic": 4,
            "max_unique_requests_per_topic": 9,
            "max_external_requests": 36,
            "model_calls": 0,
            "reranker_calls": 0,
        },
        "external_gate": {
            "status": EXTERNAL_GATE_STATUS,
            "reason": EXTERNAL_GATE_REASON,
        },
        "evaluation": {
            "qrels": QRELS_PATH,
            "relevance_threshold": 2,
            "metrics": list(EVALUATION_METRICS),
            "firewall": "frozen_manifest_required",
        },
    }


def _validate_exact(actual: object, expected: object, path: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(
            f"{path} has frozen type {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        )
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unknown = sorted(actual_keys - expected_keys)
            details = []
            if missing:
                details.append("missing=" + ",".join(str(item) for item in missing))
            if unknown:
                details.append("unknown=" + ",".join(str(item) for item in unknown))
            raise ValueError(f"{path} keys differ ({'; '.join(details)})")
        for key in expected:
            _validate_exact(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        if len(actual) != len(expected):
            raise ValueError(
                f"{path} length is frozen at {len(expected)}, got {len(actual)}"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _validate_exact(actual_item, expected_item, f"{path}[{index}]")
        return
    if actual != expected:
        raise ValueError(f"{path} is frozen at {expected!r}, got {actual!r}")


def _validate_urls(raw: dict[str, Any]) -> None:
    analyzer = urlsplit(raw["analyzer"]["url"])
    if (
        analyzer.scheme != "http"
        or analyzer.hostname != "127.0.0.1"
        or analyzer.port != 18081
        or analyzer.username is not None
        or analyzer.password is not None
        or analyzer.path not in {"", "/"}
        or analyzer.query
        or analyzer.fragment
    ):
        raise ValueError("config.analyzer.url is not the frozen loopback analyzer")
    retrieval = urlsplit(raw["retrieval"]["endpoint_url"])
    if (
        retrieval.scheme != "https"
        or retrieval.hostname != "api.castorini.uwaterloo.ca"
        or retrieval.port not in {None, 443}
        or retrieval.username is not None
        or retrieval.password is not None
        or retrieval.path != "/v1/climbmix-400b/search"
        or retrieval.query
        or retrieval.fragment
    ):
        raise ValueError("config.retrieval.endpoint_url is not the frozen endpoint")


def _as_tuple(value: list[Any]) -> tuple[Any, ...]:
    return tuple(value)


def load_det_sparse_v3_config(path: Path) -> DetSparseV3Config:
    """Load only the exact tracked v3 YAML without opening data inputs."""

    config_path = path.resolve()
    root = find_repo_root(config_path.parent)
    expected_path = (root / CONFIG_PATH).resolve()
    if config_path != expected_path:
        raise ValueError(f"config path is frozen at {expected_path}, got {config_path}")
    raw = yaml.load(
        config_path.read_text(encoding="utf-8"),
        Loader=_StrictLoader,
    )
    expected = _expected_config()
    _validate_exact(raw, expected, "config")
    assert isinstance(raw, dict)
    _validate_urls(raw)

    topics = raw["topics"]
    selection_raw = topics["selection"]
    planning = raw["query_planning"]
    window_raw = planning["candidate_window"]
    evidence_raw = planning["candidate_evidence"]
    criticality_raw = planning["criticality"]
    surfaces_raw = planning["conversational_surfaces"]
    analyzer_raw = raw["analyzer"]
    retrieval_raw = raw["retrieval"]
    ledger_raw = raw["retrieval_ledger"]
    expansion_raw = raw["expansion"]
    fusion_raw = raw["fusion"]
    cost_raw = raw["cost"]
    evaluation_raw = raw["evaluation"]

    selection = SelectionConfig(
        version=selection_raw["version"],
        seed=selection_raw["seed"],
        selected_count=selection_raw["selected_count"],
        critical_label=selection_raw["critical_label"],
        critical_intersection=selection_raw["critical_intersection"],
        digest=selection_raw["digest"],
        remaining_sort=_as_tuple(selection_raw["remaining_sort"]),
        binning=selection_raw["binning"],
        bin_count=selection_raw["bin_count"],
        merge_masked_policy=selection_raw["merge_masked_policy"],
    )
    candidate_window = CandidateWindowConfig(
        max_token_records=window_raw["max_token_records"],
        min_unique_terms=window_raw["min_unique_terms"],
        max_unique_terms=window_raw["max_unique_terms"],
        min_recurrent_terms=window_raw["min_recurrent_terms"],
        recurrence_precision_numerator=window_raw[
            "recurrence_precision_numerator"
        ],
        recurrence_precision_denominator=window_raw[
            "recurrence_precision_denominator"
        ],
        min_joint_child_df=window_raw["min_joint_child_df"],
        max_analyzed_occurrences=window_raw["max_analyzed_occurrences"],
        parent_occurrence_divisor=window_raw["parent_occurrence_divisor"],
        maximal_core_only=window_raw["maximal_core_only"],
        disjoint_maximal_core_policy=window_raw[
            "disjoint_maximal_core_policy"
        ],
        tie_break=_as_tuple(window_raw["tie_break"]),
    )
    candidate_evidence = CandidateEvidenceConfig(
        version=evidence_raw["version"],
        hash=evidence_raw["hash"],
        encoding=evidence_raw["encoding"],
        keys=_as_tuple(evidence_raw["keys"]),
        offset_unit=evidence_raw["offset_unit"],
        token_id_semantics=evidence_raw["token_id_semantics"],
    )
    criticality = CriticalityConfig(
        selection_intersection=criticality_raw["selection_intersection"],
        audit_intersection=criticality_raw["audit_intersection"],
        critical_label=criticality_raw["critical_label"],
        min_child_payload_terms=criticality_raw["min_child_payload_terms"],
        merge_masked_policy=criticality_raw["merge_masked_policy"],
    )
    conversational = ConversationalSurfaceConfig(
        version=surfaces_raw["version"],
        normalization=surfaces_raw["normalization"],
        hash_encoding=surfaces_raw["hash_encoding"],
        normalized_order=surfaces_raw["normalized_order"],
        surfaces=_as_tuple(surfaces_raw["surfaces"]),
        source_count=surfaces_raw["source_count"],
        source_sha256=surfaces_raw["source_sha256"],
        normalized_surfaces=NORMALIZED_CONVERSATIONAL_SURFACES,
        normalized_count=surfaces_raw["normalized_count"],
        normalized_sha256=surfaces_raw["normalized_sha256"],
        projection_encoding=surfaces_raw["projection_encoding"],
        expected_projection_sha256=surfaces_raw["expected_projection_sha256"],
        projected_terms_order=surfaces_raw["projected_terms_order"],
        projected_term_count=surfaces_raw["projected_term_count"],
        projected_terms_sha256=surfaces_raw["projected_terms_sha256"],
        projection_policy=surfaces_raw["projection_policy"],
    )

    analyzer = AnalyzerConfig(
        url=analyzer_raw["url"],
        expected_fingerprint_sha256=analyzer_raw["expected_fingerprint_sha256"],
    )
    retrieval = RetrievalConfig(**retrieval_raw)
    cost = CostConfig(**cost_raw)
    evaluation = EvaluationConfig(
        qrels=root / evaluation_raw["qrels"],
        relevance_threshold=evaluation_raw["relevance_threshold"],
        metrics=_as_tuple(evaluation_raw["metrics"]),
        firewall=evaluation_raw["firewall"],
    )
    return DetSparseV3Config(
        root_dir=root,
        config_path=config_path,
        schema_version=raw["schema_version"],
        experiment_id=raw["experiment"]["id"],
        output_dir=root / raw["experiment"]["output_dir"],
        topics_path=root / topics["path"],
        topics_format=topics["format"],
        candidate_topic_ids=_as_tuple(topics["candidate_ids"]),
        excluded_topic_ids=_as_tuple(topics["excluded_ids"]),
        selection=selection,
        planner_version=planning["planner_version"],
        renderer_version=planning["renderer_version"],
        splitter_version=planning["splitter_version"],
        tokenizer_version=planning["tokenizer_version"],
        anchor_selector_version=planning["anchor_selector_version"],
        max_facets=planning["max_facets"],
        min_facet_terms=planning["min_facet_terms"],
        min_units=planning["min_units"],
        parent_min_unique_terms=planning["parent_min_unique_terms"],
        candidate_window=candidate_window,
        candidate_evidence=candidate_evidence,
        criticality=criticality,
        conversational_surfaces=conversational,
        analyzer=analyzer,
        retrieval=retrieval,
        retrieval_ledger_version=ledger_raw["version"],
        retrieval_run_namespace=ledger_raw["run_namespace"],
        global_ticket_namespace=ledger_raw["global_ticket_namespace"],
        prf_formula_version=expansion_raw["formula_version"],
        prf_artifact_version=expansion_raw["artifact_version"],
        foreground_ranks=tuple(expansion_raw["foreground_ranks"]),
        background_ranks=tuple(expansion_raw["background_ranks"]),
        max_expansion_terms=expansion_raw["max_terms"],
        expand_parent_facet=expansion_raw["expand_parent_facet"],
        max_expanded_facets=expansion_raw["max_expanded_facets"],
        rrf_version=fusion_raw["version"],
        rrf_k=fusion_raw["k"],
        arms=_as_tuple(fusion_raw["arms"]),
        cost=cost,
        external_gate_status=raw["external_gate"]["status"],
        external_gate_reason=raw["external_gate"]["reason"],
        evaluation=evaluation,
    )
