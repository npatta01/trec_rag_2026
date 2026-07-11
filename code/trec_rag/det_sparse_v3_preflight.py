"""Create-only, qrels-blind recurrent-anchor preflight for sparse v3.

The public builder owns its Lucene client and has no retrieval, model,
reranker, qrels, cache, or prior-output input.  It first verifies and freezes
the conversational inventory, then reads exactly the nine candidate records.
Every candidate plan is retained so alignment, recurrence, candidate-window,
and recurrent-core evidence remain independently auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from trec_rag.det_sparse_freeze import create_freeze_manifest, verify_freeze_manifest
from trec_rag.det_sparse_v3_config import (
    ANALYZER_FINGERPRINT_SHA256,
    ARM_NAMES,
    CANDIDATE_EVIDENCE_ENCODING,
    CANDIDATE_EVIDENCE_KEYS,
    CANDIDATE_EVIDENCE_VERSION,
    CANDIDATE_TOPIC_IDS,
    CONVERSATIONAL_NORMALIZATION_VERSION,
    CONVERSATIONAL_NORMALIZED_SHA256,
    CONVERSATIONAL_SURFACES,
    CONVERSATIONAL_PROJECTED_TERM_COUNT,
    CONVERSATIONAL_PROJECTED_TERMS_SHA256,
    CONVERSATIONAL_PROJECTION_ENCODING,
    CONVERSATIONAL_PROJECTION_SHA256,
    CONVERSATIONAL_SOURCE_SHA256,
    CONVERSATIONAL_SURFACE_COUNT,
    CONVERSATIONAL_SURFACE_VERSION,
    EXCLUDED_TOPIC_IDS,
    OUTPUT_PATH,
    PILOT_TOPIC_COUNT,
    NORMALIZED_CONVERSATIONAL_SURFACES,
    DetSparseV3Config,
    ConversationalProjection,
    build_conversational_projection,
    load_det_sparse_v3_config,
)
from trec_rag.det_sparse_v3_provenance import (
    current_runtime_provenance,
    current_source_provenance,
)
from trec_rag.det_sparse_v3_selection import (
    CandidateScreen,
    SelectionOutcomeV3,
    screen_and_select_structural_topics_v3,
    semantic_plan_sha256,
)
from trec_rag.deterministic_sparse import SPLITTER_VERSION, TOKENIZER_VERSION
from trec_rag.deterministic_sparse_v3 import (
    ANCHOR_SELECTOR_VERSION,
    MAX_FACETS,
    MIN_PARENT_UNIQUE_TERMS,
    PLANNER_VERSION,
    RENDERER_VERSION,
    SELECTION_VERSION,
    DeterministicSparsePlanV3,
)
from trec_rag.query_analyzer import QueryAnalyzer, RemoteLuceneQueryAnalyzer
from trec_rag.topics import Topic, derive_title


PREFLIGHT_SCHEMA_VERSION = "det_sparse_v3_preflight_v1"
RESERVATION_SCHEMA_VERSION = "det_sparse_v3_preflight_reservation_v1"
INVENTORY_SCHEMA_VERSION = "det_sparse_v3_conversational_inventory_freeze_v1"
CRITICALITY_SCHEMA_VERSION = "det_sparse_v3_criticality_ledger_v1"
COMPLETION_SCHEMA_VERSION = "det_sparse_v3_preflight_completion_v1"
COMPLETION_PROTOCOL_VERSION = "terminal_create_only_after_postseal_replay_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_METADATA_KEYS = {
    "schema_version",
    "experiment_id",
    "config_path",
    "config_sha256",
    "topic_source_path",
    "candidate_topic_ids",
    "excluded_topic_ids",
    "selected_topic_ids",
    "critical_topic_id",
    "selected_topic_count",
    "completion_protocol_version",
    "completion_path",
    "completion_required",
    "reservation_path",
    "reservation_sha256",
    "inventory_path",
    "inventory_artifact_sha256",
    "inventory_projection_sha256",
    "inventory_projected_terms_sha256",
    "inventory_projected_term_count",
    "selection_path",
    "selection_artifact_sha256",
    "candidate_screen_path",
    "candidate_screen_artifact_sha256",
    "candidate_plans",
    "selected_plans",
    "criticality_path",
    "criticality_artifact_sha256",
    "planner_version",
    "renderer_version",
    "anchor_selector_version",
    "selection_version",
    "splitter_version",
    "tokenizer_version",
    "candidate_evidence_version",
    "candidate_evidence_encoding",
    "candidate_evidence_keys",
    "conversational_surface_version",
    "conversational_normalization_version",
    "conversational_source_count",
    "conversational_source_sha256",
    "conversational_normalized_sha256",
    "max_facets",
    "parent_min_unique_terms",
    "required_results",
    "max_attempts",
    "retry_policy",
    "redirect_policy",
    "retrieval_run_namespace",
    "global_ticket_namespace",
    "analyzer_fingerprint",
    "analyzer_fingerprint_sha256",
    "arms",
    "queries_path",
    "queries_artifact_sha256",
    "query_registry_path",
    "query_registry_artifact_sha256",
    "coverage_paths_path",
    "coverage_paths_artifact_sha256",
    "planned_base_unique_requests",
    "request_projections",
    "derived_max_total_unique_requests",
    "hard_external_request_ceiling",
    "hard_per_topic_request_ceiling",
    "external_gate_status",
    "external_gate_reason",
    "mechanical_valid",
    "fallback_topic_ids",
    "external_calls",
    "model_calls",
    "reranker_calls",
    "qrels_opened",
    "source",
    "runtime",
}

_FREEZE_METADATA_FIELDS = (
    "experiment_id",
    "candidate_topic_ids",
    "excluded_topic_ids",
    "selected_topic_ids",
    "critical_topic_id",
    "selected_topic_count",
    "completion_protocol_version",
    "completion_path",
    "completion_required",
    "planner_version",
    "renderer_version",
    "anchor_selector_version",
    "selection_version",
    "splitter_version",
    "tokenizer_version",
    "candidate_evidence_version",
    "candidate_evidence_encoding",
    "candidate_evidence_keys",
    "conversational_surface_version",
    "conversational_normalization_version",
    "conversational_source_count",
    "conversational_source_sha256",
    "conversational_normalized_sha256",
    "inventory_projection_sha256",
    "inventory_projected_terms_sha256",
    "inventory_projected_term_count",
    "retrieval_run_namespace",
    "global_ticket_namespace",
    "analyzer_fingerprint_sha256",
    "mechanical_valid",
    "fallback_topic_ids",
    "planned_base_unique_requests",
    "request_projections",
    "derived_max_total_unique_requests",
    "hard_external_request_ceiling",
    "hard_per_topic_request_ceiling",
    "external_gate_status",
    "external_gate_reason",
    "external_calls",
    "model_calls",
    "reranker_calls",
    "qrels_opened",
)


@dataclass(frozen=True)
class CandidateTopic:
    topic: Topic
    source_line_number: int
    source_line_sha256: str


@dataclass(frozen=True)
class V3PreflightResult:
    output_dir: Path
    selected_topic_ids: tuple[str, ...]
    selected_plans: tuple[DeterministicSparsePlanV3, ...]
    valid: bool
    planned_base_requests: int
    derived_max_total_requests: int
    hard_external_request_ceiling: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_normalize(value: object) -> object:
    return json.loads(_canonical_bytes(value))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=reject_constant,
    )


def _require_exact_keys(
    value: object,
    expected: set[str],
    owner: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{owner} keys differ from the frozen schema")
    return value


def _require_exact_value(actual: object, expected: object, owner: str) -> None:
    """Compare JSON values without Python's bool/int equality coercion."""

    if type(actual) is not type(expected):
        raise ValueError(f"{owner} value type differs from the frozen replay")
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        if set(actual) != set(expected):
            raise ValueError(f"{owner} keys differ from the frozen replay")
        for key in expected:
            _require_exact_value(actual[key], expected[key], f"{owner}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple))
        if len(actual) != len(expected):
            raise ValueError(f"{owner} length differs from the frozen replay")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_exact_value(
                actual_item,
                expected_item,
                f"{owner}[{index}]",
            )
    elif actual != expected:
        raise ValueError(f"{owner} differs from the frozen replay")


def _strict_int(value: object, owner: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{owner} must be an integer, not bool/float coercion")
    return value


def _strict_bool(value: object, owner: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{owner} must be boolean")
    return value


def _strict_string_list(value: object, owner: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{owner} must be a non-empty string list")
    return value


def _require_sha256(value: object, owner: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{owner} must be a lowercase SHA-256")
    return value


def _create_only(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _fixed_artifact_path(
    output: Path,
    stored_relative: object,
    expected_relative: str,
    owner: str,
) -> Path:
    if stored_relative != expected_relative:
        raise ValueError(f"{owner} must be the fixed path {expected_relative!r}")
    if output.is_symlink():
        raise ValueError("v3 preflight output directory must not be a symlink")
    current = output
    for part in Path(expected_relative).parts:
        if part in {"", ".", ".."}:
            raise ValueError(f"{owner} contains an unsafe relative component")
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{owner} must not traverse a symlink")
    if not current.is_file():
        raise ValueError(f"{owner} is missing or is not a regular file")
    try:
        current.resolve(strict=True).relative_to(output.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{owner} escapes the fixed preflight directory") from exc
    return current


def _fixed_preflight_output(config: DetSparseV3Config) -> Path:
    root = config.root_dir.resolve()
    expected = Path(os.path.abspath(os.fspath(root / OUTPUT_PATH / "preflight")))
    configured = Path(os.path.abspath(os.fspath(config.output_dir / "preflight")))
    if configured != expected:
        raise ValueError("v3 preflight output differs from the fixed lexical path")
    try:
        parts = configured.relative_to(root).parts
    except ValueError as exc:
        raise ValueError("v3 preflight output escapes the repository root") from exc
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("v3 preflight output path must not traverse a symlink")
    return configured


def _fingerprint_sha256(query_analyzer: QueryAnalyzer) -> str:
    return _sha256_bytes(
        json.dumps(
            query_analyzer.fingerprint.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _require_bound_query_analyzer(
    config: DetSparseV3Config,
    query_analyzer: QueryAnalyzer,
) -> None:
    if type(query_analyzer) is not RemoteLuceneQueryAnalyzer:
        raise ValueError("v3 preflight requires the exact RemoteLuceneQueryAnalyzer type")
    if query_analyzer.base_url != config.analyzer.url.rstrip("/"):
        raise ValueError("v3 preflight analyzer client is bound to the wrong base URL")


def _fresh_query_analyzer(config: DetSparseV3Config) -> QueryAnalyzer:
    return RemoteLuceneQueryAnalyzer(config.analyzer.url)


def _source_valid(source: Mapping[str, object]) -> bool:
    commit = source.get("commit")
    tree = source.get("tree")
    return (
        source.get("source_tree_clean") is True
        and isinstance(commit, str)
        and len(commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in commit)
        and isinstance(tree, str)
        and len(tree) in {40, 64}
        and all(character in "0123456789abcdef" for character in tree)
        and isinstance(source.get("source_files_sha256"), Mapping)
        and bool(source["source_files_sha256"])
        and isinstance(source.get("required_module_bindings"), Mapping)
        and bool(source["required_module_bindings"])
    )


def _runtime_valid(runtime: Mapping[str, object]) -> bool:
    return (
        isinstance(runtime.get("python"), str)
        and bool(runtime["python"])
        and isinstance(runtime.get("environment_file_sha256"), Mapping)
        and bool(runtime["environment_file_sha256"])
        and isinstance(runtime.get("lucene_analyzer_runtime"), Mapping)
        and bool(runtime["lucene_analyzer_runtime"])
    )


def _attest_source_runtime(
    config: DetSparseV3Config,
) -> tuple[dict[str, object], dict[str, object]]:
    source = dict(current_source_provenance(config.root_dir))
    runtime = dict(current_runtime_provenance(config.root_dir))
    if not _source_valid(source) or not _runtime_valid(runtime):
        raise ValueError("v3 source/runtime attestation is incomplete")
    return source, runtime


def _require_attestation_unchanged(
    config: DetSparseV3Config,
    expected_source: Mapping[str, object],
    expected_runtime: Mapping[str, object],
    stage: str,
) -> None:
    observed_source, observed_runtime = _attest_source_runtime(config)
    try:
        _require_exact_value(
            observed_source,
            dict(expected_source),
            f"{stage} source provenance",
        )
        _require_exact_value(
            observed_runtime,
            dict(expected_runtime),
            f"{stage} runtime provenance",
        )
    except ValueError as exc:
        raise ValueError(f"v3 source/runtime changed {stage}") from exc


def _require_canonical_config(config: DetSparseV3Config) -> DetSparseV3Config:
    canonical = load_det_sparse_v3_config(config.config_path)
    try:
        _require_exact_value(
            asdict(config),
            asdict(canonical),
            "v3 in-memory config",
        )
    except ValueError as exc:
        raise ValueError(
            "v3 config object differs from a fresh canonical reload"
        ) from exc
    bindings = {
        "planner_version": PLANNER_VERSION,
        "renderer_version": RENDERER_VERSION,
        "anchor_selector_version": ANCHOR_SELECTOR_VERSION,
        "selection_version": SELECTION_VERSION,
        "splitter_version": SPLITTER_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "selected_topic_count": PILOT_TOPIC_COUNT,
        "max_facets": MAX_FACETS,
        "parent_min_unique_terms": MIN_PARENT_UNIQUE_TERMS,
    }
    for field, expected in bindings.items():
        if getattr(canonical, field) != expected:
            raise ValueError(f"v3 config/planner binding mismatch: {field}")
    if (
        canonical.candidate_topic_ids != CANDIDATE_TOPIC_IDS
        or canonical.excluded_topic_ids != EXCLUDED_TOPIC_IDS
        or canonical.cost.max_unique_requests_per_topic != 9
        or canonical.cost.max_external_requests != 36
        or canonical.cost.model_calls != 0
        or canonical.cost.reranker_calls != 0
        or canonical.expand_parent_facet is not False
        or canonical.max_expanded_facets != 3
        or canonical.external_gate_status != "blocked"
    ):
        raise ValueError("v3 frozen universe/cost/gate binding mismatch")
    return canonical


def load_candidate_topics(config: DetSparseV3Config) -> tuple[CandidateTopic, ...]:
    """Decode only the exact nine frozen candidate narratives from raw bytes."""

    if config.topics_format != "tsv":
        raise ValueError("det_sparse_v3 supports only the frozen TSV source")
    requested = set(config.candidate_topic_ids)
    if requested.intersection(config.excluded_topic_ids):
        raise ValueError("v3 candidate and excluded topic sets overlap")
    try:
        topic_parts = config.topics_path.relative_to(config.root_dir).parts
    except ValueError as exc:
        raise ValueError("v3 topic source escapes the repository root") from exc
    current = config.root_dir
    for part in topic_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("v3 topic source path must not traverse a symlink")
    if not config.topics_path.is_file():
        raise ValueError("v3 topic source is missing or is not a regular file")
    requested_bytes = {
        topic_id.encode("ascii"): topic_id for topic_id in config.candidate_topic_ids
    }
    found: dict[str, CandidateTopic] = {}
    with config.topics_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            raw_qid, separator, raw_text = raw_line.rstrip(b"\r\n").partition(b"\t")
            # No strip/normalization is permitted on the identifier.  Rows not
            # in scope are skipped before either field is decoded.
            qid = requested_bytes.get(raw_qid)
            if qid is None:
                if raw_qid.strip() in requested_bytes:
                    raise ValueError("candidate topic ID has a whitespace alias")
                continue
            if not separator:
                raise ValueError(f"candidate topic line {line_number} lacks a tab")
            if qid in found:
                raise ValueError(f"candidate topic ID is duplicated: {qid}")
            try:
                narrative = raw_text.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"candidate topic {qid} is not UTF-8") from exc
            if not narrative.strip():
                raise ValueError(f"candidate topic {qid} has an empty narrative")
            found[qid] = CandidateTopic(
                topic=Topic(id=qid, title=derive_title(narrative), narrative=narrative),
                source_line_number=line_number,
                source_line_sha256=_sha256_bytes(raw_line),
            )
    missing = [topic_id for topic_id in config.candidate_topic_ids if topic_id not in found]
    if missing:
        raise ValueError("candidate topic IDs are missing: " + ", ".join(missing))
    return tuple(found[topic_id] for topic_id in config.candidate_topic_ids)


def _inventory_record(projection: ConversationalProjection) -> dict[str, object]:
    if (
        projection.encoding != CONVERSATIONAL_PROJECTION_ENCODING
        or projection.sha256 != CONVERSATIONAL_PROJECTION_SHA256
        or projection.projected_terms_sha256
        != CONVERSATIONAL_PROJECTED_TERMS_SHA256
        or projection.analyzer_fingerprint_sha256 != ANALYZER_FINGERPRINT_SHA256
        or len(projection.records) != CONVERSATIONAL_SURFACE_COUNT
        or tuple(row.source_surface for row in projection.records)
        != CONVERSATIONAL_SURFACES
        or tuple(row.normalized_surface for row in projection.records)
        != tuple(
            # Projection records intentionally preserve source order, unlike
            # the separately sorted normalized inventory.
            unicodedata.normalize("NFKC", surface).casefold()
            for surface in CONVERSATIONAL_SURFACES
        )
        or len(projection.projected_terms) != CONVERSATIONAL_PROJECTED_TERM_COUNT
        or tuple(
            sorted(projection.projected_terms, key=lambda item: item.encode("utf-8"))
        )
        != projection.projected_terms
        or len(set(projection.projected_terms)) != len(projection.projected_terms)
    ):
        raise ValueError("v3 conversational projection object violates frozen shape")
    # Bind the separately sorted normalized inventory as well; this detects a
    # source-order implementation accidentally substituting for that identity.
    if tuple(
        sorted(
            {row.normalized_surface for row in projection.records},
            key=lambda item: item.encode("utf-8"),
        )
    ) != NORMALIZED_CONVERSATIONAL_SURFACES:
        raise ValueError("v3 normalized conversational projection shape drifted")
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "surface_version": CONVERSATIONAL_SURFACE_VERSION,
        "normalization_version": CONVERSATIONAL_NORMALIZATION_VERSION,
        "source_count": CONVERSATIONAL_SURFACE_COUNT,
        "source_sha256": CONVERSATIONAL_SOURCE_SHA256,
        "normalized_sha256": CONVERSATIONAL_NORMALIZED_SHA256,
        "projection_encoding": CONVERSATIONAL_PROJECTION_ENCODING,
        "projection_sha256": projection.sha256,
        "records": [row.to_dict() for row in projection.records],
        "projected_term_count": len(projection.projected_terms),
        "projected_terms": list(projection.projected_terms),
        "projected_terms_sha256": projection.projected_terms_sha256,
        "analyzer_fingerprint_sha256": projection.analyzer_fingerprint_sha256,
        "state": "frozen_before_candidate_read",
    }


def _freeze_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    return {field: metadata[field] for field in _FREEZE_METADATA_FIELDS}


def _completion_record(
    metadata: Mapping[str, object],
    output: Path,
) -> dict[str, object]:
    source = metadata.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("v3 completion receipt requires source provenance")
    selected_topic_ids = metadata.get("selected_topic_ids")
    if not isinstance(selected_topic_ids, list) or any(
        not isinstance(topic_id, str) or not topic_id
        for topic_id in selected_topic_ids
    ):
        raise ValueError("v3 completion receipt requires selected topic IDs")
    mechanical_valid = _strict_bool(
        metadata.get("mechanical_valid"), "completion mechanical_valid"
    )
    return {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "protocol_version": COMPLETION_PROTOCOL_VERSION,
        "experiment_id": metadata.get("experiment_id"),
        "freeze_path": "pre_retrieval_freeze.json",
        "freeze_sha256": _sha256_file(output / "pre_retrieval_freeze.json"),
        "metadata_path": "_preflight.json",
        "metadata_sha256": _sha256_file(output / "_preflight.json"),
        "source_commit": source.get("commit"),
        "source_tree": source.get("tree"),
        "analyzer_fingerprint_sha256": metadata.get(
            "analyzer_fingerprint_sha256"
        ),
        "selected_topic_ids": list(selected_topic_ids),
        "mechanical_valid": mechanical_valid,
        "external_calls": 0,
        "model_calls": 0,
        "reranker_calls": 0,
        "qrels_opened": False,
        "state": "completed_after_postseal_attestation_and_fresh_replay",
    }


def _expected_queries_bytes(
    selected_plans: Sequence[DeterministicSparsePlanV3],
) -> bytes:
    return b"".join(
        _canonical_bytes(asdict(query))
        for plan in selected_plans
        for query in plan.query_variants()
    )


def _query_registry(
    selected_plans: Sequence[DeterministicSparsePlanV3],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for plan in selected_plans:
        variants = plan.query_variants()
        if plan.status != "ok" or len(variants) != 1 + len(plan.facets):
            raise ValueError("selected v3 plan is not a complete original+facet plan")
        signatures = [
            plan.original_bm25_signature,
            *(facet.bm25_signature for facet in plan.facets),
        ]
        coverage = [
            tuple(unit.unit_id for unit in plan.lexical_units),
            *(facet.coverage_unit_ids for facet in plan.facets),
        ]
        criticality = [
            None,
            *(
                "parent"
                if ordinal == 1
                else plan.final_criticality[ordinal - 2].label
                for ordinal in range(1, len(plan.facets) + 1)
            ),
        ]
        for ordinal, (variant, signature, unit_ids, label) in enumerate(
            zip(variants, signatures, coverage, criticality),
            start=1,
        ):
            rows.append(
                {
                    "topic_id": plan.topic_id,
                    "ordinal": ordinal,
                    "query_id": f"{plan.topic_id}:base:{ordinal:02d}",
                    "variant_name": variant.variant_name,
                    "source_type": variant.source_type,
                    "query_text": variant.query_text,
                    "query_sha256": _sha256_bytes(variant.query_text.encode("utf-8")),
                    "bm25_signature": [list(item) for item in signature],
                    "coverage_unit_ids": list(unit_ids),
                    "criticality_label": label,
                    "is_original": ordinal == 1,
                }
            )
        topic_rows = [row for row in rows if row["topic_id"] == plan.topic_id]
        if len({str(row["query_text"]) for row in topic_rows}) != len(topic_rows):
            raise ValueError("v3 query registry contains an exact alias")
        signatures_seen = {
            tuple((str(term), int(count)) for term, count in row["bm25_signature"])
            for row in topic_rows
        }
        if len(signatures_seen) != len(topic_rows):
            raise ValueError("v3 query registry contains a BM25-signature alias")
    return rows


def _coverage_paths(
    selected_plans: Sequence[DeterministicSparsePlanV3],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for plan in selected_plans:
        assignments = [
            unit_id for facet in plan.facets for unit_id in facet.coverage_unit_ids
        ]
        expected = [unit.unit_id for unit in plan.lexical_units]
        if assignments != expected or len(assignments) != len(set(assignments)):
            raise ValueError("v3 coverage ledger must assign every unit exactly once")
        by_unit = {
            unit_id: facet
            for facet in plan.facets
            for unit_id in facet.coverage_unit_ids
        }
        if set(by_unit) != set(expected):
            raise ValueError("v3 coverage ledger does not cover every unit exactly")
        final_by_facet = {row.owner_id: row for row in plan.final_criticality}
        for unit in plan.lexical_units:
            facet = by_unit[unit.unit_id]
            critical = final_by_facet.get(facet.facet_id)
            rows.append(
                {
                    "topic_id": plan.topic_id,
                    "unit_id": unit.unit_id,
                    "facet_id": facet.facet_id,
                    "variant_name": facet.variant_name,
                    "facet_bm25_signature": [list(item) for item in facet.bm25_signature],
                    "original_bm25_signature": [
                        list(item) for item in plan.original_bm25_signature
                    ],
                    "criticality_label": (
                        "parent" if critical is None else critical.label
                    ),
                    "non_original_path": (
                        facet.bm25_signature != plan.original_bm25_signature
                        and facet.query_text != plan.original_query_text
                    ),
                }
            )
    if any(row["non_original_path"] is not True for row in rows):
        raise ValueError("a v3 coverage unit lacks a non-original retrieval path")
    return rows


def _criticality_ledger(
    candidate_topics: Sequence[CandidateTopic],
    outcome: SelectionOutcomeV3,
) -> dict[str, object]:
    if tuple(outcome.plans_by_topic) != CANDIDATE_TOPIC_IDS:
        raise ValueError("v3 plan map differs from the exact candidate order")
    rows: list[dict[str, object]] = []
    for candidate in candidate_topics:
        plan = outcome.plans_by_topic[candidate.topic.id]
        rows.append(
            {
                "topic_id": plan.topic_id,
                "narrative_sha256": plan.narrative_sha256,
                "anchor_core_terms": (
                    [] if plan.anchor is None else list(plan.anchor.core_terms)
                ),
                "raw": [asdict(row) for row in plan.raw_criticality],
                "final": [asdict(row) for row in plan.final_criticality],
            }
        )
    return {
        "schema_version": CRITICALITY_SCHEMA_VERSION,
        "candidate_topic_ids": list(CANDIDATE_TOPIC_IDS),
        "critical_topic_id": outcome.selection.critical_topic_id,
        "rows": rows,
    }


def _request_projections(
    selected_plans: Sequence[DeterministicSparsePlanV3],
    config: DetSparseV3Config,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for plan in selected_plans:
        facet_count = len(plan.facets)
        base = 1 + facet_count
        # E expands O; FE expands child facets but never protected parent f01.
        # Thus M facets admit 1 + min(M-1, 3) expansion bases, equal to M
        # under the frozen 2<=M<=4 contract, and the ceiling is 1+2M<=9.
        expansion = 1 + min(facet_count - 1, config.max_expanded_facets)
        derived = base + expansion
        rows.append(
            {
                "topic_id": plan.topic_id,
                "facet_count": facet_count,
                "base_unique_requests": base,
                "eligible_expansion_bases": expansion,
                "derived_max_unique_requests": derived,
            }
        )
    return rows


def _validate_stored_request_projections(
    stored: object,
    expected: Sequence[Mapping[str, object]],
    *,
    max_expanded_facets: int,
    max_per_topic: int,
    max_global: int,
) -> list[dict[str, object]]:
    if not isinstance(stored, list) or len(stored) != len(expected):
        raise ValueError("v3 request projection count/type mismatch")
    keys = {
        "topic_id",
        "facet_count",
        "base_unique_requests",
        "eligible_expansion_bases",
        "derived_max_unique_requests",
    }
    validated: list[dict[str, object]] = []
    total = 0
    for ordinal, (raw, expected_row) in enumerate(zip(stored, expected), start=1):
        row = _require_exact_keys(raw, keys, f"request projection {ordinal}")
        if not isinstance(row.get("topic_id"), str):
            raise ValueError("v3 request projection topic_id must be text")
        facet_count = _strict_int(row.get("facet_count"), "projection facet_count")
        base = _strict_int(
            row.get("base_unique_requests"), "projection base_unique_requests"
        )
        expansion = _strict_int(
            row.get("eligible_expansion_bases"),
            "projection eligible_expansion_bases",
        )
        derived = _strict_int(
            row.get("derived_max_unique_requests"),
            "projection derived_max_unique_requests",
        )
        if (
            base != 1 + facet_count
            or expansion != 1 + min(facet_count - 1, max_expanded_facets)
            or derived != base + expansion
        ):
            raise ValueError("v3 request projection arithmetic is inconsistent")
        if not 1 <= derived <= max_per_topic:
            raise ValueError("v3 request projection exceeds the per-topic ceiling")
        if row != dict(expected_row):
            raise ValueError("v3 request projection does not replay exactly")
        total += derived
        validated.append(row)
    if total > max_global:
        raise ValueError("v3 request projection exceeds the global ceiling")
    return validated


def _plan_summary(
    plan: DeterministicSparsePlanV3,
    *,
    ordinal: int,
    relative_path: str,
    artifact_sha256: str,
) -> dict[str, object]:
    semantic = semantic_plan_sha256(plan)
    if semantic == artifact_sha256:
        raise ValueError("v3 semantic and artifact plan hashes must be distinct identities")
    return {
        "ordinal": ordinal,
        "topic_id": plan.topic_id,
        "status": plan.status,
        "plan_path": relative_path,
        "plan_semantic_sha256": semantic,
        "plan_artifact_sha256": artifact_sha256,
        "aligned_token_count": len(plan.aligned_tokens),
        "aligned_occurrence_count": len(plan.aligned_occurrences),
        "recurrence_term_count": len(plan.recurrence_audit),
        "candidate_window_count": len(plan.candidate_audit),
        "recurrent_core_count": len(plan.core_audit),
        "raw_criticality_count": len(plan.raw_criticality),
        "final_criticality_count": len(plan.final_criticality),
        "facet_count": len(plan.facets),
    }


def _candidate_table(
    candidate_topics: Sequence[CandidateTopic],
    screens: Sequence[CandidateScreen],
    candidate_plan_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    screen_by_id = {row.topic_id: row for row in screens}
    plan_by_id = {str(row["topic_id"]): row for row in candidate_plan_rows}
    if (
        len(screen_by_id) != len(screens)
        or tuple(screen_by_id) != CANDIDATE_TOPIC_IDS
        or tuple(plan_by_id) != CANDIDATE_TOPIC_IDS
    ):
        raise ValueError("candidate evidence does not cover the exact nine-topic universe")
    rows: list[dict[str, object]] = []
    for candidate in candidate_topics:
        topic_id = candidate.topic.id
        screen = screen_by_id[topic_id]
        plan_row = plan_by_id[topic_id]
        if screen.plan_semantic_sha256 != plan_row["plan_semantic_sha256"]:
            raise ValueError("candidate screen/plan semantic hashes differ")
        rows.append(
            {
                **screen.to_dict(),
                "source_line_number": candidate.source_line_number,
                "source_line_sha256": candidate.source_line_sha256,
                "candidate_plan_path": plan_row["plan_path"],
                "candidate_plan_artifact_sha256": plan_row[
                    "plan_artifact_sha256"
                ],
                "alignment_occurrence_count": plan_row[
                    "aligned_occurrence_count"
                ],
                "recurrence_term_count": plan_row["recurrence_term_count"],
                "candidate_window_count": plan_row["candidate_window_count"],
                "recurrent_core_count": plan_row["recurrent_core_count"],
            }
        )
    return rows


def _selected_critical_witness(
    outcome: SelectionOutcomeV3,
    selected_plans: Sequence[DeterministicSparsePlanV3],
) -> bool:
    selection = outcome.selection
    critical_id = selection.critical_topic_id
    if (
        selection.status != "ok"
        or critical_id is None
        or not selected_plans
        or selected_plans[0].topic_id != critical_id
        or tuple(plan.topic_id for plan in selected_plans)
        != selection.selected_topic_ids
    ):
        return False
    plan = selected_plans[0]
    anchor = getattr(plan, "anchor", None)
    core_terms = getattr(anchor, "core_terms", None)
    if (
        anchor is None
        or not isinstance(core_terms, (tuple, list))
        or len(core_terms) < 2
        or any(not isinstance(term, str) or not term for term in core_terms)
    ):
        return False

    def is_full_set_anchorless(row: object) -> bool:
        full = getattr(row, "full_core_intersection", None)
        eligible = getattr(row, "eligible_core_intersection", None)
        missing = getattr(row, "missing_full_core_terms", None)
        return bool(
            getattr(row, "label", None) == "anchorless"
            and isinstance(full, (tuple, list))
            and not full
            # Eligible intersection is retained as a separately typed audit
            # value, never substituted for the full-analyzer selection gate.
            and isinstance(eligible, (tuple, list))
            and isinstance(missing, (tuple, list))
            and tuple(missing) == tuple(core_terms)
        )

    screen = next(
        (row for row in selection.screens if row.topic_id == critical_id), None
    )
    return bool(
        screen is not None
        and screen.raw_criticality.anchorless >= 1
        and screen.final_criticality.anchorless >= 1
        and any(is_full_set_anchorless(row) for row in plan.raw_criticality)
        and any(is_full_set_anchorless(row) for row in plan.final_criticality)
    )


def _expected_materialization(
    config: DetSparseV3Config,
    candidate_topics: Sequence[CandidateTopic],
    outcome: SelectionOutcomeV3,
) -> dict[str, object]:
    if tuple(row.topic.id for row in candidate_topics) != CANDIDATE_TOPIC_IDS:
        raise ValueError("v3 candidate records differ from the frozen universe")
    if tuple(outcome.plans_by_topic) != CANDIDATE_TOPIC_IDS:
        raise ValueError("v3 selection plans differ from the frozen universe")
    selected_plans = tuple(
        outcome.plans_by_topic[topic_id]
        for topic_id in outcome.selection.selected_topic_ids
    )
    registry = _query_registry(selected_plans) if selected_plans else []
    coverage = _coverage_paths(selected_plans) if selected_plans else []
    projections = _request_projections(selected_plans, config)
    total = sum(int(row["derived_max_unique_requests"]) for row in projections)
    if any(
        type(row["derived_max_unique_requests"]) is not int
        or not 1
        <= int(row["derived_max_unique_requests"])
        <= config.cost.max_unique_requests_per_topic
        for row in projections
    ):
        raise ValueError("v3 request projection exceeds the per-topic ceiling")
    if total > config.cost.max_external_requests:
        raise ValueError("v3 request projection exceeds the global ceiling")
    return {
        "selected_plans": selected_plans,
        "query_registry": registry,
        "coverage_paths": coverage,
        "criticality_ledger": _criticality_ledger(candidate_topics, outcome),
        "request_projections": projections,
        "planned_base_requests": len(registry),
        "derived_max_total_requests": total,
    }


def build_preflight(config: DetSparseV3Config) -> V3PreflightResult:
    """Screen and freeze v3 without qrels, retrieval, models, or rerankers."""

    config = _require_canonical_config(config)
    destination = _fixed_preflight_output(config)
    if os.path.lexists(destination):
        raise FileExistsError(f"v3 preflight output already exists: {destination}")

    source, runtime = _attest_source_runtime(config)
    query_analyzer = _fresh_query_analyzer(config)
    _require_bound_query_analyzer(config, query_analyzer)
    analyzer_sha256 = _fingerprint_sha256(query_analyzer)
    if analyzer_sha256 != config.analyzer.expected_fingerprint_sha256:
        raise ValueError("v3 query analyzer fingerprint differs from frozen config")

    # The projection gate deliberately runs before directory reservation and
    # before the topics file is opened.  A mismatch leaves no ambiguous run.
    projection = build_conversational_projection(query_analyzer)
    inventory = _inventory_record(projection)
    _require_attestation_unchanged(
        config, source, runtime, "before preflight reservation"
    )

    destination.mkdir(parents=True, exist_ok=False)
    reservation_path = destination / "preflight_reservation.json"
    reservation = {
        "schema_version": RESERVATION_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "config_path": str(config.config_path),
        "config_sha256": _sha256_file(config.config_path),
        "candidate_topic_ids": list(config.candidate_topic_ids),
        "excluded_topic_ids": list(config.excluded_topic_ids),
        "analyzer_fingerprint_sha256": analyzer_sha256,
        "inventory_projection_sha256": projection.sha256,
        "inventory_projected_terms_sha256": projection.projected_terms_sha256,
        "inventory_projected_term_count": len(projection.projected_terms),
        "source": source,
        "runtime": runtime,
        "state": "reserved_before_candidate_read",
        "external_calls": 0,
        "model_calls": 0,
        "reranker_calls": 0,
        "qrels_opened": False,
    }
    _create_only(reservation_path, _canonical_bytes(reservation, pretty=True))
    inventory_path = destination / "conversational_inventory.json"
    _create_only(inventory_path, _canonical_bytes(inventory, pretty=True))

    # Both create-only pre-read records must exist before the sole candidate
    # loader is permitted to touch the topic source.
    _require_attestation_unchanged(config, source, runtime, "before candidate read")
    candidate_topics = load_candidate_topics(config)
    outcome = screen_and_select_structural_topics_v3(
        tuple(row.topic for row in candidate_topics),
        query_analyzer=query_analyzer,
        candidate_topic_ids=config.candidate_topic_ids,
        seed=config.selection_seed,
    )
    materialized = _expected_materialization(config, candidate_topics, outcome)
    selected_plans = materialized["selected_plans"]
    assert isinstance(selected_plans, tuple)

    _require_attestation_unchanged(
        config, source, runtime, "between screening and writes"
    )

    artifacts: list[Path] = [reservation_path, inventory_path]
    candidate_plan_rows: list[dict[str, object]] = []
    for ordinal, topic_id in enumerate(CANDIDATE_TOPIC_IDS, start=1):
        plan = outcome.plans_by_topic[topic_id]
        relative = f"candidate_plans/{ordinal:02d}_{topic_id}.json"
        path = destination / relative
        _create_only(path, _canonical_bytes(plan.to_dict(), pretty=True))
        artifacts.append(path)
        candidate_plan_rows.append(
            _plan_summary(
                plan,
                ordinal=ordinal,
                relative_path=relative,
                artifact_sha256=_sha256_file(path),
            )
        )

    selection_path = destination / "selection.json"
    _create_only(
        selection_path,
        _canonical_bytes(outcome.selection.to_dict(), pretty=True),
    )
    artifacts.append(selection_path)

    candidate_table = _candidate_table(
        candidate_topics,
        outcome.selection.screens,
        candidate_plan_rows,
    )
    candidate_screen_path = destination / "candidate_screen.json"
    _create_only(
        candidate_screen_path,
        _canonical_bytes(candidate_table, pretty=True),
    )
    artifacts.append(candidate_screen_path)

    selected_plan_rows: list[dict[str, object]] = []
    for ordinal, plan in enumerate(selected_plans, start=1):
        relative = f"selected_plans/{ordinal:02d}_{plan.topic_id}.json"
        path = destination / relative
        _create_only(path, _canonical_bytes(plan.to_dict(), pretty=True))
        artifacts.append(path)
        selected_plan_rows.append(
            _plan_summary(
                plan,
                ordinal=ordinal,
                relative_path=relative,
                artifact_sha256=_sha256_file(path),
            )
        )

    criticality_path = destination / "criticality_ledger.json"
    _create_only(
        criticality_path,
        _canonical_bytes(materialized["criticality_ledger"], pretty=True),
    )
    artifacts.append(criticality_path)

    queries_path = destination / "queries.jsonl"
    _create_only(queries_path, _expected_queries_bytes(selected_plans))
    artifacts.append(queries_path)
    registry_path = destination / "base_query_registry.json"
    _create_only(
        registry_path,
        _canonical_bytes(materialized["query_registry"], pretty=True),
    )
    artifacts.append(registry_path)
    coverage_path = destination / "coverage_paths.json"
    _create_only(
        coverage_path,
        _canonical_bytes(materialized["coverage_paths"], pretty=True),
    )
    artifacts.append(coverage_path)

    fallback_topic_ids = [
        plan.topic_id for plan in selected_plans if plan.status != "ok"
    ]
    mechanical_valid = bool(
        outcome.selection.status == "ok"
        and len(selected_plans) == PILOT_TOPIC_COUNT
        and len({plan.topic_id for plan in selected_plans}) == PILOT_TOPIC_COUNT
        and not fallback_topic_ids
        and _selected_critical_witness(outcome, selected_plans)
        and _source_valid(source)
        and _runtime_valid(runtime)
        and int(materialized["derived_max_total_requests"])
        <= config.cost.max_external_requests
    )

    metadata = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "config_path": str(config.config_path),
        "config_sha256": _sha256_file(config.config_path),
        "topic_source_path": str(config.topics_path),
        "candidate_topic_ids": list(config.candidate_topic_ids),
        "excluded_topic_ids": list(config.excluded_topic_ids),
        "selected_topic_ids": list(outcome.selection.selected_topic_ids),
        "critical_topic_id": outcome.selection.critical_topic_id,
        "selected_topic_count": config.selected_topic_count,
        "completion_protocol_version": COMPLETION_PROTOCOL_VERSION,
        "completion_path": "preflight_completion.json",
        "completion_required": True,
        "reservation_path": reservation_path.relative_to(destination).as_posix(),
        "reservation_sha256": _sha256_file(reservation_path),
        "inventory_path": inventory_path.relative_to(destination).as_posix(),
        "inventory_artifact_sha256": _sha256_file(inventory_path),
        "inventory_projection_sha256": projection.sha256,
        "inventory_projected_terms_sha256": projection.projected_terms_sha256,
        "inventory_projected_term_count": len(projection.projected_terms),
        "selection_path": selection_path.relative_to(destination).as_posix(),
        "selection_artifact_sha256": _sha256_file(selection_path),
        "candidate_screen_path": candidate_screen_path.relative_to(
            destination
        ).as_posix(),
        "candidate_screen_artifact_sha256": _sha256_file(candidate_screen_path),
        "candidate_plans": candidate_plan_rows,
        "selected_plans": selected_plan_rows,
        "criticality_path": criticality_path.relative_to(destination).as_posix(),
        "criticality_artifact_sha256": _sha256_file(criticality_path),
        "planner_version": PLANNER_VERSION,
        "renderer_version": RENDERER_VERSION,
        "anchor_selector_version": ANCHOR_SELECTOR_VERSION,
        "selection_version": SELECTION_VERSION,
        "splitter_version": SPLITTER_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "candidate_evidence_version": CANDIDATE_EVIDENCE_VERSION,
        "candidate_evidence_encoding": CANDIDATE_EVIDENCE_ENCODING,
        "candidate_evidence_keys": list(CANDIDATE_EVIDENCE_KEYS),
        "conversational_surface_version": CONVERSATIONAL_SURFACE_VERSION,
        "conversational_normalization_version": CONVERSATIONAL_NORMALIZATION_VERSION,
        "conversational_source_count": CONVERSATIONAL_SURFACE_COUNT,
        "conversational_source_sha256": CONVERSATIONAL_SOURCE_SHA256,
        "conversational_normalized_sha256": CONVERSATIONAL_NORMALIZED_SHA256,
        "max_facets": MAX_FACETS,
        "parent_min_unique_terms": MIN_PARENT_UNIQUE_TERMS,
        "required_results": config.retrieval.required_results,
        "max_attempts": config.retrieval.max_attempts,
        "retry_policy": config.retrieval.retry_policy,
        "redirect_policy": config.retrieval.redirect_policy,
        "retrieval_run_namespace": config.retrieval_run_namespace,
        "global_ticket_namespace": config.global_ticket_namespace,
        "analyzer_fingerprint": query_analyzer.fingerprint.to_dict(),
        "analyzer_fingerprint_sha256": analyzer_sha256,
        "arms": list(ARM_NAMES),
        "queries_path": queries_path.relative_to(destination).as_posix(),
        "queries_artifact_sha256": _sha256_file(queries_path),
        "query_registry_path": registry_path.relative_to(destination).as_posix(),
        "query_registry_artifact_sha256": _sha256_file(registry_path),
        "coverage_paths_path": coverage_path.relative_to(destination).as_posix(),
        "coverage_paths_artifact_sha256": _sha256_file(coverage_path),
        "planned_base_unique_requests": materialized["planned_base_requests"],
        "request_projections": materialized["request_projections"],
        "derived_max_total_unique_requests": materialized[
            "derived_max_total_requests"
        ],
        "hard_external_request_ceiling": config.cost.max_external_requests,
        "hard_per_topic_request_ceiling": config.cost.max_unique_requests_per_topic,
        "external_gate_status": config.external_gate_status,
        "external_gate_reason": config.external_gate_reason,
        "mechanical_valid": mechanical_valid,
        "fallback_topic_ids": fallback_topic_ids,
        "external_calls": 0,
        "model_calls": 0,
        "reranker_calls": 0,
        "qrels_opened": False,
        "source": source,
        "runtime": runtime,
    }
    metadata_path = destination / "_preflight.json"
    _create_only(metadata_path, _canonical_bytes(metadata, pretty=True))
    artifacts.append(metadata_path)
    create_freeze_manifest(
        destination / "pre_retrieval_freeze.json",
        artifact_root=destination,
        artifacts=artifacts,
        qrels_path=config.evaluation.qrels,
        metadata=_freeze_metadata(metadata),
    )
    _require_attestation_unchanged(
        config, source, runtime, "after preflight writes"
    )
    _validate_preflight_replay(
        config,
        destination,
        require_mechanical_valid=mechanical_valid,
        require_completion_receipt=False,
        forbidden_analyzer=query_analyzer,
    )
    completion_path = destination / "preflight_completion.json"
    _create_only(
        completion_path,
        _canonical_bytes(_completion_record(metadata, destination), pretty=True),
    )
    # The receipt is deliberately the terminal operation.  No external
    # attestation or other fallible protocol step follows it.
    return V3PreflightResult(
        output_dir=destination,
        selected_topic_ids=outcome.selection.selected_topic_ids,
        selected_plans=selected_plans,
        valid=mechanical_valid,
        planned_base_requests=int(materialized["planned_base_requests"]),
        derived_max_total_requests=int(materialized["derived_max_total_requests"]),
        hard_external_request_ceiling=config.cost.max_external_requests,
    )


_PLAN_ROW_KEYS = {
    "ordinal",
    "topic_id",
    "status",
    "plan_path",
    "plan_semantic_sha256",
    "plan_artifact_sha256",
    "aligned_token_count",
    "aligned_occurrence_count",
    "recurrence_term_count",
    "candidate_window_count",
    "recurrent_core_count",
    "raw_criticality_count",
    "final_criticality_count",
    "facet_count",
}


def _validate_preflight_replay(
    config: DetSparseV3Config,
    output_dir: Path,
    *,
    require_mechanical_valid: bool,
    require_completion_receipt: bool,
    forbidden_analyzer: QueryAnalyzer | None = None,
) -> dict[str, object]:
    """Replay every v3 semantic object with a fresh exact local analyzer."""

    config = _require_canonical_config(config)
    query_analyzer = _fresh_query_analyzer(config)
    if query_analyzer is forbidden_analyzer:
        raise ValueError("v3 replay requires a fresh analyzer client instance")
    _require_bound_query_analyzer(config, query_analyzer)
    expected_source, expected_runtime = _attest_source_runtime(config)

    output = Path(os.path.abspath(os.fspath(output_dir)))
    expected_output = _fixed_preflight_output(config)
    if output != expected_output:
        raise ValueError("v3 preflight path differs from the fixed experiment path")
    if output.is_symlink() or not output.is_dir():
        raise ValueError("v3 preflight output must be a real fixed directory")
    for artifact in output.rglob("*"):
        if artifact.is_symlink():
            raise ValueError("v3 preflight artifacts and directories must not be symlinks")
        if not artifact.is_file() and not artifact.is_dir():
            raise ValueError(
                "v3 preflight entries must be regular files or directories"
            )

    freeze_path = _fixed_artifact_path(
        output,
        "pre_retrieval_freeze.json",
        "pre_retrieval_freeze.json",
        "pre-retrieval freeze",
    )
    metadata_path = _fixed_artifact_path(
        output,
        "_preflight.json",
        "_preflight.json",
        "preflight metadata",
    )
    manifest = verify_freeze_manifest(
        freeze_path,
        artifact_root=output,
        expected_qrels_path=config.evaluation.qrels,
    )
    strict_manifest = _require_exact_keys(
        _load_json(freeze_path),
        {
            "schema_version",
            "artifact_root",
            "artifacts",
            "qrels_path",
            "qrels_opened_before_freeze",
            "metadata",
        },
        "v3 freeze manifest",
    )
    _require_exact_value(strict_manifest, manifest, "v3 freeze manifest")
    if freeze_path.read_bytes() != _canonical_bytes(strict_manifest):
        raise ValueError("v3 freeze manifest bytes are not canonical")
    metadata = _require_exact_keys(
        _load_json(metadata_path),
        _METADATA_KEYS,
        "v3 preflight metadata",
    )
    if metadata_path.read_bytes() != _canonical_bytes(metadata, pretty=True):
        raise ValueError("v3 preflight metadata bytes are not canonical")
    if metadata.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("v3 preflight metadata schema mismatch")
    if (
        metadata.get("completion_protocol_version")
        != COMPLETION_PROTOCOL_VERSION
        or metadata.get("completion_path") != "preflight_completion.json"
        or _strict_bool(
            metadata.get("completion_required"), "completion_required"
        )
        is not True
    ):
        raise ValueError("v3 terminal completion protocol identity mismatch")
    if (
        metadata.get("experiment_id") != config.experiment_id
        or metadata.get("config_path") != str(config.config_path)
        or metadata.get("topic_source_path") != str(config.topics_path)
    ):
        raise ValueError("v3 preflight config/experiment identity mismatch")
    if _require_sha256(metadata.get("config_sha256"), "config_sha256") != _sha256_file(
        config.config_path
    ):
        raise ValueError("v3 preflight config hash mismatch")
    if _strict_string_list(
        metadata.get("candidate_topic_ids"), "candidate_topic_ids"
    ) != list(CANDIDATE_TOPIC_IDS):
        raise ValueError("v3 candidate topic IDs differ from frozen config")
    if _strict_string_list(
        metadata.get("excluded_topic_ids"), "excluded_topic_ids"
    ) != list(EXCLUDED_TOPIC_IDS):
        raise ValueError("v3 excluded topic IDs differ from frozen config")
    if (
        metadata.get("planner_version") != PLANNER_VERSION
        or metadata.get("renderer_version") != RENDERER_VERSION
        or metadata.get("anchor_selector_version") != ANCHOR_SELECTOR_VERSION
        or metadata.get("selection_version") != SELECTION_VERSION
        or metadata.get("splitter_version") != SPLITTER_VERSION
        or metadata.get("tokenizer_version") != TOKENIZER_VERSION
        or metadata.get("candidate_evidence_version")
        != CANDIDATE_EVIDENCE_VERSION
        or metadata.get("candidate_evidence_encoding")
        != CANDIDATE_EVIDENCE_ENCODING
        or metadata.get("candidate_evidence_keys")
        != list(CANDIDATE_EVIDENCE_KEYS)
        or metadata.get("arms") != list(ARM_NAMES)
    ):
        raise ValueError("v3 planner/renderer/evidence/arm identity mismatch")
    if (
        metadata.get("conversational_surface_version")
        != CONVERSATIONAL_SURFACE_VERSION
        or metadata.get("conversational_normalization_version")
        != CONVERSATIONAL_NORMALIZATION_VERSION
        or _strict_int(
            metadata.get("conversational_source_count"),
            "conversational_source_count",
        )
        != CONVERSATIONAL_SURFACE_COUNT
        or _require_sha256(
            metadata.get("conversational_source_sha256"),
            "conversational_source_sha256",
        )
        != CONVERSATIONAL_SOURCE_SHA256
        or _require_sha256(
            metadata.get("conversational_normalized_sha256"),
            "conversational_normalized_sha256",
        )
        != CONVERSATIONAL_NORMALIZED_SHA256
    ):
        raise ValueError("v3 conversational source inventory identity mismatch")
    if (
        _strict_int(metadata.get("selected_topic_count"), "selected_topic_count")
        != PILOT_TOPIC_COUNT
        or _strict_int(metadata.get("max_facets"), "max_facets") != MAX_FACETS
        or _strict_int(
            metadata.get("parent_min_unique_terms"), "parent_min_unique_terms"
        )
        != MIN_PARENT_UNIQUE_TERMS
        or _strict_int(metadata.get("required_results"), "required_results")
        != config.retrieval.required_results
        or _strict_int(metadata.get("max_attempts"), "max_attempts")
        != config.retrieval.max_attempts
        or metadata.get("retry_policy") != config.retrieval.retry_policy
        or metadata.get("redirect_policy") != config.retrieval.redirect_policy
        or metadata.get("retrieval_run_namespace")
        != config.retrieval_run_namespace
        or metadata.get("global_ticket_namespace")
        != config.global_ticket_namespace
    ):
        raise ValueError("v3 frozen planning/retrieval contract mismatch")
    if (
        _strict_int(
            metadata.get("hard_external_request_ceiling"),
            "hard_external_request_ceiling",
        )
        != 36
        or _strict_int(
            metadata.get("hard_per_topic_request_ceiling"),
            "hard_per_topic_request_ceiling",
        )
        != 9
        or metadata.get("external_gate_status") != "blocked"
        or metadata.get("external_gate_reason") != config.external_gate_reason
    ):
        raise ValueError("v3 external cost/gate contract mismatch")
    _require_exact_value(metadata.get("source"), expected_source, "source provenance")
    _require_exact_value(
        metadata.get("runtime"), expected_runtime, "runtime provenance"
    )
    if not _source_valid(expected_source) or not _runtime_valid(expected_runtime):
        raise ValueError("v3 preflight source/runtime provenance is incomplete")

    live_fingerprint = query_analyzer.fingerprint.to_dict()
    live_fingerprint_sha256 = _fingerprint_sha256(query_analyzer)
    if live_fingerprint_sha256 != ANALYZER_FINGERPRINT_SHA256:
        raise ValueError("v3 validation analyzer differs from frozen config")
    _require_exact_value(
        metadata.get("analyzer_fingerprint"),
        _json_normalize(live_fingerprint),
        "analyzer fingerprint",
    )
    if _require_sha256(
        metadata.get("analyzer_fingerprint_sha256"),
        "analyzer_fingerprint_sha256",
    ) != live_fingerprint_sha256:
        raise ValueError("v3 preflight analyzer identity/hash mismatch")

    completion_lexical = output / "preflight_completion.json"
    if require_completion_receipt:
        completion_path = _fixed_artifact_path(
            output,
            metadata.get("completion_path"),
            "preflight_completion.json",
            "completion_path",
        )
        expected_completion = _completion_record(metadata, output)
        _require_exact_value(
            _load_json(completion_path),
            _json_normalize(expected_completion),
            "v3 terminal completion receipt",
        )
        if completion_path.read_bytes() != _canonical_bytes(
            expected_completion, pretty=True
        ):
            raise ValueError("v3 terminal completion receipt bytes are not canonical")
    elif os.path.lexists(completion_lexical):
        raise ValueError("v3 completion receipt must be absent during internal replay")

    # Verify and recompute the projection before the candidate source is read.
    projection = build_conversational_projection(query_analyzer)
    expected_inventory = _inventory_record(projection)
    inventory_path = _fixed_artifact_path(
        output,
        metadata.get("inventory_path"),
        "conversational_inventory.json",
        "inventory_path",
    )
    stored_inventory = _load_json(inventory_path)
    _require_exact_value(
        stored_inventory,
        _json_normalize(expected_inventory),
        "v3 conversational inventory",
    )
    if inventory_path.read_bytes() != _canonical_bytes(expected_inventory, pretty=True):
        raise ValueError("v3 conversational inventory bytes do not replay exactly")
    if (
        _require_sha256(
            metadata.get("inventory_artifact_sha256"),
            "inventory_artifact_sha256",
        )
        != _sha256_file(inventory_path)
        or _require_sha256(
            metadata.get("inventory_projection_sha256"),
            "inventory_projection_sha256",
        )
        != CONVERSATIONAL_PROJECTION_SHA256
        or _require_sha256(
            metadata.get("inventory_projected_terms_sha256"),
            "inventory_projected_terms_sha256",
        )
        != CONVERSATIONAL_PROJECTED_TERMS_SHA256
        or _strict_int(
            metadata.get("inventory_projected_term_count"),
            "inventory_projected_term_count",
        )
        != CONVERSATIONAL_PROJECTED_TERM_COUNT
    ):
        raise ValueError("v3 conversational inventory identity/hash mismatch")

    reservation_path = _fixed_artifact_path(
        output,
        metadata.get("reservation_path"),
        "preflight_reservation.json",
        "reservation_path",
    )
    expected_reservation = {
        "schema_version": RESERVATION_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "config_path": str(config.config_path),
        "config_sha256": _sha256_file(config.config_path),
        "candidate_topic_ids": list(config.candidate_topic_ids),
        "excluded_topic_ids": list(config.excluded_topic_ids),
        "analyzer_fingerprint_sha256": live_fingerprint_sha256,
        "inventory_projection_sha256": projection.sha256,
        "inventory_projected_terms_sha256": projection.projected_terms_sha256,
        "inventory_projected_term_count": len(projection.projected_terms),
        "source": expected_source,
        "runtime": expected_runtime,
        "state": "reserved_before_candidate_read",
        "external_calls": 0,
        "model_calls": 0,
        "reranker_calls": 0,
        "qrels_opened": False,
    }
    if reservation_path.read_bytes() != _canonical_bytes(
        expected_reservation, pretty=True
    ):
        raise ValueError("v3 preflight reservation does not replay exactly")
    if _require_sha256(
        metadata.get("reservation_sha256"), "reservation_sha256"
    ) != _sha256_file(reservation_path):
        raise ValueError("v3 preflight reservation hash mismatch")

    candidate_topics = load_candidate_topics(config)
    outcome = screen_and_select_structural_topics_v3(
        tuple(row.topic for row in candidate_topics),
        query_analyzer=query_analyzer,
        candidate_topic_ids=config.candidate_topic_ids,
        seed=config.selection_seed,
    )
    materialized = _expected_materialization(config, candidate_topics, outcome)

    raw_candidate_rows = metadata.get("candidate_plans")
    if not isinstance(raw_candidate_rows, list) or len(raw_candidate_rows) != len(
        CANDIDATE_TOPIC_IDS
    ):
        raise ValueError("v3 candidate plan count mismatch")
    replay_candidate_rows: list[dict[str, object]] = []
    candidate_plan_paths: list[Path] = []
    for ordinal, (topic_id, raw_row) in enumerate(
        zip(CANDIDATE_TOPIC_IDS, raw_candidate_rows), start=1
    ):
        row = _require_exact_keys(
            raw_row, _PLAN_ROW_KEYS, f"candidate plan row {ordinal}"
        )
        if (
            _strict_int(row.get("ordinal"), "candidate plan ordinal") != ordinal
            or row.get("topic_id") != topic_id
        ):
            raise ValueError("v3 candidate plan order/identity mismatch")
        relative = f"candidate_plans/{ordinal:02d}_{topic_id}.json"
        path = _fixed_artifact_path(
            output,
            row.get("plan_path"),
            relative,
            f"candidate plan path {ordinal}",
        )
        candidate_plan_paths.append(path)
        plan = outcome.plans_by_topic[topic_id]
        _require_exact_value(
            _load_json(path),
            _json_normalize(plan.to_dict()),
            f"candidate plan {topic_id}",
        )
        if path.read_bytes() != _canonical_bytes(plan.to_dict(), pretty=True):
            raise ValueError(f"v3 candidate plan {topic_id} bytes are not canonical")
        expected_row = _plan_summary(
            plan,
            ordinal=ordinal,
            relative_path=relative,
            artifact_sha256=_sha256_file(path),
        )
        _require_exact_value(row, expected_row, f"candidate plan row {ordinal}")
        replay_candidate_rows.append(expected_row)

    selection_path = _fixed_artifact_path(
        output,
        metadata.get("selection_path"),
        "selection.json",
        "selection_path",
    )
    _require_exact_value(
        _load_json(selection_path),
        _json_normalize(outcome.selection.to_dict()),
        "v3 structural selection",
    )
    if selection_path.read_bytes() != _canonical_bytes(
        outcome.selection.to_dict(), pretty=True
    ):
        raise ValueError("v3 selection bytes are not canonical")
    if _require_sha256(
        metadata.get("selection_artifact_sha256"),
        "selection_artifact_sha256",
    ) != _sha256_file(selection_path):
        raise ValueError("v3 selection artifact hash mismatch")

    candidate_table = _candidate_table(
        candidate_topics,
        outcome.selection.screens,
        replay_candidate_rows,
    )
    candidate_screen_path = _fixed_artifact_path(
        output,
        metadata.get("candidate_screen_path"),
        "candidate_screen.json",
        "candidate_screen_path",
    )
    _require_exact_value(
        _load_json(candidate_screen_path),
        _json_normalize(candidate_table),
        "v3 candidate structural table",
    )
    if candidate_screen_path.read_bytes() != _canonical_bytes(
        candidate_table, pretty=True
    ):
        raise ValueError("v3 candidate screen bytes are not canonical")
    if _require_sha256(
        metadata.get("candidate_screen_artifact_sha256"),
        "candidate_screen_artifact_sha256",
    ) != _sha256_file(candidate_screen_path):
        raise ValueError("v3 candidate screen artifact hash mismatch")

    if _strict_string_list(
        metadata.get("selected_topic_ids"), "selected_topic_ids"
    ) != list(outcome.selection.selected_topic_ids):
        raise ValueError("v3 selected topic IDs differ from frozen selection")
    critical_topic_id = metadata.get("critical_topic_id")
    if critical_topic_id is not None and not isinstance(critical_topic_id, str):
        raise ValueError("v3 critical_topic_id must be text or null")
    if critical_topic_id != outcome.selection.critical_topic_id:
        raise ValueError("v3 critical topic differs from frozen selection")

    selected_plans = materialized["selected_plans"]
    assert isinstance(selected_plans, tuple)
    raw_selected_rows = metadata.get("selected_plans")
    if not isinstance(raw_selected_rows, list) or len(raw_selected_rows) != len(
        selected_plans
    ):
        raise ValueError("v3 selected plan count mismatch")
    selected_plan_paths: list[Path] = []
    for ordinal, (plan, raw_row) in enumerate(
        zip(selected_plans, raw_selected_rows), start=1
    ):
        row = _require_exact_keys(
            raw_row, _PLAN_ROW_KEYS, f"selected plan row {ordinal}"
        )
        relative = f"selected_plans/{ordinal:02d}_{plan.topic_id}.json"
        path = _fixed_artifact_path(
            output,
            row.get("plan_path"),
            relative,
            f"selected plan path {ordinal}",
        )
        selected_plan_paths.append(path)
        _require_exact_value(
            _load_json(path),
            _json_normalize(plan.to_dict()),
            f"selected plan {plan.topic_id}",
        )
        if path.read_bytes() != _canonical_bytes(plan.to_dict(), pretty=True):
            raise ValueError(
                f"v3 selected plan {plan.topic_id} bytes are not canonical"
            )
        expected_row = _plan_summary(
            plan,
            ordinal=ordinal,
            relative_path=relative,
            artifact_sha256=_sha256_file(path),
        )
        _require_exact_value(row, expected_row, f"selected plan row {ordinal}")

    criticality_path = _fixed_artifact_path(
        output,
        metadata.get("criticality_path"),
        "criticality_ledger.json",
        "criticality_path",
    )
    _require_exact_value(
        _load_json(criticality_path),
        _json_normalize(materialized["criticality_ledger"]),
        "v3 criticality ledger",
    )
    if criticality_path.read_bytes() != _canonical_bytes(
        materialized["criticality_ledger"], pretty=True
    ):
        raise ValueError("v3 criticality ledger bytes are not canonical")
    if _require_sha256(
        metadata.get("criticality_artifact_sha256"),
        "criticality_artifact_sha256",
    ) != _sha256_file(criticality_path):
        raise ValueError("v3 criticality artifact hash mismatch")

    queries_path = _fixed_artifact_path(
        output,
        metadata.get("queries_path"),
        "queries.jsonl",
        "queries_path",
    )
    if queries_path.read_bytes() != _expected_queries_bytes(selected_plans):
        raise ValueError("v3 queries.jsonl does not replay exactly")
    if _require_sha256(
        metadata.get("queries_artifact_sha256"),
        "queries_artifact_sha256",
    ) != _sha256_file(queries_path):
        raise ValueError("v3 queries artifact hash mismatch")
    registry_path = _fixed_artifact_path(
        output,
        metadata.get("query_registry_path"),
        "base_query_registry.json",
        "query_registry_path",
    )
    coverage_path = _fixed_artifact_path(
        output,
        metadata.get("coverage_paths_path"),
        "coverage_paths.json",
        "coverage_paths_path",
    )
    _require_exact_value(
        _load_json(registry_path),
        _json_normalize(materialized["query_registry"]),
        "v3 query registry",
    )
    _require_exact_value(
        _load_json(coverage_path),
        _json_normalize(materialized["coverage_paths"]),
        "v3 coverage ledger",
    )
    if registry_path.read_bytes() != _canonical_bytes(
        materialized["query_registry"], pretty=True
    ):
        raise ValueError("v3 query registry bytes are not canonical")
    if coverage_path.read_bytes() != _canonical_bytes(
        materialized["coverage_paths"], pretty=True
    ):
        raise ValueError("v3 coverage ledger bytes are not canonical")
    if _require_sha256(
        metadata.get("query_registry_artifact_sha256"),
        "query_registry_artifact_sha256",
    ) != _sha256_file(registry_path):
        raise ValueError("v3 query registry artifact hash mismatch")
    if _require_sha256(
        metadata.get("coverage_paths_artifact_sha256"),
        "coverage_paths_artifact_sha256",
    ) != _sha256_file(coverage_path):
        raise ValueError("v3 coverage artifact hash mismatch")

    projections = _validate_stored_request_projections(
        metadata.get("request_projections"),
        materialized["request_projections"],
        max_expanded_facets=config.max_expanded_facets,
        max_per_topic=9,
        max_global=36,
    )
    planned_base = _strict_int(
        metadata.get("planned_base_unique_requests"),
        "planned_base_unique_requests",
    )
    derived_total = _strict_int(
        metadata.get("derived_max_total_unique_requests"),
        "derived_max_total_unique_requests",
    )
    if (
        planned_base != materialized["planned_base_requests"]
        or planned_base != len(materialized["query_registry"])
        or derived_total != materialized["derived_max_total_requests"]
        or derived_total
        != sum(int(row["derived_max_unique_requests"]) for row in projections)
    ):
        raise ValueError("v3 request projection does not replay exactly")

    fallback_topic_ids = [
        plan.topic_id for plan in selected_plans if plan.status != "ok"
    ]
    expected_mechanical_valid = bool(
        outcome.selection.status == "ok"
        and len(selected_plans) == PILOT_TOPIC_COUNT
        and len({plan.topic_id for plan in selected_plans}) == PILOT_TOPIC_COUNT
        and not fallback_topic_ids
        and _selected_critical_witness(outcome, selected_plans)
        and _source_valid(expected_source)
        and _runtime_valid(expected_runtime)
        and derived_total <= 36
    )
    if _strict_bool(metadata.get("mechanical_valid"), "mechanical_valid") is not (
        expected_mechanical_valid
    ):
        raise ValueError("v3 mechanical validity does not replay exactly")
    if require_mechanical_valid and not expected_mechanical_valid:
        raise ValueError("v3 preflight is mechanically invalid")
    if _strict_string_list(
        metadata.get("fallback_topic_ids"), "fallback_topic_ids"
    ) != fallback_topic_ids:
        raise ValueError("v3 fallback topics do not replay exactly")
    if expected_mechanical_valid and not _selected_critical_witness(
        outcome, selected_plans
    ):
        raise ValueError("v3 selected critical topic lacks its raw/final witness")
    for field in ("external_calls", "model_calls", "reranker_calls"):
        if _strict_int(metadata.get(field), field) != 0:
            raise ValueError(f"v3 preflight {field} must be zero")
    if _strict_bool(metadata.get("qrels_opened"), "qrels_opened") is not False:
        raise ValueError("v3 preflight violated the qrels firewall")

    expected_artifacts = {
        "preflight_reservation.json",
        "conversational_inventory.json",
        "selection.json",
        "candidate_screen.json",
        "criticality_ledger.json",
        "queries.jsonl",
        "base_query_registry.json",
        "coverage_paths.json",
        "_preflight.json",
        *(path.relative_to(output).as_posix() for path in candidate_plan_paths),
        *(path.relative_to(output).as_posix() for path in selected_plan_paths),
    }
    raw_manifest_artifacts = strict_manifest.get("artifacts")
    if not isinstance(raw_manifest_artifacts, list):
        raise ValueError("v3 freeze artifact table is missing")
    manifest_artifacts: set[str] = set()
    for ordinal, raw_row in enumerate(raw_manifest_artifacts, start=1):
        row = _require_exact_keys(
            raw_row,
            {"path", "size", "sha256"},
            f"freeze artifact row {ordinal}",
        )
        relative = row.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("v3 freeze artifact path must be text")
        if relative in manifest_artifacts:
            raise ValueError("v3 freeze contains duplicate artifact paths")
        manifest_artifacts.add(relative)
        size = _strict_int(row.get("size"), "freeze artifact size")
        if size < 0:
            raise ValueError("v3 freeze artifact size must be nonnegative")
        _require_sha256(row.get("sha256"), "freeze artifact sha256")
    if manifest_artifacts != expected_artifacts:
        raise ValueError("v3 freeze artifact inventory differs from fixed preflight files")
    observed_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    expected_observed_files = expected_artifacts | {"pre_retrieval_freeze.json"}
    if require_completion_receipt:
        expected_observed_files.add("preflight_completion.json")
    if observed_files != expected_observed_files:
        raise ValueError("v3 preflight directory has unsealed or missing files")
    expected_directories = {"candidate_plans"}
    if selected_plan_paths:
        expected_directories.add("selected_plans")
    observed_directories = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_dir()
    }
    if observed_directories != expected_directories:
        raise ValueError("v3 preflight directory has unexpected or missing directories")
    freeze_metadata = strict_manifest.get("metadata")
    _require_exact_keys(
        freeze_metadata,
        set(_FREEZE_METADATA_FIELDS),
        "v3 freeze metadata",
    )
    _require_exact_value(
        freeze_metadata,
        _freeze_metadata(metadata),
        "v3 freeze metadata mirror",
    )
    _require_attestation_unchanged(
        config,
        expected_source,
        expected_runtime,
        "during validation replay",
    )
    return manifest


def validate_preflight(
    config: DetSparseV3Config,
    output_dir: Path,
) -> dict[str, object]:
    """Validate one successful canonical preflight as an external-call gate."""

    return _validate_preflight_replay(
        config,
        output_dir,
        require_mechanical_valid=True,
        require_completion_receipt=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qrels-blind recurrent-anchor preflight for det_sparse_v3",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_det_sparse_v3_config(args.config)
    output = config.output_dir / "preflight"
    if args.validate_only:
        validate_preflight(config, output)
        print(f"Validated frozen v3 preflight under {output.resolve()}")
        return 0
    result = build_preflight(config)
    print(f"Wrote frozen v3 preflight under {result.output_dir}")
    print(
        "Selected topics: "
        + (", ".join(result.selected_topic_ids) if result.selected_topic_ids else "none")
    )
    print(f"Mechanical valid: {str(result.valid).lower()}")
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
