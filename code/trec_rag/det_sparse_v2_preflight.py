"""Create-only qrels-blind structural selection and preflight for v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from trec_rag.det_sparse_freeze import create_freeze_manifest, verify_freeze_manifest
from trec_rag.det_sparse_v2_config import (
    ANALYZER_FINGERPRINT_SHA256,
    ARM_NAMES,
    CANDIDATE_TOPIC_IDS,
    EXCLUDED_TOPIC_IDS,
    OUTPUT_PATH,
    PILOT_TOPIC_COUNT,
    DetSparseV2Config,
    load_det_sparse_v2_config,
)
from trec_rag.det_sparse_v2_provenance import (
    current_runtime_provenance,
    current_source_provenance,
)
from trec_rag.det_sparse_v2_selection import (
    STRATUM_ORDER,
    CandidateScreen,
    SelectionOutcome,
    screen_and_select_structural_topics,
    semantic_plan_sha256,
)
from trec_rag.deterministic_sparse_v2 import (
    CONTEXT_ALLOCATION,
    MAX_CONTEXT_UNIQUE_TERMS,
    MAX_FACETS,
    MIN_CONTEXT_UNIQUE_TERMS,
    MIN_FACET_UNIQUE_TERMS,
    MIN_PARENT_UNIQUE_TERMS,
    PLANNER_VERSION,
    RENDERER_VERSION,
    SELECTION_VERSION,
    SPLITTER_VERSION,
    TOKENIZER_VERSION,
    DeterministicSparsePlanV2,
)
from trec_rag.query_analyzer import QueryAnalyzer, RemoteLuceneQueryAnalyzer
from trec_rag.topics import Topic, derive_title


PREFLIGHT_SCHEMA_VERSION = "det_sparse_v2_preflight_v1"
RESERVATION_SCHEMA_VERSION = "det_sparse_v2_preflight_reservation_v1"
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
    "selection_strata",
    "selected_topic_count",
    "reservation_path",
    "reservation_sha256",
    "selection_path",
    "selection_sha256",
    "candidate_screen_path",
    "candidate_screen_sha256",
    "plans",
    "planner_version",
    "renderer_version",
    "selection_version",
    "splitter_version",
    "tokenizer_version",
    "parent_min_unique_terms",
    "bounded_context_min_unique_terms",
    "bounded_context_max_unique_terms",
    "bounded_context_allocation",
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
    "queries_sha256",
    "query_registry_path",
    "query_registry_sha256",
    "coverage_paths_path",
    "coverage_paths_sha256",
    "planned_base_unique_requests",
    "request_projections",
    "derived_max_total_unique_requests",
    "hard_external_request_ceiling",
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
    "planner_version",
    "renderer_version",
    "selection_version",
    "splitter_version",
    "tokenizer_version",
    "selection_strata",
    "selected_topic_count",
    "parent_min_unique_terms",
    "bounded_context_min_unique_terms",
    "bounded_context_max_unique_terms",
    "bounded_context_allocation",
    "required_results",
    "max_attempts",
    "retry_policy",
    "redirect_policy",
    "retrieval_run_namespace",
    "global_ticket_namespace",
    "analyzer_fingerprint_sha256",
    "mechanical_valid",
    "fallback_topic_ids",
    "planned_base_unique_requests",
    "request_projections",
    "derived_max_total_unique_requests",
    "hard_external_request_ceiling",
    "external_gate_status",
    "external_gate_reason",
    "external_calls",
    "model_calls",
    "reranker_calls",
)


@dataclass(frozen=True)
class CandidateTopic:
    topic: Topic
    source_line_number: int
    source_line_sha256: str


@dataclass(frozen=True)
class V2PreflightResult:
    output_dir: Path
    selected_topic_ids: tuple[str, ...]
    selected_plans: tuple[DeterministicSparsePlanV2, ...]
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
        )
        + "\n"
    ).encode("utf-8")


def _json_normalize(value: object) -> object:
    return json.loads(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def _require_exact_keys(
    value: object,
    expected: set[str],
    owner: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{owner} keys differ from the frozen schema")
    return value


def _strict_int(value: object, owner: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{owner} must be an integer, not bool/float coercion")
    return value


def _strict_string_list(value: object, owner: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{owner} must be a string list")
    return value


def _require_sha256(value: object, owner: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{owner} must be a lowercase SHA-256")
    return value


def _fixed_artifact_path(
    output: Path,
    stored_relative: object,
    expected_relative: str,
    owner: str,
) -> Path:
    """Resolve one fixed artifact without accepting aliases or symlinks."""

    if stored_relative != expected_relative:
        raise ValueError(f"{owner} must be the fixed path {expected_relative!r}")
    if output.is_symlink():
        raise ValueError("v2 preflight output directory must not be a symlink")
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


def _fixed_preflight_output(config: DetSparseV2Config) -> Path:
    """Return the lexical fixed output path while rejecting symlink parents."""

    root = config.root_dir.resolve()
    expected = Path(os.path.abspath(os.fspath(root / OUTPUT_PATH / "preflight")))
    configured = Path(
        os.path.abspath(os.fspath(config.output_dir / "preflight"))
    )
    if configured != expected:
        raise ValueError("v2 preflight output differs from the fixed lexical path")
    try:
        relative_parts = configured.relative_to(root).parts
    except ValueError as exc:
        raise ValueError("v2 preflight output escapes the repository root") from exc
    current = root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("v2 preflight output path must not traverse a symlink")
    return configured


def _expected_queries_bytes(
    selected_plans: Sequence[DeterministicSparsePlanV2],
) -> bytes:
    return b"".join(
        _canonical_bytes(asdict(query))
        for plan in selected_plans
        for query in plan.query_variants()
    )


def _freeze_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    return {field: metadata[field] for field in _FREEZE_METADATA_FIELDS}


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


def _fingerprint_sha256(query_analyzer: QueryAnalyzer) -> str:
    return hashlib.sha256(
        json.dumps(
            query_analyzer.fingerprint.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _require_bound_query_analyzer(
    config: DetSparseV2Config,
    query_analyzer: QueryAnalyzer,
) -> None:
    """Keep the public preflight on the exact local reference client."""

    if type(query_analyzer) is not RemoteLuceneQueryAnalyzer:
        raise ValueError("v2 preflight requires the exact RemoteLuceneQueryAnalyzer type")
    if query_analyzer.base_url != config.analyzer.url.rstrip("/"):
        raise ValueError("v2 preflight analyzer client is bound to the wrong base URL")


def _fresh_query_analyzer(config: DetSparseV2Config) -> QueryAnalyzer:
    return RemoteLuceneQueryAnalyzer(config.analyzer.url)


def _attest_source_runtime(
    config: DetSparseV2Config,
) -> tuple[dict[str, object], dict[str, object]]:
    source = dict(current_source_provenance(config.root_dir))
    runtime = dict(current_runtime_provenance(config.root_dir))
    if not _source_valid(source) or not _runtime_valid(runtime):
        raise ValueError("v2 source/runtime attestation is incomplete")
    return source, runtime


def _require_canonical_config(config: DetSparseV2Config) -> DetSparseV2Config:
    canonical = load_det_sparse_v2_config(config.config_path)
    if config != canonical:
        raise ValueError("v2 config object differs from a fresh canonical reload")
    expected_bindings = {
        "planner_version": PLANNER_VERSION,
        "renderer_version": RENDERER_VERSION,
        "selection_version": SELECTION_VERSION,
        "splitter_version": SPLITTER_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "selection_strata": tuple(STRATUM_ORDER),
        "selected_topic_count": PILOT_TOPIC_COUNT,
        "max_facets": MAX_FACETS,
        "min_facet_terms": MIN_FACET_UNIQUE_TERMS,
        "parent_min_unique_terms": MIN_PARENT_UNIQUE_TERMS,
        "bounded_context_min_unique_terms": MIN_CONTEXT_UNIQUE_TERMS,
        "bounded_context_max_unique_terms": MAX_CONTEXT_UNIQUE_TERMS,
        "bounded_context_allocation": CONTEXT_ALLOCATION,
    }
    for field, expected in expected_bindings.items():
        if getattr(canonical, field) != expected:
            raise ValueError(f"v2 config/planner binding mismatch: {field}")
    return canonical


def load_candidate_topics(config: DetSparseV2Config) -> tuple[CandidateTopic, ...]:
    """Decode only the 13 frozen candidate narratives, never excluded records."""

    if config.topics_format != "tsv":
        raise ValueError("det_sparse_v2 supports only the frozen TSV source")
    requested = set(config.candidate_topic_ids)
    if requested.intersection(config.excluded_topic_ids):
        raise ValueError("v2 candidate and excluded topic sets overlap")
    requested_bytes = {
        topic_id.encode("ascii"): topic_id for topic_id in config.candidate_topic_ids
    }
    found: dict[str, CandidateTopic] = {}
    with config.topics_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            raw_qid, separator, raw_text = raw_line.rstrip(b"\r\n").partition(b"\t")
            # Compare raw bytes first. Unrelated/excluded rows are never UTF-8
            # decoded, including their IDs, so malformed out-of-scope records
            # cannot widen the qrels-blind candidate read boundary.
            qid = requested_bytes.get(raw_qid)
            if qid is None:
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
                topic=Topic(
                    id=qid,
                    title=derive_title(narrative),
                    narrative=narrative,
                ),
                source_line_number=line_number,
                source_line_sha256=_sha256_bytes(raw_line),
            )
    missing = [topic_id for topic_id in config.candidate_topic_ids if topic_id not in found]
    if missing:
        raise ValueError("candidate topic IDs are missing: " + ", ".join(missing))
    return tuple(found[topic_id] for topic_id in config.candidate_topic_ids)


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


def _candidate_table(
    candidate_topics: Sequence[CandidateTopic],
    screens: Sequence[CandidateScreen],
) -> list[dict[str, object]]:
    by_id = {row.topic_id: row for row in screens}
    if len(by_id) != len(screens) or set(by_id) != set(CANDIDATE_TOPIC_IDS):
        raise ValueError("selection screen does not cover the exact candidate universe")
    return [
        {
            **by_id[row.topic.id].to_dict(),
            "source_line_number": row.source_line_number,
            "source_line_sha256": row.source_line_sha256,
        }
        for row in candidate_topics
    ]


def _query_registry(
    selected_plans: Sequence[DeterministicSparsePlanV2],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for plan in selected_plans:
        variants = plan.query_variants()
        if plan.status != "ok" or len(variants) != 1 + len(plan.facets):
            raise ValueError("selected v2 plan is not a complete original+facet plan")
        signatures = [plan.original_bm25_signature, *(facet.bm25_signature for facet in plan.facets)]
        coverage = [
            tuple(unit.unit_id for unit in plan.lexical_units),
            *(facet.coverage_unit_ids for facet in plan.facets),
        ]
        for ordinal, (variant, signature, unit_ids) in enumerate(
            zip(variants, signatures, coverage),
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
                    "is_original": ordinal == 1,
                }
            )
        topic_rows = [row for row in rows if row["topic_id"] == plan.topic_id]
        if len({str(row["query_text"]) for row in topic_rows}) != len(topic_rows):
            raise ValueError("v2 query registry contains an exact alias")
        signatures_seen = {
            tuple((str(term), int(count)) for term, count in row["bm25_signature"])
            for row in topic_rows
        }
        if len(signatures_seen) != len(topic_rows):
            raise ValueError("v2 query registry contains a BM25-signature alias")
    return rows


def _coverage_paths(
    selected_plans: Sequence[DeterministicSparsePlanV2],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for plan in selected_plans:
        assignments = [
            unit_id
            for facet in plan.facets
            for unit_id in facet.coverage_unit_ids
        ]
        expected_unit_ids = [unit.unit_id for unit in plan.lexical_units]
        if (
            assignments != expected_unit_ids
            or len(assignments) != len(set(assignments))
        ):
            raise ValueError("v2 coverage ledger does not assign every unit exactly once")
        by_unit = {
            unit_id: facet
            for facet in plan.facets
            for unit_id in facet.coverage_unit_ids
        }
        if set(by_unit) != {unit.unit_id for unit in plan.lexical_units}:
            raise ValueError("v2 coverage ledger does not cover every unit exactly")
        for unit in plan.lexical_units:
            facet = by_unit[unit.unit_id]
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
                    "non_original_path": (
                        facet.bm25_signature != plan.original_bm25_signature
                        and facet.query_text != plan.original_query_text
                    ),
                }
            )
    if any(row["non_original_path"] is not True for row in rows):
        raise ValueError("a v2 coverage unit lacks a non-original retrieval path")
    return rows


def _request_projections(
    selected_plans: Sequence[DeterministicSparsePlanV2],
) -> list[dict[str, object]]:
    rows = []
    for plan in selected_plans:
        facet_count = len(plan.facets)
        base = 1 + facet_count
        expansion_bases = facet_count
        derived = base + expansion_bases
        rows.append(
            {
                "topic_id": plan.topic_id,
                "facet_count": facet_count,
                "base_unique_requests": base,
                "eligible_expansion_bases": expansion_bases,
                "derived_max_unique_requests": derived,
            }
        )
    return rows


def _validate_stored_request_projections(
    stored: object,
    expected: Sequence[Mapping[str, object]],
    *,
    max_per_topic: int,
    max_global: int,
) -> list[dict[str, object]]:
    if not isinstance(stored, list) or len(stored) != len(expected):
        raise ValueError("v2 request projection count/type mismatch")
    exact_keys = {
        "topic_id",
        "facet_count",
        "base_unique_requests",
        "eligible_expansion_bases",
        "derived_max_unique_requests",
    }
    validated: list[dict[str, object]] = []
    total = 0
    for ordinal, (raw_row, expected_row) in enumerate(
        zip(stored, expected), start=1
    ):
        row = _require_exact_keys(raw_row, exact_keys, f"request projection {ordinal}")
        if not isinstance(row.get("topic_id"), str):
            raise ValueError("v2 request projection topic_id must be text")
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
        if base != 1 + facet_count or expansion != facet_count or derived != base + expansion:
            raise ValueError("v2 request projection arithmetic is inconsistent")
        if not 1 <= derived <= max_per_topic:
            raise ValueError("v2 request projection exceeds the per-topic ceiling")
        if row != dict(expected_row):
            raise ValueError("v2 request projection does not replay exactly")
        total += derived
        validated.append(row)
    if total > max_global:
        raise ValueError("v2 request projection exceeds the global ceiling")
    return validated


def _expected_materialization(
    config: DetSparseV2Config,
    candidate_topics: Sequence[CandidateTopic],
    selection_outcome: SelectionOutcome,
) -> dict[str, object]:
    selection = selection_outcome.selection
    selected_plans = tuple(
        selection_outcome.plans_by_topic[topic_id]
        for topic_id in selection.selected_topic_ids
    )
    candidate_table = _candidate_table(candidate_topics, selection.screens)
    query_registry = _query_registry(selected_plans) if selected_plans else []
    coverage_paths = _coverage_paths(selected_plans) if selected_plans else []
    projections = _request_projections(selected_plans)
    if any(
        type(row["derived_max_unique_requests"]) is not int
        or not 1 <= int(row["derived_max_unique_requests"]) <= 9
        for row in projections
    ):
        raise ValueError("v2 request projection exceeds the per-topic ceiling")
    total = sum(int(row["derived_max_unique_requests"]) for row in projections)
    if total > config.cost.max_external_requests:
        raise ValueError("v2 request projection exceeds the global ceiling")
    return {
        "selection": selection,
        "selected_plans": selected_plans,
        "candidate_table": candidate_table,
        "query_registry": query_registry,
        "coverage_paths": coverage_paths,
        "request_projections": projections,
        "planned_base_requests": len(query_registry),
        "derived_max_total_requests": total,
    }


def build_preflight(
    config: DetSparseV2Config,
) -> V2PreflightResult:
    """Screen, select, and freeze v2 plans without reading qrels or retrieval."""

    config = _require_canonical_config(config)
    destination = _fixed_preflight_output(config)
    if os.path.lexists(destination):
        raise FileExistsError(f"v2 preflight output already exists: {destination}")
    query_analyzer = _fresh_query_analyzer(config)
    _require_bound_query_analyzer(config, query_analyzer)
    source, runtime = _attest_source_runtime(config)
    analyzer_sha256 = _fingerprint_sha256(query_analyzer)
    if analyzer_sha256 != config.analyzer.expected_fingerprint_sha256:
        raise ValueError("v2 query analyzer fingerprint differs from frozen config")
    if _attest_source_runtime(config) != (source, runtime):
        raise ValueError("v2 source/runtime changed before preflight reservation")
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
        "source": source,
        "runtime": runtime,
        "state": "reserved_before_structural_screening",
        "external_calls": 0,
        "model_calls": 0,
        "reranker_calls": 0,
        "qrels_opened": False,
    }
    _create_only(reservation_path, _canonical_bytes(reservation, pretty=True))
    candidate_topics = load_candidate_topics(config)
    outcome = screen_and_select_structural_topics(
        tuple(row.topic for row in candidate_topics),
        query_analyzer=query_analyzer,
        candidate_topic_ids=config.candidate_topic_ids,
        seed=config.selection_seed,
    )
    materialized = _expected_materialization(config, candidate_topics, outcome)
    selection = materialized["selection"]
    assert hasattr(selection, "status")
    selected_plans = materialized["selected_plans"]
    assert isinstance(selected_plans, tuple)
    if _attest_source_runtime(config) != (source, runtime):
        raise ValueError("v2 source/runtime changed between screening and writes")
    fallback_topic_ids = [plan.topic_id for plan in selected_plans if plan.status != "ok"]
    mechanical_valid = (
        selection.status == "ok"
        and len(selected_plans) == PILOT_TOPIC_COUNT
        and not fallback_topic_ids
        and _source_valid(source)
        and _runtime_valid(runtime)
    )

    artifacts: list[Path] = [reservation_path]
    selection_path = destination / "selection.json"
    _create_only(selection_path, _canonical_bytes(selection.to_dict(), pretty=True))
    artifacts.append(selection_path)
    candidate_path = destination / "candidate_screen.json"
    _create_only(
        candidate_path,
        _canonical_bytes(materialized["candidate_table"], pretty=True),
    )
    artifacts.append(candidate_path)

    plan_rows: list[dict[str, object]] = []
    for ordinal, plan in enumerate(selected_plans, start=1):
        plan_path = destination / "plans" / f"{ordinal:02d}_{plan.topic_id}.json"
        _create_only(plan_path, _canonical_bytes(plan.to_dict(), pretty=True))
        artifacts.append(plan_path)
        plan_rows.append(
            {
                "ordinal": ordinal,
                "topic_id": plan.topic_id,
                "status": plan.status,
                "plan_path": plan_path.relative_to(destination).as_posix(),
                "plan_semantic_sha256": semantic_plan_sha256(plan),
                "plan_artifact_sha256": _sha256_file(plan_path),
                "facet_count": len(plan.facets),
            }
        )

    query_path = destination / "queries.jsonl"
    _create_only(query_path, _expected_queries_bytes(selected_plans))
    artifacts.append(query_path)
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

    metadata = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "config_path": str(config.config_path),
        "config_sha256": _sha256_file(config.config_path),
        "topic_source_path": str(config.topics_path),
        "candidate_topic_ids": list(config.candidate_topic_ids),
        "excluded_topic_ids": list(config.excluded_topic_ids),
        "selected_topic_ids": list(selection.selected_topic_ids),
        "selection_strata": list(config.selection_strata),
        "selected_topic_count": config.selected_topic_count,
        "reservation_path": reservation_path.relative_to(destination).as_posix(),
        "reservation_sha256": _sha256_file(reservation_path),
        "selection_path": selection_path.relative_to(destination).as_posix(),
        "selection_sha256": _sha256_file(selection_path),
        "candidate_screen_path": candidate_path.relative_to(destination).as_posix(),
        "candidate_screen_sha256": _sha256_file(candidate_path),
        "plans": plan_rows,
        "planner_version": PLANNER_VERSION,
        "renderer_version": RENDERER_VERSION,
        "selection_version": SELECTION_VERSION,
        "splitter_version": SPLITTER_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "parent_min_unique_terms": MIN_PARENT_UNIQUE_TERMS,
        "bounded_context_min_unique_terms": MIN_CONTEXT_UNIQUE_TERMS,
        "bounded_context_max_unique_terms": MAX_CONTEXT_UNIQUE_TERMS,
        "bounded_context_allocation": CONTEXT_ALLOCATION,
        "required_results": config.retrieval.required_results,
        "max_attempts": config.retrieval.max_attempts,
        "retry_policy": config.retrieval.retry_policy,
        "redirect_policy": config.retrieval.redirect_policy,
        "retrieval_run_namespace": config.retrieval_run_namespace,
        "global_ticket_namespace": config.global_ticket_namespace,
        "analyzer_fingerprint": query_analyzer.fingerprint.to_dict(),
        "analyzer_fingerprint_sha256": _fingerprint_sha256(query_analyzer),
        "arms": list(ARM_NAMES),
        "queries_path": query_path.relative_to(destination).as_posix(),
        "queries_sha256": _sha256_file(query_path),
        "query_registry_path": registry_path.relative_to(destination).as_posix(),
        "query_registry_sha256": _sha256_file(registry_path),
        "coverage_paths_path": coverage_path.relative_to(destination).as_posix(),
        "coverage_paths_sha256": _sha256_file(coverage_path),
        "planned_base_unique_requests": materialized["planned_base_requests"],
        "request_projections": materialized["request_projections"],
        "derived_max_total_unique_requests": materialized[
            "derived_max_total_requests"
        ],
        "hard_external_request_ceiling": config.cost.max_external_requests,
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
    if _attest_source_runtime(config) != (source, runtime):
        raise ValueError("v2 source/runtime changed before the freeze completed")
    _validate_preflight_replay(
        config,
        destination,
        require_mechanical_valid=mechanical_valid,
    )
    return V2PreflightResult(
        output_dir=destination,
        selected_topic_ids=selection.selected_topic_ids,
        selected_plans=selected_plans,
        valid=mechanical_valid,
        planned_base_requests=int(materialized["planned_base_requests"]),
        derived_max_total_requests=int(materialized["derived_max_total_requests"]),
        hard_external_request_ceiling=config.cost.max_external_requests,
    )


def _validate_preflight_replay(
    config: DetSparseV2Config,
    output_dir: Path,
    *,
    require_mechanical_valid: bool,
) -> dict[str, object]:
    """Replay v2 selection/planning and reject any semantic or hash drift."""

    config = _require_canonical_config(config)
    query_analyzer = _fresh_query_analyzer(config)
    _require_bound_query_analyzer(config, query_analyzer)
    expected_source, expected_runtime = _attest_source_runtime(config)
    output = Path(os.path.abspath(os.fspath(output_dir)))
    expected_output = _fixed_preflight_output(config)
    if output != expected_output:
        raise ValueError("v2 preflight path differs from the fixed experiment path")
    if output.is_symlink() or not output.is_dir():
        raise ValueError("v2 preflight output must be a real fixed directory")
    for artifact in output.rglob("*"):
        if artifact.is_symlink():
            raise ValueError("v2 preflight artifacts and directories must not be symlinks")
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
    metadata = _require_exact_keys(
        json.loads(metadata_path.read_text(encoding="utf-8")),
        _METADATA_KEYS,
        "v2 preflight metadata",
    )
    if metadata.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("v2 preflight metadata schema mismatch")
    if (
        metadata.get("experiment_id") != config.experiment_id
        or metadata.get("config_path") != str(config.config_path)
        or metadata.get("config_sha256") != _sha256_file(config.config_path)
        or metadata.get("topic_source_path") != str(config.topics_path)
    ):
        raise ValueError("v2 preflight config/experiment identity mismatch")
    if _require_sha256(metadata.get("config_sha256"), "config_sha256") != _sha256_file(
        config.config_path
    ):
        raise ValueError("v2 preflight config hash mismatch")
    if _strict_string_list(
        metadata.get("candidate_topic_ids"), "candidate_topic_ids"
    ) != list(config.candidate_topic_ids):
        raise ValueError("v2 candidate topic IDs differ from frozen config")
    if _strict_string_list(
        metadata.get("excluded_topic_ids"), "excluded_topic_ids"
    ) != list(config.excluded_topic_ids):
        raise ValueError("v2 excluded topic IDs differ from frozen config")
    if (
        metadata.get("planner_version") != PLANNER_VERSION
        or metadata.get("renderer_version") != RENDERER_VERSION
        or metadata.get("selection_version") != SELECTION_VERSION
        or metadata.get("splitter_version") != SPLITTER_VERSION
        or metadata.get("tokenizer_version") != TOKENIZER_VERSION
        or metadata.get("arms") != list(config.arms)
    ):
        raise ValueError("v2 planner/renderer/selection/arm identity mismatch")
    if (
        metadata.get("selection_strata") != list(config.selection_strata)
        or _strict_int(
            metadata.get("selected_topic_count"), "selected_topic_count"
        )
        != config.selected_topic_count
        or _strict_int(
            metadata.get("parent_min_unique_terms"), "parent_min_unique_terms"
        )
        != config.parent_min_unique_terms
        or _strict_int(
            metadata.get("bounded_context_min_unique_terms"),
            "bounded_context_min_unique_terms",
        )
        != config.bounded_context_min_unique_terms
        or _strict_int(
            metadata.get("bounded_context_max_unique_terms"),
            "bounded_context_max_unique_terms",
        )
        != config.bounded_context_max_unique_terms
        or metadata.get("bounded_context_allocation")
        != config.bounded_context_allocation
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
        raise ValueError("v2 frozen structural/retrieval contract mismatch")
    if (
        metadata.get("external_gate_status") != config.external_gate_status
        or metadata.get("external_gate_reason") != config.external_gate_reason
    ):
        raise ValueError("v2 external gate identity mismatch")
    if _strict_int(
        metadata.get("hard_external_request_ceiling"),
        "hard_external_request_ceiling",
    ) != config.cost.max_external_requests:
        raise ValueError("v2 hard external request ceiling mismatch")
    if metadata.get("source") != expected_source or metadata.get("runtime") != expected_runtime:
        raise ValueError("v2 preflight source/runtime provenance mismatch")
    if not _source_valid(expected_source) or not _runtime_valid(expected_runtime):
        raise ValueError("v2 preflight source/runtime provenance is incomplete")
    live_fingerprint = query_analyzer.fingerprint.to_dict()
    live_fingerprint_sha256 = _sha256_bytes(
        json.dumps(
            live_fingerprint,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    if live_fingerprint_sha256 != config.analyzer.expected_fingerprint_sha256:
        raise ValueError("v2 validation analyzer differs from frozen config")
    if (
        metadata.get("analyzer_fingerprint") != _json_normalize(live_fingerprint)
        or _require_sha256(
            metadata.get("analyzer_fingerprint_sha256"),
            "analyzer_fingerprint_sha256",
        )
        != live_fingerprint_sha256
    ):
        raise ValueError("v2 preflight analyzer identity/hash mismatch")
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
        "source": expected_source,
        "runtime": expected_runtime,
        "state": "reserved_before_structural_screening",
        "external_calls": 0,
        "model_calls": 0,
        "reranker_calls": 0,
        "qrels_opened": False,
    }
    if reservation_path.read_bytes() != _canonical_bytes(
        expected_reservation, pretty=True
    ):
        raise ValueError("v2 preflight reservation does not replay exactly")
    if _require_sha256(
        metadata.get("reservation_sha256"), "reservation_sha256"
    ) != _sha256_file(reservation_path):
        raise ValueError("v2 preflight reservation hash mismatch")

    candidate_topics = load_candidate_topics(config)
    outcome = screen_and_select_structural_topics(
        tuple(row.topic for row in candidate_topics),
        query_analyzer=query_analyzer,
        candidate_topic_ids=config.candidate_topic_ids,
        seed=config.selection_seed,
    )
    materialized = _expected_materialization(config, candidate_topics, outcome)
    selection = materialized["selection"]
    selection_path = _fixed_artifact_path(
        output,
        metadata.get("selection_path"),
        "selection.json",
        "selection_path",
    )
    candidate_path = _fixed_artifact_path(
        output,
        metadata.get("candidate_screen_path"),
        "candidate_screen.json",
        "candidate_screen_path",
    )
    if _require_sha256(metadata.get("selection_sha256"), "selection_sha256") != _sha256_file(
        selection_path
    ):
        raise ValueError("v2 selection artifact hash mismatch")
    if _require_sha256(
        metadata.get("candidate_screen_sha256"), "candidate_screen_sha256"
    ) != _sha256_file(candidate_path):
        raise ValueError("v2 candidate screen artifact hash mismatch")
    stored_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if stored_selection != _json_normalize(selection.to_dict()):
        raise ValueError("v2 structural selection does not replay exactly")
    stored_candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    if stored_candidates != _json_normalize(materialized["candidate_table"]):
        raise ValueError("v2 candidate structural table does not replay exactly")
    if _strict_string_list(
        metadata.get("selected_topic_ids"), "selected_topic_ids"
    ) != list(selection.selected_topic_ids):
        raise ValueError("v2 selected topic IDs differ from frozen selection")

    selected_plans = materialized["selected_plans"]
    assert isinstance(selected_plans, tuple)
    plan_rows = metadata.get("plans")
    if not isinstance(plan_rows, list) or len(plan_rows) != len(selected_plans):
        raise ValueError("v2 selected plan count mismatch")
    plan_paths: list[Path] = []
    plan_row_keys = {
        "ordinal",
        "topic_id",
        "status",
        "plan_path",
        "plan_semantic_sha256",
        "plan_artifact_sha256",
        "facet_count",
    }
    for ordinal, (plan, raw_row) in enumerate(zip(selected_plans, plan_rows), start=1):
        row = _require_exact_keys(raw_row, plan_row_keys, f"selected plan row {ordinal}")
        if (
            _strict_int(row.get("ordinal"), "plan ordinal") != ordinal
            or row.get("topic_id") != plan.topic_id
        ):
            raise ValueError("v2 selected plan order/identity mismatch")
        expected_plan_path = f"plans/{ordinal:02d}_{plan.topic_id}.json"
        plan_path = _fixed_artifact_path(
            output,
            row.get("plan_path"),
            expected_plan_path,
            f"plan_path {ordinal}",
        )
        plan_paths.append(plan_path)
        stored_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (
            stored_plan != _json_normalize(plan.to_dict())
            or _require_sha256(
                row.get("plan_semantic_sha256"), "plan_semantic_sha256"
            )
            != semantic_plan_sha256(plan)
            or _require_sha256(
                row.get("plan_artifact_sha256"), "plan_artifact_sha256"
            )
            != _sha256_file(plan_path)
            or row.get("status") != "ok"
            or _strict_int(row.get("facet_count"), "plan facet_count")
            != len(plan.facets)
        ):
            raise ValueError("v2 selected plan does not replay exactly")

    queries_path = _fixed_artifact_path(
        output,
        metadata.get("queries_path"),
        "queries.jsonl",
        "queries_path",
    )
    expected_query_bytes = _expected_queries_bytes(selected_plans)
    if queries_path.read_bytes() != expected_query_bytes:
        raise ValueError("v2 queries.jsonl does not replay exactly")
    if _require_sha256(metadata.get("queries_sha256"), "queries_sha256") != _sha256_file(
        queries_path
    ):
        raise ValueError("v2 queries.jsonl hash mismatch")
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
    if _require_sha256(
        metadata.get("query_registry_sha256"), "query_registry_sha256"
    ) != _sha256_file(registry_path):
        raise ValueError("v2 query registry hash mismatch")
    if _require_sha256(
        metadata.get("coverage_paths_sha256"), "coverage_paths_sha256"
    ) != _sha256_file(coverage_path):
        raise ValueError("v2 coverage path hash mismatch")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if registry != _json_normalize(materialized["query_registry"]):
        raise ValueError("v2 base query registry does not replay exactly")
    if coverage != _json_normalize(materialized["coverage_paths"]):
        raise ValueError("v2 coverage path ledger does not replay exactly")
    projections = _validate_stored_request_projections(
        metadata.get("request_projections"),
        materialized["request_projections"],
        max_per_topic=config.cost.max_unique_requests_per_topic,
        max_global=config.cost.max_external_requests,
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
        or planned_base != len(registry)
        or derived_total != materialized["derived_max_total_requests"]
        or derived_total
        != sum(int(row["derived_max_unique_requests"]) for row in projections)
    ):
        raise ValueError("v2 request projection does not replay exactly")
    if type(metadata.get("mechanical_valid")) is not bool:
        raise ValueError("v2 mechanical_valid must be boolean")
    fallback_topic_ids = [
        plan.topic_id for plan in selected_plans if plan.status != "ok"
    ]
    expected_mechanical_valid = (
        selection.status == "ok"
        and len(selected_plans) == PILOT_TOPIC_COUNT
        and not fallback_topic_ids
        and _source_valid(expected_source)
        and _runtime_valid(expected_runtime)
    )
    if metadata.get("mechanical_valid") is not expected_mechanical_valid:
        raise ValueError("v2 mechanical validity does not replay exactly")
    if require_mechanical_valid and not expected_mechanical_valid:
        raise ValueError("v2 preflight is mechanically invalid")
    if _strict_string_list(
        metadata.get("fallback_topic_ids"), "fallback_topic_ids"
    ) != fallback_topic_ids:
        raise ValueError("v2 preflight fallback topics do not replay exactly")
    if metadata.get("external_gate_status") != "blocked":
        raise ValueError("v2 external gate is not blocked")
    for field in ("external_calls", "model_calls", "reranker_calls"):
        if _strict_int(metadata.get(field), field) != 0:
            raise ValueError(f"v2 preflight {field} must be zero")
    if type(metadata.get("qrels_opened")) is not bool or metadata.get("qrels_opened") is not False:
        raise ValueError("v2 preflight violated the qrels firewall")

    expected_artifacts = {
        "preflight_reservation.json",
        "selection.json",
        "candidate_screen.json",
        "queries.jsonl",
        "base_query_registry.json",
        "coverage_paths.json",
        "_preflight.json",
        *(path.relative_to(output).as_posix() for path in plan_paths),
    }
    raw_manifest_artifacts = manifest.get("artifacts")
    if not isinstance(raw_manifest_artifacts, list):
        raise ValueError("v2 freeze artifact table is missing")
    manifest_artifacts: set[str] = set()
    for ordinal, raw_row in enumerate(raw_manifest_artifacts, start=1):
        row = _require_exact_keys(
            raw_row,
            {"path", "size", "sha256"},
            f"freeze artifact row {ordinal}",
        )
        relative = row.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("v2 freeze artifact path must be text")
        if relative in manifest_artifacts:
            raise ValueError("v2 freeze contains duplicate artifact paths")
        manifest_artifacts.add(relative)
        _strict_int(row.get("size"), "freeze artifact size")
        _require_sha256(row.get("sha256"), "freeze artifact sha256")
    if manifest_artifacts != expected_artifacts:
        raise ValueError("v2 freeze artifact inventory differs from fixed preflight files")
    observed_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_artifacts | {"pre_retrieval_freeze.json"}:
        raise ValueError("v2 preflight directory has unsealed or missing files")
    freeze_metadata = manifest.get("metadata")
    expected_freeze_metadata = _freeze_metadata(metadata)
    _require_exact_keys(
        freeze_metadata,
        set(_FREEZE_METADATA_FIELDS),
        "v2 freeze metadata",
    )
    if freeze_metadata != expected_freeze_metadata:
        raise ValueError("v2 freeze metadata does not exactly mirror preflight metadata")
    if _attest_source_runtime(config) != (expected_source, expected_runtime):
        raise ValueError("v2 source/runtime changed during validation replay")
    return manifest


def validate_preflight(
    config: DetSparseV2Config,
    output_dir: Path,
) -> dict[str, object]:
    """Validate a successful canonical preflight as an external-call gate."""

    return _validate_preflight_replay(
        config,
        output_dir,
        require_mechanical_valid=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qrels-blind structural preflight for det_sparse_v2",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_det_sparse_v2_config(args.config)
    output = config.output_dir / "preflight"
    if args.validate_only:
        validate_preflight(config, output)
        print(f"Validated frozen v2 preflight under {output.resolve()}")
        return 0
    result = build_preflight(config)
    print(f"Wrote frozen v2 preflight under {result.output_dir}")
    print(
        "Selected topics: "
        + (", ".join(result.selected_topic_ids) if result.selected_topic_ids else "none")
    )
    print(f"Mechanical valid: {str(result.valid).lower()}")
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
