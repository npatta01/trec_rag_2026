"""Authenticated, zero-call import of one frozen historical planning seed.

This module deliberately treats ``decomposition/result.json`` as an opaque
byte artifact.  The production decomposition loader validates the source and
destination, while this module authenticates the byte copy and its receipts.
No planner or hosted service is reachable from this interface.

The 22-topic/283-variant/138-lane shape is part of that seed's authenticated
identity, not a generic planning-import API. Topics outside this exact inventory
may coexist in the same run and are planned live through the production runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import TYPE_CHECKING

from trec_rag.facet_retrieval_lanes import (
    FACET_RETRIEVAL_LANE_PROJECTOR_VERSION,
    build_retrieval_lanes,
)
from trec_rag.topics import Topic

if TYPE_CHECKING:
    from trec_rag.competition_retrieval import ValidatedDecomposition


PLANNING_SEED_MODE = "validated-decomposition-byte-import-v2"
PLANNING_SEED_MANIFEST_SCHEMA = "planning_seed_manifest_v2"
PLANNING_SEED_RECEIPT_SCHEMA = "planning_seed_receipt_v2"
PLANNING_SEED_SOURCE_INVENTORY_SCHEMA = "planning_seed_source_inventory_v1"
PLANNING_SEED_MANIFEST_FILENAME = "planning-seed-manifest.json"
PLANNING_SEED_RECEIPT_FILENAME = "planning-seed-receipt.json"
EXPECTED_TOPIC_COUNT = 22
EXPECTED_PLANNER_QUERY_VARIANT_COUNT = 283
EXPECTED_RETRIEVAL_LANE_COUNT = 138
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class SeedSource:
    """One source/destination logical-path binding for an official topic."""

    topic: Topic
    source_relative_path: str
    destination_relative_path: str


@dataclass(frozen=True)
class PlanningSeedProvenance:
    """Declared historical provenance; it is not a hosted-response receipt."""

    source_run_id: str
    source_commit: str
    source_config_sha256: str
    source_inventory_sha256: str


@dataclass(frozen=True)
class PlanningLaneRecord:
    """One ordered production retrieval/scoring lane projected from a plan."""

    topic_id: str
    topic_order: int
    lane_order: int
    lane_name: str
    subnarrative_id: str | None
    retrieval_source_type: str
    retrieval_query_sha256: str
    scoring_source_type: str
    scoring_query_sha256: str

    def canonical_dict(self) -> dict[str, object]:
        return {
            "topic_id": self.topic_id,
            "topic_order": self.topic_order,
            "lane_order": self.lane_order,
            "lane_name": self.lane_name,
            "subnarrative_id": self.subnarrative_id,
            "retrieval_source_type": self.retrieval_source_type,
            "retrieval_query_sha256": self.retrieval_query_sha256,
            "scoring_source_type": self.scoring_source_type,
            "scoring_query_sha256": self.scoring_query_sha256,
        }


@dataclass(frozen=True)
class PlanningSeedRequest:
    """Complete authenticated input for a 22-topic planning seed."""

    source_root: Path
    destination_root: Path
    topics: tuple[Topic, ...]
    sources: tuple[SeedSource, ...]
    required_lane_records: tuple[PlanningLaneRecord, ...]
    provenance: PlanningSeedProvenance


@dataclass(frozen=True)
class PlanningSeedReceipt:
    """A verified aggregate receipt and its create-only filesystem location."""

    path: Path
    content_sha256: str
    content: Mapping[str, object]


@dataclass(frozen=True)
class _TopicSeed:
    source: SeedSource
    source_path: Path
    destination_path: Path
    manifest_path: Path
    source_bytes: bytes
    source_sha256: str
    validated: ValidatedDecomposition
    planner_query_records: tuple[dict[str, object], ...]
    retrieval_lane_records: tuple[PlanningLaneRecord, ...]
    manifest: dict[str, object]
    manifest_bytes: bytes
    manifest_sha256: str


def import_planning_seed(request: PlanningSeedRequest) -> PlanningSeedReceipt:
    """Validate, byte-import, and receipt a complete zero-call planning seed.

    Existing destination artifacts are never replaced.  An existing complete
    and byte-identical seed may converge; any missing, malformed, or
    contradictory state raises ``ValueError``.
    """

    with _planning_seed_lock(request.destination_root):
        prepared = _prepare(request)
        _ensure_directory(_absolute_root(request.destination_root, "destination root"))
        _preflight_existing_state(request, prepared)

        for item in prepared:
            _publish_create_only(item.destination_path, item.source_bytes)
            if _read_stable_source(item.source_path)[0] != item.source_bytes:
                raise ValueError("source changed during copy")
            _publish_create_only(item.manifest_path, item.manifest_bytes)

        _verify_prepared_files(prepared)
        aggregate_path = _absolute_root(request.destination_root, "destination root") / PLANNING_SEED_RECEIPT_FILENAME
        aggregate_bytes, aggregate_content = _aggregate_bytes(request, prepared)
        _verify_prepared_files(prepared)
        _publish_create_only(aggregate_path, aggregate_bytes)
        receipt = _read_receipt(aggregate_path, aggregate_content)
        _verify_prepared_files(prepared)
        return receipt


def verify_planning_seed(request: PlanningSeedRequest) -> PlanningSeedReceipt:
    """Revalidate every source, destination result, and receipt without a backend."""

    with _planning_seed_lock(request.destination_root):
        prepared = _prepare(request)
        destination_root = _absolute_root(request.destination_root, "destination root")
        _verify_destination_files(prepared, destination_root)

        aggregate_path = destination_root / PLANNING_SEED_RECEIPT_FILENAME
        expected_bytes, expected_content = _aggregate_bytes(request, prepared)
        _verify_prepared_files(prepared)
        actual_bytes = _read_regular(aggregate_path, "planning seed aggregate receipt")
        if actual_bytes != expected_bytes:
            raise ValueError("planning seed aggregate receipt changed or is partial")
        receipt = _read_receipt(aggregate_path, expected_content)
        _verify_prepared_files(prepared)
        return receipt


def load_seeded_decompositions(
    request: PlanningSeedRequest,
) -> tuple[ValidatedDecomposition, ...]:
    """Load the 22 verified destination decompositions with no planning call.

    The returned objects are the validated objects from the prepared source
    snapshot.  The destination files are checked byte-for-byte against that
    snapshot before return, so this function never reopens a destination path
    after verification and accidentally accepts a different valid plan for the
    same topic.
    """

    with _planning_seed_lock(request.destination_root):
        prepared = _prepare(request)
        destination_root = _absolute_root(request.destination_root, "destination root")
        _verify_destination_files(prepared, destination_root)

        aggregate_path = destination_root / PLANNING_SEED_RECEIPT_FILENAME
        expected_bytes, expected_content = _aggregate_bytes(request, prepared)
        _verify_prepared_files(prepared)
        actual_bytes = _read_regular(aggregate_path, "planning seed aggregate receipt")
        if actual_bytes != expected_bytes:
            raise ValueError("planning seed aggregate receipt changed or is partial")
        _read_receipt(aggregate_path, expected_content)
        _verify_prepared_files(prepared)
        return tuple(item.validated for item in prepared)


def planning_seed_source_inventory_sha256(
    source_root: Path,
    sources: tuple[SeedSource, ...],
) -> str:
    """Digest the exact ordered source files used by a planning seed.

    The digest binds each topic ID and canonical relative path to the byte
    length and SHA-256 of a stable, regular-file read.  It can therefore be
    computed before constructing :class:`PlanningSeedProvenance` and is
    independently recomputed during import and verification.
    """

    root = _absolute_root(source_root, "source root")
    _ensure_source_root(root)
    if type(sources) is not tuple:
        raise TypeError("sources must be an ordered tuple")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, SeedSource):
            raise TypeError("sources must contain SeedSource values")
        relative = _checked_relative_path(source.source_relative_path, "source path")
        source_key = relative.as_posix()
        if source_key in seen:
            raise ValueError("source path is duplicated")
        seen.add(source_key)
        body, _identity = _read_stable_source(
            _join_checked(root, relative, "source path")
        )
        records.append(
            {
                "topic_id": source.topic.id,
                "source_relative_path": source_key,
                "source_byte_length": len(body),
                "source_sha256": _digest(body),
            }
        )
    return _source_inventory_digest(records)


def _prepare(request: PlanningSeedRequest) -> tuple[_TopicSeed, ...]:
    source_root = _absolute_root(request.source_root, "source root")
    destination_root = _absolute_root(request.destination_root, "destination root")
    _validate_request_shape(request)
    _ensure_source_root(source_root)
    _ensure_destination_root_parent(destination_root)

    validator_sha256, validator_identity = _validator_identity()
    projector_sha256, projector_identity = _lane_projector_identity()
    prepared: list[_TopicSeed] = []
    seen_sources: set[str] = set()
    seen_destinations: set[str] = set()
    for topic_order, source in enumerate(request.sources):
        source_relative = _checked_relative_path(source.source_relative_path, "source path")
        destination_relative = _checked_relative_path(
            source.destination_relative_path, "destination path"
        )
        source_key = "/".join(source_relative.parts)
        destination_key = "/".join(destination_relative.parts)
        if source_key in seen_sources or destination_key in seen_destinations:
            raise ValueError("source or destination path is duplicated")
        seen_sources.add(source_key)
        seen_destinations.add(destination_key)

        source_path = _join_checked(source_root, source_relative, "source path")
        destination_path = _join_checked(
            destination_root, destination_relative, "destination path"
        )
        manifest_path = destination_path.parents[1] / PLANNING_SEED_MANIFEST_FILENAME
        _assert_no_symlinks(manifest_path, "planning seed manifest path")
        source_bytes, source_identity = _read_stable_source(source_path)
        validated = _validate_with_production_loader(source.topic, source_path)
        if validated.source_sha256 != _digest(source_bytes):
            raise ValueError("source changed during validation")
        if source_identity != _path_identity(source_path):
            raise ValueError("source changed during validation")
        planner_query_records = tuple(
            _planner_query_records(validated, topic_order=topic_order)
        )
        retrieval_lane_records = tuple(
            _retrieval_lane_records(
                source.topic,
                validated,
                topic_order=topic_order,
            )
        )
        manifest = _topic_manifest(
            source,
            source_bytes,
            validated,
            planner_query_records=planner_query_records,
            retrieval_lane_records=retrieval_lane_records,
            decomposition_schema=_decomposition_schema(),
            validator_sha256=validator_sha256,
            validator_identity=validator_identity,
            projector_sha256=projector_sha256,
            projector_identity=projector_identity,
        )
        manifest_bytes = _canonical_json_bytes(manifest)
        prepared.append(
            _TopicSeed(
                source=source,
                source_path=source_path,
                destination_path=destination_path,
                manifest_path=manifest_path,
                source_bytes=source_bytes,
                source_sha256=_digest(source_bytes),
                validated=validated,
                planner_query_records=planner_query_records,
                retrieval_lane_records=retrieval_lane_records,
                manifest=manifest,
                manifest_bytes=manifest_bytes,
                manifest_sha256=_digest(manifest_bytes),
            )
        )

    all_lane_records = tuple(
        record
        for item in prepared
        for record in item.retrieval_lane_records
    )
    if len(all_lane_records) != EXPECTED_RETRIEVAL_LANE_COUNT:
        raise ValueError("planning seed must project exactly 138 retrieval lanes")
    all_planner_records = tuple(
        record
        for item in prepared
        for record in item.planner_query_records
    )
    if len(all_planner_records) != EXPECTED_PLANNER_QUERY_VARIANT_COUNT:
        raise ValueError("planning seed must contain exactly 283 planner query variants")
    retrieval_hashes = [record.retrieval_query_sha256 for record in all_lane_records]
    if len(set(retrieval_hashes)) != EXPECTED_RETRIEVAL_LANE_COUNT:
        raise ValueError("duplicate retrieval query identity ambiguity")
    required = _validate_lane_tuple(request.required_lane_records)
    if all_lane_records != required:
        raise ValueError("projected retrieval lanes do not match authenticated request lanes")
    source_inventory_sha256 = _source_inventory_digest(
        [
            {
                "topic_id": item.source.topic.id,
                "source_relative_path": item.source.source_relative_path,
                "source_byte_length": len(item.source_bytes),
                "source_sha256": item.source_sha256,
            }
            for item in prepared
        ]
    )
    if source_inventory_sha256 != request.provenance.source_inventory_sha256:
        raise ValueError("source inventory SHA-256 does not match the exact source files")
    return tuple(prepared)


def _validate_request_shape(request: PlanningSeedRequest) -> None:
    if not isinstance(request, PlanningSeedRequest):
        raise TypeError("request must be a PlanningSeedRequest")
    if type(request.topics) is not tuple or len(request.topics) != EXPECTED_TOPIC_COUNT:
        raise ValueError("planning seed requires exactly 22 ordered topics")
    if any(not isinstance(topic, Topic) for topic in request.topics):
        raise TypeError("topics must contain Topic values")
    topic_ids = [topic.id for topic in request.topics]
    if len(set(topic_ids)) != EXPECTED_TOPIC_COUNT:
        raise ValueError("topic IDs must be unique")
    if type(request.sources) is not tuple or len(request.sources) != EXPECTED_TOPIC_COUNT:
        raise ValueError("planning seed requires one source for each of 22 topics")
    for topic, source in zip(request.topics, request.sources, strict=True):
        if not isinstance(source, SeedSource) or source.topic != topic:
            raise ValueError("sources must preserve the explicit topic order")
        expected_path = f"{topic.id}/decomposition/result.json"
        if (
            source.source_relative_path != expected_path
            or source.destination_relative_path != expected_path
        ):
            raise ValueError("planning seed must use the canonical topic decomposition path")
    if type(request.required_lane_records) is not tuple:
        raise TypeError("required_lane_records must be an ordered tuple")
    _validate_lane_tuple(request.required_lane_records)
    if not isinstance(request.provenance, PlanningSeedProvenance):
        raise TypeError("provenance must be PlanningSeedProvenance")
    provenance = request.provenance
    if not isinstance(provenance.source_run_id, str) or not provenance.source_run_id.strip():
        raise ValueError("source run ID must be non-empty text")
    if not isinstance(provenance.source_commit, str) or not _COMMIT.fullmatch(
        provenance.source_commit
    ):
        raise ValueError("source commit must be a 40-character SHA-1")
    for value, label in (
        (provenance.source_config_sha256, "source config SHA-256"),
        (provenance.source_inventory_sha256, "source inventory SHA-256"),
    ):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{label} is invalid")


def _topic_manifest(
    source: SeedSource,
    source_bytes: bytes,
    validated: ValidatedDecomposition,
    *,
    planner_query_records: tuple[dict[str, object], ...],
    retrieval_lane_records: tuple[PlanningLaneRecord, ...],
    decomposition_schema: str,
    validator_sha256: str,
    validator_identity: str,
    projector_sha256: str,
    projector_identity: str,
) -> dict[str, object]:
    plan_payload_sha256 = _plan_payload_digest(validated)
    source_sha256 = _digest(source_bytes)
    fallback = bool(validated.result.used_fallback)
    lane_payload = [record.canonical_dict() for record in retrieval_lane_records]
    return {
        "schema_version": PLANNING_SEED_MANIFEST_SCHEMA,
        "mode": PLANNING_SEED_MODE,
        "topic_id": source.topic.id,
        "narrative_sha256": validated.narrative_sha256,
        "source_relative_path": source.source_relative_path,
        "destination_relative_path": source.destination_relative_path,
        "source_bytes": len(source_bytes),
        "destination_bytes": len(source_bytes),
        "source_byte_length": len(source_bytes),
        "destination_byte_length": len(source_bytes),
        "source_sha256": source_sha256,
        "destination_sha256": source_sha256,
        "bytes_equal": True,
        "decomposition_schema": decomposition_schema,
        "used_fallback": fallback,
        "fallback_status": "used" if fallback else "not-used",
        "canonical_plan_payload_sha256": plan_payload_sha256,
        "plan_payload_digest": plan_payload_sha256,
        "rendered_planner_query_variant_count": len(planner_query_records),
        "rendered_planner_query_records_sha256": _digest_json(
            planner_query_records
        ),
        "retrieval_lane_count": len(retrieval_lane_records),
        "retrieval_lane_records": lane_payload,
        "retrieval_lane_records_sha256": _digest_json(lane_payload),
        "validator_module_sha256": validator_sha256,
        "validator_identity": validator_identity,
        "lane_projector_version": FACET_RETRIEVAL_LANE_PROJECTOR_VERSION,
        "lane_projector_module_sha256": projector_sha256,
        "lane_projector_identity": projector_identity,
    }


def _aggregate_bytes(
    request: PlanningSeedRequest, prepared: tuple[_TopicSeed, ...]
) -> tuple[bytes, dict[str, object]]:
    ordered_plan = [
        {
            "topic_id": item.source.topic.id,
            "canonical_plan_payload_sha256": item.manifest[
                "canonical_plan_payload_sha256"
            ],
        }
        for item in prepared
    ]
    planner_query_records = [
        record
        for item in prepared
        for record in item.planner_query_records
    ]
    retrieval_lane_records = [
        record.canonical_dict()
        for item in prepared
        for record in item.retrieval_lane_records
    ]
    unique_retrieval_query_hashes = sorted(
        {record["retrieval_query_sha256"] for record in retrieval_lane_records}
    )
    unique_scoring_query_hashes = sorted(
        {record["scoring_query_sha256"] for record in retrieval_lane_records}
    )
    validator_sha256 = prepared[0].manifest["validator_module_sha256"]
    validator_identity = prepared[0].manifest["validator_identity"]
    projector_sha256 = prepared[0].manifest["lane_projector_module_sha256"]
    projector_identity = prepared[0].manifest["lane_projector_identity"]
    required_lane_payload = [
        record.canonical_dict() for record in request.required_lane_records
    ]
    content: dict[str, object] = {
        "schema_version": PLANNING_SEED_RECEIPT_SCHEMA,
        "mode": PLANNING_SEED_MODE,
        "source_run_id": request.provenance.source_run_id,
        "source_commit": request.provenance.source_commit,
        "source_config_sha256": request.provenance.source_config_sha256,
        "source_inventory_sha256": request.provenance.source_inventory_sha256,
        "source_inventory_status": "verified-exact-files",
        "historical_provenance_status": "declared-only",
        "topic_count": len(prepared),
        "topic_ids": [item.source.topic.id for item in prepared],
        "manifest_digests": [item.manifest_sha256 for item in prepared],
        "topic_manifest_digests": [item.manifest_sha256 for item in prepared],
        "ordered_plan_digest": _digest_json(ordered_plan),
        "planner_query_variant_count": len(planner_query_records),
        "planner_query_variant_records_sha256": _digest_json(
            planner_query_records
        ),
        "retrieval_lane_occurrence_count": len(retrieval_lane_records),
        "retrieval_lane_records": retrieval_lane_records,
        "retrieval_lane_records_sha256": _digest_json(retrieval_lane_records),
        "unique_retrieval_query_hash_count": len(unique_retrieval_query_hashes),
        "unique_retrieval_query_hash_set_sha256": _digest_json(
            unique_retrieval_query_hashes
        ),
        "unique_scoring_query_hash_count": len(unique_scoring_query_hashes),
        "unique_scoring_query_hash_set_sha256": _digest_json(
            unique_scoring_query_hashes
        ),
        "authenticated_required_lane_count": len(required_lane_payload),
        "authenticated_required_lane_records_sha256": _digest_json(
            required_lane_payload
        ),
        "planning_backend_invocations": 0,
        "hosted_planning_calls": 0,
        "poison_backend_result": "passed",
        "validator_module_sha256": validator_sha256,
        "validator_identity": validator_identity,
        "lane_projector_version": FACET_RETRIEVAL_LANE_PROJECTOR_VERSION,
        "lane_projector_module_sha256": projector_sha256,
        "lane_projector_identity": projector_identity,
    }
    content["receipt_content_sha256"] = _digest_json(content)
    return _canonical_json_bytes(content), content


def _planner_query_records(
    validated: ValidatedDecomposition,
    *,
    topic_order: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for query_order, query in enumerate(validated.result.queries):
        subnarrative_id = _planner_subnarrative_id(query.variant_name)
        record = {
            "topic_id": validated.topic_id,
            "topic_order": topic_order,
            "query_order": query_order,
            "variant_name": query.variant_name,
            "source_type": query.source_type,
            "query_sha256": _digest(query.query_text.encode("utf-8")),
            "subnarrative_id": subnarrative_id,
        }
        identity = tuple(record.values())
        if identity in seen:
            raise ValueError("duplicate query identity ambiguity")
        seen.add(identity)
        records.append(record)
    return records


def _retrieval_lane_records(
    topic: Topic,
    validated: ValidatedDecomposition,
    *,
    topic_order: int,
) -> list[PlanningLaneRecord]:
    lanes = build_retrieval_lanes(
        topic,
        validated.result.queries,
        validated.result.subnarratives,
    )
    return [
        PlanningLaneRecord(
            topic_id=topic.id,
            topic_order=topic_order,
            lane_order=lane_order,
            lane_name=lane.retrieval_query.variant_name,
            subnarrative_id=lane.subnarrative_id,
            retrieval_source_type=lane.retrieval_query.source_type,
            retrieval_query_sha256=lane.bm25_query_sha256,
            scoring_source_type=lane.scoring_query.source_type,
            scoring_query_sha256=lane.semantic_query_sha256,
        )
        for lane_order, lane in enumerate(lanes)
    ]


def _planner_subnarrative_id(variant_name: object) -> str | None:
    if variant_name == "original":
        return None
    if not isinstance(variant_name, str) or not variant_name.startswith("facet:"):
        raise ValueError("query variant is not canonical")
    parts = variant_name.split(":")
    if len(parts) != 3 or not parts[1] or not re.fullmatch(r"q[1-9][0-9]*", parts[2]):
        raise ValueError("query variant has no canonical subnarrative lane")
    return parts[1]


def _plan_payload_digest(validated: ValidatedDecomposition) -> str:
    plan = validated.result.plan
    payload: object
    if plan is None:
        payload = None
    else:
        payload = {
            "schema_version": "subnarrative_queries_v1",
            "topic_id": plan.topic_id,
            "subnarratives": [
                {"subnarrative": row.text, "bm25_queries": list(row.bm25_queries)}
                for row in plan.subnarratives
            ],
        }
    return _digest_json(payload)


def _validate_lane_tuple(
    values: tuple[PlanningLaneRecord, ...],
) -> tuple[PlanningLaneRecord, ...]:
    if len(values) != EXPECTED_RETRIEVAL_LANE_COUNT:
        raise ValueError("authenticated request lanes must contain exactly 138 records")
    for record in values:
        if not isinstance(record, PlanningLaneRecord):
            raise TypeError("required_lane_records must contain PlanningLaneRecord values")
        for value, label in (
            (record.topic_id, "topic ID"),
            (record.lane_name, "lane name"),
            (record.retrieval_source_type, "retrieval source type"),
            (record.scoring_source_type, "scoring source type"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"planning lane {label} must be non-empty text")
        if record.subnarrative_id is not None and (
            not isinstance(record.subnarrative_id, str)
            or not record.subnarrative_id
        ):
            raise ValueError("planning lane subnarrative ID must be null or non-empty text")
        for value, label in (
            (record.topic_order, "topic order"),
            (record.lane_order, "lane order"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"planning lane {label} must be a non-negative integer")
        for value, label in (
            (record.retrieval_query_sha256, "retrieval query SHA-256"),
            (record.scoring_query_sha256, "scoring query SHA-256"),
        ):
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"planning lane {label} is invalid")
    if len(set(values)) != len(values):
        raise ValueError("authenticated request lanes contain duplicate records")
    return values


def _validate_with_production_loader(topic: Topic, path: Path) -> ValidatedDecomposition:
    import trec_rag.competition_retrieval as production

    loader = getattr(production, "load_validated_decomposition", None)
    if not callable(loader):
        raise ValueError("production decomposition validator is unavailable")
    return loader(topic, path)


def _validator_identity() -> tuple[str, str]:
    import trec_rag.competition_retrieval as production

    module_path = getattr(production, "__file__", None)
    if not isinstance(module_path, str):
        raise ValueError("production validator module has no source identity")
    module_bytes = _read_regular(Path(module_path), "validator module")
    return _digest(module_bytes), "trec_rag.competition_retrieval:load_validated_decomposition"


def _lane_projector_identity() -> tuple[str, str]:
    import trec_rag.facet_retrieval_lanes as projector

    module_path = getattr(projector, "__file__", None)
    if not isinstance(module_path, str):
        raise ValueError("retrieval lane projector module has no source identity")
    module_bytes = _read_regular(Path(module_path), "retrieval lane projector module")
    identity = (
        "trec_rag.facet_retrieval_lanes:build_retrieval_lanes@"
        f"{FACET_RETRIEVAL_LANE_PROJECTOR_VERSION}"
    )
    return _digest(module_bytes), identity


def _decomposition_schema() -> str:
    import trec_rag.competition_retrieval as production

    schema = getattr(production, "SCHEMA", None)
    if not isinstance(schema, str) or not schema:
        raise ValueError("production decomposition schema is unavailable")
    return schema


def _preflight_existing_state(request: PlanningSeedRequest, prepared: tuple[_TopicSeed, ...]) -> None:
    for item in prepared:
        result_exists = _lexists(item.destination_path)
        manifest_exists = _lexists(item.manifest_path)
        if result_exists != manifest_exists:
            raise ValueError("preexisting result is unreceipted or manifest is partial")
        if result_exists:
            if _read_regular(item.destination_path, "destination result") != item.source_bytes:
                raise ValueError("existing destination result is contradictory")
            if _read_regular(item.manifest_path, "planning seed manifest") != item.manifest_bytes:
                raise ValueError("existing planning seed manifest is contradictory")
            _require_canonical_object(item.manifest_bytes, "planning seed manifest")


def _verify_prepared_files(prepared: tuple[_TopicSeed, ...]) -> None:
    """Recheck the complete source/destination closure before publication."""

    validator_sha256, validator_identity = _validator_identity()
    projector_sha256, projector_identity = _lane_projector_identity()
    decomposition_schema = _decomposition_schema()
    for item in prepared:
        if item.validated.source_sha256 != item.source_sha256:
            raise ValueError("prepared source validation digest changed")
        if (
            item.manifest["validator_module_sha256"] != validator_sha256
            or item.manifest["validator_identity"] != validator_identity
        ):
            raise ValueError("decomposition validator changed during planning seed import")
        if (
            item.manifest["lane_projector_module_sha256"] != projector_sha256
            or item.manifest["lane_projector_identity"] != projector_identity
        ):
            raise ValueError("retrieval lane projector changed during planning seed import")
        if item.manifest["decomposition_schema"] != decomposition_schema:
            raise ValueError("decomposition schema changed during planning seed import")
        if _read_stable_source(item.source_path)[0] != item.source_bytes:
            raise ValueError("source changed during copy")
        if _read_stable_regular(item.destination_path, "destination result")[0] != item.source_bytes:
            raise ValueError("destination result changed during copy")
        if _read_regular(item.manifest_path, "planning seed manifest") != item.manifest_bytes:
            raise ValueError("planning seed manifest changed during copy")
        manifest = _require_canonical_object(item.manifest_bytes, "planning seed manifest")
        if manifest != item.manifest:
            raise ValueError("planning seed manifest identity changed during copy")
        if (
            manifest["source_sha256"] != item.source_sha256
            or manifest["destination_sha256"] != item.source_sha256
            or manifest["source_byte_length"] != len(item.source_bytes)
            or manifest["destination_byte_length"] != len(item.source_bytes)
            or manifest["bytes_equal"] is not True
        ):
            raise ValueError("source/destination manifest identity changed during copy")


def _verify_destination_files(
    prepared: tuple[_TopicSeed, ...], destination_root: Path
) -> None:
    if not destination_root.is_dir():
        raise ValueError("destination root is missing")
    for item in prepared:
        if _read_stable_source(item.source_path)[0] != item.source_bytes:
            raise ValueError("source result bytes or digest changed during verification")
        destination_bytes, _destination_identity = _read_stable_regular(
            item.destination_path, "destination result"
        )
        if destination_bytes != item.source_bytes:
            raise ValueError("destination result bytes or digest changed")
        validated = _validate_with_production_loader(item.source.topic, item.destination_path)
        if validated.source_sha256 != item.source_sha256:
            raise ValueError("destination source SHA-256 does not match prepared source")
        destination_bytes_after_validation, _destination_identity = _read_stable_regular(
            item.destination_path, "destination result"
        )
        if destination_bytes_after_validation != item.source_bytes:
            raise ValueError("destination result changed during validation")
        if validated.source_sha256 != _digest(destination_bytes_after_validation):
            raise ValueError("destination validation digest does not match stable bytes")
        manifest_bytes = _read_regular(item.manifest_path, "planning seed manifest")
        if manifest_bytes != item.manifest_bytes:
            raise ValueError("planning seed manifest changed")
        _require_canonical_object(manifest_bytes, "planning seed manifest")


def _read_receipt(path: Path, expected_content: Mapping[str, object]) -> PlanningSeedReceipt:
    actual = _read_regular(path, "planning seed aggregate receipt")
    if actual != _canonical_json_bytes(expected_content):
        raise ValueError("aggregate receipt bytes changed")
    parsed = _require_canonical_object(actual, "planning seed aggregate receipt")
    content_digest = parsed.get("receipt_content_sha256")
    without_digest = dict(parsed)
    without_digest.pop("receipt_content_sha256", None)
    if content_digest != _digest_json(without_digest):
        raise ValueError("aggregate receipt content digest is invalid")
    return PlanningSeedReceipt(path, _digest(actual), parsed)


def _require_canonical_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest_json(value: object) -> str:
    return _digest(_canonical_json_bytes(value))


def _source_inventory_digest(records: list[dict[str, object]]) -> str:
    return _digest_json(
        {
            "schema_version": PLANNING_SEED_SOURCE_INVENTORY_SCHEMA,
            "files": records,
        }
    )


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


@contextmanager
def _planning_seed_lock(destination_root: Path):
    root = _absolute_root(destination_root, "destination root")
    _ensure_directory(root.parent)
    lock_path = root.parent / f".{root.name}.planning-seed.lock"
    _assert_no_symlinks(lock_path, "planning seed lock")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ValueError("planning seed lock cannot be opened safely") from exc
    try:
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("planning seed lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _absolute_root(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    absolute = path.absolute()
    _assert_no_symlinks(absolute, label)
    return absolute


def _ensure_source_root(root: Path) -> None:
    if not root.is_dir():
        raise ValueError("source root is missing")
    _assert_no_symlinks(root, "source root")


def _ensure_destination_root_parent(root: Path) -> None:
    parent = root.parent
    _assert_no_symlinks(parent, "destination root parent")


def _ensure_directory(path: Path) -> None:
    parts = path.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            created = False
            try:
                current.mkdir()
                created = True
            except FileExistsError:
                pass
            mode = current.lstat().st_mode
            if created:
                _fsync_directory(current.parent)
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError("path contains a symlink or non-directory")
    _fsync_directory(path)


def _checked_relative_path(value: str, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a logical relative path")
    if "\\" in value or "\x00" in value:
        raise ValueError(f"{label} contains unsafe path characters")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} must stay within its root")
    if relative.as_posix() != value:
        raise ValueError(f"{label} is not canonical")
    return relative


def _join_checked(root: Path, relative: PurePosixPath, label: str) -> Path:
    path = root.joinpath(*relative.parts)
    _assert_no_symlinks(path, label)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} is outside its root") from exc
    return path


def _assert_no_symlinks(path: Path, label: str) -> None:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} contains a symlink")


def _path_identity(path: Path) -> tuple[int, int, int, int, int]:
    _assert_no_symlinks(path, "source path")
    try:
        info = path.stat()
    except FileNotFoundError as exc:
        raise ValueError("source result is missing") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("source result is not a regular file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_stable_source(path: Path) -> tuple[bytes, tuple[int, int, int, int, int]]:
    return _read_stable_regular(path, "source result")


def _read_stable_regular(
    path: Path, label: str
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    _assert_no_symlinks(path, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity != after_identity or identity != _path_identity(path):
        raise ValueError(f"{label} changed during read")
    body = b"".join(chunks)
    if len(body) != before.st_size:
        raise ValueError(f"{label} changed during read")
    return body, identity


def _read_regular(path: Path, label: str) -> bytes:
    _assert_no_symlinks(path, label)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular file")
    return path.read_bytes()


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _publish_create_only(path: Path, body: bytes) -> None:
    _ensure_directory(path.parent)
    _assert_no_symlinks(path, "destination artifact")
    if _lexists(path):
        if _read_regular(path, "destination artifact") != body:
            raise ValueError("contradictory republish would overwrite destination")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read_regular(path, "destination artifact") != body:
                raise ValueError("contradictory republish would overwrite destination")
        else:
            _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("directory cannot be fsynced") from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
