"""Read bounded, sealed retrieval artifacts for the competition debug report.

This module deliberately contains no retrieval, reranking, canonicalization, or
hosted-model dependency.  It is a post-run reader for a small, explicit list of
already-written artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from trec_rag.facet_pilot_config import (
    FacetPilotConfig,
    load_facet_pilot_config,
    select_configured_topics,
)
from trec_rag.topics import Topic


_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_JSONL_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SubnarrativeReport:
    subnarrative_id: str
    text: str
    bm25_queries: tuple[str, ...]
    semantic_query_sha256: str
    bm25_query_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class SelectedDocumentReport:
    docid: str
    selection_rank: int
    selected_from_lane: str
    selected_from_lane_rank: int
    text: str
    text_sha256: str


@dataclass(frozen=True)
class NewDocumentReport:
    docid: str
    first_seen_lane: str
    memberships: tuple[str, ...]
    is_new: bool
    text_sha256: str | None
    excerpt: str | None


@dataclass(frozen=True)
class TopicReport:
    topic_id: str
    narrative: str
    narrative_sha256: str
    subnarratives: tuple[SubnarrativeReport, ...]
    selected_documents: tuple[SelectedDocumentReport, ...]
    new_documents: tuple[NewDocumentReport, ...]


@dataclass(frozen=True)
class DebugReportData:
    retrieval_config_path: Path
    output_dir: Path
    topics: tuple[TopicReport, ...]
    source_sha256s: Mapping[str, str]


def load_debug_report_data(
    retrieval_config_path: Path,
    *,
    rag_config_path: Path | None = None,
    topic_ids: Sequence[str] | None = None,
) -> DebugReportData:
    """Load the immutable, bounded data foundation for a completed export.

    ``rag_config_path`` is reserved for the later optional RAG projection.  It
    is intentionally rejected here so this first-stage loader cannot discover
    or read a broader set of artifacts by accident.
    """
    if rag_config_path is not None:
        raise ValueError("RAG artifacts are not supported by the bounded data loader")
    config = load_facet_pilot_config(retrieval_config_path)
    topics = select_configured_topics(
        config,
        topic_ids=() if topic_ids is None else tuple(topic_ids),
    )
    if not topics:
        raise ValueError("at least one configured topic is required")

    output_dir = _safe_directory(config.output_dir, "configured output directory")
    receipts: dict[str, str] = {}
    export_path = _safe_file(output_dir / "retrieval_export_manifest.json", output_dir)
    export = _read_json_object(export_path, "retrieval export manifest")
    receipts[_portable_label(output_dir, export_path)] = _sha256_file(export_path)
    exported_ids = _validate_export_manifest(export, config, topics)

    configured_by_id = {topic.id: topic for topic in topics}
    selected_topics = tuple(configured_by_id[topic_id] for topic_id in exported_ids)
    if topic_ids is not None and tuple(topic.id for topic in selected_topics) != tuple(
        topic.id for topic in topics
    ):
        raise ValueError("retrieval export topics differ from requested configured topics")

    reports = tuple(
        _load_topic_report(output_dir, topic, receipts) for topic in selected_topics
    )
    return DebugReportData(
        retrieval_config_path=Path(retrieval_config_path).resolve(),
        output_dir=output_dir,
        topics=reports,
        source_sha256s=MappingProxyType(dict(sorted(receipts.items()))),
    )


def _load_topic_report(
    output_dir: Path,
    topic: Topic,
    receipts: dict[str, str],
) -> TopicReport:
    topic_root = _safe_topic_root(output_dir, topic.id)
    decomposition_path = _safe_file(topic_root / "decomposition.json", output_dir)
    selection_path = _safe_file(topic_root / "scoring" / "selection.json", output_dir)
    selected_path = _safe_file(
        topic_root / "scoring" / "selected_documents.jsonl", output_dir
    )
    for path in (decomposition_path, selection_path, selected_path):
        receipts[_portable_label(output_dir, path)] = _sha256_file(path)

    decomposition = _read_json_object(decomposition_path, "decomposition")
    subnarratives = _decode_decomposition(decomposition, topic)
    selection = _read_json_object(selection_path, "selection checkpoint")
    selected = _decode_selected_documents(_read_jsonl(selected_path, "selected documents"), topic)
    union_rows = _decode_selection(selection, topic, selected)

    audit_hashes = _load_audit_hashes(topic_root, output_dir, receipts)
    selected_by_docid = {row.docid: row for row in selected}
    for row in selected:
        audit_hash = audit_hashes.get(row.docid)
        if audit_hash is not None and audit_hash != row.text_sha256:
            raise ValueError("selected document text hash differs from retrieval audit")

    new_documents = tuple(
        NewDocumentReport(
            docid=row["docid"],
            first_seen_lane=row["first_seen_lane"],
            memberships=row["memberships"],
            is_new="original" not in row["memberships"],
            text_sha256=audit_hashes.get(
                row["docid"],
                selected_by_docid[row["docid"]].text_sha256
                if row["docid"] in selected_by_docid
                else None,
            ),
            excerpt=selected_by_docid[row["docid"]].text
            if row["docid"] in selected_by_docid
            else None,
        )
        for row in union_rows
    )
    return TopicReport(
        topic_id=topic.id,
        narrative=topic.narrative,
        narrative_sha256=_text_sha256(topic.narrative, "official narrative"),
        subnarratives=subnarratives,
        selected_documents=selected,
        new_documents=new_documents,
    )


def _validate_export_manifest(
    value: Mapping[str, Any], config: FacetPilotConfig, topics: Sequence[Topic]
) -> tuple[str, ...]:
    if value.get("schema_version") != "retrieval_export_manifest_v2":
        raise ValueError("retrieval export manifest schema is invalid")
    if value.get("run_id") != config.run_id:
        raise ValueError("retrieval export manifest run identity differs from config")
    raw_ids = value.get("selected_topic_ids")
    if not isinstance(raw_ids, list) or not raw_ids or not all(
        _is_identifier(item) for item in raw_ids
    ) or len(set(raw_ids)) != len(raw_ids):
        raise ValueError("retrieval export manifest selected topic IDs are invalid")
    configured_ids = {topic.id for topic in topics}
    if not set(raw_ids) <= configured_ids:
        raise ValueError("retrieval export manifest contains an unconfigured topic")
    ordered = tuple(topic.id for topic in topics if topic.id in set(raw_ids))
    if tuple(raw_ids) != ordered:
        raise ValueError("retrieval export manifest topic order differs from official topics")
    return ordered


def _decode_decomposition(
    value: Mapping[str, Any], topic: Topic
) -> tuple[SubnarrativeReport, ...]:
    required = {
        "schema_version", "topic_id", "narrative", "narrative_sha256", "source_sha256",
        "queries", "plan", "subnarratives",
    }
    if set(value) != required or value.get("schema_version") != "facet_pilot_v2":
        raise ValueError("decomposition schema is invalid")
    if (
        value.get("topic_id") != topic.id
        or value.get("narrative") != topic.narrative
        or value.get("narrative_sha256") != _text_sha256(topic.narrative, "official narrative")
        or not _is_sha256(value.get("source_sha256"))
    ):
        raise ValueError("decomposition topic identity differs from official topic")
    rows = value.get("subnarratives")
    plan = value.get("plan")
    if not isinstance(rows, list) or not isinstance(plan, Mapping):
        raise ValueError("decomposition subnarratives are invalid")
    plan_rows = plan.get("subnarratives")
    if plan.get("topic_id") != topic.id or not isinstance(plan_rows, list):
        raise ValueError("decomposition plan topic identity is invalid")
    if len(rows) != len(plan_rows):
        raise ValueError("decomposition plan and subnarratives differ")
    result: list[SubnarrativeReport] = []
    seen_ids: set[str] = set()
    for index, (row, plan_row) in enumerate(zip(rows, plan_rows, strict=True), start=1):
        if not isinstance(row, Mapping) or not isinstance(plan_row, Mapping):
            raise ValueError("decomposition subnarrative is invalid")
        identifier = row.get("subnarrative_id")
        text = row.get("text")
        queries = row.get("bm25_queries")
        hashes = row.get("bm25_query_sha256s")
        semantic_hash = row.get("semantic_query_sha256")
        if (
            not _is_identifier(identifier)
            or identifier in seen_ids
            or not _is_text(text)
            or not _text_list(queries)
            or not isinstance(hashes, list)
            or tuple(hashes) != tuple(_text_sha256(query, "BM25 query") for query in queries)
            or semantic_hash != _text_sha256(text, "subnarrative")
            or plan_row.get("subnarrative") != text
            or plan_row.get("bm25_queries") != queries
        ):
            raise ValueError("decomposition subnarrative identity is invalid")
        seen_ids.add(identifier)
        result.append(
            SubnarrativeReport(identifier, text, tuple(queries), semantic_hash, tuple(hashes))
        )
    _validate_decomposition_queries(value.get("queries"), topic, result)
    return tuple(result)


def _validate_decomposition_queries(
    rows: object, topic: Topic, subnarratives: Sequence[SubnarrativeReport]
) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError("decomposition queries are invalid")
    original = rows[0]
    if not isinstance(original, Mapping) or (
        original.get("topic_id"), original.get("variant_name"), original.get("query_text"), original.get("source_type")
    ) != (topic.id, "original", topic.narrative, "original_topic"):
        raise ValueError("decomposition original query differs from official narrative")
    expected = sum((list(row.bm25_queries) for row in subnarratives), [])
    actual: list[str] = []
    for row in rows[1:]:
        if not isinstance(row, Mapping) or row.get("topic_id") != topic.id or not _is_text(row.get("query_text")):
            raise ValueError("decomposition query identity is invalid")
        actual.append(row["query_text"])
    if actual != expected:
        raise ValueError("decomposition queries differ from subnarrative plan")


def _decode_selected_documents(
    rows: Sequence[Mapping[str, Any]], topic: Topic
) -> tuple[SelectedDocumentReport, ...]:
    if not rows:
        raise ValueError("selected documents are empty")
    result: list[SelectedDocumentReport] = []
    seen: set[str] = set()
    for rank, row in enumerate(rows, start=1):
        docid = row.get("docid")
        text = row.get("text")
        text_hash = row.get("text_sha256")
        lane = row.get("selected_from_lane")
        lane_rank = row.get("selected_from_lane_rank")
        if (
            row.get("topic_id") != topic.id
            or not _is_docid(docid)
            or docid in seen
            or row.get("selection_rank") != rank
            or not _is_text(text)
            or text_hash != _text_sha256(text, "selected document")
            or not _is_identifier(lane)
            or not _positive_int(lane_rank)
        ):
            raise ValueError("selected document identity, rank, or text hash is invalid")
        seen.add(docid)
        result.append(SelectedDocumentReport(docid, rank, lane, lane_rank, text, text_hash))
    return tuple(result)


def _decode_selection(
    value: Mapping[str, Any], topic: Topic, selected: Sequence[SelectedDocumentReport]
) -> tuple[dict[str, Any], ...]:
    if value.get("schema_version") != "facet_pilot_selection_v2" or value.get("topic_id") != topic.id:
        raise ValueError("selection checkpoint topic identity is invalid")
    stored_order = value.get("selected_order")
    if not isinstance(stored_order, list) or stored_order != [row.docid for row in selected]:
        raise ValueError("selected documents differ from stored selection order")
    if len(set(stored_order)) != len(stored_order):
        raise ValueError("stored selection order contains duplicate documents")
    memberships = value.get("memberships")
    if not isinstance(memberships, list) or [row.get("docid") if isinstance(row, Mapping) else None for row in memberships] != stored_order:
        raise ValueError("selection memberships differ from stored order")
    selected_by_docid = {row.docid: row for row in selected}
    for membership in memberships:
        if not isinstance(membership, Mapping) or not _valid_membership(membership, selected_by_docid):
            raise ValueError("selection membership or lane rank is invalid")

    union = value.get("union_pool")
    if not isinstance(union, list) or not union:
        raise ValueError("selection union pool is invalid")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in union:
        if not isinstance(row, Mapping):
            raise ValueError("selection union row is invalid")
        docid, lane, memberships = row.get("docid"), row.get("first_seen_lane"), row.get("memberships")
        if (
            not _is_docid(docid)
            or docid in seen
            or not _is_identifier(lane)
            or not _text_list(memberships)
            or len(set(memberships)) != len(memberships)
            or lane not in memberships
        ):
            raise ValueError("selection union row identity is invalid")
        seen.add(docid)
        result.append({"docid": docid, "first_seen_lane": lane, "memberships": tuple(memberships)})
    if not set(stored_order) <= seen:
        raise ValueError("selected document is absent from selection union pool")
    return tuple(result)


def _valid_membership(value: Mapping[str, Any], selected: Mapping[str, SelectedDocumentReport]) -> bool:
    docid, lanes = value.get("docid"), value.get("lanes")
    if docid not in selected or not isinstance(lanes, list) or not lanes:
        return False
    seen: set[str] = set()
    origin_matches = False
    for lane in lanes:
        if not isinstance(lane, Mapping):
            return False
        name = lane.get("lane_name")
        if (
            not _is_identifier(name)
            or name in seen
            or not _positive_int(lane.get("aggregate_rank"))
            or not _finite_number(lane.get("aggregate_score"))
            or not _positive_int(lane.get("bm25_rank"))
            or not _finite_number(lane.get("bm25_score"))
        ):
            return False
        seen.add(name)
        document = selected[docid]
        if name == document.selected_from_lane and lane["aggregate_rank"] == document.selected_from_lane_rank:
            origin_matches = True
    return origin_matches


def _load_audit_hashes(topic_root: Path, output_dir: Path, receipts: dict[str, str]) -> Mapping[str, str]:
    path = topic_root / "retrieval" / "audit.json"
    if not path.exists():
        return MappingProxyType({})
    path = _safe_file(path, output_dir)
    receipts[_portable_label(output_dir, path)] = _sha256_file(path)
    value = _read_json_object(path, "retrieval audit")
    lanes = value.get("lanes")
    if not isinstance(lanes, list):
        raise ValueError("retrieval audit lanes are invalid")
    hashes: dict[str, str] = {}
    for lane in lanes:
        candidates = lane.get("candidates") if isinstance(lane, Mapping) else None
        if not isinstance(candidates, list):
            raise ValueError("retrieval audit candidates are invalid")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("retrieval audit candidate is invalid")
            docid, text_hash = candidate.get("docid"), candidate.get("text_sha256")
            if not _is_docid(docid) or not _is_sha256(text_hash):
                raise ValueError("retrieval audit candidate identity is invalid")
            previous = hashes.setdefault(docid, text_hash)
            if previous != text_hash:
                raise ValueError("retrieval audit has conflicting document text hashes")
    return MappingProxyType(hashes)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    raw = _read_bounded(path, _MAX_JSON_BYTES)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> tuple[dict[str, Any], ...]:
    raw = _read_bounded(path, _MAX_JSONL_BYTES)
    if not raw.endswith(b"\n"):
        raise ValueError(f"{label} must end with LF")
    rows: list[dict[str, Any]] = []
    for number, encoded in enumerate(raw.splitlines(), start=1):
        if not encoded:
            raise ValueError(f"{label}:{number}: blank JSONL row")
        try:
            value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{label}:{number}: invalid strict JSON object") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{number}: JSONL row must be an object")
        rows.append(value)
    return tuple(rows)


def _safe_directory(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is missing")
    return resolved


def _safe_topic_root(output_dir: Path, topic_id: str) -> Path:
    if not _is_identifier(topic_id):
        raise ValueError("topic ID is not safe for an artifact path")
    path = (output_dir / topic_id).resolve()
    if path.parent != output_dir or not path.is_dir():
        raise ValueError("sealed topic artifact directory is missing or unsafe")
    return path


def _safe_file(path: Path, output_dir: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError("artifact path escapes configured output directory") from exc
    if not resolved.is_file():
        raise ValueError(f"required artifact is missing or not a regular file: {path.name}")
    return resolved


def _read_bounded(path: Path, maximum: int) -> bytes:
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise ValueError(f"artifact size is outside the bounded reader limit: {path.name}")
    return path.read_bytes()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_label(output_dir: Path, path: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _text_list(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_text(item) for item in value)


def _is_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not any(char.isspace() for char in value) and "/" not in value and "\\" not in value


def _is_docid(value: object) -> bool:
    return _is_identifier(value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _text_sha256(value: str, label: str) -> str:
    if not _is_text(value):
        raise ValueError(f"{label} must be non-empty text")
    return sha256(value.encode("utf-8")).hexdigest()


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(value)
