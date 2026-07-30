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
import zipfile

from trec_rag.competition_rag import load_documents, load_trec_run
from trec_rag.evidence_store import decode_subnarrative_selection
from trec_rag.facet_pilot_config import (
    FacetPilotConfig,
    load_facet_pilot_config,
    select_configured_topics,
)
from trec_rag.topics import Topic


_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_JSONL_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCORE_FIELDS = {
    "topic_id", "lane_name", "semantic_query_sha256", "docid", "bm25_rank",
    "bm25_score", "aggregate_rank", "aggregate_score", "long_document_raw_logit",
    "weighted_passage_raw_logit", "within_document_span_support", "winning_passages",
    "score_representation", "text_sha256", "selection_rank", "subnarrative_id",
    "bm25_queries", "bm25_query_sha256s", "downstream_only",
}
_PASSAGE_FIELDS = {"chunk_index", "start_char", "end_char", "raw_logit", "weighted_rank"}
_CANONICAL_STATES = {"complete", "empty", "fallback_extractive"}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")


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
class WinningPassageReport:
    chunk_index: int
    start_char: int
    end_char: int
    raw_logit: float
    weighted_rank: int
    text: str


@dataclass(frozen=True)
class PassageRankingReport:
    subnarrative_id: str
    docid: str
    selection_rank: int
    bm25_rank: int
    bm25_score: float
    aggregate_rank: int
    aggregate_score: float
    long_document_raw_logit: float
    weighted_passage_raw_logit: float
    within_document_span_support: int
    score_representation: str
    winning_passages: tuple[WinningPassageReport, ...]


@dataclass(frozen=True)
class CanonicalEvidenceReport:
    candidate_nugget_id: str
    candidate_kind: str
    text: str
    text_sha256: str
    docid: str
    document_sha256: str
    cluster_id: str
    raw_logit: float | None = None


@dataclass(frozen=True)
class EvidenceClusterReport:
    subnarrative_id: str
    selected_budget: int
    cluster_id: str
    representative_candidate_nugget_id: str
    representative_text: str
    representative_raw_logit: float
    evidence: tuple[CanonicalEvidenceReport, ...]


@dataclass(frozen=True)
class CanonicalNuggetReport:
    subnarrative_id: str
    selected_budget: int
    state: str
    canonical_nugget_id: str
    nugget_kind: str
    claim_text: str
    evidence: tuple[CanonicalEvidenceReport, ...]
    maximum_claims: int
    maximum_supporting_documents: int


@dataclass(frozen=True)
class RetrievalDocumentReport:
    docid: str
    rank: int
    score: float
    selection_rank: int
    selected_from_lane: str
    selected_from_lane_rank: int
    text: str
    memberships: tuple[Mapping[str, Any], ...]
    subnarrative_scores: tuple[Mapping[str, Any], ...]
    canonical_nugget_ids: tuple[str, ...]
    source_seals: Mapping[str, str]


@dataclass(frozen=True)
class RetrievalOutputReport:
    selected_pool_depth: int
    final_supported_depth: int
    documents: tuple[RetrievalDocumentReport, ...]


@dataclass(frozen=True)
class TopicReport:
    topic_id: str
    narrative: str
    narrative_sha256: str
    subnarratives: tuple[SubnarrativeReport, ...]
    selected_documents: tuple[SelectedDocumentReport, ...]
    new_documents: tuple[NewDocumentReport, ...]
    passage_rankings: tuple[PassageRankingReport, ...]
    evidence_clusters: tuple[EvidenceClusterReport, ...]
    canonical_nuggets: tuple[CanonicalNuggetReport, ...]
    retrieval_output: RetrievalOutputReport


@dataclass(frozen=True)
class _RootRetrievalArtifacts:
    run_docids: Mapping[str, tuple[str, ...]]
    provenance: Mapping[tuple[str, str], Mapping[str, Any]]
    document_text: Mapping[tuple[str, str], str]
    topic_depths: Mapping[str, tuple[int, int]]


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
    receipts[_portable_label(output_dir, export_path)] = _sha256_file(
        export_path, _MAX_JSON_BYTES
    )
    exported_ids = _validate_export_manifest(export, config, topics)

    configured_by_id = {topic.id: topic for topic in topics}
    selected_topics = tuple(configured_by_id[topic_id] for topic_id in exported_ids)
    if topic_ids is not None and tuple(topic.id for topic in selected_topics) != tuple(
        topic.id for topic in topics
    ):
        raise ValueError("retrieval export topics differ from requested configured topics")

    retrieval_artifacts = _load_root_retrieval_artifacts(
        output_dir, export, selected_topics, receipts
    )
    reports = tuple(
        _load_topic_report(config, output_dir, topic, retrieval_artifacts, receipts)
        for topic in selected_topics
    )
    return DebugReportData(
        retrieval_config_path=Path(retrieval_config_path).resolve(),
        output_dir=output_dir,
        topics=reports,
        source_sha256s=MappingProxyType(dict(sorted(receipts.items()))),
    )


def _load_topic_report(
    config: FacetPilotConfig,
    output_dir: Path,
    topic: Topic,
    retrieval_artifacts: _RootRetrievalArtifacts,
    receipts: dict[str, str],
) -> TopicReport:
    topic_root = _safe_topic_root(output_dir, topic.id)
    decomposition_path = _safe_file(topic_root / "decomposition.json", output_dir)
    selection_path = _safe_file(topic_root / "scoring" / "selection.json", output_dir)
    selected_path = _safe_file(
        topic_root / "scoring" / "selected_documents.jsonl", output_dir
    )
    for path, maximum in (
        (decomposition_path, _MAX_JSON_BYTES),
        (selection_path, _MAX_JSON_BYTES),
        (selected_path, _MAX_JSONL_BYTES),
    ):
        receipts[_portable_label(output_dir, path)] = _sha256_file(path, maximum)

    decomposition = _read_json_object(decomposition_path, "decomposition")
    subnarratives = _decode_decomposition(decomposition, topic)
    selection = _read_json_object(selection_path, "selection checkpoint")
    selected = _decode_selected_documents(_read_jsonl(selected_path, "selected documents"), topic)
    union_rows = _decode_selection(selection, topic, selected)

    audit_hashes = _load_audit_hashes(
        topic_root,
        output_dir,
        receipts,
        topic,
        decomposition["narrative_sha256"],
        decomposition["source_sha256"],
        subnarratives,
        config.retrieval.candidate_depth_per_query,
    )
    selected_by_docid = {row.docid: row for row in selected}
    for row in selected:
        audit_hash = audit_hashes.get(row.docid)
        if audit_hash is not None and audit_hash != row.text_sha256:
            raise ValueError("selected document text hash differs from retrieval audit")

    passage_rankings = _load_passage_rankings(
        topic_root, output_dir, receipts, topic, subnarratives, selected
    )
    evidence_clusters, canonical_nuggets = _load_canonical_projection(
        config, topic_root, output_dir, receipts, topic, subnarratives, selected
    )
    retrieval_output = _decode_retrieval_output(
        topic, selected, canonical_nuggets, retrieval_artifacts
    )

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
        passage_rankings=passage_rankings,
        evidence_clusters=evidence_clusters,
        canonical_nuggets=canonical_nuggets,
        retrieval_output=retrieval_output,
    )


def _load_root_retrieval_artifacts(
    output_dir: Path,
    export: Mapping[str, Any],
    topics: Sequence[Topic],
    receipts: dict[str, str],
) -> _RootRetrievalArtifacts:
    paths = {
        "run": _safe_file(output_dir / "r_output_trec_rag_2026.tsv", output_dir),
        "provenance": _safe_file(output_dir / "retrieval_provenance.jsonl", output_dir),
        "archive": _safe_file(output_dir / "retrieval_with_text.jsonl.zip", output_dir),
    }
    limits = {"run": _MAX_JSONL_BYTES, "provenance": _MAX_JSONL_BYTES, "archive": _MAX_ARCHIVE_BYTES}
    for key, path in paths.items():
        receipts[_portable_label(output_dir, path)] = _sha256_file(path, limits[key])
    _validate_export_artifact_receipts(export, paths, receipts, output_dir)

    topic_ids = {topic.id for topic in topics}
    run_topic_ids, run_row_count = _trec_run_coverage(paths["run"])
    if run_topic_ids != topic_ids:
        raise ValueError("organizer run topic coverage differs from export manifest")
    if (
        type(export.get("official_row_count")) is not int
        or export["official_row_count"] != run_row_count
    ):
        raise ValueError("retrieval export official row count differs from organizer run")
    run_rows = load_trec_run(paths["run"], topic_ids, None)
    run_docids = {topic_id: tuple(docids) for topic_id, docids in run_rows.items()}
    provenance_rows = _read_jsonl(paths["provenance"], "retrieval provenance")
    provenance: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_pairs = {
        (topic_id, docid)
        for topic_id, docids in run_docids.items()
        for docid in docids
    }
    observed_order: dict[str, list[str]] = {topic.id: [] for topic in topics}
    for row in provenance_rows:
        topic_id, docid = row.get("topic_id"), row.get("docid")
        pair = (topic_id, docid)
        if (
            topic_id not in topic_ids
            or not _is_docid(docid)
            or pair in provenance
            or not _positive_int(row.get("rank"))
            or not _finite_number(row.get("score"))
        ):
            raise ValueError("retrieval provenance identity or rank is invalid")
        observed_order[topic_id].append(docid)
        provenance[pair] = MappingProxyType(dict(row))
    if set(provenance) != expected_pairs or any(
        tuple(observed_order[topic_id]) != run_docids[topic_id]
        for topic_id in observed_order
    ):
        raise ValueError("retrieval provenance coverage differs from organizer run")

    archive_pairs = _load_bounded_archive_pairs(paths["archive"], topics)
    if set(archive_pairs) != expected_pairs:
        raise ValueError("full-text archive coverage differs from organizer run")
    production_documents = load_documents(
        paths["archive"], "retrieval_with_text.jsonl", {docid for _, docid in expected_pairs}, 10_000_000
    )
    for (_topic_id, docid), text in archive_pairs.items():
        if production_documents.get(docid) != " ".join(text.split()):
            raise ValueError("full-text archive differs from production document projection")

    depths_value = export.get("topic_depths")
    if not isinstance(depths_value, Mapping) or set(depths_value) != topic_ids:
        raise ValueError("retrieval export topic depths are invalid")
    depths: dict[str, tuple[int, int]] = {}
    for topic_id in (topic.id for topic in topics):
        row = depths_value.get(topic_id)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"official", "candidate_pool"}
            or type(row.get("official")) is not int
            or row["official"] <= 0
            or type(row.get("candidate_pool")) is not int
            or row["candidate_pool"] < row["official"]
            or row["official"] != len(run_docids[topic_id])
        ):
            raise ValueError("retrieval export topic depth is invalid")
        depths[topic_id] = (row["candidate_pool"], row["official"])
    return _RootRetrievalArtifacts(
        MappingProxyType(run_docids),
        MappingProxyType(provenance),
        MappingProxyType(archive_pairs),
        MappingProxyType(depths),
    )


def _trec_run_coverage(path: Path) -> tuple[set[str], int]:
    try:
        lines = _read_bounded(path, _MAX_JSONL_BYTES).decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("organizer run is not valid UTF-8") from exc
    topic_ids: set[str] = set()
    for line in lines:
        fields = line.split()
        if len(fields) != 6 or not _is_identifier(fields[0]):
            raise ValueError("organizer run row identity is invalid")
        topic_ids.add(fields[0])
    return topic_ids, len(lines)


def _validate_export_artifact_receipts(
    export: Mapping[str, Any],
    paths: Mapping[str, Path],
    receipts: Mapping[str, str],
    output_dir: Path,
) -> None:
    artifacts = export.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("retrieval export artifact receipts are invalid")
    for path in paths.values():
        label = _portable_label(output_dir, path)
        row = artifacts.get(path.name)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"bytes", "sha256"}
            or type(row.get("bytes")) is not int
            or row["bytes"] != path.stat().st_size
            or row.get("sha256") != receipts[label]
        ):
            raise ValueError("retrieval export artifact receipt differs from stored artifact")


def _load_bounded_archive_pairs(
    path: Path, topics: Sequence[Topic]
) -> dict[tuple[str, str], str]:
    topic_by_id = {topic.id: topic for topic in topics}
    try:
        with zipfile.ZipFile(path) as archive:
            candidates = [name for name in archive.namelist() if name.lower().endswith((".jsonl", ".json"))]
            if candidates != ["retrieval_with_text.jsonl"]:
                raise ValueError("full-text archive member set is invalid")
            info = archive.getinfo(candidates[0])
            if info.file_size <= 0 or info.file_size > _MAX_ARCHIVE_BYTES:
                raise ValueError("full-text archive member exceeds bounded reader limit")
            with archive.open(info) as source:
                body = source.read(_MAX_ARCHIVE_BYTES + 1)
    except zipfile.BadZipFile as exc:
        raise ValueError("full-text archive is invalid") from exc
    if len(body) != info.file_size or len(body) > _MAX_ARCHIVE_BYTES or not body.endswith(b"\n"):
        raise ValueError("full-text archive member is invalid or oversized")
    result: dict[tuple[str, str], str] = {}
    for number, raw in enumerate(body.splitlines(), start=1):
        try:
            row = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"full-text archive:{number}: invalid strict JSON") from exc
        query = row.get("query") if isinstance(row, Mapping) else None
        rows = row.get("candidates") if isinstance(row, Mapping) else None
        topic_id = query.get("qid") if isinstance(query, Mapping) else None
        topic = topic_by_id.get(topic_id)
        if topic is None or query.get("text") != topic.narrative or not isinstance(rows, list):
            raise ValueError("full-text archive query identity is invalid")
        for rank, candidate in enumerate(rows, start=1):
            if not isinstance(candidate, Mapping):
                raise ValueError("full-text archive candidate is invalid")
            docid, text = candidate.get("docid"), candidate.get("doc")
            pair = (topic_id, docid)
            if (
                not _is_docid(docid)
                or not _is_text(text)
                or pair in result
                or candidate.get("rank") != rank
            ):
                raise ValueError("full-text archive candidate identity is invalid")
            result[pair] = text
    return result


def _load_passage_rankings(
    topic_root: Path,
    output_dir: Path,
    receipts: dict[str, str],
    topic: Topic,
    subnarratives: Sequence[SubnarrativeReport],
    selected: Sequence[SelectedDocumentReport],
) -> tuple[PassageRankingReport, ...]:
    path = _safe_file(topic_root / "scoring" / "selected_subnarrative_scores.jsonl", output_dir)
    receipts[_portable_label(output_dir, path)] = _sha256_file(path, _MAX_JSONL_BYTES)
    rows = _read_jsonl(path, "selected subnarrative scores")
    selected_by_id = {row.docid: row for row in selected}
    subnarrative_by_id = {row.subnarrative_id: row for row in subnarratives}
    grouped: dict[str, list[PassageRankingReport]] = {
        row.subnarrative_id: [] for row in subnarratives
    }
    seen: set[tuple[str, str]] = set()
    for value in rows:
        if set(value) != _SCORE_FIELDS:
            raise ValueError("selected subnarrative score schema is invalid")
        docid, subnarrative_id = value.get("docid"), value.get("subnarrative_id")
        document = selected_by_id.get(docid)
        subnarrative = subnarrative_by_id.get(subnarrative_id)
        pair = (docid, subnarrative_id)
        if (
            value.get("topic_id") != topic.id
            or document is None
            or subnarrative is None
            or pair in seen
            or value.get("selection_rank") != document.selection_rank
            or value.get("text_sha256") != document.text_sha256
            or value.get("lane_name") != f"subnarrative:{subnarrative_id}"
            or value.get("semantic_query_sha256") != subnarrative.semantic_query_sha256
            or value.get("bm25_queries") != list(subnarrative.bm25_queries)
            or value.get("bm25_query_sha256s") != list(subnarrative.bm25_query_sha256s)
            or value.get("downstream_only") is not True
            or value.get("score_representation") != "raw_logits"
        ):
            raise ValueError("selected subnarrative score identity is invalid")
        for field in ("bm25_rank", "aggregate_rank"):
            if not _positive_int(value.get(field)):
                raise ValueError("selected subnarrative score rank is invalid")
        for field in (
            "bm25_score", "aggregate_score", "long_document_raw_logit",
            "weighted_passage_raw_logit",
        ):
            if not _finite_number(value.get(field)):
                raise ValueError("selected subnarrative score logit or score is invalid")
        if type(value.get("within_document_span_support")) is not int or value["within_document_span_support"] < 0:
            raise ValueError("selected subnarrative span support is invalid")
        passages_value = value.get("winning_passages")
        if not isinstance(passages_value, list) or not passages_value:
            raise ValueError("selected subnarrative winning passages are invalid")
        passages: list[WinningPassageReport] = []
        for passage in passages_value:
            if not isinstance(passage, Mapping) or set(passage) != _PASSAGE_FIELDS:
                raise ValueError("selected subnarrative winning passage schema is invalid")
            start, end = passage.get("start_char"), passage.get("end_char")
            if type(start) is not int or type(end) is not int or not (0 <= start < end <= len(document.text)):
                raise ValueError("winning passage offsets are outside selected document")
            if (
                type(passage.get("chunk_index")) is not int
                or passage["chunk_index"] < 0
                or not _finite_number(passage.get("raw_logit"))
                or not _positive_int(passage.get("weighted_rank"))
            ):
                raise ValueError("selected subnarrative winning passage value is invalid")
            passages.append(
                WinningPassageReport(
                    passage["chunk_index"], start, end, float(passage["raw_logit"]),
                    passage["weighted_rank"], document.text[start:end],
                )
            )
        seen.add(pair)
        grouped[subnarrative_id].append(
            PassageRankingReport(
                subnarrative_id=subnarrative_id,
                docid=docid,
                selection_rank=document.selection_rank,
                bm25_rank=value["bm25_rank"],
                bm25_score=float(value["bm25_score"]),
                aggregate_rank=value["aggregate_rank"],
                aggregate_score=float(value["aggregate_score"]),
                long_document_raw_logit=float(value["long_document_raw_logit"]),
                weighted_passage_raw_logit=float(value["weighted_passage_raw_logit"]),
                within_document_span_support=value["within_document_span_support"],
                score_representation="raw_logits",
                winning_passages=tuple(passages),
            )
        )
    if seen != {
        (document.docid, subnarrative.subnarrative_id)
        for subnarrative in subnarratives
        for document in selected
    }:
        raise ValueError("selected subnarrative score matrix is incomplete")
    result: list[PassageRankingReport] = []
    for subnarrative in subnarratives:
        ranked = sorted(grouped[subnarrative.subnarrative_id], key=lambda row: row.aggregate_rank)
        if [row.aggregate_rank for row in ranked] != list(range(1, len(ranked) + 1)):
            raise ValueError("selected subnarrative aggregate ranks are not continuous")
        result.extend(ranked)
    return tuple(result)


def _load_canonical_projection(
    config: FacetPilotConfig,
    topic_root: Path,
    output_dir: Path,
    receipts: dict[str, str],
    topic: Topic,
    subnarratives: Sequence[SubnarrativeReport],
    selected: Sequence[SelectedDocumentReport],
) -> tuple[tuple[EvidenceClusterReport, ...], tuple[CanonicalNuggetReport, ...]]:
    selection_path = _safe_file(topic_root / "canonical" / "subnarrative-selections.jsonl", output_dir)
    nugget_path = _safe_file(topic_root / "canonical" / "canonical-nuggets.jsonl", output_dir)
    manifest_path = _safe_file(topic_root / "canonical" / "canonical-nugget-manifest.json", output_dir)
    for path in (selection_path, nugget_path, manifest_path):
        maximum = _MAX_JSON_BYTES if path == manifest_path else _MAX_JSONL_BYTES
        receipts[_portable_label(output_dir, path)] = _sha256_file(path, maximum)
    manifest = _read_json_object(manifest_path, "canonical nugget manifest")
    budget = config.nuggets.evidence_budget_per_subnarrative
    maximum_claims = config.nuggets.maximum_claims_per_subnarrative
    maximum_supporting = config.nuggets.maximum_supporting_documents_per_claim
    nugget_rows = _read_jsonl(nugget_path, "canonical nuggets")
    request_sha256s = manifest.get("request_sha256s")
    if (
        manifest.get("schema_version") != "canonical_nugget_manifest_v2"
        or manifest.get("selected_budget") != budget
        or manifest.get("max_canonical_claims") != maximum_claims
        or manifest.get("max_supporting_documents_per_claim") != maximum_supporting
        or manifest.get("result_count") != len(nugget_rows)
        or manifest.get("selection_count") != len(subnarratives)
        or manifest.get("canonical_nuggets_sha256") != receipts[_portable_label(output_dir, nugget_path)]
        or not isinstance(request_sha256s, list)
        or len(request_sha256s) != len(nugget_rows)
        or any(not _is_sha256(value) for value in request_sha256s)
        or len(set(request_sha256s)) != len(request_sha256s)
    ):
        raise ValueError("canonical nugget manifest differs from retrieval config or result")

    selected_by_id = {row.docid: row for row in selected}
    selection_rows = _read_jsonl(selection_path, "subnarrative selections")
    if len(selection_rows) != len(subnarratives):
        raise ValueError("subnarrative selection set is incomplete")
    clusters: list[EvidenceClusterReport] = []
    selected_evidence: dict[tuple[str, str], CanonicalEvidenceReport] = {}
    selected_cluster_ids: dict[str, set[str]] = {}
    for raw, expected in zip(selection_rows, subnarratives, strict=True):
        try:
            selection = decode_subnarrative_selection(
                json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("subnarrative selection is invalid") from exc
        context = selection.context
        if (
            context.topic_id != topic.id
            or context.official_narrative != topic.narrative
            or context.subnarrative_id != expected.subnarrative_id
            or context.subnarrative_text != expected.text
        ):
            raise ValueError("subnarrative selection identity differs from decomposition")
        snapshots = {snapshot.budget: snapshot for snapshot in selection.snapshots}
        snapshot = snapshots.get(budget)
        if snapshot is None:
            raise ValueError("subnarrative selection lacks configured budget")
        cluster_by_id = {cluster.cluster_id: cluster for cluster in selection.clusters}
        selected_cluster_ids[expected.subnarrative_id] = set(snapshot.cluster_ids)
        for cluster_id in snapshot.cluster_ids:
            cluster = cluster_by_id.get(cluster_id)
            if cluster is None:
                raise ValueError("subnarrative selection snapshot names unknown cluster")
            evidence: list[CanonicalEvidenceReport] = []
            for support in cluster.supports:
                document = selected_by_id.get(support.docid)
                if document is None or document.text_sha256 != support.document_sha256:
                    raise ValueError("selected evidence document differs from selected pool")
                report = CanonicalEvidenceReport(
                    support.candidate_nugget_id, support.candidate_kind, support.text,
                    _text_sha256(support.text, "selected evidence text"), support.docid,
                    support.document_sha256, cluster_id, support.raw_logit,
                )
                key = (expected.subnarrative_id, support.candidate_nugget_id)
                if key in selected_evidence:
                    raise ValueError("selected evidence candidate identity is duplicated")
                selected_evidence[key] = report
                evidence.append(report)
            clusters.append(
                EvidenceClusterReport(
                    expected.subnarrative_id, budget, cluster_id,
                    cluster.representative_candidate_nugget_id, cluster.representative_text,
                    cluster.representative_raw_logit, tuple(evidence),
                )
            )

    canonical: list[CanonicalNuggetReport] = []
    states: dict[str, int] = {}
    seen_nuggets: set[str] = set()
    for index, (raw, expected) in enumerate(
        zip(nugget_rows, subnarratives, strict=True)
    ):
        required = {
            "schema_version", "topic_id", "subnarrative_id", "selected_budget",
            "request_sha256", "state", "nuggets", "metadata", "error",
        }
        state, values = raw.get("state"), raw.get("nuggets")
        if raw.get("request_sha256") != request_sha256s[index]:
            raise ValueError("canonical nugget request identity differs from manifest")
        if (
            set(raw) != required
            or raw.get("schema_version") != "canonical_nugget_result_v1"
            or raw.get("topic_id") != topic.id
            or raw.get("subnarrative_id") != expected.subnarrative_id
            or raw.get("selected_budget") != budget
            or state not in _CANONICAL_STATES
            or not isinstance(values, list)
            or len(values) > maximum_claims
            or not isinstance(raw.get("metadata"), Mapping)
            or raw.get("error") is not None and not _is_text(raw.get("error"))
        ):
            raise ValueError("canonical nugget result identity or state is invalid")
        has_selected_evidence = any(
            subnarrative_id == expected.subnarrative_id
            for subnarrative_id, _candidate_id in selected_evidence
        )
        if state == "empty":
            if has_selected_evidence or values or dict(raw["metadata"]) or raw["error"] is not None:
                raise ValueError("canonical nugget empty state semantics are invalid")
            states[state] = states.get(state, 0) + 1
            continue
        if not has_selected_evidence:
            raise ValueError("canonical nugget non-empty state lacks selected evidence")
        if state == "complete":
            if raw["error"] is not None:
                raise ValueError("canonical nugget complete state semantics are invalid")
            expected_kind = "model_claim"
        else:
            error = raw["error"]
            if (
                type(error) is not str
                or not error
                or error != error.strip()
                or len(error) > 500
                or _CONTROL.search(error) is not None
            ):
                raise ValueError("canonical nugget fallback state semantics are invalid")
            if not values:
                raise ValueError("canonical fallback must retain one exact evidence claim")
            expected_kind = "extractive_fallback"
        states[state] = states.get(state, 0) + 1
        for nugget in values:
            if not isinstance(nugget, Mapping) or set(nugget) != {
                "canonical_nugget_id", "nugget_kind", "claim_text", "evidence"
            }:
                raise ValueError("canonical nugget schema is invalid")
            nugget_id, evidence_values = nugget.get("canonical_nugget_id"), nugget.get("evidence")
            if nugget.get("nugget_kind") != expected_kind:
                raise ValueError("canonical nugget state and kind are inconsistent")
            if (
                not _is_identifier(nugget_id)
                or nugget_id in seen_nuggets
                or not _is_text(nugget.get("claim_text"))
                or not isinstance(evidence_values, list)
                or not evidence_values
                or len(evidence_values) > maximum_supporting
            ):
                raise ValueError("canonical nugget identity or supporting-document cap is invalid")
            seen_nuggets.add(nugget_id)
            evidence_reports: list[CanonicalEvidenceReport] = []
            evidence_docids: set[str] = set()
            for evidence in evidence_values:
                report = _decode_canonical_evidence(
                    evidence, expected.subnarrative_id, selected_by_id,
                    selected_evidence, selected_cluster_ids[expected.subnarrative_id],
                )
                if report.docid in evidence_docids:
                    raise ValueError("canonical nugget repeats a supporting document")
                evidence_docids.add(report.docid)
                evidence_reports.append(report)
            if state == "fallback_extractive" and (
                len(evidence_reports) != 1
                or nugget["claim_text"] != evidence_reports[0].text
            ):
                raise ValueError("canonical nugget fallback state or kind is invalid")
            canonical.append(
                CanonicalNuggetReport(
                    expected.subnarrative_id, budget, state, nugget_id,
                    nugget["nugget_kind"], nugget["claim_text"], tuple(evidence_reports),
                    maximum_claims, maximum_supporting,
                )
            )
    if manifest.get("state_counts") != dict(sorted(states.items())):
        raise ValueError("canonical nugget manifest state counts differ from results")
    return tuple(clusters), tuple(canonical)


def _decode_canonical_evidence(
    value: object,
    subnarrative_id: str,
    selected_by_id: Mapping[str, SelectedDocumentReport],
    selected_evidence: Mapping[tuple[str, str], CanonicalEvidenceReport],
    selected_cluster_ids: set[str],
) -> CanonicalEvidenceReport:
    fields = {
        "candidate_nugget_id", "candidate_kind", "text", "text_sha256", "docid",
        "document_sha256", "cluster_id",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("canonical evidence schema is invalid")
    source = selected_evidence.get((subnarrative_id, value.get("candidate_nugget_id")))
    document = selected_by_id.get(value.get("docid"))
    if (
        source is None
        or document is None
        or value.get("cluster_id") not in selected_cluster_ids
        or source.cluster_id != value.get("cluster_id")
        or source.candidate_kind != value.get("candidate_kind")
        or source.text != value.get("text")
        or source.docid != value.get("docid")
        or source.document_sha256 != value.get("document_sha256")
        or value.get("text_sha256") != _text_sha256(source.text, "canonical evidence text")
        or document.text_sha256 != value.get("document_sha256")
    ):
        raise ValueError("canonical evidence differs from selected cluster or document")
    return CanonicalEvidenceReport(
        source.candidate_nugget_id, source.candidate_kind, source.text,
        value["text_sha256"], source.docid, source.document_sha256,
        source.cluster_id, source.raw_logit,
    )


def _decode_retrieval_output(
    topic: Topic,
    selected: Sequence[SelectedDocumentReport],
    canonical: Sequence[CanonicalNuggetReport],
    artifacts: _RootRetrievalArtifacts,
) -> RetrievalOutputReport:
    selected_by_id = {row.docid: row for row in selected}
    selected_depth, final_depth = artifacts.topic_depths[topic.id]
    if selected_depth != len(selected):
        raise ValueError("retrieval export selected-pool depth differs from selected documents")
    canonical_by_docid: dict[str, set[str]] = {}
    for nugget in canonical:
        for evidence in nugget.evidence:
            canonical_by_docid.setdefault(evidence.docid, set()).add(nugget.canonical_nugget_id)
    run_docids = artifacts.run_docids[topic.id]
    if final_depth != len(run_docids) or set(run_docids) != set(canonical_by_docid):
        raise ValueError("organizer run differs from canonical supported-document projection")
    documents: list[RetrievalDocumentReport] = []
    for rank, docid in enumerate(run_docids, start=1):
        pair = (topic.id, docid)
        provenance = artifacts.provenance[pair]
        document = selected_by_id.get(docid)
        text = artifacts.document_text[pair]
        memberships = provenance.get("memberships")
        subnarrative_scores = provenance.get("subnarrative_scores")
        nuggets = provenance.get("nuggets")
        seals = provenance.get("source_seals")
        if (
            document is None
            or text != document.text
            or provenance.get("rank") != rank
            or provenance.get("selection_rank") != document.selection_rank
            or provenance.get("selected_from_lane") != document.selected_from_lane
            or provenance.get("selected_from_lane_rank") != document.selected_from_lane_rank
            or not isinstance(memberships, list)
            or not memberships
            or any(not isinstance(row, Mapping) for row in memberships)
            or not isinstance(subnarrative_scores, list)
            or any(not isinstance(row, Mapping) for row in subnarrative_scores)
            or not isinstance(nuggets, list)
            or any(not isinstance(row, Mapping) for row in nuggets)
            or not isinstance(seals, Mapping)
            or set(seals) != {"scoring_manifest_sha256", "canonical_manifest_sha256"}
            or any(not _is_sha256(value) for value in seals.values())
        ):
            raise ValueError("retrieval provenance differs from sealed selected document")
        provenance_nugget_ids = tuple(
            row.get("canonical_nugget_id") for row in nuggets
        )
        if (
            any(not _is_identifier(value) for value in provenance_nugget_ids)
            or set(provenance_nugget_ids) != canonical_by_docid[docid]
        ):
            raise ValueError("retrieval provenance canonical nugget join is invalid")
        documents.append(
            RetrievalDocumentReport(
                docid=docid,
                rank=rank,
                score=float(provenance["score"]),
                selection_rank=document.selection_rank,
                selected_from_lane=document.selected_from_lane,
                selected_from_lane_rank=document.selected_from_lane_rank,
                text=text,
                memberships=tuple(MappingProxyType(dict(row)) for row in memberships),
                subnarrative_scores=tuple(
                    MappingProxyType(dict(row)) for row in subnarrative_scores
                ),
                canonical_nugget_ids=tuple(sorted(canonical_by_docid[docid])),
                source_seals=MappingProxyType(dict(seals)),
            )
        )
    return RetrievalOutputReport(selected_depth, final_depth, tuple(documents))


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
    for row, plan_row in zip(rows, plan_rows, strict=True):
        if not isinstance(row, Mapping) or not isinstance(plan_row, Mapping):
            raise ValueError("decomposition subnarrative is invalid")
        identifier = row.get("subnarrative_id")
        text = row.get("text")
        queries = row.get("bm25_queries")
        hashes = row.get("bm25_query_sha256s")
        semantic_hash = row.get("semantic_query_sha256")
        if (
            row.get("topic_id") != topic.id
            or not _is_identifier(identifier)
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
    if not isinstance(original, Mapping) or set(original) != {
        "topic_id", "variant_name", "query_text", "source_type"
    } or (
        original.get("topic_id"), original.get("variant_name"),
        original.get("query_text"), original.get("source_type")
    ) != (topic.id, "original", topic.narrative, "original_topic"):
        raise ValueError("decomposition original query differs from official narrative")
    expected = [
        (topic.id, f"facet:{subnarrative.subnarrative_id}:q{query_index}", query,
         "generated_subnarrative_bm25")
        for subnarrative in subnarratives
        for query_index, query in enumerate(subnarrative.bm25_queries, start=1)
    ]
    actual: list[tuple[object, object, object, object]] = []
    for row in rows[1:]:
        if not isinstance(row, Mapping) or set(row) != {
            "topic_id", "variant_name", "query_text", "source_type"
        }:
            raise ValueError("decomposition query identity is invalid")
        actual.append(
            (row.get("topic_id"), row.get("variant_name"), row.get("query_text"), row.get("source_type"))
        )
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
    membership_lanes: dict[str, frozenset[str]] = {}
    for membership in memberships:
        if not isinstance(membership, Mapping) or not _valid_membership(membership, selected_by_docid):
            raise ValueError("selection membership or lane rank is invalid")
        membership_lanes[membership["docid"]] = frozenset(
            lane["lane_name"] for lane in membership["lanes"]
        )

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
    for row in result:
        if row["docid"] in membership_lanes and set(row["memberships"]) != membership_lanes[row["docid"]]:
            raise ValueError("selection union membership differs from selected provenance")
    _validate_selection_trace(value.get("trace"), selected)
    return tuple(result)


def _validate_selection_trace(
    value: object, selected: Sequence[SelectedDocumentReport]
) -> None:
    if not isinstance(value, list):
        raise ValueError("selection trace is invalid")
    selected_events: list[tuple[object, object, object, object]] = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
            "slot", "docid", "lane_name", "lane_rank", "lane_exhausted", "action"
        } or not _positive_int(row.get("slot")) or not _is_identifier(row.get("lane_name")) or type(row.get("lane_exhausted")) is not bool:
            raise ValueError("selection trace is invalid")
        action = row.get("action")
        if action not in {"selected", "duplicate_skip", "exhausted"}:
            raise ValueError("selection trace action is invalid")
        if action == "exhausted":
            if row.get("docid") is not None or row.get("lane_rank") is not None or row["lane_exhausted"] is not True:
                raise ValueError("selection trace exhaustion is invalid")
            continue
        if not _is_docid(row.get("docid")) or not _positive_int(row.get("lane_rank")):
            raise ValueError("selection trace candidate provenance is invalid")
        if action == "selected":
            selected_events.append(
                (row["slot"], row["docid"], row["lane_name"], row["lane_rank"])
            )
    expected = [
        (row.selection_rank, row.docid, row.selected_from_lane, row.selected_from_lane_rank)
        for row in selected
    ]
    if selected_events != expected:
        raise ValueError("selection trace differs from selected rank provenance")


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


def _load_audit_hashes(
    topic_root: Path,
    output_dir: Path,
    receipts: dict[str, str],
    topic: Topic,
    narrative_sha256: object,
    decomposition_source_sha256: object,
    subnarratives: Sequence[SubnarrativeReport],
    requested_depth: int,
) -> Mapping[str, str]:
    path = topic_root / "retrieval" / "audit.json"
    if not path.exists():
        return MappingProxyType({})
    path = _safe_file(path, output_dir)
    receipts[_portable_label(output_dir, path)] = _sha256_file(path, _MAX_JSON_BYTES)
    value = _read_json_object(path, "retrieval audit")
    lanes = value.get("lanes")
    expected_lanes = [
        ("original", None, narrative_sha256, narrative_sha256),
        *[
            (
                f"facet:{row.subnarrative_id}:text",
                row.subnarrative_id,
                row.semantic_query_sha256,
                row.semantic_query_sha256,
            )
            for row in subnarratives
        ],
    ]
    required = {
        "schema_version", "topic_id", "narrative_sha256",
        "decomposition_source_sha256", "requested_depth", "lanes",
    }
    if (
        set(value) != required
        or value.get("schema_version") != "facet_pilot_v2"
        or value.get("topic_id") != topic.id
        or value.get("narrative_sha256") != narrative_sha256
        or value.get("decomposition_source_sha256") != decomposition_source_sha256
        or value.get("requested_depth") != requested_depth
        or not isinstance(lanes, list)
        or len(lanes) != len(expected_lanes)
    ):
        raise ValueError("retrieval audit identity or lane set is invalid")
    hashes: dict[str, str] = {}
    for lane, expected in zip(lanes, expected_lanes, strict=True):
        if not isinstance(lane, Mapping) or set(lane) != {
            "lane_name", "subnarrative_id", "bm25_query_sha256",
            "semantic_query_sha256", "returned_count", "retained_count", "candidates",
        } or (
            lane.get("lane_name"), lane.get("subnarrative_id"),
            lane.get("bm25_query_sha256"), lane.get("semantic_query_sha256"),
        ) != expected or type(lane.get("returned_count")) is not int or lane["returned_count"] < 0 or type(lane.get("retained_count")) is not int or lane["retained_count"] < 0 or lane["retained_count"] != min(lane["returned_count"], requested_depth) or not isinstance(lane.get("candidates"), list) or len(lane["candidates"]) != lane["retained_count"]:
            raise ValueError("retrieval audit lane identity or counts are invalid")
        seen: set[str] = set()
        for rank, candidate in enumerate(lane["candidates"], start=1):
            if not isinstance(candidate, Mapping) or set(candidate) != {
                "docid", "bm25_rank", "bm25_score", "text_sha256"
            }:
                raise ValueError("retrieval audit candidate is invalid")
            docid, text_hash = candidate.get("docid"), candidate.get("text_sha256")
            if (
                not _is_docid(docid)
                or docid in seen
                or type(candidate.get("bm25_rank")) is not int
                or candidate["bm25_rank"] != rank
                or not _finite_number(candidate.get("bm25_score"))
                or not _is_sha256(text_hash)
            ):
                raise ValueError("retrieval audit candidate identity is invalid")
            seen.add(docid)
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
    _bounded_size(path, maximum)
    return path.read_bytes()


def _bounded_size(path: Path, maximum: int) -> int:
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise ValueError(f"artifact size is outside the bounded reader limit: {path.name}")
    return size


def _sha256_file(path: Path, maximum: int) -> str:
    _bounded_size(path, maximum)
    digest = sha256()
    total = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            total += len(chunk)
            if total > maximum:
                raise ValueError(
                    f"artifact size is outside the bounded reader limit: {path.name}"
                )
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
