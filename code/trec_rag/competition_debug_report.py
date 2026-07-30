"""Read bounded, sealed retrieval artifacts for the competition debug report.

This module deliberately contains no retrieval, reranking, canonicalization, or
hosted-model dependency.  It is a post-run reader for a small, explicit list of
already-written artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
import html
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, BinaryIO, Iterator, Mapping, Sequence
import zipfile

from trec_rag.competition_rag import (
    load_queries,
    load_rag_generation_config,
    load_trec_run,
    select_queries,
    validate_submission_record,
)
from trec_rag.evidence_store import decode_subnarrative_selection
from trec_rag.facet_pilot_config import (
    FacetPilotConfig,
    load_facet_pilot_config,
    select_configured_topics,
)
from trec_rag.repo_env import find_repo_root
from trec_rag.topics import Topic


_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_JSONL_BYTES = 16 * 1024 * 1024
_MAX_STREAM_RECORD_BYTES = 8 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCORE_FIELDS = {
    "topic_id", "lane_name", "semantic_query_sha256", "docid", "bm25_rank",
    "bm25_score", "aggregate_rank", "aggregate_score", "long_document_raw_logit",
    "weighted_passage_raw_logit", "within_document_span_support", "winning_passages",
    "score_representation", "text_sha256", "selection_rank", "subnarrative_id",
    "bm25_queries", "bm25_query_sha256s", "downstream_only",
}
_LANE_SCORE_FIELDS = {
    "topic_id", "lane_name", "bm25_query_sha256", "semantic_query_sha256",
    "docid", "bm25_rank", "bm25_score", "aggregate_rank", "aggregate_score",
    "long_document_raw_logit", "weighted_passage_raw_logit",
    "within_document_span_support", "winning_passages", "score_representation",
    "text_sha256",
}
_PASSAGE_FIELDS = {"chunk_index", "start_char", "end_char", "raw_logit", "weighted_rank"}
_CANONICAL_STATES = {"complete", "empty", "fallback_extractive"}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_REPORT_SCHEMA_VERSION = "competition_debug_report_v1"
_DOCUMENT_EXCERPT_CHARACTERS = 500


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
    memberships: tuple[Mapping[str, Any], ...]
    is_original_member: bool
    selection_rationale: str


@dataclass(frozen=True)
class LaneScoreProvenanceReport:
    lane_name: str
    aggregate_rank: int
    aggregate_score: float
    bm25_rank: int
    bm25_score: float
    text_sha256: str


@dataclass(frozen=True)
class NewDocumentReport:
    docid: str
    first_seen_lane: str
    memberships: tuple[str, ...]
    is_new: bool
    text_sha256: str | None
    excerpt: str | None
    lane_provenance: tuple[LaneScoreProvenanceReport, ...]


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
class CanonicalResultReport:
    subnarrative_id: str
    selected_budget: int
    state: str
    maximum_claims: int
    maximum_supporting_documents: int
    nuggets: tuple[CanonicalNuggetReport, ...]


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
    stage: str


@dataclass(frozen=True)
class RetrievalOutputReport:
    selected_pool_depth: int
    final_supported_depth: int
    documents: tuple[RetrievalDocumentReport, ...]


@dataclass(frozen=True)
class RagAnswerItemReport:
    text: str
    citations: tuple[int, ...]
    citation_docids: tuple[str, ...]


@dataclass(frozen=True)
class RagOutputReport:
    references: tuple[str, ...]
    answer_items: tuple[RagAnswerItemReport, ...]
    run_id: str
    run_desc: str
    provider: str
    model: str
    word_count: int
    output_sha256: str


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
    canonical_results: tuple[CanonicalResultReport, ...]
    retrieval_output: RetrievalOutputReport
    original_only_fallback: bool = False
    rag_output: RagOutputReport | None = None


@dataclass(frozen=True)
class _BestStoredPassage:
    subnarrative_id: str
    subnarrative_text: str
    aggregate_rank: int
    passage: WinningPassageReport


def _selected_document_reason(document: SelectedDocumentReport) -> str:
    """Describe validated selection provenance without inventing semantics."""
    membership = "Original-narrative" if document.is_original_member else "Facet-only"
    lane = document.selected_from_lane
    if lane == "original":
        lane_description = "the original lane"
    elif lane.startswith("facet:") and lane.endswith(":text"):
        lane_description = f"the {lane[len('facet:'):-len(':text')]} facet lane"
    else:
        lane_description = f"the {lane} lane"
    return (
        f"{membership} document selected at position {document.selection_rank} from "
        f"{lane_description} at rank {document.selected_from_lane_rank}."
    )


def _best_stored_passage(
    topic: TopicReport, docid: str
) -> _BestStoredPassage | None:
    """Choose by within-subnarrative rank, then sealed decomposition order."""
    subnarrative_order = {
        subnarrative.subnarrative_id: index
        for index, subnarrative in enumerate(topic.subnarratives)
    }
    subnarratives = {
        subnarrative.subnarrative_id: subnarrative
        for subnarrative in topic.subnarratives
    }
    candidates = [ranking for ranking in topic.passage_rankings if ranking.docid == docid]
    if not candidates:
        return None
    ranking = min(
        candidates,
        key=lambda candidate: (
            candidate.aggregate_rank,
            subnarrative_order[candidate.subnarrative_id],
        ),
    )
    passage = min(ranking.winning_passages, key=lambda candidate: candidate.weighted_rank)
    subnarrative = subnarratives[ranking.subnarrative_id]
    return _BestStoredPassage(
        subnarrative_id=subnarrative.subnarrative_id,
        subnarrative_text=subnarrative.text,
        aggregate_rank=ranking.aggregate_rank,
        passage=passage,
    )


@dataclass(frozen=True)
class _RootRetrievalArtifacts:
    run_docids: Mapping[str, tuple[str, ...]]
    provenance: Mapping[tuple[str, str], Mapping[str, Any]]
    document_text: Mapping[tuple[str, str], str]
    document_stage: Mapping[tuple[str, str], str]
    topic_depths: Mapping[str, tuple[int, int]]


@dataclass(frozen=True)
class DebugReportData:
    retrieval_config_path: Path
    rag_config_path: Path | None
    output_dir: Path
    topics: tuple[TopicReport, ...]
    source_sha256s: Mapping[str, str]


@dataclass(frozen=True)
class DebugReportReceipt:
    schema_version: str
    output_path: Path
    topic_ids: tuple[str, ...]
    rag_included: bool
    source_sha256s: Mapping[str, str]


def load_debug_report_data(
    retrieval_config_path: Path,
    *,
    rag_config_path: Path | None = None,
    topic_ids: Sequence[str] | None = None,
) -> DebugReportData:
    """Load immutable, bounded retrieval and optional validated RAG artifacts."""
    config = load_facet_pilot_config(retrieval_config_path)
    configured_topics = select_configured_topics(config)
    if not configured_topics:
        raise ValueError("at least one configured topic is required")

    output_dir = _safe_directory(config.output_dir, "configured output directory")
    receipts: dict[str, str] = {}
    export_path = _safe_file(output_dir / "retrieval_export_manifest.json", output_dir)
    export = _read_json_object(export_path, "retrieval export manifest")
    receipts[_portable_label(output_dir, export_path)] = _sha256_file(
        export_path, _MAX_JSON_BYTES
    )
    exported_ids = _validate_export_manifest(export, config, configured_topics)

    configured_by_id = {topic.id: topic for topic in configured_topics}
    exported_topics = tuple(configured_by_id[topic_id] for topic_id in exported_ids)
    if topic_ids is None:
        selected_topics = exported_topics
    else:
        requested_topics = select_configured_topics(config, topic_ids=tuple(topic_ids))
        exported_set = set(exported_ids)
        if any(topic.id not in exported_set for topic in requested_topics):
            raise ValueError("requested topic is absent from the retrieval export")
        selected_topics = requested_topics

    retrieval_artifacts = _load_root_retrieval_artifacts(
        output_dir,
        export,
        exported_topics,
        receipts,
        retained_topic_ids={topic.id for topic in selected_topics},
    )
    reports = tuple(
        _load_topic_report(config, output_dir, topic, retrieval_artifacts, receipts)
        for topic in selected_topics
    )
    data = DebugReportData(
        retrieval_config_path=Path(retrieval_config_path).resolve(),
        rag_config_path=None,
        output_dir=output_dir,
        topics=reports,
        source_sha256s=MappingProxyType(dict(sorted(receipts.items()))),
    )
    if rag_config_path is not None:
        data = _attach_rag_outputs(data, Path(rag_config_path), exported_topics)
    return data


def _attach_rag_outputs(
    data: DebugReportData,
    rag_config_path: Path,
    exported_topics: Sequence[Topic],
) -> DebugReportData:
    config = load_rag_generation_config(rag_config_path)
    expected_run = (data.output_dir / "r_output_trec_rag_2026.tsv").resolve()
    expected_documents = (data.output_dir / "retrieval_with_text.jsonl.zip").resolve()
    if config.run_path.resolve() != expected_run:
        raise ValueError("RAG retrieval run path is incompatible with the retrieval export")
    if config.documents_path.resolve() != expected_documents:
        raise ValueError("RAG retrieval documents path is incompatible with the retrieval export")

    configured_queries = select_queries(load_queries(config.queries_path), config.topic_ids)
    configured_ids = tuple(topic_id for topic_id, _ in configured_queries)
    if configured_ids != tuple(topic.id for topic in exported_topics):
        raise ValueError("RAG topic order or coverage is incompatible with the retrieval export")
    retrieval_narratives = {topic.id: topic.narrative for topic in exported_topics}
    for topic_id, narrative in configured_queries:
        if narrative != retrieval_narratives[topic_id]:
            raise ValueError("RAG topic narrative is incompatible with the retrieval export")

    allowed_by_topic = load_trec_run(
        config.run_path,
        set(configured_ids),
        config.top_k,
    )
    output_parent = _safe_directory(config.output_path.parent, "RAG output directory")
    output_path = _safe_file(config.output_path, output_parent)
    rows = _read_jsonl(output_path, "RAG output")
    if len(rows) != len(configured_queries):
        raise ValueError("RAG output topic coverage is incompatible with the RAG config")
    output_sha256 = _sha256_file(output_path, _MAX_JSONL_BYTES)

    rag_by_topic: dict[str, RagOutputReport] = {}
    for row, (topic_id, narrative) in zip(rows, configured_queries, strict=True):
        validate_submission_record(
            row,
            topic_id=topic_id,
            narrative=narrative,
            allowed_docids=allowed_by_topic[topic_id],
            team_id=config.team_id,
            run_id=config.run_id,
            run_desc=config.run_desc,
        )
        references = tuple(row["references"])
        answer_items = tuple(
            RagAnswerItemReport(
                text=item["text"],
                citations=tuple(item["citations"]),
                citation_docids=tuple(references[index] for index in item["citations"]),
            )
            for item in row["answer"]
        )
        metadata = row["metadata"]
        rag_by_topic[topic_id] = RagOutputReport(
            references=references,
            answer_items=answer_items,
            run_id=metadata["run_id"],
            run_desc=metadata["run_desc"],
            provider=config.provider,
            model=config.model,
            word_count=sum(len(item.text.split()) for item in answer_items),
            output_sha256=output_sha256,
        )

    sources = dict(data.source_sha256s)
    sources["rag/rag_output_trec_rag_2026.jsonl"] = output_sha256
    return replace(
        data,
        rag_config_path=rag_config_path.resolve(),
        topics=tuple(
            replace(topic, rag_output=rag_by_topic[topic.topic_id])
            for topic in data.topics
        ),
        source_sha256s=MappingProxyType(dict(sorted(sources.items()))),
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
    subnarratives, original_only_fallback = _decode_decomposition(
        decomposition, topic
    )
    selection = _read_json_object(selection_path, "selection checkpoint")
    selected = _decode_selected_documents(_read_jsonl(selected_path, "selected documents"), topic)
    union_rows, selected = _decode_selection(selection, topic, selected)
    if original_only_fallback and (
        any(row["memberships"] != ("original",) for row in union_rows)
        or any(
            row.selected_from_lane != "original"
            or not row.is_original_member
            or tuple(item.get("lane_name") for item in row.memberships)
            != ("original",)
            for row in selected
        )
    ):
        raise ValueError("original-only fallback selection provenance is invalid")

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
    lane_provenance = _load_lane_score_provenance(
        topic_root,
        output_dir,
        receipts,
        topic,
        subnarratives,
        union_rows,
        retrieval_artifacts,
    )
    for (docid, _lane_name), lane_score in lane_provenance.items():
        audit_hash = audit_hashes.get(docid)
        if audit_hash is not None and audit_hash != lane_score.text_sha256:
            raise ValueError("lane score text hash differs from retrieval audit")
    for row in selected:
        audit_hash = audit_hashes.get(row.docid)
        if audit_hash is not None and audit_hash != row.text_sha256:
            raise ValueError("selected document text hash differs from retrieval audit")
        for membership in row.memberships:
            source = lane_provenance.get(
                (row.docid, membership.get("lane_name"))
            )
            if source is None or (
                source.aggregate_rank,
                source.aggregate_score,
                source.bm25_rank,
                source.bm25_score,
                source.text_sha256,
            ) != (
                membership.get("aggregate_rank"),
                membership.get("aggregate_score"),
                membership.get("bm25_rank"),
                membership.get("bm25_score"),
                row.text_sha256,
            ):
                raise ValueError(
                    "selected document membership differs from sealed lane score"
                )

    passage_rankings = _load_passage_rankings(
        topic_root,
        output_dir,
        receipts,
        topic,
        subnarratives,
        selected,
        allow_empty=original_only_fallback,
    )
    evidence_clusters, canonical_nuggets, canonical_results = _load_canonical_projection(
        config,
        topic_root,
        output_dir,
        receipts,
        topic,
        subnarratives,
        selected,
        original_only_fallback=original_only_fallback,
    )
    retrieval_output = _decode_retrieval_output(
        topic,
        selected,
        canonical_nuggets,
        retrieval_artifacts,
        original_only_fallback=original_only_fallback,
    )

    new_documents = () if original_only_fallback else tuple(
        NewDocumentReport(
            docid=row["docid"],
            first_seen_lane=row["first_seen_lane"],
            memberships=row["memberships"],
            is_new="original" not in row["memberships"],
            text_sha256=lane_provenance[
                (row["docid"], row["memberships"][0])
            ].text_sha256,
            excerpt=selected_by_docid[row["docid"]].text
            if row["docid"] in selected_by_docid
            else None,
            lane_provenance=tuple(
                lane_provenance[(row["docid"], lane)]
                for lane in row["memberships"]
            ),
        )
        for row in union_rows
        if "original" not in row["memberships"]
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
        canonical_results=canonical_results,
        retrieval_output=retrieval_output,
        original_only_fallback=original_only_fallback,
    )


def _load_root_retrieval_artifacts(
    output_dir: Path,
    export: Mapping[str, Any],
    topics: Sequence[Topic],
    receipts: dict[str, str],
    *,
    retained_topic_ids: set[str] | None = None,
) -> _RootRetrievalArtifacts:
    retained_ids = (
        {topic.id for topic in topics}
        if retained_topic_ids is None
        else set(retained_topic_ids)
    )
    topic_ids = {topic.id for topic in topics}
    if not retained_ids <= topic_ids:
        raise ValueError("retained root artifact topic is absent from export")
    paths = {
        "run": _safe_file(output_dir / "r_output_trec_rag_2026.tsv", output_dir),
        "provenance": _safe_file(output_dir / "retrieval_provenance.jsonl", output_dir),
        "archive": _safe_file(output_dir / "retrieval_with_text.jsonl.zip", output_dir),
    }
    artifact_receipts = {
        key: _export_artifact_receipt(export, path)
        for key, path in paths.items()
    }

    with _open_receipted_file(
        paths["run"], artifact_receipts["run"]
    ) as (run_source, run_digest):
        receipts[_portable_label(output_dir, paths["run"])] = run_digest
        run_docids, run_row_count = _load_trec_run_streaming(
            run_source, paths["run"], topics
        )
    if set(run_docids) != topic_ids:
        raise ValueError("organizer run topic coverage differs from export manifest")
    if (
        type(export.get("official_row_count")) is not int
        or export["official_row_count"] != run_row_count
    ):
        raise ValueError("retrieval export official row count differs from organizer run")
    provenance: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_pairs = {
        (topic_id, docid)
        for topic_id, docids in run_docids.items()
        for docid in docids
    }
    observed_order: dict[str, list[str]] = {topic.id: [] for topic in topics}
    seen_provenance: set[tuple[str, str]] = set()
    with _open_receipted_file(
        paths["provenance"], artifact_receipts["provenance"]
    ) as (provenance_source, provenance_digest):
        receipts[_portable_label(output_dir, paths["provenance"])] = provenance_digest
        for row in _iter_strict_jsonl(
            provenance_source,
            "retrieval provenance",
            path=paths["provenance"],
        ):
            topic_id, docid = row.get("topic_id"), row.get("docid")
            pair = (topic_id, docid)
            if (
                topic_id not in topic_ids
                or not _is_docid(docid)
                or pair in seen_provenance
                or not _positive_int(row.get("rank"))
                or not _finite_number(row.get("score"))
            ):
                raise ValueError("retrieval provenance identity or rank is invalid")
            seen_provenance.add(pair)
            observed_order[topic_id].append(docid)
            if topic_id in retained_ids:
                provenance[pair] = MappingProxyType(dict(row))
    if seen_provenance != expected_pairs or any(
        tuple(observed_order[topic_id]) != run_docids[topic_id]
        for topic_id in observed_order
    ):
        raise ValueError("retrieval provenance coverage differs from organizer run")

    with _open_receipted_file(
        paths["archive"], artifact_receipts["archive"]
    ) as (archive_source, archive_digest):
        receipts[_portable_label(output_dir, paths["archive"])] = archive_digest
        archive_pairs, archive_stages = _load_streaming_archive_pairs(
            archive_source,
            paths["archive"],
            topics,
            run_docids,
            retained_ids,
        )

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
        MappingProxyType(archive_stages),
        MappingProxyType(depths),
    )


def _load_trec_run_streaming(
    source: BinaryIO, path: Path, topics: Sequence[Topic]
) -> tuple[dict[str, tuple[str, ...]], int]:
    grouped: dict[str, list[tuple[int, float, str]]] = {
        topic.id: [] for topic in topics
    }
    run_tag: str | None = None
    row_count = 0
    for row_count, encoded in enumerate(
        _iter_bounded_lines(source, "organizer run", path=path), start=1
    ):
        try:
            line = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("organizer run is not valid UTF-8") from exc
        fields = line.split()
        if len(fields) != 6:
            raise ValueError("organizer run row must contain six TREC fields")
        topic_id, q0, docid, raw_rank, raw_score, tag = fields
        if not _is_identifier(topic_id) or q0 != "Q0" or not _is_docid(docid):
            raise ValueError("organizer run row identity is invalid")
        try:
            rank = int(raw_rank)
            score = float(raw_score)
        except ValueError as exc:
            raise ValueError("organizer run rank or score is invalid") from exc
        if rank <= 0 or not math.isfinite(score) or not tag:
            raise ValueError("organizer run rank, score, or tag is invalid")
        if run_tag is None:
            run_tag = tag
        elif run_tag != tag:
            raise ValueError("organizer run tag is not stable")
        grouped.setdefault(topic_id, []).append((rank, score, docid))
    if row_count == 0:
        raise ValueError("organizer run contains no rows")

    result: dict[str, tuple[str, ...]] = {}
    for topic_id, rows in grouped.items():
        if not rows:
            continue
        ranks = [rank for rank, _score, _docid in rows]
        scores = [score for _rank, score, _docid in rows]
        docids = [docid for _rank, _score, docid in rows]
        if (
            len(ranks) != len(set(ranks))
            or ranks != sorted(ranks)
            or ranks[0] != 1
            or len(docids) != len(set(docids))
            or any(previous < current for previous, current in zip(scores, scores[1:]))
        ):
            raise ValueError("organizer run rank, score, or document order is invalid")
        result[topic_id] = tuple(docids)
    return result, row_count


def _export_artifact_receipt(
    export: Mapping[str, Any], path: Path
) -> Mapping[str, Any]:
    artifacts = export.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("retrieval export artifact receipts are invalid")
    row = artifacts.get(path.name)
    if (
        not isinstance(row, Mapping)
        or set(row) != {"bytes", "sha256"}
        or type(row.get("bytes")) is not int
        or row["bytes"] <= 0
        or not _is_sha256(row.get("sha256"))
    ):
        raise ValueError("retrieval export artifact receipt differs from stored artifact")
    return MappingProxyType(dict(row))


def _load_streaming_archive_pairs(
    source: BinaryIO,
    path: Path,
    topics: Sequence[Topic],
    run_docids: Mapping[str, tuple[str, ...]],
    retained_topic_ids: set[str],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    topic_by_id = {topic.id: topic for topic in topics}
    expected_pairs = {
        (topic_id, docid)
        for topic_id, docids in run_docids.items()
        for docid in docids
    }
    result: dict[tuple[str, str], str] = {}
    stages: dict[tuple[str, str], str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    seen_topics: set[str] = set()
    normalized_hash_by_docid: dict[str, str] = {}
    try:
        with zipfile.ZipFile(source) as archive:
            candidates = [name for name in archive.namelist() if name.lower().endswith((".jsonl", ".json"))]
            if candidates != ["retrieval_with_text.jsonl"]:
                raise ValueError("full-text archive member set is invalid")
            info = archive.getinfo(candidates[0])
            if (
                info.file_size <= 0
                or info.file_size > len(topics) * _MAX_STREAM_RECORD_BYTES
            ):
                raise ValueError("full-text archive member exceeds bounded record coverage")
            with archive.open(info) as source:
                for row in _iter_strict_jsonl(
                    source, "full-text archive", path=path
                ):
                    query = row.get("query")
                    rows = row.get("candidates")
                    topic_id = query.get("qid") if isinstance(query, Mapping) else None
                    topic = topic_by_id.get(topic_id)
                    if (
                        topic is None
                        or topic_id in seen_topics
                        or query.get("text") != topic.narrative
                        or not isinstance(rows, list)
                    ):
                        raise ValueError("full-text archive query identity is invalid")
                    seen_topics.add(topic_id)
                    observed_docids: list[str] = []
                    for rank, candidate in enumerate(rows, start=1):
                        if not isinstance(candidate, Mapping):
                            raise ValueError("full-text archive candidate is invalid")
                        docid, text = candidate.get("docid"), candidate.get("doc")
                        stage = candidate.get("stage")
                        pair = (topic_id, docid)
                        if (
                            not _is_docid(docid)
                            or not _is_text(text)
                            or pair in seen_pairs
                            or candidate.get("rank") != rank
                            or stage not in {"canonical_supported", "original_only_fallback"}
                        ):
                            raise ValueError("full-text archive candidate identity is invalid")
                        normalized_digest = _text_sha256(
                            " ".join(text.split()), "full-text archive document"
                        )
                        previous_digest = normalized_hash_by_docid.setdefault(
                            docid, normalized_digest
                        )
                        if previous_digest != normalized_digest:
                            raise ValueError(
                                "full-text archive has conflicting duplicate document text"
                            )
                        seen_pairs.add(pair)
                        observed_docids.append(docid)
                        if topic_id in retained_topic_ids:
                            result[pair] = text
                            stages[pair] = stage
                    if tuple(observed_docids) != run_docids[topic_id]:
                        raise ValueError(
                            "full-text archive document order differs from organizer run"
                        )
    except zipfile.BadZipFile as exc:
        raise ValueError("full-text archive is invalid") from exc
    if seen_topics != set(topic_by_id) or seen_pairs != expected_pairs:
        raise ValueError("full-text archive coverage differs from organizer run")
    return result, stages


def _load_passage_rankings(
    topic_root: Path,
    output_dir: Path,
    receipts: dict[str, str],
    topic: Topic,
    subnarratives: Sequence[SubnarrativeReport],
    selected: Sequence[SelectedDocumentReport],
    *,
    allow_empty: bool = False,
) -> tuple[PassageRankingReport, ...]:
    path = _safe_file(topic_root / "scoring" / "selected_subnarrative_scores.jsonl", output_dir)
    receipts[_portable_label(output_dir, path)] = _sha256_file(
        path, _MAX_JSONL_BYTES, allow_empty=allow_empty
    )
    rows = _read_jsonl(
        path, "selected subnarrative scores", allow_empty=allow_empty
    )
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
    *,
    original_only_fallback: bool = False,
) -> tuple[
    tuple[EvidenceClusterReport, ...],
    tuple[CanonicalNuggetReport, ...],
    tuple[CanonicalResultReport, ...],
]:
    selection_path = _safe_file(topic_root / "canonical" / "subnarrative-selections.jsonl", output_dir)
    nugget_path = _safe_file(topic_root / "canonical" / "canonical-nuggets.jsonl", output_dir)
    manifest_path = _safe_file(topic_root / "canonical" / "canonical-nugget-manifest.json", output_dir)
    for path in (selection_path, nugget_path, manifest_path):
        maximum = _MAX_JSON_BYTES if path == manifest_path else _MAX_JSONL_BYTES
        receipts[_portable_label(output_dir, path)] = _sha256_file(
            path,
            maximum,
            allow_empty=original_only_fallback and path != manifest_path,
        )
    manifest = _read_json_object(manifest_path, "canonical nugget manifest")
    budget = config.nuggets.evidence_budget_per_subnarrative
    maximum_claims = config.nuggets.maximum_claims_per_subnarrative
    maximum_supporting = config.nuggets.maximum_supporting_documents_per_claim
    nugget_rows = _read_jsonl(
        nugget_path,
        "canonical nuggets",
        allow_empty=original_only_fallback,
    )
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
    if original_only_fallback and any(
        manifest.get(field) != 0
        for field in (
            "hosted_llm_calls",
            "validated_cache_hits",
            "raw_cache_writes",
            "validated_cache_writes",
        )
    ):
        raise ValueError("original-only fallback canonical work is not empty")

    selected_by_id = {row.docid: row for row in selected}
    selection_rows = _read_jsonl(
        selection_path,
        "subnarrative selections",
        allow_empty=original_only_fallback,
    )
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
    canonical_results: list[CanonicalResultReport] = []
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
            canonical_results.append(
                CanonicalResultReport(
                    expected.subnarrative_id,
                    budget,
                    state,
                    maximum_claims,
                    maximum_supporting,
                    (),
                )
            )
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
        result_nuggets: list[CanonicalNuggetReport] = []
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
            ):
                raise ValueError("canonical nugget identity or supporting-document cap is invalid")
            seen_nuggets.add(nugget_id)
            evidence_reports: list[CanonicalEvidenceReport] = []
            evidence_docids: set[str] = set()
            evidence_candidate_ids: set[str] = set()
            for evidence in evidence_values:
                report = _decode_canonical_evidence(
                    evidence, expected.subnarrative_id, selected_by_id,
                    selected_evidence, selected_cluster_ids[expected.subnarrative_id],
                )
                if report.candidate_nugget_id in evidence_candidate_ids:
                    raise ValueError("canonical nugget repeats a selected evidence candidate")
                evidence_candidate_ids.add(report.candidate_nugget_id)
                evidence_docids.add(report.docid)
                evidence_reports.append(report)
            if len(evidence_docids) > maximum_supporting:
                raise ValueError("canonical nugget exceeds supporting-document cap")
            if state == "fallback_extractive" and (
                len(evidence_reports) != 1
                or nugget["claim_text"] != evidence_reports[0].text
            ):
                raise ValueError("canonical nugget fallback state or kind is invalid")
            report = CanonicalNuggetReport(
                    expected.subnarrative_id, budget, state, nugget_id,
                    nugget["nugget_kind"], nugget["claim_text"], tuple(evidence_reports),
                    maximum_claims, maximum_supporting,
            )
            canonical.append(report)
            result_nuggets.append(report)
        canonical_results.append(
            CanonicalResultReport(
                expected.subnarrative_id,
                budget,
                state,
                maximum_claims,
                maximum_supporting,
                tuple(result_nuggets),
            )
        )
    if manifest.get("state_counts") != dict(sorted(states.items())):
        raise ValueError("canonical nugget manifest state counts differ from results")
    return tuple(clusters), tuple(canonical), tuple(canonical_results)


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
    *,
    original_only_fallback: bool = False,
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
    if original_only_fallback:
        if canonical or tuple(run_docids) != tuple(row.docid for row in selected):
            raise ValueError(
                "organizer run differs from original-only selected-document projection"
            )
    elif set(run_docids) != set(canonical_by_docid):
        raise ValueError(
            "organizer run differs from canonical supported-document projection"
        )
    if final_depth != len(run_docids):
        raise ValueError("retrieval export final depth differs from organizer run")
    documents: list[RetrievalDocumentReport] = []
    for rank, docid in enumerate(run_docids, start=1):
        pair = (topic.id, docid)
        provenance = artifacts.provenance[pair]
        document = selected_by_id.get(docid)
        text = artifacts.document_text[pair]
        archive_stage = artifacts.document_stage[pair]
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
        if tuple(dict(row) for row in memberships) != tuple(
            dict(row) for row in document.memberships
        ):
            raise ValueError(
                "retrieval provenance membership differs from sealed selection"
            )
        expected_stage = (
            "original_only_fallback"
            if original_only_fallback
            else "canonical_supported"
        )
        if (
            archive_stage != expected_stage
            or (
                provenance.get("stage") != expected_stage
                if original_only_fallback
                else provenance.get("stage") is not None
            )
            or (
                original_only_fallback
                and (subnarrative_scores != [] or nuggets != [])
            )
        ):
            raise ValueError("retrieval fallback stage provenance is invalid")
        provenance_nugget_ids = tuple(
            row.get("canonical_nugget_id") for row in nuggets
        )
        if (
            any(not _is_identifier(value) for value in provenance_nugget_ids)
            or set(provenance_nugget_ids) != canonical_by_docid.get(docid, set())
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
                canonical_nugget_ids=tuple(
                    sorted(canonical_by_docid.get(docid, set()))
                ),
                source_seals=MappingProxyType(dict(seals)),
                stage=expected_stage,
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
) -> tuple[tuple[SubnarrativeReport, ...], bool]:
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
    if not isinstance(rows, list):
        raise ValueError("decomposition subnarratives are invalid")
    if plan is None:
        if rows:
            raise ValueError("original-only fallback subnarratives are invalid")
        _validate_decomposition_queries(value.get("queries"), topic, ())
        return (), True
    if not isinstance(plan, Mapping):
        raise ValueError("decomposition plan is invalid")
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
    return tuple(result), False


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
        result.append(
            SelectedDocumentReport(
                docid,
                rank,
                lane,
                lane_rank,
                text,
                text_hash,
                (),
                False,
                "",
            )
        )
    return tuple(result)


def _decode_selection(
    value: Mapping[str, Any], topic: Topic, selected: Sequence[SelectedDocumentReport]
) -> tuple[tuple[dict[str, Any], ...], tuple[SelectedDocumentReport, ...]]:
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
    membership_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for membership in memberships:
        if not isinstance(membership, Mapping) or not _valid_membership(membership, selected_by_docid):
            raise ValueError("selection membership or lane rank is invalid")
        membership_lanes[membership["docid"]] = frozenset(
            lane["lane_name"] for lane in membership["lanes"]
        )
        membership_rows[membership["docid"]] = tuple(
            MappingProxyType(dict(lane)) for lane in membership["lanes"]
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
    rationales = _validate_selection_trace(value.get("trace"), selected)
    explained = tuple(
        replace(
            document,
            memberships=membership_rows[document.docid],
            is_original_member="original" in membership_lanes[document.docid],
            selection_rationale=rationales[document.docid],
        )
        for document in selected
    )
    return tuple(result), explained


def _validate_selection_trace(
    value: object, selected: Sequence[SelectedDocumentReport]
) -> Mapping[str, str]:
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
    return MappingProxyType(
        {
            row.docid: (
                f"Selected at slot {row.selection_rank} from "
                f"{row.selected_from_lane} rank {row.selected_from_lane_rank}."
            )
            for row in selected
        }
    )


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


def _load_lane_score_provenance(
    topic_root: Path,
    output_dir: Path,
    receipts: dict[str, str],
    topic: Topic,
    subnarratives: Sequence[SubnarrativeReport],
    union_rows: Sequence[Mapping[str, Any]],
    retrieval_artifacts: _RootRetrievalArtifacts,
) -> Mapping[tuple[str, str], LaneScoreProvenanceReport]:
    manifest_path = _safe_file(topic_root / "scoring" / "complete.json", output_dir)
    sealed_manifest_digests: set[str] = set()
    for (topic_id, _docid), row in retrieval_artifacts.provenance.items():
        if topic_id != topic.id:
            continue
        seals = row.get("source_seals")
        digest = (
            seals.get("scoring_manifest_sha256")
            if isinstance(seals, Mapping)
            else None
        )
        if not _is_sha256(digest):
            raise ValueError("retrieval provenance scoring manifest seal is invalid")
        sealed_manifest_digests.add(digest)
    if len(sealed_manifest_digests) != 1:
        raise ValueError("scoring manifest differs from retrieval provenance seal")

    manifest_receipt = {
        "bytes": _bounded_size(manifest_path, _MAX_JSON_BYTES),
        "sha256": next(iter(sealed_manifest_digests)),
    }
    try:
        with _open_receipted_file(
            manifest_path,
            manifest_receipt,
            label="scoring manifest",
        ) as (manifest_source, manifest_digest):
            receipts[_portable_label(output_dir, manifest_path)] = manifest_digest
            manifest = _decode_json_object(
                manifest_source.read(_MAX_JSON_BYTES + 1),
                "scoring manifest",
            )
    except ValueError as exc:
        if str(exc) == "scoring manifest receipt differs from stored artifact":
            raise ValueError(
                "scoring manifest differs from retrieval provenance seal"
            ) from exc
        raise
    artifact_rows = manifest.get("artifacts")
    if (
        manifest.get("schema_version") != "facet_pilot_v2"
        or manifest.get("phase") != "score"
        or manifest.get("topic_id") != topic.id
        or not isinstance(artifact_rows, list)
    ):
        raise ValueError("scoring manifest identity or artifacts are invalid")
    matches = [
        row
        for row in artifact_rows
        if isinstance(row, Mapping)
        and row.get("relative_path") == "scoring/lane_scores.jsonl"
    ]
    if len(matches) != 1:
        raise ValueError("scoring manifest lane-score receipt is missing or duplicated")
    artifact_receipt = matches[0]
    if (
        set(artifact_receipt) != {"relative_path", "bytes", "sha256"}
        or type(artifact_receipt.get("bytes")) is not int
        or artifact_receipt["bytes"] <= 0
        or not _is_sha256(artifact_receipt.get("sha256"))
    ):
        raise ValueError("scoring manifest lane-score receipt is invalid")

    lane_path = _safe_file(topic_root / "scoring" / "lane_scores.jsonl", output_dir)
    expected_keys = {
        (row["docid"], lane)
        for row in union_rows
        for lane in row["memberships"]
    }
    expected_query_hashes = {
        "original": _text_sha256(topic.narrative, "official narrative"),
        **{
            f"facet:{row.subnarrative_id}:text": row.semantic_query_sha256
            for row in subnarratives
        },
    }
    result: dict[tuple[str, str], LaneScoreProvenanceReport] = {}
    text_hash_by_docid: dict[str, str] = {}
    with _open_receipted_file(
        lane_path,
        artifact_receipt,
        label="scoring lane-score artifact",
    ) as (source, lane_digest):
        receipts[_portable_label(output_dir, lane_path)] = lane_digest
        for value in _iter_strict_jsonl(source, "lane scores", path=lane_path):
            docid, lane_name = value.get("docid"), value.get("lane_name")
            key = (docid, lane_name)
            expected_query_hash = expected_query_hashes.get(lane_name)
            if (
                set(value) != _LANE_SCORE_FIELDS
                or value.get("topic_id") != topic.id
                or key not in expected_keys
                or key in result
                or expected_query_hash is None
                or value.get("bm25_query_sha256") != expected_query_hash
                or value.get("semantic_query_sha256") != expected_query_hash
                or value.get("score_representation") != "raw_logits"
                or not _is_sha256(value.get("text_sha256"))
            ):
                raise ValueError("lane score identity or query provenance is invalid")
            for field in ("bm25_rank", "aggregate_rank"):
                if not _positive_int(value.get(field)):
                    raise ValueError("lane score rank is invalid")
            for field in (
                "bm25_score",
                "aggregate_score",
                "long_document_raw_logit",
                "weighted_passage_raw_logit",
            ):
                if not _finite_number(value.get(field)):
                    raise ValueError("lane score value is not finite")
            support = value.get("within_document_span_support")
            passages = value.get("winning_passages")
            if (
                type(support) is not int
                or support < 0
                or not isinstance(passages, list)
                or not passages
            ):
                raise ValueError("lane score span or passage provenance is invalid")
            for passage in passages:
                if (
                    not isinstance(passage, Mapping)
                    or set(passage) != _PASSAGE_FIELDS
                    or type(passage.get("chunk_index")) is not int
                    or passage["chunk_index"] < 0
                    or type(passage.get("start_char")) is not int
                    or type(passage.get("end_char")) is not int
                    or not (0 <= passage["start_char"] < passage["end_char"])
                    or not _positive_int(passage.get("weighted_rank"))
                    or not _finite_number(passage.get("raw_logit"))
                ):
                    raise ValueError("lane score winning passage is invalid")
            text_hash = value["text_sha256"]
            if text_hash_by_docid.setdefault(docid, text_hash) != text_hash:
                raise ValueError("lane scores disagree on document text hash")
            result[key] = LaneScoreProvenanceReport(
                lane_name=lane_name,
                aggregate_rank=value["aggregate_rank"],
                aggregate_score=float(value["aggregate_score"]),
                bm25_rank=value["bm25_rank"],
                bm25_score=float(value["bm25_score"]),
                text_sha256=text_hash,
            )
    if set(result) != expected_keys:
        raise ValueError("lane score coverage differs from selection union pool")
    return MappingProxyType(result)


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


def _iter_bounded_lines(
    source: BinaryIO,
    label: str,
    *,
    path: Path,
) -> Iterator[bytes]:
    number = 0
    while True:
        encoded = source.readline(_MAX_STREAM_RECORD_BYTES + 1)
        if not encoded:
            return
        number += 1
        if len(encoded) > _MAX_STREAM_RECORD_BYTES:
            raise ValueError(
                f"{label}:{number}: record exceeds bounded streaming limit: {path.name}"
            )
        if not encoded.endswith(b"\n"):
            raise ValueError(f"{label}:{number}: record must end with LF")
        if encoded == b"\n":
            raise ValueError(f"{label}:{number}: blank row")
        yield encoded


def _iter_strict_jsonl(
    source: BinaryIO,
    label: str,
    *,
    path: Path,
) -> Iterator[dict[str, Any]]:
    for number, encoded in enumerate(
        _iter_bounded_lines(source, label, path=path), start=1
    ):
        try:
            value = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_no_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{label}:{number}: invalid strict JSON object") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{number}: JSONL row must be an object")
        yield value


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    raw = _read_bounded(path, _MAX_JSON_BYTES)
    return _decode_json_object(raw, label)


def _decode_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(
    path: Path, label: str, *, allow_empty: bool = False
) -> tuple[dict[str, Any], ...]:
    raw = _read_bounded(path, _MAX_JSONL_BYTES, allow_empty=allow_empty)
    if not raw and allow_empty:
        return ()
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


def _read_bounded(path: Path, maximum: int, *, allow_empty: bool = False) -> bytes:
    with path.open("rb") as source:
        raw = source.read(maximum + 1)
    if len(raw) > maximum or (not raw and not allow_empty):
        raise ValueError(
            f"artifact size is outside the bounded reader limit: {path.name}"
        )
    return raw


def _bounded_size(path: Path, maximum: int, *, allow_empty: bool = False) -> int:
    size = path.stat().st_size
    if size < 0 or (size == 0 and not allow_empty) or size > maximum:
        raise ValueError(f"artifact size is outside the bounded reader limit: {path.name}")
    return size


def _sha256_file(path: Path, maximum: int, *, allow_empty: bool = False) -> str:
    _bounded_size(path, maximum, allow_empty=allow_empty)
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


def _sha256_receipted_file(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    label: str = "retrieval export artifact",
) -> str:
    with _open_receipted_file(path, receipt, label=label) as (_source, digest):
        return digest


@contextmanager
def _open_receipted_file(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    label: str = "retrieval export artifact",
) -> Iterator[tuple[BinaryIO, str]]:
    """Yield a private snapshot containing exactly the bytes hashed for the receipt."""
    with path.open("rb") as source:
        with tempfile.TemporaryFile(mode="w+b") as snapshot:
            digest = _sha256_receipted_stream(
                source,
                receipt,
                label=label,
                snapshot=snapshot,
            )
            snapshot.flush()
            snapshot.seek(0)
            yield snapshot, digest


def _sha256_receipted_stream(
    source: BinaryIO,
    receipt: Mapping[str, Any],
    *,
    label: str,
    snapshot: BinaryIO | None = None,
) -> str:
    expected_size = receipt["bytes"]
    expected_digest = receipt["sha256"]
    digest = sha256()
    total = 0
    initial_stat = os.fstat(source.fileno())
    initial_identity = (
        initial_stat.st_dev,
        initial_stat.st_ino,
        initial_stat.st_nlink,
        initial_stat.st_size,
        initial_stat.st_mtime_ns,
        initial_stat.st_ctime_ns,
    )
    if initial_stat.st_size != expected_size:
        raise ValueError(f"{label} receipt differs from stored artifact")
    while True:
        chunk = source.read(min(1024 * 1024, expected_size - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > expected_size:
            raise ValueError(f"{label} receipt differs from stored artifact")
        digest.update(chunk)
        if snapshot is not None:
            snapshot.write(chunk)
    final_stat = os.fstat(source.fileno())
    final_identity = (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_nlink,
        final_stat.st_size,
        final_stat.st_mtime_ns,
        final_stat.st_ctime_ns,
    )
    observed_digest = digest.hexdigest()
    if (
        total != expected_size
        or observed_digest != expected_digest
        or final_identity != initial_identity
    ):
        raise ValueError(f"{label} receipt differs from stored artifact")
    return observed_digest


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


def render_debug_report(data: DebugReportData) -> str:
    """Render sealed report data as one deterministic, dependency-free HTML document.

    This is deliberately a presentation boundary: all source-derived text is
    escaped here, rather than relying on the strict artifact readers to make
    stored corpus text safe for an HTML context.
    """
    topic_links = "".join(
        '<li>'
        f'<button type="button" class="topic-tab" id="topic-tab-{_topic_anchor(topic.topic_id)}" '
        f'data-topic-target="topic-{_topic_anchor(topic.topic_id)}">'
        f'{_html(topic.topic_id)}</button>'
        f'<a class="topic-link-fallback" href="#topic-{_topic_anchor(topic.topic_id)}">'
        f'{_html(topic.topic_id)}</a></li>'
        for topic in data.topics
    )
    topics = "".join(
        _render_topic(topic, initially_open=index == 0)
        for index, topic in enumerate(data.topics)
    )
    run_summary = _render_run_summary(data)
    pipeline_legend = _render_pipeline_legend()
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Competition retrieval debug report</title>
<style>
:root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; line-height: 1.5; --page: #f5f7fb; --surface: #fff; --surface-soft: #eef3fb; --ink: #182235; --muted: #5d687b; --line: #ccd5e3; --accent: #2859c5; --accent-ink: #fff; --focus: #f59e0b; }}
* {{ box-sizing: border-box; }}
html {{ overflow-x: hidden; scroll-behavior: smooth; }}
body {{ margin: 0; min-width: 0; overflow-x: hidden; background: var(--page); color: var(--ink); }}
header, main, footer {{ width: min(100%, 82rem); margin: auto; padding: clamp(1rem, 3vw, 2rem); }}
header {{ margin-top: clamp(.5rem, 2vw, 1.5rem); border-bottom: 1px solid var(--line); }}
header h1 {{ margin-block: 0 .35rem; font-size: clamp(1.75rem, 4vw, 3rem); line-height: 1.1; letter-spacing: -.035em; }}
header p {{ max-width: 70ch; color: var(--muted); }}
nav ul {{ display: flex; flex-wrap: wrap; gap: .65rem; margin: 1.25rem 0 0; padding: 0; list-style: none; }}
.topic-tab, .topic-link-fallback {{ min-height: 44px; align-items: center; justify-content: center; padding: .65rem 1rem; border-radius: 999px; border: 1px solid var(--line); font: inherit; font-weight: 750; color: var(--ink); background: var(--surface); text-decoration: none; cursor: pointer; }}
.topic-tab {{ display: none; }}
.js .topic-tab {{ display: inline-flex; }}
.js .topic-link-fallback {{ display: none; }}
.topic-tab[aria-selected="true"] {{ color: var(--accent-ink); border-color: var(--accent); background: var(--accent); box-shadow: 0 .35rem 1rem rgb(40 89 197 / 22%); }}
section {{ min-width: 0; margin: 1.5rem 0; padding: clamp(.85rem, 2.5vw, 1.35rem); border: 1px solid var(--line); border-radius: .75rem; background: var(--surface); }}
section section {{ border-color: color-mix(in srgb, var(--line) 75%, transparent); background: color-mix(in srgb, var(--surface) 92%, var(--surface-soft)); }}
.topic-panel {{ min-width: 0; margin: 1.5rem 0; border: 1px solid var(--line); border-radius: 1rem; background: var(--surface); box-shadow: 0 .6rem 2rem rgb(30 50 90 / 8%); }}
.topic-panel > summary {{ min-height: 44px; padding: 1rem 1.25rem; cursor: pointer; font-size: 1.2rem; font-weight: 800; }}
.topic-panel > .topic-content {{ min-width: 0; padding: 0 clamp(.75rem, 2vw, 1.25rem) .25rem; }}
.js .topic-panel:not([open]) {{ display: none; }}
.js .topic-panel > summary {{ display: none; }}
.run-diagnostics {{ margin-top: 1rem; padding: .25rem 1rem 1rem; border: 1px solid var(--line); border-radius: .65rem; background: var(--surface-soft); }}
.run-diagnostics > summary {{ font-weight: 750; }}
.run-diagnostics dl {{ display: grid; grid-template-columns: minmax(9rem, auto) minmax(0, 1fr); gap: .35rem 1rem; }}
.run-diagnostics dt {{ font-weight: 750; }}
.run-diagnostics dd {{ min-width: 0; margin: 0; }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; min-width: 38rem; }}
caption {{ text-align: left; font-weight: 700; padding: .4rem 0; }}
th, td {{ text-align: left; vertical-align: top; border: 1px solid var(--line); padding: .45rem; }}
code, .break {{ overflow-wrap: anywhere; word-break: break-word; }}
details:not(.topic-panel, .run-diagnostics) {{ margin: .75rem 0; padding: .5rem; border-inline-start: .25rem solid var(--line); }}
summary {{ min-height: 44px; display: list-item; padding-block: .6rem; cursor: pointer; font-weight: 650; }}
.status {{ display: inline-block; padding: .1rem .45rem; border-radius: 999px; font-weight: 700; }}
.status-complete {{ color: #063; background: #d8f3df; }}
.status-empty {{ color: #735400; background: #fff0bd; }}
.status-fallback-extractive {{ color: #7a2300; background: #ffe0d2; }}
.subnarrative-list, .new-document-list, .selected-document-list, .retrieval-document-list, .rag-answer-list, .rag-reference-list {{ display: grid; gap: 1rem; margin: 0; padding: 0; list-style: none; }}
.subnarrative-card, .new-document-card, .selected-document-card, .retrieval-document-card, .rag-answer-item, .rag-reference-card {{ min-width: 0; padding: clamp(.85rem, 2vw, 1.15rem); border: 1px solid var(--line); border-radius: .75rem; background: var(--surface-soft); }}
.subnarrative-card h3, .new-document-card h4, .retrieval-document-card h3 {{ margin-top: 0; }}
.query-list {{ margin-bottom: 0; }}
.stage-note, .rank-caveat {{ color: var(--muted); }}
.new-document-lane, .passage-ranking-disclosure, .canonical-cluster-diagnostics, .canonical-result {{ margin-block: 1rem; border: 1px solid var(--line); border-inline-start-width: .25rem; border-radius: .65rem; background: var(--surface-soft); }}
.new-document-lane > summary, .passage-ranking-disclosure > summary, .canonical-cluster-diagnostics > summary, .canonical-result > summary {{ padding-inline: .5rem; }}
.new-document-lane > .new-document-list, .passage-ranking-disclosure > .passage-diagnostics, .canonical-cluster-diagnostics > .table-wrap, .canonical-result > .canonical-result-detail {{ margin: .5rem; }}
.card-heading {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: .45rem .75rem; margin: 0 0 .75rem; }}
.card-rank, .citation-chip {{ display: inline-flex; min-height: 2rem; align-items: center; padding: .2rem .65rem; border-radius: 999px; font-weight: 750; }}
.card-rank {{ color: var(--accent-ink); background: var(--accent); }}
.card-metadata, .rag-provenance dl {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 11rem), 1fr)); gap: .65rem 1rem; margin: .75rem 0; }}
.card-metadata > div, .rag-provenance dl > div {{ min-width: 0; }}
.card-metadata dt, .rag-provenance dt {{ color: var(--muted); font-size: .86rem; font-weight: 750; text-transform: uppercase; letter-spacing: .035em; }}
.card-metadata dd, .rag-provenance dd {{ margin: .15rem 0 0; }}
.evidence-callout {{ margin: 1rem 0; padding: .8rem 1rem; border-inline-start: .3rem solid var(--accent); border-radius: .35rem; background: var(--surface); }}
.evidence-callout h4 {{ margin: 0 0 .35rem; }}
.technical-provenance {{ margin-top: .85rem; }}
.rag-provenance {{ margin: 0 0 1.25rem; padding: 1rem; border: 1px solid var(--line); border-radius: .75rem; background: var(--surface-soft); }}
.rag-provenance h3, .rag-reference-card h4, .rag-answer-item h4 {{ margin-top: 0; }}
.rag-answer-item {{ background: var(--surface); }}
.citation-list {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .75rem; }}
.citation-chip {{ min-height: 44px; color: var(--accent); border: 1px solid var(--accent); background: var(--surface-soft); text-decoration: none; }}
.citation-chip:hover {{ color: var(--accent-ink); background: var(--accent); }}
.rag-reference-list {{ margin-top: 1rem; }}
a:focus-visible, button:focus-visible, summary:focus-visible {{ outline: .22rem solid var(--focus); outline-offset: .2rem; }}
@media (prefers-color-scheme: dark) {{ :root {{ --page: #0d1320; --surface: #141d2d; --surface-soft: #1a263a; --ink: #edf3ff; --muted: #aebbd0; --line: #39475e; --accent: #7ca1ff; --accent-ink: #10182a; --focus: #fbbf24; }} }}
@media (max-width: 42rem) {{ header, main, footer {{ padding: .75rem; }} section {{ padding: .75rem; }} nav li {{ flex: 1 1 calc(50% - .65rem); }} .topic-tab, .topic-link-fallback {{ width: 100%; }} .run-diagnostics dl {{ display: block; }} .run-diagnostics dd {{ margin: 0 0 .75rem; }} }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; }} }}
</style>
</head>
<body>
<header>
<h1>Competition retrieval debug report</h1>
<p>Read-only rendering of sealed retrieval artifacts. Corpus text and identifiers may be sensitive.</p>
<nav aria-label="Topic navigation"><ul>{topic_links}</ul></nav>
</header>
<main>
{run_summary}
{pipeline_legend}
{topics}
</main>
<footer><p>Generated deterministically from sealed report data; no retrieval, model, or network calls were made.</p></footer>
<script>
(() => {{
  const panels = Array.from(document.querySelectorAll(".topic-panel"));
  const tabs = Array.from(document.querySelectorAll(".topic-tab"));
  if (!panels.length) return;
  const tablist = document.querySelector('nav[aria-label="Topic navigation"] ul');
  let synchronizing = false;
  const panelForTab = (tab) => document.getElementById(tab.dataset.topicTarget);
  const activate = (panel, updateHash, focusTab) => {{
    if (!panel) return;
    synchronizing = true;
    panels.forEach((candidate) => {{ candidate.open = candidate === panel; }});
    tabs.forEach((tab) => {{
      const selected = panelForTab(tab) === panel;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focusTab) tab.focus();
    }});
    synchronizing = false;
    if (updateHash && location.hash !== "#" + panel.id) {{
      history.replaceState(null, "", "#" + panel.id);
    }}
  }};
  const restoreHash = () => {{
    let target = null;
    try {{
      target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    }} catch (_error) {{
      target = null;
    }}
    const requested = target && target.closest(".topic-panel");
    if (requested && panels.includes(requested)) {{
      activate(requested, false, false);
      setTimeout(() => {{
        let currentTarget = null;
        try {{
          currentTarget = document.getElementById(
            decodeURIComponent(location.hash.slice(1))
          );
        }} catch (_error) {{
          currentTarget = null;
        }}
        if (currentTarget === target) {{
          target.scrollIntoView({{ block: "start", behavior: "instant" }});
        }}
      }}, 0);
      return;
    }}
    activate(panels.find((panel) => panel.open) || panels[0], false, false);
  }};
  tabs.forEach((tab, index) => {{
    tab.addEventListener("click", () => activate(panelForTab(tab), true, false));
    tab.addEventListener("keydown", (event) => {{
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const direction = event.key === "ArrowRight" ? 1 : -1;
      const next = tabs[(index + direction + tabs.length) % tabs.length];
      activate(panelForTab(next), true, true);
    }});
  }});
  panels.forEach((panel) => panel.addEventListener("toggle", () => {{
    if (synchronizing) return;
    if (panel.open) activate(panel, false, false);
    else if (!panels.some((candidate) => candidate.open)) activate(panel, false, false);
  }}));
  window.addEventListener("hashchange", restoreHash);
  tablist.setAttribute("role", "tablist");
  tabs.forEach((tab) => {{
    const panel = panelForTab(tab);
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", panel.id);
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", tab.id);
    panel.querySelector(":scope > summary").hidden = true;
  }});
  restoreHash();
  document.documentElement.classList.add("js");
}})();
</script>
</body>
</html>'''


def _render_run_summary(data: DebugReportData) -> str:
    topic_ids = ", ".join(topic.topic_id for topic in data.topics)
    rag_path = str(data.rag_config_path) if data.rag_config_path is not None else "Not supplied"
    source_rows = "".join(
        f'<tr><th scope="row">{_html(label)}</th><td><code>{_html(digest)}</code></td></tr>'
        for label, digest in sorted(data.source_sha256s.items())
    )
    return (
        '<section id="run-summary" aria-labelledby="run-summary-title">'
        '<h2 id="run-summary-title">Run summary</h2>'
        f'<p>{_html(len(data.topics))} validated topic'
        f'{"s" if len(data.topics) != 1 else ""} ready to inspect.</p>'
        '<details class="run-diagnostics"><summary>Validation, configuration, and source receipts</summary><dl>'
        f'<dt>Retrieval config</dt><dd class="break">{_html(data.retrieval_config_path)}</dd>'
        f'<dt>RAG config</dt><dd class="break">{_html(rag_path)}</dd>'
        f'<dt>Retrieval output</dt><dd class="break">{_html(data.output_dir)}</dd>'
        f'<dt>Included topics</dt><dd>{_html(len(data.topics))}: {_html(topic_ids)}</dd>'
        '<dt>Validation state</dt><dd>Bounded sealed artifacts validated</dd>'
        f'<dt>Source receipts</dt><dd>{_html(len(data.source_sha256s))} SHA-256 values</dd>'
        '</dl>'
        + _table(
            "Bounded artifact SHA-256 receipts",
            ("Artifact", "SHA-256"),
            source_rows,
        )
        + "</details></section>"
    )


def _render_pipeline_legend() -> str:
    definitions = (
        ("Narrative", "The official topic narrative supplied to the run."),
        ("Subnarratives", "Stored decomposition facets and their literal BM25 queries."),
        (
            "New documents",
            "Union-pool documents with no original lane membership; new is relative to the original reranked eligible pool, not the corpus.",
        ),
        ("Selected documents", "The stored selected pool with lane memberships and selection rationale."),
        ("Top passages", "Every stored document ranking, grouped by subnarrative; raw model scores are logits."),
        ("Final selected nuggets", "Configured-budget evidence clusters and their final canonical claims."),
        ("Final retrieval", "Organizer-facing documents supported by at least one canonical nugget."),
        ("Final RAG", "Validated answer items, references, and resolved citations when supplied."),
    )
    rows = "".join(
        f"<dt>{_html(label)}</dt><dd>{_html(description)}</dd>"
        for label, description in definitions
    )
    return (
        '<section id="pipeline-legend" aria-labelledby="pipeline-legend-title">'
        '<h2 id="pipeline-legend-title">Pipeline legend</h2>'
        f"<dl>{rows}</dl></section>"
    )


def _render_topic(topic: TopicReport, *, initially_open: bool = False) -> str:
    anchor = _topic_anchor(topic.topic_id)
    prefix = f"stage-{anchor}"
    stages = (
        _render_narrative(topic, prefix),
        _render_subnarratives(topic, prefix),
        _render_new_documents(topic, prefix),
        _render_selected_documents(topic, prefix),
        _render_passages(topic, prefix),
        _render_nuggets(topic, prefix),
        _render_retrieval(topic, prefix),
        _render_final_rag(topic, prefix),
    )
    open_attribute = " open" if initially_open else ""
    return (
        f'<details class="topic-panel" name="competition-topic" '
        f'id="topic-{anchor}"{open_attribute}>'
        f'<summary id="topic-summary-{anchor}">Topic {_html(topic.topic_id)}</summary>'
        f'<div class="topic-content"><h1 id="topic-title-{anchor}">'
        f'Topic {_html(topic.topic_id)}</h1>{"".join(stages)}</div></details>'
    )


def _stage(prefix: str, suffix: str, title: str, body: str) -> str:
    identifier = f"{prefix}-{suffix}"
    return f'<section id="{identifier}" aria-labelledby="{identifier}-title"><h2 id="{identifier}-title">{title}</h2>{body}</section>'


def _render_narrative(topic: TopicReport, prefix: str) -> str:
    fallback = ""
    if topic.original_only_fallback:
        fallback = (
            '<p><span class="status status-fallback-extractive">'
            "Original-only fallback</span> The sealed decomposition retained only "
            "the official narrative.</p>"
        )
    return _stage(
        prefix,
        "narrative",
        "Narrative",
        f'<p class="break">{_html(topic.narrative)}</p><p>Source seal: <code>{_html(topic.narrative_sha256)}</code></p>{fallback}',
    )


def _render_subnarratives(topic: TopicReport, prefix: str) -> str:
    if not topic.subnarratives:
        return _stage(
            prefix,
            "subnarratives",
            "Subnarratives",
            "<p>No generated subnarratives; the sealed run used the original-only fallback.</p>",
        )
    cards = "".join(
        '<li><article class="subnarrative-card">'
        f'<h3><code>{_html(item.subnarrative_id)}</code></h3>'
        f'<p class="break">{_html(item.text)}</p>'
        '<h4>Literal BM25 queries</h4><ol class="query-list">'
        + "".join(
            f'<li class="break">{_html(query)}</li>' for query in item.bm25_queries
        )
        + '</ol><details class="technical-provenance"><summary>Query and seal details</summary><dl>'
        f'<dt>Semantic-query SHA-256</dt><dd><code>{_html(item.semantic_query_sha256)}</code></dd>'
        '<dt>BM25 query SHA-256 values</dt><dd><ol>'
        + "".join(
            f'<li><code>{_html(digest)}</code></li>'
            for digest in item.bm25_query_sha256s
        )
        + "</ol></dd></dl></details></article></li>"
        for item in topic.subnarratives
    )
    return _stage(
        prefix,
        "subnarratives",
        "Subnarratives",
        '<p class="stage-note">Stored decomposition facets and their literal '
        'retrieval queries.</p><ol class="subnarrative-list">'
        f"{cards}</ol>",
    )


def _render_new_documents(topic: TopicReport, prefix: str) -> str:
    grouped: dict[str, list[NewDocumentReport]] = {}
    for item in topic.new_documents:
        grouped.setdefault(item.first_seen_lane, []).append(item)
    groups: list[str] = []
    for lane_name, items in grouped.items():
        cards = "".join(
            _render_new_document_card(item)
            for item in items
        )
        groups.append(
            '<details class="new-document-lane">'
            f"<summary>{_html(lane_name)} — {_html(len(items))} documents</summary>"
            f'<ol class="new-document-list">{cards}</ol></details>'
        )
    body = (
        f"<p>{_html(len(topic.new_documents))} facet-only new documents; "
        "counts are grouped by stored first-seen lane.</p>"
        + "".join(groups)
    )
    return _stage(
        prefix,
        "new-documents",
        "New documents",
        body,
    )


def _render_new_document_card(item: NewDocumentReport) -> str:
    excerpt = (
        f'<p class="break">{_html(_bounded_excerpt(item.excerpt))}</p>'
        if item.excerpt is not None
        else '<p class="stage-note">Document text is not stored.</p>'
    )
    memberships = ", ".join(item.memberships) or "No stored memberships"
    return (
        '<li><article class="new-document-card">'
        f'<h4><code>{_html(item.docid)}</code></h4>{excerpt}'
        '<dl class="card-metadata">'
        f'<div><dt>First-seen lane</dt><dd class="break">{_html(item.first_seen_lane)}</dd></div>'
        f'<div><dt>Memberships</dt><dd class="break">{_html(memberships)}</dd></div>'
        '</dl><details class="technical-provenance"><summary>Lane and document provenance</summary><dl>'
        f'<dt>Text SHA-256</dt><dd><code>{_html(item.text_sha256 or "Not stored")}</code></dd>'
        f'<dt>New-document marker</dt><dd>{_html(item.is_new)}</dd>'
        f'<dt>Lane rank and score provenance</dt><dd>{_detail_lane_provenance(item.lane_provenance)}</dd>'
        "</dl></details></article></li>"
    )


def _render_selected_documents(topic: TopicReport, prefix: str) -> str:
    cards = "".join(
        _render_selected_document_card(topic, item)
        for item in topic.selected_documents
    )
    return _stage(
        prefix,
        "selected-documents",
        "Selected documents",
        '<p>Stored selections, explained by their sealed lane evidence and a '
        'representative available passage.</p>'
        '<p class="rank-caveat"><strong>Representative stored passage rule:</strong> '
        'lowest stored aggregate rank among subnarratives; decomposition order breaks ties. '
        'Ranks are facet-local; cross-facet logits are not compared.</p>'
        '<ol class="selected-document-list">'
        f"{cards}</ol>",
    )


def _render_selected_document_card(
    topic: TopicReport, item: SelectedDocumentReport
) -> str:
    best = _best_stored_passage(topic, item.docid)
    memberships = "; ".join(_membership_text(value) for value in item.memberships)
    membership_coverage = _membership_coverage(item.memberships)
    status = "Original member" if item.is_original_member else "Facet-only"
    if best is None:
        evidence = (
            '<div class="evidence-callout"><h4>Representative stored passage</h4>'
            "<p>No stored passage ranking is available for this selected document.</p></div>"
        )
    else:
        evidence = (
            '<div class="evidence-callout"><h4>Representative stored passage</h4>'
            f'<p class="break">{_html(_bounded_excerpt(best.passage.text))}</p>'
            '<dl class="card-metadata">'
            f'<div><dt>Subnarrative</dt><dd><code>{_html(best.subnarrative_id)}</code></dd></div>'
            f'<div><dt>Aggregate rank</dt><dd>{_html(best.aggregate_rank)}</dd></div>'
            f'<div><dt>Facet</dt><dd class="break">{_html(best.subnarrative_text)}</dd></div>'
            "</dl></div>"
        )
    return (
        '<li><article class="selected-document-card" '
        f'id="selected-document-{_topic_anchor(topic.topic_id)}-{_html(item.selection_rank)}">'
        '<h3 class="card-heading">'
        f'<span class="card-rank">#{_html(item.selection_rank)}</span>'
        f'<code>{_html(item.docid)}</code></h3>'
        f'<p class="selection-reason">{_html(_selected_document_reason(item))}</p>'
        '<dl class="card-metadata">'
        f'<div><dt>Status</dt><dd>{_html(status)}</dd></div>'
        f'<div><dt>Selected lane</dt><dd class="break">{_html(item.selected_from_lane)}</dd></div>'
        f'<div><dt>Lane rank</dt><dd>{_html(item.selected_from_lane_rank)}</dd></div>'
        f'<div><dt>Memberships</dt><dd>{_html(membership_coverage)}</dd></div>'
        f"</dl>{evidence}"
        '<details class="technical-provenance"><summary>Technical provenance</summary><dl>'
        f'<dt>Text SHA-256</dt><dd><code>{_html(item.text_sha256)}</code></dd>'
        f'<dt>Memberships and selection rationale</dt><dd class="break">{_html(memberships)}. {_html(item.selection_rationale)}</dd>'
        f'<dt>Document excerpt (first {_DOCUMENT_EXCERPT_CHARACTERS} characters)</dt><dd class="break">{_html(_bounded_excerpt(item.text))}</dd>'
        "</dl></details></article></li>"
    )


def _membership_coverage(memberships: Sequence[Mapping[str, Any]]) -> str:
    lane_names = {str(value["lane_name"]) for value in memberships}
    if not lane_names:
        return "No stored lane memberships"
    has_original = "original" in lane_names
    facet_count = sum(name.startswith("facet:") for name in lane_names)
    other_count = len(lane_names) - int(has_original) - facet_count
    if has_original and not facet_count and not other_count:
        return "Original lane"
    parts: list[str] = []
    if has_original:
        parts.append("Original")
    if facet_count:
        parts.append(
            f"{facet_count} facet lane" + ("" if facet_count == 1 else "s")
        )
    if other_count:
        parts.append(
            f"{other_count} other lane" + ("" if other_count == 1 else "s")
        )
    return " + ".join(parts)


def _render_passage_rows(rankings: Sequence[PassageRankingReport]) -> str:
    return "".join(
        "<tr>"
        f"<th scope=\"row\">{_html(item.aggregate_rank)}</th><td>{_html(item.subnarrative_id)}</td>"
        f"<td>{_html(item.docid)}</td><td>{_html(item.selection_rank)}</td>"
        f"<td>{_html(item.bm25_rank)}</td><td>{_html(_number(item.bm25_score))}</td>"
        f"<td>{_html(_number(item.aggregate_score))}</td>"
        f"<td>{_html(_number(item.long_document_raw_logit))}</td>"
        f"<td>{_html(_number(item.weighted_passage_raw_logit))}</td>"
        f"<td>{_html(item.within_document_span_support)}</td>"
        f"<td>{_detail_passages(item.winning_passages)}</td></tr>"
        for item in rankings
    )


def _render_passages(topic: TopicReport, prefix: str) -> str:
    if not topic.subnarratives:
        return _stage(
            prefix,
            "top-passages",
            "Top passages",
            "<p>No downstream passage rankings; original-only fallback bypassed downstream scoring.</p>",
        )
    headings = ("Aggregate rank", "Subnarrative", "DocID", "Selection rank", "BM25 rank", "BM25 score", "Aggregate score", "Document raw logit", "Passage raw logit", "Span support", "Winning passages")
    groups: list[str] = []
    for subnarrative in topic.subnarratives:
        rankings = tuple(
            row
            for row in topic.passage_rankings
            if row.subnarrative_id == subnarrative.subnarrative_id
        )
        visible = _table(
            f"Top stored passage rankings for {subnarrative.subnarrative_id}",
            headings,
            _render_passage_rows(rankings[:5]),
        )
        remainder = rankings[5:]
        disclosure = ""
        if remainder:
            disclosure = (
                '<details class="passage-remainder"><summary>Show remaining '
                f"{len(remainder)} stored passage rankings</summary>"
                + _table(
                    f"Remaining stored passage rankings for {subnarrative.subnarrative_id}",
                    headings,
                    _render_passage_rows(remainder),
                )
                + "</details>"
            )
        heading_id = (
            f"{prefix}-top-passages-{_topic_anchor(subnarrative.subnarrative_id)}"
        )
        visible_count = min(5, len(rankings))
        count_summary = (
            f"Top {visible_count} of {len(rankings)} stored rankings"
            if len(rankings) > visible_count
            else f"{len(rankings)} stored ranking"
            + ("" if len(rankings) == 1 else "s")
        )
        groups.append(
            '<section class="passage-ranking-group" '
            f'aria-labelledby="{_html(heading_id)}"><h3 id="{_html(heading_id)}">'
            f"Subnarrative {_html(subnarrative.subnarrative_id)}</h3>"
            '<details class="passage-ranking-disclosure" '
            f'data-subnarrative="{_html(subnarrative.subnarrative_id)}">'
            f"<summary>{_html(count_summary)} · diagnostic table</summary>"
            f'<div class="passage-diagnostics">{visible}{disclosure}</div>'
            "</details></section>"
        )
    return _stage(
        prefix,
        "top-passages",
        "Top passages",
        '<p class="stage-note">Aggregate ranks are meaningful only within each '
        'subnarrative. Open a facet to inspect its stored diagnostic table.</p>'
        + "".join(groups),
    )


def _render_nuggets(topic: TopicReport, prefix: str) -> str:
    if topic.original_only_fallback and not topic.canonical_results:
        return _stage(
            prefix,
            "final-selected-nuggets",
            "Final selected nuggets",
            "<p>No canonical result rows; original-only fallback performed no downstream canonical work.</p>",
        )
    cluster_rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{_html(item.subnarrative_id)}</th><td>{_html(item.selected_budget)}</td>"
        f"<td>{_html(item.cluster_id)}</td><td class=\"break\">{_html(item.representative_text)}</td>"
        f"<td>{_html(_number(item.representative_raw_logit))}</td>"
        f"<td>{_detail_evidence(item.evidence)}</td></tr>"
        for item in topic.evidence_clusters
    )
    cluster_count = len(topic.evidence_clusters)
    body = (
        '<details class="canonical-cluster-diagnostics"><summary>'
        f'{_html(cluster_count)} selected evidence cluster'
        f'{"" if cluster_count == 1 else "s"} · diagnostic table</summary>'
        + _table(
            "Selected evidence clusters",
            (
                "Subnarrative",
                "Budget",
                "Cluster",
                "Representative text",
                "Representative raw logit",
                "Evidence",
            ),
            cluster_rows,
        )
        + "</details>"
    )
    result_groups: list[str] = []
    for result in topic.canonical_results:
        claims = tuple(
            item
            for item in topic.canonical_nuggets
            if item.subnarrative_id == result.subnarrative_id
        )
        claim_rows = "".join(
            "<tr>"
            f"<th scope=\"row\">{_html(item.canonical_nugget_id)}</th>"
            f"<td>{_html(item.nugget_kind)}</td>"
            f"<td class=\"break\">{_html(item.claim_text)}</td>"
            f"<td>{_detail_evidence(item.evidence)}</td></tr>"
            for item in claims
        )
        claim_count = len(claims)
        claim_label = f"{claim_count} canonical claim" + (
            "" if claim_count == 1 else "s"
        )
        claims_body = (
            _table(
                f"Claims for {result.subnarrative_id}",
                ("Nugget ID", "Kind", "Claim", "Supporting evidence"),
                claim_rows,
            )
            if claims
            else "<p>No canonical claims were retained for this result.</p>"
        )
        result_groups.append(
            '<details class="canonical-result"><summary>'
            f"{_html(result.subnarrative_id)} · {_html(result.state)} · {_html(claim_label)}"
            '</summary><div class="canonical-result-detail"><dl>'
            f'<dt>State</dt><dd><span class="status {_status_class(result.state)}">{_html(result.state)}</span></dd>'
            f"<dt>Selected budget</dt><dd>{_html(result.selected_budget)}</dd>"
            f"<dt>Configured maximum claims</dt><dd>{_html(result.maximum_claims)}</dd>"
            f"<dt>Configured maximum supporting documents per claim</dt><dd>{_html(result.maximum_supporting_documents)}</dd>"
            f"<dt>Claims</dt><dd>{_html(claim_label)}</dd>"
            "</dl>"
            f"{claims_body}</div></details>"
        )
    body += "".join(result_groups)
    return _stage(prefix, "final-selected-nuggets", "Final selected nuggets", body)


def _render_retrieval(topic: TopicReport, prefix: str) -> str:
    cards = "".join(
        _render_retrieval_document_card(item)
        for item in topic.retrieval_output.documents
    )
    projection = (
        " Final retrieval uses the sealed original-only selected pool."
        if topic.original_only_fallback
        else ""
    )
    intro = f"<p>Selected-pool depth: {_html(topic.retrieval_output.selected_pool_depth)}. Final supported depth: {_html(topic.retrieval_output.final_supported_depth)}.{projection}</p>"
    return _stage(
        prefix,
        "final-retrieval",
        "Final retrieval",
        intro + f'<ol class="retrieval-document-list">{cards}</ol>',
    )


def _render_retrieval_document_card(item: RetrievalDocumentReport) -> str:
    return (
        '<li><article class="retrieval-document-card">'
        '<h3 class="card-heading">'
        f'<span class="card-rank">#{_html(item.rank)}</span>'
        f'<code>{_html(item.docid)}</code></h3>'
        '<dl class="card-metadata">'
        f'<div><dt>Score</dt><dd>{_html(_number(item.score))}</dd></div>'
        f'<div><dt>Selection rank</dt><dd>{_html(item.selection_rank)}</dd></div>'
        f'<div><dt>Source lane</dt><dd class="break">{_html(item.selected_from_lane)}</dd></div>'
        '</dl><details class="technical-provenance"><summary>'
        'Retrieval provenance and document excerpt</summary><dl>'
        f'<dt>Selected lane rank</dt><dd>{_html(item.selected_from_lane_rank)}</dd>'
        f'<dt>Stage</dt><dd>{_html(item.stage)}</dd>'
        '</dl><h4>Sealed provenance</h4>'
        f'{_retrieval_provenance_fields(item)}'
        f'<h4>Document excerpt (first {_DOCUMENT_EXCERPT_CHARACTERS} characters)</h4>'
        f'<p class="break">{_html(_bounded_excerpt(item.text))}</p>'
        "</details></article></li>"
    )


def _render_final_rag(topic: TopicReport, prefix: str) -> str:
    rag = topic.rag_output
    if rag is None:
        return _stage(
            prefix,
            "final-rag",
            "Final RAG",
            '<p><span class="status status-empty">Not included</span> RAG output not supplied.</p>',
        )
    anchor_prefix = f"rag-reference-{_topic_anchor(topic.topic_id)}"
    documents = {item.docid: item for item in topic.retrieval_output.documents}
    answers = "".join(
        _render_rag_answer_item(item, index, anchor_prefix)
        for index, item in enumerate(rag.answer_items, start=1)
    )
    references = "".join(
        _render_rag_reference_card(index, docid, documents.get(docid), anchor_prefix)
        for index, docid in enumerate(rag.references)
    )
    body = (
        '<aside class="rag-provenance" aria-label="RAG generation provenance">'
        "<h3>Generation provenance</h3><dl>"
        '<div><dt>Implementation</dt><dd><code>trec_rag.competition_rag</code></dd></div>'
        f'<div><dt>Run</dt><dd><code>{_html(rag.run_id)}</code></dd></div>'
        f'<div><dt>Description</dt><dd class="break">{_html(rag.run_desc)}</dd></div>'
        f'<div><dt>Provider</dt><dd><code>{_html(rag.provider)}</code></dd></div>'
        f'<div><dt>Model</dt><dd><code>{_html(rag.model)}</code></dd></div>'
        '<div><dt>Validation</dt><dd>Validated against standard retrieval inputs</dd></div>'
        f'<div><dt>Answer length</dt><dd>{_html(rag.word_count)} words</dd></div>'
        "</dl><p>Implementation authorship is not recorded in sealed run artifacts.</p></aside>"
        '<h3>Generated answer</h3><ol class="rag-answer-list">'
        f"{answers}</ol><h3>Referenced documents</h3>"
        '<ol class="rag-reference-list">'
        f"{references}</ol>"
        '<details class="technical-provenance"><summary>RAG output receipt</summary>'
        f'<p>Output SHA-256: <code>{_html(rag.output_sha256)}</code>.</p></details>'
    )
    return _stage(prefix, "final-rag", "Final RAG", body)


def _render_rag_answer_item(
    item: RagAnswerItemReport, index: int, anchor_prefix: str
) -> str:
    citations = "".join(
        '<a class="citation-chip" '
        f'href="#{anchor_prefix}-{_html(citation)}">'
        f'citation {_html(citation)} → {_html(docid)}</a>'
        for citation, docid in zip(
            item.citations, item.citation_docids, strict=True
        )
    )
    return (
        '<li><article class="rag-answer-item">'
        f'<h4>Answer {_html(index)}</h4>'
        f'<p class="break">{_html(item.text)}</p>'
        f'<div class="citation-list" aria-label="Resolved citations">{citations}</div>'
        "</article></li>"
    )


def _render_rag_reference_card(
    citation: int,
    docid: str,
    document: RetrievalDocumentReport | None,
    anchor_prefix: str,
) -> str:
    if document is None:
        excerpt = "Validated reference text is not stored in this report."
        detail = "No retrieval detail is stored for this reference."
    else:
        excerpt = _bounded_excerpt(document.text)
        memberships = "; ".join(
            _membership_text(value) for value in document.memberships
        ) or "None"
        seals = "; ".join(
            f"{key}: {value}" for key, value in sorted(document.source_seals.items())
        ) or "None"
        detail = (
            f"Organizer rank {document.rank}; selection rank {document.selection_rank}; "
            f"source lane {document.selected_from_lane}; memberships {memberships}; "
            f"source seals {seals}."
        )
    return (
        '<li><article class="rag-reference-card" '
        f'id="{anchor_prefix}-{_html(citation)}">'
        f'<h4>Reference {_html(citation)} · <code>{_html(docid)}</code></h4>'
        '<dl class="card-metadata">'
        f'<div><dt>Citation index</dt><dd>{_html(citation)}</dd></div>'
        f'<div><dt>DocID</dt><dd><code>{_html(docid)}</code></dd></div>'
        "</dl>"
        f'<p class="break">{_html(excerpt)}</p>'
        '<details class="technical-provenance"><summary>Technical document detail</summary>'
        f'<p class="break">{_html(detail)}</p></details>'
        "</article></li>"
    )


def _bounded_excerpt(value: str) -> str:
    excerpt = value[:_DOCUMENT_EXCERPT_CHARACTERS]
    if len(value) > _DOCUMENT_EXCERPT_CHARACTERS:
        excerpt += "…"
    return excerpt


def _table(caption: str, headings: Sequence[str], rows: str) -> str:
    header = "".join(f'<th scope="col">{_html(heading)}</th>' for heading in headings)
    return f'<div class="table-wrap"><table><caption>{_html(caption)}</caption><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'


def _detail_passages(passages: Sequence[WinningPassageReport]) -> str:
    items = "".join(
        f'<li>Chunk {_html(item.chunk_index)}, offsets {_html(item.start_char)}–{_html(item.end_char)}, raw logit {_html(_number(item.raw_logit))}: <span class="break">{_html(item.text)}</span></li>'
        for item in passages
    )
    return f'<details><summary>{_html(len(passages))} stored passage(s)</summary><ul>{items}</ul></details>'


def _detail_lane_provenance(
    lanes: Sequence[LaneScoreProvenanceReport],
) -> str:
    items = "".join(
        "<li>"
        f"{_html(item.lane_name)}: aggregate rank {_html(item.aggregate_rank)}, "
        f"aggregate score {_html(_number(item.aggregate_score))}, "
        f"BM25 rank {_html(item.bm25_rank)}, "
        f"BM25 score {_html(_number(item.bm25_score))}"
        "</li>"
        for item in lanes
    )
    return (
        "<details><summary>Sealed lane rank and score provenance</summary>"
        f"<ul>{items}</ul></details>"
    )


def _detail_evidence(evidence: Sequence[CanonicalEvidenceReport]) -> str:
    items = "".join(
        f'<li><code>{_html(item.docid)}</code>: <span class="break">{_html(item.text)}</span></li>'
        for item in evidence
    )
    return f'<details><summary>{_html(len(evidence))} evidence item(s)</summary><ul>{items}</ul></details>'


def _retrieval_provenance_fields(item: RetrievalDocumentReport) -> str:
    memberships = "; ".join(
        _membership_text(value) for value in item.memberships
    ) or "None"
    scores = "; ".join(_score_text(value) for value in item.subnarrative_scores) or "None"
    nuggets = ", ".join(item.canonical_nugget_ids) or "None"
    seals = "; ".join(f"{key}: {value}" for key, value in sorted(item.source_seals.items())) or "None"
    return (
        "<dl>"
        f"<dt>Memberships</dt><dd class=\"break\">{_html(memberships)}</dd>"
        f"<dt>Subnarrative scores</dt><dd class=\"break\">{_html(scores)}</dd>"
        f"<dt>Canonical nugget IDs</dt><dd class=\"break\">{_html(nuggets)}</dd>"
        f"<dt>Source seals</dt><dd class=\"break\">{_html(seals)}</dd></dl>"
    )


def _membership_text(value: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{key}={value[key]}" for key in ("lane_name", "aggregate_rank", "aggregate_score", "bm25_rank", "bm25_score") if key in value
    )


def _score_text(value: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{key}={value[key]}" for key in ("subnarrative_id", "aggregate_rank", "aggregate_score", "weighted_passage_raw_logit") if key in value
    )


def _number(value: int | float) -> str:
    return format(value, ".12g")


def _topic_anchor(topic_id: str) -> str:
    """Return a stable HTML-safe anchor token without trusting stored text."""
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", topic_id):
        return f"literal-{topic_id}"
    return f"hash-{sha256(topic_id.encode('utf-8')).hexdigest()[:16]}"


def _status_class(state: str) -> str:
    return {
        "complete": "status-complete",
        "empty": "status-empty",
        "fallback_extractive": "status-fallback-extractive",
    }.get(state, "status-empty")


def _html(value: object) -> str:
    return html.escape(str(value), quote=True)


def build_debug_report(
    retrieval_config_path: Path,
    *,
    rag_config_path: Path | None = None,
    topic_ids: Sequence[str] | None = None,
    output_path: Path | None = None,
) -> DebugReportReceipt:
    """Validate completed artifacts, atomically write HTML, and return its receipt."""
    data = load_debug_report_data(
        Path(retrieval_config_path),
        rag_config_path=None if rag_config_path is None else Path(rag_config_path),
        topic_ids=topic_ids,
    )
    target = _resolve_report_output(data, output_path)
    rendered = render_debug_report(data)
    _atomic_write_report(target, rendered.encode("utf-8"))
    return DebugReportReceipt(
        schema_version=_REPORT_SCHEMA_VERSION,
        output_path=target,
        topic_ids=tuple(topic.topic_id for topic in data.topics),
        rag_included=data.rag_config_path is not None,
        source_sha256s=MappingProxyType(dict(sorted(data.source_sha256s.items()))),
    )


def _resolve_report_output(data: DebugReportData, output_path: Path | None) -> Path:
    repo_root = find_repo_root(data.retrieval_config_path.parent).resolve()
    default_target = data.output_dir / "competition_debug_report.html"
    if output_path is None:
        requested = default_target
    else:
        requested = Path(output_path)
    if requested.is_symlink():
        raise ValueError("report output must not be a symbolic link")
    target = requested.resolve()
    if target.suffix.lower() != ".html":
        raise ValueError("report output must be an HTML path, not a source artifact")
    parent = target.parent
    if not parent.is_dir():
        raise ValueError("report output parent must be an existing directory")
    retrieval_output = data.output_dir.resolve()
    if not (
        target.is_relative_to(repo_root)
        or target.is_relative_to(retrieval_output)
    ):
        raise ValueError(
            "report output must remain inside the repository or retrieval output"
        )
    if target.exists() and not target.is_file():
        raise ValueError("report output must be a regular file path")
    if output_path is not None and target.exists() and target != default_target.resolve():
        with target.open("rb") as existing:
            prefix = existing.read(4096)
        if (
            not prefix.startswith(b"<!doctype html>\n")
            or b"<title>Competition retrieval debug report</title>" not in prefix
        ):
            raise ValueError("report output must not replace an existing non-report artifact")
    return target


def _atomic_write_report(target: Path, body: bytes) -> None:
    temporary_path: Path | None = None
    backup_path: Path | None = None
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            with target.open("rb") as previous:
                backup_path = _write_report_copy(target, previous, suffix=".bak")
            _fsync_directory(target.parent)
        try:
            os.replace(temporary_path, target)
            temporary_path = None
            replaced = True
            _fsync_directory(target.parent)
        except Exception as triggering_error:
            if replaced:
                try:
                    _restore_previous_report(target, backup_path)
                    if backup_path is not None:
                        _remove_backup_durably(backup_path)
                except Exception as rollback_error:
                    recovery = backup_path if backup_path is not None and backup_path.exists() else None
                    if recovery is not None:
                        backup_path = None
                    location = str(recovery) if recovery is not None else "unavailable"
                    raise RuntimeError(
                        "atomic report output failed: "
                        f"{triggering_error}; rollback failed: {rollback_error}; "
                        f"previous report recovery preserved at {location}"
                    ) from rollback_error
                backup_path = None
            raise
        if backup_path is not None:
            try:
                _remove_backup_after_success(target, backup_path)
            except _ReportRecoveryError:
                backup_path = None
                raise
            backup_path = None
    finally:
        for candidate in (temporary_path, backup_path):
            if candidate is None:
                continue
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass


def _restore_previous_report(target: Path, backup_path: Path | None) -> None:
    if backup_path is None:
        target.unlink()
        _fsync_directory(target.parent)
        return
    rollback_path: Path | None = None
    try:
        with backup_path.open("rb") as backup:
            rollback_path = _write_report_copy(target, backup, suffix=".tmp")
        os.replace(rollback_path, target)
        rollback_path = None
        _fsync_directory(target.parent)
    finally:
        if rollback_path is not None:
            try:
                rollback_path.unlink()
            except FileNotFoundError:
                pass


def _remove_backup_after_success(target: Path, backup_path: Path) -> None:
    with backup_path.open("rb") as recovery_source:
        try:
            backup_path.unlink()
            _fsync_directory(target.parent)
        except Exception as triggering_error:
            recovery_path = backup_path
            if not recovery_path.exists():
                try:
                    recovery_path = _write_report_copy(
                        target, recovery_source, suffix=".bak"
                    )
                except Exception as recovery_error:
                    try:
                        recovery_path = _write_report_copy(
                            target, recovery_source, suffix=".bak"
                        )
                        _fsync_directory(target.parent)
                    except Exception as preservation_error:
                        retained = (
                            recovery_path
                            if recovery_path.exists()
                            else None
                        )
                        location = (
                            str(retained) if retained is not None else "unavailable"
                        )
                        raise _ReportRecoveryError(
                            "backup cleanup failed: "
                            f"{triggering_error}; recovery recreation failed: "
                            f"{recovery_error}; emergency recovery preservation failed: "
                            f"{preservation_error}; recovery material at {location}",
                            retained,
                        ) from preservation_error
                    raise _ReportRecoveryError(
                        "backup cleanup failed: "
                        f"{triggering_error}; recovery recreation failed: "
                        f"{recovery_error}; rollback not attempted; previous report "
                        "recovery preserved at "
                        f"{recovery_path}",
                        recovery_path,
                    ) from recovery_error
            try:
                _fsync_directory(target.parent)
                _restore_previous_report(target, recovery_path)
                _remove_backup_durably(recovery_path)
            except Exception as rollback_error:
                retained = recovery_path if recovery_path.exists() else None
                location = str(retained) if retained is not None else "unavailable"
                raise _ReportRecoveryError(
                    "backup cleanup failed: "
                    f"{triggering_error}; rollback failed: {rollback_error}; "
                    f"previous report recovery preserved at {location}",
                    retained,
                ) from rollback_error
            raise


class _ReportRecoveryError(RuntimeError):
    def __init__(self, message: str, recovery_path: Path | None) -> None:
        super().__init__(message)
        self.recovery_path = recovery_path


def _remove_backup_durably(backup_path: Path) -> None:
    backup_path.unlink()
    _fsync_directory(backup_path.parent)


def _write_report_copy(target: Path, source: BinaryIO, *, suffix: str) -> Path:
    source.seek(0)
    copy_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=suffix,
            delete=False,
        ) as copy:
            copy_path = Path(copy.name)
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                copy.write(chunk)
            copy.flush()
            os.fsync(copy.fileno())
        return copy_path
    except Exception as copy_error:
        if copy_path is None:
            raise
        try:
            copy_path.unlink()
            _fsync_directory(copy_path.parent)
        except Exception as cleanup_error:
            location = str(copy_path) if copy_path.exists() else "unavailable"
            raise RuntimeError(
                f"report copy failed: {copy_error}; partial copy cleanup failed: "
                f"{cleanup_error}; partial recovery material at {location}"
            ) from cleanup_error
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a private post-run competition debug report."
    )
    parser.add_argument("--retrieval-config", type=Path, required=True)
    parser.add_argument("--rag-config", type=Path)
    parser.add_argument("--topic", action="append", dest="topic_ids")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    receipt = build_debug_report(
        arguments.retrieval_config,
        rag_config_path=arguments.rag_config,
        topic_ids=arguments.topic_ids,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {
                "schema_version": receipt.schema_version,
                "output_path": str(receipt.output_path),
                "topic_ids": list(receipt.topic_ids),
                "rag_included": receipt.rag_included,
                "source_sha256s": dict(receipt.source_sha256s),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
