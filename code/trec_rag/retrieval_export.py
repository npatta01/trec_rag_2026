"""Fail-closed export of sealed facet-pilot artifacts to TREC retrieval runs."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

from trec_rag.facet_pilot_config import FacetPilotConfig
from trec_rag.facet_evidence import ExtractiveCandidate, SubnarrativeSelection
from trec_rag.canonical_nuggets import (
    CANONICAL_NUGGET_SCHEMA_VERSION,
    NUGGET_IMPORTANCE_VALUES,
    PROMPT_VERSION,
    RESULT_SCHEMA_VERSION,
)
from trec_rag.document_store import DocumentStore
from trec_rag.evidence_bundle import (
    BundleDocument,
    BundleLane,
    BundleSelection,
    BundleSelectionMember,
    EvidenceBundle,
    RetrievalEvent,
)
from trec_rag.topic_records import TOPIC_RECORDS_SCHEMA_VERSION
from trec_rag.facet_retrieval import (
    SELECTION_DEPTH,
)
from trec_rag.generation_handoff import (
    GenerationTopic,
    deserialize_generation_topic,
    serialize_generation_topic,
)
from trec_rag.generation_handoff_export import (
    ValidatedGenerationSnapshot,
    prepare_generation_handoff_artifact,
    project_generation_topic,
)
from trec_rag.topics import Topic, load_narrative_topics
from trec_rag.pipeline_models import jsonable
from trec_rag.topic_records import TopicRecords, TopicRecordsReceipt


_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXPORT_SCHEMA = "retrieval_export_manifest_v6"
_LEGACY_DIAGNOSTIC_ARTIFACTS = frozenset(
    {
        "retrieval_candidate_pool.trec",
        "retrieval_provenance.jsonl",
        "resolved_config.yaml",
    }
)
_RETRIEVAL_ARTIFACTS = frozenset(
    {
        "decomposition.json",
        "retrieval/audit.json",
        "retrieval/evidence-bundle.json",
    }
)
_RETRIEVAL_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "topic_id",
        "narrative_sha256",
        "decomposition_source_sha256",
        "code_commit",
        "retriever",
        "passage_search",
        "artifacts",
    }
)
_SCORING_ARTIFACTS = frozenset(
    {
        "scoring/lane_scores.jsonl",
        "scoring/selected_documents.jsonl",
        "scoring/selection.json",
        "scoring/selected_subnarrative_scores.jsonl",
    }
)
_CANONICAL_NUGGETS = "canonical/canonical-nuggets.jsonl"
_CANONICAL_ARTIFACTS = frozenset(
    {
        "canonical/handoff/candidate-requests.jsonl",
        "canonical/handoff/selection-contexts.jsonl",
        "canonical/handoff/handoff-manifest.json",
        "records.sqlite3",
        "canonical/records-manifest.json",
        "canonical/subnarrative-selections.jsonl",
        "canonical/selection-manifest.json",
        _CANONICAL_NUGGETS,
        "canonical/canonical-nugget-manifest.json",
    }
)
_CANONICAL_PROJECTION_ARTIFACTS = frozenset(
    {
        "canonical/retrieval-projection.json",
        "canonical/retrieval-projection-manifest.json",
        "canonical/generation-projection.json",
        "canonical/generation-projection-manifest.json",
    }
)
_PROJECTION_SCHEMA = "retrieval_projection_manifest_v4"
_PROJECTION_RECEIPT_SCHEMA = "retrieval_projection_receipt_v4"
_PROJECTION_FILE = "retrieval-projection.json"
_PROJECTION_MANIFEST_FILE = "retrieval-projection-manifest.json"
_GENERATION_PROJECTION_SCHEMA = "generation_topic_projection_v2"
_GENERATION_PROJECTION_RECEIPT_SCHEMA = "generation_projection_receipt_v2"
_GENERATION_PROJECTION_FILE = "generation-projection.json"
_GENERATION_PROJECTION_MANIFEST_FILE = "generation-projection-manifest.json"
_PROJECTION_NUGGET_FIELDS = frozenset(
    {
        "canonical_nugget_id",
        "nugget_kind",
        "claim_text",
        "importance",
        "subnarrative_id",
        "evidence",
    }
)
_PROJECTION_NUGGET_EVIDENCE_FIELDS = frozenset(
    {
        "candidate_nugget_id",
        "candidate_kind",
        "text",
        "text_sha256",
        "docid",
        "document_sha256",
        "cluster_id",
    }
)
_PROJECTION_SOURCE_SEALS = frozenset(
    {
        "config_sha256",
        "official_topics_sha256",
        "narrative_sha256",
        "decomposition_source_sha256",
        "decomposition_producer_sha256",
        "retrieval_manifest_sha256",
        "scoring_manifest_sha256",
        "handoff_manifest_sha256",
        "selection_manifest_sha256",
        "canonical_nugget_manifest_sha256",
        "topic_snapshot_sha256",
    }
)
_ROOT_TOPIC_SOURCE_SEALS = _PROJECTION_SOURCE_SEALS | frozenset(
    {
        "canonical_manifest_sha256",
        "source_code_commit",
        "organizer_projection_manifest_sha256",
        "generation_projection_manifest_sha256",
        "generation_context_sha256",
    }
)
_PROJECTION_SOURCE_SCHEMA_VERSIONS = frozenset(
    {
        "retrieval_manifest",
        "scoring_manifest",
        "canonical_manifest",
        "selection_manifest",
        "canonical_nugget_manifest",
        "records_manifest",
    }
)
_PROJECTION_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "topic_id",
        "projection_filename",
        "projection_sha256",
        "projection_bytes",
        "natural_document_count",
        "official_document_count",
        "retrieval_status",
        "retrieval_stopping_reason",
        "topic_snapshot_sha256",
        "source_code_commit",
        "source_schema_versions",
        "source_seals",
        "records_receipt",
    }
)
_PROJECTION_RECEIPT_FIELDS = _PROJECTION_MANIFEST_FIELDS | {
    "receipt_schema_version",
    "manifest_filename",
    "manifest_sha256",
    "manifest_bytes",
    "canonical_complete_sha256",
    "canonical_complete_bytes",
    "generation",
}
_GENERATION_PROJECTION_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "topic_id",
        "projection_filename",
        "projection_sha256",
        "projection_bytes",
        "context_sha256",
        "citation_document_count",
        "citation_docids_sha256",
        "source_code_commit",
        "source_schema_versions",
        "source_seals",
        "records_receipt",
    }
)
_GENERATION_PROJECTION_RECEIPT_FIELDS = _GENERATION_PROJECTION_MANIFEST_FIELDS | {
    "receipt_schema_version",
    "manifest_filename",
    "manifest_sha256",
    "manifest_bytes",
}
_PYSERINI_RETRIEVER_IDENTITY_FIELDS = frozenset(
    {
        "name",
        "type",
        "index",
        "index_url",
        "hits",
        "corpus_epoch",
        "retrieval_cache_schema",
        "parser_version",
        "extractor_version",
        "field_path",
        "scoring_normalizer_version",
    }
)
_SCORING_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "topic_id",
        "narrative_sha256",
        "decomposition_source_sha256",
        "code_commit",
        "selection_schema_version",
        "retriever",
        "passage_search",
        "retrieval_manifest_sha256",
        "scorer",
        "rerank_depth",
        "selection_k",
        "selection_policy",
        "selection_scope",
        "score_policy",
        "selected_set_sha256",
        "artifacts",
    }
)
_CANONICAL_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "topic_id",
        "official_topics_sha256",
        "narrative_sha256",
        "decomposition_source_sha256",
        "scoring_manifest_sha256",
        "handoff_manifest_sha256",
        "code_commit",
        "input_roles",
        "selection_policy",
        "selected_budget",
        "canonical_claim_cap",
        "canonical_supporting_document_cap",
        "artifacts",
    }
)
_SELECTED_DOCUMENT_FIELDS = frozenset(
    {
        "topic_id",
        "docid",
        "selection_rank",
        "selected_from_lane",
        "selected_from_lane_rank",
        "text_sha256",
        "text",
    }
)
_LANE_SCORE_FIELDS = frozenset(
    {
        "topic_id",
        "lane_name",
        "bm25_query_sha256",
        "semantic_query_sha256",
        "docid",
        "bm25_rank",
        "bm25_score",
        "aggregate_rank",
        "aggregate_score",
        "long_document_raw_logit",
        "weighted_passage_raw_logit",
        "within_document_span_support",
        "winning_passages",
        "score_representation",
        "text_sha256",
    }
)
_MEMBERSHIP_FIELDS = frozenset({"docid", "lanes"})
_MEMBERSHIP_LANE_FIELDS = frozenset(
    {"lane_name", "aggregate_rank", "aggregate_score", "bm25_rank", "bm25_score"}
)
_DOWNSTREAM_SCORE_FIELDS = frozenset(
    {
        "topic_id",
        "lane_name",
        "semantic_query_sha256",
        "docid",
        "bm25_rank",
        "bm25_score",
        "aggregate_rank",
        "aggregate_score",
        "long_document_raw_logit",
        "weighted_passage_raw_logit",
        "within_document_span_support",
        "winning_passages",
        "score_representation",
        "text_sha256",
        "selection_rank",
        "subnarrative_id",
        "bm25_queries",
        "bm25_query_sha256s",
        "downstream_only",
    }
)
_CANONICAL_NUGGET_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "result_schema_version",
        "canonical_response_schema_version",
        "selection_schema_version",
        "selection_manifest_schema_version",
        "selection_file",
        "selection_manifest_file",
        "canonical_nugget_file",
        "selections_sha256",
        "selection_manifest_sha256",
        "canonical_nuggets_sha256",
        "output_sha256",
        "selected_budget",
        "selection_count",
        "result_count",
        "state_counts",
        "max_canonical_claims",
        "max_supporting_documents_per_claim",
        "request_sha256s",
        "model",
        "prompt_version",
        "hosted_llm_calls",
        "validated_cache_hits",
        "raw_cache_writes",
        "validated_cache_writes",
    }
)
_CANONICAL_RESULT_STATES = frozenset({"complete", "empty", "fallback_extractive"})
_PASSAGE_FIELDS = frozenset(
    {"chunk_index", "start_char", "end_char", "raw_logit", "weighted_rank"}
)


@dataclass(frozen=True)
class RetrievalExportReceipt:
    official_run: Path
    with_text_archive: Path
    generation_handoff: Path
    manifest: Path


@dataclass(frozen=True)
class TopicRecordsProjectionReceipt:
    """The path-free, serializable receipt for one sealed TopicRecords input."""

    schema_version: str
    topic_id: str
    database_sha256: str
    database_bytes: int
    manifest_sha256: str
    manifest_bytes: int
    semantic_sha256: str
    document_sha256s: tuple[str, ...]
    row_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "topic_id": self.topic_id,
            "database_sha256": self.database_sha256,
            "database_bytes": self.database_bytes,
            "manifest_sha256": self.manifest_sha256,
            "manifest_bytes": self.manifest_bytes,
            "semantic_sha256": self.semantic_sha256,
            "document_sha256s": list(self.document_sha256s),
            "row_counts": {key: value for key, value in self.row_counts},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TopicRecordsProjectionReceipt":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "topic_id",
            "database_sha256",
            "database_bytes",
            "manifest_sha256",
            "manifest_bytes",
            "semantic_sha256",
            "document_sha256s",
            "row_counts",
        }:
            raise ValueError("topic records projection receipt fields changed")
        schema_version = value.get("schema_version")
        topic_id = value.get("topic_id")
        if (
            schema_version != TOPIC_RECORDS_SCHEMA_VERSION
            or not isinstance(topic_id, str)
            or not topic_id
        ):
            raise ValueError("topic records projection receipt identity changed")
        for key in ("database_sha256", "manifest_sha256", "semantic_sha256"):
            digest = value.get(key)
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError("topic records projection receipt digest changed")
        for key in ("database_bytes", "manifest_bytes"):
            count = value.get(key)
            if type(count) is not int or count < 0:
                raise ValueError("topic records projection receipt byte count changed")
        document_sha256s = value.get("document_sha256s")
        if (
            not isinstance(document_sha256s, list)
            or any(not isinstance(item, str) or not _SHA256.fullmatch(item) for item in document_sha256s)
            or len(set(document_sha256s)) != len(document_sha256s)
        ):
            raise ValueError("topic records projection document closure changed")
        row_counts = value.get("row_counts")
        if (
            not isinstance(row_counts, dict)
            or any(
                not isinstance(key, str) or type(count) is not int or count < 0
                for key, count in row_counts.items()
            )
        ):
            raise ValueError("topic records projection row counts changed")
        return cls(
            schema_version,
            topic_id,
            value["database_sha256"],
            value["database_bytes"],
            value["manifest_sha256"],
            value["manifest_bytes"],
            value["semantic_sha256"],
            tuple(document_sha256s),
            tuple(sorted(row_counts.items())),
        )


@dataclass(frozen=True)
class GenerationProjectionReceipt:
    """Path-free receipt for one source-free selected-evidence projection."""

    receipt_schema_version: str
    topic_id: str
    projection_filename: str
    projection_sha256: str
    projection_bytes: int
    manifest_filename: str
    manifest_sha256: str
    manifest_bytes: int
    context_sha256: str
    citation_document_count: int
    citation_docids_sha256: str
    source_code_commit: str
    source_schema_versions: tuple[tuple[str, str], ...]
    source_seals: tuple[tuple[str, str], ...]
    records_receipt: TopicRecordsProjectionReceipt

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_schema_version": self.receipt_schema_version,
            "schema_version": _GENERATION_PROJECTION_SCHEMA,
            "phase": "generation_projection",
            "topic_id": self.topic_id,
            "projection_filename": self.projection_filename,
            "projection_sha256": self.projection_sha256,
            "projection_bytes": self.projection_bytes,
            "manifest_filename": self.manifest_filename,
            "manifest_sha256": self.manifest_sha256,
            "manifest_bytes": self.manifest_bytes,
            "context_sha256": self.context_sha256,
            "citation_document_count": self.citation_document_count,
            "citation_docids_sha256": self.citation_docids_sha256,
            "source_code_commit": self.source_code_commit,
            "source_schema_versions": dict(self.source_schema_versions),
            "source_seals": dict(self.source_seals),
            "records_receipt": self.records_receipt.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "GenerationProjectionReceipt":
        if not isinstance(value, Mapping) or set(value) != (
            _GENERATION_PROJECTION_RECEIPT_FIELDS
        ):
            raise ValueError("generation projection receipt fields changed")
        if (
            value.get("receipt_schema_version")
            != _GENERATION_PROJECTION_RECEIPT_SCHEMA
            or value.get("schema_version") != _GENERATION_PROJECTION_SCHEMA
            or value.get("phase") != "generation_projection"
            or value.get("projection_filename") != _GENERATION_PROJECTION_FILE
            or value.get("manifest_filename")
            != _GENERATION_PROJECTION_MANIFEST_FILE
        ):
            raise ValueError("generation projection receipt identity changed")
        topic_id = value.get("topic_id")
        if not isinstance(topic_id, str) or not topic_id:
            raise ValueError("generation projection topic identity changed")
        for key in (
            "projection_sha256",
            "manifest_sha256",
            "context_sha256",
            "citation_docids_sha256",
        ):
            digest = value.get(key)
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError("generation projection digest changed")
        source_code_commit = value.get("source_code_commit")
        if not isinstance(source_code_commit, str) or not _COMMIT.fullmatch(
            source_code_commit
        ):
            raise ValueError("generation projection code identity changed")
        for key in (
            "projection_bytes",
            "manifest_bytes",
            "citation_document_count",
        ):
            count = value.get(key)
            if type(count) is not int or count < 0:
                raise ValueError("generation projection count changed")
        source_schema_versions = value.get("source_schema_versions")
        source_seals = value.get("source_seals")
        if (
            not isinstance(source_schema_versions, dict)
            or set(source_schema_versions) != _PROJECTION_SOURCE_SCHEMA_VERSIONS
            or any(
                not isinstance(item, str) or not item
                for item in source_schema_versions.values()
            )
            or not isinstance(source_seals, dict)
            or set(source_seals) != _PROJECTION_SOURCE_SEALS
            or any(
                not isinstance(item, str) or not _SHA256.fullmatch(item)
                for item in source_seals.values()
            )
        ):
            raise ValueError("generation projection source identity changed")
        records_receipt = TopicRecordsProjectionReceipt.from_dict(
            value["records_receipt"]  # type: ignore[arg-type]
        )
        if records_receipt.topic_id != topic_id:
            raise ValueError("generation projection records topic changed")
        return cls(
            _GENERATION_PROJECTION_RECEIPT_SCHEMA,
            topic_id,
            _GENERATION_PROJECTION_FILE,
            value["projection_sha256"],  # type: ignore[arg-type]
            value["projection_bytes"],  # type: ignore[arg-type]
            _GENERATION_PROJECTION_MANIFEST_FILE,
            value["manifest_sha256"],  # type: ignore[arg-type]
            value["manifest_bytes"],  # type: ignore[arg-type]
            value["context_sha256"],  # type: ignore[arg-type]
            value["citation_document_count"],  # type: ignore[arg-type]
            value["citation_docids_sha256"],  # type: ignore[arg-type]
            source_code_commit,
            tuple(sorted(source_schema_versions.items())),
            tuple(sorted(source_seals.items())),
            records_receipt,
        )


def _retrieval_completion(status: object, stopping_reason: object) -> tuple[str, str]:
    if status == "complete" and stopping_reason == "coverage_sufficient":
        return status, stopping_reason
    if status == "incomplete" and stopping_reason in {
        "budget_exhausted",
        "hard_deadline",
        "retrieval_unavailable",
        "scoring_failed",
        "no_evidence",
        "evidence_validation_failed",
    }:
        assert isinstance(stopping_reason, str)
        return status, stopping_reason
    raise ValueError("retrieval completion state is invalid or unsealed")


@dataclass(frozen=True)
class TopicProjectionReceipt:
    """A path-free receipt that is safe to cross a process boundary."""

    receipt_schema_version: str
    topic_id: str
    projection_filename: str
    projection_sha256: str
    projection_bytes: int
    manifest_filename: str
    manifest_sha256: str
    manifest_bytes: int
    canonical_complete_sha256: str
    canonical_complete_bytes: int
    natural_document_count: int
    official_document_count: int
    retrieval_status: str
    retrieval_stopping_reason: str
    topic_snapshot_sha256: str
    source_code_commit: str
    source_schema_versions: tuple[tuple[str, str], ...]
    source_seals: tuple[tuple[str, str], ...]
    records_receipt: TopicRecordsProjectionReceipt
    generation: GenerationProjectionReceipt

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_schema_version": self.receipt_schema_version,
            "schema_version": _PROJECTION_SCHEMA,
            "phase": "retrieval_projection",
            "topic_id": self.topic_id,
            "projection_filename": self.projection_filename,
            "projection_sha256": self.projection_sha256,
            "projection_bytes": self.projection_bytes,
            "manifest_filename": self.manifest_filename,
            "manifest_sha256": self.manifest_sha256,
            "manifest_bytes": self.manifest_bytes,
            "canonical_complete_sha256": self.canonical_complete_sha256,
            "canonical_complete_bytes": self.canonical_complete_bytes,
            "natural_document_count": self.natural_document_count,
            "official_document_count": self.official_document_count,
            "retrieval_status": self.retrieval_status,
            "retrieval_stopping_reason": self.retrieval_stopping_reason,
            "topic_snapshot_sha256": self.topic_snapshot_sha256,
            "source_code_commit": self.source_code_commit,
            "source_schema_versions": {
                key: value for key, value in self.source_schema_versions
            },
            "source_seals": {key: value for key, value in self.source_seals},
            "records_receipt": self.records_receipt.to_dict(),
            "generation": self.generation.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TopicProjectionReceipt":
        if not isinstance(value, Mapping) or set(value) != _PROJECTION_RECEIPT_FIELDS:
            raise ValueError("topic projection receipt fields changed")
        if (
            value.get("receipt_schema_version") != _PROJECTION_RECEIPT_SCHEMA
            or value.get("schema_version") != _PROJECTION_SCHEMA
            or value.get("phase") != "retrieval_projection"
        ):
            raise ValueError("topic projection receipt schema changed")
        topic_id = value.get("topic_id")
        if not isinstance(topic_id, str) or not topic_id:
            raise ValueError("topic projection receipt topic identity changed")
        projection_filename = value.get("projection_filename")
        manifest_filename = value.get("manifest_filename")
        if (
            projection_filename != _PROJECTION_FILE
            or manifest_filename != _PROJECTION_MANIFEST_FILE
        ):
            raise ValueError("topic projection receipt filenames changed")
        for key in (
            "projection_sha256",
            "manifest_sha256",
            "canonical_complete_sha256",
            "topic_snapshot_sha256",
            "source_code_commit",
        ):
            item = value.get(key)
            pattern = _COMMIT if key == "source_code_commit" else _SHA256
            if not isinstance(item, str) or not pattern.fullmatch(item):
                raise ValueError("topic projection receipt identity changed")
        for key in (
            "projection_bytes",
            "manifest_bytes",
            "canonical_complete_bytes",
            "natural_document_count",
            "official_document_count",
        ):
            item = value.get(key)
            if type(item) is not int or item < 0:
                raise ValueError("topic projection receipt count changed")
        source_schema_versions = value.get("source_schema_versions")
        source_seals = value.get("source_seals")
        if (
            not isinstance(source_schema_versions, dict)
            or set(source_schema_versions) != _PROJECTION_SOURCE_SCHEMA_VERSIONS
            or any(not isinstance(item, str) or not item for item in source_schema_versions.values())
            or not isinstance(source_seals, dict)
            or set(source_seals) != _PROJECTION_SOURCE_SEALS
            or any(not isinstance(item, str) or not _SHA256.fullmatch(item) for item in source_seals.values())
        ):
            raise ValueError("topic projection receipt source seals changed")
        records_receipt = TopicRecordsProjectionReceipt.from_dict(value["records_receipt"])
        if records_receipt.topic_id != topic_id:
            raise ValueError("topic projection receipt records topic changed")
        retrieval_status, retrieval_stopping_reason = _retrieval_completion(
            value.get("retrieval_status"),
            value.get("retrieval_stopping_reason"),
        )
        if value["topic_snapshot_sha256"] != records_receipt.semantic_sha256:
            raise ValueError("topic projection snapshot seal changed")
        if (
            source_seals.get("topic_snapshot_sha256")
            != value["topic_snapshot_sha256"]
        ):
            raise ValueError("topic projection source snapshot seal changed")
        generation = GenerationProjectionReceipt.from_dict(
            value["generation"]  # type: ignore[arg-type]
        )
        if (
            generation.topic_id != topic_id
            or generation.source_code_commit != value["source_code_commit"]
            or generation.source_schema_versions
            != tuple(sorted(source_schema_versions.items()))
            or generation.source_seals != tuple(sorted(source_seals.items()))
            or generation.records_receipt != records_receipt
        ):
            raise ValueError("paired projection receipt identity changed")
        return cls(
            _PROJECTION_RECEIPT_SCHEMA,
            topic_id,
            projection_filename,
            value["projection_sha256"],
            value["projection_bytes"],
            manifest_filename,
            value["manifest_sha256"],
            value["manifest_bytes"],
            value["canonical_complete_sha256"],
            value["canonical_complete_bytes"],
            value["natural_document_count"],
            value["official_document_count"],
            retrieval_status,
            retrieval_stopping_reason,
            value["topic_snapshot_sha256"],
            value["source_code_commit"],
            tuple(sorted(source_schema_versions.items())),
            tuple(sorted(source_seals.items())),
            records_receipt,
            generation,
        )


@dataclass(frozen=True)
class _SelectedDocument:
    docid: str
    rank: int
    text: str
    text_sha256: str
    selected_from_lane: str
    selected_from_lane_rank: int


@dataclass(frozen=True)
class _TopicProjection:
    topic: Topic
    bundle: EvidenceBundle
    selected: tuple[_SelectedDocument, ...]
    supported_docids: frozenset[str]
    original_only_fallback: bool
    memberships: Mapping[str, tuple[dict[str, Any], ...]]
    scores: Mapping[str, tuple[dict[str, Any], ...]]
    nuggets: Mapping[str, tuple[dict[str, Any], ...]]
    scoring_manifest_sha256: str
    canonical_manifest_sha256: str
    retrieval_status: str
    retrieval_stopping_reason: str
    topic_snapshot_sha256: str
    source_code_commit: str
    generation_topic: GenerationTopic
    source_schema_versions: tuple[tuple[str, str], ...]
    source_seals: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _ValidatedEvidenceProjection:
    selections: tuple[SubnarrativeSelection, ...]
    candidates: Mapping[tuple[str, str], ExtractiveCandidate]
    canonical_requests: tuple[Any, ...]
    allowed_evidence: frozenset[tuple[object, ...]]
    retrieval_topic_sha256: str


def _passage_search_identity_for_export(
    config: FacetPilotConfig,
    topic: Topic,
) -> dict[str, object]:
    path = config.output_dir / topic.id / "scoring" / "complete.json"
    manifest = _manifest(path.read_bytes(), "scoring checkpoint manifest")
    identity = manifest.get("passage_search")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("scoring checkpoint passage search identity is invalid")
    try:
        canonical = _canonical_json_bytes(identity)
        restored = _strict_json(canonical, "passage search identity")
    except (TypeError, ValueError) as exc:
        raise ValueError("passage search identity is not canonical JSON") from exc
    if restored != identity:
        raise ValueError("passage search identity changed during canonicalization")
    return dict(identity)


def export_retrieval_run(
    config: FacetPilotConfig,
    topics: Sequence[Topic],
    projection_receipts: Sequence[TopicProjectionReceipt],
    *,
    code_commit: str,
) -> RetrievalExportReceipt:
    """Publish global outputs from already sealed per-topic projections only."""
    if not isinstance(code_commit, str) or not _COMMIT.fullmatch(code_commit):
        raise ValueError("code commit must be a full lowercase SHA-1")
    selected_topics = tuple(topics)
    _validate_topics(selected_topics)
    receipts = tuple(projection_receipts)
    ordered_receipts = _validate_projection_receipt_coverage(
        config, selected_topics, receipts
    )
    official_rows: list[tuple[str, str, int, int, str]] = []
    with_text_rows: list[dict[str, object]] = []
    generation_topics: list[GenerationTopic] = []
    topic_depths: dict[str, dict[str, int]] = {}
    topic_statuses: dict[str, dict[str, str]] = {}
    topic_receipts: list[dict[str, str]] = []
    source_seals: dict[str, dict[str, str]] = {}
    source_commits: set[str] = set()
    passage_search_identity: dict[str, object] | None = None
    for topic, receipt in zip(selected_topics, ordered_receipts, strict=True):
        _verify_canonical_completion_receipt(config, topic, receipt)
        projection_row = _read_projection_for_export(config, topic, receipt)
        generation_topic = _read_generation_projection_for_export(
            config,
            topic,
            receipt.generation,
            organizer_row=projection_row,
        )
        topic_passage_identity = _passage_search_identity_for_export(config, topic)
        if passage_search_identity is None:
            passage_search_identity = topic_passage_identity
        elif passage_search_identity != topic_passage_identity:
            raise ValueError("selected topics have different passage search identities")
        candidates = projection_row["candidates"]
        if not isinstance(candidates, list):
            raise ValueError("retrieval projection candidates changed")
        official = tuple(
            (
                topic.id,
                str(candidate["docid"]),
                int(candidate["rank"]),
                int(candidate["score"]),
                config.run_id,
            )
            for candidate in candidates
        )
        if not official:
            raise ValueError(f"selected topic {topic.id!r} has no supported document")
        official_rows.extend(official)
        with_text_rows.append(projection_row)
        generation_topics.append(generation_topic)
        topic_depths[topic.id] = {
            "natural_union": receipt.natural_document_count,
            "official": receipt.official_document_count,
        }
        topic_statuses[topic.id] = {
            "status": receipt.retrieval_status,
            "stopping_reason": receipt.retrieval_stopping_reason,
        }
        topic_receipts.append(
            {
                "topic_id": topic.id,
                "projection_manifest_sha256": receipt.manifest_sha256,
            }
        )
        source_seals[topic.id] = {
            **dict(receipt.source_seals),
            "canonical_manifest_sha256": receipt.canonical_complete_sha256,
            "source_code_commit": receipt.source_code_commit,
            "organizer_projection_manifest_sha256": receipt.manifest_sha256,
            "generation_projection_manifest_sha256": (
                receipt.generation.manifest_sha256
            ),
            "generation_context_sha256": generation_topic.context_sha256,
        }
        source_commits.add(receipt.source_code_commit)

    official_bytes = _trec_bytes(official_rows)
    _validate_trec_bytes(official_bytes, official_rows)
    with_text_jsonl = _jsonl_bytes(with_text_rows)
    archive_bytes = _deterministic_zip(with_text_jsonl)
    if passage_search_identity is None:
        raise ValueError("selected topics require a passage search identity")

    output_dir = config.output_dir
    official_run = output_dir / "r_output_trec_rag_2026.tsv"
    with_text_archive = output_dir / "retrieval_with_text.jsonl.zip"
    prepared_handoff = prepare_generation_handoff_artifact(
        output_dir=output_dir,
        retrieval_run_id=config.run_id,
        producer_revision=code_commit,
        topics=tuple(generation_topics),
    )
    generation_handoff = prepared_handoff.path
    manifest = output_dir / "retrieval_export_manifest.json"
    artifacts = {
        official_run.name: official_bytes,
        with_text_archive.name: archive_bytes,
        generation_handoff.name: prepared_handoff.body,
    }
    manifest_body = _canonical_json_bytes(
        {
            "schema_version": _EXPORT_SCHEMA,
            "export_code_commit": code_commit,
            "source_code_commits": sorted(source_commits),
            "run_id": config.run_id,
            "selected_topic_ids": [topic.id for topic in selected_topics],
            "score_semantics": "ordinal_selection_order",
            "execution": {"topic_workers": config.execution.topic_workers},
            "passage_search": passage_search_identity,
            "topic_depths": topic_depths,
            "topic_statuses": topic_statuses,
            "topic_receipts": topic_receipts,
            "source_seals": source_seals,
            "official_row_count": len(official_rows),
            "artifacts": {
                name: {"bytes": len(body), "sha256": sha256(body).hexdigest()}
                for name, body in artifacts.items()
            },
        },
        pretty=True,
    )
    topic_ids = tuple(topic.id for topic in selected_topics)
    # All source, receipt, row, TREC, ZIP, handoff, and manifest bytes are
    # validated before locking or mutating the global run namespace. The outer
    # manifest remains the completion marker.
    with _export_lock(output_dir):
        _reject_legacy_export_outputs(output_dir)
        _validate_existing_export(
            output_dir,
            run_id=config.run_id,
            required_topic_ids=topic_ids,
            allow_topic_expansion=True,
            topic_workers=config.execution.topic_workers,
        )
        if manifest.exists():
            existing_body = manifest.read_bytes()
            if existing_body == manifest_body and all(
                (output_dir / name).read_bytes() == body
                for name, body in artifacts.items()
            ):
                return RetrievalExportReceipt(
                    official_run=official_run,
                    with_text_archive=with_text_archive,
                    generation_handoff=generation_handoff,
                    manifest=manifest,
                )
            manifest.unlink()
            _fsync_directory(output_dir)
        for path, body in (
            (official_run, official_bytes),
            (with_text_archive, archive_bytes),
            (generation_handoff, prepared_handoff.body),
        ):
            _atomic_write(path, body)
        _atomic_write(manifest, manifest_body)
        _validate_existing_export(
            output_dir,
            run_id=config.run_id,
            required_topic_ids=topic_ids,
            topic_workers=config.execution.topic_workers,
        )
    return RetrievalExportReceipt(
        official_run=official_run,
        with_text_archive=with_text_archive,
        generation_handoff=generation_handoff,
        manifest=manifest,
    )


def _validate_projection_receipt_coverage(
    config: FacetPilotConfig,
    topics: tuple[Topic, ...],
    receipts: tuple[TopicProjectionReceipt, ...],
) -> tuple[TopicProjectionReceipt, ...]:
    if len(receipts) != len(topics):
        raise ValueError("projection receipt coverage differs from selected topics")
    topic_by_id = {topic.id: topic for topic in topics}
    by_topic: dict[str, TopicProjectionReceipt] = {}
    for receipt in receipts:
        if not isinstance(receipt, TopicProjectionReceipt):
            raise TypeError("projection_receipts must contain TopicProjectionReceipt values")
        _validate_projection_receipt_shape(receipt)
        if receipt.topic_id not in topic_by_id:
            raise ValueError("projection receipt contains a wrong topic")
        if receipt.topic_id in by_topic:
            raise ValueError("duplicate projection receipt topic")
        by_topic[receipt.topic_id] = receipt
    if set(by_topic) != set(topic_by_id):
        raise ValueError("projection receipt coverage is incomplete")
    return tuple(by_topic[topic.id] for topic in topics)


def _validate_projection_receipt_shape(receipt: TopicProjectionReceipt) -> None:
    if receipt.receipt_schema_version != _PROJECTION_RECEIPT_SCHEMA:
        raise ValueError("projection receipt schema changed")
    if (
        receipt.projection_filename != _PROJECTION_FILE
        or receipt.manifest_filename != _PROJECTION_MANIFEST_FILE
    ):
        raise ValueError("projection receipt filenames changed")
    for digest in (
        receipt.projection_sha256,
        receipt.manifest_sha256,
        receipt.canonical_complete_sha256,
        receipt.topic_snapshot_sha256,
        receipt.records_receipt.database_sha256,
        receipt.records_receipt.manifest_sha256,
        receipt.records_receipt.semantic_sha256,
    ):
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("projection receipt digest changed")
    if not isinstance(receipt.source_code_commit, str) or not _COMMIT.fullmatch(
        receipt.source_code_commit
    ):
        raise ValueError("projection receipt source code identity changed")
    for count in (
        receipt.projection_bytes,
        receipt.manifest_bytes,
        receipt.canonical_complete_bytes,
        receipt.natural_document_count,
        receipt.official_document_count,
        receipt.records_receipt.database_bytes,
        receipt.records_receipt.manifest_bytes,
    ):
        if type(count) is not int or count < 0:
            raise ValueError("projection receipt byte or count changed")
    if receipt.records_receipt.topic_id != receipt.topic_id:
        raise ValueError("projection receipt records topic changed")
    if not isinstance(receipt.generation, GenerationProjectionReceipt):
        raise ValueError("projection receipt generation binding changed")
    if receipt.topic_snapshot_sha256 != receipt.records_receipt.semantic_sha256:
        raise ValueError("projection receipt snapshot seal changed")
    _retrieval_completion(
        receipt.retrieval_status,
        receipt.retrieval_stopping_reason,
    )
    if dict(receipt.source_seals).keys() != _PROJECTION_SOURCE_SEALS:
        raise ValueError("projection receipt source seal set changed")
    if any(
        not isinstance(value, str) or not _SHA256.fullmatch(value)
        for value in dict(receipt.source_seals).values()
    ):
        raise ValueError("projection receipt source seal changed")
    if (
        dict(receipt.source_seals)["topic_snapshot_sha256"]
        != receipt.topic_snapshot_sha256
    ):
        raise ValueError("projection receipt source snapshot seal changed")
    if dict(receipt.source_schema_versions).keys() != _PROJECTION_SOURCE_SCHEMA_VERSIONS:
        raise ValueError("projection receipt source schema set changed")
    if any(
        not isinstance(value, str) or not value
        for value in dict(receipt.source_schema_versions).values()
    ):
        raise ValueError("projection receipt source schema version changed")
    GenerationProjectionReceipt.from_dict(receipt.generation.to_dict())
    if TopicProjectionReceipt.from_dict(receipt.to_dict()) != receipt:
        raise ValueError("projection receipt canonical form changed")


def _read_projection_for_export(
    config: FacetPilotConfig,
    topic: Topic,
    receipt: TopicProjectionReceipt,
) -> dict[str, object]:
    topic_root = config.output_dir / topic.id
    projection_path = topic_root / "canonical" / receipt.projection_filename
    manifest_path = topic_root / "canonical" / receipt.manifest_filename
    projection_bytes = projection_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    if (
        len(projection_bytes) != receipt.projection_bytes
        or sha256(projection_bytes).hexdigest() != receipt.projection_sha256
        or len(manifest_bytes) != receipt.manifest_bytes
        or sha256(manifest_bytes).hexdigest() != receipt.manifest_sha256
    ):
        raise ValueError("projection receipt artifact hash changed")
    manifest = _manifest(manifest_bytes, "retrieval projection manifest")
    expected_manifest = _projection_manifest_dict(receipt)
    if manifest_bytes != _canonical_json_bytes(expected_manifest):
        raise ValueError("retrieval projection manifest is not canonical")
    _validate_projection_manifest_sources(config, topic, manifest, receipt)
    if not projection_bytes.endswith(b"\n") or projection_bytes.count(b"\n") != 1:
        raise ValueError("retrieval projection must be exactly one final-LF JSONL row")
    row = _strict_json(projection_bytes, "retrieval projection line 1")
    if _canonical_json_bytes(row) != projection_bytes:
        raise ValueError("retrieval projection is not canonical JSONL")
    if not isinstance(row, dict):
        raise ValueError("retrieval projection must contain exactly one organizer row")
    _validate_projection_row(row, topic, receipt)
    return row


def _citation_docids_sha256(docids: Sequence[str]) -> str:
    body = json.dumps(list(docids), separators=(",", ":")).encode("utf-8")
    return sha256(body).hexdigest()


def _validate_generation_projection(
    body: bytes,
    topic: Topic,
    receipt: GenerationProjectionReceipt,
    *,
    organizer_row: Mapping[str, object],
) -> GenerationTopic:
    if (
        len(body) != receipt.projection_bytes
        or sha256(body).hexdigest() != receipt.projection_sha256
    ):
        raise ValueError("generation projection artifact hash changed")
    generation_topic = deserialize_generation_topic(body)
    candidates = organizer_row.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("organizer projection candidates changed")
    organizer_docids = tuple(str(row["docid"]) for row in candidates)
    source_seals = dict(receipt.source_seals)
    if (
        generation_topic.topic_id != topic.id
        or generation_topic.narrative != topic.narrative
        or generation_topic.context_sha256 != receipt.context_sha256
        or len(generation_topic.citation_docids)
        != receipt.citation_document_count
        or _citation_docids_sha256(generation_topic.citation_docids)
        != receipt.citation_docids_sha256
        or generation_topic.citation_docids != organizer_docids
        or generation_topic.source_receipts.retrieval_topic_sha256
        != receipt.records_receipt.semantic_sha256
        or generation_topic.source_receipts.official_topics_sha256
        != source_seals["official_topics_sha256"]
    ):
        raise ValueError("paired organizer/generation projection identity changed")
    return generation_topic


def _read_generation_projection_for_export(
    config: FacetPilotConfig,
    topic: Topic,
    receipt: GenerationProjectionReceipt,
    *,
    organizer_row: Mapping[str, object],
) -> GenerationTopic:
    canonical = config.output_dir / topic.id / "canonical"
    projection_bytes = (canonical / receipt.projection_filename).read_bytes()
    manifest_bytes = (canonical / receipt.manifest_filename).read_bytes()
    if (
        len(manifest_bytes) != receipt.manifest_bytes
        or sha256(manifest_bytes).hexdigest() != receipt.manifest_sha256
        or manifest_bytes
        != _canonical_json_bytes(_generation_projection_manifest_dict(receipt))
    ):
        raise ValueError("generation projection manifest receipt changed")
    manifest = _manifest(manifest_bytes, "generation projection manifest")
    if (
        set(manifest) != _GENERATION_PROJECTION_MANIFEST_FIELDS
        or manifest != _generation_projection_manifest_dict(receipt)
    ):
        raise ValueError("generation projection manifest fields changed")
    return _validate_generation_projection(
        projection_bytes,
        topic,
        receipt,
        organizer_row=organizer_row,
    )


def _verify_canonical_completion_receipt(
    config: FacetPilotConfig,
    topic: Topic,
    receipt: TopicProjectionReceipt,
) -> None:
    path = config.output_dir / topic.id / "canonical" / "complete.json"
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise ValueError("canonical completion receipt is missing") from exc
    if (
        len(body) != receipt.canonical_complete_bytes
        or sha256(body).hexdigest() != receipt.canonical_complete_sha256
    ):
        raise ValueError("canonical completion changed after projection")


def _validate_projection_manifest_sources(
    config: FacetPilotConfig,
    topic: Topic,
    manifest: Mapping[str, object],
    receipt: TopicProjectionReceipt,
) -> None:
    """Validate only the worker's immutable manifest/receipt binding.

    The worker already deep-validates TopicRecords and seals the source
    identities into this manifest.  The parent intentionally does not reopen,
    stream-hash, or otherwise reread those source artifacts.
    """
    if set(manifest) != _PROJECTION_MANIFEST_FIELDS:
        raise ValueError("retrieval projection manifest fields changed")
    expected = _projection_manifest_dict(receipt)
    if dict(manifest) != expected:
        raise ValueError("retrieval projection manifest receipt changed")


def _validate_projection_row(
    row: Mapping[str, object],
    topic: Topic,
    receipt: TopicProjectionReceipt,
) -> None:
    if set(row) != {
        "query",
        "candidates",
        "canonical_nuggets",
        "retrieval_status",
        "retrieval_stopping_reason",
        "topic_snapshot_sha256",
    }:
        raise ValueError("retrieval projection organizer row fields changed")
    query = row.get("query")
    candidates = row.get("candidates")
    if not isinstance(query, dict) or set(query) != {
        "qid",
        "selection_id",
        "text",
        "text_sha256",
    }:
        raise ValueError("retrieval projection query fields changed")
    narrative_digest = sha256(topic.narrative.encode("utf-8")).hexdigest()
    if (
        query.get("qid") != topic.id
        or query.get("selection_id") != "official"
        or query.get("text") != topic.narrative
        or query.get("text_sha256") != narrative_digest
    ):
        raise ValueError("retrieval projection narrative identity changed")
    canonical_nuggets = row.get("canonical_nuggets")
    if not isinstance(candidates, list) or len(candidates) != receipt.official_document_count:
        raise ValueError("retrieval projection official document count changed")
    if not isinstance(canonical_nuggets, dict):
        raise ValueError("retrieval projection canonical nugget extension is invalid")
    if (
        _retrieval_completion(
            row.get("retrieval_status"),
            row.get("retrieval_stopping_reason"),
        )
        != (receipt.retrieval_status, receipt.retrieval_stopping_reason)
        or row.get("topic_snapshot_sha256") != receipt.topic_snapshot_sha256
    ):
        raise ValueError("retrieval projection completion identity changed")
    seen: set[str] = set()
    previous_score: int | None = None
    for expected_rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict) or set(candidate) != {
            "doc",
            "docid",
            "lane_ids",
            "rank",
            "score",
            "text_sha256",
        }:
            raise ValueError("retrieval projection candidate fields changed")
        docid = candidate.get("docid")
        text = candidate.get("doc")
        lanes = candidate.get("lane_ids")
        rank = candidate.get("rank")
        score = candidate.get("score")
        text_digest = candidate.get("text_sha256")
        if (
            not isinstance(docid, str)
            or not docid
            or any(character.isspace() for character in docid)
            or docid in seen
            or not isinstance(text, str)
            or not text
            or sha256(text.encode("utf-8")).hexdigest() != text_digest
            or type(rank) is not int
            or rank != expected_rank
            or type(score) is not int
            or score != receipt.official_document_count - expected_rank + 1
            or (previous_score is not None and score >= previous_score)
            or not isinstance(lanes, list)
            or not lanes
            or any(not isinstance(lane, str) or not lane for lane in lanes)
            or len(set(lanes)) != len(lanes)
        ):
            raise ValueError("retrieval projection candidate ordering or hash changed")
        seen.add(docid)
        previous_score = score
    if set(canonical_nuggets) != seen:
        raise ValueError("retrieval projection canonical nugget document set changed")
    for docid, links in canonical_nuggets.items():
        if not isinstance(docid, str) or not isinstance(links, list):
            raise ValueError("retrieval projection canonical nugget extension is invalid")
        for link in links:
            _validate_projection_nugget_link(link, docid)


def _projection_document_records(
    projection: _TopicProjection,
) -> tuple[dict[str, object], ...]:
    """Add topic-owned canonical nugget labels without changing organizer core fields."""
    records = projection.bundle.to_document_records(selection_id="official")
    if len(records) != 1 or not isinstance(records[0].get("candidates"), list):
        raise ValueError("topic projection must contain one candidate row")
    row = dict(records[0])
    candidates = [dict(candidate) for candidate in records[0]["candidates"]]
    row["candidates"] = candidates
    row["retrieval_status"] = projection.retrieval_status
    row["retrieval_stopping_reason"] = projection.retrieval_stopping_reason
    row["topic_snapshot_sha256"] = projection.topic_snapshot_sha256
    row["canonical_nuggets"] = {
        str(candidate["docid"]): [dict(link) for link in projection.nuggets.get(str(candidate["docid"]), ())]
        for candidate in candidates
    }
    return (row,)


def _validate_projection_nugget_link(value: object, docid: str) -> None:
    if not isinstance(value, dict) or set(value) != _PROJECTION_NUGGET_FIELDS:
        raise ValueError("retrieval projection canonical nugget fields changed")
    if (
        not isinstance(value["canonical_nugget_id"], str)
        or not value["canonical_nugget_id"]
        or not isinstance(value["nugget_kind"], str)
        or not value["nugget_kind"]
        or not isinstance(value["claim_text"], str)
        or not value["claim_text"].strip()
        or value["importance"] not in NUGGET_IMPORTANCE_VALUES
        or not isinstance(value["subnarrative_id"], str)
        or not value["subnarrative_id"]
    ):
        raise ValueError("retrieval projection canonical nugget identity changed")
    evidence = value["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != _PROJECTION_NUGGET_EVIDENCE_FIELDS:
        raise ValueError("retrieval projection canonical nugget evidence changed")
    if (
        evidence["docid"] != docid
        or not isinstance(evidence["candidate_nugget_id"], str)
        or not evidence["candidate_nugget_id"]
        or not isinstance(evidence["candidate_kind"], str)
        or not evidence["candidate_kind"]
        or not isinstance(evidence["text"], str)
        or not evidence["text"].strip()
        or not isinstance(evidence["text_sha256"], str)
        or not _SHA256.fullmatch(evidence["text_sha256"])
        or sha256(evidence["text"].encode("utf-8")).hexdigest() != evidence["text_sha256"]
        or not isinstance(evidence["document_sha256"], str)
        or not _SHA256.fullmatch(evidence["document_sha256"])
        or not isinstance(evidence["cluster_id"], str)
        or not evidence["cluster_id"]
    ):
        raise ValueError("retrieval projection canonical nugget evidence identity changed")


def read_retrieval_export_receipt(
    config: FacetPilotConfig,
    topics: Sequence[Topic],
) -> RetrievalExportReceipt:
    """Re-read and validate a complete retrieval export before returning it."""
    selected_topics = tuple(topics)
    _validate_topics(selected_topics)
    manifest = config.output_dir / "retrieval_export_manifest.json"
    if not manifest.is_file():
        raise ValueError("retrieval export manifest is missing")
    _validate_existing_export(
        config.output_dir,
        run_id=config.run_id,
        required_topic_ids=tuple(topic.id for topic in selected_topics),
        topic_workers=config.execution.topic_workers,
    )
    return RetrievalExportReceipt(
        official_run=config.output_dir / "r_output_trec_rag_2026.tsv",
        with_text_archive=config.output_dir / "retrieval_with_text.jsonl.zip",
        generation_handoff=(
            config.output_dir / "generation_handoff_manifest.json"
        ),
        manifest=manifest,
    )


def read_topic_projection_receipt(
    config: FacetPilotConfig,
    topic: Topic,
) -> TopicProjectionReceipt:
    """Read and strictly validate one published topic projection receipt."""
    _validate_topics((topic,))
    topic_root = config.output_dir / topic.id
    canonical_root = topic_root / "canonical"
    complete_path = canonical_root / "complete.json"
    try:
        complete_bytes = complete_path.read_bytes()
    except OSError as exc:
        raise ValueError("expanded canonical checkpoint is missing") from exc
    complete_manifest = _manifest(complete_bytes, "canonical checkpoint manifest")
    if _canonical_json_bytes(complete_manifest) != complete_bytes:
        raise ValueError("canonical checkpoint manifest is not canonical")
    complete_artifacts = _artifact_paths(complete_manifest)
    if complete_artifacts == _CANONICAL_ARTIFACTS:
        raise ValueError("legacy canonical checkpoint is incompatible")
    if complete_artifacts != _CANONICAL_ARTIFACTS | _CANONICAL_PROJECTION_ARTIFACTS:
        raise ValueError("canonical checkpoint artifact set changed")

    try:
        projection_manifest_bytes = (
            canonical_root / _PROJECTION_MANIFEST_FILE
        ).read_bytes()
        projection_bytes = (canonical_root / _PROJECTION_FILE).read_bytes()
        generation_manifest_bytes = (
            canonical_root / _GENERATION_PROJECTION_MANIFEST_FILE
        ).read_bytes()
        generation_bytes = (
            canonical_root / _GENERATION_PROJECTION_FILE
        ).read_bytes()
    except OSError as exc:
        raise ValueError("retrieval projection checkpoint is incomplete") from exc
    projection_manifest = _manifest(
        projection_manifest_bytes, "retrieval projection manifest"
    )
    generation_manifest = _manifest(
        generation_manifest_bytes, "generation projection manifest"
    )
    generation_value = dict(generation_manifest)
    generation_value.update(
        {
            "receipt_schema_version": _GENERATION_PROJECTION_RECEIPT_SCHEMA,
            "manifest_filename": _GENERATION_PROJECTION_MANIFEST_FILE,
            "manifest_sha256": sha256(generation_manifest_bytes).hexdigest(),
            "manifest_bytes": len(generation_manifest_bytes),
        }
    )
    generation_receipt = GenerationProjectionReceipt.from_dict(generation_value)
    receipt_value = dict(projection_manifest)
    receipt_value.update(
        {
            "receipt_schema_version": _PROJECTION_RECEIPT_SCHEMA,
            "manifest_filename": _PROJECTION_MANIFEST_FILE,
            "manifest_sha256": sha256(projection_manifest_bytes).hexdigest(),
            "manifest_bytes": len(projection_manifest_bytes),
            "canonical_complete_sha256": sha256(complete_bytes).hexdigest(),
            "canonical_complete_bytes": len(complete_bytes),
            "generation": generation_receipt.to_dict(),
        }
    )
    receipt = TopicProjectionReceipt.from_dict(receipt_value)
    try:
        _validate_projection_complete(
            config,
            topic,
            complete_manifest,
            receipt,
        )
        organizer_row = _read_projection_for_export(config, topic, receipt)
        _read_generation_projection_for_export(
            config,
            topic,
            receipt.generation,
            organizer_row=organizer_row,
        )
    except OSError as exc:
        raise ValueError("retrieval projection checkpoint is incomplete") from exc
    if (
        len(projection_bytes) != receipt.projection_bytes
        or sha256(projection_bytes).hexdigest() != receipt.projection_sha256
        or len(generation_bytes) != receipt.generation.projection_bytes
        or sha256(generation_bytes).hexdigest()
        != receipt.generation.projection_sha256
    ):
        raise ValueError("projection receipt artifact hash changed")
    return receipt


def _validate_projection_complete(
    config: FacetPilotConfig,
    topic: Topic,
    complete_manifest: Mapping[str, object],
    receipt: TopicProjectionReceipt,
) -> None:
    topic_root = config.output_dir / topic.id
    scoring_bytes = (topic_root / "scoring" / "complete.json").read_bytes()
    scoring_manifest = _manifest(scoring_bytes, "scoring checkpoint manifest")
    scoring_commit = _validate_scoring_manifest(scoring_manifest, config, topic)
    canonical_commit = _validate_canonical_manifest(
        complete_manifest,
        config,
        topic,
        scoring_manifest,
        scoring_bytes,
    )
    if canonical_commit != scoring_commit or receipt.source_code_commit != scoring_commit:
        raise ValueError("projection source code identity changed")
    if _validate_receipts(topic_root, complete_manifest) != (
        _CANONICAL_ARTIFACTS | _CANONICAL_PROJECTION_ARTIFACTS
    ):
        raise ValueError("canonical checkpoint artifact set changed")

    records_manifest_path = topic_root / "canonical" / "records-manifest.json"
    records_manifest_bytes = records_manifest_path.read_bytes()
    records_manifest = _manifest(records_manifest_bytes, "topic records manifest")
    expected_records = TopicRecordsProjectionReceipt.from_dict(
        {
            "schema_version": records_manifest.get("schema_version"),
            "topic_id": records_manifest.get("topic_id"),
            "database_sha256": records_manifest.get("database_sha256"),
            "database_bytes": records_manifest.get("database_bytes"),
            "manifest_sha256": sha256(records_manifest_bytes).hexdigest(),
            "manifest_bytes": len(records_manifest_bytes),
            "semantic_sha256": records_manifest.get("semantic_sha256"),
            "document_sha256s": records_manifest.get("document_sha256s"),
            "row_counts": records_manifest.get("row_counts"),
        }
    )
    if expected_records != receipt.records_receipt:
        raise ValueError("projection topic records receipt changed")
    run_id = records_manifest.get("run_id")
    if run_id != config.run_id:
        raise ValueError("topic records manifest run identity changed")
    records_source = TopicRecordsReceipt(
        database_sha256=expected_records.database_sha256,
        database_bytes=expected_records.database_bytes,
        semantic_sha256=expected_records.semantic_sha256,
        topic_id=expected_records.topic_id,
        run_id=run_id,
        document_sha256s=expected_records.document_sha256s,
        row_counts=dict(expected_records.row_counts),
        schema_version=expected_records.schema_version,
        manifest_sha256=expected_records.manifest_sha256,
        manifest_bytes=expected_records.manifest_bytes,
    )
    if tuple(sorted(_projection_source_schema_versions(
        topic_root, scoring_manifest, complete_manifest, records_source
    ))) != tuple(sorted(receipt.source_schema_versions)):
        raise ValueError("projection source schema versions changed")
    if tuple(sorted(_projection_source_seals(
        config,
        topic,
        topic_root,
        scoring_manifest,
        complete_manifest,
        config_sha256=dict(receipt.source_seals)["config_sha256"],
        topic_snapshot_sha256=expected_records.semantic_sha256,
        decomposition_producer_sha256=dict(receipt.source_seals)[
            "decomposition_producer_sha256"
        ],
    ))) != tuple(sorted(receipt.source_seals)):
        raise ValueError("projection source seals changed")


def _artifact_paths(manifest: Mapping[str, object]) -> set[str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("checkpoint artifacts changed")
    paths: set[str] = set()
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"relative_path", "bytes", "sha256"}
            or not isinstance(artifact.get("relative_path"), str)
            or artifact["relative_path"] in paths
        ):
            raise ValueError("checkpoint artifact receipt changed")
        paths.add(artifact["relative_path"])
    return paths


def validate_retrieval_topic_checkpoints(
    config: FacetPilotConfig,
    topics: Sequence[Topic],
    *,
    expected_retriever_identity: Mapping[str, object],
    expected_decomposition_producer_sha256: Mapping[str, str],
) -> tuple[TopicProjectionReceipt, ...]:
    """Deeply validate and return selected projection receipts in topic order."""
    expected_identity = _require_expected_retriever_identity(
        expected_retriever_identity
    )
    selected_topics = tuple(topics)
    if not selected_topics:
        return ()
    _validate_topics(selected_topics)
    expected_producers = dict(expected_decomposition_producer_sha256)
    if set(expected_producers) != {topic.id for topic in selected_topics} or any(
        not isinstance(value, str) or not _SHA256.fullmatch(value)
        for value in expected_producers.values()
    ):
        raise ValueError("expected decomposition producer seals are invalid")
    receipts: list[TopicProjectionReceipt] = []
    for topic in selected_topics:
        try:
            receipt = read_topic_projection_receipt(config, topic)
            rebuilt = _load_topic_projection(
                config,
                topic,
                config_sha256=dict(receipt.source_seals)["config_sha256"],
                expected_retriever_identity=expected_identity,
                decomposition_producer_sha256=expected_producers[topic.id],
            )
        except OSError as exc:
            raise ValueError("sealed topic checkpoint is incomplete") from exc
        organizer_rows = _projection_document_records(rebuilt)
        if len(organizer_rows) != 1:
            raise ValueError("rebuilt organizer projection row count changed")
        organizer_bytes = _jsonl_bytes(organizer_rows)
        if (
            len(organizer_bytes) != receipt.projection_bytes
            or sha256(organizer_bytes).hexdigest() != receipt.projection_sha256
        ):
            raise ValueError("rebuilt organizer projection differs from receipt")
        generation_bytes = serialize_generation_topic(rebuilt.generation_topic)
        if (
            len(generation_bytes) != receipt.generation.projection_bytes
            or sha256(generation_bytes).hexdigest()
            != receipt.generation.projection_sha256
        ):
            raise ValueError("rebuilt generation projection differs from receipt")
        receipts.append(receipt)
    return tuple(receipts)


def build_topic_projection(
    config: FacetPilotConfig,
    topic: Topic,
    records: TopicRecords,
    *,
    expected_retriever_identity: Mapping[str, object],
    decomposition_producer_sha256: str,
    config_sha256: str | None = None,
    canonical_manifest_bytes: bytes | None = None,
) -> TopicProjectionReceipt:
    """Build and create-only publish paired organizer/generation projections.

    ``records`` is deliberately an already opened handle.  This function never
    opens a database and never places the process-local validation capability in
    the returned receipt.
    """
    expected_identity = _require_expected_retriever_identity(
        expected_retriever_identity
    )
    if not isinstance(config_sha256, str) or not _SHA256.fullmatch(config_sha256):
        raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
    if (
        not isinstance(decomposition_producer_sha256, str)
        or not _SHA256.fullmatch(decomposition_producer_sha256)
    ):
        raise ValueError(
            "decomposition_producer_sha256 must be a lowercase SHA-256 digest"
        )
    if type(records) is not TopicRecords:
        raise TypeError("records must be an opened TopicRecords handle")
    if canonical_manifest_bytes is not None and not isinstance(
        canonical_manifest_bytes, bytes
    ):
        raise TypeError("canonical_manifest_bytes must be bytes or None")
    authoritative_records_receipt = TopicRecords.assert_current(records)
    _validate_topics((topic,))
    if authoritative_records_receipt.topic_id != topic.id:
        raise ValueError("opened TopicRecords topic does not match projection topic")
    _reject_existing_base_only_canonical_checkpoint(config, topic)
    try:
        projection = _load_topic_projection_from_records(
            config,
            topic,
            records,
            authoritative_records_receipt,
            config_sha256=config_sha256,
            canonical_manifest_bytes=canonical_manifest_bytes,
            expected_retriever_identity=expected_identity,
            decomposition_producer_sha256=decomposition_producer_sha256,
        )
    except OSError as exc:
        raise ValueError("sealed topic checkpoint is incomplete") from exc
    official_rows = _projection_document_records(projection)
    if len(official_rows) != 1:
        raise ValueError("topic projection must contain exactly one organizer row")
    projection_bytes = _jsonl_bytes(official_rows)
    generation_bytes = serialize_generation_topic(projection.generation_topic)
    records_receipt = _topic_records_projection_receipt(authoritative_records_receipt)
    generation_receipt = GenerationProjectionReceipt(
        receipt_schema_version=_GENERATION_PROJECTION_RECEIPT_SCHEMA,
        topic_id=topic.id,
        projection_filename=_GENERATION_PROJECTION_FILE,
        projection_sha256=sha256(generation_bytes).hexdigest(),
        projection_bytes=len(generation_bytes),
        manifest_filename=_GENERATION_PROJECTION_MANIFEST_FILE,
        manifest_sha256="0" * 64,
        manifest_bytes=0,
        context_sha256=projection.generation_topic.context_sha256,
        citation_document_count=len(projection.generation_topic.citation_docids),
        citation_docids_sha256=_citation_docids_sha256(
            projection.generation_topic.citation_docids
        ),
        source_code_commit=projection.source_code_commit,
        source_schema_versions=projection.source_schema_versions,
        source_seals=projection.source_seals,
        records_receipt=records_receipt,
    )
    generation_manifest_body = _canonical_json_bytes(
        _generation_projection_manifest_dict(generation_receipt)
    )
    generation_receipt = replace(
        generation_receipt,
        manifest_sha256=sha256(generation_manifest_body).hexdigest(),
        manifest_bytes=len(generation_manifest_body),
    )
    receipt = TopicProjectionReceipt(
        receipt_schema_version=_PROJECTION_RECEIPT_SCHEMA,
        topic_id=topic.id,
        projection_filename=_PROJECTION_FILE,
        projection_sha256=sha256(projection_bytes).hexdigest(),
        projection_bytes=len(projection_bytes),
        manifest_filename=_PROJECTION_MANIFEST_FILE,
        manifest_sha256="0" * 64,
        manifest_bytes=0,
        canonical_complete_sha256="0" * 64,
        canonical_complete_bytes=0,
        natural_document_count=projection.bundle.natural_document_count,
        official_document_count=len(official_rows[0]["candidates"]),
        retrieval_status=projection.retrieval_status,
        retrieval_stopping_reason=projection.retrieval_stopping_reason,
        topic_snapshot_sha256=projection.topic_snapshot_sha256,
        source_code_commit=projection.source_code_commit,
        source_schema_versions=projection.source_schema_versions,
        source_seals=projection.source_seals,
        records_receipt=records_receipt,
        generation=generation_receipt,
    )
    manifest_body = _canonical_json_bytes(
        _projection_manifest_dict(receipt)
    )
    receipt = replace(
        receipt,
        manifest_sha256=sha256(manifest_body).hexdigest(),
        manifest_bytes=len(manifest_body),
    )
    _validate_projection_row(official_rows[0], topic, receipt)
    _validate_generation_projection(
        generation_bytes,
        topic,
        generation_receipt,
        organizer_row=official_rows[0],
    )
    with _topic_projection_lock(config.output_dir / topic.id):
        _publish_topic_projection(
            config,
            topic,
            projection_bytes,
            manifest_body,
            generation_bytes,
            generation_manifest_body,
            receipt,
        )
        if canonical_manifest_bytes is None:
            canonical_manifest_bytes = (
                config.output_dir / topic.id / "canonical" / "complete.json"
            ).read_bytes()
        canonical_complete_bytes = _reseal_canonical_complete(
            config,
            topic,
            receipt,
            base_manifest_bytes=canonical_manifest_bytes,
        )
    return replace(
        receipt,
        canonical_complete_sha256=sha256(canonical_complete_bytes).hexdigest(),
        canonical_complete_bytes=len(canonical_complete_bytes),
    )


def _projection_manifest_dict(receipt: TopicProjectionReceipt) -> dict[str, object]:
    value = receipt.to_dict()
    for key in (
        "receipt_schema_version",
        "manifest_filename",
        "manifest_sha256",
        "manifest_bytes",
        "canonical_complete_sha256",
        "canonical_complete_bytes",
        "generation",
    ):
        value.pop(key, None)
    return value


def _generation_projection_manifest_dict(
    receipt: GenerationProjectionReceipt,
) -> dict[str, object]:
    value = receipt.to_dict()
    for key in (
        "receipt_schema_version",
        "manifest_filename",
        "manifest_sha256",
        "manifest_bytes",
    ):
        value.pop(key, None)
    return value


def _topic_records_projection_receipt(
    source: TopicRecordsReceipt,
) -> TopicRecordsProjectionReceipt:
    return TopicRecordsProjectionReceipt(
        source.schema_version,
        source.topic_id,
        source.database_sha256,
        source.database_bytes,
        source.manifest_sha256,
        source.manifest_bytes,
        source.semantic_sha256,
        tuple(source.document_sha256s),
        tuple(sorted((str(key), int(value)) for key, value in source.row_counts.items())),
    )


def _publish_topic_projection(
    config: FacetPilotConfig,
    topic: Topic,
    projection_bytes: bytes,
    manifest_bytes: bytes,
    generation_bytes: bytes,
    generation_manifest_bytes: bytes,
    receipt: TopicProjectionReceipt,
) -> None:
    canonical = config.output_dir / topic.id / "canonical"
    projection_path = canonical / _PROJECTION_FILE
    manifest_path = canonical / _PROJECTION_MANIFEST_FILE
    generation_path = canonical / _GENERATION_PROJECTION_FILE
    generation_manifest_path = canonical / _GENERATION_PROJECTION_MANIFEST_FILE
    if (
        receipt.projection_bytes != len(projection_bytes)
        or receipt.projection_sha256 != sha256(projection_bytes).hexdigest()
        or receipt.manifest_bytes != len(manifest_bytes)
        or receipt.manifest_sha256 != sha256(manifest_bytes).hexdigest()
        or receipt.generation.projection_bytes != len(generation_bytes)
        or receipt.generation.projection_sha256
        != sha256(generation_bytes).hexdigest()
        or receipt.generation.manifest_bytes != len(generation_manifest_bytes)
        or receipt.generation.manifest_sha256
        != sha256(generation_manifest_bytes).hexdigest()
    ):
        raise ValueError("topic projection receipt does not match publication bytes")
    canonical.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not projection_path.exists():
        raise ValueError("topic projection manifest exists without projection")
    if generation_manifest_path.exists() and not generation_path.exists():
        raise ValueError("generation projection manifest exists without projection")
    _publish_create_only(projection_path, projection_bytes)
    _publish_create_only(generation_path, generation_bytes)
    _publish_create_only(manifest_path, manifest_bytes)
    _publish_create_only(generation_manifest_path, generation_manifest_bytes)
    if (
        projection_path.read_bytes() != projection_bytes
        or manifest_path.read_bytes() != manifest_bytes
        or generation_path.read_bytes() != generation_bytes
        or generation_manifest_path.read_bytes() != generation_manifest_bytes
    ):
        raise ValueError("topic projection publication winner changed")


@contextmanager
def _topic_projection_lock(topic_root: Path):
    topic_root.mkdir(parents=True, exist_ok=True)
    with (topic_root / ".topic-projection.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _publish_create_only(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise ValueError("conflicting create-only topic projection winner")
        finally:
            temporary.unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reseal_canonical_complete(
    config: FacetPilotConfig,
    topic: Topic,
    receipt: TopicProjectionReceipt,
    *,
    base_manifest_bytes: bytes | None = None,
) -> bytes:
    path = config.output_dir / topic.id / "canonical" / "complete.json"
    if base_manifest_bytes is None:
        base_manifest_bytes = path.read_bytes()
    manifest = _manifest(base_manifest_bytes, "canonical checkpoint manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("canonical checkpoint artifacts are invalid")
    artifact_map = {
        item["relative_path"]: item
        for item in artifacts
        if isinstance(item, dict) and isinstance(item.get("relative_path"), str)
    }
    projection_receipts = {
        "canonical/retrieval-projection.json": _artifact_receipt(
            config.output_dir / topic.id, "canonical/retrieval-projection.json"
        ),
        "canonical/retrieval-projection-manifest.json": _artifact_receipt(
            config.output_dir / topic.id, "canonical/retrieval-projection-manifest.json"
        ),
        "canonical/generation-projection.json": _artifact_receipt(
            config.output_dir / topic.id, "canonical/generation-projection.json"
        ),
        "canonical/generation-projection-manifest.json": _artifact_receipt(
            config.output_dir / topic.id,
            "canonical/generation-projection-manifest.json",
        ),
    }
    expected_projection_receipts = {
        "canonical/retrieval-projection.json": {
            "relative_path": "canonical/retrieval-projection.json",
            "bytes": receipt.projection_bytes,
            "sha256": receipt.projection_sha256,
        },
        "canonical/retrieval-projection-manifest.json": {
            "relative_path": "canonical/retrieval-projection-manifest.json",
            "bytes": receipt.manifest_bytes,
            "sha256": receipt.manifest_sha256,
        },
        "canonical/generation-projection.json": {
            "relative_path": "canonical/generation-projection.json",
            "bytes": receipt.generation.projection_bytes,
            "sha256": receipt.generation.projection_sha256,
        },
        "canonical/generation-projection-manifest.json": {
            "relative_path": "canonical/generation-projection-manifest.json",
            "bytes": receipt.generation.manifest_bytes,
            "sha256": receipt.generation.manifest_sha256,
        },
    }
    if projection_receipts != expected_projection_receipts:
        raise ValueError("canonical checkpoint projection receipt changed")
    existing = set(artifact_map)
    if existing == _CANONICAL_ARTIFACTS:
        if path.exists():
            raise ValueError("legacy canonical checkpoint is incompatible")
        artifact_map.update(projection_receipts)
    elif existing != _CANONICAL_ARTIFACTS | _CANONICAL_PROJECTION_ARTIFACTS:
        raise ValueError("canonical checkpoint artifact set changed")
    if any(artifact_map[key] != value for key, value in projection_receipts.items()):
        raise ValueError("canonical checkpoint projection receipt changed")
    ordered = [item for item in artifacts if isinstance(item, dict)]
    if not any(item.get("relative_path") == "canonical/retrieval-projection.json" for item in ordered):
        ordered.extend(projection_receipts.values())
    manifest["artifacts"] = ordered
    body = _canonical_json_bytes(manifest)
    if path.exists():
        existing = path.read_bytes()
        if existing == body:
            return body
        if existing == base_manifest_bytes:
            raise ValueError("legacy canonical checkpoint is incompatible")
        if existing != base_manifest_bytes:
            raise ValueError("canonical checkpoint publication winner changed")
    _publish_create_only(path, body)
    return body


def _artifact_receipt(topic_root: Path, relative_path: str) -> dict[str, object]:
    body = (topic_root / relative_path).read_bytes()
    return {
        "relative_path": relative_path,
        "bytes": len(body),
        "sha256": sha256(body).hexdigest(),
    }


def _validate_topics(topics: tuple[Topic, ...]) -> None:
    if not topics:
        raise ValueError("at least one selected topic is required")
    seen: set[str] = set()
    for topic in topics:
        if not isinstance(topic, Topic) or not topic.id or not topic.narrative.strip():
            raise ValueError("selected topics must have non-empty identities and narratives")
        if topic.id in seen:
            raise ValueError(f"duplicate selected topic ID: {topic.id}")
        if (
            topic.id in {".", ".."}
            or Path(topic.id).name != topic.id
            or any(character.isspace() for character in topic.id)
        ):
            raise ValueError("topic ID must be a safe path component")
        seen.add(topic.id)


def _validate_existing_export(
    output_dir: Path,
    *,
    run_id: str,
    required_topic_ids: tuple[str, ...] | None,
    allow_topic_expansion: bool = False,
    topic_workers: int,
) -> None:
    manifest_path = output_dir / "retrieval_export_manifest.json"
    if not manifest_path.exists():
        return
    manifest = _manifest(manifest_path.read_bytes(), "existing export manifest")
    if manifest.get("schema_version") != _EXPORT_SCHEMA:
        raise ValueError("unsupported existing export manifest schema")
    expected_fields = {
        "schema_version",
        "export_code_commit",
        "source_code_commits",
        "run_id",
        "selected_topic_ids",
        "score_semantics",
        "execution",
        "passage_search",
        "topic_depths",
        "topic_statuses",
        "topic_receipts",
        "source_seals",
        "official_row_count",
        "artifacts",
    }
    recorded_topic_ids = manifest.get("selected_topic_ids")
    topic_depths = manifest.get("topic_depths")
    topic_statuses = manifest.get("topic_statuses")
    topic_receipts = manifest.get("topic_receipts")
    source_seals = manifest.get("source_seals")
    execution = manifest.get("execution")
    passage_search = manifest.get("passage_search")
    if (
        set(manifest) != expected_fields
        or manifest.get("run_id") != run_id
        or not isinstance(recorded_topic_ids, list)
        or not recorded_topic_ids
        or any(
            not isinstance(topic_id, str)
            or not topic_id
            or topic_id in {".", ".."}
            or Path(topic_id).name != topic_id
            or any(character.isspace() for character in topic_id)
            for topic_id in recorded_topic_ids
        )
        or len(set(recorded_topic_ids)) != len(recorded_topic_ids)
        or not isinstance(topic_depths, dict)
        or set(topic_depths) != set(recorded_topic_ids)
        or not isinstance(topic_statuses, dict)
        or set(topic_statuses) != set(recorded_topic_ids)
        or not isinstance(topic_receipts, list)
        or [
            row.get("topic_id") if isinstance(row, dict) else None
            for row in topic_receipts
        ]
        != recorded_topic_ids
        or not isinstance(source_seals, dict)
        or set(source_seals) != set(recorded_topic_ids)
        or execution != {"topic_workers": topic_workers}
        or not isinstance(passage_search, dict)
        or not passage_search
        or (
            required_topic_ids is not None
            and (
                not set(recorded_topic_ids) <= set(required_topic_ids)
                if allow_topic_expansion
                else recorded_topic_ids != list(required_topic_ids)
            )
        )
        or manifest.get("score_semantics") != "ordinal_selection_order"
    ):
        raise ValueError("existing export manifest identity changed")
    for topic_id in recorded_topic_ids:
        status = topic_statuses[topic_id]
        if not isinstance(status, dict) or set(status) != {
            "status",
            "stopping_reason",
        }:
            raise ValueError("existing export topic completion changed")
        _retrieval_completion(status.get("status"), status.get("stopping_reason"))
    for receipt in topic_receipts:
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"topic_id", "projection_manifest_sha256"}
            or not isinstance(receipt.get("projection_manifest_sha256"), str)
            or not _SHA256.fullmatch(receipt["projection_manifest_sha256"])
        ):
            raise ValueError("existing export topic receipt changed")
    for topic_id in recorded_topic_ids:
        topic_source_seals = source_seals[topic_id]
        if (
            not isinstance(topic_source_seals, dict)
            or set(topic_source_seals) != _ROOT_TOPIC_SOURCE_SEALS
            or not isinstance(topic_source_seals.get("source_code_commit"), str)
            or not _COMMIT.fullmatch(topic_source_seals["source_code_commit"])
            or any(
                not isinstance(topic_source_seals.get(key), str)
                or not _SHA256.fullmatch(topic_source_seals[key])
                for key in _ROOT_TOPIC_SOURCE_SEALS - {"source_code_commit"}
            )
        ):
            raise ValueError("existing export topic source seals changed")
        canonical_complete = output_dir / topic_id / "canonical" / "complete.json"
        try:
            canonical_complete_bytes = canonical_complete.read_bytes()
        except OSError as exc:
            raise ValueError("canonical completion changed after export") from exc
        if (
            sha256(canonical_complete_bytes).hexdigest()
            != topic_source_seals["canonical_manifest_sha256"]
        ):
            raise ValueError("canonical completion changed after export")
    export_commit = manifest.get("export_code_commit")
    source_commits = manifest.get("source_code_commits")
    if (
        not isinstance(export_commit, str)
        or not _COMMIT.fullmatch(export_commit)
        or not isinstance(source_commits, list)
        or not source_commits
        or source_commits != sorted(set(source_commits))
        or any(not isinstance(item, str) or not _COMMIT.fullmatch(item) for item in source_commits)
    ):
        raise ValueError("existing export revision identity changed")
    artifacts = manifest.get("artifacts")
    expected_names = {
        "generation_handoff_manifest.json",
        "r_output_trec_rag_2026.tsv",
        "retrieval_with_text.jsonl.zip",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise ValueError("existing export artifact receipts changed")
    for name in sorted(expected_names):
        receipt = artifacts[name]
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"bytes", "sha256"}
            or type(receipt.get("bytes")) is not int
            or receipt["bytes"] < 0
            or not isinstance(receipt.get("sha256"), str)
            or not _SHA256.fullmatch(receipt["sha256"])
        ):
            raise ValueError("existing export artifact receipt changed")
        body = (output_dir / name).read_bytes()
        if (
            len(body) != receipt["bytes"]
            or sha256(body).hexdigest() != receipt["sha256"]
        ):
            raise ValueError("existing export artifact hash changed")


def _reject_legacy_export_outputs(output_dir: Path) -> None:
    """Fail closed instead of deleting or migrating an old export namespace."""
    present = sorted(
        name
        for name in _LEGACY_DIAGNOSTIC_ARTIFACTS
        if (output_dir / name).exists()
    )
    if present:
        raise ValueError(
            "legacy retrieval export artifacts are incompatible; use a clean "
            f"output namespace: {', '.join(present)}"
        )


def _reject_existing_base_only_canonical_checkpoint(
    config: FacetPilotConfig,
    topic: Topic,
) -> None:
    """Reject the pre-projection checkpoint instead of upgrading it in place."""
    path = config.output_dir / topic.id / "canonical" / "complete.json"
    if not path.exists():
        return
    body = path.read_bytes()
    manifest = _manifest(body, "canonical checkpoint manifest")
    if _canonical_json_bytes(manifest) != body:
        raise ValueError("canonical checkpoint manifest is not canonical")
    if _artifact_paths(manifest) == _CANONICAL_ARTIFACTS:
        raise ValueError("legacy canonical checkpoint is incompatible")


def _load_topic_projection(
    config: FacetPilotConfig,
    topic: Topic,
    *,
    config_sha256: str,
    expected_retriever_identity: Mapping[str, object] | None = None,
    decomposition_producer_sha256: str,
) -> _TopicProjection:
    from trec_rag.competition_retrieval import document_store_dir

    topic_root = config.output_dir / topic.id
    with TopicRecords.open(
        topic_root / "records.sqlite3",
        topic_root / "canonical" / "records-manifest.json",
        topic.id,
        DocumentStore(document_store_dir(config.root_dir)),
    ) as records:
        records_receipt = TopicRecords.assert_current(records)
        return _load_topic_projection_from_records(
            config,
            topic,
            records,
            records_receipt,
            config_sha256=config_sha256,
            expected_retriever_identity=expected_retriever_identity,
            decomposition_producer_sha256=decomposition_producer_sha256,
        )


def _load_topic_projection_from_records(
    config: FacetPilotConfig,
    topic: Topic,
    records: TopicRecords,
    records_receipt: TopicRecordsReceipt,
    *,
    config_sha256: str,
    canonical_manifest_bytes: bytes | None = None,
    expected_retriever_identity: Mapping[str, object] | None = None,
    decomposition_producer_sha256: str,
) -> _TopicProjection:
    if not isinstance(config_sha256, str) or not _SHA256.fullmatch(config_sha256):
        raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
    if (
        not isinstance(decomposition_producer_sha256, str)
        or not _SHA256.fullmatch(decomposition_producer_sha256)
    ):
        raise ValueError("decomposition producer SHA-256 is invalid")
    if records_receipt.topic_id != topic.id:
        raise ValueError("opened TopicRecords topic does not match projection topic")
    snapshot = records.topic_snapshot()
    retrieval_status, retrieval_stopping_reason = _retrieval_completion(
        snapshot.status,
        snapshot.stopping_reason,
    )
    topic_root = config.output_dir / topic.id
    scoring_path = topic_root / "scoring" / "complete.json"
    scoring_bytes = scoring_path.read_bytes()
    scoring_manifest = _manifest(scoring_bytes, "scoring checkpoint manifest")
    scoring_commit = _validate_scoring_manifest(
        scoring_manifest,
        config,
        topic,
        expected_retriever_identity=expected_retriever_identity,
    )
    scoring_receipts = _validate_receipts(topic_root, scoring_manifest)
    if scoring_receipts != _SCORING_ARTIFACTS:
        raise ValueError("scoring checkpoint artifact set changed")
    (
        original_only_fallback,
        decomposition,
        retrieval_audit,
        retrieval_bundle,
    ) = _validate_retrieval_source_chain(
        topic_root,
        scoring_manifest,
        config,
        topic,
        scoring_commit,
        expected_retriever_identity=expected_retriever_identity,
    )

    if canonical_manifest_bytes is None:
        canonical_bytes = (topic_root / "canonical" / "complete.json").read_bytes()
    else:
        if not isinstance(canonical_manifest_bytes, bytes):
            raise TypeError("canonical_manifest_bytes must be bytes or None")
        canonical_bytes = canonical_manifest_bytes
    canonical_manifest = _manifest(
        canonical_bytes, "canonical checkpoint manifest"
    )
    if _canonical_json_bytes(canonical_manifest) != canonical_bytes:
        raise ValueError("canonical checkpoint manifest is not canonical")
    canonical_commit = _validate_canonical_manifest(
        canonical_manifest,
        config,
        topic,
        scoring_manifest,
        scoring_bytes,
    )
    if canonical_commit != scoring_commit:
        raise ValueError("scoring and canonical checkpoint code identities differ")
    canonical_receipts = _validate_receipts(
        topic_root,
        canonical_manifest,
        records_receipt=records_receipt,
    )
    if canonical_receipts not in (
        _CANONICAL_ARTIFACTS,
        _CANONICAL_ARTIFACTS | _CANONICAL_PROJECTION_ARTIFACTS,
    ):
        raise ValueError("canonical checkpoint artifact set changed")
    handoff_bytes = (
        topic_root / "canonical" / "handoff" / "handoff-manifest.json"
    ).read_bytes()
    if canonical_manifest.get("handoff_manifest_sha256") != sha256(
        handoff_bytes
    ).hexdigest():
        raise ValueError("canonical checkpoint handoff seal changed")

    expected_empty_internal_selection = _audit_proves_empty_internal_selection(
        retrieval_audit
    )
    selected = _load_selected_documents(
        topic_root / "scoring" / "selected_documents.jsonl",
        topic.id,
        expected_empty=expected_empty_internal_selection,
    )
    selected_set_sha256 = sha256(
        json.dumps(
            [row.docid for row in selected], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if scoring_manifest.get("selected_set_sha256") != selected_set_sha256:
        raise ValueError("scoring checkpoint selected document set changed")
    lane_scores = _load_lane_scores(
        topic_root / "scoring" / "lane_scores.jsonl",
        topic.id,
        expected_empty=expected_empty_internal_selection,
    )
    from trec_rag.competition_retrieval import validate_scoring_selection

    memberships = validate_scoring_selection(
        (topic_root / "scoring" / "selection.json").read_bytes(),
        topic_id=topic.id,
        selected_documents=tuple(row.__dict__ for row in selected),
        lane_score_rows=tuple(lane_scores.values()),
        audit_lanes=retrieval_audit,
        rerank_depth=config.retrieval.documents_per_query,
        selected_set_sha256=selected_set_sha256,
    )
    cross_scores = _load_cross_scores(
        topic_root / "scoring" / "selected_subnarrative_scores.jsonl",
        topic.id,
        selected,
        expected_subnarratives=decomposition.result.subnarratives,
        expected_passage_results=retrieval_audit,
    )
    _validate_memberships(memberships, selected, lane_scores)
    if original_only_fallback:
        _validate_original_only_scoring(
            selected,
            lane_scores,
            cross_scores,
            expected_query_sha256=sha256(topic.narrative.encode("utf-8")).hexdigest(),
        )
    subnarrative_ids = _validate_cross_score_matrix(
        cross_scores,
        selected,
        expected_subnarrative_ids=frozenset(
            row.subnarrative_id for row in decomposition.result.subnarratives
        ),
        allow_empty=original_only_fallback,
    )
    candidate_bundle = retrieval_bundle.with_ranked_selection(
        selection_id="candidate_pool",
        document_ids=tuple(row.docid for row in selected),
        policy="scoring_selection",
    )
    nugget_manifest = _validate_canonical_nugget_manifest(
        topic_root,
        canonical_manifest,
        config,
        subnarrative_ids,
    )
    if original_only_fallback and any(
        nugget_manifest[field] != 0
        for field in (
            "hosted_llm_calls",
            "validated_cache_hits",
            "raw_cache_writes",
            "validated_cache_writes",
        )
    ):
        raise ValueError("original-only fallback canonical work must be empty")
    evidence_projection = _load_allowed_canonical_evidence(
        topic_root,
        topic,
        config,
        records,
        records_receipt,
        retrieval_bundle,
        decomposition.result.subnarratives,
        selected_budget=config.nuggets.evidence_budget_per_subnarrative,
        max_canonical_claims=config.nuggets.maximum_claims_per_subnarrative,
        max_supporting_documents_per_claim=(
            config.nuggets.maximum_supporting_documents_per_claim
        ),
    )
    _claim_supported_docids, nuggets, canonical_results = _load_supported_docids(
        topic_root / _CANONICAL_NUGGETS,
        topic.id,
        retrieval_bundle,
        expected_subnarrative_ids=subnarrative_ids,
        nugget_manifest=nugget_manifest,
        allowed_evidence=evidence_projection.allowed_evidence,
        requests=evidence_projection.canonical_requests,
    )
    if not evidence_projection.candidates:
        raise ValueError(f"selected topic {topic.id!r} has no supported document")
    generation_docids = {
        candidate.docid for candidate in evidence_projection.candidates.values()
    }
    ordered_generation_docids = _order_supported_docids(
        generation_docids,
        retrieval_audit,
    )
    generation_topic = project_generation_topic(
        ValidatedGenerationSnapshot(
            topic=topic,
            official_topics_sha256=_official_topics_sha256(config),
            retrieval_topic_sha256=evidence_projection.retrieval_topic_sha256,
            selections=evidence_projection.selections,
            candidates=evidence_projection.candidates,
            canonical_results=canonical_results,
            selected_budget=config.nuggets.evidence_budget_per_subnarrative,
            max_canonical_claims=config.nuggets.maximum_claims_per_subnarrative,
            max_supporting_documents_per_claim=(
                config.nuggets.maximum_supporting_documents_per_claim
            ),
            document_ranks={
                docid: rank
                for rank, docid in enumerate(ordered_generation_docids, start=1)
            },
            original_narrative_fallback=original_only_fallback,
        )
    )
    if (
        generation_topic.topic_id != topic.id
        or generation_topic.narrative != topic.narrative
    ):
        raise ValueError("generation projection topic identity changed")
    if not generation_topic.citation_docids:
        raise ValueError("selected evidence projection contains no citation document")
    retrieved_docids = {row.docid for row in retrieval_bundle.documents}
    if not set(generation_topic.citation_docids) <= retrieved_docids:
        raise ValueError("selected evidence cites a document outside sealed retrieval")
    membership_map = {
        row["docid"]: tuple(
            {
                "lane_name": lane["lane_name"],
                "aggregate_rank": lane["aggregate_rank"],
                "aggregate_score": lane["aggregate_score"],
                "bm25_rank": lane["bm25_rank"],
                "bm25_score": lane["bm25_score"],
            }
            for lane in row["lanes"]
        )
        for row in memberships
    }
    score_map = {
        document.docid: tuple(
            score for score in cross_scores if score.get("docid") == document.docid
        )
        for document in selected
    }
    final_document_ids = generation_topic.citation_docids
    bundle = candidate_bundle.with_ranked_selection(
        selection_id="official",
        document_ids=final_document_ids,
        policy="selected_cluster_support",
    )
    return _TopicProjection(
        topic,
        bundle,
        selected,
        frozenset(generation_topic.citation_docids),
        original_only_fallback,
        membership_map,
        score_map,
        nuggets,
        sha256(scoring_bytes).hexdigest(),
        sha256(canonical_bytes).hexdigest(),
        retrieval_status,
        retrieval_stopping_reason,
        records_receipt.semantic_sha256,
        scoring_commit,
        generation_topic,
        _projection_source_schema_versions(
            topic_root,
            scoring_manifest,
            canonical_manifest,
            records_receipt,
        ),
        _projection_source_seals(
            config,
            topic,
            topic_root,
            scoring_manifest,
            canonical_manifest,
            config_sha256=config_sha256,
            topic_snapshot_sha256=records_receipt.semantic_sha256,
            decomposition_producer_sha256=decomposition_producer_sha256,
        ),
    )


def _projection_source_schema_versions(
    topic_root: Path,
    scoring_manifest: Mapping[str, object],
    canonical_manifest: Mapping[str, object],
    records_receipt: TopicRecordsReceipt,
) -> tuple[tuple[str, str], ...]:
    retrieval = _manifest(
        (topic_root / "retrieval" / "complete.json").read_bytes(),
        "retrieval checkpoint manifest",
    )
    selection = _manifest(
        (topic_root / "canonical" / "selection-manifest.json").read_bytes(),
        "selection manifest",
    )
    nugget = _manifest(
        (topic_root / "canonical" / "canonical-nugget-manifest.json").read_bytes(),
        "canonical nugget manifest",
    )
    return tuple(
        sorted(
            {
                "retrieval_manifest": retrieval.get("schema_version"),
                "scoring_manifest": scoring_manifest.get("schema_version"),
                "canonical_manifest": canonical_manifest.get("schema_version"),
                "selection_manifest": selection.get("schema_version"),
                "canonical_nugget_manifest": nugget.get("schema_version"),
                "records_manifest": records_receipt.schema_version,
            }.items()
        )
    )


def _projection_source_seals(
    config: FacetPilotConfig,
    topic: Topic,
    topic_root: Path,
    scoring_manifest: Mapping[str, object],
    canonical_manifest: Mapping[str, object],
    *,
    config_sha256: str,
    topic_snapshot_sha256: str,
    decomposition_producer_sha256: str,
) -> tuple[tuple[str, str], ...]:
    seals = {
        "config_sha256": config_sha256,
        "official_topics_sha256": canonical_manifest.get("official_topics_sha256"),
        "narrative_sha256": canonical_manifest.get("narrative_sha256"),
        "decomposition_source_sha256": canonical_manifest.get(
            "decomposition_source_sha256"
        ),
        "decomposition_producer_sha256": decomposition_producer_sha256,
        "retrieval_manifest_sha256": scoring_manifest.get("retrieval_manifest_sha256"),
        "scoring_manifest_sha256": sha256(
            (topic_root / "scoring" / "complete.json").read_bytes()
        ).hexdigest(),
        "handoff_manifest_sha256": canonical_manifest.get("handoff_manifest_sha256"),
        "selection_manifest_sha256": sha256(
            (topic_root / "canonical" / "selection-manifest.json").read_bytes()
        ).hexdigest(),
        "canonical_nugget_manifest_sha256": sha256(
            (topic_root / "canonical" / "canonical-nugget-manifest.json").read_bytes()
        ).hexdigest(),
        "topic_snapshot_sha256": topic_snapshot_sha256,
    }
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in seals.values()):
        raise ValueError("projection source seals are invalid")
    return tuple(sorted(seals.items()))


def _manifest(body: bytes, label: str) -> dict[str, Any]:
    value = _strict_json(body, label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_source_manifest_identity(
    manifest: Mapping[str, object],
    *,
    expected_fields: frozenset[str],
    topic: Topic,
    phase: str,
) -> str:
    if set(manifest) != expected_fields or manifest.get("schema_version") != "facet_pilot_v2":
        raise ValueError(f"{phase} checkpoint schema changed")
    if manifest.get("topic_id") != topic.id or manifest.get("phase") != phase:
        raise ValueError(f"{phase} checkpoint topic or phase identity changed")
    commit = manifest.get("code_commit")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError(f"{phase} checkpoint code identity is invalid")
    return commit


def _validate_retriever_identity(
    value: object,
    *,
    expected: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("retriever identity is invalid")
    if value.get("type") == "pyserini_remote":
        missing = sorted(_PYSERINI_RETRIEVER_IDENTITY_FIELDS - set(value))
        if missing:
            raise ValueError(
                "retriever identity is incomplete: " + ", ".join(missing)
            )
    if expected is not None and dict(value) != dict(expected):
        raise ValueError("retriever identity changed")
    return value


def _require_expected_retriever_identity(
    value: object,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected_retriever_identity must be a mapping")
    return dict(value)


def _validate_scoring_manifest(
    manifest: Mapping[str, object],
    config: FacetPilotConfig,
    topic: Topic,
    *,
    expected_retriever_identity: Mapping[str, object] | None = None,
) -> str:
    commit = _validate_source_manifest_identity(
        manifest,
        expected_fields=_SCORING_MANIFEST_FIELDS,
        topic=topic,
        phase="score",
    )
    narrative_sha256 = sha256(topic.narrative.encode("utf-8")).hexdigest()
    retriever = manifest.get("retriever")
    passage_search = manifest.get("passage_search")
    from trec_rag.competition_retrieval import _configured_passage_search_identity

    expected_passage_search = _configured_passage_search_identity(
        retrieval_depth=config.retrieval.documents_per_query,
        passages_per_query=config.passage.passages_per_query,
        chunk_max_characters=config.passage.chunk_max_characters,
        chunk_overlap_characters=config.passage.chunk_overlap_characters,
        model=config.passage.model,
        device=config.passage.device,
    )
    if manifest.get("narrative_sha256") != narrative_sha256:
        raise ValueError("scoring checkpoint narrative identity is stale")
    _validate_retriever_identity(
        retriever,
        expected=expected_retriever_identity,
    )
    if (
        not isinstance(manifest.get("decomposition_source_sha256"), str)
        or not _SHA256.fullmatch(manifest["decomposition_source_sha256"])
        or not isinstance(retriever, Mapping)
        or retriever.get("hits") != config.retrieval.documents_per_query
        or (
            retriever.get("type") == "pyserini_remote"
            and retriever.get("index") != config.retrieval.index
        )
        or not isinstance(passage_search, Mapping)
        or dict(passage_search) != expected_passage_search
        or manifest.get("scorer") != {"source": "sealed_topic_passage_search"}
        or manifest.get("rerank_depth") != config.retrieval.documents_per_query
        or manifest.get("selection_k") != SELECTION_DEPTH
        or manifest.get("selection_scope")
        != "internal_fixed_path_projection_not_final_submission"
        or manifest.get("selection_schema_version") != "facet_pilot_selection_v2"
        or manifest.get("selection_policy")
        != "round_robin_lane_order_no_fusion"
    ):
        raise ValueError("scoring checkpoint configuration is incompatible with export")
    for digest_key in ("retrieval_manifest_sha256", "selected_set_sha256"):
        value = manifest.get(digest_key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError("scoring checkpoint digest identity is invalid")
    if manifest.get("score_policy") != _producer_score_policy():
        raise ValueError("scoring checkpoint score policy is invalid")
    return commit


def _producer_score_policy() -> dict[str, object]:
    return {
        "document_order": "best_passage_raw_logit_source_rank_docid",
        "passage_scores": "sealed_topic_passage_search",
    }


def _validate_retrieval_source_chain(
    topic_root: Path,
    scoring_manifest: Mapping[str, object],
    config: FacetPilotConfig,
    topic: Topic,
    source_commit: str,
    *,
    expected_retriever_identity: Mapping[str, object] | None = None,
) -> tuple[bool, Any, tuple[Any, ...], EvidenceBundle]:
    retrieval_path = topic_root / "retrieval" / "complete.json"
    retrieval_bytes = retrieval_path.read_bytes()
    if scoring_manifest.get("retrieval_manifest_sha256") != sha256(
        retrieval_bytes
    ).hexdigest():
        raise ValueError("scoring checkpoint retrieval manifest seal changed")
    retrieval = _manifest(retrieval_bytes, "retrieval checkpoint manifest")
    retrieval_commit = _validate_source_manifest_identity(
        retrieval,
        expected_fields=_RETRIEVAL_MANIFEST_FIELDS,
        topic=topic,
        phase="retrieve",
    )
    retriever = retrieval.get("retriever")
    _validate_retriever_identity(
        scoring_manifest.get("retriever"),
        expected=expected_retriever_identity,
    )
    _validate_retriever_identity(
        retriever,
        expected=expected_retriever_identity,
    )
    if (
        retrieval_commit != source_commit
        or retrieval.get("narrative_sha256")
        != sha256(topic.narrative.encode("utf-8")).hexdigest()
        or retrieval.get("decomposition_source_sha256")
        != scoring_manifest.get("decomposition_source_sha256")
        or retriever != scoring_manifest.get("retriever")
        or retrieval.get("passage_search")
        != scoring_manifest.get("passage_search")
        or not isinstance(retriever, Mapping)
        or retriever.get("hits") != config.retrieval.documents_per_query
        or (
            retriever.get("type") == "pyserini_remote"
            and retriever.get("index") != config.retrieval.index
        )
    ):
        raise ValueError("retrieval checkpoint identity is incompatible with export")
    if _validate_receipts(topic_root, retrieval) != _RETRIEVAL_ARTIFACTS:
        raise ValueError("retrieval checkpoint artifact set changed")
    try:
        retrieval_bundle_payload = _strict_json(
            (topic_root / "retrieval/evidence-bundle.json").read_bytes(),
            "retrieval evidence bundle",
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("retrieval evidence bundle is invalid") from exc

    from trec_rag.competition_retrieval import (
        decode_retrieval_audit,
        decode_retrieval_decomposition,
        load_validated_decomposition,
    )

    source_decomposition = load_validated_decomposition(
        topic, topic_root / "decomposition" / "result.json"
    )
    decomposition = decode_retrieval_decomposition(
        topic,
        (topic_root / "decomposition.json").read_bytes(),
        expected_source_sha256=scoring_manifest["decomposition_source_sha256"],
    )
    if (
        source_decomposition.source_sha256 != decomposition.source_sha256
        or source_decomposition.result.used_fallback
        != decomposition.result.used_fallback
        or source_decomposition.result.queries != decomposition.result.queries
        or source_decomposition.result.plan != decomposition.result.plan
        or source_decomposition.result.subnarratives
        != decomposition.result.subnarratives
    ):
        raise ValueError(
            "saved decomposition source differs from receipted canonical rendering"
        )
    audit = decode_retrieval_audit(
        topic,
        decomposition,
        (topic_root / "retrieval" / "audit.json").read_bytes(),
        requested_depth=config.retrieval.documents_per_query,
    )
    from trec_rag.competition_retrieval import document_store_dir

    try:
        retrieval_bundle = _decode_retrieval_evidence_bundle(
            retrieval_bundle_payload,
            topic,
            audit,
            DocumentStore(document_store_dir(config.root_dir)),
        )
    except (TypeError, ValueError, KeyError, OSError) as exc:
        raise ValueError("retrieval evidence bundle is invalid") from exc
    _validate_bundle_against_retrieval_audit(
        retrieval_bundle,
        topic,
        audit,
    )
    return decomposition.result.used_fallback, decomposition, audit, retrieval_bundle


def _decode_retrieval_evidence_bundle(
    payload: object,
    topic: Topic,
    audit: tuple[Any, ...],
    document_store: DocumentStore,
) -> EvidenceBundle:
    """Decode the active v2 compact passage bundle into the export view."""
    if not isinstance(payload, Mapping):
        raise ValueError("retrieval evidence bundle must be an object")
    if payload.get("schema_version") != "facet_passage_evidence_v2":
        return EvidenceBundle.from_dict(payload)
    if set(payload) != {"schema_version", "topic_id", "lanes"}:
        raise ValueError("v2 retrieval evidence bundle fields changed")
    if payload.get("topic_id") != topic.id or not isinstance(payload.get("lanes"), list):
        raise ValueError("v2 retrieval evidence bundle identity changed")
    lanes_payload = payload["lanes"]
    if len(lanes_payload) != len(audit):
        raise ValueError("v2 retrieval evidence bundle lane count changed")
    lane_rows: list[BundleLane] = []
    document_rows: dict[str, tuple[str, str, set[str]]] = {}
    events: list[dict[str, object]] = []
    from trec_rag.competition_retrieval import _passage_result_json

    for index, (raw, audited) in enumerate(zip(lanes_payload, audit, strict=True), start=1):
        if not isinstance(raw, dict) or set(raw) != {
            "lane_id", "lane_kind", "query_text", "query_text_sha256", "documents", "passages"
        }:
            raise ValueError("v2 retrieval evidence bundle lane fields changed")
        lane = audited.lane
        expected_lane_id = lane.retrieval_query.variant_name
        expected_kind = "narrative" if lane.subnarrative_id is None else "subnarrative"
        if (
            raw["lane_id"] != expected_lane_id
            or raw["lane_kind"] != expected_kind
            or raw["query_text"] != lane.scoring_query.query_text
            or raw["query_text_sha256"] != lane.semantic_query_sha256
            or not isinstance(raw["documents"], list)
            or not isinstance(raw["passages"], list)
        ):
            raise ValueError("v2 retrieval evidence bundle lane identity changed")
        passage_result = audited.passage_result
        if passage_result is None:
            raise ValueError("v2 retrieval evidence bundle lacks passage result")
        expected_documents = [jsonable(document) for document in passage_result.documents]
        expected_passages = _passage_result_json(passage_result)["passages"]
        if raw["documents"] != expected_documents or raw["passages"] != expected_passages:
            raise ValueError("v2 retrieval evidence bundle rows differ from passage audit")
        lane_rows.append(
            BundleLane(
                expected_lane_id,
                expected_kind,
                lane.scoring_query.query_text,
                lane.semantic_query_sha256,
                producer="topic_passage_search",
            )
        )
        for document in passage_result.documents:
            source = document_store.read_text(document.content_sha256)
            if sha256(source.encode("utf-8")).hexdigest() != document.content_sha256:
                raise ValueError("v2 retrieval evidence bundle source digest changed")
            prior = document_rows.get(document.docid)
            if prior is None:
                document_rows[document.docid] = (
                    source,
                    document.content_sha256,
                    {expected_lane_id},
                )
            elif prior[:2] != (source, document.content_sha256):
                raise ValueError("v2 retrieval evidence bundle document source conflicts")
            else:
                prior[2].add(expected_lane_id)
            events.append({
                "lane_id": expected_lane_id,
                "docid": document.docid,
                "rank": document.source_rank,
                "score": document.source_score,
                "event_id": (
                    f"retrieval.{index:06d}.{expected_lane_id}."
                    f"{document.source_rank:06d}"
                ),
            })
    lane_ids = tuple(sorted(row.lane_id for row in lane_rows))
    documents = tuple(
        BundleDocument(
            docid=docid,
            text=value[0],
            text_sha256=value[1],
            lane_ids=tuple(sorted(value[2])),
        )
        for docid, value in sorted(document_rows.items())
    )
    bundle = EvidenceBundle(
        topic_id=topic.id,
        lanes=tuple(lane_rows),
        documents=documents,
        retrieval_events=tuple(
            RetrievalEvent(
                event_id=str(event["event_id"]),
                lane_id=str(event["lane_id"]),
                docid=str(event["docid"]),
                rank=int(event["rank"]),
                score=float(event["score"]),
                retriever="topic_passage_search",
            )
            for event in events
        ),
        selections=(
            BundleSelection(
                "natural_union",
                lane_ids,
                tuple(BundleSelectionMember(docid=docid, included=True) for docid in sorted(document_rows)),
            ),
        ),
        natural_document_count=len(document_rows),
    )
    bundle.validate()
    return bundle


def _validate_bundle_against_retrieval_audit(
    bundle: EvidenceBundle,
    topic: Topic,
    audit: tuple[Any, ...],
) -> None:
    if bundle.topic_id != topic.id:
        raise ValueError("retrieval evidence bundle topic identity changed")
    narrative_lanes = tuple(
        lane for lane in bundle.lanes if lane.lane_kind == "narrative"
    )
    if len(narrative_lanes) != 1 or narrative_lanes[0].query_text != topic.narrative:
        raise ValueError("retrieval evidence bundle narrative identity changed")
    expected_events = {
        (lane.lane.retrieval_query.variant_name, candidate.docid): (
            candidate.bm25_rank,
            candidate.bm25_score,
            candidate.text_sha256,
        )
        for lane in audit
        for candidate in lane.candidates
    }
    documents = {document.docid: document for document in bundle.documents}
    actual_events = {
        (event.lane_id, event.docid): event
        for event in bundle.retrieval_events
    }
    if (
        len(actual_events) != len(bundle.retrieval_events)
        or len(expected_events)
        != sum(len(lane.candidates) for lane in audit)
        or set(actual_events) != set(expected_events)
    ):
        raise ValueError("retrieval evidence bundle events differ from retrieval audit")
    for key, (rank, score, text_sha256) in expected_events.items():
        event = actual_events[key]
        if (
            event.rank != rank
            or event.score != score
            or documents[event.docid].text_sha256 != text_sha256
        ):
            raise ValueError("retrieval evidence bundle event differs from retrieval audit")
    expected_queries = {
        lane.lane.retrieval_query.variant_name: lane.lane.retrieval_query.query_text
        for lane in audit
    }
    if {lane.lane_id for lane in bundle.lanes} != set(expected_queries):
        raise ValueError("retrieval evidence bundle lanes differ from retrieval audit")
    for lane in bundle.lanes:
        if lane.query_text != expected_queries[lane.lane_id]:
            raise ValueError("retrieval evidence bundle query differs from retrieval audit")


def _validate_canonical_manifest(
    manifest: Mapping[str, object],
    config: FacetPilotConfig,
    topic: Topic,
    scoring_manifest: Mapping[str, object],
    scoring_bytes: bytes,
) -> str:
    commit = _validate_source_manifest_identity(
        manifest,
        expected_fields=_CANONICAL_MANIFEST_FIELDS,
        topic=topic,
        phase="canonical",
    )
    expected_policy = {
        "budgets": [config.nuggets.evidence_budget_per_subnarrative],
        "precluster_limit": 10 * config.nuggets.evidence_budget_per_subnarrative,
        "semantic_threshold": 0.92,
        "mmr_lambda": 0.7,
    }
    if (
        manifest.get("narrative_sha256")
        != sha256(topic.narrative.encode("utf-8")).hexdigest()
    ):
        raise ValueError("canonical checkpoint narrative identity is stale")
    if (
        manifest.get("decomposition_source_sha256")
        != scoring_manifest.get("decomposition_source_sha256")
        or manifest.get("official_topics_sha256") != _official_topics_sha256(config)
        or manifest.get("scoring_manifest_sha256") != sha256(scoring_bytes).hexdigest()
        or manifest.get("selection_policy") != expected_policy
        or manifest.get("selected_budget")
        != config.nuggets.evidence_budget_per_subnarrative
        or manifest.get("canonical_claim_cap")
        != config.nuggets.maximum_claims_per_subnarrative
        or manifest.get("canonical_supporting_document_cap")
        != config.nuggets.maximum_supporting_documents_per_claim
        or manifest.get("input_roles")
        != [
            "official_topic_narrative",
            "validated_generated_decomposition",
            "sealed_scoring_checkpoint",
        ]
    ):
        raise ValueError("canonical checkpoint configuration is incompatible with export")
    handoff_digest = manifest.get("handoff_manifest_sha256")
    if not isinstance(handoff_digest, str) or not _SHA256.fullmatch(handoff_digest):
        raise ValueError("canonical checkpoint handoff digest is invalid")
    return commit


def _official_topics_sha256(config: FacetPilotConfig) -> str:
    topics = load_narrative_topics(config.topics_path)
    body = json.dumps(
        [{"id": topic.id, "narrative": topic.narrative} for topic in topics],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(body).hexdigest()


def _validate_canonical_nugget_manifest(
    topic_root: Path,
    canonical_checkpoint: Mapping[str, object],
    config: FacetPilotConfig,
    expected_subnarrative_ids: frozenset[str],
) -> dict[str, Any]:
    canonical_root = topic_root / "canonical"
    value = _manifest(
        (canonical_root / "canonical-nugget-manifest.json").read_bytes(),
        "canonical nugget manifest",
    )
    fixed_identity = {
        "schema_version": "canonical_nugget_manifest_v2",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "canonical_response_schema_version": CANONICAL_NUGGET_SCHEMA_VERSION,
        "selection_schema_version": "subnarrative_selection_v1",
        "selection_manifest_schema_version": "subnarrative_selection_manifest_v1",
        "selection_file": "subnarrative-selections.jsonl",
        "selection_manifest_file": "selection-manifest.json",
        "canonical_nugget_file": "canonical-nuggets.jsonl",
        "model": "deepseek/deepseek-v4-flash-20260423",
        "prompt_version": PROMPT_VERSION,
    }
    if (
        set(value) != _CANONICAL_NUGGET_MANIFEST_FIELDS
        or any(value.get(key) != expected for key, expected in fixed_identity.items())
    ):
        raise ValueError("canonical nugget manifest identity changed")

    nugget_bytes = (canonical_root / "canonical-nuggets.jsonl").read_bytes()
    expected_digests = {
        "selections_sha256": sha256(
            (canonical_root / "subnarrative-selections.jsonl").read_bytes()
        ).hexdigest(),
        "selection_manifest_sha256": sha256(
            (canonical_root / "selection-manifest.json").read_bytes()
        ).hexdigest(),
        "canonical_nuggets_sha256": sha256(nugget_bytes).hexdigest(),
        "output_sha256": sha256(nugget_bytes).hexdigest(),
    }
    if any(value.get(key) != digest for key, digest in expected_digests.items()):
        raise ValueError("canonical nugget manifest artifact seal changed")

    expected_count = len(expected_subnarrative_ids)
    if (
        value.get("selected_budget")
        != config.nuggets.evidence_budget_per_subnarrative
        or value.get("selected_budget") != canonical_checkpoint.get("selected_budget")
        or value.get("max_canonical_claims")
        != config.nuggets.maximum_claims_per_subnarrative
        or value.get("max_canonical_claims")
        != canonical_checkpoint.get("canonical_claim_cap")
        or value.get("max_supporting_documents_per_claim")
        != config.nuggets.maximum_supporting_documents_per_claim
        or value.get("max_supporting_documents_per_claim")
        != canonical_checkpoint.get("canonical_supporting_document_cap")
        or type(value.get("selection_count")) is not int
        or value["selection_count"] != expected_count
        or type(value.get("result_count")) is not int
        or value["result_count"] != expected_count
    ):
        raise ValueError("canonical nugget manifest configuration or counts changed")

    requests = value.get("request_sha256s")
    if (
        not isinstance(requests, list)
        or len(requests) != expected_count
        or any(type(item) is not str or not _SHA256.fullmatch(item) for item in requests)
        or len(set(requests)) != expected_count
    ):
        raise ValueError("canonical nugget manifest request digests changed")
    state_counts = value.get("state_counts")
    if (
        not isinstance(state_counts, dict)
        or not set(state_counts) <= _CANONICAL_RESULT_STATES
        or any(type(count) is not int or count < 0 for count in state_counts.values())
        or sum(state_counts.values()) != expected_count
    ):
        raise ValueError("canonical nugget manifest state counts changed")
    for field in (
        "hosted_llm_calls",
        "validated_cache_hits",
        "raw_cache_writes",
        "validated_cache_writes",
    ):
        if type(value.get(field)) is not int or value[field] < 0:
            raise ValueError("canonical nugget manifest run counters changed")
    return value


def _validate_receipts(
    topic_root: Path,
    manifest: Mapping[str, object],
    *,
    records_receipt: TopicRecordsReceipt | None = None,
) -> set[str]:
    receipts = manifest.get("artifacts")
    if not isinstance(receipts, list):
        raise ValueError("checkpoint artifacts changed")
    seen: set[str] = set()
    root = topic_root.resolve()
    authoritative_source_receipts = (
        {}
        if records_receipt is None
        else {
            "records.sqlite3": {
                "relative_path": "records.sqlite3",
                "bytes": records_receipt.database_bytes,
                "sha256": records_receipt.database_sha256,
            },
            "canonical/records-manifest.json": {
                "relative_path": "canonical/records-manifest.json",
                "bytes": records_receipt.manifest_bytes,
                "sha256": records_receipt.manifest_sha256,
            },
        }
    )
    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != {
            "relative_path",
            "bytes",
            "sha256",
        }:
            raise ValueError("checkpoint artifact receipt changed")
        relative = receipt["relative_path"]
        byte_count = receipt["bytes"]
        digest = receipt["sha256"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or Path(relative).is_absolute()
        ):
            raise ValueError("checkpoint artifact path changed")
        if relative in authoritative_source_receipts:
            if receipt != authoritative_source_receipts[relative]:
                raise ValueError("topic records source receipt changed")
            seen.add(relative)
            continue
        path = (topic_root / relative).resolve()
        if path == root or root not in path.parents:
            raise ValueError("checkpoint artifact path escapes topic root")
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError("checkpoint artifact byte count changed")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("checkpoint artifact digest changed")
        actual_bytes, actual_digest = _streamed_file_identity(path)
        if actual_bytes != byte_count or actual_digest != digest:
            raise ValueError("checkpoint artifact hash changed")
        seen.add(relative)
    return seen


def _streamed_file_identity(path: Path) -> tuple[int, str]:
    digest = sha256()
    byte_count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def _audit_proves_empty_internal_selection(
    retrieval_audit: Sequence[Any],
) -> bool:
    if not retrieval_audit:
        return False
    return all(
        audited.passage_result is not None
        and audited.passage_result.status == "incomplete"
        and audited.passage_result.scored_documents == 0
        and audited.passage_result.scored_passages == 0
        and not audited.passage_result.passages
        and all(
            document.best_passage_id is None
            and document.best_passage_raw_logit is None
            for document in audited.passage_result.documents
        )
        for audited in retrieval_audit
    )


def _load_selected_documents(
    path: Path,
    topic_id: str,
    *,
    expected_empty: bool = False,
) -> tuple[_SelectedDocument, ...]:
    records = _jsonl(path, "selected documents")
    rows: list[_SelectedDocument] = []
    seen: set[str] = set()
    for expected_rank, record in enumerate(records, start=1):
        if not isinstance(record, dict) or set(record) != _SELECTED_DOCUMENT_FIELDS:
            raise ValueError("selected document schema changed")
        docid = record.get("docid")
        rank = record.get("selection_rank")
        text = record.get("text")
        text_hash = record.get("text_sha256")
        selected_from_lane = record.get("selected_from_lane")
        selected_from_lane_rank = record.get("selected_from_lane_rank")
        if record.get("topic_id") != topic_id:
            raise ValueError("selected document topic identity changed")
        if (
            not isinstance(docid, str)
            or not docid
            or any(character.isspace() for character in docid)
            or docid in seen
        ):
            raise ValueError("duplicate or invalid selected topic-document pair")
        if type(rank) is not int or rank != expected_rank:
            raise ValueError("selected document ranks must be contiguous")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("selected document text must be non-empty")
        if text_hash != sha256(text.encode("utf-8")).hexdigest():
            raise ValueError("selected document text hash changed")
        if (
            not isinstance(selected_from_lane, str)
            or not selected_from_lane
            or isinstance(selected_from_lane_rank, bool)
            or not isinstance(selected_from_lane_rank, int)
            or selected_from_lane_rank <= 0
        ):
            raise ValueError("selected document lane provenance is invalid")
        seen.add(docid)
        rows.append(
            _SelectedDocument(
                docid,
                rank,
                text,
                text_hash,
                selected_from_lane,
                selected_from_lane_rank,
            )
        )
    if expected_empty and rows:
        raise ValueError("selected document checkpoint must be empty")
    if not rows and not expected_empty:
        raise ValueError("selected document checkpoint is empty")
    return tuple(rows)


def _load_lane_scores(
    path: Path,
    topic_id: str,
    *,
    expected_empty: bool = False,
) -> Mapping[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in _jsonl(path, "lane scores"):
        if not isinstance(record, dict) or set(record) != _LANE_SCORE_FIELDS:
            raise ValueError("lane score schema changed")
        docid = record.get("docid")
        lane_name = record.get("lane_name")
        key = (docid, lane_name)
        if (
            record.get("topic_id") != topic_id
            or not isinstance(docid, str)
            or not docid
            or not isinstance(lane_name, str)
            or not lane_name
            or key in result
        ):
            raise ValueError("lane score identity is invalid or duplicated")
        _validate_score_components(
            record,
            "lane score",
            allow_empty_passages=True,
        )
        result[key] = record
    if expected_empty and result:
        raise ValueError("lane score artifact must be empty")
    if not result and not expected_empty:
        raise ValueError("lane score artifact is empty")
    return result


def _validate_memberships(
    memberships: tuple[dict[str, Any], ...],
    selected: tuple[_SelectedDocument, ...],
    lane_scores: Mapping[tuple[str, str], dict[str, Any]],
) -> None:
    if [row.get("docid") for row in memberships] != [row.docid for row in selected]:
        raise ValueError("selection memberships do not match selected documents")
    selected_by_id = {row.docid: row for row in selected}
    for membership in memberships:
        if set(membership) != _MEMBERSHIP_FIELDS:
            raise ValueError("selection membership schema changed")
        lanes = membership.get("lanes")
        if not isinstance(lanes, list) or not lanes or any(
            not isinstance(lane, dict)
            or set(lane) != _MEMBERSHIP_LANE_FIELDS
            or not isinstance(lane.get("lane_name"), str)
            or not lane["lane_name"]
            for lane in lanes
        ):
            raise ValueError("selection membership lanes are invalid")
        if len({lane["lane_name"] for lane in lanes}) != len(lanes):
            raise ValueError("selection membership lane identities are duplicated")
        document = selected_by_id[membership["docid"]]
        sealed_lane_names = {
            lane_name
            for docid, lane_name in lane_scores
            if docid == document.docid
        }
        if {lane["lane_name"] for lane in lanes} != sealed_lane_names:
            raise ValueError("selection membership lane set differs from sealed lane scores")
        for lane in lanes:
            source = lane_scores.get((document.docid, lane["lane_name"]))
            if source is None or any(
                lane[field] != source[field]
                for field in (
                    "aggregate_rank",
                    "aggregate_score",
                    "bm25_rank",
                    "bm25_score",
                )
            ):
                raise ValueError("selection membership differs from sealed lane score")
        origin = next(
            (lane for lane in lanes if lane["lane_name"] == document.selected_from_lane),
            None,
        )
        if (
            origin is None
            or origin["aggregate_rank"] != document.selected_from_lane_rank
        ):
            raise ValueError("selected document origin differs from sealed lane score")


def _load_cross_scores(
    path: Path,
    topic_id: str,
    selected: tuple[_SelectedDocument, ...],
    *,
    expected_subnarratives: Sequence[Any],
    expected_passage_results: Sequence[Any] = (),
) -> tuple[dict[str, Any], ...]:
    expected_by_id = {
        row.subnarrative_id: row for row in expected_subnarratives
    }
    selected_by_id = {row.docid: row for row in selected}
    records = _jsonl(path, "selected subnarrative scores")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        if isinstance(record, dict) and record.get("schema_version") == "facet_passage_lane_v2":
            if set(record) != {"schema_version", "subnarrative_id", "passage_search"}:
                raise ValueError("passage lane score schema changed")
            subnarrative_id = record.get("subnarrative_id")
            if not isinstance(subnarrative_id, str) or subnarrative_id not in expected_by_id:
                raise ValueError("passage lane score identity is invalid")
            expected = next(
                (
                    audited.passage_result
                    for audited in expected_passage_results
                    if audited.lane.subnarrative_id == subnarrative_id
                ),
                None,
            )
            if expected is None:
                raise ValueError("passage lane score is missing its bound retrieval lane")
            from trec_rag.competition_retrieval import _decode_passage_result

            if _decode_passage_result(record["passage_search"]) != expected:
                raise ValueError("passage lane score differs from retrieval audit")
            rows.append(dict(record))
            continue
        if not isinstance(record, dict) or set(record) != _DOWNSTREAM_SCORE_FIELDS:
            raise ValueError("subnarrative score schema changed")
        docid = record.get("docid")
        subnarrative_id = record.get("subnarrative_id")
        document = selected_by_id.get(docid) if isinstance(docid, str) else None
        expected_subnarrative = expected_by_id.get(subnarrative_id)
        pair = (docid, subnarrative_id)
        if (
            record.get("topic_id") != topic_id
            or document is None
            or not isinstance(subnarrative_id, str)
            or expected_subnarrative is None
            or pair in seen
        ):
            raise ValueError("subnarrative score identity is invalid or duplicated")
        if (
            record.get("selection_rank") != document.rank
            or record.get("text_sha256") != document.text_sha256
        ):
            raise ValueError("subnarrative score document identity changed")
        _validate_score_components(record, "subnarrative score")
        if record.get("downstream_only") is not True:
            raise ValueError("subnarrative score must be downstream-only")
        queries = record.get("bm25_queries")
        query_hashes = record.get("bm25_query_sha256s")
        if (
            not isinstance(queries, list)
            or not queries
            or any(not isinstance(item, str) or not item for item in queries)
            or not isinstance(query_hashes, list)
            or len(query_hashes) != len(queries)
            or any(not isinstance(item, str) or not _SHA256.fullmatch(item) for item in query_hashes)
            or record.get("lane_name") != f"subnarrative:{subnarrative_id}"
            or record.get("semantic_query_sha256")
            != expected_subnarrative.semantic_query_sha256
            or queries != list(expected_subnarrative.bm25_queries)
            or query_hashes != list(expected_subnarrative.bm25_query_sha256s)
        ):
            raise ValueError("subnarrative score BM25 query provenance is invalid")
        seen.add(pair)
        rows.append(_downstream_score_provenance(record))
    return tuple(rows)


def _validate_score_components(
    record: Mapping[str, object],
    label: str,
    *,
    allow_empty_passages: bool = False,
) -> None:
    for key in ("bm25_rank", "aggregate_rank"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} rank is invalid")
    for key in (
        "bm25_score",
        "aggregate_score",
        "long_document_raw_logit",
        "weighted_passage_raw_logit",
    ):
        value = record.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label} value is not finite")
    support = record.get("within_document_span_support")
    if isinstance(support, bool) or not isinstance(support, int) or support < 0:
        raise ValueError(f"{label} span support is invalid")
    for key in ("semantic_query_sha256", "text_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{label} digest is invalid")
    if record.get("score_representation") != "raw_logits":
        raise ValueError(f"{label} representation is invalid")
    passages = record.get("winning_passages")
    if not isinstance(passages, list) or (not passages and not allow_empty_passages):
        raise ValueError(f"{label} winning passages are invalid")
    for passage in passages:
        if not isinstance(passage, dict) or set(passage) != _PASSAGE_FIELDS:
            raise ValueError(f"{label} passage schema changed")
        integer_values = tuple(
            passage[key]
            for key in ("chunk_index", "start_char", "end_char", "weighted_rank")
        )
        raw_logit = passage["raw_logit"]
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values)
            or passage["chunk_index"] < 0
            or passage["start_char"] < 0
            or passage["end_char"] <= passage["start_char"]
            or passage["weighted_rank"] <= 0
            or isinstance(raw_logit, bool)
            or not isinstance(raw_logit, (int, float))
            or not math.isfinite(raw_logit)
        ):
            raise ValueError(f"{label} passage value is invalid")


def _downstream_score_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "topic_id": record["topic_id"],
        "lane_name": record["lane_name"],
        "semantic_query_sha256": record["semantic_query_sha256"],
        "docid": record["docid"],
        "aggregate_rank": record["aggregate_rank"],
        "aggregate_score": record["aggregate_score"],
        "long_document_raw_logit": record["long_document_raw_logit"],
        "weighted_passage_raw_logit": record["weighted_passage_raw_logit"],
        "within_document_span_support": record["within_document_span_support"],
        "winning_passages": [
            {
                "chunk_index": passage["chunk_index"],
                "start_char": passage["start_char"],
                "end_char": passage["end_char"],
                "raw_logit": passage["raw_logit"],
                "weighted_rank": passage["weighted_rank"],
            }
            for passage in record["winning_passages"]
        ],
        "score_representation": record["score_representation"],
        "text_sha256": record["text_sha256"],
        "selection_rank": record["selection_rank"],
        "subnarrative_id": record["subnarrative_id"],
        "bm25_queries": list(record["bm25_queries"]),
        "bm25_query_sha256s": list(record["bm25_query_sha256s"]),
        "downstream_only": True,
    }


def _validate_cross_score_matrix(
    scores: tuple[dict[str, Any], ...],
    selected: tuple[_SelectedDocument, ...],
    *,
    expected_subnarrative_ids: frozenset[str],
    allow_empty: bool = False,
) -> frozenset[str]:
    passage_lane_rows = tuple(
        row for row in scores
        if row.get("schema_version") == "facet_passage_lane_v2"
    )
    if passage_lane_rows:
        if len(passage_lane_rows) != len(scores):
            raise ValueError("passage lane score schemas are mixed")
        subnarratives = frozenset(row["subnarrative_id"] for row in passage_lane_rows)
        if subnarratives != expected_subnarrative_ids:
            raise ValueError("passage lane score set differs from canonical plan")
        return subnarratives
    subnarratives = frozenset(row["subnarrative_id"] for row in scores)
    if not subnarratives:
        if allow_empty and not expected_subnarrative_ids:
            return frozenset()
        raise ValueError("document-by-subnarrative score matrix is empty")
    if subnarratives != expected_subnarrative_ids:
        raise ValueError("document-by-subnarrative score matrix differs from canonical plan")
    actual = {(row["docid"], row["subnarrative_id"]) for row in scores}
    expected = {
        (document.docid, subnarrative_id)
        for document in selected
        for subnarrative_id in subnarratives
    }
    if actual != expected:
        raise ValueError("document-by-subnarrative score matrix is incomplete")
    return subnarratives


def _validate_original_only_scoring(
    selected: tuple[_SelectedDocument, ...],
    lane_scores: Mapping[tuple[str, str], dict[str, Any]],
    cross_scores: tuple[dict[str, Any], ...],
    *,
    expected_query_sha256: str,
) -> None:
    if any(
        score[hash_field] != expected_query_sha256
        for score in lane_scores.values()
        for hash_field in ("bm25_query_sha256", "semantic_query_sha256")
    ):
        raise ValueError(
            "original-only fallback scores are not bound to original narrative query"
        )
    if (
        cross_scores
        or any(document.selected_from_lane != "original" for document in selected)
        or any(lane_name != "original" for _docid, lane_name in lane_scores)
    ):
        raise ValueError("original-only fallback scoring contains downstream data")


def _order_supported_docids(
    supported_docids: set[str],
    retrieval_audit: Sequence[Any],
) -> tuple[str, ...]:
    if not supported_docids:
        return ()
    evidence: dict[str, list[tuple[float, int, int]]] = {
        docid: [] for docid in supported_docids
    }
    for lane_index, audited in enumerate(retrieval_audit):
        passage_result = audited.passage_result
        if passage_result is None:
            raise ValueError("retrieval audit is missing bound passage evidence")
        for document in passage_result.documents:
            if (
                document.docid in evidence
                and document.best_passage_raw_logit is not None
            ):
                evidence[document.docid].append(
                    (
                        float(document.best_passage_raw_logit),
                        document.source_rank,
                        lane_index,
                    )
                )
    missing = sorted(docid for docid, rows in evidence.items() if not rows)
    if missing:
        raise ValueError(
            "canonical evidence document lacks scored shared passage evidence: "
            + ", ".join(missing)
        )

    def order_key(docid: str) -> tuple[float, int, int, str]:
        rows = evidence[docid]
        best_score, best_source_rank, stable_lane_order = min(
            rows,
            key=lambda row: (-row[0], row[1], row[2]),
        )
        return (-best_score, best_source_rank, stable_lane_order, docid)

    return tuple(sorted(supported_docids, key=order_key))


def _load_supported_docids(
    path: Path,
    topic_id: str,
    retrieval_bundle: EvidenceBundle,
    *,
    expected_subnarrative_ids: frozenset[str],
    nugget_manifest: Mapping[str, Any],
    allowed_evidence: frozenset[tuple[object, ...]],
    requests: Sequence[Any],
) -> tuple[
    set[str],
    dict[str, tuple[dict[str, Any], ...]],
    tuple[Mapping[str, object], ...],
]:
    from trec_rag.canonical_nuggets import validate_canonical_nugget_result

    retrieval_by_id = {
        document.docid: document for document in retrieval_bundle.documents
    }
    supported: set[str] = set()
    nuggets_by_doc: dict[str, list[dict[str, Any]]] = {}
    nugget_ids: set[str] = set()
    seen_subnarratives: set[str] = set()
    states: Counter[str] = Counter()
    records = _jsonl(path, "canonical nuggets")
    request_sha256s = [request.request_sha256 for request in requests]
    if (
        len(records) != nugget_manifest["result_count"]
        or nugget_manifest["request_sha256s"] != request_sha256s
    ):
        raise ValueError("canonical nugget result count changed")
    canonical_results: list[Mapping[str, object]] = []
    for record, request in zip(records, requests, strict=True):
        if not isinstance(record, Mapping):
            raise ValueError("canonical nugget result must be an object")
        try:
            state = validate_canonical_nugget_result(record, request)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "canonical nugget result or canonical evidence semantics are invalid"
            ) from exc
        subnarrative_id = request.subnarrative_id
        if (
            request.topic_id != topic_id
            or subnarrative_id not in expected_subnarrative_ids
            or subnarrative_id in seen_subnarratives
            or request.selected_budget != nugget_manifest["selected_budget"]
        ):
            raise ValueError("canonical nugget subnarrative identity is invalid")
        seen_subnarratives.add(subnarrative_id)
        states[state] += 1
        canonical_results.append(record)
        nuggets = record["nuggets"]
        for nugget in nuggets:
            nugget_id = nugget["canonical_nugget_id"]
            evidence = nugget["evidence"]
            if nugget_id in nugget_ids:
                raise ValueError("canonical nugget identity or evidence is invalid")
            nugget_ids.add(nugget_id)
            for item in evidence:
                docid = item["docid"]
                document = retrieval_by_id.get(docid)
                if document is None:
                    raise ValueError(
                        "canonical evidence document is absent from retrieval evidence bundle"
                    )
                if item.get("document_sha256") != document.text_sha256:
                    raise ValueError("canonical evidence document hash changed")
                evidence_text = item.get("text")
                if (
                    not isinstance(evidence_text, str)
                    or not evidence_text
                    or item.get("text_sha256")
                    != sha256(evidence_text.encode("utf-8")).hexdigest()
                ):
                    raise ValueError("canonical evidence text identity changed")
                evidence_identity = (
                    subnarrative_id,
                    item.get("cluster_id"),
                    item.get("candidate_nugget_id"),
                    item.get("candidate_kind"),
                    evidence_text,
                    docid,
                    item.get("document_sha256"),
                )
                if evidence_identity not in allowed_evidence:
                    raise ValueError(
                        "canonical evidence is absent from sealed selected candidate and cluster"
                    )
                supported.add(document.docid)
                nugget_link = {
                    "canonical_nugget_id": nugget_id,
                    "nugget_kind": nugget.get("nugget_kind"),
                    "claim_text": nugget.get("claim_text"),
                    "importance": nugget.get("importance"),
                    "subnarrative_id": subnarrative_id,
                    "evidence": {
                        "candidate_nugget_id": item["candidate_nugget_id"],
                        "candidate_kind": item["candidate_kind"],
                        "text": item["text"],
                        "text_sha256": item["text_sha256"],
                        "docid": item["docid"],
                        "document_sha256": item["document_sha256"],
                        "cluster_id": item["cluster_id"],
                    },
                }
                links = nuggets_by_doc.setdefault(document.docid, [])
                if nugget_link not in links:
                    links.append(nugget_link)
    if (
        seen_subnarratives != set(expected_subnarrative_ids)
        or len(records) != nugget_manifest["result_count"]
        or dict(sorted(states.items())) != nugget_manifest["state_counts"]
    ):
        raise ValueError("canonical nugget result set is incomplete")
    return (
        supported,
        {docid: tuple(rows) for docid, rows in nuggets_by_doc.items()},
        tuple(canonical_results),
    )


def _load_allowed_canonical_evidence(
    topic_root: Path,
    topic: Topic,
    config: FacetPilotConfig,
    records: TopicRecords,
    records_receipt: TopicRecordsReceipt,
    retrieval_bundle: EvidenceBundle,
    subnarratives: Sequence[Any],
    *,
    selected_budget: int,
    max_canonical_claims: int,
    max_supporting_documents_per_claim: int,
) -> _ValidatedEvidenceProjection:
    from trec_rag.canonical_nuggets import (
        build_canonical_nugget_request,
        load_validated_selection_artifacts,
    )
    from trec_rag.evidence_store import _validate_candidate_stage_identity
    from trec_rag.topic_records import TopicRecordsIntegrityError

    canonical_root = topic_root / "canonical"
    selections_path = canonical_root / "subnarrative-selections.jsonl"
    selection_manifest_path = canonical_root / "selection-manifest.json"
    selections, manifest, _policy = load_validated_selection_artifacts(
        selections_path, selection_manifest_path
    )
    contexts_path = canonical_root / "handoff" / "selection-contexts.jsonl"
    fixed_files = {
        "records_file": "records.sqlite3",
        "records_manifest_file": "records-manifest.json",
        "contexts_file": contexts_path.name,
    }
    digests = {
        "records_database_sha256": records_receipt.database_sha256,
        "candidate_semantic_sha256": records_receipt.semantic_sha256,
        "contexts_sha256": sha256(contexts_path.read_bytes()).hexdigest(),
    }
    if any(manifest.get(key) != value for key, value in fixed_files.items()) or any(
        manifest.get(key) != value for key, value in digests.items()
    ):
        raise ValueError("selection manifest evidence artifact seal changed")
    _validate_candidate_stage_identity(records)
    expected_subnarratives = {
        row.subnarrative_id: row.text for row in subnarratives
    }
    selection_by_id: dict[str, Any] = {}
    for selection in selections:
        context = selection.context
        if (
            context.topic_id != topic.id
            or context.official_narrative != topic.narrative
            or expected_subnarratives.get(context.subnarrative_id)
            != context.subnarrative_text
            or context.subnarrative_id in selection_by_id
        ):
            raise ValueError("canonical selection differs from canonical plan")
        selection_by_id[context.subnarrative_id] = selection
    if set(selection_by_id) != set(expected_subnarratives):
        raise ValueError("canonical selection set is incomplete")
    selected_members: list[tuple[str, str, Any]] = []
    for subnarrative_id, selection in selection_by_id.items():
        snapshots = {snapshot.budget: snapshot for snapshot in selection.snapshots}
        snapshot = snapshots.get(selected_budget)
        if snapshot is None:
            raise ValueError("canonical selection lacks configured budget")
        clusters = {cluster.cluster_id: cluster for cluster in selection.clusters}
        for cluster_id in snapshot.cluster_ids:
            cluster = clusters.get(cluster_id)
            if cluster is None:
                raise ValueError("canonical selection snapshot names an unknown cluster")
            selected_members.extend(
                (subnarrative_id, cluster_id, member)
                for member in cluster.supports
            )
    required_candidate_keys = frozenset(
        (subnarrative_id, member.candidate_nugget_id)
        for subnarrative_id, _cluster_id, member in selected_members
    )
    try:
        candidates = records.load_candidates(required_candidate_keys)
    except TopicRecordsIntegrityError as exc:
        raise ValueError(
            f"topic records source closure or seal is invalid: {exc}"
        ) from exc
    if set(candidates) != set(required_candidate_keys):
        raise ValueError("validated candidate projection differs from selected supports")
    retrieved_by_docid = {
        row.docid: row for row in retrieval_bundle.documents
    }
    allowed: set[tuple[object, ...]] = set()
    for subnarrative_id, cluster_id, member in selected_members:
        candidate = candidates.get((subnarrative_id, member.candidate_nugget_id))
        retrieved_document = (
            None if candidate is None else retrieved_by_docid.get(candidate.docid)
        )
        if candidate is not None and retrieved_document is None:
            raise ValueError(
                "canonical evidence document is absent from retrieval evidence bundle"
            )
        if (
            candidate is None
            or retrieved_document is None
            or member.candidate_kind != candidate.candidate_kind
            or member.text != candidate.text
            or member.docid != candidate.docid
            or member.document_sha256 != candidate.document_sha256
            or retrieved_document.text_sha256 != candidate.document_sha256
            or member.raw_logit != candidate.sentence_cross_encoder_score
        ):
            raise ValueError(
                "canonical selection member differs from sealed topic records"
            )
        allowed.add(
            (
                subnarrative_id,
                cluster_id,
                candidate.candidate_nugget_id,
                candidate.candidate_kind,
                candidate.text,
                candidate.docid,
                candidate.document_sha256,
            )
        )
    requests = tuple(
        build_canonical_nugget_request(
            selection,
            selected_budget,
            max_canonical_claims=max_canonical_claims,
            max_supporting_documents_per_claim=(
                max_supporting_documents_per_claim
            ),
        )
        for selection in selections
    )
    return _ValidatedEvidenceProjection(
        selections=tuple(selections),
        candidates=candidates,
        canonical_requests=requests,
        allowed_evidence=frozenset(allowed),
        retrieval_topic_sha256=records_receipt.semantic_sha256,
    )


def _jsonl(path: Path, label: str) -> tuple[object, ...]:
    body = path.read_bytes()
    if body and not body.endswith(b"\n"):
        raise ValueError(f"{label} must end with LF")
    return tuple(
        _strict_json(line, f"{label} line {line_number}")
        for line_number, line in enumerate(body.splitlines(), start=1)
        if line
    )


def _strict_json(body: bytes, label: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains non-standard JSON constant {value}")

    try:
        return json.loads(
            body,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc


def _trec_bytes(rows: Sequence[tuple[str, str, int, int, str]]) -> bytes:
    return "".join(
        f"{topic_id} Q0 {docid} {rank} {score} {run_id}\n"
        for topic_id, docid, rank, score, run_id in rows
    ).encode("utf-8")


def _validate_trec_bytes(
    body: bytes, expected: Sequence[tuple[str, str, int, int, str]]
) -> None:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("emitted TREC run is not UTF-8") from exc
    if body and not body.endswith(b"\n"):
        raise ValueError("emitted TREC run lacks final LF")
    reparsed: list[tuple[str, str, int, int, str]] = []
    pairs: set[tuple[str, str]] = set()
    last_rank: dict[str, int] = {}
    last_score: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split(" ")
        if len(fields) != 6 or any(not field for field in fields) or fields[1] != "Q0":
            raise ValueError("emitted TREC row is malformed")
        topic_id, _q0, docid, rank_text, score_text, run_id = fields
        try:
            rank = int(rank_text)
            score = int(score_text)
        except ValueError as exc:
            raise ValueError("emitted TREC rank or score is not an integer") from exc
        pair = (topic_id, docid)
        if pair in pairs:
            raise ValueError("duplicate topic-document pair in emitted TREC run")
        if rank != last_rank.get(topic_id, 0) + 1:
            raise ValueError("emitted TREC ranks are not contiguous")
        previous_score = last_score.get(topic_id)
        if previous_score is not None and score >= previous_score:
            raise ValueError("emitted TREC scores are not strictly decreasing")
        pairs.add(pair)
        last_rank[topic_id] = rank
        last_score[topic_id] = score
        reparsed.append((topic_id, docid, rank, score, run_id))
    if reparsed != list(expected):
        raise ValueError("emitted TREC run does not match its projection")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, object] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _deterministic_zip(member_body: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        member = zipfile.ZipInfo(
            "retrieval_with_text.jsonl", date_time=(1980, 1, 1, 0, 0, 0)
        )
        member.compress_type = zipfile.ZIP_DEFLATED
        member.create_system = 3
        member.external_attr = 0o100600 << 16
        archive.writestr(member, member_body)
    return buffer.getvalue()


@contextmanager
def _export_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".retrieval-export.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
