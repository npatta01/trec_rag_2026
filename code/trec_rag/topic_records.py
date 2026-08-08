"""Sealed, offset-only candidate records for one topic.

The v2 database stores integer surrogate keys for documents, candidates, and
passages, one complete passage dictionary row per ``(document, passage_id)``,
and compact ``candidate_passage_link`` rows.  Every text role is reconstructed
from the bound document body plus stored offsets, so the database never repeats
source text.

Validation is a current-process capability rather than a disk marker.  A
successful validator returns a non-serializable :class:`ValidatedTopicRecords`
bound to the exact database bytes, exact manifest bytes, validator version,
topic, semantic seal, and verified content-addressed-store receipts it checked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
from threading import RLock
from typing import Any
from types import MappingProxyType
import weakref

from .document_store import DocumentReceipt, DocumentStore, DocumentStoreIntegrityError
from .facet_evidence import (
    SCHEMA_VERSION as CANDIDATE_SCHEMA_VERSION,
    pair_is_admissible,
    SCORING_NORMALIZATION_VERSION,
    SENTENCE_SPLITTER_VERSION,
    ExtractiveCandidate,
    PassageProvenance,
    SelectionCandidate,
    SentenceEvidence,
    SourceSpan,
    SubnarrativeContext,
    _candidate_nugget_id_from_identity,
    _dependency_cue,
)
from .topic_geometry import DocumentGeometry, DocumentGeometryIndex, TopicGeometryError
from .topic_passage_search import (
    FocusedQuery,
    PassageSearchResult,
    SourceDocument,
    SourcePassage,
)


TOPIC_RECORDS_SCHEMA_VERSION = "topic-records-v4"
TOPIC_RECORDS_VALIDATOR_VERSION = "topic-records-source-validator-v2"
CANDIDATE_STAGE = "canonical-candidates-v1"

_PUBLICATION_TEST_HOOK: Callable[[str], None] | None = None

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID = re.compile(r"^ecn1_[0-9a-f]{64}$")
_CANDIDATE_KINDS = frozenset({"exact_sentence", "exact_sentence_pair"})
_HASH_CHUNK_BYTES = 1 << 20
_BUILDER_BATCH_CANDIDATES = 256

_SCOPED_TABLES = (
    "topic_identity",
    "document_binding",
    "subnarrative_identity",
    "query_identity",
    "query_facet",
    "retrieval_candidate",
    "query_passage",
    "candidate",
    "candidate_span",
    "passage",
    "candidate_passage_link",
    "researcher_handoff",
    "researcher_evidence",
    "researcher_facet_update",
    "topic_completion",
)
_SCHEMA_COLUMNS = {
    "topic_identity": ("topic_id", "run_id"),
    "document_binding": (
        "document_pk", "topic_id", "docid", "content_sha256",
        "character_count", "byte_count",
    ),
    "subnarrative_identity": (
        "topic_id", "subnarrative_id", "subnarrative_sha256", "subnarrative_text", "origin",
    ),
    "query_identity": (
        "query_id", "query_text_sha256", "primary_subnarrative_id", "status",
        "stopping_reason", "attempt_count", "requested_documents", "returned_documents",
        "scored_documents", "scored_passages", "source_exhausted",
    ),
    "query_facet": ("query_id", "subnarrative_id", "role", "ordinal"),
    "retrieval_candidate": (
        "query_id", "document_pk", "source_rank", "source_score",
        "best_passage_id", "best_passage_raw_logit",
    ),
    "query_passage": (
        "query_id", "passage_pk", "raw_logit", "passage_rank", "score_cache_key",
    ),
    "candidate": (
        "candidate_pk", "topic_id", "candidate_nugget_id", "document_pk",
        "subnarrative_id", "candidate_kind", "nugget_type", "start_char", "end_char",
        "text_sha256", "sentence_score", "document_subnarrative_rank",
        "scoring_text_sha256", "subnarrative_sha256", "sentence_splitter_version",
    ),
    "candidate_span": (
        "candidate_pk", "role", "ordinal", "start_char", "end_char",
        "start_byte", "end_byte", "text_sha256", "cross_encoder_score",
    ),
    "passage": (
        "passage_pk", "document_pk", "passage_id", "source_start_char", "source_end_char",
        "source_start_byte", "source_end_byte", "source_text_sha256",
        "scoring_text_sha256", "chunker_identity_json",
    ),
    "candidate_passage_link": (
        "candidate_pk", "document_pk", "ordinal", "passage_pk", "query_id",
    ),
    "researcher_handoff": ("researcher_id", "run_id", "round_index", "handoff_sha256"),
    "researcher_evidence": (
        "researcher_id", "subnarrative_id", "passage_pk", "relevance",
    ),
    "researcher_facet_update": ("researcher_id", "ordinal", "subnarrative_id"),
    "topic_completion": ("singleton", "status", "stopping_reason"),
    "stage_seal": ("stage", "schema_version", "semantic_sha256", "identity_json", "row_counts_json"),
}
_INTEGER_COLUMNS = frozenset({
    "document_pk", "candidate_pk", "passage_pk",
    "character_count", "byte_count", "start_char", "end_char",
    "document_subnarrative_rank", "ordinal", "start_byte", "end_byte",
    "scoring_start_char", "scoring_end_char", "source_start_char",
    "source_end_char", "source_start_byte", "source_end_byte",
    "cross_encoder_rank", "attempt_count", "requested_documents", "returned_documents",
    "scored_documents", "scored_passages", "source_exhausted", "source_rank",
    "passage_rank", "round_index", "singleton",
})
_REAL_COLUMNS = frozenset({
    "sentence_score", "cross_encoder_score", "source_score", "raw_logit",
    "best_passage_raw_logit",
})
_NULLABLE_COLUMNS = frozenset({
    ("candidate_span", "cross_encoder_score"),
    ("query_identity", "stopping_reason"),
    ("retrieval_candidate", "best_passage_id"),
    ("retrieval_candidate", "best_passage_raw_logit"),
})
# ``INTEGER PRIMARY KEY`` surrogates are rowid aliases: SQLite reports them with
# ``notnull=0`` and does not create a separate primary-key index for them.
_ROWID_PRIMARY_KEYS = {
    "document_binding": "document_pk",
    "candidate": "candidate_pk",
    "passage": "passage_pk",
    "topic_completion": "singleton",
}
_UNIQUE_KEYS = {
    "document_binding": (
        ("topic_id", "docid"),
        ("topic_id", "docid", "content_sha256"),
    ),
    "candidate": (
        ("topic_id", "candidate_nugget_id"),
        ("candidate_pk", "document_pk"),
    ),
    "passage": (
        ("passage_id",),
        ("document_pk", "passage_id"),
        ("passage_pk", "document_pk"),
    ),
    "subnarrative_identity": (("subnarrative_id",),),
    "query_facet": (("query_id", "role", "ordinal"),),
    "retrieval_candidate": (("query_id", "document_pk"),),
    "query_passage": (("query_id", "passage_rank"),),
    "candidate_passage_link": (
        ("candidate_pk", "passage_pk", "query_id"),
    ),
    "researcher_facet_update": (("researcher_id", "subnarrative_id"),),
}
_FOREIGN_KEYS = {
    "document_binding": (
        ("topic_identity", (("topic_id", "topic_id"),)),
    ),
    "subnarrative_identity": (
        ("topic_identity", (("topic_id", "topic_id"),)),
    ),
    "query_identity": (
        ("subnarrative_identity", (("primary_subnarrative_id", "subnarrative_id"),)),
    ),
    "query_facet": (
        ("query_identity", (("query_id", "query_id"),)),
        ("subnarrative_identity", (("subnarrative_id", "subnarrative_id"),)),
    ),
    "retrieval_candidate": (
        ("query_identity", (("query_id", "query_id"),)),
        ("document_binding", (("document_pk", "document_pk"),)),
    ),
    "query_passage": (
        ("query_identity", (("query_id", "query_id"),)),
        ("passage", (("passage_pk", "passage_pk"),)),
    ),
    "candidate": (
        ("document_binding", (("document_pk", "document_pk"),)),
        (
            "subnarrative_identity",
            (("topic_id", "topic_id"), ("subnarrative_id", "subnarrative_id")),
        ),
    ),
    "candidate_span": (
        ("candidate", (("candidate_pk", "candidate_pk"),)),
    ),
    "passage": (
        ("document_binding", (("document_pk", "document_pk"),)),
    ),
    "candidate_passage_link": (
        ("candidate", (("candidate_pk", "candidate_pk"), ("document_pk", "document_pk"))),
        ("passage", (("passage_pk", "passage_pk"), ("document_pk", "document_pk"))),
        ("query_passage", (("query_id", "query_id"), ("passage_pk", "passage_pk"))),
    ),
    "researcher_evidence": (
        ("researcher_handoff", (("researcher_id", "researcher_id"),)),
        ("subnarrative_identity", (("subnarrative_id", "subnarrative_id"),)),
        ("passage", (("passage_pk", "passage_pk"),)),
    ),
    "researcher_facet_update": (
        ("researcher_handoff", (("researcher_id", "researcher_id"),)),
        ("subnarrative_identity", (("subnarrative_id", "subnarrative_id"),)),
    ),
}
_PRIMARY_KEYS = {
    "topic_identity": ("topic_id",),
    "document_binding": ("document_pk",),
    "subnarrative_identity": ("topic_id", "subnarrative_id"),
    "query_identity": ("query_id",),
    "query_facet": ("query_id", "subnarrative_id"),
    "retrieval_candidate": ("query_id", "source_rank"),
    "query_passage": ("query_id", "passage_pk"),
    "candidate": ("candidate_pk",),
    "candidate_span": ("candidate_pk", "role", "ordinal"),
    "passage": ("passage_pk",),
    "candidate_passage_link": ("candidate_pk", "ordinal"),
    "researcher_handoff": ("researcher_id",),
    "researcher_evidence": ("researcher_id", "subnarrative_id", "passage_pk"),
    "researcher_facet_update": ("researcher_id", "ordinal"),
    "topic_completion": ("singleton",),
    "stage_seal": ("stage",),
}

# Semantic projections join through stable natural identities.  ``document_pk``,
# ``candidate_pk``, and ``passage_pk`` never appear in the hashed bytes or in an
# ``ORDER BY``, so surrogate allocation cannot change a seal.
_SEMANTIC_PROJECTIONS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "topic",
        ("topic_id", "run_id"),
        "SELECT topic_id, run_id FROM topic_identity ORDER BY topic_id",
    ),
    (
        "document",
        ("topic_id", "docid", "content_sha256", "character_count", "byte_count"),
        "SELECT topic_id, docid, content_sha256, character_count, byte_count "
        "FROM document_binding ORDER BY topic_id, docid",
    ),
    (
        "subnarrative",
        ("topic_id", "subnarrative_id", "subnarrative_sha256", "subnarrative_text", "origin"),
        "SELECT topic_id, subnarrative_id, subnarrative_sha256, subnarrative_text, origin "
        "FROM subnarrative_identity ORDER BY topic_id, subnarrative_id",
    ),
    (
        "query",
        (
            "query_id", "query_text_sha256", "primary_subnarrative_id", "status",
            "stopping_reason", "attempt_count", "requested_documents", "returned_documents",
            "scored_documents", "scored_passages", "source_exhausted",
        ),
        "SELECT query_id, query_text_sha256, primary_subnarrative_id, status, stopping_reason, "
        "attempt_count, requested_documents, returned_documents, scored_documents, "
        "scored_passages, source_exhausted FROM query_identity ORDER BY query_id",
    ),
    (
        "query_facet",
        ("query_id", "subnarrative_id", "role", "ordinal"),
        "SELECT query_id, subnarrative_id, role, ordinal FROM query_facet "
        "ORDER BY query_id, ordinal, subnarrative_id",
    ),
    (
        "retrieval_candidate",
        (
            "query_id", "docid", "content_sha256", "source_rank", "source_score",
            "best_passage_id", "best_passage_raw_logit",
        ),
        "SELECT r.query_id, d.docid, d.content_sha256, r.source_rank, r.source_score, "
        "r.best_passage_id, r.best_passage_raw_logit FROM retrieval_candidate AS r "
        "JOIN document_binding AS d ON d.document_pk = r.document_pk "
        "ORDER BY r.query_id, r.source_rank",
    ),
    (
        "query_passage",
        ("query_id", "passage_id", "raw_logit", "passage_rank", "score_cache_key"),
        "SELECT q.query_id, p.passage_id, q.raw_logit, q.passage_rank, q.score_cache_key "
        "FROM query_passage AS q JOIN passage AS p ON p.passage_pk = q.passage_pk "
        "ORDER BY q.query_id, q.passage_rank, p.passage_id",
    ),
    (
        "candidate",
        (
            "topic_id", "candidate_nugget_id", "docid", "content_sha256", "subnarrative_id",
            "candidate_kind", "nugget_type", "start_char", "end_char", "text_sha256",
            "sentence_score", "document_subnarrative_rank", "scoring_text_sha256",
            "subnarrative_sha256", "sentence_splitter_version",
        ),
        "SELECT c.topic_id, c.candidate_nugget_id, d.docid, d.content_sha256, "
        "c.subnarrative_id, c.candidate_kind, c.nugget_type, c.start_char, c.end_char, "
        "c.text_sha256, c.sentence_score, c.document_subnarrative_rank, "
        "c.scoring_text_sha256, c.subnarrative_sha256, c.sentence_splitter_version "
        "FROM candidate AS c "
        "JOIN document_binding AS d ON d.document_pk = c.document_pk "
        "ORDER BY c.topic_id, c.candidate_nugget_id",
    ),
    (
        "span",
        (
            "topic_id", "candidate_nugget_id", "role", "ordinal", "start_char", "end_char",
            "start_byte", "end_byte", "text_sha256", "cross_encoder_score",
        ),
        "SELECT c.topic_id, c.candidate_nugget_id, s.role, s.ordinal, s.start_char, "
        "s.end_char, s.start_byte, s.end_byte, s.text_sha256, s.cross_encoder_score "
        "FROM candidate_span AS s "
        "JOIN candidate AS c ON c.candidate_pk = s.candidate_pk "
        "ORDER BY c.topic_id, c.candidate_nugget_id, s.role, s.ordinal",
    ),
    (
        "passage",
        (
            "topic_id", "docid", "content_sha256", "passage_id", "source_start_char",
            "source_end_char", "source_start_byte", "source_end_byte", "source_text_sha256",
            "scoring_text_sha256", "chunker_identity_json",
        ),
        "SELECT d.topic_id, d.docid, d.content_sha256, p.passage_id, p.source_start_char, "
        "p.source_end_char, p.source_start_byte, p.source_end_byte, p.source_text_sha256, "
        "p.scoring_text_sha256, p.chunker_identity_json "
        "FROM passage AS p "
        "JOIN document_binding AS d ON d.document_pk = p.document_pk "
        "ORDER BY d.topic_id, d.docid, d.content_sha256, p.passage_id",
    ),
    (
        "candidate_passage_link",
        ("topic_id", "candidate_nugget_id", "ordinal", "query_id", "docid", "content_sha256", "passage_id"),
        "SELECT c.topic_id, c.candidate_nugget_id, l.ordinal, l.query_id, d.docid, "
        "d.content_sha256, p.passage_id "
        "FROM candidate_passage_link AS l "
        "JOIN candidate AS c ON c.candidate_pk = l.candidate_pk "
        "JOIN passage AS p ON p.passage_pk = l.passage_pk "
        "JOIN document_binding AS d ON d.document_pk = l.document_pk "
        "ORDER BY c.topic_id, c.candidate_nugget_id, l.ordinal",
    ),
    (
        "researcher_handoff",
        ("researcher_id", "run_id", "round_index", "handoff_sha256"),
        "SELECT researcher_id, run_id, round_index, handoff_sha256 FROM researcher_handoff "
        "ORDER BY researcher_id",
    ),
    (
        "researcher_evidence",
        ("researcher_id", "subnarrative_id", "passage_id", "relevance"),
        "SELECT e.researcher_id, e.subnarrative_id, p.passage_id, e.relevance "
        "FROM researcher_evidence AS e JOIN passage AS p ON p.passage_pk = e.passage_pk "
        "ORDER BY e.researcher_id, e.subnarrative_id, p.passage_id",
    ),
    (
        "researcher_facet_update",
        ("researcher_id", "ordinal", "subnarrative_id"),
        "SELECT researcher_id, ordinal, subnarrative_id FROM researcher_facet_update "
        "ORDER BY researcher_id, ordinal",
    ),
    (
        "completion",
        ("singleton", "status", "stopping_reason"),
        "SELECT singleton, status, stopping_reason FROM topic_completion ORDER BY singleton",
    ),
)

_SPAN_ROLES = ("evidence", "matched_paragraph", "context_before", "context_after")
_ROLE_ORDER = {role: index for index, role in enumerate(_SPAN_ROLES)}


class TopicRecordsIntegrityError(RuntimeError):
    """Raised when a topic records artifact is incomplete or inconsistent."""


@dataclass(frozen=True)
class TopicRecordsReceipt:
    database_sha256: str
    database_bytes: int
    semantic_sha256: str
    topic_id: str
    run_id: str
    document_sha256s: tuple[str, ...]
    row_counts: Mapping[str, int]
    schema_version: str
    manifest_sha256: str
    manifest_bytes: int


@dataclass(frozen=True)
class PublishedTopicRecords:
    """A published topic database plus its current-process validation session."""

    receipt: TopicRecordsReceipt
    validation_session: ValidatedTopicRecords = field(compare=False, repr=False)

    @property
    def topic_id(self) -> str:
        return self.receipt.topic_id

    @property
    def run_id(self) -> str:
        return self.receipt.run_id

    @property
    def semantic_sha256(self) -> str:
        return self.receipt.semantic_sha256

    @property
    def database_sha256(self) -> str:
        return self.receipt.database_sha256

    @property
    def database_bytes(self) -> int:
        return self.receipt.database_bytes

    @property
    def document_sha256s(self) -> tuple[str, ...]:
        return self.receipt.document_sha256s

    @property
    def row_counts(self) -> Mapping[str, int]:
        return self.receipt.row_counts


@dataclass(frozen=True)
class PreclusterPool:
    """Deterministically ordered exact-group candidates for one subnarrative."""

    candidates: tuple[SelectionCandidate, ...]
    candidate_count: int
    exact_group_count: int

    def __iter__(self):
        return iter(self.candidates)


@dataclass(frozen=True)
class FacetRecord:
    subnarrative_id: str
    text: str
    origin: str

    def __post_init__(self) -> None:
        _require_text(self.subnarrative_id, "subnarrative_id")
        _require_text(self.text, "facet text")
        if self.origin not in {"initial", "research_discovered"}:
            raise ValueError("facet origin must be initial or research_discovered")

    @property
    def text_sha256(self) -> str:
        return _digest(self.text)


@dataclass(frozen=True)
class ResearcherEvidence:
    subnarrative_id: str
    passage_id: str
    relevance: str

    def __post_init__(self) -> None:
        _require_text(self.subnarrative_id, "researcher evidence subnarrative_id")
        _require_text(self.passage_id, "researcher evidence passage_id")
        if self.relevance not in {"relevant", "supporting"}:
            raise ValueError("researcher evidence relevance must be relevant or supporting")


@dataclass(frozen=True)
class ResearcherHandoff:
    run_id: str
    researcher_id: str
    round_index: int
    evidence: tuple[ResearcherEvidence, ...]
    facet_updates: tuple[FacetRecord, ...]

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        _require_text(self.researcher_id, "researcher_id")
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, int) or self.round_index < 0:
            raise ValueError("round_index must be a non-negative integer")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(row, ResearcherEvidence) for row in self.evidence
        ):
            raise ValueError("evidence must contain ResearcherEvidence rows")
        if not isinstance(self.facet_updates, tuple) or any(
            not isinstance(row, FacetRecord) for row in self.facet_updates
        ):
            raise ValueError("facet_updates must contain FacetRecord rows")


@dataclass(frozen=True)
class StoredQuery:
    query_id: str
    query_text_sha256: str
    primary_subnarrative_id: str
    supporting_subnarrative_ids: tuple[str, ...]
    status: str
    stopping_reason: str | None
    attempt_count: int
    requested_documents: int
    returned_documents: int
    scored_documents: int
    scored_passages: int
    source_exhausted: bool


@dataclass(frozen=True)
class PassageSearchSnapshot:
    query: StoredQuery
    documents: tuple[SourceDocument, ...]
    passages: tuple[SourcePassage, ...]

    @property
    def query_id(self) -> str:
        return self.query.query_id


@dataclass(frozen=True)
class TopicEvidenceSnapshot:
    facets: tuple[FacetRecord, ...]
    queries: tuple[PassageSearchSnapshot, ...]
    documents: tuple[SourceDocument, ...]
    passages: tuple[SourcePassage, ...]
    researcher_handoffs: tuple[ResearcherHandoff, ...]
    researcher_evidence: tuple[ResearcherEvidence, ...]
    candidates: tuple[ExtractiveCandidate, ...]
    status: str | None
    stopping_reason: str | None


def _digest(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TopicRecordsIntegrityError(f"{label} must be a non-empty string")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TopicRecordsIntegrityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TopicRecordsIntegrityError(f"{label} must be finite")
    return float(value)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TopicRecordsIntegrityError(f"{label} must be a positive integer")
    return value


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TopicRecordsIntegrityError("stage identity is not canonical JSON") from exc


def _legacy_candidate_query_id(query_id: str, docid: str) -> str:
    """Namespace legacy candidate scores per document in the v3 query ledger."""
    return _canonical_json({"base_query_id": query_id, "docid": docid})


def _decode_legacy_candidate_query_id(query_id: str) -> str:
    try:
        value = json.loads(query_id)
    except (TypeError, ValueError):
        return query_id
    if (
        isinstance(value, dict)
        and set(value) == {"base_query_id", "docid"}
        and isinstance(value["base_query_id"], str)
        and isinstance(value["docid"], str)
        and value["docid"]
    ):
        return value["base_query_id"]
    return query_id


def _frame(tag: bytes, value: bytes) -> bytes:
    return tag + len(value).to_bytes(8, "big") + value


def _semantic_value(value: object) -> bytes:
    if value is None:
        return b"N"
    if isinstance(value, bool):
        return _frame(b"B", b"1" if value else b"0")
    if isinstance(value, int):
        return _frame(b"I", str(value).encode("ascii"))
    if isinstance(value, float):
        return _frame(b"F", value.hex().encode("ascii"))
    if isinstance(value, str):
        return _frame(b"S", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _frame(b"Y", value)
    raise TopicRecordsIntegrityError(f"unsupported SQLite value type {type(value).__name__}")


def _semantic_sha256(database: sqlite3.Connection) -> str:
    """Hash explicit joined natural-key projections, never physical rows."""
    digest = hashlib.sha256()
    for projection, columns, statement in _SEMANTIC_PROJECTIONS:
        digest.update(_frame(b"T", projection.encode("ascii")))
        for column in columns:
            digest.update(_frame(b"C", column.encode("utf-8")))
        for row in database.execute(statement):
            if len(row) != len(columns):
                raise TopicRecordsIntegrityError(
                    f"semantic projection {projection} returned an unexpected width"
                )
            digest.update(b"R")
            for value in row:
                digest.update(_semantic_value(value))
    return digest.hexdigest()


def _row_counts(database: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(database.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0])
        for table in (*_SCOPED_TABLES, "stage_seal")
    }


def _stream_receipt(path: Path) -> tuple[str, int]:
    """Stream-hash a file without materializing it in memory."""
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
    except OSError as exc:
        raise TopicRecordsIntegrityError(f"unable to read {path}") from exc
    return digest.hexdigest(), total


def _open_regular_readonly(path: Path, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise TopicRecordsIntegrityError(f"{label} is not a regular file")
        return descriptor
    except TopicRecordsIntegrityError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise TopicRecordsIntegrityError(f"unable to open {label}") from exc


def _read_descriptor(descriptor: int, label: str) -> bytes:
    try:
        chunks: list[bytes] = []
        offset = 0
        while True:
            chunk = os.pread(descriptor, _HASH_CHUNK_BYTES, offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        return b"".join(chunks)
    except AttributeError:
        original_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        duplicate = os.dup(descriptor)
        try:
            os.lseek(duplicate, 0, os.SEEK_SET)
            chunks = []
            while True:
                chunk = os.read(duplicate, _HASH_CHUNK_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        except OSError as exc:
            raise TopicRecordsIntegrityError(f"unable to read {label}") from exc
        finally:
            try:
                os.lseek(descriptor, original_offset, os.SEEK_SET)
            finally:
                os.close(duplicate)
    except OSError as exc:
        raise TopicRecordsIntegrityError(f"unable to read {label}") from exc


def _descriptor_receipt(descriptor: int, label: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        offset = 0
        while True:
            chunk = os.pread(descriptor, _HASH_CHUNK_BYTES, offset)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            offset += len(chunk)
        return digest.hexdigest(), total
    except AttributeError:
        body = _read_descriptor(descriptor, label)
        return hashlib.sha256(body).hexdigest(), len(body)
    except OSError as exc:
        raise TopicRecordsIntegrityError(f"unable to read {label}") from exc


def _connect_readonly_descriptor(descriptor: int) -> sqlite3.Connection:
    uri = f"file:/proc/self/fd/{descriptor}?mode=ro&immutable=1"
    database: sqlite3.Connection | None = None
    try:
        database = sqlite3.connect(uri, uri=True, check_same_thread=False)
        database.execute("PRAGMA foreign_keys=ON")
        database.execute("PRAGMA query_only=ON")
        database.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        return database
    except (OSError, sqlite3.Error) as exc:
        if database is not None:
            try:
                database.close()
            except sqlite3.Error:
                pass
        raise TopicRecordsIntegrityError("unable to open topic database") from exc


def _copy_descriptor(source: int, destination: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        offset = 0
        while True:
            chunk = os.pread(source, _HASH_CHUNK_BYTES, offset)
            if not chunk:
                break
            written = 0
            while written < len(chunk):
                written += os.write(descriptor, chunk[written:])
            offset += len(chunk)
        os.fsync(descriptor)
    except OSError as exc:
        raise TopicRecordsIntegrityError("unable to snapshot topic database") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _paths(destination: Path) -> tuple[Path, Path]:
    destination = Path(destination)
    return destination / "records.sqlite3", destination / "canonical" / "records-manifest.json"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_without_replacement(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    _fsync_directory(destination.parent)
    source.unlink(missing_ok=True)


def _publication_boundary(name: str) -> None:
    hook = _PUBLICATION_TEST_HOOK
    if hook is not None:
        hook(name)


@contextmanager
def _publication_lock(path: Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _atomic_manifest_create(path: Path, body: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        _publish_without_replacement(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _configure(database: sqlite3.Connection) -> None:
    database.execute("PRAGMA foreign_keys=ON")
    database.execute("PRAGMA journal_mode=DELETE")
    database.execute("PRAGMA synchronous=FULL")
    database.execute("PRAGMA temp_store=FILE")


def _serialized_builder_operation(method: Callable[..., Any]) -> Callable[..., Any]:
    """Serialize one complete builder transaction and its in-memory indexes."""

    @wraps(method)
    def locked(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._operation_lock:
            return method(self, *args, **kwargs)

    return locked


def _create_schema(database: sqlite3.Connection) -> None:
    database.executescript(
        """
        CREATE TABLE topic_identity (
            topic_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL
        ) STRICT;
        CREATE TABLE document_binding (
            document_pk INTEGER PRIMARY KEY,
            topic_id TEXT NOT NULL,
            docid TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            character_count INTEGER NOT NULL,
            byte_count INTEGER NOT NULL,
            UNIQUE(topic_id, docid),
            UNIQUE(topic_id, docid, content_sha256),
            FOREIGN KEY(topic_id) REFERENCES topic_identity(topic_id)
        ) STRICT;
        CREATE TABLE subnarrative_identity (
            topic_id TEXT NOT NULL,
            subnarrative_id TEXT NOT NULL,
            subnarrative_sha256 TEXT NOT NULL,
            subnarrative_text TEXT NOT NULL,
            origin TEXT NOT NULL CHECK(origin IN ('initial','research_discovered')),
            PRIMARY KEY(topic_id, subnarrative_id),
            UNIQUE(subnarrative_id),
            FOREIGN KEY(topic_id) REFERENCES topic_identity(topic_id)
        ) STRICT;
        CREATE TABLE query_identity (
            query_id TEXT PRIMARY KEY,
            query_text_sha256 TEXT NOT NULL,
            primary_subnarrative_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('complete','incomplete')),
            stopping_reason TEXT,
            attempt_count INTEGER NOT NULL,
            requested_documents INTEGER NOT NULL,
            returned_documents INTEGER NOT NULL,
            scored_documents INTEGER NOT NULL,
            scored_passages INTEGER NOT NULL,
            source_exhausted INTEGER NOT NULL CHECK(source_exhausted IN (0,1)),
            FOREIGN KEY(primary_subnarrative_id)
                REFERENCES subnarrative_identity(subnarrative_id)
        ) STRICT;
        CREATE TABLE query_facet (
            query_id TEXT NOT NULL,
            subnarrative_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('primary','supporting')),
            ordinal INTEGER NOT NULL,
            PRIMARY KEY(query_id, subnarrative_id),
            UNIQUE(query_id, role, ordinal),
            FOREIGN KEY(query_id) REFERENCES query_identity(query_id),
            FOREIGN KEY(subnarrative_id)
                REFERENCES subnarrative_identity(subnarrative_id)
        ) STRICT;
        CREATE TABLE retrieval_candidate (
            query_id TEXT NOT NULL,
            document_pk INTEGER NOT NULL,
            source_rank INTEGER NOT NULL,
            source_score REAL NOT NULL,
            best_passage_id TEXT,
            best_passage_raw_logit REAL,
            PRIMARY KEY(query_id, source_rank),
            UNIQUE(query_id, document_pk),
            FOREIGN KEY(query_id) REFERENCES query_identity(query_id),
            FOREIGN KEY(document_pk) REFERENCES document_binding(document_pk)
        ) STRICT;
        CREATE TABLE query_passage (
            query_id TEXT NOT NULL,
            passage_pk INTEGER NOT NULL,
            raw_logit REAL NOT NULL,
            passage_rank INTEGER NOT NULL,
            score_cache_key TEXT NOT NULL,
            PRIMARY KEY(query_id, passage_pk),
            UNIQUE(query_id, passage_rank),
            FOREIGN KEY(query_id) REFERENCES query_identity(query_id),
            FOREIGN KEY(passage_pk) REFERENCES passage(passage_pk)
        ) STRICT;
        CREATE TABLE candidate (
            candidate_pk INTEGER PRIMARY KEY,
            topic_id TEXT NOT NULL,
            candidate_nugget_id TEXT NOT NULL,
            document_pk INTEGER NOT NULL,
            subnarrative_id TEXT NOT NULL,
            candidate_kind TEXT NOT NULL,
            nugget_type TEXT NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            text_sha256 TEXT NOT NULL,
            sentence_score REAL NOT NULL,
            document_subnarrative_rank INTEGER NOT NULL,
            scoring_text_sha256 TEXT NOT NULL,
            subnarrative_sha256 TEXT NOT NULL,
            sentence_splitter_version TEXT NOT NULL,
            UNIQUE(topic_id, candidate_nugget_id),
            UNIQUE(candidate_pk, document_pk),
            FOREIGN KEY(document_pk)
                REFERENCES document_binding(document_pk),
            FOREIGN KEY(topic_id, subnarrative_id)
                REFERENCES subnarrative_identity(topic_id, subnarrative_id)
        ) STRICT;
        CREATE TABLE candidate_span (
            candidate_pk INTEGER NOT NULL,
            role TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            start_byte INTEGER NOT NULL,
            end_byte INTEGER NOT NULL,
            text_sha256 TEXT NOT NULL,
            cross_encoder_score REAL,
            PRIMARY KEY(candidate_pk, role, ordinal),
            FOREIGN KEY(candidate_pk)
                REFERENCES candidate(candidate_pk)
        ) STRICT;
        CREATE TABLE passage (
            passage_pk INTEGER PRIMARY KEY,
            document_pk INTEGER NOT NULL,
            passage_id TEXT NOT NULL,
            source_start_char INTEGER NOT NULL,
            source_end_char INTEGER NOT NULL,
            source_start_byte INTEGER NOT NULL,
            source_end_byte INTEGER NOT NULL,
            source_text_sha256 TEXT NOT NULL,
            scoring_text_sha256 TEXT NOT NULL,
            chunker_identity_json TEXT NOT NULL,
            UNIQUE(passage_id),
            UNIQUE(document_pk, passage_id),
            UNIQUE(passage_pk, document_pk),
            FOREIGN KEY(document_pk)
                REFERENCES document_binding(document_pk)
        ) STRICT;
        CREATE TABLE candidate_passage_link (
            candidate_pk INTEGER NOT NULL,
            document_pk INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            passage_pk INTEGER NOT NULL,
            query_id TEXT NOT NULL,
            PRIMARY KEY(candidate_pk, ordinal),
            UNIQUE(candidate_pk, passage_pk, query_id),
            FOREIGN KEY(candidate_pk, document_pk)
                REFERENCES candidate(candidate_pk, document_pk),
            FOREIGN KEY(passage_pk, document_pk)
                REFERENCES passage(passage_pk, document_pk),
            FOREIGN KEY(query_id, passage_pk)
                REFERENCES query_passage(query_id, passage_pk)
        ) STRICT;
        CREATE TABLE researcher_handoff (
            researcher_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            round_index INTEGER NOT NULL,
            handoff_sha256 TEXT NOT NULL
        ) STRICT;
        CREATE TABLE researcher_evidence (
            researcher_id TEXT NOT NULL,
            subnarrative_id TEXT NOT NULL,
            passage_pk INTEGER NOT NULL,
            relevance TEXT NOT NULL CHECK(relevance IN ('relevant','supporting')),
            PRIMARY KEY(researcher_id, subnarrative_id, passage_pk),
            FOREIGN KEY(researcher_id)
                REFERENCES researcher_handoff(researcher_id),
            FOREIGN KEY(subnarrative_id)
                REFERENCES subnarrative_identity(subnarrative_id),
            FOREIGN KEY(passage_pk)
                REFERENCES passage(passage_pk)
        ) STRICT;
        CREATE TABLE researcher_facet_update (
            researcher_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            subnarrative_id TEXT NOT NULL,
            PRIMARY KEY(researcher_id, ordinal),
            UNIQUE(researcher_id, subnarrative_id),
            FOREIGN KEY(researcher_id)
                REFERENCES researcher_handoff(researcher_id),
            FOREIGN KEY(subnarrative_id)
                REFERENCES subnarrative_identity(subnarrative_id)
        ) STRICT;
        CREATE TABLE topic_completion (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            status TEXT NOT NULL CHECK(status IN ('complete','incomplete')),
            stopping_reason TEXT NOT NULL
        ) STRICT;
        CREATE TABLE stage_seal (
            stage TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            semantic_sha256 TEXT NOT NULL,
            identity_json TEXT NOT NULL,
            row_counts_json TEXT NOT NULL
        ) STRICT;
        """
    )


def _safe_source_slice(source: str, start: int, end: int) -> str:
    return source[start:end] if 0 <= start <= end <= len(source) else ""


def _geometry_for(
    index: DocumentGeometryIndex,
    store: DocumentStore,
    content_sha256: str,
) -> DocumentGeometry:
    """Derive geometry once per content digest and reuse it afterwards."""
    existing = index.get(content_sha256)
    if existing is not None:
        return existing
    return index.admit(content_sha256, store.read_text(content_sha256))


def _check_shared_passage_geometry(
    geometry: DocumentGeometry,
    passage: SourcePassage,
    *,
    label: str = "passage",
) -> str:
    if passage.content_sha256 != geometry.content_sha256:
        raise TopicRecordsIntegrityError(f"{label} content binding is inconsistent")
    _check_offsets(
        geometry,
        passage.start_char,
        passage.end_char,
        passage.start_byte,
        passage.end_byte,
        passage.text_sha256,
        label,
    )
    if geometry.source[passage.start_char:passage.end_char] != passage.text:
        raise TopicRecordsIntegrityError(f"{label} source reconstruction is inconsistent")
    if passage.scoring_text_sha256 != _digest(passage.text):
        raise TopicRecordsIntegrityError(f"{label} scoring text hash is inconsistent")
    identity_json = _canonical_json(dict(passage.chunker_identity))
    if not dict(passage.chunker_identity):
        raise TopicRecordsIntegrityError(f"{label} chunker identity is empty")
    return identity_json


# --------------------------------------------------------------------------- #
# Exact-source checks shared by the rich and the streaming validators
# --------------------------------------------------------------------------- #


def _check_offsets(
    geometry: DocumentGeometry,
    start_char: object,
    end_char: object,
    start_byte: object,
    end_byte: object,
    text_sha256: object,
    label: str,
) -> None:
    source = geometry.source
    offsets = geometry.byte_offsets
    if (
        isinstance(start_char, bool)
        or not isinstance(start_char, int)
        or isinstance(end_char, bool)
        or not isinstance(end_char, int)
        or start_char < 0
        or end_char <= start_char
        or end_char > len(source)
        or offsets[start_char] != start_byte
        or offsets[end_char] != end_byte
        or _digest(source[start_char:end_char]) != text_sha256
    ):
        raise TopicRecordsIntegrityError(f"{label} source span is inconsistent")


def _check_span(geometry: DocumentGeometry, span: SourceSpan, label: str) -> None:
    if not isinstance(span, SourceSpan):
        raise TopicRecordsIntegrityError(f"{label} source span is inconsistent")
    _check_offsets(
        geometry,
        span.start_char,
        span.end_char,
        span.start_byte,
        span.end_byte,
        span.text_sha256,
        label,
    )
    if geometry.source[span.start_char:span.end_char] != span.text:
        raise TopicRecordsIntegrityError(f"{label} source span is inconsistent")


def _check_paragraph_context(
    geometry: DocumentGeometry,
    matched: tuple[int, int],
    before: tuple[int, int] | None,
    after: tuple[int, int] | None,
) -> None:
    if geometry.paragraph_ordinal(matched) is None:
        raise TopicRecordsIntegrityError("matched paragraph is not an exact source paragraph")
    for offset, actual in ((-1, before), (1, after)):
        neighbor = geometry.paragraph_neighbor(matched, offset)
        expected = None if neighbor is None else (neighbor.start_char, neighbor.end_char)
        if actual != expected:
            raise TopicRecordsIntegrityError("candidate paragraph context is inconsistent")


def _check_evidence_membership(
    geometry: DocumentGeometry,
    matched: tuple[int, int],
    candidate_kind: str,
    evidence: Sequence[tuple[int, int]],
) -> None:
    if len(evidence) != (1 if candidate_kind == "exact_sentence" else 2):
        raise TopicRecordsIntegrityError("candidate evidence count is inconsistent")
    ordinals: list[int] = []
    for sentence in evidence:
        ordinal = geometry.sentence_ordinal(matched, sentence)
        if ordinal is None:
            raise TopicRecordsIntegrityError("candidate evidence is not an exact source sentence")
        ordinals.append(ordinal)
    if candidate_kind == "exact_sentence_pair":
        # Same admission rule the builder used. Duplicating the condition here
        # once let the builder emit pairs this check then rejected, which aborts
        # the whole candidate stage rather than dropping one candidate.
        paragraph = geometry.paragraphs[geometry.paragraph_index[matched]]
        sentences = geometry.sentences_for(matched)
        if ordinals[1] != ordinals[0] + 1 or not pair_is_admissible(
            geometry.source,
            paragraph,
            sentences[ordinals[0]],
            sentences[ordinals[1]],
        ):
            raise TopicRecordsIntegrityError("candidate sentence pair is not admitted")


def _check_passage_provenance(
    geometry: DocumentGeometry,
    *,
    scoring_start_char: object,
    scoring_end_char: object,
    source_start_char: object,
    source_end_char: object,
    source_start_byte: object,
    source_end_byte: object,
    source_text_sha256: object,
    scoring_text_sha256: object,
    chunk_text_sha256: object,
    normalization_version: object,
    cross_encoder_score: object,
    cross_encoder_rank: object,
) -> None:
    boundaries = geometry.scoring_boundaries
    if normalization_version != SCORING_NORMALIZATION_VERSION:
        raise TopicRecordsIntegrityError("passage normalization version is inconsistent")
    if (
        isinstance(scoring_start_char, bool)
        or not isinstance(scoring_start_char, int)
        or isinstance(scoring_end_char, bool)
        or not isinstance(scoring_end_char, int)
        or scoring_start_char < 0
        or scoring_end_char <= scoring_start_char
        or scoring_end_char >= len(boundaries)
        or scoring_text_sha256 != geometry.scoring_text_sha256
    ):
        raise TopicRecordsIntegrityError("passage scoring offsets are inconsistent")
    source_start = boundaries[scoring_start_char]
    source_end = boundaries[scoring_end_char]
    chunk = geometry.scoring_text[scoring_start_char:scoring_end_char]
    source_text = geometry.source[source_start:source_end]
    if (
        source_start_char != source_start
        or source_end_char != source_end
        or source_start_byte != geometry.byte_offsets[source_start]
        or source_end_byte != geometry.byte_offsets[source_end]
        or source_text_sha256 != _digest(source_text)
        or chunk_text_sha256 != _digest(chunk)
    ):
        raise TopicRecordsIntegrityError("passage source reconstruction is inconsistent")
    _finite(cross_encoder_score, "passage score")
    _positive_int(cross_encoder_rank, "passage rank")


def _check_candidate_identity(
    *,
    topic_id: str,
    docid: str,
    content_sha256: str,
    subnarrative_id: str,
    subnarrative_sha256: str,
    candidate_kind: str,
    sentences: Sequence[SourceSpan],
    candidate_nugget_id: str,
) -> None:
    """Recompute the candidate nugget ID from its natural identity."""
    expected = _candidate_nugget_id_from_identity(
        topic_id=topic_id,
        document_id=docid,
        document_sha256=content_sha256,
        subnarrative_id=subnarrative_id,
        subnarrative_sha256=subnarrative_sha256,
        candidate_kind=candidate_kind,
        sentences=sentences,
    )
    if expected != candidate_nugget_id:
        raise TopicRecordsIntegrityError(
            "candidate nugget ID does not match its recomputed natural identity"
        )


def _validate_candidate_with_geometry(
    candidate: ExtractiveCandidate,
    geometry: DocumentGeometry,
) -> None:
    """Validate one rich candidate against already-derived document geometry."""
    if candidate.schema_version != CANDIDATE_SCHEMA_VERSION:
        raise TopicRecordsIntegrityError("candidate schema version is inconsistent")
    if candidate.nugget_type != "extractive":
        raise TopicRecordsIntegrityError("candidate nugget type is inconsistent")
    if candidate.candidate_kind not in _CANDIDATE_KINDS:
        raise TopicRecordsIntegrityError("candidate kind is inconsistent")
    if candidate.sentence_splitter_version != SENTENCE_SPLITTER_VERSION:
        raise TopicRecordsIntegrityError("candidate sentence splitter is inconsistent")
    if _CANDIDATE_ID.fullmatch(candidate.candidate_nugget_id) is None:
        raise TopicRecordsIntegrityError("candidate ID is malformed")
    if geometry.content_sha256 != candidate.document_sha256:
        raise TopicRecordsIntegrityError("candidate document hash is inconsistent")
    _require_digest(candidate.subnarrative_sha256, "candidate subnarrative hash")
    if geometry.scoring_text_sha256 != candidate.scoring_text_sha256:
        raise TopicRecordsIntegrityError("candidate scoring text hash is inconsistent")

    _check_span(geometry, candidate.matched_paragraph, "matched paragraph")
    matched = (candidate.matched_paragraph.start_char, candidate.matched_paragraph.end_char)
    for label, context in (
        ("context before", candidate.context_before),
        ("context after", candidate.context_after),
    ):
        if context is not None:
            _check_span(geometry, context, label)
    _check_paragraph_context(
        geometry,
        matched,
        None if candidate.context_before is None else (
            candidate.context_before.start_char, candidate.context_before.end_char
        ),
        None if candidate.context_after is None else (
            candidate.context_after.start_char, candidate.context_after.end_char
        ),
    )

    if len(candidate.evidence_sentences) != (
        1 if candidate.candidate_kind == "exact_sentence" else 2
    ):
        raise TopicRecordsIntegrityError("candidate evidence count is inconsistent")
    for sentence in candidate.evidence_sentences:
        _check_span(geometry, sentence, "candidate evidence")
        _finite(sentence.cross_encoder_score, "candidate sentence score")
    _check_evidence_membership(
        geometry,
        matched,
        candidate.candidate_kind,
        tuple((row.start_char, row.end_char) for row in candidate.evidence_sentences),
    )

    start_char = candidate.evidence_sentences[0].start_char
    end_char = candidate.evidence_sentences[-1].end_char
    if candidate.text != geometry.source[start_char:end_char]:
        raise TopicRecordsIntegrityError("candidate text reconstruction is inconsistent")
    if candidate.sentence_cross_encoder_score != min(
        row.cross_encoder_score for row in candidate.evidence_sentences
    ):
        raise TopicRecordsIntegrityError("candidate score is inconsistent")
    _positive_int(candidate.rank_within_document_subnarrative, "candidate rank")
    _check_candidate_identity(
        topic_id=candidate.topic_id,
        docid=candidate.docid,
        content_sha256=candidate.document_sha256,
        subnarrative_id=candidate.subnarrative_id,
        subnarrative_sha256=candidate.subnarrative_sha256,
        candidate_kind=candidate.candidate_kind,
        sentences=candidate.evidence_sentences,
        candidate_nugget_id=candidate.candidate_nugget_id,
    )

    if not candidate.passages:
        raise TopicRecordsIntegrityError("candidate is missing evidence or passages")
    seen_passages: set[str] = set()
    for passage in candidate.passages:
        if passage.passage_id in seen_passages:
            raise TopicRecordsIntegrityError("candidate passage IDs are not unique")
        seen_passages.add(passage.passage_id)
        _check_passage_provenance(
            geometry,
            scoring_start_char=passage.scoring_start_char,
            scoring_end_char=passage.scoring_end_char,
            source_start_char=passage.source_start_char,
            source_end_char=passage.source_end_char,
            source_start_byte=passage.source_start_byte,
            source_end_byte=passage.source_end_byte,
            source_text_sha256=passage.source_text_sha256,
            scoring_text_sha256=passage.scoring_text_sha256,
            chunk_text_sha256=passage.chunk_text_sha256,
            normalization_version=passage.normalization_version,
            cross_encoder_score=passage.cross_encoder_score,
            cross_encoder_rank=passage.cross_encoder_rank,
        )
        if passage.source_text != geometry.source[
            passage.source_start_char:passage.source_end_char
        ]:
            raise TopicRecordsIntegrityError("passage source reconstruction is inconsistent")


# --------------------------------------------------------------------------- #
# Direct, document-indexed whole-database validation
# --------------------------------------------------------------------------- #


class _PeekCursor:
    """A one-row lookahead over an ordered SQLite cursor."""

    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor
        self._row: tuple[Any, ...] | None = cursor.fetchone()

    def peek(self) -> tuple[Any, ...] | None:
        return self._row

    def take(self) -> tuple[Any, ...]:
        row = self._row
        if row is None:
            raise TopicRecordsIntegrityError("relational stream ended unexpectedly")
        self._row = self._cursor.fetchone()
        return row

    def exhausted(self) -> bool:
        return self._row is None


def _validate_streamed_candidate(
    geometry: DocumentGeometry,
    topic_id: str,
    docid: str,
    content_sha256: str,
    candidate_row: tuple[Any, ...],
    span_rows: Sequence[tuple[Any, ...]],
    link_rows: Sequence[tuple[Any, ...]],
    subnarrative_hashes: Mapping[str, str],
    document_passage_pks: Mapping[int, str],
) -> None:
    (
        _document_pk, _candidate_pk, row_topic_id, candidate_nugget_id, subnarrative_id,
        candidate_kind, nugget_type, start_char, end_char, text_sha256, sentence_score,
        rank, scoring_text_sha256, subnarrative_sha256, sentence_splitter_version,
    ) = candidate_row
    if row_topic_id != topic_id:
        raise TopicRecordsIntegrityError("candidate topic identity is inconsistent")
    if nugget_type != "extractive":
        raise TopicRecordsIntegrityError("candidate nugget type is inconsistent")
    if candidate_kind not in _CANDIDATE_KINDS:
        raise TopicRecordsIntegrityError("candidate kind is inconsistent")
    if sentence_splitter_version != SENTENCE_SPLITTER_VERSION:
        raise TopicRecordsIntegrityError("candidate sentence splitter is inconsistent")
    if _CANDIDATE_ID.fullmatch(str(candidate_nugget_id)) is None:
        raise TopicRecordsIntegrityError("candidate ID is malformed")
    if scoring_text_sha256 != geometry.scoring_text_sha256:
        raise TopicRecordsIntegrityError("candidate scoring text hash is inconsistent")
    _require_digest(subnarrative_sha256, "candidate subnarrative hash")
    if subnarrative_hashes.get(subnarrative_id) != subnarrative_sha256:
        raise TopicRecordsIntegrityError("candidate subnarrative identity is inconsistent")

    evidence: list[SourceSpan] = []
    matched: tuple[int, int] | None = None
    context: dict[str, tuple[int, int]] = {}
    for span_row in span_rows:
        role, ordinal = span_row[2], span_row[3]
        span_start, span_end = span_row[4], span_row[5]
        label = "candidate evidence" if role == "evidence" else str(role).replace("_", " ")
        _check_offsets(
            geometry, span_start, span_end, span_row[6], span_row[7], span_row[8], label
        )
        score = span_row[9]
        if role == "evidence":
            if ordinal != len(evidence):
                raise TopicRecordsIntegrityError("evidence span ordinals are not contiguous")
            if score is None:
                raise TopicRecordsIntegrityError("evidence span score is null")
            _finite(score, "candidate sentence score")
            evidence.append(
                SourceSpan(
                    geometry.source[span_start:span_end],
                    span_start,
                    span_end,
                    span_row[6],
                    span_row[7],
                    span_row[8],
                )
            )
        elif role in {"matched_paragraph", "context_before", "context_after"}:
            if ordinal != 0 or score is not None:
                raise TopicRecordsIntegrityError("candidate span role is invalid")
            if role == "matched_paragraph":
                if matched is not None:
                    raise TopicRecordsIntegrityError("candidate has duplicate matched paragraphs")
                matched = (span_start, span_end)
            else:
                if role in context:
                    raise TopicRecordsIntegrityError("candidate span role is invalid")
                context[role] = (span_start, span_end)
        else:
            raise TopicRecordsIntegrityError("candidate span role is invalid")
    if matched is None:
        raise TopicRecordsIntegrityError("candidate has no matched paragraph")
    if not evidence:
        raise TopicRecordsIntegrityError("candidate is missing evidence or passages")

    _check_paragraph_context(
        geometry, matched, context.get("context_before"), context.get("context_after")
    )
    _check_evidence_membership(
        geometry,
        matched,
        candidate_kind,
        tuple((row.start_char, row.end_char) for row in evidence),
    )

    if (
        start_char != evidence[0].start_char
        or end_char != evidence[-1].end_char
        or text_sha256 != _digest(geometry.source[start_char:end_char])
    ):
        raise TopicRecordsIntegrityError("candidate offsets or text hash are inconsistent")
    evidence_scores = tuple(row[9] for row in span_rows if row[2] == "evidence")
    if float(sentence_score) != min(float(value) for value in evidence_scores):
        raise TopicRecordsIntegrityError("candidate score is inconsistent")
    _finite(sentence_score, "candidate score")
    _positive_int(rank, "candidate rank")
    _check_candidate_identity(
        topic_id=topic_id,
        docid=docid,
        content_sha256=content_sha256,
        subnarrative_id=subnarrative_id,
        subnarrative_sha256=subnarrative_sha256,
        candidate_kind=candidate_kind,
        sentences=evidence,
        candidate_nugget_id=candidate_nugget_id,
    )

    if not link_rows:
        raise TopicRecordsIntegrityError("candidate is missing evidence or passages")
    seen_passages: set[str] = set()
    for ordinal, link_row in enumerate(link_rows):
        if link_row[2] != ordinal:
            raise TopicRecordsIntegrityError("passage ordinals are not contiguous")
        passage_id = document_passage_pks.get(link_row[3])
        if passage_id is None:
            raise TopicRecordsIntegrityError("candidate passage link crosses documents")
        if passage_id in seen_passages:
            raise TopicRecordsIntegrityError("candidate passage IDs are not unique")
        seen_passages.add(passage_id)


def _validate_sources(
    database: sqlite3.Connection,
    store: DocumentStore,
    topic_id: str,
    *,
    geometry_index: DocumentGeometryIndex | None = None,
) -> tuple[tuple[str, int, int], ...]:
    """Validate every derived record directly against its bound document body.

    Geometry is derived once per bound document, each unique passage is checked
    once, and candidate/span/link rows are streamed in ``(document, candidate)``
    order without reconstructing rich candidate objects.  The return value is
    the verified content-addressed-store receipt closure.
    """
    subnarrative_hashes: dict[str, str] = {}
    for subnarrative_id, subnarrative_sha256 in database.execute(
        "SELECT subnarrative_id, subnarrative_sha256 FROM subnarrative_identity "
        "WHERE topic_id=?",
        (topic_id,),
    ):
        _require_digest(subnarrative_sha256, "subnarrative hash")
        subnarrative_hashes[subnarrative_id] = subnarrative_sha256

    candidates = _PeekCursor(database.execute(
        "SELECT c.document_pk, c.candidate_pk, c.topic_id, c.candidate_nugget_id, "
        "c.subnarrative_id, c.candidate_kind, c.nugget_type, c.start_char, c.end_char, "
        "c.text_sha256, c.sentence_score, c.document_subnarrative_rank, "
        "c.scoring_text_sha256, c.subnarrative_sha256, c.sentence_splitter_version "
        "FROM candidate AS c ORDER BY c.document_pk, c.candidate_pk"
    ))
    spans = _PeekCursor(database.execute(
        "SELECT c.document_pk, s.candidate_pk, s.role, s.ordinal, s.start_char, s.end_char, "
        "s.start_byte, s.end_byte, s.text_sha256, s.cross_encoder_score "
        "FROM candidate_span AS s JOIN candidate AS c ON c.candidate_pk = s.candidate_pk "
        "ORDER BY c.document_pk, s.candidate_pk, s.role, s.ordinal"
    ))
    links = _PeekCursor(database.execute(
        "SELECT document_pk, candidate_pk, ordinal, passage_pk, query_id "
        "FROM candidate_passage_link "
        "ORDER BY document_pk, candidate_pk, ordinal"
    ))

    receipts: list[tuple[str, int, int]] = []
    for document_pk, docid, content_sha256, character_count, byte_count in database.execute(
        "SELECT document_pk, docid, content_sha256, character_count, byte_count "
        "FROM document_binding WHERE topic_id=? ORDER BY document_pk",
        (topic_id,),
    ).fetchall():
        _require_digest(content_sha256, f"document binding {docid!r} content hash")
        try:
            receipt = store.verify(content_sha256)
        except DocumentStoreIntegrityError as exc:
            raise TopicRecordsIntegrityError(
                f"document binding {docid!r} is invalid"
            ) from exc
        if receipt.character_count != character_count or receipt.byte_count != byte_count:
            raise TopicRecordsIntegrityError("document binding counts are inconsistent")
        index = DocumentGeometryIndex() if geometry_index is None else geometry_index
        geometry = _geometry_for(index, store, content_sha256)
        if geometry.character_count != character_count or geometry.byte_count != byte_count:
            raise TopicRecordsIntegrityError("document binding counts are inconsistent")

        document_passage_pks: dict[int, str] = {}
        for passage_row in database.execute(
            "SELECT passage_pk, passage_id, source_start_char, source_end_char, "
            "source_start_byte, source_end_byte, source_text_sha256, scoring_text_sha256, "
            "chunker_identity_json "
            "FROM passage WHERE document_pk=? ORDER BY passage_pk",
            (document_pk,),
        ):
            _require_text(passage_row[1], "passage_id")
            _check_offsets(
                geometry,
                passage_row[2],
                passage_row[3],
                passage_row[4],
                passage_row[5],
                passage_row[6],
                "passage",
            )
            try:
                identity = json.loads(passage_row[8])
            except (TypeError, json.JSONDecodeError) as exc:
                raise TopicRecordsIntegrityError("passage chunker identity is invalid") from exc
            if (
                not isinstance(identity, dict)
                or not identity
                or _canonical_json(identity) != passage_row[8]
            ):
                raise TopicRecordsIntegrityError("passage chunker identity is not canonical")
            passage_text_sha256 = _digest(
                geometry.source[passage_row[2]:passage_row[3]]
            )
            if passage_row[7] != passage_text_sha256:
                raise TopicRecordsIntegrityError(
                    "passage scoring text hash is inconsistent"
                )
            if identity.get("backend") == "legacy-candidate":
                score_row = database.execute(
                    "SELECT raw_logit, passage_rank FROM query_passage "
                    "WHERE passage_pk=? ORDER BY query_id LIMIT 1",
                    (passage_row[0],),
                ).fetchone()
                if score_row is None:
                    raise TopicRecordsIntegrityError("legacy passage score is absent")
                try:
                    scoring_start = geometry.scoring_boundaries.index(passage_row[2])
                    scoring_end = geometry.scoring_boundaries.index(passage_row[3])
                except ValueError as exc:
                    raise TopicRecordsIntegrityError(
                        "legacy passage source geometry is not a scoring projection"
                    ) from exc
                _check_passage_provenance(
                    geometry,
                    scoring_start_char=scoring_start,
                    scoring_end_char=scoring_end,
                    source_start_char=passage_row[2],
                    source_end_char=passage_row[3],
                    source_start_byte=passage_row[4],
                    source_end_byte=passage_row[5],
                    source_text_sha256=passage_row[6],
                    scoring_text_sha256=geometry.scoring_text_sha256,
                    chunk_text_sha256=_digest(
                        geometry.scoring_text[scoring_start:scoring_end]
                    ),
                    normalization_version=SCORING_NORMALIZATION_VERSION,
                    cross_encoder_score=score_row[0],
                    cross_encoder_rank=score_row[1],
                )
            document_passage_pks[passage_row[0]] = passage_row[1]

        while True:
            candidate_row = candidates.peek()
            if candidate_row is None or candidate_row[0] != document_pk:
                break
            candidates.take()
            key = (candidate_row[0], candidate_row[1])
            span_rows: list[tuple[Any, ...]] = []
            while True:
                row = spans.peek()
                if row is None or (row[0], row[1]) != key:
                    break
                span_rows.append(spans.take())
            span_rows.sort(key=lambda row: (_ROLE_ORDER.get(row[2], len(_SPAN_ROLES)), row[3]))
            link_rows: list[tuple[Any, ...]] = []
            while True:
                row = links.peek()
                if row is None or (row[0], row[1]) != key:
                    break
                link_rows.append(links.take())
            _validate_streamed_candidate(
                geometry,
                topic_id,
                docid,
                content_sha256,
                candidate_row,
                span_rows,
                link_rows,
                subnarrative_hashes,
                document_passage_pks,
            )

        if geometry_index is None:
            del index
        receipts.append((content_sha256, character_count, byte_count))

    if not candidates.exhausted() or not spans.exhausted() or not links.exhausted():
        raise TopicRecordsIntegrityError("relational rows reference an unbound document")
    return tuple(sorted(set(receipts)))


def _stored_facets(database: sqlite3.Connection) -> tuple[FacetRecord, ...]:
    result: list[FacetRecord] = []
    for subnarrative_id, text, origin in database.execute(
        "SELECT subnarrative_id, subnarrative_text, origin FROM subnarrative_identity "
        "ORDER BY subnarrative_id"
    ):
        if not text:
            continue
        facet = FacetRecord(subnarrative_id, text, origin)
        if _digest(text) != database.execute(
            "SELECT subnarrative_sha256 FROM subnarrative_identity WHERE subnarrative_id=?",
            (subnarrative_id,),
        ).fetchone()[0]:
            raise TopicRecordsIntegrityError("facet text hash is inconsistent")
        result.append(facet)
    return tuple(result)


def _stored_passage(
    database: sqlite3.Connection,
    store: DocumentStore,
    geometry_index: DocumentGeometryIndex,
    query_id: str,
    passage_pk: int,
    raw_logit: float,
    passage_rank: int,
    score_cache_key: str,
) -> SourcePassage:
    row = database.execute(
        "SELECT p.passage_id, d.docid, d.content_sha256, r.source_rank, r.source_score, "
        "p.source_start_char, p.source_end_char, p.source_start_byte, p.source_end_byte, "
        "p.source_text_sha256, p.scoring_text_sha256, p.chunker_identity_json "
        "FROM passage AS p JOIN document_binding AS d ON d.document_pk=p.document_pk "
        "JOIN retrieval_candidate AS r ON r.query_id=? AND r.document_pk=p.document_pk "
        "WHERE p.passage_pk=?",
        (query_id, passage_pk),
    ).fetchone()
    if row is None:
        raise TopicRecordsIntegrityError("query passage source binding is absent")
    geometry = _geometry_for(geometry_index, store, row[2])
    _check_offsets(
        geometry,
        row[5], row[6], row[7], row[8], row[9], "passage"
    )
    try:
        identity = json.loads(row[11])
    except (TypeError, json.JSONDecodeError) as exc:
        raise TopicRecordsIntegrityError("passage chunker identity is invalid") from exc
    if not isinstance(identity, dict) or not identity or _canonical_json(identity) != row[11]:
        raise TopicRecordsIntegrityError("passage chunker identity is not canonical")
    text = geometry.source[row[5]:row[6]]
    if row[10] != _digest(text):
        raise TopicRecordsIntegrityError("passage scoring text hash is inconsistent")
    return SourcePassage(
        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
        row[9], text, raw_logit, passage_rank, score_cache_key, row[10], identity,
    )


def _stored_passage_search(
    database: sqlite3.Connection,
    store: DocumentStore,
    geometry_index: DocumentGeometryIndex,
    query_id: str,
) -> PassageSearchSnapshot:
    query_row = database.execute(
        "SELECT query_id, query_text_sha256, primary_subnarrative_id, status, stopping_reason, "
        "attempt_count, requested_documents, returned_documents, scored_documents, "
        "scored_passages, source_exhausted FROM query_identity WHERE query_id=?",
        (query_id,),
    ).fetchone()
    if query_row is None:
        raise TopicRecordsIntegrityError("requested passage search is absent")
    facets = tuple(
        row[0]
        for row in database.execute(
            "SELECT subnarrative_id FROM query_facet WHERE query_id=? "
            "ORDER BY ordinal",
            (query_id,),
        )
    )
    if not facets or facets[0] != query_row[2]:
        raise TopicRecordsIntegrityError("query facet identity is inconsistent")
    documents = tuple(
        SourceDocument(*row)
        for row in database.execute(
            "SELECT d.docid, d.content_sha256, r.source_rank, r.source_score, "
            "r.best_passage_id, r.best_passage_raw_logit "
            "FROM retrieval_candidate AS r JOIN document_binding AS d "
            "ON d.document_pk=r.document_pk WHERE r.query_id=? ORDER BY r.source_rank",
            (query_id,),
        )
    )
    passages = tuple(
        _stored_passage(database, store, geometry_index, query_id, row[0], row[1], row[2], row[3])
        for row in database.execute(
            "SELECT passage_pk, raw_logit, passage_rank, score_cache_key "
            "FROM query_passage WHERE query_id=? ORDER BY passage_rank",
            (query_id,),
        )
    )
    query = StoredQuery(
        query_row[0], query_row[1], query_row[2], tuple(facets[1:]), query_row[3], query_row[4],
        query_row[5], query_row[6], query_row[7], query_row[8], query_row[9], bool(query_row[10]),
    )
    return PassageSearchSnapshot(query, documents, passages)


def _materialize_topic_snapshot(
    database: sqlite3.Connection,
    store: DocumentStore,
    geometry_index: DocumentGeometryIndex,
) -> TopicEvidenceSnapshot:
    database.execute("BEGIN")
    try:
        facets = _stored_facets(database)
        queries = tuple(
            _stored_passage_search(database, store, geometry_index, row[0])
            for row in database.execute("SELECT query_id FROM query_identity ORDER BY query_id")
        )
        documents_by_id: dict[tuple[str, str], SourceDocument] = {}
        passages_by_id: dict[str, SourcePassage] = {}
        for query in queries:
            for document in query.documents:
                documents_by_id[(document.docid, document.content_sha256)] = document
            for passage in query.passages:
                passages_by_id.setdefault(passage.passage_id, passage)
        topic_row = database.execute("SELECT topic_id FROM topic_identity").fetchall()
        if len(topic_row) != 1:
            raise TopicRecordsIntegrityError("topic identity is not singular")
        candidates: list[ExtractiveCandidate] = []
        for subnarrative_id, candidate_id in database.execute(
            "SELECT subnarrative_id, candidate_nugget_id FROM candidate "
            "WHERE topic_id=? ORDER BY subnarrative_id, candidate_nugget_id",
            (topic_row[0][0],),
        ):
            candidate, geometry = _reconstruct_candidate(
                database,
                store,
                topic_row[0][0],
                subnarrative_id,
                candidate_id,
                geometry_index=geometry_index,
            )
            _validate_candidate_with_geometry(candidate, geometry)
            candidates.append(candidate)
        evidence_rows = tuple(
            (row[0], ResearcherEvidence(row[1], row[2], row[3]))
            for row in database.execute(
                "SELECT e.researcher_id, e.subnarrative_id, p.passage_id, e.relevance "
                "FROM researcher_evidence AS e JOIN passage AS p ON p.passage_pk=e.passage_pk "
                "ORDER BY e.researcher_id, e.subnarrative_id, p.passage_id"
            )
        )
        evidence = tuple(row[1] for row in evidence_rows)
        facet_updates_by_researcher: dict[str, list[FacetRecord]] = {}
        for researcher_id, ordinal, facet_id, text, origin, text_sha256 in database.execute(
            "SELECT u.researcher_id, u.ordinal, f.subnarrative_id, f.subnarrative_text, "
            "f.origin, f.subnarrative_sha256 FROM researcher_facet_update AS u "
            "JOIN subnarrative_identity AS f ON f.subnarrative_id=u.subnarrative_id "
            "ORDER BY u.researcher_id, u.ordinal"
        ):
            rows = facet_updates_by_researcher.setdefault(researcher_id, [])
            if ordinal != len(rows):
                raise TopicRecordsIntegrityError("researcher facet update ordinals are not contiguous")
            facet = FacetRecord(facet_id, text, origin)
            if facet.text_sha256 != text_sha256:
                raise TopicRecordsIntegrityError("researcher facet update identity is inconsistent")
            rows.append(facet)
        handoffs = tuple(
            ResearcherHandoff(
                row[1], row[0], row[2],
                tuple(item for researcher_id, item in evidence_rows if researcher_id == row[0]),
                tuple(facet_updates_by_researcher.get(row[0], ())),
            )
            for row in database.execute(
                "SELECT researcher_id, run_id, round_index FROM researcher_handoff "
                "ORDER BY researcher_id"
            )
        )
        completion = database.execute(
            "SELECT status, stopping_reason FROM topic_completion WHERE singleton=1"
        ).fetchone()
        return TopicEvidenceSnapshot(
            facets,
            queries,
            tuple(
                documents_by_id[key]
                for key in sorted(documents_by_id)
            ),
            tuple(passages_by_id[key] for key in sorted(passages_by_id)),
            handoffs,
            evidence,
            tuple(candidates),
            None if completion is None else completion[0],
            None if completion is None else completion[1],
        )
    finally:
        database.rollback()


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


@dataclass
class _PendingBatch:
    """Rows staged for one bounded per-document transaction."""

    subnarratives: list[tuple[Any, ...]] = field(default_factory=list)
    queries: list[tuple[Any, ...]] = field(default_factory=list)
    query_facets: list[tuple[Any, ...]] = field(default_factory=list)
    retrieval_candidates: list[tuple[Any, ...]] = field(default_factory=list)
    query_passages: list[tuple[Any, ...]] = field(default_factory=list)
    passages: list[tuple[Any, ...]] = field(default_factory=list)
    candidates: list[tuple[Any, ...]] = field(default_factory=list)
    spans: list[tuple[Any, ...]] = field(default_factory=list)
    links: list[tuple[Any, ...]] = field(default_factory=list)

    def empty(self) -> bool:
        return not (
            self.subnarratives or self.queries or self.query_facets
            or self.retrieval_candidates or self.query_passages
            or self.passages or self.candidates or self.spans or self.links
        )

    def clear(self) -> None:
        self.subnarratives.clear()
        self.queries.clear()
        self.query_facets.clear()
        self.retrieval_candidates.clear()
        self.query_passages.clear()
        self.passages.clear()
        self.candidates.clear()
        self.spans.clear()
        self.links.clear()


@dataclass
class _CandidateStage:
    """Candidate-local overlays merged only after every derived row validates."""

    batch: _PendingBatch = field(default_factory=_PendingBatch)
    candidate_ids: set[str] = field(default_factory=set)
    subnarratives: dict[str, str] = field(default_factory=dict)
    passage_pk_by_id: dict[str, int] = field(default_factory=dict)
    passage_rows: dict[int, tuple[Any, ...]] = field(default_factory=dict)
    next_candidate_pk: int = 0
    next_passage_pk: int = 0


class TopicRecordsBuilder:
    """Build and publish one immutable topic records database."""

    def __init__(
        self,
        destination: Path,
        topic_id: str,
        document_store: DocumentStore,
        *,
        run_id: str,
    ):
        self._destination = Path(destination)
        self._operation_lock = RLock()
        self._database_path, self._manifest_path = _paths(self._destination)
        self._store = document_store
        self._topic_id = _require_text(topic_id, "topic_id")
        self._run_id = _require_text(run_id, "run_id")
        if not isinstance(document_store, DocumentStore):
            raise TypeError("document_store must be a DocumentStore")
        self._destination.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._destination, prefix=".records-", suffix=".sqlite3"
        )
        os.close(descriptor)
        self._temporary_path = Path(temporary_name)
        try:
            self._database = sqlite3.connect(
                self._temporary_path,
                check_same_thread=False,
            )
            _configure(self._database)
            _create_schema(self._database)
            self._database.execute(
                "INSERT INTO topic_identity VALUES (?, ?)",
                (self._topic_id, self._run_id),
            )
            self._database.commit()
        except BaseException:
            self._cleanup()
            raise
        self._published: PublishedTopicRecords | None = None
        self._published_identity_json: str | None = None
        self._failed = False
        self._batch = _PendingBatch()
        self._geometry_index = DocumentGeometryIndex()
        self._active_document_pk: int | None = None
        self._passage_pk_by_id: dict[str, int] = {}
        self._passage_rows: dict[int, tuple[Any, ...]] = {}
        self._documents: dict[str, tuple[int, str]] = {}
        self._subnarratives: dict[str, str] = {}
        self._facets: dict[str, FacetRecord] = {}
        self._query_ids: set[str] = set()
        # Committed query-passage rows only. Rows still in ``_batch`` are
        # consulted explicitly by add_candidate until the flush commits them.
        self._query_passage_rows: dict[tuple[str, int], tuple[Any, ...]] = {}
        self._candidate_ids: set[str] = set()
        self._next_document_pk = 1
        self._next_candidate_pk = 1
        self._next_passage_pk = 1

    @property
    def topic_id(self) -> str:
        """Immutable topic identity established when this builder was opened."""
        return self._topic_id

    @property
    def run_id(self) -> str:
        """Immutable run identity established when this builder was opened."""
        return self._run_id

    def _cleanup(self) -> None:
        database = getattr(self, "_database", None)
        if database is not None:
            database.close()
            self._database = None
        temporary = getattr(self, "_temporary_path", None)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            Path(str(temporary) + "-wal").unlink(missing_ok=True)
            Path(str(temporary) + "-shm").unlink(missing_ok=True)

    def _require_open(self) -> sqlite3.Connection:
        if self._failed:
            raise TopicRecordsIntegrityError("topic records builder failed and cannot continue")
        if self._database is None:
            raise TopicRecordsIntegrityError("topic records builder is closed")
        return self._database

    # -- batching ---------------------------------------------------------- #

    def _flush(self) -> None:
        if self._batch.empty():
            return
        database = self._require_open()
        database.execute("BEGIN IMMEDIATE")
        try:
            database.executemany(
                "INSERT INTO subnarrative_identity "
                "(topic_id, subnarrative_id, subnarrative_sha256, subnarrative_text, origin) "
                "VALUES (?, ?, ?, ?, ?)",
                self._batch.subnarratives,
            )
            database.executemany(
                "INSERT INTO query_identity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._batch.queries,
            )
            database.executemany(
                "INSERT INTO query_facet VALUES (?, ?, ?, ?)",
                self._batch.query_facets,
            )
            database.executemany(
                "INSERT INTO retrieval_candidate VALUES (?, ?, ?, ?, ?, ?)",
                self._batch.retrieval_candidates,
            )
            database.executemany(
                "INSERT INTO passage "
                "(passage_pk, document_pk, passage_id, source_start_char, source_end_char, "
                "source_start_byte, source_end_byte, source_text_sha256, scoring_text_sha256, "
                "chunker_identity_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._batch.passages,
            )
            database.executemany(
                "INSERT INTO query_passage VALUES (?, ?, ?, ?, ?)",
                self._batch.query_passages,
            )
            database.executemany(
                "INSERT INTO candidate VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._batch.candidates,
            )
            database.executemany(
                "INSERT INTO candidate_span VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._batch.spans,
            )
            database.executemany(
                "INSERT INTO candidate_passage_link VALUES (?, ?, ?, ?, ?)",
                self._batch.links,
            )
        except BaseException:
            database.rollback()
            self._failed = True
            raise
        else:
            database.commit()
            for row in self._batch.query_passages:
                self._query_passage_rows[(row[0], row[1])] = tuple(row)
            self._batch.clear()

    def _activate_document(self, document_pk: int, content_sha256: str) -> DocumentGeometry:
        if self._active_document_pk != document_pk:
            self._flush()
            self._geometry_index = DocumentGeometryIndex()
            self._passage_pk_by_id = {}
            self._passage_rows = {}
            self._active_document_pk = document_pk
        return _geometry_for(self._geometry_index, self._store, content_sha256)

    # -- public seam ------------------------------------------------------- #

    @_serialized_builder_operation
    def bind_document(
        self,
        docid: str,
        exact_text: str,
        expected_sha256: str | None = None,
    ) -> DocumentReceipt:
        _require_text(docid, "docid")
        if not isinstance(exact_text, str):
            raise TopicRecordsIntegrityError("exact document text must be a string")
        try:
            receipt = self._store.admit_text(exact_text, expected_sha256=expected_sha256)
            database = self._require_open()
            existing = self._documents.get(docid)
            values = (
                self._topic_id,
                docid,
                receipt.content_sha256,
                receipt.character_count,
                receipt.byte_count,
            )
            if existing is not None:
                if existing[1] != receipt.content_sha256:
                    raise TopicRecordsIntegrityError(
                        "document binding conflicts with existing topic document binding"
                    )
                return receipt
            database.execute("BEGIN IMMEDIATE")
            try:
                stored = database.execute(
                    "SELECT document_pk, content_sha256, character_count, byte_count "
                    "FROM document_binding WHERE topic_id=? AND docid=?",
                    (self._topic_id, docid),
                ).fetchone()
                if stored is not None:
                    if tuple(stored[1:]) != values[2:]:
                        raise TopicRecordsIntegrityError(
                            "document binding conflicts with existing topic document binding"
                        )
                    document_pk = int(stored[0])
                    self._next_document_pk = max(self._next_document_pk, document_pk + 1)
                else:
                    document_pk = self._next_document_pk
                    database.execute(
                        "INSERT INTO document_binding "
                        "(document_pk, topic_id, docid, content_sha256, character_count, byte_count) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (document_pk, *values),
                    )
            except BaseException:
                database.rollback()
                raise
            else:
                database.commit()
                if stored is None:
                    self._next_document_pk += 1
                self._documents[docid] = (document_pk, receipt.content_sha256)
            return receipt
        except TopicRecordsIntegrityError:
            raise
        except (DocumentStoreIntegrityError, sqlite3.Error) as exc:
            raise TopicRecordsIntegrityError(f"unable to bind document {docid!r}") from exc

    @_serialized_builder_operation
    def add_facets(self, facets: Sequence[FacetRecord]) -> None:
        if not isinstance(facets, Sequence) or isinstance(facets, (str, bytes)):
            raise TypeError("facets must be a sequence of FacetRecord rows")
        rows = tuple(facets)
        if any(not isinstance(row, FacetRecord) for row in rows):
            raise TypeError("facets must contain FacetRecord rows")
        if len({row.subnarrative_id for row in rows}) != len(rows):
            raise TopicRecordsIntegrityError("facet IDs are not unique")
        database = self._require_open()
        self._flush()
        database.execute("BEGIN IMMEDIATE")
        try:
            for facet in rows:
                stored = database.execute(
                    "SELECT subnarrative_sha256, subnarrative_text, origin "
                    "FROM subnarrative_identity WHERE subnarrative_id=?",
                    (facet.subnarrative_id,),
                ).fetchone()
                if stored is None:
                    database.execute(
                        "INSERT INTO subnarrative_identity "
                        "(topic_id, subnarrative_id, subnarrative_sha256, subnarrative_text, origin) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            self._topic_id,
                            facet.subnarrative_id,
                            facet.text_sha256,
                            facet.text,
                            facet.origin,
                        ),
                    )
                elif stored[0] != facet.text_sha256 or (
                    stored[1] and stored[1] != facet.text
                ) or (stored[1] and stored[2] != facet.origin):
                    raise TopicRecordsIntegrityError("facet identity conflicts")
                elif not stored[1]:
                    database.execute(
                        "UPDATE subnarrative_identity SET subnarrative_text=?, "
                        "subnarrative_sha256=?, origin=? WHERE subnarrative_id=?",
                        (facet.text, facet.text_sha256, facet.origin, facet.subnarrative_id),
                    )
        except BaseException:
            database.rollback()
            raise
        else:
            database.commit()
            for facet in rows:
                self._subnarratives[facet.subnarrative_id] = facet.text_sha256
                self._facets[facet.subnarrative_id] = facet

    def _result_document_bindings(
        self,
        database: sqlite3.Connection,
        result: PassageSearchResult,
    ) -> dict[str, tuple[int, str]]:
        bindings: dict[str, tuple[int, str]] = {}
        next_document_pk = max(
            self._next_document_pk,
            int(database.execute(
                "SELECT COALESCE(MAX(document_pk), 0) FROM document_binding"
            ).fetchone()[0]) + 1,
        )
        for document in result.documents:
            if document.docid in bindings:
                raise TopicRecordsIntegrityError("search document IDs are not unique")
            try:
                receipt = self._store.verify(document.content_sha256)
            except DocumentStoreIntegrityError as exc:
                raise TopicRecordsIntegrityError("search document CAS binding is invalid") from exc
            geometry = _geometry_for(self._geometry_index, self._store, document.content_sha256)
            if receipt.character_count != geometry.character_count or receipt.byte_count != geometry.byte_count:
                raise TopicRecordsIntegrityError("search document CAS counts are inconsistent")
            stored = database.execute(
                "SELECT document_pk, content_sha256, character_count, byte_count "
                "FROM document_binding WHERE topic_id=? AND docid=?",
                (self._topic_id, document.docid),
            ).fetchone()
            expected = (
                document.content_sha256,
                geometry.character_count,
                geometry.byte_count,
            )
            if stored is not None:
                if tuple(stored[1:]) != expected:
                    raise TopicRecordsIntegrityError("search document binding conflicts")
                bindings[document.docid] = (int(stored[0]), document.content_sha256)
            else:
                document_pk = next_document_pk
                next_document_pk += 1
                database.execute(
                    "INSERT INTO document_binding "
                    "(document_pk, topic_id, docid, content_sha256, character_count, byte_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (document_pk, self._topic_id, document.docid, *expected),
                )
                bindings[document.docid] = (document_pk, document.content_sha256)
        self._next_document_pk = next_document_pk
        return bindings

    @staticmethod
    def _query_identity_row(result: PassageSearchResult) -> tuple[Any, ...]:
        return (
            result.query.query_id,
            _digest(result.query.text),
            result.query.primary_subnarrative_id,
            result.status,
            result.stopping_reason,
            result.attempt_count,
            result.requested_documents,
            result.returned_documents,
            result.scored_documents,
            result.scored_passages,
            int(result.source_exhausted),
        )

    @_serialized_builder_operation
    def add_passage_search(self, result: PassageSearchResult) -> None:
        if not isinstance(result, PassageSearchResult):
            raise TypeError("result must be a PassageSearchResult")
        database = self._require_open()
        self._flush()
        facet_ids = (
            result.query.primary_subnarrative_id,
            *result.query.supporting_subnarrative_ids,
        )
        database.execute("BEGIN IMMEDIATE")
        try:
            for facet_id in facet_ids:
                if database.execute(
                    "SELECT 1 FROM subnarrative_identity WHERE subnarrative_id=?",
                    (facet_id,),
                ).fetchone() is None:
                    raise TopicRecordsIntegrityError("query facet is not admitted")

            existing_query = database.execute(
                "SELECT query_id, query_text_sha256, primary_subnarrative_id, status, "
                "stopping_reason, attempt_count, requested_documents, returned_documents, "
                "scored_documents, scored_passages, source_exhausted "
                "FROM query_identity WHERE query_id=?",
                (result.query.query_id,),
            ).fetchone()
            query_row = self._query_identity_row(result)
            if existing_query is not None and tuple(existing_query) != query_row:
                raise TopicRecordsIntegrityError("query identity conflicts")
            if existing_query is None:
                database.execute(
                    "INSERT INTO query_identity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    query_row,
                )

            query_facets = (
                (result.query.query_id, result.query.primary_subnarrative_id, "primary", 0),
                *(
                    (result.query.query_id, facet_id, "supporting", ordinal)
                    for ordinal, facet_id in enumerate(result.query.supporting_subnarrative_ids, 1)
                ),
            )
            stored_query_facets = tuple(database.execute(
                "SELECT query_id, subnarrative_id, role, ordinal FROM query_facet "
                "WHERE query_id=? ORDER BY ordinal",
                (result.query.query_id,),
            ).fetchall())
            if stored_query_facets and stored_query_facets != query_facets:
                raise TopicRecordsIntegrityError("query facet identity conflicts")
            if not stored_query_facets:
                database.executemany(
                    "INSERT INTO query_facet VALUES (?, ?, ?, ?)", query_facets
                )

            bindings = self._result_document_bindings(database, result)
            passage_pks: dict[str, int] = {}
            next_passage_pk = max(
                self._next_passage_pk,
                int(database.execute(
                    "SELECT COALESCE(MAX(passage_pk), 0) FROM passage"
                ).fetchone()[0]) + 1,
            )
            for passage in result.passages:
                document_pk, _ = bindings[passage.docid]
                geometry = _geometry_for(self._geometry_index, self._store, passage.content_sha256)
                identity_json = _check_shared_passage_geometry(geometry, passage)
                expected_geometry = (
                    passage.start_char,
                    passage.end_char,
                    passage.start_byte,
                    passage.end_byte,
                    passage.text_sha256,
                    passage.scoring_text_sha256,
                    identity_json,
                )
                stored = database.execute(
                    "SELECT passage_pk, source_start_char, source_end_char, source_start_byte, "
                    "source_end_byte, source_text_sha256, scoring_text_sha256, chunker_identity_json "
                    "FROM passage WHERE document_pk=? AND passage_id=?",
                    (document_pk, passage.passage_id),
                ).fetchone()
                if stored is None:
                    passage_pk = next_passage_pk
                    next_passage_pk += 1
                    database.execute(
                        "INSERT INTO passage "
                        "(passage_pk, document_pk, passage_id, source_start_char, source_end_char, "
                        "source_start_byte, source_end_byte, source_text_sha256, scoring_text_sha256, "
                        "chunker_identity_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (passage_pk, document_pk, passage.passage_id, *expected_geometry),
                    )
                else:
                    passage_pk = int(stored[0])
                    if tuple(stored[1:]) != expected_geometry:
                        raise TopicRecordsIntegrityError("passage geometry conflicts")
                passage_pks[passage.passage_id] = passage_pk

                query_passage = (
                    result.query.query_id,
                    passage_pk,
                    passage.raw_logit,
                    passage.rank,
                    passage.score_cache_key,
                )
                stored_query_passage = database.execute(
                    "SELECT query_id, passage_pk, raw_logit, passage_rank, score_cache_key "
                    "FROM query_passage WHERE query_id=? AND passage_pk=?",
                    (result.query.query_id, passage_pk),
                ).fetchone()
                if stored_query_passage is None:
                    database.execute(
                        "INSERT INTO query_passage VALUES (?, ?, ?, ?, ?)", query_passage
                    )
                elif tuple(stored_query_passage) != query_passage:
                    raise TopicRecordsIntegrityError("query passage identity conflicts")

            for document in result.documents:
                document_pk, _ = bindings[document.docid]
                retrieval_row = (
                    result.query.query_id,
                    document_pk,
                    document.source_rank,
                    document.source_score,
                    document.best_passage_id,
                    document.best_passage_raw_logit,
                )
                stored = database.execute(
                    "SELECT query_id, document_pk, source_rank, source_score, best_passage_id, "
                    "best_passage_raw_logit FROM retrieval_candidate "
                    "WHERE query_id=? AND source_rank=?",
                    (result.query.query_id, document.source_rank),
                ).fetchone()
                if stored is None:
                    database.execute(
                        "INSERT INTO retrieval_candidate VALUES (?, ?, ?, ?, ?, ?)",
                        retrieval_row,
                    )
                elif tuple(stored) != retrieval_row:
                    raise TopicRecordsIntegrityError("retrieval candidate identity conflicts")
            self._next_passage_pk = next_passage_pk
        except BaseException:
            database.rollback()
            raise
        else:
            database.commit()
            self._documents.update(bindings)
            self._query_ids.add(result.query.query_id)

    @_serialized_builder_operation
    def add_researcher_handoff(self, handoff: ResearcherHandoff) -> None:
        if not isinstance(handoff, ResearcherHandoff):
            raise TypeError("handoff must be a ResearcherHandoff")
        if handoff.run_id != self._run_id:
            raise TopicRecordsIntegrityError("researcher handoff run does not match builder run")
        handoff_payload = {
            "run_id": handoff.run_id,
            "researcher_id": handoff.researcher_id,
            "round_index": handoff.round_index,
            "evidence": [
                {
                    "passage_id": row.passage_id,
                    "relevance": row.relevance,
                    "subnarrative_id": row.subnarrative_id,
                }
                for row in handoff.evidence
            ],
            "facet_updates": [
                {"origin": row.origin, "subnarrative_id": row.subnarrative_id, "text": row.text}
                for row in handoff.facet_updates
            ],
        }
        handoff_sha256 = _digest(_canonical_json(handoff_payload))
        database = self._require_open()
        self._flush()
        database.execute("BEGIN IMMEDIATE")
        try:
            stored = database.execute(
                "SELECT handoff_sha256 FROM researcher_handoff WHERE researcher_id=?",
                (handoff.researcher_id,),
            ).fetchone()
            if stored is not None:
                if stored[0] != handoff_sha256:
                    raise TopicRecordsIntegrityError("researcher handoff conflict")
                database.rollback()
                return
            for facet in handoff.facet_updates:
                current = database.execute(
                    "SELECT subnarrative_sha256, subnarrative_text, origin "
                    "FROM subnarrative_identity WHERE subnarrative_id=?",
                    (facet.subnarrative_id,),
                ).fetchone()
                if current is None:
                    database.execute(
                        "INSERT INTO subnarrative_identity "
                        "(topic_id, subnarrative_id, subnarrative_sha256, subnarrative_text, origin) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (self._topic_id, facet.subnarrative_id, facet.text_sha256, facet.text, facet.origin),
                    )
                elif current[0] != facet.text_sha256 or (
                    current[1] and (current[1] != facet.text or current[2] != facet.origin)
                ):
                    raise TopicRecordsIntegrityError("facet identity conflicts")
                elif not current[1]:
                    database.execute(
                        "UPDATE subnarrative_identity SET subnarrative_sha256=?, "
                        "subnarrative_text=?, origin=? WHERE subnarrative_id=?",
                        (facet.text_sha256, facet.text, facet.origin, facet.subnarrative_id),
                    )
            database.execute(
                "INSERT INTO researcher_handoff VALUES (?, ?, ?, ?)",
                (handoff.researcher_id, handoff.run_id, handoff.round_index, handoff_sha256),
            )
            database.executemany(
                "INSERT INTO researcher_facet_update VALUES (?, ?, ?)",
                (
                    (handoff.researcher_id, ordinal, facet.subnarrative_id)
                    for ordinal, facet in enumerate(handoff.facet_updates)
                ),
            )
            for evidence in handoff.evidence:
                facet_exists = database.execute(
                    "SELECT 1 FROM subnarrative_identity WHERE subnarrative_id=?",
                    (evidence.subnarrative_id,),
                ).fetchone()
                if facet_exists is None:
                    raise TopicRecordsIntegrityError("researcher evidence facet is not admitted")
                passage_rows = database.execute(
                    "SELECT passage_pk FROM passage WHERE passage_id=?",
                    (evidence.passage_id,),
                ).fetchall()
                if len(passage_rows) != 1:
                    raise TopicRecordsIntegrityError("unknown passage in researcher handoff")
                database.execute(
                    "INSERT INTO researcher_evidence VALUES (?, ?, ?, ?)",
                    (handoff.researcher_id, evidence.subnarrative_id, passage_rows[0][0], evidence.relevance),
                )
        except BaseException:
            database.rollback()
            raise
        else:
            database.commit()
            for facet in handoff.facet_updates:
                self._subnarratives[facet.subnarrative_id] = facet.text_sha256
                self._facets[facet.subnarrative_id] = facet

    @_serialized_builder_operation
    def set_completion(self, status: str, stopping_reason: str) -> None:
        if status not in {"complete", "incomplete"}:
            raise ValueError("completion status must be complete or incomplete")
        allowed = (
            {"coverage_sufficient", "budget_exhausted", "agent_completed"}
            if status == "complete"
            else {
                "budget_exhausted",
                "hard_deadline",
                "retrieval_unavailable",
                "scoring_failed",
                "no_evidence",
                "zero_grounded_nuggets",
                "evidence_validation_failed",
            }
        )
        if stopping_reason not in allowed:
            raise ValueError("completion stopping_reason is invalid for status")
        database = self._require_open()
        self._flush()
        database.execute("BEGIN IMMEDIATE")
        try:
            row = (1, status, stopping_reason)
            stored = database.execute(
                "SELECT singleton, status, stopping_reason FROM topic_completion"
            ).fetchone()
            if stored is None:
                database.execute("INSERT INTO topic_completion VALUES (?, ?, ?)", row)
            elif tuple(stored) != row:
                raise TopicRecordsIntegrityError("topic completion conflicts")
        except BaseException:
            database.rollback()
            raise
        else:
            database.commit()

    @_serialized_builder_operation
    def topic_snapshot(self) -> TopicEvidenceSnapshot:
        database = self._require_open()
        self._flush()
        try:
            return _materialize_topic_snapshot(database, self._store, self._geometry_index)
        except TopicRecordsIntegrityError:
            raise
        except (DocumentStoreIntegrityError, TopicGeometryError, sqlite3.Error, ValueError, IndexError, KeyError) as exc:
            raise TopicRecordsIntegrityError("unable to materialize topic snapshot") from exc

    @_serialized_builder_operation
    def add_candidate(self, candidate: ExtractiveCandidate) -> None:
        if not isinstance(candidate, ExtractiveCandidate):
            raise TypeError("candidate must be an ExtractiveCandidate")
        if candidate.topic_id != self._topic_id:
            raise TopicRecordsIntegrityError("candidate topic_id does not match builder topic")
        self._require_open()
        try:
            binding = self._documents.get(candidate.docid)
            if binding is None or binding[1] != candidate.document_sha256:
                raise TopicRecordsIntegrityError("candidate document is not bound to this topic")
            document_pk, content_sha256 = binding
            geometry = self._activate_document(document_pk, content_sha256)
            _validate_candidate_with_geometry(candidate, geometry)
            _require_digest(candidate.subnarrative_sha256, "candidate subnarrative_sha256")

            known = self._subnarratives.get(candidate.subnarrative_id)
            if known is not None and known != candidate.subnarrative_sha256:
                raise TopicRecordsIntegrityError("subnarrative identity conflicts")
            if candidate.candidate_nugget_id in self._candidate_ids:
                raise TopicRecordsIntegrityError("candidate is already present in this topic")

            stage = _CandidateStage(
                candidate_ids={candidate.candidate_nugget_id},
                next_candidate_pk=self._next_candidate_pk,
                next_passage_pk=self._next_passage_pk,
            )
            if known is None:
                stage.subnarratives[candidate.subnarrative_id] = candidate.subnarrative_sha256
                stage.batch.subnarratives.append(
                    (
                        self._topic_id,
                        candidate.subnarrative_id,
                        candidate.subnarrative_sha256,
                        "",
                        "initial",
                    )
                )
            source_query_id = candidate.passages[0].query_id
            if any(passage.query_id != source_query_id for passage in candidate.passages):
                raise TopicRecordsIntegrityError("candidate passages use multiple query IDs")
            real_query = source_query_id in self._query_ids
            query_id = source_query_id if real_query else _legacy_candidate_query_id(
                source_query_id, candidate.docid
            )
            if not real_query and query_id not in self._query_ids:
                stage.batch.queries.append(
                    (
                        query_id,
                    _digest(source_query_id),
                        candidate.subnarrative_id,
                        "complete",
                        None,
                        1,
                        1,
                        1,
                        1,
                        len(candidate.passages),
                        0,
                    )
                )
                stage.batch.query_facets.append(
                    (query_id, candidate.subnarrative_id, "primary", 0)
                )
                best_passage = min(
                    candidate.passages,
                    key=lambda row: (-row.cross_encoder_score, row.passage_id),
                )
                stage.batch.retrieval_candidates.append(
                    (
                        query_id,
                        document_pk,
                        1,
                        0.0,
                        best_passage.passage_id,
                        best_passage.cross_encoder_score,
                    )
                )
            candidate_pk = stage.next_candidate_pk
            stage.batch.candidates.append((
                candidate_pk,
                self._topic_id,
                candidate.candidate_nugget_id,
                document_pk,
                candidate.subnarrative_id,
                candidate.candidate_kind,
                candidate.nugget_type,
                candidate.evidence_sentences[0].start_char,
                candidate.evidence_sentences[-1].end_char,
                _digest(candidate.text),
                candidate.sentence_cross_encoder_score,
                candidate.rank_within_document_subnarrative,
                candidate.scoring_text_sha256,
                candidate.subnarrative_sha256,
                candidate.sentence_splitter_version,
            ))
            for ordinal, span in enumerate(candidate.evidence_sentences):
                stage.batch.spans.append((
                    candidate_pk, "evidence", ordinal, span.start_char, span.end_char,
                    span.start_byte, span.end_byte, span.text_sha256, span.cross_encoder_score,
                ))
            for role, span in (
                ("matched_paragraph", candidate.matched_paragraph),
                ("context_before", candidate.context_before),
                ("context_after", candidate.context_after),
            ):
                if span is not None:
                    stage.batch.spans.append((
                        candidate_pk, role, 0, span.start_char, span.end_char,
                        span.start_byte, span.end_byte, span.text_sha256, None,
                    ))
            for ordinal, passage in enumerate(candidate.passages):
                passage_pk = self._passage_pk(document_pk, passage, stage)
                query_passage_key = (query_id, passage_pk)
                known_query_passage = self._query_passage_rows.get(query_passage_key)
                if known_query_passage is None:
                    known_query_passage = next(
                        (
                            tuple(row)
                            for row in reversed(self._batch.query_passages)
                            if (row[0], row[1]) == query_passage_key
                        ),
                        None,
                    )
                if known_query_passage is None:
                    stored_query_passage = self._require_open().execute(
                        "SELECT query_id, passage_pk, raw_logit, passage_rank, score_cache_key "
                        "FROM query_passage WHERE query_id=? AND passage_pk=?",
                        query_passage_key,
                    ).fetchone()
                    if stored_query_passage is not None:
                        known_query_passage = tuple(stored_query_passage)
                        self._query_passage_rows[query_passage_key] = known_query_passage
                if real_query:
                    if known_query_passage is None:
                        raise TopicRecordsIntegrityError(
                            "candidate passage is not admitted for the real query"
                        )
                    if (
                        known_query_passage[2] != passage.cross_encoder_score
                        or known_query_passage[3] != passage.cross_encoder_rank
                    ):
                        raise TopicRecordsIntegrityError(
                            "passage provenance/query passage score or rank conflicts"
                        )
                else:
                    query_passage = (
                        query_id,
                        passage_pk,
                        passage.cross_encoder_score,
                        passage.cross_encoder_rank,
                        _digest(f"{query_id}\0{passage.passage_id}"),
                    )
                    if known_query_passage is not None and known_query_passage != query_passage:
                        raise TopicRecordsIntegrityError(
                            "passage provenance/query passage conflicts"
                        )
                    if known_query_passage is None:
                        stage.batch.query_passages.append(query_passage)
                stage.batch.links.append((candidate_pk, document_pk, ordinal, passage_pk, query_id))

            stage.next_candidate_pk += 1
            try:
                self._batch.subnarratives.extend(stage.batch.subnarratives)
                self._batch.queries.extend(stage.batch.queries)
                self._batch.query_facets.extend(stage.batch.query_facets)
                self._batch.retrieval_candidates.extend(stage.batch.retrieval_candidates)
                self._batch.passages.extend(stage.batch.passages)
                self._batch.query_passages.extend(stage.batch.query_passages)
                self._batch.candidates.extend(stage.batch.candidates)
                self._batch.spans.extend(stage.batch.spans)
                self._batch.links.extend(stage.batch.links)
                self._passage_pk_by_id.update(stage.passage_pk_by_id)
                self._passage_rows.update(stage.passage_rows)
                self._candidate_ids.update(stage.candidate_ids)
                self._subnarratives.update(stage.subnarratives)
                self._query_ids.add(query_id)
                self._next_candidate_pk = stage.next_candidate_pk
                self._next_passage_pk = stage.next_passage_pk
                if len(self._batch.candidates) >= _BUILDER_BATCH_CANDIDATES:
                    self._flush()
            except BaseException:
                self._failed = True
                raise
        except TopicRecordsIntegrityError:
            raise
        except TopicGeometryError as exc:
            raise TopicRecordsIntegrityError("candidate document geometry is invalid") from exc
        except (DocumentStoreIntegrityError, sqlite3.Error, ValueError, IndexError) as exc:
            raise TopicRecordsIntegrityError("unable to add candidate") from exc

    def _passage_row(self, document_pk: int, passage_pk: int, passage: PassageProvenance):
        """Store the exact scored source slice; normalized hashes are derivable."""
        return (
            passage_pk,
            document_pk,
            passage.passage_id,
            passage.source_start_char,
            passage.source_end_char,
            passage.source_start_byte,
            passage.source_end_byte,
            passage.source_text_sha256,
            passage.source_text_sha256,
            _canonical_json({"backend": "legacy-candidate", "implementation": "v3"}),
        )

    @staticmethod
    def _candidate_passage_geometry_matches(
        stored: Sequence[object],
        expected: Sequence[object],
    ) -> bool:
        """Compare candidate geometry while preserving an admitted chunker identity."""
        return tuple(stored[:-1]) == tuple(expected[:-1])

    def _passage_pk(
        self,
        document_pk: int,
        passage: PassageProvenance,
        stage: _CandidateStage,
    ) -> int:
        """Deduplicate one complete passage tuple per ``(document, passage_id)``.

        The passage dictionary carries only source geometry in v4. Query score,
        rank, and cache identity are normalized in ``query_passage``.
        """
        known = stage.passage_pk_by_id.get(passage.passage_id)
        if known is None:
            known = self._passage_pk_by_id.get(passage.passage_id)
        if known is not None:
            known_row = stage.passage_rows.get(known)
            if known_row is None:
                known_row = self._passage_rows[known]
            if not self._candidate_passage_geometry_matches(
                known_row,
                self._passage_row(document_pk, known, passage),
            ):
                raise TopicRecordsIntegrityError(
                    "passage provenance conflicts within one document"
                )
            return known
        stored = self._require_open().execute(
            "SELECT * FROM passage WHERE document_pk=? AND passage_id=?",
            (document_pk, passage.passage_id),
        ).fetchone()
        if stored is not None:
            passage_pk = int(stored[0])
            row = self._passage_row(document_pk, passage_pk, passage)
            if not self._candidate_passage_geometry_matches(stored, row):
                raise TopicRecordsIntegrityError(
                    "passage provenance conflicts within one document"
                )
            stage.next_passage_pk = max(stage.next_passage_pk, passage_pk + 1)
        else:
            passage_pk = stage.next_passage_pk
            stage.next_passage_pk += 1
            row = self._passage_row(document_pk, passage_pk, passage)
            stage.batch.passages.append(row)
        stage.passage_pk_by_id[passage.passage_id] = passage_pk
        stage.passage_rows[passage_pk] = row
        return passage_pk

    def _seal(self, identity_json: str) -> tuple[str, dict[str, int]]:
        database = self._require_open()
        database.execute("BEGIN IMMEDIATE")
        try:
            semantic = _semantic_sha256(database)
            counts = _row_counts(database)
            counts["stage_seal"] = 1
            database.execute(
                "INSERT INTO stage_seal VALUES (?, ?, ?, ?, ?)",
                (CANDIDATE_STAGE, TOPIC_RECORDS_SCHEMA_VERSION, semantic, identity_json, _canonical_json(counts)),
            )
        except BaseException:
            database.rollback()
            raise
        else:
            database.commit()
        return semantic, counts

    @_serialized_builder_operation
    def publish(self, identity: Mapping[str, object]) -> PublishedTopicRecords:
        if not isinstance(identity, Mapping):
            raise TypeError("identity must be a mapping")
        identity_json = _canonical_json(identity)
        if self._published is not None:
            if identity_json != self._published_identity_json:
                raise TopicRecordsIntegrityError(
                    "published topic records identity conflicts with requested identity"
                )
            return self._published
        try:
            self._flush()
            semantic, counts = self._seal(identity_json)
            database = self._require_open()
            if database.execute("PRAGMA foreign_key_check").fetchall():
                raise TopicRecordsIntegrityError("foreign-key validation failed")
            if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise TopicRecordsIntegrityError("SQLite integrity validation failed")
            self._validate_builder_sources()
            document_hashes = sorted(self._document_hashes())
            database.execute("PRAGMA journal_mode=DELETE")
            database.commit()
            database.close()
            self._database = None
            if Path(str(self._temporary_path) + "-wal").exists() or Path(str(self._temporary_path) + "-shm").exists():
                raise TopicRecordsIntegrityError("topic database has WAL/SHM companions")
            database_sha256, database_bytes = _stream_receipt(self._temporary_path)
            manifest = {
                "schema_version": TOPIC_RECORDS_SCHEMA_VERSION,
                "stage": CANDIDATE_STAGE,
                "topic_id": self._topic_id,
                "run_id": self._run_id,
                "records_file": self._database_path.name,
                "manifest_file": self._manifest_path.name,
                "database_sha256": database_sha256,
                "database_bytes": database_bytes,
                "semantic_sha256": semantic,
                "identity_json": identity_json,
                "document_sha256s": document_hashes,
                "request_key_closure": [],
                "row_counts": counts,
            }
            manifest_body = (_canonical_json(manifest) + "\n").encode("utf-8")
            self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._destination / ".records-publication.lock"
            with _publication_lock(lock_path):
                db_exists = self._database_path.exists()
                manifest_exists = self._manifest_path.exists()
                if db_exists != manifest_exists:
                    raise TopicRecordsIntegrityError("partial topic records publication exists")
                if db_exists:
                    return self._record_publication(
                        self._existing_publication(semantic, identity_json), identity_json
                    )
                _publication_boundary("before_database")
                try:
                    _publish_without_replacement(self._temporary_path, self._database_path)
                except FileExistsError:
                    return self._record_publication(
                        self._existing_publication(semantic, identity_json), identity_json
                    )
                _publication_boundary("after_database")
                _publication_boundary("before_manifest")
                try:
                    _atomic_manifest_create(self._manifest_path, manifest_body)
                except FileExistsError:
                    return self._record_publication(
                        self._existing_publication(semantic, identity_json), identity_json
                    )
                _publication_boundary("after_manifest")
            records = TopicRecords.open(
                self._database_path,
                self._manifest_path,
                self._topic_id,
                self._store,
            )
            try:
                published = PublishedTopicRecords(records.receipt, records.validation_session)
            finally:
                records.close()
            return self._record_publication(published, identity_json)
        except TopicRecordsIntegrityError:
            self._cleanup()
            raise
        except (sqlite3.Error, OSError, DocumentStoreIntegrityError, ValueError, IndexError) as exc:
            self._cleanup()
            raise TopicRecordsIntegrityError("unable to publish topic records") from exc

    def _record_publication(
        self,
        published: PublishedTopicRecords,
        identity_json: str,
    ) -> PublishedTopicRecords:
        self._cleanup()
        self._published = published
        self._published_identity_json = identity_json
        return published

    def _document_hashes(self) -> tuple[str, ...]:
        database = self._require_open()
        return tuple(sorted({
            row[0]
            for row in database.execute(
                "SELECT content_sha256 FROM document_binding WHERE topic_id=? ORDER BY docid",
                (self._topic_id,),
            )
        }))

    def _existing_publication(
        self,
        semantic: str,
        identity_json: str,
    ) -> PublishedTopicRecords:
        if not self._database_path.is_file() or not self._manifest_path.is_file():
            raise TopicRecordsIntegrityError("existing topic records publication is partial")
        # A race loser fully opens and validates the winner's actual bytes; it
        # never reuses its own proof for someone else's database.
        records = TopicRecords.open(
            self._database_path,
            self._manifest_path,
            self._topic_id,
            self._store,
        )
        try:
            if records.receipt.semantic_sha256 != semantic:
                raise TopicRecordsIntegrityError(
                    "contradictory topic records publication already exists"
                )
            if records._manifest["identity_json"] != identity_json:
                raise TopicRecordsIntegrityError(
                    "topic records publication identity conflicts with requested identity"
                )
            return PublishedTopicRecords(records.receipt, records.validation_session)
        finally:
            records.close()

    def _validate_builder_sources(self) -> None:
        # A rolling per-document index keeps builder memory bounded while still
        # deriving geometry exactly once per bound document.
        _validate_sources(self._require_open(), self._store, self._topic_id)

    def __del__(self):
        try:
            self._cleanup()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Read-only seam
# --------------------------------------------------------------------------- #


class TopicRecords:
    """Read-only public seam for a sealed topic database."""

    def __enter__(self) -> "TopicRecords":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def validate_all_sources(self) -> None:
        try:
            _validate_sources(
                self._database, self._store, self.topic_id, geometry_index=self._geometry
            )
        except TopicRecordsIntegrityError:
            raise
        except TopicGeometryError as exc:
            raise TopicRecordsIntegrityError("topic source geometry is invalid") from exc
        except (DocumentStoreIntegrityError, sqlite3.Error, ValueError, IndexError, KeyError) as exc:
            raise TopicRecordsIntegrityError("topic source validation failed") from exc

    def load_candidates(
        self,
        required_keys: Iterable[tuple[str, str]] | None = None,
    ) -> Mapping[tuple[str, str], ExtractiveCandidate]:
        keys = None if required_keys is None else tuple(required_keys)
        if keys is not None and any(
            not isinstance(key, tuple) or len(key) != 2 or any(not isinstance(part, str) or not part for part in key)
            for key in keys
        ):
            raise TypeError("required_keys must contain (subnarrative_id, candidate_nugget_id) pairs")
        if keys is None:
            rows = self._database.execute(
                "SELECT subnarrative_id, candidate_nugget_id FROM candidate "
                "WHERE topic_id=? ORDER BY subnarrative_id, candidate_nugget_id",
                (self.topic_id,),
            ).fetchall()
        else:
            rows = []
            for subnarrative_id, candidate_id in sorted(set(keys)):
                row = self._database.execute(
                    "SELECT subnarrative_id, candidate_nugget_id FROM candidate "
                    "WHERE topic_id=? AND subnarrative_id=? AND candidate_nugget_id=?",
                    (self.topic_id, subnarrative_id, candidate_id),
                ).fetchone()
                if row is None:
                    raise TopicRecordsIntegrityError("requested candidate is absent")
                rows.append(row)
        result: dict[tuple[str, str], ExtractiveCandidate] = {}
        for subnarrative_id, candidate_id in rows:
            candidate, geometry = self._reconstruct(subnarrative_id, candidate_id)
            _validate_candidate_with_geometry(candidate, geometry)
            result[(subnarrative_id, candidate_id)] = candidate
        return result

    def selection_pool(self, context: SubnarrativeContext, limit: int) -> PreclusterPool:
        if not isinstance(context, SubnarrativeContext):
            raise TypeError("context must be a SubnarrativeContext")
        if context.topic_id != self.topic_id:
            raise TopicRecordsIntegrityError("selection context topic does not match records topic")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("selection pool limit must be positive")
        identity = self._database.execute(
            "SELECT subnarrative_sha256 FROM subnarrative_identity WHERE topic_id=? AND subnarrative_id=?",
            (self.topic_id, context.subnarrative_id),
        ).fetchone()
        if identity is None or identity[0] != context.subnarrative_sha256:
            raise TopicRecordsIntegrityError("selection subnarrative identity does not match")
        metadata = self._database.execute(
            "SELECT candidate_kind, text_sha256, candidate_nugget_id, sentence_score "
            "FROM candidate WHERE topic_id=? AND subnarrative_id=?",
            (self.topic_id, context.subnarrative_id),
        ).fetchall()
        groups: dict[tuple[str, str], list[tuple[str, float]]] = {}
        for kind, text_sha256, candidate_id, score in metadata:
            groups.setdefault((kind, text_sha256), []).append((candidate_id, float(score)))
        ranked_groups = sorted(
            groups.items(),
            key=lambda item: (
                -max(score for _, score in item[1]),
                min(candidate_id for candidate_id, score in item[1] if score == max(score for _, score in item[1])),
            ),
        )
        chosen = ranked_groups[:limit]
        selected_rows: list[tuple[str, str, str, str, float]] = []
        for kind, text_sha256 in ((kind, digest) for (kind, digest), _ in chosen):
            members = sorted(groups[(kind, text_sha256)], key=lambda row: (-row[1], row[0]))
            selected_rows.extend((kind, text_sha256, candidate_id, self.topic_id, score) for candidate_id, score in members)
        candidates: list[SelectionCandidate] = []
        for kind, text_sha256, candidate_id, topic_id, score in selected_rows:
            row = self._database.execute(
                "SELECT subnarrative_id, candidate_nugget_id FROM candidate "
                "WHERE topic_id=? AND subnarrative_id=? AND candidate_nugget_id=?",
                (topic_id, context.subnarrative_id, candidate_id),
            ).fetchone()
            if row is None:
                raise TopicRecordsIntegrityError("selection candidate disappeared")
            candidate, geometry = self._reconstruct(row[0], row[1])
            _validate_candidate_with_geometry(candidate, geometry)
            if candidate.candidate_kind != kind or _digest(candidate.text) != text_sha256 or candidate.sentence_cross_encoder_score != score:
                raise TopicRecordsIntegrityError("selection exact-group seal is inconsistent")
            candidates.append(SelectionCandidate(
                topic_id=candidate.topic_id,
                subnarrative_id=candidate.subnarrative_id,
                candidate_nugget_id=candidate.candidate_nugget_id,
                candidate_kind=candidate.candidate_kind,
                text=candidate.text,
                docid=candidate.docid,
                document_sha256=candidate.document_sha256,
                raw_logit=candidate.sentence_cross_encoder_score,
            ))
        return PreclusterPool(tuple(candidates), len(metadata), len(groups))

    def _reconstruct(
        self,
        subnarrative_id: str,
        candidate_id: str,
    ) -> tuple[ExtractiveCandidate, DocumentGeometry]:
        try:
            return _reconstruct_candidate(
                self._database,
                self._store,
                self.topic_id,
                subnarrative_id,
                candidate_id,
                geometry_index=self._geometry,
            )
        except TopicRecordsIntegrityError:
            raise
        except TopicGeometryError as exc:
            raise TopicRecordsIntegrityError("topic source geometry is invalid") from exc
        except (DocumentStoreIntegrityError, sqlite3.Error, ValueError, IndexError, KeyError) as exc:
            raise TopicRecordsIntegrityError("topic source validation failed") from exc


def _install_topic_records_capability():
    class _ValidationProof:
        __slots__ = (
            "process_id",
            "validator_version",
            "schema_version",
            "topic_id",
            "receipt",
            "manifest",
            "document_receipts",
            "geometry",
            "master",
            "lock",
            "closed",
            "__weakref__",
        )

        def __init__(
            self,
            *,
            receipt: TopicRecordsReceipt,
            manifest: Mapping[str, object],
            document_receipts: tuple[tuple[str, int, int], ...],
            geometry: DocumentGeometryIndex,
            master: sqlite3.Connection,
        ) -> None:
            self.process_id = os.getpid()
            self.validator_version = TOPIC_RECORDS_VALIDATOR_VERSION
            self.schema_version = TOPIC_RECORDS_SCHEMA_VERSION
            self.topic_id = receipt.topic_id
            self.receipt = receipt
            self.manifest = MappingProxyType(dict(manifest))
            self.document_receipts = document_receipts
            self.geometry = geometry
            self.master = master
            self.lock = RLock()
            self.closed = False

        def close(self) -> None:
            with self.lock:
                if self.closed:
                    return
                self.closed = True
                master = self.master
                self.master = None
            if master is not None:
                try:
                    master.close()
                except sqlite3.Error:
                    pass

        def __del__(self) -> None:
            try:
                self.close()
            except Exception:
                pass

    @dataclass
    class _Binding:
        process_id: int
        validator_version: str
        token: object
        path: Path
        manifest_path: Path
        store: DocumentStore
        lock: RLock = field(default_factory=RLock)

    class _ValidatedTopicRecords:
        __slots__ = ("__weakref__",)

        def __new__(cls, *args: object, **kwargs: object):
            raise TypeError(
                "ValidatedTopicRecords is an opaque process-local capability; "
                "only the validator may mint it"
            )

        def __reduce__(self) -> Any:
            raise TypeError(
                "ValidatedTopicRecords is a process-local capability and cannot be "
                "serialized; revalidate in the target process instead"
            )

        def __reduce_ex__(self, protocol: int) -> Any:
            return self.__reduce__()

        def __getstate__(self) -> Any:
            return self.__reduce__()

        def __copy__(self) -> Any:
            raise TypeError(
                "ValidatedTopicRecords is a process-local capability and cannot be copied"
            )

        def __deepcopy__(self, memo: dict[int, object]) -> Any:
            raise TypeError(
                "ValidatedTopicRecords is a process-local capability and cannot be copied"
            )

    _ValidatedTopicRecords.__name__ = "ValidatedTopicRecords"
    _ValidatedTopicRecords.__qualname__ = "ValidatedTopicRecords"

    token_registry: weakref.WeakKeyDictionary[object, _ValidationProof] = (
        weakref.WeakKeyDictionary()
    )
    handle_registry: weakref.WeakKeyDictionary[object, _Binding] = (
        weakref.WeakKeyDictionary()
    )
    registry_lock = RLock()
    registry_pid = os.getpid()

    def refresh_after_fork() -> None:
        nonlocal token_registry, handle_registry, registry_lock, registry_pid
        current_pid = os.getpid()
        if current_pid != registry_pid:
            token_registry = weakref.WeakKeyDictionary()
            handle_registry = weakref.WeakKeyDictionary()
            registry_lock = RLock()
            registry_pid = current_pid

    def resolve(token: object) -> _ValidationProof:
        if type(token) is not _ValidatedTopicRecords:
            raise TopicRecordsIntegrityError(
                "validation session is not an exact issued capability"
            )
        refresh_after_fork()
        with registry_lock:
            proof = token_registry.get(token)
        if proof is None:
            raise TopicRecordsIntegrityError(
                "validation session was not issued by this process"
            )
        return proof

    def mint(
        receipt: TopicRecordsReceipt,
        manifest: Mapping[str, object],
        document_receipts: tuple[tuple[str, int, int], ...],
        geometry: DocumentGeometryIndex,
        master: sqlite3.Connection,
    ) -> _ValidatedTopicRecords:
        refresh_after_fork()
        with registry_lock:
            token = object.__new__(_ValidatedTopicRecords)
            token_registry[token] = _ValidationProof(
                receipt=receipt,
                manifest=manifest,
                document_receipts=document_receipts,
                geometry=geometry,
                master=master,
            )
            return token

    def verify_document_closure(
        document_store: DocumentStore,
        document_receipts: tuple[tuple[str, int, int], ...],
    ) -> None:
        try:
            for content_sha256, character_count, byte_count in document_receipts:
                receipt = document_store.verify(content_sha256)
                if (
                    receipt.character_count != character_count
                    or receipt.byte_count != byte_count
                ):
                    raise TopicRecordsIntegrityError(
                        "validation session CAS closure does not match the document store"
                    )
        except DocumentStoreIntegrityError as exc:
            raise TopicRecordsIntegrityError(
                "validation session CAS closure does not match the document store"
            ) from exc

    def construct(
        *,
        path: Path,
        manifest_path: Path,
        document_store: DocumentStore,
        token: _ValidatedTopicRecords,
    ) -> TopicRecords:
        records = object.__new__(TopicRecords)
        binding = _Binding(
            process_id=os.getpid(),
            validator_version=TOPIC_RECORDS_VALIDATOR_VERSION,
            token=token,
            path=path,
            manifest_path=manifest_path,
            store=document_store,
        )
        refresh_after_fork()
        with registry_lock:
            handle_registry[records] = binding
        return records

    def receipt_from_manifest(
        value: Mapping[str, object],
        database_receipt: tuple[str, int],
        manifest_receipt: tuple[str, int],
    ) -> TopicRecordsReceipt:
        return TopicRecordsReceipt(
            database_sha256=database_receipt[0],
            database_bytes=database_receipt[1],
            semantic_sha256=str(value["semantic_sha256"]),
            topic_id=str(value["topic_id"]),
            run_id=str(value["run_id"]),
            document_sha256s=tuple(value["document_sha256s"]),
            row_counts=MappingProxyType(dict(value["row_counts"])),
            schema_version=str(value["schema_version"]),
            manifest_sha256=manifest_receipt[0],
            manifest_bytes=manifest_receipt[1],
        )

    @contextmanager
    def authorized(records: object) -> Iterator[tuple[_Binding, _ValidationProof]]:
        if type(records) is not TopicRecords:
            raise TopicRecordsIntegrityError("topic records handle is not exact")
        refresh_after_fork()
        with registry_lock:
            binding = handle_registry.get(records)
        if binding is None:
            raise TopicRecordsIntegrityError(
                "topic records handle is not registered or is revoked"
            )
        with binding.lock:
            refresh_after_fork()
            with registry_lock:
                if handle_registry.get(records) is not binding:
                    raise TopicRecordsIntegrityError(
                        "topic records handle is not registered or is revoked"
                    )
            if binding.process_id != os.getpid():
                raise TopicRecordsIntegrityError(
                    "topic records handle belongs to a different process"
                )
            proof = resolve(binding.token)
            if (
                proof.process_id != os.getpid()
                or binding.validator_version != TOPIC_RECORDS_VALIDATOR_VERSION
                or proof.validator_version != TOPIC_RECORDS_VALIDATOR_VERSION
                or proof.schema_version != TOPIC_RECORDS_SCHEMA_VERSION
                or proof.master is None
                or proof.closed
            ):
                raise TopicRecordsIntegrityError(
                    "topic records handle validation session is stale"
                )
            with proof.lock:
                if proof.master is None or proof.closed:
                    raise TopicRecordsIntegrityError("topic records handle is closed")
                yield binding, proof

    def assert_current(cls, records: object) -> TopicRecordsReceipt:
        if cls is not TopicRecords:
            raise TypeError("TopicRecords.assert_current requires the exact class")
        with authorized(records) as (_binding, proof):
            return proof.receipt

    def close_records(records: TopicRecords) -> None:
        if type(records) is not TopicRecords:
            return
        refresh_after_fork()
        with registry_lock:
            binding = handle_registry.get(records)
        if binding is None:
            return
        with binding.lock:
            refresh_after_fork()
            with registry_lock:
                if handle_registry.get(records) is binding:
                    handle_registry.pop(records, None)

    def receipt_property(records: object) -> TopicRecordsReceipt:
        with authorized(records) as (_binding, proof):
            return proof.receipt

    def topic_id_property(records: object) -> str:
        with authorized(records) as (_binding, proof):
            return proof.topic_id

    def run_id_property(records: object) -> str:
        with authorized(records) as (_binding, proof):
            return proof.receipt.run_id

    def validation_session_property(records: object) -> _ValidatedTopicRecords:
        with authorized(records) as (binding, _proof):
            return binding.token

    def manifest_property(records: object) -> Mapping[str, object]:
        with authorized(records) as (_binding, proof):
            return proof.manifest

    def enter_records(records: TopicRecords) -> TopicRecords:
        with authorized(records):
            return records

    def validate_all_sources(records: TopicRecords) -> None:
        try:
            with authorized(records) as (binding, proof):
                _validate_sources(
                    proof.master,
                    binding.store,
                    proof.topic_id,
                    geometry_index=proof.geometry,
                )
        except TopicRecordsIntegrityError:
            raise
        except TopicGeometryError as exc:
            raise TopicRecordsIntegrityError("topic source geometry is invalid") from exc
        except (DocumentStoreIntegrityError, sqlite3.Error, ValueError, IndexError, KeyError) as exc:
            raise TopicRecordsIntegrityError("topic source validation failed") from exc

    def load_candidates(
        records: TopicRecords,
        required_keys: Iterable[tuple[str, str]] | None = None,
    ) -> Mapping[tuple[str, str], ExtractiveCandidate]:
        try:
            with authorized(records) as (binding, proof):
                keys = None if required_keys is None else tuple(required_keys)
                if keys is not None and any(
                    not isinstance(key, tuple)
                    or len(key) != 2
                    or any(not isinstance(part, str) or not part for part in key)
                    for key in keys
                ):
                    raise TypeError(
                        "required_keys must contain (subnarrative_id, candidate_nugget_id) pairs"
                    )
                if keys is None:
                    rows = proof.master.execute(
                        "SELECT subnarrative_id, candidate_nugget_id FROM candidate "
                        "WHERE topic_id=? ORDER BY subnarrative_id, candidate_nugget_id",
                        (proof.topic_id,),
                    ).fetchall()
                else:
                    rows = []
                    for subnarrative_id, candidate_id in sorted(set(keys)):
                        row = proof.master.execute(
                            "SELECT subnarrative_id, candidate_nugget_id FROM candidate "
                            "WHERE topic_id=? AND subnarrative_id=? AND candidate_nugget_id=?",
                            (proof.topic_id, subnarrative_id, candidate_id),
                        ).fetchone()
                        if row is None:
                            raise TopicRecordsIntegrityError("requested candidate is absent")
                        rows.append(row)
                result: dict[tuple[str, str], ExtractiveCandidate] = {}
                for subnarrative_id, candidate_id in rows:
                    candidate, geometry = _reconstruct_candidate(
                        proof.master,
                        binding.store,
                        proof.topic_id,
                        subnarrative_id,
                        candidate_id,
                        geometry_index=proof.geometry,
                    )
                    _validate_candidate_with_geometry(candidate, geometry)
                    result[(subnarrative_id, candidate_id)] = candidate
                return result
        except TopicRecordsIntegrityError:
            raise
        except TopicGeometryError as exc:
            raise TopicRecordsIntegrityError("topic source geometry is invalid") from exc
        except (DocumentStoreIntegrityError, sqlite3.Error, ValueError, IndexError, KeyError) as exc:
            raise TopicRecordsIntegrityError("topic source validation failed") from exc

    def selection_pool(
        records: TopicRecords,
        context: SubnarrativeContext,
        limit: int,
    ) -> PreclusterPool:
        try:
            with authorized(records) as (binding, proof):
                if not isinstance(context, SubnarrativeContext):
                    raise TypeError("context must be a SubnarrativeContext")
                if context.topic_id != proof.topic_id:
                    raise TopicRecordsIntegrityError(
                        "selection context topic does not match records topic"
                    )
                if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                    raise ValueError("selection pool limit must be positive")
                identity = proof.master.execute(
                    "SELECT subnarrative_sha256 FROM subnarrative_identity "
                    "WHERE topic_id=? AND subnarrative_id=?",
                    (proof.topic_id, context.subnarrative_id),
                ).fetchone()
                if identity is None or identity[0] != context.subnarrative_sha256:
                    raise TopicRecordsIntegrityError(
                        "selection subnarrative identity does not match"
                    )
                metadata = proof.master.execute(
                    "SELECT candidate_kind, text_sha256, candidate_nugget_id, sentence_score "
                    "FROM candidate WHERE topic_id=? AND subnarrative_id=?",
                    (proof.topic_id, context.subnarrative_id),
                ).fetchall()
                groups: dict[tuple[str, str], list[tuple[str, float]]] = {}
                for kind, text_sha256, candidate_id, score in metadata:
                    groups.setdefault((kind, text_sha256), []).append(
                        (candidate_id, float(score))
                    )
                ranked_groups = sorted(
                    groups.items(),
                    key=lambda item: (
                        -max(score for _, score in item[1]),
                        min(
                            candidate_id
                            for candidate_id, score in item[1]
                            if score == max(score for _, score in item[1])
                        ),
                    ),
                )
                chosen = ranked_groups[:limit]
                selected_rows: list[tuple[str, str, str, str, float]] = []
                for kind, text_sha256 in (
                    (kind, digest) for (kind, digest), _ in chosen
                ):
                    members = sorted(
                        groups[(kind, text_sha256)], key=lambda row: (-row[1], row[0])
                    )
                    selected_rows.extend(
                        (kind, text_sha256, candidate_id, proof.topic_id, score)
                        for candidate_id, score in members
                    )
                candidates: list[SelectionCandidate] = []
                for kind, text_sha256, candidate_id, topic_id, score in selected_rows:
                    row = proof.master.execute(
                        "SELECT subnarrative_id, candidate_nugget_id FROM candidate "
                        "WHERE topic_id=? AND subnarrative_id=? AND candidate_nugget_id=?",
                        (topic_id, context.subnarrative_id, candidate_id),
                    ).fetchone()
                    if row is None:
                        raise TopicRecordsIntegrityError("selection candidate disappeared")
                    candidate, geometry = _reconstruct_candidate(
                        proof.master,
                        binding.store,
                        topic_id,
                        row[0],
                        row[1],
                        geometry_index=proof.geometry,
                    )
                    _validate_candidate_with_geometry(candidate, geometry)
                    if (
                        candidate.candidate_kind != kind
                        or _digest(candidate.text) != text_sha256
                        or candidate.sentence_cross_encoder_score != score
                    ):
                        raise TopicRecordsIntegrityError(
                            "selection exact-group seal is inconsistent"
                        )
                    candidates.append(
                        SelectionCandidate(
                            topic_id=candidate.topic_id,
                            subnarrative_id=candidate.subnarrative_id,
                            candidate_nugget_id=candidate.candidate_nugget_id,
                            candidate_kind=candidate.candidate_kind,
                            text=candidate.text,
                            docid=candidate.docid,
                            document_sha256=candidate.document_sha256,
                            raw_logit=candidate.sentence_cross_encoder_score,
                        )
                    )
                return PreclusterPool(tuple(candidates), len(metadata), len(groups))
        except TopicRecordsIntegrityError:
            raise
        except TopicGeometryError as exc:
            raise TopicRecordsIntegrityError("topic source geometry is invalid") from exc
        except (DocumentStoreIntegrityError, sqlite3.Error, ValueError, IndexError, KeyError) as exc:
            raise TopicRecordsIntegrityError("topic source validation failed") from exc

    def passage_search(self: TopicRecords, query_id: str) -> PassageSearchSnapshot:
        try:
            with authorized(self) as (binding, proof):
                _require_text(query_id, "query_id")
                proof.master.execute("BEGIN")
                try:
                    snapshot = _stored_passage_search(
                        proof.master, binding.store, proof.geometry, query_id
                    )
                finally:
                    proof.master.rollback()
                return snapshot
        except TopicRecordsIntegrityError:
            raise
        except (DocumentStoreIntegrityError, TopicGeometryError, sqlite3.Error, ValueError, IndexError, KeyError) as exc:
            raise TopicRecordsIntegrityError("unable to materialize passage search") from exc

    def topic_snapshot(self: TopicRecords) -> TopicEvidenceSnapshot:
        try:
            with authorized(self) as (binding, proof):
                return _materialize_topic_snapshot(
                    proof.master, binding.store, proof.geometry
                )
        except TopicRecordsIntegrityError:
            raise
        except (DocumentStoreIntegrityError, TopicGeometryError, sqlite3.Error, ValueError, IndexError, KeyError) as exc:
            raise TopicRecordsIntegrityError("unable to materialize topic snapshot") from exc

    def open_records(
        cls,
        database: Path,
        manifest: Path,
        expected_topic_id: str,
        document_store: DocumentStore,
        validation_session: _ValidatedTopicRecords | None = None,
    ) -> TopicRecords:
        if cls is not TopicRecords:
            raise TypeError("TopicRecords.open requires the exact TopicRecords class")
        database = Path(database)
        manifest = Path(manifest)
        _require_text(expected_topic_id, "expected_topic_id")
        if not isinstance(document_store, DocumentStore):
            raise TypeError("document_store must be a DocumentStore")

        proof: _ValidationProof | None = None
        if validation_session is not None:
            proof = resolve(validation_session)
            if proof.process_id != os.getpid():
                raise TopicRecordsIntegrityError(
                    "validation session belongs to a different process"
                )
            if (
                proof.validator_version != TOPIC_RECORDS_VALIDATOR_VERSION
                or proof.schema_version != TOPIC_RECORDS_SCHEMA_VERSION
            ):
                raise TopicRecordsIntegrityError("validation session version does not match")
            if proof.topic_id != expected_topic_id:
                raise TopicRecordsIntegrityError("validation session topic does not match")

        if not database.is_file() or not manifest.is_file():
            raise TopicRecordsIntegrityError("topic database and manifest must both exist")
        if Path(str(database) + "-wal").exists() or Path(str(database) + "-shm").exists():
            raise TopicRecordsIntegrityError("sealed topic database has WAL/SHM companions")

        database_descriptor: int | None = None
        manifest_descriptor: int | None = None
        audit_descriptor: int | None = None
        connection: sqlite3.Connection | None = None
        snapshot_path: Path | None = None
        snapshot_directory: Path | None = None
        minted_token: _ValidatedTopicRecords | None = None
        success = False
        try:
            manifest_descriptor = _open_regular_readonly(manifest, "records manifest")
            manifest_body = _read_descriptor(manifest_descriptor, "records manifest")
            manifest_receipt = (_digest(manifest_body), len(manifest_body))
            try:
                value = json.loads(manifest_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TopicRecordsIntegrityError(
                    "records manifest is not valid UTF-8 JSON"
                ) from exc
            if not isinstance(value, dict):
                raise TopicRecordsIntegrityError("records manifest must be an object")

            if proof is not None:
                database_descriptor = _open_regular_readonly(database, "topic database")
                database_receipt = _descriptor_receipt(database_descriptor, "topic database")
                if database_receipt != (
                    proof.receipt.database_sha256,
                    proof.receipt.database_bytes,
                ):
                    raise TopicRecordsIntegrityError(
                        "validation session database bytes do not match the sealed database"
                    )
                if manifest_receipt != (
                    proof.receipt.manifest_sha256,
                    proof.receipt.manifest_bytes,
                ):
                    raise TopicRecordsIntegrityError(
                        "validation session manifest bytes do not match the sealed manifest"
                    )
                if tuple(value["document_sha256s"]) != proof.receipt.document_sha256s:
                    raise TopicRecordsIntegrityError(
                        "validation session manifest identity does not match"
                    )
                with proof.lock:
                    verify_document_closure(document_store, proof.document_receipts)
                    records = construct(
                        path=database,
                        manifest_path=manifest,
                        document_store=document_store,
                        token=validation_session,
                    )
                success = True
                return records

            database_descriptor = _open_regular_readonly(database, "topic database")
            snapshot_directory = Path(tempfile.mkdtemp(prefix="topic-records-"))
            os.chmod(snapshot_directory, 0o700)
            snapshot_path = snapshot_directory / database.name
            _copy_descriptor(database_descriptor, snapshot_path)
            os.close(database_descriptor)
            database_descriptor = None
            audit_descriptor = _open_regular_readonly(
                snapshot_path, "topic database snapshot"
            )
            connection = _connect_readonly_descriptor(audit_descriptor)
            snapshot_path.unlink()
            snapshot_path = None
            snapshot_directory.rmdir()
            snapshot_directory = None
            database_receipt = _descriptor_receipt(audit_descriptor, "topic database")
            os.close(audit_descriptor)
            audit_descriptor = None
            _validate_manifest(
                value, database, manifest, expected_topic_id, database_receipt
            )
            _validate_database(connection, value)
            actual_semantic = _semantic_sha256(connection)
            if actual_semantic != value["semantic_sha256"]:
                raise TopicRecordsIntegrityError("semantic seal does not match database")
            stage = connection.execute(
                "SELECT * FROM stage_seal WHERE stage=?", (CANDIDATE_STAGE,)
            ).fetchone()
            if (
                stage is None
                or stage[2] != actual_semantic
                or stage[1] != TOPIC_RECORDS_SCHEMA_VERSION
                or stage[3] != value["identity_json"]
                or json.loads(stage[4]) != value["row_counts"]
            ):
                raise TopicRecordsIntegrityError("stage seal is inconsistent")
            geometry = DocumentGeometryIndex()
            document_receipts = _validate_sources(
                connection,
                document_store,
                str(value["topic_id"]),
                geometry_index=geometry,
            )
            receipt = receipt_from_manifest(value, database_receipt, manifest_receipt)
            minted_token = mint(
                receipt,
                value,
                document_receipts,
                geometry,
                connection,
            )
            resolve(minted_token)
            connection = None
            records = construct(
                path=database,
                manifest_path=manifest,
                document_store=document_store,
                token=minted_token,
            )
            success = True
            return records
        except TopicRecordsIntegrityError:
            raise
        except TopicGeometryError as exc:
            raise TopicRecordsIntegrityError("topic source geometry is invalid") from exc
        except (sqlite3.Error, OSError, DocumentStoreIntegrityError, ValueError, IndexError, KeyError) as exc:
            raise TopicRecordsIntegrityError("topic records database validation failed") from exc
        finally:
            if not success and minted_token is not None:
                with registry_lock:
                    failure_proof = token_registry.pop(minted_token, None)
                if failure_proof is not None:
                    failure_proof.close()
            if not success and connection is not None:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass
            for descriptor in (
                audit_descriptor,
                manifest_descriptor,
                database_descriptor,
            ):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            if snapshot_path is not None:
                snapshot_path.unlink(missing_ok=True)
            if snapshot_directory is not None:
                try:
                    snapshot_directory.rmdir()
                except OSError:
                    pass

    def reject_init(self, *args: object, **kwargs: object) -> None:
        raise TypeError("TopicRecords construction is internal-only")

    TopicRecords.__init__ = reject_init
    TopicRecords.open = classmethod(open_records)
    TopicRecords.assert_current = classmethod(assert_current)
    TopicRecords.close = close_records
    TopicRecords.__enter__ = enter_records
    TopicRecords.validate_all_sources = validate_all_sources
    TopicRecords.load_candidates = load_candidates
    TopicRecords.selection_pool = selection_pool
    TopicRecords.passage_search = passage_search
    TopicRecords.topic_snapshot = topic_snapshot
    TopicRecords.receipt = property(receipt_property)
    TopicRecords.topic_id = property(topic_id_property)
    TopicRecords.run_id = property(run_id_property)
    TopicRecords.validation_session = property(validation_session_property)
    TopicRecords._manifest = property(manifest_property)
    for name in ("_rebind", "_reconstruct"):
        if hasattr(TopicRecords, name):
            delattr(TopicRecords, name)
    return _ValidatedTopicRecords


ValidatedTopicRecords = _install_topic_records_capability()
del _install_topic_records_capability


def _validate_manifest(
    value: Mapping[str, object],
    database: Path,
    manifest: Path,
    expected_topic_id: str,
    database_receipt: tuple[str, int],
) -> None:
    required = {
        "schema_version", "stage", "topic_id", "run_id", "records_file", "manifest_file",
        "database_sha256", "database_bytes", "semantic_sha256", "identity_json",
        "document_sha256s", "request_key_closure", "row_counts",
    }
    if set(value) != required:
        raise TopicRecordsIntegrityError("records manifest fields are not exact")
    if value["schema_version"] != TOPIC_RECORDS_SCHEMA_VERSION or value["stage"] != CANDIDATE_STAGE:
        raise TopicRecordsIntegrityError("records manifest schema identity mismatch")
    if value["topic_id"] != expected_topic_id:
        raise TopicRecordsIntegrityError("records manifest topic identity mismatch")
    _require_text(value["run_id"], "records manifest run_id")
    if value["records_file"] != database.name or value["manifest_file"] != manifest.name:
        raise TopicRecordsIntegrityError("records manifest file identity mismatch")
    _require_digest(value["database_sha256"], "database_sha256")
    _require_digest(value["semantic_sha256"], "semantic_sha256")
    if isinstance(value["database_bytes"], bool) or not isinstance(value["database_bytes"], int) or value["database_bytes"] < 0:
        raise TopicRecordsIntegrityError("database_bytes must be non-negative")
    if not isinstance(value["identity_json"], str):
        raise TopicRecordsIntegrityError("identity_json must be text")
    try:
        identity = json.loads(value["identity_json"])
    except json.JSONDecodeError as exc:
        raise TopicRecordsIntegrityError("identity_json is not JSON") from exc
    if not isinstance(identity, dict) or _canonical_json(identity) != value["identity_json"]:
        raise TopicRecordsIntegrityError("identity_json is not canonical")
    document_hashes = value["document_sha256s"]
    if (
        not isinstance(document_hashes, list)
        or document_hashes != sorted(document_hashes)
        or len(document_hashes) != len(set(document_hashes))
        or any(_SHA256.fullmatch(str(item)) is None for item in document_hashes)
    ):
        raise TopicRecordsIntegrityError("document closure is not sorted and hashed")
    if value["request_key_closure"] != []:
        raise TopicRecordsIntegrityError("request-key closure must be empty for topic records v2")
    row_counts = value["row_counts"]
    if not isinstance(row_counts, dict) or set(row_counts) != {*_SCOPED_TABLES, "stage_seal"} or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in row_counts.values()):
        raise TopicRecordsIntegrityError("row counts are invalid")
    if database_receipt != (value["database_sha256"], value["database_bytes"]):
        raise TopicRecordsIntegrityError("database byte receipt does not match manifest")


def _expected_column_xinfo(table: str) -> tuple[tuple[object, ...], ...]:
    primary_key = _PRIMARY_KEYS[table]
    rowid_alias = _ROWID_PRIMARY_KEYS.get(table)
    result: list[tuple[object, ...]] = []
    for cid, name in enumerate(_SCHEMA_COLUMNS[table]):
        declared_type = (
            "INTEGER" if name in _INTEGER_COLUMNS
            else "REAL" if name in _REAL_COLUMNS
            else "TEXT"
        )
        primary_key_position = primary_key.index(name) + 1 if name in primary_key else 0
        not_null = 0 if (table, name) in _NULLABLE_COLUMNS or name == rowid_alias else 1
        result.append((cid, name, declared_type, not_null, None, primary_key_position, 0))
    return tuple(result)


def _expected_index_xinfo(
    table: str,
    columns: tuple[str, ...],
) -> tuple[tuple[object, ...], ...]:
    table_columns = _SCHEMA_COLUMNS[table]
    return tuple(
        (table_columns.index(column), column, 0, "BINARY", 1)
        for column in columns
    ) + ((-1, None, 0, "BINARY", 0),)


def _index_descriptors(
    database: sqlite3.Connection,
    table: str,
) -> tuple[tuple[object, ...], ...]:
    descriptors: list[tuple[object, ...]] = []
    for _sequence, name, unique, origin, partial in database.execute(
        f"PRAGMA index_list({_quote(table)})"
    ):
        rows = database.execute(f"PRAGMA index_xinfo({_quote(name)})").fetchall()
        if tuple(row[0] for row in rows) != tuple(range(len(rows))):
            raise TopicRecordsIntegrityError(f"{table} index schema is not exact")
        descriptors.append(
            (unique, origin, partial, tuple(tuple(row[1:]) for row in rows))
        )
    return tuple(sorted(descriptors, key=repr))


def _expected_index_descriptors(table: str) -> tuple[tuple[object, ...], ...]:
    descriptors: list[tuple[object, ...]] = []
    if table not in _ROWID_PRIMARY_KEYS:
        descriptors.append((1, "pk", 0, _expected_index_xinfo(table, _PRIMARY_KEYS[table])))
    descriptors.extend(
        (1, "u", 0, _expected_index_xinfo(table, columns))
        for columns in _UNIQUE_KEYS.get(table, ())
    )
    return tuple(sorted(descriptors, key=repr))


def _foreign_key_descriptors(
    database: sqlite3.Connection,
    table: str,
) -> tuple[tuple[object, ...], ...]:
    grouped: dict[int, list[tuple[object, ...]]] = {}
    for row in database.execute(f"PRAGMA foreign_key_list({_quote(table)})"):
        grouped.setdefault(row[0], []).append(tuple(row))
    descriptors: list[tuple[object, ...]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: row[1])
        if tuple(row[1] for row in rows) != tuple(range(len(rows))):
            raise TopicRecordsIntegrityError(f"{table} foreign-key schema is not exact")
        reference_table = rows[0][2]
        on_update, on_delete, match = rows[0][5:8]
        if any(
            row[2] != reference_table or row[5:8] != (on_update, on_delete, match)
            for row in rows
        ):
            raise TopicRecordsIntegrityError(f"{table} foreign-key schema is not exact")
        descriptors.append(
            (
                reference_table,
                tuple((row[3], row[4]) for row in rows),
                on_update,
                on_delete,
                match,
            )
        )
    return tuple(sorted(descriptors, key=repr))


def _expected_foreign_key_descriptors(table: str) -> tuple[tuple[object, ...], ...]:
    return tuple(sorted([
        (
            reference_table,
            columns,
            "NO ACTION",
            "NO ACTION",
            "NONE",
        )
        for reference_table, columns in _FOREIGN_KEYS.get(table, ())
    ], key=repr))


def _table_create_sql(database: sqlite3.Connection) -> dict[str, str]:
    return {
        str(name): str(sql)
        for name, sql in database.execute(
            "SELECT name, sql FROM sqlite_schema "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    }


def _reference_table_create_sql() -> dict[str, str]:
    reference = sqlite3.connect(":memory:")
    try:
        _create_schema(reference)
        return _table_create_sql(reference)
    finally:
        reference.close()


def _validate_database(database: sqlite3.Connection, manifest: Mapping[str, object]) -> None:
    if database.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise TopicRecordsIntegrityError("foreign keys are not enabled")
    if str(database.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "delete":
        raise TopicRecordsIntegrityError("topic database is not in rollback-journal mode")
    expected_table_list = {
        ("main", table, "table", len(columns), 0, 1)
        for table, columns in _SCHEMA_COLUMNS.items()
    }
    actual_table_list = {
        tuple(row)
        for row in database.execute("PRAGMA table_list")
        if not row[1].startswith("sqlite_")
    }
    if actual_table_list != expected_table_list:
        raise TopicRecordsIntegrityError("topic database schema is not exact")
    if _table_create_sql(database) != _reference_table_create_sql():
        raise TopicRecordsIntegrityError("topic database schema CREATE SQL is not exact")
    if database.execute(
        "SELECT 1 FROM sqlite_schema WHERE type IN ('view', 'trigger') LIMIT 1"
    ).fetchone() is not None:
        raise TopicRecordsIntegrityError("topic database schema has unexpected objects")
    for table in _SCHEMA_COLUMNS:
        column_xinfo = tuple(
            tuple(row)
            for row in database.execute(f"PRAGMA table_xinfo({_quote(table)})")
        )
        if column_xinfo != _expected_column_xinfo(table):
            raise TopicRecordsIntegrityError(f"{table} schema columns are not exact")
        if _index_descriptors(database, table) != _expected_index_descriptors(table):
            raise TopicRecordsIntegrityError(f"{table} index schema is not exact")
        if _foreign_key_descriptors(database, table) != _expected_foreign_key_descriptors(table):
            raise TopicRecordsIntegrityError(f"{table} foreign-key schema is not exact")
    if database.execute("PRAGMA foreign_key_check").fetchall():
        raise TopicRecordsIntegrityError("foreign-key validation failed")
    if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise TopicRecordsIntegrityError("SQLite integrity validation failed")
    topic_rows = database.execute("SELECT topic_id, run_id FROM topic_identity").fetchall()
    if len(topic_rows) != 1 or tuple(topic_rows[0]) != (
        manifest["topic_id"],
        manifest["run_id"],
    ):
        raise TopicRecordsIntegrityError("topic identity table is not one topic/run")
    for table in ("document_binding", "subnarrative_identity", "candidate"):
        if database.execute(
            f"SELECT COUNT(*) FROM {_quote(table)} WHERE topic_id IS NOT ?",
            (manifest["topic_id"],),
        ).fetchone()[0]:
            raise TopicRecordsIntegrityError("topic identity table is not one-topic")
    if _row_counts(database) != manifest["row_counts"]:
        raise TopicRecordsIntegrityError("manifest row counts do not match database")
    actual_document_hashes = sorted({
        row[0]
        for row in database.execute(
            "SELECT content_sha256 FROM document_binding WHERE topic_id=? ORDER BY docid",
            (manifest["topic_id"],),
        )
    })
    if actual_document_hashes != manifest["document_sha256s"]:
        raise TopicRecordsIntegrityError("manifest document closure does not match database")


def _reconstruct_candidate(
    database: sqlite3.Connection,
    store: DocumentStore,
    topic_id: str,
    subnarrative_id: str,
    candidate_id: str,
    *,
    geometry_index: DocumentGeometryIndex,
) -> tuple[ExtractiveCandidate, DocumentGeometry]:
    candidate_row = database.execute(
        "SELECT c.candidate_pk, c.topic_id, c.candidate_nugget_id, d.docid, d.content_sha256, "
        "c.subnarrative_id, c.candidate_kind, c.nugget_type, c.start_char, c.end_char, "
        "c.text_sha256, c.sentence_score, c.document_subnarrative_rank, c.scoring_text_sha256, "
        "c.subnarrative_sha256, c.sentence_splitter_version "
        "FROM candidate AS c JOIN document_binding AS d ON d.document_pk = c.document_pk "
        "WHERE c.topic_id=? AND c.subnarrative_id=? AND c.candidate_nugget_id=?",
        (topic_id, subnarrative_id, candidate_id),
    ).fetchone()
    if candidate_row is None:
        raise TopicRecordsIntegrityError("candidate is absent")
    candidate_pk, docid, content_sha256 = candidate_row[0], candidate_row[3], candidate_row[4]
    geometry = _geometry_for(geometry_index, store, content_sha256)
    source = geometry.source
    span_rows = database.execute(
        "SELECT role, ordinal, start_char, end_char, start_byte, end_byte, text_sha256, "
        "cross_encoder_score FROM candidate_span WHERE candidate_pk=? "
        "ORDER BY CASE role WHEN 'evidence' THEN 0 WHEN 'matched_paragraph' THEN 1 "
        "WHEN 'context_before' THEN 2 WHEN 'context_after' THEN 3 ELSE 4 END, ordinal",
        (candidate_pk,),
    ).fetchall()
    evidence: list[SentenceEvidence] = []
    matched: SourceSpan | None = None
    context_before: SourceSpan | None = None
    context_after: SourceSpan | None = None
    for role, ordinal, start_char, end_char, start_byte, end_byte, text_sha256, score in span_rows:
        span = SourceSpan(
            _safe_source_slice(source, start_char, end_char),
            start_char,
            end_char,
            start_byte,
            end_byte,
            text_sha256,
        )
        if role == "evidence":
            if ordinal != len(evidence):
                raise TopicRecordsIntegrityError("evidence span ordinals are not contiguous")
            if score is None:
                raise TopicRecordsIntegrityError("evidence span score is null")
            evidence.append(SentenceEvidence(**span.__dict__, cross_encoder_score=float(score)))
        elif role == "matched_paragraph" and ordinal == 0:
            if matched is not None:
                raise TopicRecordsIntegrityError("candidate has duplicate matched paragraphs")
            matched = span
        elif role == "context_before" and ordinal == 0:
            context_before = span
        elif role == "context_after" and ordinal == 0:
            context_after = span
        else:
            raise TopicRecordsIntegrityError("candidate span role is invalid")
    if matched is None:
        raise TopicRecordsIntegrityError("candidate has no matched paragraph")
    passage_rows = database.execute(
        "SELECT l.ordinal, l.query_id, p.passage_id, p.source_start_char, "
        "p.source_end_char, p.source_start_byte, p.source_end_byte, "
        "p.source_text_sha256, p.scoring_text_sha256, qp.raw_logit, qp.passage_rank "
        "FROM candidate_passage_link AS l "
        "JOIN passage AS p ON p.passage_pk = l.passage_pk AND p.document_pk = l.document_pk "
        "JOIN query_passage AS qp ON qp.query_id = l.query_id AND qp.passage_pk = l.passage_pk "
        "JOIN query_identity AS qi ON qi.query_id = qp.query_id "
        "WHERE l.candidate_pk=? ORDER BY l.ordinal",
        (candidate_pk,),
    ).fetchall()
    passages_list: list[PassageProvenance] = []
    for ordinal, passage_row in enumerate(passage_rows):
        if passage_row[0] != ordinal:
            raise TopicRecordsIntegrityError("passage ordinals are not contiguous")
        (
            _, query_id, passage_id, source_start_char, source_end_char,
            source_start_byte, source_end_byte, source_text_sha256,
            stored_scoring_text_sha256, raw_logit, passage_rank,
        ) = passage_row
        boundaries = geometry.scoring_boundaries
        try:
            scoring_start_char = boundaries.index(source_start_char)
            scoring_end_char = boundaries.index(source_end_char)
        except ValueError as exc:
            raise TopicRecordsIntegrityError(
                "passage source geometry is not a scoring projection"
            ) from exc
        scoring_text = geometry.scoring_text[scoring_start_char:scoring_end_char]
        passage_text = _safe_source_slice(source, source_start_char, source_end_char)
        passage_text_sha256 = _digest(passage_text)
        chunk_text_sha256 = _digest(scoring_text)
        if (
            passage_text_sha256 != source_text_sha256
            or stored_scoring_text_sha256 != passage_text_sha256
            or source_start_byte != geometry.byte_offsets[source_start_char]
            or source_end_byte != geometry.byte_offsets[source_end_char]
        ):
            raise TopicRecordsIntegrityError("passage source reconstruction is inconsistent")
        passages_list.append(PassageProvenance(
            passage_id=passage_id,
            lane_id="original",
            query_id=_decode_legacy_candidate_query_id(query_id),
            scoring_start_char=scoring_start_char,
            scoring_end_char=scoring_end_char,
            source_start_char=source_start_char,
            source_end_char=source_end_char,
            source_start_byte=source_start_byte,
            source_end_byte=source_end_byte,
            source_text=passage_text,
            source_text_sha256=source_text_sha256,
            scoring_text_sha256=geometry.scoring_text_sha256,
            chunk_text_sha256=chunk_text_sha256,
            normalization_version=SCORING_NORMALIZATION_VERSION,
            cross_encoder_score=raw_logit,
            cross_encoder_rank=passage_rank,
        ))
    passages = tuple(passages_list)
    if len(passages) != len(passage_rows):
        raise TopicRecordsIntegrityError("passage ordinals are not contiguous")
    if not evidence or not passages:
        raise TopicRecordsIntegrityError("candidate is missing evidence or passages")
    text = _safe_source_slice(source, evidence[0].start_char, evidence[-1].end_char)
    subnarrative_row = database.execute(
        "SELECT subnarrative_sha256 FROM subnarrative_identity "
        "WHERE topic_id=? AND subnarrative_id=?",
        (topic_id, subnarrative_id),
    ).fetchone()
    if subnarrative_row is None or subnarrative_row[0] != candidate_row[14]:
        raise TopicRecordsIntegrityError("candidate subnarrative identity is inconsistent")
    if (
        candidate_row[8] != evidence[0].start_char
        or candidate_row[9] != evidence[-1].end_char
        or candidate_row[10] != _digest(text)
    ):
        raise TopicRecordsIntegrityError("candidate offsets or text hash are inconsistent")
    candidate = ExtractiveCandidate(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        topic_id=candidate_row[1], docid=docid, subnarrative_id=candidate_row[5],
        candidate_nugget_id=candidate_row[2], nugget_type=candidate_row[7],
        candidate_kind=candidate_row[6],
        text=text, evidence_sentences=tuple(evidence), matched_paragraph=matched,
        context_before=context_before, context_after=context_after, passages=passages,
        sentence_cross_encoder_score=candidate_row[11],
        rank_within_document_subnarrative=candidate_row[12],
        document_sha256=content_sha256, scoring_text_sha256=candidate_row[13],
        subnarrative_sha256=candidate_row[14], sentence_splitter_version=candidate_row[15],
    )
    return candidate, geometry
