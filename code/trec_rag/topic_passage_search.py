"""Shared topic-scoped retrieval, passage scoring, and source geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from typing import Any, Protocol
from types import MappingProxyType

from .chunking import TextChunk
from .pipeline_models import QueryVariant, RetrievedCandidate


PASSAGE_SEARCH_SCHEMA_VERSION = "topic-passage-search-v1"
_SHA256_LENGTH = 64
_FOCUSED_QUERY_SOURCE_TYPE = "topic_passage_search"


class OrganizerRequestFailed(RuntimeError):
    """A retryable organizer transport failure."""


class OrganizerRequestTerminal(RuntimeError):
    """A terminal organizer transport outcome that must not be replayed."""


class PassageScoringFailed(RuntimeError):
    """A classified model/runtime failure while scoring passages."""


class DocumentStoreAdapter(Protocol):
    def admit_text(self, text: str, *, expected_sha256: str | None = None) -> Any:
        ...


class RetrieverAdapter(Protocol):
    def retrieve(
        self, query: QueryVariant, *, depth: int
    ) -> Sequence[RetrievedCandidate]:
        ...


class ChunkerAdapter(Protocol):
    def split_text(self, text: str, *, document_id: str) -> Sequence[TextChunk]:
        ...


class ScorerAdapter(Protocol):
    identity: Mapping[str, object]

    def cache_key(self, query_text: str, passage_text: str) -> str:
        ...

    def rank(self, query_text: str, chunks: Sequence[TextChunk]) -> Sequence[Any]:
        ...


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _digest_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from exc
    if value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class PassageSearchPolicy:
    retrieval_depth: int = 1000
    passage_limit: int = 100
    max_attempts: int = 3

    def __post_init__(self) -> None:
        for name in ("retrieval_depth", "passage_limit", "max_attempts"):
            _positive_int(getattr(self, name), name)


@dataclass(frozen=True)
class FocusedQuery:
    query_id: str
    text: str
    primary_subnarrative_id: str
    supporting_subnarrative_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_string(self.query_id, "query_id")
        _nonempty_string(self.text, "query text")
        _nonempty_string(self.primary_subnarrative_id, "primary_subnarrative_id")
        if not isinstance(self.supporting_subnarrative_ids, tuple):
            raise ValueError("supporting_subnarrative_ids must be a tuple")

        deduplicated: list[str] = []
        seen: set[str] = set()
        for facet_id in self.supporting_subnarrative_ids:
            _nonempty_string(facet_id, "supporting_subnarrative_id")
            if facet_id == self.primary_subnarrative_id or facet_id in seen:
                continue
            seen.add(facet_id)
            deduplicated.append(facet_id)
        object.__setattr__(self, "supporting_subnarrative_ids", tuple(deduplicated))


@dataclass(frozen=True)
class SourceDocument:
    docid: str
    content_sha256: str
    source_rank: int
    source_score: float
    best_passage_id: str | None
    best_passage_raw_logit: float | None

    def __post_init__(self) -> None:
        _nonempty_string(self.docid, "docid")
        _digest(self.content_sha256, "content_sha256")
        _positive_int(self.source_rank, "source_rank")
        object.__setattr__(self, "source_score", _finite_float(self.source_score, "source_score"))
        if (self.best_passage_id is None) != (self.best_passage_raw_logit is None):
            raise ValueError("best passage identity and score must be both present or both absent")
        if self.best_passage_id is not None:
            _nonempty_string(self.best_passage_id, "best_passage_id")
            object.__setattr__(
                self,
                "best_passage_raw_logit",
                _finite_float(self.best_passage_raw_logit, "best_passage_raw_logit"),
            )


@dataclass(frozen=True)
class SourcePassage:
    passage_id: str
    docid: str
    content_sha256: str
    source_rank: int
    source_score: float
    start_char: int
    end_char: int
    start_byte: int
    end_byte: int
    text_sha256: str
    text: str
    raw_logit: float
    rank: int
    score_cache_key: str
    scoring_text_sha256: str
    chunker_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        _nonempty_string(self.passage_id, "passage_id")
        _nonempty_string(self.docid, "docid")
        _digest(self.content_sha256, "content_sha256")
        _positive_int(self.source_rank, "source_rank")
        object.__setattr__(self, "source_score", _finite_float(self.source_score, "source_score"))
        for name in ("start_char", "end_char", "start_byte", "end_byte"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.end_char <= self.start_char or self.end_byte <= self.start_byte:
            raise ValueError("passage offsets must be non-empty ranges")
        _digest(self.text_sha256, "text_sha256")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("passage text must be non-empty")
        if _digest_text(self.text) != self.text_sha256:
            raise ValueError("text_sha256 does not match passage text")
        _digest(self.scoring_text_sha256, "scoring_text_sha256")
        if not isinstance(self.chunker_identity, Mapping):
            raise ValueError("chunker_identity must be a mapping")
        identity = dict(self.chunker_identity)
        if not identity:
            raise ValueError("chunker_identity must be a non-empty mapping")
        try:
            canonical = json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("chunker_identity must be JSON serializable") from exc
        if json.loads(canonical) != identity:
            raise ValueError("chunker_identity must be canonical JSON data")
        object.__setattr__(self, "chunker_identity", MappingProxyType(identity))
        object.__setattr__(self, "raw_logit", _finite_float(self.raw_logit, "raw_logit"))
        _positive_int(self.rank, "rank")
        _nonempty_string(self.score_cache_key, "score_cache_key")


@dataclass(frozen=True)
class _PassageDraft:
    passage_id: str
    docid: str
    content_sha256: str
    source_rank: int
    source_score: float
    start_char: int
    end_char: int
    start_byte: int
    end_byte: int
    text_sha256: str
    text: str
    raw_logit: float
    score_cache_key: str
    scoring_text_sha256: str


@dataclass(frozen=True)
class PassageSearchResult:
    query: FocusedQuery
    status: str
    stopping_reason: str | None
    requested_documents: int
    returned_documents: int
    scored_documents: int
    scored_passages: int
    documents: tuple[SourceDocument, ...]
    passages: tuple[SourcePassage, ...]
    attempt_count: int
    source_exhausted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.query, FocusedQuery):
            raise ValueError("query must be a FocusedQuery")
        if self.status not in {"complete", "incomplete"}:
            raise ValueError("status must be complete or incomplete")
        for name in ("returned_documents", "scored_documents", "scored_passages"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("attempt_count", "requested_documents"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.source_exhausted, bool):
            raise ValueError("source_exhausted must be a boolean")
        if self.returned_documents != len(self.documents):
            raise ValueError("returned_documents must equal the document row count")
        if self.returned_documents > self.requested_documents:
            raise ValueError("returned_documents cannot exceed requested_documents")
        if self.scored_documents > self.returned_documents:
            raise ValueError("scored_documents cannot exceed returned_documents")
        if self.scored_passages < len(self.passages):
            raise ValueError("scored_passages cannot be less than returned passages")

        documents_by_id: dict[str, SourceDocument] = {}
        for document in self.documents:
            if not isinstance(document, SourceDocument):
                raise ValueError("documents must contain SourceDocument rows")
            if document.docid in documents_by_id:
                raise ValueError("document IDs must be unique")
            documents_by_id[document.docid] = document

        passages_by_id: dict[str, SourcePassage] = {}
        passage_ranks: set[int] = set()
        passages_by_docid: dict[str, list[SourcePassage]] = {}
        for passage in self.passages:
            if not isinstance(passage, SourcePassage):
                raise ValueError("passages must contain SourcePassage rows")
            if passage.passage_id in passages_by_id:
                raise ValueError("passage IDs must be unique")
            if passage.rank in passage_ranks:
                raise ValueError("passage ranks must be unique")
            document = documents_by_id.get(passage.docid)
            if document is None or (
                passage.content_sha256 != document.content_sha256
                or passage.source_rank != document.source_rank
                or passage.source_score != document.source_score
            ):
                raise ValueError("passage does not match its listed source document")
            passages_by_id[passage.passage_id] = passage
            passage_ranks.add(passage.rank)
            passages_by_docid.setdefault(passage.docid, []).append(passage)

        ranks = [passage.rank for passage in self.passages]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("returned passage ranks must be contiguous from 1")
        if self.status == "complete":
            if self.stopping_reason is not None or not self.passages:
                raise ValueError("complete result requires no reason and non-empty passages")
            scored_document_count = sum(
                document.best_passage_id is not None for document in self.documents
            )
            if self.scored_documents != scored_document_count:
                raise ValueError(
                    "scored_documents must equal documents with a best passage"
                )
            expected_exhaustion = self.returned_documents < self.requested_documents
            if self.source_exhausted != expected_exhaustion:
                raise ValueError(
                    "complete result source_exhausted must match returned_documents "
                    "being less than requested_documents"
                )
            self._validate_best_passages(
                documents_by_id,
                passages_by_id,
                passages_by_docid,
            )
            return

        if self.stopping_reason not in {
            "retrieval_unavailable",
            "scoring_failed",
            "no_evidence",
        }:
            raise ValueError("incomplete result requires an enumerated stopping_reason")
        if (
            self.passages
            or self.scored_documents != 0
            or self.scored_passages != 0
            or any(document.best_passage_id is not None for document in self.documents)
        ):
            raise ValueError(
                "incomplete result cannot contain passages, scores, or document best passages"
            )
        if self.stopping_reason == "retrieval_unavailable":
            if self.documents or self.source_exhausted:
                raise ValueError(
                    "retrieval_unavailable result must have no documents and "
                    "source_exhausted False"
                )
            return
        expected_exhaustion = self.returned_documents < self.requested_documents
        if self.source_exhausted != expected_exhaustion:
            raise ValueError(
                "incomplete result source_exhausted must match returned_documents "
                "being less than requested_documents"
            )

    def _validate_best_passages(
        self,
        documents_by_id: Mapping[str, SourceDocument],
        passages_by_id: Mapping[str, SourcePassage],
        passages_by_docid: Mapping[str, Sequence[SourcePassage]],
    ) -> None:
        complete_passage_set = self.scored_passages == len(self.passages)
        for docid, document in documents_by_id.items():
            rows = tuple(passages_by_docid.get(docid, ()))
            if complete_passage_set:
                expected = min(
                    rows,
                    key=lambda row: (-row.raw_logit, row.passage_id),
                    default=None,
                )
                if expected is None:
                    if document.best_passage_id is not None:
                        raise ValueError(
                            "document best passage has no scored passage row"
                        )
                elif (
                    document.best_passage_id != expected.passage_id
                    or document.best_passage_raw_logit != expected.raw_logit
                ):
                    raise ValueError("document best passage is not its actual best passage")
                continue

            if rows and document.best_passage_id is None:
                raise ValueError("scored document is missing its best passage")
            if document.best_passage_id is None:
                continue
            returned_best = passages_by_id.get(document.best_passage_id)
            if returned_best is not None and (
                returned_best.docid != document.docid
                or returned_best.raw_logit != document.best_passage_raw_logit
            ):
                raise ValueError("document best passage binding is inconsistent")
            if rows:
                best_returned = min(
                    rows,
                    key=lambda row: (-row.raw_logit, row.passage_id),
                )
                document_best_key = (
                    -float(document.best_passage_raw_logit),
                    str(document.best_passage_id),
                )
                returned_best_key = (
                    -best_returned.raw_logit,
                    best_returned.passage_id,
                )
                if document_best_key > returned_best_key:
                    raise ValueError(
                        "document best passage cannot rank below a returned passage"
                    )


def passage_id(
    content_sha256: str,
    docid: str,
    chunk: TextChunk,
    chunker_identity: Mapping[str, object],
) -> str:
    """Return a query-independent identity for one exact source span."""
    _nonempty_string(docid, "docid")
    body = json.dumps(
        {
            "schema_version": PASSAGE_SEARCH_SCHEMA_VERSION,
            "docid": docid,
            "content_sha256": content_sha256,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "text_sha256": _digest_text(chunk.text),
            "chunker": dict(chunker_identity),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "p-" + sha256(body).hexdigest()


@dataclass(frozen=True)
class _RetainedDocument:
    candidate: RetrievedCandidate
    content_sha256: str
    text: str
    chunks: tuple[TextChunk, ...]


class TopicPassageSearch:
    """Retrieve, score, and source-bind all passages for one focused query."""

    def __init__(
        self,
        topic_id: str,
        document_store: DocumentStoreAdapter,
        retriever: RetrieverAdapter,
        chunker: ChunkerAdapter,
        scorer: ScorerAdapter,
        policy: PassageSearchPolicy | None = None,
    ) -> None:
        self.topic_id = _nonempty_string(topic_id, "topic_id")
        self._document_store = document_store
        self._retriever = retriever
        self._chunker = chunker
        self._scorer = scorer
        self._policy = policy or PassageSearchPolicy()
        scorer_identity = getattr(scorer, "identity", None)
        if not isinstance(scorer_identity, Mapping):
            raise ValueError("scorer identity must be a mapping")
        try:
            json.dumps(dict(scorer_identity), sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("scorer identity must be JSON serializable") from exc
        self._chunker_identity = self._get_chunker_identity(chunker)

    @staticmethod
    def _get_chunker_identity(chunker: ChunkerAdapter) -> dict[str, object]:
        supplied = getattr(chunker, "identity", None)
        if isinstance(supplied, Mapping):
            identity = dict(supplied)
        else:
            config = getattr(chunker, "config", None)
            identity = {
                "backend": f"{type(chunker).__module__}.{type(chunker).__qualname__}",
            }
            if config is not None:
                for name in ("max_characters", "overlap_characters", "trim"):
                    if hasattr(config, name):
                        identity[name] = getattr(config, name)
        try:
            canonical = json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("chunker identity must be JSON serializable") from exc
        if not identity or json.loads(canonical) != identity:
            raise ValueError("chunker identity must be a non-empty canonical JSON mapping")
        return identity

    def search(self, query: FocusedQuery) -> PassageSearchResult:
        if not isinstance(query, FocusedQuery):
            raise ValueError("query must be a FocusedQuery")
        request = QueryVariant(
            self.topic_id,
            query.query_id,
            query.text,
            _FOCUSED_QUERY_SOURCE_TYPE,
        )
        candidates, attempt_count = self._retrieve_with_retries(request)
        if candidates is None:
            return self._result(
                query,
                status="incomplete",
                stopping_reason="retrieval_unavailable",
                attempt_count=attempt_count,
                documents=(),
                passages=(),
                scored_documents=0,
                scored_passages=0,
                source_exhausted=False,
            )

        retained = self._retain_documents(candidates, request)
        source_exhausted = len(retained) < self._policy.retrieval_depth
        if not retained:
            return self._result(
                query,
                status="incomplete",
                stopping_reason="no_evidence",
                attempt_count=attempt_count,
                documents=(),
                passages=(),
                scored_documents=0,
                scored_passages=0,
                source_exhausted=source_exhausted,
            )

        retained = self._admit_and_chunk(retained)
        chunks = tuple(chunk for document in retained for chunk in document.chunks)
        if not chunks:
            documents = self._source_documents(retained, ())
            return self._result(
                query,
                status="incomplete",
                stopping_reason="no_evidence",
                attempt_count=attempt_count,
                documents=documents,
                passages=(),
                scored_documents=0,
                scored_passages=0,
                source_exhausted=source_exhausted,
            )

        try:
            scored_rows = self._scorer.rank(query.text, chunks)
        except PassageScoringFailed:
            documents = self._source_documents(retained, ())
            return self._result(
                query,
                status="incomplete",
                stopping_reason="scoring_failed",
                attempt_count=attempt_count,
                documents=documents,
                passages=(),
                scored_documents=0,
                scored_passages=0,
                source_exhausted=source_exhausted,
            )

        scores = self._validate_scores(scored_rows, chunks)
        drafts = self._build_passage_drafts(query.text, retained, scores)
        documents = self._source_documents(retained, drafts)
        if not drafts:
            return self._result(
                query,
                status="incomplete",
                stopping_reason="no_evidence",
                attempt_count=attempt_count,
                documents=documents,
                passages=(),
                scored_documents=0,
                scored_passages=0,
                source_exhausted=source_exhausted,
            )

        ordered = sorted(
            drafts,
            key=lambda row: (-row.raw_logit, row.source_rank, row.passage_id),
        )
        ranked = tuple(
            SourcePassage(
                row.passage_id,
                row.docid,
                row.content_sha256,
                row.source_rank,
                row.source_score,
                row.start_char,
                row.end_char,
                row.start_byte,
                row.end_byte,
                row.text_sha256,
                row.text,
                row.raw_logit,
                index,
                row.score_cache_key,
                row.scoring_text_sha256,
                self._chunker_identity,
            )
            for index, row in enumerate(ordered, start=1)
        )
        scored_document_ids = {row.docid for row in ranked}
        return self._result(
            query,
            status="complete",
            stopping_reason=None,
            attempt_count=attempt_count,
            documents=documents,
            passages=ranked[: self._policy.passage_limit],
            scored_documents=len(scored_document_ids),
            scored_passages=len(ranked),
            source_exhausted=source_exhausted,
        )

    def _retrieve_with_retries(
        self, request: QueryVariant
    ) -> tuple[tuple[RetrievedCandidate, ...] | None, int]:
        for attempt in range(1, self._policy.max_attempts + 1):
            try:
                return tuple(
                    self._retriever.retrieve(
                        request,
                        depth=self._policy.retrieval_depth,
                    )
                ), attempt
            except OrganizerRequestTerminal:
                return None, attempt
            except OrganizerRequestFailed:
                if attempt == self._policy.max_attempts:
                    return None, attempt
        raise AssertionError("retry loop did not return")

    def _retain_documents(
        self,
        candidates: Sequence[RetrievedCandidate],
        request: QueryVariant,
    ) -> tuple[_RetainedDocument, ...]:
        normalized: list[RetrievedCandidate] = []
        for candidate in candidates:
            if not isinstance(candidate, RetrievedCandidate):
                raise ValueError("retriever returned an invalid candidate")
            if candidate.topic_id != self.topic_id:
                raise ValueError("retriever returned a candidate for a different topic")
            if candidate.variant_name != request.variant_name or candidate.query_text != request.query_text:
                raise ValueError("retriever returned a candidate for a different query")
            _nonempty_string(candidate.docid, "candidate docid")
            _positive_int(candidate.rank, "candidate rank")
            _finite_float(candidate.score, "candidate score")
            if not isinstance(candidate.text, str):
                raise ValueError("candidate text must be a string")
            normalized.append(candidate)

        candidates_by_docid: dict[str, list[RetrievedCandidate]] = {}
        bodies_by_docid: dict[str, str] = {}
        for candidate in normalized:
            previous_body = bodies_by_docid.setdefault(candidate.docid, candidate.text)
            if previous_body != candidate.text:
                raise ValueError(f"conflicting duplicate docid {candidate.docid!r}")
            candidates_by_docid.setdefault(candidate.docid, []).append(candidate)

        unique: list[RetrievedCandidate] = []
        for rows in candidates_by_docid.values():
            lowest_rank = min(row.rank for row in rows)
            lowest_rank_rows = [row for row in rows if row.rank == lowest_rank]
            highest_score = max(float(row.score) for row in lowest_rank_rows)
            finalists = [
                row for row in lowest_rank_rows if float(row.score) == highest_score
            ]
            selected = finalists[0]
            if any(row != selected for row in finalists[1:]):
                raise ValueError("duplicate candidate metadata conflicts after rank and score")
            unique.append(selected)

        unique.sort(key=lambda row: (row.rank, row.docid))

        retained = unique[: self._policy.retrieval_depth]
        return tuple(
            _RetainedDocument(candidate, "", candidate.text, ()) for candidate in retained
        )

    def _admit_and_chunk(
        self, documents: Sequence[_RetainedDocument]
    ) -> tuple[_RetainedDocument, ...]:
        admitted: list[_RetainedDocument] = []
        for document in documents:
            receipt = self._document_store.admit_text(
                document.text,
                expected_sha256=_digest_text(document.text),
            )
            content_sha256 = getattr(receipt, "content_sha256")
            expected_sha256 = _digest_text(document.text)
            if content_sha256 != expected_sha256:
                raise ValueError("document store returned a conflicting content digest")
            chunks = tuple(
                self._chunker.split_text(document.text, document_id=document.candidate.docid)
            ) if document.text else ()
            self._validate_chunks(chunks, document.candidate.docid, document.text)
            admitted.append(replace(document, content_sha256=content_sha256, chunks=chunks))
        return tuple(admitted)

    @staticmethod
    def _validate_chunks(
        chunks: Sequence[TextChunk], document_id: str, document_text: str
    ) -> None:
        seen_ids: set[str] = set()
        seen_spans: set[tuple[int, int, str]] = set()
        for chunk in chunks:
            if not isinstance(chunk, TextChunk):
                raise ValueError("chunker returned an invalid text chunk")
            if chunk.document_id != document_id:
                raise ValueError("chunker returned a chunk for a different document")
            if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id:
                raise ValueError("chunker returned a chunk without an ID")
            if chunk.chunk_id in seen_ids:
                raise ValueError("chunker returned duplicate chunk IDs")
            seen_ids.add(chunk.chunk_id)
            for name in ("start_char", "end_char"):
                value = getattr(chunk, name)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("chunker returned non-integer character offsets")
            if (
                chunk.start_char < 0
                or chunk.end_char <= chunk.start_char
                or chunk.end_char > len(document_text)
                or document_text[chunk.start_char : chunk.end_char] != chunk.text
                or not chunk.text
            ):
                raise ValueError("chunker returned an invalid source-bound text chunk")
            span_identity = (chunk.start_char, chunk.end_char, chunk.text)
            if span_identity in seen_spans:
                raise ValueError("chunker returned a duplicate exact source span")
            seen_spans.add(span_identity)

    @staticmethod
    def _validate_scores(
        scored_rows: Sequence[Any], chunks: Sequence[TextChunk]
    ) -> tuple[tuple[TextChunk, float], ...]:
        try:
            rows = tuple(scored_rows)
        except TypeError as exc:
            raise ValueError("scorer must return a sequence of scores") from exc
        if len(rows) != len(chunks):
            raise ValueError("scorer must return exactly one score per input chunk")

        valid_chunks = set(chunks)
        seen: set[TextChunk] = set()
        validated: list[tuple[TextChunk, float]] = []
        for index, row in enumerate(rows):
            if isinstance(row, (int, float)) and not isinstance(row, bool):
                chunk = chunks[index]
                score = row
            else:
                chunk = getattr(row, "chunk", None)
                if not isinstance(chunk, TextChunk) or chunk not in valid_chunks:
                    raise ValueError("scorer returned an unknown text chunk")
                if hasattr(row, "relevance_score"):
                    score = getattr(row, "relevance_score")
                elif hasattr(row, "raw_logit"):
                    score = getattr(row, "raw_logit")
                else:
                    raise ValueError("scorer returned a row without a score")
            if chunk in seen:
                raise ValueError("scorer must return exactly one score per input chunk")
            validated.append((chunk, _finite_float(score, "scorer score")))
            seen.add(chunk)
        if seen != valid_chunks:
            raise ValueError("scorer must return exactly one score per input chunk")
        return tuple(validated)

    def _build_passage_drafts(
        self,
        query_text: str,
        documents: Sequence[_RetainedDocument],
        scores: Sequence[tuple[TextChunk, float]],
    ) -> tuple[_PassageDraft, ...]:
        documents_by_id = {document.candidate.docid: document for document in documents}
        passages: list[_PassageDraft] = []
        for chunk, raw_logit in scores:
            document = documents_by_id.get(chunk.document_id)
            if document is None:
                raise ValueError("scored chunk is not bound to a retained source document")
            if document.text[chunk.start_char : chunk.end_char] != chunk.text:
                raise ValueError("scored chunk is not bound to the exact source text")
            start_byte = len(document.text[: chunk.start_char].encode("utf-8"))
            end_byte = len(document.text[: chunk.end_char].encode("utf-8"))
            passages.append(
                _PassageDraft(
                    passage_id(
                        document.content_sha256,
                        document.candidate.docid,
                        chunk,
                        self._chunker_identity,
                    ),
                    document.candidate.docid,
                    document.content_sha256,
                    document.candidate.rank,
                    float(document.candidate.score),
                    chunk.start_char,
                    chunk.end_char,
                    start_byte,
                    end_byte,
                    _digest_text(chunk.text),
                    chunk.text,
                    raw_logit,
                    self._scorer.cache_key(query_text, chunk.text),
                    _digest_text(chunk.text),
                )
            )
        return tuple(passages)

    @staticmethod
    def _source_documents(
        documents: Sequence[_RetainedDocument], passages: Sequence[_PassageDraft]
    ) -> tuple[SourceDocument, ...]:
        best_by_docid: dict[str, _PassageDraft] = {}
        for passage in sorted(passages, key=lambda row: (-row.raw_logit, row.passage_id)):
            best_by_docid.setdefault(passage.docid, passage)
        return tuple(
            SourceDocument(
                document.candidate.docid,
                document.content_sha256,
                document.candidate.rank,
                float(document.candidate.score),
                (
                    best_by_docid[document.candidate.docid].passage_id
                    if document.candidate.docid in best_by_docid
                    else None
                ),
                (
                    best_by_docid[document.candidate.docid].raw_logit
                    if document.candidate.docid in best_by_docid
                    else None
                ),
            )
            for document in documents
        )

    def _result(
        self,
        query: FocusedQuery,
        *,
        status: str,
        stopping_reason: str | None,
        attempt_count: int,
        documents: Sequence[SourceDocument],
        passages: Sequence[SourcePassage],
        scored_documents: int,
        scored_passages: int,
        source_exhausted: bool,
    ) -> PassageSearchResult:
        return PassageSearchResult(
            query=query,
            status=status,
            stopping_reason=stopping_reason,
            requested_documents=self._policy.retrieval_depth,
            returned_documents=len(documents),
            scored_documents=scored_documents,
            scored_passages=scored_passages,
            documents=tuple(documents),
            passages=tuple(passages),
            attempt_count=attempt_count,
            source_exhausted=source_exhausted,
        )
