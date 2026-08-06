"""Pure projection from agentic grounded evidence to competition artifacts.

The agent ledger decides membership: every document supporting a live nugget is
submitted, while only nuggets explicitly selected into a need's bounded draft
become RAG evidence.  Source text always comes from the authenticated document
store and sealed passage snapshot, never from a model-written quote.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import re

from .deepagent_evidence import EvidenceCoverageReport, NeedReport, NuggetReport
from .deepagent_submission import AgenticDocumentRank, rank_agentic_documents
from .document_store import DocumentStore, DocumentStoreIntegrityError
from .generation_handoff import (
    ClaimHint,
    EvidenceGroup,
    EvidencePassage,
    EvidenceSourceSpan,
    GenerationTopic,
    SelectedCluster,
    TopicSourceReceipts,
)
from .topic_passage_search import SourceDocument, SourcePassage
from .topic_records import TopicEvidenceSnapshot
from .topics import Topic


AGENTIC_RETRIEVAL_TOPIC_SCHEMA = "agentic_retrieval_topic_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class AgenticProjectionError(ValueError):
    """Agentic evidence cannot be projected without breaking provenance."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AgenticProjectionError(
            f"agentic retrieval projection is not canonical JSON: {exc}"
        ) from exc


def _text_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _stable_id(kind: str, *parts: str) -> str:
    identity = _canonical_json([kind, *parts])
    return f"agentic-{kind}-{sha256(identity).hexdigest()}"


def _require_unique(
    rows: Sequence[object], *, attribute: str, label: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for row in rows:
        value = getattr(row, attribute, None)
        if not isinstance(value, str) or not value:
            raise AgenticProjectionError(f"{label} identity must be non-empty text")
        if value in result:
            raise AgenticProjectionError(f"duplicate {label} identity: {value}")
        result[value] = row
    return result


@dataclass(frozen=True)
class AgenticRetrievalRow:
    """One variable-depth retrieval row before the run tag is attached."""

    topic_id: str
    docid: str
    rank: int
    score: int

    def to_payload(self) -> dict[str, object]:
        return {
            "topic_id": self.topic_id,
            "q0": "Q0",
            "docid": self.docid,
            "rank": self.rank,
            "score": self.score,
        }

    def to_trec_fields(self, *, run_id: str) -> tuple[str, str, str, int, int, str]:
        if not isinstance(run_id, str) or not run_id or any(char.isspace() for char in run_id):
            raise AgenticProjectionError("run_id must be non-empty text without whitespace")
        return (self.topic_id, "Q0", self.docid, self.rank, self.score, run_id)


@dataclass(frozen=True)
class AgenticFullTextCandidate:
    """One authenticated full document corresponding to a retrieval row."""

    docid: str
    text: str
    rank: int
    score: int
    document_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "docid": self.docid,
            "doc": self.text,
            "rank": self.rank,
            "score": self.score,
            "lane_ids": ["agentic"],
            "text_sha256": self.document_sha256,
        }


@dataclass(frozen=True)
class AgenticTopicProjection:
    """Prepared Retrieval and Generation projections for exactly one topic."""

    topic_id: str
    narrative: str
    retrieval_rows: tuple[AgenticRetrievalRow, ...]
    full_text_candidates: tuple[AgenticFullTextCandidate, ...]
    generation_topic: GenerationTopic
    retrieval_topic_sha256: str

    @property
    def full_text_record(self) -> dict[str, object]:
        return {
            "query": {
                "qid": self.topic_id,
                "selection_id": "official",
                "text": self.narrative,
                "text_sha256": _text_digest(self.narrative),
            },
            "candidates": [row.to_payload() for row in self.full_text_candidates],
        }

    def retrieval_payload(self) -> dict[str, object]:
        return {
            "schema_version": AGENTIC_RETRIEVAL_TOPIC_SCHEMA,
            "topic_id": self.topic_id,
            "retrieval_rows": [row.to_payload() for row in self.retrieval_rows],
            "full_text_record": self.full_text_record,
        }


def serialize_agentic_retrieval_topic(projection: AgenticTopicProjection) -> bytes:
    """Return the canonical bytes authenticated by ``retrieval_topic_sha256``."""

    if not isinstance(projection, AgenticTopicProjection):
        raise AgenticProjectionError("projection must be AgenticTopicProjection")
    return _canonical_json(projection.retrieval_payload()) + b"\n"


def _serialize_retrieval_parts(
    *,
    topic_id: str,
    narrative: str,
    retrieval_rows: tuple[AgenticRetrievalRow, ...],
    full_text_candidates: tuple[AgenticFullTextCandidate, ...],
) -> bytes:
    record = {
        "schema_version": AGENTIC_RETRIEVAL_TOPIC_SCHEMA,
        "topic_id": topic_id,
        "retrieval_rows": [row.to_payload() for row in retrieval_rows],
        "full_text_record": {
            "query": {
                "qid": topic_id,
                "selection_id": "official",
                "text": narrative,
                "text_sha256": _text_digest(narrative),
            },
            "candidates": [row.to_payload() for row in full_text_candidates],
        },
    }
    return _canonical_json(record) + b"\n"


def _validate_candidate_topic_ids(
    topic_id: str, fused_candidates: Sequence[object], searches: Sequence[object]
) -> None:
    candidates = list(fused_candidates)
    for search in searches:
        search_candidates = getattr(search, "candidates", None)
        if not isinstance(search_candidates, Sequence):
            raise AgenticProjectionError("search candidates must be a sequence")
        candidates.extend(search_candidates)
    for candidate in candidates:
        candidate_topic = getattr(candidate, "topic_id", topic_id)
        if candidate_topic != topic_id:
            raise AgenticProjectionError(
                f"retrieval candidate belongs to another topic: {candidate_topic!r}"
            )


def _read_ranked_documents(
    *,
    ranked: tuple[AgenticDocumentRank, ...],
    documents: Mapping[str, SourceDocument],
    document_store: DocumentStore,
) -> tuple[AgenticFullTextCandidate, ...]:
    candidates: list[AgenticFullTextCandidate] = []
    for expected_rank, row in enumerate(ranked, start=1):
        if row.rank != expected_rank or row.score != len(ranked) - expected_rank + 1:
            raise AgenticProjectionError("agentic document ranks are not contiguous")
        source = documents.get(row.document_id)
        if source is None:
            raise AgenticProjectionError(
                f"grounded retrieval row references unknown document: {row.document_id}"
            )
        try:
            text = document_store.read_text(source.content_sha256)
        except DocumentStoreIntegrityError as exc:
            raise AgenticProjectionError(
                f"unable to verify stored document {row.document_id}: {exc}"
            ) from exc
        if not text:
            raise AgenticProjectionError(
                f"stored document is empty: {row.document_id}"
            )
        if _text_digest(text) != source.content_sha256:
            raise AgenticProjectionError(
                f"stored document hash changed: {row.document_id}"
            )
        candidates.append(
            AgenticFullTextCandidate(
                docid=row.document_id,
                text=text,
                rank=row.rank,
                score=row.score,
                document_sha256=source.content_sha256,
            )
        )
    return tuple(candidates)


def _validate_source_passage(
    *,
    reference_docid: str,
    passage: SourcePassage,
    source_document: SourceDocument,
    document_text: str,
) -> None:
    if passage.docid != reference_docid:
        raise AgenticProjectionError(
            f"cited passage document differs from ledger reference: {passage.passage_id}"
        )
    if passage.content_sha256 != source_document.content_sha256:
        raise AgenticProjectionError(
            f"cited passage document hash differs from snapshot: {passage.passage_id}"
        )
    if _text_digest(document_text) != source_document.content_sha256:
        raise AgenticProjectionError(
            f"stored document hash differs from snapshot: {source_document.docid}"
        )
    if (
        passage.end_char > len(document_text)
        or document_text[passage.start_char : passage.end_char] != passage.text
    ):
        raise AgenticProjectionError(
            f"cited passage source span differs from stored document: {passage.passage_id}"
        )
    document_bytes = document_text.encode("utf-8")
    passage_bytes = passage.text.encode("utf-8")
    if (
        passage.end_byte > len(document_bytes)
        or document_bytes[passage.start_byte : passage.end_byte] != passage_bytes
    ):
        raise AgenticProjectionError(
            f"cited passage byte source span differs from stored document: {passage.passage_id}"
        )


def _selected_generation_records(
    *,
    topic: Topic,
    report: EvidenceCoverageReport,
    passages: Mapping[str, SourcePassage],
    documents: Mapping[str, SourceDocument],
    document_texts: Mapping[str, str],
    document_ranks: Mapping[str, int],
) -> tuple[
    tuple[EvidenceGroup, ...], tuple[EvidencePassage, ...], tuple[ClaimHint, ...]
]:
    nuggets = _require_unique(report.nuggets, attribute="nugget_id", label="nugget")
    _require_unique(report.needs, attribute="need_id", label="need")
    groups: list[EvidenceGroup] = []
    evidence_rows: list[EvidencePassage] = []
    claims: list[ClaimHint] = []
    selected_count = 0

    for need_object in report.needs:
        need = need_object
        if not isinstance(need, NeedReport):
            raise AgenticProjectionError("report needs must contain NeedReport rows")
        if not need.draft_nugget_ids:
            continue
        if not need.question:
            raise AgenticProjectionError(f"selected need has no question: {need.need_id}")
        if len(set(need.draft_nugget_ids)) != len(need.draft_nugget_ids):
            raise AgenticProjectionError(
                f"selected need repeats a nugget: {need.need_id}"
            )

        group_id = _stable_id("group", topic.id, need.need_id)
        clusters: list[SelectedCluster] = []
        for cluster_ordinal, nugget_id in enumerate(need.draft_nugget_ids, start=1):
            raw_nugget = nuggets.get(nugget_id)
            if raw_nugget is None:
                raise AgenticProjectionError(
                    f"unknown selected nugget {nugget_id!r} for need {need.need_id!r}"
                )
            nugget = raw_nugget
            if not isinstance(nugget, NuggetReport):
                raise AgenticProjectionError("report nuggets must contain NuggetReport rows")
            if nugget.superseded_by is not None:
                raise AgenticProjectionError(
                    f"selected nugget is superseded: {nugget.nugget_id}"
                )
            if need.need_id not in nugget.need_ids or nugget.nugget_id not in need.nugget_ids:
                raise AgenticProjectionError(
                    f"selected nugget is not associated with need {need.need_id}: "
                    f"{nugget.nugget_id}"
                )
            if not nugget.evidence:
                raise AgenticProjectionError(
                    f"selected nugget has no grounded evidence: {nugget.nugget_id}"
                )

            cluster_id = _stable_id(
                "cluster", topic.id, need.need_id, nugget.nugget_id
            )
            cluster_evidence_ids: list[str] = []
            seen_passages: set[str] = set()
            for reference in nugget.evidence:
                passage = passages.get(reference.snippet_id)
                if passage is None:
                    raise AgenticProjectionError(
                        f"selected nugget cites unknown passage: {reference.snippet_id}"
                    )
                source_document = documents.get(reference.document_id)
                if source_document is None:
                    raise AgenticProjectionError(
                        f"selected nugget cites unknown document: {reference.document_id}"
                    )
                document_text = document_texts.get(reference.document_id)
                if document_text is None:
                    raise AgenticProjectionError(
                        f"cited document is absent from retrieval projection: "
                        f"{reference.document_id}"
                    )
                _validate_source_passage(
                    reference_docid=reference.document_id,
                    passage=passage,
                    source_document=source_document,
                    document_text=document_text,
                )
                if passage.passage_id in seen_passages:
                    continue
                seen_passages.add(passage.passage_id)
                document_rank = document_ranks.get(reference.document_id)
                if document_rank is None:
                    raise AgenticProjectionError(
                        f"cited document is absent from retrieval ranks: "
                        f"{reference.document_id}"
                    )
                evidence_id = _stable_id(
                    "evidence",
                    topic.id,
                    need.need_id,
                    nugget.nugget_id,
                    passage.docid,
                    passage.passage_id,
                )
                cluster_evidence_ids.append(evidence_id)
                evidence_rows.append(
                    EvidencePassage(
                        evidence_id=evidence_id,
                        group_id=group_id,
                        cluster_id=cluster_id,
                        cluster_ordinal=cluster_ordinal,
                        support_ordinal=len(cluster_evidence_ids),
                        candidate_kind="agentic_grounded_nugget",
                        docid=passage.docid,
                        document_rank=document_rank,
                        text=passage.text,
                        document_sha256=source_document.content_sha256,
                        source_span=EvidenceSourceSpan(
                            start_char=passage.start_char,
                            end_char=passage.end_char,
                            start_byte=passage.start_byte,
                            end_byte=passage.end_byte,
                        ),
                    )
                )
            if not cluster_evidence_ids:
                raise AgenticProjectionError(
                    f"selected nugget has no grounded evidence: {nugget.nugget_id}"
                )
            clusters.append(
                SelectedCluster(
                    cluster_id=cluster_id,
                    ordinal=cluster_ordinal,
                    representative_evidence_id=cluster_evidence_ids[0],
                    evidence_ids=tuple(cluster_evidence_ids),
                )
            )
            claims.append(
                ClaimHint(
                    claim_id=_stable_id(
                        "claim", topic.id, need.need_id, nugget.nugget_id
                    ),
                    group_id=group_id,
                    kind="agentic_grounded_nugget",
                    text=nugget.text,
                    evidence_ids=tuple(cluster_evidence_ids),
                )
            )
            selected_count += 1
        groups.append(
            EvidenceGroup(
                group_id=group_id,
                kind="generated_subnarrative",
                text=need.question,
                selected_clusters=tuple(clusters),
            )
        )

    if selected_count == 0:
        raise AgenticProjectionError("topic has no selected grounded evidence")
    return tuple(groups), tuple(evidence_rows), tuple(claims)


def prepare_agentic_topic_projection(
    *,
    topic: Topic,
    report: EvidenceCoverageReport,
    snapshot: TopicEvidenceSnapshot,
    fused_candidates: Sequence[object],
    searches: Sequence[object],
    document_store: DocumentStore,
    official_topics_sha256: str,
) -> AgenticTopicProjection:
    """Build one deterministic, provenance-closed agentic topic projection."""

    if not isinstance(topic, Topic) or not topic.id or not topic.narrative:
        raise AgenticProjectionError("topic identity and narrative must be non-empty")
    if not isinstance(report, EvidenceCoverageReport):
        raise AgenticProjectionError("report must be EvidenceCoverageReport")
    if not isinstance(snapshot, TopicEvidenceSnapshot):
        raise AgenticProjectionError("snapshot must be TopicEvidenceSnapshot")
    if not isinstance(document_store, DocumentStore):
        raise AgenticProjectionError("document_store must be DocumentStore")
    if not isinstance(official_topics_sha256, str) or _SHA256.fullmatch(
        official_topics_sha256
    ) is None:
        raise AgenticProjectionError(
            "official_topics_sha256 must be a lowercase SHA-256 digest"
        )
    if snapshot.status != "complete":
        raise AgenticProjectionError("topic snapshot must be complete before projection")

    raw_documents = _require_unique(
        snapshot.documents, attribute="docid", label="source document"
    )
    raw_passages = _require_unique(
        snapshot.passages, attribute="passage_id", label="source passage"
    )
    documents = {
        key: value
        for key, value in raw_documents.items()
        if isinstance(value, SourceDocument)
    }
    passages = {
        key: value
        for key, value in raw_passages.items()
        if isinstance(value, SourcePassage)
    }
    if len(documents) != len(raw_documents) or len(passages) != len(raw_passages):
        raise AgenticProjectionError("snapshot source rows have unexpected types")

    _validate_candidate_topic_ids(topic.id, fused_candidates, searches)
    ranked = rank_agentic_documents(
        report,
        fused_candidates=fused_candidates,
        searches=searches,
    )
    if not ranked:
        raise AgenticProjectionError("topic has no grounded retrieval documents")
    unplaced = tuple(row.document_id for row in ranked if row.order_source == "unplaced")
    if unplaced:
        raise AgenticProjectionError(
            "grounded documents have no recorded retrieval position: "
            + ", ".join(unplaced)
        )

    full_text_candidates = _read_ranked_documents(
        ranked=ranked,
        documents=documents,
        document_store=document_store,
    )
    retrieval_rows = tuple(
        AgenticRetrievalRow(
            topic_id=topic.id,
            docid=row.document_id,
            rank=row.rank,
            score=row.score,
        )
        for row in ranked
    )
    retrieval_bytes = _serialize_retrieval_parts(
        topic_id=topic.id,
        narrative=topic.narrative,
        retrieval_rows=retrieval_rows,
        full_text_candidates=full_text_candidates,
    )
    retrieval_topic_sha256 = sha256(retrieval_bytes).hexdigest()

    document_texts = {row.docid: row.text for row in full_text_candidates}
    document_ranks = {row.docid: row.rank for row in full_text_candidates}
    groups, evidence, claims = _selected_generation_records(
        topic=topic,
        report=report,
        passages=passages,
        documents=documents,
        document_texts=document_texts,
        document_ranks=document_ranks,
    )
    generation_topic = GenerationTopic(
        topic_id=topic.id,
        narrative=topic.narrative,
        groups=groups,
        evidence=evidence,
        claim_hints=claims,
        source_receipts=TopicSourceReceipts(
            official_topics_sha256=official_topics_sha256,
            retrieval_topic_sha256=retrieval_topic_sha256,
        ),
    )
    citation_docids = set(generation_topic.citation_docids)
    retrieval_docids = {row.docid for row in retrieval_rows}
    full_text_docids = {row.docid for row in full_text_candidates}
    if not citation_docids <= retrieval_docids or retrieval_docids != full_text_docids:
        raise AgenticProjectionError(
            "generation citations are not closed over retrieval and full-text documents"
        )

    projection = AgenticTopicProjection(
        topic_id=topic.id,
        narrative=topic.narrative,
        retrieval_rows=retrieval_rows,
        full_text_candidates=full_text_candidates,
        generation_topic=generation_topic,
        retrieval_topic_sha256=retrieval_topic_sha256,
    )
    if serialize_agentic_retrieval_topic(projection) != retrieval_bytes:
        raise AgenticProjectionError("retrieval projection bytes changed during assembly")
    return projection
