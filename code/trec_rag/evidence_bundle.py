"""Canonical, arm-neutral evidence bundle records and validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import math
import re
from typing import Any


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _require_identifier(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{field_name} must be a stable identifier")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _require_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _CONTROL.search(value):
        raise ValueError(f"{field_name} must be text without control characters")


def _require_docid(value: str) -> None:
    if not isinstance(value, str) or not value or _CONTROL.search(value):
        raise ValueError("docid must be non-empty text without control characters")


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class BundleLane:
    lane_id: str
    lane_kind: str
    query_text: str
    parent_lane_id: str | None = None
    producer: str | None = None


@dataclass(frozen=True)
class BundleDocument:
    docid: str
    text: str
    text_sha256: str
    lane_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalEvent:
    event_id: str
    lane_id: str
    docid: str
    rank: int
    score: float
    retriever: str
    trace_ref_id: str | None = None


@dataclass(frozen=True)
class BundleSelection:
    selection_id: str
    source_lane_ids: tuple[str, ...]
    input_document_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    input_count: int
    output_count: int
    policy: str = "natural_union"


@dataclass(frozen=True)
class EvidenceSpan:
    evidence_id: str
    docid: str
    text: str
    text_sha256: str
    start_char: int
    end_char: int
    lane_ids: tuple[str, ...]
    selector: str
    source_document_sha256: str


@dataclass(frozen=True)
class BundleNugget:
    nugget_id: str
    text: str
    text_sha256: str
    evidence_ids: tuple[str, ...]
    nugget_kind: str
    subnarrative_id: str | None = None


@dataclass(frozen=True)
class TraceReference:
    trace_ref_id: str
    trace_kind: str
    trace_sha256: str


@dataclass(frozen=True)
class EvidenceBundle:
    topic_id: str
    lanes: tuple[BundleLane, ...]
    documents: tuple[BundleDocument, ...]
    retrieval_events: tuple[RetrievalEvent, ...]
    selections: tuple[BundleSelection, ...]
    evidence: tuple[EvidenceSpan, ...] = ()
    nuggets: tuple[BundleNugget, ...] = ()
    trace_refs: tuple[TraceReference, ...] = ()
    natural_document_count: int = 0

    def validate(self) -> None:
        _require_identifier(self.topic_id, field_name="topic_id")

        lane_map: dict[str, BundleLane] = {}
        for lane in self.lanes:
            _require_identifier(lane.lane_id, field_name="lane_id")
            _require_identifier(lane.lane_kind, field_name="lane_kind")
            _require_text(lane.query_text, field_name="query_text")
            if lane.parent_lane_id is not None:
                _require_identifier(lane.parent_lane_id, field_name="parent_lane_id")
            if lane.producer is not None:
                _require_text(lane.producer, field_name="producer")
            if lane.lane_id in lane_map:
                raise ValueError(f"duplicate lane_id: {lane.lane_id}")
            lane_map[lane.lane_id] = lane
        for lane in self.lanes:
            if lane.parent_lane_id is not None and lane.parent_lane_id not in lane_map:
                raise ValueError(f"unknown parent lane: {lane.parent_lane_id}")

        event_membership: dict[str, set[str]] = defaultdict(set)
        document_map: dict[str, BundleDocument] = {}
        for document in self.documents:
            _require_docid(document.docid)
            _require_text(document.text, field_name="document text")
            _require_sha256(document.text_sha256, field_name="document text_sha256")
            if _digest(document.text) != document.text_sha256:
                raise ValueError(f"document text_sha256 mismatch for {document.docid}")
            if document.docid in document_map:
                raise ValueError(f"duplicate docid: {document.docid}")
            lane_ids = _sorted_unique(document.lane_ids)
            if lane_ids != document.lane_ids:
                raise ValueError(f"document lane_ids must be sorted and unique for {document.docid}")
            for lane_id in document.lane_ids:
                if lane_id not in lane_map:
                    raise ValueError(f"unknown lane {lane_id} in document {document.docid}")
            document_map[document.docid] = document

        if self.natural_document_count != len(document_map):
            raise ValueError("natural union document count does not match unique documents")

        trace_map: dict[str, TraceReference] = {}
        for trace_ref in self.trace_refs:
            _require_identifier(trace_ref.trace_ref_id, field_name="trace_ref_id")
            _require_identifier(trace_ref.trace_kind, field_name="trace_kind")
            _require_sha256(trace_ref.trace_sha256, field_name="trace_sha256")
            if trace_ref.trace_ref_id in trace_map:
                raise ValueError(f"duplicate trace_ref_id: {trace_ref.trace_ref_id}")
            trace_map[trace_ref.trace_ref_id] = trace_ref

        event_map: dict[str, RetrievalEvent] = {}
        for event in self.retrieval_events:
            _require_identifier(event.event_id, field_name="event_id")
            _require_identifier(event.lane_id, field_name="event lane_id")
            _require_docid(event.docid)
            _require_text(event.retriever, field_name="retriever")
            if event.event_id in event_map:
                raise ValueError(f"duplicate event_id: {event.event_id}")
            if event.lane_id not in lane_map:
                raise ValueError(f"unknown lane for retrieval event: {event.lane_id}")
            if event.docid not in document_map:
                raise ValueError(f"unknown document for retrieval event: {event.docid}")
            if not isinstance(event.rank, int) or event.rank < 1:
                raise ValueError("retrieval event rank must be a positive integer")
            if not isinstance(event.score, (int, float)) or not math.isfinite(event.score):
                raise ValueError("retrieval event score must be finite")
            if event.trace_ref_id is not None and event.trace_ref_id not in trace_map:
                raise ValueError(f"unknown trace reference: {event.trace_ref_id}")
            event_map[event.event_id] = event
            event_membership[event.docid].add(event.lane_id)

        for docid, document in document_map.items():
            seen = event_membership.get(docid)
            if seen and tuple(sorted(seen)) != document.lane_ids:
                raise ValueError(f"document lane membership mismatch for {docid}")

        selection_map: dict[str, BundleSelection] = {}
        for selection in self.selections:
            _require_identifier(selection.selection_id, field_name="selection_id")
            _require_identifier(selection.policy, field_name="selection policy")
            if selection.selection_id in selection_map:
                raise ValueError(f"duplicate selection_id: {selection.selection_id}")
            source_lane_ids = _sorted_unique(selection.source_lane_ids)
            input_document_ids = _sorted_unique(selection.input_document_ids)
            document_ids = _sorted_unique(selection.document_ids)
            if source_lane_ids != selection.source_lane_ids:
                raise ValueError(f"selection source_lane_ids must be sorted and unique for {selection.selection_id}")
            if input_document_ids != selection.input_document_ids:
                raise ValueError(f"selection input_document_ids must be sorted and unique for {selection.selection_id}")
            if document_ids != selection.document_ids:
                raise ValueError(f"selection document_ids must be sorted and unique for {selection.selection_id}")
            for lane_id in selection.source_lane_ids:
                if lane_id not in lane_map:
                    raise ValueError(f"unknown source lane {lane_id} in selection {selection.selection_id}")
            for docid in selection.input_document_ids + selection.document_ids:
                if docid not in document_map:
                    raise ValueError(f"unknown document {docid} in selection {selection.selection_id}")
            if not set(selection.document_ids).issubset(selection.input_document_ids):
                raise ValueError(f"selection output must be a subset of the input union for {selection.selection_id}")
            if selection.input_count != len(selection.input_document_ids):
                raise ValueError(f"selection input count mismatch for {selection.selection_id}")
            if selection.output_count != len(selection.document_ids):
                raise ValueError(f"selection output count mismatch for {selection.selection_id}")
            expected_input = {
                event.docid
                for event in self.retrieval_events
                if event.lane_id in selection.source_lane_ids
            }
            if expected_input and expected_input != set(selection.input_document_ids):
                raise ValueError(f"selection natural union mismatch for {selection.selection_id}")
            selection_map[selection.selection_id] = selection

        evidence_map: dict[str, EvidenceSpan] = {}
        for span in self.evidence:
            _require_identifier(span.evidence_id, field_name="evidence_id")
            _require_docid(span.docid)
            _require_text(span.text, field_name="evidence text")
            _require_sha256(span.text_sha256, field_name="evidence text_sha256")
            _require_identifier(span.selector, field_name="evidence selector")
            _require_sha256(span.source_document_sha256, field_name="source_document_sha256")
            if span.evidence_id in evidence_map:
                raise ValueError(f"duplicate evidence_id: {span.evidence_id}")
            if span.docid not in document_map:
                raise ValueError(f"unknown document for evidence span: {span.docid}")
            if _digest(span.text) != span.text_sha256:
                raise ValueError(f"evidence text_sha256 mismatch for {span.evidence_id}")
            if span.lane_ids != _sorted_unique(span.lane_ids):
                raise ValueError(f"evidence lane_ids must be sorted and unique for {span.evidence_id}")
            document = document_map[span.docid]
            if span.source_document_sha256 != document.text_sha256:
                raise ValueError(f"evidence source document hash mismatch for {span.evidence_id}")
            for lane_id in span.lane_ids:
                if lane_id not in lane_map:
                    raise ValueError(f"unknown lane {lane_id} in evidence {span.evidence_id}")
                if lane_id not in document.lane_ids:
                    raise ValueError(f"evidence lane {lane_id} is not present in document membership for {span.evidence_id}")
            if (
                not isinstance(span.start_char, int)
                or not isinstance(span.end_char, int)
                or span.start_char < 0
                or span.end_char <= span.start_char
                or span.end_char > len(document.text)
            ):
                raise ValueError(f"evidence span coordinates are invalid for {span.evidence_id}")
            if document.text[span.start_char:span.end_char] != span.text:
                raise ValueError(f"evidence span does not resolve in document text for {span.evidence_id}")
            evidence_map[span.evidence_id] = span

        nugget_map: dict[str, BundleNugget] = {}
        for nugget in self.nuggets:
            _require_identifier(nugget.nugget_id, field_name="nugget_id")
            _require_text(nugget.text, field_name="nugget text")
            _require_sha256(nugget.text_sha256, field_name="nugget text_sha256")
            _require_identifier(nugget.nugget_kind, field_name="nugget_kind")
            if nugget.subnarrative_id is not None:
                _require_identifier(nugget.subnarrative_id, field_name="subnarrative_id")
            if nugget.nugget_id in nugget_map:
                raise ValueError(f"duplicate nugget_id: {nugget.nugget_id}")
            if _digest(nugget.text) != nugget.text_sha256:
                raise ValueError(f"nugget text_sha256 mismatch for {nugget.nugget_id}")
            if nugget.evidence_ids != _sorted_unique(nugget.evidence_ids):
                raise ValueError(f"nugget evidence_ids must be sorted and unique for {nugget.nugget_id}")
            for evidence_id in nugget.evidence_ids:
                if evidence_id not in evidence_map:
                    raise ValueError(f"unknown evidence for nugget {nugget.nugget_id}: {evidence_id}")
            nugget_map[nugget.nugget_id] = nugget

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "topic_id": self.topic_id,
            "natural_document_count": self.natural_document_count,
            "lanes": [
                {
                    "lane_id": lane.lane_id,
                    "lane_kind": lane.lane_kind,
                    "query_text": lane.query_text,
                    "parent_lane_id": lane.parent_lane_id,
                    "producer": lane.producer,
                }
                for lane in sorted(self.lanes, key=lambda row: row.lane_id)
            ],
            "documents": [
                {
                    "docid": document.docid,
                    "text": document.text,
                    "text_sha256": document.text_sha256,
                    "lane_ids": list(document.lane_ids),
                }
                for document in sorted(self.documents, key=lambda row: row.docid)
            ],
            "retrieval_events": [
                {
                    "event_id": event.event_id,
                    "lane_id": event.lane_id,
                    "docid": event.docid,
                    "rank": event.rank,
                    "score": event.score,
                    "retriever": event.retriever,
                    "trace_ref_id": event.trace_ref_id,
                }
                for event in sorted(
                    self.retrieval_events,
                    key=lambda row: (row.rank, row.lane_id, row.docid, row.event_id),
                )
            ],
            "selections": [
                {
                    "selection_id": selection.selection_id,
                    "source_lane_ids": list(selection.source_lane_ids),
                    "input_document_ids": list(selection.input_document_ids),
                    "document_ids": list(selection.document_ids),
                    "input_count": selection.input_count,
                    "output_count": selection.output_count,
                    "policy": selection.policy,
                }
                for selection in sorted(self.selections, key=lambda row: row.selection_id)
            ],
            "evidence": [
                {
                    "evidence_id": span.evidence_id,
                    "docid": span.docid,
                    "text": span.text,
                    "text_sha256": span.text_sha256,
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "lane_ids": list(span.lane_ids),
                    "selector": span.selector,
                    "source_document_sha256": span.source_document_sha256,
                }
                for span in sorted(self.evidence, key=lambda row: row.evidence_id)
            ],
            "nuggets": [
                {
                    "nugget_id": nugget.nugget_id,
                    "text": nugget.text,
                    "text_sha256": nugget.text_sha256,
                    "evidence_ids": list(nugget.evidence_ids),
                    "nugget_kind": nugget.nugget_kind,
                    "subnarrative_id": nugget.subnarrative_id,
                }
                for nugget in sorted(self.nuggets, key=lambda row: row.nugget_id)
            ],
            "trace_refs": [
                {
                    "trace_ref_id": trace.trace_ref_id,
                    "trace_kind": trace.trace_kind,
                    "trace_sha256": trace.trace_sha256,
                }
                for trace in sorted(self.trace_refs, key=lambda row: row.trace_ref_id)
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvidenceBundle:
        bundle = cls(
            topic_id=payload["topic_id"],
            natural_document_count=payload["natural_document_count"],
            lanes=tuple(
                BundleLane(
                    lane_id=row["lane_id"],
                    lane_kind=row["lane_kind"],
                    query_text=row["query_text"],
                    parent_lane_id=row.get("parent_lane_id"),
                    producer=row.get("producer"),
                )
                for row in payload["lanes"]
            ),
            documents=tuple(
                BundleDocument(
                    docid=row["docid"],
                    text=row["text"],
                    text_sha256=row["text_sha256"],
                    lane_ids=tuple(row["lane_ids"]),
                )
                for row in payload["documents"]
            ),
            retrieval_events=tuple(
                RetrievalEvent(
                    event_id=row["event_id"],
                    lane_id=row["lane_id"],
                    docid=row["docid"],
                    rank=row["rank"],
                    score=row["score"],
                    retriever=row["retriever"],
                    trace_ref_id=row.get("trace_ref_id"),
                )
                for row in payload["retrieval_events"]
            ),
            selections=tuple(
                BundleSelection(
                    selection_id=row["selection_id"],
                    source_lane_ids=tuple(row["source_lane_ids"]),
                    input_document_ids=tuple(row["input_document_ids"]),
                    document_ids=tuple(row["document_ids"]),
                    input_count=row["input_count"],
                    output_count=row["output_count"],
                    policy=row.get("policy", "natural_union"),
                )
                for row in payload["selections"]
            ),
            evidence=tuple(
                EvidenceSpan(
                    evidence_id=row["evidence_id"],
                    docid=row["docid"],
                    text=row["text"],
                    text_sha256=row["text_sha256"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    lane_ids=tuple(row["lane_ids"]),
                    selector=row["selector"],
                    source_document_sha256=row["source_document_sha256"],
                )
                for row in payload.get("evidence", ())
            ),
            nuggets=tuple(
                BundleNugget(
                    nugget_id=row["nugget_id"],
                    text=row["text"],
                    text_sha256=row["text_sha256"],
                    evidence_ids=tuple(row["evidence_ids"]),
                    nugget_kind=row["nugget_kind"],
                    subnarrative_id=row.get("subnarrative_id"),
                )
                for row in payload.get("nuggets", ())
            ),
            trace_refs=tuple(
                TraceReference(
                    trace_ref_id=row["trace_ref_id"],
                    trace_kind=row["trace_kind"],
                    trace_sha256=row["trace_sha256"],
                )
                for row in payload.get("trace_refs", ())
            ),
        )
        bundle.validate()
        return bundle

    @classmethod
    def from_retrieval_rows(
        cls,
        *,
        topic_id: str,
        rows: Iterable[Mapping[str, object]],
    ) -> EvidenceBundle:
        lane_rows: dict[str, BundleLane] = {}
        document_texts: dict[str, tuple[str, str]] = {}
        document_lanes: dict[str, set[str]] = defaultdict(set)
        events: list[RetrievalEvent] = []

        for index, row in enumerate(rows, start=1):
            lane_id = _expect_str(row, "lane_id")
            lane_kind = _expect_str(row, "lane_kind")
            query_text = _expect_str(row, "query_text")
            parent_lane_id = _expect_optional_str(row, "parent_lane_id")
            producer = _expect_optional_str(row, "producer")
            docid = _expect_str(row, "docid")
            text = _expect_str(row, "text")
            retriever = _expect_str(row, "retriever")
            rank = _expect_int(row, "rank")
            score = _expect_float(row, "score")
            text_sha256 = _expect_optional_str(row, "text_sha256") or _digest(text)

            lane = BundleLane(
                lane_id=lane_id,
                lane_kind=lane_kind,
                query_text=query_text,
                parent_lane_id=parent_lane_id,
                producer=producer,
            )
            prior_lane = lane_rows.get(lane_id)
            if prior_lane is not None and prior_lane != lane:
                raise ValueError(f"inconsistent lane metadata for {lane_id}")
            lane_rows[lane_id] = lane

            prior_document = document_texts.get(docid)
            current_document = (text, text_sha256)
            if prior_document is not None and prior_document != current_document:
                raise ValueError(f"inconsistent document text for {docid}")
            if text_sha256 != _digest(text):
                raise ValueError(f"text_sha256 mismatch for retrieval row {docid}")
            document_texts[docid] = current_document
            document_lanes[docid].add(lane_id)

            events.append(
                RetrievalEvent(
                    event_id=_expect_optional_str(row, "event_id") or f"retrieval.{index:06d}",
                    lane_id=lane_id,
                    docid=docid,
                    rank=rank,
                    score=score,
                    retriever=retriever,
                    trace_ref_id=_expect_optional_str(row, "trace_ref_id"),
                )
            )

        lane_ids = tuple(sorted(lane_rows))
        document_ids = tuple(sorted(document_texts))
        bundle = cls(
            topic_id=topic_id,
            lanes=tuple(lane_rows[lane_id] for lane_id in lane_ids),
            documents=tuple(
                BundleDocument(
                    docid=docid,
                    text=document_texts[docid][0],
                    text_sha256=document_texts[docid][1],
                    lane_ids=tuple(sorted(document_lanes[docid])),
                )
                for docid in document_ids
            ),
            retrieval_events=tuple(events),
            selections=(
                BundleSelection(
                    selection_id="natural_union",
                    source_lane_ids=lane_ids,
                    input_document_ids=document_ids,
                    document_ids=document_ids,
                    input_count=len(document_ids),
                    output_count=len(document_ids),
                ),
            ),
            natural_document_count=len(document_ids),
        )
        bundle.validate()
        return bundle


def _expect_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    return value


def _expect_optional_str(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text when present")
    return value


def _expect_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _expect_float(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number
