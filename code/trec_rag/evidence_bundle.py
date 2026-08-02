"""Canonical, arm-neutral evidence bundle records and validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import zipfile


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_TEXT_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BUNDLE_SCHEMA_VERSION = "evidence_bundle_v1"
_LANE_KINDS = frozenset({"narrative", "subnarrative", "agentic"})


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _require_identifier(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{field_name} must be a stable identifier")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _require_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _UNSAFE_TEXT_CONTROL.search(value):
        raise ValueError(f"{field_name} must be text without unsafe control characters")


def _require_docid(value: str) -> None:
    if not isinstance(value, str) or not value or _CONTROL.search(value):
        raise ValueError("docid must be non-empty text without control characters")


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _event_sort_key(event: RetrievalEvent) -> tuple[int, str, str, str]:
    return (event.rank, event.lane_id, event.docid, event.event_id)


@dataclass(frozen=True)
class BundleLane:
    lane_id: str
    lane_kind: str
    query_text: str
    query_text_sha256: str
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
class BundleSelectionMember:
    docid: str
    included: bool
    output_rank: int | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class BundleSelection:
    selection_id: str
    source_lane_ids: tuple[str, ...]
    members: tuple[BundleSelectionMember, ...]
    policy: str = "natural_union"

    @property
    def input_document_ids(self) -> tuple[str, ...]:
        return tuple(member.docid for member in self.members)

    @property
    def document_ids(self) -> tuple[str, ...]:
        included = tuple(member for member in self.members if member.included)
        if all(member.output_rank is None for member in included):
            return tuple(member.docid for member in included)
        return tuple(
            member.docid
            for member in sorted(included, key=lambda member: member.output_rank or 0)
        )

    @property
    def input_count(self) -> int:
        return len(self.members)

    @property
    def output_count(self) -> int:
        return sum(member.included for member in self.members)


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
            if lane.lane_kind not in _LANE_KINDS:
                raise ValueError(f"unsupported lane_kind: {lane.lane_kind}")
            _require_text(lane.query_text, field_name="query_text")
            _require_sha256(lane.query_text_sha256, field_name="query_text_sha256")
            if _digest(lane.query_text) != lane.query_text_sha256:
                raise ValueError(f"query_text_sha256 mismatch for {lane.lane_id}")
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

        if type(self.natural_document_count) is not int:
            raise ValueError("natural_document_count must be an integer")
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
            if type(event.rank) is not int or event.rank < 1:
                raise ValueError("retrieval event rank must be a positive integer")
            if type(event.score) not in (int, float) or not math.isfinite(event.score):
                raise ValueError("retrieval event score must be finite")
            if event.trace_ref_id is not None and event.trace_ref_id not in trace_map:
                raise ValueError(f"unknown trace reference: {event.trace_ref_id}")
            event_map[event.event_id] = event
            event_membership[event.docid].add(event.lane_id)

        for docid, document in document_map.items():
            seen = tuple(sorted(event_membership.get(docid, set())))
            if seen != document.lane_ids:
                raise ValueError(f"document lane membership mismatch for {docid}")

        selection_map: dict[str, BundleSelection] = {}
        for selection in self.selections:
            _require_identifier(selection.selection_id, field_name="selection_id")
            _require_identifier(selection.policy, field_name="selection policy")
            if selection.selection_id in selection_map:
                raise ValueError(f"duplicate selection_id: {selection.selection_id}")
            source_lane_ids = _sorted_unique(selection.source_lane_ids)
            if source_lane_ids != selection.source_lane_ids:
                raise ValueError(f"selection source_lane_ids must be sorted and unique for {selection.selection_id}")
            for lane_id in selection.source_lane_ids:
                if lane_id not in lane_map:
                    raise ValueError(f"unknown source lane {lane_id} in selection {selection.selection_id}")
            member_docids = tuple(member.docid for member in selection.members)
            if member_docids != tuple(sorted(member_docids)) or len(set(member_docids)) != len(member_docids):
                raise ValueError(f"selection members must have sorted unique docids for {selection.selection_id}")
            output_ranks: list[int] = []
            for member in selection.members:
                _require_docid(member.docid)
                if member.docid not in document_map:
                    raise ValueError(f"unknown document {member.docid} in selection {selection.selection_id}")
                if type(member.included) is not bool:
                    raise ValueError(f"selection member included must be Boolean for {selection.selection_id}")
                if member.included:
                    if member.rejection_reason is not None:
                        raise ValueError(f"included selection member cannot have a rejection reason for {selection.selection_id}")
                    if member.output_rank is not None:
                        if type(member.output_rank) is not int or member.output_rank < 1:
                            raise ValueError(f"selection member output_rank must be a positive integer for {selection.selection_id}")
                        output_ranks.append(member.output_rank)
                else:
                    if member.output_rank is not None:
                        raise ValueError(f"rejected selection member cannot have an output rank for {selection.selection_id}")
                    if not isinstance(member.rejection_reason, str) or not member.rejection_reason:
                        raise ValueError(f"rejected selection member requires a rejection reason for {selection.selection_id}")
                    _require_text(member.rejection_reason, field_name="selection rejection_reason")
            if selection.policy == "natural_union":
                if any(not member.included or member.output_rank is not None for member in selection.members):
                    raise ValueError("natural_union selection must include every member without output ranks")
            elif sorted(output_ranks) != list(range(1, selection.output_count + 1)):
                raise ValueError(f"selection output ranks must be contiguous for {selection.selection_id}")
            expected_input = {
                event.docid
                for event in self.retrieval_events
                if event.lane_id in selection.source_lane_ids
            }
            if expected_input != set(selection.input_document_ids):
                raise ValueError(f"selection natural union mismatch for {selection.selection_id}")
            selection_map[selection.selection_id] = selection

        natural_union_selections = tuple(
            selection
            for selection in self.selections
            if selection.selection_id == "natural_union"
            and selection.policy == "natural_union"
        )
        if len(natural_union_selections) != 1:
            raise ValueError(
                "bundle requires exactly one natural_union selection"
            )

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
            if not span.lane_ids:
                raise ValueError(f"evidence {span.evidence_id} requires at least one lane")
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
                type(span.start_char) is not int
                or type(span.end_char) is not int
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
                subnarrative_lane = lane_map.get(nugget.subnarrative_id)
                if subnarrative_lane is None or subnarrative_lane.lane_kind != "subnarrative":
                    raise ValueError(f"nugget {nugget.nugget_id} must reference a known subnarrative lane")
            if nugget.nugget_id in nugget_map:
                raise ValueError(f"duplicate nugget_id: {nugget.nugget_id}")
            if _digest(nugget.text) != nugget.text_sha256:
                raise ValueError(f"nugget text_sha256 mismatch for {nugget.nugget_id}")
            if not nugget.evidence_ids:
                raise ValueError(f"nugget {nugget.nugget_id} requires at least one evidence ID")
            if nugget.evidence_ids != _sorted_unique(nugget.evidence_ids):
                raise ValueError(f"nugget evidence_ids must be sorted and unique for {nugget.nugget_id}")
            for evidence_id in nugget.evidence_ids:
                if evidence_id not in evidence_map:
                    raise ValueError(f"unknown evidence for nugget {nugget.nugget_id}: {evidence_id}")
            nugget_map[nugget.nugget_id] = nugget

    def to_trec_run(self, *, selection_id: str) -> tuple[tuple[str, str, int, int, str], ...]:
        selected = self._selected_documents(selection_id)
        total = len(selected)
        return tuple(
            (self.topic_id, document.docid, rank, total - rank + 1, selection_id)
            for rank, document in enumerate(selected, start=1)
        )

    def to_document_records(self, *, selection_id: str) -> tuple[dict[str, object], ...]:
        selected = self._selected_documents(selection_id)
        query = self._topic_query()
        total = len(selected)
        return (
            {
                "query": {
                    "qid": self.topic_id,
                    "selection_id": selection_id,
                    "text": query.query_text,
                    "text_sha256": query.query_text_sha256,
                },
                "candidates": [
                    {
                        "docid": document.docid,
                        "doc": document.text,
                        "rank": rank,
                        "score": total - rank + 1,
                        "lane_ids": list(document.lane_ids),
                        "text_sha256": document.text_sha256,
                    }
                    for rank, document in enumerate(selected, start=1)
                ],
            },
        )

    def to_fixed_rag_context(self, *, selection_id: str) -> tuple[dict[str, object], ...]:
        selection = self._selection(selection_id)
        selected = self._selected_documents(selection_id)
        query = self._topic_query()
        selected_source_lanes = set(selection.source_lane_ids)
        evidence_by_doc: dict[str, list[EvidenceSpan]] = defaultdict(list)
        for span in self.evidence:
            if selected_source_lanes.intersection(span.lane_ids):
                evidence_by_doc[span.docid].append(span)
        for spans in evidence_by_doc.values():
            spans.sort(key=lambda row: (row.start_char, row.end_char, row.evidence_id))

        nugget_by_id = {nugget.nugget_id: nugget for nugget in self.nuggets}
        evidence_to_nuggets: dict[str, list[str]] = defaultdict(list)
        for nugget in self.nuggets:
            for evidence_id in nugget.evidence_ids:
                evidence_to_nuggets[evidence_id].append(nugget.nugget_id)

        context: list[dict[str, object]] = []
        total = len(selected)
        for rank, document in enumerate(selected, start=1):
            spans = evidence_by_doc.get(document.docid, [])
            selected_evidence_ids = {span.evidence_id for span in spans}
            seen_nuggets: set[str] = set()
            ordered_nuggets: list[tuple[BundleNugget, tuple[str, ...]]] = []
            for span in spans:
                for nugget_id in evidence_to_nuggets.get(span.evidence_id, ()):
                    if nugget_id in seen_nuggets:
                        continue
                    nugget = nugget_by_id[nugget_id]
                    projected_evidence_ids = tuple(
                        evidence_id
                        for evidence_id in nugget.evidence_ids
                        if evidence_id in selected_evidence_ids
                    )
                    if not projected_evidence_ids:
                        continue
                    seen_nuggets.add(nugget_id)
                    ordered_nuggets.append((nugget, projected_evidence_ids))
            ordered_nuggets.sort(
                key=lambda row: (
                    row[1][0] if row[1] else "",
                    row[0].nugget_id,
                )
            )
            context.append(
                {
                    "topic_id": self.topic_id,
                    "selection_id": selection_id,
                    "query": {
                        "qid": self.topic_id,
                        "text": query.query_text,
                        "text_sha256": query.query_text_sha256,
                    },
                    "docid": document.docid,
                    "rank": rank,
                    "score": total - rank + 1,
                    "lane_ids": list(document.lane_ids),
                    "document": {
                        "text": document.text,
                        "text_sha256": document.text_sha256,
                    },
                    "evidence": [
                        {
                            "evidence_id": span.evidence_id,
                            "text": span.text,
                            "text_sha256": span.text_sha256,
                            "start_char": span.start_char,
                            "end_char": span.end_char,
                            "lane_ids": list(span.lane_ids),
                            "selector": span.selector,
                        }
                        for span in spans
                    ],
                    "nuggets": [
                        {
                            "nugget_id": nugget.nugget_id,
                            "text": nugget.text,
                            "text_sha256": nugget.text_sha256,
                            "nugget_kind": nugget.nugget_kind,
                            "subnarrative_id": nugget.subnarrative_id,
                            "evidence_ids": list(projected_evidence_ids),
                        }
                        for nugget, projected_evidence_ids in ordered_nuggets
                    ],
                }
            )
        return tuple(context)

    def write_fixed_rag_inputs(
        self,
        output_dir: Path,
        *,
        selection_id: str,
    ) -> dict[str, Path]:
        outputs = write_fixed_rag_package(
            (self,), output_dir, selection_id=selection_id
        )
        return {key: value for key, value in outputs.items() if key != "queries"}

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": _BUNDLE_SCHEMA_VERSION,
            "topic_id": self.topic_id,
            "natural_document_count": self.natural_document_count,
            "lanes": [
                {
                    "lane_id": lane.lane_id,
                    "lane_kind": lane.lane_kind,
                    "query_text": lane.query_text,
                    "query_text_sha256": lane.query_text_sha256,
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
                    "members": [
                        {
                            "docid": member.docid,
                            "included": member.included,
                            "output_rank": member.output_rank,
                            "rejection_reason": member.rejection_reason,
                        }
                        for member in selection.members
                    ],
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
        _require_payload_keys(
            payload,
            {
                "schema_version",
                "topic_id",
                "natural_document_count",
                "lanes",
                "documents",
                "retrieval_events",
                "selections",
                "evidence",
                "nuggets",
                "trace_refs",
            },
            owner="evidence bundle payload",
        )
        if payload["schema_version"] != _BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported evidence bundle schema version")
        lane_rows = _payload_rows(payload, "lanes")
        document_rows = _payload_rows(payload, "documents")
        retrieval_rows = _payload_rows(payload, "retrieval_events")
        selection_rows = _payload_rows(payload, "selections")
        evidence_rows = _payload_rows(payload, "evidence")
        nugget_rows = _payload_rows(payload, "nuggets")
        trace_rows = _payload_rows(payload, "trace_refs")
        row_key_contracts = (
            (
                lane_rows,
                {
                    "lane_id",
                    "lane_kind",
                    "query_text",
                    "query_text_sha256",
                    "parent_lane_id",
                    "producer",
                },
                "lane payload",
            ),
            (
                document_rows,
                {"docid", "text", "text_sha256", "lane_ids"},
                "document payload",
            ),
            (
                retrieval_rows,
                {
                    "event_id",
                    "lane_id",
                    "docid",
                    "rank",
                    "score",
                    "retriever",
                    "trace_ref_id",
                },
                "retrieval event payload",
            ),
            (
                evidence_rows,
                {
                    "evidence_id",
                    "docid",
                    "text",
                    "text_sha256",
                    "start_char",
                    "end_char",
                    "lane_ids",
                    "selector",
                    "source_document_sha256",
                },
                "evidence payload",
            ),
            (
                nugget_rows,
                {
                    "nugget_id",
                    "text",
                    "text_sha256",
                    "evidence_ids",
                    "nugget_kind",
                    "subnarrative_id",
                },
                "nugget payload",
            ),
            (
                trace_rows,
                {"trace_ref_id", "trace_kind", "trace_sha256"},
                "trace reference payload",
            ),
        )
        for rows, expected_keys, owner in row_key_contracts:
            for row in rows:
                _require_payload_keys(row, expected_keys, owner=owner)
        bundle = cls(
            topic_id=_payload_text(payload["topic_id"], "topic_id"),
            natural_document_count=_payload_int(
                payload["natural_document_count"], "natural_document_count"
            ),
            lanes=tuple(
                BundleLane(
                    lane_id=_payload_text(row["lane_id"], "lane_id"),
                    lane_kind=_payload_text(row["lane_kind"], "lane_kind"),
                    query_text=_payload_text(row["query_text"], "query_text"),
                    query_text_sha256=_payload_text(
                        row["query_text_sha256"], "query_text_sha256"
                    ),
                    parent_lane_id=_payload_optional_text(
                        row["parent_lane_id"], "parent_lane_id"
                    ),
                    producer=_payload_optional_text(row["producer"], "producer"),
                )
                for row in lane_rows
            ),
            documents=tuple(
                BundleDocument(
                    docid=_payload_text(row["docid"], "docid"),
                    text=_payload_text(row["text"], "document text"),
                    text_sha256=_payload_text(row["text_sha256"], "text_sha256"),
                    lane_ids=_payload_text_tuple(row["lane_ids"], "lane_ids"),
                )
                for row in document_rows
            ),
            retrieval_events=tuple(
                RetrievalEvent(
                    event_id=_payload_text(row["event_id"], "event_id"),
                    lane_id=_payload_text(row["lane_id"], "lane_id"),
                    docid=_payload_text(row["docid"], "docid"),
                    rank=_payload_int(row["rank"], "rank"),
                    score=_payload_number(row["score"], "score"),
                    retriever=_payload_text(row["retriever"], "retriever"),
                    trace_ref_id=_payload_optional_text(
                        row["trace_ref_id"], "trace_ref_id"
                    ),
                )
                for row in sorted(
                    retrieval_rows,
                    key=lambda row: (
                        _payload_int(row["rank"], "rank"),
                        _payload_text(row["lane_id"], "lane_id"),
                        _payload_text(row["docid"], "docid"),
                        _payload_text(row["event_id"], "event_id"),
                    ),
                )
            ),
            selections=tuple(_selection_from_payload(row) for row in selection_rows),
            evidence=tuple(
                EvidenceSpan(
                    evidence_id=_payload_text(row["evidence_id"], "evidence_id"),
                    docid=_payload_text(row["docid"], "docid"),
                    text=_payload_text(row["text"], "evidence text"),
                    text_sha256=_payload_text(row["text_sha256"], "text_sha256"),
                    start_char=_payload_int(row["start_char"], "start_char"),
                    end_char=_payload_int(row["end_char"], "end_char"),
                    lane_ids=_payload_text_tuple(row["lane_ids"], "lane_ids"),
                    selector=_payload_text(row["selector"], "selector"),
                    source_document_sha256=_payload_text(
                        row["source_document_sha256"], "source_document_sha256"
                    ),
                )
                for row in evidence_rows
            ),
            nuggets=tuple(
                BundleNugget(
                    nugget_id=_payload_text(row["nugget_id"], "nugget_id"),
                    text=_payload_text(row["text"], "nugget text"),
                    text_sha256=_payload_text(row["text_sha256"], "text_sha256"),
                    evidence_ids=_payload_text_tuple(row["evidence_ids"], "evidence_ids"),
                    nugget_kind=_payload_text(row["nugget_kind"], "nugget_kind"),
                    subnarrative_id=_payload_optional_text(
                        row["subnarrative_id"], "subnarrative_id"
                    ),
                )
                for row in nugget_rows
            ),
            trace_refs=tuple(
                TraceReference(
                    trace_ref_id=_payload_text(row["trace_ref_id"], "trace_ref_id"),
                    trace_kind=_payload_text(row["trace_kind"], "trace_kind"),
                    trace_sha256=_payload_text(row["trace_sha256"], "trace_sha256"),
                )
                for row in trace_rows
            ),
        )
        bundle.validate()
        return bundle

    def _selection(self, selection_id: str) -> BundleSelection:
        self.validate()
        for selection in self.selections:
            if selection.selection_id == selection_id:
                return selection
        raise ValueError(f"unknown selection_id: {selection_id}")

    def _selected_documents(self, selection_id: str) -> tuple[BundleDocument, ...]:
        selection = self._selection(selection_id)
        if selection.policy == "natural_union" or any(
            member.included and member.output_rank is None for member in selection.members
        ):
            raise ValueError(f"cannot rank non-ranked selection: {selection_id}")
        documents_by_id = {document.docid: document for document in self.documents}
        return tuple(documents_by_id[docid] for docid in selection.document_ids)

    def _topic_query(self) -> BundleLane:
        narrative_lanes = tuple(lane for lane in self.lanes if lane.lane_kind == "narrative")
        if len(narrative_lanes) != 1:
            raise ValueError(f"topic {self.topic_id} must have exactly one narrative lane")
        return narrative_lanes[0]

    @classmethod
    def from_retrieval_rows(
        cls,
        *,
        topic_id: str,
        rows: Iterable[Mapping[str, object]],
        trace_refs: Iterable[TraceReference] = (),
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
                query_text_sha256=_expect_optional_str(row, "query_text_sha256")
                or _digest(query_text),
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
            retrieval_events=tuple(sorted(events, key=_event_sort_key)),
            selections=(
                BundleSelection(
                    selection_id="natural_union",
                    source_lane_ids=lane_ids,
                    members=tuple(
                        BundleSelectionMember(docid=docid, included=True)
                        for docid in document_ids
                    ),
                ),
            ),
            trace_refs=tuple(trace_refs),
            natural_document_count=len(document_ids),
        )
        bundle.validate()
        return bundle


def _require_payload_keys(
    payload: Mapping[str, Any], expected: set[str], *, owner: str
) -> None:
    if not isinstance(payload, Mapping) or not all(
        isinstance(key, str) for key in payload
    ):
        raise ValueError(f"{owner} must be an object with text keys")
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{owner} has invalid payload keys; missing={missing}, unexpected={unexpected}"
        )


def _payload_rows(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = payload[key]
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"{key} must be a list of objects")
    return value


def _payload_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    return value


def _payload_optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _payload_text(value, field_name)


def _payload_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return value


def _payload_number(value: object, field_name: str) -> int | float:
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be numeric")
    return value


def _payload_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be Boolean")
    return value


def _payload_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of text values")
    return tuple(value)


def _selection_from_payload(row: Mapping[str, Any]) -> BundleSelection:
    _require_payload_keys(
        row,
        {
            "selection_id",
            "source_lane_ids",
            "input_document_ids",
            "document_ids",
            "input_count",
            "output_count",
            "policy",
            "members",
        },
        owner="selection payload",
    )
    member_rows = row["members"]
    if not isinstance(member_rows, list) or not all(
        isinstance(member, Mapping) for member in member_rows
    ):
        raise ValueError("selection members must be a list of objects")
    members: list[BundleSelectionMember] = []
    for member in member_rows:
        _require_payload_keys(
            member,
            {"docid", "included", "output_rank", "rejection_reason"},
            owner="selection member payload",
        )
        output_rank = member["output_rank"]
        members.append(
            BundleSelectionMember(
                docid=_payload_text(member["docid"], "selection member docid"),
                included=_payload_bool(member["included"], "selection member included"),
                output_rank=None
                if output_rank is None
                else _payload_int(output_rank, "selection member output_rank"),
                rejection_reason=_payload_optional_text(
                    member["rejection_reason"], "selection member rejection_reason"
                ),
            )
        )
    selection = BundleSelection(
        selection_id=_payload_text(row["selection_id"], "selection_id"),
        source_lane_ids=_payload_text_tuple(row["source_lane_ids"], "source_lane_ids"),
        members=tuple(members),
        policy=_payload_text(row["policy"], "selection policy"),
    )
    input_document_ids = _payload_text_tuple(
        row["input_document_ids"], "input_document_ids"
    )
    document_ids = _payload_text_tuple(row["document_ids"], "document_ids")
    input_count = _payload_int(row["input_count"], "input_count")
    output_count = _payload_int(row["output_count"], "output_count")
    if input_document_ids != selection.input_document_ids:
        raise ValueError("selection input_document_ids do not match members")
    if document_ids != selection.document_ids:
        raise ValueError("selection document_ids do not match members")
    if input_count != selection.input_count:
        raise ValueError("selection input_count does not match members")
    if output_count != selection.output_count:
        raise ValueError("selection output_count does not match members")
    return selection


def write_fixed_rag_package(
    bundles: Sequence[EvidenceBundle],
    output_dir: Path,
    *,
    selection_id: str,
) -> dict[str, Path]:
    """Write one deterministic, query-aware fixed-RAG package for many topics."""
    ordered_bundles = tuple(sorted(bundles, key=lambda bundle: bundle.topic_id))
    if not ordered_bundles:
        raise ValueError("at least one evidence bundle is required")
    topic_ids = tuple(bundle.topic_id for bundle in ordered_bundles)
    if len(set(topic_ids)) != len(topic_ids):
        raise ValueError("fixed-RAG package topic IDs must be unique")

    query_rows: list[tuple[str, str]] = []
    trec_rows: list[tuple[str, str, int, int, str]] = []
    document_records: list[Mapping[str, object]] = []
    context_records: list[Mapping[str, object]] = []
    for bundle in ordered_bundles:
        bundle.validate()
        query = bundle._topic_query()
        query_rows.append((bundle.topic_id, " ".join(query.query_text.split())))
        trec_rows.extend(bundle.to_trec_run(selection_id=selection_id))
        document_records.extend(bundle.to_document_records(selection_id=selection_id))
        context_records.extend(bundle.to_fixed_rag_context(selection_id=selection_id))

    output_dir = Path(output_dir)
    outputs = {
        "queries": output_dir / "trec_rag_2026_queries.tsv",
        "run": output_dir / "r_output_trec_rag_2026.tsv",
        "documents_jsonl": output_dir / "retrieval_with_text.jsonl",
        "documents_zip": output_dir / "retrieval_with_text.jsonl.zip",
        "context_jsonl": output_dir / "fixed_rag_context.jsonl",
    }
    query_body = "".join(f"{topic_id}\t{text}\n" for topic_id, text in query_rows).encode("utf-8")
    trec_body = _trec_bytes(trec_rows)
    documents_body = _jsonl_bytes(document_records)
    context_body = _jsonl_bytes(context_records)

    from trec_rag.retrieval_export import _validate_trec_bytes

    _validate_trec_bytes(trec_body, trec_rows)
    _atomic_write(outputs["queries"], query_body)
    _atomic_write(outputs["run"], trec_body)
    _atomic_write(outputs["documents_jsonl"], documents_body)
    _atomic_write(outputs["documents_zip"], _deterministic_zip(documents_body))
    _atomic_write(outputs["context_jsonl"], context_body)
    return outputs


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


def _trec_bytes(rows: Sequence[tuple[str, str, int, int, str]]) -> bytes:
    return "".join(
        f"{topic_id} Q0 {docid} {rank} {score} {run_id}\n"
        for topic_id, docid, rank, score, run_id in rows
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


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


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
