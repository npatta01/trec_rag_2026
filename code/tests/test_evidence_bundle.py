from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from trec_rag.evidence_bundle import (
    BundleNugget,
    EvidenceBundle,
    EvidenceSpan,
    TraceReference,
)


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _retrieval_rows() -> list[dict[str, object]]:
    return [
        {
            "lane_id": "narrative",
            "lane_kind": "narrative",
            "query_text": "Explain the documented outcome.",
            "docid": "doc-shared",
            "text": "Alpha Beta",
            "rank": 1,
            "score": 12.5,
            "retriever": "bm25",
        },
        {
            "lane_id": "agentic.a",
            "lane_kind": "agentic",
            "query_text": "Find independent corroboration.",
            "parent_lane_id": "narrative",
            "producer": "agent-step-v1",
            "docid": "doc-shared",
            "text": "Alpha Beta",
            "rank": 2,
            "score": 11.0,
            "retriever": "bm25",
        },
        {
            "lane_id": "agentic.a",
            "lane_kind": "agentic",
            "query_text": "Find independent corroboration.",
            "parent_lane_id": "narrative",
            "producer": "agent-step-v1",
            "docid": "doc-unique",
            "text": "Gamma Delta",
            "rank": 3,
            "score": 8.75,
            "retriever": "bm25",
        },
    ]


def _bundle() -> EvidenceBundle:
    return EvidenceBundle.from_retrieval_rows(topic_id="224", rows=_retrieval_rows())


def test_from_retrieval_rows_preserves_multi_lane_document_membership() -> None:
    bundle = _bundle()

    shared = next(document for document in bundle.documents if document.docid == "doc-shared")

    assert shared.lane_ids == ("agentic.a", "narrative")
    assert [
        (event.lane_id, event.docid, event.rank)
        for event in bundle.retrieval_events
        if event.docid == "doc-shared"
    ] == [
        ("narrative", "doc-shared", 1),
        ("agentic.a", "doc-shared", 2),
    ]


def test_validation_checks_natural_union_counts() -> None:
    bundle = _bundle()

    assert bundle.natural_document_count == 2
    assert bundle.selections[0].input_count == 2
    assert bundle.selections[0].output_count == 2

    broken = replace(bundle, natural_document_count=3)
    with pytest.raises(ValueError, match="natural union"):
        broken.validate()


def test_validation_rejects_document_lane_membership_without_retrieval_event() -> None:
    bundle = _bundle()
    shared = next(document for document in bundle.documents if document.docid == "doc-shared")
    broken_document = replace(shared, lane_ids=("agentic.a", "extra.lane", "narrative"))
    broken = replace(
        bundle,
        lanes=(
            *bundle.lanes,
            replace(bundle.lanes[0], lane_id="extra.lane", lane_kind="agentic"),
        ),
        documents=(broken_document, bundle.documents[1]),
    )

    with pytest.raises(ValueError, match="document lane membership mismatch"):
        broken.validate()


def test_validation_rejects_document_without_any_retrieval_event() -> None:
    bundle = _bundle()
    orphan = replace(
        bundle.documents[0],
        docid="doc-orphan",
        text="Orphan text",
        text_sha256=_digest("Orphan text"),
        lane_ids=("narrative",),
    )
    broken = replace(
        bundle,
        documents=(bundle.documents[0], bundle.documents[1], orphan),
        natural_document_count=3,
    )

    with pytest.raises(ValueError, match="document lane membership mismatch"):
        broken.validate()


def test_validation_rejects_selection_input_without_retrieval_event() -> None:
    bundle = _bundle()
    selection = bundle.selections[0]
    broken = replace(
        bundle,
        selections=(
            replace(
                selection,
                source_lane_ids=("narrative",),
                input_document_ids=("doc-shared", "doc-unique"),
                input_count=2,
            ),
        ),
    )

    with pytest.raises(ValueError, match="selection natural union mismatch"):
        broken.validate()


def test_validation_requires_exact_evidence_spans_to_resolve_in_document_text() -> None:
    bundle = _bundle()
    shared = next(document for document in bundle.documents if document.docid == "doc-shared")
    evidence = EvidenceSpan(
        evidence_id="evidence.beta",
        docid="doc-shared",
        text="Beta",
        text_sha256=_digest("Beta"),
        start_char=6,
        end_char=10,
        lane_ids=("agentic.a", "narrative"),
        selector="exact_sentence",
        source_document_sha256=shared.text_sha256,
    )

    replace(bundle, evidence=(evidence,)).validate()

    broken = replace(bundle, evidence=(replace(evidence, end_char=9),))
    with pytest.raises(ValueError, match="evidence span"):
        broken.validate()


def test_validation_requires_nugget_support_to_reference_known_evidence() -> None:
    bundle = _bundle()
    shared = next(document for document in bundle.documents if document.docid == "doc-shared")
    evidence = EvidenceSpan(
        evidence_id="evidence.beta",
        docid="doc-shared",
        text="Beta",
        text_sha256=_digest("Beta"),
        start_char=6,
        end_char=10,
        lane_ids=("narrative",),
        selector="exact_sentence",
        source_document_sha256=shared.text_sha256,
    )
    nugget = BundleNugget(
        nugget_id="nugget.beta",
        text="Beta is present in the shared document.",
        text_sha256=_digest("Beta is present in the shared document."),
        evidence_ids=("evidence.missing",),
        nugget_kind="direct",
    )

    with pytest.raises(ValueError, match="unknown evidence"):
        replace(bundle, evidence=(evidence,), nuggets=(nugget,)).validate()


def test_serialization_round_trip_is_deterministic_and_json_compatible() -> None:
    bundle = _bundle()
    shared = next(document for document in bundle.documents if document.docid == "doc-shared")
    trace = TraceReference(
        trace_ref_id="trace.001",
        trace_kind="agent-run",
        trace_sha256="a" * 64,
    )
    evidence = EvidenceSpan(
        evidence_id="evidence.beta",
        docid="doc-shared",
        text="Beta",
        text_sha256=_digest("Beta"),
        start_char=6,
        end_char=10,
        lane_ids=("agentic.a", "narrative"),
        selector="exact_sentence",
        source_document_sha256=shared.text_sha256,
    )
    nugget = BundleNugget(
        nugget_id="nugget.beta",
        text="Beta is present in the shared document.",
        text_sha256=_digest("Beta is present in the shared document."),
        evidence_ids=("evidence.beta",),
        nugget_kind="direct",
    )
    traced_events = (
        replace(bundle.retrieval_events[0], trace_ref_id="trace.001"),
        *bundle.retrieval_events[1:],
    )
    enriched = replace(
        bundle,
        retrieval_events=traced_events,
        evidence=(evidence,),
        nuggets=(nugget,),
        trace_refs=(trace,),
    )

    payload = enriched.to_dict()
    round_tripped = EvidenceBundle.from_dict(payload)

    assert json.loads(json.dumps(payload)) == payload
    assert round_tripped == enriched
    assert round_tripped.to_dict() == payload


def test_round_trip_canonicalizes_unsorted_retrieval_events() -> None:
    rows = [
        {
            "lane_id": "narrative",
            "lane_kind": "narrative",
            "query_text": "Explain the documented outcome.",
            "docid": "doc-b",
            "text": "Second",
            "rank": 2,
            "score": 10.0,
            "retriever": "bm25",
        },
        {
            "lane_id": "narrative",
            "lane_kind": "narrative",
            "query_text": "Explain the documented outcome.",
            "docid": "doc-a",
            "text": "First",
            "rank": 1,
            "score": 11.0,
            "retriever": "bm25",
        },
    ]

    bundle = EvidenceBundle.from_retrieval_rows(topic_id="224", rows=rows)
    payload = bundle.to_dict()
    round_tripped = EvidenceBundle.from_dict(payload)

    assert [(event.rank, event.docid) for event in bundle.retrieval_events] == [
        (1, "doc-a"),
        (2, "doc-b"),
    ]
    assert round_tripped == bundle
    assert round_tripped.to_dict() == payload
