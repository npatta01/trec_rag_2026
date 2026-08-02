from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import pytest

from trec_rag import evidence_bundle as evidence_bundle_module
from trec_rag.evidence_bundle import (
    BundleLane,
    BundleSelection,
    BundleSelectionMember,
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


def test_from_retrieval_rows_preserves_seeded_empty_lanes() -> None:
    query = "A generated subnarrative with no hits."
    empty_lane = BundleLane(
        lane_id="sub.empty",
        lane_kind="subnarrative",
        query_text=query,
        query_text_sha256=_digest(query),
        parent_lane_id="narrative",
        producer="retriever-v1",
    )

    bundle = EvidenceBundle.from_retrieval_rows(
        topic_id="224",
        rows=_retrieval_rows(),
        lane_records=(empty_lane,),
    )

    assert "sub.empty" in {lane.lane_id for lane in bundle.lanes}
    assert bundle.selections[0].source_lane_ids == (
        "agentic.a",
        "narrative",
        "sub.empty",
    )
    assert bundle.selections[0].document_ids == ("doc-shared", "doc-unique")


def test_validation_checks_natural_union_counts() -> None:
    bundle = _bundle()

    assert bundle.natural_document_count == 2
    assert bundle.selections[0].input_count == 2
    assert bundle.selections[0].output_count == 2

    broken = replace(bundle, natural_document_count=3)
    with pytest.raises(ValueError, match="natural union"):
        broken.validate()


def test_validation_rejects_unknown_lane_kind() -> None:
    bundle = _bundle()
    broken_lane = replace(bundle.lanes[0], lane_kind="unsupported")

    with pytest.raises(ValueError, match="unsupported lane_kind"):
        replace(bundle, lanes=(broken_lane, *bundle.lanes[1:])).validate()


def test_validation_requires_natural_union_selection() -> None:
    bundle = _bundle()

    with pytest.raises(ValueError, match="exactly one natural_union"):
        replace(bundle, selections=()).validate()


def test_from_retrieval_rows_preserves_trace_references() -> None:
    trace = TraceReference(
        trace_ref_id="trace.1",
        trace_kind="agent_trace",
        trace_sha256=_digest("trace body"),
    )
    rows = _retrieval_rows()
    rows[0]["trace_ref_id"] = trace.trace_ref_id

    bundle = EvidenceBundle.from_retrieval_rows(
        topic_id="224",
        rows=rows,
        trace_refs=(trace,),
    )

    assert bundle.trace_refs == (trace,)
    assert bundle.retrieval_events[0].trace_ref_id == trace.trace_ref_id


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
    assert payload["schema_version"] == "evidence_bundle_v1"
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


def test_deserialization_rejects_incompatible_bundle_schema_version() -> None:
    payload = _bundle().to_dict()
    incompatible = dict(payload)
    incompatible["schema_version"] = "evidence_bundle_v2"

    with pytest.raises(ValueError, match="unsupported evidence bundle schema version"):
        EvidenceBundle.from_dict(incompatible)


def test_round_trip_preserves_explicit_selection_document_order() -> None:
    bundle = _bundle()
    bundle = replace(
        bundle,
        selections=(
            bundle.selections[0],
            BundleSelection(
                selection_id="ordered",
                source_lane_ids=("agentic.a", "narrative"),
                members=(
                    BundleSelectionMember(
                        docid="doc-shared", included=True, output_rank=2
                    ),
                    BundleSelectionMember(
                        docid="doc-unique", included=True, output_rank=1
                    ),
                ),
                policy="ranked_all",
            ),
        ),
    )

    payload = bundle.to_dict()
    round_tripped = EvidenceBundle.from_dict(payload)

    ordered_payload = next(
        selection for selection in payload["selections"] if selection["selection_id"] == "ordered"
    )
    ordered_round_trip = next(
        selection for selection in round_tripped.selections if selection.selection_id == "ordered"
    )
    assert ordered_payload["document_ids"] == ["doc-unique", "doc-shared"]
    assert ordered_round_trip.document_ids == ("doc-unique", "doc-shared")


def test_validation_rejects_duplicate_selection_member_document_ids() -> None:
    bundle = replace(
        _bundle(),
        selections=(
            BundleSelection(
                selection_id="duplicate_output",
                source_lane_ids=("agentic.a", "narrative"),
                members=(
                    BundleSelectionMember(
                        docid="doc-unique", included=True, output_rank=1
                    ),
                    BundleSelectionMember(
                        docid="doc-unique", included=True, output_rank=2
                    ),
                ),
                policy="ranked_all",
            ),
        ),
    )

    with pytest.raises(ValueError, match="selection members must have sorted unique docids"):
        bundle.validate()


def _projection_bundle() -> EvidenceBundle:
    bundle = EvidenceBundle.from_retrieval_rows(
        topic_id="224",
        rows=[
            {
                "lane_id": "narrative",
                "lane_kind": "narrative",
                "query_text": "Explain the documented outcome.",
                "docid": "doc-b",
                "text": "Beta evidence lives here.",
                "rank": 1,
                "score": 10.0,
                "retriever": "bm25",
            },
            {
                "lane_id": "narrative",
                "lane_kind": "narrative",
                "query_text": "Explain the documented outcome.",
                "docid": "doc-a",
                "text": "Alpha evidence lives here.",
                "rank": 2,
                "score": 9.0,
                "retriever": "bm25",
            },
            {
                "lane_id": "agentic.a",
                "lane_kind": "agentic",
                "query_text": "Find corroborating evidence.",
                "parent_lane_id": "narrative",
                "producer": "agent-step-v1",
                "docid": "doc-c",
                "text": "Gamma corroborates alpha.",
                "rank": 1,
                "score": 8.0,
                "retriever": "bm25",
            },
            {
                "lane_id": "agentic.a",
                "lane_kind": "agentic",
                "query_text": "Find corroborating evidence.",
                "parent_lane_id": "narrative",
                "producer": "agent-step-v1",
                "docid": "doc-a",
                "text": "Alpha evidence lives here.",
                "rank": 3,
                "score": 7.0,
                "retriever": "bm25",
            },
        ],
    )
    doc_a = next(document for document in bundle.documents if document.docid == "doc-a")
    doc_b = next(document for document in bundle.documents if document.docid == "doc-b")
    evidence = (
        EvidenceSpan(
            evidence_id="evidence.alpha",
            docid="doc-a",
            text="Alpha evidence",
            text_sha256=_digest("Alpha evidence"),
            start_char=0,
            end_char=14,
            lane_ids=("agentic.a", "narrative"),
            selector="exact_span",
            source_document_sha256=doc_a.text_sha256,
        ),
        EvidenceSpan(
            evidence_id="evidence.beta",
            docid="doc-b",
            text="Beta evidence",
            text_sha256=_digest("Beta evidence"),
            start_char=0,
            end_char=13,
            lane_ids=("narrative",),
            selector="exact_span",
            source_document_sha256=doc_b.text_sha256,
        ),
    )
    nuggets = (
        BundleNugget(
            nugget_id="nugget.alpha",
            text="Alpha is directly supported.",
            text_sha256=_digest("Alpha is directly supported."),
            evidence_ids=("evidence.alpha",),
            nugget_kind="direct",
            subnarrative_id="sub.alpha",
        ),
        BundleNugget(
            nugget_id="nugget.beta",
            text="Beta is directly supported.",
            text_sha256=_digest("Beta is directly supported."),
            evidence_ids=("evidence.beta",),
            nugget_kind="direct",
        ),
    )
    selections = (
        bundle.selections[0],
        BundleSelection(
            selection_id="agentic_only",
            source_lane_ids=("agentic.a",),
            members=(
                BundleSelectionMember(docid="doc-a", included=True, output_rank=1),
                BundleSelectionMember(docid="doc-c", included=True, output_rank=2),
            ),
            policy="ranked_all",
        ),
        BundleSelection(
            selection_id="ranked_all",
            source_lane_ids=("agentic.a", "narrative"),
            members=(
                BundleSelectionMember(docid="doc-a", included=True, output_rank=1),
                BundleSelectionMember(docid="doc-b", included=True, output_rank=2),
                BundleSelectionMember(docid="doc-c", included=True, output_rank=3),
            ),
            policy="ranked_all",
        ),
    )
    narrative_lane = next(lane for lane in bundle.lanes if lane.lane_id == "narrative")
    subnarrative_text = "Explain alpha support."
    subnarrative_lane = replace(
        narrative_lane,
        lane_id="sub.alpha",
        lane_kind="subnarrative",
        query_text=subnarrative_text,
        query_text_sha256=_digest(subnarrative_text),
        parent_lane_id="narrative",
    )
    enriched = replace(
        bundle,
        lanes=(*bundle.lanes, subnarrative_lane),
        evidence=evidence,
        nuggets=nuggets,
        selections=(
            replace(
                selections[0],
                source_lane_ids=("agentic.a", "narrative", "sub.alpha"),
            ),
            *selections[1:],
        ),
    )
    enriched.validate()
    return enriched


def test_lane_kinds_compile_to_the_same_bundle_schema() -> None:
    rows = [
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
            "lane_id": "sub.alpha",
            "lane_kind": "subnarrative",
            "query_text": "Explain the documented subnarrative.",
            "parent_lane_id": "narrative",
            "docid": "doc-shared",
            "text": "Alpha Beta",
            "rank": 1,
            "score": 11.5,
            "retriever": "bm25",
        },
        {
            "lane_id": "agentic.a",
            "lane_kind": "agentic",
            "query_text": "Find independent corroboration.",
            "parent_lane_id": "sub.alpha",
            "producer": "synthetic-agent-step-v1",
            "docid": "doc-agentic",
            "text": "Gamma Delta",
            "rank": 1,
            "score": 10.5,
            "retriever": "bm25",
        },
    ]

    bundle = EvidenceBundle.from_retrieval_rows(topic_id="224", rows=rows)
    payload = bundle.to_dict()

    assert payload["schema_version"] == "evidence_bundle_v1"
    assert set(payload.keys()) == {
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
    }
    assert [lane["lane_kind"] for lane in payload["lanes"]] == [
        "agentic",
        "narrative",
        "subnarrative",
    ]
    assert payload["documents"] == [
        {
            "docid": "doc-agentic",
            "text": "Gamma Delta",
            "text_sha256": _digest("Gamma Delta"),
            "lane_ids": ["agentic.a"],
        },
        {
            "docid": "doc-shared",
            "text": "Alpha Beta",
            "text_sha256": _digest("Alpha Beta"),
            "lane_ids": ["narrative", "sub.alpha"],
        },
    ]


def _projection_bundle_with_mixed_lane_support() -> EvidenceBundle:
    bundle = _projection_bundle()
    doc_a = next(document for document in bundle.documents if document.docid == "doc-a")
    mixed_evidence = (
        *bundle.evidence,
        EvidenceSpan(
            evidence_id="evidence.alpha.agentic_only",
            docid="doc-a",
            text="evidence",
            text_sha256=_digest("evidence"),
            start_char=6,
            end_char=14,
            lane_ids=("agentic.a",),
            selector="exact_span",
            source_document_sha256=doc_a.text_sha256,
        ),
        EvidenceSpan(
            evidence_id="evidence.alpha.narrative_only",
            docid="doc-a",
            text="lives",
            text_sha256=_digest("lives"),
            start_char=15,
            end_char=20,
            lane_ids=("narrative",),
            selector="exact_span",
            source_document_sha256=doc_a.text_sha256,
        ),
    )
    mixed_nuggets = (
        *bundle.nuggets,
        BundleNugget(
            nugget_id="nugget.alpha.agentic_only",
            text="Agentic support survives selection filtering.",
            text_sha256=_digest("Agentic support survives selection filtering."),
            evidence_ids=("evidence.alpha.agentic_only",),
            nugget_kind="direct",
        ),
        BundleNugget(
            nugget_id="nugget.alpha.mixed_support",
            text="Mixed support should project only selected evidence.",
            text_sha256=_digest("Mixed support should project only selected evidence."),
            evidence_ids=("evidence.alpha.agentic_only", "evidence.alpha.narrative_only"),
            nugget_kind="direct",
        ),
        BundleNugget(
            nugget_id="nugget.alpha.narrative_only",
            text="Narrative-only support should be filtered out for agentic selection.",
            text_sha256=_digest("Narrative-only support should be filtered out for agentic selection."),
            evidence_ids=("evidence.alpha.narrative_only",),
            nugget_kind="direct",
        ),
    )
    enriched = replace(bundle, evidence=mixed_evidence, nuggets=mixed_nuggets)
    enriched.validate()
    return enriched


def test_to_trec_run_uses_explicit_selection_and_rank_order_without_top_100() -> None:
    bundle = _projection_bundle()

    assert bundle.to_trec_run(selection_id="agentic_only") == (
        ("224", "doc-a", 1, 2, "agentic_only"),
        ("224", "doc-c", 2, 1, "agentic_only"),
    )


def test_with_ranked_selection_records_rejections_and_explicit_order() -> None:
    bundle = _bundle().with_ranked_selection(
        selection_id="official",
        document_ids=("doc-unique",),
        policy="canonical_supported",
    )

    selection = next(row for row in bundle.selections if row.selection_id == "official")
    assert selection.document_ids == ("doc-unique",)
    assert [(member.docid, member.included, member.output_rank, member.rejection_reason)
            for member in selection.members] == [
        ("doc-shared", False, None, "not_selected"),
        ("doc-unique", True, 1, None),
    ]


def test_natural_union_preserves_more_than_100_unranked_documents() -> None:
    rows = [
        {
            "lane_id": "narrative",
            "lane_kind": "narrative",
            "query_text": "Explain the documented outcome.",
            "docid": f"doc-{index:03d}",
            "text": f"Document {index}",
            "rank": index,
            "score": float(1000 - index),
            "retriever": "bm25",
        }
        for index in range(1, 102)
    ]
    bundle = EvidenceBundle.from_retrieval_rows(topic_id="224", rows=rows)

    natural_union = bundle.selections[0]

    assert natural_union.input_count == 101
    assert natural_union.output_count == 101
    assert all(member.included for member in natural_union.members)
    assert all(member.output_rank is None for member in natural_union.members)


def test_to_document_records_emits_organizer_core_with_lane_metadata() -> None:
    bundle = _projection_bundle()

    records = bundle.to_document_records(selection_id="agentic_only")

    assert records == (
        {
            "query": {
                "qid": "224",
                "selection_id": "agentic_only",
                "text": "Explain the documented outcome.",
                "text_sha256": _digest("Explain the documented outcome."),
            },
            "candidates": [
                {
                    "docid": "doc-a",
                    "doc": "Alpha evidence lives here.",
                    "rank": 1,
                    "score": 2,
                    "lane_ids": ["agentic.a", "narrative"],
                    "text_sha256": _digest("Alpha evidence lives here."),
                },
                {
                    "docid": "doc-c",
                    "doc": "Gamma corroborates alpha.",
                    "rank": 2,
                    "score": 1,
                    "lane_ids": ["agentic.a"],
                    "text_sha256": _digest("Gamma corroborates alpha."),
                },
            ],
        },
    )


def test_to_fixed_rag_context_orders_documents_evidence_and_nuggets() -> None:
    bundle = _projection_bundle()

    context = bundle.to_fixed_rag_context(selection_id="ranked_all")

    assert context == (
        {
            "topic_id": "224",
            "selection_id": "ranked_all",
            "query": {
                "qid": "224",
                "text": "Explain the documented outcome.",
                "text_sha256": _digest("Explain the documented outcome."),
            },
            "docid": "doc-a",
            "rank": 1,
            "score": 3,
            "lane_ids": ["agentic.a", "narrative"],
            "document": {
                "text": "Alpha evidence lives here.",
                "text_sha256": _digest("Alpha evidence lives here."),
            },
            "evidence": [
                {
                    "evidence_id": "evidence.alpha",
                    "text": "Alpha evidence",
                    "text_sha256": _digest("Alpha evidence"),
                    "start_char": 0,
                    "end_char": 14,
                    "lane_ids": ["agentic.a", "narrative"],
                    "selector": "exact_span",
                }
            ],
            "nuggets": [
                {
                    "nugget_id": "nugget.alpha",
                    "text": "Alpha is directly supported.",
                    "text_sha256": _digest("Alpha is directly supported."),
                    "nugget_kind": "direct",
                    "subnarrative_id": "sub.alpha",
                    "evidence_ids": ["evidence.alpha"],
                }
            ],
        },
        {
            "topic_id": "224",
            "selection_id": "ranked_all",
            "query": {
                "qid": "224",
                "text": "Explain the documented outcome.",
                "text_sha256": _digest("Explain the documented outcome."),
            },
            "docid": "doc-b",
            "rank": 2,
            "score": 2,
            "lane_ids": ["narrative"],
            "document": {
                "text": "Beta evidence lives here.",
                "text_sha256": _digest("Beta evidence lives here."),
            },
            "evidence": [
                {
                    "evidence_id": "evidence.beta",
                    "text": "Beta evidence",
                    "text_sha256": _digest("Beta evidence"),
                    "start_char": 0,
                    "end_char": 13,
                    "lane_ids": ["narrative"],
                    "selector": "exact_span",
                }
            ],
            "nuggets": [
                {
                    "nugget_id": "nugget.beta",
                    "text": "Beta is directly supported.",
                    "text_sha256": _digest("Beta is directly supported."),
                    "nugget_kind": "direct",
                    "subnarrative_id": None,
                    "evidence_ids": ["evidence.beta"],
                }
            ],
        },
        {
            "topic_id": "224",
            "selection_id": "ranked_all",
            "query": {
                "qid": "224",
                "text": "Explain the documented outcome.",
                "text_sha256": _digest("Explain the documented outcome."),
            },
            "docid": "doc-c",
            "rank": 3,
            "score": 1,
            "lane_ids": ["agentic.a"],
            "document": {
                "text": "Gamma corroborates alpha.",
                "text_sha256": _digest("Gamma corroborates alpha."),
            },
            "evidence": [],
            "nuggets": [],
        },
    )


def test_to_fixed_rag_context_filters_support_to_selected_source_lanes() -> None:
    bundle = _projection_bundle_with_mixed_lane_support()

    context = bundle.to_fixed_rag_context(selection_id="agentic_only")

    assert context == (
        {
            "topic_id": "224",
            "selection_id": "agentic_only",
            "query": {
                "qid": "224",
                "text": "Explain the documented outcome.",
                "text_sha256": _digest("Explain the documented outcome."),
            },
            "docid": "doc-a",
            "rank": 1,
            "score": 2,
            "lane_ids": ["agentic.a", "narrative"],
            "document": {
                "text": "Alpha evidence lives here.",
                "text_sha256": _digest("Alpha evidence lives here."),
            },
            "evidence": [
                {
                    "evidence_id": "evidence.alpha",
                    "text": "Alpha evidence",
                    "text_sha256": _digest("Alpha evidence"),
                    "start_char": 0,
                    "end_char": 14,
                    "lane_ids": ["agentic.a", "narrative"],
                    "selector": "exact_span",
                },
                {
                    "evidence_id": "evidence.alpha.agentic_only",
                    "text": "evidence",
                    "text_sha256": _digest("evidence"),
                    "start_char": 6,
                    "end_char": 14,
                    "lane_ids": ["agentic.a"],
                    "selector": "exact_span",
                },
            ],
            "nuggets": [
                {
                    "nugget_id": "nugget.alpha",
                    "text": "Alpha is directly supported.",
                    "text_sha256": _digest("Alpha is directly supported."),
                    "nugget_kind": "direct",
                    "subnarrative_id": "sub.alpha",
                    "evidence_ids": ["evidence.alpha"],
                },
                {
                    "nugget_id": "nugget.alpha.agentic_only",
                    "text": "Agentic support survives selection filtering.",
                    "text_sha256": _digest("Agentic support survives selection filtering."),
                    "nugget_kind": "direct",
                    "subnarrative_id": None,
                    "evidence_ids": ["evidence.alpha.agentic_only"],
                },
                {
                    "nugget_id": "nugget.alpha.mixed_support",
                    "text": "Mixed support should project only selected evidence.",
                    "text_sha256": _digest("Mixed support should project only selected evidence."),
                    "nugget_kind": "direct",
                    "subnarrative_id": None,
                    "evidence_ids": ["evidence.alpha.agentic_only"],
                },
            ],
        },
        {
            "topic_id": "224",
            "selection_id": "agentic_only",
            "query": {
                "qid": "224",
                "text": "Explain the documented outcome.",
                "text_sha256": _digest("Explain the documented outcome."),
            },
            "docid": "doc-c",
            "rank": 2,
            "score": 1,
            "lane_ids": ["agentic.a"],
            "document": {
                "text": "Gamma corroborates alpha.",
                "text_sha256": _digest("Gamma corroborates alpha."),
            },
            "evidence": [],
            "nuggets": [],
        },
    )


def test_write_fixed_rag_inputs_writes_deterministic_run_and_document_sidecars(
    tmp_path: Path,
) -> None:
    bundle = _projection_bundle()

    outputs = bundle.write_fixed_rag_inputs(tmp_path, selection_id="agentic_only")

    assert outputs["run"] == tmp_path / "r_output_trec_rag_2026.tsv"
    assert outputs["documents_jsonl"] == tmp_path / "retrieval_with_text.jsonl"
    assert outputs["documents_zip"] == tmp_path / "retrieval_with_text.jsonl.zip"
    assert outputs["run"].read_text(encoding="utf-8") == (
        "224 Q0 doc-a 1 2 agentic_only\n"
        "224 Q0 doc-c 2 1 agentic_only\n"
    )
    assert outputs["documents_jsonl"].read_text(encoding="utf-8") == json.dumps(
        bundle.to_document_records(selection_id="agentic_only")[0],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"
    with zipfile.ZipFile(outputs["documents_zip"]) as archive:
        assert archive.namelist() == ["retrieval_with_text.jsonl"]
        assert archive.read("retrieval_with_text.jsonl") == outputs["documents_jsonl"].read_bytes()


def test_fixed_rag_query_projection_rejects_lossy_tsv_text(tmp_path: Path) -> None:
    query = "Preserve this line.\nAnd this one."
    bundle = EvidenceBundle.from_retrieval_rows(
        topic_id="224",
        rows=[
            {
                "lane_id": "narrative",
                "lane_kind": "narrative",
                "query_text": query,
                "docid": "doc-only",
                "text": "Only document.",
                "rank": 1,
                "score": 1.0,
                "retriever": "bm25",
            }
        ],
    ).with_ranked_selection(
        selection_id="official",
        document_ids=("doc-only",),
        policy="official",
    )

    with pytest.raises(ValueError, match="cannot represent tabs or line breaks"):
        evidence_bundle_module.write_fixed_rag_package(
            [bundle], tmp_path, selection_id="official"
        )


def test_validation_rejects_evidence_without_lane_support() -> None:
    bundle = _bundle()
    shared = next(document for document in bundle.documents if document.docid == "doc-shared")
    unsupported = EvidenceSpan(
        evidence_id="evidence.unsupported",
        docid="doc-shared",
        text="Beta",
        text_sha256=_digest("Beta"),
        start_char=6,
        end_char=10,
        lane_ids=(),
        selector="exact_span",
        source_document_sha256=shared.text_sha256,
    )

    with pytest.raises(ValueError, match="at least one lane"):
        replace(bundle, evidence=(unsupported,)).validate()


def test_validation_rejects_nugget_without_evidence_support() -> None:
    unsupported = BundleNugget(
        nugget_id="nugget.unsupported",
        text="This claim has no evidence.",
        text_sha256=_digest("This claim has no evidence."),
        evidence_ids=(),
        nugget_kind="direct",
    )

    with pytest.raises(ValueError, match="at least one evidence"):
        replace(_bundle(), nuggets=(unsupported,)).validate()


@pytest.mark.parametrize("subnarrative_id", ["missing.subnarrative", "narrative"])
def test_validation_requires_nugget_subnarrative_to_resolve_to_subnarrative_lane(
    subnarrative_id: str,
) -> None:
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
        selector="exact_span",
        source_document_sha256=shared.text_sha256,
    )
    nugget = BundleNugget(
        nugget_id="nugget.beta",
        text="Beta is present.",
        text_sha256=_digest("Beta is present."),
        evidence_ids=("evidence.beta",),
        nugget_kind="direct",
        subnarrative_id=subnarrative_id,
    )

    with pytest.raises(ValueError, match="subnarrative lane"):
        replace(bundle, evidence=(evidence,), nuggets=(nugget,)).validate()


def test_selection_members_record_inclusion_rank_and_rejection_reason() -> None:
    payload = _bundle().to_dict()
    selection = payload["selections"][0]
    natural_union = _bundle().to_dict()["selections"][0]
    selection.update(
        {
            "selection_id": "top_one",
            "policy": "ranked_top_one",
            "document_ids": ["doc-unique"],
            "output_count": 1,
            "members": [
                {
                    "docid": "doc-shared",
                    "included": False,
                    "output_rank": None,
                    "rejection_reason": "context_budget",
                },
                {
                    "docid": "doc-unique",
                    "included": True,
                    "output_rank": 1,
                    "rejection_reason": None,
                },
            ],
        }
    )
    payload["selections"].append(natural_union)

    round_tripped = EvidenceBundle.from_dict(payload)

    top_one = next(
        selection for selection in round_tripped.selections if selection.selection_id == "top_one"
    )
    assert [
        (member.docid, member.included, member.output_rank, member.rejection_reason)
        for member in top_one.members
    ] == [
        ("doc-shared", False, None, "context_budget"),
        ("doc-unique", True, 1, None),
    ]
    assert top_one.document_ids == ("doc-unique",)
    assert top_one.input_count == 2
    assert top_one.output_count == 1
    top_one_payload = next(
        item for item in round_tripped.to_dict()["selections"] if item["selection_id"] == "top_one"
    )
    assert top_one_payload["members"] == selection["members"]


def test_natural_union_is_an_unranked_relation_and_cannot_emit_trec_run() -> None:
    bundle = EvidenceBundle.from_retrieval_rows(
        topic_id="224",
        rows=[
            {
                "lane_id": "narrative",
                "lane_kind": "narrative",
                "query_text": "Question",
                "docid": "doc-z",
                "text": "First by retrieval rank.",
                "rank": 1,
                "score": 10.0,
                "retriever": "bm25",
            },
            {
                "lane_id": "narrative",
                "lane_kind": "narrative",
                "query_text": "Question",
                "docid": "doc-a",
                "text": "Second by retrieval rank.",
                "rank": 2,
                "score": 9.0,
                "retriever": "bm25",
            },
        ],
    )

    assert [member.output_rank for member in bundle.selections[0].members] == [None, None]
    with pytest.raises(ValueError, match="non-ranked selection"):
        bundle.to_trec_run(selection_id="natural_union")


def test_lane_query_hashes_round_trip_and_multiline_source_text_is_valid() -> None:
    query_text = "Question first line.\nQuestion second line."
    document_text = "Document first line.\n\nDocument second paragraph."

    bundle = EvidenceBundle.from_retrieval_rows(
        topic_id="224",
        rows=[
            {
                "lane_id": "narrative",
                "lane_kind": "narrative",
                "query_text": query_text,
                "docid": "doc-multiline",
                "text": document_text,
                "rank": 1,
                "score": 1.0,
                "retriever": "bm25",
            }
        ],
    )

    assert bundle.lanes[0].query_text_sha256 == _digest(query_text)
    assert bundle.documents[0].text == document_text
    assert EvidenceBundle.from_dict(bundle.to_dict()) == bundle

    broken_lane = replace(bundle.lanes[0], query_text_sha256="0" * 64)
    with pytest.raises(ValueError, match="query_text_sha256 mismatch"):
        replace(bundle, lanes=(broken_lane,)).validate()


@pytest.mark.parametrize("relation_key", ["evidence", "nuggets", "trace_refs"])
def test_v1_deserialization_requires_complete_relation_keys(relation_key: str) -> None:
    payload = _bundle().to_dict()
    del payload[relation_key]

    with pytest.raises(ValueError, match="payload keys"):
        EvidenceBundle.from_dict(payload)


@pytest.mark.parametrize("field_name", ["rank", "score"])
def test_v1_deserialization_rejects_boolean_retrieval_scalars(field_name: str) -> None:
    payload = _bundle().to_dict()
    payload["retrieval_events"][0][field_name] = True

    with pytest.raises(ValueError, match=field_name):
        EvidenceBundle.from_dict(payload)


def test_v1_deserialization_rejects_boolean_selection_scalars() -> None:
    bundle = EvidenceBundle.from_retrieval_rows(
        topic_id="224",
        rows=[
            {
                "lane_id": "narrative",
                "lane_kind": "narrative",
                "query_text": "Question",
                "docid": "doc-only",
                "text": "Only document.",
                "rank": 1,
                "score": 1.0,
                "retriever": "bm25",
            }
        ],
    )
    payload = bundle.to_dict()
    payload["natural_document_count"] = True
    payload["selections"][0]["input_count"] = True
    payload["selections"][0]["output_count"] = True

    with pytest.raises(ValueError, match="integer"):
        EvidenceBundle.from_dict(payload)


def test_write_fixed_rag_package_writes_one_deterministic_query_aware_multi_topic_package(
    tmp_path: Path,
) -> None:
    first = _projection_bundle()
    first_selection = next(
        selection for selection in first.selections if selection.selection_id == "agentic_only"
    )
    second_query = "Summarize the second topic."
    second = replace(
        first,
        topic_id="225",
        lanes=tuple(
            replace(
                lane,
                query_text=second_query,
                query_text_sha256=_digest(second_query),
            )
            if lane.lane_kind == "narrative"
            else lane
            for lane in first.lanes
        ),
        selections=(first.selections[0], first_selection),
    )
    first = replace(first, selections=(first.selections[0], first_selection))

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    outputs = evidence_bundle_module.write_fixed_rag_package(
        [second, first], first_dir, selection_id="agentic_only"
    )
    repeated = evidence_bundle_module.write_fixed_rag_package(
        [first, second], second_dir, selection_id="agentic_only"
    )

    assert outputs == {
        "queries": first_dir / "trec_rag_2026_queries.tsv",
        "run": first_dir / "r_output_trec_rag_2026.tsv",
        "documents_jsonl": first_dir / "retrieval_with_text.jsonl",
        "documents_zip": first_dir / "retrieval_with_text.jsonl.zip",
        "context_jsonl": first_dir / "fixed_rag_context.jsonl",
    }
    assert outputs["queries"].read_text(encoding="utf-8") == (
        "224\tExplain the documented outcome.\n"
        "225\tSummarize the second topic.\n"
    )
    assert outputs["run"].read_text(encoding="utf-8") == (
        "224 Q0 doc-a 1 2 agentic_only\n"
        "224 Q0 doc-c 2 1 agentic_only\n"
        "225 Q0 doc-a 1 2 agentic_only\n"
        "225 Q0 doc-c 2 1 agentic_only\n"
    )
    document_rows = [
        json.loads(line) for line in outputs["documents_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    context_rows = [
        json.loads(line) for line in outputs["context_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    assert [row["query"]["qid"] for row in document_rows] == ["224", "225"]
    assert document_rows[0]["query"] == {
        "qid": "224",
        "selection_id": "agentic_only",
        "text": "Explain the documented outcome.",
        "text_sha256": _digest("Explain the documented outcome."),
    }
    assert [row["query"]["qid"] for row in context_rows] == ["224", "224", "225", "225"]
    with zipfile.ZipFile(outputs["documents_zip"]) as archive:
        assert archive.namelist() == ["retrieval_with_text.jsonl"]
        assert archive.read("retrieval_with_text.jsonl") == outputs["documents_jsonl"].read_bytes()
    for key, first_path in outputs.items():
        assert first_path.read_bytes() == repeated[key].read_bytes()
