from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import pytest

from trec_rag.evidence_bundle import (
    BundleSelection,
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


def test_round_trip_preserves_explicit_selection_document_order() -> None:
    bundle = replace(
        _bundle(),
        selections=(
            BundleSelection(
                selection_id="ordered",
                source_lane_ids=("agentic.a", "narrative"),
                input_document_ids=("doc-shared", "doc-unique"),
                document_ids=("doc-unique", "doc-shared"),
                input_count=2,
                output_count=2,
            ),
        ),
    )

    payload = bundle.to_dict()
    round_tripped = EvidenceBundle.from_dict(payload)

    assert payload["selections"][0]["document_ids"] == ["doc-unique", "doc-shared"]
    assert round_tripped.selections[0].document_ids == ("doc-unique", "doc-shared")


def test_validation_rejects_duplicate_selection_output_document_ids() -> None:
    bundle = replace(
        _bundle(),
        selections=(
            BundleSelection(
                selection_id="duplicate_output",
                source_lane_ids=("agentic.a", "narrative"),
                input_document_ids=("doc-shared", "doc-unique"),
                document_ids=("doc-unique", "doc-unique"),
                input_count=2,
                output_count=2,
            ),
        ),
    )

    with pytest.raises(ValueError, match="selection document_ids must be unique"):
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
            input_document_ids=("doc-a", "doc-c"),
            document_ids=("doc-a", "doc-c"),
            input_count=2,
            output_count=2,
        ),
    )
    enriched = replace(bundle, evidence=evidence, nuggets=nuggets, selections=selections)
    enriched.validate()
    return enriched


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


def test_to_trec_run_preserves_natural_union_beyond_100_documents() -> None:
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

    projected = bundle.to_trec_run(selection_id="natural_union")

    assert len(projected) == 101
    assert projected[0] == ("224", "doc-001", 1, 101, "natural_union")
    assert projected[-1] == ("224", "doc-101", 101, 1, "natural_union")


def test_to_document_records_emits_organizer_core_with_lane_metadata() -> None:
    bundle = _projection_bundle()

    records = bundle.to_document_records(selection_id="agentic_only")

    assert records == (
        {
            "query": {"qid": "224", "selection_id": "agentic_only"},
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
                    "rank": 1,
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

    context = bundle.to_fixed_rag_context(selection_id="natural_union")

    assert context == (
        {
            "topic_id": "224",
            "selection_id": "natural_union",
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
            "selection_id": "natural_union",
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
            "selection_id": "natural_union",
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
