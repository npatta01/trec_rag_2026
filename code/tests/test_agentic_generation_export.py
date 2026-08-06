from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import pytest

from trec_rag.agentic_generation_export import (
    AgenticProjectionError,
    prepare_agentic_topic_projection,
    serialize_agentic_retrieval_topic,
)
from trec_rag.deepagent_evidence import (
    EvidenceCoverageReport,
    EvidenceReference,
    NeedReport,
    NuggetReport,
)
from trec_rag.document_store import DocumentStore
from trec_rag.generation_handoff import serialize_generation_topic
from trec_rag.pipeline_models import RankedCandidate
from trec_rag.topic_passage_search import SourceDocument, SourcePassage
from trec_rag.topic_records import TopicEvidenceSnapshot
from trec_rag.topics import Topic


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _passage(
    *,
    docid: str,
    document: str,
    text: str,
    passage_id: str,
    source_rank: int,
    rank: int,
) -> SourcePassage:
    start_char = document.index(text)
    end_char = start_char + len(text)
    start_byte = len(document[:start_char].encode("utf-8"))
    end_byte = len(document[:end_char].encode("utf-8"))
    return SourcePassage(
        passage_id=passage_id,
        docid=docid,
        content_sha256=_digest(document),
        source_rank=source_rank,
        source_score=1.0 / source_rank,
        start_char=start_char,
        end_char=end_char,
        start_byte=start_byte,
        end_byte=end_byte,
        text_sha256=_digest(text),
        text=text,
        raw_logit=4.0 - rank,
        rank=rank,
        score_cache_key=f"score-{passage_id}",
        scoring_text_sha256=_digest(text),
        chunker_identity={"fixture": "unicode-v1"},
    )


def _document(docid: str, text: str, passage: SourcePassage, rank: int) -> SourceDocument:
    return SourceDocument(
        docid=docid,
        content_sha256=_digest(text),
        source_rank=rank,
        source_score=1.0 / rank,
        best_passage_id=passage.passage_id,
        best_passage_raw_logit=passage.raw_logit,
    )


def _need(
    need_id: str,
    question: str,
    nugget_ids: tuple[str, ...],
    draft_nugget_ids: tuple[str, ...],
) -> NeedReport:
    return NeedReport(
        need_id=need_id,
        narrative_span=question,
        question=question,
        status="answerable",
        remaining_gap="",
        facet_ids=(),
        nugget_ids=nugget_ids,
        draft_answer="Grounded draft",
        draft_nugget_ids=draft_nugget_ids,
    )


def _nugget(
    nugget_id: str,
    text: str,
    need_ids: tuple[str, ...],
    *references: tuple[str, str],
) -> NuggetReport:
    return NuggetReport(
        nugget_id=nugget_id,
        text=text,
        need_ids=need_ids,
        facet_ids=(),
        evidence=tuple(
            EvidenceReference(
                document_id=docid,
                snippet_id=passage_id,
                page_index=0,
                quote="Model-visible excerpt; not the exported source text.",
            )
            for docid, passage_id in references
        ),
        contradicts=(),
        support="multi_document" if len({row[0] for row in references}) > 1 else "single_document",
        superseded_by=None,
        importance="vital",
        support_ratio=1.0,
    )


def _report(
    needs: tuple[NeedReport, ...], nuggets: tuple[NuggetReport, ...]
) -> EvidenceCoverageReport:
    return EvidenceCoverageReport(
        needs=needs,
        facets=(),
        nuggets=nuggets,
        actions=(),
        searches=(),
        documents=(),
        unresolved_need_ids=(),
        search_count=1,
        inspected_page_count=1,
        state_version=7,
        state_hash="e" * 64,
        terminal_reason="coverage_sufficient",
    )


@dataclass(frozen=True)
class _Fixture:
    topic: Topic
    report: EvidenceCoverageReport
    snapshot: TopicEvidenceSnapshot
    fused: tuple[RankedCandidate, ...]
    searches: tuple[_Search, ...]
    store: DocumentStore
    texts: dict[str, str]
    passages: dict[str, SourcePassage]


@dataclass(frozen=True)
class _Search:
    candidates: tuple[RankedCandidate, ...]


def _fixture(tmp_path: Path) -> _Fixture:
    topic = Topic(
        id="rag2026-0",
        title="unused",
        narrative="Explain the causes and consequences of the documented event.",
    )
    texts = {
        "doc-a": "Préface café.\nGrounded α source passage.\nTail A.",
        "doc-b": "Opening β.\nIndependent Ω support.\nTail B.",
        "doc-c": "Original-query-only passage that no admitted nugget cites.",
        "doc-d": "Context.\nA live but unselected grounded passage.\nTail D.",
    }
    passages = {
        "p-a": _passage(
            docid="doc-a",
            document=texts["doc-a"],
            text="Grounded α source passage.",
            passage_id="p-a",
            source_rank=2,
            rank=2,
        ),
        "p-b": _passage(
            docid="doc-b",
            document=texts["doc-b"],
            text="Independent Ω support.",
            passage_id="p-b",
            source_rank=1,
            rank=1,
        ),
        "p-c-original": _passage(
            docid="doc-c",
            document=texts["doc-c"],
            text=texts["doc-c"],
            passage_id="p-c-original",
            source_rank=3,
            rank=3,
        ),
        "p-d": _passage(
            docid="doc-d",
            document=texts["doc-d"],
            text="A live but unselected grounded passage.",
            passage_id="p-d",
            source_rank=4,
            rank=4,
        ),
    }
    documents = tuple(
        _document(docid, texts[docid], passages[passage_id], rank)
        for rank, (docid, passage_id) in enumerate(
            (
                ("doc-b", "p-b"),
                ("doc-a", "p-a"),
                ("doc-c", "p-c-original"),
                ("doc-d", "p-d"),
            ),
            start=1,
        )
    )
    store = DocumentStore(tmp_path / "documents")
    for text in texts.values():
        store.admit_text(text)

    g1 = _nugget(
        "g1",
        "The event has two independently documented causes.",
        ("n1",),
        ("doc-a", "p-a"),
        ("doc-b", "p-b"),
    )
    g2 = _nugget(
        "g2",
        "The consequence is independently supported.",
        ("n2",),
        ("doc-b", "p-b"),
    )
    g3 = _nugget(
        "g3",
        "This live nugget was grounded but not chosen for a draft.",
        ("n2",),
        ("doc-d", "p-d"),
    )
    report = _report(
        (
            _need("n1", "What caused the event?", ("g1",), ("g1",)),
            _need("n2", "What followed the event?", ("g2", "g3"), ("g2",)),
        ),
        (g1, g2, g3),
    )
    snapshot = TopicEvidenceSnapshot(
        facets=(),
        queries=(),
        documents=documents,
        passages=tuple(passages.values()),
        researcher_handoffs=(),
        researcher_evidence=(),
        candidates=(),
        status="complete",
        stopping_reason="coverage_sufficient",
    )
    ranked_candidates = tuple(
        RankedCandidate(
            topic_id=topic.id,
            docid=docid,
            rank=rank,
            score=1.0 / rank,
            text=texts[docid],
            provenance=[],
        )
        for rank, docid in enumerate(("doc-b", "doc-c", "doc-a", "doc-d"), start=1)
    )
    fused = ranked_candidates[:3]
    searches = (_Search(ranked_candidates),)
    return _Fixture(topic, report, snapshot, fused, searches, store, texts, passages)


def _project(fixture: _Fixture):
    return prepare_agentic_topic_projection(
        topic=fixture.topic,
        report=fixture.report,
        snapshot=fixture.snapshot,
        fused_candidates=fixture.fused,
        searches=fixture.searches,
        document_store=fixture.store,
        official_topics_sha256="f" * 64,
    )


def test_projection_exports_exact_grounded_unicode_evidence_and_document_closure(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    projection = _project(fixture)

    assert [row.docid for row in projection.retrieval_rows] == [
        "doc-b",
        "doc-a",
        "doc-d",
    ]
    assert [row.rank for row in projection.retrieval_rows] == [1, 2, 3]
    assert [row.score for row in projection.retrieval_rows] == [3, 2, 1]
    assert projection.full_text_record == {
        "query": {
            "qid": fixture.topic.id,
            "selection_id": "official",
            "text": fixture.topic.narrative,
            "text_sha256": _digest(fixture.topic.narrative),
        },
        "candidates": [
            {
                "docid": "doc-b",
                "doc": fixture.texts["doc-b"],
                "rank": 1,
                "score": 3,
                "lane_ids": ["agentic"],
                "text_sha256": _digest(fixture.texts["doc-b"]),
            },
            {
                "docid": "doc-a",
                "doc": fixture.texts["doc-a"],
                "rank": 2,
                "score": 2,
                "lane_ids": ["agentic"],
                "text_sha256": _digest(fixture.texts["doc-a"]),
            },
            {
                "docid": "doc-d",
                "doc": fixture.texts["doc-d"],
                "rank": 3,
                "score": 1,
                "lane_ids": ["agentic"],
                "text_sha256": _digest(fixture.texts["doc-d"]),
            },
        ],
    }

    generation = projection.generation_topic
    assert [group.text for group in generation.groups] == [
        "What caused the event?",
        "What followed the event?",
    ]
    assert [len(group.selected_clusters) for group in generation.groups] == [1, 1]
    assert [claim.text for claim in generation.claim_hints] == [
        "The event has two independently documented causes.",
        "The consequence is independently supported.",
    ]
    assert [row.text for row in generation.evidence] == [
        fixture.passages["p-a"].text,
        fixture.passages["p-b"].text,
        fixture.passages["p-b"].text,
    ]
    assert [row.document_rank for row in generation.evidence] == [2, 1, 1]
    assert generation.evidence[0].source_span.start_char == fixture.passages["p-a"].start_char
    assert generation.evidence[0].source_span.end_byte == fixture.passages["p-a"].end_byte
    assert generation.evidence[0].text != fixture.report.nuggets[0].evidence[0].quote

    trec_docids = {row.docid for row in projection.retrieval_rows}
    full_text_docids = {
        row["docid"] for row in projection.full_text_record["candidates"]
    }
    assert set(generation.citation_docids) <= trec_docids == full_text_docids
    assert generation.citation_docids == ("doc-b", "doc-a")
    assert "doc-c" not in trec_docids
    assert fixture.passages["p-c-original"].text not in {
        row.text for row in generation.evidence
    }


def test_projection_is_byte_deterministic_and_receipts_its_retrieval_projection(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    first = _project(fixture)
    second = _project(fixture)

    assert first == second
    assert serialize_agentic_retrieval_topic(first) == serialize_agentic_retrieval_topic(second)
    assert serialize_generation_topic(first.generation_topic) == serialize_generation_topic(
        second.generation_topic
    )
    assert first.retrieval_topic_sha256 == sha256(
        serialize_agentic_retrieval_topic(first)
    ).hexdigest()
    assert (
        first.generation_topic.source_receipts.retrieval_topic_sha256
        == first.retrieval_topic_sha256
    )
    evidence_ids = [row.evidence_id for row in first.generation_topic.evidence]
    assert len(evidence_ids) == len(set(evidence_ids))
    assert all(evidence_id.startswith("agentic-evidence-") for evidence_id in evidence_ids)


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("missing", "unknown selected nugget"),
        ("superseded", "superseded"),
        ("ungrounded", "has no grounded evidence"),
        ("unassociated", "not associated"),
        ("missing_document", "unknown document"),
        ("mismatched_hash", "document hash"),
        ("missing_passage", "unknown passage"),
        ("zero_selected", "no selected grounded evidence"),
    ),
)
def test_projection_fails_closed_on_invalid_selected_evidence(
    tmp_path: Path, case: str, match: str
) -> None:
    fixture = _fixture(tmp_path)
    needs = list(fixture.report.needs)
    nuggets = list(fixture.report.nuggets)
    snapshot = fixture.snapshot

    if case == "missing":
        needs[0] = replace(needs[0], draft_nugget_ids=("does-not-exist",))
    elif case == "superseded":
        nuggets[0] = replace(nuggets[0], superseded_by="g2")
    elif case == "ungrounded":
        nuggets[0] = replace(nuggets[0], evidence=())
    elif case == "unassociated":
        nuggets[0] = replace(nuggets[0], need_ids=("n2",))
    elif case == "missing_document":
        snapshot = replace(
            snapshot,
            documents=tuple(row for row in snapshot.documents if row.docid != "doc-a"),
        )
    elif case == "mismatched_hash":
        bad_passages = tuple(
            replace(row, content_sha256="a" * 64) if row.passage_id == "p-a" else row
            for row in snapshot.passages
        )
        snapshot = replace(snapshot, passages=bad_passages)
    elif case == "missing_passage":
        nuggets[0] = replace(
            nuggets[0],
            evidence=(
                replace(nuggets[0].evidence[0], snippet_id="does-not-exist"),
            ),
        )
    elif case == "zero_selected":
        needs = [replace(need, draft_nugget_ids=()) for need in needs]
    else:  # pragma: no cover - protects the fixture table
        raise AssertionError(case)

    invalid = replace(
        fixture,
        report=replace(fixture.report, needs=tuple(needs), nuggets=tuple(nuggets)),
        snapshot=snapshot,
    )
    with pytest.raises(AgenticProjectionError, match=match):
        _project(invalid)


def test_projection_rejects_passage_offsets_that_do_not_match_stored_document(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    passage = fixture.passages["p-a"]
    shifted = replace(
        passage,
        start_char=passage.start_char + 1,
        end_char=passage.end_char + 1,
        start_byte=passage.start_byte + 1,
        end_byte=passage.end_byte + 1,
    )
    invalid = replace(
        fixture,
        snapshot=replace(
            fixture.snapshot,
            passages=tuple(
                shifted if row.passage_id == shifted.passage_id else row
                for row in fixture.snapshot.passages
            ),
        ),
    )

    with pytest.raises(AgenticProjectionError, match="source span"):
        _project(invalid)


def test_projection_rejects_a_grounded_document_without_a_recorded_search_position(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    invalid = replace(fixture, searches=())

    with pytest.raises(AgenticProjectionError, match="retrieval position"):
        _project(invalid)
