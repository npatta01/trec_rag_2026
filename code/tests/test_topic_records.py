from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import gc
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
from threading import Barrier, Event
import weakref

import pytest

import trec_rag.topic_records as topic_records_module
from trec_rag.chunking import TextChunk
from trec_rag.document_store import DocumentStore
from trec_rag.facet_evidence import (
    CandidateSubnarrative,
    ExtractiveCandidateRequest,
    ScoredPassage,
    SubnarrativeContext,
    _scoring_text_and_boundaries,
    extract_document_candidates,
)
from trec_rag.topic_records import (
    TopicRecords,
    TopicRecordsBuilder,
    TopicRecordsIntegrityError,
)
from trec_rag.topic_passage_search import (
    FocusedQuery,
    PassageSearchResult,
    SourceDocument,
    SourcePassage,
    passage_id,
)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _task3_passage_result(document_text: str = "exact body") -> PassageSearchResult:
    digest = _digest(document_text)
    passage_id = "p-" + "1" * 64
    document = SourceDocument("shared-doc", digest, 1, 1.0, passage_id, 1.0)
    passage = SourcePassage(
        passage_id,
        "shared-doc",
        digest,
        1,
        1.0,
        0,
        len(document_text),
        0,
        len(document_text.encode("utf-8")),
        digest,
        document_text,
        1.0,
        1,
        "cache-q1",
        _digest(document_text),
        {"backend": "task3-fixed", "implementation": "v1"},
    )
    return PassageSearchResult(
        FocusedQuery("q1", "find exact body", "facet-a", ("facet-b",)),
        "complete",
        None,
        1,
        1,
        1,
        1,
        (document,),
        (passage,),
        1,
        False,
    )


def _task3_identical_document_result(document_text: str = "identical body") -> PassageSearchResult:
    content_sha256 = _digest(document_text)
    chunker_identity = {"backend": "task3-fixed", "implementation": "v1"}
    documents: list[SourceDocument] = []
    passages: list[SourcePassage] = []
    for rank, docid in enumerate(("doc-a", "doc-b"), start=1):
        chunk = TextChunk(docid, f"{docid}:0000", document_text, 0, len(document_text))
        source_passage_id = passage_id(
            content_sha256,
            docid,
            chunk,
            chunker_identity,
        )
        documents.append(
            SourceDocument(docid, content_sha256, rank, float(rank), source_passage_id, 1.0)
        )
        passages.append(
            SourcePassage(
                source_passage_id,
                docid,
                content_sha256,
                rank,
                float(rank),
                0,
                len(document_text),
                0,
                len(document_text.encode("utf-8")),
                content_sha256,
                document_text,
                1.0,
                rank,
                "shared-score-cache-key",
                content_sha256,
                chunker_identity,
            )
        )
    return PassageSearchResult(
        FocusedQuery("q-identical", "find identical", "facet-a"),
        "complete",
        None,
        2,
        2,
        2,
        2,
        tuple(documents),
        tuple(passages),
        1,
        False,
    )


def _task3_builder(tmp_path: Path, *, topic_id: str = "t1", run_id: str = "r1"):
    store = DocumentStore(tmp_path / "objects")
    store.admit_text("exact body")
    builder = TopicRecordsBuilder(tmp_path / "topic", topic_id, store, run_id=run_id)
    return builder, store


def _task3_open_published(tmp_path: Path, store: DocumentStore):
    return TopicRecords.open(
        tmp_path / "topic" / "records.sqlite3",
        tmp_path / "topic" / "canonical" / "records-manifest.json",
        "t1",
        store,
    )


def _task3_handoff(*, run_id: str = "r1", passage_id: str = "p-" + "1" * 64):
    evidence = (topic_records_module.ResearcherEvidence("facet-a", passage_id, "relevant"),)
    return topic_records_module.ResearcherHandoff(
        run_id,
        "worker-1",
        0,
        evidence,
        (),
    )


def test_topic_records_persists_passage_search_without_repeating_document_text(tmp_path: Path) -> None:
    builder, store = _task3_builder(tmp_path)
    result = _task3_passage_result()
    builder.add_facets(
        (
            topic_records_module.FacetRecord("facet-a", "primary facet", "initial"),
            topic_records_module.FacetRecord("facet-b", "supporting facet", "initial"),
        )
    )
    builder.add_passage_search(result)
    builder.set_completion("complete", "coverage_sufficient")
    builder.publish({"run_id": "r1"})

    with _task3_open_published(tmp_path, store) as records:
        snapshot = records.passage_search("q1")
        assert snapshot.passages[0].text == "exact body"
        assert snapshot.query.supporting_subnarrative_ids == ("facet-b",)
        assert dict(snapshot.passages[0].chunker_identity) == {
            "backend": "task3-fixed",
            "implementation": "v1",
        }
    assert b"exact body" not in (tmp_path / "topic" / "records.sqlite3").read_bytes()


def test_ordinary_ledger_methods_do_not_accept_topic_id() -> None:
    assert list(inspect.signature(TopicRecordsBuilder.add_passage_search).parameters) == [
        "self",
        "result",
    ]
    assert list(inspect.signature(TopicRecords.topic_snapshot).parameters) == ["self"]


def test_builder_exposes_immutable_topic_and_run_identity(tmp_path: Path) -> None:
    builder, _store = _task3_builder(
        tmp_path,
        topic_id="topic-owned-by-builder",
        run_id="run-owned-by-builder",
    )

    assert builder.topic_id == "topic-owned-by-builder"
    assert builder.run_id == "run-owned-by-builder"
    with pytest.raises(AttributeError):
        builder.topic_id = "another-topic"
    with pytest.raises(AttributeError):
        builder.run_id = "another-run"


def test_passage_search_supporting_query_facets_and_researcher_facets_are_persisted(tmp_path: Path) -> None:
    builder, _store = _task3_builder(tmp_path)
    result = _task3_passage_result()
    builder.add_facets(
        (
            topic_records_module.FacetRecord("facet-a", "primary facet", "initial"),
            topic_records_module.FacetRecord("facet-b", "supporting facet", "initial"),
        )
    )
    builder.add_passage_search(result)
    builder.add_researcher_handoff(_task3_handoff())

    snapshot = builder.topic_snapshot()
    assert [facet.subnarrative_id for facet in snapshot.facets] == ["facet-a", "facet-b"]
    assert snapshot.queries[0].query.supporting_subnarrative_ids == ("facet-b",)
    assert snapshot.researcher_evidence[0].subnarrative_id == "facet-a"


def test_builder_worker_thread_can_mutate_and_snapshot_one_topic(
    tmp_path: Path,
) -> None:
    builder, _store = _task3_builder(tmp_path)

    def worker():
        builder.add_facets(
            (
                topic_records_module.FacetRecord(
                    "facet-a", "primary facet", "initial"
                ),
                topic_records_module.FacetRecord(
                    "facet-b", "supporting facet", "initial"
                ),
            )
        )
        builder.add_passage_search(_task3_passage_result())
        return builder.topic_snapshot()

    with ThreadPoolExecutor(max_workers=1) as executor:
        snapshot = executor.submit(worker).result()

    assert [facet.subnarrative_id for facet in snapshot.facets] == [
        "facet-a",
        "facet-b",
    ]
    assert [query.query_id for query in snapshot.queries] == ["q1"]
    assert snapshot.passages[0].text == "exact body"


def test_passage_search_preserves_admitted_facet_hash_for_later_candidate(tmp_path: Path) -> None:
    source, candidates = _unicode_candidates()
    store = DocumentStore(tmp_path / "objects")
    builder = TopicRecordsBuilder(tmp_path / "topic", "topic-unicode", store, run_id="r1")
    builder.bind_document("shared-doc", source)
    content_sha256 = _digest(source)
    passage_id = "p-" + "2" * 64
    passage = SourcePassage(
        passage_id,
        "shared-doc",
        content_sha256,
        1,
        1.0,
        0,
        len(source),
        0,
        len(source.encode("utf-8")),
        content_sha256,
        source,
        1.0,
        1,
        "cache-safety",
        content_sha256,
        {"backend": "task3-fixed", "implementation": "v1"},
    )
    result = PassageSearchResult(
        FocusedQuery("q-safety", "find safety", "safety"),
        "complete",
        None,
        1,
        1,
        1,
        1,
        (SourceDocument("shared-doc", content_sha256, 1, 1.0, passage_id, 1.0),),
        (passage,),
        1,
        False,
    )

    builder.add_facets((topic_records_module.FacetRecord("safety", "Safety evidence", "initial"),))
    builder.add_passage_search(result)
    builder.add_candidate(candidates[0])


def test_same_researcher_handoff_is_idempotent_but_changed_handoff_conflicts(tmp_path: Path) -> None:
    builder, _store = _task3_builder(tmp_path)
    result = _task3_passage_result()
    builder.add_facets(
        (
            topic_records_module.FacetRecord("facet-a", "primary facet", "initial"),
            topic_records_module.FacetRecord("facet-b", "supporting facet", "initial"),
        )
    )
    builder.add_passage_search(result)
    handoff = _task3_handoff()
    builder.add_researcher_handoff(handoff)
    builder.add_researcher_handoff(handoff)

    with pytest.raises(TopicRecordsIntegrityError, match="handoff.*conflict"):
        builder.add_researcher_handoff(replace(handoff, evidence=()))


def test_researcher_handoff_rejects_unknown_passage_and_wrong_run(tmp_path: Path) -> None:
    builder, _store = _task3_builder(tmp_path)
    builder.add_facets((topic_records_module.FacetRecord("facet-a", "primary facet", "initial"),))

    unknown = _task3_handoff(passage_id="p-" + "0" * 64)
    with pytest.raises(TopicRecordsIntegrityError, match="unknown passage"):
        builder.add_researcher_handoff(unknown)

    wrong_run = _task3_handoff(run_id="another-run", passage_id="p-" + "0" * 64)
    with pytest.raises(TopicRecordsIntegrityError, match="run"):
        builder.add_researcher_handoff(wrong_run)


def test_incomplete_topic_completion_publishes_and_reopens_with_valid_passages(tmp_path: Path) -> None:
    builder, store = _task3_builder(tmp_path)
    result = _task3_passage_result()
    builder.add_facets(
        (
            topic_records_module.FacetRecord("facet-a", "primary facet", "initial"),
            topic_records_module.FacetRecord("facet-b", "supporting facet", "initial"),
        )
    )
    builder.add_passage_search(result)
    builder.set_completion("incomplete", "budget_exhausted")
    builder.publish({"run_id": "r1"})

    with _task3_open_published(tmp_path, store) as records:
        snapshot = records.topic_snapshot()
    assert snapshot.status == "incomplete"
    assert snapshot.stopping_reason == "budget_exhausted"
    assert snapshot.passages[0].passage_id == result.passages[0].passage_id


def test_evidence_validation_failure_publishes_as_an_incomplete_topic(
    tmp_path: Path,
) -> None:
    builder, store = _task3_builder(tmp_path)
    result = _task3_passage_result()
    builder.add_facets(
        (
            topic_records_module.FacetRecord("facet-a", "primary facet", "initial"),
            topic_records_module.FacetRecord("facet-b", "supporting facet", "initial"),
        )
    )
    builder.add_passage_search(result)
    builder.set_completion("incomplete", "evidence_validation_failed")
    builder.publish({"run_id": "r1"})

    with _task3_open_published(tmp_path, store) as records:
        snapshot = records.topic_snapshot()
    assert snapshot.status == "incomplete"
    assert snapshot.stopping_reason == "evidence_validation_failed"
    assert snapshot.passages[0].passage_id == result.passages[0].passage_id


def test_topic_snapshot_materializes_queries_passages_and_completion(tmp_path: Path) -> None:
    builder, _store = _task3_builder(tmp_path)
    result = _task3_passage_result()
    builder.add_facets(
        (
            topic_records_module.FacetRecord("facet-a", "primary facet", "initial"),
            topic_records_module.FacetRecord("facet-b", "supporting facet", "initial"),
        )
    )
    builder.add_passage_search(result)
    builder.set_completion("complete", "coverage_sufficient")

    snapshot = builder.topic_snapshot()
    assert [row.query.query_id for row in snapshot.queries] == [result.query.query_id]
    assert [row.passage_id for row in snapshot.passages] == [row.passage_id for row in result.passages]
    assert snapshot.status == "complete"


def test_task3_hardening_identical_documents_keep_distinct_citable_passages(
    tmp_path: Path,
) -> None:
    store = DocumentStore(tmp_path / "objects")
    store.admit_text("identical body")
    builder = TopicRecordsBuilder(tmp_path / "topic", "t1", store, run_id="r1")
    builder.add_facets((topic_records_module.FacetRecord("facet-a", "primary", "initial"),))
    result = _task3_identical_document_result()
    builder.add_passage_search(result)
    for researcher_id, passage in zip(("worker-a", "worker-b"), result.passages, strict=True):
        builder.add_researcher_handoff(topic_records_module.ResearcherHandoff(
            "r1",
            researcher_id,
            0,
            (topic_records_module.ResearcherEvidence("facet-a", passage.passage_id, "relevant"),),
            (),
        ))

    builder_snapshot = builder.topic_snapshot()
    assert {(row.docid, row.passage_id) for row in builder_snapshot.passages} == {
        (row.docid, row.passage_id) for row in result.passages
    }
    assert {row.passage_id for row in builder_snapshot.researcher_evidence} == {
        row.passage_id for row in result.passages
    }
    builder.publish({"fixture": "identical-documents"})

    with TopicRecords.open(
        tmp_path / "topic" / "records.sqlite3",
        _manifest_path(tmp_path / "topic"),
        "t1",
        store,
    ) as records:
        reopened = records.topic_snapshot()
    assert {(row.docid, row.passage_id) for row in reopened.passages} == {
        (row.docid, row.passage_id) for row in result.passages
    }


def test_task3_hardening_researcher_facet_updates_roundtrip_exactly(tmp_path: Path) -> None:
    builder, store = _task3_builder(tmp_path)
    result = _task3_passage_result()
    builder.add_facets((
        topic_records_module.FacetRecord("facet-a", "primary facet", "initial"),
        topic_records_module.FacetRecord("facet-b", "supporting facet", "initial"),
    ))
    builder.add_passage_search(result)
    handoff = topic_records_module.ResearcherHandoff(
        "r1",
        "worker-1",
        2,
        (topic_records_module.ResearcherEvidence(
            "facet-a", result.passages[0].passage_id, "relevant"
        ),),
        (topic_records_module.FacetRecord(
            "facet-new", "newly discovered facet", "research_discovered"
        ),),
    )

    builder.add_researcher_handoff(handoff)
    assert builder.topic_snapshot().researcher_handoffs == (handoff,)
    builder.publish({"fixture": "facet-update"})

    with _task3_open_published(tmp_path, store) as records:
        assert records.topic_snapshot().researcher_handoffs == (handoff,)


def test_task3_hardening_run_identity_survives_without_handoffs(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "objects")
    builder = TopicRecordsBuilder(tmp_path / "topic", "t1", store, run_id="run-exact")

    published = builder.publish({"fixture": "run-identity"})

    assert published.receipt.run_id == "run-exact"
    assert json.loads(
        _manifest_path(tmp_path / "topic").read_text(encoding="utf-8")
    )["run_id"] == "run-exact"
    with TopicRecords.open(
        tmp_path / "topic" / "records.sqlite3",
        _manifest_path(tmp_path / "topic"),
        "t1",
        store,
    ) as records:
        assert records.run_id == "run-exact"
        assert records.receipt.run_id == "run-exact"


def test_task3_hardening_different_run_cannot_republish_same_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    first_store = DocumentStore(tmp_path / "objects-first")
    TopicRecordsBuilder(destination, "t1", first_store, run_id="run-a").publish(
        {"fixture": "run-conflict"}
    )
    second_builder = TopicRecordsBuilder(
        destination,
        "t1",
        DocumentStore(tmp_path / "objects-second"),
        run_id="run-b",
    )

    with pytest.raises(TopicRecordsIntegrityError, match="run|contradictory|identity"):
        second_builder.publish({"fixture": "run-conflict"})


class _ConstantScorer:
    def score_pairs(self, pairs):
        return tuple(1.0 + index / 10 for index, _ in enumerate(pairs))


def _candidates_for_source(
    source: str,
    *,
    document_id: str,
    passage_id: str,
):
    scoring_text, _ = _scoring_text_and_boundaries(source)
    request = ExtractiveCandidateRequest(
        topic_id="topic-unicode",
        document_id=document_id,
        source=source,
        document_sha256=_digest(source),
        scoring_text_sha256=_digest(scoring_text),
        subnarratives=(CandidateSubnarrative("safety", "Safety evidence"),),
        passages=(ScoredPassage(
            passage_id,
            "original",
            "topic-unicode",
            0,
            len(scoring_text),
            _digest(scoring_text),
            _digest(scoring_text),
            2.0,
            1,
        ),),
    )
    return source, extract_document_candidates(request, _ConstantScorer())


def _unicode_candidates():
    return _candidates_for_source(
        (
            "Heading: Café findings\n\n"
            "Dr. Ada measured\t3.5 meters. This result confirms safety.\n\n"
            "Ωmega evidence proves durability! Another detail ends.\n"
        ),
        document_id="shared-doc",
        passage_id="passage-1",
    )


def _partial_passage_candidates():
    source = (
        "Intro material is unrelated.\n\n"
        "First  fact.\nThis confirms safety.\n\n"
        "Trailing material is also unrelated.\n"
    )
    scoring_text, _ = _scoring_text_and_boundaries(source)
    chunk_text = "First fact. This confirms safety."
    scoring_start = scoring_text.index(chunk_text)
    scoring_end = scoring_start + len(chunk_text)
    request = ExtractiveCandidateRequest(
        topic_id="topic-unicode",
        document_id="partial-doc",
        source=source,
        document_sha256=_digest(source),
        scoring_text_sha256=_digest(scoring_text),
        subnarratives=(CandidateSubnarrative("safety", "Safety evidence"),),
        passages=(ScoredPassage(
            "partial-passage",
            "original",
            "topic-unicode",
            scoring_start,
            scoring_end,
            _digest(scoring_text),
            _digest(chunk_text),
            2.0,
            1,
        ),),
    )
    return source, extract_document_candidates(request, _ConstantScorer())


def test_task3_hardening_partial_candidate_passage_roundtrips(tmp_path: Path) -> None:
    source, candidates = _partial_passage_candidates()
    assert candidates
    assert candidates[0].passages[0].source_text != source
    passage = candidates[0].passages[0]
    exact_source_text = "First  fact.\nThis confirms safety."
    assert passage.source_text == exact_source_text
    assert passage.source_text_sha256 == _digest(exact_source_text)
    assert passage.chunk_text_sha256 == (
        "03c61fce26bc28f293456759f03d3c8480cd8248da3662daf14883afc0909206"
    )
    assert passage.source_text_sha256 != passage.chunk_text_sha256
    destination = tmp_path / "topic"
    store = DocumentStore(tmp_path / "objects")
    builder = TopicRecordsBuilder(destination, "topic-unicode", store, run_id="r1")
    builder.bind_document("partial-doc", source)
    for candidate in candidates:
        builder.add_candidate(candidate)
    builder.publish({"fixture": "partial-passage"})

    connection = sqlite3.connect(destination / "records.sqlite3")
    try:
        assert connection.execute(
            "SELECT source_text_sha256, scoring_text_sha256 FROM passage "
            "WHERE passage_id='partial-passage'"
        ).fetchone() == (passage.source_text_sha256, passage.source_text_sha256)
    finally:
        connection.close()

    with TopicRecords.open(
        destination / "records.sqlite3",
        _manifest_path(destination),
        "topic-unicode",
        store,
    ) as records:
        loaded = records.load_candidates()

    assert tuple(loaded.values()) == tuple(sorted(
        candidates,
        key=lambda row: (row.subnarrative_id, row.candidate_nugget_id),
    ))


def test_task3_hardening_shared_search_passage_is_reused_by_candidate(
    tmp_path: Path,
) -> None:
    source, candidates = _candidates_for_source(
        "First\tfact. This confirms safety.",
        document_id="shared-doc",
        passage_id="shared-passage",
    )
    candidate = candidates[0]
    provenance = candidate.passages[0]
    assert provenance.source_text_sha256 != provenance.chunk_text_sha256
    store = DocumentStore(tmp_path / "objects")
    builder = TopicRecordsBuilder(
        tmp_path / "topic", "topic-unicode", store, run_id="r1"
    )
    builder.bind_document("shared-doc", source)
    builder.add_facets((
        topic_records_module.FacetRecord("safety", "Safety evidence", "initial"),
    ))
    shared_passage = SourcePassage(
        provenance.passage_id,
        "shared-doc",
        candidate.document_sha256,
        1,
        1.0,
        provenance.source_start_char,
        provenance.source_end_char,
        provenance.source_start_byte,
        provenance.source_end_byte,
        provenance.source_text_sha256,
        provenance.source_text,
        provenance.cross_encoder_score,
        provenance.cross_encoder_rank,
        "shared-cache-key",
        provenance.source_text_sha256,
        {"backend": "real-shared-chunker", "implementation": "v1"},
    )
    builder.add_passage_search(PassageSearchResult(
        FocusedQuery("q-shared", "find safety", "safety"),
        "complete",
        None,
        1,
        1,
        1,
        1,
        (SourceDocument(
            "shared-doc",
            candidate.document_sha256,
            1,
            1.0,
            provenance.passage_id,
            provenance.cross_encoder_score,
        ),),
        (shared_passage,),
        1,
        False,
    ))

    builder.add_candidate(candidate)
    assert dict(builder.topic_snapshot().passages[0].chunker_identity) == {
        "backend": "real-shared-chunker",
        "implementation": "v1",
    }
    builder.publish({"fixture": "shared-candidate-reuse"})

    with TopicRecords.open(
        tmp_path / "topic" / "records.sqlite3",
        _manifest_path(tmp_path / "topic"),
        "topic-unicode",
        store,
    ) as records:
        loaded = records.load_candidates({("safety", candidate.candidate_nugget_id)})
    assert loaded[("safety", candidate.candidate_nugget_id)] == candidate


def test_task3_hardening_semantic_hash_ignores_document_admission_order(
    tmp_path: Path,
) -> None:
    first_source, first_candidates = _candidates_for_source(
        "First fact. This confirms safety.",
        document_id="doc-a",
        passage_id="passage-a",
    )
    second_source, second_candidates = _candidates_for_source(
        "Second fact. This confirms safety.",
        document_id="doc-b",
        passage_id="passage-b",
    )
    documents = (
        ("doc-a", first_source, first_candidates),
        ("doc-b", second_source, second_candidates),
    )

    def build(destination: Path, store_root: Path, rows) -> str:
        builder = TopicRecordsBuilder(
            destination,
            "topic-unicode",
            DocumentStore(store_root),
            run_id="r1",
        )
        for docid, source, candidates in rows:
            builder.bind_document(docid, source)
            for candidate in candidates:
                builder.add_candidate(candidate)
        return builder.publish({"fixture": "document-order"}).semantic_sha256

    first_semantic = build(tmp_path / "first", tmp_path / "objects-first", documents)
    second_semantic = build(
        tmp_path / "second", tmp_path / "objects-second", tuple(reversed(documents))
    )

    assert first_semantic == second_semantic


def test_same_docid_can_bind_different_bodies_in_different_topic_records(
    tmp_path: Path,
) -> None:
    first_store = DocumentStore(tmp_path / "objects-1")
    second_store = DocumentStore(tmp_path / "objects-2")
    first = TopicRecordsBuilder(tmp_path / "topic-1", "topic-1", first_store, run_id="r1")
    second = TopicRecordsBuilder(tmp_path / "topic-2", "topic-2", second_store, run_id="r1")

    first.bind_document("shared-doc", "topic one body")
    second.bind_document("shared-doc", "topic two body")

    first_receipt = first.publish({})
    second_receipt = second.publish({})

    assert first_receipt.topic_id == "topic-1"
    assert second_receipt.topic_id == "topic-2"
    assert first_receipt.document_sha256s != second_receipt.document_sha256s
    TopicRecords.open(
        tmp_path / "topic-1" / "records.sqlite3",
        tmp_path / "topic-1" / "canonical" / "records-manifest.json",
        "topic-1",
        first_store,
    )
    TopicRecords.open(
        tmp_path / "topic-2" / "records.sqlite3",
        tmp_path / "topic-2" / "canonical" / "records-manifest.json",
        "topic-2",
        second_store,
    )


def test_same_topic_docid_cannot_bind_contradictory_body(tmp_path: Path) -> None:
    builder = TopicRecordsBuilder(
        tmp_path / "topic", "topic-1", DocumentStore(tmp_path / "objects"), run_id="r1"
    )
    builder.bind_document("shared-doc", "first body")

    with pytest.raises(TopicRecordsIntegrityError, match="document binding"):
        builder.bind_document("shared-doc", "different body")


def test_reopens_exact_candidate_roles_from_offsets_and_hashes(tmp_path: Path) -> None:
    source, candidates = _unicode_candidates()
    first_store = DocumentStore(tmp_path / "objects-1")
    second_store = DocumentStore(tmp_path / "objects-2")
    builder = TopicRecordsBuilder(tmp_path / "topic", "topic-unicode", first_store, run_id="r1")
    builder.bind_document("shared-doc", source, expected_sha256=_digest(source))
    for candidate in candidates:
        builder.add_candidate(candidate)
    builder.publish({"fixture": "unicode"})

    second_store.admit_text(source)
    records = TopicRecords.open(
        tmp_path / "topic" / "records.sqlite3",
        tmp_path / "topic" / "canonical" / "records-manifest.json",
        "topic-unicode",
        second_store,
    )
    required = frozenset(
        (candidate.subnarrative_id, candidate.candidate_nugget_id)
        for candidate in candidates
    )
    loaded = records.load_candidates(required)

    assert tuple(loaded[key] for key in required)  # every requested key resolved
    for candidate in candidates:
        actual = loaded[(candidate.subnarrative_id, candidate.candidate_nugget_id)]
        assert actual == candidate
        assert actual.text == source[actual.evidence_sentences[0].start_char:actual.evidence_sentences[-1].end_char]
        assert all(
            source.encode("utf-8")[span.start_byte:span.end_byte].decode("utf-8") == span.text
            for span in (
                *actual.evidence_sentences,
                actual.matched_paragraph,
                *(span for span in (actual.context_before, actual.context_after) if span is not None),
            )
        )
        assert all(
            passage.source_text == source[passage.source_start_char:passage.source_end_char]
            and passage.source_text_sha256 == _digest(passage.source_text)
            for passage in actual.passages
        )

    database_bytes = (tmp_path / "topic" / "records.sqlite3").read_bytes()
    manifest_bytes = _manifest_path(tmp_path / "topic").read_bytes()
    assert all(candidate.text.encode("utf-8") not in database_bytes for candidate in candidates)
    assert str(tmp_path / "objects-1").encode() not in database_bytes
    assert str(tmp_path / "objects-1").encode() not in manifest_bytes


def _build_unicode_topic(
    destination: Path,
    store_root: Path,
    candidates,
    *,
    reverse: bool = False,
):
    source, _ = _unicode_candidates()
    store = DocumentStore(store_root)
    builder = TopicRecordsBuilder(destination, "topic-unicode", store, run_id="r1")
    builder.bind_document("shared-doc", source)
    for candidate in reversed(candidates) if reverse else candidates:
        builder.add_candidate(candidate)
    receipt = builder.publish({"fixture": "unicode"})
    return receipt, TopicRecords.open(
        destination / "records.sqlite3",
        destination / "canonical" / "records-manifest.json",
        "topic-unicode",
        store,
    )


def _prepared_unicode_builder(destination: Path, store_root: Path):
    source, candidates = _unicode_candidates()
    builder = TopicRecordsBuilder(
        destination,
        "topic-unicode",
        DocumentStore(store_root),
        run_id="r1",
    )
    builder.bind_document("shared-doc", source)
    for candidate in candidates:
        builder.add_candidate(candidate)
    return builder


def test_add_candidate_rejects_validly_formatted_forged_candidate_nugget_id(
    tmp_path: Path,
) -> None:
    source, candidates = _unicode_candidates()
    builder = TopicRecordsBuilder(
        tmp_path / "topic",
        "topic-unicode",
        DocumentStore(tmp_path / "objects"),
        run_id="r1",
    )
    builder.bind_document("shared-doc", source)
    forged = replace(candidates[0], candidate_nugget_id="ecn1_" + "0" * 64)

    with pytest.raises(TopicRecordsIntegrityError, match="candidate.*identity"):
        builder.add_candidate(forged)


def test_v3_deduplicates_complete_passages_into_compact_same_document_links(
    tmp_path: Path,
) -> None:
    source, candidates = _unicode_candidates()
    destination = tmp_path / "topic"
    builder = TopicRecordsBuilder(
        destination,
        "topic-unicode",
        DocumentStore(tmp_path / "objects"),
        run_id="r1",
    )
    builder.bind_document("shared-doc", source)
    for candidate in candidates:
        builder.add_candidate(candidate)
    published = builder.publish({"fixture": "unicode"})

    receipt = getattr(published, "receipt", published)
    assert receipt.row_counts["passage"] == 1
    assert receipt.row_counts["candidate_passage_link"] == len(candidates)
    assert "candidate_passage" not in receipt.row_counts
    connection = sqlite3.connect(destination / "records.sqlite3")
    try:
        columns = tuple(
            row[1]
            for row in connection.execute("PRAGMA table_xinfo(candidate_passage_link)")
        )
        assert columns == (
            "candidate_pk", "document_pk", "ordinal", "passage_pk", "query_id"
        )
        assert connection.execute(
            "SELECT COUNT(DISTINCT passage_pk) FROM candidate_passage_link"
        ).fetchone()[0] == 1
    finally:
        connection.close()


def test_candidate_only_path_retains_legacy_query_fallback(tmp_path: Path) -> None:
    source, candidates = _unicode_candidates()
    destination = tmp_path / "topic"
    builder = TopicRecordsBuilder(
        destination,
        "topic-unicode",
        DocumentStore(tmp_path / "objects"),
        run_id="r1",
    )
    builder.bind_document("shared-doc", source)
    builder.add_candidate(candidates[0])
    builder.publish({"fixture": "candidate-only"})

    connection = sqlite3.connect(destination / "records.sqlite3")
    try:
        query_id = connection.execute(
            "SELECT query_id FROM query_identity"
        ).fetchone()[0]
        assert json.loads(query_id) == {
            "base_query_id": "topic-unicode",
            "docid": "shared-doc",
        }
        assert connection.execute(
            "SELECT query_id FROM candidate_passage_link"
        ).fetchone()[0] == query_id
    finally:
        connection.close()


def test_v2_relational_constraints_reject_cross_document_links_and_passage_ids(
    tmp_path: Path,
) -> None:
    first_source, first_candidates = _unicode_candidates()
    second_source, second_candidates = _candidates_for_source(
        "Heading: Deux\n\nFirst fact. This confirms safety.\n",
        document_id="second-doc",
        passage_id="passage-2",
    )
    destination = tmp_path / "topic"
    store = DocumentStore(tmp_path / "objects")
    builder = TopicRecordsBuilder(destination, "topic-unicode", store, run_id="r1")
    for docid, source, candidates in (
        ("shared-doc", first_source, first_candidates),
        ("second-doc", second_source, second_candidates),
    ):
        builder.bind_document(docid, source)
        for candidate in candidates:
            builder.add_candidate(candidate)
    builder.publish({"fixture": "two-documents"})

    connection = sqlite3.connect(destination / "records.sqlite3")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        candidate_pk, candidate_document_pk = connection.execute(
            "SELECT c.candidate_pk, c.document_pk FROM candidate AS c "
            "JOIN document_binding AS d ON d.document_pk=c.document_pk "
            "WHERE d.docid='shared-doc' LIMIT 1"
        ).fetchone()
        foreign_passage_pk = connection.execute(
            "SELECT p.passage_pk FROM passage AS p "
            "JOIN document_binding AS d ON d.document_pk=p.document_pk "
            "WHERE d.docid='second-doc' LIMIT 1"
        ).fetchone()[0]
        source_query_id = connection.execute(
            "SELECT query_id FROM candidate_passage_link "
            "WHERE candidate_pk=? ORDER BY ordinal LIMIT 1",
            (candidate_pk,),
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                "INSERT INTO candidate_passage_link VALUES (?, ?, ?, ?, ?)",
                (candidate_pk, candidate_document_pk, 999, foreign_passage_pk, source_query_id),
            )

        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            connection.execute(
                "INSERT INTO passage "
                "(document_pk, passage_id, source_start_char, source_end_char, "
                "source_start_byte, source_end_byte, source_text_sha256, "
                "scoring_text_sha256, chunker_identity_json) "
                "SELECT document_pk, passage_id, source_start_char, source_end_char, "
                "source_start_byte, source_end_byte, source_text_sha256, "
                "scoring_text_sha256, chunker_identity_json "
                "FROM passage LIMIT 1"
            )
    finally:
        connection.rollback()
        connection.close()


def test_publish_returns_non_pickleable_process_local_validation_session(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    builder = _prepared_unicode_builder(destination, store_root)

    published = builder.publish({"fixture": "unicode"})

    assert isinstance(published, topic_records_module.PublishedTopicRecords)
    assert isinstance(
        published.validation_session,
        topic_records_module.ValidatedTopicRecords,
    )
    with pytest.raises(TypeError, match="serial|pickle|process"):
        pickle.dumps(published.validation_session)

    records = TopicRecords.open(
        destination / "records.sqlite3",
        _manifest_path(destination),
        "topic-unicode",
        DocumentStore(store_root),
        validation_session=published.validation_session,
    )
    try:
        assert records.validation_session is published.validation_session
        assert records.receipt == published.receipt
    finally:
        records.close()


def test_validation_session_constructor_and_subclass_forgery_are_rejected() -> None:
    args = (
        object(),
        os.getpid(),
        "validator",
        "schema",
        "topic-unicode",
        "0" * 64,
        0,
        "0" * 64,
        0,
        "0" * 64,
        (),
    )

    with pytest.raises(TypeError, match="process-local|internal|mint"):
        topic_records_module.ValidatedTopicRecords(*args)

    class ForgedValidationSession(topic_records_module.ValidatedTopicRecords):
        pass

    with pytest.raises(TypeError, match="process-local|internal|mint"):
        ForgedValidationSession(*args)


def test_validation_session_has_only_weakref_slot_and_rejects_copying(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    session = published.validation_session

    assert topic_records_module.ValidatedTopicRecords.__slots__ == ("__weakref__",)
    with pytest.raises(TypeError, match="serial|pickle|process"):
        copy.copy(session)
    with pytest.raises(TypeError, match="serial|pickle|process"):
        copy.deepcopy(session)


def test_validation_session_rejects_pid_forgery_after_fork(tmp_path: Path) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("fork is unavailable on this platform")
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        outcome = b"unexpected"
        try:
            try:
                object.__setattr__(
                    published.validation_session,
                    "process_id",
                    os.getpid(),
                )
            except BaseException:
                pass
            records = TopicRecords.open(
                destination / "records.sqlite3",
                _manifest_path(destination),
                "topic-unicode",
                DocumentStore(store_root),
                validation_session=published.validation_session,
            )
            records.close()
            outcome = b"accepted"
        except TopicRecordsIntegrityError:
            outcome = b"rejected"
        except BaseException:
            outcome = b"unexpected"
        os.write(write_fd, outcome)
        os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    _, status = os.waitpid(child_pid, 0)
    outcome = os.read(read_fd, 32)
    os.close(read_fd)
    assert os.WIFEXITED(status)
    assert outcome == b"rejected"


def test_validation_session_supports_concurrent_same_process_rebinds(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )

    def rebind(_: int) -> bool:
        records = TopicRecords.open(
            destination / "records.sqlite3",
            _manifest_path(destination),
            "topic-unicode",
            DocumentStore(store_root),
            validation_session=published.validation_session,
        )
        try:
            return records.validation_session is published.validation_session
        finally:
            records.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert tuple(executor.map(rebind, range(8))) == (True,) * 8


def test_validation_session_registry_does_not_keep_token_alive(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    token_ref = weakref.ref(published.validation_session)

    del published
    gc.collect()

    assert token_ref() is None


def test_identical_publications_compare_equal_with_distinct_validation_tokens(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    first = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    second = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )

    assert first.validation_session is not second.validation_session
    assert first == second


def test_topic_records_constructor_is_internal_only(tmp_path: Path) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    database = destination / "records.sqlite3"
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        manifest_value = json.loads(_manifest_path(destination).read_text(encoding="utf-8"))
        with pytest.raises(TypeError, match="internal|constructor|validated"):
            TopicRecords(
                connection,
                database,
                _manifest_path(destination),
                manifest_value,
                DocumentStore(store_root),
                published.validation_session,
            )
    finally:
        connection.close()


def test_validation_session_rejects_equivalent_but_different_manifest_bytes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    manifest_path = _manifest_path(destination)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TopicRecordsIntegrityError, match="validation session|manifest bytes"):
        TopicRecords.open(
            destination / "records.sqlite3",
            manifest_path,
            "topic-unicode",
            DocumentStore(store_root),
            validation_session=published.validation_session,
        )

    records = TopicRecords.open(
        destination / "records.sqlite3",
        manifest_path,
        "topic-unicode",
        DocumentStore(store_root),
    )
    records.close()


def test_validation_session_rejects_changed_database_and_cas_bytes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    database_path = destination / "records.sqlite3"
    database_bytes = database_path.read_bytes()
    changed_database = bytearray(database_bytes)
    changed_database[-1] ^= 1
    database_path.write_bytes(changed_database)
    with pytest.raises(TopicRecordsIntegrityError, match="validation session|database bytes"):
        TopicRecords.open(
            database_path,
            _manifest_path(destination),
            "topic-unicode",
            DocumentStore(store_root),
            validation_session=published.validation_session,
        )
    database_path.write_bytes(database_bytes)

    digest = published.document_sha256s[0]
    object_path = store_root / "sha256" / digest[:2] / f"{digest}.utf8"
    object_bytes = object_path.read_bytes()
    object_path.write_bytes(object_bytes + b"changed")
    with pytest.raises(TopicRecordsIntegrityError, match="validation session|document|CAS"):
        TopicRecords.open(
            database_path,
            _manifest_path(destination),
            "topic-unicode",
            DocumentStore(store_root),
            validation_session=published.validation_session,
        )


def test_validation_session_rejects_object_new_capability_lookalike(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    forged = object.__new__(topic_records_module.ValidatedTopicRecords)

    with pytest.raises(TopicRecordsIntegrityError, match="validation session"):
        TopicRecords.open(
            destination / "records.sqlite3",
            _manifest_path(destination),
            "topic-unicode",
            DocumentStore(store_root),
            validation_session=forged,
        )


def test_validation_session_rejects_dataclasses_replace_proof_forgery(
    tmp_path: Path,
) -> None:
    published = _prepared_unicode_builder(
        tmp_path / "topic", tmp_path / "objects"
    ).publish({"fixture": "unicode"})

    with pytest.raises(TypeError, match="dataclass"):
        replace(published.validation_session, document_receipts=())


def test_forged_validation_session_cannot_bypass_missing_cas_object(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    forged = object.__new__(topic_records_module.ValidatedTopicRecords)
    digest = published.document_sha256s[0]
    object_path = store_root / "sha256" / digest[:2] / f"{digest}.utf8"
    object_path.unlink()

    with pytest.raises(TopicRecordsIntegrityError, match="validation session|document|CAS"):
        TopicRecords.open(
            destination / "records.sqlite3",
            _manifest_path(destination),
            "topic-unicode",
            DocumentStore(store_root),
            validation_session=forged,
        )


def test_rejected_conflicting_passage_cannot_publish_candidate_leak(
    tmp_path: Path,
) -> None:
    source, candidates = _unicode_candidates()
    assert len(candidates) >= 2
    destination = tmp_path / "topic"
    builder = TopicRecordsBuilder(
        destination,
        "topic-unicode",
        DocumentStore(tmp_path / "objects"),
        run_id="r1",
    )
    builder.bind_document("shared-doc", source)
    builder.add_candidate(candidates[0])
    original_passage = candidates[1].passages[0]
    unique_passage = replace(original_passage, passage_id="passage-unique")
    conflicting = replace(
        candidates[1],
        passages=(
            unique_passage,
            replace(
                original_passage,
                cross_encoder_score=original_passage.cross_encoder_score + 1.0,
            ),
        ),
    )

    with pytest.raises(TopicRecordsIntegrityError, match="passage provenance"):
        builder.add_candidate(conflicting)

    builder.add_candidate(candidates[1])
    published = builder.publish({"fixture": "unicode"})

    clean_receipt, clean_records = _build_unicode_topic(
        tmp_path / "clean", tmp_path / "clean-objects", candidates[:2]
    )
    clean_records.close()

    assert published.semantic_sha256 == clean_receipt.semantic_sha256
    assert published.row_counts == clean_receipt.row_counts
    connection = sqlite3.connect(destination / "records.sqlite3")
    try:
        candidate_ids = {
            row[0]
            for row in connection.execute(
                "SELECT candidate_nugget_id FROM candidate"
            )
        }
        passage_map = tuple(connection.execute(
            "SELECT p.passage_id, qp.raw_logit, qp.passage_rank "
            "FROM passage AS p JOIN query_passage AS qp "
            "ON qp.passage_pk=p.passage_pk ORDER BY p.passage_id"
        ))
    finally:
        connection.close()
    clean_connection = sqlite3.connect(tmp_path / "clean" / "records.sqlite3")
    try:
        clean_passage_map = tuple(clean_connection.execute(
            "SELECT p.passage_id, qp.raw_logit, qp.passage_rank "
            "FROM passage AS p JOIN query_passage AS qp "
            "ON qp.passage_pk=p.passage_pk ORDER BY p.passage_id"
        ))
    finally:
        clean_connection.close()
    assert candidate_ids == {
        candidate.candidate_nugget_id for candidate in candidates[:2]
    }
    assert passage_map == clean_passage_map


def test_untrusted_open_derives_geometry_once_per_bound_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_source, first_candidates = _unicode_candidates()
    second_source, second_candidates = _candidates_for_source(
        (
            "Heading: Résumé findings\n\n"
            "Dr. Bea measured 4 meters. This result confirms stability.\n\n"
            "Λambda evidence proves resilience! Another fact ends.\n"
        ),
        document_id="second-doc",
        passage_id="passage-2",
    )
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    store = DocumentStore(store_root)
    builder = TopicRecordsBuilder(destination, "topic-unicode", store, run_id="r1")
    for docid, source, candidates in (
        ("shared-doc", first_source, first_candidates),
        ("second-doc", second_source, second_candidates),
    ):
        builder.bind_document(docid, source)
        for candidate in candidates:
            builder.add_candidate(candidate)
    builder.publish({"fixture": "two-documents"})

    admitted: list[str] = []
    original_admit = topic_records_module.DocumentGeometryIndex.admit

    def counting_admit(index, content_sha256: str, source: str):
        admitted.append(content_sha256)
        return original_admit(index, content_sha256, source)

    monkeypatch.setattr(
        topic_records_module.DocumentGeometryIndex,
        "admit",
        counting_admit,
    )
    records = TopicRecords.open(
        destination / "records.sqlite3",
        _manifest_path(destination),
        "topic-unicode",
        store,
    )
    try:
        assert len(admitted) == 2
        assert set(admitted) == {_digest(first_source), _digest(second_source)}
    finally:
        records.close()


def test_untrusted_open_deep_validates_each_shared_passage_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, candidates = _unicode_candidates()
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    _build_unicode_topic(destination, store_root, candidates)[1].close()
    checked: list[str] = []
    original_check = topic_records_module._check_passage_provenance

    def counting_check(geometry, **kwargs):
        checked.append(kwargs["source_text_sha256"])
        return original_check(geometry, **kwargs)

    monkeypatch.setattr(
        topic_records_module,
        "_check_passage_provenance",
        counting_check,
    )
    records = TopicRecords.open(
        destination / "records.sqlite3",
        _manifest_path(destination),
        "topic-unicode",
        DocumentStore(store_root),
    )
    records.close()

    assert len(checked) == 1


def test_untrusted_whole_database_validation_never_reconstructs_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, candidates = _unicode_candidates()
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    _build_unicode_topic(destination, store_root, candidates)[1].close()

    def unexpected_reconstruction(*args, **kwargs):
        raise AssertionError("whole-database validation reconstructed a candidate")

    monkeypatch.setattr(
        topic_records_module,
        "_reconstruct_candidate",
        unexpected_reconstruction,
    )
    records = TopicRecords.open(
        destination / "records.sqlite3",
        _manifest_path(destination),
        "topic-unicode",
        DocumentStore(store_root),
    )
    records.close()


def test_open_rejects_explicit_v3_manifest(tmp_path: Path) -> None:
    _, candidates = _unicode_candidates()
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    _build_unicode_topic(destination, store_root, candidates)[1].close()
    _rewrite_manifest(_manifest_path(destination), schema_version="topic-records-v3")

    with pytest.raises(TopicRecordsIntegrityError, match="schema"):
        TopicRecords.open(
            destination / "records.sqlite3",
            _manifest_path(destination),
            "topic-unicode",
            DocumentStore(store_root),
        )


def test_selection_and_semantic_seal_are_stable_across_insert_order(tmp_path: Path) -> None:
    _, candidates = _unicode_candidates()
    first_receipt, first = _build_unicode_topic(
        tmp_path / "first", tmp_path / "objects-first", candidates
    )
    second_receipt, second = _build_unicode_topic(
        tmp_path / "second", tmp_path / "objects-second", candidates, reverse=True
    )
    context = SubnarrativeContext(
        "topic-unicode", "Explain safety.", "safety", "Safety evidence"
    )

    first_pool = first.selection_pool(context, 2)
    second_pool = second.selection_pool(context, 2)

    assert first_receipt.semantic_sha256 == second_receipt.semantic_sha256
    assert first_pool == second_pool
    assert first_receipt.database_sha256 == sha256(
        (tmp_path / "first" / "records.sqlite3").read_bytes()
    ).hexdigest()
    assert second_receipt.database_sha256 == sha256(
        (tmp_path / "second" / "records.sqlite3").read_bytes()
    ).hexdigest()


def test_score_change_changes_semantic_seal_and_offset_tamper_fails_at_add(
    tmp_path: Path,
) -> None:
    source, candidates = _unicode_candidates()
    changed = candidates[0]
    changed_evidence = tuple(
        replace(sentence, cross_encoder_score=sentence.cross_encoder_score + 1.0)
        for sentence in changed.evidence_sentences
    )
    changed = replace(
        changed,
        evidence_sentences=changed_evidence,
        sentence_cross_encoder_score=changed.sentence_cross_encoder_score + 1.0,
    )
    first_receipt, _ = _build_unicode_topic(
        tmp_path / "original", tmp_path / "objects-original", candidates
    )
    changed_receipt, _ = _build_unicode_topic(
        tmp_path / "changed", tmp_path / "objects-changed", (changed, *candidates[1:])
    )
    assert first_receipt.semantic_sha256 != changed_receipt.semantic_sha256

    store = DocumentStore(tmp_path / "objects-tampered")
    builder = TopicRecordsBuilder(tmp_path / "tampered", "topic-unicode", store, run_id="r1")
    builder.bind_document("shared-doc", source)
    bad_span = replace(
        candidates[0].evidence_sentences[0],
        start_char=candidates[0].evidence_sentences[0].start_char + 1,
    )
    with pytest.raises(TopicRecordsIntegrityError, match="source span"):
        builder.add_candidate(replace(candidates[0], evidence_sentences=(bad_span,)))


def _manifest_path(destination: Path) -> Path:
    return destination / "canonical" / "records-manifest.json"


def _rewrite_manifest(path: Path, **changes: object) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(changes)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _refresh_database_receipt(destination: Path) -> None:
    database_bytes = (destination / "records.sqlite3").read_bytes()
    _rewrite_manifest(
        _manifest_path(destination),
        database_sha256=sha256(database_bytes).hexdigest(),
        database_bytes=len(database_bytes),
    )


def _recreate_stage_seal(connection: sqlite3.Connection, declaration: str) -> None:
    connection.execute("ALTER TABLE stage_seal RENAME TO stage_seal_old")
    connection.execute(declaration)
    connection.execute("INSERT INTO stage_seal SELECT * FROM stage_seal_old")
    connection.execute("DROP TABLE stage_seal_old")


def test_task3_hardening_removed_status_check_is_rejected(tmp_path: Path) -> None:
    builder, store = _task3_builder(tmp_path)
    builder.add_facets((
        topic_records_module.FacetRecord("facet-a", "primary facet", "initial"),
        topic_records_module.FacetRecord("facet-b", "supporting facet", "initial"),
    ))
    builder.add_passage_search(_task3_passage_result())
    builder.set_completion("complete", "coverage_sufficient")
    builder.publish({"fixture": "check-drift"})
    destination = tmp_path / "topic"
    connection = sqlite3.connect(destination / "records.sqlite3")
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute("ALTER TABLE topic_completion RENAME TO topic_completion_old")
    connection.execute("""
        CREATE TABLE topic_completion (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            status TEXT NOT NULL,
            stopping_reason TEXT NOT NULL
        ) STRICT
    """)
    connection.execute("INSERT INTO topic_completion SELECT * FROM topic_completion_old")
    connection.execute("DROP TABLE topic_completion_old")
    connection.commit()
    connection.close()
    _refresh_database_receipt(destination)

    with pytest.raises(TopicRecordsIntegrityError, match="schema"):
        TopicRecords.open(
            destination / "records.sqlite3",
            _manifest_path(destination),
            "t1",
            store,
        )


@pytest.mark.parametrize(
    "drift",
    ("non_strict", "missing_primary_key", "extra_unique", "foreign_key_action"),
)
def test_open_rejects_self_consistent_database_with_schema_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    _, candidates = _unicode_candidates()
    destination = tmp_path / drift
    store_root = tmp_path / f"objects-{drift}"
    _, records = _build_unicode_topic(destination, store_root, candidates)
    records.close()
    connection = sqlite3.connect(destination / "records.sqlite3")
    connection.execute("PRAGMA foreign_keys=OFF")
    if drift == "non_strict":
        _recreate_stage_seal(connection, """
            CREATE TABLE stage_seal (
                stage TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                semantic_sha256 TEXT NOT NULL,
                identity_json TEXT NOT NULL,
                row_counts_json TEXT NOT NULL
            )
        """)
    elif drift == "missing_primary_key":
        _recreate_stage_seal(connection, """
            CREATE TABLE stage_seal (
                stage TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                semantic_sha256 TEXT NOT NULL,
                identity_json TEXT NOT NULL,
                row_counts_json TEXT NOT NULL
            ) STRICT
        """)
    elif drift == "extra_unique":
        connection.execute(
            "CREATE UNIQUE INDEX unexpected_candidate_unique "
            "ON candidate(candidate_nugget_id, candidate_kind)"
        )
    else:
        connection.execute(
            "ALTER TABLE candidate_passage_link RENAME TO candidate_passage_link_old"
        )
        connection.execute("""
            CREATE TABLE candidate_passage_link (
                candidate_pk INTEGER NOT NULL,
                document_pk INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                passage_pk INTEGER NOT NULL,
                PRIMARY KEY(candidate_pk, ordinal),
                UNIQUE(candidate_pk, passage_pk),
                FOREIGN KEY(candidate_pk, document_pk)
                    REFERENCES candidate(candidate_pk, document_pk)
                    ON DELETE CASCADE,
                FOREIGN KEY(passage_pk, document_pk)
                    REFERENCES passage(passage_pk, document_pk)
            ) STRICT
        """)
        connection.execute(
            "INSERT INTO candidate_passage_link "
            "(candidate_pk, document_pk, ordinal, passage_pk) "
            "SELECT candidate_pk, document_pk, ordinal, passage_pk "
            "FROM candidate_passage_link_old"
        )
        connection.execute("DROP TABLE candidate_passage_link_old")
    connection.commit()
    connection.close()
    _refresh_database_receipt(destination)

    with pytest.raises(TopicRecordsIntegrityError, match="schema"):
        TopicRecords.open(
            destination / "records.sqlite3",
            _manifest_path(destination),
            "topic-unicode",
            DocumentStore(store_root),
        )


def test_open_rejects_database_manifest_topic_and_content_corruption(tmp_path: Path) -> None:
    _, candidates = _unicode_candidates()

    corrupt_db = tmp_path / "corrupt-db"
    _build_unicode_topic(corrupt_db, tmp_path / "objects-db", candidates)
    database = corrupt_db / "records.sqlite3"
    body = bytearray(database.read_bytes())
    body[-1] ^= 1
    database.write_bytes(body)
    with pytest.raises(TopicRecordsIntegrityError, match="database byte receipt"):
        TopicRecords.open(database, _manifest_path(corrupt_db), "topic-unicode", DocumentStore(tmp_path / "objects-db"))

    corrupt_manifest = tmp_path / "corrupt-manifest"
    _, candidates = _unicode_candidates()
    _build_unicode_topic(corrupt_manifest, tmp_path / "objects-manifest", candidates)
    _rewrite_manifest(_manifest_path(corrupt_manifest), semantic_sha256="0" * 64)
    with pytest.raises(TopicRecordsIntegrityError, match="semantic"):
        TopicRecords.open(
            corrupt_manifest / "records.sqlite3",
            _manifest_path(corrupt_manifest),
            "topic-unicode",
            DocumentStore(tmp_path / "objects-manifest"),
        )

    wrong_topic = tmp_path / "wrong-topic"
    _, candidates = _unicode_candidates()
    _build_unicode_topic(wrong_topic, tmp_path / "objects-topic", candidates)
    with pytest.raises(TopicRecordsIntegrityError, match="topic"):
        TopicRecords.open(
            wrong_topic / "records.sqlite3",
            _manifest_path(wrong_topic),
            "different-topic",
            DocumentStore(tmp_path / "objects-topic"),
        )

    missing_object = tmp_path / "missing-object"
    receipt, _ = _build_unicode_topic(
        missing_object, tmp_path / "objects-missing", candidates
    )
    digest = receipt.document_sha256s[0]
    (tmp_path / "objects-missing" / "sha256" / digest[:2] / f"{digest}.utf8").unlink()
    with pytest.raises(TopicRecordsIntegrityError, match="source|document"):
        TopicRecords.open(
            missing_object / "records.sqlite3",
            _manifest_path(missing_object),
            "topic-unicode",
            DocumentStore(tmp_path / "objects-missing"),
        )


def test_open_rejects_bad_foreign_key_stale_wal_and_partial_publication(
    tmp_path: Path,
) -> None:
    _, candidates = _unicode_candidates()
    foreign_key = tmp_path / "foreign-key"
    _build_unicode_topic(foreign_key, tmp_path / "objects-fk", candidates)
    database = foreign_key / "records.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=OFF")
    valid_span = connection.execute(
        "SELECT candidate_pk, role, ordinal, start_char, end_char, start_byte, "
        "end_byte, text_sha256, cross_encoder_score "
        "FROM candidate_span ORDER BY candidate_pk, role, ordinal LIMIT 1"
    ).fetchone()
    assert valid_span is not None
    invalid_candidate_pk = connection.execute(
        "SELECT COALESCE(MAX(candidate_pk), 0) + 1 FROM candidate"
    ).fetchone()[0]
    connection.execute(
        "INSERT INTO candidate_span "
        "(candidate_pk, role, ordinal, start_char, end_char, start_byte, end_byte, "
        "text_sha256, cross_encoder_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (invalid_candidate_pk, *valid_span[1:]),
    )
    connection.commit()
    connection.close()
    mutated = database.read_bytes()
    _rewrite_manifest(
        _manifest_path(foreign_key),
        database_sha256=sha256(mutated).hexdigest(),
        database_bytes=len(mutated),
    )
    with pytest.raises(TopicRecordsIntegrityError, match="foreign-key"):
        TopicRecords.open(
            database,
            _manifest_path(foreign_key),
            "topic-unicode",
            DocumentStore(tmp_path / "objects-fk"),
        )

    stale_wal = tmp_path / "stale-wal"
    _build_unicode_topic(stale_wal, tmp_path / "objects-wal", candidates)
    (stale_wal / "records.sqlite3-wal").write_bytes(b"stale")
    (stale_wal / "records.sqlite3-shm").write_bytes(b"stale")
    with pytest.raises(TopicRecordsIntegrityError, match="WAL/SHM"):
        TopicRecords.open(
            stale_wal / "records.sqlite3",
            _manifest_path(stale_wal),
            "topic-unicode",
            DocumentStore(tmp_path / "objects-wal"),
        )

    partial = tmp_path / "partial"
    partial.mkdir()
    shutil.copy2(stale_wal / "records.sqlite3", partial / "records.sqlite3")
    with pytest.raises(TopicRecordsIntegrityError, match="both exist"):
        TopicRecords.open(
            partial / "records.sqlite3",
            partial / "canonical" / "records-manifest.json",
            "topic-unicode",
            DocumentStore(tmp_path / "objects-wal"),
        )
    assert not (partial / "canonical" / "complete.json").exists()


def test_publication_is_idempotent_first_writer_wins_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    _, candidates = _unicode_candidates()
    destination = tmp_path / "published"
    first_receipt, _ = _build_unicode_topic(
        destination, tmp_path / "objects-first-writer", candidates
    )
    winning_bytes = (destination / "records.sqlite3").read_bytes()

    second_store = DocumentStore(tmp_path / "objects-second-writer")
    second_builder = TopicRecordsBuilder(destination, "topic-unicode", second_store, run_id="r1")
    source, _ = _unicode_candidates()
    second_builder.bind_document("shared-doc", source)
    for candidate in reversed(candidates):
        second_builder.add_candidate(candidate)
    second_receipt = second_builder.publish({"fixture": "unicode"})
    assert second_receipt == first_receipt
    assert (destination / "records.sqlite3").read_bytes() == winning_bytes

    changed = replace(
        candidates[0],
        evidence_sentences=tuple(
            replace(sentence, cross_encoder_score=sentence.cross_encoder_score + 2.0)
            for sentence in candidates[0].evidence_sentences
        ),
        sentence_cross_encoder_score=candidates[0].sentence_cross_encoder_score + 2.0,
    )
    conflict_store = DocumentStore(tmp_path / "objects-conflict")
    conflict_builder = TopicRecordsBuilder(destination, "topic-unicode", conflict_store, run_id="r1")
    conflict_builder.bind_document("shared-doc", source)
    conflict_builder.add_candidate(changed)
    with pytest.raises(TopicRecordsIntegrityError, match="contradictory|partial"):
        conflict_builder.publish({"fixture": "unicode"})
    assert (destination / "records.sqlite3").read_bytes() == winning_bytes
    assert (destination / "canonical" / "records-manifest.json").is_file()


def test_same_builder_republish_rejects_changed_identity(tmp_path: Path) -> None:
    source, candidates = _unicode_candidates()
    destination = tmp_path / "published"
    store = DocumentStore(tmp_path / "objects")
    builder = TopicRecordsBuilder(destination, "topic-unicode", store, run_id="r1")
    builder.bind_document("shared-doc", source)
    for candidate in candidates:
        builder.add_candidate(candidate)
    first_receipt = builder.publish({"run": "first"})
    winning_bytes = (destination / "records.sqlite3").read_bytes()

    with pytest.raises(TopicRecordsIntegrityError, match="identity"):
        builder.publish({"run": "second"})

    assert (destination / "records.sqlite3").read_bytes() == winning_bytes
    assert first_receipt.semantic_sha256 == json.loads(
        _manifest_path(destination).read_text(encoding="utf-8")
    )["semantic_sha256"]


def test_new_builder_republish_rejects_changed_identity(tmp_path: Path) -> None:
    source, candidates = _unicode_candidates()
    destination = tmp_path / "published"
    first_receipt, records = _build_unicode_topic(
        destination, tmp_path / "objects-first", candidates
    )
    records.close()
    winning_bytes = (destination / "records.sqlite3").read_bytes()

    contender_store = DocumentStore(tmp_path / "objects-contender")
    contender = TopicRecordsBuilder(destination, "topic-unicode", contender_store, run_id="r1")
    contender.bind_document("shared-doc", source)
    for candidate in candidates:
        contender.add_candidate(candidate)

    with pytest.raises(TopicRecordsIntegrityError, match="identity"):
        contender.publish({"fixture": "different"})

    assert (destination / "records.sqlite3").read_bytes() == winning_bytes
    assert first_receipt.semantic_sha256 == json.loads(
        _manifest_path(destination).read_text(encoding="utf-8")
    )["semantic_sha256"]


def test_simultaneous_identical_publishers_converge_after_database_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    start = Barrier(3)
    database_linked = Barrier(2)
    release_manifest = Barrier(2)

    def publication_hook(boundary: str) -> None:
        if boundary == "after_database":
            database_linked.wait(timeout=5)
            release_manifest.wait(timeout=5)

    monkeypatch.setattr(
        topic_records_module,
        "_PUBLICATION_TEST_HOOK",
        publication_hook,
        raising=False,
    )

    def publish(index: int):
        builder = _prepared_unicode_builder(
            destination,
            tmp_path / f"objects-{index}",
        )
        start.wait(timeout=5)
        return builder.publish({"fixture": "unicode"})

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(publish, index) for index in range(2))
        start.wait(timeout=5)
        database_linked.wait(timeout=5)
        assert (destination / "records.sqlite3").is_file()
        assert not _manifest_path(destination).exists()
        release_manifest.wait(timeout=5)
        receipts = tuple(future.result(timeout=5) for future in futures)

    assert receipts[0] == receipts[1]
    assert _manifest_path(destination).is_file()


@pytest.mark.parametrize(
    ("boundary", "database_exists", "manifest_exists"),
    (
        ("before_database", False, False),
        ("after_database", True, False),
        ("before_manifest", True, False),
        ("after_manifest", True, True),
    ),
)
def test_publication_boundary_failure_never_names_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    database_exists: bool,
    manifest_exists: bool,
) -> None:
    destination = tmp_path / boundary
    builder = _prepared_unicode_builder(destination, tmp_path / f"objects-{boundary}")

    def publication_hook(actual_boundary: str) -> None:
        if actual_boundary == boundary:
            raise OSError(f"injected {boundary} failure")

    monkeypatch.setattr(
        topic_records_module,
        "_PUBLICATION_TEST_HOOK",
        publication_hook,
        raising=False,
    )

    with pytest.raises(TopicRecordsIntegrityError, match="publish"):
        builder.publish({"fixture": "unicode"})

    database = destination / "records.sqlite3"
    manifest = _manifest_path(destination)
    assert database.exists() is database_exists
    assert manifest.exists() is manifest_exists
    if database_exists and not manifest_exists:
        winner_bytes = database.read_bytes()
        monkeypatch.setattr(topic_records_module, "_PUBLICATION_TEST_HOOK", None)
        contender = _prepared_unicode_builder(
            destination,
            tmp_path / f"objects-{boundary}-contender",
        )
        with pytest.raises(TopicRecordsIntegrityError, match="partial"):
            contender.publish({"fixture": "unicode"})
        assert database.read_bytes() == winner_bytes
        assert not manifest.exists()
    elif manifest_exists:
        monkeypatch.setattr(topic_records_module, "_PUBLICATION_TEST_HOOK", None)
        records = TopicRecords.open(
            database,
            manifest,
            "topic-unicode",
            DocumentStore(tmp_path / f"objects-{boundary}"),
        )
        records.close()


def test_capability_internals_and_rebind_are_not_module_or_class_exposed() -> None:
    for name in (
        "_mint_validation_session",
        "_resolve_validation_session",
        "_TOPIC_RECORDS_CONSTRUCTOR_AUTHORITY",
        "_make_validation_capability",
        "_install_topic_records_capability",
    ):
        assert not hasattr(topic_records_module, name)
    assert not hasattr(TopicRecords, "_rebind")


def test_topic_records_constructor_rejects_guessed_or_recovered_authority(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    database = sqlite3.connect(
        f"file:{(destination / 'records.sqlite3').resolve()}?mode=ro", uri=True
    )
    try:
        manifest_path = _manifest_path(destination)
        manifest_value = json.loads(manifest_path.read_bytes())
        with pytest.raises(TypeError, match="construction|internal|constructor"):
            TopicRecords(
                database,
                destination / "records.sqlite3",
                manifest_path,
                manifest_value,
                DocumentStore(store_root),
                published.validation_session,
                _construction_authority=getattr(
                    topic_records_module, "_TOPIC_RECORDS_CONSTRUCTOR_AUTHORITY", object()
                ),
            )
    finally:
        database.close()


def test_assert_current_rejects_object_new_clone_with_copied_attrs(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    records = _build_unicode_topic(
        destination, store_root, _unicode_candidates()[1]
    )[1]
    try:
        clone = object.__new__(TopicRecords)
        clone.__dict__.update(vars(records))

        with pytest.raises(TopicRecordsIntegrityError, match="registered|handle"):
            TopicRecords.assert_current(clone)
    finally:
        records.close()


def test_assert_current_returns_exact_receipt_and_revokes_closed_handle(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    records = _build_unicode_topic(
        destination, store_root, _unicode_candidates()[1]
    )[1]
    receipt = TopicRecords.assert_current(records)
    assert receipt is records.receipt
    assert receipt is TopicRecords.assert_current(records)

    records.close()
    with pytest.raises(TopicRecordsIntegrityError, match="registered|closed|revoked"):
        TopicRecords.assert_current(records)


def test_receipt_pins_exact_manifest_schema_and_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    manifest_body = _manifest_path(destination).read_bytes()
    manifest = json.loads(manifest_body)
    records = TopicRecords.open(
        destination / "records.sqlite3",
        _manifest_path(destination),
        "topic-unicode",
        DocumentStore(store_root),
        validation_session=published.validation_session,
    )
    try:
        receipt = TopicRecords.assert_current(records)
        assert receipt.schema_version == manifest["schema_version"]
        assert receipt.manifest_sha256 == sha256(manifest_body).hexdigest()
        assert receipt.manifest_bytes == len(manifest_body)
    finally:
        records.close()


def test_manifest_validation_and_receipt_share_one_same_inode_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    manifest_path = _manifest_path(destination)
    original_body = manifest_path.read_bytes()
    original_manifest = json.loads(original_body)
    mutated_manifest = dict(original_manifest)
    mutated_manifest["identity_json"] = json.dumps(
        {"fixture": "mutated"}, sort_keys=True, separators=(",", ":")
    )
    mutated_body = (
        json.dumps(mutated_manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert mutated_body != original_body
    manifest_inode = manifest_path.stat().st_ino
    original_read_descriptor = topic_records_module._read_descriptor
    mutated = False

    def read_then_mutate(descriptor: int, label: str) -> bytes:
        nonlocal mutated
        body = original_read_descriptor(descriptor, label)
        if label == "records manifest" and not mutated:
            manifest_path.write_bytes(mutated_body)
            mutated = True
        return body

    monkeypatch.setattr(
        topic_records_module, "_read_descriptor", read_then_mutate
    )
    records = TopicRecords.open(
        destination / "records.sqlite3",
        manifest_path,
        "topic-unicode",
        DocumentStore(store_root),
    )
    session = records.validation_session
    receipt = records.receipt
    try:
        assert records._manifest == original_manifest
    finally:
        records.close()

    assert mutated
    assert manifest_path.stat().st_ino == manifest_inode
    rebound = None
    try:
        with pytest.raises(
            TopicRecordsIntegrityError,
            match="validation session manifest bytes",
        ):
            rebound = TopicRecords.open(
                destination / "records.sqlite3",
                manifest_path,
                "topic-unicode",
                DocumentStore(store_root),
                validation_session=session,
            )
    finally:
        if rebound is not None:
            rebound.close()
    assert receipt.manifest_sha256 == sha256(original_body).hexdigest()
    assert receipt.manifest_bytes == len(original_body)


def test_open_hashes_and_connects_to_the_same_database_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, candidates = _unicode_candidates()
    original_destination = tmp_path / "original"
    replacement_destination = tmp_path / "replacement"
    original_store = tmp_path / "objects-original"
    replacement_store = tmp_path / "objects-replacement"
    original_receipt, original_records = _build_unicode_topic(
        original_destination, original_store, candidates
    )
    original_records.close()
    changed_evidence = tuple(
        replace(sentence, cross_encoder_score=sentence.cross_encoder_score + 1.0)
        for sentence in candidates[0].evidence_sentences
    )
    changed_candidate = replace(
        candidates[0],
        evidence_sentences=changed_evidence,
        sentence_cross_encoder_score=candidates[0].sentence_cross_encoder_score + 1.0,
    )
    _, replacement_records = _build_unicode_topic(
        replacement_destination,
        replacement_store,
        (changed_candidate, *candidates[1:]),
    )
    replacement_records.close()

    original_database = original_destination / "records.sqlite3"
    replacement_database = replacement_destination / "records.sqlite3"
    assert original_database.read_bytes() != replacement_database.read_bytes()
    real_descriptor_receipt = topic_records_module._descriptor_receipt
    replaced = False

    def hash_then_replace(descriptor: int, label: str) -> tuple[str, int]:
        nonlocal replaced
        receipt = real_descriptor_receipt(descriptor, label)
        if label == "topic database" and not replaced:
            os.replace(replacement_database, original_database)
            replaced = True
        return receipt

    monkeypatch.setattr(
        topic_records_module, "_descriptor_receipt", hash_then_replace
    )
    records = TopicRecords.open(
        original_database,
        _manifest_path(original_destination),
        "topic-unicode",
        DocumentStore(original_store),
    )
    try:
        assert replaced
        assert records.receipt.database_sha256 == original_receipt.database_sha256
        loaded = records.load_candidates(
            {(candidates[0].subnarrative_id, candidates[0].candidate_nugget_id)}
        )
        assert loaded[(
            candidates[0].subnarrative_id,
            candidates[0].candidate_nugget_id,
        )] == candidates[0]
    finally:
        records.close()


def test_open_handle_stays_bound_after_database_path_replacement(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    _, candidates = _unicode_candidates()
    _receipt, records = _build_unicode_topic(destination, store_root, candidates)
    database_path = destination / "records.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(b"not a sqlite database")

    os.replace(replacement, database_path)
    try:
        key = (candidates[0].subnarrative_id, candidates[0].candidate_nugget_id)
        assert records.load_candidates({key})[key] == candidates[0]
        assert TopicRecords.assert_current(records) is records.receipt
    finally:
        records.close()


def test_open_handle_keeps_pinned_manifest_receipt_after_path_replacement(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    _published, records = _build_unicode_topic(
        destination, store_root, _unicode_candidates()[1]
    )
    pinned = records.receipt
    replacement = tmp_path / "replacement-manifest.json"
    replacement.write_text("{}\n", encoding="utf-8")

    os.replace(replacement, _manifest_path(destination))
    try:
        assert TopicRecords.assert_current(records) is pinned
        assert records.receipt.manifest_sha256 == pinned.manifest_sha256
        assert records.receipt.manifest_bytes == pinned.manifest_bytes
    finally:
        records.close()


def test_fork_inherited_handle_is_not_current(tmp_path: Path) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("fork is unavailable on this platform")
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    _published, records = _build_unicode_topic(
        destination, store_root, _unicode_candidates()[1]
    )
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        try:
            TopicRecords.assert_current(records)
        except TopicRecordsIntegrityError:
            outcome = b"rejected"
        except BaseException:
            outcome = b"unexpected"
        else:
            outcome = b"accepted"
        os.write(write_fd, outcome)
        os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    try:
        _, status = os.waitpid(child_pid, 0)
        outcome = os.read(read_fd, 32)
        assert os.WIFEXITED(status)
        assert outcome == b"rejected"
        assert TopicRecords.assert_current(records) is records.receipt
    finally:
        os.close(read_fd)
        records.close()


def test_validation_registry_does_not_keep_handle_alive(tmp_path: Path) -> None:
    _published, records = _build_unicode_topic(
        tmp_path / "topic", tmp_path / "objects", _unicode_candidates()[1]
    )
    handle_ref = weakref.ref(records)

    del records
    gc.collect()

    assert handle_ref() is None


def test_failed_regular_file_open_does_not_leak_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not Path("/proc/self/fd").is_dir():
        pytest.skip("descriptor accounting requires procfs")
    path = tmp_path / "records.sqlite3"
    path.write_bytes(b"sqlite")
    before = len(tuple(Path("/proc/self/fd").iterdir()))

    def fail_fstat(_descriptor: int) -> object:
        raise OSError("injected fstat failure")

    monkeypatch.setattr(topic_records_module.os, "fstat", fail_fstat)
    with pytest.raises(TopicRecordsIntegrityError, match="open|regular"):
        topic_records_module._open_regular_readonly(path, "topic database")

    after = len(tuple(Path("/proc/self/fd").iterdir()))
    assert after == before


def test_snapshot_copy_forces_0600_under_restrictive_umask(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "snapshot.sqlite3"
    source.write_bytes(b"snapshot bytes")
    source_descriptor = os.open(source, os.O_RDONLY)
    previous_umask = os.umask(0o777)
    try:
        topic_records_module._copy_descriptor(source_descriptor, destination)
    finally:
        os.umask(previous_umask)
        os.close(source_descriptor)

    assert destination.stat().st_mode & 0o777 == 0o600
    assert destination.read_bytes() == b"snapshot bytes"


def _deleted_fd_targets() -> set[str]:
    targets: set[str] = set()
    for descriptor in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(descriptor)
        except OSError:
            continue
        if " (deleted)" in target:
            targets.add(target)
    return targets


def test_fast_rebind_uses_original_snapshot_after_same_inode_valid_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, candidates = _unicode_candidates()
    destination = tmp_path / "original"
    replacement_destination = tmp_path / "replacement"
    store_root = tmp_path / "objects"
    replacement_store_root = tmp_path / "replacement-objects"
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    changed_evidence = tuple(
        replace(sentence, cross_encoder_score=sentence.cross_encoder_score + 1.0)
        for sentence in candidates[0].evidence_sentences
    )
    changed_candidate = replace(
        candidates[0],
        evidence_sentences=changed_evidence,
        sentence_cross_encoder_score=candidates[0].sentence_cross_encoder_score + 1.0,
    )
    _build_unicode_topic(
        replacement_destination,
        replacement_store_root,
        (changed_candidate, *candidates[1:]),
    )[1].close()

    database = destination / "records.sqlite3"
    manifest = _manifest_path(destination)
    database_inode = database.stat().st_ino
    manifest_inode = manifest.stat().st_ino
    original_descriptor_receipt = topic_records_module._descriptor_receipt
    updated = False

    def hash_then_update(descriptor: int, label: str) -> tuple[str, int]:
        nonlocal updated
        receipt = original_descriptor_receipt(descriptor, label)
        if label == "topic database" and not updated:
            database.write_bytes((replacement_destination / "records.sqlite3").read_bytes())
            updated = True
        return receipt

    monkeypatch.setattr(topic_records_module, "_descriptor_receipt", hash_then_update)
    records = TopicRecords.open(
        database,
        manifest,
        "topic-unicode",
        DocumentStore(store_root),
        validation_session=published.validation_session,
    )
    try:
        key = (candidates[0].subnarrative_id, candidates[0].candidate_nugget_id)
        assert updated
        assert database.stat().st_ino == database_inode
        assert manifest.stat().st_ino == manifest_inode
        assert records.receipt == published.receipt
        assert records.load_candidates({key})[key] == candidates[0]
    finally:
        records.close()


def test_object_new_clone_cannot_use_data_operations_or_authoritative_properties(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    records = _build_unicode_topic(
        destination, store_root, _unicode_candidates()[1]
    )[1]
    candidate = _unicode_candidates()[1][0]
    key = (candidate.subnarrative_id, candidate.candidate_nugget_id)
    clone = object.__new__(TopicRecords)
    clone.__dict__.update(vars(records))
    context = SubnarrativeContext(
        "topic-unicode",
        "Explain safety.",
        candidate.subnarrative_id,
        "Safety evidence",
    )

    try:
        for operation in (
            lambda: clone.validate_all_sources(),
            lambda: clone.load_candidates({key}),
            lambda: clone.selection_pool(context, 1),
            lambda: TopicRecords.assert_current(clone),
            lambda: clone.__enter__(),
            lambda: clone.receipt,
            lambda: clone.topic_id,
            lambda: clone.validation_session,
            lambda: clone._manifest,
        ):
            with pytest.raises(TopicRecordsIntegrityError, match="registered|current|handle|capability"):
                operation()
        assert not hasattr(TopicRecords, "_reconstruct")
    finally:
        records.close()


def test_fork_inherited_handle_rejects_all_data_operations_but_parent_remains_valid(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("fork is unavailable on this platform")
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    _published, records = _build_unicode_topic(
        destination, store_root, _unicode_candidates()[1]
    )
    candidate = _unicode_candidates()[1][0]
    context = SubnarrativeContext(
        "topic-unicode",
        "Explain safety.",
        candidate.subnarrative_id,
        "Safety evidence",
    )
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        outcomes: list[bytes] = []
        try:
            operations = (
                lambda: records.validate_all_sources(),
                lambda: records.load_candidates({(candidate.subnarrative_id, candidate.candidate_nugget_id)}),
                lambda: records.selection_pool(context, 1),
                lambda: TopicRecords.assert_current(records),
            )
            for operation in operations:
                try:
                    operation()
                except TopicRecordsIntegrityError:
                    outcomes.append(b"rejected")
                except BaseException:
                    outcomes.append(b"unexpected")
                else:
                    outcomes.append(b"accepted")
            os.write(write_fd, b"|".join(outcomes))
        finally:
            os.close(write_fd)
            os._exit(0)

    os.close(write_fd)
    try:
        _, status = os.waitpid(child_pid, 0)
        assert os.WIFEXITED(status)
        assert os.read(read_fd, 128) == b"rejected|rejected|rejected|rejected"
        assert TopicRecords.assert_current(records) is records.receipt
    finally:
        os.close(read_fd)
        records.close()


def test_last_handle_close_from_non_creator_thread_releases_master_after_token_gc(
    tmp_path: Path,
) -> None:
    if not Path("/proc/self/fd").is_dir():
        pytest.skip("descriptor accounting requires procfs")
    deleted_before = _deleted_fd_targets()
    published = _prepared_unicode_builder(
        tmp_path / "topic", tmp_path / "objects"
    ).publish({"fixture": "unicode"})
    session = published.validation_session
    records = TopicRecords.open(
        tmp_path / "topic" / "records.sqlite3",
        _manifest_path(tmp_path / "topic"),
        "topic-unicode",
        DocumentStore(tmp_path / "objects"),
        validation_session=session,
    )
    deleted_after_open = _deleted_fd_targets()
    new_deleted_targets = deleted_after_open - deleted_before
    assert new_deleted_targets

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(records.close).result(timeout=5)
    with pytest.raises(TopicRecordsIntegrityError, match="registered|revoked|closed"):
        TopicRecords.assert_current(records)
    deleted_while_token_alive = _deleted_fd_targets()
    assert new_deleted_targets <= deleted_while_token_alive

    del records
    del published
    del session
    gc.collect()
    assert not new_deleted_targets & _deleted_fd_targets()


def test_close_waits_for_in_flight_data_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    records = _build_unicode_topic(
        destination, store_root, _unicode_candidates()[1]
    )[1]
    candidate = _unicode_candidates()[1][0]
    entered = Event()
    release = Event()
    real_reconstruct = topic_records_module._reconstruct_candidate

    def blocked_reconstruct(*args: object, **kwargs: object):
        entered.set()
        assert release.wait(timeout=5)
        return real_reconstruct(*args, **kwargs)

    monkeypatch.setattr(topic_records_module, "_reconstruct_candidate", blocked_reconstruct)
    with ThreadPoolExecutor(max_workers=2) as executor:
        operation = executor.submit(
            records.load_candidates,
            {(candidate.subnarrative_id, candidate.candidate_nugget_id)},
        )
        assert entered.wait(timeout=5)
        closing = executor.submit(records.close)
        assert not closing.done()
        release.set()
        assert operation.result(timeout=5)
        assert closing.result(timeout=5) is None
    with pytest.raises(TopicRecordsIntegrityError, match="registered|revoked|closed"):
        TopicRecords.assert_current(records)


def test_fast_rebind_reuses_one_immutable_master_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "topic"
    store_root = tmp_path / "objects"
    real_connect = topic_records_module._connect_readonly_descriptor
    masters: list[sqlite3.Connection] = []

    def counted_connect(descriptor: int) -> sqlite3.Connection:
        connection = real_connect(descriptor)
        masters.append(connection)
        return connection

    monkeypatch.setattr(topic_records_module, "_connect_readonly_descriptor", counted_connect)
    published = _prepared_unicode_builder(destination, store_root).publish(
        {"fixture": "unicode"}
    )
    handles = [
        TopicRecords.open(
            destination / "records.sqlite3",
            _manifest_path(destination),
            "topic-unicode",
            DocumentStore(store_root),
            validation_session=published.validation_session,
        )
        for _ in range(3)
    ]
    try:
        assert len(masters) == 1
        assert all(records.receipt == published.receipt for records in handles)
    finally:
        for records in handles:
            records.close()
