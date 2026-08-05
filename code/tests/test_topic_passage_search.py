from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Callable, Sequence

import pytest

from trec_rag.chunking import TextChunk
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate
from trec_rag.topic_passage_search import (
    FocusedQuery,
    OrganizerRequestFailed,
    PassageScoringFailed,
    PassageSearchPolicy,
    PassageSearchResult,
    TopicPassageSearch,
)


@dataclass(frozen=True)
class MemoryReceipt:
    content_sha256: str


class MemoryDocumentStore:
    def __init__(self) -> None:
        self.admitted: list[str] = []
        self.expected_digests: list[str | None] = []
        self.objects: dict[str, str] = {}

    def admit_text(self, text: str, *, expected_sha256: str | None = None) -> MemoryReceipt:
        digest = sha256(text.encode("utf-8")).hexdigest()
        if expected_sha256 is not None and expected_sha256 != digest:
            raise AssertionError("test document store received a wrong expected digest")
        self.admitted.append(text)
        self.expected_digests.append(expected_sha256)
        self.objects[digest] = text
        return MemoryReceipt(digest)


class FixedChunker:
    identity = {"backend": "test-fixed", "implementation": "v1"}

    def split_text(self, text: str, *, document_id: str) -> list[TextChunk]:
        if not text:
            return []
        return [TextChunk(document_id, f"{document_id}:0000", text, 0, len(text))]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: TextChunk
    relevance_score: float


class RecordingScorer:
    identity = {"backend": "test-scorer", "model": "v1"}

    def __init__(self, score: Callable[[TextChunk], float] | None = None) -> None:
        self.score = score or (lambda _chunk: 1.0)
        self.calls: list[tuple[str, tuple[TextChunk, ...]]] = []

    @property
    def chunk_count(self) -> int:
        return sum(len(chunks) for _query, chunks in self.calls)

    def cache_key(self, query_text: str, passage_text: str) -> str:
        return sha256(f"{query_text}\0{passage_text}".encode("utf-8")).hexdigest()

    def rank(self, query_text: str, chunks: Sequence[TextChunk]) -> tuple[ScoredChunk, ...]:
        rows = tuple(chunks)
        self.calls.append((query_text, rows))
        return tuple(ScoredChunk(chunk, self.score(chunk)) for chunk in rows)


class RecordingUnavailableRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.query_variants: list[QueryVariant] = []
        self.depths: list[int] = []

    def retrieve(
        self, query: QueryVariant, *, depth: int
    ) -> Sequence[RetrievedCandidate]:
        self.queries.append(query.query_text)
        self.query_variants.append(query)
        self.depths.append(depth)
        raise OrganizerRequestFailed("organizer unavailable")


class RecordingRetriever:
    def __init__(self, candidates: Sequence[RetrievedCandidate]) -> None:
        self.candidates = tuple(candidates)
        self.queries: list[QueryVariant] = []
        self.depths: list[int] = []

    def retrieve(
        self, query: QueryVariant, *, depth: int
    ) -> Sequence[RetrievedCandidate]:
        self.queries.append(query)
        self.depths.append(depth)
        return self.candidates


class FailingScorer(RecordingScorer):
    def rank(self, query_text: str, chunks: Sequence[TextChunk]) -> tuple[ScoredChunk, ...]:
        self.calls.append((query_text, tuple(chunks)))
        raise PassageScoringFailed("model runtime failed")


def focused_query() -> FocusedQuery:
    return FocusedQuery("q-1", "focused text", "facet-a", ("facet-b",))


def candidate(
    docid: str,
    text: str,
    *,
    rank: int = 1,
    score: float = 1.0,
    query_id: str = "q-1",
    query_text: str = "focused text",
    retriever_name: str = "test-retriever",
) -> RetrievedCandidate:
    return RetrievedCandidate(
        "topic-1", query_id, retriever_name, query_text, docid, rank, score, text
    )


def fake_candidates(count: int) -> list[RetrievedCandidate]:
    return [candidate(f"d{index}", f"body-{index}", rank=index + 1) for index in range(count)]


def make_search(
    tmp_path: Path,
    candidates: Sequence[RetrievedCandidate] = (),
    scorer: RecordingScorer | None = None,
    retriever: object | None = None,
    chunker: object | None = None,
    policy: PassageSearchPolicy | None = None,
) -> tuple[TopicPassageSearch, MemoryDocumentStore, RecordingScorer | None]:
    store = MemoryDocumentStore()
    selected_scorer = scorer or RecordingScorer()
    selected_retriever = retriever or RecordingRetriever(candidates)
    search = TopicPassageSearch(
        "topic-1",
        store,
        selected_retriever,
        chunker or FixedChunker(),
        selected_scorer,
        policy=policy,
    )
    return search, store, selected_scorer


def test_focused_query_has_one_primary_and_deduplicated_supporting_facets() -> None:
    query = FocusedQuery("q-1", "focused text", "facet-a", ("facet-b", "facet-b", "facet-a"))
    assert query.primary_subnarrative_id == "facet-a"
    assert query.supporting_subnarrative_ids == ("facet-b",)


def test_search_stops_after_three_identical_failed_attempts(tmp_path: Path) -> None:
    retriever = RecordingUnavailableRetriever()
    search, _store, _scorer = make_search(tmp_path, retriever=retriever)

    result = search.search(focused_query())

    assert result.status == "incomplete"
    assert result.stopping_reason == "retrieval_unavailable"
    assert result.attempt_count == 3
    assert retriever.queries == [focused_query().text] * 3
    assert retriever.query_variants[0] == retriever.query_variants[1] == retriever.query_variants[2]
    assert retriever.depths == [1000, 1000, 1000]


def test_policy_defaults_and_rejects_non_boolean_positive_integer_values() -> None:
    assert PassageSearchPolicy() == PassageSearchPolicy(1000, 100, 3)
    for field in ("retrieval_depth", "passage_limit", "max_attempts"):
        with pytest.raises(ValueError, match=field):
            PassageSearchPolicy(**{field: True})
        with pytest.raises(ValueError, match=field):
            PassageSearchPolicy(**{field: 0})


def test_search_passes_custom_retrieval_depth(tmp_path: Path) -> None:
    retriever = RecordingRetriever(fake_candidates(2))
    search, _store, _scorer = make_search(
        tmp_path,
        retriever=retriever,
        policy=PassageSearchPolicy(retrieval_depth=2),
    )

    result = search.search(focused_query())

    assert result.requested_documents == 2
    assert result.returned_documents == 2
    assert retriever.depths == [2]


def test_search_scores_passages_from_every_returned_document(tmp_path: Path) -> None:
    candidates = fake_candidates(1000)
    scorer = RecordingScorer(
        score=lambda chunk: float(chunk.document_id.removeprefix("d"))
    )
    search, _store, _scorer = make_search(tmp_path, candidates, scorer)

    result = search.search(focused_query())

    assert result.status == "complete"
    assert result.returned_documents == 1000
    assert result.scored_documents == 1000
    assert result.scored_passages == 1000
    assert len(result.passages) == 100
    assert result.passages[0].docid == "d999"
    assert result.passages[0].rank == 1
    assert result.passages[-1].rank == 100
    assert scorer.chunk_count == 1000


def test_search_returns_source_exhaustion_without_hiding_documents(tmp_path: Path) -> None:
    search, _store, _scorer = make_search(tmp_path, [candidate("d1", "A body")])

    result = search.search(focused_query())

    assert result.status == "complete"
    assert result.source_exhausted is True
    assert result.returned_documents == 1
    assert len(result.documents) == 1


def test_passage_byte_offsets_are_from_exact_utf8_source(tmp_path: Path) -> None:
    text = "A café Ω end"
    search, _store, _scorer = make_search(tmp_path, [candidate("d1", text)])

    result = search.search(focused_query())
    row = result.passages[0]

    assert row.start_char == 0
    assert row.end_char == len(text)
    assert row.start_byte == 0
    assert row.end_byte == len(text.encode("utf-8"))
    assert row.text.encode("utf-8") == text.encode("utf-8")[row.start_byte : row.end_byte]


def test_passage_result_carries_explicit_chunker_identity(tmp_path: Path) -> None:
    search, _store, _scorer = make_search(tmp_path, [candidate("d1", "exact body")])

    result = search.search(focused_query())

    assert result.passages[0].chunker_identity == FixedChunker.identity
    assert result.passages[0].scoring_text_sha256 == sha256(b"exact body").hexdigest()


def test_tied_scores_use_source_rank_after_score(tmp_path: Path) -> None:
    candidates = [candidate("d2", "second", rank=2), candidate("d1", "first", rank=1)]
    search, _store, _scorer = make_search(tmp_path, candidates)

    result = search.search(focused_query())

    assert [row.docid for row in result.passages] == ["d1", "d2"]
    assert [row.rank for row in result.passages] == [1, 2]


def test_passage_identity_is_query_independent_for_the_same_source_span(tmp_path: Path) -> None:
    first_search, _first_store, _first_scorer = make_search(
        tmp_path / "first",
        [candidate("d1", "same body", query_id="q-1", query_text="first query")],
    )
    second_search, _second_store, _second_scorer = make_search(
        tmp_path / "second",
        [candidate("d1", "same body", query_id="q-2", query_text="second query")],
    )

    first = first_search.search(FocusedQuery("q-1", "first query", "facet-a"))
    second = second_search.search(FocusedQuery("q-2", "second query", "facet-b"))

    assert first.passages[0].passage_id == second.passages[0].passage_id


def test_task3_hardening_passage_identity_distinguishes_document_ids(tmp_path: Path) -> None:
    search, _store, _scorer = make_search(
        tmp_path,
        [
            candidate("doc-a", "identical body", rank=1),
            candidate("doc-b", "identical body", rank=2),
        ],
        policy=PassageSearchPolicy(retrieval_depth=2),
    )

    result = search.search(focused_query())

    assert [row.docid for row in result.passages] == ["doc-a", "doc-b"]
    assert len({row.passage_id for row in result.passages}) == 2


def test_scoring_failure_returns_validated_documents_without_inventing_passages(
    tmp_path: Path,
) -> None:
    search, store, _scorer = make_search(
        tmp_path, [candidate("d1", "exact body")], scorer=FailingScorer()
    )

    result = search.search(focused_query())

    assert result.status == "incomplete"
    assert result.stopping_reason == "scoring_failed"
    assert result.returned_documents == 1
    assert result.scored_documents == 0
    assert result.scored_passages == 0
    assert result.passages == ()
    assert result.documents[0].docid == "d1"
    assert result.documents[0].best_passage_id is None
    assert store.admitted == ["exact body"]
    assert store.expected_digests == [sha256(b"exact body").hexdigest()]


def test_no_evidence_is_incomplete_after_successful_empty_retrieval(tmp_path: Path) -> None:
    search, _store, _scorer = make_search(tmp_path, [])

    result = search.search(focused_query())

    assert result.status == "incomplete"
    assert result.stopping_reason == "no_evidence"
    assert result.returned_documents == 0
    assert result.passages == ()


def test_conflicting_duplicate_docids_remain_loud(tmp_path: Path) -> None:
    candidates = [candidate("d1", "first", rank=1), candidate("d1", "different", rank=2)]
    search, _store, _scorer = make_search(tmp_path, candidates)

    with pytest.raises(ValueError, match="conflicting.*docid"):
        search.search(focused_query())


def test_duplicate_same_body_selection_is_permutation_invariant(tmp_path: Path) -> None:
    candidates = [
        candidate("d1", "same body", rank=2, score=99.0),
        candidate("d1", "same body", rank=1, score=1.0),
        candidate("d1", "same body", rank=1, score=3.0),
    ]
    first_search, _first_store, _first_scorer = make_search(
        tmp_path / "first", candidates
    )
    second_search, _second_store, _second_scorer = make_search(
        tmp_path / "second", tuple(reversed(candidates))
    )

    first = first_search.search(focused_query())
    second = second_search.search(focused_query())

    assert first == second
    assert [(row.source_rank, row.source_score) for row in first.documents] == [(1, 3.0)]


def test_duplicate_tie_rejects_conflicting_remaining_metadata(tmp_path: Path) -> None:
    candidates = [
        candidate("d1", "same body", rank=1, score=3.0, retriever_name="first"),
        candidate("d1", "same body", rank=1, score=3.0, retriever_name="second"),
    ]
    search, _store, _scorer = make_search(tmp_path, candidates)

    with pytest.raises(ValueError, match="duplicate.*metadata"):
        search.search(focused_query())


def test_chunker_rejects_duplicate_exact_spans_with_different_ids(tmp_path: Path) -> None:
    class DuplicateSpanChunker:
        identity = {"backend": "duplicate-span-test", "implementation": "v1"}

        def split_text(self, text: str, *, document_id: str) -> list[TextChunk]:
            return [
                TextChunk(document_id, f"{document_id}:0000", text, 0, len(text)),
                TextChunk(document_id, f"{document_id}:0001", text, 0, len(text)),
            ]

    search, _store, _scorer = make_search(
        tmp_path,
        [candidate("d1", "same body")],
        chunker=DuplicateSpanChunker(),
    )

    with pytest.raises(ValueError, match="duplicate exact source span"):
        search.search(focused_query())


def test_malformed_scorer_output_remains_loud(tmp_path: Path) -> None:
    class MissingScore(RecordingScorer):
        def rank(self, query_text: str, chunks: Sequence[TextChunk]) -> tuple[ScoredChunk, ...]:
            self.calls.append((query_text, tuple(chunks)))
            return ()

    search, _store, _scorer = make_search(
        tmp_path, [candidate("d1", "body")], scorer=MissingScore()
    )

    with pytest.raises(ValueError, match="exactly one score"):
        search.search(focused_query())


def _complete_result(tmp_path: Path) -> PassageSearchResult:
    search, _store, _scorer = make_search(tmp_path, [candidate("d1", "body")])
    return search.search(focused_query())


def _full_two_passage_result(tmp_path: Path) -> PassageSearchResult:
    result = _complete_result(tmp_path)
    runner_up = replace(
        result.passages[0],
        passage_id="p-" + "1" * 64,
        raw_logit=0.5,
        rank=2,
        score_cache_key="1" * 64,
    )
    return replace(result, scored_passages=2, passages=(result.passages[0], runner_up))


def _incomplete_scoring_result(tmp_path: Path) -> PassageSearchResult:
    result = _complete_result(tmp_path)
    unscored_document = replace(
        result.documents[0],
        best_passage_id=None,
        best_passage_raw_logit=None,
    )
    return replace(
        result,
        status="incomplete",
        stopping_reason="scoring_failed",
        scored_documents=0,
        scored_passages=0,
        documents=(unscored_document,),
        passages=(),
    )


def test_result_rejects_invalid_status_and_stopping_reason(tmp_path: Path) -> None:
    result = _complete_result(tmp_path)

    with pytest.raises(ValueError, match="status"):
        replace(result, status="paused")
    with pytest.raises(ValueError, match="stopping_reason"):
        replace(result, status="incomplete", stopping_reason="unknown")
    with pytest.raises(ValueError, match="complete result"):
        replace(result, passages=(), scored_passages=0)
    with pytest.raises(ValueError, match="complete result"):
        replace(result, stopping_reason="no_evidence")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("returned_documents", -1),
        ("scored_documents", -1),
        ("scored_passages", -1),
        ("attempt_count", 0),
        ("attempt_count", True),
        ("requested_documents", 0),
        ("requested_documents", True),
        ("source_exhausted", 1),
    ],
)
def test_result_rejects_malformed_counts_and_flags(
    tmp_path: Path, field: str, value: object
) -> None:
    result = _complete_result(tmp_path)

    with pytest.raises(ValueError, match=field):
        replace(result, **{field: value})


def test_result_rejects_inconsistent_counts_and_noncontiguous_ranks(tmp_path: Path) -> None:
    result = _complete_result(tmp_path)
    bad_rank = replace(result.passages[0], rank=2)

    with pytest.raises(ValueError, match="returned_documents"):
        replace(result, returned_documents=0)
    with pytest.raises(ValueError, match="scored_documents"):
        replace(result, scored_documents=2)
    with pytest.raises(ValueError, match="scored_passages"):
        replace(result, scored_passages=0)
    with pytest.raises(ValueError, match="contiguous"):
        replace(result, passages=(bad_rank,))


def test_result_rejects_more_returned_than_requested_documents(tmp_path: Path) -> None:
    result = _complete_result(tmp_path)
    unscored_second = replace(
        result.documents[0],
        docid="d2",
        source_rank=2,
        best_passage_id=None,
        best_passage_raw_logit=None,
    )

    with pytest.raises(ValueError, match="returned_documents.*requested_documents"):
        replace(
            result,
            requested_documents=1,
            returned_documents=2,
            documents=(result.documents[0], unscored_second),
            source_exhausted=False,
        )


def test_result_rejects_duplicate_document_and_passage_ids(tmp_path: Path) -> None:
    result = _complete_result(tmp_path)
    duplicate_passage = replace(result.passages[0], rank=2)

    with pytest.raises(ValueError, match="document IDs.*unique"):
        replace(
            result,
            returned_documents=2,
            documents=(result.documents[0], result.documents[0]),
        )
    with pytest.raises(ValueError, match="passage IDs.*unique"):
        replace(
            result,
            scored_passages=2,
            passages=(result.passages[0], duplicate_passage),
        )


@pytest.mark.parametrize(
    "passage",
    [
        {"docid": "missing"},
        {"content_sha256": "0" * 64},
        {"source_rank": 2},
        {"source_score": 2.0},
    ],
)
def test_result_rejects_passages_not_bound_to_listed_source_document(
    tmp_path: Path, passage: dict[str, object]
) -> None:
    result = _complete_result(tmp_path)
    mismatched = replace(result.passages[0], **passage)

    with pytest.raises(ValueError, match="passage.*source document"):
        replace(result, passages=(mismatched,))


def test_complete_result_validates_scored_document_count_and_exhaustion(
    tmp_path: Path,
) -> None:
    result = _complete_result(tmp_path)

    with pytest.raises(ValueError, match="scored_documents"):
        replace(result, scored_documents=0)
    with pytest.raises(ValueError, match="source_exhausted"):
        replace(result, source_exhausted=False)


def test_full_passage_result_requires_each_document_actual_best_passage(
    tmp_path: Path,
) -> None:
    result = _full_two_passage_result(tmp_path)
    runner_up = result.passages[1]
    wrong_best = replace(
        result.documents[0],
        best_passage_id=runner_up.passage_id,
        best_passage_raw_logit=runner_up.raw_logit,
    )

    with pytest.raises(ValueError, match="best passage"):
        replace(result, documents=(wrong_best,))


def test_partial_passage_result_allows_unreturned_document_best(tmp_path: Path) -> None:
    result = _complete_result(tmp_path)
    unreturned_best = replace(
        result.documents[0],
        best_passage_id="p-" + "f" * 64,
        best_passage_raw_logit=2.0,
    )

    partial = replace(
        result,
        scored_passages=2,
        documents=(unreturned_best,),
    )

    assert partial.documents[0].best_passage_id == "p-" + "f" * 64


def test_incomplete_result_has_no_passages_scores_or_document_best(tmp_path: Path) -> None:
    complete = _complete_result(tmp_path)
    incomplete = _incomplete_scoring_result(tmp_path / "incomplete")

    with pytest.raises(ValueError, match="incomplete result"):
        replace(
            incomplete,
            scored_documents=1,
            scored_passages=1,
            passages=complete.passages,
        )
    with pytest.raises(ValueError, match="incomplete result"):
        replace(incomplete, scored_documents=1)
    with pytest.raises(ValueError, match="incomplete result"):
        replace(incomplete, scored_passages=1)
    with pytest.raises(ValueError, match="incomplete result"):
        replace(incomplete, documents=complete.documents)


def test_retrieval_unavailable_result_has_no_documents_and_is_not_exhausted(
    tmp_path: Path,
) -> None:
    unavailable_search, _store, _scorer = make_search(
        tmp_path / "unavailable", retriever=RecordingUnavailableRetriever()
    )
    unavailable = unavailable_search.search(focused_query())
    complete = _complete_result(tmp_path / "complete")
    unscored_document = replace(
        complete.documents[0],
        best_passage_id=None,
        best_passage_raw_logit=None,
    )

    with pytest.raises(ValueError, match="retrieval_unavailable"):
        replace(
            unavailable,
            returned_documents=1,
            documents=(unscored_document,),
        )
    with pytest.raises(ValueError, match="retrieval_unavailable"):
        replace(unavailable, source_exhausted=True)


def test_no_evidence_and_scoring_failure_exhaustion_matches_counts(tmp_path: Path) -> None:
    no_evidence_search, _store, _scorer = make_search(tmp_path / "empty", [])
    no_evidence = no_evidence_search.search(focused_query())
    scoring_failed = _incomplete_scoring_result(tmp_path / "scoring")

    with pytest.raises(ValueError, match="source_exhausted"):
        replace(no_evidence, source_exhausted=False)
    with pytest.raises(ValueError, match="source_exhausted"):
        replace(scoring_failed, source_exhausted=False)
