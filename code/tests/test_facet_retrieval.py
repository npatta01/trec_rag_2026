from __future__ import annotations

import hashlib
import json
import multiprocessing
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from trec_rag.facet_extraction import Subnarrative
from trec_rag.facet_retrieval import (
    DOCUMENT_MAX_LENGTH,
    DOCUMENT_PAIR_BUFFER_TOKENS,
    MIXEDBREAD_MODEL,
    MIXEDBREAD_REVISION,
    RERANK_DEPTH,
    RETRIEVAL_DEPTH,
    SELECTION_DEPTH,
    WINDOW_MAX_LENGTH,
    CoverageSelection,
    LaneDocumentScore,
    LaneRanking,
    MixedbreadCoverageScorer,
    PassageScore,
    RetrievalAuditCandidate,
    build_pyserini_retriever,
    build_retrieval_lanes,
    round_robin_select,
    run_facet_retrieval,
    score_selected_documents,
)
from trec_rag.pipeline_config import RetrieverConfig
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate
from trec_rag.remote_client import RemoteSearchResponse
from trec_rag.remote_config import RemotePyseriniConfig
from trec_rag.retrieval_cache import RetrievalCacheIntegrityError
from trec_rag.retrievers import PyseriniRemoteRetriever, request_cache_key
from trec_rag.topics import Topic
from trec_rag.topic_passage_search import (
    FocusedQuery,
    PassageSearchResult,
    SourceDocument,
    SourcePassage,
)


@pytest.fixture(autouse=True)
def _explicit_test_corpus_epoch(monkeypatch):
    monkeypatch.setenv("PYSERINI_CORPUS_EPOCH", "test-epoch")


def _topic(*, title: str = "ignored title") -> Topic:
    return Topic("58", title, "Compare wildfire smoke health effects and public responses.")


def _queries(topic: Topic) -> tuple[QueryVariant, ...]:
    return (
        QueryVariant(
            topic.id,
            "original",
            topic.narrative,
            "original_topic",
        ),
        QueryVariant(
            topic.id,
            "facet:subnarrative-1:q1",
            "wildfire smoke health effects particulate exposure",
            "generated_subnarrative_bm25",
        ),
        QueryVariant(
            topic.id,
            "facet:subnarrative-2:q1",
            "wildfire smoke public responses clean air shelter",
            "generated_subnarrative_bm25",
        ),
    )


def _subnarratives(topic: Topic) -> tuple[Subnarrative, ...]:
    return (
        Subnarrative(
            topic.id,
            "subnarrative-1",
            "wildfire smoke health effects",
            ("wildfire smoke health effects particulate exposure",),
        ),
        Subnarrative(
            topic.id,
            "subnarrative-2",
            "wildfire smoke public responses",
            ("wildfire smoke public responses clean air shelter",),
        ),
    )


def _single_flight_worker(
    worker_index: int,
    cache_dir: str,
    ready: multiprocessing.synchronize.Event,
    go: multiprocessing.synchronize.Event,
    entered_remote: multiprocessing.synchronize.Event,
    release_remote: multiprocessing.synchronize.Event,
    remote_calls: multiprocessing.sharedctypes.Synchronized,
    results: multiprocessing.queues.Queue,
) -> None:
    class Client:
        config = RemotePyseriniConfig(
            index_url="https://pyserini.test/search",
            api_token=None,
            hits=10,
            queries=(),
        )

        def search_raw(self, query_text: str, *, raw_sink=None) -> RemoteSearchResponse:
            with remote_calls.get_lock():
                remote_calls.value += 1
            entered_remote.set()
            if not release_remote.wait(10):
                raise TimeoutError("test remote gate was not released")
            raw = json.dumps(
                {
                    "api": "v1",
                    "index": "climbmix-400b",
                    "query": {"text": query_text},
                    "candidates": [
                        {
                            "doc": "single-flight body",
                            "docid": "doc-single-flight",
                            "rank": 1,
                            "score": 3.0,
                        }
                    ],
                }
            ).encode("utf-8")
            if raw_sink is not None:
                raw_sink(raw)
            return RemoteSearchResponse(
                raw=raw,
                payload=json.loads(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )

    ready.set()
    if not go.wait(10):
        results.put({"error": "test start gate was not released"})
        return
    if worker_index == 1 and not entered_remote.wait(10):
        results.put({"error": "first worker did not enter remote call"})
        return
    query = QueryVariant("31", "original", "single flight query", "original_topic")
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=10,
        index="climbmix-400b",
    )
    try:
        retriever = PyseriniRemoteRetriever(config, cache_dir=Path(cache_dir), client=Client())
        candidates = retriever.retrieve(query)
        results.put(
            {
                "docids": [candidate.docid for candidate in candidates],
                "summary": retriever.cache_summary(),
            }
        )
    except Exception as exc:  # pragma: no cover - surfaced by the parent assertion
        results.put({"error": f"{type(exc).__name__}: {exc}"})


def test_shared_pyserini_cache_single_flights_identical_concurrent_requests(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    start_events = [context.Event(), context.Event()]
    go = context.Event()
    entered_remote = context.Event()
    release_remote = context.Event()
    remote_calls = context.Value("i", 0)
    results = context.Queue()
    processes = [
        context.Process(
            target=_single_flight_worker,
            args=(
                index,
                str(tmp_path),
                start_events[index],
                go,
                entered_remote,
                release_remote,
                remote_calls,
                results,
            ),
        )
        for index in range(2)
    ]

    for process in processes:
        process.start()
    try:
        for event in start_events:
            assert event.wait(10)
        go.set()
        assert entered_remote.wait(10)
        deadline = time.monotonic() + 1.0
        while remote_calls.value < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert remote_calls.value == 1
    finally:
        release_remote.set()
        for process in processes:
            process.join(15)
            if process.is_alive():
                process.terminate()
                process.join()

    assert all(process.exitcode == 0 for process in processes)
    outcomes = sorted(
        (results.get(timeout=5) for _ in processes),
        key=lambda outcome: outcome.get("summary", {}).get("hits", -1),
    )
    assert [outcome["docids"] for outcome in outcomes] == [
        ["doc-single-flight"],
        ["doc-single-flight"],
    ]
    assert sorted(outcome["summary"]["hits"] for outcome in outcomes) == [0, 1]
    assert sorted(outcome["summary"]["writes"] for outcome in outcomes) == [0, 1]
    assert len(list((tmp_path / "v2" / "attempts").rglob("manifest.json"))) == 1


@pytest.mark.parametrize(
    ("state", "message"),
    (
        ("metadata-only", "missing response artifact"),
        ("response-only", "missing provenance sidecar"),
        ("stale-temporary", "incomplete cache publication state"),
        ("conflicting-pair", "cache response hash mismatch"),
    ),
)
def test_shared_pyserini_cache_fails_closed_on_partial_or_conflicting_state(
    tmp_path: Path,
    state: str,
    message: str,
) -> None:
    query = QueryVariant("31", "original", "partial cache query", "original_topic")
    config = RetrieverConfig(
        name="climbmix_bm25",
        type="pyserini_remote",
        query_variants=("original",),
        hits=10,
        index="climbmix-400b",
    )

    class FailingClient:
        config = RemotePyseriniConfig(
            index_url="https://pyserini.test/search",
            api_token=None,
            hits=10,
            queries=(),
        )

        calls = 0

        def search(self, _query: str) -> dict[str, object]:
            self.calls += 1
            raise AssertionError("partial or conflicting cache must not call remote search")

    client = FailingClient()
    retriever = PyseriniRemoteRetriever(config, cache_dir=tmp_path, client=client)
    cache = retriever.retrieval_cache
    identity = retriever._transport_identity(query)
    entry = cache._entry_path(identity.request_key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    if state == "metadata-only":
        entry.mkdir()
        (entry / "transport-manifest.json").write_text("{}", encoding="utf-8")
    elif state == "response-only":
        entry.mkdir()
        (entry / "raw.body.gz").write_bytes(
            __import__("gzip").compress(
                b'{"api":"v1","index":"climbmix-400b","candidates":[]}', mtime=0
            )
        )
    elif state == "stale-temporary":
        entry.with_name(f".{entry.name}.crashed.tmp").mkdir()
    else:
        entry.mkdir()
        (entry / "raw.body.gz").write_bytes(
            __import__("gzip").compress(
                b'{"api":"v1","index":"climbmix-400b","candidates":[]}', mtime=0
            )
        )
        (entry / "transport-manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RetrievalCacheIntegrityError, match="partial|transport|incomplete"):
        retriever.retrieve(query)

    assert client.calls == 0
    assert retriever.cache_summary() == {
        "enabled": True,
        "retrieval_cache_schema": "organizer-retrieval-cache-v2",
        "requests": 0,
        "hit_artifacts": [],
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "bypasses": 0,
    }


def _score(
    lane,
    docid: str,
    aggregate_rank: int,
    *,
    aggregate_score: float | None = None,
    text: str | None = None,
    source_rank: int | None = None,
    source_score: float | None = None,
) -> LaneDocumentScore:
    document_text = text if text is not None else f"text for {docid}"
    return LaneDocumentScore(
        topic_id=lane.retrieval_query.topic_id,
        lane_name=lane.retrieval_query.variant_name,
        bm25_query=lane.retrieval_query.query_text,
        bm25_query_sha256=lane.bm25_query_sha256,
        semantic_query=lane.scoring_query.query_text,
        semantic_query_sha256=lane.semantic_query_sha256,
        docid=docid,
        text=document_text,
        bm25_rank=source_rank if source_rank is not None else aggregate_rank,
        bm25_score=(
            source_score
            if source_score is not None
            else float(1000 - aggregate_rank)
        ),
        aggregate_rank=aggregate_rank,
        aggregate_score=(
            4.0
            if aggregate_score is None
            else aggregate_score
        ),
        long_document_raw_logit=4.0,
        weighted_passage_raw_logit=3.0,
        within_document_span_support=2,
        winning_passages=(PassageScore(0, 0, len(document_text), 3.0, 1),),
    )


def _lane(lane, docids: list[str]) -> LaneRanking:
    return LaneRanking(
        lane=lane,
        retrieval_returned_count=len(docids),
        retrieval_retained_count=len(docids),
        retrieval_audit_candidates=tuple(
            RetrievalAuditCandidate(
                docid=docid,
                bm25_rank=rank,
                bm25_score=float(1000 - rank),
                text_sha256="0" * 64,
            )
            for rank, docid in enumerate(docids, start=1)
        ),
        documents=tuple(
            _score(lane, docid, rank)
            for rank, docid in enumerate(docids, start=1)
        ),
    )


def test_lane_binding_is_narrative_only_and_title_independent() -> None:
    topic = _topic()
    changed_title = replace(topic, title="a completely different synthetic title")

    lanes = build_retrieval_lanes(topic, _queries(topic), _subnarratives(topic))
    changed = build_retrieval_lanes(
        changed_title,
        _queries(changed_title),
        _subnarratives(changed_title),
    )

    assert lanes == changed
    assert [lane.retrieval_query.variant_name for lane in lanes] == [
        "original",
        "facet:subnarrative-1:text",
        "facet:subnarrative-2:text",
    ]
    assert lanes[0].retrieval_query.query_text == topic.narrative
    assert lanes[0].scoring_query.query_text == topic.narrative
    assert all(lane.retrieval_query.source_type != "structured_facet" for lane in lanes)


def test_saved_bm25_suggestions_produce_one_active_subnarrative_lane() -> None:
    topic = _topic()
    subnarrative = Subnarrative(
        topic.id,
        "subnarrative-1",
        "Generated shared semantic need",
        ("first lexical query", "second lexical query", "third lexical query"),
    )
    queries = (
        QueryVariant(topic.id, "original", topic.narrative, "original_topic"),
        *(
            QueryVariant(
                topic.id,
                f"facet:subnarrative-1:q{index}",
                query,
                "generated_subnarrative_bm25",
            )
            for index, query in enumerate(subnarrative.bm25_queries, start=1)
        ),
    )

    lanes = build_retrieval_lanes(topic, queries, (subnarrative,))

    assert [lane.retrieval_query.variant_name for lane in lanes] == [
        "original",
        "facet:subnarrative-1:text",
    ]
    assert lanes[1].retrieval_query.query_text == "Generated shared semantic need"
    assert lanes[1].scoring_query.query_text == "Generated shared semantic need"
    assert lanes[1].retrieval_query.source_type == "generated_subnarrative"
    assert lanes[1].scoring_query.source_type == "generated_subnarrative"

    with pytest.raises(ValueError, match="ordered"):
        build_retrieval_lanes(
            topic,
            (queries[0], queries[2], queries[1], queries[3]),
            (subnarrative,),
        )


def test_retrieval_lanes_require_the_complete_canonical_query_sequence() -> None:
    topic = _topic()
    queries = _queries(topic)
    subnarratives = _subnarratives(topic)

    with pytest.raises(ValueError, match="complete ordered"):
        build_retrieval_lanes(
            topic,
            (queries[1], queries[0], queries[2]),
            subnarratives,
        )

    extra_original = QueryVariant(
        topic.id,
        "shadow-original",
        topic.narrative,
        "original_topic",
    )
    with pytest.raises(ValueError, match="complete ordered"):
        build_retrieval_lanes(
            topic,
            (*queries, extra_original),
            subnarratives,
        )


@pytest.mark.parametrize(
    "subnarrative",
    (
        Subnarrative(
            "58",
            "caller-chosen-id",
            "wildfire smoke health effects",
            ("wildfire smoke health effects",),
        ),
        Subnarrative(
            "58",
            "subnarrative-1",
            "wildfire smoke\x00health effects",
            ("wildfire smoke health effects",),
        ),
        Subnarrative(
            "58",
            "subnarrative-1",
            "wildfire smoke health effects",
            (),
        ),
    ),
)
def test_retrieval_lanes_reject_noncanonical_direct_subnarratives(
    subnarrative: Subnarrative,
) -> None:
    topic = _topic()
    queries = (
        QueryVariant(topic.id, "original", topic.narrative, "original_topic"),
        *(
            QueryVariant(
                topic.id,
                f"facet:{subnarrative.subnarrative_id}:q{index}",
                query,
                "generated_subnarrative_bm25",
            )
            for index, query in enumerate(subnarrative.bm25_queries, start=1)
        ),
    )

    with pytest.raises(ValueError, match="canonical"):
        build_retrieval_lanes(topic, queries, (subnarrative,))


def _shared_depth_1000_result(topic: Topic) -> PassageSearchResult:
    documents: list[SourceDocument] = []
    passages: list[SourcePassage] = []
    chunker_identity = {"backend": "test-chunker", "implementation": "v1"}
    for source_rank in range(1, RETRIEVAL_DEPTH + 1):
        docid = f"doc-{source_rank:04d}"
        text = f"source text for {docid}"
        text_sha256 = hashlib.sha256(text.encode()).hexdigest()
        passage_id = f"p-{docid}-0000"
        raw_logit = float(RETRIEVAL_DEPTH - source_rank)
        if source_rank == RETRIEVAL_DEPTH:
            raw_logit = float(RETRIEVAL_DEPTH + 1)
        documents.append(
            SourceDocument(
                docid,
                text_sha256,
                source_rank,
                float(RETRIEVAL_DEPTH - source_rank),
                passage_id,
                raw_logit,
            )
        )
        if source_rank <= RERANK_DEPTH - 1 or source_rank == RETRIEVAL_DEPTH:
            passages.append(
                SourcePassage(
                    passage_id,
                    docid,
                    text_sha256,
                    source_rank,
                    float(RETRIEVAL_DEPTH - source_rank),
                    0,
                    len(text),
                    0,
                    len(text.encode()),
                    text_sha256,
                    text,
                    raw_logit,
                    len(passages) + 1,
                    f"cache-{docid}",
                    text_sha256,
                    chunker_identity,
                )
            )
    return PassageSearchResult(
        FocusedQuery("original", topic.narrative, "original"),
        "complete",
        None,
        RETRIEVAL_DEPTH,
        RETRIEVAL_DEPTH,
        RETRIEVAL_DEPTH,
        RETRIEVAL_DEPTH,
        tuple(documents),
        tuple(passages),
        1,
        False,
    )


def test_fixed_path_uses_shared_passage_search_and_never_old_document_scorer() -> None:
    topic = _topic()
    shared_search = _RecordingTopicPassageSearch(_shared_depth_1000_result(topic))

    class PoisonLaneScorer:
        def score_lane(self, *_args, **_kwargs):
            raise AssertionError("the v2 fixed path must not call the old lane scorer")

    result = run_facet_retrieval(
        topic,
        _queries(topic),
        passage_search=shared_search,
        subnarratives=_subnarratives(topic),
        legacy_scorer=PoisonLaneScorer(),
    )

    assert len(shared_search.queries) == len(result.lanes)
    assert len(result.lanes[0].passage_result.passages) == RERANK_DEPTH
    assert "p-doc-1000-0000" in {
        passage.passage_id for passage in result.lanes[0].passage_result.passages
    }


class _RecordingTopicPassageSearch:
    def __init__(self, result: PassageSearchResult) -> None:
        self.result = result
        self.queries: list[FocusedQuery] = []
        self._texts = {
            document.content_sha256: f"source text for {document.docid}"
            for document in result.documents
        }

    def search(self, query: FocusedQuery) -> PassageSearchResult:
        self.queries.append(query)
        return replace(self.result, query=query)

    def read_text(self, content_sha256: str) -> str:
        return self._texts[content_sha256]


def test_round_robin_advances_duplicates_and_preserves_each_lane_score() -> None:
    topic = _topic()
    original, health, response = build_retrieval_lanes(
        topic,
        _queries(topic),
        _subnarratives(topic),
    )
    lanes = (
        _lane(original, ["d1", "d2", "d3"]),
        _lane(health, ["d1", "d4"]),
        _lane(response, ["d1"]),
    )

    selection = round_robin_select(lanes, limit=3)

    assert [row.docid for row in selection.documents] == ["d1", "d4", "d2"]
    assert selection.documents[0].selected_from_lane == "original"
    assert [score.lane_name for score in selection.documents[0].lane_scores] == [
        "original",
        "facet:subnarrative-1:text",
        "facet:subnarrative-2:text",
    ]
    assert selection.lane_statuses[1].duplicate_skips == 1
    assert [event.action for event in selection.trace] == [
        "selected",
        "duplicate_skip",
        "selected",
        "duplicate_skip",
        "exhausted",
        "selected",
    ]
    assert [(event.slot, event.lane_name, event.docid) for event in selection.trace[:3]] == [
        (1, "original", "d1"),
        (2, "facet:subnarrative-1:text", "d1"),
        (2, "facet:subnarrative-1:text", "d4"),
    ]
    assert selection.complete is True
    assert selection.all_lanes_exhausted is False


def test_round_robin_reports_exhaustion_and_is_stable() -> None:
    topic = _topic()
    original, health, _response = build_retrieval_lanes(
        topic,
        _queries(topic),
        _subnarratives(topic),
    )
    lanes = (_lane(original, ["d2", "d1"]), _lane(health, ["d2"]))

    first = round_robin_select(lanes, limit=5)
    second = round_robin_select(lanes, limit=5)

    assert first == second
    assert [row.docid for row in first.documents] == ["d2", "d1"]
    assert first.complete is False
    assert first.all_lanes_exhausted is True
    assert all(status.exhausted for status in first.lane_statuses)


def test_fake_end_to_end_retrieves_1000_reranks_100_and_selects_100() -> None:
    topic = _topic()
    queries = _queries(topic)

    class FakeRetriever:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
            self.calls.append(query.variant_name)
            prefix = query.variant_name.replace(":", "-")
            return [
                RetrievedCandidate(
                    topic.id,
                    query.variant_name,
                    "climbmix_bm25",
                    query.query_text,
                    f"{prefix}-{rank:04d}",
                    rank,
                    float(RETRIEVAL_DEPTH - rank),
                    f"document {prefix} {rank}",
                )
                for rank in range(1, RETRIEVAL_DEPTH + 1)
            ]

    class FakeScorer:
        def __init__(self) -> None:
            self.depths: list[int] = []

        def score_lane(
            self,
            lane_topic: Topic,
            lane,
            candidates: list[RetrievedCandidate],
        ) -> tuple[LaneDocumentScore, ...]:
            assert lane_topic.narrative == topic.narrative
            self.depths.append(len(candidates))
            return tuple(
                LaneDocumentScore(
                    topic_id=row.topic_id,
                    lane_name=row.variant_name,
                    bm25_query=lane.retrieval_query.query_text,
                    bm25_query_sha256=lane.bm25_query_sha256,
                    semantic_query=lane.scoring_query.query_text,
                    semantic_query_sha256=lane.semantic_query_sha256,
                    docid=row.docid,
                    text=row.text,
                    bm25_rank=row.rank,
                    bm25_score=row.score,
                    aggregate_rank=rank,
                    aggregate_score=1.75,
                    long_document_raw_logit=2.0,
                    weighted_passage_raw_logit=1.0,
                    within_document_span_support=1,
                    winning_passages=(
                        PassageScore(0, 0, len(row.text), 1.0, 1),
                    ),
                )
                for rank, row in enumerate(reversed(candidates), start=1)
            )

    retriever = FakeRetriever()
    scorer = FakeScorer()

    result = run_facet_retrieval(
        topic,
        queries,
        retriever,
        scorer,
        subnarratives=_subnarratives(topic),
    )

    assert retriever.calls == [
        "original",
        "facet:subnarrative-1:text",
        "facet:subnarrative-2:text",
    ]
    assert scorer.depths == [RERANK_DEPTH, RERANK_DEPTH, RERANK_DEPTH]
    assert [lane.retrieval_returned_count for lane in result.lanes] == [
        RETRIEVAL_DEPTH,
        RETRIEVAL_DEPTH,
        RETRIEVAL_DEPTH,
    ]
    assert all(lane.retrieval_exhausted is False for lane in result.lanes)
    assert len(result.selection.documents) == SELECTION_DEPTH
    assert result.selection.complete is True
    assert len({row.docid for row in result.selection.documents}) == SELECTION_DEPTH
    assert result.original_only_control == result.lanes[0].documents
    assert len(result.union_pool) == 3 * RERANK_DEPTH
    assert [row.first_seen_lane for row in result.union_pool[:2]] == [
        "original",
        "original",
    ]
    assert all(lane.audit_only_count == RETRIEVAL_DEPTH - RERANK_DEPTH for lane in result.lanes)
    assert all(len(lane.retrieval_audit_candidates) == RETRIEVAL_DEPTH for lane in result.lanes)
    assert all(len(lane.retrieval_audit_sha256) == 64 for lane in result.lanes)
    assert result.lanes[0].audit_only_candidates[0] == RetrievalAuditCandidate(
        docid="original-0101",
        bm25_rank=101,
        bm25_score=899.0,
        text_sha256="a21fbf7664b3d3b61c444a3c714a08b767eba3ecfc435441fb047a555c859095",
    )


def test_selection_reranking_uses_semantic_query_without_bm25_expansions() -> None:
    topic = _topic()
    queries = _queries(topic)
    observed: list[str] = []

    class Retriever:
        def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
            return [
                RetrievedCandidate(
                    topic.id,
                    query.variant_name,
                    "climbmix_bm25",
                    query.query_text,
                    f"{query.variant_name}-doc",
                    1,
                    10.0,
                    f"document for {query.variant_name}",
                )
            ]

    class Scorer:
        def score_lane(self, _topic_value, lane, candidates):
            observed.append(candidates[0].query_text)
            return (
                _score(
                    lane,
                    candidates[0].docid,
                    1,
                    text=candidates[0].text,
                    source_rank=candidates[0].rank,
                    source_score=candidates[0].score,
                ),
            )

    run_facet_retrieval(
        topic,
        queries,
        Retriever(),
        Scorer(),
        subnarratives=_subnarratives(topic),
        selection_k=3,
    )

    assert observed == [
        topic.narrative,
        "wildfire smoke health effects",
        "wildfire smoke public responses",
    ]
    assert all("particulate exposure" not in query for query in observed)
    assert all("clean air shelter" not in query for query in observed)


def test_full_top1000_text_conflict_fails_before_any_scoring() -> None:
    topic = _topic()
    calls = 0

    class Retriever:
        def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
            prefix = query.variant_name.replace(":", "-")
            rows = [
                RetrievedCandidate(
                    topic.id,
                    query.variant_name,
                    "climbmix_bm25",
                    query.query_text,
                    f"{prefix}-{rank:04d}",
                    rank,
                    float(200 - rank),
                    f"text {prefix} {rank}",
                )
                for rank in range(1, 102)
            ]
            if query.variant_name == "original":
                rows[-1] = replace(rows[-1], docid="shared", text="first source text")
            elif query.variant_name == "facet:subnarrative-1:text":
                rows[0] = replace(rows[0], docid="shared", text="conflicting source text")
            return rows

    class Scorer:
        def score_lane(self, *_args):
            nonlocal calls
            calls += 1
            raise RuntimeError("scoring must not begin before audit identity validation")

    with pytest.raises(ValueError, match="source text identity"):
        run_facet_retrieval(
            topic,
            _queries(topic),
            Retriever(),
            Scorer(),
            subnarratives=_subnarratives(topic),
        )
    assert calls == 0


def test_duplicate_docid_within_one_retrieval_lane_fails_before_scoring() -> None:
    topic = _topic()
    query = _queries(topic)[0]
    rows = [
        RetrievedCandidate(
            topic.id,
            query.variant_name,
            "climbmix_bm25",
            query.query_text,
            "duplicate",
            rank,
            float(3 - rank),
            "same source text",
        )
        for rank in (1, 2)
    ]

    class Retriever:
        def retrieve(self, _query: QueryVariant) -> list[RetrievedCandidate]:
            return rows

    class Scorer:
        def score_lane(self, *_args):
            raise RuntimeError("scoring must not begin")

    with pytest.raises(ValueError, match="duplicate document ID"):
        run_facet_retrieval(
            topic,
            (query,),
            Retriever(),
            Scorer(),
            subnarratives=(),
        )


@pytest.mark.parametrize("rank", (1.0, "1"))
def test_retrieval_boundary_rejects_noninteger_candidate_ranks_before_scoring(
    rank: object,
) -> None:
    topic = _topic()
    query = _queries(topic)[0]
    candidate = RetrievedCandidate(
        topic.id,
        query.variant_name,
        "climbmix_bm25",
        query.query_text,
        "invalid-rank",
        rank,  # type: ignore[arg-type]
        1.0,
        "candidate text",
    )
    calls = 0

    class Retriever:
        def retrieve(self, _query: QueryVariant) -> list[RetrievedCandidate]:
            return [candidate]

    class Scorer:
        def score_lane(self, *_args: object) -> tuple[LaneDocumentScore, ...]:
            nonlocal calls
            calls += 1
            raise RuntimeError("scoring must not begin for an invalid rank")

    with pytest.raises(ValueError, match="rank must be a positive integer"):
        run_facet_retrieval(
            topic,
            (query,),
            Retriever(),
            Scorer(),
            subnarratives=(),
        )
    assert calls == 0


@pytest.mark.parametrize("score", (True, False))
def test_retrieval_boundary_rejects_boolean_candidate_scores_before_scoring(
    score: bool,
) -> None:
    topic = _topic()
    query = _queries(topic)[0]
    candidate = RetrievedCandidate(
        topic.id,
        query.variant_name,
        "climbmix_bm25",
        query.query_text,
        "invalid-score",
        1,
        score,
        "candidate text",
    )
    calls = 0

    class Retriever:
        def retrieve(self, _query: QueryVariant) -> list[RetrievedCandidate]:
            return [candidate]

    class Scorer:
        def score_lane(self, *_args: object) -> tuple[LaneDocumentScore, ...]:
            nonlocal calls
            calls += 1
            raise RuntimeError("scoring must not begin for an invalid score")

    with pytest.raises(ValueError, match="score must be a finite number"):
        run_facet_retrieval(
            topic,
            (query,),
            Retriever(),
            Scorer(),
            subnarratives=(),
        )
    assert calls == 0


def test_subnarrative_cross_scores_every_selected_document_without_fusion() -> None:
    topic = _topic()
    original, health, _response = build_retrieval_lanes(
        topic,
        _queries(topic),
        _subnarratives(topic),
    )
    selection: CoverageSelection = round_robin_select(
        (_lane(original, ["d1", "d2"]),),
        limit=2,
    )
    subnarratives = (
        Subnarrative(
            topic.id,
            "subnarrative-1",
            "health effects exact anchor",
            ("bm25 health expansion", "alternate health expansion"),
        ),
        Subnarrative(
            topic.id,
            "subnarrative-2",
            "public responses exact anchor",
            ("bm25 response expansion",),
        ),
    )

    class CrossScorer:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def score_lane(
            self,
            _topic_value: Topic,
            lane,
            candidates: list[RetrievedCandidate],
        ) -> tuple[LaneDocumentScore, ...]:
            self.queries.append(lane.scoring_query.query_text)
            return tuple(
                LaneDocumentScore(
                    topic_id=candidate.topic_id,
                    lane_name=lane.retrieval_query.variant_name,
                    bm25_query=lane.retrieval_query.query_text,
                    bm25_query_sha256=lane.bm25_query_sha256,
                    semantic_query=lane.scoring_query.query_text,
                    semantic_query_sha256=lane.semantic_query_sha256,
                    docid=candidate.docid,
                    text=candidate.text,
                    bm25_rank=candidate.rank,
                    bm25_score=candidate.score,
                    aggregate_rank=rank,
                    aggregate_score=4.0,
                    long_document_raw_logit=4.0,
                    weighted_passage_raw_logit=3.0,
                    within_document_span_support=2,
                    winning_passages=(
                        PassageScore(0, 0, len(candidate.text), 3.0, 1),
                    ),
                )
                for rank, candidate in enumerate(candidates, start=1)
            )

    scorer = CrossScorer()
    rows = score_selected_documents(topic, selection, subnarratives, scorer)

    assert scorer.queries == [
        "health effects exact anchor",
        "public responses exact anchor",
    ]
    assert [(row.docid, row.subnarrative_id) for row in rows] == [
        ("d1", "subnarrative-1"),
        ("d1", "subnarrative-2"),
        ("d2", "subnarrative-1"),
        ("d2", "subnarrative-2"),
    ]
    assert [row.score.aggregate_score for row in rows] == [4.0, 4.0, 4.0, 4.0]
    assert all(row.semantic_query_sha256 for row in rows)
    assert [row.bm25_queries for row in rows[:2]] == [
        ("bm25 health expansion", "alternate health expansion"),
        ("bm25 response expansion",),
    ]
    assert all(row.bm25_query_sha256s for row in rows)


@pytest.mark.parametrize(
    "case",
    [
        "wrong-topic",
        "duplicate-docid",
        "empty-text",
        "changed-source-text",
        "noncontiguous-rank",
        "boolean-selection-rank",
        "requested-count-too-small",
        "incomplete-with-live-lane",
        "exhaustion-state-disagrees",
    ],
)
def test_selected_document_cross_scoring_rejects_invalid_selection_before_scoring(
    case: str,
) -> None:
    topic = _topic()
    original = build_retrieval_lanes(topic, (_queries(topic)[0],), ())[0]
    selection = round_robin_select(
        (_lane(original, ["d1", "d2", "d3"]),),
        limit=2,
    )
    first, second = selection.documents
    if case == "wrong-topic":
        selection = replace(
            selection,
            documents=(replace(first, topic_id="other-topic"), second),
        )
    elif case == "duplicate-docid":
        selection = replace(
            selection,
            documents=(first, replace(second, docid=first.docid)),
        )
    elif case == "empty-text":
        selection = replace(
            selection,
            documents=(replace(first, text="  "), second),
        )
    elif case == "changed-source-text":
        selection = replace(
            selection,
            documents=(replace(first, text="changed source text"), second),
        )
    elif case == "noncontiguous-rank":
        selection = replace(
            selection,
            documents=(first, replace(second, selection_rank=3)),
        )
    elif case == "boolean-selection-rank":
        selection = replace(
            selection,
            documents=(replace(first, selection_rank=True), second),
        )
    elif case == "requested-count-too-small":
        selection = replace(selection, requested_count=1)
    elif case == "incomplete-with-live-lane":
        selection = replace(selection, requested_count=3)
    elif case == "exhaustion-state-disagrees":
        selection = replace(selection, all_lanes_exhausted=True)

    calls = 0

    class Scorer:
        def score_lane(self, *_args):
            nonlocal calls
            calls += 1
            raise RuntimeError("invalid selection must fail before scoring")

    with pytest.raises(ValueError, match="selection|selected document"):
        score_selected_documents(
            topic,
            selection,
            _subnarratives(topic)[:1],
            Scorer(),
        )
    assert calls == 0


def test_lane_scorer_cannot_change_retrieval_provenance() -> None:
    topic = _topic()
    query = _queries(topic)[0]
    lane = build_retrieval_lanes(topic, (query,), ())[0]
    candidate = RetrievedCandidate(
        topic.id,
        query.variant_name,
        "climbmix_bm25",
        query.query_text,
        "d1",
        1,
        12.5,
        "exact retrieved text",
    )
    changed = LaneDocumentScore(
        topic_id=topic.id,
        lane_name=query.variant_name,
        bm25_query=lane.retrieval_query.query_text,
        bm25_query_sha256=lane.bm25_query_sha256,
        semantic_query=lane.scoring_query.query_text,
        semantic_query_sha256=lane.semantic_query_sha256,
        docid="d1",
        text="changed text",
        bm25_rank=1,
        bm25_score=12.5,
        aggregate_rank=1,
        aggregate_score=3.25,
        long_document_raw_logit=3.0,
        weighted_passage_raw_logit=3.0,
        within_document_span_support=1,
        winning_passages=(PassageScore(0, 0, 12, 3.0, 1),),
    )

    class Retriever:
        def retrieve(self, _query: QueryVariant) -> list[RetrievedCandidate]:
            return [candidate]

    class Scorer:
        def score_lane(self, *_args) -> tuple[LaneDocumentScore, ...]:
            return (changed,)

    with pytest.raises(ValueError, match="retrieval provenance"):
        run_facet_retrieval(
            topic,
            (query,),
            Retriever(),
            Scorer(),
            subnarratives=(),
        )


@pytest.mark.parametrize(
    ("changes", "case"),
    [
        ({"long_document_raw_logit": float("nan")}, "nonfinite-long-logit"),
        ({"within_document_span_support": -1}, "negative-span-support"),
        ({"within_document_span_support": 7}, "over-cap-span-support"),
        ({"within_document_span_support": True}, "boolean-span-support"),
        (
            {"winning_passages": (PassageScore(0, -1, 5, 3.0, 1),)},
            "negative-passage-offset",
        ),
        (
            {"winning_passages": (PassageScore(0, 0, 21, 3.0, 1),)},
            "passage-past-document",
        ),
        (
            {"winning_passages": (PassageScore(0, 0, 20, float("nan"), 1),)},
            "nonfinite-passage-logit",
        ),
        (
            {"winning_passages": (PassageScore(0, 0, 20, 3.0, 2),)},
            "noncontiguous-weight-rank",
        ),
        (
            {
                "winning_passages": (
                    PassageScore(0, 0, 10, 2.0, 1),
                    PassageScore(1, 10, 20, 3.0, 2),
                ),
                "weighted_passage_raw_logit": 2.3125,
            },
            "passages-not-score-ordered",
        ),
        ({"weighted_passage_raw_logit": 2.0}, "weighted-prefix-mismatch"),
        ({"aggregate_score": 99.0}, "aggregate-formula-mismatch"),
        ({"aggregate_rank": True}, "boolean-aggregate-rank"),
        ({"score_representation": "probability"}, "wrong-score-representation"),
    ],
)
def test_lane_scores_reject_invalid_components(changes, case: str) -> None:
    topic = _topic()
    query = _queries(topic)[0]
    lane = build_retrieval_lanes(topic, (query,), ())[0]
    candidate = RetrievedCandidate(
        topic.id,
        query.variant_name,
        "climbmix_bm25",
        query.query_text,
        "d1",
        1,
        10.0,
        "x" * 20,
    )
    valid = _score(
        lane,
        "d1",
        1,
        text=candidate.text,
        source_rank=1,
        source_score=10.0,
    )
    invalid = replace(valid, **changes)

    class Retriever:
        def retrieve(self, _query: QueryVariant) -> list[RetrievedCandidate]:
            return [candidate]

    class Scorer:
        def score_lane(self, *_args) -> tuple[LaneDocumentScore, ...]:
            return (invalid,)

    with pytest.raises(ValueError, match="score|support|passage|formula|rank"):
        run_facet_retrieval(
            topic,
            (query,),
            Retriever(),
            Scorer(),
            subnarratives=(),
        )


def test_real_adapters_pin_retrieval_and_content_addressed_score_identity(tmp_path) -> None:
    class Client:
        config = RemotePyseriniConfig(
            "http://api.example.test/v1/climbmix-400b/search",
            None,
            RETRIEVAL_DEPTH,
            (),
        )

    retriever = build_pyserini_retriever(
        tmp_path / "retrieval", client=Client(), corpus_epoch="test-epoch"
    )
    topic = _topic()
    original, health, _response = build_retrieval_lanes(
        topic,
        _queries(topic),
        _subnarratives(topic),
    )
    original_key = request_cache_key(
        retriever.config,
        original.retrieval_query,
        index_url=Client.config.index_url,
    )
    health_key = request_cache_key(
        retriever.config,
        health.retrieval_query,
        index_url=Client.config.index_url,
    )
    scorer = MixedbreadCoverageScorer(
        artifact_dir=tmp_path / "artifacts",
        score_cache_root=tmp_path / "score-cache",
    )

    assert retriever.config.hits == RETRIEVAL_DEPTH
    assert retriever.config.index == "climbmix-400b"
    assert original_key != health_key
    assert scorer.identity == {
        "model": MIXEDBREAD_MODEL,
        "model_revision": MIXEDBREAD_REVISION,
        "backend_version": "5.6.0",
        "score_representation": "raw_logits",
        "inference_dtype": "bfloat16",
        "document_max_length": DOCUMENT_MAX_LENGTH,
        "document_pair_buffer_tokens": DOCUMENT_PAIR_BUFFER_TOKENS,
        "window_max_length": WINDOW_MAX_LENGTH,
        "chunk_max_characters": 3500,
        "chunk_overlap_characters": 350,
        "input_policy": "trec_rag_whitespace_v1",
    }
    assert scorer.document_cache.cache_key(query_text="q1", text="doc") != (
        scorer.document_cache.cache_key(query_text="q2", text="doc")
    )


def test_configured_retriever_binds_index_and_candidate_depth(tmp_path: Path) -> None:
    class Client:
        config = RemotePyseriniConfig(
            "http://api.example.test/v1/climbmix-test/search",
            None,
            500,
            (),
        )

    retriever = build_pyserini_retriever(
        tmp_path,
        index="climbmix-test",
        hits=500,
        corpus_epoch="test-epoch",
        client=Client(),
    )

    assert retriever.config.index == "climbmix-test"
    assert retriever.config.hits == 500


def test_configured_retriever_forwards_cache_only_without_constructing_transport(
    tmp_path: Path,
) -> None:
    """Catches the fixed-path builder silently falling back to online retrieval."""
    class Client:
        config = RemotePyseriniConfig(
            "http://api.example.test/v1/climbmix-test/search",
            None,
            500,
            (),
        )

    retriever = build_pyserini_retriever(
        tmp_path,
        index="climbmix-test",
        hits=500,
        corpus_epoch="test-epoch",
        client=Client(),
        cache_only=True,
    )

    assert retriever.cache_only is True


def test_configured_retrieval_depth_limits_audit_and_reranking() -> None:
    topic = _topic()
    query = _queries(topic)[0]
    scored_depths: list[int] = []

    class Retriever:
        def retrieve(self, lane_query: QueryVariant) -> list[RetrievedCandidate]:
            return [
                RetrievedCandidate(
                    topic.id,
                    lane_query.variant_name,
                    "climbmix_bm25",
                    lane_query.query_text,
                    f"d{rank}",
                    rank,
                    float(4 - rank),
                    f"document {rank}",
                )
                for rank in range(1, 4)
            ]

    class Scorer:
        def score_lane(self, _topic_value, lane, candidates):
            scored_depths.append(len(candidates))
            return tuple(
                _score(
                    lane,
                    candidate.docid,
                    rank,
                    text=candidate.text,
                    source_rank=candidate.rank,
                    source_score=candidate.score,
                )
                for rank, candidate in enumerate(candidates, start=1)
            )

    result = run_facet_retrieval(
        topic,
        (query,),
        Retriever(),
        Scorer(),
        subnarratives=(),
        retrieval_depth=2,
        rerank_depth=1,
        selection_k=1,
    )

    assert scored_depths == [1]
    assert result.lanes[0].retrieval_requested_depth == 2
    assert result.lanes[0].retrieval_retained_count == 2
    assert len(result.lanes[0].retrieval_audit_candidates) == 2


def test_mixedbread_adapter_records_formula_components_and_reuses_cache(tmp_path) -> None:
    topic = _topic()
    query = build_retrieval_lanes(topic, (_queries(topic)[0],), ())[0]
    text = "a" * 900 + "b" * 900
    candidate = RetrievedCandidate(
        topic.id,
        query.scoring_query.variant_name,
        "climbmix_bm25",
        query.scoring_query.query_text,
        "d1",
        7,
        12.5,
        text,
    )
    loads: list[tuple[str, int, str]] = []

    class Parameter:
        dtype = "torch.bfloat16"

    class Model:
        def __init__(self, scores: list[float]) -> None:
            self.scores = scores

        def parameters(self):
            return [Parameter()]

        def predict(self, pairs, **_kwargs):
            assert len(pairs) == len(self.scores)
            return self.scores

    def loader(model: str, *, revision: str, max_length: int, device: str):
        loads.append((model, max_length, device))
        assert revision == MIXEDBREAD_REVISION
        return Model([2.0] if max_length != WINDOW_MAX_LENGTH else [3.0])

    scorer = MixedbreadCoverageScorer(
        artifact_dir=tmp_path / "artifacts",
        score_cache_root=tmp_path / "score-cache",
        device="cuda",
        model_loader=loader,
    )

    rows = scorer.score_lane(topic, query, [candidate])

    assert loads == [
        (MIXEDBREAD_MODEL, DOCUMENT_MAX_LENGTH - DOCUMENT_PAIR_BUFFER_TOKENS, "cuda"),
        (MIXEDBREAD_MODEL, WINDOW_MAX_LENGTH, "cuda"),
    ]
    assert len(rows) == 1
    assert rows[0].bm25_rank == 7
    assert rows[0].long_document_raw_logit == 2.0
    assert rows[0].weighted_passage_raw_logit == 3.0
    assert rows[0].within_document_span_support == 1
    assert rows[0].aggregate_score == 2.75
    assert [(row.start_char, row.end_char, row.raw_logit) for row in rows[0].winning_passages] == [
        (0, 1800, 3.0),
    ]

    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("warm artifacts and caches must avoid model loading")

    warm = MixedbreadCoverageScorer(
        artifact_dir=tmp_path / "artifacts",
        score_cache_root=tmp_path / "score-cache",
        model_loader=forbidden_loader,
    )
    assert warm.score_lane(topic, query, [candidate]) == rows


def _ledger_topic(topic_id: str) -> Topic:
    return Topic(topic_id, "", f"Question for ledger topic {topic_id}")


def _ledger_lane(topic: Topic):
    query = QueryVariant(topic.id, "original", topic.narrative, "original_topic")
    return build_retrieval_lanes(topic, (query,), ())[0]


def _ledger_candidate(topic: Topic) -> RetrievedCandidate:
    lane = _ledger_lane(topic)
    return RetrievedCandidate(
        topic.id,
        lane.scoring_query.variant_name,
        "climbmix_bm25",
        lane.scoring_query.query_text,
        f"doc-{topic.id}",
        1,
        1.0,
        f"Document body for ledger topic {topic.id}.",
    )


def _ledger_model_loader(*_args, **_kwargs):
    return _ledger_model_loader_with_score(1.0)(*_args, **_kwargs)


def _ledger_model_loader_with_score(score: float):
    def loader(*_args, **_kwargs):
        class Parameter:
            dtype = "torch.bfloat16"

        class Model:
            def parameters(self):
                return [Parameter()]

            def predict(self, pairs, **_kwargs):
                return [score] * len(pairs)

        return Model()

    return loader


def _gated_ledger_model_loader(score: float, barrier: threading.Barrier):
    def loader(*_args, max_length, **_kwargs):
        class Parameter:
            dtype = "torch.bfloat16"

        class Model:
            def parameters(self):
                return [Parameter()]

            def predict(self, pairs, **_kwargs):
                if max_length != WINDOW_MAX_LENGTH:
                    barrier.wait(timeout=10)
                return [score] * len(pairs)

        return Model()

    return loader


def test_topics_publish_to_disjoint_ledgers_when_scored_concurrently(tmp_path: Path) -> None:
    root = tmp_path / "scorer-ledger"
    topics = (_ledger_topic("topic-a"), _ledger_topic("topic-b"))
    barrier = threading.Barrier(len(topics))
    outcomes: list[
        tuple[Topic, MixedbreadCoverageScorer, tuple[LaneDocumentScore, ...], Path, Path]
    ] = []
    errors: list[BaseException] = []
    outcome_lock = threading.Lock()

    def run(topic: Topic) -> None:
        scorer = MixedbreadCoverageScorer(
            artifact_dir=root,
            score_cache_root=tmp_path / f"score-cache-{topic.id}",
            device="cpu",
            model_loader=_ledger_model_loader,
        )
        try:
            barrier.wait(timeout=10)
            rows = scorer.score_lane(topic, _ledger_lane(topic), [_ledger_candidate(topic)])
            with outcome_lock:
                outcomes.append(
                    (topic, scorer, rows, scorer.document_score_path, scorer.window_score_path)
                )
        except BaseException as exc:  # pragma: no cover - surfaced by assertions
            with outcome_lock:
                errors.append(exc)

    threads = [threading.Thread(target=run, args=(topic,)) for topic in topics]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)

    assert not errors
    assert len(outcomes) == len(topics)
    assert not (root / "document_scores.jsonl").exists()
    assert not (root / "window_scores.jsonl").exists()
    for topic, scorer, rows, document_score_path, window_score_path in outcomes:
        assert len(rows) == 1
        assert document_score_path == root / topic.id / "document_scores.jsonl"
        assert window_score_path == root / topic.id / "window_scores.jsonl"
        for path in (document_score_path, window_score_path):
            payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            assert payload
            assert {row["topic_id"] for row in payload} == {topic.id}


@pytest.mark.parametrize("artifact_name", ("document_scores.jsonl", "window_scores.jsonl"))
def test_partial_score_ledger_is_rejected(tmp_path: Path, artifact_name: str) -> None:
    topic = _ledger_topic("partial-topic")
    root = tmp_path / "scorer-ledger" / topic.id
    root.mkdir(parents=True)
    (root / artifact_name).write_text('{"topic_id":', encoding="utf-8")
    scorer = MixedbreadCoverageScorer(
        artifact_dir=tmp_path / "scorer-ledger",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        model_loader=lambda *_args, **_kwargs: pytest.fail("partial ledger must fail before model loading"),
    )

    with pytest.raises(ValueError, match="invalid JSONL row"):
        scorer.score_lane(topic, _ledger_lane(topic), [_ledger_candidate(topic)])


def test_stale_temporary_score_ledger_is_rejected(tmp_path: Path) -> None:
    topic = _ledger_topic("stale-topic")
    root = tmp_path / "scorer-ledger" / topic.id
    root.mkdir(parents=True)
    (root / ".document_scores.jsonl.crashed.tmp").write_text('{"partial":', encoding="utf-8")
    scorer = MixedbreadCoverageScorer(
        artifact_dir=tmp_path / "scorer-ledger",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        model_loader=lambda *_args, **_kwargs: pytest.fail("stale ledger must fail before model loading"),
    )

    with pytest.raises(ValueError, match="temporary|incomplete"):
        scorer.score_lane(topic, _ledger_lane(topic), [_ledger_candidate(topic)])


def test_identical_score_ledger_reruns_converge_to_identical_bytes(tmp_path: Path) -> None:
    topic = _ledger_topic("rerun-topic")
    lane = _ledger_lane(topic)
    candidate = _ledger_candidate(topic)
    first = MixedbreadCoverageScorer(
        artifact_dir=tmp_path / "scorer-ledger",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        model_loader=_ledger_model_loader,
    )
    first.score_lane(topic, lane, [candidate])
    expected_rows = first.score_lane(topic, lane, [candidate])
    first_bytes = {
        path.name: path.read_bytes()
        for path in (first.document_score_path, first.window_score_path)
    }

    second = MixedbreadCoverageScorer(
        artifact_dir=tmp_path / "scorer-ledger",
        score_cache_root=tmp_path / "score-cache",
        device="cpu",
        model_loader=lambda *_args, **_kwargs: pytest.fail("identical rerun should use cache"),
    )
    assert second.score_lane(topic, lane, [candidate]) == expected_rows
    assert {
        path.name: path.read_bytes()
        for path in (second.document_score_path, second.window_score_path)
    } == first_bytes


def test_contradictory_concurrent_ledger_publication_fails_closed(tmp_path: Path) -> None:
    topic = _ledger_topic("conflict-topic")
    root = tmp_path / "scorer-ledger"
    barrier = threading.Barrier(2)
    scorers = (
        MixedbreadCoverageScorer(
            artifact_dir=root,
            score_cache_root=tmp_path / "score-cache-a",
            device="cpu",
            model_loader=_gated_ledger_model_loader(1.0, barrier),
        ),
        MixedbreadCoverageScorer(
            artifact_dir=root,
            score_cache_root=tmp_path / "score-cache-b",
            device="cpu",
            model_loader=_gated_ledger_model_loader(2.0, barrier),
        ),
    )
    outcomes: list[tuple[MixedbreadCoverageScorer, BaseException | None]] = []
    lock = threading.Lock()

    def publish(scorer: MixedbreadCoverageScorer) -> None:
        try:
            scorer.score_lane(topic, _ledger_lane(topic), [_ledger_candidate(topic)])
        except BaseException as exc:  # pragma: no cover - collected below
            with lock:
                outcomes.append((scorer, exc))
        else:
            with lock:
                outcomes.append((scorer, None))

    threads = [threading.Thread(target=publish, args=(scorer,)) for scorer in scorers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)

    assert len(outcomes) == 2
    assert sum(error is None for _scorer, error in outcomes) == 1
    assert sum(isinstance(error, ValueError) for _scorer, error in outcomes) == 1
    path = root / topic.id / "document_scores.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["score"] in {1.0, 2.0}
