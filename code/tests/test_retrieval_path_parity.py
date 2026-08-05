from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import SimpleNamespace

from trec_rag.chunking import TextChunk
from trec_rag.deepagent_budget import ResearchTaskContext
from trec_rag.deepagent_research import ResearchTaskEnvelope, bind_research_task
from trec_rag.deepagent_retrieval import DeepAgentRetriever
from trec_rag.facet_extraction import GeneratedQueryPlan, Subnarrative, render_facet_queries
from trec_rag.facet_retrieval import run_facet_retrieval
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate
from trec_rag.topic_passage_search import TopicPassageSearch
from trec_rag.topics import Topic


class MemoryDocumentStore:
    def __init__(self) -> None:
        self._texts: dict[str, str] = {}

    def admit_text(self, text: str, *, expected_sha256: str | None = None):
        digest = sha256(text.encode("utf-8")).hexdigest()
        assert expected_sha256 in {None, digest}
        self._texts[digest] = text
        return SimpleNamespace(content_sha256=digest)

    def read_text(self, digest: str) -> str:
        return self._texts[digest]


class RecordingDepthRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[QueryVariant, int]] = []

    def retrieve(
        self,
        query: QueryVariant,
        *,
        depth: int,
    ) -> tuple[RetrievedCandidate, ...]:
        self.calls.append((query, depth))
        return tuple(
            RetrievedCandidate(
                topic_id=query.topic_id,
                variant_name=query.variant_name,
                retriever_name="recording-depth-retriever",
                query_text=query.query_text,
                docid=f"doc-{rank:04d}",
                rank=rank,
                score=float(1001 - rank),
                text=f"Exact source text for rank {rank}.",
            )
            for rank in range(1, depth + 1)
        )


class OnePassageChunker:
    identity = {"backend": "one-passage-fake", "version": 1}

    def split_text(self, text: str, *, document_id: str) -> tuple[TextChunk, ...]:
        return (TextChunk(document_id, f"{document_id}:0000", text, 0, len(text)),)


class Rank1000WinningScorer:
    identity = {"model": "rank-1000-winning-fake", "revision": "test"}

    def __init__(self) -> None:
        self.scored_document_ids: list[tuple[str, ...]] = []

    def cache_key(self, query_text: str, passage_text: str) -> str:
        return sha256(f"{query_text}\0{passage_text}".encode("utf-8")).hexdigest()

    def rank(
        self,
        _query_text: str,
        chunks: tuple[TextChunk, ...],
    ) -> tuple[float, ...]:
        self.scored_document_ids.append(tuple(chunk.document_id for chunk in chunks))
        return tuple(
            10_000.0
            if chunk.document_id == "doc-1000"
            else float(1000 - int(chunk.document_id.removeprefix("doc-")))
            for chunk in chunks
        )


class ReadableTopicPassageSearch(TopicPassageSearch):
    def __init__(self, *args, document_store: MemoryDocumentStore, **kwargs) -> None:
        self._readable_store = document_store
        super().__init__(*args, document_store=document_store, **kwargs)

    def read_text(self, content_sha256: str) -> str:
        return self._readable_store.read_text(content_sha256)


class TopicLedgerDouble:
    def __init__(self, topic_id: str, run_id: str) -> None:
        self.topic_id = topic_id
        self.run_id = run_id
        self.facets = {}
        self.searches = []
        self.handoffs = []
        self.completion: tuple[str, str] | None = None

    def add_facets(self, facets) -> None:
        self.facets.update((facet.subnarrative_id, facet) for facet in facets)

    def add_passage_search(self, result) -> None:
        self.searches.append(result)

    def add_researcher_handoff(self, handoff) -> None:
        self.handoffs.append(handoff)

    def set_completion(self, status: str, stopping_reason: str) -> None:
        self.completion = (status, stopping_reason)

    def topic_snapshot(self):
        return SimpleNamespace(
            status=self.completion[0] if self.completion else None,
            stopping_reason=self.completion[1] if self.completion else None,
        )


@dataclass(frozen=True)
class SharedFakeDependencies:
    passage_search: ReadableTopicPassageSearch
    retriever: RecordingDepthRetriever
    scorer: Rank1000WinningScorer


def shared_dependencies() -> SharedFakeDependencies:
    store = MemoryDocumentStore()
    retriever = RecordingDepthRetriever()
    scorer = Rank1000WinningScorer()
    search = ReadableTopicPassageSearch(
        "topic-1",
        document_store=store,
        retriever=retriever,
        chunker=OnePassageChunker(),
        scorer=scorer,
    )
    return SharedFakeDependencies(search, retriever, scorer)


def run_fixed_adapter(dependencies: SharedFakeDependencies):
    topic = Topic("topic-1", "Parity topic", "Original parity narrative.")
    subnarrative = Subnarrative(
        topic.id,
        "subnarrative-1",
        "Focused parity query.",
        ("focused parity bm25",),
    )
    plan = GeneratedQueryPlan(topic.id, (subnarrative,))
    queries = render_facet_queries(topic, plan).queries
    result = run_facet_retrieval(
        topic,
        queries,
        subnarratives=(subnarrative,),
        passage_search=dependencies.passage_search,
        retrieval_depth=1000,
        selection_k=100,
    )
    return result.lanes[1].passage_result


def run_agentic_search_tool(dependencies: SharedFakeDependencies):
    captured: dict[str, object] = {}
    records = TopicLedgerDouble("topic-1", "run-1")

    def agent_factory(_model, toolset):
        def invoke(_payload):
            toolset.update_retrieval_state(
                {
                    "add_needs": [
                        {
                            "need_id": "need-1",
                            "narrative_span": "Original parity narrative.",
                            "question": "What evidence addresses the parity topic?",
                        }
                    ],
                    "add_facets": [
                        {
                            "facet_id": "subnarrative-1",
                            "need_ids": ["need-1"],
                            "dimension": "parity",
                            "value": "Focused parity query.",
                            "origin": "narrative",
                        }
                    ],
                }
            )
            context = ResearchTaskContext(
                "researcher-1", 1, "focused", ("subnarrative-1",)
            )
            envelope = ResearchTaskEnvelope(
                research_task_id="researcher-1",
                round_index=1,
                depth="focused",
                motivating_ids=["subnarrative-1"],
                goal="Find focused parity evidence.",
            )
            assert toolset.budget.reserve_task(context).ok
            try:
                with bind_research_task(envelope):
                    captured["payload"] = json.loads(
                        toolset.search_passages(
                            "Focused parity query.",
                            ["subnarrative-1"],
                            "The focused facet has no evidence.",
                        )
                    )
            finally:
                toolset.budget.finish_task(context)
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return SimpleNamespace(invoke=invoke)

    result = DeepAgentRetriever(
        passage_search=dependencies.passage_search,
        agent_factory=agent_factory,
        tracing=SimpleNamespace(
            agent_span=lambda _narrative: _null_span(),
            retriever_span=lambda _query: _null_span(),
            snippet_span=lambda _document, _query: _null_span(),
            force_flush=lambda: True,
        ),
        model="test-model",
    ).retrieve(records, "Original parity narrative.")
    return result.searches[1], captured["payload"]


class _null_span:
    def __enter__(self):
        return None

    def __exit__(self, _exc_type, _exc, _traceback):
        return False


def _source_binding(row) -> tuple[object, ...]:
    return (
        row.passage_id,
        row.docid,
        row.content_sha256,
        row.source_rank,
        row.start_char,
        row.end_char,
        row.start_byte,
        row.end_byte,
        row.text_sha256,
        row.raw_logit,
        row.rank,
    )


def test_fixed_and_agentic_paths_share_real_depth_1000_passage_search() -> None:
    dependencies = shared_dependencies()

    fixed = run_fixed_adapter(dependencies)
    agentic, payload = run_agentic_search_tool(dependencies)

    assert fixed is not None
    assert [_source_binding(row) for row in fixed.passages] == [
        _source_binding(row) for row in agentic.passages
    ]
    assert len(fixed.passages) == 100
    assert fixed.passages[0].docid == "doc-1000"
    assert payload["documents_scored"] == 1000
    assert payload["documents_not_scored"] == 0
    assert len(payload["passages"]) == 100
    assert {depth for _query, depth in dependencies.retriever.calls} == {1000}
    assert all(
        len(document_ids) == 1000
        and "doc-1000" in document_ids
        for document_ids in dependencies.scorer.scored_document_ids
    )
