from __future__ import annotations

from collections.abc import Mapping
import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Barrier, Event
from typing import Callable, Sequence
from unittest.mock import ANY

import pytest
import trec_rag.deepagent_retrieval as deepagent_retrieval
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_openrouter import ChatOpenRouter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor
from pydantic import Field

import trec_rag.deepagent_tracing as deepagent_tracing
from trec_rag.deepagent_retrieval import (
    AgentRetrievalError,
    AgentSearch,
    DeepAgentRetriever,
    _create_agent,
    reciprocal_rank_fuse,
)
from trec_rag.deepagent_snippets import (
    RelevantSnippet,
    SnippetExtractionResult,
    SnippetPage,
)
from trec_rag.deepagent_tracing import (
    MAX_TRACE_DOCUMENT_ID_CHARACTERS,
    MAX_TRACE_DOCUMENTS,
    REDACTED_CONTENT,
    create_retrieval_tracing,
)
from trec_rag.pipeline_models import RetrievedCandidate, jsonable


def _candidate(
    *,
    query: str,
    variant: str,
    docid: str,
    rank: int,
    text: str = "Document excerpt.",
) -> RetrievedCandidate:
    return RetrievedCandidate(
        topic_id="internal-topic",
        variant_name=variant,
        retriever_name="fake",
        query_text=query,
        docid=docid,
        rank=rank,
        score=float(100 - rank),
        text=text,
    )


class FakeRetriever:
    def __init__(self, *, candidate_count: int = 1) -> None:
        self.queries = []
        self.candidate_count = candidate_count

    def retrieve(self, query):
        self.queries.append(query)
        return [
            _candidate(
                query=query.query_text,
                variant=query.variant_name,
                docid=f"{query.variant_name}-doc-{rank}",
                rank=rank,
                text=f"{query.query_text} excerpt {rank}",
            )
            for rank in range(1, self.candidate_count + 1)
        ]


class SlowFollowupRetriever(FakeRetriever):
    """Make concurrent pre-append budget checks overlap deterministically."""

    def retrieve(self, query):
        if query.variant_name != "original":
            time.sleep(0.05)
        return super().retrieve(query)


class ControlledFailureRetriever(FakeRetriever):
    """Pause one failing follow-up while later calls queue behind the SDK lock."""

    def __init__(self) -> None:
        super().__init__()
        self.attempted_queries: list[str] = []
        self.failure_started = Event()
        self.release_failure = Event()

    def retrieve(self, query):
        self.attempted_queries.append(query.query_text)
        if query.query_text == "failing query":
            self.failure_started.set()
            if not self.release_failure.wait(timeout=2):
                raise AssertionError("test did not release the controlled failure")
            raise RuntimeError("controlled retrieval failure")
        self.queries.append(query)
        return [
            _candidate(
                query=query.query_text,
                variant=query.variant_name,
                docid="shared-doc",
                rank=1,
                text="stable shared excerpt",
            )
        ]


@dataclass
class FakeAgent:
    invoke_callback: Callable[[dict[str, object]], object]

    def invoke(self, payload: dict[str, object]) -> object:
        return self.invoke_callback(payload)


class FakeTracing:
    def __init__(self) -> None:
        self.flushes = 0
        self.search_records: list[dict[str, object]] = []
        self.snippet_records: list[dict[str, object]] = []
        self.result_records: list[dict[str, object]] = []

    @contextmanager
    def agent_span(self, _narrative: str):
        yield self

    @contextmanager
    def retriever_span(self, _query: str):
        yield self

    @contextmanager
    def snippet_span(self, _document_id: str, _focus_query: str):
        yield self

    def record_search(self, **evidence: object) -> None:
        self.search_records.append(evidence)

    def record_page(self, **evidence: object) -> None:
        self.snippet_records.append(evidence)

    def record_result(self, **evidence: object) -> None:
        self.result_records.append(evidence)

    def force_flush(self) -> bool:
        self.flushes += 1
        return True


class RecordingSnippetExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str | None]] = []

    def extract(
        self,
        document_id: str,
        document_text: str,
        focus_query: str,
        cursor: str | None = None,
    ) -> SnippetExtractionResult:
        self.calls.append((document_id, document_text, focus_query, cursor))
        if cursor == "invalid-cursor":
            raise ValueError("cursor leaked /private/cache/path")
        if focus_query == "extractor failure":
            raise RuntimeError("ranker leaked /private/model/path")
        snippet_text = document_text[-40:]
        return SnippetExtractionResult(
            page=SnippetPage(
                document_id=document_id,
                focus_query=focus_query,
                snippets=(
                    RelevantSnippet(
                        chunk_id="chunk-1",
                        start_char=len(document_text) - len(snippet_text),
                        end_char=len(document_text),
                        text=snippet_text,
                        relevance_score=0.75,
                    ),
                ),
                next_cursor=None,
            ),
            cache_status="miss",
            ranker_backend="private-ranker",
            page_offset=0,
        )


class FailedFlushTracing(FakeTracing):
    def force_flush(self) -> bool:
        self.flushes += 1
        return False


class FailedSnippetLifecycleTracing(FakeTracing):
    def __init__(self, failed_phase: str) -> None:
        super().__init__()
        self.failed_phase = failed_phase

    @contextmanager
    def snippet_span(self, _document_id: str, _focus_query: str):
        if self.failed_phase == "enter":
            raise RuntimeError("trace span enter failed")
        yield self
        if self.failed_phase == "exit":
            raise RuntimeError("trace span exit failed")


class RejectedEvidenceTracing(FakeTracing):
    def __init__(self, rejected_phase: str) -> None:
        super().__init__()
        self.rejected_phase = rejected_phase

    def record_search(self, **evidence: object) -> None:
        if self.rejected_phase == "search":
            raise ValueError("trace evidence rejected")
        super().record_search(**evidence)

    def record_page(self, **evidence: object) -> None:
        if self.rejected_phase == "snippet":
            raise ValueError("trace evidence rejected")
        super().record_page(**evidence)

    def record_result(self, **evidence: object) -> None:
        if self.rejected_phase == "result":
            raise ValueError("trace evidence rejected")
        super().record_result(**evidence)


class CaptureChatModel(FakeMessagesListChatModel):
    """Offline chat model that records the tools Deep Agents exposes to it."""

    captured_tool_names: list[str] = Field(default_factory=list)
    captured_bind_settings: list[dict[str, object]] = Field(default_factory=list)

    def bind_tools(
        self, tools: Sequence[object], **kwargs: object
    ) -> "CaptureChatModel":
        self.captured_tool_names = [str(getattr(tool, "name", "")) for tool in tools]
        self.captured_bind_settings.append(dict(kwargs))
        return self


@pytest.fixture
def isolated_real_tracing() -> None:
    def reset() -> None:
        instrumentor = LangChainInstrumentor()
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
        deepagent_tracing._LIVE_PROVIDER_CACHE.clear()
        deepagent_tracing._ACTIVE_INSTRUMENTATION = None

    reset()
    yield
    reset()


def _sdk(
    retriever: FakeRetriever,
    agent_factory: Callable[
        [str, Callable[[str], str], Callable[[str, str, str | None], str]],
        FakeAgent,
    ],
    *,
    tracing: FakeTracing | None = None,
    snippet_extractor: object | None = None,
) -> DeepAgentRetriever:
    return DeepAgentRetriever(
        retriever=retriever,
        agent_factory=agent_factory,
        tracing=tracing or FakeTracing(),
        model="test-model",
        snippet_extractor=snippet_extractor,
    )


def test_fusion_deduplicates_and_sums_reciprocal_ranks() -> None:
    searches = (
        AgentSearch(
            query="original narrative",
            kind="original",
            cache_status="miss",
            candidates=(
                _candidate(
                    query="original narrative",
                    variant="original",
                    docid="shared",
                    rank=1,
                    text="First shared text.",
                ),
                _candidate(
                    query="original narrative",
                    variant="original",
                    docid="original-only",
                    rank=2,
                ),
            ),
        ),
        AgentSearch(
            query="targeted follow-up",
            kind="followup",
            cache_status="miss",
            candidates=(
                _candidate(
                    query="targeted follow-up",
                    variant="followup-query",
                    docid="shared",
                    rank=1,
                    text="",
                ),
                _candidate(
                    query="targeted follow-up",
                    variant="followup-query",
                    docid="followup-only",
                    rank=3,
                ),
            ),
        ),
    )

    fused = reciprocal_rank_fuse(searches, limit=20, rrf_k=60)

    assert [row.docid for row in fused] == ["shared", "original-only", "followup-only"]
    assert fused[0].score == pytest.approx(2 / 61)
    assert fused[0].text == "First shared text."
    assert fused[0].provenance[0]["query_kind"] == "original"
    assert [row["source_rank"] for row in fused[0].provenance] == [1, 1]


def test_fusion_provenance_is_deeply_immutable_and_jsonable() -> None:
    fused = reciprocal_rank_fuse(
        (
            AgentSearch(
                query="original narrative",
                kind="original",
                cache_status="miss",
                candidates=(
                    _candidate(
                        query="original narrative",
                        variant="original",
                        docid="doc-a",
                        rank=1,
                    ),
                ),
            ),
        )
    )

    with pytest.raises((AttributeError, TypeError)):
        fused[0].provenance.append({"query_kind": "followup"})
    assert isinstance(fused[0].provenance[0], Mapping)
    assert not isinstance(fused[0].provenance[0], dict)
    with pytest.raises(TypeError):
        fused[0].provenance[0]["query_kind"] = "followup"
    with pytest.raises((AttributeError, TypeError)):
        fused[0].provenance[0].query_kind = "followup"

    serialized = jsonable(fused[0])
    assert serialized["provenance"] == [
        {
            "cache_status": "miss",
            "query": "original narrative",
            "query_kind": "original",
            "retriever_name": "fake",
            "source_rank": 1,
            "source_score": 99.0,
            "variant_name": "original",
        }
    ]


def test_retrieve_searches_untouched_narrative_before_agent_followups() -> None:
    fake_retriever = FakeRetriever()
    factory_queries = []
    factory_models = []

    def agent_factory(
        model: str,
        search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        factory_models.append(model)
        factory_queries.extend(fake_retriever.queries)
        return FakeAgent(
            lambda _payload: (
                search_tool("targeted follow-up"),
                {
                    "messages": [
                        {"role": "assistant", "content": "Coverage is sufficient."}
                    ]
                },
            )[1]
        )

    result = _sdk(fake_retriever, agent_factory).retrieve(
        "  supplied narrative exactly  "
    )

    assert [query.query_text for query in factory_queries] == [
        "  supplied narrative exactly  "
    ]
    assert factory_models == ["test-model"]
    assert fake_retriever.queries[0].query_text == "  supplied narrative exactly  "
    assert [search.kind for search in result.searches] == ["original", "followup"]
    assert result.narrative == "  supplied narrative exactly  "
    assert result.rationale == "Coverage is sufficient."


def test_agent_sees_only_candidate_metadata_and_can_extract_original_and_followup_snippets() -> (
    None
):
    sentinel = "FULL-DOCUMENT-SENTINEL-9d64"
    extractor = RecordingSnippetExtractor()
    initial_candidates: list[dict[str, object]] = []
    followup_candidates: list[dict[str, object]] = []
    snippet_payloads: list[dict[str, object]] = []

    class SentinelRetriever(FakeRetriever):
        def retrieve(self, query):
            candidates = super().retrieve(query)
            candidate = candidates[0]
            return [
                RetrievedCandidate(
                    topic_id=candidate.topic_id,
                    variant_name=candidate.variant_name,
                    retriever_name=candidate.retriever_name,
                    query_text=candidate.query_text,
                    docid=candidate.docid,
                    rank=candidate.rank,
                    score=candidate.score,
                    text=f"Document body for {candidate.docid}. {sentinel}",
                )
            ]

    def agent_factory(
        _model: str,
        search_tool: Callable[[str], str],
        snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        def invoke(payload: dict[str, object]) -> object:
            content = str(payload["messages"][0]["content"])
            initial_candidates.extend(
                json.loads(content.split("Bounded original results:\n", 1)[1])
            )
            snippet_payloads.append(
                json.loads(snippet_tool("original-doc-1", "original focus", None))
            )
            followup = json.loads(search_tool("targeted query"))
            followup_candidates.extend(followup["candidates"])
            followup_docid = str(followup["candidates"][0]["docid"])
            snippet_payloads.append(
                json.loads(snippet_tool(followup_docid, "followup focus", None))
            )
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(
        SentinelRetriever(), agent_factory, snippet_extractor=extractor
    ).retrieve("narrative")

    assert initial_candidates == [
        {"docid": "original-doc-1", "rank": 1, "score": 99.0, "text_length": 61}
    ]
    assert set(followup_candidates[0]) == {"docid", "rank", "score", "text_length"}
    assert sentinel not in json.dumps(initial_candidates)
    assert sentinel not in json.dumps(followup_candidates)
    assert [payload["focus_query"] for payload in snippet_payloads] == [
        "original focus",
        "followup focus",
    ]
    assert all(
        set(payload) == {"document_id", "focus_query", "snippets", "next_cursor"}
        for payload in snippet_payloads
    )
    assert all(
        sentinel in payload["snippets"][0]["text"] for payload in snippet_payloads
    )
    assert all("cache_status" not in payload for payload in snippet_payloads)
    assert all("ranker_backend" not in payload for payload in snippet_payloads)
    assert all(sentinel in search.candidates[0].text for search in result.searches)
    assert all(sentinel in candidate.text for candidate in result.candidates)


def test_snippet_tool_returns_safe_errors_for_invalid_requests_and_extractor_failures() -> (
    None
):
    extractor = RecordingSnippetExtractor()
    tool_payloads: list[dict[str, object]] = []

    def agent_factory(
        _model: str,
        _search_tool: Callable[[str], str],
        snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            tool_payloads.extend(
                json.loads(response)
                for response in (
                    snippet_tool("unknown-doc", "focus", None),
                    snippet_tool("original-doc-1", "   ", None),
                    snippet_tool("original-doc-1", "focus", "invalid-cursor"),
                    snippet_tool("original-doc-1", "extractor failure", None),
                )
            )
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    _sdk(FakeRetriever(), agent_factory, snippet_extractor=extractor).retrieve(
        "narrative"
    )

    assert tool_payloads == [
        {"error": "unknown document_id"},
        {"error": "focus_query must be non-empty text"},
        {"error": "invalid cursor"},
        {"error": "snippet extraction failed"},
    ]
    assert "/private/" not in json.dumps(tool_payloads)


def test_retrieve_raises_when_one_document_id_has_conflicting_nonempty_text() -> None:
    class ConflictingRetriever(FakeRetriever):
        def retrieve(self, query):
            return [
                _candidate(
                    query=query.query_text,
                    variant=query.variant_name,
                    docid="shared-doc",
                    rank=1,
                    text=f"{query.variant_name} body",
                )
            ]

    def agent_factory(
        _model: str,
        search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        return FakeAgent(
            lambda _payload: (
                search_tool("targeted query"),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        )

    with pytest.raises(AgentRetrievalError) as raised:
        _sdk(ConflictingRetriever(), agent_factory).retrieve("narrative")

    assert str(raised.value.__cause__) == "conflicting text for document_id shared-doc"
    assert [search.kind for search in raised.value.searches] == ["original"]


def test_retrieve_rejects_empty_narrative_before_external_calls() -> None:
    fake_retriever = FakeRetriever()
    factory_calls = []

    with pytest.raises(ValueError, match="narrative must be non-empty text"):
        _sdk(
            fake_retriever,
            lambda _model, _search_tool, _snippet_tool: factory_calls.append(True),  # type: ignore[return-value]
        ).retrieve("   ")

    assert fake_retriever.queries == []
    assert factory_calls == []


def test_retrieve_limits_successful_followups_to_three_and_marks_budget_exhaustion() -> (
    None
):
    fake_retriever = FakeRetriever()
    tool_results = []

    def agent_factory(
        _model: str,
        search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            tool_results.extend(search_tool(f"query {number}") for number in range(4))
            return {
                "messages": [
                    {"role": "assistant", "content": "Stopped after coverage."}
                ]
            }

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")

    assert [query.query_text for query in fake_retriever.queries] == [
        "narrative",
        "query 0",
        "query 1",
        "query 2",
    ]
    assert [search.kind for search in result.searches] == [
        "original",
        "followup",
        "followup",
        "followup",
    ]
    assert json.loads(tool_results[-1])["error"] == "search budget exhausted"
    assert result.stopping_reason == "search_budget_exhausted"


def test_retrieve_serializes_concurrent_followups_at_three_successes() -> None:
    fake_retriever = SlowFollowupRetriever()
    tool_results = []

    def agent_factory(
        _model: str,
        search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            barrier = Barrier(4)

            def call_tool(number: int) -> str:
                barrier.wait(timeout=2)
                return search_tool(f"concurrent query {number}")

            with ThreadPoolExecutor(max_workers=4) as pool:
                tool_results.extend(pool.map(call_tool, range(4)))
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")
    payloads = [json.loads(item) for item in tool_results]

    assert len(fake_retriever.queries) == 4
    assert len(result.searches) == 4
    assert [search.kind for search in result.searches] == [
        "original",
        "followup",
        "followup",
        "followup",
    ]
    assert (
        sum(payload.get("error") == "search budget exhausted" for payload in payloads)
        == 1
    )
    assert result.stopping_reason == "search_budget_exhausted"


def test_retrieve_rejects_concurrent_duplicate_without_using_budget() -> None:
    fake_retriever = SlowFollowupRetriever()
    tool_results = []

    def agent_factory(
        _model: str,
        search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            barrier = Barrier(2)

            def call_tool(_number: int) -> str:
                barrier.wait(timeout=2)
                return search_tool("same concurrent query")

            with ThreadPoolExecutor(max_workers=2) as pool:
                tool_results.extend(pool.map(call_tool, range(2)))
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")
    payloads = [json.loads(item) for item in tool_results]

    assert len(fake_retriever.queries) == 2
    assert len(result.searches) == 2
    assert (
        sum(payload.get("error") == "duplicate follow-up query" for payload in payloads)
        == 1
    )
    assert result.stopping_reason == "agent_completed"


def test_retrieve_concurrent_blank_followups_do_not_use_budget() -> None:
    fake_retriever = SlowFollowupRetriever()
    blank_results = []
    successful_results = []

    def agent_factory(
        _model: str,
        search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            barrier = Barrier(4)

            def call_blank(number: int) -> str:
                barrier.wait(timeout=2)
                return search_tool(" " * (number + 1))

            with ThreadPoolExecutor(max_workers=4) as pool:
                blank_results.extend(pool.map(call_blank, range(4)))
            successful_results.extend(
                search_tool(f"query {number}") for number in range(3)
            )
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")

    assert [query.query_text for query in fake_retriever.queries] == [
        "narrative",
        "query 0",
        "query 1",
        "query 2",
    ]
    assert all(
        json.loads(item) == {"error": "query must be non-empty text"}
        for item in blank_results
    )
    assert [json.loads(item)["remaining_budget"] for item in successful_results] == [
        2,
        1,
        0,
    ]
    assert result.stopping_reason == "agent_completed"


def test_retrieve_failure_releases_lock_and_preserves_serialized_success_order() -> (
    None
):
    fake_retriever = ControlledFailureRetriever()
    tool_results = []

    def agent_factory(
        _model: str,
        search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            queued = Barrier(4)

            def call_success(number: int) -> str:
                queued.wait(timeout=2)
                return search_tool(f"query {number}")

            with ThreadPoolExecutor(max_workers=4) as pool:
                failed = pool.submit(search_tool, "failing query")
                assert fake_retriever.failure_started.wait(timeout=2)
                successful = [pool.submit(call_success, number) for number in range(3)]
                queued.wait(timeout=2)
                fake_retriever.release_failure.set()
                with pytest.raises(RuntimeError, match="controlled retrieval failure"):
                    failed.result(timeout=2)
                tool_results.extend(future.result(timeout=2) for future in successful)
            tool_results.append(search_tool("query beyond budget"))
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")
    recorded_queries = [query.query_text for query in fake_retriever.queries]

    assert fake_retriever.attempted_queries[0:2] == ["narrative", "failing query"]
    assert "failing query" not in recorded_queries
    assert len(recorded_queries) == 4
    assert [search.query for search in result.searches] == recorded_queries
    assert [
        item["query"] for item in result.candidates[0].provenance
    ] == recorded_queries
    assert sorted(
        json.loads(item)["remaining_budget"] for item in tool_results[:3]
    ) == [
        0,
        1,
        2,
    ]
    assert json.loads(tool_results[-1]) == {"error": "search budget exhausted"}
    assert result.stopping_reason == "search_budget_exhausted"


def test_retrieve_rejects_blank_and_original_duplicate_followups_without_using_budget() -> (
    None
):
    fake_retriever = FakeRetriever()
    tool_results = []

    def agent_factory(
        _model: str,
        search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            tool_results.extend(
                [
                    search_tool("   "),
                    search_tool("narrative"),
                    search_tool("query 0"),
                    search_tool("query 1"),
                    search_tool("query 2"),
                    search_tool("query 3"),
                ]
            )
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")

    assert [query.query_text for query in fake_retriever.queries] == [
        "narrative",
        "query 0",
        "query 1",
        "query 2",
    ]
    assert json.loads(tool_results[0])["error"] == "query must be non-empty text"
    assert json.loads(tool_results[1])["error"] == "duplicate follow-up query"
    assert json.loads(tool_results[-1])["error"] == "search budget exhausted"
    assert result.stopping_reason == "search_budget_exhausted"


def test_retrieve_uses_stable_hashes_for_private_cache_variants() -> None:
    fake_retriever = FakeRetriever()

    def agent_factory(
        _model: str,
        search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        return FakeAgent(
            lambda _payload: (
                search_tool("specific missing aspect"),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        )

    narrative = "Narrative whose ID must not be public."
    _sdk(fake_retriever, agent_factory).retrieve(narrative)

    assert (
        fake_retriever.queries[0].topic_id
        == hashlib.sha256(narrative.encode()).hexdigest()[:16]
    )
    assert fake_retriever.queries[0].variant_name == "original"
    assert (
        fake_retriever.queries[1].variant_name
        == "followup-" + hashlib.sha256(b"specific missing aspect").hexdigest()[:16]
    )


def test_retrieve_wraps_agent_failure_with_completed_searches() -> None:
    fake_retriever = FakeRetriever()
    failure = RuntimeError("provider unavailable")
    tracing = FakeTracing()

    def agent_factory(
        _model: str,
        _search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        return FakeAgent(lambda _payload: (_ for _ in ()).throw(failure))

    with pytest.raises(AgentRetrievalError) as raised:
        _sdk(fake_retriever, agent_factory, tracing=tracing).retrieve("narrative")

    assert [search.kind for search in raised.value.searches] == ["original"]
    assert raised.value.__cause__ is failure
    assert tracing.flushes == 1


def test_result_reports_successful_trace_flush() -> None:
    tracing = FakeTracing()

    result = _sdk(
        FakeRetriever(),
        lambda _model, _search_tool, _snippet_tool: FakeAgent(
            lambda _payload: {"messages": [{"role": "assistant", "content": "Done."}]}
        ),
        tracing=tracing,
    ).retrieve("narrative")

    assert result.trace_flush_succeeded is True
    assert tracing.flushes == 1
    with pytest.raises((AttributeError, TypeError)):
        result.trace_flush_succeeded = False


def test_result_reports_disabled_tracing_as_successful_noop() -> None:
    tracing = create_retrieval_tracing(environ={})

    result = DeepAgentRetriever(
        retriever=FakeRetriever(),
        agent_factory=lambda _model, _search_tool, _snippet_tool: FakeAgent(
            lambda _payload: {"messages": [{"role": "assistant", "content": "Done."}]}
        ),
        tracing=tracing,
        model="test-model",
    ).retrieve("narrative")

    assert tracing.enabled is False
    assert result.trace_flush_succeeded is True


def test_failed_trace_flush_is_reported_without_retrying_or_changing_ranking() -> None:
    retriever = FakeRetriever(candidate_count=2)
    tracing = FailedFlushTracing()
    agent_calls = 0

    def invoke(_payload: dict[str, object]) -> object:
        nonlocal agent_calls
        agent_calls += 1
        return {"messages": [{"role": "assistant", "content": "Done."}]}

    result = _sdk(
        retriever,
        lambda _model, _search_tool, _snippet_tool: FakeAgent(invoke),
        tracing=tracing,
    ).retrieve("narrative")

    assert result.trace_flush_succeeded is False
    assert [candidate.docid for candidate in result.candidates] == [
        "original-doc-1",
        "original-doc-2",
    ]
    assert [candidate.rank for candidate in result.candidates] == [1, 2]
    assert len(retriever.queries) == 1
    assert agent_calls == 1
    assert tracing.flushes == 1


@pytest.mark.parametrize("rejected_phase", ["search", "snippet", "result"])
def test_trace_evidence_rejection_does_not_change_retrieval(
    rejected_phase: str,
) -> None:
    retriever = FakeRetriever(candidate_count=2)
    tracing = RejectedEvidenceTracing(rejected_phase)

    result = _sdk(
        retriever,
        lambda _model, _search_tool, snippet_tool: FakeAgent(
            lambda _payload: (
                snippet_tool("original-doc-1", "safe focus", None),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        ),
        tracing=tracing,
        snippet_extractor=RecordingSnippetExtractor(),
    ).retrieve("narrative")

    assert [candidate.docid for candidate in result.candidates] == [
        "original-doc-1",
        "original-doc-2",
    ]
    assert result.stopping_reason == "agent_completed"
    assert len(retriever.queries) == 1
    assert tracing.flushes == 1


def test_snippet_trace_validation_rejection_does_not_change_tool_result(
    isolated_real_tracing: None,
) -> None:
    oversized_document_id = "d" * (MAX_TRACE_DOCUMENT_ID_CHARACTERS + 1)
    snippet_payloads: list[dict[str, object]] = []

    class OversizedIdRetriever(FakeRetriever):
        def retrieve(self, query):
            self.queries.append(query)
            return [
                _candidate(
                    query=query.query_text,
                    variant=query.variant_name,
                    docid=oversized_document_id,
                    rank=1,
                    text="trace rejection must not hide this text",
                )
            ]

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    tracing = create_retrieval_tracing(environ={}, tracer_provider=provider)

    result = _sdk(
        OversizedIdRetriever(),
        lambda _model, _search_tool, snippet_tool: FakeAgent(
            lambda _payload: (
                snippet_payloads.append(
                    json.loads(snippet_tool(oversized_document_id, "safe focus", None))
                ),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        ),
        tracing=tracing,
        snippet_extractor=RecordingSnippetExtractor(),
    ).retrieve("narrative")

    assert snippet_payloads[0]["document_id"] == oversized_document_id
    assert result.candidates[0].docid == oversized_document_id


@pytest.mark.parametrize("failed_phase", ["enter", "exit"])
def test_snippet_span_lifecycle_failure_does_not_change_tool_result(
    failed_phase: str,
) -> None:
    snippet_payloads: list[dict[str, object]] = []

    result = _sdk(
        FakeRetriever(),
        lambda _model, _search_tool, snippet_tool: FakeAgent(
            lambda _payload: (
                snippet_payloads.append(
                    json.loads(snippet_tool("original-doc-1", "safe focus", None))
                ),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        ),
        tracing=FailedSnippetLifecycleTracing(failed_phase),
        snippet_extractor=RecordingSnippetExtractor(),
    ).retrieve("narrative")

    assert snippet_payloads[0]["document_id"] == "original-doc-1"
    assert result.candidates[0].docid == "original-doc-1"


def test_factory_passes_only_explicit_deepagents_070_arguments(monkeypatch) -> None:
    calls = []

    def fake_create_deep_agent(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr("deepagents.create_deep_agent", fake_create_deep_agent)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def search_tool(_query: str) -> str:
        return "{}"

    def snippet_tool(
        _document_id: str, _focus_query: str, _cursor: str | None = None
    ) -> str:
        return "{}"

    agent = _create_agent("openrouter:deepseek/test-model", search_tool, snippet_tool)

    assert agent is not None
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert set(kwargs) == {"model", "tools", "system_prompt", "middleware", "backend"}
    assert isinstance(kwargs["model"], ChatOpenRouter)
    assert kwargs["model"].model_name == "deepseek/test-model"
    assert kwargs["model"].max_retries == 0
    assert kwargs["tools"] == [search_tool, snippet_tool]
    assert kwargs["system_prompt"] == ANY
    assert len(kwargs["middleware"]) == 1
    assert isinstance(
        kwargs["middleware"][0], deepagent_retrieval._RetrievalOnlyMiddleware
    )
    assert isinstance(kwargs["backend"], StateBackend)


@pytest.mark.parametrize(
    "model",
    ["test:model", "openrouter:", "openrouter:   ", " openrouter:test/model"],
)
def test_factory_rejects_invalid_openrouter_specs_before_agent_construction(
    monkeypatch, model: str
) -> None:
    calls = []
    monkeypatch.setattr(
        "deepagents.create_deep_agent",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="openrouter:<model-id>"):
        _create_agent(
            model,
            lambda _query: "{}",
            lambda _document_id, _focus_query, _cursor=None: "{}",
        )

    assert calls == []


def test_real_deepagents_factory_exposes_retrieval_and_safe_state_tools(
    monkeypatch,
) -> None:
    sdk_model = CaptureChatModel(
        responses=[AIMessage(content="Coverage is sufficient.")]
    )
    unrelated_model = CaptureChatModel(responses=[AIMessage(content="Unrelated done.")])
    resolved_model = [sdk_model]

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "deepagents.graph.resolve_model", lambda _spec: resolved_model[0]
    )
    result = DeepAgentRetriever(
        retriever=FakeRetriever(),
        model="openrouter:test/deepagent-retrieval-capture",
        tracing=FakeTracing(),
    ).retrieve("narrative")

    assert result.rationale == "Coverage is sufficient."
    assert set(sdk_model.captured_tool_names) == {
        "search_climbmix",
        "extract_relevant_snippets",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
    }
    assert "task" not in sdk_model.captured_tool_names
    assert "execute" not in sdk_model.captured_tool_names
    assert sdk_model.captured_bind_settings
    assert all(
        settings["parallel_tool_calls"] is False
        for settings in sdk_model.captured_bind_settings
    )

    def unrelated_tool(query: str) -> str:
        """Return a test-only unrelated result."""
        return query

    resolved_model[0] = unrelated_model
    unrelated_agent = create_deep_agent(
        model="openrouter:test/deepagent-retrieval-capture",
        tools=[unrelated_tool],
    )
    unrelated_agent.invoke({"messages": [{"role": "user", "content": "unrelated"}]})

    assert {"ls", "task", "unrelated_tool"} <= set(unrelated_model.captured_tool_names)


def test_retrieval_only_middleware_allows_retrieval_and_state_tools_but_denies_others() -> (
    None
):
    middleware = deepagent_retrieval._RetrievalOnlyMiddleware()
    model = CaptureChatModel(responses=[AIMessage(content="unused")])
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[
            {"name": name}
            for name in (
                "search_climbmix",
                "extract_relevant_snippets",
                "ls",
                "read_file",
                "write_file",
                "edit_file",
                "delete",
                "glob",
                "grep",
                "execute",
                "task",
                "unrelated_tool",
            )
        ],
        model_settings={"temperature": 0.25, "parallel_tool_calls": True},
    )
    seen_sync = []

    def sync_handler(filtered: ModelRequest[object]) -> AIMessage:
        seen_sync.extend(
            tool["name"] for tool in filtered.tools if isinstance(tool, dict)
        )
        filtered.model.bind_tools(filtered.tools, **(filtered.model_settings or {}))
        return AIMessage(content="ok")

    assert middleware.wrap_model_call(request, sync_handler).content == "ok"
    assert seen_sync == [
        "search_climbmix",
        "extract_relevant_snippets",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
    ]
    assert model.captured_bind_settings == [
        {"temperature": 0.25, "parallel_tool_calls": False}
    ]

    async def verify_async() -> None:
        seen_async = []

        async def async_handler(filtered: ModelRequest[object]) -> AIMessage:
            seen_async.extend(
                tool["name"] for tool in filtered.tools if isinstance(tool, dict)
            )
            filtered.model.bind_tools(filtered.tools, **(filtered.model_settings or {}))
            return AIMessage(content="ok")

        reply = await middleware.awrap_model_call(request, async_handler)
        assert reply.content == "ok"
        assert seen_async == seen_sync
        assert model.captured_bind_settings == [
            {"temperature": 0.25, "parallel_tool_calls": False},
            {"temperature": 0.25, "parallel_tool_calls": False},
        ]

        for allowed_name in ("write_file", "read_file"):
            allowed = ToolCallRequest(
                tool_call={"name": allowed_name, "args": {}, "id": "call-allowed"},
                tool=None,
                state={},
                runtime=None,
            )
            assert middleware.wrap_tool_call(allowed, lambda _request: "ok") == "ok"

            async def allowed_handler(_request: ToolCallRequest) -> str:
                return "ok"

            assert await middleware.awrap_tool_call(allowed, allowed_handler) == "ok"

        for forbidden_name in ("execute", "task", "unrelated_tool"):
            forbidden = ToolCallRequest(
                tool_call={"name": forbidden_name, "args": {}, "id": "call-forbidden"},
                tool=None,
                state={},
                runtime=None,
            )
            with pytest.raises(
                PermissionError, match="retrieval-only tool access denied"
            ):
                middleware.wrap_tool_call(
                    forbidden,
                    lambda _request: pytest.fail("forbidden tool handler was called"),
                )
            with pytest.raises(
                PermissionError, match="retrieval-only tool access denied"
            ):
                await middleware.awrap_tool_call(
                    forbidden,
                    lambda _request: pytest.fail("forbidden tool handler was called"),
                )

    asyncio.run(verify_async())


def test_tool_returns_bounded_metadata_but_result_retains_all_candidates() -> None:
    fake_retriever = FakeRetriever(candidate_count=11)
    tool_payloads = []

    def agent_factory(
        _model: str,
        search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            tool_payloads.append(json.loads(search_tool("targeted query")))
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")

    assert len(tool_payloads[0]["candidates"]) == 10
    assert all(
        set(candidate) == {"docid", "rank", "score", "text_length"}
        for candidate in tool_payloads[0]["candidates"]
    )
    assert len(result.searches[1].candidates) == 11
    assert tool_payloads[0]["remaining_budget"] == 2


def test_nondefault_retrieval_bounds_control_model_budget_trace_and_fusion() -> None:
    fake_retriever = FakeRetriever(candidate_count=4)
    tracing = FakeTracing()
    initial_payloads: list[dict[str, object]] = []
    tool_payloads: list[dict[str, object]] = []

    def agent_factory(
        _model: str,
        search_tool: Callable[[str], str],
        _snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        def invoke(payload: dict[str, object]) -> object:
            initial_payloads.append(payload)
            tool_payloads.append(json.loads(search_tool("targeted query")))
            tool_payloads.append(json.loads(search_tool("over budget query")))
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = DeepAgentRetriever(
        retriever=fake_retriever,
        agent_factory=agent_factory,
        tracing=tracing,
        model="test-model",
        hits_per_search=2,
        max_followup_searches=1,
        fused_result_limit=1,
    ).retrieve("narrative")

    assert len(fake_retriever.queries) == 2
    assert "original-doc-3" not in json.dumps(initial_payloads[0], sort_keys=True)
    assert len(tool_payloads[0]["candidates"]) == 2
    assert tool_payloads[0]["remaining_budget"] == 0
    assert tool_payloads[1] == {"error": "search budget exhausted"}
    assert result.stopping_reason == "search_budget_exhausted"
    assert len(result.candidates) == 1
    assert len(result.searches[0].candidates) == 4
    assert len(result.searches[1].candidates) == 4
    assert tracing.search_records[0]["document_ids"] == (
        "original-doc-1",
        "original-doc-2",
    )
    assert tracing.result_records[0]["fused_document_ids"] == (
        "followup-52d6e9c630f589d8-doc-1",
    )


def test_large_fused_override_keeps_root_trace_ids_at_safe_limit() -> None:
    requested_limit = MAX_TRACE_DOCUMENTS + 1
    tracing = FakeTracing()

    result = DeepAgentRetriever(
        retriever=FakeRetriever(candidate_count=requested_limit),
        agent_factory=lambda _model, _search_tool, _snippet_tool: FakeAgent(
            lambda _payload: {"messages": [{"role": "assistant", "content": "Done."}]}
        ),
        tracing=tracing,
        model="test-model",
        hits_per_search=requested_limit,
        fused_result_limit=requested_limit,
    ).retrieve("narrative")

    assert len(result.candidates) == requested_limit
    assert len(tracing.result_records[0]["fused_document_ids"]) == MAX_TRACE_DOCUMENTS


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2", None])
@pytest.mark.parametrize(
    "parameter",
    ["hits_per_search", "max_followup_searches", "fused_result_limit"],
)
def test_constructor_rejects_invalid_retrieval_bounds_before_use(
    parameter: str, value: object
) -> None:
    with pytest.raises(ValueError, match=f"{parameter} must be a positive integer"):
        DeepAgentRetriever(
            retriever=FakeRetriever(),
            tracing=FakeTracing(),
            **{parameter: value},
        )


def test_from_env_threads_nondefault_retrieval_bounds_into_remote_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    created = []

    class OfflineRemoteRetriever:
        def __init__(self, config: object, *, cache_dir: object) -> None:
            self.config = config
            self.cache_dir = cache_dir
            created.append(self)

        def retrieve(self, _query: object) -> list[RetrievedCandidate]:
            return []

    monkeypatch.setattr(
        "trec_rag.deepagent_retrieval.PyseriniRemoteRetriever", OfflineRemoteRetriever
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    sdk = DeepAgentRetriever.from_env(
        root=tmp_path,
        tracing=FakeTracing(),
        hits_per_search=7,
        max_followup_searches=2,
        fused_result_limit=5,
    )

    assert created[0].config.hits == 7
    assert sdk._hits_per_search == 7
    assert sdk._max_followup_searches == 2
    assert sdk._fused_result_limit == 5


def test_from_env_rejects_invalid_bound_before_remote_retriever_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    remote_constructions = []
    monkeypatch.setattr(
        "trec_rag.deepagent_retrieval.PyseriniRemoteRetriever",
        lambda *_args, **_kwargs: remote_constructions.append(True),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(ValueError, match="hits_per_search must be a positive integer"):
        DeepAgentRetriever.from_env(
            root=tmp_path,
            tracing=FakeTracing(),
            hits_per_search=True,
        )

    assert remote_constructions == []


def test_retrieve_uses_only_the_purpose_specific_tracing_api() -> None:
    fake_retriever = FakeRetriever()
    tracing = FakeTracing()
    result = _sdk(
        fake_retriever,
        lambda _model, _search_tool, snippet_tool: FakeAgent(
            lambda _payload: (
                snippet_tool("original-doc-1", "safe focus", None),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        ),
        tracing=tracing,
        snippet_extractor=RecordingSnippetExtractor(),
    ).retrieve("narrative")

    assert result.stopping_reason == "agent_completed"
    assert len(tracing.search_records) == 1
    assert set(tracing.search_records[0]) == {
        "document_ids",
        "source_ranks",
        "source_scores",
        "cache_status",
        "latency_ms",
        "text_lengths",
    }
    assert tracing.search_records[0]["text_lengths"] == (19,)
    assert tracing.snippet_records == [
        {
            "chunk_ids": ("chunk-1",),
            "start_chars": (0,),
            "end_chars": (19,),
            "relevance_scores": (0.75,),
            "texts": ("narrative excerpt 1",),
            "cache_status": "miss",
            "ranker_backend": "private-ranker",
            "latency_ms": ANY,
            "page_offset": 0,
            "has_next_page": False,
        }
    ]
    assert tracing.result_records == [
        {
            "fused_document_ids": ("original-doc-1",),
            "stopping_reason": "agent_completed",
        }
    ]
    assert tracing.flushes == 1


@pytest.mark.parametrize("trace_content", [True, False])
def test_retrieve_exports_complete_bounded_safe_trace_payload(
    monkeypatch: pytest.MonkeyPatch, trace_content: bool, isolated_real_tracing: None
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = create_retrieval_tracing(
        environ={}, tracer_provider=provider, trace_content=trace_content
    )
    clock = iter((10.0, 10.125, 20.0, 20.0075))
    monkeypatch.setattr(
        deepagent_retrieval, "monotonic", lambda: next(clock), raising=False
    )

    result = _sdk(
        FakeRetriever(candidate_count=2),
        lambda _model, _search_tool, snippet_tool: FakeAgent(
            lambda _payload: (
                snippet_tool("original-doc-1", "private focus query", None),
                {
                    "messages": [
                        {"role": "assistant", "content": "Coverage is sufficient."}
                    ]
                },
            )[1]
        ),
        tracing=tracing,
        snippet_extractor=RecordingSnippetExtractor(),
    ).retrieve("private narrative")

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "climbmix.retrieve",
        "deepagent.extract_relevant_snippets",
        "deepagent.retrieve",
    ]
    retriever_span, snippet_span, root_span = spans
    assert retriever_span.parent is not None
    assert retriever_span.parent.span_id == root_span.context.span_id
    assert set(retriever_span.attributes) == {
        "input.value",
        "openinference.span.kind",
        "retrieval.cache_status",
        "retrieval.document_count",
        "retrieval.document_text_lengths",
        "retrieval.document_ids",
        "retrieval.latency_ms",
        "retrieval.source_ranks",
        "retrieval.source_scores",
    }
    assert retriever_span.attributes["input.value"] == (
        "private narrative" if trace_content else REDACTED_CONTENT
    )
    assert retriever_span.attributes["retrieval.document_ids"] == (
        "original-doc-1",
        "original-doc-2",
    )
    assert retriever_span.attributes["retrieval.source_ranks"] == (1, 2)
    assert retriever_span.attributes["retrieval.source_scores"] == (99.0, 98.0)
    assert retriever_span.attributes["retrieval.cache_status"] == "not_reported"
    assert retriever_span.attributes["retrieval.latency_ms"] == 125.0
    assert retriever_span.attributes["retrieval.document_count"] == 2
    assert retriever_span.attributes["retrieval.document_text_lengths"] == (27, 27)
    assert snippet_span.parent is not None
    assert snippet_span.parent.span_id == root_span.context.span_id
    assert snippet_span.attributes["input.value"] == (
        "private focus query" if trace_content else REDACTED_CONTENT
    )
    assert snippet_span.attributes["snippet.document_id"] == "original-doc-1"
    assert snippet_span.attributes["snippet.chunk_ids"] == ("chunk-1",)
    assert snippet_span.attributes["snippet.start_chars"] == (0,)
    assert snippet_span.attributes["snippet.end_chars"] == (27,)
    assert snippet_span.attributes["snippet.relevance_scores"] == (0.75,)
    assert snippet_span.attributes["snippet.texts"] == (
        ("private narrative excerpt 1",) if trace_content else (REDACTED_CONTENT,)
    )
    assert snippet_span.attributes["snippet.cache_status"] == "miss"
    assert snippet_span.attributes["snippet.ranker_backend"] == "private-ranker"
    assert snippet_span.attributes["snippet.latency_ms"] == pytest.approx(7.5)
    assert snippet_span.attributes["snippet.page_offset"] == 0
    assert snippet_span.attributes["snippet.has_next_page"] is False
    assert set(root_span.attributes) == {
        "input.value",
        "openinference.span.kind",
        "retrieval.fused_document_ids",
        "retrieval.stopping_reason",
    }
    assert root_span.attributes["input.value"] == (
        "private narrative" if trace_content else REDACTED_CONTENT
    )
    assert root_span.attributes["retrieval.fused_document_ids"] == (
        "original-doc-1",
        "original-doc-2",
    )
    assert root_span.attributes["retrieval.stopping_reason"] == "agent_completed"
    assert [candidate.docid for candidate in result.candidates] == [
        "original-doc-1",
        "original-doc-2",
    ]


def test_from_env_uses_offline_retriever_config_cache_and_model_precedence(
    monkeypatch, tmp_path
) -> None:
    created = []
    snippet_roots = []

    class OfflineRemoteRetriever:
        def __init__(self, config: object, *, cache_dir: object) -> None:
            self.config = config
            self.cache_dir = cache_dir
            created.append(self)

        def retrieve(self, _query: object) -> list[RetrievedCandidate]:
            return []

    monkeypatch.setattr(
        "trec_rag.deepagent_retrieval.PyseriniRemoteRetriever", OfflineRemoteRetriever
    )
    monkeypatch.setattr(
        "trec_rag.deepagent_retrieval.create_default_snippet_extractor",
        lambda root: snippet_roots.append(root) or RecordingSnippetExtractor(),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("DEEPAGENT_MODEL", "openrouter:environment-model")
    tracing = FakeTracing()

    configured = DeepAgentRetriever.from_env(
        root=tmp_path,
        model="openrouter:constructor-model",
        tracing=tracing,
    )
    from_environment = DeepAgentRetriever.from_env(root=tmp_path, tracing=tracing)
    monkeypatch.delenv("DEEPAGENT_MODEL")
    defaulted = DeepAgentRetriever.from_env(root=tmp_path, tracing=tracing)

    assert [sdk._model for sdk in (configured, from_environment, defaulted)] == [
        "openrouter:constructor-model",
        "openrouter:environment-model",
        "openrouter:deepseek/deepseek-v4-flash",
    ]
    assert len(created) == 3
    assert snippet_roots == [tmp_path, tmp_path, tmp_path]
    assert created[0].config.name == "deepagent_climbmix"
    assert created[0].config.type == "pyserini_remote"
    assert created[0].config.query_variants == ("original", "followup")
    assert created[0].config.hits == 10
    assert created[0].config.index == "climbmix-400b"
    assert created[0].config.cache is True
    assert created[0].cache_dir == tmp_path / "cache" / "retrieval" / "pyserini_remote"


def test_direct_construction_creates_default_extractor_only_when_snippet_tool_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_roots: list[object] = []
    extractor = RecordingSnippetExtractor()
    monkeypatch.setattr(
        "trec_rag.deepagent_retrieval.create_default_snippet_extractor",
        lambda root: created_roots.append(root) or extractor,
    )

    _sdk(
        FakeRetriever(),
        lambda _model, _search_tool, _snippet_tool: FakeAgent(
            lambda _payload: {"messages": [{"role": "assistant", "content": "Done."}]}
        ),
    ).retrieve("narrative")
    assert created_roots == []

    snippet_payloads: list[dict[str, object]] = []

    def agent_factory(
        _model: str,
        _search_tool: Callable[[str], str],
        snippet_tool: Callable[[str, str, str | None], str],
    ) -> FakeAgent:
        return FakeAgent(
            lambda _payload: (
                snippet_payloads.append(
                    json.loads(snippet_tool("original-doc-1", "focus", None))
                ),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        )

    _sdk(FakeRetriever(), agent_factory).retrieve("narrative")

    assert len(created_roots) == 1
    assert snippet_payloads[0]["document_id"] == "original-doc-1"
