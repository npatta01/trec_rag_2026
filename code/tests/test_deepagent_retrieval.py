from __future__ import annotations

from collections.abc import Mapping
import asyncio
import hashlib
import inspect
import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Barrier, Event, Lock
from typing import Callable, Sequence
from unittest.mock import ANY
from uuid import uuid4

import pytest
import trec_rag.deepagent_retrieval as deepagent_retrieval
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openrouter import ChatOpenRouter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor
from pydantic import Field

import trec_rag.deepagent_tracing as deepagent_tracing
from trec_rag.deepagent_budget import (
    ResearchBudget,
    ResearchBudgetConfig,
    ResearchTaskContext,
)
from trec_rag.deepagent_research import (
    MainToolFilterMiddleware,
    ResearchTaskBudgetMiddleware,
    ResearchTaskEnvelope,
    bind_research_task,
)
from trec_rag.deepagent_evidence import SnippetHandle
from trec_rag.document_store import DocumentStore
from trec_rag.deepagent_retrieval import (
    AgentRetrievalError,
    AgentSearch,
    DeepAgentRetriever,
    _create_agent,
    reciprocal_rank_fuse,
)
from trec_rag.deepagent_snippets import (
    InvalidSnippetCursorError,
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
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate, jsonable
from trec_rag.topic_passage_search import (
    FocusedQuery,
    OrganizerRequestFailed,
    PassageSearchResult,
    SourceDocument,
    SourcePassage,
)
from trec_rag.topic_records import TopicRecordsBuilder


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


class GatedFollowupRetriever(FakeRetriever):
    """Hold named follow-ups inside transport until a test releases them."""

    def __init__(self, *queries: str) -> None:
        super().__init__()
        self.entered = {query: Event() for query in queries}
        self.releases = {query: Event() for query in queries}

    def retrieve(self, query):
        if query.query_text in self.entered:
            self.entered[query.query_text].set()
            if not self.releases[query.query_text].wait(timeout=2):
                raise AssertionError(f"test did not release {query.query_text}")
        return super().retrieve(query)


class GatedDuplicateRetriever(FakeRetriever):
    """Expose an unintended second transport call for one duplicate query."""

    def __init__(self) -> None:
        super().__init__()
        self.first_entered = Event()
        self.second_entered = Event()
        self.release_first = Event()
        self._followup_calls = 0
        self._calls_lock = Lock()

    def retrieve(self, query):
        if query.variant_name != "original":
            with self._calls_lock:
                self._followup_calls += 1
                call_number = self._followup_calls
            if call_number == 1:
                self.first_entered.set()
                if not self.release_first.wait(timeout=2):
                    raise AssertionError("test did not release first duplicate call")
            else:
                self.second_entered.set()
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


def _seed_test_need(
    toolset: deepagent_retrieval.AgentToolset,
    *,
    narrative_span: str = "narrative",
) -> None:
    toolset.update_retrieval_state(
        {
            "add_needs": [
                {
                    "need_id": "test-need",
                    "narrative_span": narrative_span,
                    "question": "What evidence closes the test need?",
                }
            ]
        }
    )


_TEST_CONTEXT_LOCK = Lock()


def _test_research_context(
    toolset: deepagent_retrieval.AgentToolset,
) -> tuple[ResearchTaskContext, ResearchTaskEnvelope]:
    with _TEST_CONTEXT_LOCK:
        existing = getattr(toolset, "_test_research_context", None)
        if existing is not None:
            return existing
        task_id = f"test-research-{uuid4().hex}"
        context = ResearchTaskContext(task_id, 1, "focused", ("test-need",))
        envelope = ResearchTaskEnvelope(
            research_task_id=task_id,
            round_index=1,
            depth="focused",
            motivating_ids=["test-need"],
            goal="Close the test need.",
        )
        assert toolset.budget.reserve_task(context).ok
        existing = (context, envelope)
        object.__setattr__(toolset, "_test_research_context", existing)
        return existing


def _authorized_search(
    toolset: deepagent_retrieval.AgentToolset,
    query: str,
    *,
    narrative_span: str = "narrative",
) -> str:
    _seed_test_need(toolset, narrative_span=narrative_span)
    _, envelope = _test_research_context(toolset)
    with bind_research_task(envelope):
        return toolset.search_climbmix(
            query, ["test-need"], "test need remains open"
        )


def _authorized_snippet(
    toolset: deepagent_retrieval.AgentToolset,
    document_id: str,
    focus_query: str,
    cursor: str | None = None,
    *,
    action: str = "extract",
    narrative_span: str = "narrative",
) -> str:
    _seed_test_need(toolset, narrative_span=narrative_span)
    _, envelope = _test_research_context(toolset)
    with bind_research_task(envelope):
        return toolset.extract_relevant_snippets(
            document_id,
            focus_query,
            ["test-need"],
            "test need remains open",
            cursor,
        )


def _authorized_passage(
    toolset: deepagent_retrieval.AgentToolset,
    query: str,
    *,
    motivating_ids: list[str] | None = None,
) -> str:
    _seed_test_need(toolset)
    _, envelope = _test_research_context(toolset)
    with bind_research_task(envelope):
        return toolset.search_passages(
            query,
            motivating_ids or ["test-need"],
            "test need remains open",
        )


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
            raise InvalidSnippetCursorError("cursor leaked /private/cache/path")
        if focus_query == "ranker value failure":
            raise ValueError("ranker leaked /private/model/path")
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
                page_index=0,
                residual_count=0,
                residual_top_score=None,
                returned_min_score=0.75,
                pages_estimated=1,
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


class FailedTraceLifecycleTracing(FakeTracing):
    def __init__(self, span_name: str, failed_phase: str) -> None:
        super().__init__()
        self.span_name = span_name
        self.failed_phase = failed_phase

    @contextmanager
    def _manager(self, span_name: str):
        if self.span_name == span_name and self.failed_phase == "enter":
            raise RuntimeError(f"{span_name} trace span enter failed")
        try:
            yield self
        except BaseException:
            if self.span_name == span_name and self.failed_phase == "exit":
                raise RuntimeError(f"{span_name} trace span exit failed")
            raise
        if self.span_name == span_name and self.failed_phase == "exit":
            raise RuntimeError(f"{span_name} trace span exit failed")

    def agent_span(self, _narrative: str):
        if self.span_name == "agent" and self.failed_phase == "create":
            raise RuntimeError("agent trace span creation failed")
        return self._manager("agent")

    def retriever_span(self, _query: str):
        if self.span_name == "retriever" and self.failed_phase == "create":
            raise RuntimeError("retriever trace span creation failed")
        return self._manager("retriever")


class RaisingFlushTracing(FakeTracing):
    def force_flush(self) -> bool:
        self.flushes += 1
        raise RuntimeError("trace flush failed")


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


class ToolAwareFakeMessagesListChatModel(FakeMessagesListChatModel):
    """Scripted model with the bind_tools hook required by create_agent."""

    def bind_tools(self, tools: Sequence[object], **kwargs: object):
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


class _RetrieverBackedPassageSearch:
    """Test-only bridge from historical fakes to the shared passage contract."""

    def __init__(self, retriever: FakeRetriever, topic_id: str) -> None:
        self._retriever = retriever
        self.topic_id = topic_id

    def search(self, query: FocusedQuery) -> PassageSearchResult:
        request = QueryVariant(
            self.topic_id,
            query.query_id,
            query.text,
            "topic_passage_search",
        )
        try:
            candidates = tuple(self._retriever.retrieve(request))
        except RuntimeError as exc:
            if str(exc) in {
                "controlled retrieval failure",
                "explicit continuation required",
            }:
                raise OrganizerRequestFailed("test retriever unavailable") from exc
            raise

        by_docid: dict[str, RetrievedCandidate] = {}
        for candidate in candidates:
            previous = by_docid.get(candidate.docid)
            if previous is not None and previous.text != candidate.text:
                raise ValueError(
                    f"conflicting text for document_id {candidate.docid}"
                )
            if previous is None or candidate.rank < previous.rank:
                by_docid[candidate.docid] = candidate
        retained = tuple(sorted(by_docid.values(), key=lambda row: (row.rank, row.docid)))
        if not retained:
            return PassageSearchResult(
                query,
                "incomplete",
                "no_evidence",
                1000,
                0,
                0,
                0,
                (),
                (),
                1,
                True,
            )

        passage_rows = []
        document_rows = []
        for candidate in retained:
            text = candidate.text or f"No text returned for {candidate.docid}."
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            passage_id = "p-" + hashlib.sha256(
                f"{candidate.docid}\0{text}".encode("utf-8")
            ).hexdigest()
            passage_rows.append(
                SourcePassage(
                    passage_id,
                    candidate.docid,
                    digest,
                    candidate.rank,
                    candidate.score,
                    0,
                    len(text),
                    0,
                    len(text.encode("utf-8")),
                    digest,
                    text,
                    candidate.score,
                    1,
                    "cache-" + hashlib.sha256(
                        f"{query.text}\0{text}".encode("utf-8")
                    ).hexdigest(),
                    digest,
                    {"backend": "retriever-backed-test-chunker"},
                )
            )
            document_rows.append(
                SourceDocument(
                    candidate.docid,
                    digest,
                    candidate.rank,
                    candidate.score,
                    passage_id,
                    candidate.score,
                )
            )
        passage_rows.sort(
            key=lambda row: (-row.raw_logit, row.source_rank, row.passage_id)
        )
        ranked_passages = tuple(
            SourcePassage(
                row.passage_id,
                row.docid,
                row.content_sha256,
                row.source_rank,
                row.source_score,
                row.start_char,
                row.end_char,
                row.start_byte,
                row.end_byte,
                row.text_sha256,
                row.text,
                row.raw_logit,
                rank,
                row.score_cache_key,
                row.scoring_text_sha256,
                row.chunker_identity,
            )
            for rank, row in enumerate(passage_rows[:100], start=1)
        )
        return PassageSearchResult(
            query,
            "complete",
            None,
            1000,
            len(document_rows),
            len(document_rows),
            len(passage_rows),
            tuple(document_rows),
            ranked_passages,
            1,
            len(document_rows) < 1000,
        )


class _HarnessTopicLedger:
    def __init__(self, topic_id: str) -> None:
        self.topic_id = topic_id
        self.run_id = "test-run"
        self._facets = {}
        self.searches = []
        self.handoffs = []
        self.completion: tuple[str, str] | None = None

    def add_facets(self, facets) -> None:
        self._facets.update((facet.subnarrative_id, facet) for facet in facets)

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


class _TopicBoundSdkHarness:
    def __init__(self, retriever: FakeRetriever, **sdk_kwargs: object) -> None:
        self.retriever = retriever
        self.sdk_kwargs = sdk_kwargs

    def retrieve(self, narrative: str):
        topic_id = hashlib.sha256(narrative.encode("utf-8")).hexdigest()[:16]
        sdk = DeepAgentRetriever(
            passage_search=_RetrieverBackedPassageSearch(self.retriever, topic_id),
            **self.sdk_kwargs,
        )
        return sdk.retrieve(_HarnessTopicLedger(topic_id), narrative)


def _topic_bound_sdk(retriever: FakeRetriever, **kwargs: object) -> _TopicBoundSdkHarness:
    return _TopicBoundSdkHarness(retriever, **kwargs)


def _sdk(
    retriever: FakeRetriever,
    agent_factory: Callable[
        [str, Callable[[str], str], Callable[[str, str, str | None], str]],
        FakeAgent,
    ],
    *,
    tracing: FakeTracing | None = None,
    snippet_extractor: object | None = None,
) -> _TopicBoundSdkHarness:
    return _topic_bound_sdk(
        retriever,
        agent_factory=agent_factory,
        tracing=tracing or FakeTracing(),
        model="test-model",
        snippet_extractor=snippet_extractor,
    )


def test_search_records_actual_query_motivation_and_task_context_in_one_call() -> None:
    fake_retriever = FakeRetriever()

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            toolset.update_retrieval_state(
                {
                    "add_needs": [
                        {
                            "need_id": "N1",
                            "narrative_span": "narrative",
                            "question": "What evidence closes N1?",
                        }
                    ]
                }
            )
            envelope = ResearchTaskEnvelope(
                research_task_id="R1-N1",
                round_index=1,
                depth="focused",
                motivating_ids=["N1"],
                goal="Close N1.",
            )
            context = ResearchTaskContext("R1-N1", 1, "focused", ("N1",))
            assert toolset.budget.reserve_task(context).ok
            try:
                with bind_research_task(envelope):
                    payload = json.loads(
                        toolset.search_climbmix(
                            "actual refined query",
                            ["N1"],
                            "N1 has no grounded driver evidence",
                        )
                    )
            finally:
                toolset.budget.finish_task(context)
            assert "error" not in payload
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")

    action = result.coverage_report.actions[-1]
    assert action.target == "actual refined query"
    assert action.motivating_ids == ("N1",)
    assert action.research_task_id == "R1-N1"
    assert result.budget_snapshot.remaining_retrieval_calls == 99


def test_invalid_motivation_is_rejected_before_retrieval() -> None:
    fake_retriever = FakeRetriever()
    observed: dict[str, object] = {}

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            toolset.update_retrieval_state(
                {
                    "add_needs": [
                        {
                            "need_id": "N1",
                            "narrative_span": "narrative",
                            "question": "What evidence closes N1?",
                        }
                    ]
                }
            )
            envelope = ResearchTaskEnvelope(
                research_task_id="R1-N1",
                round_index=1,
                depth="survey",
                motivating_ids=["N1"],
                goal="Close N1.",
            )
            context = ResearchTaskContext("R1-N1", 1, "survey", ("N1",))
            assert toolset.budget.reserve_task(context).ok
            try:
                with bind_research_task(envelope):
                    observed.update(
                        json.loads(
                            toolset.search_climbmix(
                                "query", ["UNKNOWN"], "find missing evidence"
                            )
                        )
                    )
            finally:
                toolset.budget.finish_task(context)
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    _sdk(fake_retriever, agent_factory).retrieve("narrative")

    assert observed["code"] == "UNKNOWN_MOTIVATION"
    assert [query.query_text for query in fake_retriever.queries] == ["narrative"]


def test_coordinator_prompt_states_the_enforced_dispatch_limits() -> None:
    prompt = deepagent_retrieval._coordinator_prompt(
        ResearchBudgetConfig(max_concurrent=2, max_researcher_invocations=7)
    )

    assert "At most 2 researchers run at once" in prompt
    assert "At most 7 researchers may run in the whole invocation" in prompt.replace(
        "\n", " "
    )
    assert "more than 2 task calls in one batch" in prompt.replace("\n", " ")


def test_coordinator_prompt_reads_correctly_for_a_single_slot() -> None:
    prompt = deepagent_retrieval._coordinator_prompt(
        ResearchBudgetConfig(max_concurrent=1)
    )

    assert "Exactly one researcher runs at a time" in prompt
    assert "single task call per turn" in prompt.replace("\n", " ")
    assert "1 researchers" not in prompt, "the singular case must not be pluralised"


def test_coordinator_prompt_tracks_a_changed_concurrency_limit() -> None:
    relaxed = deepagent_retrieval._coordinator_prompt(
        ResearchBudgetConfig(max_concurrent=5)
    )

    assert "At most 5 researchers" in relaxed, (
        "the stated limit must come from the config that enforces it"
    )
    assert "Exactly one researcher" not in relaxed


def test_unreachable_index_is_not_reported_as_missing_evidence() -> None:
    """An outage must not read as "this narrative had no evidence"."""

    class BrokenRetriever(FakeRetriever):
        def retrieve(self, query: object) -> list[RetrievedCandidate]:
            if getattr(query, "variant_name", "").startswith("followup"):
                raise RuntimeError("explicit continuation required")
            return super().retrieve(query)

    observed: dict[str, object] = {}

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            _seed_need(toolset)
            envelope = ResearchTaskEnvelope(
                research_task_id="R1-N1",
                round_index=1,
                depth="survey",
                motivating_ids=["N1"],
                goal="Close N1.",
            )
            context = ResearchTaskContext("R1-N1", 1, "survey", ("N1",))
            assert toolset.budget.reserve_task(context).ok
            try:
                with bind_research_task(envelope):
                    observed.update(
                        json.loads(
                            toolset.search_climbmix("a query", ["N1"], "find evidence")
                        )
                    )
            finally:
                toolset.budget.finish_task(context)
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(BrokenRetriever(), agent_factory).retrieve("narrative")

    assert observed["code"] == "RETRIEVAL_UNAVAILABLE"
    assert observed["must_stop"] is True
    assert result.stopping_reason == "retrieval_unavailable"
    assert result.budget_snapshot.stop_code is None, (
        "one unreachable search reports the outage without cancelling the run"
    )


def test_no_round_cap_exists_to_strand_researchers() -> None:
    config = ResearchBudgetConfig()

    assert not hasattr(config, "max_rounds"), (
        "rounds are bounded by researchers; a separate cap can only strand them"
    )
    assert config.max_main_models >= 3 * config.max_researcher_invocations, (
        "each round costs the coordinator a dispatch, a merge, and a close"
    )


def test_model_facing_snippets_show_handles_and_numbered_sentences() -> None:
    handles = (
        SnippetHandle(
            handle="S3",
            snippet_id="shard_00123_4567:0002",
            relevance_score=0.5,
            sentences=("First sentence.", "Second sentence."),
        ),
    )

    rows = deepagent_retrieval._model_facing_snippets(handles)

    assert rows == [
        {
            "cite": "S3",
            "relevance_score": 0.5,
            "sentences": [
                {"n": 1, "text": "First sentence."},
                {"n": 2, "text": "Second sentence."},
            ],
        }
    ]
    serialised = json.dumps(rows)
    assert "shard_00123_4567" not in serialised, (
        "an agent that never sees a raw identifier cannot retype one"
    )


def _seed_need(toolset: deepagent_retrieval.AgentToolset) -> None:
    toolset.update_retrieval_state(
        {
            "add_needs": [
                {
                    "need_id": "N1",
                    "narrative_span": "narrative",
                    "question": "What evidence closes N1?",
                }
            ]
        }
    )


def test_unknown_citation_rejection_is_reported_as_unadmitted() -> None:
    observed: dict[str, object] = {}

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            _seed_need(toolset)
            observed.update(
                json.loads(
                    toolset.update_retrieval_state(
                        {
                            "add_nuggets": [
                                {
                                    "nugget_id": "NUG-1",
                                    "text": "A claim whose citation was invented.",
                                    "need_ids": ["N1"],
                                    "facet_ids": [],
                                    "evidence": [{"cite": "S9"}],
                                }
                            ]
                        }
                    )
                )
            )
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(FakeRetriever(), agent_factory).retrieve("narrative")

    assert observed["accepted_ids"] == []
    assert observed["rejected_summary"] == {"UNKNOWN_CITATION": 1}
    assert observed["unadmitted_sections"] == ["add_nuggets"]
    assert "NOT admitted" in str(observed["evidence_rejection_notice"])
    assert result.coverage_report.nuggets == ()
    assert result.coverage_report.needs[0].status == "unaddressed"


def test_accepted_delta_carries_no_rejection_notice() -> None:
    observed: dict[str, object] = {}

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            observed.update(
                json.loads(
                    toolset.update_retrieval_state(
                        {
                            "add_needs": [
                                {
                                    "need_id": "N1",
                                    "narrative_span": "narrative",
                                    "question": "What evidence closes N1?",
                                }
                            ]
                        }
                    )
                )
            )
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    _sdk(FakeRetriever(), agent_factory).retrieve("narrative")

    assert observed["accepted_ids"] == ["N1"]
    assert "rejected_summary" not in observed
    assert "evidence_rejection_notice" not in observed


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
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        factory_models.append(model)
        factory_queries.extend(fake_retriever.queries)
        return FakeAgent(
            lambda _payload: (
                _authorized_search(toolset, "targeted follow-up"),
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


def test_agent_sees_candidate_metadata_without_shared_document_extraction() -> (
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
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        def invoke(payload: dict[str, object]) -> object:
            content = str(payload["messages"][0]["content"])
            initial_candidates.extend(
                json.loads(content.split("Bounded original results:\n", 1)[1])
            )
            snippet_payloads.append(
                json.loads(
                    _authorized_snippet(
                        toolset, "original-doc-1", "original focus", None
                    )
                )
            )
            followup = json.loads(_authorized_search(toolset, "targeted query"))
            followup_candidates.extend(followup["documents"])
            followup_docid = str(followup["documents"][0]["docid"])
            snippet_payloads.append(
                json.loads(
                    _authorized_snippet(toolset, followup_docid, "followup focus", None)
                )
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
    assert [payload["code"] for payload in snippet_payloads] == [
        "UNKNOWN_DOCUMENT",
        "UNKNOWN_DOCUMENT",
    ]
    assert extractor.calls == []
    assert all("cache_status" not in payload for payload in snippet_payloads)
    assert all("ranker_backend" not in payload for payload in snippet_payloads)
    assert all(sentinel in search.candidates[0].text for search in result.searches)
    assert all(sentinel in candidate.text for candidate in result.candidates)


def test_legacy_snippet_tool_returns_safe_errors_without_shared_document_text() -> (
    None
):
    extractor = RecordingSnippetExtractor()
    tool_payloads: list[dict[str, object]] = []

    def agent_factory(
        _model: str,
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            tool_payloads.extend(
                (
                    json.loads(
                        _authorized_snippet(toolset, "unknown-doc", "focus", None)
                    ),
                    json.loads(
                        _authorized_snippet(toolset, "original-doc-1", "   ", None)
                    ),
                )
            )
            tool_payloads.append(
                json.loads(_authorized_snippet(toolset, "original-doc-1", "focus", None))
            )
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    _sdk(FakeRetriever(), agent_factory, snippet_extractor=extractor).retrieve(
        "narrative"
    )

    assert [payload["error"] for payload in tool_payloads] == [
        "unknown document_id",
        "focus_query must be non-empty text",
        "unknown document_id",
    ]
    assert extractor.calls == []
    assert all("budget_snapshot" in payload for payload in tool_payloads)
    assert "/private/" not in json.dumps(tool_payloads)


def test_retrieve_does_not_treat_retained_passages_as_complete_documents() -> None:
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
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        return FakeAgent(
            lambda _payload: (
                _authorized_search(toolset, "targeted query"),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        )

    result = _sdk(ConflictingRetriever(), agent_factory).retrieve("narrative")
    assert [candidate.docid for candidate in result.candidates] == ["shared-doc"]


def test_retrieve_rejects_empty_narrative_before_external_calls() -> None:
    fake_retriever = FakeRetriever()
    factory_calls = []

    with pytest.raises(ValueError, match="narrative must be non-empty text"):
        _sdk(
            fake_retriever,
            lambda _model, _toolset: factory_calls.append(True),  # type: ignore[return-value]
        ).retrieve("   ")

    assert fake_retriever.queries == []
    assert factory_calls == []


def test_retrieve_default_budget_admits_four_followups() -> (
    None
):
    fake_retriever = FakeRetriever()
    tool_results = []

    def agent_factory(
        _model: str,
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            tool_results.extend(
                _authorized_search(toolset, f"query {number}") for number in range(4)
            )
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
        "query 3",
    ]
    assert [search.kind for search in result.searches] == [
        "original",
        "followup",
        "followup",
        "followup",
        "followup",
    ]
    assert all(json.loads(item)["ok"] for item in tool_results)
    assert result.stopping_reason == "agent_completed"


def test_retrieve_serializes_concurrent_followups_at_three_successes() -> None:
    queries = tuple(f"concurrent query {number}" for number in range(3))
    fake_retriever = GatedFollowupRetriever(*queries)
    tool_results = []

    def agent_factory(
        _model: str,
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            _seed_test_need(toolset)
            started = [Event() for _query in range(4)]

            def authorize(query: str) -> None:
                assert query

            def call_tool(index: int, query: str) -> str:
                started[index].set()
                return _authorized_search(toolset, query)

            with ThreadPoolExecutor(max_workers=4) as pool:
                authorize(queries[0])
                futures = [pool.submit(call_tool, 0, queries[0])]
                assert fake_retriever.entered[queries[0]].wait(timeout=2)

                authorize(queries[1])
                futures.append(pool.submit(call_tool, 1, queries[1]))
                assert started[1].wait(timeout=2)
                assert not fake_retriever.entered[queries[1]].wait(timeout=0.1)
                fake_retriever.releases[queries[0]].set()
                assert fake_retriever.entered[queries[1]].wait(timeout=2)

                authorize(queries[2])
                futures.append(pool.submit(call_tool, 2, queries[2]))
                assert started[2].wait(timeout=2)
                assert not fake_retriever.entered[queries[2]].wait(timeout=0.1)
                fake_retriever.releases[queries[1]].set()
                assert fake_retriever.entered[queries[2]].wait(timeout=2)

                over_budget_query = "concurrent query 3"
                authorize(over_budget_query)
                futures.append(pool.submit(call_tool, 3, over_budget_query))
                assert started[3].wait(timeout=2)
                fake_retriever.releases[queries[2]].set()
                tool_results.extend(future.result(timeout=2) for future in futures)
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")
    payloads = [json.loads(item) for item in tool_results]

    assert len(fake_retriever.queries) == 5
    assert len(result.searches) == 5
    assert [search.kind for search in result.searches] == [
        "original",
        "followup",
        "followup",
        "followup",
        "followup",
    ]
    assert all(payload["ok"] for payload in payloads)
    assert result.stopping_reason == "agent_completed"


def test_retrieve_rejects_concurrent_duplicate_without_using_budget() -> None:
    fake_retriever = GatedDuplicateRetriever()
    tool_results = []

    def agent_factory(
        _model: str,
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            _seed_test_need(toolset)
            query = "same concurrent query"

            def authorize() -> None:
                return None

            second_started = Event()
            with ThreadPoolExecutor(max_workers=2) as pool:
                authorize()
                first = pool.submit(_authorized_search, toolset, query)
                assert fake_retriever.first_entered.wait(timeout=2)

                authorize()

                def call_second() -> str:
                    second_started.set()
                    return _authorized_search(toolset, query)

                second = pool.submit(call_second)
                assert second_started.wait(timeout=2)
                assert not fake_retriever.second_entered.wait(timeout=0.1)
                fake_retriever.release_first.set()
                tool_results.extend((first.result(timeout=2), second.result(timeout=2)))
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


def test_retrieve_concurrent_blank_followups_are_budgeted_as_no_yield() -> None:
    fake_retriever = FakeRetriever()
    blank_results = []
    successful_results = []

    def agent_factory(
        _model: str,
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            barrier = Barrier(4)

            def call_blank(number: int) -> str:
                barrier.wait(timeout=2)
                return _authorized_search(toolset, " " * (number + 1))

            with ThreadPoolExecutor(max_workers=4) as pool:
                blank_results.extend(pool.map(call_blank, range(4)))
            successful_results.extend(
                _authorized_search(toolset, f"query {number}") for number in range(3)
            )
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")

    assert [query.query_text for query in fake_retriever.queries] == ["narrative"]
    assert {
        json.loads(item)["error"] for item in blank_results
    } <= {"query must be non-empty text", "retrieval budget refused"}
    assert all(
        json.loads(item)["code"] == "NO_YIELD_STOP"
        for item in successful_results
    )
    assert all(json.loads(item)["must_stop"] is True for item in successful_results)
    assert result.budget_snapshot.stop_code is None
    assert result.stopping_reason == "agent_completed"


def test_retrieve_failure_releases_lock_and_preserves_serialized_success_order() -> (
    None
):
    fake_retriever = ControlledFailureRetriever()
    tool_results = []

    def agent_factory(
        _model: str,
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            _seed_test_need(toolset)

            def authorize(query: str) -> None:
                assert query

            authorize("failing query")
            queued_started = Event()
            with ThreadPoolExecutor(max_workers=2) as pool:
                failed = pool.submit(_authorized_search, toolset, "failing query")
                assert fake_retriever.failure_started.wait(timeout=2)

                authorize("query 0")

                def call_queued_success() -> str:
                    queued_started.set()
                    return _authorized_search(toolset, "query 0")

                successful = pool.submit(call_queued_success)
                assert queued_started.wait(timeout=2)
                assert fake_retriever.attempted_queries == [
                    "narrative",
                    "failing query",
                ]
                fake_retriever.release_failure.set()
                # A transport failure is reported to the agent rather than
                # raised, but it must still release the lock in order.
                assert (
                    json.loads(failed.result(timeout=2))["code"]
                    == "RETRIEVAL_UNAVAILABLE"
                )
                tool_results.append(successful.result(timeout=2))

            tool_results.extend(
                _authorized_search(toolset, f"query {number}") for number in range(1, 3)
            )
            tool_results.append(_authorized_search(toolset, "query beyond budget"))
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")
    recorded_queries = [query.query_text for query in fake_retriever.queries]

    assert fake_retriever.attempted_queries[0:2] == ["narrative", "failing query"]
    assert "failing query" not in recorded_queries
    assert len(recorded_queries) == 5
    assert [search.query for search in result.searches] == recorded_queries
    assert [
        item["query"] for item in result.candidates[0].provenance
    ] == recorded_queries
    assert all(json.loads(item)["ok"] for item in tool_results)
    last_payload = json.loads(tool_results[-1])
    assert last_payload["must_stop"] is True
    assert last_payload["budget_snapshot"]["stop_code"] is None
    assert result.stopping_reason == "retrieval_unavailable", (
        "a run that could not reach the index says so, even though it continued"
    )


def test_retrieve_rejects_blank_and_original_duplicate_followups_with_budget_charge() -> (
    None
):
    fake_retriever = FakeRetriever()
    tool_results = []

    def agent_factory(
        _model: str,
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            tool_results.extend(
                [
                    _authorized_search(toolset, "   "),
                    _authorized_search(toolset, "narrative"),
                    _authorized_search(toolset, "query 0"),
                    _authorized_search(toolset, "query 1"),
                    _authorized_search(toolset, "query 2"),
                    _authorized_search(toolset, "query 3"),
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
        "query 3",
    ]
    assert json.loads(tool_results[0])["error"] == "query must be non-empty text"
    assert json.loads(tool_results[1])["error"] == "duplicate follow-up query"
    assert json.loads(tool_results[-1])["ok"] is True
    assert result.stopping_reason == "agent_completed"


def test_retrieve_uses_stable_hashes_for_private_cache_variants() -> None:
    fake_retriever = FakeRetriever()

    def agent_factory(
        _model: str,
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        return FakeAgent(
            lambda _payload: (
                _authorized_search(
                    toolset, "specific missing aspect", narrative_span="Narrative"
                ),
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


def test_retrieve_maps_provider_failure_to_partial_result() -> None:
    fake_retriever = FakeRetriever()
    failure = RuntimeError("provider unavailable")
    tracing = FakeTracing()

    def agent_factory(
        _model: str,
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        return FakeAgent(lambda _payload: (_ for _ in ()).throw(failure))

    result = _sdk(fake_retriever, agent_factory, tracing=tracing).retrieve("narrative")

    assert [search.kind for search in result.searches] == ["original"]
    assert result.stopping_reason == "retrieval_unavailable"
    assert result.topic_snapshot.status == "incomplete"
    assert result.topic_snapshot.stopping_reason == "retrieval_unavailable"
    assert tracing.flushes == 1


def test_result_reports_successful_trace_flush() -> None:
    tracing = FakeTracing()

    result = _sdk(
        FakeRetriever(),
        lambda _model, _toolset: FakeAgent(
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

    result = _topic_bound_sdk(
        FakeRetriever(),
        agent_factory=lambda _model, _toolset: FakeAgent(
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
        lambda _model, _toolset: FakeAgent(invoke),
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


def test_raised_trace_flush_is_reported_without_changing_result() -> None:
    retriever = FakeRetriever(candidate_count=2)
    tracing = RaisingFlushTracing()

    result = _sdk(
        retriever,
        lambda _model, _toolset: FakeAgent(
            lambda _payload: {
                "messages": [{"role": "assistant", "content": "Stable rationale."}]
            }
        ),
        tracing=tracing,
    ).retrieve("narrative")

    assert result.trace_flush_succeeded is False
    assert [candidate.docid for candidate in result.candidates] == [
        "original-doc-1",
        "original-doc-2",
    ]
    assert result.rationale == "Stable rationale."
    assert tracing.flushes == 1


@pytest.mark.parametrize("span_name", ["agent", "retriever"])
@pytest.mark.parametrize("failed_phase", ["create", "enter", "exit"])
def test_trace_span_lifecycle_failure_does_not_change_completed_result(
    span_name: str, failed_phase: str
) -> None:
    snippet_payloads: list[dict[str, object]] = []
    tracing = FailedTraceLifecycleTracing(span_name, failed_phase)

    result = _sdk(
        FakeRetriever(candidate_count=2),
        lambda _model, toolset: FakeAgent(
            lambda _payload: (
                snippet_payloads.append(
                    json.loads(
                        _authorized_snippet(toolset, "original-doc-1", "safe focus", None)
                    )
                ),
                {"messages": [{"role": "assistant", "content": "Stable rationale."}]},
            )[1]
        ),
        tracing=tracing,
        snippet_extractor=RecordingSnippetExtractor(),
    ).retrieve("narrative")

    assert snippet_payloads[0]["code"] == "UNKNOWN_DOCUMENT"
    assert [search.query for search in result.searches] == ["narrative"]
    assert [candidate.docid for candidate in result.candidates] == [
        "original-doc-1",
        "original-doc-2",
    ]
    assert result.rationale == "Stable rationale."
    assert result.stopping_reason == "agent_completed"
    assert result.trace_flush_succeeded is True
    assert tracing.flushes == 1


@pytest.mark.parametrize("span_name", ["agent", "retriever"])
def test_trace_span_exit_failure_preserves_retrieval_error(
    span_name: str,
) -> None:
    class ActualFailureRetriever(FakeRetriever):
        def retrieve(self, _query):
            raise RuntimeError("actual retrieval failed")

    with pytest.raises(RuntimeError, match="^actual retrieval failed$"):
        _sdk(
            ActualFailureRetriever(),
            lambda _model, _toolset: FakeAgent(
                lambda _payload: {
                    "messages": [{"role": "assistant", "content": "unused"}]
                }
            ),
            tracing=FailedTraceLifecycleTracing(span_name, "exit"),
        ).retrieve("narrative")


@pytest.mark.parametrize("rejected_phase", ["search", "snippet", "result"])
def test_trace_evidence_rejection_does_not_change_retrieval(
    rejected_phase: str,
) -> None:
    retriever = FakeRetriever(candidate_count=2)
    tracing = RejectedEvidenceTracing(rejected_phase)

    result = _sdk(
        retriever,
        lambda _model, toolset: FakeAgent(
            lambda _payload: (
                _authorized_snippet(toolset, "original-doc-1", "safe focus", None),
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


def test_legacy_snippet_trace_rejection_returns_unknown_document(
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
        lambda _model, toolset: FakeAgent(
            lambda _payload: (
                snippet_payloads.append(
                    json.loads(
                        _authorized_snippet(toolset, oversized_document_id, "safe focus", None)
                    )
                ),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        ),
        tracing=tracing,
        snippet_extractor=RecordingSnippetExtractor(),
    ).retrieve("narrative")

    assert snippet_payloads[0]["code"] == "UNKNOWN_DOCUMENT"
    assert result.candidates[0].docid == oversized_document_id


@pytest.mark.parametrize("failed_phase", ["enter", "exit"])
def test_snippet_span_lifecycle_failure_does_not_change_tool_result(
    failed_phase: str,
) -> None:
    snippet_payloads: list[dict[str, object]] = []

    result = _sdk(
        FakeRetriever(),
        lambda _model, toolset: FakeAgent(
            lambda _payload: (
                snippet_payloads.append(
                    json.loads(
                        _authorized_snippet(toolset, "original-doc-1", "safe focus", None)
                    )
                ),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        ),
        tracing=FailedSnippetLifecycleTracing(failed_phase),
        snippet_extractor=RecordingSnippetExtractor(),
    ).retrieve("narrative")

    assert snippet_payloads[0]["code"] == "UNKNOWN_DOCUMENT"
    assert result.candidates[0].docid == "original-doc-1"


def test_factory_passes_only_explicit_deepagents_070_arguments(monkeypatch) -> None:
    calls = []
    profiles = []

    def fake_create_deep_agent(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr("deepagents.create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(
        "deepagents.register_harness_profile",
        lambda key, profile: profiles.append((key, profile)),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def search_tool(
        _query: str, _motivating_ids: list[str], _rationale: str
    ) -> str:
        return "{}"

    def snippet_tool(
        _document_id: str,
        _focus_query: str,
        _motivating_ids: list[str],
        _rationale: str,
        _cursor: str | None = None,
    ) -> str:
        return "{}"

    def view_tool(_scope: str = "frontier") -> str:
        return "{}"

    def update_tool(_delta: dict[str, object]) -> str:
        return "{}"

    def complete_tool() -> str:
        return "{}"

    def passages_tool(_query: str, _ids: list[str], _rationale: str) -> str:
        return "{}"

    def complete_round_tool(_round_index: int) -> str:
        return "{}"

    def action_tool(
        _action: str,
        _target: str,
        _focus_query: str | None,
        _motivating_ids: list[str],
        _rationale: str,
    ) -> str:
        return "{}"

    budget_config = ResearchBudgetConfig()
    budget = ResearchBudget(budget_config)
    toolset = deepagent_retrieval.AgentToolset(
        search_climbmix=search_tool,
        search_passages=passages_tool,
        extract_relevant_snippets=snippet_tool,
        view_retrieval_state=view_tool,
        update_retrieval_state=update_tool,
        complete_retrieval=complete_tool,
        complete_research_round=complete_round_tool,
        choose_next_action=action_tool,
        budget=budget,
        budget_config=budget_config,
    )
    agent = _create_agent("openrouter:deepseek/test-model", toolset)

    assert agent is not None
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert set(kwargs) == {
        "model",
        "tools",
        "system_prompt",
        "middleware",
        "subagents",
        "backend",
    }
    assert isinstance(kwargs["model"], ChatOpenRouter)
    assert kwargs["model"].model_name == "deepseek/test-model"
    assert kwargs["model"].max_retries == 0
    assert kwargs["model"].request_timeout == 300_000
    sdk_config = kwargs["model"].client.sdk_configuration
    assert sdk_config.timeout_ms == 300_000
    assert sdk_config.retry_config.strategy == "none"
    assert sdk_config.retry_config.retry_connection_errors is False
    assert kwargs["tools"] == [
        view_tool,
        update_tool,
        complete_round_tool,
        complete_tool,
    ]
    assert kwargs["system_prompt"] == ANY
    assert len(kwargs["subagents"]) == 1
    assert kwargs["subagents"][0]["name"] == "researcher"
    assert kwargs["subagents"][0]["model"] is kwargs["model"]
    assert kwargs["subagents"][0]["tools"] == [
        passages_tool,
        view_tool,
    ]
    assert len(kwargs["middleware"]) == 4
    assert isinstance(kwargs["middleware"][2], ResearchTaskBudgetMiddleware)
    assert isinstance(kwargs["middleware"][3], MainToolFilterMiddleware)
    assert profiles[0][0] == "openrouter:deepseek/test-model"
    assert profiles[0][1].general_purpose_subagent.enabled is False
    assert isinstance(
        kwargs["middleware"][0],
        deepagent_retrieval.ModelCallLimitMiddleware,
    )
    assert isinstance(kwargs["backend"], StateBackend)


def test_main_tool_filter_reserves_the_last_model_turn_for_finalization() -> None:
    config = ResearchBudgetConfig(max_main_models=4)
    middleware = MainToolFilterMiddleware(ResearchBudget(config), config)
    model = CaptureChatModel(responses=[AIMessage(content="unused")])
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[{"name": "task"}, {"name": "view_retrieval_state"}],
        state={"messages": [], "run_model_call_count": 3},
    )
    observed: dict[str, object] = {}

    def handler(filtered: ModelRequest[object]) -> AIMessage:
        observed["tools"] = filtered.tools
        observed["system"] = (
            filtered.system_message.content if filtered.system_message else ""
        )
        return AIMessage(content="partial")

    assert middleware.wrap_model_call(request, handler).content == "partial"
    assert observed["tools"] == []
    assert "grounded partial result immediately" in str(observed["system"])


def test_main_tool_filter_keeps_tools_after_task_local_no_yield_stop() -> None:
    config = ResearchBudgetConfig(no_yield_calls=1)
    budget = ResearchBudget(config)
    context = ResearchTaskContext("R1-N1", 1, "survey", ("N1",))
    assert budget.reserve_task(context).ok
    assert budget.reserve_retrieval(context, "search_climbmix").ok
    budget.record_yield(context, ())
    middleware = MainToolFilterMiddleware(budget, config)
    model = CaptureChatModel(responses=[AIMessage(content="unused")])
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[{"name": "task"}, {"name": "view_retrieval_state"}],
        state={"messages": [], "run_model_call_count": 1},
    )
    observed: dict[str, object] = {}

    def handler(filtered: ModelRequest[object]) -> AIMessage:
        observed["tools"] = filtered.tools
        return AIMessage(content="partial")

    middleware.wrap_model_call(request, handler)

    assert budget.snapshot().stop_code is None
    assert observed["tools"] == [
        {"name": "task"},
        {"name": "view_retrieval_state"},
    ]


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
            deepagent_retrieval.AgentToolset(
                search_climbmix=lambda _query, _ids, _rationale: "{}",
                search_passages=lambda _query, _ids, _rationale: "{}",
                extract_relevant_snippets=lambda _document_id,
                _focus_query,
                _ids,
                _rationale,
                _cursor=None: "{}",
                view_retrieval_state=lambda _scope="frontier": "{}",
                update_retrieval_state=lambda _delta: "{}",
                complete_research_round=lambda _round_index: "{}",
                choose_next_action=lambda _action,
                _target,
                _focus_query,
                _motivating_ids,
                _rationale: "{}",
                budget=ResearchBudget(ResearchBudgetConfig()),
                budget_config=ResearchBudgetConfig(),
            ),
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
    result = _topic_bound_sdk(
        FakeRetriever(),
        model="openrouter:test/deepagent-retrieval-capture",
        tracing=FakeTracing(),
    ).retrieve("narrative")

    assert result.rationale == "Coverage is sufficient."
    assert set(sdk_model.captured_tool_names) == {
        "task",
        "view_retrieval_state",
        "update_retrieval_state",
        "complete_retrieval",
        "complete_research_round",
        "read_file",
    }
    assert "execute" not in sdk_model.captured_tool_names
    assert sdk_model.captured_bind_settings
    assert all(
        settings["parallel_tool_calls"] is True
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

    assert {"ls", "unrelated_tool"} <= set(unrelated_model.captured_tool_names)


def test_offline_coordinator_delegates_to_researcher_and_continues_after_bundle(
    monkeypatch,
) -> None:
    task_description = json.dumps(
        {
            "research_task_id": "R1-N1",
            "round_index": 1,
            "depth": "focused",
            "motivating_ids": ["N1"],
            "goal": "Find one source for N1.",
            "known_evidence": "",
            "remaining_gap": "No grounded source yet.",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    bundle_snapshot = ResearchBudget(ResearchBudgetConfig()).snapshot().as_dict()
    model = CaptureChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "update_retrieval_state",
                        "args": {
                            "delta": {
                                "add_needs": [
                                    {
                                        "need_id": "N1",
                                        "narrative_span": "narrative",
                                        "question": "What evidence answers the narrative?",
                                    }
                                ]
                            }
                        },
                        "id": "seed-needs",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": task_description,
                            "subagent_type": "researcher",
                        },
                        "id": "delegate-research",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_passages",
                        "args": {
                            "query": "focused researcher query",
                            "motivating_ids": ["N1"],
                            "rationale": "N1 has no grounded source",
                        },
                        "id": "research-search",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "EvidenceBundle",
                        "args": {
                            "research_task_id": "R1-N1",
                            "round_index": 1,
                            "depth": "focused",
                            "motivating_need_ids": ["N1"],
                            "candidate_nuggets": [
                                {
                                    "claim": "The retrieved source addresses N1.",
                                    "need_ids": ["N1"],
                                    "facet_ids": [],
                                    "evidence": [{"cite": "S1"}],
                                    "contradicts_claims": [],
                                }
                            ],
                            "conflicts": [],
                            "unresolved_gaps": [],
                            "suggested_followups": [],
                            "stopping_reason": "goal_satisfied",
                            "budget_snapshot": bundle_snapshot,
                        },
                        "id": "structured-bundle",
                    }
                ],
            ),
            AIMessage(content="Final grounded response after researcher bundle."),
        ]
    )
    constructor_calls: list[dict[str, object]] = []

    def fake_openrouter(**kwargs: object) -> CaptureChatModel:
        constructor_calls.append(dict(kwargs))
        return model

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("langchain_openrouter.ChatOpenRouter", fake_openrouter)
    fake_retriever = FakeRetriever()

    result = _topic_bound_sdk(
        fake_retriever,
        model="openrouter:test/offline-coordinator",
        tracing=FakeTracing(),
    ).retrieve("narrative")

    assert len(constructor_calls) == 1
    constructor = constructor_calls[0]
    assert constructor["model"] == "test/offline-coordinator"
    assert constructor["max_retries"] == 0
    assert constructor["timeout"] == 300_000
    sdk_config = constructor["client"].sdk_configuration
    assert sdk_config.timeout_ms == 300_000
    assert sdk_config.retry_config.strategy == "none"
    assert sdk_config.retry_config.retry_connection_errors is False
    assert [query.query_text for query in fake_retriever.queries] == [
        "narrative",
        "focused researcher query",
    ]
    search_action = result.coverage_report.actions[-1]
    assert search_action.research_task_id == "R1-N1"
    assert search_action.target == "focused researcher query"
    assert result.coverage_report.inspected_page_count == 2
    assert result.budget_snapshot.completed_researchers == 1
    assert result.rationale == "Final grounded response after researcher bundle."


def test_nested_state_update_tool_exposes_model_facing_delta_sections() -> None:
    captured: list[deepagent_retrieval.AgentToolset] = []

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        captured.append(toolset)
        return FakeAgent(
            lambda _payload: {
                "messages": [{"role": "assistant", "content": "Done."}]
            }
        )

    _sdk(FakeRetriever(), agent_factory).retrieve("narrative")

    schema = convert_to_openai_tool(captured[0].update_retrieval_state)
    delta = schema["function"]["parameters"]["properties"]["delta"]

    assert {
        "add_needs",
        "add_facets",
        "add_nuggets",
        "add_evidence",
        "set_facet_status",
        "set_need_status",
        "supersede_nuggets",
        "abandon_documents",
    } <= set(delta["properties"])
    assert {"need_id", "narrative_span", "question"} <= set(
        delta["properties"]["add_needs"]["items"]["properties"]
    )
    assert {"need_id", "narrative_span", "question"} <= set(
        delta["properties"]["add_needs"]["items"]["required"]
    )


def _structured_state_update_tool(
    narrative: str = "narrative",
) -> StructuredTool:
    captured: list[deepagent_retrieval.AgentToolset] = []

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        captured.append(toolset)
        return FakeAgent(
            lambda _payload: {
                "messages": [{"role": "assistant", "content": "Done."}]
            }
        )

    _sdk(FakeRetriever(), agent_factory).retrieve(narrative)
    return StructuredTool.from_function(captured[0].update_retrieval_state)


def test_structured_state_tool_description_teaches_exact_delta_shape() -> None:
    description = _structured_state_update_tool().description

    assert (
        '{"delta":{"add_needs":[{"need_id":"N1",'
        '"narrative_span":"<exact text copied from the untouched narrative>",'
        '"question":"<question derived from that span>"}]}}'
    ) in description
    for section in (
        "add_needs",
        "add_facets",
        "add_nuggets",
        "add_evidence",
        "set_facet_status",
        "set_need_status",
        "supersede_nuggets",
        "abandon_documents",
    ):
        assert section in description
    assert 'Do not use "needs"' in description
    assert 'IDs such as "N1"' in description


def test_structured_state_tool_preserves_unknown_section_rejection() -> None:
    tool = _structured_state_update_tool()

    result = json.loads(tool.invoke({"delta": {"make_up_a_section": []}}))

    assert result["accepted_ids"] == []
    assert result["rejected"] == [
        {"section": "make_up_a_section", "index": 0, "code": "UNKNOWN_SECTION"}
    ]
    assert result["valid_delta_sections"] == [
        "add_needs",
        "add_facets",
        "add_nuggets",
        "add_evidence",
        "set_facet_status",
        "set_need_status",
        "supersede_nuggets",
        "abandon_documents",
    ]
    assert result["minimal_add_needs_example"] == {
        "delta": {
            "add_needs": [
                {
                    "need_id": "N1",
                    "narrative_span": "<exact text copied from the untouched narrative>",
                    "question": "<question derived from that span>",
                }
            ]
        }
    }


def test_structured_state_tool_accepts_valid_rows_while_rejecting_unknown_section() -> (
    None
):
    tool = _structured_state_update_tool("Explain migration drivers.")

    result = json.loads(
        tool.invoke(
            {
                "delta": {
                    "add_needs": [
                        {
                            "need_id": "n1",
                            "narrative_span": "migration drivers",
                            "question": "What are the drivers?",
                        }
                    ],
                    "make_up_a_section": [],
                }
            }
        )
    )

    assert result["accepted_ids"] == ["n1"]
    assert result["rejected"] == [
        {"section": "make_up_a_section", "index": 0, "code": "UNKNOWN_SECTION"}
    ]


@pytest.mark.parametrize("delta", [{}, {"add_needs": []}])
def test_structured_state_tool_rejects_empty_delta(delta: dict[str, object]) -> None:
    tool = _structured_state_update_tool()

    result = json.loads(tool.invoke({"delta": delta}))

    assert result["accepted_ids"] == []
    assert result["rejected"] == [
        {"section": "delta", "index": 0, "code": "EMPTY_DELTA"}
    ]
    assert result["valid_delta_sections"] == [
        "add_needs",
        "add_facets",
        "add_nuggets",
        "add_evidence",
        "set_facet_status",
        "set_need_status",
        "supersede_nuggets",
        "abandon_documents",
    ]
    assert result["minimal_add_needs_example"] == {
        "delta": {
            "add_needs": [
                {
                    "need_id": "N1",
                    "narrative_span": "<exact text copied from the untouched narrative>",
                    "question": "<question derived from that span>",
                }
            ]
        }
    }


def test_structured_state_tool_allows_nullable_status_fields_to_reach_state() -> (
    None
):
    tool = _structured_state_update_tool("Explain migration drivers.")
    seeded = json.loads(
        tool.invoke(
            {
                "delta": {
                    "add_needs": [
                        {
                            "need_id": "n1",
                            "narrative_span": "migration drivers",
                            "question": "What are the drivers?",
                        }
                    ],
                    "add_facets": [
                        {
                            "facet_id": "f1",
                            "need_ids": ["n1"],
                            "dimension": "driver",
                            "value": "conflict",
                            "origin": "narrative",
                        }
                    ],
                }
            }
        )
    )

    facet = json.loads(
        tool.invoke(
            {
                "delta": {
                    "set_facet_status": [
                        {
                            "facet_id": "f1",
                            "status": "open",
                            "status_reason": None,
                            "supporting_nugget_ids": [],
                        }
                    ]
                }
            }
        )
    )
    need = json.loads(
        tool.invoke(
            {
                "delta": {
                    "set_need_status": [
                        {
                            "need_id": "n1",
                            "status": "partial",
                            "remaining_gap": "Needs evidence.",
                            "draft_answer": None,
                            "draft_nugget_ids": [],
                        }
                    ]
                }
            }
        )
    )

    assert seeded["accepted_ids"] == ["n1", "f1"]
    assert facet["accepted_ids"] == ["f1"]
    assert facet["rejected"] == []
    assert need["accepted_ids"] == ["n1"]
    assert need["rejected"] == []


def test_choose_next_action_rejects_unhashable_action_kinds_as_safe_json() -> None:
    observed: list[dict[str, object]] = []

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            for malformed in ([], {}):
                observed.append(
                    json.loads(
                        toolset.choose_next_action(
                            malformed,  # type: ignore[arg-type]
                            "original-doc-1",
                            None,
                            [],
                            "malformed actions must be rejected safely",
                        )
                    )
                )
            return {"messages": [{"role": "assistant", "content": "Stopped."}]}

        return FakeAgent(invoke)

    _sdk(FakeRetriever(), agent_factory).retrieve("narrative")

    assert observed == [
        {"code": "INVALID_ACTION", "ok": False},
        {"code": "INVALID_ACTION", "ok": False},
    ]


def test_legacy_pagination_after_shared_search_rejects_unknown_document_before_extractor_work() -> None:
    extractor = RecordingSnippetExtractor()
    observed: dict[str, object] = {}

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            observed["page"] = json.loads(
                _authorized_snippet(
                    toolset,
                    "original-doc-1",
                    "unseen focus",
                    "opaque-page-two",
                    action="paginate",
                )
            )
            return {"messages": [{"role": "assistant", "content": "Stopped."}]}

        return FakeAgent(invoke)

    result = _sdk(FakeRetriever(), agent_factory, snippet_extractor=extractor).retrieve(
        "narrative"
    )

    assert observed["page"]["code"] == "UNKNOWN_DOCUMENT"
    assert observed["page"]["error"] == "unknown document_id"
    assert extractor.calls == []
    assert result.coverage_report.actions[-1].state == "consumed"
    assert result.coverage_report.inspected_page_count == 1


def test_coverage_report_preserves_five_seeded_topic_224_needs_and_open_gaps() -> None:
    narrative = (
        "I want to understand why people immigrate or become refugees, the challenges "
        "they face, and how laws and different groups shape immigration policies. "
        "Additionally, I'm interested in how various countries and religions view "
        "immigrants, and what options migrant workers have to improve their lives."
    )
    spans = (
        "why people immigrate or become refugees",
        "the challenges they face",
        "how laws and different groups shape immigration policies",
        "how various countries and religions view immigrants",
        "what options migrant workers have to improve their lives",
    )

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        return FakeAgent(
            lambda _payload: (
                toolset.update_retrieval_state(
                    {
                        "add_needs": [
                            {
                                "need_id": f"n{index}",
                                "narrative_span": span,
                                "question": f"Need {index}?",
                            }
                            for index, span in enumerate(spans, start=1)
                        ]
                    }
                ),
                {"messages": [{"role": "assistant", "content": "Stopped early."}]},
            )[1]
        )

    result = _sdk(FakeRetriever(), agent_factory).retrieve(narrative)

    assert [need.narrative_span for need in result.coverage_report.needs] == list(spans)
    assert result.coverage_report.unresolved_need_ids == (
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
    )
    assert result.coverage_report.terminal_reason is None
    assert result.stopping_reason == "agent_completed"


def test_recorded_terminal_stop_controls_result_stopping_reason() -> None:
    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        return FakeAgent(
            lambda _payload: (
                toolset.complete_retrieval(),
                {"messages": [{"role": "assistant", "content": "Complete."}]},
            )[1]
        )

    result = _sdk(
        FakeRetriever(), agent_factory, snippet_extractor=RecordingSnippetExtractor()
    ).retrieve("narrative")

    assert result.stopping_reason == "completion"
    assert result.coverage_report.terminal_reason == "completion"
    assert result.coverage_report.actions[-1].state == "terminal"


def test_complete_retrieval_rejects_live_evidence_without_a_draft_selection() -> None:
    observed: dict[str, object] = {}

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            _add_need_and_facet(toolset, need_id="test-need", facet_id="test-facet")
            passage = json.loads(_authorized_passage(toolset, "test need evidence"))
            toolset.update_retrieval_state(
                {
                    "add_nuggets": [
                        {
                            "nugget_id": "g1",
                            "text": "Grounded test evidence.",
                            "need_ids": ["test-need"],
                            "facet_ids": ["test-facet"],
                            "evidence": [{"cite": passage["passages"][0]["cite"]}],
                        }
                    ]
                }
            )
            observed["completion"] = json.loads(toolset.complete_retrieval())
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    result = _sdk(
        FakeRetriever(), agent_factory, snippet_extractor=RecordingSnippetExtractor()
    ).retrieve("narrative")

    assert observed["completion"]["ok"] is False
    assert observed["completion"]["code"] == "INCOMPLETE_CLOSEOUT"
    assert observed["completion"]["need_ids"] == ["test-need"]
    assert result.coverage_report.terminal_reason is None
    assert result.stopping_reason == "closeout_refused"


def test_retriever_wires_a_live_closeout_predicate_into_the_agent_toolset() -> None:
    observed: dict[str, object] = {}

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        observed["predicate"] = toolset.closeout_pending

        def invoke(_payload: dict[str, object]) -> object:
            _add_need_and_facet(toolset, need_id="test-need", facet_id="test-facet")
            passage = json.loads(_authorized_passage(toolset, "test need evidence"))
            toolset.update_retrieval_state(
                {
                    "add_nuggets": [
                        {
                            "nugget_id": "g1",
                            "text": "Grounded test evidence.",
                            "need_ids": ["test-need"],
                            "facet_ids": ["test-facet"],
                            "evidence": [{"cite": passage["passages"][0]["cite"]}],
                        }
                    ]
                }
            )
            assert toolset.closeout_pending is not None
            observed["pending_with_live_evidence"] = toolset.closeout_pending()
            toolset.update_retrieval_state(
                {
                    "set_need_status": [
                        {
                            "need_id": "test-need",
                            "status": "partial",
                            "remaining_gap": "More detail would help.",
                            "draft_nugget_ids": ["g1"],
                        }
                    ]
                }
            )
            observed["pending_after_selection"] = toolset.closeout_pending()
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    _sdk(
        FakeRetriever(),
        agent_factory,
        snippet_extractor=RecordingSnippetExtractor(),
    ).retrieve("narrative")

    assert callable(observed["predicate"])
    assert observed["pending_with_live_evidence"] is True
    assert observed["pending_after_selection"] is False


def test_complete_retrieval_records_the_only_successful_completion_transition() -> None:
    observed: dict[str, object] = {}

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            _add_need_and_facet(toolset, need_id="test-need", facet_id="test-facet")
            passage = json.loads(_authorized_passage(toolset, "test need evidence"))
            toolset.update_retrieval_state(
                {
                    "add_nuggets": [
                        {
                            "nugget_id": "g1",
                            "text": "Grounded test evidence.",
                            "need_ids": ["test-need"],
                            "facet_ids": ["test-facet"],
                            "evidence": [{"cite": passage["passages"][0]["cite"]}],
                        }
                    ],
                    "set_need_status": [
                        {
                            "need_id": "test-need",
                            "status": "answerable",
                            "remaining_gap": "",
                            "draft_answer": "Grounded test answer.",
                            "draft_nugget_ids": ["g1"],
                        }
                    ],
                }
            )
            observed["completion"] = json.loads(toolset.complete_retrieval())
            return {"messages": [{"role": "assistant", "content": "Complete."}]}

        return FakeAgent(invoke)

    result = _sdk(
        FakeRetriever(), agent_factory, snippet_extractor=RecordingSnippetExtractor()
    ).retrieve("narrative")

    assert observed["completion"]["ok"] is True
    assert observed["completion"]["state"] == "terminal"
    assert result.coverage_report.terminal_reason == "completion"
    assert result.stopping_reason == "completion"


def test_production_toolset_completes_through_a_real_langgraph_loop() -> None:
    model = ToolAwareFakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "complete_retrieval",
                        "args": {},
                        "id": "complete-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="retrieval complete"),
        ]
    )

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> object:
        assert toolset.complete_retrieval is not None
        return create_agent(
            model=model,
            tools=[toolset.complete_retrieval],
            middleware=[
                MainToolFilterMiddleware(
                    toolset.budget,
                    toolset.budget_config,
                    closeout_pending=toolset.closeout_pending,
                )
            ],
        )

    result = _sdk(FakeRetriever(), agent_factory).retrieve("narrative")

    assert result.coverage_report.terminal_reason == "completion"
    assert result.stopping_reason == "completion"


def test_production_toolset_rejects_live_incomplete_completion_and_continues() -> None:
    model = ToolAwareFakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "complete_retrieval",
                        "args": {},
                        "id": "complete-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "update_retrieval_state",
                        "args": {
                            "delta": {
                                "set_need_status": [
                                    {
                                        "need_id": "test-need",
                                        "status": "partial",
                                        "remaining_gap": "More detail would help.",
                                        "draft_nugget_ids": ["g1"],
                                    }
                                ]
                            }
                        },
                        "id": "update-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="I wrote the missing draft selection."),
        ]
    )

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> object:
        _add_need_and_facet(toolset, need_id="test-need", facet_id="test-facet")
        passage = json.loads(_authorized_passage(toolset, "test need evidence"))
        toolset.update_retrieval_state(
            {
                "add_nuggets": [
                    {
                        "nugget_id": "g1",
                        "text": "Grounded test evidence.",
                        "need_ids": ["test-need"],
                        "facet_ids": ["test-facet"],
                        "evidence": [{"cite": passage["passages"][0]["cite"]}],
                    }
                ]
            }
        )
        assert toolset.complete_retrieval is not None
        return create_agent(
            model=model,
            tools=[toolset.update_retrieval_state, toolset.complete_retrieval],
            middleware=[
                MainToolFilterMiddleware(
                    toolset.budget,
                    toolset.budget_config,
                    closeout_pending=toolset.closeout_pending,
                )
            ],
        )

    result = _sdk(
        FakeRetriever(),
        agent_factory,
        snippet_extractor=RecordingSnippetExtractor(),
    ).retrieve("narrative")

    assert result.coverage_report.terminal_reason is None
    assert result.stopping_reason == "closeout_refused"
    assert result.coverage_report.unresolved_need_ids == ("test-need",)


def test_choose_next_action_cannot_bypass_explicit_completion_tool() -> None:
    observed: dict[str, object] = {}

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            observed["result"] = json.loads(
                toolset.choose_next_action(
                    "stop", "completion", None, [], "try to bypass completion tool"
                )
            )
            return {"messages": [{"role": "assistant", "content": "Stopped."}]}

        return FakeAgent(invoke)

    result = _sdk(FakeRetriever(), agent_factory).retrieve("narrative")

    assert observed["result"] == {
        "code": "COMPLETE_RETRIEVAL_REQUIRED",
        "ok": False,
    }
    assert result.coverage_report.terminal_reason is None


def test_rejected_completion_is_not_reported_as_agent_completed() -> None:
    observed: dict[str, object] = {}

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            _seed_test_need(toolset)
            observed["completion"] = json.loads(toolset.complete_retrieval())
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    result = _sdk(FakeRetriever(), agent_factory).retrieve("narrative")

    assert observed["completion"]["ok"] is False
    assert observed["completion"]["code"] == "COMPLETION_OPEN_NEEDS"
    assert observed["completion"]["need_ids"] == ["test-need"]
    assert result.stopping_reason == "closeout_refused"


def test_coverage_terminal_reason_wins_over_budget_in_result_and_trace() -> None:
    tracing = FakeTracing()

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            _add_need_and_facet(toolset, need_id="test-need", facet_id="test-facet")
            passage = json.loads(_authorized_passage(toolset, "test need evidence"))
            toolset.update_retrieval_state(
                {
                    "add_nuggets": [
                        {
                            "nugget_id": "nugget-1",
                            "text": "Grounded test evidence.",
                            "need_ids": ["test-need"],
                            "facet_ids": ["test-facet"],
                            "evidence": [{"cite": passage["passages"][0]["cite"]}],
                        }
                    ],
                    "set_need_status": [
                        {
                            "need_id": "test-need",
                            "status": "answerable",
                            "remaining_gap": "",
                            "draft_answer": "Grounded test answer.",
                            "draft_nugget_ids": ["nugget-1"],
                        }
                    ],
                }
            )
            toolset.complete_retrieval()
            return {"messages": [{"role": "assistant", "content": "Complete."}]}

        return FakeAgent(invoke)

    result = _topic_bound_sdk(
        FakeRetriever(),
        agent_factory=agent_factory,
        tracing=tracing,
        model="test-model",
        snippet_extractor=RecordingSnippetExtractor(),
        budget_config=ResearchBudgetConfig(max_retrieval_calls=1),
    ).retrieve("narrative")

    assert result.stopping_reason == "completion"
    assert result.budget_snapshot.stop_code == "RETRIEVAL_BUDGET_EXHAUSTED"
    assert tracing.result_records[0]["stopping_reason"] == result.stopping_reason
    assert tracing.result_records[0]["budget_stop_code"] == (
        "RETRIEVAL_BUDGET_EXHAUSTED"
    )


def test_run_wide_budget_stop_maps_to_public_budget_exhausted_reason() -> None:
    tracing = FakeTracing()

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        return FakeAgent(
            lambda _payload: (
                _authorized_search(toolset, "one tiny-budget query"),
                {"messages": [{"role": "assistant", "content": "Partial."}]},
            )[1]
        )

    result = _topic_bound_sdk(
        FakeRetriever(),
        agent_factory=agent_factory,
        tracing=tracing,
        model="test-model",
        snippet_extractor=RecordingSnippetExtractor(),
        budget_config=ResearchBudgetConfig(max_retrieval_calls=1),
    ).retrieve("narrative")

    assert result.stopping_reason == "budget_exhausted"
    assert result.coverage_report.terminal_reason is None
    assert result.budget_snapshot.stop_code == "RETRIEVAL_BUDGET_EXHAUSTED"
    assert tracing.result_records[0]["stopping_reason"] == "budget_exhausted"
    assert (
        tracing.result_records[0]["budget_stop_code"]
        == "RETRIEVAL_BUDGET_EXHAUSTED"
    )


def test_legacy_extraction_after_shared_search_never_crosses_into_deadline_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.seconds = 0.0

        def __call__(self) -> float:
            return self.seconds

    clock = FakeClock()
    real_budget = ResearchBudget
    tracing = FakeTracing()
    tool_payload: dict[str, object] = {}

    class DeadlineCrossingExtractor(RecordingSnippetExtractor):
        def extract(
            self,
            document_id: str,
            document_text: str,
            focus_query: str,
            cursor: str | None = None,
        ) -> SnippetExtractionResult:
            result = super().extract(
                document_id, document_text, focus_query, cursor
            )
            clock.seconds = 2.0
            return result

    monkeypatch.setattr(
        deepagent_retrieval,
        "ResearchBudget",
        lambda config: real_budget(config, clock=clock),
    )

    def agent_factory(
        _model: str, toolset: deepagent_retrieval.AgentToolset
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            tool_payload.update(
                json.loads(
                    _authorized_snippet(
                        toolset, "original-doc-1", "deadline focus", None
                    )
                )
            )
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    result = _topic_bound_sdk(
        FakeRetriever(),
        agent_factory=agent_factory,
        tracing=tracing,
        model="test-model",
        snippet_extractor=DeadlineCrossingExtractor(),
        budget_config=ResearchBudgetConfig(soft_seconds=0, hard_seconds=1),
    ).retrieve("narrative")

    assert tool_payload["ok"] is False
    assert tool_payload["code"] == "UNKNOWN_DOCUMENT"
    assert tool_payload["must_stop"] is False
    assert "snippets" not in tool_payload
    assert result.coverage_report.inspected_page_count == 1
    assert [row.focus_query for row in result.coverage_report.documents] == [
        "narrative"
    ]
    assert tracing.snippet_records == []
    assert result.stopping_reason == "agent_completed"
    assert result.budget_snapshot.stop_code is None


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
                "view_retrieval_state",
                "update_retrieval_state",
                "choose_next_action",
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
        "view_retrieval_state",
        "update_retrieval_state",
        "ls",
        "read_file",
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

        for allowed_name in ("read_file",):
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

        for forbidden_name in (
            "execute",
            "task",
            "unrelated_tool",
            "write_file",
            "edit_file",
            "delete",
        ):
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
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            tool_payloads.append(
                json.loads(_authorized_search(toolset, "targeted query"))
            )
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")

    assert len(tool_payloads[0]["documents"]) == 10
    assert all(
        set(candidate) == {"docid", "rank", "score", "text_length"}
        for candidate in tool_payloads[0]["documents"]
    )
    assert len(result.searches[1].candidates) == 11
    assert tool_payloads[0]["remaining_budget"] == 99


def test_nondefault_retrieval_bounds_control_model_budget_trace_and_fusion() -> None:
    fake_retriever = FakeRetriever(candidate_count=4)
    tracing = FakeTracing()
    initial_payloads: list[dict[str, object]] = []
    tool_payloads: list[dict[str, object]] = []

    def agent_factory(
        _model: str,
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        def invoke(payload: dict[str, object]) -> object:
            initial_payloads.append(payload)
            tool_payloads.append(
                json.loads(_authorized_search(toolset, "targeted query"))
            )
            tool_payloads.append(
                json.loads(_authorized_search(toolset, "over budget query"))
            )
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _topic_bound_sdk(
        fake_retriever,
        agent_factory=agent_factory,
        tracing=tracing,
        model="test-model",
        hits_per_search=2,
        max_followup_searches=1,
        fused_result_limit=1,
    ).retrieve("narrative")

    assert len(fake_retriever.queries) == 2
    assert "original-doc-3" not in json.dumps(initial_payloads[0], sort_keys=True)
    assert len(tool_payloads[0]["documents"]) == 2
    assert tool_payloads[0]["remaining_budget"] == 99
    assert tool_payloads[1]["code"] == "TASK_TOOL_BUDGET_EXHAUSTED"
    assert result.stopping_reason == "agent_completed"
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
    requested_limit = MAX_TRACE_DOCUMENTS
    tracing = FakeTracing()

    result = _topic_bound_sdk(
        FakeRetriever(candidate_count=requested_limit),
        agent_factory=lambda _model, _toolset: FakeAgent(
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
            passage_search=_SharedPassageSearch(),
            tracing=FakeTracing(),
            **{parameter: value},
        )


def test_from_env_threads_nondefault_bounds_around_injected_shared_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    passage_search = _SharedPassageSearch()

    sdk = DeepAgentRetriever.from_env(
        root=tmp_path,
        passage_search=passage_search,
        tracing=FakeTracing(),
        snippet_extractor=RecordingSnippetExtractor(),
        hits_per_search=7,
        max_followup_searches=2,
        fused_result_limit=5,
    )

    assert sdk._passage_search is passage_search
    assert sdk._hits_per_search == 7
    assert sdk._max_followup_searches == 2
    assert sdk._fused_result_limit == 5


def test_from_env_rejects_invalid_bound_before_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(ValueError, match="hits_per_search must be a positive integer"):
        DeepAgentRetriever.from_env(
            root=tmp_path,
            passage_search=_SharedPassageSearch(),
            tracing=FakeTracing(),
            snippet_extractor=RecordingSnippetExtractor(),
            hits_per_search=True,
        )


def test_retrieve_uses_only_the_purpose_specific_tracing_api() -> None:
    fake_retriever = FakeRetriever()
    tracing = FakeTracing()
    result = _sdk(
        fake_retriever,
        lambda _model, toolset: FakeAgent(
            lambda _payload: (
                _authorized_snippet(toolset, "original-doc-1", "safe focus", None),
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
    assert tracing.snippet_records == []
    assert tracing.result_records == [
        {
            "fused_document_ids": ("original-doc-1",),
            "stopping_reason": "agent_completed",
            "coverage_state_hash": result.coverage_report.state_hash,
            "need_count": 1,
            "answerable_need_count": 0,
            "conflicted_need_count": 0,
            "unresolved_need_count": 1,
            "nugget_count": 0,
            "action_count": 1,
            "researcher_invocation_count": 1,
            "research_round_count": 0,
            "retrieval_call_count": 1,
            "budget_stop_code": None,
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
        lambda _model, toolset: FakeAgent(
            lambda _payload: (
                _authorized_snippet(
                    toolset, "original-doc-1", "private focus query", None
                ),
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
        "deepagent.retrieve",
    ]
    retriever_span, root_span = spans
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
    assert set(root_span.attributes) == {
        "input.value",
        "openinference.span.kind",
        "retrieval.fused_document_ids",
        "retrieval.stopping_reason",
        "coverage.state_hash",
        "coverage.need_count",
        "coverage.answerable_need_count",
        "coverage.conflicted_need_count",
        "coverage.unresolved_need_count",
        "coverage.nugget_count",
        "coverage.action_count",
        "deepagent.researcher_invocation_count",
        "deepagent.research_round_count",
        "deepagent.retrieval_call_count",
    }
    assert root_span.attributes["input.value"] == (
        "private narrative" if trace_content else REDACTED_CONTENT
    )
    assert root_span.attributes["retrieval.fused_document_ids"] == (
        "original-doc-1",
        "original-doc-2",
    )
    assert root_span.attributes["retrieval.stopping_reason"] == "agent_completed"
    assert root_span.attributes["coverage.state_hash"] == result.coverage_report.state_hash
    assert root_span.attributes["coverage.need_count"] == 1
    assert root_span.attributes["coverage.answerable_need_count"] == 0
    assert root_span.attributes["coverage.conflicted_need_count"] == 0
    assert root_span.attributes["coverage.unresolved_need_count"] == 1
    assert root_span.attributes["coverage.nugget_count"] == 0
    assert root_span.attributes["coverage.action_count"] == 1
    assert root_span.attributes["deepagent.researcher_invocation_count"] == 1
    assert root_span.attributes["deepagent.research_round_count"] == 0
    assert root_span.attributes["deepagent.retrieval_call_count"] == 1
    assert [candidate.docid for candidate in result.candidates] == [
        "original-doc-1",
        "original-doc-2",
    ]


def test_retrieve_traces_research_context_on_followup_search(
    isolated_real_tracing: None,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = create_retrieval_tracing(environ={}, tracer_provider=provider)

    _sdk(
        FakeRetriever(),
        lambda _model, toolset: FakeAgent(
            lambda _payload: (
                _authorized_search(toolset, "safe refined query"),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        ),
        tracing=tracing,
    ).retrieve("private narrative")

    search_spans = [
        span for span in exporter.get_finished_spans() if span.name == "climbmix.retrieve"
    ]
    assert len(search_spans) == 2
    followup_span = search_spans[1]
    assert followup_span.attributes["deepagent.research_task_id"].startswith(
        "test-research-"
    )
    assert followup_span.attributes["deepagent.research_round_index"] == 1
    assert followup_span.attributes["deepagent.research_depth"] == "focused"
    assert followup_span.attributes["deepagent.budget_code"] == "OK"
    assert followup_span.attributes["deepagent.remaining_retrieval_calls"] == 99


def test_from_env_preserves_injected_search_and_model_precedence(
    monkeypatch, tmp_path
) -> None:
    snippet_roots = []
    monkeypatch.setattr(
        "trec_rag.deepagent_retrieval.create_default_snippet_extractor",
        lambda root: snippet_roots.append(root) or RecordingSnippetExtractor(),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("DEEPAGENT_MODEL", "openrouter:environment-model")
    tracing = FakeTracing()
    passage_search = _SharedPassageSearch()

    configured = DeepAgentRetriever.from_env(
        root=tmp_path,
        passage_search=passage_search,
        model="openrouter:constructor-model",
        tracing=tracing,
    )
    from_environment = DeepAgentRetriever.from_env(
        root=tmp_path, passage_search=passage_search, tracing=tracing
    )
    monkeypatch.delenv("DEEPAGENT_MODEL")
    defaulted = DeepAgentRetriever.from_env(
        root=tmp_path, passage_search=passage_search, tracing=tracing
    )

    assert [sdk._model for sdk in (configured, from_environment, defaulted)] == [
        "openrouter:constructor-model",
        "openrouter:environment-model",
        "openrouter:deepseek/deepseek-v4-flash",
    ]
    assert snippet_roots == [tmp_path, tmp_path, tmp_path]
    assert all(
        sdk._passage_search is passage_search
        for sdk in (configured, from_environment, defaulted)
    )


def test_direct_construction_does_not_create_extractor_for_shared_search_results(
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
        lambda _model, _toolset: FakeAgent(
            lambda _payload: {"messages": [{"role": "assistant", "content": "Done."}]}
        ),
    ).retrieve("narrative")
    assert created_roots == []

    snippet_payloads: list[dict[str, object]] = []

    def agent_factory(
        _model: str,
        toolset: deepagent_retrieval.AgentToolset,
    ) -> FakeAgent:
        return FakeAgent(
            lambda _payload: (
                snippet_payloads.append(
                    json.loads(
                        _authorized_snippet(toolset, "original-doc-1", "focus", None)
                    )
                ),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        )

    _sdk(FakeRetriever(), agent_factory).retrieve("narrative")

    assert created_roots == []
    assert snippet_payloads[0]["code"] == "UNKNOWN_DOCUMENT"


class _SharedPassageSearch:
    topic_id = "internal-topic"

    def __init__(self, failing: bool = False) -> None:
        self.failing = failing

    def search(self, query: FocusedQuery) -> PassageSearchResult:
        if self.failing:
            return PassageSearchResult(
                query,
                "incomplete",
                "scoring_failed",
                1000,
                0,
                0,
                0,
                (),
                (),
                1,
                True,
            )
        documents = []
        passages = []
        for rank in range(1, 6):
            docid = f"original-doc-{rank}"
            text = f"Grounded evidence from document {rank}."
            digest = hashlib.sha256(text.encode()).hexdigest()
            passage_id = f"p-{docid}"
            score = float(100 - rank)
            documents.append(
                SourceDocument(docid, digest, rank, score, passage_id, score)
            )
            passages.append(
                SourcePassage(
                    passage_id,
                    docid,
                    digest,
                    rank,
                    score,
                    0,
                    len(text),
                    0,
                    len(text.encode()),
                    digest,
                    text,
                    score,
                    rank,
                    f"cache-{docid}",
                    digest,
                    {"backend": "test-chunker"},
                )
            )
        return PassageSearchResult(
            query,
            "complete",
            None,
            5,
            5,
            5,
            5,
            tuple(documents),
            tuple(passages),
            1,
            False,
        )


def _passage_toolset(*, scoring_failed=False, candidate_count=5):
    captured: list[deepagent_retrieval.AgentToolset] = []

    def agent_factory(_model, toolset):
        captured.append(toolset)

        def invoke(_payload):
            return {"structured_response": None}

        return SimpleNamespace(invoke=invoke)

    sdk = deepagent_retrieval.DeepAgentRetriever(
        passage_search=_SharedPassageSearch(scoring_failed),
        model="openrouter:test/model",
        agent_factory=agent_factory,
        tracing=FakeTracing(),
    )
    try:
        sdk.retrieve(_HarnessTopicLedger("internal-topic"), "narrative")
    except Exception:
        pass
    return captured[0]


def _authorized_passage_search(toolset, query):
    _seed_test_need(toolset, narrative_span="narrative")
    _, envelope = _test_research_context(toolset)
    with bind_research_task(envelope):
        return toolset.search_passages(query, ["test-need"], "test need remains open")


def test_passage_search_returns_shared_rows_across_multiple_documents() -> None:
    toolset = _passage_toolset()

    payload = json.loads(_authorized_passage_search(toolset, "a focused query"))

    assert payload["ok"] is True
    documents = {row["document_id"] for row in payload["passages"]}
    assert len(documents) >= 3, "selection collapsed onto too few documents"
    assert payload["distinct_documents"] == len(documents)
    assert payload["documents_scored"] == payload["documents_retrieved"]


def test_passage_search_mints_citable_handles_through_the_existing_ledger() -> None:
    toolset = _passage_toolset()

    payload = json.loads(_authorized_passage_search(toolset, "a focused query"))

    cites = [row["cite"] for row in payload["passages"]]
    assert cites, "no citable passages returned"
    assert all(cite.startswith("S") for cite in cites)
    assert len(cites) == len(set(cites))
    for row in payload["passages"]:
        assert row["sentences"], "a passage carried no citable sentences"
        assert row["sentences"][0]["n"] == 1


def test_passage_search_reports_what_it_did_not_return() -> None:
    toolset = _passage_toolset()

    payload = json.loads(_authorized_passage_search(toolset, "a focused query"))

    assert payload["documents_retrieved"] == 5
    assert payload["documents_scored"] == 5
    assert payload["chunks_scored"] >= payload["returned_passages"]
    assert payload["chunks_not_returned"] == (
        payload["chunks_scored"] - payload["returned_passages"]
    )


def test_a_scoring_failure_is_classified_and_never_reads_as_finding_nothing() -> None:
    """The recurring defect class: refusing correctly but costing the wrong thing."""
    toolset = _passage_toolset(scoring_failed=True)

    raw = _authorized_passage_search(toolset, "a focused query")
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["code"] == "PASSAGE_SCORING_FAILED"
    assert payload["passages"] == []
    # Raw exception text must not reach agent context.
    assert "exploded" not in raw
    assert "/home/user/key" not in raw


def test_repeated_shared_passage_rows_eventually_trigger_the_bounded_no_yield_stop() -> None:
    toolset = _passage_toolset()

    _seed_test_need(toolset, narrative_span="narrative")
    _, envelope = _test_research_context(toolset)
    with bind_research_task(envelope):
        for index in range(3):
            payload = json.loads(
                toolset.search_passages(f"query {index}", ["test-need"], "open")
            )
            assert payload["ok"] is True, payload
        json.loads(toolset.search_passages("one too many", ["test-need"], "open"))
        refused = json.loads(
            toolset.search_passages("the stop is now enforced", ["test-need"], "open")
        )

    assert refused["ok"] is False
    assert refused["code"] == "NO_YIELD_STOP"


def test_topic_scoped_retrieval_commits_validated_handoff_and_reads_snapshot() -> None:
    class RecordingTopicLedger:
        topic_id = "internal-topic"
        run_id = "run-1"

        def __init__(self) -> None:
            self._facets = {}
            self.searches = []
            self.handoffs = []
            self.snapshots = 0
            self.completions = []

        def add_facets(self, facets) -> None:
            self._facets.update({facet.subnarrative_id: facet for facet in facets})

        def add_passage_search(self, result) -> None:
            self.searches.append(result)

        def add_researcher_handoff(self, handoff) -> None:
            self.handoffs.append(handoff)

        def topic_snapshot(self):
            self.snapshots += 1
            return {"handoffs": len(self.handoffs)}

        def set_completion(self, status, stopping_reason) -> None:
            self.completions.append((status, stopping_reason))

    ledger = RecordingTopicLedger()

    def agent_factory(_model, toolset):
        def invoke(_payload):
            _seed_test_need(toolset)
            _test_research_context(toolset)
            update = toolset.update_retrieval_state(
                {
                    "add_facets": [
                        {
                            "facet_id": "test-need",
                            "need_ids": ["test-need"],
                            "dimension": "evidence",
                            "value": "test evidence",
                            "origin": "narrative",
                        }
                    ]
                }
            )
            assert "test-need" in json.loads(update)["accepted_ids"]
            _, envelope = _test_research_context(toolset)
            with bind_research_task(envelope):
                payload = json.loads(
                    toolset.search_passages(
                        "focused evidence query",
                        ["test-need"],
                        "test need remains open",
                    )
                )
            assert payload["ok"] is True
            toolset.update_retrieval_state(
                {
                    "add_nuggets": [
                        {
                            "nugget_id": "g1",
                            "text": "Grounded test evidence.",
                            "need_ids": ["test-need"],
                            "facet_ids": ["test-need"],
                            "evidence": [{"cite": "S1"}],
                            "contradicts": [],
                        }
                    ]
                }
            )
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    result = DeepAgentRetriever(
        passage_search=_SharedPassageSearch(),
        agent_factory=agent_factory,
        tracing=FakeTracing(),
        model="test-model",
    ).retrieve(ledger, "narrative")

    assert len(ledger.searches) == 2
    assert len(ledger.handoffs) == 1
    assert ledger.handoffs[0].run_id == "run-1"
    assert ledger.handoffs[0].evidence[0].passage_id == "p-original-doc-1"
    assert ledger.snapshots == 1
    assert ledger.completions == [("incomplete", "no_evidence")]
    assert result.topic_snapshot == {"handoffs": 1}


def test_active_api_requires_shared_search_and_topic_records_builder() -> None:
    constructor = inspect.signature(DeepAgentRetriever).parameters
    retrieve = inspect.signature(DeepAgentRetriever.retrieve).parameters

    assert constructor["passage_search"].default is inspect.Parameter.empty
    assert "retriever" not in constructor
    assert list(retrieve) == ["self", "records", "narrative"]
    assert retrieve["narrative"].default is inspect.Parameter.empty


def test_from_env_uses_injected_shared_search_without_constructing_legacy_retriever(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    legacy_constructions: list[object] = []
    monkeypatch.setattr(
        deepagent_retrieval,
        "PyseriniRemoteRetriever",
        lambda *_args, **_kwargs: legacy_constructions.append(object()),
        raising=False,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    passage_search = _SharedPassageSearch()

    sdk = DeepAgentRetriever.from_env(
        root=tmp_path,
        passage_search=passage_search,
        tracing=FakeTracing(),
        snippet_extractor=RecordingSnippetExtractor(),
    )

    assert sdk._passage_search is passage_search
    assert legacy_constructions == []


def test_retrieve_rejects_cross_topic_scope_before_search_or_ledger_admission() -> None:
    class PoisonPassageSearch:
        topic_id = "topic-search"

        def __init__(self) -> None:
            self.calls = 0

        def search(self, _query: FocusedQuery) -> PassageSearchResult:
            self.calls += 1
            raise AssertionError("cross-topic search was admitted")

    class PoisonRecords:
        topic_id = "topic-records"
        run_id = "run-1"

        def __init__(self) -> None:
            self.admissions = 0

        def _poison(self, *_args, **_kwargs) -> None:
            self.admissions += 1
            raise AssertionError("cross-topic ledger admission occurred")

        add_facets = _poison
        add_passage_search = _poison
        add_researcher_handoff = _poison
        set_completion = _poison

        def topic_snapshot(self):
            self.admissions += 1
            raise AssertionError("cross-topic snapshot occurred")

    passage_search = PoisonPassageSearch()
    records = PoisonRecords()
    sdk = DeepAgentRetriever(
        passage_search=passage_search,
        agent_factory=lambda *_args: (_ for _ in ()).throw(
            AssertionError("agent construction occurred")
        ),
        tracing=FakeTracing(),
        model="test-model",
    )

    with pytest.raises(ValueError, match="topic"):
        sdk.retrieve(records, "narrative")

    assert passage_search.calls == 0
    assert records.admissions == 0


class _RecordingTopicLedger:
    def __init__(
        self,
        *,
        topic_id: str = "internal-topic",
        run_id: str = "run-1",
    ) -> None:
        self.topic_id = topic_id
        self.run_id = run_id
        self._facets = {}
        self.searches = []
        self.handoffs = []
        self.completion: tuple[str, str] | None = None
        self.events: list[tuple[object, ...]] = []
        self.snapshots = 0

    def add_facets(self, facets) -> None:
        self._facets.update((facet.subnarrative_id, facet) for facet in facets)
        self.events.append(
            ("facets", tuple(facet.subnarrative_id for facet in facets))
        )

    def add_passage_search(self, result) -> None:
        self.searches.append(result)
        self.events.append(("search", result.query.query_id))

    def add_researcher_handoff(self, handoff) -> None:
        self.handoffs.append(handoff)
        self.events.append(("handoff", handoff.researcher_id))

    def set_completion(self, status: str, stopping_reason: str) -> None:
        self.completion = (status, stopping_reason)
        self.events.append(("completion", status, stopping_reason))

    def topic_snapshot(self):
        self.snapshots += 1
        self.events.append(("snapshot", self.completion))
        return SimpleNamespace(
            status=self.completion[0] if self.completion is not None else None,
            stopping_reason=(
                self.completion[1] if self.completion is not None else None
            ),
            researcher_handoffs=tuple(self.handoffs),
        )


class _QuerySpecificPassageSearch:
    topic_id = "internal-topic"

    def search(self, query: FocusedQuery) -> PassageSearchResult:
        suffix = hashlib.sha256(query.text.encode("utf-8")).hexdigest()[:12]
        docid = f"doc-{suffix}"
        text = f"Grounded evidence for {query.text}."
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        passage_id = f"p-{suffix}"
        document = SourceDocument(docid, digest, 1, 1.0, passage_id, 1.0)
        passage = SourcePassage(
            passage_id,
            docid,
            digest,
            1,
            1.0,
            0,
            len(text),
            0,
            len(text.encode("utf-8")),
            digest,
            text,
            1.0,
            1,
            f"cache-{suffix}",
            digest,
            {"backend": "query-specific-test-chunker"},
        )
        return PassageSearchResult(
            query,
            "complete",
            None,
            1000,
            1,
            1,
            1,
            (document,),
            (passage,),
            1,
            True,
        )


class _FollowupExplodingPassageSearch(_QuerySpecificPassageSearch):
    def search(self, query: FocusedQuery) -> PassageSearchResult:
        if query.query_id == "original":
            return super().search(query)
        raise RuntimeError("private worker failure detail")


class _OffsetPassageSearch:
    topic_id = "internal-topic"

    passage_id = "p-offset-exact"
    passage_text = "The retained passage carries the grounded claim."
    prefix = "é" * 17
    document_text = prefix + passage_text + " Trailing source text."
    start_char = len(prefix)
    end_char = start_char + len(passage_text)
    start_byte = len(prefix.encode("utf-8"))
    end_byte = start_byte + len(passage_text.encode("utf-8"))

    def __init__(self, store: DocumentStore | None = None) -> None:
        if store is not None:
            store.admit_text(self.document_text)

    def search(self, query: FocusedQuery) -> PassageSearchResult:
        content_digest = hashlib.sha256(
            self.document_text.encode("utf-8")
        ).hexdigest()
        passage_digest = hashlib.sha256(
            self.passage_text.encode("utf-8")
        ).hexdigest()
        passage = SourcePassage(
            self.passage_id,
            "doc-offset",
            content_digest,
            1,
            0.95,
            self.start_char,
            self.end_char,
            self.start_byte,
            self.end_byte,
            passage_digest,
            self.passage_text,
            0.95,
            1,
            "cache-offset-exact",
            passage_digest,
            {"backend": "offset-test-chunker"},
        )
        document = SourceDocument(
            "doc-offset", content_digest, 1, 0.95, self.passage_id, 0.95
        )
        return PassageSearchResult(
            query,
            "complete",
            None,
            1000,
            1,
            1,
            1,
            (document,),
            (passage,),
            1,
            True,
        )


def _topic_sdk(
    agent_factory,
    *,
    passage_search=None,
    budget_config: ResearchBudgetConfig | None = None,
) -> DeepAgentRetriever:
    return DeepAgentRetriever(
        passage_search=passage_search or _QuerySpecificPassageSearch(),
        agent_factory=agent_factory,
        tracing=FakeTracing(),
        model="test-model",
        budget_config=budget_config,
    )


def _add_needs(toolset, *need_ids: str) -> None:
    result = json.loads(
        toolset.update_retrieval_state(
            {
                "add_needs": [
                    {
                        "need_id": need_id,
                        "narrative_span": "narrative",
                        "question": f"What evidence closes {need_id}?",
                    }
                    for need_id in need_ids
                ]
            }
        )
    )
    assert set(need_ids) <= set(result["accepted_ids"])


def _add_need_and_facet(toolset, *, need_id: str, facet_id: str) -> None:
    result = json.loads(
        toolset.update_retrieval_state(
            {
                "add_needs": [
                    {
                        "need_id": need_id,
                        "narrative_span": "narrative",
                        "question": f"What evidence closes {need_id}?",
                    }
                ],
                "add_facets": [
                    {
                        "facet_id": facet_id,
                        "need_ids": [need_id],
                        "dimension": "evidence",
                        "value": f"Evidence for {facet_id}",
                        "origin": "narrative",
                    }
                ],
            }
        )
    )
    assert {need_id, facet_id} <= set(result["accepted_ids"])


def _run_researcher_search(
    toolset,
    *,
    researcher_id: str,
    round_index: int,
    motivating_id: str,
    query: str,
) -> dict[str, object]:
    context = ResearchTaskContext(
        researcher_id,
        round_index,
        "focused",
        (motivating_id,),
    )
    envelope = ResearchTaskEnvelope(
        research_task_id=researcher_id,
        round_index=round_index,
        depth="focused",
        motivating_ids=[motivating_id],
        goal=f"Find evidence for {motivating_id}.",
    )
    assert toolset.budget.reserve_task(context).ok
    try:
        with bind_research_task(envelope):
            return json.loads(
                toolset.search_passages(
                    query,
                    [motivating_id],
                    f"{motivating_id} has no evidence.",
                )
            )
    finally:
        toolset.budget.finish_task(context)


def test_unexpected_research_tool_error_is_sanitized_and_raised_after_agent(
) -> None:
    captured: dict[str, object] = {}
    records = _RecordingTopicLedger(
        topic_id="internal-topic",
        run_id="run-tool-failure",
    )

    def agent_factory(_model, toolset):
        def invoke(_payload):
            _add_need_and_facet(toolset, need_id="need-1", facet_id="facet-1")
            captured["tool_payload"] = _run_researcher_search(
                toolset,
                researcher_id="researcher-1",
                round_index=1,
                motivating_id="facet-1",
                query="trigger worker failure",
            )
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    with pytest.raises(AgentRetrievalError, match="retrieval tool failed") as raised:
        _topic_sdk(
            agent_factory,
            passage_search=_FollowupExplodingPassageSearch(),
        ).retrieve(records, "narrative")

    tool_payload = captured["tool_payload"]
    assert isinstance(tool_payload, dict)
    assert set(tool_payload) == {
        "budget_snapshot",
        "code",
        "error",
        "must_stop",
        "ok",
        "tool",
    }
    assert tool_payload["code"] == "INTERNAL_TOOL_FAILURE"
    assert tool_payload["error"] == "internal retrieval tool failed"
    assert tool_payload["must_stop"] is True
    assert tool_payload["ok"] is False
    assert tool_payload["tool"] == "search_passages"
    assert isinstance(tool_payload["budget_snapshot"], dict)
    assert "private worker failure detail" not in json.dumps(tool_payload)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert [search.query.query_id for search in records.searches] == ["original"]
    assert records.completion is None


def _assert_returned_completion(
    result,
    records: _RecordingTopicLedger,
    *,
    status: str,
    stopping_reason: str,
) -> None:
    assert result.topic_snapshot.status == status
    assert result.topic_snapshot.stopping_reason == stopping_reason
    assert records.snapshots == 1
    assert records.events[-2:] == [
        ("completion", status, stopping_reason),
        ("snapshot", (status, stopping_reason)),
    ]


def test_direct_passage_citation_retains_exact_source_passage_and_offsets(
    tmp_path,
) -> None:
    store = DocumentStore(tmp_path / "objects")
    passage_search = _OffsetPassageSearch(store)
    records = TopicRecordsBuilder(
        tmp_path / "topic",
        passage_search.topic_id,
        store,
        run_id="run-1",
    )

    def agent_factory(_model, toolset):
        def invoke(_payload):
            _add_need_and_facet(toolset, need_id="need-1", facet_id="facet-1")
            payload = _run_researcher_search(
                toolset,
                researcher_id="researcher-1",
                round_index=1,
                motivating_id="facet-1",
                query="offset grounded evidence",
            )
            update = json.loads(
                toolset.update_retrieval_state(
                    {
                        "add_nuggets": [
                            {
                                "nugget_id": "nugget-offset",
                                "text": passage_search.passage_text,
                                "need_ids": ["need-1"],
                                "facet_ids": ["facet-1"],
                                "evidence": [{"cite": payload["passages"][0]["cite"]}],
                            }
                        ]
                    }
                )
            )
            assert update["accepted_ids"] == ["nugget-offset"]
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    result = _topic_sdk(
        agent_factory,
        passage_search=passage_search,
    ).retrieve(records, "narrative")

    assert result.stopping_reason != "evidence_validation_failed"
    recorded = next(
        passage
        for passage in result.topic_snapshot.passages
        if passage.passage_id == passage_search.passage_id
    )
    assert (
        recorded.start_char,
        recorded.end_char,
        recorded.start_byte,
        recorded.end_byte,
    ) == (
        passage_search.start_char,
        passage_search.end_char,
        passage_search.start_byte,
        passage_search.end_byte,
    )
    assert [
        (row.subnarrative_id, row.passage_id)
        for handoff in result.topic_snapshot.researcher_handoffs
        for row in handoff.evidence
    ] == [
        ("facet-1", passage_search.passage_id)
    ]


def test_direct_need_citation_retains_exact_source_passage_and_offsets(
    tmp_path,
) -> None:
    store = DocumentStore(tmp_path / "objects")
    passage_search = _OffsetPassageSearch(store)
    records = TopicRecordsBuilder(
        tmp_path / "topic",
        passage_search.topic_id,
        store,
        run_id="run-1",
    )

    def agent_factory(_model, toolset):
        def invoke(_payload):
            _add_needs(toolset, "need-1")
            payload = _run_researcher_search(
                toolset,
                researcher_id="researcher-1",
                round_index=1,
                motivating_id="need-1",
                query="direct need grounded evidence",
            )
            update = json.loads(
                toolset.update_retrieval_state(
                    {
                        "add_nuggets": [
                            {
                                "nugget_id": "nugget-unhanded-off",
                                "text": "The retained passage carries the grounded claim.",
                                "need_ids": ["need-1"],
                                "facet_ids": [],
                                "evidence": [{"cite": payload["passages"][0]["cite"]}],
                            }
                        ]
                    }
                )
            )
            assert update["accepted_ids"] == ["nugget-unhanded-off"]
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    result = _topic_sdk(
        agent_factory,
        passage_search=passage_search,
    ).retrieve(records, "narrative")

    assert result.stopping_reason != "evidence_validation_failed"
    recorded = next(
        passage
        for passage in result.topic_snapshot.passages
        if passage.passage_id == passage_search.passage_id
    )
    assert (
        recorded.start_char,
        recorded.end_char,
        recorded.start_byte,
        recorded.end_byte,
    ) == (
        passage_search.start_char,
        passage_search.end_char,
        passage_search.start_byte,
        passage_search.end_byte,
    )
    assert [
        (row.subnarrative_id, row.passage_id)
        for handoff in result.topic_snapshot.researcher_handoffs
        for row in handoff.evidence
    ] == [("need-1", passage_search.passage_id)]


def test_direct_multi_need_citation_admits_each_need_and_deduplicates_passage(
    tmp_path,
) -> None:
    store = DocumentStore(tmp_path / "objects")
    passage_search = _OffsetPassageSearch(store)
    records = TopicRecordsBuilder(
        tmp_path / "topic",
        passage_search.topic_id,
        store,
        run_id="run-1",
    )

    def agent_factory(_model, toolset):
        def invoke(_payload):
            _add_needs(toolset, "need-1", "need-2")
            payload = _run_researcher_search(
                toolset,
                researcher_id="researcher-1",
                round_index=1,
                motivating_id="need-1",
                query="multi need grounded evidence",
            )
            update = json.loads(
                toolset.update_retrieval_state(
                    {
                        "add_nuggets": [
                            {
                                "nugget_id": nugget_id,
                                "text": passage_search.passage_text,
                                "need_ids": ["need-1", "need-2"],
                                "facet_ids": [],
                                "evidence": [
                                    {"cite": payload["passages"][0]["cite"]}
                                ],
                            }
                            for nugget_id in ("nugget-one", "nugget-two")
                        ]
                    }
                )
            )
            assert set(update["accepted_ids"]) == {"nugget-one", "nugget-two"}
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    result = _topic_sdk(
        agent_factory,
        passage_search=passage_search,
    ).retrieve(records, "narrative")

    assert result.stopping_reason != "evidence_validation_failed"
    assert sorted(
        (row.subnarrative_id, row.passage_id)
        for handoff in result.topic_snapshot.researcher_handoffs
        for row in handoff.evidence
    ) == [
        ("need-1", passage_search.passage_id),
        ("need-2", passage_search.passage_id),
    ]


def test_live_coverage_citation_dropped_by_projector_fails_validation(
    monkeypatch,
) -> None:
    records = _RecordingTopicLedger()
    evidence_type = deepagent_retrieval.ResearcherEvidence

    def replace_passage_id(subnarrative_id, passage_id, relevance):
        return evidence_type(
            subnarrative_id,
            f"dropped-{passage_id}",
            relevance,
        )

    monkeypatch.setattr(
        deepagent_retrieval,
        "ResearcherEvidence",
        replace_passage_id,
    )

    def agent_factory(_model, toolset):
        def invoke(_payload):
            _add_need_and_facet(toolset, need_id="need-1", facet_id="facet-1")
            payload = _run_researcher_search(
                toolset,
                researcher_id="researcher-1",
                round_index=1,
                motivating_id="facet-1",
                query="citation dropped from durable handoff",
            )
            update = json.loads(
                toolset.update_retrieval_state(
                    {
                        "add_nuggets": [
                            {
                                "nugget_id": "nugget-dropped",
                                "text": "The retained passage carries the grounded claim.",
                                "need_ids": ["need-1"],
                                "facet_ids": ["facet-1"],
                                "evidence": [
                                    {"cite": payload["passages"][0]["cite"]}
                                ],
                            }
                        ]
                    }
                )
            )
            assert update["accepted_ids"] == ["nugget-dropped"]
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    result = _topic_sdk(
        agent_factory,
        passage_search=_OffsetPassageSearch(),
    ).retrieve(records, "narrative")

    assert result.stopping_reason == "evidence_validation_failed"
    assert result.topic_snapshot.status == "incomplete"
    assert result.topic_snapshot.stopping_reason == "evidence_validation_failed"


def test_legacy_extraction_after_shared_search_returns_unknown_document() -> None:
    records = _RecordingTopicLedger()
    extractor = RecordingSnippetExtractor()

    def agent_factory(_model, toolset):
        def invoke(_payload):
            _add_need_and_facet(toolset, need_id="need-1", facet_id="facet-1")
            payload = _run_researcher_search(
                toolset,
                researcher_id="researcher-1",
                round_index=1,
                motivating_id="facet-1",
                query="shared search before legacy extraction",
            )
            context = ResearchTaskContext("researcher-2", 1, "focused", ("facet-1",))
            envelope = ResearchTaskEnvelope(
                research_task_id="researcher-2",
                round_index=1,
                depth="focused",
                motivating_ids=["facet-1"],
                goal="Find evidence for facet-1.",
            )
            assert toolset.budget.reserve_task(context).ok
            try:
                with bind_research_task(envelope):
                    response = json.loads(
                        toolset.extract_relevant_snippets(
                            payload["passages"][0]["document_id"],
                            "shared search before legacy extraction",
                            ["facet-1"],
                            "facet-1 still needs evidence.",
                        )
                    )
            finally:
                toolset.budget.finish_task(context)
            assert response["code"] == "UNKNOWN_DOCUMENT"
            assert extractor.calls == []
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    _topic_sdk(agent_factory, passage_search=_OffsetPassageSearch()).retrieve(
        records, "narrative"
    )


def test_provider_stop_after_validated_research_returns_and_persists_partial_topic() -> None:
    records = _RecordingTopicLedger()

    def agent_factory(_model, toolset):
        def invoke(_payload):
            _add_need_and_facet(toolset, need_id="need-1", facet_id="facet-1")
            payload = _run_researcher_search(
                toolset,
                researcher_id="researcher-1",
                round_index=1,
                motivating_id="facet-1",
                query="provider partial evidence",
            )
            update = json.loads(
                toolset.update_retrieval_state(
                    {
                        "add_nuggets": [
                            {
                                "nugget_id": "nugget-1",
                                "text": "Validated provider-partial evidence.",
                                "need_ids": ["need-1"],
                                "facet_ids": ["facet-1"],
                                "evidence": [
                                    {"cite": payload["passages"][0]["cite"]}
                                ],
                            }
                        ]
                    }
                )
            )
            assert update["accepted_ids"] == ["nugget-1"]
            raise RuntimeError("provider unavailable")

        return FakeAgent(invoke)

    result = _topic_sdk(agent_factory).retrieve(records, "narrative")

    assert result.stopping_reason == "retrieval_unavailable"
    assert len(records.handoffs) == 1
    assert [row.passage_id for row in records.handoffs[0].evidence] == [
        records.searches[1].passages[0].passage_id
    ]
    _assert_returned_completion(
        result,
        records,
        status="incomplete",
        stopping_reason="retrieval_unavailable",
    )


def test_programming_failure_after_search_remains_loud() -> None:
    records = _RecordingTopicLedger()

    def agent_factory(_model, _toolset):
        return FakeAgent(
            lambda _payload: (_ for _ in ()).throw(ValueError("programming conflict"))
        )

    with pytest.raises(ValueError, match="programming conflict"):
        _topic_sdk(agent_factory).retrieve(records, "narrative")

    assert records.completion is None
    assert records.snapshots == 0


def test_two_researchers_commit_only_their_evidence_and_discovered_facets() -> None:
    records = _RecordingTopicLedger()

    def agent_factory(_model, toolset):
        def invoke(_payload):
            for need_id in ("need-1", "need-2"):
                accepted = json.loads(
                    toolset.update_retrieval_state(
                        {
                            "add_needs": [
                                {
                                    "need_id": need_id,
                                    "narrative_span": "narrative",
                                    "question": f"What closes {need_id}?",
                                }
                            ]
                        }
                    )
                )
                assert accepted["accepted_ids"] == [need_id]

            for index in (1, 2):
                researcher_id = f"researcher-{index}"
                need_id = f"need-{index}"
                facet_id = f"discovered-{index}"
                payload = _run_researcher_search(
                    toolset,
                    researcher_id=researcher_id,
                    round_index=index,
                    motivating_id=need_id,
                    query=f"disjoint query {index}",
                )
                passage = payload["passages"][0]
                envelope = ResearchTaskEnvelope(
                    research_task_id=researcher_id,
                    round_index=index,
                    depth="focused",
                    motivating_ids=[need_id],
                    goal=f"Find evidence for {need_id}.",
                )
                with bind_research_task(envelope):
                    update = json.loads(
                        toolset.update_retrieval_state(
                            {
                                "add_facets": [
                                    {
                                        "facet_id": facet_id,
                                        "need_ids": [need_id],
                                        "dimension": "discovered",
                                        "value": f"Discovered facet {index}",
                                        "origin": "snippet",
                                        "origin_snippet_id": passage["passage_id"],
                                    }
                                ],
                                "add_nuggets": [
                                    {
                                        "nugget_id": f"nugget-{index}",
                                        "text": f"Disjoint evidence {index}.",
                                        "need_ids": [need_id],
                                        "facet_ids": [facet_id],
                                        "evidence": [{"cite": passage["cite"]}],
                                    }
                                ],
                            }
                        )
                    )
                assert set(update["accepted_ids"]) == {
                    facet_id,
                    f"nugget-{index}",
                }
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    _topic_sdk(agent_factory).retrieve(records, "narrative")

    handoffs = {row.researcher_id: row for row in records.handoffs}
    assert set(handoffs) == {"researcher-1", "researcher-2"}
    assert [row.subnarrative_id for row in handoffs["researcher-1"].facet_updates] == [
        "discovered-1"
    ]
    assert [row.subnarrative_id for row in handoffs["researcher-2"].facet_updates] == [
        "discovered-2"
    ]
    assert [row.passage_id for row in handoffs["researcher-1"].evidence] == [
        records.searches[1].passages[0].passage_id
    ]
    assert [row.passage_id for row in handoffs["researcher-2"].evidence] == [
        records.searches[2].passages[0].passage_id
    ]


def test_shared_passage_discovered_facet_is_handed_off_only_by_its_creator() -> None:
    records = _RecordingTopicLedger()

    def agent_factory(_model, toolset):
        def invoke(_payload):
            for index in (1, 2):
                _add_need_and_facet(
                    toolset,
                    need_id=f"need-{index}",
                    facet_id=f"facet-{index}",
                )

            observed_passage_ids = []
            for index in (1, 2):
                researcher_id = f"researcher-{index}"
                motivating_id = f"facet-{index}"
                context = ResearchTaskContext(
                    researcher_id,
                    index,
                    "focused",
                    (motivating_id,),
                )
                envelope = ResearchTaskEnvelope(
                    research_task_id=researcher_id,
                    round_index=index,
                    depth="focused",
                    motivating_ids=[motivating_id],
                    goal=f"Find evidence for {motivating_id}.",
                )
                assert toolset.budget.reserve_task(context).ok
                try:
                    with bind_research_task(envelope):
                        payload = json.loads(
                            toolset.search_passages(
                                f"overlapping query {index}",
                                [motivating_id],
                                f"{motivating_id} has no evidence.",
                            )
                        )
                        passage = payload["passages"][0]
                        observed_passage_ids.append(passage["passage_id"])
                        if index == 1:
                            update = json.loads(
                                toolset.update_retrieval_state(
                                    {
                                        "add_facets": [
                                            {
                                                "facet_id": "creator-only-facet",
                                                "need_ids": ["need-1"],
                                                "dimension": "discovered",
                                                "value": "Created only by researcher 1",
                                                "origin": "snippet",
                                                "origin_snippet_id": passage[
                                                    "passage_id"
                                                ],
                                            }
                                        ]
                                    }
                                )
                            )
                            assert update["accepted_ids"] == ["creator-only-facet"]
                finally:
                    toolset.budget.finish_task(context)

            assert observed_passage_ids[0] == observed_passage_ids[1]
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    _topic_sdk(
        agent_factory,
        passage_search=_SharedPassageSearch(),
    ).retrieve(records, "narrative")

    handoffs = {row.researcher_id: row for row in records.handoffs}
    assert [row.subnarrative_id for row in handoffs["researcher-1"].facet_updates] == [
        "creator-only-facet"
    ]
    assert handoffs["researcher-2"].facet_updates == ()


def test_complete_topic_is_written_before_returned_snapshot() -> None:
    records = _RecordingTopicLedger()

    def agent_factory(_model, toolset):
        def invoke(_payload):
            _add_need_and_facet(toolset, need_id="need-1", facet_id="facet-1")
            passage = _run_researcher_search(
                toolset,
                researcher_id="researcher-1",
                round_index=1,
                motivating_id="facet-1",
                query="complete grounded evidence",
            )
            update = json.loads(
                toolset.update_retrieval_state(
                    {
                        "add_nuggets": [
                            {
                                "nugget_id": "nugget-1",
                                "text": "Complete grounded evidence.",
                                "need_ids": ["need-1"],
                                "facet_ids": ["facet-1"],
                                "evidence": [{"cite": passage["passages"][0]["cite"]}],
                            }
                        ],
                        "set_need_status": [
                            {
                                "need_id": "need-1",
                                "status": "answerable",
                                "remaining_gap": "",
                                "draft_answer": "Complete grounded answer.",
                                "draft_nugget_ids": ["nugget-1"],
                            }
                        ],
                    }
                )
            )
            assert "nugget-1" in update["accepted_ids"]
            assert json.loads(toolset.complete_retrieval())["ok"] is True
            return {"messages": [{"role": "assistant", "content": "Complete."}]}

        return FakeAgent(invoke)

    result = _topic_sdk(agent_factory).retrieve(records, "narrative")

    assert result.stopping_reason == "completion"
    _assert_returned_completion(
        result,
        records,
        status="complete",
        stopping_reason="coverage_sufficient",
    )


def test_budget_exhaustion_is_written_before_returned_snapshot() -> None:
    records = _RecordingTopicLedger()

    def agent_factory(_model, toolset):
        def invoke(_payload):
            _add_need_and_facet(toolset, need_id="need-1", facet_id="facet-1")
            payload = _run_researcher_search(
                toolset,
                researcher_id="researcher-1",
                round_index=1,
                motivating_id="facet-1",
                query="single budgeted query",
            )
            assert payload["ok"] is True
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    result = _topic_sdk(
        agent_factory,
        budget_config=ResearchBudgetConfig(max_retrieval_calls=1),
    ).retrieve(records, "narrative")

    assert result.stopping_reason == "budget_exhausted"
    _assert_returned_completion(
        result,
        records,
        status="incomplete",
        stopping_reason="budget_exhausted",
    )


def test_hard_deadline_is_written_before_returned_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClock:
        seconds = 0.0

        def __call__(self) -> float:
            return self.seconds

    clock = FakeClock()
    real_budget = ResearchBudget
    monkeypatch.setattr(
        deepagent_retrieval,
        "ResearchBudget",
        lambda config: real_budget(config, clock=clock),
    )
    records = _RecordingTopicLedger()

    def agent_factory(_model, _toolset):
        def invoke(_payload):
            clock.seconds = 2.0
            return {"messages": [{"role": "assistant", "content": "Partial."}]}

        return FakeAgent(invoke)

    result = _topic_sdk(
        agent_factory,
        budget_config=ResearchBudgetConfig(soft_seconds=0, hard_seconds=1),
    ).retrieve(records, "narrative")

    assert result.stopping_reason == "hard_deadline"
    _assert_returned_completion(
        result,
        records,
        status="incomplete",
        stopping_reason="hard_deadline",
    )


def test_no_evidence_is_written_before_returned_snapshot() -> None:
    records = _RecordingTopicLedger()
    result = _topic_sdk(
        lambda _model, _toolset: FakeAgent(
            lambda _payload: {
                "messages": [{"role": "assistant", "content": "No evidence."}]
            }
        )
    ).retrieve(records, "narrative")

    _assert_returned_completion(
        result,
        records,
        status="incomplete",
        stopping_reason="no_evidence",
    )
