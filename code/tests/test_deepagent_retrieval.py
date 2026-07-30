from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from unittest.mock import ANY

import pytest
import trec_rag.deepagent_retrieval as deepagent_retrieval
from deepagents import create_deep_agent
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_openrouter import ChatOpenRouter
from pydantic import Field

from trec_rag.deepagent_retrieval import (
    AgentRetrievalError,
    AgentSearch,
    DeepAgentRetriever,
    _create_agent,
    reciprocal_rank_fuse,
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


@dataclass
class FakeAgent:
    invoke_callback: Callable[[dict[str, object]], object]

    def invoke(self, payload: dict[str, object]) -> object:
        return self.invoke_callback(payload)


class FakeTracing:
    def __init__(self) -> None:
        self.flushes = 0
        self.attribute_keys = []

    @contextmanager
    def agent_span(self, _narrative: str):
        yield self

    @contextmanager
    def retriever_span(self, _query: str):
        yield self

    def set_attribute(self, key: str, _value: object) -> None:
        self.attribute_keys.append(key)

    def force_flush(self) -> bool:
        self.flushes += 1
        return True


class CaptureChatModel(FakeMessagesListChatModel):
    """Offline chat model that records the tools Deep Agents exposes to it."""

    captured_tool_names: list[str] = Field(default_factory=list)

    def bind_tools(
        self, tools: Sequence[object], **_kwargs: object
    ) -> "CaptureChatModel":
        self.captured_tool_names = [str(getattr(tool, "name", "")) for tool in tools]
        return self


def _sdk(
    retriever: FakeRetriever,
    agent_factory: Callable[[str, Callable[[str], str]], FakeAgent],
    *,
    tracing: FakeTracing | None = None,
) -> DeepAgentRetriever:
    return DeepAgentRetriever(
        retriever=retriever,
        agent_factory=agent_factory,
        tracing=tracing or FakeTracing(),
        model="test-model",
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
    with pytest.raises(TypeError):
        fused[0].provenance[0]["query_kind"] = "followup"

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

    def agent_factory(model: str, search_tool: Callable[[str], str]) -> FakeAgent:
        factory_models.append(model)
        factory_queries.extend(fake_retriever.queries)
        return FakeAgent(
            lambda _payload: (
                search_tool("targeted follow-up"),
                {"messages": [{"role": "assistant", "content": "Coverage is sufficient."}]},
            )[1]
        )

    result = _sdk(fake_retriever, agent_factory).retrieve("  supplied narrative exactly  ")

    assert [query.query_text for query in factory_queries] == ["  supplied narrative exactly  "]
    assert factory_models == ["test-model"]
    assert fake_retriever.queries[0].query_text == "  supplied narrative exactly  "
    assert [search.kind for search in result.searches] == ["original", "followup"]
    assert result.narrative == "  supplied narrative exactly  "
    assert result.rationale == "Coverage is sufficient."


def test_retrieve_rejects_empty_narrative_before_external_calls() -> None:
    fake_retriever = FakeRetriever()
    factory_calls = []

    with pytest.raises(ValueError, match="narrative must be non-empty text"):
        _sdk(
            fake_retriever,
            lambda _model, _tool: factory_calls.append(True),  # type: ignore[return-value]
        ).retrieve("   ")

    assert fake_retriever.queries == []
    assert factory_calls == []


def test_retrieve_limits_successful_followups_to_three_and_marks_budget_exhaustion() -> None:
    fake_retriever = FakeRetriever()
    tool_results = []

    def agent_factory(_model: str, search_tool: Callable[[str], str]) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            tool_results.extend(search_tool(f"query {number}") for number in range(4))
            return {"messages": [{"role": "assistant", "content": "Stopped after coverage."}]}

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


def test_retrieve_rejects_blank_and_original_duplicate_followups_without_using_budget() -> None:
    fake_retriever = FakeRetriever()
    tool_results = []

    def agent_factory(_model: str, search_tool: Callable[[str], str]) -> FakeAgent:
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

    def agent_factory(_model: str, search_tool: Callable[[str], str]) -> FakeAgent:
        return FakeAgent(
            lambda _payload: (
                search_tool("specific missing aspect"),
                {"messages": [{"role": "assistant", "content": "Done."}]},
            )[1]
        )

    narrative = "Narrative whose ID must not be public."
    _sdk(fake_retriever, agent_factory).retrieve(narrative)

    assert fake_retriever.queries[0].topic_id == hashlib.sha256(
        narrative.encode()
    ).hexdigest()[:16]
    assert fake_retriever.queries[0].variant_name == "original"
    assert fake_retriever.queries[1].variant_name == "followup-" + hashlib.sha256(
        b"specific missing aspect"
    ).hexdigest()[:16]


def test_retrieve_wraps_agent_failure_with_completed_searches() -> None:
    fake_retriever = FakeRetriever()
    failure = RuntimeError("provider unavailable")
    tracing = FakeTracing()

    def agent_factory(_model: str, _search_tool: Callable[[str], str]) -> FakeAgent:
        return FakeAgent(lambda _payload: (_ for _ in ()).throw(failure))

    with pytest.raises(AgentRetrievalError) as raised:
        _sdk(fake_retriever, agent_factory, tracing=tracing).retrieve("narrative")

    assert [search.kind for search in raised.value.searches] == ["original"]
    assert raised.value.__cause__ is failure
    assert tracing.flushes == 1


def test_factory_passes_only_explicit_deepagents_070_arguments(monkeypatch) -> None:
    calls = []

    def fake_create_deep_agent(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr("deepagents.create_deep_agent", fake_create_deep_agent)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    tool = lambda _query: "{}"

    agent = _create_agent("openrouter:deepseek/test-model", tool)

    assert agent is not None
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert set(kwargs) == {"model", "tools", "system_prompt", "middleware"}
    assert isinstance(kwargs["model"], ChatOpenRouter)
    assert kwargs["model"].model_name == "deepseek/test-model"
    assert kwargs["model"].max_retries == 0
    assert kwargs["tools"] == [tool]
    assert kwargs["system_prompt"] == ANY
    assert len(kwargs["middleware"]) == 1
    assert isinstance(kwargs["middleware"][0], deepagent_retrieval._RetrievalOnlyMiddleware)


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
        _create_agent(model, lambda _query: "{}")

    assert calls == []


def test_real_deepagents_factory_exposes_only_the_retrieval_tool(monkeypatch) -> None:
    sdk_model = CaptureChatModel(responses=[AIMessage(content="Coverage is sufficient.")])
    unrelated_model = CaptureChatModel(responses=[AIMessage(content="Unrelated done.")])
    resolved_model = [sdk_model]

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("deepagents.graph.resolve_model", lambda _spec: resolved_model[0])
    result = DeepAgentRetriever(
        retriever=FakeRetriever(),
        model="openrouter:test/deepagent-retrieval-capture",
        tracing=FakeTracing(),
    ).retrieve("narrative")

    assert result.rationale == "Coverage is sufficient."
    assert sdk_model.captured_tool_names == ["search_climbmix"]

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


def test_retrieval_only_middleware_filters_sync_async_models_and_blocks_tools() -> None:
    middleware = deepagent_retrieval._RetrievalOnlyMiddleware()
    request = ModelRequest(
        model=CaptureChatModel(responses=[AIMessage(content="unused")]),
        messages=[],
        tools=[{"name": "search_climbmix"}, {"name": "ls"}],
    )
    seen_sync = []

    def sync_handler(filtered: ModelRequest[object]) -> AIMessage:
        seen_sync.extend(tool["name"] for tool in filtered.tools if isinstance(tool, dict))
        return AIMessage(content="ok")

    assert middleware.wrap_model_call(request, sync_handler).content == "ok"
    assert seen_sync == ["search_climbmix"]

    async def verify_async() -> None:
        seen_async = []

        async def async_handler(filtered: ModelRequest[object]) -> AIMessage:
            seen_async.extend(tool["name"] for tool in filtered.tools if isinstance(tool, dict))
            return AIMessage(content="ok")

        reply = await middleware.awrap_model_call(request, async_handler)
        assert reply.content == "ok"
        assert seen_async == ["search_climbmix"]

        forbidden = ToolCallRequest(
            tool_call={"name": "ls", "args": {}, "id": "call-1"},
            tool=None,
            state={},
            runtime=None,
        )
        with pytest.raises(PermissionError, match="retrieval-only tool access denied"):
            middleware.wrap_tool_call(
                forbidden,
                lambda _request: pytest.fail("forbidden tool handler was called"),
            )
        with pytest.raises(PermissionError, match="retrieval-only tool access denied"):
            await middleware.awrap_tool_call(
                forbidden,
                lambda _request: pytest.fail("forbidden tool handler was called"),
            )

    asyncio.run(verify_async())


def test_tool_returns_bounded_excerpts_but_result_retains_all_candidates() -> None:
    fake_retriever = FakeRetriever(candidate_count=11)
    tool_payloads = []

    def agent_factory(_model: str, search_tool: Callable[[str], str]) -> FakeAgent:
        def invoke(_payload: dict[str, object]) -> object:
            tool_payloads.append(json.loads(search_tool("targeted query")))
            return {"messages": [{"role": "assistant", "content": "Done."}]}

        return FakeAgent(invoke)

    result = _sdk(fake_retriever, agent_factory).retrieve("narrative")

    assert len(tool_payloads[0]["candidates"]) == 10
    assert len(result.searches[1].candidates) == 11
    assert tool_payloads[0]["remaining_budget"] == 2


def test_retrieve_uses_only_the_safe_tracing_attribute() -> None:
    fake_retriever = FakeRetriever()
    tracing = FakeTracing()
    result = _sdk(
        fake_retriever,
        lambda _model, _tool: FakeAgent(
            lambda _payload: {"messages": [{"role": "assistant", "content": "Done."}]}
        ),
        tracing=tracing,
    ).retrieve("narrative")

    assert result.stopping_reason == "agent_completed"
    assert tracing.attribute_keys == ["retrieval.document_count"]
    assert tracing.flushes == 1


def test_from_env_uses_offline_retriever_config_cache_and_model_precedence(
    monkeypatch, tmp_path
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
    assert created[0].config.name == "deepagent_climbmix"
    assert created[0].config.type == "pyserini_remote"
    assert created[0].config.query_variants == ("original", "followup")
    assert created[0].config.hits == 10
    assert created[0].config.index == "climbmix-400b"
    assert created[0].config.cache is True
    assert created[0].cache_dir == tmp_path / "cache" / "retrieval" / "pyserini_remote"
