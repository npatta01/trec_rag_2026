"""Narrative-only Deep Agent retrieval backed by the existing ClimbMix adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from trec_rag.deepagent_tracing import RetrievalTracing, create_retrieval_tracing
from trec_rag.pipeline_config import RetrieverConfig
from trec_rag.pipeline_models import QueryVariant, RankedCandidate, RetrievedCandidate
from trec_rag.repo_env import find_repo_root, load_repo_env, repo_cache_root
from trec_rag.retrievers import PyseriniRemoteRetriever, Retriever


DEFAULT_MODEL = "openrouter:deepseek/deepseek-v4-flash"
MAX_FOLLOWUP_SEARCHES = 3
HITS_PER_SEARCH = 10
FUSED_RESULT_LIMIT = 20
EXCERPT_MAX_CHARACTERS = 1_000

RETRIEVAL_SYSTEM_PROMPT = """You are a retrieval-only research assistant.
The supplied narrative has already been searched exactly as provided. Use the
search_climbmix tool only for targeted follow-up queries that cover a specific
uncovered aspect. Do not use filesystem, shell, task, or other tools for this
retrieval. Stop when targeted follow-ups no longer add coverage, and make your
final message explain why you stopped."""


@dataclass(frozen=True)
class AgentSearch:
    """One completed original or agent-directed ClimbMix search."""

    query: str
    kind: Literal["original", "followup"]
    candidates: tuple[RetrievedCandidate, ...]
    cache_status: str


@dataclass(frozen=True)
class AgentRetrievalResult:
    """Immutable result of a narrative-driven retrieval session."""

    narrative: str
    searches: tuple[AgentSearch, ...]
    candidates: tuple[RankedCandidate, ...]
    rationale: str
    stopping_reason: str


class AgentRetrievalError(RuntimeError):
    """An agent-side failure with the searches completed before it occurred."""

    def __init__(self, message: str, *, searches: Sequence[AgentSearch]) -> None:
        super().__init__(message)
        self.searches = tuple(searches)


class _FrozenProvenance(dict[str, Any]):
    """A dict-shaped provenance record that cannot be mutated after ranking."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        dict.__init__(self, values)

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("ranked candidate provenance is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, _memo: dict[int, object]) -> dict[str, Any]:
        """Keep existing dataclass serialization compatible with plain dict output."""
        return dict(self)


class _RetrievalOnlyMiddleware(AgentMiddleware):
    """Limit one SDK-created agent to the retrieval tool at every boundary."""

    _ALLOWED_TOOL = "search_climbmix"
    _DENIED_MESSAGE = "retrieval-only tool access denied"

    @staticmethod
    def _tool_name(tool: object) -> str | None:
        if isinstance(tool, Mapping):
            name = tool.get("name")
        else:
            name = getattr(tool, "name", None)
        return name if isinstance(name, str) else None

    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        return request.override(
            tools=[
                tool for tool in request.tools if self._tool_name(tool) == self._ALLOWED_TOOL
            ]
        )

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        return handler(self._filter_tools(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Any]
    ) -> Any:
        return await handler(self._filter_tools(request))

    def _require_allowed_tool(self, request: ToolCallRequest) -> None:
        if request.tool_call.get("name") != self._ALLOWED_TOOL:
            raise PermissionError(self._DENIED_MESSAGE)

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        self._require_allowed_tool(request)
        return handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        self._require_allowed_tool(request)
        return await handler(request)


class _Agent(Protocol):
    def invoke(self, input: Mapping[str, object]) -> object: ...


class _SafeSpan(Protocol):
    def set_attribute(self, key: str, value: object) -> None: ...


class _Tracing(Protocol):
    def agent_span(self, narrative: str) -> AbstractContextManager[_SafeSpan]: ...

    def retriever_span(self, query: str) -> AbstractContextManager[_SafeSpan]: ...

    def force_flush(self) -> bool: ...


AgentFactory = Callable[[str, Callable[[str], str]], _Agent]


def reciprocal_rank_fuse(
    searches: Sequence[AgentSearch], *, limit: int = FUSED_RESULT_LIMIT, rrf_k: int = 60
) -> tuple[RankedCandidate, ...]:
    """Fuse searches by document ID using deterministic reciprocal-rank fusion."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    rows: dict[str, dict[str, object]] = {}
    for search in searches:
        for candidate in search.candidates:
            row = rows.setdefault(
                candidate.docid,
                {
                    "score": 0.0,
                    "text": "",
                    "topic_id": candidate.topic_id,
                    "provenance": [],
                },
            )
            row["score"] = float(row["score"]) + 1.0 / (rrf_k + candidate.rank)
            if not row["text"] and candidate.text:
                row["text"] = candidate.text
            provenance = cast(list[dict[str, Any]], row["provenance"])
            provenance.append(
                {
                    "query": search.query,
                    "query_kind": search.kind,
                    "variant_name": candidate.variant_name,
                    "retriever_name": candidate.retriever_name,
                    "source_rank": candidate.rank,
                    "source_score": candidate.score,
                    "cache_status": search.cache_status,
                }
            )

    ordered = sorted(
        rows.items(), key=lambda item: (-float(item[1]["score"]), item[0])
    )[:limit]
    return tuple(
        RankedCandidate(
            topic_id=str(row["topic_id"]),
            docid=docid,
            rank=rank,
            score=float(row["score"]),
            text=str(row["text"]),
            provenance=cast(
                list[dict[str, Any]],
                tuple(_FrozenProvenance(item) for item in row["provenance"]),
            ),
        )
        for rank, (docid, row) in enumerate(ordered, start=1)
    )


def _create_agent(model: str, search_tool: Callable[[str], str]) -> _Agent:
    """Keep the Deep Agents 0.7 construction surface intentionally narrow."""
    from deepagents import create_deep_agent
    from langchain_openrouter import ChatOpenRouter

    prefix = "openrouter:"
    if not isinstance(model, str) or not model.startswith(prefix):
        raise ValueError("model must match openrouter:<model-id>")
    model_id = model[len(prefix) :]
    if not model_id or model_id != model_id.strip():
        raise ValueError("model must match openrouter:<model-id>")
    provider_model = ChatOpenRouter(model=model_id, max_retries=0)

    return cast(
        _Agent,
        create_deep_agent(
            model=provider_model,
            tools=[search_tool],
            system_prompt=RETRIEVAL_SYSTEM_PROMPT,
            middleware=[_RetrievalOnlyMiddleware()],
        ),
    )


def _cache_counts(retriever: Retriever) -> Mapping[str, int] | None:
    summary = getattr(retriever, "cache_summary", None)
    if not callable(summary):
        return None
    reported = summary()
    if not isinstance(reported, Mapping):
        return None
    counts = {}
    for key in ("hits", "misses", "bypasses"):
        value = reported.get(key)
        if isinstance(value, int):
            counts[key] = value
    return counts


def _cache_status(before: Mapping[str, int] | None, after: Mapping[str, int] | None) -> str:
    if before is None or after is None:
        return "not_reported"
    for key, label in (("hits", "hit"), ("misses", "miss"), ("bypasses", "bypass")):
        if after.get(key, 0) > before.get(key, 0):
            return label
    return "not_reported"


def _bounded_candidates(candidates: Sequence[RetrievedCandidate]) -> list[dict[str, object]]:
    return [
        {
            "docid": candidate.docid,
            "rank": candidate.rank,
            "score": candidate.score,
            "excerpt": candidate.text[:EXCERPT_MAX_CHARACTERS],
        }
        for candidate in candidates[:HITS_PER_SEARCH]
    ]


def _assistant_rationale(reply: object) -> str:
    messages: object = reply.get("messages", ()) if isinstance(reply, Mapping) else ()
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ""
    for message in reversed(messages):
        role: object = None
        content: object = None
        if isinstance(message, Mapping):
            role = message.get("role") or message.get("type")
            content = message.get("content")
        else:
            role = getattr(message, "role", None) or getattr(message, "type", None)
            content = getattr(message, "content", None)
        if role not in {"assistant", "ai"}:
            continue
        if isinstance(content, str):
            return content
        if content is not None:
            return str(content)
    return ""


class DeepAgentRetriever:
    """Search a supplied narrative once, then allow bounded agent follow-ups."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        model: str = DEFAULT_MODEL,
        agent_factory: AgentFactory = _create_agent,
        tracing: _Tracing | None = None,
    ) -> None:
        self._retriever = retriever
        self._model = model
        self._agent_factory = agent_factory
        self._tracing = tracing or create_retrieval_tracing()

    @classmethod
    def from_env(
        cls,
        *,
        root: Path | None = None,
        model: str | None = None,
        agent_factory: AgentFactory = _create_agent,
        tracing: _Tracing | None = None,
    ) -> "DeepAgentRetriever":
        """Build the isolated SDK using the existing remote retriever and cache."""
        resolved_root = Path(root) if root is not None else find_repo_root()
        load_repo_env(resolved_root)
        if not os.environ.get("OPENROUTER_API_KEY", "").strip():
            raise ValueError("OPENROUTER_API_KEY is required for Deep Agent retrieval")
        config = RetrieverConfig(
            name="deepagent_climbmix",
            type="pyserini_remote",
            query_variants=("original", "followup"),
            hits=HITS_PER_SEARCH,
            index="climbmix-400b",
            cache=True,
        )
        return cls(
            retriever=PyseriniRemoteRetriever(
                config,
                cache_dir=repo_cache_root(resolved_root) / "retrieval" / "pyserini_remote",
            ),
            model=model or os.environ.get("DEEPAGENT_MODEL") or DEFAULT_MODEL,
            agent_factory=agent_factory,
            tracing=tracing or create_retrieval_tracing(environ=os.environ),
        )

    def retrieve(self, narrative: str) -> AgentRetrievalResult:
        """Retrieve from an untouched narrative without accepting a topic identifier."""
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError("narrative must be non-empty text")

        topic_component = sha256(narrative.encode()).hexdigest()[:16]
        searches: list[AgentSearch] = []
        exhausted = False
        followup_lock = Lock()

        def run_search(query: str, kind: Literal["original", "followup"], variant: str) -> AgentSearch:
            before = _cache_counts(self._retriever)
            with self._tracing.retriever_span(query) as span:
                candidates = tuple(
                    self._retriever.retrieve(
                        QueryVariant(
                            topic_id=topic_component,
                            variant_name=variant,
                            query_text=query,
                            source_type="deepagent_retrieval",
                        )
                    )
                )
                span.set_attribute("retrieval.document_count", len(candidates))
            return AgentSearch(
                query=query,
                kind=kind,
                candidates=candidates,
                cache_status=_cache_status(before, _cache_counts(self._retriever)),
            )

        def search_climbmix(query: str) -> str:
            """Search ClimbMix for a targeted, previously uncovered narrative aspect."""
            nonlocal exhausted
            with followup_lock:
                if not isinstance(query, str) or not query.strip():
                    return json.dumps({"error": "query must be non-empty text"}, sort_keys=True)
                if any(search.query == query for search in searches):
                    return json.dumps({"error": "duplicate follow-up query"}, sort_keys=True)
                if len(searches) - 1 >= MAX_FOLLOWUP_SEARCHES:
                    exhausted = True
                    return json.dumps({"error": "search budget exhausted"}, sort_keys=True)
                search = run_search(
                    query,
                    "followup",
                    "followup-" + sha256(query.encode()).hexdigest()[:16],
                )
                searches.append(search)
                return json.dumps(
                    {
                        "candidates": _bounded_candidates(search.candidates),
                        "remaining_budget": MAX_FOLLOWUP_SEARCHES - (len(searches) - 1),
                    },
                    sort_keys=True,
                )

        try:
            with self._tracing.agent_span(narrative):
                original = run_search(narrative, "original", "original")
                searches.append(original)
                agent = self._agent_factory(self._model, search_climbmix)
                initial_results = json.dumps(
                    _bounded_candidates(original.candidates), sort_keys=True
                )
                reply = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "The untouched narrative is:\n"
                                    f"{narrative}\n\n"
                                    "It has already been searched. Bounded original results:\n"
                                    f"{initial_results}"
                                ),
                            }
                        ]
                    }
                )
                rationale = _assistant_rationale(reply)
                candidates = reciprocal_rank_fuse(searches)
                stopping_reason = "search_budget_exhausted" if exhausted else "agent_completed"
                return AgentRetrievalResult(
                    narrative=narrative,
                    searches=tuple(searches),
                    candidates=candidates,
                    rationale=rationale,
                    stopping_reason=stopping_reason,
                )
        except Exception as exc:
            if searches:
                raise AgentRetrievalError(
                    "Deep Agent retrieval failed", searches=searches
                ) from exc
            raise
        finally:
            self._tracing.force_flush()
