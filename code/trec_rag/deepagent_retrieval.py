"""Narrative-only Deep Agent retrieval backed by the existing ClimbMix adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any, ClassVar, Literal, Protocol, TypeVar, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from trec_rag.deepagent_snippets import (
    InvalidSnippetCursorError,
    RelevantSnippetExtractor,
    SnippetExtractionResult,
    create_default_snippet_extractor,
)
from trec_rag.deepagent_tracing import (
    MAX_TRACE_DOCUMENTS,
    create_retrieval_tracing,
)
from trec_rag.pipeline_config import RetrieverConfig
from trec_rag.pipeline_models import QueryVariant, RankedCandidate, RetrievedCandidate
from trec_rag.repo_env import find_repo_root, load_repo_env, repo_cache_root
from trec_rag.retrievers import PyseriniRemoteRetriever, Retriever


DEFAULT_MODEL = "openrouter:deepseek/deepseek-v4-flash"
MAX_FOLLOWUP_SEARCHES = 3
HITS_PER_SEARCH = 10
FUSED_RESULT_LIMIT = 20

RETRIEVAL_SYSTEM_PROMPT = """You are a retrieval-only research assistant.
The supplied narrative has already been searched exactly as provided. Use the
search_climbmix tool only for targeted follow-up queries that cover a specific
uncovered aspect. Inspect candidate documents through extract_relevant_snippets
with a focused query, and follow next_cursor when another page would add useful
evidence. Use state-file tools only for oversized tool output or temporary notes;
the state filesystem is ephemeral. Do not use execute, task, or unrelated tools.
Stop when targeted follow-ups and document inspection no longer add coverage,
and make your final message explain why you stopped."""


@dataclass(frozen=True)
class AgentSearch:
    """One completed original or agent-directed ClimbMix search."""

    query: str
    kind: Literal["original", "followup"]
    candidates: tuple[RetrievedCandidate, ...]
    cache_status: str


@dataclass(frozen=True)
class AgentRetrievalResult:
    """Immutable retrieval result, including the optional trace flush outcome."""

    narrative: str
    searches: tuple[AgentSearch, ...]
    candidates: tuple[AgentRankedCandidate, ...]
    rationale: str
    stopping_reason: str
    trace_flush_succeeded: bool


class AgentRetrievalError(RuntimeError):
    """An agent-side failure with the searches completed before it occurred."""

    def __init__(self, message: str, *, searches: Sequence[AgentSearch]) -> None:
        super().__init__(message)
        self.searches = tuple(searches)


@dataclass(frozen=True)
class AgentCandidateProvenance(Mapping[str, object]):
    """Explicit immutable provenance for one candidate's source search."""

    query: str
    query_kind: Literal["original", "followup"]
    variant_name: str
    retriever_name: str
    source_rank: int
    source_score: float
    cache_status: str

    _KEYS: ClassVar[tuple[str, ...]] = (
        "query",
        "query_kind",
        "variant_name",
        "retriever_name",
        "source_rank",
        "source_score",
        "cache_status",
    )

    def __getitem__(self, key: str) -> object:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)


@dataclass(frozen=True)
class AgentRankedCandidate(RankedCandidate):
    """A ranked candidate whose Deep Agent provenance is deeply immutable."""

    provenance: tuple[AgentCandidateProvenance, ...]


@dataclass
class _FusionRow:
    score: float
    text: str
    topic_id: str
    provenance: list[AgentCandidateProvenance]


class _RetrievalOnlyMiddleware(AgentMiddleware):
    """Limit one SDK-created agent to retrieval and state scratch tools."""

    _ALLOWED_TOOLS = frozenset(
        {
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
    )
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
                tool
                for tool in request.tools
                if self._tool_name(tool) in self._ALLOWED_TOOLS
            ],
            model_settings={
                **(request.model_settings or {}),
                "parallel_tool_calls": False,
            },
        )

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Any]
    ) -> Any:
        return handler(self._filter_tools(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Any]
    ) -> Any:
        return await handler(self._filter_tools(request))

    def _require_allowed_tool(self, request: ToolCallRequest) -> None:
        if request.tool_call.get("name") not in self._ALLOWED_TOOLS:
            raise PermissionError(self._DENIED_MESSAGE)

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        self._require_allowed_tool(request)
        return handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        self._require_allowed_tool(request)
        return await handler(request)


class _Agent(Protocol):
    def invoke(self, input: Mapping[str, object]) -> object: ...


class _AgentTraceSpan(Protocol):
    def record_result(
        self, *, fused_document_ids: Sequence[str], stopping_reason: str
    ) -> None: ...


class _RetrieverTraceSpan(Protocol):
    def record_search(
        self,
        *,
        document_ids: Sequence[str],
        source_ranks: Sequence[int],
        source_scores: Sequence[float],
        cache_status: str,
        latency_ms: float,
        text_lengths: Sequence[int],
    ) -> None: ...


class _SnippetTraceSpan(Protocol):
    def record_page(
        self,
        *,
        chunk_ids: Sequence[str],
        start_chars: Sequence[int],
        end_chars: Sequence[int],
        relevance_scores: Sequence[float],
        texts: Sequence[str],
        cache_status: str,
        ranker_backend: str,
        latency_ms: float,
        page_offset: int,
        has_next_page: bool,
    ) -> None: ...


class _Tracing(Protocol):
    def agent_span(self, narrative: str) -> AbstractContextManager[_AgentTraceSpan]: ...

    def retriever_span(
        self, query: str
    ) -> AbstractContextManager[_RetrieverTraceSpan]: ...

    def snippet_span(
        self, document_id: str, focus_query: str
    ) -> AbstractContextManager[_SnippetTraceSpan]: ...

    def force_flush(self) -> bool: ...


_TraceSpan = TypeVar("_TraceSpan")


@contextmanager
def _isolated_trace_span(
    factory: Callable[[], AbstractContextManager[_TraceSpan]],
) -> Iterator[_TraceSpan | None]:
    """Keep optional tracing lifecycle failures out of application behavior."""
    try:
        manager = factory()
        span = manager.__enter__()
    except Exception:
        yield None
        return

    try:
        yield span
    except BaseException as exc:
        try:
            manager.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            pass
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception:
            pass


AgentFactory = Callable[
    [str, Callable[[str], str], Callable[[str, str, str | None], str]],
    _Agent,
]


def reciprocal_rank_fuse(
    searches: Sequence[AgentSearch], *, limit: int = FUSED_RESULT_LIMIT, rrf_k: int = 60
) -> tuple[AgentRankedCandidate, ...]:
    """Fuse searches by document ID using deterministic reciprocal-rank fusion."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    rows: dict[str, _FusionRow] = {}
    for search in searches:
        for candidate in search.candidates:
            row = rows.setdefault(
                candidate.docid,
                _FusionRow(
                    score=0.0,
                    text="",
                    topic_id=candidate.topic_id,
                    provenance=[],
                ),
            )
            row.score += 1.0 / (rrf_k + candidate.rank)
            if not row.text and candidate.text:
                row.text = candidate.text
            row.provenance.append(
                AgentCandidateProvenance(
                    query=search.query,
                    query_kind=search.kind,
                    variant_name=candidate.variant_name,
                    retriever_name=candidate.retriever_name,
                    source_rank=candidate.rank,
                    source_score=candidate.score,
                    cache_status=search.cache_status,
                )
            )

    ordered = sorted(rows.items(), key=lambda item: (-item[1].score, item[0]))[:limit]
    return tuple(
        AgentRankedCandidate(
            topic_id=row.topic_id,
            docid=docid,
            rank=rank,
            score=row.score,
            text=row.text,
            provenance=tuple(row.provenance),
        )
        for rank, (docid, row) in enumerate(ordered, start=1)
    )


def _create_agent(
    model: str,
    search_tool: Callable[[str], str],
    snippet_tool: Callable[[str, str, str | None], str],
) -> _Agent:
    """Keep the Deep Agents 0.7 construction surface intentionally narrow."""
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
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
            tools=[search_tool, snippet_tool],
            system_prompt=RETRIEVAL_SYSTEM_PROMPT,
            middleware=[_RetrievalOnlyMiddleware()],
            backend=StateBackend(),
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


def _cache_status(
    before: Mapping[str, int] | None, after: Mapping[str, int] | None
) -> str:
    if before is None or after is None:
        return "not_reported"
    for key, label in (("hits", "hit"), ("misses", "miss"), ("bypasses", "bypass")):
        if after.get(key, 0) > before.get(key, 0):
            return label
    return "not_reported"


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _candidate_metadata(
    candidates: Sequence[RetrievedCandidate], *, limit: int
) -> list[dict[str, object]]:
    return [
        {
            "docid": candidate.docid,
            "rank": candidate.rank,
            "score": candidate.score,
            "text_length": len(candidate.text),
        }
        for candidate in candidates[:limit]
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
        snippet_extractor: RelevantSnippetExtractor | None = None,
        hits_per_search: int = HITS_PER_SEARCH,
        max_followup_searches: int = MAX_FOLLOWUP_SEARCHES,
        fused_result_limit: int = FUSED_RESULT_LIMIT,
    ) -> None:
        self._hits_per_search = _positive_int(hits_per_search, name="hits_per_search")
        self._max_followup_searches = _positive_int(
            max_followup_searches, name="max_followup_searches"
        )
        self._fused_result_limit = _positive_int(
            fused_result_limit, name="fused_result_limit"
        )
        self._retriever = retriever
        self._model = model
        self._agent_factory = agent_factory
        self._tracing = tracing or create_retrieval_tracing()
        self._snippet_extractor = snippet_extractor
        self._snippet_extractor_lock = Lock()

    @classmethod
    def from_env(
        cls,
        *,
        root: Path | None = None,
        model: str | None = None,
        agent_factory: AgentFactory = _create_agent,
        tracing: _Tracing | None = None,
        snippet_extractor: RelevantSnippetExtractor | None = None,
        hits_per_search: int = HITS_PER_SEARCH,
        max_followup_searches: int = MAX_FOLLOWUP_SEARCHES,
        fused_result_limit: int = FUSED_RESULT_LIMIT,
    ) -> "DeepAgentRetriever":
        """Build the isolated SDK using the existing remote retriever and cache."""
        validated_hits = _positive_int(hits_per_search, name="hits_per_search")
        validated_followups = _positive_int(
            max_followup_searches, name="max_followup_searches"
        )
        validated_fused_limit = _positive_int(
            fused_result_limit, name="fused_result_limit"
        )
        resolved_root = Path(root) if root is not None else find_repo_root()
        load_repo_env(resolved_root)
        if not os.environ.get("OPENROUTER_API_KEY", "").strip():
            raise ValueError("OPENROUTER_API_KEY is required for Deep Agent retrieval")
        config = RetrieverConfig(
            name="deepagent_climbmix",
            type="pyserini_remote",
            query_variants=("original", "followup"),
            hits=validated_hits,
            index="climbmix-400b",
            cache=True,
        )
        return cls(
            retriever=PyseriniRemoteRetriever(
                config,
                cache_dir=repo_cache_root(resolved_root)
                / "retrieval"
                / "pyserini_remote",
            ),
            model=model or os.environ.get("DEEPAGENT_MODEL") or DEFAULT_MODEL,
            agent_factory=agent_factory,
            tracing=tracing or create_retrieval_tracing(environ=os.environ),
            snippet_extractor=(
                snippet_extractor
                if snippet_extractor is not None
                else create_default_snippet_extractor(resolved_root)
            ),
            hits_per_search=validated_hits,
            max_followup_searches=validated_followups,
            fused_result_limit=validated_fused_limit,
        )

    def retrieve(self, narrative: str) -> AgentRetrievalResult:
        """Retrieve from an untouched narrative without accepting a topic identifier."""
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError("narrative must be non-empty text")

        topic_component = sha256(narrative.encode()).hexdigest()[:16]
        searches: list[AgentSearch] = []
        documents: dict[str, str] = {}
        exhausted = False
        followup_lock = Lock()
        document_lock = Lock()

        def register_documents(candidates: Sequence[RetrievedCandidate]) -> None:
            with document_lock:
                for candidate in candidates:
                    if candidate.docid not in documents:
                        documents[candidate.docid] = candidate.text
                        continue
                    existing = documents[candidate.docid]
                    if existing and existing != candidate.text:
                        raise ValueError(
                            f"conflicting text for document_id {candidate.docid}"
                        )
                    if not existing and candidate.text:
                        documents[candidate.docid] = candidate.text

        def run_search(
            query: str, kind: Literal["original", "followup"], variant: str
        ) -> AgentSearch:
            before = _cache_counts(self._retriever)
            with _isolated_trace_span(
                lambda: self._tracing.retriever_span(query)
            ) as span:
                started_at = monotonic()
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
                latency_ms = (monotonic() - started_at) * 1_000
                cache_status = _cache_status(before, _cache_counts(self._retriever))
                trace_candidates = candidates[
                    : min(self._hits_per_search, MAX_TRACE_DOCUMENTS)
                ]
                if span is not None:
                    try:
                        span.record_search(
                            document_ids=tuple(
                                candidate.docid for candidate in trace_candidates
                            ),
                            source_ranks=tuple(
                                candidate.rank for candidate in trace_candidates
                            ),
                            source_scores=tuple(
                                candidate.score for candidate in trace_candidates
                            ),
                            cache_status=cache_status,
                            latency_ms=latency_ms,
                            text_lengths=tuple(
                                len(candidate.text) for candidate in trace_candidates
                            ),
                        )
                    except Exception:
                        pass
            register_documents(candidates)
            return AgentSearch(
                query=query,
                kind=kind,
                candidates=candidates,
                cache_status=cache_status,
            )

        def search_climbmix(query: str) -> str:
            """Search ClimbMix for a targeted, previously uncovered narrative aspect."""
            nonlocal exhausted
            with followup_lock:
                if not isinstance(query, str) or not query.strip():
                    return json.dumps(
                        {"error": "query must be non-empty text"}, sort_keys=True
                    )
                if any(search.query == query for search in searches):
                    return json.dumps(
                        {"error": "duplicate follow-up query"}, sort_keys=True
                    )
                if len(searches) - 1 >= self._max_followup_searches:
                    exhausted = True
                    return json.dumps(
                        {"error": "search budget exhausted"}, sort_keys=True
                    )
                search = run_search(
                    query,
                    "followup",
                    "followup-" + sha256(query.encode()).hexdigest()[:16],
                )
                searches.append(search)
                return json.dumps(
                    {
                        "documents": _candidate_metadata(
                            search.candidates, limit=self._hits_per_search
                        ),
                        "remaining_budget": self._max_followup_searches
                        - (len(searches) - 1),
                    },
                    sort_keys=True,
                )

        def extract_relevant_snippets(
            document_id: str,
            focus_query: str,
            cursor: str | None = None,
        ) -> str:
            """Return one relevance-ranked snippet page for a retrieved document."""
            if not isinstance(document_id, str) or not document_id.strip():
                return json.dumps({"error": "unknown document_id"}, sort_keys=True)
            if not isinstance(focus_query, str) or not focus_query.strip():
                return json.dumps(
                    {"error": "focus_query must be non-empty text"}, sort_keys=True
                )
            if cursor is not None and (
                not isinstance(cursor, str) or not cursor.strip()
            ):
                return json.dumps({"error": "invalid cursor"}, sort_keys=True)
            with document_lock:
                document_text = documents.get(document_id)
            if document_text is None:
                return json.dumps({"error": "unknown document_id"}, sort_keys=True)
            try:
                with _isolated_trace_span(
                    lambda: self._tracing.snippet_span(document_id, focus_query)
                ) as span:
                    with self._snippet_extractor_lock:
                        if self._snippet_extractor is None:
                            self._snippet_extractor = create_default_snippet_extractor(
                                find_repo_root()
                            )
                        extractor = self._snippet_extractor
                    started_at = monotonic()
                    result: SnippetExtractionResult = extractor.extract(
                        document_id,
                        document_text,
                        focus_query,
                        cursor,
                    )
                    latency_ms = (monotonic() - started_at) * 1_000
                    if span is not None:
                        try:
                            span.record_page(
                                chunk_ids=tuple(
                                    snippet.chunk_id
                                    for snippet in result.page.snippets
                                ),
                                start_chars=tuple(
                                    snippet.start_char
                                    for snippet in result.page.snippets
                                ),
                                end_chars=tuple(
                                    snippet.end_char for snippet in result.page.snippets
                                ),
                                relevance_scores=tuple(
                                    snippet.relevance_score
                                    for snippet in result.page.snippets
                                ),
                                texts=tuple(
                                    snippet.text for snippet in result.page.snippets
                                ),
                                cache_status=result.cache_status,
                                ranker_backend=result.ranker_backend,
                                latency_ms=latency_ms,
                                page_offset=result.page_offset,
                                has_next_page=result.page.next_cursor is not None,
                            )
                        except Exception:
                            pass
            except InvalidSnippetCursorError:
                return json.dumps({"error": "invalid cursor"}, sort_keys=True)
            except Exception:
                return json.dumps(
                    {"error": "snippet extraction failed"}, sort_keys=True
                )
            return json.dumps(result.page.as_dict(), sort_keys=True)

        try:
            with _isolated_trace_span(
                lambda: self._tracing.agent_span(narrative)
            ) as agent_span:
                original = run_search(narrative, "original", "original")
                searches.append(original)
                agent = self._agent_factory(
                    self._model,
                    search_climbmix,
                    extract_relevant_snippets,
                )
                initial_results = json.dumps(
                    _candidate_metadata(
                        original.candidates, limit=self._hits_per_search
                    ),
                    sort_keys=True,
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
                candidates = reciprocal_rank_fuse(
                    searches, limit=self._fused_result_limit
                )
                stopping_reason = (
                    "search_budget_exhausted" if exhausted else "agent_completed"
                )
                if agent_span is not None:
                    try:
                        agent_span.record_result(
                            fused_document_ids=tuple(
                                candidate.docid
                                for candidate in candidates[:MAX_TRACE_DOCUMENTS]
                            ),
                            stopping_reason=stopping_reason,
                        )
                    except Exception:
                        pass
        except Exception as exc:
            if searches:
                raise AgentRetrievalError(
                    "Deep Agent retrieval failed", searches=searches
                ) from exc
            raise
        finally:
            try:
                trace_flush_succeeded = bool(self._tracing.force_flush())
            except Exception:
                trace_flush_succeeded = False

        return AgentRetrievalResult(
            narrative=narrative,
            searches=tuple(searches),
            candidates=candidates,
            rationale=rationale,
            stopping_reason=stopping_reason,
            trace_flush_succeeded=trace_flush_succeeded,
        )
