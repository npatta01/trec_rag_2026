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

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from trec_rag.deepagent_budget import (
    BudgetDecision,
    BudgetSnapshot,
    ResearchBudget,
    ResearchBudgetConfig,
    ResearchTaskContext,
)
from trec_rag.deepagent_evidence import (
    ActionKind,
    DocumentObservation,
    EvidenceCoverageReport,
    EvidenceCoverageState,
    RetrievalStateDelta,
)
from trec_rag.deepagent_research import (
    MainToolFilterMiddleware,
    ResearchTaskBudgetMiddleware,
    build_research_subagent,
    current_research_task,
)
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
OPENROUTER_REQUEST_TIMEOUT_MS = 120_000
MAX_FOLLOWUP_SEARCHES = 8
HITS_PER_SEARCH = 10
FUSED_RESULT_LIMIT = 20

RETRIEVAL_SYSTEM_PROMPT = """You are the main retrieval research coordinator.
The supplied narrative has already been searched exactly as provided.
First decompose the untouched narrative into explicit needs and record them
with update_retrieval_state under delta.add_needs, using the exact outer shape
{"delta":{"add_needs":[{"need_id":"N1","narrative_span":"<exact text copied from the untouched narrative>","question":"<question derived from that span>"}]}}.
Preserve each need's exact narrative span.
Delegate retrieval only through `task` using compact ResearchTaskEnvelope JSON
and subagent_type "researcher". Batch independent task calls in parallel. After
each batch, merge the returned evidence bundles with exactly one batched
update_retrieval_state semantic delta, then call complete_research_round once.
Every round must finish at least one researcher before it can close. If round
completion returns ROUND_RESEARCH_REQUIRED, the next action must delegate a
researcher for that same round. If it returns ROUND_SEQUENCE_INVALID, retry once
with round_index equal to budget_snapshot.completed_rounds + 1.
Closing a round is not the end of the run. While needs remain unresolved and no
budget response says must_stop, open the next round with round_index incremented
by one and delegate a new task batch aimed at those specific gaps. Give an empty
bundle's need a materially different angle rather than repeating the goal that
already returned nothing.
Set task.description to a JSON-encoded object with exactly this shape:
{"research_task_id":"R1-N1","round_index":1,"depth":"focused","motivating_ids":["N1"],"goal":"Find grounded evidence for N1","known_evidence":"","remaining_gap":"No grounded evidence yet"}.
The JSON object must come first. Optional research instructions may follow it
after a newline; never put prose before the JSON object.
Do not call retrieval tools directly. Stop whenever a budget response says
must_stop. After soft_deadline_reached becomes true, do not launch a new survey
task; focused or deep tasks may still close a specific gap, and already-running
tasks may finish. Merge completed bundles and finalize even after the soft
deadline.
Mark a need answerable only with a draft answer and grounded nugget IDs. Report
conflicts and unresolved gaps. Caches remain tool-owned; use state scratch only
for oversized output or temporary notes. The state filesystem is ephemeral.
Only read_file is available from it."""

_VALID_DELTA_SECTIONS = (
    "add_needs",
    "add_facets",
    "add_nuggets",
    "add_evidence",
    "set_facet_status",
    "set_need_status",
    "supersede_nuggets",
    "abandon_documents",
)


def _delta_error_guidance() -> dict[str, object]:
    return {
        "valid_delta_sections": list(_VALID_DELTA_SECTIONS),
        "minimal_add_needs_example": {
            "delta": {
                "add_needs": [
                    {
                        "need_id": "N1",
                        "narrative_span": (
                            "<exact text copied from the untouched narrative>"
                        ),
                        "question": "<question derived from that span>",
                    }
                ]
            }
        },
    }


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
    coverage_report: EvidenceCoverageReport
    budget_snapshot: BudgetSnapshot
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
        self,
        *,
        fused_document_ids: Sequence[str],
        stopping_reason: str,
        coverage_state_hash: str,
        need_count: int,
        answerable_need_count: int,
        conflicted_need_count: int,
        unresolved_need_count: int,
        nugget_count: int,
        action_count: int,
        researcher_invocation_count: int,
        research_round_count: int,
        retrieval_call_count: int,
        budget_stop_code: str | None,
    ) -> None: ...


class _RetrieverTraceSpan(Protocol):
    def record_research_context(
        self,
        *,
        research_task_id: str,
        round_index: int,
        depth: str,
        code: str,
        must_stop: bool,
        snapshot: Mapping[str, object],
    ) -> None: ...

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
    def record_research_context(
        self,
        *,
        research_task_id: str,
        round_index: int,
        depth: str,
        code: str,
        must_stop: bool,
        snapshot: Mapping[str, object],
    ) -> None: ...

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
        page_index: int,
        residual_count: int,
        residual_top_score: float | None,
        returned_min_score: float | None,
        pages_estimated: int,
    ) -> None: ...


class _Tracing(Protocol):
    def agent_span(self, narrative: str) -> AbstractContextManager[_AgentTraceSpan]: ...

    def retriever_span(
        self, query: str
    ) -> AbstractContextManager[_RetrieverTraceSpan]: ...

    def snippet_span(
        self, document_id: str, focus_query: str
    ) -> AbstractContextManager[_SnippetTraceSpan]: ...

    def researcher_task_span(
        self, research_task_id: str, round_index: int, depth: str
    ) -> AbstractContextManager[object]: ...

    def researcher_dispatch_span(self) -> AbstractContextManager[object]: ...

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


@dataclass(frozen=True)
class AgentToolset:
    search_climbmix: Callable[[str, list[str], str], str]
    extract_relevant_snippets: Callable[
        [str, str, list[str], str, str | None], str
    ]
    view_retrieval_state: Callable[[str], str]
    update_retrieval_state: Callable[[RetrievalStateDelta], str]
    complete_research_round: Callable[[int], str]
    choose_next_action: Callable[[str, str, str | None, list[str], str], str]
    budget: ResearchBudget
    budget_config: ResearchBudgetConfig
    tracing: _Tracing | None = None


AgentFactory = Callable[[str, AgentToolset], _Agent]


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
    toolset: AgentToolset,
) -> _Agent:
    """Keep the Deep Agents 0.7 construction surface intentionally narrow."""
    from deepagents import (
        GeneralPurposeSubagentProfile,
        HarnessProfile,
        create_deep_agent,
        register_harness_profile,
    )
    from deepagents.backends import StateBackend
    from langchain_openrouter import ChatOpenRouter
    from openrouter import OpenRouter
    from openrouter.utils import BackoffStrategy, RetryConfig

    prefix = "openrouter:"
    if not isinstance(model, str) or not model.startswith(prefix):
        raise ValueError("model must match openrouter:<model-id>")
    model_id = model[len(prefix) :]
    if not model_id or model_id != model_id.strip():
        raise ValueError("model must match openrouter:<model-id>")
    no_retries = RetryConfig(
        strategy="none",
        backoff=BackoffStrategy(
            initial_interval=500,
            max_interval=60_000,
            exponent=1.5,
            max_elapsed_time=0,
            jitter_ms=0,
        ),
        retry_connection_errors=False,
    )
    sdk_client = OpenRouter(
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout_ms=OPENROUTER_REQUEST_TIMEOUT_MS,
        retry_config=no_retries,
    )
    provider_model = ChatOpenRouter(
        model=model_id,
        client=sdk_client,
        max_retries=0,
        timeout=OPENROUTER_REQUEST_TIMEOUT_MS,
    )
    register_harness_profile(
        model,
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
        ),
    )
    researcher = build_research_subagent(
        model=provider_model,
        tools=[
            toolset.search_climbmix,
            toolset.extract_relevant_snippets,
            toolset.view_retrieval_state,
        ],
        budget=toolset.budget,
        budget_config=toolset.budget_config,
    )

    return cast(
        _Agent,
        create_deep_agent(
            model=provider_model,
            tools=[
                toolset.view_retrieval_state,
                toolset.update_retrieval_state,
                toolset.complete_research_round,
            ],
            system_prompt=RETRIEVAL_SYSTEM_PROMPT,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=toolset.budget_config.max_main_models,
                    exit_behavior="end",
                ),
                ToolCallLimitMiddleware(
                    tool_name="task",
                    run_limit=toolset.budget_config.max_researcher_invocations,
                    exit_behavior="continue",
                ),
                ResearchTaskBudgetMiddleware(toolset.budget, tracing=toolset.tracing),
                MainToolFilterMiddleware(toolset.budget, toolset.budget_config),
            ],
            subagents=[researcher],
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


def _validated_action_kind(action: object) -> ActionKind | None:
    if not isinstance(action, str) or action not in {
        "search",
        "extract",
        "paginate",
        "refocus",
        "stop",
    }:
        return None
    return cast(ActionKind, action)


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
        budget_config: ResearchBudgetConfig | None = None,
    ) -> None:
        self._hits_per_search = _positive_int(hits_per_search, name="hits_per_search")
        self._max_followup_searches = _positive_int(
            max_followup_searches, name="max_followup_searches"
        )
        self._fused_result_limit = _positive_int(
            fused_result_limit, name="fused_result_limit"
        )
        self._budget_config = budget_config or ResearchBudgetConfig(
            max_searches_per_researcher=self._max_followup_searches
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
        budget_config: ResearchBudgetConfig | None = None,
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
            budget_config=budget_config,
        )

    def retrieve(self, narrative: str) -> AgentRetrievalResult:
        """Retrieve from an untouched narrative without accepting a topic identifier."""
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError("narrative must be non-empty text")

        topic_component = sha256(narrative.encode()).hexdigest()[:16]
        coverage_state = EvidenceCoverageState(narrative)
        budget = ResearchBudget(self._budget_config)
        searches: list[AgentSearch] = []
        documents: dict[str, str] = {}
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
            query: str,
            kind: Literal["original", "followup"],
            variant: str,
            *,
            context: ResearchTaskContext | None = None,
            decision: BudgetDecision | None = None,
        ) -> AgentSearch:
            before = _cache_counts(self._retriever)
            with _isolated_trace_span(
                lambda: self._tracing.retriever_span(query)
            ) as span:
                if span is not None and context is not None and decision is not None:
                    try:
                        span.record_research_context(
                            research_task_id=context.research_task_id,
                            round_index=context.round_index,
                            depth=context.depth,
                            code=decision.code,
                            must_stop=decision.must_stop,
                            snapshot=decision.snapshot.as_dict(),
                        )
                    except Exception:
                        pass
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
            coverage_state.record_search(
                query=query,
                kind=kind,
                documents=tuple(
                    DocumentObservation(candidate.docid, candidate.rank)
                    for candidate in candidates
                ),
            )
            return AgentSearch(
                query=query,
                kind=kind,
                candidates=candidates,
                cache_status=cache_status,
            )

        def task_context() -> ResearchTaskContext | None:
            envelope = current_research_task()
            if envelope is None:
                return None
            return ResearchTaskContext(
                envelope.research_task_id,
                envelope.round_index,
                envelope.depth,
                tuple(envelope.motivating_ids),
            )

        def budget_payload(
            decision: BudgetDecision,
            *,
            error: str | None = None,
            code: str | None = None,
        ) -> dict[str, object]:
            snapshot = decision.snapshot
            payload: dict[str, object] = {
                "ok": decision.ok,
                "code": code or decision.code,
                "must_stop": decision.must_stop or task_must_stop(snapshot),
                "budget_snapshot": snapshot.as_dict(),
            }
            if error is not None:
                payload["error"] = error
            return payload

        def record_no_yield(context: ResearchTaskContext) -> None:
            budget.record_yield(context, ())

        def task_must_stop(snapshot: BudgetSnapshot) -> bool:
            context = task_context()
            return snapshot.stop_code is not None or (
                context is not None and budget.task_stop_code(context) is not None
            )

        def search_climbmix(
            query: str,
            motivating_ids: list[str],
            rationale: str,
        ) -> str:
            """Search ClimbMix for a targeted, previously uncovered narrative aspect."""
            context = task_context()
            if context is None:
                return json.dumps(
                    {
                        "ok": False,
                        "code": "RESEARCH_TASK_REQUIRED",
                        "must_stop": True,
                        "budget_snapshot": budget.snapshot().as_dict(),
                        "error": "research task context required",
                    },
                    sort_keys=True,
                )
            decision = budget.reserve_retrieval(context, "search_climbmix")
            if not decision.ok:
                return json.dumps(
                    budget_payload(decision, error="retrieval budget refused"),
                    sort_keys=True,
                )
            with followup_lock:
                if not isinstance(query, str) or not query.strip():
                    record_no_yield(context)
                    return json.dumps(
                        budget_payload(
                            BudgetDecision(
                                False, decision.code, budget.snapshot(), decision.must_stop
                            ),
                            error="query must be non-empty text",
                            code="INVALID_QUERY",
                        ),
                        sort_keys=True,
                    )
                action_error = coverage_state.record_retrieval_action(
                    action="search",
                    target=query,
                    focus_query=None,
                    motivating_ids=motivating_ids,
                    rationale=rationale,
                    context=context,
                )
                if action_error is not None:
                    record_no_yield(context)
                    return json.dumps(
                        budget_payload(
                            BudgetDecision(
                                False, decision.code, budget.snapshot(), decision.must_stop
                            ),
                            error="retrieval action rejected",
                            code=action_error,
                        ),
                        sort_keys=True,
                    )
                if any(search.query == query for search in searches):
                    record_no_yield(context)
                    return json.dumps(
                        budget_payload(
                            BudgetDecision(
                                False, decision.code, budget.snapshot(), decision.must_stop
                            ),
                            error="duplicate follow-up query",
                            code="DUPLICATE_QUERY",
                        ),
                        sort_keys=True,
                    )
                search = run_search(
                    query,
                    "followup",
                    "followup-" + sha256(query.encode()).hexdigest()[:16],
                    context=context,
                    decision=decision,
                )
                searches.append(search)
                budget.record_yield(
                    context, (candidate.docid for candidate in search.candidates)
                )
                snapshot = budget.snapshot()
                return json.dumps(
                    {
                        "ok": True,
                        "code": decision.code,
                        "must_stop": decision.must_stop or task_must_stop(snapshot),
                        "budget_snapshot": snapshot.as_dict(),
                        "documents": _candidate_metadata(
                            search.candidates, limit=self._hits_per_search
                        ),
                        "remaining_budget": snapshot.remaining_retrieval_calls,
                    },
                    sort_keys=True,
                )

        def extract_relevant_snippets(
            document_id: str,
            focus_query: str,
            motivating_ids: list[str],
            rationale: str,
            cursor: str | None = None,
        ) -> str:
            """Return one relevance-ranked snippet page for a retrieved document."""
            context = task_context()
            if context is None:
                return json.dumps(
                    {
                        "ok": False,
                        "code": "RESEARCH_TASK_REQUIRED",
                        "must_stop": True,
                        "budget_snapshot": budget.snapshot().as_dict(),
                        "error": "research task context required",
                    },
                    sort_keys=True,
                )
            decision = budget.reserve_retrieval(
                context, "extract_relevant_snippets"
            )
            if not decision.ok:
                return json.dumps(
                    budget_payload(decision, error="retrieval budget refused"),
                    sort_keys=True,
                )
            if not isinstance(document_id, str) or not document_id.strip():
                record_no_yield(context)
                return json.dumps(
                    budget_payload(
                        BudgetDecision(False, decision.code, budget.snapshot()),
                        error="unknown document_id",
                        code="INVALID_DOCUMENT",
                    ),
                    sort_keys=True,
                )
            if not isinstance(focus_query, str) or not focus_query.strip():
                record_no_yield(context)
                return json.dumps(
                    budget_payload(
                        BudgetDecision(False, decision.code, budget.snapshot()),
                        error="focus_query must be non-empty text",
                        code="INVALID_FOCUS_QUERY",
                    ),
                    sort_keys=True,
                )
            if cursor is not None and (
                not isinstance(cursor, str) or not cursor.strip()
            ):
                record_no_yield(context)
                return json.dumps(
                    budget_payload(
                        BudgetDecision(False, decision.code, budget.snapshot()),
                        error="invalid cursor",
                        code="INVALID_CURSOR",
                    ),
                    sort_keys=True,
                )
            action = coverage_state.expected_snippet_action(
                document_id=document_id,
                focus_query=focus_query,
                cursor=cursor,
            )
            action_kind = "paginate" if cursor is not None else action or "extract"
            action_error = coverage_state.record_retrieval_action(
                action=action_kind,
                target=document_id,
                focus_query=focus_query,
                motivating_ids=motivating_ids,
                rationale=rationale,
                context=context,
            )
            if action_error is not None:
                record_no_yield(context)
                return json.dumps(
                    budget_payload(
                        BudgetDecision(False, decision.code, budget.snapshot()),
                        error="retrieval action rejected",
                        code=action_error,
                    ),
                    sort_keys=True,
                )
            with document_lock:
                document_text = documents.get(document_id)
            if document_text is None:
                record_no_yield(context)
                return json.dumps(
                    budget_payload(
                        BudgetDecision(False, decision.code, budget.snapshot()),
                        error="unknown document_id",
                        code="UNKNOWN_DOCUMENT",
                    ),
                    sort_keys=True,
                )
            if cursor is not None and action is None:
                record_no_yield(context)
                return json.dumps(
                    budget_payload(
                        BudgetDecision(False, decision.code, budget.snapshot()),
                        error="pagination requires prior snippet page",
                        code="INVALID_CURSOR",
                    ),
                    sort_keys=True,
                )
            try:
                with _isolated_trace_span(
                    lambda: self._tracing.snippet_span(document_id, focus_query)
                ) as span:
                    if span is not None:
                        try:
                            span.record_research_context(
                                research_task_id=context.research_task_id,
                                round_index=context.round_index,
                                depth=context.depth,
                                code=decision.code,
                                must_stop=decision.must_stop,
                                snapshot=decision.snapshot.as_dict(),
                            )
                        except Exception:
                            pass
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
                    snapshot = budget.snapshot()
                    if snapshot.hard_deadline_reached:
                        return json.dumps(
                            budget_payload(
                                BudgetDecision(
                                    ok=False,
                                    code="HARD_DEADLINE_REACHED",
                                    snapshot=snapshot,
                                    must_stop=True,
                                ),
                                error="retrieval budget refused",
                            ),
                            sort_keys=True,
                        )
                    coverage_state.record_snippet_page(result.page)
                    latency_ms = (monotonic() - started_at) * 1_000
                    if span is not None:
                        try:
                            span.record_page(
                                chunk_ids=tuple(
                                    snippet.chunk_id for snippet in result.page.snippets
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
                                page_index=result.page.page_index,
                                residual_count=result.page.residual_count,
                                residual_top_score=result.page.residual_top_score,
                                returned_min_score=result.page.returned_min_score,
                                pages_estimated=result.page.pages_estimated,
                            )
                        except Exception:
                            pass
            except InvalidSnippetCursorError:
                record_no_yield(context)
                return json.dumps(
                    budget_payload(
                        BudgetDecision(False, decision.code, budget.snapshot()),
                        error="invalid cursor",
                        code="INVALID_CURSOR",
                    ),
                    sort_keys=True,
                )
            except Exception:
                record_no_yield(context)
                return json.dumps(
                    budget_payload(
                        BudgetDecision(False, decision.code, budget.snapshot()),
                        error="snippet extraction failed",
                        code="SNIPPET_EXTRACTION_FAILED",
                    ),
                    sort_keys=True,
                )
            budget.record_yield(
                context, (snippet.chunk_id for snippet in result.page.snippets)
            )
            snapshot = budget.snapshot()
            payload = result.page.as_dict()
            payload.update(
                {
                    "ok": True,
                    "code": decision.code,
                    "must_stop": decision.must_stop or task_must_stop(snapshot),
                    "budget_snapshot": snapshot.as_dict(),
                }
            )
            return json.dumps(payload, sort_keys=True)

        def view_retrieval_state(scope: str = "frontier") -> str:
            """View compact invocation-local needs, gaps, evidence, or document state."""
            if not isinstance(scope, str):
                return json.dumps({"error": "invalid state scope"}, sort_keys=True)
            return coverage_state.view(scope)

        def update_retrieval_state(delta: RetrievalStateDelta) -> str:
            """Update retrieval state with this exact call shape:
            {"delta":{"add_needs":[{"need_id":"N1","narrative_span":"<exact text copied from the untouched narrative>","question":"<question derived from that span>"}]}}
            Valid delta section keys are exactly: add_needs, add_facets,
            add_nuggets, add_evidence, set_facet_status, set_need_status,
            supersede_nuggets, abandon_documents.
            Do not use "needs" or IDs such as "N1" as keys inside delta.
            """
            update = coverage_state.apply_delta(delta)
            payload = update.as_dict()
            if any(
                rejection.code in {"UNKNOWN_SECTION", "EMPTY_DELTA"}
                for rejection in update.rejected
            ):
                payload.update(_delta_error_guidance())
            return json.dumps(payload, sort_keys=True)

        def complete_research_round(round_index: int) -> str:
            """Close one coordinator round after applying its batched semantic delta."""
            decision = budget.authorize_round_completion(round_index)
            if decision.ok:
                decision = budget.complete_round(round_index, coverage_state.report())
            return json.dumps(
                {
                    "ok": decision.ok,
                    "code": decision.code,
                    "must_stop": decision.must_stop,
                    "budget_snapshot": decision.snapshot.as_dict(),
                },
                sort_keys=True,
            )

        def choose_next_action(
            action: str,
            target: str,
            focus_query: str | None,
            motivating_ids: list[str],
            rationale: str,
        ) -> str:
            """Record the coverage gap motivating the next retrieval action or stop."""
            validated_action = _validated_action_kind(action)
            invalid_focus = focus_query is not None and (
                not isinstance(focus_query, str) or not focus_query.strip()
            )
            if validated_action != "stop" or invalid_focus:
                return json.dumps(
                    {"ok": False, "code": "INVALID_ACTION"}, sort_keys=True
                )
            return json.dumps(
                coverage_state.choose_action(
                    action=validated_action,
                    target=target,
                    focus_query=focus_query,
                    motivating_ids=motivating_ids,
                    rationale=rationale,
                ),
                sort_keys=True,
            )

        try:
            with _isolated_trace_span(
                lambda: self._tracing.agent_span(narrative)
            ) as agent_span:
                original = run_search(narrative, "original", "original")
                searches.append(original)
                agent = self._agent_factory(
                    self._model,
                    AgentToolset(
                        search_climbmix=search_climbmix,
                        extract_relevant_snippets=extract_relevant_snippets,
                        view_retrieval_state=view_retrieval_state,
                        update_retrieval_state=update_retrieval_state,
                        complete_research_round=complete_research_round,
                        choose_next_action=choose_next_action,
                        budget=budget,
                        budget_config=self._budget_config,
                        tracing=self._tracing,
                    ),
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
                coverage_report = coverage_state.report()
                budget_snapshot = budget.snapshot()
                stopping_reason = (
                    coverage_report.terminal_reason
                    or (
                        "budget_exhausted"
                        if budget_snapshot.stop_code is not None
                        else None
                    )
                    or "agent_completed"
                )
                if agent_span is not None:
                    try:
                        agent_span.record_result(
                            fused_document_ids=tuple(
                                candidate.docid
                                for candidate in candidates[:MAX_TRACE_DOCUMENTS]
                            ),
                            stopping_reason=stopping_reason,
                            coverage_state_hash=coverage_report.state_hash,
                            need_count=len(coverage_report.needs),
                            answerable_need_count=sum(
                                need.status == "answerable"
                                for need in coverage_report.needs
                            ),
                            conflicted_need_count=sum(
                                need.status == "conflicted"
                                for need in coverage_report.needs
                            ),
                            unresolved_need_count=len(
                                coverage_report.unresolved_need_ids
                            ),
                            nugget_count=len(coverage_report.nuggets),
                            action_count=len(coverage_report.actions),
                            researcher_invocation_count=(
                                self._budget_config.max_researcher_invocations
                                - budget_snapshot.remaining_researchers
                            ),
                            research_round_count=budget_snapshot.completed_rounds,
                            retrieval_call_count=(
                                self._budget_config.max_retrieval_calls
                                - budget_snapshot.remaining_retrieval_calls
                            ),
                            budget_stop_code=budget_snapshot.stop_code,
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
            coverage_report=coverage_report,
            budget_snapshot=budget_snapshot,
            trace_flush_succeeded=trace_flush_succeeded,
        )
