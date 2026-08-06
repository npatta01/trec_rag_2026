"""Narrative-only Deep Agent retrieval backed by the existing ClimbMix adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import ClassVar, Literal, Protocol, TypeVar, cast

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from trec_rag.deepagent_budget import (
    BudgetDecision,
    BudgetSnapshot,
    ResearchBudget,
    ResearchBudgetConfig,
    ResearchTaskContext,
)
from trec_rag.deepagent_evidence import (
    ActionKind,
    DeltaRejection,
    DocumentObservation,
    EvidenceCoverageReport,
    EvidenceCoverageState,
    RetrievalStateDelta,
    SnippetHandle,
)
from trec_rag.deepagent_research import (
    CloseoutAttemptsExhausted,
    MainToolFilterMiddleware,
    ResearchTaskBudgetMiddleware,
    _RoleToolFilterMiddleware,
    build_research_subagent,
    current_research_task,
    is_operational_provider_stop,
)
from trec_rag.deepagent_passages import (
    group_by_document,
)
from trec_rag.deepagent_snippets import (
    InvalidSnippetCursorError,
    RelevantSnippet,
    RelevantSnippetExtractor,
    SnippetExtractionResult,
    SnippetPage,
    create_default_snippet_extractor,
)
from trec_rag.deepagent_tracing import (
    MAX_TRACE_DOCUMENTS,
    create_retrieval_tracing,
)
from trec_rag.pipeline_models import RankedCandidate, RetrievedCandidate
from trec_rag.repo_env import find_repo_root, load_repo_env
from trec_rag.topic_passage_search import (
    FocusedQuery,
    OrganizerRequestFailed,
    PassageScoringFailed,
    PassageSearchResult,
    SourcePassage,
    TopicPassageSearch,
)
from trec_rag.topic_records import (
    FacetRecord,
    ResearcherEvidence,
    ResearcherHandoff,
    TopicRecordsBuilder,
    TopicRecordsIntegrityError,
)


DEFAULT_MODEL = "openrouter:deepseek/deepseek-v4-flash"
OPENROUTER_REQUEST_TIMEOUT_MS = 300_000
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
by one and delegate a new task batch aimed at those specific gaps.
Spend researchers by priority. Every need whose grounded_nugget_count is 0 comes
first: never deepen a need that already has evidence while another has none, and
never send two researchers at the same need in one batch while an empty need is
waiting. Once every need has some grounded evidence, keep going on the needs
still marked partial, targeting each one's recorded remaining_gap; partial means
unfinished, not done. Give an empty bundle's need a materially different angle
rather than repeating the goal that already returned nothing.
Set task.description to a JSON-encoded object with exactly this shape:
{"research_task_id":"R1-N1","round_index":1,"depth":"focused","motivating_ids":["N1"],"goal":"Find grounded evidence for N1","known_evidence":"","remaining_gap":"No grounded evidence yet"}.
The JSON object must come first. Optional research instructions may follow it
after a newline; never put prose before the JSON object.
Do not call retrieval tools directly. Stop whenever a budget response says
must_stop. After soft_deadline_reached becomes true, do not launch a new survey
task; focused or deep tasks may still close a specific gap, and already-running
tasks may finish. Merge completed bundles and finalize even after the soft
deadline.
Every merge delta must also update need status with set_need_status. Move a need
to "partial" with a refreshed remaining_gap as soon as it has grounded nuggets,
and to "answerable" only with a draft_answer and grounded draft_nugget_ids.
Leaving a need "unaddressed" after its researchers returned is a reporting
error. Nugget evidence is a list of citations such as
[{"cite":"S3.2"}]; pass through the handles a researcher returned and never
write quote text yourself. Importance is not something you
write: a nugget counts as vital exactly when you select it into a need's
draft_nugget_ids, so choose those few deliberately. If add_nuggets rows come back rejected, those claims
never entered the ledger: leave the cited needs unsupported and delegate
researchers for them again. Report conflicts and unresolved gaps. Caches remain tool-owned; use state scratch only
for oversized output or temporary notes. The state filesystem is ephemeral.
When research is over, write every grounded need's best draft_nugget_ids and
then call complete_retrieval. Do not use any other completion action: the
completion tool validates the ledger before recording the terminal transition.
Only read_file is available from it."""

def _coordinator_prompt(config: ResearchBudgetConfig) -> str:
    """State the dispatch limits, which the coordinator cannot otherwise know.

    Built from the live config rather than written into the prompt text, so the
    numbers an agent is told can never drift from the ones enforced.
    """
    if config.max_concurrent == 1:
        dispatch = (
            "Exactly one researcher runs at a time. Issue a single task call per "
            "turn and wait for its bundle before dispatching the next; a second "
            "task call in the same turn is refused outright, and recovering from "
            "that refusal costs a turn you need later."
        )
    else:
        dispatch = (
            f"At most {config.max_concurrent} researchers run at once. Never put "
            f"more than {config.max_concurrent} task calls in one batch: the "
            "extras are refused outright, and recovering from that refusal costs "
            "turns you need for later rounds."
        )
    return (
        f"{RETRIEVAL_SYSTEM_PROMPT}\n{dispatch} At most "
        f"{config.max_researcher_invocations} researchers may run in the whole "
        "invocation, so spend them on needs that still have no grounded evidence."
    )


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


_EVIDENCE_REJECTION_CODES = frozenset(
    {
        "UNKNOWN_CITATION",
        "INVALID_CITATION",
        "DUPLICATE_EVIDENCE",
    }
)


def _model_facing_snippets(
    handles: Sequence[SnippetHandle],
) -> list[dict[str, object]]:
    """Show each snippet as a citable handle with numbered sentences.

    Agents never receive raw snippet identifiers or echo passage text back, so
    the only thing they transport is a short handle.
    """
    return [
        {
            # Named "cite" so it matches the evidence field an agent must fill,
            # and so sort_keys renders the handle above its own sentences.
            "cite": handle.handle,
            "relevance_score": handle.relevance_score,
            "sentences": [
                {"n": index, "text": sentence}
                for index, sentence in enumerate(handle.sentences, start=1)
            ],
        }
        for handle in handles
    ]


def agentic_passage_payload(
    result: PassageSearchResult,
    handles: Sequence[SnippetHandle] | None = None,
) -> dict[str, object]:
    """Render every shared passage row without introducing a second policy.

    ``handles`` is supplied by the live tool after the evidence validator has
    registered the rows.  The standalone form is useful to adapters and keeps
    the source passage identity visible in offline tests.
    """
    if not isinstance(result, PassageSearchResult):
        raise TypeError("result must be a PassageSearchResult")
    by_passage_id = {
        handle.snippet_id: handle for handle in (handles or ())
    }
    rows: list[dict[str, object]] = []
    for passage in result.passages:
        handle = by_passage_id.get(passage.passage_id)
        rows.append(
            {
                "cite": handle.handle if handle is not None else passage.passage_id,
                "passage_id": passage.passage_id,
                "document_id": passage.docid,
                "docid": passage.docid,
                "source_rank": passage.source_rank,
                "source_score": passage.source_score,
                "start_char": passage.start_char,
                "end_char": passage.end_char,
                "start_byte": passage.start_byte,
                "end_byte": passage.end_byte,
                "text": passage.text,
                "raw_logit": passage.raw_logit,
                "relevance_score": passage.raw_logit,
                "rank": passage.rank,
                **(
                    {
                        "sentences": [
                            {"n": index, "text": sentence}
                            for index, sentence in enumerate(
                                handle.sentences, start=1
                            )
                        ],
                    }
                    if handle is not None
                    else {}
                ),
            }
        )
    return {
        "passages": rows,
        "documents_retrieved": result.returned_documents,
        "documents_scored": result.scored_documents,
        "documents_not_scored": max(
            0, result.returned_documents - result.scored_documents
        ),
        "scored_passages": result.scored_passages,
        "returned_passages": len(result.passages),
        "distinct_documents": len({passage.docid for passage in result.passages}),
        "chunks_scored": result.scored_passages,
        "chunks_not_returned": max(0, result.scored_passages - len(result.passages)),
    }


def _rejection_summary(rejected: Sequence[DeltaRejection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rejection in rejected:
        counts[rejection.code] = counts.get(rejection.code, 0) + 1
    return counts


def _evidence_rejection_guidance(sections: Iterable[str]) -> dict[str, object]:
    """State plainly that rejected rows never entered the ledger."""
    return {
        "unadmitted_sections": sorted(set(sections)),
        "evidence_rejection_notice": (
            "These rows were NOT admitted. The needs they cited gained no "
            "grounded evidence from this round, so treat those needs as still "
            "lacking support. Evidence must cite a snippet handle returned "
            'during this run, as "S3", "S3.2", or "S3.2-4". Do not invent a '
            "handle and do not write quote text. Delegate a researcher for "
            "those needs again if no valid handle covers the claim."
        ),
    }


@dataclass(frozen=True)
class AgentSearch:
    """One completed original or agent-directed ClimbMix search."""

    query: str
    kind: Literal["original", "followup"]
    candidates: tuple[RetrievedCandidate, ...]
    cache_status: str
    passages: tuple[SourcePassage, ...] = ()


@dataclass(frozen=True)
class AgentRetrievalResult:
    """Immutable retrieval result, including the optional trace flush outcome."""

    narrative: str
    searches: tuple[AgentSearch, ...]
    candidates: tuple[AgentRankedCandidate, ...]
    rationale: str
    stopping_reason: str
    synthesis_outcome: Literal[
        "coordinator_selected",
        "deterministic_grounded_recovery",
        "zero_grounded_nuggets",
    ]
    coverage_report: EvidenceCoverageReport
    budget_snapshot: BudgetSnapshot
    trace_flush_succeeded: bool
    topic_snapshot: object | None = None


class RetrievalTransportError(RuntimeError):
    """The index could not be reached, as distinct from returning nothing."""


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


class _RetrievalOnlyMiddleware(_RoleToolFilterMiddleware):
    """Limit one SDK-created agent to retrieval and state scratch tools."""

    _ALLOWED_TOOLS = frozenset(
        {
            "search_climbmix",
            "extract_relevant_snippets",
            "view_retrieval_state",
            "update_retrieval_state",
            "ls",
            "read_file",
            "glob",
            "grep",
        }
    )
    _DENIED_MESSAGE = "retrieval-only tool access denied"


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
    search_passages: Callable[[str, list[str], str], str]
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
    complete_retrieval: Callable[[], str] | None = None
    closeout_pending: Callable[[], bool] | None = None


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
            toolset.search_passages,
            toolset.view_retrieval_state,
        ],
        budget=toolset.budget,
        budget_config=toolset.budget_config,
    )

    coordinator_tools = [
        toolset.view_retrieval_state,
        toolset.update_retrieval_state,
        toolset.complete_research_round,
    ]
    if toolset.complete_retrieval is not None:
        coordinator_tools.append(toolset.complete_retrieval)

    return cast(
        _Agent,
        create_deep_agent(
            model=provider_model,
            tools=coordinator_tools,
            system_prompt=_coordinator_prompt(toolset.budget_config),
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
                MainToolFilterMiddleware(
                    toolset.budget,
                    toolset.budget_config,
                    closeout_pending=toolset.closeout_pending,
                ),
            ],
            subagents=[researcher],
            backend=StateBackend(),
        ),
    )


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
        passage_search: TopicPassageSearch,
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
        if not callable(getattr(passage_search, "search", None)):
            raise TypeError("passage_search must provide search(FocusedQuery)")
        if not isinstance(getattr(passage_search, "topic_id", None), str) or not (
            passage_search.topic_id.strip()
        ):
            raise ValueError("passage_search must expose a non-empty topic_id")
        self._passage_search = passage_search
        self._model = model
        self._agent_factory = agent_factory
        self._tracing = tracing or create_retrieval_tracing()
        self._snippet_extractor = snippet_extractor
        self._snippet_extractor_lock = Lock()

    @classmethod
    def from_env(
        cls,
        *,
        passage_search: TopicPassageSearch,
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
        """Build agent orchestration around an already topic-scoped search."""
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
        return cls(
            passage_search=passage_search,
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

    def search_passages(self, query: FocusedQuery) -> PassageSearchResult:
        """Run the injected shared search adapter without a second policy."""
        result = self._passage_search.search(query)
        if not isinstance(result, PassageSearchResult):
            raise TypeError("passage search must return a PassageSearchResult")
        return result

    def retrieve(
        self,
        records: TopicRecordsBuilder,
        narrative: str,
    ) -> AgentRetrievalResult:
        """Run bounded research against an already topic/run-scoped ledger."""
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError("narrative must be non-empty text")
        required = (
            "add_facets",
            "add_passage_search",
            "add_researcher_handoff",
            "topic_snapshot",
            "set_completion",
        )
        if any(not callable(getattr(records, name, None)) for name in required):
            raise TypeError("records must be a topic-scoped TopicRecordsBuilder")
        records_topic_id = getattr(records, "topic_id", None)
        records_run_id = getattr(records, "run_id", None)
        if not isinstance(records_topic_id, str) or not records_topic_id.strip():
            raise ValueError("topic records builder must expose its topic identity")
        if not isinstance(records_run_id, str) or not records_run_id.strip():
            raise ValueError("topic records builder must expose its run identity")
        if self._passage_search.topic_id != records_topic_id:
            raise ValueError("passage_search topic_id does not match records topic_id")
        records_builder = cast(TopicRecordsBuilder, records)

        coverage_state = EvidenceCoverageState(narrative)
        budget = ResearchBudget(self._budget_config)
        searches: list[AgentSearch] = []
        documents: dict[str, str] = {}
        followup_lock = Lock()
        document_lock = Lock()
        shared_failure_reasons: list[str] = []
        shared_task_passages: dict[str, set[str]] = {}
        shared_task_rounds: dict[str, int] = {}
        dynamic_facet_creators: dict[str, str] = {}
        admitted_ledger_facets: set[str] = set()
        topic_snapshot: object | None = None
        operational_provider_stop = False
        closeout_provider_stop = False
        unexpected_tool_failures: list[tuple[str, Exception]] = []
        unexpected_tool_failure_lock = Lock()

        def guard_retrieval_tool(
            name: str,
            function: Callable[..., str],
        ) -> Callable[..., str]:
            @wraps(function)
            def guarded(*args: object, **kwargs: object) -> str:
                try:
                    return function(*args, **kwargs)
                except Exception as exc:
                    with unexpected_tool_failure_lock:
                        if not unexpected_tool_failures:
                            unexpected_tool_failures.append((name, exc))
                    return json.dumps(
                        {
                            "ok": False,
                            "code": "INTERNAL_TOOL_FAILURE",
                            "must_stop": True,
                            "budget_snapshot": budget.snapshot().as_dict(),
                            "error": "internal retrieval tool failed",
                            "tool": name,
                        },
                        sort_keys=True,
                    )

            return guarded

        def _records_run_id() -> str:
            return records_run_id

        def _ensure_ledger_facets(facet_ids: Sequence[str], query: str) -> None:
            rows: list[FacetRecord] = []
            coverage_report = coverage_state.report()
            coverage_facets = {
                facet.facet_id: facet for facet in coverage_report.facets
            }
            coverage_needs = {need.need_id: need for need in coverage_report.needs}
            for index, facet_id in enumerate(dict.fromkeys(facet_ids)):
                if not isinstance(facet_id, str) or not facet_id.strip():
                    continue
                if facet_id in admitted_ledger_facets:
                    continue
                coverage_facet = coverage_facets.get(facet_id)
                coverage_need = coverage_needs.get(facet_id)
                facet_text = (
                    coverage_facet.value
                    if coverage_facet is not None
                    else coverage_need.question
                    if coverage_need is not None
                    else query
                    if index == 0
                    else f"Supporting evidence for {facet_id}"
                )
                facet_origin = (
                    "research_discovered"
                    if coverage_facet is not None and coverage_facet.origin == "snippet"
                    else "initial"
                )
                rows.append(
                    FacetRecord(
                        facet_id,
                        facet_text,
                        facet_origin,
                    )
                )
            if rows:
                records_builder.add_facets(tuple(rows))
                admitted_ledger_facets.update(
                    facet.subnarrative_id for facet in rows
                )

        def _shared_query(
            query: str,
            context: ResearchTaskContext | None,
            motivating_ids: Sequence[str],
        ) -> FocusedQuery:
            if context is None:
                primary = motivating_ids[0] if motivating_ids else "original"
                supporting = tuple(motivating_ids[1:])
                query_id = "original"
            else:
                primary = motivating_ids[0] if motivating_ids else context.research_task_id
                supporting = tuple(motivating_ids[1:])
                query_id = "followup-" + sha256(query.encode()).hexdigest()[:16]
            _ensure_ledger_facets((primary, *supporting), query)
            return FocusedQuery(query_id, query, primary, supporting)

        def _register_shared_passages(
            result: PassageSearchResult,
        ) -> tuple[SnippetHandle, ...]:
            # Grouping is only an index for document metadata. Register the
            # original shared sequence so S-handles and model rows retain the
            # global (-score, source-rank, passage-id) ordering.
            observed: list[SnippetHandle] = []
            for passage in result.passages:
                page = SnippetPage(
                    document_id=passage.docid,
                    focus_query=result.query.text,
                    snippets=(
                        RelevantSnippet(
                            chunk_id=passage.passage_id,
                            start_char=passage.start_char,
                            end_char=passage.end_char,
                            text=passage.text,
                            relevance_score=passage.raw_logit,
                        ),
                    ),
                    next_cursor=None,
                    page_index=0,
                    residual_count=0,
                    residual_top_score=None,
                    returned_min_score=passage.raw_logit,
                    pages_estimated=1,
                )
                observed.extend(coverage_state.record_snippet_page(page))
            return tuple(observed)

        def _shared_agent_search(
            query: str,
            kind: Literal["original", "followup"],
            *,
            context: ResearchTaskContext | None = None,
            decision: BudgetDecision | None = None,
            motivating_ids: Sequence[str] = (),
        ) -> tuple[AgentSearch, tuple[SnippetHandle, ...], PassageSearchResult]:
            focused_query = _shared_query(query, context, motivating_ids)
            try:
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
                    result = self.search_passages(focused_query)
                    latency_ms = (monotonic() - started_at) * 1_000
                    if span is not None:
                        trace_documents = result.documents[
                            : min(self._hits_per_search, MAX_TRACE_DOCUMENTS)
                        ]
                        text_lengths = {
                            docid: len(rows[0].text)
                            for docid, rows in group_by_document(result.passages)
                            if rows
                        }
                        try:
                            span.record_search(
                                document_ids=tuple(
                                    document.docid for document in trace_documents
                                ),
                                source_ranks=tuple(
                                    document.source_rank for document in trace_documents
                                ),
                                source_scores=tuple(
                                    document.source_score for document in trace_documents
                                ),
                                cache_status="not_reported",
                                latency_ms=latency_ms,
                                text_lengths=tuple(
                                    text_lengths.get(document.docid, 0)
                                    for document in trace_documents
                                ),
                            )
                        except Exception:
                            pass
            except OrganizerRequestFailed:
                shared_failure_reasons.append("retrieval_unavailable")
                raise RetrievalTransportError("shared passage retrieval unavailable")
            except PassageScoringFailed:
                shared_failure_reasons.append("scoring_failed")
                raise RetrievalTransportError("shared passage scoring failed")

            records_builder.add_passage_search(result)
            handles = _register_shared_passages(result)
            if result.status == "incomplete" and result.stopping_reason is not None:
                shared_failure_reasons.append(result.stopping_reason)
            if context is not None:
                shared_task_passages.setdefault(context.research_task_id, set()).update(
                    passage.passage_id for passage in result.passages
                )
                shared_task_rounds[context.research_task_id] = context.round_index
            coverage_state.record_search(
                query=query,
                kind=kind,
                documents=tuple(
                    DocumentObservation(document.docid, document.source_rank)
                    for document in result.documents
                ),
            )
            passages_by_doc = {
                docid: rows[0].text
                for docid, rows in group_by_document(result.passages)
                if rows
            }
            candidates = tuple(
                RetrievedCandidate(
                    topic_id=self._passage_search.topic_id,
                    variant_name=result.query.query_id,
                    retriever_name="topic_passage_search",
                    query_text=query,
                    docid=document.docid,
                    rank=document.source_rank,
                    score=document.source_score,
                    text=passages_by_doc.get(document.docid, ""),
                )
                for document in result.documents
            )
            return (
                AgentSearch(
                    query=query,
                    kind=kind,
                    candidates=candidates,
                    cache_status="shared",
                    passages=result.passages,
                ),
                handles,
                result,
            )

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
                try:
                    search, _handles, shared_result = _shared_agent_search(
                        query,
                        "followup",
                        context=context,
                        decision=decision,
                        motivating_ids=motivating_ids,
                    )
                except RetrievalTransportError:
                    budget.note_retrieval_unavailable()
                    record_no_yield(context)
                    return json.dumps(
                        budget_payload(
                            BudgetDecision(
                                False, decision.code, budget.snapshot(), True
                            ),
                            error="shared passage search failed",
                            code="RETRIEVAL_UNAVAILABLE",
                        ),
                        sort_keys=True,
                    )
                searches.append(search)
                budget.record_yield(context, (row.passage_id for row in search.passages))
                snapshot = budget.snapshot()
                return json.dumps(
                    {
                        "ok": shared_result.status == "complete",
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

        def search_passages(
            query: str,
            motivating_ids: list[str],
            rationale: str,
        ) -> str:
            """Search ClimbMix and return the best passages across many documents."""
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
            # A pooled scoring pass costs materially more than one snippet page,
            # so it is charged as its own retrieval unit rather than reusing the
            # per-snippet-call accounting.
            decision = budget.reserve_retrieval(context, "search_passages")
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
                try:
                    search, observed, result = _shared_agent_search(
                        query,
                        "followup",
                        context=context,
                        decision=decision,
                        motivating_ids=motivating_ids,
                    )
                except RetrievalTransportError:
                    record_no_yield(context)
                    code = (
                        "PASSAGE_SCORING_FAILED"
                        if shared_failure_reasons
                        and shared_failure_reasons[-1] == "scoring_failed"
                        else "RETRIEVAL_UNAVAILABLE"
                    )
                    return json.dumps(
                        budget_payload(
                            BudgetDecision(
                                False, decision.code, budget.snapshot(), True
                            ),
                            error="shared passage search failed",
                            code=code,
                        ),
                        sort_keys=True,
                    )

                searches.append(search)
                budget.record_yield(context, (handle.snippet_id for handle in observed))
                snapshot = budget.snapshot()
                payload: dict[str, object] = {
                    "ok": result.status == "complete",
                    "code": decision.code,
                    "must_stop": decision.must_stop or task_must_stop(snapshot),
                    "budget_snapshot": snapshot.as_dict(),
                    "focus_query": query,
                    "remaining_budget": snapshot.remaining_retrieval_calls,
                }
                payload.update(agentic_passage_payload(result, observed))
                if result.status == "incomplete":
                    payload["error"] = "shared passage search incomplete"
                    payload["code"] = {
                        "retrieval_unavailable": "RETRIEVAL_UNAVAILABLE",
                        "scoring_failed": "PASSAGE_SCORING_FAILED",
                        "no_evidence": "NO_EVIDENCE",
                    }.get(result.stopping_reason, "PASSAGE_SEARCH_INCOMPLETE")
                return json.dumps(payload, sort_keys=True)

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
                    observed_handles = coverage_state.record_snippet_page(result.page)
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
            payload["snippets"] = _model_facing_snippets(observed_handles)
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
            context = task_context()
            if context is not None and isinstance(delta, Mapping):
                facet_rows = delta.get("add_facets", ())
                if isinstance(facet_rows, (list, tuple)):
                    rejected_indexes = {
                        rejection.index
                        for rejection in update.rejected
                        if rejection.section == "add_facets"
                    }
                    visible_passages = shared_task_passages.get(
                        context.research_task_id, set()
                    )
                    for index, row in enumerate(facet_rows):
                        if index in rejected_indexes or not isinstance(row, Mapping):
                            continue
                        facet_id = row.get("facet_id")
                        origin_snippet_id = row.get("origin_snippet_id")
                        if (
                            row.get("origin") == "snippet"
                            and isinstance(facet_id, str)
                            and facet_id in update.accepted_ids
                            and isinstance(origin_snippet_id, str)
                            and origin_snippet_id in visible_passages
                        ):
                            dynamic_facet_creators[facet_id] = (
                                context.research_task_id
                            )
            payload = update.as_dict()
            summary = _rejection_summary(update.rejected)
            if summary:
                payload["rejected_summary"] = summary
            if any(
                rejection.code in {"UNKNOWN_SECTION", "EMPTY_DELTA"}
                for rejection in update.rejected
            ):
                payload.update(_delta_error_guidance())
            ungrounded = [
                rejection.section
                for rejection in update.rejected
                if rejection.code in _EVIDENCE_REJECTION_CODES
            ]
            if ungrounded:
                payload.update(_evidence_rejection_guidance(ungrounded))
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

        def complete_retrieval() -> str:
            """Validate draft coverage, then record the terminal completion."""
            result = coverage_state.complete_retrieval()
            if not result.get("ok", False):
                budget.note_closeout_refused()
            return json.dumps(result, sort_keys=True)

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
            if target == "completion":
                return json.dumps(
                    {"ok": False, "code": "COMPLETE_RETRIEVAL_REQUIRED"},
                    sort_keys=True,
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

        def commit_researcher_handoffs(
            report: EvidenceCoverageReport,
        ) -> frozenset[str]:
            """Commit already validated passage citations once per researcher."""
            nuggets = tuple(report.nuggets)
            evidence_subnarrative_ids = tuple(
                dict.fromkeys(
                    subnarrative_id
                    for nugget in nuggets
                    for subnarrative_id in (nugget.facet_ids or nugget.need_ids)
                )
            )
            if evidence_subnarrative_ids:
                _ensure_ledger_facets(
                    evidence_subnarrative_ids,
                    "Grounded researcher evidence",
                )
            run_id = _records_run_id()
            admitted_passage_ids: set[str] = set()
            for researcher_id, passage_ids in shared_task_passages.items():
                evidence_rows: list[ResearcherEvidence] = []
                seen: set[tuple[str, str]] = set()
                facet_updates = [
                    FacetRecord(
                        facet.facet_id,
                        facet.value,
                        "research_discovered",
                    )
                    for facet in report.facets
                    if facet.origin == "snippet"
                    and dynamic_facet_creators.get(facet.facet_id) == researcher_id
                ]
                for nugget in nuggets:
                    for evidence in nugget.evidence:
                        if evidence.snippet_id not in passage_ids:
                            continue
                        subnarrative_ids = nugget.facet_ids or nugget.need_ids
                        for subnarrative_id in subnarrative_ids:
                            key = (subnarrative_id, evidence.snippet_id)
                            if key in seen:
                                continue
                            seen.add(key)
                            evidence_rows.append(
                                ResearcherEvidence(
                                    subnarrative_id,
                                    evidence.snippet_id,
                                    "relevant",
                                )
                            )
                unique_updates = tuple(
                    {facet.subnarrative_id: facet for facet in facet_updates}.values()
                )
                records_builder.add_researcher_handoff(
                    ResearcherHandoff(
                        run_id=run_id,
                        researcher_id=researcher_id,
                        round_index=shared_task_rounds.get(researcher_id, 1),
                        evidence=tuple(evidence_rows),
                        facet_updates=tuple(unique_updates),
                    )
                )
                admitted_passage_ids.update(
                    evidence.passage_id for evidence in evidence_rows
                )
            return frozenset(admitted_passage_ids)

        try:
            with _isolated_trace_span(
                lambda: self._tracing.agent_span(narrative)
            ) as agent_span:
                try:
                    original, _handles, _result = _shared_agent_search(
                        narrative,
                        "original",
                        motivating_ids=("original",),
                    )
                except RetrievalTransportError:
                    original = AgentSearch(
                        narrative, "original", (), "unavailable"
                    )
                searches.append(original)
                agent = self._agent_factory(
                    self._model,
                    AgentToolset(
                        search_climbmix=guard_retrieval_tool(
                            "search_climbmix", search_climbmix
                        ),
                        search_passages=guard_retrieval_tool(
                            "search_passages", search_passages
                        ),
                        extract_relevant_snippets=guard_retrieval_tool(
                            "extract_relevant_snippets", extract_relevant_snippets
                        ),
                        view_retrieval_state=guard_retrieval_tool(
                            "view_retrieval_state", view_retrieval_state
                        ),
                        update_retrieval_state=guard_retrieval_tool(
                            "update_retrieval_state", update_retrieval_state
                        ),
                        complete_retrieval=guard_retrieval_tool(
                            "complete_retrieval", complete_retrieval
                        ),
                        complete_research_round=guard_retrieval_tool(
                            "complete_research_round", complete_research_round
                        ),
                        choose_next_action=guard_retrieval_tool(
                            "choose_next_action", choose_next_action
                        ),
                        budget=budget,
                        budget_config=self._budget_config,
                        tracing=self._tracing,
                        closeout_pending=lambda: bool(
                            coverage_state.pending_closeout_need_ids()
                        ),
                    ),
                )
                initial_results = json.dumps(
                    _candidate_metadata(
                        original.candidates, limit=self._hits_per_search
                    ),
                    sort_keys=True,
                )
                try:
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
                except CloseoutAttemptsExhausted as exc:
                    cause = exc.__cause__
                    if not isinstance(cause, Exception):
                        raise
                    if isinstance(
                        cause,
                        (AssertionError, TypeError, ValueError, TopicRecordsIntegrityError),
                    ) or not is_operational_provider_stop(cause):
                        raise cause
                    closeout_provider_stop = True
                    reply = {"messages": ()}
                except Exception as exc:
                    if isinstance(
                        exc,
                        (AssertionError, TypeError, ValueError, TopicRecordsIntegrityError),
                    ) or not is_operational_provider_stop(exc):
                        raise
                    operational_provider_stop = True
                    reply = {"messages": ()}
                with unexpected_tool_failure_lock:
                    unexpected_tool_failure = (
                        unexpected_tool_failures[0]
                        if unexpected_tool_failures
                        else None
                    )
                if unexpected_tool_failure is not None:
                    raise AgentRetrievalError(
                        "agentic retrieval tool failed",
                        searches=searches,
                    ) from unexpected_tool_failure[1]
                rationale = _assistant_rationale(reply)
                candidates = reciprocal_rank_fuse(
                    searches, limit=self._fused_result_limit
                )
                recovery = coverage_state.recover_grounded_drafts()
                coverage_report = coverage_state.report()
                if not recovery.live_grounded_nugget_ids:
                    synthesis_outcome = "zero_grounded_nuggets"
                elif recovery.recovered_need_ids:
                    synthesis_outcome = "deterministic_grounded_recovery"
                else:
                    synthesis_outcome = "coordinator_selected"
                admitted_handoff_passage_ids = commit_researcher_handoffs(coverage_report)
                live_evidence_passage_ids = {
                    evidence.snippet_id
                    for nugget in coverage_report.nuggets
                    if nugget.superseded_by is None
                    for evidence in nugget.evidence
                }
                shared_passage_ids = {
                    passage_id
                    for passage_ids in shared_task_passages.values()
                    for passage_id in passage_ids
                }
                evidence_validation_failed = bool(
                    live_evidence_passage_ids.difference(shared_passage_ids)
                    or live_evidence_passage_ids.difference(
                        admitted_handoff_passage_ids
                    )
                )
                closeout_pending_after_recovery = bool(
                    coverage_state.pending_closeout_need_ids()
                )
                if closeout_pending_after_recovery:
                    budget.note_closeout_refused()
                budget_snapshot = budget.snapshot()
                if evidence_validation_failed:
                    stopping_reason = "evidence_validation_failed"
                elif "scoring_failed" in shared_failure_reasons:
                    stopping_reason = "scoring_failed"
                elif (
                    budget.retrieval_unavailable()
                    or "retrieval_unavailable" in shared_failure_reasons
                    or operational_provider_stop
                ):
                    stopping_reason = "retrieval_unavailable"
                elif budget_snapshot.hard_deadline_reached:
                    stopping_reason = "hard_deadline"
                elif synthesis_outcome == "zero_grounded_nuggets":
                    stopping_reason = "zero_grounded_nuggets"
                elif closeout_provider_stop:
                    stopping_reason = "budget_exhausted"
                elif coverage_report.terminal_reason is not None:
                    stopping_reason = coverage_report.terminal_reason
                elif budget_snapshot.stop_code is not None:
                    stopping_reason = "budget_exhausted"
                elif budget.closeout_refused() and closeout_pending_after_recovery:
                    stopping_reason = "closeout_refused"
                else:
                    stopping_reason = "agent_completed"
                incomplete_reasons = {
                    "retrieval_unavailable",
                    "scoring_failed",
                    "hard_deadline",
                    "zero_grounded_nuggets",
                    "evidence_validation_failed",
                }
                if stopping_reason in incomplete_reasons:
                    completion = ("incomplete", stopping_reason)
                elif stopping_reason == "closeout_refused" or (
                    stopping_reason == "agent_completed"
                    and coverage_report.terminal_reason is None
                ):
                    completion = ("incomplete", "no_evidence")
                elif coverage_report.terminal_reason is not None:
                    completion = ("complete", "coverage_sufficient")
                else:
                    completion = ("complete", stopping_reason)
                records_builder.set_completion(*completion)
                # Completion is part of the coordinator view. Take exactly one
                # holistic snapshot, after every handoff and completion write.
                topic_snapshot = records_builder.topic_snapshot()
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
        except Exception:
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
            synthesis_outcome=synthesis_outcome,
            coverage_report=coverage_report,
            budget_snapshot=budget_snapshot,
            trace_flush_succeeded=trace_flush_succeeded,
            topic_snapshot=topic_snapshot,
        )
