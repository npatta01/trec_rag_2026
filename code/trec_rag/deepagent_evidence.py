"""Invocation-local, mechanically grounded evidence and coverage state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import re
from threading import Lock
from typing import TYPE_CHECKING, Literal

from typing_extensions import NotRequired, TypedDict

from pydantic import ConfigDict

from trec_rag.chunking import sentence_segmenter
from trec_rag.deepagent_snippets import SnippetPage

if TYPE_CHECKING:
    from trec_rag.deepagent_budget import ResearchTaskContext


NeedStatus = Literal["unaddressed", "partial", "answerable", "conflicted"]
FacetStatus = Literal["open", "covered", "dropped"]
DocumentState = Literal["unexamined", "productive", "exhausted", "abandoned"]
ActionKind = Literal["search", "extract", "paginate", "refocus", "stop"]

MAX_FRONTIER_CHARACTERS = 2_000
SATURATION_ZERO_YIELD_PAGES = 3
MAX_SENTENCE_CHARACTERS = 600
# A cap, not a target: a need with three good nuggets must not pad to five.
# Sized against observed volume of roughly 21 nuggets per need, which gives
# about 4:1 selection pressure and 20-35 drafted nuggets per topic.
MAX_DRAFT_NUGGETS_PER_NEED = 5
_CITATION = re.compile(r"^S(\d+)(?:\.(\d+)(?:-(\d+))?)?$")


def _capped_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Trim one segment and split it so no sentence exceeds the length cap."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return []
    spans: list[tuple[int, int]] = []
    while end - start > MAX_SENTENCE_CHARACTERS:
        limit = start + MAX_SENTENCE_CHARACTERS
        cut = text.rfind(" ", start, limit)
        if cut <= start:
            cut = limit
        spans.append((start, cut))
        start = cut
        while start < end and text[start].isspace():
            start += 1
    if start < end:
        spans.append((start, end))
    return spans


def _sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Split snippet text into deterministic, length-capped sentence spans.

    Splitting happens once, when a snippet is first observed. Citations resolve
    against the stored spans and never re-split, so a handle cannot drift.

    Boundaries come from the shared segmenter, so this lane and competition
    retrieval cut text the same way. A span that does not stand on its own - a
    heading, a list caption, the "Plyler v." that once had a Supreme Court
    holding cited to it - is joined to what follows rather than left citable.
    """
    sentences = sentence_segmenter().segment(text)
    if not sentences:
        return ((0, len(text)),)
    merged: list[tuple[int, int]] = []
    pending: int | None = None
    for row in sentences:
        start = pending if pending is not None else row.start_char
        if not row.is_complete:
            pending = start
            continue
        merged.append((start, row.end_char))
        pending = None
    if pending is not None:
        # A trailing fragment has nothing after it to join, so it folds back.
        if merged:
            previous_start, _ = merged.pop()
            merged.append((previous_start, sentences[-1].end_char))
        else:
            merged.append((pending, sentences[-1].end_char))
    spans: list[tuple[int, int]] = []
    for start, end in merged:
        spans.extend(_capped_spans(text, start, end))
    return tuple(spans) or ((0, len(text)),)


_DELTA_SECTIONS = (
    "add_needs",
    "add_facets",
    "add_nuggets",
    "add_evidence",
    "set_facet_status",
    "set_need_status",
    "supersede_nuggets",
    "abandon_documents",
)


class NeedDelta(TypedDict):
    need_id: str
    narrative_span: str
    question: str


class FacetDelta(TypedDict):
    facet_id: str
    need_ids: list[str]
    dimension: str
    value: str
    origin: Literal["narrative", "snippet"]
    origin_snippet_id: NotRequired[str | None]


class EvidenceDelta(TypedDict):
    cite: str


class NuggetDelta(TypedDict):
    nugget_id: str
    text: str
    need_ids: list[str]
    facet_ids: list[str]
    evidence: list[EvidenceDelta]
    contradicts: NotRequired[list[str]]


class AddEvidenceDelta(TypedDict):
    nugget_id: str
    snippet_id: str
    quote: str


class FacetStatusDelta(TypedDict):
    facet_id: str
    status: FacetStatus
    status_reason: NotRequired[str | None]
    supporting_nugget_ids: NotRequired[list[str]]


class NeedStatusDelta(TypedDict):
    need_id: str
    status: NeedStatus
    remaining_gap: str
    draft_answer: NotRequired[str | None]
    draft_nugget_ids: NotRequired[list[str]]


class SupersedeNuggetDelta(TypedDict):
    nugget_id: str
    superseded_by: str


class AbandonDocumentDelta(TypedDict):
    document_id: str
    reason: str


class RetrievalStateDelta(TypedDict, total=False):
    """The model-facing delta accepted by :meth:`EvidenceCoverageState.apply_delta`."""

    __pydantic_config__ = ConfigDict(extra="allow")

    add_needs: list[NeedDelta]
    add_facets: list[FacetDelta]
    add_nuggets: list[NuggetDelta]
    add_evidence: list[AddEvidenceDelta]
    set_facet_status: list[FacetStatusDelta]
    set_need_status: list[NeedStatusDelta]
    supersede_nuggets: list[SupersedeNuggetDelta]
    abandon_documents: list[AbandonDocumentDelta]


@dataclass(frozen=True)
class DocumentObservation:
    document_id: str
    rank: int


@dataclass(frozen=True)
class EvidenceReference:
    document_id: str
    snippet_id: str
    page_index: int
    quote: str


@dataclass(frozen=True)
class SnippetHandle:
    """One observed snippet as an agent may cite it: a handle and its sentences."""

    handle: str
    snippet_id: str
    relevance_score: float
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class _EvidenceSpan:
    """A resolved citation. The quote is derived from this, never stored."""

    snippet_id: str
    first_sentence: int
    last_sentence: int


@dataclass(frozen=True)
class DeltaRejection:
    section: str
    index: int
    code: str


@dataclass(frozen=True)
class StateUpdateResult:
    accepted_ids: tuple[str, ...]
    rejected: tuple[DeltaRejection, ...]
    state_version: int
    state_hash: str

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted_ids": list(self.accepted_ids),
            "rejected": [asdict(item) for item in self.rejected],
            "state_version": self.state_version,
            "state_hash": self.state_hash,
        }


@dataclass(frozen=True)
class NeedReport:
    need_id: str
    narrative_span: str
    question: str
    status: NeedStatus
    remaining_gap: str
    facet_ids: tuple[str, ...]
    nugget_ids: tuple[str, ...]
    draft_answer: str | None
    draft_nugget_ids: tuple[str, ...]


@dataclass(frozen=True)
class FacetReport:
    facet_id: str
    need_ids: tuple[str, ...]
    dimension: str
    value: str
    origin: Literal["narrative", "snippet"]
    origin_snippet_id: str | None
    status: FacetStatus
    status_reason: str
    supporting_nugget_ids: tuple[str, ...]


@dataclass(frozen=True)
class NuggetReport:
    nugget_id: str
    text: str
    need_ids: tuple[str, ...]
    facet_ids: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...]
    contradicts: tuple[str, ...]
    support: Literal["single_document", "multi_document"]
    superseded_by: str | None
    importance: Literal["vital", "okay"] = "okay"
    support_ratio: float = 0.0


@dataclass(frozen=True)
class ActionReport:
    action: ActionKind
    target: str
    focus_query: str | None
    motivating_ids: tuple[str, ...]
    rationale: str
    state: Literal["pending", "consumed", "terminal"]
    research_task_id: str | None = None
    round_index: int | None = None
    depth: Literal["survey", "focused", "deep"] | None = None


@dataclass(frozen=True)
class SearchReport:
    query: str
    kind: Literal["original", "followup"]
    document_ids: tuple[str, ...]


@dataclass(frozen=True)
class DocumentFocusReport:
    document_id: str
    focus_query: str
    pages_fetched: tuple[int, ...]
    next_page_available: bool
    residual_count: int
    residual_top_score: float | None
    returned_min_score: float | None
    nugget_ids: tuple[str, ...]
    state: DocumentState
    state_reason: str


@dataclass(frozen=True)
class EvidenceCoverageReport:
    needs: tuple[NeedReport, ...]
    facets: tuple[FacetReport, ...]
    nuggets: tuple[NuggetReport, ...]
    actions: tuple[ActionReport, ...]
    searches: tuple[SearchReport, ...]
    documents: tuple[DocumentFocusReport, ...]
    unresolved_need_ids: tuple[str, ...]
    search_count: int
    inspected_page_count: int
    state_version: int
    state_hash: str
    terminal_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GroundedDraftRecovery:
    """Deterministic draft selections recovered from admitted live nuggets."""

    live_grounded_nugget_ids: tuple[str, ...]
    recovered_need_ids: tuple[str, ...]
    selected_nugget_ids: tuple[str, ...]


@dataclass
class _Need:
    need_id: str
    narrative_span: str
    question: str
    status: NeedStatus = "unaddressed"
    remaining_gap: str = ""
    facet_ids: list[str] = field(default_factory=list)
    nugget_ids: list[str] = field(default_factory=list)
    draft_answer: str | None = None
    draft_nugget_ids: list[str] = field(default_factory=list)


@dataclass
class _Facet:
    facet_id: str
    need_ids: list[str]
    dimension: str
    value: str
    origin: Literal["narrative", "snippet"]
    origin_snippet_id: str | None
    status: FacetStatus = "open"
    status_reason: str = ""
    supporting_nugget_ids: list[str] = field(default_factory=list)


@dataclass
class _Nugget:
    nugget_id: str
    text: str
    need_ids: list[str]
    facet_ids: list[str]
    evidence: list[_EvidenceSpan]
    contradicts: list[str]
    superseded_by: str | None = None
    support_ratio: float = 0.0


@dataclass
class _SnippetObservation:
    document_id: str
    snippet_id: str
    page_index: int
    text: str
    focus_query: str
    page_yield_index: int
    handle: str
    sentence_spans: tuple[tuple[int, int], ...]


@dataclass
class _DocumentFocus:
    document_id: str
    focus_query: str
    pages_fetched: list[int] = field(default_factory=list)
    next_page_available: bool = False
    residual_count: int = 0
    residual_top_score: float | None = None
    returned_min_score: float | None = None
    nugget_ids: list[str] = field(default_factory=list)
    recent_yield: bool = False
    state: DocumentState = "unexamined"
    state_reason: str = "not yet inspected"


@dataclass
class _Action:
    action: ActionKind
    target: str
    focus_query: str | None
    motivating_ids: list[str]
    rationale: str
    state: Literal["pending", "consumed", "terminal"]
    research_task_id: str | None = None
    round_index: int | None = None
    depth: Literal["survey", "focused", "deep"] | None = None


@dataclass(frozen=True)
class _Accepted:
    identifier: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _nonblank(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _string_ids(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    items = tuple(item for item in value if _nonblank(item) is not None)
    return items if len(items) == len(value) and len(set(items)) == len(items) else None


# A cited span with fewer than this many content words cannot carry a claim.
# Refusing a citation costs the researcher its claim, so this is set to the
# least aggressive value that still catches the documented harm: the live run
# attached a full Supreme Court holding to "Plyler v." (one content word) and
# other claims to "2.", "B." and "Yes." (zero to one). Two-word spans such as
# "Outside support." are left citable deliberately; the support ratio still
# records when a claim leans on one.
_MIN_CITATION_CONTENT_WORDS = 2

_STOPWORDS = frozenset(
    "the a an and or of to in for on with that this these those is are was were"
    " be been by as at from it its their they them we our you your not no can"
    " may might will would should could have has had do does did than then so"
    " such other some more most many much into over under between about which"
    " who what when where how".split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _content_word_count(text: str) -> int:
    return sum(1 for word in _WORD.findall(text) if word.lower() not in _STOPWORDS)


def claim_support_ratio(claim: str, cited_text: str) -> float:
    """Fraction of the claim's content words that appear in the text it cites.

    A screen, not an entailment proof. It cannot confirm a claim is supported,
    but a claim whose words are almost entirely absent from what it cites is
    drawing on something other than the retrieved evidence, which is the
    failure worth catching: a Supreme Court holding cited to a span reading
    only "Plyler v.".
    """
    words = [word.lower() for word in _WORD.findall(claim)]
    content = [word for word in words if word not in _STOPWORDS]
    if not content:
        return 0.0
    haystack = cited_text.lower()
    return sum(1 for word in content if word in haystack) / len(content)


def _normalised_whitespace(value: str) -> str:
    return " ".join(value.split())


class EvidenceCoverageState:
    """Own mutable grounding state for exactly one retrieval invocation."""

    def __init__(self, narrative: str) -> None:
        if _nonblank(narrative) is None:
            raise ValueError("narrative must be nonblank")
        self._narrative = narrative
        self._lock = Lock()
        self._needs: dict[str, _Need] = {}
        self._facets: dict[str, _Facet] = {}
        self._nuggets: dict[str, _Nugget] = {}
        self._snippets: dict[str, _SnippetObservation] = {}
        self._handles: dict[str, str] = {}
        self._documents: dict[tuple[str, str], _DocumentFocus] = {}
        self._searches: list[SearchReport] = []
        self._actions: list[_Action] = []
        self._page_yields: list[bool] = []
        self._state_version = 0
        self._terminal_reason: str | None = None

    def _changed(self) -> None:
        self._state_version += 1

    def _hash(self) -> str:
        payload = self._report_payload(state_hash="")
        payload.pop("state_hash")
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _report_payload(self, *, state_hash: str) -> dict[str, object]:
        needs = [
            asdict(
                NeedReport(
                    item.need_id,
                    item.narrative_span,
                    item.question,
                    item.status,
                    item.remaining_gap,
                    tuple(item.facet_ids),
                    tuple(item.nugget_ids),
                    item.draft_answer,
                    tuple(item.draft_nugget_ids),
                )
            )
            for item in self._needs.values()
        ]
        facets = [
            asdict(
                FacetReport(
                    item.facet_id,
                    tuple(item.need_ids),
                    item.dimension,
                    item.value,
                    item.origin,
                    item.origin_snippet_id,
                    item.status,
                    item.status_reason,
                    tuple(item.supporting_nugget_ids),
                )
            )
            for item in self._facets.values()
        ]
        nuggets = [asdict(self._nugget_report(item)) for item in self._nuggets.values()]
        actions = [
            asdict(
                ActionReport(
                    item.action,
                    item.target,
                    item.focus_query,
                    tuple(item.motivating_ids),
                    item.rationale,
                    item.state,
                    item.research_task_id,
                    item.round_index,
                    item.depth,
                )
            )
            for item in self._actions
        ]
        documents = [
            asdict(self._document_report(item)) for item in self._documents.values()
        ]
        unresolved = [
            item.need_id
            for item in self._needs.values()
            if item.status in {"unaddressed", "partial", "conflicted"}
        ]
        return {
            "needs": needs,
            "facets": facets,
            "nuggets": nuggets,
            "actions": actions,
            "searches": [asdict(item) for item in self._searches],
            "documents": documents,
            "unresolved_need_ids": unresolved,
            "search_count": len(self._searches),
            "inspected_page_count": len(self._page_yields),
            "state_version": self._state_version,
            "state_hash": state_hash,
            "terminal_reason": self._terminal_reason,
        }

    def _nugget_report(self, item: _Nugget) -> NuggetReport:
        references = tuple(self._evidence_reference(span) for span in item.evidence)
        support = (
            "multi_document"
            if len({row.document_id for row in references}) > 1
            else "single_document"
        )
        return NuggetReport(
            item.nugget_id,
            item.text,
            tuple(item.need_ids),
            tuple(item.facet_ids),
            references,
            tuple(item.contradicts),
            support,
            item.superseded_by,
            # Derived, never written. A nugget is vital exactly when the
            # coordinator selected it into a need's bounded draft set, so
            # the label cannot be inflated by claiming it.
            "vital" if item.nugget_id in self._drafted_nugget_ids() else "okay",
            item.support_ratio,
        )

    def _drafted_nugget_ids(self) -> frozenset[str]:
        """Nuggets the coordinator selected into some need's bounded draft set.

        This is the importance signal. It is a selection rather than a label,
        so it cannot be inflated: the draft set is capped per need, and an
        earlier design where an agent asserted its own importance collapsed to
        94% vital on a live run.
        """
        return frozenset(
            nugget_id
            for need in self._needs.values()
            for nugget_id in need.draft_nugget_ids
        )

    def _document_report(self, item: _DocumentFocus) -> DocumentFocusReport:
        return DocumentFocusReport(
            item.document_id,
            item.focus_query,
            tuple(item.pages_fetched),
            item.next_page_available,
            item.residual_count,
            item.residual_top_score,
            item.returned_min_score,
            tuple(item.nugget_ids),
            item.state,
            item.state_reason,
        )

    def report(self) -> EvidenceCoverageReport:
        with self._lock:
            return EvidenceCoverageReport(
                needs=tuple(self._need_report(item) for item in self._needs.values()),
                facets=tuple(
                    FacetReport(
                        item.facet_id,
                        tuple(item.need_ids),
                        item.dimension,
                        item.value,
                        item.origin,
                        item.origin_snippet_id,
                        item.status,
                        item.status_reason,
                        tuple(item.supporting_nugget_ids),
                    )
                    for item in self._facets.values()
                ),
                nuggets=tuple(self._nugget_report(item) for item in self._nuggets.values()),
                actions=tuple(
                    ActionReport(
                        item.action,
                        item.target,
                        item.focus_query,
                        tuple(item.motivating_ids),
                        item.rationale,
                        item.state,
                        item.research_task_id,
                        item.round_index,
                        item.depth,
                    )
                    for item in self._actions
                ),
                searches=tuple(self._searches),
                documents=tuple(self._document_report(item) for item in self._documents.values()),
                unresolved_need_ids=tuple(
                    item.need_id
                    for item in self._needs.values()
                    if item.status in {"unaddressed", "partial", "conflicted"}
                ),
                search_count=len(self._searches),
                inspected_page_count=len(self._page_yields),
                state_version=self._state_version,
                state_hash=self._hash(),
                terminal_reason=self._terminal_reason,
            )

    def _pending_closeout_need_ids_locked(self) -> tuple[str, ...]:
        live_nugget_ids = frozenset(
            item.nugget_id
            for item in self._nuggets.values()
            if item.superseded_by is None
        )
        return tuple(
            need.need_id
            for need in self._needs.values()
            if any(nugget_id in live_nugget_ids for nugget_id in need.nugget_ids)
            and (
                not need.draft_nugget_ids
                or any(
                    nugget_id not in live_nugget_ids
                    or need.need_id not in self._nuggets[nugget_id].need_ids
                    for nugget_id in need.draft_nugget_ids
                )
            )
        )

    def pending_closeout_need_ids(self) -> tuple[str, ...]:
        """Return needs with live evidence but no selected live draft nuggets."""
        with self._lock:
            return self._pending_closeout_need_ids_locked()

    def recover_grounded_drafts(self) -> GroundedDraftRecovery:
        """Select only already-admitted live nuggets for undrafted needs.

        This is the deterministic final-synthesis fallback. It has no access to
        raw search passages or model text, so it cannot widen the grounded
        evidence set. Need and nugget insertion order are both preserved.
        """
        with self._lock:
            live_nuggets = {
                nugget.nugget_id: nugget
                for nugget in self._nuggets.values()
                if nugget.superseded_by is None and bool(nugget.evidence)
            }
            recovered_need_ids: list[str] = []
            selected_nugget_ids: list[str] = []
            for need in self._needs.values():
                existing = tuple(
                    nugget_id
                    for nugget_id in need.draft_nugget_ids
                    if nugget_id in live_nuggets
                    and need.need_id in live_nuggets[nugget_id].need_ids
                )
                if existing and len(existing) == len(need.draft_nugget_ids):
                    continue
                selected = tuple(
                    nugget_id
                    for nugget_id in need.nugget_ids
                    if nugget_id in live_nuggets
                    and need.need_id in live_nuggets[nugget_id].need_ids
                )[:MAX_DRAFT_NUGGETS_PER_NEED]
                if not selected:
                    continue
                outcome = self._set_need_status(
                    {
                        "need_id": need.need_id,
                        "status": "partial",
                        "remaining_gap": need.remaining_gap
                        or "Final synthesis did not complete; grounded evidence retained.",
                        "draft_nugget_ids": list(selected),
                    }
                )
                if isinstance(outcome, str):  # guarded by the live selection above
                    raise RuntimeError(
                        f"grounded draft recovery violated state invariants: {outcome}"
                    )
                recovered_need_ids.append(need.need_id)
                selected_nugget_ids.extend(selected)
            if recovered_need_ids:
                self._changed()
            return GroundedDraftRecovery(
                live_grounded_nugget_ids=tuple(live_nuggets),
                recovered_need_ids=tuple(recovered_need_ids),
                selected_nugget_ids=tuple(selected_nugget_ids),
            )

    def record_search(
        self,
        *,
        query: str,
        kind: Literal["original", "followup"],
        documents: Sequence[DocumentObservation],
    ) -> None:
        if _nonblank(query) is None or kind not in {"original", "followup"}:
            raise ValueError("search observation is invalid")
        if any(
            _nonblank(item.document_id) is None
            or isinstance(item.rank, bool)
            or not isinstance(item.rank, int)
            or item.rank < 1
            for item in documents
        ):
            raise ValueError("document observation is invalid")
        with self._lock:
            self._searches.append(SearchReport(query, kind, tuple(item.document_id for item in documents)))
            self._changed()

    def record_snippet_page(self, page: SnippetPage) -> tuple[SnippetHandle, ...]:
        """Observe one page and return how an agent may cite each snippet.

        Handles are assigned monotonically and scoped to the whole invocation,
        so pagination never reuses one and concurrent researchers never collide.
        """
        with self._lock:
            yield_index = len(self._page_yields)
            self._page_yields.append(False)
            focus = self._documents.setdefault(
                (page.document_id, page.focus_query),
                _DocumentFocus(page.document_id, page.focus_query),
            )
            if page.page_index not in focus.pages_fetched:
                focus.pages_fetched.append(page.page_index)
            focus.next_page_available = page.next_cursor is not None
            focus.residual_count = page.residual_count
            focus.residual_top_score = page.residual_top_score
            focus.returned_min_score = page.returned_min_score
            focus.recent_yield = False
            if page.snippets:
                focus.state = "productive"
                focus.state_reason = "returned snippets"
            elif page.next_cursor is None:
                focus.state = "exhausted"
                focus.state_reason = "no snippets or next page remain"
            observed: list[SnippetHandle] = []
            for snippet in page.snippets:
                existing = self._snippets.get(snippet.chunk_id)
                if existing is None:
                    spans = _sentence_spans(snippet.text)
                    handle = f"S{len(self._handles) + 1}"
                    self._handles[handle] = snippet.chunk_id
                    self._snippets[snippet.chunk_id] = _SnippetObservation(
                        page.document_id,
                        snippet.chunk_id,
                        page.page_index,
                        snippet.text,
                        page.focus_query,
                        yield_index,
                        handle,
                        spans,
                    )
                else:
                    # A re-observed snippet keeps its original handle so an
                    # earlier citation never changes meaning mid-invocation.
                    handle = existing.handle
                    spans = existing.sentence_spans
                observation = self._snippets[snippet.chunk_id]
                observed.append(
                    SnippetHandle(
                        handle,
                        snippet.chunk_id,
                        snippet.relevance_score,
                        tuple(
                            observation.text[start:end] for start, end in spans
                        ),
                    )
                )
            self._changed()
            return tuple(observed)

    def _rejection(self, section: str, index: int, code: str) -> DeltaRejection:
        return DeltaRejection(section, index, code)

    def _resolve_citations(self, value: object) -> tuple[_EvidenceSpan, ...] | str:
        """Resolve agent citations such as ``S3``, ``S3.2``, or ``S3.2-4``.

        Nothing the agent writes becomes evidence text. A citation either names
        stored sentences or it is rejected, so an ungrounded quote is not a
        reachable outcome.
        """
        if not isinstance(value, (list, tuple)) or not value:
            return "INVALID_CITATION"
        spans: list[_EvidenceSpan] = []
        for row in value:
            if not isinstance(row, Mapping):
                return "INVALID_CITATION"
            cite = _nonblank(row.get("cite"))
            if cite is None:
                return "INVALID_CITATION"
            match = _CITATION.match(cite.strip())
            if match is None:
                return "INVALID_CITATION"
            snippet_id = self._handles.get(f"S{int(match.group(1))}")
            if snippet_id is None:
                return "UNKNOWN_CITATION"
            observation = self._snippets[snippet_id]
            sentence_count = len(observation.sentence_spans)
            if match.group(2) is None:
                first, last = 1, sentence_count
            else:
                first = int(match.group(2))
                last = int(match.group(3)) if match.group(3) is not None else first
            if first < 1 or last < first or last > sentence_count:
                return "INVALID_CITATION"
            # A citation that resolves is not the same as a citation that
            # supports. The sentence splitter breaks on list markers and on
            # abbreviations such as the "v." in a case name, producing spans
            # like "2.", "B." or "Plyler v." that are perfectly citable and
            # carry nothing. Measured live: 12 of 548 cited spans were under
            # five words, and one nugget attached a full Supreme Court holding
            # to the two-word span "Plyler v.".
            if _content_word_count(self._span_text(snippet_id, first, last)) < _MIN_CITATION_CONTENT_WORDS:
                return "DEGENERATE_CITATION"
            spans.append(_EvidenceSpan(snippet_id, first, last))
        return tuple(spans)

    def _span_text(self, snippet_id: str, first: int, last: int) -> str:
        observation = self._snippets[snippet_id]
        selected = observation.sentence_spans[first - 1 : last]
        return " ".join(observation.text[start:end] for start, end in selected)

    def _evidence_reference(self, span: _EvidenceSpan) -> EvidenceReference:
        """Derive the reported quote from stored sentences, never from an agent."""
        observation = self._snippets[span.snippet_id]
        selected = observation.sentence_spans[span.first_sentence - 1 : span.last_sentence]
        quote = " ".join(
            observation.text[start:end] for start, end in selected
        )
        return EvidenceReference(
            observation.document_id,
            span.snippet_id,
            observation.page_index,
            quote,
        )

    def _record_grounded_yield(self, evidence: Sequence[_EvidenceSpan], nugget_id: str) -> None:
        for reference in evidence:
            snippet = self._snippets[reference.snippet_id]
            self._page_yields[snippet.page_yield_index] = True
            focus = self._documents[(snippet.document_id, snippet.focus_query)]
            focus.recent_yield = True
            if nugget_id not in focus.nugget_ids:
                focus.nugget_ids.append(nugget_id)

    def apply_delta(self, delta: Mapping[str, object]) -> StateUpdateResult:
        accepted: list[str] = []
        rejected: list[DeltaRejection] = []
        with self._lock:
            if not isinstance(delta, Mapping):
                return StateUpdateResult((), (self._rejection("delta", 0, "INVALID_DELTA"),), self._state_version, self._hash())
            unknown_sections = [
                section
                for section in delta
                if not isinstance(section, str) or section not in _DELTA_SECTIONS
            ]
            if unknown_sections:
                rejected.extend(
                    self._rejection(str(section), 0, "UNKNOWN_SECTION")
                    for section in unknown_sections
                )
            if not delta:
                return StateUpdateResult(
                    (),
                    (self._rejection("delta", 0, "EMPTY_DELTA"),),
                    self._state_version,
                    self._hash(),
                )
            for section in _DELTA_SECTIONS:
                rows = delta.get(section, [])
                if not isinstance(rows, (list, tuple)):
                    rejected.append(self._rejection(section, 0, "INVALID_SECTION"))
                    continue
                for index, row in enumerate(rows):
                    if not isinstance(row, Mapping):
                        rejected.append(self._rejection(section, index, "INVALID_ITEM"))
                        continue
                    handler = getattr(self, f"_{section}")
                    outcome = handler(row)
                    if isinstance(outcome, _Accepted):
                        accepted.append(outcome.identifier)
                        self._changed()
                    else:
                        rejected.append(self._rejection(section, index, outcome))
            if not accepted and not rejected:
                rejected.append(self._rejection("delta", 0, "EMPTY_DELTA"))
            return StateUpdateResult(tuple(accepted), tuple(rejected), self._state_version, self._hash())

    def _add_needs(self, row: Mapping[str, object]) -> _Accepted | str:
        need_id = _nonblank(row.get("need_id"))
        span = _nonblank(row.get("narrative_span"))
        question = _nonblank(row.get("question"))
        if need_id is None or span is None or question is None:
            return "INVALID_NEED"
        if need_id in self._needs:
            return "DUPLICATE_ID"
        if span not in self._narrative:
            return "UNGROUNDED_NARRATIVE_SPAN"
        self._needs[need_id] = _Need(need_id, span, question)
        return _Accepted(need_id)

    def _add_facets(self, row: Mapping[str, object]) -> _Accepted | str:
        facet_id = _nonblank(row.get("facet_id"))
        need_ids = _string_ids(row.get("need_ids"))
        dimension = _nonblank(row.get("dimension"))
        value = _nonblank(row.get("value"))
        origin = row.get("origin")
        origin_snippet_id = row.get("origin_snippet_id")
        if facet_id is None or need_ids is None or not need_ids or dimension is None or value is None:
            return "INVALID_FACET"
        if facet_id in self._facets:
            return "DUPLICATE_ID"
        if any(need_id not in self._needs for need_id in need_ids):
            return "UNKNOWN_NEED"
        if origin not in {"narrative", "snippet"}:
            return "INVALID_ORIGIN"
        if origin == "snippet":
            if _nonblank(origin_snippet_id) is None:
                return "MISSING_ORIGIN_SNIPPET"
            if origin_snippet_id not in self._snippets:
                return "UNKNOWN_SNIPPET"
        elif origin_snippet_id is not None:
            return "INVALID_ORIGIN_SNIPPET"
        self._facets[facet_id] = _Facet(
            facet_id, list(need_ids), dimension, value, origin, origin_snippet_id if isinstance(origin_snippet_id, str) else None
        )
        for need_id in need_ids:
            self._needs[need_id].facet_ids.append(facet_id)
        return _Accepted(facet_id)

    def _add_nuggets(self, row: Mapping[str, object]) -> _Accepted | str:
        nugget_id = _nonblank(row.get("nugget_id"))
        text = _nonblank(row.get("text"))
        need_ids = _string_ids(row.get("need_ids"))
        facet_ids = _string_ids(row.get("facet_ids"))
        contradicts = _string_ids(row.get("contradicts", []))
        if nugget_id is None or text is None or need_ids is None or facet_ids is None or contradicts is None:
            return "INVALID_NUGGET"
        if nugget_id in self._nuggets:
            return "DUPLICATE_ID"
        if not need_ids or any(need_id not in self._needs for need_id in need_ids):
            return "UNKNOWN_NEED"
        if any(facet_id not in self._facets for facet_id in facet_ids):
            return "UNKNOWN_FACET"
        if any(other_id not in self._nuggets for other_id in contradicts):
            return "UNKNOWN_NUGGET"
        evidence = self._resolve_citations(row.get("evidence"))
        if isinstance(evidence, str):
            return evidence
        # Recorded, never used to reject. A low ratio means the claim's words
        # are largely absent from what it cites, which is worth surfacing to
        # whatever builds the answer; it is not proof of a bad claim, and
        # discarding grounded evidence on a lexical screen would cost more
        # than it saves.
        support_ratio = claim_support_ratio(
            text,
            " ".join(
                self._span_text(span.snippet_id, span.first_sentence, span.last_sentence)
                for span in evidence
            ),
        )
        self._nuggets[nugget_id] = _Nugget(
            nugget_id,
            text,
            list(need_ids),
            list(facet_ids),
            list(evidence),
            list(contradicts),
            support_ratio=support_ratio,
        )
        for need_id in need_ids:
            self._needs[need_id].nugget_ids.append(nugget_id)
        for other_id in contradicts:
            self._nuggets[other_id].contradicts.append(nugget_id)
        self._record_grounded_yield(evidence, nugget_id)
        return _Accepted(nugget_id)

    def _add_evidence(self, row: Mapping[str, object]) -> _Accepted | str:
        nugget_id = _nonblank(row.get("nugget_id"))
        if nugget_id is None or nugget_id not in self._nuggets:
            return "UNKNOWN_NUGGET"
        evidence = self._resolve_citations([row])
        if isinstance(evidence, str):
            return evidence
        nugget = self._nuggets[nugget_id]
        new_evidence = tuple(
            reference for reference in evidence if reference not in nugget.evidence
        )
        if not new_evidence:
            return "DUPLICATE_EVIDENCE"
        nugget.evidence.extend(new_evidence)
        return _Accepted(nugget_id)

    def _set_facet_status(self, row: Mapping[str, object]) -> _Accepted | str:
        facet_id = _nonblank(row.get("facet_id"))
        status = row.get("status")
        reason = row.get("status_reason")
        nugget_ids = _string_ids(row.get("supporting_nugget_ids", []))
        if facet_id is None or facet_id not in self._facets:
            return "UNKNOWN_FACET"
        if status not in {"open", "covered", "dropped"} or nugget_ids is None:
            return "INVALID_FACET_STATUS"
        facet = self._facets[facet_id]
        if status == "covered":
            if not nugget_ids or any(
                nugget_id not in self._nuggets
                or facet_id not in self._nuggets[nugget_id].facet_ids
                or not self._nuggets[nugget_id].evidence
                for nugget_id in nugget_ids
            ):
                return "MISSING_GROUNDED_NUGGET"
        if status == "dropped" and _nonblank(reason) is None:
            return "MISSING_STATUS_REASON"
        facet.status = status
        facet.status_reason = reason if isinstance(reason, str) else ""
        facet.supporting_nugget_ids = list(nugget_ids)
        return _Accepted(facet_id)

    def _set_need_status(self, row: Mapping[str, object]) -> _Accepted | str:
        need_id = _nonblank(row.get("need_id"))
        status = row.get("status")
        remaining_gap = row.get("remaining_gap")
        draft_answer = row.get("draft_answer")
        draft_nugget_ids = _string_ids(row.get("draft_nugget_ids", []))
        if need_id is None or need_id not in self._needs:
            return "UNKNOWN_NEED"
        if status not in {"unaddressed", "partial", "answerable", "conflicted"} or not isinstance(remaining_gap, str) or draft_nugget_ids is None:
            return "INVALID_NEED_STATUS"
        # Selection has to select. Unbounded, a live run drafted all 145 of its
        # nuggets, which made drafted_count equal nugget_count for every
        # document and collapsed the submission ranker's highest-weighted
        # feature into its lowest. The cap is deliberately a fixed number and
        # not derived from the evidence: a derived threshold would launder the
        # ranking judgement away again, and tuning one against development
        # qrels is unsound when most retrieved documents are unjudged.
        if len(draft_nugget_ids) > MAX_DRAFT_NUGGETS_PER_NEED:
            return "TOO_MANY_DRAFT_NUGGETS"
        grounded = all(
            nugget_id in self._nuggets
            and need_id in self._nuggets[nugget_id].need_ids
            and bool(self._nuggets[nugget_id].evidence)
            # A superseded claim was replaced, and the submission ranker
            # already refuses to let it vouch for its documents. Drafting one
            # would cite a claim whose documents the run never submits.
            and self._nuggets[nugget_id].superseded_by is None
            for nugget_id in draft_nugget_ids
        )
        # Grounding is checked for every status that supplies a draft, not just
        # the terminal ones. Selection is the importance signal, so an
        # unchecked "partial" row could list nugget ids that do not exist yet
        # and mint vital labels for them the moment they arrive - inflating the
        # submission ranker's highest-weighted feature without any real
        # selection having happened.
        if draft_nugget_ids and not grounded:
            return "MISSING_GROUNDED_DRAFT"
        if status == "answerable" and (_nonblank(draft_answer) is None or not draft_nugget_ids or not grounded):
            return "MISSING_GROUNDED_DRAFT"
        if status == "conflicted":
            linked = any(
                left in self._nuggets
                and right in self._nuggets[left].contradicts
                for left in draft_nugget_ids
                for right in draft_nugget_ids
                if left != right
            )
            if len(draft_nugget_ids) < 2 or not grounded or not linked:
                return "MISSING_CONTRADICTION_LINK"
        need = self._needs[need_id]
        need.status = status
        need.remaining_gap = remaining_gap
        need.draft_answer = draft_answer if isinstance(draft_answer, str) else None
        need.draft_nugget_ids = list(draft_nugget_ids)
        return _Accepted(need_id)

    def _supersede_nuggets(self, row: Mapping[str, object]) -> _Accepted | str:
        nugget_id = _nonblank(row.get("nugget_id"))
        superseded_by = _nonblank(row.get("superseded_by"))
        if nugget_id is None or superseded_by is None or nugget_id not in self._nuggets or superseded_by not in self._nuggets:
            return "UNKNOWN_NUGGET"
        if nugget_id == superseded_by:
            return "INVALID_SUPERSESSION"
        self._nuggets[nugget_id].superseded_by = superseded_by
        return _Accepted(nugget_id)

    def _abandon_documents(self, row: Mapping[str, object]) -> _Accepted | str:
        document_id = _nonblank(row.get("document_id"))
        reason = _nonblank(row.get("reason"))
        if document_id is None or reason is None:
            return "INVALID_ABANDONMENT"
        focuses = [item for item in self._documents.values() if item.document_id == document_id]
        if not focuses:
            return "UNKNOWN_DOCUMENT"
        for focus in focuses:
            focus.state = "abandoned"
            focus.state_reason = reason
        return _Accepted(document_id)

    def view(self, scope: str) -> str:
        with self._lock:
            if scope == "frontier":
                payload = {
                    "needs": [
                        {
                            "need_id": item.need_id,
                            "status": item.status,
                            "remaining_gap": item.remaining_gap,
                            # Counted from the ledger, not from the status an
                            # agent set, so an uncovered need is visible even
                            # when its status is stale or wrong.
                            "grounded_nugget_count": len(item.nugget_ids),
                        }
                        for item in self._needs.values()
                        if item.status in {"unaddressed", "partial", "conflicted"}
                    ],
                    "open_facets": [
                        {"facet_id": item.facet_id, "need_ids": item.need_ids, "dimension": item.dimension, "value": item.value}
                        for item in self._facets.values()
                        if item.status == "open"
                    ],
                    "documents": [
                        {
                            "document_id": item.document_id,
                            "focus_query": item.focus_query,
                            "residual_count": item.residual_count,
                            "residual_top_score": item.residual_top_score,
                            "returned_min_score": item.returned_min_score,
                            "next_page_available": item.next_page_available,
                            "recent_yield": item.recent_yield,
                        }
                        for item in self._documents.values()
                    ],
                    "state_hash": self._hash(),
                }
                encoded = _canonical_json(payload)
                if len(encoded) > MAX_FRONTIER_CHARACTERS:
                    encoded = _canonical_json({"code": "FRONTIER_TOO_LARGE", "state_hash": self._hash()})
                return encoded
            if scope.startswith("need:"):
                need_id = scope.removeprefix("need:")
                need = self._needs.get(need_id)
                if need is None:
                    return _canonical_json({"code": "UNKNOWN_SCOPE"})
                nuggets = [self._nugget_report(self._nuggets[item]) for item in need.nugget_ids]
                return _canonical_json({"need": asdict(self._need_report(need)), "nuggets": [asdict(item) for item in nuggets], "state_hash": self._hash()})
            return _canonical_json({"code": "UNKNOWN_SCOPE"})

    def _need_report(self, item: _Need) -> NeedReport:
        return NeedReport(item.need_id, item.narrative_span, item.question, item.status, item.remaining_gap, tuple(item.facet_ids), tuple(item.nugget_ids), item.draft_answer, tuple(item.draft_nugget_ids))

    def _open_motivation(self, item_id: str) -> str | None:
        need = self._needs.get(item_id)
        if need is not None:
            return None if need.status in {"unaddressed", "partial", "conflicted"} else "CLOSED_MOTIVATION"
        facet = self._facets.get(item_id)
        if facet is not None:
            return None if facet.status == "open" else "CLOSED_MOTIVATION"
        return "UNKNOWN_MOTIVATION"

    def _action_error(
        self, code: str, *, need_ids: Sequence[str] = ()
    ) -> Mapping[str, object]:
        result: dict[str, object] = {
            "ok": False,
            "code": code,
            "state_version": self._state_version,
            "state_hash": self._hash(),
        }
        if need_ids:
            result["need_ids"] = list(need_ids)
        return result

    def record_retrieval_action(
        self,
        *,
        action: Literal["search", "extract", "paginate", "refocus"],
        target: str,
        focus_query: str | None,
        motivating_ids: Sequence[str],
        rationale: str,
        context: ResearchTaskContext,
    ) -> str | None:
        """Append one consumed action using the fixed tool's actual arguments."""

        with self._lock:
            if self._terminal_reason is not None:
                return "TERMINAL_STATE"
            if action not in {"search", "extract", "paginate", "refocus"}:
                return "INVALID_ACTION"
            if _nonblank(target) is None or _nonblank(rationale) is None:
                return "INVALID_ACTION"
            if action == "search":
                if focus_query is not None:
                    return "INVALID_ACTION"
            elif _nonblank(focus_query) is None:
                return "INVALID_ACTION"

            ids = _string_ids(motivating_ids)
            if ids is None:
                return "INVALID_MOTIVATION"
            if not ids:
                return "MISSING_MOTIVATION"
            for item_id in ids:
                error = self._open_motivation(item_id)
                if error is not None:
                    return error

            context_ids = _string_ids(context.motivating_ids)
            if (
                _nonblank(context.research_task_id) is None
                or isinstance(context.round_index, bool)
                or not isinstance(context.round_index, int)
                or context.round_index < 1
                or context.depth not in {"survey", "focused", "deep"}
                or context_ids is None
                or not context_ids
                or any(item_id not in context_ids for item_id in ids)
            ):
                return "CONTEXT_MISMATCH"

            self._actions.append(
                _Action(
                    action,
                    target,
                    focus_query,
                    list(ids),
                    rationale,
                    "consumed",
                    context.research_task_id,
                    context.round_index,
                    context.depth,
                )
            )
            self._changed()
            return None

    def _choose_action_locked(
        self,
        *,
        action: ActionKind,
        target: str,
        focus_query: str | None,
        motivating_ids: Sequence[str],
        rationale: str,
    ) -> Mapping[str, object]:
        if self._terminal_reason is not None:
            return self._action_error("TERMINAL_STATE")
        if (
            action not in {"search", "extract", "paginate", "refocus", "stop"}
            or _nonblank(target) is None
            or _nonblank(rationale) is None
        ):
            return self._action_error("INVALID_ACTION")
        ids = _string_ids(motivating_ids)
        if ids is None:
            return self._action_error("INVALID_MOTIVATION")
        if action != "stop":
            if not ids:
                return self._action_error("MISSING_MOTIVATION")
            for item_id in ids:
                error = self._open_motivation(item_id)
                if error is not None:
                    return self._action_error(error)
            if any(item.state == "pending" for item in self._actions):
                return self._action_error("PENDING_ACTION_EXISTS")
            self._actions.append(
                _Action(action, target, focus_query, list(ids), rationale, "pending")
            )
            self._changed()
            return {
                "ok": True,
                "action": action,
                "state": "pending",
                "state_version": self._state_version,
                "state_hash": self._hash(),
            }
        if any(item.state == "pending" for item in self._actions):
            return self._action_error("PENDING_ACTION_EXISTS")
        if target == "completion":
            open_need_ids = tuple(
                item.need_id
                for item in self._needs.values()
                if self._open_motivation(item.need_id) is None
            )
            if open_need_ids:
                return self._action_error(
                    "COMPLETION_OPEN_NEEDS", need_ids=open_need_ids
                )
            if ids:
                return self._action_error("COMPLETION_HAS_MOTIVATION")
        elif target == "saturation":
            if not ids:
                return self._action_error("MISSING_MOTIVATION")
            for item_id in ids:
                error = self._open_motivation(item_id)
                if error is not None:
                    return self._action_error(error)
            if self._zero_yield_pages() < SATURATION_ZERO_YIELD_PAGES:
                return self._action_error("INSUFFICIENT_ZERO_YIELD_PAGES")
        else:
            return self._action_error("INVALID_STOP_TARGET")
        self._actions.append(
            _Action("stop", target, focus_query, list(ids), rationale, "terminal")
        )
        self._terminal_reason = target
        self._changed()
        return {
            "ok": True,
            "action": "stop",
            "state": "terminal",
            "state_version": self._state_version,
            "state_hash": self._hash(),
        }

    def choose_action(
        self,
        *,
        action: ActionKind,
        target: str,
        focus_query: str | None,
        motivating_ids: Sequence[str],
        rationale: str,
    ) -> Mapping[str, object]:
        """Validate, append, and expose one pending action or terminal stop."""
        with self._lock:
            return self._choose_action_locked(
                action=action,
                target=target,
                focus_query=focus_query,
                motivating_ids=motivating_ids,
                rationale=rationale,
            )

    def complete_retrieval(self) -> Mapping[str, object]:
        """Atomically validate draft coverage and record completion."""
        with self._lock:
            pending_need_ids = self._pending_closeout_need_ids_locked()
            if pending_need_ids:
                return self._action_error(
                    "INCOMPLETE_CLOSEOUT", need_ids=pending_need_ids
                )
            result = self._choose_action_locked(
                action="stop",
                target="completion",
                focus_query=None,
                motivating_ids=[],
                rationale="explicit retrieval completion",
            )
            if not result.get("ok", False) and "need_ids" not in result:
                open_need_ids = tuple(
                    item.need_id
                    for item in self._needs.values()
                    if self._open_motivation(item.need_id) is None
                )
                result = {**result, "need_ids": list(open_need_ids)}
            return result

    def _zero_yield_pages(self) -> int:
        count = 0
        for yielded in reversed(self._page_yields):
            if yielded:
                break
            count += 1
        return count

    def require_pending_action(
        self,
        *,
        action: Literal["search", "extract", "paginate", "refocus"],
        target: str,
        focus_query: str | None,
    ) -> Mapping[str, object] | None:
        """Consume a matching pending action or return a safe tool error."""
        with self._lock:
            if self._terminal_reason is not None:
                return self._action_error("TERMINAL_STATE")
            pending = next((item for item in self._actions if item.state == "pending"), None)
            if pending is None:
                return self._action_error("PENDING_ACTION_REQUIRED")
            if (pending.action, pending.target, pending.focus_query) != (action, target, focus_query):
                return self._action_error("ACTION_MISMATCH")
            pending.state = "consumed"
            self._changed()
            return None

    def expected_snippet_action(
        self, *, document_id: str, focus_query: str, cursor: str | None
    ) -> Literal["extract", "paginate", "refocus"] | None:
        with self._lock:
            if cursor is not None:
                return "paginate" if (document_id, focus_query) in self._documents else None
            if (document_id, focus_query) in self._documents:
                return "extract"
            if any(key[0] == document_id for key in self._documents):
                return "refocus"
            return "extract"
