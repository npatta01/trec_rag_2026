"""Invocation-local, mechanically grounded evidence and coverage state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from threading import Lock
from typing import Literal, NotRequired, TypedDict

from trec_rag.deepagent_snippets import SnippetPage


NeedStatus = Literal["unaddressed", "partial", "answerable", "conflicted"]
FacetStatus = Literal["open", "covered", "dropped"]
DocumentState = Literal["unexamined", "productive", "exhausted", "abandoned"]
ActionKind = Literal["search", "extract", "paginate", "refocus", "stop"]

MAX_FRONTIER_CHARACTERS = 2_000
SATURATION_ZERO_YIELD_PAGES = 3
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
    snippet_id: str
    quote: str


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
    status: Literal["open", "covered", "dropped"]
    status_reason: NotRequired[str]
    supporting_nugget_ids: NotRequired[list[str]]


class NeedStatusDelta(TypedDict):
    need_id: str
    status: NeedStatus
    remaining_gap: str
    draft_answer: NotRequired[str]
    draft_nugget_ids: NotRequired[list[str]]


class SupersedeNuggetDelta(TypedDict):
    nugget_id: str
    superseded_by: str


class AbandonDocumentDelta(TypedDict):
    document_id: str
    reason: str


class RetrievalStateDelta(TypedDict, total=False):
    """The model-facing delta accepted by :meth:`EvidenceCoverageState.apply_delta`."""

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


@dataclass(frozen=True)
class ActionReport:
    action: ActionKind
    target: str
    focus_query: str | None
    motivating_ids: tuple[str, ...]
    rationale: str
    state: Literal["pending", "consumed", "terminal"]


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
    evidence: list[EvidenceReference]
    contradicts: list[str]
    superseded_by: str | None = None


@dataclass
class _SnippetObservation:
    document_id: str
    snippet_id: str
    page_index: int
    text: str
    focus_query: str
    page_yield_index: int


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
        support = "multi_document" if len({row.document_id for row in item.evidence}) > 1 else "single_document"
        return NuggetReport(
            item.nugget_id,
            item.text,
            tuple(item.need_ids),
            tuple(item.facet_ids),
            tuple(item.evidence),
            tuple(item.contradicts),
            support,
            item.superseded_by,
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

    def record_snippet_page(self, page: SnippetPage) -> None:
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
            for snippet in page.snippets:
                self._snippets[snippet.chunk_id] = _SnippetObservation(
                    page.document_id,
                    snippet.chunk_id,
                    page.page_index,
                    snippet.text,
                    page.focus_query,
                    yield_index,
                )
            self._changed()

    def _rejection(self, section: str, index: int, code: str) -> DeltaRejection:
        return DeltaRejection(section, index, code)

    def _ground_evidence(self, value: object) -> tuple[EvidenceReference, ...] | str:
        if not isinstance(value, (list, tuple)) or not value:
            return "MISSING_EVIDENCE"
        references: list[EvidenceReference] = []
        for row in value:
            if not isinstance(row, Mapping):
                return "INVALID_EVIDENCE"
            snippet_id = _nonblank(row.get("snippet_id"))
            quote = _nonblank(row.get("quote"))
            if snippet_id is None or quote is None:
                return "INVALID_EVIDENCE"
            snippet = self._snippets.get(snippet_id)
            if snippet is None:
                return "UNKNOWN_SNIPPET"
            if _normalised_whitespace(quote) not in _normalised_whitespace(snippet.text):
                return "UNGROUNDED_QUOTE"
            references.append(
                EvidenceReference(snippet.document_id, snippet_id, snippet.page_index, quote)
            )
        return tuple(references)

    def _record_grounded_yield(self, evidence: Sequence[EvidenceReference], nugget_id: str) -> None:
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
        evidence = self._ground_evidence(row.get("evidence"))
        if isinstance(evidence, str):
            return evidence
        self._nuggets[nugget_id] = _Nugget(nugget_id, text, list(need_ids), list(facet_ids), list(evidence), list(contradicts))
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
        evidence = self._ground_evidence([row])
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
        grounded = all(
            nugget_id in self._nuggets
            and need_id in self._nuggets[nugget_id].need_ids
            and bool(self._nuggets[nugget_id].evidence)
            for nugget_id in draft_nugget_ids
        )
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
                        {"need_id": item.need_id, "status": item.status, "remaining_gap": item.remaining_gap}
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

    def _action_error(self, code: str) -> Mapping[str, object]:
        return {"ok": False, "code": code, "state_version": self._state_version, "state_hash": self._hash()}

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
            if self._terminal_reason is not None:
                return self._action_error("TERMINAL_STATE")
            if action not in {"search", "extract", "paginate", "refocus", "stop"} or _nonblank(target) is None or _nonblank(rationale) is None:
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
                self._actions.append(_Action(action, target, focus_query, list(ids), rationale, "pending"))
                self._changed()
                return {"ok": True, "action": action, "state": "pending", "state_version": self._state_version, "state_hash": self._hash()}
            if any(item.state == "pending" for item in self._actions):
                return self._action_error("PENDING_ACTION_EXISTS")
            if target == "completion":
                if any(self._open_motivation(item.need_id) is None for item in self._needs.values()):
                    return self._action_error("COMPLETION_OPEN_NEEDS")
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
            self._actions.append(_Action("stop", target, focus_query, list(ids), rationale, "terminal"))
            self._terminal_reason = target
            self._changed()
            return {"ok": True, "action": "stop", "state": "terminal", "state_version": self._state_version, "state_hash": self._hash()}

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
