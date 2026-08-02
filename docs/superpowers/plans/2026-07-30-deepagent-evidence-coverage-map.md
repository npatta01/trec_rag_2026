# Deep Agent Evidence/Coverage Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add invocation-local need, nugget, and mechanical retrieval state that makes every Deep Agent retrieval action explainable in terms of narrative coverage and gives the model useful pagination signals.

**Architecture:** Keep three logical stores inside one invocation-local `EvidenceCoverageState`: a model-authored need map, a mechanically appended retrieval ledger, and a model-authored but snippet-grounded nugget store. Add compact view/update/action tools to the existing retrieval-only Deep Agent, require a recorded action before every agent-triggered retrieval call, and return an immutable final coverage report. Keep snippet ranking/cache ownership in `deepagent_snippets` and add residual metadata derived from the same stable per-document/focus ranking.

**Tech Stack:** Python 3.12, LangChain Deep Agents 0.7, LangChain OpenRouter, dataclasses, existing Mixedbread snippet ranker/cache, OpenTelemetry/OpenInference/Phoenix, pytest, Ruff.

## Global Constraints

- Preserve the untouched narrative and the mandatory original ClimbMix search.
- Preserve the existing configured follow-up-search bound.
- Keep snippet page size as a fixed SDK configuration value; the agent cannot set it.
- Add no global cap on inspected documents, snippet pages, snippets, or total agent actions.
- Compare snippet scores only within one `(document_id, focus_query)` ranking.
- Keep search and snippet tools cache-first; expose no cache controls, paths, or keys to the model.
- Use no title, publisher, domain, or independent-source claim because ClimbMix does not return that metadata.
- Report only `single_document` versus `multi_document` nugget support.
- Keep all semantic state invocation-local and separate from optional scratch spill.
- Let deterministic code validate references, exact-quote grounding, counts, and cursor facts; never let it infer semantic equivalence or coverage.
- Require a grounded draft answer before the model may mark a need `answerable`.
- Keep verification POC-sized and targeted; do not add a production-hardening matrix.

---

## File Structure

- Modify `code/trec_rag/deepagent_snippets.py`: add residual pagination metadata to typed pages, cache serialization, and validation.
- Create `code/trec_rag/deepagent_evidence.py`: own all need/facet/nugget/retrieval/action state, delta validation, compact projections, and immutable reports.
- Modify `code/trec_rag/deepagent_retrieval.py`: construct the per-call state, expose five retrieval/state tools, enforce recorded actions, and return the coverage report.
- Modify `code/trec_rag/deepagent_tracing.py`: trace residual page signals and bounded final coverage counts/hash.
- Modify `code/trec_rag/README.md`: document the state model, tool protocol, pagination signal, and result report.
- Modify `code/tests/test_deepagent_snippets.py`: verify enriched pagination and cache integrity.
- Create `code/tests/test_deepagent_evidence.py`: verify grounding, coverage, multi-document support, projections, actions, and reports.
- Modify `code/tests/test_deepagent_retrieval.py`: verify tool integration, action enforcement, prompt behavior, and returned coverage.
- Modify `code/tests/test_deepagent_tracing.py`: verify safe bounded coverage and residual trace attributes.

---

### Task 1: Add Decision-Relevant Snippet Pagination Signals

**Files:**
- Modify: `code/trec_rag/deepagent_snippets.py:32-151,602-799,863-918,1024-1049,1106-1181`
- Test: `code/tests/test_deepagent_snippets.py:308-380,648-848`

**Interfaces:**
- Consumes: `SnippetExtractionConfig.snippets_per_page`, the existing fully sorted/de-duplicated `Sequence[ScoredTextChunk]`, and the current page offset.
- Produces: `SnippetPage.page_index: int`, `residual_count: int`, `residual_top_score: float | None`, `returned_min_score: float | None`, and `pages_estimated: int` in `SnippetPage.as_dict()` and exact cached responses.

- [ ] **Step 1: Write failing pagination metadata tests**

Extend the first-page, short-document, and continuation tests with exact assertions:

```python
def test_relevant_passage_near_document_end_leads_first_ten_item_page(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)

    result = extractor.extract("doc-a", LONG_DOCUMENT, "target passage")

    assert result.page.page_index == 0
    assert result.page.residual_count == 2
    assert result.page.residual_top_score == 0.0
    assert result.page.returned_min_score == 0.0
    assert result.page.pages_estimated == 2
    assert result.page.as_dict().keys() == {
        "document_id",
        "focus_query",
        "snippets",
        "next_cursor",
        "page_index",
        "residual_count",
        "residual_top_score",
        "returned_min_score",
        "pages_estimated",
    }


def test_continuation_reports_exhausted_residual_ranking(tmp_path: Path) -> None:
    extractor, _ranker = _extractor(tmp_path)
    first = extractor.extract("doc-a", LONG_DOCUMENT, "target passage")

    second = extractor.extract(
        "doc-a", LONG_DOCUMENT, "target passage", first.page.next_cursor
    )

    assert second.page.page_index == 1
    assert second.page.residual_count == 0
    assert second.page.residual_top_score is None
    assert second.page.returned_min_score == 0.0
    assert second.page.pages_estimated == 2
```

Also assert that an empty page reports `page_index=0`, zero residuals, both scores `None`, and `pages_estimated=0`.

- [ ] **Step 2: Run the new tests and confirm the typed page is missing the fields**

Run:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_deepagent_snippets.py::test_relevant_passage_near_document_end_leads_first_ten_item_page \
  code/tests/test_deepagent_snippets.py::test_continuation_reports_exhausted_residual_ranking
```

Expected: failures because `SnippetPage` has no residual/page metadata yet.

- [ ] **Step 3: Extend the typed page and calculate metadata from the stable ranking**

Bump `RESULT_SCHEMA_VERSION` from `2` to `3`. Do not change `CURSOR_SCHEMA_VERSION`; cursor binding and offsets do not change.

Implement the page fields and calculations:

```python
@dataclass(frozen=True)
class SnippetPage:
    document_id: str
    focus_query: str
    snippets: tuple[RelevantSnippet, ...]
    next_cursor: str | None
    page_index: int
    residual_count: int
    residual_top_score: float | None
    returned_min_score: float | None
    pages_estimated: int


def _page(
    self,
    document_id: str,
    focus_query: str,
    ranked: Sequence[ScoredTextChunk],
    offset: int,
    binding_identity: Mapping[str, object],
) -> SnippetPage:
    selected = ranked[offset : offset + self.config.snippets_per_page]
    next_offset = offset + len(selected)
    residual_count = len(ranked) - next_offset
    residual_top_score = (
        ranked[next_offset].relevance_score if residual_count else None
    )
    returned_min_score = (
        min(row.relevance_score for row in selected) if selected else None
    )
    pages_estimated = math.ceil(len(ranked) / self.config.snippets_per_page)
    return SnippetPage(
        document_id=document_id,
        focus_query=focus_query,
        snippets=tuple(
            RelevantSnippet(
                chunk_id=row.chunk.chunk_id,
                start_char=row.chunk.start_char,
                end_char=row.chunk.end_char,
                text=row.chunk.text,
                relevance_score=row.relevance_score,
            )
            for row in selected
        ),
        next_cursor=(
            self._encode_cursor(binding_identity, next_offset)
            if residual_count
            else None
        ),
        page_index=offset // self.config.snippets_per_page,
        residual_count=residual_count,
        residual_top_score=residual_top_score,
        returned_min_score=returned_min_score,
        pages_estimated=pages_estimated,
    )
```

Add all five fields to `as_dict()` and `_page_from_response()`. Validate booleans separately from integers, require finite scores, require `residual_top_score is None` exactly when `residual_count == 0`, and require `returned_min_score is None` exactly when `snippets` is empty.

- [ ] **Step 4: Add cache semantic-validation cases**

Add parameterized cache-rewrite tests that reject:

```python
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("page_index", -1),
        ("residual_count", True),
        ("residual_top_score", float("nan")),
        ("returned_min_score", "0.5"),
        ("pages_estimated", -1),
    ],
)
def test_cache_rejects_invalid_pagination_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    extractor, _ranker = _extractor(tmp_path)
    extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
    cache_file = _cache_file(tmp_path / "pages")
    payload = json.loads(cache_file.read_text())
    payload["response"][field] = value
    payload["response_sha256"] = sha256(
        deepagent_snippets._canonical_json(payload["response"]).encode()
    ).hexdigest()
    cache_file.write_text(json.dumps(payload))

    with pytest.raises(SnippetCacheIntegrityError):
        extractor.extract("doc-a", LONG_DOCUMENT, "target passage")
```

Also validate internal relationships: `page_index == page_offset // page_size`, `next_cursor` exists exactly when residuals remain, and `pages_estimated == ceil((page_offset + len(snippets) + residual_count) / page_size)`.

- [ ] **Step 5: Run the complete snippet module and lint it**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_deepagent_snippets.py
.venv/bin/python -m ruff check \
  code/trec_rag/deepagent_snippets.py \
  code/tests/test_deepagent_snippets.py
```

Expected: all snippet tests pass and Ruff reports no issues.

- [ ] **Step 6: Commit Task 1**

```bash
git add code/trec_rag/deepagent_snippets.py code/tests/test_deepagent_snippets.py
git commit -m "Add snippet residual pagination signals"
```

---

### Task 2: Build the Invocation-Local Evidence/Coverage State

**Files:**
- Create: `code/trec_rag/deepagent_evidence.py`
- Create: `code/tests/test_deepagent_evidence.py`

**Interfaces:**
- Consumes: untouched `narrative: str`, `SnippetPage`, original/follow-up search observations, and model-authored delta dictionaries.
- Produces: `EvidenceCoverageState`, `StateUpdateResult`, `EvidenceCoverageReport`, `view(scope: str) -> str`, `apply_delta(delta: Mapping[str, object]) -> StateUpdateResult`, `choose_action(...) -> Mapping[str, object]`, and mechanical registration/authorization methods used by Task 3.

- [ ] **Step 1: Write the state grounding and coverage tests**

Create model-free unit tests around this setup:

```python
def _state_with_snippet() -> EvidenceCoverageState:
    state = EvidenceCoverageState("Why do people migrate and what challenges do they face?")
    state.record_search(
        query="Why do people migrate and what challenges do they face?",
        kind="original",
        documents=(DocumentObservation("doc-a", 1),),
    )
    snippet_text = "Conflict and persecution force people to flee."
    page = SnippetPage(
        document_id="doc-a",
        focus_query="migration drivers",
        snippets=(
            RelevantSnippet(
                chunk_id="doc-a:0001",
                start_char=0,
                end_char=len(snippet_text),
                text=snippet_text,
                relevance_score=0.9,
            ),
        ),
        next_cursor=None,
        page_index=0,
        residual_count=0,
        residual_top_score=None,
        returned_min_score=0.9,
        pages_estimated=1,
    )
    state.record_snippet_page(page)
    return state
```

Cover these exact behaviors:

- `add_needs` accepts narrative-anchored `n1` and `n2` records.
- An exact supporting quote accepts nugget `g1`; an invented quote rejects only that nugget with `UNGROUNDED_QUOTE`.
- `answerable` is rejected with `MISSING_GROUNDED_DRAFT` unless `draft_answer` and grounded `draft_nugget_ids` are present.
- Several snippets from `doc-a` keep `support="single_document"`; evidence from `doc-b` changes it to `multi_document` without using the word independent.
- A snippet-origin facet without `origin_snippet_id` is rejected.
- Contradiction links are symmetric; supersession preserves both nuggets.
- `conflicted` requires grounded nuggets joined by an explicit contradiction
  link; code validates the link but not whether the claims truly conflict.

- [ ] **Step 2: Run the grounding tests and confirm the module is absent**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_deepagent_evidence.py
```

Expected: collection fails because `trec_rag.deepagent_evidence` does not exist.

- [ ] **Step 3: Define focused records and immutable outputs**

Create these public records with frozen dataclasses and exact enum literals:

```python
NeedStatus = Literal["unaddressed", "partial", "answerable", "conflicted"]
FacetStatus = Literal["open", "covered", "dropped"]
DocumentState = Literal["unexamined", "productive", "exhausted", "abandoned"]
ActionKind = Literal["search", "extract", "paginate", "refocus", "stop"]


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
```

Add private mutable records for needs, facets, nuggets, snippet observations,
document/focus observations, and action history with the same field names as
their report records plus internal cursor/yield data. Keep all mutation behind
`EvidenceCoverageState`; expose only immutable `EvidenceCoverageReport` data.

- [ ] **Step 4: Implement deterministic registration and per-item delta validation**

Use one lock and process delta sections in this fixed order so later entries may
reference accepted earlier entries from the same call:

```python
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
```

The accepted delta shape is:

```json
{
  "add_needs": [
    {
      "need_id": "n1",
      "narrative_span": "why people immigrate or become refugees",
      "question": "Why do people immigrate or become refugees?"
    }
  ],
  "add_facets": [
    {
      "facet_id": "f1",
      "need_ids": ["n1"],
      "dimension": "population",
      "value": "refugee",
      "origin": "narrative",
      "origin_snippet_id": null
    }
  ],
  "add_nuggets": [
    {
      "nugget_id": "g1",
      "text": "Conflict and persecution force displacement.",
      "need_ids": ["n1"],
      "facet_ids": ["f1"],
      "evidence": [
        {
          "snippet_id": "doc-a:0001",
          "quote": "Conflict and persecution force people to flee."
        }
      ],
      "contradicts": []
    }
  ],
  "add_evidence": [],
  "set_facet_status": [
    {
      "facet_id": "f1",
      "status": "covered",
      "status_reason": "A grounded refugee-driver nugget is available",
      "supporting_nugget_ids": ["g1"]
    }
  ],
  "set_need_status": [
    {
      "need_id": "n1",
      "status": "partial",
      "remaining_gap": "Economic and family migration drivers remain uncovered",
      "draft_answer": null,
      "draft_nugget_ids": []
    }
  ],
  "supersede_nuggets": [],
  "abandon_documents": []
}
```

Normalize whitespace only for quote containment; preserve the model's quote and
nugget text in state. Validate references against snippets actually returned in
this invocation. Commit valid entries and reject invalid entries independently;
never rewrite semantic text or silently create IDs.

New facets begin `open`. `covered` requires at least one grounded linked nugget
and `dropped` requires a nonblank reason; code validates those references but
does not decide whether the semantic facet is truly covered.

- [ ] **Step 5: Write and implement compact views, action authorization, and reports**

Add tests for:

- `view("frontier")` contains need status/gap, open facets, residual document
  signals, recent yield, and state hash but no snippet text, and is at most
  `MAX_FRONTIER_CHARACTERS = 2_000` characters.
- `view("need:n1")` includes only `n1`'s nuggets and evidence.
- unknown scopes return a safe JSON error.
- non-stop actions reject missing, closed, or unknown motivating IDs.
- completion-stop requires no open needs and no motivating IDs.
- saturation-stop requires at least one unresolved motivating ID.
- saturation-stop is accepted only after
  `SATURATION_ZERO_YIELD_PAGES = 3` consecutive inspected pages produced no
  newly accepted grounded nugget; this is a yield rule, not a total action cap.
- only one pending retrieval action may exist; a matching tool call consumes it.

Implement the exact action boundary:

```python
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


def require_pending_action(
    self,
    *,
    action: Literal["search", "extract", "paginate", "refocus"],
    target: str,
    focus_query: str | None,
) -> Mapping[str, object] | None:
    """Consume a matching pending action or return a safe tool error."""
```

`report()` must return needs, facets, nuggets, unresolved IDs, action history,
document support multiplicity, state version/hash, and counts without mutable
references. Derive state hashes from canonical JSON with sorted keys.

- [ ] **Step 6: Run the new state module and lint it**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_deepagent_evidence.py
.venv/bin/python -m ruff check \
  code/trec_rag/deepagent_evidence.py \
  code/tests/test_deepagent_evidence.py
```

Expected: all evidence-state tests pass and Ruff reports no issues.

- [ ] **Step 7: Commit Task 2**

```bash
git add code/trec_rag/deepagent_evidence.py code/tests/test_deepagent_evidence.py
git commit -m "Add invocation-local evidence coverage state"
```

---

### Task 3: Integrate State Tools and Action Enforcement into the Deep Agent

**Files:**
- Modify: `code/trec_rag/deepagent_retrieval.py:39-70,129-193,277-365,433-773`
- Modify: `code/tests/test_deepagent_retrieval.py:1-230,428-580,1164-1335,1597-1795`

**Interfaces:**
- Consumes: `EvidenceCoverageState`, enriched `SnippetPage`, and the existing retriever/snippet tools.
- Produces: `AgentToolset`, a five-tool Deep Agent, enforced retrieval-action sequencing, and `AgentRetrievalResult.coverage_report: EvidenceCoverageReport`.

- [ ] **Step 1: Write failing agent-boundary tests**

Add tests that prove:

1. The real factory exposes exactly these custom tools plus safe state-filesystem tools:

```python
expected_custom_tools = {
    "search_climbmix",
    "extract_relevant_snippets",
    "view_retrieval_state",
    "update_retrieval_state",
    "choose_next_action",
}
```

2. A follow-up search without a matching recorded action returns
   `{"error": "matching next action required"}` and makes no retriever call.
3. A matching `choose_next_action(action="search", target=query, ...)` permits
   exactly one call; replay requires another action.
4. `cursor=None` requires `extract` for the first focus and `refocus` for a new
   focus on an inspected document; a non-null cursor requires `paginate` for the
   stored document/focus.
5. The original search is recorded mechanically without requiring an action.
6. The first model message still contains the untouched narrative and
   metadata-only original results.
7. The final result exposes five seeded topic-224 needs when the fake agent
   writes them and returns unresolved gaps honestly when it stops early.

- [ ] **Step 2: Run the boundary tests and confirm the current three-argument factory fails them**

Run:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_deepagent_retrieval.py::test_factory_passes_only_explicit_deepagents_070_arguments \
  code/tests/test_deepagent_retrieval.py::test_real_deepagents_factory_exposes_retrieval_and_safe_state_tools \
  code/tests/test_deepagent_retrieval.py::test_retrieve_requires_recorded_actions_for_agent_retrieval_tools
```

Expected: failures because the state tools and action enforcement do not exist.

- [ ] **Step 3: Replace the positional factory surface with a typed toolset**

Define:

```python
@dataclass(frozen=True)
class AgentToolset:
    search_climbmix: Callable[[str], str]
    extract_relevant_snippets: Callable[[str, str, str | None], str]
    view_retrieval_state: Callable[[str], str]
    update_retrieval_state: Callable[[dict[str, Any]], str]
    choose_next_action: Callable[[str, str, str | None, list[str], str], str]


AgentFactory = Callable[[str, AgentToolset], _Agent]
```

Update `_create_agent(model, toolset)` to pass all five bound callables in
`tools=`. Extend `_RetrievalOnlyMiddleware._ALLOWED_TOOLS` with only the three
new semantic-control tools. Continue forcing `parallel_tool_calls=False`.
Migrate test factories to the typed toolset rather than adding a compatibility
shim.

- [ ] **Step 4: Construct one state per `retrieve()` and expose narrow closures**

Inside `retrieve(narrative)`, construct `EvidenceCoverageState(narrative)`.
After every successful original/follow-up search, call `record_search()` with
document ID and rank only. After every successful snippet call, call
`record_snippet_page(result.page)` before returning the page JSON.

Expose closures with these model-visible signatures:

```python
def view_retrieval_state(scope: str = "frontier") -> str:
    """View compact invocation-local needs, gaps, evidence, or document state."""
    return coverage_state.view(scope)


def update_retrieval_state(delta: dict[str, Any]) -> str:
    """Add grounded needs, facets, nuggets, evidence, and coverage judgments."""
    return json.dumps(coverage_state.apply_delta(delta).as_dict(), sort_keys=True)


def choose_next_action(
    action: str,
    target: str,
    focus_query: str | None,
    motivating_ids: list[str],
    rationale: str,
) -> str:
    """Record the coverage gap motivating the next retrieval action or stop."""
    return json.dumps(
        coverage_state.choose_action(
            action=cast(Any, action),
            target=target,
            focus_query=focus_query,
            motivating_ids=motivating_ids,
            rationale=rationale,
        ),
        sort_keys=True,
    )
```

Use an internal validator rather than relying on the `cast` for runtime safety.
All tool errors must be compact JSON and must not expose cache paths, document
text, cursor contents, or stack traces.

- [ ] **Step 5: Require and consume matching actions in search/snippet tools**

Before a follow-up search transport call, consume a pending `search` action
whose target exactly equals the query. Before snippet extraction, derive the
expected action:

```python
expected_action = coverage_state.expected_snippet_action(
    document_id=document_id,
    focus_query=focus_query,
    cursor=cursor,
)
authorization_error = coverage_state.require_pending_action(
    action=expected_action,
    target=document_id,
    focus_query=focus_query,
)
if authorization_error is not None:
    return json.dumps(authorization_error, sort_keys=True)
```

Do not require an action for the mandatory original search. Keep tool-owned
cache execution unchanged after authorization.

- [ ] **Step 6: Replace the system prompt with the evidence-control loop**

The prompt must explicitly require this order:

```text
First decompose the untouched narrative into explicit needs and record them
with update_retrieval_state. Preserve each need's exact narrative span.
Before every search or document inspection, view the frontier and record one
matching next action with its open motivating need or facet. After every
snippet page, add only claims grounded by exact returned quotes, update gaps,
and then choose whether to inspect another document, refocus, paginate, search,
or stop. A next_cursor alone is not a reason to paginate: use residual count,
within-ranking score continuity, novel nugget yield, and the remaining gap.
Mark a need answerable only with a draft answer and grounded nugget IDs. Report
conflicts and unresolved gaps. Caches remain tool-owned; use state scratch only
for oversized output or temporary notes.
```

Do not hardcode topic 224's five needs in the prompt; the fake/live topic-224
run must derive them from its narrative.

- [ ] **Step 7: Return and trace the immutable coverage report**

Add `coverage_report: EvidenceCoverageReport` to `AgentRetrievalResult`. Set it
from `coverage_state.report()` after agent completion, including when the agent
uses no semantic tools. Derive `stopping_reason` from a recorded terminal stop
when present; otherwise preserve the existing `agent_completed` or
`search_budget_exhausted` fallback.

- [ ] **Step 8: Run the retrieval module tests and lint it**

Run:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_deepagent_evidence.py \
  code/tests/test_deepagent_retrieval.py
.venv/bin/python -m ruff check \
  code/trec_rag/deepagent_evidence.py \
  code/trec_rag/deepagent_retrieval.py \
  code/tests/test_deepagent_evidence.py \
  code/tests/test_deepagent_retrieval.py
```

Expected: all evidence and retrieval tests pass and Ruff reports no issues.

- [ ] **Step 9: Commit Task 3**

```bash
git add \
  code/trec_rag/deepagent_retrieval.py \
  code/tests/test_deepagent_retrieval.py
git commit -m "Integrate evidence coverage tools into Deep Agent"
```

---

### Task 4: Expose Safe Phoenix Evidence and Document the SDK

**Files:**
- Modify: `code/trec_rag/deepagent_tracing.py:143-264`
- Modify: `code/tests/test_deepagent_tracing.py:97-205,330-410`
- Modify: `code/trec_rag/README.md:195-290`

**Interfaces:**
- Consumes: enriched snippet page fields and final `EvidenceCoverageReport` counts/hash.
- Produces: bounded Phoenix attributes for residual pagination and final coverage, plus updated SDK documentation.

- [ ] **Step 1: Write failing safe-trace tests**

Extend snippet span tests to call `record_page()` with:

```python
page_index=0,
residual_count=7,
residual_top_score=0.81,
returned_min_score=0.62,
pages_estimated=2,
```

Assert the exported span contains those five typed attributes and never contains
`next_cursor`. Add validation cases for negative counts/indexes, non-finite
scores, and inconsistent `residual_count`/`residual_top_score` nullability.

Extend the root result span test with:

```python
coverage_state_hash="a" * 64,
need_count=5,
answerable_need_count=3,
conflicted_need_count=1,
unresolved_need_count=1,
nugget_count=14,
action_count=19,
```

Assert only counts/hash are added; no narrative, nugget, quote, scratch, or
cursor content is placed in manual attributes.

- [ ] **Step 2: Run tracing tests and confirm the protocol rejects new arguments**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_deepagent_tracing.py
```

Expected: failures because `_SnippetSpan.record_page()` and
`_AgentSpan.record_result()` do not accept the new evidence.

- [ ] **Step 3: Add typed bounded attributes and wire the retrieval protocols**

Update both tracing methods and the matching protocols in
`deepagent_retrieval.py`. Validate all counts as non-boolean non-negative
integers, validate scores as finite numbers or the permitted `None`, and
validate the state hash as exactly 64 lowercase hexadecimal characters.

Use these attribute names:

```text
snippet.page_index
snippet.residual_count
snippet.residual_top_score
snippet.returned_min_score
snippet.pages_estimated
coverage.state_hash
coverage.need_count
coverage.answerable_need_count
coverage.conflicted_need_count
coverage.unresolved_need_count
coverage.nugget_count
coverage.action_count
```

Automatic LangChain tool spans will carry compact `view`, `delta`, and action
tool content according to the existing `trace_content` setting. Do not create a
second manual span for each state tool.

When an optional score is `None`, omit that OpenTelemetry attribute instead of
attempting to export a null value. Extend `_AgentSpan._STOPPING_REASONS` with
`coverage_complete` and `evidence_saturated` while retaining
`agent_completed` and `search_budget_exhausted`.

- [ ] **Step 4: Update the README with the three-store protocol and result example**

Document:

- need map versus mechanical retrieval ledger versus grounded nugget store;
- `view_retrieval_state`, `update_retrieval_state`, and
  `choose_next_action`;
- residual pagination fields and within-ranking-only score interpretation;
- exact-quote grounding and the draft-answer coverage gate;
- `single_document`/`multi_document` support without source independence;
- invocation-local state versus cache-first tools versus scratch spill;
- `result.coverage_report` inspection from Python.

Use a Python SDK example, not a CLI.

- [ ] **Step 5: Run tracing/retrieval tests and lint changed files**

Run:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_deepagent_tracing.py \
  code/tests/test_deepagent_retrieval.py
.venv/bin/python -m ruff check \
  code/trec_rag/deepagent_tracing.py \
  code/trec_rag/deepagent_retrieval.py \
  code/tests/test_deepagent_tracing.py \
  code/tests/test_deepagent_retrieval.py
```

Expected: all targeted tests pass and Ruff reports no issues.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  code/trec_rag/deepagent_tracing.py \
  code/trec_rag/deepagent_retrieval.py \
  code/trec_rag/README.md \
  code/tests/test_deepagent_tracing.py \
  code/tests/test_deepagent_retrieval.py
git commit -m "Trace and document evidence-guided retrieval"
```

---

### Task 5: Verify One Complete Topic-224 Retrieval Pass

**Files:**
- Modify after verification: `docs/superpowers/plans/2026-07-30-deepagent-evidence-coverage-map.md`

**Interfaces:**
- Consumes: completed Tasks 1-4, repository `.env`, topic 224 narrative helper, existing OpenRouter/ClimbMix/Phoenix configuration.
- Produces: targeted local verification plus one live `AgentRetrievalResult` and Phoenix trace whose coverage/action records can be inspected.

- [ ] **Step 1: Run the complete targeted POC test set**

Run:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_deepagent_snippets.py \
  code/tests/test_deepagent_evidence.py \
  code/tests/test_deepagent_retrieval.py \
  code/tests/test_deepagent_tracing.py
.venv/bin/python -m ruff check \
  code/trec_rag/deepagent_snippets.py \
  code/trec_rag/deepagent_evidence.py \
  code/trec_rag/deepagent_retrieval.py \
  code/trec_rag/deepagent_tracing.py \
  code/tests/test_deepagent_snippets.py \
  code/tests/test_deepagent_evidence.py \
  code/tests/test_deepagent_retrieval.py \
  code/tests/test_deepagent_tracing.py
```

Expected: all targeted tests pass and Ruff reports no issues. Do not expand to
the full repository suite unless a targeted failure indicates a shared-module
regression.

- [ ] **Step 2: Run topic 224 once through the Python SDK**

The linked worktree must have the ignored repository environment file. If
`.env` is absent, copy it without displaying its contents, then verify Git
ignores it:

```bash
cp /home/npatta01/data/competitions/trec_rag_2026/.env .env
git check-ignore -q .env
```

Skip the copy when `.env` already exists. From the repository root, run:

```bash
PYTHONPATH=code .venv/bin/python - <<'PY'
from pathlib import Path
import json

from trec_rag.deepagent_retrieval import DeepAgentRetriever
from trec_rag.topics import load_topic_narrative

topic_file = Path(
    "trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"
)
narrative = load_topic_narrative("224", topic_file)
result = DeepAgentRetriever.from_env(root=Path.cwd()).retrieve(narrative)
print(
    json.dumps(
        {
            "stopping_reason": result.stopping_reason,
            "searches": [search.query for search in result.searches],
            "coverage": result.coverage_report.as_dict(),
            "trace_flush_succeeded": result.trace_flush_succeeded,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
```

Expected: five narrative-anchored needs; every accepted nugget has grounded
evidence; every non-stop retrieval action names an open need/facet; stop follows
completion or saturation rules; unresolved gaps remain explicit; trace flush is
true. Pagination is judged by its recorded residual/yield decision and is not
required merely because a cursor exists.

- [ ] **Step 3: Inspect the Phoenix trace and compare the prior behavior**

Verify:

- the first model call sees no preloaded document excerpts;
- need decomposition precedes agent-triggered retrieval;
- state-tool spans are visible and not redacted when `trace_content=True`;
- snippet spans show residual signals but no cursor value;
- every agent-triggered search/snippet call has a preceding matching action;
- document abandonment and the final stop are explicit;
- cache status remains tool-owned and absent from semantic deltas.

- [ ] **Step 4: Record exact verification evidence in this plan**

Append a `## Verification Evidence` section containing the exact targeted test
count, Ruff result, live stopping reason, need/nugget/action/page counts,
pagination decisions, trace-flush result, and Phoenix trace URL or trace ID.
Do not include credentials, full snippets, raw documents, cache paths, or
scratch contents.

- [ ] **Step 5: Commit the verification record**

```bash
git add docs/superpowers/plans/2026-07-30-deepagent-evidence-coverage-map.md
git commit -m "Record evidence coverage POC verification"
```
