# Deep Agent Retrieval Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an importable Python SDK that searches a supplied narrative first, lets a Deep Agent issue bounded ClimbMix follow-up searches, deterministically fuses the results, and exports the complete trace to Phoenix Cloud.

**Architecture:** Keep Deep Agents and Phoenix behind two focused modules. `DeepAgentRetriever` owns orchestration and adapts the existing `PyseriniRemoteRetriever`; `deepagent_tracing` owns optional OpenInference setup and manual agent/retriever spans. The primary API accepts only a narrative, while topic lookup remains a separate helper.

**Tech Stack:** Python 3.12, `deepagents==0.7.0`, `langchain-openrouter==0.2.7`, `arize-phoenix-otel==0.16.1`, `openinference-instrumentation-langchain==0.1.67`, existing Pyserini HTTP adapter/cache, pytest, Phoenix Cloud.

## Global Constraints

- `DeepAgentRetriever.retrieve` accepts an arbitrary non-empty narrative string as its only required retrieval input.
- Search the untouched narrative before invoking the agent; the agent may issue at most three follow-up searches.
- Expose ten candidates per search and return at most twenty fused candidates.
- Use deterministic reciprocal-rank fusion and document-ID deduplication; never compare raw BM25 scores across queries.
- Keep topic loading separate as `load_topic_narrative(topic_id: str, path: Path) -> str`; the path is always explicit.
- Default model: `openrouter:deepseek/deepseek-v4-flash`; override through `DEEPAGENT_MODEL` or constructor input.
- Default Phoenix project: `trec-rag-deepagent-retrieval`; tracing is optional when the collector endpoint is absent.
- Pin `deepagents==0.7.0`, `langchain-openrouter==0.2.7`, `arize-phoenix-otel==0.16.1`, and `openinference-instrumentation-langchain==0.1.67` exactly.
- Use the default ephemeral Deep Agents state backend; do not grant host filesystem or shell access.
- Reuse `PyseriniRemoteRetriever`, the shared cache, rate limiter, provenance validation, and explicit-continuation policy.
- Do not change `run_pipeline`, `run_official`, organizer export, sealed artifacts, or ranking semantics outside this prototype.
- Do not persist chat state, log credentials, add secrets to git, or silently retry OpenRouter, Pyserini, or Phoenix calls.
- Phoenix spans may contain the supplied narrative, follow-up queries, and bounded excerpts, but never credentials, headers, raw responses, continuation-ticket values, or local cache paths.
- The local `.env` remains ignored and mode `600`; only source, tests, lockfile, documentation, design, and plan files may be committed.

---

## File Structure

- `pyproject.toml` and `uv.lock`: exact runtime pins plus a dev-only Phoenix client used to verify trace arrival.
- `code/trec_rag/topics.py`: the separate exact topic-narrative lookup helper.
- `code/trec_rag/deepagent_tracing.py`: optional/idempotent Phoenix setup, OpenInference masking, and manual agent/retriever spans.
- `code/trec_rag/deepagent_retrieval.py`: result records, fusion, search budget/tool adapter, Deep Agents factory, and public SDK.
- `code/tests/test_topics.py`: helper contract.
- `code/tests/test_deepagent_tracing.py`: no-network span hierarchy, masking, configuration, and idempotency contracts.
- `code/tests/test_deepagent_retrieval.py`: no-network SDK orchestration, budget, fusion, failure, and 0.7.0 factory contracts.
- `code/trec_rag/README.md`: SDK inputs, outputs, configuration, privacy, example, and validation commands.

---

### Task 1: Pin Dependencies and Add Separate Narrative Lookup

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `code/trec_rag/topics.py`
- Modify: `code/tests/test_topics.py`

**Interfaces:**
- Consumes: existing `load_narrative_topics(path: Path) -> list[Topic]`.
- Produces: `load_topic_narrative(topic_id: str, path: Path) -> str` and the exact installed dependency versions used by Tasks 2–4.

- [x] **Step 1: Add failing narrative-helper tests**

Append tests that prove the helper returns the exact stored narrative and fails clearly for missing IDs:

```python
from trec_rag.topics import load_topic_narrative


def test_load_topic_narrative_is_separate_exact_lookup(tmp_path):
    path = tmp_path / "topics.tsv"
    path.write_text("224\tProvided narrative exactly.\n", encoding="utf-8")
    assert load_topic_narrative("224", path) == "Provided narrative exactly."


def test_load_topic_narrative_rejects_unknown_id(tmp_path):
    path = tmp_path / "topics.tsv"
    path.write_text("224\tProvided narrative.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="topic ID '999'.*not found"):
        load_topic_narrative("999", path)
```

- [x] **Step 2: Run the helper tests and verify red**

Run: `.venv/bin/python -m pytest code/tests/test_topics.py -q`

Expected: collection fails because `load_topic_narrative` is not defined.

- [x] **Step 3: Implement the exact lookup**

Add the helper without a default topic path:

```python
def load_topic_narrative(topic_id: str, path: Path) -> str:
    """Return one narrative from an explicit topic source."""
    matches = [topic for topic in load_narrative_topics(Path(path)) if topic.id == topic_id]
    if not matches:
        raise ValueError(f"topic ID {topic_id!r} was not found in {path}")
    if len(matches) != 1:
        raise ValueError(f"topic ID {topic_id!r} was not unique in {path}")
    return matches[0].narrative
```

The underlying loader already rejects duplicate IDs. Do not normalize or summarize the returned narrative.

- [x] **Step 4: Pin current runtime and verification dependencies**

Add these exact runtime dependencies under `[project].dependencies`:

```toml
"deepagents==0.7.0",
"langchain-openrouter==0.2.7",
"arize-phoenix-otel==0.16.1",
"openinference-instrumentation-langchain==0.1.67",
```

Add the exact trace-query client under `[dependency-groups].dev`:

```toml
"arize-phoenix-client==2.13.0",
```

Run: `uv lock`

Expected: `uv.lock` resolves on Python 3.12 with the exact direct pins.

- [x] **Step 5: Set up the repository environment and run the focused tests**

Run: `code/tools/setup_env.sh`

Run: `.venv/bin/python -m pytest code/tests/test_topics.py -q`

Expected: all topic tests pass.

- [x] **Step 6: Verify installed versions without contacting providers**

Run:

```bash
.venv/bin/python -c "import importlib.metadata as m; expected={'deepagents':'0.7.0','langchain-openrouter':'0.2.7','arize-phoenix-otel':'0.16.1','openinference-instrumentation-langchain':'0.1.67','arize-phoenix-client':'2.13.0'}; actual={k:m.version(k) for k in expected}; assert actual == expected, actual; print(actual)"
```

Expected: the printed mapping exactly matches `expected`.

- [x] **Step 7: Commit Task 1**

```bash
git add pyproject.toml uv.lock code/trec_rag/topics.py code/tests/test_topics.py
git commit -m "Add Deep Agent prototype dependencies"
```

---

### Task 2: Add Optional Phoenix and OpenInference Tracing

**Files:**
- Create: `code/trec_rag/deepagent_tracing.py`
- Create: `code/tests/test_deepagent_tracing.py`

**Interfaces:**
- Consumes: Phoenix environment names and pinned packages from Task 1.
- Produces:
  - `create_retrieval_tracing(*, environ: Mapping[str, str] | None = None, tracer_provider: TracerProvider | None = None, trace_content: bool = True) -> RetrievalTracing`.
  - `RetrievalTracing.agent_span(narrative: str)` and `RetrievalTracing.retriever_span(query: str)` context managers.
  - `RetrievalTracing.force_flush() -> bool`.

- [x] **Step 1: Write failing configuration and span tests**

Create tests with an OpenTelemetry `TracerProvider`, `SimpleSpanProcessor`, and `InMemorySpanExporter` that verify:

```python
def test_no_phoenix_endpoint_returns_disabled_tracing():
    tracing = create_retrieval_tracing(environ={})
    assert tracing.enabled is False


def test_phoenix_cloud_requires_key():
    with pytest.raises(ValueError, match="PHOENIX_API_KEY"):
        create_retrieval_tracing(
            environ={
                "PHOENIX_COLLECTOR_ENDPOINT": "https://app.phoenix.arize.com/s/example",
            }
        )


def test_injected_provider_captures_agent_and_retriever_hierarchy(span_exporter, provider):
    tracing = create_retrieval_tracing(
        environ={"PHOENIX_PROJECT_NAME": "trec-rag-deepagent-retrieval"},
        tracer_provider=provider,
    )
    with tracing.agent_span("the narrative"):
        with tracing.retriever_span("follow-up query") as span:
            span.set_attribute("retrieval.document_count", 2)
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["climbmix.retrieve", "deepagent.retrieve"]
    assert spans[0].parent.span_id == spans[1].context.span_id
```

Also assert metadata-only mode replaces narrative/query/excerpt values with a redacted marker and that repeated setup with the same provider does not duplicate LangChain instrumentation.

- [x] **Step 2: Run tracing tests and verify red**

Run: `.venv/bin/python -m pytest code/tests/test_deepagent_tracing.py -q`

Expected: collection fails because `deepagent_tracing` is not defined.

- [x] **Step 3: Implement configuration validation and no-op behavior**

Use constants rather than embedding secrets:

```python
DEFAULT_PHOENIX_PROJECT = "trec-rag-deepagent-retrieval"
_CLOUD_HOST = "app.phoenix.arize.com"


def _settings(environ: Mapping[str, str]) -> tuple[str | None, str, bool]:
    endpoint = environ.get("PHOENIX_COLLECTOR_ENDPOINT") or None
    project = environ.get("PHOENIX_PROJECT_NAME") or DEFAULT_PHOENIX_PROJECT
    if endpoint and _CLOUD_HOST in endpoint and not environ.get("PHOENIX_API_KEY"):
        raise ValueError("PHOENIX_API_KEY is required for Phoenix Cloud")
    return endpoint, project, endpoint is not None
```

When no endpoint and no injected provider are present, return `RetrievalTracing(enabled=False)` backed by an OpenTelemetry no-op tracer. Never print the environment mapping.

- [x] **Step 4: Implement idempotent Phoenix/LangChain setup**

For live configuration, call Phoenix registration with HTTP/protobuf and the resolved project name, then instrument LangChain with the same provider and a `TraceConfig` matching `trace_content`. Track instrumented provider identities under a module lock so a notebook can construct multiple SDK objects safely:

```python
provider = phoenix_register(
    project_name=project_name,
    protocol="http/protobuf",
    batch=True,
)
LangChainInstrumentor().instrument(
    tracer_provider=provider,
    config=TraceConfig(
        hide_input_text=not trace_content,
        hide_output_text=not trace_content,
    ),
)
```

Use the injected provider directly in tests; injected providers never export over the network.

- [x] **Step 5: Implement manual agent and retriever spans**

Create an `OITracer` and context managers that name spans exactly `deepagent.retrieve` and `climbmix.retrieve`, set `openinference.span.kind` to `AGENT` and `RETRIEVER`, record exceptions/status, and expose the active span to the caller for safe attributes. Apply the same content masking to manually set narrative/query values.

- [x] **Step 6: Run the tracing tests**

Run: `.venv/bin/python -m pytest code/tests/test_deepagent_tracing.py -q`

Expected: all tracing tests pass with no network request.

- [x] **Step 7: Commit Task 2**

```bash
git add code/trec_rag/deepagent_tracing.py code/tests/test_deepagent_tracing.py
git commit -m "Add Phoenix tracing for retrieval agent"
```

---

### Task 3: Implement the Narrative-Only Deep Agent Retrieval SDK

**Files:**
- Create: `code/trec_rag/deepagent_retrieval.py`
- Create: `code/tests/test_deepagent_retrieval.py`

**Interfaces:**
- Consumes: `Retriever.retrieve(QueryVariant)`, `RetrievedCandidate`, `RankedCandidate`, `repo_cache_root`, `load_repo_env`, and Task 2 tracing.
- Produces:
  - `AgentSearch(query: str, kind: Literal["original", "followup"], candidates: tuple[RetrievedCandidate, ...], cache_status: str)`.
  - `AgentRetrievalResult(narrative: str, searches: tuple[AgentSearch, ...], candidates: tuple[RankedCandidate, ...], rationale: str, stopping_reason: str)`.
  - `reciprocal_rank_fuse(searches: Sequence[AgentSearch], *, limit: int = 20, rrf_k: int = 60) -> tuple[RankedCandidate, ...]`.
  - `DeepAgentRetriever.from_env(...) -> DeepAgentRetriever`.
  - `DeepAgentRetriever.retrieve(narrative: str) -> AgentRetrievalResult`.

- [x] **Step 1: Write failing fusion and orchestration tests**

Use fake retriever and agent factories; no test calls OpenRouter or Pyserini. Cover these exact behaviors:

```python
def test_fusion_deduplicates_and_sums_reciprocal_ranks():
    fused = reciprocal_rank_fuse(searches, limit=20, rrf_k=60)
    assert [row.docid for row in fused] == ["shared", "original-only", "followup-only"]
    assert fused[0].provenance[0]["query_kind"] == "original"


def test_retrieve_searches_untouched_narrative_before_agent_followups():
    result = sdk.retrieve("  supplied narrative exactly  ")
    assert fake_retriever.queries[0].query_text == "  supplied narrative exactly  "
    assert [search.kind for search in result.searches] == ["original", "followup"]
    assert result.narrative == "  supplied narrative exactly  "


def test_retrieve_rejects_empty_narrative_before_external_calls():
    with pytest.raises(ValueError, match="narrative must be non-empty text"):
        sdk.retrieve("   ")
    assert fake_retriever.queries == []
```

Also test the three-follow-up budget, stable query-hash cache variants, an agent exception carrying completed searches, explicit factory arguments to `create_deep_agent`, and `search_budget_exhausted` stopping behavior.

- [x] **Step 2: Run SDK tests and verify red**

Run: `.venv/bin/python -m pytest code/tests/test_deepagent_retrieval.py -q`

Expected: collection fails because `deepagent_retrieval` is not defined.

- [x] **Step 3: Add result records and reciprocal-rank fusion**

Use frozen dataclasses and existing pipeline records. Sum `1 / (rrf_k + source_rank)` per document, retain every query/rank in `RankedCandidate.provenance`, choose the first non-empty document text, sort by descending fused score then `docid`, and assign contiguous one-based output ranks. Validate positive `limit` and `rrf_k`.

- [x] **Step 4: Add the Deep Agents 0.7.0 factory boundary**

Keep the import and construction in one function:

```python
DEFAULT_MODEL = "openrouter:deepseek/deepseek-v4-flash"


def _create_agent(model: str, search_tool: Callable[[str], str]):
    return create_deep_agent(
        model=model,
        tools=[search_tool],
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
    )
```

The prompt must tell the model that the untouched narrative has already been searched, it may use only targeted follow-ups, it must not use filesystem/task tools for this retrieval, and its final message must explain why it stopped. Do not rely on the removed 0.7.0 todo prompt.

- [x] **Step 5: Implement `from_env` without logging secrets**

Load the shared and worktree environments through `load_repo_env`. Require non-empty `OPENROUTER_API_KEY`, construct `RetrieverConfig(name="deepagent_climbmix", type="pyserini_remote", query_variants=("original", "followup"), hits=10, index="climbmix-400b", cache=True)`, and point `PyseriniRemoteRetriever` at `repo_cache_root(root) / "retrieval/pyserini_remote"`. Resolve the model from the constructor argument, then `DEEPAGENT_MODEL`, then `DEFAULT_MODEL`. Construct tracing from the loaded process environment.

- [x] **Step 6: Implement mandatory original search and bounded follow-up tool**

Validate without stripping or rewriting the narrative. Use `sha256(narrative.encode()).hexdigest()[:16]` as the internal cache topic component. Search with variant `original` before agent construction. The closure tool uses `followup-<query-sha256-prefix>` variant names, rejects blank or duplicate queries, records at most three successful follow-ups, and returns JSON containing at most ten bounded excerpts plus remaining budget.

The SDK-side `AgentSearch` retains complete normalized candidates. Tool JSON and Phoenix attributes receive only the bounded excerpts.

- [x] **Step 7: Invoke the agent, fuse results, and preserve failures**

Invoke with the untouched narrative plus bounded original results. Extract the last assistant message as `rationale`; use `search_budget_exhausted` if the tool rejected an over-budget call and `agent_completed` otherwise. Wrap the operation in `deepagent.retrieve`, each search in `climbmix.retrieve`, attach only approved attributes, call `force_flush`, and return immutable records.

Define `AgentRetrievalError(RuntimeError)` with a `searches` tuple. When OpenRouter fails after the original search, raise it with those completed records and keep the original exception as `__cause__`.

- [x] **Step 8: Run focused SDK and existing retriever tests**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_deepagent_retrieval.py code/tests/test_pipeline.py code/tests/test_remote_pyserini.py -q
```

Expected: all selected tests pass with no live external call.

- [x] **Step 9: Commit Task 3**

```bash
git add code/trec_rag/deepagent_retrieval.py code/tests/test_deepagent_retrieval.py
git commit -m "Add narrative-only Deep Agent retriever"
```

---

### Task 4: Document and Verify the End-to-End Prototype

**Files:**
- Modify: `code/trec_rag/README.md`

**Interfaces:**
- Consumes: the public SDK and helper from Tasks 1–3.
- Produces: copy-paste SDK usage and fresh evidence for local tests, one live OpenRouter/Pyserini run, and Phoenix trace arrival.

- [x] **Step 1: Add README usage and boundaries**

Document this primary example:

```python
from trec_rag.deepagent_retrieval import DeepAgentRetriever

result = DeepAgentRetriever.from_env().retrieve(provided_narrative)
for candidate in result.candidates:
    print(candidate.rank, candidate.docid, candidate.score)
```

Document the separate explicit-path helper, four environment names (`OPENROUTER_API_KEY`, `INDEX_URL`, `PYSERINI_API_TOKEN`, and `PHOENIX_COLLECTOR_ENDPOINT`), optional `PHOENIX_API_KEY`/`PHOENIX_PROJECT_NAME`, default bounds, returned records, Phoenix content/privacy behavior, and the absence of official pipeline integration.

- [x] **Step 2: Run the complete focused automated verification**

Run:

```bash
.venv/bin/python -m pytest code/tests/test_topics.py code/tests/test_deepagent_tracing.py code/tests/test_deepagent_retrieval.py code/tests/test_pipeline.py code/tests/test_remote_pyserini.py -q
```

Expected: all selected tests pass.

- [x] **Step 3: Run a secret and portability scan before the live call**

Run:

```bash
git diff --check
rg -n "eyJ[A-Za-z0-9_-]{20,}\\.|sk-or-v1-|Bearer [A-Za-z0-9]" code pyproject.toml docs/superpowers
git check-ignore -q .env && test "$(stat -c %a .env)" = 600
```

Expected: `git diff --check` exits zero, the scan finds no credential values or hard-coded authorization headers, and the local credential file remains ignored with mode `600`.

- [x] **Step 4: Run one bounded live SDK retrieval**

Use the separate helper for test Topic 224, invoke the SDK once, and print only counts, IDs, query text, stopping reason, and candidate ranks—not raw environment values or full document text:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from trec_rag.deepagent_retrieval import DeepAgentRetriever
from trec_rag.topics import load_topic_narrative

narrative = load_topic_narrative(
    "224",
    Path("trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv"),
)
result = DeepAgentRetriever.from_env().retrieve(narrative)
print({
    "search_count": len(result.searches),
    "queries": [search.query for search in result.searches],
    "candidate_ids": [row.docid for row in result.candidates],
    "stopping_reason": result.stopping_reason,
})
PY
```

Expected: the original narrative is search 1, there are no more than four searches total, the fused list has no more than twenty unique IDs, and the call returns without hidden retries.

- [x] **Step 5: Confirm the root span arrived in Phoenix Cloud**

Set `PHOENIX_BASE_URL` to the same space URL as `PHOENIX_COLLECTOR_ENDPOINT` for the read client, query only recent root spans, and assert the project contains `deepagent.retrieve`:

```bash
PHOENIX_BASE_URL="$PHOENIX_COLLECTOR_ENDPOINT" .venv/bin/python - <<'PY'
from datetime import datetime, timedelta, timezone
from phoenix.client import Client

spans = Client().spans.get_spans_dataframe(
    project_identifier="trec-rag-deepagent-retrieval",
    limit=20,
    root_spans_only=True,
    start_time=datetime.now(timezone.utc) - timedelta(minutes=15),
)
names = set(spans["name"].tolist())
assert "deepagent.retrieve" in names, names
print({"project": "trec-rag-deepagent-retrieval", "root_span_seen": True})
PY
```

Expected: prints `root_span_seen: True` without printing trace content or credentials.

- [x] **Step 6: Commit Task 4**

```bash
git add code/trec_rag/README.md
git commit -m "Document Deep Agent retrieval prototype"
```

- [x] **Step 7: Run final repository checks**

Run:

```bash
.venv/bin/python -m pytest -q
git diff --check origin/master..HEAD
git status --short
```

Expected: the full Python suite passes, diff check exits zero, and status contains no tracked changes or staged secrets.
