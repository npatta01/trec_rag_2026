# Multipage Competition Debug Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a privacy-safe run-and-score summary plus one complete standalone raw debug page per selected competition topic, then validate and privately serve the 119-topic 2026 bundle.

**Architecture:** Keep `competition_debug_report.py` as the public deep module and sealed-artifact authority. Add a focused `competition_debug_bundle.py` implementation module that projects an allowlisted run summary, adapts the existing evaluation manifest, renders the summary, and publishes a create-only static bundle after rendering each topic through shared raw-report presentation functions.

**Tech Stack:** Python 3.12, frozen dataclasses, standard-library HTML/JSON/hash/filesystem APIs, pytest, repository-pinned offline evaluation and privacy helpers, dependency-free HTML/CSS/JavaScript, Chrome/Chromium smoke checks, private Tailscale Serve.

## Global Constraints

- Preserve `build_debug_report(...) -> DebugReportReceipt` and the existing `--output FILE` behavior.
- Multipage output is explicit through mutually exclusive `--output-dir DIR`; its target is create-only.
- Load and validate the sealed retrieval and optional RAG artifacts exactly once per bundle.
- Do not read candidate ledgers, qrels, gold nuggets, judge events, provider responses, or environment secrets in the bundle builder.
- `index.html` receives only the allowlisted summary model; it must contain no narratives, queries, answers, claims, passages, document IDs, citations, source paths, prompts, provider responses, full digests, secrets, or environment values.
- Keep retrieval relevance, nugget/obligation coverage, and answer/citation quality in separate metric families; never calculate a cross-family composite.
- Missing or incomplete metrics render as `Unavailable` with their exact validated reason, never as zero.
- Every raw topic page retains every stage and row rendered by the legacy report.
- HTML pages remain standalone and dependency-free, with inline CSS/JavaScript and readable no-JavaScript fallbacks.
- Write `bundle-manifest.json` last, verify every page/hash/link, then rename the completed sibling staging directory into the absent target.
- Make no network, retrieval, reranking, generation, judging, or evaluation calls.
- Raw topic pages remain private and may be served only through the already-authorized tailnet-only portal after privacy review.
- Preserve unrelated dirty-worktree changes and stage only files named by this plan.

---

## File Structure

- Create `code/trec_rag/competition_debug_bundle.py`: summary/evaluation view models, privacy projection, summary HTML, bundle manifest, create-only publication, and bundle receipt construction.
- Create `code/tests/test_competition_debug_bundle.py`: focused model, renderer, publisher, CLI, privacy, deterministic-output, and browser contracts.
- Modify `code/trec_rag/competition_debug_report.py`: reusable raw topic-content/document shell, standalone topic-page renderer, public bundle receipt/wrapper, and CLI dispatch.
- Modify `code/tests/test_competition_debug_report.py`: regression contracts for the refactored legacy shell and standalone raw topic rendering.
- Modify `code/trec_rag/README.md`: operator commands, bundle layout, score overlay, create-only semantics, size expectations, and privacy warning.
- Modify `.agents/skills/trec-rag-competition-debug-report/SKILL.md`: choose multipage output for multi-topic runs and verify the bundle receipt without changing hosted-evaluation authorization.
- Modify `code/tests/test_competition_debug_report_skill.py`: enforce the skill's multipage command, receipt, and privacy behavior.
- Add the already-approved `docs/superpowers/specs/2026-08-09-multipage-competition-debug-report-design.md` and this plan to the isolated implementation branch.

### Public interfaces locked by this plan

```text
# code/trec_rag/competition_debug_report.py
@dataclass(frozen=True)
class TopicPageNavigation:
    summary_href: str
    position: int
    total: int
    previous_href: str | None = None
    next_href: str | None = None

@dataclass(frozen=True)
class DebugReportBundleReceipt:
    schema_version: str
    output_dir: Path
    index_path: Path
    manifest_path: Path
    topic_ids: tuple[str, ...]
    rag_included: bool
    evaluation_included: bool
    page_count: int
    total_bytes: int
    bundle_manifest_sha256: str

render_debug_topic_page(
    topic: TopicReport,
    *,
    navigation: TopicPageNavigation,
) -> str

build_debug_report_bundle(
    retrieval_config_path: Path,
    *,
    rag_config_path: Path | None = None,
    topic_ids: Sequence[str] | None = None,
    evaluation_manifest_path: Path | None = None,
    output_dir: Path,
) -> DebugReportBundleReceipt
```

```text
# code/trec_rag/competition_debug_bundle.py
@dataclass(frozen=True)
class MetricAvailability:
    available: bool
    reason: str | None

@dataclass(frozen=True)
class MetricFamilyOverlay:
    key: str
    label: str
    definitions: Mapping[str, str]
    macro_rule: str
    macro: Mapping[str, float]
    macro_availability: MetricAvailability
    per_topic: Mapping[str, Mapping[str, float]]
    per_topic_availability: Mapping[str, MetricAvailability]

@dataclass(frozen=True)
class EvaluationOverlay:
    manifest_sha256: str
    topic_ids: tuple[str, ...]
    families: tuple[MetricFamilyOverlay, ...]
    source_label: str

@dataclass(frozen=True)
class TopicSummary:
    topic_id: str
    health: str
    fallback_kinds: tuple[str, ...]
    depth: int
    subnarratives: int
    queries: int
    nuggets: int
    href: str
    metrics: Mapping[str, Mapping[str, float]]
    metric_availability: Mapping[str, MetricAvailability]

@dataclass(frozen=True)
class RunSummary:
    topics: tuple[TopicSummary, ...]
    completed_topics: int
    fallback_topics: int
    submitted_documents: int
    canonical_nuggets: int
    distributions: Mapping[str, Mapping[str, int | float]]
    evaluation: EvaluationOverlay | None
    rag_included: bool

load_evaluation_overlay(path: Path, selected_topic_ids: Sequence[str]) -> EvaluationOverlay
build_run_summary(data: DebugReportData, evaluation: EvaluationOverlay | None) -> RunSummary
render_bundle_summary(summary: RunSummary, *, denylist: Sequence[str]) -> str
build_bundle_from_data(
    data: DebugReportData,
    *,
    output_dir: Path,
    evaluation_manifest_path: Path | None,
) -> DebugReportBundleReceipt
```

---

### Task 1: Allowlisted Summary and Evaluation Models

**Files:**
- Create: `code/trec_rag/competition_debug_bundle.py`
- Create: `code/tests/test_competition_debug_bundle.py`

**Interfaces:**
- Consumes: `DebugReportData`, `TopicReport`, `friendly_report.build_presentation`, and `offline_evaluation.load_manifest`.
- Produces: `MetricAvailability`, `MetricFamilyOverlay`, `EvaluationOverlay`, `TopicSummary`, `RunSummary`, `load_evaluation_overlay`, and `build_run_summary` with the exact signatures above.

- [ ] **Step 1: Write failing summary-projection tests**

Use the existing generic run fixture so the tests do not encode real 2026 paths or text:

```python
from dataclasses import fields, replace
from pathlib import Path

import pytest

from offline_evaluation_fixture import TopicSpec, build_run
from trec_rag.competition_debug_report import load_debug_report_data


def test_run_summary_projects_only_safe_counts_and_attention_state(tmp_path: Path) -> None:
    # Import inside the test for the first RED run: the absent module must make
    # this test FAIL, not abort collection.
    from trec_rag.competition_debug_bundle import build_run_summary

    fixture = build_run(
        tmp_path,
        (
            TopicSpec("alpha-topic", "private alpha narrative", candidate_documents=3),
            TopicSpec("beta-topic", "private beta narrative", candidate_documents=7),
        ),
    )
    data = load_debug_report_data(fixture.retrieval_config, rag_config_path=fixture.rag_config)
    fallback_result = replace(data.topics[1].canonical_results[0], state="fallback_extractive")
    fallback_topic = replace(
        data.topics[1],
        original_only_fallback=True,
        canonical_results=(fallback_result,),
    )

    summary = build_run_summary(replace(data, topics=(data.topics[0], fallback_topic)), None)

    assert summary.completed_topics == 2
    assert summary.fallback_topics == 1
    assert summary.submitted_documents == 2
    assert [topic.health for topic in summary.topics] == ["complete", "fallback"]
    assert summary.distributions["depth"] == {"minimum": 1, "median": 1, "maximum": 1}
    assert "narrative" not in {field.name for field in fields(summary.topics[0])}
    assert "docid" not in {field.name for field in fields(summary.topics[0])}
```

Add this second test proving query counts mean generated subnarrative BM25
queries, canonical nugget counts are exact, and topic links retain official
order:

```python
def test_run_summary_counts_generated_queries_nuggets_and_safe_topic_links(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import build_run_summary

    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    data = load_debug_report_data(fixture.retrieval_config)

    summary = build_run_summary(data, None)

    assert [topic.topic_id for topic in summary.topics] == ["alpha-topic", "beta-topic"]
    assert [topic.href for topic in summary.topics] == [
        "topics/alpha-topic.html",
        "topics/beta-topic.html",
    ]
    assert [topic.queries for topic in summary.topics] == [
        sum(len(item.bm25_queries) for item in topic.subnarratives)
        for topic in data.topics
    ]
    assert [topic.nuggets for topic in summary.topics] == [
        len(topic.canonical_nuggets) for topic in data.topics
    ]
```

- [ ] **Step 2: Run the projection tests and confirm the module is absent**

Run:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_competition_debug_bundle.py::test_run_summary_projects_only_safe_counts_and_attention_state
```

Expected: the selected test is reported as `FAILED` with
`ModuleNotFoundError: No module named 'trec_rag.competition_debug_bundle'` from
inside the test body; test collection itself succeeds.

- [ ] **Step 3: Implement immutable summary records and deterministic distributions**

Create the dataclasses in the locked interface. Build topic health from validated state only:

```python
def _topic_health(topic: TopicReport) -> tuple[str, tuple[str, ...]]:
    fallbacks = {
        result.state
        for result in topic.canonical_results
        if result.state == "fallback_extractive"
    }
    if topic.original_only_fallback:
        fallbacks.add("original_only")
    if fallbacks:
        return "fallback", tuple(sorted(fallbacks))
    if any(result.state == "empty" for result in topic.canonical_results):
        return "empty", ()
    return "complete", ()


def _distribution(values: Sequence[int]) -> Mapping[str, int | float]:
    if not values:
        raise ValueError("a summary distribution requires at least one value")
    return MappingProxyType(
        {
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
        }
    )
```

`build_run_summary` must count `queries` as
`sum(len(subnarrative.bm25_queries) for subnarrative in topic.subnarratives)`,
use `len(topic.retrieval_output.documents)` for submitted depth, use
`len(topic.canonical_nuggets)` for nuggets, and create the metric mappings from
the supplied overlay only.

- [ ] **Step 4: Run the projection tests**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_competition_debug_bundle.py -k summary_projection
```

Expected: all selected projection tests pass.

- [ ] **Step 5: Write failing evaluation-overlay tests using the real manifest schema**

Build an evaluation fixture with two arbitrary topics, deterministic full-support
judge results, and fixture qrels. Then assert family separation, subset scope,
definitions, metrics, and unavailability:

```python
from trec_rag.competition_debug_bundle import load_evaluation_overlay
from trec_rag.offline_evaluation import (
    JudgeOutcome,
    JudgeSettings,
    build_evaluation_bundle,
)


def test_evaluation_overlay_preserves_families_scope_and_unavailability(tmp_path: Path) -> None:
    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    bundle = build_evaluation_bundle(
        retrieval_config_path=fixture.retrieval_config,
        rag_config_path=fixture.rag_config,
        work_dir=tmp_path / "evaluation",
        repository_root=REPOSITORY_ROOT,
        cache_root=tmp_path / "judge-cache",
        qrels_path=fixture.qrels(),
        judge=lambda _task: JudgeOutcome(status="completed", support_label="FS"),
        judge_settings=JudgeSettings(
            provider="fixture",
            model="fixture-model",
            thinking="disabled",
            temperature=0.0,
            system_prompt="fixture prompt",
            agent_binary="fixture-agent",
        ),
        created_utc="2026-08-09T00:00:00+00:00",
    )

    overlay = load_evaluation_overlay(bundle.manifest_path, ("beta-topic",))

    assert overlay.topic_ids == ("beta-topic",)
    assert [family.key for family in overlay.families] == [
        "retrieval",
        "nugget_coverage",
        "citation_support",
    ]
    retrieval = overlay.families[0]
    assert "ndcg@10" in retrieval.per_topic["beta-topic"]
    assert retrieval.macro_availability.available is False
    assert "2-topic evaluation scope" in retrieval.macro_availability.reason
    nuggets = overlay.families[1]
    assert nuggets.macro == {}
    assert nuggets.macro_availability.available is False
    assert "no released gold-nugget file" in nuggets.macro_availability.reason
```

Also mutate copies of the manifest to verify rejection of a missing selected
topic, conflicting relative order, non-finite numeric values, duplicate topic
IDs, unavailable states without reasons, and unknown evaluation schemas.

- [ ] **Step 6: Run the overlay tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_competition_debug_bundle.py -k evaluation_overlay
```

Expected: failures identify the missing `load_evaluation_overlay` behavior.

- [ ] **Step 7: Implement strict evaluation projection**

Load and validate through repository-owned contracts before extracting metrics:

```python
_FAMILY_LABELS = (
    ("retrieval", "Retrieval relevance"),
    ("nugget_coverage", "Nugget or obligation coverage"),
    ("citation_support", "Answer and citation quality"),
)


def load_evaluation_overlay(
    path: Path, selected_topic_ids: Sequence[str]
) -> EvaluationOverlay:
    payload = Path(path).read_bytes()
    manifest = load_manifest(Path(path))
    build_presentation(manifest)
    scope = tuple(str(value) for value in manifest["scope"]["topic_ids"])
    selected = tuple(str(value) for value in selected_topic_ids)
    positions = [scope.index(topic_id) for topic_id in selected if topic_id in scope]
    if len(positions) != len(selected):
        raise EvaluationError("evaluation scope is missing a selected report topic")
    if positions != sorted(positions):
        raise EvaluationError("evaluation and report topic order conflict")
    families = tuple(
        _project_metric_family(manifest, key=key, label=label, topic_ids=selected)
        for key, label in _FAMILY_LABELS
    )
    return EvaluationOverlay(
        manifest_sha256=sha256(payload).hexdigest(),
        topic_ids=selected,
        families=families,
        source_label="Validated offline evaluation manifest",
    )
```

`_project_metric_family` must accept only finite `int`/`float` values excluding
booleans, create read-only mappings sorted by stable metric name, require
availability objects with `available: bool` and a nonempty reason when false,
take the macro rule from `metric_definitions.macro`, and subset per-topic cells
without recomputing the manifest's authoritative macro. When the selected
topics are a strict subset of the manifest scope, retain per-topic metrics. For
an otherwise-available full-scope macro, replace macro availability with
`available=False` and the exact reason `<N>-topic evaluation scope does not
match the <M>-topic report scope`. A family whose macro was already unavailable
keeps its more specific validated reason, such as missing gold nuggets; scope
mismatch must not overwrite the cause that prevented the metric from existing.

- [ ] **Step 8: Run all Task 1 tests**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_competition_debug_bundle.py -k 'summary_projection or evaluation_overlay'
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit the summary/evaluation model**

```bash
git add code/trec_rag/competition_debug_bundle.py code/tests/test_competition_debug_bundle.py
git commit -m "feat: project debug report run summaries"
```

---

### Task 2: Standalone Raw Topic Pages

**Files:**
- Modify: `code/trec_rag/competition_debug_report.py:2745-3030`
- Modify: `code/tests/test_competition_debug_report.py:2580-2800`

**Interfaces:**
- Consumes: the existing `_render_narrative` through `_render_retrieval` stage renderers.
- Produces: `TopicPageNavigation`, `_render_topic_content`, and `render_debug_topic_page` with the locked signatures.

- [ ] **Step 1: Write failing one-topic page tests**

```python
def test_standalone_topic_page_contains_one_complete_trace_and_relative_navigation(
    tmp_path: Path,
) -> None:
    config_path, _output = _write_debug_run(tmp_path)
    data = load_debug_report_data(config_path)
    page = debug_report.render_debug_topic_page(
        data.topics[0],
        navigation=debug_report.TopicPageNavigation(
            summary_href="../index.html",
            position=2,
            total=3,
            previous_href="rag2026-0.html",
            next_href="rag2026-2.html",
        ),
    )

    assert page.startswith("<!doctype html>")
    assert "Topic rag2026-0" in page
    for stage in (
        "Narrative", "Subnarratives", "Generated answer", "Funnel overview",
        "New documents", "Selected documents", "Top passages",
        "Final selected nuggets", "Final retrieval",
    ):
        assert stage in page
    assert 'href="../index.html"' in page
    assert 'href="rag2026-0.html"' in page
    assert 'href="rag2026-2.html"' in page
    assert "Topic 2 of 3" in page
    assert "https://" not in page
```

Add these boundary, escaping, anchor, and determinism assertions:

```python
def test_standalone_topic_page_omits_missing_boundary_links_and_escapes_text(
    tmp_path: Path,
) -> None:
    config_path, _output = _write_debug_run(tmp_path)
    topic = load_debug_report_data(config_path).topics[0]
    page = debug_report.render_debug_topic_page(
        topic,
        navigation=debug_report.TopicPageNavigation(
            summary_href="../index.html",
            position=1,
            total=1,
        ),
    )

    assert "Previous topic" not in page
    assert "Next topic" not in page
    assert "Original &lt;narrative&gt; &amp; &quot; in page
    assert 'id="stage-literal-rag2026-0-narrative"' in page
    assert page == debug_report.render_debug_topic_page(
        topic,
        navigation=debug_report.TopicPageNavigation(
            summary_href="../index.html",
            position=1,
            total=1,
        ),
    )
```

- [ ] **Step 2: Run the topic-page test and verify the missing interface**

Run:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_competition_debug_report.py::test_standalone_topic_page_contains_one_complete_trace_and_relative_navigation
```

Expected: FAIL because `TopicPageNavigation` or `render_debug_topic_page` is absent.

- [ ] **Step 3: Extract the shared topic body and document shell**

Refactor without changing any existing stage renderer:

```python
def _render_topic_content(topic: TopicReport) -> str:
    anchor = _topic_anchor(topic.topic_id)
    prefix = f"stage-{anchor}"
    stages = (
        _render_narrative(topic, prefix),
        _render_subnarratives(topic, prefix),
        _render_final_rag(topic, prefix),
        _render_funnel_overview(topic, prefix),
        _render_new_documents(topic, prefix),
        _render_selected_documents(topic, prefix),
        _render_passages(topic, prefix),
        _render_nuggets(topic, prefix),
        _render_retrieval(topic, prefix),
    )
    return (
        f'<div class="topic-content"><h1 id="topic-title-{anchor}">'
        f'Topic {_html(topic.topic_id)}</h1>{"".join(stages)}</div>'
    )
```

Move the legacy stylesheet byte-for-byte from `render_debug_report` into the
module constant `_DEBUG_REPORT_CSS`. Move its topic-tab script byte-for-byte to
`_LEGACY_TOPIC_SCRIPT`. Add `_render_html_document(title, header, main, script)`
and make the legacy renderer call it. The legacy result must retain its title,
topic tabs, run diagnostics, no-JavaScript disclosures, and every existing test
contract.

- [ ] **Step 4: Implement standalone topic navigation and rendering**

```python
def _topic_page_link(href: str | None, label: str) -> str:
    if href is None:
        return ""
    return f'<a class="topic-page-link" href="{_html(href)}">{_html(label)}</a>'


def render_debug_topic_page(
    topic: TopicReport, *, navigation: TopicPageNavigation
) -> str:
    if navigation.total < 1 or not 1 <= navigation.position <= navigation.total:
        raise ValueError("topic page position must belong to its total")
    header = (
        '<header><p class="eyebrow">Private raw trace</p>'
        f'<h1>Topic {_html(topic.topic_id)}</h1>'
        f'<p>Topic {_html(navigation.position)} of {_html(navigation.total)}</p>'
        '<nav class="topic-page-nav" aria-label="Report navigation">'
        f'{_topic_page_link(navigation.summary_href, "Back to summary")}'
        f'{_topic_page_link(navigation.previous_href, "Previous topic")}'
        f'{_topic_page_link(navigation.next_href, "Next topic")}'
        '</nav></header>'
    )
    return _render_html_document(
        title=f"{topic.topic_id} · Competition retrieval debug report",
        header=header,
        main=f"<main>{_render_pipeline_legend()}{_render_topic_content(topic)}</main>",
        script="",
    )
```

Add the sticky responsive navigation CSS to `_DEBUG_REPORT_CSS`, preserving
44-pixel targets, visible focus, reduced motion, and page-level overflow safety.

- [ ] **Step 5: Run topic-page and legacy renderer tests**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_competition_debug_report.py -k 'render or standalone_topic or information_hierarchy or browser'
```

Expected: all selected tests pass; Chrome-dependent tests may skip only when no
Chrome/Chromium executable is installed.

- [ ] **Step 6: Commit the shared raw renderer**

```bash
git add code/trec_rag/competition_debug_report.py code/tests/test_competition_debug_report.py
git commit -m "feat: render standalone topic debug pages"
```

---

### Task 3: Privacy-safe Summary HTML

**Files:**
- Modify: `code/trec_rag/competition_debug_bundle.py`
- Modify: `code/tests/test_competition_debug_bundle.py`

**Interfaces:**
- Consumes: `RunSummary`, `friendly_report.assert_publishable`, and a run-derived short-token denylist.
- Produces: `render_bundle_summary(summary, *, denylist) -> str`.

- [ ] **Step 1: Write failing summary-renderer contracts**

```python
from offline_evaluation_fixture import RunFixture
from trec_rag.competition_debug_bundle import (
    EvaluationOverlay,
    build_run_summary,
    load_evaluation_overlay,
    render_bundle_summary,
)
from trec_rag.competition_debug_report import DebugReportData
from trec_rag.friendly_report import ReportPrivacyError


REPOSITORY_ROOT = Path(__file__).parents[2]


def _evaluated_two_topic_data(
    tmp_path: Path,
) -> tuple[RunFixture, DebugReportData, EvaluationOverlay]:
    fixture = build_run(
        tmp_path,
        (
            TopicSpec("alpha-topic", "private alpha narrative"),
            TopicSpec("beta-topic", "private beta narrative"),
        ),
    )
    evaluation = build_evaluation_bundle(
        retrieval_config_path=fixture.retrieval_config,
        rag_config_path=fixture.rag_config,
        work_dir=tmp_path / "evaluation",
        repository_root=REPOSITORY_ROOT,
        cache_root=tmp_path / "judge-cache",
        qrels_path=fixture.qrels(),
        judge=lambda _task: JudgeOutcome(status="completed", support_label="FS"),
        judge_settings=JudgeSettings(
            provider="fixture",
            model="fixture-model",
            thinking="disabled",
            temperature=0.0,
            system_prompt="fixture prompt",
            agent_binary="fixture-agent",
        ),
        created_utc="2026-08-09T00:00:00+00:00",
    )
    data = load_debug_report_data(
        fixture.retrieval_config,
        rag_config_path=fixture.rag_config,
    )
    overlay = load_evaluation_overlay(evaluation.manifest_path, fixture.topic_ids)
    return fixture, data, overlay


def test_summary_html_shows_health_scores_and_unavailability_without_private_text(
    tmp_path: Path,
) -> None:
    fixture, data, overlay = _evaluated_two_topic_data(tmp_path)
    summary = build_run_summary(data, overlay)
    private_docids = tuple(fixture.docids.values())

    page = render_bundle_summary(summary, denylist=private_docids)

    assert page.startswith("<!doctype html>")
    assert "Run summary" in page
    assert "Retrieval relevance" in page
    assert "Nugget or obligation coverage" in page
    assert "Answer and citation quality" in page
    assert "Unavailable" in page
    assert "no released gold-nugget file" in page
    assert "private alpha narrative" not in page
    assert all(docid not in page for docid in private_docids)
    assert 'href="topics/alpha-topic.html"' in page
    assert 'data-sort-kind="number"' in page
    assert 'type="search"' in page
```

Add exact contracts for macros, deterministic columns, missing evaluation,
attention state, escaping, and the denylist:

```python
def test_summary_html_keeps_metric_families_definitions_and_macro_rules(
    tmp_path: Path,
) -> None:
    _fixture, data, overlay = _evaluated_two_topic_data(tmp_path)
    page = render_bundle_summary(build_run_summary(data, overlay), denylist=())

    assert "Unweighted mean over the topic cells" in page
    assert "Normalized discounted cumulative gain" in page
    assert page.index("Retrieval relevance") < page.index("Nugget or obligation coverage")
    assert page.index("Nugget or obligation coverage") < page.index("Answer and citation quality")
    retrieval_names = sorted(overlay.families[0].definitions)
    assert [page.index(f'data-metric="retrieval:{name}"') for name in retrieval_names] == sorted(
        page.index(f'data-metric="retrieval:{name}"') for name in retrieval_names
    )


def test_summary_without_evaluation_is_explicit_and_keeps_official_row_order(
    tmp_path: Path,
) -> None:
    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    data = load_debug_report_data(fixture.retrieval_config)
    page = render_bundle_summary(build_run_summary(data, None), denylist=())

    assert "Evaluation not supplied" in page
    assert page.index('data-topic-id="alpha-topic"') < page.index('data-topic-id="beta-topic"')
    assert "0.000000" not in page


def test_summary_privacy_scan_rejects_a_run_derived_collision(tmp_path: Path) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    data = load_debug_report_data(fixture.retrieval_config)

    with pytest.raises(ReportPrivacyError, match="forbidden private value"):
        render_bundle_summary(
            build_run_summary(data, None),
            denylist=("alpha-topic",),
        )
```

In the first renderer test, use a narrative containing `</script><script>` and
assert neither literal sequence reaches the summary because narratives are not
part of `RunSummary`. The browser test in Task 6 exercises the complete sorting
and filtering script; source-level assertions here require `aria-sort`, a search
label, and safely serialized static JavaScript with no source-derived JSON blob.

- [ ] **Step 2: Run the summary-renderer tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_competition_debug_bundle.py -k summary_html
```

Expected: failures identify the absent renderer.

- [ ] **Step 3: Implement the semantic summary shell**

Render these sections in order:

```python
def render_bundle_summary(summary: RunSummary, *, denylist: Sequence[str]) -> str:
    page = _summary_document(
        health=_render_health_cards(summary),
        distributions=_render_distributions(summary),
        score_families=_render_score_families(summary),
        topics=_render_topic_table(summary),
    )
    assert_publishable(page, denylist=tuple(denylist))
    return page
```

`_render_health_cards` shows completed/selected topics, fallback topics,
submitted documents, depth median/range, canonical nuggets, and evaluation
state. `_render_distributions` shows minimum/median/maximum for depth,
subnarratives, queries, and nuggets. `_render_score_families` renders one
separate section per family and uses `Unavailable — <reason>` when availability
is false. `_render_topic_table` uses stable base columns followed by metric
families and names; every numeric cell carries an invariant raw decimal in a
`data-sort-value` attribute and formatted visible text.

- [ ] **Step 4: Implement progressive filtering and stable sorting**

Embed a small script that does not fetch data:

```javascript
const rows = Array.from(document.querySelectorAll("#topic-table tbody tr"));
const official = new Map(rows.map((row, index) => [row, index]));
const missing = Number.POSITIVE_INFINITY;

function numericValue(row, column) {
  const cell = row.cells[column];
  return cell && cell.dataset.sortValue !== undefined
    ? Number(cell.dataset.sortValue)
    : missing;
}

function sortRows(column, direction) {
  rows.sort((left, right) => {
    const delta = numericValue(left, column) - numericValue(right, column);
    if (Number.isFinite(delta) && delta !== 0) return direction * delta;
    if (numericValue(left, column) !== numericValue(right, column)) {
      return numericValue(left, column) === missing ? 1 : -1;
    }
    return official.get(left) - official.get(right);
  });
  rows.forEach((row) => row.parentNode.appendChild(row));
}
```

The complete script must also perform case-insensitive topic/status filtering,
toggle `aria-sort` on the active header, preserve stable official-order ties,
and expose a "Needs attention" reset that restores failure/fallback/official
ordering. Button labels must name the active metric; no unlabeled primary score
or cross-family calculation is permitted.

- [ ] **Step 5: Implement the summary denylist projection**

Add `_summary_denylist(data)` that returns unique nonempty short sensitive
tokens: every document ID/reference, full source/config path, and full digest.
The allowlisted model already excludes long raw text; the generic private regex
scan catches credential names, credential-shaped strings, provider-event
fields, and private absolute paths. Call `assert_publishable` only on the final
summary page.

- [ ] **Step 6: Run all summary renderer/privacy tests**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_competition_debug_bundle.py -k 'summary_html or privacy or sort'
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the summary renderer**

```bash
git add code/trec_rag/competition_debug_bundle.py code/tests/test_competition_debug_bundle.py
git commit -m "feat: render private run score summaries"
```

---

### Task 4: Create-only Bundle Publisher and CLI

**Files:**
- Modify: `code/trec_rag/competition_debug_bundle.py`
- Modify: `code/trec_rag/competition_debug_report.py:3930-4010`
- Modify: `code/tests/test_competition_debug_bundle.py`
- Modify: `code/tests/test_competition_debug_report.py:3900-4210`

**Interfaces:**
- Consumes: one `DebugReportData`, `render_debug_topic_page`, `build_run_summary`, `render_bundle_summary`, and optional `EvaluationOverlay`.
- Produces: `build_bundle_from_data`, public `build_debug_report_bundle`, bundle manifest v1, bundle JSON CLI receipt, and unchanged legacy CLI receipt.

Use this internal receipt so one record owns both reconciliation and manifest
serialization:

```python
@dataclass(frozen=True)
class _PageReceipt:
    path: str
    bytes: int
    sha256: str
    topic_id: str | None = None

    def as_json(self) -> dict[str, str | int]:
        value: dict[str, str | int] = {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }
        if self.topic_id is not None:
            value["topic_id"] = self.topic_id
        return value
```

- [ ] **Step 1: Write failing bundle layout, receipt, and determinism tests**

```python
from hashlib import sha256
from html.parser import HTMLParser
import json
import socket

import trec_rag.competition_debug_bundle as bundle_module
import trec_rag.competition_debug_report as debug_report
from trec_rag.competition_debug_bundle import build_bundle_from_data


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "a" and dict(attrs).get("href"):
            self.hrefs.append(str(dict(attrs)["href"]))


def _local_hrefs(page: str) -> tuple[str, ...]:
    parser = _HrefParser()
    parser.feed(page)
    return tuple(
        href
        for href in parser.hrefs
        if not href.startswith(("http://", "https://", "mailto:", "#"))
    )


def test_build_bundle_writes_summary_topics_and_manifest_with_matching_receipt(
    tmp_path: Path,
) -> None:
    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    target = fixture.retrieval_output / "debug-bundle"

    receipt = debug_report.build_debug_report_bundle(
        fixture.retrieval_config,
        rag_config_path=fixture.rag_config,
        output_dir=target,
    )

    assert receipt.output_dir == target.resolve()
    assert receipt.index_path == target.resolve() / "index.html"
    assert receipt.topic_ids == ("alpha-topic", "beta-topic")
    assert receipt.page_count == 3
    manifest = json.loads(receipt.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "competition_debug_report_bundle_v1"
    assert [item["topic_id"] for item in manifest["topics"]] == list(receipt.topic_ids)
    assert manifest["index"]["sha256"] == _file_sha256(target / "index.html")
    assert all(
        item["sha256"] == _file_sha256(target / item["path"])
        for item in manifest["topics"]
    )
    assert receipt.bundle_manifest_sha256 == _file_sha256(receipt.manifest_path)
    assert receipt.total_bytes == sum(path.stat().st_size for path in target.rglob("*" ) if path.is_file())
```

Build the same input into a second absent target and use this exact determinism
and single-load contract:

```python
def test_bundle_is_deterministic_and_loads_run_data_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    first = fixture.retrieval_output / "bundle-a"
    second = fixture.retrieval_output / "bundle-b"
    real_load = debug_report.load_debug_report_data
    calls = 0

    def observed_load(*args: object, **kwargs: object) -> DebugReportData:
        nonlocal calls
        calls += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(debug_report, "load_debug_report_data", observed_load)
    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=first)
    assert calls == 1
    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=second)
    assert calls == 2
    first_files = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()}
    assert first_files == second_files
```

- [ ] **Step 2: Write failing publication safety tests**

Use these contracts for target ownership, traversal, cleanup, link integrity,
write order, and source immutability:

- an existing target directory or file is refused and unchanged;
- a symlink target is refused;
- an output outside the repository and retrieval output is refused;
- a topic ID containing `/`, `..`, or a control character is refused;
- forced rendering, summary privacy, page-write, manifest-write, directory
  fsync, and final-rename failures remove only the owned staging directory;
- `bundle-manifest.json` is the final file write in staging;
- every expected previous/next/index href resolves to a created file;
- a forced hash mismatch prevents rename; and
- the source retrieval and organizer artifacts remain byte-identical.

Use a write observer rather than wall-clock timestamps:

```python
@pytest.mark.parametrize("existing_kind", ("file", "directory", "symlink"))
def test_bundle_refuses_every_existing_target(
    tmp_path: Path, existing_kind: str
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "existing-bundle"
    if existing_kind == "file":
        target.write_text("owned by somebody else", encoding="utf-8")
    elif existing_kind == "directory":
        target.mkdir()
        (target / "owned.txt").write_text("owned by somebody else", encoding="utf-8")
    else:
        destination = fixture.retrieval_output / "symlink-destination"
        destination.mkdir()
        target.symlink_to(destination, target_is_directory=True)

    with pytest.raises(ValueError, match="bundle output.*(?:absent|symbolic link)"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert target.exists() or target.is_symlink()


@pytest.mark.parametrize("topic_id", ("bad/topic", "..", "bad\x00topic"))
def test_bundle_rejects_unsafe_topic_filenames(tmp_path: Path, topic_id: str) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    data = load_debug_report_data(fixture.retrieval_config)
    unsafe = replace(data, topics=(replace(data.topics[0], topic_id=topic_id),))
    target = fixture.retrieval_output / "unsafe-topic-bundle"

    with pytest.raises(ValueError, match="safe topic filename"):
        build_bundle_from_data(unsafe, output_dir=target, evaluation_manifest_path=None)

    assert not target.exists()


def test_bundle_rejects_an_external_output_directory(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    fixture = build_run(run_root, (TopicSpec("alpha-topic", "alpha"),))
    external_parent = tmp_path / "external"
    external_parent.mkdir()

    with pytest.raises(ValueError, match="inside the repository or retrieval output"):
        debug_report.build_debug_report_bundle(
            fixture.retrieval_config,
            output_dir=external_parent / "bundle",
        )


@pytest.mark.parametrize("failed_name", ("index.html", "bundle-manifest.json"))
def test_bundle_write_failure_removes_only_owned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_name: str
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "failed-bundle"
    sentinel = fixture.retrieval_output / "unrelated.txt"
    sentinel.write_text("keep", encoding="utf-8")
    real_write = bundle_module._write_bundle_file

    def fail_named(path: Path, payload: bytes) -> None:
        if path.name == failed_name:
            raise OSError(f"forced {failed_name} failure")
        real_write(path, payload)

    monkeypatch.setattr(bundle_module, "_write_bundle_file", fail_named)
    with pytest.raises(OSError, match="forced"):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()
    assert list(fixture.retrieval_output.glob(".failed-bundle.*.tmp")) == []
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_bundle_manifest_is_written_last_and_all_links_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    target = fixture.retrieval_output / "ordered-bundle"
    writes: list[str] = []
    real_write = bundle_module._write_bundle_file

    def observed(path: Path, payload: bytes) -> None:
        writes.append(path.name)
        real_write(path, payload)

    monkeypatch.setattr(bundle_module, "_write_bundle_file", observed)
    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert writes[-1] == "bundle-manifest.json"
    for page in target.rglob("*.html"):
        for href in _local_hrefs(page.read_text(encoding="utf-8")):
            assert (page.parent / href.split("#", 1)[0]).resolve().is_file()
```

Use one parameterized failure-injection contract for rendering, fsync, rename,
and final hash reconciliation:

```python
@pytest.mark.parametrize(
    "failure_point",
    ("topic-render", "summary-render", "directory-fsync", "rename", "hash"),
)
def test_bundle_failure_points_never_publish_a_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "partial-bundle"
    if failure_point == "topic-render":
        monkeypatch.setattr(
            bundle_module,
            "render_debug_topic_page",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("topic-render")),
        )
    elif failure_point == "summary-render":
        monkeypatch.setattr(
            bundle_module,
            "render_bundle_summary",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("summary-render")),
        )
    elif failure_point == "directory-fsync":
        monkeypatch.setattr(
            bundle_module,
            "_fsync_directory",
            lambda *_args: (_ for _ in ()).throw(OSError("directory-fsync")),
        )
    elif failure_point == "rename":
        monkeypatch.setattr(
            bundle_module.os,
            "rename",
            lambda *_args: (_ for _ in ()).throw(OSError("rename")),
        )
    else:
        monkeypatch.setattr(bundle_module, "_hash_matches", lambda *_args: False)

    with pytest.raises((OSError, RuntimeError, ValueError), match=failure_point):
        debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)

    assert not target.exists()
    assert list(fixture.retrieval_output.glob(".partial-bundle.*.tmp")) == []
```

Lock `_hash_matches(path: Path, expected_sha256: str, expected_bytes: int) -> bool`
as the final on-disk receipt check. Add this exact source immutability test:

```python
def test_bundle_build_does_not_modify_source_artifacts(tmp_path: Path) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    before = {
        path.relative_to(fixture.retrieval_output): _file_sha256(path)
        for path in fixture.retrieval_output.rglob("*")
        if path.is_file()
    }

    debug_report.build_debug_report_bundle(
        fixture.retrieval_config,
        output_dir=fixture.retrieval_output / "immutable-source-bundle",
    )

    after = {
        relative: _file_sha256(fixture.retrieval_output / relative)
        for relative in before
    }
    assert after == before
```

- [ ] **Step 3: Run publisher tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_competition_debug_bundle.py -k 'build_bundle or publication or manifest'
```

Expected: failures identify the missing publisher and public wrapper.

- [ ] **Step 4: Implement safe target resolution and per-page streaming writes**

Resolve the target inside the repository or retrieval output, require an
existing safe parent, reject all existing/symlink targets, and create staging
with restrictive permissions:

```python
staging = Path(
    tempfile.mkdtemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
)
staging.chmod(0o700)
(staging / "topics").mkdir(mode=0o700)
```

Render and write one topic at a time so the implementation never constructs an
808 MB aggregate HTML string. `_write_bundle_file` uses exclusive creation,
mode `0o600`, flush, and `os.fsync`. Record each page's relative POSIX path,
byte count, and SHA-256 from the exact encoded bytes.

- [ ] **Step 5: Implement navigation, manifest, validation, and atomic rename**

For topic `i`, construct:

```python
navigation = TopicPageNavigation(
    summary_href="../index.html",
    position=i + 1,
    total=len(data.topics),
    previous_href=None if i == 0 else f"{data.topics[i - 1].topic_id}.html",
    next_href=None if i + 1 == len(data.topics) else f"{data.topics[i + 1].topic_id}.html",
)
```

Build the manifest with this exact top-level shape. `sources` is the sorted
mapping of already validated bounded-artifact receipts; no paths outside the
retrieval output, environment values, or raw content are added:

```python
manifest = {
    "schema_version": "competition_debug_report_bundle_v1",
    "report_schema_version": _REPORT_SCHEMA_VERSION,
    "topic_ids": [topic.topic_id for topic in data.topics],
    "index": index_receipt.as_json(),
    "topics": [receipt.as_json() for receipt in topic_receipts],
    "sources": dict(sorted(data.source_sha256s.items())),
    "rag_included": data.rag_config_path is not None,
    "evaluation": (
        {
            "included": True,
            "schema_version": "trec_rag_offline_evaluation_bundle_v1",
            "sha256": evaluation.manifest_sha256,
        }
        if evaluation is not None
        else {"included": False, "schema_version": None, "sha256": None}
    ),
    "page_count": 1 + len(topic_receipts),
    "total_bytes": 0,
}
```

`PageReceipt.as_json()` returns `path`, `bytes`, and `sha256`; a topic receipt
also returns `topic_id`. Build canonical JSON with `ensure_ascii=False`,
`sort_keys=True`, compact separators, and one trailing newline. Stabilize
`total_bytes` because the manifest includes its own length:

```python
manifest["total_bytes"] = sum(page.bytes for page in pages)
while True:
    body = _canonical_json_bytes(manifest)
    total = sum(page.bytes for page in pages) + len(body)
    if manifest["total_bytes"] == total:
        break
    manifest["total_bytes"] = total
```

Before writing the manifest, verify topic coverage/order, unique safe paths,
every relative href, recorded page bytes/hashes, and no unlisted files. Write
the manifest last, fsync staging and its parent, confirm the target remains
absent, then `os.rename(staging, target)`. On failure, resolve and verify that
the cleanup candidate is the exact owned staging child before calling
`shutil.rmtree`.

- [ ] **Step 6: Add the public wrapper and bundle CLI dispatch**

In `competition_debug_report.py`, define `DebugReportBundleReceipt` and use a
function-local import to avoid a module cycle:

```python
def build_debug_report_bundle(
    retrieval_config_path: Path,
    *,
    rag_config_path: Path | None = None,
    topic_ids: Sequence[str] | None = None,
    evaluation_manifest_path: Path | None = None,
    output_dir: Path,
) -> DebugReportBundleReceipt:
    data = load_debug_report_data(
        Path(retrieval_config_path),
        rag_config_path=None if rag_config_path is None else Path(rag_config_path),
        topic_ids=topic_ids,
    )
    from trec_rag.competition_debug_bundle import build_bundle_from_data
    return build_bundle_from_data(
        data,
        output_dir=Path(output_dir),
        evaluation_manifest_path=evaluation_manifest_path,
    )
```

Use one argparse mutually exclusive group:

```python
destination = parser.add_mutually_exclusive_group()
destination.add_argument("--output", type=Path)
destination.add_argument("--output-dir", type=Path)
parser.add_argument("--evaluation-manifest", type=Path)
```

Reject `--evaluation-manifest` unless `--output-dir` is selected. Dispatch
legacy output exactly as before when `--output-dir` is absent. Bundle stdout is
one sorted compact JSON object containing the fields of
`DebugReportBundleReceipt`; legacy stdout retains its exact current field set.

- [ ] **Step 7: Run publisher and CLI tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_competition_debug_bundle.py \
  code/tests/test_competition_debug_report.py
```

Expected: all tests pass, with only environment-dependent browser skips.

- [ ] **Step 8: Verify no network or candidate-ledger access**

Add this socket/candidate-ledger guard and run it directly:

```python
def test_bundle_makes_no_network_call_or_candidate_ledger_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    target = fixture.retrieval_output / "offline-bundle"
    real_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.name == "candidates.jsonl":
            raise AssertionError("candidate ledger was opened")
        return real_open(path, *args, **kwargs)

    def forbid_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(socket, "socket", forbid_socket)

    debug_report.build_debug_report_bundle(fixture.retrieval_config, output_dir=target)
    assert (target / "bundle-manifest.json").is_file()
```

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_competition_debug_bundle.py \
  -k 'no_network or candidate_ledger or source_artifacts'
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit the bundle publisher and CLI**

```bash
git add \
  code/trec_rag/competition_debug_bundle.py \
  code/trec_rag/competition_debug_report.py \
  code/tests/test_competition_debug_bundle.py \
  code/tests/test_competition_debug_report.py
git commit -m "feat: publish multipage debug report bundles"
```

---

### Task 5: Operator and Skill Documentation

**Files:**
- Modify: `code/trec_rag/README.md:973-1008`
- Modify: `.agents/skills/trec-rag-competition-debug-report/SKILL.md:25-70`
- Verify unchanged contract: `code/tests/test_competition_debug_report_skill.py`
- Add: `docs/superpowers/specs/2026-08-09-multipage-competition-debug-report-design.md`
- Add: `docs/superpowers/plans/2026-08-09-multipage-competition-debug-report.md`

**Interfaces:**
- Consumes: the final CLI and bundle receipt.
- Produces: copy-paste operator commands and a skill contract that selects bundle output for multi-topic runs without authorizing hosted evaluation or publication.

- [ ] **Step 1: Run a RED application scenario against the current skill**

The controller dispatches a fresh agent with the current
`.agents/skills/trec-rag-competition-debug-report/SKILL.md` and this scenario:

```text
IMPORTANT: Treat this as the real operator request and give the exact command
and receipt checks you would use.

A sealed retrieval run has all 119 topics. Its existing standalone debug report
is 808 MB. A validated evaluation_manifest.json may be supplied later. The user
asks for a summary plus browsable per-topic detail and says nothing about hosted
judging or serving. Choose the report CLI form now. State whether splitting
reduces total bytes, whether any hosted call is allowed, and which receipt
fields you verify. Do not ask a follow-up question.
```

Record the response verbatim under this plan's SDD workspace. RED is confirmed
when the current skill chooses or defaults to the single-file `--output` path,
does not describe the bundle receipt, or treats an existing evaluation manifest
as authorization to run/resume judging. If the control already satisfies every
new requirement, no skill edit is necessary; update only the human README.

- [ ] **Step 2: Update the README operator section**

Keep the legacy examples and add these literal commands:

```bash
.venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml \
  --output-dir outputs/facet-deepseek-b40-v3/competition-debug-report
```

```bash
.venv/bin/python \
  -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml \
  --evaluation-manifest PRIVATE_WORK_DIR/evaluation_manifest.json \
  --output-dir outputs/facet-deepseek-b40-v3/competition-debug-report-scored
```

Document the three-file layout, create-only target, exact score families,
unavailable-not-zero rule, atomic manifest-last receipt, private raw topic
pages, and the fact that splitting reduces per-browser load rather than
guaranteeing substantial total disk reduction.

- [ ] **Step 3: Update the skill workflow with the minimum guidance that closes RED**

For one explicitly requested topic, retain the legacy single-file command. For
two or more topics or all exported topics, require `--output-dir BUNDLE_DIR` in
an ignored retrieval-output location. Add `--evaluation-manifest` only when a
validated manifest already exists and the user asks to include scores; state
that this rendering action performs no judging and creates no hosted call.

Require receipt checks for `index_path`, ordered `topic_ids`, `page_count`,
`total_bytes`, `bundle_manifest_sha256`, `rag_included`, and
`evaluation_included`. Preserve every existing authorization, privacy, cache,
probe, and resume rule verbatim.

- [ ] **Step 4: Run the GREEN application scenario with the edited skill**

Dispatch a fresh agent with the updated skill and the exact Step 1 scenario.
GREEN requires all of these observable choices:

- `--output-dir BUNDLE_DIR` for all 119 topics;
- optional `--evaluation-manifest EVALUATION_MANIFEST` only for an already
  validated local score overlay;
- no retrieval, generation, judging, hosted call, or serving action;
- explicit verification of `index_path`, ordered `topic_ids`, `page_count`,
  `total_bytes`, `bundle_manifest_sha256`, `rag_included`, and
  `evaluation_included`; and
- a clear statement that multipage output reduces the browser loading unit but
  may leave total bundle bytes near the raw trace size.

Record the response verbatim in the SDD workspace. If any item is absent,
tighten only the corresponding workflow sentence and re-run with another fresh
agent until the scenario passes.

- [ ] **Step 5: Run existing documentation and portability regression tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  code/tests/test_competition_debug_report_skill.py \
  code/tests/test_environment_portability.py -k 'competition_debug_report or skill'
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit documentation, spec, and plan**

```bash
git add \
  .agents/skills/trec-rag-competition-debug-report/SKILL.md \
  code/trec_rag/README.md \
  docs/superpowers/specs/2026-08-09-multipage-competition-debug-report-design.md \
  docs/superpowers/plans/2026-08-09-multipage-competition-debug-report.md
git commit -m "docs: explain multipage debug reports"
```

---

### Task 6: Full Verification and Private 2026 Handoff

**Files:**
- Verify only: all files changed in Tasks 1-5
- Generate ignored artifact: `outputs/facet-deepseek-b40-v3/competition-debug-report-20260809/`
- Refresh authorized derived portal artifact: `$HOME/codex-rendered/plans/trec-rag-2026-competition-debug-report/`
- Modify derived portal index: `$HOME/codex-rendered/index.html`

**Interfaces:**
- Consumes: final CLI, sealed 119-topic retrieval output, optional existing score manifest, and the authorized private Tailscale Serve mapping.
- Produces: verified tests, independent review, a private multipage bundle, and a live tailnet-only summary URL.

- [ ] **Step 1: Run focused static and unit checks**

```bash
git diff --check
.venv/bin/python -m pytest -q \
  code/tests/test_competition_debug_bundle.py \
  code/tests/test_competition_debug_report.py \
  code/tests/test_competition_debug_report_skill.py \
  code/tests/test_offline_evaluation.py
```

Expected: `git diff --check` is silent and all tests pass, with only explicitly
reported environment-dependent browser skips.

- [ ] **Step 2: Run the complete repository test suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass. Record exact passed/skipped counts.

- [ ] **Step 3: Build the cheapest real two-topic bundle**

Use an absent ignored target and two exported topics:

```bash
.venv/bin/python -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml \
  --topic rag2026-0 \
  --topic rag2026-2 \
  --output-dir outputs/facet-deepseek-b40-v3/competition-debug-report-smoke-20260809
```

Expected: a compact JSON receipt with two topics and three HTML pages. Recompute
every listed hash, open both navigation directions, and confirm the summary
contains neither known document IDs nor `/home/` paths.

- [ ] **Step 4: Run desktop and mobile browser verification**

Use the installed Chrome/Chromium executable through the existing headless
test harness. Verify the summary at 1440×1000 and 390×844, one raw topic at both
sizes, keyboard focus, filtering, low-to-high/high-to-low metric sorting when a
fixture score overlay is present, fragment navigation, disclosures, and no
page-level horizontal overflow.

Run:

```bash
.venv/bin/python -m pytest -q code/tests/test_competition_debug_bundle.py -k browser
```

Expected: browser checks pass; if neither executable exists, record the skip
and do not claim browser verification.

- [ ] **Step 5: Request independent substantive review**

Invoke `superpowers:requesting-code-review` and assign a `sol_reviewer` to
compare the branch against the approved spec. Require review of privacy,
manifest/publication safety, legacy compatibility, score semantics,
accessibility, and missing verification. Address every confirmed finding with
a failing regression test before changing implementation, then rerun the
focused suite.

- [ ] **Step 6: Build and reconcile the full 119-topic bundle**

Only after the cheap checks and review pass:

```bash
.venv/bin/python -m trec_rag.competition_debug_report \
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml \
  --output-dir outputs/facet-deepseek-b40-v3/competition-debug-report-20260809
```

Expected: 119 ordered topic pages plus `index.html`; `page_count` is 120. Verify
119 unique IDs, every manifest hash and byte count, all internal links, summary
cards, fallback rows, total bundle bytes, summary bytes, and min/median/max topic
page bytes. Confirm no evaluation is represented as zero when no 2026 score
manifest exists.

- [ ] **Step 7: Perform the portal privacy review and refresh the authorized copy**

Before copying, scan `index.html` for private paths, full digests, known source
document IDs, credential names/values, provider-event fields, narratives,
queries, answers, claims, and corpus passages. Scan the entire bundle for
credential values and unrelated local paths; raw trace text and document IDs
are expected only under `topics/` and are covered by the user's prior explicit
tailnet-only serving authorization.

Copy only the completed bundle tree to:

```text
$HOME/codex-rendered/plans/trec-rag-2026-competition-debug-report/
```

Update `$HOME/codex-rendered/index.html` with one current summary link and remove
the stale 808 MB single-file link after confirming the new bundle works. Then
delete only the derived portal copy
`$HOME/codex-rendered/plans/trec-rag-2026-competition-debug-report.html` after
verifying the canonical original remains at
`outputs/facet-deepseek-b40-v3/competition_debug_report.html`; that original is
the recovery source. Do not use Funnel, Sites, a public listener, or broaden
filesystem exposure.

- [ ] **Step 8: Verify live tailnet-only delivery**

Verify HTTP 200 and intended titles for:

```text
https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-competition-debug-report/index.html
https://npatta01-framework.tail481212.ts.net/plans/trec-rag-2026-competition-debug-report/topics/rag2026-0.html
```

Follow representative first/middle/last links, verify content lengths against
local files, and run `tailscale serve status` to confirm the mapping still says
tailnet-only. Confirm directory listing and unrelated repository paths are not
exposed.

- [ ] **Step 9: Run final verification immediately before completion**

Invoke `superpowers:verification-before-completion`, then rerun:

```bash
git status --short
git diff --check
.venv/bin/python -m pytest -q \
  code/tests/test_competition_debug_bundle.py \
  code/tests/test_competition_debug_report.py \
  code/tests/test_competition_debug_report_skill.py
```

Expected: only requested branch files are changed, diff check is silent, and
the final focused suite passes. Report the exact verification evidence, live
private URL, canonical ignored bundle path, summary size, total bundle size,
and whether scores were available.

---

## Plan Completion Criteria

- Legacy one-file reports remain compatible.
- A 119-topic build loads only one raw topic page in the browser at a time.
- The summary contains complete run health/counts and future score families but
  no raw corpus/document trace material.
- Unavailable scores are explicit and never flattened into zero or a composite.
- Every bundle file is deterministic, linked, hashed, and published create-only
  with the manifest last.
- Tests, independent review, two-topic real smoke, 119-topic reconciliation,
  desktop/mobile checks, and tailnet-only delivery are evidenced in the handoff.
