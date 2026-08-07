from __future__ import annotations

from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import trec_rag.retrieval_nugget_coverage_report as report_module
from trec_rag.facet_extraction import plan_facet_queries
from trec_rag.retrieval_nugget_coverage import (
    CoverageFacet,
    CoverageJudgment,
    CoverageNugget,
    CoverageObligation,
    CoverageReport,
    EvaluatorIdentity,
    FrozenPlan,
)
from trec_rag.retrieval_nugget_coverage_report import (
    CoverageReportData,
    CoverageReportTopic,
    CoverageRunSummary,
    RetrievalPlanContext,
    RetrievalSubnarrativeContext,
    load_coverage_report_data,
    main,
    publish_coverage_report,
    summarize_coverage_topics,
)
from trec_rag.pipeline_models import jsonable
from trec_rag.topics import Topic


def _topic(
    *,
    required_kinds: tuple[str, ...],
    label_counts: dict[str, int],
    required_coverage: float,
    strict_full_rate: float,
    nugget_count: int,
) -> CoverageReportTopic:
    obligations = tuple(
        CoverageObligation(
            obligation_id=f"obligation-{index}",
            requirement=f"requirement-{index}",
            support_test=f"support-{index}",
            kind=kind,
            narrative_spans=(f"span-{index}",),
        )
        for index, kind in enumerate(required_kinds, start=1)
    )
    plan = FrozenPlan(
        narrative="A narrative",
        schema_version="retrieval_nugget_plan_v1",
        facets=(CoverageFacet("facet-1", "Facet", obligations),),
        unmapped_narrative_spans=(),
        canonical_bytes=b"{}",
        plan_sha256="a" * 64,
    )
    report = CoverageReport(
        plan_sha256="a" * 64,
        identity=SimpleNamespace(),
        required_coverage=required_coverage,
        strict_full_rate=strict_full_rate,
        supplemental_coverage=None,
        label_counts=label_counts,
        facet_scores={},
        judgments=(),
        unmapped_narrative_spans=(),
        uncited_nugget_ids=(),
        uncited_nugget_aliases=(),
    )
    evaluation = SimpleNamespace(
        bound_input=SimpleNamespace(nuggets=tuple(object() for _ in range(nugget_count))),
        plan=plan,
        report=report,
    )
    retrieval_plan = RetrievalPlanContext(
        used_fallback=False,
        subnarratives=(
            RetrievalSubnarrativeContext("subnarrative-1", "Scope", ("scope",)),
        ),
        planner_identity={},
        manifest_sha256="b" * 64,
        result_sha256="c" * 64,
    )
    return CoverageReportTopic(evaluation=evaluation, retrieval_plan=retrieval_plan)


def test_summary_uses_literal_expected_counts_and_means() -> None:
    topics = (
        _topic(
            required_kinds=("required_explicit", "supplemental_inferred"),
            label_counts={"full": 1, "partial": 0, "unsupported": 1},
            required_coverage=0.5,
            strict_full_rate=0.5,
            nugget_count=2,
        ),
        _topic(
            required_kinds=("required_explicit", "required_explicit"),
            label_counts={"full": 1, "partial": 1, "unsupported": 0},
            required_coverage=1.0,
            strict_full_rate=0.75,
            nugget_count=3,
        ),
    )

    summary = summarize_coverage_topics(topics)

    assert summary.topic_count == 2
    assert summary.nugget_count == 5
    assert summary.required_obligation_count == 3
    assert summary.supplemental_obligation_count == 1
    assert summary.topic_macro_required_coverage == 0.75
    assert summary.topic_macro_strict_full_rate == 0.625
    assert dict(summary.label_counts) == {
        "full": 2,
        "partial": 1,
        "unsupported": 1,
    }
    assert summary.perfect_required_topic_count == 1


def test_summary_rejects_empty_topics() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_coverage_topics(())


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _saved_decomposition(
    root: Path,
    topic: Topic,
    *,
    fallback: bool = False,
    planner: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    (root / "handoff.json").touch()
    if fallback:
        planned = plan_facet_queries(topic, {"not": "a valid plan"})
    else:
        planned = plan_facet_queries(
            topic,
            {
                "schema_version": "subnarrative_queries_v1",
                "topic_id": topic.id,
                "subnarratives": [
                    {
                        "subnarrative": "First exact scope.",
                        "bm25_queries": ["first lane", "second lane"],
                    },
                    {
                        "subnarrative": "Second exact scope.",
                        "bm25_queries": ["third lane"],
                    },
                ],
            },
        )
    assert planned.used_fallback is fallback
    record = {
        "schema_version": "facet_pilot_v2",
        "topic": {"id": topic.id, "narrative": topic.narrative},
        "narrative_sha256": sha256(topic.narrative.encode("utf-8")).hexdigest(),
        "used_fallback": planned.used_fallback,
        "error": planned.error,
        "queries": jsonable(planned.queries),
        "plan": (
            None
            if planned.plan is None
            else {
                "schema_version": "subnarrative_queries_v1",
                "topic_id": topic.id,
                "subnarratives": [
                    {
                        "subnarrative": row.text,
                        "bm25_queries": list(row.bm25_queries),
                    }
                    for row in planned.plan.subnarratives
                ],
            }
        ),
        "subnarratives": jsonable(planned.subnarratives),
    }
    result_path = root / topic.id / "decomposition" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_bytes = _json_bytes(record)
    result_path.write_bytes(result_bytes)
    manifest_path = result_path.with_name("manifest.json")
    manifest_path.write_bytes(
        _json_bytes(
            {
                "schema_version": "facet-decomposition-manifest-v1",
                "planner": {"backend": "fixture"} if planner is None else planner,
                "result_file": "result.json",
                "result_bytes": len(result_bytes),
                "result_sha256": sha256(result_bytes).hexdigest(),
            }
        )
    )
    return manifest_path, result_path


def _fake_topic(topic: Topic) -> SimpleNamespace:
    return SimpleNamespace(topic_id=topic.id, narrative=topic.narrative)


def test_retrieval_plan_preserves_valid_subnarrative_text_ids_and_query_order(
    tmp_path: Path,
) -> None:
    topic = Topic("topic-plan", "", "Find the exact plan.")
    manifest_path, result_path = _saved_decomposition(tmp_path, topic)

    loaded = report_module._load_retrieval_plan_context(
        handoff_manifest_path=tmp_path / "handoff.json",
        topic=_fake_topic(topic),
    )

    assert loaded.used_fallback is False
    assert [(row.subnarrative_id, row.text, row.bm25_queries) for row in loaded.subnarratives] == [
        ("subnarrative-1", "First exact scope.", ("first lane", "second lane")),
        ("subnarrative-2", "Second exact scope.", ("third lane",)),
    ]
    assert loaded.manifest_sha256 == sha256(manifest_path.read_bytes()).hexdigest()
    assert loaded.result_sha256 == sha256(result_path.read_bytes()).hexdigest()


def test_retrieval_plan_original_fallback_has_no_generated_subnarratives(tmp_path: Path) -> None:
    topic = Topic("topic-fallback", "", "Use the original request.")
    _saved_decomposition(tmp_path, topic, fallback=True)

    loaded = report_module._load_retrieval_plan_context(
        handoff_manifest_path=tmp_path / "handoff.json",
        topic=_fake_topic(topic),
    )

    assert loaded.used_fallback is True
    assert loaded.subnarratives == ()


@pytest.mark.parametrize("field", ["id", "narrative"])
def test_retrieval_plan_rejects_wrong_saved_topic_identity(tmp_path: Path, field: str) -> None:
    topic = Topic("topic-identity", "", "The authenticated narrative.")
    manifest_path, result_path = _saved_decomposition(tmp_path, topic)
    payload = json.loads(result_path.read_bytes())
    payload["topic"][field] = "different" if field == "id" else "different narrative"
    result_bytes = _json_bytes(payload)
    result_path.write_bytes(result_bytes)
    manifest = json.loads(manifest_path.read_bytes())
    manifest["result_bytes"] = len(result_bytes)
    manifest["result_sha256"] = sha256(result_bytes).hexdigest()
    manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises(ValueError, match="narrative|topic|decomposition"):
        report_module._load_retrieval_plan_context(
            handoff_manifest_path=tmp_path / "handoff.json",
            topic=_fake_topic(topic),
        )


@pytest.mark.parametrize("mutation", ["noncanonical", "filename", "bytes", "sha", "manifest-fields", "planner"])
def test_retrieval_plan_rejects_checkpoint_integrity_failures(
    tmp_path: Path,
    mutation: str,
) -> None:
    topic = Topic("topic-invalid", "", "Validate this request.")
    manifest_path, result_path = _saved_decomposition(tmp_path, topic)
    if mutation == "noncanonical":
        result_path.write_bytes(result_path.read_bytes() + b" ")
    else:
        manifest = json.loads(manifest_path.read_bytes())
        if mutation == "filename":
            manifest["result_file"] = "other.json"
        elif mutation == "bytes":
            manifest["result_bytes"] += 1
        elif mutation == "sha":
            manifest["result_sha256"] = "0" * 64
        elif mutation == "manifest-fields":
            del manifest["planner"]
        else:
            manifest["planner"] = {"bad": object()}
        if mutation == "planner":
            manifest_path.write_bytes(b'{"planner":{"bad":null}}\n')
        else:
            manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises((ValueError, TypeError)):
        report_module._load_retrieval_plan_context(
            handoff_manifest_path=tmp_path / "handoff.json",
            topic=_fake_topic(topic),
        )


@pytest.mark.parametrize("which", ["topic", "manifest", "result"])
def test_retrieval_plan_rejects_symlinked_checkpoint_paths(tmp_path: Path, which: str) -> None:
    topic = Topic("topic-link", "", "Reject linked files.")
    manifest_path, result_path = _saved_decomposition(tmp_path, topic)
    if which == "topic":
        target = tmp_path / "real-topic"
        target.mkdir()
        (tmp_path / topic.id).rename(target)
        (tmp_path / topic.id).symlink_to(target, target_is_directory=True)
    else:
        target = tmp_path / f"{which}-target"
        source = manifest_path if which == "manifest" else result_path
        target.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(target)

    with pytest.raises((ValueError, OSError)):
        report_module._load_retrieval_plan_context(
            handoff_manifest_path=tmp_path / "handoff.json",
            topic=_fake_topic(topic),
        )


def test_retrieval_plan_rejects_topic_root_outside_handoff_parent(tmp_path: Path) -> None:
    topic = Topic("../outside", "", "Do not escape.")
    _saved_decomposition(tmp_path, topic)

    with pytest.raises((ValueError, OSError)):
        report_module._load_retrieval_plan_context(
            handoff_manifest_path=tmp_path / "handoff.json",
            topic=_fake_topic(topic),
        )


def _fake_evaluation(topic_id: str = "fixture") -> SimpleNamespace:
    return SimpleNamespace(
        bound_input=SimpleNamespace(topic_id=topic_id, nuggets=()),
        plan=SimpleNamespace(obligations=()),
        report=SimpleNamespace(
            required_coverage=0.0,
            strict_full_rate=0.0,
            label_counts={"full": 0, "partial": 0, "unsupported": 0},
        ),
    )


def _patch_discovery_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    topics: tuple[SimpleNamespace, ...],
) -> None:
    handoff = SimpleNamespace(topics=topics)
    monkeypatch.setattr(report_module, "load_generation_handoff", lambda _path: handoff)

    def select(_handoff: object, topic_ids: object) -> tuple[SimpleNamespace, ...]:
        if topic_ids is None:
            return topics
        requested = tuple(topic_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate topic ID")
        by_id = {topic.topic_id: topic for topic in topics}
        if any(topic_id not in by_id for topic_id in requested):
            raise ValueError("unknown topic ID")
        return tuple(topic for topic in topics if topic.topic_id in requested)

    monkeypatch.setattr(report_module, "select_generation_topics", select)
    monkeypatch.setattr(
        report_module,
        "load_completed_coverage_evaluation",
        lambda **kwargs: _fake_evaluation(kwargs["topic_id"]),
    )
    monkeypatch.setattr(
        report_module,
        "_load_retrieval_plan_context",
        lambda **_kwargs: RetrievalPlanContext(
            used_fallback=True,
            subnarratives=(),
            planner_identity={"fixture": True},
            manifest_sha256="a" * 64,
            result_sha256="b" * 64,
        ),
    )


def _complete_coverage_dir(root: Path, topic_id: str) -> None:
    topic_root = root / topic_id
    topic_root.mkdir()
    for name in ("input.json", "plan.json", "judgments.json", "report.json", "manifest.json"):
        (topic_root / name).write_text("fixture")


def test_report_data_discovers_manifest_order_and_preserves_explicit_selector_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = (
        SimpleNamespace(topic_id="t2", narrative="Narrative two"),
        SimpleNamespace(topic_id="t1", narrative="Narrative one"),
    )
    _patch_discovery_dependencies(monkeypatch, topics)
    coverage_root = tmp_path / "coverage"
    coverage_root.mkdir()
    _complete_coverage_dir(coverage_root, "t2")
    _complete_coverage_dir(coverage_root, "t1")

    default_data = load_coverage_report_data(
        handoff_manifest_path=tmp_path / "handoff.json",
        coverage_root=coverage_root,
    )
    selected_data = load_coverage_report_data(
        handoff_manifest_path=tmp_path / "handoff.json",
        coverage_root=coverage_root,
        topic_ids=("t1", "t2"),
    )

    assert [topic.evaluation.bound_input.topic_id for topic in default_data.topics] == [
        "t2",
        "t1",
    ]
    assert [topic.evaluation.bound_input.topic_id for topic in selected_data.topics] == [
        "t1",
        "t2",
    ]


@pytest.mark.parametrize("topic_ids", [("missing",), ("t1", "t1")])
def test_report_data_rejects_unknown_or_duplicate_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    topic_ids: tuple[str, ...],
) -> None:
    topics = (SimpleNamespace(topic_id="t1", narrative="Narrative one"),)
    _patch_discovery_dependencies(monkeypatch, topics)
    coverage_root = tmp_path / "coverage"
    coverage_root.mkdir()
    _complete_coverage_dir(coverage_root, "t1")

    with pytest.raises(ValueError):
        load_coverage_report_data(
            handoff_manifest_path=tmp_path / "handoff.json",
            coverage_root=coverage_root,
            topic_ids=topic_ids,
        )


def test_report_data_rejects_unknown_manifest_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topics = (SimpleNamespace(topic_id="t1", narrative="Narrative one"),)
    _patch_discovery_dependencies(monkeypatch, topics)
    coverage_root = tmp_path / "coverage"
    coverage_root.mkdir()
    _complete_coverage_dir(coverage_root, "t1")
    _complete_coverage_dir(coverage_root, "unknown")

    with pytest.raises(ValueError, match="unknown"):
        load_coverage_report_data(
            handoff_manifest_path=tmp_path / "handoff.json",
            coverage_root=coverage_root,
        )


def test_report_data_rejects_known_incomplete_directory_and_empty_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = (SimpleNamespace(topic_id="t1", narrative="Narrative one"),)
    _patch_discovery_dependencies(monkeypatch, topics)
    coverage_root = tmp_path / "coverage"
    coverage_root.mkdir()
    incomplete = coverage_root / "t1"
    incomplete.mkdir()
    (incomplete / "report.json").write_text("fixture")

    with pytest.raises(ValueError, match="incomplete"):
        load_coverage_report_data(
            handoff_manifest_path=tmp_path / "handoff.json",
            coverage_root=coverage_root,
        )

    for child in incomplete.iterdir():
        child.unlink()
    incomplete.rmdir()
    with pytest.raises(ValueError, match="missing"):
        load_coverage_report_data(
            handoff_manifest_path=tmp_path / "handoff.json",
            coverage_root=coverage_root,
        )


def test_report_data_ignores_unrelated_regular_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topics = (SimpleNamespace(topic_id="t1", narrative="Narrative one"),)
    _patch_discovery_dependencies(monkeypatch, topics)
    coverage_root = tmp_path / "coverage"
    coverage_root.mkdir()
    (coverage_root / "README.txt").write_text("unrelated")
    _complete_coverage_dir(coverage_root, "t1")

    data = load_coverage_report_data(
        handoff_manifest_path=tmp_path / "handoff.json",
        coverage_root=coverage_root,
    )

    assert len(data.topics) == 1


class _ReportHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_chunks: list[str] = []
        self.tags: list[str] = []
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        for key, value in attrs:
            if key == "id" and value is not None:
                self.ids.append(value)

    def handle_data(self, data: str) -> None:
        self.text_chunks.append(data)


class _ReportSemanticStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.description_list_depth = 0
        self.orphan_description_items: list[str] = []
        self.heading_levels: list[int] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "dl":
            self.description_list_depth += 1
        elif tag in {"dt", "dd"} and self.description_list_depth == 0:
            self.orphan_description_items.append(tag)
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self.heading_levels.append(int(tag[1]))

    def handle_endtag(self, tag: str) -> None:
        if tag == "dl":
            self.description_list_depth -= 1


def _html_fixture_data() -> tuple[CoverageReportData, dict[str, str]]:
    values = {
        "narrative": '<script>alert("narrative")</script> & "quotes" \u2028 \u2029',
        "facet": '<script>alert("facet")</script> & "quotes" \u2028 \u2029',
        "requirement": '<script>alert("requirement")</script> & "quotes" \u2028 \u2029',
        "support_test": '<script>alert("support")</script> & "quotes" \u2028 \u2029',
        "span": '<script>alert("span")</script> & "quotes" \u2028 \u2029',
        "missing": '<script>alert("missing")</script> & "quotes" \u2028 \u2029',
        "subnarrative": '<script>alert("subnarrative")</script> & "quotes" \u2028 \u2029',
        "query": '<script>alert("query")</script> & "quotes" \u2028 \u2029',
        "nugget": '<script>alert("nugget")</script> & "quotes" \u2028 \u2029',
        "planner": '<script>alert("planner")</script> & "quotes" \u2028 \u2029',
    }
    obligation = CoverageObligation(
        obligation_id="obligation-1",
        requirement=values["requirement"],
        support_test=values["support_test"],
        kind="required_explicit",
        narrative_spans=(values["span"],),
    )
    plan = FrozenPlan(
        narrative=values["narrative"],
        schema_version="retrieval_nugget_plan_v1",
        facets=(CoverageFacet("facet-1", values["facet"], (obligation,)),),
        unmapped_narrative_spans=("unmapped span",),
        canonical_bytes=b"{}",
        plan_sha256="a" * 64,
    )
    nugget = CoverageNugget("canonical-id-1", values["nugget"])
    judgment = CoverageJudgment(
        obligation_id=obligation.obligation_id,
        label="partial",
        supporting_nugget_ids=(nugget.nugget_id,),
        missing_elements=values["missing"],
    )
    evaluation = SimpleNamespace(
        bound_input=SimpleNamespace(
            topic_id="topic-safe",
            narrative=values["narrative"],
            nuggets=(nugget,),
            manifest_sha256="d" * 64,
            narrative_sha256="e" * 64,
            nugget_text_sha256s=("f" * 64,),
        ),
        identity=EvaluatorIdentity("eval-v1", values["planner"], "judge-v1", "planner-model", "judge-model"),
        plan=plan,
        judgments=(judgment,),
        report=CoverageReport(
            plan_sha256=plan.plan_sha256,
            identity=EvaluatorIdentity("eval-v1", values["planner"], "judge-v1", "planner-model", "judge-model"),
            required_coverage=0.5,
            strict_full_rate=0.0,
            supplemental_coverage=None,
            label_counts={"full": 0, "partial": 1, "unsupported": 0},
            facet_scores={"facet-1": 0.5},
            judgments=(judgment,),
            unmapped_narrative_spans=("unmapped span",),
            uncited_nugget_ids=(),
            uncited_nugget_aliases=(),
        ),
        artifact_hashes={"input.json": "1" * 64, "report.json": "2" * 64},
        manifest_sha256="3" * 64,
    )
    topic = CoverageReportTopic(
        evaluation=evaluation,
        retrieval_plan=RetrievalPlanContext(
            used_fallback=False,
            subnarratives=(
                RetrievalSubnarrativeContext("sub-1", values["subnarrative"], (values["query"],)),
            ),
            planner_identity={"provider": values["planner"]},
            manifest_sha256="4" * 64,
            result_sha256="5" * 64,
        ),
    )
    data = CoverageReportData(
        topics=(topic,),
        summary=CoverageRunSummary(
            topic_count=1,
            nugget_count=1,
            required_obligation_count=1,
            supplemental_obligation_count=0,
            topic_macro_required_coverage=0.5,
            topic_macro_strict_full_rate=0.0,
            label_counts={"full": 0, "partial": 1, "unsupported": 0},
            perfect_required_topic_count=0,
        ),
    )
    return data, values


def test_renderer_escapes_semantic_text_and_is_deterministic() -> None:
    data, values = _html_fixture_data()

    first = report_module.render_coverage_report_html(data)
    second = report_module.render_coverage_report_html(data)
    source = first.decode("utf-8")
    parser = _ReportHTMLParser()
    parser.feed(source)
    visible_text = "".join(parser.text_chunks)

    assert first == second
    for value in values.values():
        assert value in visible_text
    assert "<script>alert(\"narrative\")</script>" not in source
    assert "DOC-SENTINEL" not in source
    assert "PASSAGE-SENTINEL" not in source
    assert "PROVIDER-BODY-SENTINEL" not in source
    assert 'src="http' not in source
    assert 'href="http' not in source
    assert "fetch(" not in source
    assert "XMLHttpRequest" not in source
    assert "WebSocket" not in source
    assert "EventSource" not in source
    assert "innerHTML" not in source
    assert parser.tags.count("script") == 2
    assert len(parser.ids) == len(set(parser.ids))
    assert source.count('class="nugget-inventory"') == 1
    assert source.count("n001") >= 2


def test_renderer_contract_has_theme_navigation_and_accessibility_landmarks() -> None:
    data, _ = _html_fixture_data()
    source = report_module.render_coverage_report_html(data).decode("utf-8")
    parser = _ReportHTMLParser()
    parser.feed(source)

    assert source.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in source
    assert 'name="viewport"' in source
    assert "<title>Retrieval nugget coverage report</title>" in source
    assert 'class="skip-link" href="#main-content"' in source
    assert '<main id="main-content">' in source
    assert 'role="search"' in source
    assert 'for="topic-search"' in source
    assert 'for="status-filter"' in source
    assert 'for="sort-topics"' in source
    assert 'role="status" aria-live="polite"' in source
    assert "<details" in source and "<summary>" in source
    assert '<details class="obligation" open>' in source
    assert '<details class="obligation" >' not in source
    assert "--color-bg:" in source
    assert "--color-surface:" in source
    assert "--color-text:" in source
    assert '[data-theme="light"]' in source
    assert '[data-theme="dark"]' in source
    assert "@media (prefers-color-scheme: dark)" in source
    assert "@media (prefers-reduced-motion: reduce)" in source
    assert "@media print" in source
    assert "history.pushState" in source
    assert "popstate" in source
    assert "hashchange" not in source
    assert 'data-theme-choice="light"' in source
    assert 'data-theme-choice="system"' in source
    assert 'data-theme-choice="dark"' in source
    assert 'aria-pressed="false"' in source
    assert len(parser.ids) == len(set(parser.ids))


def test_renderer_nests_description_items_in_description_lists() -> None:
    data, _ = _html_fixture_data()
    parser = _ReportSemanticStructureParser()
    parser.feed(report_module.render_coverage_report_html(data).decode("utf-8"))

    assert parser.orphan_description_items == []


def test_renderer_heading_levels_do_not_skip_depth() -> None:
    data, _ = _html_fixture_data()
    parser = _ReportSemanticStructureParser()
    parser.feed(report_module.render_coverage_report_html(data).decode("utf-8"))

    assert all(
        current <= previous + 1
        for previous, current in zip(parser.heading_levels, parser.heading_levels[1:])
    )


def test_renderer_explains_metric_denominators_in_standalone_report() -> None:
    data, _ = _html_fixture_data()
    parser = _ReportHTMLParser()
    parser.feed(report_module.render_coverage_report_html(data).decode("utf-8"))
    visible_text = " ".join(" ".join(parser.text_chunks).split())

    assert "Required coverage is a facet-macro average over required obligations." in visible_text
    assert "Strict-full rate is an obligation-micro share over required obligations." in visible_text
    assert "Label counts include required and supplemental obligations." in visible_text


def test_renderer_focus_contract_moves_into_detail_and_restores_overview_target() -> None:
    data, _ = _html_fixture_data()
    source = report_module.render_coverage_report_html(data).decode("utf-8")

    assert 'id="back-to-overview"' in source
    assert 'id="topic-list"' in source
    assert 'aria-controls="topic-1-topic-safe-' in source
    assert "var returnFocusButton = null;" in source
    assert "backButton.focus();" in source
    assert "var focusTarget = restoreFocus ? returnFocusButton : null;" in source
    assert "focusTarget.focus();" in source
    assert 'showOverview("", true);' in source
    assert 'showOverview("That link was not a valid topic; showing the overview.", true);' in source
    assert 'showOverview("That topic was not found; showing the overview.", true);' in source


def _patch_cli_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    body: bytes = b"<!doctype html>\nfixture\n",
) -> list[tuple[Path, Path, tuple[str, ...]]]:
    calls: list[tuple[Path, Path, tuple[str, ...]]] = []
    data, _ = _html_fixture_data()

    def load(*, handoff_manifest_path: Path, coverage_root: Path, topic_ids: tuple[str, ...]):
        calls.append((handoff_manifest_path, coverage_root, tuple(topic_ids)))
        return data

    monkeypatch.setattr(report_module, "load_coverage_report_data", load)
    monkeypatch.setattr(report_module, "render_coverage_report_html", lambda _data: body)
    return calls


def test_cli_publishes_renderer_bytes_and_reports_zero_hosted_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _patch_cli_dependencies(monkeypatch, body=b"renderer bytes\n")
    handoff = tmp_path / "handoff.json"
    coverage_root = tmp_path / "coverage"
    coverage_root.mkdir()
    output = tmp_path / "report.html"

    assert main(
        [
            "--handoff-manifest",
            str(handoff),
            "--coverage-root",
            str(coverage_root),
            "--output",
            str(output),
            "--topic",
            "topic-safe",
            "--topic",
            "topic-other",
        ]
    ) == 0

    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {
        "status": "ok",
        "selected_topic_count": 1,
        "output": str(output),
        "output_sha256": sha256(b"renderer bytes\n").hexdigest(),
        "hosted_calls": 0,
    }
    assert output.read_bytes() == b"renderer bytes\n"
    assert calls == [(handoff, coverage_root, ("topic-safe", "topic-other"))]


@pytest.mark.parametrize(
    ("argv", "message", "expected_stage", "expected_reason"),
    [
        (("--topic", "missing"), "unknown", "load", "coverage report inputs are invalid"),
        (("--topic", "topic-safe", "--topic", "topic-safe"), "duplicate", "config", "report configuration is invalid"),
        (("--topic", ""), "empty", "config", "report configuration is invalid"),
        (("--topic", "incomplete"), "incomplete", "load", "coverage report inputs are invalid"),
        (("--topic", "contradictory"), "contradictory", "load", "coverage report inputs are invalid"),
    ],
)
def test_cli_rejects_invalid_topic_state_with_structured_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: tuple[str, ...],
    message: str,
    expected_stage: str,
    expected_reason: str,
) -> None:
    def load(**kwargs: object) -> CoverageReportData:
        del kwargs
        raise ValueError(f"{message}: {tmp_path / 'private-checkpoint.json'}")

    monkeypatch.setattr(report_module, "load_coverage_report_data", load)
    output = tmp_path / "report.html"
    output.write_bytes(b"keep me")
    result = main(
        [
            "--handoff-manifest",
            str(tmp_path / "handoff.json"),
            "--coverage-root",
            str(tmp_path / "coverage"),
            "--output",
            str(output),
            *argv,
        ]
    )

    assert result != 0
    assert output.read_bytes() == b"keep me"
    stderr = capsys.readouterr().err
    assert "traceback" not in stderr.casefold()
    error = json.loads(stderr)
    assert error == {
        "status": "error",
        "error": {
            "type": "retrieval_nugget_coverage_report_error",
            "stage": expected_stage,
            "reason": expected_reason,
        },
    }
    assert message not in stderr
    assert str(tmp_path) not in stderr


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda tmp_path: tmp_path / "report.txt",
        lambda tmp_path: _symlink_output(tmp_path),
        lambda tmp_path: _symlink_parent_output(tmp_path),
    ],
)
def test_publish_rejects_unsafe_output_without_changing_existing_file(
    tmp_path: Path,
    path_factory: object,
) -> None:
    output = path_factory(tmp_path)
    existing = output.resolve() if output.is_symlink() else output
    existing.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_symlink():
        output.write_bytes(b"keep me")
    else:
        existing.write_bytes(b"keep me")

    with pytest.raises((ValueError, OSError)):
        publish_coverage_report(output, b"new bytes")
    assert existing.read_bytes() == b"keep me"


def _symlink_output(tmp_path: Path) -> Path:
    target = tmp_path / "target.html"
    target.write_bytes(b"keep me")
    output = tmp_path / "report.html"
    output.symlink_to(target)
    return output


def _symlink_parent_output(tmp_path: Path) -> Path:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    output = tmp_path / "linked-parent" / "report.html"
    output.parent.symlink_to(real_parent, target_is_directory=True)
    (real_parent / "report.html").write_bytes(b"keep me")
    return output


def test_publish_atomically_replaces_regular_report_and_cleans_temp_files(tmp_path: Path) -> None:
    output = tmp_path / "report.html"
    output.write_bytes(b"old bytes")

    assert publish_coverage_report(output, b"new bytes") == output
    assert output.read_bytes() == b"new bytes"
    assert list(tmp_path.iterdir()) == [output]


def test_publish_rejects_unwritable_existing_report_without_changing_it(tmp_path: Path) -> None:
    output = tmp_path / "report.html"
    output.write_bytes(b"old bytes")
    output.chmod(0o444)

    with pytest.raises(OSError, match="not writable"):
        publish_coverage_report(output, b"new bytes")
    assert output.read_bytes() == b"old bytes"


def test_publish_cleans_temporary_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.html"
    output.write_bytes(b"old bytes")

    def fail_replace(_source: str, _target: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(report_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        publish_coverage_report(output, b"new bytes")
    assert output.read_bytes() == b"old bytes"
    assert list(tmp_path.iterdir()) == [output]
