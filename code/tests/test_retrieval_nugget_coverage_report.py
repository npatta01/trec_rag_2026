from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import trec_rag.retrieval_nugget_coverage_report as report_module
from trec_rag.facet_extraction import plan_facet_queries
from trec_rag.retrieval_nugget_coverage import (
    CoverageFacet,
    CoverageObligation,
    CoverageReport,
    FrozenPlan,
)
from trec_rag.retrieval_nugget_coverage_report import (
    CoverageReportTopic,
    RetrievalPlanContext,
    RetrievalSubnarrativeContext,
    load_coverage_report_data,
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
