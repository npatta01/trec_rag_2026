"""Contract tests for the privacy-safe multipage debug-report bundle models."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path

import pytest

from offline_evaluation_fixture import TopicSpec, build_run
from trec_rag.competition_debug_report import load_debug_report_data
from trec_rag.offline_evaluation import (
    EvaluationError,
    JudgeOutcome,
    JudgeSettings,
    build_evaluation_bundle,
)


REPOSITORY_ROOT = Path(__file__).parents[2]


def test_summary_projection_projects_only_safe_counts_and_attention_state(
    tmp_path: Path,
) -> None:
    # Keep the first production import in the test body so the initial RED run
    # reports a genuine test failure rather than a collection error.
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


def test_summary_projection_counts_generated_queries_nuggets_and_safe_topic_links(
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


def _evaluation_fixture(tmp_path: Path):
    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    return fixture, build_evaluation_bundle(
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


def test_evaluation_overlay_preserves_families_scope_and_unavailability(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import load_evaluation_overlay

    _fixture, bundle = _evaluation_fixture(tmp_path)
    overlay = load_evaluation_overlay(bundle.manifest_path, ("beta-topic",))

    assert overlay.topic_ids == ("beta-topic",)
    assert [family.key for family in overlay.families] == [
        "retrieval",
        "nugget_coverage",
        "citation_support",
    ]
    retrieval = overlay.families[0]
    assert retrieval.label == "Retrieval relevance"
    assert "ndcg@10" in retrieval.definitions
    assert retrieval.macro_rule.startswith("Unweighted mean")
    assert "ndcg@10" in retrieval.per_topic["beta-topic"]
    assert "run_id" not in retrieval.per_topic["beta-topic"]
    assert retrieval.macro_availability.available is False
    assert "2-topic evaluation scope" in retrieval.macro_availability.reason
    nuggets = overlay.families[1]
    assert nuggets.macro == {}
    assert nuggets.macro_availability.available is False
    assert "no released gold-nugget file" in nuggets.macro_availability.reason


def _mutated_manifest(tmp_path: Path, mutate) -> Path:
    _fixture, bundle = _evaluation_fixture(tmp_path)
    payload = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "mutated-evaluation-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "case,mutate,selected",
    (
        (
            "missing selected topic",
            lambda _payload: None,
            ("missing-topic",),
        ),
        (
            "conflicting relative order",
            lambda _payload: None,
            ("beta-topic", "alpha-topic"),
        ),
        (
            "non-finite metric",
            lambda payload: payload["metrics"]["retrieval"]["per_topic"]["alpha-topic"].update(
                {"ndcg@10": float("nan")}
            ),
            ("alpha-topic", "beta-topic"),
        ),
        (
            "duplicate topic IDs",
            lambda payload: payload["scope"]["topic_ids"].append("alpha-topic"),
            ("alpha-topic", "beta-topic"),
        ),
        (
            "unavailable state without reason",
            lambda payload: payload["metrics"]["nugget_coverage"]["macro_availability"].update(
                {"available": False, "reason": ""}
            ),
            ("alpha-topic", "beta-topic"),
        ),
        (
            "unknown evaluation schema",
            lambda payload: payload.update({"schema_version": "unknown-evaluation-schema"}),
            ("alpha-topic", "beta-topic"),
        ),
    ),
)
def test_evaluation_overlay_rejects_invalid_manifest_contract(
    tmp_path: Path, case: str, mutate, selected: tuple[str, ...]
) -> None:
    from trec_rag.competition_debug_bundle import load_evaluation_overlay

    path = _mutated_manifest(tmp_path, mutate)
    with pytest.raises(EvaluationError):
        load_evaluation_overlay(path, selected)


def test_evaluation_overlay_rejects_missing_available_selected_metric_row(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import load_evaluation_overlay

    path = _mutated_manifest(
        tmp_path,
        lambda payload: payload["metrics"]["retrieval"]["per_topic"].pop(
            "beta-topic"
        ),
    )
    with pytest.raises(EvaluationError):
        load_evaluation_overlay(path, ("beta-topic",))


def test_evaluation_overlay_rejects_incomplete_available_macro_metrics(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import load_evaluation_overlay

    path = _mutated_manifest(
        tmp_path,
        lambda payload: payload["metrics"]["retrieval"]["macro"].pop("ndcg@10"),
    )
    with pytest.raises(EvaluationError):
        load_evaluation_overlay(path, ("alpha-topic", "beta-topic"))


def test_evaluation_overlay_rejects_manifest_replacement_between_snapshot_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_rag.competition_debug_bundle as debug_bundle

    _fixture, bundle = _evaluation_fixture(tmp_path)
    original_load_manifest = debug_bundle.load_manifest

    def replace_before_load(path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metrics"]["retrieval"]["per_topic"]["alpha-topic"]["ndcg@10"] = 0.5
        path.write_text(json.dumps(payload), encoding="utf-8")
        return original_load_manifest(path)

    monkeypatch.setattr(debug_bundle, "load_manifest", replace_before_load)
    try:
        overlay = debug_bundle.load_evaluation_overlay(
            bundle.manifest_path, ("alpha-topic", "beta-topic")
        )
    except EvaluationError:
        return
    assert overlay.manifest_sha256 == sha256(bundle.manifest_path.read_bytes()).hexdigest()


def test_evaluation_overlay_rejects_aba_manifest_replacement_at_load_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_rag.competition_debug_bundle as debug_bundle

    _fixture, bundle = _evaluation_fixture(tmp_path)
    snapshot = bundle.manifest_path.read_bytes()
    snapshot_manifest = json.loads(snapshot)
    original_load_manifest = debug_bundle.load_manifest

    def replace_and_restore(path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metrics"]["retrieval"]["per_topic"]["alpha-topic"]["ndcg@10"] = 0.5
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = original_load_manifest(path)
        path.write_bytes(snapshot)
        return loaded

    monkeypatch.setattr(debug_bundle, "load_manifest", replace_and_restore)
    try:
        overlay = debug_bundle.load_evaluation_overlay(
            bundle.manifest_path, ("alpha-topic", "beta-topic")
        )
    except EvaluationError:
        return
    retrieval = overlay.families[0]
    assert retrieval.per_topic["alpha-topic"]["ndcg@10"] == snapshot_manifest[
        "metrics"
    ]["retrieval"]["per_topic"]["alpha-topic"]["ndcg@10"]
    assert overlay.manifest_sha256 == sha256(snapshot).hexdigest()
