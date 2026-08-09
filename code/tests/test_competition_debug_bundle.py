"""Contract tests for the privacy-safe multipage debug-report bundle models."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path

import pytest

from offline_evaluation_fixture import TopicSpec, build_run
from trec_rag.competition_debug_report import load_debug_report_data
from trec_rag.friendly_report import ReportPrivacyError
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


def _evaluated_two_topic_data(tmp_path: Path):
    from trec_rag.competition_debug_bundle import load_evaluation_overlay

    fixture = build_run(
        tmp_path,
        (
            TopicSpec("alpha-topic", "private alpha narrative </script><script>"),
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
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

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
    assert "</script><script>" not in page
    assert "<script>private alpha narrative" not in page
    assert all(docid not in page for docid in private_docids)
    assert 'href="topics/alpha-topic.html"' in page
    assert 'data-sort-kind="number"' in page
    assert 'type="search"' in page


def test_summary_html_keeps_metric_families_definitions_and_macro_rules(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

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
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

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
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    data = load_debug_report_data(fixture.retrieval_config)

    with pytest.raises(ReportPrivacyError, match="private input value"):
        render_bundle_summary(
            build_run_summary(data, None),
            denylist=("alpha-topic",),
        )


def test_summary_denylist_real_fixture_can_render_without_invalid_document_digest_access(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import (
        _summary_denylist,
        build_run_summary,
        render_bundle_summary,
    )

    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    data = load_debug_report_data(fixture.retrieval_config)
    denylist = _summary_denylist(data)

    page = render_bundle_summary(build_run_summary(data, None), denylist=denylist)

    assert page.startswith("<!doctype html>")
    assert denylist


def test_summary_denylist_collects_all_query_and_retrieval_digests(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import _summary_denylist

    fixture = build_run(tmp_path, (TopicSpec("alpha-topic", "alpha"),))
    data = load_debug_report_data(fixture.retrieval_config)
    topic = data.topics[0]
    subnarrative = replace(
        topic.subnarratives[0],
        semantic_query_sha256="1" * 64,
        bm25_query_sha256s=("2" * 64,),
    )
    lane_provenance = replace(
        topic.new_documents[0].lane_provenance[0], text_sha256="3" * 64
    )
    new_document = replace(
        topic.new_documents[0], lane_provenance=(lane_provenance,)
    )
    retrieval_score = dict(topic.retrieval_output.documents[0].subnarrative_scores[0])
    retrieval_score.update(
        {
            "semantic_query_sha256": "4" * 64,
            "bm25_query_sha256s": ["5" * 64],
            "text_sha256": "6" * 64,
        }
    )
    retrieval_document = replace(
        topic.retrieval_output.documents[0],
        subnarrative_scores=(retrieval_score,),
    )
    updated_topic = replace(
        topic,
        subnarratives=(subnarrative,),
        new_documents=(new_document, *topic.new_documents[1:]),
        retrieval_output=replace(
            topic.retrieval_output, documents=(retrieval_document,)
        ),
    )
    updated_data = replace(data, topics=(updated_topic,))

    denylist = set(_summary_denylist(updated_data))

    assert {str(index) * 64 for index in range(1, 7)} <= denylist


def test_summary_static_rows_remain_official_order_before_attention_enhancement(
    tmp_path: Path,
) -> None:
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    data = load_debug_report_data(fixture.retrieval_config)
    fallback_result = replace(data.topics[1].canonical_results[0], state="fallback_extractive")
    fallback_topic = replace(
        data.topics[1],
        original_only_fallback=True,
        canonical_results=(fallback_result,),
    )
    summary = build_run_summary(replace(data, topics=(data.topics[0], fallback_topic)), None)

    page = render_bundle_summary(summary, denylist=())

    assert page.index('data-topic-id="alpha-topic"') < page.index('data-topic-id="beta-topic"')


def test_summary_needs_attention_reset_clears_metric_sort_direction() -> None:
    from trec_rag.competition_debug_bundle import _SUMMARY_SCRIPT

    reset_start = _SUMMARY_SCRIPT.index('reset?.addEventListener("click"')
    reset_end = _SUMMARY_SCRIPT.index("  });\n  sortNeedsAttention();", reset_start)
    reset_block = _SUMMARY_SCRIPT[reset_start:reset_end]

    assert 'button.dataset.direction = ""' in reset_block
    assert 'header.setAttribute("aria-sort", "none")' in reset_block


def test_summary_health_cards_report_each_fallback_kind_count(tmp_path: Path) -> None:
    from trec_rag.competition_debug_bundle import build_run_summary, render_bundle_summary

    fixture = build_run(
        tmp_path,
        (TopicSpec("alpha-topic", "alpha"), TopicSpec("beta-topic", "beta")),
    )
    data = load_debug_report_data(fixture.retrieval_config)
    fallback_result = replace(data.topics[0].canonical_results[0], state="fallback_extractive")
    fallback_alpha = replace(
        data.topics[0],
        canonical_results=(fallback_result,),
    )
    fallback_beta = replace(data.topics[1], original_only_fallback=True)
    summary = build_run_summary(replace(data, topics=(fallback_alpha, fallback_beta)), None)

    page = render_bundle_summary(summary, denylist=())

    assert "fallback_extractive: 1" in page
    assert "original_only: 1" in page
