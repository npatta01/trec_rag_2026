"""Regression tests for exact friendly-report privacy sink binding."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from offline_evaluation_fixture import TopicSpec, build_run
from trec_rag import friendly_report
from trec_rag.friendly_report import (
    ReportPrivacyError,
    denylist_from_bundle,
    write_report,
)
from trec_rag.offline_evaluation import JudgeOutcome, JudgeSettings, build_evaluation_bundle


REPOSITORY_ROOT = Path(__file__).parents[2]


def _settings() -> JudgeSettings:
    return JudgeSettings(
        provider="fixture-provider",
        model="fixture/model-1",
        thinking="medium",
        temperature=None,
        system_prompt="fixture system prompt",
        agent_binary="fixture-agent",
        extension_identity="none",
    )


def _bundle(tmp_path: Path):
    (tmp_path / "fixture").mkdir()
    fixture = build_run(
        tmp_path / "fixture",
        [TopicSpec("topic", "A fixture narrative.")],
    )
    bundle = build_evaluation_bundle(
        retrieval_config_path=fixture.retrieval_config,
        rag_config_path=fixture.rag_config,
        work_dir=fixture.root / "work",
        repository_root=REPOSITORY_ROOT,
        cache_root=fixture.root / "cache",
        judge=lambda _task: JudgeOutcome(status="completed", support_label="FS"),
        judge_settings=_settings(),
        created_utc="2026-01-01T00:00:00+00:00",
    )
    return fixture, bundle


def _selected_passage(bundle) -> str:
    row = json.loads((bundle.work_dir / "support_input.jsonl").read_text(encoding="utf-8"))
    return next(iter(row["segments"].values()))


def _render_with_response_prefix(fragment: str):
    original = friendly_report.render_report

    def render(view):
        page = original(view)
        marker = '<span class="citations">'
        assert marker in page
        return page.replace(marker, f"{fragment}{marker}", 1)

    return render


@pytest.mark.parametrize(
    "strong",
    [
        True,
        False,
    ],
    ids=["strong-injection", "plain-injection"],
)
def test_selected_passage_injected_before_citations_is_not_authorized(
    tmp_path: Path, strong: bool
) -> None:
    fixture, bundle = _bundle(tmp_path)
    passage = _selected_passage(bundle)
    fragment = f"<strong>{passage}</strong>" if strong else passage
    with patch.object(
        friendly_report,
        "render_report",
        side_effect=_render_with_response_prefix(fragment),
    ):
        with pytest.raises(ReportPrivacyError, match="expected response rendering"):
            write_report(
                bundle.manifest,
                fixture.root / "injected.html",
                denylist=denylist_from_bundle(bundle.work_dir),
            )


def test_legitimate_answer_containing_selected_passage_is_redacted_and_passes(
    tmp_path: Path,
) -> None:
    fixture, bundle = _bundle(tmp_path)
    passage = _selected_passage(bundle)
    manifest = deepcopy(bundle.manifest)
    manifest["topics"][0]["answer"][0]["text"] = f"The answer repeats {passage}."

    output = fixture.root / "legitimate.html"
    write_report(
        manifest,
        output,
        denylist=denylist_from_bundle(bundle.work_dir),
    )

    assert output.is_file()
    assert f"The answer repeats {passage}." in output.read_text(encoding="utf-8")


def test_sink_count_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture, bundle = _bundle(tmp_path)
    original = friendly_report.render_report

    def render_without_subnarrative(view):
        page = original(view)
        marker = '<p class="subnarrative-text">'
        start = page.index(marker)
        end = page.index("</p>", start) + len("</p>")
        return page[:start] + page[end:]

    with patch.object(friendly_report, "render_report", side_effect=render_without_subnarrative):
        with pytest.raises(ReportPrivacyError, match="expected subnarrative sink count"):
            write_report(
                bundle.manifest,
                fixture.root / "missing-sink.html",
                denylist=denylist_from_bundle(bundle.work_dir),
            )


def test_sink_order_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture, bundle = _bundle(tmp_path)
    original = friendly_report.render_report

    def render_with_reordered_answers(view):
        page = original(view)
        first = '<p class="response-text">First supported answer.'
        second = '<p class="response-text">Second detail.'
        marker = "__SECOND_ANSWER__"
        assert page.count(first) == 1
        assert page.count(second) == 1
        return page.replace(first, marker, 1).replace(second, first, 1).replace(marker, second, 1)

    with patch.object(friendly_report, "render_report", side_effect=render_with_reordered_answers):
        with pytest.raises(ReportPrivacyError, match="expected response rendering"):
            write_report(
                bundle.manifest,
                fixture.root / "reordered-sinks.html",
                denylist=denylist_from_bundle(bundle.work_dir),
            )
