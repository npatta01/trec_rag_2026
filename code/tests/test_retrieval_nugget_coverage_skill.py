"""Contract tests for the retrieval-nugget-coverage skill route."""

from pathlib import Path
import json
import re
import shlex

import trec_rag.retrieval_nugget_coverage_report as report_module


REPOSITORY_ROOT = Path(__file__).parents[2]
SKILL = (
    REPOSITORY_ROOT
    / ".agents"
    / "skills"
    / "trec-rag-competition-debug-report"
    / "SKILL.md"
)


def _route_section() -> str:
    body = SKILL.read_text(encoding="utf-8")
    match = re.search(
        r"^## Retrieval Nugget Coverage\n(?P<section>.*?)(?=^## |\Z)",
        body,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "the retrieval nugget coverage route is missing"
    return match.group("section")


def test_retrieval_nugget_coverage_route_documents_the_cli_contract() -> None:
    section = " ".join(_route_section().split())
    for required in (
        "Retrieval Nugget Coverage",
        "-m trec_rag.retrieval_nugget_coverage",
        "--handoff-manifest",
        "--topic",
        "--allow-hosted-calls",
        "cache-only",
        "one narrative",
        "canonical retrieval nugget text",
        "maximum of two hosted calls",
        "defaults are `openai/gpt-5.6-sol` for each",
    ):
        assert required in section


def test_route_isolated_from_year_specific_inputs_and_unsafe_pipeline_actions() -> None:
    section = _route_section().casefold()
    for forbidden in (
        "2025",
        "gold",
        "qrels",
        "-m trec_rag.competition_retrieval",
        "-m trec_rag.competition_rag",
        "-m trec_rag.rerank",
        "passage upload",
        "upload passages",
        "permission to serve",
        "permission to publish",
        "publishing permission",
    ):
        assert forbidden not in section


def test_route_documents_resume_branch_for_existing_work_directory() -> None:
    section = _route_section()
    normalized = " ".join(section.split())
    assert "fresh namespace" in normalized
    assert "existing or partial work directory" in normalized
    assert "same two-command cache-first sequence" in normalized
    assert section.count("--work-dir WORK_DIR") >= 2
    assert section.count("--mode resume") >= 2


def test_route_explains_cache_only_resume_can_publish_missing_derived_artifacts() -> None:
    normalized = " ".join(_route_section().split()).casefold()
    assert "cache-only resume makes zero hosted calls" in normalized
    assert "may locally publish missing derived" in normalized
    assert "report.json" in normalized and "manifest.json" in normalized
    assert "only a fresh cache-only create is write-free" in normalized


def test_explicit_nugget_request_authorizes_only_planner_and_judge_calls() -> None:
    normalized = " ".join(_route_section().split())
    assert "explicitly asks to evaluate, score, or judge retrieval nugget coverage" in normalized
    assert "inspect, explain, debug, or audit" in normalized
    assert "does not authorize hosted calls" in normalized
    assert "only these planner and judge calls" in normalized


def test_html_report_route_executes_exact_zero_hosted_call_cli(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    section = _route_section()
    match = re.search(
        r"```bash\n(?P<command>\.venv/bin/python -m trec_rag\.retrieval_nugget_coverage_report .*?)\n```",
        section,
        flags=re.DOTALL,
    )
    assert match is not None, "the HTML report route command is missing"
    command = " ".join(line.strip().rstrip("\\") for line in match.group("command").splitlines())
    arguments = shlex.split(command)[3:]
    handoff = tmp_path / "handoff.json"
    coverage_root = tmp_path / "coverage"
    output = tmp_path / "coverage.html"
    coverage_root.mkdir()
    substitutions = {
        "HANDOFF_MANIFEST": str(handoff),
        "COVERAGE_ROOT": str(coverage_root),
        "REPORT_HTML": str(output),
        "TOPIC_ID": "topic-safe",
    }
    arguments = [substitutions.get(argument, argument) for argument in arguments]
    hosted_calls = []

    monkeypatch.setattr(
        report_module,
        "load_coverage_report_data",
        lambda **kwargs: _route_fixture_data(kwargs, hosted_calls),
    )
    monkeypatch.setattr(
        report_module,
        "render_coverage_report_html",
        lambda _data: b"route html\n",
    )

    assert report_module.main(arguments) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["hosted_calls"] == 0
    assert output.read_bytes() == b"route html\n"
    assert hosted_calls == []


def _route_fixture_data(kwargs: dict[str, object], hosted_calls: list[object]):
    if kwargs["topic_ids"] != ("topic-safe",):
        hosted_calls.append(kwargs["topic_ids"])
    return report_module.CoverageReportData(
        topics=(),
        summary=report_module.CoverageRunSummary(
            topic_count=0,
            nugget_count=0,
            required_obligation_count=0,
            supplemental_obligation_count=0,
            topic_macro_required_coverage=0.0,
            topic_macro_strict_full_rate=0.0,
            label_counts={},
            perfect_required_topic_count=0,
        ),
    )
