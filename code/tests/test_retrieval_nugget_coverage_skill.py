"""Contract tests for the retrieval-nugget-coverage skill route."""

from pathlib import Path
import re


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
    section = _route_section()
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
