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
