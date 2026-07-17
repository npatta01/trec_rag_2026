import csv
from pathlib import Path

import yaml

from trec_rag.experiment_records import build_experiment_indexes


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "reports" / "experiments"


def test_history_names_merged_decisions_and_separates_unmerged_work() -> None:
    text = (REPO_ROOT / "experiment.md").read_text(encoding="utf-8")

    required_sections = (
        "## Current retained configuration",
        "## Merged experiment chronology",
        "## Tried, learned, rejected, retained",
        "## Topic 31/300 postmortem",
        "## Active / unmerged",
        "## Next actions",
    )
    positions = [text.index(section) for section in required_sections]

    assert positions == sorted(positions)
    assert "all_topic_tethered_facet_validation_v1" in text
    assert "Retained" in text and "Rejected" in text
    assert "unjudged" in text.lower() and "unknown" in text.lower()
    assert "source-diverse RAG evidence selection" in text


def test_runs_index_contains_all_topic_record_once() -> None:
    generated_rows, _ = build_experiment_indexes(EXPERIMENTS)
    with (EXPERIMENTS / "runs.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    assert sum(
        row["experiment_id"] == "all_topic_tethered_facet_validation_v1"
        for row in rows
    ) == 1
    assert rows == generated_rows


def test_manifests_use_portable_repository_relative_artifact_paths() -> None:
    all_topic = yaml.safe_load(
        (EXPERIMENTS / "all_topic_tethered_facet_validation_v1" / "manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    coverage = yaml.safe_load(
        (EXPERIMENTS / "bm25_candidate_pool_coverage_v1" / "manifest.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert all_topic["experiment"]["split"] == "dev"
    assert all_topic["data"]["topic_count"] == 22
    assert all_topic["config"] == {
        "retriever": "pyserini_remote",
        "index": "climbmix-400b",
        "query_source": "original_narrative_and_reviewed_facets",
        "ranking": "family_balanced_rrf_and_preregistered_dual_arms",
    }
    assert all_topic["data"]["qrels"].endswith(
        "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
    )
    assert "sealed roots" in all_topic["notes"].lower()
    assert all(
        not Path(path).is_absolute()
        for path in all_topic["tracked_record_files"].values()
    )
    assert coverage["local_artifacts"]["cache_dir"] == "cache/retrieval/pyserini_remote"


def test_completed_all_topic_plan_distinguishes_unperformed_steps() -> None:
    text = (
        REPO_ROOT
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-07-16-all-topic-tethered-facet-validation.md"
    ).read_text(encoding="utf-8")

    assert "## Completion summary — 2026-07-17" in text
    assert "Unperformed" in text
    assert "private" in text.lower()
