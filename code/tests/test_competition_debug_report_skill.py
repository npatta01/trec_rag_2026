"""Contract tests for the competition debug-report agent skill."""

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).parents[2]
SKILL = (
    REPOSITORY_ROOT
    / ".agents"
    / "skills"
    / "trec-rag-competition-debug-report"
    / "SKILL.md"
)


class CompetitionDebugReportSkillContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.body = SKILL.read_text(encoding="utf-8")

    def test_skill_lives_in_repository_agent_discovery_tree(self) -> None:
        self.assertEqual(
            SKILL.relative_to(REPOSITORY_ROOT).parts,
            (
                ".agents",
                "skills",
                "trec-rag-competition-debug-report",
                "SKILL.md",
            ),
        )

    def test_trigger_covers_completed_run_inspection_requests(self) -> None:
        frontmatter = self.body.split("---", 2)[1]
        for trigger in ("inspect", "explain", "visualize", "audit", "debug"):
            self.assertIn(trigger, frontmatter)
        self.assertIn("completed TREC RAG competition", frontmatter)

    def test_invocation_uses_only_the_repository_post_run_cli(self) -> None:
        self.assertIn("Invoke only the repository CLI", self.body)
        self.assertIn(".venv/bin/python", self.body)
        # The documented form is the direct interpreter; no wrapper, no PYTHONPATH.
        self.assertNotIn("uv run --no-sync", self.body)
        self.assertNotIn("PYTHONPATH=", self.body)
        self.assertIn("-m trec_rag.competition_debug_report", self.body)
        self.assertIn("--retrieval-config RETRIEVAL_CONFIG", self.body)

    def test_privacy_and_no_rerun_boundaries_are_explicit(self) -> None:
        for forbidden in (
            "Never run retrieval",
            "reranking",
            "models",
            "hosted APIs",
            "never copy, serve, or publish",
        ):
            self.assertIn(forbidden, self.body)
        self.assertIn("must remain private", self.body)

    def test_ambiguous_completed_runs_require_one_path_question(self) -> None:
        self.assertIn("multiple completed retrieval runs remain genuinely ambiguous", self.body)
        self.assertIn("ask one short question for the retrieval-config path", self.body)
        self.assertIn("do not guess, run every candidate", self.body)

    def test_repository_discovery_uses_the_workspace_markers(self) -> None:
        discovery = (
            "walk upward from the current directory to the first root with the "
            "repository markers `pyproject.toml` and "
            "`code/trec_rag/competition_debug_report.py`; if none exists, ask "
            "for the repository path"
        )
        self.assertIn(discovery, self.body)

    def test_includes_literal_retrieval_only_and_rag_examples(self) -> None:
        retrieval_only = """.venv/bin/python \\
  -m trec_rag.competition_debug_report \\
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml"""
        retrieval_plus_rag = """.venv/bin/python \\
  -m trec_rag.competition_debug_report \\
  --retrieval-config configs/rag26_competition_retrieval_v2.yaml \\
  --rag-config configs/rag26_competition_rag_gpt_sol_v2.yaml"""
        self.assertIn(retrieval_only, self.body)
        self.assertIn(retrieval_plus_rag, self.body)

    def test_documents_the_executable_probe_then_resume_workflow(self) -> None:
        """The promised one-task probe must name the option that actually caps calls."""
        self.assertIn("--run-judge --judge-limit 1", self.body)
        self.assertIn("-m trec_rag.competition_evaluation_report", self.body)
        for requirement in (
            "hosted_calls: 1",
            "failed_judgments: 0",
            "stop and report it",
            "same** `--cache-dir`",
            "reused_from_cache >= 1",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, self.body)

    def test_workflow_is_cache_first_and_skips_a_needless_probe(self) -> None:
        self.assertIn("Always start without `--run-judge`", self.body)
        self.assertIn("Do not probe.", self.body)

    def test_probe_success_is_relative_not_an_absolute_count(self) -> None:
        """Pre-existing cache hits and shared identities both break `completed == 1`."""
        self.assertIn(
            "greater than the cache-only run's", self.body
        )
        self.assertIn("Do not require `completed_judgments` to equal 1", self.body)
        self.assertIn("more than one task at once", self.body)
        self.assertNotIn("completed_judgments: 1", self.body)

    def test_documents_the_receipt_fields_it_asks_operators_to_read(self) -> None:
        for field in ("missing_judgments", "failed_judgments", "reused_from_cache",
                      "fully_judged", "hosted_calls"):
            with self.subTest(field=field):
                self.assertIn(field, self.body)

    def test_nugget_and_cache_limitations_stay_documented(self) -> None:
        self.assertIn("Nugget coverage stays unavailable", self.body)
        self.assertIn("no `PYTHONPATH` or other environment override", self.body)


if __name__ == "__main__":
    unittest.main()
