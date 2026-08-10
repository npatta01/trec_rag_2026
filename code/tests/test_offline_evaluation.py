"""Behavioural tests for the generic offline evaluation bundle, renderer, and CLI.

Every fixture is synthetic and privacy-free, with arbitrary topic identifiers, so nothing
here depends on the names, counts, or shape of any real competition run.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import time
import unittest
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

from offline_evaluation_fixture import (
    RunFixture,
    TopicSpec,
    build_accepted_run,
    build_run,
    submission_records,
    write_generation_identity,
    write_submission,
)

from trec_rag import competition_evaluation_report as cli
import trec_rag.offline_evaluation as offline_module
from trec_rag.friendly_report import (
    ReportPrivacyError,
    build_presentation,
    denylist_from_bundle,
    render_report,
    write_report,
)
from trec_rag.judge_cache import JudgeCache
from trec_rag.offline_evaluation import (
    EvaluationError,
    JudgeOutcome,
    JudgeSettings,
    _judge_identity,
    _resolve_judgments,
    build_evaluation_bundle,
    ragdoll_identity,
)

REPOSITORY_ROOT = Path(__file__).parents[2]

ONE_TOPIC = (TopicSpec("solitary-topic", "A single arbitrary narrative.", candidate_documents=13),)
THREE_TOPICS = (
    TopicSpec("alpha-topic", "Alpha narrative about power grids.", candidate_documents=11),
    TopicSpec("beta-topic", "Beta narrative about coastal erosion.", candidate_documents=7),
    TopicSpec(
        "gamma-topic",
        "Gamma narrative about vaccine logistics.",
        candidate_documents=9,
        answers=(
            ("Gamma first claim.", (0,)),
            ("Gamma second claim.", (0,)),
            ("Gamma third claim.", (0,)),
        ),
    ),
)


def settings(**overrides: Any) -> JudgeSettings:
    base = {
        "provider": "fixture-provider",
        "model": "fixture/model-1",
        "thinking": "medium",
        "temperature": None,
        "system_prompt": "fixture system prompt",
        "agent_binary": "fixture-agent",
        "extension_identity": "none",
    }
    base.update(overrides)
    return JudgeSettings(**base)  # type: ignore[arg-type]


class RecordingJudge:
    """A mock hosted judge. Nothing in these tests may reach a real provider."""

    def __init__(self, labels: Sequence[str] | str = "FS") -> None:
        self.labels = labels
        self.calls: list[str] = []

    def __call__(self, task: dict[str, Any]) -> JudgeOutcome:
        self.calls.append(str(task["task_id"]))
        label = self.labels if isinstance(self.labels, str) else self.labels[len(self.calls) - 1]
        if label is None:
            return JudgeOutcome(status="failed", error="fixture failure")
        return JudgeOutcome(status="completed", support_label=label)


class ConcurrencyRecordingJudge:
    """A deterministic fixture judge that records concurrent hosted calls."""

    def __init__(self, label: str = "FS", delay: float = 0.01) -> None:
        self.label = label
        self.delay = delay
        self.calls: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, task: dict[str, Any]) -> JudgeOutcome:
        task_id = str(task["task_id"])
        with self._lock:
            self.calls.append(task_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return JudgeOutcome(status="completed", support_label=self.label)
        finally:
            with self._lock:
                self.active -= 1


class ControllerRecordingCache(JudgeCache):
    """Cache fixture that proves writes stay on the resolver/controller thread."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.put_thread_names: list[str] = []

    def put(self, identity: Any, *, support_label: str) -> str:
        self.put_thread_names.append(threading.current_thread().name)
        return super().put(identity, support_label=support_label)


class CheckpointThenInterruptJudge:
    """Complete the first task, then emulate an interrupted hosted job."""

    def __init__(self, first_task_id: str) -> None:
        self.first_task_id = first_task_id
        self.calls: list[str] = []

    def __call__(self, task: dict[str, Any]) -> JudgeOutcome:
        task_id = str(task["task_id"])
        self.calls.append(task_id)
        if task_id == self.first_task_id:
            return JudgeOutcome(status="completed", support_label="FS")
        time.sleep(0.05)
        raise KeyboardInterrupt("simulated operator interruption")


class Headings(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.levels: list[int] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self.levels.append(int(tag[1]))


class EvaluationCase(unittest.TestCase):
    def workspace(self) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="offline-eval-test-"))
        self.addCleanup(shutil.rmtree, directory, True)
        return directory

    def run_fixture(self, specs: Sequence[TopicSpec] = THREE_TOPICS, **kwargs: Any) -> RunFixture:
        return build_run(self.workspace(), list(specs), **kwargs)

    def accepted_fixture(self, specs: Sequence[TopicSpec] = THREE_TOPICS) -> RunFixture:
        return build_accepted_run(self.workspace(), list(specs))

    def build(
        self,
        fixture: RunFixture,
        *,
        judge: Any = None,
        cache_root: Path | None = None,
        work_dir: Path | None = None,
        judge_settings: JudgeSettings | None = None,
        **kwargs: Any,
    ):
        return build_evaluation_bundle(
            retrieval_config_path=fixture.retrieval_config,
            rag_config_path=fixture.rag_config,
            work_dir=work_dir or (fixture.root / "work"),
            repository_root=REPOSITORY_ROOT,
            cache_root=cache_root or (fixture.root / "cache"),
            judge=judge,
            judge_settings=judge_settings or settings(),
            created_utc="2026-01-01T00:00:00+00:00",
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Scope, ordering, and CLI contract
# ---------------------------------------------------------------------------


class ScopeTests(EvaluationCase):
    def test_single_arbitrary_topic(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        self.assertEqual(bundle.manifest["scope"]["topic_ids"], ["solitary-topic"])
        self.assertEqual(bundle.manifest["judgments"]["label_counts"]["FS"], 2)

    def test_three_arbitrary_topics_in_declared_order(self) -> None:
        fixture = self.run_fixture()
        bundle = self.build(fixture, judge=RecordingJudge("PS"))
        self.assertEqual(
            bundle.manifest["scope"]["topic_ids"], ["alpha-topic", "beta-topic", "gamma-topic"]
        )
        self.assertEqual(bundle.manifest["judgments"]["task_count"], 7)

    def test_no_topic_flags_select_every_topic(self) -> None:
        fixture = self.run_fixture()
        bundle = self.build(fixture, judge=RecordingJudge("FS"), topic_ids=None)
        self.assertEqual(len(bundle.manifest["scope"]["topic_ids"]), 3)

    def test_repeated_topic_flags_preserve_declared_order(self) -> None:
        fixture = self.run_fixture()
        bundle = self.build(
            fixture, judge=RecordingJudge("FS"), topic_ids=["gamma-topic", "alpha-topic"]
        )
        self.assertEqual(bundle.manifest["scope"]["topic_ids"], ["alpha-topic", "gamma-topic"])

    def test_shuffled_source_rows_keep_declared_order(self) -> None:
        fixture = build_run(self.workspace(), list(THREE_TOPICS), shuffle_rows=True)
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        self.assertEqual(
            bundle.manifest["scope"]["topic_ids"], ["alpha-topic", "beta-topic", "gamma-topic"]
        )

    def test_unknown_topic_is_refused(self) -> None:
        fixture = self.run_fixture()
        with self.assertRaises(ValueError):
            self.build(fixture, judge=RecordingJudge("FS"), topic_ids=["no-such-topic"])


class CliContractTests(EvaluationCase):
    def test_both_configs_are_required(self) -> None:
        for argv in (
            ["--retrieval-config", "a.yaml"],
            ["--rag-config", "b.yaml"],
            [],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                cli.main(argv)

    def test_cli_requires_config_mode_or_complete_accepted_mode(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(["--retrieval-config", "r.yaml", "--accepted-rag", "run.jsonl"])

    def test_cli_rejects_source_identity_in_config_mode(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(
                [
                    "--retrieval-config", "r.yaml",
                    "--rag-config", "g.yaml",
                    "--source-identity", "identity.json",
                ]
            )

    def test_accepted_mode_records_post_run_binding(self) -> None:
        fixture = self.accepted_fixture(ONE_TOPIC)
        bundle = build_evaluation_bundle(
            retrieval_config_path=fixture.retrieval_config,
            accepted_submission_path=fixture.rag_output,
            accepted_bundle_metadata_path=fixture.accepted_bundle_metadata,
            handoff_manifest_path=fixture.accepted_handoff,
            work_dir=fixture.root / "accepted-work",
            repository_root=REPOSITORY_ROOT,
            cache_root=fixture.root / "accepted-cache",
            judge=None,
            judge_settings=settings(),
            created_utc="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(
            bundle.manifest["identities"]["binding_kind"],
            "accepted_rag_evaluation_binding_v1",
        )
        self.assertFalse(bundle.manifest["identities"]["source_identity_available"])
        self.assertEqual(
            bundle.manifest["sources"]["rag/rag_output_trec_rag_2026.jsonl"],
            bundle.manifest["identities"]["submission_sha256"],
        )

    def test_accepted_mode_does_not_claim_missing_generation_identity(self) -> None:
        fixture = self.accepted_fixture(ONE_TOPIC)
        bundle = build_evaluation_bundle(
            retrieval_config_path=fixture.retrieval_config,
            accepted_submission_path=fixture.rag_output,
            accepted_bundle_metadata_path=fixture.bundle_metadata,
            handoff_manifest_path=fixture.handoff,
            work_dir=fixture.root / "accepted-work",
            repository_root=REPOSITORY_ROOT,
            cache_root=fixture.root / "accepted-cache",
            judge=None,
            judge_settings=settings(),
            created_utc="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(bundle.manifest["validation"]["handoff_bound_to_generation"])
        self.assertTrue(bundle.manifest["validation"]["accepted_submission_bound_to_handoff"])
        page = render_report(build_presentation(bundle.manifest))
        self.assertIn("Handoff bound to generation: no", page)
        self.assertIn("Original generation identity unavailable", page)
        self.assertIn("original generation identity was not preserved", page)

    def test_accepted_mode_rejects_between_phase_submission_replacement(self) -> None:
        fixture = self.accepted_fixture(ONE_TOPIC)
        real_loader = offline_module.load_debug_report_data
        binding = offline_module.build_accepted_run_binding(
            fixture.rag_output,
            fixture.bundle_metadata,
            fixture.handoff,
        )
        source = offline_module.RagArtifactSource(
            handoff_manifest_path=fixture.handoff.resolve(),
            output_path=fixture.rag_output.resolve(),
            topic_ids=binding.topic_ids,
            team_id=binding.team_id,
            run_id=binding.run_id,
            run_desc=binding.run_desc,
            provider=binding.provider,
            model=", ".join(binding.models),
            accepted_submission_sha256=binding.submission_sha256,
        )
        accepted_bytes = fixture.rag_output.read_bytes()
        replacement_bytes = accepted_bytes.replace(
            b"First supported answer.", b"Changed supported answer."
        )

        def load_then_replace(*args: Any, **kwargs: Any) -> Any:
            fixture.rag_output.write_bytes(replacement_bytes)
            try:
                return real_loader(*args, **kwargs)
            finally:
                fixture.rag_output.write_bytes(accepted_bytes)

        with patch.object(offline_module, "load_debug_report_data", side_effect=load_then_replace):
            with self.assertRaisesRegex(ValueError, "sha256|bound bytes"):
                load_then_replace(
                    fixture.retrieval_config,
                    rag_artifact_source=source,
                )

    def test_accepted_mode_rejects_changed_submission_bytes(self) -> None:
        fixture = self.accepted_fixture(ONE_TOPIC)
        fixture.rag_output.write_bytes(fixture.rag_output.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            build_evaluation_bundle(
                retrieval_config_path=fixture.retrieval_config,
                accepted_submission_path=fixture.rag_output,
                accepted_bundle_metadata_path=fixture.accepted_bundle_metadata,
                handoff_manifest_path=fixture.accepted_handoff,
                work_dir=fixture.root / "accepted-work",
                repository_root=REPOSITORY_ROOT,
                cache_root=fixture.root / "accepted-cache",
                judge=None,
                judge_settings=settings(),
            )

    def test_accepted_mode_rejects_wrong_run_description(self) -> None:
        fixture = self.accepted_fixture(ONE_TOPIC)
        metadata = json.loads(fixture.accepted_bundle_metadata.read_text(encoding="utf-8"))
        metadata["runs"][0]["run_desc"] = "Wrong accepted run description"
        fixture.accepted_bundle_metadata.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ValueError):
            build_evaluation_bundle(
                retrieval_config_path=fixture.retrieval_config,
                accepted_submission_path=fixture.rag_output,
                accepted_bundle_metadata_path=fixture.accepted_bundle_metadata,
                handoff_manifest_path=fixture.accepted_handoff,
                work_dir=fixture.root / "accepted-work",
                repository_root=REPOSITORY_ROOT,
                cache_root=fixture.root / "accepted-cache",
                judge=None,
                judge_settings=settings(),
            )

    def test_accepted_mode_rejects_handoff_incompatible_with_retrieval_export(self) -> None:
        fixture = self.accepted_fixture(ONE_TOPIC)
        other = self.accepted_fixture(ONE_TOPIC)
        with self.assertRaises(ValueError):
            build_evaluation_bundle(
                retrieval_config_path=fixture.retrieval_config,
                accepted_submission_path=fixture.rag_output,
                accepted_bundle_metadata_path=fixture.accepted_bundle_metadata,
                handoff_manifest_path=other.accepted_handoff,
                work_dir=fixture.root / "accepted-work",
                repository_root=REPOSITORY_ROOT,
                cache_root=fixture.root / "accepted-cache",
                judge=None,
                judge_settings=settings(),
            )

    def test_cli_accepted_mode_writes_portable_command_shape(self) -> None:
        import contextlib
        import io

        fixture = self.accepted_fixture(ONE_TOPIC)
        output = io.StringIO()
        report = fixture.root / "accepted-report.html"
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                cli.main(
                    [
                        "--retrieval-config", str(fixture.retrieval_config),
                        "--accepted-rag", str(fixture.rag_output),
                        "--accepted-bundle-metadata", str(fixture.accepted_bundle_metadata),
                        "--handoff-manifest", str(fixture.accepted_handoff),
                        "--work-dir", str(fixture.root / "accepted-cli-work"),
                        "--cache-dir", str(fixture.root / "accepted-cli-cache"),
                        "--output", str(report),
                    ]
                ),
                0,
            )
        receipt = json.loads(output.getvalue().strip().splitlines()[-1])
        page = report.read_text(encoding="utf-8")
        self.assertIn("--accepted-rag", page)
        self.assertNotIn(str(fixture.root), page)
        self.assertEqual(
            receipt["exact_invocation"].split()[3],
            "--retrieval-config",
        )

    def test_judge_switch_defaults_to_no_hosted_calls(self) -> None:
        parser_defaults = cli.main.__doc__ or ""
        self.assertIsInstance(parser_defaults, str)
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=None)
        self.assertEqual(bundle.manifest["judge"]["hosted_calls"], 0)
        self.assertFalse(bundle.manifest["judgments"]["fully_judged"])

    def test_hosted_judge_never_enables_the_ragdoll_ttl_cache(self) -> None:
        config = cli.local_agent_config(settings())
        self.assertIsNone(config.cache_dir)

    def test_hosted_judge_uses_the_pinned_ragdoll_settings(self) -> None:
        from ragdoll.config import DEFAULT_MODEL, DEFAULT_PROVIDER

        pinned = cli.judge_settings(REPOSITORY_ROOT)
        self.assertEqual(pinned.provider, DEFAULT_PROVIDER)
        self.assertEqual(pinned.model, DEFAULT_MODEL)
        self.assertEqual(pinned.agent_binary, cli.AGENT_BINARY)
        self.assertEqual(pinned.extension_identity, cli.EXTENSION_IDENTITY)

    def test_hosted_judge_only_receives_validated_misses(self) -> None:
        """The judge callable is invoked once per miss and never for a cache hit."""
        fixture = self.run_fixture(ONE_TOPIC)
        cache_root = fixture.root / "shared-cache"
        first = RecordingJudge("PS")
        self.build(fixture, judge=first, cache_root=cache_root)
        self.assertEqual(len(first.calls), 2)
        second = RecordingJudge("PS")
        self.build(fixture, judge=second, cache_root=cache_root, work_dir=fixture.root / "w2")
        self.assertEqual(second.calls, [])

    def test_failed_judge_results_are_not_cached(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        cache_root = fixture.root / "cache"
        bundle = self.build(fixture, judge=RecordingJudge([None, None]), cache_root=cache_root)
        self.assertEqual(bundle.manifest["judge"]["failed"], 2)
        self.assertEqual(bundle.manifest["judgments"]["completed"], 0)
        self.assertEqual(bundle.manifest["cache"]["writes"], 0)

    def test_judge_exception_becomes_a_failed_judgment_receipt(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)

        def broken_judge(_task: dict[str, Any]) -> JudgeOutcome:
            raise RuntimeError("private provider detail")

        bundle = self.build(fixture, judge=broken_judge)
        self.assertEqual(bundle.manifest["judge"]["hosted_calls"], 2)
        self.assertEqual(bundle.manifest["judge"]["failed"], 2)
        self.assertEqual(bundle.manifest["judge"]["missing"], 2)
        self.assertEqual(bundle.manifest["judge"]["conflicts"], 0)
        self.assertFalse(bundle.manifest["judgments"]["fully_judged"])
        self.assertNotIn("private provider detail", json.dumps(bundle.manifest))

    def test_concurrent_cache_conflict_keeps_first_label_and_reports_conflict(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        baseline = self.build(fixture, judge=None, work_dir=fixture.root / "baseline")
        task = _rows(baseline.work_dir / "support_tasks.jsonl")[0]
        cache = JudgeCache(fixture.root / "racing-cache")
        judge_settings = settings()
        ragdoll = ragdoll_identity(REPOSITORY_ROOT)

        def racing_judge(current: dict[str, Any]) -> JudgeOutcome:
            cache.put(
                _judge_identity(current, settings=judge_settings, ragdoll=ragdoll),
                support_label="PS",
            )
            return JudgeOutcome(status="completed", support_label="FS")

        judgments, report = _resolve_judgments(
            [task],
            cache=cache,
            judge=racing_judge,
            settings=judge_settings,
            ragdoll=ragdoll,
        )
        self.assertEqual(judgments[0]["support_label"], "PS")
        self.assertEqual(judgments[0]["label_source"], "cache_conflict")
        self.assertEqual(report["conflicts"], 1)
        self.assertEqual(report["completed"], 1)
        self.assertEqual(report["failed"], 0)


# ---------------------------------------------------------------------------
# Counts, metrics, and availability
# ---------------------------------------------------------------------------


class CountTests(EvaluationCase):
    def test_candidate_documents_use_the_sealed_pool_not_facet_new_documents(self) -> None:
        fixture = self.run_fixture()
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        counts = {topic["topic_id"]: topic["counts"] for topic in bundle.manifest["topics"]}
        self.assertEqual(counts["alpha-topic"]["candidate_documents"], 11)
        self.assertEqual(counts["beta-topic"]["candidate_documents"], 7)
        self.assertEqual(counts["gamma-topic"]["candidate_documents"], 9)
        for topic in bundle.manifest["topics"]:
            self.assertEqual(topic["candidate_pool_kind"], "natural_union")
            self.assertNotEqual(
                topic["counts"]["candidate_documents"], topic["counts"]["facet_new_documents"]
            )

    def test_candidate_pool_kind_is_recorded_verbatim(self) -> None:
        fixture = build_run(self.workspace(), list(ONE_TOPIC), depth_kind="candidate_pool")
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        self.assertEqual(bundle.manifest["topics"][0]["candidate_pool_kind"], "candidate_pool")

    def test_renderer_distinguishes_natural_union_from_candidate_pool(self) -> None:
        natural_root = self.workspace() / "natural"
        pool_root = self.workspace() / "pool"
        natural_root.mkdir()
        pool_root.mkdir()
        natural_fixture = build_run(natural_root, list(ONE_TOPIC), depth_kind="natural_union")
        pool_fixture = build_run(pool_root, list(ONE_TOPIC), depth_kind="candidate_pool")
        natural = render_report(
            build_presentation(self.build(natural_fixture, judge=RecordingJudge("FS")).manifest)
        )
        pool = render_report(
            build_presentation(self.build(pool_fixture, judge=RecordingJudge("FS")).manifest)
        )

        self.assertIn("Deduplicated union across query lanes", natural)
        self.assertNotIn("sealed candidate pool", natural)
        self.assertIn("sealed candidate pool", pool)
        self.assertNotIn("Deduplicated union across query lanes", pool)


class SupportMetricTests(EvaluationCase):
    def test_macro_is_averaged_before_rounding(self) -> None:
        """0.75 and 0.840909 must publish 0.795455, not tie-to-even 0.795454."""
        from trec_rag.offline_evaluation import _macro_from_raw

        macro = _macro_from_raw(
            {"a": {"weighted_precision_first_citation": 0.75},
             "b": {"weighted_precision_first_citation": 0.8409090909090909}},
            ["a", "b"],
        )
        self.assertEqual(macro["weighted_precision_first_citation"], 0.795455)

    def test_macro_uses_raw_cells_not_rounded_ones(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=RecordingJudge("PS"))
        support = bundle.manifest["metrics"]["citation_support"]
        self.assertTrue(support["macro_availability"]["available"])
        self.assertEqual(support["macro"]["weighted_precision_first_citation"], 0.5)

    def test_partial_judgments_make_the_topic_and_macro_unavailable(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=RecordingJudge(["FS", None]))
        support = bundle.manifest["metrics"]["citation_support"]
        self.assertFalse(support["macro_availability"]["available"])
        self.assertIn("1 of 2", support["per_topic_availability"]["solitary-topic"]["reason"])
        self.assertEqual(support["macro"], {})

    def test_completed_rag_without_any_judgments_reports_unavailable_not_zero(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=None)
        support = bundle.manifest["metrics"]["citation_support"]
        self.assertEqual(support["macro"], {})
        self.assertIn("no topic has complete", support["macro_availability"]["reason"])


class QrelsTests(EvaluationCase):
    def test_complete_qrels_score_every_topic_and_publish_a_macro(self) -> None:
        fixture = self.run_fixture()
        bundle = self.build(fixture, judge=RecordingJudge("FS"), qrels_path=fixture.qrels())
        retrieval = bundle.manifest["metrics"]["retrieval"]
        self.assertTrue(retrieval["macro_availability"]["available"])
        self.assertEqual(sorted(retrieval["per_topic"]), ["alpha-topic", "beta-topic", "gamma-topic"])
        self.assertEqual(retrieval["macro"]["ndcg@10"], 1.0)
        self.assertEqual(retrieval["per_topic"]["alpha-topic"]["recall@100"], 1.0)

    def test_partial_qrels_expose_per_topic_availability_and_no_macro(self) -> None:
        fixture = self.run_fixture()
        bundle = self.build(
            fixture, judge=RecordingJudge("FS"), qrels_path=fixture.qrels(["beta-topic"])
        )
        retrieval = bundle.manifest["metrics"]["retrieval"]
        self.assertFalse(retrieval["macro_availability"]["available"])
        self.assertEqual(retrieval["macro"], {})
        self.assertTrue(retrieval["per_topic_availability"]["beta-topic"]["available"])
        self.assertFalse(retrieval["per_topic_availability"]["alpha-topic"]["available"])
        self.assertIn("no judgments", retrieval["per_topic_availability"]["alpha-topic"]["reason"])

    def test_mismatched_qrels_score_nothing(self) -> None:
        fixture = self.run_fixture()
        qrels = fixture.root / "foreign.txt"
        qrels.write_text("some-other-topic 0 doc-original 1\n", encoding="utf-8")
        bundle = self.build(fixture, judge=RecordingJudge("FS"), qrels_path=qrels)
        retrieval = bundle.manifest["metrics"]["retrieval"]
        self.assertEqual(retrieval["per_topic"], {})
        self.assertIn("match none", retrieval["macro_availability"]["reason"])

    def test_malformed_qrels_fail_closed(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        qrels = fixture.root / "broken.txt"
        qrels.write_text("only two columns\n", encoding="utf-8")
        with self.assertRaises(EvaluationError):
            self.build(fixture, judge=RecordingJudge("FS"), qrels_path=qrels)

    def test_shuffled_qrels_lines_do_not_change_scores(self) -> None:
        fixture = self.run_fixture()
        ordered = fixture.qrels()
        lines = ordered.read_text(encoding="utf-8").splitlines()
        shuffled = fixture.root / "shuffled.txt"
        shuffled.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
        first = self.build(fixture, judge=RecordingJudge("FS"), qrels_path=ordered)
        second = self.build(
            fixture,
            judge=RecordingJudge("FS"),
            qrels_path=shuffled,
            work_dir=fixture.root / "work2",
        )
        self.assertEqual(
            first.manifest["metrics"]["retrieval"], second.manifest["metrics"]["retrieval"]
        )

    def test_absent_qrels_are_reported_exactly(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        retrieval = bundle.manifest["metrics"]["retrieval"]
        self.assertEqual(retrieval["macro_availability"]["reason"], "no qrels file was supplied")


class GoldNuggetTests(EvaluationCase):
    def test_matching_gold_still_reports_unavailable_with_the_assignment_reason(self) -> None:
        fixture = self.run_fixture()
        bundle = self.build(
            fixture, judge=RecordingJudge("FS"), gold_nuggets_path=fixture.gold_nuggets()
        )
        nuggets = bundle.manifest["metrics"]["nugget_coverage"]
        self.assertFalse(nuggets["macro_availability"]["available"])
        self.assertEqual(nuggets["matched_topic_ids"], ["alpha-topic", "beta-topic", "gamma-topic"])
        self.assertIn("no completed nugget-assignment run", nuggets["macro_availability"]["reason"])
        self.assertFalse(nuggets["generated_claims_used_as_gold"])

    def test_partial_gold_reports_the_matched_subset(self) -> None:
        fixture = self.run_fixture()
        bundle = self.build(
            fixture,
            judge=RecordingJudge("FS"),
            gold_nuggets_path=fixture.gold_nuggets(["beta-topic"]),
        )
        nuggets = bundle.manifest["metrics"]["nugget_coverage"]
        self.assertEqual(nuggets["matched_topic_ids"], ["beta-topic"])
        self.assertIn("match 1 of 3", nuggets["macro_availability"]["reason"])

    def test_gold_is_bound_to_the_authoritative_narrative_not_the_topic_id(self) -> None:
        """A gold file's own query text is replaced by this run's real narrative."""
        from hashlib import sha256

        fixture = self.run_fixture(ONE_TOPIC)
        gold = fixture.root / "wrong-query.jsonl"
        gold.write_text(
            json.dumps(
                {
                    "qid": "solitary-topic",
                    "query": "a narrative that is not this topic's",
                    "nuggets": [{"text": "n", "importance": "vital"}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        bundle = self.build(fixture, judge=RecordingJudge("FS"), gold_nuggets_path=gold)
        binding = bundle.manifest["metrics"]["nugget_coverage"]["narrative_binding_sha256s"]
        narrative = ONE_TOPIC[0].narrative
        self.assertEqual(
            binding["solitary-topic"], sha256(narrative.encode("utf-8")).hexdigest()
        )
        self.assertNotEqual(
            binding["solitary-topic"], sha256(b"solitary-topic").hexdigest()
        )

    def test_gold_naming_an_out_of_scope_topic_is_reported_exactly(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        gold = fixture.root / "foreign.jsonl"
        gold.write_text(
            json.dumps(
                {"qid": "not-in-scope", "nuggets": [{"text": "n", "importance": "vital"}]}
            )
            + "\n",
            encoding="utf-8",
        )
        bundle = self.build(fixture, judge=RecordingJudge("FS"), gold_nuggets_path=gold)
        nuggets = bundle.manifest["metrics"]["nugget_coverage"]
        self.assertEqual(nuggets["matched_topic_ids"], [])
        self.assertFalse(nuggets["macro_availability"]["available"])
        reason = nuggets["macro_availability"]["reason"]
        self.assertIn("could not be bound", reason)
        self.assertIn("not-in-scope", reason)

    def test_absent_gold_is_reported_exactly(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        self.assertEqual(
            bundle.manifest["metrics"]["nugget_coverage"]["macro_availability"]["reason"],
            "no released gold-nugget file was supplied",
        )


# ---------------------------------------------------------------------------
# Fail-closed inputs
# ---------------------------------------------------------------------------


class DriftTests(EvaluationCase):
    def test_duplicate_submission_topic_is_refused(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        records = submission_records(fixture.specs, fixture.run_id)
        write_submission(fixture.rag_output, fixture.specs, fixture.run_id, records=records * 2)
        with self.assertRaises(ValueError):
            self.build(fixture, judge=RecordingJudge("FS"))

    def test_altered_rag_output_breaks_the_handoff_binding(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        records = submission_records(fixture.specs, fixture.run_id)
        records[0]["answer"][0]["text"] = "A statement the sealed run never produced."
        write_submission(fixture.rag_output, fixture.specs, fixture.run_id, records=records)
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        self.assertIn(
            "A statement the sealed run never produced.",
            [item["text"] for item in bundle.manifest["topics"][0]["answer"]],
        )

    def test_unresolvable_citation_is_refused(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        records = submission_records(fixture.specs, fixture.run_id)
        records[0]["answer"][0]["citations"] = [7]
        write_submission(fixture.rag_output, fixture.specs, fixture.run_id, records=records)
        with self.assertRaises(ValueError):
            self.build(fixture, judge=RecordingJudge("FS"))

    def test_citation_naming_a_foreign_document_is_refused(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        records = submission_records(fixture.specs, fixture.run_id)
        records[0]["references"] = ["doc-not-in-handoff"]
        write_submission(fixture.rag_output, fixture.specs, fixture.run_id, records=records)
        with self.assertRaises(ValueError):
            self.build(fixture, judge=RecordingJudge("FS"))

    def test_generation_identity_drift_is_refused(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        identity_path = write_generation_identity(fixture)
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        identity["handoff_manifest_sha256"] = "0" * 64
        identity_path.write_text(json.dumps(identity), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.build(fixture, judge=RecordingJudge("FS"))

    def test_missing_generation_identity_is_refused(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        from trec_rag.competition_rag import load_rag_generation_config

        (load_rag_generation_config(fixture.rag_config).work_dir / "generation_identity.json").unlink()
        with self.assertRaises(EvaluationError):
            self.build(fixture, judge=RecordingJudge("FS"))

    def test_altered_sealed_source_is_refused(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        run_path = fixture.retrieval_output / "r_output_trec_rag_2026.tsv"
        run_path.write_bytes(run_path.read_bytes().replace(b"doc-original", b"doc-tampered"))
        with self.assertRaises(ValueError):
            self.build(fixture, judge=RecordingJudge("FS"))

    def test_duplicate_judgment_rows_are_refused(self) -> None:
        from trec_rag.offline_evaluation import _labels_by_citation

        row = {
            "task_id": "t",
            "support_label": "FS",
            "metadata": {"topic_id": "a", "sentence_index": 0, "citation_index": 0},
        }
        with self.assertRaises(EvaluationError):
            _labels_by_citation([row, dict(row)])


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


class CacheReuseTests(EvaluationCase):
    def independent_runs(self) -> tuple[RunFixture, RunFixture, Path]:
        """Two runs with different config paths, experiment/run/topic/task ids."""
        first = build_run(
            self.workspace(),
            [TopicSpec("first-topic", "Shared narrative text.", candidate_documents=4)],
            experiment_id="experiment-one",
        )
        second = build_run(
            self.workspace(),
            [TopicSpec("second-topic", "Shared narrative text.", candidate_documents=4)],
            experiment_id="experiment-two",
        )
        return first, second, self.workspace() / "shared-cache"

    def test_cross_run_reuse_with_no_shared_identifiers(self) -> None:
        first, second, cache_root = self.independent_runs()
        self.assertNotEqual(first.retrieval_config, second.retrieval_config)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertNotEqual(first.topic_ids, second.topic_ids)

        judge = RecordingJudge("PS")
        one = self.build(first, judge=judge, cache_root=cache_root)
        self.assertEqual(len(judge.calls), 2)

        second_judge = RecordingJudge("PS")
        two = self.build(second, judge=second_judge, cache_root=cache_root)
        self.assertEqual(second_judge.calls, [])
        self.assertEqual(two.manifest["judge"]["hosted_calls"], 0)
        self.assertEqual(two.manifest["judge"]["reused_from_cache"], 2)

        # Task ids differ, yet each run is bound to its own.
        first_tasks = {row["task_id"] for row in _rows(one.work_dir / "support_judgments.jsonl")}
        second_tasks = {row["task_id"] for row in _rows(two.work_dir / "support_judgments.jsonl")}
        self.assertFalse(first_tasks & second_tasks)
        for row in _rows(two.work_dir / "support_judgments.jsonl"):
            self.assertEqual(row["metadata"]["topic_id"], "second-topic")
            self.assertEqual(row["label_source"], "cache")
            self.assertEqual(len(row["cache_entry_sha256"]), 64)

    def test_changed_statement_invalidates_only_the_affected_entry(self) -> None:
        first, second, cache_root = self.independent_runs()
        self.build(first, judge=RecordingJudge("PS"), cache_root=cache_root)

        records = submission_records(second.specs, second.run_id)
        records[0]["answer"][0]["text"] = "A different first claim entirely."
        write_submission(second.rag_output, second.specs, second.run_id, records=records)

        judge = RecordingJudge("NS")
        bundle = self.build(second, judge=judge, cache_root=cache_root)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(bundle.manifest["judge"]["reused_from_cache"], 1)

    def test_changed_model_invalidates_every_entry(self) -> None:
        first, second, cache_root = self.independent_runs()
        self.build(first, judge=RecordingJudge("PS"), cache_root=cache_root)
        judge = RecordingJudge("FS")
        self.build(
            second, judge=judge, cache_root=cache_root, judge_settings=settings(model="other/model")
        )
        self.assertEqual(len(judge.calls), 2)

    def test_partial_hit_calls_only_the_misses(self) -> None:
        fixture = self.run_fixture(
            (TopicSpec("partial-topic", "Partial narrative.", answers=(
                ("Claim one.", (0,)),
                ("Claim two.", (0,)),
                ("Claim three.", (0,)),
            )),)
        )
        cache_root = fixture.root / "cache"
        self.build(fixture, judge=RecordingJudge(["FS", "FS", "FS"]), cache_root=cache_root)

        records = submission_records(fixture.specs, fixture.run_id)
        records[0]["answer"][2]["text"] = "Claim three, revised."
        write_submission(fixture.rag_output, fixture.specs, fixture.run_id, records=records)

        judge = RecordingJudge("PS")
        bundle = self.build(
            fixture, judge=judge, cache_root=cache_root, work_dir=fixture.root / "work2"
        )
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(bundle.manifest["cache"]["hits"], 2)
        self.assertEqual(bundle.manifest["judge"]["hosted_calls"], 1)

    def test_full_hit_reports_zero_hosted_calls(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        cache_root = fixture.root / "cache"
        self.build(fixture, judge=RecordingJudge("FS"), cache_root=cache_root)
        bundle = self.build(
            fixture, judge=RecordingJudge("FS"), cache_root=cache_root, work_dir=fixture.root / "w2"
        )
        self.assertEqual(bundle.manifest["judge"]["hosted_calls"], 0)
        self.assertEqual(bundle.manifest["cache"]["misses"], 0)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class RenderTests(EvaluationCase):
    def rendered(self, fixture: RunFixture, **kwargs: Any) -> tuple[Any, str]:
        bundle = self.build(fixture, **kwargs)
        return bundle, render_report(build_presentation(bundle.manifest))

    def test_judgment_mutation_moves_every_dependent_value(self) -> None:
        """One flipped label must move the badge, the detailed row, the totals, the
        per-topic cell, and the macro together."""
        baseline_fixture = self.run_fixture(ONE_TOPIC)
        baseline, before = self.rendered(baseline_fixture, judge=RecordingJudge(["FS", "FS"]))
        mutated_fixture = self.run_fixture(ONE_TOPIC)
        mutated, after = self.rendered(mutated_fixture, judge=RecordingJudge(["FS", "NS"]))

        judged = r'class="citation citation-judged label-(\w\w)"'
        self.assertEqual(re.findall(judged, before), ["fs", "fs"])
        self.assertEqual(re.findall(judged, after), ["fs", "ns"])

        self.assertIn('<span class="judgment-badge label-ns">NS · None</span>', after)
        self.assertNotIn("judgment-badge label-ns", before)

        self.assertEqual(baseline.manifest["judgments"]["label_counts"], {"FS": 2, "NS": 0, "PS": 0})
        self.assertEqual(mutated.manifest["judgments"]["label_counts"], {"FS": 1, "NS": 1, "PS": 0})
        self.assertIn("2 full · 0 partial · 0 none.", before)
        self.assertIn("1 full · 0 partial · 1 none.", after)
        self.assertNotIn("response-has-ns\"", before)
        self.assertIn("response-has-ns", after)

        topic = "solitary-topic"
        before_cell = baseline.manifest["metrics"]["citation_support"]["per_topic"][topic]
        after_cell = mutated.manifest["metrics"]["citation_support"]["per_topic"][topic]
        self.assertEqual(before_cell["weighted_precision_first_citation"], 1.0)
        self.assertEqual(after_cell["weighted_precision_first_citation"], 0.5)

        before_macro = baseline.manifest["metrics"]["citation_support"]["macro"]
        after_macro = mutated.manifest["metrics"]["citation_support"]["macro"]
        self.assertEqual(before_macro["weighted_precision_all_judged_citations"], 1.0)
        self.assertEqual(after_macro["weighted_precision_all_judged_citations"], 0.5)
        self.assertEqual(after_macro["hard_precision"], 0.5)

    def test_detailed_judgments_quote_the_statement(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        _, page = self.rendered(fixture, judge=RecordingJudge(["FS", "NS"]))
        self.assertIn('<q class="judgment-statement">Second detail.</q>', page)
        self.assertRegex(
            page,
            r'judgment-claim">Answer 2 · reference \[1\]'
            r'<q class="judgment-statement">Second detail\.</q></span>'
            r'<span class="judgment-badge label-ns">',
        )

    def test_headings_never_skip_a_level(self) -> None:
        fixture = self.run_fixture()
        _, page = self.rendered(fixture, judge=RecordingJudge("FS"))
        collector = Headings()
        collector.feed(page)
        self.assertEqual(collector.levels[0], 1)
        previous = collector.levels[0]
        for level in collector.levels[1:]:
            self.assertLessEqual(level - previous, 1)
            previous = level

    def test_subnarratives_stay_collapsed_and_badges_carry_text(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        _, page = self.rendered(fixture, judge=RecordingJudge("PS"))
        self.assertIn('<details class="subnarrative-disclosure">', page)
        self.assertNotIn('subnarrative-disclosure" open', page)
        self.assertIn('<span class="visually-hidden"> — partial support</span>', page)

    def test_unjudged_citations_render_without_a_label(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        _, page = self.rendered(fixture, judge=None)
        self.assertIn("citation-unjudged", page)
        self.assertIn("Not judged", page)

    def test_report_has_no_external_runtime_dependencies(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        _, page = self.rendered(fixture, judge=RecordingJudge("FS"))
        self.assertNotIn("<script", page)
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)

    def test_render_is_independent_of_worker_count(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        one = self.build(
            fixture,
            judge=RecordingJudge("FS"),
            work_dir=fixture.root / "one-worker",
            cache_root=fixture.root / "one-worker-cache",
            judge_workers=1,
        )
        many = self.build(
            fixture,
            judge=ConcurrencyRecordingJudge("FS"),
            work_dir=fixture.root / "many-workers",
            cache_root=fixture.root / "many-workers-cache",
            judge_workers=2,
        )
        self.assertEqual(
            render_report(build_presentation(one.manifest)),
            render_report(build_presentation(many.manifest)),
        )


class PrivacyTests(EvaluationCase):
    def test_dynamic_docid_leak_is_rejected(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        denylist = denylist_from_bundle(bundle.work_dir)
        self.assertTrue(denylist)
        page = render_report(build_presentation(bundle.manifest))
        leaked = page + "<p>doc-original</p>"
        with self.assertRaises(ReportPrivacyError):
            from trec_rag.friendly_report import assert_publishable

            assert_publishable(leaked, ("doc-original",))

    def test_dynamic_evidence_text_leak_is_rejected(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        rows = _rows(bundle.work_dir / "support_input.jsonl")
        evidence = next(iter(rows[0]["segments"].values()))
        page = render_report(build_presentation(bundle.manifest))
        with self.assertRaises(ReportPrivacyError):
            from trec_rag.friendly_report import assert_publishable

            assert_publishable(page + evidence, denylist_from_bundle(bundle.work_dir))

    def test_html_escaped_private_text_is_rejected(self) -> None:
        from trec_rag.friendly_report import assert_publishable

        with self.assertRaises(ReportPrivacyError):
            assert_publishable("<p>O&#x27;Brien &amp; Co.</p>", ("O'Brien & Co.",))

    def test_long_decimal_number_is_not_treated_as_a_secret(self) -> None:
        from trec_rag.friendly_report import assert_publishable

        assert_publishable("<p>Identifier 123456789012 is public.</p>")

    def test_clean_report_passes_its_own_denylist(self) -> None:
        fixture = self.run_fixture()
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        output = fixture.root / "report.html"
        write_report(bundle.manifest, output, denylist=denylist_from_bundle(bundle.work_dir))
        self.assertTrue(output.is_file())
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_a_rejected_report_leaves_a_previous_one_intact(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        output = fixture.root / "report.html"
        write_report(bundle.manifest, output)
        original = output.read_bytes()
        with self.assertRaises(ReportPrivacyError):
            write_report(bundle.manifest, output, denylist=("Private post-run evaluation",))
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual([p for p in output.parent.glob("*.tmp-*")], [])


class DeterminismTests(EvaluationCase):
    def test_two_independent_bundles_render_identical_html(self) -> None:
        """Independent invocations, separate work dirs, one shared cache."""
        fixture = self.run_fixture()
        cache_root = fixture.root / "cache"
        self.build(fixture, judge=RecordingJudge("PS"), cache_root=cache_root)

        pages = []
        manifests = []
        for index in (1, 2):
            bundle = build_evaluation_bundle(
                retrieval_config_path=fixture.retrieval_config,
                rag_config_path=fixture.rag_config,
                work_dir=fixture.root / f"independent-{index}",
                repository_root=REPOSITORY_ROOT,
                cache_root=cache_root,
                judge=RecordingJudge("PS"),
                judge_settings=settings(),
                created_utc=f"2026-0{index}-0{index}T00:00:0{index}+00:00",
            )
            manifests.append(bundle.manifest)
            output = fixture.root / f"report-{index}.html"
            write_report(bundle.manifest, output)
            pages.append(output.read_bytes())

        self.assertNotEqual(manifests[0]["created_utc"], manifests[1]["created_utc"])
        self.assertEqual(pages[0], pages[1])

    def test_volatile_receipt_fields_are_declared(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        declared = set(bundle.manifest["volatile_receipt_fields"])
        self.assertIn("created_utc", declared)
        page = render_report(build_presentation(bundle.manifest))
        self.assertNotIn(bundle.manifest["created_utc"], page)


# ---------------------------------------------------------------------------
# Published metric definitions, per-topic rendering, scope receipt, judge limit
# ---------------------------------------------------------------------------


class MetricDefinitionTests(EvaluationCase):
    """The published definitions must match the pinned RAGDoll formula exactly."""

    def denominators(self, sentences: list[dict[str, Any]]) -> dict[str, float]:
        from ragdoll.support.metrics import support_metric

        metric = support_metric({"topic_id": "t", "run_id": "r", "sentences": sentences})
        return {
            "wpf": metric.weighted_precision_first_citation,
            "wrf": metric.weighted_recall_first_citation,
            "wpa": metric.weighted_precision_all_judged_citations,
            "wra": metric.weighted_recall_all_judged_citations,
            "hp": metric.hard_precision,
            "hr": metric.hard_recall,
        }

    def test_uncited_answer_objects_lower_recall_but_not_precision(self) -> None:
        """This is the behaviour the old wording denied; it pins the real formula."""
        sentences = [
            # support "2" is FS in ragdoll's WEIGHTED_SCORES {-1:0.0, 0:0.0, 1:0.5, 2:1.0}
            {"sentenceID": 0, "text": "cited", "citations": [{"citationID": 0, "support": "2"}]},
            {"sentenceID": 1, "text": "uncited", "citations": []},
        ]
        values = self.denominators(sentences)
        self.assertEqual(values["wpf"], 1.0)  # denominator 1: only the judged object
        self.assertEqual(values["wrf"], 0.5)  # denominator 2: the uncited object stays
        self.assertEqual(values["wpa"], 1.0)
        self.assertEqual(values["wra"], 0.5)
        self.assertEqual(values["hp"], 1.0)
        self.assertEqual(values["hr"], 0.5)

    def test_unjudged_citations_leave_the_recall_denominator(self) -> None:
        sentences = [
            {"sentenceID": 0, "text": "judged", "citations": [{"citationID": 0, "support": "2"}]},
            {"sentenceID": 1, "text": "unjudged", "citations": [{"citationID": 0, "support": "-1"}]},
        ]
        values = self.denominators(sentences)
        # The -1 object is removed from both denominators, so recall equals precision.
        self.assertEqual(values["wrf"], 1.0)
        self.assertEqual(values["wra"], 1.0)
        self.assertEqual(values["hr"], 1.0)

    def test_published_definitions_state_the_real_denominators(self) -> None:
        from trec_rag.offline_evaluation import SUPPORT_METRIC_DEFINITIONS as D

        for key in ("weighted_recall_first_citation", "weighted_recall_all_judged_citations",
                    "hard_recall"):
            with self.subTest(metric=key):
                text = D[key]
                self.assertIn("every answer object in the topic", text)
                self.assertNotIn("every answer object with a completed judgment", text)
        self.assertIn("no citations stay in this denominator",
                      D["weighted_recall_first_citation"])
        for key in ("weighted_precision_first_citation",
                    "weighted_precision_all_judged_citations", "hard_precision"):
            with self.subTest(metric=key):
                self.assertIn("judged", D[key])

    def test_definitions_are_published_in_the_manifest_and_page(self) -> None:
        from trec_rag.offline_evaluation import SUPPORT_METRIC_DEFINITIONS

        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        published = bundle.manifest["metric_definitions"]["citation_support"]
        self.assertEqual(published, dict(SUPPORT_METRIC_DEFINITIONS))
        page = render_report(build_presentation(bundle.manifest))
        self.assertIn("every answer object in the topic", page)


class PerTopicRetrievalRenderTests(EvaluationCase):
    def rendered(self, fixture: RunFixture, **kwargs: Any) -> str:
        bundle = self.build(fixture, judge=RecordingJudge("FS"), **kwargs)
        return render_report(build_presentation(bundle.manifest))

    def test_single_topic_values_are_rendered(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        page = self.rendered(fixture, qrels_path=fixture.qrels())
        self.assertIn("Per-topic retrieval metrics", page)
        self.assertIn('<th scope="row">solitary-topic</th>', page)
        self.assertIn("<th scope=\"col\">ndcg@10</th>", page)
        self.assertIn("<td>1.000000</td>", page)

    def test_three_topics_render_in_declared_order(self) -> None:
        fixture = self.run_fixture()
        page = self.rendered(fixture, qrels_path=fixture.qrels())
        order = [
            match
            for match in re.findall(r'<th scope="row">([\w-]+)</th>', page)
        ]
        self.assertEqual(order[:3], ["alpha-topic", "beta-topic", "gamma-topic"])

    def test_shuffled_inputs_keep_declared_row_order(self) -> None:
        fixture = build_run(self.workspace(), list(THREE_TOPICS), shuffle_rows=True)
        page = self.rendered(fixture, qrels_path=fixture.qrels())
        order = re.findall(r'<th scope="row">([\w-]+)</th>', page)
        self.assertEqual(order[:3], ["alpha-topic", "beta-topic", "gamma-topic"])

    def test_partial_qrels_keep_unavailable_topics_explicit(self) -> None:
        fixture = self.run_fixture()
        page = self.rendered(fixture, qrels_path=fixture.qrels(["beta-topic"]))
        self.assertIn('<th scope="row">beta-topic</th>', page)
        self.assertIn("cell-na", page)
        self.assertIn("no judgments for topic alpha-topic", page)

    def test_no_qrels_renders_no_per_topic_table(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        page = self.rendered(fixture)
        self.assertNotIn("Per-topic retrieval metrics", page)

    def test_table_is_headed_and_scrollable_for_small_screens(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        page = self.rendered(fixture, qrels_path=fixture.qrels())
        self.assertIn('<div class="table-scroll">', page)
        self.assertIn('<caption id="retrieval-per-topic-caption">', page)
        self.assertIn('aria-labelledby="retrieval-per-topic-caption"', page)


class ScopeReceiptTests(EvaluationCase):
    def test_omitted_selectors_are_recorded_as_all(self) -> None:
        fixture = self.run_fixture()
        bundle = self.build(fixture, judge=RecordingJudge("FS"), topic_ids=None)
        scope = bundle.manifest["scope"]
        self.assertEqual(scope["selection"], "all")
        self.assertIsNone(scope["requested_topic_ids"])
        self.assertEqual(len(scope["topic_ids"]), 3)

    def test_explicit_selectors_are_recorded_verbatim(self) -> None:
        fixture = self.run_fixture()
        requested = ["gamma-topic", "alpha-topic"]
        bundle = self.build(fixture, judge=RecordingJudge("FS"), topic_ids=requested)
        scope = bundle.manifest["scope"]
        self.assertEqual(scope["selection"], "explicit")
        self.assertEqual(scope["requested_topic_ids"], requested)
        self.assertEqual(scope["topic_ids"], ["alpha-topic", "gamma-topic"])

    def test_selecting_every_topic_explicitly_is_still_explicit(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(
            fixture, judge=RecordingJudge("FS"), topic_ids=["solitary-topic"]
        )
        self.assertEqual(bundle.manifest["scope"]["selection"], "explicit")


class JudgeLimitTests(EvaluationCase):
    """The documented probe-then-resume workflow, with a mocked judge only."""

    def probe_fixture(self) -> RunFixture:
        return self.run_fixture(
            (
                TopicSpec(
                    "probe-topic",
                    "A probe narrative.",
                    answers=(
                        ("Claim one.", (0,)),
                        ("Claim two.", (0,)),
                        ("Claim three.", (0,)),
                    ),
                ),
            )
        )

    def test_limit_caps_hosted_calls_and_still_writes_a_valid_bundle(self) -> None:
        fixture = self.probe_fixture()
        judge = RecordingJudge("FS")
        bundle = self.build(fixture, judge=judge, judge_limit=1)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(bundle.manifest["judge"]["hosted_calls"], 1)
        self.assertEqual(bundle.manifest["judge"]["skipped_by_judge_limit"], 2)
        self.assertEqual(bundle.manifest["judge"]["judge_limit"], 1)
        self.assertFalse(bundle.manifest["judgments"]["fully_judged"])
        # A partial bundle is still valid and reports unavailability rather than a zero.
        self.assertTrue(bundle.manifest_path.is_file())
        support = bundle.manifest["metrics"]["citation_support"]
        self.assertEqual(support["macro"], {})
        self.assertFalse(support["macro_availability"]["available"])

    def test_resume_reuses_the_probe_and_calls_only_remaining_misses(self) -> None:
        fixture = self.probe_fixture()
        cache_root = fixture.root / "cache"
        probe = RecordingJudge("FS")
        self.build(fixture, judge=probe, cache_root=cache_root, judge_limit=1)
        self.assertEqual(len(probe.calls), 1)

        resume = RecordingJudge("PS")
        bundle = self.build(
            fixture, judge=resume, cache_root=cache_root, work_dir=fixture.root / "w2"
        )
        self.assertEqual(len(resume.calls), 2)
        self.assertNotIn(probe.calls[0], resume.calls)
        self.assertEqual(bundle.manifest["judge"]["hosted_calls"], 2)
        self.assertEqual(bundle.manifest["judge"]["reused_from_cache"], 1)
        self.assertEqual(bundle.manifest["judge"]["skipped_by_judge_limit"], 0)
        self.assertTrue(bundle.manifest["judgments"]["fully_judged"])
        self.assertEqual(bundle.manifest["judgments"]["label_counts"], {"FS": 1, "PS": 2, "NS": 0})

    def test_a_failed_probe_does_not_burn_the_remaining_quota(self) -> None:
        fixture = self.probe_fixture()
        judge = RecordingJudge([None, "FS", "FS"])
        bundle = self.build(fixture, judge=judge, judge_limit=1)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(bundle.manifest["judge"]["failed"], 1)
        self.assertEqual(bundle.manifest["judge"]["completed"], 0)
        self.assertEqual(bundle.manifest["cache"]["writes"], 0)

    def test_limit_is_inert_without_a_judge(self) -> None:
        fixture = self.probe_fixture()
        bundle = self.build(fixture, judge=None, judge_limit=1)
        self.assertEqual(bundle.manifest["judge"]["hosted_calls"], 0)
        self.assertEqual(bundle.manifest["judgments"]["completed"], 0)

    def test_limit_above_the_miss_count_behaves_like_no_limit(self) -> None:
        fixture = self.probe_fixture()
        judge = RecordingJudge("FS")
        bundle = self.build(fixture, judge=judge, judge_limit=99)
        self.assertEqual(len(judge.calls), 3)
        self.assertEqual(bundle.manifest["judge"]["skipped_by_judge_limit"], 0)
        self.assertTrue(bundle.manifest["judgments"]["fully_judged"])

    def test_judge_workers_bound_concurrency_and_preserve_order(self) -> None:
        fixture = self.run_fixture(THREE_TOPICS)
        judge = ConcurrencyRecordingJudge()
        bundle = self.build(fixture, judge=judge, judge_workers=2)
        rows = _rows(bundle.work_dir / "support_judgments.jsonl")
        expected = [
            row["task_id"] for row in _rows(bundle.work_dir / "support_tasks.jsonl")
        ]
        self.assertEqual(judge.max_active, 2)
        self.assertEqual([row["task_id"] for row in rows], expected)
        self.assertEqual(bundle.manifest["judge"]["judge_workers"], 2)

    def test_cache_writes_are_owned_by_the_controller_thread(self) -> None:
        fixture = self.run_fixture(THREE_TOPICS)
        baseline = self.build(fixture, judge=None, work_dir=fixture.root / "baseline")
        tasks = _rows(baseline.work_dir / "support_tasks.jsonl")
        cache = ControllerRecordingCache(fixture.root / "controller-cache")
        judgments, report = _resolve_judgments(
            tasks,
            cache=cache,
            judge=ConcurrencyRecordingJudge(),
            settings=settings(),
            ragdoll=ragdoll_identity(REPOSITORY_ROOT),
            judge_workers=3,
        )
        self.assertEqual(report["completed"], len(judgments))
        self.assertTrue(cache.put_thread_names)
        self.assertEqual(
            set(cache.put_thread_names),
            {threading.current_thread().name},
        )

    def test_completed_future_is_checkpointed_before_later_interruption(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        baseline = self.build(fixture, judge=None, work_dir=fixture.root / "baseline")
        tasks = _rows(baseline.work_dir / "support_tasks.jsonl")
        cache = JudgeCache(fixture.root / "checkpoint-cache")
        interrupted = CheckpointThenInterruptJudge(str(tasks[0]["task_id"]))

        with self.assertRaises(KeyboardInterrupt):
            _resolve_judgments(
                tasks,
                cache=cache,
                judge=interrupted,
                settings=settings(),
                ragdoll=ragdoll_identity(REPOSITORY_ROOT),
                judge_workers=2,
            )

        first_identity = _judge_identity(
            tasks[0], settings=settings(), ragdoll=ragdoll_identity(REPOSITORY_ROOT)
        )
        self.assertEqual(cache.read(first_identity).support_label, "FS")

        resume = RecordingJudge("PS")
        judgments, report = _resolve_judgments(
            tasks,
            cache=cache,
            judge=resume,
            settings=settings(),
            ragdoll=ragdoll_identity(REPOSITORY_ROOT),
            judge_workers=2,
        )
        self.assertEqual(len(resume.calls), 1)
        self.assertNotIn(str(tasks[0]["task_id"]), resume.calls)
        self.assertEqual(report["reused_from_cache"], 1)
        self.assertEqual(report["hosted_calls"], 1)
        self.assertEqual(
            [row["task_id"] for row in judgments],
            [task["task_id"] for task in tasks],
        )

    def test_probe_limit_preselects_one_miss_even_with_four_workers(self) -> None:
        fixture = self.run_fixture(THREE_TOPICS)
        judge = ConcurrencyRecordingJudge()
        bundle = self.build(fixture, judge=judge, judge_limit=1, judge_workers=4)
        self.assertEqual(bundle.manifest["judge"]["hosted_calls"], 1)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(bundle.manifest["judge"]["judge_workers"], 4)

    def test_cached_tasks_never_occupy_workers(self) -> None:
        fixture = self.run_fixture(THREE_TOPICS)
        cache_root = fixture.root / "cache"
        seed = self.build(fixture, judge=RecordingJudge("FS"), cache_root=cache_root)
        task_ids = {row["task_id"] for row in _rows(seed.work_dir / "support_tasks.jsonl")}
        judge = ConcurrencyRecordingJudge("PS")
        bundle = self.build(
            fixture,
            judge=judge,
            cache_root=cache_root,
            work_dir=fixture.root / "resume",
            judge_workers=4,
        )
        self.assertEqual(judge.calls, [])
        self.assertEqual(bundle.manifest["judge"]["reused_from_cache"], len(task_ids))
        self.assertEqual(bundle.manifest["judge"]["hosted_calls"], 0)

    def test_failed_calls_are_not_cached_and_resume_retries_only_failures(self) -> None:
        fixture = self.run_fixture(THREE_TOPICS)
        cache_root = fixture.root / "cache"
        first = RecordingJudge([None, "FS", None, "FS", None, "FS"])
        # Keep the label sequence deterministic while exercising resumability. The two
        # default-answer topics intentionally share cache identities, so there are five
        # unique judge calls for seven materialized tasks.
        partial = self.build(fixture, judge=first, cache_root=cache_root, judge_workers=1)
        self.assertEqual(partial.manifest["judge"]["failed"], 3)
        self.assertEqual(partial.manifest["cache"]["writes"], 2)

        retry = RecordingJudge("PS")
        complete = self.build(
            fixture,
            judge=retry,
            cache_root=cache_root,
            work_dir=fixture.root / "retry",
            judge_workers=2,
        )
        self.assertEqual(len(retry.calls), 3)
        self.assertEqual(
            complete.manifest["judge"]["reused_from_cache"],
            partial.manifest["judge"]["completed"],
        )
        self.assertEqual(complete.manifest["judge"]["hosted_calls"], 3)
        self.assertTrue(complete.manifest["judgments"]["fully_judged"])

    def test_judge_workers_must_be_positive(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        with self.assertRaisesRegex(EvaluationError, "judge_workers"):
            self.build(fixture, judge=None, judge_workers=0)

    def test_cli_rejects_a_limit_without_run_judge(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(
                ["--retrieval-config", "a.yaml", "--rag-config", "b.yaml", "--judge-limit", "1"]
            )

    def test_cli_rejects_a_non_positive_limit(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(
                [
                    "--retrieval-config", "a.yaml", "--rag-config", "b.yaml",
                    "--run-judge", "--judge-limit", "0",
                ]
            )

    def test_cli_rejects_non_default_workers_without_run_judge(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(
                [
                    "--retrieval-config", "a.yaml", "--rag-config", "b.yaml",
                    "--judge-workers", "2",
                ]
            )

    def test_cli_rejects_non_positive_workers(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(
                [
                    "--retrieval-config", "a.yaml", "--rag-config", "b.yaml",
                    "--run-judge", "--judge-workers", "0",
                ]
            )

    def test_cli_rejects_importing_external_or_legacy_judgments(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        with self.assertRaises(SystemExit):
            cli.main(
                [
                    "--retrieval-config", str(fixture.retrieval_config),
                    "--rag-config", str(fixture.rag_config),
                    "--seed-judgments", str(fixture.root / "legacy.jsonl"),
                ]
            )


class ReportWordingTests(EvaluationCase):
    def test_page_does_not_claim_the_command_is_exact(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        bundle = self.build(fixture, judge=RecordingJudge("FS"))
        page = render_report(
            build_presentation(bundle.manifest, commands=(".venv/bin/python -m x",))
        )
        self.assertNotIn("exact commands", page)
        self.assertIn("Command shape", page)
        self.assertIn("not the exact invocation", page)


class ReceiptContractTests(EvaluationCase):
    """The stdout receipt must expose everything the documented workflow checks."""

    REQUIRED = (
        "schema_version", "manifest_path", "report_path", "work_dir", "topic_ids",
        "judgment_tasks", "completed_judgments", "failed_judgments", "missing_judgments",
        "conflicting_judgments", "reused_from_cache", "hosted_calls", "cache", "label_counts", "fully_judged",
        "judge_limit", "judge_workers", "skipped_by_judge_limit", "exact_invocation",
    )

    def receipt(self, fixture: RunFixture, *extra: str, cache_dir: Path | None = None,
                work: str = "cli-work") -> dict[str, Any]:
        import contextlib, io

        output = io.StringIO()
        argv = [
            "--retrieval-config", str(fixture.retrieval_config),
            "--rag-config", str(fixture.rag_config),
            "--work-dir", str(fixture.root / work),
            "--cache-dir", str(cache_dir or (fixture.root / "cli-cache")),
            "--output", str(fixture.root / f"{work}.html"),
            *extra,
        ]
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(argv), 0)
        return json.loads(output.getvalue().strip().splitlines()[-1])

    def test_receipt_exposes_every_documented_field(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        receipt = self.receipt(fixture)
        for field in self.REQUIRED:
            with self.subTest(field=field):
                self.assertIn(field, receipt)
        self.assertEqual(receipt["judge_workers"], 1)
        self.assertEqual(set(receipt["cache"]), {"hits", "misses", "invalidations", "writes"})

    def test_cache_only_receipt_counts_missing_judgments(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        receipt = self.receipt(fixture)
        self.assertEqual(receipt["hosted_calls"], 0)
        self.assertEqual(receipt["completed_judgments"], 0)
        self.assertEqual(receipt["missing_judgments"], 2)
        self.assertEqual(receipt["failed_judgments"], 0)
        self.assertEqual(receipt["reused_from_cache"], 0)
        self.assertFalse(receipt["fully_judged"])

    def test_receipt_records_effective_non_default_workers(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        with patch.object(cli, "hosted_judge", return_value=RecordingJudge("FS")):
            receipt = self.receipt(
                fixture,
                "--run-judge",
                "--judge-workers",
                "2",
                work="workers",
            )
        self.assertEqual(receipt["judge_workers"], 2)
        manifest = json.loads(Path(receipt["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["judge"]["judge_workers"], 2)

    def test_receipt_fields_track_the_validated_bundle(self) -> None:
        fixture = self.run_fixture(ONE_TOPIC)
        receipt = self.receipt(fixture)
        manifest = json.loads(Path(receipt["manifest_path"]).read_text(encoding="utf-8"))
        judge = manifest["judge"]
        self.assertEqual(receipt["failed_judgments"], judge["failed"])
        self.assertEqual(receipt["conflicting_judgments"], judge["conflicts"])
        self.assertEqual(receipt["missing_judgments"], judge["missing"])
        self.assertEqual(receipt["reused_from_cache"], judge["reused_from_cache"])
        self.assertEqual(receipt["completed_judgments"], judge["completed"])
        self.assertEqual(receipt["judgment_tasks"], judge["tasks"])


class ProbeResumeWorkflowTests(EvaluationCase):
    """The exact decision points the skill tells an operator to check."""

    def probe_fixture(self, claims: Sequence[str]) -> RunFixture:
        return self.run_fixture(
            (
                TopicSpec(
                    "workflow-topic",
                    "A workflow narrative.",
                    answers=tuple((claim, (0,)) for claim in claims),
                ),
            )
        )

    def test_already_fully_cached_needs_no_probe_and_no_calls(self) -> None:
        fixture = self.probe_fixture(["One.", "Two."])
        cache_root = fixture.root / "cache"
        self.build(fixture, judge=RecordingJudge("FS"), cache_root=cache_root)

        judge = RecordingJudge("FS")
        cache_only = self.build(
            fixture, judge=None, cache_root=cache_root, work_dir=fixture.root / "w2"
        )
        self.assertTrue(cache_only.manifest["judgments"]["fully_judged"])
        self.assertEqual(cache_only.manifest["judge"]["hosted_calls"], 0)
        self.assertEqual(cache_only.manifest["judge"]["missing"], 0)
        self.assertEqual(cache_only.manifest["judge"]["reused_from_cache"], 2)
        self.assertEqual(judge.calls, [])  # no probe was ever needed

    def test_partial_cache_plus_one_call_probe(self) -> None:
        """completed_judgments must rise above the cache-only baseline, not equal 1."""
        fixture = self.probe_fixture(["One.", "Two.", "Three."])
        cache_root = fixture.root / "cache"
        # Seed exactly one identity by probing once against a throwaway work dir.
        self.build(
            fixture,
            judge=RecordingJudge("FS"),
            cache_root=cache_root,
            work_dir=fixture.root / "seed",
            judge_limit=1,
        )

        baseline = self.build(
            fixture, judge=None, cache_root=cache_root, work_dir=fixture.root / "baseline"
        )
        self.assertEqual(baseline.manifest["judge"]["completed"], 1)
        self.assertFalse(baseline.manifest["judgments"]["fully_judged"])

        judge = RecordingJudge("PS")
        probe = self.build(
            fixture,
            judge=judge,
            cache_root=cache_root,
            work_dir=fixture.root / "probe",
            judge_limit=1,
        )
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(probe.manifest["judge"]["hosted_calls"], 1)
        self.assertEqual(probe.manifest["judge"]["failed"], 0)
        self.assertGreater(
            probe.manifest["judge"]["completed"], baseline.manifest["judge"]["completed"]
        )
        self.assertEqual(probe.manifest["judge"]["reused_from_cache"], 1)

    def test_probe_receipt_exposes_a_failure(self) -> None:
        fixture = self.probe_fixture(["One.", "Two."])
        probe = self.build(fixture, judge=RecordingJudge([None]), judge_limit=1)
        judge = probe.manifest["judge"]
        self.assertEqual(judge["hosted_calls"], 1)
        self.assertEqual(judge["failed"], 1)
        self.assertEqual(judge["completed"], 0)
        self.assertEqual(judge["missing"], 2)
        self.assertFalse(probe.manifest["judgments"]["fully_judged"])

    def test_one_identity_shared_by_two_tasks_still_counts_as_progress(self) -> None:
        """Identical statements share a cache identity, so one call completes two tasks."""
        fixture = self.probe_fixture(["Repeated claim.", "Repeated claim.", "Distinct claim."])
        cache_root = fixture.root / "cache"
        baseline = self.build(
            fixture, judge=None, cache_root=cache_root, work_dir=fixture.root / "baseline"
        )
        self.assertEqual(baseline.manifest["judge"]["completed"], 0)

        judge = RecordingJudge("FS")
        probe = self.build(
            fixture,
            judge=judge,
            cache_root=cache_root,
            work_dir=fixture.root / "probe",
            judge_limit=1,
        )
        # One hosted call, but the second identical task is served from the new entry.
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(probe.manifest["judge"]["hosted_calls"], 1)
        self.assertEqual(probe.manifest["judge"]["completed"], 2)
        self.assertEqual(probe.manifest["judge"]["failed"], 0)
        self.assertGreater(
            probe.manifest["judge"]["completed"], baseline.manifest["judge"]["completed"]
        )
        # The success rule "completed == 1" would have falsely failed here.
        self.assertNotEqual(probe.manifest["judge"]["completed"], 1)

    def test_resume_exposes_reuse_and_only_remaining_misses(self) -> None:
        fixture = self.probe_fixture(["One.", "Two.", "Three."])
        cache_root = fixture.root / "cache"
        self.build(
            fixture,
            judge=RecordingJudge("FS"),
            cache_root=cache_root,
            work_dir=fixture.root / "probe",
            judge_limit=1,
        )
        resume_judge = RecordingJudge("PS")
        resume = self.build(
            fixture,
            judge=resume_judge,
            cache_root=cache_root,
            work_dir=fixture.root / "resume",
        )
        judge = resume.manifest["judge"]
        self.assertGreaterEqual(judge["reused_from_cache"], 1)
        self.assertEqual(judge["failed"], 0)
        self.assertEqual(judge["hosted_calls"], 2)
        self.assertLess(judge["hosted_calls"], judge["tasks"])
        self.assertEqual(len(resume_judge.calls), 2)
        self.assertTrue(resume.manifest["judgments"]["fully_judged"])


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
