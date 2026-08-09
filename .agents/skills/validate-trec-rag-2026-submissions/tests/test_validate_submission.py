from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib import metadata
from pathlib import Path
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "validate_submission.py"
)
SPEC = importlib.util.spec_from_file_location("rag26_submission_validator", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load validator script: {SCRIPT_PATH}")
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


class RetrievalValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_text(self, name: str, body: str) -> Path:
        path = self.root / name
        path.write_text(body, encoding="utf-8")
        return path

    def write_bytes(self, name: str, body: bytes) -> Path:
        path = self.root / name
        path.write_bytes(body)
        return path

    def write_topics_tsv(self, *rows: tuple[str, str]) -> Path:
        return self.write_text(
            "topics.tsv",
            "".join(f"{topic_id}\t{narrative}\n" for topic_id, narrative in rows),
        )

    def one_topic(self):
        return validator.load_topics(
            self.write_topics_tsv(("rag2026-0", "First narrative"))
        )

    def assert_retrieval_failure(
        self,
        body: str | bytes,
        expected_message: str,
        *,
        topics=None,
    ):
        run = (
            self.write_bytes("run.tsv", body)
            if isinstance(body, bytes)
            else self.write_text("run.tsv", body)
        )
        result = validator.validate_retrieval(run, topics or self.one_topic())
        self.assertEqual("fail", result.status)
        self.assertTrue(
            any(expected_message in finding.message for finding in result.findings),
            tuple(finding.message for finding in result.findings),
        )
        return result

    def test_retrieval_accepts_variable_depth_and_reports_counts(self) -> None:
        topics = self.write_topics_tsv(
            ("rag2026-0", "First narrative"),
            ("rag2026-1", "Second narrative"),
        )
        run = self.write_text(
            "run.tsv",
            "rag2026-0 Q0 shard_00001_1 1 2.0 run-a\n"
            "rag2026-0 Q0 shard_00002_2 2 1.0 run-a\n"
            "rag2026-1 Q0 shard_00003_3 1 9.0 run-a\n",
        )

        result = validator.validate_retrieval(run, validator.load_topics(topics))

        self.assertEqual("pass", result.status)
        self.assertEqual(3, result.row_count)
        self.assertEqual(2, result.topic_count)
        self.assertEqual((1, 2), (result.depth_min, result.depth_max))
        self.assertEqual((), result.findings)

    def test_retrieval_ignores_blank_lines(self) -> None:
        run = self.write_text(
            "run.tsv",
            "\nrag2026-0 Q0 shard_00001_1 1 1.0 run-a\n\n",
        )

        result = validator.validate_retrieval(run, self.one_topic())

        self.assertEqual("pass", result.status)
        self.assertEqual(1, result.row_count)

    def test_topics_request_jsonl_is_accepted(self) -> None:
        topics = self.write_text(
            "topics.jsonl",
            json.dumps({"request_id": "rag2026-0", "title": "First narrative"})
            + "\n",
        )

        loaded = validator.load_topics(topics)

        self.assertEqual(
            (("rag2026-0", "First narrative"),),
            tuple((topic.topic_id, topic.narrative) for topic in loaded),
        )

    def test_topics_reject_duplicate_ids(self) -> None:
        topics = self.write_topics_tsv(
            ("rag2026-0", "First narrative"),
            ("rag2026-0", "Repeated narrative"),
        )

        with self.assertRaisesRegex(ValueError, "duplicate topic ID"):
            validator.load_topics(topics)

    def test_tsv_topics_preserve_exact_narrative_whitespace(self) -> None:
        topics = self.write_topics_tsv(
            ("rag2026-0", "  First narrative with exact whitespace  ")
        )

        loaded = validator.load_topics(topics)

        self.assertEqual(
            "  First narrative with exact whitespace  ", loaded[0].narrative
        )

    def test_retrieval_rejects_malformed_column_count(self) -> None:
        self.assert_retrieval_failure(
            "rag2026-0 Q0 shard_00001_1 1 1.0\n",
            "expected six columns",
        )

    def test_retrieval_rejects_wrong_q0(self) -> None:
        self.assert_retrieval_failure(
            "rag2026-0 XX shard_00001_1 1 1.0 run-a\n",
            "column 2 must be Q0",
        )

    def test_retrieval_rejects_missing_topic(self) -> None:
        topics = validator.load_topics(
            self.write_topics_tsv(
                ("rag2026-0", "First narrative"),
                ("rag2026-1", "Second narrative"),
            )
        )
        self.assert_retrieval_failure(
            "rag2026-0 Q0 shard_00001_1 1 1.0 run-a\n",
            "missing expected topics: rag2026-1",
            topics=topics,
        )

    def test_retrieval_rejects_extra_topic(self) -> None:
        self.assert_retrieval_failure(
            "rag2026-0 Q0 shard_00001_1 1 2.0 run-a\n"
            "rag2026-1 Q0 shard_00002_2 1 1.0 run-a\n",
            "topics outside expected population: rag2026-1",
        )

    def test_retrieval_rejects_rank_not_starting_at_one(self) -> None:
        self.assert_retrieval_failure(
            "rag2026-0 Q0 shard_00001_1 2 1.0 run-a\n",
            "ranks must start at 1 and be dense",
        )

    def test_retrieval_rejects_rank_gap(self) -> None:
        self.assert_retrieval_failure(
            "rag2026-0 Q0 shard_00001_1 1 2.0 run-a\n"
            "rag2026-0 Q0 shard_00002_2 3 1.0 run-a\n",
            "ranks must start at 1 and be dense",
        )

    def test_retrieval_rejects_duplicate_topic_document_pair(self) -> None:
        self.assert_retrieval_failure(
            "rag2026-0 Q0 shard_00001_1 1 2.0 run-a\n"
            "rag2026-0 Q0 shard_00001_1 2 1.0 run-a\n",
            "duplicate document ID",
        )

    def test_retrieval_rejects_increasing_score(self) -> None:
        self.assert_retrieval_failure(
            "rag2026-0 Q0 shard_00001_1 1 1.0 run-a\n"
            "rag2026-0 Q0 shard_00002_2 2 2.0 run-a\n",
            "scores must be non-increasing",
        )

    def test_retrieval_rejects_nonfinite_scores(self) -> None:
        for score in ("nan", "inf", "-inf"):
            with self.subTest(score=score):
                self.assert_retrieval_failure(
                    f"rag2026-0 Q0 shard_00001_1 1 {score} run-a\n",
                    "score must be finite",
                )

    def test_retrieval_rejects_invalid_climbmix_id(self) -> None:
        self.assert_retrieval_failure(
            "rag2026-0 Q0 doc-a 1 1.0 run-a\n",
            "invalid ClimbMix document ID",
        )

    def test_retrieval_rejects_conflicting_run_ids(self) -> None:
        self.assert_retrieval_failure(
            "rag2026-0 Q0 shard_00001_1 1 2.0 run-a\n"
            "rag2026-0 Q0 shard_00002_2 2 1.0 run-b\n",
            "conflicting run IDs",
        )

    def test_retrieval_rejects_invalid_utf8(self) -> None:
        self.assert_retrieval_failure(
            b"rag2026-0 Q0 shard_00001_1 1 1.0 run-a\xff\n",
            "not valid UTF-8",
        )

    def test_retrieval_io_error_is_normalized(self) -> None:
        missing = self.root / "missing.tsv"

        result = validator.validate_retrieval(missing, self.one_topic())

        self.assertEqual("fail", result.status)
        self.assertTrue(
            any("could not read retrieval run" in item.message for item in result.findings)
        )

    def test_retrieval_rejects_empty_run(self) -> None:
        result = self.assert_retrieval_failure("", "run contains no rows")
        self.assertEqual(0, result.row_count)


class RagValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.topics_tsv = self.root / "topics.tsv"
        self.topics_tsv.write_text(
            "rag2026-0\tFirst narrative\n", encoding="utf-8"
        )
        self.topics_jsonl = self.root / "topics.jsonl"
        self.topics_jsonl.write_text(
            json.dumps(
                {"request_id": "rag2026-0", "title": "First narrative"}
            )
            + "\n",
            encoding="utf-8",
        )
        self.valid_rag = self.root / "rag.jsonl"
        self.valid_rag.write_text(
            json.dumps(self.report_record()) + "\n", encoding="utf-8"
        )
        self.recorded_commands: list[tuple[str, ...]] = []
        self.recorded_topics: list[str] = []
        self.runner_returncode = 0
        self.runner_stdout = "PASS: report is valid\n"
        self.runner_stderr = ""

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def report_record(*, narrative: str = "First narrative") -> dict:
        return {
            "metadata": {
                "team_id": "test-team",
                "narrative_id": "rag2026-0",
                "narrative": narrative,
                "run_id": "test-run",
                "run_desc": "Validator integration fixture",
            },
            "references": ["shard_00001_1"],
            "answer": [{"text": "Supported fact.", "citations": [0]}],
        }

    def recording_runner(self, command, **kwargs):
        command = tuple(str(part) for part in command)
        self.recorded_commands.append(command)
        topics_index = command.index("--topics") + 1
        self.recorded_topics.append(
            Path(command[topics_index]).read_text(encoding="utf-8")
        )
        return subprocess.CompletedProcess(
            command,
            self.runner_returncode,
            stdout=self.runner_stdout,
            stderr=self.runner_stderr,
        )

    def test_rag_uses_rag26_check_and_converted_tsv_topics(self) -> None:
        result = validator.validate_rag(
            self.valid_rag,
            self.topics_tsv,
            strict=False,
            runner=self.recording_runner,
        )

        command = self.recorded_commands[0]
        self.assertIn("autojudge_base.report_tool", command)
        self.assertIn("check", command)
        self.assertIn("--spec", command)
        self.assertIn("rag26", command)
        self.assertNotIn("--strict", command)
        self.assertEqual("pass", result.status)
        self.assertEqual(
            {
                "request_id": "rag2026-0",
                "title": "First narrative",
            },
            json.loads(self.recorded_topics[0]),
        )

    def test_rag_strict_flag_is_forwarded(self) -> None:
        validator.validate_rag(
            self.valid_rag,
            self.topics_tsv,
            strict=True,
            runner=self.recording_runner,
        )
        self.assertIn("--strict", self.recorded_commands[0])

    def test_compatible_local_autojudge_is_preferred(self) -> None:
        with (
            mock.patch.object(validator.metadata, "version", return_value="0.4.3"),
            mock.patch.object(validator.shutil, "which", return_value=None),
        ):
            command = validator.build_autojudge_command(strict=False)

        self.assertEqual(sys.executable, command[0])
        self.assertEqual(
            ("-m", "autojudge_base.report_tool", "check"), command[1:]
        )

    def test_compatible_pep440_post_and_local_versions_are_preferred(self) -> None:
        for installed_version in (
            "0.4.3.post1",
            "0.4.3-post",
            "0.4.3-r",
            "0.5.0+local",
        ):
            with self.subTest(installed_version=installed_version):
                with (
                    mock.patch.object(
                        validator.metadata,
                        "version",
                        return_value=installed_version,
                    ),
                    mock.patch.object(validator.shutil, "which", return_value=None),
                ):
                    command = validator.build_autojudge_command(strict=False)
                self.assertEqual(sys.executable, command[0])

    def test_minimum_pep440_prereleases_use_uv_fallback(self) -> None:
        for installed_version in ("0.4.3-rc1", "0.4.3-alpha1", "0.4.3.dev1"):
            with self.subTest(installed_version=installed_version):
                with (
                    mock.patch.object(
                        validator.metadata,
                        "version",
                        return_value=installed_version,
                    ),
                    mock.patch.object(
                        validator.shutil, "which", return_value="/usr/bin/uv"
                    ),
                ):
                    command = validator.build_autojudge_command(strict=False)
                self.assertEqual("uv", command[0])

    def test_isolated_uv_fallback_is_used_for_old_local_package(self) -> None:
        with (
            mock.patch.object(validator.metadata, "version", return_value="0.4.2"),
            mock.patch.object(validator.shutil, "which", return_value="/usr/bin/uv"),
        ):
            command = validator.build_autojudge_command(strict=False)

        self.assertEqual("uv", command[0])
        self.assertIn("--isolated", command)
        self.assertIn("--no-project", command)
        self.assertIn("autojudge-base>=0.4.3", command)

    def test_missing_autojudge_and_uv_returns_setup_failure(self) -> None:
        with (
            mock.patch.object(
                validator.metadata,
                "version",
                side_effect=metadata.PackageNotFoundError,
            ),
            mock.patch.object(validator.shutil, "which", return_value=None),
        ):
            result = validator.validate_rag(
                self.valid_rag,
                self.topics_tsv,
                strict=False,
                runner=self.recording_runner,
            )

        self.assertEqual("fail", result.status)
        self.assertTrue(
            any("autojudge-base>=0.4.3" in item.message for item in result.findings)
        )
        self.assertEqual([], self.recorded_commands)

    def test_autojudge_nonzero_exit_is_failure(self) -> None:
        self.runner_returncode = 255
        self.runner_stderr = "invalid report"

        result = validator.validate_rag(
            self.valid_rag,
            self.topics_tsv,
            strict=False,
            runner=self.recording_runner,
        )

        self.assertEqual("fail", result.status)
        self.assertIn("AutoJudge exited with status 255", result.findings[0].message)
        self.assertIn("invalid report", result.detail)

    def test_autojudge_json_parse_failure_has_concise_finding(self) -> None:
        self.runner_returncode = 1
        self.runner_stderr = (
            "json.decoder.JSONDecodeError: Expecting value: "
            "line 1 column 1 (char 0)"
        )

        result = validator.validate_rag(
            self.valid_rag,
            self.topics_tsv,
            strict=False,
            runner=self.recording_runner,
        )

        self.assertEqual("fail", result.status)
        self.assertIn("not valid JSONL", result.findings[0].message)
        self.assertIn("line 1, column 1", result.findings[0].message)
        self.assertIn("JSONDecodeError", result.detail)

    def test_autojudge_smell_is_pass_with_warnings(self) -> None:
        self.runner_stdout = "PASS\nSMELL: citation could be improved\n"

        result = validator.validate_rag(
            self.valid_rag,
            self.topics_tsv,
            strict=False,
            runner=self.recording_runner,
        )

        self.assertEqual("pass-with-warnings", result.status)
        self.assertTrue(any("SMELL" in item.message for item in result.findings))

    def test_request_jsonl_topics_are_passed_through(self) -> None:
        validator.validate_rag(
            self.valid_rag,
            self.topics_jsonl,
            strict=False,
            runner=self.recording_runner,
        )

        command = self.recorded_commands[0]
        self.assertEqual(
            str(self.topics_jsonl), command[command.index("--topics") + 1]
        )

    def test_cli_accepts_repeated_inputs_and_prints_combined_summary(self) -> None:
        retrieval = self.root / "retrieval.tsv"
        retrieval.write_text(
            "rag2026-0 Q0 shard_00001_1 1 1.0 run-a\n", encoding="utf-8"
        )
        rag_result = validator.ArtifactResult(
            "rag",
            self.valid_rag,
            "pass",
            None,
            None,
            None,
            None,
            (),
            "PASS: report is valid",
        )
        output = io.StringIO()

        with (
            mock.patch.object(validator, "validate_rag", return_value=rag_result) as rag,
            redirect_stdout(output),
        ):
            exit_code = validator.main(
                [
                    "--topics",
                    str(self.topics_tsv),
                    "--retrieval",
                    str(retrieval),
                    "--retrieval",
                    str(retrieval),
                    "--rag",
                    str(self.valid_rag),
                    "--rag",
                    str(self.valid_rag),
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual(2, rag.call_count)
        self.assertEqual(2, output.getvalue().count("[retrieval] PASS"))
        self.assertEqual(2, output.getvalue().count("[rag] PASS"))

    def test_cli_returns_nonzero_for_worst_failure(self) -> None:
        rag_result = validator.ArtifactResult(
            "rag",
            self.valid_rag,
            "fail",
            None,
            None,
            None,
            None,
            (validator.Finding("bad report"),),
        )
        with (
            mock.patch.object(validator, "validate_rag", return_value=rag_result),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = validator.main(
                [
                    "--topics",
                    str(self.topics_tsv),
                    "--rag",
                    str(self.valid_rag),
                ]
            )
        self.assertEqual(1, exit_code)

    def test_cli_reports_missing_retrieval_and_continues_to_rag(self) -> None:
        rag_result = validator.ArtifactResult(
            "rag",
            self.valid_rag,
            "pass",
            None,
            None,
            None,
            None,
            (),
        )
        output = io.StringIO()
        with (
            mock.patch.object(validator, "validate_rag", return_value=rag_result),
            redirect_stdout(output),
        ):
            exit_code = validator.main(
                [
                    "--topics",
                    str(self.topics_tsv),
                    "--retrieval",
                    str(self.root / "missing.tsv"),
                    "--rag",
                    str(self.valid_rag),
                ]
            )

        self.assertEqual(1, exit_code)
        self.assertIn("[retrieval] FAIL", output.getvalue())
        self.assertIn("[rag] PASS", output.getvalue())

    def test_cli_rejects_invocation_without_submission_artifacts(self) -> None:
        with self.assertRaises(SystemExit) as error:
            validator.main(["--topics", str(self.topics_tsv)])
        self.assertEqual(2, error.exception.code)

    @unittest.skipUnless(shutil.which("uv"), "uv required")
    def test_real_isolated_autojudge_accepts_valid_and_rejects_mismatch(self) -> None:
        mismatched = self.root / "mismatched.jsonl"
        mismatched.write_text(
            json.dumps(self.report_record(narrative="Wrong narrative")) + "\n",
            encoding="utf-8",
        )
        whitespace_topics = self.root / "whitespace-topics.tsv"
        whitespace_topics.write_text(
            "rag2026-0\t  First narrative  \n", encoding="utf-8"
        )
        with mock.patch.object(
            validator.metadata,
            "version",
            side_effect=metadata.PackageNotFoundError,
        ):
            valid = validator.validate_rag(
                self.valid_rag, self.topics_tsv, strict=False
            )
            invalid = validator.validate_rag(
                mismatched, self.topics_tsv, strict=False
            )
            whitespace_mismatch = validator.validate_rag(
                self.valid_rag, whitespace_topics, strict=False
            )

        self.assertEqual("pass", valid.status, valid.detail)
        self.assertEqual("fail", invalid.status, invalid.detail)
        self.assertEqual("fail", whitespace_mismatch.status, whitespace_mismatch.detail)


if __name__ == "__main__":
    unittest.main()
