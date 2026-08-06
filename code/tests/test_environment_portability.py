"""The evaluation workflow must run from a plain, documented repository setup.

The pinned ``ragdoll/`` submodule is a declared dependency (a uv editable path source),
so a fresh environment produced by ``code/tools/setup_env.sh`` can import it with no
``PYTHONPATH`` tweak. These tests fail if that declaration regresses.

Everything runs in a subprocess with ``PYTHONPATH`` removed and the working directory set
outside the repository. That matters: running from the repository root makes the
``ragdoll/`` directory look like an empty namespace package, which lets ``import ragdoll``
succeed while ``import ragdoll.config`` fails — exactly the failure this guards against.
No subprocess here contacts a model, a provider, or the network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PINNED_RAGDOLL_SOURCE = REPOSITORY_ROOT / "ragdoll" / "src" / "ragdoll"
TIMEOUT_SECONDS = 120


def portable_environment() -> dict[str, str]:
    """The caller's environment minus every import-path override."""
    environment = dict(os.environ)
    for variable in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        environment.pop(variable, None)
    # Keep the subprocess from writing bytecode into the read-only-ish source tree.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def run_isolated(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the configured interpreter from a neutral directory with no PYTHONPATH."""
    with tempfile.TemporaryDirectory(prefix="portability-cwd-") as neutral:
        return subprocess.run(
            [sys.executable, *arguments],
            cwd=neutral,
            env=portable_environment(),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )


class InterpreterSetupTests(unittest.TestCase):
    def test_pythonpath_is_not_leaking_into_the_subprocess(self) -> None:
        result = run_isolated(
            "-c", "import os; print(repr(os.environ.get('PYTHONPATH')))"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "None")


class RagdollImportTests(unittest.TestCase):
    """`ragdoll` must be an installed package, not a PYTHONPATH accident."""

    def test_required_ragdoll_modules_import(self) -> None:
        probe = (
            "import ragdoll, ragdoll.config, ragdoll.runner, "
            "ragdoll.support.prompts, ragdoll.support.stages, "
            "ragdoll.support.metrics, ragdoll.support.assignments;"
            "import json;"
            "print(json.dumps({"
            "'package': ragdoll.__file__,"
            "'config': ragdoll.config.__file__,"
            "'provider': ragdoll.config.DEFAULT_PROVIDER,"
            "'model': ragdoll.config.DEFAULT_MODEL,"
            "}))"
        )
        result = run_isolated("-c", probe)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(payload["provider"])
        self.assertTrue(payload["model"])
        for key in ("package", "config"):
            with self.subTest(module=key):
                self.assertTrue(
                    Path(payload[key]).is_file(),
                    msg=f"{key} did not resolve to a real file: {payload[key]}",
                )

    def test_ragdoll_resolves_to_the_repository_pinned_submodule(self) -> None:
        """A different ragdoll on the machine must not shadow the pinned one."""
        result = run_isolated("-c", "import ragdoll; print(ragdoll.__file__)")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        resolved = Path(result.stdout.strip().splitlines()[-1]).resolve()
        self.assertEqual(resolved.parent, PINNED_RAGDOLL_SOURCE.resolve())

    def test_ragdoll_is_not_an_empty_namespace_package(self) -> None:
        result = run_isolated(
            "-c",
            "import ragdoll; import sys; "
            "print(type(getattr(ragdoll, '__path__', None)).__name__)",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        # A namespace package exposes _NamespacePath; a real package exposes a plain list.
        self.assertEqual(result.stdout.strip().splitlines()[-1], "list")


class RepositoryImportTests(unittest.TestCase):
    def test_trec_rag_modules_import_without_pythonpath(self) -> None:
        probe = (
            "import trec_rag.offline_evaluation as oe, "
            "trec_rag.friendly_report as fr, "
            "trec_rag.judge_cache as jc, "
            "trec_rag.competition_evaluation_report as cli;"
            "print(oe.BUNDLE_SCHEMA_VERSION)"
        )
        result = run_isolated("-c", probe)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(
            result.stdout.strip().splitlines()[-1], "trec_rag_offline_evaluation_bundle_v1"
        )


class CliEntryPointTests(unittest.TestCase):
    """The commands the skill advertises must run as written."""

    def test_evaluation_report_help_runs(self) -> None:
        result = run_isolated("-m", "trec_rag.competition_evaluation_report", "--help")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        for flag in (
            "--retrieval-config",
            "--rag-config",
            "--topic",
            "--work-dir",
            "--qrels",
            "--gold-nuggets",
            "--output",
            "--run-judge",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, result.stdout)

    def test_debug_report_help_runs(self) -> None:
        result = run_isolated("-m", "trec_rag.competition_debug_report", "--help")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--retrieval-config", result.stdout)

    def test_evaluation_report_requires_both_configs(self) -> None:
        result = run_isolated(
            "-m", "trec_rag.competition_evaluation_report", "--retrieval-config", "x.yaml"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--rag-config", result.stderr)

    def test_pinned_judge_settings_resolve_without_pythonpath(self) -> None:
        """The judge identity reads RAGDoll's pinned settings; no hosted call is made."""
        probe = (
            "from pathlib import Path;"
            "from trec_rag.competition_evaluation_report import judge_settings, local_agent_config;"
            "import json;"
            f"s = judge_settings(Path({str(REPOSITORY_ROOT)!r}));"
            "c = local_agent_config(s);"
            "print(json.dumps({'provider': s.provider, 'model': s.model, "
            "'thinking': s.thinking, 'cache_dir': c.cache_dir}))"
        )
        result = run_isolated("-c", probe)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(payload["provider"])
        self.assertTrue(payload["model"])
        self.assertNotEqual(payload["thinking"], payload["model"])
        self.assertIsNone(payload["cache_dir"], msg="RAGDoll's TTL cache must stay off")


class CacheOnlyPathTests(unittest.TestCase):
    """A representative offline path: cache round-trip, no judge, no network."""

    def test_judge_cache_round_trip_in_a_clean_interpreter(self) -> None:
        probe = (
            "import json, tempfile;"
            "from pathlib import Path;"
            "from trec_rag.judge_cache import JudgeCache, JudgeIdentity;"
            "from ragdoll.support.prompts import render_support_prompt;"
            "d = Path(tempfile.mkdtemp());"
            "i = JudgeIdentity(evaluator='support',"
            "instruction=render_support_prompt(statement='s', citation='c'),"
            "ragdoll_version='0.0.0', ragdoll_commit='0'*40,"
            "prompt_contract_sha256='1'*64, task_schema_version='v1',"
            "provider='p', model='m', thinking='t', temperature=None,"
            "system_prompt='sp', agent_binary='b', extension_identity='none');"
            "c = JudgeCache(d);"
            "miss = c.get(i);"
            "c.put(i, support_label='FS');"
            "print(json.dumps({'miss': miss, 'hit': c.get(i), 'stats': c.stats()}))"
        )
        result = run_isolated("-c", probe)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertIsNone(payload["miss"])
        self.assertEqual(payload["hit"], "FS")
        self.assertEqual(payload["stats"]["hits"], 1)
        self.assertEqual(payload["stats"]["writes"], 1)


class DeclaredDependencyTests(unittest.TestCase):
    """The declaration itself, so the fix cannot be silently undone."""

    def test_pyproject_declares_the_local_ragdoll_source(self) -> None:
        import tomllib

        data = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertIn("ragdoll", data["project"]["dependencies"])
        source = data["tool"]["uv"]["sources"]["ragdoll"]
        self.assertEqual(source["path"], "ragdoll")
        self.assertTrue(source.get("editable"))

    def test_lockfile_pins_ragdoll_to_the_local_directory(self) -> None:
        lock = (REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8")
        self.assertIn('name = "ragdoll"', lock)
        self.assertIn('editable = "ragdoll"', lock)


if __name__ == "__main__":
    unittest.main()
