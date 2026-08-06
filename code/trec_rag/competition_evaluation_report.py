"""CLI for the full evaluated friendly post-run report.

This is the authoritative path for the response-and-judgment report. Both configs are
required: the retrieval config alone can only produce the raw debug report, which is a
different command (``trec_rag.competition_debug_report``) and must not be mistaken for
this one.

Without ``--run-judge`` the command makes no hosted calls at all. It reuses validated
cache entries and renders whatever is completely judged, reporting the rest as explicitly
unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from trec_rag.friendly_report import denylist_from_bundle, write_report
from trec_rag.offline_evaluation import (
    JudgeSettings,
    build_evaluation_bundle,
)
from trec_rag.repo_env import find_repo_root


DEFAULT_CACHE_SUBDIR = Path("cache") / "ragdoll_support_judge"
AGENT_BINARY = "pi"
# No RAGDoll agent extension is configured for support judging. If one is ever added it
# must be named here so it becomes part of the cache identity.
EXTENSION_IDENTITY = "none"


def _private_directory(path: Path | None) -> Path:
    if path is not None:
        directory = Path(path).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(stat.S_IRWXU)
        return directory
    return Path(tempfile.mkdtemp(prefix="trec-rag-offline-eval-"))


def _portable(path: Path, repository_root: Path) -> str:
    """Render a path relative to the repository root so no private layout is published."""
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(repository_root))
    except ValueError:
        return resolved.name


def judge_settings(repository_root: Path) -> JudgeSettings:
    """Read the pinned judge settings from RAGDoll rather than restating them."""
    from ragdoll.config import (
        DEFAULT_MODEL,
        DEFAULT_PROVIDER,
        DEFAULT_SYSTEM_PROMPT,
        DEFAULT_THINKING,
        resolve_thinking,
    )

    return JudgeSettings(
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL,
        thinking=resolve_thinking(DEFAULT_MODEL, DEFAULT_THINKING),
        temperature=None,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        agent_binary=AGENT_BINARY,
        extension_identity=EXTENSION_IDENTITY,
    )


def local_agent_config(settings: JudgeSettings):
    """Build the RAGDoll agent config for the hosted judge.

    ``cache_dir=None`` is required: RAGDoll's own cache is a TTL cache that expires entries
    and never revalidates that a reused row is a completed judgment. This workflow's
    validated cache replaces it, so the weaker one must stay switched off.
    """
    from ragdoll.config import LocalAgentConfig

    return LocalAgentConfig(
        agent_binary=settings.agent_binary,
        provider=settings.provider,
        model=settings.model,
        thinking=settings.thinking,
        system_prompt=settings.system_prompt,
        temperature=settings.temperature,
        cache_dir=None,
    )


def hosted_judge(settings: JudgeSettings, raw_events_dir: Path):
    """Build the real hosted judge. Only reachable behind ``--run-judge``."""
    raw_events_dir = Path(raw_events_dir)
    raw_events_dir.mkdir(parents=True, exist_ok=True)
    raw_events_dir.chmod(stat.S_IRWXU)

    def judge(task):
        import asyncio

        from ragdoll.runner import run_prompt
        from ragdoll.support.prompts import parse_support_label

        from trec_rag.offline_evaluation import JudgeOutcome

        result = asyncio.run(
            run_prompt(
                task_id=str(task["task_id"]),
                evaluator=str(task["evaluator"]),
                instruction=str(task["instruction"]),
                raw_events_dir=raw_events_dir,
                config=local_agent_config(settings),
            )
        )
        if result.get("status") != "completed":
            return JudgeOutcome(status="failed", error=str(result.get("error") or "judge failed"))
        label = parse_support_label(str(result.get("output_text") or ""))
        if label is None:
            return JudgeOutcome(status="failed", error="judge output carried no support label")
        return JudgeOutcome(status="completed", support_label=label)

    return judge


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trec_rag.competition_evaluation_report",
        description=(
            "Evaluate a completed retrieval + RAG run and render the private friendly report. "
            "Both configs are required; use trec_rag.competition_debug_report for the "
            "retrieval-only raw debug report."
        ),
    )
    parser.add_argument("--retrieval-config", type=Path, required=True)
    parser.add_argument("--rag-config", type=Path, required=True)
    parser.add_argument("--topic", action="append", dest="topic_ids", default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--qrels", type=Path, default=None)
    parser.add_argument("--gold-nuggets", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--run-judge",
        action="store_true",
        help="Allow hosted RAGDoll judge calls for validated cache misses. Omitted: zero calls.",
    )
    parser.add_argument(
        "--judge-limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap hosted judge calls to N validated cache misses for this invocation. Use "
            "--run-judge --judge-limit 1 to probe one task, then rerun without the limit so "
            "the probe is a cache hit and only remaining misses call the judge."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.judge_limit is not None and arguments.judge_limit < 1:
        parser.error("--judge-limit must be a positive number of hosted calls")
    if arguments.judge_limit is not None and not arguments.run_judge:
        parser.error("--judge-limit only applies with --run-judge, which makes hosted calls")

    repository_root = find_repo_root(Path.cwd())
    work_dir = _private_directory(arguments.work_dir)
    cache_root = (
        Path(arguments.cache_dir).expanduser().resolve()
        if arguments.cache_dir is not None
        else repository_root / DEFAULT_CACHE_SUBDIR
    )
    settings = judge_settings(repository_root)
    judge = hosted_judge(settings, work_dir / "raw-events") if arguments.run_judge else None

    bundle = build_evaluation_bundle(
        retrieval_config_path=arguments.retrieval_config,
        rag_config_path=arguments.rag_config,
        work_dir=work_dir,
        repository_root=repository_root,
        cache_root=cache_root,
        topic_ids=arguments.topic_ids,
        qrels_path=arguments.qrels,
        gold_nuggets_path=arguments.gold_nuggets,
        judge=judge,
        judge_settings=settings,
        judge_limit=arguments.judge_limit,
    )

    output_path = (
        Path(arguments.output).expanduser().resolve()
        if arguments.output is not None
        else work_dir / "evaluation_report.html"
    )
    # Published in the HTML, so it must stay repo-relative: absolute interpreter, work,
    # cache, qrels, and output paths would leak the private filesystem layout and the privacy
    # scan rejects them. This is therefore a portable command *shape*, not the exact
    # invocation; the exact one is recorded privately in the receipt below.
    command = " ".join(
        shlex.quote(part)
        for part in [
            ".venv/bin/python",
            "-m",
            "trec_rag.competition_evaluation_report",
            "--retrieval-config",
            _portable(arguments.retrieval_config, repository_root),
            "--rag-config",
            _portable(arguments.rag_config, repository_root),
            *[part for topic_id in (arguments.topic_ids or []) for part in ("--topic", topic_id)],
        ]
    )
    # The full invocation, private paths and judge options included, stays in the private
    # stdout receipt and never reaches the HTML.
    exact_invocation = " ".join(
        shlex.quote(part)
        for part in [
            sys.executable,
            "-m",
            "trec_rag.competition_evaluation_report",
            *(argv if argv is not None else sys.argv[1:]),
        ]
    )
    write_report(
        bundle.manifest,
        output_path,
        denylist=denylist_from_bundle(work_dir),
        commands=(command,),
    )

    judge_report = bundle.manifest["judge"]
    print(
        json.dumps(
            {
                "schema_version": bundle.manifest["schema_version"],
                "manifest_path": str(bundle.manifest_path),
                "report_path": str(output_path),
                "work_dir": str(work_dir),
                "topic_ids": bundle.manifest["scope"]["topic_ids"],
                "judgment_tasks": judge_report["tasks"],
                "completed_judgments": judge_report["completed"],
                # The probe/resume workflow is judged on these, so expose them rather than
                # asking an operator to infer them from the cache counters.
                "failed_judgments": judge_report["failed"],
                "conflicting_judgments": judge_report["conflicts"],
                "missing_judgments": judge_report["missing"],
                "reused_from_cache": judge_report["reused_from_cache"],
                "hosted_calls": judge_report["hosted_calls"],
                "cache": bundle.manifest["cache"],
                "label_counts": bundle.manifest["judgments"]["label_counts"],
                "fully_judged": bundle.manifest["judgments"]["fully_judged"],
                "judge_limit": arguments.judge_limit,
                "skipped_by_judge_limit": judge_report["skipped_by_judge_limit"],
                "exact_invocation": exact_invocation,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
