"""Full-run orchestration for the frozen multi-stage competition RAG path."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Sequence
import json
import os
from pathlib import Path
import sys
from typing import Any

from filelock import FileLock, Timeout as FileLockTimeout

from trec_rag.competition_rag import (
    RagGenerationConfig,
    _atomic_write_text,
    _redact,
    _validate_artifact_paths,
    _validate_exact_hint_citations,
    _validate_generated_submission_record,
    load_rag_generation_config,
)
from trec_rag.generation_handoff import (
    GenerationHandoff,
    GenerationTopic,
    load_generation_handoff,
    select_generation_topics,
)
# The bounded-revision names are a supported deadline contract shared with the
# one-topic CLI. Their identities and signatures are load-bearing for resume.
from trec_rag.narrative_blueprint_trial import (
    TRIAL_CONTRACT_VERSION,
    _bounded_identity,
    _bounded_private_root,
    _bounded_read_state,
    _bounded_revalidate_files,
    _run_bounded_revision,
)
from trec_rag.repo_env import find_repo_root, load_repo_env


MULTISTAGE_IDENTITY_VERSION = 1
_IDENTITY_FILENAME = "multistage_generation_identity.json"
_LOCK_FILENAME = ".multistage-generation.lock"
_FAILURES_FILENAME = "failures.json"
_CRASH_FALLBACK_WARNING = (
    "operation-screen call was crash-consumed; published validated draft fallback"
)
TopicRunner = Callable[..., Awaitable[Path]]


def _multistage_identity(
    config: RagGenerationConfig,
    handoff: GenerationHandoff,
    topics: Sequence[GenerationTopic],
) -> dict[str, Any]:
    return {
        "identity_version": MULTISTAGE_IDENTITY_VERSION,
        "trial_contract_version": TRIAL_CONTRACT_VERSION,
        "handoff_schema_version": handoff.schema_version,
        "handoff_manifest_sha256": handoff.manifest_sha256,
        "submission_run_id": f"{config.run_id}-final",
        "topics": [
            {
                "topic_id": topic.topic_id,
                "context_sha256": topic.context_sha256,
                "bounded_identity": _bounded_identity(config, handoff, topic),
            }
            for topic in topics
        ],
    }


def _load_final_record(
    root: Path,
    *,
    config: RagGenerationConfig,
    handoff: GenerationHandoff,
    topic: GenerationTopic,
) -> dict[str, Any]:
    expected_root = _bounded_private_root(config, topic)
    if root.resolve() != expected_root.resolve():
        raise ValueError(f"multi-stage topic root differs for {topic.topic_id}")
    state = _bounded_read_state(root)
    if state.get("identity") != _bounded_identity(config, handoff, topic):
        raise ValueError(f"multi-stage topic identity differs for {topic.topic_id}")
    _bounded_revalidate_files(root, state)
    if not state.get("stages", {}).get("final"):
        raise ValueError(f"multi-stage topic is incomplete: {topic.topic_id}")

    final_path = root / "evaluation/final/submission.jsonl"
    try:
        lines = [line for line in final_path.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != 1:
            raise ValueError("final artifact must contain exactly one row")
        record = json.loads(lines[0])
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read multi-stage final for {topic.topic_id}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"multi-stage final must be an object: {topic.topic_id}")
    _validate_generated_submission_record(
        record,
        topic_id=topic.topic_id,
        narrative=topic.narrative,
        allowed_docids=list(topic.citation_docids),
        team_id=config.team_id,
        run_id=f"{config.run_id}-final",
        run_desc=config.run_desc,
    )
    _validate_exact_hint_citations(record, topic=topic)
    return record


def _prepare_multistage_state(
    config: RagGenerationConfig,
    identity: dict[str, Any],
) -> None:
    if config.overwrite:
        raise ValueError("multi-stage generation does not support experiment.mode: overwrite")

    identity_path = config.resolved_work_dir / _IDENTITY_FILENAME
    if config.resume:
        try:
            recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read multi-stage generation identity") from exc
        if recorded != identity:
            raise ValueError("multi-stage resume identity differs from create identity")
        return

    work_dir = config.resolved_work_dir
    if work_dir.is_symlink() or (work_dir.exists() and not work_dir.is_dir()):
        raise ValueError("multi-stage generation work path is not a directory")
    has_artifacts = work_dir.is_dir() and any(
        path.name != _LOCK_FILENAME for path in work_dir.iterdir()
    )
    if config.output_path.exists() or has_artifacts:
        raise ValueError("multi-stage generation artifacts already exist")
    work_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        identity_path,
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


async def _run_multistage_locked(
    config: RagGenerationConfig,
    handoff: GenerationHandoff,
    topics: Sequence[GenerationTopic],
    *,
    api_key: str,
    topic_runner: TopicRunner,
) -> None:
    identity = _multistage_identity(config, handoff, topics)
    _prepare_multistage_state(config, identity)

    semaphore = asyncio.Semaphore(config.concurrency)

    async def run_topic(topic: GenerationTopic) -> Path:
        root = _bounded_private_root(config, topic)
        state_mode = "resume" if root.exists() or root.is_symlink() else "create"
        async with semaphore:
            return await topic_runner(
                config,
                handoff,
                topic,
                api_key=api_key,
                state_mode=state_mode,
            )

    results = await asyncio.gather(
        *(run_topic(topic) for topic in topics),
        return_exceptions=True,
    )
    failures: list[tuple[str, BaseException]] = [
        (topic.topic_id, result)
        for topic, result in zip(topics, results, strict=True)
        if isinstance(result, BaseException)
    ]
    warnings: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    if not failures:
        for topic, result in zip(topics, results, strict=True):
            if not isinstance(result, Path):
                failures.append(
                    (
                        topic.topic_id,
                        TypeError("multi-stage topic runner returned an invalid result"),
                    )
                )
                continue
            try:
                record = _load_final_record(
                    result,
                    config=config,
                    handoff=handoff,
                    topic=topic,
                )
                records.append(record)
                state = _bounded_read_state(result)
                operation_screen = state.get("operation_screen")
                if (
                    isinstance(operation_screen, dict)
                    and operation_screen.get("crash_fallback") is True
                ):
                    warnings.append(
                        {
                            "topic_id": topic.topic_id,
                            "kind": "operation_screen_crash_fallback",
                            "message": _CRASH_FALLBACK_WARNING,
                        }
                    )
            except Exception as exc:
                failures.append((topic.topic_id, exc))
    failure_report = {
        "failure_count": len(failures),
        "topic_count": len(topics),
        "warning_count": len(warnings),
        "failures": [
            {
                "topic_id": topic_id,
                "exception_type": type(exc).__name__,
                "message": _redact(str(exc), (api_key,)),
            }
            for topic_id, exc in failures
        ],
        "warnings": warnings,
    }
    _atomic_write_text(
        config.resolved_work_dir / _FAILURES_FILENAME,
        json.dumps(failure_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    for warning in warnings:
        print(
            f"topic {warning['topic_id']}: warning: {warning['message']}",
            file=sys.stderr,
        )
    if failures:
        for failure in failure_report["failures"]:
            print(
                f"topic {failure['topic_id']}: {failure['exception_type']}: "
                f"{failure['message']}",
                file=sys.stderr,
            )
        raise RuntimeError(
            "multi-stage topic execution failed for "
            f"{len(failures)} of {len(topics)} topics"
        ) from failures[0][1]
    _atomic_write_text(
        config.output_path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
    )


async def run_multistage_generation(
    config: RagGenerationConfig,
    handoff: GenerationHandoff,
    *,
    api_key: str,
    topic_runner: TopicRunner = _run_bounded_revision,
) -> None:
    """Run missing frozen multi-stage topics and publish one ordered JSONL."""

    _validate_artifact_paths(config)
    topics = select_generation_topics(handoff, config.topic_ids)
    lock_path = config.resolved_work_dir / _LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lock_path, timeout=0):
            await _run_multistage_locked(
                config,
                handoff,
                topics,
                api_key=api_key,
                topic_runner=topic_runner,
            )
    except FileLockTimeout as exc:
        raise RuntimeError(
            f"multi-stage generation is already active for {config.output_path}"
        ) from exc


def _print_dry_run(
    config: RagGenerationConfig,
    topics: Sequence[GenerationTopic],
) -> None:
    topic_count = len(topics)
    group_count = sum(len(topic.groups) for topic in topics)
    print(f"topics={topic_count},groups={group_count}")
    print(
        "calls="
        f"sol_routine:{2 * topic_count},"
        f"sol_max:{3 * topic_count},"
        f"luna_min:{topic_count + group_count},"
        f"luna_max:{2 * topic_count + group_count},"
        "provider:0"
    )
    print(f"concurrency={config.concurrency}")
    print(f"handoff={config.handoff_manifest_path}")
    print(f"output={config.output_path}")
    print(f"work={config.resolved_work_dir}")


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Competition RAG YAML.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Authenticate inputs and report the call budget without writing state.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = arguments(argv)
        config_path = args.config.resolve()
        config = load_rag_generation_config(config_path)
        handoff = load_generation_handoff(config.handoff_manifest_path)
        topics = select_generation_topics(handoff, config.topic_ids)
        _validate_artifact_paths(config)
        if args.dry_run:
            _print_dry_run(config, topics)
            return
        if config.overwrite:
            raise ValueError(
                "multi-stage generation does not support experiment.mode: overwrite"
            )
        load_repo_env(find_repo_root(config_path.parent))
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise ValueError(f"{config.api_key_env} is missing or empty")
        asyncio.run(run_multistage_generation(config, handoff, api_key=api_key))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        raise SystemExit(f"error: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
