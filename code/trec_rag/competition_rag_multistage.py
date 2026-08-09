"""Full-run orchestration for the frozen multi-stage competition RAG path."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
import json
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout as FileLockTimeout

from trec_rag.competition_rag import (
    RagGenerationConfig,
    _atomic_write_text,
    _validate_artifact_paths,
    _validate_exact_hint_citations,
    _validate_generated_submission_record,
    _work_has_artifacts,
)
from trec_rag.generation_handoff import (
    GenerationHandoff,
    GenerationTopic,
    select_generation_topics,
)
from trec_rag.narrative_blueprint_trial import (
    TRIAL_CONTRACT_VERSION,
    _bounded_identity,
    _bounded_private_root,
    _bounded_read_state,
    _bounded_revalidate_files,
    _run_bounded_revision,
)


MULTISTAGE_IDENTITY_VERSION = 1
_IDENTITY_FILENAME = "multistage_generation_identity.json"
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

    if config.output_path.exists() or _work_has_artifacts(config.resolved_work_dir):
        raise ValueError("multi-stage generation artifacts already exist")
    config.resolved_work_dir.mkdir(parents=True, exist_ok=True)
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
    failure_count = sum(isinstance(result, BaseException) for result in results)
    if failure_count:
        raise RuntimeError(
            "multi-stage topic execution failed for "
            f"{failure_count} of {len(topics)} topics"
        )
    roots = [result for result in results if isinstance(result, Path)]
    if len(roots) != len(topics):
        raise RuntimeError("multi-stage topic runner returned an invalid result")
    records = [
        _load_final_record(
            root,
            config=config,
            handoff=handoff,
            topic=topic,
        )
        for topic, root in zip(topics, roots, strict=True)
    ]
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
    lock_path = config.output_path.with_name(f".{config.output_path.name}.lock")
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
