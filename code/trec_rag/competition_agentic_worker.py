"""Assigned-only agentic retrieval worker.

The coordinator owns cohort selection and aggregate export.  This entry point
accepts only the coordinator's authenticated plan bytes and a disjoint topic
subset, then seals successful topics locally without touching run-level files.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from .agentic_run_state import (
    AgenticRunStateError,
    install_run_plan,
    resume_run_plan,
    select_run_topics,
)
from .competition_agentic_retrieval import (
    AgenticRepositoryBinding,
    AgenticRunnerError,
    AgenticTopicOutcome,
    AgenticWorkerError,
    TopicExecutionRequest,
    TopicExecutionResult,
    _build_production_topic_executor,
    _check_secret_presence,
    _execute_agentic_topic_selection,
    _load_environment,
    _planned_topics,
    _probe_repository,
    _read_stable,
    _require_unchanged,
)


@dataclass(frozen=True)
class AgenticWorkerReceipt:
    """Machine-readable outcomes for one assigned worker invocation."""

    run_id: str
    output_dir: Path
    plan_sha256: str
    planned_topic_ids: tuple[str, ...]
    assigned_topic_ids: tuple[str, ...]
    outcomes: tuple[AgenticTopicOutcome, ...]

    @property
    def complete(self) -> bool:
        return all(outcome.status in {"complete", "skipped"} for outcome in self.outcomes)

    def to_payload(self) -> dict[str, object]:
        return {
            "status": "complete" if self.complete else "incomplete",
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "plan_sha256": self.plan_sha256,
            "planned_topic_ids": list(self.planned_topic_ids),
            "assigned_topic_ids": list(self.assigned_topic_ids),
            "outcomes": [outcome.to_payload() for outcome in self.outcomes],
        }


def _worker_error(code: str, message: str) -> AgenticWorkerError:
    return AgenticWorkerError(code, message)


def _run_agentic_worker(
    config: str | Path,
    *,
    plan_body: bytes,
    assigned_topic_ids: Sequence[str],
    environment_loader: Callable[[Path], None],
    environ: Mapping[str, str] | None,
    repository_probe: Callable[[Path], AgenticRepositoryBinding],
    topic_executor_factory: Callable[[Any], Callable[[TopicExecutionRequest], TopicExecutionResult]],
) -> AgenticWorkerReceipt:
    """Run only assigned topics against an authenticated coordinator plan."""

    if not isinstance(plan_body, bytes) or not plan_body:
        raise _worker_error("plan_invalid", "worker plan bytes are required")
    if isinstance(assigned_topic_ids, (str, bytes)):
        raise TypeError("assigned_topic_ids must be a sequence of topic IDs")
    assigned = tuple(assigned_topic_ids)
    if not assigned or any(not isinstance(topic_id, str) for topic_id in assigned):
        raise _worker_error("assignment_invalid", "assigned topic subset is required")
    if len(set(assigned)) != len(assigned):
        raise _worker_error("assignment_invalid", "assigned topic subset repeats an identity")

    config_path = Path(config).resolve()
    config_bytes = _read_stable(config_path, label="agentic config")
    try:
        from .agentic_retrieval_config import (
            load_agentic_retrieval_config,
            select_agentic_topics,
        )

        loaded = load_agentic_retrieval_config(config_path, source_bytes=config_bytes)
        topic_source_bytes = _read_stable(
            loaded.topics_path, label="official topic source"
        )
        official_topics = tuple(select_agentic_topics(loaded))
    except AgenticRunnerError:
        raise
    except Exception as exc:
        raise _worker_error(
            "config_invalid", "agentic configuration or topic source is invalid"
        ) from exc
    if not official_topics:
        raise _worker_error("topic_selection_empty", "worker topic source is empty")

    work_dir = loaded.output_dir / "work"
    try:
        # Installation is create-only/identical-only and authenticates the
        # exact bytes before any topic state is inspected.
        installed = install_run_plan(work_dir=work_dir, body=plan_body)
        installed_body = (work_dir / "run_plan.json").read_bytes()
        if installed_body != plan_body:
            raise _worker_error("plan_invalid", "installed plan bytes differ")
        binding = repository_probe(loaded.root_dir)
        if not isinstance(binding, AgenticRepositoryBinding):
            raise TypeError("repository_probe must return AgenticRepositoryBinding")
        cohort = _planned_topics(installed, official_topics)
        plan = resume_run_plan(
            work_dir=work_dir,
            run_id=loaded.run_id,
            config_bytes=config_bytes,
            topics=cohort,
            official_topics_sha256=sha256_bytes(topic_source_bytes),
            source_revision=binding.source_revision,
            submodule_revisions=binding.submodule_revisions,
        )
        if any(topic_id not in plan.planned_topic_ids for topic_id in assigned):
            raise _worker_error(
                "assignment_invalid",
                "assigned topic subset contains a topic outside the authenticated plan",
            )
        _require_unchanged(config_path, config_bytes, label="agentic config")
        _require_unchanged(
            loaded.topics_path,
            topic_source_bytes,
            label="official topic source",
        )
    except AgenticWorkerError:
        raise
    except AgenticRunStateError as exc:
        raise _worker_error("plan_invalid", str(exc)) from exc
    except AgenticRunnerError as exc:
        raise _worker_error("plan_invalid", str(exc)) from exc

    environment_loader(loaded.root_dir)
    runtime_environ = os.environ if environ is None else environ
    _check_secret_presence(runtime_environ)
    os.environ["HF_HOME"] = str(loaded.caches.model_cache_dir)
    try:
        selection = select_run_topics(
            work_dir=work_dir,
            plan=plan,
            topic_ids=assigned,
            validate_all=False,
        )
        _executed, all_outcomes = _execute_agentic_topic_selection(
            loaded=loaded,
            work_dir=work_dir,
            plan=plan,
            cohort=cohort,
            selection=selection,
            official_topics_sha256=sha256_bytes(topic_source_bytes),
            topic_executor_factory=topic_executor_factory,
        )
    except AgenticRunStateError as exc:
        raise _worker_error("run_state_invalid", str(exc)) from exc
    outcome_by_id = {outcome.topic_id: outcome for outcome in all_outcomes}
    outcomes = tuple(outcome_by_id[topic_id] for topic_id in assigned if topic_id in outcome_by_id)
    if {outcome.topic_id for outcome in outcomes} != set(assigned):
        raise _worker_error("worker_incomplete", "worker did not return every assigned topic")
    return AgenticWorkerReceipt(
        run_id=plan.run_id,
        output_dir=loaded.output_dir,
        plan_sha256=plan.plan_sha256,
        planned_topic_ids=plan.planned_topic_ids,
        assigned_topic_ids=assigned,
        outcomes=outcomes,
    )


def sha256_bytes(body: bytes) -> str:
    from hashlib import sha256

    return sha256(body).hexdigest()


def run_agentic_worker(
    config: str | Path,
    *,
    plan_body: bytes,
    assigned_topic_ids: Sequence[str],
) -> AgenticWorkerReceipt:
    """Production worker entry point using the pinned plan and live adapters."""

    return _run_agentic_worker(
        config,
        plan_body=plan_body,
        assigned_topic_ids=assigned_topic_ids,
        environment_loader=_load_environment,
        environ=None,
        repository_probe=_probe_repository,
        topic_executor_factory=_build_production_topic_executor,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute only assigned topics from an authenticated agentic run plan."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--topic", action="append", dest="topic_ids", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run_agentic_worker(
            args.config,
            plan_body=args.plan.read_bytes(),
            assigned_topic_ids=tuple(args.topic_ids),
        )
    except AgenticRunnerError as exc:
        print(
            json.dumps(
                {"status": "error", "code": exc.code, "message": str(exc)},
                sort_keys=True,
            ),
            file=os.sys.stderr,
        )
        return 2
    print(json.dumps(receipt.to_payload(), sort_keys=True))
    return 0 if receipt.complete else 2


__all__ = [
    "AgenticWorkerError",
    "AgenticWorkerReceipt",
    "main",
    "run_agentic_worker",
]


if __name__ == "__main__":
    raise SystemExit(main())
