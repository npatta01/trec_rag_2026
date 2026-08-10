"""Build the canonical private evaluation bundle for a completed retrieval + RAG run.

Everything is resolved from the two configs and the authenticated manifests they name:
the retrieval export manifest, the generation handoff, and the generation identity. No
experiment name, output directory, topic id, topic count, citation count, or expected
label is assumed here.

The bundle's ``evaluation_manifest.json`` is the only thing the friendly renderer reads.
It binds the validated sources, the derived artifacts, the judgments, and the metrics
together so the renderer never has to guess which loose files belong to one another.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

from trec_rag.accepted_rag_evaluation import (
    AcceptedRunBinding,
    build_accepted_run_binding,
    write_accepted_run_binding,
)
from trec_rag.competition_debug_report import (
    DebugReportData,
    RagArtifactSource,
    load_debug_report_data,
)
from trec_rag.competition_rag import ANSWER_WORD_LIMIT, load_rag_generation_config
from trec_rag.judge_cache import JudgeCache, JudgeCacheConflict, JudgeIdentity, canonical_bytes
from trec_rag.ragdoll_io import (
    SUPPORT_LABELS,
    selected_evidence_support_rows,
    validate_support_judgments,
)


BUNDLE_SCHEMA_VERSION = "trec_rag_offline_evaluation_bundle_v1"
TASK_SCHEMA_VERSION = "ragdoll_support_task_v1"

# Manifest fields that are receipts of one invocation rather than inputs to the report.
# The renderer must never read these, so independently built bundles render identically.
VOLATILE_RECEIPT_FIELDS = ("created_utc", "environment", "artifacts", "volatile_receipt_fields")

SUPPORT_METRIC_PAIRS = (
    ("weighted_precision_first_citation", "weighted_recall_first_citation"),
    ("weighted_precision_all_judged_citations", "weighted_recall_all_judged_citations"),
    ("hard_precision", "hard_recall"),
)

# These restate the pinned ``ragdoll.support.metrics.support_metric`` formula. Precision and
# recall differ only in their denominator: precision counts answer objects that actually
# carry a judged citation, while recall starts from *every* answer object in the topic and
# removes only those whose citations are unjudged (support ``-1``). An answer object with no
# citations at all is therefore never removed, so it stays in the recall denominator and
# lowers recall while leaving precision untouched.
SUPPORT_METRIC_DEFINITIONS = {
    "weighted_precision_first_citation": (
        "Mean support weight (FS=1, PS=0.5, NS=0) of each answer object's first citation, "
        "divided by the number of answer objects whose first citation is judged."
    ),
    "weighted_recall_first_citation": (
        "The same weighted first-citation sum, divided instead by every answer object in the "
        "topic except those whose first citation is unjudged. Answer objects with no "
        "citations stay in this denominator, so they reduce recall but not precision."
    ),
    "weighted_precision_all_judged_citations": (
        "Per answer object, the mean support weight across all of its judged citations, "
        "divided by the number of answer objects that have at least one judged citation."
    ),
    "weighted_recall_all_judged_citations": (
        "The same all-citation sum, divided instead by every answer object in the topic "
        "except those that have citations but none judged. Answer objects with no citations "
        "stay in this denominator."
    ),
    "hard_precision": (
        "First-citation variant crediting only full support (FS=1, PS=0, NS=0), divided by "
        "the number of answer objects whose first citation is judged."
    ),
    "hard_recall": (
        "The same hard first-citation sum, divided instead by every answer object in the "
        "topic except those whose first citation is unjudged."
    ),
}

RETRIEVAL_METRIC_NAMES = ("ndcg@10", "ndcg@100", "recall@100", "recall@1000", "judged_rate@100")
RETRIEVAL_RELEVANCE_THRESHOLD = 1
RETRIEVAL_METRIC_DEFINITIONS = {
    "ndcg@10": "Normalized discounted cumulative gain over the top 10 submitted documents.",
    "ndcg@100": "Normalized discounted cumulative gain over the top 100 submitted documents.",
    "recall@100": "Fraction of qrels-relevant documents present in the top 100 submitted documents.",
    "recall@1000": "Fraction of qrels-relevant documents present in the top 1000 submitted documents.",
    "judged_rate@100": "Fraction of the top 100 submitted documents that carry any qrels judgment.",
}


class EvaluationError(ValueError):
    """Raised when the evaluation inputs cannot be bound into one validated bundle."""


@dataclass(frozen=True)
class JudgeSettings:
    """Output-affecting judge settings; provenance-only values do not belong here.

    Settings deliberately excluded from the cache identity are the ones that cannot change
    the judge's answer: ``timeout_seconds`` (turns a call into a failure, and failures are
    never cached), ``cache_dir`` (this cache replaces it), and ``agent_state_dir`` (copied
    into a throwaway per-call directory by RAGDoll's runner).
    """

    provider: str
    model: str
    thinking: str
    temperature: float | None
    system_prompt: str
    agent_binary: str
    extension_identity: str = ""


@dataclass(frozen=True)
class JudgeOutcome:
    """One hosted judge result. Only ``completed`` rows may ever be cached."""

    status: str
    support_label: str | None = None
    error: str | None = None


JudgeCallable = Callable[[Mapping[str, Any]], JudgeOutcome]


@dataclass(frozen=True)
class EvaluationBundle:
    work_dir: Path
    manifest_path: Path
    manifest: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Pinned RAGDoll identity
# ---------------------------------------------------------------------------


def ragdoll_identity(repository_root: Path) -> dict[str, str]:
    """Record the pinned RAGDoll project version, commit, and prompt contract."""
    import ragdoll
    from ragdoll.support import prompts

    submodule = repository_root / "ragdoll"
    commit = _git_commit(submodule)
    version = getattr(ragdoll, "__version__", None)
    if not isinstance(version, str) or not version:
        version = _project_version(submodule / "pyproject.toml")
    return {
        "project_version": version,
        "commit": commit,
        "prompt_contract_sha256": sha256(
            prompts.SUPPORT_EVAL_PROMPT.encode("utf-8")
        ).hexdigest(),
        "task_schema_version": TASK_SCHEMA_VERSION,
    }


def _git_commit(directory: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        raise EvaluationError(f"{directory}: cannot resolve the pinned RAGDoll commit") from None
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise EvaluationError(f"{directory}: pinned RAGDoll commit is not a full sha")
    return commit


def _project_version(pyproject: Path) -> str:
    import tomllib

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        raise EvaluationError(f"{pyproject}: cannot resolve the RAGDoll project version") from None
    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise EvaluationError(f"{pyproject}: RAGDoll project version is missing")
    return version


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------


def build_evaluation_bundle(
    *,
    retrieval_config_path: Path,
    work_dir: Path,
    repository_root: Path,
    cache_root: Path,
    rag_config_path: Path | None = None,
    accepted_submission_path: Path | None = None,
    accepted_bundle_metadata_path: Path | None = None,
    handoff_manifest_path: Path | None = None,
    source_identity_path: Path | None = None,
    topic_ids: Sequence[str] | None = None,
    qrels_path: Path | None = None,
    gold_nuggets_path: Path | None = None,
    judge: JudgeCallable | None = None,
    judge_settings: JudgeSettings,
    judge_limit: int | None = None,
    judge_workers: int = 1,
    created_utc: str | None = None,
) -> EvaluationBundle:
    """Validate every input, resolve or run judgments, and seal the bundle manifest."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    _require_private_directory(work_dir)
    if type(judge_workers) is not int or judge_workers < 1:
        raise EvaluationError("judge_workers must be a positive integer")

    accepted_paths = (
        accepted_submission_path,
        accepted_bundle_metadata_path,
        handoff_manifest_path,
    )
    accepted_mode = any(path is not None for path in accepted_paths)
    if rag_config_path is not None and accepted_mode:
        raise EvaluationError("RAG config and accepted RAG artifacts are mutually exclusive")
    if rag_config_path is None and not all(path is not None for path in accepted_paths):
        raise EvaluationError(
            "provide either rag_config_path or accepted submission, bundle metadata, and handoff"
        )
    if rag_config_path is not None and source_identity_path is not None:
        raise EvaluationError("source identity is only valid for accepted RAG artifacts")

    rag_config = None
    identity_path: Path | None = None
    accepted_binding: AcceptedRunBinding | None = None
    accepted_binding_path: Path | None = None
    if rag_config_path is not None:
        rag_config = load_rag_generation_config(Path(rag_config_path))
        identity_path = rag_config.work_dir / "generation_identity.json"
        if not identity_path.is_file():
            raise EvaluationError(f"{identity_path}: generation identity is missing")
        from trec_rag.generation_handoff import load_generation_handoff, select_generation_topics

        handoff = load_generation_handoff(rag_config.handoff_manifest_path)
        selected_topics = select_generation_topics(handoff, rag_config.topic_ids)
        rag_source = RagArtifactSource(
            handoff_manifest_path=rag_config.handoff_manifest_path,
            output_path=rag_config.output_path,
            topic_ids=tuple(topic.topic_id for topic in selected_topics),
            team_id=rag_config.team_id,
            run_id=rag_config.run_id,
            run_desc=rag_config.run_desc,
            provider=rag_config.provider,
            model=rag_config.model,
        )
    else:
        assert accepted_submission_path is not None
        assert accepted_bundle_metadata_path is not None
        assert handoff_manifest_path is not None
        accepted_binding = build_accepted_run_binding(
            Path(accepted_submission_path),
            Path(accepted_bundle_metadata_path),
            Path(handoff_manifest_path),
            source_identity_path=(
                None if source_identity_path is None else Path(source_identity_path)
            ),
        )
        accepted_binding_path = write_accepted_run_binding(
            accepted_binding, work_dir / "accepted_run_binding.json"
        )
        rag_source = RagArtifactSource(
            handoff_manifest_path=Path(handoff_manifest_path).resolve(),
            output_path=Path(accepted_submission_path).resolve(),
            topic_ids=accepted_binding.topic_ids,
            team_id=accepted_binding.team_id,
            run_id=accepted_binding.run_id,
            run_desc=accepted_binding.run_desc,
            provider=accepted_binding.provider,
            model=", ".join(accepted_binding.models),
            accepted_submission_sha256=accepted_binding.submission_sha256,
            submission_snapshot=accepted_binding.submission_snapshot,
            output_snapshot=accepted_binding.submission_snapshot,
            handoff_snapshot=accepted_binding.handoff_snapshot,
            handoff=accepted_binding.handoff,
        )

    data = load_debug_report_data(
        Path(retrieval_config_path),
        rag_config_path=(None if rag_config_path is None else Path(rag_config_path)),
        rag_artifact_source=(None if rag_config_path is not None else rag_source),
        topic_ids=list(topic_ids) if topic_ids is not None else None,
    )
    if not data.topics:
        raise EvaluationError("the selected scope contains no topics")
    if any(topic.rag_output is None for topic in data.topics):
        raise EvaluationError("every selected topic must have a completed RAG output")

    ordered_topic_ids = tuple(topic.topic_id for topic in data.topics)
    support_rows = _selected_support_rows(
        rag_source,
        identity_path if identity_path is not None else accepted_binding_path,
        ordered_topic_ids,
    )

    ragdoll = ragdoll_identity(Path(repository_root))
    tasks = _support_tasks(support_rows, ordered_topic_ids)

    cache = JudgeCache(Path(cache_root))
    judgments, judge_report = _resolve_judgments(
        tasks,
        cache=cache,
        judge=judge,
        settings=judge_settings,
        ragdoll=ragdoll,
        judge_limit=judge_limit,
        judge_workers=judge_workers,
    )

    derived: dict[str, Path] = {}
    derived["support_input.jsonl"] = _write_jsonl(work_dir / "support_input.jsonl", support_rows)
    derived["support_tasks.jsonl"] = _write_jsonl(work_dir / "support_tasks.jsonl", tasks)
    derived["support_judgments.jsonl"] = _write_jsonl(work_dir / "support_judgments.jsonl", judgments)

    fully_judged = judge_report["missing"] == 0
    if fully_judged:
        validated = validate_support_judgments(
            derived["support_input.jsonl"], derived["support_judgments.jsonl"]
        )
        if validated != len(tasks):
            raise EvaluationError("validated judgment count disagrees with the materialized tasks")

    assignments = _assignments(derived["support_input.jsonl"], derived["support_judgments.jsonl"])
    derived["support_assignments.jsonl"] = _write_jsonl(
        work_dir / "support_assignments.jsonl", assignments
    )

    support = _support_metrics(assignments, tasks, judgments, ordered_topic_ids)
    derived["support_metrics.jsonl"] = _write_jsonl(
        work_dir / "support_metrics.jsonl", list(support["per_topic"].values())
    )

    retrieval = _retrieval_metrics(data, qrels_path)
    nuggets = _nugget_availability(
        gold_nuggets_path,
        ordered_topic_ids,
        {topic.topic_id: topic.narrative for topic in data.topics},
    )

    # All accepted sources were captured before any phase began.  Reject a path replacement
    # before sealing, even though every derived object above was built from the snapshots.
    if accepted_binding is not None:
        accepted_binding.verify_sources()

    manifest = _manifest(
        data=data,
        rag_config=rag_config,
        rag_config_path=(None if rag_config_path is None else Path(rag_config_path)),
        retrieval_config_path=Path(retrieval_config_path),
        identity_path=identity_path,
        rag_artifact_source=rag_source,
        accepted_binding=accepted_binding,
        accepted_binding_path=accepted_binding_path,
        ordered_topic_ids=ordered_topic_ids,
        requested_topic_ids=topic_ids,
        ragdoll=ragdoll,
        judge_settings=judge_settings,
        judge_report=judge_report,
        judge_limit=judge_limit,
        judge_workers=judge_workers,
        cache=cache,
        derived=derived,
        work_dir=work_dir,
        support=support,
        retrieval=retrieval,
        nuggets=nuggets,
        tasks=tasks,
        judgments=judgments,
        qrels_path=qrels_path,
        gold_nuggets_path=gold_nuggets_path,
        created_utc=created_utc,
    )
    if accepted_binding is not None:
        accepted_binding.verify_sources()
    manifest_path = work_dir / "evaluation_manifest.json"
    _atomic_write(manifest_path, canonical_bytes(manifest) + b"\n")
    return EvaluationBundle(work_dir=work_dir, manifest_path=manifest_path, manifest=manifest)


def _require_private_directory(path: Path) -> None:
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        path.chmod(0o700)


def _selected_support_rows(
    rag_source: RagArtifactSource,
    evidence_binding_path: Path | None,
    ordered_topic_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if evidence_binding_path is None:
        raise EvaluationError("RAG evidence binding is missing")
    rows = selected_evidence_support_rows(
        rag_source.output_path,
        rag_source.handoff_manifest_path,
        evidence_binding_path,
        submission_snapshot=rag_source.submission_snapshot,
        handoff_snapshot=rag_source.handoff_snapshot,
        handoff=rag_source.handoff,
    )
    by_topic: dict[str, dict[str, Any]] = {}
    for row in rows:
        topic_id = str(row["topic_id"])
        if topic_id in by_topic:
            raise EvaluationError(f"duplicate submission row for topic {topic_id}")
        by_topic[topic_id] = row
    missing = [topic_id for topic_id in ordered_topic_ids if topic_id not in by_topic]
    if missing:
        raise EvaluationError(f"submission has no row for topic(s) {', '.join(missing)}")
    return [by_topic[topic_id] for topic_id in ordered_topic_ids]


def _support_tasks(
    support_rows: Sequence[Mapping[str, Any]], ordered_topic_ids: Sequence[str]
) -> list[dict[str, Any]]:
    from ragdoll.support.stages import iter_support_tasks

    tasks = iter_support_tasks([dict(row) for row in support_rows])
    if not tasks:
        raise EvaluationError("the selected scope produced no citation-support tasks")
    seen: set[str] = set()
    allowed = set(ordered_topic_ids)
    for task in tasks:
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise EvaluationError("a materialized support task has no task id")
        if task_id in seen:
            raise EvaluationError(f"duplicate support task {task_id}")
        seen.add(task_id)
        source = task.get("metadata", {}).get("source", {})
        if str(source.get("topic_id")) not in allowed:
            raise EvaluationError(f"support task {task_id} is outside the selected scope")
    return tasks


def _judge_identity(
    task: Mapping[str, Any], *, settings: JudgeSettings, ragdoll: Mapping[str, str]
) -> JudgeIdentity:
    instruction = task.get("instruction")
    evaluator = task.get("evaluator")
    if not isinstance(instruction, str) or not instruction:
        raise EvaluationError(f"support task {task.get('task_id')!r} has no instruction")
    if not isinstance(evaluator, str) or not evaluator:
        raise EvaluationError(f"support task {task.get('task_id')!r} has no evaluator")
    return JudgeIdentity(
        evaluator=evaluator,
        instruction=instruction,
        ragdoll_version=ragdoll["project_version"],
        ragdoll_commit=ragdoll["commit"],
        prompt_contract_sha256=ragdoll["prompt_contract_sha256"],
        task_schema_version=ragdoll["task_schema_version"],
        provider=settings.provider,
        model=settings.model,
        thinking=settings.thinking,
        temperature=settings.temperature,
        system_prompt=settings.system_prompt,
        agent_binary=settings.agent_binary,
        extension_identity=settings.extension_identity,
    )


def _resolve_judgments(
    tasks: Sequence[Mapping[str, Any]],
    *,
    cache: JudgeCache,
    judge: JudgeCallable | None,
    settings: JudgeSettings,
    ragdoll: Mapping[str, str],
    judge_limit: int | None = None,
    judge_workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reuse validated cache entries and call the judge only for validated misses.

    ``judge_limit`` caps hosted calls for this invocation, which is what makes the
    documented probe-then-resume workflow possible: probe with a limit of 1, confirm the
    single judgment completed, then rerun without the limit so the probe is a cache hit and
    only the remaining misses call the judge. A failed call still consumes the limit, so a
    broken probe stops instead of burning the rest of the quota.
    """
    if type(judge_workers) is not int or judge_workers < 1:
        raise EvaluationError("judge_workers must be a positive integer")
    if judge_limit is not None and (type(judge_limit) is not int or judge_limit < 1):
        raise EvaluationError("judge_limit must be a positive integer")

    # Resolve every cache hit on the controller first. A miss is selected at most once by
    # identity, so repeated task payloads cannot consume multiple probe calls. Nothing is
    # submitted until this pass completes, which makes --judge-limit a strict bound even
    # when a worker pool is wider than the probe quota.
    identities: list[JudgeIdentity] = []
    labels: list[str | None] = []
    label_sources: list[str] = []
    reused = 0
    skipped_by_limit = 0
    selected: list[tuple[int, Mapping[str, Any], JudgeIdentity]] = []
    selected_by_digest: dict[str, int] = {}
    for index, task in enumerate(tasks):
        identity = _judge_identity(task, settings=settings, ragdoll=ragdoll)
        identities.append(identity)
        entry = cache.read(identity)
        if entry is not None:
            labels.append(entry.support_label)
            label_sources.append("cache")
            reused += 1
            continue
        labels.append(None)
        label_sources.append("missing")
        if judge is None:
            continue
        digest = identity.digest
        if digest in selected_by_digest:
            # The selected call will populate all tasks sharing this identity.
            continue
        if judge_limit is not None and len(selected) >= judge_limit:
            skipped_by_limit += 1
            continue
        selected_by_digest[digest] = index
        selected.append((index, task, identity))

    # Hosted calls may run concurrently, but every completion is consumed and checkpointed on
    # this controller thread as soon as it is available. Final judgments are still assembled
    # by declared task index below, so completion timing cannot reorder support_judgments.jsonl.
    if selected and judge is not None:
        hosted_calls = len(selected)
    else:
        hosted_calls = 0
    failed = 0
    conflicts = 0
    outcome_by_digest: dict[str, tuple[JudgeOutcome | None, str | None, str]] = {}

    def checkpoint(
        identity: JudgeIdentity, outcome: JudgeOutcome | None
    ) -> None:
        """Validate and persist one worker outcome on the controller thread."""
        nonlocal failed, conflicts
        label: str | None = None
        source = "missing"
        if outcome is not None and (
            outcome.status == "completed" and outcome.support_label in SUPPORT_LABELS
        ):
            label = outcome.support_label
            try:
                cache.put(identity, support_label=label)
                source = "judge"
            except JudgeCacheConflict:
                existing = cache.read(identity)
                if existing is None:
                    failed += 1
                    label = None
                else:
                    label = existing.support_label
                    source = "cache_conflict"
                    conflicts += 1
        else:
            failed += 1
        outcome_by_digest[identity.digest] = (outcome, label, source)

    if selected and judge is not None:
        with ThreadPoolExecutor(max_workers=judge_workers) as executor:
            future_to_selected = {
                executor.submit(judge, task): (index, task, identity)
                for index, task, identity in selected
            }
            for future in as_completed(future_to_selected):
                _, _, identity = future_to_selected[future]
                try:
                    outcome = future.result()
                except Exception:
                    outcome = None
                checkpoint(identity, outcome)

    # Apply each completed outcome to all same-identity task rows. This is what lets a probe
    # report progress for duplicate statement/citation tasks while still making one hosted call.
    for index, identity in enumerate(identities):
        if labels[index] is not None:
            continue
        selected_outcome = outcome_by_digest.get(identity.digest)
        if selected_outcome is None:
            continue
        _, label, source = selected_outcome
        labels[index] = label
        label_sources[index] = source

    judgments: list[dict[str, Any]] = []
    missing = 0
    for index, task in enumerate(tasks):
        label = labels[index]
        if label is None:
            missing += 1
            continue
        metadata = dict(task.get("metadata", {}))
        judgments.append(
            {
                "task_id": task["task_id"],
                "status": "completed",
                "support_label": label,
                "statement": metadata.get("statement"),
                "citation": metadata.get("citation"),
                "metadata": dict(metadata.get("source", {})),
                "cache_entry_sha256": identities[index].digest,
                "label_source": label_sources[index],
            }
        )
    report = {
        "tasks": len(tasks),
        "completed": len(judgments),
        "missing": missing,
        "failed": failed,
        "conflicts": conflicts,
        "hosted_calls": hosted_calls,
        "reused_from_cache": reused,
        "skipped_by_judge_limit": skipped_by_limit,
    }
    return judgments, report


def _assignments(support_input: Path, judgments_path: Path) -> list[dict[str, Any]]:
    from ragdoll.support.assignments import assemble_support_assignments

    return assemble_support_assignments(support_input, [judgments_path])


def _support_metrics(
    assignments: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    ordered_topic_ids: Sequence[str],
) -> dict[str, Any]:
    """Compute RAGDoll's own support metrics, per topic, only where fully judged."""
    from ragdoll.support.metrics import metric_row, support_metric

    judged_ids = {str(row["task_id"]) for row in judgments}
    expected_by_topic: dict[str, int] = {topic_id: 0 for topic_id in ordered_topic_ids}
    judged_by_topic: dict[str, int] = {topic_id: 0 for topic_id in ordered_topic_ids}
    labels_by_topic: dict[str, dict[str, int]] = {
        topic_id: {label: 0 for label in sorted(SUPPORT_LABELS)} for topic_id in ordered_topic_ids
    }
    label_by_task = {str(row["task_id"]): str(row["support_label"]) for row in judgments}
    for task in tasks:
        topic_id = str(task["metadata"]["source"]["topic_id"])
        expected_by_topic[topic_id] += 1
        task_id = str(task["task_id"])
        if task_id in judged_ids:
            judged_by_topic[topic_id] += 1
            labels_by_topic[topic_id][label_by_task[task_id]] += 1

    # Per-topic cells stay exactly what RAGDoll publishes (metric_row rounds them). The macro
    # is averaged from the raw SupportMetric floats and rounded once, at publication.
    raw: dict[str, dict[str, float]] = {}
    published: dict[str, dict[str, Any]] = {}
    for row in assignments:
        metric = support_metric(dict(row))
        raw[metric.topic_id] = {
            key: float(getattr(metric, key))
            for pair in SUPPORT_METRIC_PAIRS
            for key in pair
        }
        published[metric.topic_id] = dict(metric_row(metric))

    per_topic: dict[str, dict[str, Any]] = {}
    availability: dict[str, dict[str, Any]] = {}
    for topic_id in ordered_topic_ids:
        expected = expected_by_topic[topic_id]
        judged = judged_by_topic[topic_id]
        if expected and judged == expected and topic_id in published:
            per_topic[topic_id] = published[topic_id]
            availability[topic_id] = {"available": True, "reason": None}
        else:
            availability[topic_id] = {
                "available": False,
                "reason": (
                    f"{expected - judged} of {expected} citation-support judgments are not "
                    "completed for this topic"
                ),
            }

    available_topics = [
        topic_id for topic_id in ordered_topic_ids if availability[topic_id]["available"]
    ]
    complete = bool(available_topics) and len(available_topics) == len(ordered_topic_ids)
    macro = _macro_from_raw(raw, available_topics) if complete else {}
    macro_availability = _macro_availability(
        ordered_topic_ids, available_topics, "completed citation-support judgments"
    )
    if not available_topics:
        macro_availability["reason"] = "no topic has complete citation-support judgments"
    return {
        "per_topic": per_topic,
        "per_topic_availability": availability,
        "macro": macro,
        "macro_availability": macro_availability,
        "expected_tasks": expected_by_topic,
        "completed_tasks": judged_by_topic,
        "label_counts": labels_by_topic,
    }


def _retrieval_metrics(data: DebugReportData, qrels_path: Path | None) -> dict[str, Any]:
    """Score submitted documents only against qrels that match the selected topics.

    The repository evaluator owns the formulas. It is handed ``RankedCandidate`` rows, the
    full qrels mapping, the explicit metric names, the relevance threshold, and the exact
    topic scope, so a topic outside the request can never be scored by accident.
    """
    ordered = [topic.topic_id for topic in data.topics]
    if qrels_path is None:
        return _unavailable_metric_block(ordered, "no qrels file was supplied")

    from trec_rag.evaluation import evaluate_ranked, parse_qrels
    from trec_rag.pipeline_models import RankedCandidate

    try:
        qrels = parse_qrels(Path(qrels_path))
    except (OSError, ValueError) as error:
        raise EvaluationError(f"{qrels_path}: qrels could not be parsed: {error}") from error

    scorable = [
        topic.topic_id for topic in data.topics if qrels.get(topic.topic_id)
    ]
    availability: dict[str, dict[str, Any]] = {
        topic_id: (
            {"available": True, "reason": None}
            if topic_id in scorable
            else {
                "available": False,
                "reason": f"the supplied qrels contain no judgments for topic {topic_id}",
            }
        )
        for topic_id in ordered
    }

    per_topic: dict[str, dict[str, Any]] = {}
    raw: dict[str, dict[str, float]] = {}
    if scorable:
        ranked = [
            RankedCandidate(
                topic_id=topic.topic_id,
                docid=document.docid,
                rank=document.rank,
                score=document.score,
                text="",
                provenance=[],
            )
            for topic in data.topics
            if topic.topic_id in scorable
            for document in topic.retrieval_output.documents
        ]
        scored = evaluate_ranked(
            ranked,
            qrels,
            metric_names=RETRIEVAL_METRIC_NAMES,
            relevance_threshold=RETRIEVAL_RELEVANCE_THRESHOLD,
            topic_ids=scorable,
        )
        per_topic_scores = scored.get("per_topic", scored)
        for topic_id in ordered:
            if topic_id not in scorable:
                continue
            values = per_topic_scores.get(topic_id)
            if not isinstance(values, Mapping):
                raise EvaluationError(f"{topic_id}: the evaluator returned no metric row")
            raw[topic_id] = {str(key): float(value) for key, value in values.items()}
            per_topic[topic_id] = {key: round(value, 6) for key, value in raw[topic_id].items()}

    # The macro is computed from unrounded cells and rounded once, at publication.
    complete = len(scorable) == len(ordered) and bool(scorable)
    macro = _macro_from_raw(raw, scorable) if complete else {}
    return {
        "per_topic": per_topic,
        "per_topic_availability": availability,
        "macro": macro,
        "macro_availability": _macro_availability(ordered, scorable, "the supplied qrels"),
    }


def _unavailable_metric_block(ordered: Sequence[str], reason: str) -> dict[str, Any]:
    return {
        "per_topic": {},
        "per_topic_availability": {
            topic_id: {"available": False, "reason": reason} for topic_id in ordered
        },
        "macro": {},
        "macro_availability": {"available": False, "scope_topic_ids": [], "reason": reason},
    }


def _macro_from_raw(
    raw: Mapping[str, Mapping[str, float]], scope: Sequence[str]
) -> dict[str, float]:
    """Average the unrounded cells, then round once for publication."""
    if not scope:
        return {}
    keys = sorted({key for topic_id in scope for key in raw[topic_id]})
    return {
        key: _publish(sum(raw[topic_id][key] for topic_id in scope) / len(scope)) for key in keys
    }


def _publish(value: float) -> float:
    """Round half away from zero at the publication boundary.

    Python's ``round`` is tie-to-even and its float inputs sit a hair below the tie, so
    averaging cells such as 0.75 and 0.840909 published 0.795454 where the pinned RAGDoll
    formatting publishes 0.795455. Decimal rounding removes both effects.
    """
    return float(Decimal(repr(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _macro_availability(
    ordered: Sequence[str], scope: Sequence[str], subject: str
) -> dict[str, Any]:
    if scope and len(scope) == len(ordered):
        return {"available": True, "scope_topic_ids": list(scope), "reason": None}
    reason = (
        f"{subject} match none of the selected topics"
        if not scope
        else (
            f"{subject} match only {len(scope)} of {len(ordered)} selected topics, so this is "
            "not a full-scope macro"
        )
    )
    return {"available": False, "scope_topic_ids": list(scope), "reason": reason}


def _nugget_availability(
    gold_nuggets_path: Path | None,
    ordered_topic_ids: Sequence[str],
    narratives: Mapping[str, str],
) -> dict[str, Any]:
    """Nugget coverage needs released gold *and* completed assignments; never claims.

    Gold alone can never make this available: RAGDoll scores nugget coverage from completed
    assignment rows, and this workflow does not produce them. Supplying gold therefore only
    sharpens the unavailability reason, it never turns into a number.
    """
    binding: dict[str, str] = {}
    if gold_nuggets_path is None:
        reason = "no released gold-nugget file was supplied"
        matched: list[str] = []
    else:
        from trec_rag.ragdoll_io import gold_nugget_rows

        try:
            # No topic_ids filter: the adapter *raises* when a requested topic is absent,
            # which would turn a legitimate partial match into an error. Bind everything the
            # file offers, then intersect with the requested scope here.
            rows = [
                row
                for row in gold_nugget_rows(
                    Path(gold_nuggets_path), narratives=dict(narratives)
                )
                if str(row.get("qid")) in set(ordered_topic_ids)
            ]
        except (ValueError, OSError, KeyError) as error:
            rows = []
            reason = f"the supplied gold nuggets could not be bound to this scope: {error}"
        else:
            reason = ""
        matched = sorted({str(row.get("qid")) for row in rows}) if rows else []
        # Record what each matched row was bound to, so a regression that bound topic ids
        # instead of the authoritative narratives is visible rather than silent.
        binding = {
            str(row["qid"]): sha256(str(row["query"]).encode("utf-8")).hexdigest()
            for row in rows
        }
        if not reason:
            if not matched:
                reason = "the supplied gold nuggets match none of the selected topics"
            else:
                reason = (
                    f"released gold nuggets match {len(matched)} of {len(ordered_topic_ids)} "
                    "selected topics, but no completed nugget-assignment run exists for them, "
                    "so coverage cannot be computed"
                )
    return {
        "per_topic": {},
        "per_topic_availability": {
            topic_id: {"available": False, "reason": reason} for topic_id in ordered_topic_ids
        },
        "macro": {},
        "macro_availability": {"available": False, "scope_topic_ids": [], "reason": reason},
        "matched_topic_ids": matched,
        "narrative_binding_sha256s": binding,
        "generated_claims_used_as_gold": False,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _manifest(
    *,
    data: DebugReportData,
    rag_config: Any | None,
    rag_config_path: Path | None,
    retrieval_config_path: Path,
    identity_path: Path | None,
    rag_artifact_source: RagArtifactSource,
    accepted_binding: AcceptedRunBinding | None,
    accepted_binding_path: Path | None,
    ordered_topic_ids: Sequence[str],
    requested_topic_ids: Sequence[str] | None,
    ragdoll: Mapping[str, str],
    judge_settings: JudgeSettings,
    judge_report: Mapping[str, int],
    judge_limit: int | None,
    judge_workers: int,
    cache: JudgeCache,
    derived: Mapping[str, Path],
    work_dir: Path,
    support: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    nuggets: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    qrels_path: Path | None,
    gold_nuggets_path: Path | None,
    created_utc: str | None,
) -> dict[str, Any]:
    identity: Mapping[str, Any] = {}
    if identity_path is not None:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        selected_contexts = _selected_topic_contexts(identity, identity_path)
    elif accepted_binding is not None:
        selected_contexts = dict(accepted_binding.topic_context_sha256s)
    else:
        raise EvaluationError("RAG provenance identity is missing")
    evidence = _handoff_evidence_counts(
        rag_artifact_source.handoff_manifest_path,
        handoff=rag_artifact_source.handoff,
    )
    if accepted_binding is not None:
        report_submission_sha256 = data.source_sha256s.get(
            "rag/rag_output_trec_rag_2026.jsonl"
        )
        if report_submission_sha256 != accepted_binding.submission_sha256:
            raise EvaluationError(
                "report RAG output digest does not match accepted binding submission_sha256"
            )
    labels = _labels_by_citation(judgments)
    topics = [_topic_projection(topic, evidence, labels) for topic in data.topics]
    fully_judged = judge_report["missing"] == 0
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        # Receipt-only, deliberately volatile: never a render input, so two independent
        # invocations over identical effective inputs still produce identical HTML.
        "created_utc": created_utc or datetime.now(UTC).isoformat(),
        "volatile_receipt_fields": VOLATILE_RECEIPT_FIELDS,
        "scope": {
            "topic_ids": list(ordered_topic_ids),
            "topic_count": len(ordered_topic_ids),
            # Whether the caller passed topic selectors, not whether the result is nonempty:
            # a valid bundle always has topics, so length can never distinguish these.
            "selection": "explicit" if requested_topic_ids is not None else "all",
            "requested_topic_ids": (
                list(requested_topic_ids) if requested_topic_ids is not None else None
            ),
        },
        "configs": {
            "retrieval": _file_receipt(retrieval_config_path),
            "rag": None if rag_config_path is None else _file_receipt(rag_config_path),
        },
        "identities": _identity_projection(
            rag_artifact_source,
            identity=identity,
            identity_path=identity_path,
            accepted_binding=accepted_binding,
            selected_contexts=selected_contexts,
            ordered_topic_ids=ordered_topic_ids,
        ),
        "sources": {
            **dict(sorted(data.source_sha256s.items())),
            **(
                {}
                if accepted_binding is None
                else {
                    "rag/accepted_bundle_metadata.json": (
                        accepted_binding.bundle_metadata_snapshot.sha256
                        if accepted_binding.bundle_metadata_snapshot is not None
                        else accepted_binding.bundle_metadata_sha256
                    ),
                    **(
                        {}
                        if accepted_binding.source_identity_snapshot is None
                        else {
                            "rag/source_identity.json": accepted_binding.source_identity_snapshot.sha256
                        }
                    ),
                }
            ),
            **(
                {}
                if accepted_binding_path is None
                else {"rag/accepted_run_binding.json": _file_receipt(accepted_binding_path)["sha256"]}
            ),
        },
        "artifacts": {
            name: _file_receipt(path, relative_to=work_dir) for name, path in sorted(derived.items())
        },
        "ragdoll": dict(ragdoll),
        "judge": {
            "provider": judge_settings.provider,
            "model": judge_settings.model,
            "thinking": judge_settings.thinking,
            "temperature": judge_settings.temperature,
            "agent_binary": judge_settings.agent_binary,
            "extension_identity": judge_settings.extension_identity,
            "system_prompt_sha256": sha256(judge_settings.system_prompt.encode("utf-8")).hexdigest(),
            "tasks": judge_report["tasks"],
            "completed": judge_report["completed"],
            "missing": judge_report["missing"],
            "failed": judge_report["failed"],
            "conflicts": judge_report["conflicts"],
            "hosted_calls": judge_report["hosted_calls"],
            "reused_from_cache": judge_report["reused_from_cache"],
            "skipped_by_judge_limit": judge_report["skipped_by_judge_limit"],
            "judge_limit": judge_limit,
            "judge_workers": judge_workers,
        },
        "cache": dict(cache.stats()),
        "metric_definitions": {
            "citation_support": dict(SUPPORT_METRIC_DEFINITIONS),
            "retrieval": dict(RETRIEVAL_METRIC_DEFINITIONS),
            "macro": "Unweighted mean over the topic cells whose required labels are complete.",
        },
        "metrics": {
            "citation_support": {
                key: value for key, value in support.items() if key != "label_counts"
            },
            "retrieval": dict(retrieval),
            "nugget_coverage": dict(nuggets),
        },
        "judgments": {
            "task_count": len(tasks),
            "completed": len(judgments),
            "label_counts_per_topic": support["label_counts"],
            "label_counts": _total_labels(support["label_counts"]),
            "fully_judged": fully_judged,
        },
        "topics": topics,
        "contract": {
            "answer_word_limit": ANSWER_WORD_LIMIT,
        },
        "validation": {
            "sources_hashed": True,
            "topic_order_matches_export": True,
            "handoff_bound_to_generation": (
                identity_path is not None
                or (accepted_binding is not None and accepted_binding.source_identity_available)
            ),
            "support_tasks_unique": True,
            "judgments_validated": fully_judged,
            "labels_valid": True,
            **(
                {"accepted_submission_bound_to_handoff": True}
                if accepted_binding is not None
                else {}
            ),
        },
        "inputs": {
            "qrels_supplied": qrels_path is not None,
            "gold_nuggets_supplied": gold_nuggets_path is not None,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
        },
    }


def _total_labels(per_topic: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    totals = {label: 0 for label in sorted(SUPPORT_LABELS)}
    for counts in per_topic.values():
        for label, count in counts.items():
            totals[label] += count
    return totals


def _identity_projection(
    rag_source: RagArtifactSource,
    *,
    identity: Mapping[str, Any],
    identity_path: Path | None,
    accepted_binding: AcceptedRunBinding | None,
    selected_contexts: Mapping[str, str],
    ordered_topic_ids: Sequence[str],
) -> dict[str, Any]:
    base = {
        "run_id": rag_source.run_id,
        "team_id": rag_source.team_id,
        "run_desc": rag_source.run_desc,
        "handoff_manifest_sha256": (
            identity.get("handoff_manifest_sha256")
            if identity_path is not None
            else accepted_binding.handoff_manifest_sha256
        ),
        "handoff_schema_version": (
            identity.get("handoff_schema_version")
            if identity_path is not None
            else accepted_binding.handoff_schema_version
        ),
        "selected_topic_context_sha256s": {
            topic_id: selected_contexts[topic_id] for topic_id in ordered_topic_ids
        },
    }
    if identity_path is not None:
        base.update(
            {
                "prompt_contract_version": identity.get("prompt_contract_version"),
                "generation_identity_version": identity.get("identity_version"),
                "generation_identity_sha256": _file_receipt(identity_path)["sha256"],
                "source_identity_available": True,
                "source_identity_reason": None,
            }
        )
    else:
        assert accepted_binding is not None
        base.update(
            {
                "binding_kind": accepted_binding.schema_version,
                "source_identity_available": accepted_binding.source_identity_available,
                "source_identity_sha256": accepted_binding.source_identity_sha256,
                "source_identity_reason": accepted_binding.source_identity_reason,
                "generation_identity_sha256": accepted_binding.source_identity_sha256,
                "submission_sha256": accepted_binding.submission_sha256,
                "bundle_metadata_sha256": accepted_binding.bundle_metadata_sha256,
                "models": list(accepted_binding.models),
            }
        )
    return base


def _selected_topic_contexts(identity: Mapping[str, Any], path: Path) -> dict[str, str]:
    """Normalize the generation identity's selected-topic list into a mapping."""
    selected = identity.get("selected_topics")
    if not isinstance(selected, list) or not selected:
        raise EvaluationError(f"{path}: selected_topics must be a nonempty list")
    contexts: dict[str, str] = {}
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            raise EvaluationError(f"{path}: selected_topics[{index}] is not an object")
        topic_id = item.get("topic_id")
        context = item.get("context_sha256")
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise EvaluationError(f"{path}: selected_topics[{index}] has no topic id")
        if topic_id in contexts:
            raise EvaluationError(f"{path}: duplicate selected topic {topic_id}")
        if not isinstance(context, str) or len(context) != 64:
            raise EvaluationError(f"{path}: selected_topics[{index}] has no context digest")
        contexts[topic_id] = context
    return contexts


def _handoff_evidence_counts(
    handoff_manifest_path: Path,
    *,
    handoff: Any | None = None,
) -> dict[str, dict[str, int]]:
    """Count authenticated selected evidence per topic straight from the handoff."""
    from trec_rag.generation_handoff import load_generation_handoff

    if handoff is None:
        handoff = load_generation_handoff(handoff_manifest_path)
    counts: dict[str, dict[str, int]] = {}
    for topic in handoff.topics:
        counts[topic.topic_id] = {
            "evidence_documents": len({evidence.docid for evidence in topic.evidence}),
            "evidence_passages": len(topic.evidence),
        }
    return counts


def _labels_by_citation(
    judgments: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int], str]:
    labels: dict[tuple[str, int, int], str] = {}
    for row in judgments:
        metadata = row.get("metadata", {})
        key = (
            str(metadata.get("topic_id")),
            int(metadata.get("sentence_index", -1)),
            int(metadata.get("citation_index", -1)),
        )
        if key in labels:
            raise EvaluationError(f"duplicate judgment for citation {key}")
        labels[key] = str(row["support_label"])
    return labels


def _topic_projection(
    topic: Any,
    evidence: Mapping[str, Mapping[str, int]],
    labels: Mapping[tuple[str, int, int], str],
) -> dict[str, Any]:
    """Project one topic into the manifest, keeping evidence text and docids out."""
    rag = topic.rag_output
    answer: list[dict[str, Any]] = []
    for sentence_index, item in enumerate(rag.answer_items):
        citations = []
        for citation_index, citation in enumerate(item.citations):
            position = (
                (citation + 1)
                if type(citation) is int
                else (list(rag.references).index(citation) + 1)
            )
            citations.append(
                {
                    "position": position,
                    "support_label": labels.get(
                        (topic.topic_id, sentence_index, citation_index)
                    ),
                }
            )
        answer.append({"text": item.text, "citations": citations})
    citation_instances = sum(len(item["citations"]) for item in answer)
    cited_positions = {
        citation["position"] for item in answer for citation in item["citations"]
    }
    return {
        "topic_id": topic.topic_id,
        "narrative": topic.narrative,
        "subnarratives": [subnarrative.text for subnarrative in topic.subnarratives],
        "answer": answer,
        "candidate_pool_kind": topic.retrieval_output.candidate_pool_kind,
        "counts": {
            # The sealed export's deduplicated pool depth, not the facet-only new-document
            # count, which is a strictly smaller and different quantity.
            "candidate_documents": topic.retrieval_output.selected_pool_depth,
            "facet_new_documents": len(topic.new_documents),
            "submitted_documents": len(topic.retrieval_output.documents),
            "evidence_documents": evidence[topic.topic_id]["evidence_documents"],
            "evidence_passages": evidence[topic.topic_id]["evidence_passages"],
            "answer_references": len(rag.references),
            "answer_objects": len(rag.answer_items),
            "citations": citation_instances,
            "uncited_references": len(rag.references) - len(cited_positions),
            "answer_words": rag.word_count,
        },
    }


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------


def _file_receipt(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    payload = path.read_bytes()
    name = str(path.relative_to(relative_to)) if relative_to is not None else path.name
    return {"path": name, "bytes": len(payload), "sha256": sha256(payload).hexdigest()}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    body = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    _atomic_write(path, body)
    return path


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a bundle manifest, refusing anything that is not this schema."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"{path}: manifest is not an object")
    if value.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise EvaluationError(f"{path}: manifest schema is not {BUNDLE_SCHEMA_VERSION}")
    return value
