"""Durable create/resume CLI for grounded agentic competition retrieval.

The runner owns only lifecycle orchestration.  Semantic configuration remains
immutable, completed topics are authenticated through per-topic seals, and an
aggregate organizer export is published only when the original run-plan cohort
is complete.  Production retrieval/model dependencies are imported and built
only after every local preflight and run-state check succeeds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from .agentic_generation_export import AgenticTopicProjection
from .agentic_run_state import (
    AgenticRunPlan,
    AgenticRunStateError,
    SubmoduleRevision,
    TopicAttempt,
    allocate_topic_attempt,
    create_run_plan,
    load_run_plan,
    resume_run_plan,
    seal_topic_success,
    select_run_topics,
)
from .topics import Topic


_REQUIRED_SECRETS = (
    "INDEX_URL",
    "PYSERINI_API_TOKEN",
    "OPENROUTER_API_KEY",
)
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_CODE = re.compile(r"[a-z][a-z0-9_]*\Z")
_FAILURE_SCHEMA = "agentic_topic_failure_v1"


class AgenticRunnerError(RuntimeError):
    """Safe operator-facing failure raised before or between topic attempts."""

    def __init__(self, code: str, message: str) -> None:
        if _SAFE_CODE.fullmatch(code) is None:
            raise ValueError("runner error code must be safe snake-case text")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("runner error message must be non-empty text")
        super().__init__(message)
        self.code = code


class TopicOperationalError(RuntimeError):
    """One topic stopped for an operational reason safe to retry manually."""

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or _SAFE_CODE.fullmatch(reason) is None:
            raise ValueError("topic operational reason must be safe snake-case text")
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class AgenticRepositoryBinding:
    """Clean source identities authenticated into the immutable run plan."""

    source_revision: str
    submodule_revisions: tuple[SubmoduleRevision, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_revision, str)
            or _REVISION.fullmatch(self.source_revision) is None
        ):
            raise ValueError(
                "source_revision must be a lowercase 40-character Git object ID"
            )
        if any(
            not isinstance(row, SubmoduleRevision)
            for row in self.submodule_revisions
        ):
            raise TypeError("submodule_revisions must contain typed records")
        paths = tuple(row.path for row in self.submodule_revisions)
        if len(set(paths)) != len(paths):
            raise ValueError("submodule_revisions repeat a path")


@dataclass(frozen=True)
class TopicExecutionRequest:
    """Everything one fresh, topic-scoped transaction is allowed to use."""

    topic: Topic
    plan: AgenticRunPlan
    attempt: TopicAttempt
    official_topics_sha256: str


@dataclass(frozen=True)
class TopicExecutionResult:
    """A complete projection ready to seal, or a bounded incomplete attempt."""

    status: str
    stopping_reason: str
    synthesis_outcome: str | None
    projection: AgenticTopicProjection | None
    records_receipt: object | None

    def __post_init__(self) -> None:
        if self.status not in {"complete", "incomplete"}:
            raise ValueError("topic execution status must be complete or incomplete")
        if (
            not isinstance(self.stopping_reason, str)
            or _SAFE_CODE.fullmatch(self.stopping_reason) is None
        ):
            raise ValueError("topic stopping reason must be safe snake-case text")
        if self.status == "complete":
            if self.synthesis_outcome not in {
                "coordinator_selected",
                "deterministic_grounded_recovery",
            }:
                raise ValueError("complete topic requires a successful synthesis outcome")
            if not isinstance(self.projection, AgenticTopicProjection):
                raise TypeError("complete topic requires an agentic projection")
            if self.records_receipt is None:
                raise ValueError("complete topic requires a records receipt")
        elif any(
            value is not None
            for value in (
                self.synthesis_outcome,
                self.projection,
                self.records_receipt,
            )
        ):
            raise ValueError("incomplete topic cannot carry sealable projection data")


@dataclass(frozen=True)
class AgenticRunReceipt:
    """Concise, non-secret outcome for one create or resume invocation."""

    run_id: str
    output_dir: Path
    planned_topic_ids: tuple[str, ...]
    skipped_topic_ids: tuple[str, ...]
    executed_topic_ids: tuple[str, ...]
    unresolved_topic_ids: tuple[str, ...]
    export: Any | None
    resume_command: str | None

    @property
    def complete(self) -> bool:
        return not self.unresolved_topic_ids and self.export is not None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": "complete" if self.complete else "incomplete",
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "planned_topic_ids": list(self.planned_topic_ids),
            "skipped_topic_ids": list(self.skipped_topic_ids),
            "executed_topic_ids": list(self.executed_topic_ids),
            "unresolved_topic_ids": list(self.unresolved_topic_ids),
        }
        if self.export is not None:
            payload.update(
                {
                    "export_manifest": str(self.export.manifest),
                    "export_manifest_sha256": self.export.manifest_sha256,
                }
            )
        if self.resume_command is not None:
            payload["resume_command"] = self.resume_command
        return payload


def _run_git(repo: Path, arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgenticRunnerError(
            "repository_preflight_failed",
            "unable to verify the repository source binding",
        ) from exc
    return completed.stdout


def _probe_repository(repo: Path) -> AgenticRepositoryBinding:
    """Reject tracked or submodule drift and return exact committed identities."""

    root = Path(repo).resolve()
    status = _run_git(
        root,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            "--ignore-submodules=none",
        ),
    )
    if status.strip():
        raise AgenticRunnerError(
            "dirty_worktree",
            "tracked worktree or submodule changes are not allowed",
        )
    revision = _run_git(root, ("rev-parse", "--verify", "HEAD")).strip()
    if _REVISION.fullmatch(revision) is None:
        raise AgenticRunnerError(
            "repository_preflight_failed",
            "repository HEAD is not a lowercase 40-character Git object ID",
        )
    submodules: list[SubmoduleRevision] = []
    for line in _run_git(root, ("submodule", "status", "--recursive")).splitlines():
        if not line or line[0] != " ":
            raise AgenticRunnerError(
                "submodule_drift",
                "submodules must be initialized at their pinned revisions",
            )
        fields = line[1:].split()
        if len(fields) < 2 or _REVISION.fullmatch(fields[0]) is None:
            raise AgenticRunnerError(
                "submodule_drift", "unable to authenticate submodule revisions"
            )
        submodules.append(SubmoduleRevision(fields[1], fields[0]))
    return AgenticRepositoryBinding(revision, tuple(submodules))


def _read_stable(path: Path, *, label: str) -> bytes:
    try:
        before = path.read_bytes()
        if not before:
            raise AgenticRunnerError(
                "input_invalid", f"{label} must be a non-empty regular file"
            )
        if path.is_symlink() or not path.is_file():
            raise AgenticRunnerError(
                "input_invalid", f"{label} must be a non-empty regular file"
            )
        return before
    except AgenticRunnerError:
        raise
    except OSError as exc:
        raise AgenticRunnerError(
            "input_invalid", f"unable to read {label}"
        ) from exc


def _require_unchanged(path: Path, expected: bytes, *, label: str) -> None:
    try:
        current = path.read_bytes()
    except OSError as exc:
        raise AgenticRunnerError(
            "input_changed", f"{label} changed during preflight"
        ) from exc
    if current != expected:
        raise AgenticRunnerError(
            "input_changed", f"{label} changed during preflight"
        )


def _load_environment(root: Path) -> None:
    from .repo_env import load_repo_env

    load_repo_env(root)


def _check_secret_presence(environ: Mapping[str, str]) -> None:
    missing = tuple(
        name
        for name in _REQUIRED_SECRETS
        if not isinstance(environ.get(name), str) or not environ[name].strip()
    )
    if missing:
        raise AgenticRunnerError(
            "missing_secrets",
            "missing required secret names: " + ", ".join(missing),
        )


def _planned_topics(
    stored: AgenticRunPlan, official_topics: Sequence[Topic]
) -> tuple[Topic, ...]:
    by_id = {topic.id: topic for topic in official_topics}
    if len(by_id) != len(official_topics):
        raise AgenticRunnerError(
            "topic_source_invalid", "official topic source repeats an identity"
        )
    missing = tuple(
        topic_id
        for topic_id in stored.planned_topic_ids
        if topic_id not in by_id
    )
    if missing:
        raise AgenticRunnerError(
            "topic_source_drift",
            "stored topic cohort is missing from the official topic source",
        )
    return tuple(by_id[topic_id] for topic_id in stored.planned_topic_ids)


def _failure_body(request: TopicExecutionRequest, reason: str) -> bytes:
    safe_reason = reason if _SAFE_CODE.fullmatch(reason) is not None else "operational_failure"
    return _canonical_line(
        {
            "schema_version": _FAILURE_SCHEMA,
            "topic_id": request.topic.id,
            "attempt_number": request.attempt.attempt_number,
            "reason": safe_reason,
            "cache_reuse_available": True,
        }
    )


def _finalize_topic_execution_result(
    *,
    work_dir: Path,
    plan: AgenticRunPlan,
    request: TopicExecutionRequest,
    result: TopicExecutionResult,
) -> None:
    if not isinstance(result, TopicExecutionResult):
        raise TypeError("topic executor must return TopicExecutionResult")
    if result.status == "incomplete":
        request.attempt.write_artifact(
            "failure.json",
            _failure_body(request, result.stopping_reason),
        )
        return
    try:
        seal_topic_success(
            work_dir=work_dir,
            plan=plan,
            projection=result.projection,
            attempt=request.attempt,
            status="complete",
            stopping_reason=result.stopping_reason,
            synthesis_outcome=result.synthesis_outcome,
            records_receipt=result.records_receipt,
        )
    except AgenticRunStateError as exc:
        raise AgenticRunnerError("topic_seal_invalid", str(exc)) from exc


def _canonical_line(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - fixed records only
        raise AgenticRunnerError(
            "diagnostic_failed", "unable to serialize a topic diagnostic"
        ) from exc


def _resume_command(config_argument: str, topic_ids: Sequence[str]) -> str:
    arguments = [
        ".venv/bin/python-rocm",
        "-m",
        "trec_rag.competition_agentic_retrieval",
        config_argument,
        "--resume",
    ]
    for topic_id in topic_ids:
        arguments.extend(("--topic", topic_id))
    return " ".join(shlex.quote(argument) for argument in arguments)


def _build_production_topic_executor(config: Any) -> Callable[[TopicExecutionRequest], TopicExecutionResult]:
    """Build cache-first production components after preflight has succeeded."""

    from .chunking import ChunkingConfig, SemanticTextChunker
    from .competition_retrieval import _build_topic_passage_search
    from .deepagent_retrieval import AgentRetrievalError, DeepAgentRetriever
    from .deepagent_snippets import (
        LocalMixedbreadSnippetRanker,
        RelevantSnippetExtractor,
        SnippetExtractionConfig,
        SnippetResultCache,
    )
    from .document_store import DocumentStore
    from .facet_retrieval import build_pyserini_retriever
    from .mixedbread_passage_scorer import MixedbreadPassageScorer
    from .topic_records import TopicEvidenceSnapshot, TopicRecordsBuilder

    def execute(request: TopicExecutionRequest) -> TopicExecutionResult:
        document_store = DocumentStore(config.caches.document_store_dir)
        pyserini = build_pyserini_retriever(
            config.retrieval.cache_dir,
            index=config.retrieval.index,
            hits=config.retrieval.documents_per_query,
            corpus_epoch=config.retrieval.corpus_epoch,
        )
        passage_scorer = MixedbreadPassageScorer(
            score_cache_root=config.passage.score_cache_dir,
            device=config.passage.device,
        )
        chunker = SemanticTextChunker(
            ChunkingConfig(
                max_characters=config.passage.chunk_max_characters,
                overlap_characters=config.passage.chunk_overlap_characters,
            )
        )
        snippet_extractor = RelevantSnippetExtractor(
            ranker=LocalMixedbreadSnippetRanker(
                score_cache_root=config.passage.score_cache_dir,
                device=config.passage.device,
            ),
            result_cache=SnippetResultCache(config.snippets.result_cache_dir),
            config=SnippetExtractionConfig(
                snippets_per_page=config.snippets.snippets_per_page,
                chunk_max_characters=config.passage.chunk_max_characters,
                chunk_overlap_characters=config.passage.chunk_overlap_characters,
            ),
            chunker=chunker,
        )
        passage_search = _build_topic_passage_search(
            request.topic,
            retriever=pyserini,
            scorer=passage_scorer,
            document_store_root=config.caches.document_store_dir,
            retrieval_cache_dir=config.retrieval.cache_dir,
            retrieval_index=config.retrieval.index,
            corpus_epoch=config.retrieval.corpus_epoch,
            score_cache_root=config.passage.score_cache_dir,
            device=config.passage.device,
            retrieval_depth=config.retrieval.documents_per_query,
            passages_per_query=config.passage.passages_per_query,
            chunk_max_characters=config.passage.chunk_max_characters,
            chunk_overlap_characters=config.passage.chunk_overlap_characters,
        )
        records = TopicRecordsBuilder(
            request.attempt.path / "topic_records",
            request.topic.id,
            document_store,
            run_id=request.plan.run_id,
        )
        retriever = DeepAgentRetriever.from_env(
            passage_search=passage_search,
            root=config.root_dir,
            model=config.models.coordinator_and_researcher,
            snippet_extractor=snippet_extractor,
            hits_per_search=config.retrieval.hits_per_search,
            max_followup_searches=config.budget.max_searches_per_researcher,
            fused_result_limit=config.agent.fused_result_limit,
            budget_config=config.budget,
        )
        try:
            result = retriever.retrieve(records, request.topic.narrative)
        except AgentRetrievalError as exc:
            raise TopicOperationalError("agent_retrieval_failure") from exc
        except (ConnectionError, TimeoutError) as exc:
            raise TopicOperationalError("provider_failure") from exc

        snapshot = result.topic_snapshot
        reason = result.stopping_reason
        if (
            result.synthesis_outcome == "zero_grounded_nuggets"
            or not isinstance(snapshot, TopicEvidenceSnapshot)
            or snapshot.status != "complete"
        ):
            return TopicExecutionResult(
                status="incomplete",
                stopping_reason=(
                    reason if _SAFE_CODE.fullmatch(reason) is not None else "operational_failure"
                ),
                synthesis_outcome=None,
                projection=None,
                records_receipt=None,
            )

        published = records.publish(
            {
                "schema_version": "agentic_topic_records_identity_v1",
                "run_plan_sha256": request.plan.plan_sha256,
                "topic_id": request.topic.id,
                "attempt_number": request.attempt.attempt_number,
            }
        )
        from .agentic_generation_export import prepare_agentic_topic_projection

        projection = prepare_agentic_topic_projection(
            topic=request.topic,
            report=result.coverage_report,
            snapshot=snapshot,
            fused_candidates=result.candidates,
            searches=result.searches,
            document_store=document_store,
            official_topics_sha256=request.official_topics_sha256,
        )
        return TopicExecutionResult(
            status="complete",
            stopping_reason=snapshot.stopping_reason,
            synthesis_outcome=result.synthesis_outcome,
            projection=projection,
            records_receipt=published.receipt,
        )

    return execute


def _run_production_topic_request(
    config: Any,
    request: TopicExecutionRequest,
) -> TopicExecutionResult:
    """Build and run one topic inside a spawn-safe production worker."""

    return _build_production_topic_executor(config)(request)


def run_agentic_retrieval(
    config: str | Path,
    *,
    resume: bool = False,
    topic_ids: Sequence[str] | None = None,
) -> AgenticRunReceipt:
    """Run create, resume-all, or targeted repair with production dependencies."""

    return _run_agentic_retrieval(
        config,
        resume=resume,
        topic_ids=topic_ids,
        environment_loader=_load_environment,
        environ=None,
        repository_probe=_probe_repository,
        topic_executor_factory=_build_production_topic_executor,
        production_dispatch=True,
    )


def _run_agentic_retrieval(
    config: str | Path,
    *,
    resume: bool,
    topic_ids: Sequence[str] | None,
    environment_loader: Callable[[Path], None],
    environ: Mapping[str, str] | None,
    repository_probe: Callable[[Path], AgenticRepositoryBinding],
    topic_executor_factory: Callable[[Any], Callable[[TopicExecutionRequest], TopicExecutionResult]],
    export_publisher: Callable[..., Any] | None = None,
    production_dispatch: bool = False,
) -> AgenticRunReceipt:
    """Injected implementation used by offline orchestration tests."""

    if not isinstance(resume, bool):
        raise TypeError("resume must be Boolean")
    if not isinstance(production_dispatch, bool):
        raise TypeError("production_dispatch must be Boolean")
    if topic_ids is None:
        requested_ids: tuple[str, ...] = ()
    else:
        if isinstance(topic_ids, (str, bytes)):
            raise TypeError("topic_ids must be a sequence of topic IDs")
        requested_ids = tuple(topic_ids)
        if any(not isinstance(topic_id, str) for topic_id in requested_ids):
            raise TypeError("topic_ids must contain text identities")

    config_argument = str(config)
    config_path = Path(config).resolve()
    config_bytes = _read_stable(config_path, label="agentic config")
    try:
        from .agentic_retrieval_config import (
            load_agentic_retrieval_config,
            select_agentic_topics,
        )

        loaded = load_agentic_retrieval_config(
            config_path, source_bytes=config_bytes
        )
        topic_source_bytes = _read_stable(
            loaded.topics_path, label="official topic source"
        )
        official_topics = tuple(select_agentic_topics(loaded))
    except AgenticRunnerError:
        raise
    except Exception as exc:
        raise AgenticRunnerError(
            "config_invalid", "agentic configuration or topic source is invalid"
        ) from exc
    if not official_topics:
        raise AgenticRunnerError(
            "topic_selection_empty", "agentic runs require at least one official topic"
        )

    output_dir = loaded.output_dir
    work_dir = output_dir / "work"
    if not resume and (output_dir.exists() or output_dir.is_symlink()):
        raise AgenticRunnerError(
            "namespace_exists",
            f"agentic run namespace already exists: {output_dir}",
        )

    environment_loader(loaded.root_dir)
    runtime_environ = os.environ if environ is None else environ
    _check_secret_presence(runtime_environ)
    binding = repository_probe(loaded.root_dir)
    if not isinstance(binding, AgenticRepositoryBinding):
        raise TypeError("repository_probe must return AgenticRepositoryBinding")
    _require_unchanged(config_path, config_bytes, label="agentic config")
    _require_unchanged(
        loaded.topics_path,
        topic_source_bytes,
        label="official topic source",
    )
    official_topics_sha256 = sha256(topic_source_bytes).hexdigest()

    try:
        if resume:
            stored = load_run_plan(work_dir)
            cohort = _planned_topics(stored, official_topics)
            plan = resume_run_plan(
                work_dir=work_dir,
                run_id=loaded.run_id,
                config_bytes=config_bytes,
                topics=cohort,
                official_topics_sha256=official_topics_sha256,
                source_revision=binding.source_revision,
                submodule_revisions=binding.submodule_revisions,
            )
            selection = select_run_topics(
                work_dir=work_dir,
                plan=plan,
                topic_ids=requested_ids or None,
            )
        else:
            cohort = tuple(
                select_agentic_topics(loaded, topic_ids=requested_ids)
            )
            if not cohort:
                raise AgenticRunnerError(
                    "topic_selection_empty",
                    "create requires at least one selected topic",
                )
            plan = create_run_plan(
                work_dir=work_dir,
                run_id=loaded.run_id,
                config_bytes=config_bytes,
                topics=cohort,
                official_topics_sha256=official_topics_sha256,
                source_revision=binding.source_revision,
                submodule_revisions=binding.submodule_revisions,
            )
            selection = select_run_topics(
                work_dir=work_dir,
                plan=plan,
                topic_ids=None,
            )
    except AgenticRunnerError:
        raise
    except AgenticRunStateError as exc:
        raise AgenticRunnerError("run_state_invalid", str(exc)) from exc
    except Exception as exc:
        raise AgenticRunnerError(
            "topic_selection_invalid", "topic selection is invalid"
        ) from exc

    os.environ["HF_HOME"] = str(loaded.caches.model_cache_dir)
    topic_by_id = {topic.id: topic for topic in cohort}
    executed: list[str] = []
    requests: list[TopicExecutionRequest] = []
    if selection.execute_topic_ids:
        executor = None
        if not production_dispatch:
            executor = topic_executor_factory(loaded)
            if not callable(executor):
                raise TypeError("topic_executor_factory must return a callable")
        invoked: set[str] = set()
        for topic_id in selection.execute_topic_ids:
            if topic_id in invoked:
                raise AgenticRunnerError(
                    "duplicate_execution", "one topic cannot execute twice in one invocation"
                )
            invoked.add(topic_id)
            topic = topic_by_id[topic_id]
            try:
                attempt = allocate_topic_attempt(
                    work_dir=work_dir,
                    plan=plan,
                    topic_id=topic_id,
                )
            except AgenticRunStateError as exc:
                raise AgenticRunnerError("run_state_invalid", str(exc)) from exc
            request = TopicExecutionRequest(
                topic=topic,
                plan=plan,
                attempt=attempt,
                official_topics_sha256=official_topics_sha256,
            )
            requests.append(request)
            executed.append(topic_id)

        def finalize(request: TopicExecutionRequest, result: TopicExecutionResult) -> None:
            _finalize_topic_execution_result(
                work_dir=work_dir,
                plan=plan,
                request=request,
                result=result,
            )

        if loaded.execution.topic_workers == 1:
            for request in requests:
                try:
                    if production_dispatch:
                        result = _run_production_topic_request(loaded, request)
                    else:
                        assert executor is not None
                        result = executor(request)
                except TopicOperationalError as exc:
                    request.attempt.write_artifact(
                        "failure.json", _failure_body(request, exc.reason)
                    )
                    continue
                finalize(request, result)
        elif production_dispatch:
            errors: list[Exception] = []
            with ProcessPoolExecutor(
                max_workers=loaded.execution.topic_workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = {
                    pool.submit(_run_production_topic_request, loaded, request): request
                    for request in requests
                }
                for future in as_completed(futures):
                    request = futures[future]
                    try:
                        result = future.result()
                    except TopicOperationalError as exc:
                        request.attempt.write_artifact(
                            "failure.json", _failure_body(request, exc.reason)
                        )
                    except Exception as exc:
                        errors.append(exc)
                    else:
                        try:
                            finalize(request, result)
                        except Exception as exc:
                            errors.append(exc)
            if errors:
                raise errors[0]
        else:
            errors = []
            assert executor is not None
            with ThreadPoolExecutor(
                max_workers=loaded.execution.topic_workers
            ) as pool:
                futures = {
                    pool.submit(executor, request): request for request in requests
                }
                for future in as_completed(futures):
                    request = futures[future]
                    try:
                        result = future.result()
                    except TopicOperationalError as exc:
                        request.attempt.write_artifact(
                            "failure.json", _failure_body(request, exc.reason)
                        )
                    except Exception as exc:
                        errors.append(exc)
                    else:
                        try:
                            finalize(request, result)
                        except Exception as exc:
                            errors.append(exc)
            if errors:
                raise errors[0]

    try:
        final_selection = select_run_topics(
            work_dir=work_dir,
            plan=plan,
            topic_ids=None,
        )
    except AgenticRunStateError as exc:
        raise AgenticRunnerError("run_state_invalid", str(exc)) from exc
    completed = set(final_selection.completed_topic_ids)
    unresolved = tuple(
        topic_id
        for topic_id in plan.planned_topic_ids
        if topic_id not in completed
    )
    export = None
    if not unresolved:
        if export_publisher is None:
            from .agentic_retrieval_export import publish_agentic_retrieval_export

            publisher = publish_agentic_retrieval_export
        else:
            publisher = export_publisher
        export = publisher(
            output_dir=output_dir,
            work_dir=work_dir,
            plan=plan,
            producer_revision=binding.source_revision,
        )
    resume_command = (
        _resume_command(config_argument, unresolved) if unresolved else None
    )
    return AgenticRunReceipt(
        run_id=plan.run_id,
        output_dir=output_dir,
        planned_topic_ids=plan.planned_topic_ids,
        skipped_topic_ids=selection.completed_topic_ids,
        executed_topic_ids=tuple(executed),
        unresolved_topic_ids=unresolved,
        export=export,
        resume_command=resume_command,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or resume a sealed agentic TREC RAG retrieval run. "
            "Resume selectors narrow execution; the original run plan always "
            "controls final artifact membership."
        ),
        epilog=(
            "Create: %(prog)s CONFIG [--topic ID ...]\n"
            "Resume all outstanding: %(prog)s CONFIG --resume\n"
            "Repair selected topics: %(prog)s CONFIG --resume --topic ID"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path, metavar="CONFIG")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate the existing run plan and execute outstanding topics",
    )
    parser.add_argument(
        "--topic",
        action="append",
        dest="topic_ids",
        metavar="ID",
        help=(
            "repeat to define the create cohort or target incomplete resume topics"
        ),
    )
    return parser


def _main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., AgenticRunReceipt] = run_agentic_retrieval,
) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = runner(
            args.config,
            resume=args.resume,
            topic_ids=(
                None if args.topic_ids is None else tuple(args.topic_ids)
            ),
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
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "code": "unexpected_failure",
                    "exception_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            file=os.sys.stderr,
        )
        return 2
    target = os.sys.stdout if receipt.complete else os.sys.stderr
    print(json.dumps(receipt.to_payload(), sort_keys=True), file=target)
    return 0 if receipt.complete else 2


def main(argv: Sequence[str] | None = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AgenticRepositoryBinding",
    "AgenticRunReceipt",
    "AgenticRunnerError",
    "TopicExecutionRequest",
    "TopicExecutionResult",
    "TopicOperationalError",
    "main",
    "run_agentic_retrieval",
]
