"""Process-safe, manifest-last dispatch for independent topic jobs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal


_SCHEMA_VERSION = "topic-job-receipt-v4"
_RECEIPT_FILENAME = "topic-job-receipt.json"
_OFFLINE_RECEIPT_FILENAME = "topic-job-receipt.offline-cache-only.json"
_CACHED_RESCORE_RECEIPT_FILENAME = "topic-job-receipt.cached-upstream-rescore.json"
_PROJECTION_MANIFEST = Path("canonical/retrieval-projection-manifest.json")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INCOMPLETE_REASONS = {
    "budget_exhausted",
    "hard_deadline",
    "retrieval_unavailable",
    "scoring_failed",
    "no_evidence",
    "evidence_validation_failed",
}
ExecutionPolicy = Literal[
    "online",
    "offline-cache-only",
    "cached-upstream-rescore",
]
_EXECUTION_POLICIES = frozenset(
    {"online", "offline-cache-only", "cached-upstream-rescore"}
)


class TopicDispatchError(RuntimeError):
    """One or more topic workers failed after dispatch began."""


class TopicDispatchIntegrityError(ValueError):
    """A sealed topic receipt conflicts with its immutable identity."""


@dataclass(frozen=True)
class TopicJob:
    """Serializable identity for one independently executable topic."""

    topic_id: str
    run_id: str
    config_path: Path
    config_bytes: bytes
    config_sha256: str
    topic_root: Path
    execution_policy: ExecutionPolicy = "online"

    def __post_init__(self) -> None:
        if not isinstance(self.topic_id, str) or not _SAFE_ID.fullmatch(self.topic_id):
            raise ValueError("topic_id must be a safe identifier")
        if not isinstance(self.run_id, str) or not _SAFE_ID.fullmatch(self.run_id):
            raise ValueError("run_id must be a safe identifier")
        for name in ("config_path", "topic_root"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute Path")
        if not isinstance(self.config_bytes, bytes) or not self.config_bytes:
            raise ValueError("config_bytes must be non-empty bytes")
        if (
            not isinstance(self.config_sha256, str)
            or not _SHA256.fullmatch(self.config_sha256)
            or sha256(self.config_bytes).hexdigest() != self.config_sha256
        ):
            raise ValueError("config_sha256 must bind the exact config_bytes")
        if (
            not isinstance(self.execution_policy, str)
            or self.execution_policy not in _EXECUTION_POLICIES
        ):
            raise ValueError("execution_policy is invalid")

    @property
    def offline_cache_only(self) -> bool:
        """Compatibility predicate for the fully offline replay policy."""
        return self.execution_policy == "offline-cache-only"


@dataclass(frozen=True)
class TopicJobReceipt:
    """Path-free completion receipt returned across a process boundary."""

    topic_id: str
    projection_manifest_sha256: str
    status: str
    stopping_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.topic_id, str) or not _SAFE_ID.fullmatch(self.topic_id):
            raise ValueError("receipt topic_id must be a safe identifier")
        if (
            not isinstance(self.projection_manifest_sha256, str)
            or not _SHA256.fullmatch(self.projection_manifest_sha256)
        ):
            raise ValueError("projection_manifest_sha256 must be a lowercase SHA-256")
        if self.status == "complete":
            if self.stopping_reason != "coverage_sufficient":
                raise ValueError("complete receipt must stop at coverage_sufficient")
        elif self.status == "incomplete":
            if self.stopping_reason not in _INCOMPLETE_REASONS:
                raise ValueError("incomplete receipt has an invalid stopping_reason")
        else:
            raise ValueError("receipt status must be complete or incomplete")


TopicWorker = Callable[[TopicJob], TopicJobReceipt]


def dispatch_topics(
    jobs: Sequence[TopicJob],
    worker: TopicWorker,
    *,
    max_workers: int,
) -> tuple[TopicJobReceipt, ...]:
    """Run unsealed topics and return validated receipts in input order.

    All pending process workers are allowed to finish after any sibling fails.
    Their successfully sealed receipts remain resumable, while the aggregate
    call raises and therefore cannot publish a global run manifest.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if not callable(worker):
        raise TypeError("worker must be callable")
    ordered_jobs = tuple(jobs)
    if any(not isinstance(job, TopicJob) for job in ordered_jobs):
        raise TypeError("jobs must contain TopicJob values")

    receipts: list[TopicJobReceipt | None] = [None] * len(ordered_jobs)
    pending: list[tuple[int, TopicJob]] = []
    for index, job in enumerate(ordered_jobs):
        sealed = read_topic_receipt(job, missing_ok=True)
        if sealed is None:
            pending.append((index, job))
        else:
            receipts[index] = sealed

    failures: list[tuple[int, TopicJob, Exception]] = []
    if max_workers == 1:
        for index, job in pending:
            try:
                returned = _execute_and_publish(job, worker)
                receipts[index] = _validate_returned_receipt(job, returned)
            except Exception as exc:  # preserve other independently useful jobs
                failures.append((index, job, exc))
    elif pending:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Future[TopicJobReceipt], tuple[int, TopicJob]] = {
                executor.submit(_execute_and_publish, job, worker): (index, job)
                for index, job in pending
            }
            for future in as_completed(futures):
                index, job = futures[future]
                try:
                    returned = future.result()
                    receipts[index] = _validate_returned_receipt(job, returned)
                except Exception as exc:
                    failures.append((index, job, exc))

    if failures:
        failures.sort(key=lambda row: row[0])
        summary = ", ".join(
            f"{job.topic_id}: {type(exc).__name__}: {exc}"
            for _, job, exc in failures
        )
        error = TopicDispatchError(f"topic dispatch failed for {summary}")
        raise error from failures[0][2]
    if any(receipt is None for receipt in receipts):
        raise TopicDispatchIntegrityError("topic dispatch did not produce every receipt")
    return tuple(receipt for receipt in receipts if receipt is not None)


def publish_topic_receipt(job: TopicJob, receipt: TopicJobReceipt) -> TopicJobReceipt:
    """Create-only publish a canonical receipt after its projection manifest."""
    _validate_job_receipt_pair(job, receipt)
    destination = _receipt_path(job)
    body = _receipt_bytes(job, receipt)
    if destination.exists():
        if destination.read_bytes() != body:
            raise TopicDispatchIntegrityError(
                "conflicting topic receipt for the same run identity"
            )
        restored = read_topic_receipt(job)
        assert restored is not None
        return restored

    _validate_projection_manifest(job, receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".topic-job-receipt-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != body:
                raise TopicDispatchIntegrityError(
                    "conflicting topic receipt for the same run identity"
                )
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    restored = read_topic_receipt(job)
    assert restored is not None
    return restored


def read_topic_receipt(
    job: TopicJob,
    *,
    missing_ok: bool = False,
) -> TopicJobReceipt | None:
    """Read and fully validate one manifest-last topic dispatch receipt."""
    if not isinstance(job, TopicJob):
        raise TypeError("job must be a TopicJob")
    path = _receipt_path(job)
    try:
        body = path.read_bytes()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise TopicDispatchIntegrityError("topic receipt is missing") from None
    try:
        value = json.loads(body, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, TopicDispatchIntegrityError) as exc:
        raise TopicDispatchIntegrityError("topic receipt is invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_id",
        "topic_id",
        "config_sha256",
        "mode",
        "projection_manifest_sha256",
        "status",
        "stopping_reason",
    }:
        raise TopicDispatchIntegrityError("topic receipt fields changed")
    if value.get("schema_version") != _SCHEMA_VERSION:
        raise TopicDispatchIntegrityError("topic receipt schema changed")
    if value.get("run_id") != job.run_id:
        raise TopicDispatchIntegrityError("topic receipt run identity changed")
    if value.get("topic_id") != job.topic_id:
        raise TopicDispatchIntegrityError("topic receipt topic identity changed")
    if value.get("config_sha256") != job.config_sha256:
        raise TopicDispatchIntegrityError("topic receipt config identity changed")
    if value.get("mode") != _job_mode(job):
        raise TopicDispatchIntegrityError("topic receipt execution mode changed")
    try:
        receipt = TopicJobReceipt(
            topic_id=value["topic_id"],
            projection_manifest_sha256=value["projection_manifest_sha256"],
            status=value["status"],
            stopping_reason=value["stopping_reason"],
        )
    except (TypeError, ValueError) as exc:
        raise TopicDispatchIntegrityError("topic receipt values changed") from exc
    if _receipt_bytes(job, receipt) != body:
        raise TopicDispatchIntegrityError("topic receipt is not canonical")
    _validate_projection_manifest(job, receipt)
    return receipt


def _execute_and_publish(job: TopicJob, worker: TopicWorker) -> TopicJobReceipt:
    receipt = worker(job)
    if not isinstance(receipt, TopicJobReceipt):
        raise TypeError("topic worker must return TopicJobReceipt")
    return publish_topic_receipt(job, receipt)


def _validate_returned_receipt(
    job: TopicJob,
    returned: TopicJobReceipt,
) -> TopicJobReceipt:
    _validate_job_receipt_pair(job, returned)
    sealed = read_topic_receipt(job)
    if sealed != returned:
        raise TopicDispatchIntegrityError("returned topic receipt differs from disk seal")
    return returned


def _validate_job_receipt_pair(job: TopicJob, receipt: TopicJobReceipt) -> None:
    if not isinstance(job, TopicJob):
        raise TypeError("job must be a TopicJob")
    if not isinstance(receipt, TopicJobReceipt):
        raise TypeError("receipt must be a TopicJobReceipt")
    if receipt.topic_id != job.topic_id:
        raise TopicDispatchIntegrityError("topic worker returned a wrong topic")


def _validate_projection_manifest(job: TopicJob, receipt: TopicJobReceipt) -> None:
    try:
        body = (job.topic_root / _PROJECTION_MANIFEST).read_bytes()
    except OSError as exc:
        raise TopicDispatchIntegrityError("topic projection manifest is missing") from exc
    if sha256(body).hexdigest() != receipt.projection_manifest_sha256:
        raise TopicDispatchIntegrityError("topic projection manifest hash changed")


def _receipt_path(job: TopicJob) -> Path:
    filename = {
        "online": _RECEIPT_FILENAME,
        "offline-cache-only": _OFFLINE_RECEIPT_FILENAME,
        "cached-upstream-rescore": _CACHED_RESCORE_RECEIPT_FILENAME,
    }[job.execution_policy]
    return job.topic_root / filename


def _job_mode(job: TopicJob) -> str:
    return job.execution_policy


def _receipt_bytes(job: TopicJob, receipt: TopicJobReceipt) -> bytes:
    return json.dumps(
        {
            "schema_version": _SCHEMA_VERSION,
            "run_id": job.run_id,
            "topic_id": receipt.topic_id,
            "config_sha256": job.config_sha256,
            "mode": _job_mode(job),
            "projection_manifest_sha256": receipt.projection_manifest_sha256,
            "status": receipt.status,
            "stopping_reason": receipt.stopping_reason,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TopicDispatchIntegrityError(f"duplicate receipt field: {key}")
        value[key] = item
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ExecutionPolicy",
    "TopicDispatchError",
    "TopicDispatchIntegrityError",
    "TopicJob",
    "TopicJobReceipt",
    "dispatch_topics",
    "publish_topic_receipt",
    "read_topic_receipt",
]
