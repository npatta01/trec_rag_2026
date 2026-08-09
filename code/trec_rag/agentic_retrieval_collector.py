"""Incrementally collect authenticated agentic retrieval topic bundles.

The producer publishes ``bundle.tar.zst`` first and the small
``bundle-complete.json`` marker last.  The collector treats the marker as an
eligibility hint only: every downloaded pair is verified offline through the
topic-bundle module before it is imported.  Local state is append-only so an
interrupted watch cycle can be restarted without repeating successful imports.

The HF transport and the bundle operations are deliberately dependency
injected.  This keeps the collector testable without credentials or a network
and leaves the private ``hf`` CLI details at the process boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from typing import Protocol

from .hf_bucket_listing import HFListingError, parse_hf_bucket_listing


WAVE_RECEIPTS_FILENAME = "collector-wave-receipts.jsonl"
COLLECTOR_JOURNAL_FILENAME = "collector-journal.jsonl"
COLLECTOR_LOCK_FILENAME = ".collector.lock"
EXPORT_RECEIPT_FILENAME = "collector-export-receipt.json"
QUARANTINE_DIRNAME = "quarantine"
INCOMING_DIRNAME = ".incoming"
_SCHEMA = "agentic_retrieval_collector_wave_v1"
_EXPORT_SCHEMA = "agentic_retrieval_collector_export_v1"
_ARCHIVE_NAME = "bundle.tar.zst"
_MARKER_NAME = "bundle-complete.json"
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")


class CollectorError(ValueError):
    """The collector configuration or durable local state is invalid."""


class HFTransport(Protocol):
    """Minimal private-bucket seam used by :class:`IncrementalCollector`."""

    def list(self, prefix: str) -> str | bytes | Sequence[Mapping[str, object]]:
        ...

    def download(self, prefix: str, destination: Path) -> object:
        ...


@dataclass(frozen=True)
class WaveReceipt:
    """The durable result of one bounded listing/import cycle."""

    wave: int
    imported: tuple[str, ...]
    present: tuple[str, ...]
    rejected: tuple[str, ...]
    failed: tuple[str, ...]
    missing: tuple[str, ...]
    complete: bool
    exported: bool = False
    export_error: str | None = None
    topic_digests: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "wave": self.wave,
            "imported": list(self.imported),
            "present": list(self.present),
            "rejected": list(self.rejected),
            "failed": list(self.failed),
            "missing": list(self.missing),
            "complete": self.complete,
            "exported": self.exported,
            "export_error": self.export_error,
            "topic_digests": {
                topic_id: dict(digests)
                for topic_id, digests in self.topic_digests.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WaveReceipt":
        if payload.get("schema_version") != _SCHEMA:
            raise CollectorError("collector wave receipt schema changed")
        wave = payload.get("wave")
        if type(wave) is not int or wave <= 0:
            raise CollectorError("collector wave number is invalid")

        def ids(name: str) -> tuple[str, ...]:
            value = payload.get(name)
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                raise CollectorError(f"collector receipt {name} is invalid")
            if len(set(value)) != len(value):
                raise CollectorError(f"collector receipt {name} repeats a topic")
            return tuple(value)

        complete = payload.get("complete")
        exported = payload.get("exported", False)
        error = payload.get("export_error")
        if type(complete) is not bool or type(exported) is not bool:
            raise CollectorError("collector receipt completion fields are invalid")
        if error is not None and not isinstance(error, str):
            raise CollectorError("collector receipt export error is invalid")
        raw_digests = payload.get("topic_digests", {})
        if not isinstance(raw_digests, Mapping):
            raise CollectorError("collector receipt topic digests are invalid")
        topic_digests: dict[str, Mapping[str, str]] = {}
        for topic_id, value in raw_digests.items():
            if not isinstance(topic_id, str) or not isinstance(value, Mapping):
                raise CollectorError("collector receipt topic digest row is invalid")
            if any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
                raise CollectorError("collector receipt topic digest values are invalid")
            topic_digests[topic_id] = dict(value)
        return cls(
            wave,
            ids("imported"),
            ids("present"),
            ids("rejected"),
            ids("failed"),
            ids("missing"),
            complete,
            exported,
            error,
            topic_digests,
        )


@dataclass(frozen=True)
class _Listing:
    complete: tuple[str, ...]
    malformed: tuple[str, ...]


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CollectorError(f"collector state is not canonical JSON: {exc}") from exc


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("ab") as stream:
        stream.write(_canonical(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise CollectorError(f"cannot read collector state: {path}") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CollectorError(
                f"collector state line {line_number} is not JSON"
            ) from exc
        if not isinstance(value, dict):
            raise CollectorError(f"collector state line {line_number} is not an object")
        rows.append(value)
    return rows


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise CollectorError(f"{label} is not a safe path component")
    return value


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CollectorError(f"collector path is not a directory: {path}")
    os.chmod(path, 0o700)


@contextmanager
def _collector_lock(path: Path) -> Iterator[None]:
    _ensure_private_directory(path.parent)
    with path.open("a+b") as stream:
        os.chmod(path, 0o600)
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _plan_details(plan: object) -> tuple[object, tuple[str, ...], str]:
    if isinstance(plan, (str, Path)):
        from .agentic_run_state import load_run_plan

        path = Path(plan)
        plan = load_run_plan(path.parent if path.is_file() else path)
    if isinstance(plan, Mapping):
        ids = plan.get("planned_topic_ids")
        if ids is None:
            raw_topics = plan.get("topics")
            if isinstance(raw_topics, list):
                ids = tuple(
                    row.get("topic_id")
                    for row in raw_topics
                    if isinstance(row, Mapping)
                )
        digest = plan.get("plan_sha256")
    else:
        ids = getattr(plan, "planned_topic_ids", None)
        digest = getattr(plan, "plan_sha256", None)
    if isinstance(ids, str) or not isinstance(ids, Sequence) or not ids:
        raise CollectorError("frozen plan has no ordered topic IDs")
    topic_ids = tuple(_safe_component(item, "planned topic ID") for item in ids)
    if len(set(topic_ids)) != len(topic_ids):
        raise CollectorError("frozen plan repeats a topic ID")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        # Synthetic dependency-injected plans may use an opaque digest.  Real
        # AgenticRunPlan instances always use a lowercase SHA-256.
        if not digest or not isinstance(digest, str):
            raise CollectorError("frozen plan has no plan digest")
    return plan, topic_ids, digest


def _parse_listing(
    raw: str | bytes | Sequence[Mapping[str, object]],
    *,
    bucket_prefix: str,
) -> _Listing:
    if isinstance(raw, (str, bytes)):
        try:
            records = parse_hf_bucket_listing(raw)
        except HFListingError as exc:
            raise CollectorError(str(exc)) from exc
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        records = tuple(raw)
    else:
        raise CollectorError("Hugging Face listing must be JSON or records")

    prefix = bucket_prefix.rstrip("/")
    by_topic: dict[str, set[str]] = {}
    malformed: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise CollectorError("Hugging Face listing entries must be objects")
        raw_path = record.get("path")
        if not isinstance(raw_path, str):
            raise CollectorError("Hugging Face listing entry has no path")
        if raw_path == prefix:
            continue
        prefix_marker = prefix + "/"
        if raw_path.startswith(prefix_marker):
            relative = raw_path[len(prefix_marker) :]
        elif "://" not in prefix and "/" not in prefix and not raw_path.startswith("/"):
            relative = raw_path
        else:
            # A listing can include sibling paths.  They are not candidates
            # for this exact prefix and must never satisfy a topic selector.
            continue
        parts = tuple(relative.split("/"))
        if len(parts) != 2 or any(
            not part or part in {".", ".."} or _SAFE_COMPONENT.fullmatch(part) is None
            for part in parts
        ):
            if parts and parts[0]:
                malformed.add(parts[0])
            continue
        topic_id, name = parts
        if record.get("type") != "file":
            continue
        key = (topic_id, name)
        if key in seen:
            malformed.add(topic_id)
            continue
        seen.add(key)
        if name not in {_ARCHIVE_NAME, _MARKER_NAME}:
            malformed.add(topic_id)
            continue
        by_topic.setdefault(topic_id, set()).add(name)

    complete = tuple(
        sorted(
            topic_id
            for topic_id, names in by_topic.items()
            if names == {_ARCHIVE_NAME, _MARKER_NAME}
            and topic_id not in malformed
        )
    )
    return _Listing(complete=complete, malformed=tuple(sorted(malformed)))


def _marker_topic(marker_path: Path) -> str:
    try:
        value = json.loads(marker_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectorError("bundle completion marker is malformed") from exc
    if not isinstance(value, Mapping):
        raise CollectorError("bundle completion marker is not an object")
    return _safe_component(value.get("topic_id"), "bundle topic ID")


def _topic_manifest_path(run_dir: Path, topic_id: str) -> Path:
    return run_dir / "topics" / topic_id / "topic_projection_manifest.json"


def _is_present(run_dir: Path, topic_id: str, plan: object) -> bool:
    manifest = _topic_manifest_path(run_dir, topic_id)
    try:
        info = manifest.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CollectorError(f"topic seal is not a regular file: {manifest}")
    # A real AgenticRunPlan gives us the authenticated, full seal check.  Fakes
    # used by offline callers only need the manifest existence seam.
    if plan.__class__.__name__ == "AgenticRunPlan":
        try:
            from .agentic_run_state import load_topic_seal

            load_topic_seal(work_dir=run_dir, plan=plan, topic_id=topic_id)
        except Exception:
            return False
    return True


def _default_verify(archive: Path, marker: Path, expected_plan_sha256: str) -> object:
    try:
        from .agentic_retrieval_shard_bundle import verify_topic_bundle
    except ImportError as exc:
        raise CollectorError("topic bundle verifier is unavailable") from exc
    return verify_topic_bundle(archive, marker, expected_plan_sha256)


def _default_import(archive: Path, marker: Path, destination: Path) -> object:
    try:
        from .agentic_retrieval_shard_bundle import import_topic_bundle
    except ImportError as exc:
        raise CollectorError("topic bundle importer is unavailable") from exc
    return import_topic_bundle(archive, marker, destination)


def _default_summarize(destination: Path) -> Mapping[str, object]:
    try:
        from .agentic_retrieval_shard_bundle import summarize_staging
    except ImportError as exc:
        raise CollectorError("topic bundle status operation is unavailable") from exc
    return summarize_staging(destination)


def _verified_topic(value: object, marker: Path) -> str:
    if isinstance(value, Mapping):
        candidate = value.get("topic_id")
    else:
        candidate = getattr(value, "topic_id", None)
    if candidate is None:
        candidate = _marker_topic(marker)
    return _safe_component(candidate, "verified topic ID")


def _call_download(transport: object, prefix: str, destination: Path) -> None:
    method = getattr(transport, "download", None)
    if method is None and callable(transport):
        method = transport
    if method is None:
        raise CollectorError("HF transport has no download operation")
    try:
        method(prefix, destination)
    except TypeError:
        # A few small test/process adapters use ``download(topic_id, path)``.
        # Retry only the alternate transport shape; the caller still receives
        # the original exception for all other failures.
        method(prefix.rstrip("/").split("/")[-1], destination)


class IncrementalCollector:
    """Collect one wave at a time, or keep polling until the cohort is sealed."""

    def __init__(
        self,
        *,
        bucket_prefix: str,
        plan: object,
        staging_root: Path,
        transport: HFTransport | object | None = None,
        destination_run_dir: Path | None = None,
        poll_interval: float = 30.0,
        expected_topic_ids: Sequence[str] | None = None,
        verify_bundle: Callable[[Path, Path, str], object] | None = None,
        import_bundle: Callable[[Path, Path, Path], object] | None = None,
        summarize_staging: Callable[[Path], Mapping[str, object]] | None = None,
        listing_fn: Callable[[str], str | bytes | Sequence[Mapping[str, object]]] | None = None,
        download_fn: Callable[[str, Path], object] | None = None,
        export_fn: Callable[..., object] | None = None,
        export_output_dir: Path | None = None,
        producer_revision: str = "0" * 40,
    ) -> None:
        if not isinstance(bucket_prefix, str) or not bucket_prefix.strip("/"):
            raise CollectorError("bucket prefix is required")
        prefix_parts = bucket_prefix.split("://", 1)[-1].strip("/").split("/")
        if any(part in {"", ".", ".."} for part in prefix_parts):
            raise CollectorError("bucket prefix contains an unsafe path component")
        if poll_interval < 0:
            raise CollectorError("poll interval cannot be negative")
        self.bucket_prefix = bucket_prefix.rstrip("/")
        self.plan, self.cohort_topic_ids, self.plan_sha256 = _plan_details(plan)
        requested = self.cohort_topic_ids if expected_topic_ids is None else tuple(expected_topic_ids)
        if not requested:
            raise CollectorError("expected topic subset cannot be empty")
        if len(set(requested)) != len(requested):
            raise CollectorError("expected topic subset repeats an identity")
        unknown = set(requested) - set(self.cohort_topic_ids)
        if unknown:
            raise CollectorError(f"expected topic is outside frozen plan: {sorted(unknown)[0]}")
        self.expected_topic_ids = tuple(
            topic_id for topic_id in self.cohort_topic_ids if topic_id in set(requested)
        )
        self.staging_root = Path(staging_root)
        self.destination_run_dir = Path(destination_run_dir or staging_root)
        self.poll_interval = float(poll_interval)
        if transport is None:
            if listing_fn is None or download_fn is None:
                raise CollectorError("transport or listing/download functions are required")

            class _FunctionTransport:
                def list(self, prefix: str):
                    return listing_fn(prefix)

                def download(self, prefix: str, destination: Path):
                    return download_fn(prefix, destination)

            transport = _FunctionTransport()
        self.transport = transport
        self.verify_bundle = verify_bundle or _default_verify
        self.import_bundle = import_bundle or _default_import
        self.summarize_staging = summarize_staging or _default_summarize
        self.export_fn = export_fn
        self._uses_default_export = export_fn is None
        self.export_output_dir = Path(export_output_dir or self.destination_run_dir)
        self.producer_revision = producer_revision
        self._topic_digests: dict[str, Mapping[str, str]] = {}
        _ensure_private_directory(self.staging_root)
        _ensure_private_directory(self.destination_run_dir)
        _ensure_private_directory(self.staging_root / QUARANTINE_DIRNAME)
        _ensure_private_directory(self.staging_root / INCOMING_DIRNAME)

    @property
    def wave_receipts_path(self) -> Path:
        return self.staging_root / WAVE_RECEIPTS_FILENAME

    @property
    def journal_path(self) -> Path:
        return self.staging_root / COLLECTOR_JOURNAL_FILENAME

    def _next_wave(self) -> int:
        rows = _read_jsonl(self.wave_receipts_path)
        if not rows:
            return 1
        receipts = [WaveReceipt.from_payload(row) for row in rows]
        numbers = [receipt.wave for receipt in receipts]
        if len(set(numbers)) != len(numbers) or numbers != list(range(1, len(numbers) + 1)):
            raise CollectorError("collector wave receipts are not append-only")
        return len(receipts) + 1

    def _present_ids(self) -> set[str]:
        summary_ids: set[str] | None = None
        try:
            summary = self.summarize_staging(self.destination_run_dir)
            completed = summary.get("completed_topic_ids")
            if isinstance(completed, list) and all(isinstance(item, str) for item in completed):
                summary_ids = set(completed)
        except (CollectorError, OSError):
            # A synthetic offline plan may not have a bundle-status provider;
            # the immutable topic seal path remains a safe fallback seam.
            summary_ids = None
        return {
            topic_id
            for topic_id in self.cohort_topic_ids
            if (summary_ids is None or topic_id in summary_ids)
            and _is_present(self.destination_run_dir, topic_id, self.plan)
        }

    def _quarantine(self, source: Path | None, topic_id: str, wave: int) -> None:
        name = f"{topic_id}.{wave}.{uuid.uuid4().hex}"
        destination = self.staging_root / QUARANTINE_DIRNAME / name
        if source is None:
            destination.mkdir(mode=0o700)
            return
        try:
            shutil.move(str(source), str(destination))
        except OSError:
            destination.mkdir(mode=0o700)
            (destination / "quarantine-error.txt").write_text("bundle could not be moved\n")

    def _journal(self, wave: int, topic_id: str, status: str, error: str | None = None) -> None:
        payload: dict[str, object] = {
            "schema_version": _SCHEMA,
            "wave": wave,
            "topic_id": topic_id,
            "status": status,
        }
        if error is not None:
            payload["error"] = error
        _append_jsonl(self.journal_path, payload)

    def _export_receipt_valid(self) -> bool:
        path = self.staging_root / EXPORT_RECEIPT_FILENAME
        if not path.exists():
            return False
        try:
            rows = _read_jsonl(path)
        except CollectorError:
            return False
        if len(rows) != 1:
            return False
        row = rows[0]
        valid = (
            row.get("schema_version") == _EXPORT_SCHEMA
            and type(row.get("wave")) is int
            and row["wave"] > 0
            and row.get("topic_ids") == list(self.cohort_topic_ids)
        )
        if not valid:
            return False
        if self.plan.__class__.__name__ == "AgenticRunPlan":
            try:
                from .agentic_retrieval_export import load_agentic_retrieval_export

                load_agentic_retrieval_export(
                    output_dir=self.export_output_dir,
                    work_dir=self.destination_run_dir,
                    plan=self.plan,
                )
            except Exception:
                return False
        return True

    def _process_topic_unlocked(self, topic_id: str, wave: int, *, rejected: bool = False) -> str:
        incoming = self.staging_root / INCOMING_DIRNAME / f"{topic_id}.{wave}.{uuid.uuid4().hex}"
        incoming.mkdir(mode=0o700)
        remote_prefix = f"{self.bucket_prefix}/{topic_id}"
        try:
            _call_download(self.transport, remote_prefix, incoming)
        except Exception as exc:
            self._quarantine(incoming, topic_id, wave)
            self._journal(wave, topic_id, "failed", str(exc))
            return "failed"
        archive = incoming / _ARCHIVE_NAME
        marker = incoming / _MARKER_NAME
        try:
            if not archive.is_file() or not marker.is_file():
                raise CollectorError("downloaded topic bundle is incomplete")
            if rejected:
                raise CollectorError("bundle is outside the requested cohort or listing contract")
            verified = self.verify_bundle(archive, marker, self.plan_sha256)
            if _verified_topic(verified, marker) != topic_id:
                raise CollectorError("verified bundle topic differs from remote path")
            seal_digest = (
                verified.get("topic_seal_sha256")
                if isinstance(verified, Mapping)
                else getattr(verified, "topic_seal_sha256", None)
            )
            if not isinstance(seal_digest, str):
                try:
                    marker_payload = json.loads(marker.read_bytes())
                    seal_digest = marker_payload.get("topic_seal_sha256")
                except (OSError, json.JSONDecodeError):
                    seal_digest = None
            if not isinstance(seal_digest, str):
                raise CollectorError("verified bundle has no topic-seal digest")
            self._topic_digests[topic_id] = {
                "plan_sha256": self.plan_sha256,
                "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
                "marker_sha256": sha256(marker.read_bytes()).hexdigest(),
                "topic_seal_sha256": seal_digest,
            }
        except Exception as exc:
            self._quarantine(incoming, topic_id, wave)
            self._journal(wave, topic_id, "rejected", str(exc))
            return "rejected"
        try:
            with _collector_lock(self.export_output_dir / ".agentic-retrieval-export.lock"):
                self.import_bundle(archive, marker, self.destination_run_dir)
                if not _is_present(self.destination_run_dir, topic_id, self.plan):
                    raise CollectorError("import completed without a valid topic seal")
        except Exception as exc:
            self._quarantine(incoming, topic_id, wave)
            self._journal(wave, topic_id, "failed", str(exc))
            return "failed"
        self._journal(wave, topic_id, "imported")
        return "imported"

    def _process_topic(self, topic_id: str, wave: int, *, rejected: bool = False) -> str:
        # Download and verify remain outside the shared mutation lock.  The
        # unlocked helper takes that lock only around import and seal validation.
        return self._process_topic_unlocked(topic_id, wave, rejected=rejected)

    def _export_complete_cohort(self, wave: int) -> tuple[bool, str | None]:
        if not self._uses_default_export and self.export_fn is None:
            return False, None
        if self._export_receipt_valid():
            return True, None
        status = self.status()
        try:
            # Injected fakes receive a compact status mapping.  The production
            # exporter receives its documented keyword interface.
            if self._uses_default_export:
                from .agentic_retrieval_export import publish_agentic_retrieval_export

                publish_agentic_retrieval_export(
                    output_dir=self.export_output_dir,
                    work_dir=self.destination_run_dir,
                    plan=self.plan,
                    producer_revision=self.producer_revision,
                )
            elif getattr(self.export_fn, "__module__", "") == "trec_rag.agentic_retrieval_export":
                self.export_fn(
                    output_dir=self.export_output_dir,
                    work_dir=self.destination_run_dir,
                    plan=self.plan,
                    producer_revision=self.producer_revision,
                )
            else:
                self.export_fn(status)
        except Exception as exc:
            return False, str(exc)
        _append_jsonl(
            self.staging_root / EXPORT_RECEIPT_FILENAME,
            {
                "schema_version": _EXPORT_SCHEMA,
                "wave": wave,
                "topic_ids": list(self.cohort_topic_ids),
            },
        )
        return True, None

    def run_once(self) -> WaveReceipt:
        """Perform one listing, download, verify, import, and receipt cycle."""
        with _collector_lock(self.destination_run_dir / COLLECTOR_LOCK_FILENAME):
            wave = self._next_wave()
            self._topic_digests = {}
            present_before = self._present_ids()
            try:
                raw_listing = self.transport.list(self.bucket_prefix)  # type: ignore[attr-defined]
                listing = _parse_listing(raw_listing, bucket_prefix=self.bucket_prefix)
            except Exception as exc:
                remaining = tuple(
                    topic_id for topic_id in self.expected_topic_ids if topic_id not in present_before
                )
                receipt = WaveReceipt(
                    wave, (), tuple(self.expected_topic_ids[i] for i in range(len(self.expected_topic_ids)) if self.expected_topic_ids[i] in present_before), (), remaining, remaining, False, False, str(exc)
                )
                _append_jsonl(self.wave_receipts_path, receipt.to_payload())
                return receipt

            imported: list[str] = []
            present: list[str] = [
                topic_id for topic_id in self.expected_topic_ids if topic_id in present_before
            ]
            rejected: list[str] = []
            failed: list[str] = []
            handled = set(present)

            # Complete bundles outside the frozen cohort are retained only in
            # quarantine.  Exact component matching means ``14`` cannot match
            # ``144``.
            for topic_id in sorted(set(listing.complete) | set(listing.malformed)):
                if topic_id in handled:
                    continue
                if _SAFE_COMPONENT.fullmatch(topic_id) is None:
                    rejected.append(topic_id)
                    self._quarantine(None, "malformed-path", wave)
                    self._journal(wave, "malformed-path", "rejected", "unsafe listing topic component")
                    handled.add(topic_id)
                    continue
                if topic_id in self.cohort_topic_ids and topic_id not in self.expected_topic_ids:
                    # A caller-provided subset bounds this collector's work;
                    # leave other valid cohort topics for their own wave.
                    continue
                if topic_id not in self.expected_topic_ids:
                    outcome = self._process_topic(topic_id, wave, rejected=True)
                    if outcome in {"rejected", "failed"}:
                        (rejected if outcome == "rejected" else failed).append(topic_id)
                    handled.add(topic_id)
                    continue
                if topic_id in listing.malformed:
                    outcome = self._process_topic(topic_id, wave, rejected=True)
                else:
                    outcome = self._process_topic(topic_id, wave)
                if outcome == "imported":
                    imported.append(topic_id)
                elif outcome == "rejected":
                    rejected.append(topic_id)
                else:
                    failed.append(topic_id)
                handled.add(topic_id)

            missing = [
                topic_id
                for topic_id in self.expected_topic_ids
                if topic_id not in handled
            ]
            complete = set(self._present_ids()) == set(self.cohort_topic_ids)
            exported = False
            export_error: str | None = None
            if complete:
                exported, export_error = self._export_complete_cohort(wave)
            receipt = WaveReceipt(
                wave,
                tuple(imported),
                tuple(present),
                tuple(rejected),
                tuple(failed),
                tuple(missing),
                complete,
                exported,
                export_error,
                dict(self._topic_digests),
            )
            _append_jsonl(self.wave_receipts_path, receipt.to_payload())
            return receipt

    def status(self) -> dict[str, object]:
        present = tuple(
            topic_id for topic_id in self.cohort_topic_ids if _is_present(self.destination_run_dir, topic_id, self.plan)
        )
        exported = (self.staging_root / EXPORT_RECEIPT_FILENAME).exists()
        return {
            "topic_ids": list(self.cohort_topic_ids),
            "present": list(present),
            "missing": [topic_id for topic_id in self.cohort_topic_ids if topic_id not in present],
            "complete": len(present) == len(self.cohort_topic_ids),
            "wave_count": len(_read_jsonl(self.wave_receipts_path)),
            "exported": exported,
        }

    def watch(self, *, max_cycles: int | None = None) -> Iterator[WaveReceipt]:
        """Yield bounded cycles until the complete frozen cohort is present."""
        if max_cycles is not None and max_cycles <= 0:
            return
        cycle = 0
        while max_cycles is None or cycle < max_cycles:
            try:
                receipt = self.run_once()
            except KeyboardInterrupt:
                return
            yield receipt
            cycle += 1
            if receipt.complete:
                return
            if max_cycles is None or cycle < max_cycles:
                try:
                    time.sleep(self.poll_interval)
                except KeyboardInterrupt:
                    return


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket-prefix", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--destination-run-dir", type=Path)
    parser.add_argument("--topic", action="append", dest="topic_ids")
    parser.add_argument("--interval", type=float, default=30.0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--hf", default="hf", help="hf executable")
    return parser


class _CliTransport:
    def __init__(self, executable: str) -> None:
        self.executable = executable

    def list(self, prefix: str) -> bytes:
        result = subprocess.run(
            [self.executable, "buckets", "list", prefix, "--format", "json"],
            check=True,
            capture_output=True,
        )
        return result.stdout

    def download(self, prefix: str, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        subprocess.run(
            [self.executable, "buckets", "sync", f"{prefix}/", str(destination)],
            check=True,
            capture_output=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        from .agentic_run_state import load_run_plan

        plan = load_run_plan(args.plan if args.plan.is_dir() else args.plan.parent)
        collector = IncrementalCollector(
            bucket_prefix=args.bucket_prefix,
            plan=plan,
            staging_root=args.staging_root,
            destination_run_dir=args.destination_run_dir,
            transport=_CliTransport(args.hf),
            poll_interval=args.interval,
            expected_topic_ids=args.topic_ids,
        )
        receipts = [collector.run_once()] if args.once else collector.watch()
        for receipt in receipts:
            print(json.dumps(receipt.to_payload(), ensure_ascii=False, sort_keys=True))
        return 0
    except (CollectorError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLLECTOR_JOURNAL_FILENAME",
    "COLLECTOR_LOCK_FILENAME",
    "EXPORT_RECEIPT_FILENAME",
    "IncrementalCollector",
    "CollectorError",
    "HFTransport",
    "WAVE_RECEIPTS_FILENAME",
    "WaveReceipt",
    "main",
]
