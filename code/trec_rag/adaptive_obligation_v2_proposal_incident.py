"""Authenticate and freeze the immutable aborted R1 proposal attempt."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from .adaptive_obligation_v2_ledger import LEDGER_SCHEMA_VERSION, classify_completion
from .adaptive_obligation_v2_propose import (
    PRIMARY_JOB_COUNT,
    PRIMARY_MAX_NEW_TOKENS,
    PROPOSAL_SCHEMA,
    _build_run_anchor,
    _capture_and_verify_preflight,
    _capture_inference_approval,
    _compact_bytes,
    _open_directory_no_symlinks,
    _pretty_bytes,
    _read_stable_regular_at,
    verify_proposal_preflight,
)


INCIDENT_SCHEMA_VERSION = "adaptive-obligation-v2-proposal-incident-r1"
EXPECTED_JOB_COUNT = 48
R1_FAILED_LEDGER_ROOT_NAMES = frozenset(
    {"events.jsonl", "head.json", "anchor.json", "mutation.lock", "raw"}
)
R1_RAW_BYTES = 546
R1_RAW_SHA256 = "683174f0ca3353b8dc901ec889974d83b2c02bac9cc79d00db6a08f26e4bb667"
R1_RAW_PATH = (
    "87df2b737285e7243ba3726dc621bf207164cb614bf548f7b3bad2f9d661ba5b"
    ".1.completion"
)
R1_OUTPUT_TOKENS = 186

_STARTED_EVENT_KEYS = frozenset(
    {
        "schema_version",
        "sequence",
        "previous_event_sha256",
        "event_sha256",
        "state",
        "stage",
        "job_id",
        "attempt_ordinal",
        "request_sha256",
        "max_new_tokens",
    }
)
_TERMINAL_EVENT_KEYS = _STARTED_EVENT_KEYS | {
    "classification",
    "output_token_count",
    "raw_path",
    "raw_bytes",
    "raw_sha256",
}
_HEAD_KEYS = frozenset({"schema_version", "event_count", "head_sha256"})
_ZERO_SHA256 = "0" * 64


def _sha256(source: bytes) -> str:
    return sha256(source).hexdigest()


def _directory_identity(observed: os.stat_result) -> tuple[int, ...]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _canonical_object(source: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or source != _compact_bytes(value):
        raise ValueError(f"{label} is not canonical")
    return value


def _canonical_events(source: bytes) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in source.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("R1 failed ledger event is invalid JSON") from exc
        if not isinstance(value, dict) or line != _compact_bytes(value):
            raise ValueError("R1 failed ledger event is not canonical")
        events.append(value)
    return events


def _anchor_attempts(
    anchor: Mapping[str, object],
) -> dict[tuple[str, int], dict[str, object]]:
    jobs = anchor.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("R1 failed ledger anchor inventory differs")
    attempts: dict[tuple[str, int], dict[str, object]] = {}
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("attempts"), list):
            raise ValueError("R1 failed ledger anchor inventory differs")
        job_id = job.get("job_id")
        if not isinstance(job_id, str):
            raise ValueError("R1 failed ledger anchor inventory differs")
        for attempt in job["attempts"]:
            if not isinstance(attempt, dict):
                raise ValueError("R1 failed ledger anchor inventory differs")
            ordinal = attempt.get("attempt_ordinal")
            if type(ordinal) is not int or (job_id, ordinal) in attempts:
                raise ValueError("R1 failed ledger anchor inventory differs")
            attempts[(job_id, ordinal)] = attempt
    return attempts


def _verify_failed_ledger_material(
    *,
    anchor_source: bytes,
    head_source: bytes,
    events_source: bytes,
    raw_name: str,
    raw: bytes,
) -> dict[str, object]:
    anchor = _canonical_object(anchor_source, "R1 failed ledger anchor")
    head = _canonical_object(head_source, "R1 failed ledger head")
    events = _canonical_events(events_source)
    attempts = _anchor_attempts(anchor)
    if set(head) != _HEAD_KEYS or head.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ValueError("R1 failed ledger head differs")

    previous = _ZERO_SHA256
    pending: set[tuple[str, int]] = set()
    completed: set[tuple[str, int]] = set()
    raw_names: set[str] = set()
    for sequence, event in enumerate(events, start=1):
        state = event.get("state")
        expected_keys = (
            _STARTED_EVENT_KEYS if state == "started" else _TERMINAL_EVENT_KEYS
        )
        payload = {key: value for key, value in event.items() if key != "event_sha256"}
        claimed = event.get("event_sha256")
        if (
            set(event) != expected_keys
            or event.get("schema_version") != LEDGER_SCHEMA_VERSION
            or event.get("sequence") != sequence
            or event.get("previous_event_sha256") != previous
            or claimed != _sha256(_compact_bytes(payload))
        ):
            raise ValueError("R1 failed ledger event hash chain differs")
        previous = str(claimed)

        job_id = event.get("job_id")
        ordinal = event.get("attempt_ordinal")
        if (
            not isinstance(job_id, str)
            or type(ordinal) is not int
            or ordinal != 1
            or event.get("max_new_tokens") != PRIMARY_MAX_NEW_TOKENS
        ):
            raise ValueError("R1 failed ledger observed primary attempt differs")
        key = (job_id, ordinal)
        anchored = attempts.get(key)
        if (
            anchored is None
            or event.get("stage") != anchor.get("stage")
            or event.get("request_sha256") != anchored.get("request_sha256")
            or event.get("max_new_tokens") != anchored.get("max_new_tokens")
        ):
            raise ValueError("R1 failed ledger attempt differs from anchor")
        if state == "started":
            if key in pending or key in completed:
                raise ValueError("R1 failed ledger attempt lifecycle differs")
            pending.add(key)
            continue
        if state != "terminal" or key not in pending:
            raise ValueError("R1 failed ledger attempt lifecycle differs")
        raw_path = event.get("raw_path")
        expected_raw_name = f"{job_id}.{ordinal}.completion"
        if (
            raw_path != expected_raw_name
            or raw_path != R1_RAW_PATH
            or raw_path != raw_name
            or raw_path in raw_names
            or type(event.get("output_token_count")) is not int
            or type(event.get("raw_bytes")) is not int
            or event.get("raw_bytes") != len(raw)
            or event.get("raw_sha256") != _sha256(raw)
        ):
            raise ValueError("R1 failed ledger raw binding differs")
        pending.remove(key)
        completed.add(key)
        raw_names.add(raw_path)

    if pending:
        raise ValueError("R1 failed ledger contains an incomplete attempt")
    if raw_names != {raw_name}:
        raise ValueError("R1 failed ledger raw inventory differs")
    if (
        type(head.get("event_count")) is not int
        or head.get("event_count") != len(events)
        or head.get("head_sha256") != previous
    ):
        raise ValueError("R1 failed ledger head differs")
    return {
        "anchor": anchor,
        "anchor_source": anchor_source,
        "head": head,
        "head_source": head_source,
        "events": events,
        "events_source": events_source,
        "raw_name": raw_name,
        "raw": raw,
    }


def _capture_failed_r1_ledger(ledger_dir: Path) -> dict[str, object]:
    """Capture and verify the unsealed R1 ledger without opening it for mutation."""

    root_fd = -1
    raw_fd = -1
    try:
        root_fd = _open_directory_no_symlinks(Path(ledger_dir))
        root_before = os.fstat(root_fd)
        names_before = set(os.listdir(root_fd))
        if names_before != R1_FAILED_LEDGER_ROOT_NAMES:
            raise OSError("failed ledger root inventory differs")
        anchor_source = _read_stable_regular_at(
            root_fd, "anchor.json", require_single_link=True
        )
        head_source = _read_stable_regular_at(
            root_fd, "head.json", require_single_link=True
        )
        events_source = _read_stable_regular_at(
            root_fd, "events.jsonl", require_single_link=True
        )
        lock_source = _read_stable_regular_at(
            root_fd, "mutation.lock", require_single_link=True
        )
        if lock_source:
            raise OSError("failed ledger lock differs")

        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        raw_fd = os.open("raw", flags, dir_fd=root_fd)
        raw_before = os.fstat(raw_fd)
        if not stat.S_ISDIR(raw_before.st_mode):
            raise OSError("failed ledger raw path is not a directory")
        raw_names_before = os.listdir(raw_fd)
        if len(raw_names_before) != 1:
            raise OSError("failed ledger raw inventory differs")
        raw_name = raw_names_before[0]
        raw = _read_stable_regular_at(raw_fd, raw_name, require_single_link=True)
        raw_names_after = os.listdir(raw_fd)
        raw_after = os.fstat(raw_fd)
        if (
            raw_names_after != raw_names_before
            or _directory_identity(raw_after) != _directory_identity(raw_before)
        ):
            raise OSError("failed ledger raw directory changed while captured")

        names_after = set(os.listdir(root_fd))
        root_after = os.fstat(root_fd)
        if (
            names_after != names_before
            or _directory_identity(root_after) != _directory_identity(root_before)
        ):
            raise OSError("failed ledger root changed while captured")
    except (OSError, ValueError) as exc:
        raise ValueError("R1 failed ledger is missing, changed, or unsafe") from exc
    finally:
        if raw_fd >= 0:
            os.close(raw_fd)
        if root_fd >= 0:
            os.close(root_fd)
    try:
        return _verify_failed_ledger_material(
            anchor_source=anchor_source,
            head_source=head_source,
            events_source=events_source,
            raw_name=raw_name,
            raw=raw,
        )
    except ValueError as exc:
        if str(exc).startswith("R1 failed ledger"):
            raise
        raise ValueError("R1 failed ledger differs") from exc


def _prove_rationale_length_is_sole_defect(
    raw: bytes, allowed_support_unit_ids: Sequence[str]
) -> str:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("rationale length is not the sole schema defect") from exc
    if not isinstance(value, dict) or not isinstance(value.get("o1"), dict):
        raise ValueError("rationale length is not the sole schema defect")
    rationale = value["o1"].get("scope_rationale")
    if not isinstance(rationale, str) or len(rationale) != 245:
        raise ValueError("rationale length is not the sole schema defect")
    original = classify_completion(
        raw,
        schema=PROPOSAL_SCHEMA,
        output_token_count=R1_OUTPUT_TOKENS,
        max_new_tokens=PRIMARY_MAX_NEW_TOKENS,
        allowed_support_unit_ids=allowed_support_unit_ids,
    )
    diagnostic_value = deepcopy(value)
    diagnostic_value["o1"]["scope_rationale"] = "Within the supported scope."
    diagnostic_raw = _compact_bytes(diagnostic_value).rstrip(b"\n")
    diagnostic = classify_completion(
        diagnostic_raw,
        schema=PROPOSAL_SCHEMA,
        output_token_count=R1_OUTPUT_TOKENS,
        max_new_tokens=PRIMARY_MAX_NEW_TOKENS,
        allowed_support_unit_ids=allowed_support_unit_ids,
    )
    if (
        original.get("classification") != "schema_error"
        or diagnostic.get("classification") != "valid"
    ):
        raise ValueError("rationale length is not the sole schema defect")
    return rationale


def _file_binding(path: Path, source: bytes) -> dict[str, object]:
    return {"path": str(path), "bytes": len(source), "sha256": _sha256(source)}


def _failed_ledger_binding(
    ledger_dir: Path, failed: Mapping[str, object]
) -> dict[str, object]:
    return {
        "path": str(ledger_dir),
        "anchor": _file_binding(
            Path(ledger_dir) / "anchor.json", failed["anchor_source"]  # type: ignore[arg-type]
        ),
        "head": _file_binding(
            Path(ledger_dir) / "head.json", failed["head_source"]  # type: ignore[arg-type]
        ),
        "events": _file_binding(
            Path(ledger_dir) / "events.jsonl", failed["events_source"]  # type: ignore[arg-type]
        ),
        "event_count": len(failed["events"]),  # type: ignore[arg-type]
    }


def _raw_binding(terminal: Mapping[str, object], raw: bytes) -> dict[str, object]:
    return {
        "path": str(Path("raw") / str(terminal["raw_path"])),
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def build_r1_incident_receipt(
    *, preflight_dir: Path, approval_path: Path, ledger_dir: Path
) -> dict[str, object]:
    approval = _capture_inference_approval(Path(approval_path))
    captured = _capture_and_verify_preflight(
        Path(preflight_dir),
        expected_receipt_sha256=str(approval.value["preflight_sha256"]),
        verifier=verify_proposal_preflight,
    )
    if (
        len(captured.jobs) != EXPECTED_JOB_COUNT
        or EXPECTED_JOB_COUNT != PRIMARY_JOB_COUNT
    ):
        raise ValueError("R1 incident job inventory differs")
    anchor = _build_run_anchor(
        captured.jobs,
        preflight_sha256=captured.receipt_sha256,
        approval_sha256=approval.sha256,
    )
    failed = _capture_failed_r1_ledger(Path(ledger_dir))
    if failed["anchor"] != anchor:
        raise ValueError("R1 incident anchor differs")
    events = failed["events"]
    if not isinstance(events, list) or len(events) != 2:
        raise ValueError("R1 incident event count differs")
    started, terminal = events
    if (
        started.get("state") != "started"
        or terminal.get("state") != "terminal"
        or terminal.get("classification") != "schema_error"
        or started.get("job_id") != captured.jobs[0]["job_id"]
        or terminal.get("job_id") != captured.jobs[0]["job_id"]
    ):
        raise ValueError("R1 incident terminal state differs")
    raw = failed["raw"]
    if not isinstance(raw, bytes):
        raise ValueError("R1 incident raw identity differs")
    if (
        terminal.get("raw_bytes") != R1_RAW_BYTES
        or terminal.get("raw_sha256") != R1_RAW_SHA256
        or terminal.get("output_token_count") != R1_OUTPUT_TOKENS
        or len(raw) != R1_RAW_BYTES
        or sha256(raw).hexdigest() != R1_RAW_SHA256
    ):
        raise ValueError("R1 incident raw identity differs")
    allowed_ids = captured.jobs[0].get("input_unit_ids")
    if not isinstance(allowed_ids, list):
        raise ValueError("R1 incident support inventory differs")
    rationale = _prove_rationale_length_is_sole_defect(raw, allowed_ids)
    return {
        "schema_version": INCIDENT_SCHEMA_VERSION,
        "status": "aborted",
        "reason_code": "scope_rationale_length_exceeded",
        "attempted_job_count": 1,
        "terminal_schema_error_count": 1,
        "uncalled_job_count": 47,
        "observed_rationale_characters": len(rationale),
        "accepted_rationale_maximum": 240,
        "output_token_count": R1_OUTPUT_TOKENS,
        "preflight": _file_binding(
            Path(preflight_dir) / "receipt.json", captured.contents["receipt.json"]
        ),
        "approval": _file_binding(Path(approval_path), approval.source),
        "ledger": _failed_ledger_binding(Path(ledger_dir), failed),
        "raw_completion": _raw_binding(terminal, raw),
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "qrels_opened": False,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic create-only rename is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(f"create-only R1 incident exists: {destination}")
    raise OSError(error, os.strerror(error), str(destination))


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def publish_r1_incident(
    *,
    preflight_dir: Path,
    approval_path: Path,
    ledger_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    receipt = build_r1_incident_receipt(
        preflight_dir=Path(preflight_dir),
        approval_path=Path(approval_path),
        ledger_dir=Path(ledger_dir),
    )
    destination = Path(output_dir)
    if _path_present(destination):
        raise FileExistsError(f"create-only R1 incident exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ValueError("R1 incident parent is unsafe")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    published = False
    try:
        receipt_path = staging / "receipt.json"
        with receipt_path.open("xb") as sink:
            sink.write(_pretty_bytes(receipt))
            sink.flush()
            os.fsync(sink.fileno())
        entries = list(staging.iterdir())
        observed = receipt_path.stat(follow_symlinks=False)
        if (
            {entry.name for entry in entries} != {"receipt.json"}
            or receipt_path.is_symlink()
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or receipt_path.read_bytes() != _pretty_bytes(receipt)
        ):
            raise ValueError("staged R1 incident receipt differs")
        _fsync_directory(staging)
        _rename_noreplace(staging, destination)
        published = True
        _fsync_directory(destination.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return receipt


def _capture_published_receipt(output_dir: Path) -> bytes:
    directory_fd = -1
    try:
        directory_fd = _open_directory_no_symlinks(Path(output_dir))
        before = os.fstat(directory_fd)
        names_before = set(os.listdir(directory_fd))
        if names_before != {"receipt.json"}:
            raise OSError("incident inventory differs")
        source = _read_stable_regular_at(
            directory_fd, "receipt.json", require_single_link=True
        )
        names_after = set(os.listdir(directory_fd))
        after = os.fstat(directory_fd)
        if (
            names_after != names_before
            or _directory_identity(after) != _directory_identity(before)
        ):
            raise OSError("incident directory changed while captured")
        return source
    except OSError as exc:
        raise ValueError("R1 incident receipt differs") from exc
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def verify_r1_incident(
    *,
    preflight_dir: Path,
    approval_path: Path,
    ledger_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    expected = build_r1_incident_receipt(
        preflight_dir=Path(preflight_dir),
        approval_path=Path(approval_path),
        ledger_dir=Path(ledger_dir),
    )
    if _capture_published_receipt(Path(output_dir)) != _pretty_bytes(expected):
        raise ValueError("R1 incident receipt differs")
    return expected
