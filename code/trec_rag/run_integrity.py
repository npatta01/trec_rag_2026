"""Filesystem integrity helpers for resumable long-running score jobs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
from typing import Any


class ExclusiveFileLock:
    """A small O_EXCL-backed writer marker for one shared filesystem path."""

    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path
        self.payload = payload
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(
                f"run already has an active writer marker: {self.path}"
            ) from exc
        try:
            encoded = (json.dumps(self.payload, sort_keys=True) + "\n").encode("utf-8")
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        self.path.unlink(missing_ok=True)
        self.acquired = False


class ThreadSafeJsonStatusWriter:
    """Serialize atomic JSON status replacement across worker threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        with self._lock:
            temporary_path = self.path.with_name(
                f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                temporary_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_path, self.path)
            finally:
                temporary_path.unlink(missing_ok=True)


def validate_or_repair_final_jsonl(
    path: Path,
    *,
    archive_dir: Path,
) -> dict[str, object]:
    """Validate JSONL, truncating only a non-newline final fragment.

    Every newline-terminated row is parsed before any mutation.  Therefore an
    invalid earlier row is rejected.  A final non-newline byte sequence is
    archived with its digest and then removed; a resume can deterministically
    recreate that row from its inputs/cache.
    """

    if not path.exists():
        return {"path": str(path), "state": "missing", "repaired": False}
    data = path.read_bytes()
    complete_end = len(data) if data.endswith(b"\n") else data.rfind(b"\n") + 1
    complete = data[:complete_end]
    fragment = data[complete_end:]

    for line_number, line in enumerate(complete.splitlines(), start=1):
        if not line:
            raise ValueError(f"{path}:{line_number}: blank JSONL row")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid earlier JSONL row"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row must be an object")

    if not fragment:
        return {
            "path": str(path),
            "state": "valid",
            "repaired": False,
            "rows": complete.count(b"\n"),
        }

    archive_dir.mkdir(parents=True, exist_ok=True)
    fragment_digest = hashlib.sha256(fragment).hexdigest()
    archive_path = archive_dir / f"{path.name}.{fragment_digest[:16]}.torn-fragment"
    if archive_path.exists():
        raise FileExistsError(f"torn-fragment archive already exists: {archive_path}")
    archive_path.write_bytes(fragment)
    with path.open("r+b") as destination:
        destination.truncate(complete_end)
        destination.flush()
        os.fsync(destination.fileno())
    return {
        "path": str(path),
        "state": "repaired_torn_final_fragment",
        "repaired": True,
        "rows_retained": complete.count(b"\n"),
        "bytes_removed": len(fragment),
        "fragment_sha256": fragment_digest,
        "fragment_archive": str(archive_path),
    }
