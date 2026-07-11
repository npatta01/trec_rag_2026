"""Experiment-global external-call tickets shared by linked worktrees."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from trec_rag.det_sparse_config import EXPERIMENT_ID, RETRIEVAL_ENDPOINT_URL
from trec_rag.det_sparse_ledger import (
    RetrievalRequest,
    RetrievalRequestIdentity,
    RetrievalTransport,
)


GLOBAL_BUDGET_SCHEMA_VERSION = "det_sparse_global_external_budget_v1"


def global_budget_dir(repo_root: Path) -> Path:
    """Return one budget root shared by every linked worktree of this checkout."""

    completed = subprocess.run(
        ("git", "rev-parse", "--path-format=absolute", "--git-common-dir"),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    common = Path(completed.stdout.strip()).resolve()
    if not common.is_dir():
        raise ValueError("git common directory is unavailable for global budget state")
    return common / "trec-rag-external-budgets" / EXPERIMENT_ID


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _create_only(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as sink:
            sink.write(_canonical_bytes(value))
            sink.flush()
            os.fsync(sink.fileno())
    finally:
        os.close(descriptor)


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"global budget artifact is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"global budget artifact is not an object: {path}")
    return value


class GlobalExternalBudget:
    """Durably burn a ticket before each transport entry across worktrees."""

    def __init__(self, root: Path, *, max_calls: int = 36, max_per_topic: int = 9):
        if max_calls != 36 or max_per_topic != 9:
            raise ValueError("global budget is frozen at 36 total and 9 per topic")
        self.root = root.resolve()
        self.tickets_dir = self.root / "tickets"
        self.max_calls = max_calls
        self.max_per_topic = max_per_topic
        self.tickets_dir.mkdir(parents=True, exist_ok=True)
        expected = {
            "schema_version": GLOBAL_BUDGET_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "endpoint_url": RETRIEVAL_ENDPOINT_URL,
            "max_calls": max_calls,
            "max_calls_per_topic": max_per_topic,
        }
        manifest_path = self.root / "budget.json"
        try:
            _create_only(manifest_path, expected)
        except FileExistsError:
            if _read_object(manifest_path) != expected:
                raise ValueError("global budget policy differs from frozen protocol")

    class _Lock:
        def __init__(self, path: Path):
            self.path = path
            self.descriptor: int | None = None

        def __enter__(self):
            self.descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self.descriptor, fcntl.LOCK_EX)

        def __exit__(self, exc_type, exc, traceback):
            assert self.descriptor is not None
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)

    def _tickets(self) -> list[dict[str, object]]:
        rows = []
        for path in sorted(self.tickets_dir.glob("*.json")):
            row = _read_object(path)
            key = row.get("request_key")
            identity = row.get("identity")
            if (
                set(row)
                != {
                    "schema_version",
                    "experiment_id",
                    "request_key",
                    "identity",
                    "query_text",
                    "global_ordinal",
                    "reserved_at",
                }
                or row.get("schema_version") != GLOBAL_BUDGET_SCHEMA_VERSION
                or row.get("experiment_id") != EXPERIMENT_ID
                or not isinstance(key, str)
                or path.name != f"{key}.json"
                or not isinstance(identity, Mapping)
                or identity.get("index_url") != RETRIEVAL_ENDPOINT_URL
                or type(row.get("global_ordinal")) is not int
                or not isinstance(row.get("reserved_at"), str)
            ):
                raise ValueError(f"global budget ticket is malformed: {path}")
            try:
                typed_identity = RetrievalRequestIdentity(**dict(identity))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"global budget ticket identity is invalid: {path}") from exc
            query_text = row.get("query_text")
            if (
                typed_identity.request_key != key
                or not isinstance(query_text, str)
                or hashlib.sha256(query_text.encode("utf-8")).hexdigest()
                != typed_identity.query_sha256
            ):
                raise ValueError(f"global budget ticket query identity differs: {path}")
            rows.append(row)
        ordinals = sorted(int(row["global_ordinal"]) for row in rows)
        if ordinals != list(range(1, len(rows) + 1)):
            raise ValueError("global budget ticket ordinals are not contiguous")
        if len(rows) > self.max_calls:
            raise ValueError("global budget already exceeds its hard ceiling")
        return rows

    def reserve(self, request: RetrievalRequest) -> dict[str, object]:
        identity = request.identity
        if identity.index_url != RETRIEVAL_ENDPOINT_URL:
            raise ValueError("global budget request endpoint differs from approved URL")
        with self._Lock(self.root / ".budget.lock"):
            rows = self._tickets()
            key = identity.request_key
            if any(row["request_key"] == key for row in rows):
                raise ValueError("global budget refuses replay of an existing request")
            topic_calls = sum(
                isinstance(row["identity"], Mapping)
                and row["identity"].get("topic_id") == identity.topic_id
                for row in rows
            )
            if topic_calls >= self.max_per_topic:
                raise ValueError("global per-topic external-call budget is exhausted")
            if len(rows) >= self.max_calls:
                raise ValueError("global external-call budget is exhausted")
            ticket = {
                "schema_version": GLOBAL_BUDGET_SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "request_key": key,
                "identity": identity.canonical_dict(),
                "query_text": request.query_text,
                "global_ordinal": len(rows) + 1,
                "reserved_at": datetime.now(timezone.utc).isoformat(),
            }
            _create_only(self.tickets_dir / f"{key}.json", ticket)
            return ticket

    def verify_receipts(
        self,
        receipts: list[dict[str, object]],
        requests: Mapping[str, RetrievalRequest],
    ) -> None:
        if len(receipts) != len(requests):
            raise ValueError("global budget receipt/request counts differ")
        by_key = {str(row.get("request_key")): row for row in receipts}
        if set(by_key) != set(requests):
            raise ValueError("global budget receipt/request keys differ")
        current = {str(row["request_key"]): row for row in self._tickets()}
        for key, request in requests.items():
            row = by_key[key]
            if (
                current.get(key) != row
                or row.get("identity") != request.identity.canonical_dict()
                or row.get("query_text") != request.query_text
            ):
                raise ValueError("global budget receipt differs from durable ticket")


class GloballyBudgetedTransport:
    """Reserve the global ticket immediately before entering the one-shot transport."""

    def __init__(self, budget: GlobalExternalBudget, transport: RetrievalTransport):
        self.budget = budget
        self.transport = transport
        self.receipts: list[dict[str, object]] = []

    def __call__(self, request: RetrievalRequest):
        receipt = self.budget.reserve(request)
        self.receipts.append(receipt)
        return self.transport(request)
