"""A content-addressed, cross-run cache for RAGDoll support-judge results.

RAGDoll ships a cache in ``ragdoll.runner`` keyed by ``cache_key(config, instruction)``,
but it is a TTL cache that silently expires entries, stores the raw agent result, and
performs no validation that a reused row is a completed judgment carrying exactly one
support label. Post-run evaluation needs the opposite properties: entries that never
expire on their own, that are rejected unless they still validate, and that never
resurrect a failure as a completed judgment. So the behaviour lives here instead.

Two identities are deliberately kept apart:

* The **cache identity** (:class:`JudgeIdentity`) covers only values that change the
  effective judge request — the rendered instruction, the pinned RAGDoll revision, the
  prompt/task contract, and every output-affecting agent setting. It excludes config
  paths, experiment/run identifiers, topic and task identifiers, and timestamps, so two
  independently named runs that ask the judge the same question share one entry.
* The **run binding** lives in the judgment rows the caller materializes. A cross-run
  hit still produces a fresh row bound to that run's task id, handoff, and generation
  identity, with the reused entry digest recorded as private provenance.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


CACHE_SCHEMA_VERSION = "ragdoll_support_judge_cache_v1"
IDENTITY_SCHEMA_VERSION = "ragdoll_support_judge_identity_v1"
SUPPORT_LABELS = ("FS", "PS", "NS")


class JudgeCacheConflict(ValueError):
    """Raised when a valid entry already records a different label for one identity."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class JudgeIdentity:
    """Everything that changes the judge's answer, and nothing that does not."""

    evaluator: str
    instruction: str
    ragdoll_version: str
    ragdoll_commit: str
    prompt_contract_sha256: str
    task_schema_version: str
    provider: str
    model: str
    thinking: str
    temperature: float | None
    system_prompt: str
    agent_binary: str
    # The agent runtime/extension identity. A different extension or agent build can change
    # the answer for an otherwise identical request, so it belongs in the key.
    extension_identity: str = ""

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["identity_schema_version"] = IDENTITY_SCHEMA_VERSION
        return payload

    @property
    def digest(self) -> str:
        return sha256(canonical_bytes(self.payload())).hexdigest()


@dataclass(frozen=True)
class CachedJudgment:
    support_label: str
    identity_sha256: str


def _valid_label(value: Any) -> bool:
    return isinstance(value, str) and value in SUPPORT_LABELS


class JudgeCache:
    """A private on-disk cache of validated support labels, one file per identity."""

    DIRECTORY_MODE = 0o700
    FILE_MODE = 0o600

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(self.DIRECTORY_MODE)
        self._hits = 0
        self._misses = 0
        self._invalidations = 0
        self._writes = 0

    # -- layout ---------------------------------------------------------------

    def path_for(self, digest: str) -> Path:
        if not isinstance(digest, str) or len(digest) != 64 or not all(
            character in "0123456789abcdef" for character in digest
        ):
            raise ValueError("cache digest must be a lowercase sha256 hex string")
        return self.root / digest[:2] / f"{digest}.json"

    # -- reading --------------------------------------------------------------

    def read(self, identity: JudgeIdentity) -> CachedJudgment | None:
        """Return a validated entry, counting a corrupted one as an invalidation."""
        digest = identity.digest
        path = self.path_for(digest)
        if not path.is_file():
            self._misses += 1
            return None
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._invalidations += 1
            return None
        if not self._entry_is_valid(stored, identity, digest):
            self._invalidations += 1
            return None
        self._hits += 1
        return CachedJudgment(support_label=stored["support_label"], identity_sha256=digest)

    def get(self, identity: JudgeIdentity) -> str | None:
        entry = self.read(identity)
        return None if entry is None else entry.support_label

    def _entry_is_valid(self, stored: Any, identity: JudgeIdentity, digest: str) -> bool:
        if not isinstance(stored, dict):
            return False
        if stored.get("schema_version") != CACHE_SCHEMA_VERSION:
            return False
        if stored.get("identity_sha256") != digest:
            return False
        if stored.get("identity") != identity.payload():
            return False
        if stored.get("status") != "completed":
            return False
        return _valid_label(stored.get("support_label"))

    # -- writing --------------------------------------------------------------

    def put(self, identity: JudgeIdentity, *, support_label: str) -> str:
        """Store one completed label atomically; never store a failure."""
        if not _valid_label(support_label):
            raise ValueError(
                f"refusing to cache {support_label!r}; a completed judgment carries exactly "
                f"one of {', '.join(SUPPORT_LABELS)}"
            )
        digest = identity.digest
        path = self.path_for(digest)
        payload = canonical_bytes(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "identity_sha256": digest,
                "identity": identity.payload(),
                "status": "completed",
                "support_label": support_label,
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(self.DIRECTORY_MODE)

        # Publish by linking a fully written temp file into place. The final path therefore
        # never exists in a partial state, so a concurrent loser always reads complete
        # content instead of racing through an empty file and overwriting the winner.
        temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, self.FILE_MODE)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self._peek(path, identity, digest)
                if existing == support_label:
                    return digest
                if existing is not None:
                    raise JudgeCacheConflict(
                        f"cache entry {digest[:12]} already records {existing}; refusing to "
                        f"overwrite it with {support_label}"
                    ) from None
                # A complete but no longer valid entry is the only one we may heal.
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        self._writes += 1
        return digest

    def _peek(self, path: Path, identity: JudgeIdentity, digest: str) -> str | None:
        """Read without touching counters; an unusable entry is replaceable."""
        if not path.is_file():
            return None
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not self._entry_is_valid(stored, identity, digest):
            return None
        return str(stored["support_label"])

    # -- bookkeeping ----------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "invalidations": self._invalidations,
            "writes": self._writes,
        }


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Replace ``path`` atomically, leaving no partial file behind on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
