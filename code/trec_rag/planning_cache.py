"""Portable immutable cache for validated deterministic planning payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


PLANNING_CACHE_SCHEMA_VERSION = "planning-cache-entry-v1"
PLANNING_CACHE_DIRECTORY = "planning-cache-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENTRY_FIELDS = frozenset(
    {"schema_version", "cache_key", "identity", "payload", "payload_sha256"}
)
_IDENTITY_FIELDS = frozenset(
    {
        "endpoint",
        "model",
        "prompt_version",
        "schema_version",
        "topic_id",
        "narrative_sha256",
        "request_body_sha256",
    }
)

__all__ = [
    "PLANNING_CACHE_DIRECTORY",
    "PLANNING_CACHE_SCHEMA_VERSION",
    "PlanningCache",
    "PlanningCacheError",
    "PlanningCacheIdentity",
    "PlanningCacheIntegrityError",
    "PlanningCacheMiss",
    "build_planning_cache_identity",
]


class PlanningCacheError(RuntimeError):
    """Base class for deterministic planning cache failures."""


class PlanningCacheMiss(PlanningCacheError):
    """The exact validated planning payload is absent."""


class PlanningCacheIntegrityError(PlanningCacheError):
    """A planning entry is malformed or conflicts with immutable state."""


@dataclass(frozen=True)
class PlanningCacheIdentity:
    endpoint: str
    model: str
    prompt_version: str
    schema_version: str
    topic_id: str
    narrative_sha256: str
    request_body_sha256: str

    def as_dict(self) -> dict[str, str]:
        values = {
            "endpoint": self.endpoint,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "topic_id": self.topic_id,
            "narrative_sha256": self.narrative_sha256,
            "request_body_sha256": self.request_body_sha256,
        }
        for field in (
            "endpoint",
            "model",
            "prompt_version",
            "schema_version",
            "topic_id",
        ):
            value = values[field]
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"planning identity {field} must be non-empty text")
        for field in ("narrative_sha256", "request_body_sha256"):
            if not isinstance(values[field], str) or not _SHA256.fullmatch(values[field]):
                raise ValueError(f"planning identity {field} must be a lowercase SHA-256")
        return values

    @property
    def cache_key(self) -> str:
        return sha256(_canonical_json(self.as_dict())).hexdigest()


def build_planning_cache_identity(
    *,
    request_body: bytes,
    endpoint: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    topic_id: str,
    narrative: str,
) -> PlanningCacheIdentity:
    """Bind every exact deterministic request input without retaining narrative text."""
    if not isinstance(request_body, bytes) or not request_body:
        raise ValueError("planning request body must be non-empty bytes")
    if not isinstance(narrative, str) or not narrative:
        raise ValueError("planning narrative must be non-empty text")
    identity = PlanningCacheIdentity(
        endpoint=endpoint,
        model=model,
        prompt_version=prompt_version,
        schema_version=schema_version,
        topic_id=topic_id,
        narrative_sha256=sha256(narrative.encode("utf-8")).hexdigest(),
        request_body_sha256=sha256(request_body).hexdigest(),
    )
    identity.as_dict()
    return identity


class PlanningCache:
    """Read and create canonical planning entries without mutable repair paths."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError("planning cache root must be a Path")
        self.root = root

    def entry_path(self, identity: PlanningCacheIdentity) -> Path:
        if not isinstance(identity, PlanningCacheIdentity):
            raise TypeError("identity must be PlanningCacheIdentity")
        key = identity.cache_key
        return self.root / PLANNING_CACHE_DIRECTORY / key[:2] / f"{key}.json"

    def load(self, identity: PlanningCacheIdentity) -> dict[str, object]:
        path = self.entry_path(identity)
        try:
            source = path.read_bytes()
        except FileNotFoundError as exc:
            raise PlanningCacheMiss(f"planning cache miss for {identity.cache_key}") from exc
        except OSError as exc:
            raise PlanningCacheIntegrityError(f"planning cache read failed: {path}") from exc
        return self._decode_entry(source, identity, path)

    def store(
        self,
        identity: PlanningCacheIdentity,
        payload: Mapping[str, object],
    ) -> Path:
        if not isinstance(payload, Mapping):
            raise TypeError("validated planning payload must be a mapping")
        canonical_payload = _canonical_mapping(payload, "validated planning payload")
        payload_bytes = _canonical_json(canonical_payload)
        entry = {
            "schema_version": PLANNING_CACHE_SCHEMA_VERSION,
            "cache_key": identity.cache_key,
            "identity": identity.as_dict(),
            "payload": canonical_payload,
            "payload_sha256": sha256(payload_bytes).hexdigest(),
        }
        source = _canonical_json(entry) + b"\n"
        path = self.entry_path(identity)
        if path.exists():
            existing = self._decode_entry(path.read_bytes(), identity, path)
            if existing != canonical_payload or path.read_bytes() != source:
                raise PlanningCacheIntegrityError(
                    f"planning cache immutable conflict: {path}"
                )
            return path
        _create_only(path, source)
        existing = self._decode_entry(path.read_bytes(), identity, path)
        if existing != canonical_payload or path.read_bytes() != source:
            raise PlanningCacheIntegrityError(f"planning cache immutable conflict: {path}")
        return path

    @staticmethod
    def _decode_entry(
        source: bytes,
        identity: PlanningCacheIdentity,
        path: Path,
    ) -> dict[str, object]:
        try:
            value = json.loads(
                source.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_invalid_constant,
            )
            if not isinstance(value, dict) or set(value) != _ENTRY_FIELDS:
                raise ValueError("entry fields differ")
            stored_identity = value["identity"]
            if (
                value["schema_version"] != PLANNING_CACHE_SCHEMA_VERSION
                or value["cache_key"] != identity.cache_key
                or not isinstance(stored_identity, dict)
                or set(stored_identity) != _IDENTITY_FIELDS
                or stored_identity != identity.as_dict()
                or not isinstance(value["payload"], dict)
            ):
                raise ValueError("entry identity differs")
            canonical_payload = _canonical_mapping(value["payload"], "cached payload")
            payload_bytes = _canonical_json(canonical_payload)
            if value["payload_sha256"] != sha256(payload_bytes).hexdigest():
                raise ValueError("payload digest differs")
            if source != _canonical_json(value) + b"\n":
                raise ValueError("entry is not canonical JSON")
            return canonical_payload
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise PlanningCacheIntegrityError(f"planning cache entry is invalid: {path}") from exc


def _canonical_mapping(value: Mapping[str, object], label: str) -> dict[str, object]:
    canonical = _canonical_value(value, label)
    if not isinstance(canonical, dict):  # pragma: no cover - Mapping guarantees this
        raise TypeError(f"{label} must be a mapping")
    return canonical


def _canonical_value(value: object, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} keys must be strings")
            result[key] = _canonical_value(item, f"{label}.{key}")
        return dict(sorted(result.items()))
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, label) for item in value]
    raise TypeError(f"{label} contains a non-JSON value")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _create_only(path: Path, source: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as sink:
            os.fchmod(sink.fileno(), 0o600)
            sink.write(source)
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return
        os.chmod(path, 0o600)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
