"""Deterministic, hostile-input-safe competition cache shard bundles.

The archive format is deliberately small: a canonical manifest is the first
member, every other regular file is declared by digest and size, and no tar
metadata is trusted during verification.  Bundle completion is published last
as a separate canonical receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import sqlite3
import stat
import tarfile
import tempfile
from typing import Any, BinaryIO, Callable, Sequence
import unicodedata
from urllib.parse import quote

import yaml
import zstandard

from trec_rag.facet_pilot_config import (
    FacetPilotConfig,
    load_facet_pilot_config,
    select_configured_topics,
)
from trec_rag.repo_env import repo_cache_root


BUNDLE_SCHEMA_VERSION = "trec-rag-cache-bundle-v1"
BUNDLE_ARCHIVE_NAME = "bundle.tar.zst"
BUNDLE_COMPLETE_NAME = "bundle-complete.json"
BUNDLE_MANIFEST_NAME = "bundle-manifest.json"
MERGE_STATE_DIRECTORY = ".cache-bundle-merges"
MERGE_PREPARE_NAME = "prepare.json"
MERGE_COMPLETE_NAME = "complete.json"
MERGE_CONFLICTS_NAME = "conflicts.json"

# Bounds are intentionally independent.  A small compressed archive may still
# be a decompression bomb, and one oversized member should fail before its body
# is read.
MAX_COMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 32 * 1024 * 1024 * 1024
MAX_MEMBER_BYTES = 4 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_MEMBERS = 1_000_000
MAX_ZSTD_WINDOW_BYTES = 128 * 1024 * 1024
_COPY_CHUNK = 1024 * 1024
_HEX_DIGEST_LENGTH = 64

_FORBIDDEN_SUFFIXES = (
    ".lock",
    ".sqlite",
    ".sqlite3",
    ".sqlite-wal",
    ".sqlite-shm",
    ".sqlite3-wal",
    ".sqlite3-shm",
    ".wal",
    ".shm",
)
_FORBIDDEN_PARTS = frozenset(
    {
        "attempt",
        "attempts",
        "answers",
        "claims",
        "full-corpus",
        "generated-answers",
        "generation",
        "gold",
        "gold-nuggets",
        "locks",
        "qrels",
        "rag-answers",
        "ragdoll",
        "ragdoll-scores",
        "raw",
        "responses",
        "provider-responses",
        "work",
    }
)
_CACHE_TOP_LEVEL = frozenset(
    {
        "canonical",
        "documents",
        "planning-cache-v1",
        "retrieval",
        "similarity-cache-v1",
    }
)

_TOPIC_PHASE_ARTIFACTS = {
    "retrieval": frozenset(
        {
            "decomposition.json",
            "retrieval/audit.json",
            "retrieval/evidence-bundle.json",
        }
    ),
    "scoring": frozenset(
        {
            "scoring/lane_scores.jsonl",
            "scoring/selected_documents.jsonl",
            "scoring/selection.json",
            "scoring/selected_subnarrative_scores.jsonl",
        }
    ),
    "canonical": frozenset(
        {
            "canonical/handoff/candidate-requests.jsonl",
            "canonical/handoff/selection-contexts.jsonl",
            "canonical/handoff/handoff-manifest.json",
            "records.sqlite3",
            "canonical/records-manifest.json",
            "canonical/subnarrative-selections.jsonl",
            "canonical/selection-manifest.json",
            "canonical/canonical-nuggets.jsonl",
            "canonical/canonical-nugget-manifest.json",
            "canonical/retrieval-projection.json",
            "canonical/retrieval-projection-manifest.json",
            "canonical/generation-projection.json",
            "canonical/generation-projection-manifest.json",
        }
    ),
}
_TOPIC_RECEIPT = "topic-job-receipt.json"
_CACHE_OPERATION_RECEIPT = "cache-operation-receipt.json"
_CACHE_OPERATION_RECEIPT_SCHEMA = "cache-operation-receipt-v1"
_CACHE_OPERATION_PHASES = ("planning", "retrieval", "scoring", "canonical")
_CACHE_OPERATION_STAGES = (
    "planning",
    "retrieval",
    "passage_scores",
    "sentence_scores",
    "similarity",
    "canonicalization",
)
_CACHE_OPERATION_COUNTERS = (
    "cache_hits",
    "cache_misses",
    "network_calls",
    "provider_calls",
    "model_batches",
)
_LIVE_DECOMPOSITION_RESULT = "decomposition/result.json"
_LIVE_DECOMPOSITION_MANIFEST = "decomposition/manifest.json"
_SEED_DECOMPOSITION_MANIFEST = "planning-seed-manifest.json"
_SEED_DECOMPOSITION_RECEIPT = "planning-seed-receipt.json"


class CacheBundleError(RuntimeError):
    """Base class for deterministic bundle failures."""


class CacheBundleIntegrityError(CacheBundleError):
    """A source, archive, marker, or extraction target is unsafe."""


class CacheBundleConflictError(CacheBundleError):
    """Immutable destination content contradicts a verified bundle."""


@dataclass(frozen=True)
class BundleMember:
    path: str
    kind: str
    size: int
    sha256: str
    mode: int = 0o600


@dataclass(frozen=True)
class BundleReceipt:
    topic_id: str
    archive_sha256: str
    archive_size: int
    manifest_sha256: str
    member_count: int


@dataclass(frozen=True)
class VerifiedBundle:
    bundle_dir: Path
    topic_id: str
    experiment_id: str
    archive_sha256: str
    archive_size: int
    manifest_sha256: str
    source_config_sha256: str
    members: tuple[BundleMember, ...]


@dataclass(frozen=True)
class MergeReceipt:
    merge_id: str
    completion_path: Path
    bundle_count: int
    operation_count: int
    installed_count: int
    identical_count: int
    kept_conflict_count: int
    score_import_count: int


@dataclass(frozen=True)
class _SourceMember:
    archive_path: str
    source_path: Path
    kind: str
    size: int
    sha256: str

    @property
    def public(self) -> BundleMember:
        return BundleMember(
            path=self.archive_path,
            kind=self.kind,
            size=self.size,
            sha256=self.sha256,
        )


@dataclass(frozen=True)
class _InstallOperation:
    member: BundleMember
    staged_path: Path
    target_root_name: str
    relative_path: PurePosixPath

    def target(self, cache_root: Path, outputs_root: Path) -> Path:
        root = cache_root if self.target_root_name == "cache" else outputs_root
        return root.joinpath(*self.relative_path.parts)


@dataclass(frozen=True)
class _ScoreOperation:
    member: BundleMember
    staged_path: Path


class _BoundedReader:
    """Count bytes returned by a streaming decompressor."""

    def __init__(self, source: BinaryIO, maximum: int) -> None:
        self._source = source
        self._maximum = maximum
        self._count = 0

    def read(self, size: int = -1) -> bytes:
        value = self._source.read(size)
        self._count += len(value)
        if self._count > self._maximum:
            raise CacheBundleIntegrityError("bundle decompressed-size limit exceeded")
        return value

    @property
    def count(self) -> int:
        return self._count


class _DigestingReader:
    """Detect a source changing after its manifest receipt was computed."""

    def __init__(self, source: BinaryIO) -> None:
        self._source = source
        self._digest = hashlib.sha256()
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        value = self._source.read(size)
        self._digest.update(value)
        self.count += len(value)
        return value

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CacheBundleIntegrityError(
            "bundle metadata is not canonical JSON"
        ) from exc
    return (encoded + "\n").encode("utf-8")


def _canonical_or_pretty_json(value: object, body: bytes) -> bool:
    """Accept only the two canonical encodings used by production manifests."""
    try:
        pretty = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CacheBundleIntegrityError("checkpoint metadata is not JSON") from exc
    return body in {_canonical_json(value), pretty}


def _strict_json(value: bytes, label: str) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise CacheBundleIntegrityError(
                    f"{label} contains a duplicate JSON key"
                )
            result[key] = item
        return result

    def reject_constant(constant: str) -> object:
        raise CacheBundleIntegrityError(
            f"{label} contains non-standard JSON constant {constant}"
        )

    try:
        return json.loads(
            value,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheBundleIntegrityError(f"{label} is not strict UTF-8 JSON") from exc


def _require_exact_fields(
    value: object,
    expected: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise CacheBundleIntegrityError(
            f"{label} fields differ: expected={sorted(expected)!r}, actual={actual!r}"
        )
    return value


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HEX_DIGEST_LENGTH
        or value != value.lower()
    ):
        raise CacheBundleIntegrityError(f"{label} is not a lowercase SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise CacheBundleIntegrityError(
            f"{label} is not a lowercase SHA-256 digest"
        ) from exc
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CacheBundleIntegrityError(f"{label} must be a non-negative integer")
    return value


def _file_receipt(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK), b""):
            count += len(chunk)
            digest.update(chunk)
    return count, digest.hexdigest()


def _require_absolute_directory(path: Path, label: str, *, create: bool) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError(f"{label} must be an absolute directory")
    if any(part in {".", ".."} for part in raw.parts):
        raise CacheBundleIntegrityError(f"{label} must be lexically normalized")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise CacheBundleIntegrityError(
                f"{label} contains a symbolic-link component"
            )
    if create:
        raw.mkdir(parents=True, exist_ok=True)
    if not raw.is_dir():
        raise CacheBundleIntegrityError(f"{label} must be a directory")
    return raw.resolve()


def _validate_archive_path(name: str, seen: dict[str, str]) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise CacheBundleIntegrityError("archive member has an unsafe path")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != name
    ):
        raise CacheBundleIntegrityError(f"archive member path is unsafe: {name!r}")
    collision_key = unicodedata.normalize("NFC", name).casefold()
    prior = seen.get(collision_key)
    if prior is not None:
        if prior == name:
            raise CacheBundleIntegrityError(f"duplicate archive member: {name}")
        raise CacheBundleIntegrityError(
            f"archive member canonical/case collision: {prior!r}, {name!r}"
        )
    seen[collision_key] = name
    return name


def _is_forbidden(relative: PurePosixPath) -> bool:
    folded = tuple(part.casefold() for part in relative.parts)
    if any(
        part in _FORBIDDEN_PARTS or PurePosixPath(part).stem in _FORBIDDEN_PARTS
        for part in folded
    ):
        return True
    name = relative.name.casefold()
    return name.startswith(".") or name.endswith(_FORBIDDEN_SUFFIXES)


def _require_regular_source(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise CacheBundleIntegrityError(
            "bundle source escaped its declared root"
        ) from exc
    current = root
    for part in relative.parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CacheBundleIntegrityError(f"bundle source contains a link: {path}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise CacheBundleIntegrityError(f"bundle source is not a regular file: {path}")


def _source_member(
    source: Path,
    *,
    source_root: Path,
    archive_path: str,
    kind: str,
) -> _SourceMember:
    _validate_archive_path(archive_path, {})
    _require_regular_source(source, source_root)
    size, digest = _file_receipt(source)
    if size > MAX_MEMBER_BYTES:
        raise CacheBundleIntegrityError(f"bundle source member is too large: {source}")
    return _SourceMember(archive_path, source, kind, size, digest)


def _checkpoint_manifest_sources(topic_root: Path, phase: str) -> tuple[Path, ...]:
    manifest_path = topic_root / phase / "complete.json"
    try:
        body = manifest_path.read_bytes()
    except OSError as exc:
        raise CacheBundleIntegrityError(
            f"selected topic {phase} checkpoint manifest is missing"
        ) from exc
    value = _strict_json(body, f"{phase} checkpoint manifest")
    if (
        not isinstance(value, dict)
        or not _canonical_or_pretty_json(value, body)
        or value.get("topic_id") != topic_root.name
        or value.get("phase") != ("retrieve" if phase == "retrieval" else phase)
    ):
        raise CacheBundleIntegrityError(
            f"selected topic {phase} checkpoint manifest is invalid"
        )
    receipts = value.get("artifacts")
    if not isinstance(receipts, list):
        raise CacheBundleIntegrityError("checkpoint artifacts must be a list")
    found: dict[str, Path] = {}
    for index, raw in enumerate(receipts):
        receipt = _require_exact_fields(
            raw,
            {"bytes", "relative_path", "sha256"},
            f"checkpoint artifact {index}",
        )
        relative = receipt["relative_path"]
        if not isinstance(relative, str):
            raise CacheBundleIntegrityError("checkpoint artifact path must be text")
        _validate_archive_path(relative, {})
        if relative in found:
            raise CacheBundleIntegrityError("checkpoint artifact receipt is duplicated")
        artifact = topic_root.joinpath(*PurePosixPath(relative).parts)
        _require_regular_source(artifact, topic_root)
        size, digest = _file_receipt(artifact)
        if size != _require_nonnegative_int(receipt["bytes"], "artifact bytes"):
            raise CacheBundleIntegrityError("checkpoint artifact size mismatch")
        if digest != _require_digest(receipt["sha256"], "artifact sha256"):
            raise CacheBundleIntegrityError("checkpoint artifact digest mismatch")
        found[relative] = artifact
    if set(found) != _TOPIC_PHASE_ARTIFACTS[phase]:
        raise CacheBundleIntegrityError(
            f"selected topic {phase} checkpoint artifact set changed"
        )
    return (manifest_path, *(found[path] for path in sorted(found)))


def _decomposition_sources(
    config: FacetPilotConfig,
    topic_root: Path,
) -> tuple[tuple[Path, Path], ...]:
    result = topic_root / _LIVE_DECOMPOSITION_RESULT
    _require_regular_source(result, topic_root)
    retrieval = _strict_json(
        (topic_root / "retrieval/complete.json").read_bytes(),
        "retrieval checkpoint manifest",
    )
    if (
        not isinstance(retrieval, dict)
        or retrieval.get("decomposition_source_sha256")
        != hashlib.sha256(result.read_bytes()).hexdigest()
    ):
        raise CacheBundleIntegrityError(
            "decomposition source differs from its checkpoint identity"
        )
    live = topic_root / _LIVE_DECOMPOSITION_MANIFEST
    seeded = topic_root / _SEED_DECOMPOSITION_MANIFEST
    if live.is_file() == seeded.is_file():
        raise CacheBundleIntegrityError(
            "decomposition requires exactly one authenticated producer manifest"
        )
    if live.is_file():
        _require_regular_source(live, topic_root)
        body = live.read_bytes()
        value = _require_exact_fields(
            _strict_json(body, "decomposition producer manifest"),
            {
                "planner",
                "result_bytes",
                "result_file",
                "result_sha256",
                "schema_version",
            },
            "decomposition producer manifest",
        )
        if (
            not _canonical_or_pretty_json(value, body)
            or value["schema_version"] != "facet-decomposition-manifest-v1"
            or value["result_file"] != result.name
            or value["result_bytes"] != result.stat().st_size
            or value["result_sha256"] != hashlib.sha256(result.read_bytes()).hexdigest()
            or not isinstance(value["planner"], dict)
        ):
            raise CacheBundleIntegrityError("decomposition producer manifest is stale")
        return ((result, topic_root), (live, topic_root))
    _require_regular_source(seeded, topic_root)
    aggregate = config.output_dir / _SEED_DECOMPOSITION_RECEIPT
    _require_regular_source(aggregate, config.output_dir)
    # The production validator has already authenticated the full seeded
    # producer chain.  Retain both receipts so the merged checkpoint preserves
    # that chain rather than silently degrading it to a result-only artifact.
    return (
        (result, topic_root),
        (seeded, topic_root),
        (aggregate, config.output_dir),
    )


def _validate_production_topic_checkpoint(
    config: FacetPilotConfig,
    topic: Any,
    *,
    config_path: Path,
    config_bytes: bytes,
) -> None:
    """Use production dispatch/projection readers before selecting any bytes."""
    from trec_rag.competition_retrieval import _decomposition_producer_sha256
    from trec_rag.retrieval_export import (
        read_topic_projection_receipt,
        validate_retrieval_topic_checkpoints,
    )
    from trec_rag.topic_dispatch import TopicJob, read_topic_receipt

    topic_root = (config.output_dir / topic.id).resolve()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    try:
        job = TopicJob(
            topic_id=topic.id,
            run_id=config.run_id,
            config_path=config_path.resolve(),
            config_bytes=config_bytes,
            config_sha256=config_sha256,
            topic_root=topic_root,
        )
        dispatch = read_topic_receipt(job)
        projection = read_topic_projection_receipt(config, topic)
        source_seals = dict(projection.source_seals)
        if source_seals.get("config_sha256") != config_sha256:
            raise ValueError("projection config identity changed")
        producer_sha256 = _decomposition_producer_sha256(topic, config.output_dir, None)
        if source_seals.get("decomposition_producer_sha256") != producer_sha256:
            raise ValueError("projection decomposition producer identity changed")
        retrieval = _strict_json(
            (topic_root / "retrieval/complete.json").read_bytes(),
            "retrieval checkpoint manifest",
        )
        if not isinstance(retrieval, dict) or not isinstance(
            retrieval.get("retriever"), dict
        ):
            raise ValueError("retrieval checkpoint identity is invalid")
        validated = validate_retrieval_topic_checkpoints(
            config,
            (topic,),
            expected_retriever_identity=retrieval["retriever"],
            expected_decomposition_producer_sha256={topic.id: producer_sha256},
        )
        if (
            len(validated) != 1
            or validated[0] != projection
            or dispatch is None
            or dispatch.topic_id != topic.id
            or dispatch.projection_manifest_sha256 != projection.manifest_sha256
            or dispatch.status != projection.retrieval_status
            or dispatch.stopping_reason != projection.retrieval_stopping_reason
        ):
            raise ValueError("dispatch and projection receipts differ")
    except (CacheBundleError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise CacheBundleIntegrityError(
            f"selected topic production checkpoint is invalid: {topic.id}"
        ) from exc


def _validate_online_cache_operation_receipt(
    source: Path,
    *,
    config_sha256: str,
    run_id: str,
    topic_id: str,
    projection_manifest_sha256: str,
) -> None:
    """Validate the Task 5 wire receipt without importing the runner module."""
    try:
        metadata = source.lstat()
        body = source.read_bytes()
    except OSError as exc:
        raise CacheBundleIntegrityError("cache operation receipt is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CacheBundleIntegrityError("cache operation receipt is not a regular file")
    if len(body) > MAX_MANIFEST_BYTES:
        raise CacheBundleIntegrityError("cache operation receipt size limit exceeded")
    value = _require_exact_fields(
        _strict_json(body, "cache operation receipt"),
        {
            "schema_version",
            "mode",
            "run_id",
            "topic_id",
            "config_sha256",
            "projection_manifest_sha256",
            "phases",
            "stages",
            "receipt_content_sha256",
        },
        "cache operation receipt",
    )
    if _canonical_json(value) != body:
        raise CacheBundleIntegrityError("cache operation receipt is not canonical JSON")
    content = dict(value)
    receipt_content_sha256 = _require_digest(
        content.pop("receipt_content_sha256"),
        "cache operation receipt content digest",
    )
    if receipt_content_sha256 != hashlib.sha256(_canonical_json(content)).hexdigest():
        raise CacheBundleIntegrityError(
            "cache operation receipt content digest differs"
        )
    if (
        value["schema_version"] != _CACHE_OPERATION_RECEIPT_SCHEMA
        or value["mode"] != "online"
        or value["run_id"] != run_id
        or value["topic_id"] != topic_id
        or _require_digest(
            value["config_sha256"], "cache operation receipt config digest"
        )
        != config_sha256
        or _require_digest(
            value["projection_manifest_sha256"],
            "cache operation receipt projection digest",
        )
        != projection_manifest_sha256
    ):
        raise CacheBundleIntegrityError("cache operation receipt identity changed")
    phases = _require_exact_fields(
        value["phases"],
        set(_CACHE_OPERATION_PHASES),
        "cache operation receipt phases",
    )
    for phase_name in _CACHE_OPERATION_PHASES:
        phase = _require_exact_fields(
            phases[phase_name],
            {"resumed"},
            f"cache operation receipt {phase_name} phase",
        )
        if type(phase["resumed"]) is not bool:
            raise CacheBundleIntegrityError(
                "cache operation receipt phase resumed value is not Boolean"
            )
    stages = _require_exact_fields(
        value["stages"],
        set(_CACHE_OPERATION_STAGES),
        "cache operation receipt stages",
    )
    for stage_name in _CACHE_OPERATION_STAGES:
        counters = _require_exact_fields(
            stages[stage_name],
            set(_CACHE_OPERATION_COUNTERS),
            f"cache operation receipt {stage_name} counters",
        )
        for counter_name in _CACHE_OPERATION_COUNTERS:
            _require_nonnegative_int(
                counters[counter_name],
                f"cache operation receipt {stage_name} {counter_name}",
            )


def _collect_topic_members(
    config: FacetPilotConfig,
    topic: Any,
    *,
    config_path: Path,
    config_bytes: bytes,
) -> list[_SourceMember]:
    topic_id = topic.id
    topic_root = config.output_dir / topic_id
    if not topic_root.is_dir() or topic_root.is_symlink():
        raise CacheBundleIntegrityError(
            f"selected topic checkpoint is missing: {topic_id}"
        )
    _validate_production_topic_checkpoint(
        config,
        topic,
        config_path=config_path,
        config_bytes=config_bytes,
    )
    allowed_sensitive = {
        PurePosixPath("records.sqlite3"),
        PurePosixPath(".topic-projection.lock"),
        PurePosixPath(".records-publication.lock"),
    }
    for source in sorted(topic_root.rglob("*")):
        if source.is_dir() and not source.is_symlink():
            continue
        relative = PurePosixPath(source.relative_to(topic_root).as_posix())
        if _is_forbidden(relative) and relative not in allowed_sensitive:
            raise CacheBundleIntegrityError(
                f"selected topic contains a forbidden sensitive extra: {relative}"
            )

    sources: dict[Path, Path] = {}
    for phase in ("retrieval", "scoring", "canonical"):
        for source in _checkpoint_manifest_sources(topic_root, phase):
            sources[source.resolve()] = topic_root
    projection_manifest = topic_root / "canonical/retrieval-projection-manifest.json"
    operation_receipt = topic_root / _CACHE_OPERATION_RECEIPT
    _validate_online_cache_operation_receipt(
        operation_receipt,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        run_id=config.run_id,
        topic_id=topic_id,
        projection_manifest_sha256=hashlib.sha256(
            projection_manifest.read_bytes()
        ).hexdigest(),
    )
    sources[operation_receipt.resolve()] = topic_root
    receipt = topic_root / _TOPIC_RECEIPT
    _require_regular_source(receipt, topic_root)
    sources[receipt.resolve()] = topic_root
    for source, source_root in _decomposition_sources(config, topic_root):
        sources[source.resolve()] = source_root

    prefix = PurePosixPath("outputs", config.experiment.id, topic_id)
    members: list[_SourceMember] = []
    for source, source_root in sorted(
        sources.items(), key=lambda item: item[0].as_posix()
    ):
        if source_root == config.output_dir:
            archive_path = PurePosixPath(
                "outputs", config.experiment.id, source.name
            ).as_posix()
        else:
            relative = PurePosixPath(source.relative_to(topic_root).as_posix())
            archive_path = (prefix / relative).as_posix()
        members.append(
            _source_member(
                source,
                source_root=source_root,
                archive_path=archive_path,
                kind="checkpoint",
            )
        )
    return members


def _validate_planning_entry(cache_root: Path, source: Path, topic_id: str) -> None:
    """Adapter around the evolving planning-cache public validation API."""
    from trec_rag.planning_cache import PlanningCache, PlanningCacheIdentity

    raw = source.read_bytes()
    value = _strict_json(raw, f"planning cache entry {source}")
    if not isinstance(value, dict) or not isinstance(value.get("identity"), dict):
        raise CacheBundleIntegrityError(f"planning cache entry is invalid: {source}")
    try:
        identity = PlanningCacheIdentity(**value["identity"])
        if identity.topic_id != topic_id:
            raise ValueError("planning cache entry belongs to another topic")
        cache = PlanningCache(cache_root)
        if cache.entry_path(identity).resolve() != source.resolve():
            raise ValueError("planning cache entry path differs from its identity")
        cache.load(identity)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CacheBundleIntegrityError(
            f"planning cache entry is invalid: {source}"
        ) from exc


def _validate_similarity_entry(cache_root: Path, source: Path) -> Any:
    """Adapter around the evolving similarity-cache public validation API."""
    from trec_rag.similarity_cache import SimilarityCache, SimilarityCacheIdentity

    raw = source.read_bytes()
    value = _strict_json(raw, f"similarity cache entry {source}")
    if not isinstance(value, dict) or not isinstance(value.get("identity"), dict):
        raise CacheBundleIntegrityError(f"similarity cache entry is invalid: {source}")
    identity_value = value["identity"]
    model_identity = identity_value.get("model_identity")
    text_sha256s = identity_value.get("text_sha256s")
    if not isinstance(model_identity, dict) or not isinstance(text_sha256s, list):
        raise CacheBundleIntegrityError(f"similarity cache entry is invalid: {source}")
    try:
        identity = SimilarityCacheIdentity(
            **model_identity,
            text_sha256s=tuple(text_sha256s),
        )
        cache = SimilarityCache(cache_root)
        if cache.entry_path(identity).resolve() != source.resolve():
            raise ValueError("similarity cache entry path differs from its identity")
        cache.load(identity)
        return identity
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CacheBundleIntegrityError(
            f"similarity cache entry is invalid: {source}"
        ) from exc


def _validate_existing_similarity_conflict(
    operation: _InstallOperation,
    *,
    cache_root: Path,
    target: Path,
) -> None:
    """Require both conflicting matrices to decode under one exact identity."""
    from trec_rag.similarity_cache import SimilarityCache

    staged_cache_root = operation.staged_path
    for _part in operation.relative_path.parts:
        staged_cache_root = staged_cache_root.parent
    try:
        identity = _validate_similarity_entry(staged_cache_root, operation.staged_path)
        destination_cache = SimilarityCache(cache_root)
        if destination_cache.entry_path(identity).resolve() != target.resolve():
            raise ValueError("destination similarity path differs from source identity")
        destination_cache.load(identity)
    except (CacheBundleError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise CacheBundleIntegrityError(
            f"destination similarity entry is invalid: {target}"
        ) from exc


def _validate_canonical_entry(source: Path) -> None:
    """Validate the portable portion of an existing canonical provider entry."""
    raw = source.read_bytes()
    value = _require_exact_fields(
        _strict_json(raw, f"validated canonical cache entry {source}"),
        {
            "content",
            "content_sha256",
            "metadata",
            "metadata_entries",
            "request_sha256",
            "response_body_sha256s",
            "schema_version",
            "status",
        },
        "validated canonical cache entry",
    )
    from trec_rag.canonical_nuggets import VALIDATED_CACHE_SCHEMA_VERSION

    content = value["content"]
    request_sha256 = value["request_sha256"]
    response_hashes = value["response_body_sha256s"]
    metadata_entries = value["metadata_entries"]
    if (
        value["schema_version"] != VALIDATED_CACHE_SCHEMA_VERSION
        or not isinstance(content, str)
        or hashlib.sha256(content.encode("utf-8")).hexdigest()
        != value["content_sha256"]
        or _canonical_json(_strict_json(content.encode("utf-8"), "canonical content"))[
            :-1
        ]
        != content.encode("utf-8")
        or _require_digest(request_sha256, "canonical request sha256") != source.stem
        or isinstance(value["status"], bool)
        or not isinstance(value["status"], int)
        or not isinstance(response_hashes, list)
        or not response_hashes
        or any(
            _require_digest(item, "canonical response digest") != item
            for item in response_hashes
        )
        or not isinstance(metadata_entries, list)
        or len(metadata_entries) != len(response_hashes)
        or any(not isinstance(item, dict) for item in metadata_entries)
        or not isinstance(value["metadata"], dict)
        or raw != _canonical_json(value)
    ):
        raise CacheBundleIntegrityError(
            f"validated canonical cache entry is invalid: {source}"
        )


def _validated_retrieval_files(
    config: FacetPilotConfig,
    cache_root: Path,
) -> set[Path]:
    """Validate complete retrieval entries through their pure read adapter."""
    from trec_rag.document_store import DocumentStore
    from trec_rag.retrieval_cache import (
        DerivationIdentity,
        OrganizerTextNormalizer,
        RetrievalCache,
        TransportIdentity,
    )

    retrieval_root = config.retrieval.cache_dir
    v2_root = retrieval_root if retrieval_root.name == "v2" else retrieval_root / "v2"
    if not v2_root.exists():
        return set()
    if v2_root.is_symlink() or not v2_root.is_dir():
        raise CacheBundleIntegrityError("retrieval v2 cache root is unsafe")
    allowed: set[Path] = set()
    transport_manifests = sorted(
        v2_root.glob("[0-9a-f][0-9a-f]/*/transport-manifest.json")
    )
    for manifest_path in transport_manifests:
        try:
            manifest = _require_exact_fields(
                _strict_json(
                    manifest_path.read_bytes(), "retrieval transport manifest"
                ),
                {
                    "gzip_length",
                    "gzip_sha256",
                    "query_sha256",
                    "raw_filename",
                    "raw_length",
                    "raw_sha256",
                    "request_key",
                    "schema_version",
                    "transport_identity",
                },
                "retrieval transport manifest",
            )
            if not isinstance(manifest["transport_identity"], dict):
                raise ValueError("transport identity is not an object")
            identity = TransportIdentity(**manifest["transport_identity"])
            entry = manifest_path.parent
            if (
                identity.index_id != config.retrieval.index
                or identity.corpus_epoch != config.retrieval.corpus_epoch
                or identity.hits != config.retrieval.documents_per_query
                or identity.request_key != manifest["request_key"]
                or entry.name != identity.request_key
                or entry.parent.name != identity.request_key[:2]
                or manifest["query_sha256"] != identity.query_sha256
                or manifest["raw_filename"] != "raw.body.gz"
            ):
                raise ValueError("transport path identity changed")
            raw_path = entry / "raw.body.gz"
            compressed = raw_path.read_bytes()
            if (
                len(compressed) != manifest["gzip_length"]
                or hashlib.sha256(compressed).hexdigest() != manifest["gzip_sha256"]
            ):
                raise ValueError("transport gzip receipt changed")
            raw = gzip.decompress(compressed)
            if (
                len(raw) != manifest["raw_length"]
                or hashlib.sha256(raw).hexdigest() != manifest["raw_sha256"]
                or len(raw) > MAX_MEMBER_BYTES
            ):
                raise ValueError("transport raw receipt changed")
            response = _strict_json(raw, "retrieval organizer response")
            if (
                not isinstance(response, dict)
                or not isinstance(response.get("query"), dict)
                or not isinstance(response["query"].get("text"), str)
            ):
                raise ValueError("transport response query is invalid")
            query_text = response["query"]["text"]
            derivation_manifests = sorted(
                (entry / "derived").glob("*/derivation-manifest.json")
            )
            if not derivation_manifests:
                raise ValueError("transport has no complete derivation")
            allowed.update({manifest_path.resolve(), raw_path.resolve()})
            for derivation_manifest_path in derivation_manifests:
                derivation_manifest = _strict_json(
                    derivation_manifest_path.read_bytes(),
                    "retrieval derivation manifest",
                )
                if not isinstance(derivation_manifest, dict) or not isinstance(
                    derivation_manifest.get("derivation_identity"), dict
                ):
                    raise ValueError("derivation identity is invalid")
                derivation = DerivationIdentity(
                    **derivation_manifest["derivation_identity"]
                )
                if derivation_manifest_path.parent.name != derivation.derivation_key:
                    raise ValueError("derivation path identity changed")
                normalizer = OrganizerTextNormalizer(
                    version=derivation.extractor_version,
                    parser_version=derivation.parser_version,
                    scoring_version=derivation.scoring_normalizer_version,
                )
                cache = RetrievalCache(
                    retrieval_root,
                    DocumentStore(cache_root / "documents" / "v1"),
                    normalizer,
                )
                if cache.lookup_complete(identity, derivation, query_text) is None:
                    raise ValueError("retrieval derivation is incomplete")
                hits = derivation_manifest_path.parent / "hits.json"
                allowed.update({derivation_manifest_path.resolve(), hits.resolve()})
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise CacheBundleIntegrityError(
                f"retrieval document closure or cache entry is invalid: {manifest_path}"
            ) from exc
    return allowed


def _collect_cache_members(
    config: FacetPilotConfig,
    topic_id: str,
) -> list[_SourceMember]:
    cache_root = repo_cache_root(config.root_dir).resolve()
    if not cache_root.exists():
        return []
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise CacheBundleIntegrityError("cache root is not a safe directory")
    members: list[_SourceMember] = []
    document_closure: set[str] = set()
    retrieval_root = cache_root / "retrieval"
    if retrieval_root.is_dir():
        for manifest_path in sorted(retrieval_root.rglob("derivation-manifest.json")):
            _require_regular_source(manifest_path, cache_root)
            manifest = _strict_json(
                manifest_path.read_bytes(), f"retrieval derivation {manifest_path}"
            )
            if not isinstance(manifest, dict) or not isinstance(
                manifest.get("document_closure"), list
            ):
                raise CacheBundleIntegrityError(
                    "retrieval derivation has no valid document closure"
                )
            for raw_digest in manifest["document_closure"]:
                document_closure.add(
                    _require_digest(raw_digest, "retrieval document closure digest")
                )
    validated_retrieval_files = _validated_retrieval_files(config, cache_root)
    for source in sorted(cache_root.rglob("*")):
        if source.is_dir() and not source.is_symlink():
            continue
        relative = PurePosixPath(source.relative_to(cache_root).as_posix())
        if not relative.parts or relative.parts[0].casefold() not in _CACHE_TOP_LEVEL:
            continue
        # Document objects are selected only through authenticated retrieval
        # derivation closures; copying a whole corpus CAS would be both unsafe
        # and non-topic-specific.
        if relative.parts[0].casefold() == "documents":
            continue
        if _is_forbidden(relative):
            continue
        top_level = relative.parts[0].casefold()
        if (
            top_level == "retrieval"
            and source.resolve() not in validated_retrieval_files
        ):
            raise CacheBundleIntegrityError(
                "retrieval document closure/cache contains an unsealed or "
                f"undeclared file: {source}"
            )
        if top_level == "planning-cache-v1":
            _validate_planning_entry(cache_root, source, topic_id)
        elif top_level == "similarity-cache-v1":
            _validate_similarity_entry(cache_root, source)
        elif top_level == "canonical":
            if (
                len(relative.parts) != 4
                or relative.parts[2].casefold() != "validated"
                or not relative.name.endswith(".json")
            ):
                raise CacheBundleIntegrityError(
                    f"canonical cache contains an undeclared portable path: {source}"
                )
            _require_digest(relative.stem, "canonical request sha256")
            _validate_canonical_entry(source)
        kind = "similarity" if top_level == "similarity-cache-v1" else "immutable"
        members.append(
            _source_member(
                source,
                source_root=cache_root,
                archive_path=(PurePosixPath("cache") / relative).as_posix(),
                kind=kind,
            )
        )
    document_root = cache_root / "documents" / "v1"
    for digest in sorted(document_closure):
        source = document_root / "sha256" / digest[:2] / f"{digest}.utf8"
        try:
            size, actual = _file_receipt(source)
        except OSError as exc:
            raise CacheBundleIntegrityError(
                f"retrieval document closure is incomplete: {digest}"
            ) from exc
        if actual != digest:
            raise CacheBundleIntegrityError(
                f"retrieval document closure digest mismatch: {digest}"
            )
        try:
            source.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CacheBundleIntegrityError(
                f"retrieval document closure is not UTF-8: {digest}"
            ) from exc
        if size > MAX_MEMBER_BYTES:
            raise CacheBundleIntegrityError(
                f"retrieval document closure member is too large: {digest}"
            )
        members.append(
            _source_member(
                source,
                source_root=cache_root,
                archive_path=(
                    PurePosixPath("cache")
                    / PurePosixPath(source.relative_to(cache_root).as_posix())
                ).as_posix(),
                kind="immutable",
            )
        )
    return members


def _score_cache_for_database(score_root: Path, database: Path):
    """Discover and validate one score context, then return its read-only API."""
    from trec_rag.rerank_score_cache import (
        SCORE_CACHE_CONTEXT_VERSION,
        GlobalScoreCache,
        ScoreCacheContext,
    )

    for suffix in ("-wal", "-shm"):
        if database.with_name(database.name + suffix).exists():
            raise CacheBundleIntegrityError(
                f"score cache must be quiescent before portable export: {database}"
            )
    uri = f"file:{quote(str(database.resolve()), safe='/')}?mode=ro&immutable=1"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            rows = dict(connection.execute("SELECT key, value FROM cache_meta"))
        if set(rows) != {"schema_version", "context_sha256", "context_json"}:
            raise ValueError("score cache metadata fields differ")
        context_value = _strict_json(
            rows["context_json"].encode("utf-8"), "score cache context"
        )
        if not isinstance(context_value, dict):
            raise ValueError("score cache context is not an object")
        if (
            context_value.pop("context_schema_version", None)
            != SCORE_CACHE_CONTEXT_VERSION
        ):
            raise ValueError("score cache context schema changed")
        context = ScoreCacheContext(**context_value)
        cache = GlobalScoreCache(score_root, context, read_only=True)
        if cache.path.resolve() != database.resolve():
            raise ValueError("score cache database path differs from its context")
        if cache.context_sha256 != rows["context_sha256"]:
            raise ValueError("score cache context digest differs")
        return cache
    except (sqlite3.Error, TypeError, ValueError, OSError) as exc:
        raise CacheBundleIntegrityError(
            f"score cache is not safely portable: {database}"
        ) from exc


def _collect_score_members(
    config: FacetPilotConfig,
    staging_root: Path,
) -> list[_SourceMember]:
    """Export SQLite scores through the public portable-cache adapter."""
    score_root = config.passage.score_cache_dir
    if not score_root.exists():
        return []
    if score_root.is_symlink() or not score_root.is_dir():
        raise CacheBundleIntegrityError("score cache root is not a safe directory")
    results: list[_SourceMember] = []
    seen_contexts: set[str] = set()
    for database in sorted(score_root.rglob("*.sqlite3")):
        _require_regular_source(database, score_root)
        cache = _score_cache_for_database(score_root, database)
        context_sha256 = cache.context_sha256
        if context_sha256 in seen_contexts:
            raise CacheBundleIntegrityError(
                f"duplicate score cache context: {context_sha256}"
            )
        seen_contexts.add(context_sha256)
        destination = staging_root / f"{context_sha256}.jsonl"
        try:
            export_receipt = cache.export_portable_jsonl(destination)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise CacheBundleIntegrityError(
                f"score cache portable export failed: {database}"
            ) from exc
        size, digest = _file_receipt(destination)
        if (
            export_receipt.get("context_sha256") != context_sha256
            or export_receipt.get("sha256") != digest
        ):
            raise CacheBundleIntegrityError(
                f"score cache portable export receipt differs: {database}"
            )
        results.append(
            _source_member(
                destination,
                source_root=staging_root,
                archive_path=f"portable-scores/{context_sha256}.jsonl",
                kind="score",
            )
        )
    return results


def _manifest_payload(
    *,
    config: FacetPilotConfig,
    config_sha256: str,
    topic_id: str,
    members: Sequence[_SourceMember],
) -> dict[str, object]:
    return {
        "experiment_id": config.experiment.id,
        "members": [
            {
                "kind": member.kind,
                "mode": 0o600,
                "path": member.archive_path,
                "sha256": member.sha256,
                "size": member.size,
            }
            for member in members
        ],
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_config_sha256": config_sha256,
        "topic_id": topic_id,
    }


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o600
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_archive(
    destination: Path,
    manifest_bytes: bytes,
    members: Sequence[_SourceMember],
) -> None:
    compressor = zstandard.ZstdCompressor(
        level=19,
        threads=0,
        write_checksum=True,
        write_content_size=False,
    )
    with destination.open("xb") as raw_sink:
        os.fchmod(raw_sink.fileno(), 0o600)
        with compressor.stream_writer(raw_sink, closefd=False) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w|",
                format=tarfile.USTAR_FORMAT,
            ) as archive:
                archive.addfile(
                    _tar_info(BUNDLE_MANIFEST_NAME, len(manifest_bytes)),
                    io.BytesIO(manifest_bytes),
                )
                for member in members:
                    with member.source_path.open("rb") as source:
                        digesting = _DigestingReader(source)
                        archive.addfile(
                            _tar_info(member.archive_path, member.size),
                            digesting,
                        )
                    if (
                        digesting.count != member.size
                        or digesting.hexdigest != member.sha256
                    ):
                        raise CacheBundleIntegrityError(
                            f"bundle source changed while packing: {member.source_path}"
                        )
        raw_sink.flush()
        os.fsync(raw_sink.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        metadata = destination.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise CacheBundleConflictError(
                f"immutable bundle output is not a regular file: {destination}"
            )
        if _file_receipt(temporary) != _file_receipt(destination):
            raise CacheBundleConflictError(
                f"immutable bundle output already differs: {destination}"
            )
    else:
        os.chmod(destination, 0o600)
        _fsync_directory(destination.parent)


def pack_bundle(
    config_path: str | Path,
    topic_id: str,
    destination: str | Path,
) -> BundleReceipt:
    """Pack one sealed topic and its portable cache into a deterministic shard."""
    if not isinstance(topic_id, str) or not topic_id.strip():
        raise ValueError("topic must be non-empty text")
    topic_id = topic_id.strip()
    bundle_dir = _require_absolute_directory(
        Path(destination), "destination", create=True
    )
    unexpected = {
        entry.name
        for entry in bundle_dir.iterdir()
        if entry.name not in {BUNDLE_ARCHIVE_NAME, BUNDLE_COMPLETE_NAME}
    }
    if unexpected:
        raise CacheBundleIntegrityError(
            f"destination contains undeclared bundle files: {sorted(unexpected)!r}"
        )
    source_config = Path(config_path).resolve()
    config_bytes = source_config.read_bytes()
    config = load_facet_pilot_config(source_config, source_bytes=config_bytes)
    selected = select_configured_topics(config, topic_ids=(topic_id,))
    if len(selected) != 1 or selected[0].id != topic_id:
        raise CacheBundleIntegrityError("topic selection changed while packing")

    with tempfile.TemporaryDirectory(prefix="trec-rag-score-export-") as temporary:
        return _pack_validated_sources(
            config=config,
            config_bytes=config_bytes,
            source_config=source_config,
            topic=selected[0],
            topic_id=topic_id,
            bundle_dir=bundle_dir,
            score_staging=Path(temporary),
        )


def _pack_validated_sources(
    *,
    config: FacetPilotConfig,
    config_bytes: bytes,
    source_config: Path,
    topic: Any,
    topic_id: str,
    bundle_dir: Path,
    score_staging: Path,
) -> BundleReceipt:
    """Build and publish a bundle while portable score exports remain staged."""

    sources = [
        _source_member(
            source_config,
            source_root=config.root_dir,
            archive_path="source-config/config.yaml",
            kind="config",
        ),
        *_collect_cache_members(config, topic_id),
        *_collect_score_members(config, score_staging),
        *_collect_topic_members(
            config,
            topic,
            config_path=source_config,
            config_bytes=config_bytes,
        ),
    ]
    sources.sort(key=lambda member: member.archive_path)
    seen: dict[str, str] = {}
    for source in sources:
        _validate_archive_path(source.archive_path, seen)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if sources[0].source_path != source_config or sources[0].sha256 != config_sha256:
        # Sorting may move the config; bind it without relying on its position.
        config_member = next(
            member
            for member in sources
            if member.archive_path == "source-config/config.yaml"
        )
        if config_member.sha256 != config_sha256:
            raise CacheBundleIntegrityError("source config changed while packing")
    manifest_bytes = _canonical_json(
        _manifest_payload(
            config=config,
            config_sha256=config_sha256,
            topic_id=topic_id,
            members=sources,
        )
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{BUNDLE_ARCHIVE_NAME}.", suffix=".tmp", dir=bundle_dir
    )
    os.close(descriptor)
    temporary_archive = Path(temporary_name)
    temporary_archive.unlink()
    try:
        _write_archive(temporary_archive, manifest_bytes, sources)
        archive_size, archive_sha256 = _file_receipt(temporary_archive)
        if archive_size > MAX_COMPRESSED_BYTES:
            raise CacheBundleIntegrityError("compressed bundle size limit exceeded")
        receipt = BundleReceipt(
            topic_id=topic_id,
            archive_sha256=archive_sha256,
            archive_size=archive_size,
            manifest_sha256=manifest_sha256,
            member_count=len(sources),
        )
        complete_bytes = _canonical_json(
            {
                "archive": BUNDLE_ARCHIVE_NAME,
                "archive_sha256": receipt.archive_sha256,
                "archive_size": receipt.archive_size,
                "manifest_sha256": receipt.manifest_sha256,
                "member_count": receipt.member_count,
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "topic_id": receipt.topic_id,
            }
        )
        _publish_immutable(temporary_archive, bundle_dir / BUNDLE_ARCHIVE_NAME)
        marker_tmp = bundle_dir / f".{BUNDLE_COMPLETE_NAME}.{os.getpid()}.tmp"
        try:
            with marker_tmp.open("xb") as sink:
                os.fchmod(sink.fileno(), 0o600)
                sink.write(complete_bytes)
                sink.flush()
                os.fsync(sink.fileno())
            _publish_immutable(marker_tmp, bundle_dir / BUNDLE_COMPLETE_NAME)
        finally:
            marker_tmp.unlink(missing_ok=True)
        return receipt
    finally:
        temporary_archive.unlink(missing_ok=True)


def _parse_complete(bundle_dir: Path) -> BundleReceipt:
    marker = bundle_dir / BUNDLE_COMPLETE_NAME
    archive = bundle_dir / BUNDLE_ARCHIVE_NAME
    for path in (marker, archive):
        if (
            path.is_symlink()
            or not path.is_file()
            or not stat.S_ISREG(path.lstat().st_mode)
        ):
            raise CacheBundleIntegrityError(
                f"bundle is missing regular file {path.name}"
            )
    if marker.stat().st_size > MAX_MANIFEST_BYTES:
        raise CacheBundleIntegrityError("bundle completion marker size limit exceeded")
    marker_bytes = marker.read_bytes()
    value = _require_exact_fields(
        _strict_json(marker_bytes, "bundle completion marker"),
        {
            "archive",
            "archive_sha256",
            "archive_size",
            "manifest_sha256",
            "member_count",
            "schema_version",
            "topic_id",
        },
        "bundle completion marker",
    )
    if _canonical_json(value) != marker_bytes:
        raise CacheBundleIntegrityError(
            "bundle completion marker is not canonical JSON"
        )
    if value["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise CacheBundleIntegrityError("unsupported bundle completion schema")
    if value["archive"] != BUNDLE_ARCHIVE_NAME:
        raise CacheBundleIntegrityError("bundle completion archive name changed")
    topic_id = value["topic_id"]
    if not isinstance(topic_id, str) or not topic_id:
        raise CacheBundleIntegrityError("bundle completion topic is invalid")
    return BundleReceipt(
        topic_id=topic_id,
        archive_sha256=_require_digest(value["archive_sha256"], "archive_sha256"),
        archive_size=_require_nonnegative_int(value["archive_size"], "archive_size"),
        manifest_sha256=_require_digest(value["manifest_sha256"], "manifest_sha256"),
        member_count=_require_nonnegative_int(value["member_count"], "member_count"),
    )


def _parse_manifest(
    value: bytes,
) -> tuple[str, str, str, tuple[BundleMember, ...]]:
    raw = _require_exact_fields(
        _strict_json(value, "bundle manifest"),
        {
            "experiment_id",
            "members",
            "schema_version",
            "source_config_sha256",
            "topic_id",
        },
        "bundle manifest",
    )
    if _canonical_json(raw) != value:
        raise CacheBundleIntegrityError("bundle manifest is not canonical JSON")
    if raw["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise CacheBundleIntegrityError("unsupported bundle manifest schema")
    topic_id = raw["topic_id"]
    experiment_id = raw["experiment_id"]
    if not isinstance(topic_id, str) or not topic_id:
        raise CacheBundleIntegrityError("bundle manifest topic is invalid")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise CacheBundleIntegrityError("bundle manifest experiment is invalid")
    source_config_sha256 = _require_digest(
        raw["source_config_sha256"], "source_config_sha256"
    )
    members_raw = raw["members"]
    if not isinstance(members_raw, list) or len(members_raw) > MAX_MEMBERS:
        raise CacheBundleIntegrityError("bundle manifest member count is invalid")
    seen: dict[str, str] = {
        unicodedata.normalize(
            "NFC", BUNDLE_MANIFEST_NAME
        ).casefold(): BUNDLE_MANIFEST_NAME
    }
    members: list[BundleMember] = []
    for index, item in enumerate(members_raw):
        row = _require_exact_fields(
            item,
            {"kind", "mode", "path", "sha256", "size"},
            f"bundle manifest member {index}",
        )
        path = row["path"]
        kind = row["kind"]
        if not isinstance(path, str):
            raise CacheBundleIntegrityError("bundle member path must be text")
        _validate_archive_path(path, seen)
        if kind not in {"checkpoint", "config", "immutable", "similarity", "score"}:
            raise CacheBundleIntegrityError(f"unsupported bundle member kind: {kind!r}")
        size = _require_nonnegative_int(row["size"], "bundle member size")
        if size > MAX_MEMBER_BYTES:
            raise CacheBundleIntegrityError("bundle member size limit exceeded")
        if row["mode"] != 0o600:
            raise CacheBundleIntegrityError("bundle member mode is not canonical")
        members.append(
            BundleMember(
                path=path,
                kind=kind,
                size=size,
                sha256=_require_digest(row["sha256"], "bundle member sha256"),
            )
        )
    if [member.path for member in members] != sorted(member.path for member in members):
        raise CacheBundleIntegrityError("bundle manifest members are not sorted")
    canonical_paths = {
        unicodedata.normalize("NFC", member.path).casefold() for member in members
    }
    for member in members:
        canonical = PurePosixPath(unicodedata.normalize("NFC", member.path).casefold())
        if any(
            parent.as_posix() in canonical_paths
            for parent in canonical.parents
            if parent != PurePosixPath(".")
        ):
            raise CacheBundleIntegrityError(
                "bundle member paths have an ancestor collision"
            )
    return topic_id, experiment_id, source_config_sha256, tuple(members)


def _validate_member_layout(
    *,
    topic_id: str,
    experiment_id: str,
    source_config_sha256: str,
    members: Sequence[BundleMember],
) -> None:
    """Enforce the exact semantic namespaces emitted by ``pack``."""
    configs = [member for member in members if member.kind == "config"]
    if len(configs) != 1 or configs[0].path != "source-config/config.yaml":
        raise CacheBundleIntegrityError(
            "bundle requires exactly one authenticated source config"
        )
    if configs[0].sha256 != source_config_sha256:
        raise CacheBundleIntegrityError("source config digest differs from manifest")
    topic_prefix = ("outputs", experiment_id, topic_id)
    run_seed_path = PurePosixPath("outputs", experiment_id, _SEED_DECOMPOSITION_RECEIPT)
    for member in members:
        path = PurePosixPath(member.path)
        if member.kind == "config":
            if path != PurePosixPath("source-config/config.yaml"):
                raise CacheBundleIntegrityError(
                    "config kind/path combination is invalid"
                )
            continue
        if path.parts[:1] == ("source-config",):
            raise CacheBundleIntegrityError(
                "source-config path has the wrong member kind"
            )
        if member.kind == "checkpoint":
            if path == run_seed_path:
                continue
            if path.parts[:3] != topic_prefix or len(path.parts) < 4:
                raise CacheBundleIntegrityError(
                    "checkpoint kind/path combination is invalid"
                )
            relative = PurePosixPath(*path.parts[3:])
            if _is_forbidden(relative) and relative != PurePosixPath("records.sqlite3"):
                raise CacheBundleIntegrityError(
                    f"checkpoint member path is forbidden: {member.path}"
                )
            continue
        if member.kind == "score":
            if (
                len(path.parts) != 2
                or path.parts[0] != "portable-scores"
                or not path.name.endswith(".jsonl")
                or _require_digest(path.stem, "portable score identity") != path.stem
            ):
                raise CacheBundleIntegrityError(
                    "portable score kind/path combination is invalid"
                )
            continue
        if path.parts[:1] != ("cache",) or len(path.parts) < 3:
            raise CacheBundleIntegrityError("cache kind/path combination is invalid")
        relative = PurePosixPath(*path.parts[1:])
        if _is_forbidden(relative):
            raise CacheBundleIntegrityError(
                f"cache member path is forbidden: {member.path}"
            )
        namespace = relative.parts[0]
        if namespace == "planning-cache-v1":
            valid = (
                member.kind == "immutable"
                and len(relative.parts) == 3
                and len(relative.parts[1]) == 2
                and relative.name.endswith(".json")
                and _require_digest(relative.stem, "planning cache identity")
                == relative.stem
                and relative.parts[1] == relative.stem[:2]
            )
        elif namespace == "similarity-cache-v1":
            valid = (
                member.kind == "similarity"
                and len(relative.parts) == 3
                and len(relative.parts[1]) == 2
                and relative.name.endswith(".json")
                and _require_digest(relative.stem, "similarity cache identity")
                == relative.stem
                and relative.parts[1] == relative.stem[:2]
            )
        elif namespace == "documents":
            valid = (
                member.kind == "immutable"
                and len(relative.parts) == 5
                and relative.parts[1:3] == ("v1", "sha256")
                and len(relative.parts[3]) == 2
                and relative.name.endswith(".utf8")
                and _require_digest(relative.stem, "document identity") == relative.stem
                and relative.parts[3] == relative.stem[:2]
            )
        elif namespace == "retrieval":
            valid = member.kind == "immutable" and len(relative.parts) >= 3
        elif namespace == "canonical":
            valid = (
                member.kind == "immutable"
                and len(relative.parts) == 4
                and relative.parts[2] == "validated"
                and relative.name.endswith(".json")
                and _require_digest(relative.stem, "canonical request identity")
                == relative.stem
            )
        else:
            valid = False
        if not valid:
            raise CacheBundleIntegrityError(
                f"bundle member kind/path combination is invalid: {member.path}"
            )


def _consume_member(
    source: BinaryIO,
    expected: BundleMember,
    destination: Path | None = None,
) -> None:
    digest = hashlib.sha256()
    count = 0
    sink: BinaryIO | None = None
    try:
        if destination is not None:
            destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            sink = destination.open("xb")
            os.fchmod(sink.fileno(), 0o600)
        while True:
            chunk = source.read(min(_COPY_CHUNK, expected.size - count + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > expected.size:
                raise CacheBundleIntegrityError(
                    f"bundle member size mismatch: {expected.path}"
                )
            digest.update(chunk)
            if sink is not None:
                sink.write(chunk)
        if sink is not None:
            sink.flush()
            os.fsync(sink.fileno())
    finally:
        if sink is not None:
            sink.close()
    if count != expected.size:
        raise CacheBundleIntegrityError(f"bundle member size mismatch: {expected.path}")
    if digest.hexdigest() != expected.sha256:
        raise CacheBundleIntegrityError(
            f"bundle member digest mismatch: {expected.path}"
        )


def _verify_archive(
    archive_path: Path,
    receipt: BundleReceipt,
    *,
    extraction_root: Path | None = None,
) -> tuple[str, str, tuple[BundleMember, ...]]:
    if archive_path.stat().st_size > MAX_COMPRESSED_BYTES:
        raise CacheBundleIntegrityError("compressed bundle size limit exceeded")
    archive_size, archive_digest = _file_receipt(archive_path)
    if archive_size != receipt.archive_size or archive_digest != receipt.archive_sha256:
        raise CacheBundleIntegrityError("bundle archive digest or size mismatch")
    with archive_path.open("rb") as compressed_source:
        decompressor = zstandard.ZstdDecompressor(
            max_window_size=MAX_ZSTD_WINDOW_BYTES
        ).stream_reader(compressed_source, read_across_frames=True)
        bounded = _BoundedReader(decompressor, MAX_DECOMPRESSED_BYTES)
        try:
            with tarfile.open(fileobj=bounded, mode="r|") as archive:
                manifest_bytes: bytes | None = None
                declared: tuple[BundleMember, ...] | None = None
                topic_id = ""
                experiment_id = ""
                source_config_sha256 = ""
                seen: dict[str, str] = {}
                member_index = -1
                for member_index, info in enumerate(archive):
                    if member_index > MAX_MEMBERS:
                        raise CacheBundleIntegrityError(
                            "archive member-count limit exceeded"
                        )
                    name = _validate_archive_path(info.name, seen)
                    if not info.isfile() or info.issym() or info.islnk():
                        raise CacheBundleIntegrityError(
                            f"archive member is not a regular file: {name}"
                        )
                    if info.size < 0 or info.size > MAX_MEMBER_BYTES:
                        raise CacheBundleIntegrityError(
                            f"archive member size limit exceeded: {name}"
                        )
                    if (
                        info.mtime != 0
                        or info.uid != 0
                        or info.gid != 0
                        or info.mode != 0o600
                        or info.uname not in {"", None}
                        or info.gname not in {"", None}
                    ):
                        raise CacheBundleIntegrityError(
                            f"archive member metadata is not canonical: {name}"
                        )
                    extracted = archive.extractfile(info)
                    if extracted is None:
                        raise CacheBundleIntegrityError(
                            f"archive member body is unavailable: {name}"
                        )
                    if member_index == 0:
                        if (
                            name != BUNDLE_MANIFEST_NAME
                            or info.size > MAX_MANIFEST_BYTES
                        ):
                            raise CacheBundleIntegrityError(
                                "bundle manifest must be the bounded first archive member"
                            )
                        manifest_bytes = extracted.read(MAX_MANIFEST_BYTES + 1)
                        if len(manifest_bytes) != info.size:
                            raise CacheBundleIntegrityError(
                                "bundle manifest size mismatch"
                            )
                        if (
                            hashlib.sha256(manifest_bytes).hexdigest()
                            != receipt.manifest_sha256
                        ):
                            raise CacheBundleIntegrityError(
                                "bundle manifest digest mismatch"
                            )
                        (
                            topic_id,
                            experiment_id,
                            source_config_sha256,
                            declared,
                        ) = _parse_manifest(manifest_bytes)
                        if topic_id != receipt.topic_id:
                            raise CacheBundleIntegrityError(
                                "bundle topic identity mismatch"
                            )
                        if len(declared) != receipt.member_count:
                            raise CacheBundleIntegrityError(
                                "bundle member count mismatch"
                            )
                        continue
                    assert declared is not None
                    declared_index = member_index - 1
                    if declared_index >= len(declared):
                        raise CacheBundleIntegrityError(
                            f"undeclared archive member: {name}"
                        )
                    expected = declared[declared_index]
                    if name != expected.path:
                        raise CacheBundleIntegrityError(
                            f"undeclared or reordered archive member: {name}"
                        )
                    if info.size != expected.size:
                        raise CacheBundleIntegrityError(
                            f"bundle member header size mismatch: {name}"
                        )
                    extraction_path = (
                        None
                        if extraction_root is None
                        else extraction_root.joinpath(*PurePosixPath(name).parts)
                    )
                    _consume_member(extracted, expected, extraction_path)
                if manifest_bytes is None or declared is None:
                    raise CacheBundleIntegrityError("bundle archive has no manifest")
                if member_index != len(declared):
                    raise CacheBundleIntegrityError(
                        "bundle archive is missing declared members"
                    )
                # tarfile stops at its end markers; consume the decompressor so
                # hidden bytes or concatenated zstd frames cannot sit behind a
                # valid declared archive.
                while bounded.read(_COPY_CHUNK):
                    pass
                logical_bytes = 512 + ((len(manifest_bytes) + 511) // 512) * 512
                logical_bytes += sum(
                    512 + ((member.size + 511) // 512) * 512 for member in declared
                )
                expected_tar_bytes = (
                    (logical_bytes + 1024 + tarfile.RECORDSIZE - 1)
                    // tarfile.RECORDSIZE
                ) * tarfile.RECORDSIZE
                if bounded.count != expected_tar_bytes:
                    raise CacheBundleIntegrityError(
                        "bundle has trailing or non-canonical decompressed bytes"
                    )
                _validate_member_layout(
                    topic_id=topic_id,
                    experiment_id=experiment_id,
                    source_config_sha256=source_config_sha256,
                    members=declared,
                )
        except (tarfile.TarError, zstandard.ZstdError, EOFError) as exc:
            raise CacheBundleIntegrityError("bundle archive stream is invalid") from exc
        finally:
            decompressor.close()
    return experiment_id, source_config_sha256, declared


def _portable_cache_path(
    config_bytes: bytes, section: str, field: str
) -> PurePosixPath:
    try:
        raw = yaml.safe_load(config_bytes)
        value = raw[section][field]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise CacheBundleIntegrityError("source config cache path is invalid") from exc
    if not isinstance(value, str) or not value or "\\" in value:
        raise CacheBundleIntegrityError("source config cache path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CacheBundleIntegrityError("source config cache path is not portable")
    parts = path.parts[1:] if path.parts[:1] == ("cache",) else path.parts
    if not parts:
        raise CacheBundleIntegrityError("source config cache path is invalid")
    return PurePosixPath(*parts)


def _staged_config(
    extraction_root: Path,
    *,
    experiment_id: str,
) -> tuple[FacetPilotConfig, bytes]:
    config_path = extraction_root / "source-config/config.yaml"
    config_bytes = config_path.read_bytes()
    # ``find_repo_root`` needs a local boundary; this marker lives only in the
    # isolated verifier temporary directory and is never an archive member.
    (extraction_root / "AGENTS.md").write_bytes(b"bundle validation root\n")
    try:
        loaded = load_facet_pilot_config(config_path, source_bytes=config_bytes)
    except (OSError, TypeError, ValueError) as exc:
        raise CacheBundleIntegrityError(
            "authenticated source config is invalid"
        ) from exc
    if loaded.experiment.id != experiment_id:
        raise CacheBundleIntegrityError(
            "source config experiment differs from bundle manifest"
        )
    retrieval_relative = _portable_cache_path(config_bytes, "retrieval", "cache_dir")
    score_relative = _portable_cache_path(config_bytes, "passage", "score_cache_dir")
    staged = replace(
        loaded,
        root_dir=extraction_root,
        retrieval=replace(
            loaded.retrieval,
            cache_dir=extraction_root / "cache" / retrieval_relative,
        ),
        passage=replace(
            loaded.passage,
            score_cache_dir=extraction_root / "cache" / score_relative,
        ),
    )
    return staged, config_bytes


def _validate_document_member(source: Path) -> str:
    digest = source.stem
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise CacheBundleIntegrityError("document cache identity differs from content")
    try:
        source.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CacheBundleIntegrityError("document cache member is not UTF-8") from exc
    return digest


def _validate_extracted_cache(
    extraction_root: Path,
    *,
    config: FacetPilotConfig,
    topic_id: str,
    members: Sequence[BundleMember],
) -> None:
    cache_root = extraction_root / "cache"
    cache_members = [
        member
        for member in members
        if PurePosixPath(member.path).parts[:1] == ("cache",)
    ]
    retrieval_sources: set[Path] = set()
    document_digests: set[str] = set()
    for member in cache_members:
        relative = PurePosixPath(*PurePosixPath(member.path).parts[1:])
        source = cache_root.joinpath(*relative.parts)
        namespace = relative.parts[0]
        if namespace == "planning-cache-v1":
            _validate_planning_entry(cache_root, source, topic_id)
        elif namespace == "similarity-cache-v1":
            _validate_similarity_entry(cache_root, source)
        elif namespace == "canonical":
            _validate_canonical_entry(source)
        elif namespace == "documents":
            document_digests.add(_validate_document_member(source))
        elif namespace == "retrieval":
            retrieval_sources.add(source.resolve())
        else:  # layout validation should make this unreachable
            raise CacheBundleIntegrityError("bundle cache namespace changed")

    allowed_retrieval = _validated_retrieval_files(config, cache_root)
    if retrieval_sources != allowed_retrieval:
        raise CacheBundleIntegrityError(
            "retrieval cache bundle differs from its authenticated entry closure"
        )
    expected_documents: set[str] = set()
    retrieval_root = cache_root / "retrieval"
    if retrieval_root.exists():
        for manifest_path in sorted(retrieval_root.rglob("derivation-manifest.json")):
            value = _strict_json(
                manifest_path.read_bytes(), "retrieval derivation manifest"
            )
            if not isinstance(value, dict) or not isinstance(
                value.get("document_closure"), list
            ):
                raise CacheBundleIntegrityError(
                    "retrieval derivation document closure is invalid"
                )
            expected_documents.update(
                _require_digest(item, "retrieval document closure digest")
                for item in value["document_closure"]
            )
    if document_digests != expected_documents:
        raise CacheBundleIntegrityError(
            "document cache bundle differs from retrieval closure"
        )

    for member in members:
        if member.kind == "score":
            source = extraction_root.joinpath(*PurePosixPath(member.path).parts)
            _portable_score_context_and_rows(source)


def _validate_extracted_checkpoint(
    extraction_root: Path,
    *,
    config: FacetPilotConfig,
    config_bytes: bytes,
    topic_id: str,
    members: Sequence[BundleMember],
) -> None:
    from trec_rag.topic_dispatch import TopicJob, read_topic_receipt

    topic_root = config.output_dir / topic_id
    expected: set[str] = set()
    prefix = PurePosixPath("outputs", config.experiment.id, topic_id)
    for phase in ("retrieval", "scoring", "canonical"):
        for source in _checkpoint_manifest_sources(topic_root, phase):
            relative = PurePosixPath(source.relative_to(topic_root).as_posix())
            expected.add((prefix / relative).as_posix())
    receipt_path = topic_root / _TOPIC_RECEIPT
    _require_regular_source(receipt_path, topic_root)
    expected.add((prefix / _TOPIC_RECEIPT).as_posix())
    for source, source_root in _decomposition_sources(config, topic_root):
        if source_root == config.output_dir:
            expected.add(
                PurePosixPath("outputs", config.experiment.id, source.name).as_posix()
            )
        else:
            expected.add(
                (
                    prefix / PurePosixPath(source.relative_to(topic_root).as_posix())
                ).as_posix()
            )

    actual = {member.path for member in members if member.kind == "checkpoint"}
    projection_manifest = topic_root / "canonical/retrieval-projection-manifest.json"
    operation_path = topic_root / _CACHE_OPERATION_RECEIPT
    _validate_online_cache_operation_receipt(
        operation_path,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        run_id=config.run_id,
        topic_id=topic_id,
        projection_manifest_sha256=hashlib.sha256(
            projection_manifest.read_bytes()
        ).hexdigest(),
    )
    expected.add((prefix / _CACHE_OPERATION_RECEIPT).as_posix())
    if actual != expected:
        raise CacheBundleIntegrityError(
            "checkpoint members differ from authenticated receipt closure"
        )
    config_path = extraction_root / "source-config/config.yaml"
    try:
        job = TopicJob(
            topic_id=topic_id,
            run_id=config.run_id,
            config_path=config_path.resolve(),
            config_bytes=config_bytes,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            topic_root=topic_root.resolve(),
        )
        receipt = read_topic_receipt(job)
    except (OSError, TypeError, ValueError) as exc:
        raise CacheBundleIntegrityError(
            "checkpoint dispatch receipt is invalid"
        ) from exc
    if receipt is None:
        raise CacheBundleIntegrityError("checkpoint dispatch receipt is missing")


def _validate_extracted_semantics(
    extraction_root: Path,
    *,
    topic_id: str,
    experiment_id: str,
    members: Sequence[BundleMember],
) -> None:
    config, config_bytes = _staged_config(extraction_root, experiment_id=experiment_id)
    _validate_extracted_cache(
        extraction_root,
        config=config,
        topic_id=topic_id,
        members=members,
    )
    _validate_extracted_checkpoint(
        extraction_root,
        config=config,
        config_bytes=config_bytes,
        topic_id=topic_id,
        members=members,
    )


def verify_bundle(bundle_dir: str | Path) -> VerifiedBundle:
    """Authenticate structurally, then semantically stage in isolation."""
    root = Path(bundle_dir)
    if root.is_symlink() or not root.is_dir():
        raise CacheBundleIntegrityError("bundle directory is missing or unsafe")
    actual_names = {entry.name for entry in root.iterdir()}
    expected_names = {BUNDLE_ARCHIVE_NAME, BUNDLE_COMPLETE_NAME}
    if actual_names != expected_names:
        raise CacheBundleIntegrityError(
            "bundle directory contains undeclared files: "
            f"{sorted(actual_names - expected_names)!r}"
        )
    receipt = _parse_complete(root)
    experiment_id, source_config_sha256, members = _verify_archive(
        root / BUNDLE_ARCHIVE_NAME, receipt
    )
    # Hostile input completes an extraction-free pass first.  Only then is the
    # same authenticated stream staged under an isolated temporary directory
    # for cache/checkpoint semantic validation.
    with tempfile.TemporaryDirectory(prefix="trec-rag-bundle-verify-") as temporary:
        extraction_root = Path(temporary)
        staged_experiment, staged_config_sha256, staged_members = _verify_archive(
            root / BUNDLE_ARCHIVE_NAME,
            receipt,
            extraction_root=extraction_root,
        )
        if (
            staged_experiment != experiment_id
            or staged_config_sha256 != source_config_sha256
            or staged_members != members
        ):
            raise CacheBundleIntegrityError(
                "bundle changed between structural and semantic verification"
            )
        _validate_extracted_semantics(
            extraction_root,
            topic_id=receipt.topic_id,
            experiment_id=experiment_id,
            members=members,
        )
    return VerifiedBundle(
        bundle_dir=root.resolve(),
        topic_id=receipt.topic_id,
        experiment_id=experiment_id,
        archive_sha256=receipt.archive_sha256,
        archive_size=receipt.archive_size,
        manifest_sha256=receipt.manifest_sha256,
        source_config_sha256=source_config_sha256,
        members=members,
    )


def _validate_destination_root(path: str | Path, label: str) -> Path:
    root = Path(path)
    if not root.is_absolute():
        raise ValueError(f"{label} must be an absolute directory")
    if any(part in {".", ".."} for part in root.parts):
        raise CacheBundleIntegrityError(f"{label} must be lexically normalized")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise CacheBundleIntegrityError(
                f"{label} contains a symbolic-link component"
            )
        if current == root and not stat.S_ISDIR(metadata.st_mode):
            raise CacheBundleIntegrityError(f"{label} is not a safe directory")
    return root.resolve(strict=False)


def _secure_mkdir(path: Path) -> None:
    if not path.is_absolute():
        raise CacheBundleIntegrityError(
            "secure directory creation requires an absolute path"
        )
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                metadata = current.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise CacheBundleIntegrityError(
                        f"unsafe destination directory component: {current}"
                    )
            else:
                _fsync_directory(current.parent)
                continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise CacheBundleIntegrityError(
                f"unsafe destination directory component: {current}"
            )


def _write_durable_create_only(path: Path, content: bytes) -> None:
    _secure_mkdir(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            os.fchmod(sink.fileno(), 0o600)
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise CacheBundleConflictError(
                    f"durable merge state already differs: {path}"
                )
        else:
            os.chmod(path, 0o600)
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_merge_journal_content(
    *,
    state_name: str,
    prepare_value: object,
    prepare_bytes: bytes,
    conflicts_value: object,
    conflicts_bytes: bytes,
    completion_value: object,
    completion_receipt: MergeReceipt,
) -> None:
    prepare = _require_exact_fields(
        prepare_value,
        {
            "bundles",
            "cache_root",
            "merge_id",
            "operations",
            "outputs_root",
            "schema_version",
            "score_conflicts",
        },
        "merge prepare journal",
    )
    conflicts = _require_exact_fields(
        conflicts_value,
        {"conflicts", "merge_id", "schema_version"},
        "merge conflict audit",
    )
    if _canonical_json(prepare) != prepare_bytes:
        raise CacheBundleIntegrityError("merge prepare journal is not canonical")
    if _canonical_json(conflicts) != conflicts_bytes:
        raise CacheBundleIntegrityError("merge conflict audit is not canonical")
    if (
        prepare["schema_version"] != BUNDLE_SCHEMA_VERSION
        or conflicts["schema_version"] != BUNDLE_SCHEMA_VERSION
        or prepare["merge_id"] != state_name
        or conflicts["merge_id"] != state_name
        or completion_receipt.merge_id != state_name
    ):
        raise CacheBundleIntegrityError("merge journal identity changed")
    bundles = prepare["bundles"]
    operations = prepare["operations"]
    conflict_rows = conflicts["conflicts"]
    if not isinstance(bundles, list) or not isinstance(operations, list):
        raise CacheBundleIntegrityError("merge prepare journal lists are invalid")
    if not isinstance(conflict_rows, list):
        raise CacheBundleIntegrityError("merge conflict audit list is invalid")
    install_operation_keys: set[tuple[str, str]] = set()
    score_operation_count = 0
    for operation in operations:
        if not isinstance(operation, dict):
            raise CacheBundleIntegrityError("merge prepare operation is invalid")
        target_root = operation.get("target_root")
        relative_path = operation.get("relative_path")
        if target_root == "score-import" and operation.get("kind") == "score":
            score_operation_count += 1
        elif target_root in {"cache", "outputs"} and isinstance(relative_path, str):
            install_operation_keys.add((target_root, relative_path))
        else:
            raise CacheBundleIntegrityError("merge prepare operation target is invalid")
    kept_similarity_keys: set[tuple[str, str]] = set()
    similarity_row_count = 0
    for row in conflict_rows:
        if not isinstance(row, dict):
            raise CacheBundleIntegrityError("merge conflict row is invalid")
        if row.get("kind") != "similarity":
            continue
        similarity_row_count += 1
        target_root = row.get("target_root")
        relative_path = row.get("relative_path")
        key = (target_root, relative_path)
        if (
            row.get("resolution") != "kept-existing"
            or not isinstance(target_root, str)
            or not isinstance(relative_path, str)
            or key not in install_operation_keys
        ):
            raise CacheBundleIntegrityError(
                "merge similarity conflict does not match an operation"
            )
        kept_similarity_keys.add(key)
    if len(kept_similarity_keys) != similarity_row_count:
        raise CacheBundleIntegrityError("merge similarity conflict is duplicated")
    complete = _require_exact_fields(
        completion_value,
        {
            "bundle_count",
            "conflicts_sha256",
            "identical_count",
            "installed_count",
            "kept_conflict_count",
            "merge_id",
            "operation_count",
            "prepare_sha256",
            "schema_version",
            "score_import_count",
        },
        "merge completion receipt",
    )
    if (
        completion_receipt.bundle_count != len(bundles)
        or completion_receipt.operation_count != len(operations)
        or completion_receipt.installed_count + completion_receipt.identical_count
        != len(install_operation_keys) - len(kept_similarity_keys)
        or completion_receipt.kept_conflict_count != len(conflict_rows)
        or completion_receipt.score_import_count != score_operation_count
        or complete["schema_version"] != BUNDLE_SCHEMA_VERSION
    ):
        raise CacheBundleIntegrityError("merge completion counters are invalid")


def incomplete_merge_journals(cache_root: str | Path) -> tuple[Path, ...]:
    """Return prepare journals that have no valid committed completion receipt."""
    root = Path(cache_root)
    state_root = root / MERGE_STATE_DIRECTORY
    if not state_root.exists():
        return ()
    if state_root.is_symlink() or not state_root.is_dir():
        return (state_root,)
    incomplete: list[Path] = []
    for state in sorted(state_root.iterdir()):
        if state.is_symlink() or not state.is_dir():
            incomplete.append(state)
            continue
        prepare = state / MERGE_PREPARE_NAME
        if not prepare.is_file() or prepare.is_symlink():
            incomplete.append(state)
            continue
        complete = state / MERGE_COMPLETE_NAME
        if not complete.is_file() or complete.is_symlink():
            incomplete.append(prepare)
            continue
        try:
            prepare_bytes = prepare.read_bytes()
            complete_bytes = complete.read_bytes()
            prepare_value = _strict_json(prepare_bytes, "merge prepare journal")
            completion_receipt = _parse_merge_receipt(complete, complete_bytes)
            if (
                not isinstance(prepare_value, dict)
                or prepare_value.get("merge_id") != state.name
                or completion_receipt.merge_id != state.name
            ):
                incomplete.append(prepare)
                continue
            complete_value = _strict_json(complete_bytes, "merge completion receipt")
            if (
                not isinstance(complete_value, dict)
                or complete_value.get("prepare_sha256")
                != hashlib.sha256(prepare_bytes).hexdigest()
            ):
                incomplete.append(prepare)
                continue
            conflicts = state / MERGE_CONFLICTS_NAME
            if conflicts.is_symlink() or not conflicts.is_file():
                incomplete.append(prepare)
                continue
            conflicts_bytes = conflicts.read_bytes()
            if hashlib.sha256(conflicts_bytes).hexdigest() != complete_value.get(
                "conflicts_sha256"
            ):
                incomplete.append(prepare)
                continue
            _validate_merge_journal_content(
                state_name=state.name,
                prepare_value=prepare_value,
                prepare_bytes=prepare_bytes,
                conflicts_value=_strict_json(conflicts_bytes, "merge conflict audit"),
                conflicts_bytes=conflicts_bytes,
                completion_value=complete_value,
                completion_receipt=completion_receipt,
            )
        except (CacheBundleError, OSError):
            incomplete.append(prepare)
    return tuple(incomplete)


def assert_no_incomplete_cache_bundle_merge(cache_root: str | Path) -> None:
    """Fail closed before a runner uses a cache with an interrupted merge."""
    journals = incomplete_merge_journals(cache_root)
    if journals:
        raise CacheBundleIntegrityError(
            "cache root has an incomplete cache-bundle merge journal: "
            + ", ".join(str(path) for path in journals)
        )


def _existing_target_receipt(target: Path, root: Path) -> tuple[int, str] | None:
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise CacheBundleIntegrityError(
            "merge target escaped its destination root"
        ) from exc
    current = root
    if not current.exists():
        return None
    root_metadata = current.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise CacheBundleIntegrityError(f"merge destination root is unsafe: {root}")
    for part in relative.parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise CacheBundleIntegrityError(
                f"merge destination parent is unsafe: {current}"
            )
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CacheBundleIntegrityError(f"merge target is not a regular file: {target}")
    return _file_receipt(target)


def _install_create_only(operation: _InstallOperation, target: Path) -> bool:
    _secure_mkdir(target.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with (
            os.fdopen(descriptor, "wb") as sink,
            operation.staged_path.open("rb") as source,
        ):
            os.fchmod(sink.fileno(), 0o600)
            shutil.copyfileobj(source, sink, length=_COPY_CHUNK)
            sink.flush()
            os.fsync(sink.fileno())
        if _file_receipt(temporary) != (operation.member.size, operation.member.sha256):
            raise CacheBundleIntegrityError(
                "staged merge member changed during install"
            )
        try:
            os.link(temporary, target)
        except FileExistsError:
            metadata = target.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or _file_receipt(target)
                != (operation.member.size, operation.member.sha256)
            ):
                raise CacheBundleConflictError(
                    f"immutable merge target changed after preflight: {target}"
                )
            return False
        os.chmod(target, 0o600)
        _fsync_directory(target.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _merge_id(
    bundles: Sequence[VerifiedBundle],
    *,
    cache_root: Path,
    outputs_root: Path,
    score_conflicts: str,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "archives": [bundle.archive_sha256 for bundle in bundles],
                "cache_root": str(cache_root),
                "outputs_root": str(outputs_root),
                "score_conflicts": score_conflicts,
            }
        )
    ).hexdigest()


def _operation_payload(operation: _InstallOperation) -> dict[str, object]:
    return {
        "kind": operation.member.kind,
        "relative_path": operation.relative_path.as_posix(),
        "sha256": operation.member.sha256,
        "size": operation.member.size,
        "target_root": operation.target_root_name,
    }


def _score_operation_payload(operation: _ScoreOperation) -> dict[str, object]:
    return {
        "kind": "score",
        "relative_path": PurePosixPath(operation.member.path).as_posix(),
        "sha256": operation.member.sha256,
        "size": operation.member.size,
        "target_root": "score-import",
    }


def _prepare_operations(
    bundles: Sequence[VerifiedBundle],
    staging_root: Path,
) -> tuple[tuple[_InstallOperation, ...], tuple[_ScoreOperation, ...]]:
    by_target: dict[tuple[str, str], _InstallOperation] = {}
    scores_by_digest: dict[str, _ScoreOperation] = {}
    for bundle_index, bundle in enumerate(bundles):
        extracted = staging_root / str(bundle_index)
        receipt = BundleReceipt(
            topic_id=bundle.topic_id,
            archive_sha256=bundle.archive_sha256,
            archive_size=bundle.archive_size,
            manifest_sha256=bundle.manifest_sha256,
            member_count=len(bundle.members),
        )
        experiment_id, source_config_sha256, extracted_members = _verify_archive(
            bundle.bundle_dir / BUNDLE_ARCHIVE_NAME,
            receipt,
            extraction_root=extracted,
        )
        if (
            experiment_id != bundle.experiment_id
            or source_config_sha256 != bundle.source_config_sha256
            or extracted_members != bundle.members
        ):
            raise CacheBundleIntegrityError(
                "bundle changed between verification and staging"
            )
        for member in bundle.members:
            path = PurePosixPath(member.path)
            if member.kind == "config":
                if path.parts[:1] != ("source-config",):
                    raise CacheBundleIntegrityError(
                        "config member has an invalid archive path"
                    )
                continue
            if member.kind == "score":
                # Imported through the score-cache transaction adapter, never
                # installed as a database file.
                if path.parts[:1] != ("portable-scores",) or len(path.parts) != 2:
                    raise CacheBundleIntegrityError(
                        "portable score member has an invalid archive path"
                    )
                scores_by_digest.setdefault(
                    member.sha256,
                    _ScoreOperation(
                        member=member,
                        staged_path=extracted.joinpath(*path.parts),
                    ),
                )
                continue
            if path.parts[:1] == ("cache",):
                target_root_name = "cache"
            elif path.parts[:1] == ("outputs",):
                target_root_name = "outputs"
            else:
                raise CacheBundleIntegrityError(
                    f"installable bundle member has no target root: {member.path}"
                )
            if member.kind == "checkpoint" and target_root_name != "outputs":
                raise CacheBundleIntegrityError("checkpoint member is outside outputs")
            if (
                member.kind in {"immutable", "similarity"}
                and target_root_name != "cache"
            ):
                raise CacheBundleIntegrityError("cache member is outside cache")
            relative = PurePosixPath(*path.parts[1:])
            operation = _InstallOperation(
                member=member,
                staged_path=extracted.joinpath(*path.parts),
                target_root_name=target_root_name,
                relative_path=relative,
            )
            key = (target_root_name, relative.as_posix())
            prior = by_target.get(key)
            if prior is not None and prior.member != operation.member:
                raise CacheBundleConflictError(
                    "verified bundles contradict each other for "
                    f"{target_root_name}/{relative.as_posix()}"
                )
            by_target[key] = prior or operation
    return (
        tuple(by_target[key] for key in sorted(by_target)),
        tuple(scores_by_digest[key] for key in sorted(scores_by_digest)),
    )


def _portable_score_context_and_rows(source: Path):
    """Decode the public portable score envelope behind a private adapter."""
    from trec_rag.rerank_score_cache import (
        PORTABLE_SCORE_CACHE_SCHEMA_VERSION,
        SCORE_CACHE_CONTEXT_VERSION,
        ScoreCacheContext,
    )

    raw = source.read_bytes()
    lines = raw.splitlines()
    if not lines or raw != b"\n".join(lines) + b"\n":
        raise CacheBundleIntegrityError(
            "portable score JSONL is not newline terminated"
        )
    metadata = _require_exact_fields(
        _strict_json(lines[0], "portable score metadata"),
        {"context", "context_sha256", "portable_schema_version", "type"},
        "portable score metadata",
    )
    if _canonical_json(metadata)[:-1] != lines[0]:
        raise CacheBundleIntegrityError("portable score metadata is not canonical")
    context_value = metadata["context"]
    if not isinstance(context_value, dict):
        raise CacheBundleIntegrityError("portable score context is not an object")
    context_payload = dict(context_value)
    if (
        context_payload.pop("context_schema_version", None)
        != SCORE_CACHE_CONTEXT_VERSION
    ):
        raise CacheBundleIntegrityError("portable score context schema changed")
    try:
        context = ScoreCacheContext(**context_payload)
    except (TypeError, ValueError) as exc:
        raise CacheBundleIntegrityError("portable score context is invalid") from exc
    if (
        metadata["type"] != "score-cache-metadata"
        or metadata["portable_schema_version"] != PORTABLE_SCORE_CACHE_SCHEMA_VERSION
        or metadata["context_sha256"] != context.context_sha256
        or source.stem != context.context_sha256
    ):
        raise CacheBundleIntegrityError("portable score context identity changed")
    rows: list[dict[str, object]] = []
    prior_key: str | None = None
    for line in lines[1:]:
        record = _require_exact_fields(
            _strict_json(line, "portable score row"),
            {"cache_key", "query_sha256", "score_hex", "text_sha256", "type"},
            "portable score row",
        )
        if _canonical_json(record)[:-1] != line or record["type"] != "score":
            raise CacheBundleIntegrityError("portable score row is not canonical")
        cache_key = _require_digest(record["cache_key"], "portable score cache key")
        query_sha256 = _require_digest(
            record["query_sha256"], "portable score query digest"
        )
        text_sha256 = _require_digest(
            record["text_sha256"], "portable score text digest"
        )
        score_hex = record["score_hex"]
        if not isinstance(score_hex, str):
            raise CacheBundleIntegrityError(
                "portable score value is not float.hex text"
            )
        try:
            score = float.fromhex(score_hex)
        except ValueError as exc:
            raise CacheBundleIntegrityError("portable score value is invalid") from exc
        if score.hex() != score_hex:
            raise CacheBundleIntegrityError("portable score value is not canonical")
        if prior_key is not None and cache_key <= prior_key:
            raise CacheBundleIntegrityError(
                "portable score rows are not uniquely sorted"
            )
        prior_key = cache_key
        rows.append(
            {
                "cache_key": cache_key,
                "query_sha256": query_sha256,
                "score": score,
                "score_hex": score_hex,
                "text_sha256": text_sha256,
            }
        )
    return context, tuple(rows)


def _preflight_score_operations(
    operations: Sequence[_ScoreOperation],
    *,
    cache_root: Path,
    validation_root: Path,
    conflict_policy: str,
) -> tuple[
    list[dict[str, object]],
    tuple[tuple[dict[str, str], ...], ...],
]:
    """Validate imports in scratch and compare destination scores read-only."""
    from trec_rag.rerank_score_cache import GlobalScoreCache

    conflicts: list[dict[str, object]] = []
    expected_import_conflicts: list[tuple[dict[str, str], ...]] = []
    effective_scores: dict[tuple[str, str], float] = {}
    for operation in operations:
        context, rows = _portable_score_context_and_rows(operation.staged_path)
        scratch = GlobalScoreCache(validation_root / "reranker", context)
        try:
            scratch_receipt = scratch.import_portable_jsonl(
                operation.staged_path,
                conflict_policy=conflict_policy,
            )
        except ValueError as exc:
            if "conflict" in str(exc).casefold():
                raise CacheBundleConflictError(
                    f"portable score bundles conflict: {operation.member.path}"
                ) from exc
            raise CacheBundleIntegrityError(
                f"portable score validation failed: {operation.member.path}"
            ) from exc
        except (OSError, TypeError, RuntimeError) as exc:
            raise CacheBundleIntegrityError(
                f"portable score validation failed: {operation.member.path}"
            ) from exc
        finally:
            scratch.close()
        for row in scratch_receipt.get("conflicts", []):
            conflicts.append(
                {
                    **row,
                    "context_sha256": context.context_sha256,
                    "kind": "score",
                    "resolution": "kept-existing",
                    "scope": "bundle",
                }
            )

        target_path = (cache_root / "reranker").joinpath(*context.path_parts)
        if _existing_target_receipt(target_path, cache_root) is None:
            found: Sequence[float | None] = (None,) * len(rows)
        else:
            target = None
            try:
                target = GlobalScoreCache(
                    cache_root / "reranker", context, read_only=True
                )
                found = target.lookup_many(
                    [
                        {
                            "cache_key": row["cache_key"],
                            "query_sha256": row["query_sha256"],
                            "text_sha256": row["text_sha256"],
                        }
                        for row in rows
                    ]
                )
            except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as exc:
                raise CacheBundleIntegrityError(
                    f"destination score database is invalid: {target_path}"
                ) from exc
            finally:
                if target is not None:
                    target.close()
        operation_conflicts: list[dict[str, str]] = []
        for row, existing in zip(rows, found, strict=True):
            if existing is not None and existing.hex() != row["score_hex"]:
                if conflict_policy == "strict":
                    raise CacheBundleConflictError(
                        "numerical score conflicts with an existing destination value"
                    )
                conflicts.append(
                    {
                        "cache_key": row["cache_key"],
                        "context_sha256": context.context_sha256,
                        "existing_score_hex": existing.hex(),
                        "kind": "score",
                        "query_sha256": row["query_sha256"],
                        "resolution": "kept-existing",
                        "scope": "destination",
                        "source_score_hex": row["score_hex"],
                        "text_sha256": row["text_sha256"],
                    }
                )
            key = (context.context_sha256, str(row["cache_key"]))
            effective = effective_scores.get(key)
            if effective is None and existing is not None:
                effective = existing
                effective_scores[key] = effective
            source_score = float(row["score"])
            if effective is None:
                effective_scores[key] = source_score
                continue
            if effective.hex() == row["score_hex"]:
                continue
            if conflict_policy == "strict":
                raise CacheBundleConflictError(
                    "numerical score conflicts with an existing destination value"
                )
            conflict = {
                "cache_key": str(row["cache_key"]),
                "existing_score_hex": effective.hex(),
                "query_sha256": str(row["query_sha256"]),
                "source_score_hex": str(row["score_hex"]),
                "text_sha256": str(row["text_sha256"]),
            }
            operation_conflicts.append(conflict)
        expected_import_conflicts.append(tuple(operation_conflicts))
    return (
        sorted(
            conflicts,
            key=lambda row: (
                str(row.get("context_sha256")),
                str(row.get("cache_key")),
                str(row.get("scope")),
            ),
        ),
        tuple(expected_import_conflicts),
    )


def _import_score_operation(
    operation: _ScoreOperation,
    *,
    cache_root: Path,
    conflict_policy: str,
) -> dict[str, object]:
    """Import one fully preflighted portable artifact in the cache API transaction."""
    from trec_rag.rerank_score_cache import GlobalScoreCache

    context, _rows = _portable_score_context_and_rows(operation.staged_path)
    cache = GlobalScoreCache(cache_root / "reranker", context)
    try:
        return cache.import_portable_jsonl(
            operation.staged_path,
            conflict_policy=conflict_policy,
        )
    finally:
        cache.close()


def _reconcile_completed_score_operation(
    operation: _ScoreOperation,
    *,
    cache_root: Path,
    conflict_policy: str,
) -> None:
    """Re-import and prove every portable row after a completed-merge retry."""
    from trec_rag.rerank_score_cache import GlobalScoreCache

    context, rows = _portable_score_context_and_rows(operation.staged_path)
    try:
        _import_score_operation(
            operation,
            cache_root=cache_root,
            conflict_policy=conflict_policy,
        )
        cache = GlobalScoreCache(cache_root / "reranker", context, read_only=True)
        try:
            found = cache.lookup_many(
                [
                    {
                        "cache_key": row["cache_key"],
                        "query_sha256": row["query_sha256"],
                        "text_sha256": row["text_sha256"],
                    }
                    for row in rows
                ]
            )
        finally:
            cache.close()
    except CacheBundleConflictError:
        raise
    except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as exc:
        raise CacheBundleIntegrityError(
            f"completed merge score state is invalid: {operation.member.path}"
        ) from exc
    if any(value is None for value in found):
        raise CacheBundleIntegrityError(
            f"completed merge is missing portable scores: {operation.member.path}"
        )
    if conflict_policy == "strict" and any(
        value is None or value.hex() != row["score_hex"]
        for row, value in zip(rows, found, strict=True)
    ):
        raise CacheBundleIntegrityError(
            f"completed merge portable scores differ: {operation.member.path}"
        )


def _parse_merge_receipt(path: Path, body: bytes) -> MergeReceipt:
    raw = _require_exact_fields(
        _strict_json(body, "merge completion receipt"),
        {
            "bundle_count",
            "conflicts_sha256",
            "identical_count",
            "installed_count",
            "kept_conflict_count",
            "merge_id",
            "operation_count",
            "prepare_sha256",
            "schema_version",
            "score_import_count",
        },
        "merge completion receipt",
    )
    if _canonical_json(raw) != body:
        raise CacheBundleIntegrityError("merge completion receipt is not canonical")
    if raw["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise CacheBundleIntegrityError("merge completion schema changed")
    return MergeReceipt(
        merge_id=_require_digest(raw["merge_id"], "merge_id"),
        completion_path=path,
        bundle_count=_require_nonnegative_int(raw["bundle_count"], "bundle_count"),
        operation_count=_require_nonnegative_int(
            raw["operation_count"], "operation_count"
        ),
        installed_count=_require_nonnegative_int(
            raw["installed_count"], "installed_count"
        ),
        identical_count=_require_nonnegative_int(
            raw["identical_count"], "identical_count"
        ),
        kept_conflict_count=_require_nonnegative_int(
            raw["kept_conflict_count"], "kept_conflict_count"
        ),
        score_import_count=_require_nonnegative_int(
            raw["score_import_count"], "score_import_count"
        ),
    )


def _read_regular_merge_state(
    path: Path,
    label: str,
    *,
    expected_size: int | None = None,
) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CacheBundleIntegrityError(f"completed merge {label} is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CacheBundleIntegrityError(
            f"completed merge {label} is not a regular file"
        )
    if expected_size is not None and metadata.st_size != expected_size:
        raise CacheBundleIntegrityError(
            f"completed merge {label} size differs from current content"
        )
    if expected_size is None and metadata.st_size > MAX_MANIFEST_BYTES:
        raise CacheBundleIntegrityError(f"completed merge {label} size limit exceeded")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CacheBundleIntegrityError(
            f"completed merge {label} could not be read"
        ) from exc


def _authenticate_completed_merge(
    *,
    complete_path: Path,
    prepare_path: Path,
    conflicts_path: Path,
    expected_prepare: bytes,
    expected_conflicts: bytes,
    merge_id: str,
    bundle_count: int,
    install_operation_count: int,
    operation_count: int,
    kept_operation_count: int,
    conflict_count: int,
    score_operation_count: int,
) -> MergeReceipt:
    """Authenticate a completion receipt from current inputs and postconditions."""
    actual_prepare = _read_regular_merge_state(
        prepare_path,
        "prepare journal",
        expected_size=len(expected_prepare),
    )
    if actual_prepare != expected_prepare:
        raise CacheBundleIntegrityError(
            "completed merge prepare journal differs from current inputs"
        )
    actual_conflicts = _read_regular_merge_state(
        conflicts_path,
        "conflict audit",
        expected_size=len(expected_conflicts),
    )
    if actual_conflicts != expected_conflicts:
        raise CacheBundleIntegrityError(
            "completed merge conflict audit differs from current destinations"
        )
    complete_bytes = _read_regular_merge_state(complete_path, "completion receipt")
    receipt = _parse_merge_receipt(complete_path, complete_bytes)
    raw = _strict_json(complete_bytes, "merge completion receipt")
    if not isinstance(raw, dict):
        raise CacheBundleIntegrityError("merge completion receipt is not an object")
    expected_install_outcomes = install_operation_count - kept_operation_count
    if (
        receipt.merge_id != merge_id
        or receipt.bundle_count != bundle_count
        or receipt.operation_count != operation_count
        or receipt.installed_count + receipt.identical_count
        != expected_install_outcomes
        or receipt.kept_conflict_count != conflict_count
        or receipt.score_import_count != score_operation_count
        or raw["prepare_sha256"] != hashlib.sha256(expected_prepare).hexdigest()
        or raw["conflicts_sha256"] != hashlib.sha256(expected_conflicts).hexdigest()
    ):
        raise CacheBundleIntegrityError(
            "completed merge receipt differs from authenticated merge content"
        )
    return receipt


def merge_bundles(
    *,
    cache_root: str | Path,
    outputs_root: str | Path,
    bundle_dirs: Sequence[str | Path],
    score_conflicts: str = "strict",
    publication_hook: Callable[[str], None] | None = None,
) -> MergeReceipt:
    """Stage and converge verified bundle files under durable merge state."""
    if score_conflicts not in {"strict", "keep-existing"}:
        raise ValueError("score_conflicts must be strict or keep-existing")
    if not bundle_dirs:
        raise ValueError("at least one bundle directory is required")
    cache_destination = _validate_destination_root(cache_root, "cache-root")
    outputs_destination = _validate_destination_root(outputs_root, "outputs-root")

    # No destination path is created until every hostile archive has completed
    # a bounded, extraction-free validation pass.
    unique: dict[str, VerifiedBundle] = {}
    for raw_bundle in bundle_dirs:
        verified = verify_bundle(raw_bundle)
        prior = unique.get(verified.archive_sha256)
        if prior is not None and prior != verified:
            raise CacheBundleIntegrityError(
                "duplicate archive digest has conflicting identity"
            )
        unique[verified.archive_sha256] = verified
    bundles = tuple(
        sorted(unique.values(), key=lambda item: (item.topic_id, item.archive_sha256))
    )
    merge_id = _merge_id(
        bundles,
        cache_root=cache_destination,
        outputs_root=outputs_destination,
        score_conflicts=score_conflicts,
    )

    if cache_destination.exists():
        unrelated = tuple(
            path
            for path in incomplete_merge_journals(cache_destination)
            if path.parent.name != merge_id
        )
        if unrelated:
            raise CacheBundleIntegrityError(
                "cache root has another incomplete bundle merge: "
                + ", ".join(str(path) for path in unrelated)
            )

    with tempfile.TemporaryDirectory(prefix="trec-rag-cache-merge-") as temporary:
        staging_root = Path(temporary)
        operations, score_operations = _prepare_operations(bundles, staging_root)
        conflicts, expected_score_import_conflicts = _preflight_score_operations(
            score_operations,
            cache_root=cache_destination,
            validation_root=staging_root / "score-validation",
            conflict_policy=score_conflicts,
        )
        missing: list[_InstallOperation] = []
        identical = 0
        kept: set[tuple[str, str]] = set()
        kept_conflicts: dict[tuple[str, str], dict[str, object]] = {}
        for operation in operations:
            target = operation.target(cache_destination, outputs_destination)
            target_root = (
                cache_destination
                if operation.target_root_name == "cache"
                else outputs_destination
            )
            existing = _existing_target_receipt(target, target_root)
            expected = (operation.member.size, operation.member.sha256)
            if existing is None:
                missing.append(operation)
            elif existing == expected:
                identical += 1
            elif (
                score_conflicts == "keep-existing"
                and operation.member.kind == "similarity"
            ):
                _validate_existing_similarity_conflict(
                    operation,
                    cache_root=cache_destination,
                    target=target,
                )
                key = (operation.target_root_name, operation.relative_path.as_posix())
                kept.add(key)
                conflict = {
                    "existing_sha256": existing[1],
                    "existing_size": existing[0],
                    "kind": "similarity",
                    "relative_path": operation.relative_path.as_posix(),
                    "resolution": "kept-existing",
                    "source_sha256": operation.member.sha256,
                    "source_size": operation.member.size,
                    "target_root": operation.target_root_name,
                }
                kept_conflicts[key] = conflict
                conflicts.append(conflict)
            else:
                raise CacheBundleConflictError(f"immutable merge conflict at {target}")

        prepare_payload = {
            "bundles": [
                {
                    "archive_sha256": bundle.archive_sha256,
                    "manifest_sha256": bundle.manifest_sha256,
                    "topic_id": bundle.topic_id,
                }
                for bundle in bundles
            ],
            "cache_root": str(cache_destination),
            "merge_id": merge_id,
            "operations": [
                *[_operation_payload(operation) for operation in operations],
                *[
                    _score_operation_payload(operation)
                    for operation in score_operations
                ],
            ],
            "outputs_root": str(outputs_destination),
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "score_conflicts": score_conflicts,
        }
        prepare_bytes = _canonical_json(prepare_payload)
        prepare_sha256 = hashlib.sha256(prepare_bytes).hexdigest()
        state_root = cache_destination / MERGE_STATE_DIRECTORY / merge_id
        prepare_path = state_root / MERGE_PREPARE_NAME
        conflicts_path = state_root / MERGE_CONFLICTS_NAME
        complete_path = state_root / MERGE_COMPLETE_NAME
        conflicts_bytes = _canonical_json(
            {
                "conflicts": conflicts,
                "merge_id": merge_id,
                "schema_version": BUNDLE_SCHEMA_VERSION,
            }
        )
        conflicts_sha256 = hashlib.sha256(conflicts_bytes).hexdigest()

        try:
            complete_metadata = complete_path.lstat()
        except FileNotFoundError:
            completed = False
        except OSError as exc:
            raise CacheBundleIntegrityError(
                "merge completion receipt could not be inspected"
            ) from exc
        else:
            if stat.S_ISLNK(complete_metadata.st_mode) or not stat.S_ISREG(
                complete_metadata.st_mode
            ):
                raise CacheBundleIntegrityError(
                    "merge completion receipt is not a regular file"
                )
            completed = True

        if completed:
            if missing:
                raise CacheBundleIntegrityError(
                    "completed merge receipt has a missing immutable destination"
                )
            prior = _authenticate_completed_merge(
                complete_path=complete_path,
                prepare_path=prepare_path,
                conflicts_path=conflicts_path,
                expected_prepare=prepare_bytes,
                expected_conflicts=conflicts_bytes,
                merge_id=merge_id,
                bundle_count=len(bundles),
                install_operation_count=len(operations),
                operation_count=len(operations) + len(score_operations),
                kept_operation_count=len(kept),
                conflict_count=len(conflicts),
                score_operation_count=len(score_operations),
            )
            for operation in score_operations:
                _reconcile_completed_score_operation(
                    operation,
                    cache_root=cache_destination,
                    conflict_policy=score_conflicts,
                )
            return prior

        _secure_mkdir(cache_destination)
        _secure_mkdir(outputs_destination)
        _write_durable_create_only(prepare_path, prepare_bytes)
        if publication_hook is not None:
            publication_hook("prepare")

        score_import_count = 0
        for operation, expected_import_conflicts in zip(
            score_operations,
            expected_score_import_conflicts,
            strict=True,
        ):
            try:
                import_receipt = _import_score_operation(
                    operation,
                    cache_root=cache_destination,
                    conflict_policy=score_conflicts,
                )
            except ValueError as exc:
                if "conflict" in str(exc).casefold():
                    raise CacheBundleConflictError(
                        "transactional score import conflicts changed after preflight"
                    ) from exc
                raise CacheBundleIntegrityError(
                    "transactional score import failed after preflight"
                ) from exc
            if import_receipt.get("conflict_count") != len(
                expected_import_conflicts
            ) or import_receipt.get("conflicts") != list(expected_import_conflicts):
                raise CacheBundleConflictError(
                    "transactional score import conflicts changed after preflight"
                )
            score_import_count += 1
            if publication_hook is not None:
                publication_hook(f"score-imported:{operation.member.sha256}")

        installed = 0
        for operation in missing:
            key = (operation.target_root_name, operation.relative_path.as_posix())
            if key in kept:
                continue
            target = operation.target(cache_destination, outputs_destination)
            created = _install_create_only(operation, target)
            installed += int(created)
            identical += int(not created)
            if publication_hook is not None:
                publication_hook(
                    f"installed:{operation.target_root_name}/{operation.relative_path.as_posix()}"
                )

        for operation in operations:
            key = (operation.target_root_name, operation.relative_path.as_posix())
            target = operation.target(cache_destination, outputs_destination)
            target_root = (
                cache_destination
                if operation.target_root_name == "cache"
                else outputs_destination
            )
            existing = _existing_target_receipt(target, target_root)
            if key not in kept:
                if existing != (operation.member.size, operation.member.sha256):
                    raise CacheBundleConflictError(
                        f"immutable merge target changed after preflight: {target}"
                    )
                continue
            if existing is None:
                raise CacheBundleConflictError(
                    f"kept similarity target changed after preflight: {target}"
                )
            _validate_existing_similarity_conflict(
                operation,
                cache_root=cache_destination,
                target=target,
            )
            current_conflict = {
                "existing_sha256": existing[1],
                "existing_size": existing[0],
                "kind": "similarity",
                "relative_path": operation.relative_path.as_posix(),
                "resolution": "kept-existing",
                "source_sha256": operation.member.sha256,
                "source_size": operation.member.size,
                "target_root": operation.target_root_name,
            }
            if current_conflict != kept_conflicts[key]:
                raise CacheBundleConflictError(
                    f"kept similarity conflict changed after preflight: {target}"
                )

        _write_durable_create_only(conflicts_path, conflicts_bytes)
        complete_bytes = _canonical_json(
            {
                "bundle_count": len(bundles),
                "conflicts_sha256": conflicts_sha256,
                "identical_count": identical,
                "installed_count": installed,
                "kept_conflict_count": len(conflicts),
                "merge_id": merge_id,
                "operation_count": len(operations) + len(score_operations),
                "prepare_sha256": prepare_sha256,
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "score_import_count": score_import_count,
            }
        )
        try:
            _write_durable_create_only(complete_path, complete_bytes)
        except CacheBundleConflictError as collision:
            try:
                return _authenticate_completed_merge(
                    complete_path=complete_path,
                    prepare_path=prepare_path,
                    conflicts_path=conflicts_path,
                    expected_prepare=prepare_bytes,
                    expected_conflicts=conflicts_bytes,
                    merge_id=merge_id,
                    bundle_count=len(bundles),
                    install_operation_count=len(operations),
                    operation_count=len(operations) + len(score_operations),
                    kept_operation_count=len(kept),
                    conflict_count=len(conflicts),
                    score_operation_count=len(score_operations),
                )
            except CacheBundleIntegrityError as authentication_error:
                raise collision from authentication_error
        if publication_hook is not None:
            publication_hook("complete")
        return _authenticate_completed_merge(
            complete_path=complete_path,
            prepare_path=prepare_path,
            conflicts_path=conflicts_path,
            expected_prepare=prepare_bytes,
            expected_conflicts=conflicts_bytes,
            merge_id=merge_id,
            bundle_count=len(bundles),
            install_operation_count=len(operations),
            operation_count=len(operations) + len(score_operations),
            kept_operation_count=len(kept),
            conflict_count=len(conflicts),
            score_operation_count=len(score_operations),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack, verify, and merge deterministic competition cache shards."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--config", type=Path, required=True)
    pack.add_argument("--topic", required=True)
    pack.add_argument("--destination", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("bundle_dir", type=Path)
    merge = commands.add_parser("merge")
    merge.add_argument("--cache-root", type=Path, required=True)
    merge.add_argument("--outputs-root", type=Path, required=True)
    merge.add_argument(
        "--score-conflicts",
        choices=("strict", "keep-existing"),
        default="strict",
    )
    merge.add_argument("bundle_dirs", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "pack":
        receipt = pack_bundle(args.config, args.topic, args.destination)
        print(f"archive_sha256={receipt.archive_sha256}")
        return 0
    if args.command == "verify":
        verified = verify_bundle(args.bundle_dir)
        print(f"archive_sha256={verified.archive_sha256}")
        return 0
    receipt = merge_bundles(
        cache_root=args.cache_root,
        outputs_root=args.outputs_root,
        bundle_dirs=args.bundle_dirs,
        score_conflicts=args.score_conflicts,
    )
    print(f"merge_id={receipt.merge_id}")
    print(f"completion={receipt.completion_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
