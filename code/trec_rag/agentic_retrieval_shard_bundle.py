"""Authenticated, projection-only bundles for completed agentic topics.

The archive deliberately contains only the three public projections and the
manifest-last topic seal.  It is not a TopicRecords or retrieval-cache replay
format.  The verifier treats the tar stream and all metadata as hostile input;
the importer verifies the complete archive before taking its process lock and
publishes files create-only into a run-specific staging tree.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
import unicodedata
import uuid
from typing import Iterator, Mapping, Sequence

import zstandard

from .agentic_run_state import (
    TOPIC_SEAL_FILENAME,
    TOPIC_SEAL_SCHEMA,
    AgenticRunStateError,
    load_run_plan,
    load_topic_seal,
)
from .generation_handoff import deserialize_generation_topic


BUNDLE_SCHEMA_VERSION = "agentic-retrieval-topic-bundle-v1"
# The completion marker uses the bundle schema version, matching the existing
# cache-bundle convention: a marker is a receipt for one specific schema, not
# a second independently evolving format.
MARKER_SCHEMA_VERSION = BUNDLE_SCHEMA_VERSION
BUNDLE_MANIFEST_NAME = "bundle-manifest.json"
BUNDLE_COMPLETE_NAME = "bundle-complete.json"
TOPIC_SEAL_MEMBER = TOPIC_SEAL_FILENAME
TOPIC_MEMBER_NAMES = (
    "generation_topic.json",
    "retrieval_topic.json",
    "topic_records_receipt.json",
    TOPIC_SEAL_MEMBER,
)
_TOPIC_MEMBER_SET = frozenset(TOPIC_MEMBER_NAMES)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

MAX_COMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_MARKER_BYTES = 64 * 1024
MAX_MEMBERS = 16
_COPY_CHUNK = 1024 * 1024
_RECEIPT_ROW_COUNTS = frozenset(
    {
        "candidate",
        "candidate_passage_link",
        "candidate_span",
        "document_binding",
        "passage",
        "query_facet",
        "query_identity",
        "query_passage",
        "researcher_evidence",
        "researcher_facet_update",
        "researcher_handoff",
        "retrieval_candidate",
        "stage_seal",
        "subnarrative_identity",
        "topic_completion",
        "topic_identity",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "database_sha256",
        "database_bytes",
        "semantic_sha256",
        "topic_id",
        "run_id",
        "document_sha256s",
        "row_counts",
        "schema_version",
        "manifest_sha256",
        "manifest_bytes",
    }
)
_JOURNAL_DIRECTORY = ".agentic-bundle-imports"
_LOCK_NAME = ".agentic-bundle-import.lock"


class AgenticShardBundleError(ValueError):
    """The bundle, marker, or destination is invalid."""


class AgenticShardBundleIntegrityError(AgenticShardBundleError):
    """A hostile or corrupted bundle failed closed."""


class AgenticShardBundleConflictError(AgenticShardBundleError):
    """Create-only publication found contradictory immutable bytes."""


@dataclass(frozen=True)
class TopicBundleMember:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class VerifiedTopicBundle:
    topic_id: str
    run_plan_sha256: str
    topic_seal_sha256: str
    archive_sha256: str
    archive_size: int
    manifest_sha256: str
    members: tuple[TopicBundleMember, ...]
    bundle_schema: str = BUNDLE_SCHEMA_VERSION

    @property
    def archive_bytes(self) -> int:
        return self.archive_size


def _canonical(value: object) -> bytes:
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
    except (TypeError, ValueError) as exc:
        raise AgenticShardBundleIntegrityError("metadata is not canonical JSON") from exc


def _strict_json(body: bytes, label: str) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AgenticShardBundleIntegrityError(f"{label} has duplicate keys")
            result[key] = value
        return result

    def reject(value: str) -> object:
        raise AgenticShardBundleIntegrityError(f"{label} has non-finite JSON")

    try:
        value = json.loads(
            body.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject
        )
    except AgenticShardBundleIntegrityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgenticShardBundleIntegrityError(f"{label} is not strict JSON") from exc
    return value


def _exact_mapping(value: object, fields: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise AgenticShardBundleIntegrityError(f"{label} must be an object")
    if set(value) != fields:
        raise AgenticShardBundleIntegrityError(f"{label} fields changed")
    return value


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AgenticShardBundleIntegrityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise AgenticShardBundleIntegrityError(f"{label} must be a safe identifier")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgenticShardBundleIntegrityError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    value = _nonnegative_int(value, label)
    if value == 0:
        raise AgenticShardBundleIntegrityError(f"{label} must be positive")
    return value


def _regular(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise AgenticShardBundleIntegrityError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise AgenticShardBundleIntegrityError(f"{label} must be a regular file")


def _read_regular(path: Path, label: str) -> bytes:
    _regular(path, label)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise AgenticShardBundleIntegrityError(f"unable to read {label}") from exc


def _ensure_directory(path: Path, label: str, *, create: bool = False, private: bool = True) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        info = path.lstat()
    except OSError as exc:
        raise AgenticShardBundleIntegrityError(f"{label} is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise AgenticShardBundleIntegrityError(f"{label} must be a directory")
    if private:
        os.chmod(path, 0o700)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_identical(path: Path, body: bytes, label: str) -> None:
    _ensure_directory(path.parent, f"{label} parent", create=True, private=False)
    if path.exists() or path.is_symlink():
        existing = _read_regular(path, label)
        if existing != body:
            raise AgenticShardBundleConflictError(f"{label} publication conflict")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if _read_regular(path, label) != body:
                raise AgenticShardBundleConflictError(f"{label} publication conflict")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_member_path(name: object, seen: dict[str, str]) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise AgenticShardBundleIntegrityError("archive member path is unsafe")
    if name.startswith("/"):
        raise AgenticShardBundleIntegrityError("archive member path is absolute")
    path = PurePosixPath(name)
    if path == PurePosixPath(".") or any(part in {"", ".", ".."} for part in path.parts):
        raise AgenticShardBundleIntegrityError("archive member path is unsafe")
    normalized = unicodedata.normalize("NFC", name)
    key = normalized.casefold()
    prior = seen.get(key)
    if prior is not None:
        raise AgenticShardBundleIntegrityError(f"duplicate or colliding archive member: {name}")
    seen[key] = name
    return name


def _validate_prefixes(paths: Sequence[str]) -> None:
    canonical = {PurePosixPath(unicodedata.normalize("NFC", p).casefold()) for p in paths}
    for path in canonical:
        if any(parent in canonical for parent in path.parents):
            raise AgenticShardBundleIntegrityError("archive member paths have a prefix collision")


def _member_payload(path: str, body: bytes) -> dict[str, object]:
    return {"mode": 0o600, "path": path, "sha256": _digest(body), "size": len(body)}


def _manifest_payload(*, plan_sha256: str, topic_id: str, narrative_sha256: str,
                      topic_seal_sha256: str, members: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "members": list(members),
        "run_plan_sha256": plan_sha256,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "topic_id": topic_id,
        "topic_seal_sha256": topic_seal_sha256,
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


def _write_archive(path: Path, manifest: bytes, members: Sequence[tuple[str, bytes]]) -> None:
    compressor = zstandard.ZstdCompressor(level=19, threads=0, write_checksum=True, write_content_size=False)
    with path.open("xb") as raw:
        os.fchmod(raw.fileno(), 0o600)
        with compressor.stream_writer(raw, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                archive.addfile(_tar_info(BUNDLE_MANIFEST_NAME, len(manifest)), io.BytesIO(manifest))
                for name, body in members:
                    archive.addfile(_tar_info(name, len(body)), io.BytesIO(body))
        raw.flush()
        os.fsync(raw.fileno())


def _file_receipt(path: Path) -> tuple[int, str]:
    _regular(path, "archive")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_COPY_CHUNK):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _parse_marker(path: Path) -> Mapping[str, object]:
    body = _read_regular(path, "bundle completion marker")
    if len(body) > MAX_MARKER_BYTES:
        raise AgenticShardBundleIntegrityError("bundle completion marker is too large")
    value = _strict_json(body, "bundle completion marker")
    row = _exact_mapping(
        value,
        {
            "archive_sha256",
            "archive_size",
            "run_plan_sha256",
            "schema_version",
            "topic_id",
            "topic_seal_sha256",
        },
        "bundle completion marker",
    )
    if _canonical(row) != body:
        raise AgenticShardBundleIntegrityError("bundle completion marker is not canonical")
    if row["schema_version"] != MARKER_SCHEMA_VERSION:
        raise AgenticShardBundleIntegrityError("unsupported bundle completion schema")
    _require_digest(row["archive_sha256"], "archive_sha256")
    _positive_int(row["archive_size"], "archive_size")
    _require_digest(row["run_plan_sha256"], "run_plan_sha256")
    _require_digest(row["topic_seal_sha256"], "topic_seal_sha256")
    _require_safe_id(row["topic_id"], "topic_id")
    return row


def _parse_manifest(body: bytes) -> tuple[str, str, str, tuple[TopicBundleMember, ...]]:
    if len(body) > MAX_MANIFEST_BYTES:
        raise AgenticShardBundleIntegrityError("bundle manifest is too large")
    value = _strict_json(body, "bundle manifest")
    row = _exact_mapping(
        value,
        {"members", "run_plan_sha256", "schema_version", "topic_id", "topic_seal_sha256"},
        "bundle manifest",
    )
    if _canonical(row) != body:
        raise AgenticShardBundleIntegrityError("bundle manifest is not canonical")
    if row["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise AgenticShardBundleIntegrityError("unsupported bundle manifest schema")
    plan_sha256 = _require_digest(row["run_plan_sha256"], "run_plan_sha256")
    seal_sha256 = _require_digest(row["topic_seal_sha256"], "topic_seal_sha256")
    topic_id = _require_safe_id(row["topic_id"], "topic_id")
    raw_members = row["members"]
    if not isinstance(raw_members, list) or len(raw_members) != len(TOPIC_MEMBER_NAMES):
        raise AgenticShardBundleIntegrityError("bundle member allowlist changed")
    seen: dict[str, str] = {unicodedata.normalize("NFC", BUNDLE_MANIFEST_NAME).casefold(): BUNDLE_MANIFEST_NAME}
    members: list[TopicBundleMember] = []
    for raw in raw_members:
        item = _exact_mapping(raw, {"mode", "path", "sha256", "size"}, "bundle member")
        path = _validate_member_path(item["path"], seen)
        if path not in _TOPIC_MEMBER_SET or item["mode"] != 0o600:
            raise AgenticShardBundleIntegrityError("bundle contains a forbidden member")
        size = _positive_int(item["size"], "bundle member size")
        if size > MAX_MEMBER_BYTES:
            raise AgenticShardBundleIntegrityError("bundle member is too large")
        members.append(TopicBundleMember(path, size, _require_digest(item["sha256"], "bundle member sha256")))
    if tuple(member.path for member in members) != tuple(sorted(TOPIC_MEMBER_NAMES)):
        raise AgenticShardBundleIntegrityError("bundle members are not canonical and sorted")
    _validate_prefixes([BUNDLE_MANIFEST_NAME, *(member.path for member in members)])
    return topic_id, plan_sha256, seal_sha256, tuple(members)


def _decompress_archive(path: Path) -> bytes:
    size = path.stat().st_size
    if size > MAX_COMPRESSED_BYTES:
        raise AgenticShardBundleIntegrityError("compressed bundle size limit exceeded")
    source = path.open("rb")
    try:
        decoder = zstandard.ZstdDecompressor(max_window_size=128 * 1024 * 1024).decompressobj()
        chunks: list[bytes] = []
        total = 0
        try:
            while raw := source.read(_COPY_CHUNK):
                if decoder.eof:
                    raise AgenticShardBundleIntegrityError("bundle has trailing compressed bytes")
                output = decoder.decompress(raw)
                total += len(output)
                if total > MAX_DECOMPRESSED_BYTES:
                    raise AgenticShardBundleIntegrityError("decompressed bundle size limit exceeded")
                chunks.append(output)
                if decoder.eof:
                    # ``unused_data`` is the suffix in this input chunk; bytes
                    # still in the source are a second frame or trailing data.
                    if decoder.unused_data or source.read(1):
                        raise AgenticShardBundleIntegrityError("bundle has trailing compressed bytes")
                    break
            if not decoder.eof:
                raise AgenticShardBundleIntegrityError("bundle compressed stream is truncated")
        except zstandard.ZstdError as exc:
            raise AgenticShardBundleIntegrityError("bundle compressed stream is invalid") from exc
    finally:
        source.close()
    return b"".join(chunks)


def _validate_seal_payloads(
    payloads: Mapping[str, bytes], *, topic_id: str, plan_sha256: str, seal_sha256: str
) -> None:
    seal_body = payloads[TOPIC_SEAL_MEMBER]
    seal = _strict_json(seal_body, "topic seal")
    seal_row = _exact_mapping(
        seal,
        {
            "artifacts",
            "attempt_number",
            "full_text_record_sha256",
            "retrieval_topic_sha256",
            "run_plan_sha256",
            "schema_version",
            "seal_sha256",
            "status",
            "stopping_reason",
            "synthesis_outcome",
            "topic",
        },
        "topic seal",
    )
    if _canonical(seal_row) != seal_body:
        raise AgenticShardBundleIntegrityError("topic seal is not canonical")
    if seal_row["schema_version"] != TOPIC_SEAL_SCHEMA or seal_row["run_plan_sha256"] != plan_sha256:
        raise AgenticShardBundleIntegrityError("topic seal belongs to another run plan")
    if _require_digest(seal_row["seal_sha256"], "topic seal digest") != seal_sha256:
        raise AgenticShardBundleIntegrityError("topic seal digest differs from manifest")
    without_digest = {key: value for key, value in seal_row.items() if key != "seal_sha256"}
    # ``agentic_run_state`` hashes the canonical object without its JSONL
    # newline, while the published seal itself is a canonical JSONL record.
    if _digest(_canonical(without_digest).rstrip(b"\n")) != seal_sha256:
        raise AgenticShardBundleIntegrityError("topic seal digest does not match contents")
    topic = _exact_mapping(seal_row["topic"], {"narrative_sha256", "topic_id"}, "sealed topic")
    if topic["topic_id"] != topic_id:
        raise AgenticShardBundleIntegrityError("topic seal topic identity differs")
    _require_digest(topic["narrative_sha256"], "sealed narrative")
    if seal_row["status"] != "complete" or not isinstance(seal_row["stopping_reason"], str) or not seal_row["stopping_reason"]:
        raise AgenticShardBundleIntegrityError("topic seal is not successful")
    if seal_row["synthesis_outcome"] not in {"coordinator_selected", "deterministic_grounded_recovery"}:
        raise AgenticShardBundleIntegrityError("topic seal synthesis outcome is invalid")
    _positive_int(seal_row["attempt_number"], "topic attempt number")

    artifacts = seal_row["artifacts"]
    if not isinstance(artifacts, list) or [item.get("name") for item in artifacts if isinstance(item, Mapping)] != [
        "retrieval_topic.json", "generation_topic.json", "topic_records_receipt.json"
    ]:
        raise AgenticShardBundleIntegrityError("topic seal artifacts changed")
    for artifact in artifacts:
        item = _exact_mapping(artifact, {"bytes", "name", "sha256"}, "sealed artifact")
        name = item["name"]
        if name not in payloads or name == TOPIC_SEAL_MEMBER:
            raise AgenticShardBundleIntegrityError("topic seal artifact is not bundled")
        if _positive_int(item["bytes"], "sealed artifact bytes") != len(payloads[name]):
            raise AgenticShardBundleIntegrityError("sealed artifact size differs")
        if _require_digest(item["sha256"], "sealed artifact sha256") != _digest(payloads[name]):
            raise AgenticShardBundleIntegrityError("sealed artifact digest differs")

    retrieval_body = payloads["retrieval_topic.json"]
    retrieval = _strict_json(retrieval_body, "retrieval topic")
    if _canonical(retrieval) != retrieval_body or not isinstance(retrieval, Mapping) or retrieval.get("topic_id") != topic_id:
        raise AgenticShardBundleIntegrityError("retrieval topic projection is invalid")
    retrieval_digest = _require_digest(seal_row["retrieval_topic_sha256"], "retrieval topic sha256")
    if _digest(retrieval_body) != retrieval_digest:
        raise AgenticShardBundleIntegrityError("retrieval topic digest differs")
    full_text = retrieval.get("full_text_record")
    if not isinstance(full_text, Mapping):
        raise AgenticShardBundleIntegrityError("retrieval topic lacks full-text record")
    if _digest(_canonical(dict(full_text)).rstrip(b"\n")) != _require_digest(seal_row["full_text_record_sha256"], "full-text record sha256"):
        raise AgenticShardBundleIntegrityError("full-text record digest differs")
    candidates = full_text.get("candidates")
    if not isinstance(candidates, list):
        raise AgenticShardBundleIntegrityError("retrieval candidates are invalid")
    projected_documents: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise AgenticShardBundleIntegrityError("retrieval candidate is invalid")
        projected_documents.append(_require_digest(candidate.get("text_sha256"), "projected document"))

    generation_body = payloads["generation_topic.json"]
    try:
        generation = deserialize_generation_topic(generation_body)
    except Exception as exc:
        raise AgenticShardBundleIntegrityError("generation topic projection is invalid") from exc
    if (
        generation.topic_id != topic_id
        or generation.narrative_sha256 != topic["narrative_sha256"]
        or generation.source_receipts.retrieval_topic_sha256 != retrieval_digest
    ):
        raise AgenticShardBundleIntegrityError("generation topic identity differs")

    receipt_body = payloads["topic_records_receipt.json"]
    receipt = _strict_json(receipt_body, "topic records receipt")
    receipt_row = _exact_mapping(receipt, set(_RECEIPT_FIELDS), "topic records receipt")
    if _canonical(receipt_row) != receipt_body:
        raise AgenticShardBundleIntegrityError("topic records receipt is not canonical")
    if receipt_row["topic_id"] != topic_id:
        raise AgenticShardBundleIntegrityError("topic records receipt topic differs")
    _require_safe_id(receipt_row["run_id"], "topic records receipt run id")
    _require_digest(receipt_row["database_sha256"], "topic records database")
    _require_digest(receipt_row["semantic_sha256"], "topic records semantic digest")
    _require_digest(receipt_row["manifest_sha256"], "topic records manifest digest")
    _nonnegative_int(receipt_row["database_bytes"], "topic records database bytes")
    _nonnegative_int(receipt_row["manifest_bytes"], "topic records manifest bytes")
    docs = receipt_row["document_sha256s"]
    if not isinstance(docs, list) or any(not isinstance(item, str) or _SHA256.fullmatch(item) is None for item in docs) or docs != sorted(docs) or len(docs) != len(set(docs)):
        raise AgenticShardBundleIntegrityError("topic records document closure is invalid")
    if not set(projected_documents) <= set(docs):
        raise AgenticShardBundleIntegrityError("topic records receipt omits projected evidence")
    row_counts = receipt_row["row_counts"]
    if not isinstance(row_counts, Mapping) or set(row_counts) != set(_RECEIPT_ROW_COUNTS) or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in row_counts.values()):
        raise AgenticShardBundleIntegrityError("topic records row counts are invalid")


def _verify_archive(archive_path: Path, marker: Mapping[str, object], expected_plan: str) -> VerifiedTopicBundle:
    compressed_size, archive_sha256 = _file_receipt(archive_path)
    if compressed_size != marker["archive_size"] or archive_sha256 != marker["archive_sha256"]:
        raise AgenticShardBundleIntegrityError("bundle archive digest or size mismatch")
    if marker["run_plan_sha256"] != expected_plan:
        raise AgenticShardBundleIntegrityError("bundle belongs to another run plan")
    tar_bytes = _decompress_archive(archive_path)
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
            infos = archive.getmembers()
            if len(infos) != len(TOPIC_MEMBER_NAMES) + 1:
                raise AgenticShardBundleIntegrityError("bundle member count changed")
            seen: dict[str, str] = {}
            payloads: dict[str, bytes] = {}
            for index, info in enumerate(infos):
                name = _validate_member_path(info.name, seen)
                if info.type != tarfile.REGTYPE or info.issym() or info.islnk() or info.pax_headers:
                    raise AgenticShardBundleIntegrityError("archive member is not a canonical regular file")
                if info.size < 0 or info.size > MAX_MEMBER_BYTES:
                    raise AgenticShardBundleIntegrityError("archive member size limit exceeded")
                if info.mode != 0o600 or info.mtime != 0 or info.uid != 0 or info.gid != 0 or info.uname not in {"", None} or info.gname not in {"", None}:
                    raise AgenticShardBundleIntegrityError("archive member metadata is not canonical")
                stream = archive.extractfile(info)
                if stream is None:
                    raise AgenticShardBundleIntegrityError("archive member body is unavailable")
                body = stream.read(info.size + 1)
                if len(body) != info.size:
                    raise AgenticShardBundleIntegrityError("archive member stream is truncated")
                if index == 0:
                    if name != BUNDLE_MANIFEST_NAME:
                        raise AgenticShardBundleIntegrityError("bundle manifest must be first")
                    topic_id, plan_sha256, seal_sha256, members = _parse_manifest(body)
                    if plan_sha256 != expected_plan or marker["topic_id"] != topic_id or marker["topic_seal_sha256"] != seal_sha256:
                        raise AgenticShardBundleIntegrityError("bundle identity differs between marker and manifest")
                    manifest_sha256 = _digest(body)
                    declared = members
                    continue
                if index == 1 and declared is None:
                    raise AgenticShardBundleIntegrityError("bundle has no manifest")
                expected = declared[index - 1]
                if name != expected.path or info.size != expected.size or _digest(body) != expected.sha256:
                    raise AgenticShardBundleIntegrityError(f"bundle member digest or size mismatch: {name}")
                payloads[name] = body
            if declared is None or tuple(payloads) != tuple(member.path for member in declared):
                raise AgenticShardBundleIntegrityError("bundle members are missing or reordered")
            _validate_seal_payloads(payloads, topic_id=topic_id, plan_sha256=expected_plan, seal_sha256=seal_sha256)
            logical = 512 + ((infos[0].size + 511) // 512) * 512
            logical += sum(512 + ((info.size + 511) // 512) * 512 for info in infos[1:])
            expected_tar_size = ((logical + 1024 + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE) * tarfile.RECORDSIZE
            if len(tar_bytes) != expected_tar_size:
                raise AgenticShardBundleIntegrityError("bundle has trailing or non-canonical tar bytes")
    except (tarfile.TarError, EOFError) as exc:
        raise AgenticShardBundleIntegrityError("bundle tar stream is invalid") from exc
    return VerifiedTopicBundle(
        topic_id=topic_id,
        run_plan_sha256=expected_plan,
        topic_seal_sha256=seal_sha256,
        archive_sha256=archive_sha256,
        archive_size=compressed_size,
        manifest_sha256=manifest_sha256,
        members=declared,
    )


def verify_topic_bundle(
    archive_path: str | Path,
    marker_path: str | Path,
    expected_run_plan_sha256: str,
) -> VerifiedTopicBundle:
    """Verify marker, archive, topic seal, and projection/receipt closure."""
    expected = _require_digest(expected_run_plan_sha256, "expected run plan sha256")
    archive = Path(archive_path)
    marker = _parse_marker(Path(marker_path))
    return _verify_archive(archive, marker, expected)


def pack_topic(
    source_run_dir: str | Path,
    topic_id: str,
    archive_path: str | Path,
    marker_path: str | Path,
) -> VerifiedTopicBundle:
    """Pack one already sealed topic and publish archive then completion marker."""
    source = Path(source_run_dir)
    _ensure_directory(source, "source run directory")
    try:
        plan = load_run_plan(source)
        seal = load_topic_seal(work_dir=source, plan=plan, topic_id=topic_id)
    except AgenticRunStateError as exc:
        raise AgenticShardBundleIntegrityError(f"source topic is not sealed: {exc}") from exc
    topic_dir = source / "topics" / topic_id
    source_payloads = {
        "retrieval_topic.json": _read_regular(topic_dir / "retrieval_topic.json", "retrieval topic"),
        "generation_topic.json": _read_regular(topic_dir / "generation_topic.json", "generation topic"),
        "topic_records_receipt.json": _read_regular(topic_dir / "topic_records_receipt.json", "topic records receipt"),
        TOPIC_SEAL_MEMBER: _read_regular(topic_dir / TOPIC_SEAL_MEMBER, "topic seal"),
    }
    members = tuple(
        _member_payload(name, source_payloads[name])
        for name in sorted(TOPIC_MEMBER_NAMES)
    )
    manifest = _canonical(
        _manifest_payload(
            plan_sha256=plan.plan_sha256,
            topic_id=topic_id,
            narrative_sha256=seal.narrative_sha256,
            topic_seal_sha256=seal.seal_sha256,
            members=members,
        )
    )
    archive = Path(archive_path)
    marker = Path(marker_path)
    _ensure_directory(archive.parent, "archive parent", create=True, private=False)
    _ensure_directory(marker.parent, "marker parent", create=True, private=False)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{archive.name}.", suffix=".tmp", dir=archive.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _write_archive(temporary, manifest, [(name, source_payloads[name]) for name in sorted(TOPIC_MEMBER_NAMES)])
        archive_size, archive_sha256 = _file_receipt(temporary)
        marker_body = _canonical(
            {
                "archive_sha256": archive_sha256,
                "archive_size": archive_size,
                "run_plan_sha256": plan.plan_sha256,
                "schema_version": MARKER_SCHEMA_VERSION,
                "topic_id": topic_id,
                "topic_seal_sha256": seal.seal_sha256,
            }
        )
        _verify_archive(temporary, _strict_marker_from_body(marker_body), plan.plan_sha256)
        _publish_identical(archive, temporary.read_bytes(), "bundle archive")
        _publish_identical(marker, marker_body, "bundle completion marker")
    finally:
        temporary.unlink(missing_ok=True)
    return verify_topic_bundle(archive, marker, plan.plan_sha256)


def _strict_marker_from_body(body: bytes) -> Mapping[str, object]:
    value = _strict_json(body, "bundle completion marker")
    row = _exact_mapping(value, {"archive_sha256", "archive_size", "run_plan_sha256", "schema_version", "topic_id", "topic_seal_sha256"}, "bundle completion marker")
    return row


def _extract_payloads(archive_path: Path, verified: VerifiedTopicBundle) -> dict[str, bytes]:
    tar_bytes = _decompress_archive(archive_path)
    payloads: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
        for info in archive.getmembers()[1:]:
            stream = archive.extractfile(info)
            if stream is None:
                raise AgenticShardBundleIntegrityError("bundle member unavailable during import")
            payloads[info.name] = stream.read(info.size)
    if set(payloads) != {member.path for member in verified.members}:
        raise AgenticShardBundleIntegrityError("bundle extraction closure changed")
    return payloads


def _journal_body(*, state: str, verified: VerifiedTopicBundle) -> bytes:
    if state not in {"prepare", "conflict", "complete"}:
        raise ValueError(state)
    return _canonical(
        {
            "archive_sha256": verified.archive_sha256,
            "run_plan_sha256": verified.run_plan_sha256,
            "state": state,
            "topic_id": verified.topic_id,
            "topic_seal_sha256": verified.topic_seal_sha256,
        }
    )


def _write_journal_state(path: Path, body: bytes) -> None:
    """Advance one mutable recovery journal atomically."""
    _ensure_directory(path.parent, "import journal", create=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _import_lock(destination: Path) -> Iterator[None]:
    lock_path = destination / _LOCK_NAME
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _link_or_identical(source: Path, destination: Path, label: str) -> None:
    if destination.exists() or destination.is_symlink():
        if _read_regular(destination, label) != _read_regular(source, label):
            raise AgenticShardBundleConflictError(f"{label} conflicts with immutable destination")
        return
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError:
        if _read_regular(destination, label) != _read_regular(source, label):
            raise AgenticShardBundleConflictError(f"{label} conflicts with immutable destination")
    os.chmod(destination, 0o600)
    _fsync_directory(destination.parent)


def import_topic_bundle(
    archive_path: str | Path,
    marker_path: str | Path,
    destination_run_dir: str | Path,
) -> VerifiedTopicBundle:
    """Verify first, then transactionally install one topic create-only."""
    archive = Path(archive_path)
    marker = Path(marker_path)
    marker_row = _parse_marker(marker)
    verified = verify_topic_bundle(archive, marker, str(marker_row["run_plan_sha256"]))
    destination = Path(destination_run_dir)
    _ensure_directory(destination, "destination run directory", create=True)
    journal_root = destination / _JOURNAL_DIRECTORY
    _ensure_directory(journal_root, "import journal", create=True)
    journal = journal_root / f"{verified.archive_sha256}.json"
    if journal.exists() or journal.is_symlink():
        journal_body = _read_regular(journal, "import journal")
        existing_journal = _strict_json(journal_body, "import journal")
        journal_row = _exact_mapping(
            existing_journal,
            {"archive_sha256", "run_plan_sha256", "state", "topic_id", "topic_seal_sha256"},
            "import journal",
        )
        if _canonical(journal_row) != journal_body or journal_row["state"] not in {"prepare", "conflict", "complete"}:
            raise AgenticShardBundleIntegrityError("import journal is not canonical")
        if (
            journal_row["archive_sha256"] != verified.archive_sha256
            or journal_row["run_plan_sha256"] != verified.run_plan_sha256
            or journal_row["topic_id"] != verified.topic_id
            or journal_row["topic_seal_sha256"] != verified.topic_seal_sha256
        ):
            raise AgenticShardBundleConflictError("import journal identity differs")
        if journal_row["state"] != "complete":
            _write_journal_state(journal, _journal_body(state="prepare", verified=verified))
    else:
        _publish_identical(journal, _journal_body(state="prepare", verified=verified), "prepare journal")
    payloads = _extract_payloads(archive, verified)
    with _import_lock(destination):
        topic_root = destination / "topics" / verified.topic_id
        _ensure_directory(topic_root.parent, "topics directory", create=True)
        existing_seal = topic_root / TOPIC_SEAL_MEMBER
        if existing_seal.exists() or existing_seal.is_symlink():
            try:
                if _read_regular(existing_seal, "installed topic seal") != payloads[TOPIC_SEAL_MEMBER]:
                    raise AgenticShardBundleConflictError("installed topic seal conflicts")
                for name in TOPIC_MEMBER_NAMES:
                    if _read_regular(topic_root / name, "installed topic artifact") != payloads[name]:
                        raise AgenticShardBundleConflictError("installed topic artifact conflicts")
            except AgenticShardBundleError:
                _write_journal_state(journal, _journal_body(state="conflict", verified=verified))
                raise
            _write_journal_state(journal, _journal_body(state="complete", verified=verified))
            return verified

        temporary_root = destination / f".agentic-bundle-staging-{uuid.uuid4().hex}"
        temporary_topic = temporary_root / "topics" / verified.topic_id
        try:
            _ensure_directory(temporary_topic, "temporary topic", create=True)
            attempt_number = _strict_json(payloads[TOPIC_SEAL_MEMBER], "topic seal")["attempt_number"]
            attempt_dir = temporary_topic / "attempts" / f"{int(attempt_number):06d}"
            _ensure_directory(attempt_dir, "temporary successful attempt", create=True)
            for name in TOPIC_MEMBER_NAMES:
                _publish_identical(temporary_topic / name, payloads[name], "temporary topic artifact")
            _ensure_directory(topic_root, "destination topic", create=True)
            _ensure_directory(topic_root / "attempts", "destination attempts", create=True)
            _ensure_directory(topic_root / "attempts" / attempt_dir.name, "destination successful attempt", create=True)
            for name in ("retrieval_topic.json", "generation_topic.json", "topic_records_receipt.json"):
                _link_or_identical(temporary_topic / name, topic_root / name, "topic artifact")
            # The seal is published last, preserving the source manifest-last protocol.
            _link_or_identical(temporary_topic / TOPIC_SEAL_MEMBER, topic_root / TOPIC_SEAL_MEMBER, "topic seal")
            plan_path = destination / "run_plan.json"
            if plan_path.exists() or plan_path.is_symlink():
                try:
                    plan = load_run_plan(destination)
                    if plan.plan_sha256 != verified.run_plan_sha256:
                        raise AgenticShardBundleConflictError("destination run plan differs")
                    load_topic_seal(work_dir=destination, plan=plan, topic_id=verified.topic_id)
                except (AgenticRunStateError, AgenticShardBundleError) as exc:
                    _write_journal_state(journal, _journal_body(state="conflict", verified=verified))
                    raise AgenticShardBundleIntegrityError("installed topic seal failed validation") from exc
            _write_journal_state(journal, _journal_body(state="complete", verified=verified))
        except AgenticShardBundleError:
            raise
        finally:
            for path in sorted(temporary_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    path.rmdir()
            temporary_root.rmdir() if temporary_root.exists() else None
    return verified


def summarize_staging(destination_run_dir: str | Path) -> dict[str, list[str]]:
    """Return completed and missing topic IDs without touching final exports."""
    destination = Path(destination_run_dir)
    _ensure_directory(destination, "destination run directory")
    planned: tuple[str, ...] = ()
    authenticated_plan = None
    plan_path = destination / "run_plan.json"
    if plan_path.exists() or plan_path.is_symlink():
        try:
            authenticated_plan = load_run_plan(destination)
            planned = authenticated_plan.planned_topic_ids
        except AgenticRunStateError as exc:
            raise AgenticShardBundleIntegrityError("staging run plan is invalid") from exc
    topics = destination / "topics"
    discovered = []
    if topics.exists():
        _ensure_directory(topics, "staging topics")
        discovered = sorted(path.name for path in topics.iterdir() if path.is_dir() and _SAFE_ID.fullmatch(path.name))
    topic_ids = list(planned or tuple(discovered))
    completed: list[str] = []
    for topic_id in topic_ids:
        seal = topics / topic_id / TOPIC_SEAL_MEMBER
        if seal.is_file() and not seal.is_symlink():
            if authenticated_plan is not None:
                try:
                    load_topic_seal(work_dir=destination, plan=authenticated_plan, topic_id=topic_id)
                except AgenticRunStateError as exc:
                    raise AgenticShardBundleIntegrityError("staging topic seal is invalid") from exc
            completed.append(topic_id)
    missing = [topic_id for topic_id in topic_ids if topic_id not in completed]
    return {"topic_ids": topic_ids, "completed_topic_ids": completed, "missing_topic_ids": missing}


def _json_result(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _bundle_payload(bundle: VerifiedTopicBundle) -> dict[str, object]:
    return {
        "archive_sha256": bundle.archive_sha256,
        "archive_size": bundle.archive_size,
        "bundle_schema": bundle.bundle_schema,
        "manifest_sha256": bundle.manifest_sha256,
        "members": [{"path": m.path, "sha256": m.sha256, "size": m.size} for m in bundle.members],
        "run_plan_sha256": bundle.run_plan_sha256,
        "topic_id": bundle.topic_id,
        "topic_seal_sha256": bundle.topic_seal_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("source_run_dir", type=Path)
    pack.add_argument("topic_id")
    pack.add_argument("archive_path", type=Path)
    pack.add_argument("marker_path", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("archive_path", type=Path)
    verify.add_argument("marker_path", type=Path)
    verify.add_argument("expected_run_plan_sha256")
    imp = commands.add_parser("import")
    imp.add_argument("archive_path", type=Path)
    imp.add_argument("marker_path", type=Path)
    imp.add_argument("destination_run_dir", type=Path)
    status = commands.add_parser("status")
    status.add_argument("destination_run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            _json_result(_bundle_payload(pack_topic(args.source_run_dir, args.topic_id, args.archive_path, args.marker_path)))
        elif args.command == "verify":
            _json_result(_bundle_payload(verify_topic_bundle(args.archive_path, args.marker_path, args.expected_run_plan_sha256)))
        elif args.command == "import":
            _json_result(_bundle_payload(import_topic_bundle(args.archive_path, args.marker_path, args.destination_run_dir)))
        else:
            _json_result(summarize_staging(args.destination_run_dir))
    except AgenticShardBundleError as exc:
        print(f"agentic topic bundle: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BUNDLE_COMPLETE_NAME",
    "BUNDLE_MANIFEST_NAME",
    "BUNDLE_SCHEMA_VERSION",
    "AgenticShardBundleConflictError",
    "AgenticShardBundleError",
    "AgenticShardBundleIntegrityError",
    "MARKER_SCHEMA_VERSION",
    "TOPIC_MEMBER_NAMES",
    "TopicBundleMember",
    "VerifiedTopicBundle",
    "import_topic_bundle",
    "main",
    "pack_topic",
    "summarize_staging",
    "verify_topic_bundle",
]
