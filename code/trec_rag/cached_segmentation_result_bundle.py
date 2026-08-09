"""Deterministic private bundles for cached segmentation A/B validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import tarfile
import tempfile
from typing import Any, BinaryIO, Sequence
import unicodedata

import zstandard

from trec_rag.generation_handoff import load_generation_handoff
from trec_rag.retrieval_nugget_coverage import (
    load_completed_coverage_evaluation,
)


BUNDLE_SCHEMA = "cached-segmentation-result-bundle-v1"
COMPLETE_SCHEMA = "cached-segmentation-result-bundle-complete-v1"
ARCHIVE_NAME = "bundle.tar.zst"
COMPLETE_NAME = "bundle-complete.json"
MANIFEST_NAME = "bundle-manifest.json"
DIAGNOSTIC_STATUS_NAME = "diagnostic-status.json"
DIAGNOSTIC_SCHEMA = "cached-segmentation-diagnostic-v1"
COVERAGE_FILES = (
    "input.json",
    "plan.json",
    "judgments.json",
    "report.json",
    "manifest.json",
)
EXPORT_FILES = (
    "generation_handoff_manifest.json",
    "r_output_trec_rag_2026.tsv",
    "retrieval_with_text.jsonl.zip",
    "retrieval_export_manifest.json",
    "cache-operation-manifest.json",
)
PHASE_ARTIFACTS = {
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
OPERATION_STAGES = (
    "planning",
    "retrieval",
    "passage_scores",
    "sentence_scores",
    "similarity",
    "canonicalization",
)
OPERATION_COUNTERS = (
    "cache_hits",
    "cache_misses",
    "network_calls",
    "provider_calls",
    "model_batches",
)
UPSTREAM_STAGES = ("planning", "retrieval", "passage_scores")
MAX_COMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 24 * 1024 * 1024 * 1024
MAX_MEMBER_BYTES = 4 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_MEMBERS = 10_000
MAX_ZSTD_WINDOW_BYTES = 128 * 1024 * 1024
_CHUNK = 1024 * 1024


class ResultBundleError(RuntimeError):
    """Base error for private validation bundles."""


class ResultBundleIntegrityError(ResultBundleError):
    """The source closure, marker, or archive is incomplete or unsafe."""


class ResultBundleConflictError(ResultBundleError):
    """A create-only bundle destination already contains different bytes."""


@dataclass(frozen=True)
class BundleMember:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class VerifiedResultBundle:
    bundle_dir: Path
    bundle_kind: str
    run_id: str
    git_revision: str
    topic_ids: tuple[str, ...]
    archive_sha256: str
    archive_size: int
    manifest_sha256: str
    members: tuple[BundleMember, ...]


@dataclass(frozen=True)
class _SourceMember:
    path: str
    source: Path
    size: int
    sha256: str

    @property
    def public(self) -> BundleMember:
        return BundleMember(self.path, self.size, self.sha256)


class _BoundedReader:
    def __init__(self, source: BinaryIO, maximum: int) -> None:
        self.source = source
        self.maximum = maximum
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        body = self.source.read(size)
        self.count += len(body)
        if self.count > self.maximum:
            raise ResultBundleIntegrityError("decompressed-size limit exceeded")
        return body


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
        raise ResultBundleIntegrityError("bundle metadata is not canonical JSON") from exc


def _strict_json(body: bytes, label: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ResultBundleIntegrityError(f"{label} has a duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            body.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ResultBundleIntegrityError(
                    f"{label} has non-standard JSON constant {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultBundleIntegrityError(f"{label} is not strict JSON") from exc


def _fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ResultBundleIntegrityError(f"{label} fields changed")
    return value


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ResultBundleIntegrityError(f"{label} is not a SHA-256 digest")
    return value


def _nonnegative(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ResultBundleIntegrityError(f"{label} must be a non-negative integer")
    return value


def _file_receipt(path: Path) -> tuple[int, str]:
    digest = sha256()
    count = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK), b""):
            count += len(chunk)
            digest.update(chunk)
    return count, digest.hexdigest()


def _safe_path(name: str, seen: dict[str, str]) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ResultBundleIntegrityError("archive member has an unsafe path")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != name
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ResultBundleIntegrityError(f"archive member path is unsafe: {name!r}")
    key = unicodedata.normalize("NFC", name).casefold()
    if key in seen:
        raise ResultBundleIntegrityError(
            "duplicate archive member" if seen[key] == name else "archive path collision"
        )
    seen[key] = name
    return name


def _require_regular(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ResultBundleIntegrityError("bundle source escaped its root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ResultBundleIntegrityError("bundle source contains a symbolic link")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ResultBundleIntegrityError("bundle source is not a regular file")


def _source(path: Path, root: Path, archive_path: str) -> _SourceMember:
    _safe_path(archive_path, {})
    _require_regular(path, root)
    size, digest = _file_receipt(path)
    if size > MAX_MEMBER_BYTES:
        raise ResultBundleIntegrityError("bundle source member is too large")
    return _SourceMember(archive_path, path, size, digest)


def _source_root(path: Path, label: str) -> Path:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"{label} must be absolute")
    current = Path(source.anchor)
    for part in source.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise ResultBundleIntegrityError(f"{label} is missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ResultBundleIntegrityError(f"{label} contains a symbolic link")
    if not source.is_dir():
        raise ResultBundleIntegrityError(f"{label} is not a directory")
    return source.resolve()


def _destination(path: Path) -> Path:
    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("destination must be absolute")
    current = Path(destination.anchor)
    for part in destination.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ResultBundleIntegrityError("destination contains a symbolic link")
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or not destination.is_dir():
        raise ResultBundleIntegrityError("destination is not a safe directory")
    extras = {
        item.name
        for item in destination.iterdir()
        if item.name not in {ARCHIVE_NAME, COMPLETE_NAME}
    }
    if extras:
        raise ResultBundleIntegrityError(
            f"destination has undeclared files: {sorted(extras)!r}"
        )
    return destination


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o600
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_archive(path: Path, manifest: bytes, sources: Sequence[_SourceMember]) -> None:
    with path.open("xb") as raw:
        compressor = zstandard.ZstdCompressor(
            level=10,
            threads=0,
            write_checksum=True,
            write_content_size=True,
        )
        with compressor.stream_writer(raw, closefd=False) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w|",
                format=tarfile.GNU_FORMAT,
            ) as archive:
                archive.addfile(_tar_info(MANIFEST_NAME, len(manifest)), io.BytesIO(manifest))
                for member in sources:
                    with member.source.open("rb") as body:
                        archive.addfile(_tar_info(member.path, member.size), body)
        raw.flush()
        os.fsync(raw.fileno())


def _publish(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file():
            raise ResultBundleConflictError("bundle destination is unsafe")
        if _file_receipt(temporary) != _file_receipt(destination):
            raise ResultBundleConflictError("bundle destination already differs")


def _manifest_payload(
    *,
    kind: str,
    run_id: str,
    revision: str,
    topics: Sequence[str],
    members: Sequence[_SourceMember],
) -> dict[str, object]:
    return {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_kind": kind,
        "run_id": run_id,
        "git_revision": revision,
        "topic_ids": list(topics),
        "members": [
            {"path": row.path, "size": row.size, "sha256": row.sha256}
            for row in members
        ],
    }


def _pack(
    destination: Path,
    *,
    kind: str,
    run_id: str,
    revision: str,
    topics: Sequence[str],
    sources: Sequence[_SourceMember],
) -> VerifiedResultBundle:
    root = _destination(destination)
    ordered = tuple(sorted(sources, key=lambda row: row.path))
    seen: dict[str, str] = {MANIFEST_NAME.casefold(): MANIFEST_NAME}
    for member in ordered:
        _safe_path(member.path, seen)
    manifest = _canonical(
        _manifest_payload(
            kind=kind,
            run_id=run_id,
            revision=revision,
            topics=topics,
            members=ordered,
        )
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{ARCHIVE_NAME}.", suffix=".tmp", dir=root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _write_archive(temporary, manifest, ordered)
        archive_size, archive_sha = _file_receipt(temporary)
        if archive_size > MAX_COMPRESSED_BYTES:
            raise ResultBundleIntegrityError("compressed-size limit exceeded")
        complete = _canonical(
            {
                "schema_version": COMPLETE_SCHEMA,
                "archive": ARCHIVE_NAME,
                "archive_size": archive_size,
                "archive_sha256": archive_sha,
                "manifest_sha256": sha256(manifest).hexdigest(),
                "member_count": len(ordered),
                "bundle_kind": kind,
                "run_id": run_id,
                "git_revision": revision,
                "topic_ids": list(topics),
            }
        )
        _verify_archive_bytes(temporary, complete)
        _publish(temporary, root / ARCHIVE_NAME)
        marker = root / f".{COMPLETE_NAME}.{os.getpid()}.tmp"
        marker.write_bytes(complete)
        try:
            _publish(marker, root / COMPLETE_NAME)
        finally:
            marker.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return _verify(root, expected_kind=kind)


def pack_baseline_bundle(
    handoff: str | Path,
    coverage_root: str | Path,
    destination: str | Path,
) -> VerifiedResultBundle:
    """Pack only one authenticated handoff and its complete coverage states."""

    raw_handoff = Path(handoff)
    source_root = _source_root(raw_handoff.parent, "baseline source root")
    if raw_handoff.is_symlink() or not raw_handoff.is_file():
        raise ResultBundleIntegrityError("baseline handoff is missing or unsafe")
    handoff_path = raw_handoff.resolve()
    loaded = load_generation_handoff(handoff_path)
    topics = tuple(topic.topic_id for topic in loaded.topics)
    if not topics or len(set(topics)) != len(topics):
        raise ResultBundleIntegrityError("baseline topic order is invalid")
    coverage = _source_root(Path(coverage_root), "baseline coverage root")
    if {item.name for item in coverage.iterdir()} != set(topics):
        raise ResultBundleIntegrityError("baseline coverage topic set changed")
    sources = [_source(handoff_path, source_root, "generation_handoff_manifest.json")]
    for topic_id in topics:
        work = coverage / topic_id
        try:
            load_completed_coverage_evaluation(
                handoff_manifest_path=handoff_path,
                topic_id=topic_id,
                work_dir=work,
            )
        except Exception as exc:
            raise ResultBundleIntegrityError(
                f"baseline coverage is invalid for topic {topic_id}"
            ) from exc
        if {item.name for item in work.iterdir()} != set(COVERAGE_FILES):
            raise ResultBundleIntegrityError("baseline coverage files changed")
        for name in COVERAGE_FILES:
            sources.append(
                _source(
                    work / name,
                    coverage,
                    f"retrieval_nugget_coverage_v2/{topic_id}/{name}",
                )
            )
    return _pack(
        Path(destination),
        kind="baseline",
        run_id=loaded.producer.retrieval_run_id,
        revision=loaded.producer.producer_revision,
        topics=topics,
        sources=sources,
    )


def _parse_manifest(
    body: bytes,
) -> tuple[str, str, str, tuple[str, ...], tuple[BundleMember, ...]]:
    value = _fields(
        _strict_json(body, "bundle manifest"),
        {"schema_version", "bundle_kind", "run_id", "git_revision", "topic_ids", "members"},
        "bundle manifest",
    )
    if value["schema_version"] != BUNDLE_SCHEMA or _canonical(value) != body:
        raise ResultBundleIntegrityError("bundle manifest schema or encoding changed")
    kind = value["bundle_kind"]
    run_id = value["run_id"]
    revision = value["git_revision"]
    topics = value["topic_ids"]
    if (
        kind not in {"baseline", "result", "diagnostic"}
        or not isinstance(run_id, str)
        or not run_id
        or not isinstance(revision, str)
        or len(revision) != 40
        or not isinstance(topics, list)
        or not topics
        or any(not isinstance(topic, str) or not topic for topic in topics)
        or len(set(topics)) != len(topics)
    ):
        raise ResultBundleIntegrityError("bundle manifest identity changed")
    raw_members = value["members"]
    if not isinstance(raw_members, list) or len(raw_members) > MAX_MEMBERS:
        raise ResultBundleIntegrityError("bundle member list changed")
    seen: dict[str, str] = {MANIFEST_NAME.casefold(): MANIFEST_NAME}
    members: list[BundleMember] = []
    for raw in raw_members:
        row = _fields(raw, {"path", "size", "sha256"}, "bundle member")
        path = _safe_path(row["path"], seen)
        members.append(
            BundleMember(
                path,
                _nonnegative(row["size"], "member size"),
                _digest(row["sha256"], "member sha256"),
            )
        )
        if members[-1].size > MAX_MEMBER_BYTES:
            raise ResultBundleIntegrityError("member size exceeds limit")
    if tuple(member.path for member in members) != tuple(
        sorted(member.path for member in members)
    ):
        raise ResultBundleIntegrityError("bundle members are not ordered")
    return kind, run_id, revision, tuple(topics), tuple(members)


def _parse_complete(body: bytes) -> dict[str, Any]:
    value = _fields(
        _strict_json(body, "bundle completion marker"),
        {
            "schema_version",
            "archive",
            "archive_size",
            "archive_sha256",
            "manifest_sha256",
            "member_count",
            "bundle_kind",
            "run_id",
            "git_revision",
            "topic_ids",
        },
        "bundle completion marker",
    )
    if (
        value["schema_version"] != COMPLETE_SCHEMA
        or value["archive"] != ARCHIVE_NAME
        or _canonical(value) != body
    ):
        raise ResultBundleIntegrityError("bundle completion marker changed")
    _nonnegative(value["archive_size"], "archive size")
    _nonnegative(value["member_count"], "member count")
    _digest(value["archive_sha256"], "archive sha256")
    _digest(value["manifest_sha256"], "manifest sha256")
    return value


def _consume_member(source: BinaryIO, member: BundleMember, target: Path) -> None:
    digest = sha256()
    remaining = member.size
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as sink:
        while remaining:
            chunk = source.read(min(_CHUNK, remaining))
            if not chunk:
                raise ResultBundleIntegrityError("archive member is truncated")
            remaining -= len(chunk)
            digest.update(chunk)
            sink.write(chunk)
        if source.read(1):
            raise ResultBundleIntegrityError("archive member exceeds declared size")
    if digest.hexdigest() != member.sha256:
        raise ResultBundleIntegrityError("archive member digest changed")


def _verify_archive_bytes(
    archive_path: Path,
    complete_body: bytes,
) -> tuple[VerifiedResultBundle, Path | None]:
    marker = _parse_complete(complete_body)
    archive_size, archive_sha = _file_receipt(archive_path)
    if archive_size != marker["archive_size"] or archive_sha != marker["archive_sha256"]:
        raise ResultBundleIntegrityError("archive differs from completion marker")
    if archive_size > MAX_COMPRESSED_BYTES:
        raise ResultBundleIntegrityError("compressed-size limit exceeded")
    try:
        with archive_path.open("rb") as frame_source:
            frame = zstandard.get_frame_parameters(frame_source.read(18))
    except (OSError, zstandard.ZstdError) as exc:
        raise ResultBundleIntegrityError("bundle zstd frame is invalid") from exc
    if frame.window_size > MAX_ZSTD_WINDOW_BYTES:
        raise ResultBundleIntegrityError("zstd window exceeds limit")
    temporary_context = tempfile.TemporaryDirectory(prefix="segmentation-bundle-")
    extraction = Path(temporary_context.name)
    manifest_body: bytes | None = None
    identity: tuple[str, str, str, tuple[str, ...], tuple[BundleMember, ...]] | None = None
    member_index = 0
    with archive_path.open("rb") as raw:
        decompressor = zstandard.ZstdDecompressor(
            max_window_size=MAX_ZSTD_WINDOW_BYTES
        ).stream_reader(raw, read_across_frames=True)
        bounded = _BoundedReader(decompressor, MAX_DECOMPRESSED_BYTES)
        try:
            with tarfile.open(fileobj=bounded, mode="r|") as archive:
                for member_index, info in enumerate(archive, start=1):
                    if (
                        not info.isreg()
                        or info.mode != 0o600
                        or info.uid != 0
                        or info.gid != 0
                        or info.uname
                        or info.gname
                        or info.mtime != 0
                        or info.pax_headers
                    ):
                        raise ResultBundleIntegrityError("archive member metadata is unsafe")
                    body = archive.extractfile(info)
                    if body is None:
                        raise ResultBundleIntegrityError("archive member body is missing")
                    if member_index == 1:
                        if info.name != MANIFEST_NAME or info.size > MAX_MANIFEST_BYTES:
                            raise ResultBundleIntegrityError("manifest must be the first member")
                        manifest_body = body.read(MAX_MANIFEST_BYTES + 1)
                        if len(manifest_body) != info.size:
                            raise ResultBundleIntegrityError("manifest size changed")
                        if sha256(manifest_body).hexdigest() != marker["manifest_sha256"]:
                            raise ResultBundleIntegrityError("manifest digest changed")
                        identity = _parse_manifest(manifest_body)
                        continue
                    if identity is None:
                        raise ResultBundleIntegrityError("archive has no manifest")
                    declared = identity[4]
                    index = member_index - 2
                    if index >= len(declared):
                        raise ResultBundleIntegrityError("archive has an undeclared member")
                    expected = declared[index]
                    if info.name != expected.path or info.size != expected.size:
                        raise ResultBundleIntegrityError("archive member order or size changed")
                    _consume_member(
                        body,
                        expected,
                        extraction.joinpath(*PurePosixPath(expected.path).parts),
                    )
            while bounded.read(_CHUNK):
                pass
        except (tarfile.TarError, zstandard.ZstdError, EOFError) as exc:
            raise ResultBundleIntegrityError("bundle archive stream is invalid") from exc
        finally:
            decompressor.close()
    if identity is None or manifest_body is None:
        raise ResultBundleIntegrityError("archive has no manifest")
    kind, run_id, revision, topics, members = identity
    if member_index != len(members) + 1:
        raise ResultBundleIntegrityError("archive is missing declared members")
    logical = 512 + ((len(manifest_body) + 511) // 512) * 512
    logical += sum(512 + ((row.size + 511) // 512) * 512 for row in members)
    expected_tar = (
        (logical + 1024 + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE
    ) * tarfile.RECORDSIZE
    if bounded.count != expected_tar:
        raise ResultBundleIntegrityError("archive has trailing decompressed bytes")
    if (
        marker["bundle_kind"] != kind
        or marker["run_id"] != run_id
        or marker["git_revision"] != revision
        or marker["topic_ids"] != list(topics)
        or marker["member_count"] != len(members)
    ):
        raise ResultBundleIntegrityError("marker and manifest identity differ")
    verified = VerifiedResultBundle(
        bundle_dir=archive_path.parent.resolve(),
        bundle_kind=kind,
        run_id=run_id,
        git_revision=revision,
        topic_ids=topics,
        archive_sha256=archive_sha,
        archive_size=archive_size,
        manifest_sha256=sha256(manifest_body).hexdigest(),
        members=members,
    )
    temporary_context.cleanup()
    return verified, None


def _verify_baseline_semantics(verified: VerifiedResultBundle, root: Path) -> None:
    handoff_path = root / "generation_handoff_manifest.json"
    handoff = load_generation_handoff(handoff_path)
    if (
        handoff.producer.retrieval_run_id != verified.run_id
        or handoff.producer.producer_revision != verified.git_revision
        or tuple(topic.topic_id for topic in handoff.topics) != verified.topic_ids
    ):
        raise ResultBundleIntegrityError("baseline handoff identity changed")
    expected = {"generation_handoff_manifest.json"}
    for topic_id in verified.topic_ids:
        work = root / "retrieval_nugget_coverage_v2" / topic_id
        try:
            load_completed_coverage_evaluation(
                handoff_manifest_path=handoff_path,
                topic_id=topic_id,
                work_dir=work,
            )
        except Exception as exc:
            raise ResultBundleIntegrityError(
                f"baseline coverage failed validation for topic {topic_id}"
            ) from exc
        expected.update(
            f"retrieval_nugget_coverage_v2/{topic_id}/{name}"
            for name in COVERAGE_FILES
        )
    if {member.path for member in verified.members} != expected:
        raise ResultBundleIntegrityError("baseline member allowlist changed")


def _verify(bundle_dir: Path, *, expected_kind: str) -> VerifiedResultBundle:
    root = Path(bundle_dir)
    if root.is_symlink() or not root.is_dir():
        raise ResultBundleIntegrityError("bundle directory is missing or unsafe")
    if {item.name for item in root.iterdir()} != {ARCHIVE_NAME, COMPLETE_NAME}:
        raise ResultBundleIntegrityError("bundle directory has undeclared files")
    for path in (root / ARCHIVE_NAME, root / COMPLETE_NAME):
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise ResultBundleIntegrityError("bundle files are missing or unsafe")
    if (root / COMPLETE_NAME).stat().st_size > MAX_MANIFEST_BYTES:
        raise ResultBundleIntegrityError("bundle completion marker exceeds size limit")
    complete = (root / COMPLETE_NAME).read_bytes()
    verified, _ = _verify_archive_bytes(root / ARCHIVE_NAME, complete)
    if verified.bundle_kind != expected_kind:
        raise ResultBundleIntegrityError("bundle kind changed")
    with tempfile.TemporaryDirectory(prefix="segmentation-bundle-semantic-") as temp:
        extraction = Path(temp)
        _extract_verified(root / ARCHIVE_NAME, complete, extraction)
        if expected_kind == "baseline":
            _verify_baseline_semantics(verified, extraction)
        elif expected_kind == "result":
            _verify_result_semantics(verified, extraction)
        else:
            _verify_diagnostic_semantics(verified, extraction)
    return verified


def _extract_verified(archive_path: Path, complete: bytes, destination: Path) -> None:
    marker = _parse_complete(complete)
    with archive_path.open("rb") as raw:
        decompressed = zstandard.ZstdDecompressor(
            max_window_size=MAX_ZSTD_WINDOW_BYTES
        ).stream_reader(raw, read_across_frames=True)
        bounded = _BoundedReader(decompressed, MAX_DECOMPRESSED_BYTES)
        identity = None
        with tarfile.open(fileobj=bounded, mode="r|") as archive:
            for index, info in enumerate(archive):
                body = archive.extractfile(info)
                if body is None:
                    raise ResultBundleIntegrityError("archive member body is missing")
                if index == 0:
                    manifest = body.read(MAX_MANIFEST_BYTES + 1)
                    if sha256(manifest).hexdigest() != marker["manifest_sha256"]:
                        raise ResultBundleIntegrityError("manifest digest changed")
                    identity = _parse_manifest(manifest)
                    continue
                if identity is None:
                    raise ResultBundleIntegrityError("archive has no manifest")
                member = identity[4][index - 1]
                _consume_member(
                    body,
                    member,
                    destination.joinpath(*PurePosixPath(member.path).parts),
                )
        decompressed.close()


def verify_baseline_bundle(bundle_dir: str | Path) -> VerifiedResultBundle:
    return _verify(Path(bundle_dir), expected_kind="baseline")


def restore_baseline_bundle(
    bundle_dir: str | Path,
    destination: str | Path,
) -> VerifiedResultBundle:
    """Verify and extract a baseline bundle into one fresh private directory."""

    verified = verify_baseline_bundle(bundle_dir)
    target = Path(destination)
    if not target.is_absolute():
        raise ValueError("baseline restore destination must be absolute")
    if target.exists():
        if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
            raise ResultBundleIntegrityError(
                "baseline restore destination must be an empty safe directory"
            )
    else:
        target.mkdir(parents=True, mode=0o700)
    complete = (Path(bundle_dir) / COMPLETE_NAME).read_bytes()
    _extract_verified(Path(bundle_dir) / ARCHIVE_NAME, complete, target)
    _verify_baseline_semantics(verified, target)
    return verified


def _strict_file(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ResultBundleIntegrityError(f"{label} is missing or unsafe")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ResultBundleIntegrityError(f"{label} exceeds size limit")
    body = path.read_bytes()
    value = _strict_json(body, label)
    if not isinstance(value, dict):
        raise ResultBundleIntegrityError(f"{label} must contain an object")
    return body, value


def _validate_receipt_digest(body: bytes, value: dict[str, Any], label: str) -> None:
    content = dict(value)
    claimed = _digest(content.pop("receipt_content_sha256", None), f"{label} content")
    if claimed != sha256(_canonical(content)).hexdigest() or _canonical(value) != body:
        raise ResultBundleIntegrityError(f"{label} content digest changed")


def _validate_stages(value: object, label: str) -> dict[str, dict[str, int]]:
    stages = _fields(value, set(OPERATION_STAGES), label)
    result: dict[str, dict[str, int]] = {}
    for stage_name in OPERATION_STAGES:
        counters = _fields(
            stages[stage_name], set(OPERATION_COUNTERS), f"{label} {stage_name}"
        )
        result[stage_name] = {
            counter: _nonnegative(counters[counter], f"{stage_name} {counter}")
            for counter in OPERATION_COUNTERS
        }
    for stage_name in UPSTREAM_STAGES:
        for counter in (
            "cache_misses",
            "network_calls",
            "provider_calls",
            "model_batches",
        ):
            if result[stage_name][counter] != 0:
                raise ResultBundleIntegrityError(
                    "result operation receipt contains forbidden upstream work"
                )
    return result


def _validate_phases(value: object, label: str) -> None:
    phases = _fields(value, {"planning", "retrieval", "scoring", "canonical"}, label)
    for phase_name, raw in phases.items():
        phase = _fields(raw, {"resumed"}, f"{label} {phase_name}")
        if type(phase["resumed"]) is not bool:
            raise ResultBundleIntegrityError(f"{label} resumed flag changed")


def _phase_sources(
    topic_root: Path,
    run_root: Path,
    topic_id: str,
    phase: str,
) -> list[_SourceMember]:
    complete_path = topic_root / phase / "complete.json"
    body, manifest = _strict_file(complete_path, f"{phase} checkpoint")
    receipts = manifest.get("artifacts")
    if manifest.get("topic_id") != topic_id or not isinstance(receipts, list):
        raise ResultBundleIntegrityError(f"{phase} checkpoint identity changed")
    found: dict[str, Path] = {}
    for raw in receipts:
        row = _fields(raw, {"relative_path", "bytes", "sha256"}, "artifact receipt")
        relative = row["relative_path"]
        if not isinstance(relative, str):
            raise ResultBundleIntegrityError("artifact receipt path changed")
        _safe_path(relative, {})
        if relative in found:
            raise ResultBundleIntegrityError("artifact receipt is duplicated")
        source = topic_root.joinpath(*PurePosixPath(relative).parts)
        _require_regular(source, topic_root)
        size, digest = _file_receipt(source)
        if (
            size != _nonnegative(row["bytes"], "artifact bytes")
            or digest != _digest(row["sha256"], "artifact sha256")
        ):
            raise ResultBundleIntegrityError("checkpoint artifact changed")
        found[relative] = source
    if set(found) != PHASE_ARTIFACTS[phase]:
        raise ResultBundleIntegrityError(f"{phase} checkpoint allowlist changed")
    sources = [
        _source(
            complete_path,
            run_root,
            f"run/{topic_id}/{phase}/complete.json",
        )
    ]
    del body
    sources.extend(
        _source(
            found[relative],
            run_root,
            f"run/{topic_id}/{relative}",
        )
        for relative in sorted(found)
    )
    return sources


def _comparison_sources(
    validation_root: Path,
    run_root: Path,
    topics: tuple[str, ...],
    candidate_handoff_sha256: str,
    run_id: str,
    config_sha256: str,
    selected_workers: int,
) -> list[_SourceMember]:
    sources: list[_SourceMember] = []
    for kind in ("structural", "semantic"):
        comparison_path = validation_root / f"{kind}-comparison.json"
        manifest_path = validation_root / f"{kind}-comparison-manifest.json"
        comparison_body, comparison = _strict_file(
            comparison_path, f"{kind} comparison"
        )
        manifest_body, manifest = _strict_file(
            manifest_path, f"{kind} comparison manifest"
        )
        expected_schema = f"cached-segmentation-{kind}-comparison-v1"
        expected_manifest_schema = f"cached-segmentation-{kind}-manifest-v1"
        if (
            comparison.get("schema_version") != expected_schema
            or comparison.get("topic_ids") != list(topics)
            or comparison.get("candidate_handoff_sha256")
            != candidate_handoff_sha256
            or comparison.get("gates_passed") is not True
            or (
                kind == "structural"
                and comparison.get("improvement_observed") is not True
            )
            or manifest.get("schema_version") != expected_manifest_schema
            or manifest.get("comparison_file") != comparison_path.name
            or manifest.get("comparison_bytes") != len(comparison_body)
            or manifest.get("comparison_sha256")
            != sha256(comparison_body).hexdigest()
            or manifest.get("topic_ids") != list(topics)
            or manifest.get("gates_passed") is not True
        ):
            raise ResultBundleIntegrityError(f"{kind} comparison identity changed")
        if kind == "semantic" and (
            comparison.get("planner_calls") != 0
            or comparison.get("candidate_judge_calls") != len(topics)
        ):
            raise ResultBundleIntegrityError("semantic comparison call count changed")
        del manifest_body
        sources.extend(
            (
                _source(
                    comparison_path,
                    validation_root,
                    f"validation/{comparison_path.name}",
                ),
                _source(
                    manifest_path,
                    validation_root,
                    f"validation/{manifest_path.name}",
                ),
            )
        )
    decision_path = validation_root / "concurrency-decision.json"
    decision_body, decision = _strict_file(
        decision_path, "concurrency decision"
    )
    expected_decision_fields = {
        "schema_version",
        "run_id",
        "config_sha256",
        "gpu_name",
        "gpu_uuid",
        "driver_version",
        "probe_topics",
        "probe_workers",
        "probe_duration_seconds",
        "probe_exit_status",
        "minimum_peak_delta_mib",
        "selected_workers",
        "elapsed_seconds",
        "idle_memory_mib",
        "peak_memory_mib",
        "total_memory_mib",
        "projected_selected_worker_memory_mib",
        "safe_limit_memory_mib",
    }
    integer_fields = expected_decision_fields - {
        "schema_version",
        "run_id",
        "config_sha256",
        "gpu_name",
        "gpu_uuid",
        "driver_version",
        "probe_topics",
    }
    gpu_name = decision.get("gpu_name")
    gpu_uuid = decision.get("gpu_uuid")
    driver_version = decision.get("driver_version")
    if (
        set(decision) != expected_decision_fields
        or _canonical(decision) != decision_body
        or decision.get("schema_version")
        != "cached-segmentation-concurrency-decision-v3"
        or decision.get("run_id") != run_id
        or decision.get("config_sha256") != config_sha256
        or not isinstance(gpu_name, str)
        or not any(model in gpu_name for model in ("H100", "H200"))
        or not isinstance(gpu_uuid, str)
        or not gpu_uuid.startswith("GPU-")
        or not isinstance(driver_version, str)
        or not driver_version
        or decision.get("probe_topics") != ["14", "31"]
        or decision.get("probe_workers") != 2
        or decision.get("probe_duration_seconds") != 900
        or decision.get("probe_exit_status") not in {0, 124}
        or decision.get("minimum_peak_delta_mib") != 4096
        or decision.get("selected_workers") != selected_workers
        or type(selected_workers) is not int
        or selected_workers != 20
        or selected_workers % decision.get("probe_workers", 1) != 0
        or any(type(decision.get(field)) is not int for field in integer_fields)
    ):
        raise ResultBundleIntegrityError("concurrency decision identity changed")
    idle = decision["idle_memory_mib"]
    peak = decision["peak_memory_mib"]
    total = decision["total_memory_mib"]
    elapsed = decision["elapsed_seconds"]
    probe_duration = decision["probe_duration_seconds"]
    probe_status = decision["probe_exit_status"]
    expected_projected = idle + (
        selected_workers // decision["probe_workers"]
    ) * max(0, peak - idle)
    expected_safe_limit = int(total * 0.90)
    if (
        elapsed <= 0
        or (
            probe_status == 124
            and not probe_duration <= elapsed <= probe_duration + 60
        )
        or (probe_status == 0 and elapsed > probe_duration)
        or idle < 0
        or peak < idle
        or peak - idle < decision["minimum_peak_delta_mib"]
        or total <= 0
        or peak > total
        or decision["projected_selected_worker_memory_mib"] != expected_projected
        or decision["safe_limit_memory_mib"] != expected_safe_limit
        or expected_projected > expected_safe_limit
    ):
        raise ResultBundleIntegrityError("concurrency decision identity changed")
    sources.append(
        _source(
            decision_path,
            validation_root,
            "validation/concurrency-decision.json",
        )
    )
    return sources


def _validated_result_sources(
    run_root: Path,
    validation_root: Path,
) -> tuple[str, str, tuple[str, ...], list[_SourceMember]]:
    run = _source_root(Path(run_root), "result run root")
    validation = _source_root(Path(validation_root), "result validation root")
    handoff_path = run / "generation_handoff_manifest.json"
    handoff = load_generation_handoff(handoff_path)
    run_id = handoff.producer.retrieval_run_id
    revision = handoff.producer.producer_revision
    topics = tuple(topic.topic_id for topic in handoff.topics)
    if not topics or len(set(topics)) != len(topics):
        raise ResultBundleIntegrityError("result topic order changed")

    export_body, export = _strict_file(
        run / "retrieval_export_manifest.json", "retrieval export manifest"
    )
    execution = _fields(
        export.get("execution"), {"topic_workers"}, "retrieval export execution"
    )
    selected_workers = execution.get("topic_workers")
    artifacts = export.get("artifacts")
    expected_export_artifacts = {
        "generation_handoff_manifest.json",
        "r_output_trec_rag_2026.tsv",
        "retrieval_with_text.jsonl.zip",
    }
    if (
        export.get("run_id") != run_id
        or export.get("export_code_commit") != revision
        or export.get("selected_topic_ids") != list(topics)
        or type(selected_workers) is not int
        or selected_workers != 20
        or not isinstance(artifacts, dict)
        or set(artifacts) != expected_export_artifacts
    ):
        raise ResultBundleIntegrityError("retrieval export identity changed")
    for name in expected_export_artifacts:
        row = _fields(artifacts[name], {"bytes", "sha256"}, "export artifact")
        size, digest = _file_receipt(run / name)
        if size != _nonnegative(row["bytes"], "export bytes") or digest != _digest(
            row["sha256"], "export sha256"
        ):
            raise ResultBundleIntegrityError("retrieval export artifact changed")

    operation_body, raw_operation = _strict_file(
        run / "cache-operation-manifest.json", "cache operation manifest"
    )
    operation = _fields(
        raw_operation,
        {
            "schema_version",
            "mode",
            "run_id",
            "config_sha256",
            "topic_ids",
            "topic_receipt_sha256s",
            "projection_manifest_sha256s",
            "retrieval_export_manifest",
            "retrieval_export_manifest_sha256",
            "totals",
            "receipt_content_sha256",
        },
        "cache operation manifest",
    )
    _validate_receipt_digest(operation_body, operation, "cache operation manifest")
    config_sha = _digest(operation.get("config_sha256"), "operation config")
    projection_digests = operation.get("projection_manifest_sha256s")
    receipt_digests = operation.get("topic_receipt_sha256s")
    if (
        operation.get("schema_version") != "cache-operation-manifest-v1"
        or operation.get("mode") != "cached-upstream-rescore"
        or operation.get("run_id") != run_id
        or operation.get("topic_ids") != list(topics)
        or operation.get("retrieval_export_manifest")
        != "retrieval_export_manifest.json"
        or operation.get("retrieval_export_manifest_sha256")
        != sha256(export_body).hexdigest()
        or not isinstance(projection_digests, list)
        or not isinstance(receipt_digests, list)
        or len(projection_digests) != len(topics)
        or len(receipt_digests) != len(topics)
    ):
        raise ResultBundleIntegrityError("cache operation manifest identity changed")
    totals = _validate_stages(operation.get("totals"), "operation totals")
    computed_totals = {
        stage: {counter: 0 for counter in OPERATION_COUNTERS}
        for stage in OPERATION_STAGES
    }
    del operation_body

    sources = [
        _source(run / name, run, f"run/{name}")
        for name in EXPORT_FILES
    ]
    export_topic_receipts = export.get("topic_receipts")
    if not isinstance(export_topic_receipts, list) or len(export_topic_receipts) != len(topics):
        raise ResultBundleIntegrityError("retrieval export topic receipts changed")
    for index, topic_id in enumerate(topics):
        topic_root = run / topic_id
        operation_receipt_path = (
            topic_root / "cache-operation-receipt.cached-upstream-rescore.json"
        )
        receipt_body, raw_receipt = _strict_file(
            operation_receipt_path, "cache operation topic receipt"
        )
        receipt = _fields(
            raw_receipt,
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
            "cache operation topic receipt",
        )
        _validate_receipt_digest(
            receipt_body, receipt, "cache operation topic receipt"
        )
        projection_sha = _digest(
            projection_digests[index], "operation projection digest"
        )
        if (
            receipt.get("schema_version") != "cache-operation-receipt-v1"
            or receipt.get("mode") != "cached-upstream-rescore"
            or receipt.get("run_id") != run_id
            or receipt.get("topic_id") != topic_id
            or receipt.get("config_sha256") != config_sha
            or receipt.get("projection_manifest_sha256") != projection_sha
            or sha256(receipt_body).hexdigest()
            != _digest(receipt_digests[index], "operation receipt digest")
        ):
            raise ResultBundleIntegrityError("cache operation topic identity changed")
        topic_stages = _validate_stages(
            receipt.get("stages"), "topic operation stages"
        )
        _validate_phases(receipt.get("phases"), "topic operation phases")
        for stage_name in OPERATION_STAGES:
            for counter_name in OPERATION_COUNTERS:
                computed_totals[stage_name][counter_name] += topic_stages[stage_name][
                    counter_name
                ]
        source_receipt = export_topic_receipts[index]
        if (
            not isinstance(source_receipt, dict)
            or source_receipt.get("topic_id") != topic_id
            or source_receipt.get("projection_manifest_sha256") != projection_sha
        ):
            raise ResultBundleIntegrityError("retrieval export topic order changed")
        job_path = topic_root / "topic-job-receipt.cached-upstream-rescore.json"
        job_body, raw_job = _strict_file(job_path, "topic dispatch receipt")
        job = _fields(
            raw_job,
            {
                "schema_version",
                "run_id",
                "topic_id",
                "config_sha256",
                "mode",
                "projection_manifest_sha256",
                "status",
                "stopping_reason",
            },
            "topic dispatch receipt",
        )
        if (
            job.get("schema_version") != "topic-job-receipt-v4"
            or job.get("run_id") != run_id
            or job.get("topic_id") != topic_id
            or job.get("config_sha256") != config_sha
            or job.get("mode") != "cached-upstream-rescore"
            or job.get("projection_manifest_sha256") != projection_sha
            or job.get("status") != "complete"
            or _canonical(job) != job_body
        ):
            raise ResultBundleIntegrityError("topic dispatch receipt identity changed")
        decomposition_result = topic_root / "decomposition/result.json"
        decomposition_manifest_path = topic_root / "decomposition/manifest.json"
        decomposition_body = decomposition_result.read_bytes()
        _, decomposition_manifest = _strict_file(
            decomposition_manifest_path, "decomposition manifest"
        )
        if (
            decomposition_manifest.get("result_file") != "result.json"
            or decomposition_manifest.get("result_bytes") != len(decomposition_body)
            or decomposition_manifest.get("result_sha256")
            != sha256(decomposition_body).hexdigest()
        ):
            raise ResultBundleIntegrityError("decomposition receipt changed")
        sources.extend(
            (
                _source(
                    operation_receipt_path,
                    run,
                    f"run/{topic_id}/{operation_receipt_path.name}",
                ),
                _source(job_path, run, f"run/{topic_id}/{job_path.name}"),
                _source(
                    decomposition_result,
                    run,
                    f"run/{topic_id}/decomposition/result.json",
                ),
                _source(
                    decomposition_manifest_path,
                    run,
                    f"run/{topic_id}/decomposition/manifest.json",
                ),
            )
        )
        for phase in ("retrieval", "scoring", "canonical"):
            sources.extend(_phase_sources(topic_root, run, topic_id, phase))
        projection_path = (
            topic_root / "canonical/retrieval-projection-manifest.json"
        )
        if _file_receipt(projection_path)[1] != projection_sha:
            raise ResultBundleIntegrityError(
                "operation projection digest differs from canonical checkpoint"
            )

    if computed_totals != totals:
        raise ResultBundleIntegrityError("cache operation totals changed")

    candidate_handoff_sha = sha256(handoff_path.read_bytes()).hexdigest()
    sources.extend(
        _comparison_sources(
            validation,
            run,
            topics,
            candidate_handoff_sha,
            run_id,
            config_sha,
            selected_workers,
        )
    )
    coverage_root = validation / "retrieval_nugget_coverage_v2"
    if {item.name for item in coverage_root.iterdir()} != set(topics):
        raise ResultBundleIntegrityError("candidate coverage topic set changed")
    for topic_id in topics:
        work = coverage_root / topic_id
        try:
            load_completed_coverage_evaluation(
                handoff_manifest_path=handoff_path,
                topic_id=topic_id,
                work_dir=work,
            )
        except Exception as exc:
            raise ResultBundleIntegrityError(
                f"candidate coverage is invalid for topic {topic_id}"
            ) from exc
        if {item.name for item in work.iterdir()} != set(COVERAGE_FILES):
            raise ResultBundleIntegrityError("candidate coverage files changed")
        for name in COVERAGE_FILES:
            sources.append(
                _source(
                    work / name,
                    validation,
                    f"validation/retrieval_nugget_coverage_v2/{topic_id}/{name}",
                )
            )
    return run_id, revision, topics, sources


def _verify_result_semantics(verified: VerifiedResultBundle, root: Path) -> None:
    run_id, revision, topics, sources = _validated_result_sources(
        root / "run", root / "validation"
    )
    if (
        run_id != verified.run_id
        or revision != verified.git_revision
        or topics != verified.topic_ids
        or {source.path for source in sources}
        != {member.path for member in verified.members}
    ):
        raise ResultBundleIntegrityError("result bundle member closure changed")


def _optional_source_root(path: Path, label: str) -> Path | None:
    if not path.exists():
        return None
    return _source_root(path, label)


def _canonical_object_file(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    body, value = _strict_file(path, label)
    if _canonical(value) != body:
        raise ResultBundleIntegrityError(f"{label} encoding changed")
    return body, value


def _ordered_topic_subset(
    value: object,
    requested_topics: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(topic, str) or not topic for topic in value)
        or len(set(value)) != len(value)
    ):
        raise ResultBundleIntegrityError(f"{label} topic set changed")
    topics = tuple(value)
    selected = set(topics)
    if tuple(topic for topic in requested_topics if topic in selected) != topics:
        raise ResultBundleIntegrityError(f"{label} topic order changed")
    return topics


def _diagnostic_comparison_sources(
    validation_root: Path,
    requested_topics: tuple[str, ...],
    candidate_handoff_sha256: str | None,
) -> list[_SourceMember]:
    sources: list[_SourceMember] = []
    for relative_root in (Path(), Path("canary")):
        comparison_root = validation_root / relative_root
        for kind in ("structural", "semantic"):
            comparison_path = comparison_root / f"{kind}-comparison.json"
            manifest_path = comparison_root / f"{kind}-comparison-manifest.json"
            present = (comparison_path.exists(), manifest_path.exists())
            if not any(present):
                continue
            if not all(present):
                # An interruption can land between the create-only comparison
                # and its hash-binding manifest. Preserve other completed
                # evidence, but never admit the unsealed half-pair.
                continue
            comparison_body, comparison = _canonical_object_file(
                comparison_path, f"diagnostic {kind} comparison"
            )
            _, manifest = _canonical_object_file(
                manifest_path, f"diagnostic {kind} comparison manifest"
            )
            topics = _ordered_topic_subset(
                comparison.get("topic_ids"),
                requested_topics,
                f"diagnostic {kind} comparison",
            )
            gates_passed = comparison.get("gates_passed")
            comparison_candidate_sha = _digest(
                comparison.get("candidate_handoff_sha256"),
                f"diagnostic {kind} candidate handoff",
            )
            if (
                comparison.get("schema_version")
                != f"cached-segmentation-{kind}-comparison-v1"
                or type(gates_passed) is not bool
                or (
                    relative_root == Path()
                    and candidate_handoff_sha256 is not None
                    and comparison_candidate_sha != candidate_handoff_sha256
                )
                or manifest.get("schema_version")
                != f"cached-segmentation-{kind}-manifest-v1"
                or manifest.get("comparison_file") != comparison_path.name
                or manifest.get("comparison_bytes") != len(comparison_body)
                or manifest.get("comparison_sha256")
                != sha256(comparison_body).hexdigest()
                or manifest.get("topic_ids") != list(topics)
                or manifest.get("gates_passed") is not gates_passed
            ):
                raise ResultBundleIntegrityError(
                    f"diagnostic {kind} comparison identity changed"
                )
            prefix = "validation"
            if relative_root.parts:
                prefix += f"/{relative_root.as_posix()}"
            sources.extend(
                (
                    _source(
                        comparison_path,
                        validation_root,
                        f"{prefix}/{comparison_path.name}",
                    ),
                    _source(
                        manifest_path,
                        validation_root,
                        f"{prefix}/{manifest_path.name}",
                    ),
                )
            )
    return sources


def _diagnostic_evidence_sources(
    run_root: Path,
    validation_root: Path,
    *,
    run_id: str,
    revision: str,
    requested_topics: tuple[str, ...],
) -> list[_SourceMember]:
    sources: list[_SourceMember] = []
    run = _optional_source_root(Path(run_root), "diagnostic run root")
    validation = _optional_source_root(
        Path(validation_root), "diagnostic validation root"
    )
    candidate_handoff_path: Path | None = None
    candidate_handoff_sha256: str | None = None
    candidate_topics: tuple[str, ...] = ()

    if run is not None:
        handoff_path = run / "generation_handoff_manifest.json"
        if handoff_path.exists():
            try:
                handoff = load_generation_handoff(handoff_path)
            except Exception as exc:
                raise ResultBundleIntegrityError(
                    "diagnostic generation handoff is invalid"
                ) from exc
            candidate_topics = tuple(topic.topic_id for topic in handoff.topics)
            _ordered_topic_subset(
                list(candidate_topics), requested_topics, "diagnostic generation handoff"
            )
            if (
                handoff.producer.retrieval_run_id != run_id
                or handoff.producer.producer_revision != revision
            ):
                raise ResultBundleIntegrityError(
                    "diagnostic generation handoff identity changed"
                )
            candidate_handoff_path = handoff_path
            candidate_handoff_sha256 = _file_receipt(handoff_path)[1]
            sources.append(
                _source(
                    handoff_path,
                    run,
                    "run/generation_handoff_manifest.json",
                )
            )

        for name in ("retrieval_export_manifest.json", "cache-operation-manifest.json"):
            path = run / name
            if not path.exists():
                continue
            body, value = _canonical_object_file(path, f"diagnostic {name}")
            if value.get("run_id") != run_id:
                raise ResultBundleIntegrityError(
                    f"diagnostic {name} run identity changed"
                )
            if name == "cache-operation-manifest.json":
                _validate_receipt_digest(body, value, "diagnostic operation manifest")
            sources.append(_source(path, run, f"run/{name}"))

        for topic_id in requested_topics:
            topic_root = run / topic_id
            if not topic_root.exists():
                continue
            if topic_root.is_symlink() or not topic_root.is_dir():
                raise ResultBundleIntegrityError(
                    "diagnostic topic root is missing or unsafe"
                )
            for name in (
                "cache-operation-receipt.cached-upstream-rescore.json",
                "topic-job-receipt.cached-upstream-rescore.json",
            ):
                path = topic_root / name
                if not path.exists():
                    continue
                body, value = _canonical_object_file(
                    path, f"diagnostic {topic_id} {name}"
                )
                if value.get("run_id") != run_id or value.get("topic_id") != topic_id:
                    raise ResultBundleIntegrityError(
                        "diagnostic topic receipt identity changed"
                    )
                if name.startswith("cache-operation-receipt"):
                    _validate_receipt_digest(
                        body, value, "diagnostic topic operation receipt"
                    )
                sources.append(_source(path, run, f"run/{topic_id}/{name}"))

            decomposition_result = topic_root / "decomposition/result.json"
            decomposition_manifest = topic_root / "decomposition/manifest.json"
            decomposition_present = (
                decomposition_result.exists(),
                decomposition_manifest.exists(),
            )
            if all(decomposition_present):
                result_body = decomposition_result.read_bytes()
                _, manifest = _canonical_object_file(
                    decomposition_manifest, "diagnostic decomposition manifest"
                )
                if (
                    manifest.get("result_file") != "result.json"
                    or manifest.get("result_bytes") != len(result_body)
                    or manifest.get("result_sha256")
                    != sha256(result_body).hexdigest()
                ):
                    raise ResultBundleIntegrityError(
                        "diagnostic decomposition receipt changed"
                    )
                sources.extend(
                    (
                        _source(
                            decomposition_result,
                            run,
                            f"run/{topic_id}/decomposition/result.json",
                        ),
                        _source(
                            decomposition_manifest,
                            run,
                            f"run/{topic_id}/decomposition/manifest.json",
                        ),
                    )
                )
            for phase in ("retrieval", "scoring", "canonical"):
                if (topic_root / phase / "complete.json").exists():
                    sources.extend(_phase_sources(topic_root, run, topic_id, phase))

    if validation is not None:
        sources.extend(
            _diagnostic_comparison_sources(
                validation,
                requested_topics,
                candidate_handoff_sha256,
            )
        )
        decision_path = validation / "concurrency-decision.json"
        if decision_path.exists():
            _, decision = _canonical_object_file(
                decision_path, "diagnostic concurrency decision"
            )
            if (
                decision.get("schema_version")
                != "cached-segmentation-concurrency-decision-v3"
                or decision.get("run_id") != run_id
            ):
                raise ResultBundleIntegrityError(
                    "diagnostic concurrency decision identity changed"
                )
            sources.append(
                _source(
                    decision_path,
                    validation,
                    "validation/concurrency-decision.json",
                )
            )

        coverage_root = validation / "retrieval_nugget_coverage_v2"
        if coverage_root.exists():
            if candidate_handoff_path is None:
                raise ResultBundleIntegrityError(
                    "diagnostic coverage has no candidate handoff"
                )
            coverage = _source_root(coverage_root, "diagnostic coverage root")
            for topic_id in candidate_topics:
                work = coverage / topic_id
                if not work.exists() or {item.name for item in work.iterdir()} != set(
                    COVERAGE_FILES
                ):
                    continue
                try:
                    load_completed_coverage_evaluation(
                        handoff_manifest_path=candidate_handoff_path,
                        topic_id=topic_id,
                        work_dir=work,
                    )
                except Exception as exc:
                    raise ResultBundleIntegrityError(
                        f"diagnostic coverage is invalid for topic {topic_id}"
                    ) from exc
                for name in COVERAGE_FILES:
                    sources.append(
                        _source(
                            work / name,
                            validation,
                            f"validation/retrieval_nugget_coverage_v2/{topic_id}/{name}",
                        )
                    )
    return sources


def _verify_diagnostic_semantics(
    verified: VerifiedResultBundle,
    root: Path,
) -> None:
    status_body, status = _canonical_object_file(
        root / DIAGNOSTIC_STATUS_NAME, "diagnostic status"
    )
    expected_fields = {
        "schema_version",
        "promotion_eligible",
        "run_id",
        "git_revision",
        "requested_topic_ids",
        "stage",
        "exit_status",
        "evidence_members",
    }
    evidence_members = status.get("evidence_members")
    if (
        set(status) != expected_fields
        or status.get("schema_version") != DIAGNOSTIC_SCHEMA
        or status.get("promotion_eligible") is not False
        or status.get("run_id") != verified.run_id
        or status.get("git_revision") != verified.git_revision
        or status.get("requested_topic_ids") != list(verified.topic_ids)
        or not isinstance(status.get("stage"), str)
        or not status["stage"]
        or type(status.get("exit_status")) is not int
        or not 1 <= status["exit_status"] <= 255
        or not isinstance(evidence_members, list)
        or any(not isinstance(path, str) for path in evidence_members)
        or evidence_members != sorted(evidence_members)
        or len(set(evidence_members)) != len(evidence_members)
        or _canonical(status) != status_body
    ):
        raise ResultBundleIntegrityError("diagnostic status identity changed")
    evidence = _diagnostic_evidence_sources(
        root / "run",
        root / "validation",
        run_id=verified.run_id,
        revision=verified.git_revision,
        requested_topics=verified.topic_ids,
    )
    actual_evidence = sorted(source.path for source in evidence)
    expected_members = {DIAGNOSTIC_STATUS_NAME, *actual_evidence}
    if evidence_members != actual_evidence or {
        member.path for member in verified.members
    } != expected_members:
        raise ResultBundleIntegrityError("diagnostic bundle member closure changed")


def pack_diagnostic_bundle(
    run_root: str | Path,
    validation_root: str | Path,
    destination: str | Path,
    *,
    run_id: str,
    revision: str,
    topic_ids: Sequence[str],
    stage: str,
    exit_status: int,
) -> VerifiedResultBundle:
    """Preserve authenticated partial evidence without making it promotable."""

    topics = tuple(topic_ids)
    if type(exit_status) is not int or not 1 <= exit_status <= 255:
        raise ValueError("diagnostic bundles require a failure exit status")
    raw_run_root = Path(run_root)
    raw_validation_root = Path(validation_root)
    if not raw_run_root.is_absolute() or not raw_validation_root.is_absolute():
        raise ValueError("diagnostic source roots must be absolute")
    if (
        not run_id
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not topics
        or len(set(topics)) != len(topics)
        or any(not topic for topic in topics)
        or not stage
        or len(stage) > 80
        or any(not (character.isalnum() or character in "_-") for character in stage)
    ):
        raise ValueError("diagnostic identity is invalid")
    evidence = _diagnostic_evidence_sources(
        raw_run_root,
        raw_validation_root,
        run_id=run_id,
        revision=revision,
        requested_topics=topics,
    )
    evidence_paths = sorted(source.path for source in evidence)
    with tempfile.TemporaryDirectory(prefix="segmentation-diagnostic-status-") as temp:
        status_path = Path(temp) / DIAGNOSTIC_STATUS_NAME
        status_path.write_bytes(
            _canonical(
                {
                    "schema_version": DIAGNOSTIC_SCHEMA,
                    "promotion_eligible": False,
                    "run_id": run_id,
                    "git_revision": revision,
                    "requested_topic_ids": list(topics),
                    "stage": stage,
                    "exit_status": exit_status,
                    "evidence_members": evidence_paths,
                }
            )
        )
        sources = [
            _source(status_path, Path(temp), DIAGNOSTIC_STATUS_NAME),
            *evidence,
        ]
        return _pack(
            Path(destination),
            kind="diagnostic",
            run_id=run_id,
            revision=revision,
            topics=topics,
            sources=sources,
        )


def verify_diagnostic_bundle(bundle_dir: str | Path) -> VerifiedResultBundle:
    return _verify(Path(bundle_dir), expected_kind="diagnostic")


def pack_result_bundle(
    run_root: str | Path,
    validation_root: str | Path,
    destination: str | Path,
) -> VerifiedResultBundle:
    run_id, revision, topics, sources = _validated_result_sources(
        Path(run_root), Path(validation_root)
    )
    return _pack(
        Path(destination),
        kind="result",
        run_id=run_id,
        revision=revision,
        topics=topics,
        sources=sources,
    )


def verify_result_bundle(bundle_dir: str | Path) -> VerifiedResultBundle:
    return _verify(Path(bundle_dir), expected_kind="result")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack_baseline = commands.add_parser("pack-baseline")
    pack_baseline.add_argument("--handoff", type=Path, required=True)
    pack_baseline.add_argument("--coverage-root", type=Path, required=True)
    pack_baseline.add_argument("--destination", type=Path, required=True)
    verify_baseline = commands.add_parser("verify-baseline")
    verify_baseline.add_argument("bundle", type=Path)
    restore_baseline = commands.add_parser("restore-baseline")
    restore_baseline.add_argument("bundle", type=Path)
    restore_baseline.add_argument("--destination", type=Path, required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--run-root", type=Path, required=True)
    pack.add_argument("--validation-root", type=Path, required=True)
    pack.add_argument("--destination", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    pack_diagnostic = commands.add_parser("pack-diagnostic")
    pack_diagnostic.add_argument("--run-root", type=Path, required=True)
    pack_diagnostic.add_argument("--validation-root", type=Path, required=True)
    pack_diagnostic.add_argument("--destination", type=Path, required=True)
    pack_diagnostic.add_argument("--run-id", required=True)
    pack_diagnostic.add_argument("--revision", required=True)
    pack_diagnostic.add_argument("--topic", action="append", required=True)
    pack_diagnostic.add_argument("--stage", required=True)
    pack_diagnostic.add_argument("--exit-status", type=int, required=True)
    verify_diagnostic = commands.add_parser("verify-diagnostic")
    verify_diagnostic.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    if args.command == "pack-baseline":
        pack_baseline_bundle(args.handoff, args.coverage_root, args.destination)
    elif args.command == "verify-baseline":
        verify_baseline_bundle(args.bundle)
    elif args.command == "restore-baseline":
        restore_baseline_bundle(args.bundle, args.destination)
    elif args.command == "pack":
        pack_result_bundle(args.run_root, args.validation_root, args.destination)
    elif args.command == "verify":
        verify_result_bundle(args.bundle)
    elif args.command == "pack-diagnostic":
        pack_diagnostic_bundle(
            args.run_root,
            args.validation_root,
            args.destination,
            run_id=args.run_id,
            revision=args.revision,
            topic_ids=args.topic,
            stage=args.stage,
            exit_status=args.exit_status,
        )
    else:
        verify_diagnostic_bundle(args.bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
