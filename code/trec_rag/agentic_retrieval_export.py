"""Manifest-last aggregate publication for completed agentic retrieval runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Callable, Iterator
import zipfile

from .agentic_generation_export import AGENTIC_RETRIEVAL_TOPIC_SCHEMA
from .agentic_run_state import (
    AgenticRunPlan,
    AgenticRunStateError,
    ValidatedTopicSeal,
    load_run_plan,
    load_sealed_topics,
)
from .generation_handoff import GenerationTopic
from .generation_handoff_export import prepare_generation_handoff_artifact


RETRIEVAL_RUN_FILENAME = "r_output_trec_rag_2026.tsv"
RETRIEVAL_WITH_TEXT_FILENAME = "retrieval_with_text.jsonl.zip"
GENERATION_HANDOFF_FILENAME = "generation_handoff_manifest.json"
EXPORT_MANIFEST_FILENAME = "retrieval_export_manifest.json"
EXPORT_SCHEMA = "agentic_retrieval_export_manifest_v1"
_PAYLOAD_NAMES = (
    RETRIEVAL_RUN_FILENAME,
    RETRIEVAL_WITH_TEXT_FILENAME,
    GENERATION_HANDOFF_FILENAME,
)
_MAX_EXPORT_BYTES = 2 * 1024 * 1024 * 1024
_LOWERCASE_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")

_PUBLICATION_TEST_HOOK: Callable[[str], None] | None = None


class AgenticRetrievalExportError(ValueError):
    """The complete run cannot be published or its root receipt is invalid."""


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
        raise AgenticRetrievalExportError(
            f"retrieval export is not canonical JSON: {exc}"
        ) from exc


def _canonical_line(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _digest(body: bytes) -> str:
    return sha256(body).hexdigest()


def _exact_mapping(
    value: object, fields: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise AgenticRetrievalExportError(f"{label} must be an object")
    if set(value) != fields:
        raise AgenticRetrievalExportError(f"{label} fields changed")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AgenticRetrievalExportError(f"{label} must be a positive integer")
    return value


def _read_regular(path: Path, *, label: str) -> bytes:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise AgenticRetrievalExportError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise AgenticRetrievalExportError(f"{label} must be a regular file")
    if info.st_size > _MAX_EXPORT_BYTES:
        raise AgenticRetrievalExportError(f"{label} is too large")
    return path.read_bytes()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise AgenticRetrievalExportError(f"export directory is unsafe: {path}")
    os.chmod(path, 0o700)


def _publish_identical(path: Path, body: bytes) -> None:
    if path.exists() or path.is_symlink():
        if _read_regular(path, label="root export payload") != body:
            raise AgenticRetrievalExportError(
                f"root export publication conflict: {path.name}"
            )
        os.chmod(path, 0o600)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if _read_regular(path, label="root export payload") != body:
                raise AgenticRetrievalExportError(
                    f"root export publication conflict: {path.name}"
                )
            os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _export_lock(output_dir: Path) -> Iterator[None]:
    _ensure_private_directory(output_dir)
    path = output_dir / ".agentic-retrieval-export.lock"
    with path.open("a+b") as lock:
        os.chmod(path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _authenticated_seals(
    *, work_dir: Path, plan: AgenticRunPlan
) -> tuple[ValidatedTopicSeal, ...]:
    try:
        if load_run_plan(work_dir) != plan:
            raise AgenticRunStateError(
                "run plan argument differs from authenticated state"
            )
        return load_sealed_topics(work_dir=work_dir, plan=plan)
    except AgenticRunStateError as exc:
        raise AgenticRetrievalExportError(str(exc)) from exc


@dataclass(frozen=True)
class _PreparedTopic:
    topic_id: str
    trec_rows: tuple[tuple[str, str, str, int, int, str], ...]
    full_text_record: Mapping[str, object]
    generation_topic: GenerationTopic
    seal: ValidatedTopicSeal
    document_count: int


def _prepare_topic(
    *, plan: AgenticRunPlan, seal: ValidatedTopicSeal
) -> _PreparedTopic:
    payload = seal.retrieval_payload
    row = _exact_mapping(
        payload,
        {
            "schema_version",
            "topic_id",
            "retrieval_rows",
            "full_text_record",
        },
        "sealed retrieval topic",
    )
    if (
        row["schema_version"] != AGENTIC_RETRIEVAL_TOPIC_SCHEMA
        or row["topic_id"] != seal.topic_id
    ):
        raise AgenticRetrievalExportError(
            "sealed retrieval topic identity changed"
        )
    raw_rows = row["retrieval_rows"]
    full_text = row["full_text_record"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise AgenticRetrievalExportError(
            "sealed retrieval topic has no rows"
        )
    full_text_row = _exact_mapping(
        full_text, {"query", "candidates"}, "full-text record"
    )
    query = _exact_mapping(
        full_text_row["query"],
        {"qid", "selection_id", "text", "text_sha256"},
        "full-text query",
    )
    if (
        query["qid"] != seal.topic_id
        or query["selection_id"] != "official"
        or not isinstance(query["text"], str)
        or _digest(query["text"].encode("utf-8")) != seal.narrative_sha256
        or query["text_sha256"] != seal.narrative_sha256
    ):
        raise AgenticRetrievalExportError(
            "full-text query differs from the planned topic"
        )
    candidates = full_text_row["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(raw_rows):
        raise AgenticRetrievalExportError(
            "retrieval rows and full-text candidates differ"
        )
    parsed_rows: list[tuple[str, str, str, int, int, str]] = []
    seen_docids: set[str] = set()
    previous_score: int | None = None
    for expected_rank, (raw_row, raw_candidate) in enumerate(
        zip(raw_rows, candidates, strict=True), start=1
    ):
        retrieval = _exact_mapping(
            raw_row,
            {"topic_id", "q0", "docid", "rank", "score"},
            "retrieval row",
        )
        candidate = _exact_mapping(
            raw_candidate,
            {"docid", "doc", "rank", "score", "lane_ids", "text_sha256"},
            "full-text candidate",
        )
        docid = retrieval["docid"]
        rank = retrieval["rank"]
        score = retrieval["score"]
        text = candidate["doc"]
        if (
            retrieval["topic_id"] != seal.topic_id
            or retrieval["q0"] != "Q0"
            or not isinstance(docid, str)
            or not docid
            or any(character.isspace() for character in docid)
            or docid in seen_docids
            or type(rank) is not int
            or rank != expected_rank
            or type(score) is not int
            or score != len(raw_rows) - expected_rank + 1
            or (previous_score is not None and score >= previous_score)
            or candidate["docid"] != docid
            or candidate["rank"] != rank
            or candidate["score"] != score
            or candidate["lane_ids"] != ["agentic"]
            or not isinstance(text, str)
            or not text
            or candidate["text_sha256"] != _digest(text.encode("utf-8"))
        ):
            raise AgenticRetrievalExportError(
                f"retrieval/full-text closure changed for {seal.topic_id}"
            )
        seen_docids.add(docid)
        previous_score = score
        parsed_rows.append(
            (seal.topic_id, "Q0", docid, rank, score, plan.run_id)
        )
    generation = seal.generation_topic
    if (
        generation.topic_id != seal.topic_id
        or generation.narrative_sha256 != seal.narrative_sha256
        or not set(generation.citation_docids) <= seen_docids
    ):
        raise AgenticRetrievalExportError(
            "generation citations escape the retrieval projection"
        )
    return _PreparedTopic(
        topic_id=seal.topic_id,
        trec_rows=tuple(parsed_rows),
        full_text_record=full_text_row,
        generation_topic=generation,
        seal=seal,
        document_count=len(parsed_rows),
    )


def _trec_bytes(topics: Sequence[_PreparedTopic]) -> bytes:
    return "".join(
        f"{topic_id} {q0} {docid} {rank} {score} {run_id}\n"
        for topic in topics
        for topic_id, q0, docid, rank, score, run_id in topic.trec_rows
    ).encode("utf-8")


def _jsonl_bytes(topics: Sequence[_PreparedTopic]) -> bytes:
    return b"".join(
        _canonical_line(topic.full_text_record) for topic in topics
    )


def _deterministic_zip(member_body: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        member = zipfile.ZipInfo(
            "retrieval_with_text.jsonl",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        member.compress_type = zipfile.ZIP_DEFLATED
        member.create_system = 3
        member.external_attr = 0o100600 << 16
        archive.writestr(member, member_body)
    return buffer.getvalue()


@dataclass(frozen=True)
class ExportArtifact:
    name: str
    bytes: int
    sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class AgenticRetrievalExportReceipt:
    output_dir: Path
    run_file: Path
    with_text_archive: Path
    generation_handoff: Path
    manifest: Path
    manifest_sha256: str
    topic_ids: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedExport:
    payloads: Mapping[str, bytes]
    manifest_body: bytes
    manifest_sha256: str
    topic_ids: tuple[str, ...]


def _prepare_export(
    *,
    output_dir: Path,
    plan: AgenticRunPlan,
    seals: Sequence[ValidatedTopicSeal],
    producer_revision: str,
) -> _PreparedExport:
    if (
        not isinstance(producer_revision, str)
        or _LOWERCASE_GIT_REVISION.fullmatch(producer_revision) is None
        or producer_revision != plan.source_revision
    ):
        raise AgenticRetrievalExportError(
            "producer_revision must exactly match the frozen lowercase "
            "40-character source revision"
        )
    if tuple(seal.topic_id for seal in seals) != plan.planned_topic_ids:
        raise AgenticRetrievalExportError(
            "sealed topics do not match the original run plan order"
        )
    topics = tuple(
        _prepare_topic(plan=plan, seal=seal) for seal in seals
    )
    trec_body = _trec_bytes(topics)
    zip_body = _deterministic_zip(_jsonl_bytes(topics))
    handoff = prepare_generation_handoff_artifact(
        output_dir=output_dir,
        retrieval_run_id=plan.run_id,
        producer_revision=producer_revision,
        topics=tuple(topic.generation_topic for topic in topics),
    )
    if handoff.path.name != GENERATION_HANDOFF_FILENAME:
        raise AgenticRetrievalExportError(
            "generation handoff filename changed"
        )
    payloads: dict[str, bytes] = {
        RETRIEVAL_RUN_FILENAME: trec_body,
        RETRIEVAL_WITH_TEXT_FILENAME: zip_body,
        GENERATION_HANDOFF_FILENAME: handoff.body,
    }
    artifacts = tuple(
        ExportArtifact(name, len(payloads[name]), _digest(payloads[name]))
        for name in _PAYLOAD_NAMES
    )
    without_digest: dict[str, object] = {
        "schema_version": EXPORT_SCHEMA,
        "run_id": plan.run_id,
        "run_plan_sha256": plan.plan_sha256,
        "producer_revision": producer_revision,
        "topic_count": len(topics),
        "planned_topic_ids": list(plan.planned_topic_ids),
        "topics": [
            {
                "topic_id": topic.topic_id,
                "status": topic.seal.status,
                "stopping_reason": topic.seal.stopping_reason,
                "synthesis_outcome": topic.seal.synthesis_outcome,
                "topic_seal_sha256": topic.seal.seal_sha256,
                "retrieval_topic_sha256": (
                    topic.seal.retrieval_topic_sha256
                ),
                "generation_context_sha256": (
                    topic.generation_topic.context_sha256
                ),
                "document_count": topic.document_count,
                "citation_document_count": len(
                    topic.generation_topic.citation_docids
                ),
            }
            for topic in topics
        ],
        "artifacts": [artifact.to_payload() for artifact in artifacts],
    }
    manifest_sha256 = _digest(_canonical(without_digest))
    manifest_body = _canonical_line(
        {**without_digest, "manifest_sha256": manifest_sha256}
    )
    return _PreparedExport(
        payloads=payloads,
        manifest_body=manifest_body,
        manifest_sha256=manifest_sha256,
        topic_ids=plan.planned_topic_ids,
    )


def _receipt(
    output_dir: Path, prepared: _PreparedExport
) -> AgenticRetrievalExportReceipt:
    return AgenticRetrievalExportReceipt(
        output_dir=output_dir,
        run_file=output_dir / RETRIEVAL_RUN_FILENAME,
        with_text_archive=output_dir / RETRIEVAL_WITH_TEXT_FILENAME,
        generation_handoff=output_dir / GENERATION_HANDOFF_FILENAME,
        manifest=output_dir / EXPORT_MANIFEST_FILENAME,
        manifest_sha256=prepared.manifest_sha256,
        topic_ids=prepared.topic_ids,
    )


def _preflight_existing(
    output_dir: Path, prepared: _PreparedExport
) -> None:
    manifest = output_dir / EXPORT_MANIFEST_FILENAME
    manifest_exists = manifest.exists() or manifest.is_symlink()
    for name, expected in prepared.payloads.items():
        path = output_dir / name
        exists = path.exists() or path.is_symlink()
        if exists and _read_regular(
            path, label="root export payload"
        ) != expected:
            raise AgenticRetrievalExportError(
                f"root export publication conflict: {name}"
            )
        if manifest_exists and not exists:
            raise AgenticRetrievalExportError(
                f"published export manifest has missing payload: {name}"
            )
    if manifest_exists and _read_regular(
        manifest, label="retrieval export manifest"
    ) != prepared.manifest_body:
        raise AgenticRetrievalExportError(
            "retrieval export manifest publication conflict"
        )


def publish_agentic_retrieval_export(
    *,
    output_dir: Path,
    work_dir: Path,
    plan: AgenticRunPlan,
    producer_revision: str,
) -> AgenticRetrievalExportReceipt:
    """Publish the complete original cohort, with the outer receipt last."""

    destination = Path(output_dir)
    seals = _authenticated_seals(work_dir=work_dir, plan=plan)
    prepared = _prepare_export(
        output_dir=destination,
        plan=plan,
        seals=seals,
        producer_revision=producer_revision,
    )
    _ensure_private_directory(destination)
    _preflight_existing(destination, prepared)
    with _export_lock(destination):
        _preflight_existing(destination, prepared)
        for name in _PAYLOAD_NAMES:
            _publish_identical(destination / name, prepared.payloads[name])
            if _PUBLICATION_TEST_HOOK is not None:
                _PUBLICATION_TEST_HOOK(name)
        _publish_identical(
            destination / EXPORT_MANIFEST_FILENAME,
            prepared.manifest_body,
        )
    return load_agentic_retrieval_export(
        output_dir=destination,
        work_dir=work_dir,
        plan=plan,
    )


def load_agentic_retrieval_export(
    *,
    output_dir: Path,
    work_dir: Path,
    plan: AgenticRunPlan,
) -> AgenticRetrievalExportReceipt:
    """Authenticate an already published complete root export."""

    destination = Path(output_dir)
    manifest_path = destination / EXPORT_MANIFEST_FILENAME
    manifest_body = _read_regular(
        manifest_path, label="retrieval export manifest"
    )
    try:
        payload = json.loads(manifest_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgenticRetrievalExportError(
            "retrieval export manifest is invalid JSON"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or manifest_body != _canonical_line(payload)
    ):
        raise AgenticRetrievalExportError(
            "retrieval export manifest is not canonical"
        )
    row = _exact_mapping(
        payload,
        {
            "schema_version",
            "run_id",
            "run_plan_sha256",
            "producer_revision",
            "topic_count",
            "planned_topic_ids",
            "topics",
            "artifacts",
            "manifest_sha256",
        },
        "retrieval export manifest",
    )
    if row["schema_version"] != EXPORT_SCHEMA:
        raise AgenticRetrievalExportError(
            "retrieval export manifest schema changed"
        )
    received_digest = row["manifest_sha256"]
    if not isinstance(received_digest, str):
        raise AgenticRetrievalExportError(
            "retrieval export manifest digest is invalid"
        )
    without_digest = {
        key: value
        for key, value in row.items()
        if key != "manifest_sha256"
    }
    if _digest(_canonical(without_digest)) != received_digest:
        raise AgenticRetrievalExportError(
            "retrieval export manifest digest changed"
        )
    if (
        row["run_id"] != plan.run_id
        or row["run_plan_sha256"] != plan.plan_sha256
        or row["planned_topic_ids"] != list(plan.planned_topic_ids)
        or row["topic_count"] != len(plan.planned_topic_ids)
        or row["producer_revision"] != plan.source_revision
    ):
        raise AgenticRetrievalExportError(
            "retrieval export manifest differs from its run plan"
        )
    seals = _authenticated_seals(work_dir=work_dir, plan=plan)
    prepared = _prepare_export(
        output_dir=destination,
        plan=plan,
        seals=seals,
        producer_revision=row["producer_revision"],
    )
    if prepared.manifest_body != manifest_body:
        raise AgenticRetrievalExportError(
            "retrieval export manifest differs from authenticated topics"
        )
    for name, expected in prepared.payloads.items():
        if _read_regular(
            destination / name, label="root export payload"
        ) != expected:
            raise AgenticRetrievalExportError(
                f"root export payload changed: {name}"
            )
    return _receipt(destination, prepared)


__all__ = [
    "EXPORT_MANIFEST_FILENAME",
    "GENERATION_HANDOFF_FILENAME",
    "RETRIEVAL_RUN_FILENAME",
    "RETRIEVAL_WITH_TEXT_FILENAME",
    "AgenticRetrievalExportError",
    "AgenticRetrievalExportReceipt",
    "load_agentic_retrieval_export",
    "publish_agentic_retrieval_export",
]
