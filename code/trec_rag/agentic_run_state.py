"""Authenticated run plans and resumable per-topic agentic state.

The run plan freezes the original cohort and every semantic input before live
work starts. A topic becomes resumable only after its three payloads and a
manifest-last success seal validate byte-for-byte. Failed attempts remain
private and never count as completed topics.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Iterator

from .agentic_generation_export import (
    AgenticTopicProjection,
    serialize_agentic_retrieval_topic,
)
from .generation_handoff import (
    deserialize_generation_topic,
    serialize_generation_topic,
)
from .topic_records import TopicRecordsReceipt
from .topics import Topic


RUN_PLAN_FILENAME = "run_plan.json"
TOPIC_SEAL_FILENAME = "topic_projection_manifest.json"
RUN_PLAN_SCHEMA = "agentic_run_plan_v1"
TOPIC_SEAL_SCHEMA = "agentic_topic_projection_manifest_v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SAFE_OUTCOME = re.compile(r"[a-z][a-z0-9_]*\Z")
_SUCCESS_OUTCOMES = frozenset(
    {"coordinator_selected", "deterministic_grounded_recovery"}
)
_ARTIFACT_NAMES = (
    "retrieval_topic.json",
    "generation_topic.json",
    "topic_records_receipt.json",
)
_MAX_STATE_BYTES = 512 * 1024 * 1024


class AgenticRunStateError(ValueError):
    """Run state is missing, incompatible, corrupt, or publication-conflicted."""


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
        raise AgenticRunStateError(f"state is not canonical JSON: {exc}") from exc


def _canonical_line(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _digest(body: bytes) -> str:
    return sha256(body).hexdigest()


def _text_digest(text: str) -> str:
    return _digest(text.encode("utf-8"))


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AgenticRunStateError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_revision(value: object, label: str) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise AgenticRunStateError(f"{label} must be a lowercase source revision")
    return value


def _require_safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise AgenticRunStateError(f"{label} must be a safe identifier")
    return value


def _exact_mapping(
    value: object, fields: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise AgenticRunStateError(f"{label} must be an object")
    if set(value) != fields:
        raise AgenticRunStateError(f"{label} fields changed")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AgenticRunStateError(f"{label} must be a positive integer")
    return value


def _read_regular(path: Path, *, label: str) -> bytes:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise AgenticRunStateError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise AgenticRunStateError(f"{label} must be a regular file")
    if info.st_size > _MAX_STATE_BYTES:
        raise AgenticRunStateError(f"{label} is too large")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise AgenticRunStateError(f"unable to read {label}: {path}") from exc


def _parse_json_line(body: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgenticRunStateError(f"{label} is not valid canonical JSON") from exc
    if not isinstance(value, Mapping):
        raise AgenticRunStateError(f"{label} must be a JSON object")
    if body != _canonical_line(value):
        raise AgenticRunStateError(f"{label} is not canonical")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        info = path.lstat()
    except OSError as exc:  # pragma: no cover
        raise AgenticRunStateError(
            f"unable to inspect state directory: {path}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise AgenticRunStateError(f"state directory is unsafe: {path}")
    os.chmod(path, 0o700)


def _publish_identical(path: Path, body: bytes, *, label: str) -> None:
    """Create one file atomically, accepting only byte-identical recovery."""

    _ensure_private_directory(path.parent)
    if path.exists() or path.is_symlink():
        existing = _read_regular(path, label=label)
        if existing != body:
            raise AgenticRunStateError(f"{label} publication conflict: {path}")
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
            existing = _read_regular(path, label=label)
            if existing != body:
                raise AgenticRunStateError(f"{label} publication conflict: {path}")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    _ensure_private_directory(path.parent)
    with path.open("a+b") as lock:
        os.chmod(path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class SubmoduleRevision:
    path: str
    revision: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or not self.path
            or Path(self.path).is_absolute()
            or ".." in Path(self.path).parts
        ):
            raise AgenticRunStateError("submodule path must be safe relative text")
        _require_revision(self.revision, "submodule revision")

    def to_payload(self) -> dict[str, str]:
        return {"path": self.path, "revision": self.revision}


@dataclass(frozen=True)
class PlannedTopic:
    topic_id: str
    narrative_sha256: str

    def __post_init__(self) -> None:
        _require_safe_id(self.topic_id, "topic id")
        _require_digest(self.narrative_sha256, "topic narrative")

    def to_payload(self) -> dict[str, str]:
        return {
            "topic_id": self.topic_id,
            "narrative_sha256": self.narrative_sha256,
        }


@dataclass(frozen=True)
class AgenticRunPlan:
    run_id: str
    config_bytes: bytes
    config_sha256: str
    topics_source_sha256: str
    topics: tuple[PlannedTopic, ...]
    source_revision: str
    submodule_revisions: tuple[SubmoduleRevision, ...]
    plan_sha256: str

    @property
    def planned_topic_ids(self) -> tuple[str, ...]:
        return tuple(topic.topic_id for topic in self.topics)

    def _payload_without_digest(self) -> dict[str, object]:
        return {
            "schema_version": RUN_PLAN_SCHEMA,
            "run_id": self.run_id,
            "config": {
                "base64": base64.b64encode(self.config_bytes).decode("ascii"),
                "sha256": self.config_sha256,
            },
            "topics": {
                "source_sha256": self.topics_source_sha256,
                "planned": [topic.to_payload() for topic in self.topics],
            },
            "source": {
                "revision": self.source_revision,
                "submodules": [
                    row.to_payload() for row in self.submodule_revisions
                ],
            },
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_digest(),
            "plan_sha256": self.plan_sha256,
        }


def _build_run_plan(
    *,
    run_id: str,
    config_bytes: bytes,
    topics: Sequence[Topic],
    official_topics_sha256: str,
    source_revision: str,
    submodule_revisions: Sequence[SubmoduleRevision],
) -> AgenticRunPlan:
    validated_run_id = _require_safe_id(run_id, "run id")
    if not isinstance(config_bytes, bytes) or not config_bytes:
        raise AgenticRunStateError("config bytes must be non-empty bytes")
    if (
        isinstance(topics, (str, bytes))
        or not isinstance(topics, Sequence)
        or not topics
    ):
        raise AgenticRunStateError("run plan requires at least one topic")
    planned: list[PlannedTopic] = []
    seen_topics: set[str] = set()
    for topic in topics:
        if not isinstance(topic, Topic) or not topic.narrative:
            raise AgenticRunStateError(
                "run plan topics must contain exact narratives"
            )
        topic_id = _require_safe_id(topic.id, "topic id")
        if topic_id in seen_topics:
            raise AgenticRunStateError(
                f"run plan topics repeat topic id: {topic_id}"
            )
        seen_topics.add(topic_id)
        planned.append(PlannedTopic(topic_id, _text_digest(topic.narrative)))
    source_digest = _require_digest(official_topics_sha256, "topic source")
    revision = _require_revision(source_revision, "source revision")
    submodules = tuple(submodule_revisions)
    if any(not isinstance(row, SubmoduleRevision) for row in submodules):
        raise AgenticRunStateError(
            "submodule revisions must be typed records"
        )
    paths = tuple(row.path for row in submodules)
    if len(set(paths)) != len(paths):
        raise AgenticRunStateError("submodule revisions repeat a path")
    provisional = AgenticRunPlan(
        run_id=validated_run_id,
        config_bytes=config_bytes,
        config_sha256=_digest(config_bytes),
        topics_source_sha256=source_digest,
        topics=tuple(planned),
        source_revision=revision,
        submodule_revisions=submodules,
        plan_sha256="0" * 64,
    )
    return AgenticRunPlan(
        run_id=provisional.run_id,
        config_bytes=provisional.config_bytes,
        config_sha256=provisional.config_sha256,
        topics_source_sha256=provisional.topics_source_sha256,
        topics=provisional.topics,
        source_revision=provisional.source_revision,
        submodule_revisions=provisional.submodule_revisions,
        plan_sha256=_digest(
            _canonical(provisional._payload_without_digest())
        ),
    )


def serialize_run_plan(plan: AgenticRunPlan) -> bytes:
    if not isinstance(plan, AgenticRunPlan):
        raise AgenticRunStateError("plan must be AgenticRunPlan")
    return _canonical_line(plan.to_payload())


def _plans_match_or_raise(
    stored: AgenticRunPlan, expected: AgenticRunPlan
) -> None:
    if stored.run_id != expected.run_id:
        raise AgenticRunStateError("run id differs from stored run plan")
    if (
        stored.config_bytes != expected.config_bytes
        or stored.config_sha256 != expected.config_sha256
    ):
        raise AgenticRunStateError("config bytes differ from stored run plan")
    if stored.planned_topic_ids != expected.planned_topic_ids:
        raise AgenticRunStateError("topic cohort differs from stored run plan")
    for stored_topic, expected_topic in zip(
        stored.topics, expected.topics, strict=True
    ):
        if stored_topic.narrative_sha256 != expected_topic.narrative_sha256:
            raise AgenticRunStateError(
                "topic narrative differs from stored run plan: "
                f"{stored_topic.topic_id}"
            )
    if stored.topics_source_sha256 != expected.topics_source_sha256:
        raise AgenticRunStateError(
            "topic source differs from stored run plan"
        )
    if stored.source_revision != expected.source_revision:
        raise AgenticRunStateError(
            "source revision differs from stored run plan"
        )
    if stored.submodule_revisions != expected.submodule_revisions:
        raise AgenticRunStateError(
            "submodule revisions differ from stored run plan"
        )


def create_run_plan(
    *,
    work_dir: Path,
    run_id: str,
    config_bytes: bytes,
    topics: Sequence[Topic],
    official_topics_sha256: str,
    source_revision: str,
    submodule_revisions: Sequence[SubmoduleRevision],
    allow_existing_identical: bool = False,
) -> AgenticRunPlan:
    work = Path(work_dir)
    expected = _build_run_plan(
        run_id=run_id,
        config_bytes=config_bytes,
        topics=topics,
        official_topics_sha256=official_topics_sha256,
        source_revision=source_revision,
        submodule_revisions=submodule_revisions,
    )
    plan_path = work / RUN_PLAN_FILENAME
    if plan_path.exists() or plan_path.is_symlink():
        if not allow_existing_identical:
            raise AgenticRunStateError(
                f"run plan already exists: {plan_path}"
            )
        stored = load_run_plan(work)
        _plans_match_or_raise(stored, expected)
        return stored
    if work.exists() and any(work.iterdir()):
        raise AgenticRunStateError(
            "run state namespace already exists without a plan"
        )
    _ensure_private_directory(work)
    _publish_identical(
        plan_path, serialize_run_plan(expected), label="run plan"
    )
    return expected


def load_run_plan(work_dir: Path) -> AgenticRunPlan:
    path = Path(work_dir) / RUN_PLAN_FILENAME
    body = _read_regular(path, label="run plan")
    payload = _parse_json_line(body, label="run plan")
    row = _exact_mapping(
        payload,
        {
            "schema_version",
            "run_id",
            "config",
            "topics",
            "source",
            "plan_sha256",
        },
        "run plan",
    )
    if row["schema_version"] != RUN_PLAN_SCHEMA:
        raise AgenticRunStateError("run plan schema changed")
    received_digest = _require_digest(
        row["plan_sha256"], "run plan digest"
    )
    without_digest = {
        key: value for key, value in row.items() if key != "plan_sha256"
    }
    if _digest(_canonical(without_digest)) != received_digest:
        raise AgenticRunStateError(
            "run plan digest does not match its contents"
        )
    config = _exact_mapping(
        row["config"], {"base64", "sha256"}, "run plan config"
    )
    if not isinstance(config["base64"], str):
        raise AgenticRunStateError(
            "run plan config base64 must be text"
        )
    try:
        config_bytes = base64.b64decode(
            config["base64"], validate=True
        )
    except (ValueError, base64.binascii.Error) as exc:
        raise AgenticRunStateError(
            "run plan config base64 is invalid"
        ) from exc
    config_digest = _require_digest(
        config["sha256"], "run plan config"
    )
    if not config_bytes or _digest(config_bytes) != config_digest:
        raise AgenticRunStateError(
            "run plan config bytes do not match their digest"
        )
    topic_data = _exact_mapping(
        row["topics"],
        {"source_sha256", "planned"},
        "run plan topics",
    )
    source_digest = _require_digest(
        topic_data["source_sha256"], "run plan topic source"
    )
    raw_topics = topic_data["planned"]
    if not isinstance(raw_topics, list) or not raw_topics:
        raise AgenticRunStateError(
            "run plan requires at least one planned topic"
        )
    topics: list[PlannedTopic] = []
    for raw_topic in raw_topics:
        item = _exact_mapping(
            raw_topic,
            {"topic_id", "narrative_sha256"},
            "planned topic",
        )
        topics.append(
            PlannedTopic(
                _require_safe_id(item["topic_id"], "topic id"),
                _require_digest(
                    item["narrative_sha256"], "topic narrative"
                ),
            )
        )
    if len({item.topic_id for item in topics}) != len(topics):
        raise AgenticRunStateError(
            "run plan topics repeat an identity"
        )
    source = _exact_mapping(
        row["source"], {"revision", "submodules"}, "run plan source"
    )
    revision = _require_revision(
        source["revision"], "source revision"
    )
    raw_submodules = source["submodules"]
    if not isinstance(raw_submodules, list):
        raise AgenticRunStateError(
            "run plan submodules must be an array"
        )
    submodules: list[SubmoduleRevision] = []
    for raw_submodule in raw_submodules:
        item = _exact_mapping(
            raw_submodule, {"path", "revision"}, "submodule"
        )
        path_value = item["path"]
        revision_value = item["revision"]
        if not isinstance(path_value, str) or not isinstance(
            revision_value, str
        ):
            raise AgenticRunStateError(
                "submodule identity must be text"
            )
        submodules.append(
            SubmoduleRevision(path_value, revision_value)
        )
    if len({item.path for item in submodules}) != len(submodules):
        raise AgenticRunStateError(
            "run plan submodule paths repeat"
        )
    plan = AgenticRunPlan(
        run_id=_require_safe_id(row["run_id"], "run id"),
        config_bytes=config_bytes,
        config_sha256=config_digest,
        topics_source_sha256=source_digest,
        topics=tuple(topics),
        source_revision=revision,
        submodule_revisions=tuple(submodules),
        plan_sha256=received_digest,
    )
    if serialize_run_plan(plan) != body:
        raise AgenticRunStateError(
            "run plan canonical bytes changed"
        )
    return plan


def resume_run_plan(
    *,
    work_dir: Path,
    run_id: str,
    config_bytes: bytes,
    topics: Sequence[Topic],
    official_topics_sha256: str,
    source_revision: str,
    submodule_revisions: Sequence[SubmoduleRevision],
) -> AgenticRunPlan:
    stored = load_run_plan(work_dir)
    expected = _build_run_plan(
        run_id=run_id,
        config_bytes=config_bytes,
        topics=topics,
        official_topics_sha256=official_topics_sha256,
        source_revision=source_revision,
        submodule_revisions=submodule_revisions,
    )
    _plans_match_or_raise(stored, expected)
    return stored


def _validate_plan_argument(
    work_dir: Path, plan: AgenticRunPlan
) -> None:
    if (
        not isinstance(plan, AgenticRunPlan)
        or load_run_plan(work_dir) != plan
    ):
        raise AgenticRunStateError(
            "run plan argument differs from authenticated state"
        )


def _planned_topic(
    plan: AgenticRunPlan, topic_id: str
) -> PlannedTopic:
    _require_safe_id(topic_id, "topic id")
    matches = tuple(
        topic for topic in plan.topics if topic.topic_id == topic_id
    )
    if len(matches) != 1:
        raise AgenticRunStateError(
            f"topic is outside the run plan: {topic_id}"
        )
    return matches[0]


@dataclass(frozen=True)
class TopicAttempt:
    topic_id: str
    attempt_number: int
    path: Path

    def write_artifact(self, name: str, body: bytes) -> Path:
        if (
            not isinstance(name, str)
            or _SAFE_ID.fullmatch(name) is None
        ):
            raise AgenticRunStateError(
                "attempt artifact name must be safe"
            )
        if not isinstance(body, bytes):
            raise AgenticRunStateError(
                "attempt artifact body must be bytes"
            )
        destination = self.path / name
        _publish_identical(
            destination, body, label="attempt artifact"
        )
        return destination


def _topic_dir(work_dir: Path, topic_id: str) -> Path:
    return Path(work_dir) / "topics" / topic_id


def allocate_topic_attempt(
    *, work_dir: Path, plan: AgenticRunPlan, topic_id: str
) -> TopicAttempt:
    _validate_plan_argument(work_dir, plan)
    _planned_topic(plan, topic_id)
    topic_dir = _topic_dir(work_dir, topic_id)
    _ensure_private_directory(topic_dir)
    with _state_lock(topic_dir / ".topic-state.lock"):
        if (topic_dir / TOPIC_SEAL_FILENAME).exists():
            load_topic_seal(
                work_dir=work_dir, plan=plan, topic_id=topic_id
            )
            raise AgenticRunStateError(
                f"topic is already sealed: {topic_id}"
            )
        attempts_dir = topic_dir / "attempts"
        _ensure_private_directory(attempts_dir)
        numbers = [
            int(path.name)
            for path in attempts_dir.iterdir()
            if path.is_dir()
            and re.fullmatch(r"[0-9]{6}", path.name)
        ]
        number = max(numbers, default=0) + 1
        path = attempts_dir / f"{number:06d}"
        try:
            path.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise AgenticRunStateError(
                "topic attempt allocation raced"
            ) from exc
        _fsync_directory(attempts_dir)
    return TopicAttempt(topic_id, number, path)


@dataclass(frozen=True)
class SealedArtifact:
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
class ValidatedTopicSeal:
    topic_id: str
    narrative_sha256: str
    run_plan_sha256: str
    attempt_number: int
    status: str
    stopping_reason: str
    synthesis_outcome: str
    retrieval_topic_sha256: str
    full_text_record_sha256: str
    artifacts: tuple[SealedArtifact, ...]
    seal_sha256: str
    retrieval_topic_bytes: bytes
    generation_topic_bytes: bytes
    records_receipt_bytes: bytes

    @property
    def generation_topic(self):
        return deserialize_generation_topic(
            self.generation_topic_bytes
        )

    @property
    def retrieval_payload(self) -> Mapping[str, object]:
        return _parse_json_line(
            self.retrieval_topic_bytes,
            label="sealed retrieval topic",
        )

    @property
    def full_text_record(self) -> Mapping[str, object]:
        value = self.retrieval_payload.get("full_text_record")
        if not isinstance(value, Mapping):
            raise AgenticRunStateError(
                "sealed retrieval topic lacks full-text record"
            )
        return value


def topic_records_receipt_payload(
    receipt: TopicRecordsReceipt,
) -> dict[str, object]:
    if not isinstance(receipt, TopicRecordsReceipt):
        raise AgenticRunStateError(
            "receipt must be TopicRecordsReceipt"
        )
    return {
        "database_sha256": receipt.database_sha256,
        "database_bytes": receipt.database_bytes,
        "semantic_sha256": receipt.semantic_sha256,
        "topic_id": receipt.topic_id,
        "run_id": receipt.run_id,
        "document_sha256s": list(receipt.document_sha256s),
        "row_counts": dict(sorted(receipt.row_counts.items())),
        "schema_version": receipt.schema_version,
        "manifest_sha256": receipt.manifest_sha256,
        "manifest_bytes": receipt.manifest_bytes,
    }


def _receipt_mapping(
    value: Mapping[str, object] | TopicRecordsReceipt,
) -> dict[str, object]:
    if isinstance(value, TopicRecordsReceipt):
        return topic_records_receipt_payload(value)
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise AgenticRunStateError(
            "topic records receipt must be an object"
        )
    canonical = _canonical(dict(value))
    parsed = json.loads(canonical.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise AgenticRunStateError(
            "topic records receipt must be an object"
        )
    return parsed


def _artifact_rows(
    payloads: Mapping[str, bytes],
) -> tuple[SealedArtifact, ...]:
    return tuple(
        SealedArtifact(
            name=name,
            bytes=len(payloads[name]),
            sha256=_digest(payloads[name]),
        )
        for name in _ARTIFACT_NAMES
    )


def _seal_payload(
    *,
    plan: AgenticRunPlan,
    topic: PlannedTopic,
    attempt_number: int,
    status: str,
    stopping_reason: str,
    synthesis_outcome: str,
    retrieval_topic_sha256: str,
    full_text_record_sha256: str,
    artifacts: tuple[SealedArtifact, ...],
) -> dict[str, object]:
    without_digest: dict[str, object] = {
        "schema_version": TOPIC_SEAL_SCHEMA,
        "run_plan_sha256": plan.plan_sha256,
        "topic": topic.to_payload(),
        "attempt_number": attempt_number,
        "status": status,
        "stopping_reason": stopping_reason,
        "synthesis_outcome": synthesis_outcome,
        "retrieval_topic_sha256": retrieval_topic_sha256,
        "full_text_record_sha256": full_text_record_sha256,
        "artifacts": [row.to_payload() for row in artifacts],
    }
    return {
        **without_digest,
        "seal_sha256": _digest(_canonical(without_digest)),
    }


def seal_topic_success(
    *,
    work_dir: Path,
    plan: AgenticRunPlan,
    projection: AgenticTopicProjection,
    attempt: TopicAttempt,
    status: str,
    stopping_reason: str,
    synthesis_outcome: str,
    records_receipt: Mapping[str, object] | TopicRecordsReceipt,
) -> ValidatedTopicSeal:
    _validate_plan_argument(work_dir, plan)
    if not isinstance(projection, AgenticTopicProjection):
        raise AgenticRunStateError(
            "projection must be AgenticTopicProjection"
        )
    topic = _planned_topic(plan, projection.topic_id)
    if _text_digest(projection.narrative) != topic.narrative_sha256:
        raise AgenticRunStateError(
            "projection narrative differs from the run plan"
        )
    if status != "complete":
        raise AgenticRunStateError(
            "only complete topics can be sealed"
        )
    if not isinstance(stopping_reason, str) or not stopping_reason:
        raise AgenticRunStateError(
            "stopping_reason must be non-empty text"
        )
    if (
        not isinstance(synthesis_outcome, str)
        or _SAFE_OUTCOME.fullmatch(synthesis_outcome) is None
        or synthesis_outcome not in _SUCCESS_OUTCOMES
    ):
        raise AgenticRunStateError(
            "synthesis_outcome is not a successful outcome"
        )
    if (
        not isinstance(attempt, TopicAttempt)
        or attempt.topic_id != topic.topic_id
    ):
        raise AgenticRunStateError(
            "topic attempt identity differs from projection"
        )
    expected_attempt_path = (
        _topic_dir(work_dir, topic.topic_id)
        / "attempts"
        / f"{attempt.attempt_number:06d}"
    )
    if (
        attempt.path != expected_attempt_path
        or not attempt.path.is_dir()
    ):
        raise AgenticRunStateError(
            "topic attempt path is not authenticated"
        )

    retrieval_bytes = serialize_agentic_retrieval_topic(projection)
    if _digest(retrieval_bytes) != projection.retrieval_topic_sha256:
        raise AgenticRunStateError(
            "retrieval topic digest differs from projection"
        )
    generation_bytes = serialize_generation_topic(
        projection.generation_topic
    )
    generation_topic = deserialize_generation_topic(generation_bytes)
    if (
        generation_topic.topic_id != topic.topic_id
        or generation_topic.narrative_sha256
        != topic.narrative_sha256
        or generation_topic.source_receipts.retrieval_topic_sha256
        != projection.retrieval_topic_sha256
        or generation_topic.source_receipts.official_topics_sha256
        != plan.topics_source_sha256
    ):
        raise AgenticRunStateError(
            "generation topic identity differs from run plan"
        )
    receipt_payload = _receipt_mapping(records_receipt)
    if receipt_payload.get("topic_id") != topic.topic_id:
        raise AgenticRunStateError(
            "topic records receipt topic differs from projection"
        )
    if receipt_payload.get("run_id") != plan.run_id:
        raise AgenticRunStateError(
            "topic records receipt run id differs from plan"
        )
    receipt_bytes = _canonical_line(receipt_payload)
    payloads = {
        "retrieval_topic.json": retrieval_bytes,
        "generation_topic.json": generation_bytes,
        "topic_records_receipt.json": receipt_bytes,
    }
    artifacts = _artifact_rows(payloads)
    full_text_digest = _digest(
        _canonical(projection.full_text_record)
    )
    manifest_payload = _seal_payload(
        plan=plan,
        topic=topic,
        attempt_number=attempt.attempt_number,
        status=status,
        stopping_reason=stopping_reason,
        synthesis_outcome=synthesis_outcome,
        retrieval_topic_sha256=projection.retrieval_topic_sha256,
        full_text_record_sha256=full_text_digest,
        artifacts=artifacts,
    )
    manifest_bytes = _canonical_line(manifest_payload)
    topic_dir = _topic_dir(work_dir, topic.topic_id)
    _ensure_private_directory(topic_dir)
    with _state_lock(topic_dir / ".topic-state.lock"):
        manifest_path = topic_dir / TOPIC_SEAL_FILENAME
        if manifest_path.exists() or manifest_path.is_symlink():
            existing = _read_regular(
                manifest_path, label="topic seal"
            )
            if existing != manifest_bytes:
                raise AgenticRunStateError(
                    "topic seal publication conflict"
                )
            for name, artifact_body in payloads.items():
                if (
                    _read_regular(
                        topic_dir / name,
                        label="topic seal artifact",
                    )
                    != artifact_body
                ):
                    raise AgenticRunStateError(
                        "topic seal artifact publication conflict"
                    )
        else:
            for name, artifact_body in payloads.items():
                _publish_identical(
                    topic_dir / name,
                    artifact_body,
                    label="topic seal artifact",
                )
            _publish_identical(
                manifest_path,
                manifest_bytes,
                label="topic seal",
            )
    return load_topic_seal(
        work_dir=work_dir, plan=plan, topic_id=topic.topic_id
    )


def load_topic_seal(
    *, work_dir: Path, plan: AgenticRunPlan, topic_id: str
) -> ValidatedTopicSeal:
    _validate_plan_argument(work_dir, plan)
    planned = _planned_topic(plan, topic_id)
    topic_dir = _topic_dir(work_dir, topic_id)
    manifest_path = topic_dir / TOPIC_SEAL_FILENAME
    if (
        not manifest_path.exists()
        and not manifest_path.is_symlink()
    ):
        raise AgenticRunStateError(
            f"topic is not sealed: {topic_id}"
        )
    body = _read_regular(manifest_path, label="topic seal")
    payload = _parse_json_line(body, label="topic seal")
    row = _exact_mapping(
        payload,
        {
            "schema_version",
            "run_plan_sha256",
            "topic",
            "attempt_number",
            "status",
            "stopping_reason",
            "synthesis_outcome",
            "retrieval_topic_sha256",
            "full_text_record_sha256",
            "artifacts",
            "seal_sha256",
        },
        "topic seal",
    )
    if row["schema_version"] != TOPIC_SEAL_SCHEMA:
        raise AgenticRunStateError("topic seal schema changed")
    received_seal = _require_digest(
        row["seal_sha256"], "topic seal digest"
    )
    without_digest = {
        key: value for key, value in row.items()
        if key != "seal_sha256"
    }
    if _digest(_canonical(without_digest)) != received_seal:
        raise AgenticRunStateError(
            "topic seal digest does not match its contents"
        )
    if row["run_plan_sha256"] != plan.plan_sha256:
        raise AgenticRunStateError(
            "topic seal belongs to another run plan"
        )
    topic_row = _exact_mapping(
        row["topic"],
        {"topic_id", "narrative_sha256"},
        "sealed topic",
    )
    if (
        topic_row["topic_id"] != planned.topic_id
        or topic_row["narrative_sha256"]
        != planned.narrative_sha256
    ):
        raise AgenticRunStateError(
            "topic seal topic identity differs from run plan"
        )
    attempt_number = _positive_int(
        row["attempt_number"], "topic attempt number"
    )
    attempt_dir = (
        _topic_dir(work_dir, topic_id)
        / "attempts"
        / f"{attempt_number:06d}"
    )
    if not attempt_dir.is_dir():
        raise AgenticRunStateError(
            "topic seal attempt is missing"
        )
    if row["status"] != "complete":
        raise AgenticRunStateError(
            "topic seal status is not complete"
        )
    stopping_reason = row["stopping_reason"]
    outcome = row["synthesis_outcome"]
    if not isinstance(stopping_reason, str) or not stopping_reason:
        raise AgenticRunStateError(
            "topic seal stopping_reason is invalid"
        )
    if (
        not isinstance(outcome, str)
        or outcome not in _SUCCESS_OUTCOMES
    ):
        raise AgenticRunStateError(
            "topic seal synthesis_outcome is invalid"
        )
    retrieval_digest = _require_digest(
        row["retrieval_topic_sha256"], "retrieval topic"
    )
    full_text_digest = _require_digest(
        row["full_text_record_sha256"], "full text record"
    )
    raw_artifacts = row["artifacts"]
    if (
        not isinstance(raw_artifacts, list)
        or len(raw_artifacts) != len(_ARTIFACT_NAMES)
    ):
        raise AgenticRunStateError(
            "topic seal artifacts changed"
        )
    artifacts: list[SealedArtifact] = []
    payload_bytes: dict[str, bytes] = {}
    for expected_name, raw_artifact in zip(
        _ARTIFACT_NAMES, raw_artifacts, strict=True
    ):
        item = _exact_mapping(
            raw_artifact,
            {"name", "bytes", "sha256"},
            "sealed artifact",
        )
        if item["name"] != expected_name:
            raise AgenticRunStateError(
                "topic seal artifact order or name changed"
            )
        size = _positive_int(
            item["bytes"], "sealed artifact bytes"
        )
        artifact_digest = _require_digest(
            item["sha256"], "sealed artifact"
        )
        artifact_body = _read_regular(
            topic_dir / expected_name,
            label="sealed artifact",
        )
        if len(artifact_body) != size:
            raise AgenticRunStateError(
                f"sealed artifact bytes changed: {expected_name}"
            )
        if _digest(artifact_body) != artifact_digest:
            raise AgenticRunStateError(
                f"sealed artifact sha256 changed: {expected_name}"
            )
        artifacts.append(
            SealedArtifact(expected_name, size, artifact_digest)
        )
        payload_bytes[expected_name] = artifact_body
    retrieval_bytes = payload_bytes["retrieval_topic.json"]
    if _digest(retrieval_bytes) != retrieval_digest:
        raise AgenticRunStateError(
            "sealed retrieval topic sha256 changed"
        )
    retrieval_payload = _parse_json_line(
        retrieval_bytes, label="sealed retrieval topic"
    )
    if retrieval_payload.get("topic_id") != topic_id:
        raise AgenticRunStateError(
            "sealed retrieval topic identity changed"
        )
    full_text_record = retrieval_payload.get(
        "full_text_record"
    )
    if (
        not isinstance(full_text_record, Mapping)
        or _digest(_canonical(full_text_record))
        != full_text_digest
    ):
        raise AgenticRunStateError(
            "sealed full-text record sha256 changed"
        )
    generation_bytes = payload_bytes["generation_topic.json"]
    try:
        generation = deserialize_generation_topic(
            generation_bytes
        )
    except Exception as exc:
        raise AgenticRunStateError(
            "sealed generation topic is invalid"
        ) from exc
    if (
        generation.topic_id != topic_id
        or generation.narrative_sha256
        != planned.narrative_sha256
        or generation.source_receipts.retrieval_topic_sha256
        != retrieval_digest
        or generation.source_receipts.official_topics_sha256
        != plan.topics_source_sha256
    ):
        raise AgenticRunStateError(
            "sealed generation topic identity changed"
        )
    records_bytes = payload_bytes[
        "topic_records_receipt.json"
    ]
    records = _parse_json_line(
        records_bytes, label="topic records receipt"
    )
    if (
        records.get("topic_id") != topic_id
        or records.get("run_id") != plan.run_id
    ):
        raise AgenticRunStateError(
            "topic records receipt identity changed"
        )
    return ValidatedTopicSeal(
        topic_id=topic_id,
        narrative_sha256=planned.narrative_sha256,
        run_plan_sha256=plan.plan_sha256,
        attempt_number=attempt_number,
        status="complete",
        stopping_reason=stopping_reason,
        synthesis_outcome=outcome,
        retrieval_topic_sha256=retrieval_digest,
        full_text_record_sha256=full_text_digest,
        artifacts=tuple(artifacts),
        seal_sha256=received_seal,
        retrieval_topic_bytes=retrieval_bytes,
        generation_topic_bytes=generation_bytes,
        records_receipt_bytes=records_bytes,
    )


@dataclass(frozen=True)
class RunTopicSelection:
    planned_topic_ids: tuple[str, ...]
    completed_topic_ids: tuple[str, ...]
    execute_topic_ids: tuple[str, ...]


def select_run_topics(
    *,
    work_dir: Path,
    plan: AgenticRunPlan,
    topic_ids: Sequence[str] | None = None,
) -> RunTopicSelection:
    _validate_plan_argument(work_dir, plan)
    if topic_ids is None:
        requested = plan.planned_topic_ids
    else:
        if (
            isinstance(topic_ids, str)
            or not isinstance(topic_ids, Sequence)
            or not topic_ids
        ):
            raise AgenticRunStateError(
                "topic selection requires at least one topic"
            )
        requested = tuple(topic_ids)
        if len(set(requested)) != len(requested):
            raise AgenticRunStateError(
                "topic selection repeats an identity"
            )
        for topic_id in requested:
            _planned_topic(plan, topic_id)
    completed: list[str] = []
    for topic_id in plan.planned_topic_ids:
        manifest = (
            _topic_dir(work_dir, topic_id)
            / TOPIC_SEAL_FILENAME
        )
        if manifest.exists() or manifest.is_symlink():
            load_topic_seal(
                work_dir=work_dir,
                plan=plan,
                topic_id=topic_id,
            )
            completed.append(topic_id)
    completed_set = set(completed)
    execute = tuple(
        topic_id
        for topic_id in requested
        if topic_id not in completed_set
    )
    return RunTopicSelection(
        plan.planned_topic_ids,
        tuple(completed),
        execute,
    )


def load_sealed_topics(
    *, work_dir: Path, plan: AgenticRunPlan
) -> tuple[ValidatedTopicSeal, ...]:
    _validate_plan_argument(work_dir, plan)
    return tuple(
        load_topic_seal(
            work_dir=work_dir,
            plan=plan,
            topic_id=topic_id,
        )
        for topic_id in plan.planned_topic_ids
    )


__all__ = [
    "RUN_PLAN_FILENAME",
    "TOPIC_SEAL_FILENAME",
    "AgenticRunPlan",
    "AgenticRunStateError",
    "RunTopicSelection",
    "SealedArtifact",
    "SubmoduleRevision",
    "TopicAttempt",
    "ValidatedTopicSeal",
    "allocate_topic_attempt",
    "create_run_plan",
    "load_run_plan",
    "load_sealed_topics",
    "load_topic_seal",
    "resume_run_plan",
    "seal_topic_success",
    "select_run_topics",
    "serialize_run_plan",
    "topic_records_receipt_payload",
]
