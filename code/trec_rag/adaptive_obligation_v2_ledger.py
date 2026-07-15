"""Crash-safe append-only attempt ledger for adaptive-obligation v2 inference."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Mapping, Sequence


LEDGER_SCHEMA_VERSION = "adaptive-obligation-v2-attempt-ledger-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROOT_NAMES = frozenset(
    {"events.jsonl", "head.json", "anchor.json", "mutation.lock", "raw"}
)
_ANCHOR_SCHEMA_VERSION = "adaptive-obligation-v2-run-anchor-v1"
_COMPLETION_SCHEMA_VERSION = "adaptive-obligation-v2-run-completion-v1"
_PROPOSAL_SCHEMA_SHA256 = (
    "dc09881657e08a49f98757505fb725534b13c579596f550056f1df889ac634dd"
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())


def _claim_directory_no_symlinks(path: Path) -> None:
    target = Path(path)
    if not target.name or target.name in (".", ".."):
        raise OSError("output directory name is unsafe")
    parent = target.parent
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/" if parent.is_absolute() else ".", flags)
    try:
        for component in parent.parts:
            if component in (parent.anchor, "", "."):
                continue
            if component == "..":
                raise OSError("output parent traversal is unsafe")
            child = os.open(component, flags | nofollow, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        os.mkdir(target.name, 0o700, dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_fsynced(path: Path, content: bytes) -> None:
    descriptor, raw_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class AttemptSpec:
    stage: str
    job_id: str
    attempt_ordinal: int
    request_sha256: str
    max_new_tokens: int

    def __post_init__(self) -> None:
        if self.stage != "proposal":
            raise ValueError("attempt stage must be proposal")
        if not _SHA256.fullmatch(self.job_id):
            raise ValueError("attempt job_id must be a lowercase SHA-256")
        if self.attempt_ordinal not in (1, 2) or isinstance(self.attempt_ordinal, bool):
            raise ValueError("attempt ordinal must be exactly 1 or 2")
        if not _SHA256.fullmatch(self.request_sha256):
            raise ValueError("attempt request_sha256 must be a lowercase SHA-256")
        if self.max_new_tokens not in (256, 512) or isinstance(
            self.max_new_tokens, bool
        ):
            raise ValueError("attempt output ceiling must be exactly 256 or 512")
        if self.max_new_tokens != {1: 256, 2: 512}[self.attempt_ordinal]:
            raise ValueError("attempt ordinal and output ceiling differ")


class _JSONPrefixState(Enum):
    COMPLETE = auto()
    INCOMPLETE = auto()
    INVALID = auto()


class _IncompleteJSONPrefix(Exception):
    pass


class _InvalidJSONPrefix(Exception):
    pass


class _JSONPrefixParser:
    """Classify one UTF-8 JSON byte prefix without repairing earlier syntax."""

    _WHITESPACE = frozenset(b" \t\r\n")
    _HEX = frozenset(b"0123456789abcdefABCDEF")
    _SIMPLE_ESCAPES = frozenset(b'"\\/bfnrt')

    def __init__(self, source: bytes) -> None:
        self.source = source
        self.length = len(source)

    def classify(self) -> _JSONPrefixState:
        try:
            position = self._parse_value(self._skip_whitespace(0))
            position = self._skip_whitespace(position)
        except _IncompleteJSONPrefix:
            return _JSONPrefixState.INCOMPLETE
        except _InvalidJSONPrefix:
            return _JSONPrefixState.INVALID
        return (
            _JSONPrefixState.COMPLETE
            if position == self.length
            else _JSONPrefixState.INVALID
        )

    def _skip_whitespace(self, position: int) -> int:
        while position < self.length and self.source[position] in self._WHITESPACE:
            position += 1
        return position

    def _parse_value(self, position: int) -> int:
        if position >= self.length:
            raise _IncompleteJSONPrefix
        token = self.source[position]
        if token == ord('"'):
            return self._parse_string(position)
        if token == ord("{"):
            return self._parse_object(position)
        if token == ord("["):
            return self._parse_array(position)
        if token == ord("t"):
            return self._parse_literal(position, b"true")
        if token == ord("f"):
            return self._parse_literal(position, b"false")
        if token == ord("n"):
            return self._parse_literal(position, b"null")
        if token == ord("-") or ord("0") <= token <= ord("9"):
            return self._parse_number(position)
        raise _InvalidJSONPrefix

    def _parse_literal(self, position: int, literal: bytes) -> int:
        available = self.source[position : position + len(literal)]
        if len(available) < len(literal):
            if literal.startswith(available):
                raise _IncompleteJSONPrefix
            raise _InvalidJSONPrefix
        if available != literal:
            raise _InvalidJSONPrefix
        return position + len(literal)

    def _parse_string(self, position: int) -> int:
        position += 1
        while position < self.length:
            token = self.source[position]
            if token == ord('"'):
                return position + 1
            if token < 0x20:
                raise _InvalidJSONPrefix
            if token == ord("\\"):
                position += 1
                if position >= self.length:
                    raise _IncompleteJSONPrefix
                escaped = self.source[position]
                if escaped in self._SIMPLE_ESCAPES:
                    position += 1
                    continue
                if escaped != ord("u"):
                    raise _InvalidJSONPrefix
                for _offset in range(4):
                    position += 1
                    if position >= self.length:
                        raise _IncompleteJSONPrefix
                    if self.source[position] not in self._HEX:
                        raise _InvalidJSONPrefix
                position += 1
                continue
            if token < 0x80:
                position += 1
                continue
            position = self._parse_utf8_character(position)
        raise _IncompleteJSONPrefix

    def _parse_utf8_character(self, position: int) -> int:
        first = self.source[position]
        if 0xC2 <= first <= 0xDF:
            width = 2
            second_min, second_max = 0x80, 0xBF
        elif first == 0xE0:
            width = 3
            second_min, second_max = 0xA0, 0xBF
        elif 0xE1 <= first <= 0xEC or 0xEE <= first <= 0xEF:
            width = 3
            second_min, second_max = 0x80, 0xBF
        elif first == 0xED:
            width = 3
            second_min, second_max = 0x80, 0x9F
        elif first == 0xF0:
            width = 4
            second_min, second_max = 0x90, 0xBF
        elif 0xF1 <= first <= 0xF3:
            width = 4
            second_min, second_max = 0x80, 0xBF
        elif first == 0xF4:
            width = 4
            second_min, second_max = 0x80, 0x8F
        else:
            raise _InvalidJSONPrefix
        for offset in range(1, width):
            index = position + offset
            if index >= self.length:
                raise _IncompleteJSONPrefix
            token = self.source[index]
            if offset == 1:
                if not second_min <= token <= second_max:
                    raise _InvalidJSONPrefix
            elif not 0x80 <= token <= 0xBF:
                raise _InvalidJSONPrefix
        return position + width

    def _parse_number(self, position: int) -> int:
        if self.source[position] == ord("-"):
            position += 1
            if position >= self.length:
                raise _IncompleteJSONPrefix
        if self.source[position] == ord("0"):
            position += 1
        elif ord("1") <= self.source[position] <= ord("9"):
            position += 1
            while (
                position < self.length
                and ord("0") <= self.source[position] <= ord("9")
            ):
                position += 1
        else:
            raise _InvalidJSONPrefix
        if position < self.length and self.source[position] == ord("."):
            position += 1
            if position >= self.length:
                raise _IncompleteJSONPrefix
            if not ord("0") <= self.source[position] <= ord("9"):
                raise _InvalidJSONPrefix
            while (
                position < self.length
                and ord("0") <= self.source[position] <= ord("9")
            ):
                position += 1
        if position < self.length and self.source[position] in b"eE":
            position += 1
            if position >= self.length:
                raise _IncompleteJSONPrefix
            if self.source[position] in b"+-":
                position += 1
                if position >= self.length:
                    raise _IncompleteJSONPrefix
            if not ord("0") <= self.source[position] <= ord("9"):
                raise _InvalidJSONPrefix
            while (
                position < self.length
                and ord("0") <= self.source[position] <= ord("9")
            ):
                position += 1
        return position

    def _parse_array(self, position: int) -> int:
        position = self._skip_whitespace(position + 1)
        if position >= self.length:
            raise _IncompleteJSONPrefix
        if self.source[position] == ord("]"):
            return position + 1
        while True:
            position = self._parse_value(position)
            position = self._skip_whitespace(position)
            if position >= self.length:
                raise _IncompleteJSONPrefix
            token = self.source[position]
            if token == ord("]"):
                return position + 1
            if token != ord(","):
                raise _InvalidJSONPrefix
            position = self._skip_whitespace(position + 1)
            if position >= self.length:
                raise _IncompleteJSONPrefix

    def _parse_object(self, position: int) -> int:
        position = self._skip_whitespace(position + 1)
        if position >= self.length:
            raise _IncompleteJSONPrefix
        if self.source[position] == ord("}"):
            return position + 1
        while True:
            if self.source[position] != ord('"'):
                raise _InvalidJSONPrefix
            position = self._parse_string(position)
            position = self._skip_whitespace(position)
            if position >= self.length:
                raise _IncompleteJSONPrefix
            if self.source[position] != ord(":"):
                raise _InvalidJSONPrefix
            position = self._skip_whitespace(position + 1)
            position = self._parse_value(position)
            position = self._skip_whitespace(position)
            if position >= self.length:
                raise _IncompleteJSONPrefix
            token = self.source[position]
            if token == ord("}"):
                return position + 1
            if token != ord(","):
                raise _InvalidJSONPrefix
            position = self._skip_whitespace(position + 1)
            if position >= self.length:
                raise _IncompleteJSONPrefix


def _incomplete_json(source: bytes) -> bool:
    return _JSONPrefixParser(source).classify() is _JSONPrefixState.INCOMPLETE


def _schema_error(value: object) -> str | None:
    if not isinstance(value, dict):
        return "schema: proposal must be an object"
    if set(value) != {"status", "reason_code", "o1"}:
        return "schema: proposal fields differ"
    status = value.get("status")
    reason = value.get("reason_code")
    o1 = value.get("o1")
    if status == "UNSUPPORTED":
        if reason not in {
            "NO_ABSTRACT_CHILD",
            "ONLY_ANSWER_FACTS",
            "INSUFFICIENT_SCOPE",
        } or o1 is not None:
            return "schema: unsupported proposal fields differ"
        return None
    if status != "SUPPORTED" or reason != "SUPPORTED" or not isinstance(o1, dict):
        return "schema: status or reason_code differs"
    if set(o1) != {"label", "scope_rationale", "support_unit_ids"}:
        return "schema: o1 fields differ"
    label = o1.get("label")
    rationale = o1.get("scope_rationale")
    support = o1.get("support_unit_ids")
    if not isinstance(label, str) or not 3 <= len(label) <= 120:
        return "schema: o1 label differs"
    if not isinstance(rationale, str) or not 3 <= len(rationale) <= 240:
        return "schema: o1 scope_rationale differs"
    if (
        not isinstance(support, list)
        or not 1 <= len(support) <= 2
        or any(not isinstance(item, str) or not _SHA256.fullmatch(item) for item in support)
        or len(set(support)) != len(support)
    ):
        return "schema: o1 support_unit_ids differ"
    return None


def classify_completion(
    raw_completion: bytes,
    *,
    schema: Mapping[str, object],
    output_token_count: int,
    max_new_tokens: int,
    allowed_support_unit_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Parse one JSON value and classify parse, schema, and semantic outcomes."""

    if not isinstance(raw_completion, bytes):
        raise TypeError("raw completion must be bytes")
    if not isinstance(schema, Mapping):
        raise TypeError("proposal schema must be a mapping")
    schema_bytes = json.dumps(
        schema,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if _sha256(schema_bytes) != _PROPOSAL_SCHEMA_SHA256:
        return {
            "classification": "schema_error",
            "error": "schema: supplied proposal schema differs",
            "output_token_count": output_token_count,
        }
    if (
        isinstance(output_token_count, bool)
        or not isinstance(output_token_count, int)
        or output_token_count < 0
        or isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
        or output_token_count > max_new_tokens
    ):
        raise ValueError("completion token count is invalid")
    try:
        value = json.loads(raw_completion)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        classification = (
            "truncated_at_ceiling"
            if output_token_count == max_new_tokens and _incomplete_json(raw_completion)
            else "parse_error"
        )
        return {
            "classification": classification,
            "error": f"parse: {exc}",
            "output_token_count": output_token_count,
        }
    error = _schema_error(value)
    if error is not None:
        return {
            "classification": "schema_error",
            "error": error,
            "output_token_count": output_token_count,
        }
    if isinstance(value, dict) and value.get("status") == "SUPPORTED":
        allowed = set(allowed_support_unit_ids or ())
        cited = set(value["o1"]["support_unit_ids"])
        if not cited <= allowed:
            return {
                "classification": "semantic_error",
                "error": "semantic: support unit escapes the proposal job",
                "output_token_count": output_token_count,
            }
    return {
        "classification": "valid",
        "value": value,
        "output_token_count": output_token_count,
    }


class AppendOnlyAttemptLedger:
    """Raw-first attempts anchored to one approved immutable job inventory."""

    _COMMON_EVENT_KEYS = {
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
    _TERMINAL_KEYS = _COMMON_EVENT_KEYS | {
        "classification",
        "output_token_count",
        "raw_path",
        "raw_bytes",
        "raw_sha256",
    }

    def __init__(
        self,
        root: Path,
        *,
        expected_anchor: Mapping[str, object] | None = None,
        create_only: bool,
    ) -> None:
        if expected_anchor is None:
            raise ValueError("proposal ledger expected anchor is required")
        self.root = Path(root)
        self.events_path = self.root / "events.jsonl"
        self.head_path = self.root / "head.json"
        self.anchor_path = self.root / "anchor.json"
        self.lock_path = self.root / "mutation.lock"
        self.completion_path = self.root / "completion.json"
        self.raw_dir = self.root / "raw"
        self._active_raw_name: str | None = None
        self.anchor = self._validated_anchor(expected_anchor)
        self._anchor_source = _canonical_bytes(self.anchor)
        self._anchor_sha256 = _sha256(self._anchor_source)
        self._expected_specs, self._support_by_job = self._anchor_indexes(self.anchor)
        if create_only:
            try:
                _claim_directory_no_symlinks(self.root)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"create-only proposal output exists: {self.root}"
                ) from exc
            except OSError as exc:
                raise ValueError("proposal output parent is missing or unsafe") from exc
            self.raw_dir.mkdir(mode=0o700)
            _write_exclusive(self.events_path, b"")
            _write_exclusive(self.head_path, _canonical_bytes(self._head(0, "0" * 64)))
            _write_exclusive(self.anchor_path, self._anchor_source)
            _write_exclusive(self.lock_path, b"")
            _fsync_directory(self.raw_dir)
            _fsync_directory(self.root)
        elif not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("proposal ledger root is unsafe")
        events = self._verify(allow_incomplete=False)
        if not create_only and events and not self.completion_path.is_file():
            raise ValueError("proposal ledger nonempty reopen requires completion seal")

    @staticmethod
    def _validated_anchor(anchor: Mapping[str, object]) -> dict[str, object]:
        try:
            value = json.loads(_canonical_bytes(dict(anchor)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("proposal ledger anchor is invalid") from exc
        required = {
            "schema_version",
            "stage",
            "preflight_sha256",
            "approval_sha256",
            "job_count",
            "job_inventory_sha256",
            "jobs",
            "schema",
        }
        jobs = value.get("jobs")
        if (
            set(value) != required
            or value.get("schema_version") != _ANCHOR_SCHEMA_VERSION
            or value.get("stage") != "proposal"
            or not isinstance(value.get("preflight_sha256"), str)
            or not _SHA256.fullmatch(value["preflight_sha256"])
            or not isinstance(value.get("approval_sha256"), str)
            or not _SHA256.fullmatch(value["approval_sha256"])
            or type(value.get("job_count")) is not int
            or not isinstance(jobs, list)
            or value.get("job_count") != len(jobs)
            or value.get("job_inventory_sha256")
            != _sha256(
                json.dumps(
                    jobs,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            or not isinstance(value.get("schema"), dict)
            or _sha256(
                json.dumps(
                    value["schema"],
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            != _PROPOSAL_SCHEMA_SHA256
        ):
            raise ValueError("proposal ledger anchor differs")
        seen: set[str] = set()
        for job in jobs:
            if not isinstance(job, dict) or set(job) != {
                "job_id",
                "input_unit_ids",
                "attempts",
            }:
                raise ValueError("proposal ledger anchor job differs")
            job_id = job.get("job_id")
            unit_ids = job.get("input_unit_ids")
            attempts = job.get("attempts")
            if (
                not isinstance(job_id, str)
                or not _SHA256.fullmatch(job_id)
                or job_id in seen
                or not isinstance(unit_ids, list)
                or len(set(unit_ids)) != len(unit_ids)
                or any(
                    not isinstance(item, str) or not _SHA256.fullmatch(item)
                    for item in unit_ids
                )
                or not isinstance(attempts, list)
                or len(attempts) != 2
            ):
                raise ValueError("proposal ledger anchor job inventory differs")
            seen.add(job_id)
            for expected_ordinal, attempt in enumerate(attempts, start=1):
                if not isinstance(attempt, dict) or set(attempt) != {
                    "attempt_ordinal",
                    "request_sha256",
                    "max_new_tokens",
                }:
                    raise ValueError("proposal ledger anchor attempt differs")
                AttemptSpec(
                    stage="proposal",
                    job_id=job_id,
                    attempt_ordinal=attempt.get("attempt_ordinal"),
                    request_sha256=attempt.get("request_sha256"),
                    max_new_tokens=attempt.get("max_new_tokens"),
                )
                if attempt.get("attempt_ordinal") != expected_ordinal:
                    raise ValueError("proposal ledger anchor attempt order differs")
        return value

    @staticmethod
    def _anchor_indexes(
        anchor: Mapping[str, object],
    ) -> tuple[dict[tuple[str, int], AttemptSpec], dict[str, list[str]]]:
        specs: dict[tuple[str, int], AttemptSpec] = {}
        supports: dict[str, list[str]] = {}
        for job in anchor["jobs"]:  # type: ignore[index]
            job_id = str(job["job_id"])
            supports[job_id] = list(job["input_unit_ids"])
            for attempt in job["attempts"]:
                spec = AttemptSpec(
                    stage="proposal",
                    job_id=job_id,
                    attempt_ordinal=attempt["attempt_ordinal"],
                    request_sha256=attempt["request_sha256"],
                    max_new_tokens=attempt["max_new_tokens"],
                )
                specs[(job_id, spec.attempt_ordinal)] = spec
        return specs, supports

    @staticmethod
    def _head(count: int, head_sha256: str) -> dict[str, object]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event_count": count,
            "head_sha256": head_sha256,
        }

    def _load_events(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for line in self.events_path.read_bytes().splitlines():
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("proposal ledger event is invalid") from exc
            if not isinstance(row, dict) or line + b"\n" != _canonical_bytes(row):
                raise ValueError("proposal ledger event is not canonical")
            rows.append(row)
        return rows

    def _expected_completion(
        self, events: Sequence[Mapping[str, object]], head: Mapping[str, object]
    ) -> dict[str, object]:
        valid_jobs = sorted(
            {
                str(event["job_id"])
                for event in events
                if event.get("state") == "terminal"
                and event.get("classification") == "valid"
            }
        )
        return {
            "schema_version": _COMPLETION_SCHEMA_VERSION,
            "anchor_sha256": self._anchor_sha256,
            "event_count": len(events),
            "final_head_sha256": head["head_sha256"],
            "completed_job_count": len(valid_jobs),
            "completed_job_ids_sha256": _sha256(_canonical_bytes(valid_jobs)),
        }

    def _verify(self, *, allow_incomplete: bool) -> list[dict[str, object]]:
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("proposal ledger root is unsafe")
        entries = list(self.root.iterdir())
        names = {entry.name for entry in entries}
        if names not in (_ROOT_NAMES, _ROOT_NAMES | {"completion.json"}):
            raise ValueError("proposal ledger contains extra or missing artifacts")
        regular = (self.events_path, self.head_path, self.anchor_path, self.lock_path)
        if any(path.is_symlink() or not path.is_file() for path in regular) or (
            self.raw_dir.is_symlink() or not self.raw_dir.is_dir()
        ):
            raise ValueError("proposal ledger inventory is unsafe")
        if self.lock_path.stat().st_size != 0:
            raise ValueError("proposal ledger lock artifact differs")
        if self.anchor_path.read_bytes() != self._anchor_source:
            raise ValueError("proposal ledger anchor differs")
        try:
            head_source = self.head_path.read_bytes()
            head = json.loads(head_source)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("proposal ledger head is invalid") from exc
        if not isinstance(head, dict) or head_source != _canonical_bytes(head):
            raise ValueError("proposal ledger head is not canonical")
        events = self._load_events()
        previous = "0" * 64
        pending: dict[tuple[str, int], AttemptSpec] = {}
        completed: set[tuple[str, int]] = set()
        classifications: dict[tuple[str, int], str] = {}
        raw_names: set[str] = set()
        for index, event in enumerate(events, start=1):
            state = event.get("state")
            expected_keys = (
                self._COMMON_EVENT_KEYS if state == "started" else self._TERMINAL_KEYS
            )
            if set(event) != expected_keys:
                raise ValueError("proposal ledger event key set differs")
            claimed = event.get("event_sha256")
            payload = {key: value for key, value in event.items() if key != "event_sha256"}
            if (
                event.get("schema_version") != LEDGER_SCHEMA_VERSION
                or event.get("sequence") != index
                or event.get("previous_event_sha256") != previous
                or claimed != _sha256(_canonical_bytes(payload))
            ):
                raise ValueError("proposal ledger hash chain differs")
            previous = str(claimed)
            try:
                spec = AttemptSpec(
                    stage=event["stage"],
                    job_id=event["job_id"],
                    attempt_ordinal=event["attempt_ordinal"],
                    request_sha256=event["request_sha256"],
                    max_new_tokens=event["max_new_tokens"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("proposal ledger attempt spec differs") from exc
            key = (spec.job_id, spec.attempt_ordinal)
            if self._expected_specs.get(key) != spec:
                raise ValueError("proposal ledger attempt spec differs from anchor")
            if state == "started":
                if key in pending or key in completed:
                    raise ValueError("proposal ledger contains duplicate ordinals")
                if spec.attempt_ordinal == 2 and classifications.get(
                    (spec.job_id, 1)
                ) != "truncated_at_ceiling":
                    raise ValueError("proposal retry requires recomputed primary truncation")
                pending[key] = spec
                continue
            if state != "terminal" or pending.get(key) != spec:
                raise ValueError("proposal ledger terminal spec is reordered or differs")
            raw_name = event.get("raw_path")
            expected_raw_name = f"{spec.job_id}.{spec.attempt_ordinal}.completion"
            raw_path = self.raw_dir / expected_raw_name
            output_count = event.get("output_token_count")
            if (
                raw_name != expected_raw_name
                or raw_name in raw_names
                or type(output_count) is not int
                or not 0 <= output_count <= spec.max_new_tokens
                or raw_path.is_symlink()
                or not raw_path.is_file()
            ):
                raise ValueError("proposal ledger raw name or token count differs")
            raw = raw_path.read_bytes()
            if (
                type(event.get("raw_bytes")) is not int
                or event.get("raw_bytes") != len(raw)
                or event.get("raw_sha256") != _sha256(raw)
            ):
                raise ValueError("proposal ledger raw completion differs")
            recomputed = classify_completion(
                raw,
                schema=self.anchor["schema"],  # type: ignore[arg-type]
                output_token_count=output_count,
                max_new_tokens=spec.max_new_tokens,
                allowed_support_unit_ids=self._support_by_job[spec.job_id],
            )
            if event.get("classification") != recomputed.get("classification"):
                raise ValueError("proposal ledger classification differs from raw bytes")
            raw_names.add(expected_raw_name)
            classifications[key] = str(recomputed["classification"])
            del pending[key]
            completed.add(key)
        if head != self._head(len(events), previous):
            raise ValueError("proposal ledger deletion or head tampering detected")
        raw_entries = list(self.raw_dir.iterdir())
        actual_raw = {path.name for path in raw_entries}
        if any(path.is_symlink() or not path.is_file() for path in raw_entries):
            raise ValueError("proposal ledger raw inventory is unsafe")
        if actual_raw != raw_names:
            active_raw = {self._active_raw_name} if self._active_raw_name else set()
            if not allow_incomplete or actual_raw - raw_names != active_raw:
                raise ValueError("proposal ledger contains incomplete or extra raw state")
        if pending and not allow_incomplete:
            raise ValueError("proposal ledger contains an incomplete started attempt")
        if len(pending) > 1:
            raise ValueError("proposal ledger contains multiple incomplete attempts")
        if self.completion_path.exists() or self.completion_path.is_symlink():
            if self.completion_path.is_symlink() or not self.completion_path.is_file():
                raise ValueError("proposal ledger completion is unsafe")
            try:
                source = self.completion_path.read_bytes()
                completion = json.loads(source)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("proposal ledger completion is invalid") from exc
            if (
                not isinstance(completion, dict)
                or source != _canonical_bytes(completion)
                or completion != self._expected_completion(events, head)
                or completion.get("completed_job_count") != self.anchor["job_count"]
            ):
                raise ValueError("proposal ledger completion final head differs")
        return events

    @contextmanager
    def _mutation_lock(self) -> object:
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.lock_path, flags)
        try:
            observed = os.fstat(descriptor)
            if not stat.S_ISREG(observed.st_mode) or observed.st_size != 0:
                raise ValueError("proposal ledger lock artifact differs")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("proposal ledger is locked by another executor") from exc
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def read_events(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._verify(allow_incomplete=True)]

    def _append(self, event: Mapping[str, object]) -> dict[str, object]:
        events = self._verify(allow_incomplete=True)
        previous = str(events[-1]["event_sha256"]) if events else "0" * 64
        payload = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "sequence": len(events) + 1,
            "previous_event_sha256": previous,
            **dict(event),
        }
        row = {**payload, "event_sha256": _sha256(_canonical_bytes(payload))}
        with self.events_path.open("ab") as sink:
            sink.write(_canonical_bytes(row))
            sink.flush()
            os.fsync(sink.fileno())
        _replace_fsynced(
            self.head_path,
            _canonical_bytes(self._head(len(events) + 1, str(row["event_sha256"]))),
        )
        return row

    def run_attempt(
        self,
        spec: AttemptSpec,
        generate: Callable[[object, object, int], object],
        *,
        messages: object = (),
        schema: Mapping[str, object] | None = None,
        allowed_support_unit_ids: Sequence[str] | None = None,
    ) -> dict[str, object]:
        with self._mutation_lock():
            existing = self._verify(allow_incomplete=False)
            key = (spec.job_id, spec.attempt_ordinal)
            if self._expected_specs.get(key) != spec:
                raise ValueError("proposal attempt differs from anchored job inventory")
            if any(
                (event.get("job_id"), event.get("attempt_ordinal")) == key
                for event in existing
            ):
                raise ValueError("proposal ledger contains duplicate ordinals")
            if spec.attempt_ordinal == 2 and not any(
                event.get("state") == "terminal"
                and event.get("job_id") == spec.job_id
                and event.get("attempt_ordinal") == 1
                and event.get("classification") == "truncated_at_ceiling"
                for event in existing
            ):
                raise ValueError("proposal retry requires a truncated primary attempt")
            self._append({"state": "started", **asdict(spec)})
            generated = generate(messages, schema, spec.max_new_tokens)
            if not (
                isinstance(generated, tuple)
                and len(generated) == 2
                and isinstance(generated[0], bytes)
                and type(generated[1]) is int
            ):
                raise TypeError(
                    "model generation must return raw bytes and an exact token count"
                )
            raw_completion, output_token_count = generated
            raw_name = f"{spec.job_id}.{spec.attempt_ordinal}.completion"
            self._active_raw_name = raw_name
            raw_path = self.raw_dir / raw_name
            _write_exclusive(raw_path, raw_completion)
            _fsync_directory(self.raw_dir)
            if schema is None:
                raise ValueError("proposal schema is required for ledger classification")
            if list(allowed_support_unit_ids or ()) != self._support_by_job[spec.job_id]:
                raise ValueError("proposal support inventory differs from run anchor")
            result = classify_completion(
                raw_completion,
                schema=schema,
                output_token_count=output_token_count,
                max_new_tokens=spec.max_new_tokens,
                allowed_support_unit_ids=allowed_support_unit_ids,
            )
            self._append(
                {
                    "state": "terminal",
                    **asdict(spec),
                    "classification": result["classification"],
                    "output_token_count": output_token_count,
                    "raw_path": raw_name,
                    "raw_bytes": len(raw_completion),
                    "raw_sha256": _sha256(raw_completion),
                }
            )
            self._active_raw_name = None
            self._verify(allow_incomplete=False)
            return result

    def seal_completion(self) -> dict[str, object]:
        with self._mutation_lock():
            events = self._verify(allow_incomplete=False)
            if self.completion_path.exists() or self.completion_path.is_symlink():
                raise FileExistsError("proposal ledger completion already exists")
            head = json.loads(self.head_path.read_bytes())
            completion = self._expected_completion(events, head)
            if completion["completed_job_count"] != self.anchor["job_count"]:
                raise ValueError("proposal ledger cannot seal an incomplete run")
            _write_exclusive(self.completion_path, _canonical_bytes(completion))
            _fsync_directory(self.root)
            self._verify(allow_incomplete=False)
            return completion
