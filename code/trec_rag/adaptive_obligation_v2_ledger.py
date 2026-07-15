"""Crash-safe append-only attempt ledger for adaptive-obligation v2 inference."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


LEDGER_SCHEMA_VERSION = "adaptive-obligation-v2-attempt-ledger-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROOT_NAMES = frozenset({"events.jsonl", "head.json", "raw"})
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


def _incomplete_json(source: bytes) -> bool:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError:
        return False
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return False
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        if exc.msg.startswith("Unterminated string"):
            return True
        return exc.pos >= len(text.rstrip()) and exc.msg in {
            "Expecting value",
            "Expecting property name enclosed in double quotes",
            "Expecting ':' delimiter",
            "Expecting ',' delimiter",
        }
    return False


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
    """Durable raw-first attempts with an anchored SHA-256 event chain."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.events_path = self.root / "events.jsonl"
        self.head_path = self.root / "head.json"
        self.raw_dir = self.root / "raw"
        self._active_key: tuple[str, int] | None = None
        self._active_raw_name: str | None = None
        if not self.root.exists():
            self.root.mkdir(parents=True)
            self.raw_dir.mkdir()
            _write_exclusive(self.events_path, b"")
            _write_exclusive(self.head_path, _canonical_bytes(self._head(0, "0" * 64)))
            _fsync_directory(self.raw_dir)
            _fsync_directory(self.root)
        self._verify(allow_incomplete=False)

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

    def _verify(self, *, allow_incomplete: bool) -> list[dict[str, object]]:
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("proposal ledger root is unsafe")
        entries = list(self.root.iterdir())
        if {entry.name for entry in entries} != _ROOT_NAMES:
            raise ValueError("proposal ledger contains extra or missing artifacts")
        if (
            self.events_path.is_symlink()
            or not self.events_path.is_file()
            or self.head_path.is_symlink()
            or not self.head_path.is_file()
            or self.raw_dir.is_symlink()
            or not self.raw_dir.is_dir()
        ):
            raise ValueError("proposal ledger inventory is unsafe")
        try:
            head_source = self.head_path.read_bytes()
            head = json.loads(head_source)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("proposal ledger head is invalid") from exc
        if not isinstance(head, dict) or head_source != _canonical_bytes(head):
            raise ValueError("proposal ledger head is not canonical")
        events = self._load_events()
        previous = "0" * 64
        pending: dict[tuple[str, int], dict[str, object]] = {}
        completed: set[tuple[str, int]] = set()
        raw_names: set[str] = set()
        for index, event in enumerate(events, start=1):
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
            key = (str(event.get("job_id")), int(event.get("attempt_ordinal", -1)))
            state = event.get("state")
            if state == "started":
                if key in pending or key in completed:
                    raise ValueError("proposal ledger contains duplicate ordinals")
                pending[key] = event
            elif state == "terminal":
                if key not in pending:
                    raise ValueError("proposal ledger terminal event is reordered")
                raw_name = event.get("raw_path")
                if not isinstance(raw_name, str) or raw_name in raw_names:
                    raise ValueError("proposal ledger raw binding differs")
                raw_path = self.raw_dir / raw_name
                if (
                    "/" in raw_name
                    or raw_path.is_symlink()
                    or not raw_path.is_file()
                    or event.get("raw_bytes") != raw_path.stat().st_size
                    or event.get("raw_sha256") != _sha256(raw_path.read_bytes())
                ):
                    raise ValueError("proposal ledger raw completion differs")
                raw_names.add(raw_name)
                del pending[key]
                completed.add(key)
            else:
                raise ValueError("proposal ledger state differs")
        if (
            head
            != self._head(len(events), previous)
        ):
            raise ValueError("proposal ledger deletion or head tampering detected")
        actual_raw = {path.name for path in self.raw_dir.iterdir()}
        if any(path.is_symlink() or not path.is_file() for path in self.raw_dir.iterdir()):
            raise ValueError("proposal ledger raw inventory is unsafe")
        if actual_raw != raw_names:
            active_raw = (
                {self._active_raw_name}
                if self._active_raw_name is not None
                else set()
            )
            if not allow_incomplete or actual_raw - raw_names != active_raw:
                raise ValueError("proposal ledger contains incomplete or extra raw state")
        if pending and not allow_incomplete:
            raise ValueError("proposal ledger contains an incomplete started attempt")
        if len(pending) > 1:
            raise ValueError("proposal ledger contains multiple incomplete attempts")
        return events

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
        existing = self._verify(allow_incomplete=False)
        key = (spec.job_id, spec.attempt_ordinal)
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
        self._active_key = key
        generated = generate(messages, schema, spec.max_new_tokens)
        if isinstance(generated, bytes):
            raw_completion = generated
            output_token_count = spec.max_new_tokens
        elif (
            isinstance(generated, tuple)
            and len(generated) == 2
            and isinstance(generated[0], bytes)
            and isinstance(generated[1], int)
            and not isinstance(generated[1], bool)
        ):
            raw_completion, output_token_count = generated
        else:
            raise TypeError("model generation must return raw bytes and an exact token count")
        raw_name = f"{spec.job_id}.{spec.attempt_ordinal}.completion"
        self._active_raw_name = raw_name
        raw_path = self.raw_dir / raw_name
        _write_exclusive(raw_path, raw_completion)
        _fsync_directory(self.raw_dir)
        if schema is None:
            try:
                value = json.loads(raw_completion)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                result = {
                    "classification": "parse_error",
                    "error": f"parse: {exc}",
                    "output_token_count": output_token_count,
                }
            else:
                result = {
                    "classification": "valid",
                    "value": value,
                    "output_token_count": output_token_count,
                }
        else:
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
        self._verify(allow_incomplete=False)
        self._active_key = None
        self._active_raw_name = None
        return result
