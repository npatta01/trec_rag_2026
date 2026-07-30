"""Build durable organizer Pi trace bundles and export them to Phoenix."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import BinaryIO, Iterator
import zipfile

from trec_rag.organizer_pi_inputs import OrganizerTopic, select_topic
from trec_rag.tracing.phoenix_export import (
    ExportReceipt,
    PhoenixSettings,
    assert_no_secrets,
    export_trace_bundle,
)
from trec_rag.pi_event_trace import (
    DEFAULT_PROJECT_NAME,
    build_fixed_trace,
    build_piika_trace,
    load_pi_events,
)
from trec_rag.tracing.models import (
    TraceBundle,
    _strict_json_loads,
    read_trace_bundle,
    write_trace_bundle,
)
from trec_rag.repo_env import find_repo_root, load_repo_env


FIXED_DOCUMENT_COUNT = 100
FIXED_DOCUMENT_WORD_CAP = 1000


class _UsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build a local strict-JSON trace bundle")
    build.add_argument(
        "--baseline", choices=("piika-agentic", "ragnarok-fixed"), required=True
    )
    build.add_argument("--topic", required=True)
    build.add_argument("--topic-tsv", type=Path, required=True)
    build.add_argument("--events", type=Path, required=True)
    build.add_argument("--record", type=Path, required=True)
    build.add_argument("--bundle", type=Path, required=True)
    build.add_argument("--session", help="defaults to <topic>-comparison")
    build.add_argument("--ranked-run", type=Path)
    build.add_argument("--documents", type=Path)
    build.add_argument("--organizer-script", type=Path)

    export = commands.add_parser("export", help="export a saved bundle to Phoenix")
    export.add_argument("--bundle", type=Path, required=True)
    export.add_argument("--receipt", type=Path, required=True)
    return parser


def _default_ignored_checker(path: Path) -> bool:
    repo_root = find_repo_root()
    target = Path(path).resolve()
    try:
        relative = target.relative_to(repo_root.resolve())
    except ValueError:
        return False
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "-q", "--", str(relative)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _require_ignored(path: Path, checker: Callable[[Path], bool]) -> None:
    if not checker(Path(path)):
        raise ValueError("trace bundles and receipts must be written to ignored paths")


def _read_json_object(path: Path) -> dict[str, object]:
    source = Path(path)
    try:
        body = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("output record must be UTF-8") from error
    value = _strict_json_loads(body)
    if not isinstance(value, dict):
        raise ValueError("output record must be one JSON object")
    return value


def _metadata(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, Mapping) else None


def _validate_output_identity(
    record: Mapping[str, object], topic: OrganizerTopic, *, baseline: str
) -> None:
    output = record.get("trec_rag_output")
    if baseline == "piika-agentic":
        if record.get("query_id") != topic.topic_id:
            raise ValueError("output record query_id does not match selected topic")
        metadata = _metadata(output)
    else:
        if "query_id" in record and record.get("query_id") != topic.topic_id:
            raise ValueError("output record query_id does not match selected topic")
        metadata = _metadata(output) or _metadata(record)
    if metadata is None:
        if baseline == "ragnarok-fixed":
            if record.get("query_id") == topic.topic_id and record.get("status") != "completed":
                return
            raise ValueError("fixed output record is missing topic identity metadata")
        return
    if metadata.get("narrative_id") != topic.topic_id:
        raise ValueError("output record narrative_id does not match selected topic")
    if metadata.get("narrative") != topic.narrative:
        raise ValueError("output record narrative does not match selected topic")


def _normalized_record(
    record: dict[str, object], topic: OrganizerTopic, *, baseline: str
) -> dict[str, object]:
    _validate_output_identity(record, topic, baseline=baseline)
    if (
        baseline == "ragnarok-fixed"
        and "trec_rag_output" not in record
        and _metadata(record) is not None
    ):
        return {"status": "completed", "trec_rag_output": record}
    return record


def _ranked_docids(path: Path, topic_id: str) -> list[str]:
    rows: list[tuple[int, str]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if fields[0] != topic_id:
                continue
            if len(fields) != 6:
                raise ValueError(f"ranked run row {line_number} must have six columns")
            try:
                rank = int(fields[3])
            except ValueError as error:
                raise ValueError(f"ranked run row {line_number} has an invalid rank") from error
            rows.append((rank, fields[2]))
    rows.sort()
    ranks = [rank for rank, _ in rows]
    docids = [docid for _, docid in rows]
    if len(rows) != FIXED_DOCUMENT_COUNT:
        raise ValueError(f"fixed ranked run must contain exactly {FIXED_DOCUMENT_COUNT} rows")
    if ranks != list(range(1, FIXED_DOCUMENT_COUNT + 1)):
        raise ValueError("fixed ranked run ranks must be contiguous from 1 through 100")
    if len(set(docids)) != len(docids):
        raise ValueError("fixed ranked run contains duplicate document IDs")
    return docids


def _archive_lines(path: Path):
    archive = zipfile.ZipFile(path)
    try:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".jsonl", ".json"))
        ]
        if len(names) != 1:
            raise ValueError("document ZIP must contain exactly one JSONL member")
        raw = archive.open(names[0])
        text = io.TextIOWrapper(raw, encoding="utf-8")
        try:
            yield from text
        finally:
            text.close()
    finally:
        archive.close()


def _document_value(record: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_documents(path: Path, docids: Sequence[str]) -> list[dict[str, object]]:
    wanted = set(docids)
    found: dict[str, str] = {}
    for line_number, line in enumerate(_archive_lines(Path(path)), start=1):
        if not line.strip():
            continue
        try:
            value = _strict_json_loads(line)
        except ValueError as error:
            raise ValueError(f"invalid document JSON at row {line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"document row {line_number} must be an object")
        candidates = value.get("candidates")
        records = candidates if isinstance(candidates, list) else [value]
        for record in records:
            if not isinstance(record, Mapping):
                continue
            docid = _document_value(record, "docid", "id", "_id")
            if docid not in wanted:
                continue
            text = _document_value(record, "text", "doc", "contents", "body")
            if not text:
                continue
            truncated = " ".join(text.split()[:FIXED_DOCUMENT_WORD_CAP])
            if docid in found and found[docid] != truncated:
                raise ValueError(f"document archive contains conflicting duplicate {docid}")
            found[docid] = truncated
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"document archive is missing {len(missing)} ranked documents")
    return [
        {"rank": rank, "docid": docid, "text": found[docid]}
        for rank, docid in enumerate(docids, start=1)
    ]


def _load_organizer_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_trec_rag_fixed_organizer", path)
    if spec is None or spec.loader is None:
        raise ValueError("organizer script cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not isinstance(getattr(module, "SYSTEM_PROMPT", None), str):
        raise ValueError("organizer script SYSTEM_PROMPT must be a string")
    if not callable(getattr(module, "prompt", None)):
        raise ValueError("organizer script prompt must be callable")
    return module


def _message_text(message: Mapping[str, object]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = [
        item.get("text")
        for item in content
        if isinstance(item, Mapping)
        and item.get("type") in {"text", "input_text"}
        and isinstance(item.get("text"), str)
    ]
    return "\n".join(parts) if parts else None


def _verify_native_user_prompt(events: Sequence[Mapping[str, object]], rendered: str) -> None:
    native_prompts: list[str] = []
    for event in events:
        if event.get("type") not in {"message_start", "message_end"}:
            continue
        message = event.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        text = _message_text(message)
        if text is not None:
            native_prompts.append(text)
    def matches(native: str) -> bool:
        if native == rendered:
            return True
        wrapper = re.fullmatch(
            r'<file name="/[^"\n]*/prompt\.txt">\n(?P<body>[\s\S]*)</file>\n',
            native,
        )
        return wrapper is not None and wrapper.group("body") == f"{rendered}\n"

    if not native_prompts or not all(matches(native) for native in native_prompts):
        raise ValueError("rendered organizer prompt does not match native Pi user event")


def _build_bundle(arguments: argparse.Namespace) -> TraceBundle:
    topic = select_topic(arguments.topic_tsv, arguments.topic)
    events = load_pi_events(arguments.events)
    record = _normalized_record(
        _read_json_object(arguments.record), topic, baseline=arguments.baseline
    )
    session_id = arguments.session or f"{topic.topic_id}-comparison"
    if arguments.baseline == "piika-agentic":
        return build_piika_trace(
            topic=topic,
            events=events,
            run_record=record,
            session_id=session_id,
        )

    docids = _ranked_docids(arguments.ranked_run, topic.topic_id)
    documents = _load_documents(arguments.documents, docids)
    organizer = _load_organizer_script(arguments.organizer_script)
    texts = {str(document["docid"]): str(document["text"]) for document in documents}
    rendered = organizer.prompt(topic.narrative, docids, texts)
    if not isinstance(rendered, str):
        raise ValueError("organizer script prompt must return a string")
    _verify_native_user_prompt(events, rendered)
    return build_fixed_trace(
        topic=topic,
        events=events,
        run_record=record,
        system_prompt=organizer.SYSTEM_PROMPT,
        user_prompt=rendered,
        documents=documents,
        session_id=session_id,
    )


def _write_json(temporary: BinaryIO, value: object) -> None:
    temporary.write(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


@contextmanager
def _staged_atomic_json(path: Path) -> Iterator[BinaryIO]:
    """Reserve an atomic destination, replacing it only after a clean body write."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        temporary = os.fdopen(descriptor, "wb")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise
    try:
        with temporary:
            yield temporary
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _credential_values(settings: PhoenixSettings) -> tuple[object, ...]:
    values: list[object] = [settings.api_key]
    suffixes = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
    values.extend(value for name, value in os.environ.items() if name.endswith(suffixes))
    return tuple(values)


def _load_repository_environment() -> None:
    load_repo_env(find_repo_root())


def main(
    argv: Sequence[str] | None = None,
    *,
    bundle_loader: Callable[[Path], TraceBundle] = read_trace_bundle,
    settings_loader: Callable[[], PhoenixSettings] = PhoenixSettings.from_env,
    exporter: Callable[
        [TraceBundle, PhoenixSettings], ExportReceipt
    ] = export_trace_bundle,
    ignored_checker: Callable[[Path], bool] | None = None,
    environment_loader: Callable[[], None] = _load_repository_environment,
    credential_loader: Callable[[PhoenixSettings], Sequence[object]] = _credential_values,
) -> int:
    """Run the trace CLI and return a process-style exit code."""
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
        checker = ignored_checker or _default_ignored_checker
        if arguments.command == "build":
            fixed_values = (
                arguments.ranked_run,
                arguments.documents,
                arguments.organizer_script,
            )
            if arguments.baseline == "ragnarok-fixed" and not all(fixed_values):
                raise _UsageError(
                    "ragnarok-fixed requires --ranked-run, --documents, and --organizer-script"
                )
            if arguments.baseline == "piika-agentic" and any(fixed_values):
                raise _UsageError(
                    "fixed-path arguments are not valid for piika-agentic"
                )
            _require_ignored(arguments.bundle, checker)
            arguments.bundle.parent.mkdir(parents=True, exist_ok=True)
            write_trace_bundle(_build_bundle(arguments), arguments.bundle)
            return 0

        _require_ignored(arguments.bundle, checker)
        _require_ignored(arguments.receipt, checker)
        environment_loader()
        settings = settings_loader()
        bundle = bundle_loader(arguments.bundle)
        if bundle.project_name != settings.project_name:
            raise ValueError("bundle project does not match PHOENIX_PROJECT_NAME")
        assert_no_secrets(bundle, credential_loader(settings))
        with _staged_atomic_json(arguments.receipt) as temporary_receipt:
            receipt = exporter(bundle, settings)
            if not isinstance(receipt, ExportReceipt):
                raise TypeError("exporter must return ExportReceipt")
            _write_json(temporary_receipt, asdict(receipt))
        return 0
    except _UsageError as error:
        print(f"usage error: {error}", file=sys.stderr)
        return 2
    except SystemExit as error:
        return int(error.code or 0)
    except Exception as error:
        print(f"error: {type(error).__name__}: command failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
