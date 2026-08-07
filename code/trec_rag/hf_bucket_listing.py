"""Fail-closed parsing for Hugging Face Bucket listing output.

The ``hf`` CLI emits no bytes for an empty prefix, a JSON array for normal
listings, and some versions/modes may emit one JSON object per line.  Keep that
transport variation separate from the immutable two-file shard contract.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


_BUNDLE_FILES = frozenset({"bundle.tar.zst", "bundle-complete.json"})
_CONTAINER_KEYS = ("data", "files", "items", "results")
_ENTRY_TYPES = frozenset({"directory", "file"})
_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_SAFE_TOPIC_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class HFListingError(ValueError):
    """The Bucket listing is malformed or contradicts the shard contract."""


def _records(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, dict):
        for key in _CONTAINER_KEYS:
            nested = value.get(key)
            if isinstance(nested, list):
                return _records(nested)
        return (dict(value),)
    if not isinstance(value, list):
        raise HFListingError("Hugging Face listing JSON must contain objects")
    records: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise HFListingError("Hugging Face listing entries must be objects")
        records.append(dict(item))
    return tuple(records)


def parse_hf_bucket_listing(raw: str | bytes) -> tuple[Mapping[str, Any], ...]:
    """Parse empty, JSON-array, wrapped-JSON, or JSONL CLI output."""
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HFListingError("Hugging Face listing is not UTF-8") from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise TypeError("raw listing must be text or bytes")
    if not text.strip():
        return ()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        values: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise HFListingError(
                    f"Hugging Face listing line {line_number} is not valid JSON"
                ) from exc
        flattened: list[Mapping[str, Any]] = []
        for item in values:
            flattened.extend(_records(item))
        return tuple(flattened)
    return _records(value)


def _path_parts(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise HFListingError(f"{label} has no safe path")
    parts = tuple(value.split("/"))
    if (
        value.startswith("/")
        or value.endswith("/")
        or any(
            not part
            or part in {".", ".."}
            or _SAFE_PATH_COMPONENT.fullmatch(part) is None
            for part in parts
        )
    ):
        raise HFListingError(f"{label} has no safe path")
    return parts


def _target_entries(
    raw: str | bytes,
    *,
    topic_prefix: str,
) -> tuple[tuple[tuple[str, ...], str], ...]:
    target_parts = _path_parts(topic_prefix, label="Hugging Face topic prefix")
    if (
        len(target_parts) < 2
        or _SAFE_TOPIC_COMPONENT.fullmatch(target_parts[-1]) is None
    ):
        raise HFListingError("Hugging Face topic prefix has no safe topic ID")
    run_parent = target_parts[:-1]
    target_topic = target_parts[-1]

    seen_paths: set[tuple[str, ...]] = set()
    target_entries: list[tuple[tuple[str, ...], str]] = []
    for record in parse_hf_bucket_listing(raw):
        path_parts = _path_parts(
            record.get("path"), label="Hugging Face listing entry"
        )
        entry_type = record.get("type")
        if entry_type not in _ENTRY_TYPES:
            raise HFListingError("Hugging Face listing entry has no known type")
        if path_parts in seen_paths:
            raise HFListingError("Hugging Face listing has a duplicate path")
        seen_paths.add(path_parts)

        if (
            len(path_parts) < len(run_parent) + 1
            or path_parts[: len(run_parent)] != run_parent
        ):
            raise HFListingError(
                "Hugging Face listing entry is outside the selected run prefix"
            )
        sibling_topic = path_parts[len(run_parent)]
        if _SAFE_TOPIC_COMPONENT.fullmatch(sibling_topic) is None:
            raise HFListingError(
                "Hugging Face listing entry has no safe sibling topic ID"
            )
        if len(path_parts) == len(run_parent) + 1:
            if entry_type != "directory":
                raise HFListingError(
                    "Hugging Face topic root listing entry is not a directory"
                )
            continue
        if sibling_topic == target_topic:
            target_entries.append((path_parts[len(target_parts) :], entry_type))
    return tuple(target_entries)


def require_empty_listing(raw: str | bytes, *, topic_prefix: str) -> None:
    if _target_entries(raw, topic_prefix=topic_prefix):
        raise HFListingError("immutable Hugging Face shard prefix is not empty")


def require_bundle_listing(raw: str | bytes, *, topic_prefix: str) -> None:
    names: list[str] = []
    for relative_parts, entry_type in _target_entries(
        raw, topic_prefix=topic_prefix
    ):
        if len(relative_parts) != 1 or entry_type != "file":
            raise HFListingError(
                "Hugging Face shard prefix must contain exactly the two bundle files"
            )
        names.append(relative_parts[0])
    if frozenset(names) != _BUNDLE_FILES:
        raise HFListingError(
            "Hugging Face shard prefix must contain exactly the two bundle files"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate machine-readable Hugging Face Bucket listings."
    )
    parser.add_argument("mode", choices=("require-empty", "require-bundle"))
    parser.add_argument("listing", type=Path)
    parser.add_argument(
        "--topic-prefix",
        required=True,
        help="Exact bucket path below the listed run parent.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        raw = args.listing.read_bytes()
        if args.mode == "require-empty":
            require_empty_listing(raw, topic_prefix=args.topic_prefix)
        else:
            require_bundle_listing(raw, topic_prefix=args.topic_prefix)
    except (OSError, HFListingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
