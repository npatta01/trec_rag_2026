"""Fail-closed parsing for Hugging Face Bucket listing output.

The ``hf`` CLI emits no bytes for an empty prefix, a JSON array for normal
listings, and some versions/modes may emit one JSON object per line.  Keep that
transport variation separate from the immutable two-file shard contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence


_BUNDLE_FILES = frozenset({"bundle.tar.zst", "bundle-complete.json"})
_CONTAINER_KEYS = ("data", "files", "items", "results")


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


def require_empty_listing(raw: str | bytes) -> None:
    records = parse_hf_bucket_listing(raw)
    if records:
        raise HFListingError("immutable Hugging Face shard prefix is not empty")


def require_bundle_listing(raw: str | bytes) -> None:
    records = parse_hf_bucket_listing(raw)
    names: list[str] = []
    for record in records:
        path = record.get("path")
        if not isinstance(path, str) or not path or "\\" in path:
            raise HFListingError("Hugging Face listing entry has no safe path")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or pure.name in {"", ".", ".."}:
            raise HFListingError("Hugging Face listing entry has no safe path")
        names.append(pure.name)
    if len(set(names)) != len(names):
        raise HFListingError("Hugging Face bundle listing has a duplicate file name")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        raw = args.listing.read_bytes()
        if args.mode == "require-empty":
            require_empty_listing(raw)
        else:
            require_bundle_listing(raw)
    except (OSError, HFListingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
