"""Order-independent semantic comparison for reranker artifact JSONL rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


ArtifactKey = tuple[str, ...]
ArtifactRow = dict[str, Any]


def _artifact_key(row: Mapping[str, Any], *, window: bool) -> ArtifactKey:
    topic_id = str(row["topic_id"])
    docid = str(row["docid"])
    if window:
        return (topic_id, docid, str(int(row["chunk_index"])))
    return (topic_id, docid)


def _index_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: bool,
    label: str,
) -> dict[ArtifactKey, ArtifactRow]:
    indexed: dict[ArtifactKey, ArtifactRow] = {}
    for row_number, source_row in enumerate(rows, start=1):
        row = dict(source_row)
        key = _artifact_key(row, window=window)
        if key in indexed:
            raise ValueError(
                f"{label}: duplicate artifact key {key!r} at row {row_number}"
            )
        indexed[key] = row
    return indexed


def _canonical_digest(indexed: Mapping[ArtifactKey, ArtifactRow]) -> str:
    digest = hashlib.sha256()
    for key in sorted(indexed):
        encoded = json.dumps(
            indexed[key],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(encoded.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def compare_artifact_rows(
    canonical_rows: Iterable[Mapping[str, Any]],
    regenerated_rows: Iterable[Mapping[str, Any]],
    *,
    window: bool,
) -> dict[str, object]:
    """Require key-indexed row equality while deliberately ignoring row order."""

    canonical = _index_rows(canonical_rows, window=window, label="canonical")
    regenerated = _index_rows(regenerated_rows, window=window, label="regenerated")
    canonical_keys = set(canonical)
    regenerated_keys = set(regenerated)
    missing = sorted(canonical_keys - regenerated_keys)
    extra = sorted(regenerated_keys - canonical_keys)
    if missing or extra:
        raise ValueError(
            "artifact keys differ: "
            f"missing={missing[:5]!r} extra={extra[:5]!r} "
            f"missing_count={len(missing)} extra_count={len(extra)}"
        )

    for key in sorted(canonical):
        if canonical[key] == regenerated[key]:
            continue
        all_fields = sorted(set(canonical[key]) | set(regenerated[key]))
        changed_fields = [
            field
            for field in all_fields
            if canonical[key].get(field) != regenerated[key].get(field)
            or (field in canonical[key]) != (field in regenerated[key])
        ]
        raise ValueError(
            f"artifact row differs for key {key!r}; changed_fields={changed_fields!r}"
        )

    canonical_digest = _canonical_digest(canonical)
    regenerated_digest = _canonical_digest(regenerated)
    if canonical_digest != regenerated_digest:
        raise ValueError("canonicalized artifact digests differ after row comparison")
    return {
        "rows": len(canonical),
        "semantic_equal": True,
        "canonicalized_sha256": canonical_digest,
        "row_order_ignored": True,
    }


def load_jsonl_rows(path: str) -> list[ArtifactRow]:
    """Load JSONL rows for semantic artifact comparison."""

    rows: list[ArtifactRow] = []
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: artifact row must be an object"
                )
            rows.append(row)
    return rows
