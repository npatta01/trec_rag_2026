"""Freeze deterministic exact evidence units for adaptive obligation search v2.

This stage is deliberately read-only with respect to the authenticated Task 1
contract and Task 3 base scores.  It performs no retrieval, model, tokenizer,
inference, discovery, or qrels work.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .adaptive_evidence_score import load_score_contract, verify_local_scoring


SCHEMA_VERSION = "adaptive-obligation-v2-contract-v1"
PARENT_SCHEMA_VERSION = "adaptive-obligation-v2-parent-v1"
RESERVOIR_SCHEMA_VERSION = "adaptive-obligation-v2-reservoir-v1"
UNIT_SCHEMA_VERSION = "adaptive-obligation-v2-unit-v1"
SOURCE_CONTRACT_SCHEMA_VERSION = "adaptive-evidence-contract-v1"
EXPECTED_DOCUMENT_COUNT = 8_114
EXPECTED_BROAD_COUNT = 4
EXPECTED_O0_COUNT = 24
EXPECTED_WINDOW_COUNT = 98_053
EXPECTED_PAIR_COUNT = 96_911
EXPECTED_SHARD_COUNT = 28
EXPECTED_RESERVOIR_COUNT = 48
RESERVOIR_DOCUMENT_LIMIT = 10

_OUTPUT_NAMES = frozenset(
    {"manifest.json", "parents.jsonl", "reservoirs.jsonl", "units.jsonl", "receipt.json"}
)
_ZERO_COUNTERS = (
    "network_call_count",
    "retrieval_call_count",
    "hosted_inference_call_count",
    "paid_call_count",
    "model_load_count",
    "tokenizer_load_count",
    "inference_count",
)
_UNIT_RE = re.compile(
    r"(?m)(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+[^\n]+"
    r"|[^.!?\n]+(?:[.!?]+|$)"
)
_HEX_RE = re.compile(r"[0-9a-f]{64}")

T = TypeVar("T")


def canonical_sha256(value: object) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reject_protected_before_access(
    topic_ids: Iterable[object], source_loader: Callable[[], T]
) -> T:
    for topic_id in map(str, topic_ids):
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
    return source_loader()


def document_fold(topic_id: str, document_id: str) -> int:
    reject_protected_before_access([topic_id], lambda: None)
    digest = hashlib.sha256(f"{topic_id}\0{document_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 2


def sentence_and_list_item_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _UNIT_RE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def split_exact_units(
    *,
    topic_id: str,
    parent_id: str,
    fold: int,
    document_id: str,
    window_id: str,
    text: str,
) -> list[dict[str, object]]:
    if not isinstance(text, str):
        raise ValueError("unit source text must be text")
    if fold not in (0, 1) or isinstance(fold, bool):
        raise ValueError("unit fold must be exactly 0 or 1")
    rows: list[dict[str, object]] = []
    for start, end in sentence_and_list_item_spans(text):
        exact = text[start:end]
        identity = {
            "topic_id": topic_id,
            "parent_id": parent_id,
            "fold": fold,
            "document_id": document_id,
            "window_id": window_id,
            "start": start,
            "end": end,
            "text": exact,
        }
        rows.append(
            {
                "schema_version": UNIT_SCHEMA_VERSION,
                "unit_id": canonical_sha256(identity),
                **identity,
                "text_sha256": sha256_text(exact),
            }
        )
    return rows


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _HEX_RE.fullmatch(value) is not None


def _compact_bytes(value: object) -> bytes:
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


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_compact_bytes(dict(row)) for row in rows)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        source = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    return _parse_json_bytes(source, label)


def _parse_json_bytes(source: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    try:
        source = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    return _parse_jsonl_bytes(source, label)


def _parse_jsonl_bytes(source: bytes, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line:
            raise ValueError(f"{label}:{line_number} is blank")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label}:{line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_number} must be a JSON object")
        if _compact_bytes(row).rstrip(b"\n") != line:
            raise ValueError(f"{label}:{line_number} is not canonical JSON")
        rows.append(row)
    return rows


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    label: str
    content: bytes
    device: int
    inode: int
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _snapshot_regular_file(path: Path, label: str) -> _FileSnapshot:
    """Read one immutable no-follow snapshot and bind its descriptor identity."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("source snapshot O_NOFOLLOW is unavailable")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be a readable non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        content = b"".join(chunks)
        if _stat_identity(before) != _stat_identity(after) or len(content) != before.st_size:
            raise ValueError(f"{label} changed while its snapshot was read")
        return _FileSnapshot(
            path=Path(path),
            label=label,
            content=content,
            device=before.st_dev,
            inode=before.st_ino,
            size=before.st_size,
            mode=before.st_mode,
            mtime_ns=before.st_mtime_ns,
            ctime_ns=before.st_ctime_ns,
        )
    finally:
        os.close(descriptor)


def _assert_snapshot_current(snapshot: _FileSnapshot) -> None:
    """Detect path replacement or in-place rewrite after deep verification."""

    try:
        observed = os.lstat(snapshot.path)
    except OSError as exc:
        raise ValueError(f"{snapshot.label} changed after snapshot") from exc
    expected = (
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mode,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )
    if _stat_identity(observed) != expected or not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{snapshot.label} changed after snapshot")


def _require_snapshot_binding(
    snapshot: _FileSnapshot,
    binding: Mapping[str, object],
    *,
    expected_rows: int | None = None,
) -> None:
    if binding.get("sha256") != _sha256_bytes(snapshot.content):
        raise ValueError(f"{snapshot.label} hash differs from authenticated receipt")
    if "bytes" in binding and binding.get("bytes") != len(snapshot.content):
        raise ValueError(f"{snapshot.label} byte count differs from authenticated receipt")
    if expected_rows is not None and binding.get("rows") != expected_rows:
        raise ValueError(f"{snapshot.label} row count differs from authenticated receipt")


def _mapping(data: object, name: str) -> dict[str, object]:
    if not isinstance(data, Mapping) or not isinstance(data.get(name), Mapping):
        raise ValueError(f"v2 source {name} must be an object")
    return dict(data[name])  # type: ignore[index]


def _records(data: object, name: str) -> list[dict[str, object]]:
    if not isinstance(data, Mapping):
        raise ValueError("v2 source must be an object")
    value = data.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"v2 source {name} must be an array")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"v2 source {name} must contain objects")
    return [dict(row) for row in value]  # type: ignore[arg-type]


def _topic_ids(data: object) -> list[str]:
    if not isinstance(data, Mapping):
        raise ValueError("v2 source must be an object")
    raw = data.get("topic_ids")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("v2 source topic_ids must be an array")
    topic_ids = [str(value) for value in raw]
    reject_protected_before_access(topic_ids, lambda: None)
    if len(topic_ids) != len(PILOT_TOPIC_IDS) or set(topic_ids) != set(PILOT_TOPIC_IDS):
        raise ValueError(
            "v2 source topic set must contain exactly the four pilot topics "
            f"{list(PILOT_TOPIC_IDS)}"
        )
    return list(PILOT_TOPIC_IDS)


def _source_bindings(data: object) -> dict[str, object]:
    bindings = _mapping(data, "source_bindings")
    common = {
        "mode",
        "contract_summary_sha256",
        "base_score_receipt_sha256",
        "document_count",
        "broad_obligation_count",
        "o0_obligation_count",
        "window_count",
        "pair_count",
        "shard_count",
    }
    mode = bindings.get("mode")
    if mode == "fixture_only":
        required = common
    elif mode == "authenticated_paths":
        required = {*common, "contract_dir", "base_scores_dir"}
    else:
        raise ValueError("v2 source mode must be fixture_only or authenticated_paths")
    if set(bindings) != required or any("discovery" in name for name in bindings):
        raise ValueError("v2 source bindings fields differ or include forbidden discovery")
    if not _is_sha256(bindings.get("contract_summary_sha256")) or not _is_sha256(
        bindings.get("base_score_receipt_sha256")
    ):
        raise ValueError("v2 source receipt hash is invalid")
    for name in (
        "document_count",
        "broad_obligation_count",
        "o0_obligation_count",
        "window_count",
        "pair_count",
        "shard_count",
    ):
        value = bindings.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"v2 source {name} count is invalid")
    if bindings["broad_obligation_count"] != EXPECTED_BROAD_COUNT:
        raise ValueError("v2 source must contain exactly four broad obligations")
    if bindings["o0_obligation_count"] != EXPECTED_O0_COUNT:
        raise ValueError("v2 source must contain exactly 24 O0 parents")
    if bindings["shard_count"] != EXPECTED_SHARD_COUNT:
        raise ValueError("v2 source must bind exactly 28 score shards")
    if mode == "authenticated_paths" and any(
        not isinstance(bindings.get(name), str) or not bindings[name]
        for name in ("contract_dir", "base_scores_dir")
    ):
        raise ValueError("authenticated v2 source paths are invalid")
    return bindings


def _normalize_parents(
    obligations: Sequence[Mapping[str, object]], topic_ids: Sequence[str]
) -> tuple[list[dict[str, object]], set[str]]:
    topic_order = {topic_id: index for index, topic_id in enumerate(topic_ids)}
    seen: set[str] = set()
    broad_ids: set[str] = set()
    broad_topics: list[str] = []
    parents: list[dict[str, object]] = []
    broad_count = 0
    for source in obligations:
        row = dict(source)
        topic_id = row.get("topic_id")
        obligation_id = row.get("obligation_id")
        kind = row.get("kind")
        if not isinstance(topic_id, str):
            raise ValueError("obligation topic identity is invalid")
        reject_protected_before_access([topic_id], lambda: None)
        if topic_id not in topic_order or not isinstance(obligation_id, str) or not obligation_id:
            raise ValueError("obligation identity is outside the pilot contract")
        if obligation_id in seen:
            raise ValueError("obligations contain a duplicate identity")
        seen.add(obligation_id)
        if kind == "broad":
            broad_count += 1
            broad_ids.add(obligation_id)
            broad_topics.append(topic_id)
            continue
        if kind != "o0":
            raise ValueError("Task 1 contract may contain only broad and O0 obligations")
        order = row.get("manifest_order")
        text = row.get("text")
        query = row.get("query")
        if (
            isinstance(order, bool)
            or not isinstance(order, int)
            or order < 0
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(query, str)
            or not query.strip()
            or row.get("source_facet_id") not in (None, obligation_id)
        ):
            raise ValueError("O0 parent schema is invalid")
        parents.append(
            {
                "schema_version": PARENT_SCHEMA_VERSION,
                "topic_id": topic_id,
                "parent_id": obligation_id,
                "manifest_order": order,
                "text": text,
                "text_sha256": sha256_text(text),
                "query": query,
                "query_sha256": sha256_text(query),
            }
        )
    if (
        broad_count != EXPECTED_BROAD_COUNT
        or len(set(broad_topics)) != EXPECTED_BROAD_COUNT
        or set(broad_topics) != set(topic_ids)
    ):
        raise ValueError(
            "v2 source must contain exactly one broad obligation per pilot topic"
        )
    if len(parents) != EXPECTED_O0_COUNT:
        raise ValueError("v2 source must contain exactly 24 O0 parents")
    if len({int(row["manifest_order"]) for row in parents}) != len(parents):
        raise ValueError("O0 parent manifest orders must be distinct")
    parents.sort(
        key=lambda row: (
            topic_order[str(row["topic_id"])],
            int(row["manifest_order"]),
            str(row["parent_id"]),
        )
    )
    return parents, broad_ids


def _normalize_documents(
    rows: Sequence[Mapping[str, object]], topic_ids: Sequence[str]
) -> dict[tuple[str, str], dict[str, object]]:
    allowed = set(topic_ids)
    documents: dict[tuple[str, str], dict[str, object]] = {}
    for source in rows:
        row = dict(source)
        topic_id = row.get("topic_id")
        document_id = row.get("document_id")
        text = row.get("text")
        fold = row.get("fold")
        if not isinstance(topic_id, str):
            raise ValueError("document topic identity is invalid")
        reject_protected_before_access([topic_id], lambda: None)
        if (
            topic_id not in allowed
            or not isinstance(document_id, str)
            or not document_id
            or not isinstance(text, str)
            or row.get("text_sha256") != sha256_text(text)
        ):
            raise ValueError("document identity or text hash differs")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold not in (0, 1):
            raise ValueError("document fold must be exactly 0 or 1")
        if fold != document_fold(topic_id, document_id):
            raise ValueError("document fold differs from deterministic assignment")
        key = (topic_id, document_id)
        if key in documents:
            raise ValueError("documents contain a duplicate topic-document identity")
        documents[key] = row
    if {key[0] for key in documents} != allowed:
        raise ValueError("document topics differ from the four pilot topics")
    return documents


def _finite_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("MiniLM score must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("MiniLM score is nonfinite")
    return score


def _score_rank_key(row: Mapping[str, object]) -> tuple[float, int, int, str, str]:
    return (
        -float(row["score"]),
        int(row["document_start_token"]),
        int(row["document_end_token"]),
        str(row["window_id"]),
        str(row["document_id"]),
    )


def _best_parent_document_windows(
    score_rows: Sequence[Mapping[str, object]],
    *,
    parents: Sequence[Mapping[str, object]],
    broad_ids: set[str],
    documents: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[tuple[str, str, str], dict[str, object]]:
    parent_topics = {str(row["parent_id"]): str(row["topic_id"]) for row in parents}
    known_obligations = {*parent_topics, *broad_ids}
    best: dict[tuple[str, str, str], dict[str, object]] = {}
    seen_windows: set[tuple[str, str, str, str]] = set()
    for source in score_rows:
        row = dict(source)
        topic_id = row.get("topic_id")
        parent_id = row.get("variant", row.get("obligation_id"))
        document_id = row.get("document_id")
        if not isinstance(topic_id, str):
            raise ValueError("score topic identity is invalid")
        reject_protected_before_access([topic_id], lambda: None)
        if (
            not isinstance(parent_id, str)
            or parent_id not in known_obligations
            or not isinstance(document_id, str)
            or (topic_id, document_id) not in documents
        ):
            raise ValueError("score obligation or document identity is invalid")
        if parent_id in parent_topics and parent_topics[parent_id] != topic_id:
            raise ValueError("score parent topic differs")
        document = documents[(topic_id, document_id)]
        if row.get("document_sha256") != document.get("text_sha256"):
            raise ValueError("score document hash differs")
        window_id = row.get("window_id")
        window_text = row.get("window_text")
        start = row.get("document_start_token")
        end = row.get("document_end_token")
        if (
            not isinstance(window_id, str)
            or not window_id
            or not isinstance(window_text, str)
            or row.get("window_sha256") != sha256_text(window_text)
        ):
            raise ValueError("score window identity or hash differs")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end <= start
        ):
            raise ValueError("score window span is invalid")
        score = _finite_score(row.get("score"))
        window_key = (topic_id, parent_id, document_id, window_id)
        if window_key in seen_windows:
            raise ValueError("score rows contain a duplicate window identity")
        seen_windows.add(window_key)
        if parent_id in broad_ids:
            continue
        normalized = {
            "topic_id": topic_id,
            "parent_id": parent_id,
            "document_id": document_id,
            "document_sha256": document["text_sha256"],
            "fold": document["fold"],
            "window_id": window_id,
            "window_text": window_text,
            "window_sha256": row["window_sha256"],
            "document_start_token": start,
            "document_end_token": end,
            "score": score,
        }
        key = (topic_id, parent_id, document_id)
        if key not in best or _score_rank_key(normalized) < _score_rank_key(best[key]):
            best[key] = normalized
    return best


def build_v2_contract(data: object) -> dict[str, object]:
    """Build exact parent/fold reservoirs from already authenticated source rows."""

    topic_ids = _topic_ids(data)
    bindings = _source_bindings(data)
    obligations = _records(data, "obligations")
    document_rows = _records(data, "documents")
    score_rows = _records(data, "score_rows")
    if bindings["document_count"] != len(document_rows):
        raise ValueError("v2 source document count differs")
    if bindings["window_count"] != len(score_rows):
        raise ValueError("v2 source window count differs")
    score_pairs: set[tuple[str, str]] = set()
    for row in score_rows:
        query = row.get("query")
        window_text = row.get("window_text")
        if (
            not isinstance(query, str)
            or not isinstance(window_text, str)
            or row.get("query_sha256") != sha256_text(query)
        ):
            raise ValueError("score query identity or hash differs")
        score_pairs.add((query, window_text))
    if bindings["pair_count"] != len(score_pairs):
        raise ValueError("v2 source unique pair count differs")
    parents, broad_ids = _normalize_parents(obligations, topic_ids)
    if sum(row.get("kind") == "broad" for row in obligations) != bindings[
        "broad_obligation_count"
    ] or sum(row.get("kind") == "o0" for row in obligations) != bindings[
        "o0_obligation_count"
    ]:
        raise ValueError("v2 source obligation count differs")
    documents = _normalize_documents(document_rows, topic_ids)
    best = _best_parent_document_windows(
        score_rows,
        parents=parents,
        broad_ids=broad_ids,
        documents=documents,
    )

    reservoirs: list[dict[str, object]] = []
    units: list[dict[str, object]] = []
    for parent in parents:
        topic_id = str(parent["topic_id"])
        parent_id = str(parent["parent_id"])
        for fold in (0, 1):
            local = sorted(
                (
                    row
                    for (row_topic, row_parent, _document), row in best.items()
                    if row_topic == topic_id
                    and row_parent == parent_id
                    and row["fold"] == fold
                ),
                key=_score_rank_key,
            )
            if len(local) < RESERVOIR_DOCUMENT_LIMIT:
                raise ValueError(
                    f"parent {parent_id} fold {fold} has fewer than ten distinct documents"
                )
            selected = local[:RESERVOIR_DOCUMENT_LIMIT]
            reservoir_documents: list[dict[str, object]] = []
            for rank, selected_row in enumerate(selected, start=1):
                window_text = str(selected_row["window_text"])
                exact_units = split_exact_units(
                    topic_id=topic_id,
                    parent_id=parent_id,
                    fold=fold,
                    document_id=str(selected_row["document_id"]),
                    window_id=str(selected_row["window_id"]),
                    text=window_text,
                )
                if not exact_units:
                    raise ValueError("selected reservoir window has no nonempty exact units")
                for unit in exact_units:
                    start, end = int(unit["start"]), int(unit["end"])
                    if window_text[start:end] != unit["text"]:
                        raise ValueError("unit span differs from immutable window text")
                units.extend(exact_units)
                reservoir_documents.append(
                    {
                        "rank": rank,
                        "document_id": selected_row["document_id"],
                        "document_sha256": selected_row["document_sha256"],
                        "window_id": selected_row["window_id"],
                        "window_text": window_text,
                        "window_sha256": selected_row["window_sha256"],
                        "document_start_token": selected_row["document_start_token"],
                        "document_end_token": selected_row["document_end_token"],
                        "score": selected_row["score"],
                        "unit_ids": [row["unit_id"] for row in exact_units],
                    }
                )
            document_ids = [str(row["document_id"]) for row in reservoir_documents]
            window_ids = [str(row["window_id"]) for row in reservoir_documents]
            reservoir_id = canonical_sha256(
                {
                    "topic_id": topic_id,
                    "parent_id": parent_id,
                    "fold": fold,
                    "document_ids": document_ids,
                    "window_ids": window_ids,
                }
            )
            reservoirs.append(
                {
                    "schema_version": RESERVOIR_SCHEMA_VERSION,
                    "reservoir_id": reservoir_id,
                    "topic_id": topic_id,
                    "parent_id": parent_id,
                    "fold": fold,
                    "document_count": RESERVOIR_DOCUMENT_LIMIT,
                    "document_ids": document_ids,
                    "window_ids": window_ids,
                    "documents": reservoir_documents,
                }
            )
    if len(reservoirs) != EXPECTED_RESERVOIR_COUNT:
        raise ValueError("v2 contract must contain exactly 48 parent/fold reservoirs")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "parent_count": len(parents),
        "reservoir_count": len(reservoirs),
        "reservoir_document_limit": RESERVOIR_DOCUMENT_LIMIT,
        "reservoir_document_slot_count": sum(
            int(row["document_count"]) for row in reservoirs
        ),
        "unit_count": len(units),
        "protected_topic_count": 0,
        "source_bindings": bindings,
        "qrels_opened": False,
        **{name: 0 for name in _ZERO_COUNTERS},
        "external_cost_usd": 0.0,
    }
    return {
        "manifest": manifest,
        "parents": parents,
        "reservoirs": reservoirs,
        "units": units,
        "source_bindings": bindings,
    }


def _artifact(path: str, content: bytes, rows: int) -> dict[str, object]:
    return {
        "path": path,
        "rows": rows,
        "bytes": len(content),
        "sha256": _sha256_bytes(content),
    }


def _publication_bytes(
    contract: Mapping[str, object],
) -> tuple[dict[str, bytes], dict[str, object]]:
    parents = contract.get("parents")
    reservoirs = contract.get("reservoirs")
    units = contract.get("units")
    raw_manifest = contract.get("manifest")
    if (
        not isinstance(parents, list)
        or not isinstance(reservoirs, list)
        or not isinstance(units, list)
        or not isinstance(raw_manifest, Mapping)
    ):
        raise ValueError("v2 contract publication data is invalid")
    parents_bytes = _jsonl_bytes(parents)
    reservoirs_bytes = _jsonl_bytes(reservoirs)
    units_bytes = _jsonl_bytes(units)
    data_artifacts = {
        "parents.jsonl": _artifact("parents.jsonl", parents_bytes, len(parents)),
        "reservoirs.jsonl": _artifact(
            "reservoirs.jsonl", reservoirs_bytes, len(reservoirs)
        ),
        "units.jsonl": _artifact("units.jsonl", units_bytes, len(units)),
    }
    manifest = {**dict(raw_manifest), "artifacts": data_artifacts}
    manifest_bytes = _pretty_bytes(manifest)
    all_artifacts = {
        "manifest.json": _artifact("manifest.json", manifest_bytes, 1),
        **data_artifacts,
    }
    receipt = {
        **dict(raw_manifest),
        "artifacts": all_artifacts,
    }
    contents = {
        "manifest.json": manifest_bytes,
        "parents.jsonl": parents_bytes,
        "reservoirs.jsonl": reservoirs_bytes,
        "units.jsonl": units_bytes,
        "receipt.json": _pretty_bytes(receipt),
    }
    return contents, receipt


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _write_fsynced(path: Path, content: bytes) -> None:
    with Path(path).open("xb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic create-only rename is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(f"create-only v2 contract output exists: {destination}")
    raise OSError(error, os.strerror(error), str(destination))


def _require_output_inventory(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("v2 contract output inventory root must not be a symlink")
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise ValueError("v2 contract output inventory is unreadable") from exc
    if {entry.name for entry in entries} != _OUTPUT_NAMES or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("v2 contract output inventory is partial, unexpected, or unsafe")


def _publish_v2_contract(
    contract: Mapping[str, object], output_dir: Path
) -> dict[str, object]:
    """Publish five artifacts atomically and create-only via a sibling staging dir."""

    destination = Path(output_dir)
    if _path_present(destination):
        raise FileExistsError(f"create-only v2 contract output exists: {destination}")
    if destination.parent.is_symlink():
        raise ValueError("v2 contract output parent must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("v2 contract output parent must not be a symlink")
    contents, receipt = _publication_bytes(contract)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=str(destination.parent)
        )
    )
    published = False
    try:
        for name in (
            "parents.jsonl",
            "reservoirs.jsonl",
            "units.jsonl",
            "manifest.json",
            "receipt.json",
        ):
            _write_fsynced(staging / name, contents[name])
        _require_output_inventory(staging)
        for name, expected in contents.items():
            if (staging / name).read_bytes() != expected:
                raise ValueError(f"staged {name} byte binding differs")
        _fsync_directory(staging)
        _rename_noreplace(staging, destination)
        published = True
        _fsync_directory(destination.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return receipt


def _safe_source_receipt(receipt: Mapping[str, object], label: str) -> None:
    if (
        receipt.get("qrels_opened") is not False
        or receipt.get("network_call_count") != 0
        or receipt.get("retrieval_call_count") != 0
        or receipt.get("hosted_inference_call_count") != 0
        or receipt.get("paid_call_count") != 0
        or receipt.get("external_cost_usd") != 0.0
    ):
        raise ValueError(f"{label} safety receipt differs")


def load_authenticated_v2_sources(
    contract_dir: Path, scores_dir: Path
) -> dict[str, object]:
    """Authenticate and consume Task 1/3 from immutable byte snapshots."""

    contract_root, scores_root = Path(contract_dir), Path(scores_dir)
    for root, label in ((contract_root, "contract"), (scores_root, "scores")):
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"authenticated {label} root must be a regular directory")

    manifest_snapshot = _snapshot_regular_file(
        contract_root / "manifest.json", "contract manifest"
    )
    summary_snapshot = _snapshot_regular_file(
        contract_root / "summary.json", "contract summary"
    )
    receipt_snapshot = _snapshot_regular_file(
        scores_root / "receipt.json", "base score receipt"
    )
    manifest = _parse_json_bytes(manifest_snapshot.content, manifest_snapshot.label)
    raw_topics = manifest.get("topic_ids")
    if not isinstance(raw_topics, list):
        raise ValueError("contract manifest topic_ids must be an array")
    topics = [str(value) for value in raw_topics]
    reject_protected_before_access(topics, lambda: None)
    if (
        topics != list(PILOT_TOPIC_IDS)
        or manifest.get("schema_version") != SOURCE_CONTRACT_SCHEMA_VERSION
        or manifest.get("document_count") != EXPECTED_DOCUMENT_COUNT
        or manifest.get("broad_obligation_count") != EXPECTED_BROAD_COUNT
        or manifest.get("o0_obligation_count") != EXPECTED_O0_COUNT
        or manifest.get("qrels_opened") is not False
    ):
        raise ValueError("contract manifest canonical counts or topics differ")
    summary = _parse_json_bytes(summary_snapshot.content, summary_snapshot.label)
    summary_topics = summary.get("topic_ids")
    if not isinstance(summary_topics, list):
        raise ValueError("contract summary topic_ids must be an array")
    reject_protected_before_access(summary_topics, lambda: None)
    if (
        summary_topics != list(PILOT_TOPIC_IDS)
        or summary.get("schema_version") != SOURCE_CONTRACT_SCHEMA_VERSION
        or summary.get("status") != "complete"
        or summary.get("document_count") != EXPECTED_DOCUMENT_COUNT
        or summary.get("broad_obligation_count") != EXPECTED_BROAD_COUNT
        or summary.get("o0_obligation_count") != EXPECTED_O0_COUNT
        or summary.get("protected_topic_count") != 0
        or summary.get("qrels_opened") is not False
    ):
        raise ValueError("contract summary canonical counts or safety differ")
    summary_sha256 = _sha256_bytes(summary_snapshot.content)

    score_receipt = _parse_json_bytes(receipt_snapshot.content, receipt_snapshot.label)
    raw_shards = score_receipt.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("base score shard metadata is missing")
    shard_topics: list[str] = []
    shard_rows = 0
    shard_queues: set[tuple[str, str]] = set()
    shard_paths: set[str] = set()
    for raw in raw_shards:
        if not isinstance(raw, Mapping):
            raise ValueError("base score shard metadata is invalid")
        topic_id = raw.get("topic_id")
        obligation_id = raw.get("obligation_id")
        relative = raw.get("path")
        if not isinstance(topic_id, str) or not topic_id:
            raise ValueError("base score shard topic identity is invalid")
        reject_protected_before_access([topic_id], lambda: None)
        if topic_id not in PILOT_TOPIC_IDS:
            raise ValueError("base score shard topic coverage differs")
        if not isinstance(obligation_id, str) or not obligation_id:
            raise ValueError("base score shard queue identity is invalid")
        queue = (topic_id, obligation_id)
        if queue in shard_queues:
            raise ValueError("base score receipt has a duplicate shard queue identity")
        shard_queues.add(queue)
        if (
            not isinstance(relative, str)
            or not relative.startswith("shards/")
            or Path(relative).parts[:1] != ("shards",)
            or len(Path(relative).parts) != 2
        ):
            raise ValueError("base score shard path is invalid")
        if relative in shard_paths:
            raise ValueError("base score receipt has a duplicate shard path")
        shard_paths.add(relative)
        if not _is_sha256(raw.get("sha256")):
            raise ValueError("base score shard hash binding is invalid")
        byte_count = raw.get("bytes")
        rows = raw.get("rows")
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 0
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError("base score shard row or byte count is invalid")
        shard_topics.append(topic_id)
        shard_rows += rows
    _safe_source_receipt(score_receipt, "base score")
    if (
        set(shard_topics) != set(PILOT_TOPIC_IDS)
        or len(shard_queues) != EXPECTED_SHARD_COUNT
        or score_receipt.get("completed_window_count") != EXPECTED_WINDOW_COUNT
        or score_receipt.get("unique_pair_count") != EXPECTED_PAIR_COUNT
        or score_receipt.get("shard_count") != EXPECTED_SHARD_COUNT
        or len(raw_shards) != EXPECTED_SHARD_COUNT
        or shard_rows != EXPECTED_WINDOW_COUNT
    ):
        raise ValueError("base score topic coverage or canonical counts differ")
    score_receipt_sha256 = _sha256_bytes(receipt_snapshot.content)

    # No large artifact or callback is reached until the complete small receipt
    # inventory above has exact pilot topics and 28 distinct queue identities.
    raw_artifacts = summary.get("artifact_sha256")
    expected_contract_names = {
        "manifest.json",
        "obligations.jsonl",
        "documents.jsonl",
        "folds.jsonl",
    }
    if not isinstance(raw_artifacts, Mapping) or set(raw_artifacts) != expected_contract_names:
        raise ValueError("contract summary artifact hash inventory differs")
    contract_snapshots: dict[str, _FileSnapshot] = {"manifest.json": manifest_snapshot}
    for name in ("obligations.jsonl", "documents.jsonl", "folds.jsonl"):
        contract_snapshots[name] = _snapshot_regular_file(
            contract_root / name, f"contract {name}"
        )
    for name, snapshot in contract_snapshots.items():
        expected_hash = raw_artifacts.get(name)
        if expected_hash != _sha256_bytes(snapshot.content):
            raise ValueError(f"contract {name} hash differs from authenticated summary")

    obligations = _parse_jsonl_bytes(
        contract_snapshots["obligations.jsonl"].content, "contract obligations snapshot"
    )
    documents = _parse_jsonl_bytes(
        contract_snapshots["documents.jsonl"].content, "contract documents snapshot"
    )
    # folds.jsonl is not consumed by v2.  Its exact bytes are authenticated by
    # the Task 1 summary above; the fold value v2 does consume comes from each
    # captured document row and is independently recomputed in build_v2_contract.
    _parse_jsonl_bytes(
        contract_snapshots["folds.jsonl"].content, "contract folds snapshot"
    )
    reject_protected_before_access(
        [row.get("topic_id") for row in obligations]
        + [row.get("topic_id") for row in documents],
        lambda: None,
    )
    if (
        len(documents) != EXPECTED_DOCUMENT_COUNT
        or sum(row.get("kind") == "broad" for row in obligations)
        != EXPECTED_BROAD_COUNT
        or sum(row.get("kind") == "o0" for row in obligations) != EXPECTED_O0_COUNT
    ):
        raise ValueError("captured contract row counts differ")

    shards_root = scores_root / "shards"
    if shards_root.is_symlink() or not shards_root.is_dir():
        raise ValueError("base score shards root must be a regular directory")
    shard_snapshots: list[_FileSnapshot] = []
    score_rows: list[dict[str, object]] = []
    for raw in raw_shards:
        assert isinstance(raw, Mapping)
        relative = str(raw["path"])
        snapshot = _snapshot_regular_file(
            scores_root / relative, f"base score shard {relative}"
        )
        parsed = _parse_jsonl_bytes(snapshot.content, snapshot.label)
        _require_snapshot_binding(snapshot, raw, expected_rows=len(parsed))
        topic_id, obligation_id = str(raw["topic_id"]), str(raw["obligation_id"])
        if any(
            row.get("topic_id") != topic_id
            or row.get("variant", row.get("obligation_id")) != obligation_id
            for row in parsed
        ):
            raise ValueError("captured score shard queue rows differ from receipt")
        shard_snapshots.append(snapshot)
        score_rows.extend(parsed)
    if len(score_rows) != EXPECTED_WINDOW_COUNT:
        raise ValueError("captured score shard population differs")
    reject_protected_before_access(
        [row.get("topic_id") for row in score_rows], lambda: None
    )
    pairs: set[tuple[str, str]] = set()
    for row in score_rows:
        query, window_text = row.get("query"), row.get("window_text")
        if not isinstance(query, str) or not isinstance(window_text, str):
            raise ValueError("captured score pair identity is invalid")
        pairs.add((query, window_text))
    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise ValueError("captured score unique pair count differs")

    # Deep verifiers may reopen paths, so their complete semantic returns must
    # equal the captured snapshots.  For scores, that exact normalized receipt
    # comparison combines with each captured shard's hash/bytes/rows/queue
    # binding above to cover the score rows v2 consumes.
    verified_contract = load_score_contract(contract_root)
    expected_verified_contract = {
        "schema_version": SOURCE_CONTRACT_SCHEMA_VERSION,
        "obligations": obligations,
        "documents": documents,
    }
    if (
        not isinstance(verified_contract, Mapping)
        or dict(verified_contract) != expected_verified_contract
    ):
        raise ValueError("deep-validated contract rows differ from captured snapshot")
    for snapshot in (manifest_snapshot, summary_snapshot, *contract_snapshots.values()):
        _assert_snapshot_current(snapshot)
    verified_scores = verify_local_scoring(scores_root)
    expected_verified_scores = dict(score_receipt)
    expected_verified_scores.pop("restored_cache_pair_count", None)
    expected_verified_scores.pop("resumed_shard_count", None)
    if (
        not isinstance(verified_scores, Mapping)
        or dict(verified_scores) != expected_verified_scores
    ):
        raise ValueError("deep-validated score receipt differs from captured snapshot")
    _safe_source_receipt(verified_scores, "verified base score")
    if (
        verified_scores.get("completed_window_count") != EXPECTED_WINDOW_COUNT
        or verified_scores.get("unique_pair_count") != EXPECTED_PAIR_COUNT
        or verified_scores.get("shard_count") != EXPECTED_SHARD_COUNT
    ):
        raise ValueError("verified base score canonical counts differ")
    seen_snapshots: set[Path] = set()
    for snapshot in (
        manifest_snapshot,
        summary_snapshot,
        receipt_snapshot,
        *contract_snapshots.values(),
        *shard_snapshots,
    ):
        if snapshot.path not in seen_snapshots:
            _assert_snapshot_current(snapshot)
            seen_snapshots.add(snapshot.path)
    return {
        "topic_ids": list(PILOT_TOPIC_IDS),
        "obligations": obligations,
        "documents": documents,
        "score_rows": score_rows,
        "source_bindings": {
            "mode": "authenticated_paths",
            "contract_dir": str(contract_root.resolve()),
            "contract_summary_sha256": summary_sha256,
            "base_scores_dir": str(scores_root.resolve()),
            "base_score_receipt_sha256": score_receipt_sha256,
            "document_count": EXPECTED_DOCUMENT_COUNT,
            "broad_obligation_count": EXPECTED_BROAD_COUNT,
            "o0_obligation_count": EXPECTED_O0_COUNT,
            "window_count": EXPECTED_WINDOW_COUNT,
            "pair_count": EXPECTED_PAIR_COUNT,
            "shard_count": EXPECTED_SHARD_COUNT,
        },
    }


def _verify_artifact_bindings(
    root: Path, receipt: Mapping[str, object]
) -> dict[str, bytes]:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _OUTPUT_NAMES - {
        "receipt.json"
    }:
        raise ValueError("v2 contract artifact bindings differ")
    contents: dict[str, bytes] = {}
    for name in sorted(artifacts):
        binding = artifacts[name]
        if not isinstance(binding, Mapping) or binding.get("path") != name:
            raise ValueError("v2 contract artifact path binding is invalid")
        content = (root / name).read_bytes()
        if (
            binding.get("bytes") != len(content)
            or binding.get("sha256") != _sha256_bytes(content)
        ):
            raise ValueError(f"v2 contract {name} hash or byte count differs")
        contents[name] = content
    return contents


def _verify_v2_contract(
    output_dir: Path,
    *,
    required_mode: str,
    fixture_source: object | None = None,
) -> dict[str, object]:
    root = Path(output_dir)
    _require_output_inventory(root)
    receipt = _read_json(root / "receipt.json", "v2 contract receipt")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or receipt.get("parent_count") != EXPECTED_O0_COUNT
        or receipt.get("reservoir_count") != EXPECTED_RESERVOIR_COUNT
        or receipt.get("reservoir_document_limit") != RESERVOIR_DOCUMENT_LIMIT
        or receipt.get("reservoir_document_slot_count")
        != EXPECTED_RESERVOIR_COUNT * RESERVOIR_DOCUMENT_LIMIT
        or receipt.get("protected_topic_count") != 0
        or receipt.get("qrels_opened") is not False
        or any(receipt.get(name) != 0 for name in _ZERO_COUNTERS)
        or receipt.get("external_cost_usd") != 0.0
    ):
        raise ValueError("v2 contract receipt counts or safety counters differ")
    bindings = receipt.get("source_bindings")
    if not isinstance(bindings, Mapping) or bindings.get("mode") != required_mode:
        raise ValueError(f"v2 verification requires exact {required_mode} sources")
    contents = _verify_artifact_bindings(root, receipt)
    parents = _read_jsonl(root / "parents.jsonl", "v2 parents")
    reservoirs = _read_jsonl(root / "reservoirs.jsonl", "v2 reservoirs")
    units = _read_jsonl(root / "units.jsonl", "v2 units")
    if (
        not isinstance(receipt.get("unit_count"), int)
        or receipt.get("unit_count") != len(units)
        or len(parents) != EXPECTED_O0_COUNT
        or len(reservoirs) != EXPECTED_RESERVOIR_COUNT
    ):
        raise ValueError("v2 contract artifact row counts differ")
    manifest = _read_json(root / "manifest.json", "v2 manifest")
    data_bindings = {
        name: dict(value)
        for name, value in receipt["artifacts"].items()  # type: ignore[union-attr]
        if name != "manifest.json"
    }
    expected_manifest = {
        **{key: value for key, value in receipt.items() if key != "artifacts"},
        "artifacts": data_bindings,
    }
    if manifest != expected_manifest or contents["manifest.json"] != _pretty_bytes(manifest):
        raise ValueError("v2 contract manifest differs from receipt")

    if required_mode == "fixture_only":
        if fixture_source is None:
            raise ValueError("fixture verification requires exact source data")
        source = fixture_source
    else:
        contract_path = bindings.get("contract_dir")
        scores_path = bindings.get("base_scores_dir")
        if not isinstance(contract_path, str) or not isinstance(scores_path, str):
            raise ValueError("authenticated v2 source paths are invalid")
        source = load_authenticated_v2_sources(
            Path(contract_path), Path(scores_path)
        )
    expected_contract = build_v2_contract(source)
    if expected_contract["source_bindings"] != dict(bindings):
        raise ValueError("v2 authenticated source bindings differ")
    expected_contents, expected_receipt = _publication_bytes(expected_contract)
    observed_contents = {
        name: (root / name).read_bytes() for name in _OUTPUT_NAMES
    }
    if observed_contents != expected_contents or receipt != expected_receipt:
        raise ValueError("v2 contract differs from deterministic source reconstruction")
    return receipt


def _verify_fixture_v2_contract(
    output_dir: Path, *, source: object
) -> dict[str, object]:
    return _verify_v2_contract(
        output_dir, required_mode="fixture_only", fixture_source=source
    )


def verify_v2_contract(output_dir: Path) -> dict[str, object]:
    """Verify and reconstruct only from the two authenticated bound sources."""

    return _verify_v2_contract(output_dir, required_mode="authenticated_paths")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="freeze the v2 evidence contract")
    build.add_argument("--contract", type=Path, required=True)
    build.add_argument("--scores", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify the v2 evidence contract")
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        if _path_present(args.output):
            raise FileExistsError(
                f"create-only v2 contract output exists: {args.output}"
            )
        source = load_authenticated_v2_sources(args.contract, args.scores)
        receipt = _publish_v2_contract(build_v2_contract(source), args.output)
        print(
            "status=complete "
            f"parents={receipt['parent_count']} "
            f"reservoirs={receipt['reservoir_count']} "
            f"slots={receipt['reservoir_document_slot_count']} "
            f"units={receipt['unit_count']} protected=0 "
            "network=false model_loads=0 inference=0 qrels_opened=false"
        )
    else:
        receipt = verify_v2_contract(args.output)
        print(
            "status=verified "
            f"parents={receipt['parent_count']} "
            f"reservoirs={receipt['reservoir_count']} "
            f"slots={receipt['reservoir_document_slot_count']} "
            f"units={receipt['unit_count']} protected=0 "
            "network=false model_loads=0 inference=0 qrels_opened=false"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
