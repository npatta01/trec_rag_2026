"""Typed records exchanged between RAG pipeline stages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryVariant:
    topic_id: str
    variant_name: str
    query_text: str
    source_type: str


@dataclass(frozen=True)
class RetrievedCandidate:
    topic_id: str
    variant_name: str
    retriever_name: str
    query_text: str
    docid: str
    rank: int
    score: float
    text: str


@dataclass(frozen=True)
class RankedCandidate:
    topic_id: str
    docid: str
    rank: int
    score: float
    text: str
    provenance: list[dict[str, Any]]


@dataclass(frozen=True)
class EvidenceRecord:
    topic_id: str
    docid: str
    citation_index: int
    rank: int
    score: float
    text: str
    provenance: list[dict[str, Any]]


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_jsonl(records: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
