"""Topic loading helpers for TREC RAG experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Topic:
    id: str
    title: str
    narrative: str


def derive_title(text: str, max_words: int = 12) -> str:
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", " ".join(text.split()))
    if not words:
        return "Untitled topic"
    return " ".join(words[:max_words])


def _required_text(record: dict[str, object], key: str, line_number: int) -> str:
    if key not in record or record[key] is None:
        raise ValueError(f"line {line_number}: missing required field {key!r}")
    value = " ".join(str(record[key]).split())
    if not value:
        raise ValueError(f"line {line_number}: empty required field {key!r}")
    return value


def _load_jsonl_topics(path: Path) -> list[Topic]:
    topics: list[Topic] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON topic record") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number}: topic record must be an object")
        topics.append(
            Topic(
                id=_required_text(record, "id", line_number),
                title=_required_text(record, "title", line_number),
                narrative=_required_text(record, "narrative", line_number),
            )
        )
    return topics


def _load_tsv_topics(path: Path, title_words: int) -> list[Topic]:
    topics: list[Topic] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        if "\t" not in raw_line:
            raise ValueError(f"line {line_number}: expected qid<TAB>text")
        qid, text = raw_line.split("\t", 1)
        qid = " ".join(qid.split())
        narrative = " ".join(text.split())
        if not qid:
            raise ValueError(f"line {line_number}: empty qid")
        if not narrative:
            raise ValueError(f"line {line_number}: empty topic text")
        topics.append(
            Topic(
                id=qid,
                title=derive_title(narrative, max_words=title_words),
                narrative=narrative,
            )
        )
    return topics


def load_topics(
    path: Path,
    *,
    title_words: int = 12,
    topic_format: str | None = None,
) -> list[Topic]:
    normalized_format = topic_format.strip().lower() if topic_format else None
    if normalized_format and normalized_format not in {"jsonl", "tsv"}:
        raise ValueError(f"unsupported topic format: {topic_format}")

    if normalized_format == "tsv" or (normalized_format is None and path.suffix.lower() == ".tsv"):
        topics = _load_tsv_topics(path, title_words)
    else:
        topics = _load_jsonl_topics(path)
    if not topics:
        raise ValueError(f"{path} did not contain any topics")
    return topics


def write_topics_jsonl(topics: Iterable[Topic], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as sink:
        for topic in topics:
            sink.write(
                json.dumps(
                    {
                        "id": topic.id,
                        "title": topic.title,
                        "narrative": topic.narrative,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count
