"""Prepare byte-faithful one-topic inputs for organizer Pi baselines."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class OrganizerTopic:
    topic_id: str
    narrative: str


def select_topic(query_path: Path, topic_id: str) -> OrganizerTopic:
    """Return the single non-empty narrative selected from a query TSV."""
    matches = []
    with Path(query_path).open(encoding="utf-8", newline="") as source:
        for fields in csv.reader(source, delimiter="\t"):
            if fields and fields[0].strip() == topic_id:
                matches.append(OrganizerTopic(topic_id, "\t".join(fields[1:]).strip()))
    if len(matches) != 1 or not matches[0].narrative:
        raise ValueError(f"expected exactly one non-empty topic {topic_id}")
    return matches[0]


def _atomic_write(path: Path, body: bytes) -> Path:
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(body)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def write_topic_tsv(topic: OrganizerTopic, path: Path) -> Path:
    """Write an organizer query TSV containing exactly one selected topic."""
    return _atomic_write(Path(path), f"{topic.topic_id}\t{topic.narrative}\n".encode("utf-8"))


def write_topic_run(
    run_path: Path, topic_id: str, path: Path, *, expected_depth: int
) -> Path:
    """Validate and write a sorted fixed-depth TREC run for one topic."""
    if expected_depth <= 0:
        raise ValueError("expected_depth must be positive")

    rows: list[tuple[int, str, bytes]] = []
    with Path(run_path).open("rb") as source:
        for raw_line in source.read().splitlines(keepends=True):
            fields = raw_line.rstrip(b"\r\n").decode("utf-8").split()
            if not fields or fields[0] != topic_id:
                continue
            if len(fields) != 6:
                raise ValueError(f"expected six columns for topic {topic_id}")
            try:
                rank = int(fields[3])
            except ValueError as error:
                raise ValueError(f"invalid rank for topic {topic_id}") from error
            if rank <= 0:
                raise ValueError(f"rank must be positive for topic {topic_id}")
            rows.append((rank, fields[2], raw_line))

    ranks = [rank for rank, _, _ in rows]
    docids = [docid for _, docid, _ in rows]
    if len(rows) != expected_depth:
        raise ValueError(f"expected {expected_depth} rows for topic {topic_id}")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate ranks for topic {topic_id}")
    if sorted(ranks) != list(range(1, expected_depth + 1)):
        raise ValueError(f"ranks must be contiguous for topic {topic_id}")
    if len(set(docids)) != len(docids):
        raise ValueError(f"duplicate document IDs for topic {topic_id}")

    return _atomic_write(Path(path), b"".join(row for _, _, row in sorted(rows)))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
