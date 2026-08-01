"""Build fixed-retrieval RAG inputs for development topics from archived Pyserini responses.

The development-topic Pyserini archives under ``cache/retrieval/pyserini_remote`` predate the
retriever's provenance sidecar requirement (``retrievers.py`` raises
``unverified cache missing provenance sidecar``), so they cannot be replayed through
``trec_rag.pipeline``. This module reads them directly as an archive rather than as a verified
cache, and never writes a sidecar: synthesizing provenance would defeat that check.

The emitted ordering is therefore raw BM25 as archived, not reranked. That is adequate for
wiring and integration work; a measured run should come from a verified retrieval export.

Outputs match what ``trec_rag.competition_rag`` consumes:

* a six-column TREC run, ``topic_id Q0 docid rank score run_id``
* a document JSONL, one row per topic, ``{"query": {"qid": ...}, "candidates": [...]}``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

ARCHIVE_VARIANT = "original"
ARCHIVE_RETRIEVER = "climbmix_bm25"
DEFAULT_DEPTH = 100


@dataclass(frozen=True)
class ArchivedCandidate:
    """One archived retrieval candidate with its document text."""

    docid: str
    score: float
    text: str


@dataclass(frozen=True)
class ArchivedTopic:
    """A topic's archived query text and its deduplicated candidates."""

    topic_id: str
    query_text: str
    candidates: tuple[ArchivedCandidate, ...]


def archive_path(cache_dir: Path, topic_id: str) -> Path:
    """Return the single archive for ``topic_id``, or raise when it is not unique."""
    pattern = f"{topic_id}__{ARCHIVE_VARIANT}__{ARCHIVE_RETRIEVER}__*.json"
    matches = sorted(
        path for path in cache_dir.glob(pattern) if not path.name.endswith(".meta.json")
    )
    if not matches:
        raise ValueError(f"{topic_id}: no archived retrieval response in {cache_dir}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"{topic_id}: ambiguous archived responses: {names}")
    return matches[0]


def load_archive(path: Path, *, topic_id: str, depth: int) -> ArchivedTopic:
    """Read one archive, keeping the best-ranked entry per docid up to ``depth``."""
    if depth <= 0:
        raise ValueError("depth must be a positive integer")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: archive is not a JSON object")
    if str(payload.get("topic_id")) != topic_id:
        raise ValueError(f"{path}: archive topic_id does not match {topic_id}")
    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"{path}: archive has no response object")
    query = response.get("query")
    query_text = query.get("text") if isinstance(query, dict) else payload.get("query")
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError(f"{path}: archive has no query text")

    rows = response.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: archive has no candidates")

    ordered = sorted(rows, key=_rank_of)
    seen: set[str] = set()
    kept: list[ArchivedCandidate] = []
    for row in ordered:
        if len(kept) == depth:
            break
        if not isinstance(row, dict):
            raise ValueError(f"{path}: candidate is not an object")
        docid = row.get("docid")
        text = row.get("doc")
        if not isinstance(docid, str) or not docid.strip():
            raise ValueError(f"{path}: candidate is missing a docid")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: {docid} is missing document text")
        if docid in seen:
            continue
        seen.add(docid)
        kept.append(ArchivedCandidate(docid=docid, score=float(row["score"]), text=text))

    if not kept:
        raise ValueError(f"{path}: no usable candidates")
    return ArchivedTopic(topic_id=topic_id, query_text=query_text, candidates=tuple(kept))


def _rank_of(row: object) -> int:
    if not isinstance(row, dict):
        raise ValueError("candidate is not an object")
    rank = row.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError("candidate has a non-integer rank")
    return rank


def trec_run_lines(topics: Sequence[ArchivedTopic], *, run_id: str) -> Iterator[str]:
    """Yield six-column TREC rows with dense ranks and non-increasing scores.

    ``competition_rag.load_trec_run`` rejects duplicate ranks, rank gaps, and scores that rise
    with rank, so ranks are renumbered after deduplication and scores are clamped to stay
    monotonically non-increasing.
    """
    if not run_id.strip() or any(character.isspace() for character in run_id):
        raise ValueError("run_id must be non-empty and whitespace-free")
    for topic in topics:
        previous: float | None = None
        for rank, candidate in enumerate(topic.candidates, start=1):
            score = candidate.score
            if previous is not None and score > previous:
                score = previous
            previous = score
            yield f"{topic.topic_id} Q0 {candidate.docid} {rank} {score:.6f} {run_id}"


def document_rows(topics: Sequence[ArchivedTopic]) -> Iterator[dict[str, object]]:
    """Yield ``competition_rag.load_documents`` rows, one per topic."""
    for topic in topics:
        yield {
            "query": {"qid": topic.topic_id, "text": topic.query_text},
            "candidates": [
                {"docid": candidate.docid, "doc": candidate.text}
                for candidate in topic.candidates
            ],
        }


def build(
    *,
    topic_ids: Sequence[str],
    cache_dir: Path,
    run_path: Path,
    documents_path: Path,
    run_id: str,
    depth: int = DEFAULT_DEPTH,
) -> list[ArchivedTopic]:
    """Write the TREC run and document JSONL for ``topic_ids`` and return what was read."""
    if not topic_ids:
        raise ValueError("at least one topic id is required")
    if len(set(topic_ids)) != len(topic_ids):
        raise ValueError("topic ids must be unique")

    topics = [
        load_archive(archive_path(cache_dir, topic_id), topic_id=topic_id, depth=depth)
        for topic_id in topic_ids
    ]

    run_path.parent.mkdir(parents=True, exist_ok=True)
    documents_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(
        "".join(f"{line}\n" for line in trec_run_lines(topics, run_id=run_id)),
        encoding="utf-8",
    )
    documents_path.write_text(
        "".join(
            f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n"
            for row in document_rows(topics)
        ),
        encoding="utf-8",
    )
    return topics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", action="append", dest="topics", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--documents-path", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    args = parser.parse_args(argv)

    try:
        topics = build(
            topic_ids=args.topics,
            cache_dir=args.cache_dir,
            run_path=args.run_path,
            documents_path=args.documents_path,
            run_id=args.run_id,
            depth=args.depth,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {type(error).__name__}: {error}") from error

    for topic in topics:
        print(f"{topic.topic_id}: {len(topic.candidates)} candidates")
    print(f"run={args.run_path} documents={args.documents_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
