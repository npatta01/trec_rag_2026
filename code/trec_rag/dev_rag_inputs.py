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
import re
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


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for from by with
    about into over after before between out against during without within along across is are
    was were be been being do does did have has had can could should would may might will just
    i me my we our you your it its as so such not no nor only own same too very s t don now""".split()
)


def _content_terms(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]


def select_passages(
    text: str,
    query: str,
    *,
    budget_words: int,
    window_words: int = 120,
    stride_words: int = 60,
) -> str:
    """Return the most query-relevant windows of ``text``, in document order.

    The generator prompt previously took each document's first ``max_document_words``. With a
    median document of about 3,400 words that shows roughly a sixth of the pool, always the
    opening, which on scraped web pages is often navigation furniture rather than evidence.

    Windows are scored by how many distinct query terms they contain, which rewards a passage
    covering several facets of the narrative over one repeating a single term. Selection is
    greedy and non-overlapping, and the kept windows are re-emitted in their original order so
    the text still reads forwards.
    """
    if budget_words <= 0:
        raise ValueError("budget_words must be a positive integer")
    words = text.split()
    if len(words) <= budget_words:
        return text

    wanted = set(_content_terms(query))
    if not wanted:
        return " ".join(words[:budget_words])

    scored: list[tuple[float, int]] = []
    for start in range(0, len(words), stride_words):
        window = words[start : start + window_words]
        if len(window) < min(window_words, 20):
            break
        terms = _content_terms(" ".join(window))
        present = wanted.intersection(terms)
        if not present:
            continue
        matches = sum(1 for term in terms if term in wanted)
        scored.append((len(present) + 0.05 * matches, start))

    if not scored:
        return " ".join(words[:budget_words])

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[int] = []
    used = 0
    for _, start in scored:
        if any(abs(start - other) < window_words for other in chosen):
            continue
        take = min(window_words, len(words) - start)
        if used + take > budget_words:
            continue
        chosen.append(start)
        used += take
        if used >= budget_words:
            break

    if not chosen:
        return " ".join(words[:budget_words])

    chosen.sort()
    parts = [" ".join(words[start : start + window_words]) for start in chosen]
    return " ... ".join(parts)


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


def document_rows(
    topics: Sequence[ArchivedTopic], *, passage_words: int | None = None
) -> Iterator[dict[str, object]]:
    """Yield ``competition_rag.load_documents`` rows, one per topic.

    With ``passage_words`` set, each document is reduced to its most query-relevant windows
    instead of being left for the generator to head-truncate.
    """
    for topic in topics:
        yield {
            "query": {"qid": topic.topic_id, "text": topic.query_text},
            "candidates": [
                {
                    "docid": candidate.docid,
                    "doc": (
                        candidate.text
                        if passage_words is None
                        else select_passages(
                            candidate.text, topic.query_text, budget_words=passage_words
                        )
                    ),
                }
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
    passage_words: int | None = None,
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
            for row in document_rows(topics, passage_words=passage_words)
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
    parser.add_argument(
        "--passage-words",
        type=int,
        help="Reduce each document to its most query-relevant windows of this many words.",
    )
    args = parser.parse_args(argv)

    try:
        topics = build(
            topic_ids=args.topics,
            cache_dir=args.cache_dir,
            run_path=args.run_path,
            documents_path=args.documents_path,
            run_id=args.run_id,
            depth=args.depth,
            passage_words=args.passage_words,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {type(error).__name__}: {error}") from error

    for topic in topics:
        print(f"{topic.topic_id}: {len(topic.candidates)} candidates")
    print(f"run={args.run_path} documents={args.documents_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
