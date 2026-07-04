"""BM25 retrieval baseline for TREC RAG 2026."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Protocol

from trec_rag.remote_pyserini import (
    RemotePyseriniClient,
    RemotePyseriniConfig,
    find_repo_root,
    load_repo_env,
    normalize_candidates,
)
from trec_rag.topics import Topic, load_topics


DEFAULT_OUTPUT = Path("outputs/baseline/r_output_trec_rag_2026.tsv")
DEFAULT_CACHE_DIR = Path("outputs/baseline/cache")
DEFAULT_RUN_ID = "pyserini_climbmix_bm25_top100"


class SearchClient(Protocol):
    def search(self, query: str) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class RetrievalRow:
    topic_id: str
    docid: str
    rank: int
    score: float
    run_id: str

    def to_trec(self) -> str:
        return f"{self.topic_id} Q0 {self.docid} {self.rank} {self.score} {self.run_id}"


def build_query(topic: Topic) -> str:
    return " ".join(topic.narrative.split())


def _candidate_rank(candidate: dict[str, object]) -> int:
    try:
        return int(candidate.get("rank") or 0)
    except (TypeError, ValueError):
        return 0


def _candidate_score(candidate: dict[str, object]) -> float:
    score = candidate.get("score")
    if score is None:
        return 0.0
    try:
        return float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidate {candidate.get('docid')!r} has non-numeric score") from exc


def candidates_to_run_rows(
    topic: Topic,
    candidates: Iterable[dict[str, object]],
    *,
    run_id: str,
    depth: int,
) -> list[RetrievalRow]:
    if depth < 1:
        raise ValueError("depth must be at least 1")

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (_candidate_rank(candidate), -_candidate_score(candidate)),
    )
    rows: list[RetrievalRow] = []
    for output_rank, candidate in enumerate(sorted_candidates[:depth], start=1):
        docid = candidate.get("docid")
        if not docid or not str(docid).strip():
            raise ValueError(f"topic {topic.id}: candidate rank {output_rank} is missing docid")
        rows.append(
            RetrievalRow(
                topic_id=topic.id,
                docid=str(docid),
                rank=output_rank,
                score=_candidate_score(candidate),
                run_id=run_id,
            )
        )
    return rows


def write_retrieval_run(rows: Iterable[RetrievalRow], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(row.to_trec() + "\n")
            count += 1
    return count


def _cache_path(cache_dir: Path, topic_id: str) -> Path:
    safe_topic_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", topic_id).strip("._") or "topic"
    return cache_dir / f"{safe_topic_id}.json"


def run_bm25_retrieval(
    topics: Iterable[Topic],
    *,
    client: SearchClient,
    output_path: Path,
    cache_dir: Path,
    run_id: str = DEFAULT_RUN_ID,
    depth: int = 100,
) -> list[RetrievalRow]:
    all_rows: list[RetrievalRow] = []
    cache_dir.mkdir(parents=True, exist_ok=True)

    for topic in topics:
        query = build_query(topic)
        response = client.search(query)
        _cache_path(cache_dir, topic.id).write_text(
            json.dumps({"query": query, "response": response}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        candidates = normalize_candidates(response)
        all_rows.extend(candidates_to_run_rows(topic, candidates, run_id=run_id, depth=depth))

    write_retrieval_run(all_rows, output_path)
    return all_rows


def validate_retrieval_run(runfile: Path, *, expected_topic_ids: Iterable[str]) -> None:
    expected = {str(topic_id) for topic_id in expected_topic_ids}
    seen_topics: set[str] = set()
    ranks_by_topic: dict[str, list[int]] = {}
    docids_by_topic: dict[str, set[str]] = {}
    errors: list[str] = []

    for line_number, raw_line in enumerate(runfile.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        columns = raw_line.split()
        if len(columns) != 6:
            errors.append(f"line {line_number}: expected six columns")
            continue

        topic_id, q0, docid, rank_text, score_text, _run_id = columns
        if q0 != "Q0":
            errors.append(f"line {line_number}: second column must be Q0")
        try:
            rank = int(rank_text)
        except ValueError:
            errors.append(f"line {line_number}: rank must be an integer")
            continue
        if rank < 1:
            errors.append(f"line {line_number}: rank must be positive")
        try:
            float(score_text)
        except ValueError:
            errors.append(f"line {line_number}: score must be numeric")

        seen_topics.add(topic_id)
        ranks_by_topic.setdefault(topic_id, []).append(rank)
        topic_docids = docids_by_topic.setdefault(topic_id, set())
        if docid in topic_docids:
            errors.append(f"duplicate docid {docid} for topic {topic_id}")
        topic_docids.add(docid)

    missing_topics = sorted(expected - seen_topics)
    if missing_topics:
        errors.append(f"missing topics: {', '.join(missing_topics)}")

    for topic_id, ranks in sorted(ranks_by_topic.items()):
        expected_ranks = list(range(1, len(ranks) + 1))
        if sorted(ranks) != expected_ranks:
            errors.append(f"topic {topic_id} ranks must be contiguous from 1")

    if errors:
        raise ValueError("; ".join(errors))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run topic-text BM25 retrieval against the hosted Pyserini ClimbMix index."
    )
    parser.add_argument("--topics", type=Path, required=True, help="Topic JSONL or dev TSV file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--hits", type=int, default=100)
    parser.add_argument("--title-words", type=int, default=12)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = find_repo_root()
    load_repo_env(repo_root)
    topics = load_topics(args.topics, title_words=args.title_words)
    config = replace(RemotePyseriniConfig.from_env(), hits=args.hits)
    client = RemotePyseriniClient(config)

    rows = run_bm25_retrieval(
        topics,
        client=client,
        output_path=args.output,
        cache_dir=args.cache_dir,
        run_id=args.run_id,
        depth=args.hits,
    )
    validate_retrieval_run(args.output, expected_topic_ids=[topic.id for topic in topics])
    print(f"Wrote {len(rows)} retrieval rows for {len(topics)} topics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
