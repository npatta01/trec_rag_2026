"""Adapters between this repository's RAG artifacts and RAGDoll's evaluation inputs.

RAGDoll resolves a topic id with ``row.get("qid") or row.get("topic_id") or row.get("query_id")``
(``ragdoll/src/ragdoll/trec_io.py``) and has no ``metadata.narrative_id`` fallback. Our
submission rows carry the id only under ``metadata``, so feeding them to RAGDoll unchanged makes
``iter_assign_payloads_from_files`` skip every answer and report empty metrics instead of
failing. The organizer contract forbids fixing that in place: ``ragnarok_style_ag.py`` rejects
any row whose root keys are not exactly ``metadata``/``references``/``answer``, and
``competition_rag._validate_generated_submission_record`` mirrors it.

So the submission file is left untouched and a separate RAGDoll-facing answers file is derived
from it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

GOLD_NUGGET_IMPORTANCE = frozenset({"vital", "okay"})


@dataclass(frozen=True)
class AnswerRow:
    """A RAGDoll answers row derived from one submission record."""

    run_id: str
    qid: str
    query: str
    answer_text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "qid": self.qid,
            "topic_id": self.qid,
            "query": self.query,
            "answer_text": self.answer_text,
            "response_length": len(self.answer_text.split()),
        }


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: row is not a JSON object")
            yield row


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def answer_rows(submission_path: Path) -> list[AnswerRow]:
    """Derive RAGDoll answers rows from an organizer submission JSONL."""
    rows: list[AnswerRow] = []
    seen: set[str] = set()
    for record in _read_jsonl(submission_path):
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{submission_path}: record is missing a metadata object")
        qid = metadata.get("narrative_id")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"{submission_path}: record is missing metadata.narrative_id")
        if qid in seen:
            raise ValueError(f"{submission_path}: duplicate narrative_id {qid}")
        seen.add(qid)

        answer = record.get("answer")
        if not isinstance(answer, list) or not answer:
            raise ValueError(f"{qid}: answer must be a nonempty list")
        texts: list[str] = []
        for index, item in enumerate(answer):
            if not isinstance(item, dict):
                raise ValueError(f"{qid}: answer[{index}] is not an object")
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{qid}: answer[{index}] has empty text")
            texts.append(text.strip())

        narrative = metadata.get("narrative")
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError(f"{qid}: record is missing metadata.narrative")
        run_id = metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"{qid}: record is missing metadata.run_id")

        rows.append(
            AnswerRow(
                run_id=run_id,
                qid=qid,
                query=narrative,
                answer_text=" ".join(texts),
            )
        )
    if not rows:
        raise ValueError(f"{submission_path}: no submission records")
    return rows


def gold_nugget_rows(
    nuggets_path: Path,
    *,
    narratives: dict[str, str],
    topic_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Reshape released gold nuggets into RAGDoll's nuggets shape.

    Drops ``mapped_sub_narrative`` (its released values contain literal embedded double quotes)
    and ``source``, keeping only the ``text``/``importance`` pair RAGDoll scores on.
    """
    wanted = set(topic_ids) if topic_ids is not None else None
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in _read_jsonl(nuggets_path):
        qid = record.get("qid")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"{nuggets_path}: record is missing qid")
        if wanted is not None and qid not in wanted:
            continue
        if qid in seen:
            raise ValueError(f"{nuggets_path}: duplicate qid {qid}")
        seen.add(qid)

        nuggets = record.get("nuggets")
        if not isinstance(nuggets, list) or not nuggets:
            raise ValueError(f"{qid}: nuggets must be a nonempty list")
        reshaped: list[dict[str, str]] = []
        for index, nugget in enumerate(nuggets):
            if not isinstance(nugget, dict):
                raise ValueError(f"{qid}: nuggets[{index}] is not an object")
            text = nugget.get("text")
            importance = nugget.get("importance")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{qid}: nuggets[{index}] has empty text")
            if importance not in GOLD_NUGGET_IMPORTANCE:
                raise ValueError(f"{qid}: nuggets[{index}] has importance {importance!r}")
            reshaped.append({"text": text.strip(), "importance": importance})

        narrative = narratives.get(qid)
        if narrative is None:
            raise ValueError(f"{qid}: no narrative available for gold nuggets")
        rows.append({"qid": qid, "query": narrative, "nuggets": reshaped})

    if wanted is not None:
        missing = sorted(wanted - seen)
        if missing:
            raise ValueError(f"{nuggets_path}: no gold nuggets for {', '.join(missing)}")
    if not rows:
        raise ValueError(f"{nuggets_path}: no gold nugget records selected")
    return rows


def assert_join(answers: Sequence[AnswerRow], nuggets: Sequence[dict[str, object]]) -> set[str]:
    """Return the shared topic ids, raising when the join would silently be empty."""
    answer_ids = {row.qid for row in answers}
    nugget_ids = {str(row["qid"]) for row in nuggets}
    shared = answer_ids & nugget_ids
    if not shared:
        raise ValueError(
            "answers and gold nuggets share no topic ids; "
            f"answers={sorted(answer_ids)} nuggets={sorted(nugget_ids)}"
        )
    return shared


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive RAGDoll inputs from RAG artifacts.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--gold-nuggets", type=Path, required=True)
    parser.add_argument("--answers-out", type=Path, required=True)
    parser.add_argument("--nuggets-out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        answers = answer_rows(args.submission)
        narratives = {row.qid: row.query for row in answers}
        nuggets = gold_nugget_rows(
            args.gold_nuggets,
            narratives=narratives,
            topic_ids=[row.qid for row in answers],
        )
        shared = assert_join(answers, nuggets)
        _write_jsonl(args.answers_out, [row.as_dict() for row in answers])
        _write_jsonl(args.nuggets_out, nuggets)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {type(error).__name__}: {error}") from error

    for row in nuggets:
        print(f"{row['qid']}: {len(row['nuggets'])} gold nuggets")  # type: ignore[arg-type]
    print(f"joined topics={len(shared)} answers={args.answers_out} nuggets={args.nuggets_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
