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


def resolved_support_rows(
    submission_path: Path,
    documents_path: Path,
    *,
    archive_member: str | None = None,
    max_words: int = 1000,
) -> list[dict[str, object]]:
    """Attach cited document text to submission records for ``ragdoll support judge``.

    ``support/stages.py`` already falls back to ``metadata.narrative_id``, so the submission
    shape needs no id adapter here — only the ``segments`` map it cannot resolve offline. A
    citation whose docid is absent from ``segments`` is silently skipped by the judge, so every
    cited docid is required up front instead.

    ``documents_path`` must be the same document file the generator read. Judging citations
    against text the model never saw produces spurious No Support: an earlier run of this
    comparison passed head-truncated text against an extractive generation and inflated the No
    Support rate from 4.8% to 32.7%, inverting the conclusion.

    Note also that organizers resolve references from the index, that is full documents, so
    scoring against any truncated view understates support for every arm.
    """
    from trec_rag.competition_rag import load_documents

    records = list(_read_jsonl(submission_path))
    if not records:
        raise ValueError(f"{submission_path}: no submission records")

    cited: set[str] = set()
    for record in records:
        references = record.get("references")
        if not isinstance(references, list) or not references:
            raise ValueError(f"{submission_path}: record has no references")
        answer = record.get("answer")
        if not isinstance(answer, list) or not answer:
            raise ValueError(f"{submission_path}: record has no answer objects")
        for item in answer:
            if not isinstance(item, dict):
                raise ValueError(f"{submission_path}: answer object is not an object")
            for citation in item.get("citations") or []:
                if type(citation) is int:
                    if not 0 <= citation < len(references):
                        raise ValueError(f"{submission_path}: citation {citation} out of range")
                    cited.add(str(references[citation]))
                elif isinstance(citation, str):
                    cited.add(citation)
                else:
                    raise ValueError(f"{submission_path}: unsupported citation {citation!r}")

    if not cited:
        raise ValueError(f"{submission_path}: no citations to resolve")
    documents = load_documents(documents_path, archive_member, cited, max_words)

    rows: list[dict[str, object]] = []
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{submission_path}: record is missing a metadata object")
        references = [str(docid) for docid in record["references"]]  # type: ignore[index]
        missing = [docid for docid in references if docid not in documents]
        if missing:
            raise ValueError(
                f"{metadata.get('narrative_id')}: no document text for {', '.join(missing)}"
            )
        rows.append(
            {
                "topic_id": str(metadata.get("narrative_id")),
                "run_id": str(metadata.get("run_id")),
                "metadata": metadata,
                "references": references,
                "segments": {docid: documents[docid] for docid in references},
                "answer": record["answer"],
            }
        )
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
    parser.add_argument(
        "--documents",
        type=Path,
        help="Document JSONL/ZIP; enables --support-out reference resolution.",
    )
    parser.add_argument("--archive-member")
    parser.add_argument("--max-document-words", type=int, default=1000)
    parser.add_argument(
        "--support-out",
        type=Path,
        help="Write resolved rows for `ragdoll support judge` (requires --documents).",
    )
    args = parser.parse_args(argv)

    if bool(args.support_out) != bool(args.documents):
        parser.error("--support-out and --documents must be given together")

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
        support_count = 0
        if args.support_out is not None:
            support_rows = resolved_support_rows(
                args.submission,
                args.documents,
                archive_member=args.archive_member,
                max_words=args.max_document_words,
            )
            support_count = _write_jsonl(args.support_out, support_rows)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {type(error).__name__}: {error}") from error

    for row in nuggets:
        print(f"{row['qid']}: {len(row['nuggets'])} gold nuggets")  # type: ignore[arg-type]
    print(f"joined topics={len(shared)} answers={args.answers_out} nuggets={args.nuggets_out}")
    if args.support_out is not None:
        print(f"support rows={support_count} -> {args.support_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
