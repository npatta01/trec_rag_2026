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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trec_rag.topics import load_narrative_topics

GOLD_NUGGET_IMPORTANCE = frozenset({"vital", "okay"})
SUPPORT_LABELS = frozenset({"FS", "PS", "NS"})


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


def answer_rows(
    submission_path: Path,
    *,
    narratives: Mapping[str, str],
) -> list[AnswerRow]:
    """Derive answers while binding every question to an authoritative topic."""
    if not isinstance(narratives, Mapping):
        raise TypeError("narratives must be a topic-to-narrative mapping")
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

        metadata_narrative = metadata.get("narrative")
        if not isinstance(metadata_narrative, str) or not metadata_narrative.strip():
            raise ValueError(f"{qid}: record is missing metadata.narrative")
        narrative = narratives.get(qid)
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError(f"{qid}: no authoritative topic narrative is available")
        if metadata_narrative != narrative:
            raise ValueError(f"{qid}: metadata narrative differs from authoritative topic narrative")
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


def selected_evidence_support_rows(
    submission_path: Path,
    handoff_manifest_path: Path,
    generation_identity_path: Path,
) -> list[dict[str, object]]:
    """Attach exactly the topic-owned selected passages shown to this generation run."""
    from trec_rag.generation_handoff import (
        HANDOFF_SCHEMA_VERSION,
        PROMPT_CONTRACT_VERSION,
        load_generation_handoff,
    )

    records, topic_docids = _support_inputs(submission_path)
    handoff = load_generation_handoff(handoff_manifest_path)
    identity = _load_generation_identity(generation_identity_path)
    if identity["handoff_schema_version"] != HANDOFF_SCHEMA_VERSION:
        raise ValueError(
            f"{generation_identity_path}: handoff_schema_version is not "
            f"{HANDOFF_SCHEMA_VERSION}"
        )
    if identity["prompt_contract_version"] != PROMPT_CONTRACT_VERSION:
        raise ValueError(
            f"{generation_identity_path}: prompt_contract_version is not "
            f"{PROMPT_CONTRACT_VERSION}"
        )
    if identity["handoff_manifest_sha256"] != handoff.manifest_sha256:
        raise ValueError(
            f"{generation_identity_path}: handoff_manifest_sha256 does not match "
            f"{handoff_manifest_path}"
        )

    run_ids = {
        str(record["metadata"]["run_id"])
        for record in records
    }
    if run_ids != {identity["run_id"]}:
        raise ValueError(
            f"{generation_identity_path}: run_id does not match submission run IDs "
            f"{sorted(run_ids)}"
        )

    identity_topics = identity["selected_topics"]
    topics = {topic.topic_id: topic for topic in handoff.topics}
    records_by_topic = {
        str(record["metadata"]["narrative_id"]).strip(): record
        for record in records
    }
    for topic_id, record in records_by_topic.items():
        topic = topics.get(topic_id)
        if topic is None:
            raise ValueError(
                f"{handoff_manifest_path}: no selected evidence for topic {topic_id}"
            )
        context_sha256 = identity_topics.get(topic_id)
        if context_sha256 != topic.context_sha256:
            raise ValueError(
                f"{generation_identity_path}: selected topic {topic_id} is absent or its "
                "context_sha256 does not match the handoff"
            )
        metadata = record["metadata"]
        if metadata.get("narrative") != topic.narrative:
            raise ValueError(
                f"{topic_id}: submission narrative does not match selected-evidence handoff"
            )
        evidence_docids = {evidence.docid for evidence in topic.evidence}
        foreign_references = sorted(
            str(docid)
            for docid in record["references"]
            if str(docid) not in evidence_docids
        )
        if foreign_references:
            raise ValueError(
                f"{topic_id}: references outside selected-evidence handoff: "
                f"{', '.join(foreign_references)}"
            )

    documents_by_topic: dict[str, dict[str, str]] = {}
    for topic_id, cited_docids in topic_docids.items():
        topic = topics[topic_id]
        passages: dict[str, list[str]] = {docid: [] for docid in cited_docids}
        seen: dict[str, set[str]] = {docid: set() for docid in cited_docids}
        for evidence in topic.evidence:
            if evidence.docid not in cited_docids or evidence.text in seen[evidence.docid]:
                continue
            seen[evidence.docid].add(evidence.text)
            passages[evidence.docid].append(evidence.text)
        missing = sorted(docid for docid, rows in passages.items() if not rows)
        if missing:
            raise ValueError(
                f"{topic_id}: no selected evidence for cited docids {', '.join(missing)}"
            )
        documents_by_topic[topic_id] = {
            docid: "\n\n".join(rows) for docid, rows in passages.items()
        }
    return _assemble_support_rows(
        submission_path,
        records=records,
        topic_docids=topic_docids,
        documents_by_topic=documents_by_topic,
    )


def _load_generation_identity(path: Path) -> dict[str, Any]:
    """Load the immutable generation receipt fields needed to bind an evidence view."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid generation identity JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: generation identity is not an object")

    identity_version = value.get("identity_version")
    schema_version = value.get("handoff_schema_version")
    prompt_contract_version = value.get("prompt_contract_version")
    digest = value.get("handoff_manifest_sha256")
    run_id = value.get("run_id")
    selected = value.get("selected_topics")
    if (
        isinstance(identity_version, bool)
        or not isinstance(identity_version, int)
        or identity_version <= 0
    ):
        raise ValueError(f"{path}: invalid identity_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError(f"{path}: invalid handoff_schema_version")
    if not isinstance(prompt_contract_version, str) or not prompt_contract_version:
        raise ValueError(f"{path}: invalid prompt_contract_version")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{path}: invalid handoff_manifest_sha256")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"{path}: invalid run_id")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"{path}: selected_topics must be a nonempty list")

    topics: dict[str, str] = {}
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: selected_topics[{index}] is not an object")
        topic_id = item.get("topic_id")
        context_sha256 = item.get("context_sha256")
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise ValueError(f"{path}: selected_topics[{index}] has invalid topic_id")
        if topic_id in topics:
            raise ValueError(f"{path}: duplicate selected topic {topic_id}")
        if not isinstance(context_sha256, str) or len(context_sha256) != 64:
            raise ValueError(
                f"{path}: selected_topics[{index}] has invalid context_sha256"
            )
        topics[topic_id] = context_sha256
    return {
        "identity_version": identity_version,
        "handoff_schema_version": schema_version,
        "prompt_contract_version": prompt_contract_version,
        "handoff_manifest_sha256": digest,
        "run_id": run_id,
        "selected_topics": topics,
    }


def _support_inputs(
    submission_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    records = list(_read_jsonl(submission_path))
    if not records:
        raise ValueError(f"{submission_path}: no submission records")

    topic_docids: dict[str, set[str]] = {}
    seen_topics: set[str] = set()
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{submission_path}: record is missing a metadata object")
        topic_id = metadata.get("narrative_id")
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise ValueError(f"{submission_path}: record is missing a narrative_id")
        topic_id = topic_id.strip()
        if topic_id in seen_topics:
            raise ValueError(f"{submission_path}: duplicate narrative_id {topic_id}")
        seen_topics.add(topic_id)
        run_id = metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"{submission_path}: record is missing metadata.run_id")
        references = record.get("references")
        if (
            not isinstance(references, list)
            or not references
            or not all(
                isinstance(docid, str) and docid.strip() and docid == docid.strip()
                for docid in references
            )
            or len(references) != len(set(references))
        ):
            raise ValueError(
                f"{submission_path}: references must be unique nonempty docid strings"
            )
        answer = record.get("answer")
        if not isinstance(answer, list) or not answer:
            raise ValueError(f"{submission_path}: record has no answer objects")
        record_cited: set[str] = set()
        for item in answer:
            if not isinstance(item, dict):
                raise ValueError(f"{submission_path}: answer object is not an object")
            for citation in item.get("citations") or []:
                if type(citation) is int:
                    if not 0 <= citation < len(references):
                        raise ValueError(f"{submission_path}: citation {citation} out of range")
                    docid = str(references[citation])
                    record_cited.add(docid)
                elif isinstance(citation, str):
                    # A docid citation naming something outside references would be dropped from
                    # segments, and ragdoll support silently skips a citation it cannot resolve.
                    # The skipped judgment leaves the metric denominator, understating failure.
                    if citation not in references:
                        raise ValueError(
                            f"{submission_path}: citation {citation!r} is not in references"
                        )
                    record_cited.add(citation)
                else:
                    raise ValueError(f"{submission_path}: unsupported citation {citation!r}")
        if record_cited:
            topic_docids.setdefault(topic_id, set()).update(record_cited)

    if not topic_docids:
        raise ValueError(f"{submission_path}: no citations to resolve")
    return records, topic_docids


def _assemble_support_rows(
    submission_path: Path,
    *,
    records: Sequence[dict[str, Any]],
    topic_docids: Mapping[str, set[str]],
    documents_by_topic: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{submission_path}: record is missing a metadata object")
        topic_id = metadata.get("narrative_id")
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise ValueError(f"{submission_path}: record is missing a narrative_id")
        topic_id = topic_id.strip()
        documents = documents_by_topic.get(topic_id, {})
        references = [str(docid) for docid in record["references"]]  # type: ignore[index]
        # Only cited documents need text. Uncited references are valid per the task reference
        # and simply carry no support judgment.
        topic_cited = topic_docids.get(topic_id, set())
        missing = [docid for docid in references if docid in topic_cited and docid not in documents]
        if missing:
            raise ValueError(
                f"{metadata.get('narrative_id')}: no document text for {', '.join(missing)}"
            )
        rows.append(
            {
                "topic_id": topic_id,
                "run_id": str(metadata.get("run_id")),
                "metadata": metadata,
                "references": references,
                "segments": {
                    docid: documents[docid] for docid in references if docid in documents
                },
                "answer": record["answer"],
            }
        )
    return rows


def validate_support_judgments(
    support_input_path: Path,
    judgments_path: Path,
) -> int:
    """Fail unless RAGDoll completed exactly every expected citation judgment."""
    expected: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(support_input_path):
        topic_id = row.get("topic_id")
        run_id = row.get("run_id")
        references = row.get("references")
        segments = row.get("segments")
        answer = row.get("answer")
        if not isinstance(topic_id, str) or not topic_id:
            raise ValueError(f"{support_input_path}: support row has no topic_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"{support_input_path}: support row has no run_id")
        if not isinstance(references, list) or not isinstance(segments, dict):
            raise ValueError(f"{support_input_path}: invalid references or segments")
        if not isinstance(answer, list):
            raise ValueError(f"{support_input_path}: invalid answer")
        for sentence_index, sentence in enumerate(answer):
            if not isinstance(sentence, dict) or not isinstance(sentence.get("text"), str):
                raise ValueError(
                    f"{support_input_path}: answer[{sentence_index}] has no text"
                )
            citations = sentence.get("citations")
            if not isinstance(citations, list):
                raise ValueError(
                    f"{support_input_path}: answer[{sentence_index}] has invalid citations"
                )
            for citation_index, citation in enumerate(citations):
                if type(citation) is int and 0 <= citation < len(references):
                    docid = str(references[citation])
                elif isinstance(citation, str):
                    docid = citation
                else:
                    raise ValueError(
                        f"{support_input_path}: unsupported citation {citation!r}"
                    )
                citation_text = segments.get(docid)
                if not isinstance(citation_text, str):
                    raise ValueError(
                        f"{support_input_path}: citation {docid!r} has no segment"
                    )
                task_id = f"{run_id}:{topic_id}:s{sentence_index}:c{citation_index}"
                if task_id in expected:
                    raise ValueError(f"{support_input_path}: duplicate task {task_id}")
                expected[task_id] = {
                    "statement": sentence["text"],
                    "citation": citation_text,
                    "topic_id": topic_id,
                    "run_id": run_id,
                    "sentence_index": sentence_index,
                    "citation_index": citation_index,
                    "docid": docid,
                }
    if not expected:
        raise ValueError(f"{support_input_path}: no support tasks")

    completed: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(judgments_path):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in expected:
            raise ValueError(f"{judgments_path}: unexpected task {task_id}")

        expected_row = expected[task_id]
        if row.get("statement") != expected_row["statement"]:
            raise ValueError(f"{task_id}: statement differs from support input")
        if row.get("citation") != expected_row["citation"]:
            raise ValueError(f"{task_id}: citation differs from support input")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{task_id}: judgment has no metadata")
        for field in (
            "topic_id",
            "run_id",
            "sentence_index",
            "citation_index",
            "docid",
        ):
            if metadata.get(field) != expected_row[field]:
                raise ValueError(f"{task_id}: metadata.{field} differs from support input")

        status = row.get("status")
        if status == "failed":
            if row.get("support_label") is not None:
                raise ValueError(f"{task_id}: failed judgment carries a support_label")
            continue
        if status != "completed":
            raise ValueError(f"{task_id}: invalid judgment status {status!r}")
        if row.get("support_label") not in SUPPORT_LABELS:
            raise ValueError(f"{task_id}: invalid support_label {row.get('support_label')!r}")
        if task_id in completed:
            raise ValueError(f"{judgments_path}: duplicate completed task {task_id}")
        completed[task_id] = row

    missing = sorted(set(expected) - set(completed))
    if missing:
        raise ValueError(
            f"{judgments_path}: missing {len(missing)} expected task(s): "
            f"{', '.join(missing[:3])}"
        )
    return len(completed)


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
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--gold-nuggets", type=Path, required=True)
    parser.add_argument("--answers-out", type=Path, required=True)
    parser.add_argument("--nuggets-out", type=Path, required=True)
    parser.add_argument(
        "--handoff-manifest",
        type=Path,
        help="Sealed selected-evidence handoff read by the v2 generator; enables --support-out.",
    )
    parser.add_argument(
        "--generation-identity",
        type=Path,
        help="Generation receipt that binds the answer run to the exact handoff digest.",
    )
    parser.add_argument(
        "--support-out",
        type=Path,
        help=(
            "Write rows for `ragdoll support judge`; requires --handoff-manifest "
            "and --generation-identity."
        ),
    )
    parser.add_argument(
        "--support-judgments",
        type=Path,
        help="Fail unless this RAGDoll output completed every task in --support-out.",
    )
    args = parser.parse_args(argv)

    has_handoff = args.handoff_manifest is not None
    has_identity = args.generation_identity is not None
    if bool(args.support_out) != has_handoff or bool(args.support_out) != has_identity:
        parser.error(
            "--support-out, --handoff-manifest, and --generation-identity "
            "must be provided together"
        )
    if args.support_judgments is not None and args.support_out is None:
        parser.error("--support-judgments requires --support-out")

    try:
        narratives = {
            topic.id: topic.narrative for topic in load_narrative_topics(args.topics)
        }
        answers = answer_rows(args.submission, narratives=narratives)
        nuggets = gold_nugget_rows(
            args.gold_nuggets,
            narratives=narratives,
            topic_ids=[row.qid for row in answers],
        )
        shared = assert_join(answers, nuggets)
        _write_jsonl(args.answers_out, [row.as_dict() for row in answers])
        _write_jsonl(args.nuggets_out, nuggets)
        support_count = 0
        judgment_count = 0
        if args.support_out is not None:
            support_rows = selected_evidence_support_rows(
                args.submission,
                args.handoff_manifest,
                args.generation_identity,
            )
            support_count = _write_jsonl(args.support_out, support_rows)
            if args.support_judgments is not None:
                judgment_count = validate_support_judgments(
                    args.support_out,
                    args.support_judgments,
                )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {type(error).__name__}: {error}") from error

    for row in nuggets:
        print(f"{row['qid']}: {len(row['nuggets'])} gold nuggets")  # type: ignore[arg-type]
    print(f"joined topics={len(shared)} answers={args.answers_out} nuggets={args.nuggets_out}")
    if args.support_out is not None:
        print(f"support rows={support_count} -> {args.support_out}")
    if args.support_judgments is not None:
        print(f"validated support judgments={judgment_count} -> {args.support_judgments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
