"""Deterministic input and artifact contracts for the Topic 213 handover.

The loader deliberately keeps organizer qrel grades, reviewer support scores,
and raw source text separate.  Raw text is available only to the shortlisting
and review stages; the durable handover renderer never emits it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


TOPIC_ID = "213"
CANONICAL_ELIGIBLE_DOCUMENT_COUNT = 173
CANONICAL_SUB_NARRATIVE_COUNT = 10
_ELIGIBLE_GRADES = frozenset({2, 3, 4})
_FORBIDDEN_TEXT = ("/home/", "PYSERINI_API_TOKEN")


@dataclass(frozen=True)
class EligibleDocument:
    """One organizer-eligible Topic 213 document and its source text."""

    document_id: str
    text: str
    topic_qrel_grade: int


@dataclass(frozen=True)
class TopicEvidenceInputs:
    """The typed, joined inputs used for Topic 213 evidence shortlisting."""

    topic_id: str
    narrative: str
    sub_narratives: tuple[str, ...]
    documents: tuple[EligibleDocument, ...]


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_number} is invalid JSON") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"{label}:{line_number} must be a JSON object")
        rows.append(row)
    return rows


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _load_topic_narrative(topic_tsv: Path) -> str:
    matches: list[str] = []
    try:
        lines = Path(topic_tsv).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"topic TSV is unreadable: {topic_tsv}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        if "\t" not in raw_line:
            raise ValueError(f"topic TSV:{line_number} must contain qid and narrative")
        topic_id, narrative = raw_line.split("\t", 1)
        if topic_id.strip() == TOPIC_ID:
            matches.append(_required_string(narrative, f"topic TSV:{line_number} narrative"))
    if len(matches) != 1:
        raise ValueError(f"topic TSV must contain exactly one Topic {TOPIC_ID} narrative")
    return matches[0]


def _load_sub_narratives(nuggets_jsonl: Path) -> tuple[str, ...]:
    matches = [row for row in _read_jsonl(nuggets_jsonl, "nuggets JSONL") if row.get("qid") == TOPIC_ID]
    if len(matches) != 1:
        raise ValueError(f"nuggets JSONL must contain exactly one Topic {TOPIC_ID} record")
    nuggets = matches[0].get("nuggets")
    if not isinstance(nuggets, list) or not nuggets:
        raise ValueError("Topic 213 nuggets must be a nonempty array")
    labels: list[str] = []
    seen: set[str] = set()
    for index, nugget in enumerate(nuggets, start=1):
        if not isinstance(nugget, Mapping):
            raise ValueError(f"Topic 213 nugget {index} must be an object")
        label = _required_string(
            nugget.get("mapped_sub_narrative"),
            f"Topic 213 nugget {index} mapped_sub_narrative",
        )
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return tuple(labels)


def _load_eligible_grades(qrels: Path) -> dict[str, int]:
    eligible: dict[str, int] = {}
    seen_topic_docids: set[str] = set()
    try:
        lines = Path(qrels).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"qrels is unreadable: {qrels}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        columns = raw_line.split()
        if len(columns) != 4:
            raise ValueError(f"qrels:{line_number} must have four whitespace-delimited columns")
        topic_id, _iteration, document_id, raw_grade = columns
        if topic_id != TOPIC_ID:
            continue
        if document_id in seen_topic_docids:
            raise ValueError(f"qrels contains a duplicate Topic {TOPIC_ID} document ID: {document_id}")
        seen_topic_docids.add(document_id)
        try:
            grade = int(raw_grade)
        except ValueError as exc:
            raise ValueError(f"qrels:{line_number} grade must be an integer") from exc
        if grade in _ELIGIBLE_GRADES:
            eligible[document_id] = grade
    if not eligible:
        raise ValueError("qrels contains no eligible Topic 213 documents")
    return eligible


def _load_document_texts(
    rows: Iterable[Mapping[str, object]],
    *,
    label: str,
    topic_scoped: bool,
    eligible_docids: set[str],
) -> dict[str, str]:
    documents: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        if topic_scoped and row.get("topic_id") != TOPIC_ID:
            continue
        document_id = row.get("document_id")
        if document_id not in eligible_docids:
            continue
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"{label}:{index} has an invalid document ID")
        if document_id in documents:
            raise ValueError(f"{label} contains a duplicate document ID: {document_id}")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label}:{index} has missing text for document ID {document_id}")
        documents[document_id] = text
    return documents


def load_topic213_inputs(
    *,
    topic_tsv: Path,
    nuggets_jsonl: Path,
    qrels: Path,
    accepted_union: Path,
    supplemental_documents: Path | None = None,
    canonical: bool = True,
) -> TopicEvidenceInputs:
    """Load and join the Topic 213 evidence population.

    ``supplemental_documents`` is a local JSONL source for organizer-eligible
    documents absent from the authenticated accepted union.  It contains
    ``document_id`` and ``text`` fields only; no network retrieval occurs.
    """

    narrative = _load_topic_narrative(topic_tsv)
    sub_narratives = _load_sub_narratives(nuggets_jsonl)
    eligible_grades = _load_eligible_grades(qrels)
    eligible_docids = set(eligible_grades)
    accepted_texts = _load_document_texts(
        _read_jsonl(accepted_union, "accepted union"),
        label="accepted union",
        topic_scoped=True,
        eligible_docids=eligible_docids,
    )
    supplemental_texts: dict[str, str] = {}
    if supplemental_documents is not None:
        supplemental_texts = _load_document_texts(
            _read_jsonl(supplemental_documents, "supplemental documents"),
            label="supplemental documents",
            topic_scoped=False,
            eligible_docids=eligible_docids,
        )
    overlap = set(accepted_texts).intersection(supplemental_texts)
    conflicting = sorted(
        document_id
        for document_id in overlap
        if accepted_texts[document_id] != supplemental_texts[document_id]
    )
    if conflicting:
        raise ValueError(
            "accepted union and supplemental documents contain conflicting text for document IDs: "
            + ", ".join(conflicting)
        )
    document_texts = {**supplemental_texts, **accepted_texts}
    missing = sorted(eligible_docids - set(document_texts))
    if missing:
        raise ValueError(
            "eligible Topic 213 documents are missing text: " + ", ".join(missing[:5])
        )
    if canonical and len(eligible_grades) != CANONICAL_ELIGIBLE_DOCUMENT_COUNT:
        raise ValueError(
            "canonical Topic 213 eligible population must contain "
            f"{CANONICAL_ELIGIBLE_DOCUMENT_COUNT} documents, found {len(eligible_grades)}"
        )
    if canonical and len(sub_narratives) != CANONICAL_SUB_NARRATIVE_COUNT:
        raise ValueError(
            "canonical Topic 213 must contain "
            f"{CANONICAL_SUB_NARRATIVE_COUNT} unique mapped_sub_narrative values"
        )
    return TopicEvidenceInputs(
        topic_id=TOPIC_ID,
        narrative=narrative,
        sub_narratives=sub_narratives,
        documents=tuple(
            EligibleDocument(
                document_id=document_id,
                text=document_texts[document_id],
                topic_qrel_grade=eligible_grades[document_id],
            )
            for document_id in sorted(eligible_grades)
        ),
    )


def _contains_forbidden_content(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            key == "text" or _contains_forbidden_content(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_content(child) for child in value)
    return isinstance(value, str) and any(forbidden in value for forbidden in _FORBIDDEN_TEXT)


def validate_reviewed_handover(
    handover: Mapping[str, object],
    *,
    eligible_docids: set[str],
) -> None:
    """Reject a final handover that violates its review and sanitization contract."""

    if not isinstance(handover, Mapping):
        raise ValueError("handover must be an object")
    if handover.get("topic_id") != TOPIC_ID:
        raise ValueError("handover topic_id must be '213'")
    if _contains_forbidden_content(handover):
        raise ValueError("handover contains raw document text, a secret, or an absolute machine path")
    rows = handover.get("sub_narratives")
    if not isinstance(rows, list) or len(rows) != CANONICAL_SUB_NARRATIVE_COUNT:
        raise ValueError("handover must contain exactly 10 sub_narratives")
    seen_sub_narratives: set[str] = set()
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"sub_narratives[{row_index}] must be an object")
        sub_narrative = _required_string(
            row.get("sub_narrative"), f"sub_narratives[{row_index}].sub_narrative"
        )
        if sub_narrative in seen_sub_narratives:
            raise ValueError("sub_narratives must be unique")
        seen_sub_narratives.add(sub_narrative)
        documents = row.get("documents")
        if not isinstance(documents, list) or len(documents) != 5:
            raise ValueError(f"{sub_narrative!r} must contain exactly five documents")
        seen_documents: set[str] = set()
        for document_index, document in enumerate(documents, start=1):
            if not isinstance(document, Mapping):
                raise ValueError(f"{sub_narrative!r} document {document_index} must be an object")
            document_id = _required_string(
                document.get("document_id"),
                f"{sub_narrative!r} document {document_index}.document_id",
            )
            if document_id not in eligible_docids:
                raise ValueError(f"{sub_narrative!r} includes an ineligible document ID")
            if document_id in seen_documents:
                raise ValueError(f"{sub_narrative!r} documents must be unique")
            seen_documents.add(document_id)
            grade = document.get("topic_qrel_grade")
            if isinstance(grade, bool) or not isinstance(grade, int) or grade not in _ELIGIBLE_GRADES:
                raise ValueError(f"{sub_narrative!r} document topic_qrel_grade must be 2, 3, or 4")
            support_score = document.get("support_score")
            if (
                isinstance(support_score, bool)
                or not isinstance(support_score, int)
                or support_score not in {2, 3}
            ):
                raise ValueError(f"{sub_narrative!r} document support_score must be 2 or 3")
            claims = document.get("claims")
            if (
                not isinstance(claims, list)
                or not claims
                or any(not isinstance(claim, str) or not claim.strip() for claim in claims)
            ):
                raise ValueError(f"{sub_narrative!r} document claims must be a nonempty string array")


def render_handover_markdown(handover: Mapping[str, object]) -> str:
    """Render reviewed claims and scores without emitting raw document text."""

    if not isinstance(handover, Mapping):
        raise ValueError("handover must be an object")
    topic_id = _required_string(handover.get("topic_id"), "handover topic_id")
    narrative = _required_string(handover.get("narrative"), "handover narrative")
    rows = handover.get("sub_narratives")
    if not isinstance(rows, list):
        raise ValueError("handover sub_narratives must be an array")
    lines = [f"# Topic {topic_id} evidence handover", "", narrative]
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("handover sub_narrative must be an object")
        sub_narrative = _required_string(row.get("sub_narrative"), "sub_narrative")
        documents = row.get("documents")
        if not isinstance(documents, list):
            raise ValueError(f"{sub_narrative!r} documents must be an array")
        lines.extend(("", f"## {sub_narrative}"))
        for document in documents:
            if not isinstance(document, Mapping):
                raise ValueError(f"{sub_narrative!r} document must be an object")
            document_id = _required_string(document.get("document_id"), "document_id")
            grade = document.get("topic_qrel_grade")
            support_score = document.get("support_score")
            claims = document.get("claims")
            if isinstance(grade, bool) or not isinstance(grade, int):
                raise ValueError("topic_qrel_grade must be an integer")
            if isinstance(support_score, bool) or not isinstance(support_score, int):
                raise ValueError("support_score must be an integer")
            if not isinstance(claims, list):
                raise ValueError("claims must be an array")
            lines.extend(
                (
                    "",
                    f"### Document `{document_id}`",
                    f"- topic qrel grade: {grade}",
                    f"- support score: {support_score}",
                    "- claims:",
                )
            )
            for claim in claims:
                lines.append(f"  - {_required_string(claim, 'claim')}")
    return "\n".join(lines) + "\n"
