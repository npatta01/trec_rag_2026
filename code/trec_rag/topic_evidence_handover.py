"""Deterministic input and artifact contracts for the Topic 213 handover.

The loader deliberately keeps organizer qrel grades, reviewer support scores,
and raw source text separate.  Raw text is available only to the shortlisting
and review stages; the durable handover renderer never emits it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .chunking import ChunkingConfig, SemanticTextChunker
from .remote_client import RemotePyseriniThrottled, extract_text, rate_limited_session
from .remote_config import RemotePyseriniConfig
from .repo_env import find_repo_root, load_repo_env


TOPIC_ID = "213"
CANONICAL_ELIGIBLE_DOCUMENT_COUNT = 173
CANONICAL_SUB_NARRATIVES = (
    '"What triggered the Korean War?"',
    '"How did Cold War strategy influence US actions?"',
    '"What motivated US involvement in the Korean War?"',
    '"What major strategic or political mistakes were made during the war?"',
    '"What impact did the Korean War have on US politics?"',
    '"How did the Korean War conclude?"',
    '"How did US presidents differ in their views on the Korean War?"',
    "New: How does the Korean War affect Korea",
    "New: How does the Korean War affect UN",
    "New: What motivated China involvement in te Korean War?",
)
CANONICAL_SUB_NARRATIVE_COUNT = len(CANONICAL_SUB_NARRATIVES)
_ELIGIBLE_GRADES = frozenset({2, 3, 4})
_HANDOVER_KEYS = frozenset({"topic_id", "narrative", "sub_narratives"})
_SUB_NARRATIVE_KEYS = frozenset({"sub_narrative", "documents"})
_DOCUMENT_KEYS = frozenset(
    {"document_id", "topic_qrel_grade", "support_score", "claims", "review_rationale"}
)
_CREDENTIAL_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|credential|cookie|password|secret|token)", re.IGNORECASE
)
_SENSITIVE_CREDENTIAL_VALUE_RE = re.compile(
    r"\b(?:PYSERINI_API_TOKEN|api[_-]?key|authorization|credential|cookie|password|secret|token)"
    r"\s*(?:[:=]\s*|\s+Bearer\s+)\S+|\bBearer\s+\S+",
    re.IGNORECASE,
)
_POSIX_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9+.\-:/])/[^\s/]+(?:/[^\s/]+)*")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9+.\-:/\\])[A-Za-z]:[\\/]")
_HOME_PATH_RE = re.compile(r"(?<![A-Za-z0-9+.\-:/~])~[\\/]")

MODEL_ID = "mixedbread-ai/mxbai-rerank-base-v2"
MODEL_REVISION = "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
DEFAULT_OUTPUT_DIRECTORY = Path("outputs/rag25_topic213_evidence_handover_v1")
DEFAULT_ACCEPTED_UNION = Path(
    "outputs/all_topic_tethered_facet_validation_v1/retrieval/accepted_union.jsonl"
)
DEFAULT_TOPIC_TSV = Path(
    "trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"
)
DEFAULT_NUGGETS_JSONL = Path(
    "trec-rag-data/trec-rag-2026/development-data/rag25-dev-nuggets/"
    "rag25-dev-nuggets.jsonl"
)
DEFAULT_QRELS = Path(
    "trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/"
    "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
)

PairScorer = Callable[[str, list[str]], Sequence[float]]


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
    if canonical and sub_narratives != CANONICAL_SUB_NARRATIVES:
        raise ValueError(
            "canonical Topic 213 must preserve the exact released "
            "mapped_sub_narrative tuple"
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


def _require_allowlisted_keys(
    value: object,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    credential_fields = sorted(
        key for key in actual if _CREDENTIAL_KEY_RE.search(str(key))
    )
    if credential_fields:
        raise ValueError(f"{label} contains credential-like fields")
    unexpected = sorted(actual - allowed)
    if unexpected:
        raise ValueError(f"{label} contains fields outside the allowlisted schema")
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"{label} is missing required allowlisted fields")
    return value


def _reject_sensitive_values(value: object) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_sensitive_values(child)
        return
    if isinstance(value, list):
        for child in value:
            _reject_sensitive_values(child)
        return
    if isinstance(value, str):
        if _SENSITIVE_CREDENTIAL_VALUE_RE.search(value):
            raise ValueError("handover contains a sensitive credential value")
        if (
            _POSIX_ABSOLUTE_PATH_RE.search(value)
            or _WINDOWS_ABSOLUTE_PATH_RE.search(value)
            or _HOME_PATH_RE.search(value)
        ):
            raise ValueError("handover contains an absolute filesystem path")


def _provenance_grades(
    *,
    inputs: TopicEvidenceInputs | None,
    qrel_grades: Mapping[str, int] | None,
) -> Mapping[str, int]:
    if (inputs is None) == (qrel_grades is None):
        raise ValueError("provide exactly one of inputs or qrel_grades")
    if inputs is not None:
        if inputs.topic_id != TOPIC_ID:
            raise ValueError("TopicEvidenceInputs must be for Topic 213")
        grades = {document.document_id: document.topic_qrel_grade for document in inputs.documents}
        if len(grades) != len(inputs.documents):
            raise ValueError("TopicEvidenceInputs contains duplicate document IDs")
    else:
        assert qrel_grades is not None
        grades = dict(qrel_grades)
    if not grades:
        raise ValueError("qrel-grade provenance cannot be empty")
    for document_id, grade in grades.items():
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("qrel-grade provenance contains an invalid document ID")
        if isinstance(grade, bool) or not isinstance(grade, int) or grade not in _ELIGIBLE_GRADES:
            raise ValueError("qrel-grade provenance must contain grades 2, 3, or 4")
    return grades


def validate_reviewed_handover(
    handover: Mapping[str, object],
    *,
    inputs: TopicEvidenceInputs | None = None,
    qrel_grades: Mapping[str, int] | None = None,
) -> None:
    """Reject a final handover that violates its review and sanitization contract."""

    provenance = _provenance_grades(inputs=inputs, qrel_grades=qrel_grades)
    handover = _require_allowlisted_keys(
        handover,
        allowed=_HANDOVER_KEYS,
        required=_HANDOVER_KEYS,
        label="handover",
    )
    if handover.get("topic_id") != TOPIC_ID:
        raise ValueError("handover topic_id must be '213'")
    _required_string(handover.get("narrative"), "handover narrative")
    _reject_sensitive_values(handover)
    rows = handover.get("sub_narratives")
    if not isinstance(rows, list) or len(rows) != CANONICAL_SUB_NARRATIVE_COUNT:
        raise ValueError("handover must contain exactly 10 sub_narratives")
    sub_narratives: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        row = _require_allowlisted_keys(
            row,
            allowed=_SUB_NARRATIVE_KEYS,
            required=_SUB_NARRATIVE_KEYS,
            label=f"sub_narratives[{row_index}]",
        )
        sub_narrative = _required_string(
            row.get("sub_narrative"), f"sub_narratives[{row_index}].sub_narrative"
        )
        sub_narratives.append(sub_narrative)
        documents = row.get("documents")
        if not isinstance(documents, list) or len(documents) != 5:
            raise ValueError(f"{sub_narrative!r} must contain exactly five documents")
        seen_documents: set[str] = set()
        for document_index, document in enumerate(documents, start=1):
            document = _require_allowlisted_keys(
                document,
                allowed=_DOCUMENT_KEYS,
                required=_DOCUMENT_KEYS - {"review_rationale"},
                label=f"{sub_narrative!r} document {document_index}",
            )
            document_id = _required_string(
                document.get("document_id"),
                f"{sub_narrative!r} document {document_index}.document_id",
            )
            if document_id not in provenance:
                raise ValueError(f"{sub_narrative!r} includes an ineligible document ID")
            if document_id in seen_documents:
                raise ValueError(f"{sub_narrative!r} documents must be unique")
            seen_documents.add(document_id)
            grade = document.get("topic_qrel_grade")
            if isinstance(grade, bool) or not isinstance(grade, int) or grade not in _ELIGIBLE_GRADES:
                raise ValueError(f"{sub_narrative!r} document topic_qrel_grade must be 2, 3, or 4")
            if grade != provenance[document_id]:
                raise ValueError(
                    f"{sub_narrative!r} document topic_qrel_grade does not match input provenance"
                )
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
            if "review_rationale" in document:
                _required_string(
                    document["review_rationale"],
                    f"{sub_narrative!r} document review_rationale",
                )
    if tuple(sub_narratives) != CANONICAL_SUB_NARRATIVES:
        raise ValueError(
            "handover sub_narratives must equal the exact released "
            "mapped_sub_narrative tuple"
        )


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


def missing_document_ids(
    *,
    eligible_docids: set[str],
    available_docids: set[str],
) -> list[str]:
    """Return the deterministic fetch plan for organizer-eligible documents."""

    return sorted(eligible_docids - available_docids)


def score_sub_narrative_pairs(
    inputs: TopicEvidenceInputs,
    scorer: PairScorer,
    *,
    shortlist_depth: int = 12,
) -> dict[str, object]:
    """Score every sub-narrative/document pair and retain per-label leaders.

    ``scorer`` receives each sub-narrative and its flattened document chunks,
    then returns one raw model logit per chunk.  This keeps model execution
    injected while the whole-document max-chunk and shortlist contracts remain
    deterministic and independently testable.
    """

    if inputs.topic_id != TOPIC_ID:
        raise ValueError("TopicEvidenceInputs must be for Topic 213")
    if shortlist_depth <= 0:
        raise ValueError("shortlist_depth must be positive")
    if len({document.document_id for document in inputs.documents}) != len(inputs.documents):
        raise ValueError("TopicEvidenceInputs contains duplicate document IDs")

    chunker = SemanticTextChunker(ChunkingConfig())
    chunks_by_document = {
        document.document_id: chunker.split_text(document.text, document_id=document.document_id)
        for document in inputs.documents
    }
    empty_chunks = sorted(
        document_id for document_id, chunks in chunks_by_document.items() if not chunks
    )
    if empty_chunks:
        raise ValueError("eligible documents must produce at least one text chunk")

    shortlists: dict[str, list[dict[str, object]]] = {}
    for sub_narrative in inputs.sub_narratives:
        chunk_texts = [
            chunk.text
            for document in inputs.documents
            for chunk in chunks_by_document[document.document_id]
        ]
        raw_scores = scorer(sub_narrative, chunk_texts)
        if len(raw_scores) != len(chunk_texts):
            raise ValueError("scorer must return one score for every document chunk")

        scores_by_document: dict[str, list[float]] = {
            document.document_id: [] for document in inputs.documents
        }
        score_index = 0
        for document in inputs.documents:
            for _chunk in chunks_by_document[document.document_id]:
                raw_score = raw_scores[score_index]
                score_index += 1
                if isinstance(raw_score, bool):
                    raise ValueError("scorer scores must be finite floats")
                score = float(raw_score)
                if not math.isfinite(score):
                    raise ValueError("scorer scores must be finite floats")
                scores_by_document[document.document_id].append(score)

        ranked = sorted(
            (
                (document, max(scores_by_document[document.document_id]))
                for document in inputs.documents
            ),
            key=lambda item: (-item[1], item[0].document_id),
        )
        shortlists[sub_narrative] = [
            {
                "document_id": document.document_id,
                "topic_qrel_grade": document.topic_qrel_grade,
                "model_score": float(score),
                "model_rank": rank,
            }
            for rank, (document, score) in enumerate(ranked[:shortlist_depth], start=1)
        ]

    return {
        "topic_id": inputs.topic_id,
        "pair_count": len(inputs.sub_narratives) * len(inputs.documents),
        "shortlist_depth": shortlist_depth,
        "shortlists": shortlists,
    }


def _load_supplemental_document_records(path: Path, *, eligible_docids: set[str]) -> dict[str, str]:
    if not path.exists():
        return {}
    records: dict[str, str] = {}
    for index, row in enumerate(_read_jsonl(path, "supplemental documents"), start=1):
        document_id = _required_string(row.get("document_id"), f"supplemental documents:{index} document_id")
        text = _required_string(row.get("text"), f"supplemental documents:{index} text")
        response_sha256 = row.get("response_sha256")
        if (
            not isinstance(response_sha256, str)
            or len(response_sha256) != 64
            or any(character not in "0123456789abcdef" for character in response_sha256)
        ):
            raise ValueError(f"supplemental documents:{index} response_sha256 must be lowercase SHA-256")
        if document_id not in eligible_docids:
            raise ValueError(f"supplemental documents:{index} has an ineligible document ID")
        if document_id in records:
            raise ValueError(f"supplemental documents contains a duplicate document ID: {document_id}")
        records[document_id] = text
    return records


def _document_endpoint(index_url: str, document_id: str) -> str:
    base_url = index_url.rstrip("/")
    if not base_url.endswith("/search"):
        raise ValueError("INDEX_URL must name the configured Pyserini search endpoint")
    return f"{base_url.removesuffix('/search')}/doc/{quote(document_id, safe='')}"


def fetch_missing_documents(
    *,
    topic_tsv: Path,
    nuggets_jsonl: Path,
    qrels: Path,
    accepted_union: Path,
    supplemental_documents: Path,
    limit_documents: int | None = None,
) -> dict[str, int]:
    """Fetch absent eligible text exactly once per document into resumable JSONL."""

    if limit_documents is not None and limit_documents <= 0:
        raise ValueError("limit_documents must be positive")
    # Validate the static topic and nugget inputs before making an authenticated request.
    _load_topic_narrative(Path(topic_tsv))
    _load_sub_narratives(Path(nuggets_jsonl))
    eligible_grades = _load_eligible_grades(Path(qrels))
    eligible_docids = set(eligible_grades)
    accepted_texts = _load_document_texts(
        _read_jsonl(Path(accepted_union), "accepted union"),
        label="accepted union",
        topic_scoped=True,
        eligible_docids=eligible_docids,
    )
    supplemental_texts = _load_supplemental_document_records(
        Path(supplemental_documents), eligible_docids=eligible_docids
    )
    missing = missing_document_ids(
        eligible_docids=eligible_docids,
        available_docids=set(accepted_texts).union(supplemental_texts),
    )
    planned = missing if limit_documents is None else missing[:limit_documents]

    repo_root = find_repo_root(Path(__file__))
    load_repo_env(repo_root)
    config = RemotePyseriniConfig.from_env()
    session = rate_limited_session(config)
    destination = Path(supplemental_documents)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fetched = 0
    with destination.open("a", encoding="utf-8") as sink:
        for document_id in planned:
            response = session.get(
                _document_endpoint(config.index_url, document_id),
                headers={
                    "Accept": "application/json",
                    **(
                        {"Authorization": f"Bearer {config.api_token}"}
                        if config.api_token
                        else {}
                    ),
                },
                timeout=30,
                allow_redirects=False,
            )
            raw = response.content
            if response.status_code == 429:
                raise RemotePyseriniThrottled(None)
            response.raise_for_status()
            try:
                payload: object = response.json()
            except ValueError:
                payload = raw.decode("utf-8", errors="replace")
            text = extract_text(payload)
            if not text.strip():
                raise ValueError(f"document endpoint returned no text for document ID {document_id}")
            row = {
                "document_id": document_id,
                "text": text,
                "response_sha256": hashlib.sha256(raw).hexdigest(),
            }
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            fetched += 1
    return {
        "fetched_count": fetched,
        "remaining_missing_count": len(missing) - fetched,
    }


def _model_snapshot() -> Path:
    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshot = cache_root / "models--mixedbread-ai--mxbai-rerank-base-v2" / "snapshots" / MODEL_REVISION
    if not snapshot.is_dir():
        raise FileNotFoundError(
            "the pinned Mixedbread snapshot is unavailable locally; model downloads are forbidden"
        )
    return snapshot


class _LocalMixedbreadScorer:
    """Local-only Mixedbread wrapper that returns untransformed model logits."""

    def __init__(self) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise RuntimeError("Mixedbread scoring requires an available ROCm torch cuda device")
        self._identity = torch.nn.Identity()
        self._model = CrossEncoder(
            str(_model_snapshot()),
            device="cuda",
            local_files_only=True,
            trust_remote_code=False,
            max_length=32_768,
            activation_fn=self._identity,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    def __call__(self, sub_narrative: str, chunk_texts: list[str]) -> list[float]:
        scores = self._model.predict(
            [(sub_narrative, chunk_text) for chunk_text in chunk_texts],
            batch_size=8,
            show_progress_bar=False,
            activation_fn=self._identity,
            apply_softmax=False,
            convert_to_numpy=True,
        )
        return [float(score) for score in scores]


def _default_path(path: Path) -> Path:
    return find_repo_root(Path(__file__)) / path


def _load_cli_inputs(args: argparse.Namespace) -> TopicEvidenceInputs:
    return load_topic213_inputs(
        topic_tsv=args.topic_tsv,
        nuggets_jsonl=args.nuggets_jsonl,
        qrels=args.qrels,
        accepted_union=args.accepted_union,
        supplemental_documents=args.supplemental_documents,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_input_paths(command: argparse.ArgumentParser) -> None:
        command.add_argument("--topic-tsv", type=Path, default=_default_path(DEFAULT_TOPIC_TSV))
        command.add_argument("--nuggets-jsonl", type=Path, default=_default_path(DEFAULT_NUGGETS_JSONL))
        command.add_argument("--qrels", type=Path, default=_default_path(DEFAULT_QRELS))
        command.add_argument("--accepted-union", type=Path, default=_default_path(DEFAULT_ACCEPTED_UNION))
        command.add_argument(
            "--supplemental-documents",
            type=Path,
            default=_default_path(DEFAULT_OUTPUT_DIRECTORY / "supplemental_documents.jsonl"),
        )

    fetch = commands.add_parser("fetch-missing", help="Fetch absent eligible documents once.")
    add_input_paths(fetch)
    fetch.add_argument("--limit-documents", type=int)

    shortlist = commands.add_parser("shortlist", help="Score local document/sub-narrative pairs.")
    add_input_paths(shortlist)
    shortlist.add_argument("--limit-documents", type=int)
    shortlist.add_argument("--shortlist-depth", type=int, default=12)
    shortlist.add_argument(
        "--output",
        type=Path,
        default=_default_path(DEFAULT_OUTPUT_DIRECTORY / "shortlist.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fetch-missing":
        result = fetch_missing_documents(
            topic_tsv=args.topic_tsv,
            nuggets_jsonl=args.nuggets_jsonl,
            qrels=args.qrels,
            accepted_union=args.accepted_union,
            supplemental_documents=args.supplemental_documents,
            limit_documents=args.limit_documents,
        )
        print(
            f"fetched_records={result['fetched_count']} "
            f"remaining_missing={result['remaining_missing_count']}"
        )
        return 0

    inputs = _load_cli_inputs(args)
    if args.limit_documents is not None:
        if args.limit_documents <= 0:
            raise ValueError("limit_documents must be positive")
        inputs = TopicEvidenceInputs(
            topic_id=inputs.topic_id,
            narrative=inputs.narrative,
            sub_narratives=inputs.sub_narratives,
            documents=inputs.documents[: args.limit_documents],
        )
    result = score_sub_narrative_pairs(
        inputs,
        _LocalMixedbreadScorer(),
        shortlist_depth=args.shortlist_depth,
    )
    payload = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "score_representation": "raw_logits",
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"scored_pairs={result['pair_count']} "
        f"shortlists={len(result['shortlists'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
