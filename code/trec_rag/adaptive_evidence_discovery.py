"""One-pass cross-fitted discovery of derived evidence obligations and nuggets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .adaptive_evidence_local_model import (
    DISCOVERY_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SNAPSHOT,
    LocalJsonModel,
)
from .adaptive_evidence_score import load_score_contract, verify_local_scoring


DISCOVERY_PREFLIGHT_SCHEMA_VERSION = "adaptive-evidence-discovery-preflight-v1"
DISCOVERY_PROPOSAL_SCHEMA_VERSION = "adaptive-evidence-discovery-proposal-v1"
DISCOVERY_RECEIPT_SCHEMA_VERSION = "adaptive-evidence-discovery-v1"
RESERVOIR_LIMIT = 10
DISCOVERY_MAX_NEW_TOKENS = 700


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}
_FACT_PATTERNS = (
    re.compile(r"(?:[$€£¥]|\b(?:usd|eur|gbp|dollars?|euros?|pounds?)\b)", re.IGNORECASE),
    re.compile(r"\b\d+(?:[.,]\d+)?\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent|per cent)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december|monday|tuesday|wednesday|thursday|friday|"
        r"saturday|sunday)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:people|users?|patients?|students?|"
        r"kilograms?|metric tons?|megawatts?|gigawatts?|hours?|days?|years?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|"
        r"thousand|million|billion|trillion|dozens?|scores?)\b",
        re.IGNORECASE,
    ),
)
_ABSTRACT_CATEGORY_HEADS = frozenset(
    {
        "access",
        "barriers",
        "benefits",
        "causes",
        "challenges",
        "concerns",
        "consequences",
        "effects",
        "experiences",
        "factors",
        "impacts",
        "issues",
        "mechanisms",
        "needs",
        "opportunities",
        "outcomes",
        "patterns",
        "practices",
        "responses",
        "risks",
    }
)
_LABEL_FORBIDDEN_WORDS = frozenset(
    {
        "a",
        "about",
        "across",
        "after",
        "although",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "during",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "must",
        "of",
        "on",
        "once",
        "or",
        "our",
        "shall",
        "she",
        "should",
        "since",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "these",
        "they",
        "this",
        "those",
        "though",
        "through",
        "to",
        "under",
        "unless",
        "until",
        "upon",
        "was",
        "we",
        "were",
        "when",
        "whereas",
        "while",
        "will",
        "with",
        "within",
        "without",
        "would",
        "you",
        "your",
    }
)
_IRREGULAR_ASSERTION_VERBS = frozenset(
    {
        "became",
        "began",
        "brought",
        "drove",
        "fell",
        "found",
        "gave",
        "got",
        "grew",
        "kept",
        "led",
        "left",
        "made",
        "rose",
        "saw",
        "took",
        "went",
    }
)
_ASSERTION_VERB_BASES = frozenset(
    {
        "adopt",
        "boost",
        "cause",
        "cost",
        "decrease",
        "enable",
        "improve",
        "increase",
        "lead",
        "lower",
        "prevent",
        "produce",
        "provide",
        "raise",
        "reduce",
        "respond",
        "result",
        "save",
    }
)
_LABEL_SENTENCE_PUNCTUATION_RE = re.compile(r"[.!?;,:/—–-]|--|[\r\n]")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_compact_bytes(dict(row)) for row in rows)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    try:
        source = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    output: list[dict[str, object]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line:
            raise ValueError(f"{label}:{line_number} is blank")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label}:{line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number} must be an object")
        if _compact_bytes(value).rstrip(b"\n") != line:
            raise ValueError(f"{label}:{line_number} is not canonical JSON")
        output.append(value)
    return output


def _exclusive_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("xb") as sink:
        sink.write(value)
        sink.flush()
        os.fsync(sink.fileno())


def _reject_protected(topic_ids: Sequence[object]) -> None:
    for topic_id in map(str, topic_ids):
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")


def _require_pilot_topics(topic_ids: Sequence[object]) -> None:
    received = {str(value) for value in topic_ids}
    if received != set(PILOT_TOPIC_IDS):
        raise ValueError("discovery requires exactly pilot topics 219,72,300,84")


def _snapshot_manifest() -> dict[str, object]:
    if not MODEL_SNAPSHOT.is_dir():
        raise RuntimeError(f"pinned local model snapshot is missing: {MODEL_SNAPSHOT}")
    files: list[dict[str, object]] = []
    for path in sorted(MODEL_SNAPSHOT.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        target = path.resolve()
        blob_id = target.name
        content_sha256 = (
            blob_id
            if len(blob_id) == 64 and all(c in "0123456789abcdef" for c in blob_id)
            else _sha256_file(target)
        )
        files.append(
            {
                "name": path.name,
                "bytes": target.stat().st_size,
                "content_sha256": content_sha256,
            }
        )
    if not files or not any(row["name"] == "model.safetensors.index.json" for row in files):
        raise RuntimeError("pinned local model snapshot is incomplete")
    payload = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": files,
    }
    return {**payload, "manifest_sha256": _sha256_bytes(_compact_bytes(payload))}


def _tokens(value: object) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(str(value))}


def _stemmed_content_terms(value: object) -> set[str]:
    output: set[str] = set()
    for token in _tokens(value):
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        if token:
            output.add(token)
    return output


def attach_contract_folds(
    passages: Sequence[Mapping[str, object]],
    documents: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Join authenticated Task 1 fold assignments onto authenticated score rows."""

    folds: dict[tuple[str, str], int] = {}
    for document in documents:
        key = (str(document.get("topic_id", "")), str(document.get("document_id", "")))
        fold = document.get("fold")
        if (
            not all(key)
            or isinstance(fold, bool)
            or fold not in (0, 1)
            or key in folds
        ):
            raise ValueError("contract document fold identity is invalid or duplicate")
        folds[key] = int(fold)
    output: list[dict[str, object]] = []
    for passage in passages:
        key = (str(passage.get("topic_id", "")), str(passage.get("document_id", "")))
        fold = folds.get(key)
        if fold is None:
            raise ValueError("base score document is absent from the contract folds")
        received = passage.get("fold")
        if received is not None and received != fold:
            raise ValueError("base score fold conflicts with the contract")
        output.append({**passage, "fold": fold})
    return output


def build_reservoirs(
    obligations: Sequence[Mapping[str, object]],
    passages: Sequence[Mapping[str, object]],
    *,
    limit: int = 10,
) -> dict[tuple[str, int], list[dict[str, object]]]:
    """Select the highest-scoring distinct documents per O0 and fold."""

    if limit < 1:
        raise ValueError("reservoir limit must be positive")
    parents = {
        str(row["obligation_id"])
        for row in obligations
        if row.get("kind") == "o0"
    }
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in passages:
        parent_id = str(row.get("variant", row.get("obligation_id", "")))
        if parent_id not in parents:
            continue
        fold = row.get("fold")
        if isinstance(fold, bool) or fold not in (0, 1):
            raise ValueError("passage fold must be 0 or 1")
        grouped[(parent_id, int(fold))].append(row)

    output: dict[tuple[str, int], list[dict[str, object]]] = {}
    for parent_id in sorted(parents):
        for fold in (0, 1):
            ranked = sorted(
                grouped[(parent_id, fold)],
                key=lambda row: (
                    -float(row["score"]),
                    str(row["document_id"]),
                    str(row.get("window_id", "")),
                ),
            )
            seen: set[str] = set()
            selected: list[dict[str, object]] = []
            for row in ranked:
                document_id = str(row["document_id"])
                if document_id in seen:
                    continue
                seen.add(document_id)
                selected.append(dict(row))
                if len(selected) == limit:
                    break
            output[(parent_id, fold)] = selected
    return output


def _looks_like_answer_fact(label: str) -> bool:
    return any(pattern.search(label) for pattern in _FACT_PATTERNS)


def _is_abstract_o1_category(label: str) -> bool:
    """Accept only short nominal labels ending in a frozen abstract head noun."""

    if _LABEL_SENTENCE_PUNCTUATION_RE.search(label) or _looks_like_answer_fact(label):
        return False
    tokens = [token.casefold() for token in _TOKEN_RE.findall(label)]
    if not 2 <= len(tokens) <= 8 or tokens[-1] not in _ABSTRACT_CATEGORY_HEADS:
        return False
    if any(token in _LABEL_FORBIDDEN_WORDS for token in tokens):
        return False
    modifiers = tokens[:-1]
    return not any(
        token in _IRREGULAR_ASSERTION_VERBS
        or (len(token) > 4 and token.endswith(("ed", "ing")))
        or (token.endswith("s") and token[:-1] in _ASSERTION_VERB_BASES)
        for token in modifiers
    )


def _parent_scope_preserved(
    proposal: Mapping[str, object],
    parent: Mapping[str, object],
) -> bool:
    def content_terms(value: object) -> set[str]:
        return _stemmed_content_terms(value) - _CONTENT_STOPWORDS

    def listed_terms(name: str) -> set[str]:
        raw = parent.get(name, [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return set()
        return set().union(*(content_terms(value) for value in raw))

    parent_text = content_terms(parent.get("text", ""))
    parent_subject = listed_terms("anchor_terms") or parent_text
    parent_relation = listed_terms("relation_terms") | parent_text
    parent_population = listed_terms("population_terms")
    if not parent_population:
        parent_population = listed_terms("relation_terms") | parent_text
    parent_domain = listed_terms("domain_terms") or (
        listed_terms("anchor_terms") | parent_text
    )

    proposed = {
        name: content_terms(proposal.get(name, ""))
        for name in ("subject", "population", "domain", "relation")
    }
    allowed = {
        "subject": parent_subject,
        "population": parent_population,
        "domain": parent_domain,
        "relation": parent_relation,
    }
    if any(not proposed[name] or not proposed[name].issubset(allowed[name]) for name in allowed):
        return False

    proposed_domain = proposed["domain"]
    for pattern in parent.get("wrong_domain_patterns", []):
        pattern_terms = content_terms(pattern)
        if pattern_terms and pattern_terms.issubset(proposed_domain):
            return False
    return True


def validate_proposal(
    proposal: Mapping[str, object],
    parent: Mapping[str, object],
    *,
    opposite_fold_support: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply deterministic O1 scope and independent-fold acceptance checks."""

    reasons: list[str] = []
    if proposal.get("parent_id") != parent.get("obligation_id"):
        reasons.append("parent")
    required = (
        "label",
        "scope_rationale",
        "subject",
        "population",
        "domain",
        "relation",
        "support_span",
    )
    if any(
        name not in proposal
        or not isinstance(proposal.get(name), str)
        or not str(proposal.get(name)).strip()
        for name in required
    ):
        reasons.append("schema")
    if not _is_abstract_o1_category(str(proposal.get("label", ""))):
        reasons.append("candidate_answer")
    distinct = {
        str(row["document_id"])
        for row in opposite_fold_support
        if row.get("qualified") is True and row.get("document_id") is not None
    }
    source_document_id = str(
        proposal.get("source_document_id", proposal.get("document_id", ""))
    )
    if not distinct or source_document_id in distinct:
        reasons.append("cross_fold_support")
    if not _parent_scope_preserved(proposal, parent):
        reasons.append("scope")
    unique_reasons = sorted(set(reasons))
    return {"accepted": not unique_reasons, "reasons": unique_reasons}


def _triple_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return tuple(
        " ".join(sorted(_stemmed_content_terms(row.get(name, ""))))
        for name in ("subject", "relation", "object")
    )  # type: ignore[return-value]


_SRO_NONATOMIC_RE = re.compile(
    r"(?:[,;:/]|--|[—–]|[\r\n]|\b(?:and|or|but|when|if|after|before|while|"
    r"because|although|though|whereas|unless|since|once|as|which|who|that)\b)",
    re.IGNORECASE,
)


def _without_one_trailing_terminator(value: str) -> str:
    stripped = value.strip()
    if stripped.endswith((".", "!", "?")):
        return stripped[:-1].rstrip()
    return stripped


def validate_nugget_atomicity(nugget: Mapping[str, object]) -> dict[str, object]:
    """Require one fully quoted, non-coordinated subject-relation-object fact."""

    reasons: list[str] = []
    fields: dict[str, str] = {}
    for name in ("subject", "relation", "object"):
        value = nugget.get(name)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"{name}_missing")
            continue
        fields[name] = value.strip()
        if _SRO_NONATOMIC_RE.search(value) or re.search(r"[.!?]", value):
            reasons.append(f"{name}_coordination")

    support = nugget.get("support_span")
    if not isinstance(support, str) or not support.strip():
        reasons.append("support_missing")
    else:
        support = support.strip()
        support_body = _without_one_trailing_terminator(support)
        if re.search(r"[.!?]", support_body):
            reasons.append("support_multiple_sentences")
        if _SRO_NONATOMIC_RE.search(support_body):
            reasons.append("support_multiple_clauses")
        if len(fields) == 3:
            support_terms = _stemmed_content_terms(support)
            assertion_terms = set().union(
                *(_stemmed_content_terms(fields[name]) for name in fields)
            )
            if not assertion_terms or not assertion_terms.issubset(support_terms):
                reasons.append("unsupported_sro")

    unique_reasons = sorted(set(reasons))
    return {"accepted": not unique_reasons, "reasons": unique_reasons}


def merge_nuggets(
    nuggets: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Merge exact triples, then lexical paraphrases at Jaccard >= 0.80."""

    groups: list[list[dict[str, object]]] = []
    by_triple: dict[tuple[str, str, str], int] = {}
    for raw in nuggets:
        row = dict(raw)
        key = _triple_key(row)
        group_index = by_triple.get(key)
        if group_index is None:
            by_triple[key] = len(groups)
            groups.append([row])
        else:
            groups[group_index].append(row)

    merged: list[list[dict[str, object]]] = []
    for group in groups:
        terms = _stemmed_content_terms(group[0].get("object", ""))
        target: list[dict[str, object]] | None = None
        for candidate in merged:
            other = _stemmed_content_terms(candidate[0].get("object", ""))
            union = terms | other
            jaccard = len(terms & other) / len(union) if union else 1.0
            if jaccard >= 0.80:
                target = candidate
                break
        if target is None:
            merged.append(group)
        else:
            target.extend(group)

    output: list[dict[str, object]] = []
    for rows in merged:
        documents = sorted({str(row.get("document_id", "")) for row in rows})
        representative = min(
            rows,
            key=lambda row: (
                str(row.get("subject", "")),
                str(row.get("relation", "")),
                str(row.get("object", "")),
                str(row.get("document_id", "")),
            ),
        )
        output.append(
            {
                **representative,
                "supporting_document_ids": documents,
                "support_count": len(documents),
                "singleton": len(documents) == 1,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row.get("topic_id", "")),
            str(row.get("parent_id", "")),
            str(row.get("subject", "")),
            str(row.get("relation", "")),
            str(row.get("object", "")),
        ),
    )


def validate_model_response(
    response: Mapping[str, object],
    passages: Sequence[Mapping[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    """Retain only model records whose quoted span occurs in its supplied passage."""

    by_document = {
        str(row["document_id"]): str(
            row.get("passage_text", row.get("window_text", ""))
        )
        for row in passages
    }
    accepted: dict[str, list[dict[str, object]]] = {"o1": [], "n1": []}
    rejected: list[dict[str, object]] = []
    for kind in ("o1", "n1"):
        raw_rows = response.get(kind, [])
        if not isinstance(raw_rows, list):
            raise ValueError(f"model response {kind} must be an array")
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                rejected.append({"kind": kind, "reason": "schema"})
                continue
            row = dict(raw)
            document_id = str(row.get("support_document_id", ""))
            span = str(row.get("support_span", ""))
            passage = by_document.get(document_id)
            if passage is None:
                rejected.append({**row, "kind": kind, "reason": "support_document"})
            elif not span or span not in passage:
                rejected.append({**row, "kind": kind, "reason": "support_span"})
            elif kind == "n1" and not (
                atomicity := validate_nugget_atomicity(row)
            )["accepted"]:
                rejected.append(
                    {
                        **row,
                        "kind": kind,
                        "reason": "n1_atomicity",
                        "atomicity_reasons": atomicity["reasons"],
                    }
                )
            else:
                accepted[kind].append(row)
    return accepted, rejected


def extract_repeated_phrases(
    passages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Emit 2--5 analyzed-token phrases occurring in documents in both folds."""

    occurrences: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    for row in passages:
        document_id = str(row.get("document_id", ""))
        fold = row.get("fold")
        if not document_id or isinstance(fold, bool) or fold not in (0, 1):
            raise ValueError("phrase-control passages require document and binary fold")
        passage = str(row.get("passage_text", row.get("window_text", "")))
        normalized_content = " ".join(passage.split()).casefold()
        content_sha256 = _sha256_bytes(normalized_content.encode("utf-8"))
        analyzed = [
            token.casefold()
            for token in _TOKEN_RE.findall(passage)
            if token.casefold() not in _CONTENT_STOPWORDS
        ]
        seen: set[str] = set()
        for size in range(2, 6):
            for start in range(0, len(analyzed) - size + 1):
                seen.add(" ".join(analyzed[start : start + size]))
        for phrase in seen:
            occurrences[phrase].add((document_id, content_sha256, int(fold)))
    output: list[dict[str, object]] = []
    for phrase, identities in occurrences.items():
        documents = sorted({document_id for document_id, _hash, _fold in identities})
        hashes = sorted({_hash for _document_id, _hash, _fold in identities})
        folds = sorted({fold for _document_id, _hash, fold in identities})
        hashes_by_fold = {
            fold: {
                content_hash
                for _document_id, content_hash, identity_fold in identities
                if identity_fold == fold
            }
            for fold in (0, 1)
        }
        distinct_across_folds = any(
            left != right
            for left in hashes_by_fold[0]
            for right in hashes_by_fold[1]
        )
        if len(documents) >= 2 and len(hashes) >= 2 and distinct_across_folds:
            output.append(
                {
                    "phrase": phrase,
                    "token_count": len(phrase.split()),
                    "document_ids": documents,
                    "content_sha256s": hashes,
                    "folds": folds,
                }
            )
    return sorted(output, key=lambda row: (row["token_count"], row["phrase"]))


def _phrase_control_scope_preserved(
    phrase: str,
    parent: Mapping[str, object],
) -> bool:
    """Check phrase scope without pretending the phrase is a structured O1."""

    def content_terms(value: object) -> set[str]:
        return _stemmed_content_terms(value) - _CONTENT_STOPWORDS

    def listed_terms(name: str) -> set[str]:
        raw = parent.get(name, [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return set()
        return set().union(*(content_terms(value) for value in raw))

    phrase_terms = content_terms(phrase)
    anchors = listed_terms("anchor_terms")
    scope_terms = set().union(
        content_terms(parent.get("text", "")),
        anchors,
        listed_terms("relation_terms"),
        listed_terms("population_terms"),
        listed_terms("domain_terms"),
    )
    if (
        not phrase_terms
        or not phrase_terms.issubset(scope_terms)
        or (anchors and not phrase_terms.intersection(anchors))
    ):
        return False
    return not any(
        (pattern_terms := content_terms(pattern))
        and pattern_terms.issubset(phrase_terms)
        for pattern in parent.get("wrong_domain_patterns", [])
    )


def build_phrase_controls(
    parents: Sequence[Mapping[str, object]],
    passages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build deterministic no-LLM phrase controls for fresh preflight output."""

    output: list[dict[str, object]] = []
    for parent in sorted(parents, key=lambda row: str(row.get("obligation_id", ""))):
        parent_id = str(parent.get("obligation_id", ""))
        parent_passages = [
            row for row in passages if str(row.get("parent_id", "")) == parent_id
        ]
        for row in extract_repeated_phrases(parent_passages):
            phrase = str(row["phrase"])
            hashes = row.get("content_sha256s", [])
            documents = row.get("document_ids", [])
            content_distinct = (
                isinstance(hashes, list)
                and len(set(map(str, hashes))) >= 2
                and isinstance(documents, list)
                and len(set(map(str, documents))) >= 2
                and row.get("folds") == [0, 1]
            )
            scope_preserved = _phrase_control_scope_preserved(phrase, parent)
            abstract_category = _is_abstract_o1_category(phrase)
            output.append(
                {
                    "schema_version": "adaptive-evidence-repeated-phrase-v1",
                    "topic_id": str(parent.get("topic_id", "")),
                    "parent_id": parent_id,
                    **row,
                    "content_distinct": content_distinct,
                    "scope_preserved": scope_preserved,
                    "candidate_answer": not abstract_category,
                    "abstract_category": abstract_category,
                    "accepted_control": (
                        content_distinct and scope_preserved and abstract_category
                    ),
                }
            )
    return sorted(
        output,
        key=lambda row: (
            str(row["topic_id"]),
            str(row["parent_id"]),
            int(row["token_count"]),
            str(row["phrase"]),
        ),
    )


def build_derived_query(
    broad: Mapping[str, object],
    parent: Mapping[str, object],
    label: str,
) -> str:
    narrative = str(broad.get("text", broad.get("query", "")))
    parent_text = str(parent.get("text", ""))
    if not narrative or not parent_text or not label.strip():
        raise ValueError("derived query requires narrative, complete parent, and label")
    return (
        f"{narrative}\n\nExplicit obligation:\n{parent_text}"
        f"\n\nCorpus-derived sub-obligation:\n{label.strip()}"
    )


def freeze_o1(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Freeze at most one accepted child per parent and four per topic."""

    def lexical_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
        normalized_label = " ".join(_TOKEN_RE.findall(str(row.get("label", "")).casefold()))
        return (
            str(row.get("topic_id", "")).casefold(),
            normalized_label,
            str(row.get("parent_id", "")).casefold(),
            str(row.get("proposal_id", row.get("obligation_id", ""))).casefold(),
        )

    ranked = sorted(
        (dict(row) for row in rows if row.get("accepted") is True),
        key=lexical_key,
    )
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    seen_parents: set[str] = set()
    topic_counts: dict[str, int] = defaultdict(int)
    for row in ranked:
        topic_id = str(row.get("topic_id", ""))
        parent_id = str(row.get("parent_id", ""))
        if parent_id in seen_parents or topic_counts[topic_id] >= 4:
            rejected.append({**row, "accepted": False, "reasons": ["freeze_limit"]})
            continue
        seen_parents.add(parent_id)
        topic_counts[topic_id] += 1
        accepted.append(row)
    return accepted, rejected


_DISCOVERY_INSTRUCTIONS = (
    "Use only the supplied parent and passages. You must preserve the parent "
    "subject, population, domain, and relation. Propose abstract O1 categories "
    "separately from specific N1 facts. Quote support spans as exact substrings "
    "from the supplied passages; never use outside knowledge. If the passages "
    "do not support a record, return unsupported rather than inventing evidence. "
    "O1 labels must be short abstract information categories already inside the "
    "frozen parent scope, never candidate answers, assertions, numbers, dates, "
    "currencies, quantities, causes, or outcomes. "
    "Every N1 must be exactly one atomic subject-relation-object assertion "
    "supported by one exact span: no coordinated subjects, relations, or objects; "
    "no lists, multiple clauses, or multiple sentences."
)
_SCHEMA_INSTRUCTIONS = (
    "The response_json_schema is authoritative. Return exactly one JSON object "
    "matching it, with every required key and no additional keys; do not wrap it "
    "in Markdown or explanatory text."
)


def _proposal_prompt_contract_sha256() -> str:
    return _sha256_bytes(
        _compact_bytes(
            {
                "instructions": f"{_DISCOVERY_INSTRUCTIONS} {_SCHEMA_INSTRUCTIONS}",
                "response_json_schema": DISCOVERY_SCHEMA,
            }
        )
    )


def build_discovery_messages(
    parent: Mapping[str, object],
    passages: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Build the frozen one-parent, one-fold discovery prompt."""

    payload = {
        "parent": dict(parent),
        "passages": [
            {
                "document_id": str(row["document_id"]),
                "fold": int(row["fold"]),
                "passage_text": str(
                    row.get("passage_text", row.get("window_text", ""))
                ),
            }
            for row in passages
        ],
        "response_json_schema": DISCOVERY_SCHEMA,
    }
    return [
        {
            "role": "system",
            "content": f"{_DISCOVERY_INSTRUCTIONS} {_SCHEMA_INSTRUCTIONS}",
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    ]


def _legacy_integration_preflight_messages(
    parent: Mapping[str, object],
    passages: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Reconstruct the exact first-reservoir prompt rejected before artifacts."""

    payload = {
        "parent": dict(parent),
        "passages": [
            {
                "document_id": str(row["document_id"]),
                "fold": int(row["fold"]),
                "passage_text": str(
                    row.get("passage_text", row.get("window_text", ""))
                ),
            }
            for row in passages
        ],
        "response_contract": {
            "status": "supported or unsupported",
            "o1": "zero or one abstract sub-obligation category",
            "n1": "zero to four atomic subject-relation-object facts",
        },
    }
    return [
        {"role": "system", "content": _DISCOVERY_INSTRUCTIONS},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    ]


def _sentence_spans(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text)
        if match.group(0).strip()
    ]


def qualify_opposite_support(
    proposal: Mapping[str, object],
    parent: Mapping[str, object],
    score_rows: Sequence[Mapping[str, object]],
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Deterministically qualify exact corroborating spans from opposite-fold scores."""

    source_fold = proposal.get("source_fold", proposal.get("fold"))
    if isinstance(source_fold, bool) or source_fold not in (0, 1):
        raise ValueError("proposal source_fold must be 0 or 1")
    obligation_id = str(
        proposal.get("obligation_id", proposal.get("proposal_id", ""))
    )
    source_document = str(
        proposal.get("source_document_id", proposal.get("document_id", ""))
    )
    source_sha256 = str(proposal.get("source_document_sha256", ""))
    parent_anchor = set().union(
        *(_tokens(value) for value in parent.get("anchor_terms", []))
    )
    parent_relation = set().union(
        *(_tokens(value) for value in parent.get("relation_terms", []))
    )
    label_terms = _tokens(proposal.get("label", "")) - _CONTENT_STOPWORDS
    parent_terms = _tokens(parent.get("text", "")) | parent_anchor | parent_relation
    specialized_terms = label_terms - parent_terms
    if not specialized_terms:
        specialized_terms = label_terms
    wrong_domains = [
        str(value).casefold()
        for value in parent.get("wrong_domain_patterns", [])
        if str(value).strip()
    ]

    ranked = sorted(
        (
            row
            for row in score_rows
            if str(row.get("variant", row.get("obligation_id", "")))
            == obligation_id
            and row.get("fold") == 1 - int(source_fold)
        ),
        key=lambda row: (
            -float(row["score"]),
            str(row.get("document_id", "")),
            str(row.get("window_id", "")),
        ),
    )
    selected: list[Mapping[str, object]] = []
    seen_documents: set[str] = set()
    for row in ranked:
        document_id = str(row.get("document_id", ""))
        if not document_id or document_id in seen_documents:
            continue
        seen_documents.add(document_id)
        selected.append(row)
        if len(selected) == limit:
            break

    output: list[dict[str, object]] = []
    for row in selected:
        document_id = str(row.get("document_id", ""))
        document_sha256 = str(row.get("document_sha256", ""))
        if document_id == source_document or (
            source_sha256 and document_sha256 == source_sha256
        ):
            continue
        passage = str(row.get("window_text", row.get("passage_text", "")))
        passage_folded = passage.casefold()
        if any(pattern in passage_folded for pattern in wrong_domains):
            continue
        support_span = ""
        for sentence in _sentence_spans(passage):
            terms = _tokens(sentence)
            if (
                (not parent_anchor or bool(terms & parent_anchor))
                and (not parent_relation or bool(terms & parent_relation))
                and (not specialized_terms or bool(terms & specialized_terms))
            ):
                support_span = sentence
                break
        if not support_span or support_span not in passage:
            continue
        output.append(
            {
                "document_id": document_id,
                "document_sha256": document_sha256,
                "fold": int(row["fold"]),
                "window_id": str(row.get("window_id", "")),
                "score": float(row["score"]),
                "support_span": support_span,
                "passage_text": passage,
                "span_valid": True,
                "anchor_match": True,
                "relation_match": True,
                "domain_coherent": True,
                "nonduplicate": True,
                "qualified": True,
            }
        )
    return output


def finalize_discovery_records(
    *,
    proposals: Sequence[Mapping[str, object]],
    nuggets: Sequence[Mapping[str, object]],
    parents: Sequence[Mapping[str, object]],
    broad_by_topic: Mapping[str, Mapping[str, object]],
    score_rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Validate, cap, and freeze O1/N1 records without another model pass."""

    parent_by_id = {str(row["obligation_id"]): row for row in parents}
    validated: list[dict[str, object]] = []
    initially_rejected: list[dict[str, object]] = []
    for raw in proposals:
        proposal = dict(raw)
        parent_id = str(proposal.get("parent_id", ""))
        parent = parent_by_id.get(parent_id)
        if parent is None:
            initially_rejected.append(
                {
                    **proposal,
                    "accepted": False,
                    "status": "unsupported",
                    "reasons": ["parent"],
                    "opposite_fold_support": [],
                }
            )
            continue
        support = qualify_opposite_support(proposal, parent, score_rows)
        decision = validate_proposal(
            proposal,
            parent,
            opposite_fold_support=support,
        )
        source_document = str(
            proposal.get("source_document_id", proposal.get("document_id", ""))
        )
        document_ids = {
            source_document,
            *(str(row["document_id"]) for row in support),
        }
        document_ids.discard("")
        source_hashes = {
            str(proposal.get("source_document_sha256", "")),
            *(str(row.get("document_sha256", "")) for row in support),
        }
        source_hashes.discard("")
        row = {
            **proposal,
            **decision,
            "status": "accepted" if decision["accepted"] else "unsupported",
            "validation_method": "deterministic_opposite_fold_minilm",
            "opposite_fold_support": support,
            "validating_document_count": len(document_ids),
            "independent_stream_count": len(
                {
                    int(proposal.get("source_fold", proposal.get("fold", -1))),
                    *(int(item["fold"]) for item in support),
                }
            ),
            "source_diversity": len(source_hashes),
            "parent_local_rank": int(proposal.get("source_parent_rank", 0)),
        }
        validated.append(row)
        if not decision["accepted"]:
            initially_rejected.append(row)

    accepted, freeze_rejected = freeze_o1(validated)
    frozen: list[dict[str, object]] = []
    for row in accepted:
        topic_id = str(row["topic_id"])
        parent = parent_by_id[str(row["parent_id"])]
        broad = broad_by_topic.get(topic_id)
        if broad is None:
            raise ValueError(f"broad obligation for topic {topic_id} is missing")
        frozen.append(
            {
                **row,
                "kind": "o1",
                "status": "accepted",
                "query": build_derived_query(broad, parent, str(row["label"])),
            }
        )
    rejected = [
        *initially_rejected,
        *({**row, "status": "unsupported"} for row in freeze_rejected),
    ]
    atomic_nuggets: list[dict[str, object]] = []
    rejected_n1: list[dict[str, object]] = []
    for raw in nuggets:
        nugget = dict(raw)
        atomicity = validate_nugget_atomicity(nugget)
        if atomicity["accepted"]:
            atomic_nuggets.append(nugget)
        else:
            rejected_n1.append(
                {
                    **nugget,
                    "kind": "n1",
                    "status": "unsupported",
                    "reasons": atomicity["reasons"],
                }
            )
    accepted_n1 = [
        {**row, "kind": "n1", "status": "accepted"}
        for row in merge_nuggets(atomic_nuggets)
    ]
    return {
        "validated_o1": sorted(
            validated,
            key=lambda row: str(row.get("proposal_id", row.get("obligation_id", ""))),
        ),
        "accepted_o1": sorted(
            frozen,
            key=lambda row: (str(row["topic_id"]), str(row["parent_id"])),
        ),
        "rejected_o1": sorted(
            rejected,
            key=lambda row: str(row.get("proposal_id", row.get("obligation_id", ""))),
        ),
        "accepted_n1": accepted_n1,
        "rejected_n1": rejected_n1,
    }


def run_discovery_preflight(
    *,
    contract_dir: Path,
    scores_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Authenticate inputs, freeze reservoirs, and probe the pinned local model."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"create-only discovery output exists: {destination}")
    manifest_path = Path(contract_dir) / "manifest.json"
    manifest = _read_json(manifest_path, "contract manifest metadata")
    topic_ids = manifest.get("topic_ids")
    if not isinstance(topic_ids, list):
        raise ValueError("contract manifest topic_ids must be an array")
    _reject_protected(topic_ids)
    _require_pilot_topics(topic_ids)

    contract = load_score_contract(Path(contract_dir))
    obligations = list(contract["obligations"])  # type: ignore[index]
    _reject_protected([row.get("topic_id") for row in obligations])
    parents = [dict(row) for row in obligations if row.get("kind") == "o0"]
    broad = [dict(row) for row in obligations if row.get("kind") == "broad"]
    if len(parents) != 24 or len(broad) != 4:
        raise ValueError("discovery requires exactly 24 O0 and four broad obligations")

    base_receipt = verify_local_scoring(Path(scores_dir))
    if (
        base_receipt.get("qrels_opened") is not False
        or base_receipt.get("network_call_count") != 0
        or base_receipt.get("retrieval_call_count") != 0
        or base_receipt.get("hosted_inference_call_count") != 0
        or base_receipt.get("paid_call_count") != 0
    ):
        raise ValueError("base scores violate the frozen safety receipt")
    o0_ids = {str(row["obligation_id"]) for row in parents}
    shard_records = base_receipt.get("shards")
    if not isinstance(shard_records, list):
        raise ValueError("base score receipt shard inventory is missing")
    base_rows: list[dict[str, object]] = []
    for shard in shard_records:
        if not isinstance(shard, Mapping):
            raise ValueError("base score shard receipt is invalid")
        obligation_id = str(shard.get("obligation_id", ""))
        if obligation_id not in o0_ids:
            continue
        relative = shard.get("path")
        if not isinstance(relative, str) or not relative.startswith("shards/"):
            raise ValueError("base score shard path is invalid")
        base_rows.extend(
            _read_jsonl(Path(scores_dir) / relative, f"base scores {obligation_id}")
        )
    base_rows = attach_contract_folds(
        base_rows,
        list(contract["documents"]),  # type: ignore[index]
    )
    reservoirs = build_reservoirs(parents, base_rows, limit=RESERVOIR_LIMIT)
    reservoir_rows: list[dict[str, object]] = []
    for (parent_id, fold), rows in sorted(reservoirs.items()):
        if len(rows) != RESERVOIR_LIMIT:
            raise ValueError(f"reservoir {parent_id} fold {fold} does not contain ten documents")
        if len({str(row["document_id"]) for row in rows}) != RESERVOIR_LIMIT:
            raise ValueError("reservoir document identities are not distinct")
        for rank, row in enumerate(rows, start=1):
            reservoir_rows.append(
                {
                    "schema_version": "adaptive-evidence-reservoir-v1",
                    "topic_id": str(row["topic_id"]),
                    "parent_id": parent_id,
                    "fold": fold,
                    "reservoir_rank": rank,
                    "document_id": str(row["document_id"]),
                    "document_sha256": str(row["document_sha256"]),
                    "parent_local_rank": int(row["rank"]),
                    "window_id": str(row["window_id"]),
                    "passage_text": str(row["window_text"]),
                    "base_minilm_score": float(row["score"]),
                    "base_model": str(row["model"]),
                    "base_model_revision": str(row["model_revision"]),
                }
            )

    repeated_rows = build_phrase_controls(parents, reservoir_rows)

    snapshot = _snapshot_manifest()
    model = LocalJsonModel()
    probe = model.generate(
        [
            {
                "role": "system",
                "content": "Return one exact JSON object with no markdown.",
            },
            {
                "role": "user",
                "content": '{"status":"unsupported","o1":[],"n1":[]}',
            },
        ],
        DISCOVERY_SCHEMA,
        max_new_tokens=64,
    )
    if probe != {"status": "unsupported", "o1": [], "n1": []}:
        raise ValueError("local model schema probe did not return the frozen sentinel")
    model_receipt = model.execution_receipt()
    if model_receipt.get("generation_count") != 1:
        raise ValueError("local model schema probe count differs")

    payloads = {
        "parents.jsonl": _jsonl_bytes(parents),
        "broad.jsonl": _jsonl_bytes(broad),
        "reservoirs.jsonl": _jsonl_bytes(reservoir_rows),
        "repeated_phrases.jsonl": _jsonl_bytes(repeated_rows),
    }
    artifacts = {
        name: {
            "rows": len(payload.splitlines()),
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        for name, payload in payloads.items()
    }
    receipt: dict[str, object] = {
        "schema_version": DISCOVERY_PREFLIGHT_SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(PILOT_TOPIC_IDS),
        "parent_count": len(parents),
        "folds": [0, 1],
        "reservoir_limit": RESERVOIR_LIMIT,
        "reservoir_count": len(reservoir_rows),
        "reservoir_group_count": len(reservoirs),
        "repeated_phrase_count": len(repeated_rows),
        "accepted_control_phrase_count": sum(
            row["accepted_control"] is True for row in repeated_rows
        ),
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "external_cost_usd": 0.0,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot": snapshot,
        "model_preflight": model_receipt,
        "prompt_sha256": _sha256_bytes(_DISCOVERY_INSTRUCTIONS.encode("utf-8")),
        "schema_sha256": _sha256_bytes(_compact_bytes(DISCOVERY_SCHEMA)),
        "schema": DISCOVERY_SCHEMA,
        "artifacts": artifacts,
        "bindings": {
            "contract": {
                "path": str(Path(contract_dir).resolve()),
                "summary_sha256": _sha256_file(Path(contract_dir) / "summary.json"),
            },
            "base_scores": {
                "path": str(Path(scores_dir).resolve()),
                "receipt_sha256": _sha256_file(Path(scores_dir) / "receipt.json"),
                "completed_window_count": base_receipt["completed_window_count"],
            },
            "code_sha256": {
                "adaptive_evidence_discovery.py": _sha256_file(Path(__file__)),
                "adaptive_evidence_local_model.py": _sha256_file(
                    Path(__file__).with_name("adaptive_evidence_local_model.py")
                ),
                "adaptive_evidence_score.py": _sha256_file(
                    Path(__file__).with_name("adaptive_evidence_score.py")
                ),
            },
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    for name, payload in payloads.items():
        _exclusive_bytes(destination / name, payload)
    _exclusive_bytes(destination / "preflight.json", _pretty_bytes(receipt))
    return receipt


def _load_discovery_preflight(
    output_dir: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    root = Path(output_dir)
    receipt = _read_json(root / "preflight.json", "discovery preflight")
    if (
        receipt.get("schema_version") != DISCOVERY_PREFLIGHT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or receipt.get("folds") != [0, 1]
        or receipt.get("reservoir_limit") != RESERVOIR_LIMIT
        or receipt.get("qrels_opened") is not False
        or receipt.get("network_call_count") != 0
        or receipt.get("retrieval_call_count") != 0
        or receipt.get("hosted_inference_call_count") != 0
        or receipt.get("paid_call_count") != 0
        or receipt.get("model") != MODEL_ID
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("prompt_sha256")
        != _sha256_bytes(_DISCOVERY_INSTRUCTIONS.encode("utf-8"))
        or receipt.get("schema_sha256")
        != _sha256_bytes(_compact_bytes(DISCOVERY_SCHEMA))
    ):
        raise ValueError("discovery preflight differs from the frozen contract")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("discovery preflight artifact bindings are missing")
    loaded: dict[str, list[dict[str, object]]] = {}
    for name in ("parents.jsonl", "broad.jsonl", "reservoirs.jsonl"):
        binding = artifacts.get(name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"discovery preflight binding {name} is missing")
        path = root / name
        rows = _read_jsonl(path, name)
        if (
            binding.get("rows") != len(rows)
            or binding.get("bytes") != path.stat().st_size
            or binding.get("sha256") != _sha256_file(path)
        ):
            raise ValueError(f"discovery preflight artifact {name} differs")
        loaded[name] = rows
    parents = loaded["parents.jsonl"]
    broad = loaded["broad.jsonl"]
    reservoirs = loaded["reservoirs.jsonl"]
    _reject_protected(
        [row.get("topic_id") for row in parents]
        + [row.get("topic_id") for row in broad]
        + [row.get("topic_id") for row in reservoirs]
    )
    if (
        len(parents) != 24
        or len(broad) != 4
        or len(reservoirs) != 24 * 2 * RESERVOIR_LIMIT
    ):
        raise ValueError("discovery preflight row counts differ")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in reservoirs:
        groups[(str(row["parent_id"]), int(row["fold"]))].add(
            str(row["document_id"])
        )
    if len(groups) != 48 or any(len(documents) != 10 for documents in groups.values()):
        raise ValueError("discovery preflight reservoirs are incomplete")
    return receipt, parents, broad, reservoirs


def run_proposal_pass(
    output_dir: Path,
) -> dict[str, object]:
    """Run exactly one local Qwen proposal/atomization pass over all reservoirs."""

    root = Path(output_dir)
    proposal_names = (
        "integration_preflight_failure.json",
        "corrected_pass_started.json",
        "proposed_o1.jsonl",
        "proposed_n1.jsonl",
        "rejected_model_records.jsonl",
        "prompt_receipts.jsonl",
        "proposal_receipt.json",
    )
    if any((root / name).exists() for name in proposal_names):
        raise FileExistsError("one-pass discovery proposal output already exists")
    preflight, parents, broad, reservoirs = _load_discovery_preflight(root)
    parent_by_id = {str(row["obligation_id"]): row for row in parents}
    broad_by_topic = {str(row["topic_id"]): row for row in broad}
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in reservoirs:
        groups[(str(row["parent_id"]), int(row["fold"]))].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["reservoir_rank"]))

    first_parent_id = sorted(parent_by_id)[0]
    first_passages = groups[(first_parent_id, 0)]
    legacy_messages = _legacy_integration_preflight_messages(
        parent_by_id[first_parent_id],
        first_passages,
    )
    integration_failure = {
        "schema_version": "adaptive-evidence-integration-preflight-failure-v1",
        "status": "failed",
        "phase": "model_facing_schema_integration_preflight",
        "completed_proposal_pass_count": 0,
        "qwen_reservoir_generation_call_count": 1,
        "proposal_artifact_count": 0,
        "parent_id": first_parent_id,
        "source_fold": 0,
        "messages": legacy_messages,
        "messages_sha256": _sha256_bytes(_compact_bytes(legacy_messages)),
        "schema": DISCOVERY_SCHEMA,
        "schema_sha256": _sha256_bytes(_compact_bytes(DISCOVERY_SCHEMA)),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot_manifest_sha256": preflight["model_snapshot"][  # type: ignore[index]
            "manifest_sha256"
        ],
        "model_loading": {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "torch_dtype": "bfloat16",
            "eval_mode": True,
            "device": "cuda",
            "execution_backend": "rocm",
            "seed": 0,
            "do_sample": False,
        },
        "error_type": "ValueError",
        "error_message": "JSON schema $ is missing status",
        "error_location": (
            "adaptive_evidence_local_model.LocalJsonModel.generate."
            "validate_json_schema"
        ),
        "model_output_retained": False,
        "preflight_sha256": _sha256_file(root / "preflight.json"),
        "resolution": "user_authorized_exact_schema_prompt_correction",
    }
    _exclusive_bytes(
        root / "integration_preflight_failure.json",
        _pretty_bytes(integration_failure),
    )
    corrected_marker = {
        "schema_version": "adaptive-evidence-corrected-pass-start-v1",
        "status": "started",
        "corrected_completed_pass_ordinal": 1,
        "expected_generation_count": 48,
        "no_further_retry_authorized": True,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_contract_sha256": _proposal_prompt_contract_sha256(),
        "schema_sha256": _sha256_bytes(_compact_bytes(DISCOVERY_SCHEMA)),
        "integration_preflight_failure_sha256": _sha256_file(
            root / "integration_preflight_failure.json"
        ),
    }
    _exclusive_bytes(
        root / "corrected_pass_started.json",
        _pretty_bytes(corrected_marker),
    )

    model = LocalJsonModel()
    started = time.perf_counter()
    proposed_o1: list[dict[str, object]] = []
    proposed_n1: list[dict[str, object]] = []
    rejected_model: list[dict[str, object]] = []
    prompt_receipts: list[dict[str, object]] = []
    prompt_index = 0
    for parent_id in sorted(parent_by_id):
        parent = parent_by_id[parent_id]
        topic_id = str(parent["topic_id"])
        broad_row = broad_by_topic[topic_id]
        for fold in (0, 1):
            prompt_index += 1
            passages = groups[(parent_id, fold)]
            messages = build_discovery_messages(parent, passages)
            response = model.generate(
                messages,
                DISCOVERY_SCHEMA,
                max_new_tokens=DISCOVERY_MAX_NEW_TOKENS,
            )
            if response["status"] == "unsupported" and (
                response["o1"] or response["n1"]
            ):
                for kind in ("o1", "n1"):
                    for raw in response[kind]:
                        rejected_model.append(
                            {**raw, "kind": kind, "reason": "unsupported_status"}
                        )
                accepted = {"o1": [], "n1": []}
                rejected: list[dict[str, object]] = []
            else:
                accepted, rejected = validate_model_response(response, passages)
                rejected_model.extend(rejected)
            passage_by_document = {
                str(row["document_id"]): row for row in passages
            }
            for index, raw in enumerate(accepted["o1"]):
                source = passage_by_document[str(raw["support_document_id"])]
                obligation_id = f"{parent_id}:f{fold}:o1:{index}"
                proposed_o1.append(
                    {
                        "schema_version": "adaptive-evidence-o1-proposal-v1",
                        "proposal_id": obligation_id,
                        "obligation_id": obligation_id,
                        "topic_id": topic_id,
                        "kind": "o1",
                        "parent_id": parent_id,
                        "source_fold": fold,
                        "source_document_id": str(source["document_id"]),
                        "source_document_sha256": str(source["document_sha256"]),
                        "source_window_id": str(source["window_id"]),
                        "source_parent_rank": int(source["parent_local_rank"]),
                        "source_reservoir_rank": int(source["reservoir_rank"]),
                        "label": str(raw["label"]),
                        "scope_rationale": str(raw["scope_rationale"]),
                        "subject": str(raw["subject"]),
                        "population": str(raw["population"]),
                        "domain": str(raw["domain"]),
                        "relation": str(raw["relation"]),
                        "support_span": str(raw["support_span"]),
                        "query": build_derived_query(
                            broad_row,
                            parent,
                            str(raw["label"]),
                        ),
                    }
                )
            for index, raw in enumerate(accepted["n1"]):
                source = passage_by_document[str(raw["support_document_id"])]
                proposed_n1.append(
                    {
                        "schema_version": "adaptive-evidence-n1-proposal-v1",
                        "nugget_id": f"{parent_id}:f{fold}:n1:{index}",
                        "topic_id": topic_id,
                        "kind": "n1",
                        "parent_id": parent_id,
                        "fold": fold,
                        "document_id": str(source["document_id"]),
                        "document_sha256": str(source["document_sha256"]),
                        "window_id": str(source["window_id"]),
                        "subject": str(raw["subject"]),
                        "relation": str(raw["relation"]),
                        "object": str(raw["object"]),
                        "support_span": str(raw["support_span"]),
                    }
                )
            prompt_receipts.append(
                {
                    "schema_version": "adaptive-evidence-prompt-receipt-v1",
                    "prompt_index": prompt_index,
                    "topic_id": topic_id,
                    "parent_id": parent_id,
                    "source_fold": fold,
                    "reservoir_document_ids": [
                        str(row["document_id"]) for row in passages
                    ],
                    "messages_sha256": _sha256_bytes(_compact_bytes(messages)),
                    "response_sha256": _sha256_bytes(_compact_bytes(response)),
                    "status": str(response["status"]),
                    "proposed_o1_count": len(accepted["o1"]),
                    "proposed_n1_count": len(accepted["n1"]),
                    "rejected_span_count": len(rejected),
                }
            )
            print(
                "progress=local_qwen_discovery "
                f"prompt={prompt_index}/48 parent={parent_id} fold={fold} "
                f"o1={len(accepted['o1'])} n1={len(accepted['n1'])}",
                file=sys.stderr,
                flush=True,
            )
    if prompt_index != 48:
        raise ValueError("discovery did not cover exactly 24 parents and two folds")
    runtime = model.execution_receipt()
    elapsed = time.perf_counter() - started
    if runtime.get("generation_count") != 48:
        raise ValueError("one-pass discovery generation count differs")
    payloads = {
        "proposed_o1.jsonl": _jsonl_bytes(proposed_o1),
        "proposed_n1.jsonl": _jsonl_bytes(proposed_n1),
        "rejected_model_records.jsonl": _jsonl_bytes(rejected_model),
        "prompt_receipts.jsonl": _jsonl_bytes(prompt_receipts),
    }
    artifacts = {
        name: {
            "rows": len(payload.splitlines()),
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        for name, payload in payloads.items()
    }
    receipt: dict[str, object] = {
        "schema_version": DISCOVERY_PROPOSAL_SCHEMA_VERSION,
        "status": "complete",
        "proposal_pass_count": 1,
        "corrected_completed_pass_count": 1,
        "prior_integration_preflight_failure_count": 1,
        "qwen_proposal_generation_call_count": 49,
        "completed_pass_generation_count": 48,
        "no_further_retry_authorized": True,
        "recursive_discovery": False,
        "model_selected_stop": False,
        "folds": [0, 1],
        "parent_count": 24,
        "prompt_count": 48,
        "proposed_o1_count": len(proposed_o1),
        "proposed_n1_count": len(proposed_n1),
        "rejected_model_record_count": len(rejected_model),
        "invalid_support_span_count": sum(
            row.get("reason") == "support_span" for row in rejected_model
        ),
        "all_proposed_spans_exact": not rejected_model,
        "elapsed_seconds": elapsed,
        "model_execution": runtime,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_sha256": _proposal_prompt_contract_sha256(),
        "preflight_prompt_sha256": preflight["prompt_sha256"],
        "schema_sha256": preflight["schema_sha256"],
        "preflight_sha256": _sha256_file(root / "preflight.json"),
        "artifacts": artifacts,
        "integration_preflight_failure_sha256": _sha256_file(
            root / "integration_preflight_failure.json"
        ),
        "corrected_pass_started_sha256": _sha256_file(
            root / "corrected_pass_started.json"
        ),
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "external_cost_usd": 0.0,
    }
    for name, payload in payloads.items():
        _exclusive_bytes(root / name, payload)
    _exclusive_bytes(root / "proposal_receipt.json", _pretty_bytes(receipt))
    return receipt


_FAILURE_HISTORY_NAMES = (
    "preflight.json",
    "integration_preflight_failure.json",
    "corrected_pass_started.json",
    "corrected_pass_failure.json",
)
_TERMINAL_ALLOWED_NAMES = {
    *_FAILURE_HISTORY_NAMES,
    "receipt.json",
    "verification.json",
    "broad.jsonl",
    "parents.jsonl",
    "reservoirs.jsonl",
    "repeated_phrases.jsonl",
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _authenticate_failure_history(root: Path) -> dict[str, object]:
    """Authenticate the four immutable records and derive terminal call counts."""

    preflight_path = root / "preflight.json"
    integration_path = root / "integration_preflight_failure.json"
    marker_path = root / "corrected_pass_started.json"
    failure_path = root / "corrected_pass_failure.json"
    preflight = _read_json(preflight_path, "discovery preflight")
    integration = _read_json(integration_path, "integration preflight failure")
    marker = _read_json(marker_path, "corrected pass start marker")
    failure = _read_json(failure_path, "corrected pass failure")

    preflight_schema = preflight.get("schema")
    preflight_model = preflight.get("model_preflight")
    if (
        preflight.get("schema_version") != DISCOVERY_PREFLIGHT_SCHEMA_VERSION
        or preflight.get("status") != "complete"
        or preflight.get("model") != MODEL_ID
        or preflight.get("model_revision") != MODEL_REVISION
        or not isinstance(preflight_model, Mapping)
        or preflight_model.get("generation_count") != 1
        or not isinstance(preflight_schema, Mapping)
        or preflight.get("schema_sha256")
        != _sha256_bytes(_compact_bytes(preflight_schema))
        or not _is_sha256(preflight.get("prompt_sha256"))
        or preflight.get("qrels_opened") is not False
        or preflight.get("network_call_count") != 0
        or preflight.get("retrieval_call_count") != 0
        or preflight.get("hosted_inference_call_count") != 0
        or preflight.get("paid_call_count") != 0
        or preflight.get("external_cost_usd") != 0.0
    ):
        raise ValueError("discovery failure history has an invalid successful preflight")

    integration_messages = integration.get("messages")
    if (
        integration.get("schema_version")
        != "adaptive-evidence-integration-preflight-failure-v1"
        or integration.get("status") != "failed"
        or integration.get("phase")
        != "model_facing_schema_integration_preflight"
        or integration.get("completed_proposal_pass_count") != 0
        or integration.get("qwen_reservoir_generation_call_count") != 1
        or integration.get("proposal_artifact_count") != 0
        or integration.get("model") != MODEL_ID
        or integration.get("model_revision") != MODEL_REVISION
        or integration.get("error_type") != "ValueError"
        or integration.get("error_message") != "JSON schema $ is missing status"
        or integration.get("model_output_retained") is not False
        or integration.get("preflight_sha256") != _sha256_file(preflight_path)
        or integration.get("schema_sha256") != preflight.get("schema_sha256")
        or not isinstance(integration.get("schema"), Mapping)
        or integration.get("schema_sha256")
        != _sha256_bytes(_compact_bytes(integration["schema"]))
        or not isinstance(integration_messages, list)
        or integration.get("messages_sha256")
        != _sha256_bytes(_compact_bytes(integration_messages))
    ):
        raise ValueError("discovery failure history has an invalid integration failure")

    prompt_contract_sha256 = marker.get("prompt_contract_sha256")
    schema_sha256 = marker.get("schema_sha256")
    if (
        marker.get("schema_version")
        != "adaptive-evidence-corrected-pass-start-v1"
        or marker.get("status") != "started"
        or marker.get("corrected_completed_pass_ordinal") != 1
        or marker.get("expected_generation_count") != 48
        or marker.get("no_further_retry_authorized") is not True
        or marker.get("model") != MODEL_ID
        or marker.get("model_revision") != MODEL_REVISION
        or not _is_sha256(prompt_contract_sha256)
        or schema_sha256 != integration.get("schema_sha256")
        or marker.get("integration_preflight_failure_sha256")
        != _sha256_file(integration_path)
    ):
        raise ValueError("discovery failure history has an invalid corrected-pass marker")

    if (
        failure.get("schema_version")
        != "adaptive-evidence-corrected-pass-failure-v1"
        or failure.get("status") != "failed"
        or failure.get("phase") != "corrected_complete_proposal_pass"
        or failure.get("failed_generation_ordinal") != 1
        or failure.get("completed_generation_count") != 0
        or failure.get("expected_generation_count") != 48
        or failure.get("completed_proposal_pass_count") != 0
        or failure.get("model") != MODEL_ID
        or failure.get("model_revision") != MODEL_REVISION
        or failure.get("max_new_tokens") != DISCOVERY_MAX_NEW_TOKENS
        or failure.get("prompt_contract_sha256") != prompt_contract_sha256
        or failure.get("schema_sha256") != schema_sha256
        or not isinstance(failure.get("schema"), Mapping)
        or failure.get("schema_sha256")
        != _sha256_bytes(_compact_bytes(failure["schema"]))
        or failure.get("error_type") != "ValueError"
        or failure.get("error_message")
        != "local model completion is not one exact JSON value"
        or failure.get("cause_type") != "JSONDecodeError"
        or failure.get("failure_classification")
        != "schema_json_truncated_at_generation_ceiling"
        or failure.get("model_output_retained") is not False
        or failure.get("proposal_artifact_count") != 0
        or failure.get("no_further_retry_authorized") is not True
        or failure.get("corrected_pass_started_sha256") != _sha256_file(marker_path)
        or failure.get("integration_preflight_failure_sha256")
        != _sha256_file(integration_path)
        or not _is_sha256(failure.get("messages_sha256"))
    ):
        raise ValueError("discovery failure history is not the exact terminal history")

    integration_calls = int(integration["qwen_reservoir_generation_call_count"])
    corrected_calls = int(failure["failed_generation_ordinal"])
    total_calls = integration_calls + corrected_calls
    if total_calls != 2:
        raise ValueError("discovery failure history does not contain exactly two calls")
    return {
        "preflight": preflight,
        "prompt_contract_sha256": prompt_contract_sha256,
        "schema_sha256": schema_sha256,
        "preflight_schema_probe_generation_call_count": int(
            preflight_model["generation_count"]
        ),
        "prior_integration_preflight_failure_count": 1,
        "corrected_pass_failure_count": 1,
        "completed_proposal_pass_count": int(
            failure["completed_proposal_pass_count"]
        ),
        "total_qwen_proposal_generation_call_count": total_calls,
        "hashes": {
            name: _sha256_file(root / name) for name in _FAILURE_HISTORY_NAMES
        },
    }


def _reject_terminal_extras(root: Path) -> None:
    unexpected = sorted(
        path.name for path in root.iterdir() if path.name not in _TERMINAL_ALLOWED_NAMES
    )
    downstream = [
        root.parent / "scoring" / name
        for name in (
            "provisional-o1",
            "provisional_o1",
            "o1",
            "accepted-o1",
            "accepted_o1",
        )
    ]
    if unexpected or any(path.exists() for path in downstream):
        raise ValueError(
            "terminal discovery unexpectedly contains partial or downstream artifacts"
        )


def record_discovery_unavailable(output_dir: Path) -> dict[str, object]:
    """Freeze a receipt derived only from the complete immutable failure history."""

    root = Path(output_dir)
    if (root / "receipt.json").exists():
        raise FileExistsError("create-only terminal discovery receipt already exists")
    history = _authenticate_failure_history(root)
    _reject_terminal_extras(root)
    preflight = history["preflight"]
    assert isinstance(preflight, Mapping)
    hashes = history["hashes"]
    assert isinstance(hashes, Mapping)
    receipt: dict[str, object] = {
        "schema_version": DISCOVERY_RECEIPT_SCHEMA_VERSION,
        "status": "discovery_unavailable",
        "reason": "corrected_pass_schema_json_truncation",
        "completed_proposal_pass_count": history["completed_proposal_pass_count"],
        "prior_integration_preflight_failure_count": history[
            "prior_integration_preflight_failure_count"
        ],
        "corrected_pass_failure_count": history["corrected_pass_failure_count"],
        "total_qwen_proposal_generation_call_count": history[
            "total_qwen_proposal_generation_call_count"
        ],
        "preflight_schema_probe_generation_call_count": history[
            "preflight_schema_probe_generation_call_count"
        ],
        "no_further_retry_authorized": True,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_contract_sha256": history["prompt_contract_sha256"],
        "schema_sha256": history["schema_sha256"],
        "repeated_phrase_count": preflight.get("repeated_phrase_count", 0),
        "accepted_control_phrase_count": preflight.get(
            "accepted_control_phrase_count", 0
        ),
        "proposed_o1_count": 0,
        "validated_o1_count": 0,
        "accepted_o1_count": 0,
        "rejected_o1_count": 0,
        "proposed_n1_count": 0,
        "validated_n1_count": 0,
        "accepted_n1_count": 0,
        "rejected_n1_count": 0,
        "provisional_o1_score_status": "not_run",
        "provisional_o1_candidate_count": 0,
        "provisional_o1_window_count": 0,
        "deterministic_validation_status": "not_run",
        "validation_model_call_count": 0,
        "accepted_o1_score_status": "not_run",
        "accepted_o1_candidate_count": 0,
        "accepted_o1_window_count": 0,
        "artifacts": {
            name: {"sha256": hashes[name]} for name in _FAILURE_HISTORY_NAMES
        },
        "bindings": {
            "integration_preflight_failure_sha256": hashes[
                "integration_preflight_failure.json"
            ],
            "preflight_sha256": hashes["preflight.json"],
            "model_preflight": preflight.get("model_preflight"),
            "code_sha256": {
                "adaptive_evidence_discovery.py": _sha256_file(Path(__file__)),
                "adaptive_evidence_local_model.py": _sha256_file(
                    Path(__file__).with_name("adaptive_evidence_local_model.py")
                ),
                "adaptive_evidence_score.py": _sha256_file(
                    Path(__file__).with_name("adaptive_evidence_score.py")
                ),
            },
        },
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "external_cost_usd": 0.0,
    }
    _exclusive_bytes(root / "receipt.json", _pretty_bytes(receipt))
    return receipt


def verify_discovery_terminal(output_dir: Path) -> dict[str, object]:
    """Authenticate the terminal discovery-unavailable state and zero downstream work."""

    root = Path(output_dir)
    receipt = _read_json(root / "receipt.json", "terminal discovery receipt")
    history = _authenticate_failure_history(root)
    if (
        receipt.get("schema_version") != DISCOVERY_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "discovery_unavailable"
        or receipt.get("reason") != "corrected_pass_schema_json_truncation"
        or receipt.get("completed_proposal_pass_count")
        != history["completed_proposal_pass_count"]
        or receipt.get("prior_integration_preflight_failure_count")
        != history["prior_integration_preflight_failure_count"]
        or receipt.get("corrected_pass_failure_count")
        != history["corrected_pass_failure_count"]
        or receipt.get("total_qwen_proposal_generation_call_count")
        != history["total_qwen_proposal_generation_call_count"]
        or receipt.get("preflight_schema_probe_generation_call_count")
        != history["preflight_schema_probe_generation_call_count"]
        or receipt.get("no_further_retry_authorized") is not True
        or receipt.get("model") != MODEL_ID
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("prompt_contract_sha256")
        != history["prompt_contract_sha256"]
        or receipt.get("schema_sha256") != history["schema_sha256"]
        or receipt.get("qrels_opened") is not False
        or receipt.get("network_call_count") != 0
        or receipt.get("retrieval_call_count") != 0
        or receipt.get("hosted_inference_call_count") != 0
        or receipt.get("paid_call_count") != 0
        or receipt.get("external_cost_usd") != 0.0
    ):
        raise ValueError("terminal discovery receipt differs from the failure contract")
    zero_fields = (
        "proposed_o1_count",
        "validated_o1_count",
        "accepted_o1_count",
        "rejected_o1_count",
        "proposed_n1_count",
        "validated_n1_count",
        "accepted_n1_count",
        "rejected_n1_count",
        "provisional_o1_candidate_count",
        "provisional_o1_window_count",
        "validation_model_call_count",
        "accepted_o1_candidate_count",
        "accepted_o1_window_count",
    )
    if any(receipt.get(name) != 0 for name in zero_fields):
        raise ValueError("terminal discovery receipt has nonzero downstream work")
    if any(
        receipt.get(name) != "not_run"
        for name in (
            "provisional_o1_score_status",
            "deterministic_validation_status",
            "accepted_o1_score_status",
        )
    ):
        raise ValueError("terminal discovery downstream status differs")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("terminal discovery artifacts are missing")
    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("terminal discovery bindings are missing")
    hashes = history["hashes"]
    assert isinstance(hashes, Mapping)
    for name in _FAILURE_HISTORY_NAMES:
        binding = artifacts.get(name)
        if name == "preflight.json" and not isinstance(binding, Mapping):
            binding = {"sha256": bindings.get("preflight_sha256")}
        elif (
            name == "integration_preflight_failure.json"
            and not isinstance(binding, Mapping)
        ):
            binding = {
                "sha256": bindings.get("integration_preflight_failure_sha256")
            }
        if (
            not isinstance(binding, Mapping)
            or binding.get("sha256") != hashes[name]
        ):
            raise ValueError(f"terminal discovery artifact hash differs: {name}")
    _reject_terminal_extras(root)
    return receipt


def _load_proposal_artifacts(
    root: Path,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    receipt = _read_json(root / "proposal_receipt.json", "proposal receipt")
    if (
        receipt.get("schema_version") != DISCOVERY_PROPOSAL_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("proposal_pass_count") != 1
        or receipt.get("corrected_completed_pass_count") != 1
        or receipt.get("prior_integration_preflight_failure_count") != 1
        or receipt.get("qwen_proposal_generation_call_count") != 49
        or receipt.get("completed_pass_generation_count") != 48
        or receipt.get("no_further_retry_authorized") is not True
        or receipt.get("recursive_discovery") is not False
        or receipt.get("folds") != [0, 1]
        or receipt.get("prompt_count") != 48
        or receipt.get("model") != MODEL_ID
        or receipt.get("model_revision") != MODEL_REVISION
        or receipt.get("prompt_sha256") != _proposal_prompt_contract_sha256()
        or receipt.get("qrels_opened") is not False
        or receipt.get("network_call_count") != 0
        or receipt.get("retrieval_call_count") != 0
        or receipt.get("hosted_inference_call_count") != 0
        or receipt.get("paid_call_count") != 0
    ):
        raise ValueError("proposal receipt differs from the frozen one-pass contract")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("proposal artifact bindings are missing")
    loaded: list[list[dict[str, object]]] = []
    for name in (
        "proposed_o1.jsonl",
        "proposed_n1.jsonl",
        "rejected_model_records.jsonl",
    ):
        path = root / name
        rows = _read_jsonl(path, name)
        binding = artifacts.get(name)
        if (
            not isinstance(binding, Mapping)
            or binding.get("rows") != len(rows)
            or binding.get("bytes") != path.stat().st_size
            or binding.get("sha256") != _sha256_file(path)
        ):
            raise ValueError(f"proposal artifact {name} differs")
        loaded.append(rows)
    proposals, nuggets, rejected = loaded
    _reject_protected(
        [row.get("topic_id") for row in proposals]
        + [row.get("topic_id") for row in nuggets]
    )
    return receipt, proposals, nuggets, rejected


def _load_provisional_score_rows(
    stage_dir: Path,
    proposal_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    root = Path(stage_dir)
    receipt = _read_json(root / "receipt.json", "provisional O1 score receipt")
    source = receipt.get("source")
    if (
        receipt.get("schema_version") != "adaptive-evidence-o1-score-stage-v1"
        or receipt.get("status") != "complete"
        or receipt.get("stage") != "provisional_o1"
        or receipt.get("opposite_fold_only") is not True
        or receipt.get("population_violation_count") != 0
        or receipt.get("qrels_opened") is not False
        or receipt.get("network_call_count") != 0
        or receipt.get("retrieval_call_count") != 0
        or receipt.get("hosted_inference_call_count") != 0
        or receipt.get("paid_call_count") != 0
        or not isinstance(source, Mapping)
        or source.get("sha256") != _sha256_file(proposal_path)
        or source.get("bytes") != proposal_path.stat().st_size
    ):
        raise ValueError("provisional O1 score stage differs from proposals or fold contract")
    if receipt.get("record_count") == 0:
        return receipt, []
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("provisional O1 score artifacts are missing")
    score_binding = artifacts.get("scores")
    if (
        not isinstance(score_binding, Mapping)
        or score_binding.get("sha256") != _sha256_file(root / "scores/receipt.json")
    ):
        raise ValueError("provisional O1 score receipt hash differs")
    score_receipt = verify_local_scoring(root / "scores")
    rows: list[dict[str, object]] = []
    for shard in score_receipt["shards"]:  # type: ignore[index]
        if not isinstance(shard, Mapping):
            raise ValueError("provisional score shard receipt is invalid")
        rows.extend(
            _read_jsonl(
                root / "scores" / str(shard["path"]),
                "provisional O1 score shard",
            )
        )
    if len(rows) != receipt.get("window_count"):
        raise ValueError("provisional O1 window count differs")
    return receipt, rows


def run_discovery_validation(
    *,
    output_dir: Path,
    opposite_fold_scores: Path,
) -> dict[str, object]:
    """Deterministically corroborate proposals and freeze O1/N1 records."""

    root = Path(output_dir)
    final_names = (
        "validated_o1.jsonl",
        "accepted_o1.jsonl",
        "rejected_o1.jsonl",
        "validated_n1.jsonl",
        "accepted_n1.jsonl",
        "rejected_n1.jsonl",
        "receipt.json",
    )
    if any((root / name).exists() for name in final_names):
        raise FileExistsError("create-only discovery validation output already exists")
    _preflight, parents, broad, _reservoirs = _load_discovery_preflight(root)
    proposal_receipt, proposals, nuggets, model_rejected = _load_proposal_artifacts(
        root
    )
    score_receipt, score_rows = _load_provisional_score_rows(
        Path(opposite_fold_scores),
        root / "proposed_o1.jsonl",
    )
    broad_by_topic = {str(row["topic_id"]): row for row in broad}
    finalized = finalize_discovery_records(
        proposals=proposals,
        nuggets=nuggets,
        parents=parents,
        broad_by_topic=broad_by_topic,
        score_rows=score_rows,
    )
    rejected_n1 = [
        {**row, "status": "unsupported"}
        for row in model_rejected
        if row.get("kind") == "n1"
    ]
    validated_n1 = [dict(row) for row in nuggets]
    finalized["rejected_n1"].extend(rejected_n1)
    payload_rows = {
        "validated_o1.jsonl": finalized["validated_o1"],
        "accepted_o1.jsonl": finalized["accepted_o1"],
        "rejected_o1.jsonl": finalized["rejected_o1"],
        "validated_n1.jsonl": validated_n1,
        "accepted_n1.jsonl": finalized["accepted_n1"],
        "rejected_n1.jsonl": finalized["rejected_n1"],
    }
    payloads = {name: _jsonl_bytes(rows) for name, rows in payload_rows.items()}
    artifacts = {
        name: {
            "rows": len(payload_rows[name]),
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        for name, payload in payloads.items()
    }
    reason_counts: dict[str, int] = defaultdict(int)
    for row in finalized["rejected_o1"]:
        for reason in row.get("reasons", []):
            reason_counts[str(reason)] += 1
    receipt: dict[str, object] = {
        "schema_version": DISCOVERY_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "proposal_pass_count": 1,
        "recursive_discovery": False,
        "validation_method": "deterministic_opposite_fold_minilm",
        "validation_model_call_count": 0,
        "folds": [0, 1],
        "proposed_o1_count": len(proposals),
        "validated_o1_count": len(finalized["validated_o1"]),
        "accepted_o1_count": len(finalized["accepted_o1"]),
        "rejected_o1_count": len(finalized["rejected_o1"]),
        "proposed_n1_count": len(nuggets),
        "validated_n1_count": len(validated_n1),
        "accepted_n1_count": len(finalized["accepted_n1"]),
        "rejected_n1_count": len(finalized["rejected_n1"]),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "opposite_fold_score_record_count": score_receipt["record_count"],
        "opposite_fold_candidate_count": score_receipt["candidate_count"],
        "opposite_fold_window_count": score_receipt["window_count"],
        "opposite_fold_population_violation_count": score_receipt[
            "population_violation_count"
        ],
        "all_accepted_spans_exact": all(
            str(support["support_span"]) in str(support["passage_text"])
            for row in finalized["accepted_o1"]
            for support in row["opposite_fold_support"]
        ),
        "parent_limit_violation_count": 0,
        "topic_limit_violation_count": 0,
        "artifacts": artifacts,
        "bindings": {
            "proposal_receipt_sha256": _sha256_file(root / "proposal_receipt.json"),
            "proposal_model_execution": proposal_receipt["model_execution"],
            "provisional_score_receipt_sha256": _sha256_file(
                Path(opposite_fold_scores) / "receipt.json"
            ),
        },
        "qrels_opened": False,
        "network_call_count": 0,
        "retrieval_call_count": 0,
        "hosted_inference_call_count": 0,
        "paid_call_count": 0,
        "external_cost_usd": 0.0,
    }
    parent_ids = [str(row["parent_id"]) for row in finalized["accepted_o1"]]
    topic_counts: dict[str, int] = defaultdict(int)
    for row in finalized["accepted_o1"]:
        topic_counts[str(row["topic_id"])] += 1
    if len(parent_ids) != len(set(parent_ids)) or any(
        count > 4 for count in topic_counts.values()
    ):
        raise ValueError("frozen O1 complexity safeguard was violated")
    for name, payload in payloads.items():
        _exclusive_bytes(root / name, payload)
    _exclusive_bytes(root / "receipt.json", _pretty_bytes(receipt))
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="authenticate sources and freeze fold reservoirs"
    )
    preflight.add_argument("--contract", type=Path, required=True)
    preflight.add_argument("--scores", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    propose = subparsers.add_parser(
        "propose", help="run the single local Qwen discovery pass"
    )
    propose.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate", help="deterministically validate opposite-fold support"
    )
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--opposite-fold-scores", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        receipt = run_discovery_preflight(
            contract_dir=args.contract,
            scores_dir=args.scores,
            output_dir=args.output,
        )
        print(
            "status=complete "
            f"reservoirs={receipt['reservoir_group_count']} "
            f"documents={receipt['reservoir_count']} "
            f"control_phrases={receipt['repeated_phrase_count']} "
            "model_preflight=passed network=false qrels_opened=false external_cost=$0"
        )
    elif args.command == "propose":
        receipt = run_proposal_pass(args.output)
        print(
            "status=complete proposal_passes=1 "
            f"prompts={receipt['prompt_count']} "
            f"o1={receipt['proposed_o1_count']} "
            f"n1={receipt['proposed_n1_count']} "
            f"invalid_spans={receipt['invalid_support_span_count']} "
            "network=false qrels_opened=false external_cost=$0"
        )
    else:
        receipt = run_discovery_validation(
            output_dir=args.output,
            opposite_fold_scores=args.opposite_fold_scores,
        )
        print(
            "status=complete "
            f"accepted_o1={receipt['accepted_o1_count']} "
            f"rejected_o1={receipt['rejected_o1_count']} "
            f"accepted_n1={receipt['accepted_n1_count']} "
            f"validation_model_calls={receipt['validation_model_call_count']} "
            "network=false qrels_opened=false external_cost=$0"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
