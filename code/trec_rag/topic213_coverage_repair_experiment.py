"""Run a coverage-oriented, audit-repair Topic 213 GPT generation experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import os
import re
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import yaml

from trec_rag.repo_env import find_repo_root, load_repo_env
from trec_rag.topic213_controlled_generator_benchmark import (
    GenerationCompletion,
    GeneratorClient,
    SingleCandidateJsonClient,
    _artifact_row,
    _make_judge_client,
    _usage_cost,
    build_candidate_generation,
    build_generation_payload,
    build_official_entry,
    render_markdown,
    semantic_request_sha256,
    validate_generation_response,
    validate_manifest_hashes,
    validate_official_entry,
)
from trec_rag.topic213_ragnarok_experiment import SpacySentenceTokenizer
from trec_rag.topic213_response_experiment import (
    OpenAICompatibleJsonClient,
    PassageCorpus,
    _audit_section,
    _require_nonempty_string,
    _sha256,
    _write_json,
    _write_jsonl,
    compute_evaluation_metrics,
    evaluate_nuggets,
    load_nuggets,
    load_passage_corpus,
)


SCHEMA_VERSION = "topic213-coverage-repair-v2"
PROMPT_VERSION = "coverage-repair-atomic-generation-v2"
FREEZE_VERSION = "coverage-repair-generation-freeze-v2"
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_INLINE_CITATION_RE = re.compile(r"\[[^\]]+\]")
_STOP_WORDS = frozenset(
    """
    a an and are as at be been being by did do does for from had has have he her his in into is it
    its of on or that the their them they this to under was were which who with would korean korea
    war united states state us u s
    """.split()
)


GENERATION_SYSTEM_PROMPT = """You produce a citation-grounded TREC RAG answer from a frozen,
nugget-blind evidence ledger. Return one JSON object with a `sentences` array and exactly one item
for every supplied claim slot, in the same order. Every item has exactly `claim_id` and `text`.

Write one polished, self-contained sentence per slot. Preserve the source claim's explicit subject,
action, causal relationship, names, dates, and qualifications. Do not add stronger causality,
motives, evaluations, or historical details that the source claim does not state. Do not merge or
drop slots, use headings, add introductions or conclusions, or emit citation markers. Prefer direct
wording over rhetorical compression. Follow each facet word target and keep the whole answer inside
the supplied word band; most sentences should be roughly 17-24 words. Citations are attached after
generation."""


LEDGER_REPAIR_SYSTEM_PROMPT = """Rewrite one proposed evidence claim into one concise atomic
sentence using only the supplied verbatim passages. Preserve the most distinctive supported fact,
but remove every unsupported adjective, causal link, number, motive, or conclusion. Return one JSON
object with exactly one key, `text`. Do not include citations or outside knowledge."""


FACET_PLANNER_SYSTEM_PROMPT = """Rank evidence candidates for one research sub-narrative.
Return one JSON object with `ranked_candidate_ids`, containing every supplied candidate ID exactly
once from most useful to least useful. Put direct answers first. Maximize distinct coverage of
actors, policies, dates, decisions, causes, consequences, and disagreements while suppressing
paraphrases. Prefer concrete claims with clear evidence over broad commentary. Use only supplied
candidate IDs; organizer nuggets are unavailable."""


GENERATION_REPAIR_SYSTEM_PROMPT = """Repair rejected answer sentences using only their frozen
source claims and cited verbatim passages. Return one JSON object with a `sentences` array in the
supplied order. Every item must have exactly `claim_id` and `text`. Keep each claim ID unchanged and
write exactly one self-contained sentence that states only details directly supported by its cited
passages. Do not add citation markers, outside knowledge, introductions, or conclusions."""


ProgressCallback = Callable[[dict[str, object]], None]


def _content_terms(text: str) -> set[str]:
    return {
        term
        for term in _WORD_RE.findall(text.casefold())
        if len(term) > 2 and term not in _STOP_WORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def select_covering_passages(
    claim_text: str,
    passages: Sequence[Mapping[str, object]],
    *,
    maximum_citations: int = 3,
) -> list[dict[str, object]]:
    """Select up to three distinct documents that jointly cover claim terms."""

    if maximum_citations < 1 or maximum_citations > 3:
        raise ValueError("maximum_citations must be between 1 and 3")
    rows = [dict(row) for row in passages]
    if not rows:
        raise ValueError("a claim must have at least one supporting passage")
    claim_terms = _content_terms(claim_text)
    uncovered = set(claim_terms)
    selected: list[dict[str, object]] = []
    used_documents: set[str] = set()
    remaining = list(enumerate(rows))
    while remaining and len(selected) < maximum_citations:
        scored: list[tuple[int, int, int, dict[str, object]]] = []
        for index, row in remaining:
            document_id = _require_nonempty_string(row.get("document_id"), "document_id")
            if document_id in used_documents:
                continue
            passage_terms = _content_terms(str(row.get("text", "")))
            gain = len(uncovered & passage_terms)
            total_overlap = len(claim_terms & passage_terms)
            scored.append((gain, total_overlap, -index, row))
        if not scored:
            break
        gain, _, _, best = max(scored, key=lambda item: item[:3])
        if selected and gain == 0:
            break
        selected.append(best)
        used_documents.add(str(best["document_id"]))
        uncovered -= _content_terms(str(best.get("text", "")))
        remaining = [(index, row) for index, row in remaining if row is not best]
        if not uncovered:
            break
    if not selected:
        selected.append(rows[0])
    return selected


def _candidate_quality(
    candidate: Mapping[str, object],
    *,
    idf: Mapping[str, float],
    maximum_idf: float,
) -> float:
    terms = _content_terms(str(candidate.get("text", "")))
    passage_terms = _content_terms(
        " ".join(
            str(row.get("text", ""))
            for row in candidate.get("supporting_passages", [])
            if isinstance(row, Mapping)
        )
    )
    denominator = sum(idf.get(term, 1.0) for term in terms) or 1.0
    support = sum(idf.get(term, 1.0) for term in terms & passage_terms) / denominator
    specificity = (
        sum(idf.get(term, 1.0) for term in terms) / max(len(terms), 1) / maximum_idf
    )
    concrete_bonus = 0.03 if re.search(r"\b(?:18|19|20)\d{2}\b|\d+%|\b[A-Z]{2,}-?\d*\b", str(candidate.get("text", ""))) else 0.0
    return min(1.0, 0.70 * support + 0.30 * specificity + concrete_bonus)


def select_diverse_candidates(
    candidates: Sequence[Mapping[str, object]],
    *,
    quota: int,
    reserve_count: int = 5,
    duplicate_threshold: float = 0.68,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Rank supported candidates with hard paraphrase suppression and MMR novelty."""

    if quota < 1 or reserve_count < 0:
        raise ValueError("quota must be positive and reserve_count nonnegative")
    unique: dict[str, dict[str, object]] = {}
    for raw in candidates:
        row = dict(raw)
        text = _require_nonempty_string(row.get("text"), "candidate text")
        key = " ".join(text.casefold().split())
        current = unique.get(key)
        if current is None or len(row.get("supporting_passages", [])) > len(
            current.get("supporting_passages", [])
        ):
            unique[key] = row
    rows = list(unique.values())
    if len(rows) < quota:
        raise ValueError(f"only {len(rows)} unique candidates are available for quota {quota}")

    term_sets = [_content_terms(str(row["text"])) for row in rows]
    document_frequency = Counter(term for terms in term_sets for term in terms)
    idf = {
        term: math.log((len(rows) + 1) / (frequency + 1)) + 1
        for term, frequency in document_frequency.items()
    }
    maximum_idf = max(idf.values(), default=1.0)
    qualities = [
        _candidate_quality(row, idf=idf, maximum_idf=maximum_idf) for row in rows
    ]
    ranked: list[int] = []
    target_count = min(len(rows), quota + reserve_count)
    while len(ranked) < target_count:
        unranked = [index for index in range(len(rows)) if index not in ranked]
        nonduplicates = [
            index
            for index in unranked
            if all(
                _jaccard(term_sets[index], term_sets[chosen]) < duplicate_threshold
                for chosen in ranked
            )
        ]
        pool = nonduplicates or unranked

        def score(index: int) -> tuple[float, float, str]:
            redundancy = max(
                (_jaccard(term_sets[index], term_sets[chosen]) for chosen in ranked),
                default=0.0,
            )
            candidate_documents = {
                str(row.get("document_id"))
                for row in rows[index].get("supporting_passages", [])
                if isinstance(row, Mapping)
            }
            document_overlap = max(
                (
                    bool(
                        candidate_documents
                        & {
                            str(row.get("document_id"))
                            for row in rows[chosen].get("supporting_passages", [])
                            if isinstance(row, Mapping)
                        }
                    )
                    for chosen in ranked
                ),
                default=False,
            )
            value = 0.58 * qualities[index] + 0.42 * (1.0 - redundancy)
            if document_overlap:
                value -= 0.05
            return value, qualities[index], str(rows[index].get("candidate_id", ""))

        ranked.append(max(pool, key=score))
    selected = [copy.deepcopy(rows[index]) for index in ranked[:quota]]
    reserves = [copy.deepcopy(rows[index]) for index in ranked[quota:]]
    return selected, reserves


def build_frozen_evidence_from_claims(
    *,
    topic_id: str,
    narrative: str,
    source_experiment_id: str,
    labels: Sequence[str],
    claims_by_label: Mapping[str, Sequence[Mapping[str, object]]],
    sentence_quotas: Mapping[str, int],
    facet_word_targets: Mapping[str, int],
    minimum_words: int,
    maximum_words: int,
    source_metadata: Mapping[str, object],
    input_accounting: Mapping[str, object],
) -> dict[str, object]:
    if minimum_words > maximum_words or maximum_words > 1024:
        raise ValueError("invalid answer word band")
    facets: list[dict[str, object]] = []
    position = 0
    source_words = 0
    used_ids: set[str] = set()
    for facet_number, label in enumerate(labels, 1):
        quota = int(sentence_quotas[label])
        claims = [dict(row) for row in claims_by_label[label]]
        if len(claims) != quota:
            raise ValueError(f"{label} requires {quota} grounded claims, found {len(claims)}")
        frozen_claims: list[dict[str, object]] = []
        for raw in claims:
            position += 1
            text = _require_nonempty_string(raw.get("text"), "grounded claim text")
            claim_id = str(raw.get("claim_id") or f"A{position:03d}")
            if claim_id in used_ids:
                raise ValueError(f"duplicate frozen claim ID: {claim_id}")
            used_ids.add(claim_id)
            passages = raw.get("selected_citation_passages") or raw.get(
                "supporting_passages"
            )
            if not isinstance(passages, list):
                raise ValueError(f"{claim_id} has no supporting passages")
            selected = select_covering_passages(text, passages, maximum_citations=3)
            source_words += len(text.split())
            frozen_claims.append(
                {
                    "position": position,
                    "claim_id": claim_id,
                    "source_candidate_id": raw.get("candidate_id", claim_id),
                    "text": text,
                    "sub_narrative": label,
                    "document_ids": [str(row["document_id"]) for row in selected],
                    "selected_citation_passages": selected,
                    "available_supporting_passage_count": len(passages),
                    "ledger_repaired": bool(raw.get("ledger_repaired", False)),
                }
            )
        facets.append(
            {
                "facet_number": facet_number,
                "sub_narrative": label,
                "sentence_quota": quota,
                "word_target": int(facet_word_targets[label]),
                "source_claims": frozen_claims,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "topic_id": str(topic_id),
        "narrative": narrative,
        "source_experiment_id": source_experiment_id,
        "organizer_nuggets_available": False,
        "selection_strategy": "full-map-candidate-mmr-with-pre-generation-support-repair",
        "supported_source_claim_count": position,
        "source_claim_word_count": source_words,
        "exact_total_sentence_count": position,
        "word_band": {"minimum": minimum_words, "maximum": maximum_words},
        "facet_word_target_total": sum(int(facet_word_targets[label]) for label in labels),
        "input_accounting": dict(input_accounting),
        "source_metadata": dict(source_metadata),
        "facets": facets,
    }


def apply_audited_repairs(
    candidate: Mapping[str, object],
    *,
    initial_audits: Sequence[Mapping[str, object]],
    repair_text_by_claim_id: Mapping[str, str],
    repair_audits: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Keep supported originals and supported repairs, excluding only failed repairs."""

    final = copy.deepcopy(candidate)
    candidate_ids = {
        str(claim["claim_id"])
        for section in final["sections"]
        for claim in section["claims"]
    }
    initial_by_id = {str(row["claim_id"]): dict(row) for row in initial_audits}
    if set(initial_by_id) != candidate_ids:
        raise ValueError("initial support audit must cover every candidate claim")
    repair_by_id = {str(row["claim_id"]): dict(row) for row in repair_audits}
    kept: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    retained_source_ids: set[str] = set()
    attempted = 0
    retained_repairs = 0
    for section in final["sections"]:
        kept_claims: list[dict[str, object]] = []
        for claim in section["claims"]:
            claim_id = str(claim["claim_id"])
            initial = initial_by_id[claim_id]
            if initial.get("status") == "supported":
                kept_claims.append(claim)
                kept.append(initial)
                retained_source_ids.update(map(str, claim["evidence_claim_ids"]))
                continue
            attempted += 1
            repair_text = repair_text_by_claim_id.get(claim_id)
            repair_audit = repair_by_id.get(claim_id)
            if repair_text and repair_audit and repair_audit.get("status") == "supported":
                claim["text"] = repair_text
                claim["repair"] = {
                    "initial_status": initial.get("status"),
                    "initial_notes": initial.get("notes", ""),
                    "applied": True,
                }
                kept_claims.append(claim)
                kept.append(repair_audit)
                retained_source_ids.update(map(str, claim["evidence_claim_ids"]))
                retained_repairs += 1
            else:
                excluded.append(
                    {
                        **dict(claim),
                        "initial_support_audit": initial,
                        "repair_support_audit": repair_audit,
                    }
                )
        section["claims"] = kept_claims
    final["evidence_ledger"] = [
        row
        for row in final["evidence_ledger"]
        if str(row["claim_id"]) in retained_source_ids
    ]
    final["support_filter"] = {
        "candidate_sentence_count": len(candidate_ids),
        "submitted_sentence_count": len(kept),
        "excluded_sentence_count": len(excluded),
        "repair_attempted_count": attempted,
        "repair_retained_count": retained_repairs,
    }
    return final, kept, excluded


def _normalize_label(value: object) -> str:
    return " ".join(str(value).strip().strip('"').casefold().split())


def _match_label(value: object, labels: Sequence[str]) -> str:
    normalized = _normalize_label(value)
    exact = {_normalize_label(label): label for label in labels}
    if normalized in exact:
        return exact[normalized]
    simplified = normalized.replace(" te ", " the ")
    typo_normalized = {
        key.replace(" te ", " the "): label for key, label in exact.items()
    }
    if simplified in typo_normalized:
        return typo_normalized[simplified]
    raise ValueError(f"map candidate has unknown sub-narrative: {value}")


def _candidate_id(label: str, text: str, passage_ids: Sequence[str]) -> str:
    canonical = json.dumps(
        [label, " ".join(text.casefold().split()), list(passage_ids)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "M" + hashlib.sha256(canonical).hexdigest()[:15].upper()


def load_map_candidate_pool(
    *, checkpoint_dir: Path,
    corpus: PassageCorpus,
    labels: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    passage_lookup = {row.passage_id: row for row in corpus.passages}
    passages_by_label: dict[str, list[dict[str, object]]] = {label: [] for label in labels}
    for passage in corpus.passages:
        label = _match_label(passage.sub_narrative, labels)
        passages_by_label[label].append(
            {
                "passage_id": passage.passage_id,
                "document_id": passage.document_id,
                "text": passage.text,
            }
        )
    checkpoint_paths = sorted(checkpoint_dir.glob("map_evidence__*.json"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"no map evidence checkpoints in {checkpoint_dir}")
    candidates: list[dict[str, object]] = []
    exact_seen: set[tuple[str, str]] = set()
    for checkpoint_path in checkpoint_paths:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        rows = payload.get("claims", [])
        if not isinstance(rows, list):
            raise ValueError(f"invalid map checkpoint: {checkpoint_path}")
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            label = _match_label(raw.get("sub_narrative"), labels)
            text = _require_nonempty_string(raw.get("text"), "map claim text")
            exact_key = (label, " ".join(text.casefold().split()))
            if exact_key in exact_seen:
                continue
            passage_ids = [
                str(item)
                for item in raw.get("passage_ids", [])
                if str(item) in passage_lookup
            ]
            if not passage_ids:
                continue
            passage_rows = [
                {
                    "passage_id": passage_lookup[item].passage_id,
                    "document_id": passage_lookup[item].document_id,
                    "text": passage_lookup[item].text,
                }
                for item in passage_ids
            ]
            claim_terms = _content_terms(text)
            lexical_support = sorted(
                passages_by_label[label],
                key=lambda row: (
                    len(claim_terms & _content_terms(str(row["text"]))),
                    len(claim_terms & _content_terms(str(row["text"])))
                    / max(len(claim_terms), 1),
                    -len(str(row["text"])),
                    str(row["passage_id"]),
                ),
                reverse=True,
            )[:12]
            combined_passages: list[dict[str, object]] = []
            seen_passage_ids: set[str] = set()
            for passage in [*passage_rows, *lexical_support]:
                passage_id = str(passage["passage_id"])
                if passage_id not in seen_passage_ids:
                    combined_passages.append(passage)
                    seen_passage_ids.add(passage_id)
            selected_passages = select_covering_passages(
                text, combined_passages, maximum_citations=3
            )
            candidates.append(
                {
                    "candidate_id": _candidate_id(label, text, passage_ids),
                    "text": text,
                    "sub_narrative": label,
                    "source_checkpoint": checkpoint_path.name,
                    "source_passage_ids": passage_ids,
                    "retrieved_supporting_passage_count": len(combined_passages),
                    "supporting_passages": selected_passages,
                }
            )
            exact_seen.add(exact_key)
    aggregate = hashlib.sha256(
        "".join(
            f"{path.name}:{_sha256(path)}\n" for path in checkpoint_paths
        ).encode("utf-8")
    ).hexdigest()
    summary = {
        "checkpoint_file_count": len(checkpoint_paths),
        "checkpoint_aggregate_sha256": aggregate,
        "candidate_count": len(candidates),
        "candidate_count_by_facet": {
            label: sum(1 for row in candidates if row["sub_narrative"] == label)
            for label in labels
        },
    }
    return candidates, summary


def _audit_claim(
    client: OpenAICompatibleJsonClient,
    *,
    claim: Mapping[str, object],
    audit_config: Mapping[str, object],
) -> dict[str, object]:
    rows = _audit_section(
        client,
        sub_narrative=str(claim["sub_narrative"]),
        claims=[claim],
        max_tokens=int(audit_config.get("max_tokens", 350)),
        temperature=float(audit_config.get("temperature", 0.0)),
        validation_attempts=int(audit_config.get("validation_attempts", 3)),
    )
    return rows[0]


def _repair_ledger_claim(
    client: OpenAICompatibleJsonClient,
    *,
    claim: Mapping[str, object],
    audit_notes: str,
    audit_config: Mapping[str, object],
    tokenizer: SpacySentenceTokenizer,
) -> str:
    payload: dict[str, object] = {
        "task": "Rewrite this proposed claim to the strongest atomic fact directly supported by the passages.",
        "proposed_claim": claim["text"],
        "audit_notes": audit_notes,
        "cited_passages": claim["supporting_passages"],
    }
    attempts = int(audit_config.get("validation_attempts", 3))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        value = client.complete_json(
            stage="repair_ledger_claim",
            system_prompt=LEDGER_REPAIR_SYSTEM_PROMPT,
            payload=payload,
            max_tokens=int(audit_config.get("repair_max_tokens", 180)),
            temperature=0.0,
        )
        try:
            if set(value) != {"text"}:
                raise ValueError("ledger repair must contain only text")
            text = _require_nonempty_string(value.get("text"), "ledger repair text")
            if _INLINE_CITATION_RE.search(text):
                raise ValueError("ledger repair contains a citation marker")
            if len(tokenizer.tokenize(text)) != 1:
                raise ValueError("ledger repair is not exactly one sentence")
            return text
        except ValueError as exc:
            last_error = exc
            payload = {**payload, "validation_feedback": str(exc), "attempt": attempt + 1}
    raise RuntimeError("ledger claim repair failed validation") from last_error


def _plan_facet_candidates(
    client: OpenAICompatibleJsonClient,
    *,
    label: str,
    candidates: Sequence[Mapping[str, object]],
    quota: int,
    audit_config: Mapping[str, object],
) -> list[dict[str, object]]:
    lookup = {str(row["candidate_id"]): dict(row) for row in candidates}
    payload: dict[str, object] = {
        "task": "Rank all candidates; the first entries fill the answer quota and later entries are support-audit reserves.",
        "sub_narrative": label,
        "answer_quota": quota,
        "candidates": [
            {"candidate_id": candidate_id, "text": row["text"]}
            for candidate_id, row in lookup.items()
        ],
    }
    attempts = int(audit_config.get("validation_attempts", 3))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        value = client.complete_json(
            stage="plan_facet_coverage",
            system_prompt=FACET_PLANNER_SYSTEM_PROMPT,
            payload=payload,
            max_tokens=int(audit_config.get("planner_max_tokens", 800)),
            temperature=0.0,
        )
        try:
            raw_ids = value.get("ranked_candidate_ids")
            if not isinstance(raw_ids, list):
                raise ValueError("planner output must contain ranked_candidate_ids")
            requested_ids = list(map(str, raw_ids))
            ids = list(
                dict.fromkeys(candidate_id for candidate_id in requested_ids if candidate_id in lookup)
            )
            if len(ids) < quota:
                raise ValueError(f"planner must rank at least {quota} known candidate IDs")
            ids.extend(candidate_id for candidate_id in lookup if candidate_id not in ids)
            return [lookup[candidate_id] for candidate_id in ids]
        except ValueError as exc:
            last_error = exc
            payload = {**payload, "validation_feedback": str(exc), "attempt": attempt + 1}
    raise RuntimeError(f"facet planner failed for {label}") from last_error


def ground_evidence_plan(
    *,
    candidate_pool: Sequence[Mapping[str, object]],
    labels: Sequence[str],
    sentence_quotas: Mapping[str, int],
    reserve_count: int,
    duplicate_threshold: float,
    judge: OpenAICompatibleJsonClient,
    audit_config: Mapping[str, object],
    tokenizer: SpacySentenceTokenizer,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]], list[dict[str, object]]]:
    progress = progress or (lambda _event: None)
    grounded: dict[str, list[dict[str, object]]] = {}
    audit_log: list[dict[str, object]] = []
    selection_log: list[dict[str, object]] = []
    total = sum(int(sentence_quotas[label]) for label in labels)
    completed = 0
    for label in labels:
        quota = int(sentence_quotas[label])
        facet_pool = [row for row in candidate_pool if row["sub_narrative"] == label]
        selected, reserves = select_diverse_candidates(
            facet_pool,
            quota=quota,
            reserve_count=reserve_count,
            duplicate_threshold=duplicate_threshold,
        )
        ranked = _plan_facet_candidates(
            judge,
            label=label,
            candidates=[*selected, *reserves],
            quota=quota,
            audit_config=audit_config,
        )
        kept: list[dict[str, object]] = []
        for rank, raw in enumerate(ranked, 1):
            if len(kept) == quota:
                break
            candidate = copy.deepcopy(raw)
            candidate["claim_id"] = str(candidate["candidate_id"])
            candidate["document_ids"] = [
                str(row["document_id"]) for row in candidate["supporting_passages"]
            ]
            audit = _audit_claim(judge, claim=candidate, audit_config=audit_config)
            audit_log.append({**audit, "phase": "ledger_initial", "candidate_rank": rank})
            repaired = False
            if audit["status"] != "supported":
                repaired_text = _repair_ledger_claim(
                    judge,
                    claim=candidate,
                    audit_notes=str(audit.get("notes", "")),
                    audit_config=audit_config,
                    tokenizer=tokenizer,
                )
                candidate["text"] = repaired_text
                candidate["ledger_repaired"] = True
                candidate["ledger_original_text"] = raw["text"]
                repair_audit = _audit_claim(
                    judge, claim=candidate, audit_config=audit_config
                )
                audit_log.append(
                    {**repair_audit, "phase": "ledger_repair", "candidate_rank": rank}
                )
                audit = repair_audit
                repaired = True
            accepted = audit["status"] == "supported"
            selection_log.append(
                {
                    "sub_narrative": label,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_rank": rank,
                    "selected_initially": rank <= quota,
                    "ledger_repaired": repaired,
                    "accepted": accepted,
                    "final_status": audit["status"],
                    "text": candidate["text"],
                    "document_ids": candidate["document_ids"],
                }
            )
            if accepted:
                kept.append(candidate)
                completed += 1
                progress(
                    {
                        "stage": "grounding",
                        "message": f"Grounded {completed}/{total} evidence claims",
                        "completed": completed,
                        "total": total,
                    }
                )
        if len(kept) != quota:
            raise RuntimeError(f"only {len(kept)}/{quota} claims survived grounding for {label}")
        grounded[label] = kept
    return grounded, audit_log, selection_log


def _audit_generation(
    generation: Mapping[str, object],
    *,
    judge: OpenAICompatibleJsonClient,
    audit_config: Mapping[str, object],
    only_claim_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for section in generation["sections"]:
        claims = [
            claim
            for claim in section["claims"]
            if only_claim_ids is None or str(claim["claim_id"]) in only_claim_ids
        ]
        if not claims:
            continue
        rows.extend(
            _audit_section(
                judge,
                sub_narrative=str(section["sub_narrative"]),
                claims=claims,
                max_tokens=int(audit_config.get("max_tokens", 350)),
                temperature=float(audit_config.get("temperature", 0.0)),
                validation_attempts=int(audit_config.get("validation_attempts", 3)),
            )
        )
    return rows


def _repair_generation_payload(
    candidate: Mapping[str, object],
    rejected_audits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rejected_by_id = {str(row["claim_id"]): row for row in rejected_audits}
    ledger = {str(row["claim_id"]): row for row in candidate["evidence_ledger"]}
    sentences: list[dict[str, object]] = []
    for section in candidate["sections"]:
        for claim in section["claims"]:
            claim_id = str(claim["claim_id"])
            if claim_id not in rejected_by_id:
                continue
            source_id = str(claim["evidence_claim_ids"][0])
            source = ledger[source_id]
            sentences.append(
                {
                    "claim_id": claim_id,
                    "rejected_text": claim["text"],
                    "audit_notes": rejected_by_id[claim_id].get("notes", ""),
                    "source_claim": source["text"],
                    "cited_passages": source["supporting_passages"],
                }
            )
    return {
        "task": "Repair every rejected sentence without dropping a slot.",
        "output_contract": {
            "root_key": "sentences",
            "item_keys": ["claim_id", "text"],
            "claim_order_must_match_input": True,
            "one_sentence_per_item": True,
        },
        "sentences": sentences,
    }


def _validate_generation_repairs(
    value: Mapping[str, object],
    *,
    payload: Mapping[str, object],
    tokenizer: SpacySentenceTokenizer,
) -> dict[str, str]:
    expected = [str(row["claim_id"]) for row in payload["sentences"]]
    rows = value.get("sentences")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(f"repair response must contain exactly {len(expected)} sentences")
    output: dict[str, str] = {}
    for index, (row, expected_id) in enumerate(zip(rows, expected, strict=True), 1):
        if not isinstance(row, Mapping) or set(row) != {"claim_id", "text"}:
            raise ValueError(f"repair sentence {index} has invalid keys")
        if str(row["claim_id"]) != expected_id:
            raise ValueError(f"repair claim order mismatch at {index}")
        text = _require_nonempty_string(row.get("text"), "repair text")
        if _INLINE_CITATION_RE.search(text) or len(tokenizer.tokenize(text)) != 1:
            raise ValueError(f"repair {expected_id} must be one sentence without citations")
        output[expected_id] = text
    return output


def _generation_with_texts(
    candidate: Mapping[str, object], texts: Mapping[str, str]
) -> dict[str, object]:
    updated = copy.deepcopy(candidate)
    for section in updated["sections"]:
        for claim in section["claims"]:
            claim_id = str(claim["claim_id"])
            if claim_id in texts:
                claim["text"] = texts[claim_id]
    return updated


def _source_fallback_texts(
    candidate: Mapping[str, object], claim_ids: set[str]
) -> dict[str, str]:
    ledger = {str(row["claim_id"]): row for row in candidate["evidence_ledger"]}
    output: dict[str, str] = {}
    for section in candidate["sections"]:
        for claim in section["claims"]:
            claim_id = str(claim["claim_id"])
            if claim_id in claim_ids:
                source_id = str(claim["evidence_claim_ids"][0])
                output[claim_id] = str(ledger[source_id]["text"])
    return output


def _resolve_shared_input(repo_root: Path, relative: str) -> Path:
    requested = Path(relative)
    if requested.is_absolute() and requested.exists():
        return requested
    local = repo_root / requested
    if local.exists():
        return local
    shared_env = os.environ.get("TREC_RAG_SHARED_ROOT")
    if shared_env:
        shared = Path(shared_env).expanduser().resolve() / requested
        if shared.exists():
            return shared
    git_file = repo_root / ".git"
    if git_file.is_file():
        content = git_file.read_text(encoding="utf-8").strip()
        if content.startswith("gitdir:"):
            git_dir = Path(content.split(":", 1)[1].strip()).resolve()
            marker = f"{os.sep}.git{os.sep}"
            text = str(git_dir)
            if marker in text:
                shared_root = Path(text.split(marker, 1)[0])
                shared = shared_root / requested
                if shared.exists():
                    return shared
    raise FileNotFoundError(local)


def _generator_client(
    *,
    output_dir: Path,
    config: Mapping[str, object],
    checkpoint_name: str,
) -> SingleCandidateJsonClient:
    api_base = os.environ.get(
        str(config.get("api_base_env", "")), str(config["api_base"])
    )
    api_key = os.environ.get(
        str(config.get("api_key_env", "")), str(config.get("api_key", ""))
    )
    if not api_key:
        raise RuntimeError(f"missing API key: {config.get('api_key_env')}")
    return SingleCandidateJsonClient(
        api_base=api_base,
        model=str(config["model"]),
        api_key=api_key,
        checkpoint_path=output_dir / checkpoint_name,
        call_log_path=output_dir / "generation_calls.jsonl",
        timeout_seconds=float(config.get("timeout_seconds", 360)),
        transport_max_attempts=int(config.get("transport_max_attempts", 3)),
        request_overrides=config.get("request_overrides", {}),
    )


def _emit(progress: ProgressCallback | None, **event: object) -> None:
    if progress:
        progress(dict(event))


def _sum_usage_cost(receipts: Sequence[Mapping[str, object]]) -> float:
    return sum(_usage_cost(receipt) for receipt in receipts)


def _comparison_report(
    *, metrics: Mapping[str, object], baseline: Mapping[str, object]
) -> str:
    current_all = metrics["nuggets"]["all"]
    current_vital = metrics["nuggets"]["vital"]
    baseline_all = baseline["nuggets"]["all"]
    baseline_vital = baseline["nuggets"]["vital"]
    official = metrics["official_submission"]
    return f"""# GPT-5.6 Sol coverage-repair experiment

This development experiment rebuilt the evidence ledger from the nugget-blind map-stage pool over
all 1,478 passages. It applied semantic deduplication, complexity-weighted facet quotas, up to three
covering citations per claim, pre-generation support repair, and post-generation audit repair.

| Run | Strict | Partial | Vital strict | Claims | Words | Excluded |
|---|---:|---:|---:|---:|---:|---:|
| Previous controlled GPT-5.6 Sol | {baseline_all['strict_coverage']:.3f} | {baseline_all['partial_credit_coverage']:.3f} | {baseline_vital['strict_coverage']:.3f} | {baseline['claim_retention']['submitted_source_claims']} | {baseline['official_submission']['word_count']} | {baseline['official_submission']['excluded_after_support_audit']} |
| Coverage-repair GPT-5.6 Sol | **{current_all['strict_coverage']:.3f}** | **{current_all['partial_credit_coverage']:.3f}** | **{current_vital['strict_coverage']:.3f}** | {official['sentence_count']} | {official['word_count']} | {official['excluded_after_repair']} |

## Change

- Strict: {current_all['strict_coverage'] - baseline_all['strict_coverage']:+.3f}
- Partial credit: {current_all['partial_credit_coverage'] - baseline_all['partial_credit_coverage']:+.3f}
- Vital strict: {current_vital['strict_coverage'] - baseline_vital['strict_coverage']:+.3f}

Organizer nuggets were not loaded until the repaired submission was frozen. The quota design was
informed by earlier Topic 213 development error analysis, so these metrics are development-tuned and
must not be treated as held-out generalization.
"""


def _html_report(
    *, metrics: Mapping[str, object], baseline: Mapping[str, object]
) -> str:
    current_all = metrics["nuggets"]["all"]
    current_vital = metrics["nuggets"]["vital"]
    baseline_all = baseline["nuggets"]["all"]
    baseline_vital = baseline["nuggets"]["vital"]
    rows = [
        ("Strict", baseline_all["strict_coverage"], current_all["strict_coverage"]),
        (
            "Partial credit",
            baseline_all["partial_credit_coverage"],
            current_all["partial_credit_coverage"],
        ),
        ("Vital strict", baseline_vital["strict_coverage"], current_vital["strict_coverage"]),
    ]
    table_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{old:.3f}</td><td>{new:.3f}</td><td class='delta'>{new-old:+.3f}</td></tr>"
        for label, old, new in rows
    )
    official = metrics["official_submission"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPT-5.6 Sol coverage repair</title><style>
body{{font-family:Inter,Segoe UI,sans-serif;margin:0;color:#18201d;background:#f5f7f4}}main{{max-width:960px;margin:auto;padding:48px 24px}}
h1{{font-size:3.25rem;line-height:1.02;margin:0 0 16px}}p{{max-width:72ch;line-height:1.65}}table{{width:100%;border-collapse:collapse;background:white;margin:28px 0}}
th,td{{padding:14px;border-bottom:1px solid #d9dfdb;text-align:left}}thead th{{background:#1d3128;color:white}}.delta{{font-weight:700;color:#0b6b43}}
.band{{border-left:5px solid #e9a23b;padding:12px 18px;background:white}}code{{font-family:ui-monospace,monospace}}@media(max-width:600px){{main{{padding:32px 16px}}h1{{font-size:2.25rem}}table{{font-size:.875rem}}th,td{{padding:10px 8px}}}}</style></head>
<body><main><p><strong>Topic 213 development experiment</strong></p><h1>Coverage before eloquence.</h1>
<p>GPT-5.6 Sol was rerun after rebuilding its evidence ledger from all 1,478 mapped passages, suppressing duplicate claims, assigning facet-specific quotas, selecting up to three covering citations, and repairing support failures instead of deleting them.</p>
<table><thead><tr><th>Metric</th><th>Previous GPT</th><th>Coverage repair</th><th>Change</th></tr></thead><tbody>{table_rows}</tbody></table>
<p class="band">The repaired organizer answer contains <strong>{official['sentence_count']} sentences</strong>, <strong>{official['word_count']} words</strong>, and <strong>{official['excluded_after_repair']} exclusions</strong> after audit.</p>
<h2>Methodological note</h2><p>Nuggets remained unavailable until generation was frozen. The quota design was informed by earlier development-topic errors, so this is a development-tuned result rather than a held-out estimate.</p>
</main></body></html>"""


def run_from_config(
    config_path: Path,
    *,
    generator_client: GeneratorClient | None = None,
    repair_client: GeneratorClient | None = None,
    judge_client: OpenAICompatibleJsonClient | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    config_path = config_path.expanduser().resolve()
    repo_root = find_repo_root(config_path.parent)
    load_repo_env(repo_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    experiment = config["experiment"]
    inputs = config["inputs"]
    selection = config["selection"]
    generation_config = config["generation"]
    audit_config = config["audit"]
    evaluation_config = config["evaluation"]
    output_dir = repo_root / str(experiment["output_dir"])
    report_dir = repo_root / str(experiment["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = SpacySentenceTokenizer()

    source_path = _resolve_shared_input(repo_root, str(inputs["source_generation"]))
    checkpoint_dir = _resolve_shared_input(repo_root, str(inputs["map_checkpoint_dir"]))
    passages_path = _resolve_shared_input(repo_root, str(inputs["passages"]))
    source_generation = json.loads(source_path.read_text(encoding="utf-8"))
    labels = [str(section["sub_narrative"]) for section in source_generation["sections"]]
    facet_config = selection["facets"]
    configured_labels = [str(row["label"]) for row in facet_config]
    if configured_labels != labels:
        raise ValueError("configured facet order must match the source generation")
    sentence_quotas = {str(row["label"]): int(row["sentence_quota"]) for row in facet_config}
    facet_word_targets = {str(row["label"]): int(row["word_target"]) for row in facet_config}

    _emit(progress, stage="evidence", message="Loading full map-stage evidence")
    corpus = load_passage_corpus(passages_path)
    pool, pool_summary = load_map_candidate_pool(
        checkpoint_dir=checkpoint_dir, corpus=corpus, labels=labels
    )
    pool_summary.update(
        {
            "passages_path": str(passages_path),
            "passages_sha256": _sha256(passages_path),
            "passages_processed": len(corpus.passages),
            "organizer_nuggets_available": False,
        }
    )
    judge = judge_client or _make_judge_client(
        output_dir=output_dir, config=audit_config, purpose="coverage_grounding"
    )
    grounded, ledger_audits, selection_log = ground_evidence_plan(
        candidate_pool=pool,
        labels=labels,
        sentence_quotas=sentence_quotas,
        reserve_count=int(selection.get("reserve_count", 7)),
        duplicate_threshold=float(selection.get("duplicate_threshold", 0.68)),
        judge=judge,
        audit_config=audit_config,
        tokenizer=tokenizer,
        progress=progress,
    )
    frozen = build_frozen_evidence_from_claims(
        topic_id=str(source_generation["topic_id"]),
        narrative=str(source_generation["narrative"]),
        source_experiment_id=str(source_generation["experiment_id"]),
        labels=labels,
        claims_by_label=grounded,
        sentence_quotas=sentence_quotas,
        facet_word_targets=facet_word_targets,
        minimum_words=int(selection["minimum_candidate_words"]),
        maximum_words=int(selection["maximum_candidate_words"]),
        source_metadata={
            **dict(source_generation.get("source_metadata", {})),
            "map_checkpoint_aggregate_sha256": pool_summary[
                "checkpoint_aggregate_sha256"
            ],
            "passages_sha256": pool_summary["passages_sha256"],
        },
        input_accounting={
            **dict(source_generation.get("input_accounting", {})),
            "map_candidate_count": pool_summary["candidate_count"],
        },
    )
    payload = build_generation_payload(frozen)
    max_tokens = int(generation_config.get("max_tokens", 4500))
    temperature = float(generation_config.get("temperature", 0.0))
    semantic_hash = semantic_request_sha256(
        system_prompt=GENERATION_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    generator = generator_client or _generator_client(
        output_dir=output_dir,
        config=generation_config,
        checkpoint_name=f"generation_checkpoint__{semantic_hash}.json",
    )

    shutil.copy2(config_path, output_dir / "config.yaml")
    _write_json(output_dir / "candidate_pool_summary.json", pool_summary)
    _write_jsonl(output_dir / "evidence_selection.jsonl", selection_log)
    _write_jsonl(output_dir / "ledger_support_audit.jsonl", ledger_audits)
    _write_json(output_dir / "frozen_evidence_ledger.json", frozen)
    _write_json(output_dir / "frozen_generation_payload.json", payload)
    (output_dir / "generation_system_prompt.txt").write_text(
        GENERATION_SYSTEM_PROMPT + "\n", encoding="utf-8"
    )

    _emit(progress, stage="generation", message="Generating with GPT-5.6 Sol")
    completion = generator.complete_once(
        system_prompt=GENERATION_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    normalized = validate_generation_response(
        completion.parsed, frozen=frozen, tokenizer=tokenizer
    )
    candidate = build_candidate_generation(
        source_generation=source_generation,
        frozen=frozen,
        normalized_sentences=normalized,
        model_key=str(generation_config["key"]),
        model_identity=str(generation_config["model_identity"]),
        receipt=completion.receipt,
        experiment_id=str(experiment["id"]),
        run_id=str(experiment["run_id"]),
    )
    candidate["context_policy"].update(
        {
            "kind": "full-map-diverse-atomic-claims-with-audit-repair",
            "llm_rewrite_or_repair_requests": 0,
        }
    )
    _emit(progress, stage="audit", message="Auditing generated sentences")
    initial_audits = _audit_generation(
        candidate, judge=judge, audit_config=audit_config
    )
    rejected = [row for row in initial_audits if row["status"] != "supported"]
    repair_texts: dict[str, str] = {}
    repair_audits: list[dict[str, object]] = []
    repair_completion: GenerationCompletion | None = None
    if rejected:
        _emit(
            progress,
            stage="repair",
            message=f"Repairing {len(rejected)} rejected sentences with GPT-5.6 Sol",
        )
        repair_payload = _repair_generation_payload(candidate, rejected)
        repair_hash = semantic_request_sha256(
            system_prompt=GENERATION_REPAIR_SYSTEM_PROMPT,
            payload=repair_payload,
            max_tokens=int(generation_config.get("repair_max_tokens", 1800)),
            temperature=0.0,
        )
        repair_generator = repair_client or _generator_client(
            output_dir=output_dir,
            config=generation_config,
            checkpoint_name=f"repair_checkpoint__{repair_hash}.json",
        )
        repair_completion = repair_generator.complete_once(
            system_prompt=GENERATION_REPAIR_SYSTEM_PROMPT,
            payload=repair_payload,
            max_tokens=int(generation_config.get("repair_max_tokens", 1800)),
            temperature=0.0,
        )
        repair_texts = _validate_generation_repairs(
            repair_completion.parsed, payload=repair_payload, tokenizer=tokenizer
        )
        repaired_generation = _generation_with_texts(candidate, repair_texts)
        repair_audits = _audit_generation(
            repaired_generation,
            judge=judge,
            audit_config=audit_config,
            only_claim_ids=set(repair_texts),
        )
        failed_repair_ids = {
            str(row["claim_id"])
            for row in repair_audits
            if row["status"] != "supported"
        }
        if failed_repair_ids:
            fallback_texts = _source_fallback_texts(candidate, failed_repair_ids)
            fallback_generation = _generation_with_texts(candidate, fallback_texts)
            fallback_audits = _audit_generation(
                fallback_generation,
                judge=judge,
                audit_config=audit_config,
                only_claim_ids=failed_repair_ids,
            )
            fallback_by_id = {str(row["claim_id"]): row for row in fallback_audits}
            repair_audits = [
                fallback_by_id.get(str(row["claim_id"]), row) for row in repair_audits
            ]
            repair_texts.update(fallback_texts)

    final, kept_audits, excluded = apply_audited_repairs(
        candidate,
        initial_audits=initial_audits,
        repair_text_by_claim_id=repair_texts,
        repair_audits=repair_audits,
    )
    if excluded and bool(selection.get("require_full_retention", True)):
        raise RuntimeError(f"{len(excluded)} claims still failed after repair")
    final["context_policy"]["llm_rewrite_or_repair_requests"] = 1 if rejected else 0
    official = build_official_entry(
        generation=final,
        team_id=str(experiment["team_id"]),
        run_desc=str(experiment["run_desc"]),
    )
    official_words = validate_official_entry(official, tokenizer=tokenizer)
    minimum_final_words = int(selection.get("minimum_final_words", 900))
    if official_words < minimum_final_words:
        raise ValueError(
            f"repaired answer has {official_words} words; minimum is {minimum_final_words}"
        )
    markdown = render_markdown(final)

    _write_json(output_dir / "raw_generation.json", completion.parsed)
    (output_dir / "raw_generation.txt").write_text(
        completion.raw_content, encoding="utf-8"
    )
    _write_json(output_dir / "generation_receipt.json", completion.receipt)
    if repair_completion:
        _write_json(output_dir / "raw_repair_generation.json", repair_completion.parsed)
        (output_dir / "raw_repair_generation.txt").write_text(
            repair_completion.raw_content, encoding="utf-8"
        )
        _write_json(output_dir / "repair_receipt.json", repair_completion.receipt)
    _write_json(output_dir / "response_generation.candidate.json", candidate)
    _write_jsonl(output_dir / "generation_support_audit.jsonl", initial_audits)
    _write_jsonl(output_dir / "repair_support_audit.jsonl", repair_audits)
    _write_json(output_dir / "response_generation.json", final)
    _write_jsonl(output_dir / "claim_support_audit.jsonl", kept_audits)
    _write_jsonl(output_dir / "excluded_sentences.jsonl", excluded)
    _write_jsonl(output_dir / "rag_output_trec_rag_2026.jsonl", [official])
    (output_dir / "generated_response.md").write_text(markdown, encoding="utf-8")

    generation_names = [
        "config.yaml",
        "candidate_pool_summary.json",
        "evidence_selection.jsonl",
        "ledger_support_audit.jsonl",
        "frozen_evidence_ledger.json",
        "frozen_generation_payload.json",
        "generation_system_prompt.txt",
        "raw_generation.json",
        "raw_generation.txt",
        "generation_receipt.json",
        "response_generation.candidate.json",
        "generation_support_audit.jsonl",
        "repair_support_audit.jsonl",
        "response_generation.json",
        "claim_support_audit.jsonl",
        "excluded_sentences.jsonl",
        "rag_output_trec_rag_2026.jsonl",
        "generated_response.md",
    ]
    if repair_completion:
        generation_names.extend(
            ["raw_repair_generation.json", "raw_repair_generation.txt", "repair_receipt.json"]
        )
    freeze = {
        "schema_version": FREEZE_VERSION,
        "organizer_nuggets_read": False,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [_artifact_row(output_dir, name) for name in generation_names],
    }
    _write_json(output_dir / "generation_freeze.json", freeze)
    freeze_sha256 = _sha256(output_dir / "generation_freeze.json")

    _emit(progress, stage="evaluation", message="Evaluating frozen answer against nuggets")
    nuggets_path = _resolve_shared_input(repo_root, str(inputs["nuggets"]))
    nuggets = load_nuggets(
        nuggets_path,
        topic_id=str(frozen["topic_id"]),
        expected_count=int(evaluation_config.get("expected_nugget_count", 50)),
    )
    evaluator = _make_judge_client(
        output_dir=output_dir, config=audit_config, purpose="coverage_nugget_evaluation"
    )
    comparison = evaluate_nuggets(
        nuggets=nuggets,
        generation=final,
        client=evaluator,
        evaluation_config=evaluation_config,
    )
    metrics = compute_evaluation_metrics(comparison, kept_audits, response_text=markdown)
    receipts = [completion.receipt]
    if repair_completion:
        receipts.append(repair_completion.receipt)
    metrics.update(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment["id"],
            "run_id": experiment["run_id"],
            "topic_id": frozen["topic_id"],
            "model": {
                "key": generation_config["key"],
                "display_name": generation_config["display_name"],
                "identity": generation_config["model_identity"],
                "provider": generation_config["provider"],
            },
            "generation_freeze_sha256": freeze_sha256,
            "nuggets_loaded_after_generation_frozen": True,
            "claim_retention": {
                "candidate_source_claims": int(frozen["supported_source_claim_count"]),
                "submitted_source_claims": len(kept_audits),
                "retention_rate": len(kept_audits)
                / int(frozen["supported_source_claim_count"]),
            },
            "official_submission": {
                "candidate_sentence_count": int(frozen["exact_total_sentence_count"]),
                "candidate_word_count": sum(
                    len(str(claim["text"]).split())
                    for section in candidate["sections"]
                    for claim in section["claims"]
                ),
                "sentence_count": len(official["answer"]),
                "word_count": official_words,
                "reference_count": len(official["references"]),
                "initially_rejected": len(rejected),
                "repaired_and_retained": final["support_filter"][
                    "repair_retained_count"
                ],
                "excluded_after_repair": len(excluded),
                "maximum_words": 1024,
            },
            "coverage_repair": {
                "map_candidates": pool_summary["candidate_count"],
                "passages_processed": pool_summary["passages_processed"],
                "ledger_claims": frozen["supported_source_claim_count"],
                "ledger_repairs": sum(
                    1
                    for facet in frozen["facets"]
                    for claim in facet["source_claims"]
                    if claim["ledger_repaired"]
                ),
                "maximum_citations_per_sentence": 3,
                "organizer_nuggets_available_during_generation": False,
                "development_tuned_quota_design": True,
            },
            "generation": {
                "semantic_generation_request_count": 1,
                "semantic_repair_request_count": 1 if repair_completion else 0,
                "usage": [receipt.get("usage", {}) for receipt in receipts],
                "cost": _sum_usage_cost(receipts),
            },
        }
    )
    baseline_path = _resolve_shared_input(repo_root, str(inputs["baseline_metrics"]))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    _write_jsonl(output_dir / "nugget_comparison.jsonl", comparison)
    _write_json(output_dir / "metrics.json", metrics)
    report = _comparison_report(metrics=metrics, baseline=baseline)
    (output_dir / "comparison_report.md").write_text(report, encoding="utf-8")
    (output_dir / "coverage-repair-report.html").write_text(
        _html_report(metrics=metrics, baseline=baseline), encoding="utf-8"
    )

    publish_names = [
        *generation_names,
        "generation_freeze.json",
        "nugget_comparison.jsonl",
        "metrics.json",
        "comparison_report.md",
        "coverage-repair-report.html",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment["id"],
        "generation_freeze_sha256": freeze_sha256,
        "nuggets_loaded_after_generation_frozen": True,
        "artifacts": [_artifact_row(output_dir, name) for name in publish_names],
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    validate_manifest_hashes(manifest, artifact_root=output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in [*publish_names, "manifest.yaml"]:
        source = output_dir / name
        destination = report_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    _emit(
        progress,
        stage="complete",
        message=(
            f"Strict coverage {metrics['nuggets']['all']['strict_coverage']:.3f}; "
            f"{official_words} submitted words"
        ),
    )
    return report_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)

    def progress(event: dict[str, object]) -> None:
        print(f"[{event.get('stage')}] {event.get('message')}", flush=True)

    report_dir = run_from_config(args.config, progress=progress)
    print(report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
