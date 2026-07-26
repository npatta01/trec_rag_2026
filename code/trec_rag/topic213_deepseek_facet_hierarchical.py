"""Run the two-stage DeepSeek facet hierarchy for Topic 213."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import requests
import yaml

from trec_rag.repo_env import find_repo_root, load_repo_env
from trec_rag.topic213_2026_submission import (
    _read_jsonl,
    build_official_entry,
    render_markdown,
    validate_submission_entry,
)
from trec_rag.topic213_ragnarok_experiment import SpacySentenceTokenizer
from trec_rag.topic213_response_experiment import (
    OpenAICompatibleJsonClient,
    _audit_section,
    _parse_json_object,
    _require_nonempty_string,
    _sha256,
    _write_json,
    _write_jsonl,
    compute_evaluation_metrics,
    evaluate_nuggets,
    load_nuggets,
)
from trec_rag.topic213_utokyo_experiment import normalize_answer_label


SCHEMA_VERSION = "topic213-deepseek-v4-flash-facet-hierarchical-v1"
PROMPT_VERSION = "supported-claims-facet-extract-then-synthesize-v1"
EXPECTED_FACET_COUNT = 10
EXPECTED_DEEPSEEK_CALLS = 11
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

EXTRACTION_SYSTEM_PROMPT = """You extract atomic candidate facts for one sub-narrative from a
frozen list of source claims that were already independently judged fully supported. These are
generation-time candidate facts, not organizer evaluation nuggets. You do not know the organizer
nuggets and must not infer or mention them.

Return one JSON object with a `candidate_facts` array. Every item must contain exactly `text` and
`source_claim_ids`. Use only facts stated by the supplied source claim text. Make each fact one
grammatical sentence and as atomic as practical. Preserve distinct useful details, merge exact
duplicates, and do not introduce outside knowledge. Cite one to three supplied source claim IDs for
each fact; prefer one ID when it fully supports the fact."""

SYNTHESIS_SYSTEM_PROMPT = """You synthesize a concise report exclusively from a ledger of candidate
facts produced by an earlier extraction stage. Return one JSON object with a `sections` array.
Preserve every supplied sub-narrative exactly once and in the supplied order. Every section must
contain `sub_narrative` and a nonempty `answer` array. Every answer item must contain exactly `text`
and `extracted_fact_ids`.

Each answer text must be one polished grammatical sentence supported in full by one to three fact
IDs from the same section. Combine compatible facts when useful, but do not add facts, headings,
introductions, conclusions, citation markers, or outside knowledge. Cover as many distinct candidate
facts as possible within the supplied sentence and word budgets."""


class JsonClient(Protocol):
    model: str

    def complete_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: Mapping[str, object],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]: ...


class SingleShotOpenRouterJsonClient:
    """OpenAI-compatible JSON client with exactly one HTTP attempt per cache miss."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str,
        checkpoint_dir: Path,
        call_log_path: Path,
        timeout_seconds: float,
        provider_order: Sequence[str] = (),
        allow_provider_fallbacks: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.checkpoint_dir = checkpoint_dir
        self.call_log_path = call_log_path
        self.timeout_seconds = timeout_seconds
        self.provider_order = list(provider_order)
        self.allow_provider_fallbacks = allow_provider_fallbacks
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.call_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, object]] = []
        self.network_request_count = 0

    def _log(self, row: Mapping[str, object]) -> None:
        record = dict(row)
        self.records.append(record)
        with self.call_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def complete_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: Mapping[str, object],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]:
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "include_reasoning": False,
            "reasoning": {"enabled": False},
        }
        if self.provider_order:
            request_body["provider"] = {
                "order": self.provider_order,
                "allow_fallbacks": self.allow_provider_fallbacks,
                "require_parameters": True,
            }
        canonical = json.dumps(request_body, ensure_ascii=False, sort_keys=True).encode("utf-8")
        request_hash = hashlib.sha256(canonical).hexdigest()
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage)
        legacy_checkpoint = self.checkpoint_dir / f"{safe_stage}__{request_hash}.json"
        short_checkpoint = self.checkpoint_dir / f"{request_hash}.json"
        checkpoint = legacy_checkpoint if legacy_checkpoint.exists() else short_checkpoint
        if checkpoint.exists():
            stored = json.loads(checkpoint.read_text(encoding="utf-8"))
            parsed = stored.get("parsed")
            if not isinstance(parsed, dict):
                raise ValueError(f"invalid DeepSeek checkpoint: {checkpoint}")
            self._log(
                {
                    "stage": stage,
                    "request_sha256": request_hash,
                    "cache_hit": True,
                    "network_request": False,
                    "model": self.model,
                    "response_model": stored.get("response_model"),
                    "provider": stored.get("provider"),
                    "finish_reason": stored.get("finish_reason"),
                    "usage": stored.get("usage", {}),
                }
            )
            return parsed

        self.network_request_count += 1
        started = time.monotonic()
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/npatta01/trec_rag_2026",
            "X-Title": "TREC RAG Topic 213 facet hierarchy",
        }
        try:
            response = requests.post(
                url,
                headers=headers,
                json=request_body,
                timeout=(10, self.timeout_seconds),
            )
            response.raise_for_status()
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("completion content is not text")
            parsed = _parse_json_object(content)
        except Exception as exc:
            self._log(
                {
                    "stage": stage,
                    "request_sha256": request_hash,
                    "cache_hit": False,
                    "network_request": True,
                    "model": self.model,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "error": str(exc),
                }
            )
            raise RuntimeError(f"single-shot DeepSeek call failed at {stage}") from exc

        stored = {
            "request_sha256": request_hash,
            "parsed": parsed,
            "response_id": response.headers.get("X-Generation-Id") or envelope.get("id"),
            "response_model": envelope.get("model"),
            "provider": envelope.get("provider"),
            "finish_reason": envelope.get("choices", [{}])[0].get("finish_reason"),
            "usage": envelope.get("usage", {}),
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(
            json.dumps(stored, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._log(
            {
                "stage": stage,
                "request_sha256": request_hash,
                "cache_hit": False,
                "network_request": True,
                "model": self.model,
                "response_model": envelope.get("model"),
                "provider": envelope.get("provider"),
                "response_id": stored["response_id"],
                "finish_reason": stored["finish_reason"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "input_chars": sum(len(str(row["content"])) for row in request_body["messages"]),
                "output_chars": len(content),
                "usage": envelope.get("usage", {}),
            }
        )
        return parsed


def load_supported_source_claims(
    source_generation: Mapping[str, object],
    support_rows: Sequence[Mapping[str, object]],
    *,
    expected_claim_count: int = 42,
) -> tuple[list[str], dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
    sections = source_generation.get("sections")
    if not isinstance(sections, list) or len(sections) != EXPECTED_FACET_COUNT:
        raise ValueError(f"source generation must have {EXPECTED_FACET_COUNT} sections")
    status_by_id: dict[str, str] = {}
    for row in support_rows:
        claim_id = _require_nonempty_string(row.get("claim_id"), "audit claim_id")
        if claim_id in status_by_id:
            raise ValueError(f"duplicate support audit claim: {claim_id}")
        status_by_id[claim_id] = _require_nonempty_string(row.get("status"), "audit status")

    labels: list[str] = []
    claims_by_label: dict[str, list[dict[str, object]]] = {}
    source_by_id: dict[str, dict[str, object]] = {}
    for section in sections:
        if not isinstance(section, Mapping):
            raise ValueError("source section must be an object")
        label = _require_nonempty_string(section.get("sub_narrative"), "sub_narrative")
        if label in claims_by_label:
            raise ValueError(f"duplicate sub-narrative: {label}")
        labels.append(label)
        claims_by_label[label] = []
        claims = section.get("claims")
        if not isinstance(claims, list):
            raise ValueError("source claims must be an array")
        for raw_claim in claims:
            if not isinstance(raw_claim, Mapping):
                raise ValueError("source claim must be an object")
            claim_id = _require_nonempty_string(raw_claim.get("claim_id"), "source claim_id")
            if status_by_id.get(claim_id) != "supported":
                continue
            claim = {**dict(raw_claim), "sub_narrative": label}
            claims_by_label[label].append(claim)
            source_by_id[claim_id] = claim
        if not claims_by_label[label]:
            raise ValueError(f"sub-narrative has no supported source claims: {label}")
    if len(source_by_id) != expected_claim_count:
        raise ValueError(
            f"expected {expected_claim_count} supported source claims, found {len(source_by_id)}"
        )
    return labels, claims_by_label, source_by_id


def build_generation_input(
    labels: Sequence[str], claims_by_label: Mapping[str, Sequence[Mapping[str, object]]]
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "organizer_nuggets_available": False,
        "supported_source_claim_count": sum(len(claims_by_label[label]) for label in labels),
        "sections": [
            {
                "sub_narrative": label,
                "source_claims": [
                    {"claim_id": claim["claim_id"], "text": claim["text"]}
                    for claim in claims_by_label[label]
                ],
            }
            for label in labels
        ],
    }


def extract_candidate_facts(
    *,
    client: JsonClient,
    labels: Sequence[str],
    claims_by_label: Mapping[str, Sequence[Mapping[str, object]]],
    tokenizer: SpacySentenceTokenizer,
    max_tokens: int,
    temperature: float,
    maximum_facts_per_facet: int,
) -> list[dict[str, object]]:
    extracted_sections: list[dict[str, object]] = []
    for facet_number, label in enumerate(labels, 1):
        input_claims = [
            {"claim_id": str(claim["claim_id"]), "text": str(claim["text"])}
            for claim in claims_by_label[label]
        ]
        allowed_ids = {row["claim_id"] for row in input_claims}
        raw = client.complete_json(
            stage=f"deepseek_extract_facet_{facet_number:02d}",
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            payload={
                "task": "Extract atomic supported candidate facts for this sub-narrative.",
                "sub_narrative": label,
                "source_claims": input_claims,
                "maximum_candidate_facts": maximum_facts_per_facet,
            },
            max_tokens=max_tokens,
            temperature=temperature,
        )
        rows = raw.get("candidate_facts")
        if not isinstance(rows, list) or not rows or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError(f"facet {facet_number} candidate_facts must be a nonempty object array")
        normalized: list[dict[str, object]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for row in rows:
            text = _require_nonempty_string(row.get("text"), "candidate fact text")
            sentences = tokenizer.tokenize(text)
            if len(sentences) != 1:
                raise ValueError(f"candidate fact must be exactly one SpaCy sentence: {text}")
            source_ids = row.get("source_claim_ids")
            if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 3:
                raise ValueError("candidate fact must cite one to three source claim IDs")
            source_ids = list(dict.fromkeys(map(str, source_ids)))
            if any(source_id not in allowed_ids for source_id in source_ids):
                raise ValueError("candidate fact cites an unknown or cross-facet source claim")
            key = (sentences[0].casefold(), tuple(source_ids))
            if key in seen:
                continue
            seen.add(key)
            normalized.append({"text": sentences[0], "source_claim_ids": source_ids})
        for fact_number, row in enumerate(normalized, 1):
            row["fact_id"] = f"F{facet_number:02d}-{fact_number:03d}"
        extracted_sections.append(
            {
                "facet_number": facet_number,
                "sub_narrative": label,
                "input_source_claim_ids": [row["claim_id"] for row in input_claims],
                "requested_maximum_candidate_facts": maximum_facts_per_facet,
                "model_candidate_fact_count": len(normalized),
                "candidate_fact_budget_exceeded": len(normalized) > maximum_facts_per_facet,
                "candidate_facts": normalized,
            }
        )
    return extracted_sections


def synthesize_from_extractions(
    *,
    client: JsonClient,
    extracted_sections: Sequence[Mapping[str, object]],
    tokenizer: SpacySentenceTokenizer,
    max_tokens: int,
    temperature: float,
    maximum_sentences_per_section: int,
    target_maximum_words: int,
    maximum_words: int = 1024,
) -> list[dict[str, object]]:
    labels = [str(section["sub_narrative"]) for section in extracted_sections]
    fact_by_id: dict[str, Mapping[str, object]] = {}
    stage_a_outputs: list[dict[str, object]] = []
    for section in extracted_sections:
        facts = section["candidate_facts"]
        if not isinstance(facts, list):
            raise ValueError("candidate facts must be an array")
        normalized_facts = []
        for fact in facts:
            if not isinstance(fact, Mapping):
                raise ValueError("candidate fact must be an object")
            fact_id = str(fact["fact_id"])
            fact_by_id[fact_id] = fact
            normalized_facts.append({"fact_id": fact_id, "text": fact["text"]})
        stage_a_outputs.append(
            {"sub_narrative": section["sub_narrative"], "candidate_facts": normalized_facts}
        )
    raw = client.complete_json(
        stage="deepseek_synthesize_all_facets",
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        payload={
            "task": "Synthesize all ten sub-narratives from only these Stage-A outputs.",
            "maximum_sentences_per_section": maximum_sentences_per_section,
            "target_maximum_total_words": target_maximum_words,
            "stage_a_outputs": stage_a_outputs,
        },
        max_tokens=max_tokens,
        temperature=temperature,
    )
    raw_sections = raw.get("sections")
    if not isinstance(raw_sections, list) or any(
        not isinstance(section, Mapping) for section in raw_sections
    ):
        raise ValueError("synthesis sections must be an array of objects")
    by_label: dict[str, Mapping[str, object]] = {}
    for section in raw_sections:
        label = normalize_answer_label(section.get("sub_narrative"), labels)
        if label in by_label:
            raise ValueError(f"duplicate synthesized section: {label}")
        by_label[label] = section
    if list(by_label) != labels:
        raise ValueError("synthesis must preserve every Stage-A section in order")

    normalized: list[dict[str, object]] = []
    total_words = 0
    for section_index, label in enumerate(labels):
        facts = extracted_sections[section_index]["candidate_facts"]
        allowed_ids = {str(fact["fact_id"]) for fact in facts}
        answer = by_label[label].get("answer")
        if not isinstance(answer, list) or not answer or any(
            not isinstance(item, Mapping) for item in answer
        ):
            raise ValueError(f"synthesis section has no answer: {label}")
        normalized_answer: list[dict[str, object]] = []
        for item in answer:
            text = _require_nonempty_string(item.get("text"), "synthesized sentence")
            sentences = tokenizer.tokenize(text)
            if len(sentences) != 1:
                raise ValueError(f"synthesized answer item must be one SpaCy sentence: {text}")
            fact_ids = item.get("extracted_fact_ids")
            if not isinstance(fact_ids, list) or not fact_ids:
                raise ValueError("synthesized sentence must cite extracted fact IDs")
            fact_ids = list(dict.fromkeys(map(str, fact_ids)))
            if any(fact_id not in allowed_ids for fact_id in fact_ids):
                raise ValueError("synthesized sentence cites an unknown or cross-facet fact")
            total_words += len(sentences[0].split())
            normalized_answer.append(
                {"text": sentences[0], "extracted_fact_ids": fact_ids}
            )
        if len(normalized_answer) > maximum_sentences_per_section:
            raise ValueError(f"synthesis section exceeds sentence budget: {label}")
        normalized.append({"sub_narrative": label, "answer": normalized_answer})
    if total_words > maximum_words:
        raise ValueError(f"synthesis has {total_words} words; organizer maximum is {maximum_words}")
    return normalized


def _passage_score(text: str, passage: Mapping[str, object]) -> tuple[float, int, int]:
    sentence_terms = set(_WORD_RE.findall(text.casefold()))
    passage_terms = set(_WORD_RE.findall(str(passage.get("text", "")).casefold()))
    overlap = len(sentence_terms & passage_terms)
    coverage = overlap / len(sentence_terms) if sentence_terms else 0.0
    return coverage, overlap, len(str(passage.get("text", "")))


def select_citation_passages(
    text: str,
    source_claim_ids: Sequence[str],
    source_by_id: Mapping[str, Mapping[str, object]],
    *,
    maximum_citations: int = 3,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for source_id in source_claim_ids:
        source = source_by_id[source_id]
        passages = source.get("supporting_passages")
        if not isinstance(passages, list) or not passages or any(
            not isinstance(passage, Mapping) for passage in passages
        ):
            raise ValueError(f"source claim {source_id} has no supporting passage")
        best = max(passages, key=lambda passage: _passage_score(text, passage))
        candidates.append(dict(best))
    candidates.sort(key=lambda passage: _passage_score(text, passage), reverse=True)
    selected: list[dict[str, object]] = []
    seen_documents: set[str] = set()
    for passage in candidates:
        document_id = str(passage.get("document_id", ""))
        if not document_id.startswith("shard_"):
            raise ValueError(f"non-ClimbMix document ID: {document_id}")
        if document_id in seen_documents:
            continue
        selected.append(passage)
        seen_documents.add(document_id)
        if len(selected) == maximum_citations:
            break
    if not selected:
        raise ValueError("synthesized sentence has no citation passage")
    return selected


def build_candidate_generation(
    *,
    source_generation: Mapping[str, object],
    extracted_sections: Sequence[Mapping[str, object]],
    synthesized_sections: Sequence[Mapping[str, object]],
    source_by_id: Mapping[str, Mapping[str, object]],
    experiment: Mapping[str, object],
    generation_config: Mapping[str, object],
) -> dict[str, object]:
    fact_by_id = {
        str(fact["fact_id"]): {**dict(fact), "sub_narrative": section["sub_narrative"]}
        for section in extracted_sections
        for fact in section["candidate_facts"]
    }
    output_sections: list[dict[str, object]] = []
    used_fact_ids: list[str] = []
    used_source_ids: list[str] = []
    sentence_number = 0
    for section in synthesized_sections:
        claims: list[dict[str, object]] = []
        for item in section["answer"]:
            sentence_number += 1
            fact_ids = list(map(str, item["extracted_fact_ids"]))
            source_ids = list(
                dict.fromkeys(
                    str(source_id)
                    for fact_id in fact_ids
                    for source_id in fact_by_id[fact_id]["source_claim_ids"]
                )
            )
            passages = select_citation_passages(
                str(item["text"]), source_ids, source_by_id, maximum_citations=3
            )
            claims.append(
                {
                    "claim_id": f"H{sentence_number:03d}",
                    "text": item["text"],
                    "extracted_fact_ids": fact_ids,
                    "evidence_claim_ids": source_ids,
                    "document_ids": [str(passage["document_id"]) for passage in passages],
                    "supporting_passages": passages,
                }
            )
            for fact_id in fact_ids:
                if fact_id not in used_fact_ids:
                    used_fact_ids.append(fact_id)
            for source_id in source_ids:
                if source_id not in used_source_ids:
                    used_source_ids.append(source_id)
        output_sections.append(
            {"sub_narrative": section["sub_narrative"], "claims": claims}
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "experiment_id": experiment["id"],
        "run_id": experiment["run_id"],
        "topic_id": source_generation["topic_id"],
        "narrative": source_generation["narrative"],
        "model": generation_config["model"],
        "model_identity": generation_config["model"],
        "temperature": generation_config.get("temperature", 0.0),
        "context_policy": {
            "kind": "supported_claims_facet_extraction_then_global_synthesis",
            "source_experiment_id": source_generation["experiment_id"],
            "supported_source_claim_count": len(source_by_id),
            "deepseek_generation_calls": EXPECTED_DEEPSEEK_CALLS,
            "nuggets_available_during_generation": False,
            "synthesis_input": "stage_a_outputs_only",
        },
        "input_accounting": source_generation["input_accounting"],
        "source_metadata": source_generation["source_metadata"],
        "generation_config": dict(generation_config),
        "prompts": {
            "extraction_system": EXTRACTION_SYSTEM_PROMPT,
            "synthesis_system": SYNTHESIS_SYSTEM_PROMPT,
        },
        "extracted_fact_ledger": [fact_by_id[fact_id] for fact_id in used_fact_ids],
        "evidence_ledger": [source_by_id[source_id] for source_id in used_source_ids],
        "sections": output_sections,
    }


def filter_supported_sentences(
    generation: dict[str, object],
    audit_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    audit_by_id = {str(row["claim_id"]): dict(row) for row in audit_rows}
    kept_audits: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for section in generation["sections"]:
        kept_claims = []
        for claim in section["claims"]:
            audit = audit_by_id[str(claim["claim_id"])]
            if audit["status"] == "supported":
                kept_claims.append(claim)
                kept_audits.append(audit)
            else:
                excluded.append({**dict(claim), "support_audit": audit})
        section["claims"] = kept_claims
    used_fact_ids = {
        fact_id
        for section in generation["sections"]
        for claim in section["claims"]
        for fact_id in claim["extracted_fact_ids"]
    }
    used_source_ids = {
        source_id
        for section in generation["sections"]
        for claim in section["claims"]
        for source_id in claim["evidence_claim_ids"]
    }
    generation["extracted_fact_ledger"] = [
        row for row in generation["extracted_fact_ledger"] if row["fact_id"] in used_fact_ids
    ]
    generation["evidence_ledger"] = [
        row for row in generation["evidence_ledger"] if row["claim_id"] in used_source_ids
    ]
    return kept_audits, excluded


def build_lineage_rows(
    generation: Mapping[str, object],
    extracted_sections: Sequence[Mapping[str, object]],
    source_by_id: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    fact_by_id = {
        str(fact["fact_id"]): fact
        for section in extracted_sections
        for fact in section["candidate_facts"]
    }
    rows: list[dict[str, object]] = []
    for section in generation["sections"]:
        for claim in section["claims"]:
            facts = []
            for fact_id in claim["extracted_fact_ids"]:
                fact = fact_by_id[str(fact_id)]
                sources = []
                for source_id in fact["source_claim_ids"]:
                    source = source_by_id[str(source_id)]
                    sources.append(
                        {
                            "source_claim_id": source_id,
                            "text": source["text"],
                            "supporting_passages": source["supporting_passages"],
                        }
                    )
                facts.append(
                    {
                        "fact_id": fact_id,
                        "text": fact["text"],
                        "source_claims": sources,
                    }
                )
            rows.append(
                {
                    "claim_id": claim["claim_id"],
                    "sub_narrative": section["sub_narrative"],
                    "text": claim["text"],
                    "official_document_citations": claim["document_ids"],
                    "official_supporting_passages": claim["supporting_passages"],
                    "extracted_facts": facts,
                }
            )
    return rows


def summarize_deepseek_calls(client: SingleShotOpenRouterJsonClient) -> dict[str, object]:
    if len(client.records) != EXPECTED_DEEPSEEK_CALLS:
        raise ValueError(
            f"expected {EXPECTED_DEEPSEEK_CALLS} logical DeepSeek calls, found {len(client.records)}"
        )
    stage_counts = Counter(
        "extraction" if str(row["stage"]).startswith("deepseek_extract_facet_") else "synthesis"
        for row in client.records
    )
    if stage_counts != Counter({"extraction": 10, "synthesis": 1}):
        raise ValueError(f"unexpected DeepSeek stage calls: {dict(stage_counts)}")
    usage_totals: dict[str, float] = {}
    for record in client.records:
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage_totals[key] = usage_totals.get(key, 0.0) + float(value)
    return {
        "expected_logical_calls": EXPECTED_DEEPSEEK_CALLS,
        "logical_calls": len(client.records),
        "extraction_calls": stage_counts["extraction"],
        "synthesis_calls": stage_counts["synthesis"],
        "network_requests_this_invocation": client.network_request_count,
        "cache_hits_this_invocation": sum(bool(row.get("cache_hit")) for row in client.records),
        "originating_network_calls": len(client.records),
        "requested_model": client.model,
        "response_models": list(
            dict.fromkeys(
                str(row["response_model"])
                for row in client.records
                if row.get("response_model")
            )
        ),
        "providers": list(
            dict.fromkeys(str(row["provider"]) for row in client.records if row.get("provider"))
        ),
        "usage_totals": usage_totals,
        "calls": client.records,
    }


def write_generation_freeze(output_dir: Path, relative_paths: Sequence[str]) -> dict[str, object]:
    artifacts = []
    for relative in relative_paths:
        path = output_dir / relative
        if not path.is_file():
            raise ValueError(f"cannot freeze missing artifact: {relative}")
        artifacts.append({"path": relative.replace("\\", "/"), "sha256": _sha256(path)})
    freeze = {
        "schema_version": "generation-freeze-v1",
        "organizer_nuggets_read": False,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "generation_freeze.json", freeze)
    return freeze


def validate_manifest_hashes(
    manifest: Mapping[str, object], *, artifact_root: Path, repo_root: Path
) -> int:
    checked = 0
    for group, base in (("source_inputs", repo_root), ("artifacts", artifact_root)):
        rows = manifest.get(group)
        if not isinstance(rows, list):
            raise ValueError(f"manifest {group} must be an array")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"manifest {group} row must be an object")
            relative = _require_nonempty_string(row.get("path"), "manifest path")
            expected = _require_nonempty_string(row.get("sha256"), "manifest sha256")
            actual = _sha256(base / relative)
            if actual != expected:
                raise ValueError(f"manifest hash mismatch: {relative}")
            checked += 1
    return checked


def _artifact_rows(output_dir: Path, relative_paths: Sequence[str]) -> list[dict[str, str]]:
    return [
        {"path": relative.replace("\\", "/"), "sha256": _sha256(output_dir / relative)}
        for relative in relative_paths
    ]


def _publish_report_artifacts(
    output_dir: Path, report_dir: Path, relative_paths: Sequence[str]
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for relative in relative_paths:
        source = output_dir / relative
        destination = report_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _comparison_summary(current: Mapping[str, object], baseline: Mapping[str, object]) -> dict[str, object]:
    current_nuggets = current["nuggets"]
    baseline_nuggets = baseline["nuggets"]
    return {
        "baseline_experiment_id": baseline["experiment_id"],
        "strict_coverage_delta": (
            current_nuggets["all"]["strict_coverage"]
            - baseline_nuggets["all"]["strict_coverage"]
        ),
        "partial_credit_coverage_delta": (
            current_nuggets["all"]["partial_credit_coverage"]
            - baseline_nuggets["all"]["partial_credit_coverage"]
        ),
        "vital_strict_coverage_delta": (
            current_nuggets["vital"]["strict_coverage"]
            - baseline_nuggets["vital"]["strict_coverage"]
        ),
    }


def _render_report(
    *, metrics: Mapping[str, object], baseline: Mapping[str, object], call_summary: Mapping[str, object]
) -> str:
    current_all = metrics["nuggets"]["all"]
    baseline_all = baseline["nuggets"]["all"]
    current_official = metrics["official_submission"]
    baseline_official = baseline["official_submission"]
    strict_delta = current_all["strict_coverage"] - baseline_all["strict_coverage"]
    partial_delta = current_all["partial_credit_coverage"] - baseline_all["partial_credit_coverage"]
    winner = (
        "the hierarchical DeepSeek run"
        if (current_all["strict_coverage"], current_all["partial_credit_coverage"])
        > (baseline_all["strict_coverage"], baseline_all["partial_credit_coverage"])
        else "the format-compliant full-evidence Qwen baseline"
    )
    usage = call_summary["usage_totals"]
    return f"""# Topic 213: DeepSeek V4 Flash facet hierarchy

This controlled run used only the 42 source claims previously judged supported. DeepSeek made one
atomic-fact extraction call for each of the ten sub-narratives, followed by one synthesis call over
only those extraction outputs. Local Qwen independently audited the synthesized sentences and
evaluated organizer nuggets only after all generation artifacts and the official submission were
frozen.

## Organizer-format result

- Submitted sentences: {current_official['sentence_count']}
- Answer words: {current_official['word_count']} / 1024
- Direct `shard_*` references: {current_official['reference_count']}
- Excluded unsupported sentences: {current_official['excluded_after_support_audit']}
- Citation coverage: {metrics['answer_claims']['citation_coverage']:.3f}
- Unsupported submitted sentences: {metrics['answer_claims']['unsupported_claim_count']}

## Nugget coverage

| Metric | Qwen one-shot baseline | DeepSeek facet hierarchy | Delta |
|---|---:|---:|---:|
| Strict coverage | {baseline_all['strict_coverage']:.3f} | {current_all['strict_coverage']:.3f} | {strict_delta:+.3f} |
| Partial-credit coverage | {baseline_all['partial_credit_coverage']:.3f} | {current_all['partial_credit_coverage']:.3f} | {partial_delta:+.3f} |
| Vital strict coverage | {baseline['nuggets']['vital']['strict_coverage']:.3f} | {metrics['nuggets']['vital']['strict_coverage']:.3f} | {metrics['nuggets']['vital']['strict_coverage'] - baseline['nuggets']['vital']['strict_coverage']:+.3f} |
| Submitted sentences | {baseline_official['sentence_count']} | {current_official['sentence_count']} | {current_official['sentence_count'] - baseline_official['sentence_count']:+d} |
| Answer words | {baseline_official['word_count']} | {current_official['word_count']} | {current_official['word_count'] - baseline_official['word_count']:+d} |

## Call accounting

- DeepSeek model requested: `{call_summary['requested_model']}`
- Logical generation calls: {call_summary['logical_calls']} ({call_summary['extraction_calls']} extraction + {call_summary['synthesis_calls']} synthesis)
- Originating model generations represented: {call_summary['originating_network_calls']}
- Network requests in this invocation: {call_summary['network_requests_this_invocation']}
- Resumed checkpoint reads in this invocation: {call_summary['cache_hits_this_invocation']}
- Prompt tokens: {int(usage.get('prompt_tokens', 0))}
- Completion tokens: {int(usage.get('completion_tokens', 0))}
- Total tokens: {int(usage.get('total_tokens', 0))}

## Recommendation

On strict nugget coverage, {winner} is the better of these two controlled format-compliant runs.
The row-level nugget comparison and full sentence-to-fact-to-source-to-passage lineage are retained
for diagnosing which facets gained or lost coverage. This report does not compare the separate
DeepSeek one-shot and GPT one-shot worktrees.
"""


def run_from_config(config_path: Path) -> Path:
    repo_root = find_repo_root(config_path.parent)
    load_repo_env(repo_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    experiment = config["experiment"]
    inputs = config["inputs"]
    generation_config = config["generation"]
    audit_config = config["audit"]
    evaluation_config = config["evaluation"]
    if not all(
        isinstance(section, Mapping)
        for section in (experiment, inputs, generation_config, audit_config, evaluation_config)
    ):
        raise ValueError("config sections must be mappings")

    output_dir = repo_root / str(experiment["output_dir"])
    report_dir = repo_root / str(experiment["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_generation_path = repo_root / str(inputs["source_generation"])
    source_audit_path = repo_root / str(inputs["source_support_audit"])
    nuggets_path = repo_root / str(inputs["nuggets"])
    baseline_metrics_path = repo_root / str(inputs["baseline_metrics"])
    source_generation = json.loads(source_generation_path.read_text(encoding="utf-8"))
    source_audit = _read_jsonl(source_audit_path)
    labels, claims_by_label, source_by_id = load_supported_source_claims(
        source_generation,
        source_audit,
        expected_claim_count=int(generation_config.get("expected_supported_source_claims", 42)),
    )
    tokenizer = SpacySentenceTokenizer(str(generation_config.get("spacy_model", "")) or None)

    deepseek_client = SingleShotOpenRouterJsonClient(
        api_base=str(generation_config.get("api_base", "https://openrouter.ai/api/v1")),
        model=str(generation_config["model"]),
        api_key=os.environ.get(str(generation_config.get("api_key_env", "OPENROUTER_API_KEY")), ""),
        checkpoint_dir=output_dir / "deepseek_checkpoints",
        call_log_path=output_dir / "deepseek_calls.jsonl",
        timeout_seconds=float(generation_config.get("timeout_seconds", 240)),
        provider_order=list(map(str, generation_config.get("provider_order", []))),
        allow_provider_fallbacks=bool(
            generation_config.get("allow_provider_fallbacks", False)
        ),
    )
    generation_input = build_generation_input(labels, claims_by_label)
    extracted_sections = extract_candidate_facts(
        client=deepseek_client,
        labels=labels,
        claims_by_label=claims_by_label,
        tokenizer=tokenizer,
        max_tokens=int(generation_config.get("extraction_max_tokens", 1200)),
        temperature=float(generation_config.get("temperature", 0.0)),
        maximum_facts_per_facet=int(generation_config.get("maximum_facts_per_facet", 12)),
    )
    synthesized_sections = synthesize_from_extractions(
        client=deepseek_client,
        extracted_sections=extracted_sections,
        tokenizer=tokenizer,
        max_tokens=int(generation_config.get("synthesis_max_tokens", 4000)),
        temperature=float(generation_config.get("temperature", 0.0)),
        maximum_sentences_per_section=int(
            generation_config.get("maximum_sentences_per_section", 4)
        ),
        target_maximum_words=int(generation_config.get("target_maximum_words", 900)),
    )
    call_summary = summarize_deepseek_calls(deepseek_client)
    candidate = build_candidate_generation(
        source_generation=source_generation,
        extracted_sections=extracted_sections,
        synthesized_sections=synthesized_sections,
        source_by_id=source_by_id,
        experiment=experiment,
        generation_config=generation_config,
    )

    qwen_client = OpenAICompatibleJsonClient(
        api_base=str(audit_config.get("api_base", "http://localhost:4000/v1")),
        model=str(audit_config.get("model", "qwen-local")),
        api_key=str(audit_config.get("api_key", "none")),
        checkpoint_dir=output_dir / "qwen_checkpoints",
        call_log_path=output_dir / "qwen_calls.jsonl",
        timeout_seconds=float(audit_config.get("timeout_seconds", 240)),
        max_attempts=int(audit_config.get("http_max_attempts", 4)),
    )
    full_audit: list[dict[str, object]] = []
    for section in candidate["sections"]:
        full_audit.extend(
            _audit_section(
                qwen_client,
                sub_narrative=str(section["sub_narrative"]),
                claims=section["claims"],
                max_tokens=int(audit_config.get("max_tokens", 350)),
                temperature=0.0,
                validation_attempts=int(audit_config.get("validation_attempts", 3)),
            )
        )
    candidate_before_audit = json.loads(json.dumps(candidate))
    final_audit, excluded = filter_supported_sentences(candidate, full_audit)
    official_entry = build_official_entry(
        generation=candidate,
        team_id=str(experiment["team_id"]),
        run_desc=str(experiment["run_desc"]),
    )
    official_words = validate_submission_entry(official_entry)
    for answer_item in official_entry["answer"]:
        if len(tokenizer.tokenize(str(answer_item["text"]))) != 1:
            raise ValueError("official answer item is not exactly one SpaCy sentence")

    facet_dir = output_dir / "facet_extractions"
    facet_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "generation_input_claims.json", generation_input)
    _write_jsonl(output_dir / "facet_extractions.jsonl", extracted_sections)
    for section in extracted_sections:
        _write_json(
            facet_dir / f"facet_{int(section['facet_number']):02d}.json",
            section,
        )
    _write_json(
        output_dir / "deepseek_synthesis.json",
        {"schema_version": SCHEMA_VERSION, "sections": synthesized_sections},
    )
    _write_json(output_dir / "candidate_response_generation.json", candidate_before_audit)
    _write_json(output_dir / "response_generation.json", candidate)
    response_text = render_markdown(candidate)
    (output_dir / "generated_response.md").write_text(response_text, encoding="utf-8")
    _write_jsonl(output_dir / "generation_support_audit.jsonl", full_audit)
    _write_jsonl(output_dir / "claim_support_audit.jsonl", final_audit)
    _write_jsonl(output_dir / "excluded_sentences.jsonl", excluded)
    _write_jsonl(
        output_dir / "lineage.jsonl",
        build_lineage_rows(candidate, extracted_sections, source_by_id),
    )
    _write_jsonl(output_dir / "rag_output_trec_rag_2026.jsonl", [official_entry])
    _write_json(output_dir / "deepseek_call_summary.json", call_summary)
    _write_json(output_dir / "config.resolved.json", json.loads(json.dumps(config)))

    pre_evaluation_paths = [
        "generation_input_claims.json",
        "facet_extractions.jsonl",
        *[f"facet_extractions/facet_{number:02d}.json" for number in range(1, 11)],
        "deepseek_synthesis.json",
        "candidate_response_generation.json",
        "response_generation.json",
        "generated_response.md",
        "generation_support_audit.jsonl",
        "claim_support_audit.jsonl",
        "excluded_sentences.jsonl",
        "lineage.jsonl",
        "rag_output_trec_rag_2026.jsonl",
        "deepseek_calls.jsonl",
        "deepseek_call_summary.json",
        "config.resolved.json",
    ]
    freeze = write_generation_freeze(output_dir, pre_evaluation_paths)

    # Organizer nuggets are intentionally unavailable until every generation artifact is frozen.
    nuggets = load_nuggets(
        nuggets_path,
        topic_id=str(source_generation["topic_id"]),
        expected_count=int(evaluation_config.get("expected_nugget_count", 50)),
    )
    comparison = evaluate_nuggets(
        nuggets=nuggets,
        generation=candidate,
        client=qwen_client,
        evaluation_config=evaluation_config,
    )
    _write_jsonl(output_dir / "nugget_comparison.jsonl", comparison)
    metrics = compute_evaluation_metrics(comparison, final_audit, response_text=response_text)
    fact_count = sum(len(section["candidate_facts"]) for section in extracted_sections)
    referenced_source_ids = {
        str(source_id)
        for section in extracted_sections
        for fact in section["candidate_facts"]
        for source_id in fact["source_claim_ids"]
    }
    metrics.update(
        {
            "experiment_id": experiment["id"],
            "run_id": experiment["run_id"],
            "topic_id": source_generation["topic_id"],
            "models": {
                "generation": generation_config["model"],
                "audit_and_evaluation": audit_config.get("model_identity", audit_config["model"]),
            },
            "hierarchy": {
                "supported_source_claims": len(source_by_id),
                "source_claims_referenced_by_extractions": len(referenced_source_ids),
                "candidate_facts": fact_count,
                "candidate_sentences": sum(
                    len(section["claims"]) for section in candidate_before_audit["sections"]
                ),
                "submitted_sentences": len(official_entry["answer"]),
            },
            "generation_calls": call_summary,
            "official_submission": {
                "word_count": official_words,
                "sentence_count": len(official_entry["answer"]),
                "reference_count": len(official_entry["references"]),
                "excluded_after_support_audit": len(excluded),
                "maximum_words": 1024,
            },
            "generation_freeze_sha256": _sha256(output_dir / "generation_freeze.json"),
            "frozen_generation_sha256": _sha256(output_dir / "response_generation.json"),
            "frozen_extractions_sha256": _sha256(output_dir / "facet_extractions.jsonl"),
            "frozen_submission_sha256": _sha256(
                output_dir / "rag_output_trec_rag_2026.jsonl"
            ),
        }
    )
    baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    metrics["comparison_to_qwen_baseline"] = _comparison_summary(metrics, baseline)
    _write_json(output_dir / "metrics.json", metrics)
    report = _render_report(metrics=metrics, baseline=baseline, call_summary=call_summary)
    (output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")

    final_artifact_paths = [
        *pre_evaluation_paths,
        "generation_freeze.json",
        "nugget_comparison.jsonl",
        "metrics.json",
        "evaluation_report.md",
        "qwen_calls.jsonl",
    ]
    source_rows = [
        {"path": str(source_generation_path.relative_to(repo_root)).replace("\\", "/"), "sha256": _sha256(source_generation_path)},
        {"path": str(source_audit_path.relative_to(repo_root)).replace("\\", "/"), "sha256": _sha256(source_audit_path)},
        {"path": str(nuggets_path.relative_to(repo_root)).replace("\\", "/"), "sha256": _sha256(nuggets_path)},
        {"path": str(baseline_metrics_path.relative_to(repo_root)).replace("\\", "/"), "sha256": _sha256(baseline_metrics_path)},
    ]
    manifest = {
        "schema_version": "topic213-experiment-manifest-v1",
        "experiment_id": experiment["id"],
        "branch": "codex/deepseek-facet-hierarchical",
        "source_inputs": source_rows,
        "artifacts": _artifact_rows(output_dir, final_artifact_paths),
        "generation_freeze": freeze,
        "deepseek_generation_calls": call_summary,
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )
    checked = validate_manifest_hashes(manifest, artifact_root=output_dir, repo_root=repo_root)
    _write_json(
        output_dir / "manifest_validation.json",
        {"status": "passed", "checked_hashes": checked},
    )
    publish_paths = [*final_artifact_paths, "manifest.yaml", "manifest_validation.json"]
    _publish_report_artifacts(output_dir, report_dir, publish_paths)
    report_manifest = yaml.safe_load((report_dir / "manifest.yaml").read_text(encoding="utf-8"))
    validate_manifest_hashes(report_manifest, artifact_root=report_dir, repo_root=repo_root)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run_from_config(args.config.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
