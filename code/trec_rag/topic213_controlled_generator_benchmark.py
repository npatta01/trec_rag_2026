"""Run a claim-preserving, three-model Topic 213 generation benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import requests
import yaml

from trec_rag.repo_env import find_repo_root, load_repo_env
from trec_rag.topic213_2026_submission import validate_submission_entry
from trec_rag.topic213_deepseek_facet_hierarchical import load_supported_source_claims
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


SCHEMA_VERSION = "topic213-controlled-generator-benchmark-v1"
PROMPT_VERSION = "claim-preserving-42-slot-generation-v1"
FREEZE_VERSION = "controlled-generator-freeze-v1"
EXPECTED_FACET_COUNT = 10
EXPECTED_MODEL_COUNT = 3
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_INLINE_CITATION_RE = re.compile(r"\[[^\]]+\]")
_TRANSIENT_HTTP_STATUSES = {408, 409, 429, 500, 502, 503, 504}


GENERATION_SYSTEM_PROMPT = """You rewrite a frozen ledger of independently supported factual
claims into one official TREC RAG 2026 answer. Return exactly one JSON object with a `sentences`
array. The user supplies an ordered list of claim slots grouped into facets. Return exactly one item
for every slot, in exactly the supplied order. Each item must have exactly two keys: `claim_id` and
`text`, and `claim_id` must match the slot being rewritten.

Each `text` must be exactly one polished, self-contained grammatical sentence. Preserve every
factual detail in that slot's source claim without adding outside knowledge, inference, headings,
introductions, conclusions, citation markers, or meta-commentary. Do not merge slots and do not
split one slot across multiple items. Use the facet word targets to remain balanced and keep the
complete answer within the supplied total word band. Citations are assigned deterministically after
generation, so do not mention document IDs or citations in the text."""


class GeneratorClient(Protocol):
    model: str

    def complete_once(
        self,
        *,
        system_prompt: str,
        payload: Mapping[str, object],
        max_tokens: int,
        temperature: float,
    ) -> "GenerationCompletion": ...


@dataclass(frozen=True)
class GenerationCompletion:
    parsed: dict[str, object]
    raw_content: str
    receipt: dict[str, object]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def semantic_request_sha256(
    *,
    system_prompt: str,
    payload: Mapping[str, object],
    max_tokens: int,
    temperature: float,
) -> str:
    semantic_request = {
        "system_prompt": system_prompt,
        "payload": payload,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    return hashlib.sha256(_canonical_json(semantic_request)).hexdigest()


class SingleCandidateJsonClient:
    """OpenAI-compatible client with one semantic request and transient retries only."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str,
        checkpoint_path: Path,
        call_log_path: Path,
        timeout_seconds: float,
        transport_max_attempts: int,
        request_overrides: Mapping[str, object] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("generation model must be nonempty")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.checkpoint_path = checkpoint_path
        self.call_log_path = call_log_path
        self.timeout_seconds = timeout_seconds
        self.transport_max_attempts = transport_max_attempts
        self.request_overrides = dict(request_overrides or {})
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.call_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, row: Mapping[str, object]) -> None:
        with self.call_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def complete_once(
        self,
        *,
        system_prompt: str,
        payload: Mapping[str, object],
        max_tokens: int,
        temperature: float,
    ) -> GenerationCompletion:
        semantic_hash = semantic_request_sha256(
            system_prompt=system_prompt,
            payload=payload,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if self.checkpoint_path.exists():
            cached = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            parsed = cached.get("parsed")
            raw_content = cached.get("raw_content")
            receipt = cached.get("receipt")
            if not isinstance(parsed, dict) or not isinstance(raw_content, str) or not isinstance(receipt, dict):
                raise ValueError(f"invalid generation checkpoint: {self.checkpoint_path}")
            self._log(
                {
                    "semantic_request_sha256": semantic_hash,
                    "request_sha256": receipt.get("request_sha256"),
                    "cache_hit": True,
                    "network_requests": 0,
                }
            )
            return GenerationCompletion(parsed, raw_content, dict(receipt))

        request_body: dict[str, object] = {
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
        }
        request_body.update(copy.deepcopy(self.request_overrides))
        request_hash = hashlib.sha256(_canonical_json(request_body)).hexdigest()
        url = f"{self.api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(1, self.transport_max_attempts + 1):
            started = time.monotonic()
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=request_body,
                    timeout=(15, self.timeout_seconds),
                )
            except requests.RequestException as exc:
                last_error = exc
                self._log(
                    {
                        "semantic_request_sha256": semantic_hash,
                        "request_sha256": request_hash,
                        "cache_hit": False,
                        "attempt": attempt,
                        "transport_error": str(exc),
                    }
                )
                if attempt == self.transport_max_attempts:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
                continue

            if response.status_code in _TRANSIENT_HTTP_STATUSES:
                last_error = requests.HTTPError(
                    f"transient generation HTTP {response.status_code}", response=response
                )
                self._log(
                    {
                        "semantic_request_sha256": semantic_hash,
                        "request_sha256": request_hash,
                        "cache_hit": False,
                        "attempt": attempt,
                        "http_status": response.status_code,
                        "transient": True,
                    }
                )
                if attempt == self.transport_max_attempts:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
                continue

            if response.status_code >= 400:
                try:
                    provider_error: object = response.json()
                except ValueError:
                    provider_error = {"body": response.text[:2000]}
                self._log(
                    {
                        "semantic_request_sha256": semantic_hash,
                        "request_sha256": request_hash,
                        "cache_hit": False,
                        "attempt": attempt,
                        "http_status": response.status_code,
                        "provider_error": provider_error,
                    }
                )
                raise RuntimeError(
                    f"generation request rejected with HTTP {response.status_code}: "
                    f"{json.dumps(provider_error, ensure_ascii=False)}"
                )

            try:
                envelope = response.json()
                raw_content = envelope["choices"][0]["message"]["content"]
                if not isinstance(raw_content, str) or not raw_content.strip():
                    raise ValueError("generation content is empty")
                parsed = _parse_json_object(raw_content)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._log(
                    {
                        "semantic_request_sha256": semantic_hash,
                        "request_sha256": request_hash,
                        "cache_hit": False,
                        "attempt": attempt,
                        "http_status": response.status_code,
                        "semantic_completion_received": True,
                        "validation_error": str(exc),
                    }
                )
                raise ValueError("single candidate returned invalid JSON; no semantic retry was made") from exc

            receipt = {
                "model": self.model,
                "response_model": envelope.get("model"),
                "provider": envelope.get("provider"),
                "response_id": envelope.get("id"),
                "semantic_request_sha256": semantic_hash,
                "request_sha256": request_hash,
                "raw_content_sha256": hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "semantic_generation_request_count": 1,
                "transport_attempt_count": attempt,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "usage": envelope.get("usage", {}),
                "request_overrides": self.request_overrides,
                "checkpoint_replay": False,
            }
            _write_json(
                self.checkpoint_path,
                {"parsed": parsed, "raw_content": raw_content, "receipt": receipt},
            )
            self._log(
                {
                    "semantic_request_sha256": semantic_hash,
                    "request_sha256": request_hash,
                    "cache_hit": False,
                    "attempt": attempt,
                    "http_status": response.status_code,
                    "elapsed_seconds": receipt["elapsed_seconds"],
                    "usage": receipt["usage"],
                }
            )
            return GenerationCompletion(parsed, raw_content, receipt)

        raise RuntimeError("generation transport failed after identical retries") from last_error


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def _resolve_config_path(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _passage_score(claim_text: str, passage: Mapping[str, object]) -> tuple[float, int, int]:
    claim_terms = set(_WORD_RE.findall(claim_text.casefold()))
    passage_terms = set(_WORD_RE.findall(str(passage.get("text", "")).casefold()))
    overlap = len(claim_terms & passage_terms)
    return (
        overlap / len(claim_terms) if claim_terms else 0.0,
        overlap,
        len(str(passage.get("text", ""))),
    )


def select_frozen_citation_passages(
    claim: Mapping[str, object], *, maximum_citations: int = 1
) -> list[dict[str, object]]:
    passages = claim.get("supporting_passages")
    if not isinstance(passages, list) or not passages or any(
        not isinstance(row, Mapping) for row in passages
    ):
        raise ValueError(f"claim {claim.get('claim_id')} has no supporting passages")
    claim_text = _require_nonempty_string(claim.get("text"), "claim text")
    ranked = sorted(
        enumerate(passages),
        key=lambda item: (*_passage_score(claim_text, item[1]), -item[0]),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    seen_documents: set[str] = set()
    for _, raw_passage in ranked:
        passage = dict(raw_passage)
        document_id = _require_nonempty_string(passage.get("document_id"), "document_id")
        if not document_id.startswith("shard_"):
            raise ValueError(f"non-ClimbMix document ID: {document_id}")
        if document_id in seen_documents:
            continue
        selected.append(passage)
        seen_documents.add(document_id)
        if len(selected) == maximum_citations:
            break
    return selected


def build_frozen_evidence(
    *,
    source_generation: Mapping[str, object],
    support_rows: Sequence[Mapping[str, object]],
    sentence_quotas: Sequence[int],
    facet_word_targets: Sequence[int],
    expected_claim_count: int,
    expected_source_word_count: int,
    minimum_words: int,
    maximum_words: int,
    tokenizer: SpacySentenceTokenizer,
) -> dict[str, object]:
    if len(sentence_quotas) != EXPECTED_FACET_COUNT or len(facet_word_targets) != EXPECTED_FACET_COUNT:
        raise ValueError(f"facet quota arrays must each contain {EXPECTED_FACET_COUNT} values")
    if minimum_words > maximum_words or maximum_words > 1024:
        raise ValueError("invalid controlled word band")
    labels, claims_by_label, _ = load_supported_source_claims(
        source_generation, support_rows, expected_claim_count=expected_claim_count
    )
    facets: list[dict[str, object]] = []
    source_word_count = 0
    position = 0
    for facet_number, (label, sentence_quota, word_target) in enumerate(
        zip(labels, sentence_quotas, facet_word_targets, strict=True), 1
    ):
        claims = claims_by_label[label]
        if len(claims) != sentence_quota:
            raise ValueError(
                f"facet {facet_number} quota is {sentence_quota}, but it has {len(claims)} supported claims"
            )
        frozen_claims: list[dict[str, object]] = []
        for claim in claims:
            text = _require_nonempty_string(claim.get("text"), "source claim text")
            if len(tokenizer.tokenize(text)) != 1:
                raise ValueError(f"source claim is not exactly one sentence: {claim['claim_id']}")
            position += 1
            source_word_count += len(text.split())
            selected = select_frozen_citation_passages(claim)
            frozen_claims.append(
                {
                    "position": position,
                    "claim_id": claim["claim_id"],
                    "text": text,
                    "sub_narrative": label,
                    "document_ids": [str(row["document_id"]) for row in selected],
                    "selected_citation_passages": selected,
                    "available_supporting_passage_count": len(claim["supporting_passages"]),
                }
            )
        facets.append(
            {
                "facet_number": facet_number,
                "sub_narrative": label,
                "sentence_quota": sentence_quota,
                "word_target": word_target,
                "source_claims": frozen_claims,
            }
        )
    if source_word_count != expected_source_word_count:
        raise ValueError(
            f"expected {expected_source_word_count} supported source words, found {source_word_count}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "topic_id": str(source_generation["topic_id"]),
        "narrative": source_generation["narrative"],
        "source_experiment_id": source_generation["experiment_id"],
        "organizer_nuggets_available": False,
        "supported_source_claim_count": expected_claim_count,
        "source_claim_word_count": source_word_count,
        "exact_total_sentence_count": sum(sentence_quotas),
        "word_band": {"minimum": minimum_words, "maximum": maximum_words},
        "facet_word_target_total": sum(facet_word_targets),
        "input_accounting": source_generation.get("input_accounting", {}),
        "source_metadata": source_generation.get("source_metadata", {}),
        "facets": facets,
    }


def build_generation_payload(frozen: Mapping[str, object]) -> dict[str, object]:
    return {
        "task": "Rewrite every frozen supported claim into its matching output slot.",
        "narrative": frozen["narrative"],
        "exact_total_sentence_count": frozen["exact_total_sentence_count"],
        "total_answer_word_band": frozen["word_band"],
        "output_contract": {
            "root_key": "sentences",
            "item_keys": ["claim_id", "text"],
            "claim_order_must_match_input": True,
            "one_spacy_sentence_per_item": True,
            "one_item_per_claim": True,
        },
        "facets": [
            {
                "facet_number": facet["facet_number"],
                "sub_narrative": facet["sub_narrative"],
                "exact_sentence_count": facet["sentence_quota"],
                "word_target": facet["word_target"],
                "claims": [
                    {
                        "position": claim["position"],
                        "claim_id": claim["claim_id"],
                        "text": claim["text"],
                    }
                    for claim in facet["source_claims"]
                ],
            }
            for facet in frozen["facets"]
        ],
    }


def build_response_schema(expected_sentence_count: int) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["sentences"],
        "properties": {
            "sentences": {
                "type": "array",
                "minItems": expected_sentence_count,
                "maxItems": expected_sentence_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["claim_id", "text"],
                    "properties": {
                        "claim_id": {"type": "string"},
                        "text": {"type": "string"},
                    },
                },
            }
        },
    }


def _flatten_frozen_claims(frozen: Mapping[str, object]) -> list[dict[str, object]]:
    return [dict(claim) for facet in frozen["facets"] for claim in facet["source_claims"]]


def validate_generation_response(
    value: Mapping[str, object],
    *,
    frozen: Mapping[str, object],
    tokenizer: SpacySentenceTokenizer,
) -> list[dict[str, object]]:
    raw_sentences = value.get("sentences")
    expected_claims = _flatten_frozen_claims(frozen)
    if not isinstance(raw_sentences, list) or len(raw_sentences) != len(expected_claims):
        raise ValueError(f"response must contain exactly {len(expected_claims)} sentences")
    normalized: list[dict[str, object]] = []
    seen_texts: set[str] = set()
    total_words = 0
    for index, (raw_item, expected) in enumerate(
        zip(raw_sentences, expected_claims, strict=True), 1
    ):
        if not isinstance(raw_item, Mapping) or set(raw_item) != {"claim_id", "text"}:
            raise ValueError(f"sentence {index} must contain exactly claim_id and text")
        claim_id = _require_nonempty_string(raw_item.get("claim_id"), "claim_id")
        if claim_id != expected["claim_id"]:
            raise ValueError(f"claim order mismatch at sentence {index}")
        text = _require_nonempty_string(raw_item.get("text"), "sentence text")
        if _INLINE_CITATION_RE.search(text):
            raise ValueError(f"sentence {index} contains an inline citation marker")
        if len(tokenizer.tokenize(text)) != 1:
            raise ValueError(f"sentence {index} is not exactly one SpaCy sentence")
        normalized_text = " ".join(text.casefold().split())
        if normalized_text in seen_texts:
            raise ValueError(f"sentence {index} duplicates an earlier sentence")
        seen_texts.add(normalized_text)
        total_words += len(text.split())
        normalized.append(
            {
                "position": index,
                "claim_id": claim_id,
                "sub_narrative": expected["sub_narrative"],
                "text": text,
            }
        )
    minimum = int(frozen["word_band"]["minimum"])
    maximum = int(frozen["word_band"]["maximum"])
    if not minimum <= total_words <= maximum:
        raise ValueError(
            f"candidate has {total_words} words; controlled band is {minimum}-{maximum}"
        )
    return normalized


def build_candidate_generation(
    *,
    source_generation: Mapping[str, object],
    frozen: Mapping[str, object],
    normalized_sentences: Sequence[Mapping[str, object]],
    model_key: str,
    model_identity: str,
    receipt: Mapping[str, object],
    experiment_id: str,
    run_id: str,
) -> dict[str, object]:
    frozen_by_id = {str(row["claim_id"]): row for row in _flatten_frozen_claims(frozen)}
    sentences_by_label: dict[str, list[Mapping[str, object]]] = {
        str(facet["sub_narrative"]): [] for facet in frozen["facets"]
    }
    for sentence in normalized_sentences:
        sentences_by_label[str(sentence["sub_narrative"])].append(sentence)
    sections: list[dict[str, object]] = []
    sentence_number = 0
    for facet in frozen["facets"]:
        label = str(facet["sub_narrative"])
        claims: list[dict[str, object]] = []
        for sentence in sentences_by_label[label]:
            sentence_number += 1
            source_id = str(sentence["claim_id"])
            source = frozen_by_id[source_id]
            claims.append(
                {
                    "claim_id": f"S{sentence_number:03d}",
                    "text": sentence["text"],
                    "evidence_claim_ids": [source_id],
                    "document_ids": list(source["document_ids"]),
                    "supporting_passages": copy.deepcopy(source["selected_citation_passages"]),
                }
            )
        sections.append({"sub_narrative": label, "claims": claims})
    evidence_ledger = [
        {
            "claim_id": row["claim_id"],
            "sub_narrative": row["sub_narrative"],
            "text": row["text"],
            "document_ids": list(row["document_ids"]),
            "supporting_passages": copy.deepcopy(row["selected_citation_passages"]),
        }
        for row in _flatten_frozen_claims(frozen)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "topic_id": str(source_generation["topic_id"]),
        "narrative": source_generation["narrative"],
        "model_key": model_key,
        "model": model_identity,
        "temperature": 0.0,
        "context_policy": {
            "kind": "one-to-one-rewrite-of-frozen-supported-claim-ledger",
            "source_experiment_id": source_generation["experiment_id"],
            "supported_source_claim_count": frozen["supported_source_claim_count"],
            "exact_total_sentence_count": frozen["exact_total_sentence_count"],
            "word_band": frozen["word_band"],
            "semantic_request_sha256": receipt["semantic_request_sha256"],
            "semantic_generation_request_count": 1,
            "llm_rewrite_or_repair_requests": 0,
            "organizer_nuggets_available": False,
        },
        "input_accounting": frozen.get("input_accounting", {}),
        "source_metadata": frozen.get("source_metadata", {}),
        "prompts": {"generation_system": GENERATION_SYSTEM_PROMPT},
        "evidence_ledger": evidence_ledger,
        "sections": sections,
    }


def filter_supported_generation(
    candidate: Mapping[str, object], audit_rows: Sequence[Mapping[str, object]]
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    final = copy.deepcopy(candidate)
    expected_ids = {
        str(claim["claim_id"])
        for section in final["sections"]
        for claim in section["claims"]
    }
    audit_by_id = {str(row["claim_id"]): dict(row) for row in audit_rows}
    if set(audit_by_id) != expected_ids:
        raise ValueError("support audit must cover every candidate sentence exactly once")
    kept: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    retained_source_ids: set[str] = set()
    for section in final["sections"]:
        kept_claims: list[dict[str, object]] = []
        for claim in section["claims"]:
            audit = audit_by_id[str(claim["claim_id"])]
            if audit.get("status") == "supported":
                kept_claims.append(claim)
                kept.append(audit)
                retained_source_ids.update(map(str, claim["evidence_claim_ids"]))
            else:
                excluded.append({**dict(claim), "support_audit": audit})
        section["claims"] = kept_claims
    final["evidence_ledger"] = [
        row for row in final["evidence_ledger"] if str(row["claim_id"]) in retained_source_ids
    ]
    final["support_filter"] = {
        "candidate_sentence_count": len(expected_ids),
        "submitted_sentence_count": len(kept),
        "excluded_sentence_count": len(excluded),
        "rewrite_or_repair_attempted": False,
    }
    return final, kept, excluded


def build_official_entry(
    *,
    generation: Mapping[str, object],
    team_id: str,
    run_desc: str,
) -> dict[str, object]:
    references: list[str] = []
    answer: list[dict[str, object]] = []
    for section in generation["sections"]:
        for claim in section["claims"]:
            citations = list(claim["document_ids"])
            for citation in citations:
                if citation not in references:
                    references.append(citation)
            answer.append({"text": claim["text"], "citations": citations})
    return {
        "metadata": {
            "team_id": team_id,
            "narrative_id": str(generation["topic_id"]),
            "narrative": generation["narrative"],
            "run_id": generation["run_id"],
            "run_desc": run_desc,
            "generator": generation["model"],
            "source_experiment_id": generation["context_policy"]["source_experiment_id"],
        },
        "references": references,
        "answer": answer,
    }


def validate_official_entry(
    entry: Mapping[str, object], *, tokenizer: SpacySentenceTokenizer
) -> int:
    word_count = validate_submission_entry(entry, maximum_words=1024)
    first_use: list[str] = []
    for index, item in enumerate(entry["answer"]):
        if len(tokenizer.tokenize(str(item["text"]))) != 1:
            raise ValueError(f"answer[{index}] is not exactly one sentence")
        citations = list(item["citations"])
        if len(citations) != len(set(citations)):
            raise ValueError(f"answer[{index}] has duplicate citations")
        if any(not str(citation).startswith("shard_") for citation in citations):
            raise ValueError(f"answer[{index}] has a non-ClimbMix citation")
        for citation in citations:
            if citation not in first_use:
                first_use.append(citation)
    if list(entry["references"]) != first_use:
        raise ValueError("references must be unique and ordered by first citation use")
    return word_count


def render_markdown(generation: Mapping[str, object]) -> str:
    lines = ["# Topic 213: Korean War", ""]
    for section in generation["sections"]:
        label = str(section["sub_narrative"]).strip().strip('"')
        if label.startswith("New: "):
            label = label[5:]
        lines.extend([f"## {label}", ""])
        if not section["claims"]:
            lines.extend(["No sentence survived the independent support audit.", ""])
        for claim in section["claims"]:
            lines.extend([f"{claim['text']} [{'; '.join(claim['document_ids'])}]", ""])
    return "\n".join(lines).rstrip() + "\n"


def _artifact_row(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_model_freeze(model_dir: Path, artifact_names: Sequence[str]) -> dict[str, object]:
    freeze = {
        "schema_version": FREEZE_VERSION,
        "organizer_nuggets_read": False,
        "artifacts": [_artifact_row(model_dir, name) for name in artifact_names],
    }
    _write_json(model_dir / "generation_freeze.json", freeze)
    return freeze


def _write_aggregate_freeze(
    *, output_dir: Path, shared_names: Sequence[str], model_keys: Sequence[str]
) -> dict[str, object]:
    models = []
    for model_key in model_keys:
        freeze_path = output_dir / model_key / "generation_freeze.json"
        if not freeze_path.is_file():
            raise ValueError(f"cannot freeze missing model generation: {model_key}")
        models.append(
            {
                "model_key": model_key,
                "generation_freeze_sha256": _sha256(freeze_path),
            }
        )
    freeze = {
        "schema_version": "controlled-benchmark-aggregate-freeze-v1",
        "organizer_nuggets_read": False,
        "shared_artifacts": [_artifact_row(output_dir, name) for name in shared_names],
        "models": models,
    }
    _write_json(output_dir / "benchmark_generation_freeze.json", freeze)
    return freeze


def _usage_cost(receipt: Mapping[str, object]) -> float:
    usage = receipt.get("usage", {})
    if not isinstance(usage, Mapping):
        return 0.0
    value = usage.get("cost", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def choose_winner(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot choose a winner from no results")
    return dict(
        max(
            rows,
            key=lambda row: (
                float(row["strict_coverage"]),
                float(row["vital_strict_coverage"]),
                float(row["partial_credit_coverage"]),
                -int(row["unsupported_submitted"]),
                -float(row.get("cost", math.inf)),
            ),
        )
    )


def validate_manifest_hashes(
    manifest: Mapping[str, object], *, artifact_root: Path
) -> int:
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise ValueError("manifest artifacts must be an array")
    checked = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest artifact row must be an object")
        relative = _require_nonempty_string(row.get("path"), "manifest path")
        expected = _require_nonempty_string(row.get("sha256"), "manifest sha256")
        if _sha256(artifact_root / relative) != expected:
            raise ValueError(f"manifest hash mismatch: {relative}")
        checked += 1
    return checked


def _make_judge_client(
    *, output_dir: Path, config: Mapping[str, object], purpose: str
) -> OpenAICompatibleJsonClient:
    checkpoint_dir_names = {
        "support_audit": "support_audit_checkpoints",
        "nugget_evaluation": "nugget_cache",
    }
    checkpoint_dir_name = checkpoint_dir_names.get(purpose, f"{purpose}_cache")
    api_base = os.environ.get(
        str(config.get("api_base_env", "LITELLM_BASE_URL")),
        str(config.get("api_base", "http://localhost:4000/v1")),
    )
    api_key = os.environ.get(
        str(config.get("api_key_env", "LITELLM_API_KEY")),
        str(config.get("api_key", "none")),
    )
    return OpenAICompatibleJsonClient(
        api_base=api_base,
        model=str(config.get("model", "qwen-local")),
        api_key=api_key,
        checkpoint_dir=output_dir / checkpoint_dir_name,
        call_log_path=output_dir / f"{purpose}_calls.jsonl",
        timeout_seconds=float(config.get("timeout_seconds", 240)),
        max_attempts=int(config.get("http_max_attempts", 4)),
    )


def _model_client(
    *, model_dir: Path, config: Mapping[str, object]
) -> SingleCandidateJsonClient:
    api_base = os.environ.get(
        str(config.get("api_base_env", "")), str(config["api_base"])
    )
    api_key = os.environ.get(
        str(config.get("api_key_env", "")), str(config.get("api_key", ""))
    )
    return SingleCandidateJsonClient(
        api_base=api_base,
        model=str(config["model"]),
        api_key=api_key,
        checkpoint_path=model_dir / "generation_checkpoint.json",
        call_log_path=model_dir / "generation_calls.jsonl",
        timeout_seconds=float(config.get("timeout_seconds", 300)),
        transport_max_attempts=int(config.get("transport_max_attempts", 3)),
        request_overrides=config.get("request_overrides", {}),
    )


def _per_model_report(metrics: Mapping[str, object]) -> str:
    all_nuggets = metrics["nuggets"]["all"]
    vital = metrics["nuggets"]["vital"]
    official = metrics["official_submission"]
    retention = metrics["claim_retention"]
    return f"""# {metrics['model']['display_name']}: controlled Topic 213 result

This candidate used the same frozen 42-claim ledger, exact facet quotas, shared prompt, and shared
900-1,000-word candidate band as the other generators. Nugget evaluation was loaded only after all
three audited organizer submissions were frozen.

## Result

- Strict nugget coverage: {all_nuggets['strict_coverage']:.3f}
- Partial-credit coverage: {all_nuggets['partial_credit_coverage']:.3f}
- Vital strict coverage: {vital['strict_coverage']:.3f}
- Candidate claim retention: {retention['candidate_source_claims']}/42
- Submitted claim retention: {retention['submitted_source_claims']}/42
- Candidate words: {official['candidate_word_count']}
- Submitted words: {official['word_count']} / 1024
- Submitted sentences: {official['sentence_count']}
- Excluded after support audit: {official['excluded_after_support_audit']}
- Citation coverage: {metrics['answer_claims']['citation_coverage']:.3f}
- Unsupported submitted sentences: {metrics['answer_claims']['unsupported_claim_count']}
"""


def _comparison_report(
    *,
    rows: Sequence[Mapping[str, object]],
    winner: Mapping[str, object],
    original_baseline: Mapping[str, object],
    compliant_baseline: Mapping[str, object],
) -> str:
    lines = [
        "# Topic 213 controlled generator benchmark",
        "",
        "All three generators received the same frozen 42-claim evidence ledger, exact per-facet",
        "sentence quotas, prompt, response contract, and 900-1,000-word candidate target. Citation",
        "selection and local-Qwen support auditing were identical and model-blind.",
        "",
        "## Nugget coverage",
        "",
        "| Run | Strict | Partial | Vital strict | Candidate words | Submitted words | Retained claims | Excluded | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Original local Qwen, unconstrained | {original_baseline['nuggets']['all']['strict_coverage']:.3f} | "
            f"{original_baseline['nuggets']['all']['partial_credit_coverage']:.3f} | "
            f"{original_baseline['nuggets']['vital']['strict_coverage']:.3f} | n/a | "
            f"{original_baseline['response']['word_count']} | {original_baseline['answer_claims']['total']} | "
            f"{original_baseline['answer_claims']['unsupported_claim_count']} unsupported | n/a |"
        ),
        (
            f"| Existing organizer-compliant Qwen | {compliant_baseline['nuggets']['all']['strict_coverage']:.3f} | "
            f"{compliant_baseline['nuggets']['all']['partial_credit_coverage']:.3f} | "
            f"{compliant_baseline['nuggets']['vital']['strict_coverage']:.3f} | n/a | "
            f"{compliant_baseline['official_submission']['word_count']} | "
            f"{compliant_baseline['answer_claims']['total']} | "
            f"{compliant_baseline['official_submission'].get('excluded_after_support_audit', 0)} | n/a |"
        ),
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['strict_coverage']:.3f} | "
            f"{row['partial_credit_coverage']:.3f} | {row['vital_strict_coverage']:.3f} | "
            f"{row['candidate_words']} | {row['submitted_words']} | {row['submitted_claims']}/42 | "
            f"{row['excluded']} | ${row['cost']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"The selected generator is **{winner['display_name']}**. Selection is lexicographic by strict coverage,",
            "then vital strict coverage, partial-credit coverage, unsupported submitted sentences, and cost.",
            "",
            "## Interpretation",
            "",
            "This experiment isolates the generator over one development topic; it does not establish held-out",
            "generalization. Local Qwen also serves as the blinded support and nugget judge, so evaluator-style bias",
            "remains a limitation even though model identity is never included in judge payloads.",
        ]
    )
    return "\n".join(lines) + "\n"


def _copy_report_artifacts(
    *, output_dir: Path, report_dir: Path, relative_paths: Sequence[str]
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for relative in relative_paths:
        source = output_dir / relative
        destination = report_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run_from_config(
    config_path: Path,
    *,
    generator_clients: Mapping[str, GeneratorClient] | None = None,
) -> Path:
    config_path = _resolve_config_path(config_path)
    repo_root = find_repo_root(config_path.parent)
    load_repo_env(repo_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    experiment = config["experiment"]
    inputs = config["inputs"]
    shared = config["shared_generation"]
    audit_config = config["audit"]
    evaluation_config = config["evaluation"]
    model_configs = config["models"]
    if not isinstance(model_configs, list) or len(model_configs) != EXPECTED_MODEL_COUNT:
        raise ValueError(f"models must contain exactly {EXPECTED_MODEL_COUNT} entries")

    output_dir = repo_root / str(experiment["output_dir"])
    report_dir = repo_root / str(experiment["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_generation_path = repo_root / str(inputs["source_generation"])
    source_audit_path = repo_root / str(inputs["source_support_audit"])
    source_generation = json.loads(source_generation_path.read_text(encoding="utf-8"))
    source_audit = _read_jsonl(source_audit_path)
    tokenizer = SpacySentenceTokenizer()
    frozen = build_frozen_evidence(
        source_generation=source_generation,
        support_rows=source_audit,
        sentence_quotas=list(map(int, shared["sentence_quotas"])),
        facet_word_targets=list(map(int, shared["facet_word_targets"])),
        expected_claim_count=int(shared["expected_supported_claim_count"]),
        expected_source_word_count=int(shared["expected_source_word_count"]),
        minimum_words=int(shared["minimum_words"]),
        maximum_words=int(shared["maximum_words"]),
        tokenizer=tokenizer,
    )
    payload = build_generation_payload(frozen)
    response_schema = build_response_schema(int(frozen["exact_total_sentence_count"]))
    max_tokens = int(shared["max_tokens"])
    temperature = float(shared["temperature"])
    semantic_hash = semantic_request_sha256(
        system_prompt=GENERATION_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    _write_json(output_dir / "frozen_evidence_ledger.json", frozen)
    _write_json(output_dir / "frozen_generation_payload.json", payload)
    _write_json(output_dir / "response_schema.json", response_schema)
    (output_dir / "generation_system_prompt.txt").write_text(
        GENERATION_SYSTEM_PROMPT + "\n", encoding="utf-8"
    )
    _write_json(
        output_dir / "semantic_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "semantic_request_sha256": semantic_hash,
            "system_prompt_sha256": hashlib.sha256(GENERATION_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "payload_sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
            "response_schema_sha256": hashlib.sha256(_canonical_json(response_schema)).hexdigest(),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model_identity_excluded_from_semantic_hash": True,
        },
    )
    shutil.copy2(config_path, output_dir / "config.yaml")

    successful: dict[str, dict[str, object]] = {}
    model_keys: list[str] = []
    supplied_clients = dict(generator_clients or {})
    for model_config in model_configs:
        if not isinstance(model_config, Mapping):
            raise ValueError("model config must be an object")
        model_key = str(model_config["key"])
        if model_key in model_keys:
            raise ValueError(f"duplicate model key: {model_key}")
        model_keys.append(model_key)
        model_dir = output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        client = supplied_clients.get(model_key) or _model_client(
            model_dir=model_dir, config=model_config
        )
        completion = client.complete_once(
            system_prompt=GENERATION_SYSTEM_PROMPT,
            payload=payload,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if completion.receipt.get("semantic_request_sha256") != semantic_hash:
            raise ValueError(f"semantic request hash mismatch for {model_key}")
        normalized = validate_generation_response(
            completion.parsed, frozen=frozen, tokenizer=tokenizer
        )
        candidate = build_candidate_generation(
            source_generation=source_generation,
            frozen=frozen,
            normalized_sentences=normalized,
            model_key=model_key,
            model_identity=str(model_config["model_identity"]),
            receipt=completion.receipt,
            experiment_id=str(experiment["id"]),
            run_id=f"{experiment['run_id']}-{model_key}",
        )
        judge = _make_judge_client(
            output_dir=model_dir, config=audit_config, purpose="support_audit"
        )
        audit_rows: list[dict[str, object]] = []
        for section in candidate["sections"]:
            audit_rows.extend(
                _audit_section(
                    judge,
                    sub_narrative=str(section["sub_narrative"]),
                    claims=section["claims"],
                    max_tokens=int(audit_config.get("max_tokens", 350)),
                    temperature=float(audit_config.get("temperature", 0.0)),
                    validation_attempts=int(audit_config.get("validation_attempts", 3)),
                )
            )
        final, kept_audits, excluded = filter_supported_generation(candidate, audit_rows)
        official = build_official_entry(
            generation=final,
            team_id=str(experiment["team_id"]),
            run_desc=f"{experiment['run_desc']} Generator: {model_config['model_identity']}.",
        )
        official_words = validate_official_entry(official, tokenizer=tokenizer)
        markdown = render_markdown(final)

        _write_json(model_dir / "raw_generation.json", completion.parsed)
        (model_dir / "raw_generation.txt").write_text(
            completion.raw_content, encoding="utf-8"
        )
        _write_json(model_dir / "generation_receipt.json", completion.receipt)
        _write_json(model_dir / "response_generation.candidate.json", candidate)
        _write_json(model_dir / "response_generation.json", final)
        _write_jsonl(model_dir / "generation_support_audit.jsonl", audit_rows)
        _write_jsonl(model_dir / "claim_support_audit.jsonl", kept_audits)
        _write_jsonl(model_dir / "excluded_sentences.jsonl", excluded)
        _write_jsonl(model_dir / "rag_output_trec_rag_2026.jsonl", [official])
        (model_dir / "generated_response.md").write_text(markdown, encoding="utf-8")
        lineage = [
            {
                "sentence_claim_id": claim["claim_id"],
                "source_claim_id": claim["evidence_claim_ids"][0],
                "sub_narrative": section["sub_narrative"],
                "document_ids": claim["document_ids"],
                "submitted": any(
                    claim["claim_id"] == kept["claim_id"] for kept in kept_audits
                ),
            }
            for section in candidate["sections"]
            for claim in section["claims"]
        ]
        _write_jsonl(model_dir / "lineage.jsonl", lineage)
        frozen_names = [
            "raw_generation.json",
            "raw_generation.txt",
            "generation_receipt.json",
            "response_generation.candidate.json",
            "response_generation.json",
            "generation_support_audit.jsonl",
            "claim_support_audit.jsonl",
            "excluded_sentences.jsonl",
            "rag_output_trec_rag_2026.jsonl",
            "generated_response.md",
            "lineage.jsonl",
        ]
        model_freeze = _write_model_freeze(model_dir, frozen_names)
        successful[model_key] = {
            "config": dict(model_config),
            "candidate": candidate,
            "final": final,
            "official": official,
            "official_words": official_words,
            "candidate_words": sum(
                len(str(claim["text"]).split())
                for section in candidate["sections"]
                for claim in section["claims"]
            ),
            "kept_audits": kept_audits,
            "excluded": excluded,
            "markdown": markdown,
            "receipt": dict(completion.receipt),
            "model_freeze": model_freeze,
        }

    shared_freeze_names = [
        "config.yaml",
        "frozen_evidence_ledger.json",
        "frozen_generation_payload.json",
        "response_schema.json",
        "generation_system_prompt.txt",
        "semantic_contract.json",
    ]
    aggregate_freeze = _write_aggregate_freeze(
        output_dir=output_dir, shared_names=shared_freeze_names, model_keys=model_keys
    )
    aggregate_freeze_sha256 = _sha256(output_dir / "benchmark_generation_freeze.json")

    nuggets_path = repo_root / str(inputs["nuggets"])
    nuggets = load_nuggets(
        nuggets_path,
        topic_id=str(frozen["topic_id"]),
        expected_count=int(evaluation_config.get("expected_nugget_count", 50)),
    )
    original_baseline_path = repo_root / str(inputs["original_baseline_metrics"])
    compliant_baseline_path = repo_root / str(inputs["compliant_baseline_metrics"])
    original_baseline = json.loads(original_baseline_path.read_text(encoding="utf-8"))
    compliant_baseline = json.loads(compliant_baseline_path.read_text(encoding="utf-8"))

    comparison_rows: list[dict[str, object]] = []
    model_metric_paths: list[str] = []
    for model_key in model_keys:
        state = successful[model_key]
        model_dir = output_dir / model_key
        evaluator = _make_judge_client(
            output_dir=model_dir, config=audit_config, purpose="nugget_evaluation"
        )
        comparison = evaluate_nuggets(
            nuggets=nuggets,
            generation=state["final"],
            client=evaluator,
            evaluation_config=evaluation_config,
        )
        metrics = compute_evaluation_metrics(
            comparison,
            state["kept_audits"],
            response_text=state["markdown"],
        )
        official = state["official"]
        submitted_source_ids = {
            source_id
            for section in state["final"]["sections"]
            for claim in section["claims"]
            for source_id in claim["evidence_claim_ids"]
        }
        metrics.update(
            {
                "experiment_id": experiment["id"],
                "run_id": state["final"]["run_id"],
                "topic_id": frozen["topic_id"],
                "model": {
                    "key": model_key,
                    "display_name": state["config"]["display_name"],
                    "identity": state["config"]["model_identity"],
                    "provider": state["config"]["provider"],
                },
                "semantic_request_sha256": semantic_hash,
                "aggregate_generation_freeze_sha256": aggregate_freeze_sha256,
                "nuggets_loaded_after_all_generations_frozen": True,
                "claim_retention": {
                    "candidate_source_claims": int(frozen["supported_source_claim_count"]),
                    "submitted_source_claims": len(submitted_source_ids),
                    "retention_rate": len(submitted_source_ids)
                    / int(frozen["supported_source_claim_count"]),
                },
                "official_submission": {
                    "candidate_sentence_count": int(frozen["exact_total_sentence_count"]),
                    "candidate_word_count": state["candidate_words"],
                    "sentence_count": len(official["answer"]),
                    "word_count": state["official_words"],
                    "reference_count": len(official["references"]),
                    "excluded_after_support_audit": len(state["excluded"]),
                    "maximum_words": 1024,
                },
                "generation": {
                    "semantic_generation_request_count": 1,
                    "semantic_repair_request_count": 0,
                    "transport_attempt_count": state["receipt"].get(
                        "transport_attempt_count", 0
                    ),
                    "elapsed_seconds": state["receipt"].get("elapsed_seconds", 0),
                    "usage": state["receipt"].get("usage", {}),
                    "cost": _usage_cost(state["receipt"]),
                },
            }
        )
        _write_jsonl(model_dir / "nugget_comparison.jsonl", comparison)
        _write_json(model_dir / "metrics.json", metrics)
        (model_dir / "evaluation_report.md").write_text(
            _per_model_report(metrics), encoding="utf-8"
        )
        model_metric_paths.extend(
            [
                f"{model_key}/nugget_comparison.jsonl",
                f"{model_key}/metrics.json",
                f"{model_key}/evaluation_report.md",
            ]
        )
        comparison_rows.append(
            {
                "model_key": model_key,
                "display_name": state["config"]["display_name"],
                "strict_coverage": metrics["nuggets"]["all"]["strict_coverage"],
                "partial_credit_coverage": metrics["nuggets"]["all"][
                    "partial_credit_coverage"
                ],
                "vital_strict_coverage": metrics["nuggets"]["vital"][
                    "strict_coverage"
                ],
                "candidate_words": state["candidate_words"],
                "submitted_words": state["official_words"],
                "submitted_claims": len(submitted_source_ids),
                "excluded": len(state["excluded"]),
                "unsupported_submitted": metrics["answer_claims"][
                    "unsupported_claim_count"
                ],
                "citation_coverage": metrics["answer_claims"]["citation_coverage"],
                "cost": _usage_cost(state["receipt"]),
                "elapsed_seconds": state["receipt"].get("elapsed_seconds", 0),
            }
        )

    winner = choose_winner(comparison_rows)
    comparison_metrics = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment["id"],
        "topic_id": frozen["topic_id"],
        "semantic_request_sha256": semantic_hash,
        "aggregate_generation_freeze_sha256": aggregate_freeze_sha256,
        "selection_policy": [
            "strict_coverage_desc",
            "vital_strict_coverage_desc",
            "partial_credit_coverage_desc",
            "unsupported_submitted_asc",
            "cost_asc",
        ],
        "winner": winner,
        "models": comparison_rows,
        "baselines": {
            "original_qwen": original_baseline,
            "organizer_compliant_qwen": compliant_baseline,
        },
    }
    _write_json(output_dir / "comparison_metrics.json", comparison_metrics)
    (output_dir / "comparison_report.md").write_text(
        _comparison_report(
            rows=comparison_rows,
            winner=winner,
            original_baseline=original_baseline,
            compliant_baseline=compliant_baseline,
        ),
        encoding="utf-8",
    )

    frozen_model_paths = [
        f"{model_key}/{name}"
        for model_key in model_keys
        for name in [
            "raw_generation.json",
            "raw_generation.txt",
            "generation_receipt.json",
            "response_generation.candidate.json",
            "response_generation.json",
            "generation_support_audit.jsonl",
            "claim_support_audit.jsonl",
            "excluded_sentences.jsonl",
            "rag_output_trec_rag_2026.jsonl",
            "generated_response.md",
            "lineage.jsonl",
            "generation_freeze.json",
        ]
    ]
    publish_paths = [
        *shared_freeze_names,
        "benchmark_generation_freeze.json",
        *frozen_model_paths,
        *model_metric_paths,
        "comparison_metrics.json",
        "comparison_report.md",
    ]
    manifest = {
        "schema_version": "controlled-generator-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment["id"],
        "topic_id": frozen["topic_id"],
        "semantic_request_sha256": semantic_hash,
        "source_inputs": [
            {"path": str(path.relative_to(repo_root)).replace("\\", "/"), "sha256": _sha256(path)}
            for path in [
                config_path,
                source_generation_path,
                source_audit_path,
                nuggets_path,
                original_baseline_path,
                compliant_baseline_path,
            ]
        ],
        "models": [
            {
                "key": key,
                "identity": successful[key]["config"]["model_identity"],
                "provider": successful[key]["config"]["provider"],
                "generation_receipt_sha256": _sha256(
                    output_dir / key / "generation_receipt.json"
                ),
            }
            for key in model_keys
        ],
        "winner": winner,
        "artifacts": [_artifact_row(output_dir, relative) for relative in publish_paths],
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    publish_paths.append("manifest.yaml")
    _copy_report_artifacts(
        output_dir=output_dir, report_dir=report_dir, relative_paths=publish_paths
    )
    validate_manifest_hashes(manifest, artifact_root=report_dir)
    return report_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report_dir = run_from_config(args.config)
    print(f"Wrote controlled benchmark report to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
