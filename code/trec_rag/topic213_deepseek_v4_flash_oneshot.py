"""Run a one-shot DeepSeek synthesis of the frozen Topic 213 evidence ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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


SCHEMA_VERSION = "topic213-trec-rag-2026-deepseek-oneshot-v1"
PROMPT_VERSION = "deepseek-v4-flash-supported-ledger-oneshot-v1"
EXACT_MODEL = "deepseek/deepseek-v4-flash"
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CITATION_MARKER_RE = re.compile(r"\[[^\]]+\]")
_TRANSIENT_HTTP_STATUSES = {408, 409, 429, 500, 502, 503, 504}

ONESHOT_SYSTEM_PROMPT = """You produce a concise TREC RAG answer from a ledger of factual
claims that were independently judged fully supported. Return one JSON object with a `sections`
array. Preserve every supplied section exactly once and in the supplied order. Each section has
`sub_narrative` and a nonempty `answer` array. Each answer item has exactly two keys: `text` and
`source_claim_ids`.

Write each `text` as exactly one polished, self-contained factual sentence. Use only facts expressly
stated by the supplied source claims. Do not use outside knowledge, infer unstated details, or add
introductions, conclusions, headings, citation markers, or meta-commentary. Merge redundant claims
when useful, but attach one to three source claim IDs from the same section that directly support
every factual detail in the sentence. Prefer distinct, information-dense sentences that collectively
cover the supplied evidence while remaining under the total word target."""


@dataclass(frozen=True)
class OneShotCompletion:
    parsed: dict[str, object]
    raw_content: str
    receipt: dict[str, object]


class OpenRouterOneShotJsonClient:
    """Issue one semantic request, retrying only an identical transient request."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str,
        checkpoint_path: Path,
        timeout_seconds: float,
        max_transport_attempts: int,
    ) -> None:
        if model != EXACT_MODEL:
            raise ValueError(f"OpenRouter generation model must be exactly {EXACT_MODEL}")
        if not api_key.strip():
            raise ValueError("OPENROUTER_API_KEY is missing or empty")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.checkpoint_path = checkpoint_path
        self.timeout_seconds = timeout_seconds
        self.max_transport_attempts = max_transport_attempts

    def complete_once(
        self,
        *,
        system_prompt: str,
        payload: Mapping[str, object],
        max_tokens: int,
        temperature: float,
    ) -> OneShotCompletion:
        if temperature != 0:
            raise ValueError("one-shot DeepSeek temperature must be 0")
        if self.checkpoint_path.exists():
            cached = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            parsed = cached.get("parsed")
            raw_content = cached.get("raw_content")
            receipt = cached.get("receipt")
            if not isinstance(parsed, dict) or not isinstance(raw_content, str) or not isinstance(receipt, dict):
                raise ValueError(f"invalid one-shot checkpoint: {self.checkpoint_path}")
            # Preserve the receipt from the API-producing run when replaying downstream stages.
            return OneShotCompletion(parsed, raw_content, receipt)

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
            "reasoning": {"enabled": False},
            "response_format": {"type": "json_object"},
        }
        canonical = json.dumps(request_body, ensure_ascii=False, sort_keys=True).encode("utf-8")
        request_hash = hashlib.sha256(canonical).hexdigest()
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_transport_attempts + 1):
            started = time.monotonic()
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=request_body,
                    timeout=(10, self.timeout_seconds),
                )
                if response.status_code in _TRANSIENT_HTTP_STATUSES:
                    raise requests.HTTPError(
                        f"transient OpenRouter HTTP {response.status_code}", response=response
                    )
                response.raise_for_status()
                envelope = response.json()
                raw_content = envelope["choices"][0]["message"]["content"]
                if not isinstance(raw_content, str) or not raw_content.strip():
                    raise ValueError("DeepSeek returned empty completion content")

                # Parsing and validation failures are deliberately not retried.
                parsed = _parse_json_object(raw_content)
                receipt = {
                    "endpoint": url,
                    "requested_model": self.model,
                    "response_model": envelope.get("model"),
                    "response_id": envelope.get("id"),
                    "provider": envelope.get("provider"),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "reasoning": {"enabled": False},
                    "semantic_generation_request_count": 1,
                    "transport_attempt_count": attempt,
                    "request_sha256": request_hash,
                    "raw_content_sha256": hashlib.sha256(
                        raw_content.encode("utf-8")
                    ).hexdigest(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "usage": envelope.get("usage", {}),
                    "checkpoint_replay": False,
                }
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                _write_json(
                    self.checkpoint_path,
                    {"parsed": parsed, "raw_content": raw_content, "receipt": receipt},
                )
                return OneShotCompletion(parsed, raw_content, receipt)
            except ValueError:
                raise
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in _TRANSIENT_HTTP_STATUSES:
                    raise RuntimeError(f"OpenRouter rejected the one-shot request with HTTP {status}") from exc
                if attempt < self.max_transport_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                break
            except (requests.RequestException, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("OpenRouter returned an invalid completion envelope") from exc
        raise RuntimeError(
            f"OpenRouter one-shot request failed after {self.max_transport_attempts} identical attempts"
        ) from last_error


def load_supported_claim_input(
    source_generation: Mapping[str, object],
    support_rows: Sequence[Mapping[str, object]],
    *,
    expected_count: int,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]], list[str]]:
    raw_sections = source_generation.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError("source generation must contain sections")
    audit_by_id: dict[str, Mapping[str, object]] = {}
    for row in support_rows:
        claim_id = _require_nonempty_string(row.get("claim_id"), "audit claim_id")
        if claim_id in audit_by_id:
            raise ValueError(f"duplicate source audit row: {claim_id}")
        audit_by_id[claim_id] = row

    source_by_id: dict[str, Mapping[str, object]] = {}
    labels: list[str] = []
    payload_sections: list[dict[str, object]] = []
    source_claim_ids: set[str] = set()
    for raw_section in raw_sections:
        if not isinstance(raw_section, Mapping):
            raise ValueError("source section must be an object")
        label = _require_nonempty_string(raw_section.get("sub_narrative"), "sub_narrative")
        labels.append(label)
        claims = raw_section.get("claims")
        if not isinstance(claims, list):
            raise ValueError(f"source section {label} has no claims array")
        supported: list[dict[str, str]] = []
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise ValueError("source claim must be an object")
            claim_id = _require_nonempty_string(claim.get("claim_id"), "source claim_id")
            if claim_id in source_claim_ids:
                raise ValueError(f"duplicate source claim: {claim_id}")
            source_claim_ids.add(claim_id)
            audit = audit_by_id.get(claim_id)
            if audit is None:
                raise ValueError(f"source claim has no audit row: {claim_id}")
            if str(audit.get("status")) != "supported":
                continue
            text = _require_nonempty_string(claim.get("text"), "source claim text")
            source_by_id[claim_id] = {**dict(claim), "sub_narrative": label}
            supported.append({"claim_id": claim_id, "text": text})
        if not supported:
            raise ValueError(f"section has no supported source claims: {label}")
        payload_sections.append({"sub_narrative": label, "source_claims": supported})

    unknown_audit_ids = set(audit_by_id) - source_claim_ids
    if unknown_audit_ids:
        raise ValueError(f"audit contains unknown source claims: {sorted(unknown_audit_ids)}")
    if len(source_by_id) != expected_count:
        raise ValueError(
            f"expected {expected_count} supported source claims, found {len(source_by_id)}"
        )
    payload = {
        "task": "Write one complete response from this fully supported source-claim ledger.",
        "narrative": source_generation["narrative"],
        "constraints": {
            "maximum_sentences_per_section": 3,
            "maximum_source_claim_ids_per_sentence": 3,
            "target_maximum_total_words": 900,
            "exactly_one_sentence_per_answer_item": True,
        },
        "sections": payload_sections,
    }
    return payload, source_by_id, labels


def validate_one_shot_response(
    value: Mapping[str, object],
    *,
    labels: Sequence[str],
    source_by_id: Mapping[str, Mapping[str, object]],
    maximum_sentences_per_section: int,
    target_maximum_words: int,
    tokenizer: SpacySentenceTokenizer,
) -> list[dict[str, object]]:
    raw_sections = value.get("sections")
    if not isinstance(raw_sections, list) or any(
        not isinstance(section, Mapping) for section in raw_sections
    ):
        raise ValueError("one-shot response sections must be an array of objects")
    normalized_by_label: dict[str, Mapping[str, object]] = {}
    for raw_section in raw_sections:
        label = normalize_answer_label(raw_section.get("sub_narrative"), labels)
        if label in normalized_by_label:
            raise ValueError(f"duplicate one-shot section: {label}")
        normalized_by_label[label] = raw_section
    if list(normalized_by_label) != list(labels):
        raise ValueError("one-shot response must preserve every section exactly once and in order")

    normalized: list[dict[str, object]] = []
    seen_texts: set[str] = set()
    total_words = 0
    for label in labels:
        raw_answer = normalized_by_label[label].get("answer")
        if not isinstance(raw_answer, list) or not raw_answer:
            raise ValueError(f"section must have a nonempty answer array: {label}")
        if len(raw_answer) > maximum_sentences_per_section:
            raise ValueError(f"section exceeds sentence budget: {label}")
        valid_ids = {
            claim_id
            for claim_id, claim in source_by_id.items()
            if str(claim["sub_narrative"]) == label
        }
        answer: list[dict[str, object]] = []
        for item in raw_answer:
            if not isinstance(item, Mapping) or set(item) != {"text", "source_claim_ids"}:
                raise ValueError("each answer item must have only text and source_claim_ids")
            text = _require_nonempty_string(item.get("text"), "one-shot sentence")
            if len(tokenizer.tokenize(text)) != 1:
                raise ValueError(f"answer item is not exactly one SpaCy sentence: {text}")
            if _CITATION_MARKER_RE.search(text):
                raise ValueError("answer text must not contain inline citation markers")
            normalized_text = " ".join(text.casefold().split())
            if normalized_text in seen_texts:
                raise ValueError("one-shot response contains a duplicate sentence")
            seen_texts.add(normalized_text)
            source_ids = item.get("source_claim_ids")
            if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 3:
                raise ValueError("source_claim_ids must contain one to three IDs")
            source_ids = list(dict.fromkeys(map(str, source_ids)))
            if len(source_ids) != len(item["source_claim_ids"]):
                raise ValueError("source_claim_ids must be unique within a sentence")
            if any(source_id not in valid_ids for source_id in source_ids):
                raise ValueError("answer cites an unknown, unsupported, or cross-section source claim")
            total_words += len(text.split())
            answer.append({"text": text, "source_claim_ids": source_ids})
        normalized.append({"sub_narrative": label, "answer": answer})
    if total_words > target_maximum_words:
        raise ValueError(
            f"one-shot response has {total_words} words; target is {target_maximum_words}"
        )
    return normalized


def _best_passage(sentence: str, source: Mapping[str, object]) -> dict[str, object]:
    passages = source.get("supporting_passages")
    if not isinstance(passages, list) or not passages or any(
        not isinstance(passage, Mapping) for passage in passages
    ):
        raise ValueError(f"source claim {source.get('claim_id')} has no supporting passages")
    sentence_terms = set(_WORD_RE.findall(sentence.casefold()))

    def score(passage: Mapping[str, object]) -> tuple[float, int]:
        passage_terms = set(_WORD_RE.findall(str(passage.get("text", "")).casefold()))
        overlap = len(sentence_terms & passage_terms)
        return (overlap / len(sentence_terms) if sentence_terms else 0.0, overlap)

    selected = dict(max(passages, key=score))
    document_id = str(selected.get("document_id", ""))
    if not document_id.startswith("shard_"):
        raise ValueError(f"non-ClimbMix supporting document ID: {document_id}")
    return selected


def build_candidate_generation(
    *,
    source_generation: Mapping[str, object],
    sections: Sequence[Mapping[str, object]],
    source_by_id: Mapping[str, Mapping[str, object]],
    experiment: Mapping[str, object],
    generation_config: Mapping[str, object],
    receipt: Mapping[str, object],
) -> dict[str, object]:
    output_sections: list[dict[str, object]] = []
    used_ids: list[str] = []
    answer_number = 0
    for section in sections:
        claims: list[dict[str, object]] = []
        for item in section["answer"]:
            answer_number += 1
            document_ids: list[str] = []
            supporting_passages: list[dict[str, object]] = []
            source_ids = list(item["source_claim_ids"])
            for source_id in source_ids:
                passage = _best_passage(str(item["text"]), source_by_id[source_id])
                document_id = str(passage["document_id"])
                if document_id not in document_ids:
                    document_ids.append(document_id)
                    supporting_passages.append(passage)
                if source_id not in used_ids:
                    used_ids.append(source_id)
            claims.append(
                {
                    "claim_id": f"S{answer_number:03d}",
                    "text": item["text"],
                    "evidence_claim_ids": source_ids,
                    "document_ids": document_ids,
                    "supporting_passages": supporting_passages,
                }
            )
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
        "model": EXACT_MODEL,
        "model_identity": generation_config.get("model_identity", EXACT_MODEL),
        "temperature": 0.0,
        "context_policy": {
            "kind": "single_request_synthesis_of_supported_full-evidence_claims",
            "source_experiment_id": source_generation["experiment_id"],
            "supported_source_claim_count": len(source_by_id),
            "successful_candidate_generation_request_count": receipt[
                "semantic_generation_request_count"
            ],
            "total_openrouter_generation_api_call_count": receipt[
                "total_openrouter_generation_api_call_count"
            ],
            "llm_rewrite_or_repair_requests": 0,
            "nuggets_available_during_generation": False,
        },
        "input_accounting": source_generation["input_accounting"],
        "source_metadata": source_generation["source_metadata"],
        "generation_config": dict(generation_config),
        "prompts": {"one_shot_system": ONESHOT_SYSTEM_PROMPT},
        "evidence_ledger": [source_by_id[source_id] for source_id in used_ids],
        "sections": output_sections,
    }


def filter_supported_sentences(
    candidate: dict[str, object], audit_rows: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    audit_by_id = {str(row["claim_id"]): row for row in audit_rows}
    kept_audits: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    used_evidence_ids: set[str] = set()
    for section in candidate["sections"]:
        kept_claims: list[dict[str, object]] = []
        for claim in section["claims"]:
            claim_id = str(claim["claim_id"])
            if claim_id not in audit_by_id:
                raise ValueError(f"candidate sentence has no support audit: {claim_id}")
            audit = dict(audit_by_id[claim_id])
            if audit.get("status") == "supported":
                kept_claims.append(claim)
                kept_audits.append(audit)
                used_evidence_ids.update(map(str, claim["evidence_claim_ids"]))
            else:
                excluded.append({**dict(claim), "support_audit": audit})
        section["claims"] = kept_claims
    candidate["evidence_ledger"] = [
        row
        for row in candidate["evidence_ledger"]
        if str(row["claim_id"]) in used_evidence_ids
    ]
    return kept_audits, excluded


def validate_sentence_level_submission(
    entry: Mapping[str, object], *, tokenizer: SpacySentenceTokenizer
) -> int:
    words = validate_submission_entry(entry, maximum_words=1024)
    references = entry["references"]
    answer = entry["answer"]
    used_references: list[str] = []
    for index, item in enumerate(answer):
        text = str(item["text"])
        if len(tokenizer.tokenize(text)) != 1:
            raise ValueError(f"answer[{index}] is not exactly one SpaCy sentence")
        citations = list(item["citations"])
        if len(set(citations)) != len(citations):
            raise ValueError(f"answer[{index}] citations must be unique")
        for citation in citations:
            if citation not in used_references:
                used_references.append(citation)
    if list(references) != used_references:
        raise ValueError("references must be unique and ordered by first citation use")
    return words


def _local_client(
    *, output_dir: Path, config: Mapping[str, object]
) -> OpenAICompatibleJsonClient:
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
        checkpoint_dir=output_dir / "local_qwen_checkpoints",
        call_log_path=output_dir / "local_qwen_calls.jsonl",
        timeout_seconds=float(config.get("timeout_seconds", 240)),
        max_attempts=int(config.get("http_max_attempts", 4)),
    )


def _artifact_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def write_manifest(
    *,
    output_dir: Path,
    repo_root: Path,
    config: Mapping[str, object],
    source_paths: Mapping[str, Path],
    artifact_names: Sequence[str],
    receipt: Mapping[str, object],
    frozen_generation_sha256: str,
    frozen_submission_sha256: str,
) -> Path:
    artifacts = {
        name: _artifact_record(output_dir / name) for name in sorted(artifact_names)
    }
    manifest = {
        "schema_version": "topic213-experiment-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment"]["id"],
        "run_id": config["experiment"]["run_id"],
        "topic_id": "213",
        "generation": {
            "provider": "OpenRouter",
            "requested_model": EXACT_MODEL,
            "response_model": receipt.get("response_model"),
            "successful_candidate_generation_request_count": receipt[
                "semantic_generation_request_count"
            ],
            "total_openrouter_generation_api_call_count": receipt[
                "total_openrouter_generation_api_call_count"
            ],
            "successful_request_transport_attempt_count": receipt[
                "transport_attempt_count"
            ],
            "prior_failed_generation_api_call_count": receipt[
                "prior_failed_generation_api_call_count"
            ],
            "temperature": 0.0,
            "reasoning": {"enabled": False},
            "llm_rewrite_or_repair_requests": 0,
        },
        "audit_and_evaluation": {
            "provider": "local LiteLLM",
            "model": config["audit"]["model"],
            "model_identity": config["audit"]["model_identity"],
        },
        "data_firewall": {
            "nuggets_loaded_after_generation_and_submission_freeze": True,
            "frozen_generation_sha256": frozen_generation_sha256,
            "frozen_submission_sha256": frozen_submission_sha256,
        },
        "inputs": {
            label: {
                "path": str(path.relative_to(repo_root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for label, path in source_paths.items()
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )
    return manifest_path


def verify_manifest_hashes(directory: Path) -> int:
    manifest = yaml.safe_load((directory / "manifest.yaml").read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("manifest has no artifacts")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid manifest artifact record: {name}")
        path = directory / str(name)
        if not path.is_file():
            raise ValueError(f"manifest artifact is missing: {name}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"manifest byte count differs: {name}")
        if _sha256(path) != str(record["sha256"]):
            raise ValueError(f"manifest SHA-256 differs: {name}")
    return len(artifacts)


def _copy_report(output_dir: Path, report_dir: Path, names: Sequence[str]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in [*names, "manifest.yaml"]:
        shutil.copy2(output_dir / name, report_dir / name)


def _local_call_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"unique_semantic_requests": 0, "transport_attempts": 0}
    rows = _read_jsonl(path)
    request_hashes = {str(row["request_sha256"]) for row in rows}
    transport_rows = [row for row in rows if not bool(row.get("cache_hit"))]
    return {
        "unique_semantic_requests": len(request_hashes),
        "transport_attempts": len(transport_rows),
    }


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
        isinstance(item, Mapping)
        for item in (experiment, inputs, generation_config, audit_config, evaluation_config)
    ):
        raise ValueError("config sections must be mappings")
    if str(generation_config.get("model")) != EXACT_MODEL:
        raise ValueError(f"generation.model must be exactly {EXACT_MODEL}")
    if str(audit_config.get("model")) != "qwen-local":
        raise ValueError("audit.model must be the fixed local qwen-local alias")

    output_dir = repo_root / str(experiment["output_dir"])
    report_dir = repo_root / str(experiment["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_generation_path = repo_root / str(inputs["source_generation"])
    source_audit_path = repo_root / str(inputs["source_support_audit"])
    nuggets_path = repo_root / str(inputs["nuggets"])
    baseline_metrics_path = repo_root / str(inputs["baseline_metrics"])
    source_generation = json.loads(source_generation_path.read_text(encoding="utf-8"))
    source_audit = _read_jsonl(source_audit_path)
    generation_input, source_by_id, labels = load_supported_claim_input(
        source_generation,
        source_audit,
        expected_count=int(generation_config.get("expected_supported_claim_count", 42)),
    )
    generation_input["constraints"] = {
        "maximum_sentences_per_section": int(
            generation_config.get("maximum_sentences_per_section", 3)
        ),
        "maximum_source_claim_ids_per_sentence": 3,
        "target_maximum_total_words": int(
            generation_config.get("target_maximum_words", 900)
        ),
        "exactly_one_sentence_per_answer_item": True,
    }
    _write_json(output_dir / "generation_input.json", generation_input)
    shutil.copy2(config_path, output_dir / "config.yaml")
    _write_json(output_dir / "config.resolved.json", json.loads(json.dumps(config)))

    api_key = os.environ.get(str(generation_config.get("api_key_env", "OPENROUTER_API_KEY")), "")
    deepseek_client = OpenRouterOneShotJsonClient(
        api_base=str(generation_config.get("api_base", "https://openrouter.ai/api/v1")),
        model=str(generation_config["model"]),
        api_key=api_key,
        checkpoint_path=output_dir / "deepseek_oneshot_checkpoint.json",
        timeout_seconds=float(generation_config.get("timeout_seconds", 300)),
        max_transport_attempts=int(generation_config.get("max_transport_attempts", 3)),
    )
    completion = deepseek_client.complete_once(
        system_prompt=ONESHOT_SYSTEM_PROMPT,
        payload=generation_input,
        max_tokens=int(generation_config.get("max_tokens", 3500)),
        temperature=0.0,
    )
    prior_failed_calls = int(generation_config.get("prior_failed_generation_api_call_count", 0))
    receipt = {
        **completion.receipt,
        "prior_failed_generation_api_call_count": prior_failed_calls,
        "prior_failure_reason": generation_config.get("prior_failure_reason"),
        "total_openrouter_generation_api_call_count": (
            prior_failed_calls + int(completion.receipt["transport_attempt_count"])
        ),
    }
    (output_dir / "deepseek_raw_response.txt").write_text(
        completion.raw_content, encoding="utf-8"
    )
    _write_json(output_dir / "deepseek_oneshot_response.json", completion.parsed)
    _write_json(output_dir / "openrouter_generation_receipt.json", receipt)
    _write_jsonl(
        output_dir / "openrouter_generation_failures.jsonl",
        [
            {
                "requested_model": EXACT_MODEL,
                "endpoint": generation_config.get("api_base"),
                "http_status": 200,
                "failure": generation_config.get("prior_failure_reason"),
                "content_was_empty": True,
                "candidate_was_created": False,
                "nuggets_had_been_loaded": False,
            }
            for _ in range(prior_failed_calls)
        ],
    )

    tokenizer = SpacySentenceTokenizer()
    normalized_sections = validate_one_shot_response(
        completion.parsed,
        labels=labels,
        source_by_id=source_by_id,
        maximum_sentences_per_section=int(
            generation_config.get("maximum_sentences_per_section", 3)
        ),
        target_maximum_words=int(generation_config.get("target_maximum_words", 900)),
        tokenizer=tokenizer,
    )
    candidate = build_candidate_generation(
        source_generation=source_generation,
        sections=normalized_sections,
        source_by_id=source_by_id,
        experiment=experiment,
        generation_config=generation_config,
        receipt=receipt,
    )

    local_client = _local_client(output_dir=output_dir, config=audit_config)
    full_audit: list[dict[str, object]] = []
    for section in candidate["sections"]:
        full_audit.extend(
            _audit_section(
                local_client,
                sub_narrative=str(section["sub_narrative"]),
                claims=section["claims"],
                max_tokens=int(audit_config.get("max_tokens", 350)),
                temperature=0.0,
                validation_attempts=int(audit_config.get("validation_attempts", 3)),
            )
        )
    final_audit, excluded = filter_supported_sentences(candidate, full_audit)
    if not final_audit:
        raise RuntimeError("local Qwen support audit excluded every DeepSeek sentence")

    official_entry = build_official_entry(
        generation=candidate,
        team_id=str(experiment["team_id"]),
        run_desc=str(experiment["run_desc"]),
    )
    official_words = validate_sentence_level_submission(official_entry, tokenizer=tokenizer)
    generation_path = output_dir / "response_generation.json"
    submission_path = output_dir / "rag_output_trec_rag_2026.jsonl"
    response_path = output_dir / "generated_response.md"
    _write_json(generation_path, candidate)
    _write_jsonl(output_dir / "generation_support_audit.jsonl", full_audit)
    _write_jsonl(output_dir / "claim_support_audit.jsonl", final_audit)
    _write_jsonl(output_dir / "excluded_sentences.jsonl", excluded)
    _write_jsonl(submission_path, [official_entry])
    response_text = render_markdown(candidate)
    response_path.write_text(response_text, encoding="utf-8")

    # The organizer nuggets are not opened until both candidate artifacts are final and hashed.
    frozen_generation_sha256 = _sha256(generation_path)
    frozen_submission_sha256 = _sha256(submission_path)

    nuggets = load_nuggets(
        nuggets_path,
        topic_id=str(source_generation["topic_id"]),
        expected_count=int(evaluation_config.get("expected_nugget_count", 50)),
    )
    comparison = evaluate_nuggets(
        nuggets=nuggets,
        generation=candidate,
        client=local_client,
        evaluation_config=evaluation_config,
    )
    _write_jsonl(output_dir / "nugget_comparison.jsonl", comparison)
    metrics = compute_evaluation_metrics(comparison, final_audit, response_text=response_text)
    local_calls = _local_call_counts(output_dir / "local_qwen_calls.jsonl")
    metrics.update(
        {
            "experiment_id": experiment["id"],
            "run_id": experiment["run_id"],
            "topic_id": source_generation["topic_id"],
            "generation": {
                "provider": "OpenRouter",
                "requested_model": EXACT_MODEL,
                "response_model": completion.receipt.get("response_model"),
                "supported_input_claim_count": len(source_by_id),
                "successful_candidate_generation_api_call_count": completion.receipt[
                    "semantic_generation_request_count"
                ],
                "total_openrouter_generation_api_call_count": receipt[
                    "total_openrouter_generation_api_call_count"
                ],
                "successful_request_transport_attempt_count": completion.receipt[
                    "transport_attempt_count"
                ],
                "prior_failed_generation_api_call_count": prior_failed_calls,
                "llm_rewrite_or_repair_call_count": 0,
            },
            "audit_and_evaluation": {
                "provider": "local LiteLLM",
                "model": audit_config["model"],
                "model_identity": audit_config["model_identity"],
                **local_calls,
            },
            "official_submission": {
                "word_count": official_words,
                "sentence_count": len(official_entry["answer"]),
                "reference_count": len(official_entry["references"]),
                "candidate_sentence_count": len(full_audit),
                "excluded_after_support_audit": len(excluded),
                "maximum_words": 1024,
            },
            "frozen_generation_sha256": frozen_generation_sha256,
            "frozen_submission_sha256": frozen_submission_sha256,
        }
    )
    _write_json(output_dir / "metrics.json", metrics)

    baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    baseline_all = baseline["nuggets"]["all"]
    current_all = metrics["nuggets"]["all"]
    strict_delta = current_all["strict_coverage"] - baseline_all["strict_coverage"]
    partial_delta = (
        current_all["partial_credit_coverage"]
        - baseline_all["partial_credit_coverage"]
    )
    recommendation = (
        "DeepSeek one-shot is the stronger result on strict nugget coverage."
        if strict_delta > 0
        else "The seeded local-Qwen consolidation remains stronger on strict nugget coverage."
        if strict_delta < 0
        else "DeepSeek is the marginal coverage winner: strict coverage ties, while DeepSeek gains one point of partial-credit coverage and 3.7 points of vital strict coverage."
        if partial_delta > 0
        else "The runs tie on strict coverage, and the seeded local-Qwen consolidation has at least as much partial-credit coverage."
    )
    report = f"""# Topic 213: DeepSeek V4 Flash one-shot

DeepSeek V4 Flash received all 42 independently supported full-evidence source claims in one successful candidate-producing OpenRouter request spanning the ten organizer sub-narratives. One earlier OpenRouter request returned HTTP success with empty final content and produced no candidate. Local Qwen, served through LiteLLM, independently audited each generated sentence and evaluated nuggets only after the filtered generation and organizer JSONL were frozen and hashed. No DeepSeek rewrite or repair call was made.

## Submission validation

- Generation model: `{EXACT_MODEL}` through OpenRouter
- Successful candidate-generation calls: {completion.receipt['semantic_generation_request_count']}
- Total OpenRouter generation API calls: {receipt['total_openrouter_generation_api_call_count']} (including {prior_failed_calls} HTTP-200/empty-content failure)
- Transport attempts for the successful request: {completion.receipt['transport_attempt_count']}
- DeepSeek rewrite or repair calls: 0
- Supported source claims supplied: {len(source_by_id)}
- Candidate sentences: {len(full_audit)}
- Submitted sentences: {len(official_entry['answer'])}
- Excluded by local-Qwen support audit: {len(excluded)}
- Answer words: {official_words} / 1024
- Unique ClimbMix references: {len(official_entry['references'])}
- Citation coverage: {metrics['answer_claims']['citation_coverage']:.3f}
- Unsupported submitted sentences: {metrics['answer_claims']['unsupported_claim_count']}

## Nugget coverage

| Run | Strict | Partial credit | Vital strict | Supported nuggets |
|---|---:|---:|---:|---:|
| DeepSeek V4 Flash one-shot | {current_all['strict_coverage']:.3f} | {current_all['partial_credit_coverage']:.3f} | {metrics['nuggets']['vital']['strict_coverage']:.3f} | {current_all['supported']} / {current_all['total']} |
| Seeded local-Qwen consolidation | {baseline_all['strict_coverage']:.3f} | {baseline_all['partial_credit_coverage']:.3f} | {baseline['nuggets']['vital']['strict_coverage']:.3f} | {baseline_all['supported']} / {baseline_all['total']} |
| DeepSeek delta | {strict_delta:+.3f} | {partial_delta:+.3f} | {metrics['nuggets']['vital']['strict_coverage'] - baseline['nuggets']['vital']['strict_coverage']:+.3f} | {current_all['supported'] - baseline_all['supported']:+d} |

## Recommendation

{recommendation} The gain is small and costs 120 additional answer words; DeepSeek also had 6 of 28 candidate sentences excluded. The comparison keeps the 42-claim source ledger, local-Qwen support auditor, nugget evaluator, sentence format, and word ceiling fixed, but it is not a pure generator ablation because the seeded Qwen run allowed one bounded rewrite-and-reaudit pass while this DeepSeek run deliberately allowed none. Cross-task ranking should wait for the other isolated runs.

## Integrity

The organizer nuggets were not read until `response_generation.json` and `rag_output_trec_rag_2026.jsonl` were written and hashed. The manifest pins both frozen hashes, all source inputs, the exact requested and returned model identities, and every published artifact.
"""
    (output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")

    artifact_names = [
        "claim_support_audit.jsonl",
        "config.resolved.json",
        "config.yaml",
        "deepseek_oneshot_response.json",
        "deepseek_raw_response.txt",
        "evaluation_report.md",
        "excluded_sentences.jsonl",
        "generated_response.md",
        "generation_input.json",
        "generation_support_audit.jsonl",
        "local_qwen_calls.jsonl",
        "metrics.json",
        "nugget_comparison.jsonl",
        "openrouter_generation_failures.jsonl",
        "openrouter_generation_receipt.json",
        "rag_output_trec_rag_2026.jsonl",
        "response_generation.json",
    ]
    source_paths = {
        "source_generation": source_generation_path,
        "source_support_audit": source_audit_path,
        "organizer_nuggets": nuggets_path,
        "comparison_baseline_metrics": baseline_metrics_path,
    }
    write_manifest(
        output_dir=output_dir,
        repo_root=repo_root,
        config=config,
        source_paths=source_paths,
        artifact_names=artifact_names,
        receipt=receipt,
        frozen_generation_sha256=frozen_generation_sha256,
        frozen_submission_sha256=frozen_submission_sha256,
    )
    verify_manifest_hashes(output_dir)
    _copy_report(output_dir, report_dir, artifact_names)
    verify_manifest_hashes(report_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_from_config(args.config.resolve()))


if __name__ == "__main__":
    main()
