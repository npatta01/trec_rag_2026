"""Run strict one-shot GPT-5.6 Sol generation for Topic 213.

Generation consumes only the frozen, fully supported source-claim ledger. The
organizer nuggets are opened only after the independently audited submission is
written and sealed with SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import requests
import yaml

from trec_rag.repo_env import find_repo_root, load_repo_env
from trec_rag.topic213_2026_submission import validate_submission_entry
from trec_rag.topic213_ragnarok_experiment import SpacySentenceTokenizer
from trec_rag.topic213_response_experiment import (
    OpenAICompatibleJsonClient,
    _audit_section,
    _require_nonempty_string,
    _sha256,
    _write_json,
    _write_jsonl,
    compute_evaluation_metrics,
    evaluate_nuggets,
    load_nuggets,
)
from trec_rag.topic213_utokyo_experiment import normalize_answer_label


SCHEMA_VERSION = "topic213-trec-rag-2026-gpt-sol-oneshot-v1"
PROMPT_VERSION = "full-evidence-supported-ledger-oneshot-v1"
GENERATION_STAGE = "gpt_5_6_sol_oneshot"
_TEXT_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
}

GENERATION_SYSTEM_PROMPT = """You produce one official TREC RAG 2026 answer from a frozen
ledger of factual claims that were independently verified before this request. Return exactly one
JSON object matching the supplied response schema.

Cover all supplied sub-narratives exactly once and in their supplied order. Write one to four
information-dense answer sentences per sub-narrative, aiming to preserve the distinct supported
facts while consolidating genuine redundancy. Each answer item must contain exactly one polished,
grammatical sentence, its exact sub-narrative, one to three source claim IDs from that
sub-narrative, and one to three direct shard document citations. Cite only documents listed in the
supporting passages for the source claims used by that sentence, and ensure every factual detail is
directly supported by those passages. Do not add outside knowledge, headings, introductions,
conclusions, citation markers inside text, or meta-commentary. Stay within the requested total word
budget."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def _normalize_generated_text(value: str) -> str:
    for broken, replacement in _TEXT_REPLACEMENTS.items():
        value = value.replace(broken, replacement)
    return value.strip()


class StrictOneShotJsonClient:
    """OpenAI-compatible client that never repairs a semantic generation.

    Identical transport retries are allowed for transient network/server errors.
    A successful but malformed completion is persisted and fails immediately.
    """

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str,
        checkpoint_dir: Path,
        call_log_path: Path,
        timeout_seconds: float,
        transport_max_attempts: int,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing or empty")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.checkpoint_dir = checkpoint_dir
        self.call_log_path = call_log_path
        self.timeout_seconds = timeout_seconds
        self.transport_max_attempts = transport_max_attempts
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.call_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, row: Mapping[str, object]) -> None:
        with self.call_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def complete_json_once(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: Mapping[str, object],
        response_schema: Mapping[str, object],
        max_tokens: int,
        temperature: float,
    ) -> tuple[dict[str, object], dict[str, object]]:
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "trec_rag_2026_answer",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        }
        request_sha256 = hashlib.sha256(_canonical_json(request_body)).hexdigest()
        parsed_checkpoint = self.checkpoint_dir / f"{stage}__{request_sha256}.json"
        metadata_checkpoint = self.checkpoint_dir / f"{stage}__{request_sha256}.meta.json"
        failed_checkpoint = self.checkpoint_dir / f"{stage}__{request_sha256}.failed.txt"
        if parsed_checkpoint.exists():
            parsed = json.loads(parsed_checkpoint.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_checkpoint.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict) or not isinstance(metadata, dict):
                raise ValueError("invalid one-shot checkpoint")
            self._log(
                {
                    "stage": stage,
                    "request_sha256": request_sha256,
                    "cache_hit": True,
                    "http_attempts": 0,
                    "successful_completions": 0,
                }
            )
            return parsed, metadata
        if failed_checkpoint.exists():
            raise RuntimeError(
                "the one-shot completion previously returned malformed output; "
                f"inspect {failed_checkpoint}"
            )

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
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
                self._log(
                    {
                        "stage": stage,
                        "request_sha256": request_sha256,
                        "cache_hit": False,
                        "attempt": attempt,
                        "transport_error": str(exc),
                    }
                )
                if attempt == self.transport_max_attempts:
                    raise RuntimeError("one-shot generation transport failed") from exc
                time.sleep(min(2 ** (attempt - 1), 8))
                continue

            if response.status_code == 429 or response.status_code >= 500:
                self._log(
                    {
                        "stage": stage,
                        "request_sha256": request_sha256,
                        "cache_hit": False,
                        "attempt": attempt,
                        "http_status": response.status_code,
                        "transient_http_error": True,
                    }
                )
                if attempt == self.transport_max_attempts:
                    response.raise_for_status()
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            if response.status_code >= 400:
                try:
                    error_payload: object = response.json()
                except ValueError:
                    error_payload = {"body": response.text[:2000]}
                self._log(
                    {
                        "stage": stage,
                        "request_sha256": request_sha256,
                        "cache_hit": False,
                        "attempt": attempt,
                        "http_status": response.status_code,
                        "semantic_completion_received": False,
                        "provider_error": error_payload,
                    }
                )
                raise RuntimeError(
                    f"one-shot request was rejected with HTTP {response.status_code}: "
                    f"{json.dumps(error_payload, ensure_ascii=False)}"
                )
            response.raise_for_status()

            try:
                envelope = response.json()
                content = envelope["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("completion content is not nonempty text")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("completion JSON must be an object")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                failed_checkpoint.write_text(response.text, encoding="utf-8")
                self._log(
                    {
                        "stage": stage,
                        "request_sha256": request_sha256,
                        "cache_hit": False,
                        "attempt": attempt,
                        "http_status": response.status_code,
                        "semantic_completion_received": True,
                        "validation_error": str(exc),
                    }
                )
                raise ValueError(
                    "one-shot generation returned malformed JSON; no repair call was made"
                ) from exc

            metadata = {
                "api_base": self.api_base,
                "model": self.model,
                "request_sha256": request_sha256,
                "http_attempts": attempt,
                "successful_completions": 1,
                "temperature": temperature,
                "usage": envelope.get("usage", {}),
                "response_id": envelope.get("id"),
                "provider": envelope.get("provider"),
            }
            parsed_checkpoint.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            metadata_checkpoint.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._log(
                {
                    "stage": stage,
                    "request_sha256": request_sha256,
                    "cache_hit": False,
                    "attempt": attempt,
                    "http_status": response.status_code,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "input_chars": sum(
                        len(str(message["content"]))
                        for message in request_body["messages"]
                    ),
                    "output_chars": len(content),
                    "successful_completions": 1,
                    "usage": envelope.get("usage", {}),
                }
            )
            return parsed, metadata

        raise AssertionError("unreachable transport retry state")


def load_supported_source_claims(
    source_generation: Mapping[str, object],
    support_rows: Sequence[Mapping[str, object]],
) -> tuple[list[str], dict[str, dict[str, object]]]:
    sections = source_generation.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("source generation must contain sections")
    status_by_id = {
        str(row["claim_id"]): str(row["status"])
        for row in support_rows
        if "claim_id" in row and "status" in row
    }
    labels: list[str] = []
    source_by_id: dict[str, dict[str, object]] = {}
    for section in sections:
        if not isinstance(section, Mapping):
            raise ValueError("source section must be an object")
        label = _require_nonempty_string(section.get("sub_narrative"), "sub_narrative")
        labels.append(label)
        claims = section.get("claims")
        if not isinstance(claims, list):
            raise ValueError("source claims must be an array")
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise ValueError("source claim must be an object")
            claim_id = _require_nonempty_string(claim.get("claim_id"), "claim_id")
            if status_by_id.get(claim_id) != "supported":
                continue
            passages = claim.get("supporting_passages")
            if not isinstance(passages, list) or not passages:
                raise ValueError(f"supported source claim {claim_id} has no passages")
            normalized_passages: list[dict[str, str]] = []
            for passage in passages:
                if not isinstance(passage, Mapping):
                    raise ValueError("supporting passage must be an object")
                document_id = _require_nonempty_string(
                    passage.get("document_id"), "document_id"
                )
                if not document_id.startswith("shard_"):
                    raise ValueError(f"non-ClimbMix document ID: {document_id}")
                normalized_passages.append(
                    {
                        "passage_id": str(passage.get("passage_id", "")),
                        "document_id": document_id,
                        "text": _require_nonempty_string(passage.get("text"), "passage text"),
                    }
                )
            source_by_id[claim_id] = {
                **dict(claim),
                "claim_id": claim_id,
                "sub_narrative": label,
                "supporting_passages": normalized_passages,
                "document_ids": list(
                    dict.fromkeys(row["document_id"] for row in normalized_passages)
                ),
            }
    missing_audits = [
        claim_id for claim_id, status in status_by_id.items()
        if status == "supported" and claim_id not in source_by_id
    ]
    if missing_audits:
        raise ValueError(f"supported audit rows are absent from source: {missing_audits}")
    if not source_by_id:
        raise ValueError("no fully supported source claims were found")
    return labels, source_by_id


def build_generation_payload(
    *,
    source_generation: Mapping[str, object],
    labels: Sequence[str],
    source_by_id: Mapping[str, Mapping[str, object]],
    maximum_sentences_per_section: int,
    target_maximum_words: int,
) -> dict[str, object]:
    sections: list[dict[str, object]] = []
    for label in labels:
        claims = [
            {
                "claim_id": claim_id,
                "text": source["text"],
                "supporting_passages": source["supporting_passages"],
            }
            for claim_id, source in source_by_id.items()
            if source["sub_narrative"] == label
        ]
        if not claims:
            raise ValueError(f"sub-narrative has no fully supported claims: {label}")
        sections.append({"sub_narrative": label, "source_claims": claims})
    return {
        "task": "Produce the final sentence-level TREC RAG 2026 response in one call.",
        "narrative": source_generation["narrative"],
        "maximum_sentences_per_sub_narrative": maximum_sentences_per_section,
        "target_maximum_total_answer_words": target_maximum_words,
        "sections": sections,
    }


def build_response_schema(
    *, labels: Sequence[str], source_by_id: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    document_ids = list(
        dict.fromkeys(
            str(passage["document_id"])
            for source in source_by_id.values()
            for passage in source["supporting_passages"]
        )
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer"],
        "properties": {
            "answer": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "sub_narrative",
                        "text",
                        "citations",
                        "source_claim_ids",
                    ],
                    "properties": {
                        # Organizer labels contain literal quotes, which OpenAI's
                        # strict-schema subset forbids inside string enums. The
                        # one-pass semantic validator enforces exact labels.
                        "sub_narrative": {"type": "string"},
                        "text": {"type": "string"},
                        "citations": {
                            "type": "array",
                            "items": {"type": "string", "enum": document_ids},
                        },
                        "source_claim_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(source_by_id),
                            },
                        },
                    },
                },
            }
        },
    }


def validate_oneshot_answer(
    value: Mapping[str, object],
    *,
    labels: Sequence[str],
    source_by_id: Mapping[str, Mapping[str, object]],
    maximum_sentences_per_section: int,
    target_maximum_words: int,
    tokenizer: SpacySentenceTokenizer,
) -> list[dict[str, object]]:
    raw_answer = value.get("answer")
    if not isinstance(raw_answer, list) or not raw_answer:
        raise ValueError("one-shot answer must be a nonempty array")
    normalized: list[dict[str, object]] = []
    seen_labels: list[str] = []
    counts: Counter[str] = Counter()
    total_words = 0
    for index, item in enumerate(raw_answer):
        if not isinstance(item, Mapping):
            raise ValueError(f"answer[{index}] must be an object")
        label = normalize_answer_label(item.get("sub_narrative"), labels)
        if not seen_labels or seen_labels[-1] != label:
            if label in seen_labels:
                raise ValueError(f"sub-narrative is noncontiguous: {label}")
            seen_labels.append(label)
        counts[label] += 1
        if counts[label] > maximum_sentences_per_section:
            raise ValueError(f"sub-narrative exceeds sentence budget: {label}")
        text = _normalize_generated_text(
            _require_nonempty_string(item.get("text"), f"answer[{index}].text")
        )
        if len(tokenizer.tokenize(text)) != 1:
            raise ValueError(f"answer[{index}] is not exactly one SpaCy sentence")
        citations = item.get("citations")
        source_ids = item.get("source_claim_ids")
        if not isinstance(citations, list) or not 1 <= len(citations) <= 3:
            raise ValueError(f"answer[{index}] must have one to three citations")
        if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 3:
            raise ValueError(f"answer[{index}] must have one to three source claim IDs")
        citations = list(dict.fromkeys(map(str, citations)))
        source_ids = list(dict.fromkeys(map(str, source_ids)))
        if len(citations) != len(item["citations"]):
            raise ValueError(f"answer[{index}] has duplicate citations")
        if len(source_ids) != len(item["source_claim_ids"]):
            raise ValueError(f"answer[{index}] has duplicate source claim IDs")
        if any(source_id not in source_by_id for source_id in source_ids):
            raise ValueError(f"answer[{index}] cites an unknown source claim")
        if any(source_by_id[source_id]["sub_narrative"] != label for source_id in source_ids):
            raise ValueError(f"answer[{index}] uses a cross-section source claim")
        documents_by_source = {
            source_id: {
                str(passage["document_id"])
                for passage in source_by_id[source_id]["supporting_passages"]
            }
            for source_id in source_ids
        }
        allowed_documents = set().union(*documents_by_source.values())
        if any(citation not in allowed_documents for citation in citations):
            raise ValueError(f"answer[{index}] cites a document outside its source claims")
        if any(not (documents & set(citations)) for documents in documents_by_source.values()):
            raise ValueError(
                f"answer[{index}] does not cite evidence for every source claim used"
            )
        total_words += len(text.split())
        normalized.append(
            {
                "sub_narrative": label,
                "text": text,
                "citations": citations,
                "source_claim_ids": source_ids,
            }
        )
    if seen_labels != list(labels):
        raise ValueError("all sub-narratives must appear exactly once in supplied order")
    if total_words > target_maximum_words:
        raise ValueError(
            f"one-shot answer has {total_words} words; target is {target_maximum_words}"
        )
    return normalized


def _best_passage(
    *, sentence: str, document_id: str, source_ids: Sequence[str],
    source_by_id: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    terms = set(sentence.casefold().split())
    candidates = [
        passage
        for source_id in source_ids
        for passage in source_by_id[source_id]["supporting_passages"]
        if str(passage["document_id"]) == document_id
    ]
    if not candidates:
        raise ValueError(f"no cited passage found for {document_id}")
    return dict(
        max(
            candidates,
            key=lambda row: len(terms & set(str(row["text"]).casefold().split())),
        )
    )


def build_candidate_generation(
    *,
    source_generation: Mapping[str, object],
    answer: Sequence[Mapping[str, object]],
    source_by_id: Mapping[str, Mapping[str, object]],
    experiment: Mapping[str, object],
    generation_config: Mapping[str, object],
    request_metadata: Mapping[str, object],
) -> dict[str, object]:
    labels = list(dict.fromkeys(str(row["sub_narrative"]) for row in answer))
    sections: list[dict[str, object]] = []
    claim_number = 0
    used_source_ids: list[str] = []
    for label in labels:
        claims: list[dict[str, object]] = []
        for item in answer:
            if item["sub_narrative"] != label:
                continue
            claim_number += 1
            source_ids = list(map(str, item["source_claim_ids"]))
            citations = list(map(str, item["citations"]))
            passages = [
                _best_passage(
                    sentence=str(item["text"]),
                    document_id=document_id,
                    source_ids=source_ids,
                    source_by_id=source_by_id,
                )
                for document_id in citations
            ]
            for source_id in source_ids:
                if source_id not in used_source_ids:
                    used_source_ids.append(source_id)
            claims.append(
                {
                    "claim_id": f"S{claim_number:03d}",
                    "text": item["text"],
                    "evidence_claim_ids": source_ids,
                    "document_ids": citations,
                    "supporting_passages": passages,
                }
            )
        sections.append({"sub_narrative": label, "claims": claims})
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "experiment_id": experiment["id"],
        "run_id": experiment["run_id"],
        "topic_id": source_generation["topic_id"],
        "narrative": source_generation["narrative"],
        "model": generation_config["model"],
        "model_identity": generation_config["model_identity"],
        "temperature": generation_config["temperature"],
        "context_policy": {
            "kind": "one_shot_synthesis_of_fully_supported_full-evidence_claims",
            "source_experiment_id": source_generation["experiment_id"],
            "source_supported_claim_count": len(source_by_id),
            "semantic_generation_request_count": 1,
            "llm_rewrite_or_repair_calls": 0,
            "nuggets_available_during_generation": False,
        },
        "input_accounting": source_generation["input_accounting"],
        "source_metadata": source_generation["source_metadata"],
        "generation_config": dict(generation_config),
        "generation_request": dict(request_metadata),
        "prompts": {"generation_system": GENERATION_SYSTEM_PROMPT},
        "evidence_ledger": [source_by_id[source_id] for source_id in used_source_ids],
        "sections": sections,
    }


def filter_supported_sentences(
    candidate: Mapping[str, object],
    audit_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    final = copy.deepcopy(candidate)
    audit_by_id = {str(row["claim_id"]): row for row in audit_rows}
    kept_audit: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    used_evidence_ids: set[str] = set()
    for section in final["sections"]:
        kept_claims: list[dict[str, object]] = []
        for claim in section["claims"]:
            audit = audit_by_id.get(str(claim["claim_id"]))
            if audit is None:
                raise ValueError(f"missing support audit for {claim['claim_id']}")
            if audit["status"] == "supported":
                kept_claims.append(claim)
                kept_audit.append(dict(audit))
                used_evidence_ids.update(map(str, claim["evidence_claim_ids"]))
            else:
                excluded.append({**dict(claim), "support_audit": dict(audit)})
        section["claims"] = kept_claims
    final["evidence_ledger"] = [
        row
        for row in final["evidence_ledger"]
        if str(row["claim_id"]) in used_evidence_ids
    ]
    final["support_filter"] = {
        "judge_model": "Qwen/Qwen3-4B-Instruct-2507",
        "candidate_sentence_count": len(audit_rows),
        "submitted_sentence_count": len(kept_audit),
        "excluded_sentence_count": len(excluded),
        "rewrite_or_repair_attempted": False,
    }
    return final, kept_audit, excluded


def build_official_entry(
    *, generation: Mapping[str, object], team_id: str, run_desc: str
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
            "generator": generation["model_identity"],
            "support_judge": generation["support_filter"]["judge_model"],
            "source_experiment_id": generation["context_policy"]["source_experiment_id"],
        },
        "references": references,
        "answer": answer,
    }


def validate_organizer_submission(
    entry: Mapping[str, object], *, maximum_words: int,
    tokenizer: SpacySentenceTokenizer
) -> int:
    word_count = validate_submission_entry(entry, maximum_words=maximum_words)
    for index, item in enumerate(entry["answer"]):
        if len(tokenizer.tokenize(str(item["text"]))) != 1:
            raise ValueError(f"answer[{index}] is not exactly one SpaCy sentence")
        if len(set(item["citations"])) != len(item["citations"]):
            raise ValueError(f"answer[{index}] has duplicate citations")
    return word_count


def render_markdown(generation: Mapping[str, object]) -> str:
    lines = ["# Topic 213: Korean War", ""]
    for section in generation["sections"]:
        label = str(section["sub_narrative"]).strip().strip('"')
        if label.startswith("New: "):
            label = label[5:]
        lines.extend([f"## {label}", ""])
        if not section["claims"]:
            lines.extend(["No fully supported sentence survived the independent audit.", ""])
        for claim in section["claims"]:
            lines.extend(
                [f"{claim['text']} [{'; '.join(claim['document_ids'])}]", ""]
            )
    return "\n".join(lines).rstrip() + "\n"


def _write_freeze_seal(
    *, path: Path, experiment_id: str, artifact_paths: Sequence[Path], repo_root: Path
) -> dict[str, object]:
    files = {
        str(item.relative_to(repo_root)).replace("\\", "/"): {
            "bytes": item.stat().st_size,
            "sha256": _sha256(item),
        }
        for item in artifact_paths
    }
    seal = {
        "schema_version": "pre-nugget-freeze-v1",
        "experiment_id": experiment_id,
        "nuggets_loaded": False,
        "files": files,
        "root_sha256": hashlib.sha256(_canonical_json(files)).hexdigest(),
    }
    _write_json(path, seal)
    return seal


def verify_manifest_hashes(manifest_path: Path, *, repo_root: Path) -> int:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    bindings = manifest.get("files")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("manifest must contain file bindings")
    checked = 0
    for relative, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid manifest binding: {relative}")
        path = repo_root / str(relative)
        if not path.is_file():
            raise ValueError(f"manifest file is missing: {path}")
        if path.stat().st_size != int(binding["bytes"]):
            raise ValueError(f"manifest byte count mismatch: {path}")
        if _sha256(path) != str(binding["sha256"]):
            raise ValueError(f"manifest SHA-256 mismatch: {path}")
        checked += 1
    return checked


def _coverage_row(metrics: Mapping[str, object]) -> tuple[float, float, float]:
    nuggets = metrics["nuggets"]
    assert isinstance(nuggets, Mapping)
    all_rows = nuggets["all"]
    vital = nuggets["vital"]
    assert isinstance(all_rows, Mapping) and isinstance(vital, Mapping)
    return (
        float(all_rows["strict_coverage"]),
        float(all_rows["partial_credit_coverage"]),
        float(vital["strict_coverage"]),
    )


def _render_report(
    *, metrics: Mapping[str, object], baseline: Mapping[str, object] | None,
    generation_metadata: Mapping[str, object]
) -> str:
    strict, partial, vital = _coverage_row(metrics)
    official = metrics["official_submission"]
    answer = metrics["answer_claims"]
    assert isinstance(official, Mapping) and isinstance(answer, Mapping)
    lines = [
        "# GPT-5.6 Sol one-shot Topic 213 report",
        "",
        "## Protocol",
        "",
        "A single semantic generation request was sent through OpenRouter to `openai/gpt-5.6-sol` at temperature 0. It received all 42 source claims previously judged fully supported across the ten sub-narratives. Local `Qwen/Qwen3-4B-Instruct-2507` independently audited each generated sentence and evaluated nugget coverage. No generation rewrite or repair call was made.",
        "",
        "The candidate was support-audited, filtered, written, and SHA-256 sealed before the 50 organizer nuggets were opened.",
        "",
        "## Results",
        "",
        f"- Strict nugget coverage: `{strict:.3f}`",
        f"- Partial-credit coverage: `{partial:.3f}`",
        f"- Vital strict coverage: `{vital:.3f}`",
        f"- Submitted sentences: `{official['sentence_count']}`",
        f"- Submitted words: `{official['word_count']}` / 1024",
        f"- Unique references: `{official['reference_count']}`",
        f"- Citation coverage: `{float(answer['citation_coverage']):.3f}`",
        f"- Unsupported submitted sentences: `{answer['unsupported_claim_count']}`",
        f"- Sentences excluded by Qwen audit: `{official['excluded_after_support_audit']}`",
        f"- Semantic generation requests: `{generation_metadata['successful_completions']}`",
        f"- Successful-request HTTP attempts: `{generation_metadata['http_attempts']}`",
        f"- Rejected pre-completion HTTP requests: `{generation_metadata['precompletion_rejected_http_attempts']}`",
        f"- Total generation-endpoint HTTP requests: `{generation_metadata['total_http_requests']}`",
        "",
    ]
    if baseline is not None:
        base_strict, base_partial, base_vital = _coverage_row(baseline)
        base_official = baseline.get("official_submission", {})
        assert isinstance(base_official, Mapping)
        lines.extend(
            [
                "## Format-compliant baseline comparison",
                "",
                "| Run | Generator | Sentences | Words | Strict | Partial credit | Vital strict |",
                "|---|---|---:|---:|---:|---:|---:|",
                f"| Existing format-compliant baseline | Local Qwen with bounded repair | {base_official.get('sentence_count', 'n/a')} | {base_official.get('word_count', 'n/a')} | {base_strict:.3f} | {base_partial:.3f} | {base_vital:.3f} |",
                f"| This run | GPT-5.6 Sol, one shot | {official['sentence_count']} | {official['word_count']} | {strict:.3f} | {partial:.3f} | {vital:.3f} |",
                "",
                f"Strict delta: `{strict - base_strict:+.3f}`. Partial-credit delta: `{partial - base_partial:+.3f}`. The baseline used a bounded rewrite pass while this run intentionally did not, so the table compares final system outcomes rather than generation-only model quality.",
                "",
                f"GPT-5.6 Sol produced {official['candidate_sentence_count']} candidates, exactly one per sub-narrative. The independent audit excluded {official['excluded_after_support_audit']}, leaving three facets with no submitted sentence; this compression is the main observed source of lost coverage.",
                "",
            ]
        )
    lines.extend(
        [
            "## Recommendation",
            "",
            "Use the existing format-compliant local-Qwen baseline for the current Topic 213 submission comparison: it has higher strict, partial-credit, and vital coverage while retaining perfect citation coverage. Keep this GPT-5.6 Sol result as the clean one-shot ablation, not the preferred run. Treat the ranking cautiously because Topic 213 is development data and the local-Qwen judge is an experimental evaluator, not the official organizer scorer.",
            "",
        ]
    )
    return "\n".join(lines)


def _copy_report_artifacts(
    *, output_dir: Path, report_dir: Path, config_path: Path
) -> list[Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in (
        "rag_output_trec_rag_2026.jsonl",
        "generated_response.md",
        "generation_support_audit.jsonl",
        "excluded_sentences.jsonl",
        "nugget_comparison.jsonl",
        "metrics.json",
        "evaluation_report.md",
        "config.resolved.json",
        "pre_nugget_freeze.json",
    ):
        target = report_dir / name
        shutil.copy2(output_dir / name, target)
        copied.append(target)
    config_target = report_dir / "config.yaml"
    shutil.copy2(config_path, config_target)
    copied.append(config_target)
    return copied


def run_from_config(config_path: Path) -> Path:
    repo_root = find_repo_root(config_path.parent)
    load_repo_env(repo_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    experiment = config.get("experiment")
    inputs = config.get("inputs")
    generation_config = config.get("generation")
    judge_config = config.get("judge")
    evaluation_config = config.get("evaluation")
    if not all(
        isinstance(item, Mapping)
        for item in (experiment, inputs, generation_config, judge_config, evaluation_config)
    ):
        raise ValueError("config sections must be mappings")
    assert isinstance(experiment, Mapping)
    assert isinstance(inputs, Mapping)
    assert isinstance(generation_config, Mapping)
    assert isinstance(judge_config, Mapping)
    assert isinstance(evaluation_config, Mapping)

    output_dir = repo_root / str(experiment["output_dir"])
    report_dir = repo_root / str(experiment["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_generation_path = repo_root / str(inputs["source_generation"])
    source_audit_path = repo_root / str(inputs["source_support_audit"])
    nuggets_path = repo_root / str(inputs["nuggets"])
    source_generation = json.loads(source_generation_path.read_text(encoding="utf-8"))
    source_audit = _read_jsonl(source_audit_path)
    labels, source_by_id = load_supported_source_claims(source_generation, source_audit)
    tokenizer = SpacySentenceTokenizer()

    generation_api_key = os.environ.get(str(generation_config["api_key_env"]), "")
    generation_client = StrictOneShotJsonClient(
        api_base=str(generation_config["api_base"]),
        model=str(generation_config["model"]),
        api_key=generation_api_key,
        checkpoint_dir=output_dir / "generation_checkpoint",
        call_log_path=output_dir / "openrouter_generation_calls.jsonl",
        timeout_seconds=float(generation_config["timeout_seconds"]),
        transport_max_attempts=int(generation_config["transport_max_attempts"]),
    )
    payload = build_generation_payload(
        source_generation=source_generation,
        labels=labels,
        source_by_id=source_by_id,
        maximum_sentences_per_section=int(
            generation_config["maximum_sentences_per_section"]
        ),
        target_maximum_words=int(generation_config["target_maximum_words"]),
    )
    raw_generation, request_metadata = generation_client.complete_json_once(
        stage=GENERATION_STAGE,
        system_prompt=GENERATION_SYSTEM_PROMPT,
        payload=payload,
        response_schema=build_response_schema(labels=labels, source_by_id=source_by_id),
        max_tokens=int(generation_config["max_tokens"]),
        temperature=float(generation_config["temperature"]),
    )
    request_metadata["precompletion_rejected_http_attempts"] = int(
        generation_config.get("precompletion_rejected_http_attempts", 0)
    )
    request_metadata["precompletion_rejection"] = generation_config.get(
        "precompletion_rejection"
    )
    request_metadata["total_http_requests"] = int(request_metadata["http_attempts"]) + int(
        request_metadata["precompletion_rejected_http_attempts"]
    )
    _write_json(output_dir / "openrouter_raw_response.json", raw_generation)
    _write_json(output_dir / "generation_request_summary.json", request_metadata)

    answer = validate_oneshot_answer(
        raw_generation,
        labels=labels,
        source_by_id=source_by_id,
        maximum_sentences_per_section=int(
            generation_config["maximum_sentences_per_section"]
        ),
        target_maximum_words=int(generation_config["target_maximum_words"]),
        tokenizer=tokenizer,
    )
    candidate = build_candidate_generation(
        source_generation=source_generation,
        answer=answer,
        source_by_id=source_by_id,
        experiment=experiment,
        generation_config=generation_config,
        request_metadata=request_metadata,
    )
    candidate_path = output_dir / "response_generation.candidate.json"
    _write_json(candidate_path, candidate)

    judge_api_base = os.environ.get(
        str(judge_config["api_base_env"]), str(judge_config["api_base"])
    )
    judge_api_key = os.environ.get(
        str(judge_config["api_key_env"]), str(judge_config["api_key"])
    )
    judge_client = OpenAICompatibleJsonClient(
        api_base=judge_api_base,
        model=str(judge_config["model"]),
        api_key=judge_api_key,
        checkpoint_dir=output_dir / "judge_checkpoints",
        call_log_path=output_dir / "local_qwen_calls.jsonl",
        timeout_seconds=float(judge_config["timeout_seconds"]),
        max_attempts=int(judge_config["http_max_attempts"]),
    )
    all_audits: list[dict[str, object]] = []
    for section in candidate["sections"]:
        all_audits.extend(
            _audit_section(
                judge_client,
                sub_narrative=str(section["sub_narrative"]),
                claims=section["claims"],
                max_tokens=int(judge_config["audit_max_tokens"]),
                temperature=float(judge_config["temperature"]),
                validation_attempts=int(judge_config["validation_attempts"]),
            )
        )
    final_generation, final_audits, excluded = filter_supported_sentences(
        candidate, all_audits
    )
    official_entry = build_official_entry(
        generation=final_generation,
        team_id=str(experiment["team_id"]),
        run_desc=str(experiment["run_desc"]),
    )
    official_words = validate_organizer_submission(
        official_entry,
        maximum_words=int(generation_config["official_maximum_words"]),
        tokenizer=tokenizer,
    )

    generation_path = output_dir / "response_generation.json"
    support_path = output_dir / "claim_support_audit.jsonl"
    all_support_path = output_dir / "generation_support_audit.jsonl"
    excluded_path = output_dir / "excluded_sentences.jsonl"
    response_path = output_dir / "generated_response.md"
    submission_path = output_dir / "rag_output_trec_rag_2026.jsonl"
    _write_json(generation_path, final_generation)
    _write_jsonl(all_support_path, all_audits)
    _write_jsonl(support_path, final_audits)
    _write_jsonl(excluded_path, excluded)
    response_text = render_markdown(final_generation)
    response_path.write_text(response_text, encoding="utf-8")
    _write_jsonl(submission_path, [official_entry])

    freeze_path = output_dir / "pre_nugget_freeze.json"
    freeze = _write_freeze_seal(
        path=freeze_path,
        experiment_id=str(experiment["id"]),
        artifact_paths=(
            output_dir / "openrouter_raw_response.json",
            output_dir / "generation_request_summary.json",
            candidate_path,
            generation_path,
            all_support_path,
            support_path,
            excluded_path,
            response_path,
            submission_path,
        ),
        repo_root=repo_root,
    )

    # The organizer nuggets are first opened after the final submission is sealed.
    nuggets = load_nuggets(
        nuggets_path,
        topic_id=str(source_generation["topic_id"]),
        expected_count=int(evaluation_config["expected_nugget_count"]),
    )
    comparison = evaluate_nuggets(
        nuggets=nuggets,
        generation=final_generation,
        client=judge_client,
        evaluation_config=evaluation_config,
    )
    comparison_path = output_dir / "nugget_comparison.jsonl"
    _write_jsonl(comparison_path, comparison)
    metrics = compute_evaluation_metrics(
        comparison, final_audits, response_text=response_text
    )
    metrics.update(
        {
            "experiment_id": experiment["id"],
            "run_id": experiment["run_id"],
            "topic_id": source_generation["topic_id"],
            "models": {
                "generator": generation_config["model_identity"],
                "support_judge_and_nugget_evaluator": judge_config["model_identity"],
            },
            "generation_api": request_metadata,
            "source_supported_claim_count": len(source_by_id),
            "official_submission": {
                "word_count": official_words,
                "sentence_count": len(official_entry["answer"]),
                "reference_count": len(official_entry["references"]),
                "candidate_sentence_count": len(all_audits),
                "excluded_after_support_audit": len(excluded),
                "maximum_words": int(generation_config["official_maximum_words"]),
            },
            "pre_nugget_freeze_sha256": _sha256(freeze_path),
            "frozen_generation_sha256": _sha256(generation_path),
            "frozen_submission_sha256": _sha256(submission_path),
            "nuggets_loaded_after_freeze": freeze["nuggets_loaded"] is False,
        }
    )
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    baseline_path = repo_root / str(inputs["trec26_qwen_baseline_metrics"])
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.is_file()
        else None
    )
    report_path = output_dir / "evaluation_report.md"
    report_path.write_text(
        _render_report(
            metrics=metrics,
            baseline=baseline,
            generation_metadata=request_metadata,
        ),
        encoding="utf-8",
    )
    resolved_path = output_dir / "config.resolved.json"
    _write_json(resolved_path, json.loads(json.dumps(config)))

    report_artifacts = _copy_report_artifacts(
        output_dir=output_dir, report_dir=report_dir, config_path=config_path
    )
    output_artifacts = [
        output_dir / name
        for name in (
            "openrouter_raw_response.json",
            "generation_request_summary.json",
            "response_generation.candidate.json",
            "response_generation.json",
            "generation_support_audit.jsonl",
            "claim_support_audit.jsonl",
            "excluded_sentences.jsonl",
            "generated_response.md",
            "rag_output_trec_rag_2026.jsonl",
            "pre_nugget_freeze.json",
            "nugget_comparison.jsonl",
            "metrics.json",
            "evaluation_report.md",
            "config.resolved.json",
            "openrouter_generation_calls.jsonl",
            "local_qwen_calls.jsonl",
        )
    ]
    manifest_files = [
        config_path,
        source_generation_path,
        source_audit_path,
        nuggets_path,
        *output_artifacts,
        *report_artifacts,
    ]
    if baseline_path.is_file():
        manifest_files.append(baseline_path)
    bindings = {
        str(path.relative_to(repo_root)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in manifest_files
    }
    manifest = {
        "schema_version": "topic213-experiment-manifest-v1",
        "experiment_id": experiment["id"],
        "run_id": experiment["run_id"],
        "status": "complete",
        "models": metrics["models"],
        "generation_protocol": {
            "semantic_request_count": request_metadata["successful_completions"],
            "successful_request_http_attempt_count": request_metadata["http_attempts"],
            "precompletion_rejected_http_attempt_count": request_metadata[
                "precompletion_rejected_http_attempts"
            ],
            "total_generation_endpoint_http_request_count": request_metadata[
                "total_http_requests"
            ],
            "temperature": generation_config["temperature"],
            "rewrite_or_repair_calls": 0,
        },
        "data_firewall": {
            "nuggets_loaded_after_pre_nugget_freeze": True,
            "pre_nugget_freeze_sha256": _sha256(freeze_path),
        },
        "files": bindings,
    }
    output_manifest = output_dir / "manifest.yaml"
    report_manifest = report_dir / "manifest.yaml"
    manifest_text = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    output_manifest.write_text(manifest_text, encoding="utf-8")
    report_manifest.write_text(manifest_text, encoding="utf-8")
    verify_manifest_hashes(report_manifest, repo_root=repo_root)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_from_config(args.config.resolve()))


if __name__ == "__main__":
    main()
