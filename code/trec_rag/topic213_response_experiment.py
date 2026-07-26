"""Generate and evaluate the Topic 213 response from the released passage packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import requests
import yaml

from trec_rag.repo_env import find_repo_root, load_repo_env


SCHEMA_VERSION = "topic213-response-experiment-v1"
PROMPT_VERSION = "topic213-full-passage-map-reduce-v1"
ALLOWED_NUGGET_STATUSES = {
    "supported",
    "partially_supported",
    "missing",
    "contradicted",
}
ALLOWED_SUPPORT_STATUSES = {
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
}
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_T = TypeVar("_T")
MAX_SUPPORTING_PASSAGES_PER_CLAIM = 8


MAP_SYSTEM_PROMPT = """You are an evidence extraction component. Use only the supplied verbatim passages.
Return JSON with a single `claims` array. Each claim must contain `text`, `sub_narrative`, and
`passage_ids`. Write concise atomic claims, preserve uncertainty or disagreement, cite at least one
supplied passage ID, and do not add outside knowledge. Cover distinct useful facts without repeating
paraphrases. Organizer evaluation nuggets are unavailable and must not be inferred or requested."""

REDUCE_SYSTEM_PROMPT = """You consolidate evidence claims for one research question. Return JSON with
a single `claims` array. Each claim must contain `text` and `source_claim_ids`. Merge only true
duplicates, retain concrete dates/names/causal distinctions, preserve source disagreement, and use
only supplied claim IDs. Do not add outside knowledge."""

ANSWER_SYSTEM_PROMPT = """You write one section of a cited research answer from a verified claim
ledger. Return JSON with a single `claims` array. Each item must contain `text` and
`evidence_claim_ids`. Every sentence must be a self-contained factual claim supported by at least one
supplied evidence claim. Prefer coverage and precision over rhetoric. Do not add introductions,
conclusions, uncited transitions, or outside knowledge."""

SUPPORT_AUDIT_SYSTEM_PROMPT = """Audit generated claims against their cited verbatim evidence.
Return JSON with an `assessments` array containing exactly one item per claim ID, with `claim_id`,
`status`, and `notes`. Status must be supported, partially_supported, unsupported, or contradicted.
Topical similarity is not support; the cited text must justify the claim. Treat conflicting or weak
sources explicitly."""

NUGGET_EVALUATION_SYSTEM_PROMPT = """Evaluate a frozen generated answer against organizer nuggets.
The nuggets are evaluation references, not assumed factual truth. Return JSON with an `assessments`
array containing exactly one item per nugget ID, with `nugget_id`, `status`, `matched_claim_ids`,
`evidence_claim_ids`, and `notes`. Status must be supported, partially_supported, missing, or
contradicted. `supported` requires both that the answer communicates the nugget and that cited release
evidence directly supports it. `partially_supported` also requires a matched answer claim and evidence.
Use `missing` when the answer does not communicate the nugget. Use `contradicted` only when a cited
answer claim explicitly states the opposite, and include that claim ID. Keyword overlap is insufficient.
Use only the supplied answer and release evidence; do not rely on outside historical knowledge. Flag
ambiguity or source disagreement."""


@dataclass(frozen=True)
class PassageUnit:
    passage_id: str
    document_id: str
    sub_narrative: str
    text: str


@dataclass(frozen=True)
class PassageCorpus:
    document_ids: tuple[str, ...]
    sub_narratives: tuple[str, ...]
    passages: tuple[PassageUnit, ...]
    evidence_assignment_count: int


class JsonCompletionClient(Protocol):
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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}: line {line_number} must be a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def load_passage_corpus(path: Path) -> PassageCorpus:
    rows = _read_jsonl(path)
    document_ids: list[str] = []
    labels: list[str] = []
    passages: list[PassageUnit] = []
    assignments = 0
    seen_documents: set[str] = set()

    for row_number, row in enumerate(rows, 1):
        document_id = _require_nonempty_string(row.get("document_id"), "document_id")
        if document_id in seen_documents:
            raise ValueError(f"duplicate document_id in passage artifact: {document_id}")
        seen_documents.add(document_id)
        document_ids.append(document_id)
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"document row {row_number} must have nonempty evidence")
        for evidence_number, item in enumerate(evidence, 1):
            if not isinstance(item, dict):
                raise ValueError(f"document row {row_number} evidence {evidence_number} is invalid")
            label = _require_nonempty_string(item.get("sub_narrative"), "sub_narrative")
            raw_passages = item.get("passages")
            if not isinstance(raw_passages, list) or not raw_passages:
                raise ValueError(f"{document_id}/{label} must have nonempty passages")
            assignments += 1
            if label not in labels:
                labels.append(label)
            for text in raw_passages:
                passage_text = _require_nonempty_string(text, "passage text")
                passages.append(
                    PassageUnit(
                        passage_id=f"P{len(passages) + 1:06d}",
                        document_id=document_id,
                        sub_narrative=label,
                        text=passage_text,
                    )
                )

    if not rows:
        raise ValueError("passage artifact is empty")
    return PassageCorpus(
        document_ids=tuple(document_ids),
        sub_narratives=tuple(labels),
        passages=tuple(passages),
        evidence_assignment_count=assignments,
    )


def batch_passages(
    passages: Sequence[PassageUnit],
    *,
    max_input_chars: int,
    max_passages: int,
) -> list[tuple[PassageUnit, ...]]:
    if max_input_chars <= 0 or max_passages <= 0:
        raise ValueError("passage batch limits must be positive")
    batches: list[tuple[PassageUnit, ...]] = []
    current: list[PassageUnit] = []
    current_chars = 0
    for passage in passages:
        cost = len(passage.text)
        if current and (current_chars + cost > max_input_chars or len(current) >= max_passages):
            batches.append(tuple(current))
            current = []
            current_chars = 0
        current.append(passage)
        current_chars += cost
    if current:
        batches.append(tuple(current))
    return batches


def load_release_manifest(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("release manifest must be a YAML mapping")
    return value


def validate_release_accounting(
    corpus: PassageCorpus,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    counts = manifest.get("population_counts")
    extraction = manifest.get("passage_extraction")
    if not isinstance(counts, Mapping) or not isinstance(extraction, Mapping):
        raise ValueError("release manifest lacks passage accounting")

    expected = {
        "passage_output_documents": len(corpus.document_ids),
        "passage_evidence_assignments": corpus.evidence_assignment_count,
        "verbatim_passages": len(corpus.passages),
    }
    for key, actual in expected.items():
        if counts.get(key) != actual:
            raise ValueError(f"release manifest {key}={counts.get(key)!r}, artifact has {actual}")

    attempted = int(counts.get("passage_model_calls_attempted", 0))
    valid = int(counts.get("passage_valid_responses", 0))
    failed = int(counts.get("passage_failed_responses", 0))
    failed_ids = extraction.get("failed_document_ids")
    if not isinstance(failed_ids, list) or any(not isinstance(item, str) for item in failed_ids):
        raise ValueError("failed_document_ids must be an array of strings")
    if attempted != valid + failed or valid != len(corpus.document_ids) or failed != len(failed_ids):
        raise ValueError("release passage call accounting is inconsistent")

    return {
        "documents_eligible": int(counts.get("passage_documents_eligible", attempted)),
        "document_calls_attempted": attempted,
        "documents_processed": valid,
        "documents_unavailable": failed,
        "failed_document_ids": list(failed_ids),
        "evidence_assignments_processed": corpus.evidence_assignment_count,
        "passages_processed": len(corpus.passages),
        "sub_narratives_processed": len(corpus.sub_narratives),
    }


def load_topic_narrative(path: Path, *, topic_id: str) -> str:
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"topics line {line_number} must contain qid and narrative")
        if parts[0] == topic_id:
            return _require_nonempty_string(parts[1], "topic narrative")
    raise ValueError(f"topic {topic_id} not found in {path}")


def load_nuggets(path: Path, *, topic_id: str, expected_count: int = 50) -> list[dict[str, object]]:
    topic_rows = [row for row in _read_jsonl(path) if str(row.get("qid")) == topic_id]
    if len(topic_rows) != 1:
        raise ValueError(f"expected exactly one nugget row for topic {topic_id}")
    raw_nuggets = topic_rows[0].get("nuggets")
    if not isinstance(raw_nuggets, list) or len(raw_nuggets) != expected_count:
        actual = len(raw_nuggets) if isinstance(raw_nuggets, list) else "invalid"
        raise ValueError(f"expected {expected_count} nuggets for topic {topic_id}, found {actual}")

    nuggets: list[dict[str, object]] = []
    for index, raw in enumerate(raw_nuggets, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"nugget {index} must be an object")
        importance = _require_nonempty_string(raw.get("importance"), "nugget importance")
        if importance not in {"vital", "okay"}:
            raise ValueError(f"nugget {index} has invalid importance: {importance}")
        nuggets.append(
            {
                "nugget_id": f"{topic_id}-N{index:03d}",
                "text": _require_nonempty_string(raw.get("text"), "nugget text"),
                "mapped_sub_narrative": _require_nonempty_string(
                    raw.get("mapped_sub_narrative"), "mapped_sub_narrative"
                ),
                "importance": importance,
                "source": raw.get("source"),
            }
        )
    return nuggets


class OpenAICompatibleJsonClient:
    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str,
        checkpoint_dir: Path,
        call_log_path: Path,
        timeout_seconds: float,
        max_attempts: int,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.checkpoint_dir = checkpoint_dir
        self.call_log_path = call_log_path
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.call_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, row: Mapping[str, object]) -> None:
        with self.call_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

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
        }
        canonical = json.dumps(request_body, ensure_ascii=False, sort_keys=True).encode("utf-8")
        request_hash = hashlib.sha256(canonical).hexdigest()
        checkpoint = self.checkpoint_dir / f"{stage}__{request_hash}.json"
        if checkpoint.exists():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if not isinstance(cached, dict):
                raise ValueError(f"invalid checkpoint: {checkpoint}")
            self._log({"stage": stage, "request_sha256": request_hash, "cache_hit": True})
            return cached

        url = f"{self.api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        current_request_body = request_body
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            content: str | None = None
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=current_request_body,
                    timeout=(10, self.timeout_seconds),
                )
                response.raise_for_status()
                envelope = response.json()
                content = envelope["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("completion content is not text")
                parsed = _parse_json_object(content)
                checkpoint.write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                usage = envelope.get("usage", {})
                self._log(
                    {
                        "stage": stage,
                        "request_sha256": request_hash,
                        "cache_hit": False,
                        "attempt": attempt,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "input_chars": sum(
                            len(str(message["content"]))
                            for message in current_request_body["messages"]
                        ),
                        "output_chars": len(content),
                        "usage": usage,
                    }
                )
                return parsed
            except (requests.RequestException, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                self._log(
                    {
                        "stage": stage,
                        "request_sha256": request_hash,
                        "cache_hit": False,
                        "attempt": attempt,
                        "error": str(exc),
                    }
                )
                if attempt < self.max_attempts:
                    if content is not None:
                        current_request_body = {
                            **request_body,
                            "messages": [
                                *request_body["messages"],
                                {
                                    "role": "user",
                                    "content": (
                                        "A prior response was invalid or truncated JSON. Return a materially "
                                        "shorter, complete JSON object that follows the requested keys."
                                    ),
                                },
                            ],
                        }
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"{stage} completion failed after {self.max_attempts} attempts") from last_error

    def complete_text(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Return a cached plain-text chat completion from the same local endpoint."""
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        canonical = json.dumps(request_body, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
        request_hash = hashlib.sha256(canonical).hexdigest()
        checkpoint = self.checkpoint_dir / f"{stage}__text__{request_hash}.txt"
        if checkpoint.exists():
            cached = checkpoint.read_text(encoding="utf-8")
            if not cached.strip():
                raise ValueError(f"invalid empty checkpoint: {checkpoint}")
            self._log({"stage": stage, "request_sha256": request_hash, "cache_hit": True})
            return cached

        url = f"{self.api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
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
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("completion content is not nonempty text")
                checkpoint.write_text(content, encoding="utf-8")
                self._log(
                    {
                        "stage": stage,
                        "request_sha256": request_hash,
                        "cache_hit": False,
                        "attempt": attempt,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "input_chars": sum(
                            len(str(message["content"]))
                            for message in request_body["messages"]
                        ),
                        "output_chars": len(content),
                        "usage": envelope.get("usage", {}),
                    }
                )
                return content
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                self._log(
                    {
                        "stage": stage,
                        "request_sha256": request_hash,
                        "cache_hit": False,
                        "attempt": attempt,
                        "error": str(exc),
                    }
                )
                if attempt < self.max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"{stage} completion failed after {self.max_attempts} attempts") from last_error


def _parse_json_object(content: str) -> dict[str, object]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("completion did not contain a JSON object")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("completion JSON must be an object")
    return parsed


def _complete_validated(
    client: JsonCompletionClient,
    *,
    stage: str,
    system_prompt: str,
    payload: Mapping[str, object],
    max_tokens: int,
    temperature: float,
    validator: Callable[[Mapping[str, object]], _T],
    validation_attempts: int,
) -> _T:
    current_payload = dict(payload)
    last_error: Exception | None = None
    for validation_attempt in range(1, validation_attempts + 1):
        result = client.complete_json(
            stage=stage if validation_attempt == 1 else f"{stage}_repair_{validation_attempt}",
            system_prompt=system_prompt,
            payload=current_payload,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            return validator(result)
        except ValueError as exc:
            last_error = exc
            current_payload = {
                **payload,
                "repair_instruction": (
                    f"The prior JSON failed validation: {exc}. Return a complete corrected JSON object."
                ),
                "invalid_prior_response": result,
            }
    raise RuntimeError(f"{stage} returned invalid structured output") from last_error


def _claim_rows(value: Mapping[str, object], *, label: str) -> list[Mapping[str, object]]:
    rows = value.get("claims")
    if not isinstance(rows, list):
        raise ValueError(f"{label}.claims must be an array")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label}.claims items must be objects")
    return list(rows)


def _map_batch(
    client: JsonCompletionClient,
    *,
    batch: Sequence[PassageUnit],
    narrative: str,
    allowed_labels: Sequence[str],
    max_claims: int,
    max_tokens: int,
    temperature: float,
    validation_attempts: int,
) -> list[dict[str, object]]:
    lookup = {item.passage_id: item for item in batch}
    payload = {
        "task": "Extract atomic answer-worthy claims from every supplied passage.",
        "topic_narrative": narrative,
        "allowed_sub_narratives": list(allowed_labels),
        "maximum_claims": max_claims,
        "passages": [
            {
                "passage_id": item.passage_id,
                "document_id": item.document_id,
                "sub_narrative": item.sub_narrative,
                "text": item.text,
            }
            for item in batch
        ],
    }

    def validate(value: Mapping[str, object]) -> list[dict[str, object]]:
        rows = _claim_rows(value, label="map")
        if len(rows) > max_claims:
            raise ValueError(f"map returned more than {max_claims} claims")
        result: list[dict[str, object]] = []
        for row in rows:
            text = _require_nonempty_string(row.get("text"), "map claim text")
            sub_narrative = _normalize_sub_narrative(
                row.get("sub_narrative"), allowed_labels
            )
            raw_ids = row.get("passage_ids")
            if not isinstance(raw_ids, list) or not raw_ids:
                raise ValueError("map claim passage_ids must be nonempty")
            passage_ids = list(dict.fromkeys(map(str, raw_ids)))
            valid_passage_ids = [
                item
                for item in passage_ids
                if item in lookup and lookup[item].sub_narrative == sub_narrative
            ]
            if not valid_passage_ids:
                continue
            claim = _claim_from_passages(
                text=text,
                sub_narrative=sub_narrative,
                passage_ids=valid_passage_ids,
                passage_lookup=lookup,
            )
            discarded = [item for item in passage_ids if item not in valid_passage_ids]
            if discarded:
                claim["discarded_out_of_batch_passage_ids"] = discarded
            result.append(claim)
        return result

    return _complete_validated(
        client,
        stage="map_evidence",
        system_prompt=MAP_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
        validator=validate,
        validation_attempts=validation_attempts,
    )


def _claim_from_passages(
    *,
    text: str,
    sub_narrative: str,
    passage_ids: Sequence[str],
    passage_lookup: Mapping[str, PassageUnit],
) -> dict[str, object]:
    evidence = [passage_lookup[item] for item in passage_ids]
    return {
        "text": text,
        "sub_narrative": sub_narrative,
        "passage_ids": list(passage_ids),
        "document_ids": list(dict.fromkeys(item.document_id for item in evidence)),
        "supporting_passages": [
            {
                "passage_id": item.passage_id,
                "document_id": item.document_id,
                "text": item.text,
            }
            for item in evidence
        ],
    }


def _batch_claims(
    claims: Sequence[Mapping[str, object]], *, max_input_chars: int, max_claims: int
) -> list[list[Mapping[str, object]]]:
    batches: list[list[Mapping[str, object]]] = []
    current: list[Mapping[str, object]] = []
    current_chars = 0
    for claim in claims:
        cost = len(str(claim.get("text", ""))) + sum(
            len(str(document_id)) for document_id in claim.get("document_ids", [])
        )
        if current and (current_chars + cost > max_input_chars or len(current) >= max_claims):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(claim)
        current_chars += cost
    if current:
        batches.append(current)
    return batches


def _reduce_claim_batch(
    client: JsonCompletionClient,
    *,
    sub_narrative: str,
    claims: Sequence[Mapping[str, object]],
    max_output_claims: int,
    max_tokens: int,
    temperature: float,
    validation_attempts: int,
) -> list[dict[str, object]]:
    lookup = {str(item["claim_id"]): item for item in claims}
    payload = {
        "task": "Consolidate these claims while retaining distinct supported facts.",
        "sub_narrative": sub_narrative,
        "maximum_output_claims": max_output_claims,
        "claims": [
            {"claim_id": key, "text": value["text"], "document_ids": value["document_ids"]}
            for key, value in lookup.items()
        ],
    }

    def validate(value: Mapping[str, object]) -> list[dict[str, object]]:
        rows = _claim_rows(value, label="reduce")
        if not rows:
            raise ValueError("reduce must retain at least one claim")
        if len(rows) > max_output_claims:
            raise ValueError(f"reduce returned more than {max_output_claims} claims")
        result: list[dict[str, object]] = []
        for row in rows:
            text = _require_nonempty_string(row.get("text"), "reduced claim text")
            raw_ids = row.get("source_claim_ids")
            if not isinstance(raw_ids, list) or not raw_ids:
                raise ValueError("reduced source_claim_ids must be nonempty")
            requested_source_ids = list(dict.fromkeys(map(str, raw_ids)))
            source_ids = [item for item in requested_source_ids if item in lookup]
            if not source_ids:
                continue
            sources = [lookup[item] for item in source_ids]
            all_passage_rows = _unique_dict_rows(
                row
                for source in sources
                for row in source.get("supporting_passages", [])
                if isinstance(row, Mapping)
            )
            passage_rows = _select_supporting_passages(all_passage_rows)
            source_passage_ids = list(
                dict.fromkeys(
                    str(passage_id)
                    for source in sources
                    for passage_id in source.get("source_passage_ids", source.get("passage_ids", []))
                )
            )
            reduced_claim = {
                    "text": text,
                    "sub_narrative": sub_narrative,
                    "source_claim_ids": source_ids,
                    "map_claim_ids": list(
                        dict.fromkeys(
                            claim_id
                            for source in sources
                            for claim_id in source.get("map_claim_ids", [source["claim_id"]])
                        )
                    ),
                    "passage_ids": [str(row["passage_id"]) for row in passage_rows],
                    "source_passage_ids": source_passage_ids,
                    "source_passage_count": len(source_passage_ids),
                    "document_ids": list(
                        dict.fromkeys(str(row["document_id"]) for row in passage_rows)
                    ),
                    "supporting_passages": passage_rows,
                }
            discarded = [item for item in requested_source_ids if item not in lookup]
            if discarded:
                reduced_claim["discarded_out_of_batch_source_claim_ids"] = discarded
            result.append(reduced_claim)
        if not result:
            source = next(iter(lookup.values()))
            result.append(
                {
                    **dict(source),
                    "source_claim_ids": [str(source["claim_id"])],
                    "map_claim_ids": list(source.get("map_claim_ids", [source["claim_id"]])),
                }
            )
        return result

    return _complete_validated(
        client,
        stage="reduce_evidence",
        system_prompt=REDUCE_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
        validator=validate,
        validation_attempts=validation_attempts,
    )


def _unique_dict_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(dict(row))
    return result


def _select_supporting_passages(
    rows: Sequence[Mapping[str, object]],
    *,
    limit: int = MAX_SUPPORTING_PASSAGES_PER_CLAIM,
) -> list[dict[str, object]]:
    unique = _unique_dict_rows(rows)
    selected: list[dict[str, object]] = []
    selected_keys: set[str] = set()
    seen_documents: set[str] = set()
    for row in unique:
        document_id = str(row.get("document_id", ""))
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if document_id not in seen_documents:
            selected.append(row)
            selected_keys.add(key)
            seen_documents.add(document_id)
            if len(selected) == limit:
                return selected
    for row in unique:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)
            if len(selected) == limit:
                break
    return selected


def _consolidate_label_claims(
    client: JsonCompletionClient,
    *,
    sub_narrative: str,
    claims: Sequence[dict[str, object]],
    max_input_chars: int,
    max_claims_per_batch: int,
    max_output_claims: int,
    max_tokens: int,
    temperature: float,
    validation_attempts: int,
) -> list[dict[str, object]]:
    if not claims:
        return []
    current = [dict(item) for item in claims]
    for round_number in range(1, 12):
        batches = _batch_claims(
            current,
            max_input_chars=max_input_chars,
            max_claims=max_claims_per_batch,
        )
        reduced: list[dict[str, object]] = []
        for batch in batches:
            output = _reduce_claim_batch(
                client,
                sub_narrative=sub_narrative,
                claims=batch,
                max_output_claims=max_output_claims,
                max_tokens=max_tokens,
                temperature=temperature,
                validation_attempts=validation_attempts,
            )
            for item in output:
                item["claim_id"] = f"R{round_number:02d}-{len(reduced) + 1:04d}"
                reduced.append(item)
        if len(batches) == 1:
            return reduced
        if len(reduced) >= len(current):
            raise RuntimeError("evidence reduction did not shrink a multi-batch claim set")
        current = reduced
    raise RuntimeError("evidence reduction exceeded maximum rounds")


def _generate_section(
    client: JsonCompletionClient,
    *,
    narrative: str,
    sub_narrative: str,
    evidence_claims: Sequence[Mapping[str, object]],
    max_answer_claims: int,
    max_tokens: int,
    temperature: float,
    validation_attempts: int,
) -> list[dict[str, object]]:
    if not evidence_claims:
        return []
    lookup = {str(item["claim_id"]): item for item in evidence_claims}
    payload = {
        "task": "Write a concise answer section as separately cited factual sentences.",
        "topic_narrative": narrative,
        "sub_narrative": sub_narrative,
        "maximum_answer_claims": max_answer_claims,
        "evidence_claims": [
            {
                "evidence_claim_id": key,
                "text": value["text"],
                "document_ids": value["document_ids"],
            }
            for key, value in lookup.items()
        ],
    }

    def validate(value: Mapping[str, object]) -> list[dict[str, object]]:
        rows = _claim_rows(value, label="answer")
        if not rows or len(rows) > max_answer_claims:
            raise ValueError(f"answer must contain 1-{max_answer_claims} claims")
        result: list[dict[str, object]] = []
        for row in rows:
            text = _require_nonempty_string(row.get("text"), "answer claim text")
            raw_ids = row.get("evidence_claim_ids")
            if not isinstance(raw_ids, list) or not raw_ids:
                raise ValueError("answer evidence_claim_ids must be nonempty")
            ids = list(dict.fromkeys(map(str, raw_ids)))
            if any(item not in lookup for item in ids):
                raise ValueError("answer cites an unknown evidence claim")
            sources = [lookup[item] for item in ids]
            all_support = _unique_dict_rows(
                row
                for source in sources
                for row in source.get("supporting_passages", [])
                if isinstance(row, Mapping)
            )
            support = _select_supporting_passages(all_support)
            result.append(
                {
                    "text": text,
                    "evidence_claim_ids": ids,
                    "document_ids": list(
                        dict.fromkeys(str(row["document_id"]) for row in support)
                    ),
                    "supporting_passages": support,
                    "available_supporting_passage_count": len(all_support),
                }
            )
        return result

    return _complete_validated(
        client,
        stage="generate_section",
        system_prompt=ANSWER_SYSTEM_PROMPT,
        payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
        validator=validate,
        validation_attempts=validation_attempts,
    )


def _audit_section(
    client: JsonCompletionClient,
    *,
    sub_narrative: str,
    claims: Sequence[Mapping[str, object]],
    max_tokens: int,
    temperature: float,
    validation_attempts: int,
) -> list[dict[str, object]]:
    if not claims:
        return []
    result: list[dict[str, object]] = []
    for source in claims:
        claim_id = str(source["claim_id"])
        payload = {
            "task": "Judge this answer claim only against its cited passages.",
            "sub_narrative": sub_narrative,
            "claims": [
                {
                    "claim_id": claim_id,
                    "text": source["text"],
                    "cited_passages": source["supporting_passages"],
                }
            ],
        }

        def validate(value: Mapping[str, object]) -> dict[str, object]:
            rows = value.get("assessments")
            if (
                not isinstance(rows, list)
                or len(rows) != 1
                or not isinstance(rows[0], Mapping)
                or str(rows[0].get("claim_id")) != claim_id
            ):
                raise ValueError("support audit must assess the supplied claim exactly once")
            status = _require_nonempty_string(rows[0].get("status"), "support status")
            if status not in ALLOWED_SUPPORT_STATUSES:
                raise ValueError(f"invalid support status: {status}")
            return {
                "claim_id": claim_id,
                "sub_narrative": sub_narrative,
                "claim_text": source["text"],
                "status": status,
                "document_ids": source["document_ids"],
                "supporting_passages": source["supporting_passages"],
                "notes": str(rows[0].get("notes", "")).strip(),
            }

        result.append(
            _complete_validated(
                client,
                stage="audit_support",
                system_prompt=SUPPORT_AUDIT_SYSTEM_PROMPT,
                payload=payload,
                max_tokens=max_tokens,
                temperature=temperature,
                validator=validate,
                validation_attempts=validation_attempts,
            )
        )
    return result


def generate_response(
    *,
    topic_id: str,
    narrative: str,
    corpus: PassageCorpus,
    release_accounting: Mapping[str, object],
    client: JsonCompletionClient,
    generation_config: Mapping[str, object],
    experiment_id: str,
    run_id: str,
    source_metadata: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    max_input_chars = int(generation_config.get("max_input_chars", 7000))
    temperature = float(generation_config.get("temperature", 0.0))
    validation_attempts = int(generation_config.get("validation_attempts", 3))
    batches = [
        batch
        for label in corpus.sub_narratives
        for batch in batch_passages(
            [passage for passage in corpus.passages if passage.sub_narrative == label],
            max_input_chars=max_input_chars,
            max_passages=int(generation_config.get("max_passages_per_batch", 32)),
        )
    ]
    mapped_claims: list[dict[str, object]] = []
    for batch_number, batch in enumerate(batches, 1):
        claims = _map_batch(
            client,
            batch=batch,
            narrative=narrative,
            allowed_labels=(batch[0].sub_narrative,),
            max_claims=int(generation_config.get("map_max_claims", 14)),
            max_tokens=int(generation_config.get("map_max_tokens", 1000)),
            temperature=temperature,
            validation_attempts=validation_attempts,
        )
        for claim in claims:
            claim["claim_id"] = f"M{len(mapped_claims) + 1:05d}"
            claim["map_claim_ids"] = [claim["claim_id"]]
            claim["map_batch"] = batch_number
            mapped_claims.append(claim)

    claims_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for claim in mapped_claims:
        claims_by_label[str(claim["sub_narrative"])].append(claim)

    evidence_ledger: list[dict[str, object]] = []
    sections: list[dict[str, object]] = []
    support_audit: list[dict[str, object]] = []
    answer_claim_number = 0
    for label in corpus.sub_narratives:
        canonical = _consolidate_label_claims(
            client,
            sub_narrative=label,
            claims=claims_by_label[label],
            max_input_chars=int(generation_config.get("reduce_max_input_chars", 6500)),
            max_claims_per_batch=int(generation_config.get("reduce_max_claims_per_batch", 20)),
            max_output_claims=int(generation_config.get("reduce_max_output_claims", 8)),
            max_tokens=int(generation_config.get("reduce_max_tokens", 900)),
            temperature=temperature,
            validation_attempts=validation_attempts,
        )
        label_index = len(sections) + 1
        for index, claim in enumerate(canonical, 1):
            claim["claim_id"] = f"E{label_index:02d}-{index:03d}"
            evidence_ledger.append(claim)
        answer_claims = _generate_section(
            client,
            narrative=narrative,
            sub_narrative=label,
            evidence_claims=canonical,
            max_answer_claims=int(generation_config.get("max_answer_claims_per_section", 5)),
            max_tokens=int(generation_config.get("answer_max_tokens", 700)),
            temperature=temperature,
            validation_attempts=validation_attempts,
        )
        for claim in answer_claims:
            answer_claim_number += 1
            claim["claim_id"] = f"A{answer_claim_number:03d}"
        section = {"sub_narrative": label, "claims": answer_claims}
        sections.append(section)
        support_audit.extend(
            _audit_section(
                client,
                sub_narrative=label,
                claims=answer_claims,
                max_tokens=int(generation_config.get("audit_max_tokens", 700)),
                temperature=float(generation_config.get("audit_temperature", 0.0)),
                validation_attempts=validation_attempts,
            )
        )

    generation = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "topic_id": topic_id,
        "narrative": narrative,
        "model": client.model,
        "prompt_version": PROMPT_VERSION,
        "temperature": temperature,
        "context_policy": {
            "kind": "full_passage_map_reduce",
            "map_batch_partition": "sub_narrative_then_size",
            "passage_batches": len(batches),
            "max_input_chars": max_input_chars,
            "all_passages_processed_once_in_map_stage": True,
            "nuggets_available_during_generation": False,
        },
        "input_accounting": dict(release_accounting),
        "source_metadata": dict(source_metadata),
        "generation_config": dict(generation_config),
        "prompts": {
            "map_system": MAP_SYSTEM_PROMPT,
            "reduce_system": REDUCE_SYSTEM_PROMPT,
            "answer_system": ANSWER_SYSTEM_PROMPT,
            "support_audit_system": SUPPORT_AUDIT_SYSTEM_PROMPT,
        },
        "map_claim_count": len(mapped_claims),
        "evidence_ledger": evidence_ledger,
        "sections": sections,
    }
    return generation, support_audit


def _clean_heading(label: str) -> str:
    cleaned = label.strip().strip('"').strip()
    if cleaned.startswith("New: "):
        cleaned = cleaned[5:]
    return cleaned


def _normalize_sub_narrative(value: object, allowed_labels: Sequence[str]) -> str:
    candidate = _require_nonempty_string(value, "sub_narrative")
    if candidate in allowed_labels:
        return candidate
    if len(allowed_labels) == 1:
        return allowed_labels[0]
    normalized = _clean_heading(candidate).casefold()
    matches = [label for label in allowed_labels if _clean_heading(label).casefold() == normalized]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"unknown sub_narrative: {candidate}")


def render_generated_response(generation: Mapping[str, object]) -> str:
    lines = ["# Topic 213: Korean War", ""]
    sections = generation.get("sections")
    if not isinstance(sections, list):
        raise ValueError("generation sections must be an array")
    for section in sections:
        if not isinstance(section, Mapping):
            raise ValueError("generation section must be an object")
        lines.extend([f"## {_clean_heading(str(section['sub_narrative']))}", ""])
        claims = section.get("claims")
        if not isinstance(claims, list):
            raise ValueError("generation section claims must be an array")
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise ValueError("generation claim must be an object")
            document_ids = claim.get("document_ids")
            if not isinstance(document_ids, list) or not document_ids:
                raise ValueError("every generated claim must cite at least one document")
            lines.append(f"{str(claim['text']).strip()} [{'; '.join(map(str, document_ids))}]")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def evaluate_nuggets(
    *,
    nuggets: Sequence[Mapping[str, object]],
    generation: Mapping[str, object],
    client: JsonCompletionClient,
    evaluation_config: Mapping[str, object],
) -> list[dict[str, object]]:
    sections = generation.get("sections")
    ledger = generation.get("evidence_ledger")
    if not isinstance(sections, list) or not isinstance(ledger, list):
        raise ValueError("frozen generation lacks sections or evidence ledger")
    sections_by_label = {
        str(section["sub_narrative"]): section
        for section in sections
        if isinstance(section, Mapping)
    }
    ledger_by_id = {
        str(claim["claim_id"]): claim for claim in ledger if isinstance(claim, Mapping)
    }
    nuggets_by_label: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for nugget in nuggets:
        nuggets_by_label[str(nugget["mapped_sub_narrative"])].append(nugget)

    comparison_by_id: dict[str, dict[str, object]] = {}
    temperature = float(evaluation_config.get("temperature", 0.0))
    validation_attempts = int(evaluation_config.get("validation_attempts", 3))
    for label, label_nuggets in nuggets_by_label.items():
        if label not in sections_by_label:
            raise ValueError(f"nugget label has no generated section: {label}")
        section = sections_by_label[label]
        answer_claims = section.get("claims", [])
        if not isinstance(answer_claims, list):
            raise ValueError("section claims must be an array")
        answer_by_id = {
            str(claim["claim_id"]): claim
            for claim in answer_claims
            if isinstance(claim, Mapping)
        }
        label_ledger = {
            key: value
            for key, value in ledger_by_id.items()
            if str(value.get("sub_narrative")) == label
        }
        payload = {
            "task": "Compare every nugget against this frozen cited answer section.",
            "sub_narrative": label,
            "nuggets": [
                {"nugget_id": row["nugget_id"], "text": row["text"]}
                for row in label_nuggets
            ],
            "answer_claims": [
                {
                    "claim_id": key,
                    "text": value["text"],
                    "evidence_claim_ids": value["evidence_claim_ids"],
                }
                for key, value in answer_by_id.items()
            ],
            "release_evidence_claims": [
                {
                    "evidence_claim_id": key,
                    "text": value["text"],
                    "document_ids": value["document_ids"],
                }
                for key, value in label_ledger.items()
            ],
        }
        expected_ids = {str(row["nugget_id"]) for row in label_nuggets}

        def validate(value: Mapping[str, object]) -> list[dict[str, object]]:
            rows = value.get("assessments")
            if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
                raise ValueError("nugget assessments must be an array of objects")
            by_id = {str(row.get("nugget_id")): row for row in rows}
            if set(by_id) != expected_ids:
                raise ValueError("nugget evaluation must assess every supplied nugget exactly once")
            output: list[dict[str, object]] = []
            for nugget in label_nuggets:
                nugget_id = str(nugget["nugget_id"])
                assessment = by_id[nugget_id]
                status = _require_nonempty_string(assessment.get("status"), "nugget status")
                if status not in ALLOWED_NUGGET_STATUSES:
                    raise ValueError(f"invalid nugget status: {status}")
                matched_ids = _validated_id_list(
                    assessment.get("matched_claim_ids"), answer_by_id, "matched claim"
                )
                evidence_ids = _validated_id_list(
                    assessment.get("evidence_claim_ids"), label_ledger, "evidence claim"
                )
                if not evidence_ids:
                    evidence_ids = list(
                        dict.fromkeys(
                            str(evidence_id)
                            for claim_id in matched_ids
                            for evidence_id in answer_by_id[claim_id].get("evidence_claim_ids", [])
                            if str(evidence_id) in label_ledger
                        )
                    )
                normalization_notes: list[str] = []
                if status in {"supported", "partially_supported", "contradicted"} and not matched_ids:
                    normalization_notes.append(
                        f"Evaluator status {status} was normalized to missing because no answer claim was matched."
                    )
                    status = "missing"
                if status in {"supported", "partially_supported"} and not evidence_ids:
                    normalization_notes.append(
                        f"Evaluator status {status} was normalized to missing because no release evidence was linked."
                    )
                    status = "missing"
                evidence_claims = [label_ledger[item] for item in evidence_ids]
                supporting_passages = _unique_dict_rows(
                    row
                    for claim in evidence_claims
                    for row in claim.get("supporting_passages", [])
                    if isinstance(row, Mapping)
                )
                output.append(
                    {
                        **dict(nugget),
                        "status": status,
                        "matched_claim_ids": matched_ids,
                        "matched_answer_claims": [answer_by_id[item]["text"] for item in matched_ids],
                        "supporting_evidence_claim_ids": evidence_ids,
                        "supporting_document_ids": list(
                            dict.fromkeys(str(row["document_id"]) for row in supporting_passages)
                        ),
                        "supporting_passages": supporting_passages,
                        "evaluator_notes": " ".join(
                            [str(assessment.get("notes", "")).strip(), *normalization_notes]
                        ).strip(),
                    }
                )
            return output

        assessed = _complete_validated(
            client,
            stage="evaluate_nuggets",
            system_prompt=NUGGET_EVALUATION_SYSTEM_PROMPT,
            payload=payload,
            max_tokens=int(evaluation_config.get("max_tokens", 1400)),
            temperature=temperature,
            validator=validate,
            validation_attempts=validation_attempts,
        )
        for row in assessed:
            comparison_by_id[str(row["nugget_id"])] = row

    if set(comparison_by_id) != {str(row["nugget_id"]) for row in nuggets}:
        raise ValueError("final nugget comparison is incomplete")
    return [comparison_by_id[str(row["nugget_id"])] for row in nuggets]


def _validated_id_list(
    value: object, lookup: Mapping[str, object], label: str
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} IDs must be an array")
    ids = list(dict.fromkeys(map(str, value)))
    if any(item not in lookup for item in ids):
        raise ValueError(f"unknown {label} ID")
    return ids


def _coverage_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts = Counter(str(row["status"]) for row in rows)
    total = len(rows)
    return {
        "total": total,
        "supported": counts["supported"],
        "partially_supported": counts["partially_supported"],
        "missing": counts["missing"],
        "contradicted": counts["contradicted"],
        "strict_coverage": counts["supported"] / total if total else 0.0,
        "partial_credit_coverage": (
            (counts["supported"] + 0.5 * counts["partially_supported"]) / total
            if total
            else 0.0
        ),
    }


def _redundancy_metrics(claim_texts: Sequence[str]) -> dict[str, object]:
    normalized = [" ".join(_WORD_RE.findall(text.lower())) for text in claim_texts]
    exact_duplicates = len(normalized) - len(set(normalized))
    near_duplicates = 0
    for index, left in enumerate(normalized):
        left_terms = set(left.split())
        for right in normalized[index + 1 :]:
            right_terms = set(right.split())
            union = left_terms | right_terms
            if union and len(left_terms & right_terms) / len(union) >= 0.8:
                near_duplicates += 1
    return {
        "exact_duplicate_claim_count": exact_duplicates,
        "near_duplicate_claim_pair_count": near_duplicates,
        "near_duplicate_jaccard_threshold": 0.8,
    }


def compute_evaluation_metrics(
    comparison: Sequence[Mapping[str, object]],
    support_audit: Sequence[Mapping[str, object]],
    *,
    response_text: str,
) -> dict[str, object]:
    vital = [row for row in comparison if row.get("importance") == "vital"]
    okay = [row for row in comparison if row.get("importance") == "okay"]
    per_label: dict[str, dict[str, object]] = {}
    for label in dict.fromkeys(str(row["mapped_sub_narrative"]) for row in comparison):
        per_label[label] = _coverage_summary(
            [row for row in comparison if str(row["mapped_sub_narrative"]) == label]
        )
    cited = sum(bool(row.get("document_ids")) for row in support_audit)
    support_counts = Counter(str(row.get("status")) for row in support_audit)
    claim_texts = [str(row.get("claim_text", "")) for row in support_audit]
    return {
        "nuggets": {
            "all": _coverage_summary(list(comparison)),
            "vital": _coverage_summary(vital),
            "okay": _coverage_summary(okay),
            "per_sub_narrative": per_label,
        },
        "answer_claims": {
            "total": len(support_audit),
            "citation_coverage": cited / len(support_audit) if support_audit else 0.0,
            "supported": support_counts["supported"],
            "partially_supported": support_counts["partially_supported"],
            "unsupported_claim_count": support_counts["unsupported"],
            "contradicted_claim_count": support_counts["contradicted"],
        },
        "response": {
            "word_count": len(_WORD_RE.findall(response_text)),
            "character_count": len(response_text),
            **_redundancy_metrics(claim_texts),
        },
    }


def render_evaluation_report(
    *,
    generation: Mapping[str, object],
    metrics: Mapping[str, object],
) -> str:
    nugget_metrics = metrics["nuggets"]
    answer_metrics = metrics["answer_claims"]
    response_metrics = metrics["response"]
    accounting = generation["input_accounting"]
    assert isinstance(nugget_metrics, Mapping)
    assert isinstance(answer_metrics, Mapping)
    assert isinstance(response_metrics, Mapping)
    assert isinstance(accounting, Mapping)
    all_metrics = nugget_metrics["all"]
    vital_metrics = nugget_metrics["vital"]
    okay_metrics = nugget_metrics["okay"]
    assert isinstance(all_metrics, Mapping)
    assert isinstance(vital_metrics, Mapping)
    assert isinstance(okay_metrics, Mapping)
    lines = [
        "# Topic 213 response-generation evaluation",
        "",
        "## Run",
        "",
        f"- Experiment: `{generation['experiment_id']}`",
        f"- Run ID: `{generation['run_id']}`",
        f"- Model: `{generation['generation_config'].get('model_identity', generation['model'])}` "
        f"(served as `{generation['model']}`)",
        f"- vLLM: `{generation['generation_config'].get('vllm_version', 'not recorded')}`; "
        f"LiteLLM: `{generation['generation_config'].get('litellm_version', 'not recorded')}`",
        f"- Prompt version: `{generation['prompt_version']}`",
        f"- Available documents processed: {accounting['documents_processed']} of {accounting['documents_eligible']}",
        f"- Passages processed: {accounting['passages_processed']}",
        f"- Unavailable provider-failure documents: {', '.join(accounting['failed_document_ids'])}",
        "",
        "The answer was frozen before organizer nuggets were loaded for evaluation.",
        "",
        "## Aggregate results",
        "",
        "| Population | Total | Supported | Partial | Missing | Contradicted | Strict coverage | Partial-credit coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (("Vital", vital_metrics), ("Okay", okay_metrics), ("All", all_metrics)):
        lines.append(
            f"| {name} | {row['total']} | {row['supported']} | {row['partially_supported']} | "
            f"{row['missing']} | {row['contradicted']} | {float(row['strict_coverage']):.3f} | "
            f"{float(row['partial_credit_coverage']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Answer support",
            "",
            f"- Factual claims: {answer_metrics['total']}",
            f"- Citation coverage: {float(answer_metrics['citation_coverage']):.3f}",
            f"- Unsupported claims: {answer_metrics['unsupported_claim_count']}",
            f"- Contradicted claims: {answer_metrics['contradicted_claim_count']}",
            f"- Response length: {response_metrics['word_count']} words",
            f"- Exact duplicate claims: {response_metrics['exact_duplicate_claim_count']}",
            f"- Near-duplicate claim pairs: {response_metrics['near_duplicate_claim_pair_count']}",
            "",
            "## Per sub-narrative",
            "",
            "| Sub-narrative | Total | Supported | Partial | Missing | Contradicted | Strict coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    per_label = nugget_metrics["per_sub_narrative"]
    assert isinstance(per_label, Mapping)
    for label, row in per_label.items():
        assert isinstance(row, Mapping)
        lines.append(
            f"| {_clean_heading(str(label))} | {row['total']} | {row['supported']} | "
            f"{row['partially_supported']} | {row['missing']} | {row['contradicted']} | "
            f"{float(row['strict_coverage']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Strict coverage counts only fully supported nuggets. Partial-credit coverage assigns 0.5 to partially supported nuggets. The organizer nuggets are treated as evaluation references rather than unquestioned factual truth; source ambiguity remains visible in the row-level comparison.",
            "",
            "The full released passage packet was consumed during generation. Two eligible documents are absent because their original provider calls failed before inference, and this experiment does not fabricate or replace their evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve(repo_root: Path, value: object, label: str) -> Path:
    path = Path(_require_nonempty_string(value, label))
    return path if path.is_absolute() else repo_root / path


def _copy_config(config: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(config))


def run_from_config(config_path: Path) -> Path:
    repo_root = find_repo_root(config_path.parent)
    load_repo_env(repo_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a YAML mapping")
    experiment = config.get("experiment")
    inputs = config.get("inputs")
    generation_config = config.get("generation")
    evaluation_config = config.get("evaluation")
    if not all(isinstance(item, Mapping) for item in (experiment, inputs, generation_config, evaluation_config)):
        raise ValueError("config requires experiment, inputs, generation, and evaluation mappings")
    assert isinstance(experiment, Mapping)
    assert isinstance(inputs, Mapping)
    assert isinstance(generation_config, Mapping)
    assert isinstance(evaluation_config, Mapping)

    experiment_id = _require_nonempty_string(experiment.get("id"), "experiment.id")
    run_id = _require_nonempty_string(experiment.get("run_id"), "experiment.run_id")
    topic_id = str(experiment.get("topic_id", "213"))
    output_dir = _resolve(repo_root, experiment.get("output_dir"), "experiment.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    passages_path = _resolve(repo_root, inputs.get("passages"), "inputs.passages")
    manifest_path = _resolve(repo_root, inputs.get("release_manifest"), "inputs.release_manifest")
    topics_path = _resolve(repo_root, inputs.get("topics"), "inputs.topics")
    nuggets_path = _resolve(repo_root, inputs.get("nuggets"), "inputs.nuggets")
    for key, path in (
        ("passages_sha256", passages_path),
        ("release_manifest_sha256", manifest_path),
        ("topics_sha256", topics_path),
        ("nuggets_sha256", nuggets_path),
    ):
        expected_hash = inputs.get(key)
        if expected_hash and str(expected_hash) != _sha256(path):
            raise ValueError(f"{key} does not match {path}")
    corpus = load_passage_corpus(passages_path)
    release_manifest = load_release_manifest(manifest_path)
    accounting = validate_release_accounting(corpus, release_manifest)
    narrative = load_topic_narrative(topics_path, topic_id=topic_id)

    api_base_env = str(generation_config.get("api_base_env", "LITELLM_BASE_URL"))
    model_env = str(generation_config.get("model_env", "LITELLM_MODEL"))
    api_key_env = str(generation_config.get("api_key_env", "LITELLM_API_KEY"))
    api_base = os.environ.get(api_base_env, str(generation_config.get("api_base", "http://localhost:4000/v1")))
    model = os.environ.get(model_env, str(generation_config.get("model", "qwen-local")))
    api_key = os.environ.get(api_key_env, str(generation_config.get("api_key", "none")))
    client = OpenAICompatibleJsonClient(
        api_base=api_base,
        model=model,
        api_key=api_key,
        checkpoint_dir=output_dir / "checkpoints",
        call_log_path=output_dir / "llm_calls.jsonl",
        timeout_seconds=float(generation_config.get("timeout_seconds", 240)),
        max_attempts=int(generation_config.get("http_max_attempts", 4)),
    )

    source_metadata = {
        "passages_path": str(passages_path.relative_to(repo_root)),
        "passages_sha256": _sha256(passages_path),
        "passages_release_url": inputs.get("passages_release_url"),
        "release_manifest_path": str(manifest_path.relative_to(repo_root)),
        "release_manifest_sha256": _sha256(manifest_path),
        "topics_path": str(topics_path.relative_to(repo_root)),
        "topics_sha256": _sha256(topics_path),
        "nuggets_path": str(nuggets_path.relative_to(repo_root)),
        "nuggets_sha256": _sha256(nuggets_path),
    }
    generation, support_audit = generate_response(
        topic_id=topic_id,
        narrative=narrative,
        corpus=corpus,
        release_accounting=accounting,
        client=client,
        generation_config=generation_config,
        experiment_id=experiment_id,
        run_id=run_id,
        source_metadata=source_metadata,
    )

    # Freeze generation artifacts before the nugget file is read.
    generation_path = output_dir / "response_generation.json"
    response_path = output_dir / "generated_response.md"
    support_path = output_dir / "claim_support_audit.jsonl"
    _write_json(generation_path, generation)
    response_text = render_generated_response(generation)
    response_path.write_text(response_text, encoding="utf-8")
    _write_jsonl(support_path, support_audit)
    frozen_generation_sha256 = _sha256(generation_path)
    frozen_response_sha256 = _sha256(response_path)

    nuggets = load_nuggets(
        nuggets_path,
        topic_id=topic_id,
        expected_count=int(evaluation_config.get("expected_nugget_count", 50)),
    )
    comparison = evaluate_nuggets(
        nuggets=nuggets,
        generation=generation,
        client=client,
        evaluation_config=evaluation_config,
    )
    comparison_path = output_dir / "nugget_comparison.jsonl"
    _write_jsonl(comparison_path, comparison)
    metrics = compute_evaluation_metrics(comparison, support_audit, response_text=response_text)
    metrics.update(
        {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "topic_id": topic_id,
            "frozen_generation_sha256": frozen_generation_sha256,
            "frozen_response_sha256": frozen_response_sha256,
        }
    )
    _write_json(output_dir / "metrics.json", metrics)
    (output_dir / "evaluation_report.md").write_text(
        render_evaluation_report(generation=generation, metrics=metrics), encoding="utf-8"
    )
    _write_json(output_dir / "config.resolved.json", _copy_config(config))
    return output_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    output_dir = run_from_config(args.config.resolve())
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
