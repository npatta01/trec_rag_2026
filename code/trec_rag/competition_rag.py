"""Generate organizer-format TREC RAG answers from a sealed evidence handoff."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, Sequence
from urllib.parse import unquote

import requests
import yaml
from filelock import FileLock, Timeout as FileLockTimeout

from trec_rag.generation_handoff import (
    PROMPT_CONTRACT_VERSION,
    ClaimHint,
    GenerationHandoff,
    GenerationTopic,
    load_generation_handoff,
    render_generation_evidence,
    select_generation_topics,
)
from trec_rag.repo_env import find_repo_root, load_repo_env, shared_checkout_root


_SCHEMA_VERSION = "competition_rag_config_v2"
_SAFE_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
# strict_schema pins provider selection to schema-capable endpoints; json_object is the
# portable fallback; none sends no response_format at all.
STRUCTURED_OUTPUT_MODES = frozenset({"strict_schema", "json_object", "none"})
# The organizer answer cap. Kept here so validation, trimming, and post-run reporting all
# read one contract value instead of repeating the literal.
ANSWER_WORD_LIMIT = 1024
_MAX_REDACTION_NORMALIZATION_ROUNDS = 32
MAX_SEMANTIC_ATTEMPTS = 2
_CITATION_VALIDATION_CONTRACT_VERSION = "exact_hint_linked_docids_v1"

SYSTEM_PROMPT = """You are a selected-evidence RAG answer-generation agent. Use only the
provided selected passages and the user's task instructions. Treat claim hints as advisory.
Do not invent evidence, document identifiers, references, or source-specific claims."""

SELECTED_EVIDENCE_USER_PROMPT = """Answer the official narrative using only the frozen
selected retrieval evidence below. Selected passages are factual authority; canonical claim
hints are advisory and must be checked against their linked passage text.

Cover distinct answer-relevant evidence, tradeoffs, constraints, and uncertainty without
padding or repetition. Write each answer object as one self-contained sentence stating one
claim. The complete answer must never exceed 1,024 whitespace-separated words.

Give each answer object one to three unique raw ClimbMix docid strings in its citations,
ordered strongest support first. Do not use numeric citation indexes. Include a raw ClimbMix
docid in references only when an answer object cites it. Never cite a document unless its
selected passage fully supports that answer object.

Return one JSON object with exactly references and answer; no Markdown.

{evidence}

Restating the output contract: one JSON object with exactly references and answer; each answer
object is one self-contained sentence with one claim and one to three unique raw ClimbMix
docids in citations, never numeric indexes; never exceed 1,024 words."""

SEMANTIC_RETRY_INSTRUCTION = """

Previous completion failed local validation. Return a fresh answer that satisfies the JSON
schema and all evidence rules; do not describe or repair the previous completion. Citations
must be exact raw ClimbMix docid strings from the supplied evidence; never use numeric indexes.
If an answer sentence reuses an advisory claim hint, cite only docids from that hint's listed
evidence_ids, mapping each evidence_id to its [EVIDENCE ... docid=...] block."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError("YAML mapping keys must be hashable") from exc
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class RagGenerationConfig:
    schema_version: str
    handoff_manifest_path: Path
    output_path: Path
    work_dir: Path
    team_id: str
    run_id: str
    run_desc: str
    topic_ids: tuple[str, ...] | None
    concurrency: int
    resume: bool
    overwrite: bool
    provider: str
    api_base: str
    api_key_env: str
    model: str
    reasoning_effort: str
    structured_output: str
    temperature: float | None
    max_tokens: int
    timeout_seconds: float
    transport_max_attempts: int

    @property
    def resolved_work_dir(self) -> Path:
        return self.work_dir


def load_rag_generation_config(path: Path) -> RagGenerationConfig:
    """Load a fully specified, duplicate-key-safe competition RAG config."""
    config_path = Path(path).resolve()
    root_dir = find_repo_root(config_path.parent)
    raw = _load_yaml(config_path)
    config = _mapping(raw, "config")
    _reject_unknown(
        config,
        {"schema_version", "experiment", "submission", "inputs", "generation"},
        "config",
    )
    _require_fields(
        config,
        {"schema_version", "experiment", "submission", "inputs", "generation"},
        "config",
    )
    if _text(config, "schema_version", "config") != _SCHEMA_VERSION:
        raise ValueError(f"config.schema_version must be {_SCHEMA_VERSION}")

    experiment = _section(
        config,
        "experiment",
        {"id", "output_dir", "mode", "topic_ids"},
        required={"id", "output_dir", "mode"},
    )
    experiment_id = _text(experiment, "id", "experiment")
    if not _SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError("experiment.id must be a safe identifier")
    output_dir = _output_path(root_dir, _text(experiment, "output_dir", "experiment"))
    mode = _text(experiment, "mode", "experiment").lower()
    if mode not in {"create", "resume", "overwrite"}:
        raise ValueError("experiment.mode must be create, resume, or overwrite")

    submission = _section(config, "submission", {"team_id", "run_desc"})
    inputs = _section(
        config,
        "inputs",
        {"handoff_manifest"},
    )
    generation = _section(
        config,
        "generation",
        {
            "type",
            "api_base",
            "api_key_env",
            "model",
            "reasoning_effort",
            "structured_output",
            "temperature",
            "max_tokens",
            "timeout_seconds",
            "transport_max_attempts",
            "concurrency",
        },
        required={
            "type",
            "api_base",
            "api_key_env",
            "model",
            "reasoning_effort",
            "temperature",
            "max_tokens",
            "timeout_seconds",
            "transport_max_attempts",
            "concurrency",
        },
    )
    if _text(generation, "type", "generation").lower() != "openrouter":
        raise ValueError("generation.type must be openrouter")
    reasoning_effort = _text(generation, "reasoning_effort", "generation").lower()
    if reasoning_effort not in _REASONING_EFFORTS:
        raise ValueError("generation.reasoning_effort is unsupported")
    structured_output = (
        _optional_text(generation, "structured_output", "generation") or "strict_schema"
    )
    if structured_output not in STRUCTURED_OUTPUT_MODES:
        raise ValueError("generation.structured_output is unsupported")

    return RagGenerationConfig(
        schema_version=_SCHEMA_VERSION,
        handoff_manifest_path=_input_path(
            root_dir, _text(inputs, "handoff_manifest", "inputs")
        ),
        output_path=output_dir / "rag_output_trec_rag_2026.jsonl",
        work_dir=output_dir / "work",
        team_id=_text(submission, "team_id", "submission"),
        run_id=experiment_id,
        run_desc=_text(submission, "run_desc", "submission"),
        topic_ids=_topic_ids(experiment, owner="experiment"),
        concurrency=_positive_int(generation, "concurrency", "generation"),
        resume=mode == "resume",
        overwrite=mode == "overwrite",
        provider=_text(generation, "type", "generation").lower(),
        api_base=_text(generation, "api_base", "generation"),
        api_key_env=_text(generation, "api_key_env", "generation"),
        model=_text(generation, "model", "generation"),
        reasoning_effort=reasoning_effort,
        structured_output=structured_output,
        temperature=_optional_finite_float(generation, "temperature", "generation"),
        max_tokens=_positive_int(generation, "max_tokens", "generation"),
        timeout_seconds=_positive_float(generation, "timeout_seconds", "generation"),
        transport_max_attempts=_positive_int(
            generation, "transport_max_attempts", "generation"
        ),
    )


def _load_yaml(path: Path) -> object:
    try:
        return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML config: {path}") from exc


def _mapping(value: object, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{owner} must be a mapping")
    return value


def _section(
    config: dict[str, Any],
    name: str,
    allowed: set[str],
    *,
    required: set[str] | None = None,
) -> dict[str, Any]:
    section = _mapping(config.get(name), name)
    _reject_unknown(section, allowed, name)
    _require_fields(section, required or allowed, name)
    return section


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], owner: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{owner} has unknown field(s): {', '.join(unknown)}")


def _require_fields(mapping: dict[str, Any], required: set[str], owner: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"{owner} has missing field(s): {', '.join(missing)}")


def _text(mapping: dict[str, Any], key: str, owner: str) -> str:
    value = mapping.get(key)
    if not _nonempty_text(value):
        raise ValueError(f"{owner}.{key} must be non-empty text")
    return value.strip()


def _optional_text(mapping: dict[str, Any], key: str, owner: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not _nonempty_text(value):
        raise ValueError(f"{owner}.{key} must be non-empty text or null")
    return value.strip()


def _topic_ids(section: dict[str, Any], *, owner: str) -> tuple[str, ...] | None:
    value = section.get("topic_ids")
    if value is None and "topic_ids" not in section:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"{owner}.topic_ids must be a non-empty list")
    normalized: list[str] = []
    seen: set[str] = set()
    for topic_id in value:
        if not _nonempty_text(topic_id):
            raise ValueError(f"{owner}.topic_ids must contain non-empty text")
        clean = topic_id.strip()
        if clean in seen:
            raise ValueError(f"duplicate topic ID: {clean}")
        seen.add(clean)
        normalized.append(clean)
    return tuple(normalized)


def _positive_int(mapping: dict[str, Any], key: str, owner: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{owner}.{key} must be a positive integer")
    return value


def _positive_float(mapping: dict[str, Any], key: str, owner: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{owner}.{key} must be a positive number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{owner}.{key} must be a positive number")
    return parsed


def _optional_finite_float(mapping: dict[str, Any], key: str, owner: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{owner}.{key} must be a finite number or null")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{owner}.{key} must be a finite number or null")
    return parsed


def _input_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    active_path = root_dir / path
    if active_path.exists():
        return active_path
    shared_root = shared_checkout_root(root_dir)
    if shared_root is not None:
        shared_path = shared_root / path
        if shared_path.exists():
            return shared_path
    return active_path


def _output_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "experiment.output_dir must be a relative dedicated child of outputs/"
        )
    outputs_root = (root_dir / "outputs").resolve()
    resolved = (root_dir / path).resolve()
    if resolved == outputs_root or outputs_root not in resolved.parents:
        raise ValueError(
            "experiment.output_dir must be a relative dedicated child of outputs/"
        )
    return resolved


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def output_schema(citation_mode: str = "docid") -> dict[str, Any]:
    """Return the provider-safe schema; cross-field rules are validated locally."""
    if citation_mode not in {"docid", "index"}:
        raise ValueError(f"unsupported citation mode: {citation_mode}")
    citation_schema = (
        {"type": "string", "minLength": 1}
        if citation_mode == "docid"
        else {"type": "integer"}
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["references", "answer"],
        "properties": {
            "references": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "answer": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "citations"],
                    "properties": {
                        "text": {"type": "string"},
                        "citations": {
                            "type": "array",
                            "items": citation_schema,
                            "minItems": 1,
                            "maxItems": 3,
                        },
                    },
                },
            },
        },
    }


class JsonGenerator(Protocol):
    def complete_json(
        self,
        *,
        topic_id: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return one parsed completion and its raw provider response."""


class SemanticCompletionError(ValueError):
    """A successful provider response that cannot be accepted without repair."""

    def __init__(
        self, message: str, raw_response: object, *, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.retryable = retryable


class OpenRouterJsonGenerator:
    """One-completion OpenRouter client with retries only for transient transport."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        reasoning_effort: str,
        temperature: float | None,
        max_tokens: int,
        timeout_seconds: float,
        transport_max_attempts: int,
        structured_output: str = "strict_schema",
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is missing or empty")
        if structured_output not in STRUCTURED_OUTPUT_MODES:
            raise ValueError(f"unsupported structured_output mode: {structured_output}")
        self.api_base = api_base.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.transport_max_attempts = transport_max_attempts
        self.structured_output = structured_output

    def complete_json(
        self,
        *,
        topic_id: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del topic_id
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "reasoning": {"effort": self.reasoning_effort, "exclude": True},
        }
        if self.structured_output == "strict_schema":
            # require_parameters excludes any provider that cannot honour the schema. For some
            # models that excludes the vendor's own API and routes to a third party whose
            # grammar-constrained decoding may only treat the schema as a hint.
            request_body["provider"] = {"require_parameters": True}
            request_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "trec_rag_2026_answer",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        elif self.structured_output == "json_object":
            # The widely supported fallback. Shape is validated locally either way.
            request_body["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            request_body["temperature"] = self.temperature
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.api_base}/chat/completions"
        for attempt in range(1, self.transport_max_attempts + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=request_body,
                    timeout=(15, self.timeout_seconds),
                )
            except requests.RequestException as exc:
                if attempt == self.transport_max_attempts:
                    raise RuntimeError("OpenRouter generation transport failed") from exc
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            response_is_json, envelope, safe_response = _safe_http_response(
                response, self._api_key
            )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.transport_max_attempts:
                    raise SemanticCompletionError(
                        f"OpenRouter HTTP {response.status_code}; no repair call was made",
                        safe_response,
                    )
                time.sleep(_retry_delay(response.headers.get("Retry-After"), attempt))
                continue
            if response.status_code >= 400:
                raise SemanticCompletionError(
                    f"OpenRouter HTTP {response.status_code}; no repair call was made",
                    safe_response,
                )
            if not response_is_json:
                raise SemanticCompletionError(
                    "OpenRouter returned a non-JSON response; no repair call was made",
                    safe_response,
                    retryable=True,
                )
            if not isinstance(envelope, dict):
                raise SemanticCompletionError(
                    "OpenRouter response must be a JSON object; no repair call was made",
                    safe_response,
                    retryable=True,
                )
            # Checked outside the parse guard below, which catches ValueError and would
            # otherwise rewrite this as a generic malformed-completion error.
            # Reasoning tokens share the max_tokens budget, so truncation is a real risk, and
            # grammar-constrained decoding can close the JSON validly while the answer is cut
            # short, producing a record that passes every downstream check and scores badly.
            choices = envelope.get("choices")
            finish_reason = (
                choices[0].get("finish_reason")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict)
                else None
            )
            if finish_reason == "length":
                raise SemanticCompletionError(
                    "OpenRouter truncated the completion at max_tokens; raise "
                    "generation.max_tokens rather than accepting a shortened answer",
                    safe_response,
                )
            if finish_reason != "stop":
                # An allowlist, not a denylist. OpenRouter returns HTTP 200 with
                # finish_reason "error" and partial content, and "content_filter" or an absent
                # reason are equally unsafe to publish. Rejecting a topic is recoverable by
                # resuming; publishing a partial answer is not.
                raise SemanticCompletionError(
                    f"OpenRouter finish_reason was {finish_reason!r} rather than 'stop'; "
                    "no repair call was made",
                    safe_response,
                    retryable=finish_reason is None,
                )
            try:
                message = envelope["choices"][0]["message"]
                generated = parse_generated_json(_message_text(message.get("content")))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise SemanticCompletionError(
                    "OpenRouter returned a malformed semantic completion; no repair call was made",
                    safe_response,
                    retryable=True,
                ) from exc
            return generated, safe_response
        raise AssertionError("unreachable OpenRouter retry state")


def _safe_http_response(
    response: requests.Response, api_key: str
) -> tuple[bool, object | None, object]:
    try:
        envelope = response.json()
    except ValueError:
        body_text = response.text
        body_bytes = body_text.encode("utf-8")
        return False, None, {
            "http_status": response.status_code,
            "body_omitted": True,
            "body_utf8_byte_length": len(body_bytes),
            "body_utf8_sha256": sha256(body_bytes).hexdigest(),
        }

    safe_envelope = _redact(envelope, (api_key,))
    if response.status_code >= 400:
        return True, envelope, {
            "http_status": response.status_code,
            "envelope": safe_envelope,
        }
    return True, envelope, safe_envelope


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 30.0)
        except ValueError:
            pass
    return float(min(2 ** (attempt - 1), 8))


def _message_text(content: object) -> str:
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "output_text"}
            and isinstance(item.get("text"), str)
        ]
        if parts:
            return "\n".join(parts).strip()
    raise ValueError("completion content is not nonempty text")


def render_prompt(topic: GenerationTopic) -> str:
    """Render the one pinned writer prompt from a validated evidence topic."""
    if not isinstance(topic, GenerationTopic):
        raise TypeError("topic must be GenerationTopic")
    return SELECTED_EVIDENCE_USER_PROMPT.format(
        evidence=render_generation_evidence(topic)
    )


def parse_generated_json(text: str) -> dict[str, Any]:
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text.strip(), re.DOTALL)
    try:
        value = json.loads(fence.group(1) if fence else text)
    except json.JSONDecodeError as exc:
        raise ValueError("generation is not one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("generation is not one JSON object")
    return value


def _metadata(
    *,
    topic_id: str,
    narrative: str,
    team_id: str,
    run_id: str,
    run_desc: str,
) -> dict[str, str]:
    return {
        "team_id": team_id,
        "narrative_id": topic_id,
        "narrative": narrative,
        "run_id": run_id,
        "run_desc": run_desc,
    }


def build_submission_record(
    generated: dict[str, Any],
    *,
    topic_id: str,
    narrative: str,
    team_id: str,
    run_id: str,
    run_desc: str,
) -> dict[str, Any]:
    if set(generated) != {"references", "answer"}:
        raise ValueError(f"{topic_id}: generated root must contain exactly references and answer")
    return {
        "metadata": _metadata(
            topic_id=topic_id,
            narrative=narrative,
            team_id=team_id,
            run_id=run_id,
            run_desc=run_desc,
        ),
        "references": generated["references"],
        "answer": generated["answer"],
    }


def validate_submission_record(
    record: dict[str, Any],
    *,
    topic_id: str,
    narrative: str,
    allowed_docids: list[str],
    team_id: str,
    run_id: str,
    run_desc: str,
) -> None:
    expected_metadata = _metadata(
        topic_id=topic_id,
        narrative=narrative,
        team_id=team_id,
        run_id=run_id,
        run_desc=run_desc,
    )
    metadata = record.get("metadata")
    if (
        set(record) != {"metadata", "references", "answer"}
        or not isinstance(metadata, dict)
        or any(metadata.get(key) != value for key, value in expected_metadata.items())
    ):
        raise ValueError(f"{topic_id}: invalid root object or metadata")
    references = record.get("references")
    if (
        not isinstance(references, list)
        or not references
        or not all(isinstance(docid, str) and docid.strip() and docid == docid.strip() for docid in references)
    ):
        raise ValueError(f"{topic_id}: references must be nonempty docid strings")
    if len(references) != len(set(references)) or not set(references) <= set(allowed_docids):
        raise ValueError(
            f"{topic_id}: references are duplicated or outside the "
            "selected-evidence citation domain"
        )
    answer = record.get("answer")
    if not isinstance(answer, list) or not answer:
        raise ValueError(f"{topic_id}: answer must be a nonempty list")
    words = 0
    for index, item in enumerate(answer):
        if not isinstance(item, dict) or set(item) != {"text", "citations"}:
            raise ValueError(f"{topic_id}: answer[{index}] has invalid fields")
        text = item.get("text")
        citations = item.get("citations")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{topic_id}: answer[{index}] has empty text")
        words += len(text.split())
        if not isinstance(citations, list) or len(citations) > 3:
            raise ValueError(f"{topic_id}: answer[{index}] must have 0-3 citations")
        if any(
            not (
                (type(citation) is int and 0 <= citation < len(references))
                or (isinstance(citation, str) and citation in references)
            )
            for citation in citations
        ):
            raise ValueError(f"{topic_id}: answer[{index}] has an invalid citation")
    if words > ANSWER_WORD_LIMIT:
        raise ValueError(f"{topic_id}: answer exceeds {ANSWER_WORD_LIMIT:,} words")


def normalize_generated_record(
    record: dict[str, Any],
    *,
    allowed_docids: Sequence[str] | None = None,
    citation_mode: str = "index",
    ranked_docids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Rebuild ``references`` from the citations actually used, then re-index them.

    The organizer baseline validator requires every reference to be cited. Asking the model to
    satisfy that is the single largest source of first-attempt generation failures, and it also
    drives citation padding: to cover a long reference list the model must attach two or three
    citations to nearly every object, and the trailing ones frequently do not support the claim.

    Deriving the reference list from the citations instead makes the constraint true by
    construction. Answer text is never touched, and each citation keeps pointing at exactly the
    document it pointed at before, so no claim changes its evidence.
    """
    if citation_mode not in {"index", "docid"}:
        raise ValueError(f"unsupported citation mode: {citation_mode}")
    if ranked_docids is not None:
        if allowed_docids is not None and list(allowed_docids) != list(ranked_docids):
            raise ValueError("allowed_docids and ranked_docids disagree")
        allowed_docids = ranked_docids
    if citation_mode == "docid" and allowed_docids is None:
        raise ValueError("docid citation mode requires ranked_docids")

    references = record.get("references")
    answer = record.get("answer")
    # Runs before validate_submission_record, so it cannot assume a well-formed record. A model
    # that returns a malformed shape must produce a clear error here rather than a TypeError.
    if not isinstance(references, list) or not references:
        raise ValueError("generated references must be a nonempty list")
    if not isinstance(answer, list) or not answer:
        raise ValueError("generated answer must be a nonempty list")
    for index, item in enumerate(answer):
        if not isinstance(item, dict):
            raise ValueError(f"answer[{index}] is not an object")
        if not isinstance(item.get("text"), str):
            raise ValueError(f"answer[{index}] has no text string")
        if not isinstance(item.get("citations"), list):
            raise ValueError(f"answer[{index}] has no citations list")

    if allowed_docids is not None:
        allowed = set(allowed_docids)
        if not allowed or any(
            not isinstance(docid, str) or not docid or docid != docid.strip()
            for docid in allowed_docids
        ):
            raise ValueError("allowed_docids must contain exact nonempty docid strings")
        if not all(
            isinstance(docid, str) and docid and docid == docid.strip()
            for docid in references
        ):
            raise ValueError("references must be exact raw docid strings")
        foreign_references = [docid for docid in references if docid not in allowed]
        if foreign_references:
            raise ValueError(
                "references outside supplied ranked documents: "
                + ", ".join(foreign_references)
            )

        order: list[str] = []
        resolved: list[list[str]] = []
        for index, item in enumerate(answer):
            cites: list[str] = []
            for citation in item["citations"]:
                if (
                    not isinstance(citation, str)
                    or not citation
                    or citation != citation.strip()
                ):
                    raise ValueError(
                        f"answer[{index}] citations must be raw docid strings"
                    )
                if citation not in allowed:
                    raise ValueError(
                        f"answer[{index}] citation {citation!r} is outside supplied ranked documents"
                    )
                if citation not in cites:
                    cites.append(citation)
                if citation not in order:
                    order.append(citation)
            resolved.append(cites)
        if not order:
            raise ValueError("generated answer cites no valid reference")
        remap = {docid: index for index, docid in enumerate(order)}
        return {
            **record,
            "references": order,
            "answer": [
                {"text": item["text"], "citations": [remap[docid] for docid in cites]}
                for item, cites in zip(answer, resolved)
            ],
        }

    # Only prune uncited references. An out-of-range or non-integer citation is a model error
    # that validate_submission_record is responsible for reporting, so it must not be silently
    # dropped here: doing so would hide fabricated citation indexes behind a later, misleading
    # "must have 1-3 citations" failure.
    # A model may list the same docid at two positions. Collapse those to the first position so
    # two citations that name one document do not survive as a duplicate pair.
    canonical: dict[str, int] = {}
    for position, docid in enumerate(references):
        canonical.setdefault(str(docid), position)

    order: list[int] = []
    resolved: list[list[int]] = []
    for index, item in enumerate(answer):
        cites: list[int] = []
        for citation in item["citations"]:
            if type(citation) is not int or not 0 <= citation < len(references):
                raise ValueError(f"answer[{index}] has an invalid citation: {citation!r}")
            position = canonical[str(references[citation])]
            if position not in cites:
                cites.append(position)
            if position not in order:
                order.append(position)
        resolved.append(cites)
    if not order:
        raise ValueError("generated answer cites no valid reference")

    remap = {old: new for new, old in enumerate(order)}
    return {
        **record,
        "references": [references[old] for old in order],
        "answer": [
            {"text": item["text"], "citations": [remap[c] for c in cites]}
            for item, cites in zip(answer, resolved)
        ],
    }


def trim_to_word_limit(record: dict[str, Any], *, max_words: int = ANSWER_WORD_LIMIT) -> dict[str, Any]:
    """Drop trailing answer objects until the record fits the organizer word cap.

    Models track a running word budget poorly: answers land at 950 to 1000 words and tip over
    often enough that the cap is a leading cause of generation failure. Trimming the tail is
    strictly better than discarding the whole topic, and later objects are the least load-bearing
    because the answer is written most-important-first.

    Runs before ``normalize_generated_record`` so that references orphaned by the trim are then
    rebuilt away.
    """
    answer = record.get("answer")
    if not isinstance(answer, list) or not answer:
        raise ValueError("generated answer must be a nonempty list")
    for index, item in enumerate(answer):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError(f"answer[{index}] is not an object with text")
    kept: list[dict[str, Any]] = []
    words = 0
    for item in answer:
        length = len(item["text"].split())
        if words + length > max_words:
            break
        kept.append(item)
        words += length
    if not kept:
        raise ValueError("first answer object alone exceeds the word limit")
    if len(kept) == len(answer):
        return record
    return {**record, "answer": kept}


def _validate_generated_submission_record(
    record: dict[str, Any],
    *,
    topic_id: str,
    narrative: str,
    allowed_docids: list[str],
    team_id: str,
    run_id: str,
    run_desc: str,
) -> None:
    """Enforce this generator's strict organizer-baseline output profile."""
    validate_submission_record(
        record,
        topic_id=topic_id,
        narrative=narrative,
        allowed_docids=allowed_docids,
        team_id=team_id,
        run_id=run_id,
        run_desc=run_desc,
    )
    if set(record["metadata"]) != {
        "team_id",
        "narrative_id",
        "narrative",
        "run_id",
        "run_desc",
    }:
        raise ValueError(f"{topic_id}: generated metadata must contain exactly five fields")
    used: set[int] = set()
    for index, item in enumerate(record["answer"]):
        citations = item["citations"]
        if (
            not 1 <= len(citations) <= 3
            or any(type(citation) is not int for citation in citations)
            or len(citations) != len(set(citations))
        ):
            raise ValueError(
                f"{topic_id}: answer[{index}] must have 1-3 unique integer citations"
            )
        used.update(citations)
    if used != set(range(len(record["references"]))):
        raise ValueError(f"{topic_id}: generated answer has uncited references")


def _claim_match_key(text: str) -> str:
    """Conservatively match a generated sentence to a near-verbatim claim hint."""
    normalized = " ".join(text.split()).casefold()
    return normalized[:-1] if normalized.endswith(".") else normalized


def _validate_exact_hint_citations(
    record: dict[str, Any],
    *,
    topic: GenerationTopic,
) -> None:
    """Reject knowable hint-to-document citation mismatches without rewriting output."""
    hints_by_text: dict[str, list[ClaimHint]] = {}
    for hint in topic.claim_hints:
        hints_by_text.setdefault(_claim_match_key(hint.text), []).append(hint)

    evidence_by_id = {evidence.evidence_id: evidence for evidence in topic.evidence}
    references = record["references"]
    mismatches: list[str] = []
    for answer_index, item in enumerate(record["answer"]):
        matches = hints_by_text.get(_claim_match_key(item["text"]), [])
        if len(matches) != 1:
            # Do not guess when two hints normalize to the same sentence.
            continue
        hint = matches[0]
        linked_docids = tuple(
            dict.fromkeys(
                evidence_by_id[evidence_id].docid for evidence_id in hint.evidence_ids
            )
        )
        cited_docids = tuple(references[citation] for citation in item["citations"])
        unlinked_docids = tuple(
            docid for docid in cited_docids if docid not in linked_docids
        )
        if unlinked_docids:
            mismatches.append(
                f"answer[{answer_index}] claim_hint={hint.claim_id} "
                f"unlinked_docids={','.join(unlinked_docids)} "
                f"linked_docids={','.join(linked_docids)}"
            )
    if mismatches:
        raise ValueError(
            f"{topic.topic_id}: exact claim-hint citation mismatches: "
            + "; ".join(mismatches)
        )


def _safe_topic_name(topic_id: str) -> str:
    prefix = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in topic_id
    )
    return f"{prefix}-{sha256(topic_id.encode('utf-8')).hexdigest()[:8]}"


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    active = tuple(secret for secret in secrets if secret)
    if isinstance(value, str):
        return _redact_text(value, active)
    if isinstance(value, list):
        return [_redact(item, active) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, active) for item in value)
    if isinstance(value, dict):
        return {
            _redact(key, active) if isinstance(key, str) else key: _redact(item, active)
            for key, item in value.items()
        }
    return value


def _redact_text(value: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
        normalized = value
        seen: set[str] = set()
        for _ in range(_MAX_REDACTION_NORMALIZATION_ROUNDS):
            if normalized in seen:
                break
            seen.add(normalized)
            if secret in normalized:
                return "[REDACTED]"
            for decode in (unquote, _decode_unicode_escapes):
                normalized = decode(normalized)
                if secret in normalized:
                    return "[REDACTED]"
    return value


def _decode_unicode_escapes(value: str) -> str:
    decoded: list[str] = []
    offset = 0
    while offset < len(value):
        match = re.match(r"\\u([0-9A-Fa-f]{4})", value[offset:])
        if match is None:
            decoded.append(value[offset])
            offset += 1
            continue
        codepoint = int(match.group(1), 16)
        end = offset + 6
        if 0xD800 <= codepoint <= 0xDBFF:
            low = re.match(r"\\u([0-9A-Fa-f]{4})", value[end:])
            if low is not None:
                low_codepoint = int(low.group(1), 16)
                if 0xDC00 <= low_codepoint <= 0xDFFF:
                    codepoint = (
                        0x10000
                        + ((codepoint - 0xD800) << 10)
                        + (low_codepoint - 0xDC00)
                    )
                    end += 6
        decoded.append(chr(codepoint))
        offset = end
    return "".join(decoded)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, value: object, *, compact: bool = False) -> None:
    if compact:
        contents = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    else:
        contents = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(path, contents)


def _saved_record(
    path: Path,
    *,
    topic: GenerationTopic,
    config: RagGenerationConfig,
) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return None
        _validate_generated_submission_record(
            record,
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            allowed_docids=list(topic.citation_docids),
            team_id=config.team_id,
            run_id=config.run_id,
            run_desc=config.run_desc,
        )
        _validate_exact_hint_citations(record, topic=topic)
        return record
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _next_raw_attempt_number(raw_dir: Path, topic_name: str) -> int:
    """Return the next monotonic raw-attempt number for one topic."""
    if not raw_dir.exists():
        return 1
    pattern = re.compile(
        rf"^{re.escape(topic_name)}\.attempt-(?P<number>[1-9][0-9]*)(?:\.failed)?\.json$"
    )
    highest = 0
    for path in raw_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None:
            highest = max(highest, int(match.group("number")))
    return highest + 1


async def _generate_topic(
    *,
    topic: GenerationTopic,
    generator: JsonGenerator,
    config: RagGenerationConfig,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> tuple[str, str | None]:
    topic_id = topic.topic_id
    work_dir = config.resolved_work_dir
    topic_name = _safe_topic_name(topic_id)
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    secrets = (os.environ.get(config.api_key_env, ""),)
    next_attempt = _next_raw_attempt_number(raw_dir, topic_name)
    last_error: str | None = None
    for semantic_attempt in range(MAX_SEMANTIC_ATTEMPTS):
        attempt_number = next_attempt + semantic_attempt
        raw_path = raw_dir / f"{topic_name}.attempt-{attempt_number}.json"
        failed_path = raw_dir / f"{topic_name}.attempt-{attempt_number}.failed.json"
        raw_written = False
        try:
            user_prompt = render_prompt(topic)
            if semantic_attempt:
                user_prompt += SEMANTIC_RETRY_INSTRUCTION
            async with semaphore:
                completion = executor.submit(
                    partial(
                        generator.complete_json,
                        topic_id=topic_id,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        response_schema=output_schema(),
                    )
                )
                # Polling avoids relying on the event loop's cross-thread wakeup descriptor,
                # which is unavailable in some constrained runner environments.
                while not completion.done():
                    await asyncio.sleep(0.01)
                generated, raw_response = completion.result()
            _write_json(raw_path, _redact(raw_response, secrets))
            raw_written = True
            record = normalize_generated_record(
                build_submission_record(
                    generated,
                    topic_id=topic_id,
                    narrative=topic.narrative,
                    team_id=config.team_id,
                    run_id=config.run_id,
                    run_desc=config.run_desc,
                ),
                allowed_docids=list(topic.citation_docids),
            )
            _validate_generated_submission_record(
                record,
                topic_id=topic_id,
                narrative=topic.narrative,
                allowed_docids=list(topic.citation_docids),
                team_id=config.team_id,
                run_id=config.run_id,
                run_desc=config.run_desc,
            )
            _validate_exact_hint_citations(record, topic=topic)
            _write_json(work_dir / "rows" / f"{topic_name}.json", record, compact=True)
            error_path = work_dir / "errors" / f"{topic_name}.txt"
            if error_path.exists():
                error_path.unlink()
                _fsync_directory(error_path.parent)
            return topic_id, None
        except SemanticCompletionError as exc:
            _write_json(failed_path, _redact(exc.raw_response, secrets))
            last_error = _redact(f"{type(exc).__name__}: {exc}", secrets)
            if not exc.retryable:
                break
        except ValueError as exc:
            last_error = _redact(f"{type(exc).__name__}: {exc}", secrets)
            if not raw_written:
                _write_json(failed_path, {"error": last_error})
        except Exception as exc:
            last_error = _redact(f"{type(exc).__name__}: {exc}", secrets)
            if not raw_written:
                _write_json(failed_path, {"error": last_error})
            break

    if last_error is None:
        last_error = "RuntimeError: semantic attempt budget must be positive"
    error_path = work_dir / "errors" / f"{topic_name}.txt"
    _atomic_write_text(error_path, f"{last_error}\n")
    return topic_id, last_error


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    return first == second or first in second.parents or second in first.parents


def _validate_artifact_paths(config: RagGenerationConfig) -> None:
    inputs = {
        "handoff_manifest_path": config.handoff_manifest_path,
    }
    targets = {
        "output_path": config.output_path,
        "resolved_work_dir": config.resolved_work_dir,
    }
    for input_name, input_path in inputs.items():
        for target_name, target_path in targets.items():
            if _paths_overlap(input_path, target_path):
                raise ValueError(
                    f"generation input/output paths overlap: {input_name} and {target_name}"
                )
    if _paths_overlap(config.output_path, config.resolved_work_dir):
        raise ValueError("generation output and work paths overlap")


def _work_has_artifacts(work_dir: Path) -> bool:
    if work_dir.is_symlink() or (work_dir.exists() and not work_dir.is_dir()):
        return True
    return work_dir.is_dir() and next(work_dir.iterdir(), None) is not None


def _clear_generation_artifacts(config: RagGenerationConfig) -> None:
    work_dir = config.resolved_work_dir
    if work_dir.is_symlink() or (work_dir.exists() and not work_dir.is_dir()):
        work_dir.unlink()
        _fsync_directory(work_dir.parent)
    elif work_dir.exists():
        shutil.rmtree(work_dir)
        _fsync_directory(work_dir.parent)
    if config.output_path.exists() or config.output_path.is_symlink():
        config.output_path.unlink()
        _fsync_directory(config.output_path.parent)


def _generation_identity(
    config: RagGenerationConfig,
    handoff: GenerationHandoff,
    topics: Sequence[GenerationTopic],
) -> dict[str, Any]:
    """Return the settings that make already-generated rows comparable.

    Resuming reuses rows produced by an earlier invocation. Those rows are revalidated for
    shape, but shape cannot detect that they came from a different model or prompt, so a resume
    after any of these changed would publish one file containing answers from two systems.

    The sealed handoff and rendered prompt are included so replacing selected evidence between
    create and resume cannot silently mix answers grounded in different contexts.
    """
    return {
        # Version 6 also binds deterministic claim-hint citation validation.
        "identity_version": 6,
        "handoff_schema_version": handoff.schema_version,
        "handoff_manifest_sha256": handoff.manifest_sha256,
        "selected_topics": [
            {
                "topic_id": topic.topic_id,
                "context_sha256": topic.context_sha256,
                "prompt_sha256": sha256(render_prompt(topic).encode("utf-8")).hexdigest(),
            }
            for topic in topics
        ],
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "system_prompt_sha256": sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "response_schema_sha256": sha256(
            json.dumps(output_schema(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "semantic_attempt_policy": {
            "max_attempts": MAX_SEMANTIC_ATTEMPTS,
            "retryable_failures": ["local_validation", "malformed_semantic_completion"],
            "retry_instruction_sha256": sha256(
                SEMANTIC_RETRY_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
        },
        "citation_validation_contract_version": _CITATION_VALIDATION_CONTRACT_VERSION,
        "team_id": config.team_id,
        "run_id": config.run_id,
        "run_desc": config.run_desc,
        "model": config.model,
        "provider": config.provider,
        "api_base": config.api_base,
        "api_key_env": config.api_key_env,
        "reasoning_effort": config.reasoning_effort,
        "structured_output": config.structured_output,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout_seconds": config.timeout_seconds,
        "transport_max_attempts": config.transport_max_attempts,
        "concurrency": config.concurrency,
    }


def _enforce_generation_identity(
    config: RagGenerationConfig,
    work_dir: Path,
    handoff: GenerationHandoff,
    topics: Sequence[GenerationTopic],
) -> None:
    """Refuse to resume into rows generated under different settings.

    Fails closed. Rows that predate identity tracking, or that were recorded by an older
    identity version, cannot be shown to match the current settings, so they are rejected rather
    than adopted: adopting them would publish one file mixing two systems.
    """
    identity_path = work_dir / "generation_identity.json"
    identity = _generation_identity(config, handoff, topics)
    if identity_path.exists():
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded.get("identity_version") != identity["identity_version"]:
            raise ValueError(
                "existing generation rows were recorded by an older revision and cannot be "
                "verified against the current settings; use experiment.mode: overwrite or a "
                "new experiment.id"
            )
        if recorded != identity:
            changed = sorted(
                key for key in set(recorded) | set(identity)
                if recorded.get(key) != identity.get(key)
            )
            raise ValueError(
                "existing generation rows were produced under different settings "
                f"({', '.join(changed)}); use experiment.mode: overwrite or a new experiment.id"
            )
        return
    if _work_has_artifacts(work_dir):
        raise ValueError(
            "existing generation rows predate settings tracking and cannot be verified "
            "against the current settings; use experiment.mode: overwrite or a new "
            "experiment.id"
        )
    _write_json(identity_path, identity)


async def _run_generation_locked(
    config: RagGenerationConfig,
    generator: JsonGenerator,
    handoff: GenerationHandoff,
    topics: Sequence[GenerationTopic],
) -> None:
    work_dir = config.resolved_work_dir
    if config.overwrite:
        _clear_generation_artifacts(config)
    elif not config.resume and (
        config.output_path.exists() or _work_has_artifacts(work_dir)
    ):
        raise ValueError(
            "generation artifacts exist; set experiment.mode: resume or "
            "experiment.mode: overwrite"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    _enforce_generation_identity(config, work_dir, handoff, topics)

    rows_dir = work_dir / "rows"
    pending: list[GenerationTopic] = []
    for topic in topics:
        topic_id = topic.topic_id
        row_path = rows_dir / f"{_safe_topic_name(topic_id)}.json"
        saved = (
            _saved_record(
                row_path,
                topic=topic,
                config=config,
            )
            if config.resume
            else None
        )
        if saved is None:
            pending.append(topic)
    semaphore = asyncio.Semaphore(config.concurrency)
    with ThreadPoolExecutor(
        max_workers=config.concurrency,
        thread_name_prefix="trec-rag-generation",
    ) as executor:
        results = await asyncio.gather(
            *[
                _generate_topic(
                    topic=topic,
                    generator=generator,
                    config=config,
                    semaphore=semaphore,
                    executor=executor,
                )
                for topic in pending
            ]
        )
    failures = [(topic_id, error) for topic_id, error in results if error]
    if failures:
        raise RuntimeError(
            f"{len(failures)} topic(s) failed; inspect {work_dir / 'errors'} and rerun with "
            "experiment.mode: resume"
        )
    final_records: list[dict[str, Any]] = []
    for topic in topics:
        topic_id = topic.topic_id
        record = _saved_record(
            rows_dir / f"{_safe_topic_name(topic_id)}.json",
            topic=topic,
            config=config,
        )
        if record is None:
            raise RuntimeError(f"missing valid generated row for {topic_id}")
        final_records.append(record)
    _atomic_write_text(
        config.output_path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in final_records
        ),
    )


async def run_generation(
    config: RagGenerationConfig,
    generator: JsonGenerator,
    *,
    handoff: GenerationHandoff | None = None,
) -> None:
    """Generate missing topic rows and atomically publish the organizer JSONL."""
    _validate_artifact_paths(config)
    if handoff is None:
        handoff = load_generation_handoff(config.handoff_manifest_path)
    topics = select_generation_topics(handoff, config.topic_ids)
    lock_path = config.output_path.with_name(f".{config.output_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lock_path, timeout=0):
            await _run_generation_locked(config, generator, handoff, topics)
    except FileLockTimeout as exc:
        raise RuntimeError(
            f"generation is already active for {config.output_path}"
        ) from exc


def arguments(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Competition RAG YAML.")
    return parser.parse_args(argv).config


def main() -> None:
    try:
        config_path = arguments()
        config = load_rag_generation_config(config_path)
        handoff = load_generation_handoff(config.handoff_manifest_path)
        select_generation_topics(handoff, config.topic_ids)
        load_repo_env(find_repo_root(config_path.resolve().parent))
        generator = OpenRouterJsonGenerator(
            api_base=config.api_base,
            api_key=os.environ.get(config.api_key_env, ""),
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            structured_output=config.structured_output,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            transport_max_attempts=config.transport_max_attempts,
        )
        asyncio.run(run_generation(config, generator, handoff=handoff))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        raise SystemExit(f"error: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
