"""Generate organizer-format TREC RAG answers from a fixed retrieval run.

The command consumes a two-column topic TSV, a six-column TREC run, and a
JSONL/ZIP document collection. GPT Sol receives the ranked documents for one
topic in one request. The application injects deterministic metadata, validates
the complete organizer contract, and writes one compact JSON object per topic.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import math
import os
import re
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Protocol, TextIO

import requests
import yaml

from trec_rag.repo_env import find_repo_root, load_repo_env, shared_checkout_root


SYSTEM_PROMPT = """You are a reference-document RAG answer-generation agent. Use only the
provided reference documents and the user's task instructions.

No search, browsing, file, or other tools are available. Treat the numbered documents as the
complete retrieval context. Read the full set, compare evidence across documents, and do not
answer from only the first few documents. Do not use outside knowledge except basic reasoning
needed to synthesize the evidence. If the documents do not answer part of the question, say so.
Never invent document identifiers, references, or source-specific evidence."""


USER_PROMPT = """Answer the question using only the reference documents below.

Workflow:
1. Break the question into its main facets, decisions, comparisons, constraints, entities, and
   evidence gaps.
2. Read every numbered reference document before writing.
3. Cover every major answer-relevant facet supported by the documents, including concrete facts,
   examples, tradeoffs, risks, constraints, and uncertainty.
4. Compare evidence across documents before making important claims.
5. Stop when the question is answered well; do not pad the response.

Submission requirements:
- Keep the complete answer at or under 1,024 whitespace-separated words.
- Break the answer into sentence-level objects, each grounded in one to three documents.
- Put each cited raw ClimbMix docid in references exactly once.
- citations contains zero-based integer indexes into references, not document numbers or docids.
- Every answer object has one to three unique citation indexes.
- Every reference is cited by at least one answer object.
- Cite only documents supplied below and only when they directly support the sentence.
- Return one JSON object with exactly references and answer. Do not add commentary or Markdown.

Reference documents:
{documents}

Output shape:
{{
  "references": ["<raw ClimbMix docid>"],
  "answer": [
    {{"text": "<one grounded answer sentence>", "citations": [0]}}
  ]
}}

Question: {question}
"""


def output_schema() -> dict[str, Any]:
    """Return the provider-safe strict schema; cross-field rules are checked locally."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["references", "answer"],
        "properties": {
            "references": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Unique cited raw ClimbMix document identifiers.",
            },
            "answer": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "citations"],
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "One evidence-grounded answer sentence.",
                        },
                        "citations": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "One to three zero-based indexes into references.",
                        },
                    },
                },
            },
        },
    }


@dataclass(frozen=True)
class RagGenerationConfig:
    queries_path: Path
    run_path: Path
    documents_path: Path
    output_path: Path
    team_id: str
    run_id: str
    run_desc: str
    work_dir: Path | None = None
    archive_member: str | None = None
    top_k: int | None = None
    max_document_words: int = 1000
    concurrency: int = 4
    resume: bool = False
    overwrite: bool = False
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    model: str = "openai/gpt-5.6-sol"
    reasoning_effort: str = "medium"
    temperature: float = 0.0
    max_tokens: int = 6000
    timeout_seconds: float = 900.0
    transport_max_attempts: int = 3

    def __post_init__(self) -> None:
        positive = {
            "max_document_words": self.max_document_words,
            "concurrency": self.concurrency,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "transport_max_attempts": self.transport_max_attempts,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {', '.join(invalid)}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when supplied")
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        for name in ("team_id", "run_id", "run_desc"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be nonempty")

    @property
    def resolved_work_dir(self) -> Path:
        return self.work_dir or self.output_path.with_name(
            self.output_path.stem + ".work"
        )


def _config_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _config_section(
    config: dict[str, Any], name: str, allowed_fields: set[str]
) -> dict[str, Any]:
    section = _config_mapping(config.get(name), name)
    unknown = sorted(set(section) - allowed_fields)
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(unknown)}")
    return section


def _config_text(
    mapping: dict[str, Any], key: str, owner: str, default: str | None = None
) -> str:
    value = mapping.get(key, default)
    if value is None or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}.{key} must be nonempty text")
    return value.strip()


def _optional_config_text(
    mapping: dict[str, Any], key: str, owner: str
) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}.{key} must be nonempty text or null")
    return value.strip()


def _positive_config_int(
    mapping: dict[str, Any], key: str, owner: str, default: int
) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{owner}.{key} must be a positive integer")
    return value


def _positive_config_float(
    mapping: dict[str, Any], key: str, owner: str, default: float
) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{owner}.{key} must be a positive number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{owner}.{key} must be a positive number")
    return parsed


def _resolve_config_input(root_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    active_path = root_dir / path
    if active_path.exists():
        return active_path
    shared_root = shared_checkout_root(root_dir)
    if shared_root:
        shared_path = shared_root / path
        if shared_path.exists():
            return shared_path
    return active_path


def _resolve_config_output(root_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root_dir / path


def load_rag_generation_config(path: Path) -> RagGenerationConfig:
    """Load one strict competition answer-generation YAML file."""
    config_path = path.resolve()
    root_dir = find_repo_root(config_path.parent)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = _config_mapping(raw, "config")
    expected_sections = {
        "experiment",
        "submission",
        "inputs",
        "retrieval",
        "generation",
    }
    unknown_sections = sorted(set(config) - expected_sections)
    missing_sections = sorted(expected_sections - set(config))
    if unknown_sections:
        raise ValueError(f"unknown config section(s): {', '.join(unknown_sections)}")
    if missing_sections:
        raise ValueError(f"missing config section(s): {', '.join(missing_sections)}")

    experiment = _config_section(
        config, "experiment", {"id", "output_dir", "mode"}
    )
    experiment_id = _config_text(experiment, "id", "experiment")
    output_dir = _resolve_config_output(
        root_dir,
        experiment.get("output_dir") or Path("outputs") / experiment_id,
    )
    mode = _config_text(experiment, "mode", "experiment", "create").lower()
    if mode not in {"create", "resume", "overwrite"}:
        raise ValueError("experiment.mode must be create, resume, or overwrite")

    submission = _config_section(config, "submission", {"team_id", "run_desc"})
    inputs = _config_section(
        config, "inputs", {"queries", "run", "documents", "archive_member"}
    )
    retrieval = _config_section(
        config, "retrieval", {"top_k", "max_document_words"}
    )
    generation = _config_section(
        config,
        "generation",
        {
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
    generation_type = _config_text(
        generation, "type", "generation", "openrouter"
    ).lower()
    if generation_type != "openrouter":
        raise ValueError("generation.type must be openrouter")

    top_k = retrieval.get("top_k")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
    ):
        raise ValueError("retrieval.top_k must be a positive integer or null")
    temperature = generation.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise ValueError("generation.temperature must be a finite number")
    temperature = float(temperature)
    if not math.isfinite(temperature):
        raise ValueError("generation.temperature must be a finite number")
    reasoning_effort = _config_text(
        generation, "reasoning_effort", "generation", "medium"
    ).lower()
    if reasoning_effort not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ValueError("generation.reasoning_effort is unsupported")

    return RagGenerationConfig(
        queries_path=_resolve_config_input(
            root_dir, _config_text(inputs, "queries", "inputs")
        ),
        run_path=_resolve_config_input(
            root_dir, _config_text(inputs, "run", "inputs")
        ),
        documents_path=_resolve_config_input(
            root_dir, _config_text(inputs, "documents", "inputs")
        ),
        output_path=output_dir / "rag_output_trec_rag_2026.jsonl",
        work_dir=output_dir / "work",
        archive_member=_optional_config_text(inputs, "archive_member", "inputs"),
        top_k=top_k,
        max_document_words=_positive_config_int(
            retrieval, "max_document_words", "retrieval", 1000
        ),
        team_id=_config_text(submission, "team_id", "submission"),
        run_id=experiment_id,
        run_desc=_config_text(submission, "run_desc", "submission"),
        api_base=_config_text(
            generation,
            "api_base",
            "generation",
            "https://openrouter.ai/api/v1",
        ),
        api_key_env=_config_text(
            generation, "api_key_env", "generation", "OPENROUTER_API_KEY"
        ),
        model=_config_text(
            generation, "model", "generation", "openai/gpt-5.6-sol"
        ),
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_tokens=_positive_config_int(
            generation, "max_tokens", "generation", 6000
        ),
        timeout_seconds=_positive_config_float(
            generation, "timeout_seconds", "generation", 900.0
        ),
        transport_max_attempts=_positive_config_int(
            generation, "transport_max_attempts", "generation", 3
        ),
        concurrency=_positive_config_int(
            generation, "concurrency", "generation", 4
        ),
        resume=mode == "resume",
        overwrite=mode == "overwrite",
    )


class JsonGenerator(Protocol):
    def complete_json(
        self,
        *,
        topic_id: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the parsed model object and the raw provider response."""


class SemanticCompletionError(ValueError):
    """A successful HTTP response whose completion cannot be accepted."""

    def __init__(self, message: str, raw_response: object) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class OpenRouterJsonGenerator:
    """One-completion OpenRouter client with identical transient retries only."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        reasoning_effort: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        transport_max_attempts: int,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is missing or empty")
        self.api_base = api_base.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.transport_max_attempts = transport_max_attempts

    def complete_json(
        self,
        *,
        topic_id: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del topic_id
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning": {"effort": self.reasoning_effort, "exclude": True},
            "provider": {"require_parameters": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "trec_rag_2026_answer",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        }
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

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.transport_max_attempts:
                    response.raise_for_status()
                delay = _retry_delay(response.headers.get("Retry-After"), attempt)
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                try:
                    detail = json.dumps(response.json(), ensure_ascii=False)
                except ValueError:
                    detail = response.text
                raise RuntimeError(
                    f"OpenRouter rejected generation with HTTP {response.status_code}: "
                    f"{detail[:2000]}"
                )

            try:
                envelope = response.json()
            except ValueError as exc:
                raise SemanticCompletionError(
                    "OpenRouter returned a non-JSON response; no repair call was made",
                    {"http_status": response.status_code, "body": response.text},
                ) from exc
            if not isinstance(envelope, dict):
                raise SemanticCompletionError(
                    "OpenRouter response must be a JSON object; no repair call was made",
                    envelope,
                )
            try:
                message = envelope["choices"][0]["message"]
                content = _message_text(message.get("content"))
                generated = parse_generated_json(content)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise SemanticCompletionError(
                    "OpenRouter returned a malformed semantic completion; no repair call was made",
                    envelope,
                ) from exc
            return generated, envelope

        raise AssertionError("unreachable OpenRouter retry state")


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
            str(item["text"])
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "output_text"}
            and isinstance(item.get("text"), str)
        ]
        if parts:
            return "\n".join(parts).strip()
    raise ValueError("completion content is not nonempty text")


def load_queries(path: Path) -> list[tuple[str, str]]:
    topics: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not fields or not any(field.strip() for field in fields):
                continue
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: expected two TSV columns")
            topic_id = fields[0].strip()
            narrative = "\t".join(fields[1:]).strip()
            if not topics and topic_id.lower() in {"qid", "query_id", "topic_id"}:
                continue
            if not topic_id or not narrative:
                raise ValueError(f"{path}:{line_number}: empty topic id or narrative")
            topics.append((topic_id, narrative))
    topic_ids = [topic_id for topic_id, _ in topics]
    if not topics:
        raise ValueError(f"{path}: no topics found")
    if len(topic_ids) != len(set(topic_ids)):
        raise ValueError(f"{path}: duplicate topic ids")
    return topics


def load_trec_run(
    path: Path, topic_ids: set[str], top_k: int | None
) -> dict[str, list[str]]:
    found: dict[str, list[tuple[int, str]]] = {
        topic_id: [] for topic_id in sorted(topic_ids)
    }
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"{path}:{line_number}: expected six TREC columns")
        topic_id, docid = fields[0], fields[2]
        if topic_id not in found:
            continue
        try:
            rank = int(fields[3])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid rank") from exc
        if rank > 0:
            found[topic_id].append((rank, docid))

    ranked: dict[str, list[str]] = {}
    for topic_id, hits in found.items():
        hits.sort()
        ranks = [rank for rank, _ in hits]
        docids = [docid for _, docid in hits]
        if not hits:
            raise ValueError(f"{path}: {topic_id} has no positive-ranked documents")
        if len(ranks) != len(set(ranks)):
            raise ValueError(f"{path}: {topic_id} contains duplicate ranks")
        if len(docids) != len(set(docids)):
            raise ValueError(f"{path}: {topic_id} contains duplicate docids")
        ranked[topic_id] = docids[:top_k] if top_k is not None else docids
    return ranked


@contextmanager
def _document_lines(path: Path, archive_member: str | None) -> Iterator[TextIO]:
    if not zipfile.is_zipfile(path):
        if archive_member:
            raise ValueError("archive_member requires ZIP document input")
        with path.open(encoding="utf-8") as handle:
            yield handle
        return

    with zipfile.ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".jsonl", ".json"))
        ]
        member = archive_member or (candidates[0] if len(candidates) == 1 else "")
        if not member or member not in archive.namelist():
            raise ValueError(
                "choose one JSONL ZIP member with archive_member; "
                f"found {candidates}"
            )
        with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8") as handle:
            yield handle


def _string_value(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_documents(
    path: Path,
    archive_member: str | None,
    wanted_docids: set[str],
    max_words: int,
) -> dict[str, str]:
    documents: dict[str, str] = {}
    with _document_lines(path, archive_member) as lines:
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            nested = record.get("candidates")
            rows = nested if isinstance(nested, list) else [record]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                docid = _string_value(row, "docid", "id", "_id")
                text = _string_value(row, "text", "doc", "contents", "body")
                if docid not in wanted_docids or not text:
                    continue
                text = " ".join(text.split()[:max_words])
                if docid in documents and documents[docid] != text:
                    raise ValueError(
                        f"{path}:{line_number}: conflicting duplicate document {docid}"
                    )
                documents[docid] = text
    missing = sorted(wanted_docids - documents.keys())
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{path}: missing {len(missing)} ranked documents"
            + (f" ({preview})" if preview else "")
        )
    return documents


def render_prompt(
    narrative: str, ranked_docids: list[str], documents: dict[str, str]
) -> str:
    context = "\n\n".join(
        f"[{index}] docid: {docid}\n{documents[docid]}"
        for index, docid in enumerate(ranked_docids, 1)
    )
    return USER_PROMPT.format(documents=context, question=narrative)


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
    return {
        "metadata": _metadata(
            topic_id=topic_id,
            narrative=narrative,
            team_id=team_id,
            run_id=run_id,
            run_desc=run_desc,
        ),
        "references": generated.get("references"),
        "answer": generated.get("answer"),
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
    if set(record) != {"metadata", "references", "answer"}:
        raise ValueError(f"{topic_id}: invalid root object or metadata")
    if record.get("metadata") != expected_metadata:
        raise ValueError(f"{topic_id}: invalid root object or metadata")

    references = record.get("references")
    if (
        not isinstance(references, list)
        or not references
        or not all(
            isinstance(docid, str) and docid.strip() and docid == docid.strip()
            for docid in references
        )
    ):
        raise ValueError(f"{topic_id}: references must be nonempty docid strings")
    if len(references) != len(set(references)) or not set(references) <= set(
        allowed_docids
    ):
        raise ValueError(
            f"{topic_id}: references are duplicated or outside selected TREC rows"
        )

    answer = record.get("answer")
    if not isinstance(answer, list) or not answer:
        raise ValueError(f"{topic_id}: answer must be a nonempty list")
    used: set[int] = set()
    word_count = 0
    for index, item in enumerate(answer):
        if not isinstance(item, dict) or set(item) != {"text", "citations"}:
            raise ValueError(f"{topic_id}: answer[{index}] has invalid fields")
        text = item.get("text")
        citations = item.get("citations")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{topic_id}: answer[{index}] has empty text")
        word_count += len(text.split())
        if not isinstance(citations, list) or not 1 <= len(citations) <= 3:
            raise ValueError(
                f"{topic_id}: answer[{index}] must have 1-3 unique citations"
            )
        if any(type(citation) is not int for citation in citations):
            raise ValueError(f"{topic_id}: answer[{index}] has an invalid citation")
        if len(citations) != len(set(citations)):
            raise ValueError(
                f"{topic_id}: answer[{index}] must have 1-3 unique citations"
            )
        if any(not 0 <= citation < len(references) for citation in citations):
            raise ValueError(f"{topic_id}: answer[{index}] has an invalid citation")
        used.update(citations)

    if word_count > 1024:
        raise ValueError(f"{topic_id}: answer exceeds 1,024 words")
    if used != set(range(len(references))):
        raise ValueError(f"{topic_id}: answer has uncited references")


def _safe_topic_name(topic_id: str) -> str:
    prefix = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in topic_id
    )
    return f"{prefix}-{sha256(topic_id.encode('utf-8')).hexdigest()[:8]}"


def _write_json(path: Path, value: object, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if compact:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _saved_record(
    path: Path,
    *,
    topic_id: str,
    narrative: str,
    allowed_docids: list[str],
    config: RagGenerationConfig,
) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return None
        validate_submission_record(
            record,
            topic_id=topic_id,
            narrative=narrative,
            allowed_docids=allowed_docids,
            team_id=config.team_id,
            run_id=config.run_id,
            run_desc=config.run_desc,
        )
        return record
    except (OSError, ValueError, json.JSONDecodeError):
        return None


async def _generate_topic(
    *,
    topic_id: str,
    narrative: str,
    ranked_docids: list[str],
    documents: dict[str, str],
    generator: JsonGenerator,
    config: RagGenerationConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None]:
    work_dir = config.resolved_work_dir
    topic_name = _safe_topic_name(topic_id)
    try:
        async with semaphore:
            generated, raw_response = await asyncio.to_thread(
                generator.complete_json,
                topic_id=topic_id,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=render_prompt(narrative, ranked_docids, documents),
                response_schema=output_schema(),
            )
        _write_json(work_dir / "raw" / f"{topic_name}.json", raw_response)
        record = build_submission_record(
            generated,
            topic_id=topic_id,
            narrative=narrative,
            team_id=config.team_id,
            run_id=config.run_id,
            run_desc=config.run_desc,
        )
        validate_submission_record(
            record,
            topic_id=topic_id,
            narrative=narrative,
            allowed_docids=ranked_docids,
            team_id=config.team_id,
            run_id=config.run_id,
            run_desc=config.run_desc,
        )
        _write_json(work_dir / "rows" / f"{topic_name}.json", record, compact=True)
        print(f"completed {topic_id}", flush=True)
        return topic_id, None
    except Exception as exc:
        if isinstance(exc, SemanticCompletionError):
            _write_json(
                work_dir / "raw" / f"{topic_name}.failed.json",
                exc.raw_response,
            )
        error_path = work_dir / "errors" / f"{topic_name}.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(f"failed {topic_id}: {exc}", flush=True)
        return topic_id, f"{type(exc).__name__}: {exc}"


async def run_generation(
    config: RagGenerationConfig, generator: JsonGenerator
) -> None:
    topics = load_queries(config.queries_path)
    ranked = load_trec_run(
        config.run_path,
        {topic_id for topic_id, _ in topics},
        config.top_k,
    )
    wanted_docids = {
        docid for topic_docids in ranked.values() for docid in topic_docids
    }
    print(f"loading {len(wanted_docids)} ranked documents", flush=True)
    documents = load_documents(
        config.documents_path,
        config.archive_member,
        wanted_docids,
        config.max_document_words,
    )

    work_dir = config.resolved_work_dir
    rows_dir = work_dir / "rows"
    existing_rows = list(rows_dir.glob("*.json")) if rows_dir.exists() else []
    if (
        not config.resume
        and not config.overwrite
        and (config.output_path.exists() or existing_rows)
    ):
        raise ValueError("generation artifacts exist; use --resume or --overwrite")
    if config.overwrite and config.output_path.exists():
        config.output_path.unlink()

    pending: list[tuple[str, str]] = []
    for topic_id, narrative in topics:
        row_path = rows_dir / f"{_safe_topic_name(topic_id)}.json"
        saved = (
            _saved_record(
                row_path,
                topic_id=topic_id,
                narrative=narrative,
                allowed_docids=ranked[topic_id],
                config=config,
            )
            if config.resume
            else None
        )
        if saved is None:
            pending.append((topic_id, narrative))

    print(
        f"topics={len(topics)} reused={len(topics) - len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )
    semaphore = asyncio.Semaphore(config.concurrency)
    results = await asyncio.gather(
        *[
            _generate_topic(
                topic_id=topic_id,
                narrative=narrative,
                ranked_docids=ranked[topic_id],
                documents=documents,
                generator=generator,
                config=config,
                semaphore=semaphore,
            )
            for topic_id, narrative in pending
        ]
    )
    failures = [(topic_id, error) for topic_id, error in results if error]
    if failures:
        raise RuntimeError(
            f"{len(failures)} topic(s) failed; inspect {work_dir / 'errors'} "
            "and rerun with --resume"
        )

    final_records: list[dict[str, Any]] = []
    for topic_id, narrative in topics:
        row_path = rows_dir / f"{_safe_topic_name(topic_id)}.json"
        record = _saved_record(
            row_path,
            topic_id=topic_id,
            narrative=narrative,
            allowed_docids=ranked[topic_id],
            config=config,
        )
        if record is None:
            raise RuntimeError(f"missing valid generated row for {topic_id}")
        final_records.append(record)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_path.with_name(config.output_path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in final_records
        ),
        encoding="utf-8",
    )
    temporary.replace(config.output_path)
    print(
        f"wrote {len(final_records)} topics to {config.output_path}",
        flush=True,
    )


def arguments(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Competition RAG YAML.")
    return parser.parse_args(argv).config


def main() -> None:
    try:
        config_path = arguments()
        load_repo_env(find_repo_root(config_path.resolve().parent))
        config = load_rag_generation_config(config_path)
        api_key = os.environ.get(config.api_key_env, "")
        generator = OpenRouterJsonGenerator(
            api_base=config.api_base,
            api_key=api_key,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            transport_max_attempts=config.transport_max_attempts,
        )
        asyncio.run(run_generation(config, generator))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        raise SystemExit(f"error: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
