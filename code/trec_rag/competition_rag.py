"""Generate organizer-format TREC RAG answers from a fixed retrieval run."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import re
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence, TextIO
from urllib.parse import unquote

import requests
import yaml
from filelock import FileLock, Timeout as FileLockTimeout

from trec_rag.repo_env import find_repo_root, load_repo_env, shared_checkout_root
from trec_rag.topics import load_narrative_topics


_SCHEMA_VERSION = "competition_rag_config_v1"
_SAFE_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
# strict_schema pins provider selection to schema-capable endpoints; json_object is the
# portable fallback; none sends no response_format at all.
STRUCTURED_OUTPUT_MODES = frozenset({"strict_schema", "json_object", "none"})
_MAX_REDACTION_NORMALIZATION_ROUNDS = 32

SYSTEM_PROMPT = """You are a reference-document RAG answer-generation agent. Use only the
provided reference documents and the user's task instructions. Do not invent evidence,
document identifiers, references, or source-specific claims."""

USER_PROMPT = """Answer the question using only the reference documents below.

Read every reference document before writing. Cover answer-relevant evidence,
tradeoffs, constraints, and uncertainty without padding. The complete answer must be at most
1,024 whitespace-separated words. Each answer object must have one to three unique zero-based
citation indexes into references. When an answer object cites more than one reference, order its
citation indexes from strongest to weakest support for that object. Include each cited raw
ClimbMix docid once in references, and cite every reference. Return one JSON object with exactly
references and answer; no Markdown.

Reference documents:
{documents}

Question: {question}
"""

ATOMIC_USER_PROMPT = """Answer the question using only the reference documents below.

Read every reference document before writing. Cover answer-relevant evidence,
tradeoffs, constraints, and uncertainty without padding. The complete answer must be at most
1,024 whitespace-separated words.

Write each answer object as one self-contained sentence stating a single claim. Do not join
several claims into one answer object and do not prefix an object with a section heading or
label, because a cited document must be able to support the whole object on its own.

Each answer object must have one to three unique zero-based citation indexes into references.
When an answer object cites more than one reference, order its citation indexes from strongest
to weakest support for that object. Include each cited raw ClimbMix docid once in references,
and cite every reference. Return one JSON object with exactly references and answer; no
Markdown.

Reference documents:
{documents}

Question: {question}
"""

FULL_BUDGET_USER_PROMPT = """Answer the question using only the reference documents below.

Read every reference document before writing. Aim for roughly 900 whitespace-separated words
and never exceed 1,024, counting every answer object together. A longer answer is only better
when the extra words carry new evidence, so keep adding distinct, answer-relevant points that
the reference documents support, covering the question's different aspects, tradeoffs,
constraints and uncertainty. Do not pad, repeat a point already made, or restate the
question.

Write each answer object as one self-contained sentence stating a single claim. Do not join
several claims into one answer object and do not prefix an object with a section heading or
label, because a cited document must be able to support the whole object on its own.

Each answer object must have one to three unique zero-based citation indexes into references.
When an answer object cites more than one reference, order its citation indexes from strongest
to weakest support for that object. Include each cited raw ClimbMix docid once in references,
and cite every reference. Return one JSON object with exactly references and answer; no
Markdown.

Reference documents:
{documents}

Question: {question}
"""

FOCUSED_CITATION_USER_PROMPT = """Answer the question using only the reference documents below.

Read every reference document before writing. Aim for roughly 900 whitespace-separated words
and never exceed 1,024, counting every answer object together. A longer answer is only better
when the extra words carry new evidence, so keep adding distinct, answer-relevant points that
the reference documents support, covering the question's different aspects, tradeoffs,
constraints and uncertainty. Do not pad, repeat a point already made, or restate the
question.

Write each answer object as one self-contained sentence stating a single claim. Do not join
several claims into one answer object and do not prefix an object with a section heading or
label, because a cited document must be able to support the whole object on its own.

Cite the single document that best supports each answer object. Add a second or third citation
only when that document by itself also fully supports the same object, and order citations from
strongest to weakest support. Never add a citation to spread coverage across documents: an
extra citation that does not support the object on its own is worse than no extra citation.
List a document in references only if some answer object cites it.

Give each answer object one to three unique zero-based citation indexes into references. Return
one JSON object with exactly references and answer; no Markdown.

Reference documents:
{documents}

Question: {question}
"""

JSON_EXAMPLE_USER_PROMPT = FOCUSED_CITATION_USER_PROMPT.replace(
    """Give each answer object one to three unique zero-based citation indexes into references. Return
one JSON object with exactly references and answer; no Markdown.""",
    """Give each answer object one to three unique zero-based citation indexes into references.
Return one json object with exactly references and answer; no Markdown. Use exactly this json
shape, in which every answer object states a single claim and cites the one document supporting
it:

{{"references": ["shard_00459_61697", "shard_01012_88420"],
 "answer": [{{"text": "Commercial reactors generate heat by fissioning uranium.", "citations": [0]}},
            {{"text": "Spent fuel needs long-term isolation from the environment.", "citations": [1]}}]}}""",
)
assert JSON_EXAMPLE_USER_PROMPT != FOCUSED_CITATION_USER_PROMPT, "json example substitution failed"


TAIL_CONTRACT_USER_PROMPT = FOCUSED_CITATION_USER_PROMPT + """
Restating the output contract, which the reference documents above have separated from this
point by a long span of text:

- one json object with exactly references and answer, no Markdown
- each answer object is one self-contained sentence stating a single claim
- cite the document that best supports each object, one to three unique zero-based indexes,
  strongest first, and never add a citation that does not support the object on its own
- list a document in references only if some answer object cites it
- roughly 900 whitespace-separated words in total, never more than 1,024
"""

CONTRACT_SYSTEM_PROMPT = """You are a reference-document RAG answer-generation agent. Use only the
provided reference documents and the user's task instructions. Do not invent evidence,
document identifiers, references, or source-specific claims.

Always produce one json object with exactly two keys, references and answer, and no Markdown.

- Each answer object is one self-contained sentence stating a single claim. Never join several
  claims into one object and never prefix an object with a section heading or label, because a
  cited document must support the whole object on its own.
- Cite the document that best supports each object. Give one to three unique zero-based indexes
  into references, strongest support first. Never add a citation that does not support the
  object on its own.
- List a document in references only if some answer object cites it.
- Write roughly 900 whitespace-separated words in total and never exceed 1,024."""

CONTRACT_IN_SYSTEM_USER_PROMPT = """Answer the question using only the reference documents below,
following the output contract in your instructions.

Read every reference document before writing. Keep adding distinct, answer-relevant points that
the documents support, covering the question's different aspects, tradeoffs, constraints and
uncertainty. Do not pad, repeat a point already made, or restate the question.

Reference documents:
{documents}

Question: {question}
"""

PROMPT_PROFILES = {
    "default": USER_PROMPT,
    "atomic_claims": ATOMIC_USER_PROMPT,
    "atomic_claims_full_budget": FULL_BUDGET_USER_PROMPT,
    "focused_citations": FOCUSED_CITATION_USER_PROMPT,
    # DeepSeek documents that JSON output needs the literal word "json" plus a worked
    # example of the desired shape, not just a response_format setting.
    "focused_citations_json_example": JSON_EXAMPLE_USER_PROMPT,
    # The contract sits ~120k tokens before the generation point once documents are
    # inserted; this profile restates it compactly after the question.
    "focused_citations_tail": TAIL_CONTRACT_USER_PROMPT,
    # Moves the contract into the system message, the channel models weight persistently,
    # instead of burying it 120k tokens above the generation point.
    "contract_in_system": CONTRACT_IN_SYSTEM_USER_PROMPT,
}

# Profiles that carry their contract in the system message need a matching system prompt.
SYSTEM_PROMPT_PROFILES = {"contract_in_system": CONTRACT_SYSTEM_PROMPT}


def system_prompt_for(prompt_profile: str) -> str:
    """Return the system message a prompt profile expects."""
    if prompt_profile not in PROMPT_PROFILES:
        raise ValueError(f"unsupported prompt profile: {prompt_profile}")
    return SYSTEM_PROMPT_PROFILES.get(prompt_profile, SYSTEM_PROMPT)


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
    queries_path: Path
    run_path: Path
    documents_path: Path
    output_path: Path
    work_dir: Path
    team_id: str
    run_id: str
    run_desc: str
    archive_member: str | None
    topic_ids: tuple[str, ...] | None
    top_k: int | None
    max_document_words: int
    concurrency: int
    resume: bool
    overwrite: bool
    provider: str
    api_base: str
    api_key_env: str
    model: str
    reasoning_effort: str
    prompt_profile: str
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
        {"schema_version", "experiment", "submission", "inputs", "retrieval", "generation"},
        "config",
    )
    _require_fields(
        config,
        {"schema_version", "experiment", "submission", "inputs", "retrieval", "generation"},
        "config",
    )
    if _text(config, "schema_version", "config") != _SCHEMA_VERSION:
        raise ValueError(f"config.schema_version must be {_SCHEMA_VERSION}")

    experiment = _section(config, "experiment", {"id", "output_dir", "mode"})
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
        {"queries", "run", "documents", "archive_member", "topic_ids"},
        required={"queries", "run", "documents"},
    )
    retrieval = _section(
        config,
        "retrieval",
        {"top_k", "max_document_words"},
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
            "prompt_profile",
            "structured_output",
            "temperature",
            "max_tokens",
            "timeout_seconds",
            "transport_max_attempts",
            "concurrency",
        },
        # prompt_profile is optional so existing configs keep the default prompt.
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
    prompt_profile = _optional_text(generation, "prompt_profile", "generation") or "default"
    if prompt_profile not in PROMPT_PROFILES:
        raise ValueError("generation.prompt_profile is unsupported")
    structured_output = (
        _optional_text(generation, "structured_output", "generation") or "strict_schema"
    )
    if structured_output not in STRUCTURED_OUTPUT_MODES:
        raise ValueError("generation.structured_output is unsupported")

    return RagGenerationConfig(
        schema_version=_SCHEMA_VERSION,
        queries_path=_input_path(root_dir, _text(inputs, "queries", "inputs")),
        run_path=_input_path(root_dir, _text(inputs, "run", "inputs")),
        documents_path=_input_path(root_dir, _text(inputs, "documents", "inputs")),
        output_path=output_dir / "rag_output_trec_rag_2026.jsonl",
        work_dir=output_dir / "work",
        team_id=_text(submission, "team_id", "submission"),
        run_id=experiment_id,
        run_desc=_text(submission, "run_desc", "submission"),
        archive_member=_optional_text(inputs, "archive_member", "inputs"),
        topic_ids=_topic_ids(inputs),
        top_k=_optional_positive_int(inputs=retrieval, key="top_k", owner="retrieval"),
        max_document_words=_positive_int(retrieval, "max_document_words", "retrieval"),
        concurrency=_positive_int(generation, "concurrency", "generation"),
        resume=mode == "resume",
        overwrite=mode == "overwrite",
        provider=_text(generation, "type", "generation").lower(),
        api_base=_text(generation, "api_base", "generation"),
        api_key_env=_text(generation, "api_key_env", "generation"),
        model=_text(generation, "model", "generation"),
        reasoning_effort=reasoning_effort,
        prompt_profile=prompt_profile,
        structured_output=structured_output,
        temperature=_optional_finite_float(generation, "temperature", "generation"),
        max_tokens=_positive_int(generation, "max_tokens", "generation"),
        timeout_seconds=_positive_float(generation, "timeout_seconds", "generation"),
        transport_max_attempts=_positive_int(
            generation, "transport_max_attempts", "generation"
        ),
    )


def load_queries(path: Path) -> list[tuple[str, str]]:
    """Read the canonical headerless ``narrative_id<TAB>narrative`` topic TSV."""
    topics = load_narrative_topics(path)
    if topics[0].id.strip().casefold() in {
        "qid",
        "query_id",
        "topic_id",
        "narrative_id",
    }:
        raise ValueError(f"{path}:1: topic TSV must not have a header")
    queries = [(topic.id, topic.narrative) for topic in topics]
    return queries


def select_queries(
    queries: Sequence[tuple[str, str]], topic_ids: Sequence[str] | None
) -> list[tuple[str, str]]:
    """Select an explicit unique topic subset while retaining official TSV order."""
    if topic_ids is None:
        return list(queries)
    if not topic_ids:
        raise ValueError("inputs.topic_ids must contain at least one topic ID")
    available = {topic_id for topic_id, _ in queries}
    requested: set[str] = set()
    for value in topic_ids:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("inputs.topic_ids must contain non-empty text")
        topic_id = value.strip()
        if topic_id in requested:
            raise ValueError(f"duplicate topic ID: {topic_id}")
        if topic_id not in available:
            raise ValueError(f"unknown topic ID: {topic_id}")
        requested.add(topic_id)
    return [query for query in queries if query[0] in requested]


def load_trec_run(
    path: Path, topic_ids: set[str], top_k: int | None
) -> dict[str, list[str]]:
    """Decode one strictly valid six-column TREC run for the selected topics."""
    if not topic_ids:
        raise ValueError("at least one topic ID is required")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
        raise ValueError("top_k must be a positive integer or null")
    grouped: dict[str, list[tuple[int, float, str]]] = {}
    run_tag: str | None = None
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            raise ValueError(f"{path}:{line_number}: blank TREC rows are invalid")
        fields = raw_line.split()
        if len(fields) != 6:
            raise ValueError(f"{path}:{line_number}: expected six TREC fields")
        topic_id, q0, docid, raw_rank, raw_score, tag = fields
        if q0 != "Q0":
            raise ValueError(f"{path}:{line_number}: TREC second field must be literal Q0")
        if not topic_id or not docid:
            raise ValueError(f"{path}:{line_number}: topic id and docid must be non-empty")
        try:
            rank = int(raw_rank)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid rank") from exc
        if rank <= 0:
            raise ValueError(f"{path}:{line_number}: rank must be positive")
        try:
            score = float(raw_score)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric score") from exc
        if not math.isfinite(score):
            raise ValueError(f"{path}:{line_number}: score must be finite")
        if not tag:
            raise ValueError(f"{path}:{line_number}: run tag must be non-empty")
        if run_tag is None:
            run_tag = tag
        elif tag != run_tag:
            raise ValueError(f"{path}:{line_number}: run tag must be stable")
        grouped.setdefault(topic_id, []).append((rank, score, docid))
    if run_tag is None:
        raise ValueError(f"{path}: no TREC rows found")

    ranked: dict[str, list[str]] = {}
    for topic_id, rows in grouped.items():
        ranks = [rank for rank, _, _ in rows]
        docids = [docid for _, _, docid in rows]
        if len(ranks) != len(set(ranks)):
            raise ValueError(f"{path}: {topic_id} contains duplicate rank")
        if ranks != sorted(ranks):
            raise ValueError(f"{path}: {topic_id} rows are out of rank order")
        if ranks[0] != 1:
            raise ValueError(f"{path}: {topic_id} ranks must start at 1")
        if len(docids) != len(set(docids)):
            raise ValueError(f"{path}: {topic_id} contains duplicate docid")
        scores = [score for _, score, _ in rows]
        if any(previous < current for previous, current in zip(scores, scores[1:])):
            raise ValueError(f"{path}: {topic_id} scores must be non-increasing by rank")
        if topic_id in topic_ids:
            selected = docids if top_k is None else docids[:top_k]
            ranked[topic_id] = selected
    missing = sorted(topic_ids - ranked.keys())
    if missing:
        raise ValueError(f"{path}: missing ranked documents for {', '.join(missing)}")
    return ranked


@contextmanager
def _document_lines(path: Path, archive_member: str | None) -> Iterator[TextIO]:
    document_path = Path(path)
    expects_zip = document_path.suffix.lower() == ".zip"
    if not expects_zip and not zipfile.is_zipfile(document_path):
        if archive_member is not None:
            raise ValueError("archive_member requires a ZIP document input")
        with document_path.open(encoding="utf-8") as handle:
            try:
                yield handle
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{document_path}: document input is not valid UTF-8"
                ) from exc
        return
    try:
        with zipfile.ZipFile(document_path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".jsonl", ".json"))
            ]
            member = archive_member or (candidates[0] if len(candidates) == 1 else None)
            if member is None or member not in candidates:
                raise ValueError(
                    f"{document_path}: choose one JSONL ZIP member with archive_member; "
                    f"found {candidates}"
                )
            with archive.open(member) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8") as handle:
                    try:
                        yield handle
                    except UnicodeDecodeError as exc:
                        raise ValueError(
                            f"{document_path}: selected document member is not valid UTF-8"
                        ) from exc
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{document_path}: invalid ZIP document archive") from exc


def load_documents(
    path: Path,
    archive_member: str | None,
    wanted_docids: set[str],
    max_words: int,
) -> dict[str, str]:
    """Read organizer query-bundled documents while allowing extension fields."""
    if not wanted_docids:
        raise ValueError("at least one wanted docid is required")
    if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words <= 0:
        raise ValueError("max_words must be a positive integer")
    documents: dict[str, str] = {}
    with _document_lines(path, archive_member) as lines:
        for line_number, raw_line in enumerate(lines, 1):
            if not raw_line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL rows are invalid")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            query = record.get("query")
            candidates = record.get("candidates")
            if not isinstance(query, dict) or not _nonempty_text(query.get("qid")):
                raise ValueError(f"{path}:{line_number}: organizer query core is invalid")
            if not isinstance(candidates, list):
                raise ValueError(f"{path}:{line_number}: organizer candidates core is invalid")
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError(f"{path}:{line_number}: candidate must be an object")
                docid = candidate.get("docid")
                text = candidate.get("doc")
                if not _nonempty_text(docid) or not _nonempty_text(text):
                    raise ValueError(f"{path}:{line_number}: candidate docid and doc are required")
                normalized_docid = docid.strip()
                normalized_text = " ".join(text.split())
                if normalized_docid not in wanted_docids:
                    continue
                existing = documents.get(normalized_docid)
                if existing is not None and existing != normalized_text:
                    raise ValueError(
                        f"{path}:{line_number}: conflicting duplicate document {normalized_docid}"
                    )
                documents[normalized_docid] = normalized_text
    missing = sorted(wanted_docids - documents.keys())
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} ranked documents ({', '.join(missing[:5])})")
    return {
        docid: " ".join(text.split()[:max_words]) for docid, text in documents.items()
    }


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


def _topic_ids(inputs: dict[str, Any]) -> tuple[str, ...] | None:
    value = inputs.get("topic_ids")
    if value is None and "topic_ids" not in inputs:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("inputs.topic_ids must be a non-empty list")
    normalized: list[str] = []
    seen: set[str] = set()
    for topic_id in value:
        if not _nonempty_text(topic_id):
            raise ValueError("inputs.topic_ids must contain non-empty text")
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


def _optional_positive_int(*, inputs: dict[str, Any], key: str, owner: str) -> int | None:
    value = inputs.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{owner}.{key} must be a positive integer or null")
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


def output_schema() -> dict[str, Any]:
    """Return the provider-safe schema; cross-field rules are validated locally."""
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
                            "items": {"type": "integer"},
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

    def __init__(self, message: str, raw_response: object) -> None:
        super().__init__(message)
        self.raw_response = raw_response


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
                )
            if not isinstance(envelope, dict):
                raise SemanticCompletionError(
                    "OpenRouter response must be a JSON object; no repair call was made",
                    safe_response,
                )
            # Checked outside the parse guard below, which catches ValueError and would
            # otherwise rewrite this as a generic malformed-completion error.
            # Reasoning tokens share the max_tokens budget, so truncation is a real risk, and
            # grammar-constrained decoding can close the JSON validly while the answer is cut
            # short, producing a record that passes every downstream check and scores badly.
            choices = envelope.get("choices")
            if (
                isinstance(choices, list)
                and choices
                and isinstance(choices[0], dict)
                and choices[0].get("finish_reason") == "length"
            ):
                raise SemanticCompletionError(
                    "OpenRouter truncated the completion at max_tokens; raise "
                    "generation.max_tokens rather than accepting a shortened answer",
                    safe_response,
                )
            try:
                message = envelope["choices"][0]["message"]
                generated = parse_generated_json(_message_text(message.get("content")))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise SemanticCompletionError(
                    "OpenRouter returned a malformed semantic completion; no repair call was made",
                    safe_response,
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


def render_prompt(
    narrative: str,
    ranked_docids: list[str],
    documents: dict[str, str],
    prompt_profile: str = "default",
) -> str:
    template = PROMPT_PROFILES.get(prompt_profile)
    if template is None:
        raise ValueError(f"unsupported prompt profile: {prompt_profile}")
    context = "\n\n".join(
        f"Reference document docid: {docid}\n{documents[docid]}"
        for docid in ranked_docids
    )
    return template.format(documents=context, question=narrative)


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
        raise ValueError(f"{topic_id}: references are duplicated or outside selected TREC rows")
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
    if words > 1024:
        raise ValueError(f"{topic_id}: answer exceeds 1,024 words")


def normalize_generated_record(record: dict[str, Any]) -> dict[str, Any]:
    """Rebuild ``references`` from the citations actually used, then re-index them.

    The organizer baseline validator requires every reference to be cited. Asking the model to
    satisfy that is the single largest source of first-attempt generation failures, and it also
    drives citation padding: to cover a long reference list the model must attach two or three
    citations to nearly every object, and the trailing ones frequently do not support the claim.

    Deriving the reference list from the citations instead makes the constraint true by
    construction. Answer text is never touched, and each citation keeps pointing at exactly the
    document it pointed at before, so no claim changes its evidence.
    """
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


def trim_to_word_limit(record: dict[str, Any], *, max_words: int = 1024) -> dict[str, Any]:
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
    topic_id: str,
    narrative: str,
    allowed_docids: list[str],
    config: RagGenerationConfig,
) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return None
        _validate_generated_submission_record(
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
    secrets = (os.environ.get(config.api_key_env, ""),)
    try:
        async with semaphore:
            generated, raw_response = await asyncio.to_thread(
                generator.complete_json,
                topic_id=topic_id,
                system_prompt=system_prompt_for(config.prompt_profile),
                user_prompt=render_prompt(
                    narrative, ranked_docids, documents, config.prompt_profile
                ),
                response_schema=output_schema(),
            )
        _write_json(
            work_dir / "raw" / f"{topic_name}.json",
            _redact(raw_response, secrets),
        )
        record = normalize_generated_record(
            trim_to_word_limit(
                build_submission_record(
                generated,
                topic_id=topic_id,
                narrative=narrative,
                team_id=config.team_id,
                run_id=config.run_id,
                    run_desc=config.run_desc,
                )
            )
        )
        _validate_generated_submission_record(
            record,
            topic_id=topic_id,
            narrative=narrative,
            allowed_docids=ranked_docids,
            team_id=config.team_id,
            run_id=config.run_id,
            run_desc=config.run_desc,
        )
        _write_json(work_dir / "rows" / f"{topic_name}.json", record, compact=True)
        return topic_id, None
    except Exception as exc:
        if isinstance(exc, SemanticCompletionError):
            _write_json(
                work_dir / "raw" / f"{topic_name}.failed.json",
                _redact(exc.raw_response, secrets),
            )
        error_path = work_dir / "errors" / f"{topic_name}.txt"
        sanitized_error = _redact(f"{type(exc).__name__}: {exc}", secrets)
        _atomic_write_text(error_path, f"{sanitized_error}\n")
        return topic_id, sanitized_error


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    return first == second or first in second.parents or second in first.parents


def _validate_artifact_paths(config: RagGenerationConfig) -> None:
    inputs = {
        "queries_path": config.queries_path,
        "run_path": config.run_path,
        "documents_path": config.documents_path,
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


def _file_digest(path: Path) -> str:
    """Return a streaming sha256 so large document archives do not load into memory."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _generation_identity(config: RagGenerationConfig) -> dict[str, Any]:
    """Return the settings that make already-generated rows comparable.

    Resuming reuses rows produced by an earlier invocation. Those rows are revalidated for
    shape, but shape cannot detect that they came from a different model or prompt, so a resume
    after any of these changed would publish one file containing answers from two systems.

    The retrieval inputs are digested too. Regenerating them in place between create and resume
    would otherwise leave earlier rows grounded in evidence that no longer exists, and they
    would still validate whenever their references happen to survive.
    """
    return {
        # Bump when the recorded fields change, so an identity written by an older revision
        # fails closed instead of comparing unequal for a reason the operator cannot see.
        "identity_version": 2,
        "run_sha256": _file_digest(config.run_path),
        "documents_sha256": _file_digest(config.documents_path),
        "archive_member": config.archive_member,
        "model": config.model,
        "provider": config.provider,
        "api_base": config.api_base,
        "reasoning_effort": config.reasoning_effort,
        "prompt_profile": config.prompt_profile,
        "structured_output": config.structured_output,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_k": config.top_k,
        "max_document_words": config.max_document_words,
    }


def _enforce_generation_identity(config: RagGenerationConfig, work_dir: Path) -> None:
    """Refuse to resume into rows generated under different settings.

    Fails closed. Rows that predate identity tracking, or that were recorded by an older
    identity version, cannot be shown to match the current settings, so they are rejected rather
    than adopted: adopting them would publish one file mixing two systems.
    """
    identity_path = work_dir / "generation_identity.json"
    identity = _generation_identity(config)
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
    config: RagGenerationConfig, generator: JsonGenerator
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

    topics = select_queries(load_queries(config.queries_path), config.topic_ids)
    ranked = load_trec_run(config.run_path, {topic_id for topic_id, _ in topics}, config.top_k)
    wanted_docids = {docid for docids in ranked.values() for docid in docids}
    documents = load_documents(
        config.documents_path,
        config.archive_member,
        wanted_docids,
        config.max_document_words,
    )
    # After the fallible input loading, so a failed overwrite leaves no work directory behind.
    work_dir.mkdir(parents=True, exist_ok=True)
    _enforce_generation_identity(config, work_dir)

    rows_dir = work_dir / "rows"
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
            f"{len(failures)} topic(s) failed; inspect {work_dir / 'errors'} and rerun with "
            "experiment.mode: resume"
        )
    final_records: list[dict[str, Any]] = []
    for topic_id, narrative in topics:
        record = _saved_record(
            rows_dir / f"{_safe_topic_name(topic_id)}.json",
            topic_id=topic_id,
            narrative=narrative,
            allowed_docids=ranked[topic_id],
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


async def run_generation(config: RagGenerationConfig, generator: JsonGenerator) -> None:
    """Generate missing topic rows and atomically publish the organizer JSONL."""
    _validate_artifact_paths(config)
    lock_path = config.output_path.with_name(f".{config.output_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lock_path, timeout=0):
            await _run_generation_locked(config, generator)
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
        select_queries(load_queries(config.queries_path), config.topic_ids)
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
        asyncio.run(run_generation(config, generator))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        raise SystemExit(f"error: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
