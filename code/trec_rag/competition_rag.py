"""Strict configuration and organizer-compatible inputs for competition RAG."""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

import yaml

from trec_rag.repo_env import find_repo_root, shared_checkout_root


_SCHEMA_VERSION = "competition_rag_config_v1"
_SAFE_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


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
    api_base: str
    api_key_env: str
    model: str
    reasoning_effort: str
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
        api_base=_text(generation, "api_base", "generation"),
        api_key_env=_text(generation, "api_key_env", "generation"),
        model=_text(generation, "model", "generation"),
        reasoning_effort=reasoning_effort,
        temperature=_optional_finite_float(generation, "temperature", "generation"),
        max_tokens=_positive_int(generation, "max_tokens", "generation"),
        timeout_seconds=_positive_float(generation, "timeout_seconds", "generation"),
        transport_max_attempts=_positive_int(
            generation, "transport_max_attempts", "generation"
        ),
    )


def load_queries(path: Path) -> list[tuple[str, str]]:
    """Read the canonical headerless ``narrative_id<TAB>narrative`` topic TSV."""
    queries: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = raw_line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected exactly two TSV fields")
        raw_topic_id, narrative = fields
        topic_id = raw_topic_id.strip()
        if line_number == 1 and topic_id.lower() in {"qid", "query_id", "topic_id", "narrative_id"}:
            raise ValueError(f"{path}:{line_number}: topic TSV must not have a header")
        if raw_topic_id != topic_id:
            raise ValueError(f"{path}:{line_number}: topic id must not have surrounding whitespace")
        if not topic_id or not narrative.strip():
            raise ValueError(f"{path}:{line_number}: empty topic id or narrative")
        queries.append((topic_id, narrative))
    if not queries:
        raise ValueError(f"{path}: no topics found")
    ids = [topic_id for topic_id, _ in queries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate topic ids")
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
    if not zipfile.is_zipfile(path):
        if archive_member is not None:
            raise ValueError("archive_member requires a ZIP document input")
        with Path(path).open(encoding="utf-8") as handle:
            yield handle
        return
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name for name in archive.namelist() if name.lower().endswith((".jsonl", ".json"))
        ]
        member = archive_member or (candidates[0] if len(candidates) == 1 else None)
        if member is None or member not in candidates:
            raise ValueError(
                f"{path}: choose one JSONL ZIP member with archive_member; "
                f"found {candidates}"
            )
        with archive.open(member) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as handle:
                yield handle


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
