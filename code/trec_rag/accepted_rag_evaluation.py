"""Authenticate and normalize provenance for immutable accepted RAG runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Any

from trec_rag.generation_handoff import PROMPT_CONTRACT_VERSION, load_generation_handoff


ACCEPTED_BINDING_SCHEMA_VERSION = "accepted_rag_evaluation_binding_v1"
MISSING_SOURCE_IDENTITY_REASON = "original generation identity was not preserved"
SINGLEPASS_IDENTITY_VERSION = 6
MULTISTAGE_IDENTITY_VERSION = 1
MULTISTAGE_TRIAL_CONTRACT_VERSION = "bounded_narrative_revision_trial_v8_screen_liveness"


@dataclass(frozen=True)
class AcceptedRunBinding:
    """Post-run receipt binding one accepted JSONL to its evidence handoff."""

    schema_version: str
    run_id: str
    run_desc: str
    team_id: str
    provider: str
    models: tuple[str, ...]
    submission_sha256: str
    bundle_metadata_sha256: str
    handoff_schema_version: str
    handoff_manifest_sha256: str
    topic_ids: tuple[str, ...]
    topic_context_sha256s: Mapping[str, str]
    source_identity_available: bool
    source_identity_sha256: str | None
    source_identity_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_desc": self.run_desc,
            "team_id": self.team_id,
            "provider": self.provider,
            "models": list(self.models),
            "submission_sha256": self.submission_sha256,
            "bundle_metadata_sha256": self.bundle_metadata_sha256,
            "handoff_schema_version": self.handoff_schema_version,
            "handoff_manifest_sha256": self.handoff_manifest_sha256,
            "topic_ids": list(self.topic_ids),
            "topic_context_sha256s": dict(sorted(self.topic_context_sha256s.items())),
            "source_identity_available": self.source_identity_available,
            "source_identity_sha256": self.source_identity_sha256,
            "source_identity_reason": self.source_identity_reason,
        }


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _require_sha256(value: object, field: str, path: Path) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{path}: invalid {field}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{path}: invalid {field}") from error
    return value


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        source = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{path}: cannot read JSON") from error
    try:
        value = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root is not an object")
    return value, source


def _load_submission(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        source = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{path}: cannot read accepted submission") from error
    rows: list[dict[str, Any]] = []
    lines = source.splitlines()
    if not lines:
        raise ValueError(f"{path}: accepted submission is empty")
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{path}:{number}: accepted submission has a blank line")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{number}: invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: row is not a JSON object")
        rows.append(value)
    return rows, source


def _topic_contexts_from_handoff(handoff: Any, path: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    topic_ids = tuple(topic.topic_id for topic in handoff.topics)
    contexts = {topic.topic_id: topic.context_sha256 for topic in handoff.topics}
    if len(topic_ids) != len(contexts):
        raise ValueError(f"{path}: accepted handoff has duplicate topic IDs")
    return topic_ids, contexts


def _metadata_run(metadata: Mapping[str, Any], submission_path: Path) -> Mapping[str, Any]:
    runs = metadata.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"{submission_path}: accepted bundle metadata has no runs")
    suffix_matches: list[Mapping[str, Any]] = []
    basename_matches: list[Mapping[str, Any]] = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        run_path = run.get("path")
        if not isinstance(run_path, str) or not run_path:
            continue
        # Bundle paths are relative to the bundle directory.  The basename check
        # intentionally tolerates callers passing either that path or an extracted
        # absolute path while still requiring an exact submission suffix.
        if submission_path.as_posix().endswith("/" + run_path.lstrip("/")):
            suffix_matches.append(run)
        if Path(run_path).name == submission_path.name:
            basename_matches.append(run)
    matches = suffix_matches if suffix_matches else basename_matches
    if len(matches) != 1:
        raise ValueError(
            f"{submission_path}: accepted bundle metadata run entry does not match "
            "submission suffix"
        )
    return matches[0]


def _topic_ids_from_source_identity(
    source: Mapping[str, Any],
    source_path: Path,
    *,
    handoff_schema_version: str,
    handoff_manifest_sha256: str,
    run_id: str,
    handoff_topic_ids: tuple[str, ...],
    handoff_contexts: Mapping[str, str],
) -> None:
    """Validate either the preserved multi-stage or historical identity receipt."""
    source_run_id = source.get("submission_run_id")
    topic_items = source.get("topics")
    if source_run_id is not None or topic_items is not None:
        if source.get("identity_version") != MULTISTAGE_IDENTITY_VERSION:
            raise ValueError("accepted source identity does not match multi-stage identity version")
        if source.get("trial_contract_version") != MULTISTAGE_TRIAL_CONTRACT_VERSION:
            raise ValueError("accepted source identity does not match multi-stage trial contract")
        if not isinstance(source_run_id, str) or not source_run_id.strip():
            raise ValueError(f"{source_path}: accepted source identity has no submission_run_id")
        if source_run_id != run_id:
            raise ValueError(f"accepted source identity does not match submission run_id")
        if source.get("handoff_schema_version") != handoff_schema_version:
            raise ValueError(f"accepted source identity does not match handoff schema")
        if source.get("handoff_manifest_sha256") != handoff_manifest_sha256:
            raise ValueError(f"accepted source identity does not match handoff manifest")
        if not isinstance(topic_items, list) or not topic_items:
            raise ValueError(f"{source_path}: accepted source identity has no topics")
        source_topic_ids: list[str] = []
        source_contexts: dict[str, str] = {}
        for index, item in enumerate(topic_items):
            if not isinstance(item, Mapping):
                raise ValueError(f"{source_path}: source identity topic {index} is invalid")
            topic_id = item.get("topic_id")
            context = item.get("context_sha256")
            if not isinstance(topic_id, str) or not topic_id.strip():
                raise ValueError(f"{source_path}: source identity topic {index} has invalid topic_id")
            if topic_id in source_contexts:
                raise ValueError(f"{source_path}: source identity has duplicate topic_id {topic_id}")
            source_topic_ids.append(topic_id)
            source_contexts[topic_id] = _require_sha256(context, "context_sha256", source_path)
        if tuple(source_topic_ids) != handoff_topic_ids or source_contexts != dict(handoff_contexts):
            raise ValueError(f"accepted source identity does not match topic contexts")
        return

    # This branch is useful when a caller preserved a historical single-pass
    # identity alongside an accepted file.  It is not synthesized when missing.
    source_run_id = source.get("run_id")
    selected_items = source.get("selected_topics")
    if source.get("identity_version") != SINGLEPASS_IDENTITY_VERSION:
        raise ValueError("accepted source identity does not match single-pass identity version")
    if source.get("prompt_contract_version") != PROMPT_CONTRACT_VERSION:
        raise ValueError("accepted source identity does not match single-pass prompt contract")
    if not isinstance(source_run_id, str) or source_run_id != run_id:
        raise ValueError(f"accepted source identity does not match submission run_id")
    if source.get("handoff_schema_version") != handoff_schema_version:
        raise ValueError(f"accepted source identity does not match handoff schema")
    if source.get("handoff_manifest_sha256") != handoff_manifest_sha256:
        raise ValueError(f"accepted source identity does not match handoff manifest")
    if not isinstance(selected_items, list) or not selected_items:
        raise ValueError(f"{source_path}: source identity has no selected_topics")
    source_topic_ids: list[str] = []
    source_contexts: dict[str, str] = {}
    for index, item in enumerate(selected_items):
        if not isinstance(item, Mapping):
            raise ValueError(f"{source_path}: selected topic {index} is invalid")
        topic_id = item.get("topic_id")
        if not isinstance(topic_id, str) or not topic_id.strip() or topic_id in source_contexts:
            raise ValueError(f"{source_path}: selected topic {index} has invalid topic_id")
        source_topic_ids.append(topic_id)
        source_contexts[topic_id] = _require_sha256(
            item.get("context_sha256"), "context_sha256", source_path
        )
    if tuple(source_topic_ids) != handoff_topic_ids or source_contexts != dict(handoff_contexts):
        raise ValueError(f"accepted source identity does not match topic contexts")


def build_accepted_run_binding(
    submission_path: Path,
    bundle_metadata_path: Path,
    handoff_manifest_path: Path,
    *,
    source_identity_path: Path | None = None,
) -> AcceptedRunBinding:
    """Authenticate one immutable accepted submission and its evidence handoff."""
    submission_path = Path(submission_path)
    bundle_metadata_path = Path(bundle_metadata_path)
    handoff_manifest_path = Path(handoff_manifest_path)
    records, submission_bytes = _load_submission(submission_path)
    metadata, metadata_bytes = _load_json(bundle_metadata_path)
    run = _metadata_run(metadata, submission_path)
    handoff = load_generation_handoff(handoff_manifest_path)
    handoff_topic_ids, topic_contexts = _topic_contexts_from_handoff(
        handoff, handoff_manifest_path
    )

    expected_sha = _require_sha256(run.get("sha256"), "sha256", bundle_metadata_path)
    actual_sha = _digest(submission_bytes)
    if expected_sha != actual_sha:
        raise ValueError("accepted submission receipt does not match submission bytes")
    expected_bytes = run.get("bytes")
    if type(expected_bytes) is not int or expected_bytes != len(submission_bytes):
        raise ValueError("accepted submission receipt does not match submission bytes")
    expected_lines = run.get("line_count")
    if type(expected_lines) is not int or expected_lines != len(records):
        raise ValueError("accepted submission receipt does not match submission line count")

    run_id = run.get("run_id")
    run_desc = run.get("run_desc")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"{submission_path}: accepted run entry has invalid run_id")
    if not isinstance(run_desc, str) or not run_desc.strip():
        raise ValueError(f"{submission_path}: accepted run entry has invalid run_desc")
    record_ids: list[str] = []
    team_ids: set[str] = set()
    record_descs: set[str] = set()
    for index, record in enumerate(records):
        metadata_record = record.get("metadata")
        if not isinstance(metadata_record, Mapping):
            raise ValueError(f"{submission_path}:{index + 1}: missing metadata")
        embedded_run_id = metadata_record.get("run_id")
        if embedded_run_id != run_id:
            raise ValueError("accepted run metadata does not match embedded run_id")
        topic_id = metadata_record.get("narrative_id")
        if not isinstance(topic_id, str) or not topic_id.strip() or topic_id in record_ids:
            raise ValueError("accepted submission topic IDs do not match handoff")
        handoff_topic = next(
            (topic for topic in handoff.topics if topic.topic_id == topic_id), None
        )
        if handoff_topic is None or metadata_record.get("narrative") != handoff_topic.narrative:
            raise ValueError("accepted submission narrative does not match handoff")
        record_ids.append(topic_id)
        team_id = metadata_record.get("team_id")
        if not isinstance(team_id, str) or not team_id.strip():
            raise ValueError(f"{submission_path}:{index + 1}: invalid team_id")
        team_ids.add(team_id)
        embedded_desc = metadata_record.get("run_desc")
        if not isinstance(embedded_desc, str) or not embedded_desc.strip():
            raise ValueError(f"{submission_path}:{index + 1}: invalid run_desc")
        record_descs.add(embedded_desc)
    if tuple(record_ids) != handoff_topic_ids:
        raise ValueError("accepted submission topic IDs do not match handoff")
    if len(team_ids) != 1 or len(record_descs) != 1:
        raise ValueError("accepted submission metadata is not consistent")
    if record_descs != {run_desc}:
        raise ValueError("accepted run metadata does not match embedded run_desc")

    source = metadata.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{bundle_metadata_path}: accepted bundle metadata has no source")
    source_handoff_hash = _require_sha256(
        source.get("generation_handoff_manifest_sha256"),
        "generation_handoff_manifest_sha256",
        bundle_metadata_path,
    )
    if source_handoff_hash != handoff.manifest_sha256:
        raise ValueError("accepted bundle metadata does not match handoff manifest")
    source_topic_count = source.get("topic_count")
    if type(source_topic_count) is not int or source_topic_count != len(records):
        raise ValueError("accepted bundle metadata does not match topic count")
    if handoff.topic_count != len(records):
        raise ValueError("accepted handoff does not match submission topic count")
    metadata_topic_count = metadata.get("topic_count")
    if metadata_topic_count is not None and (
        type(metadata_topic_count) is not int or metadata_topic_count != len(records)
    ):
        raise ValueError("accepted bundle metadata does not match topic count")

    source_identity_available = False
    source_identity_sha256: str | None = None
    source_identity_reason: str | None = MISSING_SOURCE_IDENTITY_REASON
    if source_identity_path is not None:
        source_identity_path = Path(source_identity_path)
        source_payload, source_bytes = _load_json(source_identity_path)
        _topic_ids_from_source_identity(
            source_payload,
            source_identity_path,
            handoff_schema_version=handoff.schema_version,
            handoff_manifest_sha256=handoff.manifest_sha256,
            run_id=run_id,
            handoff_topic_ids=handoff_topic_ids,
            handoff_contexts=topic_contexts,
        )
        source_identity_available = True
        source_identity_sha256 = _digest(source_bytes)
        source_identity_reason = None

    provider = run.get("provider") or metadata.get("provider") or "unknown"
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"{bundle_metadata_path}: invalid provider")
    model_value = run.get("models")
    if model_value is None:
        by_model = run.get("provider_calls_by_model")
        if isinstance(by_model, Mapping):
            model_value = list(by_model)
    if model_value is None and isinstance(run.get("model"), str):
        model_value = [run["model"]]
    if not isinstance(model_value, list) or any(
        not isinstance(model, str) or not model.strip() for model in model_value
    ):
        raise ValueError(f"{bundle_metadata_path}: invalid models")
    models = tuple(sorted(dict.fromkeys(model_value)))
    return AcceptedRunBinding(
        schema_version=ACCEPTED_BINDING_SCHEMA_VERSION,
        run_id=run_id,
        run_desc=run_desc,
        team_id=next(iter(team_ids)),
        provider=provider,
        models=models,
        submission_sha256=actual_sha,
        bundle_metadata_sha256=_digest(metadata_bytes),
        handoff_schema_version=handoff.schema_version,
        handoff_manifest_sha256=handoff.manifest_sha256,
        topic_ids=handoff_topic_ids,
        topic_context_sha256s=dict(topic_contexts),
        source_identity_available=source_identity_available,
        source_identity_sha256=source_identity_sha256,
        source_identity_reason=source_identity_reason,
    )


def write_accepted_run_binding(binding: AcceptedRunBinding, output_path: Path) -> Path:
    """Atomically write a canonical private accepted-run receipt."""
    if not isinstance(binding, AcceptedRunBinding):
        raise TypeError("binding must be an AcceptedRunBinding")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_path.parent, 0o700)
    if output_path.is_symlink():
        raise ValueError(f"binding destination is a symbolic link: {output_path}")
    contents = (
        json.dumps(
            binding.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(handle.fileno(), 0o600)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        os.chmod(output_path, 0o600)
        descriptor = os.open(output_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output_path


__all__ = [
    "ACCEPTED_BINDING_SCHEMA_VERSION",
    "AcceptedRunBinding",
    "MISSING_SOURCE_IDENTITY_REASON",
    "MULTISTAGE_IDENTITY_VERSION",
    "MULTISTAGE_TRIAL_CONTRACT_VERSION",
    "SINGLEPASS_IDENTITY_VERSION",
    "build_accepted_run_binding",
    "write_accepted_run_binding",
]
