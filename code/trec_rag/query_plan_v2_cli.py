"""Crash-safe serial runner for query-planner v2 experiments.

The runner treats model responses as experimental evidence.  It freezes the
reference analyzer before the first request, writes exact HTTP bytes before
decoding, and commits one immutable terminal outcome per topic.  JSONL files
and the manifest are derived views that can be rebuilt without a model or
analyzer service.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from trec_rag.query_analyzer import RemoteLuceneQueryAnalyzer
from trec_rag.query_planner import (
    TOKENIZER_VERSION,
    V2_PROMPT_VERSION,
    V2_RENDERER_VERSION,
    V2_SCHEMA_VERSION,
    QueryPlanGeneratorV2,
    QueryPlanHttpTransportError,
    query_plan_v2_json_schema,
    tokenize_narrative,
)
from trec_rag.repo_env import find_repo_root, load_repo_env
from trec_rag.topics import Topic, load_topics


DIAGNOSTIC_TOPIC_IDS = ("144", "213", "224", "407", "515")
GPT_OSS_20B_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
PINNED_VLLM_IMAGE_DIGEST = (
    "sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28"
)
SYNTHETIC_SMOKE_TOPIC = Topic(
    id="synthetic_transport_v2_1",
    title="Synthetic versioned transport-only planner check",
    narrative=(
        "Compare battery-free water-quality sensors used in mountain streams and "
        "urban canals, and identify the maintenance indicators reported for each "
        "setting."
    ),
)
RUN_KINDS = ("synthetic_transport_smoke", "diagnostic_first_emission")
OUTCOME_STATUSES = frozenset(
    {
        "success",
        "invalid_json",
        "plan_validation_error",
        "render_validation_error",
        "http_error",
        "timeout",
        "analyzer_integrity_error",
        "unexpected_exception",
    }
)
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_request_json(value: object) -> str:
    """Match the generator's frozen request/schema JSON hash serialization."""

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _atomic_write(path: Path, content: str, *, replace: bool) -> None:
    """Fsync a file and its directory; create-only writes never overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
        if replace:
            os.replace(temporary_name, path)
        else:
            # Hard-linking is an atomic create-if-absent operation on the same
            # filesystem.  It closes the preflight/commit overwrite race.
            os.link(temporary_name, path)
            os.unlink(temporary_name)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, payload: Mapping[str, object], *, replace: bool) -> None:
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        replace=replace,
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_write(path, content, replace=True)


def _validate_run_id(value: str) -> str:
    if value in {".", ".."} or not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(
            "run_id must contain only letters, digits, dot, underscore, or hyphen"
        )
    return value


def _is_loopback_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def _topic_file_component(topic_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]", "_", topic_id).strip(".") or "topic"
    digest = hashlib.sha256(topic_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:80]}-{digest}"


def _outcome_path(outcome_dir: Path, ordinal: int, topic: Topic) -> Path:
    return outcome_dir / f"{ordinal:02d}_{_topic_file_component(topic.id)}.json"


def _raw_path(outcome_dir: Path, ordinal: int, topic: Topic) -> Path:
    return (
        outcome_dir
        / "raw_responses"
        / f"{ordinal:02d}_{_topic_file_component(topic.id)}.json"
    )


def _parse_topic_ids(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("topic IDs must not be empty")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("topic IDs must be unique")
    return result


def _select_topics(topics: list[Topic], topic_ids: tuple[str, ...]) -> list[Topic]:
    by_id = {topic.id: topic for topic in topics}
    missing = [topic_id for topic_id in topic_ids if topic_id not in by_id]
    if missing:
        raise ValueError("unknown topic ID(s): " + ", ".join(missing))
    return [by_id[topic_id] for topic_id in topic_ids]


def _topics_from_metadata(metadata: Mapping[str, object]) -> list[Topic]:
    values = metadata.get("topics")
    if not isinstance(values, list) or not values:
        raise ValueError("run metadata lacks ordered topic records")
    topics: list[Topic] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"run metadata topic {index} is not an object")
        try:
            topics.append(
                Topic(
                    id=str(value["id"]),
                    title=str(value["title"]),
                    narrative=str(value["narrative"]),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"run metadata topic {index} lacks {exc.args[0]}"
            ) from exc
    if metadata.get("request_order") != [topic.id for topic in topics]:
        raise ValueError("run metadata topics do not match request_order")
    return topics


def _serialize(value: object) -> object:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    return value


def _response_summary(response: Mapping[str, object]) -> dict[str, object]:
    usage = response.get("usage")
    return {
        "http_status": getattr(response, "http_status", None),
        "response_id": response.get("id"),
        "response_model": response.get("model"),
        "usage": dict(usage) if isinstance(usage, Mapping) else {},
    }


def _frozen_request_config(
    generator: QueryPlanGeneratorV2,
    topic: Topic,
    *,
    analyzer_fingerprint: Mapping[str, object],
    analyzer_fingerprint_sha256: str,
) -> dict[str, object]:
    request = generator.request_payload(topic)
    response_format = request.get("response_format")
    schema: object = None
    if isinstance(response_format, Mapping):
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, Mapping):
            schema = json_schema.get("schema")
    messages = request.get("messages")
    prompt = None
    if isinstance(messages, list) and messages and isinstance(messages[0], Mapping):
        prompt = messages[0].get("content")
    return {
        "provider": "openai_compatible_chat_completions",
        "base_url": generator.base_url,
        "model": generator.model,
        "model_revision": generator.model_revision,
        "schema_version": V2_SCHEMA_VERSION,
        "prompt_version": V2_PROMPT_VERSION,
        "renderer_version": V2_RENDERER_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "schema_sha256": _sha256_request_json(schema),
        "prompt_sha256": hashlib.sha256(str(prompt).encode("utf-8")).hexdigest(),
        "request_sha256": _sha256_request_json(request),
        "request_payload": request,
        "analyzer_fingerprint": dict(analyzer_fingerprint),
        "analyzer_fingerprint_sha256": analyzer_fingerprint_sha256,
        "instruction_role": generator.instruction_role,
        "reasoning_effort": generator.reasoning_effort,
        "enable_thinking": generator.enable_thinking,
        "max_tokens": generator.max_tokens,
        "temperature": generator.temperature,
        "top_p": generator.top_p,
        "top_k": generator.top_k,
        "presence_penalty": generator.presence_penalty,
        "seed": generator.seed,
    }


def _validate_frozen_request_config(
    config: Mapping[str, object],
    *,
    run_settings: Mapping[str, object],
    fingerprint_sha256: str,
    path: Path,
) -> None:
    fingerprint = config.get("analyzer_fingerprint")
    if (
        not isinstance(fingerprint, Mapping)
        or _sha256_json(fingerprint) != fingerprint_sha256
        or config.get("analyzer_fingerprint_sha256") != fingerprint_sha256
    ):
        raise ValueError(f"outcome analyzer fingerprint mismatch: {path}")
    shared_keys = (
        "provider",
        "base_url",
        "model",
        "model_revision",
        "schema_version",
        "prompt_version",
        "renderer_version",
        "tokenizer_version",
        "instruction_role",
        "reasoning_effort",
        "enable_thinking",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "seed",
    )
    mismatches = [
        key for key in shared_keys if config.get(key) != run_settings.get(key)
    ]
    if mismatches:
        raise ValueError(
            f"outcome frozen settings mismatch in {path}: {', '.join(mismatches)}"
        )
    request = config.get("request_payload")
    if not isinstance(request, Mapping):
        raise ValueError(f"outcome lacks frozen request payload: {path}")
    if config.get("request_sha256") != _sha256_request_json(request):
        raise ValueError(f"outcome request hash mismatch: {path}")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], Mapping):
        raise ValueError(f"outcome request lacks its planner prompt: {path}")
    prompt = messages[0].get("content")
    if config.get("prompt_sha256") != hashlib.sha256(
        str(prompt).encode("utf-8")
    ).hexdigest():
        raise ValueError(f"outcome prompt hash mismatch: {path}")
    response_format = request.get("response_format")
    schema: object = None
    if isinstance(response_format, Mapping):
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, Mapping):
            schema = json_schema.get("schema")
    if config.get("schema_sha256") != _sha256_request_json(schema):
        raise ValueError(f"outcome schema hash mismatch: {path}")


def _validate_nested_provenance(
    provenance: object,
    *,
    config: Mapping[str, object],
    fingerprint_sha256: str,
    path: Path,
) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError(f"outcome lacks generation provenance: {path}")
    keys = (
        "provider",
        "base_url",
        "model",
        "model_revision",
        "schema_version",
        "prompt_version",
        "renderer_version",
        "tokenizer_version",
        "schema_sha256",
        "prompt_sha256",
        "request_sha256",
        "instruction_role",
        "reasoning_effort",
        "enable_thinking",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "seed",
    )
    mismatches = [key for key in keys if provenance.get(key) != config.get(key)]
    if mismatches:
        raise ValueError(
            f"generation provenance mismatch in {path}: {', '.join(mismatches)}"
        )
    fingerprint = provenance.get("analyzer_fingerprint")
    if not isinstance(fingerprint, Mapping) or _sha256_json(fingerprint) != fingerprint_sha256:
        raise ValueError(f"generation provenance analyzer mismatch: {path}")


def _context_preflight(
    *,
    generator: QueryPlanGeneratorV2,
    topic: Topic,
    tokenizer_url: str,
    timeout: float,
) -> dict[str, object]:
    """Count the exact chat prompt locally without running model inference."""

    request = generator.request_payload(topic)
    payload: dict[str, object] = {
        "model": request["model"],
        "messages": request["messages"],
        "add_generation_prompt": True,
        "return_token_strs": False,
    }
    if "chat_template_kwargs" in request:
        payload["chat_template_kwargs"] = request["chat_template_kwargs"]
    http_request = urllib.request.Request(
        tokenizer_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("server tokenizer response must be an object")
    prompt_tokens = value.get("count")
    max_model_len = value.get("max_model_len")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 1
        or isinstance(max_model_len, bool)
        or not isinstance(max_model_len, int)
        or max_model_len < 1
    ):
        raise ValueError("server tokenizer response lacks valid context counts")
    requested_total = prompt_tokens + generator.max_tokens
    if requested_total > max_model_len:
        raise ValueError(
            f"topic {topic.id} prompt plus max output exceeds server context: "
            f"{prompt_tokens}+{generator.max_tokens}>{max_model_len}"
        )
    return {
        "topic_id": topic.id,
        "tokenizer_url": tokenizer_url,
        "prompt_tokens": prompt_tokens,
        "max_output_tokens": generator.max_tokens,
        "requested_total_tokens": requested_total,
        "max_model_len": max_model_len,
        "remaining_context_tokens": max_model_len - requested_total,
    }


def _launch_command_pins_xgrammar(command: object) -> bool:
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        return False
    for index, item in enumerate(command):
        if item == "--structured-outputs-config" and index + 1 < len(command):
            try:
                value = json.loads(command[index + 1])
            except json.JSONDecodeError:
                return False
            return isinstance(value, Mapping) and value.get("backend") == "xgrammar"
        if item == "--structured-outputs-config.backend" and index + 1 < len(command):
            return command[index + 1] == "xgrammar"
    return False


def _launch_value(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _verify_schema_preflight(
    path: Path,
    *,
    topics: list[Topic],
) -> dict[str, object]:
    """Bind compiler-only schema evidence to the exact pending run."""

    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("schema preflight manifest must be an object")
    expected = {
        "manifest_version": "query_plan_v2_schema_preflight_v1",
        "schema_version": V2_SCHEMA_VERSION,
        "prompt_version": V2_PROMPT_VERSION,
        "required_server_backend": "xgrammar",
        "all_passed": True,
    }
    mismatches = [key for key, item in expected.items() if value.get(key) != item]
    if mismatches:
        raise ValueError(
            "schema preflight manifest mismatch: " + ", ".join(mismatches)
        )
    compiler = value.get("compiler")
    if not isinstance(compiler, Mapping) or any(
        compiler.get(key) != expected_value
        for key, expected_value in {
            "vllm": "0.24.0",
            "xgrammar": "0.2.3",
            "xgrammar_strict": "pass",
            "llguidance_check": "pass",
        }.items()
    ):
        raise ValueError("schema preflight compiler identity/result mismatch")
    container = value.get("container")
    launch_command = (
        container.get("launch_command") if isinstance(container, Mapping) else None
    )
    if (
        not isinstance(container, Mapping)
        or container.get("image_digest") != PINNED_VLLM_IMAGE_DIGEST
        or not isinstance(launch_command, list)
        or not all(isinstance(item, str) for item in launch_command)
        or not launch_command
        or launch_command[0] != "openai/gpt-oss-20b"
        or _launch_value(launch_command, "--revision") != GPT_OSS_20B_REVISION
        or _launch_value(launch_command, "--served-model-name") != "gpt-oss-local"
        or _launch_value(launch_command, "--max-model-len") != "8192"
        or not _launch_command_pins_xgrammar(launch_command)
    ):
        raise ValueError(
            "schema preflight container must pin the expected image and XGrammar backend"
        )
    rows = value.get("schemas")
    if not isinstance(rows, list):
        raise ValueError("schema preflight manifest lacks schema rows")
    by_id = {
        str(row.get("topic_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("topic_id") is not None
    }
    for topic in topics:
        row = by_id.get(topic.id)
        if not isinstance(row, Mapping):
            raise ValueError(f"schema preflight lacks topic {topic.id}")
        tape = tokenize_narrative(topic.narrative)
        schema = query_plan_v2_json_schema(
            topic_id=topic.id,
            token_count=tape.token_count,
        )
        if (
            row.get("narrative_sha256") != tape.narrative_sha256
            or row.get("token_count") != tape.token_count
            or row.get("schema_sha256") != _sha256_request_json(schema)
            or row.get("unsupported_feature_count") != 0
            or row.get("xgrammar_strict") != "pass"
        ):
            raise ValueError(f"schema preflight evidence mismatch for topic {topic.id}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "manifest": value,
    }


def _read_local_json(url: str, *, timeout: float) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"local runtime endpoint did not return an object: {url}")
    return value


def _verify_live_runtime(
    schema_preflight: Mapping[str, object],
    *,
    base_url: str,
    tokenizer_url: str,
    expected_model: str,
    timeout: float,
) -> dict[str, object]:
    """Prove the loopback endpoint is the exact preflighted Podman runtime."""

    manifest = schema_preflight.get("manifest")
    container_record = (
        manifest.get("container") if isinstance(manifest, Mapping) else None
    )
    if not isinstance(container_record, Mapping):
        raise ValueError("schema preflight lacks container runtime identity")
    container_name = container_record.get("container_name")
    if not isinstance(container_name, str) or not container_name:
        raise ValueError("schema preflight lacks a container name")
    inspected_process = subprocess.run(
        ["podman", "inspect", container_name],
        text=True,
        capture_output=True,
        check=True,
    )
    inspected = json.loads(inspected_process.stdout)
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise ValueError("live container inspection did not return one record")
    row = inspected[0]
    state = row.get("State")
    config = row.get("Config")
    network = row.get("NetworkSettings")
    if (
        not isinstance(state, Mapping)
        or state.get("Running") is not True
        or not isinstance(config, Mapping)
        or not isinstance(network, Mapping)
    ):
        raise ValueError("preflighted local model container is not running")
    launch_command = config.get("Cmd")
    if (
        row.get("Image") != container_record.get("image_id")
        or launch_command != container_record.get("launch_command")
    ):
        raise ValueError("live container image or launch command changed after preflight")
    image_process = subprocess.run(
        ["podman", "image", "inspect", str(row["Image"])],
        text=True,
        capture_output=True,
        check=True,
    )
    image_rows = json.loads(image_process.stdout)
    if (
        not isinstance(image_rows, list)
        or len(image_rows) != 1
        or image_rows[0].get("Digest") != container_record.get("image_digest")
    ):
        raise ValueError("live container image digest changed after preflight")

    parsed_base = urllib.parse.urlparse(base_url)
    parsed_tokenizer = urllib.parse.urlparse(tokenizer_url)
    if (
        parsed_base.hostname != parsed_tokenizer.hostname
        or parsed_base.port != parsed_tokenizer.port
    ):
        raise ValueError("model and tokenizer endpoints must share one loopback server")
    container_port = _launch_value(list(launch_command), "--port")
    ports = network.get("Ports")
    bindings = (
        ports.get(f"{container_port}/tcp")
        if isinstance(ports, Mapping) and container_port is not None
        else None
    )
    if not isinstance(bindings, list) or not any(
        isinstance(binding, Mapping)
        and binding.get("HostIp") == parsed_base.hostname
        and binding.get("HostPort") == str(parsed_base.port)
        for binding in bindings
    ):
        raise ValueError("loopback model endpoint is not bound to the preflighted container")

    server_root = base_url.removesuffix("/v1")
    version_response = _read_local_json(
        server_root + "/version", timeout=timeout
    )
    models_response = _read_local_json(base_url + "/models", timeout=timeout)
    compiler = manifest.get("compiler")
    if (
        not isinstance(compiler, Mapping)
        or version_response.get("version") != compiler.get("vllm")
    ):
        raise ValueError("live vLLM version differs from compiler preflight")
    models = models_response.get("data")
    matching = [
        model
        for model in models
        if isinstance(model, Mapping) and model.get("id") == expected_model
    ] if isinstance(models, list) else []
    if len(matching) != 1:
        raise ValueError("live server does not expose the frozen served model alias")
    model = matching[0]
    if (
        model.get("root") != "openai/gpt-oss-20b"
        or model.get("max_model_len") != 8192
    ):
        raise ValueError("live served model root/context differs from preflight")
    return {
        "verified_at": _utc_now(),
        "container_name": container_name,
        "image_id": row.get("Image"),
        "image_digest": image_rows[0].get("Digest"),
        "launch_command": launch_command,
        "port_bindings": bindings,
        "version_response": version_response,
        "model_record": dict(model),
    }


def _exception_status(exc: Exception) -> str:
    if isinstance(exc, QueryPlanHttpTransportError):
        return "http_error"
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    if any(isinstance(item, (TimeoutError, urllib.error.URLError)) for item in chain):
        return "timeout" if any(isinstance(item, TimeoutError) for item in chain) else "http_error"
    lowered = str(exc).casefold()
    if "analyzer" in lowered or "fingerprint" in lowered:
        return "analyzer_integrity_error"
    return "unexpected_exception"


def _fallback_query(topic: Topic) -> dict[str, object]:
    return {
        "variant_name": "original",
        "source_type": "original",
        "query_text": topic.narrative,
        "components": [topic.narrative],
        "unique_content_tokens": [],
        "renderer_budget_exception": False,
    }


def _assert_result_integrity(
    *,
    topic: Topic,
    result: object,
    analyzer_fingerprint: Mapping[str, object],
    expected_response_model: str,
) -> None:
    outcome = result.outcome
    provenance = result.provenance
    if provenance.response_model != expected_response_model:
        raise ValueError(
            "served response model differs from the frozen local model alias"
        )
    if provenance.analyzer_fingerprint != dict(analyzer_fingerprint):
        raise ValueError("result analyzer fingerprint differs from frozen run")
    if outcome.used_fallback:
        if outcome.plan is not None or outcome.failure is None:
            raise ValueError("fallback outcome has inconsistent plan/failure fields")
        if len(outcome.rendered_queries) != 1:
            raise ValueError("fallback outcome must contain one original query")
        fallback = outcome.rendered_queries[0]
        if (
            fallback.variant_name != "original"
            or fallback.source_type != "original"
            or fallback.query_text != topic.narrative
            or fallback.components != (topic.narrative,)
        ):
            raise ValueError("fallback outcome is not exact original-only retrieval")
    elif outcome.plan is None or outcome.failure is not None:
        raise ValueError("successful outcome has inconsistent plan/failure fields")
    elif outcome.plan.analyzer_fingerprint.to_dict() != dict(analyzer_fingerprint):
        raise ValueError("plan analyzer fingerprint differs from frozen run")


def _validate_raw_record(
    raw_path: Path,
    *,
    run_id: str,
    run_kind: str,
    ordinal: int,
    topic_id: str,
) -> dict[str, object]:
    value = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"raw response is not an object: {raw_path}")
    expected = {
        "run_id": run_id,
        "run_kind": run_kind,
        "request_ordinal": ordinal,
        "topic_id": topic_id,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError(f"raw response identity mismatch: {raw_path}")
    encoded = value.get("raw_body_base64")
    if not isinstance(encoded, str):
        raise ValueError(f"raw response lacks base64 bytes: {raw_path}")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError(f"raw response base64 is invalid: {raw_path}") from exc
    if value.get("raw_body_bytes") != len(raw):
        raise ValueError(f"raw response byte length mismatch: {raw_path}")
    if value.get("raw_body_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"raw response hash mismatch: {raw_path}")
    return value


def _rebuild(
    *,
    output: Path,
    failure_output: Path,
    outcome_dir: Path,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    topics = _topics_from_metadata(metadata)
    run_id = str(metadata.get("run_id"))
    run_kind = str(metadata.get("run_kind"))
    frozen = metadata.get("frozen_model_settings")
    if not isinstance(frozen, Mapping):
        raise ValueError("run metadata lacks frozen_model_settings")
    fingerprint = frozen.get("analyzer_fingerprint")
    fingerprint_sha = frozen.get("analyzer_fingerprint_sha256")
    if not isinstance(fingerprint, Mapping) or fingerprint_sha != _sha256_json(fingerprint):
        raise ValueError("run metadata analyzer fingerprint/hash mismatch")

    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    missing: list[str] = []
    raw_only_topics: list[str] = []
    status_counts: dict[str, int] = {}
    expected_outcomes: set[Path] = set()
    expected_raw: set[Path] = set()
    for ordinal, topic in enumerate(topics, start=1):
        path = _outcome_path(outcome_dir, ordinal, topic)
        expected_outcomes.add(path.resolve())
        topic_raw_path = _raw_path(outcome_dir, ordinal, topic).resolve()
        raw_record: dict[str, object] | None = None
        if topic_raw_path.exists():
            expected_raw.add(topic_raw_path)
            raw_record = _validate_raw_record(
                topic_raw_path,
                run_id=run_id,
                run_kind=run_kind,
                ordinal=ordinal,
                topic_id=topic.id,
            )
            raw_config = raw_record.get("frozen_request_config")
            if not isinstance(raw_config, Mapping):
                raise ValueError(
                    f"raw response lacks frozen request config: {topic_raw_path}"
                )
            _validate_frozen_request_config(
                raw_config,
                run_settings=frozen,
                fingerprint_sha256=str(fingerprint_sha),
                path=topic_raw_path,
            )
        if not path.exists():
            missing.append(topic.id)
            if raw_record is not None:
                raw_only_topics.append(topic.id)
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"outcome is not an object: {path}")
        expected_identity = {
            "run_id": run_id,
            "run_kind": run_kind,
            "request_ordinal": ordinal,
            "topic_id": topic.id,
        }
        if any(value.get(key) != item for key, item in expected_identity.items()):
            raise ValueError(f"outcome identity mismatch: {path}")
        status = value.get("status")
        if status not in OUTCOME_STATUSES:
            raise ValueError(f"unsupported outcome status in {path}: {status}")
        status_counts[str(status)] = status_counts.get(str(status), 0) + 1
        frozen_config = value.get("frozen_request_config")
        if not isinstance(frozen_config, Mapping):
            raise ValueError(f"outcome lacks frozen request config: {path}")
        _validate_frozen_request_config(
            frozen_config,
            run_settings=frozen,
            fingerprint_sha256=str(fingerprint_sha),
            path=path,
        )
        raw_name = value.get("raw_response_path")
        if raw_name is not None:
            raw_path = Path(str(raw_name)).resolve()
            if raw_path != topic_raw_path or outcome_dir.resolve() not in raw_path.parents:
                raise ValueError(f"raw response path mismatch: {path}")
            if not raw_path.exists():
                raise ValueError(f"raw response is missing: {raw_path}")
            if raw_record is None:
                raise ValueError(f"raw response failed validation: {raw_path}")
            raw_config = raw_record.get("frozen_request_config")
            if (
                not isinstance(raw_config, Mapping)
                or _sha256_json(raw_config) != _sha256_json(frozen_config)
            ):
                raise ValueError(f"raw/outcome frozen request mismatch: {path}")
        elif raw_record is not None:
            raise ValueError(f"outcome does not reference its raw response: {path}")
        if status in {
            "success",
            "invalid_json",
            "plan_validation_error",
            "render_validation_error",
        } and raw_record is None:
            raise ValueError(f"model outcome lacks required raw response: {path}")
        if status == "success":
            record = value.get("record")
            if not isinstance(record, dict) or value.get("failure") is not None:
                raise ValueError(f"successful outcome is malformed: {path}")
            record_topic = record.get("topic")
            nested_identity = {
                "run_id": record.get("run_id"),
                "run_kind": record.get("run_kind"),
                "request_ordinal": record.get("request_ordinal"),
                "topic_id": (
                    record_topic.get("id")
                    if isinstance(record_topic, Mapping)
                    else None
                ),
            }
            if nested_identity != expected_identity:
                raise ValueError(f"successful record identity mismatch: {path}")
            _validate_nested_provenance(
                record.get("provenance"),
                config=frozen_config,
                fingerprint_sha256=str(fingerprint_sha),
                path=path,
            )
            plan = record.get("plan")
            plan_fingerprint = plan.get("analyzer_fingerprint") if isinstance(plan, Mapping) else None
            if (
                not isinstance(plan, Mapping)
                or plan.get("topic_id") != topic.id
                or not isinstance(plan_fingerprint, Mapping)
                or _sha256_json(plan_fingerprint) != fingerprint_sha
            ):
                raise ValueError(f"successful plan identity/analyzer mismatch: {path}")
            records.append(record)
        else:
            failure = value.get("failure")
            if not isinstance(failure, dict) or value.get("record") is not None:
                raise ValueError(f"failed outcome is malformed: {path}")
            nested_identity = {
                "run_id": failure.get("run_id"),
                "run_kind": failure.get("run_kind"),
                "request_ordinal": failure.get("request_ordinal"),
                "topic_id": failure.get("topic_id"),
            }
            if nested_identity != expected_identity or failure.get("status") != status:
                raise ValueError(f"failure record identity/status mismatch: {path}")
            if status in {
                "invalid_json",
                "plan_validation_error",
                "render_validation_error",
            }:
                _validate_nested_provenance(
                    failure.get("provenance"),
                    config=frozen_config,
                    fingerprint_sha256=str(fingerprint_sha),
                    path=path,
                )
            failures.append(failure)

    actual_outcomes = {
        path.resolve() for path in outcome_dir.glob("*.json") if path.name != "_run.json"
    }
    extra_outcomes = sorted(str(path) for path in actual_outcomes - expected_outcomes)
    if extra_outcomes:
        raise ValueError("unexpected outcome artifact(s): " + ", ".join(extra_outcomes))
    raw_dir = outcome_dir / "raw_responses"
    actual_raw = {path.resolve() for path in raw_dir.glob("*.json")} if raw_dir.exists() else set()
    extra_raw = sorted(str(path) for path in actual_raw - expected_raw)
    if extra_raw:
        raise ValueError("orphan raw response artifact(s): " + ", ".join(extra_raw))

    _write_jsonl(output, records)
    _write_jsonl(failure_output, failures)
    manifest_path = output.with_name(f"{output.stem}.manifest.json")
    manifest: dict[str, object] = {
        "run_id": run_id,
        "run_kind": run_kind,
        "run_started_at": metadata.get("run_started_at"),
        "summary_written_at": _utc_now(),
        "request_order": [topic.id for topic in topics],
        "successful": len(records),
        "failed": len(failures),
        "missing": len(missing),
        "missing_topics": missing,
        "raw_only_topics": raw_only_topics,
        "status_counts": status_counts,
        "serial_execution": True,
        "frozen_model_settings": dict(frozen),
        "outcome_dir": str(outcome_dir.resolve()),
        "output": str(output.resolve()),
        "failure_output": str(failure_output.resolve()),
    }
    _write_json(manifest_path, manifest, replace=True)
    return manifest


def generate_plans(args: argparse.Namespace) -> int:
    repo_root = find_repo_root(Path.cwd())
    load_repo_env(repo_root)
    failure_output = args.failure_output or args.output.with_name(
        f"{args.output.stem}.failures.jsonl"
    )
    manifest_path = args.output.with_name(f"{args.output.stem}.manifest.json")
    summary_paths = [args.output.resolve(), failure_output.resolve(), manifest_path.resolve()]
    if len(summary_paths) != len(set(summary_paths)):
        raise ValueError("output, failure output, and manifest paths must be distinct")

    if args.rebuild_only:
        if args.outcome_dir is None:
            raise ValueError("--rebuild-only requires --outcome-dir")
        metadata_path = args.outcome_dir / "_run.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("run metadata is not an object")
        manifest = _rebuild(
            output=args.output,
            failure_output=failure_output,
            outcome_dir=args.outcome_dir,
            metadata=metadata,
        )
        return 1 if manifest["failed"] or manifest["missing"] else 0

    if not _is_loopback_url(args.base_url):
        raise ValueError("v2 model experiments require a loopback endpoint")
    if not _is_loopback_url(args.analyzer_url):
        raise ValueError("the reference analyzer must use a loopback endpoint")
    if not _is_loopback_url(args.server_tokenizer_url):
        raise ValueError("the tokenizer preflight must use a loopback endpoint")
    if not isinstance(args.model_revision, str) or not args.model_revision.strip():
        raise ValueError("v2 runs require a pinned non-empty model revision")
    if args.model != "gpt-oss-local" or args.model_revision != GPT_OSS_20B_REVISION:
        raise ValueError("this v2 milestone is pinned to the local gpt-oss-20b arm")

    if args.run_kind == "synthetic_transport_smoke":
        if args.topic_ids is not None:
            raise ValueError("synthetic smoke does not accept --topic-ids")
        topics = [SYNTHETIC_SMOKE_TOPIC]
    else:
        if args.topic_ids is None:
            raise ValueError("diagnostic run requires explicit --topic-ids")
        if tuple(args.topic_ids) != DIAGNOSTIC_TOPIC_IDS:
            raise ValueError(
                "diagnostic_first_emission is locked to topics "
                + ",".join(DIAGNOSTIC_TOPIC_IDS)
                + " in that order"
            )
        topics = _select_topics(
            load_topics(args.topics, topic_format=args.topic_format), args.topic_ids
        )
    if args.expected_analyzer_fingerprint_sha256 is None:
        raise ValueError(
            "v2 runs require --expected-analyzer-fingerprint-sha256"
        )
    run_id = _validate_run_id(
        args.run_id
        or "query_plan_v2_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    outcome_dir = args.outcome_dir or args.output.parent / "outcomes" / run_id
    if outcome_dir.exists() and any(outcome_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty outcome directory: {outcome_dir}")
    collision_paths = [args.output, failure_output, manifest_path, outcome_dir / "_run.json"]
    for ordinal, topic in enumerate(topics, start=1):
        collision_paths.extend(
            (_outcome_path(outcome_dir, ordinal, topic), _raw_path(outcome_dir, ordinal, topic))
        )
    existing = [str(path) for path in collision_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing run artifact(s): " + ", ".join(existing)
        )

    resolved_artifacts = [path.resolve() for path in collision_paths]
    if len(resolved_artifacts) != len(set(resolved_artifacts)):
        raise ValueError("run artifact paths must not alias one another")

    schema_preflight = _verify_schema_preflight(
        args.schema_preflight_manifest,
        topics=topics,
    )
    live_runtime = _verify_live_runtime(
        schema_preflight,
        base_url=args.base_url,
        tokenizer_url=args.server_tokenizer_url,
        expected_model=args.model,
        timeout=args.tokenizer_timeout,
    )

    analyzer_probe = RemoteLuceneQueryAnalyzer(
        args.analyzer_url, timeout=args.analyzer_timeout
    )
    fingerprint_object = analyzer_probe.fingerprint
    fingerprint = fingerprint_object.to_dict()
    fingerprint_sha = _sha256_json(fingerprint)
    if (
        args.expected_analyzer_fingerprint_sha256 is not None
        and fingerprint_sha != args.expected_analyzer_fingerprint_sha256
    ):
        raise ValueError(
            "query analyzer fingerprint does not match the expected SHA-256"
        )
    analyzer = RemoteLuceneQueryAnalyzer(
        args.analyzer_url,
        timeout=args.analyzer_timeout,
        expected_fingerprint=fingerprint_object,
    )
    if analyzer.fingerprint != fingerprint_object:
        raise ValueError("query analyzer fingerprint changed after pinning")
    # Preflight every topic before any model call.  The client also checks that
    # every later response retains this exact fingerprint.
    for topic in topics:
        analyzed = analyzer.analyze(topic.narrative)
        if analyzed.fingerprint.to_dict() != fingerprint:
            raise ValueError("query analyzer fingerprint changed during preflight")

    generator = QueryPlanGeneratorV2(
        query_analyzer=analyzer,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        model_revision=args.model_revision,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=args.enable_thinking,
        instruction_role=args.instruction_role,
        seed=args.seed,
    )
    context_preflight = [
        _context_preflight(
            generator=generator,
            topic=topic,
            tokenizer_url=args.server_tokenizer_url,
            timeout=args.tokenizer_timeout,
        )
        for topic in topics
    ]
    run_started_at = _utc_now()
    settings = {
        "provider": "openai_compatible_chat_completions",
        "base_url": generator.base_url,
        "model": generator.model,
        "model_revision": generator.model_revision,
        "schema_version": V2_SCHEMA_VERSION,
        "prompt_version": V2_PROMPT_VERSION,
        "renderer_version": V2_RENDERER_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "instruction_role": generator.instruction_role,
        "reasoning_effort": generator.reasoning_effort,
        "enable_thinking": generator.enable_thinking,
        "max_tokens": generator.max_tokens,
        "temperature": generator.temperature,
        "top_p": generator.top_p,
        "top_k": generator.top_k,
        "presence_penalty": generator.presence_penalty,
        "seed": generator.seed,
        "timeout_seconds": generator.timeout,
        "analyzer_url": analyzer.base_url,
        "analyzer_fingerprint": fingerprint,
        "analyzer_fingerprint_sha256": fingerprint_sha,
        "analyzer_interpretation": "pinned_upstream_reference_not_verified_hosted",
        "context_preflight": context_preflight,
        "schema_preflight": schema_preflight,
        "live_runtime": live_runtime,
        "external_cost_policy": {
            "model_endpoint_loopback": _is_loopback_url(generator.base_url),
            "retrieval_calls": 0,
            "reranker_calls": 0,
            "paid_api_calls_expected": 0,
        },
    }
    metadata: dict[str, object] = {
        "run_id": run_id,
        "run_kind": args.run_kind,
        "run_started_at": run_started_at,
        "serial_execution": True,
        "concurrency": 1,
        "topics": [asdict(topic) for topic in topics],
        "request_order": [topic.id for topic in topics],
        "frozen_model_settings": settings,
        "outcome_dir": str(outcome_dir.resolve()),
        "output": str(args.output.resolve()),
        "failure_output": str(failure_output.resolve()),
    }
    _write_json(outcome_dir / "_run.json", metadata, replace=False)

    abort_after_integrity_failure = False
    for ordinal, topic in enumerate(topics, start=1):
        if abort_after_integrity_failure:
            break
        started_at = _utc_now()
        started = time.perf_counter()
        outcome_path = _outcome_path(outcome_dir, ordinal, topic)
        raw_path = _raw_path(outcome_dir, ordinal, topic)
        frozen_config = _frozen_request_config(
            generator,
            topic,
            analyzer_fingerprint=fingerprint,
            analyzer_fingerprint_sha256=fingerprint_sha,
        )
        raw_written = False
        response_summary: dict[str, object] = {}

        def preserve_raw(response: Mapping[str, object], elapsed: float) -> None:
            nonlocal raw_written, response_summary
            if raw_written:
                raise FileExistsError(f"raw response hook invoked twice: {raw_path}")
            raw_body = getattr(response, "raw_body", None)
            if not isinstance(raw_body, bytes):
                raise ValueError("formal v2 run requires exact HTTP response bytes")
            response_summary = _response_summary(response)
            record: dict[str, object] = {
                "run_id": run_id,
                "run_kind": args.run_kind,
                "request_ordinal": ordinal,
                "topic_id": topic.id,
                "request_started_at": started_at,
                "response_received_at": _utc_now(),
                "request_elapsed_seconds": elapsed,
                "capture_level": "exact_http_body_before_decode_or_json_parse",
                "http_status": getattr(response, "http_status", None),
                "response_headers": dict(
                    getattr(response, "response_headers", {}) or {}
                ),
                "response_headers_capture_level": "normalized_mapping_duplicates_collapsed",
                "raw_body_bytes": len(raw_body),
                "raw_body_base64": base64.b64encode(raw_body).decode("ascii"),
                "raw_body_sha256": hashlib.sha256(raw_body).hexdigest(),
                "raw_body_utf8_preview": raw_body.decode("utf-8", errors="replace"),
                "frozen_request_config": frozen_config,
            }
            _write_json(raw_path, record, replace=False)
            raw_written = True

        try:
            result = generator.generate(topic, response_hook=preserve_raw)
            _assert_result_integrity(
                topic=topic,
                result=result,
                analyzer_fingerprint=fingerprint,
                expected_response_model=generator.model,
            )
            finished_at = _utc_now()
            elapsed = time.perf_counter() - started
            provenance = asdict(result.provenance)
            if result.outcome.used_fallback:
                failure = {
                    "run_id": run_id,
                    "run_kind": args.run_kind,
                    "request_ordinal": ordinal,
                    "topic_id": topic.id,
                    "topic": asdict(topic),
                    "status": result.outcome.failure.status,
                    "failure_kind": "plan_validation",
                    "request_started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_seconds": elapsed,
                    "failure": _serialize(result.outcome.failure),
                    "fallback_rendered_queries": [
                        asdict(row) for row in result.outcome.rendered_queries
                    ],
                    "expansion_audit": _serialize(result.outcome.expansion_audit),
                    "provenance": provenance,
                    "token_tape": result.token_tape.to_dict(),
                    "raw_response_path": str(raw_path.resolve()) if raw_written else None,
                }
                status = result.outcome.failure.status
                outcome = {
                    "run_id": run_id,
                    "run_kind": args.run_kind,
                    "request_ordinal": ordinal,
                    "topic_id": topic.id,
                    "status": status,
                    "request_started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_seconds": elapsed,
                    "serial_execution": True,
                    "frozen_request_config": frozen_config,
                    "raw_response_path": str(raw_path.resolve()) if raw_written else None,
                    "response": response_summary,
                    "record": None,
                    "failure": failure,
                }
                message = f"{topic.id}: FAILED [{status}]"
            else:
                record = {
                    "run_id": run_id,
                    "run_kind": args.run_kind,
                    "request_ordinal": ordinal,
                    "topic": asdict(topic),
                    "plan": result.outcome.plan.to_dict(),
                    "rendered_queries": [
                        asdict(row) for row in result.outcome.rendered_queries
                    ],
                    "expansion_audit": _serialize(result.outcome.expansion_audit),
                    "provenance": provenance,
                    "token_tape": result.token_tape.to_dict(),
                    "outcome_path": str(outcome_path.resolve()),
                    "raw_response_path": str(raw_path.resolve()),
                }
                outcome = {
                    "run_id": run_id,
                    "run_kind": args.run_kind,
                    "request_ordinal": ordinal,
                    "topic_id": topic.id,
                    "status": "success",
                    "request_started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_seconds": elapsed,
                    "serial_execution": True,
                    "frozen_request_config": frozen_config,
                    "raw_response_path": str(raw_path.resolve()),
                    "response": response_summary,
                    "record": record,
                    "failure": None,
                }
                message = (
                    f"{topic.id}: {len(result.outcome.plan.facets)} facets, "
                    f"{elapsed:.2f}s"
                )
            _write_json(outcome_path, outcome, replace=False)
            print(message)
        except Exception as exc:
            if outcome_path.exists():
                raise
            finished_at = _utc_now()
            elapsed = time.perf_counter() - started
            status = _exception_status(exc)
            if (
                raw_written
                and not isinstance(exc, QueryPlanHttpTransportError)
                and status in {"http_error", "timeout", "unexpected_exception"}
            ):
                # Once model bytes are safely captured, later network/timeout
                # errors can only come from analyzer/validation work.  Stop the
                # batch so a broken reference service cannot spend more calls.
                status = "analyzer_integrity_error"
            if status in {
                "analyzer_integrity_error",
                "http_error",
                "timeout",
                "unexpected_exception",
            }:
                abort_after_integrity_failure = True
            failure = {
                "run_id": run_id,
                "run_kind": args.run_kind,
                "request_ordinal": ordinal,
                "topic_id": topic.id,
                "topic": asdict(topic),
                "status": status,
                "failure_kind": (
                    "analyzer_integrity"
                    if status == "analyzer_integrity_error"
                    else "transport"
                    if status in {"http_error", "timeout"}
                    else "unexpected"
                ),
                "request_started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": elapsed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "fallback_rendered_queries": [_fallback_query(topic)],
                "raw_response_path": str(raw_path.resolve()) if raw_written else None,
            }
            outcome = {
                "run_id": run_id,
                "run_kind": args.run_kind,
                "request_ordinal": ordinal,
                "topic_id": topic.id,
                "status": status,
                "request_started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": elapsed,
                "serial_execution": True,
                "frozen_request_config": frozen_config,
                "raw_response_path": str(raw_path.resolve()) if raw_written else None,
                "response": response_summary,
                "record": None,
                "failure": failure,
            }
            _write_json(outcome_path, outcome, replace=False)
            print(f"{topic.id}: FAILED [{status}]: {exc}")

    manifest = _rebuild(
        output=args.output,
        failure_output=failure_output,
        outcome_dir=outcome_dir,
        metadata=metadata,
    )
    print(f"Wrote {manifest['successful']} valid plan(s) to {args.output}")
    if manifest["failed"] or manifest["missing"]:
        print(
            f"Run has {manifest['failed']} failure(s) and "
            f"{manifest['missing']} missing outcome(s)"
        )
    return 1 if manifest["failed"] or manifest["missing"] else 0


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = find_repo_root(Path.cwd())
    parser = argparse.ArgumentParser(
        description="Run query-planner v2 serially with immutable raw-first ledgers."
    )
    parser.add_argument("--run-kind", choices=RUN_KINDS, default=RUN_KINDS[0])
    parser.add_argument(
        "--topics",
        type=Path,
        default=(
            repo_root
            / "trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"
        ),
    )
    parser.add_argument("--topic-format", choices=("tsv", "jsonl"), default="tsv")
    parser.add_argument("--topic-ids", type=_parse_topic_ids)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "outputs/query_planner_v2/plans.jsonl",
    )
    parser.add_argument("--failure-output", type=Path)
    parser.add_argument("--outcome-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--rebuild-only", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gpt-oss-local")
    parser.add_argument("--model-revision", default=GPT_OSS_20B_REVISION)
    parser.add_argument("--api-key")
    parser.add_argument("--instruction-role", choices=("developer", "system"), default="developer")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-tokens", type=int, default=5400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--analyzer-url", default="http://127.0.0.1:18081")
    parser.add_argument("--analyzer-timeout", type=float, default=10.0)
    parser.add_argument("--expected-analyzer-fingerprint-sha256")
    parser.add_argument(
        "--schema-preflight-manifest",
        type=Path,
        default=(
            repo_root
            / "reports/experiments/query_planner_v2_schema_compat_v2_1/compiler_manifest_xgrammar.json"
        ),
    )
    parser.add_argument(
        "--server-tokenizer-url", default="http://127.0.0.1:8000/tokenize"
    )
    parser.add_argument("--tokenizer-timeout", type=float, default=60.0)
    return parser


def main() -> int:
    return generate_plans(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
