"""Run a small LiteLLM request using ChatGPT OAuth and no API keys."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from contextlib import chdir, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from trec_rag.repo_env import find_repo_root


SCHEMA_VERSION = "chatgpt_oauth_smoke_v1"
REPORT_SCHEMA_VERSION = "chatgpt_oauth_smoke_report_v1"
FORBIDDEN_API_KEY_ENV = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "LITELLM_API_KEY",
)
_AUTH_FIELDS = (
    "access_token",
    "refresh_token",
    "id_token",
    "account_id",
    "expires_at",
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError("YAML mapping keys must be hashable") from exc
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class AuthSettings:
    token_store: Path


@dataclass(frozen=True)
class ModelSettings:
    name: str
    originator: str
    timeout_seconds: float


@dataclass(frozen=True)
class ProbeSettings:
    prompt: str
    expected_response: str


@dataclass(frozen=True)
class SmokeConfig:
    root_dir: Path
    experiment_id: str
    auth: AuthSettings
    model: ModelSettings
    probe: ProbeSettings
    report_path: Path


@dataclass(frozen=True)
class RuntimeBindings:
    responses: Callable[..., object]
    litellm_version: str


def load_smoke_config(path: Path) -> SmokeConfig:
    """Load and validate the OAuth smoke-test configuration."""
    config_path = Path(path).resolve()
    root_dir = find_repo_root(config_path.parent)
    try:
        raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML config: {config_path}") from exc

    document = _mapping(raw, "config")
    _reject_unknown(
        document,
        {"schema_version", "experiment", "auth", "model", "probe", "output"},
        "config",
    )
    if _text(document, "schema_version", "config") != SCHEMA_VERSION:
        raise ValueError(f"config.schema_version must be {SCHEMA_VERSION}")

    experiment = _mapping(document.get("experiment"), "experiment")
    _reject_unknown(experiment, {"id"}, "experiment")
    experiment_id = _text(experiment, "id", "experiment")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", experiment_id):
        raise ValueError("experiment.id must be a safe identifier")

    auth_raw = _mapping(document.get("auth"), "auth")
    _reject_unknown(auth_raw, {"token_store"}, "auth")
    token_store = _expand_path(root_dir, _text(auth_raw, "token_store", "auth"))
    _require_outside_repo(root_dir, token_store, "auth.token_store")

    model_raw = _mapping(document.get("model"), "model")
    _reject_unknown(model_raw, {"name", "originator", "timeout_seconds"}, "model")
    model_name = _text(model_raw, "name", "model")
    if not model_name.startswith("chatgpt/"):
        raise ValueError("model.name must use the chatgpt/ LiteLLM provider route")
    model = ModelSettings(
        name=model_name,
        originator=_text(model_raw, "originator", "model"),
        timeout_seconds=_positive_number(model_raw, "timeout_seconds", "model"),
    )

    probe_raw = _mapping(document.get("probe"), "probe")
    _reject_unknown(probe_raw, {"prompt", "expected_response"}, "probe")
    probe = ProbeSettings(
        prompt=_text(probe_raw, "prompt", "probe"),
        expected_response=_text(probe_raw, "expected_response", "probe"),
    )

    output_raw = _mapping(document.get("output"), "output")
    _reject_unknown(output_raw, {"report_path"}, "output")
    report_path = _expand_path(root_dir, _text(output_raw, "report_path", "output"))
    _require_beneath(root_dir / "outputs", report_path, "output.report_path")

    return SmokeConfig(
        root_dir=root_dir,
        experiment_id=experiment_id,
        auth=AuthSettings(token_store=token_store),
        model=model,
        probe=probe,
        report_path=report_path,
    )


def execute_smoke(
    config: SmokeConfig,
    *,
    runtime: RuntimeBindings | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, object]:
    """Execute the OAuth-only request and persist a sanitized report."""
    environment = os.environ if environ is None else environ
    started_at = _utc_now()
    source_schema = "unknown"
    bindings = runtime

    try:
        _assert_oauth_only_environment(environment)
        auth_record, source_schema = normalize_oauth_record(config.auth.token_store)

        with TemporaryDirectory(prefix="trec-rag-chatgpt-oauth-") as temporary:
            temporary_dir = Path(temporary)
            temporary_auth = temporary_dir / "auth.json"
            temporary_auth.write_text(
                json.dumps(auth_record, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                temporary_auth.chmod(0o600)
            except OSError:
                pass

            updates = {
                "CHATGPT_TOKEN_DIR": str(temporary_dir),
                "CHATGPT_AUTH_FILE": temporary_auth.name,
                "CHATGPT_ORIGINATOR": config.model.originator,
            }
            # LiteLLM imports python-dotenv at module import time. Importing from
            # an empty temporary cwd prevents discovery of the repository .env.
            with _temporary_environment(environment, updates), chdir(temporary_dir):
                if bindings is None:
                    bindings = _load_runtime()
                _assert_oauth_only_environment(environment)
                response = bindings.responses(
                    model=config.model.name,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": config.probe.prompt}
                            ],
                        }
                    ],
                    timeout=config.model.timeout_seconds,
                )
                answer = collect_response_text(response)

        checks = {
            "exact_response": answer.strip() == config.probe.expected_response,
            "api_key_environment_absent": True,
            "temporary_auth_copy_removed": not temporary_auth.exists(),
        }
        status = "passed" if all(checks.values()) else "failed"
        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "experiment_id": config.experiment_id,
            "status": status,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "model": config.model.name,
            "auth": {
                "method": "chatgpt_oauth",
                "source_schema": source_schema,
                "token_store": _display_path(config.auth.token_store),
                "source_file_modified": False,
            },
            "probe": {
                "prompt": config.probe.prompt,
                "expected_response": config.probe.expected_response,
                "actual_response": _redact(answer)[:2000],
            },
            "checks": checks,
            "package_versions": {"litellm": bindings.litellm_version},
        }
    except Exception as exc:
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "experiment_id": config.experiment_id,
            "status": "error",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "model": config.model.name,
            "auth": {
                "method": "chatgpt_oauth",
                "source_schema": source_schema,
                "token_store": _display_path(config.auth.token_store),
                "source_file_modified": False,
            },
            "checks": {
                "exact_response": False,
                "api_key_environment_absent": False,
                "temporary_auth_copy_removed": True,
            },
            "error": {"type": type(exc).__name__, "message": _redact(str(exc))[:2000]},
            "package_versions": (
                {} if bindings is None else {"litellm": bindings.litellm_version}
            ),
        }

    _write_report(config.report_path, report)
    return report


def normalize_oauth_record(path: Path) -> tuple[dict[str, object], str]:
    """Return only OAuth fields in the flat schema LiteLLM expects."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"ChatGPT OAuth token store not found at {_display_path(path)}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError("ChatGPT OAuth token store contains invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("ChatGPT OAuth token store must contain a JSON object")

    nested = raw.get("tokens")
    if isinstance(nested, Mapping):
        source: Mapping[str, object] = nested
        source_schema = "codex_nested"
    else:
        source = raw
        source_schema = "litellm_flat"

    normalized = {
        field: source[field]
        for field in _AUTH_FIELDS
        if field in source and source[field] is not None
    }
    if not isinstance(normalized.get("access_token"), str) and not isinstance(
        normalized.get("refresh_token"), str
    ):
        raise ValueError("ChatGPT OAuth token store has no access or refresh token")
    return normalized, source_schema


def collect_response_text(response: object) -> str:
    """Collect text from either an aggregated response or forced OAuth stream."""
    output_text = _field(response, "output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    if isinstance(response, (str, bytes, Mapping)) or not isinstance(response, Iterable):
        raise ValueError("LiteLLM response contains no output text")

    parts: list[str] = []
    for event in response:
        if _field(event, "type") != "response.output_text.delta":
            continue
        delta = _field(event, "delta")
        if isinstance(delta, str):
            parts.append(delta)
    if not parts:
        raise ValueError("LiteLLM response stream contains no output text deltas")
    return "".join(parts)


def _load_runtime() -> RuntimeBindings:
    try:
        litellm = importlib.import_module("litellm")
        version = importlib.metadata.version("litellm")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            "LiteLLM is missing; install the chatgpt-oauth-poc dependency group"
        ) from exc
    return RuntimeBindings(responses=litellm.responses, litellm_version=version)


def _assert_oauth_only_environment(environment: Mapping[str, str]) -> None:
    present = [name for name in FORBIDDEN_API_KEY_ENV if environment.get(name)]
    if present:
        raise RuntimeError(
            "OAuth-only smoke test refuses API-key environment variable(s): "
            + ", ".join(present)
        )


@contextmanager
def _temporary_environment(
    environment: MutableMapping[str, str], updates: Mapping[str, str]
) -> Iterable[None]:
    previous = {key: environment.get(key) for key in updates}
    environment.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _mapping(value: object, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    return value


def _reject_unknown(mapping: Mapping[str, object], allowed: set[str], owner: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{owner} has unknown field(s): {', '.join(sorted(unknown))}")


def _text(mapping: Mapping[str, object], key: str, owner: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}.{key} must be non-empty text")
    return value.strip()


def _positive_number(mapping: Mapping[str, object], key: str, owner: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{owner}.{key} must be a positive number")
    return float(value)


def _expand_path(root_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root_dir / path).resolve()


def _require_outside_repo(root_dir: Path, path: Path, owner: str) -> None:
    try:
        path.relative_to(root_dir.resolve())
    except ValueError:
        return
    raise ValueError(f"{owner} must remain outside the repository")


def _require_beneath(parent: Path, path: Path, owner: str) -> None:
    try:
        path.relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError(f"{owner} must remain beneath {parent.resolve()}") from exc


def _display_path(path: Path) -> str:
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)


def _redact(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to the smoke-test YAML config")
    args = parser.parse_args(argv)
    config = load_smoke_config(args.config)
    report = execute_smoke(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
