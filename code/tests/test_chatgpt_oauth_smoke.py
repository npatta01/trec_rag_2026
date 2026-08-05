"""Tests for the LiteLLM ChatGPT OAuth smoke runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_rag.chatgpt_oauth_smoke import (
    AuthSettings,
    ModelSettings,
    ProbeSettings,
    RuntimeBindings,
    SmokeConfig,
    collect_response_text,
    execute_smoke,
    load_smoke_config,
    normalize_oauth_record,
)


def _config(tmp_path: Path) -> SmokeConfig:
    root = tmp_path / "repo"
    root.mkdir()
    token_store = tmp_path / "credentials" / "auth.json"
    token_store.parent.mkdir()
    token_store.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": "must-not-be-copied",
                "tokens": {
                    "access_token": "test-access-token",
                    "refresh_token": "test-refresh-token",
                    "account_id": "test-account",
                },
            }
        ),
        encoding="utf-8",
    )
    return SmokeConfig(
        root_dir=root,
        experiment_id="oauth_smoke_test",
        auth=AuthSettings(token_store=token_store),
        model=ModelSettings(
            name="chatgpt/gpt-test",
            originator="test-suite",
            timeout_seconds=10,
        ),
        probe=ProbeSettings(prompt="Say passed.", expected_response="passed"),
        report_path=root / "outputs" / "oauth-smoke" / "result.json",
    )


def test_normalize_oauth_record_flattens_codex_tokens_only(tmp_path: Path) -> None:
    config = _config(tmp_path)

    record, schema = normalize_oauth_record(config.auth.token_store)

    assert schema == "codex_nested"
    assert record == {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "account_id": "test-account",
    }
    assert "OPENAI_API_KEY" not in record


def test_execute_smoke_uses_structured_input_and_collects_stream(tmp_path: Path) -> None:
    config = _config(tmp_path)
    captured: dict[str, object] = {}

    def responses(**kwargs: object):
        captured.update(kwargs)
        return iter(
            [
                SimpleNamespace(type="response.created"),
                SimpleNamespace(type="response.output_text.delta", delta="pass"),
                SimpleNamespace(type="response.output_text.delta", delta="ed"),
            ]
        )

    environment: dict[str, str] = {}
    report = execute_smoke(
        config,
        runtime=RuntimeBindings(responses=responses, litellm_version="test"),
        environ=environment,
    )

    assert report["status"] == "passed"
    assert captured["model"] == "chatgpt/gpt-test"
    assert captured["timeout"] == 10
    assert captured["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Say passed."}],
        }
    ]
    assert "api_key" not in captured
    assert report["auth"]["source_schema"] == "codex_nested"
    assert report["auth"]["source_file_modified"] is False
    assert report["checks"]["temporary_auth_copy_removed"] is True
    assert environment == {}
    assert config.report_path.exists()


def test_execute_smoke_refuses_api_key_environment(tmp_path: Path) -> None:
    config = _config(tmp_path)
    called = False

    def responses(**kwargs: object):
        nonlocal called
        called = True
        return {"output_text": "passed"}

    report = execute_smoke(
        config,
        runtime=RuntimeBindings(responses=responses, litellm_version="test"),
        environ={"OPENROUTER_API_KEY": "secret-value"},
    )

    assert report["status"] == "error"
    assert report["error"]["type"] == "RuntimeError"
    assert "OPENROUTER_API_KEY" in report["error"]["message"]
    assert "secret-value" not in json.dumps(report)
    assert called is False


def test_collect_response_text_accepts_aggregated_response() -> None:
    assert collect_response_text({"output_text": "complete"}) == "complete"


def test_collect_response_text_rejects_empty_stream() -> None:
    with pytest.raises(ValueError, match="no output text deltas"):
        collect_response_text(iter([SimpleNamespace(type="response.completed")]))


def test_example_config_loads() -> None:
    root = Path(__file__).resolve().parents[2]

    config = load_smoke_config(root / "configs" / "chatgpt_oauth_litellm_smoke_v1.yaml")

    assert config.model.name == "chatgpt/gpt-5.4"
    assert config.auth.token_store == Path.home() / ".codex" / "auth.json"
    assert config.report_path == root / "outputs" / config.experiment_id / "result.json"


def test_config_rejects_token_store_inside_repo(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("instructions", encoding="utf-8")
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        """
schema_version: chatgpt_oauth_smoke_v1
experiment: {id: oauth_test}
auth: {token_store: local-auth.json}
model:
  name: chatgpt/gpt-test
  originator: test-suite
  timeout_seconds: 10
probe:
  prompt: Say passed.
  expected_response: passed
output: {report_path: outputs/oauth-test/result.json}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="auth.token_store must remain outside"):
        load_smoke_config(config_path)
