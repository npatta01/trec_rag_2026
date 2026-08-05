import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from trec_rag.remote_pyserini import (
    RemotePyseriniClient,
    RemotePyseriniConfig,
    RemotePyseriniThrottled,
    extract_text,
    load_dotenv,
    load_repo_env,
    normalize_candidates,
    rate_limited_session,
)


def test_config_uses_index_url_and_typed_values():
    env = {
        "INDEX_URL": "http://api.example.test/v1/climbmix-400b/search",
        "PYSERINI_API_TOKEN": "secret-token",
        "EXTERNAL_PYSERINI_HITS": "7",
        "SAMPLE_QUERIES": "alpha; beta ; ; gamma",
    }

    config = RemotePyseriniConfig.from_env(env)

    assert config.index_url == "http://api.example.test/v1/climbmix-400b/search"
    assert config.api_token == "secret-token"
    assert config.hits == 7
    assert config.queries == ("alpha", "beta", "gamma")


def test_config_requires_remote_endpoint_env():
    with pytest.raises(ValueError, match="Set INDEX_URL"):
        RemotePyseriniConfig.from_env({})


def test_config_defaults_to_rag25_dev_topic_query():
    config = RemotePyseriniConfig.from_env(
        {"INDEX_URL": "http://api.example.test/v1/climbmix-400b/search"}
    )

    assert len(config.queries) == 1
    assert "environmental and health impacts of e-waste" in config.queries[0]
    assert "responsible waste handling" in config.queries[0]


def test_load_repo_env_reads_shared_checkout_env(tmp_path, monkeypatch):
    shared_checkout = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    git_dir = shared_checkout / ".git" / "worktrees" / "wt"
    shared_checkout.mkdir()
    worktree.mkdir()
    git_dir.mkdir(parents=True)
    (shared_checkout / ".env").write_text(
        "PYSERINI_API_TOKEN=from-shared\nINDEX_URL=http://shared/v1/climbmix-400b/search\n",
        encoding="utf-8",
    )
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    monkeypatch.delenv("PYSERINI_API_TOKEN", raising=False)
    monkeypatch.delenv("INDEX_URL", raising=False)

    load_repo_env(worktree)

    assert os.environ["PYSERINI_API_TOKEN"] == "from-shared"
    assert os.environ["INDEX_URL"] == "http://shared/v1/climbmix-400b/search"


def test_load_repo_env_lets_env_local_override_dotenv(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text(
        "PYSERINI_API_TOKEN=from-env\nINDEX_URL=http://env/v1/climbmix-400b/search\n",
        encoding="utf-8",
    )
    (repo / ".env.local").write_text(
        "PYSERINI_API_TOKEN=from-local\nINDEX_URL=http://local/v1/climbmix-400b/search\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PYSERINI_API_TOKEN", raising=False)
    monkeypatch.delenv("INDEX_URL", raising=False)

    load_repo_env(repo)

    assert os.environ["PYSERINI_API_TOKEN"] == "from-local"
    assert os.environ["INDEX_URL"] == "http://local/v1/climbmix-400b/search"


def test_load_repo_env_keeps_existing_environment_values(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env.local").write_text(
        "PYSERINI_API_TOKEN=from-file\nINDEX_URL=http://local/v1/climbmix-400b/search\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYSERINI_API_TOKEN", "from-process")
    monkeypatch.delenv("INDEX_URL", raising=False)

    load_repo_env(repo)

    assert os.environ["PYSERINI_API_TOKEN"] == "from-process"
    assert os.environ["INDEX_URL"] == "http://local/v1/climbmix-400b/search"


def test_env_example_contains_public_remote_defaults(monkeypatch):
    env_example = Path(__file__).resolve().parents[2] / ".env.example"
    for key in (
        "EXTERNAL_PYSERINI_HITS",
        "INDEX_URL",
        "PYSERINI_API_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    load_dotenv(env_example)
    config = RemotePyseriniConfig.from_env()

    assert config.index_url == "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
    assert config.api_token == ""
    assert config.hits == 5


def test_client_builds_authenticated_search_request():
    captured = {}

    class FakeResponse:
        content = json.dumps({"candidates": []}).encode("utf-8")
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

    class FakeSession:
        def get(self, url, **kwargs):
            captured.update(url=url, **kwargs)
            return FakeResponse()

    config = RemotePyseriniConfig(
        index_url="http://api.example.test/v1/climbmix-400b/search",
        api_token="secret-token",
        hits=3,
        queries=("wildfire smoke",),
    )

    response = RemotePyseriniClient(config, session=FakeSession()).search("wildfire smoke")

    parsed_url = urlparse(captured["url"])
    assert response == {"candidates": []}
    assert parsed_url.scheme == "http"
    assert parsed_url.netloc == "api.example.test"
    assert parsed_url.path == "/v1/climbmix-400b/search"
    assert captured["params"] == {"query": "wildfire smoke", "hits": "3"}
    assert captured["headers"]["Accept"] == "application/json"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["timeout"] == 300
    assert captured["allow_redirects"] is False


@pytest.mark.parametrize("timeout", [True, False, 0, -1, 1.5, "30"])
def test_client_rejects_non_positive_or_non_integer_timeout(timeout):
    config = RemotePyseriniConfig("https://api.test/search", None, 3, ())
    with pytest.raises(ValueError, match="timeout must be a positive integer"):
        RemotePyseriniClient(config, session=object(), timeout=timeout)


def test_client_accepts_explicit_positive_timeout_override():
    config = RemotePyseriniConfig("https://api.test/search", None, 3, ())
    client = RemotePyseriniClient(config, session=object(), timeout=17)
    assert client.timeout == 17


def test_client_persists_raw_bytes_before_decoding():
    events = []

    class Response:
        content = b"not-json"
        status_code = 200
        headers = {}
        def raise_for_status(self): pass

    class Session:
        def get(self, *_args, **_kwargs): return Response()

    config = RemotePyseriniConfig("https://api.test/search", None, 3, ())
    with pytest.raises(json.JSONDecodeError):
        RemotePyseriniClient(config, session=Session()).search_raw(
            "query", raw_sink=lambda raw: events.append(raw)
        )
    assert events == [b"not-json"]


def test_client_429_exposes_retry_after_without_hidden_retry():
    calls = []

    class Response:
        content = b'{"error":"slow down"}'
        status_code = 429
        headers = {"Retry-After": "1"}

    class Session:
        def get(self, *_args, **kwargs):
            calls.append(kwargs)
            return Response()

    config = RemotePyseriniConfig("https://api.test/search", "do-not-log", 3, ())
    with pytest.raises(RemotePyseriniThrottled, match="no retry") as exc:
        RemotePyseriniClient(config, session=Session()).search("query")
    assert exc.value.retry_after_seconds == 1
    assert len(calls) == 1
    assert "do-not-log" not in str(exc.value)


def test_rate_limiter_delays_before_second_transport_entry(tmp_path):
    entered = []

    class RecordingAdapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            entered.append(time.monotonic())
            response = requests.Response()
            response.status_code = 200
            response._content = b"{}"
            response.request = request
            return response
        def close(self): pass

    config = RemotePyseriniConfig(
        "https://api.test/search", None, 3, (),
        min_interval_seconds=1,
        limiter_state_path=tmp_path / "rate.sqlite",
    )
    session = rate_limited_session(config)
    session.mount("https://", RecordingAdapter())
    session.get(config.index_url)
    session.get(config.index_url)
    assert entered[1] - entered[0] >= 0.9


def test_extract_text_and_normalize_candidates():
    response = {
        "candidates": [
            {
                "docid": "shard_04149_6425",
                "score": 16.0,
                "rank": 1,
                "doc": {
                    "id": "shard_04149_6425",
                    "url": "https://example.test/wildfire-smoke",
                    "text": "Health wildfire smoke response",
                },
            },
            {
                "id": "fallback-id",
                "score": 9.0,
                "doc": json.dumps({"contents": "Serialized contents", "language": "en"}),
            },
        ]
    }

    rows = normalize_candidates(response)

    assert extract_text({"body": "Body text"}) == "Body text"
    assert rows == [
        {
            "rank": 1,
            "docid": "shard_04149_6425",
            "score": 16.0,
            "text": "Health wildfire smoke response",
            "text_length": 30,
        },
        {
            "rank": 2,
            "docid": "fallback-id",
            "score": 9.0,
            "text": "Serialized contents",
            "text_length": 19,
        },
    ]


def test_normalize_candidates_handles_hosted_api_string_docs():
    rows = normalize_candidates(
        {
            "candidates": [
                {
                    "doc": "A hosted API document body.",
                    "docid": "shard_04149_6425",
                    "rank": 1,
                    "score": 12.5,
                }
            ]
        }
    )

    assert rows == [
        {
            "rank": 1,
            "docid": "shard_04149_6425",
            "score": 12.5,
            "text": "A hosted API document body.",
            "text_length": 27,
        }
    ]


def test_config_rejects_bad_hits():
    with pytest.raises(ValueError, match="EXTERNAL_PYSERINI_HITS must be an integer"):
        RemotePyseriniConfig.from_env(
            {
                "INDEX_URL": "http://api.example.test/v1/climbmix-400b/search",
                "EXTERNAL_PYSERINI_HITS": "many",
            }
        )


def test_rate_policy_defaults_are_gentle_and_overridable():
    """The shared endpoint throttled us at three seconds; six leaves headroom."""
    from trec_rag.remote_config import DEFAULT_MIN_INTERVAL_SECONDS, RemotePyseriniConfig

    default = RemotePyseriniConfig.from_env({"INDEX_URL": "https://index.test/search"})
    assert default.min_interval_seconds == DEFAULT_MIN_INTERVAL_SECONDS
    assert DEFAULT_MIN_INTERVAL_SECONDS >= 6.0
    assert default.burst == 1

    tuned = RemotePyseriniConfig.from_env(
        {"INDEX_URL": "https://index.test/search", "PYSERINI_MIN_INTERVAL_SECONDS": "9"}
    )
    assert tuned.min_interval_seconds == 9.0
