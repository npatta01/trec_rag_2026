import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from trec_rag.remote_pyserini import (
    RemotePyseriniClient,
    RemotePyseriniConfig,
    extract_text,
    load_repo_env,
    normalize_candidates,
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


def test_config_builds_index_url_from_env_index_and_base_url():
    config = RemotePyseriniConfig.from_env(
        {
            "PYSERINI_INDEX": "custom index",
            "PYSERINI_BASE_URL": "http://api.example.test/",
        }
    )

    assert config.index_url == "http://api.example.test/v1/custom%20index/search"


def test_config_builds_index_url_from_default_env_names():
    config = RemotePyseriniConfig.from_env(
        {
            "DEFAULT_INDEX": "climbmix-400b",
            "DEFAULT_BASE_URL": "http://api.example.test",
        }
    )

    assert config.index_url == "http://api.example.test/v1/climbmix-400b/search"


def test_config_requires_remote_endpoint_env():
    with pytest.raises(ValueError, match="Set INDEX_URL.*PYSERINI_INDEX.*DEFAULT_INDEX"):
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


def test_client_builds_authenticated_search_request():
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"candidates": []}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        return FakeResponse()

    config = RemotePyseriniConfig(
        index_url="http://api.example.test/v1/climbmix-400b/search",
        api_token="secret-token",
        hits=3,
        queries=("wildfire smoke",),
    )

    response = RemotePyseriniClient(config, opener=fake_urlopen).search("wildfire smoke")

    parsed_url = urlparse(captured["url"])
    assert response == {"candidates": []}
    assert parsed_url.scheme == "http"
    assert parsed_url.netloc == "api.example.test"
    assert parsed_url.path == "/v1/climbmix-400b/search"
    assert parse_qs(parsed_url.query) == {"query": ["wildfire smoke"], "hits": ["3"]}
    assert captured["headers"]["Accept"] == "application/json"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["timeout"] == 30


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
