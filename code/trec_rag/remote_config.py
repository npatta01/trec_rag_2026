"""Configuration for the hosted Pyserini search API."""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import Mapping

from .env_config import env_get, env_int, env_queries


DEFAULT_QUERIES = (
    "I'm seeking to understand the environmental and health impacts of e-waste and other "
    "waste, along with how proper recycling benefits sustainability and local economies. "
    "Could you explain the risks of improper waste management, recent recycling "
    "innovations, and practical steps individuals and businesses can take for responsible "
    "waste handling?"
)
DEFAULT_QUERY_SOURCE = "rag25-topics-dev.tsv topic 31"


def remote_index_url(env: Mapping[str, str] | None = None) -> str:
    index_url = (env_get(env, "INDEX_URL") or "").strip().rstrip("?")
    if index_url:
        return index_url

    index = (env_get(env, "PYSERINI_INDEX") or env_get(env, "DEFAULT_INDEX") or "").strip()
    base_url = (
        env_get(env, "PYSERINI_BASE_URL") or env_get(env, "DEFAULT_BASE_URL") or ""
    ).strip().rstrip("/")
    if not index or not base_url:
        raise ValueError(
            "Set INDEX_URL, or set index/base env vars "
            "(PYSERINI_INDEX/PYSERINI_BASE_URL or DEFAULT_INDEX/DEFAULT_BASE_URL)."
        )
    return f"{base_url}/v1/{urllib.parse.quote(index)}/search"


@dataclass(frozen=True)
class RemotePyseriniConfig:
    index_url: str
    api_token: str | None
    hits: int
    queries: tuple[str, ...]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RemotePyseriniConfig":
        return cls(
            index_url=remote_index_url(env),
            api_token=env_get(env, "PYSERINI_API_TOKEN"),
            hits=env_int("EXTERNAL_PYSERINI_HITS", 5, env),
            queries=env_queries("SAMPLE_QUERIES", DEFAULT_QUERIES, env),
        )
