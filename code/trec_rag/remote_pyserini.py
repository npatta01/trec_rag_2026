"""Compatibility exports for hosted Pyserini helpers."""

from __future__ import annotations

from .env_config import env_get, env_int, env_queries
from .remote_client import RemotePyseriniClient, extract_text, normalize_candidates
from .remote_config import (
    DEFAULT_QUERIES,
    DEFAULT_QUERY_SOURCE,
    RemotePyseriniConfig,
    remote_index_url,
)
from .repo_env import find_repo_root, load_dotenv, load_repo_env, shared_checkout_root

_env_get = env_get

__all__ = [
    "DEFAULT_QUERIES",
    "DEFAULT_QUERY_SOURCE",
    "RemotePyseriniClient",
    "RemotePyseriniConfig",
    "_env_get",
    "env_get",
    "env_int",
    "env_queries",
    "extract_text",
    "find_repo_root",
    "load_dotenv",
    "load_repo_env",
    "normalize_candidates",
    "remote_index_url",
    "shared_checkout_root",
]
