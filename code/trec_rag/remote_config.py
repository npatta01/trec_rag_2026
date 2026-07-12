"""Configuration for the hosted Pyserini search API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    raise ValueError("Set INDEX_URL to the hosted Pyserini search endpoint.")


@dataclass(frozen=True)
class RemotePyseriniConfig:
    index_url: str
    api_token: str | None
    hits: int
    queries: tuple[str, ...]
    min_interval_seconds: float = 3.0
    burst: int = 1
    limiter_state_path: Path = Path("cache/retrieval/pyserini_remote/rate-limit.sqlite")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RemotePyseriniConfig":
        interval_text = env_get(env, "PYSERINI_MIN_INTERVAL_SECONDS") or "3"
        burst_text = env_get(env, "PYSERINI_RATE_BURST") or "1"
        try:
            interval = float(interval_text)
            burst = int(burst_text)
        except ValueError as exc:
            raise ValueError("Pyserini rate policy values must be numeric") from exc
        if interval <= 0 or burst != 1:
            raise ValueError("Pyserini rate policy requires a positive interval and burst=1")
        return cls(
            index_url=remote_index_url(env),
            api_token=env_get(env, "PYSERINI_API_TOKEN"),
            hits=env_int("EXTERNAL_PYSERINI_HITS", 5, env),
            queries=env_queries("SAMPLE_QUERIES", DEFAULT_QUERIES, env),
            min_interval_seconds=interval,
            burst=burst,
            limiter_state_path=Path(
                env_get(env, "PYSERINI_LIMITER_STATE_PATH")
                or "cache/retrieval/pyserini_remote/rate-limit.sqlite"
            ),
        )
