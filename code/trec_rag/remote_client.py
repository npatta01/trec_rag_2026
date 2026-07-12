"""Hosted Pyserini search client and candidate normalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
from time import time
from typing import Any, Callable

import requests
from pyrate_limiter import FileLockSQLiteBucket, Limiter, RequestRate
from requests_ratelimiter import LimiterSession

from .remote_config import RemotePyseriniConfig


def extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        try:
            return extract_text(json.loads(payload))
        except json.JSONDecodeError:
            return " ".join(payload.split())
    if isinstance(payload, dict):
        for key in ("contents", "text", "body", "passage", "abstract"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        return " ".join(text for text in (extract_text(value) for value in payload.values()) if text)
    if isinstance(payload, list):
        return " ".join(text for text in (extract_text(value) for value in payload) if text)
    return ""


def normalize_candidates(response: dict[str, Any]) -> list[dict[str, Any]]:
    rows = response.get("candidates") or response.get("hits") or response.get("results") or []
    normalized = []
    for rank, row in enumerate(rows, start=1):
        doc_payload = row.get("doc") or row.get("contents") or row
        text = extract_text(doc_payload)
        normalized.append(
            {
                "rank": int(row.get("rank") or rank),
                "docid": row.get("docid") or row.get("id") or row.get("_id"),
                "score": row.get("score"),
                "text": text,
                "text_length": len(text),
            }
        )
    return normalized


@dataclass(frozen=True)
class RemoteSearchResponse:
    raw: bytes
    payload: dict[str, Any]
    sha256: str


class RemotePyseriniThrottled(RuntimeError):
    """A single immutable attempt was rejected; callers must explicitly continue."""

    def __init__(self, retry_after_seconds: float | None) -> None:
        self.retry_after_seconds = retry_after_seconds
        detail = "" if retry_after_seconds is None else f"; retry after {retry_after_seconds:g}s"
        super().__init__(f"Hosted Pyserini returned HTTP 429{detail}; no retry was attempted")


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


def rate_limited_session(config: RemotePyseriniConfig) -> requests.Session:
    """Build a no-retry, persistent, per-host session with pacing before send()."""
    config.limiter_state_path.parent.mkdir(parents=True, exist_ok=True)
    limiter = Limiter(
        RequestRate(1, math.ceil(config.min_interval_seconds)),
        bucket_class=FileLockSQLiteBucket,
        bucket_kwargs={"path": config.limiter_state_path},
        time_function=time,
    )
    session = LimiterSession(
        limiter=limiter,
        per_host=True,
        max_delay=None,
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=0)
    # LimiterSession applies limiting in its request mixin; plain adapters ensure no retries.
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class RemotePyseriniClient:
    def __init__(
        self,
        config: RemotePyseriniConfig,
        session: requests.Session | None = None,
        timeout: int = 30,
    ) -> None:
        self.config = config
        self.session = session or rate_limited_session(config)
        self.timeout = timeout

    def search_raw(
        self, query: str, *, raw_sink: Callable[[bytes], None] | None = None
    ) -> RemoteSearchResponse:
        response = self.session.get(
            self.config.index_url,
            params={"query": query, "hits": str(self.config.hits)},
            headers={
                "Accept": "application/json",
                **(
                    {"Authorization": f"Bearer {self.config.api_token}"}
                    if self.config.api_token
                    else {}
                ),
            },
            timeout=self.timeout,
            allow_redirects=False,
        )
        raw = response.content
        if raw_sink is not None:
            raw_sink(raw)
        if response.status_code == 429:
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            raise RemotePyseriniThrottled(retry_after)
        response.raise_for_status()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Hosted Pyserini response must be a JSON object")
        return RemoteSearchResponse(raw=raw, payload=payload, sha256=hashlib.sha256(raw).hexdigest())

    def search(self, query: str) -> dict[str, Any]:
        return self.search_raw(query).payload

    def candidates(self, query: str) -> list[dict[str, Any]]:
        return normalize_candidates(self.search(query))
