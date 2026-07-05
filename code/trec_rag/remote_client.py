"""Hosted Pyserini search client and candidate normalization."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from email.utils import parsedate_to_datetime
from typing import Any, Callable

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


class RemotePyseriniClient:
    def __init__(
        self,
        config: RemotePyseriniConfig,
        opener: Callable[..., Any] = urllib.request.urlopen,
        timeout: int = 30,
        max_retries: int = 4,
        backoff_seconds: float = 5.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.opener = opener
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sleeper = sleeper

    def search(self, query: str) -> dict[str, Any]:
        params = urllib.parse.urlencode({"query": query, "hits": str(self.config.hits)})
        request = urllib.request.Request(
            f"{self.config.index_url}?{params}",
            headers={
                "Accept": "application/json",
                **(
                    {"Authorization": f"Bearer {self.config.api_token}"}
                    if self.config.api_token
                    else {}
                ),
            },
        )
        attempt = 0
        while True:
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise
                self.sleeper(_retry_delay(exc, self.backoff_seconds, attempt))
                attempt += 1

    def candidates(self, query: str) -> list[dict[str, Any]]:
        return normalize_candidates(self.search(query))


def _retry_delay(
    exc: urllib.error.HTTPError,
    backoff_seconds: float,
    attempt: int,
) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(retry_after).timestamp() - time.time()))
            except (TypeError, ValueError, IndexError, OverflowError):
                pass
    return backoff_seconds * (2**attempt)
