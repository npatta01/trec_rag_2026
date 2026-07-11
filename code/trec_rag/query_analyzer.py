"""Frozen query-analyzer contracts for planner budgets and term auditing."""

from __future__ import annotations

import json
import threading
import urllib.request
from dataclasses import asdict, dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class AnalyzerFingerprint:
    contract_version: str
    implementation: str
    lucene_version: str
    analyzer_class: str
    tokenizer: str
    filters: tuple[str, ...]
    stopword_sha256: str
    unicode_version: str | None
    index_id: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AnalyzerFingerprint":
        filters = value.get("filters")
        if not isinstance(filters, list) or not all(
            isinstance(item, str) for item in filters
        ):
            raise ValueError("analyzer fingerprint filters must be a string array")
        required = (
            "contract_version",
            "implementation",
            "lucene_version",
            "analyzer_class",
            "tokenizer",
            "stopword_sha256",
        )
        missing = [key for key in required if not isinstance(value.get(key), str)]
        if missing:
            raise ValueError(
                "analyzer fingerprint lacks text field(s): " + ", ".join(missing)
            )
        return cls(
            contract_version=str(value["contract_version"]),
            implementation=str(value["implementation"]),
            lucene_version=str(value["lucene_version"]),
            analyzer_class=str(value["analyzer_class"]),
            tokenizer=str(value["tokenizer"]),
            filters=tuple(filters),
            stopword_sha256=str(value["stopword_sha256"]),
            unicode_version=(
                str(value["unicode_version"])
                if value.get("unicode_version") is not None
                else None
            ),
            index_id=(
                str(value["index_id"])
                if value.get("index_id") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class AnalyzedQuery:
    tokens: tuple[str, ...]
    unique_tokens: tuple[str, ...]
    fingerprint: AnalyzerFingerprint


class QueryAnalyzer(Protocol):
    @property
    def fingerprint(self) -> AnalyzerFingerprint:
        ...

    def analyze(self, text: str) -> AnalyzedQuery:
        ...


def stable_unique(tokens: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return tuple(result)


class RemoteLuceneQueryAnalyzer:
    """Read-only client for the pinned local Lucene analyzer sidecar."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18081",
        *,
        timeout: float = 10.0,
        expected_fingerprint: AnalyzerFingerprint | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._expected_fingerprint = expected_fingerprint
        self._fingerprint: AnalyzerFingerprint | None = None
        self._cache: dict[str, AnalyzedQuery] = {}
        self._lock = threading.Lock()

    @property
    def fingerprint(self) -> AnalyzerFingerprint:
        if self._fingerprint is None:
            self._fingerprint = self._read_fingerprint()
        return self._fingerprint

    def _read_fingerprint(self) -> AnalyzerFingerprint:
        request = urllib.request.Request(
            f"{self.base_url}/health",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or not isinstance(
            payload.get("fingerprint"), dict
        ):
            raise ValueError("analyzer health response lacks a fingerprint")
        fingerprint = AnalyzerFingerprint.from_mapping(payload["fingerprint"])
        self._verify_fingerprint(fingerprint)
        return fingerprint

    def _verify_fingerprint(self, fingerprint: AnalyzerFingerprint) -> None:
        if (
            self._expected_fingerprint is not None
            and fingerprint != self._expected_fingerprint
        ):
            raise ValueError("query analyzer fingerprint does not match frozen run")
        if self._fingerprint is not None and fingerprint != self._fingerprint:
            raise ValueError("query analyzer fingerprint changed during the run")

    def analyze(self, text: str) -> AnalyzedQuery:
        if not isinstance(text, str):
            raise TypeError("query analyzer input must be text")
        with self._lock:
            cached = self._cache.get(text)
        if cached is not None:
            return cached
        request = urllib.request.Request(
            f"{self.base_url}/analyze",
            data=text.encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "text/plain; charset=utf-8",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("analyzer response must be an object")
        raw_tokens = payload.get("tokens")
        raw_fingerprint = payload.get("fingerprint")
        if not isinstance(raw_tokens, list) or not all(
            isinstance(token, str) and token for token in raw_tokens
        ):
            raise ValueError("analyzer response tokens must be non-empty strings")
        if not isinstance(raw_fingerprint, dict):
            raise ValueError("analyzer response lacks a fingerprint")
        fingerprint = AnalyzerFingerprint.from_mapping(raw_fingerprint)
        self._verify_fingerprint(fingerprint)
        if self._fingerprint is None:
            self._fingerprint = fingerprint
        tokens = tuple(raw_tokens)
        analyzed = AnalyzedQuery(tokens, stable_unique(tokens), fingerprint)
        with self._lock:
            self._cache[text] = analyzed
        return analyzed
