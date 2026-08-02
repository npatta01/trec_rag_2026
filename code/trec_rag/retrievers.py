"""Retriever adapters and candidate normalization for the RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from filelock import FileLock

from trec_rag.pipeline_config import RetrieverConfig
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate
from trec_rag.remote_pyserini import (
    RemotePyseriniClient,
    RemotePyseriniConfig,
    RemotePyseriniThrottled,
    normalize_candidates,
)


class Retriever(Protocol):
    def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
        ...


@dataclass
class RetrieverCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    bypasses: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "bypasses": self.bypasses,
        }


def _safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "part"


MAX_SHORT_THROTTLE_RETRIES = 2
SHORT_RETRY_AFTER_SECONDS = 5.0


def cache_path(
    topic_id: str,
    variant_name: str,
    retriever_name: str,
    request_key: str | None = None,
) -> Path:
    base = f"{_safe_part(topic_id)}__{_safe_part(variant_name)}__{_safe_part(retriever_name)}"
    if request_key:
        return Path(f"{base}__{_safe_part(request_key)}.json")
    return Path(f"{base}.json")


def request_cache_key(
    config: RetrieverConfig,
    query: QueryVariant,
    *,
    index_url: str,
) -> str:
    fingerprint = {
        "retriever_name": config.name,
        "retriever_type": config.type,
        "index": config.index,
        "index_url": index_url,
        "hits": config.hits,
        "query_text": query.query_text,
    }
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def normalize_retrieved_candidates(
    response: dict[str, object],
    *,
    query: QueryVariant,
    retriever_name: str,
) -> list[RetrievedCandidate]:
    candidates: list[RetrievedCandidate] = []
    for row in normalize_candidates(response):
        docid = row.get("docid")
        if not docid or not str(docid).strip():
            raise ValueError(f"{query.topic_id}/{query.variant_name}: candidate missing docid")
        candidates.append(
            RetrievedCandidate(
                topic_id=query.topic_id,
                variant_name=query.variant_name,
                retriever_name=retriever_name,
                query_text=query.query_text,
                docid=str(docid),
                rank=int(row.get("rank") or len(candidates) + 1),
                score=float(row.get("score") or 0.0),
                text=str(row.get("text") or ""),
            )
        )
    return candidates


class PyseriniRemoteRetriever:
    def __init__(
        self,
        config: RetrieverConfig,
        *,
        cache_dir: Path,
        client: RemotePyseriniClient | None = None,
        continuation_ticket: str | None = None,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.client = client or RemotePyseriniClient(_remote_config(config))
        self.continuation_ticket = continuation_ticket or os.environ.get(
            "PYSERINI_CONTINUATION_TICKET"
        )
        self.cache_stats = RetrieverCacheStats()
        self.cache_hit_artifacts: list[dict[str, object]] = []

    def _sleep(self, seconds: float) -> None:
        """Wait out a short throttle. Overridable so tests need not really wait."""
        if seconds > 0:
            time.sleep(seconds)

    def cache_summary(self) -> dict[str, object]:
        requests = self.cache_stats.hits + self.cache_stats.misses + self.cache_stats.bypasses
        return {
            "enabled": self.config.cache,
            "requests": requests,
            "hit_artifacts": self.cache_hit_artifacts,
            **self.cache_stats.as_dict(),
        }

    def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
        request_key = request_cache_key(
            self.config,
            query,
            index_url=self.client.config.index_url,
        )
        cache_file = self.cache_dir / cache_path(
            query.topic_id,
            query.variant_name,
            self.config.name,
            request_key,
        )
        if self.config.cache and cache_file.exists():
            cache_bytes = cache_file.read_bytes()
            metadata_file = cache_file.with_suffix(".meta.json")
            if not metadata_file.exists():
                raise ValueError(f"unverified cache missing provenance sidecar: {cache_file}")
            metadata = json.loads(metadata_file.read_bytes())
            expected_identity = {
                "cache_key": request_key,
                "query": query.query_text,
                "index": self.config.index,
                "index_url": self.client.config.index_url,
                "hits": self.config.hits,
            }
            if any(metadata.get(key) != value for key, value in expected_identity.items()):
                raise ValueError(f"cache provenance identity mismatch: {metadata_file}")
            if metadata.get("response_sha256") != hashlib.sha256(cache_bytes).hexdigest():
                raise ValueError(f"cache response hash mismatch: {cache_file}")
            payload = json.loads(cache_bytes)
            response = payload.get("response") if "response" in payload else payload
            if not isinstance(response, dict):
                raise ValueError(f"cache file missing response object: {cache_file}")
            candidates = normalize_retrieved_candidates(
                response,
                query=query,
                retriever_name=self.config.name,
            )
            self.cache_stats.hits += 1
            self.cache_hit_artifacts.append(
                {
                    "file": cache_file.name,
                    "sha256": hashlib.sha256(cache_bytes).hexdigest(),
                    "candidate_count": len(candidates),
                }
            )
            return candidates
        if self.config.cache:
            self.cache_stats.misses += 1
        else:
            self.cache_stats.bypasses += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "cache_key": request_key,
            "query": query.query_text,
            "index": self.config.index,
            "index_url": self.client.config.index_url,
            "hits": self.config.hits,
        }
        ticket_file = self.cache_dir / "continuation-ticket.json"
        consumed_ticket_file = self.cache_dir / "continuation-in-progress.json"
        attempt_id = uuid.uuid4().hex
        ledger_file = self.cache_dir / "external-call-ledger.jsonl"
        ledger_lock = FileLock(str(ledger_file) + ".lock")
        with ledger_lock:
            continuation_provenance = {}
            if ticket_file.exists():
                ticket = json.loads(ticket_file.read_bytes())
                if self.continuation_ticket != ticket.get("ticket"):
                    raise RuntimeError(
                        "explicit continuation required: a throttled request is "
                        "pending and blocks every other query. Inspect it with "
                        "'.venv/bin/python -m trec_rag.continuation', then --resume to "
                        "retry it or --discard to drop it. "
                        f"(ticket {ticket.get('ticket')}, "
                        f"query {ticket.get('query')!r})"
                    )
                if any(ticket.get(key) != value for key, value in identity.items()):
                    raise RuntimeError("continuation ticket request identity mismatch")
                if time.time() < float(ticket.get("not_before_unix", 0)):
                    raise RuntimeError("Retry-After delay has not elapsed for continuation")
                continuation_provenance = {
                    "continuation_of": ticket["failed_attempt_id"],
                    "continuation_ticket_sha256": hashlib.sha256(
                        ticket["ticket"].encode("utf-8")
                    ).hexdigest(),
                }
                ticket_file.replace(consumed_ticket_file)
            elif consumed_ticket_file.exists():
                raise RuntimeError("a continuation for this external budget is already in progress")
            elif self.continuation_ticket:
                raise RuntimeError("continuation ticket is invalid or has already been consumed")
            ledger_record = {
                "event": "reserved",
                "attempt_id": attempt_id,
                "reserved_unix": time.time(),
                **continuation_provenance,
                **identity,
            }
            with ledger_file.open("a", encoding="utf-8") as ledger:
                ledger.write(json.dumps(ledger_record, sort_keys=True) + "\n")
        raw_result = None
        if hasattr(self.client, "search_raw"):
            attempt_dir = self.cache_dir / "attempts"
            attempt_dir.mkdir(exist_ok=True)
            attempt_file = attempt_dir / f"{attempt_id}.response"
            throttle_retries = 0
            while raw_result is None:
                try:
                    raw_result = self.client.search_raw(
                        query.query_text,
                        raw_sink=attempt_file.write_bytes,
                    )
                except RemotePyseriniThrottled as exc:
                    retry_after = exc.retry_after_seconds or 0.0
                    if (
                        throttle_retries < MAX_SHORT_THROTTLE_RETRIES
                        and retry_after <= SHORT_RETRY_AFTER_SECONDS
                    ):
                        # A short back-off is worth waiting out in place. Minting
                        # a ticket blocks every later query in the run, which is
                        # a ruinous price to pay for a one-second delay.
                        throttle_retries += 1
                        self._sleep(retry_after)
                        # Every transport entry stays one immutable, separately
                        # reserved attempt, so a retry is a new attempt rather
                        # than a silent second use of the first one.
                        retried_from = attempt_id
                        attempt_id = uuid.uuid4().hex
                        attempt_file = attempt_dir / f"{attempt_id}.response"
                        with ledger_lock:
                            with ledger_file.open("a", encoding="utf-8") as ledger:
                                ledger.write(
                                    json.dumps(
                                        {
                                            "event": "reserved",
                                            "attempt_id": attempt_id,
                                            "reserved_unix": time.time(),
                                            "retry_of_attempt_id": retried_from,
                                            "retry_index": throttle_retries,
                                            "throttled_retry_after_seconds": retry_after,
                                            **identity,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                        continue
                    ticket = uuid.uuid4().hex
                    with ledger_lock:
                        ticket_file.write_text(
                            json.dumps(
                                {
                                    "ticket": ticket,
                                    "not_before_unix": time.time() + retry_after,
                                    "retry_after_seconds": exc.retry_after_seconds,
                                    "failed_attempt_id": attempt_id,
                                    **identity,
                                },
                                sort_keys=True,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        consumed_ticket_file.unlink(missing_ok=True)
                    exc.continuation_ticket = ticket
                    raise
                except Exception as exc:
                    # Only an explicit throttle latches the shared budget. A
                    # transport failure fails its own request and is recorded, but
                    # must not gate every later request behind a ticket bound to
                    # this one query: the caller may never issue that query again,
                    # which wedges the retriever until someone replays it by hand.
                    with ledger_lock:
                        with ledger_file.open("a", encoding="utf-8") as ledger:
                            ledger.write(
                                json.dumps(
                                    {
                                        "event": "failed",
                                        "attempt_id": attempt_id,
                                        "failed_unix": time.time(),
                                        "failure_type": type(exc).__name__,
                                        **identity,
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                        consumed_ticket_file.unlink(missing_ok=True)
                    raise
            if self.config.cache:
                cache_file.write_bytes(attempt_file.read_bytes())
        response = raw_result.payload if raw_result is not None else self.client.search(query.query_text)
        if self.config.cache:
            raw_bytes = raw_result.raw if raw_result is not None else json.dumps(response).encode("utf-8")
            if raw_result is None:
                cache_file.write_bytes(raw_bytes)
            # The primary artifact is byte-for-byte transport output. The client invokes
            # its sink before decoding; identity/provenance lives in a separate sidecar.
            cache_file.with_suffix(".meta.json").write_text(
                json.dumps(
                    {
                        "cache_key": request_key,
                        "query": query.query_text,
                        "topic_id": query.topic_id,
                        "variant_name": query.variant_name,
                        "retriever_name": self.config.name,
                        "retriever_type": self.config.type,
                        "index": self.config.index,
                        "index_url": self.client.config.index_url,
                        "hits": self.config.hits,
                        "response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        "rate_policy": {
                            "min_interval_seconds": self.client.config.min_interval_seconds,
                            "burst": self.client.config.burst,
                            "per_host": True,
                        },
                    }, sort_keys=True, indent=2,
                ), encoding="utf-8",
            )
            self.cache_stats.writes += 1
        with ledger_lock:
            consumed_ticket_file.unlink(missing_ok=True)
        return normalize_retrieved_candidates(response, query=query, retriever_name=self.config.name)


def _remote_config(config: RetrieverConfig) -> RemotePyseriniConfig:
    env = dict(os.environ)
    if config.index:
        index_url = env.get("INDEX_URL")
        if index_url:
            url_index = _index_from_search_url(index_url)
            if url_index != config.index:
                raise ValueError(
                    "INDEX_URL conflicts with retrievers[].index "
                    f"({url_index or 'unknown'} != {config.index})"
                )
    remote = RemotePyseriniConfig.from_env(env)
    return replace(remote, hits=config.hits)


def pyserini_factory(config: RetrieverConfig, cache_dir: Path) -> PyseriniRemoteRetriever:
    return PyseriniRemoteRetriever(config, cache_dir=cache_dir)


def _index_from_search_url(index_url: str) -> str | None:
    path_parts = [
        urllib.parse.unquote(part)
        for part in urllib.parse.urlparse(index_url.rstrip("?")).path.strip("/").split("/")
        if part
    ]
    if len(path_parts) >= 3 and path_parts[-3] == "v1" and path_parts[-1] == "search":
        return path_parts[-2]
    return None
