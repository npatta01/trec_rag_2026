"""Retriever adapters and candidate normalization for the RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Thread
from typing import Iterator, Protocol

from filelock import FileLock

from trec_rag.document_store import DocumentStore
from trec_rag.pipeline_config import RetrieverConfig
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate
from trec_rag.remote_pyserini import (
    RemotePyseriniClient,
    RemotePyseriniConfig,
    RemotePyseriniThrottled,
    normalize_candidates,
)
from trec_rag.retrieval_cache import (
    CachedRetrieval,
    DerivationIdentity,
    OrganizerTextNormalizer,
    RETRIEVAL_CACHE_SCHEMA_VERSION,
    RetrievalCache,
    RetrievalCacheMiss,
    TransportIdentity,
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
TOPIC_LEASE_SECONDS = 300.0
TOPIC_STATE_DIRNAME = "topic-state"
TOPIC_TICKET_NAME = "ticket"
TOPIC_IN_PROGRESS_NAME = "in-progress"
TOPIC_LEDGER_NAME = "ledger"
TOPIC_COMPLETION_NAME = "completion"
CONTINUATION_COMPLETION_SCHEMA_VERSION = "continuation-completion-v1"
KNOWN_TOPIC_LEASE_STATES = frozenset(
    {"awaiting_continuation", "active_recovery", "active_transport"}
)


class CompletionMarkerPublicationError(RuntimeError):
    """The completion receipt could not be durably published."""


def cache_path(
    topic_id: str,
    variant_name: str,
    retriever_name: str,
    request_key: str | None = None,
) -> Path:
    """Return the old display-only path shape for callers that still report it."""
    base = f"{_safe_part(topic_id)}__{_safe_part(variant_name)}__{_safe_part(retriever_name)}"
    if request_key:
        return Path(f"{base}__{_safe_part(request_key)}.json")
    return Path(f"{base}.json")


def transport_identity_for(
    config: RetrieverConfig,
    query: QueryVariant,
    *,
    index_url: str,
    corpus_epoch: str | None = None,
) -> TransportIdentity:
    index_id = config.index
    if not isinstance(index_id, str) or not index_id.strip():
        raise ValueError("retriever.index is required for a production Pyserini identity")
    url_index = _index_from_search_url(index_url)
    if url_index is not None and url_index != index_id:
        raise ValueError(
            "retriever.index conflicts with the hosted endpoint index "
            f"({index_id} != {url_index})"
        )
    epoch = corpus_epoch
    if epoch is None:
        epoch = config.corpus_epoch
    if epoch is None:
        epoch = os.environ.get("PYSERINI_CORPUS_EPOCH")
    return TransportIdentity.from_query(
        query_text=query.query_text,
        index_id=index_id,
        endpoint_identity=index_url.rstrip("?"),
        corpus_epoch=epoch,
        hits=config.hits,
    )


def request_cache_key(
    config: RetrieverConfig,
    query: QueryVariant,
    *,
    index_url: str,
    corpus_epoch: str | None = None,
) -> str:
    """Return the full v2 transport key; topic labels are not key inputs."""
    return transport_identity_for(
        config,
        query,
        index_url=index_url,
        corpus_epoch=corpus_epoch,
    ).request_key


def normalize_retrieved_candidates(
    response: dict[str, object],
    *,
    query: QueryVariant,
    retriever_name: str,
) -> list[RetrievedCandidate]:
    """Keep the historical in-memory adapter for non-remote retrievers."""
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
        retrieval_cache: RetrievalCache | None = None,
        normalizer: OrganizerTextNormalizer | None = None,
        corpus_epoch: str | None = None,
        continuation_ticket: str | None = None,
        offline: bool = False,
        cache_only: bool = False,
    ) -> None:
        resolved_continuation_ticket = continuation_ticket or os.environ.get(
            "PYSERINI_CONTINUATION_TICKET"
        )
        if cache_only and resolved_continuation_ticket:
            raise ValueError(
                "cache-only retrieval cannot use a continuation ticket"
            )
        self.config = config
        self.cache_dir = Path(cache_dir)
        self._client = client
        self.corpus_epoch = (
            corpus_epoch
            if corpus_epoch is not None
            else config.corpus_epoch
            if config.corpus_epoch is not None
            else os.environ.get("PYSERINI_CORPUS_EPOCH")
        )
        if not isinstance(self.corpus_epoch, str) or not self.corpus_epoch.strip():
            raise ValueError("corpus_epoch must be explicit for Pyserini retrieval")
        if self.corpus_epoch.strip().lower() in {"unspecified", "unknown"}:
            raise ValueError("corpus_epoch must not be synthetic")
        if not isinstance(config.index, str) or not config.index.strip():
            raise ValueError("retriever.index must be explicit for Pyserini retrieval")
        if config.index.strip().lower() in {"unknown", "unknown-index", "unspecified"}:
            raise ValueError("retriever.index must not be synthetic")
        self._endpoint = (
            str(client.config.index_url).rstrip("?")
            if client is not None
            else (os.environ.get("INDEX_URL") or "").strip().rstrip("?")
        )
        if self._endpoint:
            _validate_configured_endpoint(config, self._endpoint)
        if retrieval_cache is None:
            document_root = _document_store_root(self.cache_dir)
            retrieval_cache = RetrievalCache(
                self.cache_dir,
                DocumentStore(document_root),
                normalizer or OrganizerTextNormalizer(),
            )
        self.retrieval_cache = retrieval_cache
        self._checkpoint_derivation = DerivationIdentity.from_normalizer(
            self.retrieval_cache.normalizer
        )
        self.continuation_ticket = resolved_continuation_ticket
        self.offline = offline
        self.cache_only = cache_only
        self.cache_stats = RetrieverCacheStats()
        self.cache_hit_artifacts: list[dict[str, object]] = []
        self.transport_calls = 0

    @property
    def client(self) -> RemotePyseriniClient:
        """Construct the hosted client only after a cache miss needs transport."""
        if self._client is None:
            remote_config = _remote_config(self.config)
            if self._endpoint:
                remote_config = replace(remote_config, index_url=self._endpoint)
            client = RemotePyseriniClient(remote_config)
            client_endpoint = client.config.index_url.rstrip("?")
            if self._endpoint and client_endpoint != self._endpoint:
                raise ValueError(
                    "lazy Pyserini client endpoint differs from construction identity"
                )
            self._endpoint = client_endpoint
            self._client = client
        return self._client

    @property
    def identity(self) -> dict[str, object]:
        """Return a construction-bound checkpoint identity without touching the client."""
        endpoint = self._endpoint
        if not endpoint:
            raise ValueError(
                "INDEX_URL is required to expose the immutable Pyserini identity"
            )
        derivation = self._checkpoint_derivation.canonical_dict()
        return {
            "name": self.config.name,
            "type": self.config.type,
            "index": self.config.index,
            "index_url": endpoint,
            "hits": self.config.hits,
            "corpus_epoch": self.corpus_epoch,
            "retrieval_cache_schema": RETRIEVAL_CACHE_SCHEMA_VERSION,
            "parser_version": derivation["parser_version"],
            "extractor_version": derivation["extractor_version"],
            "field_path": list(derivation["field_path"]),
            "scoring_normalizer_version": derivation[
                "scoring_normalizer_version"
            ],
        }

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
            "retrieval_cache_schema": "organizer-retrieval-cache-v2",
            **self.cache_stats.as_dict(),
        }

    def _transport_identity(self, query: QueryVariant) -> TransportIdentity:
        endpoint = self._endpoint
        if not endpoint and self.continuation_ticket:
            endpoint = self._continuation_endpoint_identity(query)
        return transport_identity_for(
            self.config,
            query,
            index_url=endpoint or self._endpoint_identity(),
            corpus_epoch=self.corpus_epoch,
        )

    def _continuation_endpoint_identity(self, query: QueryVariant) -> str:
        """Recover the authenticated endpoint without constructing a client."""
        state = self.cache_dir / TOPIC_STATE_DIRNAME / _safe_part(query.topic_id)
        if not state.is_dir():
            raise RuntimeError(
                "continuation ticket is invalid or has already been consumed"
            )
        with FileLock(str(state / "state.lock")):
            ticket_path = state / TOPIC_TICKET_NAME
            completion_path = state / TOPIC_COMPLETION_NAME
            ticket = self._state_json(ticket_path) if ticket_path.exists() else None
            completion = (
                self._state_json(completion_path)
                if completion_path.exists()
                else None
            )
            if ticket is not None:
                if ticket.get("ticket") != self.continuation_ticket:
                    raise RuntimeError(
                        "continuation ticket is invalid or has already been consumed"
                    )
                record = ticket
            elif completion is not None:
                if completion.get("ticket_sha256") != self._token_digest(
                    self.continuation_ticket
                ):
                    raise RuntimeError(
                        "continuation ticket is invalid or has already been consumed"
                    )
                record = completion
            else:
                raise RuntimeError(
                    "continuation ticket is invalid or has already been consumed"
                )
            if (
                record.get("topic_id") != query.topic_id
                or record.get("query") != query.query_text
                or record.get("index") != self.config.index
                or record.get("hits") != self.config.hits
                or record.get("corpus_epoch") != self.corpus_epoch
            ):
                raise RuntimeError("continuation ticket request identity mismatch")
            endpoint = record.get("index_url")
            if not isinstance(endpoint, str) or not endpoint.strip():
                raise RuntimeError("continuation ticket endpoint identity is invalid")
            endpoint = endpoint.strip().rstrip("?")
            _validate_configured_endpoint(self.config, endpoint)
            self._endpoint = endpoint
            return endpoint

    def _endpoint_identity(self) -> str:
        if self._endpoint:
            return self._endpoint
        endpoint = (os.environ.get("INDEX_URL") or "").strip().rstrip("?")
        if not endpoint:
            raise ValueError("INDEX_URL is required to identify a Pyserini request")
        _validate_configured_endpoint(self.config, endpoint)
        self._endpoint = endpoint
        return endpoint

    def _derivation_identity(self) -> DerivationIdentity:
        return self._checkpoint_derivation

    def _materialize(
        self, cached: object, query: QueryVariant
    ) -> list[RetrievedCandidate]:
        result = []
        for hit in cached.hits:
            try:
                text = self.retrieval_cache.document_store.read_text(hit.content_sha256)
            except Exception as exc:
                raise ValueError("cached document object cannot be reconstructed") from exc
            result.append(
                RetrievedCandidate(
                    topic_id=query.topic_id,
                    variant_name=query.variant_name,
                    retriever_name=self.config.name,
                    query_text=query.query_text,
                    docid=hit.docid,
                    rank=hit.rank,
                    score=hit.score,
                    text=text,
                )
            )
        return result

    def _reconcile_cached_continuation(
        self,
        query: QueryVariant,
        identity: TransportIdentity,
        cached: CachedRetrieval,
    ) -> None:
        """Consume a continuation only after a locked, verified cache repair."""
        self._validate_cached_result(cached, identity, query)
        state = self._topic_state(query.topic_id)
        completion_path = state / TOPIC_COMPLETION_NAME
        ticket_path = state / TOPIC_TICKET_NAME
        progress_path = state / TOPIC_IN_PROGRESS_NAME
        with FileLock(str(state / "state.lock")):
            now = time.time()
            expected = self._identity_record(query, identity)
            marker = self._state_json(completion_path) if completion_path.exists() else None
            ticket = self._state_json(ticket_path) if ticket_path.exists() else None
            progress = self._state_json(progress_path) if progress_path.exists() else None

            if marker is not None:
                marker_not_before = self._validate_completion_record(
                    marker,
                    expected,
                    cached,
                    identity,
                    now=now,
                )
                if ticket is not None:
                    ticket_not_before = self._validate_ticket_record(
                        ticket,
                        expected,
                        now=now,
                    )
                    if ticket_not_before != marker_not_before:
                        raise RuntimeError(
                            "continuation completion marker not-before mismatch"
                        )
                    if ticket.get("failed_attempt_id") != marker.get(
                        "failed_attempt_id"
                    ):
                        raise RuntimeError(
                            "continuation completion marker attempt mismatch"
                        )
                if progress is not None:
                    lease_state, lease_expires = self._validate_progress_record(
                        progress, expected
                    )
                    if (
                        progress.get("attempt_id")
                        != marker.get("completed_progress_attempt_id")
                        or progress.get("owner")
                        != marker.get("completed_progress_owner")
                        or lease_state
                        != marker.get("completed_progress_state")
                    ):
                        if lease_expires > now and lease_state in {
                            "active_recovery",
                            "active_transport",
                        }:
                            raise RuntimeError(
                                f"topic {query.topic_id!r} has an active {lease_state} lease"
                            )
                        raise RuntimeError(
                            "continuation completion marker residual lease mismatch"
                        )
                self._cleanup_topic_files_locked(state)
                return

            if ticket is None:
                raise RuntimeError("continuation ticket is invalid or has already been consumed")
            not_before = self._validate_ticket_record(ticket, expected, now=now)
            if progress is None:
                raise RuntimeError(
                    "continuation state was consumed without a completion marker"
                )
            lease_state, lease_expires = self._validate_progress_record(progress, expected)
            if lease_state == "awaiting_continuation":
                if progress.get("attempt_id") != ticket.get("failed_attempt_id"):
                    raise RuntimeError("awaiting continuation attempt does not match its ticket")
            elif lease_expires > now:
                raise RuntimeError(
                    f"topic {query.topic_id!r} has an active {lease_state} lease"
                )
            self._write_state(
                completion_path,
                self._completion_record(
                    query,
                    identity,
                    cached,
                    not_before=not_before,
                    ticket=ticket,
                    progress=progress,
                ),
            )
            self._cleanup_topic_files_locked(state)

    def retrieve(self, query: QueryVariant) -> list[RetrievedCandidate]:
        identity = self._transport_identity(query)
        derivation = self._derivation_identity()
        if self.cache_only:
            cached = self.retrieval_cache.lookup_complete(
                identity,
                derivation,
                query.query_text,
            )
            if cached is None:
                self.cache_stats.misses += 1
                raise RetrievalCacheMiss(
                    f"cache-only retrieval miss for request {identity.request_key}"
                )
            self.cache_stats.hits += 1
            self.cache_hit_artifacts.append(
                {
                    "request_key": identity.request_key,
                    "raw_sha256": cached.raw_sha256,
                    "candidate_count": len(cached.hits),
                }
            )
            return self._materialize(cached, query)
        if self.config.cache or self.offline:
            cached = self.retrieval_cache.lookup(
                identity,
                derivation,
                query.query_text,
                offline=self.offline,
            )
            if cached is not None:
                self.cache_stats.hits += 1
                self.cache_hit_artifacts.append(
                    {
                        "request_key": identity.request_key,
                        "raw_sha256": cached.raw_sha256,
                        "candidate_count": len(cached.hits),
                    }
                )
                materialized = self._materialize(cached, query)
                if self.continuation_ticket:
                    self._reconcile_cached_continuation(query, identity, cached)
                return materialized

        if not self.config.cache:
            self.cache_stats.bypasses += 1
            return self._retrieve_uncached(query, identity=identity, derivation=derivation)

        self.cache_stats.misses += 1
        lock = FileLock(str(self.cache_dir / "v2" / ".locks" / f"{identity.request_key}.request"))
        with lock:
            # The first caller may have published while this caller was waiting.
            cached = self.retrieval_cache.lookup(
                identity, derivation, query.query_text, offline=self.offline
            )
            if cached is not None:
                self.cache_stats.misses -= 1
                self.cache_stats.hits += 1
                self.cache_hit_artifacts.append(
                    {
                        "request_key": identity.request_key,
                        "raw_sha256": cached.raw_sha256,
                        "candidate_count": len(cached.hits),
                    }
                )
                materialized = self._materialize(cached, query)
                if self.continuation_ticket:
                    self._reconcile_cached_continuation(query, identity, cached)
                return materialized
            return self._retrieve_uncached(query, identity=identity, derivation=derivation)

    def _topic_state(self, topic_id: str) -> Path:
        root = self.cache_dir / TOPIC_STATE_DIRNAME
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        state = root / _safe_part(topic_id)
        state.mkdir(parents=True, exist_ok=True)
        os.chmod(state, 0o700)
        return state

    def _state_json(self, path: Path) -> dict[str, object]:
        try:
            value = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid topic retrieval state: {path}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid topic retrieval state: {path}")
        return value

    @staticmethod
    def _finite_state_time(value: object, field: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid continuation {field}") from exc
        if not math.isfinite(result):
            raise RuntimeError(f"invalid continuation {field}")
        return result

    @staticmethod
    def _token_digest(token: object) -> str:
        if not isinstance(token, str) or not token:
            raise RuntimeError("continuation ticket is invalid or has already been consumed")
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_identity_record(
        record: dict[str, object],
        expected: dict[str, object],
        label: str,
    ) -> None:
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"{label} request identity mismatch")

    def _validate_ticket_record(
        self,
        ticket: dict[str, object],
        expected: dict[str, object],
        *,
        now: float,
    ) -> float:
        required = {
            "ticket",
            "not_before_unix",
            "retry_after_seconds",
            "failed_attempt_id",
            *expected,
        }
        if set(ticket) != required:
            raise RuntimeError("unknown or incomplete continuation ticket state")
        token = ticket.get("ticket")
        if not self.continuation_ticket:
            raise RuntimeError(
                "explicit continuation required; retry the recorded request"
            )
        if token != self.continuation_ticket:
            raise RuntimeError("continuation ticket is invalid or has already been consumed")
        self._token_digest(token)
        self._validate_identity_record(ticket, expected, "continuation ticket")
        failed_attempt_id = ticket.get("failed_attempt_id")
        if not isinstance(failed_attempt_id, str) or not failed_attempt_id:
            raise RuntimeError("continuation ticket is missing its failed attempt identity")
        not_before = self._finite_state_time(
            ticket.get("not_before_unix"), "not-before time"
        )
        if now < not_before:
            raise RuntimeError("Retry-After delay has not elapsed for continuation")
        return not_before

    def _validate_progress_record(
        self,
        progress: dict[str, object],
        expected: dict[str, object],
    ) -> tuple[str, float]:
        required = {
            "owner",
            "attempt_id",
            "lease_state",
            "lease_expires_unix",
            *expected,
        }
        fields = set(progress)
        optional = {"leased_at_unix", "renewed_at_unix", "retry_of_attempt_id"}
        if not required <= fields or not fields <= required | optional:
            raise RuntimeError("unknown or incomplete retrieval lease state")
        self._validate_identity_record(progress, expected, "retrieval lease")
        lease_state = progress.get("lease_state")
        self._validate_known_lease_state(progress)
        owner = progress.get("owner")
        attempt_id = progress.get("attempt_id")
        if not isinstance(owner, str) or not owner or not isinstance(attempt_id, str) or not attempt_id:
            raise RuntimeError("invalid retrieval lease identity")
        lease_expires = self._finite_state_time(
            progress.get("lease_expires_unix"), "lease expiry"
        )
        for field in ("leased_at_unix", "renewed_at_unix"):
            if field in progress:
                self._finite_state_time(progress[field], field.replace("_", " "))
        retry_of = progress.get("retry_of_attempt_id")
        if retry_of is not None and (not isinstance(retry_of, str) or not retry_of):
            raise RuntimeError("invalid retrieval retry identity")
        return str(lease_state), lease_expires

    @staticmethod
    def _validate_known_lease_state(progress: dict[str, object]) -> str:
        lease_state = progress.get("lease_state")
        if lease_state not in KNOWN_TOPIC_LEASE_STATES:
            raise RuntimeError(f"unknown retrieval lease state: {lease_state!r}")
        return str(lease_state)

    @staticmethod
    def _validate_cached_result(
        cached: CachedRetrieval,
        identity: TransportIdentity,
        query: QueryVariant,
    ) -> None:
        if (
            cached.request_key != identity.request_key
            or cached.transport_identity.canonical_dict() != identity.canonical_dict()
            or cached.query_text != query.query_text
            or not isinstance(cached.raw_sha256, str)
            or not cached.raw_sha256
        ):
            raise RuntimeError("cached continuation result identity mismatch")

    def _completion_record(
        self,
        query: QueryVariant,
        identity: TransportIdentity,
        cached: CachedRetrieval,
        *,
        not_before: float,
        ticket: dict[str, object],
        progress: dict[str, object],
    ) -> dict[str, object]:
        token_digest = self._token_digest(self.continuation_ticket)
        record: dict[str, object] = {
            "schema_version": CONTINUATION_COMPLETION_SCHEMA_VERSION,
            "state": "completed",
            "ticket_sha256": token_digest,
            "raw_sha256": cached.raw_sha256,
            "request_digest": identity.request_key,
            "derivation_key": cached.derivation_key,
            "hit_count": len(cached.hits),
            "ordered_hit_refs_sha256": self._ordered_cached_hit_digest(cached),
            "not_before_unix": not_before,
            "failed_attempt_id": ticket["failed_attempt_id"],
            "completed_progress_attempt_id": progress["attempt_id"],
            "completed_progress_owner": progress["owner"],
            "completed_progress_state": progress["lease_state"],
            **self._identity_record(query, identity),
        }
        record["completion_sha256"] = self._canonical_state_digest(record)
        return record

    @staticmethod
    def _canonical_state_digest(value: dict[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _ordered_cached_hit_digest(cached: CachedRetrieval) -> str:
        payload = [
            [hit.rank, hit.score.hex(), hit.docid, hit.content_sha256]
            for hit in cached.hits
        ]
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _validate_completion_record(
        self,
        marker: dict[str, object],
        expected: dict[str, object],
        cached: CachedRetrieval,
        identity: TransportIdentity,
        *,
        now: float,
    ) -> float:
        required = {
            "schema_version",
            "state",
            "ticket_sha256",
            "raw_sha256",
            "request_digest",
            "derivation_key",
            "hit_count",
            "ordered_hit_refs_sha256",
            "not_before_unix",
            "failed_attempt_id",
            "completed_progress_attempt_id",
            "completed_progress_owner",
            "completed_progress_state",
            "completion_sha256",
            *expected,
        }
        if set(marker) != required:
            raise RuntimeError("unknown or incomplete continuation completion marker")
        if (
            marker.get("schema_version") != CONTINUATION_COMPLETION_SCHEMA_VERSION
            or marker.get("state") != "completed"
            or marker.get("ticket_sha256") != self._token_digest(self.continuation_ticket)
            or marker.get("raw_sha256") != cached.raw_sha256
            or marker.get("request_digest") != identity.request_key
            or marker.get("derivation_key") != cached.derivation_key
            or marker.get("hit_count") != len(cached.hits)
            or marker.get("ordered_hit_refs_sha256")
            != self._ordered_cached_hit_digest(cached)
        ):
            raise RuntimeError("continuation completion marker does not match the request")
        marker_body = dict(marker)
        marker_digest = marker_body.pop("completion_sha256")
        if marker_digest != self._canonical_state_digest(marker_body):
            raise RuntimeError("continuation completion marker digest mismatch")
        if (
            not isinstance(marker.get("failed_attempt_id"), str)
            or not marker["failed_attempt_id"]
            or not isinstance(marker.get("completed_progress_attempt_id"), str)
            or not marker["completed_progress_attempt_id"]
            or not isinstance(marker.get("completed_progress_owner"), str)
            or not marker["completed_progress_owner"]
            or marker.get("completed_progress_state")
            not in KNOWN_TOPIC_LEASE_STATES
        ):
            raise RuntimeError("continuation completion marker state is invalid")
        self._validate_identity_record(marker, expected, "continuation completion marker")
        not_before = self._finite_state_time(
            marker.get("not_before_unix"), "completion not-before time"
        )
        if now < not_before:
            raise RuntimeError("Retry-After delay has not elapsed for continuation")
        return not_before

    @staticmethod
    def _unlink_state_file(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _cleanup_topic_files_locked(self, state: Path) -> None:
        self._unlink_state_file(state / TOPIC_IN_PROGRESS_NAME)
        self._unlink_state_file(state / TOPIC_TICKET_NAME)

    def _write_state(self, path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as sink:
                os.fchmod(sink.fileno(), 0o600)
                json.dump(value, sink, sort_keys=True, indent=2)
                sink.write("\n")
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _identity_record(self, query: QueryVariant, identity: TransportIdentity) -> dict[str, object]:
        return {
            "topic_id": query.topic_id,
            "query": query.query_text,
            "request_key": identity.request_key,
            "index": self.config.index,
            "index_url": self._endpoint_identity(),
            "hits": self.config.hits,
            "corpus_epoch": identity.corpus_epoch,
        }

    def _reserve_topic_attempt(
        self, query: QueryVariant, identity: TransportIdentity
    ) -> tuple[Path, Path, str, dict[str, object], str]:
        state = self._topic_state(query.topic_id)
        state.mkdir(parents=True, exist_ok=True)
        os.chmod(state, 0o700)
        state_lock = FileLock(str(state / "state.lock"))
        owner = f"{os.getpid()}-{uuid.uuid4().hex}"
        attempt_id = uuid.uuid4().hex
        ticket_path = state / TOPIC_TICKET_NAME
        progress_path = state / TOPIC_IN_PROGRESS_NAME
        with state_lock:
            now = time.time()
            expected = self._identity_record(query, identity)
            ticket = self._state_json(ticket_path) if ticket_path.exists() else None
            completion_path = state / TOPIC_COMPLETION_NAME
            if completion_path.exists():
                if (
                    self.continuation_ticket
                    or ticket is not None
                    or progress_path.exists()
                ):
                    raise RuntimeError(
                        "completion marker requires a verified cache hit before retry"
                    )
                self._unlink_state_file(completion_path)
            continuation: dict[str, object] = {}
            if ticket is not None:
                self._validate_ticket_record(ticket, expected, now=now)
                continuation = {
                    "continuation_of": ticket["failed_attempt_id"],
                    "continuation_ticket_sha256": hashlib.sha256(
                        str(ticket["ticket"]).encode("utf-8")
                    ).hexdigest(),
                }
            elif self.continuation_ticket:
                raise RuntimeError(
                    "explicit continuation ticket is invalid or has already been consumed"
                )
            if progress_path.exists():
                progress = self._state_json(progress_path)
                lease_state, lease_expires = self._validate_progress_record(progress, expected)
                lease_expired = lease_expires <= now
                if ticket is not None:
                    awaiting = lease_state == "awaiting_continuation"
                    if awaiting:
                        if progress.get("attempt_id") != ticket.get("failed_attempt_id"):
                            raise RuntimeError(
                                "awaiting continuation attempt does not match its ticket"
                            )
                    elif lease_state == "active_recovery" and not lease_expired:
                        raise RuntimeError(
                            f"topic {query.topic_id!r} has an active recovery lease"
                        )
                    elif not lease_expired:
                        raise RuntimeError(
                            f"topic {query.topic_id!r} has an active retrieval lease"
                        )
                elif not lease_expired:
                    raise RuntimeError(
                        f"topic {query.topic_id!r} has an active retrieval lease"
                    )
            elif ticket is not None:
                raise RuntimeError("continuation state is missing its retrieval lease")
            progress = {
                "owner": owner,
                "attempt_id": attempt_id,
                "lease_state": (
                    "active_recovery" if ticket is not None else "active_transport"
                ),
                "leased_at_unix": now,
                "lease_expires_unix": now + TOPIC_LEASE_SECONDS,
                **expected,
            }
            self._write_state(progress_path, progress)
            ledger_record = {
                "event": "reserved",
                "attempt_id": attempt_id,
                "reserved_unix": now,
                "owner": owner,
                **continuation,
                **expected,
            }
            ledger_path = state / TOPIC_LEDGER_NAME
            ledger_fd = os.open(
                ledger_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            os.chmod(ledger_path, 0o600)
            with os.fdopen(ledger_fd, "a", encoding="utf-8") as ledger:
                ledger.write(json.dumps(ledger_record, sort_keys=True) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
        self.retrieval_cache.v2_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.retrieval_cache.v2_root, 0o700)
        attempts_root = self.retrieval_cache.v2_root / "attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        os.chmod(attempts_root, 0o700)
        attempt_dir = attempts_root / identity.request_key / attempt_id
        attempt_dir.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(attempt_dir.parent, 0o700)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(attempt_dir, 0o700)
        self._write_state(
            attempt_dir / "request.json",
            {
                "schema_version": "organizer-retrieval-cache-v2",
                "request_key": identity.request_key,
                "transport_identity": identity.canonical_dict(),
                "query_text": query.query_text,
                "attempt_id": attempt_id,
            },
        )
        return state, progress_path, attempt_id, continuation, owner

    def _renew_topic_lease(
        self,
        progress_path: Path,
        attempt_id: str,
        owner: str,
        *,
        retry_after: float | None = None,
    ) -> None:
        state = progress_path.parent
        with FileLock(str(state / "state.lock")):
            progress = self._state_json(progress_path)
            self._validate_known_lease_state(progress)
            if (
                progress.get("attempt_id") != attempt_id
                or progress.get("owner") != owner
            ):
                raise RuntimeError("retrieval lease owner changed")
            now = time.time()
            progress["renewed_at_unix"] = now
            lease_extension = TOPIC_LEASE_SECONDS
            if retry_after is not None:
                lease_extension = max(
                    lease_extension,
                    float(retry_after) + 1.0,
                )
            progress["lease_expires_unix"] = now + lease_extension
            self._write_state(progress_path, progress)

    @contextmanager
    def _lease_heartbeat(
        self,
        progress_path: Path,
        attempt_id: str,
        owner: str,
    ) -> Iterator[None]:
        """Keep one owned attempt live and surface any ownership loss."""
        self._renew_topic_lease(progress_path, attempt_id, owner)
        stopped = Event()
        failures: list[BaseException] = []
        interval = max(0.01, min(30.0, TOPIC_LEASE_SECONDS / 3.0))

        def heartbeat() -> None:
            while not stopped.wait(interval):
                try:
                    self._renew_topic_lease(progress_path, attempt_id, owner)
                except BaseException as exc:
                    failures.append(exc)
                    stopped.set()
                    return

        worker = Thread(
            target=heartbeat,
            name=f"retrieval-lease-{attempt_id[:8]}",
            daemon=True,
        )
        worker.start()
        try:
            yield
        except BaseException:
            stopped.set()
            worker.join()
            raise
        else:
            stopped.set()
            worker.join()
            if failures:
                raise RuntimeError("retrieval lease heartbeat lost ownership") from failures[0]

    def _publish_completion_marker_locked(
        self,
        state: Path,
        query: QueryVariant,
        identity: TransportIdentity,
        cached: CachedRetrieval,
        *,
        progress: dict[str, object] | None = None,
    ) -> None:
        ticket_path = state / TOPIC_TICKET_NAME
        if not ticket_path.exists():
            raise CompletionMarkerPublicationError(
                "completion marker cannot be published without a continuation ticket"
            )
        expected = self._identity_record(query, identity)
        ticket = self._state_json(ticket_path)
        if progress is None:
            progress = self._state_json(state / TOPIC_IN_PROGRESS_NAME)
        try:
            not_before = self._validate_ticket_record(
                ticket,
                expected,
                now=time.time(),
            )
            marker = self._completion_record(
                query,
                identity,
                cached,
                not_before=not_before,
                ticket=ticket,
                progress=progress,
            )
            marker_path = state / TOPIC_COMPLETION_NAME
            if marker_path.exists():
                self._validate_completion_record(
                    self._state_json(marker_path),
                    expected,
                    cached,
                    identity,
                    now=time.time(),
                )
                return
            self._write_state(marker_path, marker)
        except CompletionMarkerPublicationError:
            raise
        except Exception as exc:
            raise CompletionMarkerPublicationError(
                "completion marker publication failed; continuation state was retained"
            ) from exc

    def _finish_topic(
        self,
        state: Path,
        progress_path: Path,
        *,
        owner: str,
        attempt_id: str,
        clear_ticket: bool = True,
        completion: tuple[QueryVariant, TransportIdentity, CachedRetrieval] | None = None,
    ) -> None:
        with FileLock(str(state / "state.lock")):
            if not progress_path.exists():
                return
            progress = self._state_json(progress_path)
            if (
                progress.get("owner") != owner
                or progress.get("attempt_id") != attempt_id
            ):
                return
            self._validate_known_lease_state(progress)
            if completion is not None:
                self._publish_completion_marker_locked(
                    state,
                    *completion,
                    progress=progress,
                )
            if clear_ticket:
                self._cleanup_topic_files_locked(state)
            else:
                self._unlink_state_file(progress_path)

    def _write_attempt_response(self, attempt_id: str, raw: bytes) -> None:
        attempt_dir = self.retrieval_cache.v2_root / "attempts"
        paths = list(attempt_dir.glob(f"*/{attempt_id}/response.bin"))
        if not paths:
            return
        path = paths[0]
        with path.open("wb") as sink:
            sink.write(raw)
            sink.flush()
            os.fsync(sink.fileno())

    def _append_failure(
        self,
        state: Path,
        query: QueryVariant,
        identity: TransportIdentity,
        attempt_id: str,
        exc: Exception,
    ) -> None:
        with FileLock(str(state / "state.lock")):
            ledger_path = state / TOPIC_LEDGER_NAME
            ledger_fd = os.open(ledger_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(ledger_fd, "a", encoding="utf-8") as ledger:
                ledger.write(
                    json.dumps(
                        {
                            "event": "failed",
                            "attempt_id": attempt_id,
                            "failed_unix": time.time(),
                            "failure_type": type(exc).__name__,
                            **self._identity_record(query, identity),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                ledger.flush()
                os.fsync(ledger.fileno())

    def _retrieve_uncached(
        self,
        query: QueryVariant,
        *,
        identity: TransportIdentity,
        derivation: DerivationIdentity,
    ) -> list[RetrievedCandidate]:
        state, progress_path, attempt_id, continuation, owner = self._reserve_topic_attempt(
            query, identity
        )
        attempt_dir = (
            self.retrieval_cache.v2_root
            / "attempts"
            / identity.request_key
            / attempt_id
        )
        response_path = attempt_dir / "response.bin"
        raw_result = None
        cached_result: CachedRetrieval | None = None
        try:
            with self._lease_heartbeat(progress_path, attempt_id, owner):
                client = self.client
            if self.config.cache and not hasattr(client, "search_raw"):
                raise TypeError("cache-enabled Pyserini clients must expose search_raw")
            if hasattr(client, "search_raw"):
                throttle_retries = 0
                while raw_result is None:
                    try:
                        with self._lease_heartbeat(
                            progress_path,
                            attempt_id,
                            owner,
                        ):
                            self.transport_calls += 1
                            raw_result = client.search_raw(
                                query.query_text,
                                raw_sink=lambda raw: self._write_raw_attempt(
                                    response_path,
                                    raw,
                                ),
                            )
                    except RemotePyseriniThrottled as exc:
                        retry_after = exc.retry_after_seconds or 0.0
                        self._renew_topic_lease(
                            progress_path, attempt_id, owner, retry_after=retry_after
                        )
                        if (
                            throttle_retries < MAX_SHORT_THROTTLE_RETRIES
                            and retry_after <= SHORT_RETRY_AFTER_SECONDS
                        ):
                            throttle_retries += 1
                            self._sleep(retry_after)
                            retried_from = attempt_id
                            state, progress_path, attempt_id, continuation, owner = self._reserve_retry(
                                state,
                                progress_path,
                                query,
                                identity,
                                retried_from,
                                throttle_retries,
                                retry_after,
                                continuation,
                                owner,
                            )
                            attempt_dir = self.retrieval_cache.v2_root / "attempts" / identity.request_key / attempt_id
                            response_path = attempt_dir / "response.bin"
                            continue
                        ticket = uuid.uuid4().hex
                        with FileLock(str(state / "state.lock")):
                            progress = self._state_json(progress_path)
                            if (
                                progress.get("owner") != owner
                                or progress.get("attempt_id") != attempt_id
                            ):
                                raise RuntimeError(
                                    "retrieval lease owner changed before continuation"
                                )
                            self._validate_progress_record(
                                progress,
                                self._identity_record(query, identity),
                            )
                            ticket_record = {
                                "ticket": ticket,
                                "not_before_unix": time.time() + retry_after,
                                "retry_after_seconds": exc.retry_after_seconds,
                                "failed_attempt_id": attempt_id,
                                **self._identity_record(query, identity),
                            }
                            progress["lease_state"] = "awaiting_continuation"
                            self._write_state(progress_path, progress)
                            self._write_state(state / TOPIC_TICKET_NAME, ticket_record)
                        exc.continuation_ticket = ticket
                        # Leave the renewed lease and ticket durable for recovery.
                        raise
                    except Exception as exc:
                        self._append_failure(state, query, identity, attempt_id, exc)
                        self._finish_topic(
                            state,
                            progress_path,
                            owner=owner,
                            attempt_id=attempt_id,
                            clear_ticket=bool(continuation),
                        )
                        raise
            else:
                with self._lease_heartbeat(progress_path, attempt_id, owner):
                    self.transport_calls += 1
                    response = client.search(query.query_text)
                raw_result = None
            with self._lease_heartbeat(progress_path, attempt_id, owner):
                response = raw_result.payload if raw_result is not None else response
                if self.config.cache:
                    if raw_result is None:
                        raise TypeError("cache-enabled Pyserini clients must expose search_raw")
                    raw = raw_result.raw
                    self._write_raw_attempt(response_path, raw)
                    self.retrieval_cache.seal_attempt(
                        attempt_dir,
                        identity,
                        query.query_text,
                        raw,
                    )
                    cached_result = self.retrieval_cache.commit(
                        identity, derivation, query.query_text, raw
                    )
                    self.cache_stats.writes += 1
                    result = self._materialize(cached_result, query)
                else:
                    result = self._normalize_uncached_response(response, query)
                if self.config.cache:
                    self._append_ledger_event(
                        state,
                        {
                            "event": "committed",
                            "attempt_id": attempt_id,
                            "attempt_manifest": str(
                                (attempt_dir / "manifest.json").relative_to(self.cache_dir)
                            ),
                            "raw_sha256": cached_result.raw_sha256,
                        },
                    )
            self._finish_topic(
                state,
                progress_path,
                owner=owner,
                attempt_id=attempt_id,
                completion=(query, identity, cached_result)
                if continuation and cached_result is not None
                else None,
            )
            return result
        except RemotePyseriniThrottled:
            # The ticket and renewed lease are the recoverable state. Do not clear them.
            raise
        except CompletionMarkerPublicationError:
            raise
        except Exception:
            if progress_path.exists():
                self._finish_topic(
                    state,
                    progress_path,
                    owner=owner,
                    attempt_id=attempt_id,
                    clear_ticket=bool(continuation),
                    completion=(query, identity, cached_result)
                    if continuation and cached_result is not None
                    else None,
                )
            raise

    def _reserve_retry(
        self,
        state: Path,
        progress_path: Path,
        query: QueryVariant,
        identity: TransportIdentity,
        retried_from: str,
        retry_index: int,
        retry_after: float,
        continuation: dict[str, object],
        owner: str,
    ) -> tuple[Path, Path, str, dict[str, object], str]:
        with FileLock(str(state / "state.lock")):
            attempt_id = uuid.uuid4().hex
            now = time.time()
            progress = self._state_json(progress_path)
            if (
                progress.get("owner") != owner
                or progress.get("attempt_id") != retried_from
            ):
                raise RuntimeError("retrieval lease owner changed before retry")
            self._validate_progress_record(
                progress,
                self._identity_record(query, identity),
            )
            progress.update(
                {
                    "attempt_id": attempt_id,
                    "leased_at_unix": now,
                    "lease_expires_unix": now + TOPIC_LEASE_SECONDS,
                    "retry_of_attempt_id": retried_from,
                }
            )
            self._write_state(progress_path, progress)
            ledger_path = state / TOPIC_LEDGER_NAME
            ledger_fd = os.open(ledger_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(ledger_fd, "a", encoding="utf-8") as ledger:
                ledger.write(
                    json.dumps(
                        {
                            "event": "reserved",
                            "attempt_id": attempt_id,
                            "reserved_unix": now,
                            "retry_of_attempt_id": retried_from,
                            "retry_index": retry_index,
                            "throttled_retry_after_seconds": retry_after,
                            **self._identity_record(query, identity),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                ledger.flush()
                os.fsync(ledger.fileno())
        self.retrieval_cache.v2_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.retrieval_cache.v2_root, 0o700)
        attempts_root = self.retrieval_cache.v2_root / "attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        os.chmod(attempts_root, 0o700)
        attempt_dir = attempts_root / identity.request_key / attempt_id
        attempt_dir.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(attempt_dir.parent, 0o700)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(attempt_dir, 0o700)
        self._write_state(
            attempt_dir / "request.json",
            {
                "schema_version": "organizer-retrieval-cache-v2",
                "request_key": identity.request_key,
                "transport_identity": identity.canonical_dict(),
                "query_text": query.query_text,
                "attempt_id": attempt_id,
            },
        )
        return state, progress_path, attempt_id, continuation, owner

    def _append_ledger_event(self, state: Path, record: dict[str, object]) -> None:
        ledger_path = state / TOPIC_LEDGER_NAME
        ledger_fd = os.open(ledger_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            with os.fdopen(ledger_fd, "a", encoding="utf-8") as ledger:
                ledger.write(json.dumps(record, sort_keys=True) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
        finally:
            os.chmod(ledger_path, 0o600)

    @staticmethod
    def _write_raw_attempt(path: Path, raw: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        with path.open("wb") as sink:
            os.fchmod(sink.fileno(), 0o600)
            sink.write(raw)
            sink.flush()
            os.fsync(sink.fileno())

    def _normalize_uncached_response(
        self, response: dict[str, object], query: QueryVariant
    ) -> list[RetrievedCandidate]:
        raw = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            parsed = self.retrieval_cache.normalizer.parse(raw)
        except Exception as exc:
            if not self.config.cache:
                # Explicitly non-authoritative compatibility for cache=False only.
                return normalize_retrieved_candidates(
                    response,
                    query=query,
                    retriever_name=self.config.name,
                )
            raise ValueError("strict organizer response parsing failed") from exc
        return [
            RetrievedCandidate(
                topic_id=query.topic_id,
                variant_name=query.variant_name,
                retriever_name=self.config.name,
                query_text=query.query_text,
                docid=row.docid,
                rank=row.rank,
                score=row.score,
                text=row.text,
            )
            for row in parsed
        ]


def _remote_config(config: RetrieverConfig) -> RemotePyseriniConfig:
    env = dict(os.environ)
    if config.index:
        index_url = env.get("INDEX_URL")
        if index_url:
            _validate_configured_endpoint(config, index_url)
    remote = RemotePyseriniConfig.from_env(env)
    return replace(remote, hits=config.hits)


def _validate_configured_endpoint(config: RetrieverConfig, index_url: str) -> None:
    if config.index:
        url_index = _index_from_search_url(index_url)
        if url_index is not None and url_index != config.index:
            raise ValueError(
                "INDEX_URL conflicts with retrievers[].index "
                f"({url_index} != {config.index})"
            )


def _document_store_root(cache_dir: Path) -> Path:
    """Resolve the one shared cache document namespace without basename heuristics."""
    cache_dir = Path(cache_dir)
    if cache_dir.parent.name == "retrieval":
        cache_root = cache_dir.parent.parent
    else:
        cache_root = cache_dir
    return cache_root / "documents" / "v1"


def pyserini_factory(config: RetrieverConfig, cache_dir: Path) -> PyseriniRemoteRetriever:
    if (
        not isinstance(config.corpus_epoch, str)
        or not config.corpus_epoch.strip()
        or config.corpus_epoch.strip().lower() in {"unknown", "unspecified"}
    ):
        raise ValueError("retrievers[].corpus_epoch is required for production Pyserini builders")
    return PyseriniRemoteRetriever(
        config,
        cache_dir=cache_dir,
        corpus_epoch=config.corpus_epoch,
    )


def _index_from_search_url(index_url: str) -> str | None:
    path_parts = [
        urllib.parse.unquote(part)
        for part in urllib.parse.urlparse(index_url.rstrip("?")).path.strip("/").split("/")
        if part
    ]
    if len(path_parts) >= 3 and path_parts[-3] == "v1" and path_parts[-1] == "search":
        return path_parts[-2]
    return None
