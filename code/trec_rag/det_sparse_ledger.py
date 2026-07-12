"""Crash-safe, raw-first ledger for externally billed sparse retrieval.

The ledger deliberately does not implement an HTTP client.  Callers inject a
transport which returns the exact response bytes, making network access an
explicit and testable boundary.  A reservation is durably committed before the
transport is entered; reservations are never removed or retried.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


LEDGER_SCHEMA_VERSION = "det-sparse-retrieval-ledger-v1"
RETRIEVER_VERSION = "pyserini_remote_raw_first_v1"
DEFAULT_MAX_CALLS = 36
DEFAULT_MAX_CALLS_PER_TOPIC = 9
DEFAULT_MIN_RESULTS = 50
DEFAULT_REQUIRED_TEXT_RESULTS = 50
MAX_RESULTS = 100
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RetrievalLedgerError(RuntimeError):
    """Base class for ledger, validation, and retrieval failures."""


class LedgerIntegrityError(RetrievalLedgerError):
    """An on-disk artifact is missing, inconsistent, or corrupt."""


class CallBudgetExceeded(RetrievalLedgerError):
    """The run has no external-call reservations remaining."""


class ReplayRefused(RetrievalLedgerError):
    """This exact request already has a pending or failed attempt."""


class ResponseValidationError(RetrievalLedgerError):
    """The raw response cannot be admitted as normalized candidates."""


@dataclass(frozen=True)
class RetrievalRequestIdentity:
    """Every field which can change the semantic retrieval request."""

    topic_id: str
    variant_name: str
    retriever_version: str
    query_sha256: str
    index_url: str
    index_id: str
    hits: int
    analyzer_fingerprint_sha256: str
    bm25_k1: float | None = None
    bm25_b: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.topic_id, "topic_id")
        _require_nonempty(self.variant_name, "variant_name")
        _require_nonempty(self.retriever_version, "retriever_version")
        _require_sha256(self.query_sha256, "query_sha256")
        _require_nonempty(self.index_url, "index_url")
        _require_nonempty(self.index_id, "index_id")
        if isinstance(self.hits, bool) or not isinstance(self.hits, int) or self.hits <= 0:
            raise ValueError("hits must be a positive integer")
        _require_sha256(
            self.analyzer_fingerprint_sha256,
            "analyzer_fingerprint_sha256",
        )
        if (self.bm25_k1 is None) != (self.bm25_b is None):
            raise ValueError("bm25_k1 and bm25_b must be provided together")
        if self.bm25_k1 is not None:
            if (
                isinstance(self.bm25_k1, bool)
                or not isinstance(self.bm25_k1, (int, float))
                or not math.isfinite(float(self.bm25_k1))
                or self.bm25_k1 < 0
            ):
                raise ValueError("bm25_k1 must be a finite non-negative number")
            if (
                isinstance(self.bm25_b, bool)
                or not isinstance(self.bm25_b, (int, float))
                or not math.isfinite(float(self.bm25_b))
                or not 0 <= self.bm25_b <= 1
            ):
                raise ValueError("bm25_b must be a finite number between zero and one")

    @classmethod
    def from_query(
        cls,
        *,
        topic_id: str,
        variant_name: str,
        query_text: str,
        index_url: str,
        index_id: str,
        hits: int,
        analyzer_fingerprint_sha256: str,
        retriever_version: str = RETRIEVER_VERSION,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
    ) -> "RetrievalRequestIdentity":
        if not isinstance(query_text, str):
            raise TypeError("query_text must be a string")
        _require_nonempty(query_text, "query_text")
        return cls(
            topic_id=topic_id,
            variant_name=variant_name,
            retriever_version=retriever_version,
            query_sha256=_sha256(query_text.encode("utf-8")),
            index_url=index_url,
            index_id=index_id,
            hits=hits,
            analyzer_fingerprint_sha256=analyzer_fingerprint_sha256,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )

    def canonical_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "analyzer_fingerprint_sha256": self.analyzer_fingerprint_sha256,
            "hits": self.hits,
            "index_id": self.index_id,
            "index_url": self.index_url,
            "query_sha256": self.query_sha256,
            "retriever_version": self.retriever_version,
            "topic_id": self.topic_id,
            "variant_name": self.variant_name,
        }
        if self.bm25_k1 is not None:
            value["bm25_k1"] = float(self.bm25_k1)
            value["bm25_b"] = float(self.bm25_b)
        return value

    @property
    def request_key(self) -> str:
        return _sha256(_canonical_bytes(self.canonical_dict()))


@dataclass(frozen=True)
class RetrievalRequest:
    """An identity plus the exact query text required by the transport."""

    identity: RetrievalRequestIdentity
    query_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.query_text, str):
            raise TypeError("query_text must be a string")
        _require_nonempty(self.query_text, "query_text")
        actual = _sha256(self.query_text.encode("utf-8"))
        if actual != self.identity.query_sha256:
            raise ValueError("query_text does not match identity.query_sha256")

    @classmethod
    def from_query(
        cls,
        *,
        topic_id: str,
        variant_name: str,
        query_text: str,
        index_url: str,
        index_id: str,
        hits: int,
        analyzer_fingerprint_sha256: str,
        retriever_version: str = RETRIEVER_VERSION,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
    ) -> "RetrievalRequest":
        identity = RetrievalRequestIdentity.from_query(
            topic_id=topic_id,
            variant_name=variant_name,
            query_text=query_text,
            index_url=index_url,
            index_id=index_id,
            hits=hits,
            analyzer_fingerprint_sha256=analyzer_fingerprint_sha256,
            retriever_version=retriever_version,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
        return cls(identity=identity, query_text=query_text)


@dataclass(frozen=True)
class RawTransportResponse:
    """Exact output of the injected external transport."""

    status: int
    headers: Mapping[str, str]
    body: bytes
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if isinstance(self.status, bool) or not isinstance(self.status, int):
            raise TypeError("transport status must be an integer")
        if not 100 <= self.status <= 599:
            raise ValueError("transport status must be between 100 and 599")
        if not isinstance(self.body, bytes):
            raise TypeError("transport body must be bytes")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be a finite non-negative number")
        _normalize_headers(self.headers)


class RetrievalTransport(Protocol):
    def __call__(self, request: RetrievalRequest) -> RawTransportResponse:
        ...


@dataclass(frozen=True)
class NormalizedCandidate:
    docid: str
    rank: int
    score: float
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    request_key: str
    candidates: tuple[NormalizedCandidate, ...]
    response_sha256: str
    candidates_sha256: str
    cache_hit: bool
    external_calls: int


@dataclass(frozen=True)
class RunValidationReport:
    reservations: int
    successes: int
    failures: int
    pending: int
    cache_hits: int
    per_topic_external_calls: dict[str, int]

    @property
    def external_calls(self) -> int:
        return self.reservations

    @property
    def planned_requests(self) -> int:
        return self.reservations + self.cache_hits


class RetrievalLedger:
    """Durable call ledger with an exact-identity, self-verifying shared cache."""

    def __init__(
        self,
        run_dir: Path,
        *,
        shared_cache_dir: Path | None = None,
        max_calls: int = DEFAULT_MAX_CALLS,
        max_calls_per_topic: int = DEFAULT_MAX_CALLS_PER_TOPIC,
        min_results: int = DEFAULT_MIN_RESULTS,
        required_text_results: int = DEFAULT_REQUIRED_TEXT_RESULTS,
    ) -> None:
        if (
            isinstance(max_calls, bool)
            or not isinstance(max_calls, int)
            or not 1 <= max_calls <= DEFAULT_MAX_CALLS
        ):
            raise ValueError(
                f"max_calls must be between 1 and the hard ceiling {DEFAULT_MAX_CALLS}"
            )
        if (
            isinstance(max_calls_per_topic, bool)
            or not isinstance(max_calls_per_topic, int)
            or not 1 <= max_calls_per_topic <= DEFAULT_MAX_CALLS_PER_TOPIC
        ):
            raise ValueError(
                "max_calls_per_topic must be between 1 and the hard ceiling "
                f"{DEFAULT_MAX_CALLS_PER_TOPIC}"
            )
        if (
            isinstance(min_results, bool)
            or not isinstance(min_results, int)
            or not 1 <= min_results <= MAX_RESULTS
        ):
            raise ValueError(f"min_results must be between 1 and {MAX_RESULTS}")
        if (
            isinstance(required_text_results, bool)
            or not isinstance(required_text_results, int)
            or required_text_results < DEFAULT_REQUIRED_TEXT_RESULTS
            or required_text_results > MAX_RESULTS
        ):
            raise ValueError(
                "required_text_results cannot weaken the mandatory top-50 text check"
            )
        self.run_dir = Path(run_dir)
        self.shared_cache_dir = (
            Path(shared_cache_dir) if shared_cache_dir is not None else None
        )
        self.max_calls = max_calls
        self.max_calls_per_topic = max_calls_per_topic
        self.min_results = min_results
        self.required_text_results = required_text_results
        self._initialize_run()

    @property
    def attempts_dir(self) -> Path:
        return self.run_dir / "attempts"

    @property
    def raw_dir(self) -> Path:
        return self.run_dir / "raw"

    @property
    def candidates_dir(self) -> Path:
        return self.run_dir / "candidates"

    @property
    def outcomes_dir(self) -> Path:
        return self.run_dir / "outcomes"

    @property
    def cache_hits_dir(self) -> Path:
        return self.run_dir / "cache_hits"

    def reservation_path(self, request_key: str) -> Path:
        _require_sha256(request_key, "request_key")
        return self.attempts_dir / f"{request_key}.reservation.json"

    def raw_path(self, request_key: str) -> Path:
        _require_sha256(request_key, "request_key")
        return self.raw_dir / f"{request_key}.body"

    def raw_metadata_path(self, request_key: str) -> Path:
        _require_sha256(request_key, "request_key")
        return self.raw_dir / f"{request_key}.metadata.json"

    def candidates_path(self, request_key: str) -> Path:
        _require_sha256(request_key, "request_key")
        return self.candidates_dir / f"{request_key}.json"

    def outcome_path(self, request_key: str) -> Path:
        _require_sha256(request_key, "request_key")
        return self.outcomes_dir / f"{request_key}.json"

    def cache_hit_path(self, request_key: str) -> Path:
        _require_sha256(request_key, "request_key")
        return self.cache_hits_dir / f"{request_key}.json"

    def call_count(self) -> int:
        """Return durable reservations; crashed/pending attempts are included."""

        with self._run_lock():
            return len(self._validated_reservations())

    def has_verified_cache(self, request: RetrievalRequest) -> bool:
        """Check an exact shared-cache entry without recording an invocation."""

        if self.shared_cache_dir is None:
            return False
        with self._shared_cache_lock(request.identity.request_key):
            return self._load_cache(request) is not None

    def retrieve(
        self,
        request: RetrievalRequest,
        transport: RetrievalTransport,
    ) -> RetrievalResult:
        """Load an exact cache hit or make exactly one newly reserved call."""

        if request.identity.hits < self.min_results:
            raise ResponseValidationError(
                f"request asks for {request.identity.hits} hits but run requires at "
                f"least {self.min_results}; refusing before reservation"
            )
        if request.identity.hits > MAX_RESULTS:
            raise ResponseValidationError(
                f"request asks for {request.identity.hits} hits but frozen depth is "
                f"at most {MAX_RESULTS}; refusing before reservation"
            )

        if self.shared_cache_dir is None:
            return self._retrieve_locked(request, transport)
        with self._shared_cache_lock(request.identity.request_key):
            # The cache must be rechecked only after acquiring this exact-key lock.
            # This prevents two runs from paying for the same simultaneous miss.
            return self._retrieve_locked(request, transport)

    def _retrieve_locked(
        self,
        request: RetrievalRequest,
        transport: RetrievalTransport,
    ) -> RetrievalResult:

        key = request.identity.request_key
        self._refuse_existing_incomplete_or_failed(request)

        cache_evidence_exists = self.cache_hit_path(request.identity.request_key).exists()
        cached = self._load_cache(request)
        if cached is not None:
            if cache_evidence_exists:
                self._validate_cache_hit_evidence(request, cached)
                raise ReplayRefused(
                    f"request {key} already has a recorded cache-hit invocation"
                )
            self._record_cache_hit(request, cached)
            return cached
        if cache_evidence_exists:
            raise LedgerIntegrityError(
                "run records a cache hit but its shared cache entry is unavailable"
            )

        reservation = self._reserve(request)
        ordinal = int(reservation["ordinal"])
        try:
            response = transport(request)
            if not isinstance(response, RawTransportResponse):
                raise TypeError("transport must return RawTransportResponse")
        except Exception as exc:
            self._write_failure_outcome(
                identity=request.identity,
                ordinal=ordinal,
                failure_type="transport_error",
                message=_exception_message(exc),
                raw_present=False,
            )
            raise RetrievalLedgerError(
                f"external retrieval failed for {key}: {_exception_message(exc)}"
            ) from exc

        # No body inspection, decoding, or JSON parsing may precede this write.
        _atomic_create_bytes(self.raw_path(key), response.body)
        response_sha256 = _sha256(response.body)
        raw_metadata = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "request_key": key,
            "identity": request.identity.canonical_dict(),
            "ordinal": ordinal,
            "http_status": response.status,
            "response_headers": _normalize_headers(response.headers),
            "elapsed_seconds": float(response.elapsed_seconds),
            "raw_body_bytes": len(response.body),
            "response_sha256": response_sha256,
            "capture_level": "exact_http_body_before_utf8_decode_or_json_parse",
        }
        _atomic_create_json(self.raw_metadata_path(key), raw_metadata)

        try:
            if not 200 <= response.status <= 299:
                raise ResponseValidationError(
                    f"HTTP status {response.status} is not successful"
                )
            candidates = _decode_and_normalize(
                response.body,
                min_results=self.min_results,
                required_text_results=self.required_text_results,
            )
            candidate_bytes = _candidate_bytes(candidates)
            candidates_sha256 = _sha256(candidate_bytes)
            if self.shared_cache_dir is not None:
                self._commit_cache(
                    request=request,
                    response=response,
                    candidates=candidates,
                    response_sha256=response_sha256,
                    candidates_sha256=candidates_sha256,
                )
            _atomic_create_bytes(self.candidates_path(key), candidate_bytes)
            outcome = {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "request_key": key,
                "identity": request.identity.canonical_dict(),
                "ordinal": ordinal,
                "status": "success",
                "raw_body_bytes": len(response.body),
                "response_sha256": response_sha256,
                "candidate_count": len(candidates),
                "candidates_sha256": candidates_sha256,
            }
            _atomic_create_json(self.outcome_path(key), outcome)
        except Exception as exc:
            if (
                not self.outcome_path(key).exists()
                and not self.candidates_path(key).exists()
            ):
                self._write_failure_outcome(
                    identity=request.identity,
                    ordinal=ordinal,
                    failure_type=(
                        "response_validation_error"
                        if isinstance(exc, (ResponseValidationError, UnicodeDecodeError, json.JSONDecodeError))
                        else "artifact_commit_error"
                    ),
                    message=_exception_message(exc),
                    raw_present=True,
                    response_sha256=response_sha256,
                )
            if isinstance(exc, RetrievalLedgerError):
                raise
            raise ResponseValidationError(
                f"retrieval response rejected for {key}: {_exception_message(exc)}"
            ) from exc

        return RetrievalResult(
            request_key=key,
            candidates=candidates,
            response_sha256=response_sha256,
            candidates_sha256=candidates_sha256,
            cache_hit=False,
            external_calls=1,
        )

    def rebuild_candidates_from_raw(
        self,
        request_key: str,
    ) -> tuple[NormalizedCandidate, ...]:
        """Offline-only reconstruction from the committed exact response body."""

        reservation = self._load_reservation(request_key)
        identity = _identity_from_mapping(reservation.get("identity"))
        _validate_stored_query(reservation, identity, "reservation")
        if identity.request_key != request_key:
            raise LedgerIntegrityError("reservation identity/request key mismatch")
        raw, metadata = self._load_verified_raw(identity)
        if not 200 <= _strict_int(metadata.get("http_status"), "http_status") <= 299:
            raise ResponseValidationError("raw response has a non-success HTTP status")
        return _decode_and_normalize(
            raw,
            min_results=self.min_results,
            required_text_results=self.required_text_results,
        )

    def load_verified_result(self, request: RetrievalRequest) -> RetrievalResult:
        """Load one completed run result after full offline integrity checks."""

        self.validate_run()
        key = request.identity.request_key
        reservation_path = self.reservation_path(key)
        cache_hit_path = self.cache_hit_path(key)
        if reservation_path.exists() and cache_hit_path.exists():
            raise LedgerIntegrityError(
                "request cannot be both an external attempt and a cache hit"
            )
        if reservation_path.exists():
            reservation = self._load_reservation(key)
            identity = _identity_from_mapping(reservation.get("identity"))
            if identity != request.identity or _validate_stored_query(
                reservation, identity, "reservation"
            ) != request.query_text:
                raise LedgerIntegrityError("verified result request identity mismatch")
            outcome = _read_json_object(self.outcome_path(key))
            self._validate_common_record(outcome, identity, reservation)
            if outcome.get("status") != "success":
                raise LedgerIntegrityError("verified result is not successful")
            self._validate_success_artifacts(identity, reservation, outcome)
            candidates = self._load_run_candidates(identity)
            return RetrievalResult(
                request_key=key,
                candidates=candidates,
                response_sha256=_require_mapping_text(outcome, "response_sha256"),
                candidates_sha256=_require_mapping_text(
                    outcome,
                    "candidates_sha256",
                ),
                cache_hit=False,
                external_calls=1,
            )
        if cache_hit_path.exists():
            cached = self._load_cache(request)
            if cached is None:
                raise LedgerIntegrityError(
                    "cache-hit evidence lacks its verified shared cache"
                )
            self._validate_cache_hit_evidence(request, cached)
            return cached
        raise LedgerIntegrityError(f"request has no completed ledger result: {key}")

    def validate_run(self) -> RunValidationReport:
        """Validate every committed artifact without issuing external calls."""

        reservations = self._validated_reservations()
        cache_hit_rows = self._validated_cache_hits()
        reservation_keys = {
            _require_mapping_text(row, "request_key") for row in reservations
        }
        cache_hit_keys = {
            _require_mapping_text(row, "request_key") for row in cache_hit_rows
        }
        if reservation_keys & cache_hit_keys:
            raise LedgerIntegrityError(
                "a request key cannot be both an external attempt and a cache hit"
            )
        self._validate_no_orphan_run_artifacts(
            reservation_keys
        )
        successes = failures = pending = 0
        for reservation in reservations:
            key = _require_mapping_text(reservation, "request_key")
            identity = _identity_from_mapping(reservation.get("identity"))
            outcome_path = self.outcome_path(key)
            raw_exists = self.raw_path(key).exists() or self.raw_metadata_path(key).exists()
            if not outcome_path.exists():
                if raw_exists or self.candidates_path(key).exists():
                    raise LedgerIntegrityError(
                        f"raw-only or derived-only attempt has no committed outcome: {key}"
                    )
                pending += 1
                continue
            outcome = _read_json_object(outcome_path)
            self._validate_common_record(outcome, identity, reservation)
            status = outcome.get("status")
            if status == "success":
                self._validate_success_artifacts(identity, reservation, outcome)
                successes += 1
            elif status == "failure":
                self._validate_failure_artifacts(identity, reservation, outcome)
                failures += 1
            else:
                raise LedgerIntegrityError(f"unsupported outcome status: {status!r}")
        for cache_hit in cache_hit_rows:
            identity = _identity_from_mapping(cache_hit.get("identity"))
            request = RetrievalRequest(
                identity=identity,
                query_text=_validate_stored_query(
                    cache_hit, identity, "cache-hit evidence"
                ),
            )
            cached = self._load_cache(request)
            if cached is None:
                raise LedgerIntegrityError(
                    "cache-hit evidence has no shared cache entry"
                )
            self._validate_cache_hit_evidence(request, cached)
        return RunValidationReport(
            reservations=len(reservations),
            successes=successes,
            failures=failures,
            pending=pending,
            cache_hits=len(cache_hit_rows),
            per_topic_external_calls={
                topic_id: sum(
                    _identity_from_mapping(row.get("identity")).topic_id == topic_id
                    for row in reservations
                )
                for topic_id in sorted(
                    {
                        _identity_from_mapping(row.get("identity")).topic_id
                        for row in reservations
                    }
                )
            },
        )

    def _initialize_run(self) -> None:
        _ensure_directory(self.run_dir)
        for directory in (
            self.attempts_dir,
            self.raw_dir,
            self.candidates_dir,
            self.outcomes_dir,
            self.cache_hits_dir,
        ):
            _ensure_directory(directory)
        manifest_path = self.run_dir / "ledger.json"
        expected = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "max_calls": self.max_calls,
            "max_calls_per_topic": self.max_calls_per_topic,
            "min_results": self.min_results,
            "required_text_results": self.required_text_results,
        }
        try:
            _atomic_create_json(manifest_path, expected)
        except FileExistsError:
            actual = _read_json_object(manifest_path)
            if actual != expected:
                raise LedgerIntegrityError(
                    "run ledger policy differs from the already frozen policy"
                )

    def _reserve(self, request: RetrievalRequest) -> dict[str, object]:
        identity = request.identity
        key = identity.request_key
        with self._run_lock():
            existing = self.reservation_path(key)
            if existing.exists():
                self._raise_replay(existing, key)
            reservations = self._validated_reservations()
            topic_reservations = sum(
                _identity_from_mapping(row.get("identity")).topic_id
                == identity.topic_id
                for row in reservations
            )
            if topic_reservations >= self.max_calls_per_topic:
                raise CallBudgetExceeded(
                    "external retrieval per-topic call ceiling reached "
                    f"({self.max_calls_per_topic}) for {identity.topic_id}; refusing "
                    "another reservation"
                )
            if len(reservations) >= self.max_calls:
                raise CallBudgetExceeded(
                    f"external retrieval call ceiling reached ({self.max_calls}); "
                    "refusing another reservation"
                )
            reservation: dict[str, object] = {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "request_key": key,
                "identity": identity.canonical_dict(),
                "query_text": request.query_text,
                "ordinal": len(reservations) + 1,
                "state": "reserved_pending",
                "reserved_at": _utc_now(),
            }
            # This path uses O_EXCL specifically: it is the durable call ticket.
            _exclusive_reservation_create(self.reservation_path(key), reservation)
            return reservation

    def _refuse_existing_incomplete_or_failed(
        self,
        request: RetrievalRequest,
    ) -> None:
        identity = request.identity
        key = identity.request_key
        with self._run_lock():
            path = self.reservation_path(key)
            if not path.exists():
                orphaned = [
                    artifact
                    for artifact in (
                        self.raw_path(key),
                        self.raw_metadata_path(key),
                        self.candidates_path(key),
                        self.outcome_path(key),
                    )
                    if artifact.exists()
                ]
                if orphaned:
                    raise LedgerIntegrityError(
                        "request artifacts exist without their reservation: "
                        + ", ".join(str(artifact) for artifact in orphaned)
                    )
                return
            reservation = self._load_reservation(key)
            stored_identity = _identity_from_mapping(reservation.get("identity"))
            if stored_identity != identity:
                raise LedgerIntegrityError("request reservation identity mismatch")
            if _validate_stored_query(
                reservation, stored_identity, "reservation"
            ) != request.query_text:
                raise LedgerIntegrityError("request reservation exact query mismatch")
            outcome_path = self.outcome_path(key)
            if not outcome_path.exists():
                raise ReplayRefused(
                    f"request {key} has a pending/crashed attempt; replay refused"
                )
            outcome = _read_json_object(outcome_path)
            self._validate_common_record(outcome, identity, reservation)
            if outcome.get("status") == "success":
                self._validate_success_artifacts(identity, reservation, outcome)
                raise ReplayRefused(
                    f"request {key} already completed; one planned invocation per key"
                )
            if outcome.get("status") == "failure":
                self._validate_failure_artifacts(identity, reservation, outcome)
                raise ReplayRefused(
                    f"request {key} already failed; one-attempt policy refuses replay"
                )
            raise LedgerIntegrityError("request outcome has an unsupported status")

    def _raise_replay(self, path: Path, key: str) -> None:
        # Parsing prevents a corrupt reservation from being treated as a normal replay.
        self._load_reservation(key)
        raise ReplayRefused(f"request {key} already has a reservation: {path}")

    def _validated_reservations(self) -> list[dict[str, object]]:
        if not self.attempts_dir.exists():
            return []
        rows: list[dict[str, object]] = []
        for path in sorted(self.attempts_dir.glob("*.reservation.json")):
            row = _read_json_object(path)
            _require_exact_fields(
                row,
                {
                    "schema_version",
                    "request_key",
                    "identity",
                    "query_text",
                    "ordinal",
                    "state",
                    "reserved_at",
                },
                "reservation",
            )
            key = _require_mapping_text(row, "request_key")
            _require_artifact_sha256(key, "reservation.request_key")
            if path.name != f"{key}.reservation.json":
                raise LedgerIntegrityError(f"reservation filename/key mismatch: {path}")
            identity = _identity_from_mapping(row.get("identity"))
            if identity.request_key != key:
                raise LedgerIntegrityError(f"reservation identity/key mismatch: {path}")
            _validate_stored_query(row, identity, "reservation")
            if row.get("schema_version") != LEDGER_SCHEMA_VERSION:
                raise LedgerIntegrityError(f"reservation schema mismatch: {path}")
            if row.get("state") != "reserved_pending":
                raise LedgerIntegrityError(f"reservation state mismatch: {path}")
            if not isinstance(row.get("reserved_at"), str) or not row["reserved_at"]:
                raise LedgerIntegrityError(f"reservation timestamp is invalid: {path}")
            ordinal = _strict_int(row.get("ordinal"), "reservation.ordinal")
            if ordinal <= 0:
                raise LedgerIntegrityError("reservation ordinal must be positive")
            rows.append(row)
        ordinals = sorted(_strict_int(row["ordinal"], "ordinal") for row in rows)
        if ordinals != list(range(1, len(rows) + 1)):
            raise LedgerIntegrityError("reservation ordinals are not contiguous and unique")
        return rows

    def _load_reservation(self, request_key: str) -> dict[str, object]:
        path = self.reservation_path(request_key)
        if not path.exists():
            raise LedgerIntegrityError(f"missing request reservation: {path}")
        row = _read_json_object(path)
        _require_exact_fields(
            row,
            {
                "schema_version",
                "request_key",
                "identity",
                "query_text",
                "ordinal",
                "state",
                "reserved_at",
            },
            "reservation",
        )
        if row.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise LedgerIntegrityError("reservation schema mismatch")
        if row.get("request_key") != request_key:
            raise LedgerIntegrityError("reservation request key mismatch")
        identity = _identity_from_mapping(row.get("identity"))
        if identity.request_key != request_key:
            raise LedgerIntegrityError("reservation identity hash mismatch")
        _validate_stored_query(row, identity, "reservation")
        if row.get("state") != "reserved_pending":
            raise LedgerIntegrityError("reservation state mismatch")
        if _strict_int(row.get("ordinal"), "reservation.ordinal") <= 0:
            raise LedgerIntegrityError("reservation ordinal must be positive")
        if not isinstance(row.get("reserved_at"), str) or not row["reserved_at"]:
            raise LedgerIntegrityError("reservation timestamp is invalid")
        return row

    def _validated_cache_hits(self) -> list[dict[str, object]]:
        if not self.cache_hits_dir.exists():
            return []
        rows: list[dict[str, object]] = []
        for path in sorted(self.cache_hits_dir.glob("*.json")):
            row = _read_json_object(path)
            _require_exact_fields(
                row,
                {
                    "schema_version",
                    "request_key",
                    "identity",
                    "query_text",
                    "status",
                    "recorded_at",
                    "candidate_count",
                    "response_sha256",
                    "candidates_sha256",
                    "external_calls",
                },
                "cache-hit evidence",
            )
            key = _require_mapping_text(row, "request_key")
            _require_artifact_sha256(key, "cache_hit.request_key")
            if path.name != f"{key}.json":
                raise LedgerIntegrityError(f"cache-hit filename/key mismatch: {path}")
            if row.get("schema_version") != LEDGER_SCHEMA_VERSION:
                raise LedgerIntegrityError(f"cache-hit schema mismatch: {path}")
            if row.get("status") != "cache_hit":
                raise LedgerIntegrityError(f"cache-hit status mismatch: {path}")
            if row.get("external_calls") != 0:
                raise LedgerIntegrityError(
                    f"cache-hit external_calls is not zero: {path}"
                )
            identity = _identity_from_mapping(row.get("identity"))
            if identity.request_key != key:
                raise LedgerIntegrityError(f"cache-hit identity/key mismatch: {path}")
            _validate_stored_query(row, identity, "cache-hit evidence")
            _require_artifact_sha256(
                row.get("response_sha256"), "cache_hit.response_sha256"
            )
            _require_artifact_sha256(
                row.get("candidates_sha256"), "cache_hit.candidates_sha256"
            )
            count = _strict_int(row.get("candidate_count"), "cache_hit.candidate_count")
            if count < self.min_results:
                raise LedgerIntegrityError("cache-hit candidate count is below minimum")
            rows.append(row)
        return rows

    def _validate_no_orphan_run_artifacts(self, reservation_keys: set[str]) -> None:
        artifact_keys: list[tuple[Path, str]] = []
        if self.raw_dir.exists():
            artifact_keys.extend(
                (path, path.name.removesuffix(".body"))
                for path in self.raw_dir.glob("*.body")
            )
            artifact_keys.extend(
                (path, path.name.removesuffix(".metadata.json"))
                for path in self.raw_dir.glob("*.metadata.json")
            )
        for directory in (self.candidates_dir, self.outcomes_dir):
            if directory.exists():
                artifact_keys.extend(
                    (path, path.name.removesuffix(".json"))
                    for path in directory.glob("*.json")
                )
        for path, key in artifact_keys:
            try:
                _require_artifact_sha256(key, "artifact request key")
            except LedgerIntegrityError as exc:
                raise LedgerIntegrityError(f"malformed run artifact filename: {path}") from exc
            if key not in reservation_keys:
                raise LedgerIntegrityError(
                    f"run artifact has no matching request reservation: {path}"
                )

    def _load_verified_raw(
        self,
        identity: RetrievalRequestIdentity,
    ) -> tuple[bytes, dict[str, object]]:
        key = identity.request_key
        body_path = self.raw_path(key)
        metadata_path = self.raw_metadata_path(key)
        if not body_path.is_file() or not metadata_path.is_file():
            raise LedgerIntegrityError(f"raw response artifact is incomplete: {key}")
        body = body_path.read_bytes()
        metadata = _read_json_object(metadata_path)
        _require_exact_fields(
            metadata,
            {
                "schema_version",
                "request_key",
                "identity",
                "ordinal",
                "http_status",
                "response_headers",
                "elapsed_seconds",
                "raw_body_bytes",
                "response_sha256",
                "capture_level",
            },
            "raw metadata",
        )
        if metadata.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise LedgerIntegrityError("raw metadata schema mismatch")
        if metadata.get("request_key") != key:
            raise LedgerIntegrityError("raw metadata request key mismatch")
        if _identity_from_mapping(metadata.get("identity")) != identity:
            raise LedgerIntegrityError("raw metadata identity mismatch")
        if metadata.get("raw_body_bytes") != len(body):
            raise LedgerIntegrityError("raw response length mismatch")
        if metadata.get("response_sha256") != _sha256(body):
            raise LedgerIntegrityError("raw response hash mismatch")
        _require_artifact_sha256(
            metadata.get("response_sha256"), "raw.response_sha256"
        )
        status = _strict_int(metadata.get("http_status"), "raw.http_status")
        if not 100 <= status <= 599:
            raise LedgerIntegrityError("raw HTTP status is outside 100..599")
        ordinal = _strict_int(metadata.get("ordinal"), "raw.ordinal")
        if ordinal <= 0:
            raise LedgerIntegrityError("raw ordinal must be positive")
        headers = metadata.get("response_headers")
        if not isinstance(headers, Mapping):
            raise LedgerIntegrityError("raw response headers are not an object")
        try:
            normalized_headers = _normalize_headers(headers)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise LedgerIntegrityError("raw response headers are invalid") from exc
        if dict(headers) != normalized_headers:
            raise LedgerIntegrityError("raw response headers are not canonical")
        elapsed = metadata.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise LedgerIntegrityError("raw elapsed_seconds is invalid")
        if metadata.get("capture_level") != (
            "exact_http_body_before_utf8_decode_or_json_parse"
        ):
            raise LedgerIntegrityError("raw response capture level is invalid")
        return body, metadata

    def _load_run_candidates(
        self,
        identity: RetrievalRequestIdentity,
    ) -> tuple[NormalizedCandidate, ...]:
        path = self.candidates_path(identity.request_key)
        if not path.is_file():
            raise LedgerIntegrityError(f"missing normalized candidate artifact: {path}")
        return _load_canonical_candidates(
            path.read_bytes(),
            min_results=self.min_results,
            required_text_results=self.required_text_results,
        )

    def _validate_success_artifacts(
        self,
        identity: RetrievalRequestIdentity,
        reservation: Mapping[str, object],
        outcome: Mapping[str, object],
    ) -> None:
        raw, metadata = self._load_verified_raw(identity)
        if metadata.get("ordinal") != reservation.get("ordinal"):
            raise LedgerIntegrityError("raw metadata/reservation ordinal mismatch")
        status = _strict_int(metadata.get("http_status"), "raw.http_status")
        if not 200 <= status <= 299:
            raise LedgerIntegrityError("successful attempt has non-success HTTP status")
        candidates = self._load_run_candidates(identity)
        if outcome.get("response_sha256") != _sha256(raw):
            raise LedgerIntegrityError("success outcome response hash mismatch")
        if outcome.get("raw_body_bytes") != len(raw):
            raise LedgerIntegrityError("success outcome raw byte length mismatch")
        candidate_bytes = _candidate_bytes(candidates)
        if outcome.get("candidates_sha256") != _sha256(candidate_bytes):
            raise LedgerIntegrityError("success outcome candidate hash mismatch")
        if outcome.get("candidate_count") != len(candidates):
            raise LedgerIntegrityError("success outcome candidate count mismatch")
        rebuilt = _decode_and_normalize(
            raw,
            min_results=self.min_results,
            required_text_results=self.required_text_results,
        )
        if rebuilt != candidates:
            raise LedgerIntegrityError(
                "run candidates do not reconstruct from the raw response"
            )

    def _validate_failure_artifacts(
        self,
        identity: RetrievalRequestIdentity,
        reservation: Mapping[str, object],
        outcome: Mapping[str, object],
    ) -> None:
        if self.candidates_path(identity.request_key).exists():
            raise LedgerIntegrityError(
                "failed attempt must not contain normalized candidates"
            )
        raw_exists = self.raw_path(identity.request_key).exists()
        metadata_exists = self.raw_metadata_path(identity.request_key).exists()
        if outcome.get("raw_present") is True:
            raw, metadata = self._load_verified_raw(identity)
            if metadata.get("ordinal") != reservation.get("ordinal"):
                raise LedgerIntegrityError("raw metadata/reservation ordinal mismatch")
            if outcome.get("response_sha256") != _sha256(raw):
                raise LedgerIntegrityError("failure outcome response hash mismatch")
        elif raw_exists or metadata_exists:
            raise LedgerIntegrityError("failure outcome/raw presence mismatch")

    def _validate_common_record(
        self,
        row: Mapping[str, object],
        identity: RetrievalRequestIdentity,
        reservation: Mapping[str, object],
    ) -> None:
        status = row.get("status")
        if status == "success":
            _require_exact_fields(
                row,
                {
                    "schema_version",
                    "request_key",
                    "identity",
                    "ordinal",
                    "status",
                    "raw_body_bytes",
                    "response_sha256",
                    "candidate_count",
                    "candidates_sha256",
                },
                "success outcome",
            )
        elif status == "failure":
            raw_present = row.get("raw_present")
            if not isinstance(raw_present, bool):
                raise LedgerIntegrityError("failure outcome raw_present must be boolean")
            expected = {
                "schema_version",
                "request_key",
                "identity",
                "ordinal",
                "status",
                "failure_type",
                "message",
                "raw_present",
            }
            if raw_present:
                expected.add("response_sha256")
            _require_exact_fields(row, expected, "failure outcome")
            if row.get("failure_type") not in {
                "transport_error",
                "response_validation_error",
                "artifact_commit_error",
            }:
                raise LedgerIntegrityError("failure outcome type is invalid")
            if not isinstance(row.get("message"), str) or not row["message"]:
                raise LedgerIntegrityError("failure outcome message is invalid")
        else:
            raise LedgerIntegrityError(f"unsupported outcome status: {status!r}")
        if row.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise LedgerIntegrityError("outcome schema mismatch")
        if row.get("request_key") != identity.request_key:
            raise LedgerIntegrityError("outcome request key mismatch")
        if _identity_from_mapping(row.get("identity")) != identity:
            raise LedgerIntegrityError("outcome identity mismatch")
        if row.get("ordinal") != reservation.get("ordinal"):
            raise LedgerIntegrityError("outcome/reservation ordinal mismatch")

    def _write_failure_outcome(
        self,
        *,
        identity: RetrievalRequestIdentity,
        ordinal: int,
        failure_type: str,
        message: str,
        raw_present: bool,
        response_sha256: str | None = None,
    ) -> None:
        outcome: dict[str, object] = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "request_key": identity.request_key,
            "identity": identity.canonical_dict(),
            "ordinal": ordinal,
            "status": "failure",
            "failure_type": failure_type,
            "message": message,
            "raw_present": raw_present,
        }
        if response_sha256 is not None:
            outcome["response_sha256"] = response_sha256
        _atomic_create_json(self.outcome_path(identity.request_key), outcome)

    def _cache_paths(self, request_key: str) -> tuple[Path, Path, Path]:
        if self.shared_cache_dir is None:
            raise AssertionError("shared cache is not configured")
        prefix = self.shared_cache_dir / request_key[:2]
        return (
            prefix / f"{request_key}.body",
            prefix / f"{request_key}.candidates.json",
            prefix / f"{request_key}.manifest.json",
        )

    def _load_cache(self, request: RetrievalRequest) -> RetrievalResult | None:
        if self.shared_cache_dir is None:
            return None
        identity = request.identity
        raw_path, candidate_path, manifest_path = self._cache_paths(identity.request_key)
        exists = (raw_path.exists(), candidate_path.exists(), manifest_path.exists())
        if not any(exists):
            return None
        if not all(exists):
            raise LedgerIntegrityError(
                f"shared cache entry is partial for {identity.request_key}"
            )
        manifest = _read_json_object(manifest_path)
        _require_exact_fields(
            manifest,
            {
                "schema_version",
                "request_key",
                "identity",
                "query_text",
                "http_status",
                "response_headers",
                "elapsed_seconds",
                "raw_body_bytes",
                "response_sha256",
                "candidate_count",
                "candidates_sha256",
            },
            "shared cache manifest",
        )
        if manifest.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise LedgerIntegrityError("shared cache schema mismatch")
        if manifest.get("request_key") != identity.request_key:
            raise LedgerIntegrityError("shared cache request key mismatch")
        if _identity_from_mapping(manifest.get("identity")) != identity:
            raise LedgerIntegrityError("shared cache exact identity mismatch")
        if _validate_stored_query(
            manifest, identity, "shared cache manifest"
        ) != request.query_text:
            raise LedgerIntegrityError("shared cache exact query mismatch")
        raw = raw_path.read_bytes()
        if manifest.get("raw_body_bytes") != len(raw):
            raise LedgerIntegrityError("shared cache raw byte length mismatch")
        response_sha256 = _sha256(raw)
        if manifest.get("response_sha256") != response_sha256:
            raise LedgerIntegrityError("shared cache response hash mismatch")
        candidate_bytes = candidate_path.read_bytes()
        candidates = _load_canonical_candidates(
            candidate_bytes,
            min_results=self.min_results,
            required_text_results=self.required_text_results,
        )
        candidates_sha256 = _sha256(candidate_bytes)
        if manifest.get("candidates_sha256") != candidates_sha256:
            raise LedgerIntegrityError("shared cache candidate hash mismatch")
        if manifest.get("candidate_count") != len(candidates):
            raise LedgerIntegrityError("shared cache candidate count mismatch")
        rebuilt = _decode_and_normalize(
            raw,
            min_results=self.min_results,
            required_text_results=self.required_text_results,
        )
        if rebuilt != candidates:
            raise LedgerIntegrityError(
                "shared cache candidates do not reconstruct from the raw response"
            )
        status = _strict_int(manifest.get("http_status"), "cache.http_status")
        if not 200 <= status <= 299:
            raise LedgerIntegrityError("shared cache contains a non-success response")
        headers = manifest.get("response_headers")
        if not isinstance(headers, Mapping):
            raise LedgerIntegrityError("shared cache response headers are not an object")
        try:
            normalized_headers = _normalize_headers(headers)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise LedgerIntegrityError("shared cache response headers are invalid") from exc
        if dict(headers) != normalized_headers:
            raise LedgerIntegrityError("shared cache response headers are not canonical")
        elapsed = manifest.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise LedgerIntegrityError("shared cache elapsed_seconds is invalid")
        return RetrievalResult(
            request_key=identity.request_key,
            candidates=candidates,
            response_sha256=response_sha256,
            candidates_sha256=candidates_sha256,
            cache_hit=True,
            external_calls=0,
        )

    def _commit_cache(
        self,
        *,
        request: RetrievalRequest,
        response: RawTransportResponse,
        candidates: tuple[NormalizedCandidate, ...],
        response_sha256: str,
        candidates_sha256: str,
    ) -> None:
        identity = request.identity
        raw_path, candidate_path, manifest_path = self._cache_paths(identity.request_key)
        if any(path.exists() for path in (raw_path, candidate_path, manifest_path)):
            # A concurrent complete writer is admissible only if all hashes verify.
            cached = self._load_cache(request)
            if cached is None:
                raise LedgerIntegrityError("shared cache disappeared during validation")
            if (
                cached.response_sha256 != response_sha256
                or cached.candidates_sha256 != candidates_sha256
            ):
                raise LedgerIntegrityError("shared cache conflicts with fresh response")
            return
        candidate_bytes = _candidate_bytes(candidates)
        _atomic_create_bytes(raw_path, response.body)
        _atomic_create_bytes(candidate_path, candidate_bytes)
        manifest = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "request_key": identity.request_key,
            "identity": identity.canonical_dict(),
            "query_text": request.query_text,
            "http_status": response.status,
            "response_headers": _normalize_headers(response.headers),
            "elapsed_seconds": float(response.elapsed_seconds),
            "raw_body_bytes": len(response.body),
            "response_sha256": response_sha256,
            "candidate_count": len(candidates),
            "candidates_sha256": candidates_sha256,
        }
        _atomic_create_json(manifest_path, manifest)

    def _record_cache_hit(
        self,
        request: RetrievalRequest,
        result: RetrievalResult,
    ) -> None:
        path = self.cache_hit_path(request.identity.request_key)
        evidence = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "request_key": request.identity.request_key,
            "identity": request.identity.canonical_dict(),
            "query_text": request.query_text,
            "status": "cache_hit",
            "recorded_at": _utc_now(),
            "candidate_count": len(result.candidates),
            "response_sha256": result.response_sha256,
            "candidates_sha256": result.candidates_sha256,
            "external_calls": 0,
        }
        try:
            _atomic_create_json(path, evidence)
        except FileExistsError:
            self._validate_cache_hit_evidence(request, result)
            raise ReplayRefused(
                f"request {request.identity.request_key} already has cache-hit evidence"
            )

    def _validate_cache_hit_evidence(
        self,
        request: RetrievalRequest,
        result: RetrievalResult,
    ) -> None:
        path = self.cache_hit_path(request.identity.request_key)
        existing = _read_json_object(path)
        expected_fields = {
            "schema_version",
            "request_key",
            "identity",
            "query_text",
            "status",
            "recorded_at",
            "candidate_count",
            "response_sha256",
            "candidates_sha256",
            "external_calls",
        }
        _require_exact_fields(existing, expected_fields, "cache-hit evidence")
        if existing.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise LedgerIntegrityError("cache-hit evidence schema mismatch")
        if existing.get("request_key") != request.identity.request_key:
            raise LedgerIntegrityError("cache-hit evidence request key mismatch")
        if _identity_from_mapping(existing.get("identity")) != request.identity:
            raise LedgerIntegrityError("cache-hit evidence identity mismatch")
        if _validate_stored_query(
            existing, request.identity, "cache-hit evidence"
        ) != request.query_text:
            raise LedgerIntegrityError("cache-hit evidence exact query mismatch")
        if existing.get("status") != "cache_hit":
            raise LedgerIntegrityError("cache-hit evidence status mismatch")
        if existing.get("external_calls") != 0:
            raise LedgerIntegrityError("cache-hit evidence must record zero external calls")
        if existing.get("candidate_count") != len(result.candidates):
            raise LedgerIntegrityError("cache-hit evidence candidate count mismatch")
        if existing.get("response_sha256") != result.response_sha256:
            raise LedgerIntegrityError("cache-hit evidence response hash mismatch")
        if existing.get("candidates_sha256") != result.candidates_sha256:
            raise LedgerIntegrityError("cache-hit evidence candidate hash mismatch")
        if not isinstance(existing.get("recorded_at"), str) or not existing["recorded_at"]:
            raise LedgerIntegrityError("cache-hit evidence lacks recorded_at")

    class _Lock:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.descriptor: int | None = None

        def __enter__(self) -> None:
            _ensure_directory(self.path.parent)
            self.descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self.descriptor, fcntl.LOCK_EX)

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            assert self.descriptor is not None
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)

    def _run_lock(self) -> "RetrievalLedger._Lock":
        return self._Lock(self.run_dir / ".ledger.lock")

    def _shared_cache_lock(self, request_key: str) -> "RetrievalLedger._Lock":
        if self.shared_cache_dir is None:
            raise AssertionError("shared cache is not configured")
        _require_sha256(request_key, "request_key")
        return self._Lock(
            self.shared_cache_dir / ".locks" / f"{request_key}.lock"
        )


def canonical_request_key(identity: RetrievalRequestIdentity) -> str:
    """Public helper for the full SHA-256 canonical request key."""

    return identity.request_key


def normalize_response_candidates(
    response: Mapping[str, Any],
    *,
    min_results: int = DEFAULT_MIN_RESULTS,
    required_text_results: int = DEFAULT_REQUIRED_TEXT_RESULTS,
) -> tuple[NormalizedCandidate, ...]:
    """Normalize and mechanically validate a decoded retrieval response."""

    if (
        isinstance(min_results, bool)
        or not isinstance(min_results, int)
        or not 1 <= min_results <= MAX_RESULTS
    ):
        raise ValueError(f"min_results must be between 1 and {MAX_RESULTS}")
    if (
        isinstance(required_text_results, bool)
        or not isinstance(required_text_results, int)
        or required_text_results < DEFAULT_REQUIRED_TEXT_RESULTS
        or required_text_results > MAX_RESULTS
    ):
        raise ValueError(
            "required_text_results cannot weaken the mandatory top-50 text check"
        )

    rows: object | None = None
    for name in ("candidates", "hits", "results"):
        if name in response:
            rows = response[name]
            break
    if not isinstance(rows, list):
        raise ResponseValidationError(
            "response must contain a candidates, hits, or results list"
        )
    if len(rows) < min_results:
        raise ResponseValidationError(
            f"response has {len(rows)} candidates; at least {min_results} required"
        )
    if len(rows) > MAX_RESULTS:
        raise ResponseValidationError(
            f"response has {len(rows)} candidates; frozen depth is at most {MAX_RESULTS}"
        )

    normalized: list[NormalizedCandidate] = []
    seen_docids: set[str] = set()
    seen_ranks: set[int] = set()
    for position, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise ResponseValidationError(f"candidate {position} is not an object")
        raw_docid = raw_row.get("docid") or raw_row.get("id") or raw_row.get("_id")
        if not isinstance(raw_docid, str) or not raw_docid.strip():
            raise ResponseValidationError(f"candidate {position} has no nonempty docid")
        docid = raw_docid.strip()
        if docid in seen_docids:
            raise ResponseValidationError(f"duplicate candidate docid: {docid}")
        raw_rank = raw_row.get("rank", position)
        if isinstance(raw_rank, bool) or not isinstance(raw_rank, int):
            raise ResponseValidationError(f"candidate {position} rank is not an integer")
        rank = raw_rank
        if rank <= 0:
            raise ResponseValidationError(f"candidate {position} rank is not positive")
        if rank != position:
            raise ResponseValidationError(
                f"candidate {position} rank must equal its one-based position"
            )
        if rank in seen_ranks:
            raise ResponseValidationError(f"duplicate candidate rank: {rank}")
        raw_score = raw_row.get("score", 0.0)
        if raw_score is None:
            raw_score = 0.0
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ResponseValidationError(f"candidate {position} score is not numeric")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ResponseValidationError(f"candidate {position} score is not finite")
        document = raw_row.get("doc") or raw_row.get("contents") or raw_row
        text = _extract_text(document)
        if (
            position <= required_text_results or rank <= required_text_results
        ) and not text:
            raise ResponseValidationError(
                f"candidate {position} (rank {rank}) has empty text in the required top "
                f"{required_text_results}"
            )
        normalized.append(
            NormalizedCandidate(docid=docid, rank=rank, score=score, text=text)
        )
        seen_docids.add(docid)
        seen_ranks.add(rank)
    return tuple(normalized)


def _decode_and_normalize(
    raw: bytes,
    *,
    min_results: int,
    required_text_results: int,
) -> tuple[NormalizedCandidate, ...]:
    try:
        decoded_text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResponseValidationError("response body is not valid UTF-8") from exc
    try:
        decoded = json.loads(decoded_text)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError("response body is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ResponseValidationError("response JSON must be an object")
    return normalize_response_candidates(
        decoded,
        min_results=min_results,
        required_text_results=required_text_results,
    )


def _load_canonical_candidates(
    content: bytes,
    *,
    min_results: int,
    required_text_results: int,
) -> tuple[NormalizedCandidate, ...]:
    try:
        decoded = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerIntegrityError("candidate artifact is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, list):
        raise LedgerIntegrityError("candidate artifact must be a JSON list")
    candidates: list[NormalizedCandidate] = []
    for position, row in enumerate(decoded, start=1):
        if not isinstance(row, Mapping) or set(row) != {"docid", "rank", "score", "text"}:
            raise LedgerIntegrityError(f"candidate artifact row {position} has invalid fields")
        try:
            candidate = NormalizedCandidate(
                docid=str(row["docid"]),
                rank=_strict_int(row["rank"], "candidate rank"),
                score=float(row["score"]),
                text=str(row["text"]),
            )
        except (TypeError, ValueError) as exc:
            raise LedgerIntegrityError(
                f"candidate artifact row {position} has invalid values"
            ) from exc
        candidates.append(candidate)
    result = tuple(candidates)
    if _candidate_bytes(result) != content:
        raise LedgerIntegrityError("candidate artifact is not canonical JSON")
    # Reuse the public validator to enforce count, identity, ranks, and top text.
    normalized_again = normalize_response_candidates(
        {"candidates": [asdict(candidate) for candidate in result]},
        min_results=min_results,
        required_text_results=required_text_results,
    )
    if normalized_again != result:
        raise LedgerIntegrityError("candidate artifact normalization is unstable")
    return result


def _candidate_bytes(candidates: Sequence[NormalizedCandidate]) -> bytes:
    return _canonical_bytes([asdict(candidate) for candidate in candidates])


def _extract_text(value: object) -> str:
    if isinstance(value, str):
        try:
            nested = json.loads(value)
        except json.JSONDecodeError:
            return " ".join(value.split())
        return _extract_text(nested)
    if isinstance(value, Mapping):
        for key in ("contents", "text", "body", "passage", "abstract"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return " ".join(candidate.split())
        return " ".join(
            text
            for text in (
                _extract_text(item)
                for item in value.values()
                if isinstance(item, (Mapping, list))
            )
            if text
        )
    if isinstance(value, list):
        return " ".join(text for text in (_extract_text(item) for item in value) if text)
    return ""


def _identity_from_mapping(value: object) -> RetrievalRequestIdentity:
    if not isinstance(value, Mapping):
        raise LedgerIntegrityError("artifact identity is not an object")
    expected = {
        "topic_id",
        "variant_name",
        "retriever_version",
        "query_sha256",
        "index_url",
        "index_id",
        "hits",
        "analyzer_fingerprint_sha256",
    }
    bm25_fields = {"bm25_k1", "bm25_b"}
    if set(value) not in (expected, expected | bm25_fields):
        raise LedgerIntegrityError("artifact identity fields are not exact")
    try:
        return RetrievalRequestIdentity(
            topic_id=value["topic_id"],  # type: ignore[arg-type]
            variant_name=value["variant_name"],  # type: ignore[arg-type]
            retriever_version=value["retriever_version"],  # type: ignore[arg-type]
            query_sha256=value["query_sha256"],  # type: ignore[arg-type]
            index_url=value["index_url"],  # type: ignore[arg-type]
            index_id=value["index_id"],  # type: ignore[arg-type]
            hits=_strict_int(value["hits"], "identity.hits"),
            analyzer_fingerprint_sha256=value["analyzer_fingerprint_sha256"],  # type: ignore[arg-type]
            bm25_k1=value.get("bm25_k1"),  # type: ignore[arg-type]
            bm25_b=value.get("bm25_b"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise LedgerIntegrityError("artifact identity is invalid") from exc


def _validate_stored_query(
    value: Mapping[str, object],
    identity: RetrievalRequestIdentity,
    artifact_name: str,
) -> str:
    query_text = value.get("query_text")
    if not isinstance(query_text, str):
        raise LedgerIntegrityError(f"{artifact_name} lacks the exact query text")
    if _sha256(query_text.encode("utf-8")) != identity.query_sha256:
        raise LedgerIntegrityError(f"{artifact_name} query text/hash mismatch")
    return query_text


def _require_exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    artifact_name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise LedgerIntegrityError(
            f"{artifact_name} fields are not exact; missing={missing}, extra={extra}"
        )


def _require_artifact_sha256(value: object, name: str) -> None:
    try:
        _require_sha256(value, name)
    except ValueError as exc:
        raise LedgerIntegrityError(str(exc)) from exc


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LedgerIntegrityError("value cannot be serialized as canonical JSON") from exc


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerIntegrityError(f"{name} must be an integer")
    return value


def _require_mapping_text(value: Mapping[str, object], name: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result:
        raise LedgerIntegrityError(f"{name} must be a nonempty string")
    return result


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(headers, Mapping):
        raise TypeError("transport headers must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("transport header names and values must be strings")
        normalized[key.lower()] = value
    return dict(sorted(normalized.items()))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerIntegrityError(f"cannot read valid JSON artifact: {path}") from exc
    if not isinstance(decoded, dict):
        raise LedgerIntegrityError(f"JSON artifact is not an object: {path}")
    return decoded


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    """Create each missing directory and fsync its parent entry."""

    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if cursor.exists() and not cursor.is_dir():
        raise NotADirectoryError(cursor)
    for directory in reversed(missing):
        try:
            os.mkdir(directory)
        except FileExistsError:
            if not directory.is_dir():
                raise
        _fsync_directory(directory.parent)


def _atomic_create_bytes(path: Path, content: bytes) -> None:
    """Fsync content, publish with create-only hard link, then fsync directory."""

    _ensure_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary_name, path)
        os.unlink(temporary_name)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_create_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_create_bytes(path, _canonical_bytes(dict(value)))


def _exclusive_reservation_create(path: Path, value: Mapping[str, object]) -> None:
    """Durably claim ``path`` with O_EXCL before publishing complete metadata."""

    _ensure_directory(path.parent)
    content = _canonical_bytes(dict(value))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
        claim = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(claim)
        # Only this process can own the just-created final path.
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _exception_message(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return message[:1000] or type(exc).__name__
