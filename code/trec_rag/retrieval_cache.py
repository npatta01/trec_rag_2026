"""Immutable, raw-authoritative organizer retrieval cache v2.

The cache deliberately keeps transport identity, local derivation identity,
and document-content identity separate.  A complete manifest is the only
commit marker that makes an on-disk entry readable.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Callable, Mapping
import uuid

from filelock import FileLock

from .document_store import DocumentStore, DocumentStoreIntegrityError


RETRIEVAL_CACHE_SCHEMA_VERSION = "organizer-retrieval-cache-v2"
RETRIEVAL_TEXT_NORMALIZER_VERSION = "organizer-exact-doc-string-v1"
RETRIEVAL_SCORE_NORMALIZER_VERSION = "whitespace-score-v1"
_DIGEST_LENGTH = 64
_LEGACY_CACHE_KEY = re.compile(r"[0-9a-f]{16}\Z")
_LEGACY_SIDECAR_FIELDS = {
    "cache_key",
    "hits",
    "index",
    "index_url",
    "query",
    "rate_policy",
    "response_sha256",
    "retriever_name",
    "retriever_type",
    "topic_id",
    "variant_name",
}


class RetrievalCacheError(RuntimeError):
    """Base class for retrieval-cache failures."""


class RetrievalCacheIntegrityError(RetrievalCacheError):
    """A cache entry, organizer record, or document object is invalid."""


class RetrievalCacheConflictError(RetrievalCacheError):
    """Immutable state already exists with contradictory content."""


class RetrievalCacheMiss(RetrievalCacheError):
    """An offline lookup has no complete cache entry."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != _DIGEST_LENGTH:
        raise RetrievalCacheIntegrityError(f"{field} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RetrievalCacheIntegrityError(
            f"{field} must be a lowercase SHA-256 digest"
        ) from exc
    if value != value.lower():
        raise RetrievalCacheIntegrityError(f"{field} must be lowercase")
    return value


def _canonical_value(value: object, field: str = "value") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"{field} mapping keys must be strings")
            result[key] = _canonical_value(value[key], f"{field}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item, field) for item in value]
    raise TypeError(f"{field} contains a non-JSON value: {type(value).__name__}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _exact_fields(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RetrievalCacheIntegrityError(f"{label} must be an object")
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise RetrievalCacheIntegrityError(
            f"{label} fields mismatch; missing={missing}, unknown={unknown}"
        )
    return value


def _strict_int(value: object, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RetrievalCacheIntegrityError(f"{field} must be an integer")
    if positive and value <= 0:
        raise RetrievalCacheIntegrityError(f"{field} must be positive")
    return value


def _strict_score(value: object, field: str = "score") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalCacheIntegrityError(f"{field} must be a number")
    score = float(value)
    if not math.isfinite(score):
        raise RetrievalCacheIntegrityError(f"{field} must be finite")
    return score


def _write_durable(path: Path, content: bytes) -> None:
    _private_mkdir(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as sink:
            os.fchmod(sink.fileno(), 0o600)
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _private_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class TransportIdentity:
    """The full identity of a remote organizer result."""

    def __init__(
        self,
        query_sha256: str | None = None,
        index_id: str | None = None,
        endpoint_identity: str | None = None,
        corpus_epoch: str | None = None,
        hits: int | None = None,
        remote_parameters: Mapping[str, object] | None = None,
        *,
        query_text: str | None = None,
        index: str | None = None,
        endpoint: str | None = None,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        if query_text is not None:
            if not isinstance(query_text, str):
                raise TypeError("query_text must be a string")
            derived_query_sha256 = _sha256(query_text.encode("utf-8"))
            if query_sha256 is not None and query_sha256 != derived_query_sha256:
                raise ValueError("query_sha256 does not match query_text")
            query_sha256 = derived_query_sha256
        if index_id is None:
            index_id = index
        if endpoint_identity is None:
            endpoint_identity = endpoint
        if remote_parameters is None:
            remote_parameters = parameters
        elif parameters is not None and dict(remote_parameters) != dict(parameters):
            raise ValueError("remote_parameters and parameters disagree")
        if not isinstance(query_sha256, str):
            raise ValueError("query_sha256 is required")
        _digest(query_sha256, "query_sha256")
        for value, field in (
            (index_id, "index_id"),
            (endpoint_identity, "endpoint_identity"),
            (corpus_epoch, "corpus_epoch"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
            if field in {"index_id", "corpus_epoch"} and value.strip().lower() in {
                "unknown",
                "unknown-index",
                "unspecified",
            }:
                raise ValueError(f"{field} must be explicit; synthetic values are forbidden")
        if isinstance(hits, bool) or not isinstance(hits, int) or hits <= 0:
            raise ValueError("hits must be a positive integer")
        canonical_parameters = _canonical_value(
            remote_parameters or {}, "remote_parameters"
        )
        if not isinstance(canonical_parameters, dict):
            raise TypeError("remote_parameters must be a mapping")
        self.query_sha256 = query_sha256
        self.index_id = index_id
        self.endpoint_identity = endpoint_identity
        self.corpus_epoch = corpus_epoch
        self.hits = hits
        self.remote_parameters = canonical_parameters

    @classmethod
    def from_query(
        cls,
        *,
        query_text: str,
        index_id: str,
        endpoint_identity: str,
        corpus_epoch: str,
        hits: int,
        remote_parameters: Mapping[str, object] | None = None,
    ) -> "TransportIdentity":
        return cls(
            index_id=index_id,
            endpoint_identity=endpoint_identity,
            corpus_epoch=corpus_epoch,
            hits=hits,
            remote_parameters=remote_parameters,
            query_text=query_text,
        )

    @property
    def index(self) -> str:
        return self.index_id

    @property
    def endpoint(self) -> str:
        return self.endpoint_identity

    @property
    def parameters(self) -> dict[str, object]:
        return dict(self.remote_parameters)

    def canonical_dict(self) -> dict[str, object]:
        return {
            "corpus_epoch": self.corpus_epoch,
            "endpoint_identity": self.endpoint_identity,
            "hits": self.hits,
            "index_id": self.index_id,
            "query_sha256": self.query_sha256,
            "remote_parameters": self.remote_parameters,
        }

    @property
    def request_key(self) -> str:
        return _sha256(_canonical_bytes(self.canonical_dict()))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TransportIdentity) and self.canonical_dict() == other.canonical_dict()

    def __hash__(self) -> int:
        return hash(self.request_key)


class DerivationIdentity:
    """The versioned local parser, field extractor, and scoring normalizer."""

    def __init__(
        self,
        parser_version: str = "organizer-json-v1",
        extractor_version: str = RETRIEVAL_TEXT_NORMALIZER_VERSION,
        field_path: str | tuple[str, ...] = "text",
        scoring_normalizer_version: str = RETRIEVAL_SCORE_NORMALIZER_VERSION,
        *,
        normalizer_version: str | None = None,
        text_normalizer_version: str | None = None,
    ) -> None:
        if normalizer_version is not None:
            extractor_version = normalizer_version
        if text_normalizer_version is not None:
            extractor_version = text_normalizer_version
        if isinstance(field_path, str):
            field_path = tuple(part for part in field_path.split(".") if part)
        if not field_path or any(not isinstance(part, str) or not part for part in field_path):
            raise ValueError("field_path must contain non-empty field names")
        for value, field in (
            (parser_version, "parser_version"),
            (extractor_version, "extractor_version"),
            (scoring_normalizer_version, "scoring_normalizer_version"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        self.parser_version = parser_version
        self.extractor_version = extractor_version
        self.field_path = tuple(field_path)
        self.scoring_normalizer_version = scoring_normalizer_version

    @classmethod
    def from_normalizer(cls, normalizer: object) -> "DerivationIdentity":
        return cls(
            parser_version=str(getattr(normalizer, "parser_version", "organizer-json-v1")),
            extractor_version=str(
                getattr(normalizer, "version", RETRIEVAL_TEXT_NORMALIZER_VERSION)
            ),
            field_path=getattr(normalizer, "field_path", ("text",)),
            scoring_normalizer_version=str(
                getattr(normalizer, "scoring_version", RETRIEVAL_SCORE_NORMALIZER_VERSION)
            ),
        )

    def canonical_dict(self) -> dict[str, object]:
        return {
            "extractor_version": self.extractor_version,
            "field_path": list(self.field_path),
            "parser_version": self.parser_version,
            "scoring_normalizer_version": self.scoring_normalizer_version,
        }

    @property
    def derivation_key(self) -> str:
        return _sha256(_canonical_bytes(self.canonical_dict()))

    @property
    def normalizer_version(self) -> str:
        return self.extractor_version

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DerivationIdentity) and self.canonical_dict() == other.canonical_dict()

    def __hash__(self) -> int:
        return hash(self.derivation_key)


@dataclass(frozen=True)
class CachedHit:
    rank: int
    score: float
    docid: str
    content_sha256: str

    @property
    def content_hash(self) -> str:
        return self.content_sha256


@dataclass(frozen=True)
class CachedRetrieval:
    transport_identity: TransportIdentity
    derivation_identity: DerivationIdentity
    query_text: str
    hits: tuple[CachedHit, ...]
    raw_response: bytes
    raw_sha256: str

    @property
    def request_key(self) -> str:
        return self.transport_identity.request_key

    @property
    def derivation_key(self) -> str:
        return self.derivation_identity.derivation_key

    @property
    def response_sha256(self) -> str:
        return self.raw_sha256

    @property
    def raw(self) -> bytes:
        return self.raw_response

    @property
    def hit_refs(self) -> tuple[CachedHit, ...]:
        return self.hits


@dataclass(frozen=True)
class LegacyPromotionReceipt:
    request_key: str
    derivation_key: str
    source_sha256: str
    raw_sha256: str
    receipt_path: Path
    receipt_sha256: str = ""


@dataclass(frozen=True)
class _ParsedHit:
    rank: int
    score: float
    docid: str
    text: str


class OrganizerTextNormalizer:
    """Parse the organizer's versioned exact-document-string contract."""

    parser_version = "organizer-response-v2"
    scoring_version = RETRIEVAL_SCORE_NORMALIZER_VERSION

    def __init__(
        self,
        *,
        version: str = RETRIEVAL_TEXT_NORMALIZER_VERSION,
        field: str | tuple[str, ...] = "text",
        field_path: str | tuple[str, ...] | None = None,
        parser_version: str | None = None,
        scoring_version: str | None = None,
    ) -> None:
        self.version = version
        selected = field_path if field_path is not None else field
        if selected not in ("text", ("text",)):
            raise ValueError(
                "organizer v2 uses candidate.doc as an exact plain-text string; "
                "object-field selectors require a distinct legacy normalizer"
            )
        self.field_path = ("doc",)
        if parser_version is not None:
            self.parser_version = parser_version
        if scoring_version is not None:
            self.scoring_version = scoring_version

    def parse(self, raw_response: bytes | Mapping[str, object]) -> tuple[_ParsedHit, ...]:
        if isinstance(raw_response, bytes):
            try:
                payload = json.loads(raw_response.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RetrievalCacheIntegrityError("organizer response is not valid UTF-8 JSON") from exc
        elif isinstance(raw_response, Mapping):
            payload = raw_response
        else:
            raise TypeError("raw_response must be bytes or a mapping")
        if not isinstance(payload, Mapping):
            raise RetrievalCacheIntegrityError("organizer response must be a JSON object")
        payload = _exact_fields(
            payload,
            {"api", "candidates", "index", "query"},
            "organizer response",
        )
        if payload["api"] != "v1":
            raise RetrievalCacheIntegrityError("organizer response api must be v1")
        if not isinstance(payload["index"], str) or not payload["index"].strip():
            raise RetrievalCacheIntegrityError("organizer response index must be non-empty text")
        query = _exact_fields(payload["query"], {"text"}, "organizer response query")
        if not isinstance(query["text"], str):
            raise RetrievalCacheIntegrityError(
                "organizer response query.text must be exact text"
            )
        rows = payload["candidates"]
        if not isinstance(rows, list):
            raise RetrievalCacheIntegrityError("organizer response candidates must be a list")
        parsed: list[_ParsedHit] = []
        ranks: set[int] = set()
        doc_bodies: dict[str, str] = {}
        for position, row in enumerate(rows, start=1):
            row = _exact_fields(
                row,
                {"doc", "docid", "rank", "score"},
                f"candidate {position}",
            )
            rank = _strict_int(row["rank"], f"candidate {position}.rank", positive=True)
            if rank in ranks:
                raise RetrievalCacheIntegrityError(f"duplicate candidate rank {rank}")
            ranks.add(rank)
            score = _strict_score(row["score"], f"candidate {position}.score")
            docid = row["docid"]
            if not isinstance(docid, str) or not docid.strip():
                raise RetrievalCacheIntegrityError(f"candidate {position}.docid must be non-empty text")
            selected = row["doc"]
            if not isinstance(selected, str):
                raise RetrievalCacheIntegrityError(
                    f"candidate {position}.doc must be exact plain text"
                )
            previous = doc_bodies.get(docid)
            if previous is not None and previous != selected:
                raise RetrievalCacheIntegrityError(
                    f"docid {docid!r} has conflicting bodies"
                )
            doc_bodies[docid] = selected
            parsed.append(_ParsedHit(rank, score, docid, selected))
        return tuple(parsed)


class RetrievalCache:
    """Crash-safe immutable transport and derivation cache."""

    def __init__(
        self,
        root: Path,
        document_store: DocumentStore,
        normalizer: OrganizerTextNormalizer | object | None = None,
        *,
        publication_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.v2_root = self.root if self.root.name == "v2" else self.root / "v2"
        self.document_store = document_store
        self.normalizer = normalizer or OrganizerTextNormalizer()
        if not hasattr(self.normalizer, "parse"):
            raise TypeError("normalizer must expose parse(raw_response)")
        self._bound_derivation_identity = DerivationIdentity.from_normalizer(
            self.normalizer
        )
        self.publication_hook = publication_hook

    def _bind_derivation_identity(
        self,
        derivation_identity: DerivationIdentity,
    ) -> None:
        if not isinstance(derivation_identity, DerivationIdentity):
            raise TypeError("derivation_identity must be a DerivationIdentity")
        if derivation_identity != self._bound_derivation_identity:
            raise RetrievalCacheIntegrityError(
                "derivation identity does not match the cache normalizer"
            )

    def lookup(
        self,
        transport_identity: TransportIdentity,
        derivation_identity: DerivationIdentity,
        query_text: str,
        *,
        offline: bool = False,
    ) -> CachedRetrieval | None:
        self._bind_derivation_identity(derivation_identity)
        self._validate_query(transport_identity, query_text)
        _private_mkdir(self.v2_root)
        with FileLock(str(self._lock_path(transport_identity.request_key))):
            result = self._lookup_locked(transport_identity, derivation_identity, query_text)
        if result is None and offline:
            raise RetrievalCacheMiss(
                f"offline cache miss for request {transport_identity.request_key}"
            )
        return result

    def commit(
        self,
        transport_identity: TransportIdentity,
        derivation_identity: DerivationIdentity,
        query_text: str,
        raw_response: bytes | bytearray | Mapping[str, object] | object,
    ) -> CachedRetrieval:
        self._bind_derivation_identity(derivation_identity)
        self._validate_query(transport_identity, query_text)
        raw = self._raw_bytes(raw_response)
        self._validate_response_identity(raw, transport_identity, query_text)
        parsed = self._parse(raw)
        _private_mkdir(self.v2_root)
        request_key = transport_identity.request_key
        with FileLock(str(self._lock_path(request_key))):
            entry = self._entry_path(request_key)
            if entry.exists():
                existing_raw = self._load_transport(entry, transport_identity, query_text)
                if existing_raw != raw:
                    self._record_conflict(request_key, raw, "raw-authority")
                    raise RetrievalCacheConflictError(
                        f"raw authority conflict for request {request_key}"
                    )
                existing = self._load_or_publish_derivation_locked(
                    entry, transport_identity, derivation_identity, query_text, existing_raw
                )
                return existing

            attempt_raw = self._recover_attempt_locked(
                transport_identity, query_text
            )
            if attempt_raw is not None:
                if attempt_raw != raw:
                    self._record_conflict(request_key, raw, "attempt-raw-authority")
                    raise RetrievalCacheConflictError(
                        f"complete attempts conflict for request {request_key}"
                    )
                raw = attempt_raw
            else:
                self._write_attempt_locked(transport_identity, query_text, raw)
            hits = self._admit_hits(parsed)
            self._publish_transport_locked(transport_identity, raw)
            entry = self._entry_path(request_key)
            return self._publish_derivation_locked(
                entry,
                transport_identity,
                derivation_identity,
                query_text,
                raw,
                hits,
            )

    def promote_legacy(
        self,
        legacy_path: Path,
        transport_identity: TransportIdentity,
        derivation_identity: DerivationIdentity,
        query_text: str,
        *,
        sidecar_path: Path | None = None,
        operator_attestation: Mapping[str, object] | None = None,
    ) -> LegacyPromotionReceipt:
        """Promote a legacy response only with an exact sidecar and epoch attestation."""
        self._bind_derivation_identity(derivation_identity)
        source = Path(legacy_path)
        raw = source.read_bytes()
        sidecar = Path(sidecar_path) if sidecar_path is not None else source.with_suffix(".meta.json")
        try:
            sidecar_bytes = sidecar.read_bytes()
            sidecar_value = json.loads(sidecar_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetrievalCacheIntegrityError("legacy response requires a valid sidecar") from exc
        sidecar_value = _exact_fields(
            sidecar_value,
            _LEGACY_SIDECAR_FIELDS,
            "legacy sidecar",
        )
        legacy_key = sidecar_value["cache_key"]
        if not isinstance(legacy_key, str) or _LEGACY_CACHE_KEY.fullmatch(legacy_key) is None:
            raise RetrievalCacheIntegrityError(
                "legacy sidecar cache_key must be a lowercase 16-character digest"
            )
        for field in (
            "index",
            "index_url",
            "query",
            "retriever_name",
            "retriever_type",
            "topic_id",
            "variant_name",
        ):
            if not isinstance(sidecar_value[field], str) or not sidecar_value[field]:
                raise RetrievalCacheIntegrityError(
                    f"legacy sidecar {field} must be non-empty text"
                )
        hits = _strict_int(sidecar_value["hits"], "legacy sidecar hits", positive=True)
        response_sha256 = _digest(
            sidecar_value["response_sha256"],
            "legacy sidecar response_sha256",
        )
        rate_policy = _exact_fields(
            sidecar_value["rate_policy"],
            {"burst", "min_interval_seconds", "per_host"},
            "legacy sidecar rate_policy",
        )
        _strict_int(
            rate_policy["burst"],
            "legacy sidecar rate_policy.burst",
            positive=True,
        )
        minimum_interval = _strict_score(
            rate_policy["min_interval_seconds"],
            "legacy sidecar rate_policy.min_interval_seconds",
        )
        if minimum_interval < 0:
            raise RetrievalCacheIntegrityError(
                "legacy sidecar rate_policy.min_interval_seconds must be non-negative"
            )
        if rate_policy["per_host"] is not True:
            raise RetrievalCacheIntegrityError(
                "legacy sidecar rate_policy.per_host must be true"
            )
        expected_legacy_key = _historical_retrieval_cache_key(sidecar_value)
        if legacy_key != expected_legacy_key:
            raise RetrievalCacheIntegrityError(
                "legacy sidecar cache_key does not match its historical request identity"
            )
        if not source.name.endswith(f"__{legacy_key}.json"):
            raise RetrievalCacheIntegrityError(
                "legacy response filename does not end with its historical cache key"
            )
        if sidecar_value["query"] != query_text:
            raise RetrievalCacheIntegrityError("legacy sidecar query mismatch")
        if response_sha256 != _sha256(raw):
            raise RetrievalCacheIntegrityError("legacy sidecar response hash mismatch")
        if not isinstance(operator_attestation, Mapping):
            raise RetrievalCacheIntegrityError(
                "legacy promotion requires explicit operator epoch attestation"
            )
        attestation = _exact_fields(
            operator_attestation,
            {"operator", "corpus_epoch"},
            "operator attestation",
        )
        if (
            not isinstance(attestation["operator"], str)
            or not attestation["operator"].strip()
            or not isinstance(attestation["corpus_epoch"], str)
        ):
            raise RetrievalCacheIntegrityError(
                "legacy promotion requires an explicit operator corpus epoch"
            )
        promoted_identity = TransportIdentity.from_query(
            query_text=query_text,
            index_id=str(sidecar_value["index"]),
            endpoint_identity=str(sidecar_value["index_url"]),
            corpus_epoch=str(attestation["corpus_epoch"]),
            hits=hits,
        )
        if promoted_identity != transport_identity:
            raise RetrievalCacheIntegrityError(
                "operator-attested v2 transport identity does not match the archive request"
            )
        parsed = self._parse(raw)
        self._validate_response_identity(raw, promoted_identity, query_text)
        if len(parsed) != hits:
            raise RetrievalCacheIntegrityError(
                "legacy response candidate count does not match sidecar hits"
            )
        result = self.commit(
            promoted_identity,
            derivation_identity,
            query_text,
            raw,
        )
        receipt_dir = self.v2_root / "legacy-promotions"
        _private_mkdir(receipt_dir)
        source_sha256 = _sha256(raw)
        receipt_path = receipt_dir / (
            f"{source_sha256}-{result.request_key}-{result.derivation_key}.json"
        )
        receipt_payload = {
            "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "source_path": str(source),
            "sidecar_path": str(sidecar),
            "sidecar_sha256": _sha256(sidecar_bytes),
            "legacy_cache_key": legacy_key,
            "request_key": result.request_key,
            "derivation_key": result.derivation_key,
            "raw_sha256": result.raw_sha256,
            "operator_attestation": dict(attestation),
        }
        receipt_digest = _sha256(_json_bytes(receipt_payload))
        receipt = {**receipt_payload, "receipt_sha256": receipt_digest}
        receipt_bytes = _json_bytes(receipt)
        if receipt_path.exists():
            existing = receipt_path.read_bytes()
            try:
                existing_value = json.loads(existing)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RetrievalCacheIntegrityError("legacy promotion receipt is corrupt") from exc
            if existing_value != receipt:
                raise RetrievalCacheConflictError("legacy promotion receipt conflict")
        else:
            _write_create_only(receipt_path, receipt_bytes)
        return LegacyPromotionReceipt(
            result.request_key,
            result.derivation_key,
            source_sha256,
            result.raw_sha256,
            receipt_path,
            receipt_digest,
        )

    import_legacy = promote_legacy

    def seal_attempt(
        self,
        attempt_path: Path,
        transport_identity: TransportIdentity,
        query_text: str,
        raw_response: bytes,
    ) -> Path:
        """Seal the already-reserved attempt; never mint a second attempt."""
        if not isinstance(raw_response, bytes):
            raise TypeError("attempt authority must be exact response bytes")
        self._validate_query(transport_identity, query_text)
        attempt = Path(attempt_path)
        if not attempt.is_dir():
            raise RetrievalCacheIntegrityError("reserved retrieval attempt is missing")
        request_bytes = (attempt / "request.json").read_bytes()
        try:
            request = json.loads(request_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetrievalCacheIntegrityError("reserved retrieval attempt request is corrupt") from exc
        request = _exact_fields(
            request,
            {"schema_version", "request_key", "transport_identity", "query_text", "attempt_id"},
            "attempt request",
        )
        if (
            request["schema_version"] != RETRIEVAL_CACHE_SCHEMA_VERSION
            or request["request_key"] != transport_identity.request_key
            or request["transport_identity"] != transport_identity.canonical_dict()
            or request["query_text"] != query_text
            or request["attempt_id"] != attempt.name
        ):
            raise RetrievalCacheIntegrityError("reserved retrieval attempt identity mismatch")
        response_bytes = (attempt / "response.bin").read_bytes()
        if response_bytes != raw_response:
            raise RetrievalCacheIntegrityError("transport receipt differs from persisted raw response")
        manifest = {
            "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
            "request_key": transport_identity.request_key,
            "attempt_id": attempt.name,
            "status": "success",
            "response_filename": "response.bin",
            "raw_sha256": _sha256(response_bytes),
            "raw_length": len(response_bytes),
        }
        manifest_path = attempt / "manifest.json"
        _write_create_only(manifest_path, _json_bytes(manifest))
        _fsync_directory(attempt)
        _fsync_directory(attempt.parent)
        self._hook("attempt_manifest")
        return attempt

    def _validate_query(self, identity: TransportIdentity, query_text: str) -> None:
        if not isinstance(query_text, str):
            raise TypeError("query_text must be a string")
        if _sha256(query_text.encode("utf-8")) != identity.query_sha256:
            raise RetrievalCacheIntegrityError(
                "query_text does not match transport identity query_sha256"
            )

    def _raw_bytes(self, raw_response: bytes | bytearray | Mapping[str, object] | object) -> bytes:
        if isinstance(raw_response, bytes):
            return raw_response
        raise TypeError("raw_response must contain exact response bytes")

    def _parse(self, raw: bytes) -> tuple[_ParsedHit, ...]:
        try:
            parsed = self.normalizer.parse(raw)
        except RetrievalCacheError:
            raise
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RetrievalCacheIntegrityError("organizer response failed strict parsing") from exc
        return tuple(parsed)

    @staticmethod
    def _validate_response_identity(
        raw: bytes,
        identity: TransportIdentity,
        query_text: str,
    ) -> None:
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetrievalCacheIntegrityError("organizer response is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise RetrievalCacheIntegrityError("organizer response must be an object")
        if payload.get("index") != identity.index_id:
            raise RetrievalCacheIntegrityError(
                "organizer response index does not match transport identity"
            )
        if payload.get("query") != {"text": query_text}:
            raise RetrievalCacheIntegrityError(
                "organizer response query does not match exact request query"
            )

    def _admit_hits(self, parsed: tuple[_ParsedHit, ...]) -> tuple[CachedHit, ...]:
        result: list[CachedHit] = []
        receipts: dict[str, str] = {}
        for row in parsed:
            content_sha256 = _sha256(row.text.encode("utf-8"))
            if content_sha256 in receipts:
                result.append(CachedHit(row.rank, row.score, row.docid, content_sha256))
                continue
            try:
                receipt = self.document_store.admit_text(row.text)
            except DocumentStoreIntegrityError as exc:
                raise RetrievalCacheIntegrityError("document admission failed") from exc
            receipts[content_sha256] = receipt.content_sha256
            result.append(CachedHit(row.rank, row.score, row.docid, receipt.content_sha256))
        return tuple(result)

    def _verify_hits(self, parsed: tuple[_ParsedHit, ...]) -> tuple[CachedHit, ...]:
        result: list[CachedHit] = []
        verified: set[str] = set()
        for row in parsed:
            digest = _sha256(row.text.encode("utf-8"))
            if digest not in verified:
                try:
                    self.document_store.verify(digest)
                except DocumentStoreIntegrityError as exc:
                    raise RetrievalCacheIntegrityError(
                        "document object is missing or corrupt"
                    ) from exc
                verified.add(digest)
            result.append(CachedHit(row.rank, row.score, row.docid, digest))
        return tuple(result)

    def _entry_path(self, request_key: str) -> Path:
        return self.v2_root / request_key[:2] / request_key

    def _lock_path(self, request_key: str) -> Path:
        path = self.v2_root / ".locks" / f"{request_key}.lock"
        _private_mkdir(path.parent)
        return path

    def _lookup_locked(
        self,
        transport_identity: TransportIdentity,
        derivation_identity: DerivationIdentity,
        query_text: str,
    ) -> CachedRetrieval | None:
        entry = self._entry_path(transport_identity.request_key)
        attempt_raw = self._recover_attempt_locked(transport_identity, query_text)
        stale_temporary = sorted(entry.parent.glob(f".{entry.name}.*.tmp"))
        if stale_temporary:
            if attempt_raw is None:
                raise RetrievalCacheIntegrityError(
                    f"incomplete cache publication state: {stale_temporary[0]}"
                )
            for staging in stale_temporary:
                self._remove_validated_transport_staging_locked(
                    staging,
                    transport_identity,
                    attempt_raw,
                )
        if entry.exists():
            try:
                raw = self._load_transport(entry, transport_identity, query_text)
            except RetrievalCacheIntegrityError:
                if attempt_raw is None:
                    raise
                raw_path = entry / "raw.body.gz"
                manifest_path = entry / "transport-manifest.json"
                if raw_path.exists():
                    try:
                        existing_compressed = raw_path.read_bytes()
                    except OSError:
                        raise
                    if existing_compressed != _gzip_bytes(attempt_raw):
                        raise RetrievalCacheIntegrityError(
                            "partial transport raw bytes conflict with sealed authority"
                        )
                if raw_path.exists() and manifest_path.exists():
                    raise RetrievalCacheIntegrityError(
                        "complete transport artifacts are corrupt"
                    )
                self._publish_transport_locked(transport_identity, attempt_raw)
                raw = self._load_transport(entry, transport_identity, query_text)
            if attempt_raw is not None and attempt_raw != raw:
                self._record_conflict(
                    transport_identity.request_key, attempt_raw, "attempt-raw-authority"
                )
                raise RetrievalCacheConflictError(
                    f"complete attempt conflicts with transport authority for request "
                    f"{transport_identity.request_key}"
                )
            return self._load_or_publish_derivation_locked(
                entry, transport_identity, derivation_identity, query_text, raw
            )
        raw = attempt_raw
        if raw is None:
            return None
        self._publish_transport_locked(transport_identity, raw)
        return self._load_or_publish_derivation_locked(
            self._entry_path(transport_identity.request_key),
            transport_identity,
            derivation_identity,
            query_text,
            raw,
        )

    def _remove_validated_transport_staging_locked(
        self,
        staging: Path,
        identity: TransportIdentity,
        sealed_raw: bytes,
    ) -> None:
        """Remove only an exact uncommitted staging tree matching sealed authority."""
        expected_name = re.compile(
            rf"\.{re.escape(identity.request_key)}\.[0-9a-f]{{32}}\.tmp\Z"
        )
        try:
            staging_stat = staging.lstat()
        except OSError as exc:
            raise RetrievalCacheIntegrityError(
                "stale transport staging cannot be authenticated"
            ) from exc
        if (
            staging.parent != self._entry_path(identity.request_key).parent
            or expected_name.fullmatch(staging.name) is None
            or not stat.S_ISDIR(staging_stat.st_mode)
        ):
            raise RetrievalCacheIntegrityError(
                "stale transport staging path cannot be authenticated"
            )
        compressed, manifest_bytes = _transport_artifacts(identity, sealed_raw)
        expected_artifacts = {
            "raw.body.gz": compressed,
            "transport-manifest.json": manifest_bytes,
        }
        try:
            children = list(staging.iterdir())
        except OSError as exc:
            raise RetrievalCacheIntegrityError(
                "stale transport staging cannot be inspected"
            ) from exc
        if any(child.name not in expected_artifacts for child in children):
            raise RetrievalCacheIntegrityError(
                "stale transport staging contains an unknown artifact"
            )
        for child in children:
            try:
                child_stat = child.lstat()
                child_bytes = child.read_bytes()
            except OSError as exc:
                raise RetrievalCacheIntegrityError(
                    "stale transport staging artifact cannot be authenticated"
                ) from exc
            if (
                not stat.S_ISREG(child_stat.st_mode)
                or child_bytes != expected_artifacts[child.name]
            ):
                raise RetrievalCacheIntegrityError(
                    "stale transport staging conflicts with sealed authority"
                )
        shutil.rmtree(staging)
        _fsync_directory(staging.parent)

    def _load_transport(
        self, entry: Path, identity: TransportIdentity, query_text: str | None
    ) -> bytes:
        if not entry.is_dir():
            raise RetrievalCacheIntegrityError(f"transport entry is not a directory: {entry}")
        raw_path = entry / "raw.body.gz"
        manifest_path = entry / "transport-manifest.json"
        if not raw_path.exists() or not manifest_path.exists():
            raise RetrievalCacheIntegrityError(f"partial transport entry: {entry}")
        try:
            manifest = json.loads(manifest_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetrievalCacheIntegrityError("invalid transport manifest") from exc
        manifest = _exact_fields(
            manifest,
            {
                "schema_version",
                "request_key",
                "transport_identity",
                "query_sha256",
                "raw_sha256",
                "raw_length",
                "gzip_sha256",
                "gzip_length",
                "raw_filename",
            },
            "transport manifest",
        )
        if manifest["schema_version"] != RETRIEVAL_CACHE_SCHEMA_VERSION:
            raise RetrievalCacheIntegrityError("transport cache schema mismatch")
        if manifest["request_key"] != identity.request_key:
            raise RetrievalCacheIntegrityError("transport request key mismatch")
        if manifest["transport_identity"] != identity.canonical_dict():
            raise RetrievalCacheIntegrityError("transport identity mismatch")
        if manifest["query_sha256"] != identity.query_sha256:
            raise RetrievalCacheIntegrityError("transport query hash mismatch")
        if manifest["raw_filename"] != "raw.body.gz":
            raise RetrievalCacheIntegrityError("transport raw filename mismatch")
        compressed = raw_path.read_bytes()
        if manifest["gzip_length"] != len(compressed) or manifest["gzip_sha256"] != _sha256(compressed):
            raise RetrievalCacheIntegrityError("transport gzip receipt mismatch")
        try:
            decompressed = gzip.decompress(compressed)
        except (OSError, EOFError) as exc:
            raise RetrievalCacheIntegrityError("transport gzip is corrupt") from exc
        expected_compressed = _gzip_bytes(decompressed)
        if expected_compressed != compressed:
            raise RetrievalCacheIntegrityError("transport gzip is not deterministic")
        raw = decompressed
        if manifest["raw_length"] != len(raw) or manifest["raw_sha256"] != _sha256(raw):
            raise RetrievalCacheIntegrityError("transport raw receipt mismatch")
        if query_text is not None and _sha256(query_text.encode("utf-8")) != identity.query_sha256:
            raise RetrievalCacheIntegrityError("transport query text mismatch")
        return raw

    def _load_or_publish_derivation_locked(
        self,
        entry: Path,
        transport_identity: TransportIdentity,
        derivation_identity: DerivationIdentity,
        query_text: str,
        raw: bytes,
    ) -> CachedRetrieval:
        self._bind_derivation_identity(derivation_identity)
        self._validate_response_identity(raw, transport_identity, query_text)
        parsed = self._parse(raw)
        derived = entry / "derived" / derivation_identity.derivation_key
        if derived.exists():
            hits = self._verify_hits(parsed)
            return self._finish_derivation_locked(
                derived,
                transport_identity,
                derivation_identity,
                query_text,
                raw,
                hits,
            )
        hits = self._admit_hits(parsed)
        return self._publish_derivation_locked(
            entry, transport_identity, derivation_identity, query_text, raw, hits
        )

    def _load_derivation(
        self,
        derived: Path,
        transport_identity: TransportIdentity,
        derivation_identity: DerivationIdentity,
        query_text: str,
        raw: bytes,
        expected_hits: tuple[CachedHit, ...],
    ) -> CachedRetrieval:
        hits_path = derived / "hits.json"
        manifest_path = derived / "derivation-manifest.json"
        if not hits_path.exists() or not manifest_path.exists():
            raise RetrievalCacheIntegrityError(f"partial derivation entry: {derived}")
        try:
            hits_bytes = hits_path.read_bytes()
            manifest_bytes = manifest_path.read_bytes()
            stored = json.loads(hits_bytes)
            manifest = json.loads(manifest_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetrievalCacheIntegrityError("invalid derivation artifacts") from exc
        stored = _exact_fields(stored, {"hits"}, "hits receipt")
        manifest = _exact_fields(
            manifest,
            {
                "schema_version",
                "derivation_key",
                "derivation_identity",
                "parent_request_key",
                "parent_raw_sha256",
                "hits_filename",
                "hits_sha256",
                "hits_length",
                "hit_count",
                "ordered_semantic_hit_digest",
                "document_closure",
            },
            "derivation manifest",
        )
        hit_rows = stored["hits"]
        if not isinstance(hit_rows, list):
            raise RetrievalCacheIntegrityError("hits receipt must contain a list")
        actual_hits = self._decode_hits(hit_rows)
        if actual_hits != expected_hits:
            raise RetrievalCacheIntegrityError("hits do not reconstruct from raw authority")
        if manifest["schema_version"] != RETRIEVAL_CACHE_SCHEMA_VERSION:
            raise RetrievalCacheIntegrityError("derivation schema mismatch")
        if manifest["derivation_key"] != derivation_identity.derivation_key:
            raise RetrievalCacheIntegrityError("derivation key mismatch")
        if manifest["derivation_identity"] != derivation_identity.canonical_dict():
            raise RetrievalCacheIntegrityError("derivation identity mismatch")
        if manifest["parent_request_key"] != transport_identity.request_key:
            raise RetrievalCacheIntegrityError("derivation parent request mismatch")
        if manifest["parent_raw_sha256"] != _sha256(raw):
            raise RetrievalCacheIntegrityError("derivation parent raw mismatch")
        if manifest["hits_filename"] != "hits.json":
            raise RetrievalCacheIntegrityError("derivation hits filename mismatch")
        if manifest["hits_sha256"] != _sha256(hits_bytes) or manifest["hits_length"] != len(hits_bytes):
            raise RetrievalCacheIntegrityError("derivation hits receipt mismatch")
        if manifest["hit_count"] != len(actual_hits):
            raise RetrievalCacheIntegrityError("derivation hit count mismatch")
        if manifest["ordered_semantic_hit_digest"] != _semantic_digest(actual_hits):
            raise RetrievalCacheIntegrityError("derivation semantic digest mismatch")
        closure = sorted({hit.content_sha256 for hit in actual_hits})
        if manifest["document_closure"] != closure:
            raise RetrievalCacheIntegrityError("derivation document closure mismatch")
        for content_sha256 in closure:
            try:
                self.document_store.verify(content_sha256)
            except DocumentStoreIntegrityError as exc:
                raise RetrievalCacheIntegrityError("derivation document closure is invalid") from exc
        return CachedRetrieval(
            transport_identity,
            derivation_identity,
            query_text,
            actual_hits,
            raw,
            _sha256(raw),
        )

    def _decode_hits(self, rows: object) -> tuple[CachedHit, ...]:
        if not isinstance(rows, list):
            raise RetrievalCacheIntegrityError("hit rows must be a list")
        decoded: list[CachedHit] = []
        for position, row in enumerate(rows, start=1):
            row = _exact_fields(
                row,
                {"rank", "score", "docid", "content_sha256"},
                f"hit {position}",
            )
            decoded.append(
                CachedHit(
                    _strict_int(row["rank"], f"hit {position}.rank", positive=True),
                    _strict_score(row["score"], f"hit {position}.score"),
                    self._strict_docid(row["docid"], f"hit {position}.docid"),
                    _digest(row["content_sha256"], f"hit {position}.content_sha256"),
                )
            )
        return tuple(decoded)

    @staticmethod
    def _strict_docid(value: object, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise RetrievalCacheIntegrityError(f"{field} must be non-empty text")
        return value

    def _publish_derivation_locked(
        self,
        entry: Path,
        transport_identity: TransportIdentity,
        derivation_identity: DerivationIdentity,
        query_text: str,
        raw: bytes,
        hits: tuple[CachedHit, ...],
    ) -> CachedRetrieval:
        self._bind_derivation_identity(derivation_identity)
        derived = entry / "derived" / derivation_identity.derivation_key
        if derived.exists():
            return self._finish_derivation_locked(
                derived,
                transport_identity,
                derivation_identity,
                query_text,
                raw,
                hits,
            )
        _private_mkdir(derived.parent)
        staging = derived.parent / f".{derivation_identity.derivation_key}.{uuid.uuid4().hex}.tmp"
        _private_mkdir(staging)
        try:
            hit_payload = {"hits": [_hit_dict(hit) for hit in hits]}
            hit_bytes = _json_bytes(hit_payload)
            _write_durable(staging / "hits.json", hit_bytes)
            self._hook("derived_hits")
            manifest = {
                "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
                "derivation_key": derivation_identity.derivation_key,
                "derivation_identity": derivation_identity.canonical_dict(),
                "parent_request_key": transport_identity.request_key,
                "parent_raw_sha256": _sha256(raw),
                "hits_filename": "hits.json",
                "hits_sha256": _sha256(hit_bytes),
                "hits_length": len(hit_bytes),
                "hit_count": len(hits),
                "ordered_semantic_hit_digest": _semantic_digest(hits),
                "document_closure": sorted({hit.content_sha256 for hit in hits}),
            }
            _write_durable(staging / "derivation-manifest.json", _json_bytes(manifest))
            derived.mkdir(mode=0o700)
            os.chmod(derived, 0o700)
            _link_create_only(staging / "hits.json", derived / "hits.json")
            _fsync_directory(derived)
            self._hook("derived_hits_linked")
            _link_create_only(
                staging / "derivation-manifest.json",
                derived / "derivation-manifest.json",
            )
            _fsync_directory(derived)
            _fsync_directory(derived.parent)
            self._hook("derived_manifest_linked")
            return self._load_derivation(
                derived, transport_identity, derivation_identity, query_text, raw, hits
            )
        except FileExistsError:
            if derived.exists():
                return self._load_derivation(
                    derived, transport_identity, derivation_identity, query_text, raw, hits
                )
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _finish_derivation_locked(
        self,
        derived: Path,
        transport_identity: TransportIdentity,
        derivation_identity: DerivationIdentity,
        query_text: str,
        raw: bytes,
        hits: tuple[CachedHit, ...],
    ) -> CachedRetrieval:
        """Complete a partial derivation only from the validated raw authority."""
        if not derived.is_dir():
            raise RetrievalCacheIntegrityError(f"derivation entry is not a directory: {derived}")
        hit_bytes = _json_bytes({"hits": [_hit_dict(hit) for hit in hits]})
        manifest_bytes = _json_bytes(
            {
                "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
                "derivation_key": derivation_identity.derivation_key,
                "derivation_identity": derivation_identity.canonical_dict(),
                "parent_request_key": transport_identity.request_key,
                "parent_raw_sha256": _sha256(raw),
                "hits_filename": "hits.json",
                "hits_sha256": _sha256(hit_bytes),
                "hits_length": len(hit_bytes),
                "hit_count": len(hits),
                "ordered_semantic_hit_digest": _semantic_digest(hits),
                "document_closure": sorted({hit.content_sha256 for hit in hits}),
            }
        )
        hits_path = derived / "hits.json"
        manifest_path = derived / "derivation-manifest.json"
        if hits_path.exists() and hits_path.read_bytes() != hit_bytes:
            raise RetrievalCacheIntegrityError("partial derivation hits conflict with raw authority")
        if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
            raise RetrievalCacheIntegrityError("partial derivation manifest conflicts with raw authority")
        if not hits_path.exists():
            _write_create_only(hits_path, hit_bytes)
            _fsync_directory(derived)
            self._hook("derived_hits_linked")
        if not manifest_path.exists():
            _write_create_only(manifest_path, manifest_bytes)
            _fsync_directory(derived)
            _fsync_directory(derived.parent)
            self._hook("derived_manifest_linked")
        return self._load_derivation(
            derived, transport_identity, derivation_identity, query_text, raw, hits
        )

    def _publish_transport_locked(self, identity: TransportIdentity, raw: bytes) -> None:
        entry = self._entry_path(identity.request_key)
        if entry.exists():
            try:
                existing = self._load_transport(entry, identity, None)
            except RetrievalCacheIntegrityError:
                return self._finish_transport_locked(entry, identity, raw)
            if existing != raw:
                self._record_conflict(identity.request_key, raw, "raw-authority")
                raise RetrievalCacheConflictError(
                    f"raw authority conflict for request {identity.request_key}"
                )
            return
        _private_mkdir(entry.parent)
        staging = entry.parent / f".{entry.name}.{uuid.uuid4().hex}.tmp"
        _private_mkdir(staging)
        compressed, manifest_bytes = _transport_artifacts(identity, raw)
        try:
            _write_durable(staging / "raw.body.gz", compressed)
            _write_durable(staging / "transport-manifest.json", manifest_bytes)
            self._hook("transport_raw")
            entry.mkdir()
            os.chmod(entry, 0o700)
            _link_create_only(staging / "raw.body.gz", entry / "raw.body.gz")
            _fsync_directory(entry)
            self._hook("transport_raw_linked")
            _link_create_only(
                staging / "transport-manifest.json", entry / "transport-manifest.json"
            )
            _fsync_directory(entry)
            _fsync_directory(entry.parent)
            self._hook("transport_manifest_linked")
        except FileExistsError:
            if entry.exists():
                existing = self._load_transport(entry, identity, None)
                if existing == raw:
                    return
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _finish_transport_locked(
        self, entry: Path, identity: TransportIdentity, raw: bytes
    ) -> None:
        """Finish a partial entry only when its existing bytes agree exactly."""
        if not entry.is_dir():
            raise RetrievalCacheIntegrityError(f"transport entry is not a directory: {entry}")
        _private_mkdir(entry)
        compressed, manifest_bytes = _transport_artifacts(identity, raw)
        raw_path = entry / "raw.body.gz"
        manifest_path = entry / "transport-manifest.json"
        if raw_path.exists():
            existing_compressed = raw_path.read_bytes()
            if existing_compressed != compressed:
                raise RetrievalCacheConflictError(
                    f"partial raw authority conflict for request {identity.request_key}"
                )
        else:
            _write_create_only(raw_path, compressed)
            _fsync_directory(entry)
            self._hook("transport_raw_linked")
        if manifest_path.exists():
            if manifest_path.read_bytes() != manifest_bytes:
                raise RetrievalCacheIntegrityError(
                    f"partial transport manifest conflicts for request {identity.request_key}"
                )
        else:
            _write_create_only(manifest_path, manifest_bytes)
            _fsync_directory(entry)
            _fsync_directory(entry.parent)
            self._hook("transport_manifest_linked")

    def _write_attempt_locked(
        self, identity: TransportIdentity, query_text: str, raw: bytes
    ) -> Path:
        request_key = identity.request_key
        attempts_base = self.v2_root / "attempts"
        _private_mkdir(attempts_base)
        attempts_root = attempts_base / request_key
        _private_mkdir(attempts_root)
        attempt = attempts_root / uuid.uuid4().hex
        _private_mkdir(attempt)
        request = {
            "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
            "request_key": request_key,
            "transport_identity": identity.canonical_dict(),
            "query_text": query_text,
            "attempt_id": attempt.name,
        }
        _write_durable(attempt / "request.json", _json_bytes(request))
        _write_durable(attempt / "response.bin", raw)
        self._hook("attempt_response")
        manifest = {
            "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
            "request_key": request_key,
            "attempt_id": attempt.name,
            "status": "success",
            "response_filename": "response.bin",
            "raw_sha256": _sha256(raw),
            "raw_length": len(raw),
        }
        _write_durable(attempt / "manifest.json", _json_bytes(manifest))
        _fsync_directory(attempt)
        _fsync_directory(attempt.parent)
        self._hook("attempt_manifest")
        return attempt

    def _recover_attempt_locked(
        self, identity: TransportIdentity, query_text: str
    ) -> bytes | None:
        attempts_root = self.v2_root / "attempts" / identity.request_key
        if not attempts_root.exists():
            return None
        complete: list[bytes] = []
        for attempt in sorted(attempts_root.iterdir()):
            if not attempt.is_dir():
                continue
            manifest_path = attempt / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                request = json.loads((attempt / "request.json").read_bytes())
                manifest = json.loads(manifest_path.read_bytes())
                request = _exact_fields(
                    request,
                    {"schema_version", "request_key", "transport_identity", "query_text", "attempt_id"},
                    "attempt request",
                )
                manifest = _exact_fields(
                    manifest,
                    {"schema_version", "request_key", "attempt_id", "status", "response_filename", "raw_sha256", "raw_length"},
                    "attempt manifest",
                )
                if (
                    request["schema_version"] != RETRIEVAL_CACHE_SCHEMA_VERSION
                    or request["request_key"] != identity.request_key
                    or request["transport_identity"] != identity.canonical_dict()
                    or request["query_text"] != query_text
                    or request["attempt_id"] != attempt.name
                    or manifest["schema_version"] != RETRIEVAL_CACHE_SCHEMA_VERSION
                    or manifest["request_key"] != identity.request_key
                    or manifest["attempt_id"] != attempt.name
                    or manifest["status"] != "success"
                    or manifest["response_filename"] != "response.bin"
                ):
                    raise RetrievalCacheIntegrityError("attempt identity mismatch")
                raw = (attempt / "response.bin").read_bytes()
                if manifest["raw_length"] != len(raw) or manifest["raw_sha256"] != _sha256(raw):
                    raise RetrievalCacheIntegrityError("attempt raw receipt mismatch")
            except FileNotFoundError as exc:
                raise RetrievalCacheIntegrityError("complete attempt is missing an artifact") from exc
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RetrievalCacheIntegrityError("invalid complete attempt") from exc
            complete.append(raw)
        if not complete:
            return None
        winner = complete[0]
        if any(raw != winner for raw in complete[1:]):
            self._record_conflict(identity.request_key, complete[-1], "attempt-raw-authority")
            raise RetrievalCacheConflictError(
                f"complete attempts conflict for request {identity.request_key}"
            )
        return winner

    def _record_conflict(self, request_key: str, raw: bytes, reason: str) -> None:
        conflicts_root = self.v2_root / "conflicts"
        _private_mkdir(conflicts_root)
        request_conflicts = conflicts_root / request_key
        _private_mkdir(request_conflicts)
        directory = request_conflicts / f"{reason}-{uuid.uuid4().hex}"
        _private_mkdir(directory)
        compressed = _gzip_bytes(raw)
        _write_durable(directory / "raw.body.gz", compressed)
        _write_durable(
            directory / "conflict.json",
            _json_bytes(
                {
                    "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
                    "request_key": request_key,
                    "reason": reason,
                    "raw_sha256": _sha256(raw),
                    "raw_length": len(raw),
                    "gzip_sha256": _sha256(compressed),
                }
            ),
        )

    def _hook(self, phase: str) -> None:
        if self.publication_hook is not None:
            self.publication_hook(phase)


def _gzip_bytes(raw: bytes) -> bytes:
    return gzip.compress(raw, compresslevel=9, mtime=0)


def _transport_artifacts(
    identity: TransportIdentity,
    raw: bytes,
) -> tuple[bytes, bytes]:
    compressed = _gzip_bytes(raw)
    manifest = {
        "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
        "request_key": identity.request_key,
        "transport_identity": identity.canonical_dict(),
        "query_sha256": identity.query_sha256,
        "raw_sha256": _sha256(raw),
        "raw_length": len(raw),
        "gzip_sha256": _sha256(compressed),
        "gzip_length": len(compressed),
        "raw_filename": "raw.body.gz",
    }
    return compressed, _json_bytes(manifest)


def _historical_retrieval_cache_key(sidecar: Mapping[str, object]) -> str:
    identity = {
        "retriever_name": sidecar["retriever_name"],
        "retriever_type": sidecar["retriever_type"],
        "index": sidecar["index"],
        "index_url": sidecar["index_url"],
        "hits": sidecar["hits"],
        "query_text": sidecar["query"],
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(encoded)[:16]


def _link_create_only(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError:
        if destination.read_bytes() != source.read_bytes():
            raise RetrievalCacheConflictError(
                f"immutable artifact conflict at {destination}"
            )


def _write_create_only(path: Path, content: bytes) -> None:
    """Create one exact private file without replacing an existing artifact."""
    _private_mkdir(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as sink:
            os.fchmod(sink.fileno(), 0o600)
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RetrievalCacheConflictError(
                    f"immutable artifact conflict at {path}"
                )
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _hit_dict(hit: CachedHit) -> dict[str, object]:
    return {
        "rank": hit.rank,
        "score": hit.score,
        "docid": hit.docid,
        "content_sha256": hit.content_sha256,
    }


def _semantic_digest(hits: tuple[CachedHit, ...]) -> str:
    semantic = [
        {
            "rank": hit.rank,
            "score_hex": hit.score.hex(),
            "docid": hit.docid,
            "content_sha256": hit.content_sha256,
        }
        for hit in hits
    ]
    return _sha256(_canonical_bytes(semantic))


__all__ = [
    "CachedHit",
    "CachedRetrieval",
    "DerivationIdentity",
    "LegacyPromotionReceipt",
    "OrganizerTextNormalizer",
    "RETRIEVAL_CACHE_SCHEMA_VERSION",
    "RETRIEVAL_SCORE_NORMALIZER_VERSION",
    "RETRIEVAL_TEXT_NORMALIZER_VERSION",
    "RetrievalCache",
    "RetrievalCacheConflictError",
    "RetrievalCacheError",
    "RetrievalCacheIntegrityError",
    "RetrievalCacheMiss",
    "TransportIdentity",
]
