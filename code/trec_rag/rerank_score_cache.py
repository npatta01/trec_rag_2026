"""Build cached cross-encoder reranker scores from shared retrieval caches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import secrets
import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker
from trec_rag.pipeline import pipeline_cache_dir
from trec_rag.pipeline_config import PipelineConfig, RetrieverConfig, load_pipeline_config
from trec_rag.pipeline_models import QueryVariant, RetrievedCandidate, jsonable
from trec_rag.query_understanding import build_query_variants
from trec_rag.ranking import passthrough_rank
from trec_rag.remote_config import RemotePyseriniConfig
from trec_rag.repo_env import load_repo_env, repo_cache_root, shared_checkout_root
from trec_rag.retrievers import cache_path, normalize_retrieved_candidates, request_cache_key
from trec_rag.topics import Topic, load_topics


DEFAULT_INDEX_URL = "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
DEFAULT_MODEL_REVISION = "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
DEFAULT_BACKEND_VERSION = "5.6.0"
DEFAULT_SCORE_REPRESENTATION = "raw_logits"
DEFAULT_INFERENCE_DTYPE = "bfloat16"
ARTIFACT_SCHEMA_VERSION = 2
SCORE_CACHE_SCHEMA_VERSION = "score-cache-v2"
SCORE_CACHE_CONTEXT_VERSION = 2
_HASH_BYTES = 32
_CLAIM_TOKEN_BYTES = 16
_SQLITE_BUSY_TIMEOUT_MS = 60_000
_MAX_FINITE_SCORE = 1.7976931348623157e+308
_FALSE_LEGACY_INPUT_POLICIES = frozenset(
    {"trec_rag_raw_v2", "extractive_sentence_pair_v1"}
)
_EFFECTIVE_INPUT_POLICY = "trec_rag_whitespace_v1"
_LEGACY_IDENTITY_FIELDS = (
    "backend",
    "backend_version",
    "model",
    "model_revision",
    "score_representation",
    "inference_dtype",
    "input_policy",
    "max_length",
    "score_kind",
)


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    if topic_id.isdigit():
        return (0, int(topic_id))
    return (1, topic_id)


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScoreCacheContext:
    backend: str
    model: str
    max_length: int
    score_kind: str
    model_revision: str = "unversioned"
    backend_version: str = "unversioned"
    score_representation: str = DEFAULT_SCORE_REPRESENTATION
    inference_dtype: str = "unspecified"
    input_policy: str = "unspecified"
    requested_max_length: int | None = None
    pair_buffer_tokens: int = 0
    chunk_max_characters: int | None = None
    chunk_overlap_characters: int | None = None
    effective_max_length: int | None = None
    scoring_contract: str = "cross-encoder-score-v2"
    transformers_version: str | None = None
    torch_version: str | None = None
    device_family: str = "unspecified"
    fixed_batch_policy: str = "unspecified"

    @property
    def context_payload(self) -> dict[str, object]:
        payload = {"context_schema_version": SCORE_CACHE_CONTEXT_VERSION, **asdict(self)}
        payload["effective_max_length"] = (
            self.max_length if self.effective_max_length is None else self.effective_max_length
        )
        return payload

    @property
    def context_json(self) -> str:
        return json.dumps(self.context_payload, sort_keys=True, separators=(",", ":"))

    @property
    def context_sha256(self) -> str:
        return _sha256_text(self.context_json)

    @property
    def path_parts(self) -> tuple[str, ...]:
        return (
            "score-cache-v2",
            _slug(self.backend),
            _slug(self.model),
            f"{_slug(self.score_kind)}--{self.context_sha256}.sqlite3",
        )

    @property
    def cache_identity_metadata(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "model": self.model,
            "model_revision": self.model_revision,
            "score_representation": self.score_representation,
            "inference_dtype": self.inference_dtype,
            "input_policy": self.input_policy,
            "scoring_contract": self.scoring_contract,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "device_family": self.device_family,
            "fixed_batch_policy": self.fixed_batch_policy,
        }

    @property
    def artifact_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            **self.cache_identity_metadata,
            "max_length": self.max_length,
            "score_kind": self.score_kind,
            "pair_buffer_tokens": self.pair_buffer_tokens,
            "scoring_contract": self.scoring_contract,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "device_family": self.device_family,
            "fixed_batch_policy": self.fixed_batch_policy,
            "effective_max_length": (
                self.max_length
                if self.effective_max_length is None
                else self.effective_max_length
            ),
        }
        if self.requested_max_length is not None:
            metadata["requested_max_length"] = self.requested_max_length
        if self.chunk_max_characters is not None:
            metadata["chunk_max_characters"] = self.chunk_max_characters
        if self.chunk_overlap_characters is not None:
            metadata["chunk_overlap_characters"] = self.chunk_overlap_characters
        return metadata


@dataclass(frozen=True)
class _Pair:
    raw: Any
    key: bytes
    query_sha256: bytes
    text_sha256: bytes


@dataclass(frozen=True)
class _ScoredPair:
    pair: _Pair
    score: float


@dataclass(frozen=True)
class _Claim:
    pair: _Pair
    owner: str
    token: bytes


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _hash_text(value: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError("score-cache pair text must be str")
    return hashlib.sha256(value.encode("utf-8")).digest()


def _hash_blob(value: str | bytes | bytearray) -> bytes:
    if isinstance(value, str):
        if len(value) != 64:
            raise ValueError("score-cache hashes must be 64 hexadecimal characters")
        try:
            result = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError("score-cache hashes must be hexadecimal") from exc
    else:
        result = bytes(value)
    if len(result) != _HASH_BYTES:
        raise ValueError("score-cache hashes must be 32 bytes")
    return result


def score_cache_key_from_hashes(
    context: ScoreCacheContext,
    *,
    query_sha256: str | bytes | bytearray,
    text_sha256: str | bytes | bytearray,
) -> str:
    """Derive the canonical v2 score-cache key from content hashes."""
    query_hash = _hash_blob(query_sha256)
    text_hash = _hash_blob(text_sha256)
    return hashlib.sha256(
        _canonical_json(
            {
                "cache_key_schema_version": 2,
                "context_sha256": context.context_sha256,
                "query_sha256": query_hash.hex(),
                "text_sha256": text_hash.hex(),
            }
        )
    ).hexdigest()


def _finite_score(value: Any, *, source: str = "score") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{source} must be a finite real number, not bool")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{source} must be finite")
    if abs(result) > _MAX_FINITE_SCORE:
        raise ValueError(f"{source} exceeds SQLite REAL range")
    return result


class GlobalScoreCache:
    """Content-addressed SQLite score cache shared across experiments."""

    schema_version = 2

    def __init__(self, root_dir: Path, context: ScoreCacheContext) -> None:
        self.context = context
        self.root_dir = Path(root_dir)
        self.path = self.root_dir.joinpath(*context.path_parts)
        self._local = threading.local()
        self._owner_prefix = f"{os.getpid()}:{uuid4().hex}"
        self._integrity_check_lock = threading.Lock()
        self._integrity_checked = False
        self.connection

    @property
    def context_sha256(self) -> str:
        return self.context.context_sha256

    @property
    def context_json(self) -> str:
        return self.context.context_json

    @staticmethod
    def _schema_sql() -> str:
        return """
        CREATE TABLE IF NOT EXISTS cache_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) STRICT;
        CREATE TABLE IF NOT EXISTS scores (
            key_sha256 BLOB PRIMARY KEY CHECK(length(key_sha256) = 32),
            query_sha256 BLOB NOT NULL CHECK(length(query_sha256) = 32),
            text_sha256 BLOB NOT NULL CHECK(length(text_sha256) = 32),
            score REAL NOT NULL CHECK(
                typeof(score) = 'real'
                AND score = score
                AND abs(score) <= 1.7976931348623157e+308
            )
        ) STRICT;
        CREATE TABLE IF NOT EXISTS claims (
            key_sha256 BLOB PRIMARY KEY CHECK(length(key_sha256) = 32),
            query_sha256 BLOB NOT NULL CHECK(length(query_sha256) = 32),
            text_sha256 BLOB NOT NULL CHECK(length(text_sha256) = 32),
            owner TEXT NOT NULL,
            token BLOB NOT NULL CHECK(length(token) = 16),
            claimed_at REAL NOT NULL,
            lease_expires_at REAL NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS claims_expiry_idx ON claims(lease_expires_at);
        CREATE TABLE IF NOT EXISTS imports (
            source_sha256 BLOB PRIMARY KEY CHECK(length(source_sha256) = 32),
            source_path TEXT NOT NULL,
            source_row_count INTEGER NOT NULL CHECK(source_row_count >= 0),
            inserted_count INTEGER NOT NULL CHECK(inserted_count >= 0),
            logical_digest BLOB NOT NULL CHECK(length(logical_digest) = 32),
            authorization_sha256 BLOB CHECK(
                authorization_sha256 IS NULL OR length(authorization_sha256) = 32
            )
        ) STRICT;
        """

    @staticmethod
    def _normalize_schema_sql(value: str) -> str:
        normalized = value.lower()
        for character in "(),=":
            normalized = normalized.replace(character, f" {character} ")
        return " ".join(normalized.split())

    def _validate_existing_schema(self, connection: sqlite3.Connection, database: Path) -> None:
        expected = {
            ("table", "cache_meta"),
            ("table", "scores"),
            ("table", "claims"),
            ("table", "imports"),
            ("index", "claims_expiry_idx"),
        }
        actual_rows = connection.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        actual = {(str(row[0]), str(row[1])) for row in actual_rows}
        if actual != expected:
            raise ValueError(f"{database}: score-cache-v2 schema objects do not match exactly")
        expected_sql = {
            ("table", "cache_meta"): "CREATE TABLE cache_meta ( key TEXT PRIMARY KEY, value TEXT NOT NULL ) STRICT",
            ("table", "scores"): "CREATE TABLE scores ( key_sha256 BLOB PRIMARY KEY CHECK(length(key_sha256) = 32), query_sha256 BLOB NOT NULL CHECK(length(query_sha256) = 32), text_sha256 BLOB NOT NULL CHECK(length(text_sha256) = 32), score REAL NOT NULL CHECK(typeof(score) = 'real' AND score = score AND abs(score) <= 1.7976931348623157e+308) ) STRICT",
            ("table", "claims"): "CREATE TABLE claims ( key_sha256 BLOB PRIMARY KEY CHECK(length(key_sha256) = 32), query_sha256 BLOB NOT NULL CHECK(length(query_sha256) = 32), text_sha256 BLOB NOT NULL CHECK(length(text_sha256) = 32), owner TEXT NOT NULL, token BLOB NOT NULL CHECK(length(token) = 16), claimed_at REAL NOT NULL, lease_expires_at REAL NOT NULL ) STRICT",
            ("table", "imports"): "CREATE TABLE imports ( source_sha256 BLOB PRIMARY KEY CHECK(length(source_sha256) = 32), source_path TEXT NOT NULL, source_row_count INTEGER NOT NULL CHECK(source_row_count >= 0), inserted_count INTEGER NOT NULL CHECK(inserted_count >= 0), logical_digest BLOB NOT NULL CHECK(length(logical_digest) = 32), authorization_sha256 BLOB CHECK(authorization_sha256 IS NULL OR length(authorization_sha256) = 32) ) STRICT",
            ("index", "claims_expiry_idx"): "CREATE INDEX claims_expiry_idx ON claims(lease_expires_at)",
        }
        for row in actual_rows:
            identity = (str(row[0]), str(row[1]))
            if self._normalize_schema_sql(str(row[2])) != self._normalize_schema_sql(
                expected_sql[identity]
            ):
                raise ValueError(f"{database}: score-cache-v2 schema definition mismatch")

    def _new_connection(
        self,
        path: Path | None = None,
        *,
        validate_integrity: bool = True,
    ) -> sqlite3.Connection:
        database = path or self.path
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            database,
            timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            connection.execute("PRAGMA busy_timeout = 60000")
            connection.execute("PRAGMA foreign_keys = ON")
            deadline = time.monotonic() + (_SQLITE_BUSY_TIMEOUT_MS / 1000)
            while True:
                try:
                    journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            if str(journal_mode).lower() != "wal":
                raise ValueError(f"score cache requires WAL journal mode, found {journal_mode!r}")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                schema_exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1"
                ).fetchone() is not None
                if schema_exists:
                    self._validate_existing_schema(connection, database)
                else:
                    self._initialize_schema(connection)
                self._validate_or_initialize_meta(
                    connection,
                    initialize=not schema_exists,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            if validate_integrity and database == self.path:
                with self._integrity_check_lock:
                    if not self._integrity_checked:
                        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                            raise ValueError(f"{database}: SQLite integrity check failed")
                        self._integrity_checked = True
            return connection
        except BaseException:
            connection.close()
            raise

    def _initialize_schema(self, connection: sqlite3.Connection) -> None:
        statement = ""
        for line in self._schema_sql().splitlines(keepends=True):
            statement += line
            if sqlite3.complete_statement(statement):
                connection.execute(statement)
                statement = ""
        if statement.strip():  # pragma: no cover - static schema is always complete
            raise ValueError("score-cache-v2 schema contains an incomplete statement")

    def _validate_or_initialize_meta(
        self,
        connection: sqlite3.Connection,
        *,
        initialize: bool,
    ) -> None:
        expected = {
            "schema_version": SCORE_CACHE_SCHEMA_VERSION,
            "context_sha256": self.context_sha256,
            "context_json": self.context_json,
        }
        rows = dict(connection.execute("SELECT key, value FROM cache_meta"))
        if not rows and initialize:
            connection.executemany(
                "INSERT INTO cache_meta(key, value) VALUES (?, ?)", expected.items()
            )
            return
        if not rows:
            raise ValueError(f"{self.path}: score-cache-v2 metadata is empty")
        if rows != expected:
            raise ValueError(f"{self.path}: score-cache-v2 schema/context metadata mismatch")

    def _get_connection(self) -> sqlite3.Connection:
        process_id = os.getpid()
        thread_id = threading.get_ident()
        state = getattr(self._local, "state", None)
        if state is not None and (state[0] != process_id or state[1] != thread_id):
            try:
                state[2].close()
            except sqlite3.Error:
                pass
            state = None
        if state is None:
            state = (process_id, thread_id, self._new_connection())
            self._local.state = state
        return state[2]

    @property
    def connection(self) -> sqlite3.Connection:
        return self._get_connection()

    @property
    def connection_identity(self) -> tuple[int, int]:
        self._get_connection()
        return (os.getpid(), threading.get_ident())

    def close(self) -> None:
        state = getattr(self._local, "state", None)
        if state is not None:
            state[2].close()
            self._local.state = None

    def _load(self) -> dict[str, float]:
        return {
            bytes(key).hex(): _finite_score(score, source="stored score")
            for key, score in self.connection.execute("SELECT key_sha256, score FROM scores")
        }

    @property
    def scores(self) -> dict[str, float]:
        return self._load()

    @scores.setter
    def scores(self, _value: object) -> None:
        # Compatibility lock adapters may assign a stale snapshot. SQLite is
        # still the only mutable state and subsequent reads query it directly.
        return

    def _key_from_hashes(self, query_sha256: bytes, text_sha256: bytes) -> bytes:
        return bytes.fromhex(
            score_cache_key_from_hashes(
                self.context,
                query_sha256=query_sha256,
                text_sha256=text_sha256,
            )
        )

    def cache_key(self, *, query_text: str, text: str) -> str:
        return self._key_from_hashes(_hash_text(query_text), _hash_text(text)).hex()

    def _legacy_key_from_hashes(
        self,
        query_sha256: bytes,
        text_sha256: bytes,
        *,
        context: ScoreCacheContext | None = None,
    ) -> str:
        legacy_context = self.context if context is None else context
        return hashlib.sha256(
            _canonical_json(
                {
                    "schema_version": self.schema_version,
                    "backend": legacy_context.backend,
                    "model": legacy_context.model,
                    "max_length": legacy_context.max_length,
                    "score_kind": legacy_context.score_kind,
                    "backend_version": legacy_context.backend_version,
                    "model_revision": legacy_context.model_revision,
                    "score_representation": legacy_context.score_representation,
                    "inference_dtype": legacy_context.inference_dtype,
                    "input_policy": legacy_context.input_policy,
                    "query_sha256": query_sha256.hex(),
                    "text_sha256": text_sha256.hex(),
                }
            )
        ).hexdigest()

    def _normalize_pair(self, raw: Any) -> _Pair:
        supplied_key: bytes | None = None
        if isinstance(raw, _Pair):
            return raw
        if isinstance(raw, dict):
            query_text = raw.get("query_text")
            text = raw.get("text")
            query_sha256 = _hash_text(query_text) if query_text is not None else _hash_blob(raw["query_sha256"])
            text_sha256 = _hash_text(text) if text is not None else _hash_blob(raw["text_sha256"])
            if "key" in raw:
                supplied_key = _hash_blob(raw["key"])
            elif "cache_key" in raw:
                supplied_key = _hash_blob(raw["cache_key"])
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            query_text, text = raw
            query_sha256, text_sha256 = _hash_text(query_text), _hash_text(text)
        elif isinstance(raw, (tuple, list)) and len(raw) == 3:
            query_text, text, _score = raw
            query_sha256, text_sha256 = _hash_text(query_text), _hash_text(text)
        elif isinstance(raw, (tuple, list)) and len(raw) == 4:
            supplied_key = _hash_blob(raw[0])
            query_sha256, text_sha256 = _hash_blob(raw[1]), _hash_blob(raw[2])
        else:
            query_text = getattr(raw, "query_text", None)
            text = getattr(raw, "text", None)
            query_sha256 = _hash_text(query_text) if query_text is not None else _hash_blob(raw.query_sha256)
            text_sha256 = _hash_text(text) if text is not None else _hash_blob(raw.text_sha256)
            supplied_key_value = getattr(raw, "key", None)
            supplied_key = _hash_blob(supplied_key_value) if supplied_key_value is not None else None
        expected_key = self._key_from_hashes(query_sha256, text_sha256)
        if supplied_key is not None and supplied_key != expected_key:
            raise ValueError("score-cache key/hash mismatch")
        return _Pair(raw=raw, key=expected_key, query_sha256=query_sha256, text_sha256=text_sha256)

    def _normalize_pairs(self, pairs: Iterable[Any]) -> tuple[list[_Pair], list[_Pair]]:
        unique: dict[bytes, _Pair] = {}
        ordered: list[_Pair] = []
        for raw in pairs:
            pair = self._normalize_pair(raw)
            prior = unique.get(pair.key)
            if prior is not None and (
                prior.query_sha256 != pair.query_sha256 or prior.text_sha256 != pair.text_sha256
            ):
                raise ValueError("score-cache key/hash mismatch")
            if prior is None:
                unique[pair.key] = pair
            ordered.append(unique[pair.key])
        return list(unique.values()), ordered

    def _normalize_scored(self, raw: Any) -> _ScoredPair:
        if isinstance(raw, dict):
            score = raw.get("score")
            pair = self._normalize_pair(raw)
        elif isinstance(raw, (tuple, list)) and len(raw) == 3:
            pair = self._normalize_pair(raw[:2])
            score = raw[2]
        elif isinstance(raw, (tuple, list)) and len(raw) == 4:
            pair = self._normalize_pair(raw[:3] + (raw[3],))
            score = raw[3]
        else:
            score = getattr(raw, "score")
            pair = self._normalize_pair(raw)
        value = _finite_score(score, source="global score cache score")
        return _ScoredPair(pair=pair, score=value)

    def _lookup_normalized(self, pairs: Sequence[_Pair]) -> dict[bytes, float]:
        result: dict[bytes, float] = {}
        keys = list(pairs)
        for offset in range(0, len(keys), 500):
            batch = keys[offset : offset + 500]
            pairs_by_key = {pair.key: pair for pair in batch}
            placeholders = ", ".join("?" for _ in batch)
            rows = self.connection.execute(
                "SELECT key_sha256, query_sha256, text_sha256, score "
                f"FROM scores WHERE key_sha256 IN ({placeholders})",
                tuple(pair.key for pair in batch),
            )
            for key, query_hash, text_hash, score in rows:
                key_bytes = bytes(key)
                pair = pairs_by_key[key_bytes]
                if bytes(query_hash) != pair.query_sha256 or bytes(text_hash) != pair.text_sha256:
                    raise ValueError("score-cache key/hash mismatch")
                result[key_bytes] = _finite_score(score, source="stored score")
        return result

    def lookup_many(self, pairs: Iterable[Any]) -> list[float | None]:
        _unique, ordered = self._normalize_pairs(pairs)
        if not ordered:
            return []
        found = self._lookup_normalized(_unique)
        return [found.get(pair.key) for pair in ordered]

    def get(self, *, query_text: str, text: str) -> float | None:
        return self.lookup_many([(query_text, text)])[0]

    def _insert_score(
        self,
        connection: sqlite3.Connection,
        row: _ScoredPair,
        claim: _Claim | None,
    ) -> int:
        score = _finite_score(row.score, source="global score cache score")
        existing = connection.execute(
            "SELECT query_sha256, text_sha256, score FROM scores WHERE key_sha256 = ?",
            (row.pair.key,),
        ).fetchone()
        if existing is not None:
            if bytes(existing[0]) != row.pair.query_sha256 or bytes(existing[1]) != row.pair.text_sha256:
                raise ValueError("score-cache key/hash mismatch")
            if _finite_score(existing[2], source="stored score") != score:
                raise ValueError("conflicting score for existing global cache key")
            return 0
        if claim is not None:
            current = connection.execute(
                "SELECT owner, token, lease_expires_at FROM claims WHERE key_sha256 = ?",
                (row.pair.key,),
            ).fetchone()
            if (
                current is None
                or str(current[0]) != claim.owner
                or bytes(current[1]) != claim.token
                or float(current[2]) <= time.time()
            ):
                raise ValueError("score-cache claim lost before commit")
        connection.execute(
            "INSERT INTO scores(key_sha256, query_sha256, text_sha256, score) VALUES (?, ?, ?, ?)",
            (row.pair.key, row.pair.query_sha256, row.pair.text_sha256, score),
        )
        if claim is not None:
            connection.execute(
                "DELETE FROM claims WHERE key_sha256 = ? AND owner = ? AND token = ?",
                (row.pair.key, claim.owner, claim.token),
            )
        return 1

    def _seed_batch(self, rows: Sequence[_ScoredPair]) -> int:
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        inserted = 0
        try:
            for row in rows:
                inserted += self._insert_score(connection, row, claim=None)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return inserted

    @staticmethod
    def _receipt_result(
        row_count: int,
        inserted_count: int,
        logical_digest: bytes,
        authorization_sha256: bytes | None,
        *,
        legacy_context: ScoreCacheContext,
        target_context: ScoreCacheContext,
    ) -> dict[str, object]:
        return {
            "source_row_count": row_count,
            "inserted_count": inserted_count,
            "logical_digest": logical_digest.hex(),
            "authorization_sha256": (
                authorization_sha256.hex() if authorization_sha256 is not None else None
            ),
            "legacy_context_sha256": legacy_context.context_sha256,
            "target_context_sha256": target_context.context_sha256,
            "declared_input_policy": legacy_context.input_policy,
            "effective_input_policy": target_context.input_policy,
        }

    def _authorization_digest(self, legacy_context: ScoreCacheContext) -> bytes:
        return hashlib.sha256(
            _canonical_json(
                {
                    "authorization_schema_version": 1,
                    "legacy_context_sha256": legacy_context.context_sha256,
                    "target_context_sha256": self.context_sha256,
                }
            )
        ).digest()

    def _validate_legacy_context_rebinding(
        self,
        legacy_context: ScoreCacheContext,
    ) -> None:
        target_context = self.context
        for field, legacy_value in asdict(legacy_context).items():
            if field == "input_policy":
                continue
            if legacy_value != getattr(target_context, field):
                raise ValueError(
                    "legacy score JSONL context rebinding mismatch for "
                    f"{field}: legacy={legacy_value!r} target={getattr(target_context, field)!r}"
                )

        legacy_policy = legacy_context.input_policy
        target_policy = target_context.input_policy
        if legacy_policy in _FALSE_LEGACY_INPUT_POLICIES:
            if target_policy != _EFFECTIVE_INPUT_POLICY:
                raise ValueError(
                    "legacy score JSONL input policy rebinding is allowed only to "
                    f"{_EFFECTIVE_INPUT_POLICY}"
                )
        elif legacy_policy != target_policy:
            raise ValueError("legacy score JSONL input policy rebinding mismatch")

    def seed_many(
        self,
        source: Iterable[Any],
        *,
        source_path: str | Path | None = None,
        source_sha256: str | bytes | None = None,
        batch_size: int = 512,
    ) -> int:
        if source_path is None or source_sha256 is None:
            if isinstance(source, dict):
                source_path = source.get("source_path", source.get("path"))
                source_sha256 = source.get("source_sha256", source.get("sha256"))
                source = source.get("rows", source.get("source"))
            else:
                source_path = getattr(source, "source_path", getattr(source, "path", None))
                source_sha256 = getattr(
                    source, "source_sha256", getattr(source, "sha256", None)
                )
                source = getattr(source, "rows", source)
        if not str(source_path):
            raise ValueError("authenticated score source requires a path or identifier")
        if source_sha256 is None:
            raise ValueError("authenticated score source requires a SHA-256")
        if batch_size <= 0:
            raise ValueError("seed batch_size must be positive")
        source_digest = _hash_blob(source_sha256)
        row_count = 0
        logical_hasher = hashlib.sha256()
        rows: list[_ScoredPair] = []
        for raw in source:
            row = self._normalize_scored(raw)
            row_count += 1
            logical_hasher.update(
                _canonical_json(
                    [
                        row.pair.key.hex(),
                        row.pair.query_sha256.hex(),
                        row.pair.text_sha256.hex(),
                        row.score.hex(),
                    ]
                )
                + b"\n"
            )
            rows.append(row)
        logical_digest = logical_hasher.digest()
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                "SELECT source_row_count, inserted_count, logical_digest, authorization_sha256 "
                "FROM imports WHERE source_sha256 = ?",
                (source_digest,),
            ).fetchone()
            if existing is not None:
                if int(existing[0]) != row_count or bytes(existing[2]) != logical_digest:
                    raise ValueError("score source conflict")
                connection.commit()
                return int(existing[1])
            inserted = 0
            for offset in range(0, len(rows), batch_size):
                for row in rows[offset : offset + batch_size]:
                    inserted += self._insert_score(connection, row, claim=None)
            connection.execute(
                "INSERT INTO imports(source_sha256, source_path, source_row_count, inserted_count, logical_digest, authorization_sha256) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                (source_digest, str(source_path), row_count, inserted, logical_digest),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return inserted

    @staticmethod
    def _score_rows_logical_digest(rows: Sequence[_ScoredPair]) -> bytes:
        """Hash normalized score rows without depending on their source format."""
        logical_hasher = hashlib.sha256()
        for row in sorted(rows, key=lambda item: item.pair.key):
            logical_hasher.update(
                _canonical_json(
                    [
                        row.pair.key.hex(),
                        row.pair.query_sha256.hex(),
                        row.pair.text_sha256.hex(),
                        row.score.hex(),
                    ]
                )
                + b"\n"
            )
        return logical_hasher.digest()

    def import_scores(
        self,
        source: Iterable[Any],
        *,
        source_path: str | Path,
        source_sha256: str | bytes,
        authorization_sha256: str | bytes | None = None,
        expected_row_count: int | None = None,
        expected_logical_digest: str | bytes | None = None,
    ) -> dict[str, object]:
        """Import normalized score rows in one durable transaction.

        The primitive deliberately knows nothing about JSONL or any producer
        artifact.  The caller supplies normalized rows and an authenticated
        source identity.  All source iteration and duplicate/conflict checks
        happen before the destination transaction; publication of scores and
        their internal import receipt is one SQLite transaction.
        """
        if not str(source_path):
            raise ValueError("score import requires a source path or identifier")
        source_digest = _hash_blob(source_sha256)
        authorization_digest = (
            _hash_blob(authorization_sha256) if authorization_sha256 is not None else None
        )
        normalized: list[_ScoredPair] = []
        by_key: dict[bytes, _ScoredPair] = {}
        raw_row_count = 0
        for raw in source:
            raw_row_count += 1
            row = self._normalize_scored(raw)
            prior = by_key.get(row.pair.key)
            if prior is not None:
                if (
                    prior.pair.query_sha256 != row.pair.query_sha256
                    or prior.pair.text_sha256 != row.pair.text_sha256
                    or prior.score != row.score
                ):
                    raise ValueError("conflicting score import row")
                continue
            by_key[row.pair.key] = row
            normalized.append(row)

        row_count = raw_row_count
        duplicate_count = row_count - len(normalized)
        if expected_row_count is not None and expected_row_count != row_count:
            raise ValueError(
                f"score import row count mismatch: expected {expected_row_count}, found {row_count}"
            )
        logical_digest = self._score_rows_logical_digest(normalized)
        if expected_logical_digest is not None:
            expected_digest = _hash_blob(expected_logical_digest)
            if expected_digest != logical_digest:
                raise ValueError("score import logical digest mismatch")

        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        inserted_keys: list[bytes] = []
        try:
            existing = connection.execute(
                "SELECT source_row_count, inserted_count, logical_digest, authorization_sha256 "
                "FROM imports WHERE source_sha256 = ?",
                (source_digest,),
            ).fetchone()
            if existing is not None:
                existing_authorization = (
                    bytes(existing[3]) if existing[3] is not None else None
                )
                if (
                    int(existing[0]) != row_count
                    or bytes(existing[2]) != logical_digest
                    or existing_authorization != authorization_digest
                ):
                    raise ValueError("score import source conflict")
                connection.commit()
                return {
                    "source_path": str(source_path),
                    "source_sha256": source_digest.hex(),
                    "source_row_count": row_count,
                    "unique_row_count": len(normalized),
                    "duplicate_count": duplicate_count,
                    "inserted_count": 0,
                    "already_identical_count": len(normalized),
                    "logical_digest": logical_digest.hex(),
                    "authorization_sha256": (
                        authorization_digest.hex()
                        if authorization_digest is not None
                        else None
                    ),
                    "receipt_reused": True,
                    "_inserted_keys": (),
                    "_new_import": False,
                }

            for row in normalized:
                if self._insert_score(connection, row, claim=None):
                    inserted_keys.append(row.pair.key)
            connection.execute(
                "INSERT INTO imports(source_sha256, source_path, source_row_count, inserted_count, logical_digest, authorization_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source_digest,
                    str(source_path),
                    row_count,
                    len(inserted_keys),
                    logical_digest,
                    authorization_digest,
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return {
            "source_path": str(source_path),
            "source_sha256": source_digest.hex(),
            "source_row_count": row_count,
            "unique_row_count": len(normalized),
            "duplicate_count": duplicate_count,
            "inserted_count": len(inserted_keys),
            "already_identical_count": len(normalized) - len(inserted_keys),
            "logical_digest": logical_digest.hex(),
            "authorization_sha256": (
                authorization_digest.hex() if authorization_digest is not None else None
            ),
            "receipt_reused": False,
            "_inserted_keys": tuple(inserted_keys),
            "_new_import": True,
        }

    def _rollback_score_import(self, receipt: dict[str, object]) -> None:
        """Undo only rows published by one import after receipt publication fails."""
        if not receipt.get("_new_import"):
            return
        source_digest = _hash_blob(str(receipt["source_sha256"]))
        keys = tuple(receipt.get("_inserted_keys", ()))
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            for key in keys:
                connection.execute("DELETE FROM scores WHERE key_sha256 = ?", (key,))
            connection.execute("DELETE FROM imports WHERE source_sha256 = ?", (source_digest,))
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def add_many(self, rows: Iterable[tuple[str, str, float]]) -> int:
        materialized = list(rows)
        if not materialized:
            return 0
        return self._seed_batch([self._normalize_scored(row) for row in materialized])

    def _claim_keys(
        self,
        pairs: Sequence[_Pair],
        owner: str,
        token: bytes,
        lease_seconds: float,
    ) -> list[_Claim]:
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        now = time.time()
        claimed: list[_Claim] = []
        try:
            for pair in pairs:
                score = connection.execute(
                    "SELECT query_sha256, text_sha256 FROM scores WHERE key_sha256 = ?",
                    (pair.key,),
                ).fetchone()
                if score is not None:
                    if bytes(score[0]) != pair.query_sha256 or bytes(score[1]) != pair.text_sha256:
                        raise ValueError("score-cache key/hash mismatch")
                    continue
                existing = connection.execute(
                    "SELECT query_sha256, text_sha256, lease_expires_at FROM claims WHERE key_sha256 = ?",
                    (pair.key,),
                ).fetchone()
                if existing is not None:
                    if bytes(existing[0]) != pair.query_sha256 or bytes(existing[1]) != pair.text_sha256:
                        raise ValueError("score-cache key/hash mismatch")
                    if float(existing[2]) > now:
                        continue
                    connection.execute(
                        "UPDATE claims SET query_sha256 = ?, text_sha256 = ?, owner = ?, token = ?, claimed_at = ?, lease_expires_at = ? WHERE key_sha256 = ?",
                        (pair.query_sha256, pair.text_sha256, owner, token, now, now + lease_seconds, pair.key),
                    )
                else:
                    connection.execute(
                        "INSERT INTO claims(key_sha256, query_sha256, text_sha256, owner, token, claimed_at, lease_expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (pair.key, pair.query_sha256, pair.text_sha256, owner, token, now, now + lease_seconds),
                    )
                claimed.append(_Claim(pair, owner, token))
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return claimed

    def _heartbeat(self, owner: str, token: bytes, lease_seconds: float) -> None:
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "UPDATE claims SET lease_expires_at = ? WHERE owner = ? AND token = ?",
                (time.time() + lease_seconds, owner, token),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def _release(self, owner: str, token: bytes) -> None:
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DELETE FROM claims WHERE owner = ? AND token = ?", (owner, token))
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def score_many(
        self,
        pairs: Iterable[Any],
        compute_batch: Callable[[Sequence[Any]], Sequence[float]],
        batch_size: int,
        lease_seconds: float = 120,
        *,
        _stats: dict[str, int] | None = None,
    ) -> list[float]:
        if batch_size <= 0:
            raise ValueError("score batch_size must be positive")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        unique, ordered = self._normalize_pairs(pairs)
        if not ordered:
            return []
        scores = self._lookup_normalized(unique)
        if _stats is not None:
            _stats["cache_hits"] = len(scores)
        pending = {pair.key: pair for pair in unique if pair.key not in scores}
        owner = f"{self._owner_prefix}:{uuid4().hex}"
        token = secrets.token_bytes(_CLAIM_TOKEN_BYTES)
        stop = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        wait_seconds = 0.05

        def heartbeat_loop() -> None:
            interval = max(0.001, min(1.0, lease_seconds / 3.0))
            while not stop.wait(interval):
                try:
                    self._heartbeat(owner, token, lease_seconds)
                except sqlite3.Error:
                    continue

        try:
            while pending:
                pending_pairs = list(pending.values())[:batch_size]
                claims = self._claim_keys(
                    pending_pairs,
                    owner,
                    token,
                    lease_seconds,
                )
                if claims:
                    if heartbeat_thread is None:
                        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
                        heartbeat_thread.start()
                    self._heartbeat(owner, token, lease_seconds)
                    batch_claims = claims
                    predicted = list(
                        compute_batch([claim.pair.raw for claim in batch_claims])
                    )
                    if len(predicted) != len(batch_claims):
                        raise ValueError("compute_batch returned the wrong number of scores")
                    connection = self.connection
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        for claim, score in zip(batch_claims, predicted, strict=True):
                            value = _finite_score(score, source="global score cache score")
                            self._insert_score(
                                connection,
                                _ScoredPair(claim.pair, value),
                                claim=claim,
                            )
                            scores[claim.pair.key] = value
                            pending.pop(claim.pair.key, None)
                        connection.commit()
                    except BaseException:
                        connection.rollback()
                        raise
                refreshed = self._lookup_normalized(list(pending.values()))
                scores.update(refreshed)
                for key in refreshed:
                    pending.pop(key, None)
                if pending and not claims:
                    time.sleep(wait_seconds)
                    wait_seconds = min(wait_seconds * 2.0, 1.0)
                else:
                    wait_seconds = 0.05
            return [scores[pair.key] for pair in ordered]
        finally:
            stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=max(1.0, lease_seconds))
            try:
                self._release(owner, token)
            except sqlite3.Error:
                pass

    def logical_binding(self, pairs: Iterable[Any] | None = None) -> str:
        if pairs is None:
            rows = self.connection.execute(
                "SELECT key_sha256, query_sha256, text_sha256, score FROM scores"
            )
            digests = [
                hashlib.sha256(
                    bytes(key)
                    + bytes(query_hash)
                    + bytes(text_hash)
                    + _finite_score(score, source="stored score").hex().encode()
                ).digest()
                for key, query_hash, text_hash, score in rows
            ]
        else:
            unique, _ordered = self._normalize_pairs(pairs)
            found = self._lookup_normalized(unique)
            digests = []
            for pair in unique:
                if pair.key not in found:
                    raise KeyError(f"missing score for {pair.key.hex()}")
                digests.append(
                    hashlib.sha256(
                        pair.key
                        + pair.query_sha256
                        + pair.text_sha256
                        + float(found[pair.key]).hex().encode()
                    ).digest()
                )
        return hashlib.sha256(
            bytes.fromhex(self.context_sha256) + b"".join(sorted(digests))
        ).hexdigest()

    def _legacy_import_row(
        self,
        row: dict[str, Any],
        *,
        legacy_context: ScoreCacheContext | None,
    ) -> _ScoredPair:
        if row.get("schema_version") != self.schema_version:
            raise ValueError("legacy score JSONL schema mismatch")
        if legacy_context is None:
            raise ValueError(
                "legacy score JSONL import requires explicit legacy context authorization"
            )
        for field, expected in legacy_context.artifact_metadata.items():
            if field in row and row[field] != expected:
                raise ValueError(f"legacy score JSONL context mismatch for {field}")
        if "context_sha256" in row and row["context_sha256"] != legacy_context.context_sha256:
            raise ValueError("legacy score JSONL context mismatch for context_sha256")
        for field in _LEGACY_IDENTITY_FIELDS:
            expected = getattr(legacy_context, field)
            if field not in row or row[field] != expected:
                raise ValueError(f"legacy score JSONL context mismatch for {field}")
        query_hash = _hash_blob(row["query_sha256"])
        text_hash = _hash_blob(row["text_sha256"])
        supplied_key = str(row.get("cache_key", ""))
        if supplied_key not in {
            self._legacy_key_from_hashes(query_hash, text_hash, context=legacy_context),
            self._key_from_hashes(query_hash, text_hash).hex(),
        }:
            raise ValueError("legacy score JSONL key/hash mismatch")
        raw = {
            "query_sha256": query_hash,
            "text_sha256": text_hash,
            "key": self._key_from_hashes(query_hash, text_hash),
        }
        return self._normalize_scored({**raw, "score": row.get("score")})

    def _logical_digest_from_connection(
        self,
        connection: sqlite3.Connection,
        *,
        table: str = "scores",
    ) -> bytes:
        if table == "scores":
            query = (
                "SELECT key_sha256, query_sha256, text_sha256, score "
                "FROM scores ORDER BY key_sha256"
            )
        elif table == "temp.legacy_import_staging":
            query = (
                "SELECT key_sha256, query_sha256, text_sha256, MIN(score) "
                "FROM temp.legacy_import_staging "
                "GROUP BY key_sha256, query_sha256, text_sha256 ORDER BY key_sha256"
            )
        else:  # pragma: no cover - only internal table names are supported
            raise ValueError(f"unsupported logical digest table: {table}")
        logical_hasher = hashlib.sha256()
        logical_hasher.update(bytes.fromhex(self.context_sha256))
        for key, query_hash, text_hash, score in connection.execute(query):
            logical_hasher.update(
                hashlib.sha256(
                    bytes(key)
                    + bytes(query_hash)
                    + bytes(text_hash)
                    + _finite_score(score, source="stored score").hex().encode()
                ).digest()
            )
        return logical_hasher.digest()

    @staticmethod
    def _validate_legacy_staging_duplicates(connection: sqlite3.Connection) -> None:
        conflict = connection.execute(
            "SELECT 1 FROM temp.legacy_import_staging "
            "GROUP BY key_sha256 "
            "HAVING COUNT(DISTINCT query_sha256 || text_sha256) > 1 "
            "OR COUNT(DISTINCT score) > 1 LIMIT 1"
        ).fetchone()
        if conflict is not None:
            raise ValueError("legacy import contains conflicting duplicate scores")

    @staticmethod
    def _validate_legacy_target_conflicts(connection: sqlite3.Connection) -> None:
        conflict = connection.execute(
            "SELECT 1 "
            "FROM ("
            "  SELECT key_sha256, query_sha256, text_sha256, MIN(score) AS score "
            "  FROM temp.legacy_import_staging "
            "  GROUP BY key_sha256, query_sha256, text_sha256"
            ") AS staged "
            "JOIN scores AS existing ON existing.key_sha256 = staged.key_sha256 "
            "WHERE existing.query_sha256 != staged.query_sha256 "
            "OR existing.text_sha256 != staged.text_sha256 "
            "OR existing.score != staged.score LIMIT 1"
        ).fetchone()
        if conflict is not None:
            raise ValueError("legacy import conflicts with an existing score")

    def import_legacy_jsonl(
        self,
        path: Path,
        *,
        legacy_context: ScoreCacheContext | None = None,
    ) -> dict[str, object]:
        """Import one old JSONL file through an authorized, atomic promotion."""
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if legacy_context is None:
            raise ValueError(
                "legacy score JSONL import requires explicit legacy context authorization"
            )
        self._validate_legacy_context_rebinding(legacy_context)
        authorization_sha256 = self._authorization_digest(legacy_context)
        connection = self.connection
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute("DROP TABLE IF EXISTS temp.legacy_import_staging")
        connection.execute(
            "CREATE TEMP TABLE legacy_import_staging ("
            "key_sha256 BLOB NOT NULL CHECK(length(key_sha256) = 32), "
            "query_sha256 BLOB NOT NULL CHECK(length(query_sha256) = 32), "
            "text_sha256 BLOB NOT NULL CHECK(length(text_sha256) = 32), "
            "score REAL NOT NULL CHECK("
            "typeof(score) = 'real' AND score = score "
            "AND abs(score) <= 1.7976931348623157e+308)"
            ") STRICT"
        )
        source_hasher = hashlib.sha256()
        row_count = 0
        try:
            connection.execute("BEGIN")
            try:
                batch: list[tuple[bytes, bytes, bytes, float]] = []
                with source_path.open("rb") as source:
                    for line_number, raw_line in enumerate(source, start=1):
                        source_hasher.update(raw_line)
                        if not raw_line.strip():
                            continue
                        try:
                            row = json.loads(raw_line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise ValueError(
                                f"{source_path}:{line_number}: invalid legacy JSONL row"
                            ) from exc
                        if not isinstance(row, dict):
                            raise ValueError(
                                f"{source_path}:{line_number}: legacy row must be an object"
                            )
                        row_count += 1
                        item = self._legacy_import_row(row, legacy_context=legacy_context)
                        batch.append(
                            (
                                item.pair.key,
                                item.pair.query_sha256,
                                item.pair.text_sha256,
                                item.score,
                            )
                        )
                        if len(batch) >= 512:
                            connection.executemany(
                                "INSERT INTO temp.legacy_import_staging "
                                "(key_sha256, query_sha256, text_sha256, score) "
                                "VALUES (?, ?, ?, ?)",
                                batch,
                            )
                            batch.clear()
                if batch:
                    connection.executemany(
                        "INSERT INTO temp.legacy_import_staging "
                        "(key_sha256, query_sha256, text_sha256, score) VALUES (?, ?, ?, ?)",
                        batch,
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

            self._validate_legacy_staging_duplicates(connection)
            source_digest = source_hasher.digest()
            logical_digest = self._logical_digest_from_connection(
                connection,
                table="temp.legacy_import_staging",
            )

            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT source_row_count, inserted_count, logical_digest, authorization_sha256 "
                    "FROM imports WHERE source_sha256 = ?",
                    (source_digest,),
                ).fetchone()
                if existing is not None:
                    if (
                        int(existing[0]) != row_count
                        or bytes(existing[2]) != logical_digest
                        or existing[3] is None
                        or bytes(existing[3]) != authorization_sha256
                    ):
                        raise ValueError("legacy import source conflict")
                    connection.commit()
                    return self._receipt_result(
                        int(existing[0]),
                        int(existing[1]),
                        bytes(existing[2]),
                        bytes(existing[3]),
                        legacy_context=legacy_context,
                        target_context=self.context,
                    )

                self._validate_legacy_target_conflicts(connection)
                staged_scores_sql = (
                    "SELECT key_sha256, query_sha256, text_sha256, MIN(score) AS score "
                    "FROM temp.legacy_import_staging "
                    "GROUP BY key_sha256, query_sha256, text_sha256"
                )
                inserted = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM ("
                        + staged_scores_sql
                        + ") AS staged "
                        "LEFT JOIN scores AS existing "
                        "ON existing.key_sha256 = staged.key_sha256 "
                        "WHERE existing.key_sha256 IS NULL"
                    ).fetchone()[0]
                )
                connection.execute(
                    "INSERT INTO scores(key_sha256, query_sha256, text_sha256, score) "
                    "SELECT staged.key_sha256, staged.query_sha256, "
                    "staged.text_sha256, staged.score FROM ("
                    + staged_scores_sql
                    + ") AS staged "
                    "LEFT JOIN scores AS existing "
                    "ON existing.key_sha256 = staged.key_sha256 "
                    "WHERE existing.key_sha256 IS NULL ORDER BY staged.key_sha256"
                )
                connection.execute(
                    "INSERT INTO imports(source_sha256, source_path, source_row_count, inserted_count, logical_digest, authorization_sha256) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        source_digest,
                        str(source_path),
                        row_count,
                        inserted,
                        logical_digest,
                        authorization_sha256,
                    ),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            return self._receipt_result(
                row_count,
                inserted,
                logical_digest,
                authorization_sha256,
                legacy_context=legacy_context,
                target_context=self.context,
            )
        finally:
            try:
                connection.execute("DROP TABLE IF EXISTS temp.legacy_import_staging")
            except sqlite3.Error:
                pass

    import_jsonl = import_legacy_jsonl

    def _cache_row(self, query_text: str, text: str, score: float) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.context.backend,
            "model": self.context.model,
            "max_length": self.context.max_length,
            "score_kind": self.context.score_kind,
            **self.context.cache_identity_metadata,
            "context_sha256": self.context_sha256,
            "cache_key": self.cache_key(query_text=query_text, text=text),
            "query_sha256": _sha256_text(query_text),
            "text_sha256": _sha256_text(text),
            "score": _finite_score(score, source="artifact score"),
        }


def global_score_cache_dir(root_dir: Path) -> Path:
    return repo_cache_root(root_dir) / "reranker"


def shared_output_path(root_dir: Path, path: Path) -> Path:
    """Resolve configured score output paths into the shared checkout when possible."""
    shared_root = shared_checkout_root(root_dir) or root_dir
    if not path.is_absolute():
        return shared_root / path
    try:
        return shared_root / path.relative_to(root_dir)
    except ValueError:
        return path


def _validate_artifact_row(
    row: dict[str, Any],
    context: ScoreCacheContext,
    *,
    path: Path,
    line_number: int,
) -> None:
    for field, expected in context.artifact_metadata.items():
        if row.get(field) != expected:
            raise ValueError(
                f"{path}:{line_number}: {field} must be {expected!r}; found {row.get(field)!r}"
            )


def _read_document_scores(
    path: Path,
    *,
    context: ScoreCacheContext | None = None,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    scores: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return scores
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        if context:
            _validate_artifact_row(row, context, path=path, line_number=line_number)
        score = _finite_score(row["score"], source=f"{path}:{line_number}: document score")
        row["score"] = score
        scores[(str(row["topic_id"]), str(row["docid"]))].append(row)
    return dict(scores)


def _read_window_scores(
    path: Path,
    *,
    context: ScoreCacheContext | None = None,
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    scores: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return scores
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        if context:
            _validate_artifact_row(row, context, path=path, line_number=line_number)
        score = _finite_score(row["score"], source=f"{path}:{line_number}: window score")
        key = (str(row["topic_id"]), str(row["docid"]), int(row["chunk_index"]))
        row["score"] = score
        scores[key].append(row)
    return dict(scores)


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    if not materialized:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        for row in materialized:
            sink.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
    return len(materialized)


def _matching_document_rows(
    candidate: RetrievedCandidate,
    rows: Iterable[dict[str, Any]],
    score_cache: GlobalScoreCache,
) -> list[dict[str, Any]]:
    expected_cache_key = score_cache.cache_key(
        query_text=candidate.query_text,
        text=candidate.text,
    )
    query_sha256 = _sha256_text(candidate.query_text)
    text_sha256 = _sha256_text(candidate.text)
    return [
        row
        for row in rows
        if row.get("score_cache_key") == expected_cache_key
        and row.get("query_sha256") == query_sha256
        and row.get("text_sha256") == text_sha256
    ]


def _matching_window_rows(
    candidate: RetrievedCandidate,
    chunk: Any,
    chunk_index: int,
    chunk_count: int,
    rows: Iterable[dict[str, Any]],
    score_cache: GlobalScoreCache,
) -> list[dict[str, Any]]:
    expected_cache_key = score_cache.cache_key(
        query_text=candidate.query_text,
        text=chunk.text,
    )
    return [
        row
        for row in rows
        if row.get("score_cache_key") == expected_cache_key
        and row.get("query_sha256") == _sha256_text(candidate.query_text)
        and row.get("document_text_sha256") == _sha256_text(candidate.text)
        and row.get("text_sha256") == _sha256_text(chunk.text)
        and int(row.get("chunk_index", -1)) == chunk_index
        and int(row.get("chunk_count", -1)) == chunk_count
        and int(row.get("start_char", -1)) == chunk.start_char
        and int(row.get("end_char", -1)) == chunk.end_char
    ]


def _consistent_score(rows: Sequence[dict[str, Any]], *, label: str) -> float | None:
    if not rows:
        return None
    scores = {_finite_score(row["score"], source=f"score for {label}") for row in rows}
    if len(scores) != 1:
        raise ValueError(f"conflicting duplicate scores for {label}")
    return scores.pop()


def _seed_document_score_cache(
    *,
    topic: Topic,
    candidates: list[RetrievedCandidate],
    existing_scores: dict[tuple[str, str], list[dict[str, Any]]],
    score_cache: GlobalScoreCache,
) -> int:
    rows: list[tuple[str, str, float]] = []
    candidates_to_seed: list[tuple[str, str, float]] = []
    for candidate in candidates:
        matches = _matching_document_rows(
            candidate,
            existing_scores.get((topic.id, candidate.docid), []),
            score_cache,
        )
        score = _consistent_score(matches, label=f"topic={topic.id} docid={candidate.docid}")
        if score is not None:
            candidates_to_seed.append((candidate.query_text, candidate.text, score))
    found = score_cache.lookup_many(
        [(query_text, text) for query_text, text, _score in candidates_to_seed]
    )
    rows.extend(
        row
        for row, cached in zip(candidates_to_seed, found, strict=True)
        if cached is None
    )
    return score_cache.add_many(rows)


def _seed_window_score_cache(
    *,
    topic: Topic,
    candidates: list[RetrievedCandidate],
    existing_scores: dict[tuple[str, str, int], list[dict[str, Any]]],
    score_cache: GlobalScoreCache,
    chunker: SemanticTextChunker,
) -> int:
    candidates_to_seed: list[tuple[str, str, float]] = []
    for candidate in candidates:
        chunks = chunker.split_text(candidate.text, document_id=candidate.docid)
        chunk_count = len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            matches = _matching_window_rows(
                candidate,
                chunk,
                chunk_index,
                chunk_count,
                existing_scores.get((topic.id, candidate.docid, chunk_index), []),
                score_cache,
            )
            score = _consistent_score(
                matches,
                label=f"topic={topic.id} docid={candidate.docid} chunk={chunk_index}",
            )
            if score is None:
                continue
            candidates_to_seed.append((candidate.query_text, chunk.text, score))
    found = score_cache.lookup_many(
        [(query_text, text) for query_text, text, _score in candidates_to_seed]
    )
    rows = [
        row
        for row, cached in zip(candidates_to_seed, found, strict=True)
        if cached is None
    ]
    return score_cache.add_many(rows)


def _load_cached_candidates(
    *,
    query: QueryVariant,
    retriever: RetrieverConfig,
    cache_dir: Path,
    index_url: str,
) -> list[RetrievedCandidate]:
    request_key = request_cache_key(retriever, query, index_url=index_url)
    candidate_cache = cache_dir / cache_path(
        query.topic_id,
        query.variant_name,
        retriever.name,
        request_key,
    )
    if not candidate_cache.exists():
        raise FileNotFoundError(
            f"missing retrieval cache for topic={query.topic_id}: {candidate_cache}"
        )
    payload = json.loads(candidate_cache.read_text(encoding="utf-8"))
    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"cache file missing response object: {candidate_cache}")
    return normalize_retrieved_candidates(response, query=query, retriever_name=retriever.name)


def _queries_by_topic(config: PipelineConfig) -> dict[str, QueryVariant]:
    topics = load_topics(config.topics.path, topic_format=config.topics.format)
    variant_configs = [{"name": variant.name, "type": variant.type} for variant in config.query_variants]
    queries: dict[str, QueryVariant] = {}
    for topic in topics:
        variants = build_query_variants(topic, variant_configs=variant_configs)
        if len(variants) != 1:
            raise ValueError("rerank score caching currently expects one query variant per topic")
        queries[topic.id] = variants[0]
    return queries


def _topics(config: PipelineConfig, requested_topic_ids: Sequence[str]) -> list[Topic]:
    topics = load_topics(config.topics.path, topic_format=config.topics.format)
    if requested_topic_ids:
        requested = set(requested_topic_ids)
        topics = [topic for topic in topics if topic.id in requested]
        missing = sorted(requested - {topic.id for topic in topics}, key=_topic_sort_key)
        if missing:
            raise ValueError(f"unknown topic ids: {', '.join(missing)}")
    return topics


def _topic_candidates(
    *,
    config: PipelineConfig,
    topic: Topic,
    query: QueryVariant,
    retriever: RetrieverConfig,
    cache_dir: Path,
    index_url: str,
    limit: int | None,
) -> list[RetrievedCandidate]:
    candidates = _load_cached_candidates(
        query=query,
        retriever=retriever,
        cache_dir=cache_dir,
        index_url=index_url,
    )
    ranked = passthrough_rank(candidates)
    selected = [row for row in ranked if row.topic_id == topic.id]
    if limit is not None:
        selected = selected[:limit]
    return [
        RetrievedCandidate(
            topic_id=row.topic_id,
            variant_name=query.variant_name,
            retriever_name=retriever.name,
            query_text=query.query_text,
            docid=row.docid,
            rank=row.rank,
            score=row.score,
            text=row.text,
        )
        for row in selected
    ]


def _scores_to_list(scores: Any) -> list[float]:
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, float | int):
        return [_finite_score(scores, source="model score")]
    return [_finite_score(score, source="model score") for score in scores]


def _identity(value: Any) -> Any:
    return value


def _predict(
    model: Any,
    pairs: list[tuple[str, str]],
    *,
    batch_size: int,
    score_representation: str,
) -> list[float]:
    if not pairs:
        return []
    predict_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "show_progress_bar": False,
        "convert_to_tensor": True,
    }
    if score_representation == "raw_logits":
        predict_kwargs["activation_fn"] = _identity
    elif score_representation != "model_default":
        raise ValueError(f"unknown score representation: {score_representation}")
    scores = model.predict(pairs, **predict_kwargs)
    if hasattr(scores, "detach"):
        scores = scores.detach().float().cpu()
    return _scores_to_list(scores)


def _choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_cross_encoder(
    model_name: str,
    *,
    revision: str,
    max_length: int,
    device: str,
) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for reranker score caching. "
            "Run code/tools/setup_env.sh, then use .venv/bin/python "
            "-m trec_rag.rerank_score_cache --topics 14 --device cpu"
        ) from exc
    return CrossEncoder(
        model_name,
        revision=revision,
        max_length=max_length,
        device=device,
    )


def _validate_model_dtype(model: Any, expected: str) -> None:
    parameter_dtypes = {
        str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters()
    }
    if parameter_dtypes != {expected}:
        found = ", ".join(sorted(parameter_dtypes)) or "no parameters"
        raise RuntimeError(f"model dtype does not match cache context ({found} != {expected})")


def _document_artifact_row(
    *,
    topic: Topic,
    candidate: RetrievedCandidate,
    score: float,
    score_cache: GlobalScoreCache,
) -> dict[str, Any]:
    return {
        "topic_id": topic.id,
        "docid": candidate.docid,
        "rank": candidate.rank,
        "score": score,
        **score_cache.context.artifact_metadata,
        "query_sha256": _sha256_text(candidate.query_text),
        "text_sha256": _sha256_text(candidate.text),
        "score_cache_key": score_cache.cache_key(
            query_text=candidate.query_text,
            text=candidate.text,
        ),
    }


def _window_artifact_row(
    *,
    topic: Topic,
    candidate: RetrievedCandidate,
    chunk: Any,
    chunk_index: int,
    chunk_count: int,
    score: float,
    score_cache: GlobalScoreCache,
) -> dict[str, Any]:
    return {
        "topic_id": topic.id,
        "docid": candidate.docid,
        "rank": candidate.rank,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "chunk_id": chunk.chunk_id,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "score": score,
        **score_cache.context.artifact_metadata,
        "query_sha256": _sha256_text(candidate.query_text),
        "document_text_sha256": _sha256_text(candidate.text),
        "text_sha256": _sha256_text(chunk.text),
        "score_cache_key": score_cache.cache_key(
            query_text=candidate.query_text,
            text=chunk.text,
        ),
    }


def _score_document_rows(
    *,
    model: Any,
    topic: Topic,
    candidates: list[RetrievedCandidate],
    existing_scores: dict[tuple[str, str], list[dict[str, Any]]],
    batch_size: int,
    score_cache: GlobalScoreCache,
    score_kind: str,
) -> list[dict[str, Any]]:
    if score_kind != score_cache.context.score_kind:
        raise ValueError("document score kind does not match the cache context")
    rows: list[dict[str, Any]] = []
    pending_by_key: dict[str, list[RetrievedCandidate]] = defaultdict(list)
    for candidate in candidates:
        key = (topic.id, candidate.docid)
        matches = _matching_document_rows(
            candidate,
            existing_scores.get(key, []),
            score_cache,
        )
        if _consistent_score(matches, label=f"topic={topic.id} docid={candidate.docid}") is not None:
            continue
        pending_by_key[
            score_cache.cache_key(
                query_text=candidate.query_text,
                text=candidate.text,
            )
        ].append(candidate)
    representatives = [group[0] for group in pending_by_key.values()]
    representative_pairs = [
        (candidate.query_text, candidate.text) for candidate in representatives
    ]
    score_stats: dict[str, int] = {}
    scores = score_cache.score_many(
        representative_pairs,
        lambda batch: _predict(
            model,
            list(batch),
            batch_size=batch_size,
            score_representation=score_cache.context.score_representation,
        ),
        batch_size=batch_size,
        _stats=score_stats,
    )
    for representative, score in zip(representatives, scores, strict=True):
        cache_key = score_cache.cache_key(
            query_text=representative.query_text,
            text=representative.text,
        )
        for candidate in pending_by_key[cache_key]:
            row = _document_artifact_row(
                topic=topic,
                candidate=candidate,
                score=score,
                score_cache=score_cache,
            )
            rows.append(row)
            existing_scores.setdefault((topic.id, candidate.docid), []).append(row)
    print(
        "  document_global_cache_hits="
        f"{score_stats.get('cache_hits', 0)} "
        f"document_model_scores={len(representatives) - score_stats.get('cache_hits', 0)}",
        flush=True,
    )
    return rows


def _score_window_rows(
    *,
    model: Any,
    topic: Topic,
    candidates: list[RetrievedCandidate],
    existing_scores: dict[tuple[str, str, int], list[dict[str, Any]]],
    batch_size: int,
    chunker: SemanticTextChunker,
    score_cache: GlobalScoreCache,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending_by_key: dict[str, list[tuple[RetrievedCandidate, Any, int]]] = defaultdict(list)
    chunk_counts: dict[tuple[str, str], int] = {}
    for candidate in candidates:
        chunks = chunker.split_text(candidate.text, document_id=candidate.docid)
        if not chunks:
            raise ValueError(f"no windows for topic={topic.id} docid={candidate.docid}")
        chunk_count = len(chunks)
        chunk_counts[(topic.id, candidate.docid)] = chunk_count
        for chunk_index, chunk in enumerate(chunks):
            key = (topic.id, candidate.docid, chunk_index)
            matches = _matching_window_rows(
                candidate,
                chunk,
                chunk_index,
                chunk_count,
                existing_scores.get(key, []),
                score_cache,
            )
            if _consistent_score(
                matches,
                label=f"topic={topic.id} docid={candidate.docid} chunk={chunk_index}",
            ) is not None:
                continue
            pending_by_key[
                score_cache.cache_key(
                    query_text=candidate.query_text,
                    text=chunk.text,
                )
            ].append((candidate, chunk, chunk_index))

    representatives = [group[0] for group in pending_by_key.values()]
    representative_pairs = [
        (candidate.query_text, chunk.text)
        for candidate, chunk, _ in representatives
    ]
    score_stats: dict[str, int] = {}
    scores = score_cache.score_many(
        representative_pairs,
        lambda batch: _predict(
            model,
            list(batch),
            batch_size=batch_size,
            score_representation=score_cache.context.score_representation,
        ),
        batch_size=batch_size,
        _stats=score_stats,
    )
    for (representative, representative_chunk, _), score in zip(
        representatives,
        scores,
        strict=True,
    ):
        cache_key = score_cache.cache_key(
            query_text=representative.query_text,
            text=representative_chunk.text,
        )
        for candidate, chunk, chunk_index in pending_by_key[cache_key]:
            key = (topic.id, candidate.docid, chunk_index)
            row = _window_artifact_row(
                topic=topic,
                candidate=candidate,
                chunk=chunk,
                chunk_index=chunk_index,
                chunk_count=chunk_counts[(topic.id, candidate.docid)],
                score=score,
                score_cache=score_cache,
            )
            rows.append(row)
            existing_scores.setdefault(key, []).append(row)
    print(
        "  window_global_cache_hits="
        f"{score_stats.get('cache_hits', 0)} "
        f"window_model_scores={len(representatives) - score_stats.get('cache_hits', 0)}",
        flush=True,
    )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build resumable Mixedbread reranker score JSONL artifacts from cached BM25 candidates."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/rag25_bm25_mixedbread_rerank_v1.yaml"))
    parser.add_argument("--topics", nargs="*", default=[], help="Optional topic ids to score.")
    parser.add_argument("--limit-per-topic", type=int, default=None)
    parser.add_argument("--score-kind", choices=["document", "window", "both"], default="both")
    parser.add_argument("--document-score-path", type=Path, default=None)
    parser.add_argument("--window-score-path", type=Path, default=None)
    parser.add_argument("--document-max-length", type=int, default=32768)
    parser.add_argument("--document-pair-buffer-tokens", type=int, default=512)
    parser.add_argument("--window-max-length", type=int, default=1024)
    parser.add_argument("--chunk-max-characters", type=int, default=3500)
    parser.add_argument("--chunk-overlap-characters", type=int, default=350)
    parser.add_argument("--document-batch-size", type=int, default=1)
    parser.add_argument("--window-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, etc. PyTorch ROCm uses cuda.")
    parser.add_argument("--sleep-between-topics", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_pipeline_config(args.config)
    reranker = config.ranking.reranker
    if not reranker:
        raise ValueError("config must use a cached-artifact reranker")
    if len(config.retrievers) != 1:
        raise ValueError("reranker score caching currently expects one retriever")
    if args.limit_per_topic is not None and args.limit_per_topic <= 0:
        raise ValueError("--limit-per-topic must be positive")
    if args.document_pair_buffer_tokens < 0:
        raise ValueError("--document-pair-buffer-tokens must not be negative")
    if args.window_max_length <= 0 or args.chunk_max_characters <= 0:
        raise ValueError("window max length and chunk max characters must be positive")
    if args.chunk_overlap_characters < 0:
        raise ValueError("chunk overlap characters must not be negative")
    document_model_max_length = args.document_max_length - args.document_pair_buffer_tokens
    if document_model_max_length <= 0:
        raise ValueError("document pair buffer must be smaller than document max length")
    if reranker.artifact_schema_version not in {None, ARTIFACT_SCHEMA_VERSION}:
        raise ValueError(
            f"reranker score caching requires artifact_schema_version: {ARTIFACT_SCHEMA_VERSION}"
        )
    configured_policy = {
        "document_max_length": reranker.document_max_length,
        "document_pair_buffer_tokens": reranker.document_pair_buffer_tokens,
        "window_max_length": reranker.window_max_length,
        "chunk_max_characters": reranker.chunk_max_characters,
        "chunk_overlap_characters": reranker.chunk_overlap_characters,
    }
    actual_policy = {
        "document_max_length": args.document_max_length,
        "document_pair_buffer_tokens": args.document_pair_buffer_tokens,
        "window_max_length": args.window_max_length,
        "chunk_max_characters": args.chunk_max_characters,
        "chunk_overlap_characters": args.chunk_overlap_characters,
    }
    mismatches = [
        f"{field}={actual_policy[field]} (config: {configured})"
        for field, configured in configured_policy.items()
        if configured is not None and configured != actual_policy[field]
    ]
    if mismatches:
        raise ValueError("score-cache CLI policy differs from config: " + ", ".join(mismatches))

    load_repo_env(config.root_dir)
    os.environ.setdefault("INDEX_URL", DEFAULT_INDEX_URL)
    index_url = RemotePyseriniConfig.from_env().index_url
    cache_dir = pipeline_cache_dir(config.root_dir, config.run_id)
    retriever = config.retrievers[0]
    topics = _topics(config, args.topics)
    queries = _queries_by_topic(config)
    candidate_limit = (
        args.limit_per_topic
        if args.limit_per_topic is not None
        else reranker.candidate_depth
    )

    model_name = reranker.model
    model_revision = reranker.model_revision or DEFAULT_MODEL_REVISION
    backend_version = reranker.backend_version or DEFAULT_BACKEND_VERSION
    score_representation = (
        reranker.score_representation or DEFAULT_SCORE_REPRESENTATION
    )
    inference_dtype = reranker.inference_dtype or DEFAULT_INFERENCE_DTYPE
    input_policy = reranker.input_policy or "trec_rag_raw_v2"
    if score_representation != DEFAULT_SCORE_REPRESENTATION:
        raise ValueError("reranker score caching requires score_representation: raw_logits")

    document_score_path = shared_output_path(
        config.root_dir,
        args.document_score_path or reranker.document_score_path,
    )
    window_score_path = shared_output_path(
        config.root_dir,
        args.window_score_path or reranker.window_score_path,
    )
    score_cache_root = global_score_cache_dir(config.root_dir)
    document_score_kind = (
        f"doc_max_{args.document_max_length}_buf{args.document_pair_buffer_tokens}"
    )
    context_kwargs = {
        "backend": "sentence-transformers-cross-encoder",
        "model": model_name,
        "model_revision": model_revision,
        "backend_version": backend_version,
        "score_representation": score_representation,
        "inference_dtype": inference_dtype,
        "input_policy": input_policy,
    }
    document_score_cache = GlobalScoreCache(
        score_cache_root,
        ScoreCacheContext(
            **context_kwargs,
            max_length=document_model_max_length,
            score_kind=document_score_kind,
            requested_max_length=args.document_max_length,
            pair_buffer_tokens=args.document_pair_buffer_tokens,
        ),
    )
    window_score_cache = GlobalScoreCache(
        score_cache_root,
        ScoreCacheContext(
            **context_kwargs,
            max_length=args.window_max_length,
            score_kind="window",
            requested_max_length=args.window_max_length,
            chunk_max_characters=args.chunk_max_characters,
            chunk_overlap_characters=args.chunk_overlap_characters,
        ),
    )
    existing_document_scores = (
        _read_document_scores(document_score_path, context=document_score_cache.context)
        if args.score_kind in {"document", "both"}
        else {}
    )
    existing_window_scores = (
        _read_window_scores(window_score_path, context=window_score_cache.context)
        if args.score_kind in {"window", "both"}
        else {}
    )
    chunker = SemanticTextChunker(
        ChunkingConfig(
            max_characters=args.chunk_max_characters,
            overlap_characters=args.chunk_overlap_characters,
        )
    )
    candidates_by_topic: dict[str, list[RetrievedCandidate]] = {}
    for topic in topics:
        candidates_by_topic[topic.id] = _topic_candidates(
            config=config,
            topic=topic,
            query=queries[topic.id],
            retriever=retriever,
            cache_dir=cache_dir,
            index_url=index_url,
            limit=candidate_limit,
        )

    print(f"cache_dir={cache_dir}", flush=True)
    print(f"document_score_path={document_score_path}", flush=True)
    print(f"window_score_path={window_score_path}", flush=True)
    print(f"global_document_score_cache={document_score_cache.path}", flush=True)
    print(f"global_window_score_cache={window_score_cache.path}", flush=True)
    print(f"topics={','.join(topic.id for topic in topics)}", flush=True)
    print(f"candidate_limit={candidate_limit}", flush=True)
    print(f"model_revision={model_revision}", flush=True)
    print(f"score_representation={score_representation}", flush=True)
    print(f"inference_dtype={inference_dtype}", flush=True)
    print(f"backend_version={backend_version}", flush=True)
    print(f"index_url={index_url}", flush=True)
    if args.dry_run:
        for topic in topics:
            candidates = candidates_by_topic[topic.id]
            missing_docs = sum(
                _consistent_score(
                    _matching_document_rows(
                        candidate,
                        existing_document_scores.get((topic.id, candidate.docid), []),
                        document_score_cache,
                    ),
                    label=f"topic={topic.id} docid={candidate.docid}",
                )
                is None
                for candidate in candidates
            )
            missing_window_chunks = 0
            missing_window_docs = 0
            for candidate in candidates:
                chunks = chunker.split_text(candidate.text, document_id=candidate.docid)
                missing_for_doc = 0
                for chunk_index, chunk in enumerate(chunks):
                    matches = _matching_window_rows(
                        candidate,
                        chunk,
                        chunk_index,
                        len(chunks),
                        existing_window_scores.get(
                            (topic.id, candidate.docid, chunk_index), []
                        ),
                        window_score_cache,
                    )
                    if _consistent_score(
                        matches,
                        label=(
                            f"topic={topic.id} docid={candidate.docid} chunk={chunk_index}"
                        ),
                    ) is None:
                        missing_for_doc += 1
                missing_window_chunks += missing_for_doc
                missing_window_docs += bool(missing_for_doc)
            global_document_hits = sum(
                _consistent_score(
                    _matching_document_rows(
                        candidate,
                        existing_document_scores.get((topic.id, candidate.docid), []),
                        document_score_cache,
                    ),
                    label=f"topic={topic.id} docid={candidate.docid}",
                )
                is None
                and document_score_cache.lookup_many(
                    [(candidate.query_text, candidate.text)]
                )[0]
                is not None
                for candidate in candidates
            )
            print(
                f"DRY topic={topic.id} candidates={len(candidates)} "
                f"missing_document_scores={missing_docs} "
                f"global_document_hits={global_document_hits} "
                f"missing_window_docs={missing_window_docs} "
                f"missing_window_scores={missing_window_chunks}",
                flush=True,
            )
        return 0

    document_seeded = 0
    window_seeded = 0
    for topic in topics:
        candidates = candidates_by_topic[topic.id]
        if args.score_kind in {"document", "both"}:
            document_seeded += _seed_document_score_cache(
                topic=topic,
                candidates=candidates,
                existing_scores=existing_document_scores,
                score_cache=document_score_cache,
            )
        if args.score_kind in {"window", "both"}:
            window_seeded += _seed_window_score_cache(
                topic=topic,
                candidates=candidates,
                existing_scores=existing_window_scores,
                score_cache=window_score_cache,
                chunker=chunker,
            )
    print(
        f"document_global_cache_seeded={document_seeded} "
        f"window_global_cache_seeded={window_seeded}",
        flush=True,
    )

    device = _choose_device(args.device)
    print(f"model={model_name} device={device}", flush=True)

    document_model_required = False
    window_model_required = False
    if args.score_kind in {"document", "both"}:
        document_model_required = any(
            _consistent_score(
                _matching_document_rows(
                    candidate,
                    existing_document_scores.get((topic.id, candidate.docid), []),
                    document_score_cache,
                ),
                label=f"topic={topic.id} docid={candidate.docid}",
            )
            is None
            and document_score_cache.lookup_many(
                [(candidate.query_text, candidate.text)]
            )[0]
            is None
            for topic in topics
            for candidate in candidates_by_topic[topic.id]
        )
    if args.score_kind in {"window", "both"}:
        for topic in topics:
            for candidate in candidates_by_topic[topic.id]:
                chunks = chunker.split_text(candidate.text, document_id=candidate.docid)
                for chunk_index, chunk in enumerate(chunks):
                    matches = _matching_window_rows(
                        candidate,
                        chunk,
                        chunk_index,
                        len(chunks),
                        existing_window_scores.get(
                            (topic.id, candidate.docid, chunk_index), []
                        ),
                        window_score_cache,
                    )
                    if _consistent_score(
                        matches,
                        label=(
                            f"topic={topic.id} docid={candidate.docid} chunk={chunk_index}"
                        ),
                    ) is None and window_score_cache.lookup_many(
                        [(candidate.query_text, chunk.text)]
                    )[0] is None:
                        window_model_required = True
                        break
                if window_model_required:
                    break
            if window_model_required:
                break

    if document_model_required or window_model_required:
        try:
            installed_backend_version = importlib.metadata.version("sentence-transformers")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("sentence-transformers is required for score generation") from exc
        if installed_backend_version != backend_version:
            raise RuntimeError(
                "sentence-transformers version does not match the configured cache context "
                f"({installed_backend_version} != {backend_version})"
            )
    print(
        f"document_model_required={document_model_required} "
        f"window_model_required={window_model_required}",
        flush=True,
    )

    document_model = None
    window_model = None
    try:
        if document_model_required:
            document_model = _load_cross_encoder(
                model_name,
                revision=model_revision,
                max_length=document_model_max_length,
                device=device,
            )
            _validate_model_dtype(document_model, inference_dtype)
        if window_model_required:
            window_model = _load_cross_encoder(
                model_name,
                revision=model_revision,
                max_length=args.window_max_length,
                device=device,
            )
            _validate_model_dtype(window_model, inference_dtype)

        for topic_index, topic in enumerate(topics, start=1):
            candidates = candidates_by_topic[topic.id]
            print(
                f"TOPIC {topic.id} ({topic_index}/{len(topics)}) candidates={len(candidates)}",
                flush=True,
            )
            if args.score_kind in {"document", "both"}:
                rows = _score_document_rows(
                    model=document_model,
                    topic=topic,
                    candidates=candidates,
                    existing_scores=existing_document_scores,
                    batch_size=args.document_batch_size,
                    score_cache=document_score_cache,
                    score_kind=document_score_kind,
                )
                written = _append_jsonl(document_score_path, rows)
                print(f"  document_scores_written={written}", flush=True)
            if args.score_kind in {"window", "both"}:
                rows = _score_window_rows(
                    model=window_model,
                    topic=topic,
                    candidates=candidates,
                    existing_scores=existing_window_scores,
                    batch_size=args.window_batch_size,
                    chunker=chunker,
                    score_cache=window_score_cache,
                )
                written = _append_jsonl(window_score_path, rows)
                print(f"  window_scores_written={written}", flush=True)
            if topic_index < len(topics):
                time.sleep(args.sleep_between_topics)
    finally:
        del document_model
        del window_model
        gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
