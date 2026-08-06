"""Portable immutable cache for deterministic MiniLM similarity matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from uuid import uuid4


SIMILARITY_CACHE_SCHEMA_VERSION = "similarity-cache-entry-v1"
SIMILARITY_CACHE_DIRECTORY = "similarity-cache-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_FIELDS = frozenset(
    {
        "backend",
        "embedding_representation",
        "local_files_only",
        "model",
        "model_revision",
        "score_kind",
    }
)
_ENTRY_FIELDS = frozenset(
    {"schema_version", "cache_key", "identity", "matrix_hex", "matrix_sha256"}
)

__all__ = [
    "SIMILARITY_CACHE_DIRECTORY",
    "SIMILARITY_CACHE_SCHEMA_VERSION",
    "SimilarityCache",
    "SimilarityCacheError",
    "SimilarityCacheIdentity",
    "SimilarityCacheIntegrityError",
    "SimilarityCacheMiss",
    "build_similarity_cache_identity",
]


class SimilarityCacheError(RuntimeError):
    """Base class for deterministic similarity cache failures."""


class SimilarityCacheMiss(SimilarityCacheError):
    """The exact ordered similarity matrix is absent."""


class SimilarityCacheIntegrityError(SimilarityCacheError):
    """A similarity entry is malformed or conflicts with immutable state."""


@dataclass(frozen=True)
class SimilarityCacheIdentity:
    backend: str
    embedding_representation: str
    local_files_only: bool
    model: str
    model_revision: str
    score_kind: str
    text_sha256s: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        values: dict[str, object] = {
            "model_identity": {
                "backend": self.backend,
                "embedding_representation": self.embedding_representation,
                "local_files_only": self.local_files_only,
                "model": self.model,
                "model_revision": self.model_revision,
                "score_kind": self.score_kind,
            },
            "text_sha256s": list(self.text_sha256s),
        }
        model_identity = values["model_identity"]
        assert isinstance(model_identity, dict)
        for field in _MODEL_FIELDS - {"local_files_only"}:
            value = model_identity[field]
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"similarity identity {field} must be non-empty text")
        if self.local_files_only is not True:
            raise ValueError("similarity identity requires local_files_only=True")
        if any(
            not isinstance(value, str) or not _SHA256.fullmatch(value)
            for value in self.text_sha256s
        ):
            raise ValueError("similarity text identities must be lowercase SHA-256 digests")
        return values

    @property
    def cache_key(self) -> str:
        return sha256(_canonical_json(self.as_dict())).hexdigest()


def build_similarity_cache_identity(
    *,
    model_identity: Mapping[str, object],
    texts: Sequence[str],
) -> SimilarityCacheIdentity:
    """Bind one model revision to exact ordered text hashes, without raw text."""
    if not isinstance(model_identity, Mapping) or set(model_identity) != _MODEL_FIELDS:
        raise ValueError("similarity model identity has unexpected or missing fields")
    rows = tuple(texts)
    if any(not isinstance(text, str) or not text for text in rows):
        raise ValueError("similarity texts must be non-empty strings")
    identity = SimilarityCacheIdentity(
        backend=model_identity["backend"],
        embedding_representation=model_identity["embedding_representation"],
        local_files_only=model_identity["local_files_only"],
        model=model_identity["model"],
        model_revision=model_identity["model_revision"],
        score_kind=model_identity["score_kind"],
        text_sha256s=tuple(sha256(text.encode("utf-8")).hexdigest() for text in rows),
    )
    identity.as_dict()
    return identity


class SimilarityCache:
    """Read and create exact finite matrix entries without mutable repair paths."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError("similarity cache root must be a Path")
        self.root = root

    def entry_path(self, identity: SimilarityCacheIdentity) -> Path:
        if not isinstance(identity, SimilarityCacheIdentity):
            raise TypeError("identity must be SimilarityCacheIdentity")
        key = identity.cache_key
        return self.root / SIMILARITY_CACHE_DIRECTORY / key[:2] / f"{key}.json"

    def load(self, identity: SimilarityCacheIdentity) -> tuple[tuple[float, ...], ...]:
        path = self.entry_path(identity)
        try:
            source = path.read_bytes()
        except FileNotFoundError as exc:
            raise SimilarityCacheMiss(
                f"similarity cache miss for {identity.cache_key}"
            ) from exc
        except OSError as exc:
            raise SimilarityCacheIntegrityError(f"similarity cache read failed: {path}") from exc
        return self._decode_entry(source, identity, path)

    def store(
        self,
        identity: SimilarityCacheIdentity,
        matrix: Sequence[Sequence[float]],
    ) -> Path:
        matrix_hex, decoded = _validated_matrix(matrix, len(identity.text_sha256s))
        matrix_bytes = _canonical_json(matrix_hex)
        entry = {
            "schema_version": SIMILARITY_CACHE_SCHEMA_VERSION,
            "cache_key": identity.cache_key,
            "identity": identity.as_dict(),
            "matrix_hex": matrix_hex,
            "matrix_sha256": sha256(matrix_bytes).hexdigest(),
        }
        source = _canonical_json(entry) + b"\n"
        path = self.entry_path(identity)
        if path.exists():
            existing = self._decode_entry(path.read_bytes(), identity, path)
            if existing != decoded or path.read_bytes() != source:
                raise SimilarityCacheIntegrityError(
                    f"similarity cache immutable conflict: {path}"
                )
            return path
        _create_only(path, source)
        existing_source = path.read_bytes()
        existing = self._decode_entry(existing_source, identity, path)
        if existing != decoded or existing_source != source:
            raise SimilarityCacheIntegrityError(f"similarity cache immutable conflict: {path}")
        return path

    @staticmethod
    def _decode_entry(
        source: bytes,
        identity: SimilarityCacheIdentity,
        path: Path,
    ) -> tuple[tuple[float, ...], ...]:
        try:
            value = json.loads(
                source.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_invalid_constant,
            )
            if not isinstance(value, dict) or set(value) != _ENTRY_FIELDS:
                raise ValueError("entry fields differ")
            matrix_hex = value["matrix_hex"]
            if (
                value["schema_version"] != SIMILARITY_CACHE_SCHEMA_VERSION
                or value["cache_key"] != identity.cache_key
                or value["identity"] != identity.as_dict()
                or value["matrix_sha256"] != sha256(_canonical_json(matrix_hex)).hexdigest()
            ):
                raise ValueError("entry identity or digest differs")
            canonical_hex, decoded = _validated_hex_matrix(
                matrix_hex, len(identity.text_sha256s)
            )
            if canonical_hex != matrix_hex or source != _canonical_json(value) + b"\n":
                raise ValueError("entry is not canonical JSON and float.hex data")
            return decoded
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            OverflowError,
            TypeError,
            ValueError,
        ) as exc:
            raise SimilarityCacheIntegrityError(
                f"similarity cache entry is invalid: {path}"
            ) from exc


def _validated_matrix(
    matrix: Sequence[Sequence[float]],
    size: int,
) -> tuple[list[list[str]], tuple[tuple[float, ...], ...]]:
    try:
        rows = tuple(tuple(row) for row in matrix)
    except TypeError as exc:
        raise ValueError("similarity matrix must be a square sequence") from exc
    if len(rows) != size or any(len(row) != size for row in rows):
        raise ValueError("similarity matrix shape must match ordered texts")
    decoded: list[tuple[float, ...]] = []
    encoded: list[list[str]] = []
    for row in rows:
        decoded_row: list[float] = []
        encoded_row: list[str] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("similarity matrix values must be finite numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("similarity matrix values must be finite")
            decoded_row.append(number)
            encoded_row.append(number.hex())
        decoded.append(tuple(decoded_row))
        encoded.append(encoded_row)
    return encoded, tuple(decoded)


def _validated_hex_matrix(
    matrix: object,
    size: int,
) -> tuple[list[list[str]], tuple[tuple[float, ...], ...]]:
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError("similarity matrix shape is invalid")
    decoded: list[tuple[float, ...]] = []
    canonical: list[list[str]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("similarity matrix shape is invalid")
        decoded_row: list[float] = []
        canonical_row: list[str] = []
        for value in row:
            if not isinstance(value, str):
                raise ValueError("similarity matrix values must be float.hex strings")
            number = float.fromhex(value)
            if not math.isfinite(number):
                raise ValueError("similarity matrix values must be finite")
            decoded_row.append(number)
            canonical_row.append(number.hex())
        decoded.append(tuple(decoded_row))
        canonical.append(canonical_row)
    return canonical, tuple(decoded)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _create_only(path: Path, source: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as sink:
            os.fchmod(sink.fileno(), 0o600)
            sink.write(source)
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return
        os.chmod(path, 0o600)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
