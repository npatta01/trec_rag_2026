"""Cache-first extraction of relevance-ranked snippets from one document."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from filelock import FileLock

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker, TextChunk, TextChunker


CURSOR_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = 1


class SnippetCacheIntegrityError(RuntimeError):
    """Raised when a cached snippet response is malformed or cannot be trusted."""


@dataclass(frozen=True)
class SnippetExtractionConfig:
    snippets_per_page: int = 10
    chunk_max_characters: int = 3_500
    chunk_overlap_characters: int = 350
    duplicate_overlap_ratio: float = 0.8

    def __post_init__(self) -> None:
        if isinstance(self.snippets_per_page, bool) or self.snippets_per_page <= 0:
            raise ValueError("snippets_per_page must be positive")
        if isinstance(self.chunk_max_characters, bool) or self.chunk_max_characters <= 0:
            raise ValueError("chunk_max_characters must be positive")
        if isinstance(self.chunk_overlap_characters, bool) or self.chunk_overlap_characters < 0:
            raise ValueError("chunk_overlap_characters must be non-negative")
        if self.chunk_overlap_characters >= self.chunk_max_characters:
            raise ValueError("chunk_overlap_characters must be smaller than chunk_max_characters")
        if (
            isinstance(self.duplicate_overlap_ratio, bool)
            or not math.isfinite(self.duplicate_overlap_ratio)
            or not 0.0 <= self.duplicate_overlap_ratio <= 1.0
        ):
            raise ValueError("duplicate_overlap_ratio must be finite and between 0 and 1")


@dataclass(frozen=True)
class ScoredTextChunk:
    chunk: TextChunk
    relevance_score: float


@dataclass(frozen=True)
class RelevantSnippet:
    chunk_id: str
    start_char: int
    end_char: int
    text: str
    relevance_score: float


@dataclass(frozen=True)
class SnippetPage:
    document_id: str
    focus_query: str
    snippets: tuple[RelevantSnippet, ...]
    next_cursor: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "focus_query": self.focus_query,
            "snippets": [asdict(snippet) for snippet in self.snippets],
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True)
class SnippetExtractionResult:
    page: SnippetPage
    cache_status: Literal["hit", "miss"]
    ranker_backend: str
    page_offset: int


class SnippetRanker(Protocol):
    @property
    def identity(self) -> Mapping[str, object]:
        raise NotImplementedError

    def rank(
        self, focus_query: str, chunks: Sequence[TextChunk]
    ) -> tuple[ScoredTextChunk, ...]:
        raise NotImplementedError


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("snippet cache identity must be JSON serializable") from exc


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


class SnippetResultCache:
    """Content-addressed persistent cache for complete, model-visible pages."""

    schema_version = RESULT_SCHEMA_VERSION

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)

    def _digest(self, identity: Mapping[str, object]) -> str:
        return _sha256_text(_canonical_json(identity))

    def _path(self, identity: Mapping[str, object]) -> Path:
        digest = self._digest(identity)
        return self.root_dir / f"schema_v{self.schema_version}" / digest[:2] / f"{digest}.json"

    @staticmethod
    def _lock_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".lock")

    def get(self, identity: Mapping[str, object]) -> tuple[SnippetPage, int] | None:
        path = self._path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self._lock_path(path))):
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or payload.get("identity") != dict(identity):
                    raise ValueError("identity mismatch")
                response = payload.get("response")
                if not isinstance(response, dict):
                    raise ValueError("missing response")
                if payload.get("response_sha256") != _sha256_text(_canonical_json(response)):
                    raise ValueError("response digest mismatch")
                page = self._page_from_response(response)
                if (
                    page.document_id != identity.get("document_id")
                    or page.focus_query != identity.get("focus_query")
                ):
                    raise ValueError("response binding mismatch")
                page_offset = self._valid_page_offset(payload.get("page_offset"))
                return page, page_offset
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SnippetCacheIntegrityError("invalid snippet cache entry") from exc

    def put(self, identity: Mapping[str, object], page: SnippetPage, *, page_offset: int) -> None:
        page_offset = self._valid_page_offset(page_offset)
        response = page.as_dict()
        payload = {
            "identity": dict(identity),
            "response": response,
            "page_offset": page_offset,
            "response_sha256": _sha256_text(_canonical_json(response)),
        }
        path = self._path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self._lock_path(path))):
            temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                temporary_path.write_text(
                    _canonical_json(payload) + "\n", encoding="utf-8"
                )
                with temporary_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary_path, path)
            finally:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _valid_page_offset(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("page_offset must be a non-negative integer")
        return value

    @staticmethod
    def _page_from_response(response: Mapping[str, object]) -> SnippetPage:
        expected_keys = {"document_id", "focus_query", "snippets", "next_cursor"}
        if set(response) != expected_keys:
            raise ValueError("unexpected response fields")
        document_id = response["document_id"]
        focus_query = response["focus_query"]
        snippets_value = response["snippets"]
        next_cursor = response["next_cursor"]
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("invalid document_id")
        if not isinstance(focus_query, str) or not focus_query.strip():
            raise ValueError("invalid focus_query")
        if not isinstance(snippets_value, list):
            raise ValueError("invalid snippets")
        if next_cursor is not None and not isinstance(next_cursor, str):
            raise ValueError("invalid next_cursor")
        snippets = tuple(SnippetResultCache._snippet_from_dict(row) for row in snippets_value)
        return SnippetPage(document_id, focus_query, snippets, next_cursor)

    @staticmethod
    def _snippet_from_dict(value: object) -> RelevantSnippet:
        if not isinstance(value, Mapping) or set(value) != {
            "chunk_id",
            "start_char",
            "end_char",
            "text",
            "relevance_score",
        }:
            raise ValueError("invalid snippet")
        chunk_id = value["chunk_id"]
        start_char = value["start_char"]
        end_char = value["end_char"]
        text = value["text"]
        score = value["relevance_score"]
        if not isinstance(chunk_id, str) or not chunk_id.strip() or not isinstance(text, str):
            raise ValueError("invalid snippet text")
        if (
            isinstance(start_char, bool)
            or not isinstance(start_char, int)
            or isinstance(end_char, bool)
            or not isinstance(end_char, int)
            or start_char < 0
            or end_char <= start_char
        ):
            raise ValueError("invalid snippet range")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("invalid snippet score")
        return RelevantSnippet(chunk_id, start_char, end_char, text, float(score))


class RelevantSnippetExtractor:
    """Extract stable relevance-ranked pages while keeping cache metadata private."""

    def __init__(
        self,
        *,
        ranker: SnippetRanker,
        result_cache: SnippetResultCache,
        config: SnippetExtractionConfig | None = None,
        chunker: TextChunker | None = None,
    ) -> None:
        self._ranker = ranker
        self._result_cache = result_cache
        self.config = config or SnippetExtractionConfig()
        self._chunker = chunker or SemanticTextChunker(
            ChunkingConfig(
                max_characters=self.config.chunk_max_characters,
                overlap_characters=self.config.chunk_overlap_characters,
            )
        )
        self._ranker_identity = self._validated_ranker_identity(ranker.identity)

    @staticmethod
    def _validated_ranker_identity(value: Mapping[str, object]) -> dict[str, object]:
        identity = dict(value)
        backend = identity.get("backend")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("ranker identity requires a nonblank backend")
        _canonical_json(identity)
        return identity

    def extract(
        self,
        document_id: str,
        document_text: str,
        focus_query: str,
        cursor: str | None = None,
    ) -> SnippetExtractionResult:
        self._validate_request(document_id, document_text, focus_query, cursor)
        identity = self._result_identity(document_id, document_text, focus_query, cursor)
        cached = self._result_cache.get(identity)
        if cached is not None:
            cached_page, cached_offset = cached
            return SnippetExtractionResult(
                cached_page,
                "hit",
                str(self._ranker_identity["backend"]),
                cached_offset,
            )

        binding_identity = self._identity_without_cursor(identity)
        offset = self._decode_and_validate_cursor(cursor, binding_identity)
        chunks = tuple(self._chunker.split_text(document_text, document_id=document_id))
        self._validate_chunks(chunks, document_id, document_text)
        ranked = sorted(
            self._validated_scored_chunks(self._ranker.rank(focus_query, chunks), chunks),
            key=lambda row: (-row.relevance_score, row.chunk.chunk_id),
        )
        deduplicated = self._deduplicate(ranked)
        page = self._page(document_id, focus_query, deduplicated, offset, binding_identity)
        self._result_cache.put(identity, page, page_offset=offset)
        return SnippetExtractionResult(
            page,
            "miss",
            str(self._ranker_identity["backend"]),
            offset,
        )

    def _result_identity(
        self, document_id: str, document_text: str, focus_query: str, cursor: str | None
    ) -> dict[str, object]:
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "cursor_schema_version": CURSOR_SCHEMA_VERSION,
            "document_id": document_id,
            "focus_query": focus_query,
            "cursor": cursor,
            "document_sha256": _sha256_text(document_text),
            "snippets_per_page": self.config.snippets_per_page,
            "chunk_max_characters": self.config.chunk_max_characters,
            "chunk_overlap_characters": self.config.chunk_overlap_characters,
            "duplicate_overlap_ratio": self.config.duplicate_overlap_ratio,
            "ranker": self._ranker_identity,
        }

    @staticmethod
    def _identity_without_cursor(identity: Mapping[str, object]) -> dict[str, object]:
        return {key: value for key, value in identity.items() if key != "cursor"}

    @staticmethod
    def _validate_request(
        document_id: object, document_text: object, focus_query: object, cursor: object
    ) -> None:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id is required")
        if not isinstance(document_text, str):
            raise ValueError("document_text must be a string")
        if not isinstance(focus_query, str) or not focus_query.strip():
            raise ValueError("focus_query is required")
        if cursor is not None and not isinstance(cursor, str):
            raise ValueError("cursor must be a string or None")

    @staticmethod
    def _validate_chunks(
        chunks: Sequence[TextChunk], document_id: str, document_text: str
    ) -> None:
        for chunk in chunks:
            if (
                chunk.document_id != document_id
                or not chunk.chunk_id.strip()
                or chunk.start_char < 0
                or chunk.end_char <= chunk.start_char
                or chunk.end_char > len(document_text)
                or document_text[chunk.start_char : chunk.end_char] != chunk.text
            ):
                raise ValueError("chunker returned an invalid text chunk")

    @staticmethod
    def _validated_scored_chunks(
        scored: Sequence[ScoredTextChunk], chunks: Sequence[TextChunk]
    ) -> tuple[ScoredTextChunk, ...]:
        valid_chunks = set(chunks)
        rows: list[ScoredTextChunk] = []
        for row in scored:
            if not isinstance(row, ScoredTextChunk) or row.chunk not in valid_chunks:
                raise ValueError("ranker returned an unknown text chunk")
            score = row.relevance_score
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("ranker relevance scores must be finite")
            rows.append(ScoredTextChunk(row.chunk, float(score)))
        return tuple(rows)

    def _deduplicate(self, ranked: Sequence[ScoredTextChunk]) -> tuple[ScoredTextChunk, ...]:
        selected: list[ScoredTextChunk] = []
        normalized_texts: set[str] = set()
        for row in ranked:
            normalized = " ".join(row.chunk.text.casefold().split())
            if normalized in normalized_texts:
                continue
            if any(self._overlaps_too_much(row.chunk, existing.chunk) for existing in selected):
                continue
            normalized_texts.add(normalized)
            selected.append(row)
        return tuple(selected)

    def _overlaps_too_much(self, left: TextChunk, right: TextChunk) -> bool:
        intersection = max(0, min(left.end_char, right.end_char) - max(left.start_char, right.start_char))
        shorter_span = min(left.end_char - left.start_char, right.end_char - right.start_char)
        return intersection / shorter_span >= self.config.duplicate_overlap_ratio

    def _page(
        self,
        document_id: str,
        focus_query: str,
        ranked: Sequence[ScoredTextChunk],
        offset: int,
        binding_identity: Mapping[str, object],
    ) -> SnippetPage:
        selected = ranked[offset : offset + self.config.snippets_per_page]
        snippets = tuple(
            RelevantSnippet(
                chunk_id=row.chunk.chunk_id,
                start_char=row.chunk.start_char,
                end_char=row.chunk.end_char,
                text=row.chunk.text,
                relevance_score=row.relevance_score,
            )
            for row in selected
        )
        next_offset = offset + len(snippets)
        next_cursor = (
            self._encode_cursor(binding_identity, next_offset)
            if next_offset < len(ranked)
            else None
        )
        return SnippetPage(document_id, focus_query, snippets, next_cursor)

    @staticmethod
    def _cursor_binding_digest(binding_identity: Mapping[str, object]) -> str:
        return _sha256_text(_canonical_json(binding_identity))

    def _encode_cursor(self, binding_identity: Mapping[str, object], next_offset: int) -> str:
        payload = {
            "schema_version": CURSOR_SCHEMA_VERSION,
            "binding_digest": self._cursor_binding_digest(binding_identity),
            "next_offset": next_offset,
        }
        return base64.urlsafe_b64encode(_canonical_json(payload).encode("utf-8")).decode("ascii").rstrip("=")

    def _decode_and_validate_cursor(
        self, cursor: str | None, binding_identity: Mapping[str, object]
    ) -> int:
        if cursor is None:
            return 0
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            decoded = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
            payload = json.loads(decoded.decode("utf-8"))
            if not isinstance(payload, dict) or set(payload) != {
                "schema_version",
                "binding_digest",
                "next_offset",
            }:
                raise ValueError("invalid cursor fields")
            if payload["schema_version"] != CURSOR_SCHEMA_VERSION:
                raise ValueError("wrong cursor schema")
            if payload["binding_digest"] != self._cursor_binding_digest(binding_identity):
                raise ValueError("wrong cursor binding")
            return SnippetResultCache._valid_page_offset(payload["next_offset"])
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError, binascii.Error) as exc:
            raise ValueError("invalid cursor") from exc
