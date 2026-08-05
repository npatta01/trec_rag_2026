"""Cache-first extraction of relevance-ranked snippets from one document."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, Literal, Protocol
from uuid import uuid4

from filelock import FileLock

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker, TextChunk, TextChunker
from trec_rag.rerank_score_cache import (
    DEFAULT_BACKEND_VERSION,
    GlobalScoreCache,
    ScoreCacheContext,
    _choose_device,
)
from trec_rag.repo_env import repo_cache_root


CURSOR_SCHEMA_VERSION = 2
RESULT_SCHEMA_VERSION = 3
IMPLEMENTATION_VERSION = 3
DEFAULT_SNIPPET_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
DEFAULT_SNIPPET_MODEL_REVISION = "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
# The model is Qwen2-based with max_position_embeddings=32768, so 512 was our
# cap, not its limit. At 512 a chunk's tail was truncated before scoring while
# still being returned to the agent with citable spans over it, so an agent
# could cite sentences the scorer never read and the score was not a claim
# about them. Measured: an answering sentence at 98% of a 3,500-character chunk
# scored exactly the filler baseline at 512 and recovers fully at 1024.
#
# 1024 covers every chunk this config produces (a 3,500-character chunk reaches
# ~838 tokens), and 2048 measured byte-identical, so more only costs time.
# This buys correctness, not accuracy: ranking quality was unchanged within
# noise and reranking costs about 75% more.
_SNIPPET_MAX_LENGTH = 1024
_SNIPPET_BATCH_SIZE = 32
_SNIPPET_SCORE_KIND = "snippet_relevance_v1"
_SMALL_LLM_SCORE_REPRESENTATION = "json_scalar"
_SMALL_LLM_MAX_LENGTH = 0
_PATH_LOCKS: dict[str, Lock] = {}
_PATH_LOCKS_GUARD = Lock()


class SnippetCacheIntegrityError(RuntimeError):
    """Raised when a cached snippet response is malformed or cannot be trusted."""


class InvalidSnippetCursorError(ValueError):
    """Raised only when a snippet cursor fails request validation."""


def _thread_lock_for(path: Path) -> Lock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, Lock())


def _create_score_cache(
    root_dir: Path, context: ScoreCacheContext
) -> GlobalScoreCache:
    root_dir = Path(root_dir)
    cache_path = root_dir.joinpath(*context.path_parts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with _thread_lock_for(lock_path), FileLock(str(lock_path)):
        return GlobalScoreCache(root_dir, context)


@dataclass(frozen=True)
class SnippetExtractionConfig:
    snippets_per_page: int = 10
    chunk_max_characters: int = 3_500
    chunk_overlap_characters: int = 350
    duplicate_overlap_ratio: float = 0.8

    def __post_init__(self) -> None:
        if (
            isinstance(self.snippets_per_page, bool)
            or not isinstance(self.snippets_per_page, int)
            or self.snippets_per_page <= 0
        ):
            raise ValueError("snippets_per_page must be positive")
        if (
            isinstance(self.chunk_max_characters, bool)
            or not isinstance(self.chunk_max_characters, int)
            or self.chunk_max_characters <= 0
        ):
            raise ValueError("chunk_max_characters must be positive")
        if (
            isinstance(self.chunk_overlap_characters, bool)
            or not isinstance(self.chunk_overlap_characters, int)
            or self.chunk_overlap_characters < 0
        ):
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
    page_index: int
    residual_count: int
    residual_top_score: float | None
    returned_min_score: float | None
    pages_estimated: int

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "focus_query": self.focus_query,
            "snippets": [asdict(snippet) for snippet in self.snippets],
            "next_cursor": self.next_cursor,
            "page_index": self.page_index,
            "residual_count": self.residual_count,
            "residual_top_score": self.residual_top_score,
            "returned_min_score": self.returned_min_score,
            "pages_estimated": self.pages_estimated,
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


def _identity_activation(value: Any) -> Any:
    return value


def _load_local_cross_encoder(
    *, model_name: str, revision: str, max_length: int, device: str
) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for local snippet ranking. "
            "Run code/tools/setup_env.sh and pre-cache the pinned model revision."
        ) from exc
    return CrossEncoder(
        model_name,
        revision=revision,
        max_length=max_length,
        device=device,
        local_files_only=True,
    )


def _finite_score(value: object, *, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{source} score must be finite")
    return float(value)


def _model_scores(values: Any, *, expected_count: int) -> tuple[float, ...]:
    if hasattr(values, "detach"):
        values = values.detach().float().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float, bool)):
        values = [values]
    if not isinstance(values, (list, tuple)):
        raise ValueError("local model scores must be a sequence")
    scores = tuple(_finite_score(value, source="local model") for value in values)
    if len(scores) != expected_count:
        raise ValueError("local model score count must match missing chunks")
    return scores


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _score_cache_input_policy(name: str, **settings: object) -> str:
    return f"{name}|{json.dumps(settings, sort_keys=True, separators=(',', ':'))}"


class LocalMixedbreadSnippetRanker:
    """Lazy, cache-aware local Mixedbread cross-encoder snippet ranker."""

    def __init__(
        self,
        *,
        score_cache_root: Path,
        device: str = "auto",
        model_loader: Callable[..., Any] = _load_local_cross_encoder,
        model_name: str = DEFAULT_SNIPPET_MODEL,
        model_revision: str = DEFAULT_SNIPPET_MODEL_REVISION,
        backend_version: str = DEFAULT_BACKEND_VERSION,
        max_length: int = _SNIPPET_MAX_LENGTH,
        batch_size: int = _SNIPPET_BATCH_SIZE,
    ) -> None:
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a nonblank string")
        for name, value in (("max_length", max_length), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._model_name = model_name
        self._model_revision = model_revision
        self._backend_version = backend_version
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = _choose_device(device)
        self._model_loader = model_loader
        self._model: Any | None = None
        self._model_lock = Lock()
        self._score_cache = _create_score_cache(
            Path(score_cache_root),
            ScoreCacheContext(
                backend="sentence_transformers_cross_encoder",
                backend_version=self._backend_version,
                model=self._model_name,
                model_revision=self._model_revision,
                max_length=self._max_length,
                score_kind=_SNIPPET_SCORE_KIND,
                score_representation="raw_logits",
                input_policy=_score_cache_input_policy(
                    "focus_query_chunk_v1",
                    batch_size=self._batch_size,
                    device=self._device,
                ),
            ),
        )

    @property
    def identity(self) -> Mapping[str, object]:
        return {
            "backend": "sentence_transformers_cross_encoder",
            "backend_version": self._backend_version,
            "model": self._model_name,
            "model_revision": self._model_revision,
            "score_representation": "raw_logits",
            "max_length": self._max_length,
            "batch_size": self._batch_size,
            "device": self._device,
            "implementation_version": 1,
        }

    def rank(
        self, focus_query: str, chunks: Sequence[TextChunk]
    ) -> tuple[ScoredTextChunk, ...]:
        rows = tuple(chunks)
        pairs = tuple((focus_query, chunk.text) for chunk in rows)

        def compute_batch(
            pending: Sequence[tuple[str, str]],
        ) -> Sequence[float]:
            model = self._get_model()
            return _model_scores(
                model.predict(
                    pending,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    activation_fn=_identity_activation,
                ),
                expected_count=len(pending),
            )

        values = self._score_cache.score_many(
            pairs,
            compute_batch,
            batch_size=self._batch_size,
        )
        return tuple(
            ScoredTextChunk(
                chunk,
                _finite_score(value, source="cached local"),
            )
            for chunk, value in zip(rows, values, strict=True)
        )

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            if self._model_loader is _load_local_cross_encoder:
                try:
                    installed_backend_version = version("sentence-transformers")
                except PackageNotFoundError as exc:
                    raise RuntimeError(
                        "sentence-transformers is required for local snippet ranking. "
                        "Run code/tools/setup_env.sh and pre-cache the pinned model revision."
                    ) from exc
                if installed_backend_version != self._backend_version:
                    raise RuntimeError(
                        "installed sentence-transformers version does not match the "
                        "snippet ranker cache identity"
                    )
            self._model = self._model_loader(
                model_name=self._model_name,
                revision=self._model_revision,
                max_length=self._max_length,
                device=self._device,
            )
        return self._model


class SmallLLMSnippetRanker:
    """Cache-aware JSON-only ranker for an explicitly injected small chat model."""

    def __init__(
        self,
        *,
        chat_model: Any,
        score_cache_root: Path,
        model_name: str,
        model_revision: str = "unversioned",
        backend_version: str = "unversioned",
        batch_size: int = 10,
    ) -> None:
        if not callable(getattr(chat_model, "invoke", None)):
            raise ValueError("chat_model must provide invoke")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be nonblank")
        self._chat_model = chat_model
        self._model_name = model_name
        self._model_revision = model_revision
        self._backend_version = backend_version
        self._batch_size = batch_size
        self._score_cache = _create_score_cache(
            Path(score_cache_root),
            ScoreCacheContext(
                backend="small_llm_json",
                backend_version=self._backend_version,
                model=self._model_name,
                model_revision=self._model_revision,
                max_length=_SMALL_LLM_MAX_LENGTH,
                score_kind=_SNIPPET_SCORE_KIND,
                score_representation=_SMALL_LLM_SCORE_REPRESENTATION,
                input_policy=_score_cache_input_policy(
                    "exact_prompt_chunk_record_v2",
                    batch_size=self._batch_size,
                ),
            ),
        )

    @property
    def identity(self) -> Mapping[str, object]:
        return {
            "backend": "small_llm_json",
            "backend_version": self._backend_version,
            "model": self._model_name,
            "model_revision": self._model_revision,
            "score_representation": _SMALL_LLM_SCORE_REPRESENTATION,
            "batch_size": self._batch_size,
            "implementation_version": 2,
        }

    def rank(
        self, focus_query: str, chunks: Sequence[TextChunk]
    ) -> tuple[ScoredTextChunk, ...]:
        rows = tuple(chunks)
        if len({chunk.chunk_id for chunk in rows}) != len(rows):
            raise ValueError("small-LLM ranking requires unique chunk IDs")
        scores: dict[str, float] = {}
        for batch in self._prompt_batches(rows):
            prompt = self._prompt(focus_query, batch)
            records = {
                chunk.chunk_id: _canonical_json(
                    {"chunk_id": chunk.chunk_id, "text": chunk.text}
                )
                for chunk in batch
            }
            pairs = tuple((prompt, records[chunk.chunk_id]) for chunk in batch)
            chunk_by_record = {
                records[chunk.chunk_id]: chunk for chunk in batch
            }

            def compute_batch(
                pending: Sequence[tuple[str, str]],
            ) -> Sequence[float]:
                returned_scores = self._invoke_scores(prompt, batch)
                return [
                    returned_scores[chunk_by_record[text].chunk_id]
                    for _query, text in pending
                ]

            returned = self._score_cache.score_many(
                pairs,
                compute_batch,
                batch_size=self._batch_size,
            )
            scores.update(
                {
                    chunk.chunk_id: _finite_score(value, source="cached small-LLM")
                    for chunk, value in zip(batch, returned, strict=True)
                }
            )
        return tuple(
            sorted(
                (
                    ScoredTextChunk(chunk, scores[chunk.chunk_id])
                    for chunk in rows
                ),
                key=lambda row: (-row.relevance_score, row.chunk.chunk_id),
            )
        )

    def _prompt_batches(
        self, chunks: Sequence[TextChunk]
    ) -> tuple[tuple[TextChunk, ...], ...]:
        by_document: dict[str, list[TextChunk]] = {}
        for chunk in chunks:
            by_document.setdefault(chunk.document_id, []).append(chunk)
        batches: list[tuple[TextChunk, ...]] = []
        for document_chunks in by_document.values():
            maximum = self._batch_size
            if len(document_chunks) > 1:
                maximum = min(maximum, (len(document_chunks) + 1) // 2)
            for offset in range(0, len(document_chunks), maximum):
                batches.append(tuple(document_chunks[offset : offset + maximum]))
        return tuple(batches)

    @staticmethod
    def _prompt(focus_query: str, chunks: Sequence[TextChunk]) -> str:
        return (
            "Score each chunk's relevance to the focus query. Return only strict JSON "
            'matching {"scores":[{"chunk_id":"...","score":0.0}]}.'
            "\n"
            + json.dumps(
                {
                    "focus_query": focus_query,
                    "chunks": [
                        {"chunk_id": chunk.chunk_id, "text": chunk.text}
                        for chunk in chunks
                    ],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    def _invoke_scores(
        self, prompt: str, chunks: Sequence[TextChunk]
    ) -> dict[str, float]:
        reply = self._chat_model.invoke(prompt)
        content = reply if isinstance(reply, str) else getattr(reply, "content", None)
        if not isinstance(content, str):
            raise ValueError("small-LLM score response must be JSON text")
        try:
            payload = json.loads(
                content,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("small-LLM score response must be valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"scores"} or not isinstance(payload["scores"], list):
            raise ValueError("small-LLM score response must contain scores")
        expected_ids = {chunk.chunk_id for chunk in chunks}
        returned: dict[str, float] = {}
        for item in payload["scores"]:
            if not isinstance(item, dict) or set(item) != {"chunk_id", "score"}:
                raise ValueError("small-LLM score response contains an invalid score")
            chunk_id = item["chunk_id"]
            if not isinstance(chunk_id, str) or chunk_id not in expected_ids or chunk_id in returned:
                raise ValueError("small-LLM score response contains invalid chunk IDs")
            try:
                returned[chunk_id] = _finite_score(item["score"], source="small-LLM")
            except ValueError as exc:
                raise ValueError("small-LLM score response contains an invalid score") from exc
        if set(returned) != expected_ids:
            raise ValueError("small-LLM score response must score every requested chunk")
        return returned


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


def _source_chunk_record(chunk: TextChunk) -> dict[str, object]:
    return {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "text_sha256": _sha256_text(chunk.text),
    }


def _source_chunk_tuple(value: object) -> tuple[str, str, int, int, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "document_id",
        "chunk_id",
        "start_char",
        "end_char",
        "text_sha256",
    }:
        raise ValueError("invalid source chunk record")
    document_id = value["document_id"]
    chunk_id = value["chunk_id"]
    start_char = value["start_char"]
    end_char = value["end_char"]
    text_sha256 = value["text_sha256"]
    if (
        not isinstance(document_id, str)
        or not document_id.strip()
        or not isinstance(chunk_id, str)
        or not chunk_id.strip()
        or isinstance(start_char, bool)
        or not isinstance(start_char, int)
        or isinstance(end_char, bool)
        or not isinstance(end_char, int)
        or start_char < 0
        or end_char <= start_char
        or not isinstance(text_sha256, str)
        or len(text_sha256) != 64
    ):
        raise ValueError("invalid source chunk record")
    return document_id, chunk_id, start_char, end_char, text_sha256


def _scored_chunk_record(row: ScoredTextChunk) -> dict[str, object]:
    return {
        "chunk_id": row.chunk.chunk_id,
        "relevance_score": row.relevance_score,
    }


def _scored_chunk_tuple(value: object) -> tuple[str, float]:
    if not isinstance(value, Mapping) or set(value) != {
        "chunk_id",
        "relevance_score",
    }:
        raise ValueError("invalid scored chunk record")
    chunk_id = value["chunk_id"]
    relevance_score = value["relevance_score"]
    if (
        not isinstance(chunk_id, str)
        or not chunk_id.strip()
        or isinstance(relevance_score, bool)
        or not isinstance(relevance_score, (int, float))
        or not math.isfinite(relevance_score)
    ):
        raise ValueError("invalid scored chunk record")
    return chunk_id, float(relevance_score)


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

    @contextmanager
    def locked(self, identity: Mapping[str, object]) -> Iterator[None]:
        path = self._path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_path(path)
        with _thread_lock_for(lock_path), FileLock(str(lock_path)):
            yield

    def get(
        self, identity: Mapping[str, object]
    ) -> tuple[
        SnippetPage,
        int,
        tuple[tuple[str, str, int, int, str], ...],
        tuple[tuple[str, float], ...],
    ] | None:
        with self.locked(identity):
            return self._get_locked(identity)

    def _get_locked(
        self, identity: Mapping[str, object]
    ) -> tuple[
        SnippetPage,
        int,
        tuple[tuple[str, str, int, int, str], ...],
        tuple[tuple[str, float], ...],
    ] | None:
        path = self._path(identity)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != {
                "identity",
                "response",
                "page_offset",
                "response_sha256",
                "source_chunks",
                "source_chunks_sha256",
                "scored_chunks",
                "scored_chunks_sha256",
            }:
                raise ValueError("unexpected cache fields")
            if payload.get("identity") != dict(identity):
                raise ValueError("identity mismatch")
            response = payload.get("response")
            if not isinstance(response, dict):
                raise ValueError("missing response")
            if payload.get("response_sha256") != _sha256_text(_canonical_json(response)):
                raise ValueError("response digest mismatch")
            source_chunks = payload.get("source_chunks")
            if not isinstance(source_chunks, list):
                raise ValueError("missing source chunks")
            if payload.get("source_chunks_sha256") != _sha256_text(
                _canonical_json(source_chunks)
            ):
                raise ValueError("source chunk digest mismatch")
            source_chunk_records = tuple(
                _source_chunk_tuple(record) for record in source_chunks
            )
            scored_chunks = payload.get("scored_chunks")
            if not isinstance(scored_chunks, list):
                raise ValueError("missing scored chunks")
            if payload.get("scored_chunks_sha256") != _sha256_text(
                _canonical_json(scored_chunks)
            ):
                raise ValueError("scored chunk digest mismatch")
            scored_chunk_records = tuple(
                _scored_chunk_tuple(record) for record in scored_chunks
            )
            page = self._page_from_response(response)
            if (
                page.document_id != identity.get("document_id")
                or page.focus_query != identity.get("focus_query")
            ):
                raise ValueError("response binding mismatch")
            page_offset = self._valid_page_offset(payload.get("page_offset"))
            return page, page_offset, source_chunk_records, scored_chunk_records
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SnippetCacheIntegrityError("invalid snippet cache entry") from exc

    def put(
        self,
        identity: Mapping[str, object],
        page: SnippetPage,
        *,
        page_offset: int,
        source_chunks: Sequence[TextChunk] | None = None,
        scored_chunks: Sequence[ScoredTextChunk] | None = None,
    ) -> None:
        with self.locked(identity):
            self._put_locked(
                identity,
                page,
                page_offset=page_offset,
                source_chunks=source_chunks,
                scored_chunks=scored_chunks,
            )

    def _put_locked(
        self,
        identity: Mapping[str, object],
        page: SnippetPage,
        *,
        page_offset: int,
        source_chunks: Sequence[TextChunk] | None = None,
        scored_chunks: Sequence[ScoredTextChunk] | None = None,
    ) -> None:
        page_offset = self._valid_page_offset(page_offset)
        response = page.as_dict()
        if source_chunks is None:
            source_chunks = tuple(
                TextChunk(
                    document_id=page.document_id,
                    chunk_id=snippet.chunk_id,
                    text=snippet.text,
                    start_char=snippet.start_char,
                    end_char=snippet.end_char,
                )
                for snippet in page.snippets
            )
        source_chunk_records = [
            _source_chunk_record(chunk) for chunk in source_chunks
        ]
        if scored_chunks is None:
            scores_by_chunk_id = {
                snippet.chunk_id: snippet.relevance_score for snippet in page.snippets
            }
            scored_chunks = tuple(
                ScoredTextChunk(chunk, scores_by_chunk_id[chunk.chunk_id])
                for chunk in source_chunks
            )
        scored_chunk_records = [
            _scored_chunk_record(row) for row in scored_chunks
        ]
        payload = {
            "identity": dict(identity),
            "response": response,
            "page_offset": page_offset,
            "response_sha256": _sha256_text(_canonical_json(response)),
            "source_chunks": source_chunk_records,
            "source_chunks_sha256": _sha256_text(
                _canonical_json(source_chunk_records)
            ),
            "scored_chunks": scored_chunk_records,
            "scored_chunks_sha256": _sha256_text(
                _canonical_json(scored_chunk_records)
            ),
        }
        path = self._path(identity)
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
        expected_keys = {
            "document_id",
            "focus_query",
            "snippets",
            "next_cursor",
            "page_index",
            "residual_count",
            "residual_top_score",
            "returned_min_score",
            "pages_estimated",
        }
        if set(response) != expected_keys:
            raise ValueError("unexpected response fields")
        document_id = response["document_id"]
        focus_query = response["focus_query"]
        snippets_value = response["snippets"]
        next_cursor = response["next_cursor"]
        page_index = response["page_index"]
        residual_count = response["residual_count"]
        residual_top_score = response["residual_top_score"]
        returned_min_score = response["returned_min_score"]
        pages_estimated = response["pages_estimated"]
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("invalid document_id")
        if not isinstance(focus_query, str) or not focus_query.strip():
            raise ValueError("invalid focus_query")
        if not isinstance(snippets_value, list):
            raise ValueError("invalid snippets")
        if next_cursor is not None and not isinstance(next_cursor, str):
            raise ValueError("invalid next_cursor")
        for value, source in (
            (page_index, "page_index"),
            (residual_count, "residual_count"),
            (pages_estimated, "pages_estimated"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid {source}")
        for value, source in (
            (residual_top_score, "residual_top_score"),
            (returned_min_score, "returned_min_score"),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"invalid {source}")
        snippets = tuple(SnippetResultCache._snippet_from_dict(row) for row in snippets_value)
        if (residual_top_score is None) != (residual_count == 0):
            raise ValueError("invalid residual score")
        if (returned_min_score is None) != (not snippets):
            raise ValueError("invalid returned score")
        return SnippetPage(
            document_id,
            focus_query,
            snippets,
            next_cursor,
            page_index,
            residual_count,
            None if residual_top_score is None else float(residual_top_score),
            None if returned_min_score is None else float(returned_min_score),
            pages_estimated,
        )

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
        if chunker is None:
            chunker = SemanticTextChunker(
                ChunkingConfig(
                    max_characters=self.config.chunk_max_characters,
                    overlap_characters=self.config.chunk_overlap_characters,
                )
            )
        self._chunker = chunker
        self._ranker_identity = self._validated_ranker_identity(ranker.identity)
        self._chunker_identity = self._validated_chunker_identity(chunker)

    @property
    def ranker(self) -> SnippetRanker:
        """The scorer, so a cross-document pass scores on the same scale."""
        return self._ranker

    @property
    def chunker(self) -> TextChunker:
        """The chunker, so cross-document chunk ids match the per-document ones.

        Sharing this instance is what lets the global per-chunk score cache hit
        across queries instead of scoring identical text twice.
        """
        return self._chunker

    @staticmethod
    def _validated_ranker_identity(value: Mapping[str, object]) -> dict[str, object]:
        identity = dict(value)
        backend = identity.get("backend")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("ranker identity requires a nonblank backend")
        _canonical_json(identity)
        return identity

    @staticmethod
    def _validated_chunker_identity(chunker: TextChunker) -> dict[str, object]:
        if type(chunker) is SemanticTextChunker:
            try:
                backend_version = version("semantic-text-splitter")
            except PackageNotFoundError:
                backend_version = "unavailable"
            identity: dict[str, object] = {
                "backend": "semantic_text_splitter",
                "backend_version": backend_version,
                "implementation": "trec_rag.chunking.SemanticTextChunker",
                "implementation_version": 1,
                "max_characters": chunker.config.max_characters,
                "overlap_characters": chunker.config.overlap_characters,
                "trim": chunker.config.trim,
            }
        else:
            supplied = getattr(chunker, "identity", None)
            if not isinstance(supplied, Mapping):
                raise ValueError("injected chunker requires an explicit chunker identity")
            identity = dict(supplied)
        backend = identity.get("backend")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("chunker identity requires a nonblank backend")
        try:
            _canonical_json(identity)
        except ValueError as exc:
            raise ValueError("chunker identity must be JSON serializable") from exc
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
        binding_identity = self._identity_without_cursor(identity)
        offset = self._decode_and_validate_cursor(cursor, binding_identity)
        with self._result_cache.locked(identity):
            cached = self._result_cache._get_locked(identity)
            if cached is not None:
                (
                    cached_page,
                    cached_offset,
                    source_chunk_records,
                    scored_chunk_records,
                ) = cached
                self._validate_cached_page(
                    cached_page,
                    cached_offset,
                    offset,
                    binding_identity,
                    document_text,
                    source_chunk_records,
                    scored_chunk_records,
                )
                return SnippetExtractionResult(
                    cached_page,
                    "hit",
                    str(self._ranker_identity["backend"]),
                    cached_offset,
                )

            chunks = tuple(
                self._chunker.split_text(document_text, document_id=document_id)
            )
            self._validate_chunks(chunks, document_id, document_text)
            ranked = sorted(
                self._validated_scored_chunks(
                    self._ranker.rank(focus_query, chunks), chunks
                ),
                key=lambda row: (-row.relevance_score, row.chunk.chunk_id),
            )
            deduplicated = self._deduplicate(ranked)
            page = self._page(
                document_id, focus_query, deduplicated, offset, binding_identity
            )
            self._result_cache._put_locked(
                identity,
                page,
                page_offset=offset,
                source_chunks=chunks,
                scored_chunks=ranked,
            )
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
            "chunker": self._chunker_identity,
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
            raise InvalidSnippetCursorError("invalid cursor")

    def _validate_chunks(
        self, chunks: Sequence[TextChunk], document_id: str, document_text: str
    ) -> None:
        chunk_ids: set[str] = set()
        for chunk in chunks:
            if not isinstance(chunk, TextChunk):
                raise ValueError("chunker returned an invalid text chunk")
            if (
                chunk.document_id != document_id
                or not isinstance(chunk.chunk_id, str)
                or not chunk.chunk_id.strip()
                or not chunk.chunk_id.startswith(f"{document_id}:")
                or chunk.chunk_id in chunk_ids
                or not isinstance(chunk.text, str)
                or isinstance(chunk.start_char, bool)
                or not isinstance(chunk.start_char, int)
                or isinstance(chunk.end_char, bool)
                or not isinstance(chunk.end_char, int)
                or chunk.start_char < 0
                or chunk.end_char <= chunk.start_char
                or chunk.end_char > len(document_text)
                or document_text[chunk.start_char : chunk.end_char] != chunk.text
                or len(chunk.text) > self.config.chunk_max_characters
            ):
                raise ValueError("chunker returned an invalid text chunk")
            chunk_ids.add(chunk.chunk_id)

    @staticmethod
    def _validated_scored_chunks(
        scored: Sequence[ScoredTextChunk], chunks: Sequence[TextChunk]
    ) -> tuple[ScoredTextChunk, ...]:
        valid_chunks = set(chunks)
        if len(scored) != len(chunks):
            raise ValueError("ranker must return exactly one score per input chunk")
        seen_chunks: set[TextChunk] = set()
        rows: list[ScoredTextChunk] = []
        for row in scored:
            if not isinstance(row, ScoredTextChunk) or row.chunk not in valid_chunks:
                raise ValueError("ranker returned an unknown text chunk")
            if row.chunk in seen_chunks:
                raise ValueError("ranker must return exactly one score per input chunk")
            score = row.relevance_score
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("ranker relevance scores must be finite")
            rows.append(ScoredTextChunk(row.chunk, float(score)))
            seen_chunks.add(row.chunk)
        if seen_chunks != valid_chunks:
            raise ValueError("ranker must return exactly one score per input chunk")
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
        next_offset = offset + len(selected)
        residual_count = len(ranked) - next_offset
        residual_top_score = (
            ranked[next_offset].relevance_score if residual_count else None
        )
        returned_min_score = (
            min(row.relevance_score for row in selected) if selected else None
        )
        pages_estimated = math.ceil(len(ranked) / self.config.snippets_per_page)
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
        next_cursor = (
            self._encode_cursor(binding_identity, next_offset)
            if residual_count
            else None
        )
        return SnippetPage(
            document_id,
            focus_query,
            snippets,
            next_cursor,
            offset // self.config.snippets_per_page,
            residual_count,
            residual_top_score,
            returned_min_score,
            pages_estimated,
        )

    @staticmethod
    def _cursor_binding_digest(
        binding_identity: Mapping[str, object], next_offset: int
    ) -> str:
        return _sha256_text(
            _canonical_json(
                {
                    "binding_identity": dict(binding_identity),
                    "next_offset": next_offset,
                }
            )
        )

    def _encode_cursor(self, binding_identity: Mapping[str, object], next_offset: int) -> str:
        payload = {
            "schema_version": CURSOR_SCHEMA_VERSION,
            "binding_digest": self._cursor_binding_digest(
                binding_identity, next_offset
            ),
            "next_offset": next_offset,
        }
        return base64.urlsafe_b64encode(_canonical_json(payload).encode("utf-8")).decode("ascii").rstrip("=")

    def _decode_and_validate_cursor(
        self, cursor: str | None, binding_identity: Mapping[str, object]
    ) -> int:
        if cursor is None:
            return 0
        padded = cursor + "=" * (-len(cursor) % 4)
        try:
            encoded_cursor = padded.encode("ascii")
        except UnicodeEncodeError as exc:
            raise InvalidSnippetCursorError("invalid cursor") from exc
        try:
            decoded = base64.b64decode(encoded_cursor, altchars=b"-_", validate=True)
            payload = json.loads(decoded.decode("utf-8"))
            if not isinstance(payload, dict) or set(payload) != {
                "schema_version",
                "binding_digest",
                "next_offset",
            }:
                raise ValueError("invalid cursor fields")
            if payload["schema_version"] != CURSOR_SCHEMA_VERSION:
                raise ValueError("wrong cursor schema")
            next_offset = SnippetResultCache._valid_page_offset(
                payload["next_offset"]
            )
            if payload["binding_digest"] != self._cursor_binding_digest(
                binding_identity, next_offset
            ):
                raise ValueError("wrong cursor binding")
            return next_offset
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError, binascii.Error) as exc:
            raise InvalidSnippetCursorError("invalid cursor") from exc

    def _validate_cached_page(
        self,
        page: SnippetPage,
        page_offset: int,
        request_offset: int,
        binding_identity: Mapping[str, object],
        document_text: str,
        source_chunk_records: Sequence[tuple[str, str, int, int, str]],
        scored_chunk_records: Sequence[tuple[str, float]],
    ) -> None:
        try:
            if request_offset != page_offset:
                raise ValueError("cached page offset does not match request cursor")
            if len(page.snippets) > self.config.snippets_per_page:
                raise ValueError("cached page exceeds configured page size")
            if page.page_index != page_offset // self.config.snippets_per_page:
                raise ValueError("cached page index does not match page offset")
            if (page.next_cursor is not None) != (page.residual_count > 0):
                raise ValueError("cached next cursor does not match residual count")
            if page.pages_estimated != math.ceil(
                (page_offset + len(page.snippets) + page.residual_count)
                / self.config.snippets_per_page
            ):
                raise ValueError("cached page estimate does not match ranking size")
            expected_returned_min_score = (
                min(snippet.relevance_score for snippet in page.snippets)
                if page.snippets
                else None
            )
            if page.returned_min_score != expected_returned_min_score:
                raise ValueError("cached returned score does not match snippets")
            valid_source_chunks: set[tuple[str, str, int, int, str]] = set()
            source_chunk_ids: set[str] = set()
            source_chunks_by_id: dict[str, TextChunk] = {}
            for record in source_chunk_records:
                document_id, chunk_id, start_char, end_char, text_sha256 = record
                if (
                    document_id != page.document_id
                    or not chunk_id.startswith(f"{page.document_id}:")
                    or chunk_id in source_chunk_ids
                    or record in valid_source_chunks
                    or end_char > len(document_text)
                    or end_char - start_char > self.config.chunk_max_characters
                    or _sha256_text(document_text[start_char:end_char])
                    != text_sha256
                ):
                    raise ValueError("cached source chunks do not match document")
                valid_source_chunks.add(record)
                source_chunk_ids.add(chunk_id)
                source_chunks_by_id[chunk_id] = TextChunk(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    text=document_text[start_char:end_char],
                    start_char=start_char,
                    end_char=end_char,
                )
            scored_chunk_ids: set[str] = set()
            ranked_from_cache: list[ScoredTextChunk] = []
            for chunk_id, relevance_score in scored_chunk_records:
                if chunk_id not in source_chunks_by_id or chunk_id in scored_chunk_ids:
                    raise ValueError("cached scores do not match source chunks")
                scored_chunk_ids.add(chunk_id)
                ranked_from_cache.append(
                    ScoredTextChunk(source_chunks_by_id[chunk_id], relevance_score)
                )
            if scored_chunk_ids != source_chunk_ids:
                raise ValueError("cached scores do not cover source chunks")
            ranked_from_cache.sort(
                key=lambda row: (-row.relevance_score, row.chunk.chunk_id)
            )
            expected_page = self._page(
                page.document_id,
                page.focus_query,
                self._deduplicate(ranked_from_cache),
                page_offset,
                binding_identity,
            )
            if page != expected_page:
                raise ValueError("cached page does not match ranked evidence")
            chunk_ids: set[str] = set()
            records: set[tuple[int, int, str]] = set()
            for snippet in page.snippets:
                record = (snippet.start_char, snippet.end_char, snippet.text)
                source_record = (
                    page.document_id,
                    snippet.chunk_id,
                    snippet.start_char,
                    snippet.end_char,
                    _sha256_text(snippet.text),
                )
                if (
                    not snippet.chunk_id.startswith(f"{page.document_id}:")
                    or snippet.chunk_id in chunk_ids
                    or record in records
                    or source_record not in valid_source_chunks
                    or snippet.end_char > len(document_text)
                    or snippet.end_char - snippet.start_char != len(snippet.text)
                    or document_text[snippet.start_char : snippet.end_char]
                    != snippet.text
                    or len(snippet.text) > self.config.chunk_max_characters
                    or not math.isfinite(snippet.relevance_score)
                ):
                    raise ValueError("cached snippet does not match source document")
                if (
                    snippet.start_char == 0
                    and snippet.end_char == len(document_text)
                    and len(document_text) > self.config.chunk_max_characters
                ):
                    raise ValueError("cached complete document exceeds chunk maximum")
                chunk_ids.add(snippet.chunk_id)
                records.add(record)
            if page.next_cursor is not None:
                next_offset = self._decode_and_validate_cursor(
                    page.next_cursor, binding_identity
                )
                if (
                    not page.snippets
                    or len(page.snippets) != self.config.snippets_per_page
                    or next_offset <= page_offset
                    or next_offset != page_offset + len(page.snippets)
                ):
                    raise ValueError("cached next cursor does not follow page")
        except ValueError as exc:
            raise SnippetCacheIntegrityError("invalid snippet cache entry") from exc


def create_default_snippet_extractor(
    root: Path,
    config: SnippetExtractionConfig | None = None,
    device: str = "auto",
) -> RelevantSnippetExtractor:
    """Build the standard cache-first extractor without loading the local model."""
    effective_config = config or SnippetExtractionConfig()
    root = Path(root)
    return RelevantSnippetExtractor(
        ranker=LocalMixedbreadSnippetRanker(
            score_cache_root=repo_cache_root(root) / "reranker" / "score_cache",
            device=device,
        ),
        chunker=SemanticTextChunker(
            ChunkingConfig(
                max_characters=effective_config.chunk_max_characters,
                overlap_characters=effective_config.chunk_overlap_characters,
            )
        ),
        result_cache=SnippetResultCache(
            repo_cache_root(root) / "reranker" / "deepagent_snippets"
        ),
        config=effective_config,
    )
