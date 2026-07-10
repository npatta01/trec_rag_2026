"""Build cached cross-encoder reranker scores from shared retrieval caches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    @property
    def path_parts(self) -> tuple[str, ...]:
        return (
            "schema_v2",
            _slug(self.backend),
            f"backend_{_slug(self.backend_version)}",
            _slug(self.model),
            f"revision_{_slug(self.model_revision)}",
            f"representation_{_slug(self.score_representation)}",
            f"dtype_{_slug(self.inference_dtype)}",
            f"max_length_{self.max_length}",
            f"{_slug(self.score_kind)}.jsonl",
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
        }

    @property
    def artifact_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            **self.cache_identity_metadata,
            "max_length": self.max_length,
            "score_kind": self.score_kind,
            "pair_buffer_tokens": self.pair_buffer_tokens,
        }
        if self.requested_max_length is not None:
            metadata["requested_max_length"] = self.requested_max_length
        if self.chunk_max_characters is not None:
            metadata["chunk_max_characters"] = self.chunk_max_characters
        if self.chunk_overlap_characters is not None:
            metadata["chunk_overlap_characters"] = self.chunk_overlap_characters
        return metadata


class GlobalScoreCache:
    """Content-addressed cross-encoder score cache shared across experiments."""

    schema_version = 2

    def __init__(self, root_dir: Path, context: ScoreCacheContext) -> None:
        self.context = context
        self.path = root_dir.joinpath(*context.path_parts)
        self.scores = self._load()

    def _load(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not self.path.exists():
            return scores
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self.path}:{line_number}: invalid JSONL row") from exc
            if row.get("schema_version") != self.schema_version:
                continue
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError(f"{self.path}:{line_number}: cached score must be finite")
            cache_key = str(row["cache_key"])
            if cache_key in scores and scores[cache_key] != score:
                raise ValueError(
                    f"{self.path}:{line_number}: conflicting duplicate cache score"
                )
            scores[cache_key] = score
        return scores

    def cache_key(self, *, query_text: str, text: str) -> str:
        payload = {
            "schema_version": self.schema_version,
            "backend": self.context.backend,
            "model": self.context.model,
            "max_length": self.context.max_length,
            "score_kind": self.context.score_kind,
            **self.context.cache_identity_metadata,
            "query_sha256": _sha256_text(query_text),
            "text_sha256": _sha256_text(text),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, *, query_text: str, text: str) -> float | None:
        return self.scores.get(self.cache_key(query_text=query_text, text=text))

    def add_many(self, rows: Iterable[tuple[str, str, float]]) -> int:
        materialized: list[dict[str, Any]] = []
        staged_scores: dict[str, float] = {}
        for query_text, text, score in rows:
            if not math.isfinite(score):
                raise ValueError("global score cache only accepts finite scores")
            cache_key = self.cache_key(query_text=query_text, text=text)
            if cache_key in self.scores:
                if self.scores[cache_key] != float(score):
                    raise ValueError("conflicting score for existing global cache key")
                continue
            if cache_key in staged_scores:
                if staged_scores[cache_key] != float(score):
                    raise ValueError("conflicting score for new global cache key")
                continue
            query_hash = _sha256_text(query_text)
            text_hash = _sha256_text(text)
            staged_scores[cache_key] = float(score)
            materialized.append(
                {
                    "schema_version": self.schema_version,
                    "backend": self.context.backend,
                    "model": self.context.model,
                    "max_length": self.context.max_length,
                    "score_kind": self.context.score_kind,
                    **self.context.cache_identity_metadata,
                    "cache_key": cache_key,
                    "query_sha256": query_hash,
                    "text_sha256": text_hash,
                    "score": float(score),
                }
            )
        written = _append_jsonl(self.path, materialized)
        self.scores.update(staged_scores)
        return written


def global_score_cache_dir(root_dir: Path) -> Path:
    return repo_cache_root(root_dir) / "reranker" / "score_cache"


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
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"{path}:{line_number}: document score must be finite")
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
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"{path}:{line_number}: window score must be finite")
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
    scores = {float(row["score"]) for row in rows}
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
    for candidate in candidates:
        matches = _matching_document_rows(
            candidate,
            existing_scores.get((topic.id, candidate.docid), []),
            score_cache,
        )
        score = _consistent_score(matches, label=f"topic={topic.id} docid={candidate.docid}")
        if score is not None and score_cache.get(
            query_text=candidate.query_text,
            text=candidate.text,
        ) is None:
            rows.append((candidate.query_text, candidate.text, score))
    return score_cache.add_many(rows)


def _seed_window_score_cache(
    *,
    topic: Topic,
    candidates: list[RetrievedCandidate],
    existing_scores: dict[tuple[str, str, int], list[dict[str, Any]]],
    score_cache: GlobalScoreCache,
    chunker: SemanticTextChunker,
) -> int:
    rows: list[tuple[str, str, float]] = []
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
            if score_cache.get(query_text=candidate.query_text, text=chunk.text) is None:
                rows.append((candidate.query_text, chunk.text, score))
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
        return [float(scores)]
    return [float(score) for score in scores]


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
    cache_hits = 0
    for candidate in candidates:
        key = (topic.id, candidate.docid)
        matches = _matching_document_rows(
            candidate,
            existing_scores.get(key, []),
            score_cache,
        )
        if _consistent_score(matches, label=f"topic={topic.id} docid={candidate.docid}") is not None:
            continue
        score = score_cache.get(query_text=candidate.query_text, text=candidate.text)
        if score is None:
            pending_by_key[
                score_cache.cache_key(
                    query_text=candidate.query_text,
                    text=candidate.text,
                )
            ].append(candidate)
            continue
        cache_hits += 1
        row = _document_artifact_row(
            topic=topic,
            candidate=candidate,
            score=score,
            score_cache=score_cache,
        )
        rows.append(row)
        existing_scores.setdefault(key, []).append(row)
    representatives = [group[0] for group in pending_by_key.values()]
    for offset in range(0, len(representatives), batch_size):
        batch = representatives[offset : offset + batch_size]
        scores = _predict(
            model,
            [(candidate.query_text, candidate.text) for candidate in batch],
            batch_size=batch_size,
            score_representation=score_cache.context.score_representation,
        )
        score_cache.add_many(
            (candidate.query_text, candidate.text, score)
            for candidate, score in zip(batch, scores, strict=True)
        )
        for representative, score in zip(batch, scores, strict=True):
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
        f"{cache_hits} document_model_scores={len(representatives)}",
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
    cache_hits = 0
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
            score = score_cache.get(query_text=candidate.query_text, text=chunk.text)
            if score is None:
                pending_by_key[
                    score_cache.cache_key(
                        query_text=candidate.query_text,
                        text=chunk.text,
                    )
                ].append((candidate, chunk, chunk_index))
                continue
            cache_hits += 1
            row = _window_artifact_row(
                topic=topic,
                candidate=candidate,
                chunk=chunk,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                score=score,
                score_cache=score_cache,
            )
            rows.append(row)
            existing_scores.setdefault(key, []).append(row)

    representatives = [group[0] for group in pending_by_key.values()]
    pairs = [(candidate.query_text, chunk.text) for candidate, chunk, _ in representatives]
    scores = _predict(
        model,
        pairs,
        batch_size=batch_size,
        score_representation=score_cache.context.score_representation,
    )
    score_cache.add_many(
        (candidate.query_text, chunk.text, score)
        for (candidate, chunk, _), score in zip(representatives, scores, strict=True)
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
        f"  window_global_cache_hits={cache_hits} window_model_scores={len(representatives)}",
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
                and document_score_cache.get(query_text=candidate.query_text, text=candidate.text) is not None
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
            and document_score_cache.get(
                query_text=candidate.query_text,
                text=candidate.text,
            )
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
                    ) is None and window_score_cache.get(
                        query_text=candidate.query_text,
                        text=chunk.text,
                    ) is None:
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
