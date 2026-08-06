"""Lazy, content-cached Mixedbread scoring for topic passages."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from .chunking import TextChunk
from .rerank_score_cache import (
    DEFAULT_BACKEND_VERSION,
    DEFAULT_INFERENCE_DTYPE,
    DEFAULT_MODEL_REVISION,
    DEFAULT_SCORE_REPRESENTATION,
    GlobalScoreCache,
    ScoreCacheContext,
    _choose_device,
    _validate_model_dtype,
)


MIXEDBREAD_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
MIXEDBREAD_REVISION = DEFAULT_MODEL_REVISION
BACKEND = "sentence-transformers-cross-encoder"
BACKEND_VERSION = DEFAULT_BACKEND_VERSION
SCORE_REPRESENTATION = DEFAULT_SCORE_REPRESENTATION
INFERENCE_DTYPE = DEFAULT_INFERENCE_DTYPE
MAX_LENGTH = 1024
INPUT_POLICY = "topic_passage_query_text_v1"
SCORE_KIND = "topic_passage_relevance_v1"


@dataclass(frozen=True)
class ScoredPassage:
    """One input chunk and its raw Mixedbread relevance logit."""

    chunk: TextChunk
    relevance_score: float


# The two retrieval modes use the same simple row shape without importing one
# orchestration mode into the shared scorer.
ScoredTextChunk = ScoredPassage


def _identity(value: Any) -> Any:
    return value


def _contains_boolean(value: Any) -> bool:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def _model_scores(values: Any, *, expected_count: int) -> tuple[float, ...]:
    """Normalize model output without allowing shape or numeric coercion loss."""
    if _contains_boolean(values):
        raise ValueError("model scores must not contain Boolean values")
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        values = [values]
    if not isinstance(values, (list, tuple)):
        raise ValueError("model scores must be a sequence")
    if len(values) != expected_count:
        raise ValueError("model score count must match missing passage count")
    scores: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("model scores must be finite real numbers")
        if not math.isfinite(value):
            raise ValueError("model scores must be finite")
        scores.append(float(value))
    return tuple(scores)


def _cached_scores(values: Any, *, expected_count: int) -> tuple[float, ...]:
    """Validate every value returned by the shared cache before row materialization."""
    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise ValueError("cached passage scores must be a sequence") from exc
    if len(normalized) != expected_count:
        raise ValueError("cached passage score count must match input passage count")
    scores: list[float] = []
    for value in normalized:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("cached passage scores must be finite real numbers")
        if not math.isfinite(value):
            raise ValueError("cached passage scores must be finite")
        scores.append(float(value))
    return tuple(scores)


def load_pinned_cross_encoder(
    *,
    model_name: str = MIXEDBREAD_MODEL,
    revision: str = MIXEDBREAD_REVISION,
    max_length: int = MAX_LENGTH,
    device: str = "auto",
    local_files_only: bool = True,
) -> Any:
    """Load exactly the repository-pinned offline bfloat16 CrossEncoder."""
    if model_name != MIXEDBREAD_MODEL:
        raise ValueError("passage scorer requires the pinned Mixedbread model")
    if revision != MIXEDBREAD_REVISION:
        raise ValueError("passage scorer requires the pinned Mixedbread revision")
    if max_length != MAX_LENGTH:
        raise ValueError("passage scorer requires max_length=1024")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a nonblank string")
    if local_files_only is not True:
        raise ValueError("passage scoring requires local_files_only=True")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for local topic passage scoring. "
            "Run code/tools/setup_env.sh and pre-cache the pinned model revision."
        ) from exc
    model = CrossEncoder(
        MIXEDBREAD_MODEL,
        revision=MIXEDBREAD_REVISION,
        max_length=MAX_LENGTH,
        device=_choose_device(device),
        local_files_only=True,
    )
    _validate_model_dtype(model, INFERENCE_DTYPE)
    return model


class MixedbreadPassageScorer:
    """Rank topic chunks by raw, uncalibrated CrossEncoder logits with a shared cache."""

    def __init__(
        self,
        score_cache_root: Path,
        device: str = "auto",
        model_loader: Callable[..., Any] = load_pinned_cross_encoder,
        batch_size: int = 8,
        read_only: bool = False,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a nonblank string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not callable(model_loader):
            raise TypeError("model_loader must be callable")
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be a bool")

        self.batch_size = batch_size
        self._device = _choose_device(device)
        self._model_loader = model_loader
        self._model: Any | None = None
        self._model_lock = Lock()
        self._stats_lock = Lock()
        self._stats = {"cache_hits": 0, "cache_misses": 0, "model_batches": 0}
        self.score_cache = GlobalScoreCache(
            Path(score_cache_root),
            ScoreCacheContext(
                backend=BACKEND,
                backend_version=BACKEND_VERSION,
                model=MIXEDBREAD_MODEL,
                model_revision=MIXEDBREAD_REVISION,
                max_length=MAX_LENGTH,
                requested_max_length=MAX_LENGTH,
                score_kind=SCORE_KIND,
                score_representation=SCORE_REPRESENTATION,
                inference_dtype=INFERENCE_DTYPE,
                input_policy=INPUT_POLICY,
            ),
            read_only=read_only,
        )

    @property
    def identity(self) -> dict[str, object]:
        identity = {
            "backend": BACKEND,
            "backend_version": BACKEND_VERSION,
            "model": MIXEDBREAD_MODEL,
            "model_revision": MIXEDBREAD_REVISION,
            "score_representation": SCORE_REPRESENTATION,
            "inference_dtype": INFERENCE_DTYPE,
            "max_length": MAX_LENGTH,
            "batch_size": self.batch_size,
            "device": self._device,
            "input_policy": INPUT_POLICY,
            "implementation_version": 1,
        }
        json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return identity

    def cache_key(self, query_text: str, passage_text: str) -> str:
        return self.score_cache.cache_key(query_text=query_text, text=passage_text)

    @property
    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def rank(self, query_text: str, chunks: Sequence[TextChunk]) -> tuple[ScoredPassage, ...]:
        rows = tuple(chunks)
        if any(not isinstance(chunk, TextChunk) for chunk in rows):
            raise TypeError("chunks must contain TextChunk values")
        pairs = tuple((query_text, chunk.text) for chunk in rows)

        def compute_batch(batch: Sequence[tuple[str, str]]) -> tuple[float, ...]:
            model = self._get_model()
            predicted = model.predict(
                list(batch),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                activation_fn=_identity,
            )
            return _model_scores(predicted, expected_count=len(batch))

        call_stats: dict[str, int] = {}
        try:
            cached = self.score_cache.score_many(
                pairs,
                compute_batch,
                batch_size=self.batch_size,
                _stats=call_stats,
            )
        finally:
            with self._stats_lock:
                for name in self._stats:
                    self._stats[name] += call_stats.get(name, 0)
        scores = _cached_scores(cached, expected_count=len(rows))
        return tuple(
            ScoredPassage(chunk=chunk, relevance_score=score)
            for chunk, score in zip(rows, scores, strict=True)
        )

    def _get_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    model = self._model_loader(
                        model_name=MIXEDBREAD_MODEL,
                        revision=MIXEDBREAD_REVISION,
                        max_length=MAX_LENGTH,
                        device=self._device,
                        local_files_only=True,
                    )
                    _validate_model_dtype(model, INFERENCE_DTYPE)
                    self._model = model
        return self._model


__all__ = [
    "BACKEND",
    "BACKEND_VERSION",
    "INFERENCE_DTYPE",
    "INPUT_POLICY",
    "MAX_LENGTH",
    "MIXEDBREAD_MODEL",
    "MIXEDBREAD_REVISION",
    "MixedbreadPassageScorer",
    "SCORE_KIND",
    "SCORE_REPRESENTATION",
    "ScoredPassage",
    "ScoredTextChunk",
    "load_pinned_cross_encoder",
]
