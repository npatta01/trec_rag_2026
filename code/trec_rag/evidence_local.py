"""Pinned local Mixedbread scoring and MiniLM similarity for facet evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from trec_rag.facet_evidence import SentencePair
from trec_rag.rerank_score_cache import (
    DEFAULT_BACKEND_VERSION,
    DEFAULT_INFERENCE_DTYPE,
    DEFAULT_MODEL_REVISION,
    DEFAULT_SCORE_REPRESENTATION,
    GlobalScoreCache,
    ScoreCacheContext,
    _choose_device,
    _predict,
    _validate_model_dtype,
)


MIXEDBREAD_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
MIXEDBREAD_REVISION = DEFAULT_MODEL_REVISION
BACKEND_VERSION = DEFAULT_BACKEND_VERSION
SCORE_REPRESENTATION = DEFAULT_SCORE_REPRESENTATION
INFERENCE_DTYPE = DEFAULT_INFERENCE_DTYPE
SENTENCE_MAX_LENGTH = 512
SCORE_KIND = "extractive_sentence_v1"
_BACKEND = "sentence-transformers-cross-encoder"
_INPUT_POLICY = "extractive_sentence_pair_v1"

__all__ = [
    "BACKEND_VERSION",
    "INFERENCE_DTYPE",
    "LocalMiniLMSimilarity",
    "MINILM_MODEL",
    "MINILM_REVISION",
    "MIXEDBREAD_MODEL",
    "MIXEDBREAD_REVISION",
    "MixedbreadSentencePairScorer",
    "SCORE_KIND",
    "SCORE_REPRESENTATION",
    "SENTENCE_MAX_LENGTH",
    "minilm_similarity_identity",
    "mixedbread_sentence_scorer_identity",
]


def mixedbread_sentence_scorer_identity() -> dict[str, object]:
    """Return the pinned scorer identity without touching a cache or model."""
    return {
        "model": MIXEDBREAD_MODEL,
        "model_revision": MIXEDBREAD_REVISION,
        "backend_version": BACKEND_VERSION,
        "score_representation": SCORE_REPRESENTATION,
        "inference_dtype": INFERENCE_DTYPE,
        "score_kind": SCORE_KIND,
        "sentence_max_length": SENTENCE_MAX_LENGTH,
    }


def _load_local_cross_encoder(
    model_name: str,
    *,
    revision: str,
    max_length: int,
    device: str,
    local_files_only: bool,
) -> Any:
    if local_files_only is not True:
        raise ValueError("extractive sentence scoring requires local_files_only=True")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for local extractive sentence scoring. "
            "Run code/tools/setup_env.sh and pre-cache the pinned model revision."
        ) from exc
    return CrossEncoder(
        model_name,
        revision=revision,
        max_length=max_length,
        device=device,
        local_files_only=True,
    )


def _reject_boolean_cache_scores(path: Path) -> None:
    if not path.exists():
        return
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # GlobalScoreCache reports the canonical malformed-row error.
        if isinstance(row, dict) and isinstance(row.get("score"), bool):
            raise ValueError(f"{path}:{line_number}: cached score must not be Boolean")


def _contains_boolean(value: Any) -> bool:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


class _LazyPinnedModel:
    def __init__(self, *, device: str, loader: Callable[..., Any]) -> None:
        self._device = device
        self._loader = loader
        self._model: Any | None = None

    def predict(self, pairs: Any, **kwargs: Any) -> Any:
        if self._model is None:
            self._model = self._loader(
                MIXEDBREAD_MODEL,
                revision=MIXEDBREAD_REVISION,
                max_length=SENTENCE_MAX_LENGTH,
                device=_choose_device(self._device),
                local_files_only=True,
            )
            _validate_model_dtype(self._model, INFERENCE_DTYPE)
        scores = self._model.predict(pairs, **kwargs)
        if _contains_boolean(scores):
            raise ValueError("model score must not be Boolean")
        return scores


class MixedbreadSentencePairScorer:
    """Score sentence pairs with a pinned local cross encoder and shared cache."""

    def __init__(
        self,
        *,
        score_cache_root: Path,
        device: str = "auto",
        model_loader: Callable[..., Any] = _load_local_cross_encoder,
        batch_size: int = 32,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        context = ScoreCacheContext(
            backend=_BACKEND,
            model=MIXEDBREAD_MODEL,
            model_revision=MIXEDBREAD_REVISION,
            backend_version=BACKEND_VERSION,
            max_length=SENTENCE_MAX_LENGTH,
            score_kind=SCORE_KIND,
            score_representation=SCORE_REPRESENTATION,
            inference_dtype=INFERENCE_DTYPE,
            input_policy=_INPUT_POLICY,
            requested_max_length=SENTENCE_MAX_LENGTH,
        )
        score_cache_root = Path(score_cache_root)
        _reject_boolean_cache_scores(score_cache_root.joinpath(*context.path_parts))
        self.score_cache = GlobalScoreCache(
            score_cache_root,
            context,
        )
        self._model = _LazyPinnedModel(device=device, loader=model_loader)

    @property
    def identity(self) -> dict[str, object]:
        return mixedbread_sentence_scorer_identity()

    def score_pairs(self, pairs: Sequence[SentencePair]) -> tuple[float, ...]:
        rows = tuple(pairs)
        if any(not isinstance(pair, SentencePair) for pair in rows):
            raise TypeError("pairs must contain SentencePair values")
        missing: dict[str, SentencePair] = {}
        scores: dict[str, float] = {}
        for pair in rows:
            key = self.score_cache.cache_key(query_text=pair.query_text, text=pair.sentence_text)
            score = self.score_cache.get(query_text=pair.query_text, text=pair.sentence_text)
            if score is None:
                missing.setdefault(key, pair)
            else:
                scores[key] = self._finite_score(score)
        representatives = tuple(missing.values())
        for offset in range(0, len(representatives), self.batch_size):
            batch = representatives[offset : offset + self.batch_size]
            predicted = tuple(
                self._finite_score(score)
                for score in _predict(
                    self._model,
                    [(pair.query_text, pair.sentence_text) for pair in batch],
                    batch_size=self.batch_size,
                    score_representation=SCORE_REPRESENTATION,
                )
            )
            if len(predicted) != len(batch):
                raise ValueError("model score count must match sentence pair count")
            self.score_cache.add_many(
                (pair.query_text, pair.sentence_text, score)
                for pair, score in zip(batch, predicted, strict=True)
            )
            scores.update(
                (
                    self.score_cache.cache_key(query_text=pair.query_text, text=pair.sentence_text),
                    score,
                )
                for pair, score in zip(batch, predicted, strict=True)
            )
        return tuple(
            scores[self.score_cache.cache_key(query_text=pair.query_text, text=pair.sentence_text)]
            for pair in rows
        )

    @staticmethod
    def _finite_score(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("sentence score must be finite")
        return float(value)

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def minilm_similarity_identity(*, device: str, batch_size: int) -> dict[str, object]:
    """Return the pinned embedding identity without constructing a model adapter."""
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return {
        "backend": "sentence-transformers",
        "embedding_representation": "l2_normalized_float",
        "local_files_only": True,
        "model": MINILM_MODEL,
        "model_revision": MINILM_REVISION,
        "score_kind": "normalized_vector_cosine",
        "batch_size": batch_size,
        "device": device,
    }


class LocalMiniLMSimilarity:
    """Load a pinned local snapshot lazily and return normalized-vector cosine."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_size: int = 256,
        loader: Callable[..., Any] | None = None,
    ) -> None:
        minilm_similarity_identity(device=device, batch_size=batch_size)
        self._device = _choose_device(device)
        self._batch_size = batch_size
        self._loader = loader
        self._model: Any | None = None

    @property
    def model_identity(self) -> dict[str, object]:
        identity = minilm_similarity_identity(
            device=self._device,
            batch_size=self._batch_size,
        )
        return {
            key: value
            for key, value in identity.items()
            if key not in {"batch_size", "device"}
        }

    @property
    def identity(self) -> dict[str, object]:
        return minilm_similarity_identity(
            device=self._device,
            batch_size=self._batch_size,
        )

    def _load(self) -> Any:
        if self._model is None:
            loader = self._loader
            if loader is None:
                from sentence_transformers import SentenceTransformer

                loader = SentenceTransformer
            self._model = loader(
                MINILM_MODEL,
                revision=MINILM_REVISION,
                local_files_only=True,
                device=self._device,
            )
        return self._model

    def cosine_matrix(self, texts: Sequence[str]) -> Any:
        import numpy as np

        rows = tuple(texts)
        if any(not isinstance(text, str) or not text for text in rows):
            raise ValueError("embedding texts must be non-empty strings")
        if not rows:
            return np.empty((0, 0), dtype=np.float64)
        encoded = self._load().encode(
            rows,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        try:
            vectors = np.asarray(encoded, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding matrix must contain numeric values") from exc
        if vectors.ndim != 2:
            raise ValueError("embedding matrix must be two-dimensional")
        if vectors.shape[0] != len(rows):
            raise ValueError("embedding count must match requested text count")
        if vectors.shape[1] == 0:
            raise ValueError("embedding vectors must have a non-empty shared dimension")
        if not np.isfinite(vectors).all():
            raise ValueError("embedding matrix must be finite")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms == 0.0):
            raise ValueError("embedding vectors must have a finite non-zero norm")
        normalized = vectors / norms
        if not np.isfinite(normalized).all():
            raise ValueError("normalized embedding matrix must be finite")
        cosine = np.matmul(normalized, normalized.T)
        if not np.isfinite(cosine).all():
            raise ValueError("cosine matrix must be finite")
        return np.clip(cosine, -1.0, 1.0)
