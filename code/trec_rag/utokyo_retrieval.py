"""UTokyo-HitU retrieval primitives for the Topic 213 experiment."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


@dataclass(frozen=True)
class RankedPassage:
    passage_id: str
    rank: int
    score: float


class TextPassage(Protocol):
    passage_id: str
    text: str


def tokenize_for_bm25(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def rank_scores(
    passage_ids: Sequence[str], scores: Sequence[float], *, top_k: int
) -> list[RankedPassage]:
    if len(passage_ids) != len(scores):
        raise ValueError("passage IDs and scores must have the same length")
    if len(set(passage_ids)) != len(passage_ids):
        raise ValueError("passage IDs must be unique")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    order = sorted(
        range(len(passage_ids)),
        key=lambda index: (-float(scores[index]), passage_ids[index]),
    )[:top_k]
    return [
        RankedPassage(
            passage_id=passage_ids[index],
            rank=rank,
            score=float(scores[index]),
        )
        for rank, index in enumerate(order, 1)
    ]


def bm25_rank(
    passages: Sequence[TextPassage], *, query: str, top_k: int
) -> list[RankedPassage]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError(
            "rank-bm25 is required; run `uv sync --group utokyo`"
        ) from exc
    tokenized_corpus = [tokenize_for_bm25(passage.text) for passage in passages]
    model = BM25Okapi(tokenized_corpus)
    scores = model.get_scores(tokenize_for_bm25(query))
    return rank_scores(
        [passage.passage_id for passage in passages], scores, top_k=top_k
    )


def hyde_vector_mix(
    original_vector: Sequence[float],
    hypothetical_vector: Sequence[float],
    *,
    alpha: float,
) -> list[float]:
    if len(original_vector) != len(hypothetical_vector) or not original_vector:
        raise ValueError("HyDE vectors must be nonempty and have equal dimensions")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("HyDE alpha must be between 0 and 1")

    def unit(vector: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(float(value) ** 2 for value in vector))
        if norm == 0.0:
            raise ValueError("HyDE vectors must have nonzero norm")
        return [float(value) / norm for value in vector]

    original = unit(original_vector)
    hypothetical = unit(hypothetical_vector)
    mixed = [
        (1.0 - alpha) * original_value + alpha * hypothetical_value
        for original_value, hypothetical_value in zip(original, hypothetical, strict=True)
    ]
    return unit(mixed)


def reciprocal_rank_fusion(
    streams: Mapping[str, Sequence[RankedPassage]],
    *,
    k: int = 60,
    top_k: int = 1000,
) -> tuple[list[RankedPassage], dict[str, dict[str, object]]]:
    if not streams:
        raise ValueError("RRF requires at least one stream")
    if k < 0 or top_k <= 0:
        raise ValueError("RRF k must be nonnegative and top_k must be positive")
    scores: dict[str, float] = {}
    provenance: dict[str, dict[str, object]] = {}
    for stream_name, stream in streams.items():
        seen: set[str] = set()
        for expected_rank, row in enumerate(stream, 1):
            if row.passage_id in seen:
                raise ValueError(f"duplicate passage in {stream_name}: {row.passage_id}")
            if row.rank != expected_rank:
                raise ValueError(f"non-contiguous ranks in stream {stream_name}")
            seen.add(row.passage_id)
            contribution = 1.0 / (k + row.rank)
            scores[row.passage_id] = scores.get(row.passage_id, 0.0) + contribution
            item = provenance.setdefault(
                row.passage_id,
                {"stream_ranks": {}, "stream_scores": {}, "rrf_contributions": {}},
            )
            item["stream_ranks"][stream_name] = row.rank  # type: ignore[index]
            item["stream_scores"][stream_name] = row.score  # type: ignore[index]
            item["rrf_contributions"][stream_name] = contribution  # type: ignore[index]
    ranked = rank_scores(list(scores), list(scores.values()), top_k=top_k)
    for row in ranked:
        provenance[row.passage_id]["rrf_rank"] = row.rank
        provenance[row.passage_id]["rrf_score"] = row.score
    return ranked, provenance


def sliding_window_rerank(
    passage_ids: Sequence[str],
    *,
    rerank_window: Callable[[Sequence[str], int, int], Sequence[str]],
    window_size: int,
    stride: int,
    num_passes: int,
) -> tuple[list[str], list[dict[str, object]]]:
    if len(set(passage_ids)) != len(passage_ids):
        raise ValueError("sliding-window input IDs must be unique")
    if window_size <= 1 or stride <= 0 or stride >= window_size or num_passes <= 0:
        raise ValueError("invalid sliding-window parameters")
    order = list(passage_ids)
    audit: list[dict[str, object]] = []
    for pass_index in range(1, num_passes + 1):
        end = len(order)
        window_index = 0
        while end > 0:
            start = max(0, end - window_size)
            before = order[start:end]
            window_index += 1
            after = list(rerank_window(before, pass_index, window_index))
            if len(after) != len(before) or set(after) != set(before):
                raise ValueError("reranker must return an exact permutation of its window")
            order[start:end] = after
            audit.append(
                {
                    "pass": pass_index,
                    "window": window_index,
                    "start": start,
                    "end": end,
                    "before": before,
                    "after": after,
                }
            )
            if start == 0:
                break
            end -= stride
    return order, audit


def corpus_sha256(passages: Sequence[TextPassage]) -> str:
    digest = hashlib.sha256()
    for passage in passages:
        digest.update(passage.passage_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(passage.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def score_cache_key(*, corpus_hash: str, query: str, method: Mapping[str, object]) -> str:
    payload = {
        "corpus_sha256": corpus_hash,
        "query": query,
        "method": dict(method),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_score_cache(
    path: Path,
    *,
    expected_passage_ids: Sequence[str],
    expected_cache_key: str,
) -> list[float] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("cache_key") != expected_cache_key:
        return None
    if payload.get("passage_ids") != list(expected_passage_ids):
        return None
    scores = payload.get("scores")
    if not isinstance(scores, list) or len(scores) != len(expected_passage_ids):
        return None
    return [float(score) for score in scores]


def write_score_cache(
    path: Path,
    *,
    cache_key: str,
    passage_ids: Sequence[str],
    scores: Sequence[float],
    metadata: Mapping[str, object],
) -> None:
    if len(passage_ids) != len(scores):
        raise ValueError("cannot cache mismatched passage IDs and scores")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "metadata": dict(metadata),
                "passage_ids": list(passage_ids),
                "scores": [float(score) for score in scores],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def dense_hyde_scores(
    passages: Sequence[TextPassage],
    *,
    query: str,
    hypothetical_answer: str,
    model_name: str,
    model_revision: str | None,
    alpha: float,
    batch_size: int,
    device: str,
    max_seq_length: int,
    query_prompt_name: str | None = None,
    query_prefix: str = "",
) -> list[float]:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers and numpy are required; run `uv sync --group utokyo`"
        ) from exc

    model = SentenceTransformer(
        model_name,
        revision=model_revision,
        device=device,
        trust_remote_code=True,
    )
    model.max_seq_length = max_seq_length
    encode_query_kwargs: dict[str, object] = {
        "batch_size": 1,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    if query_prompt_name:
        encode_query_kwargs["prompt_name"] = query_prompt_name
    original = model.encode(query_prefix + query, **encode_query_kwargs)
    hypothetical = model.encode(query_prefix + hypothetical_answer, **encode_query_kwargs)
    mixed = np.asarray(
        hyde_vector_mix(original.tolist(), hypothetical.tolist(), alpha=alpha),
        dtype="float32",
    )
    document_embeddings = model.encode(
        [passage.text for passage in passages],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    scores = document_embeddings @ mixed
    result = [float(value) for value in scores]
    del document_embeddings, model
    gc.collect()
    return result


def splade_scores(
    passages: Sequence[TextPassage],
    *,
    query: str,
    model_name: str,
    model_revision: str | None,
    batch_size: int,
    device: str,
    max_length: int,
) -> list[float]:
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "torch and transformers are required; run `uv sync --group utokyo`"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    model = AutoModelForMaskedLM.from_pretrained(model_name, revision=model_revision)
    model.to(device)
    model.eval()

    def encode(texts: Sequence[str]):
        tokens = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        logits = model(**tokens).logits
        weighted = torch.log1p(torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1)
        return torch.max(weighted, dim=1).values

    with torch.inference_mode():
        query_vector = encode([query]).squeeze(0)
        scores: list[float] = []
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            vectors = encode([passage.text for passage in batch])
            scores.extend(float(value) for value in (vectors @ query_vector).detach().cpu())
    del model, tokenizer
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores
