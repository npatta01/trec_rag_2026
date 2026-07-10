"""Benchmark reranker runtime on cached or synthetic candidate documents."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker
from trec_rag.pipeline import pipeline_cache_dir
from trec_rag.pipeline_config import load_pipeline_config
from trec_rag.pipeline_models import RetrievedCandidate
from trec_rag.remote_config import RemotePyseriniConfig
from trec_rag.repo_env import load_repo_env
from trec_rag.rerank_score_cache import DEFAULT_INDEX_URL, _queries_by_topic, _topic_candidates, _topics


DEFAULT_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"


def _emit(label: str, **fields: Any) -> None:
    print(json.dumps({"label": label, **fields}, sort_keys=True), flush=True)


def _choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sync_if_needed(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize()


def _torch_info(device: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_imported": False}
    info: dict[str, Any] = {
        "torch_imported": True,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def _repeated_text(target_chars: int, *, marker: str) -> str:
    paragraph = (
        f"{marker} This document discusses the topic in detail, with names, dates, "
        "locations, causes, evidence, counter evidence, and repeated explanatory "
        "passages. It includes enough natural language to exercise tokenizer and "
        "attention runtime for reranking benchmarks. "
    )
    repeats = max(1, target_chars // len(paragraph) + 1)
    return (paragraph * repeats)[:target_chars]


def _synthetic_candidates(*, count: int, chars: int, topic_id: str, query: str) -> list[RetrievedCandidate]:
    return [
        RetrievedCandidate(
            topic_id=topic_id,
            variant_name="synthetic",
            retriever_name="synthetic",
            query_text=query,
            docid=f"synthetic-{topic_id}-{index:04d}",
            rank=index + 1,
            score=float(count - index),
            text=_repeated_text(chars, marker=f"doc {index}"),
        )
        for index in range(count)
    ]


def _cached_candidates(
    *,
    config_path: Path,
    topic_id: str,
    limit: int | None,
) -> list[RetrievedCandidate]:
    config = load_pipeline_config(config_path)
    load_repo_env(config.root_dir)
    os.environ.setdefault("INDEX_URL", DEFAULT_INDEX_URL)
    index_url = RemotePyseriniConfig.from_env().index_url
    cache_dir = pipeline_cache_dir(config.root_dir, config.run_id)
    topic = _topics(config, [topic_id])[0]
    queries = _queries_by_topic(config)
    return _topic_candidates(
        config=config,
        topic=topic,
        query=queries[topic.id],
        retriever=config.retrievers[0],
        cache_dir=cache_dir,
        index_url=index_url,
        limit=limit,
    )


def _load_candidates(args: argparse.Namespace, *, kind: str) -> list[RetrievedCandidate]:
    if args.source == "synthetic":
        if kind == "window":
            return _synthetic_candidates(
                count=args.limit,
                chars=args.synthetic_window_chars,
                topic_id="synthetic-window",
                query=args.synthetic_query,
            )
        return _synthetic_candidates(
            count=args.long_doc_count,
            chars=args.synthetic_long_chars,
            topic_id="synthetic-long",
            query=args.synthetic_query,
        )

    topic_id = args.topic if kind == "window" else args.long_topic
    limit = args.limit if kind == "window" else None
    candidates = _cached_candidates(config_path=args.config, topic_id=topic_id, limit=limit)
    if kind == "document":
        candidates = sorted(candidates, key=lambda candidate: len(candidate.text), reverse=True)[: args.long_doc_count]
    return candidates


def _load_cross_encoder(model_name: str, *, max_length: int, device: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=max_length, device=device)


def _predict(model: Any, pairs: list[tuple[str, str]], *, batch_size: int, device: str) -> int:
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    _sync_if_needed(device)
    return int(scores.shape[0] if hasattr(scores, "shape") else len(scores))


def _benchmark_windows(args: argparse.Namespace, *, device: str) -> None:
    candidates = _load_candidates(args, kind="window")
    char_lengths = [len(candidate.text) for candidate in candidates]
    chunker = SemanticTextChunker(
        ChunkingConfig(
            max_characters=args.chunk_max_characters,
            overlap_characters=args.chunk_overlap_characters,
        )
    )
    start = time.perf_counter()
    pending = [
        (candidate.query_text, chunk.text)
        for candidate in candidates
        for chunk in chunker.split_text(candidate.text, document_id=candidate.docid)
    ]
    chunk_seconds = time.perf_counter() - start
    _emit(
        "window_input",
        source=args.source,
        docs=len(candidates),
        mean_chars=round(statistics.mean(char_lengths), 2),
        max_chars=max(char_lengths),
        chunks=len(pending),
        chunk_seconds=round(chunk_seconds, 4),
    )

    model = _load_cross_encoder(args.model, max_length=args.window_max_length, device=device)
    dtype = str(next(model.model.parameters()).dtype)
    _predict(model, pending[: min(4, len(pending))], batch_size=min(4, args.window_batch_size), device=device)
    start = time.perf_counter()
    scored = _predict(model, pending, batch_size=args.window_batch_size, device=device)
    seconds = time.perf_counter() - start
    _emit(
        "window_predict",
        max_length=args.window_max_length,
        batch_size=args.window_batch_size,
        scores=scored,
        seconds=round(seconds, 4),
        scores_per_second=round(scored / seconds, 4),
        dtype=dtype,
    )


def _benchmark_documents(args: argparse.Namespace, *, device: str) -> None:
    candidates = _load_candidates(args, kind="document")
    pairs = [(candidate.query_text, candidate.text) for candidate in candidates]
    char_lengths = [len(candidate.text) for candidate in candidates]
    _emit(
        "document_input",
        source=args.source,
        docs=len(candidates),
        mean_chars=round(statistics.mean(char_lengths), 2),
        max_chars=max(char_lengths),
        docids=[candidate.docid for candidate in candidates],
    )

    for max_length in args.document_max_lengths:
        model = _load_cross_encoder(args.model, max_length=max_length, device=device)
        dtype = str(next(model.model.parameters()).dtype)
        start = time.perf_counter()
        tokenized = model.tokenizer(
            [query for query, _ in pairs],
            [text for _, text in pairs],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        token_seconds = time.perf_counter() - start
        token_lengths = [len(row) for row in tokenized["input_ids"]]
        _predict(model, pairs[:1], batch_size=1, device=device)
        start = time.perf_counter()
        scored = _predict(model, pairs, batch_size=args.document_batch_size, device=device)
        seconds = time.perf_counter() - start
        _emit(
            "document_predict",
            max_length=max_length,
            batch_size=args.document_batch_size,
            scores=scored,
            seconds=round(seconds, 4),
            scores_per_second=round(scored / seconds, 4),
            token_seconds=round(token_seconds, 4),
            token_lengths=token_lengths,
            dtype=dtype,
        )
        del model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["cache", "synthetic"], default="cache")
    parser.add_argument("--config", type=Path, default=Path("configs/rag25_bm25_mixedbread_rerank_v1.yaml"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topic", default="14")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--long-topic", default="273")
    parser.add_argument("--long-doc-count", type=int, default=4)
    parser.add_argument("--window-max-length", type=int, default=1024)
    parser.add_argument("--document-max-lengths", type=int, nargs="+", default=[4096, 8192, 32768])
    parser.add_argument("--window-batch-size", type=int, default=8)
    parser.add_argument("--document-batch-size", type=int, default=1)
    parser.add_argument("--chunk-max-characters", type=int, default=3500)
    parser.add_argument("--chunk-overlap-characters", type=int, default=350)
    parser.add_argument("--synthetic-query", default="What evidence answers this benchmark topic?")
    parser.add_argument("--synthetic-window-chars", type=int, default=25_000)
    parser.add_argument("--synthetic-long-chars", type=int, default=800_000)
    parser.add_argument("--skip-window", action="store_true")
    parser.add_argument("--skip-document", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    device = _choose_device(args.device)
    _emit("environment", **_torch_info(device))
    if not args.skip_window:
        _benchmark_windows(args, device=device)
    if not args.skip_document:
        _benchmark_documents(args, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
