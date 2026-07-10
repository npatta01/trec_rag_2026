#!/usr/bin/env python3
"""Standalone Mixedbread reranker benchmark for CUDA/ROCm/vLLM machines.

Install for local model timing:

    python3 -m pip install -U sentence-transformers

Run local sentence-transformers timing:

    python3 standalone_reranker_bench.py --backend sentence-transformers --device cuda

Run against a vLLM OpenAI-compatible server:

    python3 standalone_reranker_bench.py --backend vllm --vllm-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"


def emit(label: str, **fields: Any) -> None:
    print(json.dumps({"label": label, **fields}, sort_keys=True), flush=True)


def repeated_text(target_chars: int, *, marker: str) -> str:
    paragraph = (
        f"{marker}. This benchmark document discusses the topic in detail, with names, "
        "dates, locations, causes, evidence, counter evidence, and repeated explanatory "
        "passages. It includes enough natural language to exercise tokenizer and long "
        "attention runtime for reranking benchmarks. "
    )
    repeats = max(1, target_chars // len(paragraph) + 1)
    return (paragraph * repeats)[:target_chars]


def synthetic_pairs(*, count: int, chars: int, query: str) -> list[tuple[str, str]]:
    return [(query, repeated_text(chars, marker=f"doc {index}")) for index in range(count)]


def chunk_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")
    chunks: list[str] = []
    step = max_chars - overlap_chars
    start = 0
    while start < len(text):
        chunk = text[start : start + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        if start + max_chars >= len(text):
            break
        start += step
    return chunks


def build_window_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    docs = synthetic_pairs(count=args.window_docs, chars=args.window_chars, query=args.query)
    start = time.perf_counter()
    pairs = [
        (query, chunk)
        for query, text in docs
        for chunk in chunk_text(
            text,
            max_chars=args.chunk_max_characters,
            overlap_chars=args.chunk_overlap_characters,
        )
    ]
    emit(
        "window_input",
        docs=len(docs),
        chunks=len(pairs),
        doc_chars=args.window_chars,
        chunk_seconds=round(time.perf_counter() - start, 4),
    )
    return pairs


def torch_info(device: str) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"torch_imported": False, "torch_error": repr(exc), "device": device}
    info: dict[str, Any] = {
        "torch_imported": True,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        try:
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
        except Exception:
            pass
    return info


def sync_if_needed(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize()


def score_sentence_transformers(
    *,
    pairs: list[tuple[str, str]],
    model_name: str,
    max_length: int,
    batch_size: int,
    device: str,
) -> tuple[int, str]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, max_length=max_length, device=device)
    dtype = str(next(model.model.parameters()).dtype)
    warmup_pairs = pairs[: min(2, len(pairs))]
    if warmup_pairs:
        model.predict(
            warmup_pairs,
            batch_size=min(batch_size, len(warmup_pairs)),
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        sync_if_needed(device)
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    sync_if_needed(device)
    return int(scores.shape[0] if hasattr(scores, "shape") else len(scores)), dtype


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} failed with HTTP {exc.code}: {body[:1000]}") from exc


def score_vllm(
    *,
    pairs: list[tuple[str, str]],
    model_name: str,
    base_url: str,
    batch_size: int,
    max_length: int,
    cache_salt_prefix: str,
) -> tuple[int, str]:
    endpoint = base_url.rstrip("/") + "/score"
    scored = 0
    max_query_tokens = min(128, max(1, max_length // 4))
    max_doc_tokens = max(1, max_length - max_query_tokens - 16)
    for offset in range(0, len(pairs), batch_size):
        batch = pairs[offset : offset + batch_size]
        payload = {
            "model": model_name,
            "queries": [query for query, _ in batch],
            "documents": [text for _, text in batch],
            "truncate_prompt_tokens": max_length,
            "truncation_side": "right",
            "max_tokens_per_query": max_query_tokens,
            "max_tokens_per_doc": max_doc_tokens,
        }
        if cache_salt_prefix:
            payload["cache_salt"] = f"{cache_salt_prefix}-{time.time_ns()}-{offset}"
        response = post_json(endpoint, payload)
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected vLLM /score response: {response}")
        scored += len(data)
    return scored, "vllm-http"


def benchmark_pairs(
    *,
    label: str,
    pairs: list[tuple[str, str]],
    args: argparse.Namespace,
    max_length: int,
    batch_size: int,
) -> None:
    char_lengths = [len(text) for _, text in pairs]
    emit(
        label + "_batch",
        pairs=len(pairs),
        mean_chars=round(statistics.mean(char_lengths), 2),
        max_chars=max(char_lengths),
        max_length=max_length,
        batch_size=batch_size,
        backend=args.backend,
    )
    start = time.perf_counter()
    if args.backend == "sentence-transformers":
        scored, dtype = score_sentence_transformers(
            pairs=pairs,
            model_name=args.model,
            max_length=max_length,
            batch_size=batch_size,
            device=args.device,
        )
    else:
        scored, dtype = score_vllm(
            pairs=pairs,
            model_name=args.model,
            base_url=args.vllm_url,
            batch_size=batch_size,
            max_length=max_length,
            cache_salt_prefix=args.vllm_cache_salt_prefix,
        )
    seconds = time.perf_counter() - start
    emit(
        label + "_predict",
        scores=scored,
        seconds=round(seconds, 4),
        scores_per_second=round(scored / seconds, 4),
        dtype=dtype,
        max_length=max_length,
        batch_size=batch_size,
        backend=args.backend,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone synthetic reranker benchmark.")
    parser.add_argument("--backend", choices=["sentence-transformers", "vllm"], default="sentence-transformers")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument(
        "--vllm-cache-salt-prefix",
        default="",
        help="Set to a non-empty value to avoid reusing vLLM prefix cache across benchmark requests.",
    )
    parser.add_argument("--query", default="What evidence answers this benchmark topic?")
    parser.add_argument("--window-docs", type=int, default=20)
    parser.add_argument("--window-chars", type=int, default=25_000)
    parser.add_argument("--long-docs", type=int, default=4)
    parser.add_argument("--long-chars", type=int, default=800_000)
    parser.add_argument("--window-max-length", type=int, default=1024)
    parser.add_argument("--document-max-lengths", type=int, nargs="+", default=[4096, 8192, 32768])
    parser.add_argument("--window-batch-size", type=int, default=8)
    parser.add_argument("--document-batch-size", type=int, default=1)
    parser.add_argument("--chunk-max-characters", type=int, default=3500)
    parser.add_argument("--chunk-overlap-characters", type=int, default=350)
    parser.add_argument("--skip-window", action="store_true")
    parser.add_argument("--skip-document", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.backend == "sentence-transformers":
        emit("environment", **torch_info(args.device))
    else:
        emit("environment", backend="vllm", vllm_url=args.vllm_url, model=args.model)

    if not args.skip_window:
        benchmark_pairs(
            label="window",
            pairs=build_window_pairs(args),
            args=args,
            max_length=args.window_max_length,
            batch_size=args.window_batch_size,
        )

    if not args.skip_document:
        doc_pairs = synthetic_pairs(count=args.long_docs, chars=args.long_chars, query=args.query)
        emit("document_input", docs=len(doc_pairs), doc_chars=args.long_chars)
        for max_length in args.document_max_lengths:
            benchmark_pairs(
                label="document",
                pairs=doc_pairs,
                args=args,
                max_length=max_length,
                batch_size=args.document_batch_size,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
