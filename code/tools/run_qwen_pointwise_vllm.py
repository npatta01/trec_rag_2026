"""Score the controlled BM25 pool through a running local vLLM server.

Only the config path and the operation are command-line inputs.  Model,
precision, prompt, candidate depth, endpoint, batching, and smoke sample are
all pinned in the benchmark configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from trec_rag.organizer_reranking import (  # noqa: E402
    POINTWISE_SCHEMA,
    RerankCandidate,
    iter_batches,
    load_bm25_candidate_sample,
    load_bm25_candidates,
    pointwise_score_row,
    qwen_label_token_ids,
    qwen_pointwise_batch_token_ids,
)
from trec_rag.retrieval_ranking_benchmark import load_benchmark_config  # noqa: E402


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        raise ValueError(f"incomplete pointwise score artifact: {path}")
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != POINTWISE_SCHEMA:
                raise ValueError(f"{path}:{line_number}: score schema mismatch")
            key = (str(row["topic_id"]), str(row["docid"]))
            if key in keys:
                raise ValueError(f"{path}:{line_number}: duplicate score")
            keys.add(key)
    return keys


def _select_smoke(
    candidate_path: Path,
    settings: dict[str, object],
) -> list[RerankCandidate]:
    local = settings["local_vllm"]
    topic_id = str(local["smoke_topic_id"])
    ranks = [int(rank) for rank in local["smoke_ranks"]]
    return load_bm25_candidate_sample(
        candidate_path,
        topic_id=topic_id,
        ranks=ranks,
        expected_depth=int(settings["candidate_depth"]),
    )


def _score(
    *,
    rows: list[RerankCandidate],
    output_path: Path,
    settings: dict[str, object],
    overwrite: bool,
) -> dict[str, object]:
    model = str(settings["model"])
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    revision = str(getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or "unresolved")
    yes_id, no_id = qwen_label_token_ids(tokenizer)
    local = settings["local_vllm"]
    endpoint = str(local["endpoint"])
    request_timeout = float(local["request_timeout_seconds"])
    batch_size = int(settings["batch_size"])
    max_length = int(settings["max_length"])
    instruction = str(settings["instruction"])
    if overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete = _existing_keys(output_path)
    pending = [row for row in rows if (row.topic_id, row.docid) not in complete]

    started = time.perf_counter()
    prompt_tokens = 0
    request_count = 0
    pending_by_query: dict[tuple[str, str], list[RerankCandidate]] = {}
    for row in pending:
        pending_by_query.setdefault((row.topic_id, row.query_text), []).append(row)
    with requests.Session() as session, output_path.open("a", encoding="utf-8", newline="\n") as sink:
        batches = (
            batch
            for topic_rows in pending_by_query.values()
            for batch in iter_batches(topic_rows, batch_size)
        )
        for batch in batches:
            token_ids = qwen_pointwise_batch_token_ids(
                tokenizer,
                query=batch[0].query_text,
                documents=[row.text for row in batch],
                max_length=max_length,
                instruction=instruction,
            )
            if len({row.query_text for row in batch}) != 1:
                raise ValueError("pointwise request batch crossed topic queries")
            response = session.post(
                endpoint,
                json={
                    "model": model,
                    "query": [],
                    "items": token_ids,
                    "label_token_ids": [yes_id, no_id],
                    "apply_softmax": True,
                    "add_special_tokens": False,
                },
                timeout=request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            scores = payload.get("data")
            if not isinstance(scores, list) or len(scores) != len(batch):
                raise ValueError(f"invalid vLLM generative-scoring response: {payload}")
            for index, (candidate, item, input_ids) in enumerate(
                zip(batch, scores, token_ids, strict=True)
            ):
                if int(item["index"]) != index:
                    raise ValueError("vLLM response indices are not contiguous")
                record = pointwise_score_row(
                    candidate,
                    score=float(item["score"]),
                    model=model,
                    model_revision=revision,
                    dtype=str(settings["dtype"]),
                    max_length=max_length,
                    prompt_tokens=len(input_ids),
                    backend="vllm-generative-scoring-local",
                )
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            prompt_tokens += int(payload.get("usage", {}).get("prompt_tokens", sum(map(len, token_ids))))
            request_count += 1

    elapsed = time.perf_counter() - started
    total_rows = len(rows)
    scored_rows = len(pending)
    average_tokens = prompt_tokens / scored_rows if scored_rows else 0.0
    tokens_per_second = prompt_tokens / elapsed if elapsed else 0.0
    projected_seconds = (
        average_tokens * total_rows / tokens_per_second if tokens_per_second else 0.0
    )
    return {
        "schema_version": "local-pointwise-runtime-v1",
        "model": model,
        "dtype": settings["dtype"],
        "backend": "vllm-generative-scoring-local",
        "rows_requested": total_rows,
        "rows_scored_this_invocation": scored_rows,
        "rows_preexisting": len(complete),
        "request_count": request_count,
        "prompt_tokens": prompt_tokens,
        "average_prompt_tokens": average_tokens,
        "elapsed_seconds": elapsed,
        "prompt_tokens_per_second": tokens_per_second,
        "projected_seconds_for_requested_rows": projected_seconds,
        "output_path": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("smoke", "full"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    settings = dict(config.raw["organizer_pointwise"])
    if args.operation == "smoke":
        rows = _select_smoke(config.path("bm25_candidates"), settings)
        score_path = config.path("organizer_pointwise_scores").with_name(
            "qwen3_reranker_8b_bf16_smoke_scores.jsonl"
        )
        receipt_path = config.path("output_dir") / "local_pointwise_smoke.json"
        overwrite = True
    else:
        candidates = load_bm25_candidates(
            config.path("bm25_candidates"),
            expected_depth=int(settings["candidate_depth"]),
        )
        rows = [row for topic_rows in candidates.values() for row in topic_rows]
        score_path = config.path("organizer_pointwise_scores")
        receipt_path = config.path("output_dir") / "local_pointwise_runtime.json"
        overwrite = False
    receipt = _score(
        rows=rows,
        output_path=score_path,
        settings=settings,
        overwrite=overwrite,
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
