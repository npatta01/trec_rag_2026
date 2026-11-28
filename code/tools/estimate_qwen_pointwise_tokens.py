"""Measure the exact Qwen pointwise prompt-token population from config."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from transformers import AutoTokenizer


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from trec_rag.organizer_reranking import (  # noqa: E402
    load_bm25_candidates,
    qwen_pointwise_token_ids,
)
from trec_rag.retrieval_ranking_benchmark import load_benchmark_config  # noqa: E402


def _percentile(sorted_values: list[int], probability: float) -> int:
    return sorted_values[min(len(sorted_values) - 1, math.floor(probability * len(sorted_values)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    settings = config.raw["organizer_pointwise"]
    tokenizer = AutoTokenizer.from_pretrained(str(settings["model"]), trust_remote_code=False)
    candidates = load_bm25_candidates(
        config.path("bm25_candidates"),
        expected_depth=int(settings["candidate_depth"]),
    )
    lengths = sorted(
        len(
            qwen_pointwise_token_ids(
                tokenizer,
                query=row.query_text,
                document=row.text,
                max_length=int(settings["max_length"]),
                instruction=str(settings["instruction"]),
            )
        )
        for rows in candidates.values()
        for row in rows
    )
    payload = {
        "schema_version": "qwen-pointwise-token-estimate-v1",
        "model": settings["model"],
        "max_length": int(settings["max_length"]),
        "rows": len(lengths),
        "total_prompt_tokens": sum(lengths),
        "average_prompt_tokens": sum(lengths) / len(lengths),
        "median_prompt_tokens": _percentile(lengths, 0.5),
        "p90_prompt_tokens": _percentile(lengths, 0.9),
        "p95_prompt_tokens": _percentile(lengths, 0.95),
        "maximum_prompt_tokens": lengths[-1],
        "maximum_length_rows": sum(length == int(settings["max_length"]) for length in lengths),
    }
    output = config.path("output_dir") / "pointwise_token_estimate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
