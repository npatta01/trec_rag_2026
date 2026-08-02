"""Build the exact Mixedbread top-100 seed consumed by FIRST on Modal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trec_rag.organizer_reranking import (
    LISTWISE_SEED_SCHEMA,
    load_bm25_candidates,
    sha256_canonical_text,
    sha256_text,
)
from trec_rag.retrieval_ranking_benchmark import (
    build_current_pointwise_ranking,
    load_benchmark_config,
)


DEFAULT_CONFIG = Path("configs/rag25_pointwise_listwise_ndcg_v1.yaml")


def build_seed(config_path: Path) -> Path:
    config = load_benchmark_config(config_path)
    depth = int(config.raw["organizer_listwise"]["candidate_depth"])
    candidates = load_bm25_candidates(
        config.path("bm25_candidates"),
        expected_depth=int(config.raw["organizer_pointwise"]["candidate_depth"]),
    )
    ranking = build_current_pointwise_ranking(config, candidates)
    source = {
        (candidate.topic_id, candidate.docid): candidate
        for rows in candidates.values()
        for candidate in rows
    }
    by_topic: dict[str, list[object]] = {}
    for row in ranking:
        by_topic.setdefault(row.topic_id, []).append(row)

    output_path = config.path("current_listwise_seed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    with partial_path.open("w", encoding="utf-8", newline="\n") as sink:
        for topic_id in sorted(by_topic, key=int):
            rows = sorted(by_topic[topic_id], key=lambda row: row.rank)[:depth]
            if len(rows) != depth or [row.rank for row in rows] != list(range(1, depth + 1)):
                raise ValueError(f"topic {topic_id}: current ranking does not contain top {depth}")
            for row in rows:
                candidate = source[(topic_id, row.docid)]
                sink.write(
                    json.dumps(
                        {
                            "schema_version": LISTWISE_SEED_SCHEMA,
                            "topic_id": topic_id,
                            "docid": row.docid,
                            "rank": row.rank,
                            "score": row.score,
                            "query_sha256": sha256_text(candidate.query_text),
                            "text_sha256": sha256_canonical_text(candidate.text),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    partial_path.replace(output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    output_path = build_seed(args.config)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
