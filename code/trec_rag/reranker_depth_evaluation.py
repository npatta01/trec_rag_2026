"""Evaluate max-passage document rankings from cached window-score artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from trec_rag.evaluation import evaluate_ranked, parse_qrels
from trec_rag.pipeline_models import RankedCandidate


IDENTITY_FIELDS = (
    "artifact_schema_version",
    "backend",
    "backend_version",
    "model",
    "model_revision",
    "score_representation",
    "inference_dtype",
    "input_policy",
    "max_length",
    "score_kind",
    "chunk_max_characters",
    "chunk_overlap_characters",
)
METRICS = (
    "ndcg@10",
    "ndcg@20",
    "judged_count@10",
    "judged_rate@10",
    "judged_count@20",
    "judged_rate@20",
)


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    return (0, int(topic_id)) if topic_id.isdigit() else (1, topic_id)


@dataclass(frozen=True)
class WindowDocument:
    topic_id: str
    docid: str
    bm25_rank: int
    max_score: float
    chunk_count: int
    query_sha256: str
    document_text_sha256: str


@dataclass(frozen=True)
class WindowArtifact:
    path: Path
    identity: dict[str, object]
    documents: dict[str, dict[str, WindowDocument]]
    total_chunks: int

    @property
    def model(self) -> str:
        return str(self.identity["model"])

    @property
    def topic_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.documents, key=_topic_sort_key))

    @property
    def total_documents(self) -> int:
        return sum(len(rows) for rows in self.documents.values())


def _required(row: Mapping[str, Any], field: str, path: Path, line_number: int) -> Any:
    if field not in row:
        raise ValueError(f"{path}:{line_number}: missing {field}")
    return row[field]


def load_window_artifact(
    path: Path,
    *,
    expected_depth: int,
    expected_topics: Sequence[str] | None = None,
) -> WindowArtifact:
    """Strictly load one complete window artifact and aggregate document maxima."""

    path = Path(path)
    if expected_depth <= 0:
        raise ValueError("expected_depth must be positive")
    identities: dict[str, dict[str, object]] = {}
    chunks: dict[tuple[str, str], dict[int, tuple[float, int, int, str, str]]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("row is not an object")
                identity = {
                    field: _required(row, field, path, line_number)
                    for field in IDENTITY_FIELDS
                }
                identity_key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
                identities[identity_key] = identity
                topic_id = str(_required(row, "topic_id", path, line_number))
                docid = str(_required(row, "docid", path, line_number))
                rank = int(_required(row, "rank", path, line_number))
                chunk_index = int(_required(row, "chunk_index", path, line_number))
                chunk_count = int(_required(row, "chunk_count", path, line_number))
                score = float(_required(row, "score", path, line_number))
                query_hash = str(_required(row, "query_sha256", path, line_number))
                document_hash = str(
                    _required(row, "document_text_sha256", path, line_number)
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid window row") from exc
            if not topic_id or not docid or not query_hash or not document_hash:
                raise ValueError(f"{path}:{line_number}: empty window identity field")
            if rank < 1 or chunk_count < 1 or not 0 <= chunk_index < chunk_count:
                raise ValueError(f"{path}:{line_number}: invalid rank or chunk bounds")
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line_number}: non-finite score")
            key = (topic_id, docid)
            per_doc = chunks.setdefault(key, {})
            if chunk_index in per_doc:
                raise ValueError(f"{path}:{line_number}: duplicate chunk for {key}")
            per_doc[chunk_index] = (score, rank, chunk_count, query_hash, document_hash)

    if not chunks:
        raise ValueError(f"{path}: no window rows")
    if len(identities) != 1:
        raise ValueError(f"{path}: multiple artifact identities")

    documents: dict[str, dict[str, WindowDocument]] = {}
    total_chunks = 0
    for (topic_id, docid), per_doc in chunks.items():
        values = list(per_doc.values())
        ranks = {value[1] for value in values}
        counts = {value[2] for value in values}
        query_hashes = {value[3] for value in values}
        document_hashes = {value[4] for value in values}
        if len(ranks) != 1 or len(counts) != 1 or len(query_hashes) != 1 or len(document_hashes) != 1:
            raise ValueError(f"{path}: conflicting chunk metadata for {(topic_id, docid)}")
        chunk_count = counts.pop()
        if set(per_doc) != set(range(chunk_count)):
            raise ValueError(f"{path}: incomplete chunks for {(topic_id, docid)}")
        documents.setdefault(topic_id, {})[docid] = WindowDocument(
            topic_id=topic_id,
            docid=docid,
            bm25_rank=ranks.pop(),
            max_score=max(value[0] for value in values),
            chunk_count=chunk_count,
            query_sha256=query_hashes.pop(),
            document_text_sha256=document_hashes.pop(),
        )
        total_chunks += chunk_count

    for topic_id, topic_docs in documents.items():
        ranks = sorted(row.bm25_rank for row in topic_docs.values())
        if ranks != list(range(1, expected_depth + 1)):
            raise ValueError(
                f"{path}: topic {topic_id} BM25 ranks are not contiguous depth {expected_depth}"
            )
        if len({row.query_sha256 for row in topic_docs.values()}) != 1:
            raise ValueError(f"{path}: topic {topic_id} has multiple query identities")

    actual_topics = tuple(sorted(documents, key=_topic_sort_key))
    if expected_topics is not None:
        wanted = tuple(sorted((str(topic) for topic in expected_topics), key=_topic_sort_key))
        if actual_topics != wanted:
            raise ValueError(f"{path}: topic population mismatch")
    identity = next(iter(identities.values()))
    return WindowArtifact(path, identity, documents, total_chunks)


def _candidate_identity(artifact: WindowArtifact) -> dict[tuple[str, str], tuple[object, ...]]:
    return {
        (topic_id, docid): (
            row.bm25_rank,
            row.query_sha256,
            row.document_text_sha256,
        )
        for topic_id, topic_docs in artifact.documents.items()
        for docid, row in topic_docs.items()
    }


def _ranked(
    artifact: WindowArtifact,
    *,
    depth: int,
    use_bm25: bool,
) -> list[RankedCandidate]:
    ranked: list[RankedCandidate] = []
    for topic_id in artifact.topic_ids:
        rows = [
            row
            for row in artifact.documents[topic_id].values()
            if row.bm25_rank <= depth
        ]
        if use_bm25:
            rows.sort(key=lambda row: (row.bm25_rank, row.docid))
        else:
            rows.sort(key=lambda row: (-row.max_score, row.bm25_rank, row.docid))
        ranked.extend(
            RankedCandidate(topic_id, row.docid, rank, row.max_score, "", [])
            for rank, row in enumerate(rows, start=1)
        )
    return ranked


def paired_bootstrap_interval(
    deltas: Sequence[float], *, samples: int = 10_000, seed: int = 20260806
) -> dict[str, float | int]:
    if not deltas:
        raise ValueError("paired bootstrap requires topic deltas")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    values = [float(value) for value in deltas]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("paired bootstrap deltas must be finite")
    generator = random.Random(seed)
    means = sorted(
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    lower_index = max(0, math.floor(0.025 * (samples - 1)))
    upper_index = min(samples - 1, math.ceil(0.975 * (samples - 1)))
    return {
        "mean": sum(values) / len(values),
        "lower_95": means[lower_index],
        "upper_95": means[upper_index],
        "samples": samples,
        "seed": seed,
    }


def evaluate_max_passage_systems(
    systems: Mapping[str, WindowArtifact],
    *,
    qrels_path: Path,
    candidate_depths: Sequence[int],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260806,
) -> dict[str, object]:
    if len(systems) != 2:
        raise ValueError("exactly two reranker systems are required")
    labels = tuple(systems)
    left_label, right_label = labels
    left, right = systems[left_label], systems[right_label]
    if left.topic_ids != right.topic_ids or _candidate_identity(left) != _candidate_identity(right):
        raise ValueError("cross-model candidate identity mismatch")
    if left.identity == right.identity:
        raise ValueError("reranker systems must have distinct identities")
    qrels = parse_qrels(Path(qrels_path))
    if set(qrels) != set(left.topic_ids):
        raise ValueError("qrels topic population differs from artifacts")
    max_depth = min(len(rows) for rows in left.documents.values())
    depths = sorted(set(int(depth) for depth in candidate_depths))
    if not depths or depths[0] <= 0 or depths[-1] > max_depth:
        raise ValueError("candidate depths must be positive and available in the artifacts")

    report_depths: dict[str, object] = {}
    for depth in depths:
        baseline = evaluate_ranked(
            _ranked(left, depth=depth, use_bm25=True),
            qrels,
            metric_names=METRICS,
            relevance_threshold=2,
            topic_ids=left.topic_ids,
        )
        evaluated = {
            label: evaluate_ranked(
                _ranked(artifact, depth=depth, use_bm25=False),
                qrels,
                metric_names=METRICS,
                relevance_threshold=2,
                topic_ids=left.topic_ids,
            )
            for label, artifact in systems.items()
        }
        per_topic_delta: dict[str, dict[str, float]] = {}
        right_minus_left: list[float] = []
        wins = {left_label: 0, right_label: 0, "ties": 0}
        for topic_id in left.topic_ids:
            left_score = float(evaluated[left_label]["per_topic"][topic_id]["ndcg@10"])
            right_score = float(evaluated[right_label]["per_topic"][topic_id]["ndcg@10"])
            delta = right_score - left_score
            per_topic_delta[topic_id] = {
                left_label: left_score,
                right_label: right_score,
                f"{right_label}_minus_{left_label}": delta,
            }
            right_minus_left.append(delta)
            if delta > 1e-12:
                wins[right_label] += 1
            elif delta < -1e-12:
                wins[left_label] += 1
            else:
                wins["ties"] += 1

        ranked_orders = {
            label: {
                topic_id: [
                    row.docid
                    for row in sorted(
                        (item for item in _ranked(artifact, depth=depth, use_bm25=False) if item.topic_id == topic_id),
                        key=lambda item: item.rank,
                    )
                ]
                for topic_id in left.topic_ids
            }
            for label, artifact in systems.items()
        }
        overlaps: dict[str, object] = {}
        for cutoff in (1, 10, 20):
            effective = min(cutoff, depth)
            values = [
                len(
                    set(ranked_orders[left_label][topic_id][:effective])
                    & set(ranked_orders[right_label][topic_id][:effective])
                )
                / effective
                for topic_id in left.topic_ids
            ]
            overlaps[str(cutoff)] = {
                "mean_overlap_fraction": sum(values) / len(values),
                "per_topic": dict(zip(left.topic_ids, values, strict=True)),
            }
        report_depths[str(depth)] = {
            "baseline": baseline,
            "systems": evaluated,
            "comparison": {
                "labels": list(labels),
                "per_topic_delta": per_topic_delta,
                "topic_wins": wins,
                "paired_bootstrap_ndcg_at_10": paired_bootstrap_interval(
                    right_minus_left,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                ),
                "top_k_overlap": overlaps,
            },
        }

    return {
        "schema_version": "reranker-depth-evaluation-v1",
        "qrels_path": str(qrels_path),
        "topic_count": len(left.topic_ids),
        "topic_ids": list(left.topic_ids),
        "systems": {
            label: {
                "source_artifact": str(artifact.path),
                "identity": artifact.identity,
                "document_count": artifact.total_documents,
                "chunk_count": artifact.total_chunks,
            }
            for label, artifact in systems.items()
        },
        "depths": report_depths,
    }


def write_evaluation(report: Mapping[str, object], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_tmp = output_dir / ".metrics.json.tmp"
    metrics_path = output_dir / "metrics.json"
    metrics_tmp.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_tmp.replace(metrics_path)

    topic_rows: list[dict[str, object]] = []
    fieldnames = ["depth", "topic_id"]
    depths = report.get("depths", {})
    if not isinstance(depths, Mapping):
        raise ValueError("evaluation report depths must be a mapping")
    for depth, depth_payload in depths.items():
        comparison = depth_payload["comparison"]
        per_topic = comparison["per_topic_delta"]
        for topic_id, values in per_topic.items():
            row = {"depth": depth, "topic_id": topic_id, **values}
            topic_rows.append(row)
            for key in values:
                if key not in fieldnames:
                    fieldnames.append(key)
    csv_tmp = output_dir / ".topic_metrics.csv.tmp"
    with csv_tmp.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topic_rows)
    csv_tmp.replace(output_dir / "topic_metrics.csv")


def _system_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("system must be LABEL=PATH")
    return label, Path(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", type=_system_argument, action="append", required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--expected-depth", type=int, required=True)
    parser.add_argument("--depth", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    systems = dict(args.system)
    if len(systems) != len(args.system):
        raise ValueError("system labels must be unique")
    qrels = parse_qrels(args.qrels)
    artifacts = {
        label: load_window_artifact(
            path,
            expected_depth=args.expected_depth,
            expected_topics=tuple(qrels),
        )
        for label, path in systems.items()
    }
    report = evaluate_max_passage_systems(
        artifacts,
        qrels_path=args.qrels,
        candidate_depths=args.depth,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_evaluation(report, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "topic_count": report["topic_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
