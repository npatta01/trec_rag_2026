"""Evaluate organizer-style retrieval runs against projected development qrels."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Collection, Sequence
from hashlib import sha256
from pathlib import Path

from trec_rag.evaluation import evaluate_ranked
from trec_rag.pipeline_models import RankedCandidate


DEFAULT_METRIC_NAMES = (
    "ndcg@10",
    "judged_count@10",
    "judged_rate@10",
    "precision@10",
    "recall@10",
    "hit_rate@10",
    "relevant_count@10",
    "graded_recall@10",
    "judged_count@50",
    "judged_rate@50",
    "precision@50",
    "recall@50",
    "hit_rate@50",
    "relevant_count@50",
    "graded_recall@50",
    "ideal_dcg_coverage@50",
    "judged_count@100",
    "judged_rate@100",
    "recall@100",
)

PINNED_ASSESSOR_VARIANT = "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1"
PINNED_QRELS_SHA256 = "42bf933ae06eb22213312b22e3f2bc39f3dcc2d54e87ebcd8125e9528ddfcc37"
PINNED_QRELS_REPO_PATH = (
    "trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/"
    "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
)
PINNED_QRELS_TOPIC_IDS = (
    "14",
    "31",
    "37",
    "58",
    "72",
    "84",
    "144",
    "161",
    "200",
    "213",
    "219",
    "224",
    "225",
    "233",
    "273",
    "300",
    "407",
    "477",
    "499",
    "515",
    "707",
    "897",
)
SUPPORTED_METRIC_NAMES = frozenset(
    {
        "graded_recall",
        "hit_rate",
        "ideal_dcg_coverage",
        "judged_count",
        "judged_rate",
        "ndcg",
        "precision",
        "recall",
        "relevant_count",
    }
)


def _validated_topic_ids(topic_ids: Collection[str]) -> list[str]:
    normalized = [str(topic_id) for topic_id in topic_ids]
    if not normalized:
        raise ValueError("expected topic population must not be empty")
    if any(not topic_id or topic_id != topic_id.strip() for topic_id in normalized):
        raise ValueError("expected topic IDs must be non-empty and contain no surrounding whitespace")
    if len(normalized) != len(set(normalized)):
        raise ValueError("expected topic IDs must not contain duplicates")
    return sorted(normalized)


def parse_trec_retrieval_run(
    path: Path,
    *,
    expected_topic_ids: Collection[str],
) -> list[RankedCandidate]:
    return _parse_trec_retrieval_run_snapshot(
        path.read_bytes(),
        expected_topic_ids=expected_topic_ids,
    )


def _parse_trec_retrieval_run_snapshot(
    run_bytes: bytes,
    *,
    expected_topic_ids: Collection[str],
) -> list[RankedCandidate]:
    expected = _validated_topic_ids(expected_topic_ids)
    expected_set = set(expected)
    ranked: list[RankedCandidate] = []
    seen_ranks: dict[str, set[int]] = {}
    seen_docids: dict[str, set[str]] = {}
    previous_scores: dict[str, float] = {}
    run_tag: str | None = None

    for line_number, raw_line in enumerate(run_bytes.decode("utf-8").splitlines(), start=1):
        parts = raw_line.split()
        if len(parts) != 6:
            raise ValueError(f"line {line_number}: expected six TREC run columns")
        topic_id, q0, docid, rank_text, score_text, row_run_tag = parts
        if not topic_id or not docid or not row_run_tag:
            raise ValueError(f"line {line_number}: topic ID, document ID, and run tag must be non-empty")
        if q0 != "Q0":
            raise ValueError(f"line {line_number}: column 2 must be Q0")
        try:
            rank = int(rank_text)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: rank must be an integer") from exc
        if rank <= 0:
            raise ValueError(f"line {line_number}: rank must be positive")
        try:
            score = float(score_text)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: score must be a number") from exc
        if not math.isfinite(score):
            raise ValueError(f"line {line_number}: score must be finite")

        topic_ranks = seen_ranks.setdefault(topic_id, set())
        expected_rank = len(topic_ranks) + 1
        if rank != expected_rank:
            if rank in topic_ranks:
                raise ValueError(
                    f"line {line_number}: duplicate rank {rank} for topic {topic_id}; "
                    f"ranks must be dense in file order (expected {expected_rank}, got {rank})"
                )
            raise ValueError(
                f"line {line_number}: topic {topic_id} ranks must start at 1 and be dense "
                f"in file order (expected {expected_rank}, got {rank})"
            )
        topic_docids = seen_docids.setdefault(topic_id, set())
        if docid in topic_docids:
            raise ValueError(
                f"line {line_number}: duplicate document ID {docid} for topic {topic_id}"
            )
        if run_tag is None:
            run_tag = row_run_tag
        elif row_run_tag != run_tag:
            raise ValueError(
                f"line {line_number}: conflicting run tags {run_tag!r} and {row_run_tag!r}"
            )
        previous_score = previous_scores.get(topic_id)
        if previous_score is not None and score > previous_score:
            raise ValueError(
                f"line {line_number}: scores must be non-increasing for topic {topic_id}"
            )

        topic_ranks.add(rank)
        topic_docids.add(docid)
        previous_scores[topic_id] = score
        ranked.append(
            RankedCandidate(
                topic_id=topic_id,
                docid=docid,
                rank=rank,
                score=score,
                text="",
                provenance=[{"run_id": row_run_tag}],
            )
        )

    actual_topic_ids = set(seen_ranks)
    extra = sorted(actual_topic_ids - expected_set)
    if extra:
        raise ValueError(f"topics outside the expected population: {', '.join(extra)}")
    missing = sorted(expected_set - actual_topic_ids)
    if missing:
        raise ValueError(f"missing expected topics: {', '.join(missing)}")
    return ranked


def _validate_projected_qrels(
    qrels_bytes: bytes,
    *,
    expected_sha256: str,
    expected_topic_ids: Collection[str],
) -> tuple[dict[str, dict[str, int]], str]:
    expected_topics = set(_validated_topic_ids(expected_topic_ids))
    qrels_digest = sha256(qrels_bytes).hexdigest()
    qrels: dict[str, dict[str, int]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    actual_topics: set[str] = set()

    for line_number, raw_line in enumerate(
        qrels_bytes.decode("utf-8").splitlines(),
        start=1,
    ):
        parts = raw_line.split()
        if len(parts) != 4:
            raise ValueError(f"line {line_number}: expected four qrels columns")
        topic_id, second_column, docid, grade_text = parts
        if not topic_id or not docid:
            raise ValueError(f"line {line_number}: qrels topic and document IDs must be non-empty")
        if second_column != "0":
            raise ValueError(f"line {line_number}: qrels column 2 must be 0")
        try:
            grade = int(grade_text)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: qrels grade must be an integer") from exc
        if not 0 <= grade <= 4:
            raise ValueError(f"line {line_number}: qrels grade must be in the range 0..4")
        pair = (topic_id, docid)
        if pair in seen_pairs:
            raise ValueError(
                f"line {line_number}: duplicate qrels topic/document pair {topic_id}/{docid}"
            )
        seen_pairs.add(pair)
        actual_topics.add(topic_id)
        qrels.setdefault(topic_id, {})[docid] = grade

    extra = sorted(actual_topics - expected_topics)
    if extra:
        raise ValueError(f"qrels topics outside the expected population: {', '.join(extra)}")
    missing = sorted(expected_topics - actual_topics)
    if missing:
        raise ValueError(f"qrels missing expected topics: {', '.join(missing)}")
    if qrels_digest != expected_sha256:
        raise ValueError(
            "qrels SHA-256 does not match the pinned assessor: "
            f"expected {expected_sha256}, got {qrels_digest}"
        )

    return qrels, qrels_digest


def _validated_metric_names(metric_names: Sequence[str]) -> list[str]:
    selected = list(metric_names)
    if not selected:
        raise ValueError("metric names must not be empty")
    if len(selected) != len(set(selected)):
        raise ValueError("metric names must be unique")
    for metric in selected:
        if not isinstance(metric, str) or metric.count("@") != 1:
            raise ValueError("metric must use NAME@CUTOFF")
        name, cutoff_text = metric.split("@")
        if not name or not cutoff_text:
            raise ValueError("metric must use NAME@CUTOFF")
        if name not in SUPPORTED_METRIC_NAMES:
            raise ValueError(f"unsupported metric: {name}")
        try:
            cutoff = int(cutoff_text)
        except ValueError as exc:
            raise ValueError(f"metric cutoff must be an integer: {metric}") from exc
        if cutoff <= 0:
            raise ValueError(f"metric cutoff must be positive: {metric}")
    return selected


def _paths_alias(left: Path, right: Path) -> bool:
    if left.resolve(strict=False) == right.resolve(strict=False):
        return True
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def _validate_output_path(output_path: Path, *, run_path: Path, qrels_path: Path) -> None:
    if _paths_alias(output_path, run_path) or _paths_alias(output_path, qrels_path):
        raise ValueError("output path must not alias the run or qrels input")


def evaluate_competition_retrieval_run(
    run_path: Path,
    qrels_path: Path,
    *,
    topic_ids: Collection[str],
    metric_names: Sequence[str],
    relevance_threshold: int = 2,
    expected_qrels_sha256: str = PINNED_QRELS_SHA256,
    expected_qrels_topic_ids: Collection[str] = PINNED_QRELS_TOPIC_IDS,
    assessor_variant: str = PINNED_ASSESSOR_VARIANT,
) -> dict[str, object]:
    evaluated_topic_ids = _validated_topic_ids(topic_ids)
    expected_qrels_topics = _validated_topic_ids(expected_qrels_topic_ids)
    unexpected_topics = sorted(set(evaluated_topic_ids) - set(expected_qrels_topics))
    if unexpected_topics:
        raise ValueError(
            "evaluation topics outside the qrels population: " + ", ".join(unexpected_topics)
        )
    selected_metrics = _validated_metric_names(metric_names)
    if (
        not isinstance(relevance_threshold, int)
        or isinstance(relevance_threshold, bool)
        or not 1 <= relevance_threshold <= 4
    ):
        raise ValueError("relevance threshold must be an integer from 1 through 4")

    run_bytes = run_path.read_bytes()
    qrels_bytes = qrels_path.read_bytes()
    ranked = _parse_trec_retrieval_run_snapshot(
        run_bytes,
        expected_topic_ids=evaluated_topic_ids,
    )
    qrels, qrels_digest = _validate_projected_qrels(
        qrels_bytes,
        expected_sha256=expected_qrels_sha256,
        expected_topic_ids=expected_qrels_topics,
    )
    if expected_qrels_sha256 != PINNED_QRELS_SHA256 and assessor_variant == PINNED_ASSESSOR_VARIANT:
        raise ValueError("a non-default qrels digest requires an explicit assessor variant")
    evaluation = evaluate_ranked(
        ranked,
        qrels,
        metric_names=selected_metrics,
        relevance_threshold=relevance_threshold,
        topic_ids=evaluated_topic_ids,
    )
    is_pinned_assessor = (
        expected_qrels_sha256 == PINNED_QRELS_SHA256
        and assessor_variant == PINNED_ASSESSOR_VARIANT
        and set(expected_qrels_topics) == set(PINNED_QRELS_TOPIC_IDS)
    )
    return {
        "schema_version": 1,
        "projected_development_qrels": True,
        "topic_ids": evaluated_topic_ids,
        "run": {
            "path": str(run_path),
            "sha256": sha256(run_bytes).hexdigest(),
        },
        "qrels": {
            "path": str(qrels_path),
            "sha256": qrels_digest,
        },
        "assessor": {
            "variant": assessor_variant,
            "path": PINNED_QRELS_REPO_PATH if is_pinned_assessor else str(qrels_path),
            "input_path": str(qrels_path),
            "sha256": qrels_digest,
        },
        "relevance_threshold": relevance_threshold,
        "metric_names": selected_metrics,
        "metrics": evaluation["metrics"],
        "per_topic": evaluation["per_topic"],
    }


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a retrieval run against projected development qrels."
    )
    parser.add_argument("--run", required=True, type=Path, help="Six-column TREC retrieval run")
    parser.add_argument(
        "--qrels",
        required=True,
        type=Path,
        help="Projected development qrels (not official ground truth)",
    )
    parser.add_argument(
        "--topic",
        required=True,
        action="append",
        dest="topic_ids",
        help="Expected topic ID; repeat for the exact evaluation population",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination JSON report")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metric_names",
        help="Metric name and cutoff; repeat to override the default metric suite",
    )
    parser.add_argument("--relevance-threshold", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_output_path(args.output, run_path=args.run, qrels_path=args.qrels)
        report = evaluate_competition_retrieval_run(
            args.run,
            args.qrels,
            topic_ids=args.topic_ids,
            metric_names=args.metric_names or DEFAULT_METRIC_NAMES,
            relevance_threshold=args.relevance_threshold,
        )
        _atomic_write(args.output, _canonical_json(report))
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
