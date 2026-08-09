"""One-process, canary-gated remote candidate-core scoring orchestration."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Protocol, Sequence

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker
from trec_rag.mixedbread_passage_scorer import MixedbreadPassageScorer
from trec_rag.retrieval_candidate_core import candidate_core_from_dict
from trec_rag.retrieval_baseline_input_bundle import (
    export_portable_score_cache,
    import_portable_score_file,
    import_portable_scores,
    verify_input_directory,
)
from trec_rag.retrieval_baseline_runs import (
    TopicMatrix,
    export_runs,
    load_topic_input,
    read_topic_matrix,
    score_topic,
    topic_sort_key,
    write_topic_matrix,
)


OFFICIAL_TOPIC_IDS = tuple(f"rag2026-{index}" for index in range(119))
CANARY_TOPIC_IDS = ("rag2026-1", "rag2026-18")


class _RemoteScorer(Protocol):
    @property
    def identity(self) -> dict[str, object]: ...

    @property
    def stats(self) -> dict[str, int]: ...

    @property
    def score_cache(self) -> object: ...


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_json(value))


def _load_candidate_core(input_dir: Path, topic_id: str):
    path = Path(input_dir) / "candidate-cores" / f"{topic_id}.json"
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("candidate core must be a JSON object")
    core = candidate_core_from_dict(value)
    if core.topic_id != topic_id:
        raise ValueError("candidate-core topic identity differs")
    return core


def _compare_matrix_artifacts(first: Path, second: Path) -> None:
    for name in ("topic-matrix.jsonl", "topic-matrix-manifest.json"):
        if (Path(first) / name).read_bytes() != (Path(second) / name).read_bytes():
            raise ValueError(f"cache-only replay differs: {name}")


def _compare_run_artifacts(first: Path, second: Path) -> None:
    relative_paths = (
        "retrieval-baseline-runs-manifest.json",
        "narrative/r_output_trec_rag_2026.tsv",
        "combo/r_output_trec_rag_2026.tsv",
        "breadth/r_output_trec_rag_2026.tsv",
    )
    for relative in relative_paths:
        if (Path(first) / relative).read_bytes() != (Path(second) / relative).read_bytes():
            raise ValueError(f"cache-only run replay differs: {relative}")


def ordered_scoring_phases(
    topic_ids: tuple[str, ...],
    canary_topic_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate production assignment and return canaries before remaining topics."""

    if tuple(sorted(topic_ids, key=topic_sort_key)) != OFFICIAL_TOPIC_IDS:
        raise ValueError("remote scoring requires the exact 119-topic set")
    if canary_topic_ids != CANARY_TOPIC_IDS:
        raise ValueError("remote scoring requires the fixed canary topic order")
    remaining = tuple(topic_id for topic_id in OFFICIAL_TOPIC_IDS if topic_id not in CANARY_TOPIC_IDS)
    return CANARY_TOPIC_IDS, remaining


def run_canary_gated_scoring(
    *,
    topic_ids: tuple[str, ...],
    canary_topic_ids: tuple[str, ...],
    score_one: Callable[[str], object],
    verify_canaries: Callable[[tuple[str, ...]], object],
) -> None:
    """Score canaries, require their replay gate, then score every remaining topic."""

    if not callable(score_one) or not callable(verify_canaries):
        raise TypeError("scoring and canary verification callbacks must be callable")
    canaries, remaining = ordered_scoring_phases(topic_ids, canary_topic_ids)
    for topic_id in canaries:
        score_one(topic_id)
    verify_canaries(canaries)
    for topic_id in remaining:
        score_one(topic_id)


def run_remote_scoring(
    *,
    input_dir: Path,
    work_root: Path,
    publication_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
    scorer_factory: Callable[[Path, bool], _RemoteScorer] | None = None,
) -> dict[str, object]:
    """Score all topics in one model process with canary and final fresh replay."""

    input_dir = Path(input_dir).resolve()
    work_root = Path(work_root).resolve()
    publication_dir = Path(publication_dir).resolve()
    if len({input_dir, work_root, publication_dir}) != 3:
        raise ValueError("input, work, and publication roots must be distinct")
    if publication_dir.exists() and any(publication_dir.iterdir()):
        raise ValueError("publication directory must be empty")
    publication_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    verified = verify_input_directory(
        input_dir,
        expected_topics=OFFICIAL_TOPIC_IDS,
    )
    input_manifest_sha256 = sha256(
        (input_dir / "input-manifest.json").read_bytes()
    ).hexdigest()
    source_revision = os.environ.get("TREC_RAG_SOURCE_REVISION")
    if source_revision is None or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError(
            "remote scoring requires TREC_RAG_SOURCE_REVISION as a Git commit"
        )
    topic_ids = tuple(verified["topic_ids"])
    canary_topic_ids = tuple(verified["canary_topic_ids"])
    ordered_scoring_phases(topic_ids, canary_topic_ids)
    source_run_id = str(verified["source_run_id"])
    source_dir = input_dir / "source" / source_run_id
    document_store = input_dir / "documents/v1"
    live_cache = work_root / "live-cache"
    import_receipt = import_portable_scores(input_dir, live_cache)
    expected_hits = int(verified["cache_stats"]["hits"])
    if (
        import_receipt.get("source_row_count") != expected_hits
        or import_receipt.get("inserted_count") != expected_hits
    ):
        raise ValueError("sealed input cache hits were not imported exactly")

    if scorer_factory is None:
        scorer_factory = lambda root, read_only: MixedbreadPassageScorer(
            score_cache_root=root,
            device=device,
            batch_size=batch_size,
            read_only=read_only,
        )
    chunker = SemanticTextChunker(
        ChunkingConfig(max_characters=3_500, overlap_characters=350)
    )
    matrices_root = publication_dir / "matrices"
    replay_receipts = publication_dir / "replay-receipts"
    live_scorer = scorer_factory(live_cache, False)
    live_matrices: dict[str, TopicMatrix] = {}
    canary_replay_started = False
    canary_replay_verified = False

    def score_one(topic_id: str) -> None:
        core = _load_candidate_core(input_dir, topic_id)
        topic = load_topic_input(
            source_dir,
            topic_id,
            document_store,
            selected_docids=core.candidate_docids,
        )
        matrix = score_topic(
            topic,
            candidate_core=core,
            scorer=live_scorer,
            chunker=chunker,
        )
        write_topic_matrix(matrix, matrices_root / topic_id)
        live_matrices[topic_id] = matrix
        _write_new_json(
            matrices_root / topic_id / "score-receipt.json",
            {
                "cache_stats": dict(matrix.cache_stats),
                "candidate_count": len(matrix.documents),
                "passage_pair_count": len(matrix.passages),
                "status": "complete",
                "topic_id": topic_id,
            },
        )

    def replay_topics(
        selected_topic_ids: tuple[str, ...],
        *,
        portable_path: Path,
        replay_cache: Path,
        replay_matrices_root: Path,
        receipt_phase: str,
    ) -> tuple[TopicMatrix, ...]:
        import_portable_score_file(portable_path, replay_cache)
        replay_scorer = scorer_factory(replay_cache, True)
        replayed: list[TopicMatrix] = []
        try:
            for topic_id in selected_topic_ids:
                core = _load_candidate_core(input_dir, topic_id)
                topic = load_topic_input(
                    source_dir,
                    topic_id,
                    document_store,
                    selected_docids=core.candidate_docids,
                )
                matrix = score_topic(
                    topic,
                    candidate_core=core,
                    scorer=replay_scorer,
                    chunker=chunker,
                )
                if (
                    matrix.cache_stats.get("cache_misses") != 0
                    or matrix.cache_stats.get("model_batches") != 0
                    or matrix.cache_stats.get("cache_hits") != len(matrix.passages)
                ):
                    raise ValueError("fresh-cache replay was not completely cache-only")
                write_topic_matrix(matrix, replay_matrices_root / topic_id)
                _compare_matrix_artifacts(
                    matrices_root / topic_id,
                    replay_matrices_root / topic_id,
                )
                replayed.append(matrix)
                _write_new_json(
                    replay_receipts / receipt_phase / f"{topic_id}.json",
                    {
                        "cache_stats": dict(matrix.cache_stats),
                        "matrix_sha256": sha256(
                            (replay_matrices_root / topic_id / "topic-matrix.jsonl").read_bytes()
                        ).hexdigest(),
                        "phase": receipt_phase,
                        "status": "verified",
                        "topic_id": topic_id,
                    },
                )
        finally:
            replay_scorer.score_cache.close()  # type: ignore[attr-defined]
        return tuple(replayed)

    def verify_canaries(selected_topic_ids: tuple[str, ...]) -> None:
        nonlocal canary_replay_started, canary_replay_verified
        canary_replay_started = True
        portable_path = work_root / "canary-cache.jsonl"
        live_scorer.score_cache.export_portable_jsonl(portable_path)  # type: ignore[attr-defined]
        replay_topics(
            selected_topic_ids,
            portable_path=portable_path,
            replay_cache=work_root / "canary-replay-cache",
            replay_matrices_root=work_root / "canary-replay-matrices",
            receipt_phase="canary",
        )
        canary_replay_verified = True

    try:
        try:
            run_canary_gated_scoring(
                topic_ids=topic_ids,
                canary_topic_ids=canary_topic_ids,
                score_one=score_one,
                verify_canaries=verify_canaries,
            )
        except Exception as exc:
            if not canary_replay_verified:
                _write_new_json(
                    publication_dir / "remote-scoring-failure-receipt.json",
                    {
                        "canary_topic_ids": list(CANARY_TOPIC_IDS),
                        "failure_stage": (
                            "canary_replay"
                            if canary_replay_started
                            else "canary_scoring"
                        ),
                        "failure_type": type(exc).__name__,
                        "input_manifest_sha256": input_manifest_sha256,
                        "schema_version": "retrieval-baseline-remote-failure-v1",
                        "scored_topic_ids": list(live_matrices),
                        "source_revision": source_revision,
                        "status": "failed",
                    },
                )
            raise
    finally:
        live_scorer.score_cache.close()  # type: ignore[attr-defined]

    complete_scores = publication_dir / "portable-scores/complete.jsonl"
    cache_export_receipt = export_portable_score_cache(live_cache, complete_scores)
    expected_rows = int(verified["cache_stats"]["hits"]) + int(
        verified["cache_stats"]["misses"]
    )
    if cache_export_receipt.get("row_count") != expected_rows:
        raise ValueError("complete remote cache row count differs from sealed workload")
    _write_new_json(
        publication_dir / "cache-export-receipt.json", cache_export_receipt
    )
    replayed = replay_topics(
        OFFICIAL_TOPIC_IDS,
        portable_path=complete_scores,
        replay_cache=work_root / "final-replay-cache",
        replay_matrices_root=work_root / "final-replay-matrices",
        receipt_phase="final",
    )
    matrices = tuple(
        read_topic_matrix(matrices_root / topic_id) for topic_id in OFFICIAL_TOPIC_IDS
    )
    if tuple(matrix.topic_id for matrix in replayed) != OFFICIAL_TOPIC_IDS:
        raise ValueError("final replay topic order differs")
    runs_root = publication_dir / "runs"
    export_runs(matrices, runs_root)
    replay_runs_root = work_root / "final-replay-runs"
    export_runs(replayed, replay_runs_root)
    _compare_run_artifacts(runs_root, replay_runs_root)

    receipt = {
        "cache_stats": dict(verified["cache_stats"]),
        "canary_topic_ids": list(CANARY_TOPIC_IDS),
        "input_manifest_sha256": input_manifest_sha256,
        "portable_cache_row_count": cache_export_receipt["row_count"],
        "portable_cache_sha256": cache_export_receipt["sha256"],
        "schema_version": "retrieval-baseline-remote-scoring-v1",
        "status": "complete",
        "topic_count": len(matrices),
        "topic_ids": list(OFFICIAL_TOPIC_IDS),
    }
    _write_new_json(publication_dir / "remote-scoring-receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--publication-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = run_remote_scoring(
        input_dir=args.input_dir,
        work_root=args.work_root,
        publication_dir=args.publication_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "CANARY_TOPIC_IDS",
    "OFFICIAL_TOPIC_IDS",
    "ordered_scoring_phases",
    "run_remote_scoring",
    "run_canary_gated_scoring",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
