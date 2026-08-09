"""Verify remote retrieval scoring, merge its cache, and replay locally."""

from __future__ import annotations

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Callable, Sequence

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker
from trec_rag.mixedbread_passage_scorer import MixedbreadPassageScorer
from trec_rag.retrieval_candidate_core import candidate_core_from_dict
from trec_rag.retrieval_baseline_input_bundle import (
    import_portable_score_file,
    verify_input_directory,
)
from trec_rag.retrieval_baseline_remote_worker import OFFICIAL_TOPIC_IDS
from trec_rag.retrieval_baseline_runs import (
    TopicMatrix,
    export_runs,
    load_topic_input,
    read_topic_matrix,
    score_topic,
    write_topic_matrix,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


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


def verify_publication_closure(
    publication_dir: Path,
    *,
    expected_input_manifest_sha256: str,
    expected_source_revision: str,
) -> dict[str, object]:
    """Verify the manifest-last publication and its exact checksum closure."""

    root = Path(publication_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("publication root must be a regular directory")
    if _SHA256.fullmatch(expected_input_manifest_sha256) is None:
        raise ValueError("expected input manifest identity is invalid")
    if _REVISION.fullmatch(expected_source_revision) is None:
        raise ValueError("expected source revision is invalid")
    sums_path = root / "SHA256SUMS"
    manifest_path = root / "publication-manifest.json"
    if any(path.is_symlink() or not path.is_file() for path in (sums_path, manifest_path)):
        raise ValueError("publication checksum or manifest file is invalid")
    expected: dict[str, str] = {}
    prior: str | None = None
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", line)
        if match is None:
            raise ValueError("SHA256SUMS contains an invalid row")
        relative = match.group(2)
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or relative in expected
            or (prior is not None and relative <= prior)
        ):
            raise ValueError("SHA256SUMS paths are unsafe, repeated, or unsorted")
        path = root.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError("SHA256SUMS declares a missing or unsafe member")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != match.group(1):
            raise ValueError("SHA256SUMS member digest differs")
        expected[relative] = digest
        prior = relative
    if not expected:
        raise ValueError("SHA256SUMS cannot be empty")
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("publication contains a symlink")
        if path.is_file() and path not in {sums_path, manifest_path}:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise ValueError("publication file set differs from SHA256SUMS")
    try:
        manifest_body = manifest_path.read_bytes()
        manifest = json.loads(manifest_body, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("publication manifest is invalid") from exc
    manifest_fields = {
        "input_manifest_sha256",
        "remote_scoring_receipt_sha256",
        "schema_version",
        "sha256s_sha256",
        "source_revision",
        "status",
        "task_name",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != manifest_fields
        or manifest_body != _canonical_json(manifest)
        or manifest.get("schema_version") != "retrieval-baseline-publication-v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("publication manifest schema or canonical bytes differ")
    if manifest.get("input_manifest_sha256") != expected_input_manifest_sha256:
        raise ValueError("publication input manifest identity differs")
    if manifest.get("source_revision") != expected_source_revision:
        raise ValueError("publication source revision differs")
    if manifest.get("sha256s_sha256") != sha256(sums_path.read_bytes()).hexdigest():
        raise ValueError("publication SHA256SUMS digest differs")
    remote_digest = manifest.get("remote_scoring_receipt_sha256")
    if not isinstance(remote_digest, str) or _SHA256.fullmatch(remote_digest) is None:
        raise ValueError("publication remote receipt identity is invalid")
    return manifest


def merge_portable_scores(
    portable_scores: Path,
    shared_cache_root: Path,
) -> dict[str, object]:
    """Strictly import portable scores under the shared local writer lock."""

    portable_scores = Path(portable_scores)
    if portable_scores.is_symlink() or not portable_scores.is_file():
        raise ValueError("portable score source must be a regular file")
    shared_cache_root = Path(shared_cache_root)
    if shared_cache_root.is_symlink() or (
        shared_cache_root.exists() and not shared_cache_root.is_dir()
    ):
        raise ValueError("shared score cache root must be a regular directory")
    shared_cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = shared_cache_root / ".retrieval-baseline-merge.lock"
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        scorer = MixedbreadPassageScorer(
            shared_cache_root,
            device="cpu",
            batch_size=32,
            read_only=False,
        )
        try:
            before = int(
                scorer.score_cache.connection.execute(
                    "SELECT COUNT(*) FROM scores"
                ).fetchone()[0]
            )
            receipt = scorer.score_cache.import_portable_jsonl(
                portable_scores,
                conflict_policy="strict",
            )
            after = int(
                scorer.score_cache.connection.execute(
                    "SELECT COUNT(*) FROM scores"
                ).fetchone()[0]
            )
            if after - before != receipt["inserted_count"]:
                raise ValueError("shared cache row delta differs from import receipt")
            public_receipt = {
                key: value
                for key, value in receipt.items()
                if not key.startswith("_")
            }
            return {
                **public_receipt,
                "after_row_count": after,
                "before_row_count": before,
                "context_sha256": scorer.score_cache.context_sha256,
                "lock_path": str(lock_path.resolve()),
            }
        finally:
            scorer.score_cache.close()
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def _compare_bytes(first: Path, second: Path, relative_paths: tuple[str, ...]) -> None:
    for relative in relative_paths:
        if (first / relative).read_bytes() != (second / relative).read_bytes():
            raise ValueError(f"cache-only replay differs: {relative}")


def _load_core(input_dir: Path, topic_id: str):
    path = input_dir / "candidate-cores" / f"{topic_id}.json"
    value = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("candidate core must be a JSON object")
    core = candidate_core_from_dict(value)
    if core.topic_id != topic_id:
        raise ValueError("candidate-core topic identity differs")
    return core


def replay_cache_only(
    *,
    input_dir: Path,
    publication_dir: Path,
    score_cache_root: Path,
    output_dir: Path,
    source_revision: str,
    scorer_factory: Callable[[Path], object] | None = None,
) -> dict[str, object]:
    """Rebuild all remote matrices and runs from one read-only local cache."""

    input_dir = Path(input_dir)
    publication_dir = Path(publication_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("cache-only replay output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    verified_input = verify_input_directory(
        input_dir,
        expected_topics=OFFICIAL_TOPIC_IDS,
    )
    source_dir = input_dir / "source" / str(verified_input["source_run_id"])
    document_store = input_dir / "documents/v1"
    if scorer_factory is None:
        scorer_factory = lambda root: MixedbreadPassageScorer(
            root,
            device="cuda",
            batch_size=32,
            read_only=True,
        )
    scorer = scorer_factory(Path(score_cache_root))
    chunker = SemanticTextChunker(
        ChunkingConfig(max_characters=3_500, overlap_characters=350)
    )
    matrices: list[TopicMatrix] = []
    total_hits = 0
    try:
        for topic_id in OFFICIAL_TOPIC_IDS:
            core = _load_core(input_dir, topic_id)
            topic = load_topic_input(
                source_dir,
                topic_id,
                document_store,
                selected_docids=core.candidate_docids,
            )
            matrix = score_topic(
                topic,
                candidate_core=core,
                scorer=scorer,  # type: ignore[arg-type]
                chunker=chunker,
            )
            if (
                matrix.cache_stats.get("cache_misses") != 0
                or matrix.cache_stats.get("model_batches") != 0
                or matrix.cache_stats.get("cache_hits") != len(matrix.passages)
            ):
                raise ValueError("local replay was not completely cache-only")
            write_topic_matrix(matrix, output_dir / "matrices" / topic_id)
            _compare_bytes(
                publication_dir / "matrices" / topic_id,
                output_dir / "matrices" / topic_id,
                ("topic-matrix.jsonl", "topic-matrix-manifest.json"),
            )
            total_hits += int(matrix.cache_stats["cache_hits"])
            matrices.append(matrix)
    finally:
        scorer.score_cache.close()  # type: ignore[attr-defined]
    previous_revision = os.environ.get("TREC_RAG_SOURCE_REVISION")
    os.environ["TREC_RAG_SOURCE_REVISION"] = source_revision
    try:
        export_runs(tuple(matrices), output_dir / "runs")
    finally:
        if previous_revision is None:
            os.environ.pop("TREC_RAG_SOURCE_REVISION", None)
        else:
            os.environ["TREC_RAG_SOURCE_REVISION"] = previous_revision
    _compare_bytes(
        publication_dir / "runs",
        output_dir / "runs",
        (
            "retrieval-baseline-runs-manifest.json",
            "narrative/r_output_trec_rag_2026.tsv",
            "combo/r_output_trec_rag_2026.tsv",
            "breadth/r_output_trec_rag_2026.tsv",
        ),
    )
    return {
        "cache_hits": total_hits,
        "cache_misses": 0,
        "model_batches": 0,
        "status": "verified",
        "topic_count": len(matrices),
    }


def collect_remote_scoring(
    *,
    publication_dir: Path,
    input_dir: Path,
    expected_input_manifest_sha256: str,
    expected_source_revision: str,
    shared_cache_root: Path,
    work_root: Path,
    output_dir: Path,
    scorer_factory: Callable[[Path], object] | None = None,
) -> dict[str, object]:
    """Verify, replay, merge, and reproduce one complete remote publication."""

    publication_dir = Path(publication_dir).resolve()
    input_dir = Path(input_dir).resolve()
    shared_cache_root = Path(shared_cache_root).resolve()
    work_root = Path(work_root).resolve()
    output_dir = Path(output_dir).resolve()
    roots = (
        publication_dir,
        input_dir,
        shared_cache_root,
        work_root,
        output_dir,
    )
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("collection roots must not overlap")
    if work_root.exists() and any(work_root.iterdir()):
        raise ValueError("collection work root must be empty")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("collection output root must be empty")
    work_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    publication_manifest = verify_publication_closure(
        publication_dir,
        expected_input_manifest_sha256=expected_input_manifest_sha256,
        expected_source_revision=expected_source_revision,
    )
    input_manifest = verify_input_directory(
        input_dir,
        expected_topics=OFFICIAL_TOPIC_IDS,
    )
    if sha256((input_dir / "input-manifest.json").read_bytes()).hexdigest() != (
        expected_input_manifest_sha256
    ):
        raise ValueError("local sealed input manifest differs from approved identity")
    remote_receipt_path = publication_dir / "remote-scoring-receipt.json"
    remote_receipt_body = remote_receipt_path.read_bytes()
    if sha256(remote_receipt_body).hexdigest() != publication_manifest[
        "remote_scoring_receipt_sha256"
    ]:
        raise ValueError("remote scoring receipt digest differs")
    remote_receipt = json.loads(
        remote_receipt_body, object_pairs_hook=_unique_object
    )
    if (
        not isinstance(remote_receipt, dict)
        or remote_receipt.get("schema_version")
        != "retrieval-baseline-remote-scoring-v1"
        or remote_receipt.get("status") != "complete"
        or remote_receipt.get("topic_count") != 119
        or remote_receipt.get("topic_ids") != list(OFFICIAL_TOPIC_IDS)
        or remote_receipt.get("input_manifest_sha256")
        != expected_input_manifest_sha256
    ):
        raise ValueError("remote scoring receipt is incomplete or mismatched")
    portable = publication_dir / "portable-scores/complete.jsonl"
    if sha256(portable.read_bytes()).hexdigest() != remote_receipt.get(
        "portable_cache_sha256"
    ):
        raise ValueError("remote portable cache digest differs")
    matrix_topics = tuple(
        sorted(
            (
                path.name
                for path in (publication_dir / "matrices").iterdir()
                if path.is_dir()
            ),
            key=lambda value: int(value.removeprefix("rag2026-")),
        )
    )
    if matrix_topics != OFFICIAL_TOPIC_IDS:
        raise ValueError("remote publication matrix topic set differs")
    for topic_id in OFFICIAL_TOPIC_IDS:
        read_topic_matrix(publication_dir / "matrices" / topic_id)

    fresh_cache = work_root / "fresh-cache"
    fresh_import = import_portable_score_file(portable, fresh_cache)
    if fresh_import.get("source_row_count") != remote_receipt.get(
        "portable_cache_row_count"
    ):
        raise ValueError("fresh replay cache row count differs")
    fresh_replay = replay_cache_only(
        input_dir=input_dir,
        publication_dir=publication_dir,
        score_cache_root=fresh_cache,
        output_dir=work_root / "fresh-replay",
        source_revision=expected_source_revision,
        scorer_factory=scorer_factory,
    )

    merge_receipt = merge_portable_scores(portable, shared_cache_root)
    (output_dir / "merge-receipt.json").write_bytes(
        _canonical_json(merge_receipt)
    )
    final_replay = replay_cache_only(
        input_dir=input_dir,
        publication_dir=publication_dir,
        score_cache_root=shared_cache_root,
        output_dir=output_dir / "final",
        source_revision=expected_source_revision,
        scorer_factory=scorer_factory,
    )
    receipt = {
        "expected_input_manifest_sha256": expected_input_manifest_sha256,
        "expected_source_revision": expected_source_revision,
        "fresh_replay": fresh_replay,
        "input_source_export_manifest_sha256": input_manifest[
            "source_export_manifest_sha256"
        ],
        "merge_receipt_sha256": sha256(
            (output_dir / "merge-receipt.json").read_bytes()
        ).hexdigest(),
        "portable_cache_sha256": remote_receipt["portable_cache_sha256"],
        "schema_version": "retrieval-baseline-local-collection-v1",
        "shared_cache_final_replay": final_replay,
        "status": "complete",
        "topic_count": 119,
    }
    (output_dir / "collection-receipt.json").write_bytes(_canonical_json(receipt))
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--shared-cache", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = collect_remote_scoring(
        publication_dir=args.publication_dir,
        input_dir=args.input_dir,
        expected_input_manifest_sha256=args.input_manifest_sha256,
        expected_source_revision=args.source_revision,
        shared_cache_root=args.shared_cache,
        work_root=args.work_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "collect_remote_scoring",
    "main",
    "merge_portable_scores",
    "replay_cache_only",
    "verify_publication_closure",
]


if __name__ == "__main__":
    raise SystemExit(main())
