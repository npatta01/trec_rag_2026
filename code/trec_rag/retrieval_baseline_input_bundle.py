"""Private authenticated inputs for retrieval-baseline GPU workers."""

from __future__ import annotations

import argparse
import errno
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Mapping, Sequence

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker
from trec_rag.mixedbread_passage_scorer import MixedbreadPassageScorer
from trec_rag.retrieval_baseline_runs import load_topic_input, topic_sort_key


SCHEMA_VERSION = "retrieval-baseline-private-input-v1"
MANIFEST_NAME = "input-manifest.json"
_TOPIC_ID = re.compile(r"rag2026-[0-9]+\Z")
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_ROOTS = {"source", "documents", "portable-scores"}
_SOURCE_RELATIVE_PATHS = (
    "topic-job-receipt.json",
    "decomposition.json",
    "decomposition/manifest.json",
    "decomposition/result.json",
    "retrieval/audit.json",
    "retrieval/complete.json",
    "retrieval/evidence-bundle.json",
    "scoring/complete.json",
    "scoring/lane_scores.jsonl",
    "scoring/selected_documents.jsonl",
    "scoring/selected_subnarrative_scores.jsonl",
    "scoring/selection.json",
    "canonical/retrieval-projection-manifest.json",
    "canonical/retrieval-projection.json",
)


class InputBundleError(ValueError):
    """A private worker input is incomplete, malformed, or corrupted."""


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


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InputBundleError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _file_receipt(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _create_or_verify(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise InputBundleError(f"conflicting immutable input: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _copy_create_only(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise InputBundleError(f"input source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        if _file_receipt(source) != _file_receipt(destination):
            raise InputBundleError(f"conflicting immutable input: {destination}") from None
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if _file_receipt(source) != _file_receipt(destination):
                    raise InputBundleError(
                        f"conflicting immutable input: {destination}"
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)


def _safe_relative(path: Path, root: Path) -> str:
    try:
        relative = PurePosixPath(path.relative_to(root).as_posix())
    except ValueError as exc:
        raise InputBundleError("bundle member escaped its root") from exc
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] not in _ALLOWED_ROOTS
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise InputBundleError("bundle member path is invalid")
    return relative.as_posix()


def seal_input_directory(
    root: Path,
    *,
    topic_ids: tuple[str, ...],
    source_run_id: str,
    cache_stats: Mapping[str, int],
) -> Path:
    """Seal all existing private members with a deterministic manifest."""

    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise InputBundleError("input root must be a regular directory")
    if (
        not topic_ids
        or len(set(topic_ids)) != len(topic_ids)
        or any(_TOPIC_ID.fullmatch(topic_id) is None for topic_id in topic_ids)
    ):
        raise InputBundleError("topic IDs must be unique official identifiers")
    ordered_topics = tuple(sorted(topic_ids, key=topic_sort_key))
    if _SAFE_RUN_ID.fullmatch(source_run_id) is None:
        raise InputBundleError("source run ID is invalid")
    normalized_stats: dict[str, int] = {}
    for name, value in cache_stats.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise InputBundleError("cache statistics are invalid")
        normalized_stats[name] = value

    members: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path == root / MANIFEST_NAME:
            continue
        if path.is_symlink():
            raise InputBundleError("bundle members cannot be symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise InputBundleError("bundle member is not a regular file")
        relative = _safe_relative(path, root)
        size, digest = _file_receipt(path)
        members.append({"path": relative, "sha256": digest, "size": size})
    if not members:
        raise InputBundleError("input directory has no members")
    manifest = {
        "cache_stats": dict(sorted(normalized_stats.items())),
        "member_count": len(members),
        "members": members,
        "schema_version": SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "topic_ids": list(ordered_topics),
    }
    manifest_path = root / MANIFEST_NAME
    _create_or_verify(manifest_path, _canonical_json(manifest))
    return manifest_path


def verify_input_directory(
    root: Path,
    *,
    expected_topics: tuple[str, ...] = (),
) -> dict[str, object]:
    """Verify manifest bytes, every member, and the complete file allowlist."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise InputBundleError("input root must be a regular directory")
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise InputBundleError("input manifest must be a regular file")
    try:
        body = manifest_path.read_bytes()
        value = json.loads(body, object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, InputBundleError) as exc:
        raise InputBundleError("input manifest is invalid") from exc
    if not isinstance(value, dict) or set(value) != {
        "cache_stats",
        "member_count",
        "members",
        "schema_version",
        "source_run_id",
        "topic_ids",
    }:
        raise InputBundleError("input manifest fields changed")
    if body != _canonical_json(value) or value["schema_version"] != SCHEMA_VERSION:
        raise InputBundleError("input manifest is not canonical or has a wrong schema")
    topic_ids = value["topic_ids"]
    if (
        not isinstance(topic_ids, list)
        or not topic_ids
        or any(not isinstance(item, str) or _TOPIC_ID.fullmatch(item) is None for item in topic_ids)
        or topic_ids != sorted(set(topic_ids), key=topic_sort_key)
    ):
        raise InputBundleError("input manifest topics are invalid")
    if expected_topics and list(sorted(expected_topics, key=topic_sort_key)) != topic_ids:
        raise InputBundleError("input manifest topic assignment differs")
    if not isinstance(value["source_run_id"], str) or _SAFE_RUN_ID.fullmatch(
        value["source_run_id"]
    ) is None:
        raise InputBundleError("input manifest source run is invalid")
    cache_stats = value["cache_stats"]
    if not isinstance(cache_stats, dict) or any(
        not isinstance(name, str)
        or not name
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in cache_stats.items()
    ):
        raise InputBundleError("input manifest cache statistics are invalid")
    members = value["members"]
    if (
        not isinstance(members, list)
        or isinstance(value["member_count"], bool)
        or value["member_count"] != len(members)
        or not members
    ):
        raise InputBundleError("input manifest member count is invalid")

    expected: dict[str, tuple[int, str]] = {}
    for raw in members:
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size"}:
            raise InputBundleError("input manifest member is invalid")
        relative = raw["path"]
        size = raw["size"]
        digest = raw["sha256"]
        if (
            not isinstance(relative, str)
            or relative in expected
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise InputBundleError("input manifest member is invalid")
        path = root.joinpath(*PurePosixPath(relative).parts)
        if _safe_relative(path, root) != relative:
            raise InputBundleError("input manifest member path changed")
        expected[relative] = (size, digest)

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path == manifest_path:
            continue
        if path.is_symlink():
            raise InputBundleError("input directory contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise InputBundleError("input directory contains a non-file member")
        relative = _safe_relative(path, root)
        actual.add(relative)
        receipt = _file_receipt(path)
        if expected.get(relative) != receipt:
            raise InputBundleError(f"input member size or digest differs: {relative}")
        if relative.startswith("documents/"):
            match = re.fullmatch(
                r"documents/v1/sha256/([0-9a-f]{2})/([0-9a-f]{64})\.utf8",
                relative,
            )
            if match is None or match.group(1) != match.group(2)[:2] or receipt[1] != match.group(2):
                raise InputBundleError("document member content address differs")
    if actual != set(expected):
        raise InputBundleError("input directory contains undeclared or missing files")
    return value


def build_input_directory(
    *,
    source_dir: Path,
    document_store_root: Path,
    score_cache_root: Path,
    topic_ids: tuple[str, ...],
    output_dir: Path,
) -> dict[str, object]:
    """Build one immutable one- or two-topic worker input from existing artifacts."""

    if not topic_ids or len(topic_ids) > 2 or len(set(topic_ids)) != len(topic_ids):
        raise InputBundleError("one or two unique topics are required")
    source_dir = Path(source_dir).resolve()
    document_store_root = Path(document_store_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise InputBundleError("output directory must be empty")
    source_run_id = source_dir.name
    topics = tuple(
        load_topic_input(source_dir, topic_id, document_store_root)
        for topic_id in sorted(topic_ids, key=topic_sort_key)
    )
    for topic in topics:
        for relative in _SOURCE_RELATIVE_PATHS:
            _copy_create_only(
                source_dir / topic.topic_id / relative,
                output_dir / "source" / source_run_id / topic.topic_id / relative,
            )
        for document in topic.documents:
            source = (
                document_store_root
                / "sha256"
                / document.content_sha256[:2]
                / f"{document.content_sha256}.utf8"
            )
            destination = (
                output_dir
                / "documents/v1/sha256"
                / document.content_sha256[:2]
                / f"{document.content_sha256}.utf8"
            )
            _copy_create_only(source, destination)

    chunker = SemanticTextChunker(
        ChunkingConfig(max_characters=3_500, overlap_characters=350)
    )
    scorer = MixedbreadPassageScorer(
        score_cache_root=score_cache_root,
        device="cpu",
        batch_size=32,
        read_only=True,
    )
    hit_pairs: list[tuple[str, str]] = []
    hits = 0
    misses = 0
    try:
        for topic in topics:
            chunks = tuple(
                chunk
                for document in topic.documents
                for chunk in chunker.split_text(document.text, document_id=document.docid)
            )
            query_texts = (topic.narrative,) + tuple(
                row.text for row in topic.subnarratives
            )
            for query_text in query_texts:
                pairs = tuple((query_text, chunk.text) for chunk in chunks)
                results = scorer.score_cache.lookup_many(pairs)
                unique: dict[str, tuple[tuple[str, str], float | None]] = {}
                for pair, result in zip(pairs, results, strict=True):
                    unique.setdefault(
                        scorer.score_cache.cache_key(
                            query_text=pair[0], text=pair[1]
                        ),
                        (pair, result),
                    )
                for pair, result in unique.values():
                    if result is None:
                        misses += 1
                    else:
                        hits += 1
                        hit_pairs.append(pair)
        portable_dir = output_dir / "portable-scores"
        portable_dir.mkdir(parents=True, exist_ok=True)
        portable_path = portable_dir / f"{scorer.score_cache.context_sha256}.jsonl"
        export_receipt = scorer.score_cache.export_portable_jsonl(
            portable_path,
            pairs=hit_pairs,
        )
    finally:
        scorer.score_cache.close()
    cache_stats = {
        "hits": hits,
        "misses": misses,
        "portable_rows": int(export_receipt["row_count"]),
    }
    manifest_path = seal_input_directory(
        output_dir,
        topic_ids=tuple(topic.topic_id for topic in topics),
        source_run_id=source_run_id,
        cache_stats=cache_stats,
    )
    verified = verify_input_directory(
        output_dir,
        expected_topics=tuple(topic.topic_id for topic in topics),
    )
    return {
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "member_count": verified["member_count"],
        "source_run_id": source_run_id,
        "topic_ids": list(topic.topic_id for topic in topics),
        "cache_stats": cache_stats,
    }


def import_portable_scores(input_dir: Path, score_cache_root: Path) -> dict[str, object]:
    """Verify an input and transactionally import its one pinned cache context."""

    verified = verify_input_directory(input_dir)
    portable = tuple(sorted((Path(input_dir) / "portable-scores").glob("*.jsonl")))
    if len(portable) != 1:
        raise InputBundleError("input must contain exactly one portable score context")
    scorer = MixedbreadPassageScorer(
        score_cache_root=score_cache_root,
        device="cuda",
        batch_size=32,
        read_only=False,
    )
    try:
        context_sha256 = scorer.score_cache.context_sha256
        receipt = scorer.score_cache.import_portable_jsonl(portable[0])
    finally:
        scorer.score_cache.close()
    return {
        "context_sha256": context_sha256,
        "inserted_count": receipt["inserted_count"],
        "source_row_count": receipt["source_row_count"],
        "topic_ids": verified["topic_ids"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument("--document-store", type=Path, required=True)
    build.add_argument("--score-cache", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--topic", action="append", dest="topic_ids", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("input_dir", type=Path)
    verify.add_argument("--topic", action="append", dest="topic_ids", default=[])
    import_scores = commands.add_parser("import-scores")
    import_scores.add_argument("input_dir", type=Path)
    import_scores.add_argument("--score-cache", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        receipt = build_input_directory(
            source_dir=args.source_dir,
            document_store_root=args.document_store,
            score_cache_root=args.score_cache,
            topic_ids=tuple(args.topic_ids),
            output_dir=args.output_dir,
        )
    elif args.command == "verify":
        verified = verify_input_directory(
            args.input_dir,
            expected_topics=tuple(args.topic_ids),
        )
        receipt = {
            "cache_stats": verified["cache_stats"],
            "manifest_sha256": sha256(
                (args.input_dir / MANIFEST_NAME).read_bytes()
            ).hexdigest(),
            "member_count": verified["member_count"],
            "source_run_id": verified["source_run_id"],
            "topic_ids": verified["topic_ids"],
        }
    else:
        receipt = import_portable_scores(args.input_dir, args.score_cache)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
