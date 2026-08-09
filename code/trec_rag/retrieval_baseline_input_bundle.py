"""Private authenticated inputs for retrieval-baseline GPU workers."""

from __future__ import annotations

import argparse
import errno
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from typing import Mapping, Sequence

from trec_rag.chunking import ChunkingConfig, SemanticTextChunker
from trec_rag.mixedbread_passage_scorer import MixedbreadPassageScorer
from trec_rag.retrieval_candidate_core import (
    candidate_core_from_dict,
    candidate_core_to_dict,
    derive_candidate_core,
)
from trec_rag.retrieval_baseline_runs import load_topic_input, topic_sort_key


SCHEMA_VERSION = "retrieval-baseline-private-input-v2"
MANIFEST_NAME = "input-manifest.json"
_TOPIC_ID = re.compile(r"rag2026-[0-9]+\Z")
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_ROOTS = {"source", "documents", "portable-scores", "candidate-cores"}
_OFFICIAL_TOPIC_IDS = tuple(f"rag2026-{index}" for index in range(119))
_CANARY_TOPIC_IDS = ("rag2026-1", "rag2026-18")
_MAX_ARCHIVE_MEMBERS = 200_000
_MAX_ARCHIVE_BYTES = 5_000_000_000
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
_TOPIC_STAT_FIELDS = {
    "cache_hits",
    "cache_misses",
    "candidate_core_sha256",
    "candidate_count",
    "chunk_count",
    "document_semantic_pair_count",
    "first_seen_cache_key_count",
    "passage_pair_count",
    "reused_prior_topic_key_count",
    "semantic_unit_count",
    "unique_cache_key_count",
}


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


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    try:
        body = Path(path).read_bytes()
        value = json.loads(body, object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, InputBundleError) as exc:
        raise InputBundleError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise InputBundleError(f"{label} must be a JSON object")
    return value, body


def _normalized_topic_stats(
    topic_ids: tuple[str, ...],
    topic_stats: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if not isinstance(topic_stats, Mapping) or set(topic_stats) != set(topic_ids):
        raise InputBundleError("topic statistics must exactly cover selected topics")
    normalized: dict[str, dict[str, object]] = {}
    for topic_id in topic_ids:
        raw = topic_stats[topic_id]
        if not isinstance(raw, Mapping) or set(raw) != _TOPIC_STAT_FIELDS:
            raise InputBundleError("topic statistic fields changed")
        row: dict[str, object] = {}
        for name in sorted(_TOPIC_STAT_FIELDS):
            value = raw[name]
            if name == "candidate_core_sha256":
                if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                    raise InputBundleError("candidate-core statistic digest is invalid")
                row[name] = value
            else:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise InputBundleError("topic statistics must be nonnegative integers")
                row[name] = value
        if row["candidate_count"] == 0 or row["semantic_unit_count"] == 0:
            raise InputBundleError("topic candidate and semantic-unit counts must be positive")
        if row["cache_hits"] + row["cache_misses"] != row[
            "first_seen_cache_key_count"
        ]:
            raise InputBundleError("topic first-seen cache accounting differs")
        if row["first_seen_cache_key_count"] + row[
            "reused_prior_topic_key_count"
        ] != row["unique_cache_key_count"]:
            raise InputBundleError("topic unique cache-key accounting differs")
        if row["unique_cache_key_count"] > row["passage_pair_count"]:
            raise InputBundleError("topic unique cache keys exceed passage pairs")
        if row["document_semantic_pair_count"] != row["candidate_count"] * row[
            "semantic_unit_count"
        ]:
            raise InputBundleError("topic document/semantic pair count differs")
        normalized[topic_id] = row
    return normalized


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
    canary_topic_ids: tuple[str, ...],
    topic_stats: Mapping[str, Mapping[str, object]],
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
    if (
        len(set(canary_topic_ids)) != len(canary_topic_ids)
        or any(topic_id not in ordered_topics for topic_id in canary_topic_ids)
        or tuple(canary_topic_ids)
        != tuple(topic_id for topic_id in _CANARY_TOPIC_IDS if topic_id in ordered_topics)
    ):
        raise InputBundleError("canary topics differ from the fixed selected subset")
    normalized_topic_stats = _normalized_topic_stats(
        ordered_topics, topic_stats
    )
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
    required_cache_stats = {
        "candidate_documents",
        "document_semantic_pairs",
        "hits",
        "misses",
        "passage_pairs",
        "portable_rows",
        "unique_cache_keys",
    }
    if set(normalized_stats) != required_cache_stats:
        raise InputBundleError("cache statistic fields changed")
    if normalized_stats["hits"] + normalized_stats["misses"] != normalized_stats[
        "unique_cache_keys"
    ]:
        raise InputBundleError("global unique cache-key accounting differs")
    if normalized_stats["portable_rows"] != normalized_stats["hits"]:
        raise InputBundleError("portable rows must equal existing cache hits")
    if normalized_stats["candidate_documents"] != sum(
        int(row["candidate_count"]) for row in normalized_topic_stats.values()
    ):
        raise InputBundleError("global candidate-document count differs")
    if normalized_stats["document_semantic_pairs"] != sum(
        int(row["document_semantic_pair_count"])
        for row in normalized_topic_stats.values()
    ):
        raise InputBundleError("global document/semantic pair count differs")
    if normalized_stats["passage_pairs"] != sum(
        int(row["passage_pair_count"]) for row in normalized_topic_stats.values()
    ):
        raise InputBundleError("global passage-pair count differs")
    if normalized_stats["hits"] != sum(
        int(row["cache_hits"]) for row in normalized_topic_stats.values()
    ) or normalized_stats["misses"] != sum(
        int(row["cache_misses"]) for row in normalized_topic_stats.values()
    ):
        raise InputBundleError("global cache hit/miss accounting differs")

    export_manifest_path = root / "source" / source_run_id / "retrieval_export_manifest.json"
    export_manifest, export_manifest_body = _read_json_object(
        export_manifest_path, label="source export manifest"
    )
    export_commit = export_manifest.get("export_code_commit")
    if (
        export_manifest.get("run_id") != source_run_id
        or not isinstance(export_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", export_commit) is None
    ):
        raise InputBundleError("source export identity differs")
    selected_export_topics = export_manifest.get("selected_topic_ids")
    if (
        not isinstance(selected_export_topics, list)
        or any(not isinstance(item, str) for item in selected_export_topics)
        or not set(ordered_topics).issubset(selected_export_topics)
    ):
        raise InputBundleError("source export does not contain every selected topic")

    for topic_id in ordered_topics:
        core_path = root / "candidate-cores" / f"{topic_id}.json"
        core_value, core_body = _read_json_object(core_path, label="candidate core")
        try:
            core = candidate_core_from_dict(core_value)
        except (TypeError, ValueError) as exc:
            raise InputBundleError("candidate core is invalid") from exc
        if core.topic_id != topic_id or core_body != _canonical_json(
            candidate_core_to_dict(core)
        ):
            raise InputBundleError("candidate core identity or canonical bytes differ")
        if normalized_topic_stats[topic_id]["candidate_count"] != len(
            core.candidate_docids
        ) or normalized_topic_stats[topic_id]["candidate_core_sha256"] != sha256(
            core_body
        ).hexdigest():
            raise InputBundleError("candidate-core topic statistics differ")

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
        "canary_topic_ids": list(canary_topic_ids),
        "member_count": len(members),
        "members": members,
        "schema_version": SCHEMA_VERSION,
        "source_export_code_commit": export_commit,
        "source_export_manifest_sha256": sha256(export_manifest_body).hexdigest(),
        "source_run_id": source_run_id,
        "topic_ids": list(ordered_topics),
        "topic_stats": normalized_topic_stats,
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
        "canary_topic_ids",
        "member_count",
        "members",
        "schema_version",
        "source_export_code_commit",
        "source_export_manifest_sha256",
        "source_run_id",
        "topic_ids",
        "topic_stats",
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
    canary_topic_ids = value["canary_topic_ids"]
    if (
        not isinstance(canary_topic_ids, list)
        or any(not isinstance(item, str) for item in canary_topic_ids)
        or canary_topic_ids
        != [topic_id for topic_id in _CANARY_TOPIC_IDS if topic_id in topic_ids]
    ):
        raise InputBundleError("input manifest canary assignment differs")
    if not isinstance(value["source_run_id"], str) or _SAFE_RUN_ID.fullmatch(
        value["source_run_id"]
    ) is None:
        raise InputBundleError("input manifest source run is invalid")
    if (
        not isinstance(value["source_export_code_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", value["source_export_code_commit"])
        is None
        or not isinstance(value["source_export_manifest_sha256"], str)
        or _SHA256.fullmatch(value["source_export_manifest_sha256"]) is None
    ):
        raise InputBundleError("input manifest source export identity is invalid")
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
    normalized_topic_stats = _normalized_topic_stats(
        tuple(topic_ids),
        value["topic_stats"] if isinstance(value["topic_stats"], dict) else {},
    )
    if value["topic_stats"] != normalized_topic_stats:
        raise InputBundleError("input manifest topic statistics are not canonical")
    required_cache_stats = {
        "candidate_documents",
        "document_semantic_pairs",
        "hits",
        "misses",
        "passage_pairs",
        "portable_rows",
        "unique_cache_keys",
    }
    if set(cache_stats) != required_cache_stats:
        raise InputBundleError("input manifest cache statistic fields changed")
    if (
        cache_stats["hits"] + cache_stats["misses"]
        != cache_stats["unique_cache_keys"]
        or cache_stats["portable_rows"] != cache_stats["hits"]
        or cache_stats["candidate_documents"]
        != sum(int(row["candidate_count"]) for row in normalized_topic_stats.values())
        or cache_stats["document_semantic_pairs"]
        != sum(
            int(row["document_semantic_pair_count"])
            for row in normalized_topic_stats.values()
        )
        or cache_stats["passage_pairs"]
        != sum(int(row["passage_pair_count"]) for row in normalized_topic_stats.values())
        or cache_stats["hits"]
        != sum(int(row["cache_hits"]) for row in normalized_topic_stats.values())
        or cache_stats["misses"]
        != sum(int(row["cache_misses"]) for row in normalized_topic_stats.values())
    ):
        raise InputBundleError("input manifest global cache accounting differs")
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
    source_run_id = str(value["source_run_id"])
    export_path = root / "source" / source_run_id / "retrieval_export_manifest.json"
    export_manifest, export_body = _read_json_object(
        export_path, label="source export manifest"
    )
    if (
        sha256(export_body).hexdigest() != value["source_export_manifest_sha256"]
        or export_manifest.get("run_id") != source_run_id
        or export_manifest.get("export_code_commit")
        != value["source_export_code_commit"]
        or not set(topic_ids).issubset(export_manifest.get("selected_topic_ids", []))
    ):
        raise InputBundleError("source export identity differs from input manifest")
    for topic_id in topic_ids:
        core_path = root / "candidate-cores" / f"{topic_id}.json"
        core_value, core_body = _read_json_object(core_path, label="candidate core")
        try:
            core = candidate_core_from_dict(core_value)
        except (TypeError, ValueError) as exc:
            raise InputBundleError("candidate core is invalid") from exc
        topic_stat = normalized_topic_stats[topic_id]
        if (
            core.topic_id != topic_id
            or core_body != _canonical_json(candidate_core_to_dict(core))
            or sha256(core_body).hexdigest() != topic_stat["candidate_core_sha256"]
            or len(core.candidate_docids) != topic_stat["candidate_count"]
        ):
            raise InputBundleError("candidate core differs from input manifest")
    return value


def build_input_directory(
    *,
    source_dir: Path,
    document_store_root: Path,
    score_cache_root: Path,
    topic_ids: tuple[str, ...],
    output_dir: Path,
    require_official_topic_set: bool = False,
) -> dict[str, object]:
    """Build one immutable candidate-only worker input from existing artifacts."""

    if not isinstance(require_official_topic_set, bool):
        raise TypeError("require_official_topic_set must be boolean")
    if (
        not topic_ids
        or len(set(topic_ids)) != len(topic_ids)
        or any(_TOPIC_ID.fullmatch(topic_id) is None for topic_id in topic_ids)
    ):
        raise InputBundleError("unique official topic identifiers are required")
    ordered_topic_ids = tuple(sorted(topic_ids, key=topic_sort_key))
    if require_official_topic_set and ordered_topic_ids != _OFFICIAL_TOPIC_IDS:
        raise InputBundleError("production input requires the exact 119-topic set")
    source_dir = Path(source_dir).resolve()
    document_store_root = Path(document_store_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise InputBundleError("output directory must be empty")
    source_run_id = source_dir.name
    export_manifest, export_body = _read_json_object(
        source_dir / "retrieval_export_manifest.json",
        label="source export manifest",
    )
    export_topics = export_manifest.get("selected_topic_ids")
    if (
        export_manifest.get("run_id") != source_run_id
        or not isinstance(export_manifest.get("export_code_commit"), str)
        or re.fullmatch(
            r"[0-9a-f]{40}", str(export_manifest.get("export_code_commit"))
        )
        is None
        or not isinstance(export_topics, list)
        or any(not isinstance(topic_id, str) for topic_id in export_topics)
        or not set(ordered_topic_ids).issubset(export_topics)
    ):
        raise InputBundleError("source export identity or selected topics differ")
    if require_official_topic_set and tuple(export_topics) != _OFFICIAL_TOPIC_IDS:
        raise InputBundleError("source export does not authenticate the 119-topic set")
    _copy_create_only(
        source_dir / "retrieval_export_manifest.json",
        output_dir / "source" / source_run_id / "retrieval_export_manifest.json",
    )
    topics = tuple(
        load_topic_input(source_dir, topic_id, document_store_root)
        for topic_id in ordered_topic_ids
    )
    candidate_cores = {}
    for topic in topics:
        candidate_core = derive_candidate_core(
            topic_id=topic.topic_id,
            lane_scores_path=source_dir
            / topic.topic_id
            / "scoring/lane_scores.jsonl",
            expected_lane_names=("original",)
            + tuple(
                f"facet:{row.subnarrative_id}:text"
                for row in topic.subnarratives
            ),
            expected_docids=frozenset(row.docid for row in topic.documents),
            best_retrieval_ranks={
                row.docid: row.best_retrieval_rank for row in topic.documents
            },
        )
        if candidate_core.lane_scores_sha256 != topic.source_sha256s.get(
            "scoring/lane_scores.jsonl"
        ):
            raise InputBundleError("derived candidate core differs from source hash chain")
        candidate_cores[topic.topic_id] = candidate_core
        _create_or_verify(
            output_dir / "candidate-cores" / f"{topic.topic_id}.json",
            _canonical_json(candidate_core_to_dict(candidate_core)),
        )
        for relative in _SOURCE_RELATIVE_PATHS:
            _copy_create_only(
                source_dir / topic.topic_id / relative,
                output_dir / "source" / source_run_id / topic.topic_id / relative,
            )
        documents_by_id = {row.docid: row for row in topic.documents}
        for docid in candidate_core.candidate_docids:
            document = documents_by_id[docid]
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
    seen_cache_keys: set[str] = set()
    hits = 0
    misses = 0
    passage_pairs = 0
    candidate_documents = 0
    document_semantic_pairs = 0
    topic_stats: dict[str, dict[str, object]] = {}
    try:
        for topic in topics:
            candidate_core = candidate_cores[topic.topic_id]
            documents_by_id = {row.docid: row for row in topic.documents}
            candidate_documents_for_topic = tuple(
                documents_by_id[docid] for docid in candidate_core.candidate_docids
            )
            chunks = tuple(
                chunk
                for document in candidate_documents_for_topic
                for chunk in chunker.split_text(document.text, document_id=document.docid)
            )
            query_texts = (topic.narrative,) + tuple(
                row.text for row in topic.subnarratives
            )
            topic_pair_count = len(chunks) * len(query_texts)
            topic_seen_keys: set[str] = set()
            topic_hits = 0
            topic_misses = 0
            reused_prior_topic_keys = 0
            for query_text in query_texts:
                pairs = tuple((query_text, chunk.text) for chunk in chunks)
                results = scorer.score_cache.lookup_many(pairs)
                for pair, result in zip(pairs, results, strict=True):
                    cache_key = scorer.score_cache.cache_key(
                        query_text=pair[0], text=pair[1]
                    )
                    if cache_key in topic_seen_keys:
                        continue
                    topic_seen_keys.add(cache_key)
                    if cache_key in seen_cache_keys:
                        reused_prior_topic_keys += 1
                        continue
                    seen_cache_keys.add(cache_key)
                    if result is None:
                        misses += 1
                        topic_misses += 1
                    else:
                        hits += 1
                        topic_hits += 1
                        hit_pairs.append(pair)
            candidate_count = len(candidate_core.candidate_docids)
            semantic_unit_count = len(query_texts)
            candidate_documents += candidate_count
            document_semantic_pairs += candidate_count * semantic_unit_count
            passage_pairs += topic_pair_count
            core_path = output_dir / "candidate-cores" / f"{topic.topic_id}.json"
            topic_stats[topic.topic_id] = {
                "cache_hits": topic_hits,
                "cache_misses": topic_misses,
                "candidate_core_sha256": sha256(core_path.read_bytes()).hexdigest(),
                "candidate_count": candidate_count,
                "chunk_count": len(chunks),
                "document_semantic_pair_count": candidate_count
                * semantic_unit_count,
                "first_seen_cache_key_count": topic_hits + topic_misses,
                "passage_pair_count": topic_pair_count,
                "reused_prior_topic_key_count": reused_prior_topic_keys,
                "semantic_unit_count": semantic_unit_count,
                "unique_cache_key_count": len(topic_seen_keys),
            }
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
        "candidate_documents": candidate_documents,
        "document_semantic_pairs": document_semantic_pairs,
        "hits": hits,
        "misses": misses,
        "passage_pairs": passage_pairs,
        "portable_rows": int(export_receipt["row_count"]),
        "unique_cache_keys": len(seen_cache_keys),
    }
    manifest_path = seal_input_directory(
        output_dir,
        topic_ids=tuple(topic.topic_id for topic in topics),
        source_run_id=source_run_id,
        cache_stats=cache_stats,
        canary_topic_ids=tuple(
            topic_id for topic_id in _CANARY_TOPIC_IDS if topic_id in ordered_topic_ids
        ),
        topic_stats=topic_stats,
    )
    verified = verify_input_directory(
        output_dir,
        expected_topics=tuple(topic.topic_id for topic in topics),
    )
    return {
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "member_count": verified["member_count"],
        "source_run_id": source_run_id,
        "source_export_manifest_sha256": sha256(export_body).hexdigest(),
        "topic_ids": list(topic.topic_id for topic in topics),
        "topic_stats": verified["topic_stats"],
        "cache_stats": cache_stats,
    }


def import_portable_scores(input_dir: Path, score_cache_root: Path) -> dict[str, object]:
    """Verify an input and transactionally import its one pinned cache context."""

    verified = verify_input_directory(input_dir)
    portable = tuple(sorted((Path(input_dir) / "portable-scores").glob("*.jsonl")))
    if len(portable) != 1:
        raise InputBundleError("input must contain exactly one portable score context")
    receipt = import_portable_score_file(portable[0], score_cache_root)
    return {**receipt, "topic_ids": verified["topic_ids"]}


def export_portable_score_cache(
    score_cache_root: Path,
    destination: Path,
) -> dict[str, object]:
    """Export the complete pinned cache context as immutable portable JSONL."""

    destination = Path(destination)
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise InputBundleError("portable score destination must be a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    scorer = MixedbreadPassageScorer(
        score_cache_root=score_cache_root,
        device="cpu",
        batch_size=32,
        read_only=True,
    )
    try:
        receipt = scorer.score_cache.export_portable_jsonl(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _file_receipt(temporary) != _file_receipt(destination):
                raise InputBundleError(
                    f"conflicting immutable portable score artifact: {destination}"
                ) from None
    finally:
        scorer.score_cache.close()
        temporary.unlink(missing_ok=True)
    size, digest = _file_receipt(destination)
    if digest != receipt["sha256"]:
        raise InputBundleError("portable score export digest changed")
    return {**receipt, "byte_count": size}


def import_portable_score_file(
    source: Path,
    score_cache_root: Path,
) -> dict[str, object]:
    """Strictly import one authenticated pinned-context portable cache file."""

    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise InputBundleError("portable score source must be a regular file")
    scorer = MixedbreadPassageScorer(
        score_cache_root=score_cache_root,
        device="cpu",
        batch_size=32,
        read_only=False,
    )
    try:
        context_sha256 = scorer.score_cache.context_sha256
        receipt = scorer.score_cache.import_portable_jsonl(source)
    finally:
        scorer.score_cache.close()
    return {
        "context_sha256": context_sha256,
        "inserted_count": receipt["inserted_count"],
        "source_row_count": receipt["source_row_count"],
        "source_sha256": receipt["source_sha256"],
    }


def create_input_archive(input_dir: Path, destination: Path) -> dict[str, object]:
    """Create immutable byte-deterministic gzip/tar bytes for one sealed input."""

    root = Path(input_dir)
    verified = verify_input_directory(root)
    destination = Path(destination)
    if destination.is_symlink() or (
        destination.exists() and not destination.is_file()
    ):
        raise InputBundleError("input archive destination must be a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_stream,
                mtime=0,
            ) as gzip_stream:
                with tarfile.open(
                    fileobj=gzip_stream,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    paths = sorted(
                        (path for path in root.rglob("*") if path.is_file()),
                        key=lambda path: path.relative_to(root).as_posix(),
                    )
                    for path in paths:
                        if path.is_symlink():
                            raise InputBundleError("input archive source contains a symlink")
                        relative = path.relative_to(root).as_posix()
                        info = tarfile.TarInfo(relative)
                        info.size = path.stat().st_size
                        info.mode = 0o600
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        with path.open("rb") as source:
                            archive.addfile(info, source)
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _file_receipt(temporary) != _file_receipt(destination):
                raise InputBundleError(
                    f"conflicting immutable input archive: {destination}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    archive_size, archive_sha256 = _file_receipt(destination)
    return {
        "archive_byte_count": archive_size,
        "archive_sha256": archive_sha256,
        "manifest_sha256": sha256((root / MANIFEST_NAME).read_bytes()).hexdigest(),
        "member_count": verified["member_count"],
        "topic_ids": verified["topic_ids"],
    }


def _archive_relative(name: str) -> str | None:
    pure = PurePosixPath(name)
    if pure.is_absolute():
        raise InputBundleError("input archive contains an absolute path")
    parts = list(pure.parts)
    while parts and parts[0] == ".":
        parts.pop(0)
    if not parts:
        return None
    if any(part in {"", ".", ".."} for part in parts):
        raise InputBundleError("input archive member path is invalid")
    relative = PurePosixPath(*parts).as_posix()
    if relative != MANIFEST_NAME and parts[0] not in _ALLOWED_ROOTS:
        raise InputBundleError("input archive member is outside the allowlist")
    return relative


def extract_input_archive(
    archive: Path,
    output_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_topics: tuple[str, ...],
) -> dict[str, object]:
    """Safely extract a regular-file-only archive and verify its sealed input."""

    archive = Path(archive)
    output_dir = Path(output_dir)
    if archive.is_symlink() or not archive.is_file():
        raise InputBundleError("input archive must be a regular file")
    if _SHA256.fullmatch(expected_manifest_sha256) is None:
        raise InputBundleError("expected input manifest digest is invalid")
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise InputBundleError("archive output must be a regular directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise InputBundleError("archive output directory must be empty")

    try:
        with tarfile.open(archive, mode="r:gz") as stream:
            members = stream.getmembers()
            if not members or len(members) > _MAX_ARCHIVE_MEMBERS:
                raise InputBundleError("input archive member count is invalid")
            planned: list[tuple[tarfile.TarInfo, str]] = []
            seen: set[str] = set()
            total_size = 0
            for member in members:
                relative = _archive_relative(member.name)
                if relative is None:
                    if not member.isdir():
                        raise InputBundleError("input archive root member is invalid")
                    continue
                if relative in seen:
                    raise InputBundleError("input archive repeats a member path")
                seen.add(relative)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise InputBundleError("input archive may contain only regular files")
                total_size += member.size
                if total_size > _MAX_ARCHIVE_BYTES:
                    raise InputBundleError("input archive expands beyond its size limit")
                planned.append((member, relative))
            if not planned:
                raise InputBundleError("input archive contains no regular files")
            for member, relative in planned:
                destination = output_dir.joinpath(*PurePosixPath(relative).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise InputBundleError("input archive member could not be read")
                with source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
    except (tarfile.TarError, OSError) as exc:
        raise InputBundleError("input archive is invalid or unreadable") from exc

    verified = verify_input_directory(
        output_dir,
        expected_topics=expected_topics,
    )
    manifest_sha256 = sha256((output_dir / MANIFEST_NAME).read_bytes()).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise InputBundleError("extracted input manifest digest differs from approved digest")
    archive_size, archive_sha256 = _file_receipt(archive)
    return {
        "archive_byte_count": archive_size,
        "archive_sha256": archive_sha256,
        "cache_stats": verified["cache_stats"],
        "canary_topic_ids": verified["canary_topic_ids"],
        "manifest_sha256": manifest_sha256,
        "member_count": verified["member_count"],
        "source_export_code_commit": verified["source_export_code_commit"],
        "source_export_manifest_sha256": verified[
            "source_export_manifest_sha256"
        ],
        "source_run_id": verified["source_run_id"],
        "topic_ids": verified["topic_ids"],
        "topic_stats": verified["topic_stats"],
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
    build.add_argument("--require-official-topic-set", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("input_dir", type=Path)
    verify.add_argument("--topic", action="append", dest="topic_ids", default=[])
    import_scores = commands.add_parser("import-scores")
    import_scores.add_argument("input_dir", type=Path)
    import_scores.add_argument("--score-cache", type=Path, required=True)
    export_cache = commands.add_parser("export-cache")
    export_cache.add_argument("--score-cache", type=Path, required=True)
    export_cache.add_argument("--output", type=Path, required=True)
    import_cache = commands.add_parser("import-cache")
    import_cache.add_argument("--portable", type=Path, required=True)
    import_cache.add_argument("--score-cache", type=Path, required=True)
    archive = commands.add_parser("create-archive")
    archive.add_argument("--input-dir", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser("extract-archive")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--input-manifest-sha256", required=True)
    extract.add_argument("--topic", action="append", dest="topic_ids", default=[])
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
            require_official_topic_set=args.require_official_topic_set,
        )
    elif args.command == "verify":
        verified = verify_input_directory(
            args.input_dir,
            expected_topics=tuple(args.topic_ids),
        )
        receipt = {
            "cache_stats": verified["cache_stats"],
            "canary_topic_ids": verified["canary_topic_ids"],
            "manifest_sha256": sha256(
                (args.input_dir / MANIFEST_NAME).read_bytes()
            ).hexdigest(),
            "member_count": verified["member_count"],
            "source_export_code_commit": verified["source_export_code_commit"],
            "source_export_manifest_sha256": verified[
                "source_export_manifest_sha256"
            ],
            "source_run_id": verified["source_run_id"],
            "topic_ids": verified["topic_ids"],
            "topic_stats": verified["topic_stats"],
        }
    elif args.command == "import-scores":
        receipt = import_portable_scores(args.input_dir, args.score_cache)
    elif args.command == "export-cache":
        receipt = export_portable_score_cache(args.score_cache, args.output)
    elif args.command == "import-cache":
        receipt = import_portable_score_file(args.portable, args.score_cache)
    elif args.command == "create-archive":
        receipt = create_input_archive(args.input_dir, args.output)
    else:
        receipt = extract_input_archive(
            args.archive,
            args.output_dir,
            expected_manifest_sha256=args.input_manifest_sha256,
            expected_topics=tuple(args.topic_ids),
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
