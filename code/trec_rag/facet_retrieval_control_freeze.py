"""Create-only offline freezer for facet-control ranking alternatives."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .det_sparse_ledger import RetrievalLedger, RetrievalRequest
from .facet_retrieval_control_experiment import (
    EXPECTED_ALTERNATIVE_NAMES,
    FACET_FAMILY_WEIGHT,
    ORIGINAL_FAMILY_WEIGHT,
    RANKING_DEPTH,
    RRF_K,
    build_topic_alternatives,
    index_control_streams,
)
from .facet_retrieval_control_inspector import inspect_stream, load_inspection_streams
from .facet_retrieval_control_manifest import (
    ANALYZER_FINGERPRINT_SHA256,
    R1_MANIFEST_SHA256,
    PROTECTED_TOPIC_IDS,
    ControlManifest,
    _validate_manifest,
    load_control_manifest,
)
from .facet_retrieval_control_run import build_control_requests
from .pipeline_models import RankedCandidate, RetrievedCandidate, jsonable


FREEZE_SCHEMA_VERSION = "facet-control-ranking-freeze-v2"
CANDIDATE_SNAPSHOT_SCHEMA_VERSION = "facet-control-candidate-snapshot-v1"
_CANDIDATE_MANIFEST_SCHEMA_VERSION = "facet-control-candidate-streams-v1"
_CANDIDATE_ROW_SCHEMA_VERSION = "facet-control-candidate-row-v1"
PRIOR_FREEZE_FILE_SHA256 = (
    "4a78b44ede4b979a3cb3ec96348088e4e08626e2cc4c92b464c6e36097a71389"
)
_PRIOR_FREEZE_INTERNAL_SHA256 = (
    "ba820c7cce7b3a1d85b7ff184d3be2c7a3b88af31cdb86e7b406a34e1e6c53f2"
)
_PRIOR_FUSION_SHA256 = (
    "8b1092d37376ce986950e00c3bedf51e31f50aa9c9bc9f7f82705e428eb3d3c9"
)
_PRIOR_MANIFEST_SHA256 = {
    "R0": "6dd55352786cf696391181a718f5d9ef7a9a3768cb229d18e3a8a12c6a8ea1ce",
    "R1": R1_MANIFEST_SHA256,
}
_PRIOR_TOPIC_IDS = ("200", "225", "707", "897")
_PRIOR_RANKING_NAMES = frozenset(
    f"{arm}:{fusion}"
    for arm in ("ALL", "F0", "O", "R0", "R1")
    for fusion in ("family_rrf", "interleave", "uniform_rrf")
)
_PRIOR_ENDPOINT = "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
_PRIOR_INDEX_ID = "climbmix-400b"
_PRIOR_RETRIEVER_VERSION = "pyserini_remote_raw_first_v1"
_R1_SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "experiments"
    / "sparse_relevance_pilot_v1"
    / "r1_manifest.json"
)
_BASE_ARM_QUERIES = (
    (
        "200",
        "prompt_lab_v1:original",
        "I want to deeply understand the Holocaust: what it was, why and how it transpired, who was responsible, and its profound historical and societal impact, particularly on European Jewry. I'm also curious about its conclusion, lasting effects, and how it aligns with other destructive historical events like Sodom and Gomorrah.",
    ),
    (
        "225",
        "prompt_lab_v1:original",
        "I'm exploring how exposure to violent video games and gory content affects human aggression, desensitization, and addiction. I'm also interested in the broader causes of aggression in both children and adults, the positive sides of gaming, and the historical background of these media trends.",
    ),
    ("225", "prompt_lab_v1:facet:f06", "video games benefits"),
    ("225", "prompt_lab_v1:facet:f07", "violent video games gory content history"),
    (
        "707",
        "prompt_lab_v1:original",
        "I'm trying to understand the health risks and potential dangers of various chemicals and substances, such as those found in antiperspirants, sorbitol, and organophosphate poisoning. I'd also like to know how integrating different actions at the operational level might impact health outcomes.",
    ),
    ("707", "prompt_lab_v1:facet:f01", "antiperspirants health risks"),
    (
        "707",
        "prompt_lab_v1:facet:f03",
        "organophosphate poisoning health risks",
    ),
    (
        "897",
        "prompt_lab_v1:original",
        "I'm trying to understand how alcohol use affects neighborhood quality of life, including genetic and gender factors in dependency. I also need to grasp the main causes, risks, and fatal consequences of alcohol and drug use, and their connection to broader community health issues.",
    ),
    ("897", "prompt_lab_v1:facet:f01", "alcohol use neighborhood quality life"),
)
_ALL_BASE_QUERIES = (
    *_BASE_ARM_QUERIES,
    ("200", "prompt_lab_v1:facet:f01", "Holocaust definition"),
    ("200", "prompt_lab_v1:facet:f02", "Holocaust causes"),
    ("200", "prompt_lab_v1:facet:f03", "Holocaust implementation process"),
    ("200", "prompt_lab_v1:facet:f04", "Holocaust responsibility"),
    (
        "200",
        "prompt_lab_v1:facet:f05",
        "Holocaust historical societal impact European Jewry",
    ),
    ("200", "prompt_lab_v1:facet:f06", "Holocaust conclusion end"),
    ("200", "prompt_lab_v1:facet:f07", "Holocaust lasting effects"),
    ("225", "prompt_lab_v1:facet:f01", "violent video games gory content aggression"),
    (
        "225",
        "prompt_lab_v1:facet:f02",
        "violent video games gory content desensitization",
    ),
    ("225", "prompt_lab_v1:facet:f03", "violent video games gory content addiction"),
    ("225", "prompt_lab_v1:facet:f04", "human aggression causes children"),
    ("225", "prompt_lab_v1:facet:f05", "human aggression causes adults"),
    ("707", "prompt_lab_v1:facet:f02", "sorbitol health risks"),
    (
        "897",
        "prompt_lab_v1:facet:f02",
        "alcohol dependence genetic gender factors",
    ),
    ("897", "prompt_lab_v1:facet:f03", "alcohol drug use causes"),
    ("897", "prompt_lab_v1:facet:f04", "alcohol drug use health risks"),
    ("897", "prompt_lab_v1:facet:f05", "alcohol drug use fatal consequences"),
    (
        "897",
        "prompt_lab_v1:facet:f06",
        "alcohol drug use community health effects",
    ),
)
_EXPECTED_R1_ARM_STREAMS_BY_TOPIC = {"200": 10, "225": 8, "707": 4, "897": 9}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_CANDIDATE_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "topic_id",
        "stream_id",
        "arm_id",
        "query_sha256",
        "rank",
        "docid",
    }
)
_CANDIDATE_STREAM_FIELDS = frozenset(
    {
        "topic_id",
        "stream_id",
        "arm_id",
        "role",
        "source_kind",
        "query_sha256",
        "request_sha256",
        "response_sha256",
        "source_candidates_sha256",
        "expected_depth",
        "row_count",
        "stream_rows_sha256",
    }
)
_CANDIDATE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_schema_version",
        "candidate_file",
        "candidate_file_sha256",
        "stream_count",
        "row_count",
        "streams",
    }
)


@dataclass(frozen=True)
class CandidateSnapshot:
    """Exact pre-serialized candidate leaf snapshot held entirely in memory."""

    schema_version: str
    manifest_bytes: bytes
    candidate_bytes: bytes


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_write(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _publish_directory_noreplace(stage: Path, destination: Path) -> None:
    """Atomically publish a directory with Linux ``RENAME_NOREPLACE``."""

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise OSError(
            errno.ENOSYS,
            "atomic no-replace directory publication is unavailable",
            destination,
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(stage),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _ranking_bytes(rows: Sequence[RankedCandidate]) -> bytes:
    return b"".join(
        json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _ranking_sha256(rows: Sequence[RankedCandidate]) -> str:
    payload = [jsonable(row) for row in rows]
    canonical = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _candidate_row_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )


def _snapshot_stream_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (str(row["topic_id"]), str(row["stream_id"]), str(row["arm_id"]))


def _expected_snapshot_streams() -> dict[tuple[str, str, str], tuple[str, str]]:
    expected: dict[tuple[str, str, str], tuple[str, str]] = {
        (topic_id, "original", "O"): ("original", "prior_base")
        for topic_id in _PRIOR_TOPIC_IDS
    }
    for topic_id, stream_id in (
        ("200", "f07a"),
        ("225", "f02"),
        ("225", "f04"),
        ("707", "f02"),
    ):
        for arm_id in ("B0", "W0", "W1", "W2"):
            expected[(topic_id, stream_id, arm_id)] = (
                "facet",
                "prior_r1" if arm_id == "B0" else "control",
            )
    return expected


def _validate_snapshot_source_rows(
    label: str,
    rows: Sequence[RetrievedCandidate],
    *,
    topic_id: str,
) -> tuple[RetrievedCandidate, ...]:
    ordered = tuple(sorted(rows, key=lambda row: (row.rank, row.docid, -row.score)))
    if (
        len(ordered) != 100
        or [row.rank for row in ordered] != list(range(1, 101))
        or len({row.docid for row in ordered}) != 100
        or any(
            row.topic_id != topic_id
            or not isinstance(row.docid, str)
            or not row.docid
            or isinstance(row.rank, bool)
            or not isinstance(row.rank, int)
            or not isinstance(row.query_text, str)
            or not row.query_text
            for row in ordered
        )
        or len({(row.variant_name, row.retriever_name, row.query_text) for row in ordered})
        != 1
    ):
        raise ValueError(f"candidate snapshot source {label} is not exact depth 100")
    return ordered


def build_candidate_snapshot(
    r1_arm: Sequence[RetrievedCandidate],
    control_rows: Sequence[RetrievedCandidate],
    manifest: ControlManifest,
    source_lineage: Mapping[tuple[str, str], Mapping[str, str]],
) -> CandidateSnapshot:
    """Copy the exact 20 leaf streams from already verified in-memory rows."""

    protected = sorted(
        {
            *(row.topic_id for row in r1_arm),
            *(row.topic_id for row in control_rows),
            *(stream.topic_id for stream in manifest.streams),
        }
        & set(PROTECTED_TOPIC_IDS)
    )
    if protected:
        raise ValueError(f"protected topic {protected[0]} is forbidden")
    _validate_manifest(manifest)
    try:
        indexed = index_control_streams(r1_arm, control_rows, manifest)
    except ValueError as exc:
        raise ValueError(f"candidate snapshot source is invalid: {exc}") from exc

    sources: list[
        tuple[str, str, str, str, str, tuple[RetrievedCandidate, ...]]
    ] = []
    for topic_id in _PRIOR_TOPIC_IDS:
        rows = _validate_snapshot_source_rows(
            f"{topic_id}/original/O",
            [
                row
                for row in r1_arm
                if row.topic_id == topic_id
                and row.variant_name == "prompt_lab_v1:original"
            ],
            topic_id=topic_id,
        )
        sources.append((topic_id, "original", "O", "original", "prior_base", rows))
    for stream in manifest.streams:
        for arm_id in ("B0", "W0", "W1", "W2"):
            rows = _validate_snapshot_source_rows(
                f"{stream.topic_id}/{stream.stream_id}/{arm_id}",
                indexed[(stream.topic_id, stream.stream_id, arm_id)],
                topic_id=stream.topic_id,
            )
            sources.append(
                (
                    stream.topic_id,
                    stream.stream_id,
                    arm_id,
                    "facet",
                    "prior_r1" if arm_id == "B0" else "control",
                    rows,
                )
            )

    expected = _expected_snapshot_streams()
    observed = {(topic, stream, arm) for topic, stream, arm, *_rest in sources}
    if observed != set(expected) or len(sources) != 20:
        raise ValueError("candidate snapshot differs from the exact 20-stream set")

    all_candidate_rows: list[dict[str, object]] = []
    manifest_entries: list[dict[str, object]] = []
    for topic_id, stream_id, arm_id, role, source_kind, rows in sorted(
        sources, key=lambda item: item[:3]
    ):
        variant_name = rows[0].variant_name
        lineage = source_lineage.get((topic_id, variant_name))
        if not isinstance(lineage, Mapping) or set(lineage) != {
            "request_sha256",
            "response_sha256",
            "candidate_sha256",
        }:
            raise ValueError(
                f"candidate snapshot source lineage is missing for {topic_id}/{variant_name}"
            )
        for field, value in lineage.items():
            _validate_sha256(value, f"candidate snapshot {topic_id}/{stream_id}/{arm_id}.{field}")
        query_sha256 = hashlib.sha256(rows[0].query_text.encode("utf-8")).hexdigest()
        candidate_rows = [
            {
                "schema_version": _CANDIDATE_ROW_SCHEMA_VERSION,
                "topic_id": topic_id,
                "stream_id": stream_id,
                "arm_id": arm_id,
                "query_sha256": query_sha256,
                "rank": row.rank,
                "docid": row.docid,
            }
            for row in rows
        ]
        stream_bytes = _candidate_row_bytes(candidate_rows)
        all_candidate_rows.extend(candidate_rows)
        manifest_entries.append(
            {
                "topic_id": topic_id,
                "stream_id": stream_id,
                "arm_id": arm_id,
                "role": role,
                "source_kind": source_kind,
                "query_sha256": query_sha256,
                "request_sha256": lineage["request_sha256"],
                "response_sha256": lineage["response_sha256"],
                "source_candidates_sha256": lineage["candidate_sha256"],
                "expected_depth": 100,
                "row_count": 100,
                "stream_rows_sha256": hashlib.sha256(stream_bytes).hexdigest(),
            }
        )
    all_candidate_rows.sort(
        key=lambda row: (
            row["topic_id"], row["stream_id"], row["arm_id"], row["rank"]
        )
    )
    candidate_bytes = _candidate_row_bytes(all_candidate_rows)
    candidate_manifest = {
        "schema_version": _CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "candidate_schema_version": _CANDIDATE_ROW_SCHEMA_VERSION,
        "candidate_file": "candidates.jsonl",
        "candidate_file_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "stream_count": 20,
        "row_count": 2000,
        "streams": manifest_entries,
    }
    snapshot = CandidateSnapshot(
        schema_version=CANDIDATE_SNAPSHOT_SCHEMA_VERSION,
        manifest_bytes=_canonical_json(candidate_manifest),
        candidate_bytes=candidate_bytes,
    )
    validate_candidate_snapshot(snapshot)
    return snapshot


def validate_candidate_snapshot(
    snapshot: CandidateSnapshot,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Validate exact candidate bytes, manifest, hashes, and stream invariants."""

    if (
        not isinstance(snapshot, CandidateSnapshot)
        or snapshot.schema_version != CANDIDATE_SNAPSHOT_SCHEMA_VERSION
    ):
        raise ValueError("candidate snapshot schema differs")
    try:
        manifest = json.loads(snapshot.manifest_bytes)
        rows = [json.loads(line) for line in snapshot.candidate_bytes.splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("candidate snapshot is not valid canonical JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != _CANDIDATE_MANIFEST_FIELDS:
        raise ValueError("candidate snapshot manifest fields differ")
    if _canonical_json(manifest) != snapshot.manifest_bytes:
        raise ValueError("candidate snapshot manifest is not canonical")
    if (
        manifest.get("schema_version") != _CANDIDATE_MANIFEST_SCHEMA_VERSION
        or manifest.get("candidate_schema_version") != _CANDIDATE_ROW_SCHEMA_VERSION
        or manifest.get("candidate_file") != "candidates.jsonl"
        or manifest.get("stream_count") != 20
        or manifest.get("row_count") != 2000
        or len(rows) != 2000
        or hashlib.sha256(snapshot.candidate_bytes).hexdigest()
        != manifest.get("candidate_file_sha256")
    ):
        raise ValueError("candidate snapshot counts or file SHA-256 differ")
    if _candidate_row_bytes(rows) != snapshot.candidate_bytes:
        raise ValueError("candidate snapshot JSONL is not canonical")
    if any(
        not isinstance(row, dict)
        or set(row) != _CANDIDATE_ROW_FIELDS
        or row.get("schema_version") != _CANDIDATE_ROW_SCHEMA_VERSION
        or row.get("topic_id") in PROTECTED_TOPIC_IDS
        or not isinstance(row.get("topic_id"), str)
        or not isinstance(row.get("stream_id"), str)
        or not isinstance(row.get("arm_id"), str)
        or not isinstance(row.get("docid"), str)
        or not row.get("docid")
        or isinstance(row.get("rank"), bool)
        or not isinstance(row.get("rank"), int)
        for row in rows
    ):
        raise ValueError("candidate snapshot contains an invalid candidate row")
    canonical_order = sorted(
        rows,
        key=lambda row: (
            row["topic_id"], row["stream_id"], row["arm_id"], row["rank"]
        ),
    )
    if rows != canonical_order:
        raise ValueError("candidate snapshot rows are not in canonical order")

    entries = manifest.get("streams")
    if not isinstance(entries, list) or len(entries) != 20 or any(
        not isinstance(entry, dict) or set(entry) != _CANDIDATE_STREAM_FIELDS
        for entry in entries
    ):
        raise ValueError("candidate snapshot stream manifest differs")
    if entries != sorted(
        entries, key=lambda entry: (entry["topic_id"], entry["stream_id"], entry["arm_id"])
    ):
        raise ValueError("candidate snapshot stream manifest is not canonical")
    expected = _expected_snapshot_streams()
    if {_snapshot_stream_key(entry) for entry in entries} != set(expected):
        raise ValueError("candidate snapshot differs from the exact 20-stream set")
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_snapshot_stream_key(row)].append(row)
    if set(grouped) != set(expected):
        raise ValueError("candidate snapshot rows omit or add a stream")
    for entry in entries:
        key = _snapshot_stream_key(entry)
        role, source_kind = expected[key]
        stream_rows = grouped[key]
        if (
            entry.get("role") != role
            or entry.get("source_kind") != source_kind
            or entry.get("expected_depth") != 100
            or entry.get("row_count") != 100
            or len(stream_rows) != 100
            or [row["rank"] for row in stream_rows] != list(range(1, 101))
            or len({row["docid"] for row in stream_rows}) != 100
            or {row["query_sha256"] for row in stream_rows}
            != {entry.get("query_sha256")}
            or hashlib.sha256(_candidate_row_bytes(stream_rows)).hexdigest()
            != entry.get("stream_rows_sha256")
        ):
            raise ValueError(f"candidate snapshot stream {key!r} is corrupt")
        for field in (
            "query_sha256",
            "request_sha256",
            "response_sha256",
            "source_candidates_sha256",
            "stream_rows_sha256",
        ):
            _validate_sha256(entry.get(field), f"candidate snapshot {key!r}.{field}")
    return manifest, rows


def _validate_sha256(value: object, label: str) -> None:
    if isinstance(value, str):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{label} must be a lowercase SHA-256")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty text")
            _validate_sha256(nested, f"{label}.{key}")
        return
    raise ValueError(f"{label} must contain SHA-256 values")


def _validate_bindings(bindings: Mapping[str, object]) -> dict[str, object]:
    required = {
        "manifest_sha256",
        "prior_freeze_sha256",
        "request_sha256",
        "response_sha256",
        "candidate_sha256",
    }
    allowed = required | {
        "ledger_sha256",
        "r1_arm_sha256",
        "candidate_streams_sha256",
        "candidates_sha256",
        "candidate_stream_rows_sha256",
    }
    if not required <= set(bindings) or not set(bindings) <= allowed:
        raise ValueError(
            f"freeze bindings must include exactly the required hashes {sorted(required)!r}"
        )
    result = dict(bindings)
    for key, value in result.items():
        _validate_sha256(value, key)
    return result


def create_control_freeze(
    output_dir: Path,
    rankings: Mapping[str, Sequence[RankedCandidate]],
    *,
    inspections: Mapping[str, object],
    bindings: Mapping[str, object],
    candidate_snapshot: CandidateSnapshot,
) -> dict[str, object]:
    """Precompute every byte, stage privately, then atomically publish once."""

    if frozenset(rankings) != frozenset(EXPECTED_ALTERNATIVE_NAMES):
        raise ValueError("control freeze rankings differ from the exact 25-name set")
    for key, rows in rankings.items():
        topic_id = key.split(":", 2)[1]
        if len(rows) != RANKING_DEPTH or any(row.topic_id != topic_id for row in rows):
            raise ValueError(f"ranking {key} must contain exactly 100 target-topic rows")
        if [row.rank for row in rows] != list(range(1, RANKING_DEPTH + 1)):
            raise ValueError(f"ranking {key} has non-canonical final ranks")

    # Serialization and hashing deliberately precede any destination creation.
    snapshot_manifest, _snapshot_rows = validate_candidate_snapshot(candidate_snapshot)
    snapshot_binding_fields = {
        "candidate_streams_sha256",
        "candidates_sha256",
        "candidate_stream_rows_sha256",
    }
    if set(bindings) & snapshot_binding_fields:
        raise ValueError("candidate snapshot bindings are freezer-owned")
    source_bindings = _validate_bindings(bindings)
    request_hashes = source_bindings["request_sha256"]
    response_hashes = source_bindings["response_sha256"]
    candidate_hashes = source_bindings["candidate_sha256"]
    if not all(
        isinstance(mapping, Mapping)
        for mapping in (request_hashes, response_hashes, candidate_hashes)
    ):
        raise ValueError("candidate snapshot source bindings must be mappings")
    for entry in snapshot_manifest["streams"]:
        request_sha256 = entry["request_sha256"]
        if (
            request_hashes.get(request_sha256) != request_sha256
            or response_hashes.get(request_sha256) != entry["response_sha256"]
            or candidate_hashes.get(request_sha256)
            != entry["source_candidates_sha256"]
        ):
            raise ValueError("candidate snapshot source hashes differ from freeze bindings")
    stream_hashes = {
        f"{entry['topic_id']}/{entry['stream_id']}/{entry['arm_id']}": entry[
            "stream_rows_sha256"
        ]
        for entry in snapshot_manifest["streams"]
    }
    frozen_bindings = {
        **source_bindings,
        "candidate_streams_sha256": hashlib.sha256(
            candidate_snapshot.manifest_bytes
        ).hexdigest(),
        "candidates_sha256": hashlib.sha256(
            candidate_snapshot.candidate_bytes
        ).hexdigest(),
        "candidate_stream_rows_sha256": stream_hashes,
    }
    _validate_bindings(frozen_bindings)
    artifacts: dict[str, bytes] = {}
    artifacts["candidate_streams.json"] = candidate_snapshot.manifest_bytes
    artifacts["candidates.jsonl"] = candidate_snapshot.candidate_bytes
    ranking_manifest: dict[str, dict[str, object]] = {}
    for key in sorted(rankings):
        filename = key.replace(":", "__") + ".jsonl"
        content = _ranking_bytes(rankings[key])
        artifacts[f"rankings/{filename}"] = content
        ranking_manifest[key] = {
            "path": f"rankings/{filename}",
            "rows": len(rankings[key]),
            "sha256": _ranking_sha256(rankings[key]),
            "file_sha256": hashlib.sha256(content).hexdigest(),
        }

    inspection_bytes = _canonical_json(inspections)
    artifacts["inspection.json"] = inspection_bytes
    fusion = {
        "k": RRF_K,
        "limit": RANKING_DEPTH,
        "original_family_weight": ORIGINAL_FAMILY_WEIGHT,
        "facet_family_weight": FACET_FAMILY_WEIGHT,
        "facet_stream_weight": "0.5 / active topic facet streams",
    }
    fusion_bytes = _canonical_json(fusion)
    artifacts["fusion.json"] = fusion_bytes

    payload: dict[str, object] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen_before_qrels",
        "bindings": frozen_bindings,
        "inspection_sha256": hashlib.sha256(inspection_bytes).hexdigest(),
        "fusion_sha256": hashlib.sha256(fusion_bytes).hexdigest(),
        "rankings": ranking_manifest,
    }
    payload["freeze_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    artifacts["freeze.json"] = _canonical_json(payload)

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        (stage / "rankings").mkdir()
        for relative, content in artifacts.items():
            _exclusive_write(stage / relative, content)
        _publish_directory_noreplace(stage, output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return payload


def _validate_prior_ranked_rows(key: str, decoded: object) -> list[dict[str, object]]:
    if not isinstance(decoded, list) or len(decoded) != 400:
        raise ValueError(f"prior ranking {key} must contain exactly 400 rows")
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(decoded):
        if not isinstance(raw, dict) or set(raw) != {
            "topic_id",
            "docid",
            "rank",
            "score",
            "text",
            "provenance",
        }:
            raise ValueError(f"prior ranking {key} row {index} is not RankedCandidate")
        try:
            candidate = RankedCandidate(**raw)
        except TypeError as exc:
            raise ValueError(
                f"prior ranking {key} row {index} is not RankedCandidate"
            ) from exc
        if (
            not isinstance(candidate.topic_id, str)
            or candidate.topic_id not in _PRIOR_TOPIC_IDS
            or not isinstance(candidate.docid, str)
            or not candidate.docid
            or isinstance(candidate.rank, bool)
            or not isinstance(candidate.rank, int)
            or isinstance(candidate.score, bool)
            or not isinstance(candidate.score, (int, float))
            or not math.isfinite(float(candidate.score))
            or not isinstance(candidate.text, str)
            or not isinstance(candidate.provenance, list)
            or not all(isinstance(item, dict) for item in candidate.provenance)
        ):
            raise ValueError(f"prior ranking {key} contains invalid RankedCandidate")
        rows.append(raw)
    for topic_id in _PRIOR_TOPIC_IDS:
        topic_rows = [row for row in rows if row["topic_id"] == topic_id]
        if len(topic_rows) != 100:
            raise ValueError(f"prior ranking {key}/{topic_id} depth is not 100")
        if [row["rank"] for row in topic_rows] != list(range(1, 101)):
            raise ValueError(f"prior ranking {key}/{topic_id} ranks are invalid")
        if len({row["docid"] for row in topic_rows}) != 100:
            raise ValueError(f"prior ranking {key}/{topic_id} docids are not unique")
    return rows


def verify_prior_freeze(path: Path) -> str:
    """Verify the one exact immutable prior freeze and all 15 rankings."""

    freeze_path = Path(path)
    if freeze_path.is_dir():
        freeze_path = freeze_path / "freeze.json"
    source = freeze_path.read_bytes()
    actual_file_sha256 = hashlib.sha256(source).hexdigest()
    if actual_file_sha256 != PRIOR_FREEZE_FILE_SHA256:
        raise ValueError("input differs from the exact immutable prior freeze")
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError("prior freeze is not valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "sparse-relevance-ranking-freeze-v1"
        or payload.get("status") != "frozen_before_qrels"
        or payload.get("freeze_sha256") != _PRIOR_FREEZE_INTERNAL_SHA256
        or payload.get("topic_ids") != list(_PRIOR_TOPIC_IDS)
        or payload.get("manifest_sha256") != _PRIOR_MANIFEST_SHA256
        or payload.get("fusion_definitions_sha256") != _PRIOR_FUSION_SHA256
    ):
        raise ValueError("exact immutable prior freeze contract differs")
    fusion = payload.get("fusion_definitions")
    fusion_sha256 = hashlib.sha256(
        json.dumps(fusion, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    ).hexdigest()
    if fusion_sha256 != _PRIOR_FUSION_SHA256:
        raise ValueError("exact immutable prior fusion definitions differ")
    expected_self = payload.get("freeze_sha256")
    _validate_sha256(expected_self, "prior freeze_sha256")
    without_self = dict(payload)
    without_self.pop("freeze_sha256")
    actual_self = hashlib.sha256(
        json.dumps(
            without_self,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()
    if actual_self != expected_self:
        raise ValueError("prior freeze self SHA-256 is invalid")
    rankings = payload.get("rankings")
    if not isinstance(rankings, dict) or frozenset(rankings) != _PRIOR_RANKING_NAMES:
        raise ValueError("exact immutable prior freeze must name exactly 15 rankings")
    for key, raw_record in rankings.items():
        if not isinstance(key, str) or not isinstance(raw_record, dict):
            raise ValueError("prior freeze ranking records are invalid")
        expected = raw_record.get("sha256")
        _validate_sha256(expected, f"prior ranking {key}")
        relative = raw_record.get("path")
        ranking_path = (
            freeze_path.parent / relative
            if isinstance(relative, str)
            else freeze_path.parent / "rankings" / f"{key.replace(':', '__')}.jsonl"
        )
        try:
            decoded_rows = [
                json.loads(line)
                for line in ranking_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"prior ranking is unreadable for {key}") from exc
        validated_rows = _validate_prior_ranked_rows(key, decoded_rows)
        actual = hashlib.sha256(
            (
                json.dumps(
                    validated_rows,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"prior ranking SHA-256 mismatch for {key}")
        rows = raw_record.get("rows")
        if rows != 400:
            raise ValueError(f"prior ranking row count is invalid for {key}")
        if len(validated_rows) != rows:
            raise ValueError(f"prior ranking row count mismatch for {key}")
    return actual_file_sha256


def _ledger_from_existing(run_dir: Path, shared_cache: Path | None) -> RetrievalLedger:
    policy_path = Path(run_dir) / "ledger.json"
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load ledger policy: {policy_path}") from exc
    if not isinstance(policy, dict):
        raise ValueError(f"ledger policy must be an object: {policy_path}")
    ledger = RetrievalLedger(
        run_dir,
        shared_cache_dir=shared_cache,
        max_calls=policy.get("max_calls"),
        max_calls_per_topic=policy.get("max_calls_per_topic"),
        min_results=policy.get("min_results"),
        required_text_results=policy.get("required_text_results"),
    )
    report = ledger.validate_run()
    if report.failures or report.pending:
        raise ValueError(f"ledger is not complete and successful: {run_dir}")
    return ledger


def _ledger_tree_sha256(run_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item for item in Path(run_dir).rglob("*") if item.is_file() and item.name != ".ledger.lock"
    ):
        digest.update(path.relative_to(run_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _prior_request(
    topic_id: str, variant_name: str, query_text: str
) -> RetrievalRequest:
    return RetrievalRequest.from_query(
        topic_id=topic_id,
        variant_name=variant_name,
        query_text=query_text,
        index_url=_PRIOR_ENDPOINT,
        index_id=_PRIOR_INDEX_ID,
        hits=100,
        analyzer_fingerprint_sha256=ANALYZER_FINGERPRINT_SHA256,
        retriever_version=_PRIOR_RETRIEVER_VERSION,
    )


def build_expected_r1_requests(
    manifest: ControlManifest,
    r1_source_path: Path = _R1_SOURCE_PATH,
) -> tuple[tuple[RetrievalRequest, ...], tuple[RetrievalRequest, ...]]:
    """Build the exact nine base and 22 repair requests that form frozen R1."""

    protected = sorted(
        {stream.topic_id for stream in manifest.streams} & set(PROTECTED_TOPIC_IDS)
    )
    if protected:
        raise ValueError(f"protected topic {protected[0]} is forbidden")
    _validate_manifest(manifest)
    source = Path(r1_source_path).read_bytes()
    if hashlib.sha256(source).hexdigest() != R1_MANIFEST_SHA256:
        raise ValueError("R1 source manifest SHA-256 differs from the frozen source")
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError("R1 source manifest is not valid JSON") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if (
        not isinstance(streams, list)
        or len(streams) != 22
        or not all(isinstance(stream, dict) for stream in streams)
    ):
        raise ValueError("R1 source manifest must contain exactly 22 repair streams")
    base_requests = tuple(_prior_request(*spec) for spec in _BASE_ARM_QUERIES)
    r1_requests: list[RetrievalRequest] = []
    seen: set[tuple[str, str]] = set()
    for stream in streams:
        topic_id = stream.get("topic_id")
        stream_id = stream.get("stream_id")
        query = stream.get("query")
        if not all(isinstance(value, str) and value for value in (topic_id, stream_id, query)):
            raise ValueError("R1 source contains an invalid repair request")
        boundary = (topic_id, stream_id)
        if boundary in seen:
            raise ValueError(f"R1 source contains duplicate stream {boundary!r}")
        seen.add(boundary)
        r1_requests.append(
            _prior_request(topic_id, f"sparse_relevance_v1:R1:{stream_id}", query)
        )
    return base_requests, tuple(r1_requests)


def _request_stream_key(request: RetrievalRequest) -> tuple[str, str, str, str]:
    return (
        request.identity.topic_id,
        request.identity.variant_name,
        request.identity.retriever_version,
        request.query_text,
    )


def validate_verified_r1_arm(
    rows: Sequence[RetrievedCandidate],
    base_requests: Sequence[RetrievalRequest],
    r1_requests: Sequence[RetrievalRequest],
) -> list[RetrievedCandidate]:
    """Reject any missing, extra, substituted, shallow, or duplicate R1 stream."""

    protected = sorted({row.topic_id for row in rows} & set(PROTECTED_TOPIC_IDS))
    if protected:
        raise ValueError(f"protected topic {protected[0]} is forbidden")
    expected_requests = (*base_requests, *r1_requests)
    if len(base_requests) != 9 or len(r1_requests) != 22:
        raise ValueError("R1 arm request lineage must contain exactly 9 base and 22 repairs")
    expected = {_request_stream_key(request) for request in expected_requests}
    if len(expected) != 31:
        raise ValueError("R1 arm request lineage contains duplicate streams")
    grouped: dict[tuple[str, str, str, str], list[RetrievedCandidate]] = defaultdict(list)
    for row in rows:
        grouped[(row.topic_id, row.variant_name, row.retriever_name, row.query_text)].append(
            row
        )
    if set(grouped) != expected:
        raise ValueError("R1 arm streams differ from the exact frozen lineage")
    for key, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda row: (row.rank, row.docid, -row.score))
        if len(ordered) != 100 or [row.rank for row in ordered] != list(range(1, 101)):
            raise ValueError(f"R1 arm stream {key!r} does not have exact depth 100")
        if len({row.docid for row in ordered}) != 100:
            raise ValueError(f"R1 arm stream {key!r} has duplicate document IDs")
        if any(
            not row.docid
            or isinstance(row.score, bool)
            or not isinstance(row.score, (int, float))
            or not math.isfinite(float(row.score))
            or not isinstance(row.text, str)
            for row in ordered
        ):
            raise ValueError(f"R1 arm stream {key!r} has invalid candidates")
    observed_counts = {
        topic_id: sum(key[0] == topic_id for key in grouped)
        for topic_id in _PRIOR_TOPIC_IDS
    }
    if observed_counts != _EXPECTED_R1_ARM_STREAMS_BY_TOPIC:
        raise ValueError("R1 arm per-topic stream counts differ from 10/8/4/9")
    return sorted(
        rows,
        key=lambda row: (
            row.topic_id,
            row.variant_name,
            row.retriever_name,
            row.rank,
            row.docid,
        ),
    )


def _load_requests_from_ledger(
    ledger: RetrievalLedger,
    requests: Sequence[RetrievalRequest],
) -> tuple[list[RetrievedCandidate], dict[str, str], dict[str, str], dict[str, str]]:
    rows: list[RetrievedCandidate] = []
    request_hashes: dict[str, str] = {}
    response_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    for request in requests:
        result = ledger.load_verified_result(request)
        key = request.identity.request_key
        request_hashes[key] = key
        response_hashes[key] = result.response_sha256
        candidate_hashes[key] = result.candidates_sha256
        rows.extend(
            RetrievedCandidate(
                topic_id=request.identity.topic_id,
                variant_name=request.identity.variant_name,
                retriever_name=request.identity.retriever_version,
                query_text=request.query_text,
                docid=candidate.docid,
                rank=candidate.rank,
                score=candidate.score,
                text=candidate.text,
            )
            for candidate in result.candidates
        )
    return rows, request_hashes, response_hashes, candidate_hashes


def _source_lineage_for_requests(
    requests: Sequence[RetrievalRequest],
    response_hashes: Mapping[str, str],
    candidate_hashes: Mapping[str, str],
) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for request in requests:
        request_sha256 = request.identity.request_key
        result[(request.identity.topic_id, request.identity.variant_name)] = {
            "request_sha256": request_sha256,
            "response_sha256": response_hashes[request_sha256],
            "candidate_sha256": candidate_hashes[request_sha256],
        }
    return result


def load_verified_r1_arm(
    manifest: ControlManifest,
    *,
    base_run: Path,
    base_cache: Path,
    r1_run: Path,
    r1_cache: Path,
    r1_source_path: Path = _R1_SOURCE_PATH,
) -> tuple[list[RetrievedCandidate], dict[str, object]]:
    """Reconstruct R1 solely from its exact independently verified ledgers."""

    base_requests, r1_requests = build_expected_r1_requests(manifest, r1_source_path)
    all_base_requests = tuple(_prior_request(*spec) for spec in _ALL_BASE_QUERIES)
    if len({_request_stream_key(request) for request in all_base_requests}) != 27:
        raise AssertionError("frozen base request namespace must contain exactly 27 streams")
    base_ledger = _ledger_from_existing(base_run, base_cache)
    if base_ledger.validate_run().planned_requests != 27:
        raise ValueError("base ledger must contain exactly 27 frozen requests")
    r1_ledger = _ledger_from_existing(r1_run, r1_cache)
    if r1_ledger.validate_run().planned_requests != 22:
        raise ValueError("R1 ledger must contain exactly 22 frozen requests")
    all_base = _load_requests_from_ledger(base_ledger, all_base_requests)
    repairs = _load_requests_from_ledger(r1_ledger, r1_requests)
    selected_base_keys = {_request_stream_key(request) for request in base_requests}
    selected_base_rows = [
        row
        for row in all_base[0]
        if (row.topic_id, row.variant_name, row.retriever_name, row.query_text)
        in selected_base_keys
    ]
    rows = validate_verified_r1_arm(
        [*selected_base_rows, *repairs[0]], base_requests, r1_requests
    )
    encoded = _canonical_json(rows)
    return rows, {
        "r1_arm_sha256": hashlib.sha256(encoded).hexdigest(),
        "ledger_sha256": {
            "base": _ledger_tree_sha256(base_run),
            "r1": _ledger_tree_sha256(r1_run),
        },
        "request_sha256": {**all_base[1], **repairs[1]},
        "response_sha256": {**all_base[2], **repairs[2]},
        "candidate_sha256": {**all_base[3], **repairs[3]},
        "snapshot_lineage": {
            **_source_lineage_for_requests(all_base_requests, all_base[2], all_base[3]),
            **_source_lineage_for_requests(r1_requests, repairs[2], repairs[3]),
        },
    }


def _load_control_candidates(
    manifest: ControlManifest,
    ledger_dir: Path,
    shared_cache: Path,
) -> tuple[list[RetrievedCandidate], dict[str, str], dict[str, str], dict[str, str]]:
    ledger = _ledger_from_existing(ledger_dir, shared_cache)
    requests = build_control_requests(manifest, endpoint=_PRIOR_ENDPOINT)
    if ledger.validate_run().planned_requests != len(requests):
        raise ValueError("control ledger does not contain exactly the 12 frozen requests")
    return _load_requests_from_ledger(ledger, requests)


def freeze_control_experiment(
    *,
    output_dir: Path,
    r1_source_manifest: Path,
    r1_arm: Sequence[RetrievedCandidate],
    control_rows: Sequence[RetrievedCandidate],
    manifest: ControlManifest,
    bindings: Mapping[str, object],
    source_lineage: Mapping[tuple[str, str], Mapping[str, str]],
) -> dict[str, object]:
    """Inspect all 16 stream arms and create all 25 alternatives without qrels."""

    indexed = index_control_streams(r1_arm, control_rows, manifest)
    specs = load_inspection_streams(r1_source_manifest, manifest)
    inspections: dict[str, object] = {}
    for stream in manifest.streams:
        for arm in ("B0", "W0", "W1", "W2"):
            key = f"{stream.topic_id}/{stream.stream_id}/{arm}"
            inspections[key] = jsonable(
                inspect_stream(
                    specs[(stream.topic_id, stream.stream_id)],
                    indexed[(stream.topic_id, stream.stream_id, arm)],
                )
            )
    rankings = build_topic_alternatives(r1_arm, control_rows, manifest)
    candidate_snapshot = build_candidate_snapshot(
        r1_arm,
        control_rows,
        manifest,
        source_lineage,
    )
    return create_control_freeze(
        output_dir,
        rankings,
        inspections=inspections,
        bindings=bindings,
        candidate_snapshot=candidate_snapshot,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior-freeze", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--r1-cache", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--control-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_control_manifest(args.manifest)
    # The protected namespace gate precedes prior-freeze, ledger, cache, and fusion access.
    protected = sorted(
        {stream.topic_id for stream in manifest.streams} & set(PROTECTED_TOPIC_IDS)
    )
    if protected:
        raise ValueError(f"protected topic {protected[0]} is forbidden")
    if args.output.exists():
        raise FileExistsError(f"create-only output already exists: {args.output}")

    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    prior_freeze_sha256 = verify_prior_freeze(args.prior_freeze)
    r1_arm, lineage = load_verified_r1_arm(
        manifest,
        base_run=args.base_run,
        base_cache=args.base_cache,
        r1_run=args.r1_run,
        r1_cache=args.r1_cache,
    )
    control_rows, request_hashes, response_hashes, candidate_hashes = (
        _load_control_candidates(
            manifest,
            args.control_run,
            args.control_cache,
        )
    )
    bindings: dict[str, object] = {
        "manifest_sha256": manifest_sha256,
        "prior_freeze_sha256": prior_freeze_sha256,
        "ledger_sha256": {
            **lineage["ledger_sha256"],
            "control": _ledger_tree_sha256(args.control_run),
        },
        "r1_arm_sha256": lineage["r1_arm_sha256"],
        "request_sha256": {**lineage["request_sha256"], **request_hashes},
        "response_sha256": {**lineage["response_sha256"], **response_hashes},
        "candidate_sha256": {**lineage["candidate_sha256"], **candidate_hashes},
    }
    control_requests = build_control_requests(manifest, endpoint=_PRIOR_ENDPOINT)
    source_lineage = {
        **lineage.get("snapshot_lineage", {}),
        **_source_lineage_for_requests(
            control_requests,
            response_hashes,
            candidate_hashes,
        ),
    }
    freeze = freeze_control_experiment(
        output_dir=args.output,
        r1_source_manifest=_R1_SOURCE_PATH,
        r1_arm=r1_arm,
        control_rows=control_rows,
        manifest=manifest,
        bindings=bindings,
        source_lineage=source_lineage,
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
