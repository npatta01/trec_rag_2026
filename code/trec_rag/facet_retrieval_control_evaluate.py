"""Verify and evaluate the self-contained facet retrieval-control freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping, Sequence

from . import facet_retrieval_control_freeze as freeze_contract
from .evaluation import evaluate_ranked, parse_qrels_bytes
from .facet_retrieval_control_experiment import (
    EXPECTED_ALTERNATIVE_NAMES,
    FACET_FAMILY_WEIGHT,
    ORIGINAL_FAMILY_WEIGHT,
    RANKING_DEPTH,
    RRF_K,
    StreamArmEvaluation,
    evaluate_stream_arm,
    retrieval_repair_decision,
    select_stream_arm,
    selected_ranking_references,
)
from .facet_retrieval_control_freeze import (
    CANDIDATE_SNAPSHOT_SCHEMA_VERSION,
    FREEZE_SCHEMA_VERSION,
    FREEZE_SCHEMA_VERSION_V2,
    CandidateSnapshot,
    _canonical_json,
    _publish_directory_noreplace,
    _ranking_sha256,
    _validate_bindings,
    _validate_candidate_snapshot_source_bindings,
    _validate_prior_ranked_rows,
    validate_candidate_snapshot,
)
from .facet_retrieval_control_inspector import InspectionResult
from .facet_retrieval_control_manifest import PROTECTED_TOPIC_IDS
from .pipeline_models import RankedCandidate, RetrievedCandidate, jsonable


EVALUATION_TOPIC_IDS = ("200", "225", "707", "897")
STREAM_IDS = ("200/f07a", "225/f02", "225/f04", "707/f02")
ARM_IDS = ("B0", "W0", "W1", "W2")
OUTPUT_NAMES = (
    "stream_evaluation.json",
    "selection.json",
    "evaluation.json",
    "decision.json",
)
SYSTEM_METRICS = (
    "ndcg@10",
    "graded_recall@100",
    "recall@100",
    "precision@10",
    "relevant_count@10",
    "judged_rate@10",
    "judged_rate@100",
)
_ROOT_FIELDS = {
    "schema_version",
    "status",
    "bindings",
    "inspection_sha256",
    "fusion_sha256",
    "rankings",
    "freeze_sha256",
}
_V2_BINDING_FIELDS = {
    "manifest_sha256",
    "prior_freeze_sha256",
    "request_sha256",
    "response_sha256",
    "candidate_sha256",
    "ledger_sha256",
    "r1_arm_sha256",
    "candidate_streams_sha256",
    "candidates_sha256",
    "candidate_stream_rows_sha256",
}
_INSPECTION_FIELDS = {field.name for field in fields(InspectionResult)}


@dataclass(frozen=True)
class FrozenCandidateRow:
    topic_id: str
    stream_id: str
    arm_id: str
    query_sha256: str
    rank: int
    docid: str


@dataclass(frozen=True)
class VerifiedControlFreeze:
    """One in-memory snapshot of every verified control artifact."""

    payload: Mapping[str, object]
    file_sha256: str
    inspections: Mapping[str, InspectionResult]
    rankings: Mapping[str, tuple[RankedCandidate, ...]]
    ranking_sha256: Mapping[str, str]
    candidate_streams: Mapping[
        tuple[str, str, str], tuple[FrozenCandidateRow, ...]
    ]


@dataclass(frozen=True)
class VerifiedPriorFreeze:
    """The exact verified prior bytes decoded once for subsequent evaluation."""

    file_sha256: str
    rankings: Mapping[str, tuple[RankedCandidate, ...]]


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _artifact_path(directory: Path, relative: object, expected: str) -> Path:
    if relative != expected:
        raise ValueError(f"control freeze artifact path differs for {expected}")
    candidate = directory / expected
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError("freeze artifact path escapes its directory") from exc
    return candidate


def _decode_ranking(
    key: str,
    content: bytes,
    record: Mapping[str, object],
) -> tuple[RankedCandidate, ...]:
    if hashlib.sha256(content).hexdigest() != _require_sha256(
        record.get("file_sha256"), f"control ranking file {key}"
    ):
        raise ValueError(f"control ranking file SHA-256 mismatch for {key}")
    try:
        decoded = [json.loads(line) for line in content.splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"control ranking is invalid JSONL for {key}") from exc
    if len(decoded) != 100 or record.get("rows") != 100:
        raise ValueError(f"control ranking row count differs for {key}")
    expected_fields = {"topic_id", "docid", "rank", "score", "text", "provenance"}
    topic_id = key.split(":", 2)[1]
    rows: list[RankedCandidate] = []
    for rank, raw in enumerate(decoded, start=1):
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ValueError(f"control ranking row is invalid for {key}")
        try:
            row = RankedCandidate(**raw)
        except TypeError as exc:
            raise ValueError(f"control ranking row is invalid for {key}") from exc
        if (
            row.topic_id != topic_id
            or row.topic_id in PROTECTED_TOPIC_IDS
            or not isinstance(row.docid, str)
            or not row.docid
            or isinstance(row.rank, bool)
            or row.rank != rank
            or isinstance(row.score, bool)
            or not isinstance(row.score, (int, float))
            or not math.isfinite(float(row.score))
            or not isinstance(row.text, str)
            or not isinstance(row.provenance, list)
            or not all(isinstance(item, dict) for item in row.provenance)
        ):
            raise ValueError(f"control ranking row is invalid for {key}")
        rows.append(row)
    if len({row.docid for row in rows}) != 100:
        raise ValueError(f"control ranking has duplicate docids for {key}")
    if _ranking_sha256(rows) != _require_sha256(
        record.get("sha256"), f"control ranking {key}"
    ):
        raise ValueError(f"control ranking SHA-256 mismatch for {key}")
    return tuple(rows)


def _inspection_decision(
    coherence_failed: bool,
    domain_warning: bool,
    content_warning: bool,
) -> str:
    if coherence_failed:
        return "reject_coherence"
    if domain_warning and content_warning:
        return "reject_independent_warnings"
    if domain_warning or content_warning:
        return "keep_with_warning"
    return "keep"


def _validate_inspections(
    decoded: object,
    candidate_streams: Mapping[
        tuple[str, str, str], tuple[FrozenCandidateRow, ...]
    ],
) -> dict[str, InspectionResult]:
    expected = {
        f"{stream_id}/{arm_id}" for stream_id in STREAM_IDS for arm_id in ARM_IDS
    }
    if not isinstance(decoded, dict) or set(decoded) != expected:
        raise ValueError("control freeze inspections differ from the exact 16-arm set")
    result: dict[str, InspectionResult] = {}
    count_fields = (
        "inspected_top5",
        "inspected_top10",
        "anchor_top5_count",
        "anchor_top10_count",
        "anchor_intent_cohit_top5_count",
        "anchor_intent_cohit_top10_count",
        "domain_drift_top5_count",
        "domain_drift_top10_count",
        "content_quality_top5_count",
        "content_quality_top10_count",
    )
    boolean_fields = (
        "coherence_failed",
        "domain_drift_warning",
        "content_quality_warning",
        "rejected",
    )
    for key in sorted(expected):
        raw = decoded[key]
        if not isinstance(raw, dict) or set(raw) != _INSPECTION_FIELDS:
            raise ValueError(f"control freeze inspection fields differ for {key}")
        stream_id, arm_id = key.rsplit("/", 1)
        topic_id, facet_id = stream_id.split("/", 1)
        if raw.get("topic_id") != topic_id or raw.get("stream_id") != facet_id:
            raise ValueError(f"control freeze inspection identity differs for {key}")
        if any(
            isinstance(raw.get(field), bool)
            or not isinstance(raw.get(field), int)
            or raw[field] < 0
            for field in count_fields
        ) or any(not isinstance(raw.get(field), bool) for field in boolean_fields):
            raise ValueError(f"control freeze inspection types differ for {key}")
        if (
            raw["inspected_top5"] != 5
            or raw["inspected_top10"] != 10
            or any(raw[field] > 5 for field in count_fields if field.endswith("top5_count"))
            or any(raw[field] > 10 for field in count_fields if field.endswith("top10_count"))
            or raw["anchor_top5_count"] > raw["anchor_top10_count"]
            or raw["anchor_intent_cohit_top5_count"]
            > raw["anchor_intent_cohit_top10_count"]
            or raw["domain_drift_top5_count"] > raw["domain_drift_top10_count"]
            or raw["content_quality_top5_count"]
            > raw["content_quality_top10_count"]
            or raw["anchor_intent_cohit_top5_count"] > raw["anchor_top5_count"]
            or raw["anchor_intent_cohit_top10_count"] > raw["anchor_top10_count"]
        ):
            raise ValueError(f"control freeze inspection counts differ for {key}")
        top_docids = raw.get("top_docids")
        top_snippets = raw.get("top_snippets")
        expected_docids = [
            row.docid for row in candidate_streams[(topic_id, facet_id, arm_id)][:10]
        ]
        if (
            not isinstance(top_docids, list)
            or top_docids != expected_docids
            or len(set(top_docids)) != 10
            or not isinstance(top_snippets, list)
            or len(top_snippets) != 10
            or not all(isinstance(value, str) and len(value) <= 400 for value in top_snippets)
        ):
            raise ValueError(f"control freeze inspection representative results differ for {key}")
        coherence = (
            raw["inspected_top5"] < 5
            or raw["anchor_top5_count"] < 3
            or raw["anchor_intent_cohit_top5_count"] < 3
        )
        domain_warning = raw["domain_drift_top5_count"] >= 2
        content_warning = raw["content_quality_top5_count"] >= 2
        rejected = coherence or (domain_warning and content_warning)
        decision = _inspection_decision(coherence, domain_warning, content_warning)
        if (
            raw["coherence_failed"] is not coherence
            or raw["domain_drift_warning"] is not domain_warning
            or raw["content_quality_warning"] is not content_warning
            or raw["rejected"] is not rejected
            or raw.get("decision") != decision
        ):
            raise ValueError(f"control freeze inspection decision is inconsistent for {key}")
        result[key] = InspectionResult(
            **{
                **raw,
                "top_docids": tuple(top_docids),
                "top_snippets": tuple(top_snippets),
            }
        )
    return result


def verify_control_freeze(path: Path) -> VerifiedControlFreeze:
    """Read and verify every v2 control artifact exactly once."""

    freeze_path = Path(path)
    if freeze_path.is_dir():
        freeze_path = freeze_path / "freeze.json"
    try:
        root_bytes = freeze_path.read_bytes()
        payload = json.loads(root_bytes)
    except OSError as exc:
        raise ValueError(f"control freeze is unreadable: {freeze_path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("control freeze is not valid canonical JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != _ROOT_FIELDS
        or _canonical_json(payload) != root_bytes
        or payload.get("status") != "frozen_before_qrels"
    ):
        raise ValueError("control freeze metadata is corrupt or incomplete")
    schema = payload.get("schema_version")
    if schema not in {FREEZE_SCHEMA_VERSION, FREEZE_SCHEMA_VERSION_V2}:
        raise ValueError(f"unsupported control freeze schema: {schema!r}")
    expected_self = _require_sha256(payload.get("freeze_sha256"), "control freeze")
    without_self = dict(payload)
    without_self.pop("freeze_sha256")
    if hashlib.sha256(_canonical_json(without_self)).hexdigest() != expected_self:
        raise ValueError("control freeze self SHA-256 mismatch")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("control freeze bindings are corrupt or incomplete")
    _validate_bindings(bindings)
    if schema == FREEZE_SCHEMA_VERSION:
        raise ValueError("candidate snapshot required for marginal evaluation")
    if set(bindings) != _V2_BINDING_FIELDS:
        raise ValueError("control freeze v2 writer bindings are incomplete")

    directory = freeze_path.parent
    try:
        manifest_bytes = (directory / "candidate_streams.json").read_bytes()
        candidate_bytes = (directory / "candidates.jsonl").read_bytes()
    except OSError as exc:
        raise ValueError("candidate snapshot artifact is unreadable") from exc
    if (
        hashlib.sha256(manifest_bytes).hexdigest()
        != bindings["candidate_streams_sha256"]
        or hashlib.sha256(candidate_bytes).hexdigest() != bindings["candidates_sha256"]
    ):
        raise ValueError("candidate snapshot file SHA-256 differs from freeze bindings")
    snapshot = CandidateSnapshot(
        schema_version=CANDIDATE_SNAPSHOT_SCHEMA_VERSION,
        manifest_bytes=manifest_bytes,
        candidate_bytes=candidate_bytes,
    )
    snapshot_manifest, raw_candidates = validate_candidate_snapshot(snapshot)
    _validate_candidate_snapshot_source_bindings(snapshot_manifest, bindings)
    stream_hashes = {
        f"{entry['topic_id']}/{entry['stream_id']}/{entry['arm_id']}": entry[
            "stream_rows_sha256"
        ]
        for entry in snapshot_manifest["streams"]
    }
    if bindings["candidate_stream_rows_sha256"] != stream_hashes:
        raise ValueError("candidate snapshot stream hashes differ from freeze bindings")
    candidate_streams: dict[
        tuple[str, str, str], list[FrozenCandidateRow]
    ] = {}
    for raw in raw_candidates:
        row = FrozenCandidateRow(
            topic_id=raw["topic_id"],
            stream_id=raw["stream_id"],
            arm_id=raw["arm_id"],
            query_sha256=raw["query_sha256"],
            rank=raw["rank"],
            docid=raw["docid"],
        )
        candidate_streams.setdefault(
            (row.topic_id, row.stream_id, row.arm_id), []
        ).append(row)
    frozen_streams = {key: tuple(rows) for key, rows in candidate_streams.items()}

    try:
        inspection_bytes = (directory / "inspection.json").read_bytes()
    except OSError as exc:
        raise ValueError("control freeze inspection artifact is unreadable") from exc
    if hashlib.sha256(inspection_bytes).hexdigest() != _require_sha256(
        payload.get("inspection_sha256"), "control freeze inspection"
    ):
        raise ValueError("control freeze inspection SHA-256 mismatch")
    try:
        inspections = _validate_inspections(json.loads(inspection_bytes), frozen_streams)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("control freeze inspection is not valid JSON") from exc

    try:
        fusion_bytes = (directory / "fusion.json").read_bytes()
        fusion = json.loads(fusion_bytes)
    except OSError as exc:
        raise ValueError("control freeze fusion artifact is unreadable") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("control freeze fusion is not valid JSON") from exc
    if (
        hashlib.sha256(fusion_bytes).hexdigest()
        != _require_sha256(payload.get("fusion_sha256"), "control freeze fusion")
        or fusion
        != {
            "k": RRF_K,
            "limit": RANKING_DEPTH,
            "original_family_weight": ORIGINAL_FAMILY_WEIGHT,
            "facet_family_weight": FACET_FAMILY_WEIGHT,
            "facet_stream_weight": "0.5 / active topic facet streams",
        }
    ):
        raise ValueError("control freeze fusion contract differs")

    records = payload.get("rankings")
    if not isinstance(records, dict) or set(records) != set(EXPECTED_ALTERNATIVE_NAMES):
        raise ValueError("control freeze rankings differ from the exact 25-name set")
    rankings: dict[str, tuple[RankedCandidate, ...]] = {}
    ranking_hashes: dict[str, str] = {}
    for key in EXPECTED_ALTERNATIVE_NAMES:
        record = records[key]
        if not isinstance(record, dict) or set(record) != {
            "path",
            "rows",
            "sha256",
            "file_sha256",
        }:
            raise ValueError(f"control freeze ranking metadata is invalid for {key}")
        relative = f"rankings/{key.replace(':', '__')}.jsonl"
        ranking_path = _artifact_path(directory, record.get("path"), relative)
        try:
            ranking_bytes = ranking_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"control ranking is unreadable for {key}") from exc
        rankings[key] = _decode_ranking(key, ranking_bytes, record)
        ranking_hashes[key] = record["sha256"]
    return VerifiedControlFreeze(
        payload=payload,
        file_sha256=hashlib.sha256(root_bytes).hexdigest(),
        inspections=inspections,
        rankings=rankings,
        ranking_sha256=ranking_hashes,
        candidate_streams=frozen_streams,
    )


def _prior_semantic_sha(rows: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                list(rows),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def verify_prior_freeze_snapshot(path: Path) -> VerifiedPriorFreeze:
    """Verify all prior rankings once and retain the exact O/F0/R1 rows used."""

    freeze_path = Path(path)
    if freeze_path.is_dir():
        freeze_path = freeze_path / "freeze.json"
    try:
        root_bytes = freeze_path.read_bytes()
        payload = json.loads(root_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prior freeze is unreadable or invalid") from exc
    file_sha256 = hashlib.sha256(root_bytes).hexdigest()
    if file_sha256 != freeze_contract.PRIOR_FREEZE_FILE_SHA256:
        raise ValueError("input differs from the exact immutable prior freeze")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "sparse-relevance-ranking-freeze-v1"
        or payload.get("status") != "frozen_before_qrels"
        or payload.get("freeze_sha256")
        != freeze_contract._PRIOR_FREEZE_INTERNAL_SHA256
        or payload.get("topic_ids") != list(freeze_contract._PRIOR_TOPIC_IDS)
        or payload.get("manifest_sha256") != freeze_contract._PRIOR_MANIFEST_SHA256
        or payload.get("fusion_definitions_sha256")
        != freeze_contract._PRIOR_FUSION_SHA256
    ):
        raise ValueError("exact immutable prior freeze contract differs")
    fusion_sha256 = hashlib.sha256(
        (
            json.dumps(
                payload.get("fusion_definitions"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if fusion_sha256 != freeze_contract._PRIOR_FUSION_SHA256:
        raise ValueError("exact immutable prior fusion definitions differ")
    without_self = dict(payload)
    without_self.pop("freeze_sha256", None)
    actual_self = hashlib.sha256(
        (
            json.dumps(
                without_self,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if actual_self != payload.get("freeze_sha256"):
        raise ValueError("prior freeze self SHA-256 is invalid")
    records = payload.get("rankings")
    if not isinstance(records, dict) or frozenset(records) != freeze_contract._PRIOR_RANKING_NAMES:
        raise ValueError("exact immutable prior freeze must name exactly 15 rankings")
    retained: dict[str, tuple[RankedCandidate, ...]] = {}
    for key in sorted(records):
        record = records[key]
        if not isinstance(record, dict) or record.get("rows") != 400:
            raise ValueError(f"prior ranking metadata is invalid for {key}")
        relative = record.get("path")
        if not isinstance(relative, str):
            relative = f"rankings/{key.replace(':', '__')}.jsonl"
        ranking_path = freeze_path.parent / relative
        try:
            ranking_path.resolve().relative_to(freeze_path.parent.resolve())
            ranking_bytes = ranking_path.read_bytes()
            decoded = [json.loads(line) for line in ranking_bytes.splitlines() if line]
        except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"prior ranking is unreadable for {key}") from exc
        rows = _validate_prior_ranked_rows(key, decoded)
        if _prior_semantic_sha(rows) != _require_sha256(
            record.get("sha256"), f"prior ranking {key}"
        ):
            raise ValueError(f"prior ranking SHA-256 mismatch for {key}")
        arm, fusion = key.split(":", 1)
        if arm in {"O", "F0", "R1"} and fusion == "family_rrf":
            retained[arm] = tuple(RankedCandidate(**row) for row in rows)
    if set(retained) != {"O", "F0", "R1"}:
        raise ValueError("prior freeze lacks exact O/F0/R1 family rankings")
    return VerifiedPriorFreeze(file_sha256=file_sha256, rankings=retained)


def _self_bound_payload(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def publish_control_evaluation(
    output_dir: Path,
    payloads: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Publish four self-hashed files as one create-only directory."""

    output = Path(output_dir)
    if _lexists(output):
        raise FileExistsError(f"create-only output already exists: {output}")
    if set(payloads) != set(OUTPUT_NAMES):
        raise ValueError("evaluation publication requires exactly four named JSON files")
    frozen = {name: _self_bound_payload(payloads[name]) for name in OUTPUT_NAMES}
    encoded = {name: _canonical_json(frozen[name]) for name in OUTPUT_NAMES}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for name in OUTPUT_NAMES:
            (stage / name).write_bytes(encoded[name])
        _publish_directory_noreplace(stage, output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return frozen


def _evaluate_system(
    rows: Sequence[RankedCandidate],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    if {row.topic_id for row in rows} != set(EVALUATION_TOPIC_IDS):
        raise ValueError("system ranking topics differ from the exact evaluation set")
    for topic_id in EVALUATION_TOPIC_IDS:
        topic_rows = sorted(
            (row for row in rows if row.topic_id == topic_id), key=lambda row: row.rank
        )
        if (
            len(topic_rows) != 100
            or [row.rank for row in topic_rows] != list(range(1, 101))
            or len({row.docid for row in topic_rows}) != 100
        ):
            raise ValueError(f"system ranking is invalid for topic {topic_id}")
    return evaluate_ranked(
        list(rows),
        {topic_id: dict(qrels.get(topic_id, {})) for topic_id in EVALUATION_TOPIC_IDS},
        metric_names=SYSTEM_METRICS,
        relevance_threshold=2,
        topic_ids=EVALUATION_TOPIC_IDS,
    )


def _retrieved_snapshot_rows(
    rows: Sequence[FrozenCandidateRow],
) -> tuple[RetrievedCandidate, ...]:
    return tuple(
        RetrievedCandidate(
            topic_id=row.topic_id,
            variant_name=f"frozen:{row.stream_id}:{row.arm_id}",
            retriever_name="facet-control-ranking-freeze-v2",
            query_text=row.query_sha256,
            docid=row.docid,
            rank=row.rank,
            score=float(101 - row.rank),
            text="",
        )
        for row in rows
    )


def _selection_eligibility(
    arms: Mapping[str, StreamArmEvaluation],
) -> dict[str, dict[str, object]]:
    baseline = arms["B0"]
    result: dict[str, dict[str, object]] = {}
    for arm_id in ARM_IDS:
        arm = arms[arm_id]
        joint_noise = (
            arm.domain_drift_top10_count > baseline.domain_drift_top10_count
            and arm.content_quality_top10_count > baseline.content_quality_top10_count
        )
        reasons = []
        if arm.coherence_failed:
            reasons.append("coherence_failed")
        if joint_noise:
            reasons.append("both_noise_families_increased_vs_B0")
        result[arm_id] = {"eligible": not reasons, "exclusion_reasons": reasons}
    return result


def evaluate_control_freeze(
    control_freeze_path: Path,
    qrels_path: Path,
    *,
    prior_freeze_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, dict[str, object]]:
    """Verify immutable snapshots, then read qrels once and evaluate from memory."""

    if output_dir is not None and _lexists(Path(output_dir)):
        raise FileExistsError(f"create-only output already exists: {output_dir}")
    control = verify_control_freeze(control_freeze_path)
    if prior_freeze_path is None:
        raise ValueError("prior freeze is required")
    prior = verify_prior_freeze_snapshot(prior_freeze_path)
    if control.payload["bindings"]["prior_freeze_sha256"] != prior.file_sha256:
        raise ValueError("control freeze prior-freeze binding mismatch")

    # This is intentionally the first qrels read. No frozen path is reopened below.
    qrels_bytes = Path(qrels_path).read_bytes()
    qrels_sha256 = hashlib.sha256(qrels_bytes).hexdigest()
    all_qrels = parse_qrels_bytes(qrels_bytes)
    protected_qrels_topics = sorted(set(all_qrels) & set(PROTECTED_TOPIC_IDS))
    qrels = {
        topic_id: dict(all_qrels.get(topic_id, {})) for topic_id in EVALUATION_TOPIC_IDS
    }

    original = {
        topic_id: _retrieved_snapshot_rows(
            control.candidate_streams[(topic_id, "original", "O")]
        )
        for topic_id in EVALUATION_TOPIC_IDS
    }
    stream_results: dict[str, dict[str, StreamArmEvaluation]] = {}
    eligibility: dict[str, dict[str, dict[str, object]]] = {}
    selected: dict[str, str] = {}
    for stream_key in STREAM_IDS:
        topic_id, stream_id = stream_key.split("/", 1)
        arms = {
            arm_id: evaluate_stream_arm(
                arm_id,
                _retrieved_snapshot_rows(
                    control.candidate_streams[(topic_id, stream_id, arm_id)]
                ),
                original[topic_id],
                qrels[topic_id],
                control.inspections[f"{stream_key}/{arm_id}"],
            )
            for arm_id in ARM_IDS
        }
        stream_results[stream_key] = arms
        eligibility[stream_key] = _selection_eligibility(arms)
        selected[stream_key] = select_stream_arm(arms).arm_id

    references = selected_ranking_references(selected)
    r2_rows: list[RankedCandidate] = []
    selected_hashes: dict[str, str] = {}
    for topic_id in EVALUATION_TOPIC_IDS:
        reference = references[topic_id]
        r2_rows.extend(control.rankings[reference])
        selected_hashes[topic_id] = control.ranking_sha256[reference]
    systems = {
        arm: _evaluate_system(rows, qrels) for arm, rows in prior.rankings.items()
    }
    systems["R2"] = _evaluate_system(r2_rows, qrels)
    ndcg_deltas = {
        topic_id: systems["R2"]["per_topic"][topic_id]["ndcg@10"]
        - systems["R1"]["per_topic"][topic_id]["ndcg@10"]
        for topic_id in EVALUATION_TOPIC_IDS
    }
    selected_noise = sum(
        stream_results[stream_id][selected[stream_id]].top10_noise
        for stream_id in STREAM_IDS
    )
    b0_noise = sum(stream_results[stream_id]["B0"].top10_noise for stream_id in STREAM_IDS)
    decision = retrieval_repair_decision(
        r2_graded_recall=systems["R2"]["metrics"]["graded_recall@100"],
        r1_graded_recall=systems["R1"]["metrics"]["graded_recall@100"],
        r2_ndcg=systems["R2"]["metrics"]["ndcg@10"],
        r1_ndcg=systems["R1"]["metrics"]["ndcg@10"],
        per_topic_ndcg_deltas=tuple(ndcg_deltas.values()),
        selected_noise=selected_noise,
        b0_noise=b0_noise,
    )
    bindings = control.payload["bindings"]
    provenance = {
        "control_freeze_file_sha256": control.file_sha256,
        "control_freeze_sha256": control.payload["freeze_sha256"],
        "manifest_sha256": bindings["manifest_sha256"],
        "prior_freeze_sha256": prior.file_sha256,
        "qrels_sha256": qrels_sha256,
        "qrels_name": Path(qrels_path).name,
    }
    payloads: dict[str, dict[str, object]] = {
        "stream_evaluation.json": {
            "schema_version": "facet-control-stream-evaluation-v1",
            "provenance": provenance,
            "qrels_policy": {
                "relevance_threshold": 2,
                "missing_judgments_grade": 0,
                "graded_recall_denominator": "sum of non-negative qrel grades",
                "zero_denominator_value": 0.0,
                "protected_qrels_topics_skipped": protected_qrels_topics,
            },
            "streams": {
                stream_id: {
                    "arms": {
                        arm_id: {
                            "metrics": jsonable(stream_results[stream_id][arm_id]),
                            "inspection": jsonable(
                                control.inspections[f"{stream_id}/{arm_id}"]
                            ),
                        }
                        for arm_id in ARM_IDS
                    }
                }
                for stream_id in STREAM_IDS
            },
        },
        "selection.json": {
            "schema_version": "facet-control-selection-v1",
            "provenance": provenance,
            "selected": selected,
            "eligibility": eligibility,
            "selected_rankings": references,
            "selected_ranking_sha256": selected_hashes,
            "frozen_alternative_count": len(control.rankings),
        },
        "evaluation.json": {
            "schema_version": "facet-control-system-evaluation-v1",
            "provenance": provenance,
            "systems": systems,
            "r2_selected_rankings": references,
            "r2_selected_ranking_sha256": selected_hashes,
            "per_topic_ndcg_delta_vs_r1": ndcg_deltas,
        },
        "decision.json": {
            "schema_version": "facet-control-decision-v1",
            "provenance": provenance,
            "decision": decision,
            "reranker_gate_opened": False,
            "evidence": {
                "r2_graded_recall_at_100": systems["R2"]["metrics"]["graded_recall@100"],
                "r1_graded_recall_at_100": systems["R1"]["metrics"]["graded_recall@100"],
                "r2_ndcg_at_10": systems["R2"]["metrics"]["ndcg@10"],
                "r1_ndcg_at_10": systems["R1"]["metrics"]["ndcg@10"],
                "per_topic_ndcg_deltas": ndcg_deltas,
                "selected_top10_noise": selected_noise,
                "b0_top10_noise": b0_noise,
            },
        },
    }
    if output_dir is not None:
        return publish_control_evaluation(Path(output_dir), payloads)
    return payloads


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--prior-freeze", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payloads = evaluate_control_freeze(
        args.freeze_dir,
        args.qrels,
        prior_freeze_path=args.prior_freeze,
        output_dir=args.output,
    )
    print(json.dumps(payloads["decision.json"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
