"""Verify and evaluate the frozen facet retrieval-control pilot offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .evaluation import evaluate_ranked, parse_qrels_bytes
from .facet_retrieval_control_experiment import (
    EXPECTED_ALTERNATIVE_NAMES,
    FACET_FAMILY_WEIGHT,
    ORIGINAL_FAMILY_WEIGHT,
    RANKING_DEPTH,
    RRF_K,
    StreamArmEvaluation,
    evaluate_stream_arm,
    index_control_streams,
    retrieval_repair_decision,
    select_stream_arm,
    selected_ranking_references,
)
from .facet_retrieval_control_freeze import (
    FREEZE_SCHEMA_VERSION,
    _canonical_json,
    _load_control_candidates,
    _publish_directory_noreplace,
    _ranking_sha256,
    _validate_bindings,
    _validate_prior_ranked_rows,
    _ledger_tree_sha256,
    load_verified_r1_arm,
    verify_prior_freeze,
)
from .facet_retrieval_control_manifest import (
    PROTECTED_TOPIC_IDS,
    load_control_manifest,
)
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


@dataclass(frozen=True)
class VerifiedControlFreeze:
    """Fully decoded control freeze whose complete artifact set has verified."""

    directory: Path
    payload: dict[str, object]
    file_sha256: str
    inspections: dict[str, dict[str, object]]
    rankings: dict[str, tuple[RankedCandidate, ...]]


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_artifact_path(directory: Path, relative: object, expected: str) -> Path:
    if relative != expected:
        raise ValueError(f"control freeze artifact path differs for {expected}")
    candidate = directory / expected
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError("control freeze artifact path escapes its directory") from exc
    return candidate


def _decode_control_ranking(
    key: str,
    path: Path,
    record: Mapping[str, object],
) -> tuple[RankedCandidate, ...]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"control freeze ranking is unreadable for {key}") from exc
    if hashlib.sha256(content).hexdigest() != _require_sha256(
        record.get("file_sha256"), f"control ranking file {key}"
    ):
        raise ValueError(f"control ranking file SHA-256 mismatch for {key}")
    try:
        decoded = [json.loads(line) for line in content.splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"control freeze ranking is invalid JSONL for {key}") from exc
    if len(decoded) != RANKING_DEPTH or record.get("rows") != RANKING_DEPTH:
        raise ValueError(f"control freeze ranking row count differs for {key}")
    rows: list[RankedCandidate] = []
    expected_fields = {"topic_id", "docid", "rank", "score", "text", "provenance"}
    topic_id = key.split(":", 2)[1]
    for index, raw in enumerate(decoded, start=1):
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ValueError(f"control freeze ranking row is invalid for {key}")
        try:
            row = RankedCandidate(**raw)
        except TypeError as exc:
            raise ValueError(f"control freeze ranking row is invalid for {key}") from exc
        if (
            row.topic_id != topic_id
            or row.topic_id in PROTECTED_TOPIC_IDS
            or not isinstance(row.docid, str)
            or not row.docid
            or isinstance(row.rank, bool)
            or row.rank != index
            or isinstance(row.score, bool)
            or not isinstance(row.score, (int, float))
            or not math.isfinite(float(row.score))
            or not isinstance(row.text, str)
            or not isinstance(row.provenance, list)
            or not all(isinstance(item, dict) for item in row.provenance)
        ):
            raise ValueError(f"control freeze ranking row is invalid for {key}")
        rows.append(row)
    if len({row.docid for row in rows}) != RANKING_DEPTH:
        raise ValueError(f"control freeze ranking has duplicate docids for {key}")
    if _ranking_sha256(rows) != _require_sha256(
        record.get("sha256"), f"control ranking {key}"
    ):
        raise ValueError(f"control ranking SHA-256 mismatch for {key}")
    return tuple(rows)


def _validate_inspections(decoded: object) -> dict[str, dict[str, object]]:
    expected = {
        f"{stream_id}/{arm_id}"
        for stream_id in STREAM_IDS
        for arm_id in ARM_IDS
    }
    if not isinstance(decoded, dict) or set(decoded) != expected:
        raise ValueError("control freeze inspections differ from the exact 16-arm set")
    result: dict[str, dict[str, object]] = {}
    for key, raw in decoded.items():
        if not isinstance(raw, dict):
            raise ValueError(f"control freeze inspection is invalid for {key}")
        stream_id, arm_id = key.rsplit("/", 1)
        topic_id, facet_id = stream_id.split("/", 1)
        if (
            raw.get("topic_id") != topic_id
            or raw.get("stream_id") != facet_id
            or arm_id not in ARM_IDS
            or not isinstance(raw.get("coherence_failed"), bool)
        ):
            raise ValueError(f"control freeze inspection identity is invalid for {key}")
        for field in ("domain_drift_top10_count", "content_quality_top10_count"):
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"control freeze inspection {field} is invalid for {key}")
        result[key] = dict(raw)
    return result


def verify_control_freeze(path: Path) -> VerifiedControlFreeze:
    """Verify freeze metadata, hashes, inspections, and all 25 ranking files."""

    freeze_path = Path(path)
    if freeze_path.is_dir():
        freeze_path = freeze_path / "freeze.json"
    try:
        source = freeze_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"control freeze is unreadable: {freeze_path}") from exc
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError("control freeze is not valid JSON") from exc
    required_fields = {
        "schema_version",
        "status",
        "bindings",
        "inspection_sha256",
        "fusion_sha256",
        "rankings",
        "freeze_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required_fields
        or payload.get("schema_version") != FREEZE_SCHEMA_VERSION
        or payload.get("status") != "frozen_before_qrels"
    ):
        raise ValueError("control freeze metadata is corrupt or incomplete")
    expected_self = _require_sha256(payload.get("freeze_sha256"), "control freeze")
    without_self = dict(payload)
    without_self.pop("freeze_sha256")
    if hashlib.sha256(_canonical_json(without_self)).hexdigest() != expected_self:
        raise ValueError("control freeze self SHA-256 mismatch")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("control freeze bindings are corrupt or incomplete")
    _validate_bindings(bindings)

    directory = freeze_path.parent
    inspection_path = _safe_artifact_path(
        directory, "inspection.json", "inspection.json"
    )
    try:
        inspection_bytes = inspection_path.read_bytes()
    except OSError as exc:
        raise ValueError("control freeze inspection artifact is unreadable") from exc
    if hashlib.sha256(inspection_bytes).hexdigest() != _require_sha256(
        payload.get("inspection_sha256"), "control freeze inspection"
    ):
        raise ValueError("control freeze inspection SHA-256 mismatch")
    try:
        inspections = _validate_inspections(json.loads(inspection_bytes))
    except json.JSONDecodeError as exc:
        raise ValueError("control freeze inspection is not valid JSON") from exc

    fusion_path = _safe_artifact_path(directory, "fusion.json", "fusion.json")
    try:
        fusion_bytes = fusion_path.read_bytes()
    except OSError as exc:
        raise ValueError("control freeze fusion artifact is unreadable") from exc
    if hashlib.sha256(fusion_bytes).hexdigest() != _require_sha256(
        payload.get("fusion_sha256"), "control freeze fusion"
    ):
        raise ValueError("control freeze fusion SHA-256 mismatch")
    try:
        fusion = json.loads(fusion_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("control freeze fusion is not valid JSON") from exc
    if fusion != {
        "k": RRF_K,
        "limit": RANKING_DEPTH,
        "original_family_weight": ORIGINAL_FAMILY_WEIGHT,
        "facet_family_weight": FACET_FAMILY_WEIGHT,
        "facet_stream_weight": "0.5 / active topic facet streams",
    }:
        raise ValueError("control freeze fusion contract differs")

    ranking_records = payload.get("rankings")
    if (
        not isinstance(ranking_records, dict)
        or set(ranking_records) != set(EXPECTED_ALTERNATIVE_NAMES)
    ):
        raise ValueError("control freeze rankings differ from the exact 25-name set")
    rankings: dict[str, tuple[RankedCandidate, ...]] = {}
    for key in EXPECTED_ALTERNATIVE_NAMES:
        record = ranking_records.get(key)
        if not isinstance(record, dict) or set(record) != {
            "path",
            "rows",
            "sha256",
            "file_sha256",
        }:
            raise ValueError(f"control freeze ranking metadata is invalid for {key}")
        relative = f"rankings/{key.replace(':', '__')}.jsonl"
        ranking_path = _safe_artifact_path(directory, record.get("path"), relative)
        rankings[key] = _decode_control_ranking(key, ranking_path, record)
    return VerifiedControlFreeze(
        directory=directory,
        payload=payload,
        file_sha256=hashlib.sha256(source).hexdigest(),
        inspections=inspections,
        rankings=rankings,
    )


def _self_bound_payload(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def publish_control_evaluation(
    output_dir: Path,
    payloads: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Publish the four self-hashed JSON artifacts as one create-only directory."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    if set(payloads) != set(OUTPUT_NAMES):
        raise ValueError("evaluation publication requires exactly four named JSON files")
    frozen = {name: _self_bound_payload(payloads[name]) for name in OUTPUT_NAMES}
    encoded = {name: _canonical_json(frozen[name]) for name in OUTPUT_NAMES}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        for name in OUTPUT_NAMES:
            (stage / name).write_bytes(encoded[name])
        _publish_directory_noreplace(stage, output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return frozen


def _load_prior_system_rankings(
    prior_freeze_path: Path,
) -> tuple[str, dict[str, tuple[RankedCandidate, ...]]]:
    prior_sha256 = verify_prior_freeze(prior_freeze_path)
    freeze_path = Path(prior_freeze_path)
    if freeze_path.is_dir():
        freeze_path = freeze_path / "freeze.json"
    payload = json.loads(freeze_path.read_bytes())
    records = payload["rankings"]
    result: dict[str, tuple[RankedCandidate, ...]] = {}
    for arm in ("O", "F0", "R1"):
        key = f"{arm}:family_rrf"
        record = records[key]
        relative = record.get("path")
        ranking_path = (
            freeze_path.parent / relative
            if isinstance(relative, str)
            else freeze_path.parent / "rankings" / f"{key.replace(':', '__')}.jsonl"
        )
        decoded = [
            json.loads(line)
            for line in ranking_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        validated = _validate_prior_ranked_rows(key, decoded)
        result[arm] = tuple(RankedCandidate(**raw) for raw in validated)
    return prior_sha256, result


def _evaluate_system(
    rows: Sequence[RankedCandidate],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    topics = {row.topic_id for row in rows}
    if topics != set(EVALUATION_TOPIC_IDS):
        raise ValueError("system ranking topics differ from the exact evaluation set")
    for topic_id in EVALUATION_TOPIC_IDS:
        topic_rows = sorted(
            (row for row in rows if row.topic_id == topic_id), key=lambda row: row.rank
        )
        if (
            len(topic_rows) != RANKING_DEPTH
            or [row.rank for row in topic_rows] != list(range(1, 101))
            or len({row.docid for row in topic_rows}) != RANKING_DEPTH
        ):
            raise ValueError(f"system ranking is invalid for topic {topic_id}")
    return evaluate_ranked(
        list(rows),
        {topic_id: dict(qrels.get(topic_id, {})) for topic_id in EVALUATION_TOPIC_IDS},
        metric_names=SYSTEM_METRICS,
        relevance_threshold=2,
        topic_ids=EVALUATION_TOPIC_IDS,
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
            and arm.content_quality_top10_count
            > baseline.content_quality_top10_count
        )
        reasons = []
        if arm.coherence_failed:
            reasons.append("coherence_failed")
        if joint_noise:
            reasons.append("both_noise_families_increased_vs_B0")
        result[arm_id] = {"eligible": not reasons, "exclusion_reasons": reasons}
    return result


def _verified_source_rows(
    *,
    verified: VerifiedControlFreeze,
    manifest_path: Path,
    prior_freeze_path: Path,
    base_run: Path,
    base_cache: Path,
    r1_run: Path,
    r1_cache: Path,
    control_run: Path,
    control_cache: Path,
) -> tuple[
    list[RetrievedCandidate],
    list[RetrievedCandidate],
    dict[str, tuple[RankedCandidate, ...]],
    str,
    str,
]:
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = load_control_manifest(manifest_path)
    protected = sorted(
        {stream.topic_id for stream in manifest.streams} & set(PROTECTED_TOPIC_IDS)
    )
    if protected:
        raise ValueError(f"protected topic {protected[0]} is forbidden")
    prior_sha256, prior_rankings = _load_prior_system_rankings(prior_freeze_path)
    r1_arm, lineage = load_verified_r1_arm(
        manifest,
        base_run=base_run,
        base_cache=base_cache,
        r1_run=r1_run,
        r1_cache=r1_cache,
    )
    control_rows, request_hashes, response_hashes, candidate_hashes = (
        _load_control_candidates(manifest, control_run, control_cache)
    )
    expected_bindings: dict[str, object] = {
        "manifest_sha256": manifest_sha256,
        "prior_freeze_sha256": prior_sha256,
        "ledger_sha256": {
            **lineage["ledger_sha256"],
            "control": _ledger_tree_sha256(control_run),
        },
        "r1_arm_sha256": lineage["r1_arm_sha256"],
        "request_sha256": {**lineage["request_sha256"], **request_hashes},
        "response_sha256": {**lineage["response_sha256"], **response_hashes},
        "candidate_sha256": {**lineage["candidate_sha256"], **candidate_hashes},
    }
    if verified.payload["bindings"] != expected_bindings:
        raise ValueError("control freeze source or prior-freeze binding mismatch")
    return r1_arm, control_rows, prior_rankings, manifest_sha256, prior_sha256


def evaluate_control_freeze(
    control_freeze_path: Path,
    qrels_path: Path,
    *,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
    prior_freeze_path: Path | None = None,
    base_run: Path | None = None,
    base_cache: Path | None = None,
    r1_run: Path | None = None,
    r1_cache: Path | None = None,
    control_run: Path | None = None,
    control_cache: Path | None = None,
) -> dict[str, dict[str, object]]:
    """Verify every frozen input, then open qrels once and publish evaluation."""

    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f"create-only output already exists: {output_dir}")
    verified = verify_control_freeze(control_freeze_path)
    required_sources = {
        "manifest_path": manifest_path,
        "prior_freeze_path": prior_freeze_path,
        "base_run": base_run,
        "base_cache": base_cache,
        "r1_run": r1_run,
        "r1_cache": r1_cache,
        "control_run": control_run,
        "control_cache": control_cache,
    }
    missing = [name for name, value in required_sources.items() if value is None]
    if missing:
        raise ValueError(
            "verified freeze evaluation requires source artifacts: " + ", ".join(missing)
        )
    r1_arm, control_rows, prior_rankings, manifest_sha256, prior_sha256 = (
        _verified_source_rows(
            verified=verified,
            manifest_path=Path(manifest_path),
            prior_freeze_path=Path(prior_freeze_path),
            base_run=Path(base_run),
            base_cache=Path(base_cache),
            r1_run=Path(r1_run),
            r1_cache=Path(r1_cache),
            control_run=Path(control_run),
            control_cache=Path(control_cache),
        )
    )

    # This is intentionally the first qrels read, after every freeze/source/output gate.
    qrels_bytes = Path(qrels_path).read_bytes()
    qrels_sha256 = hashlib.sha256(qrels_bytes).hexdigest()
    all_qrels = parse_qrels_bytes(qrels_bytes)
    protected_qrels_topics = sorted(set(all_qrels) & set(PROTECTED_TOPIC_IDS))
    qrels = {
        topic_id: dict(all_qrels.get(topic_id, {}))
        for topic_id in EVALUATION_TOPIC_IDS
    }

    manifest = load_control_manifest(Path(manifest_path))
    indexed = index_control_streams(r1_arm, control_rows, manifest)
    original_by_topic: dict[str, tuple[RetrievedCandidate, ...]] = {}
    for topic_id in EVALUATION_TOPIC_IDS:
        original_by_topic[topic_id] = tuple(
            sorted(
                (
                    row
                    for row in r1_arm
                    if row.topic_id == topic_id
                    and row.variant_name == "prompt_lab_v1:original"
                ),
                key=lambda row: (row.rank, row.docid),
            )
        )

    stream_results: dict[str, dict[str, StreamArmEvaluation]] = {}
    selected: dict[str, str] = {}
    eligibility: dict[str, dict[str, dict[str, object]]] = {}
    for stream in manifest.streams:
        stream_key = f"{stream.topic_id}/{stream.stream_id}"
        arms = {
            arm_id: evaluate_stream_arm(
                arm_id,
                indexed[(stream.topic_id, stream.stream_id, arm_id)],
                original_by_topic[stream.topic_id],
                qrels[stream.topic_id],
                verified.inspections[f"{stream_key}/{arm_id}"],
            )
            for arm_id in ARM_IDS
        }
        stream_results[stream_key] = arms
        selected[stream_key] = select_stream_arm(arms).arm_id
        eligibility[stream_key] = _selection_eligibility(arms)

    references = selected_ranking_references(selected)
    r2_rows: list[RankedCandidate] = []
    selected_ranking_hashes: dict[str, str] = {}
    ranking_records = verified.payload["rankings"]
    for topic_id in EVALUATION_TOPIC_IDS:
        reference = references[topic_id]
        if reference not in verified.rankings:
            raise ValueError(f"selected frozen ranking is missing: {reference}")
        selected_ranking_hashes[topic_id] = ranking_records[reference]["sha256"]
        r2_rows.extend(verified.rankings[reference])
    systems = {
        arm: _evaluate_system(rows, qrels)
        for arm, rows in prior_rankings.items()
    }
    systems["R2"] = _evaluate_system(r2_rows, qrels)
    r1_per_topic = systems["R1"]["per_topic"]
    r2_per_topic = systems["R2"]["per_topic"]
    ndcg_deltas = {
        topic_id: r2_per_topic[topic_id]["ndcg@10"]
        - r1_per_topic[topic_id]["ndcg@10"]
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
    provenance = {
        "control_freeze_file_sha256": verified.file_sha256,
        "control_freeze_sha256": verified.payload["freeze_sha256"],
        "manifest_sha256": manifest_sha256,
        "prior_freeze_sha256": prior_sha256,
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
                            "inspection": verified.inspections[
                                f"{stream_id}/{arm_id}"
                            ],
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
            "selected_ranking_sha256": selected_ranking_hashes,
            "frozen_alternative_count": len(verified.rankings),
        },
        "evaluation.json": {
            "schema_version": "facet-control-system-evaluation-v1",
            "provenance": provenance,
            "systems": systems,
            "r2_selected_rankings": references,
            "r2_selected_ranking_sha256": selected_ranking_hashes,
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
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior-freeze", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--r1-cache", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--control-cache", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payloads = evaluate_control_freeze(
        args.freeze,
        args.qrels,
        output_dir=args.output,
        manifest_path=args.manifest,
        prior_freeze_path=args.prior_freeze,
        base_run=args.base_run,
        base_cache=args.base_cache,
        r1_run=args.r1_run,
        r1_cache=args.r1_cache,
        control_run=args.control_run,
        control_cache=args.control_cache,
    )
    print(json.dumps(payloads["decision.json"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
