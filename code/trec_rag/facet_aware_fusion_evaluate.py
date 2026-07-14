"""One-time, mechanically decided evaluation of frozen facet-aware rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from .facet_aware_fusion_rank import (
    ARM_NAMES,
    PRIOR_PILOT_TOPIC_IDS,
    PROTECTED_TOPIC_IDS,
    verify_ranking_freeze,
)


TOPIC_IDS = ("233", "273", "161", "14")
TOPIC_ID_SET = frozenset(TOPIC_IDS)
RELEVANCE_THRESHOLD = 2
PREFUSION_FACET_DEPTH = 20
RULE_TOLERANCE = 1e-12
EXPERIMENT_ID = "rag25_facet_aware_fusion_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ARTIFACT_NAMES = (
    "metrics.json",
    "gains_losses.json",
    "decision.json",
)
_DEFAULT_EXPERIMENT_DIR = Path("outputs/rag25_facet_aware_fusion_v1")
_DEFAULT_QRELS_DIR = (
    _DEFAULT_EXPERIMENT_DIR / "authorized_inputs/pilot_qrels_projection_v1"
)
_DEFAULT_QRELS_PATH = _DEFAULT_QRELS_DIR / "pilot_topics_233_273_161_14.qrels"
_DEFAULT_QRELS_MANIFEST = _DEFAULT_QRELS_DIR / "manifest.json"
_DEFAULT_QRELS_APPROVAL = _DEFAULT_EXPERIMENT_DIR / "approvals/qrels_access_v1.json"


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def _validated_sha256(value: object, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def add_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe payload bound to its own canonical content."""

    result = json.loads(_canonical_json_bytes(dict(payload)))
    result.pop("self_sha256", None)
    result["self_sha256"] = _sha256(_canonical_json_bytes(result))
    return result


def validate_self_hash(payload: Mapping[str, Any]) -> bool:
    """Validate an artifact created by :func:`add_self_hash`."""

    claimed = _validated_sha256(payload.get("self_sha256"), "artifact self hash")
    without_hash = dict(payload)
    without_hash.pop("self_sha256", None)
    if _sha256(_canonical_json_bytes(without_hash)) != claimed:
        raise ValueError("artifact self SHA-256 mismatch")
    return True


def _parse_json_object(source: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_exact_topic_order(topic_ids: Iterable[object], label: str) -> None:
    observed = tuple(str(topic_id) for topic_id in topic_ids)
    observed_set = set(observed)
    protected = observed_set & set(PROTECTED_TOPIC_IDS)
    if protected:
        raise ValueError(
            f"{label} contains protected topics: {', '.join(sorted(protected))}"
        )
    prior = observed_set & set(PRIOR_PILOT_TOPIC_IDS)
    if prior:
        raise ValueError(
            f"{label} contains prior-pilot topics: {', '.join(sorted(prior))}"
        )
    outside = observed_set - TOPIC_ID_SET
    if outside:
        raise ValueError(
            f"{label} contains topics outside the authorization: "
            + ", ".join(sorted(outside))
        )
    if observed_set == TOPIC_ID_SET and observed != TOPIC_IDS:
        raise ValueError(f"{label} topic order must be {', '.join(TOPIC_IDS)}")
    if observed != TOPIC_IDS:
        raise ValueError(f"{label} must contain exactly {', '.join(TOPIC_IDS)}")


def _dcg(grades: Sequence[int | float]) -> float:
    return math.fsum(
        (2 ** float(grade) - 1.0) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
        if float(grade) > 0.0
    )


def _normalized_qrels(qrels: Mapping[str, int | float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for docid, raw_grade in qrels.items():
        if not isinstance(docid, str) or not docid:
            raise ValueError("qrels document IDs must be non-empty text")
        if isinstance(raw_grade, bool) or not isinstance(raw_grade, (int, float)):
            raise ValueError("qrels grades must be numeric")
        grade = float(raw_grade)
        if not math.isfinite(grade) or grade < 0:
            raise ValueError("qrels grades must be finite and non-negative")
        result[docid] = grade
    return result


def evaluate_ranking(
    ranking: Sequence[str], qrels: Mapping[str, int | float]
) -> dict[str, int | float]:
    """Compute the five preregistered metrics for one topic and ranking."""

    docids = [str(docid) for docid in ranking]
    if any(not docid for docid in docids) or len(docids) != len(set(docids)):
        raise ValueError("ranking document IDs must be non-empty and unique")
    normalized = _normalized_qrels(qrels)
    top10 = docids[:10]
    top100 = docids[:100]
    positive_gain = sum(grade for grade in normalized.values() if grade > 0)
    retrieved_gain = sum(max(0.0, normalized.get(docid, 0.0)) for docid in top100)
    relevant = {
        docid for docid, grade in normalized.items() if grade >= RELEVANCE_THRESHOLD
    }
    ideal = sorted((grade for grade in normalized.values() if grade > 0), reverse=True)[
        :10
    ]
    actual = [max(0.0, normalized.get(docid, 0.0)) for docid in top10]
    ideal_dcg = _dcg(ideal)
    return {
        "graded_recall@100": retrieved_gain / positive_gain if positive_gain else 0.0,
        "ndcg@10": _dcg(actual) / ideal_dcg if ideal_dcg else 0.0,
        "relevant@10": sum(docid in relevant for docid in top10),
        "judged_rate@10": sum(docid in normalized for docid in top10) / 10.0,
        "judged_rate@100": sum(docid in normalized for docid in top100) / 100.0,
    }


def _validate_rankings(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
) -> None:
    if tuple(rankings) != tuple(ARM_NAMES):
        if set(rankings) != set(ARM_NAMES):
            raise ValueError("rankings must contain the exact six frozen arms")
        raise ValueError("ranking arm order differs from the frozen contract")
    for arm in ARM_NAMES:
        per_topic = rankings[arm]
        _validate_exact_topic_order(per_topic, f"{arm} rankings")
        for topic_id in TOPIC_IDS:
            ranking = per_topic[topic_id]
            if (
                not isinstance(ranking, Sequence)
                or isinstance(ranking, (str, bytes))
                or len(ranking) != 100
                or len(set(ranking)) != 100
                or any(not isinstance(docid, str) or not docid for docid in ranking)
            ):
                raise ValueError(
                    f"ranking {arm}/{topic_id} must have exactly 100 unique documents"
                )


def build_evaluation(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    qrels: Mapping[str, Mapping[str, int | float]],
    prefusion_candidates: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Build all system metrics and RRF-relative gains/losses in memory."""

    _validate_rankings(rankings)
    _validate_exact_topic_order(qrels, "qrels")
    _validate_exact_topic_order(prefusion_candidates, "pre-fusion candidates")
    metric_names = (
        "graded_recall@100",
        "ndcg@10",
        "relevant@10",
        "judged_rate@10",
        "judged_rate@100",
    )
    systems: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        per_topic = {
            topic_id: evaluate_ranking(rankings[arm][topic_id], qrels[topic_id])
            for topic_id in TOPIC_IDS
        }
        systems[arm] = {
            "per_topic": per_topic,
            "aggregate": {
                name: fmean(float(per_topic[topic_id][name]) for topic_id in TOPIC_IDS)
                for name in metric_names
            },
        }

    relevant_by_topic = {
        topic_id: {
            docid
            for docid, grade in _normalized_qrels(qrels[topic_id]).items()
            if grade >= RELEVANCE_THRESHOLD
        }
        for topic_id in TOPIC_IDS
    }
    rrf_sets = {
        topic_id: set(rankings["RRF"][topic_id]) for topic_id in TOPIC_IDS
    }
    prefusion_novel = {
        topic_id: (
            set(str(docid) for docid in prefusion_candidates[topic_id])
            & relevant_by_topic[topic_id]
        )
        - rrf_sets[topic_id]
        for topic_id in TOPIC_IDS
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        per_topic: dict[str, dict[str, Any]] = {}
        retained_total = 0
        denominator_total = 0
        for topic_id in TOPIC_IDS:
            candidate_set = set(rankings[arm][topic_id])
            candidate_relevant = candidate_set & relevant_by_topic[topic_id]
            rrf_relevant = rrf_sets[topic_id] & relevant_by_topic[topic_id]
            retained = candidate_set & prefusion_novel[topic_id]
            denominator = prefusion_novel[topic_id]
            retained_total += len(retained)
            denominator_total += len(denominator)
            per_topic[topic_id] = {
                "metric_deltas": {
                    name: float(systems[arm]["per_topic"][topic_id][name])
                    - float(systems["RRF"]["per_topic"][topic_id][name])
                    for name in metric_names
                },
                "relevant_gained_vs_rrf": sorted(candidate_relevant - rrf_relevant),
                "relevant_lost_vs_rrf": sorted(rrf_relevant - candidate_relevant),
                "pre_fusion_novel_relevant_docids": sorted(denominator),
                "novel_relevant_retained_docids": sorted(retained),
                "novel_relevant_retained": len(retained),
                "novel_retention_fraction": (
                    len(retained) / len(denominator) if denominator else 0.0
                ),
            }
        comparisons[arm] = {
            "baseline_arm": "RRF",
            "aggregate_deltas": {
                name: float(systems[arm]["aggregate"][name])
                - float(systems["RRF"]["aggregate"][name])
                for name in metric_names
            },
            "per_topic": per_topic,
            "pre_fusion_novel_relevant": denominator_total,
            "novel_relevant_retained": retained_total,
            "novel_retention_fraction": (
                retained_total / denominator_total if denominator_total else 0.0
            ),
            "topics_with_positive_novel_retention": sum(
                bool(per_topic[topic_id]["novel_relevant_retained"])
                for topic_id in TOPIC_IDS
            ),
        }
    return {
        "topic_ids": list(TOPIC_IDS),
        "systems": systems,
        "comparisons": comparisons,
    }


def _comparison(value: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    comparisons = value.get("comparisons")
    if not isinstance(comparisons, Mapping) or not isinstance(
        comparisons.get(arm), Mapping
    ):
        raise ValueError(f"evaluation lacks {arm} comparison metrics")
    return comparisons[arm]


def _system_aggregate(value: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    systems = value.get("systems")
    system = systems.get(arm) if isinstance(systems, Mapping) else None
    aggregate = system.get("aggregate") if isinstance(system, Mapping) else None
    if not isinstance(aggregate, Mapping):
        raise ValueError(f"evaluation lacks {arm} aggregate metrics")
    return aggregate


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _ndcg_guardrails(comparison: Mapping[str, Any]) -> tuple[bool, bool]:
    deltas = comparison.get("aggregate_deltas")
    if not isinstance(deltas, Mapping):
        raise ValueError("comparison lacks aggregate deltas")
    aggregate_ok = (
        _number(deltas.get("ndcg@10"), "aggregate nDCG delta")
        >= -0.01 - RULE_TOLERANCE
    )
    per_topic = comparison.get("per_topic")
    if not isinstance(per_topic, Mapping):
        raise ValueError("comparison lacks per-topic deltas")
    _validate_exact_topic_order(per_topic, "decision comparison")
    topic_values: list[float] = []
    for topic_id in TOPIC_IDS:
        row = per_topic[topic_id]
        metric_deltas = row.get("metric_deltas") if isinstance(row, Mapping) else None
        if not isinstance(metric_deltas, Mapping):
            raise ValueError("comparison lacks per-topic metric deltas")
        topic_values.append(
            _number(metric_deltas.get("ndcg@10"), f"{topic_id} nDCG delta")
        )
    return aggregate_ok, min(topic_values) >= -0.10 - RULE_TOLERANCE


def decide(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered CXQ/XQ/RRF promotion rule without tuning."""

    cxq = _comparison(evaluation, "CXQ")
    xq = _comparison(evaluation, "XQ")
    cxq_delta = cxq.get("aggregate_deltas")
    xq_delta = xq.get("aggregate_deltas")
    if not isinstance(cxq_delta, Mapping) or not isinstance(xq_delta, Mapping):
        raise ValueError("xQuAD comparisons lack aggregate deltas")
    cxq_ndcg, cxq_topic_ndcg = _ndcg_guardrails(cxq)
    xq_ndcg, xq_topic_ndcg = _ndcg_guardrails(xq)
    cxq_recall = _number(
        _system_aggregate(evaluation, "CXQ").get("graded_recall@100"),
        "CXQ graded recall",
    )
    xq_recall = _number(
        _system_aggregate(evaluation, "XQ").get("graded_recall@100"),
        "XQ graded recall",
    )
    cxq_novel = _number(cxq.get("novel_relevant_retained"), "CXQ novel retention")
    xq_novel = _number(xq.get("novel_relevant_retained"), "XQ novel retention")
    checks = {
        "graded_recall_improves": _number(
            cxq_delta.get("graded_recall@100"), "CXQ graded recall delta"
        )
        > 0.0,
        "novel_retention_fraction": _number(
            cxq.get("novel_retention_fraction"), "CXQ novel retention fraction"
        )
        >= 0.25,
        "positive_novel_retention_topics": _number(
            cxq.get("topics_with_positive_novel_retention"),
            "CXQ positive retention topics",
        )
        >= 3,
        "aggregate_ndcg_guardrail": cxq_ndcg,
        "per_topic_ndcg_guardrail": cxq_topic_ndcg,
        "not_worse_than_xq_on_both_primary_metrics": (
            cxq_recall >= xq_recall and cxq_novel >= xq_novel
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    xq_checks = {
        "graded_recall_improves": _number(
            xq_delta.get("graded_recall@100"), "XQ graded recall delta"
        )
        > 0.0,
        "aggregate_ndcg_guardrail": xq_ndcg,
        "per_topic_ndcg_guardrail": xq_topic_ndcg,
    }
    if not failed:
        promoted = "CXQ"
    elif failed == ["not_worse_than_xq_on_both_primary_metrics"] and all(
        xq_checks.values()
    ):
        promoted = "XQ"
    else:
        promoted = "RRF"
    return {
        "promoted_arm": promoted,
        "checks": checks,
        "failed_checks": failed,
        "xq_fallback_checks": xq_checks,
        "rule_version": "facet-aware-fusion-promotion-v1",
    }


def read_qrels(
    path: Path,
    *,
    expected_sha256: str | None = None,
    authorized_directory: Path | None = None,
    authorized_directory_identity: tuple[int, int] | None = None,
    authorized_projection_name: str | None = None,
) -> dict[str, dict[str, int]]:
    """Read one authorized synthetic/real projection in exact topic-block order."""

    if authorized_directory is None:
        try:
            source = Path(path).read_bytes()
        except OSError as exc:
            raise ValueError("authorized qrels projection is missing or invalid") from exc
    else:
        descriptor: int | None = None
        directory_descriptor: int | None = None
        try:
            directory_descriptor = os.open(
                Path(authorized_directory),
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            directory_stat = os.fstat(directory_descriptor)
            if authorized_directory_identity is not None and (
                directory_stat.st_dev,
                directory_stat.st_ino,
            ) != authorized_directory_identity:
                raise ValueError(
                    "authorized qrels directory changed after authorization"
                )
            projection_name = authorized_projection_name or Path(path).name
            if Path(projection_name).name != projection_name:
                raise ValueError("authorized qrels projection name is unsafe")
            descriptor = os.open(
                projection_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("authorized qrels projection must be a regular file")
            descriptor_target = Path(f"/proc/self/fd/{descriptor}").resolve()
            try:
                descriptor_target.relative_to(Path(authorized_directory).resolve())
            except ValueError as exc:
                raise ValueError(
                    "authorized qrels projection escaped its directory"
                ) from exc
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            source = b"".join(chunks)
        except OSError as exc:
            raise ValueError("authorized qrels projection is missing or invalid") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if directory_descriptor is not None:
                os.close(directory_descriptor)
    if expected_sha256 is not None and _sha256(source) != _validated_sha256(
        expected_sha256, "authorized qrels projection hash"
    ):
        raise ValueError("authorized qrels projection SHA-256 mismatch")
    try:
        lines = source.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("authorized qrels projection is not UTF-8") from exc
    result: dict[str, dict[str, int]] = defaultdict(dict)
    topic_blocks: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid qrels row at line {line_number}")
        topic_id, _iteration, docid, raw_grade = fields
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"qrels projection contains protected topic {topic_id}")
        if topic_id in PRIOR_PILOT_TOPIC_IDS:
            raise ValueError(f"qrels projection contains prior-pilot topic {topic_id}")
        if topic_id not in TOPIC_ID_SET:
            raise ValueError(
                f"qrels projection contains topic outside the authorization: {topic_id}"
            )
        if not topic_blocks or topic_blocks[-1] != topic_id:
            topic_blocks.append(topic_id)
        if not docid:
            raise ValueError(f"invalid qrels document ID at line {line_number}")
        try:
            grade = int(raw_grade)
        except ValueError as exc:
            raise ValueError(f"invalid qrels grade at line {line_number}") from exc
        if grade < 0:
            raise ValueError(f"invalid qrels grade at line {line_number}")
        previous = result[topic_id].get(docid)
        if previous is not None and previous != grade:
            raise ValueError(f"conflicting duplicate qrels row at line {line_number}")
        result[topic_id][docid] = grade
    _validate_exact_topic_order(topic_blocks, "qrels projection")
    return {topic_id: result[topic_id] for topic_id in TOPIC_IDS}


def trusted_qrels_consumption_dir() -> Path:
    """Return the path-independent registry namespace shared by worktrees."""

    completed = subprocess.run(
        ("git", "rev-parse", "--path-format=absolute", "--git-common-dir"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    common = Path(completed.stdout.strip()).resolve()
    if not common.is_dir():
        raise ValueError("git common directory is unavailable for qrels guard state")
    return common / "trec-rag-qrels-consumptions" / EXPERIMENT_ID


def qrels_consumption_identity(qrels_approval: Path) -> dict[str, str]:
    """Derive approval identity from bytes and hash bindings, never caller path."""

    try:
        source = Path(qrels_approval).read_bytes()
    except OSError as exc:
        raise ValueError("qrels access approval is missing or invalid") from exc
    approval = _parse_json_object(source, "qrels access approval")
    if (
        approval.get("schema_version") != "pilot-qrels-access-approval-v1"
        or approval.get("status") != "approved"
    ):
        raise ValueError("qrels access approval contract differs")
    _validate_exact_topic_order(approval.get("topic_ids", ()), "qrels access approval")
    semantic_identity = {
        "experiment_id": EXPERIMENT_ID,
        "qrels_manifest_sha256": _validated_sha256(
            approval.get("qrels_manifest_sha256"), "approval manifest hash"
        ),
        "qrels_projection_sha256": _validated_sha256(
            approval.get("qrels_projection_sha256"), "approval projection hash"
        ),
        "ranking_freeze_sha256": _validated_sha256(
            approval.get("ranking_freeze_sha256"), "approval ranking freeze hash"
        ),
    }
    return {
        **semantic_identity,
        "qrels_approval_sha256": _sha256(source),
        "identity_sha256": _sha256(_canonical_json_bytes(semantic_identity)),
    }


def qrels_consumption_registry_path(qrels_approval: Path) -> Path:
    identity = qrels_consumption_identity(Path(qrels_approval))
    return trusted_qrels_consumption_dir() / f"{identity['identity_sha256']}.json"


def _create_registry(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    registry = add_self_hash(payload)
    validate_self_hash(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError("qrels approval was already consumed") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(_canonical_json_bytes(registry, pretty=True))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return registry


def validate_qrels_authorization(
    qrels_path: Path,
    qrels_manifest: Path,
    qrels_approval: Path,
    *,
    ranking_freeze_sha256: str,
) -> dict[str, Any]:
    """Validate sidecar hashes and paths without opening the projection."""

    ranking_hash = _validated_sha256(
        ranking_freeze_sha256, "ranking freeze hash"
    )
    try:
        manifest_source = Path(qrels_manifest).read_bytes()
        approval_source = Path(qrels_approval).read_bytes()
    except OSError as exc:
        raise ValueError("qrels authorization sidecars are missing or invalid") from exc
    manifest = _parse_json_object(manifest_source, "qrels projection manifest")
    approval = _parse_json_object(approval_source, "qrels access approval")
    if (
        manifest.get("schema_version") != "pilot-qrels-projection-v1"
        or manifest.get("status") != "authorized_projection"
    ):
        raise ValueError("qrels projection manifest contract differs")
    _validate_exact_topic_order(manifest.get("topic_ids", ()), "qrels manifest")
    relative = manifest.get("projection_path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or len(Path(relative).parts) != 1
    ):
        raise ValueError("qrels projection path must remain inside its directory")
    declared = Path(qrels_manifest).parent / relative
    try:
        declared.resolve().relative_to(Path(qrels_manifest).parent.resolve())
    except ValueError as exc:
        raise ValueError(
            "qrels projection path must remain inside its authorized directory"
        ) from exc
    if declared.resolve() != Path(qrels_path).resolve():
        raise ValueError("qrels projection path differs from its manifest")
    projection_hash = _validated_sha256(
        manifest.get("projection_sha256"), "qrels projection hash"
    )
    if (
        approval.get("schema_version") != "pilot-qrels-access-approval-v1"
        or approval.get("status") != "approved"
    ):
        raise ValueError("qrels access approval contract differs")
    _validate_exact_topic_order(approval.get("topic_ids", ()), "qrels approval")
    manifest_hash = _sha256(manifest_source)
    if approval.get("qrels_manifest_sha256") != manifest_hash:
        raise ValueError("qrels manifest hash binding differs")
    if approval.get("qrels_projection_sha256") != projection_hash:
        raise ValueError("qrels projection hash binding differs")
    if approval.get("ranking_freeze_sha256") != ranking_hash:
        raise ValueError("ranking freeze hash binding differs")
    directory_stat = Path(qrels_manifest).parent.stat()
    return {
        "approval_sha256": _sha256(approval_source),
        "manifest_sha256": manifest_hash,
        "projection_sha256": projection_hash,
        "ranking_freeze_sha256": ranking_hash,
        "authorized_directory": Path(qrels_manifest).parent,
        "authorized_directory_identity": (
            directory_stat.st_dev,
            directory_stat.st_ino,
        ),
        "projection_name": relative,
    }


def _read_bound_artifact(
    root: Path, freeze: Mapping[str, Any], relative: str
) -> bytes:
    artifacts = freeze.get("artifacts")
    record = artifacts.get(relative) if isinstance(artifacts, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"ranking freeze lacks bound artifact {relative}")
    try:
        source = (root / relative).read_bytes()
    except OSError as exc:
        raise ValueError(f"ranking freeze artifact is unreadable: {relative}") from exc
    if record.get("bytes") != len(source) or record.get("sha256") != _sha256(source):
        raise ValueError(f"ranking freeze artifact hash mismatch: {relative}")
    return source


def _parse_canonical_jsonl(source: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} row {line_number} is invalid") from exc
        if not isinstance(row, dict) or line + b"\n" != _canonical_json_bytes(row):
            raise ValueError(f"{label} row {line_number} is not canonical JSON")
        rows.append(row)
    return rows


def _provenance_topic(row: Mapping[str, Any], label: str) -> str:
    topic_id = row.get("topic_id")
    if not isinstance(topic_id, str):
        raise ValueError(f"{label} topic is invalid")
    if topic_id in PROTECTED_TOPIC_IDS:
        raise ValueError(f"{label} contains protected topic {topic_id}")
    if topic_id in PRIOR_PILOT_TOPIC_IDS:
        raise ValueError(f"{label} contains prior-pilot topic {topic_id}")
    if topic_id not in TOPIC_ID_SET:
        raise ValueError(f"{label} contains outside topic {topic_id}")
    return topic_id


def _ranked_provenance_identity(
    row: Mapping[str, Any],
    *,
    label: str,
    expected_family: str,
    maximum_rank: int,
) -> tuple[str, int, str]:
    topic_id = _provenance_topic(row, label)
    if row.get("family") != expected_family:
        raise ValueError(f"{label} family must be {expected_family}")
    rank = row.get("rank")
    docid = row.get("document_id", row.get("docid"))
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 1 <= rank <= maximum_rank
        or not isinstance(docid, str)
        or not docid
    ):
        raise ValueError(f"{label} rank/document identity is invalid")
    return topic_id, rank, docid


def validate_prefusion_provenance(
    provenance: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    candidate_pool_sha256: Mapping[str, str],
) -> dict[str, list[str]]:
    """Authenticate the accepted facet top-20 denominator against Task 3."""

    pool_topics = {
        _provenance_topic(
            {"topic_id": topic_id}, "candidate pool hashes"
        )
        for topic_id in candidate_pool_sha256
    }
    if pool_topics != TOPIC_ID_SET:
        raise ValueError("candidate pool hashes must contain the exact topics")
    originals: dict[str, dict[int, str]] = {topic_id: {} for topic_id in TOPIC_IDS}
    facet_ranks: dict[tuple[str, str], dict[int, str]] = {}
    facet_metadata: dict[tuple[str, str], tuple[int, bool]] = {}
    for row in provenance:
        if not isinstance(row, Mapping):
            raise ValueError("candidate provenance row is invalid")
        stage = row.get("provenance_stage")
        if stage == "original_rank":
            topic_id, rank, docid = _ranked_provenance_identity(
                row,
                label="original provenance",
                expected_family="original",
                maximum_rank=100,
            )
            if rank in originals[topic_id] or docid in originals[topic_id].values():
                raise ValueError("original provenance ranks/documents must be unique")
            originals[topic_id][rank] = docid
            continue

        topic_id = _provenance_topic(row, "candidate provenance")
        if stage != "facet_minilm_rank":
            continue
        topic_id, rank, docid = _ranked_provenance_identity(
            row,
            label="facet MiniLM provenance",
            expected_family="facet",
            maximum_rank=50,
        )
        facet_id = row.get("facet_id", row.get("variant"))
        manifest_order = row.get("manifest_order")
        accepted = row.get("accepted")
        if (
            not isinstance(facet_id, str)
            or not facet_id
            or isinstance(manifest_order, bool)
            or not isinstance(manifest_order, int)
            or manifest_order < 0
            or not isinstance(accepted, bool)
        ):
            raise ValueError("facet MiniLM provenance identity/gate is invalid")
        key = (topic_id, facet_id)
        metadata = (manifest_order, accepted)
        if facet_metadata.setdefault(key, metadata) != metadata:
            raise ValueError("facet MiniLM provenance disagrees within a facet")
        ranks = facet_ranks.setdefault(key, {})
        if rank in ranks or docid in ranks.values():
            raise ValueError("facet MiniLM provenance ranks/documents must be unique")
        ranks[rank] = docid

    for topic_id in TOPIC_IDS:
        if set(originals[topic_id]) != set(range(1, 101)):
            raise ValueError(f"original provenance rank coverage is incomplete for {topic_id}")
        topic_facets = [key for key in facet_ranks if key[0] == topic_id]
        if not topic_facets:
            raise ValueError(f"facet MiniLM provenance is incomplete for {topic_id}")
        orders = [facet_metadata[key][0] for key in topic_facets]
        if len(orders) != len(set(orders)):
            raise ValueError(f"facet manifest orders are duplicated for {topic_id}")
        for key in topic_facets:
            if set(facet_ranks[key]) != set(range(1, 51)):
                raise ValueError(
                    f"facet MiniLM rank coverage is incomplete for {key[0]}/{key[1]}"
                )

    gate_metadata: dict[tuple[str, str], tuple[int, bool]] = {}
    for gate in gates:
        if not isinstance(gate, Mapping):
            raise ValueError("gate row is invalid")
        topic_id = _provenance_topic(gate, "gate diagnostics")
        facet_id = gate.get("facet_id")
        manifest_order = gate.get("manifest_order")
        accepted = gate.get("accepted")
        if (
            not isinstance(facet_id, str)
            or not facet_id
            or isinstance(manifest_order, bool)
            or not isinstance(manifest_order, int)
            or manifest_order < 0
            or not isinstance(accepted, bool)
        ):
            raise ValueError("gate identity/disposition is invalid")
        key = (topic_id, facet_id)
        if key in gate_metadata:
            raise ValueError("gate diagnostics contain a duplicate facet")
        gate_metadata[key] = (manifest_order, accepted)
    if gate_metadata != facet_metadata:
        raise ValueError("gate diagnostics disagree with facet MiniLM provenance")

    prefusion: dict[str, list[str]] = {topic_id: [] for topic_id in TOPIC_IDS}
    for topic_id in TOPIC_IDS:
        pool = set(originals[topic_id].values())
        for key, ranks in facet_ranks.items():
            if key[0] != topic_id or not facet_metadata[key][1]:
                continue
            pool.update(ranks.values())
            prefusion[topic_id].extend(
                ranks[rank] for rank in range(1, PREFUSION_FACET_DEPTH + 1)
            )
        expected_pool_hash = _validated_sha256(
            candidate_pool_sha256.get(topic_id),
            f"{topic_id} candidate pool hash",
        )
        if _sha256(_canonical_json_bytes(sorted(pool))) != expected_pool_hash:
            raise ValueError(f"candidate pool hash differs for topic {topic_id}")
        prefusion[topic_id] = list(dict.fromkeys(prefusion[topic_id]))
    return prefusion


def load_verified_freeze(
    freeze_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, list[str]]],
    dict[str, list[str]],
    str,
]:
    """Authenticate Task 3 and retain only verified in-memory ranking inputs."""

    root = Path(freeze_path)
    try:
        freeze = dict(verify_ranking_freeze(root))
        freeze_source = (root / "freeze.json").read_bytes()
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("ranking freeze verification failed") from exc
    if (
        freeze.get("complete") is not True
        or freeze.get("qrels_opened") is not False
        or freeze.get("topic_ids") != list(TOPIC_IDS)
        or freeze.get("arms") != list(ARM_NAMES)
        or freeze.get("depth") != 100
    ):
        raise ValueError("ranking freeze header or exact topic order differs")

    rankings: dict[str, dict[str, list[str]]] = {arm: {} for arm in ARM_NAMES}
    for arm in ARM_NAMES:
        relative = f"rankings/{arm}.jsonl"
        rows = _parse_canonical_jsonl(
            _read_bound_artifact(root, freeze, relative), f"{arm} ranking"
        )
        for topic_id in TOPIC_IDS:
            topic_rows = [row for row in rows if row.get("topic_id") == topic_id]
            if [row.get("rank") for row in topic_rows] != list(range(1, 101)):
                raise ValueError(f"ranking freeze ranks are incomplete for {topic_id}/{arm}")
            rankings[arm][topic_id] = [str(row.get("docid", "")) for row in topic_rows]
        if len(rows) != 400:
            raise ValueError(f"ranking freeze has extra rows for {arm}")
    _validate_rankings(rankings)

    provenance = _parse_canonical_jsonl(
        _read_bound_artifact(root, freeze, "candidate_provenance.jsonl"),
        "candidate provenance",
    )
    try:
        gates = json.loads(_read_bound_artifact(root, freeze, "gates.json"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("gate diagnostics are invalid JSON") from exc
    if not isinstance(gates, list) or any(not isinstance(row, dict) for row in gates):
        raise ValueError("gate diagnostics must be a JSON array of objects")
    candidate_pool_hashes = freeze.get("candidate_pool_sha256")
    if not isinstance(candidate_pool_hashes, Mapping):
        raise ValueError("ranking freeze candidate pool hashes are invalid")
    prefusion = validate_prefusion_provenance(
        provenance,
        gates,
        candidate_pool_hashes,
    )

    try:
        verified_again = dict(verify_ranking_freeze(root))
        final_freeze_source = (root / "freeze.json").read_bytes()
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("ranking freeze changed during verification") from exc
    if verified_again != freeze or final_freeze_source != freeze_source:
        raise ValueError("ranking freeze changed during verification")
    return freeze, rankings, prefusion, _sha256(freeze_source)


def _create_receipt(
    path: Path, *, identity: Mapping[str, str], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = add_self_hash(
        {
            "schema_version": "facet-aware-fusion-qrels-access-receipt-v1",
            "status": "qrels_access_consumed",
            "experiment_id": EXPERIMENT_ID,
            "topic_ids": list(TOPIC_IDS),
            "identity_sha256": identity["identity_sha256"],
            "qrels_approval_sha256": authorization["approval_sha256"],
            "qrels_manifest_sha256": authorization["manifest_sha256"],
            "qrels_projection_sha256": authorization["projection_sha256"],
            "ranking_freeze_sha256": authorization["ranking_freeze_sha256"],
        }
    )
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(_canonical_json_bytes(receipt, pretty=True).decode("utf-8"))
    except FileExistsError as exc:
        raise FileExistsError("repeat qrels access is refused") from exc
    return receipt


def _publish_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(_canonical_json_bytes(payload, pretty=True).decode("utf-8"))
    except FileExistsError as exc:
        raise FileExistsError(f"create-only evaluation artifact exists: {path.name}") from exc


def evaluate(
    freeze: Path,
    qrels_path: Path,
    output: Path,
    *,
    qrels_manifest: Path | None = None,
    qrels_approval: Path | None = None,
) -> dict[str, Any]:
    """Verify, consume one qrels authorization, evaluate, and publish atomically."""

    output_path = Path(output)
    if output_path.exists() or os.path.lexists(output_path):
        raise FileExistsError(f"create-only evaluation output exists: {output_path}")
    freeze_payload, rankings, prefusion, freeze_hash = load_verified_freeze(Path(freeze))
    del freeze_payload
    projection = Path(qrels_path)
    if qrels_manifest is not None:
        manifest = Path(qrels_manifest)
    else:
        established_manifest = projection.parent / "manifest.json"
        fallback_manifest = projection.parent / "qrels_manifest.json"
        manifest = (
            established_manifest
            if established_manifest.exists() or not fallback_manifest.exists()
            else fallback_manifest
        )
    if qrels_approval is not None:
        approval = Path(qrels_approval)
    elif projection.parent.parent.name == "authorized_inputs":
        approval = projection.parent.parent.parent / "approvals/qrels_access_v1.json"
    else:
        approval = projection.parent / "qrels_approval.json"
    authorization = validate_qrels_authorization(
        projection,
        manifest,
        approval,
        ranking_freeze_sha256=freeze_hash,
    )
    identity = qrels_consumption_identity(approval)
    if any(
        identity.get(identity_field) != authorization.get(auth_field)
        for identity_field, auth_field in (
            ("qrels_approval_sha256", "approval_sha256"),
            ("qrels_manifest_sha256", "manifest_sha256"),
            ("qrels_projection_sha256", "projection_sha256"),
            ("ranking_freeze_sha256", "ranking_freeze_sha256"),
        )
    ):
        raise ValueError("qrels consumption identity differs from authorization")
    registry_path = (
        trusted_qrels_consumption_dir() / f"{identity['identity_sha256']}.json"
    )
    if registry_path.exists() or registry_path.is_symlink():
        raise FileExistsError("qrels approval was already consumed")
    receipt_path = output_path / "qrels_access_receipt.json"
    registry = _create_registry(
        registry_path,
        {
            "schema_version": "facet-aware-fusion-qrels-consumption-v1",
            "status": "qrels_access_consumed",
            "registry_namespace": "git_common_dir_path_independent_identity",
            "experiment_id": EXPERIMENT_ID,
            "topic_ids": list(TOPIC_IDS),
            **identity,
            "canonical_output_path": str(output_path.resolve()),
            "output_receipt_path": str(receipt_path.resolve()),
        },
    )
    output_path.mkdir(parents=True, exist_ok=False)
    receipt = _create_receipt(
        receipt_path, identity=identity, authorization=authorization
    )
    qrels = read_qrels(
        projection,
        expected_sha256=str(authorization["projection_sha256"]),
        authorized_directory=Path(authorization["authorized_directory"]),
        authorized_directory_identity=authorization[
            "authorized_directory_identity"
        ],
        authorized_projection_name=str(authorization["projection_name"]),
    )
    evaluation = build_evaluation(rankings, qrels, prefusion)
    decision = decide(evaluation)
    bindings = {
        "ranking_freeze_sha256": freeze_hash,
        "qrels_approval_sha256": authorization["approval_sha256"],
        "qrels_manifest_sha256": authorization["manifest_sha256"],
        "qrels_projection_sha256": authorization["projection_sha256"],
    }
    artifacts = {
        "metrics.json": add_self_hash(
            {
                "schema_version": "facet-aware-fusion-metrics-v1",
                "bindings": bindings,
                "topic_ids": list(TOPIC_IDS),
                "systems": evaluation["systems"],
            }
        ),
        "gains_losses.json": add_self_hash(
            {
                "schema_version": "facet-aware-fusion-gains-losses-v1",
                "bindings": bindings,
                "topic_ids": list(TOPIC_IDS),
                "comparisons": evaluation["comparisons"],
            }
        ),
        "decision.json": add_self_hash(
            {
                "schema_version": "facet-aware-fusion-decision-v1",
                "bindings": bindings,
                "topic_ids": list(TOPIC_IDS),
                "decision": decision,
            }
        ),
    }
    for name in EVALUATION_ARTIFACT_NAMES:
        _publish_artifact(output_path / name, artifacts[name])
    return {
        **evaluation,
        "decision": decision,
        "qrels_access_receipt": receipt,
        "qrels_consumption_registry": registry,
        "artifacts": artifacts,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--freeze", type=Path, required=True)
    evaluate_parser.add_argument(
        "--qrels", type=Path, default=_DEFAULT_QRELS_PATH
    )
    evaluate_parser.add_argument(
        "--qrels-manifest", type=Path, default=_DEFAULT_QRELS_MANIFEST
    )
    evaluate_parser.add_argument(
        "--qrels-approval", type=Path, default=_DEFAULT_QRELS_APPROVAL
    )
    evaluate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    result = evaluate(
        args.freeze,
        args.qrels,
        args.output,
        qrels_manifest=args.qrels_manifest,
        qrels_approval=args.qrels_approval,
    )
    print(
        json.dumps(
            {
                "promoted_arm": result["decision"]["promoted_arm"],
                "status": "evaluation_complete",
                "topic_ids": list(TOPIC_IDS),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
