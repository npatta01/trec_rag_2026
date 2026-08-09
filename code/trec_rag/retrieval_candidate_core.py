"""Deterministic candidate-core selection from authenticated retrieval lane scores."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import median
from types import MappingProxyType
from typing import Mapping


_SCHEMA_VERSION = "retrieval-candidate-core-v1"
_TOPIC_ID = re.compile(r"rag2026-[0-9]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAD_SCALE = 2.5 * 1.4826


def _docid_digest(docids: tuple[str, ...]) -> str:
    return sha256(b"".join(docid.encode("utf-8") + b"\n" for docid in docids)).hexdigest()


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


@dataclass(frozen=True)
class CandidateLaneStat:
    """Robust admission statistics for one authenticated retrieval lane."""

    lane_name: str
    median: float
    mad: float
    threshold: float
    comparison: str
    observed_count: int
    admitted_count: int
    admitted_docids_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.lane_name, str) or not self.lane_name:
            raise ValueError("lane name must be nonempty text")
        center = _finite_float(self.median, label="lane median")
        deviation = _finite_float(self.mad, label="lane MAD")
        threshold = _finite_float(self.threshold, label="lane threshold")
        if deviation < 0.0:
            raise ValueError("lane MAD must be nonnegative")
        observed = _positive_int(self.observed_count, label="observed count")
        admitted = _nonnegative_int(self.admitted_count, label="admitted count")
        if admitted > observed:
            raise ValueError("admitted count cannot exceed observed count")
        expected_comparison = (
            "greater_than_or_equal" if deviation > 0.0 else "strictly_greater_than"
        )
        expected_threshold = center + _MAD_SCALE * deviation
        if self.comparison != expected_comparison:
            raise ValueError("lane comparison differs from MAD rule")
        if threshold != expected_threshold:
            raise ValueError("lane threshold differs from MAD rule")
        if not isinstance(self.admitted_docids_sha256, str) or _SHA256.fullmatch(
            self.admitted_docids_sha256
        ) is None:
            raise ValueError("admitted-docids SHA-256 must be lowercase hexadecimal")


@dataclass(frozen=True)
class CandidateCore:
    """One authenticated variable-depth candidate set shared by all baselines."""

    topic_id: str
    lane_scores_sha256: str
    candidate_docids: tuple[str, ...]
    pre_fallback_count: int
    fallback_used: bool
    admission_multiplicity_histogram: Mapping[str, int]
    lanes: tuple[CandidateLaneStat, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.topic_id, str) or _TOPIC_ID.fullmatch(self.topic_id) is None:
            raise ValueError("candidate-core topic ID is invalid")
        if not isinstance(self.lane_scores_sha256, str) or _SHA256.fullmatch(
            self.lane_scores_sha256
        ) is None:
            raise ValueError("lane-score SHA-256 must be lowercase hexadecimal")
        if not isinstance(self.candidate_docids, tuple) or not self.candidate_docids:
            raise ValueError("candidate docids must be a nonempty tuple")
        if any(not isinstance(docid, str) or not docid for docid in self.candidate_docids):
            raise ValueError("candidate docids must be nonempty text")
        expected_order = tuple(
            sorted(set(self.candidate_docids), key=lambda value: value.encode("utf-8"))
        )
        if self.candidate_docids != expected_order:
            raise ValueError("candidate docids must be unique and UTF-8 bytewise sorted")
        pre_fallback = _nonnegative_int(
            self.pre_fallback_count, label="pre-fallback count"
        )
        if not isinstance(self.fallback_used, bool):
            raise TypeError("fallback_used must be boolean")
        if not isinstance(self.lanes, tuple) or not self.lanes:
            raise ValueError("candidate core must contain lane statistics")
        if any(not isinstance(row, CandidateLaneStat) for row in self.lanes):
            raise TypeError("candidate-core lanes must be CandidateLaneStat values")
        lane_names = tuple(row.lane_name for row in self.lanes)
        if len(set(lane_names)) != len(lane_names) or lane_names[0] != "original":
            raise ValueError("candidate-core lanes must be unique with original first")

        histogram: dict[str, int] = {}
        if not isinstance(self.admission_multiplicity_histogram, Mapping):
            raise TypeError("admission multiplicity histogram must be a mapping")
        for raw_key, raw_count in self.admission_multiplicity_histogram.items():
            if (
                not isinstance(raw_key, str)
                or not raw_key.isdigit()
                or int(raw_key) <= 0
            ):
                raise ValueError("admission multiplicity keys must be positive integers")
            histogram[raw_key] = _positive_int(
                raw_count, label="admission multiplicity count"
            )
        histogram = dict(sorted(histogram.items(), key=lambda row: int(row[0])))
        object.__setattr__(
            self, "admission_multiplicity_histogram", MappingProxyType(histogram)
        )

        admitted_total = sum(row.admitted_count for row in self.lanes)
        histogram_docs = sum(histogram.values())
        histogram_admissions = sum(int(key) * value for key, value in histogram.items())
        if self.fallback_used:
            if (
                pre_fallback != 0
                or len(self.candidate_docids) != 1
                or admitted_total != 0
                or histogram
            ):
                raise ValueError("fallback state is inconsistent")
        elif (
            pre_fallback == 0
            or len(self.candidate_docids) != pre_fallback
            or histogram_docs != pre_fallback
            or histogram_admissions != admitted_total
        ):
            raise ValueError("non-fallback admission state is inconsistent")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def derive_candidate_core(
    *,
    topic_id: str,
    lane_scores_path: Path,
    expected_lane_names: tuple[str, ...],
    expected_docids: frozenset[str],
    best_retrieval_ranks: Mapping[str, int],
) -> CandidateCore:
    """Robust-threshold each source lane and return their deterministic union."""

    if not isinstance(topic_id, str) or _TOPIC_ID.fullmatch(topic_id) is None:
        raise ValueError("topic_id must match the official rag2026-N pattern")
    if (
        not isinstance(expected_lane_names, tuple)
        or not expected_lane_names
        or expected_lane_names[0] != "original"
        or len(set(expected_lane_names)) != len(expected_lane_names)
        or any(not isinstance(name, str) or not name for name in expected_lane_names)
    ):
        raise ValueError("expected lane names must be unique with original first")
    if not isinstance(expected_docids, frozenset) or not expected_docids:
        raise ValueError("expected docids must be a nonempty frozenset")
    if any(not isinstance(docid, str) or not docid for docid in expected_docids):
        raise ValueError("expected docids must be nonempty text")
    if set(best_retrieval_ranks) != set(expected_docids):
        raise ValueError("retrieval-rank map must exactly cover expected documents")
    ranks: dict[str, int] = {}
    for docid, rank in best_retrieval_ranks.items():
        ranks[docid] = _positive_int(rank, label="best retrieval rank")

    path = Path(lane_scores_path)
    body = path.read_bytes()
    if not body:
        raise ValueError("lane-score file must be nonempty")
    scores_by_lane: dict[str, dict[str, float]] = {
        lane_name: {} for lane_name in expected_lane_names
    }
    seen_lanes: set[str] = set()
    for line_number, line in enumerate(body.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"lane-score line {line_number} is empty")
        try:
            value = json.loads(line, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"lane-score line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"lane-score line {line_number} must be an object")
        if value.get("topic_id") != topic_id:
            raise ValueError("lane-score topic identity differs")
        lane_name = _required_text(value.get("lane_name"), label="lane name")
        docid = _required_text(value.get("docid"), label="document ID")
        if lane_name not in scores_by_lane:
            seen_lanes.add(lane_name)
            continue
        if docid not in expected_docids:
            raise ValueError("lane score refers to a document outside the expected union")
        if docid in scores_by_lane[lane_name]:
            raise ValueError("duplicate lane/document score")
        score = _finite_float(value.get("aggregate_score"), label="aggregate_score")
        scores_by_lane[lane_name][docid] = score
        seen_lanes.add(lane_name)
    if seen_lanes != set(expected_lane_names) or any(
        not scores_by_lane[name] for name in expected_lane_names
    ):
        raise ValueError("lane scores do not contain the exact authenticated lane set")

    admitted_by_lane: dict[str, tuple[str, ...]] = {}
    lane_stats: list[CandidateLaneStat] = []
    admission_count_by_docid: dict[str, int] = {}
    for lane_name in expected_lane_names:
        scores = scores_by_lane[lane_name]
        center = float(median(scores.values()))
        deviation = float(median(abs(score - center) for score in scores.values()))
        if deviation > 0.0:
            threshold = center + _MAD_SCALE * deviation
            comparison = "greater_than_or_equal"
            admitted = tuple(
                sorted(
                    (docid for docid, score in scores.items() if score >= threshold),
                    key=lambda value: value.encode("utf-8"),
                )
            )
        else:
            threshold = center
            comparison = "strictly_greater_than"
            admitted = tuple(
                sorted(
                    (docid for docid, score in scores.items() if score > center),
                    key=lambda value: value.encode("utf-8"),
                )
            )
        admitted_by_lane[lane_name] = admitted
        for docid in admitted:
            admission_count_by_docid[docid] = admission_count_by_docid.get(docid, 0) + 1
        lane_stats.append(
            CandidateLaneStat(
                lane_name=lane_name,
                median=center,
                mad=deviation,
                threshold=threshold,
                comparison=comparison,
                observed_count=len(scores),
                admitted_count=len(admitted),
                admitted_docids_sha256=_docid_digest(admitted),
            )
        )

    candidate_set = set(admission_count_by_docid)
    pre_fallback_count = len(candidate_set)
    fallback_used = not candidate_set
    if fallback_used:
        original_scores = scores_by_lane["original"]
        candidate_set.add(
            min(
                original_scores,
                key=lambda docid: (
                    -original_scores[docid],
                    ranks[docid],
                    docid.encode("utf-8"),
                ),
            )
        )

    histogram: dict[str, int] = {}
    for count in admission_count_by_docid.values():
        key = str(count)
        histogram[key] = histogram.get(key, 0) + 1
    return CandidateCore(
        topic_id=topic_id,
        lane_scores_sha256=sha256(body).hexdigest(),
        candidate_docids=tuple(
            sorted(candidate_set, key=lambda value: value.encode("utf-8"))
        ),
        pre_fallback_count=pre_fallback_count,
        fallback_used=fallback_used,
        admission_multiplicity_histogram=histogram,
        lanes=tuple(lane_stats),
    )


def candidate_core_to_dict(core: CandidateCore) -> dict[str, object]:
    """Return the canonical JSON-compatible candidate-core record."""

    if not isinstance(core, CandidateCore):
        raise TypeError("core must be a CandidateCore")
    return {
        "schema_version": _SCHEMA_VERSION,
        "topic_id": core.topic_id,
        "lane_scores_sha256": core.lane_scores_sha256,
        "candidate_docids": list(core.candidate_docids),
        "pre_fallback_count": core.pre_fallback_count,
        "fallback_used": core.fallback_used,
        "admission_multiplicity_histogram": dict(
            core.admission_multiplicity_histogram
        ),
        "lanes": [
            {
                "lane_name": row.lane_name,
                "median": row.median,
                "mad": row.mad,
                "threshold": row.threshold,
                "comparison": row.comparison,
                "observed_count": row.observed_count,
                "admitted_count": row.admitted_count,
                "admitted_docids_sha256": row.admitted_docids_sha256,
            }
            for row in core.lanes
        ],
    }


def candidate_core_from_dict(value: Mapping[str, object]) -> CandidateCore:
    """Parse and validate one canonical candidate-core record."""

    if not isinstance(value, Mapping):
        raise TypeError("candidate core must be a mapping")
    expected_keys = {
        "schema_version",
        "topic_id",
        "lane_scores_sha256",
        "candidate_docids",
        "pre_fallback_count",
        "fallback_used",
        "admission_multiplicity_histogram",
        "lanes",
    }
    if set(value) != expected_keys or value.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("candidate-core schema differs")
    raw_docids = value.get("candidate_docids")
    raw_lanes = value.get("lanes")
    raw_histogram = value.get("admission_multiplicity_histogram")
    if not isinstance(raw_docids, list):
        raise TypeError("candidate docids must be a list")
    if not isinstance(raw_lanes, list):
        raise TypeError("candidate-core lanes must be a list")
    if not isinstance(raw_histogram, Mapping):
        raise TypeError("admission multiplicity histogram must be a mapping")
    lanes: list[CandidateLaneStat] = []
    lane_keys = {
        "lane_name",
        "median",
        "mad",
        "threshold",
        "comparison",
        "observed_count",
        "admitted_count",
        "admitted_docids_sha256",
    }
    for raw in raw_lanes:
        if not isinstance(raw, Mapping) or set(raw) != lane_keys:
            raise ValueError("candidate lane schema differs")
        lanes.append(
            CandidateLaneStat(
                lane_name=raw.get("lane_name"),  # type: ignore[arg-type]
                median=raw.get("median"),  # type: ignore[arg-type]
                mad=raw.get("mad"),  # type: ignore[arg-type]
                threshold=raw.get("threshold"),  # type: ignore[arg-type]
                comparison=raw.get("comparison"),  # type: ignore[arg-type]
                observed_count=raw.get("observed_count"),  # type: ignore[arg-type]
                admitted_count=raw.get("admitted_count"),  # type: ignore[arg-type]
                admitted_docids_sha256=raw.get(  # type: ignore[arg-type]
                    "admitted_docids_sha256"
                ),
            )
        )
    return CandidateCore(
        topic_id=value.get("topic_id"),  # type: ignore[arg-type]
        lane_scores_sha256=value.get("lane_scores_sha256"),  # type: ignore[arg-type]
        candidate_docids=tuple(raw_docids),  # type: ignore[arg-type]
        pre_fallback_count=value.get("pre_fallback_count"),  # type: ignore[arg-type]
        fallback_used=value.get("fallback_used"),  # type: ignore[arg-type]
        admission_multiplicity_histogram=dict(raw_histogram),  # type: ignore[arg-type]
        lanes=tuple(lanes),
    )
