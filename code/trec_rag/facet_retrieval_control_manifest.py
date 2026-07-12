"""Frozen four-stream manifest for the facet retrieval-control pilot."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from trec_rag.sparse_relevance_manifest import (
    ManifestValidationError,
    load_repair_manifest,
)


SCHEMA_VERSION = "facet-retrieval-control-manifest-v1"
R1_MANIFEST_SHA256 = (
    "769388cd828167667657549ca86137f1c7eb774a0ab6ee7ad211769fdae9115a"
)
R1_PLANNER_VERSION = "coverage-first-bm25-v4"
ANALYZER_FINGERPRINT_SHA256 = (
    "f9bbd4e7af26c532105f6dd7e49ce15fa11afd1f0fe7d387847ce41ff7d8def4"
)
PROTECTED_TOPIC_IDS = ("144", "213", "224", "407", "515")
HITS = 100
MIN_INTERVAL_SECONDS = 10.0
MAX_EXTERNAL_REQUESTS = 12

_DEFAULT_R1_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "experiments"
    / "sparse_relevance_pilot_v1"
    / "r1_manifest.json"
)
_QUERY_SPEC = {
    ("200", "f07a"): (
        "Holocaust enduring effects European Jews",
        "Holocaust Holocaust enduring effects European European Jews Jews",
    ),
    ("225", "f02"): (
        "violent video games exposure desensitization violence research",
        "violent video games violent video games exposure desensitization violence research",
    ),
    ("225", "f04"): (
        "aggressive behavior children risk factors psychology",
        "aggressive aggressive behavior children children risk factors psychology",
    ),
    ("707", "f02"): (
        "sorbitol human health adverse effects safety",
        "sorbitol sorbitol human human health adverse effects safety",
    ),
}
_ARM_SPEC = (
    ("B0", 0.9, 0.4, False),
    ("W0", 0.9, 0.4, True),
    ("W1", 0.4, 0.4, True),
    ("W2", 0.4, 0.0, True),
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class ControlManifestError(ValueError):
    """The facet retrieval-control manifest violates its frozen contract."""


@dataclass(frozen=True)
class ControlArm:
    arm_id: str
    k1: float
    b: float
    external: bool


@dataclass(frozen=True)
class ControlStream:
    topic_id: str
    stream_id: str
    baseline_query: str
    reweighted_query: str
    arms: tuple[ControlArm, ...]


@dataclass(frozen=True)
class ControlManifest:
    r1_manifest_sha256: str
    r1_planner_version: str
    analyzer_fingerprint_sha256: str
    protected_topic_ids: tuple[str, ...]
    hits: int
    min_interval_seconds: float
    max_external_requests: int
    streams: tuple[ControlStream, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-ready representation."""

        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _analyzed_vocabulary(text: str) -> frozenset[str]:
    """Return normalized surfaces; repetition cannot introduce analyzer terms."""

    return frozenset(_TOKEN_RE.findall(text.lower()))


def _frozen_arms() -> tuple[ControlArm, ...]:
    return tuple(ControlArm(*values) for values in _ARM_SPEC)


def _validate_manifest(manifest: ControlManifest) -> ControlManifest:
    if manifest.r1_manifest_sha256 != R1_MANIFEST_SHA256:
        raise ControlManifestError("R1 manifest SHA-256 differs from frozen source")
    if manifest.r1_planner_version != R1_PLANNER_VERSION:
        raise ControlManifestError("R1 planner version differs from frozen source")
    if manifest.analyzer_fingerprint_sha256 != ANALYZER_FINGERPRINT_SHA256:
        raise ControlManifestError("analyzer fingerprint differs from frozen source")
    if manifest.protected_topic_ids != PROTECTED_TOPIC_IDS:
        raise ControlManifestError("protected topic namespace differs from frozen set")
    if isinstance(manifest.hits, bool) or manifest.hits != HITS:
        raise ControlManifestError("hits must be frozen at 100")
    if (
        isinstance(manifest.min_interval_seconds, bool)
        or manifest.min_interval_seconds != MIN_INTERVAL_SECONDS
    ):
        raise ControlManifestError("minimum interval must be frozen at ten seconds")
    if (
        isinstance(manifest.max_external_requests, bool)
        or manifest.max_external_requests != MAX_EXTERNAL_REQUESTS
    ):
        raise ControlManifestError("maximum external requests must be frozen at 12")

    boundaries = tuple((stream.topic_id, stream.stream_id) for stream in manifest.streams)
    if len(set(boundaries)) != len(boundaries):
        raise ControlManifestError("duplicate control stream boundary")
    if any(stream.topic_id in PROTECTED_TOPIC_IDS for stream in manifest.streams):
        topic_id = next(
            stream.topic_id
            for stream in manifest.streams
            if stream.topic_id in PROTECTED_TOPIC_IDS
        )
        raise ControlManifestError(f"protected topic {topic_id} is forbidden")
    if len(manifest.streams) != 4:
        raise ControlManifestError("control manifest must contain exactly four streams")
    if boundaries != tuple(_QUERY_SPEC):
        raise ControlManifestError("control streams differ from exact R1 baseline namespace")

    expected_arms = _frozen_arms()
    for stream in manifest.streams:
        expected_baseline, expected_reweighted = _QUERY_SPEC[
            (stream.topic_id, stream.stream_id)
        ]
        if (
            stream.baseline_query != expected_baseline
            or stream.reweighted_query != expected_reweighted
        ):
            raise ControlManifestError(
                "control queries differ from exact R1 baseline namespace"
            )
        if stream.arms != expected_arms:
            raise ControlManifestError(
                f"stream {stream.topic_id}/{stream.stream_id} has invalid arms"
            )
        if not _analyzed_vocabulary(stream.reweighted_query) <= _analyzed_vocabulary(
            stream.baseline_query
        ):
            raise ControlManifestError(
                f"stream {stream.topic_id}/{stream.stream_id} adds analyzed vocabulary"
            )

    external = sum(
        arm.external for stream in manifest.streams for arm in stream.arms
    )
    if external != MAX_EXTERNAL_REQUESTS:
        raise ControlManifestError("control manifest must contain 12 external arms")
    return manifest


def build_control_manifest(r1_path: Path | None = None) -> ControlManifest:
    """Build the frozen control manifest from the exact prior R1 manifest."""

    source_path = Path(r1_path) if r1_path is not None else _DEFAULT_R1_PATH
    if _sha256(source_path) != R1_MANIFEST_SHA256:
        raise ControlManifestError("input differs from exact R1 baseline namespace")
    try:
        r1 = load_repair_manifest(source_path)
    except (ManifestValidationError, json.JSONDecodeError) as exc:
        raise ControlManifestError(
            f"input differs from exact R1 baseline namespace: {exc}"
        ) from exc
    if (
        r1.renderer != "R1"
        or r1.planner_version != R1_PLANNER_VERSION
        or r1.analyzer_fingerprint_sha256 != ANALYZER_FINGERPRINT_SHA256
        or r1.protected_topic_ids != PROTECTED_TOPIC_IDS
    ):
        raise ControlManifestError("input differs from exact R1 baseline namespace")

    r1_queries = {
        (stream.topic_id, stream.stream_id): stream.query for stream in r1.streams
    }
    if any(
        r1_queries.get(boundary) != baseline
        for boundary, (baseline, _reweighted) in _QUERY_SPEC.items()
    ):
        raise ControlManifestError("input differs from exact R1 baseline namespace")

    manifest = ControlManifest(
        r1_manifest_sha256=R1_MANIFEST_SHA256,
        r1_planner_version=R1_PLANNER_VERSION,
        analyzer_fingerprint_sha256=ANALYZER_FINGERPRINT_SHA256,
        protected_topic_ids=PROTECTED_TOPIC_IDS,
        hits=HITS,
        min_interval_seconds=MIN_INTERVAL_SECONDS,
        max_external_requests=MAX_EXTERNAL_REQUESTS,
        streams=tuple(
            ControlStream(
                topic_id=topic_id,
                stream_id=stream_id,
                baseline_query=baseline,
                reweighted_query=reweighted,
                arms=_frozen_arms(),
            )
            for (topic_id, stream_id), (baseline, reweighted) in _QUERY_SPEC.items()
        ),
    )
    return _validate_manifest(manifest)


def _require_object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ControlManifestError(f"{field} must be an object")
    return value


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ControlManifestError(f"{field} must be non-empty text")
    return value


def _require_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlManifestError(f"{field} must be a number")
    return float(value)


def _parse_arm(value: object, index: int) -> ControlArm:
    row = _require_object(value, f"arms[{index}]")
    external = row.get("external")
    if not isinstance(external, bool):
        raise ControlManifestError(f"arms[{index}].external must be boolean")
    return ControlArm(
        arm_id=_require_text(row.get("arm_id"), f"arms[{index}].arm_id"),
        k1=_require_number(row.get("k1"), f"arms[{index}].k1"),
        b=_require_number(row.get("b"), f"arms[{index}].b"),
        external=external,
    )


def _parse_stream(value: object, index: int) -> ControlStream:
    row = _require_object(value, f"streams[{index}]")
    raw_arms = row.get("arms")
    if not isinstance(raw_arms, list):
        raise ControlManifestError(f"streams[{index}].arms must be a list")
    return ControlStream(
        topic_id=_require_text(row.get("topic_id"), f"streams[{index}].topic_id"),
        stream_id=_require_text(
            row.get("stream_id"), f"streams[{index}].stream_id"
        ),
        baseline_query=_require_text(
            row.get("baseline_query"), f"streams[{index}].baseline_query"
        ),
        reweighted_query=_require_text(
            row.get("reweighted_query"), f"streams[{index}].reweighted_query"
        ),
        arms=tuple(_parse_arm(arm, arm_index) for arm_index, arm in enumerate(raw_arms)),
    )


def load_control_manifest(path: Path) -> ControlManifest:
    """Load and fully validate a frozen control manifest JSON file."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlManifestError(f"cannot load control manifest: {exc}") from exc
    row = _require_object(payload, "manifest")
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ControlManifestError(f"schema_version must be {SCHEMA_VERSION}")
    raw_protected = row.get("protected_topic_ids")
    if not isinstance(raw_protected, list):
        raise ControlManifestError("protected_topic_ids must be a list")
    raw_streams = row.get("streams")
    if not isinstance(raw_streams, list):
        raise ControlManifestError("streams must be a list")
    hits = row.get("hits")
    max_external_requests = row.get("max_external_requests")
    if isinstance(hits, bool) or not isinstance(hits, int):
        raise ControlManifestError("hits must be an integer")
    if isinstance(max_external_requests, bool) or not isinstance(
        max_external_requests, int
    ):
        raise ControlManifestError("max_external_requests must be an integer")
    manifest = ControlManifest(
        r1_manifest_sha256=_require_text(
            row.get("r1_manifest_sha256"), "r1_manifest_sha256"
        ),
        r1_planner_version=_require_text(
            row.get("r1_planner_version"), "r1_planner_version"
        ),
        analyzer_fingerprint_sha256=_require_text(
            row.get("analyzer_fingerprint_sha256"),
            "analyzer_fingerprint_sha256",
        ),
        protected_topic_ids=tuple(
            _require_text(value, "protected_topic_ids") for value in raw_protected
        ),
        hits=hits,
        min_interval_seconds=_require_number(
            row.get("min_interval_seconds"), "min_interval_seconds"
        ),
        max_external_requests=max_external_requests,
        streams=tuple(
            _parse_stream(stream, index) for index, stream in enumerate(raw_streams)
        ),
    )
    return _validate_manifest(manifest)
