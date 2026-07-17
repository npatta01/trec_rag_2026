"""Immutable 31-stream source manifest for the facet-local MiniLM pilot.

This module is the only boundary that may reopen the two verified retrieval
ledgers.  It copies their exact candidates into a self-contained snapshot;
downstream stages consume that snapshot and its receipt instead.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .det_sparse_ledger import (
    RetrievalLedger,
    RetrievalRequest,
    RetrievalResult,
)
from .repo_env import repo_cache_root


SCHEMA_VERSION = "facet-local-minilm-manifest-v1"
SOURCE_RECEIPT_SCHEMA_VERSION = "facet-local-minilm-source-receipt-v1"
CANDIDATE_ROW_SCHEMA_VERSION = "facet-local-minilm-candidate-row-v1"
R1_MANIFEST_SHA256 = (
    "769388cd828167667657549ca86137f1c7eb774a0ab6ee7ad211769fdae9115a"
)
PRIOR_FREEZE_SHA256 = (
    "4a78b44ede4b979a3cb3ec96348088e4e08626e2cc4c92b464c6e36097a71389"
)
ANALYZER_FINGERPRINT_SHA256 = (
    "f9bbd4e7af26c532105f6dd7e49ce15fa11afd1f0fe7d387847ce41ff7d8def4"
)
PROTECTED_TOPIC_IDS = ("144", "213", "224", "407", "515")
PILOT_TOPIC_IDS = ("200", "225", "707", "897")
ORIGINAL_VARIANT = "prompt_lab_v1:original"
RETRIEVER_VERSION = "pyserini_remote_raw_first_v1"
INDEX_URL = "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
INDEX_ID = "climbmix-400b"
EXPECTED_DEPTH = 100

# This is deliberately copied into this new contract.  Importing the historical
# experiment implementation would make the frozen namespace mutable by accident.
KEPT_BASE_FACETS = frozenset(
    {
        ("225", "prompt_lab_v1:facet:f06"),
        ("225", "prompt_lab_v1:facet:f07"),
        ("707", "prompt_lab_v1:facet:f01"),
        ("707", "prompt_lab_v1:facet:f03"),
        ("897", "prompt_lab_v1:facet:f01"),
    }
)

_BASE_SPECS = (
    (
        "200",
        ORIGINAL_VARIANT,
        "I want to deeply understand the Holocaust: what it was, why and how it "
        "transpired, who was responsible, and its profound historical and societal "
        "impact, particularly on European Jewry. I'm also curious about its "
        "conclusion, lasting effects, and how it aligns with other destructive "
        "historical events like Sodom and Gomorrah.",
    ),
    (
        "225",
        ORIGINAL_VARIANT,
        "I'm exploring how exposure to violent video games and gory content affects "
        "human aggression, desensitization, and addiction. I'm also interested in "
        "the broader causes of aggression in both children and adults, the positive "
        "sides of gaming, and the historical background of these media trends.",
    ),
    ("225", "prompt_lab_v1:facet:f06", "video games benefits"),
    (
        "225",
        "prompt_lab_v1:facet:f07",
        "violent video games gory content history",
    ),
    (
        "707",
        ORIGINAL_VARIANT,
        "I'm trying to understand the health risks and potential dangers of various "
        "chemicals and substances, such as those found in antiperspirants, sorbitol, "
        "and organophosphate poisoning. I'd also like to know how integrating "
        "different actions at the operational level might impact health outcomes.",
    ),
    ("707", "prompt_lab_v1:facet:f01", "antiperspirants health risks"),
    (
        "707",
        "prompt_lab_v1:facet:f03",
        "organophosphate poisoning health risks",
    ),
    (
        "897",
        ORIGINAL_VARIANT,
        "I'm trying to understand how alcohol use affects neighborhood quality of "
        "life, including genetic and gender factors in dependency. I also need to "
        "grasp the main causes, risks, and fatal consequences of alcohol and drug "
        "use, and their connection to broader community health issues.",
    ),
    (
        "897",
        "prompt_lab_v1:facet:f01",
        "alcohol use neighborhood quality life",
    ),
)

_R1_SPECS = (
    ("200", "f01", "Holocaust history European Jewry overview"),
    ("200", "f02", "Holocaust historical origins why transpired"),
    ("200", "f03", "Holocaust systematic process how transpired"),
    ("200", "f04", "Holocaust perpetrators responsibility"),
    ("200", "f05a", "Holocaust historical consequences European Jews"),
    ("200", "f05b", "Holocaust social consequences"),
    ("200", "f06", "Holocaust historical ending"),
    ("200", "f07a", "Holocaust enduring effects European Jews"),
    ("200", "f07b", "Holocaust enduring social consequences"),
    (
        "225",
        "f01",
        "violent video games exposure aggressive behavior research",
    ),
    (
        "225",
        "f02",
        "violent video games exposure desensitization violence research",
    ),
    ("225", "f03", "video gaming addiction disorder risk"),
    ("225", "f04", "aggressive behavior children risk factors psychology"),
    ("225", "f05", "aggressive behavior adults risk factors psychology"),
    ("707", "f02", "sorbitol human health adverse effects safety"),
    ("897", "f02a", "alcohol dependence genetic risk factors"),
    ("897", "f02b", "alcohol dependence gender differences"),
    ("897", "f03", "alcohol drug misuse addiction risk factors"),
    ("897", "f04a", "alcohol misuse health harms"),
    ("897", "f04b", "drug misuse health harms"),
    ("897", "f05", "alcohol drug mortality deaths"),
    (
        "897",
        "f06",
        "alcohol drug misuse public health community impact",
    ),
)

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_ROOT = repo_cache_root(_DEFAULT_REPO_ROOT)
_DEFAULT_R1_MANIFEST = Path(
    "reports/experiments/sparse_relevance_pilot_v1/r1_manifest.json"
)
_DEFAULT_PRIOR_FREEZE = Path("outputs/rag25_sparse_relevance_paired_v1/freeze_v1")
_DEFAULT_BASE_RUN = Path(
    "outputs/rag25_det_sparse_prompt_lab_v1/rate_limited_continuation_v1"
)
_DEFAULT_BASE_CACHE = (
    _DEFAULT_CACHE_ROOT / "retrieval/rag25_det_sparse_prompt_lab_base_http_restart_v1"
)
_DEFAULT_R1_RUN = Path("outputs/rag25_sparse_relevance_paired_v1/run_v2/R1/ledger")
_DEFAULT_R1_CACHE = _DEFAULT_CACHE_ROOT / "retrieval/rag25_sparse_relevance_paired_v1"
_DEFAULT_SOURCE_OUTPUT = Path("outputs/rag25_facet_local_minilm_v1/source_v1")

_STREAM_FIELDS = frozenset(
    {
        "topic_id",
        "family",
        "variant",
        "query",
        "query_sha256",
        "source_kind",
        "source_ledger",
        "source_cache",
        "source_request_sha256",
        "source_response_sha256",
        "source_candidates_sha256",
        "expected_rows",
        "candidates_sha256",
    }
)
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "topic_ids",
        "protected_topic_ids",
        "r1_manifest_sha256",
        "prior_freeze_sha256",
        "base_ledger_sha256",
        "r1_ledger_sha256",
        "candidate_schema_version",
        "candidate_file",
        "stream_count",
        "candidate_rows",
        "candidate_bytes",
        "candidates_sha256",
        "source_receipt_schema_version",
        "source_receipt_file",
        "source_receipt_sha256",
        "streams",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "topic_id",
        "family",
        "variant",
        "query",
        "query_sha256",
        "rank",
        "document_id",
        "text",
        "text_sha256",
        "source_score",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "r1_manifest_sha256",
        "prior_freeze_sha256",
        "base_ledger_sha256",
        "r1_ledger_sha256",
        "candidate_schema_version",
        "candidate_file",
        "stream_count",
        "candidate_rows",
        "candidate_bytes",
        "candidates_sha256",
    }
)


@dataclass(frozen=True)
class FacetLocalStream:
    topic_id: str
    family: str
    variant: str
    query: str
    query_sha256: str
    source_kind: str
    source_ledger: str
    source_cache: str
    source_request_sha256: str
    source_response_sha256: str
    source_candidates_sha256: str
    expected_rows: int
    candidates_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "topic_id": self.topic_id,
            "family": self.family,
            "variant": self.variant,
            "query": self.query,
            "query_sha256": self.query_sha256,
            "source_kind": self.source_kind,
            "source_ledger": self.source_ledger,
            "source_cache": self.source_cache,
            "source_request_sha256": self.source_request_sha256,
            "source_response_sha256": self.source_response_sha256,
            "source_candidates_sha256": self.source_candidates_sha256,
            "expected_rows": self.expected_rows,
            "candidates_sha256": self.candidates_sha256,
        }


@dataclass(frozen=True)
class FacetLocalManifest:
    topic_ids: tuple[str, ...]
    protected_topic_ids: tuple[str, ...]
    r1_manifest_sha256: str
    prior_freeze_sha256: str
    base_ledger_sha256: str
    r1_ledger_sha256: str
    candidate_schema_version: str
    candidate_file: str
    candidate_rows: int
    candidate_bytes: int
    candidates_sha256: str
    source_receipt_schema_version: str
    source_receipt_file: str
    source_receipt_sha256: str
    streams: tuple[FacetLocalStream, ...]

    @property
    def stream_counts(self) -> dict[str, int]:
        counts = Counter(stream.topic_id for stream in self.streams)
        return {topic_id: counts[topic_id] for topic_id in self.topic_ids}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "frozen_source_snapshot",
            "topic_ids": list(self.topic_ids),
            "protected_topic_ids": list(self.protected_topic_ids),
            "r1_manifest_sha256": self.r1_manifest_sha256,
            "prior_freeze_sha256": self.prior_freeze_sha256,
            "base_ledger_sha256": self.base_ledger_sha256,
            "r1_ledger_sha256": self.r1_ledger_sha256,
            "candidate_schema_version": self.candidate_schema_version,
            "candidate_file": self.candidate_file,
            "stream_count": len(self.streams),
            "candidate_rows": self.candidate_rows,
            "candidate_bytes": self.candidate_bytes,
            "candidates_sha256": self.candidates_sha256,
            "source_receipt_schema_version": self.source_receipt_schema_version,
            "source_receipt_file": self.source_receipt_file,
            "source_receipt_sha256": self.source_receipt_sha256,
            "streams": [stream.to_dict() for stream in self.streams],
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class _LoadedSource:
    source_kind: str
    source_ledger: str
    source_cache: str
    request: RetrievalRequest
    result: RetrievalResult


@dataclass(frozen=True)
class _LoadedSources:
    streams: tuple[_LoadedSource, ...]
    base_ledger_sha256: str
    r1_ledger_sha256: str


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        json.dumps(
            row, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _reject_protected(topic_ids: Sequence[str]) -> None:
    for topic_id in topic_ids:
        if str(topic_id) in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")


def _resolve_input(path: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return _DEFAULT_REPO_ROOT / value


def _verify_exact_source_paths(
    *, base_run: Path, base_cache: Path, r1_run: Path, r1_cache: Path
) -> None:
    supplied = (base_run, base_cache, r1_run, r1_cache)
    expected = (
        _DEFAULT_BASE_RUN,
        _DEFAULT_BASE_CACHE,
        _DEFAULT_R1_RUN,
        _DEFAULT_R1_CACHE,
    )
    if any(
        _resolve_input(actual).resolve() != _resolve_input(frozen).resolve()
        for actual, frozen in zip(supplied, expected, strict=True)
    ):
        raise ValueError("source ledger/cache paths differ from the frozen inputs")


def _expected_queries() -> dict[tuple[str, str], str]:
    rows = {(topic_id, variant): query for topic_id, variant, query in _BASE_SPECS}
    rows.update(
        {
            (topic_id, f"sparse_relevance_v1:R1:{stream_id}"): query
            for topic_id, stream_id, query in _R1_SPECS
        }
    )
    return rows


def _canonical_stream_keys() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            _expected_queries(),
            key=lambda key: (
                PILOT_TOPIC_IDS.index(key[0]),
                key[1] != ORIGINAL_VARIANT,
                not key[1].startswith("prompt_lab_v1:"),
                key[1],
            ),
        )
    )


def _verify_r1_manifest(path: Path) -> None:
    source = _resolve_input(path).read_bytes()
    if _sha256(source) != R1_MANIFEST_SHA256:
        raise ValueError("input differs from the exact R1 manifest")
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError("R1 manifest is not valid JSON") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    observed = (
        [
            (stream.get("topic_id"), stream.get("stream_id"), stream.get("query"))
            for stream in streams
        ]
        if isinstance(streams, list) and all(isinstance(stream, dict) for stream in streams)
        else None
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "sparse-relevance-repair-manifest-v1"
        or payload.get("renderer") != "R1"
        or payload.get("protected_topic_ids") != list(PROTECTED_TOPIC_IDS)
        or observed != list(_R1_SPECS)
    ):
        raise ValueError("input differs from the exact R1 manifest")


def _verify_prior_freeze(path: Path) -> None:
    source_path = _resolve_input(path)
    if source_path.is_dir():
        source_path = source_path / "freeze.json"
    source = source_path.read_bytes()
    if _sha256(source) != PRIOR_FREEZE_SHA256:
        raise ValueError("input differs from the exact immutable prior freeze")
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError("prior freeze is not valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "sparse-relevance-ranking-freeze-v1"
        or payload.get("status") != "frozen_before_qrels"
        or payload.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or payload.get("manifest_sha256", {}).get("R1") != R1_MANIFEST_SHA256
    ):
        raise ValueError("input differs from the exact immutable prior freeze")


def _request(topic_id: str, variant: str, query: str) -> RetrievalRequest:
    return RetrievalRequest.from_query(
        topic_id=topic_id,
        variant_name=variant,
        query_text=query,
        index_url=INDEX_URL,
        index_id=INDEX_ID,
        hits=EXPECTED_DEPTH,
        analyzer_fingerprint_sha256=ANALYZER_FINGERPRINT_SHA256,
        retriever_version=RETRIEVER_VERSION,
    )


def _ledger_from_existing(
    run_dir: Path, cache_dir: Path, *, expected_requests: int
) -> RetrievalLedger:
    resolved_run = _resolve_input(run_dir)
    resolved_cache = _resolve_input(cache_dir)
    try:
        policy = json.loads((resolved_run / "ledger.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load verified source ledger: {run_dir}") from exc
    if not isinstance(policy, dict):
        raise ValueError(f"source ledger policy is invalid: {run_dir}")
    ledger = RetrievalLedger(
        resolved_run,
        shared_cache_dir=resolved_cache,
        max_calls=policy.get("max_calls"),
        max_calls_per_topic=policy.get("max_calls_per_topic"),
        min_results=policy.get("min_results"),
        required_text_results=policy.get("required_text_results"),
    )
    report = ledger.validate_run()
    if report.failures or report.pending or report.planned_requests != expected_requests:
        raise ValueError(
            f"source ledger must contain exactly {expected_requests} verified requests"
        )
    return ledger


def _ledger_tree_sha256(run_dir: Path) -> str:
    root = _resolve_input(run_dir)
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item.name != ".ledger.lock"
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_verified_sources(
    *,
    base_run: Path,
    base_cache: Path,
    r1_run: Path,
    r1_cache: Path,
) -> _LoadedSources:
    base_ledger = _ledger_from_existing(base_run, base_cache, expected_requests=27)
    r1_ledger = _ledger_from_existing(r1_run, r1_cache, expected_requests=22)
    loaded: list[_LoadedSource] = []
    for topic_id, variant, query in _BASE_SPECS:
        request = _request(topic_id, variant, query)
        loaded.append(
            _LoadedSource(
                source_kind="base",
                source_ledger=str(_DEFAULT_BASE_RUN),
                source_cache=str(_DEFAULT_BASE_CACHE),
                request=request,
                result=base_ledger.load_verified_result(request),
            )
        )
    for topic_id, stream_id, query in _R1_SPECS:
        request = _request(
            topic_id, f"sparse_relevance_v1:R1:{stream_id}", query
        )
        loaded.append(
            _LoadedSource(
                source_kind="r1",
                source_ledger=str(_DEFAULT_R1_RUN),
                source_cache=str(_DEFAULT_R1_CACHE),
                request=request,
                result=r1_ledger.load_verified_result(request),
            )
        )
    return _LoadedSources(
        streams=tuple(loaded),
        base_ledger_sha256=_ledger_tree_sha256(base_run),
        r1_ledger_sha256=_ledger_tree_sha256(r1_run),
    )


def _candidate_rows(source: _LoadedSource) -> list[dict[str, object]]:
    request = source.request
    candidates = sorted(source.result.candidates, key=lambda row: row.rank)
    if (
        len(candidates) != EXPECTED_DEPTH
        or [candidate.rank for candidate in candidates]
        != list(range(1, EXPECTED_DEPTH + 1))
        or len({candidate.docid for candidate in candidates}) != EXPECTED_DEPTH
    ):
        raise ValueError(
            f"source stream {request.identity.topic_id}/{request.identity.variant_name} "
            "must contain unique ranks 1 through 100"
        )
    family = "original" if request.identity.variant_name == ORIGINAL_VARIANT else "facet"
    query_sha256 = _sha256(request.query_text.encode("utf-8"))
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        if (
            not isinstance(candidate.docid, str)
            or not candidate.docid
            or not isinstance(candidate.text, str)
            or not candidate.text
            or isinstance(candidate.score, bool)
            or not isinstance(candidate.score, (int, float))
            or not math.isfinite(float(candidate.score))
        ):
            raise ValueError("source stream contains an invalid candidate")
        rows.append(
            {
                "schema_version": CANDIDATE_ROW_SCHEMA_VERSION,
                "topic_id": request.identity.topic_id,
                "family": family,
                "variant": request.identity.variant_name,
                "query": request.query_text,
                "query_sha256": query_sha256,
                "rank": candidate.rank,
                "document_id": candidate.docid,
                "text": candidate.text,
                "text_sha256": _sha256(candidate.text.encode("utf-8")),
                "source_score": float(candidate.score),
            }
        )
    return rows


def _validate_loaded_sources(value: object) -> _LoadedSources:
    if not isinstance(value, _LoadedSources):
        raise TypeError("source_loader must return the verified source bundle")
    expected = _expected_queries()
    observed = {
        (source.request.identity.topic_id, source.request.identity.variant_name):
        source.request.query_text
        for source in value.streams
    }
    if len(value.streams) != 31 or observed != expected:
        raise ValueError("source loader differs from the exact 31-stream namespace")
    _require_sha256(value.base_ledger_sha256, "base_ledger_sha256")
    _require_sha256(value.r1_ledger_sha256, "r1_ledger_sha256")
    return value


def _exclusive_write(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def build_facet_local_manifest(
    *,
    r1_manifest_path: Path = _DEFAULT_R1_MANIFEST,
    prior_freeze_path: Path = _DEFAULT_PRIOR_FREEZE,
    base_run: Path = _DEFAULT_BASE_RUN,
    base_cache: Path = _DEFAULT_BASE_CACHE,
    r1_run: Path = _DEFAULT_R1_RUN,
    r1_cache: Path = _DEFAULT_R1_CACHE,
    source_output: Path = _DEFAULT_SOURCE_OUTPUT,
    source_loader: Callable[[], object] | None = None,
) -> FacetLocalManifest:
    """Build and create-only publish the exact immutable source snapshot."""

    _reject_protected(PILOT_TOPIC_IDS)
    output = _resolve_input(source_output)
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {source_output}")
    _verify_exact_source_paths(
        base_run=base_run,
        base_cache=base_cache,
        r1_run=r1_run,
        r1_cache=r1_cache,
    )
    _verify_r1_manifest(r1_manifest_path)
    _verify_prior_freeze(prior_freeze_path)
    loaded = _validate_loaded_sources(
        source_loader()
        if source_loader is not None
        else _load_verified_sources(
            base_run=base_run,
            base_cache=base_cache,
            r1_run=r1_run,
            r1_cache=r1_cache,
        )
    )

    source_by_key = {
        (source.request.identity.topic_id, source.request.identity.variant_name): source
        for source in loaded.streams
    }
    sources = [source_by_key[key] for key in _canonical_stream_keys()]
    all_rows: list[dict[str, object]] = []
    streams: list[FacetLocalStream] = []
    for source in sources:
        rows = _candidate_rows(source)
        stream_bytes = _jsonl_bytes(rows)
        all_rows.extend(rows)
        request = source.request
        streams.append(
            FacetLocalStream(
                topic_id=request.identity.topic_id,
                family=(
                    "original"
                    if request.identity.variant_name == ORIGINAL_VARIANT
                    else "facet"
                ),
                variant=request.identity.variant_name,
                query=request.query_text,
                query_sha256=_sha256(request.query_text.encode("utf-8")),
                source_kind=source.source_kind,
                source_ledger=source.source_ledger,
                source_cache=source.source_cache,
                source_request_sha256=request.identity.request_key,
                source_response_sha256=source.result.response_sha256,
                source_candidates_sha256=source.result.candidates_sha256,
                expected_rows=EXPECTED_DEPTH,
                candidates_sha256=_sha256(stream_bytes),
            )
        )
    candidate_bytes = _jsonl_bytes(all_rows)
    receipt = {
        "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
        "status": "frozen_source_snapshot",
        "r1_manifest_sha256": R1_MANIFEST_SHA256,
        "prior_freeze_sha256": PRIOR_FREEZE_SHA256,
        "base_ledger_sha256": loaded.base_ledger_sha256,
        "r1_ledger_sha256": loaded.r1_ledger_sha256,
        "candidate_schema_version": CANDIDATE_ROW_SCHEMA_VERSION,
        "candidate_file": "candidates.jsonl",
        "stream_count": 31,
        "candidate_rows": len(all_rows),
        "candidate_bytes": len(candidate_bytes),
        "candidates_sha256": _sha256(candidate_bytes),
    }
    receipt_bytes = _canonical_json(receipt)
    manifest = FacetLocalManifest(
        topic_ids=PILOT_TOPIC_IDS,
        protected_topic_ids=PROTECTED_TOPIC_IDS,
        r1_manifest_sha256=R1_MANIFEST_SHA256,
        prior_freeze_sha256=PRIOR_FREEZE_SHA256,
        base_ledger_sha256=loaded.base_ledger_sha256,
        r1_ledger_sha256=loaded.r1_ledger_sha256,
        candidate_schema_version=CANDIDATE_ROW_SCHEMA_VERSION,
        candidate_file="candidates.jsonl",
        candidate_rows=len(all_rows),
        candidate_bytes=len(candidate_bytes),
        candidates_sha256=_sha256(candidate_bytes),
        source_receipt_schema_version=SOURCE_RECEIPT_SCHEMA_VERSION,
        source_receipt_file="source_receipt.json",
        source_receipt_sha256=_sha256(receipt_bytes),
        streams=tuple(streams),
    )
    _validate_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"create-only output already exists: {source_output}"
        ) from exc
    _exclusive_write(output / manifest.candidate_file, candidate_bytes)
    _exclusive_write(output / manifest.source_receipt_file, receipt_bytes)
    load_facet_local_source_snapshot(output, manifest)
    return manifest


def _validate_manifest(manifest: FacetLocalManifest) -> FacetLocalManifest:
    if manifest.topic_ids != ("200", "225", "707", "897"):
        raise ValueError("manifest topic IDs differ from the frozen pilot")
    _reject_protected(manifest.topic_ids)
    if manifest.protected_topic_ids != PROTECTED_TOPIC_IDS:
        raise ValueError("protected topic namespace differs")
    if (
        manifest.r1_manifest_sha256 != R1_MANIFEST_SHA256
        or manifest.prior_freeze_sha256 != PRIOR_FREEZE_SHA256
        or manifest.candidate_schema_version != CANDIDATE_ROW_SCHEMA_VERSION
        or manifest.candidate_file != "candidates.jsonl"
        or manifest.source_receipt_schema_version != SOURCE_RECEIPT_SCHEMA_VERSION
        or manifest.source_receipt_file != "source_receipt.json"
    ):
        raise ValueError("manifest source bindings differ from the frozen contract")
    for field, value in (
        ("base_ledger_sha256", manifest.base_ledger_sha256),
        ("r1_ledger_sha256", manifest.r1_ledger_sha256),
        ("candidates_sha256", manifest.candidates_sha256),
        ("source_receipt_sha256", manifest.source_receipt_sha256),
    ):
        _require_sha256(value, field)
    expected = _expected_queries()
    observed = {(stream.topic_id, stream.variant): stream.query for stream in manifest.streams}
    if len(manifest.streams) != 31 or observed != expected:
        raise ValueError("manifest differs from the exact 31-stream namespace")
    if tuple(observed) != _canonical_stream_keys():
        raise ValueError("manifest streams are not in canonical order")
    if manifest.stream_counts != {"200": 10, "225": 8, "707": 4, "897": 9}:
        raise ValueError("manifest stream counts differ from 10/8/4/9")
    if (
        manifest.candidate_rows != 3100
        or manifest.candidate_bytes <= 0
        or sum(stream.expected_rows for stream in manifest.streams) != 3100
    ):
        raise ValueError("manifest candidate counts differ from the frozen contract")
    for stream in manifest.streams:
        if stream.expected_rows != EXPECTED_DEPTH:
            raise ValueError("stream expected_rows must be exactly 100")
        expected_family = "original" if stream.variant == ORIGINAL_VARIANT else "facet"
        expected_source = "base" if stream.variant.startswith("prompt_lab_v1:") else "r1"
        expected_ledger = str(
            _DEFAULT_BASE_RUN if expected_source == "base" else _DEFAULT_R1_RUN
        )
        expected_cache = str(
            _DEFAULT_BASE_CACHE if expected_source == "base" else _DEFAULT_R1_CACHE
        )
        if (
            stream.family != expected_family
            or stream.source_kind != expected_source
            or stream.query_sha256 != _sha256(stream.query.encode("utf-8"))
            or stream.source_ledger != expected_ledger
            or stream.source_cache != expected_cache
        ):
            raise ValueError("manifest stream source binding is invalid")
        for field, value in (
            ("source_request_sha256", stream.source_request_sha256),
            ("source_response_sha256", stream.source_response_sha256),
            ("source_candidates_sha256", stream.source_candidates_sha256),
            ("candidates_sha256", stream.candidates_sha256),
        ):
            _require_sha256(value, field)
    return manifest


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _parse_stream(value: object) -> FacetLocalStream:
    if not isinstance(value, dict) or set(value) != _STREAM_FIELDS:
        raise ValueError("manifest stream fields differ")
    expected_rows = value.get("expected_rows")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
        raise ValueError("expected_rows must be an integer")
    return FacetLocalStream(
        topic_id=_required_text(value.get("topic_id"), "topic_id"),
        family=_required_text(value.get("family"), "family"),
        variant=_required_text(value.get("variant"), "variant"),
        query=_required_text(value.get("query"), "query"),
        query_sha256=_require_sha256(value.get("query_sha256"), "query_sha256"),
        source_kind=_required_text(value.get("source_kind"), "source_kind"),
        source_ledger=_required_text(value.get("source_ledger"), "source_ledger"),
        source_cache=_required_text(value.get("source_cache"), "source_cache"),
        source_request_sha256=_require_sha256(
            value.get("source_request_sha256"), "source_request_sha256"
        ),
        source_response_sha256=_require_sha256(
            value.get("source_response_sha256"), "source_response_sha256"
        ),
        source_candidates_sha256=_require_sha256(
            value.get("source_candidates_sha256"), "source_candidates_sha256"
        ),
        expected_rows=expected_rows,
        candidates_sha256=_require_sha256(
            value.get("candidates_sha256"), "candidates_sha256"
        ),
    )


def load_facet_local_manifest(path: Path) -> FacetLocalManifest:
    """Load and fully validate a durable facet-local MiniLM manifest."""

    try:
        source = Path(path).read_bytes()
        payload = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load facet-local MiniLM manifest") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != _ROOT_FIELDS
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_source_snapshot"
        or _canonical_json(payload) != source
    ):
        raise ValueError("facet-local MiniLM manifest is not canonical")
    topic_ids = payload.get("topic_ids")
    protected = payload.get("protected_topic_ids")
    raw_streams = payload.get("streams")
    if not all(isinstance(value, list) for value in (topic_ids, protected, raw_streams)):
        raise ValueError("facet-local MiniLM manifest lists are invalid")
    integers = (payload.get("stream_count"), payload.get("candidate_rows"), payload.get("candidate_bytes"))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise ValueError("facet-local MiniLM manifest counts are invalid")
    streams = tuple(_parse_stream(value) for value in raw_streams)
    if payload.get("stream_count") != len(streams):
        raise ValueError("manifest stream count differs")
    return _validate_manifest(
        FacetLocalManifest(
            topic_ids=tuple(_required_text(value, "topic_ids") for value in topic_ids),
            protected_topic_ids=tuple(
                _required_text(value, "protected_topic_ids") for value in protected
            ),
            r1_manifest_sha256=_require_sha256(
                payload.get("r1_manifest_sha256"), "r1_manifest_sha256"
            ),
            prior_freeze_sha256=_require_sha256(
                payload.get("prior_freeze_sha256"), "prior_freeze_sha256"
            ),
            base_ledger_sha256=_require_sha256(
                payload.get("base_ledger_sha256"), "base_ledger_sha256"
            ),
            r1_ledger_sha256=_require_sha256(
                payload.get("r1_ledger_sha256"), "r1_ledger_sha256"
            ),
            candidate_schema_version=_required_text(
                payload.get("candidate_schema_version"), "candidate_schema_version"
            ),
            candidate_file=_required_text(payload.get("candidate_file"), "candidate_file"),
            candidate_rows=payload["candidate_rows"],
            candidate_bytes=payload["candidate_bytes"],
            candidates_sha256=_require_sha256(
                payload.get("candidates_sha256"), "candidates_sha256"
            ),
            source_receipt_schema_version=_required_text(
                payload.get("source_receipt_schema_version"),
                "source_receipt_schema_version",
            ),
            source_receipt_file=_required_text(
                payload.get("source_receipt_file"), "source_receipt_file"
            ),
            source_receipt_sha256=_require_sha256(
                payload.get("source_receipt_sha256"), "source_receipt_sha256"
            ),
            streams=streams,
        )
    )


def write_facet_local_manifest(path: Path, manifest: FacetLocalManifest) -> None:
    """Create one durable manifest without replacing an existing artifact."""

    _validate_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_write(output, manifest.to_json_bytes())


def load_facet_local_source_snapshot(
    source_output: Path, manifest: FacetLocalManifest
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Load candidates from the immutable snapshot/receipt boundary only."""

    _validate_manifest(manifest)
    source_dir = Path(source_output)
    candidate_path = source_dir / manifest.candidate_file
    receipt_path = source_dir / manifest.source_receipt_file
    if not candidate_path.is_file() or not receipt_path.is_file():
        raise ValueError("downstream inputs must be the frozen snapshot and receipt only")
    candidate_bytes = candidate_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    try:
        receipt = json.loads(receipt_bytes)
        rows = tuple(
            json.loads(line) for line in candidate_bytes.splitlines() if line
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source snapshot is not canonical JSON") from exc
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _RECEIPT_FIELDS
        or _canonical_json(receipt) != receipt_bytes
        or receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "frozen_source_snapshot"
        or receipt.get("r1_manifest_sha256") != manifest.r1_manifest_sha256
        or receipt.get("prior_freeze_sha256") != manifest.prior_freeze_sha256
        or receipt.get("base_ledger_sha256") != manifest.base_ledger_sha256
        or receipt.get("r1_ledger_sha256") != manifest.r1_ledger_sha256
        or receipt.get("candidate_schema_version") != manifest.candidate_schema_version
        or receipt.get("candidate_file") != manifest.candidate_file
        or receipt.get("stream_count") != len(manifest.streams)
        or receipt.get("candidate_rows") != manifest.candidate_rows
        or receipt.get("candidate_bytes") != manifest.candidate_bytes
        or receipt.get("candidates_sha256") != manifest.candidates_sha256
        or _sha256(receipt_bytes) != manifest.source_receipt_sha256
        or len(candidate_bytes) != manifest.candidate_bytes
        or _sha256(candidate_bytes) != manifest.candidates_sha256
        or len(rows) != manifest.candidate_rows
        or _jsonl_bytes(rows) != candidate_bytes
    ):
        raise ValueError("source snapshot receipt or byte binding differs")
    if any(
        not isinstance(row, dict)
        or set(row) != _CANDIDATE_FIELDS
        or row.get("schema_version") != CANDIDATE_ROW_SCHEMA_VERSION
        for row in rows
    ):
        raise ValueError("source snapshot candidate schema differs")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        topic_id = row.get("topic_id")
        variant = row.get("variant")
        query = row.get("query")
        text = row.get("text")
        rank = row.get("rank")
        score = row.get("source_score")
        document_id = row.get("document_id")
        if (
            not isinstance(topic_id, str)
            or topic_id in PROTECTED_TOPIC_IDS
            or not isinstance(variant, str)
            or not isinstance(query, str)
            or not isinstance(text, str)
            or not text
            or not isinstance(document_id, str)
            or not document_id
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or row.get("query_sha256") != _sha256(query.encode("utf-8"))
            or row.get("text_sha256") != _sha256(text.encode("utf-8"))
        ):
            raise ValueError("source snapshot candidate is invalid")
        grouped[(topic_id, variant)].append(row)
    if set(grouped) != set(_expected_queries()):
        raise ValueError("source snapshot stream namespace differs")
    expected_order = tuple(
        (stream.topic_id, stream.variant, rank)
        for stream in manifest.streams
        for rank in range(1, 101)
    )
    observed_order = tuple(
        (str(row["topic_id"]), str(row["variant"]), int(row["rank"]))
        for row in rows
    )
    if observed_order != expected_order:
        raise ValueError("source snapshot rows are not in canonical order")
    for stream in manifest.streams:
        stream_rows = grouped[(stream.topic_id, stream.variant)]
        if (
            len(stream_rows) != 100
            or [row["rank"] for row in stream_rows] != list(range(1, 101))
            or len({row["document_id"] for row in stream_rows}) != 100
            or {row["query"] for row in stream_rows} != {stream.query}
            or {row["family"] for row in stream_rows} != {stream.family}
            or _sha256(_jsonl_bytes(stream_rows)) != stream.candidates_sha256
        ):
            raise ValueError("source snapshot stream candidates differ")
    return rows, receipt
