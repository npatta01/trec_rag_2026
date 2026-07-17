"""All-topic tokenizer preflight and approval-gated local MiniLM scoring.

The preflight path authenticates the sealed planning/retrieval chain, loads only
the pinned local tokenizer, and freezes exact query/document windows.  It has no
network, qrels, model-construction, or inference interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import struct
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from itertools import zip_longest
from pathlib import Path

from . import facet_local_minilm_preflight as _window_module
from .all_topic_facet_contract import ALL_TOPIC_IDS, EXPERIMENT_ID, verify_planning
from .all_topic_facet_retrieve import APPROVED_ORIGINAL_CACHE_ROOT, verify_retrieval
from .facet_local_minilm_preflight import (
    MODEL_ID,
    MODEL_REVISION,
    PAIR_MAX_TOKENS,
    WINDOW_SCHEMA_VERSION,
    WINDOW_POLICY_VERSION,
    WindowPlanRow,
    _window_id,
    build_window_plan,
    load_verified_materialization,
    load_verified_tokenizer,
    score_cache_context,
)
from .facet_local_minilm_rank import aggregate_top4
from .rerank_score_cache import GlobalScoreCache
from .tethered_facet_minilm_score import (
    SCORE_CACHE_ROOT,
    TokenizerOnlyAuto,
    _LocalMiniLMRunner,
    _cache_binding,
    _selected_top4,
)


PLANNING_ROOT_SHA256 = "bc1351cca8aa05dd0979a342c8dd5f72395f2b7a9207ae20668aba05f1ab85a2"
RETRIEVAL_ROOT_SHA256 = "f2191c295f600d0243f4dcd2b1dc23a05b9fa8a7d9a4d6258983a0d3c9433aeb"
APPROVED_SCORE_PLAN_ROOT_SHA256 = "48c8b9be21adece21c1f17cdeec4de8694a83f5ee7fccaa9fc07ff8ca36944c3"
MANIFEST_SHA256 = "03a55f669f312b09434226928d9eb3a61f49f47f463b88acf9bd4d57d743f1d5"
ACCEPTED_UNION_SHA256 = "f91c38f721e6821015d28ae9928637757f2763eeb5c7270822d4e695b312d1d9"
EXPECTED_UNION_ROWS = 45_144
EXPECTED_FACET_ASSOCIATIONS = 29_600
EXPECTED_PAIR_COUNT = 119_888
EXPECTED_FACET_COUNT = 148
PREFLIGHT_SCHEMA_VERSION = "all-topic-tethered-score-preflight-v1"
PAIR_SCHEMA_VERSION = "all-topic-tethered-score-pair-v1"
SCORE_SCHEMA_VERSION = "all-topic-tethered-window-score-v1"
DOCUMENT_SCORE_SCHEMA_VERSION = "all-topic-tethered-document-score-v1"
FEATURE_SCHEMA_VERSION = "all-topic-tethered-percentile-feature-v1"
SCORING_RECEIPT_SCHEMA_VERSION = "all-topic-tethered-scoring-receipt-v1"
SCORE_LEDGER_SCHEMA_VERSION = "all-topic-tethered-score-ledger-v1"
PLAN_SEAL_SCHEMA_VERSION = "all-topic-tethered-score-plan-seal-v1"
SCORING_SEAL_SCHEMA_VERSION = "all-topic-tethered-scoring-seal-v1"
DEFAULT_MODEL_RECEIPT = Path("outputs/rag25_facet_local_minilm_v1/model_v1/materialization.json")
REFERENCE_RECEIPT = Path(
    "outputs/rag25_deep_facet_candidates_v1/"
    "post_qrels_tethered_facet_minilm_v1/scoring/scoring_receipt.json"
)
BATCH_SIZE = 32

_ZERO_EXTERNAL_CALLS = {
    "model_load": 0,
    "model_inference": 0,
    "network": 0,
    "hosted": 0,
    "paid": 0,
    "qrels": 0,
}


class _PreparedDocumentTokenizer:
    """Cache only the current document tokens; query/window behavior is unchanged."""

    def __init__(self, backend: object) -> None:
        self._backend = backend
        self._document_text: str | None = None
        self._document_tokens: list[object] | None = None

    def prepare_document(self, text: str) -> None:
        if text != self._document_text:
            self._document_text = text
            self._document_tokens = list(
                self._backend.encode(  # type: ignore[attr-defined]
                    text, add_special_tokens=False, truncation=False
                )
            )

    def encode(
        self, text: str, *, add_special_tokens: bool = False, truncation: bool = False
    ) -> list[object]:
        if (
            not add_special_tokens
            and not truncation
            and text == self._document_text
            and self._document_tokens is not None
        ):
            return list(self._document_tokens)
        return list(
            self._backend.encode(  # type: ignore[attr-defined]
                text,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
            )
        )

    def decode(self, token_ids: Sequence[object], **kwargs: object) -> str:
        return str(self._backend.decode(token_ids, **kwargs))  # type: ignore[attr-defined]

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        return int(
            self._backend.num_special_tokens_to_add(pair=pair)  # type: ignore[attr-defined]
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _compact_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _read_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        source = Path(path).read_bytes()
        value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, source


def _iter_jsonl(path: Path, label: str) -> Iterable[dict[str, object]]:
    try:
        source = Path(path).open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    with source:
        for number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label}:{number} is invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{number} must be an object")
            yield value


def _file_binding(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def render_tethered_query(narrative: str, facet_query: str) -> str:
    """Render exactly one narrative plus exactly one originating facet."""

    narrative = narrative.strip()
    facet_query = facet_query.strip()
    if not narrative or not facet_query:
        raise ValueError("narrative and facet query must be non-empty")
    return f"{narrative}\n\nFocus: {facet_query}"


def _manifest_indexes(
    manifest: Mapping[str, object], *, production: bool = False
) -> tuple[dict[str, str], dict[str, dict[str, object]], dict[str, str]]:
    raw_topic_ids = manifest.get("topic_ids")
    if not isinstance(raw_topic_ids, list) or any(
        not isinstance(value, str) for value in raw_topic_ids
    ):
        raise ValueError("manifest topic IDs are invalid")
    if production and tuple(raw_topic_ids) != ALL_TOPIC_IDS:
        raise ValueError("production scoring requires the exact ordered 22-topic scope")
    if len(set(raw_topic_ids)) != len(raw_topic_ids) or not raw_topic_ids:
        raise ValueError("manifest topic IDs must be unique and non-empty")
    topics = manifest.get("topics")
    facets = manifest.get("facets")
    if not isinstance(topics, list) or not isinstance(facets, list):
        raise ValueError("manifest topics/facets are invalid")
    narratives: dict[str, str] = {}
    for row in topics:
        if not isinstance(row, Mapping):
            raise ValueError("manifest topic row is invalid")
        topic_id = row.get("topic_id")
        narrative = row.get("narrative")
        if topic_id not in raw_topic_ids or not isinstance(narrative, str) or not narrative:
            raise ValueError("manifest topic narrative is invalid")
        if row.get("narrative_sha256") != _sha256_text(narrative):
            raise ValueError("manifest narrative hash differs")
        if str(topic_id) in narratives:
            raise ValueError("manifest topic is duplicated")
        narratives[str(topic_id)] = narrative
    if set(narratives) != set(raw_topic_ids):
        raise ValueError("manifest topic coverage differs")
    facet_by_id: dict[str, dict[str, object]] = {}
    common_terms: dict[str, list[str]] = {topic_id: [] for topic_id in raw_topic_ids}
    for row in facets:
        if not isinstance(row, Mapping):
            raise ValueError("manifest facet row is invalid")
        topic_id = row.get("topic_id")
        facet_id = row.get("facet_id")
        query = row.get("query")
        terms = row.get("analyzer_terms")
        if (
            topic_id not in raw_topic_ids
            or not isinstance(facet_id, str)
            or not facet_id
            or not isinstance(query, str)
            or not query
            or row.get("query_sha256") != _sha256_text(query)
            or not isinstance(terms, list)
            or not terms
            or any(not isinstance(term, str) or not term for term in terms)
        ):
            raise ValueError("manifest facet identity/query is invalid")
        if facet_id in facet_by_id:
            raise ValueError("manifest facet is duplicated")
        facet_by_id[facet_id] = dict(row)
        for term in terms:
            if term not in common_terms[str(topic_id)]:
                common_terms[str(topic_id)].append(term)
    common_queries = {
        topic_id: " ".join(common_terms[topic_id]) for topic_id in raw_topic_ids
    }
    if any(not query for query in common_queries.values()):
        raise ValueError("each topic requires a non-empty common/global query")
    return narratives, facet_by_id, common_queries


def _facet_ids_from_provenance(
    row: Mapping[str, object], facet_by_id: Mapping[str, Mapping[str, object]]
) -> list[str]:
    raw = row.get("stream_provenance")
    if not isinstance(raw, list) or not raw:
        raise ValueError("union stream provenance is invalid")
    output: list[str] = []
    seen: set[str] = set()
    for source in raw:
        if not isinstance(source, Mapping):
            raise ValueError("union stream provenance row is invalid")
        stream_id = source.get("stream_id")
        if stream_id == "original":
            continue
        if not isinstance(stream_id, str) or stream_id not in facet_by_id:
            raise ValueError("union stream is absent from the facet manifest")
        if stream_id in seen:
            raise ValueError("union repeats one facet/document provenance")
        seen.add(stream_id)
        output.append(stream_id)
    return output


def _iter_pair_rows(
    union_rows: Iterable[Mapping[str, object]], manifest: Mapping[str, object]
) -> Iterable[dict[str, object]]:
    narratives, facets, common_queries = _manifest_indexes(manifest)
    topic_ranks: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for row in union_rows:
        topic_id = row.get("topic_id")
        document_id = row.get("document_id")
        text = row.get("text")
        if (
            topic_id not in narratives
            or not isinstance(document_id, str)
            or not document_id
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError("union identity/text is invalid")
        identity = (str(topic_id), document_id)
        if identity in seen:
            raise ValueError("union topic/document identity is duplicated")
        seen.add(identity)
        topic_ranks[str(topic_id)] += 1
        rank = topic_ranks[str(topic_id)]
        text_sha = _sha256_text(text)
        provenance = row.get("stream_provenance")
        assert isinstance(provenance, list)
        if any(
            not isinstance(source, Mapping)
            or not isinstance(source.get("text_sha256"), str)
            or len(str(source["text_sha256"])) != 64
            or any(character not in "0123456789abcdef" for character in str(source["text_sha256"]))
            for source in provenance
        ):
            raise ValueError("union provenance text hash is invalid")
        base = {
            "topic_id": str(topic_id),
            "document_id": document_id,
            "text": text,
            "text_sha256": text_sha,
        }
        for family, query_id, query in (
            ("narrative", "n", narratives[str(topic_id)]),
            ("common", "g", common_queries[str(topic_id)]),
        ):
            yield {
                **base,
                "family": family,
                "variant": f"{topic_id}:{query_id}",
                "query_id": query_id,
                "rank": rank,
                "query": query,
                "query_sha256": _sha256_text(query),
            }
        for facet_id in _facet_ids_from_provenance(row, facets):
            facet = facets[facet_id]
            if facet.get("topic_id") != topic_id:
                raise ValueError("facet provenance crosses topics")
            query = render_tethered_query(narratives[str(topic_id)], str(facet["query"]))
            yield {
                **base,
                "family": "tethered_facet",
                "variant": facet_id,
                "query_id": facet_id,
                "facet_id": facet_id,
                "manifest_order": facet.get("manifest_order"),
                "rank": rank,
                "query": query,
                "query_sha256": _sha256_text(query),
            }


def _pair_rows(
    union_rows: Sequence[Mapping[str, object]], manifest: Mapping[str, object]
) -> list[dict[str, object]]:
    return list(_iter_pair_rows(union_rows, manifest))


def _authorized_window_plan(
    pair: Mapping[str, object], backend: object
) -> tuple[WindowPlanRow, ...]:
    """Reuse the frozen builder without weakening its historical topic guard."""

    topic_id = str(pair.get("topic_id"))
    if topic_id not in ALL_TOPIC_IDS:
        # Pure synthetic tests may use a manifest subset of the authorized list only.
        raise ValueError(f"topic {topic_id} is outside the experiment allowlist")
    prepare = getattr(backend, "prepare_document", None)
    if callable(prepare):
        prepare(str(pair["text"]))
    proxy = pair
    if topic_id in {"144", "213", "224", "407", "515"}:
        proxy = {**pair, "topic_id": "14"}
    rows = build_window_plan(proxy, backend, query=str(pair["query"]))
    if proxy is pair:
        return rows
    materialized: list[WindowPlanRow] = []
    for row in rows:
        values = row.to_dict()
        values.pop("schema_version")
        values["topic_id"] = topic_id
        values["window_id"] = _window_id(values)
        materialized.append(WindowPlanRow(**values))
    return tuple(materialized)


def build_score_plan(
    union_rows: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
    *,
    backend: object,
    cache: object | None = None,
) -> dict[str, object]:
    """Pure tokenizer-only plan used by tests and the production persister."""

    pairs = _pair_rows(union_rows, manifest)
    prepared_backend = _PreparedDocumentTokenizer(backend)
    windows: list[dict[str, object]] = []
    unique_keys: set[str] = set()
    unique_misses: set[str] = set()
    hits = 0
    # The frozen context is deterministic.  Resolve package metadata once rather
    # than once per window; this changes no cache key bytes.
    original_context_factory = _window_module.score_cache_context
    frozen_context = original_context_factory()
    _window_module.score_cache_context = lambda: frozen_context
    try:
        for pair in pairs:
            for planned in _authorized_window_plan(pair, prepared_backend):
                cached = False
                if cache is not None:
                    cached = cache.get(  # type: ignore[attr-defined]
                        query_text=planned.query, text=planned.window_text
                    ) is not None
                row = replace(planned, cache_hit=cached).to_dict()
                row["query_id"] = pair["query_id"]
                if "facet_id" in pair:
                    row["facet_id"] = pair["facet_id"]
                    row["manifest_order"] = pair["manifest_order"]
                windows.append(row)
                hits += int(cached)
                unique_keys.add(planned.cache_key)
                if not cached:
                    unique_misses.add(planned.cache_key)
    finally:
        _window_module.score_cache_context = original_context_factory
    return {
        "pairs": pairs,
        "windows": windows,
        "pair_count": len(pairs),
        "window_count": len(windows),
        "cache_hit_count": hits,
        "cache_miss_count": len(windows) - hits,
        "unique_pair_count": len(unique_keys),
        "unique_cache_miss_count": len(unique_misses),
        "external_calls": dict(_ZERO_EXTERNAL_CALLS),
    }


def percentile_features(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Add deterministic average-rank percentiles within each exact query only."""

    grouped: dict[tuple[str, str], list[tuple[int, Mapping[str, object]]]] = defaultdict(list)
    identities: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        topic_id, query_id, document_id = (
            row.get("topic_id"), row.get("query_id"), row.get("document_id")
        )
        score = row.get("score")
        if (
            not isinstance(topic_id, str)
            or not isinstance(query_id, str)
            or not isinstance(document_id, str)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("percentile score row is invalid")
        identity = (topic_id, query_id, document_id)
        if identity in identities:
            raise ValueError("duplicate query/document score identity")
        identities.add(identity)
        grouped[(topic_id, query_id)].append((index, row))
    percentiles: dict[int, float] = {}
    for members in grouped.values():
        score_groups: dict[float, list[int]] = defaultdict(list)
        for index, row in members:
            score_groups[float(row["score"])].append(index)
        count = len(members)
        first_rank = 1
        for score in sorted(score_groups, reverse=True):
            last_rank = first_rank + len(score_groups[score]) - 1
            average_rank = (first_rank + last_rank) / 2.0
            value = (
                1.0
                if count == 1
                else (count - average_rank) / (count - 1.0)
            )
            for index in score_groups[score]:
                percentiles[index] = value
            first_rank = last_rank + 1
    return [{**row, "percentile": percentiles[index]} for index, row in enumerate(rows)]


def _reference_projection(reference_receipt: Path, miss_count: int) -> dict[str, object]:
    receipt, source = _read_json(reference_receipt, "reference scoring receipt")
    pairs = receipt.get("unique_forward_pair_count")
    elapsed = receipt.get("elapsed_seconds")
    device_memory = receipt.get("peak_device_memory_bytes")
    host_memory = receipt.get("peak_host_memory_bytes")
    if (
        isinstance(pairs, bool)
        or not isinstance(pairs, int)
        or pairs <= 0
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or float(elapsed) <= 0
        or isinstance(device_memory, bool)
        or not isinstance(device_memory, int)
        or device_memory <= 0
        or isinstance(host_memory, bool)
        or not isinstance(host_memory, int)
        or host_memory <= 0
    ):
        raise ValueError("reference scoring receipt lacks runtime/memory evidence")
    rate = pairs / float(elapsed)
    return {
        "policy": "pinned_tethered_rocm_observed_rate_v1",
        "reference_receipt": str(reference_receipt.resolve()),
        "reference_receipt_sha256": _sha256_bytes(source),
        "reference_unique_forward_pair_count": pairs,
        "reference_elapsed_seconds": float(elapsed),
        "projected_pairs_per_second": rate,
        "projected_unique_cache_miss_count": miss_count,
        "projected_runtime_seconds": miss_count / rate,
        "estimated_peak_device_memory_bytes": device_memory,
        "estimated_peak_host_memory_bytes": host_memory,
    }


def create_preflight(
    retrieval_dir: Path,
    output_dir: Path,
    *,
    planning_dir: Path | None = None,
    model_receipt: Path = DEFAULT_MODEL_RECEIPT,
    cache_root: Path = SCORE_CACHE_ROOT,
    approved_original_cache_root: Path = APPROVED_ORIGINAL_CACHE_ROOT,
    reference_receipt: Path = REFERENCE_RECEIPT,
) -> dict[str, object]:
    """Authenticate upstream artifacts and freeze a create-only tokenizer plan."""

    retrieval = Path(retrieval_dir).resolve()
    planning = (
        Path(planning_dir).resolve()
        if planning_dir is not None
        else (retrieval.parent / "planning").resolve()
    )
    destination = Path(output_dir)
    if destination.exists() or os.path.lexists(destination):
        raise FileExistsError(f"create-only scoring preflight exists: {destination}")
    planning_evidence = verify_planning(
        planning, approved_cache_root=approved_original_cache_root
    )
    retrieval_evidence = verify_retrieval(
        retrieval,
        planning,
        approved_original_cache_root=approved_original_cache_root,
    )
    if (
        planning_evidence.get("root_sha256") != PLANNING_ROOT_SHA256
        or retrieval_evidence.get("root_sha256") != RETRIEVAL_ROOT_SHA256
        or retrieval_evidence.get("accepted_union_rows") != EXPECTED_UNION_ROWS
        or retrieval_evidence.get("topic_count") != len(ALL_TOPIC_IDS)
    ):
        raise ValueError("upstream production roots/counts differ from Task 1/2")
    manifest, manifest_source = _read_json(planning / "manifest.json", "manifest")
    _manifest_indexes(manifest, production=True)
    materialization = load_verified_materialization(model_receipt)
    tokenizer = load_verified_tokenizer(
        materialization.receipt_path, auto_tokenizer_cls=TokenizerOnlyAuto
    )
    cache = GlobalScoreCache(Path(cache_root), score_cache_context())
    cache_before = _cache_binding(cache)
    union_rows = list(_iter_jsonl(retrieval / "accepted_union.jsonl", "accepted union"))
    started = time.perf_counter()
    plan = build_score_plan(union_rows, manifest, backend=tokenizer, cache=cache)
    elapsed = time.perf_counter() - started
    facet_count = sum(
        row.get("family") == "tethered_facet" for row in plan["pairs"]  # type: ignore[index]
    )
    if (
        len(union_rows) != EXPECTED_UNION_ROWS
        or facet_count != EXPECTED_FACET_ASSOCIATIONS
        or plan["pair_count"] != EXPECTED_PAIR_COUNT
        or len({row["query_id"] for row in plan["pairs"] if row["family"] == "tethered_facet"})  # type: ignore[index]
        != EXPECTED_FACET_COUNT
    ):
        raise ValueError("exact production score inventory differs")
    if _cache_binding(cache) != cache_before:
        raise ValueError("score cache changed during tokenizer-only preflight")
    runtime = _reference_projection(reference_receipt, int(plan["unique_cache_miss_count"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        pair_path = destination / "pairs.jsonl"
        window_path = destination / "windows.jsonl"
        with pair_path.open("xb") as sink:
            for row in plan["pairs"]:  # type: ignore[index]
                compact = {key: value for key, value in row.items() if key != "text"}
                sink.write(_compact_bytes({"schema_version": PAIR_SCHEMA_VERSION, **compact}) + b"\n")
            sink.flush()
            os.fsync(sink.fileno())
        with window_path.open("xb") as sink:
            for row in plan["windows"]:  # type: ignore[index]
                sink.write(_compact_bytes(row) + b"\n")
            sink.flush()
            os.fsync(sink.fileno())
        pair_binding = _file_binding(pair_path)
        window_binding = _file_binding(window_path)
        payload: dict[str, object] = {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "tokenizer_only_preflight_complete",
            "experiment_id": EXPERIMENT_ID,
            "topic_ids": list(ALL_TOPIC_IDS),
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_materialization_receipt": str(materialization.receipt_path),
            "model_materialization_receipt_sha256": materialization.sha256,
            "tokenizer": {
                "load_count": 1,
                "class": type(tokenizer).__name__,
                "local_files_only": True,
                "trust_remote_code": False,
            },
            "window_policy": WINDOW_POLICY_VERSION,
            "pair_max_tokens": PAIR_MAX_TOKENS,
            "pair_count": plan["pair_count"],
            "window_count": plan["window_count"],
            "cache_hit_count": plan["cache_hit_count"],
            "cache_miss_count": plan["cache_miss_count"],
            "unique_pair_count": plan["unique_pair_count"],
            "unique_cache_miss_count": plan["unique_cache_miss_count"],
            "facet_association_count": facet_count,
            "query_count": 2 * len(ALL_TOPIC_IDS) + EXPECTED_FACET_COUNT,
            "inventory_estimate_pair_count": EXPECTED_PAIR_COUNT,
            "inventory_pair_delta": int(plan["pair_count"]) - EXPECTED_PAIR_COUNT,
            "tokenizer_planning_elapsed_seconds": elapsed,
            "runtime_projection": runtime,
            "external_calls": dict(_ZERO_EXTERNAL_CALLS),
            "model_constructed": False,
            "inference_authorized": False,
            "score_cache": {
                "context": score_cache_context().artifact_metadata,
                "binding": cache_before,
            },
            "sources": {
                "planning_root_sha256": PLANNING_ROOT_SHA256,
                "retrieval_root_sha256": RETRIEVAL_ROOT_SHA256,
                "manifest": {
                    "path": str((planning / "manifest.json").resolve()),
                    "sha256": _sha256_bytes(manifest_source),
                },
                "accepted_union": {
                    "path": str((retrieval / "accepted_union.jsonl").resolve()),
                    **_file_binding(retrieval / "accepted_union.jsonl"),
                },
            },
            "artifacts": {"pairs.jsonl": pair_binding, "windows.jsonl": window_binding},
        }
        _write_exclusive(destination / "preflight.json", _pretty_bytes(payload))
        files = {
            name: _file_binding(destination / name)
            for name in ("pairs.jsonl", "windows.jsonl", "preflight.json")
        }
        seal = {
            "schema_version": PLAN_SEAL_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "files": files,
            "root_sha256": _sha256_bytes(_compact_bytes(files)),
        }
        _write_exclusive(destination / "SCORE_PLAN_SEALED.json", _pretty_bytes(seal))
        return payload
    except BaseException:
        shutil.rmtree(destination)
        raise


def _validate_plan_rows(
    scoring: Path, payload: Mapping[str, object]
) -> tuple[set[tuple[str, str, str]], set[str]]:
    pair_identities: set[tuple[str, str, str]] = set()
    for row in _iter_jsonl(scoring / "pairs.jsonl", "score pairs"):
        topic_id = row.get("topic_id")
        query_id = row.get("query_id")
        document_id = row.get("document_id")
        query = row.get("query")
        text_sha = row.get("text_sha256")
        if (
            row.get("schema_version") != PAIR_SCHEMA_VERSION
            or topic_id not in ALL_TOPIC_IDS
            or not isinstance(query_id, str)
            or not query_id
            or not isinstance(document_id, str)
            or not document_id
            or not isinstance(query, str)
            or row.get("query_sha256") != _sha256_text(query)
            or not isinstance(text_sha, str)
            or len(text_sha) != 64
        ):
            raise ValueError("score pair row identity/lineage is invalid")
        identity = (str(topic_id), query_id, document_id)
        if identity in pair_identities:
            raise ValueError("score pair identity is duplicated")
        pair_identities.add(identity)

    window_pair_identities: set[tuple[str, str, str]] = set()
    window_ids: set[str] = set()
    cache_keys: set[str] = set()
    unique_misses: set[str] = set()
    hits = 0
    misses = 0
    original_context_factory = _window_module.score_cache_context
    frozen_context = original_context_factory()
    _window_module.score_cache_context = lambda: frozen_context
    try:
        for row in _iter_jsonl(scoring / "windows.jsonl", "score windows"):
            topic_id = row.get("topic_id")
            query_id = row.get("query_id")
            document_id = row.get("document_id")
            query = row.get("query")
            window_text = row.get("window_text")
            window_id = row.get("window_id")
            cache_key = row.get("cache_key")
            if (
                row.get("schema_version") != WINDOW_SCHEMA_VERSION
                or topic_id not in ALL_TOPIC_IDS
                or not isinstance(query_id, str)
                or not isinstance(document_id, str)
                or not isinstance(query, str)
                or not isinstance(window_text, str)
                or row.get("query_sha256") != _sha256_text(query)
                or row.get("window_sha256") != _sha256_text(window_text)
                or cache_key != _window_module._score_cache_key(query, window_text)
                or window_id != _window_id(row)
                or not isinstance(row.get("cache_hit"), bool)
                or not isinstance(row.get("pair_token_count"), int)
                or int(row["pair_token_count"]) > PAIR_MAX_TOKENS
            ):
                raise ValueError("score window identity/content is invalid")
            identity = (str(topic_id), query_id, document_id)
            if identity not in pair_identities:
                raise ValueError("score window is absent from the pair plan")
            if window_id in window_ids:
                raise ValueError("score window ID is duplicated")
            window_ids.add(str(window_id))
            window_pair_identities.add(identity)
            cache_keys.add(str(cache_key))
            if row["cache_hit"]:
                hits += 1
            else:
                misses += 1
                unique_misses.add(str(cache_key))
    finally:
        _window_module.score_cache_context = original_context_factory
    if (
        pair_identities != window_pair_identities
        or len(pair_identities) != payload.get("pair_count")
        or len(window_ids) != payload.get("window_count")
        or hits != payload.get("cache_hit_count")
        or misses != payload.get("cache_miss_count")
        or len(cache_keys) != payload.get("unique_pair_count")
        or len(unique_misses) != payload.get("unique_cache_miss_count")
    ):
        raise ValueError("score-plan semantic counters/coverage differ")
    return pair_identities, window_ids


def _replay_plan_derivation(
    scoring: Path,
    payload: Mapping[str, object],
    *,
    tokenizer: object | None = None,
    cache: object | None = None,
    authenticate_upstream: bool = True,
) -> tuple[set[tuple[str, str, str]], set[str]]:
    """Re-derive every pair/window from authenticated source semantics."""

    sources = payload.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("preflight semantic sources are missing")
    manifest_binding = sources.get("manifest")
    union_binding = sources.get("accepted_union")
    if not isinstance(manifest_binding, Mapping) or not isinstance(union_binding, Mapping):
        raise ValueError("preflight semantic source bindings are invalid")
    manifest_path = Path(str(manifest_binding.get("path"))).resolve()
    union_path = Path(str(union_binding.get("path"))).resolve()
    planning = manifest_path.parent
    retrieval = union_path.parent
    if authenticate_upstream:
        planning_evidence = verify_planning(
            planning, approved_cache_root=APPROVED_ORIGINAL_CACHE_ROOT
        )
        retrieval_evidence = verify_retrieval(
            retrieval,
            planning,
            approved_original_cache_root=APPROVED_ORIGINAL_CACHE_ROOT,
        )
        if (
            planning_evidence.get("root_sha256") != PLANNING_ROOT_SHA256
            or retrieval_evidence.get("root_sha256") != RETRIEVAL_ROOT_SHA256
        ):
            raise ValueError("semantic replay upstream roots differ")
    manifest, manifest_source = _read_json(manifest_path, "semantic replay manifest")
    if (
        _sha256_bytes(manifest_source) != manifest_binding.get("sha256")
        or _file_binding(union_path) != {
            "bytes": union_binding.get("bytes"),
            "sha256": union_binding.get("sha256"),
        }
    ):
        raise ValueError("semantic replay source bytes differ")
    _manifest_indexes(manifest, production=authenticate_upstream)
    if tokenizer is None:
        tokenizer = load_verified_tokenizer(
            Path(str(payload.get("model_materialization_receipt"))),
            auto_tokenizer_cls=TokenizerOnlyAuto,
        )
    prepared = _PreparedDocumentTokenizer(tokenizer)
    if cache is None:
        cache_evidence = payload.get("score_cache")
        if not isinstance(cache_evidence, Mapping):
            raise ValueError("semantic replay cache evidence is missing")
        binding = cache_evidence.get("binding")
        if not isinstance(binding, Mapping):
            raise ValueError("semantic replay cache binding is invalid")
        cache_path = Path(str(binding.get("path")))
        context = score_cache_context()
        root = cache_path
        for _ in context.path_parts:
            root = root.parent
        cache = GlobalScoreCache(root, context)
        if _cache_binding(cache) != binding:
            raise ValueError("semantic replay cache bytes differ from preflight")

    persisted_pairs = _iter_jsonl(scoring / "pairs.jsonl", "score pairs")
    persisted_windows = _iter_jsonl(scoring / "windows.jsonl", "score windows")
    pair_iterator = iter(persisted_pairs)
    window_iterator = iter(persisted_windows)
    pair_identities: set[tuple[str, str, str]] = set()
    window_ids: set[str] = set()
    cache_keys: set[str] = set()
    unique_misses: set[str] = set()
    hits = misses = 0
    original_context_factory = _window_module.score_cache_context
    frozen_context = original_context_factory()
    _window_module.score_cache_context = lambda: frozen_context
    try:
        union_rows = _iter_jsonl(union_path, "semantic replay accepted union")
        for derived_pair in _iter_pair_rows(union_rows, manifest):
            observed_pair = next(pair_iterator, None)
            expected_pair = {
                "schema_version": PAIR_SCHEMA_VERSION,
                **{key: value for key, value in derived_pair.items() if key != "text"},
            }
            if observed_pair != expected_pair:
                raise ValueError("score pair differs from semantic derivation")
            identity = (
                str(derived_pair["topic_id"]),
                str(derived_pair["query_id"]),
                str(derived_pair["document_id"]),
            )
            if identity in pair_identities:
                raise ValueError("semantic pair identity is duplicated")
            pair_identities.add(identity)
            for planned in _authorized_window_plan(derived_pair, prepared):
                cached = cache.get(  # type: ignore[attr-defined]
                    query_text=planned.query, text=planned.window_text
                ) is not None
                expected_window = replace(planned, cache_hit=cached).to_dict()
                expected_window["query_id"] = derived_pair["query_id"]
                if "facet_id" in derived_pair:
                    expected_window["facet_id"] = derived_pair["facet_id"]
                    expected_window["manifest_order"] = derived_pair["manifest_order"]
                observed_window = next(window_iterator, None)
                if observed_window != expected_window:
                    raise ValueError("score window differs from semantic derivation")
                window_id = str(planned.window_id)
                if window_id in window_ids:
                    raise ValueError("semantic window identity is duplicated")
                window_ids.add(window_id)
                cache_keys.add(planned.cache_key)
                if cached:
                    hits += 1
                else:
                    misses += 1
                    unique_misses.add(planned.cache_key)
        if next(pair_iterator, None) is not None or next(window_iterator, None) is not None:
            raise ValueError("score plan has rows beyond semantic derivation")
    finally:
        _window_module.score_cache_context = original_context_factory
    if (
        len(pair_identities) != payload.get("pair_count")
        or len(window_ids) != payload.get("window_count")
        or hits != payload.get("cache_hit_count")
        or misses != payload.get("cache_miss_count")
        or len(cache_keys) != payload.get("unique_pair_count")
        or len(unique_misses) != payload.get("unique_cache_miss_count")
    ):
        raise ValueError("semantic replay counters differ")
    return pair_identities, window_ids


def verify_preflight(
    scoring_dir: Path,
    *,
    expected_root_sha256: str = APPROVED_SCORE_PLAN_ROOT_SHA256,
    semantic_replay: bool = True,
    authenticate_upstream: bool = True,
    tokenizer: object | None = None,
    cache: object | None = None,
) -> dict[str, object]:
    """Verify the sealed tokenizer plan without opening a model or qrels."""

    scoring = Path(scoring_dir)
    payload, preflight_source = _read_json(scoring / "preflight.json", "score preflight")
    seal, _ = _read_json(scoring / "SCORE_PLAN_SEALED.json", "score-plan seal")
    files = seal.get("files")
    if (
        seal.get("schema_version") != PLAN_SEAL_SCHEMA_VERSION
        or seal.get("experiment_id") != EXPERIMENT_ID
        or not isinstance(files, Mapping)
        or set(files) != {"pairs.jsonl", "windows.jsonl", "preflight.json"}
        or seal.get("root_sha256") != _sha256_bytes(_compact_bytes(files))
        or seal.get("root_sha256") != expected_root_sha256
    ):
        raise ValueError("score-plan seal is invalid")
    for name, binding in files.items():
        if not isinstance(binding, Mapping) or dict(binding) != _file_binding(scoring / name):
            raise ValueError(f"score-plan seal differs for {name}")
    artifacts = payload.get("artifacts")
    runtime = payload.get("runtime_projection")
    sources = payload.get("sources")
    tokenizer_evidence = payload.get("tokenizer")
    cache_evidence = payload.get("score_cache")
    if (
        payload.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or payload.get("status") != "tokenizer_only_preflight_complete"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("topic_ids") != list(ALL_TOPIC_IDS)
        or payload.get("model") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("pair_count") != EXPECTED_PAIR_COUNT
        or payload.get("facet_association_count") != EXPECTED_FACET_ASSOCIATIONS
        or payload.get("inventory_pair_delta") != 0
        or payload.get("external_calls") != _ZERO_EXTERNAL_CALLS
        or payload.get("model_constructed") is not False
        or payload.get("inference_authorized") is not False
        or not isinstance(sources, Mapping)
        or sources.get("planning_root_sha256") != PLANNING_ROOT_SHA256
        or sources.get("retrieval_root_sha256") != RETRIEVAL_ROOT_SHA256
        or not isinstance(sources.get("manifest"), Mapping)
        or sources["manifest"].get("sha256") != MANIFEST_SHA256  # type: ignore[union-attr]
        or not isinstance(sources.get("accepted_union"), Mapping)
        or sources["accepted_union"].get("sha256") != ACCEPTED_UNION_SHA256  # type: ignore[union-attr]
        or not isinstance(tokenizer_evidence, Mapping)
        or tokenizer_evidence.get("load_count") != 1
        or tokenizer_evidence.get("local_files_only") is not True
        or not isinstance(cache_evidence, Mapping)
        or cache_evidence.get("context") != score_cache_context().artifact_metadata
        or not isinstance(artifacts, Mapping)
        or artifacts.get("pairs.jsonl") != _file_binding(scoring / "pairs.jsonl")
        or artifacts.get("windows.jsonl") != _file_binding(scoring / "windows.jsonl")
        or not isinstance(runtime, Mapping)
        or runtime.get("projected_unique_cache_miss_count")
        != payload.get("unique_cache_miss_count")
    ):
        raise ValueError("score preflight counters or safety evidence differ")
    if files["preflight.json"] != {  # type: ignore[index]
        "bytes": len(preflight_source),
        "sha256": _sha256_bytes(preflight_source),
    }:
        raise ValueError("score preflight byte binding differs")
    materialization = load_verified_materialization(
        Path(str(payload.get("model_materialization_receipt")))
    )
    if materialization.sha256 != payload.get("model_materialization_receipt_sha256"):
        raise ValueError("model materialization differs from the approved score plan")
    if semantic_replay:
        pair_identities, window_ids = _replay_plan_derivation(
            scoring,
            payload,
            tokenizer=tokenizer,
            cache=cache,
            authenticate_upstream=authenticate_upstream,
        )
    else:
        pair_identities, window_ids = _validate_plan_rows(scoring, payload)
    pair_count = len(pair_identities)
    window_count = len(window_ids)
    return {
        "verified": True,
        "root_sha256": seal["root_sha256"],
        "pair_count": pair_count,
        "window_count": window_count,
        "cache_hit_count": payload["cache_hit_count"],
        "cache_miss_count": payload["cache_miss_count"],
        "unique_cache_miss_count": payload["unique_cache_miss_count"],
        "projected_runtime_seconds": runtime["projected_runtime_seconds"],
        "estimated_peak_device_memory_bytes": runtime["estimated_peak_device_memory_bytes"],
        "estimated_peak_host_memory_bytes": runtime["estimated_peak_host_memory_bytes"],
        "external_calls": dict(_ZERO_EXTERNAL_CALLS),
    }


def _float32(value: object) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("MiniLM score must be finite")
    return struct.unpack(">f", struct.pack(">f", converted))[0]


def _aggregate_query_documents(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        topic_id = row.get("topic_id")
        query_id = row.get("query_id")
        document_id = row.get("document_id")
        if (
            topic_id not in ALL_TOPIC_IDS
            or not isinstance(query_id, str)
            or not query_id
            or not isinstance(document_id, str)
            or not document_id
        ):
            raise ValueError("scored window query/document identity is invalid")
        grouped[(str(topic_id), query_id, document_id)].append(row)
    output: list[dict[str, object]] = []
    for (topic_id, query_id, document_id), windows in grouped.items():
        variants = {row.get("variant") for row in windows}
        query_hashes = {row.get("query_sha256") for row in windows}
        text_hashes = {row.get("document_sha256") for row in windows}
        if len(variants) != 1 or len(query_hashes) != 1 or len(text_hashes) != 1:
            raise ValueError("document windows differ on query/text lineage")
        selected = _selected_top4(windows)
        output.append(
            {
                "topic_id": topic_id,
                "query_id": query_id,
                "variant": next(iter(variants)),
                "document_id": document_id,
                "score": aggregate_top4(windows),
                "selected_window_count": len(selected),
                "selected_window_ids": [str(row["window_id"]) for row in selected],
                "query_sha256": next(iter(query_hashes)),
                "text_sha256": next(iter(text_hashes)),
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
    return output


def _load_score_ledger(
    path: Path,
) -> tuple[dict[str, float], Counter[str], dict[str, tuple[str, str]]]:
    scores: dict[str, float] = {}
    sources: Counter[str] = Counter()
    lineage: dict[str, tuple[str, str]] = {}
    if not path.exists():
        return scores, sources, lineage
    for row in _iter_jsonl(path, "run-local score ledger"):
        cache_key = row.get("cache_key")
        score = row.get("score")
        source = row.get("source")
        query_sha = row.get("query_sha256")
        window_sha = row.get("window_sha256")
        if (
            row.get("schema_version") != SCORE_LEDGER_SCHEMA_VERSION
            or not isinstance(cache_key, str)
            or len(cache_key) != 64
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or source not in {"preflight_cache", "local_runner"}
            or not isinstance(query_sha, str)
            or len(query_sha) != 64
            or not isinstance(window_sha, str)
            or len(window_sha) != 64
        ):
            raise ValueError("run-local score ledger row is invalid")
        value = _float32(score)
        if cache_key in scores:
            if scores[cache_key] != value:
                raise ValueError("run-local score ledger has conflicting duplicates")
            continue
        scores[cache_key] = value
        lineage[cache_key] = (query_sha, window_sha)
        sources[str(source)] += 1
    return scores, sources, lineage


def _append_score_ledger(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as sink:
        for row in rows:
            sink.write(_compact_bytes(row) + b"\n")
        sink.flush()
        os.fsync(sink.fileno())


def _flush_file(handle: object) -> None:
    handle.flush()  # type: ignore[attr-defined]
    os.fsync(handle.fileno())  # type: ignore[attr-defined]


def run_scores(
    scoring_dir: Path,
    *,
    cache_root: Path = SCORE_CACHE_ROOT,
    runner: object | None = None,
    publish_hook: object | None = None,
) -> dict[str, object]:
    """Resume local scores into a run ledger, then publish terminal files."""

    scoring = Path(scoring_dir)
    preflight_verification = verify_preflight(scoring)
    terminal_names = (
        "scores.jsonl", "document_scores.jsonl", "features.jsonl",
        "scoring_receipt.json", "SCORING_SEALED.json",
    )
    if (scoring / "SCORING_SEALED.json").exists():
        verify_scores(scoring)
        return _read_json(scoring / "scoring_receipt.json", "scoring receipt")[0]
    # A crash during terminal publication is recoverable from the ledger.  No
    # unsealed terminal file is authoritative.
    for name in terminal_names:
        path = scoring / name
        if path.exists():
            path.unlink()
    preflight, preflight_source = _read_json(scoring / "preflight.json", "score preflight")
    cache = GlobalScoreCache(Path(cache_root), score_cache_context())
    if _cache_binding(cache) != preflight["score_cache"]["binding"]:  # type: ignore[index]
        raise ValueError("score cache changed after preflight")
    ledger_path = scoring / "score_ledger.jsonl"
    ledger_scores, ledger_sources, ledger_lineage = _load_score_ledger(ledger_path)
    scorer = runner
    started = time.perf_counter()
    seen_keys: set[str] = set()
    batch: list[Mapping[str, object]] = []
    cached_ledger_batch: list[dict[str, object]] = []

    def flush_cached_ledger() -> None:
        if not cached_ledger_batch:
            return
        _append_score_ledger(ledger_path, cached_ledger_batch)
        for row in cached_ledger_batch:
            key = str(row["cache_key"])
            ledger_scores[key] = float(row["score"])
            ledger_lineage[key] = (
                str(row["query_sha256"]), str(row["window_sha256"])
            )
            ledger_sources["preflight_cache"] += 1
        cached_ledger_batch.clear()

    def flush_batch() -> None:
        nonlocal scorer
        if not batch:
            return
        if scorer is None:
            scorer = _LocalMiniLMRunner(
                Path(str(preflight["model_materialization_receipt"]))
            )
        values = scorer.score(batch)  # type: ignore[union-attr]
        if len(values) != len(batch):
            raise ValueError("MiniLM runner returned the wrong score count")
        ledger_rows = [
            {
                "schema_version": SCORE_LEDGER_SCHEMA_VERSION,
                "cache_key": str(row["cache_key"]),
                "query_sha256": str(row["query_sha256"]),
                "window_sha256": str(row["window_sha256"]),
                "score": _float32(value),
                "source": "local_runner",
            }
            for row, value in zip(batch, values, strict=True)
        ]
        _append_score_ledger(ledger_path, ledger_rows)
        for row in ledger_rows:
            ledger_scores[str(row["cache_key"])] = float(row["score"])
            ledger_lineage[str(row["cache_key"])] = (
                str(row["query_sha256"]), str(row["window_sha256"])
            )
            ledger_sources["local_runner"] += 1
        batch.clear()

    for row in _iter_jsonl(scoring / "windows.jsonl", "score windows"):
        key = str(row["cache_key"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if key in ledger_scores:
            if ledger_lineage[key] != (
                str(row["query_sha256"]), str(row["window_sha256"])
            ):
                raise ValueError("run-local ledger lineage differs from score plan")
            continue
        cached = cache.get(query_text=str(row["query"]), text=str(row["window_text"]))
        if cached is not None:
            ledger_row = {
                "schema_version": SCORE_LEDGER_SCHEMA_VERSION,
                "cache_key": key,
                "query_sha256": str(row["query_sha256"]),
                "window_sha256": str(row["window_sha256"]),
                "score": _float32(cached),
                "source": "preflight_cache",
            }
            cached_ledger_batch.append(ledger_row)
            if len(cached_ledger_batch) == BATCH_SIZE:
                flush_cached_ledger()
        else:
            batch.append(row)
            if len(batch) == BATCH_SIZE:
                flush_batch()
    flush_batch()
    flush_cached_ledger()
    if len(ledger_scores) != preflight.get("unique_pair_count"):
        raise ValueError("run-local ledger does not cover every unique planned pair")

    stage = scoring / ".score-stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    document_rows: list[dict[str, object]] = []
    score_count = 0
    current_identity: tuple[str, str, str] | None = None
    current_rows: list[dict[str, object]] = []
    scores_path = stage / "scores.jsonl"
    documents_path = stage / "document_scores.jsonl"
    with scores_path.open("xb") as score_sink, documents_path.open("xb") as document_sink:
        for planned in _iter_jsonl(scoring / "windows.jsonl", "score windows"):
            key = str(planned["cache_key"])
            if key not in ledger_scores:
                raise ValueError("run-local ledger lacks a planned window score")
            scored = {
                **planned,
                "schema_version": SCORE_SCHEMA_VERSION,
                "score": ledger_scores[key],
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
            score_sink.write(_compact_bytes(scored) + b"\n")
            score_count += 1
            identity = (
                str(scored["topic_id"]),
                str(scored["query_id"]),
                str(scored["document_id"]),
            )
            if current_identity is not None and identity != current_identity:
                document = {
                    "schema_version": DOCUMENT_SCORE_SCHEMA_VERSION,
                    **_aggregate_query_documents(current_rows)[0],
                }
                document_sink.write(_compact_bytes(document) + b"\n")
                document_rows.append(document)
                current_rows = []
            current_identity = identity
            current_rows.append(scored)
        if current_rows:
            document = {
                "schema_version": DOCUMENT_SCORE_SCHEMA_VERSION,
                **_aggregate_query_documents(current_rows)[0],
            }
            document_sink.write(_compact_bytes(document) + b"\n")
            document_rows.append(document)
        _flush_file(score_sink)
        _flush_file(document_sink)
    feature_rows = [
        {**row, "schema_version": FEATURE_SCHEMA_VERSION}
        for row in percentile_features(document_rows)
    ]
    features_path = stage / "features.jsonl"
    with features_path.open("xb") as feature_sink:
        for row in feature_rows:
            feature_sink.write(_compact_bytes(row) + b"\n")
        _flush_file(feature_sink)
    elapsed = time.perf_counter() - started
    output_names = ("scores.jsonl", "document_scores.jsonl", "features.jsonl")
    receipt = {
        "schema_version": SCORING_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "preflight_sha256": _sha256_bytes(preflight_source),
        "score_plan_root_sha256": preflight_verification["root_sha256"],
        "planned_window_count": int(preflight["window_count"]),
        "completed_window_count": score_count,
        "document_score_count": len(document_rows),
        "feature_count": len(feature_rows),
        "unique_forward_pair_count": ledger_sources["local_runner"],
        "cache_reuse_pair_count": ledger_sources["preflight_cache"],
        "elapsed_seconds": elapsed,
        "peak_device_memory_bytes": int(getattr(scorer, "peak_device_memory_bytes", 0)),
        "peak_host_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "device": "cuda",
        "execution_backend": "rocm",
        "network_calls": 0,
        "hosted_calls": 0,
        "paid_calls": 0,
        "qrels_opened": False,
        "ledger": _file_binding(ledger_path),
        "shared_cache_mutated": False,
        "artifacts": {name: _file_binding(stage / name) for name in output_names},
    }
    _write_exclusive(stage / "scoring_receipt.json", _pretty_bytes(receipt))
    files = {
        name: _file_binding(stage / name)
        for name in (*output_names, "scoring_receipt.json")
    }
    files["score_ledger.jsonl"] = _file_binding(ledger_path)
    seal = {
        "schema_version": SCORING_SEAL_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "score_plan_root_sha256": preflight_verification["root_sha256"],
        "files": files,
        "root_sha256": _sha256_bytes(_compact_bytes(files)),
    }
    _write_exclusive(stage / "SCORING_SEALED.json", _pretty_bytes(seal))
    for name in (*output_names, "scoring_receipt.json"):
        os.replace(stage / name, scoring / name)
        if callable(publish_hook):
            publish_hook(name)
    os.replace(stage / "SCORING_SEALED.json", scoring / "SCORING_SEALED.json")
    stage.rmdir()
    return receipt


def verify_scores(scoring_dir: Path) -> dict[str, object]:
    """Verify terminal local-score coverage and its sealed percentile features."""

    scoring = Path(scoring_dir)
    preflight = verify_preflight(scoring)
    preflight_payload, preflight_source = _read_json(
        scoring / "preflight.json", "score preflight"
    )
    receipt, receipt_source = _read_json(
        scoring / "scoring_receipt.json", "scoring receipt"
    )
    seal, _ = _read_json(scoring / "SCORING_SEALED.json", "scoring seal")
    files = seal.get("files")
    if (
        seal.get("schema_version") != SCORING_SEAL_SCHEMA_VERSION
        or seal.get("experiment_id") != EXPERIMENT_ID
        or seal.get("score_plan_root_sha256") != preflight["root_sha256"]
        or not isinstance(files, Mapping)
        or set(files)
        != {
            "scores.jsonl",
            "document_scores.jsonl",
            "features.jsonl",
            "scoring_receipt.json",
            "score_ledger.jsonl",
        }
        or seal.get("root_sha256") != _sha256_bytes(_compact_bytes(files))
    ):
        raise ValueError("scoring seal is invalid")
    for name, binding in files.items():
        if not isinstance(binding, Mapping) or dict(binding) != _file_binding(scoring / name):
            raise ValueError(f"scoring seal differs for {name}")
    artifacts = receipt.get("artifacts")
    if (
        receipt.get("status") != "complete"
        or receipt.get("schema_version") != SCORING_RECEIPT_SCHEMA_VERSION
        or receipt.get("preflight_sha256") != _sha256_bytes(preflight_source)
        or receipt.get("planned_window_count") != preflight["window_count"]
        or receipt.get("completed_window_count") != preflight["window_count"]
        or receipt.get("document_score_count") != EXPECTED_PAIR_COUNT
        or receipt.get("feature_count") != EXPECTED_PAIR_COUNT
        or receipt.get("network_calls") != 0
        or receipt.get("hosted_calls") != 0
        or receipt.get("paid_calls") != 0
        or receipt.get("qrels_opened") is not False
        or receipt.get("shared_cache_mutated") is not False
        or receipt.get("score_plan_root_sha256") != preflight["root_sha256"]
        or receipt.get("ledger") != _file_binding(scoring / "score_ledger.jsonl")
        or not isinstance(artifacts, Mapping)
        or set(artifacts)
        != {"scores.jsonl", "document_scores.jsonl", "features.jsonl"}
        or any(
            not isinstance(binding, Mapping)
            or dict(binding) != _file_binding(scoring / name)
            for name, binding in artifacts.items()
        )
        or files.get("scoring_receipt.json")
        != {"bytes": len(receipt_source), "sha256": _sha256_bytes(receipt_source)}
    ):
        raise ValueError("scoring coverage or safety evidence differs")
    pair_identities, planned_window_ids = _validate_plan_rows(
        scoring, preflight_payload
    )
    ledger_scores, ledger_sources, ledger_lineage = _load_score_ledger(
        scoring / "score_ledger.jsonl"
    )
    if (
        len(ledger_scores) != preflight_payload.get("unique_pair_count")
        or ledger_sources["local_runner"] != receipt.get("unique_forward_pair_count")
        or ledger_sources["preflight_cache"] != receipt.get("cache_reuse_pair_count")
    ):
        raise ValueError("run-local ledger coverage differs")
    scored_window_ids: set[str] = set()
    recomputed_documents: list[dict[str, object]] = []
    current_identity: tuple[str, str, str] | None = None
    current_rows: list[dict[str, object]] = []
    planned_rows = _iter_jsonl(scoring / "windows.jsonl", "score windows")
    score_rows = _iter_jsonl(scoring / "scores.jsonl", "window scores")
    for planned, row in zip_longest(planned_rows, score_rows):
        if planned is None or row is None:
            raise ValueError("window score row count differs from plan")
        window_id = row.get("window_id")
        score = row.get("score")
        expected_lineage = {
            **planned,
            "schema_version": SCORE_SCHEMA_VERSION,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
        observed_lineage = {key: value for key, value in row.items() if key != "score"}
        if (
            row.get("schema_version") != SCORE_SCHEMA_VERSION
            or window_id not in planned_window_ids
            or window_id in scored_window_ids
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or row.get("model") != MODEL_ID
            or row.get("model_revision") != MODEL_REVISION
            or observed_lineage != expected_lineage
            or ledger_scores.get(str(row.get("cache_key"))) != _float32(score)
            or ledger_lineage.get(str(row.get("cache_key")))
            != (str(row.get("query_sha256")), str(row.get("window_sha256")))
        ):
            raise ValueError("window score identity/coverage is invalid")
        scored_window_ids.add(str(window_id))
        identity = (
            str(row["topic_id"]), str(row["query_id"]), str(row["document_id"])
        )
        if current_identity is not None and identity != current_identity:
            recomputed_documents.append(
                {
                    "schema_version": DOCUMENT_SCORE_SCHEMA_VERSION,
                    **_aggregate_query_documents(current_rows)[0],
                }
            )
            current_rows = []
        current_identity = identity
        current_rows.append(row)
    if current_rows:
        recomputed_documents.append(
            {
                "schema_version": DOCUMENT_SCORE_SCHEMA_VERSION,
                **_aggregate_query_documents(current_rows)[0],
            }
        )
    if scored_window_ids != planned_window_ids:
        raise ValueError("window score coverage differs from the plan")

    document_identities: set[tuple[str, str, str]] = set()
    for expected, row in zip_longest(
        recomputed_documents,
        _iter_jsonl(scoring / "document_scores.jsonl", "document scores"),
    ):
        if expected is None or row is None or row != expected:
            raise ValueError("document score differs from aggregate_top4 recomputation")
        identity = (
            str(row.get("topic_id")),
            str(row.get("query_id")),
            str(row.get("document_id")),
        )
        score = row.get("score")
        if (
            row.get("schema_version") != DOCUMENT_SCORE_SCHEMA_VERSION
            or identity not in pair_identities
            or identity in document_identities
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("document score identity/coverage is invalid")
        document_identities.add(identity)
    if document_identities != pair_identities:
        raise ValueError("document score coverage differs from the pair plan")

    recomputed_features = [
        {**row, "schema_version": FEATURE_SCHEMA_VERSION}
        for row in percentile_features(recomputed_documents)
    ]
    feature_identities: set[tuple[str, str, str]] = set()
    for expected, row in zip_longest(
        recomputed_features,
        _iter_jsonl(scoring / "features.jsonl", "percentile features"),
    ):
        if expected is None or row is None or row != expected:
            raise ValueError("percentile feature differs from recomputation")
        identity = (
            str(row.get("topic_id")),
            str(row.get("query_id")),
            str(row.get("document_id")),
        )
        percentile = row.get("percentile")
        if (
            row.get("schema_version") != FEATURE_SCHEMA_VERSION
            or identity not in document_identities
            or identity in feature_identities
            or isinstance(percentile, bool)
            or not isinstance(percentile, (int, float))
            or not math.isfinite(float(percentile))
            or not 0.0 <= float(percentile) <= 1.0
        ):
            raise ValueError("percentile feature identity/coverage is invalid")
        feature_identities.add(identity)
    if feature_identities != document_identities:
        raise ValueError("percentile feature coverage differs from document scores")
    return {
        "verified": True,
        "root_sha256": seal["root_sha256"],
        "window_count": receipt["completed_window_count"],
        "feature_count": receipt["feature_count"],
        "hosted_paid_network_qrels_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--retrieval", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)
    preflight.add_argument("--planning", type=Path)
    preflight.add_argument("--model-receipt", type=Path, default=DEFAULT_MODEL_RECEIPT)
    preflight.add_argument("--cache-root", type=Path, default=SCORE_CACHE_ROOT)
    verify_preflight_parser = subparsers.add_parser("verify-preflight")
    verify_preflight_parser.add_argument("--scoring", required=True, type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--scoring", required=True, type=Path)
    run.add_argument("--cache-root", type=Path, default=SCORE_CACHE_ROOT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--scoring", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = create_preflight(
            args.retrieval,
            args.output,
            planning_dir=args.planning,
            model_receipt=args.model_receipt,
            cache_root=args.cache_root,
        )
    elif args.command == "verify-preflight":
        result = verify_preflight(args.scoring)
    elif args.command == "run":
        result = run_scores(args.scoring, cache_root=args.cache_root)
    else:
        result = verify_scores(args.scoring)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
