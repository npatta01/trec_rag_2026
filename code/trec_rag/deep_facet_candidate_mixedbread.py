"""Protected-head Mixedbread ranking for complete downstream RAG candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from .chunking import ChunkingConfig, SemanticTextChunker
from .deep_facet_candidate_evaluate import evaluate_ranking
from .deep_facet_candidate_manifest import TOPIC_IDS
from .deep_facet_candidate_rank import ARM_NAMES, verify_seal
from .rerank_score_cache import (
    GlobalScoreCache,
    ScoreCacheContext,
    _predict,
    _validate_model_dtype,
    global_score_cache_dir,
)


PROTECTED_TOPIC_IDS = ("144", "213", "224", "407", "515")
RRF_HEAD_DEPTH = 10
MIXEDBREAD_END_DEPTH = 500
RRF_POOL_DEPTH = 500
GLOBAL_POOL_DEPTH = 500
DUAL_POOL_DEPTH = 1000
TOP4_WEIGHTS = (0.55, 0.25, 0.13, 0.07)
MODEL_ID = "mixedbread-ai/mxbai-rerank-base-v2"
MODEL_REVISION = "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
BACKEND_VERSION = "5.6.0"
MODEL_DTYPE = "bfloat16"
MODEL_BACKEND = "sentence-transformers-cross-encoder"
SCORE_REPRESENTATION = "raw_logits"
INPUT_POLICY = "trec_rag_raw_v2"
MAX_LENGTH = 1024
CHUNK_MAX_CHARACTERS = 3500
CHUNK_OVERLAP_CHARACTERS = 350
BATCH_SIZE = 32
ARM = "RRF-MIXEDBREAD-DUAL"
SCHEMA_VERSION = "deep-facet-candidate-mixedbread-v1"
MODEL_SNAPSHOT = Path(
    "/home/npatta01/.cache/huggingface/hub/"
    "models--mixedbread-ai--mxbai-rerank-base-v2/snapshots/"
    + MODEL_REVISION
)

T = TypeVar("T")


def build_structured_query(narrative: str, obligations: Sequence[str]) -> str:
    """Render one comparable narrative-plus-facets query for a topic."""

    narrative = narrative.strip()
    cleaned = tuple(obligation.strip() for obligation in obligations)
    if not narrative:
        raise ValueError("narrative is required")
    if not cleaned or any(not obligation for obligation in cleaned):
        raise ValueError("at least one non-empty facet obligation is required")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("duplicate facet obligation")
    items = "\n".join(
        f"{index}. {obligation}" for index, obligation in enumerate(cleaned, start=1)
    )
    return f"Narrative:\n{narrative}\n\nInformation needs:\n{items}"


def reject_protected_topics_before_access(
    topic_ids: Iterable[str], source_loader: Callable[[], T]
) -> T:
    """Reject protected identities before invoking a source loader or join."""

    for topic_id in map(str, topic_ids):
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
    return source_loader()


def _validate_order(order: Sequence[str], *, label: str) -> tuple[str, ...]:
    result = tuple(map(str, order))
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate documents")
    return result


def build_residual_pool(
    rrf: Sequence[str], global_order: Sequence[str], dual: Sequence[str]
) -> tuple[str, ...]:
    """Return the frozen disagreement union in DUAL tie-break order."""

    rrf_order = _validate_order(rrf, label="RRF")
    global_ranked = _validate_order(global_order, label="GLOBAL")
    dual_order = _validate_order(dual, label="DUAL")
    populations = (set(rrf_order), set(global_ranked), set(dual_order))
    if populations[0] != populations[1] or populations[0] != populations[2]:
        raise ValueError("RRF, GLOBAL, and DUAL must contain the same complete population")
    protected = set(rrf_order[:RRF_HEAD_DEPTH])
    selected = (
        set(rrf_order[:RRF_POOL_DEPTH])
        | set(global_ranked[:GLOBAL_POOL_DEPTH])
        | set(dual_order[:DUAL_POOL_DEPTH])
    ) - protected
    return tuple(document_id for document_id in dual_order if document_id in selected)


def aggregate_top4(scores: Sequence[float]) -> float:
    """Aggregate the strongest four window logits with the frozen weights."""

    values = tuple(float(score) for score in scores)
    if not values:
        raise ValueError("at least one window score is required")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("window scores must be finite")
    strongest = sorted(values, reverse=True)[: len(TOP4_WEIGHTS)]
    weights = TOP4_WEIGHTS[: len(strongest)]
    return sum(score * weight for score, weight in zip(strongest, weights, strict=True)) / sum(
        weights
    )


def assemble_complete_ranking(
    rrf: Sequence[str], dual: Sequence[str], residual_scores: dict[str, float]
) -> tuple[str, ...]:
    """Protect the RRF head, rerank the middle, and append the complete DUAL tail."""

    rrf_order = _validate_order(rrf, label="RRF")
    dual_order = _validate_order(dual, label="DUAL")
    if set(rrf_order) != set(dual_order):
        raise ValueError("RRF and DUAL must contain the same complete population")
    if len(rrf_order) < MIXEDBREAD_END_DEPTH:
        raise ValueError(f"candidate population must contain at least {MIXEDBREAD_END_DEPTH} documents")
    head = tuple(rrf_order[:RRF_HEAD_DEPTH])
    head_set = set(head)
    unknown = set(residual_scores) - set(rrf_order)
    if unknown:
        raise ValueError("residual scores contain documents outside the complete population")
    if head_set.intersection(residual_scores):
        raise ValueError("residual scores must exclude the protected RRF head")
    if any(not math.isfinite(float(score)) for score in residual_scores.values()):
        raise ValueError("residual scores contain a non-finite value")
    dual_rank = {document_id: rank for rank, document_id in enumerate(dual_order)}
    middle_count = MIXEDBREAD_END_DEPTH - RRF_HEAD_DEPTH
    middle = tuple(
        sorted(
            residual_scores,
            key=lambda document_id: (-float(residual_scores[document_id]), dual_rank[document_id]),
        )[:middle_count]
    )
    if len(middle) != middle_count:
        raise ValueError(f"residual scores must cover at least {middle_count} documents")
    selected = head_set | set(middle)
    tail = tuple(document_id for document_id in dual_order if document_id not in selected)
    ranking = head + middle + tail
    if len(ranking) != len(rrf_order) or set(ranking) != set(rrf_order):
        raise AssertionError("complete-ranking invariant failed")
    return ranking


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _exclusive_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as sink:
        sink.write(value)
        sink.flush()
        os.fsync(sink.fileno())


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        source = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    with source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label}:{line_number} is invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label}:{line_number} must be an object")
            rows.append(row)
    return rows


def _artifact(path: Path) -> dict[str, object]:
    value = path.read_bytes()
    return {"bytes": len(value), "sha256": _sha256_bytes(value)}


def _validate_model_materialization() -> dict[str, object]:
    if importlib.metadata.version("sentence-transformers") != BACKEND_VERSION:
        raise ValueError("sentence-transformers backend version differs")
    if not MODEL_SNAPSHOT.is_dir() or not (MODEL_SNAPSHOT / "model.safetensors").is_file():
        raise ValueError("pinned Mixedbread model is not materialized locally")
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_path": str(MODEL_SNAPSHOT),
        "model_safetensors": _artifact(MODEL_SNAPSHOT / "model.safetensors"),
        "backend_version": BACKEND_VERSION,
    }


def _load_source_orders(freeze_dir: Path) -> dict[str, dict[str, list[str]]]:
    verify_seal(freeze_dir)
    collected: dict[str, dict[str, list[tuple[int, str]]]] = {
        topic: {arm: [] for arm in ("RRF", "GLOBAL", "DUAL")} for topic in TOPIC_IDS
    }
    for row in _read_jsonl(freeze_dir / "rankings.jsonl", "source rankings"):
        topic_id = str(row.get("topic_id"))
        arm = str(row.get("arm"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id not in collected or arm not in ARM_NAMES:
            raise ValueError("source rankings contain an unexpected topic or arm")
        if arm in collected[topic_id]:
            collected[topic_id][arm].append((int(row["rank"]), str(row["document_id"])))
    result: dict[str, dict[str, list[str]]] = {}
    for topic_id, arms in collected.items():
        result[topic_id] = {}
        for arm, values in arms.items():
            ordered = sorted(values)
            if [rank for rank, _ in ordered] != list(range(1, len(ordered) + 1)):
                raise ValueError(f"source {arm} ranking for topic {topic_id} is not contiguous")
            result[topic_id][arm] = [document_id for _, document_id in ordered]
        populations = [set(order) for order in result[topic_id].values()]
        if not populations[0] or any(population != populations[0] for population in populations[1:]):
            raise ValueError("source ranking arms do not share one complete population")
    return result


def _load_queries(
    manifest_path: Path, accepted_facet_ids: set[str]
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    manifest = _read_object(manifest_path, "experiment manifest")
    topic_rows = manifest.get("topics")
    facet_rows = manifest.get("facets")
    if not isinstance(topic_rows, list) or not isinstance(facet_rows, list):
        raise ValueError("manifest lacks topics or facets")
    topic_ids = tuple(str(row.get("topic_id")) for row in topic_rows if isinstance(row, Mapping))
    if topic_ids != TOPIC_IDS:
        raise ValueError("manifest topic order differs from the frozen pilot")
    for topic_id in topic_ids:
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
    narratives = {
        str(row["topic_id"]): str(row["query"])
        for row in topic_rows
        if isinstance(row, Mapping)
    }
    obligations: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    manifest_facet_ids: set[str] = set()
    for row in facet_rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest facet must be an object")
        topic_id = str(row.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id not in TOPIC_IDS:
            raise ValueError("manifest facet has an unexpected topic")
        facet_id = str(row.get("facet_id"))
        if facet_id in manifest_facet_ids:
            raise ValueError("manifest contains duplicate facet IDs")
        manifest_facet_ids.add(facet_id)
        if facet_id in accepted_facet_ids:
            obligations[topic_id].append(
                (int(row["manifest_order"]), facet_id, str(row["obligation"]))
            )
    if accepted_facet_ids != manifest_facet_ids.intersection(accepted_facet_ids):
        raise ValueError("accepted facet IDs differ from the manifest")
    selected_ids = {
        facet_id for values in obligations.values() for _order, facet_id, _obligation in values
    }
    if selected_ids != accepted_facet_ids:
        raise ValueError("accepted facet IDs differ from the manifest")
    ordered = {
        topic_id: [obligation for _, _facet_id, obligation in sorted(obligations[topic_id])]
        for topic_id in topic_ids
    }
    ordered_ids = {
        topic_id: [facet_id for _, facet_id, _obligation in sorted(obligations[topic_id])]
        for topic_id in topic_ids
    }
    if any(not ordered[topic_id] for topic_id in topic_ids):
        raise ValueError("every topic requires at least one accepted facet")
    return (
        {
            topic_id: build_structured_query(narratives[topic_id], ordered[topic_id])
            for topic_id in topic_ids
        },
        ordered,
        ordered_ids,
    )


def _load_authenticated_gate(
    gate_dir: Path,
) -> tuple[dict[str, object], set[str], dict[str, object], dict[str, object]]:
    """Authenticate gate decisions and return their accepted facet identities."""

    gate_dir = Path(gate_dir)
    summary_path = gate_dir / "summary.json"
    gates_path = gate_dir / "gates.json"
    accepted_path = gate_dir / "u_accepted.jsonl"
    summary = _read_object(summary_path, "gate summary")
    gates_artifact = _artifact(gates_path)
    accepted_artifact = _artifact(accepted_path)
    artifacts = summary.get("artifacts")
    if (
        summary.get("status") != "complete"
        or summary.get("qrels_opened") is not False
        or not isinstance(artifacts, Mapping)
        or artifacts.get("gates.json") != gates_artifact
        or artifacts.get("u_accepted.jsonl") != accepted_artifact
    ):
        raise ValueError("gate summary does not authenticate decisions and U_accepted")
    gates = _read_object(gates_path, "facet gates").get("gates")
    if not isinstance(gates, list):
        raise ValueError("facet gates lack gate rows")
    seen: set[str] = set()
    accepted: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for row in gates:
        if not isinstance(row, Mapping):
            raise ValueError("facet gate row must be an object")
        topic_id = str(row.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id not in TOPIC_IDS:
            raise ValueError("facet gates contain an unexpected topic")
        facet_id = str(row.get("facet_id"))
        if facet_id in seen:
            raise ValueError("facet gates contain duplicate facet IDs")
        seen.add(facet_id)
        if row.get("accepted") is True:
            accepted.add(facet_id)
            counts[topic_id] += 1
        elif row.get("accepted") is not False:
            raise ValueError("facet gate acceptance must be boolean")
    topic_counts = summary.get("topic_counts")
    if (
        len(accepted) != summary.get("accepted_facet_count")
        or not isinstance(topic_counts, Mapping)
        or any(
            not isinstance(topic_counts.get(topic), Mapping)
            or topic_counts[topic].get("accepted_facets") != counts[topic]  # type: ignore[index]
            for topic in TOPIC_IDS
        )
    ):
        raise ValueError("facet gate counts differ from the authenticated summary")
    return summary, accepted, gates_artifact, accepted_artifact


def _score_cache_context() -> ScoreCacheContext:
    return ScoreCacheContext(
        backend=MODEL_BACKEND,
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        backend_version=BACKEND_VERSION,
        score_representation=SCORE_REPRESENTATION,
        inference_dtype=MODEL_DTYPE,
        input_policy=INPUT_POLICY,
        max_length=MAX_LENGTH,
        score_kind="window",
        requested_max_length=MAX_LENGTH,
        chunk_max_characters=CHUNK_MAX_CHARACTERS,
        chunk_overlap_characters=CHUNK_OVERLAP_CHARACTERS,
    )


def _score_cache() -> GlobalScoreCache:
    repo_root = Path(__file__).resolve().parents[2]
    return GlobalScoreCache(global_score_cache_dir(repo_root), _score_cache_context())


def _score_policy_identity() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": MODEL_BACKEND,
        "backend_version": BACKEND_VERSION,
        "score_representation": SCORE_REPRESENTATION,
        "inference_dtype": MODEL_DTYPE,
        "input_policy": INPUT_POLICY,
        "max_length": MAX_LENGTH,
        "score_kind": "window",
    }


def _validate_score_row(row: Mapping[str, object], window: Mapping[str, object]) -> float:
    window_fields = (
        "topic_id",
        "document_id",
        "chunk_index",
        "query_sha256",
        "window_text_sha256",
    )
    if any(row.get(field) != window.get(field) for field in window_fields):
        raise ValueError("score identity differs from its frozen window")
    if any(row.get(field) != value for field, value in _score_policy_identity().items()):
        raise ValueError("score model/policy identity differs")
    expected_source = "global_cache" if window.get("cache_hit") is True else "inference"
    if row.get("score_source") != expected_source:
        raise ValueError("score source differs from the frozen cache audit")
    score = float(row.get("score", float("nan")))
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    return score


def create_preflight(
    *,
    manifest_path: Path,
    freeze_dir: Path,
    gate_dir: Path,
    output_dir: Path,
    measured_pairs_per_second: float,
    benchmark_batch_size: int,
) -> dict[str, object]:
    """Create the exact qrels-free Mixedbread window plan and budget."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"create-only preflight already exists: {output_dir}")
    if measured_pairs_per_second <= 0 or benchmark_batch_size <= 0:
        raise ValueError("benchmark rate and batch size must be positive")
    gate_summary, accepted_facet_ids, gates_artifact, accepted_artifact = (
        _load_authenticated_gate(Path(gate_dir))
    )
    queries, obligations, obligation_ids = _load_queries(
        Path(manifest_path), accepted_facet_ids
    )
    source_orders = _load_source_orders(Path(freeze_dir))
    accepted_path = Path(gate_dir) / "u_accepted.jsonl"
    model = _validate_model_materialization()
    score_cache = _score_cache()
    cache_artifact = _artifact(score_cache.path) if score_cache.path.exists() else None
    residual = {
        topic: set(
            build_residual_pool(
                source_orders[topic]["RRF"],
                source_orders[topic]["GLOBAL"],
                source_orders[topic]["DUAL"],
            )
        )
        for topic in TOPIC_IDS
    }
    accepted: dict[tuple[str, str], dict[str, object]] = {}
    for row in _read_jsonl(accepted_path, "accepted union"):
        topic_id = str(row.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id not in residual:
            raise ValueError("accepted union contains an unexpected topic")
        document_id = str(row.get("document_id"))
        text = row.get("text")
        if not isinstance(text, str) or _sha256_text(text) != row.get("text_sha256"):
            raise ValueError("accepted union text identity differs")
        identity = (topic_id, document_id)
        if identity in accepted:
            raise ValueError("accepted union contains duplicate topic/document")
        accepted[identity] = row
    for topic in TOPIC_IDS:
        expected = set(source_orders[topic]["RRF"])
        actual = {document_id for row_topic, document_id in accepted if row_topic == topic}
        if actual != expected:
            raise ValueError("accepted union differs from the complete ranking population")
    chunker = SemanticTextChunker(
        ChunkingConfig(
            max_characters=CHUNK_MAX_CHARACTERS,
            overlap_characters=CHUNK_OVERLAP_CHARACTERS,
        )
    )
    candidate_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    topic_summary: dict[str, object] = {}
    for topic_id in TOPIC_IDS:
        dual_rank = {
            document_id: rank
            for rank, document_id in enumerate(source_orders[topic_id]["DUAL"], start=1)
        }
        topic_windows = 0
        for document_id in sorted(residual[topic_id], key=dual_rank.__getitem__):
            source = accepted[(topic_id, document_id)]
            text = str(source["text"])
            chunks = chunker.split_text(text, document_id=document_id)
            if not chunks:
                raise ValueError("residual candidate produced no passage windows")
            candidate_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "topic_id": topic_id,
                    "document_id": document_id,
                    "dual_rank": dual_rank[document_id],
                    "query_sha256": _sha256_text(queries[topic_id]),
                    "text_sha256": _sha256_text(text),
                    "chunk_count": len(chunks),
                }
            )
            for chunk_index, chunk in enumerate(chunks):
                cache_key = score_cache.cache_key(
                    query_text=queries[topic_id], text=chunk.text
                )
                window_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "topic_id": topic_id,
                        "document_id": document_id,
                        "dual_rank": dual_rank[document_id],
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "query": queries[topic_id],
                        "query_sha256": _sha256_text(queries[topic_id]),
                        "window_text": chunk.text,
                        "window_text_sha256": _sha256_text(chunk.text),
                        "score_cache_key": cache_key,
                        "cache_hit": cache_key in score_cache.scores,
                    }
                )
            topic_windows += len(chunks)
        topic_summary[topic_id] = {
            "accepted_document_count": len(source_orders[topic_id]["RRF"]),
            "residual_document_count": len(residual[topic_id]),
            "window_count": topic_windows,
            "obligation_count": len(obligations[topic_id]),
            "accepted_facet_ids": obligation_ids[topic_id],
            "accepted_facet_ids_sha256": _sha256_bytes(
                _canonical_bytes(obligation_ids[topic_id])
            ),
            "query_characters": len(queries[topic_id]),
            "query_sha256": _sha256_text(queries[topic_id]),
        }
    candidate_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in candidate_rows)
    window_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in window_rows)
    output_dir.mkdir(parents=True)
    _exclusive_bytes(output_dir / "candidates.jsonl", candidate_bytes)
    _exclusive_bytes(output_dir / "windows.jsonl", window_bytes)
    projected_seconds = len(window_rows) / measured_pairs_per_second
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "preflight_complete",
        "post_qrels_diagnostic": True,
        "qrels_read": False,
        "network_access": False,
        "retrieval_calls": 0,
        "paid_cost_usd": 0,
        "topic_ids": list(TOPIC_IDS),
        "arm": ARM,
        "model": model,
        "scoring_policy": {
            "score_representation": "raw_logits",
            "inference_dtype": MODEL_DTYPE,
            "max_length": MAX_LENGTH,
            "chunk_max_characters": CHUNK_MAX_CHARACTERS,
            "chunk_overlap_characters": CHUNK_OVERLAP_CHARACTERS,
            "aggregation": "chunk_top4_weighted",
            "top4_weights": list(TOP4_WEIGHTS),
            "batch_size": BATCH_SIZE,
        },
        "ranking_policy": {
            "protected_rrf_ranks": [1, RRF_HEAD_DEPTH],
            "mixedbread_ranks": [RRF_HEAD_DEPTH + 1, MIXEDBREAD_END_DEPTH],
            "complete_dual_tail_start": MIXEDBREAD_END_DEPTH + 1,
            "complete_permutation_required": True,
        },
        "benchmark": {
            "measured_pairs_per_second": measured_pairs_per_second,
            "batch_size": benchmark_batch_size,
            "projected_scoring_seconds": projected_seconds,
        },
        "summary": {
            "accepted_document_count": len(accepted),
            "residual_document_count": len(candidate_rows),
            "window_count": len(window_rows),
            "cache_hit_window_count": sum(bool(row["cache_hit"]) for row in window_rows),
            "uncached_window_count": sum(not bool(row["cache_hit"]) for row in window_rows),
            "topic": topic_summary,
        },
        "source": {
            "manifest": {"path": str(Path(manifest_path).resolve()), **_artifact(Path(manifest_path))},
            "source_freeze_dir_path": str(Path(freeze_dir).resolve()),
            "gate_dir_path": str(Path(gate_dir).resolve()),
            "source_seal_root_sha256": verify_seal(Path(freeze_dir))["root_sha256"],
            "source_rankings": _artifact(Path(freeze_dir) / "rankings.jsonl"),
            "gate_summary": _artifact(Path(gate_dir) / "summary.json"),
            "facet_gates": gates_artifact,
            "accepted_union": accepted_artifact,
            "score_cache": {
                "path": str(score_cache.path),
                "artifact": cache_artifact,
                "context": _score_cache_context().artifact_metadata,
            },
        },
        "artifacts": {
            "candidates.jsonl": {"bytes": len(candidate_bytes), "sha256": _sha256_bytes(candidate_bytes)},
            "windows.jsonl": {"bytes": len(window_bytes), "sha256": _sha256_bytes(window_bytes)},
        },
    }
    _exclusive_bytes(output_dir / "preflight.json", _pretty_bytes(payload))
    return payload


def _verify_preflight(preflight_dir: Path) -> dict[str, object]:
    payload = _read_object(preflight_dir / "preflight.json", "Mixedbread preflight")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "preflight_complete"
        or payload.get("qrels_read") is not False
        or payload.get("topic_ids") != list(TOPIC_IDS)
    ):
        raise ValueError("Mixedbread preflight contract differs")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Mixedbread preflight lacks artifacts")
    for name in ("candidates.jsonl", "windows.jsonl"):
        if artifacts.get(name) != _artifact(preflight_dir / name):
            raise ValueError(f"Mixedbread preflight {name} was mutated")
    materialized = _validate_model_materialization()
    model = payload.get("model")
    policy = payload.get("scoring_policy")
    if model != materialized or not isinstance(policy, Mapping):
        raise ValueError("Mixedbread preflight model identity differs")
    expected_policy = {
        "score_representation": SCORE_REPRESENTATION,
        "inference_dtype": MODEL_DTYPE,
        "max_length": MAX_LENGTH,
        "chunk_max_characters": CHUNK_MAX_CHARACTERS,
        "chunk_overlap_characters": CHUNK_OVERLAP_CHARACTERS,
        "aggregation": "chunk_top4_weighted",
        "top4_weights": list(TOP4_WEIGHTS),
        "batch_size": BATCH_SIZE,
    }
    if any(policy.get(field) != value for field, value in expected_policy.items()):
        raise ValueError("Mixedbread preflight scoring policy differs")
    _verify_preflight_semantics(preflight_dir, payload)
    return payload


def _verified_bound_score_cache(preflight: Mapping[str, object]) -> GlobalScoreCache:
    source = preflight.get("source")
    binding = source.get("score_cache") if isinstance(source, Mapping) else None
    if not isinstance(binding, Mapping):
        raise ValueError("Mixedbread preflight lacks a score-cache binding")
    cache = _score_cache()
    actual_artifact = _artifact(cache.path) if cache.path.exists() else None
    if (
        binding.get("path") != str(cache.path)
        or binding.get("artifact") != actual_artifact
        or binding.get("context") != _score_cache_context().artifact_metadata
    ):
        raise ValueError("Mixedbread score cache changed after preflight")
    return cache


def _verify_preflight_semantics(
    preflight_dir: Path, preflight: Mapping[str, object]
) -> None:
    """Recompute every candidate/window row from authenticated frozen sources."""

    source = preflight.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Mixedbread preflight lacks source bindings")
    manifest_binding = source.get("manifest")
    if not isinstance(manifest_binding, Mapping):
        raise ValueError("Mixedbread preflight lacks a manifest binding")
    try:
        manifest_path = Path(str(manifest_binding["path"]))
        source_freeze_dir = Path(str(source["source_freeze_dir_path"]))
        gate_dir = Path(str(source["gate_dir_path"]))
    except KeyError as exc:
        raise ValueError("Mixedbread preflight lacks a frozen source path") from exc
    gate_summary, accepted_facet_ids, gates_artifact, accepted_artifact = (
        _load_authenticated_gate(gate_dir)
    )
    queries, obligations, obligation_ids = _load_queries(
        manifest_path, accepted_facet_ids
    )
    source_orders = _load_source_orders(source_freeze_dir)
    if (
        dict(manifest_binding) != {"path": str(manifest_path.resolve()), **_artifact(manifest_path)}
        or source.get("source_seal_root_sha256")
        != verify_seal(source_freeze_dir)["root_sha256"]
        or source.get("source_rankings")
        != _artifact(source_freeze_dir / "rankings.jsonl")
        or source.get("gate_summary") != _artifact(gate_dir / "summary.json")
        or source.get("facet_gates") != gates_artifact
        or source.get("accepted_union") != accepted_artifact
    ):
        raise ValueError("Mixedbread preflight frozen source binding differs")
    _verified_bound_score_cache(preflight)
    accepted: dict[tuple[str, str], dict[str, object]] = {}
    for row in _read_jsonl(gate_dir / "u_accepted.jsonl", "accepted union"):
        topic_id = str(row.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id not in TOPIC_IDS:
            raise ValueError("accepted union contains an unexpected topic")
        document_id = str(row.get("document_id"))
        text = row.get("text")
        if not isinstance(text, str) or row.get("text_sha256") != _sha256_text(text):
            raise ValueError("accepted union text identity differs")
        identity = (topic_id, document_id)
        if identity in accepted:
            raise ValueError("accepted union contains duplicate topic/document")
        accepted[identity] = row
    candidates = _read_jsonl(preflight_dir / "candidates.jsonl", "Mixedbread candidates")
    windows = _read_jsonl(preflight_dir / "windows.jsonl", "Mixedbread windows")
    cache = _score_cache()
    chunker = SemanticTextChunker(
        ChunkingConfig(
            max_characters=CHUNK_MAX_CHARACTERS,
            overlap_characters=CHUNK_OVERLAP_CHARACTERS,
        )
    )
    candidate_index = 0
    window_index = 0
    topic_summary: dict[str, object] = {}
    for topic_id in TOPIC_IDS:
        population = set(source_orders[topic_id]["RRF"])
        accepted_population = {
            document_id for row_topic, document_id in accepted if row_topic == topic_id
        }
        if population != accepted_population:
            raise ValueError("accepted union differs from the complete ranking population")
        residual = build_residual_pool(
            source_orders[topic_id]["RRF"],
            source_orders[topic_id]["GLOBAL"],
            source_orders[topic_id]["DUAL"],
        )
        dual_rank = {
            document_id: rank
            for rank, document_id in enumerate(source_orders[topic_id]["DUAL"], start=1)
        }
        topic_windows = 0
        for document_id in residual:
            source_row = accepted[(topic_id, document_id)]
            text = str(source_row["text"])
            chunks = chunker.split_text(text, document_id=document_id)
            expected_candidate = {
                "schema_version": SCHEMA_VERSION,
                "topic_id": topic_id,
                "document_id": document_id,
                "dual_rank": dual_rank[document_id],
                "query_sha256": _sha256_text(queries[topic_id]),
                "text_sha256": _sha256_text(text),
                "chunk_count": len(chunks),
            }
            if candidate_index >= len(candidates) or candidates[candidate_index] != expected_candidate:
                raise ValueError("Mixedbread preflight candidate semantic recomputation differs")
            candidate_index += 1
            for chunk_index, chunk in enumerate(chunks):
                cache_key = cache.cache_key(
                    query_text=queries[topic_id], text=chunk.text
                )
                expected_window = {
                    "schema_version": SCHEMA_VERSION,
                    "topic_id": topic_id,
                    "document_id": document_id,
                    "dual_rank": dual_rank[document_id],
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "query": queries[topic_id],
                    "query_sha256": _sha256_text(queries[topic_id]),
                    "window_text": chunk.text,
                    "window_text_sha256": _sha256_text(chunk.text),
                    "score_cache_key": cache_key,
                    "cache_hit": cache_key in cache.scores,
                }
                if window_index >= len(windows) or windows[window_index] != expected_window:
                    raise ValueError("Mixedbread preflight window semantic recomputation differs")
                window_index += 1
            topic_windows += len(chunks)
        topic_summary[topic_id] = {
            "accepted_document_count": len(population),
            "residual_document_count": len(residual),
            "window_count": topic_windows,
            "obligation_count": len(obligations[topic_id]),
            "accepted_facet_ids": obligation_ids[topic_id],
            "accepted_facet_ids_sha256": _sha256_bytes(
                _canonical_bytes(obligation_ids[topic_id])
            ),
            "query_characters": len(queries[topic_id]),
            "query_sha256": _sha256_text(queries[topic_id]),
        }
    if candidate_index != len(candidates) or window_index != len(windows):
        raise ValueError("Mixedbread preflight contains extra candidate or window rows")
    summary = preflight.get("summary")
    expected_summary = {
        "accepted_document_count": len(accepted),
        "residual_document_count": len(candidates),
        "window_count": len(windows),
        "cache_hit_window_count": sum(bool(row["cache_hit"]) for row in windows),
        "uncached_window_count": sum(not bool(row["cache_hit"]) for row in windows),
        "topic": topic_summary,
    }
    if summary != expected_summary:
        raise ValueError("Mixedbread preflight summary semantic recomputation differs")


def _load_cross_encoder_local():
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(str(MODEL_SNAPSHOT), max_length=MAX_LENGTH, device="cuda")
    _validate_model_dtype(model, MODEL_DTYPE)
    return model


def score_preflight(*, preflight_dir: Path, scoring_dir: Path) -> dict[str, object]:
    """Score the exact frozen windows locally; resume only a verified prefix."""

    preflight_dir, scoring_dir = Path(preflight_dir), Path(scoring_dir)
    preflight = _verify_preflight(preflight_dir)
    scoring_dir.mkdir(parents=True, exist_ok=True)
    score_path = scoring_dir / "scores.jsonl"
    receipt_path = scoring_dir / "receipt.json"
    if receipt_path.exists():
        raise FileExistsError("scoring receipt already exists; run is complete")
    windows = _read_jsonl(preflight_dir / "windows.jsonl", "Mixedbread windows")
    score_cache = _verified_bound_score_cache(preflight)
    existing = _read_jsonl(score_path, "Mixedbread scores") if score_path.exists() else []
    if len(existing) > len(windows):
        raise ValueError("score prefix is longer than the frozen window plan")
    for index, row in enumerate(existing):
        window = windows[index]
        _validate_score_row(row, window)
    import torch

    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise RuntimeError("Mixedbread scoring requires local ROCm")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = None
    offset = len(existing)
    inferred_count = 0
    cached_count = 0
    mode = "ab" if score_path.exists() else "xb"
    with score_path.open(mode) as sink:
        while offset < len(windows):
            batch = windows[offset : offset + BATCH_SIZE]
            batch_scores: list[float | None] = []
            missing: list[dict[str, object]] = []
            for row in batch:
                cached = score_cache.get(
                    query_text=str(row["query"]), text=str(row["window_text"])
                )
                if (cached is not None) != (row.get("cache_hit") is True):
                    raise ValueError("score-cache hit status differs from the frozen audit")
                batch_scores.append(cached)
                if cached is None:
                    missing.append(row)
                else:
                    cached_count += 1
            if missing:
                if model is None:
                    model = _load_cross_encoder_local()
                inferred = iter(
                    _predict(
                        model,
                        [(str(row["query"]), str(row["window_text"])) for row in missing],
                        batch_size=BATCH_SIZE,
                        score_representation=SCORE_REPRESENTATION,
                    )
                )
                for index, value in enumerate(batch_scores):
                    if value is None:
                        batch_scores[index] = float(next(inferred))
                        inferred_count += 1
            for row, score in zip(batch, batch_scores, strict=True):
                assert score is not None
                output_row = {
                    **_score_policy_identity(),
                    "topic_id": row["topic_id"],
                    "document_id": row["document_id"],
                    "chunk_index": row["chunk_index"],
                    "query_sha256": row["query_sha256"],
                    "window_text_sha256": row["window_text_sha256"],
                    "score": float(score),
                    "batch_size": BATCH_SIZE,
                    "score_source": (
                        "global_cache" if row.get("cache_hit") is True else "inference"
                    ),
                }
                sink.write(_canonical_bytes(output_row) + b"\n")
            sink.flush()
            os.fsync(sink.fileno())
            offset += len(batch)
            if offset % (BATCH_SIZE * 10) == 0 or offset == len(windows):
                elapsed = time.monotonic() - started
                print(
                    f"SCORE windows={offset}/{len(windows)} "
                    f"elapsed={elapsed:.1f}s inferred_rate={inferred_count/elapsed:.2f}/s",
                    flush=True,
                )
    if model is not None:
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    scores_artifact = _artifact(score_path)
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "post_qrels_diagnostic": True,
        "qrels_read": False,
        "network_access": False,
        "retrieval_calls": 0,
        "paid_cost_usd": 0,
        **_score_policy_identity(),
        "device": "cuda",
        "execution_backend": "rocm",
        "inference_dtype": MODEL_DTYPE,
        "batch_size": BATCH_SIZE,
        "window_count": len(windows),
        "resumed_score_count": len(existing),
        "new_score_row_count": len(windows) - len(existing),
        "cache_reused_window_count": int(preflight["summary"]["cache_hit_window_count"]),  # type: ignore[index]
        "inferred_window_count": int(preflight["summary"]["uncached_window_count"]),  # type: ignore[index]
        "new_cache_reused_window_count": cached_count,
        "new_inferred_window_count": inferred_count,
        "elapsed_seconds": elapsed,
        "observed_new_pairs_per_second": (
            inferred_count / elapsed if inferred_count else None
        ),
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_host_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "preflight_sha256": _artifact(preflight_dir / "preflight.json")["sha256"],
        "windows_sha256": preflight["artifacts"]["windows.jsonl"]["sha256"],  # type: ignore[index]
        "scores": scores_artifact,
    }
    _exclusive_bytes(receipt_path, _pretty_bytes(receipt))
    return receipt


def _validate_scoring_receipt(
    receipt: Mapping[str, object],
    preflight: Mapping[str, object],
    preflight_dir: Path,
    score_path: Path,
) -> None:
    summary = preflight.get("summary")
    artifacts = preflight.get("artifacts")
    if not isinstance(summary, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("Mixedbread preflight lacks scoring bindings")
    expected = {
        **_score_policy_identity(),
        "status": "complete",
        "post_qrels_diagnostic": True,
        "qrels_read": False,
        "network_access": False,
        "retrieval_calls": 0,
        "paid_cost_usd": 0,
        "window_count": summary.get("window_count"),
        "cache_reused_window_count": summary.get("cache_hit_window_count"),
        "inferred_window_count": summary.get("uncached_window_count"),
        "preflight_sha256": _artifact(preflight_dir / "preflight.json")["sha256"],
        "windows_sha256": None,
        "scores": _artifact(score_path),
    }
    expected["windows_sha256"] = (
        artifacts.get("windows.jsonl", {}).get("sha256")
        if isinstance(artifacts.get("windows.jsonl"), Mapping)
        else None
    )
    if any(receipt.get(field) != value for field, value in expected.items()):
        raise ValueError("Mixedbread scoring receipt model/policy binding differs")


def freeze_rankings(
    *,
    preflight_dir: Path,
    scoring_dir: Path,
    source_freeze_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Freeze complete protected-head Mixedbread rankings without reading qrels."""

    preflight_dir, scoring_dir = Path(preflight_dir), Path(scoring_dir)
    source_freeze_dir, output_dir = Path(source_freeze_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"create-only ranking freeze already exists: {output_dir}")
    preflight = _verify_preflight(preflight_dir)
    receipt = _read_object(scoring_dir / "receipt.json", "Mixedbread scoring receipt")
    score_path = scoring_dir / "scores.jsonl"
    _validate_scoring_receipt(receipt, preflight, preflight_dir, score_path)
    windows = _read_jsonl(preflight_dir / "windows.jsonl", "Mixedbread windows")
    scores = _read_jsonl(score_path, "Mixedbread scores")
    if len(scores) != len(windows):
        raise ValueError("Mixedbread score count differs from the frozen window plan")
    by_document: dict[tuple[str, str], list[float]] = defaultdict(list)
    for window, score in zip(windows, scores, strict=True):
        value = _validate_score_row(score, window)
        by_document[(str(window["topic_id"]), str(window["document_id"]))].append(
            value
        )
    source_orders = _load_source_orders(source_freeze_dir)
    rows: list[dict[str, object]] = []
    topic_summary: dict[str, object] = {}
    for topic_id in TOPIC_IDS:
        residual = build_residual_pool(
            source_orders[topic_id]["RRF"],
            source_orders[topic_id]["GLOBAL"],
            source_orders[topic_id]["DUAL"],
        )
        topic_scores = {
            document_id: aggregate_top4(by_document[(topic_id, document_id)])
            for document_id in residual
        }
        if {document_id for row_topic, document_id in by_document if row_topic == topic_id} != set(
            residual
        ):
            raise ValueError("scored documents differ from the frozen residual pool")
        ranking = assemble_complete_ranking(
            source_orders[topic_id]["RRF"], source_orders[topic_id]["DUAL"], topic_scores
        )
        rrf_rank = {
            document_id: rank
            for rank, document_id in enumerate(source_orders[topic_id]["RRF"], start=1)
        }
        dual_rank = {
            document_id: rank
            for rank, document_id in enumerate(source_orders[topic_id]["DUAL"], start=1)
        }
        for rank, document_id in enumerate(ranking, start=1):
            component = "RRF_HEAD" if rank <= RRF_HEAD_DEPTH else (
                "MIXEDBREAD" if rank <= MIXEDBREAD_END_DEPTH else "DUAL_TAIL"
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "topic_id": topic_id,
                    "arm": ARM,
                    "rank": rank,
                    "document_id": document_id,
                    "component": component,
                    "rrf_rank": rrf_rank[document_id],
                    "dual_rank": dual_rank[document_id],
                    "mixedbread_score": topic_scores.get(document_id),
                }
            )
        topic_summary[topic_id] = {
            "candidate_count": len(ranking),
            "protected_head_count": RRF_HEAD_DEPTH,
            "mixedbread_middle_count": MIXEDBREAD_END_DEPTH - RRF_HEAD_DEPTH,
            "complete_tail_count": len(ranking) - MIXEDBREAD_END_DEPTH,
            "scored_residual_document_count": len(residual),
        }
    output_dir.mkdir(parents=True)
    ranking_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    parameters = {
        "schema_version": SCHEMA_VERSION,
        "post_qrels_diagnostic": True,
        "qrels_read": False,
        "topic_ids": list(TOPIC_IDS),
        "arm": ARM,
        "protected_rrf_ranks": [1, RRF_HEAD_DEPTH],
        "mixedbread_ranks": [RRF_HEAD_DEPTH + 1, MIXEDBREAD_END_DEPTH],
        "dual_tail_start": MIXEDBREAD_END_DEPTH + 1,
        "complete_permutation_required": True,
        "aggregation": "chunk_top4_weighted",
        "top4_weights": list(TOP4_WEIGHTS),
        "retrieval_calls": 0,
        "paid_cost_usd": 0,
    }
    binding = {
        "schema_version": SCHEMA_VERSION,
        "preflight_dir_path": str(preflight_dir.resolve()),
        "scoring_dir_path": str(scoring_dir.resolve()),
        "source_freeze_dir_path": str(source_freeze_dir.resolve()),
        "preflight": _artifact(preflight_dir / "preflight.json"),
        "windows": _artifact(preflight_dir / "windows.jsonl"),
        "scoring_receipt": _artifact(scoring_dir / "receipt.json"),
        "scores": _artifact(score_path),
        "source_seal_root_sha256": verify_seal(source_freeze_dir)["root_sha256"],
        "source_rankings": _artifact(source_freeze_dir / "rankings.jsonl"),
    }
    _exclusive_bytes(output_dir / "parameters.json", _pretty_bytes(parameters))
    _exclusive_bytes(output_dir / "input_binding.json", _pretty_bytes(binding))
    _exclusive_bytes(output_dir / "rankings.jsonl", ranking_bytes)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "post_qrels_diagnostic": True,
        "qrels_read": False,
        "topic_ids": list(TOPIC_IDS),
        "arm": ARM,
        "ranking_row_count": len(rows),
        "topic": topic_summary,
        "artifacts": {
            name: _artifact(output_dir / name)
            for name in ("parameters.json", "input_binding.json", "rankings.jsonl")
        },
    }
    _exclusive_bytes(output_dir / "summary.json", _pretty_bytes(summary))
    artifact_names = ("parameters.json", "input_binding.json", "rankings.jsonl", "summary.json")
    seal_material = {
        "topic_ids": list(TOPIC_IDS),
        "source_seal_root_sha256": binding["source_seal_root_sha256"],
        "artifacts": {name: _artifact(output_dir / name) for name in artifact_names},
    }
    seal = {
        "schema_version": SCHEMA_VERSION,
        "status": "rankings_frozen_before_diagnostic_evaluation",
        "post_qrels_diagnostic": True,
        "qrels_read": False,
        **seal_material,
        "root_sha256": _sha256_bytes(_canonical_bytes(seal_material)),
    }
    _exclusive_bytes(output_dir / "SEALED.json", _pretty_bytes(seal))
    verify_ranking_freeze(output_dir)
    verify_ranking_semantics(output_dir)
    return summary


def verify_ranking_freeze(output_dir: Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    seal = _read_object(output_dir / "SEALED.json", "Mixedbread ranking seal")
    if (
        seal.get("schema_version") != SCHEMA_VERSION
        or seal.get("status") != "rankings_frozen_before_diagnostic_evaluation"
        or seal.get("qrels_read") is not False
        or seal.get("topic_ids") != list(TOPIC_IDS)
    ):
        raise ValueError("Mixedbread ranking seal contract differs")
    expected = {"SEALED.json", "parameters.json", "input_binding.json", "rankings.jsonl", "summary.json"}
    if {path.name for path in output_dir.iterdir() if path.is_file()} != expected:
        raise ValueError("Mixedbread ranking freeze has missing or extra files")
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Mixedbread ranking seal lacks artifacts")
    actual = {
        name: _artifact(output_dir / name)
        for name in ("parameters.json", "input_binding.json", "rankings.jsonl", "summary.json")
    }
    if actual != artifacts:
        raise ValueError("Mixedbread ranking artifacts were mutated")
    material = {
        "topic_ids": seal["topic_ids"],
        "source_seal_root_sha256": seal["source_seal_root_sha256"],
        "artifacts": artifacts,
    }
    if seal.get("root_sha256") != _sha256_bytes(_canonical_bytes(material)):
        raise ValueError("Mixedbread ranking seal root differs")
    rows = _read_jsonl(output_dir / "rankings.jsonl", "Mixedbread frozen rankings")
    by_topic: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        topic_id = str(row.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id not in TOPIC_IDS or row.get("arm") != ARM:
            raise ValueError("Mixedbread ranking has an unexpected topic or arm")
        by_topic[topic_id].append(row)
    for topic_id in TOPIC_IDS:
        ordered = sorted(by_topic[topic_id], key=lambda row: int(row["rank"]))
        if [int(row["rank"]) for row in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError("Mixedbread ranking is not contiguous")
        ids = [str(row["document_id"]) for row in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("Mixedbread ranking contains duplicate documents")
        if any(row.get("component") != "RRF_HEAD" for row in ordered[:RRF_HEAD_DEPTH]):
            raise ValueError("Mixedbread ranking did not preserve its protected head")
    return seal


def verify_ranking_semantics(output_dir: Path) -> dict[str, object]:
    """Recompute every ranking row from its sealed sources and compare exactly."""

    output_dir = Path(output_dir)
    seal = verify_ranking_freeze(output_dir)
    binding = _read_object(output_dir / "input_binding.json", "Mixedbread input binding")
    try:
        preflight_dir = Path(str(binding["preflight_dir_path"]))
        scoring_dir = Path(str(binding["scoring_dir_path"]))
        source_freeze_dir = Path(str(binding["source_freeze_dir_path"]))
    except KeyError as exc:
        raise ValueError("Mixedbread input binding lacks a source path") from exc
    bound_artifacts = {
        "preflight": _artifact(preflight_dir / "preflight.json"),
        "windows": _artifact(preflight_dir / "windows.jsonl"),
        "scoring_receipt": _artifact(scoring_dir / "receipt.json"),
        "scores": _artifact(scoring_dir / "scores.jsonl"),
        "source_seal_root_sha256": verify_seal(source_freeze_dir)["root_sha256"],
        "source_rankings": _artifact(source_freeze_dir / "rankings.jsonl"),
    }
    for field, actual in bound_artifacts.items():
        if binding.get(field) != actual:
            raise ValueError(f"Mixedbread bound source {field} was mutated")
    windows = _read_jsonl(preflight_dir / "windows.jsonl", "Mixedbread windows")
    scores = _read_jsonl(scoring_dir / "scores.jsonl", "Mixedbread scores")
    if len(windows) != len(scores):
        raise ValueError("Mixedbread semantic verification score count differs")
    preflight = _verify_preflight(preflight_dir)
    receipt = _read_object(scoring_dir / "receipt.json", "Mixedbread scoring receipt")
    _validate_scoring_receipt(
        receipt, preflight, preflight_dir, scoring_dir / "scores.jsonl"
    )
    by_document: dict[tuple[str, str], list[float]] = defaultdict(list)
    for window, score in zip(windows, scores, strict=True):
        value = _validate_score_row(score, window)
        by_document[(str(window["topic_id"]), str(window["document_id"]))].append(
            value
        )
    source_orders = _load_source_orders(source_freeze_dir)
    expected_rows: list[dict[str, object]] = []
    for topic_id in TOPIC_IDS:
        residual = build_residual_pool(
            source_orders[topic_id]["RRF"],
            source_orders[topic_id]["GLOBAL"],
            source_orders[topic_id]["DUAL"],
        )
        topic_scores = {
            document_id: aggregate_top4(by_document[(topic_id, document_id)])
            for document_id in residual
        }
        if {document_id for row_topic, document_id in by_document if row_topic == topic_id} != set(
            residual
        ):
            raise ValueError("Mixedbread semantic verification population differs")
        ranking = assemble_complete_ranking(
            source_orders[topic_id]["RRF"], source_orders[topic_id]["DUAL"], topic_scores
        )
        rrf_rank = {
            document_id: rank
            for rank, document_id in enumerate(source_orders[topic_id]["RRF"], start=1)
        }
        dual_rank = {
            document_id: rank
            for rank, document_id in enumerate(source_orders[topic_id]["DUAL"], start=1)
        }
        for rank, document_id in enumerate(ranking, start=1):
            expected_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "topic_id": topic_id,
                    "arm": ARM,
                    "rank": rank,
                    "document_id": document_id,
                    "component": "RRF_HEAD" if rank <= RRF_HEAD_DEPTH else (
                        "MIXEDBREAD" if rank <= MIXEDBREAD_END_DEPTH else "DUAL_TAIL"
                    ),
                    "rrf_rank": rrf_rank[document_id],
                    "dual_rank": dual_rank[document_id],
                    "mixedbread_score": topic_scores.get(document_id),
                }
            )
    actual_rows = _read_jsonl(output_dir / "rankings.jsonl", "Mixedbread frozen rankings")
    if actual_rows != expected_rows:
        raise ValueError("Mixedbread ranking semantic recomputation differs")
    return seal


def _load_frozen_rankings(output_dir: Path) -> dict[str, list[str]]:
    verify_ranking_semantics(output_dir)
    rows = _read_jsonl(Path(output_dir) / "rankings.jsonl", "Mixedbread frozen rankings")
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["topic_id"])].append((int(row["rank"]), str(row["document_id"])))
    return {
        topic_id: [document_id for _, document_id in sorted(grouped[topic_id])]
        for topic_id in TOPIC_IDS
    }


def _authenticate_evaluation_inputs(
    qrels_projection_path: Path, source_metrics_path: Path
) -> dict[str, object]:
    """Verify the existing one-way qrels evaluation chain before any join."""

    qrels_projection_path = Path(qrels_projection_path)
    source_metrics_path = Path(source_metrics_path)
    if qrels_projection_path.parent.resolve() != source_metrics_path.parent.resolve():
        raise ValueError("qrels projection and source metrics must share one evaluation root")
    root = source_metrics_path.parent
    receipt = _read_object(root / "qrels_access_receipt.json", "qrels access receipt")
    summary = _read_object(root / "summary.json", "source evaluation summary")
    decision_path = root / "decision.json"
    projection_rows = _read_jsonl(qrels_projection_path, "projected qrels")
    if (
        receipt.get("schema_version") != "deep-facet-candidate-evaluation-v1"
        or receipt.get("status") != "qrels_access_boundary_crossed"
        or receipt.get("qrels_opened") is not True
        or receipt.get("topic_ids") != list(TOPIC_IDS)
        or receipt.get("qrels_projection_rows") != len(projection_rows)
        or receipt.get("qrels_projection_sha256")
        != _artifact(qrels_projection_path)["sha256"]
    ):
        raise ValueError("qrels access receipt does not authenticate the projection")
    if (
        summary.get("schema_version") != "deep-facet-candidate-evaluation-v1"
        or summary.get("status") != "complete"
        or summary.get("qrels_opened") is not True
        or summary.get("topic_ids") != list(TOPIC_IDS)
        or summary.get("metrics_sha256") != _artifact(source_metrics_path)["sha256"]
        or summary.get("decision_sha256") != _artifact(decision_path)["sha256"]
    ):
        raise ValueError("source evaluation summary does not authenticate its artifacts")
    metrics = _read_object(source_metrics_path, "source metrics")
    if (
        metrics.get("schema_version") != "deep-facet-candidate-evaluation-v1"
        or metrics.get("topic_ids") != list(TOPIC_IDS)
    ):
        raise ValueError("source metrics contract differs")
    return metrics


def evaluate_frozen_rankings(
    *,
    freeze_dir: Path,
    qrels_projection_path: Path,
    source_metrics_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Evaluate the sealed post-qrels diagnostic against the existing projection."""

    freeze_dir, output_dir = Path(freeze_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"create-only evaluation already exists: {output_dir}")
    source_metrics = _authenticate_evaluation_inputs(
        Path(qrels_projection_path), Path(source_metrics_path)
    )
    rankings = _load_frozen_rankings(freeze_dir)
    qrels: dict[str, dict[str, int]] = {topic: {} for topic in TOPIC_IDS}
    for row in _read_jsonl(Path(qrels_projection_path), "projected qrels"):
        topic_id = str(row.get("topic_id"))
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id not in qrels:
            raise ValueError("projected qrels contain an unexpected topic")
        document_id = str(row["document_id"])
        if document_id in qrels[topic_id]:
            raise ValueError("projected qrels contain duplicate topic-document identities")
        qrels[topic_id][document_id] = int(row["grade"])
    if any(not qrels[topic] for topic in TOPIC_IDS):
        raise ValueError("projected qrels lack one or more frozen topics")
    discovery = source_metrics.get("discovery")
    source_per_topic = source_metrics.get("per_topic")
    if not isinstance(discovery, Mapping) or not isinstance(source_per_topic, Mapping):
        raise ValueError("source metrics lack discovery or per-topic evidence")
    per_topic: dict[str, object] = {}
    aggregate_values: dict[str, list[float]] = defaultdict(list)
    aggregate_novel: dict[str, int] = defaultdict(int)
    novel_total = 0
    for topic_id in TOPIC_IDS:
        evaluated = evaluate_ranking(
            rankings[topic_id],
            qrels[topic_id],
            depths=(100, 500, 1000, len(rankings[topic_id])),
        )
        discovery_row = discovery.get(topic_id)
        if not isinstance(discovery_row, Mapping) or not isinstance(
            discovery_row.get("novel_relevant_ids"), list
        ):
            raise ValueError("source metrics lack topic novel-relevance evidence")
        novel = set(map(str, discovery_row["novel_relevant_ids"]))
        novel_total += len(novel)
        row: dict[str, object] = {
            "candidate_count": len(rankings[topic_id]),
            "ndcg@10": float(evaluated["ndcg@10"]),
            "ndcg@100": float(evaluated["ndcg@100"]),
        }
        for metric in ("ndcg@10", "ndcg@100"):
            aggregate_values[metric].append(float(row[metric]))
        for depth in (100, 500, 1000, len(rankings[topic_id])):
            metrics_at_depth = evaluated[str(depth)]
            assert isinstance(metrics_at_depth, Mapping)
            label = "full" if depth == len(rankings[topic_id]) else str(depth)
            for field in ("recall", "graded_recall", "judged_rate", "relevant_count"):
                key = f"{field}@{label}"
                row[key] = metrics_at_depth[field]
                if field != "relevant_count":
                    aggregate_values[key].append(float(metrics_at_depth[field]))
            retained = len(novel.intersection(rankings[topic_id][:depth]))
            row[f"novel_retained@{label}"] = retained
            row[f"novel_retention@{label}"] = retained / len(novel) if novel else None
            aggregate_novel[f"novel_retained@{label}"] += retained
        source_topic = source_per_topic.get(topic_id)
        if not isinstance(source_topic, Mapping) or not isinstance(source_topic.get("RRF"), Mapping):
            raise ValueError("source metrics lack topic RRF evidence")
        source_rrf = source_topic["RRF"]
        assert isinstance(source_rrf, Mapping)
        row["graded_recall@500_delta_vs_RRF"] = float(row["graded_recall@500"]) - float(
            source_rrf["graded_recall@500"]
        )
        per_topic[topic_id] = row
    aggregate: dict[str, object] = {
        key: sum(values) / len(values) for key, values in aggregate_values.items()
    }
    for key, count in aggregate_novel.items():
        label = key.split("@", 1)[1]
        aggregate[key] = count
        aggregate[f"novel_retention@{label}"] = count / novel_total if novel_total else None
    source_aggregate = source_metrics.get("aggregate")
    baseline = {}
    if isinstance(source_aggregate, Mapping):
        for arm in ("RRF", "GLOBAL", "DUAL"):
            if isinstance(source_aggregate.get(arm), Mapping):
                baseline[arm] = source_aggregate[arm]
    freeze_binding = _read_object(
        freeze_dir / "input_binding.json", "Mixedbread freeze binding"
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "post_qrels_diagnostic": True,
        "confirmatory_evidence": False,
        "topic_ids": list(TOPIC_IDS),
        "relevance_threshold": 2,
        "arm": ARM,
        "aggregate": {ARM: aggregate},
        "per_topic": per_topic,
        "baseline": baseline,
        "interpretation": {
            "complete_ranking": True,
            "full_depth_is_population_property": True,
            "ranking_value_is_speed_of_relevant_and_novel_evidence_into_practical_depths": True,
        },
        "source": {
            "ranking_seal_root_sha256": verify_ranking_freeze(freeze_dir)["root_sha256"],
            "scoring_receipt": freeze_binding.get("scoring_receipt"),
            "qrels_projection": _artifact(Path(qrels_projection_path)),
            "source_metrics": _artifact(Path(source_metrics_path)),
        },
    }
    output_dir.mkdir(parents=True)
    _exclusive_bytes(output_dir / "metrics.json", _pretty_bytes(payload))
    rrf_baseline = baseline.get("RRF")
    global_baseline = baseline.get("GLOBAL")
    if not isinstance(rrf_baseline, Mapping) or not isinstance(global_baseline, Mapping):
        raise ValueError("source metrics lack aggregate RRF or GLOBAL evidence")
    guards = {
        "ndcg_10_exactly_preserves_rrf": math.isclose(
            float(aggregate["ndcg@10"]), float(rrf_baseline["ndcg@10"]), abs_tol=1e-15
        ),
        "graded_recall_500_beats_rrf": float(aggregate["graded_recall@500"])
        > float(rrf_baseline["graded_recall@500"]),
        "graded_recall_500_beats_global": float(aggregate["graded_recall@500"])
        > float(global_baseline["graded_recall@500"]),
        "graded_recall_1000_at_least_rrf": float(aggregate["graded_recall@1000"])
        >= float(rrf_baseline["graded_recall@1000"]),
        "novel_retention_1000_at_least_80_percent": float(
            aggregate["novel_retention@1000"]
        )
        >= 0.80,
        "per_topic_graded_recall_500_floor": all(
            float(per_topic[topic]["graded_recall@500_delta_vs_RRF"]) >= -0.02  # type: ignore[index]
            for topic in TOPIC_IDS
        ),
    }
    decision = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "post_qrels_diagnostic": True,
        "confirmatory_evidence": False,
        "guards": guards,
        "mechanical_guards_pass": all(guards.values()),
        "failed_guards": [name for name, passed in guards.items() if not passed],
        "fresh_validation_eligible_on_mechanical_evidence": all(guards.values()),
        "advance_to_fresh_validation": False,
        "judged_coverage_review_required": True,
        "coverage_assessment": "inconclusive_on_exposed_projected_qrels",
        "production_promotion_authorized": False,
        "no_retuning_on_exposed_topics": True,
    }
    _exclusive_bytes(output_dir / "decision.json", _pretty_bytes(decision))
    _exclusive_bytes(
        output_dir / "summary.json",
        _pretty_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "post_qrels_diagnostic": True,
                "qrels_opened": True,
                "topic_ids": list(TOPIC_IDS),
                "metrics_sha256": _artifact(output_dir / "metrics.json")["sha256"],
                "decision_sha256": _artifact(output_dir / "decision.json")["sha256"],
                "ranking_seal_root_sha256": payload["source"]["ranking_seal_root_sha256"],  # type: ignore[index]
            }
        ),
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--manifest", required=True, type=Path)
    preflight.add_argument("--source-freeze", required=True, type=Path)
    preflight.add_argument("--gate-dir", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)
    preflight.add_argument("--measured-pairs-per-second", required=True, type=float)
    preflight.add_argument("--benchmark-batch-size", required=True, type=int)
    score = subparsers.add_parser("score")
    score.add_argument("--preflight", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--preflight", required=True, type=Path)
    freeze.add_argument("--scoring", required=True, type=Path)
    freeze.add_argument("--source-freeze", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--freeze", required=True, type=Path)
    evaluate.add_argument("--qrels-projection", required=True, type=Path)
    evaluate.add_argument("--source-metrics", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "preflight":
        result = create_preflight(
            manifest_path=args.manifest,
            freeze_dir=args.source_freeze,
            gate_dir=args.gate_dir,
            output_dir=args.output,
            measured_pairs_per_second=args.measured_pairs_per_second,
            benchmark_batch_size=args.benchmark_batch_size,
        )
    elif args.command == "score":
        result = score_preflight(preflight_dir=args.preflight, scoring_dir=args.output)
    elif args.command == "freeze":
        result = freeze_rankings(
            preflight_dir=args.preflight,
            scoring_dir=args.scoring,
            source_freeze_dir=args.source_freeze,
            output_dir=args.output,
        )
    else:
        result = evaluate_frozen_rankings(
            freeze_dir=args.freeze,
            qrels_projection_path=args.qrels_projection,
            source_metrics_path=args.source_metrics,
            output_dir=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
