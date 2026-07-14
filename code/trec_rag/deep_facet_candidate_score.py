"""Local-only MiniLM window planning and scoring for deep facet candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .deep_facet_candidate_manifest import (
    EXCLUDED_TOPIC_IDS,
    EXPERIMENT_ID,
    assert_mutation_allowed,
    load_manifest,
)
from .facet_local_minilm_preflight import (
    MODEL_ID,
    MODEL_REVISION,
    PAIR_MAX_TOKENS,
    WindowPlanRow,
    build_window_plan,
    load_verified_materialization,
    load_verified_tokenizer,
    score_cache_context,
)
from .facet_local_minilm_rank import aggregate_top4
from .rerank_score_cache import GlobalScoreCache


SOURCE_RECEIPT_PATH = Path(
    "outputs/rag25_facet_aware_fusion_v1/preflight_v1/preflight.json"
)
SOURCE_RECEIPT_SHA256 = (
    "346f3239373924d3ed8dc76076dfb3221c0a29ab8ab24ce71a1e5f98ce180122"
)
SCORE_CACHE_ROOT = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/reranker"
)
MAX_PHASE_SECONDS = 600.0
REFERENCE_PAIRS_PER_SECOND = 300.0
REFERENCE_FIXED_SECONDS = 30.0
BATCH_SIZE = 32
PREFLIGHT_SCHEMA_VERSION = "deep-facet-candidate-minilm-preflight-v1"
SCORE_SCHEMA_VERSION = "deep-facet-candidate-minilm-score-v1"
RECEIPT_SCHEMA_VERSION = "deep-facet-candidate-minilm-receipt-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _exclusive_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as sink:
        sink.write(value)
        sink.flush()
        os.fsync(sink.fileno())


def _exclusive_json(path: Path, value: Mapping[str, object]) -> None:
    _exclusive_bytes(path, _pretty_bytes(value))


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_number} must be an object")
        rows.append(row)
    return rows


def verify_source_receipt(path: Path, expected_sha256: str) -> str:
    try:
        source = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError("authenticated source receipt is required") from exc
    actual = _sha256_bytes(source)
    if actual != expected_sha256:
        raise ValueError("authenticated source receipt SHA-256 differs")
    return actual


def _reject_excluded(topic_id: object) -> str:
    value = str(topic_id)
    if value in EXCLUDED_TOPIC_IDS:
        raise ValueError(f"excluded topic {value} is forbidden")
    return value


def phase1_candidates(
    manifest: Mapping[str, object],
    retrieval_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Bind every retrieved row to its exact facet-local MiniLM query."""

    if manifest.get("qrels_opened") is not False:
        raise ValueError("phase 1 requires qrels_opened=false")
    raw_facets = manifest.get("facets")
    if not isinstance(raw_facets, list):
        raise ValueError("manifest facets must be an array")
    facets: dict[str, Mapping[str, object]] = {}
    for facet in raw_facets:
        if not isinstance(facet, Mapping):
            raise ValueError("manifest facet must be an object")
        _reject_excluded(facet.get("topic_id"))
        facets[str(facet.get("facet_id"))] = facet
    candidates: list[dict[str, object]] = []
    observed: dict[str, int] = {facet_id: 0 for facet_id in facets}
    for row in retrieval_rows:
        topic_id = _reject_excluded(row.get("topic_id"))
        facet_id = str(row.get("facet_id"))
        facet = facets.get(facet_id)
        if facet is None or str(facet.get("topic_id")) != topic_id:
            raise ValueError("retrieval row facet identity differs from manifest")
        query = str(facet.get("query"))
        query_sha = _sha256_bytes(query.encode("utf-8"))
        text = row.get("text")
        docid = row.get("docid")
        rank = row.get("rank")
        if (
            row.get("query_sha256") != query_sha
            or not isinstance(text, str)
            or not text
            or not isinstance(docid, str)
            or not docid
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 1
        ):
            raise ValueError("retrieval candidate query, text, ID, or rank is invalid")
        candidates.append(
            {
                "topic_id": topic_id,
                "family": "facet",
                "variant": facet_id,
                "facet_id": facet_id,
                "manifest_order": facet["manifest_order"],
                "rank": rank,
                "document_id": docid,
                "docid": docid,
                "query": query,
                "query_sha256": query_sha,
                "text": text,
                "text_sha256": _sha256_bytes(text.encode("utf-8")),
                "retrieval_score": float(row.get("score", 0.0)),
            }
        )
        observed[facet_id] += 1
    if any(count != 200 for count in observed.values()) and len(facets) > 1:
        raise ValueError("phase 1 requires exactly 200 candidates per facet")
    return candidates


def build_phase_preflight(
    candidates: Sequence[Mapping[str, object]],
    tokenizer: object,
    *,
    cache_lookup: Callable[[str, str], float | None],
    pairs_per_second: float = REFERENCE_PAIRS_PER_SECOND,
    fixed_seconds: float = REFERENCE_FIXED_SECONDS,
) -> dict[str, object]:
    """Build a bounded window plan and estimate one phase without inference."""

    if pairs_per_second <= 0 or fixed_seconds < 0:
        raise ValueError("runtime projection inputs must be positive/non-negative")
    windows: list[WindowPlanRow] = []
    for candidate in candidates:
        query = str(candidate.get("query"))
        for row in build_window_plan(candidate, tokenizer, query=query):
            cached = cache_lookup(row.query, row.window_text) is not None
            windows.append(replace(row, cache_hit=cached))
    unique_pairs = {row.cache_key for row in windows}
    unique_misses = {row.cache_key for row in windows if not row.cache_hit}
    projected = fixed_seconds + len(unique_misses) / pairs_per_second
    if projected > MAX_PHASE_SECONDS:
        raise ValueError(
            f"projected MiniLM phase exceeds 600 seconds: {projected:.3f}"
        )
    return {
        "windows": [row.to_dict() for row in windows],
        "summary": {
            "document_count": len(candidates),
            "window_count": len(windows),
            "unique_pair_count": len(unique_pairs),
            "unique_uncached_pair_count": len(unique_misses),
            "cache_hit_window_count": sum(row.cache_hit for row in windows),
        },
        "pairs_per_second": float(pairs_per_second),
        "fixed_seconds": float(fixed_seconds),
        "projected_runtime_seconds": projected,
    }


def _source_bindings() -> tuple[dict[str, object], Path, str, str]:
    receipt_path = SOURCE_RECEIPT_PATH.resolve()
    receipt_sha = verify_source_receipt(receipt_path, SOURCE_RECEIPT_SHA256)
    receipt = _read_json(receipt_path, "authenticated source preflight")
    model_receipt_value = receipt.get("model_materialization_receipt")
    model_receipt_sha = receipt.get("model_materialization_receipt_sha256")
    if not isinstance(model_receipt_value, str) or not isinstance(model_receipt_sha, str):
        raise ValueError("authenticated source receipt lacks model bindings")
    model_receipt_path = Path(model_receipt_value).resolve()
    verified = load_verified_materialization(model_receipt_path)
    if verified.sha256 != model_receipt_sha:
        raise ValueError("materialization receipt differs from authenticated source")
    if (
        verified.payload.get("model_id") != MODEL_ID
        or verified.payload.get("revision") != MODEL_REVISION
    ):
        raise ValueError("materialized MiniLM model identity differs")
    return receipt, model_receipt_path, model_receipt_sha, receipt_sha


def _load_authenticated_retrieval(
    retrieval_dir: Path, manifest: Mapping[str, object]
) -> list[dict[str, object]]:
    summary_path = Path(retrieval_dir) / "retrieval_summary.json"
    candidates_path = Path(retrieval_dir) / "candidates.jsonl"
    summary = _read_json(summary_path, "retrieval summary")
    source = candidates_path.read_bytes()
    if (
        summary.get("complete") is not True
        or summary.get("qrels_opened") is not False
        or summary.get("candidate_rows") != 5000
        or summary.get("candidates_sha256") != _sha256_bytes(source)
        or summary.get("manifest_sha256") != _sha256_bytes(_canonical_bytes(manifest))
    ):
        raise ValueError("retrieval summary does not authenticate 5,000 candidates")
    rows = _read_jsonl(candidates_path, "retrieval candidates")
    if len(rows) != 5000:
        raise ValueError("retrieval candidates must contain exactly 5,000 rows")
    return rows


def create_phase1_preflight(
    manifest: Mapping[str, object],
    retrieval_dir: Path,
    output: Path,
    *,
    cache_root: Path = SCORE_CACHE_ROOT,
) -> dict[str, object]:
    output = Path(output)
    assert_mutation_allowed(output)
    assert_mutation_allowed(output.parent)
    for name in ("preflight.json", "windows.jsonl", "candidates.jsonl", "scores.jsonl"):
        if (output / name).exists():
            raise FileExistsError(f"create-only phase output already exists: {output / name}")
    _source_receipt, model_receipt, model_receipt_sha, source_receipt_sha = (
        _source_bindings()
    )
    tokenizer = load_verified_tokenizer(model_receipt)
    cache = GlobalScoreCache(Path(cache_root), score_cache_context())
    retrieval_rows = _load_authenticated_retrieval(retrieval_dir, manifest)
    candidates = phase1_candidates(manifest, retrieval_rows)
    plan = build_phase_preflight(
        candidates,
        tokenizer,
        cache_lookup=lambda query, text: cache.get(query_text=query, text=text),
    )
    candidate_bytes = _jsonl_bytes(candidates)
    window_rows = plan.pop("windows")
    window_bytes = _jsonl_bytes(window_rows)  # type: ignore[arg-type]
    _exclusive_bytes(output / "candidates.jsonl", candidate_bytes)
    _exclusive_bytes(output / "windows.jsonl", window_bytes)
    payload: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase": "phase1_facet_local",
        "status": "tokenizer_only_preflight_complete",
        "qrels_opened": False,
        "retrieval_path_supported": False,
        "network_access_supported": False,
        "hosted_inference_supported": False,
        "model_constructed": False,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_receipt_path": str(SOURCE_RECEIPT_PATH),
        "source_receipt_sha256": source_receipt_sha,
        "model_materialization_receipt": str(model_receipt),
        "model_materialization_receipt_sha256": model_receipt_sha,
        "retrieval_summary_sha256": _sha256_bytes(
            (Path(retrieval_dir) / "retrieval_summary.json").read_bytes()
        ),
        "candidates_sha256": _sha256_bytes(candidate_bytes),
        "windows_sha256": _sha256_bytes(window_bytes),
        "score_cache_path": str(cache.path),
        "score_cache_bytes": cache.path.stat().st_size if cache.path.exists() else 0,
        "summary": plan["summary"],
        "pairs_per_second": plan["pairs_per_second"],
        "fixed_seconds": plan["fixed_seconds"],
        "projected_runtime_seconds": plan["projected_runtime_seconds"],
        "runtime_ceiling_seconds": MAX_PHASE_SECONDS,
        "window_policy": {
            "pair_max_tokens": 512,
            "query_max_tokens": 192,
            "minimum_passage_tokens": 256,
            "passage_overlap_tokens": 64,
            "maximum_windows_per_document": 32,
            "span_distinct_minimum_new_tokens": 128,
            "top4_weights": [0.55, 0.25, 0.13, 0.07],
        },
        "scorer_code_sha256": _sha256_bytes(Path(__file__).read_bytes()),
    }
    _exclusive_json(output / "preflight.json", payload)
    return payload


def _float32(value: object) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("MiniLM score must be finite")
    return struct.unpack(">f", struct.pack(">f", converted))[0]


def _host_memory_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _load_model(model_receipt: Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    verified = load_verified_materialization(model_receipt)
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise RuntimeError("ROCm MiniLM scoring requires an available torch cuda device")
    tokenizer = AutoTokenizer.from_pretrained(
        verified.snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        verified.snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
    )
    model = model.float().eval().to("cuda")
    return torch, tokenizer, model


def run_local_scoring(
    preflight_path: Path,
    *,
    cache_root: Path = SCORE_CACHE_ROOT,
) -> dict[str, object]:
    preflight_path = Path(preflight_path)
    output = preflight_path.parent
    assert_mutation_allowed(output)
    assert_mutation_allowed(output.parent)
    if (output / "scores.jsonl").exists() or (output / "scoring_receipt.json").exists():
        raise FileExistsError("create-only scoring output already exists")
    preflight = _read_json(preflight_path, "MiniLM preflight")
    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or preflight.get("status") != "tokenizer_only_preflight_complete"
        or preflight.get("qrels_opened") is not False
        or preflight.get("model") != MODEL_ID
        or preflight.get("model_revision") != MODEL_REVISION
        or preflight.get("scorer_code_sha256")
        != _sha256_bytes(Path(__file__).read_bytes())
        or float(preflight.get("projected_runtime_seconds", MAX_PHASE_SECONDS + 1))
        > MAX_PHASE_SECONDS
    ):
        raise ValueError("MiniLM preflight differs from the frozen scorer contract")
    verify_source_receipt(SOURCE_RECEIPT_PATH, SOURCE_RECEIPT_SHA256)
    model_receipt = Path(str(preflight["model_materialization_receipt"]))
    verified = load_verified_materialization(model_receipt)
    if verified.sha256 != preflight.get("model_materialization_receipt_sha256"):
        raise ValueError("model receipt differs from MiniLM preflight")
    window_source = (output / "windows.jsonl").read_bytes()
    candidate_source = (output / "candidates.jsonl").read_bytes()
    if (
        _sha256_bytes(window_source) != preflight.get("windows_sha256")
        or _sha256_bytes(candidate_source) != preflight.get("candidates_sha256")
    ):
        raise ValueError("window or candidate bytes differ from MiniLM preflight")
    raw_windows = _read_jsonl(output / "windows.jsonl", "MiniLM windows")
    windows = [WindowPlanRow.from_dict(row) for row in raw_windows]
    cache = GlobalScoreCache(Path(cache_root), score_cache_context())
    by_key: dict[str, WindowPlanRow] = {}
    for window in windows:
        by_key.setdefault(window.cache_key, window)
    misses = [
        row
        for key, row in sorted(by_key.items())
        if cache.get(query_text=row.query, text=row.window_text) is None
    ]
    planned_misses = int(preflight["summary"]["unique_uncached_pair_count"])  # type: ignore[index]
    if len(misses) != planned_misses:
        raise ValueError("score-cache misses changed after MiniLM preflight")
    started = time.perf_counter()
    peak_device = 0
    peak_host = _host_memory_bytes()
    if misses:
        torch, tokenizer, model = _load_model(model_receipt)
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(misses), BATCH_SIZE):
            batch = misses[start : start + BATCH_SIZE]
            encoded = tokenizer(
                [row.query for row in batch],
                [row.window_text for row in batch],
                padding=True,
                truncation=False,
                max_length=PAIR_MAX_TOKENS,
                return_tensors="pt",
            )
            inputs = {name: tensor.to("cuda") for name, tensor in encoded.items()}
            with torch.inference_mode():
                output_value = model(**inputs)
            values = output_value.logits.detach().float().cpu().tolist()
            if len(values) != len(batch) or any(
                not isinstance(value, list) or len(value) != 1 for value in values
            ):
                raise ValueError("MiniLM logits must have shape (batch, 1)")
            scores = [_float32(value[0]) for value in values]
            cache.add_many(
                (row.query, row.window_text, score)
                for row, score in zip(batch, scores, strict=True)
            )
            peak_device = max(peak_device, int(torch.cuda.max_memory_allocated()))
            peak_host = max(peak_host, _host_memory_bytes())
    elapsed = time.perf_counter() - started
    score_rows: list[dict[str, object]] = []
    for row in windows:
        score = cache.get(query_text=row.query, text=row.window_text)
        if score is None:
            raise ValueError("scoring cache does not cover a frozen window")
        score_rows.append(
            {
                **row.to_dict(),
                "schema_version": SCORE_SCHEMA_VERSION,
                "score": _float32(score),
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "score_representation": "raw_logits",
                "inference_dtype": "float32",
            }
        )
    score_bytes = _jsonl_bytes(score_rows)
    _exclusive_bytes(output / "scores.jsonl", score_bytes)
    receipt: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "phase": preflight["phase"],
        "qrels_opened": False,
        "network_access_supported": False,
        "hosted_inference_supported": False,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preflight_sha256": _sha256_bytes(preflight_path.read_bytes()),
        "windows_sha256": preflight["windows_sha256"],
        "scores_sha256": _sha256_bytes(score_bytes),
        "planned_window_count": len(windows),
        "completed_window_count": len(score_rows),
        "unique_forward_pair_count": len(misses),
        "cache_reuse_pair_count": len(by_key) - len(misses),
        "elapsed_seconds": elapsed,
        "peak_device_memory_bytes": peak_device,
        "peak_host_memory_bytes": peak_host,
        "device": "cuda",
        "execution_backend": "rocm",
        "score_cache_path": str(cache.path),
    }
    _exclusive_json(output / "scoring_receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight-phase1")
    preflight.add_argument("--manifest", required=True, type=Path)
    preflight.add_argument("--retrieval", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)
    score = subparsers.add_parser("score-phase1")
    score.add_argument("--preflight", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight-phase1":
        manifest = load_manifest(
            args.manifest,
            cache_root=Path(
                "/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote"
            ),
        )
        result = create_phase1_preflight(
            manifest, args.retrieval, args.output
        )
    else:
        result = run_local_scoring(args.preflight)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
