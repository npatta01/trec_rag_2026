"""Plan query-local MiniLM score coverage without running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .facet_local_minilm_preflight import (
    MAX_WINDOWS_PER_DOCUMENT,
    MIN_PASSAGE_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    PAIR_MAX_TOKENS,
    PASSAGE_OVERLAP_TOKENS,
    QUERY_MAX_TOKENS,
    WINDOW_POLICY_VERSION,
    WindowPlanRow,
    build_window_plan,
    load_verified_materialization,
    load_verified_tokenizer,
    score_cache_context,
)
from .repo_env import find_repo_root, repo_cache_root
from .rerank_score_cache import GlobalScoreCache


PREFLIGHT_SCHEMA_VERSION = "adaptive-evidence-score-preflight-v1"
CONTRACT_SCHEMA_VERSION = "adaptive-evidence-contract-v1"
EXPECTED_DOCUMENT_COUNT = 8_114
EXPECTED_BROAD_COUNT = 4
EXPECTED_O0_COUNT = 24
REFERENCE_PAIRS_PER_SECOND = 300.0
REFERENCE_FIXED_SECONDS = 30.0
DEFAULT_MODEL_RECEIPT = Path(
    "outputs/rag25_facet_local_minilm_v1/model_v1/materialization.json"
)


def _generated_by(document: Mapping[str, object], facet_id: str) -> bool:
    return any(
        row.get("family") == "facet" and row.get("facet_id") == facet_id
        for row in document["provenance"]  # type: ignore[index]
    )


def _reject_protected(topic_ids: Sequence[object]) -> None:
    for topic_id in map(str, topic_ids):
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_compact_json_bytes(dict(row)) for row in rows)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    try:
        source = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line:
            raise ValueError(f"{label}:{line_number} is blank")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label}:{line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_number} must be a JSON object")
        if _compact_json_bytes(row).rstrip(b"\n") != line:
            raise ValueError(f"{label}:{line_number} is not canonical JSON")
        rows.append(row)
    return rows


def load_score_contract(
    contract_dir: Path,
    *,
    expected_document_count: int = EXPECTED_DOCUMENT_COUNT,
    expected_o0_count: int = EXPECTED_O0_COUNT,
    expected_broad_count: int = EXPECTED_BROAD_COUNT,
) -> dict[str, object]:
    """Load and authenticate the Task 1 contract before model/cache access."""

    root = Path(contract_dir)
    manifest = _read_json(root / "manifest.json", "contract manifest")
    raw_topic_ids = manifest.get("topic_ids")
    if not isinstance(raw_topic_ids, list):
        raise ValueError("contract manifest topic_ids must be an array")
    topic_ids = [str(value) for value in raw_topic_ids]
    _reject_protected(topic_ids)
    if len(topic_ids) != len(PILOT_TOPIC_IDS) or set(topic_ids) != set(
        PILOT_TOPIC_IDS
    ):
        raise ValueError("contract must contain exactly the four pilot topics")
    if (
        manifest.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or manifest.get("qrels_opened") is not False
        or manifest.get("document_count") != expected_document_count
        or manifest.get("broad_obligation_count") != expected_broad_count
        or manifest.get("o0_obligation_count") != expected_o0_count
    ):
        raise ValueError("contract manifest counts or safety binding differ")

    summary = _read_json(root / "summary.json", "contract summary")
    if (
        summary.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or summary.get("status") != "complete"
        or summary.get("topic_ids") != raw_topic_ids
        or summary.get("qrels_opened") is not False
        or summary.get("protected_topic_count") != 0
        or summary.get("document_count") != expected_document_count
        or summary.get("broad_obligation_count") != expected_broad_count
        or summary.get("o0_obligation_count") != expected_o0_count
    ):
        raise ValueError("contract summary counts or safety binding differ")
    raw_hashes = summary.get("artifact_sha256")
    if not isinstance(raw_hashes, Mapping):
        raise ValueError("contract summary artifact hashes are missing")
    artifact_names = (
        "manifest.json",
        "obligations.jsonl",
        "documents.jsonl",
        "folds.jsonl",
    )
    for name in artifact_names:
        expected = raw_hashes.get(name)
        try:
            observed = _sha256_file(root / name)
        except OSError as exc:
            raise ValueError(f"contract {name} is unreadable") from exc
        if expected != observed:
            raise ValueError(f"contract {name} hash differs")

    obligations = _read_jsonl(root / "obligations.jsonl", "contract obligations")
    documents = _read_jsonl(root / "documents.jsonl", "contract documents")
    _reject_protected(
        [row.get("topic_id") for row in obligations]
        + [row.get("topic_id") for row in documents]
    )
    if (
        len(documents) != expected_document_count
        or sum(row.get("kind") == "broad" for row in obligations)
        != expected_broad_count
        or sum(row.get("kind") == "o0" for row in obligations) != expected_o0_count
    ):
        raise ValueError("contract row counts differ from its receipt")
    if {str(row.get("topic_id")) for row in obligations} != set(PILOT_TOPIC_IDS):
        raise ValueError("contract obligation topics differ from the pilot set")
    if {str(row.get("topic_id")) for row in documents} != set(PILOT_TOPIC_IDS):
        raise ValueError("contract document topics differ from the pilot set")
    for row in documents:
        text = row.get("text")
        if not isinstance(text, str) or row.get("text_sha256") != _sha256_bytes(
            text.encode("utf-8")
        ):
            raise ValueError("contract document text hash differs")
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "obligations": obligations,
        "documents": documents,
    }


def build_score_candidates(
    contract: Mapping[str, object],
    *,
    derived: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Build broad-universal, O0 facet-local, and O1 parent-local rows."""

    obligations = [*contract["obligations"], *derived]  # type: ignore[index]
    documents = list(contract["documents"])  # type: ignore[index]
    _reject_protected(
        [row["topic_id"] for row in obligations]
        + [row["topic_id"] for row in documents]
    )
    by_id = {str(row["obligation_id"]): row for row in obligations}
    output: list[dict[str, object]] = []
    for document in documents:
        topic_id = str(document["topic_id"])
        for obligation in obligations:
            if str(obligation["topic_id"]) != topic_id:
                continue
            kind = str(obligation["kind"])
            if kind == "o0" and not _generated_by(
                document, str(obligation["source_facet_id"])
            ):
                continue
            if kind == "o1":
                parent = by_id[str(obligation["parent_id"])]
                if not _generated_by(document, str(parent["source_facet_id"])):
                    continue
            query = str(obligation["query"])
            text = str(document["text"])
            output.append(
                {
                    "topic_id": topic_id,
                    "obligation_id": str(obligation["obligation_id"]),
                    "family": kind,
                    "variant": str(obligation["obligation_id"]),
                    "rank": int(document["union_order"]),
                    "document_id": str(document["document_id"]),
                    "query": query,
                    "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "fold": int(document["fold"]),
                }
            )
    return sorted(
        output,
        key=lambda row: (row["topic_id"], row["obligation_id"], row["rank"]),
    )


def _coverage_row(
    *,
    document_count: int,
    candidate_count: int,
    windows: Sequence[WindowPlanRow],
) -> dict[str, object]:
    keys = {row.cache_key for row in windows}
    hit_keys = {row.cache_key for row in windows if row.cache_hit}
    return {
        "document_count": document_count,
        "candidate_count": candidate_count,
        "window_count": len(windows),
        "unique_pair_count": len(keys),
        "cache_hit_count": len(hit_keys),
        "cache_miss_count": len(keys - hit_keys),
        "cache_hit_window_count": sum(row.cache_hit for row in windows),
        "cache_miss_window_count": sum(not row.cache_hit for row in windows),
    }


def build_score_preflight(
    candidates: Sequence[Mapping[str, object]],
    tokenizer: object,
    cache_lookup: Callable[[str, str], float | None],
) -> dict[str, object]:
    """Build complete candidate/window/cache coverage without inference."""

    materialized = tuple(candidates)
    _reject_protected([row.get("topic_id") for row in materialized])
    if not materialized:
        raise ValueError("score preflight requires at least one candidate")
    windows: list[WindowPlanRow] = []
    cache_results: dict[str, bool] = {}
    for candidate in materialized:
        query = str(candidate["query"])
        candidate_windows = build_window_plan(candidate, tokenizer, query=query)
        for row in candidate_windows:
            if row.cache_key not in cache_results:
                cache_results[row.cache_key] = (
                    cache_lookup(row.query, row.window_text) is not None
                )
            hit = cache_results[row.cache_key]
            windows.append(replace(row, cache_hit=hit))

    by_obligation: dict[tuple[str, str], list[WindowPlanRow]] = defaultdict(list)
    by_document: dict[tuple[str, str, str], list[WindowPlanRow]] = defaultdict(list)
    by_topic: dict[str, list[WindowPlanRow]] = defaultdict(list)
    for row in windows:
        by_obligation[(row.topic_id, row.variant)].append(row)
        by_document[(row.topic_id, row.variant, row.document_id)].append(row)
        by_topic[row.topic_id].append(row)
    obligation_candidates: dict[tuple[str, str], int] = defaultdict(int)
    topic_candidates: dict[str, int] = defaultdict(int)
    obligation_documents: dict[tuple[str, str], set[str]] = defaultdict(set)
    topic_documents: dict[str, set[str]] = defaultdict(set)
    all_documents: set[tuple[str, str]] = set()
    for row in materialized:
        topic_id = str(row["topic_id"])
        obligation_id = str(row["obligation_id"])
        document_id = str(row["document_id"])
        obligation_candidates[(topic_id, obligation_id)] += 1
        topic_candidates[topic_id] += 1
        obligation_documents[(topic_id, obligation_id)].add(document_id)
        topic_documents[topic_id].add(document_id)
        all_documents.add((topic_id, document_id))

    return {
        "qrels_opened": False,
        "inference_count": 0,
        "summary": _coverage_row(
            document_count=len(all_documents),
            candidate_count=len(materialized),
            windows=windows,
        ),
        "topics": [
            {
                "topic_id": topic_id,
                **_coverage_row(
                    document_count=len(topic_documents[topic_id]),
                    candidate_count=topic_candidates[topic_id],
                    windows=by_topic[topic_id],
                ),
            }
            for topic_id in sorted(by_topic)
        ],
        "obligations": [
            {
                "topic_id": topic_id,
                "obligation_id": obligation_id,
                **_coverage_row(
                    document_count=len(
                        obligation_documents[(topic_id, obligation_id)]
                    ),
                    candidate_count=obligation_candidates[(topic_id, obligation_id)],
                    windows=by_obligation[(topic_id, obligation_id)],
                ),
            }
            for topic_id, obligation_id in sorted(by_obligation)
        ],
        "documents": [
            {
                "topic_id": topic_id,
                "obligation_id": obligation_id,
                "document_id": document_id,
                "document_token_coverage_fraction": rows[0].document_token_coverage_fraction,
                **_coverage_row(document_count=1, candidate_count=1, windows=rows),
            }
            for (topic_id, obligation_id, document_id), rows in sorted(
                by_document.items()
            )
        ],
        "windows": [row.to_dict() for row in windows],
    }


def _file_binding(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {
            "state": "absent",
            "path": str(resolved),
            "bytes": 0,
            "sha256": None,
        }
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        raise ValueError(f"bound input is not a readable regular file: {resolved}")
    return {
        "state": "present",
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def persist_score_preflight(
    output_dir: Path,
    *,
    candidates: Sequence[Mapping[str, object]],
    preflight: Mapping[str, object],
    bindings: Mapping[str, object],
) -> dict[str, object]:
    """Persist create-only candidates, windows, and their no-inference receipt."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"create-only preflight output exists: {destination}")
    raw_windows = preflight.get("windows")
    if not isinstance(raw_windows, list) or any(
        not isinstance(row, Mapping) for row in raw_windows
    ):
        raise ValueError("preflight windows must be an array of objects")
    summary = preflight.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("preflight summary must be an object")
    candidate_bytes = _jsonl_bytes(candidates)
    window_bytes = _jsonl_bytes(raw_windows)  # type: ignore[arg-type]
    unique_misses = int(summary["cache_miss_count"])
    projected = REFERENCE_FIXED_SECONDS + (
        unique_misses / REFERENCE_PAIRS_PER_SECOND
    )
    receipt: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "tokenizer_only_preflight_complete",
        "qrels_opened": False,
        "network": False,
        "external_cost_usd": 0.0,
        "inference_count": 0,
        "inference_authorized": False,
        "model_constructed": False,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "bindings": dict(bindings),
        "artifacts": {
            "candidates.jsonl": {
                "rows": len(candidates),
                "bytes": len(candidate_bytes),
                "sha256": _sha256_bytes(candidate_bytes),
            },
            "windows.jsonl": {
                "rows": len(raw_windows),
                "bytes": len(window_bytes),
                "sha256": _sha256_bytes(window_bytes),
            },
        },
        "window_policy": {
            "version": WINDOW_POLICY_VERSION,
            "pair_max_tokens": PAIR_MAX_TOKENS,
            "query_max_tokens": QUERY_MAX_TOKENS,
            "minimum_passage_tokens": MIN_PASSAGE_TOKENS,
            "passage_overlap_tokens": PASSAGE_OVERLAP_TOKENS,
            "maximum_windows_per_document": MAX_WINDOWS_PER_DOCUMENT,
        },
        "summary": dict(summary),
        "coverage_matrix": {
            "topics": list(preflight.get("topics", [])),
            "obligations": list(preflight.get("obligations", [])),
            "documents": list(preflight.get("documents", [])),
        },
        "projection_basis": {
            "pairs_per_second": REFERENCE_PAIRS_PER_SECOND,
            "fixed_seconds": REFERENCE_FIXED_SECONDS,
            "pair_count": unique_misses,
            "device_class": "local_rocm",
        },
        "projected_rocm_runtime_seconds": projected,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    for name, content in (
        ("candidates.jsonl", candidate_bytes),
        ("windows.jsonl", window_bytes),
        ("preflight.json", _pretty_json_bytes(receipt)),
    ):
        with (destination / name).open("xb") as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
    return receipt


def run_score_preflight(
    *,
    contract_dir: Path,
    output_dir: Path,
    model_receipt_path: Path | None = None,
    score_cache_root: Path | None = None,
) -> dict[str, object]:
    """Authenticate local inputs and materialize the score-coverage preflight."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"create-only preflight output exists: {destination}")
    contract = load_score_contract(Path(contract_dir))
    candidates = build_score_candidates(contract)

    repo_root = find_repo_root()
    materialization_path = Path(
        model_receipt_path or repo_root / DEFAULT_MODEL_RECEIPT
    ).resolve()
    materialization = load_verified_materialization(materialization_path)
    if (
        materialization.payload.get("model_id") != MODEL_ID
        or materialization.payload.get("revision") != MODEL_REVISION
    ):
        raise ValueError("materialized model identity differs from pinned MiniLM")
    tokenizer = load_verified_tokenizer(materialization_path)

    context = score_cache_context()
    cache_root = Path(
        score_cache_root or repo_cache_root(repo_root) / "reranker"
    ).resolve()
    expected_cache_path = cache_root.joinpath(*context.path_parts)
    cache_binding = _file_binding(expected_cache_path)
    cache = GlobalScoreCache(cache_root, context)
    if cache.path.resolve() != expected_cache_path.resolve():
        raise ValueError("score cache path differs from the pinned cache context")
    if _file_binding(expected_cache_path) != cache_binding:
        raise ValueError("score cache changed while it was being loaded")
    plan = build_score_preflight(
        candidates,
        tokenizer,
        cache_lookup=lambda query, text: cache.get(query_text=query, text=text),
    )
    if materialization.receipt_path.read_bytes() != materialization.source:
        raise ValueError("model materialization receipt changed during preflight")
    if _file_binding(expected_cache_path) != cache_binding:
        raise ValueError("score cache changed during preflight")

    contract_summary_path = Path(contract_dir) / "summary.json"
    contract_summary = _read_json(contract_summary_path, "contract summary")
    source_files = {
        "adaptive_evidence_score.py": Path(__file__),
        "adaptive_evidence_contract.py": Path(__file__).with_name(
            "adaptive_evidence_contract.py"
        ),
        "facet_local_minilm_preflight.py": Path(__file__).with_name(
            "facet_local_minilm_preflight.py"
        ),
        "rerank_score_cache.py": Path(__file__).with_name("rerank_score_cache.py"),
    }
    bindings: dict[str, object] = {
        "contract": {
            "path": str(Path(contract_dir).resolve()),
            "summary_sha256": _sha256_file(contract_summary_path),
            "artifact_sha256": contract_summary["artifact_sha256"],
        },
        "model_materialization": {
            "receipt_path": str(materialization.receipt_path),
            "receipt_sha256": materialization.sha256,
            "snapshot_sha256": materialization.payload["snapshot_sha256"],
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "local_files_only": True,
            "trust_remote_code": False,
            "use_fast": True,
        },
        "score_cache": {
            "root": str(cache_root),
            "path": str(cache.path.resolve()),
            "context": context.artifact_metadata,
            "binding": cache_binding,
        },
        "code_sha256": {
            name: _sha256_file(path) for name, path in source_files.items()
        },
    }
    return persist_score_preflight(
        destination,
        candidates=candidates,
        preflight=plan,
        bindings=bindings,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="audit local MiniLM score coverage without inference"
    )
    preflight.add_argument("--contract", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command != "preflight":  # pragma: no cover - argparse enforces this.
        parser.error("unsupported command")
    receipt = run_score_preflight(
        contract_dir=args.contract,
        output_dir=args.output,
    )
    summary = receipt["summary"]
    assert isinstance(summary, Mapping)
    print(
        "status=complete "
        f"documents={summary['document_count']} "
        f"candidates={summary['candidate_count']} "
        f"windows={summary['window_count']} "
        f"pairs={summary['unique_pair_count']} "
        f"cache_hits={summary['cache_hit_count']} "
        f"cache_misses={summary['cache_miss_count']} "
        f"projected_rocm_seconds={receipt['projected_rocm_runtime_seconds']:.3f} "
        "network=false qrels_opened=false external_cost=$0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
