"""Authenticated, tokenizer-only preflight for narrative-tethered facet scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from .facet_local_minilm_preflight import (
    MODEL_ID,
    MODEL_REVISION,
    build_window_plan,
    load_verified_materialization,
    load_verified_tokenizer,
    score_cache_context,
)
from .rerank_score_cache import GlobalScoreCache


TOPIC_IDS = ("219", "72", "300", "84")
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
PAIR_COUNT = 4_800
WINDOW_CEILING = 25_000
RUNTIME_CEILING_SECONDS = 600.0
SCORE_CACHE_ROOT = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/reranker"
)
PREFLIGHT_SCHEMA_VERSION = "tethered-facet-minilm-preflight-v1"
CANDIDATE_SCHEMA_VERSION = "tethered-facet-minilm-candidate-v1"


class _TokenizerOnlyBackend:
    """Small adapter exposing only the tokenization operations used by planning."""

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> list[int]:
        if add_special_tokens or truncation:
            raise ValueError("tokenizer-only planning requires raw, untruncated tokens")
        return list(
            self._tokenizer.encode(  # type: ignore[attr-defined]
                text, add_special_tokens=False
            ).ids
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        if skip_special_tokens or clean_up_tokenization_spaces:
            raise ValueError("tokenizer-only planning requires exact token decoding")
        return str(
            self._tokenizer.decode(  # type: ignore[attr-defined]
                list(token_ids), skip_special_tokens=False
            )
        )

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        return int(
            self._tokenizer.post_processor.num_special_tokens_to_add(  # type: ignore[attr-defined]
                pair
            )
        )


class TokenizerOnlyAuto:
    """`AutoTokenizer`-shaped local loader that never imports a model runtime."""

    @classmethod
    def from_pretrained(
        cls,
        snapshot: Path,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
        use_fast: bool,
    ) -> _TokenizerOnlyBackend:
        if not local_files_only or trust_remote_code or not use_fast:
            raise ValueError("tokenizer-only loader requires the frozen safe settings")
        from tokenizers import Tokenizer

        return _TokenizerOnlyBackend(Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json")))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _compact_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_compact_bytes(row) + b"\n" for row in rows)


def _read_object(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        source = Path(path).read_bytes()
        value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, source


def _read_jsonl_source(source: bytes, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = source.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    for number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{number} must be an object")
        rows.append(value)
    return rows


def _exclusive_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as sink:
            sink.write(value)
            sink.flush()
            os.fsync(sink.fileno())
    finally:
        os.close(descriptor)


def render_tethered_query(narrative: str, facet_query: str) -> str:
    narrative = narrative.strip()
    facet_query = facet_query.strip()
    if not narrative:
        raise ValueError("narrative is required")
    if not facet_query:
        raise ValueError("facet query is required")
    return f"{narrative}\n\nFocus: {facet_query}"


def reject_topic(value: object) -> str:
    topic_id = str(value)
    if topic_id in PROTECTED_TOPIC_IDS:
        raise ValueError(f"protected topic {topic_id} is forbidden")
    if topic_id not in TOPIC_IDS:
        raise ValueError(f"unexpected topic {topic_id}")
    return topic_id


def _rows(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{label} rows must be objects")
    return list(value)  # type: ignore[arg-type]


def _required_text(row: Mapping[str, object], key: str, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} {key} must be non-empty text")
    return value


def build_tethered_candidates(
    manifest: Mapping[str, object],
    phase1_rows: Sequence[Mapping[str, object]],
    gate_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Join accepted facets to their phase-1 candidates without losing identity."""

    topic_ids = manifest.get("topic_ids")
    if topic_ids != list(TOPIC_IDS):
        raise ValueError("manifest topic IDs differ from the frozen order")
    topics = _rows(manifest.get("topics"), "manifest topics")
    facets = _rows(manifest.get("facets"), "manifest facets")

    narratives: dict[str, str] = {}
    for topic in topics:
        topic_id = reject_topic(topic.get("topic_id"))
        if topic_id in narratives:
            raise ValueError("manifest has duplicate topic IDs")
        narratives[topic_id] = _required_text(topic, "query", "manifest topic")
    if set(narratives) != set(TOPIC_IDS):
        raise ValueError("manifest topics do not cover the frozen topic IDs")

    facet_by_id: dict[str, Mapping[str, object]] = {}
    for facet in facets:
        topic_id = reject_topic(facet.get("topic_id"))
        facet_id = _required_text(facet, "facet_id", "manifest facet")
        facet_query = _required_text(facet, "query", "manifest facet")
        order = facet.get("manifest_order")
        if type(order) is not int or order < 0:
            raise ValueError("manifest facet order must be a non-negative integer")
        if facet_id in facet_by_id:
            raise ValueError("manifest has duplicate facet IDs")
        facet_by_id[facet_id] = {
            **facet,
            "topic_id": topic_id,
            "query": facet_query,
        }

    accepted: dict[str, Mapping[str, object]] = {}
    for gate in gate_rows:
        topic_id = reject_topic(gate.get("topic_id"))
        facet_id = _required_text(gate, "facet_id", "gate")
        if gate.get("status") != "accepted":
            continue
        facet = facet_by_id.get(facet_id)
        if facet is None or facet["topic_id"] != topic_id:
            raise ValueError("accepted gate is absent from the manifest")
        if gate.get("manifest_order") != facet.get("manifest_order"):
            raise ValueError("accepted gate manifest order differs")
        if facet_id in accepted:
            raise ValueError("duplicate accepted gate")
        accepted[facet_id] = facet

    output: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for source in phase1_rows:
        topic_id = reject_topic(source.get("topic_id"))
        facet_id = _required_text(source, "facet_id", "phase-1 candidate")
        facet = accepted.get(facet_id)
        if facet is None:
            continue
        if topic_id != facet["topic_id"]:
            raise ValueError("phase-1 candidate topic differs from facet")
        if source.get("manifest_order") != facet.get("manifest_order"):
            raise ValueError("phase-1 candidate manifest order differs")
        facet_query = str(facet["query"])
        if source.get("query") != facet_query:
            raise ValueError("phase-1 candidate query differs from manifest")
        source_query_hash = source.get("query_sha256")
        if source_query_hash is not None and source_query_hash != _sha256_text(
            facet_query
        ):
            raise ValueError("phase-1 candidate query hash differs")
        document_id = _required_text(
            source, "document_id", "phase-1 candidate"
        )
        if source.get("docid", document_id) != document_id:
            raise ValueError("phase-1 candidate document identities differ")
        identity = (facet_id, document_id)
        if identity in seen:
            raise ValueError("duplicate phase-1 facet/document identity")
        seen.add(identity)
        rank = source.get("rank")
        if type(rank) is not int or rank <= 0:
            raise ValueError("phase-1 candidate rank must be a positive integer")
        text = _required_text(source, "text", "phase-1 candidate")
        source_text_hash = source.get("text_sha256")
        if source_text_hash is not None and source_text_hash != _sha256_text(text):
            raise ValueError("phase-1 candidate text hash differs")
        query = render_tethered_query(narratives[topic_id], facet_query)
        output.append(
            {
                **source,
                "topic_id": topic_id,
                "family": "tethered_facet",
                "variant": facet_id,
                "facet_id": facet_id,
                "facet_query": facet_query,
                "facet_query_sha256": _sha256_text(facet_query),
                "query": query,
                "query_sha256": _sha256_text(query),
                "document_id": document_id,
                "docid": document_id,
                "text": text,
                "text_sha256": _sha256_text(text),
                "prior_bm25_rank": rank,
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )

    missing = set(accepted) - {str(row["facet_id"]) for row in output}
    if missing:
        raise ValueError("accepted facets lack phase-1 candidates")
    output.sort(
        key=lambda row: (
            TOPIC_IDS.index(str(row["topic_id"])),
            int(row["manifest_order"]),
            int(row["prior_bm25_rank"]),
            str(row["document_id"]),
        )
    )
    return output


def enforce_preflight_ceiling(
    pair_count: int, window_count: int, projected_seconds: float
) -> None:
    """Reject any drift from the frozen population or resource ceilings."""

    if (
        pair_count != PAIR_COUNT
        or window_count > WINDOW_CEILING
        or projected_seconds > RUNTIME_CEILING_SECONDS
    ):
        raise ValueError(
            "preflight ceiling violated: "
            f"pairs={pair_count}, windows={window_count}, seconds={projected_seconds}"
        )


def _source_path(
    sources: Mapping[str, object], key: str, *, child: str | None = None
) -> Path:
    value = sources.get(key)
    if not isinstance(value, (str, Path)):
        raise ValueError(f"preflight source {key} is required")
    path = Path(value)
    return path / child if child is not None else path


def _validate_manifest_hashes(manifest: Mapping[str, object]) -> None:
    for topic in _rows(manifest.get("topics"), "manifest topics"):
        reject_topic(topic.get("topic_id"))
    for facet in _rows(manifest.get("facets"), "manifest facets"):
        reject_topic(facet.get("topic_id"))
    if manifest.get("qrels_opened") not in (None, False):
        raise ValueError("manifest records qrels access")
    hashes = manifest.get("hashes")
    if hashes is None:
        return
    if not isinstance(hashes, Mapping):
        raise ValueError("manifest hashes must be an object")
    expected_topics = _sha256_bytes(_compact_bytes(manifest.get("topics")) + b"\n")
    expected_facets = _sha256_bytes(_compact_bytes(manifest.get("facets")) + b"\n")
    unhashed = {key: value for key, value in manifest.items() if key != "hashes"}
    expected_freeze = _sha256_bytes(_compact_bytes(unhashed) + b"\n")
    if (
        hashes.get("topics_sha256") != expected_topics
        or hashes.get("facets_sha256") != expected_facets
        or hashes.get("freeze_sha256") != expected_freeze
    ):
        raise ValueError("manifest hashes differ from canonical content")


def _cache_binding(cache: object) -> dict[str, object]:
    value = getattr(cache, "path", None)
    if not isinstance(value, (str, Path)):
        raise ValueError("score cache must expose its bound path")
    path = Path(value).resolve()
    if not path.exists():
        return {"state": "absent", "path": str(path), "bytes": 0, "sha256": None}
    if not path.is_file():
        raise ValueError("score cache path must be a regular file")
    source = path.read_bytes()
    return {
        "state": "present",
        "path": str(path),
        "bytes": len(source),
        "sha256": _sha256_bytes(source),
    }


def _materialization_path(
    sources: Mapping[str, object],
    phase1_dir: Path,
    scoring_receipt: Mapping[str, object],
) -> tuple[Path, str | None]:
    explicit = sources.get("model_receipt")
    expected = scoring_receipt.get("model_materialization_receipt_sha256")
    if isinstance(explicit, (str, Path)):
        return Path(explicit), str(expected) if isinstance(expected, str) else None
    receipt_value = scoring_receipt.get("model_materialization_receipt")
    if isinstance(receipt_value, str):
        return Path(receipt_value), str(expected) if isinstance(expected, str) else None
    preflight, _source = _read_object(
        phase1_dir / "preflight.json", "phase-1 preflight receipt"
    )
    value = preflight.get("model_materialization_receipt")
    sha = preflight.get("model_materialization_receipt_sha256")
    if not isinstance(value, str) or not isinstance(sha, str):
        raise ValueError("phase-1 receipt lacks model materialization bindings")
    return Path(value), sha


def create_preflight(
    sources: Mapping[str, object],
    output: Path,
    *,
    tokenizer: object | None = None,
    cache: object | None = None,
) -> dict[str, object]:
    """Authenticate sealed inputs and persist a create-only tokenizer plan."""

    started = time.perf_counter()
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"create-only preflight output exists: {destination}")

    manifest_path = _source_path(sources, "manifest")
    phase1_dir = _source_path(sources, "phase1")
    gate_dir = _source_path(sources, "gate")
    manifest, manifest_source = _read_object(manifest_path, "deep-facet manifest")
    _validate_manifest_hashes(manifest)
    scoring_receipt_path = phase1_dir / "scoring_receipt.json"
    scoring_receipt, scoring_source = _read_object(
        scoring_receipt_path, "phase-1 scoring receipt"
    )
    if (
        scoring_receipt.get("status") != "complete"
        or scoring_receipt.get("phase") != "phase1_facet_local"
        or scoring_receipt.get("qrels_opened") is not False
        or scoring_receipt.get("network_access_supported") is not False
        or scoring_receipt.get("hosted_inference_supported") is not False
        or scoring_receipt.get("model") != MODEL_ID
        or scoring_receipt.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("phase-1 scoring receipt identity differs")
    expected_manifest_sha = scoring_receipt.get("manifest_sha256")
    if isinstance(expected_manifest_sha, str) and expected_manifest_sha != _sha256_bytes(
        manifest_source
    ):
        raise ValueError("phase-1 manifest hash differs")

    phase1_preflight: dict[str, object] | None = None
    expected_preflight_sha = scoring_receipt.get("preflight_sha256")
    if isinstance(expected_preflight_sha, str):
        phase1_preflight_path = phase1_dir / "preflight.json"
        phase1_preflight, phase1_preflight_source = _read_object(
            phase1_preflight_path, "phase-1 preflight receipt"
        )
        if _sha256_bytes(phase1_preflight_source) != expected_preflight_sha:
            raise ValueError("phase-1 preflight receipt hash differs")
        if (
            phase1_preflight.get("status") != "tokenizer_only_preflight_complete"
            or phase1_preflight.get("qrels_opened") is not False
        ):
            raise ValueError("phase-1 preflight receipt identity differs")
    for filename, field in (
        ("windows.jsonl", "windows_sha256"),
        ("scores.jsonl", "scores_sha256"),
    ):
        expected = scoring_receipt.get(field)
        if isinstance(expected, str) and _sha256_file(phase1_dir / filename) != expected:
            raise ValueError(f"phase-1 {filename} receipt hash differs")

    gate_summary_path = gate_dir / "summary.json"
    gates_path = gate_dir / "gates.json"
    gate_summary, gate_summary_source = _read_object(gate_summary_path, "gate summary")
    gates, gates_source = _read_object(gates_path, "gates")
    artifact = gate_summary.get("artifacts")
    gate_binding = artifact.get("gates.json") if isinstance(artifact, Mapping) else None
    if (
        gate_summary.get("status") != "complete"
        or gate_summary.get("qrels_opened") is not False
        or not isinstance(gate_binding, Mapping)
        or gate_binding.get("bytes") != len(gates_source)
        or gate_binding.get("sha256") != _sha256_bytes(gates_source)
    ):
        raise ValueError("gate summary does not authenticate gates.json")
    raw_gates = gates.get("gates")
    gate_rows = _rows(raw_gates, "gates")
    accepted_count = sum(row.get("status") == "accepted" for row in gate_rows)
    if gate_summary.get("accepted_facet_count") != accepted_count:
        raise ValueError("gate accepted-facet count differs")

    model_receipt_path, expected_model_sha = _materialization_path(
        sources, phase1_dir, scoring_receipt
    )
    materialization = load_verified_materialization(model_receipt_path)
    if expected_model_sha is not None and materialization.sha256 != expected_model_sha:
        raise ValueError("model materialization receipt hash differs")
    if (
        materialization.payload.get("model_id") != MODEL_ID
        or materialization.payload.get("revision") != MODEL_REVISION
    ):
        raise ValueError("model materialization identity differs")

    # Candidate bytes are opened only after all controlling receipts are authenticated.
    candidates_path = phase1_dir / "candidates.jsonl"
    try:
        phase1_source = candidates_path.read_bytes()
    except OSError as exc:
        raise ValueError("phase-1 candidates are unreadable") from exc
    expected_candidates_sha = scoring_receipt.get("candidates_sha256")
    if expected_candidates_sha is None and phase1_preflight is not None:
        expected_candidates_sha = phase1_preflight.get("candidates_sha256")
    if isinstance(expected_candidates_sha, str) and expected_candidates_sha != _sha256_bytes(
        phase1_source
    ):
        raise ValueError("phase-1 candidates hash differs")
    phase1_rows = _read_jsonl_source(phase1_source, "phase-1 candidates")
    expected_rows = scoring_receipt.get("candidate_rows")
    if isinstance(expected_rows, int) and expected_rows != len(phase1_rows):
        raise ValueError("phase-1 candidate row count differs")
    candidates = build_tethered_candidates(manifest, phase1_rows, gate_rows)
    if isinstance(expected_rows, int) and len(candidates) != expected_rows:
        raise ValueError("accepted facets do not cover the authenticated population")
    facet_counts = Counter(str(row["facet_id"]) for row in candidates)
    if accepted_count == 0 or set(facet_counts.values()) != {len(candidates) // accepted_count}:
        raise ValueError("accepted facets do not have an exact equal population")
    if len(candidates) == PAIR_COUNT and set(facet_counts.values()) != {200}:
        raise ValueError("each accepted facet requires exactly 200 candidates")

    if tokenizer is None:
        tokenizer = load_verified_tokenizer(
            model_receipt_path, auto_tokenizer_cls=TokenizerOnlyAuto
        )
    if cache is None:
        cache_root = sources.get("cache_root", SCORE_CACHE_ROOT)
        cache = GlobalScoreCache(Path(cache_root), score_cache_context())
    cache_before = _cache_binding(cache)

    window_rows: list[dict[str, object]] = []
    hits = 0
    topic_windows: Counter[str] = Counter()
    facet_windows: Counter[str] = Counter()
    coverage: dict[str, list[float]] = {}
    for candidate in candidates:
        planned = build_window_plan(candidate, tokenizer, query=str(candidate["query"]))
        local_coverage: list[float] = []
        for row in planned:
            cached = cache.get(query_text=row.query, text=row.window_text) is not None  # type: ignore[attr-defined]
            hits += int(cached)
            materialized = replace(row, cache_hit=cached).to_dict()
            materialized["facet_id"] = candidate["facet_id"]
            materialized["manifest_order"] = candidate["manifest_order"]
            materialized["prior_bm25_rank"] = candidate["prior_bm25_rank"]
            window_rows.append(materialized)
            local_coverage.append(row.document_token_coverage_fraction)
            topic_windows[str(candidate["topic_id"])] += 1
            facet_windows[str(candidate["facet_id"])] += 1
        coverage[f"{candidate['facet_id']}:{candidate['document_id']}"] = local_coverage
    elapsed = time.perf_counter() - started
    if len(window_rows) > WINDOW_CEILING or elapsed > RUNTIME_CEILING_SECONDS:
        raise ValueError("preflight ceiling violated while planning windows")
    if _cache_binding(cache) != cache_before:
        raise ValueError("score cache changed during tokenizer-only preflight")

    candidate_output = [
        {"schema_version": CANDIDATE_SCHEMA_VERSION, **row} for row in candidates
    ]
    candidate_bytes = _jsonl_bytes(candidate_output)
    window_bytes = _jsonl_bytes(window_rows)
    topic_counts = Counter(str(row["topic_id"]) for row in candidates)
    summary: dict[str, object] = {
        "query_document_pair_count": len(candidates),
        "window_count": len(window_rows),
        "cache_hit_window_count": hits,
        "cache_miss_window_count": len(window_rows) - hits,
        "topic_pair_counts": dict(sorted(topic_counts.items(), key=lambda item: TOPIC_IDS.index(item[0]))),
        "facet_pair_counts": dict(sorted(facet_counts.items())),
        "topic_window_counts": dict(sorted(topic_windows.items(), key=lambda item: TOPIC_IDS.index(item[0]))),
        "facet_window_counts": dict(sorted(facet_windows.items())),
        "document_window_coverage": coverage,
    }
    source_bindings = {
        "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256_bytes(manifest_source)},
        "phase1_scoring_receipt": {"path": str(scoring_receipt_path.resolve()), "sha256": _sha256_bytes(scoring_source)},
        "phase1_candidates": {"path": str(candidates_path.resolve()), "sha256": _sha256_bytes(phase1_source)},
        "gate_summary": {"path": str(gate_summary_path.resolve()), "sha256": _sha256_bytes(gate_summary_source)},
        "gates": {"path": str(gates_path.resolve()), "sha256": _sha256_bytes(gates_source)},
    }
    payload: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "tokenizer_only_preflight_complete",
        "qrels_opened": False,
        "retrieval_path_supported": False,
        "network_access_supported": False,
        "hosted_inference_supported": False,
        "model_constructed": False,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_materialization_receipt": str(materialization.receipt_path),
        "model_materialization_receipt_sha256": materialization.sha256,
        "tokenizer": {"class": type(tokenizer).__name__, "local_files_only": True},
        "score_cache": {"context": score_cache_context().artifact_metadata, "binding": cache_before},
        "sources": source_bindings,
        "candidates_file": "candidates.jsonl",
        "candidates_sha256": _sha256_bytes(candidate_bytes),
        "windows_file": "windows.jsonl",
        "windows_sha256": _sha256_bytes(window_bytes),
        "summary": summary,
        "ceilings": {
            "exact_query_document_pair_count": PAIR_COUNT,
            "maximum_window_count": WINDOW_CEILING,
            "maximum_runtime_seconds": RUNTIME_CEILING_SECONDS,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    _exclusive_write(destination / "candidates.jsonl", candidate_bytes)
    _exclusive_write(destination / "windows.jsonl", window_bytes)
    _exclusive_write(destination / "preflight.json", _pretty_bytes(payload))
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--manifest", required=True, type=Path)
    preflight.add_argument("--phase1", required=True, type=Path)
    preflight.add_argument("--gate", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)
    preflight.add_argument("--cache-root", type=Path, default=SCORE_CACHE_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = create_preflight(
        {
            "manifest": args.manifest,
            "phase1": args.phase1,
            "gate": args.gate,
            "cache_root": args.cache_root,
        },
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
