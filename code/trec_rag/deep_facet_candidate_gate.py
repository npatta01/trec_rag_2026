"""Qrels-blind stream gate and raw/accepted unions for deep facet retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from .deep_facet_candidate_manifest import (
    EXCLUDED_TOPIC_IDS,
    TOPIC_IDS,
    assert_mutation_allowed,
    load_manifest,
)
from .facet_aware_fusion_rank import (
    quality_gate as _legacy_quality_gate,
)
from .facet_local_minilm_rank import aggregate_top4
from .remote_client import extract_text


GATE_SCHEMA_VERSION = "deep-facet-candidate-gate-v1"
UNION_SCHEMA_VERSION = "deep-facet-candidate-union-v1"
PREFIX_DEPTHS = (50, 100, 200)
SOURCE_CACHE_ROOT = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote"
)


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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    rows: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{number} must be an object")
        rows.append(row)
    return rows


def _docid(row: Mapping[str, object]) -> str:
    value = row.get("document_id", row.get("docid"))
    if not isinstance(value, str) or not value:
        raise ValueError("candidate document ID must be non-empty text")
    return value


def quality_gate(facet, ranked_docs):
    """Apply only the three top-five checks frozen in the approved design.

    The older helper also reports whether singular literal gate terms occur in
    the query text.  That remains useful diagnostic metadata, but morphology in
    an already frozen query cannot reject an otherwise coherent result stream.
    """

    decision = _legacy_quality_gate(facet, ranked_docs)
    failed = tuple(check for check in decision.failed_checks if check != "structural")
    return replace(decision, accepted=not failed, failed_checks=failed)


def _topic(row: Mapping[str, object]) -> str:
    value = str(row.get("topic_id"))
    if value in EXCLUDED_TOPIC_IDS:
        raise ValueError(f"excluded topic {value} is forbidden")
    return value


def _ordered_ids(rows: Sequence[Mapping[str, object]], depth: int | None = None) -> list[str]:
    ordered = sorted(rows, key=lambda row: (int(row["rank"]), _docid(row)))
    if depth is not None:
        ordered = ordered[:depth]
    result: list[str] = []
    seen: set[str] = set()
    for row in ordered:
        docid = _docid(row)
        if docid not in seen:
            seen.add(docid)
            result.append(docid)
    return result


def rank_facet_documents_200(
    candidates: Sequence[Mapping[str, object]],
    scored_windows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate and rank one complete depth-200 stream without old depth caps."""

    if len(candidates) != 200:
        raise ValueError("facet-local ranking requires exactly 200 candidates")
    ordered = sorted(candidates, key=lambda row: (int(row["rank"]), _docid(row)))
    candidate_ids = [_docid(row) for row in ordered]
    if len(set(candidate_ids)) != 200:
        raise ValueError("facet-local candidates must have unique document IDs")
    by_document: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for window in scored_windows:
        by_document[_docid(window)].append(window)
    if set(by_document) != set(candidate_ids):
        raise ValueError("scored windows must cover all 200 facet candidates")
    aggregated: list[dict[str, object]] = []
    for candidate in ordered:
        document_id = _docid(candidate)
        row = dict(candidate)
        row.update(
            {
                "document_id": document_id,
                "retrieval_rank": int(candidate["rank"]),
                "score": aggregate_top4(by_document[document_id]),
            }
        )
        aggregated.append(row)
    aggregated.sort(
        key=lambda row: (
            -float(row["score"]),
            int(row["retrieval_rank"]),
            str(row["document_id"]),
        )
    )
    for rank, row in enumerate(aggregated, start=1):
        row["rank"] = rank
    return aggregated


def _stable_union(parts: Sequence[Sequence[str]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        for docid in part:
            if docid not in seen:
                seen.add(docid)
                result.append(docid)
    return result


def build_unions(
    original: Mapping[str, Sequence[Mapping[str, object]]],
    streams: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, list[str]]]:
    """Return per-topic raw and gate-accepted document-ID unions."""

    result: dict[str, dict[str, list[str]]] = {}
    topics = list(original)
    for topic_id in topics:
        if topic_id in EXCLUDED_TOPIC_IDS:
            raise ValueError(f"excluded topic {topic_id} is forbidden")
        original_ids = _ordered_ids(original[topic_id])
        topic_streams = [row for row in streams if str(row.get("topic_id")) == topic_id]
        raw_parts = [original_ids]
        accepted_parts = [original_ids]
        for stream in topic_streams:
            bm25 = stream.get("bm25")
            if not isinstance(bm25, Sequence) or isinstance(bm25, (str, bytes)):
                raise ValueError("stream bm25 ranking must be a sequence")
            ids = _ordered_ids(bm25)  # type: ignore[arg-type]
            raw_parts.append(ids)
            if stream.get("accepted") is True:
                accepted_parts.append(ids)
        result[topic_id] = {
            "raw_docids": _stable_union(raw_parts),
            "accepted_docids": _stable_union(accepted_parts),
        }
    return result


def build_prefix_unions(
    original: Mapping[str, Sequence[Mapping[str, object]]],
    streams: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, list[str]]]:
    """Freeze accepted BM25 and MiniLM facet-prefix unions at 50/100/200."""

    result: dict[str, dict[str, list[str]]] = {}
    for topic_id, original_rows in original.items():
        original_ids = _ordered_ids(original_rows)
        accepted = [
            row
            for row in streams
            if str(row.get("topic_id")) == topic_id and row.get("accepted") is True
        ]
        result[topic_id] = {}
        for family in ("bm25", "minilm"):
            for depth in PREFIX_DEPTHS:
                parts = [original_ids]
                for stream in accepted:
                    ranking = stream.get(family)
                    if not isinstance(ranking, Sequence) or isinstance(
                        ranking, (str, bytes)
                    ):
                        raise ValueError(f"stream {family} ranking must be a sequence")
                    parts.append(_ordered_ids(ranking, depth))  # type: ignore[arg-type]
                result[topic_id][f"{family}@{depth}"] = _stable_union(parts)
    return result


def _load_original_candidates(
    manifest: Mapping[str, object], cache_root: Path
) -> dict[str, list[dict[str, object]]]:
    raw_topics = manifest.get("topics")
    if not isinstance(raw_topics, list):
        raise ValueError("manifest topics must be an array")
    result: dict[str, list[dict[str, object]]] = {}
    for topic in raw_topics:
        if not isinstance(topic, Mapping):
            raise ValueError("manifest topic must be an object")
        topic_id = str(topic.get("topic_id"))
        if topic_id in EXCLUDED_TOPIC_IDS:
            raise ValueError(f"excluded topic {topic_id} is forbidden")
        path = Path(cache_root) / str(topic.get("original_cache_filename"))
        raw = path.read_bytes()
        if _sha256(raw) != topic.get("original_cache_sha256"):
            raise ValueError("original cache hash differs from manifest")
        payload = json.loads(raw)
        candidates = payload.get("response", {}).get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1000:
            raise ValueError("original cache must contain exactly 1,000 candidates")
        rows: list[dict[str, object]] = []
        for position, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, Mapping):
                raise ValueError("original candidate must be an object")
            docid = candidate.get("docid")
            text = extract_text(candidate.get("doc") or candidate)
            if not isinstance(docid, str) or not docid or not text:
                raise ValueError("original candidate lacks ID or text")
            rows.append(
                {
                    "topic_id": topic_id,
                    "document_id": docid,
                    "docid": docid,
                    "rank": position,
                    "text": text,
                    "score": float(candidate.get("score", 0.0)),
                    "source": "original",
                }
            )
        result[topic_id] = rows
    if list(result) != list(TOPIC_IDS):
        raise ValueError("original candidates differ from frozen topic order")
    return result


def _rank_streams(
    manifest: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    scores: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    raw_facets = manifest.get("facets")
    if not isinstance(raw_facets, list):
        raise ValueError("manifest facets must be an array")
    candidate_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    score_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in candidates:
        candidate_groups[str(row.get("facet_id"))].append(row)
    for row in scores:
        score_groups[str(row.get("variant"))].append(row)
    streams: list[dict[str, object]] = []
    for facet in raw_facets:
        if not isinstance(facet, Mapping):
            raise ValueError("manifest facet must be an object")
        topic_id = _topic(facet)
        facet_id = str(facet.get("facet_id"))
        bm25 = sorted(
            candidate_groups.get(facet_id, []),
            key=lambda row: (int(row["rank"]), _docid(row)),
        )
        if len(bm25) != 200:
            raise ValueError("each facet requires exactly 200 phase-1 candidates")
        minilm = rank_facet_documents_200(bm25, score_groups.get(facet_id, []))
        decision = quality_gate(facet, minilm)
        streams.append(
            {
                "topic_id": topic_id,
                "facet_id": facet_id,
                "manifest_order": facet["manifest_order"],
                "accepted": decision.accepted,
                "status": "accepted" if decision.accepted else "rejected",
                "gate": decision.to_dict(),
                "bm25": [dict(row) for row in bm25],
                "minilm": minilm,
            }
        )
    return streams


def _union_rows(
    original: Mapping[str, Sequence[Mapping[str, object]]],
    streams: Sequence[Mapping[str, object]],
    union_ids: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    kind: str,
) -> list[dict[str, object]]:
    docs: dict[tuple[str, str], dict[str, object]] = {}
    provenance: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for topic_id, rows in original.items():
        for row in rows:
            key = (topic_id, _docid(row))
            docs[key] = dict(row)
            provenance[key].append({"family": "original", "rank": row["rank"]})
    for stream in streams:
        topic_id = str(stream["topic_id"])
        facet_id = str(stream["facet_id"])
        for row in stream["bm25"]:  # type: ignore[index]
            key = (topic_id, _docid(row))
            existing = docs.get(key)
            if existing is not None and existing.get("text") != row.get("text"):
                raise ValueError("one document ID has conflicting text across streams")
            docs.setdefault(key, dict(row))
            provenance[key].append(
                {
                    "family": "facet",
                    "facet_id": facet_id,
                    "accepted": stream["accepted"],
                    "bm25_rank": row["rank"],
                }
            )
    field = "raw_docids" if kind == "raw" else "accepted_docids"
    output: list[dict[str, object]] = []
    for topic_id in TOPIC_IDS:
        for union_order, docid in enumerate(union_ids[topic_id][field], start=1):
            row = docs[(topic_id, docid)]
            text = str(row["text"])
            output.append(
                {
                    "schema_version": UNION_SCHEMA_VERSION,
                    "union": kind,
                    "topic_id": topic_id,
                    "document_id": docid,
                    "text": text,
                    "text_sha256": _sha256(text.encode("utf-8")),
                    "union_order": union_order,
                    "provenance": provenance[(topic_id, docid)],
                }
            )
    return output


def freeze_gate(
    manifest: Mapping[str, object],
    retrieval_dir: Path,
    phase1_dir: Path,
    output: Path,
    *,
    cache_root: Path = SOURCE_CACHE_ROOT,
) -> dict[str, object]:
    output = Path(output)
    assert_mutation_allowed(output)
    assert_mutation_allowed(output.parent)
    if output.exists():
        raise FileExistsError(f"create-only gate output already exists: {output}")
    retrieval_summary = _read_json(
        Path(retrieval_dir) / "retrieval_summary.json", "retrieval summary"
    )
    phase1_preflight = _read_json(
        Path(phase1_dir) / "preflight.json", "phase-1 preflight"
    )
    scoring_receipt = _read_json(
        Path(phase1_dir) / "scoring_receipt.json", "phase-1 scoring receipt"
    )
    if (
        retrieval_summary.get("complete") is not True
        or retrieval_summary.get("qrels_opened") is not False
        or scoring_receipt.get("status") != "complete"
        or scoring_receipt.get("qrels_opened") is not False
        or scoring_receipt.get("preflight_sha256")
        != _sha256((Path(phase1_dir) / "preflight.json").read_bytes())
        or scoring_receipt.get("scores_sha256")
        != _sha256((Path(phase1_dir) / "scores.jsonl").read_bytes())
    ):
        raise ValueError("phase-1 evidence is incomplete or unauthenticated")
    candidates = _read_jsonl(Path(phase1_dir) / "candidates.jsonl", "phase-1 candidates")
    scores = _read_jsonl(Path(phase1_dir) / "scores.jsonl", "phase-1 scores")
    if (
        len(candidates) != 5000
        or len(scores) != int(scoring_receipt["completed_window_count"])
        or phase1_preflight.get("candidates_sha256")
        != _sha256((Path(phase1_dir) / "candidates.jsonl").read_bytes())
    ):
        raise ValueError("phase-1 candidates or scores are incomplete")
    streams = _rank_streams(manifest, candidates, scores)
    original = _load_original_candidates(manifest, cache_root)
    unions = build_unions(original, streams)
    prefixes = build_prefix_unions(original, streams)
    if any(len(value["accepted_docids"]) < 1000 for value in unions.values()):
        raise ValueError("accepted union must retain all 1,000 original candidates")
    raw_rows = _union_rows(original, streams, unions, kind="raw")
    accepted_rows = _union_rows(original, streams, unions, kind="accepted")
    gate_rows = [
        {
            "topic_id": stream["topic_id"],
            "facet_id": stream["facet_id"],
            "manifest_order": stream["manifest_order"],
            "status": stream["status"],
            "accepted": stream["accepted"],
            "gate": stream["gate"],
        }
        for stream in streams
    ]
    stream_rows: list[dict[str, object]] = []
    for stream in streams:
        for family in ("bm25", "minilm"):
            for row in stream[family]:  # type: ignore[index]
                stream_rows.append(
                    {
                        "topic_id": stream["topic_id"],
                        "facet_id": stream["facet_id"],
                        "manifest_order": stream["manifest_order"],
                        "accepted": stream["accepted"],
                        "stream_family": family,
                        "family": family,
                        **dict(row),
                    }
                )
    output.mkdir(parents=True)
    artifacts = {
        "gates.json": _pretty_bytes({"schema_version": GATE_SCHEMA_VERSION, "gates": gate_rows}),
        "streams.jsonl": _jsonl_bytes(stream_rows),
        "u_raw.jsonl": _jsonl_bytes(raw_rows),
        "u_accepted.jsonl": _jsonl_bytes(accepted_rows),
        "prefix_unions.json": _pretty_bytes(prefixes),
    }
    for name, content in artifacts.items():
        _exclusive_bytes(output / name, content)
    summary: dict[str, object] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": "complete",
        "qrels_opened": False,
        "facet_count": len(streams),
        "accepted_facet_count": sum(stream["accepted"] is True for stream in streams),
        "rejected_facet_count": sum(stream["accepted"] is False for stream in streams),
        "topic_counts": {
            topic_id: {
                "accepted_facets": sum(
                    stream["accepted"] is True and stream["topic_id"] == topic_id
                    for stream in streams
                ),
                "raw_union": len(unions[topic_id]["raw_docids"]),
                "accepted_union": len(unions[topic_id]["accepted_docids"]),
            }
            for topic_id in TOPIC_IDS
        },
        "artifacts": {
            name: {"bytes": len(content), "sha256": _sha256(content)}
            for name, content in artifacts.items()
        },
    }
    _exclusive_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--manifest", required=True, type=Path)
    freeze.add_argument("--retrieval", required=True, type=Path)
    freeze.add_argument("--phase1", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest, cache_root=SOURCE_CACHE_ROOT)
    result = freeze_gate(
        manifest, args.retrieval, args.phase1, args.output
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
