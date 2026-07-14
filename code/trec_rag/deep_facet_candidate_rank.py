"""Build and seal qrels-blind deep-facet candidate rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .deep_facet_candidate_manifest import (
    EXCLUDED_TOPIC_IDS,
    TOPIC_IDS,
    analyze_terms,
    assert_mutation_allowed,
    load_manifest,
)
from .deep_facet_candidate_score import aggregate_top4


SCHEMA_VERSION = "deep-facet-candidate-rank-freeze-v1"
SEAL_SCHEMA_VERSION = "deep-facet-candidate-seal-v1"
ARM_NAMES = ("RRF", "GLOBAL", "FACET", "DUAL", "DUAL-NR")
PREFIX_DEPTHS = (100, 500, 1000)
RRF_K = 60


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    source = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(source), "sha256": _sha256(source)}


def _exclusive_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as sink:
        sink.write(value)
        sink.flush()
        os.fsync(sink.fileno())


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _iter_jsonl(path: Path, label: str):
    try:
        source = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    with source:
        for number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label}:{number} is invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label}:{number} must be an object")
            yield row


def _topic(value: object) -> str:
    topic_id = str(value)
    if topic_id in EXCLUDED_TOPIC_IDS:
        raise ValueError(f"excluded topic {topic_id} is forbidden")
    if topic_id not in TOPIC_IDS:
        raise ValueError(f"unexpected topic {topic_id}")
    return topic_id


def rank_percentile(scores: Mapping[str, float | int]) -> dict[str, float]:
    """Convert finite scores to average-rank percentiles, best equal to one."""

    if not scores:
        return {}
    groups: dict[float, list[str]] = defaultdict(list)
    for document_id, raw in scores.items():
        value = float(raw)
        if not document_id or not math.isfinite(value):
            raise ValueError("percentile inputs require document IDs and finite scores")
        groups[value].append(document_id)
    n = len(scores)
    result: dict[str, float] = {}
    first_rank = 1
    for value in sorted(groups, reverse=True):
        ids = groups[value]
        last_rank = first_rank + len(ids) - 1
        average_rank = (first_rank + last_rank) / 2.0
        percentile = (n - average_rank + 1.0) / n
        for document_id in ids:
            result[document_id] = percentile
        first_rank = last_rank + 1
    return result


def redundancy_penalty(jaccard: float) -> float:
    value = float(jaccard)
    if not 0.0 <= value <= 1.0 or not math.isfinite(value):
        raise ValueError("Jaccard must be finite and within [0,1]")
    if value < 0.80:
        return 0.0
    if value == 1.0:
        return 1.0
    return (value - 0.80) / 0.20


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    if len(smaller) / len(larger) < 0.80:
        return 0.0
    intersection = len(smaller.intersection(larger))
    union = len(smaller) + len(larger) - intersection
    return intersection / union if union else 0.0


def _coerce_score_map(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    result = {str(key): float(score) for key, score in value.items()}
    if not all(key and math.isfinite(score) for key, score in result.items()):
        raise ValueError(f"{label} contains an invalid score")
    return result


def _features(inputs: Mapping[str, object]) -> dict[str, object]:
    topic_id = _topic(inputs.get("topic_id"))
    raw_docids = inputs.get("docids")
    if not isinstance(raw_docids, Sequence) or isinstance(raw_docids, (str, bytes)):
        raise ValueError("docids must be a sequence")
    docids = sorted(str(value) for value in raw_docids)
    if not docids or len(set(docids)) != len(docids) or any(not value for value in docids):
        raise ValueError("docids must be unique non-empty values")
    docset = set(docids)
    texts_value = inputs.get("texts")
    if not isinstance(texts_value, Mapping) or set(map(str, texts_value)) != docset:
        raise ValueError("texts must cover the accepted union exactly")
    texts = {str(key): str(value) for key, value in texts_value.items()}
    original_raw = inputs.get("original_rank")
    if not isinstance(original_raw, Mapping):
        raise ValueError("original_rank must be a mapping")
    original_rank = {str(key): int(value) for key, value in original_raw.items()}
    if not set(original_rank) <= docset or any(value <= 0 for value in original_rank.values()):
        raise ValueError("original ranks are invalid")

    raw_facets = inputs.get("facets")
    if not isinstance(raw_facets, Sequence) or isinstance(raw_facets, (str, bytes)):
        raise ValueError("facets must be a sequence")
    facets: list[dict[str, object]] = []
    for raw in raw_facets:
        if not isinstance(raw, Mapping):
            raise ValueError("facet must be an object")
        scores = _coerce_score_map(raw.get("scores"), "facet scores")
        bm25 = raw.get("bm25_rank")
        if not isinstance(bm25, Mapping):
            raise ValueError("facet bm25_rank must be a mapping")
        bm25_rank = {str(key): int(value) for key, value in bm25.items()}
        if set(scores) != set(bm25_rank) or not set(scores) <= docset:
            raise ValueError("facet score/rank populations differ")
        facets.append(
            {
                "facet_id": str(raw.get("facet_id")),
                "manifest_order": int(raw.get("manifest_order")),
                "scores": scores,
                "bm25_rank": bm25_rank,
            }
        )
    facets.sort(key=lambda row: (int(row["manifest_order"]), str(row["facet_id"])))
    if len({row["facet_id"] for row in facets}) != len(facets):
        raise ValueError("facet identities must be unique")

    common_scores = _coerce_score_map(inputs.get("common_scores"), "common scores")
    narrative_scores = _coerce_score_map(inputs.get("narrative_scores"), "narrative scores")
    if set(common_scores) != docset or set(narrative_scores) != docset:
        raise ValueError("common and narrative scores must cover U_accepted")

    f_maps = [rank_percentile(row["scores"]) for row in facets]
    g = rank_percentile(common_scores)
    n_score = rank_percentile(narrative_scores)
    facet_weight = 0.5 / len(facets) if facets else 0.0
    original_weight = 0.5 if facets else 1.0
    raw_rrf: dict[str, float] = {}
    for document_id in docids:
        score = original_weight / (RRF_K + original_rank[document_id]) if document_id in original_rank else 0.0
        score += math.fsum(
            facet_weight / (RRF_K + row["bm25_rank"][document_id])
            for row in facets
            if document_id in row["bm25_rank"]
        )
        raw_rrf[document_id] = score
    r = rank_percentile(raw_rrf)
    best_stream_rank = {
        document_id: min(
            [original_rank.get(document_id, 10**9)]
            + [row["bm25_rank"].get(document_id, 10**9) for row in facets]
        )
        for document_id in docids
    }
    local_rank_maps = [
        {
            document_id: rank
            for rank, document_id in enumerate(
                sorted(
                    row["scores"],
                    key=lambda value: (-row["scores"][value], row["bm25_rank"][value], value),
                ),
                start=1,
            )
        }
        for row in facets
    ]
    best_facet_rank = {
        document_id: min(
            [rank_map.get(document_id, 10**9) for rank_map in local_rank_maps],
            default=10**9,
        )
        for document_id in docids
    }
    return {
        "topic_id": topic_id,
        "docids": docids,
        "texts": texts,
        "original_rank": original_rank,
        "facets": facets,
        "F": f_maps,
        "G": g,
        "N": n_score,
        "raw_RRF": raw_rrf,
        "R": r,
        "best_stream_rank": best_stream_rank,
        "best_facet_rank": best_facet_rank,
    }


def _greedy(
    features: Mapping[str, object], arm: str
) -> tuple[list[str], dict[str, dict[str, object]]]:
    docids: list[str] = features["docids"]  # type: ignore[assignment]
    facets: list[dict[str, object]] = features["facets"]  # type: ignore[assignment]
    f_maps: list[dict[str, float]] = features["F"]  # type: ignore[assignment]
    g: dict[str, float] = features["G"]  # type: ignore[assignment]
    n_score: dict[str, float] = features["N"]  # type: ignore[assignment]
    r: dict[str, float] = features["R"]  # type: ignore[assignment]
    original_rank: dict[str, int] = features["original_rank"]  # type: ignore[assignment]
    best_facet_rank: dict[str, int] = features["best_facet_rank"]  # type: ignore[assignment]
    texts: dict[str, str] = features["texts"]  # type: ignore[assignment]
    count = len(docids)
    matrix = np.zeros((count, len(facets)), dtype=np.float64)
    for column, values in enumerate(f_maps):
        matrix[:, column] = [values.get(document_id, 0.0) for document_id in docids]
    local = matrix.max(axis=1) if len(facets) else np.zeros(count)
    local_facet = matrix.argmax(axis=1) if len(facets) else np.zeros(count, dtype=int)
    coverage = np.zeros(len(facets), dtype=np.float64)
    redundancy = np.zeros(count, dtype=np.float64)
    remaining = np.ones(count, dtype=bool)
    token_sets = [frozenset(analyze_terms(texts[document_id])) for document_id in docids]
    selected: list[str] = []
    audit: dict[str, dict[str, object]] = {}
    for _position in range(count):
        if len(facets):
            gains = matrix * (1.0 - coverage)
            bonus = gains.max(axis=1)
            bonus_facet = gains.argmax(axis=1)
        else:
            bonus = np.zeros(count)
            bonus_facet = np.zeros(count, dtype=int)
        if arm == "FACET":
            objective = 0.70 * local + 0.30 * bonus
        else:
            objective = np.asarray(
                [0.35 * g[d] + 0.15 * n_score[d] + 0.15 * r[d] for d in docids]
            ) + 0.25 * local + 0.10 * bonus
            if arm == "DUAL":
                objective -= 0.15 * redundancy
        objective[~remaining] = -np.inf
        maximum = float(objective.max())
        tied = np.flatnonzero(objective == maximum)
        if arm == "FACET":
            chosen = min(
                tied,
                key=lambda idx: (
                    -r[docids[idx]],
                    int(local_facet[idx]),
                    int(bonus_facet[idx]),
                    original_rank.get(docids[idx], 10**9),
                    best_facet_rank[docids[idx]],
                    docids[idx],
                ),
            )
        else:
            chosen = min(
                tied,
                key=lambda idx: (
                    -r[docids[idx]],
                    -float(local[idx]),
                    original_rank.get(docids[idx], 10**9),
                    best_facet_rank[docids[idx]],
                    docids[idx],
                ),
            )
        document_id = docids[int(chosen)]
        selected.append(document_id)
        audit[document_id] = {
            "objective": maximum,
            "coverage_bonus": float(bonus[chosen]),
            "coverage_facet": (
                str(facets[int(bonus_facet[chosen])]["facet_id"]) if facets else None
            ),
            "redundancy_penalty": float(redundancy[chosen]) if arm == "DUAL" else 0.0,
        }
        remaining[chosen] = False
        if len(facets):
            coverage = np.maximum(coverage, matrix[chosen])
        if arm == "DUAL":
            selected_tokens = token_sets[chosen]
            for index in np.flatnonzero(remaining):
                penalty = redundancy_penalty(_jaccard(token_sets[index], selected_tokens))
                if penalty > redundancy[index]:
                    redundancy[index] = penalty
    return selected, audit


def _build_with_audit(
    inputs: Mapping[str, object]
) -> tuple[dict[str, list[str]], dict[str, dict[str, dict[str, object]]], dict[str, object]]:
    features = _features(inputs)
    docids: list[str] = features["docids"]  # type: ignore[assignment]
    facets: list[dict[str, object]] = features["facets"]  # type: ignore[assignment]
    original_rank: dict[str, int] = features["original_rank"]  # type: ignore[assignment]
    if not facets:
        fallback = sorted(docids, key=lambda d: (original_rank.get(d, 10**9), d))
        return (
            {arm: list(fallback) for arm in ARM_NAMES},
            {arm: {d: {"objective": None, "coverage_bonus": 0.0, "coverage_facet": None, "redundancy_penalty": 0.0} for d in fallback} for arm in ARM_NAMES},
            features,
        )
    raw_rrf: dict[str, float] = features["raw_RRF"]  # type: ignore[assignment]
    best_stream_rank: dict[str, int] = features["best_stream_rank"]  # type: ignore[assignment]
    rrf = sorted(docids, key=lambda d: (-raw_rrf[d], best_stream_rank[d], d))
    g: dict[str, float] = features["G"]  # type: ignore[assignment]
    n_score: dict[str, float] = features["N"]  # type: ignore[assignment]
    r: dict[str, float] = features["R"]  # type: ignore[assignment]
    global_order = sorted(docids, key=lambda d: (-(0.70 * g[d] + 0.30 * n_score[d]), -r[d], d))
    facet, facet_audit = _greedy(features, "FACET")
    dual, dual_audit = _greedy(features, "DUAL")
    dual_nr, dual_nr_audit = _greedy(features, "DUAL-NR")
    empty_audit = {
        d: {"objective": raw_rrf[d], "coverage_bonus": 0.0, "coverage_facet": None, "redundancy_penalty": 0.0}
        for d in docids
    }
    global_audit = {
        d: {"objective": 0.70 * g[d] + 0.30 * n_score[d], "coverage_bonus": 0.0, "coverage_facet": None, "redundancy_penalty": 0.0}
        for d in docids
    }
    return (
        {"RRF": rrf, "GLOBAL": global_order, "FACET": facet, "DUAL": dual, "DUAL-NR": dual_nr},
        {"RRF": empty_audit, "GLOBAL": global_audit, "FACET": facet_audit, "DUAL": dual_audit, "DUAL-NR": dual_nr_audit},
        features,
    )


def build_permutations(inputs: Mapping[str, object]) -> dict[str, list[str]]:
    return _build_with_audit(inputs)[0]


def _aggregate_scores(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    result: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    current: tuple[str, str, str] | None = None
    windows: list[dict[str, object]] = []

    def flush() -> None:
        if current is not None:
            topic_id, family, document_id = current
            result[(topic_id, family)][document_id] = aggregate_top4(windows)

    for row in _iter_jsonl(path, "phase-2 scores"):
        topic_id = _topic(row.get("topic_id"))
        family = str(row.get("family"))
        if family not in {"common", "narrative"}:
            raise ValueError("phase-2 score family is invalid")
        key = (topic_id, family, str(row.get("document_id")))
        if current is not None and key != current:
            flush()
            windows = []
        current = key
        windows.append(row)
    flush()
    return result


def _stream_kind(row: Mapping[str, object]) -> str:
    """Recover the outer BM25/MiniLM identity, including legacy gate_v1 rows."""

    declared = row.get("stream_family")
    if declared in {"bm25", "minilm"}:
        return str(declared)
    # Legacy gate_v1 flattened ``family`` before expanding the candidate row,
    # whose ``family=facet`` overwrote it. Only aggregated MiniLM rows have both
    # an aggregate score and the original retrieval rank.
    if row.get("family") == "facet":
        return "minilm" if "score" in row and "retrieval_rank" in row else "bm25"
    if row.get("family") in {"bm25", "minilm"}:
        return str(row["family"])
    raise ValueError("gate stream family cannot be identified")


def _load_topic_inputs(
    manifest: Mapping[str, object], gate_dir: Path, phase2_dir: Path
) -> dict[str, dict[str, object]]:
    documents: dict[str, dict[str, str]] = {topic: {} for topic in TOPIC_IDS}
    original_rank: dict[str, dict[str, int]] = {topic: {} for topic in TOPIC_IDS}
    facet_bm25: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for row in _iter_jsonl(gate_dir / "u_accepted.jsonl", "accepted union"):
        topic_id = _topic(row.get("topic_id"))
        document_id = str(row.get("document_id"))
        documents[topic_id][document_id] = str(row.get("text"))
        provenance = row.get("provenance")
        if not isinstance(provenance, list):
            raise ValueError("accepted union provenance must be an array")
        for source in provenance:
            if not isinstance(source, Mapping):
                raise ValueError("accepted union provenance row must be an object")
            if source.get("family") == "original":
                original_rank[topic_id][document_id] = int(source["rank"])
            elif source.get("family") == "facet" and source.get("accepted") is True:
                facet_bm25[(topic_id, str(source["facet_id"]))][document_id] = int(source["bm25_rank"])

    local_scores: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    facet_metadata: dict[tuple[str, str], int] = {}
    for row in _iter_jsonl(gate_dir / "streams.jsonl", "gate streams"):
        if _stream_kind(row) != "minilm" or row.get("accepted") is not True:
            continue
        topic_id = _topic(row.get("topic_id"))
        facet_id = str(row.get("facet_id"))
        facet_metadata[(topic_id, facet_id)] = int(row["manifest_order"])
        local_scores[(topic_id, facet_id)][str(row.get("document_id"))] = float(row["score"])
    semantic = _aggregate_scores(phase2_dir / "scores.jsonl")
    result: dict[str, dict[str, object]] = {}
    for topic_id in TOPIC_IDS:
        facets = [
            {
                "facet_id": facet_id,
                "manifest_order": order,
                "scores": local_scores[(topic_id, facet_id)],
                "bm25_rank": facet_bm25[(topic_id, facet_id)],
            }
            for (facet_topic, facet_id), order in facet_metadata.items()
            if facet_topic == topic_id
        ]
        result[topic_id] = {
            "topic_id": topic_id,
            "docids": list(documents[topic_id]),
            "texts": documents[topic_id],
            "original_rank": original_rank[topic_id],
            "facets": facets,
            "common_scores": semantic[(topic_id, "common")],
            "narrative_scores": semantic[(topic_id, "narrative")],
        }
    return result


def _binding(path: Path) -> dict[str, object]:
    return _file_record(Path(path))


def _verify_inputs(gate_dir: Path, phase2_dir: Path) -> None:
    gate = _read_json(gate_dir / "summary.json", "gate summary")
    phase2 = _read_json(phase2_dir / "scoring_receipt.json", "phase-2 receipt")
    artifacts = gate.get("artifacts")
    if gate.get("status") != "complete" or gate.get("qrels_opened") is not False or not isinstance(artifacts, Mapping):
        raise ValueError("gate is incomplete")
    for name in ("streams.jsonl", "u_accepted.jsonl", "u_raw.jsonl", "prefix_unions.json"):
        value = artifacts.get(name)
        source = (gate_dir / name).read_bytes()
        if not isinstance(value, Mapping) or value.get("bytes") != len(source) or value.get("sha256") != _sha256(source):
            raise ValueError(f"gate artifact differs: {name}")
    scores = (phase2_dir / "scores.jsonl").read_bytes()
    if (
        phase2.get("status") != "complete"
        or phase2.get("qrels_opened") is not False
        or phase2.get("scores_sha256") != _sha256(scores)
    ):
        raise ValueError("phase-2 evidence is incomplete or unauthenticated")


def ranking_parameters() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "arms": list(ARM_NAMES),
        "topic_ids": list(TOPIC_IDS),
        "prefix_depths": list(PREFIX_DEPTHS),
        "rrf_k": RRF_K,
        "rrf_family_weights": {"original": 0.5, "accepted_facets_total": 0.5},
        "global": {"G": 0.70, "N": 0.30},
        "facet": {"L": 0.70, "B": 0.30},
        "dual": {"G": 0.35, "N": 0.15, "R": 0.15, "L": 0.25, "B": 0.10, "D": -0.15},
        "dual_nr_redundancy_fixed_zero": True,
        "redundancy_jaccard_threshold": 0.80,
        "percentile": "(n-average_rank+1)/n; absent facet=0",
        "raw_scores_cross_query_boundaries": False,
        "narrative_is_eligibility_gate": False,
        "qrels_opened": False,
    }


def _root_files(root: Path, *, seal_path: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.resolve() != seal_path.resolve()),
        key=lambda path: str(path.relative_to(root)),
    )


def create_seal(
    *, manifest_path: Path, source_dirs: Sequence[Path], freeze_dir: Path,
    topic_ids: Sequence[str],
) -> dict[str, object]:
    freeze_dir = Path(freeze_dir).resolve()
    seal_path = freeze_dir / "SEALED.json"
    if seal_path.exists():
        raise FileExistsError(f"create-only seal already exists: {seal_path}")
    roots = [Path(value).resolve() for value in source_dirs] + [freeze_dir]
    root_rows: list[dict[str, object]] = []
    all_paths: dict[str, Path] = {str(Path(manifest_path).resolve()): Path(manifest_path).resolve()}
    for root in roots:
        files = _root_files(root, seal_path=seal_path)
        root_rows.append({"path": str(root), "files": [str(path.relative_to(root)) for path in files]})
        all_paths.update({str(path.resolve()): path.resolve() for path in files})
    records = [_file_record(all_paths[key]) for key in sorted(all_paths)]
    root_material = {"topic_ids": list(topic_ids), "roots": root_rows, "artifacts": records}
    payload: dict[str, object] = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": "sealed_before_qrels",
        "qrels_opened": False,
        **root_material,
        "root_sha256": _sha256(_canonical_bytes(root_material)),
    }
    _exclusive_bytes(seal_path, _pretty_bytes(payload))
    return payload


def verify_seal(freeze_dir: Path) -> dict[str, object]:
    freeze_dir = Path(freeze_dir).resolve()
    seal_path = freeze_dir / "SEALED.json"
    payload = _read_json(seal_path, "seal")
    if (
        payload.get("schema_version") != SEAL_SCHEMA_VERSION
        or payload.get("status") != "sealed_before_qrels"
        or payload.get("qrels_opened") is not False
        or payload.get("topic_ids") != list(TOPIC_IDS)
    ):
        raise ValueError("seal contract differs")
    roots = payload.get("roots")
    records = payload.get("artifacts")
    if not isinstance(roots, list) or not isinstance(records, list):
        raise ValueError("seal records are missing")
    for root_row in roots:
        if not isinstance(root_row, Mapping) or not isinstance(root_row.get("files"), list):
            raise ValueError("seal root record is invalid")
        root = Path(str(root_row["path"]))
        actual = [str(path.relative_to(root)) for path in _root_files(root, seal_path=seal_path)]
        expected = list(root_row["files"])
        if actual != expected:
            raise ValueError(f"sealed root has missing or extra files: {root}")
    actual_records = [_file_record(Path(str(row["path"]))) for row in records if isinstance(row, Mapping)]
    if actual_records != records:
        raise ValueError("sealed artifact bytes were mutated")
    material = {"topic_ids": payload["topic_ids"], "roots": roots, "artifacts": records}
    if payload.get("root_sha256") != _sha256(_canonical_bytes(material)):
        raise ValueError("seal root hash differs")
    return payload


def freeze_rankings(
    manifest_path: Path, retrieval_dir: Path, phase1_dir: Path, gate_dir: Path,
    phase2_dir: Path, output: Path,
) -> dict[str, object]:
    output = Path(output)
    assert_mutation_allowed(output.parent)
    if output.exists():
        raise FileExistsError(f"create-only freeze output already exists: {output}")
    manifest = load_manifest(
        manifest_path,
        cache_root=Path("/home/npatta01/data/competitions/trec_rag_2026/cache/retrieval/pyserini_remote"),
    )
    _verify_inputs(Path(gate_dir), Path(phase2_dir))
    inputs = _load_topic_inputs(manifest, Path(gate_dir), Path(phase2_dir))
    ranking_rows: list[dict[str, object]] = []
    prefixes: dict[str, dict[str, dict[str, list[str]]]] = {}
    topic_summary: dict[str, dict[str, object]] = {}
    for topic_id in TOPIC_IDS:
        permutations, audits, features = _build_with_audit(inputs[topic_id])
        prefixes[topic_id] = {}
        f_maps: list[dict[str, float]] = features["F"]  # type: ignore[assignment]
        facets: list[dict[str, object]] = features["facets"]  # type: ignore[assignment]
        for arm, ordered in permutations.items():
            if len(ordered) != len(set(ordered)) or set(ordered) != set(inputs[topic_id]["docids"]):
                raise ValueError("ranking is not a complete accepted-union permutation")
            prefixes[topic_id][arm] = {str(depth): ordered[:depth] for depth in PREFIX_DEPTHS}
            for rank, document_id in enumerate(ordered, start=1):
                facet_values = {str(facets[i]["facet_id"]): f_maps[i].get(document_id, 0.0) for i in range(len(facets))}
                ranking_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "topic_id": topic_id,
                        "arm": arm,
                        "rank": rank,
                        "document_id": document_id,
                        "G": features["G"][document_id],  # type: ignore[index]
                        "N": features["N"][document_id],  # type: ignore[index]
                        "R": features["R"][document_id],  # type: ignore[index]
                        "L": max(facet_values.values(), default=0.0),
                        "facet_percentiles": facet_values,
                        "original_rank": features["original_rank"].get(document_id),  # type: ignore[union-attr]
                        "best_facet_rank": features["best_facet_rank"][document_id],  # type: ignore[index]
                        "text_sha256": _sha256(inputs[topic_id]["texts"][document_id].encode("utf-8")),  # type: ignore[index,union-attr]
                        **audits[arm][document_id],
                    }
                )
        topic_summary[topic_id] = {
            "accepted_union_count": len(inputs[topic_id]["docids"]),
            "accepted_facet_count": len(facets),
            "zero_facet_fallback": not facets,
        }
    output.mkdir(parents=True)
    parameters = ranking_parameters()
    inputs_payload = {
        "schema_version": SCHEMA_VERSION,
        "qrels_opened": False,
        "manifest": _binding(Path(manifest_path)),
        "retrieval_summary": _binding(Path(retrieval_dir) / "retrieval_summary.json"),
        "phase1_receipt": _binding(Path(phase1_dir) / "scoring_receipt.json"),
        "gate_summary": _binding(Path(gate_dir) / "summary.json"),
        "phase2_receipt": _binding(Path(phase2_dir) / "scoring_receipt.json"),
    }
    artifacts = {
        "parameters.json": _pretty_bytes(parameters),
        "input_bindings.json": _pretty_bytes(inputs_payload),
        "rankings.jsonl": _jsonl_bytes(ranking_rows),
        "prefixes.json": _pretty_bytes(prefixes),
    }
    for name, content in artifacts.items():
        _exclusive_bytes(output / name, content)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "rankings_frozen_before_qrels",
        "qrels_opened": False,
        "topic_ids": list(TOPIC_IDS),
        "arms": list(ARM_NAMES),
        "ranking_row_count": len(ranking_rows),
        "topic_summary": topic_summary,
        "artifacts": {name: {"bytes": len(content), "sha256": _sha256(content)} for name, content in artifacts.items()},
    }
    _exclusive_bytes(output / "summary.json", _pretty_bytes(summary))
    create_seal(
        manifest_path=Path(manifest_path),
        source_dirs=[Path(retrieval_dir), Path(phase1_dir), Path(gate_dir), Path(phase2_dir)],
        freeze_dir=output,
        topic_ids=TOPIC_IDS,
    )
    verify_seal(output)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--manifest", required=True, type=Path)
    freeze.add_argument("--retrieval", required=True, type=Path)
    freeze.add_argument("--phase1", required=True, type=Path)
    freeze.add_argument("--gate", required=True, type=Path)
    freeze.add_argument("--phase2", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--freeze", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify_seal(args.freeze)
    else:
        result = freeze_rankings(args.manifest, args.retrieval, args.phase1, args.gate, args.phase2, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
