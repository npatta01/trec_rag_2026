"""Freeze qrels-blind static and prefix-reinitialized all-topic DUAL rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np

from .all_topic_facet_contract import ALL_TOPIC_IDS, EXPERIMENT_ID, verify_planning
from .all_topic_facet_retrieve import APPROVED_ORIGINAL_CACHE_ROOT, verify_retrieval
from .all_topic_tethered_score import verify_scores
from .deep_facet_candidate_manifest import analyze_terms
from .deep_facet_candidate_rank import (
    _features,
    _greedy,
    _jaccard,
    ranking_parameters,
    redundancy_penalty,
)


ARMS = (
    "RRF",
    "DUAL",
    "RRF100-STATIC-DUAL",
    "RRF500-STATIC-DUAL",
    "RRF100-REINIT-DUAL",
    "RRF500-REINIT-DUAL",
)
PREFIX_DEPTHS = (100, 500)
SCHEMA_VERSION = "all-topic-tethered-ranking-freeze-v1"
SEAL_SCHEMA_VERSION = "all-topic-tethered-ranking-seal-v1"
PLANNING_ROOT_SHA256 = "bc1351cca8aa05dd0979a342c8dd5f72395f2b7a9207ae20668aba05f1ab85a2"
RETRIEVAL_ROOT_SHA256 = "f2191c295f600d0243f4dcd2b1dc23a05b9fa8a7d9a4d6258983a0d3c9433aeb"
SCORE_PLAN_ROOT_SHA256 = "48c8b9be21adece21c1f17cdeec4de8694a83f5ee7fccaa9fc07ff8ca36944c3"
SCORING_ROOT_SHA256 = "4f67107770b1cd35598adc75beeeafe205ba984f201cedf4b505c4d20515e990"


def _compact_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_compact_bytes(row) + b"\n" for row in rows)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(content), "sha256": _sha256(content)}


def _exclusive_write(path: Path, content: bytes) -> None:
    with path.open("xb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())


def _read_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        source = path.read_bytes()
        value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, source


def _iter_jsonl(path: Path, label: str):
    try:
        source = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    with source:
        for number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label}:{number} is invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label}:{number} must be an object")
            yield row


def _reject_qrels(value: object, path: str = "input") -> None:
    """Reject any recursively supplied qrels field before ranking work begins."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            label = str(key)
            if "qrel" in label.casefold():
                raise ValueError(f"qrels input is forbidden: {path}.{label}")
            _reject_qrels(child, f"{path}.{label}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_qrels(child, f"{path}[{index}]")


def _complete_order(value: object, expected: set[str], label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    result = [str(document_id) for document_id in value]
    if len(result) != len(expected) or len(set(result)) != len(result) or set(result) != expected:
        raise ValueError(f"{label} must be one complete accepted-union permutation")
    return result


def _dual_features(topic_input: Mapping[str, object]) -> dict[str, object]:
    """Use the established feature math without its superseded four-topic gate."""

    adapted = dict(topic_input)
    adapted["topic_id"] = "219"
    features = _features(adapted)
    features["topic_id"] = str(topic_input.get("topic_id"))
    return features


def _splice_prefix(prefix_source: Sequence[str], tail: Sequence[str], depth: int) -> list[str]:
    prefix = list(prefix_source[:depth])
    seen = set(prefix)
    return prefix + [document_id for document_id in tail if document_id not in seen]


def _reinitialized_from_features(
    features: Mapping[str, object], prefix: Sequence[str]
) -> tuple[list[str], dict[str, object]]:
    """Replay a protected prefix, then continue the unchanged DUAL objective."""

    docids: list[str] = features["docids"]  # type: ignore[assignment]
    facets: list[dict[str, object]] = features["facets"]  # type: ignore[assignment]
    f_maps: list[dict[str, float]] = features["F"]  # type: ignore[assignment]
    g: dict[str, float] = features["G"]  # type: ignore[assignment]
    n_score: dict[str, float] = features["N"]  # type: ignore[assignment]
    r: dict[str, float] = features["R"]  # type: ignore[assignment]
    original_rank: dict[str, int] = features["original_rank"]  # type: ignore[assignment]
    best_facet_rank: dict[str, int] = features["best_facet_rank"]  # type: ignore[assignment]
    texts: dict[str, str] = features["texts"]  # type: ignore[assignment]
    index_by_id = {document_id: index for index, document_id in enumerate(docids)}
    if len(prefix) != len(set(prefix)) or not set(prefix) <= set(docids):
        raise ValueError("reinitialized prefix is invalid")

    count = len(docids)
    matrix = np.zeros((count, len(facets)), dtype=np.float64)
    for column, values in enumerate(f_maps):
        matrix[:, column] = [values.get(document_id, 0.0) for document_id in docids]
    local = matrix.max(axis=1) if facets else np.zeros(count)
    coverage = np.zeros(len(facets), dtype=np.float64)
    redundancy = np.zeros(count, dtype=np.float64)
    remaining = np.ones(count, dtype=bool)
    token_sets = [frozenset(analyze_terms(texts[document_id])) for document_id in docids]
    selected = list(prefix)
    for document_id in prefix:
        chosen = index_by_id[document_id]
        remaining[chosen] = False
        if facets:
            coverage = np.maximum(coverage, matrix[chosen])
        selected_tokens = token_sets[chosen]
        for index in np.flatnonzero(remaining):
            penalty = redundancy_penalty(_jaccard(token_sets[index], selected_tokens))
            if penalty > redundancy[index]:
                redundancy[index] = penalty

    seed_coverage = {
        str(facet["facet_id"]): float(coverage[index]) for index, facet in enumerate(facets)
    }
    document_audit: dict[str, dict[str, object]] = {}
    for _position in range(len(prefix), count):
        if facets:
            gains = matrix * (1.0 - coverage)
            bonus = gains.max(axis=1)
            bonus_facet = gains.argmax(axis=1)
        else:
            bonus = np.zeros(count)
            bonus_facet = np.zeros(count, dtype=int)
        objective = np.asarray(
            [0.35 * g[d] + 0.15 * n_score[d] + 0.15 * r[d] for d in docids]
        ) + 0.25 * local + 0.10 * bonus - 0.15 * redundancy
        objective[~remaining] = -np.inf
        maximum = float(objective.max())
        tied = np.flatnonzero(objective == maximum)
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
        document_audit[document_id] = {
            "objective": maximum,
            "coverage_bonus": float(bonus[chosen]),
            "coverage_facet": str(facets[int(bonus_facet[chosen])]["facet_id"]) if facets else None,
            "redundancy_penalty": float(redundancy[chosen]),
        }
        remaining[chosen] = False
        if facets:
            coverage = np.maximum(coverage, matrix[chosen])
        selected_tokens = token_sets[chosen]
        for index in np.flatnonzero(remaining):
            penalty = redundancy_penalty(_jaccard(token_sets[index], selected_tokens))
            if penalty > redundancy[index]:
                redundancy[index] = penalty
    return selected, {
        "seed_document_count": len(prefix),
        "seed_coverage": seed_coverage,
        "seed_redundancy_replayed": True,
        "documents": document_audit,
    }


def reinitialized_dual(
    topic_input: Mapping[str, object], protected_prefix: Sequence[str]
) -> tuple[list[str], dict[str, object]]:
    """Public deterministic DUAL continuation from replayed prefix state."""

    _reject_qrels(topic_input)
    return _reinitialized_from_features(_dual_features(topic_input), protected_prefix)


def build_rankings(
    topic_input: Mapping[str, object], controls: Mapping[str, object]
) -> tuple[dict[str, list[str]], dict[str, object]]:
    """Build all six preregistered complete rankings for one accepted union."""

    _reject_qrels(topic_input)
    _reject_qrels(controls)
    features = _dual_features(topic_input)
    docids: list[str] = features["docids"]  # type: ignore[assignment]
    rrf = _complete_order(controls.get("RRF"), set(docids), "RRF control")
    dual, dual_audit = _greedy(features, "DUAL")
    reinit100, audit100 = _reinitialized_from_features(features, rrf[:100])
    reinit500, audit500 = _reinitialized_from_features(features, rrf[:500])
    rankings = {
        "RRF": rrf,
        "DUAL": dual,
        "RRF100-STATIC-DUAL": _splice_prefix(rrf, dual, 100),
        "RRF500-STATIC-DUAL": _splice_prefix(rrf, dual, 500),
        "RRF100-REINIT-DUAL": reinit100,
        "RRF500-REINIT-DUAL": reinit500,
    }
    expected = set(docids)
    if any(len(order) != len(expected) or set(order) != expected for order in rankings.values()):
        raise ValueError("ranking is not a complete accepted-union permutation")
    prefix100, prefix500 = set(rrf[:100]), set(rrf[:500])
    audit: dict[str, object] = {
        "DUAL": {"seed_document_count": 0, "seed_coverage": {}, "documents": dual_audit},
        "RRF100-STATIC-DUAL": {
            "seed_document_count": 0,
            "protected_document_count": min(100, len(rrf)),
            "objective_state": "empty",
            "documents": {document_id: dual_audit[document_id] for document_id in dual if document_id not in prefix100},
        },
        "RRF500-STATIC-DUAL": {
            "seed_document_count": 0,
            "protected_document_count": min(500, len(rrf)),
            "objective_state": "empty",
            "documents": {document_id: dual_audit[document_id] for document_id in dual if document_id not in prefix500},
        },
        "RRF100-REINIT-DUAL": audit100,
        "RRF500-REINIT-DUAL": audit500,
    }
    return rankings, audit


def _rrf_control(topic_input: Mapping[str, object]) -> list[str]:
    features = _dual_features(topic_input)
    docids: list[str] = features["docids"]  # type: ignore[assignment]
    raw_rrf: dict[str, float] = features["raw_RRF"]  # type: ignore[assignment]
    best_stream_rank: dict[str, int] = features["best_stream_rank"]  # type: ignore[assignment]
    return sorted(docids, key=lambda document_id: (-raw_rrf[document_id], best_stream_rank[document_id], document_id))


def _load_topic_inputs(planning: Path, retrieval: Path, scoring: Path) -> dict[str, dict[str, object]]:
    manifest, _ = _read_json(planning / "manifest.json", "planning manifest")
    facets_raw = manifest.get("facets")
    if not isinstance(facets_raw, list):
        raise ValueError("planning facets are missing")
    facet_meta: dict[str, list[tuple[int, str]]] = {topic: [] for topic in ALL_TOPIC_IDS}
    for row in facets_raw:
        if not isinstance(row, Mapping):
            raise ValueError("planning facet is invalid")
        topic_id, facet_id = str(row.get("topic_id")), str(row.get("facet_id"))
        if topic_id not in facet_meta or not facet_id:
            raise ValueError("planning facet identity is invalid")
        facet_meta[topic_id].append((int(row["manifest_order"]), facet_id))
    for values in facet_meta.values():
        values.sort()

    documents: dict[str, dict[str, str]] = {topic: {} for topic in ALL_TOPIC_IDS}
    original_rank: dict[str, dict[str, int]] = {topic: {} for topic in ALL_TOPIC_IDS}
    facet_rank: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for row in _iter_jsonl(retrieval / "accepted_union.jsonl", "accepted union"):
        topic_id, document_id = str(row.get("topic_id")), str(row.get("document_id"))
        if topic_id not in documents or not document_id or document_id in documents[topic_id]:
            raise ValueError("accepted-union identity is invalid or duplicated")
        documents[topic_id][document_id] = str(row.get("text"))
        provenance = row.get("stream_provenance")
        if not isinstance(provenance, list):
            raise ValueError("accepted-union provenance is invalid")
        for source in provenance:
            if not isinstance(source, Mapping):
                raise ValueError("accepted-union stream provenance is invalid")
            stream_id, rank = str(source.get("stream_id")), int(source["stream_rank"])
            if stream_id == "original":
                original_rank[topic_id][document_id] = rank
            else:
                facet_rank[(topic_id, stream_id)][document_id] = rank

    scores: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in _iter_jsonl(scoring / "features.jsonl", "tethered percentile features"):
        topic_id, query_id, document_id = (
            str(row.get("topic_id")), str(row.get("query_id")), str(row.get("document_id"))
        )
        key = (topic_id, query_id)
        if topic_id not in documents or document_id not in documents[topic_id] or document_id in scores[key]:
            raise ValueError("tethered feature identity is invalid or duplicated")
        percentile = row.get("percentile")
        if isinstance(percentile, bool) or not isinstance(percentile, (int, float)) or not math.isfinite(float(percentile)):
            raise ValueError("tethered percentile is invalid")
        scores[key][document_id] = float(percentile)

    result: dict[str, dict[str, object]] = {}
    for topic_id in ALL_TOPIC_IDS:
        docset = set(documents[topic_id])
        facets = []
        for order, facet_id in facet_meta[topic_id]:
            if set(scores[(topic_id, facet_id)]) != set(facet_rank[(topic_id, facet_id)]):
                raise ValueError(f"facet feature/retrieval population differs: {topic_id}/{facet_id}")
            facets.append({
                "facet_id": facet_id,
                "manifest_order": order,
                "scores": scores[(topic_id, facet_id)],
                "bm25_rank": facet_rank[(topic_id, facet_id)],
            })
        if set(scores[(topic_id, "g")]) != docset or set(scores[(topic_id, "n")]) != docset:
            raise ValueError(f"global feature coverage differs for topic {topic_id}")
        result[topic_id] = {
            "topic_id": topic_id,
            "docids": list(documents[topic_id]),
            "texts": documents[topic_id],
            "original_rank": original_rank[topic_id],
            "facets": facets,
            "common_scores": scores[(topic_id, "g")],
            "narrative_scores": scores[(topic_id, "n")],
        }
    extra = set(scores) - {
        (topic, query)
        for topic in ALL_TOPIC_IDS
        for query in ("g", "n", *(facet for _, facet in facet_meta[topic]))
    }
    if extra:
        raise ValueError("unexpected tethered feature query identities")
    return result


def _parameters(topic_ids: Sequence[str]) -> dict[str, object]:
    inherited = ranking_parameters()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "topic_ids": list(topic_ids),
        "arms": list(ARMS),
        "prefix_depths": list(PREFIX_DEPTHS),
        "rrf_k": inherited["rrf_k"],
        "rrf_family_weights": inherited["rrf_family_weights"],
        "dual": inherited["dual"],
        "redundancy_jaccard_threshold": inherited["redundancy_jaccard_threshold"],
        "facet_score_source": "narrative_tethered_percentile",
        "static_state": "empty",
        "reinitialized_state": "protected RRF prefix replay",
        "tie_breaking": "established DUAL tie break ending in document_id",
        "qrels_opened": False,
        "network_calls": 0,
        "model_loads": 0,
        "inference_calls": 0,
    }


def _freeze_loaded_topics(
    topic_inputs: Mapping[str, Mapping[str, object]],
    controls: Mapping[str, Mapping[str, object]],
    output: Path,
    *,
    input_bindings: Mapping[str, object],
    expected_topic_ids: Sequence[str],
) -> dict[str, object]:
    _reject_qrels(input_bindings)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"create-only ranking output already exists: {output}")
    if set(topic_inputs) != set(expected_topic_ids) or set(controls) != set(expected_topic_ids):
        raise ValueError("topic input/control population differs")
    ranking_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    topic_summary: dict[str, object] = {}
    for topic_id in expected_topic_ids:
        rankings, audit = build_rankings(topic_inputs[topic_id], controls[topic_id])
        hashes: dict[str, str] = {}
        for arm in ARMS:
            order = rankings[arm]
            hashes[arm] = _sha256(_compact_bytes(order))
            ranking_rows.extend(
                {
                    "schema_version": SCHEMA_VERSION,
                    "topic_id": topic_id,
                    "arm": arm,
                    "rank": rank,
                    "document_id": document_id,
                }
                for rank, document_id in enumerate(order, 1)
            )
            arm_audit = audit.get(arm)
            if isinstance(arm_audit, Mapping):
                audit_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "topic_id": topic_id,
                    "arm": arm,
                    "kind": "state",
                    **{key: value for key, value in arm_audit.items() if key != "documents"},
                })
                documents_audit = arm_audit.get("documents")
                if isinstance(documents_audit, Mapping):
                    for document_id, components in documents_audit.items():
                        audit_rows.append({
                            "schema_version": SCHEMA_VERSION,
                            "topic_id": topic_id,
                            "arm": arm,
                            "kind": "selection",
                            "document_id": document_id,
                            **(dict(components) if isinstance(components, Mapping) else {}),
                        })
        topic_summary[topic_id] = {
            "accepted_union_count": len(rankings["RRF"]),
            "ranking_sha256": hashes,
            "complete_permutations": {arm: True for arm in ARMS},
            "protected_prefixes_exact": {str(depth): True for depth in PREFIX_DEPTHS},
        }

    parameters = _pretty_bytes(_parameters(expected_topic_ids))
    bindings = _pretty_bytes({"schema_version": SCHEMA_VERSION, "qrels_opened": False, **input_bindings})
    rankings_bytes = _jsonl_bytes(ranking_rows)
    audit_bytes = _jsonl_bytes(audit_rows)
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "rankings_frozen_before_qrels",
        "qrels_opened": False,
        "topic_count": len(expected_topic_ids),
        "arm_count": len(ARMS),
        "ranking_row_count": len(ranking_rows),
        "audit_row_count": len(audit_rows),
        "network_calls": 0,
        "model_loads": 0,
        "inference_calls": 0,
        "topic_summary": topic_summary,
    }
    artifacts = {
        "parameters.json": parameters,
        "input_bindings.json": bindings,
        "rankings.jsonl": rankings_bytes,
        "audit.jsonl": audit_bytes,
        "summary.json": _pretty_bytes(summary_payload),
    }
    output.mkdir(parents=True)
    for name, content in artifacts.items():
        _exclusive_write(output / name, content)
    files = {
        name: {"bytes": len(content), "sha256": _sha256(content)}
        for name, content in artifacts.items()
    }
    seal = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "sealed_before_qrels",
        "qrels_opened": False,
        "topic_ids": list(expected_topic_ids),
        "files": files,
        "root_sha256": _sha256(_compact_bytes(files)),
    }
    _exclusive_write(output / "SEALED.json", _pretty_bytes(seal))
    return summary_payload


def freeze_rankings(planning: Path, retrieval: Path, scoring: Path, output: Path) -> dict[str, object]:
    """Verify every upstream producer, load offline features, and create the freeze."""

    planning, retrieval, scoring = map(Path, (planning, retrieval, scoring))
    planning_evidence = verify_planning(
        planning, approved_cache_root=APPROVED_ORIGINAL_CACHE_ROOT
    )
    retrieval_evidence = verify_retrieval(retrieval, planning)
    scoring_evidence = verify_scores(scoring)
    observed = (
        planning_evidence.get("root_sha256"),
        retrieval_evidence.get("root_sha256"),
        scoring_evidence.get("root_sha256"),
    )
    expected = (PLANNING_ROOT_SHA256, RETRIEVAL_ROOT_SHA256, SCORING_ROOT_SHA256)
    if observed != expected:
        raise ValueError(f"upstream root identity differs: {observed!r}")
    score_plan, _ = _read_json(scoring / "SCORE_PLAN_SEALED.json", "score-plan seal")
    if score_plan.get("root_sha256") != SCORE_PLAN_ROOT_SHA256:
        raise ValueError("score-plan root identity differs")
    topic_inputs = _load_topic_inputs(planning, retrieval, scoring)
    controls = {topic: {"RRF": _rrf_control(value)} for topic, value in topic_inputs.items()}
    input_bindings = {
        "planning_root_sha256": PLANNING_ROOT_SHA256,
        "retrieval_root_sha256": RETRIEVAL_ROOT_SHA256,
        "score_plan_root_sha256": SCORE_PLAN_ROOT_SHA256,
        "scoring_root_sha256": SCORING_ROOT_SHA256,
        "planning_seal": _binding(planning / "SEALED.json"),
        "retrieval_seal": _binding(retrieval / "RETRIEVAL_SEALED.json"),
        "score_plan_seal": _binding(scoring / "SCORE_PLAN_SEALED.json"),
        "scoring_seal": _binding(scoring / "SCORING_SEALED.json"),
        "accepted_union": _binding(retrieval / "accepted_union.jsonl"),
        "features": _binding(scoring / "features.jsonl"),
    }
    return _freeze_loaded_topics(
        topic_inputs, controls, output, input_bindings=input_bindings, expected_topic_ids=ALL_TOPIC_IDS
    )


def verify_rankings(path: Path) -> dict[str, object]:
    """Verify file hashes, complete permutations, prefix contracts, and ranking hashes."""

    root = Path(path)
    seal, _ = _read_json(root / "SEALED.json", "ranking seal")
    files = seal.get("files")
    topic_ids = seal.get("topic_ids")
    if (
        seal.get("schema_version") != SEAL_SCHEMA_VERSION
        or seal.get("experiment_id") != EXPERIMENT_ID
        or seal.get("status") != "sealed_before_qrels"
        or seal.get("qrels_opened") is not False
        or not isinstance(topic_ids, list)
        or not isinstance(files, Mapping)
        or set(files) != {"parameters.json", "input_bindings.json", "rankings.jsonl", "audit.jsonl", "summary.json"}
        or seal.get("root_sha256") != _sha256(_compact_bytes(files))
    ):
        raise ValueError("ranking seal contract differs")
    actual_names = {entry.name for entry in root.iterdir() if entry.is_file()}
    if actual_names != set(files) | {"SEALED.json"}:
        raise ValueError("ranking seal has missing or extra files")
    for name, expected in files.items():
        if not isinstance(expected, Mapping) or dict(expected) != {key: value for key, value in _binding(root / name).items() if key != "path"}:
            raise ValueError(f"ranking seal detects mutated artifact: {name}")
    parameters, _ = _read_json(root / "parameters.json", "ranking parameters")
    bindings, _ = _read_json(root / "input_bindings.json", "ranking input bindings")
    summary, _ = _read_json(root / "summary.json", "ranking summary")
    if parameters != _parameters([str(value) for value in topic_ids]):
        raise ValueError("ranking parameters differ")
    if bindings.get("qrels_opened") is not False or summary.get("qrels_opened") is not False:
        raise ValueError("qrels firewall evidence differs")
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    row_count = 0
    for row in _iter_jsonl(root / "rankings.jsonl", "rankings"):
        topic_id, arm = str(row.get("topic_id")), str(row.get("arm"))
        if row.get("schema_version") != SCHEMA_VERSION or topic_id not in topic_ids or arm not in ARMS:
            raise ValueError("ranking row contract differs")
        grouped[(topic_id, arm)].append((int(row["rank"]), str(row["document_id"])))
        row_count += 1
    topic_summary = summary.get("topic_summary")
    if not isinstance(topic_summary, Mapping):
        raise ValueError("ranking topic summary is missing")
    for topic_id in map(str, topic_ids):
        orders: dict[str, list[str]] = {}
        for arm in ARMS:
            pairs = grouped.get((topic_id, arm), [])
            if [rank for rank, _ in pairs] != list(range(1, len(pairs) + 1)):
                raise ValueError("ranking positions are not contiguous")
            orders[arm] = [document_id for _, document_id in pairs]
        expected = set(orders["RRF"])
        if not expected or any(len(order) != len(expected) or set(order) != expected for order in orders.values()):
            raise ValueError("ranking is not a complete permutation")
        if any(
            orders[f"RRF{depth}-{mode}-DUAL"][:depth] != orders["RRF"][:depth]
            for depth in PREFIX_DEPTHS for mode in ("STATIC", "REINIT")
        ):
            raise ValueError("protected prefix differs")
        claimed = topic_summary.get(topic_id)
        hashes = claimed.get("ranking_sha256") if isinstance(claimed, Mapping) else None
        if hashes != {arm: _sha256(_compact_bytes(orders[arm])) for arm in ARMS}:
            raise ValueError("ranking hash differs")
    if row_count != summary.get("ranking_row_count") or len(grouped) != len(topic_ids) * len(ARMS):
        raise ValueError("ranking row coverage differs")
    audit_count = sum(1 for _ in _iter_jsonl(root / "audit.jsonl", "ranking audit"))
    if audit_count != summary.get("audit_row_count"):
        raise ValueError("ranking audit coverage differs")
    return {
        "verified": True,
        "root_sha256": seal["root_sha256"],
        "topic_count": len(topic_ids),
        "arm_count": len(ARMS),
        "ranking_row_count": row_count,
        "qrels_network_model_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--planning", type=Path, required=True)
    freeze.add_argument("--retrieval", type=Path, required=True)
    freeze.add_argument("--scoring", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--rankings", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        payload = freeze_rankings(args.planning, args.retrieval, args.scoring, args.output)
    else:
        payload = verify_rankings(args.rankings)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
