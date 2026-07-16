"""Offline complete-union rankings with narrative-tethered facet scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

from .adaptive_evidence_contract import PILOT_TOPIC_IDS, PROTECTED_TOPIC_IDS
from .adaptive_evidence_rank import verify_baseline_rankings
from .deep_facet_candidate_rank import (
    _build_with_audit,
    _features,
    _load_topic_inputs,
    _verify_inputs,
    ranking_parameters,
    verify_seal as verify_prior_seal,
)
from .tethered_facet_minilm_score import verify_scoring


ARMS = (
    "RRF",
    "NARRATIVE",
    "FIXED-O0",
    "TETHERED-DUAL",
    "TETHERED-DUAL-NR",
    "RRF100-TETHERED-DUAL",
)
HEAD_SIZE = 100
SCHEMA_VERSION = "tethered-facet-soft-coverage-freeze-v1"
SEAL_SCHEMA_VERSION = "tethered-facet-soft-coverage-seal-v1"
_COUNTERS = (
    "network_call_count",
    "retrieval_call_count",
    "model_load_count",
    "inference_count",
    "hosted_inference_call_count",
    "paid_call_count",
)
_DEEP_ROOT = Path("outputs/rag25_deep_facet_candidates_v1")
_TETHERED = _DEEP_ROOT / "post_qrels_tethered_facet_minilm_v1" / "scoring"
_BASELINES = _DEEP_ROOT / "adaptive_evidence_ranker_v1" / "rankings"


def protect_head(
    head: Sequence[str], tail: Sequence[str], size: int = HEAD_SIZE
) -> list[str]:
    """Keep a prefix and append every unseen tail document in order."""

    protected = list(head[:size])
    seen = set(protected)
    return protected + [document_id for document_id in tail if document_id not in seen]


def _complete_order(
    value: object, *, arm: str, expected: set[str]
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{arm} control must be a sequence")
    ordered = [str(document_id) for document_id in value]
    if len(ordered) != len(expected) or len(set(ordered)) != len(ordered):
        raise ValueError(f"{arm} control is not a complete permutation")
    if set(ordered) != expected:
        raise ValueError(f"{arm} control population differs from accepted union")
    return ordered


def build_soft_permutations(
    topic_input: Mapping[str, object], controls: Mapping[str, object]
) -> tuple[dict[str, list[str]], dict[str, object]]:
    """Reuse DUAL unchanged after callers substitute tethered facet score maps."""

    raw_docids = topic_input.get("docids")
    if not isinstance(raw_docids, Sequence) or isinstance(raw_docids, (str, bytes)):
        raise ValueError("docids must be a sequence")
    expected = {str(document_id) for document_id in raw_docids}
    if len(expected) != len(raw_docids):
        raise ValueError("docids must be unique")
    control_orders = {
        arm: _complete_order(controls.get(arm), arm=arm, expected=expected)
        for arm in ("RRF", "NARRATIVE", "FIXED-O0")
    }
    established, established_audit, _features = _build_with_audit(topic_input)
    rankings = {
        **control_orders,
        "TETHERED-DUAL": established["DUAL"],
        "TETHERED-DUAL-NR": established["DUAL-NR"],
    }
    rankings["RRF100-TETHERED-DUAL"] = protect_head(
        rankings["RRF"], rankings["TETHERED-DUAL"]
    )
    parameters = ranking_parameters()
    audit: dict[str, object] = {
        "parameters": {
            "dual": parameters["dual"],
            "dual_nr_redundancy_fixed_zero": True,
            "facet_score_source": "narrative_tethered",
        },
        "TETHERED-DUAL": established_audit["DUAL"],
        "TETHERED-DUAL-NR": established_audit["DUAL-NR"],
        "RRF100-TETHERED-DUAL": established_audit["DUAL"],
    }
    return rankings, audit


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _exclusive_write(path: Path, content: bytes) -> None:
    with path.open("xb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        source = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"JSONL source is unreadable: {path}") from exc
    with source:
        for number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{number}")
            rows.append(row)
    return rows


def _reject_topics(topic_ids: Sequence[object]) -> list[str]:
    normalized = [str(topic_id) for topic_id in topic_ids]
    for topic_id in normalized:
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
    unexpected = set(normalized) - set(PILOT_TOPIC_IDS)
    if unexpected:
        raise ValueError(f"unexpected topic {min(unexpected)}")
    return normalized


def load_topic_rows(
    topic_ids: Sequence[object],
    paths: Sequence[Path] = (),
    *,
    reader: Callable[[Path], Sequence[Mapping[str, object]]] = _read_jsonl,
) -> list[dict[str, object]]:
    """Reject protected scope before invoking any supplied source reader."""

    allowed = set(_reject_topics(topic_ids))
    rows: list[dict[str, object]] = []
    for path in paths:
        for raw in reader(Path(path)):
            row = dict(raw)
            topic_id = str(row.get("topic_id"))
            _reject_topics([topic_id])
            if topic_id not in allowed:
                raise ValueError(f"row topic {topic_id} is outside requested scope")
            rows.append(row)
    return rows


def _replace_facet_scores(
    topic_inputs: Mapping[str, Mapping[str, object]],
    tethered_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Copy DUAL inputs while replacing only each facet's local score map."""

    scores: dict[tuple[str, str], dict[str, float]] = {}
    for row in tethered_rows:
        topic_id = str(row.get("topic_id"))
        _reject_topics([topic_id])
        facet_id, document_id = str(row.get("facet_id")), str(row.get("document_id"))
        key = (topic_id, facet_id)
        if not facet_id or not document_id or document_id in scores.setdefault(key, {}):
            raise ValueError("tethered facet score identity is invalid or duplicated")
        scores[key][document_id] = float(row["score"])
    output: dict[str, dict[str, object]] = {}
    used: set[tuple[str, str]] = set()
    for topic_id, raw_topic in topic_inputs.items():
        _reject_topics([topic_id, raw_topic.get("topic_id")])
        topic = dict(raw_topic)
        raw_facets = raw_topic.get("facets")
        if not isinstance(raw_facets, Sequence) or isinstance(raw_facets, (str, bytes)):
            raise ValueError("DUAL facets must be a sequence")
        facets: list[dict[str, object]] = []
        for raw_facet in raw_facets:
            if not isinstance(raw_facet, Mapping):
                raise ValueError("DUAL facet must be an object")
            facet = dict(raw_facet)
            key = (topic_id, str(facet.get("facet_id")))
            replacement = scores.get(key)
            original = facet.get("scores")
            if not isinstance(original, Mapping) or replacement is None:
                raise ValueError(f"missing tethered scores for {topic_id}/{key[1]}")
            if set(map(str, original)) != set(replacement):
                raise ValueError(f"tethered score population differs for {topic_id}/{key[1]}")
            facet["scores"] = dict(replacement)
            facets.append(facet)
            used.add(key)
        topic["facets"] = facets
        output[topic_id] = topic
    if used != set(scores):
        raise ValueError("tethered score rows contain unexpected topic/facet identities")
    return output


def _orders_from_rows(
    rows: Sequence[Mapping[str, object]],
    arms: Sequence[str],
    *,
    rank_field: str,
    source_arm: str | None = None,
) -> dict[str, dict[str, list[str]]]:
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = {}
    allowed_arms = set(arms)
    for row in rows:
        topic_id = str(row.get("topic_id"))
        arm = source_arm if source_arm is not None else str(row.get("arm"))
        _reject_topics([topic_id])
        if arm not in allowed_arms:
            continue
        grouped.setdefault((topic_id, arm), []).append(
            (int(row[rank_field]), str(row["document_id"]))
        )
    result = {topic_id: {} for topic_id in PILOT_TOPIC_IDS}
    for topic_id in PILOT_TOPIC_IDS:
        for arm in arms:
            pairs = sorted(grouped.get((topic_id, arm), []))
            if [rank for rank, _document_id in pairs] != list(range(1, len(pairs) + 1)):
                raise ValueError(f"{topic_id}/{arm} ranks are not contiguous")
            ordered = [document_id for _rank, document_id in pairs]
            if not ordered or len(ordered) != len(set(ordered)):
                raise ValueError(f"{topic_id}/{arm} is empty or duplicated")
            result[topic_id][arm] = ordered
    return result


def load_authenticated_inputs(
    deep_root: Path = _DEEP_ROOT,
    tethered: Path = _TETHERED,
    baselines: Path = _BASELINES,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, Path],
]:
    """Authenticate historical artifacts before reading the four-topic rows."""

    deep_root, tethered, baselines = Path(deep_root), Path(tethered), Path(baselines)
    gate, phase2, prior = deep_root / "gate_v1", deep_root / "phase2_v1", deep_root / "freeze_v1"
    # Repository verifiers bind the accepted union, phase-2 scores, historical
    # rank features, tethered score lineage, and both comparison controls.
    verify_prior_seal(prior)
    _verify_inputs(gate, phase2)
    verify_scoring(tethered / "preflight.json")
    verify_baseline_rankings(baselines)
    base_inputs = _load_topic_inputs({}, gate, phase2)
    source_paths = {
        "prior_seal": prior / "SEALED.json",
        "prior_rankings": prior / "rankings.jsonl",
        "accepted_union": gate / "u_accepted.jsonl",
        "gate_summary": gate / "summary.json",
        "phase2_scores": phase2 / "scores.jsonl",
        "phase2_receipt": phase2 / "scoring_receipt.json",
        "tethered_document_scores": tethered / "document_scores.jsonl",
        "tethered_scoring_receipt": tethered / "scoring_receipt.json",
        "baseline_receipt": baselines / "receipt.json",
        "narrative_rankings": baselines / "narrative.jsonl",
        "fixed_o0_rankings": baselines / "fixed_o0.jsonl",
    }
    if any(not path.is_file() or path.is_symlink() for path in source_paths.values()):
        raise ValueError("an authenticated source is missing or unsafe")
    tethered_rows = load_topic_rows(
        PILOT_TOPIC_IDS, [source_paths["tethered_document_scores"]]
    )
    topic_inputs = _replace_facet_scores(base_inputs, tethered_rows)
    prior_rows = load_topic_rows(PILOT_TOPIC_IDS, [source_paths["prior_rankings"]])
    rrf = _orders_from_rows(prior_rows, ["RRF"], rank_field="rank")
    baseline_orders: dict[str, dict[str, list[str]]] = {
        topic_id: {} for topic_id in PILOT_TOPIC_IDS
    }
    for arm, source_name in (
        ("NARRATIVE", "narrative_rankings"),
        ("FIXED-O0", "fixed_o0_rankings"),
    ):
        rows = load_topic_rows(PILOT_TOPIC_IDS, [source_paths[source_name]])
        arm_orders = _orders_from_rows(
            rows, [arm], rank_field="topic_rank", source_arm=arm
        )
        for topic_id in PILOT_TOPIC_IDS:
            baseline_orders[topic_id][arm] = arm_orders[topic_id][arm]
    controls: dict[str, dict[str, object]] = {}
    for topic_id in PILOT_TOPIC_IDS:
        controls[topic_id] = {
            "RRF": rrf[topic_id]["RRF"],
            "NARRATIVE": baseline_orders[topic_id]["NARRATIVE"],
            "FIXED-O0": baseline_orders[topic_id]["FIXED-O0"],
        }
    return topic_inputs, controls, source_paths


def _parameters() -> dict[str, object]:
    counters = {name: 0 for name in _COUNTERS}
    return {
        "schema_version": SCHEMA_VERSION,
        "arms": list(ARMS),
        "topic_ids": list(PILOT_TOPIC_IDS),
        "head_size": HEAD_SIZE,
        "dual": ranking_parameters()["dual"],
        "dual_nr_redundancy_fixed_zero": True,
        "facet_score_source": "narrative_tethered",
        "raw_scores_cross_query_boundaries": False,
        "qrels_opened": False,
        "protected_topic_count": 0,
        **counters,
        "external_cost_usd": 0.0,
    }


def _relative_binding(path: Path) -> dict[str, object]:
    path = Path(path)
    content = path.read_bytes()
    try:
        relative = str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        relative = path.name
    return {
        "path": relative,
        "bytes": len(content),
        "rows": len(content.splitlines()),
        "sha256": _sha256(content),
    }


def _ranking_rows(
    topic_id: str,
    topic_input: Mapping[str, object],
    rankings: Mapping[str, Sequence[str]],
    audit: Mapping[str, object],
) -> list[dict[str, object]]:
    features = _features(topic_input)
    facets: list[dict[str, object]] = features["facets"]  # type: ignore[assignment]
    f_maps: list[dict[str, float]] = features["F"]  # type: ignore[assignment]
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        arm_audit = audit.get(arm)
        for rank, document_id in enumerate(rankings[arm], start=1):
            values = {
                str(facets[index]["facet_id"]): f_maps[index].get(document_id, 0.0)
                for index in range(len(facets))
            }
            detail = (
                arm_audit.get(document_id, {})
                if isinstance(arm_audit, Mapping)
                else {}
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "topic_id": topic_id,
                    "arm": arm,
                    "rank": rank,
                    "document_id": document_id,
                    "G": features["G"][document_id],  # type: ignore[index]
                    "N": features["N"][document_id],  # type: ignore[index]
                    "R": features["R"][document_id],  # type: ignore[index]
                    "L": max(values.values(), default=0.0),
                    "facet_percentiles": values,
                    "objective": detail.get("objective"),
                    "coverage_bonus": detail.get("coverage_bonus", 0.0),
                    "coverage_facet": detail.get("coverage_facet"),
                    "redundancy_penalty": detail.get("redundancy_penalty", 0.0),
                    "protected_head": arm == "RRF100-TETHERED-DUAL" and rank <= HEAD_SIZE,
                }
            )
    return rows


def freeze_soft_rankings(
    *,
    topic_inputs: Mapping[str, Mapping[str, object]],
    controls: Mapping[str, Mapping[str, object]],
    input_paths: Mapping[str, Path],
    output: Path,
) -> dict[str, object]:
    """Create and seal qrels-isolated soft-coverage ranking artifacts."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"create-only freeze output already exists: {output}")
    if set(topic_inputs) != set(PILOT_TOPIC_IDS) or set(controls) != set(PILOT_TOPIC_IDS):
        _reject_topics([*topic_inputs, *controls])
        raise ValueError("inputs must contain the exact four pilot topics")
    if not input_paths:
        raise ValueError("authenticated input paths are required")
    ranking_rows: list[dict[str, object]] = []
    topic_summary: dict[str, dict[str, object]] = {}
    for topic_id in PILOT_TOPIC_IDS:
        _reject_topics([topic_id, topic_inputs[topic_id].get("topic_id")])
        rankings, audit = build_soft_permutations(topic_inputs[topic_id], controls[topic_id])
        expected = {str(value) for value in topic_inputs[topic_id]["docids"]}  # type: ignore[index]
        complete = {
            arm: len(rankings[arm]) == len(expected) and set(rankings[arm]) == expected
            for arm in ARMS
        }
        if not all(complete.values()):
            raise ValueError(f"topic {topic_id} has an incomplete permutation")
        ranking_rows.extend(_ranking_rows(topic_id, topic_inputs[topic_id], rankings, audit))
        topic_summary[topic_id] = {
            "accepted_union_count": len(expected),
            "ranking_counts": {arm: len(rankings[arm]) for arm in ARMS},
            "complete_permutations": complete,
        }
    parameters = _parameters()
    bindings = {
        "schema_version": SCHEMA_VERSION,
        "qrels_opened": False,
        "inputs": {
            name: _relative_binding(path) for name, path in sorted(input_paths.items())
        },
    }
    artifacts = {
        "parameters.json": _pretty_bytes(parameters),
        "input_bindings.json": _pretty_bytes(bindings),
        "rankings.jsonl": _jsonl_bytes(ranking_rows),
    }
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "rankings_frozen_before_evaluation",
        "qrels_opened": False,
        "topic_ids": list(PILOT_TOPIC_IDS),
        "arms": list(ARMS),
        "ranking_row_count": len(ranking_rows),
        "protected_topic_count": 0,
        **{name: 0 for name in _COUNTERS},
        "external_cost_usd": 0.0,
        "topic_summary": topic_summary,
        "artifacts": {
            name: {"bytes": len(content), "sha256": _sha256(content)}
            for name, content in artifacts.items()
        },
    }
    artifacts["summary.json"] = _pretty_bytes(summary)
    output.mkdir(parents=True)
    for name, content in artifacts.items():
        _exclusive_write(output / name, content)
    seal_material = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": "sealed_before_evaluation",
        "qrels_opened": False,
        "files": {
            name: {"bytes": len(content), "sha256": _sha256(content)}
            for name, content in sorted(artifacts.items())
        },
    }
    seal = {**seal_material, "root_sha256": _sha256(_canonical_bytes(seal_material))}
    _exclusive_write(output / "SEALED.json", _pretty_bytes(seal))
    verify_soft_freeze(output)
    return summary


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def verify_soft_freeze(path: Path) -> dict[str, object]:
    """Verify the seal, safety receipt, and complete-permutation semantics."""

    root = Path(path)
    expected_names = {
        "parameters.json", "input_bindings.json", "rankings.jsonl", "summary.json", "SEALED.json"
    }
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise ValueError("freeze is unreadable") from exc
    if {entry.name for entry in entries} != expected_names or not all(
        entry.is_file() and not entry.is_symlink() for entry in entries
    ):
        raise ValueError("freeze has missing, extra, or unsafe files")
    seal = _read_object(root / "SEALED.json", "seal")
    material = {
        key: seal.get(key) for key in ("schema_version", "status", "qrels_opened", "files")
    }
    if (
        material["schema_version"] != SEAL_SCHEMA_VERSION
        or material["status"] != "sealed_before_evaluation"
        or material["qrels_opened"] is not False
        or seal.get("root_sha256") != _sha256(_canonical_bytes(material))
        or not isinstance(material["files"], Mapping)
    ):
        raise ValueError("seal SHA-256 or contract differs")
    for name in sorted(expected_names - {"SEALED.json"}):
        content = (root / name).read_bytes()
        if material["files"].get(name) != {  # type: ignore[union-attr]
            "bytes": len(content), "sha256": _sha256(content)
        }:
            raise ValueError(f"artifact SHA-256 differs: {name}")
    parameters = _read_object(root / "parameters.json", "parameters")
    summary = _read_object(root / "summary.json", "summary")
    bindings = _read_object(root / "input_bindings.json", "input bindings")
    if parameters != _parameters():
        raise ValueError("ranking parameters differ")
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != "rankings_frozen_before_evaluation"
        or summary.get("qrels_opened") is not False
        or summary.get("topic_ids") != list(PILOT_TOPIC_IDS)
        or summary.get("arms") != list(ARMS)
        or summary.get("protected_topic_count") != 0
        or any(summary.get(name) != 0 for name in _COUNTERS)
        or summary.get("external_cost_usd") != 0.0
        or bindings.get("schema_version") != SCHEMA_VERSION
        or bindings.get("qrels_opened") is not False
    ):
        raise ValueError("freeze safety or semantic contract differs")
    rows = _read_jsonl(root / "rankings.jsonl")
    by_topic_arm: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        topic_id, arm = str(row.get("topic_id")), str(row.get("arm"))
        _reject_topics([topic_id])
        if arm not in ARMS:
            raise ValueError("ranking arm is invalid")
        by_topic_arm.setdefault((topic_id, arm), []).append(row)
    populations: dict[str, set[str]] = {}
    for topic_id in PILOT_TOPIC_IDS:
        for arm in ARMS:
            arm_rows = by_topic_arm.get((topic_id, arm), [])
            ranks = [row.get("rank") for row in arm_rows]
            if ranks != list(range(1, len(arm_rows) + 1)):
                raise ValueError("ranking positions are not contiguous")
            population = {str(row.get("document_id")) for row in arm_rows}
            if len(population) != len(arm_rows):
                raise ValueError("ranking contains duplicate documents")
            if topic_id in populations and population != populations[topic_id]:
                raise ValueError("ranking populations differ across arms")
            populations[topic_id] = population
    if len(rows) != summary.get("ranking_row_count"):
        raise ValueError("ranking row count differs")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--freeze", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        result = verify_soft_freeze(args.freeze)
    else:
        topic_inputs, controls, source_paths = load_authenticated_inputs()
        result = freeze_soft_rankings(
            topic_inputs=topic_inputs,
            controls=controls,
            input_paths=source_paths,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
