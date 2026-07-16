"""Projection-only evaluation for protected tethered facet rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .deep_facet_candidate_evaluate import evaluate_ranking
from .deep_facet_candidate_rank import verify_seal as verify_prior_seal
from .tethered_facet_two_basket import (
    PILOT_TOPIC_IDS,
    average_rank_percentiles,
    verify_freeze,
)
from .tethered_facet_minilm_score import _selected_top4


TOPIC_IDS = PILOT_TOPIC_IDS
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
NOVEL_RELEVANT_TOTAL = 177
SCHEMA_VERSION = "tethered-facet-evaluation-v1"
PRIOR_SCHEMA_VERSION = "deep-facet-candidate-evaluation-v1"
PRIOR_EVALUATION_FILES = frozenset(
    {
        "qrels_projection.jsonl",
        "qrels_access_receipt.json",
        "metrics.json",
        "decision.json",
        "summary.json",
    }
)
PRIOR_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "qrels_opened",
        "seal_sha256",
        "topic_ids",
        "upstream_mutation_forbidden",
        "seal_root_sha256",
        "qrels_source_name",
        "qrels_projection_rows",
        "qrels_projection_sha256",
        "evaluator_code_sha256",
    }
)
# This is a post-qrels historical-integrity anchor for one already-exposed
# evaluation. It is not evidence of qrels blindness or fresh generalization.
HISTORICAL_PRIOR_EVALUATION_IDENTITY: Mapping[str, object] = {
    "schema_version": PRIOR_SCHEMA_VERSION,
    "topic_ids": list(TOPIC_IDS),
    "qrels_projection_rows": 4633,
    "files": {
        "qrels_access_receipt.json": (
            "eaa4cdbe63f9a5ef8d93c8ddd16f8dc2f361867986310d8dfab78e3a005c1cca"
        ),
        "qrels_projection.jsonl": (
            "03fc4bd18be36b7ea2d446975fec9fe17ac6698dcf068918c6bb228e9aab5e87"
        ),
        "metrics.json": (
            "9a5ca726c9e99bab634997ccf0cf632ea096f696f17631a477c505f004a19196"
        ),
        "decision.json": (
            "f7ebefa9bc9564e6f264f519fc46734f4583a05e7883a76c29af60dbb2f0ee3c"
        ),
        "summary.json": (
            "4a1e0bdfc6fa0221431397bc3114ac0d0359d2c375288dd5e2df032d0c6dd61f"
        ),
    },
}

NOISE_PATTERN_REGEX: Mapping[str, str] = {
    "wrong_domain": r"\b(?:wrong domain|another domain|unrelated field|different industry)\b",
    "generic_process": r"\b(?:step[- ]by[- ]step|general process|how to|best practices)\b",
    "dictionary_or_scrabble": r"\b(?:dictionary|scrabble|word finder|anagram)\b",
    "essay_or_homework": r"\b(?:essay|homework|assignment|term paper)\b",
    "pet_health": r"\b(?:pet health|veterinar(?:y|ian)|dog health|cat health)\b",
}


@dataclass(frozen=True)
class Task3Snapshot:
    freeze_dir: Path
    seal: Mapping[str, object]
    seal_bytes: bytes
    bindings: Mapping[str, object]
    summary: Mapping[str, object]
    artifact_buffers: Mapping[str, bytes]
    producer_buffers: Mapping[str, bytes]
    producer_hashes: Mapping[str, str]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_object_bytes(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, content


def _exclusive_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())


def _binding(path: Path, content: bytes) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": len(content),
        "sha256": _sha256(content),
    }


def _parse_jsonl(content: bytes, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(content.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is invalid") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} rows must be objects")
        rows.append(row)
    return rows


def _read_jsonl_bytes(path: Path, label: str) -> tuple[list[dict[str, object]], bytes]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    return _parse_jsonl(content, label), content


def _bound_source(inputs: Mapping[str, object], name: str) -> tuple[bytes, str]:
    raw = inputs.get(name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Task 3 lacks required representative source: {name}")
    path = Path(str(raw.get("path")))
    content = path.read_bytes()
    digest = _sha256(content)
    if raw != {"path": str(path), "bytes": len(content), "sha256": digest}:
        raise ValueError(f"Task 3 representative source hash drifted: {name}")
    return content, digest


def _representative_candidate_maps(
    contents: Mapping[str, bytes],
) -> tuple[
    dict[tuple[str, str, str], dict[str, object]],
    dict[tuple[str, str, str], dict[str, object]],
]:
    facet_rows = _parse_jsonl(contents["facet_candidates"], "facet candidates")
    tethered_rows = _parse_jsonl(contents["tethered_candidates"], "tethered candidates")
    facet = {
        (str(row.get("topic_id")), str(row.get("facet_id")), str(row.get("document_id"))): row
        for row in facet_rows
    }
    tethered = {
        (str(row.get("topic_id")), str(row.get("facet_id")), str(row.get("document_id"))): row
        for row in tethered_rows
    }
    if len(facet) != len(facet_rows) or set(facet) != set(tethered):
        raise ValueError("Task 3 representative candidate coverage drifted")
    return facet, tethered


def _task3_snapshot(freeze_dir: Path) -> Task3Snapshot:
    """Capture and verify one immutable Task 3 filesystem view."""

    freeze_dir = Path(freeze_dir)
    names = (
        "SEALED.json", "parameters.json", "input_bindings.json",
        "rankings.jsonl", "prefixes.json", "summary.json",
    )
    entries = list(freeze_dir.iterdir())
    if {path.name for path in entries} != set(names) or not all(
        path.is_file() and not path.is_symlink() for path in entries
    ):
        raise ValueError("Task 3 snapshot has missing or extra artifacts")
    artifact_buffers: dict[str, bytes] = {}
    for name in names:
        path = freeze_dir / name
        if path.is_symlink():
            raise ValueError(f"Task 3 snapshot source is unsafe: {name}")
        try:
            artifact_buffers[name] = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Task 3 snapshot source is unreadable: {name}") from exc
    try:
        seal = json.loads(artifact_buffers["SEALED.json"])
        bindings = json.loads(artifact_buffers["input_bindings.json"])
        summary = json.loads(artifact_buffers["summary.json"])
    except json.JSONDecodeError as exc:
        raise ValueError("Task 3 snapshot metadata is invalid") from exc
    if not all(isinstance(value, Mapping) for value in (seal, bindings, summary)):
        raise ValueError("Task 3 snapshot metadata must be objects")
    seal_material = {
        key: seal.get(key)
        for key in ("schema_version", "status", "qrels_opened", "files")
    }
    if (
        seal.get("schema_version") != "tethered-facet-two-basket-seal-v1"
        or seal.get("status") != "sealed_before_qrels"
        or seal.get("qrels_opened") is not False
        or seal.get("root_sha256") != _sha256(
            json.dumps(
                seal_material, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
        )
    ):
        raise ValueError("Task 3 snapshot seal root or contract differs")
    inputs = bindings.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("Task 3 snapshot input bindings are missing")
    producer_buffers: dict[str, bytes] = {}
    for name, raw in inputs.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Task 3 snapshot input binding is invalid: {name}")
        path = Path(str(raw.get("path")))
        if path.is_symlink():
            raise ValueError(f"Task 3 snapshot producer is unsafe: {name}")
        try:
            producer_buffers[str(name)] = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Task 3 snapshot producer is unreadable: {name}") from exc
    verify_freeze(
        freeze_dir,
        artifact_buffers=artifact_buffers,
        input_buffers=producer_buffers,
    )
    return Task3Snapshot(
        freeze_dir=freeze_dir,
        seal=seal,
        seal_bytes=artifact_buffers["SEALED.json"],
        bindings=bindings,
        summary=summary,
        artifact_buffers=artifact_buffers,
        producer_buffers=producer_buffers,
        producer_hashes={
            name: _sha256(content) for name, content in producer_buffers.items()
        },
    )


def _representative_sources(
    bindings: Mapping[str, object], *, include_telemetry: bool = False,
    snapshot_buffers: Mapping[str, bytes] | None = None,
) -> tuple[dict[str, bytes], dict[str, str]]:
    inputs = bindings.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Task 3 input bindings are missing")
    names = [
        "facet_candidates",
        "facet_window_scores",
        "tethered_candidates",
        "tethered_window_scores",
        "tethered_document_scores",
    ]
    if include_telemetry:
        names.extend(("tethered_preflight", "tethered_scoring_receipt"))
    contents: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for name in names:
        if snapshot_buffers is None:
            contents[name], hashes[name] = _bound_source(inputs, name)
        else:
            raw = inputs.get(name)
            content = snapshot_buffers.get(name)
            if not isinstance(raw, Mapping) or not isinstance(content, bytes):
                raise ValueError(f"Task 3 snapshot lacks required source: {name}")
            digest = _sha256(content)
            if raw != {
                "path": str(raw.get("path")), "bytes": len(content), "sha256": digest,
            }:
                raise ValueError(f"Task 3 snapshot source hash drifted: {name}")
            contents[name], hashes[name] = content, digest
    return contents, hashes


def _window_maps(
    contents: Mapping[str, bytes],
) -> tuple[
    dict[tuple[str, str, str], list[dict[str, object]]],
    dict[tuple[str, str, str], list[dict[str, object]]],
    dict[tuple[str, str, str], dict[str, object]],
]:
    facet_rows = _parse_jsonl(contents["facet_window_scores"], "facet window scores")
    tethered_rows = _parse_jsonl(contents["tethered_window_scores"], "tethered window scores")
    document_rows = _parse_jsonl(contents["tethered_document_scores"], "tethered document scores")
    facet: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    tethered: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for target, rows in ((facet, facet_rows), (tethered, tethered_rows)):
        for row in rows:
            target[(str(row.get("topic_id")), str(row.get("facet_id", row.get("variant"))), str(row.get("document_id")))].append(row)
    documents = {
        (str(row.get("topic_id")), str(row.get("facet_id")), str(row.get("document_id"))): row
        for row in document_rows
    }
    if len(documents) != len(document_rows):
        raise ValueError("Task 2 document score identity is duplicated")
    return facet, tethered, documents


def _passage_evidence(
    *,
    identity: tuple[str, str, str],
    arm: str,
    candidate: Mapping[str, object],
    windows: Mapping[tuple[str, str, str], list[dict[str, object]]],
    tethered_documents: Mapping[tuple[str, str, str], Mapping[str, object]],
    hashes: Mapping[str, str],
    expected_model: str,
    expected_model_revision: str,
) -> tuple[str, dict[str, object]]:
    raw = windows.get(identity, [])
    if not raw:
        raise ValueError("representative lacks authenticated scored windows")
    selected = _selected_top4(raw)
    if arm == "TETHERED-2B":
        document = tethered_documents.get(identity)
        expected = document.get("window_hashes") if isinstance(document, Mapping) else None
        if (
            not isinstance(expected, list)
            or [row.get("window_sha256") for row in selected] != expected
            or document.get("model") != expected_model
            or document.get("model_revision") != expected_model_revision
        ):
            raise ValueError("Task 2 selected passage provenance drifted")
    chosen = min(
        selected,
        key=lambda row: (-float(row["score"]), int(row["document_start_token"]), str(row["window_id"])),
    )
    passage = chosen.get("window_text")
    if (
        not isinstance(passage, str)
        or not passage
        or chosen.get("window_sha256") != _sha256(passage.encode("utf-8"))
        or chosen.get("query_sha256") != candidate.get("query_sha256")
        or chosen.get("document_sha256", chosen.get("text_sha256")) != candidate.get("text_sha256")
        or chosen.get("model") != expected_model
        or chosen.get("model_revision") != expected_model_revision
    ):
        raise ValueError("representative passage query/text/window provenance drifted")
    prefix = "tethered" if arm == "TETHERED-2B" else "facet"
    return passage, {
        "candidate_source_sha256": hashes[f"{prefix}_candidates"],
        "window_score_source_sha256": hashes[f"{prefix}_window_scores"],
        "document_score_source_sha256": hashes.get("tethered_document_scores") if arm == "TETHERED-2B" else None,
        "query_sha256": chosen["query_sha256"],
        "text_sha256": candidate["text_sha256"],
        "window_sha256": chosen["window_sha256"],
        "window_id": chosen["window_id"],
        "model": chosen["model"],
        "model_revision": chosen["model_revision"],
        "document_start_token": chosen["document_start_token"],
        "document_end_token": chosen["document_end_token"],
        "rank_source": "prior_bm25_rank",
    }


def build_representatives(
    freeze_dir: Path,
    qrels: Mapping[str, Mapping[str, int]],
    *,
    max_per_class: int = 4,
    snapshot: Task3Snapshot | None = None,
) -> list[dict[str, object]]:
    """Build bounded movement evidence from Task 3-bound raw score sources."""

    if set(qrels) != set(PILOT_TOPIC_IDS) or any(topic in PROTECTED_TOPIC_IDS for topic in qrels):
        raise ValueError("qrels must contain the exact protected pilot topics")
    if type(max_per_class) is not int or max_per_class <= 0 or max_per_class > 8:
        raise ValueError("representative bound must be between 1 and 8")
    snapshot = snapshot or _task3_snapshot(freeze_dir)
    bindings = snapshot.bindings
    ranking_bytes = snapshot.artifact_buffers["rankings.jsonl"]
    raw_inputs = bindings.get("topic_inputs")
    if not isinstance(raw_inputs, Mapping):
        raise ValueError("Task 3 semantic score maps are missing")
    contents, hashes = _representative_sources(
        bindings, snapshot_buffers=snapshot.producer_buffers
    )
    facet_candidates, tethered_candidates = _representative_candidate_maps(contents)
    facet_windows, tethered_windows, tethered_documents = _window_maps(contents)
    rankings = _parse_jsonl(ranking_bytes, "Task 3 rankings")
    ranking_hash = _sha256(ranking_bytes)
    by_identity: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rankings:
        by_identity[(str(row["topic_id"]), str(row["document_id"]))][str(row["arm"])] = row
    percentile_maps: dict[tuple[str, str, str], dict[str, float]] = {}
    facet_provenance: dict[tuple[str, str, str], tuple[str, str]] = {}
    for arm in ("FACET-2B", "TETHERED-2B"):
        arm_inputs = raw_inputs.get(arm)
        if not isinstance(arm_inputs, Mapping):
            raise ValueError("Task 3 semantic arm score maps are missing")
        for topic in PILOT_TOPIC_IDS:
            topic_input = arm_inputs.get(topic)
            facets = topic_input.get("facets") if isinstance(topic_input, Mapping) else None
            if not isinstance(facets, list):
                raise ValueError("Task 3 semantic facet scores are missing")
            for facet in facets:
                if not isinstance(facet, Mapping) or not isinstance(facet.get("scores"), Mapping):
                    raise ValueError("Task 3 semantic facet score map is invalid")
                percentile_maps[(arm, topic, str(facet.get("facet_id")))] = average_rank_percentiles(facet["scores"])  # type: ignore[arg-type]
                facet_provenance[(arm, topic, str(facet.get("facet_id")))] = (
                    str(facet.get("model")), str(facet.get("model_revision"))
                )
    movement: dict[str, list[tuple[str, str, str]]] = {"promoted": [], "demoted": []}
    for (topic, document), arms in by_identity.items():
        left, right = arms.get("FACET-2B"), arms.get("TETHERED-2B")
        if not left or not right:
            raise ValueError("Task 3 ranking arm coverage drifted")
        if left.get("source") == "facet_basket" and right.get("source") != "facet_basket":
            movement["demoted"].append((topic, str(left["generating_facet"]), document))
        if right.get("source") == "facet_basket" and left.get("source") != "facet_basket":
            movement["promoted"].append((topic, str(right["generating_facet"]), document))
    output: list[dict[str, object]] = []
    for movement_class, passage_arm in (("promoted", "TETHERED-2B"), ("demoted", "FACET-2B")):
        identities = sorted(movement[movement_class], key=lambda value: (PILOT_TOPIC_IDS.index(value[0]), value[1], value[2]))[:max_per_class]
        if not identities:
            raise ValueError(f"Task 4 diagnostics lack {movement_class} movement evidence")
        for identity in identities:
            topic, facet, document = identity
            key = (topic, facet, document)
            facet_candidate, tethered_candidate = facet_candidates[key], tethered_candidates[key]
            query, facet_query = tethered_candidate.get("query"), tethered_candidate.get("facet_query")
            suffix = "\n\nFocus: " + str(facet_query)
            if not isinstance(query, str) or not isinstance(facet_query, str) or not query.endswith(suffix):
                raise ValueError("representative query lacks exact narrative + Focus structure")
            narrative = query[:-len(suffix)]
            if not narrative or query != narrative + suffix:
                raise ValueError("representative query lacks exact narrative + Focus structure")
            windows = tethered_windows if passage_arm == "TETHERED-2B" else facet_windows
            candidate = tethered_candidate if passage_arm == "TETHERED-2B" else facet_candidate
            passage, passage_provenance = _passage_evidence(
                identity=key, arm=passage_arm, candidate=candidate, windows=windows,
                tethered_documents=tethered_documents, hashes=hashes,
                expected_model=facet_provenance[(passage_arm, topic, facet)][0],
                expected_model_revision=facet_provenance[(passage_arm, topic, facet)][1],
            )
            rows = by_identity[(topic, document)]
            left, right = rows["FACET-2B"], rows["TETHERED-2B"]
            facet_percentile = percentile_maps[("FACET-2B", topic, facet)].get(document)
            tethered_percentile = percentile_maps[("TETHERED-2B", topic, facet)].get(document)
            if facet_percentile is None or tethered_percentile is None:
                raise ValueError("representative lacks same-facet query-local percentiles")
            output.append({
                "topic_id": topic,
                "facet_id": facet,
                "movement": movement_class,
                "document_id": document,
                "narrative": narrative,
                "facet_query": facet_query,
                "selected_passage": passage,
                "facet_only_percentile": facet_percentile,
                "tethered_percentile": tethered_percentile,
                "qrels_grade": int(qrels.get(topic, {}).get(document, 0)),
                "facet_only_final_rank": int(left["rank"]),
                "tethered_final_rank": int(right["rank"]),
                "prior_bm25_rank": int(tethered_candidate["prior_bm25_rank"]),
                "passage_provenance": passage_provenance,
                "ranking_provenance": {
                    "task3_rankings_sha256": ranking_hash,
                    "facet_only_source": left["source"],
                    "tethered_source": right["source"],
                    "generating_facet": facet,
                    "percentile_method": "query_local_average_rank",
                    "rank_source": "Task 3 sealed rankings.jsonl",
                },
            })
    return output


def validate_protected_head(*, rrf: Sequence[object], arm: Sequence[object]) -> None:
    """Require exact ordered identity of the protected first 100 documents."""

    if len(rrf) < 100 or len(arm) < 100 or list(rrf[:100]) != list(arm[:100]):
        raise ValueError("arm top 100 must exactly match RRF top 100")


def _parse_projection(content: bytes) -> dict[str, dict[str, int]]:
    lines = content.splitlines()
    parsed: list[tuple[str, str, int]] = []
    observed_order: list[str] = []
    previous: str | None = None
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise TypeError
            raw_topic, raw_document, raw_grade = (
                row["topic_id"], row["document_id"], row["grade"]
            )
            if (
                not isinstance(raw_topic, str)
                or not isinstance(raw_document, str)
                or not isinstance(raw_grade, int)
                or isinstance(raw_grade, bool)
            ):
                raise TypeError
            topic_id, document_id, grade = raw_topic, raw_document, raw_grade
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"qrels projection line {line_number} is invalid") from exc
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"qrels projection contains protected topic {topic_id}")
        if topic_id not in TOPIC_IDS:
            raise ValueError(f"qrels projection contains unexpected topic {topic_id}")
        if not document_id:
            raise ValueError("qrels projection contains an empty document ID")
        if topic_id != previous:
            observed_order.append(topic_id)
            previous = topic_id
        parsed.append((topic_id, document_id, grade))
    if observed_order != list(TOPIC_IDS):
        raise ValueError("qrels projection must contain the exact topics in order")
    result: dict[str, dict[str, int]] = {topic: {} for topic in TOPIC_IDS}
    for topic_id, document_id, grade in parsed:
        if document_id in result[topic_id]:
            raise ValueError("qrels projection contains duplicate topic-document identities")
        result[topic_id][document_id] = grade
    if any(not result[topic] for topic in TOPIC_IDS):
        raise ValueError("qrels projection must contain the exact topics in order")
    return result


def load_projection(path: Path) -> dict[str, dict[str, int]]:
    """Read the sealed JSONL projection and enforce its topic firewall."""

    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"qrels projection is unreadable: {path}") from exc
    return _parse_projection(content)


def _require_exact_topics(value: Mapping[str, object], label: str) -> None:
    if list(value) != list(TOPIC_IDS):
        raise ValueError(f"{label} must contain the exact topics in order")


def derive_novel_set(
    qrels: Mapping[str, Mapping[str, int]],
    accepted_facet_candidates: Mapping[str, Sequence[str] | set[str]],
    original_at_1000: Mapping[str, Sequence[str]],
    *,
    expected_total: int = NOVEL_RELEVANT_TOTAL,
) -> dict[str, set[str]]:
    """Derive relevant accepted-facet candidates absent from original@1000."""

    for value, label in (
        (qrels, "qrels"),
        (accepted_facet_candidates, "accepted facet candidates"),
        (original_at_1000, "original rankings"),
    ):
        _require_exact_topics(value, label)
    result = {
        topic: {
            document_id
            for document_id in map(str, accepted_facet_candidates[topic])
            if int(qrels[topic].get(document_id, 0)) >= 2
            and document_id not in set(map(str, original_at_1000[topic][:1000]))
        }
        for topic in TOPIC_IDS
    }
    observed = sum(map(len, result.values()))
    if observed != expected_total:
        raise ValueError(
            f"frozen novel relevant set must contain exactly {expected_total} documents; "
            f"found {observed}"
        )
    return result


def _mean(values: Sequence[object]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return math.fsum(present) / len(present) if present else None


def evaluate_arm(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    novel_set: Mapping[str, set[str]],
    *,
    ranking_rows: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    depths: Sequence[int] = (500, 1000),
) -> dict[str, object]:
    """Evaluate one frozen arm with shared ranking metrics and diagnostics."""

    for value, label in ((rankings, "rankings"), (qrels, "qrels"), (novel_set, "novel set")):
        _require_exact_topics(value, label)
    ordered_depths = tuple(dict.fromkeys(int(depth) for depth in depths))
    if not ordered_depths or any(depth <= 0 for depth in ordered_depths):
        raise ValueError("evaluation depths must be positive")
    per_topic: dict[str, dict[str, object]] = {}
    for topic in TOPIC_IDS:
        ranking = list(map(str, rankings[topic]))
        metrics = evaluate_ranking(ranking, qrels[topic], depths=ordered_depths)
        topic_row: dict[str, object] = {
            "ndcg@10": metrics["ndcg@10"],
            "ndcg@100": metrics["ndcg@100"],
        }
        for depth in ordered_depths:
            values = metrics[str(depth)]
            if not isinstance(values, Mapping):
                raise ValueError("shared evaluator returned invalid depth metrics")
            retained = len(set(ranking[:depth]).intersection(novel_set[topic]))
            topic_row.update(
                {
                    f"recall@{depth}": values["recall"],
                    f"graded_recall@{depth}": values["graded_recall"],
                    f"judged_rate@{depth}": values["judged_rate"],
                    f"novel_retained@{depth}": retained,
                    f"novel_retention@{depth}": (
                        retained / len(novel_set[topic]) if novel_set[topic] else None
                    ),
                }
            )
        per_topic[topic] = topic_row
    novel_total = sum(map(len, novel_set.values()))
    aggregate: dict[str, object] = {
        "ndcg@10": _mean([per_topic[topic]["ndcg@10"] for topic in TOPIC_IDS]),
        "ndcg@100": _mean([per_topic[topic]["ndcg@100"] for topic in TOPIC_IDS]),
    }
    for depth in ordered_depths:
        for field in ("recall", "graded_recall", "judged_rate"):
            aggregate[f"{field}@{depth}"] = _mean(
                [per_topic[topic][f"{field}@{depth}"] for topic in TOPIC_IDS]
            )
        retained = sum(int(per_topic[topic][f"novel_retained@{depth}"]) for topic in TOPIC_IDS)
        aggregate[f"novel_retained@{depth}"] = retained
        aggregate[f"novel_retention@{depth}"] = retained / novel_total if novel_total else None

    basket: dict[str, dict[str, int]] = defaultdict(
        lambda: {"selected_count": 0, "relevant_count": 0}
    )
    facets: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"selected_count": 0, "relevant_count": 0}
    )
    if ranking_rows is not None:
        _require_exact_topics(ranking_rows, "ranking rows")
        diagnostic_depth = 500 if 500 in ordered_depths else max(ordered_depths)
        for topic in TOPIC_IDS:
            for row in ranking_rows[topic]:
                rank = int(row.get("rank", 0))
                if not 1 <= rank <= diagnostic_depth:
                    continue
                document_id = str(row.get("document_id"))
                source = str(row.get("source"))
                relevant = int(qrels[topic].get(document_id, 0)) >= 2
                basket[source]["selected_count"] += 1
                basket[source]["relevant_count"] += int(relevant)
                facet_id = row.get("generating_facet")
                if facet_id is not None:
                    facet = facets[str(facet_id)]
                    facet["selected_count"] = int(facet["selected_count"]) + 1
                    facet["relevant_count"] = int(facet["relevant_count"]) + int(relevant)
        for facet in facets.values():
            facet["relevant_yield"] = int(facet["relevant_count"]) / int(facet["selected_count"])
    return {
        "per_topic": per_topic,
        "aggregate": aggregate,
        "basket_contributions": dict(sorted(basket.items())),
        "facet_yield": dict(sorted(facets.items())),
    }


def _at_least(value: object, *controls: object) -> bool:
    return all(float(value) >= float(control) for control in controls)


def decide(evidence: Mapping[str, object]) -> dict[str, object]:
    """Apply the frozen diagnostic rule without discretionary interpretation."""

    aggregate = evidence["aggregate"]
    per_topic = evidence["per_topic"]
    if not isinstance(aggregate, Mapping) or not isinstance(per_topic, Mapping):
        raise ValueError("decision evidence lacks aggregate or per-topic metrics")
    rrf = aggregate["RRF"]
    facet = aggregate["FACET-2B"]
    tethered = aggregate["TETHERED-2B"]
    if not all(isinstance(value, Mapping) for value in (rrf, facet, tethered)):
        raise ValueError("decision evidence lacks required arms")
    if not (
        isinstance(rrf, Mapping)
        and isinstance(facet, Mapping)
        and isinstance(tethered, Mapping)
    ):
        raise ValueError("decision evidence lacks required arm metrics")
    recall500 = all(
        _at_least(tethered[f"{field}@500"], rrf[f"{field}@500"], facet[f"{field}@500"])
        for field in ("recall", "graded_recall")
    ) and any(
        float(tethered[f"{field}@500"]) > float(facet[f"{field}@500"])
        for field in ("recall", "graded_recall")
    )
    recall1000 = all(
        _at_least(tethered[f"{field}@1000"], rrf[f"{field}@1000"], facet[f"{field}@1000"])
        for field in ("recall", "graded_recall")
    )
    per_topic_loss = True
    for topic in TOPIC_IDS:
        row = per_topic.get(topic)
        if not isinstance(row, Mapping):
            raise ValueError("decision evidence lacks exact per-topic metrics")
        candidate = row["TETHERED-2B"]
        if not isinstance(candidate, Mapping):
            raise ValueError("decision evidence lacks TETHERED-2B per-topic metrics")
        for control_name in ("RRF", "FACET-2B"):
            control = row[control_name]
            if not isinstance(control, Mapping):
                raise ValueError("decision evidence lacks control per-topic metrics")
            for field in ("recall", "graded_recall"):
                if (
                    float(candidate[f"{field}@500"])
                    - float(control[f"{field}@500"])
                    < -0.02 - 1e-12
                ):
                    per_topic_loss = False
    judged_coverage = all(
        float(tethered[f"judged_rate@{depth}"])
        - float(control[f"judged_rate@{depth}"])
        >= -0.05 - 1e-12
        for depth in (500, 1000)
        for control in (rrf, facet)
    )
    guards = {
        "top100_identity": evidence.get("top100_identity") is True,
        "recall500": recall500,
        "novel500": int(tethered["novel_retained@500"]) >= 89,
        "recall1000": recall1000,
        "novel1000": int(tethered["novel_retained@1000"]) >= 142,
        "per_topic_loss": per_topic_loss,
        "judged_coverage": judged_coverage,
        "basket_capacity": evidence.get("documented_basket_shortage") is not True,
    }
    failed = [name for name, passed in guards.items() if not passed]
    if not failed:
        label = "mechanical_pass"
    elif set(failed).issubset({"judged_coverage", "basket_capacity"}):
        label = "inconclusive"
    else:
        label = "mechanical_fail"
    return {"label": label, "guards": guards, "failed_guards": failed}


def _load_frozen_rankings(
    snapshot: Task3Snapshot,
) -> tuple[
    dict[str, dict[str, list[str]]],
    dict[str, dict[str, list[dict[str, object]]]],
    dict[str, object],
    dict[str, object],
    bytes,
    bytes,
]:
    bindings = dict(snapshot.bindings)
    summary = dict(snapshot.summary)
    binding_bytes = snapshot.artifact_buffers["input_bindings.json"]
    ranking_bytes = snapshot.artifact_buffers["rankings.jsonl"]
    raw_topic_inputs = bindings.get("topic_inputs")
    if not isinstance(raw_topic_inputs, Mapping) or set(raw_topic_inputs) != {
        "FACET-2B",
        "TETHERED-2B",
    }:
        raise ValueError("freeze lacks exact arm semantic inputs")
    facet_inputs = raw_topic_inputs["FACET-2B"]
    if not isinstance(facet_inputs, Mapping) or set(facet_inputs) != set(TOPIC_IDS):
        raise ValueError("freeze semantic inputs lack exact topics in order")
    for topic in TOPIC_IDS:
        raw = facet_inputs[topic]
        if not isinstance(raw, Mapping) or not isinstance(raw.get("rrf"), list):
            raise ValueError("freeze semantic topic input is invalid")

    collected: dict[str, dict[str, list[dict[str, object]]]] = {
        topic: {"FACET-2B": [], "TETHERED-2B": []} for topic in TOPIC_IDS
    }
    for line_number, line in enumerate(ranking_bytes.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"frozen rankings line {line_number} is invalid") from exc
        if not isinstance(row, dict):
            raise ValueError("frozen ranking rows must be objects")
        topic, arm = str(row.get("topic_id")), str(row.get("arm"))
        if topic in PROTECTED_TOPIC_IDS:
            raise ValueError("frozen rankings contain a protected topic")
        if topic not in collected or arm not in collected[topic]:
            raise ValueError("frozen rankings contain an unexpected topic or arm")
        collected[topic][arm].append(row)
    rankings: dict[str, dict[str, list[str]]] = {
        arm: {} for arm in ("FACET-2B", "TETHERED-2B")
    }
    for topic in TOPIC_IDS:
        for arm in ("FACET-2B", "TETHERED-2B"):
            rows = sorted(collected[topic][arm], key=lambda row: int(row["rank"]))
            if [int(row["rank"]) for row in rows] != list(range(1, len(rows) + 1)):
                raise ValueError("frozen ranking is not contiguous")
            ids = [str(row["document_id"]) for row in rows]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("frozen ranking is empty or contains duplicates")
            collected[topic][arm] = rows
            rankings[arm][topic] = ids
    return rankings, collected, bindings, summary, binding_bytes, ranking_bytes


def _validate_prior_novel_set(
    metrics: Mapping[str, object], novel: Mapping[str, set[str]]
) -> None:
    if metrics.get("topic_ids") != list(TOPIC_IDS):
        raise ValueError("prior metrics lack exact topics in order")
    discovery = metrics.get("discovery")
    if not isinstance(discovery, Mapping) or set(discovery) != set(TOPIC_IDS):
        raise ValueError("prior metrics lack frozen discovery evidence")
    for topic in TOPIC_IDS:
        row = discovery[topic]
        if not isinstance(row, Mapping) or not isinstance(row.get("novel_relevant_ids"), list):
            raise ValueError("prior metrics lack frozen novel-set evidence")
        if set(map(str, row["novel_relevant_ids"])) != novel[topic]:
            raise ValueError("derived novel set differs from prior frozen metrics")


def _load_prior_rrf(
    prior_freeze: Path,
) -> tuple[dict[str, list[str]], dict[str, set[str]], dict[str, list[str]], bytes]:
    """Load RRF and novel-set evidence only after its containing seal verifies."""

    rankings_path = prior_freeze / "rankings.jsonl"
    try:
        content = rankings_path.read_bytes()
    except OSError as exc:
        raise ValueError("verified prior rankings are unreadable") from exc
    rows: dict[str, list[dict[str, object]]] = {topic: [] for topic in TOPIC_IDS}
    for line_number, line in enumerate(content.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"verified prior rankings line {line_number} is invalid") from exc
        if not isinstance(row, dict):
            raise ValueError("verified prior ranking rows must be objects")
        topic, arm = str(row.get("topic_id")), str(row.get("arm"))
        if topic in PROTECTED_TOPIC_IDS:
            raise ValueError("verified prior rankings contain a protected topic")
        if topic not in rows:
            raise ValueError("verified prior rankings contain an unexpected topic")
        if arm == "RRF":
            rows[topic].append(row)
    rankings: dict[str, list[str]] = {}
    accepted_facets: dict[str, set[str]] = {}
    original_at_1000: dict[str, list[str]] = {}
    for topic in TOPIC_IDS:
        ordered = sorted(rows[topic], key=lambda row: int(row["rank"]))
        if [int(row["rank"]) for row in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError("verified prior RRF ranking is not contiguous")
        document_ids = [str(row["document_id"]) for row in ordered]
        if not document_ids or len(document_ids) != len(set(document_ids)):
            raise ValueError("verified prior RRF ranking is empty or contains duplicates")
        rankings[topic] = document_ids
        accepted_facets[topic] = set()
        original_rows: list[tuple[int, str]] = []
        for row in ordered:
            document_id = str(row["document_id"])
            original_rank = row.get("original_rank")
            if original_rank is not None:
                rank = int(original_rank)
                if rank <= 0:
                    raise ValueError("verified prior original rank is invalid")
                if rank <= 1000:
                    original_rows.append((rank, document_id))
            best_facet_rank = row.get("best_facet_rank")
            percentiles = row.get("facet_percentiles")
            has_facet_percentile = isinstance(percentiles, Mapping) and any(
                float(value) > 0.0 for value in percentiles.values()
            )
            if (
                original_rank is None
                and isinstance(best_facet_rank, int)
                and not isinstance(best_facet_rank, bool)
                and 0 < best_facet_rank < 10**9
                and has_facet_percentile
            ):
                accepted_facets[topic].add(document_id)
        original_at_1000[topic] = [document_id for _rank, document_id in sorted(original_rows)]
    return rankings, accepted_facets, original_at_1000, content


def _require_prior_metrics_equal_recomputation(
    metrics: Mapping[str, object],
    recomputed: Mapping[str, object],
    novel: Mapping[str, set[str]],
) -> None:
    """Reject even self-consistently restamped prior metric claims."""

    _validate_prior_novel_set(metrics, novel)
    if (
        metrics.get("schema_version") != PRIOR_SCHEMA_VERSION
        or metrics.get("novel_relevant_count") != sum(map(len, novel.values()))
    ):
        raise ValueError("prior metrics differ from recomputed RRF evidence")
    prior_aggregate = metrics.get("aggregate")
    prior_per_topic = metrics.get("per_topic")
    aggregate = recomputed.get("aggregate")
    per_topic = recomputed.get("per_topic")
    if not (
        isinstance(prior_aggregate, Mapping)
        and isinstance(prior_per_topic, Mapping)
        and isinstance(aggregate, Mapping)
        and isinstance(per_topic, Mapping)
    ):
        raise ValueError("prior metrics lack recomputable RRF evidence")
    claimed_aggregate = prior_aggregate.get("RRF")
    if not isinstance(claimed_aggregate, Mapping):
        raise ValueError("prior metrics lack recomputable RRF aggregate")
    fields = ("ndcg@10", "ndcg@100") + tuple(
        f"{field}@{depth}"
        for depth in (100, 500, 1000)
        for field in (
            "recall",
            "graded_recall",
            "judged_rate",
            "novel_retained",
            "novel_retention",
        )
    )
    if any(claimed_aggregate.get(field) != aggregate.get(field) for field in fields):
        raise ValueError("prior metrics differ from recomputed RRF aggregate")
    for topic in TOPIC_IDS:
        claimed_topic = prior_per_topic.get(topic)
        recomputed_topic = per_topic.get(topic)
        if not isinstance(claimed_topic, Mapping) or not isinstance(recomputed_topic, Mapping):
            raise ValueError("prior metrics lack recomputable RRF per-topic evidence")
        claimed_rrf = claimed_topic.get("RRF")
        if not isinstance(claimed_rrf, Mapping) or any(
            claimed_rrf.get(field) != recomputed_topic.get(field) for field in fields
        ):
            raise ValueError("prior metrics differ from recomputed RRF per-topic evidence")


def _require_historical_identity(contents: Mapping[str, bytes]) -> None:
    """Require the committed identity of the one historical evaluation leaf."""

    anchor = HISTORICAL_PRIOR_EVALUATION_IDENTITY
    expected_files = anchor.get("files")
    if (
        anchor.get("schema_version") != PRIOR_SCHEMA_VERSION
        or anchor.get("topic_ids") != list(TOPIC_IDS)
        or not isinstance(anchor.get("qrels_projection_rows"), int)
        or isinstance(anchor.get("qrels_projection_rows"), bool)
        or int(anchor.get("qrels_projection_rows", 0)) <= 0
        or not isinstance(expected_files, Mapping)
        or set(expected_files) != PRIOR_EVALUATION_FILES
        or set(contents) != PRIOR_EVALUATION_FILES
    ):
        raise ValueError("committed historical identity contract is invalid")
    if len(contents["qrels_projection.jsonl"].splitlines()) != int(
        anchor["qrels_projection_rows"]
    ):
        raise ValueError("prior evaluation differs from committed historical identity")
    for name in sorted(PRIOR_EVALUATION_FILES):
        expected = expected_files.get(name)
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or _sha256(contents[name]) != expected
        ):
            raise ValueError(
                f"prior evaluation differs from committed historical identity: {name}"
            )


def _per_topic_deltas(
    arms: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    fields = tuple(
        f"{metric}@{depth}"
        for depth in (500, 1000)
        for metric in ("recall", "graded_recall", "judged_rate")
    )
    result: dict[str, dict[str, dict[str, float | None]]] = {}
    for topic in TOPIC_IDS:
        result[topic] = {}
        for arm, control in (
            ("FACET-2B", "RRF"),
            ("TETHERED-2B", "RRF"),
            ("TETHERED-2B", "FACET-2B"),
        ):
            arm_topics = arms[arm]["per_topic"]
            control_topics = arms[control]["per_topic"]
            if not isinstance(arm_topics, Mapping) or not isinstance(control_topics, Mapping):
                raise ValueError("arm per-topic metrics are invalid")
            left, right = arm_topics[topic], control_topics[topic]
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                raise ValueError("arm topic metrics are invalid")
            result[topic][f"{arm}_vs_{control}"] = {
                field: (
                    float(left[field]) - float(right[field])
                    if left[field] is not None and right[field] is not None
                    else None
                )
                for field in fields
            }
    return result


def _extended_diagnostics(
    task3: Path | Task3Snapshot,
    qrels: Mapping[str, Mapping[str, int]],
    arms: Mapping[str, Mapping[str, object]],
    ranking_rows: Mapping[str, Mapping[str, list[dict[str, object]]]],
) -> dict[str, object]:
    snapshot = task3 if isinstance(task3, Task3Snapshot) else _task3_snapshot(task3)
    bindings, summary = snapshot.bindings, snapshot.summary
    contents, hashes = _representative_sources(
        bindings, include_telemetry=True,
        snapshot_buffers=snapshot.producer_buffers,
    )
    facet_candidates, tethered_candidates = _representative_candidate_maps(contents)
    texts = {
        identity: str(row.get("text")) for identity, row in tethered_candidates.items()
    }
    compiled = {
        name: re.compile(pattern, flags=re.IGNORECASE)
        for name, pattern in NOISE_PATTERN_REGEX.items()
    }
    noise_rows: list[dict[str, object]] = []
    for arm in ("FACET-2B", "TETHERED-2B"):
        selected: dict[str, list[str]] = defaultdict(list)
        for topic in TOPIC_IDS:
            for row in ranking_rows[topic][arm]:
                if row.get("source") == "facet_basket":
                    selected[str(row.get("generating_facet"))].append(str(row["document_id"]))
        for facet in sorted(selected):
            topic = next(topic for topic in TOPIC_IDS if facet.startswith(f"{topic}-"))
            values = {name: 0 for name in compiled}
            for document in selected[facet]:
                text = texts.get((topic, facet, document), "")
                for name, pattern in compiled.items():
                    values[name] += int(bool(pattern.search(text)))
            noise_rows.append({
                "arm": arm,
                "facet_id": facet,
                "selected_count": len(selected[facet]),
                **values,
            })

    facet_yield = {
        arm: arms[arm].get("facet_yield", {}) for arm in ("FACET-2B", "TETHERED-2B")
    }
    facet_changes: list[dict[str, object]] = []
    facets = sorted(set(facet_yield["FACET-2B"]) | set(facet_yield["TETHERED-2B"]))  # type: ignore[arg-type]
    for facet in facets:
        left = facet_yield["FACET-2B"].get(facet, {})  # type: ignore[union-attr]
        right = facet_yield["TETHERED-2B"].get(facet, {})  # type: ignore[union-attr]
        left_count = int(left.get("relevant_count", 0)) if isinstance(left, Mapping) else 0
        right_count = int(right.get("relevant_count", 0)) if isinstance(right, Mapping) else 0
        classification = (
            "zero" if left_count == right_count == 0
            else "rose" if right_count > left_count
            else "fell" if right_count < left_count
            else "unchanged"
        )
        facet_changes.append({
            "facet_id": facet,
            "facet_only_relevant_count": left_count,
            "tethered_relevant_count": right_count,
            "delta": right_count - left_count,
            "classification": classification,
        })

    below: list[dict[str, object]] = []
    for arm in ("FACET-2B", "TETHERED-2B"):
        for topic in TOPIC_IDS:
            for row in ranking_rows[topic][arm]:
                document = str(row["document_id"])
                grade = int(qrels[topic].get(document, 0))
                final_rank = int(row["rank"])
                if grade < 2 or final_rank <= 500:
                    continue
                reason = row.get("facet_selection_outcome")
                if reason not in {
                    "not_in_facet_candidate_pool",
                    "facet_quota_exhausted",
                    "facet_basket_capacity_exhausted",
                }:
                    raise ValueError("Task 3 below-500 rank trace is invalid")
                below.append({
                    "arm": arm,
                    "topic_id": topic,
                    "document_id": document,
                    "qrels_grade": grade,
                    "final_rank": final_rank,
                    "best_facet": row.get("best_candidate_facet"),
                    "best_facet_percentile": row.get("best_candidate_percentile"),
                    "prior_bm25_rank": row.get("best_candidate_bm25_rank"),
                    "reason": reason,
                })

    topic_summary = summary.get("topic_summary")
    if not isinstance(topic_summary, Mapping):
        raise ValueError("Task 3 topic summary is missing from diagnostics")
    pressure: list[dict[str, object]] = []
    for topic in TOPIC_IDS:
        record = topic_summary.get(topic)
        duplicates = record.get("duplicate_skip_totals") if isinstance(record, Mapping) else None
        shortages = record.get("facet_shortage_counts") if isinstance(record, Mapping) else None
        if not isinstance(duplicates, Mapping) or not isinstance(shortages, Mapping):
            raise ValueError("Task 3 duplicate/shortage totals are missing")
        for arm in ("FACET-2B", "TETHERED-2B"):
            arm_duplicates = duplicates.get(arm)
            arm_shortages = shortages.get(arm)
            if not isinstance(arm_duplicates, Mapping) or not isinstance(arm_shortages, Mapping):
                raise ValueError("Task 3 arm duplicate/shortage totals are missing")
            pressure.append({
                "topic_id": topic,
                "arm": arm,
                "duplicate_skip_totals": dict(sorted((str(key), int(value)) for key, value in arm_duplicates.items())),
                "duplicate_skip_total": sum(int(value) for value in arm_duplicates.values()),
                "shortage_counts": dict(sorted((str(key), int(value)) for key, value in arm_shortages.items())),
                "shortage_total": sum(int(value) for value in arm_shortages.values()),
            })

    try:
        preflight = json.loads(contents["tethered_preflight"])
        receipt = json.loads(contents["tethered_scoring_receipt"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Task 1/2 telemetry sources are invalid") from exc
    if not isinstance(preflight, Mapping) or not isinstance(receipt, Mapping):
        raise ValueError("Task 1/2 telemetry sources must be objects")
    preflight_summary = preflight.get("summary")
    runtime = preflight.get("runtime_evidence")
    if not isinstance(preflight_summary, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("Task 1 telemetry summary/runtime evidence is missing")
    try:
        cache_hits = int(receipt["cache_hit_count"])
        forward_pairs = int(receipt["unique_forward_pair_count"])
        available_unique_pairs = preflight_summary.get("unique_pair_count")
        unique_pairs = (
            int(available_unique_pairs)
            if available_unique_pairs is not None
            else cache_hits + forward_pairs
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Task 1/2 scoring pair telemetry is invalid") from exc
    if (
        min(cache_hits, forward_pairs, unique_pairs) < 0
        or cache_hits + forward_pairs != unique_pairs
        or (
            receipt.get("cache_reuse_pair_count") is not None
            and int(receipt["cache_reuse_pair_count"]) != cache_hits
        )
    ):
        raise ValueError("Task 1/2 unique scoring pair accounting differs")
    telemetry = {
        "preflight_source_sha256": hashes["tethered_preflight"],
        "scoring_receipt_source_sha256": hashes["tethered_scoring_receipt"],
        "model": receipt.get("model"),
        "model_revision": receipt.get("model_revision"),
        "query_document_pair_count": preflight_summary.get("query_document_pair_count"),
        "planned_window_count": receipt.get("planned_window_count"),
        "completed_window_count": receipt.get("completed_window_count"),
        "document_score_count": receipt.get("document_score_count"),
        "cache_hit_count": cache_hits,
        "cache_miss_count": forward_pairs,
        "unique_forward_pair_count": forward_pairs,
        "unique_scoring_pair_count": unique_pairs,
        "elapsed_seconds": receipt.get("elapsed_seconds"),
        "projected_inference_seconds": runtime.get("projected_inference_seconds"),
        "peak_device_memory_bytes": receipt.get("peak_device_memory_bytes"),
        "peak_host_memory_bytes": receipt.get("peak_host_memory_bytes"),
    }
    return {
        "noise_pattern_definitions": dict(NOISE_PATTERN_REGEX),
        "noise_pattern_counts": noise_rows,
        "facet_yield_changes": facet_changes,
        "relevant_below_500": below,
        "duplicate_and_quota_pressure": pressure,
        "scoring_telemetry": telemetry,
    }


def evaluate(
    freeze_dir: Path, prior_evaluation: Path, output: Path
) -> dict[str, object]:
    """Evaluate from the exact authenticated prior evaluation directory."""

    freeze_dir, prior_evaluation, output = map(
        Path, (freeze_dir, prior_evaluation, output)
    )
    task3 = _task3_snapshot(freeze_dir)
    if output.exists():
        raise FileExistsError(f"create-only evaluation output already exists: {output}")
    if (
        prior_evaluation.name != "evaluation_v1"
        or prior_evaluation.is_symlink()
        or not prior_evaluation.is_dir()
    ):
        raise ValueError("prior evaluation must be the exact evaluation_v1 directory")
    entries = list(prior_evaluation.iterdir())
    if (
        {path.name for path in entries} != PRIOR_EVALUATION_FILES
        or not all(path.is_file() and not path.is_symlink() for path in entries)
    ):
        raise ValueError("prior evaluation has missing or extra files")

    prior_freeze = prior_evaluation.parent / "freeze_v1"
    if prior_freeze.is_symlink() or not prior_freeze.is_dir():
        raise ValueError("prior evaluation requires sibling freeze_v1")
    prior_seal = verify_prior_seal(prior_freeze)
    if not isinstance(prior_seal, Mapping):
        raise ValueError("verified prior seal is invalid")
    prior_seal_path = prior_freeze / "SEALED.json"
    try:
        prior_seal_bytes = prior_seal_path.read_bytes()
    except OSError as exc:
        raise ValueError("verified prior seal bytes are unreadable") from exc
    prior_root = prior_seal.get("root_sha256")
    try:
        reread_prior_seal = json.loads(prior_seal_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("verified prior seal bytes are invalid") from exc
    if (
        reread_prior_seal != prior_seal
        or not isinstance(prior_root, str)
        or len(prior_root) != 64
    ):
        raise ValueError("verified prior seal root is invalid")

    projection = prior_evaluation / "qrels_projection.jsonl"
    receipt_path = prior_evaluation / "qrels_access_receipt.json"
    prior_metrics_path = prior_evaluation / "metrics.json"
    prior_decision_path = prior_evaluation / "decision.json"
    prior_summary_path = prior_evaluation / "summary.json"
    receipt, receipt_bytes = _read_object_bytes(receipt_path, "qrels access receipt")
    prior_metrics, prior_metrics_bytes = _read_object_bytes(prior_metrics_path, "prior metrics")
    _prior_decision, prior_decision_bytes = _read_object_bytes(
        prior_decision_path, "prior decision"
    )
    prior_summary, prior_summary_bytes = _read_object_bytes(prior_summary_path, "prior summary")
    if (
        prior_summary.get("schema_version") != PRIOR_SCHEMA_VERSION
        or prior_summary.get("status") != "complete"
        or prior_summary.get("qrels_opened") is not True
        or prior_summary.get("topic_ids") != list(TOPIC_IDS)
        or prior_summary.get("metrics_sha256") != _sha256(prior_metrics_bytes)
    ):
        raise ValueError("prior metrics SHA-256 differs from prior summary")
    if prior_summary.get("decision_sha256") != _sha256(prior_decision_bytes):
        raise ValueError("prior decision SHA-256 differs from prior summary")
    if (
        set(receipt) != PRIOR_RECEIPT_FIELDS
        or receipt.get("schema_version") != PRIOR_SCHEMA_VERSION
        or receipt.get("status") != "qrels_access_boundary_crossed"
        or receipt.get("qrels_opened") is not True
        or receipt.get("upstream_mutation_forbidden") is not True
        or receipt.get("topic_ids") != list(TOPIC_IDS)
        or not isinstance(receipt.get("qrels_source_name"), str)
        or not receipt.get("qrels_source_name")
        or not isinstance(receipt.get("qrels_projection_rows"), int)
        or isinstance(receipt.get("qrels_projection_rows"), bool)
        or int(receipt.get("qrels_projection_rows", 0)) <= 0
        or not isinstance(receipt.get("evaluator_code_sha256"), str)
        or len(str(receipt.get("evaluator_code_sha256"))) != 64
    ):
        raise ValueError("qrels access receipt contract differs")
    if (
        receipt.get("seal_sha256") != _sha256(prior_seal_bytes)
        or receipt.get("seal_root_sha256") != prior_root
    ):
        raise ValueError("qrels access receipt differs from verified prior seal")
    try:
        projection_bytes = projection.read_bytes()
    except OSError as exc:
        raise ValueError("qrels projection is unreadable") from exc
    projection_sha256 = _sha256(projection_bytes)
    if receipt.get("qrels_projection_sha256") != projection_sha256:
        raise ValueError("qrels projection SHA-256 differs from sealed receipt")
    if receipt.get("qrels_projection_rows") != len(projection_bytes.splitlines()):
        raise ValueError("qrels projection row count differs from sealed receipt")
    qrels = _parse_projection(projection_bytes)

    prior_rrf, prior_facet_candidates, original_at_1000, prior_ranking_bytes = (
        _load_prior_rrf(prior_freeze)
    )
    novel = derive_novel_set(
        qrels,
        prior_facet_candidates,
        original_at_1000,
        expected_total=NOVEL_RELEVANT_TOTAL,
    )
    recomputed_rrf = evaluate_arm(
        prior_rrf, qrels, novel, depths=(100, 500, 1000)
    )
    if prior_summary.get("novel_relevant_count") != sum(map(len, novel.values())):
        raise ValueError("prior summary differs from recomputed novel set")
    _require_prior_metrics_equal_recomputation(
        prior_metrics, recomputed_rrf, novel
    )
    _require_historical_identity(
        {
            "qrels_access_receipt.json": receipt_bytes,
            "qrels_projection.jsonl": projection_bytes,
            "metrics.json": prior_metrics_bytes,
            "decision.json": prior_decision_bytes,
            "summary.json": prior_summary_bytes,
        }
    )

    rankings, ranking_rows, freeze_bindings, task3_summary, freeze_binding_bytes, freeze_ranking_bytes = _load_frozen_rankings(
        task3
    )
    for topic in TOPIC_IDS:
        for arm in ("FACET-2B", "TETHERED-2B"):
            if set(rankings[arm][topic]) != set(prior_rrf[topic]):
                raise ValueError("Task 3 arm population differs from verified prior RRF")
            validate_protected_head(rrf=prior_rrf[topic], arm=rankings[arm][topic])

    row_views = {
        arm: {topic: ranking_rows[topic][arm] for topic in TOPIC_IDS}
        for arm in ("FACET-2B", "TETHERED-2B")
    }
    arms = {
        "RRF": recomputed_rrf,
        "FACET-2B": evaluate_arm(
            rankings["FACET-2B"], qrels, novel,
            ranking_rows=row_views["FACET-2B"], depths=(100, 500, 1000),
        ),
        "TETHERED-2B": evaluate_arm(
            rankings["TETHERED-2B"], qrels, novel,
            ranking_rows=row_views["TETHERED-2B"], depths=(100, 500, 1000),
        ),
    }
    topic_summary = task3_summary.get("topic_summary")
    if not isinstance(topic_summary, Mapping):
        raise ValueError("Task 3 topic summary is missing")
    documented_shortage = False
    for topic in TOPIC_IDS:
        record = topic_summary.get(topic)
        shortages = record.get("facet_shortage_counts") if isinstance(record, Mapping) else None
        if not isinstance(shortages, Mapping):
            raise ValueError("Task 3 complete shortage evidence is missing")
        for arm in ("FACET-2B", "TETHERED-2B"):
            arm_shortages = shortages.get(arm)
            if not isinstance(arm_shortages, Mapping):
                raise ValueError("Task 3 arm shortage evidence is missing")
            documented_shortage = documented_shortage or any(
                int(value) > 0 for value in arm_shortages.values()
            )
    aggregate = {arm: result["aggregate"] for arm, result in arms.items()}
    per_topic = {topic: {} for topic in TOPIC_IDS}
    for topic in TOPIC_IDS:
        for arm, result in arms.items():
            topics = result["per_topic"]
            if not isinstance(topics, Mapping):
                raise ValueError("arm per-topic metrics are invalid")
            per_topic[topic][arm] = topics[topic]
    decision = decide(
        {
            "top100_identity": True,
            "aggregate": aggregate,
            "per_topic": per_topic,
            "documented_basket_shortage": documented_shortage,
        }
    )
    novel_relevant_count = sum(map(len, novel.values()))
    decision["novel_relevant_count"] = novel_relevant_count
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "topic_ids": list(TOPIC_IDS),
        "novel_relevant_count": novel_relevant_count,
        "arms": arms,
    }
    extended = _extended_diagnostics(task3, qrels, arms, ranking_rows)
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "topic_ids": list(TOPIC_IDS),
        "per_topic_deltas": _per_topic_deltas(arms),
        "basket_contributions": {
            arm: arms[arm]["basket_contributions"] for arm in ("FACET-2B", "TETHERED-2B")
        },
        "facet_yield": {
            arm: arms[arm]["facet_yield"] for arm in ("FACET-2B", "TETHERED-2B")
        },
        "novel_relevant_ids": {topic: sorted(novel[topic]) for topic in TOPIC_IDS},
        "documented_basket_shortage": documented_shortage,
        "representatives": build_representatives(freeze_dir, qrels, snapshot=task3),
        **extended,
    }
    seal_path = freeze_dir / "SEALED.json"
    seal_bytes, seal = task3.seal_bytes, task3.seal
    input_bindings = {
        "schema_version": SCHEMA_VERSION,
        "task3_root_sha256": seal["root_sha256"],
        "task3_seal": _binding(seal_path, seal_bytes),
        "task3_input_bindings": _binding(freeze_dir / "input_bindings.json", freeze_binding_bytes),
        "task3_rankings": _binding(
            freeze_dir / "rankings.jsonl", freeze_ranking_bytes
        ),
        "task3_producer_sources": freeze_bindings.get("inputs"),
        "prior_freeze_root_sha256": prior_root,
        "prior_freeze_seal": _binding(prior_seal_path, prior_seal_bytes),
        "prior_freeze_rankings": _binding(
            prior_freeze / "rankings.jsonl", prior_ranking_bytes
        ),
        "qrels_projection": _binding(projection, projection_bytes),
        "qrels_access_receipt": _binding(receipt_path, receipt_bytes),
        "prior_metrics": _binding(prior_metrics_path, prior_metrics_bytes),
        "prior_decision": _binding(prior_decision_path, prior_decision_bytes),
        "prior_summary": _binding(prior_summary_path, prior_summary_bytes),
        "historical_integrity_anchor": HISTORICAL_PRIOR_EVALUATION_IDENTITY,
        "historical_integrity_only": True,
        "blind_generalization_evidence": False,
        "original_qrels_opened": False,
    }
    payloads = {
        "metrics.json": _pretty_bytes(metrics),
        "diagnostics.json": _pretty_bytes(diagnostics),
        "decision.json": _pretty_bytes(decision),
        "input_bindings.json": _pretty_bytes(input_bindings),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "label": decision["label"],
        "topic_ids": list(TOPIC_IDS),
        "novel_relevant_count": novel_relevant_count,
        "post_qrels_diagnostic": True,
        "historical_integrity_only": True,
        "blind_generalization_evidence": False,
        "production_validation": False,
        "no_new_retrieval": True,
        "original_qrels_opened": False,
        "artifacts": {
            name: {"bytes": len(content), "sha256": _sha256(content)}
            for name, content in payloads.items()
        },
    }
    payloads["summary.json"] = _pretty_bytes(summary)
    output.mkdir(parents=True)
    for name, content in payloads.items():
        _exclusive_bytes(output / name, content)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluate", nargs="?")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--prior-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate(args.freeze, args.prior_evaluation, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
