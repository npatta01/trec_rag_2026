"""Build the canonical portable-report source for the facet-control pilot.

This module does not render HTML.  It converts already frozen and evaluated
JSON artifacts into the native Data Analytics artifact contract that the
packaged portable report builder consumes in the following task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .facet_retrieval_control_experiment import (
    EXPECTED_ALTERNATIVE_NAMES,
    retrieval_repair_decision,
    selected_ranking_references,
)
from .facet_retrieval_control_freeze import _BASE_ARM_QUERIES
from .facet_retrieval_control_manifest import (
    PROTECTED_TOPIC_IDS,
    _ARM_SPEC,
    _QUERY_SPEC,
)


TITLE = "Facet Retrieval Controls: Four-Topic Pilot"
STREAM_IDS = ("200/f07a", "225/f02", "225/f04", "707/f02")
ARM_IDS = ("B0", "W0", "W1", "W2")
TOPIC_IDS = ("200", "225", "707", "897")
SYSTEM_IDS = ("O", "F0", "R1", "R2")
SYSTEM_METRICS = (
    "ndcg@10",
    "graded_recall@100",
    "recall@100",
    "precision@10",
    "relevant_count@10",
    "judged_rate@10",
    "judged_rate@100",
)
_CANDIDATE_MANIFEST_FIELDS = {
    "schema_version",
    "candidate_schema_version",
    "candidate_file",
    "candidate_file_sha256",
    "stream_count",
    "row_count",
    "streams",
}
_CANDIDATE_STREAM_FIELDS = {
    "topic_id",
    "stream_id",
    "arm_id",
    "role",
    "source_kind",
    "query_sha256",
    "request_sha256",
    "response_sha256",
    "source_candidates_sha256",
    "expected_depth",
    "row_count",
    "stream_rows_sha256",
}


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_manifest(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    if manifest.get("schema_version") != "facet-retrieval-control-manifest-v1":
        raise ValueError("manifest schema differs")
    if tuple(manifest.get("protected_topic_ids", ())) != PROTECTED_TOPIC_IDS:
        raise ValueError("manifest protected-topic boundary differs")
    streams = [_object(value, "manifest stream") for value in _list(manifest.get("streams"), "manifest streams")]
    indexed = {
        (str(stream.get("topic_id")), str(stream.get("stream_id"))): stream
        for stream in streams
    }
    if set(indexed) != set(_QUERY_SPEC) or len(streams) != len(_QUERY_SPEC):
        raise ValueError("manifest stream boundary differs")
    expected_arms = [
        {"arm_id": arm_id, "k1": k1, "b": b, "external": external}
        for arm_id, k1, b, external in _ARM_SPEC
    ]
    for boundary, (baseline, reweighted) in _QUERY_SPEC.items():
        stream = indexed[boundary]
        if (
            stream.get("baseline_query") != baseline
            or stream.get("reweighted_query") != reweighted
            or stream.get("arms") != expected_arms
        ):
            raise ValueError(f"manifest query or arm contract differs for {boundary[0]}/{boundary[1]}")
    return [indexed[(topic_id, stream_id)] for topic_id, stream_id in _QUERY_SPEC]


def _narratives() -> dict[str, str]:
    result = {
        topic_id: query
        for topic_id, variant_name, query in _BASE_ARM_QUERIES
        if variant_name == "prompt_lab_v1:original"
    }
    if set(result) != set(TOPIC_IDS):
        raise ValueError("tracked original narrative contract differs")
    return result


def _validate_candidate_snapshot(
    control_freeze: Mapping[str, object],
    candidate_streams: Mapping[str, object],
    manifest_streams: Sequence[Mapping[str, object]],
) -> None:
    if (
        control_freeze.get("schema_version") != "facet-control-ranking-freeze-v2"
        or control_freeze.get("status") != "frozen_before_qrels"
    ):
        raise ValueError("control freeze must be the frozen v2 snapshot")
    rankings = _object(control_freeze.get("rankings"), "control freeze rankings")
    if set(rankings) != set(EXPECTED_ALTERNATIVE_NAMES):
        raise ValueError("control freeze ranking alternatives differ")
    if set(candidate_streams) != _CANDIDATE_MANIFEST_FIELDS:
        raise ValueError("candidate snapshot manifest fields differ")
    if (
        candidate_streams.get("schema_version") != "facet-control-candidate-streams-v1"
        or candidate_streams.get("candidate_schema_version")
        != "facet-control-candidate-row-v1"
        or candidate_streams.get("candidate_file") != "candidates.jsonl"
        or candidate_streams.get("stream_count") != 20
        or candidate_streams.get("row_count") != 2000
    ):
        raise ValueError("candidate snapshot manifest counts differ")
    entries = [
        _object(value, "candidate snapshot stream")
        for value in _list(candidate_streams.get("streams"), "candidate snapshot streams")
    ]
    if len(entries) != 20 or any(set(entry) != _CANDIDATE_STREAM_FIELDS for entry in entries):
        raise ValueError("candidate snapshot must contain the exact 20 stream records")
    indexed = {
        (str(entry["topic_id"]), str(entry["stream_id"]), str(entry["arm_id"])): entry
        for entry in entries
    }
    expected = {
        *((topic_id, "original", "O") for topic_id in TOPIC_IDS),
        *(
            (stream_id.split("/", 1)[0], stream_id.split("/", 1)[1], arm_id)
            for stream_id in STREAM_IDS
            for arm_id in ARM_IDS
        ),
    }
    if set(indexed) != expected or len(indexed) != 20:
        raise ValueError("candidate snapshot stream namespace differs")
    bindings = _object(control_freeze.get("bindings"), "control freeze bindings")
    expected_manifest_hash = hashlib.sha256(_canonical_json(candidate_streams)).hexdigest()
    if bindings.get("candidate_streams_sha256") != expected_manifest_hash:
        raise ValueError("candidate snapshot manifest hash differs from freeze")
    frozen_stream_hashes = _object(
        bindings.get("candidate_stream_rows_sha256"),
        "control freeze candidate stream hashes",
    )
    actual_stream_hashes = {
        f"{key[0]}/{key[1]}/{key[2]}": entry["stream_rows_sha256"]
        for key, entry in indexed.items()
    }
    if frozen_stream_hashes != actual_stream_hashes:
        raise ValueError("candidate snapshot stream hashes differ from freeze")
    narratives = _narratives()
    for topic_id, narrative in narratives.items():
        entry = indexed[(topic_id, "original", "O")]
        if (
            entry.get("role") != "original"
            or entry.get("source_kind") != "prior_base"
            or entry.get("expected_depth") != 100
            or entry.get("row_count") != 100
            or entry.get("query_sha256")
            != hashlib.sha256(narrative.encode("utf-8")).hexdigest()
        ):
            raise ValueError(f"narrative differs from verified snapshot for topic {topic_id}")
    streams_by_key = {
        (str(stream["topic_id"]), str(stream["stream_id"])): stream
        for stream in manifest_streams
    }
    for stream_id in STREAM_IDS:
        topic_id, facet_id = stream_id.split("/", 1)
        stream = streams_by_key[(topic_id, facet_id)]
        for arm_id in ARM_IDS:
            query = (
                str(stream["baseline_query"])
                if arm_id == "B0"
                else str(stream["reweighted_query"])
            )
            if indexed[(topic_id, facet_id, arm_id)].get("query_sha256") != hashlib.sha256(
                query.encode("utf-8")
            ).hexdigest():
                raise ValueError(f"facet query differs from verified snapshot for {stream_id}/{arm_id}")


def _validate_inspection(inspection: Mapping[str, object]) -> None:
    expected = {f"{stream_id}/{arm_id}" for stream_id in STREAM_IDS for arm_id in ARM_IDS}
    if set(inspection) != expected:
        raise ValueError("inspection differs from the exact 16-arm boundary")
    for key in sorted(expected):
        row = _object(inspection[key], f"inspection {key}")
        stream_id, _arm_id = key.rsplit("/", 1)
        topic_id, facet_id = stream_id.split("/", 1)
        if row.get("topic_id") != topic_id or row.get("stream_id") != facet_id:
            raise ValueError(f"inspection identity differs for {key}")
        for field, maximum in (
            ("domain_drift_top5_count", 5),
            ("domain_drift_top10_count", 10),
            ("content_quality_top5_count", 5),
            ("content_quality_top10_count", 10),
        ):
            value = _integer(row.get(field), f"inspection {key}.{field}")
            if not 0 <= value <= maximum:
                raise ValueError(f"inspection count differs for {key}.{field}")
        docids = _list(row.get("top_docids"), f"inspection {key}.top_docids")
        snippets = _list(row.get("top_snippets"), f"inspection {key}.top_snippets")
        if (
            len(docids) != 10
            or len(snippets) != 10
            or len(set(docids)) != 10
            or not all(isinstance(value, str) and value for value in docids)
            or not all(isinstance(value, str) and len(value) <= 400 for value in snippets)
        ):
            raise ValueError(f"inspection representative results differ for {key}")


def _validate_stream_evaluation(
    stream_evaluation: Mapping[str, object], inspection: Mapping[str, object]
) -> Mapping[str, object]:
    if stream_evaluation.get("schema_version") != "facet-control-stream-evaluation-v1":
        raise ValueError("stream evaluation schema differs")
    streams = _object(stream_evaluation.get("streams"), "stream evaluation streams")
    if set(streams) != set(STREAM_IDS):
        raise ValueError("stream evaluation boundary differs")
    for stream_id in STREAM_IDS:
        arms = _object(_object(streams[stream_id], f"stream evaluation {stream_id}").get("arms"), f"stream evaluation {stream_id}.arms")
        if set(arms) != set(ARM_IDS):
            raise ValueError(f"stream evaluation arms differ for {stream_id}")
        for arm_id in ARM_IDS:
            record = _object(arms[arm_id], f"stream evaluation {stream_id}/{arm_id}")
            metrics = _object(record.get("metrics"), f"stream metrics {stream_id}/{arm_id}")
            embedded = _object(record.get("inspection"), f"stream inspection {stream_id}/{arm_id}")
            if embedded != inspection[f"{stream_id}/{arm_id}"]:
                raise ValueError(f"stream evaluation inspection differs for {stream_id}/{arm_id}")
            if metrics.get("arm_id") != arm_id:
                raise ValueError(f"stream evaluation arm identity differs for {stream_id}/{arm_id}")
            for field in (
                "overlap_with_original_top100",
                "relevant_at_10",
                "relevant_at_100",
                "unique_relevant_contribution",
                "unique_graded_gain",
                "domain_drift_top10_count",
                "content_quality_top10_count",
                "relevant_denominator",
                "graded_recall_denominator",
            ):
                if _integer(metrics.get(field), f"stream metrics {stream_id}/{arm_id}.{field}") < 0:
                    raise ValueError(f"stream metric must be non-negative for {stream_id}/{arm_id}.{field}")
            for field in (
                "graded_recall_at_100",
                "ndcg_at_10",
                "recall_at_100",
                "precision_at_10",
                "judged_rate_at_10",
                "judged_rate_at_100",
            ):
                _number(metrics.get(field), f"stream metrics {stream_id}/{arm_id}.{field}")
            if (
                metrics.get("domain_drift_top10_count")
                != embedded.get("domain_drift_top10_count")
                or metrics.get("content_quality_top10_count")
                != embedded.get("content_quality_top10_count")
                or metrics.get("coherence_failed")
                is not embedded.get("coherence_failed")
            ):
                raise ValueError(f"stream metrics and inspection differ for {stream_id}/{arm_id}")
    return streams


def _validate_selection(selection: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if selection.get("schema_version") != "facet-control-selection-v1":
        raise ValueError("selection schema differs")
    selected = _object(selection.get("selected"), "selection selected arms")
    if set(selected) != set(STREAM_IDS) or any(selected[key] not in ARM_IDS for key in selected):
        raise ValueError("selection differs from the exact four-stream boundary")
    references = _object(selection.get("selected_rankings"), "selection ranking references")
    if references != selected_ranking_references({key: str(value) for key, value in selected.items()}):
        raise ValueError("selection ranking references are inconsistent")
    if selection.get("frozen_alternative_count") != 25:
        raise ValueError("selection frozen alternative count differs")
    return selected, references


def _validate_system_evaluation(system_evaluation: Mapping[str, object]) -> Mapping[str, object]:
    if system_evaluation.get("schema_version") != "facet-control-system-evaluation-v1":
        raise ValueError("system evaluation schema differs")
    systems = _object(system_evaluation.get("systems"), "system evaluation systems")
    if set(systems) != set(SYSTEM_IDS):
        raise ValueError("system evaluation must contain exact O/F0/R1/R2 systems")
    for system_id in SYSTEM_IDS:
        system = _object(systems[system_id], f"system evaluation {system_id}")
        metrics = _object(system.get("metrics"), f"system evaluation {system_id}.metrics")
        per_topic = _object(system.get("per_topic"), f"system evaluation {system_id}.per_topic")
        if set(per_topic) != set(TOPIC_IDS):
            raise ValueError(f"system evaluation topic boundary differs for {system_id}")
        for field in SYSTEM_METRICS:
            _number(metrics.get(field), f"system evaluation {system_id}.metrics.{field}")
        for topic_id in TOPIC_IDS:
            topic = _object(per_topic[topic_id], f"system evaluation {system_id}.{topic_id}")
            for field in SYSTEM_METRICS:
                _number(topic.get(field), f"system evaluation {system_id}.{topic_id}.{field}")
    deltas = _object(
        system_evaluation.get("per_topic_ndcg_delta_vs_r1"),
        "system evaluation per-topic deltas",
    )
    if set(deltas) != set(TOPIC_IDS):
        raise ValueError("system evaluation per-topic deltas differ")
    for topic_id in TOPIC_IDS:
        expected = (
            float(_object(_object(systems["R2"], "R2").get("per_topic"), "R2 per-topic")[topic_id]["ndcg@10"])
            - float(_object(_object(systems["R1"], "R1").get("per_topic"), "R1 per-topic")[topic_id]["ndcg@10"])
        )
        if not math.isclose(_number(deltas[topic_id], f"system delta {topic_id}"), expected, abs_tol=1e-12):
            raise ValueError(f"system evaluation per-topic delta is inconsistent for {topic_id}")
    return systems


def _validate_decision(
    decision: Mapping[str, object],
    systems: Mapping[str, object],
    stream_rows: Mapping[str, object],
    selected: Mapping[str, object],
) -> None:
    if decision.get("schema_version") != "facet-control-decision-v1":
        raise ValueError("decision schema differs")
    if decision.get("reranker_gate_opened") is not False:
        raise ValueError("decision must not open a reranker gate")
    evidence = _object(decision.get("evidence"), "decision evidence")
    r1 = _object(_object(systems["R1"], "R1").get("metrics"), "R1 metrics")
    r2 = _object(_object(systems["R2"], "R2").get("metrics"), "R2 metrics")
    per_topic = _object(evidence.get("per_topic_ndcg_deltas"), "decision per-topic deltas")
    if set(per_topic) != set(TOPIC_IDS):
        raise ValueError("decision per-topic delta boundary differs")
    expected_values = {
        "r2_graded_recall_at_100": r2["graded_recall@100"],
        "r1_graded_recall_at_100": r1["graded_recall@100"],
        "r2_ndcg_at_10": r2["ndcg@10"],
        "r1_ndcg_at_10": r1["ndcg@10"],
    }
    for field, expected in expected_values.items():
        if not math.isclose(_number(evidence.get(field), f"decision {field}"), float(expected), abs_tol=1e-12):
            raise ValueError(f"decision evidence is inconsistent for {field}")
    selected_noise = 0
    b0_noise = 0
    for stream_id in STREAM_IDS:
        arms = _object(_object(stream_rows[stream_id], stream_id).get("arms"), f"{stream_id}.arms")
        chosen = _object(_object(arms[str(selected[stream_id])], "selected arm").get("metrics"), "selected metrics")
        baseline = _object(_object(arms["B0"], "B0 arm").get("metrics"), "B0 metrics")
        selected_noise += int(chosen["domain_drift_top10_count"]) + int(chosen["content_quality_top10_count"])
        b0_noise += int(baseline["domain_drift_top10_count"]) + int(baseline["content_quality_top10_count"])
    if evidence.get("selected_top10_noise") != selected_noise or evidence.get("b0_top10_noise") != b0_noise:
        raise ValueError("decision noise evidence is inconsistent")
    expected_decision = retrieval_repair_decision(
        r2_graded_recall=float(r2["graded_recall@100"]),
        r1_graded_recall=float(r1["graded_recall@100"]),
        r2_ndcg=float(r2["ndcg@10"]),
        r1_ndcg=float(r1["ndcg@10"]),
        per_topic_ndcg_deltas=tuple(float(per_topic[topic_id]) for topic_id in TOPIC_IDS),
        selected_noise=selected_noise,
        b0_noise=b0_noise,
    )
    if decision.get("decision") != expected_decision:
        raise ValueError("decision classification is inconsistent")


def _validate_provenance(*payloads: Mapping[str, object]) -> None:
    provenances = [_object(payload.get("provenance"), "evaluation provenance") for payload in payloads]
    if any(provenance != provenances[0] for provenance in provenances[1:]):
        raise ValueError("evaluation provenance is inconsistent")


def _source(
    source_id: str,
    label: str,
    path: str,
    paths: Sequence[str],
    description: str,
    source_hash: str,
) -> dict[str, object]:
    quoted = ", ".join(f"'{value}'" for value in paths)
    sql = f"SELECT * FROM read_json_auto([{quoted}], union_by_name = true);"
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "id": f"sha256:{_sha256(source_hash, f'{source_id} source hash')}",
            "language": "sql",
            "description": description,
            "sql": sql,
            "tables_used": list(paths),
            "filters": [
                "Only the four registered facet streams and four evaluated pilot topics",
                "Protected topics and raw relevance-judgment rows excluded from report data",
                "Canonical source shape checked before validate_artifact and portable packaging",
            ],
        },
    }


def _payload_hash(payload: Mapping[str, object]) -> str:
    value = payload.get("artifact_sha256")
    if isinstance(value, str) and len(value) == 64:
        return _sha256(value, "evaluation artifact hash")
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sources(
    control_freeze: Mapping[str, object],
    stream_evaluation: Mapping[str, object],
    system_evaluation: Mapping[str, object],
    decision: Mapping[str, object],
) -> list[dict[str, object]]:
    manifest_path = "reports/experiments/facet_retrieval_control_pilot_v1/manifest.json"
    freeze_path = "outputs/rag25_facet_retrieval_control_v1/freeze_v1/freeze.json"
    candidate_path = "outputs/rag25_facet_retrieval_control_v1/freeze_v1/candidate_streams.json"
    inspection_path = "outputs/rag25_facet_retrieval_control_v1/freeze_v1/inspection.json"
    stream_path = "outputs/rag25_facet_retrieval_control_v1/evaluation_v1/stream_evaluation.json"
    selection_path = "outputs/rag25_facet_retrieval_control_v1/evaluation_v1/selection.json"
    evaluation_path = "outputs/rag25_facet_retrieval_control_v1/evaluation_v1/evaluation.json"
    decision_path = "outputs/rag25_facet_retrieval_control_v1/evaluation_v1/decision.json"
    return [
        _source(
            "decision_evidence",
            "Frozen retrieval-repair decision",
            decision_path,
            (decision_path,),
            "Loads the saved mechanical retrieval-repair decision and gate state.",
            _payload_hash(decision),
        ),
        _source(
            "query_evidence",
            "Frozen narratives, queries, and selected arms",
            manifest_path,
            (manifest_path, freeze_path, candidate_path, selection_path),
            "Loads registered control queries, snapshot-bound narrative identities, and selected arms.",
            str(_object(control_freeze["bindings"], "freeze bindings")["candidate_streams_sha256"]),
        ),
        _source(
            "noise_evidence",
            "Frozen qrels-free noise inspection",
            inspection_path,
            (inspection_path, stream_path, selection_path),
            "Loads bounded top-five/top-ten drift and content-quality diagnostics plus selected arms. Chart map: grouped categorical bar with stream-arm categories and two same-unit warning series.",
            str(control_freeze["inspection_sha256"]),
        ),
        _source(
            "facet_evidence",
            "Selected facet contribution evaluation",
            stream_path,
            (stream_path, selection_path),
            "Loads saved selected-arm marginal relevant and graded contribution metrics. Chart map: single-series categorical bar across the four selected facet streams.",
            _payload_hash(stream_evaluation),
        ),
        _source(
            "system_evidence",
            "Frozen O/F0/R1/R2 system evaluation",
            evaluation_path,
            (evaluation_path, selection_path, decision_path),
            "Loads exact aggregate and per-topic system metrics, ranking references, and decision evidence. Visual map: spacious exact lookup table because audit precision is the purpose.",
            _payload_hash(system_evaluation),
        ),
        _source(
            "pilot_contract",
            "Facet-control pilot contract and limitations",
            manifest_path,
            (manifest_path, freeze_path, decision_path),
            "Loads the frozen pilot boundary, available controls, and gate outcome used for caveats.",
            str(control_freeze["freeze_sha256"]),
        ),
    ]


def _table(
    table_id: str,
    title: str,
    subtitle: str,
    dataset: str,
    source_id: str,
    columns: Sequence[tuple[str, str, str]],
    sort_field: str,
) -> dict[str, object]:
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "sourceId": source_id,
        "density": "spacious",
        "layout": "full",
        "defaultSort": {"field": sort_field, "direction": "asc"},
        "columns": [
            {"field": field, "label": label, "type": kind}
            for field, label, kind in columns
        ],
    }


def _artifact_rows(
    manifest_streams: Sequence[Mapping[str, object]],
    inspection: Mapping[str, object],
    stream_rows: Mapping[str, object],
    selected: Mapping[str, object],
    selection: Mapping[str, object],
    references: Mapping[str, object],
    systems: Mapping[str, object],
    system_evaluation: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    narratives = _narratives()
    query_rows: list[dict[str, object]] = []
    for stream in manifest_streams:
        topic_id = str(stream["topic_id"])
        stream_id = str(stream["stream_id"])
        key = f"{topic_id}/{stream_id}"
        chosen = str(selected[key])
        arm = next(value for value in stream["arms"] if value["arm_id"] == chosen)
        query_rows.append(
            {
                "topic_id": topic_id,
                "stream_id": stream_id,
                "stream_label": key,
                "full_narrative": narratives[topic_id],
                "existing_query": stream["baseline_query"],
                "reweighted_query": stream["reweighted_query"],
                "selected_arm": chosen,
                "k1": arm["k1"],
                "b": arm["b"],
            }
        )
    query_rows.append(
        {
            "topic_id": "897",
            "stream_id": "original",
            "stream_label": "897/original (unchanged)",
            "full_narrative": narratives["897"],
            "existing_query": "Not applicable — no tested facet stream",
            "reweighted_query": "Not applicable — unchanged R1 topic",
            "selected_arm": "Unchanged R1",
            "k1": None,
            "b": None,
        }
    )

    noise_rows: list[dict[str, object]] = []
    representative_results: list[dict[str, object]] = []
    facet_evidence: list[dict[str, object]] = []
    eligibility = _object(selection.get("eligibility"), "selection eligibility")
    for stream_id in STREAM_IDS:
        topic_id, facet_id = stream_id.split("/", 1)
        chosen = str(selected[stream_id])
        arms = _object(_object(stream_rows[stream_id], stream_id).get("arms"), f"{stream_id}.arms")
        for arm_id in ARM_IDS:
            metrics = _object(_object(arms[arm_id], f"{stream_id}/{arm_id}").get("metrics"), "stream metrics")
            inspect = _object(inspection[f"{stream_id}/{arm_id}"], "inspection")
            for noise_type, top5_field, top10_field in (
                ("Wrong-domain", "domain_drift_top5_count", "domain_drift_top10_count"),
                ("Content-quality", "content_quality_top5_count", "content_quality_top10_count"),
            ):
                noise_rows.append(
                    {
                        "topic_id": topic_id,
                        "stream_id": facet_id,
                        "stream_label": stream_id,
                        "arm_id": arm_id,
                        "stream_arm": f"{stream_id} · {arm_id}",
                        "noise_type": noise_type,
                        "top5_count": inspect[top5_field],
                        "top10_count": inspect[top10_field],
                        "selected": arm_id == chosen,
                    }
                )
        selected_record = _object(arms[chosen], f"selected {stream_id}")
        selected_metrics = _object(selected_record.get("metrics"), f"selected metrics {stream_id}")
        selected_inspection = _object(inspection[f"{stream_id}/{chosen}"], "selected inspection")
        eligible_records = _object(eligibility.get(stream_id), f"selection eligibility {stream_id}")
        eligible_arms = [
            arm_id
            for arm_id in ARM_IDS
            if _object(eligible_records.get(arm_id), f"eligibility {stream_id}/{arm_id}").get("eligible")
            is True
        ]
        if chosen not in eligible_arms:
            raise ValueError(f"selection chose an ineligible arm for {stream_id}")
        selection_rationale = (
            f"Selected {chosen} from eligible arms {', '.join(eligible_arms)} by highest "
            "unique graded gain, then graded Recall@100, nDCG@10, lower combined noise, "
            "and the least-changed arm."
        )
        for index in range(3):
            representative_results.append(
                {
                    "topic_id": topic_id,
                    "stream_id": facet_id,
                    "stream_label": stream_id,
                    "selected_arm": chosen,
                    "rank": index + 1,
                    "document_id": selected_inspection["top_docids"][index],
                    "result_excerpt": selected_inspection["top_snippets"][index],
                }
            )
        facet_evidence.append(
            {
                "topic_id": topic_id,
                "stream_id": facet_id,
                "stream_label": stream_id,
                "selected_arm": chosen,
                "unique_graded_gain": selected_metrics["unique_graded_gain"],
                "unique_relevant_beyond_original": selected_metrics[
                    "unique_relevant_contribution"
                ],
                "graded_recall_at_100": selected_metrics["graded_recall_at_100"],
                "relevant_at_100": selected_metrics["relevant_at_100"],
                "relevant_denominator": selected_metrics["relevant_denominator"],
                "selection_rationale": selection_rationale,
            }
        )

    per_topic_deltas = _object(
        system_evaluation.get("per_topic_ndcg_delta_vs_r1"),
        "system per-topic deltas",
    )
    system_comparison: list[dict[str, object]] = []
    for system_id in SYSTEM_IDS:
        system = _object(systems[system_id], system_id)
        metrics = _object(system.get("metrics"), f"{system_id} metrics")
        system_comparison.append(
            {
                "row_scope": "Aggregate",
                "topic_id": "All four",
                "system": system_id,
                "ranking_reference": (
                    "Selected frozen topic rankings" if system_id == "R2" else system_id
                ),
                "ndcg_at_10": metrics["ndcg@10"],
                "graded_recall_at_100": metrics["graded_recall@100"],
                "recall_at_100": metrics["recall@100"],
                "precision_at_10": metrics["precision@10"],
                "relevant_at_10": metrics["relevant_count@10"],
                "judged_rate_at_10": metrics["judged_rate@10"],
                "judged_rate_at_100": metrics["judged_rate@100"],
                "ndcg_delta_vs_r1": (
                    float(metrics["ndcg@10"])
                    - float(_object(_object(systems["R1"], "R1").get("metrics"), "R1 metrics")["ndcg@10"])
                ),
            }
        )
        per_topic = _object(system.get("per_topic"), f"{system_id} per-topic")
        for topic_id in TOPIC_IDS:
            row = _object(per_topic[topic_id], f"{system_id}/{topic_id}")
            system_comparison.append(
                {
                    "row_scope": "Topic",
                    "topic_id": topic_id,
                    "system": system_id,
                    "ranking_reference": (
                        references[topic_id] if system_id == "R2" else system_id
                    ),
                    "ndcg_at_10": row["ndcg@10"],
                    "graded_recall_at_100": row["graded_recall@100"],
                    "recall_at_100": row["recall@100"],
                    "precision_at_10": row["precision@10"],
                    "relevant_at_10": row["relevant_count@10"],
                    "judged_rate_at_10": row["judged_rate@10"],
                    "judged_rate_at_100": row["judged_rate@100"],
                    "ndcg_delta_vs_r1": (
                        per_topic_deltas[topic_id] if system_id == "R2" else float(row["ndcg@10"])
                        - float(_object(_object(systems["R1"], "R1").get("per_topic"), "R1 per-topic")[topic_id]["ndcg@10"])
                    ),
                }
            )
    return {
        "query_rows": query_rows,
        "noise_comparison": noise_rows,
        "representative_results": representative_results,
        "facet_evidence": facet_evidence,
        "system_comparison": system_comparison,
    }


def build_artifact(
    *,
    manifest: Mapping[str, object],
    control_freeze: Mapping[str, object],
    candidate_streams: Mapping[str, object],
    inspection: Mapping[str, object],
    stream_evaluation: Mapping[str, object],
    selection: Mapping[str, object],
    system_evaluation: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict[str, object]:
    """Return one deterministic, bounded canonical report artifact.

    All arguments are decoded, already verified experimental artifacts.  This
    presentation boundary repeats the critical cross-file checks it relies on
    and fails closed rather than filling absent or inconsistent values.
    """

    manifest = _object(manifest, "manifest")
    control_freeze = _object(control_freeze, "control freeze")
    candidate_streams = _object(candidate_streams, "candidate streams")
    inspection = _object(inspection, "inspection")
    stream_evaluation = _object(stream_evaluation, "stream evaluation")
    selection = _object(selection, "selection")
    system_evaluation = _object(system_evaluation, "system evaluation")
    decision = _object(decision, "decision")
    manifest_streams = _validate_manifest(manifest)
    _validate_candidate_snapshot(control_freeze, candidate_streams, manifest_streams)
    _validate_inspection(inspection)
    stream_rows = _validate_stream_evaluation(stream_evaluation, inspection)
    selected, references = _validate_selection(selection)
    systems = _validate_system_evaluation(system_evaluation)
    if system_evaluation.get("r2_selected_rankings") != references:
        raise ValueError("system evaluation and selection ranking references differ")
    _validate_provenance(stream_evaluation, selection, system_evaluation, decision)
    _validate_decision(decision, systems, stream_rows, selected)

    rows = _artifact_rows(
        manifest_streams,
        inspection,
        stream_rows,
        selected,
        selection,
        references,
        systems,
        system_evaluation,
    )
    decision_name = str(decision["decision"])
    success = decision_name == "retrieval_repair_success"
    outcome = (
        "The saved rule classifies the retrieval repair as successful."
        if success
        else "The saved rule classifies the retrieval repair as unsuccessful."
    )
    selected_noise = decision["evidence"]["selected_top10_noise"]
    b0_noise = decision["evidence"]["b0_top10_noise"]
    sources = _sources(
        control_freeze,
        stream_evaluation,
        system_evaluation,
        decision,
    )
    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": f"# {TITLE}"},
        {
            "id": "executive_summary",
            "type": "markdown",
            "layout": "full",
            "sourceId": "decision_evidence",
            "body": (
                "## Executive Summary\n\n"
                f"- **{outcome}** R2 was judged only by the preregistered comparison with R1.\n"
                "- **The four stream outcomes remain visible.** Each selected arm is tied to a pre-qrels frozen ranking; no ranking was created after evaluation.\n"
                "- **The next gate stays separate.** Cross-encoder not run. A cross-encoder comparison requires separate approval even when retrieval repair succeeds."
            ),
        },
        {
            "id": "queries_section",
            "type": "markdown",
            "layout": "full",
            "sourceId": "query_evidence",
            "body": (
                "## The queries stayed tied to the narrative\n\n"
                "**The repeated terms add emphasis, not new concepts.** The table keeps the full user narrative separate from the existing facet query and reweighted query, then shows the selected arm's k1 and b settings. Topic 897 is included as the unchanged fourth system topic and had no tested facet-control stream."
            ),
        },
        {"id": "queries_table_block", "type": "table", "tableId": "query_details", "layout": "full"},
        {
            "id": "noise_section",
            "type": "markdown",
            "layout": "full",
            "sourceId": "noise_evidence",
            "body": (
                "## Did the controls remove noise?\n\n"
                f"**The mechanical comparison uses the same two warning families for every arm.** Across the selected arms, the combined top-10 wrong-domain and content-quality count is **{selected_noise}**, versus **{b0_noise}** for B0. The chart separates both warning families by stream and arm; these are qrels-free diagnostics, not relevance labels."
            ),
        },
        {"id": "noise_chart_block", "type": "chart", "chartId": "noise_comparison", "layout": "full"},
        {
            "id": "representative_results_note",
            "type": "markdown",
            "layout": "full",
            "sourceId": "noise_evidence",
            "body": (
                "**Representative results make the diagnostic concrete.** The table shows only the first three saved results from each selected facet arm. It is deliberately bounded and does not expose candidate lists or relevance judgments."
            ),
        },
        {"id": "representative_results_block", "type": "table", "tableId": "representative_results", "layout": "full"},
        {
            "id": "facet_evidence_section",
            "type": "markdown",
            "layout": "full",
            "sourceId": "facet_evidence",
            "body": (
                "## Did the facets add relevant evidence?\n\n"
                "**Selection protects marginal evidence first.** Each bar shows the selected arm's unique graded gain beyond the original top 100. The backing data also retains unique relevant documents, graded Recall@100, selected arm, topic, and stream so the gain can be audited in context."
            ),
        },
        {"id": "facet_evidence_chart_block", "type": "chart", "chartId": "unique_facet_evidence", "layout": "full"},
        {
            "id": "system_section",
            "type": "markdown",
            "layout": "full",
            "sourceId": "system_evidence",
            "body": (
                "## Did the final sparse system improve?\n\n"
                f"**{outcome}** The exact O, F0, R1, and R2 aggregate and per-topic values appear below. The rule requires higher R2 graded Recall@100, nDCG@10 no more than 0.02 below R1, no topic nDCG@10 loss greater than 0.10, and selected top-10 noise no higher than B0."
            ),
        },
        {"id": "system_table_block", "type": "table", "tableId": "system_comparison", "layout": "full"},
        {
            "id": "next_steps",
            "type": "markdown",
            "layout": "full",
            "sourceId": "decision_evidence",
            "body": (
                "## What happens next\n\n"
                + (
                    "- Proceed only to a separately approved comparison of original-only and facet-augmented candidates.\n"
                    if success
                    else "- Stop this REST/BM25 repair path and explain that the available controls did not resolve the observed limitation.\n"
                )
                + "- Preserve the frozen R2 decision and all four topic outcomes.\n"
                + "- **Cross-encoder not run.** No reranker inference or automatic reranker gate was opened."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Further questions\n\n"
                "- Would the selected arms retain their marginal evidence on untouched topics?\n"
                "- Are the remaining drift and content-quality warnings genuine relevance failures or harmless lexical noise?\n"
                "- If separately approved, does a cross-encoder benefit facet-augmented candidates more than original-only candidates?"
            ),
        },
        {
            "id": "caveats",
            "type": "markdown",
            "layout": "full",
            "sourceId": "pilot_contract",
            "body": (
                "## Caveats and assumptions\n\n"
                "- This is a four-topic pilot and does not support production generalization.\n"
                "- Projected relevance judgments are development evidence, not a hidden test set.\n"
                "- Wrong-domain and content-quality counts are heuristic diagnostics, not relevance labels.\n"
                "- The hosted REST interface limited controls to query text, hit depth, k1, and b.\n"
                "- No production claim, recursive search, or cross-encoder result is included."
            ),
        },
    ]
    charts = [
        {
            "id": "noise_comparison",
            "title": "Top-10 noise counts by stream and arm",
            "subtitle": "Wrong-domain and content-quality result counts across four registered streams; lower is better.",
            "type": "bar",
            "dataset": "noise_comparison",
            "sourceId": "noise_evidence",
            "valueFormat": "number",
            "layout": "full",
            "encodings": {
                "x": {"field": "stream_arm", "type": "nominal", "label": "Stream and arm"},
                "y": {"field": "top10_count", "type": "quantitative", "label": "Top-10 results", "format": "number"},
                "color": {"field": "noise_type", "type": "nominal", "label": "Diagnostic"},
                "label": {"field": "top10_count", "type": "quantitative", "label": "Count"},
                "tooltip": [
                    {"field": "top5_count", "type": "quantitative", "label": "Top-5 count"},
                    {"field": "selected", "type": "nominal", "label": "Selected arm"},
                ],
            },
        },
        {
            "id": "unique_facet_evidence",
            "title": "Unique graded gain beyond original retrieval",
            "subtitle": "Selected facet arm for each of four streams; gain is outside the original top 100.",
            "type": "bar",
            "dataset": "facet_evidence",
            "sourceId": "facet_evidence",
            "valueFormat": "number",
            "layout": "full",
            "encodings": {
                "x": {"field": "stream_label", "type": "nominal", "label": "Facet stream"},
                "y": {"field": "unique_graded_gain", "type": "quantitative", "label": "Unique graded gain", "format": "number"},
                "label": {"field": "unique_graded_gain", "type": "quantitative", "label": "Gain"},
                "tooltip": [
                    {"field": "selected_arm", "type": "nominal", "label": "Selected arm"},
                    {"field": "unique_relevant_beyond_original", "type": "quantitative", "label": "Unique relevant beyond original"},
                    {"field": "graded_recall_at_100", "type": "quantitative", "label": "Graded Recall@100", "format": "number"},
                    {"field": "relevant_denominator", "type": "quantitative", "label": "Relevant denominator"},
                    {"field": "selection_rationale", "type": "nominal", "label": "Selected-arm rationale"},
                ],
            },
        },
    ]
    tables = [
        _table(
            "query_details",
            "Narratives and facet-control queries",
            "Full narratives are separate from the tested query strings; topic 897 remained unchanged.",
            "query_rows",
            "query_evidence",
            (
                ("topic_id", "Topic", "text"),
                ("stream_id", "Stream", "text"),
                ("full_narrative", "Full narrative", "text"),
                ("existing_query", "Existing query", "text"),
                ("reweighted_query", "Reweighted query", "text"),
                ("selected_arm", "Selected arm", "text"),
                ("k1", "k1", "number"),
                ("b", "b", "number"),
            ),
            "topic_id",
        ),
        _table(
            "representative_results",
            "Representative selected-arm results",
            "First three saved results per selected facet stream; twelve bounded rows.",
            "representative_results",
            "noise_evidence",
            (
                ("stream_label", "Facet stream", "text"),
                ("selected_arm", "Selected arm", "text"),
                ("rank", "Rank", "number"),
                ("document_id", "Document", "text"),
                ("result_excerpt", "Saved excerpt", "text"),
            ),
            "stream_label",
        ),
        _table(
            "system_comparison",
            "Exact sparse-system comparison",
            "Aggregate and per-topic O/F0/R1/R2 metrics with frozen ranking references and nDCG deltas versus R1.",
            "system_comparison",
            "system_evidence",
            (
                ("row_scope", "Scope", "text"),
                ("topic_id", "Topic", "text"),
                ("system", "System", "text"),
                ("ranking_reference", "Frozen reference", "text"),
                ("ndcg_at_10", "nDCG@10", "number"),
                ("graded_recall_at_100", "Graded Recall@100", "number"),
                ("recall_at_100", "Recall@100", "number"),
                ("precision_at_10", "P@10", "number"),
                ("relevant_at_10", "Relevant@10", "number"),
                ("judged_rate_at_10", "Judged rate@10", "number"),
                ("judged_rate_at_100", "Judged rate@100", "number"),
                ("ndcg_delta_vs_r1", "nDCG delta vs R1", "number"),
            ),
            "row_scope",
        ),
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Decision-ready four-topic facet retrieval-control pilot report.",
            "charts": charts,
            "tables": tables,
            "sources": [
                {"id": source["id"], "label": source["label"], "path": source["path"]}
                for source in sources
            ],
            "blocks": blocks,
        },
        "snapshot": {"version": 1, "status": "ready", "datasets": rows},
        "sources": sources,
    }


def _load_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable or invalid JSON") from exc
    return _object(value, label)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = build_artifact(
        manifest=_load_json(args.manifest, "manifest"),
        control_freeze=_load_json(args.freeze_dir / "freeze.json", "control freeze"),
        candidate_streams=_load_json(
            args.freeze_dir / "candidate_streams.json", "candidate streams"
        ),
        inspection=_load_json(args.freeze_dir / "inspection.json", "inspection"),
        stream_evaluation=_load_json(
            args.evaluation_dir / "stream_evaluation.json", "stream evaluation"
        ),
        selection=_load_json(args.evaluation_dir / "selection.json", "selection"),
        system_evaluation=_load_json(
            args.evaluation_dir / "evaluation.json", "system evaluation"
        ),
        decision=_load_json(args.evaluation_dir / "decision.json", "decision"),
    )
    if args.output.exists():
        raise FileExistsError(f"create-only report source already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
