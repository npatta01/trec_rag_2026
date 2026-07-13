import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from trec_rag.build_facet_retrieval_control_report import build_artifact
from trec_rag.facet_retrieval_control_experiment import (
    EXPECTED_ALTERNATIVE_NAMES,
    StreamArmEvaluation,
    select_stream_arm,
    selected_ranking_references,
)
from trec_rag.facet_retrieval_control_freeze import (
    _BASE_ARM_QUERIES,
    _canonical_json as _producer_canonical_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT
    / "reports"
    / "experiments"
    / "facet_retrieval_control_pilot_v1"
    / "manifest.json"
)
README_PATH = MANIFEST_PATH.with_name("README.md")
STREAM_IDS = ("200/f07a", "225/f02", "225/f04", "707/f02")
ARM_IDS = ("B0", "W0", "W1", "W2")
TOPIC_IDS = ("200", "225", "707", "897")
SYSTEM_IDS = ("O", "F0", "R1", "R2")
METRICS = (
    "ndcg@10",
    "graded_recall@100",
    "recall@100",
    "precision@10",
    "relevant_count@10",
    "judged_rate@10",
    "judged_rate@100",
)


def _canonical_json(value):
    return _producer_canonical_json(value)


def _seal(payload):
    sealed = copy.deepcopy(payload)
    sealed.pop("artifact_sha256", None)
    sealed["artifact_sha256"] = hashlib.sha256(_canonical_json(sealed)).hexdigest()
    return sealed


def _reseal(data, *names):
    for name in names:
        data[name] = _seal(data[name])


def _mean_metric(per_topic, metric):
    return sum(per_topic[topic_id][metric] for topic_id in TOPIC_IDS) / len(TOPIC_IDS)


def _narratives():
    return {
        topic_id: query
        for topic_id, variant_name, query in _BASE_ARM_QUERIES
        if variant_name == "prompt_lab_v1:original"
    }


def _candidate_streams():
    narratives = _narratives()
    entries = []
    for topic_id in TOPIC_IDS:
        query_hash = hashlib.sha256(narratives[topic_id].encode()).hexdigest()
        entries.append(
            {
                "topic_id": topic_id,
                "stream_id": "original",
                "arm_id": "O",
                "role": "original",
                "source_kind": "prior_base",
                "query_sha256": query_hash,
                "request_sha256": hashlib.sha256(
                    f"request/{topic_id}/original".encode()
                ).hexdigest(),
                "response_sha256": hashlib.sha256(
                    f"response/{topic_id}/original".encode()
                ).hexdigest(),
                "source_candidates_sha256": hashlib.sha256(
                    f"candidates/{topic_id}/original".encode()
                ).hexdigest(),
                "expected_depth": 100,
                "row_count": 100,
                "stream_rows_sha256": hashlib.sha256(
                    f"rows/{topic_id}/original".encode()
                ).hexdigest(),
            }
        )
    manifest = json.loads(MANIFEST_PATH.read_text())
    queries = {
        (stream["topic_id"], stream["stream_id"], arm["arm_id"]): (
            stream["baseline_query"]
            if arm["arm_id"] == "B0"
            else stream["reweighted_query"]
        )
        for stream in manifest["streams"]
        for arm in stream["arms"]
    }
    for topic_id, stream_id in (item.split("/") for item in STREAM_IDS):
        for arm_id in ARM_IDS:
            query = queries[(topic_id, stream_id, arm_id)]
            entries.append(
                {
                    "topic_id": topic_id,
                    "stream_id": stream_id,
                    "arm_id": arm_id,
                    "role": "facet",
                    "source_kind": "prior_r1" if arm_id == "B0" else "control",
                    "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                    "request_sha256": hashlib.sha256(
                        f"request/{topic_id}/{stream_id}/{arm_id}".encode()
                    ).hexdigest(),
                    "response_sha256": hashlib.sha256(
                        f"response/{topic_id}/{stream_id}/{arm_id}".encode()
                    ).hexdigest(),
                    "source_candidates_sha256": hashlib.sha256(
                        f"candidates/{topic_id}/{stream_id}/{arm_id}".encode()
                    ).hexdigest(),
                    "expected_depth": 100,
                    "row_count": 100,
                    "stream_rows_sha256": hashlib.sha256(
                        f"rows/{topic_id}/{stream_id}/{arm_id}".encode()
                    ).hexdigest(),
                }
            )
    entries.sort(key=lambda row: (row["topic_id"], row["stream_id"], row["arm_id"]))
    return {
        "schema_version": "facet-control-candidate-streams-v1",
        "candidate_schema_version": "facet-control-candidate-row-v1",
        "candidate_file": "candidates.jsonl",
        "candidate_file_sha256": "a" * 64,
        "stream_count": 20,
        "row_count": 2000,
        "streams": entries,
    }


def _inspection(stream_id, arm_id, ordinal):
    topic_id, facet_id = stream_id.split("/")
    drift10 = (ordinal + (arm_id == "B0")) % 4
    content10 = (ordinal + ARM_IDS.index(arm_id)) % 3
    return {
        "topic_id": topic_id,
        "stream_id": facet_id,
        "inspected_top5": 5,
        "inspected_top10": 10,
        "anchor_top5_count": 5,
        "anchor_top10_count": 10,
        "anchor_intent_cohit_top5_count": 4,
        "anchor_intent_cohit_top10_count": 8,
        "domain_drift_top5_count": min(drift10, 2),
        "domain_drift_top10_count": drift10,
        "content_quality_top5_count": min(content10, 2),
        "content_quality_top10_count": content10,
        "coherence_failed": False,
        "domain_drift_warning": drift10 >= 2,
        "content_quality_warning": content10 >= 2,
        "rejected": False,
        "decision": "keep",
        "top_docids": [f"{topic_id}-{facet_id}-{arm_id}-{rank}" for rank in range(1, 11)],
        "top_snippets": [
            f"Representative {topic_id}/{facet_id} {arm_id} result {rank}."
            for rank in range(1, 11)
        ],
    }


def _arm_metrics(stream_id, arm_id, ordinal, inspection):
    arm_index = ARM_IDS.index(arm_id)
    return {
        "arm_id": arm_id,
        "overlap_with_original_top100": 20 + ordinal + arm_index,
        "relevant_at_10": 2 + (arm_index % 2),
        "relevant_at_100": 9 + arm_index,
        "graded_recall_at_100": 0.30 + ordinal * 0.02 + arm_index * 0.01,
        "ndcg_at_10": 0.20 + ordinal * 0.02 + arm_index * 0.01,
        "unique_relevant_contribution": ordinal + arm_index + 1,
        "unique_graded_gain": ordinal * 2 + arm_index + 1,
        "domain_drift_top10_count": inspection["domain_drift_top10_count"],
        "content_quality_top10_count": inspection["content_quality_top10_count"],
        "coherence_failed": False,
        "recall_at_100": 0.40 + ordinal * 0.01 + arm_index * 0.01,
        "precision_at_10": 0.20 + arm_index * 0.01,
        "judged_rate_at_10": 0.80,
        "judged_rate_at_100": 0.55,
        "relevant_denominator": 30,
        "graded_recall_denominator": 55,
    }


def _system_metrics(system_id):
    base = {"O": 0.34, "F0": 0.35, "R1": 0.40, "R2": 0.395}[system_id]
    graded = {"O": 0.48, "F0": 0.49, "R1": 0.50, "R2": 0.52}[system_id]
    per_topic_ndcg = {
        "O": (0.32, 0.34, 0.35, 0.35),
        "F0": (0.33, 0.36, 0.35, 0.36),
        "R1": (0.40, 0.40, 0.40, 0.40),
        "R2": (0.41, 0.38, 0.39, 0.40),
    }[system_id]
    per_topic = {}
    for index, topic_id in enumerate(TOPIC_IDS):
        per_topic[topic_id] = {
            "ndcg@10": per_topic_ndcg[index],
            "graded_recall@100": graded + (index - 1.5) * 0.01,
            "recall@100": graded - 0.05 + index * 0.002,
            "precision@10": 0.25 + base / 10 + index * 0.001,
            "relevant_count@10": 2 + index + SYSTEM_IDS.index(system_id),
            "judged_rate@10": 0.78 + index * 0.01,
            "judged_rate@100": 0.53 + index * 0.01,
        }
    aggregate = {metric: _mean_metric(per_topic, metric) for metric in METRICS}
    return {"metrics": aggregate, "per_topic": per_topic}


@pytest.fixture
def report_inputs():
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_file_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    candidate_streams = _candidate_streams()
    candidate_file_bytes = b"synthetic frozen candidate rows\n"
    candidate_streams["candidate_file_sha256"] = hashlib.sha256(
        candidate_file_bytes
    ).hexdigest()
    inspections = {}
    stream_eval = {}
    for ordinal, stream_id in enumerate(STREAM_IDS):
        arms = {}
        for arm_id in ARM_IDS:
            inspection = _inspection(stream_id, arm_id, ordinal)
            inspections[f"{stream_id}/{arm_id}"] = inspection
            arms[arm_id] = {
                "metrics": _arm_metrics(stream_id, arm_id, ordinal, inspection),
                "inspection": copy.deepcopy(inspection),
            }
        stream_eval[stream_id] = {"arms": arms}
    stream_hashes = {
        f"{row['topic_id']}/{row['stream_id']}/{row['arm_id']}": row[
            "stream_rows_sha256"
        ]
        for row in candidate_streams["streams"]
    }
    freeze = {
        "schema_version": "facet-control-ranking-freeze-v2",
        "status": "frozen_before_qrels",
        "bindings": {
            "manifest_sha256": manifest_file_sha256,
            "prior_freeze_sha256": "b" * 64,
            "request_sha256": {},
            "response_sha256": {},
            "candidate_sha256": {},
            "ledger_sha256": {"base": "c" * 64, "r1": "d" * 64, "control": "e" * 64},
            "r1_arm_sha256": "f" * 64,
            "candidate_streams_sha256": hashlib.sha256(
                _canonical_json(candidate_streams)
            ).hexdigest(),
            "candidates_sha256": candidate_streams["candidate_file_sha256"],
            "candidate_stream_rows_sha256": stream_hashes,
        },
        "inspection_sha256": hashlib.sha256(_canonical_json(inspections)).hexdigest(),
        "fusion_sha256": "3" * 64,
        "rankings": {
            name: {
                "path": f"rankings/{name.replace(':', '__')}.jsonl",
                "rows": 100,
                "sha256": "4" * 64,
                "file_sha256": "5" * 64,
            }
            for name in EXPECTED_ALTERNATIVE_NAMES
        },
    }
    freeze["freeze_sha256"] = hashlib.sha256(_canonical_json(freeze)).hexdigest()
    control_freeze_bytes = _canonical_json(freeze)
    control_freeze_file_sha256 = hashlib.sha256(control_freeze_bytes).hexdigest()
    selected = {
        stream_id: select_stream_arm(
            {
                arm_id: StreamArmEvaluation(
                    **stream_eval[stream_id]["arms"][arm_id]["metrics"]
                )
                for arm_id in ARM_IDS
            }
        ).arm_id
        for stream_id in STREAM_IDS
    }
    assert selected == {stream_id: "W2" for stream_id in STREAM_IDS}
    references = selected_ranking_references(selected)
    provenance = {
        "control_freeze_file_sha256": control_freeze_file_sha256,
        "control_freeze_sha256": freeze["freeze_sha256"],
        "manifest_sha256": manifest_file_sha256,
        "prior_freeze_sha256": freeze["bindings"]["prior_freeze_sha256"],
        "qrels_sha256": "8" * 64,
        "qrels_name": "projected_qrels.tsv",
    }
    stream_evaluation = _seal({
        "schema_version": "facet-control-stream-evaluation-v1",
        "provenance": provenance,
        "qrels_policy": {
            "relevance_threshold": 2,
            "missing_judgments_grade": 0,
            "graded_recall_denominator": "sum of non-negative qrel grades",
            "zero_denominator_value": 0.0,
            "protected_qrels_topics_skipped": [],
        },
        "streams": stream_eval,
    })
    eligibility = {}
    for stream_id in STREAM_IDS:
        baseline = stream_eval[stream_id]["arms"]["B0"]["metrics"]
        eligibility[stream_id] = {}
        for arm_id in ARM_IDS:
            metrics = stream_eval[stream_id]["arms"][arm_id]["metrics"]
            reasons = []
            if metrics["coherence_failed"]:
                reasons.append("coherence_failed")
            if (
                metrics["domain_drift_top10_count"]
                > baseline["domain_drift_top10_count"]
                and metrics["content_quality_top10_count"]
                > baseline["content_quality_top10_count"]
            ):
                reasons.append("both_noise_families_increased_vs_B0")
            eligibility[stream_id][arm_id] = {
                "eligible": not reasons,
                "exclusion_reasons": reasons,
            }
    selection = _seal({
        "schema_version": "facet-control-selection-v1",
        "provenance": provenance,
        "selected": selected,
        "eligibility": eligibility,
        "selected_rankings": references,
        "selected_ranking_sha256": {
            topic_id: freeze["rankings"][references[topic_id]]["sha256"]
            for topic_id in TOPIC_IDS
        },
        "frozen_alternative_count": 25,
    })
    systems = {system_id: _system_metrics(system_id) for system_id in SYSTEM_IDS}
    deltas = {
        topic_id: systems["R2"]["per_topic"][topic_id]["ndcg@10"]
        - systems["R1"]["per_topic"][topic_id]["ndcg@10"]
        for topic_id in TOPIC_IDS
    }
    system_evaluation = _seal({
        "schema_version": "facet-control-system-evaluation-v1",
        "provenance": provenance,
        "systems": systems,
        "r2_selected_rankings": references,
        "r2_selected_ranking_sha256": selection["selected_ranking_sha256"],
        "per_topic_ndcg_delta_vs_r1": deltas,
    })
    selected_noise = sum(
        stream_eval[stream_id]["arms"][selected[stream_id]]["metrics"][
            "domain_drift_top10_count"
        ]
        + stream_eval[stream_id]["arms"][selected[stream_id]]["metrics"][
            "content_quality_top10_count"
        ]
        for stream_id in STREAM_IDS
    )
    b0_noise = sum(
        stream_eval[stream_id]["arms"]["B0"]["metrics"][
            "domain_drift_top10_count"
        ]
        + stream_eval[stream_id]["arms"]["B0"]["metrics"][
            "content_quality_top10_count"
        ]
        for stream_id in STREAM_IDS
    )
    decision = _seal({
        "schema_version": "facet-control-decision-v1",
        "provenance": provenance,
        "decision": "retrieval_repair_success",
        "reranker_gate_opened": False,
        "evidence": {
            "r2_graded_recall_at_100": systems["R2"]["metrics"]["graded_recall@100"],
            "r1_graded_recall_at_100": systems["R1"]["metrics"]["graded_recall@100"],
            "r2_ndcg_at_10": systems["R2"]["metrics"]["ndcg@10"],
            "r1_ndcg_at_10": systems["R1"]["metrics"]["ndcg@10"],
            "per_topic_ndcg_deltas": deltas,
            "selected_top10_noise": selected_noise,
            "b0_top10_noise": b0_noise,
        },
    })
    return {
        "manifest": manifest,
        "manifest_bytes": manifest_bytes,
        "control_freeze": freeze,
        "control_freeze_bytes": control_freeze_bytes,
        "candidate_streams": candidate_streams,
        "candidate_file_bytes": candidate_file_bytes,
        "inspection": inspections,
        "stream_evaluation": stream_evaluation,
        "selection": selection,
        "system_evaluation": system_evaluation,
        "decision": decision,
    }


def _refresh_integrity(data):
    freeze = data["control_freeze"]
    freeze["bindings"]["candidate_streams_sha256"] = hashlib.sha256(
        _canonical_json(data["candidate_streams"])
    ).hexdigest()
    freeze["bindings"]["candidate_stream_rows_sha256"] = {
        f"{row['topic_id']}/{row['stream_id']}/{row['arm_id']}": row[
            "stream_rows_sha256"
        ]
        for row in data["candidate_streams"]["streams"]
    }
    freeze["inspection_sha256"] = hashlib.sha256(
        _canonical_json(data["inspection"])
    ).hexdigest()
    freeze.pop("freeze_sha256", None)
    freeze["freeze_sha256"] = hashlib.sha256(_canonical_json(freeze)).hexdigest()
    data["control_freeze_bytes"] = _canonical_json(freeze)
    provenance = copy.deepcopy(data["stream_evaluation"]["provenance"])
    provenance.update(
        {
            "control_freeze_file_sha256": hashlib.sha256(
                data["control_freeze_bytes"]
            ).hexdigest(),
            "control_freeze_sha256": freeze["freeze_sha256"],
            "manifest_sha256": hashlib.sha256(data["manifest_bytes"]).hexdigest(),
            "prior_freeze_sha256": freeze["bindings"]["prior_freeze_sha256"],
        }
    )
    for name in ("stream_evaluation", "selection", "system_evaluation", "decision"):
        data[name]["provenance"] = copy.deepcopy(provenance)
        data[name] = _seal(data[name])


def _set_saved_selection(data, stream_id, arm_id):
    data["selection"]["selected"][stream_id] = arm_id
    references = selected_ranking_references(data["selection"]["selected"])
    data["selection"]["selected_rankings"] = references
    data["system_evaluation"]["r2_selected_rankings"] = copy.deepcopy(references)
    selected = data["selection"]["selected"]
    streams = data["stream_evaluation"]["streams"]
    data["decision"]["evidence"]["selected_top10_noise"] = sum(
        streams[key]["arms"][selected[key]]["metrics"]["domain_drift_top10_count"]
        + streams[key]["arms"][selected[key]]["metrics"][
            "content_quality_top10_count"
        ]
        for key in STREAM_IDS
    )
    _refresh_integrity(data)


def _refresh_system_aggregate(data, system_id):
    system = data["system_evaluation"]["systems"][system_id]
    system["metrics"] = {
        metric: _mean_metric(system["per_topic"], metric) for metric in METRICS
    }


def test_report_exposes_queries_noise_marginal_gain_and_decision(report_inputs):
    artifact = build_artifact(**report_inputs)
    text = json.dumps(artifact)
    for required in (
        "Existing query",
        "Reweighted query",
        "k1",
        "b",
        "Wrong-domain",
        "Unique relevant beyond original",
        "Selected arm",
        "R2",
        "Cross-encoder not run",
    ):
        assert required in text
    assert artifact["snapshot"]["status"] == "ready"


def test_artifact_has_canonical_shape_and_executive_reading_path(report_inputs):
    artifact = build_artifact(**report_inputs)

    assert set(artifact) == {"surface", "manifest", "snapshot", "sources"}
    assert artifact["surface"] == artifact["manifest"]["surface"] == "report"
    assert artifact["manifest"]["version"] == artifact["snapshot"]["version"] == 1
    blocks = artifact["manifest"]["blocks"]
    title = artifact["manifest"]["title"]
    assert blocks[0]["type"] == "markdown"
    assert blocks[0]["body"] == f"# {title}"
    assert blocks[1]["body"].startswith("## Executive Summary\n")
    headings = [
        block["body"].splitlines()[0]
        for block in blocks
        if block["type"] == "markdown" and block["body"].startswith("## ")
    ]
    assert headings == [
        "## Executive Summary",
        "## The queries stayed tied to the narrative",
        "## Did the controls remove noise?",
        "## Did the facets add relevant evidence?",
        "## Did the final sparse system improve?",
        "## What happens next",
        "## Further questions",
        "## Caveats and assumptions",
    ]
    assert all(body.count("\n## ") == 0 for body in (b["body"] for b in blocks if b["type"] == "markdown"))


def test_narratives_are_bound_to_verified_original_snapshot_hashes(report_inputs):
    artifact = build_artifact(**report_inputs)
    narratives = artifact["snapshot"]["datasets"]["narrative_rows"]
    query_arms = artifact["snapshot"]["datasets"]["query_rows"]

    assert {row["topic_id"] for row in narratives} == set(TOPIC_IDS)
    assert len(narratives) == 4
    assert any(
        row["topic_id"] == "897" and row["stream_scope"] == "unchanged"
        for row in narratives
    )
    assert len(query_arms) == 17
    assert all("full_narrative" not in row for row in query_arms)

    tampered = copy.deepcopy(report_inputs)
    original = next(
        row
        for row in tampered["candidate_streams"]["streams"]
        if row["topic_id"] == "200" and row["stream_id"] == "original"
    )
    original["query_sha256"] = "0" * 64
    _refresh_integrity(tampered)
    with pytest.raises(ValueError, match="narrative.*snapshot"):
        build_artifact(**tampered)


def test_query_evidence_uses_narrow_accessible_tables(report_inputs):
    artifact = build_artifact(**report_inputs)
    tables = {table["id"]: table for table in artifact["manifest"]["tables"]}

    assert [column["field"] for column in tables["narrative_details"]["columns"]] == [
        "topic_id",
        "stream_scope",
        "full_narrative",
    ]
    assert [column["field"] for column in tables["query_details"]["columns"]] == [
        "topic_id",
        "stream_id",
        "arm_settings",
        "query_text",
    ]
    query_rows = artifact["snapshot"]["datasets"]["query_rows"]
    assert sum("SELECTED" in row["arm_settings"] for row in query_rows) == 5
    blocks = [block["id"] for block in artifact["manifest"]["blocks"]]
    assert blocks.index("narratives_table_block") < blocks.index("query_arms_note")
    assert blocks.index("query_arms_note") < blocks.index("queries_table_block")


def test_audit_tables_bound_width_and_avoid_unbroken_document_ids(report_inputs):
    artifact = build_artifact(**report_inputs)
    tables = {table["id"]: table for table in artifact["manifest"]["tables"]}

    assert max(len(table["columns"]) for table in tables.values()) <= 6
    assert "document_id" not in {
        column["field"] for column in tables["representative_results"]["columns"]
    }
    assert {
        "system_primary",
        "system_supporting",
        "r2_ranking_references",
    } <= set(tables)
    blocks = [block["id"] for block in artifact["manifest"]["blocks"]]
    assert blocks.index("system_table_block") < blocks.index("system_supporting_note")
    assert blocks.index("system_supporting_note") < blocks.index(
        "system_supporting_table_block"
    )
    assert blocks.index("system_supporting_table_block") < blocks.index(
        "ranking_references_note"
    )
    assert blocks.index("ranking_references_note") < blocks.index(
        "ranking_references_table_block"
    )


def test_visual_contracts_and_adjacency_are_canonical(report_inputs):
    artifact = build_artifact(**report_inputs)
    manifest = artifact["manifest"]
    datasets = artifact["snapshot"]["datasets"]
    charts = {chart["id"]: chart for chart in manifest["charts"]}
    tables = {table["id"]: table for table in manifest["tables"]}
    blocks = manifest["blocks"]

    noise = charts["noise_comparison"]
    assert noise["type"] == "bar"
    assert noise["encodings"]["x"]["field"] == "stream_arm"
    assert noise["encodings"]["y"]["field"] == "top10_count"
    assert noise["encodings"]["color"]["field"] == "noise_type"
    assert len(datasets[noise["dataset"]]) == 32
    assert {row["noise_type"] for row in datasets[noise["dataset"]]} == {
        "Wrong-domain",
        "Content-quality",
    }
    gain = charts["unique_facet_evidence"]
    assert gain["type"] == "bar"
    assert gain["encodings"]["y"]["field"] == "unique_graded_gain"
    assert "color" not in gain["encodings"]
    gain_rows = datasets[gain["dataset"]]
    assert len(gain_rows) == 4
    assert {
        "topic_id",
        "stream_id",
        "stream_label",
        "selected_arm",
        "unique_graded_gain",
        "unique_relevant_beyond_original",
        "graded_recall_at_100",
        "selection_rationale",
    } <= set(gain_rows[0])
    assert all("unique graded gain" in row["selection_rationale"] for row in gain_rows)
    system = tables["system_primary"]
    assert system["density"] == "spacious"
    assert {row["system"] for row in datasets[system["dataset"]]} == set(SYSTEM_IDS)
    assert {row["row_scope"] for row in datasets[system["dataset"]]} == {
        "Aggregate",
        "Topic",
    }

    for index, block in enumerate(blocks):
        if block["type"] not in {"chart", "table"}:
            continue
        neighbors = blocks[max(index - 1, 0) : index] + blocks[index + 1 : index + 2]
        assert any(neighbor["type"] == "markdown" for neighbor in neighbors)
        item = charts[block["chartId"]] if block["type"] == "chart" else tables[block["tableId"]]
        assert item["sourceId"] in {source["id"] for source in artifact["sources"]}


def test_quantitative_blocks_and_native_evidence_have_provenance(report_inputs):
    artifact = build_artifact(**report_inputs)
    source_ids = {source["id"] for source in artifact["sources"]}
    manifest_source_ids = {source["id"] for source in artifact["manifest"]["sources"]}
    assert source_ids == manifest_source_ids
    for source in artifact["sources"]:
        assert not Path(source["path"]).is_absolute()
        assert ".." not in Path(source["path"]).parts
        serialized_source = json.dumps(source, sort_keys=True)
        assert "freeze_v1" not in serialized_source
        assert "evaluation_v1" not in serialized_source
        assert source["query"]["sql"].lstrip().upper().startswith(("SELECT", "WITH"))
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", source["query"]["id"])
        assert any("validate_artifact" in note for note in source["query"]["filters"])
    descriptions = {source["id"]: source["query"]["description"] for source in artifact["sources"]}
    assert "grouped categorical bar" in descriptions["noise_evidence"]
    assert "single-series categorical bar" in descriptions["facet_evidence"]
    assert "exact lookup table" in descriptions["system_evidence"]
    for block in artifact["manifest"]["blocks"]:
        if block["type"] == "markdown" and re.search(r"\d", block["body"]):
            assert block.get("sourceId") in source_ids
    for kind in ("charts", "tables", "cards"):
        for item in artifact["manifest"].get(kind, []):
            assert item.get("sourceId") in source_ids


def test_report_is_bounded_safe_and_deterministic_under_input_reordering(report_inputs):
    first = build_artifact(**report_inputs)
    reordered = copy.deepcopy(report_inputs)
    reordered["inspection"] = dict(reversed(list(reordered["inspection"].items())))
    reordered["stream_evaluation"]["streams"] = dict(
        reversed(list(reordered["stream_evaluation"]["streams"].items()))
    )
    reordered["selection"]["selected"] = dict(
        reversed(list(reordered["selection"]["selected"].items()))
    )
    reordered["system_evaluation"]["systems"] = dict(
        reversed(list(reordered["system_evaluation"]["systems"].items()))
    )
    reordered["candidate_streams"] = dict(
        reversed(list(reordered["candidate_streams"].items()))
    )
    reordered["candidate_streams"]["streams"] = [
        dict(reversed(list(row.items())))
        for row in reordered["candidate_streams"]["streams"]
    ]
    assert build_artifact(**reordered) == first

    encoded = json.dumps(first, sort_keys=True)
    assert "/home/" not in encoded
    assert "api.castorini" not in encoded
    assert "endpoint" not in encoded.lower()
    assert "credential" not in encoded.lower()
    assert "token" not in encoded.lower()
    assert not set(("144", "213", "224", "407", "515")) & set(re.findall(r'"(\d+)"', encoded))
    assert "qrels_sha256" not in encoded
    assert "projected_qrels.tsv" not in encoded
    assert "candidates.jsonl" not in encoded
    assert len(first["snapshot"]["datasets"]["representative_results"]) == 12


def test_saved_selection_is_recomputed_from_canonical_task5_semantics(report_inputs):
    assert report_inputs["selection"]["selected"]["200/f07a"] == "W2"
    broken = copy.deepcopy(report_inputs)
    _set_saved_selection(broken, "200/f07a", "W1")

    with pytest.raises(ValueError, match="canonical.*selection|selected arm"):
        build_artifact(**broken)


def test_ineligible_high_gain_arm_cannot_be_selected(report_inputs):
    broken = copy.deepcopy(report_inputs)
    key = "200/f07a/W2"
    broken["inspection"][key]["coherence_failed"] = True
    broken["stream_evaluation"]["streams"]["200/f07a"]["arms"]["W2"][
        "inspection"
    ]["coherence_failed"] = True
    broken["stream_evaluation"]["streams"]["200/f07a"]["arms"]["W2"][
        "metrics"
    ]["coherence_failed"] = True
    _refresh_integrity(broken)

    with pytest.raises(ValueError, match="canonical.*selection|selected arm"):
        build_artifact(**broken)


def test_exact_tie_uses_b0_w0_w1_w2_preference(report_inputs):
    broken = copy.deepcopy(report_inputs)
    stream_id = "200/f07a"
    arms = broken["stream_evaluation"]["streams"][stream_id]["arms"]
    baseline = arms["B0"]["metrics"]
    for arm_id in ARM_IDS:
        for field in (
            "unique_graded_gain",
            "graded_recall_at_100",
            "ndcg_at_10",
            "domain_drift_top10_count",
            "content_quality_top10_count",
        ):
            arms[arm_id]["metrics"][field] = baseline[field]
        arms[arm_id]["metrics"]["coherence_failed"] = False
        for field in (
            "domain_drift_top10_count",
            "content_quality_top10_count",
            "coherence_failed",
        ):
            arms[arm_id]["inspection"][field] = arms[arm_id]["metrics"][field]
            broken["inspection"][f"{stream_id}/{arm_id}"][field] = arms[arm_id][
                "metrics"
            ][field]
    _set_saved_selection(broken, stream_id, "W0")

    with pytest.raises(ValueError, match="canonical.*selection|selected arm"):
        build_artifact(**broken)


def test_aggregate_metrics_are_exact_means_of_per_topic_rows(report_inputs):
    assert report_inputs["system_evaluation"]["systems"]["R2"]["metrics"][
        "ndcg@10"
    ] == 0.395
    broken = copy.deepcopy(report_inputs)
    broken["system_evaluation"]["systems"]["R2"]["metrics"]["ndcg@10"] = 0.39
    _reseal(broken, "system_evaluation")

    with pytest.raises(ValueError, match="aggregate.*ndcg@10"):
        build_artifact(**broken)


def test_decision_flip_is_recomputed_from_per_topic_metrics(report_inputs):
    broken = copy.deepcopy(report_inputs)
    systems = broken["system_evaluation"]["systems"]
    systems["R2"]["per_topic"]["200"]["ndcg@10"] = 0.29
    _refresh_system_aggregate(broken, "R2")
    deltas = {
        topic_id: systems["R2"]["per_topic"][topic_id]["ndcg@10"]
        - systems["R1"]["per_topic"][topic_id]["ndcg@10"]
        for topic_id in TOPIC_IDS
    }
    broken["system_evaluation"]["per_topic_ndcg_delta_vs_r1"] = deltas
    broken["decision"]["evidence"].update(
        {
            "r2_ndcg_at_10": systems["R2"]["metrics"]["ndcg@10"],
            "per_topic_ndcg_deltas": deltas,
        }
    )
    assert broken["decision"]["decision"] == "retrieval_repair_success"
    _refresh_integrity(broken)

    with pytest.raises(ValueError, match="decision classification"):
        build_artifact(**broken)


def test_evaluation_self_hash_rejects_stale_zero(report_inputs):
    broken = copy.deepcopy(report_inputs)
    broken["selection"]["artifact_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="selection.*self-hash"):
        build_artifact(**broken)


def test_tampered_payload_with_stale_hash_fails_before_selection(report_inputs):
    broken = copy.deepcopy(report_inputs)
    broken["selection"]["selected"]["200/f07a"] = "W1"

    with pytest.raises(ValueError, match="selection.*self-hash"):
        build_artifact(**broken)


def test_selected_ranking_hashes_match_authenticated_freeze_records(report_inputs):
    selection = report_inputs["selection"]
    system = report_inputs["system_evaluation"]
    freeze = report_inputs["control_freeze"]

    for topic_id, reference in selection["selected_rankings"].items():
        expected = freeze["rankings"][reference]["sha256"]
        assert selection["selected_ranking_sha256"][topic_id] == expected
        assert system["r2_selected_ranking_sha256"][topic_id] == expected


def test_coordinated_selected_ranking_hash_substitution_is_rejected(report_inputs):
    broken = copy.deepcopy(report_inputs)
    wrong = "a" * 64
    broken["selection"]["selected_ranking_sha256"]["200"] = wrong
    broken["system_evaluation"]["r2_selected_ranking_sha256"]["200"] = wrong
    _reseal(broken, "selection", "system_evaluation")

    with pytest.raises(ValueError, match="ranking hash.*freeze|authenticated ranking"):
        build_artifact(**broken)


def test_selected_reference_tamper_is_rejected_after_freeze_reseal(report_inputs):
    broken = copy.deepcopy(report_inputs)
    reference = broken["selection"]["selected_rankings"]["225"]
    broken["control_freeze"]["rankings"][reference]["sha256"] = "a" * 64
    _refresh_integrity(broken)

    with pytest.raises(ValueError, match="ranking hash.*freeze|authenticated ranking"):
        build_artifact(**broken)


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param("missing", id="missing"),
        pytest.param(None, id="null"),
        pytest.param({"200": "4" * 64}, id="partial"),
        pytest.param(
            {
                "200": "4" * 64,
                "225": "4" * 64,
                "707": "4" * 64,
                "897": "4" * 64,
                "extra": "4" * 64,
            },
            id="extra",
        ),
        pytest.param(["4" * 64] * 4, id="wrong-type"),
    ],
)
def test_system_selected_ranking_hash_map_is_required_and_exact(
    report_inputs, replacement
):
    broken = copy.deepcopy(report_inputs)
    if replacement == "missing":
        broken["system_evaluation"].pop("r2_selected_ranking_sha256")
    else:
        broken["system_evaluation"]["r2_selected_ranking_sha256"] = replacement
    _reseal(broken, "system_evaluation")

    with pytest.raises(ValueError, match="system evaluation ranking hash"):
        build_artifact(**broken)


def test_candidate_file_hash_is_bound_in_manifest_and_freeze(report_inputs):
    expected = hashlib.sha256(report_inputs["candidate_file_bytes"]).hexdigest()
    assert report_inputs["candidate_streams"]["candidate_file_sha256"] == expected
    assert (
        report_inputs["control_freeze"]["bindings"]["candidates_sha256"]
        == expected
    )


def test_coordinated_candidate_metadata_rehash_cannot_replace_loaded_file(report_inputs):
    broken = copy.deepcopy(report_inputs)
    wrong = "a" * 64
    broken["candidate_streams"]["candidate_file_sha256"] = wrong
    broken["control_freeze"]["bindings"]["candidates_sha256"] = wrong
    _refresh_integrity(broken)

    with pytest.raises(ValueError, match="candidate file.*hash|candidates.*binding"):
        build_artifact(**broken)


def test_tampered_candidate_file_bytes_fail_bound_hashes(report_inputs):
    broken = copy.deepcopy(report_inputs)
    broken["candidate_file_bytes"] += b"tampered\n"

    with pytest.raises(ValueError, match="candidate file.*hash|candidates.*binding"):
        build_artifact(**broken)


def test_all_evaluation_files_cannot_agree_on_wrong_manifest_hash(report_inputs):
    broken = copy.deepcopy(report_inputs)
    wrong = "a" * 64
    broken["control_freeze"]["bindings"]["manifest_sha256"] = wrong
    for name in ("stream_evaluation", "selection", "system_evaluation", "decision"):
        broken[name]["provenance"]["manifest_sha256"] = wrong
    _refresh_integrity(broken)
    # Reapply the wrong cross-binding after refresh and authentically reseal everything.
    broken["control_freeze"]["bindings"]["manifest_sha256"] = wrong
    broken["control_freeze"].pop("freeze_sha256")
    broken["control_freeze"]["freeze_sha256"] = hashlib.sha256(
        _canonical_json(broken["control_freeze"])
    ).hexdigest()
    broken["control_freeze_bytes"] = _canonical_json(broken["control_freeze"])
    for name in ("stream_evaluation", "selection", "system_evaluation", "decision"):
        broken[name]["provenance"].update(
            {
                "manifest_sha256": wrong,
                "control_freeze_sha256": broken["control_freeze"]["freeze_sha256"],
                "control_freeze_file_sha256": hashlib.sha256(
                    broken["control_freeze_bytes"]
                ).hexdigest(),
            }
        )
        broken[name] = _seal(broken[name])

    with pytest.raises(ValueError, match="manifest.*file|manifest.*binding"):
        build_artifact(**broken)


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("selection", "control_freeze_sha256"),
        ("system_evaluation", "manifest_sha256"),
        ("decision", "qrels_sha256"),
    ],
)
def test_wrong_cross_artifact_binding_fails_even_with_valid_self_hash(
    report_inputs, name, field
):
    broken = copy.deepcopy(report_inputs)
    broken[name]["provenance"][field] = "0" * 64
    broken[name] = _seal(broken[name])

    with pytest.raises(ValueError, match="provenance|binding"):
        build_artifact(**broken)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.pop("decision"), "decision"),
        (
            lambda data: data["selection"]["selected"].__setitem__("200/f07a", "W1"),
            "selection",
        ),
        (
            lambda data: data["decision"]["evidence"].__setitem__(
                "r2_ndcg_at_10", 0.99
            ),
            "decision",
        ),
        (
            lambda data: data["control_freeze"].__setitem__("status", "draft"),
            "freeze",
        ),
    ],
)
def test_missing_or_inconsistent_decision_inputs_fail_closed(report_inputs, mutate, message):
    broken = copy.deepcopy(report_inputs)
    mutate(broken)
    with pytest.raises((TypeError, ValueError), match=message):
        build_artifact(**broken)


def test_readme_documents_report_inputs_validation_and_task7_packaging():
    text = README_PATH.read_text()

    for required in (
        "candidate_streams.json",
        "stream_evaluation.json",
        "selection.json",
        "evaluation.json",
        "decision.json",
        "artifact.json",
        "report.html",
        "deliver_portable_artifact.mjs",
        "validate_artifact",
        "Cross-encoder not run",
        "canonical Task 5 selection",
        "arithmetic mean",
        "self-hash",
        "qrels SHA-256",
        "loaded `candidates.jsonl` bytes",
        "authenticated freeze ranking",
        "required exact four-topic map",
    ):
        assert required in text
