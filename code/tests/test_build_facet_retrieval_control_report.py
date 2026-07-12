import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from trec_rag.build_facet_retrieval_control_report import build_artifact
from trec_rag.facet_retrieval_control_experiment import (
    EXPECTED_ALTERNATIVE_NAMES,
    selected_ranking_references,
)
from trec_rag.facet_retrieval_control_freeze import _BASE_ARM_QUERIES


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
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


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
    base = {"O": 0.34, "F0": 0.35, "R1": 0.40, "R2": 0.39}[system_id]
    graded = {"O": 0.48, "F0": 0.49, "R1": 0.50, "R2": 0.52}[system_id]
    per_topic_ndcg = {
        "O": (0.32, 0.34, 0.35, 0.35),
        "F0": (0.33, 0.36, 0.35, 0.36),
        "R1": (0.40, 0.40, 0.40, 0.40),
        "R2": (0.41, 0.38, 0.39, 0.40),
    }[system_id]
    aggregate = {
        "ndcg@10": base,
        "graded_recall@100": graded,
        "recall@100": graded - 0.05,
        "precision@10": 0.25 + base / 10,
        "relevant_count@10": 11 + SYSTEM_IDS.index(system_id),
        "judged_rate@10": 0.80,
        "judged_rate@100": 0.55,
    }
    per_topic = {}
    for index, topic_id in enumerate(TOPIC_IDS):
        per_topic[topic_id] = {
            **aggregate,
            "ndcg@10": per_topic_ndcg[index],
            "graded_recall@100": graded + (index - 1.5) * 0.01,
            "relevant_count@10": 2 + index,
        }
    return {"metrics": aggregate, "per_topic": per_topic}


@pytest.fixture
def report_inputs():
    manifest = json.loads(MANIFEST_PATH.read_text())
    candidate_streams = _candidate_streams()
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
            "manifest_sha256": manifest["r1_manifest_sha256"],
            "prior_freeze_sha256": "b" * 64,
            "request_sha256": {},
            "response_sha256": {},
            "candidate_sha256": {},
            "ledger_sha256": {"base": "c" * 64, "r1": "d" * 64, "control": "e" * 64},
            "r1_arm_sha256": "f" * 64,
            "candidate_streams_sha256": hashlib.sha256(
                _canonical_json(candidate_streams)
            ).hexdigest(),
            "candidates_sha256": "1" * 64,
            "candidate_stream_rows_sha256": stream_hashes,
        },
        "inspection_sha256": "2" * 64,
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
        "freeze_sha256": "6" * 64,
    }
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
    selected = {
        "200/f07a": "W1",
        "225/f02": "W0",
        "225/f04": "W2",
        "707/f02": "B0",
    }
    references = selected_ranking_references(selected)
    provenance = {
        "control_freeze_file_sha256": "7" * 64,
        "control_freeze_sha256": freeze["freeze_sha256"],
        "manifest_sha256": manifest["r1_manifest_sha256"],
        "prior_freeze_sha256": freeze["bindings"]["prior_freeze_sha256"],
        "qrels_sha256": "8" * 64,
        "qrels_name": "projected_qrels.tsv",
    }
    stream_evaluation = {
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
    }
    selection = {
        "schema_version": "facet-control-selection-v1",
        "provenance": provenance,
        "selected": selected,
        "eligibility": {
            stream_id: {
                arm_id: {"eligible": True, "exclusion_reasons": []}
                for arm_id in ARM_IDS
            }
            for stream_id in STREAM_IDS
        },
        "selected_rankings": references,
        "selected_ranking_sha256": {topic_id: "9" * 64 for topic_id in TOPIC_IDS},
        "frozen_alternative_count": 25,
    }
    systems = {system_id: _system_metrics(system_id) for system_id in SYSTEM_IDS}
    deltas = {
        topic_id: systems["R2"]["per_topic"][topic_id]["ndcg@10"]
        - systems["R1"]["per_topic"][topic_id]["ndcg@10"]
        for topic_id in TOPIC_IDS
    }
    system_evaluation = {
        "schema_version": "facet-control-system-evaluation-v1",
        "provenance": provenance,
        "systems": systems,
        "r2_selected_rankings": references,
        "r2_selected_ranking_sha256": selection["selected_ranking_sha256"],
        "per_topic_ndcg_delta_vs_r1": deltas,
    }
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
    decision = {
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
    }
    return {
        "manifest": manifest,
        "control_freeze": freeze,
        "candidate_streams": candidate_streams,
        "inspection": inspections,
        "stream_evaluation": stream_evaluation,
        "selection": selection,
        "system_evaluation": system_evaluation,
        "decision": decision,
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
    rows = artifact["snapshot"]["datasets"]["query_rows"]

    assert {row["topic_id"] for row in rows} == set(TOPIC_IDS)
    assert any(row["topic_id"] == "897" and row["stream_id"] == "original" for row in rows)
    assert all(row["full_narrative"] != row["existing_query"] for row in rows if row["stream_id"] != "original")

    tampered = copy.deepcopy(report_inputs)
    original = next(
        row
        for row in tampered["candidate_streams"]["streams"]
        if row["topic_id"] == "200" and row["stream_id"] == "original"
    )
    original["query_sha256"] = "0" * 64
    tampered["control_freeze"]["bindings"]["candidate_streams_sha256"] = hashlib.sha256(
        _canonical_json(tampered["candidate_streams"])
    ).hexdigest()
    with pytest.raises(ValueError, match="narrative.*snapshot"):
        build_artifact(**tampered)


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
    system = tables["system_comparison"]
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


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.pop("decision"), "decision"),
        (
            lambda data: data["selection"]["selected"].__setitem__("200/f07a", "W2"),
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
    ):
        assert required in text
