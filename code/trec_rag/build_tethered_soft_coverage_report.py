"""Build the v3 tethered soft-coverage proxy report from approved artifacts."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from pathlib import Path

from . import tethered_facet_soft_coverage as soft_freeze
from . import tethered_facet_soft_coverage_evaluate as soft_evaluate


SCHEMA_VERSION = "tethered-facet-soft-coverage-report-v3"
TOPIC_IDS = ("219", "72", "300", "84")
ARMS = (
    "RRF",
    "NARRATIVE",
    "FIXED-O0",
    "TETHERED-DUAL",
    "TETHERED-DUAL-NR",
    "RRF100-TETHERED-DUAL",
)
DEPTHS = (100, 250, 500, 1000, 1500)
CALL_FIELDS = {
    "retrieval": "retrieval_call_count",
    "inference": "inference_count",
    "model_load": "model_load_count",
    "hosted_inference": "hosted_inference_call_count",
    "network": "network_call_count",
    "paid": "paid_call_count",
    "cost_usd": "external_cost_usd",
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, content


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or invalid")
    return value


def _verify_artifacts(
    directory: Path, summary: Mapping[str, object], names: tuple[str, ...], label: str
) -> dict[str, dict[str, object]]:
    declared = _mapping(summary.get("artifacts"), f"{label} artifacts")
    sources: dict[str, dict[str, object]] = {}
    for name in names:
        path = directory / name
        content = path.read_bytes()
        binding = _mapping(declared.get(name), f"{label} {name} binding")
        digest = _sha256(content)
        if binding.get("sha256") != digest or binding.get("bytes") != len(content):
            raise ValueError(f"{label} {name} differs from its approved hash")
        sources[f"{label.lower()}_{name.replace('.', '_')}"] = {
            "label": f"{label} {name}",
            "bytes": len(content),
            "sha256": digest,
        }
    return sources


def _external_calls(summary: Mapping[str, object], label: str) -> dict[str, object]:
    calls = {
        output_name: summary.get(source_name)
        for output_name, source_name in CALL_FIELDS.items()
    }
    expected = {name: (0.0 if name == "cost_usd" else 0) for name in CALL_FIELDS}
    if calls != expected:
        raise ValueError(f"{label} does not have a zero-call receipt")
    return calls


def _bound_path(binding: Mapping[str, object], label: str) -> Path:
    path = Path(str(binding.get("path", "")))
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} binding is unreadable or unsafe")
    return path


def _overlap_decomposition(
    evaluation_bindings: Mapping[str, object], relevance_threshold: int
) -> tuple[list[dict[str, object]], dict[str, int]]:
    qrels_binding = _mapping(
        evaluation_bindings.get("qrels_projection"), "qrels projection binding"
    )
    union_binding = _mapping(
        evaluation_bindings.get("accepted_union"), "accepted union binding"
    )
    qrels_path = _bound_path(qrels_binding, "qrels projection")
    union_path = _bound_path(union_binding, "accepted union")

    relevant: dict[str, set[str]] = {topic: set() for topic in TOPIC_IDS}
    grade_counts = {
        "grade_0": 0,
        "grade_1_excluded": 0,
        "binary_relevant_grade_2_plus": 0,
    }
    qrels_digest = hashlib.sha256()
    qrels_rows = 0
    with qrels_path.open("rb") as handle:
        for raw_line in handle:
            qrels_digest.update(raw_line)
            qrels_rows += 1
            row = json.loads(raw_line)
            topic = str(row.get("topic_id"))
            grade = int(row.get("grade", 0))
            if grade == 0:
                grade_counts["grade_0"] += 1
            elif grade == 1:
                grade_counts["grade_1_excluded"] += 1
            elif grade >= relevance_threshold:
                grade_counts["binary_relevant_grade_2_plus"] += 1
            else:
                raise ValueError("qrels grade is outside the report contract")
            if topic in relevant and grade >= relevance_threshold:
                relevant[topic].add(str(row.get("document_id")))
    if (
        qrels_binding.get("sha256") != qrels_digest.hexdigest()
        or qrels_binding.get("rows") != qrels_rows
    ):
        raise ValueError("qrels projection differs from its approved binding")

    counts = {
        topic: {"original_only": 0, "facet_only": 0, "overlap": 0}
        for topic in TOPIC_IDS
    }
    union_digest = hashlib.sha256()
    union_rows = 0
    with union_path.open("rb") as handle:
        for raw_line in handle:
            union_digest.update(raw_line)
            union_rows += 1
            row = json.loads(raw_line)
            topic = str(row.get("topic_id"))
            document = str(row.get("document_id"))
            if topic not in relevant or document not in relevant[topic]:
                continue
            provenance = row.get("provenance")
            if not isinstance(provenance, list):
                raise ValueError("accepted union provenance is invalid")
            families = {
                str(item.get("family"))
                for item in provenance
                if isinstance(item, Mapping)
            }
            original = "original" in families
            facet = "facet" in families
            if original and facet:
                counts[topic]["overlap"] += 1
            elif original:
                counts[topic]["original_only"] += 1
            elif facet:
                counts[topic]["facet_only"] += 1
            else:
                raise ValueError("relevant union row lacks original/facet provenance")
    if (
        union_binding.get("sha256") != union_digest.hexdigest()
        or union_binding.get("rows") != union_rows
    ):
        raise ValueError("accepted union differs from its approved binding")

    output: list[dict[str, object]] = []
    for topic in TOPIC_IDS:
        original_only = counts[topic]["original_only"]
        facet_only = counts[topic]["facet_only"]
        overlap = counts[topic]["overlap"]
        union_relevant = original_only + facet_only + overlap
        output.append(
            {
                "topic_id": topic,
                "original_relevant": original_only + overlap,
                "facet_relevant": facet_only + overlap,
                "original_only_relevant": original_only,
                "overlap_relevant": overlap,
                "facet_only_relevant": facet_only,
                "incremental_relevant": facet_only,
                "union_relevant": union_relevant,
                "missed_relevant": len(relevant[topic]) - union_relevant,
                "total_relevant": len(relevant[topic]),
            }
        )
    return output, grade_counts


def build_report_payload(
    freeze: Path | str, evaluation: Path | str, prior_summary: Path | str
) -> dict[str, object]:
    """Authenticate approved inputs and return a sanitized report payload."""

    freeze_dir = Path(freeze)
    evaluation_dir = Path(evaluation)
    prior_path = Path(prior_summary)
    verified_freeze = soft_freeze.verify_soft_freeze(freeze_dir)
    verified_evaluation = soft_evaluate.verify_evaluation(evaluation_dir)
    freeze_summary, freeze_summary_bytes = _read_json(
        freeze_dir / "summary.json", "freeze summary"
    )
    if verified_freeze != freeze_summary:
        raise ValueError("verified freeze summary differs from the loaded summary")
    evaluation_summary, evaluation_summary_bytes = _read_json(
        evaluation_dir / "summary.json", "evaluation summary"
    )
    if verified_evaluation != evaluation_summary:
        raise ValueError("verified evaluation summary differs from the loaded summary")
    metrics, metrics_bytes = _read_json(evaluation_dir / "metrics.json", "metrics")
    diagnostics, diagnostics_bytes = _read_json(
        evaluation_dir / "diagnostics.json", "diagnostics"
    )
    evaluation_bindings, evaluation_bindings_bytes = _read_json(
        evaluation_dir / "input_bindings.json", "evaluation bindings"
    )
    prior, prior_bytes = _read_json(prior_path, "prior v2 summary")

    if freeze_summary.get("status") != "rankings_frozen_before_evaluation":
        raise ValueError("freeze is not the approved pre-evaluation snapshot")
    if evaluation_summary.get("status") != "complete":
        raise ValueError("evaluation is incomplete")
    for label, value in (
        ("freeze", freeze_summary),
        ("evaluation", evaluation_summary),
        ("metrics", metrics),
    ):
        if value.get("topic_ids") != list(TOPIC_IDS):
            raise ValueError(f"{label} topic set differs")
    if prior.get("schema_version") != "tethered-facet-diagnostic-report-v2":
        raise ValueError("prior summary is not v2")
    if metrics.get("arms") is None or set(_mapping(metrics["arms"], "metric arms")) != set(ARMS):
        raise ValueError("evaluation arm set differs")
    relevance_threshold = metrics.get("relevance_threshold")
    if type(relevance_threshold) is not int or relevance_threshold != 2:
        raise ValueError("binary relevance threshold differs from qrels grade >= 2")

    sources = {
        "freeze_summary_json": {
            "label": "Approved freeze summary.json",
            "bytes": len(freeze_summary_bytes),
            "sha256": _sha256(freeze_summary_bytes),
        },
        "evaluation_summary_json": {
            "label": "Approved evaluation summary.json",
            "bytes": len(evaluation_summary_bytes),
            "sha256": _sha256(evaluation_summary_bytes),
        },
        "prior_v2_summary_json": {
            "label": "Prior v2 summary.json",
            "bytes": len(prior_bytes),
            "sha256": _sha256(prior_bytes),
        },
    }
    sources.update(
        _verify_artifacts(
            freeze_dir,
            freeze_summary,
            ("input_bindings.json", "parameters.json", "rankings.jsonl"),
            "Freeze",
        )
    )
    evaluation_seal, evaluation_seal_content = _read_json(
        evaluation_dir / "SEALED.json", "evaluation seal"
    )
    sources["evaluation_sealed_json"] = {
        "label": "Approved evaluation SEALED.json",
        "bytes": len(evaluation_seal_content),
        "sha256": _sha256(evaluation_seal_content),
    }
    sources.update(
        _verify_artifacts(
            evaluation_dir,
            evaluation_summary,
            ("input_bindings.json", "metrics.json", "diagnostics.json"),
            "Evaluation",
        )
    )
    seal, seal_content = _read_json(freeze_dir / "SEALED.json", "freeze seal")
    sources["freeze_sealed_json"] = {
        "label": "Approved freeze SEALED.json",
        "bytes": len(seal_content),
        "sha256": _sha256(seal_content),
    }
    freeze_binding = _mapping(evaluation_bindings.get("freeze"), "evaluation freeze binding")
    if (
        seal.get("status") != "sealed_before_evaluation"
        or seal.get("root_sha256") != freeze_binding.get("seal_root_sha256")
    ):
        raise ValueError("freeze seal does not match the approved evaluation binding")
    verified_artifacts = _mapping(
        verified_freeze.get("artifacts"), "verified freeze artifacts"
    )
    evaluation_rankings = _mapping(
        evaluation_bindings.get("rankings"), "evaluation rankings binding"
    )
    verified_rankings = _mapping(
        verified_artifacts.get("rankings.jsonl"), "verified rankings binding"
    )
    if (
        evaluation_rankings.get("bytes") != verified_rankings.get("bytes")
        or evaluation_rankings.get("sha256") != verified_rankings.get("sha256")
    ):
        raise ValueError("verified freeze rankings differ from the approved evaluation binding")
    if sources["evaluation_metrics_json"]["sha256"] != _sha256(metrics_bytes):
        raise ValueError("metrics source hash differs")
    if sources["evaluation_diagnostics_json"]["sha256"] != _sha256(diagnostics_bytes):
        raise ValueError("diagnostics source hash differs")
    if sources["evaluation_input_bindings_json"]["sha256"] != _sha256(
        evaluation_bindings_bytes
    ):
        raise ValueError("evaluation bindings source hash differs")

    freeze_calls = _external_calls(freeze_summary, "freeze")
    evaluation_calls = _external_calls(evaluation_summary, "evaluation")
    if freeze_calls != evaluation_calls:
        raise ValueError("freeze and evaluation zero-call receipts differ")

    arm_metrics: list[dict[str, object]] = []
    arms = _mapping(metrics["arms"], "metric arms")
    for arm in ARMS:
        aggregate = _mapping(
            _mapping(arms.get(arm), f"{arm} metrics").get("aggregate"),
            f"{arm} aggregate metrics",
        )
        record: dict[str, object] = {
            "arm": arm,
            "ndcg@10": aggregate["ndcg@10"],
            "ndcg@100": aggregate["ndcg@100"],
            "ndcg@1000": aggregate["ndcg@1000"],
            "ndcg@500": aggregate["ndcg@500"],
            "ndcg@1500": aggregate["ndcg@1500"],
            "recall_auc": aggregate["recall_auc"],
            "relevant_count_full": aggregate["relevant_count_full"],
            "binary_recall_full": aggregate["binary_recall_full"],
            "graded_recall_full": aggregate["graded_recall_full"],
            "facet_only_relevant_retained_full": aggregate["facet_only_relevant_retained_full"],
            "facet_only_relevant_retention_full": aggregate["facet_only_relevant_retention_full"],
        }
        for depth in DEPTHS:
            record[f"relevant_count@{depth}"] = aggregate[f"relevant_count@{depth}"]
            record[f"binary_recall@{depth}"] = aggregate[f"binary_recall@{depth}"]
            record[f"graded_recall@{depth}"] = aggregate[f"graded_recall@{depth}"]
            record[f"facet_only_relevant_retained@{depth}"] = aggregate[
                f"facet_only_relevant_retained@{depth}"
            ]
            record[f"facet_only_relevant_retention@{depth}"] = aggregate[
                f"facet_only_relevant_retention@{depth}"
            ]
        arm_metrics.append(record)

    by_arm = {str(row["arm"]): row for row in arm_metrics}
    rrf = by_arm["RRF"]
    protected = by_arm["RRF100-TETHERED-DUAL"]
    total_relevant = int(
        _mapping(
            _mapping(arms["RRF"], "RRF metrics")["aggregate"], "RRF aggregate"
        )["total_relevant"]
    )
    findings = {
        "rrf_relevant_at_1000": rrf["relevant_count@1000"],
        "protected_relevant_at_1000": protected["relevant_count@1000"],
        "relevant_delta_at_1000": int(protected["relevant_count@1000"])
        - int(rrf["relevant_count@1000"]),
        "rrf_facet_only_at_1000": rrf["facet_only_relevant_retained@1000"],
        "protected_facet_only_at_1000": protected[
            "facet_only_relevant_retained@1000"
        ],
        "rrf_relevant_at_100": rrf["relevant_count@100"],
        "protected_relevant_at_100": protected["relevant_count@100"],
        "rrf_ndcg_at_100": rrf["ndcg@100"],
        "protected_ndcg_at_100": protected["ndcg@100"],
        "full_union_relevant": protected["relevant_count_full"],
        "total_relevant": total_relevant,
        "full_union_recall": protected["binary_recall_full"],
    }
    if findings != {
        **findings,
        "rrf_relevant_at_1000": 712,
        "protected_relevant_at_1000": 764,
        "relevant_delta_at_1000": 52,
        "rrf_facet_only_at_1000": 95,
        "protected_facet_only_at_1000": 160,
        "rrf_relevant_at_100": 299,
        "protected_relevant_at_100": 299,
        "rrf_ndcg_at_100": findings["protected_ndcg_at_100"],
        "full_union_relevant": 875,
        "total_relevant": 2817,
        "full_union_recall": 875 / 2817,
    }:
        raise ValueError("approved headline findings differ")

    topic_deltas: list[dict[str, object]] = []
    protected_topics = _mapping(arms["RRF100-TETHERED-DUAL"], "protected metrics")
    protected_topics = _mapping(protected_topics.get("per_topic"), "protected topics")
    rrf_topics = _mapping(_mapping(arms["RRF"], "RRF metrics").get("per_topic"), "RRF topics")
    for topic in TOPIC_IDS:
        candidate = _mapping(protected_topics[topic], f"protected topic {topic}")
        baseline = _mapping(rrf_topics[topic], f"RRF topic {topic}")
        topic_deltas.append(
            {
                "topic_id": topic,
                **{
                    f"relevant_delta@{depth}": int(candidate[f"relevant_count@{depth}"])
                    - int(baseline[f"relevant_count@{depth}"])
                    for depth in DEPTHS
                },
                "ndcg_delta@100": float(candidate["ndcg@100"])
                - float(baseline["ndcg@100"]),
                "ndcg_delta@1000": float(candidate["ndcg@1000"])
                - float(baseline["ndcg@1000"]),
                "recall_auc_delta": float(candidate["recall_auc"])
                - float(baseline["recall_auc"]),
            }
        )

    overlap, qrels_grade_counts = _overlap_decomposition(
        evaluation_bindings, relevance_threshold
    )
    if sum(int(row["union_relevant"]) for row in overlap) != 875:
        raise ValueError("overlap decomposition does not reconcile to full union")
    if sum(int(row["facet_only_relevant"]) for row in overlap) != 177:
        raise ValueError("overlap decomposition does not reconcile to facet-only total")
    if qrels_grade_counts != {
        "grade_0": 360,
        "grade_1_excluded": 1456,
        "binary_relevant_grade_2_plus": 2817,
    }:
        raise ValueError("qrels grade distribution differs from the approved evaluation")
    for source_id, binding_name, label in (
        ("accepted_union_jsonl", "accepted_union", "Authenticated accepted candidate union"),
        ("qrels_projection_jsonl", "qrels_projection", "Authenticated qrels projection"),
    ):
        binding = _mapping(evaluation_bindings.get(binding_name), f"{label} binding")
        sources[source_id] = {
            "label": label,
            "bytes": int(binding["bytes"]),
            "sha256": str(binding["sha256"]),
        }

    proxy_root = _mapping(diagnostics.get("coverage_proxy"), "coverage proxy")
    attribution = _mapping(
        proxy_root.get("qrels_positive_facet_attribution"), "facet attribution"
    )
    proxy_by_topic = []
    proxy_sensitivity = []
    attribution_by_arm = {
        arm: _mapping(attribution.get(arm), f"{arm} facet attribution")
        for arm in ("TETHERED-DUAL", "TETHERED-DUAL-NR")
    }
    for topic in TOPIC_IDS:
        facet_counts = _mapping(attribution_by_arm["TETHERED-DUAL"].get(topic), f"topic {topic} facets")
        dual_count = sum(int(value) for value in facet_counts.values())
        nr_counts = _mapping(attribution_by_arm["TETHERED-DUAL-NR"].get(topic), f"topic {topic} NR facets")
        nr_count = sum(int(value) for value in nr_counts.values())
        proxy_by_topic.append(
            {
                "topic_id": topic,
                "binary_relevant_attributions": dual_count,
                "facets_with_positive_attribution": sum(
                    1 for value in facet_counts.values() if int(value) > 0
                ),
            }
        )
        proxy_sensitivity.append({
            "topic_id": topic,
            "tethered_dual_attributions": dual_count,
            "tethered_dual_nr_attributions": nr_count,
            "delta_without_redundancy": nr_count - dual_count,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "title": "Protected RRF head improves deep soft coverage, not total candidate recall",
        "topic_ids": list(TOPIC_IDS),
        "relevance_threshold": relevance_threshold,
        "qrels_grade_counts": qrels_grade_counts,
        "findings": findings,
        "arm_metrics": arm_metrics,
        "topic_deltas": topic_deltas,
        "overlap_decomposition": overlap,
        "facet_proxy_by_topic": proxy_by_topic,
        "facet_proxy_sensitivity": proxy_sensitivity,
        "metric_definitions": {
            "binary_recall": {
                "aggregation": "pooled_micro",
                "definition": "Unique binary-relevant documents (qrels grade >= 2) retrieved across four topics divided by all binary-relevant documents across those topics.",
            },
            "ndcg": {
                "aggregation": "macro_topic_mean",
                "definition": "Mean topic-level normalized discounted cumulative gain; it rewards graded relevance near the top.",
            },
            "recall_auc": {
                "aggregation": "macro_topic_mean",
                "definition": "For each topic, the mean binary recall at every rank from 1 through that topic's full candidate depth; then the arithmetic mean of those topic values.",
            },
            "facet_proxy": {
                "aggregation": "binary_relevant_facet_attribution_count",
                "definition": "A binary-relevant document (qrels grade >= 2) whose marginal admission is attributed to a facet stream; it does not prove passage support for that facet.",
            },
        },
        "source_hashes": sources,
        "external_calls": freeze_calls,
        "freeze_verification": {
            "status": verified_freeze["status"],
            "ranking_row_count": verified_freeze["ranking_row_count"],
            "seal_root_sha256": seal["root_sha256"],
            "evaluation_ranking_sha256": evaluation_rankings["sha256"],
        },
        "evaluation_verification": {
            "status": verified_evaluation["status"],
            "seal_root_sha256": evaluation_seal["root_sha256"],
        },
        "prior_v2": {
            "schema_version": prior["schema_version"],
            "status": prior.get("status"),
            "new_retrieval": prior.get("new_retrieval"),
        },
        "report_boundaries": {
            "post_qrels_diagnostic": True,
            "new_retrieval": False,
            "answer_generation_evaluation": False,
            "true_nugget_coverage": False,
            "production_validation": False,
        },
    }


def _pct(value: object, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def _signed(value: object, digits: int = 0) -> str:
    number = float(value)
    if digits == 0:
        return f"{int(number):+d}"
    return f"{number:+.{digits}f}"


def _table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    head = "".join(f'<th scope="col">{html.escape(value)}</th>' for value in headers)
    body = "".join(
        "<tr>"
        + "".join(
            (
                f'<th scope="row">{html.escape(cell)}</th>'
                if index == 0
                else f"<td>{html.escape(cell)}</td>"
            )
            for index, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap" tabindex="0">'
        f"<table><caption>{html.escape(caption)}</caption><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def render_report(payload: Mapping[str, object]) -> str:
    """Render a self-contained, responsive, accessible HTML report."""

    findings = _mapping(payload["findings"], "findings")
    arm_metrics = list(payload["arm_metrics"])
    by_arm = {str(row["arm"]): row for row in arm_metrics}
    recall_rows = []
    for arm in ARMS:
        row = by_arm[arm]
        recall_rows.append(
            [
                arm,
                *[f'{row[f"relevant_count@{depth}"]} ({_pct(row[f"binary_recall@{depth}"], 2)})' for depth in DEPTHS],
                f'{row["relevant_count_full"]} ({_pct(row["binary_recall_full"], 4)})',
            ]
        )
    graded_rows = [[arm, *[_pct(by_arm[arm][f"graded_recall@{depth}"], 2) for depth in DEPTHS], _pct(by_arm[arm]["graded_recall_full"], 4)] for arm in ARMS]
    retention_rows = [[arm, *[f'{by_arm[arm][f"facet_only_relevant_retained@{depth}"]} ({_pct(by_arm[arm][f"facet_only_relevant_retention@{depth}"], 1)})' for depth in DEPTHS], f'{by_arm[arm]["facet_only_relevant_retained_full"]} ({_pct(by_arm[arm]["facet_only_relevant_retention_full"], 1)})'] for arm in ARMS]
    quality_rows = [
        [
            arm,
            f'{float(by_arm[arm]["ndcg@10"]):.4f}',
            f'{float(by_arm[arm]["ndcg@100"]):.4f}',
            f'{float(by_arm[arm]["ndcg@500"]):.4f}',
            f'{float(by_arm[arm]["ndcg@1000"]):.4f}',
            f'{float(by_arm[arm]["ndcg@1500"]):.4f}',
            f'{float(by_arm[arm]["recall_auc"]):.4f}',
        ]
        for arm in ARMS
    ]
    overlap_rows = [
        [
            str(row["topic_id"]),
            str(row["original_relevant"]),
            str(row["facet_relevant"]),
            str(row["overlap_relevant"]),
            str(row["original_only_relevant"]),
            str(row["facet_only_relevant"]),
            str(row["union_relevant"]),
            str(row["missed_relevant"]),
        ]
        for row in payload["overlap_decomposition"]
    ]
    topic_rows = [
        [
            str(row["topic_id"]),
            *[_signed(row[f"relevant_delta@{depth}"]) for depth in DEPTHS],
            _signed(float(row["ndcg_delta@100"]) * 100, 2) + " pp",
            _signed(float(row["ndcg_delta@1000"]) * 100, 2) + " pp",
        ]
        for row in payload["topic_deltas"]
    ]
    proxy_rows = [
        [
            str(row["topic_id"]),
            str(row["binary_relevant_attributions"]),
            str(row["facets_with_positive_attribution"]),
        ]
        for row in payload["facet_proxy_by_topic"]
    ]
    sensitivity_rows = [[str(row["topic_id"]), str(row["tethered_dual_attributions"]), str(row["tethered_dual_nr_attributions"]), _signed(row["delta_without_redundancy"])] for row in payload["facet_proxy_sensitivity"]]
    source_rows = [
        [str(source["label"]), str(source["bytes"]), str(source["sha256"])]
        for source in payload["source_hashes"].values()
    ]
    call_rows = [[name.replace("_", " ").title(), str(value)] for name, value in payload["external_calls"].items()]

    recall_table = _table(
        ["Arm", "@100", "@250", "@500", "@1,000", "@1,500", "Full"],
        recall_rows,
        "Pooled binary-relevant document counts by depth (qrels grade >= 2); the full binary-relevant union is identical for every reordering arm.",
    )
    graded_table = _table(["Arm", "@100", "@250", "@500", "@1,000", "@1,500", "Full"], graded_rows, "Pooled graded-gain recall by depth for every arm.")
    retention_table = _table(["Arm", "@100", "@250", "@500", "@1,000", "@1,500", "Full"], retention_rows, "Facet-only binary-relevant documents retained (count and percent of the 177 facet-only total).")
    quality_table = _table(
        ["Arm", "nDCG@10", "nDCG@100", "nDCG@500", "nDCG@1,000", "nDCG@1,500", "Recall AUC"],
        quality_rows,
        "Macro mean across four topics. nDCG and recall AUC are not pooled document recall.",
    )
    overlap_table = _table(
        ["Topic", "Original binary-relevant", "Facet binary-relevant", "Overlap", "Original only", "Facet only / incremental", "Binary-relevant union", "Missed binary-relevant"],
        overlap_rows,
        "Nonadditive overlap decomposition. Original relevant and facet relevant both include overlap; union equals original only + overlap + facet only.",
    )
    topic_table = _table(
        ["Topic", "Δ @100", "Δ @250", "Δ @500", "Δ @1,000", "Δ @1,500", "Δ nDCG@100", "Δ nDCG@1,000"],
        topic_rows,
        "RRF100-TETHERED-DUAL minus RRF. Positive binary-relevant document deltas are gains; negative deltas are regressions.",
    )
    proxy_table = _table(
        ["Topic", "Binary-relevant attributions", "Facets represented"],
        proxy_rows,
        "TETHERED-DUAL marginal facet attribution among binary-relevant documents (qrels grade >= 2); this is a proxy, not semantic facet support.",
    )
    sensitivity_table = _table(["Topic", "TETHERED-DUAL", "TETHERED-DUAL-NR", "Δ without redundancy"], sensitivity_rows, "Sensitivity of binary-relevant facet attribution to the DUAL no-redundancy variant.")
    source_table = _table(["Source", "Bytes", "SHA-256"], source_rows, "Exact canonical source identities used to build this report.")
    call_table = _table(["External operation", "Count / cost"], call_rows, "Freeze and evaluation receipts agree that all external operations and cost were zero.")

    title = html.escape(str(payload["title"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
<style>
:root {{ --bg:#f4f7fb; --paper:#fff; --ink:#172033; --muted:#526176; --line:#c8d2df; --navy:#123b66; --blue:#1769aa; --cyan:#0b7285; --good:#146c43; --warn:#8a4b08; --focus:#ffbf47; }}
@media (prefers-color-scheme:dark) {{ :root {{ --bg:#0e1621; --paper:#162231; --ink:#edf4fb; --muted:#b7c5d3; --line:#415367; --navy:#8fc5f2; --blue:#69b7ef; --cyan:#6bd5e1; --good:#72d59e; --warn:#ffc078; --focus:#ffd166; }} }}
* {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.58 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
a {{ color:var(--blue); }} a:focus-visible, [tabindex]:focus-visible {{ outline:3px solid var(--focus); outline-offset:3px; }}
.skip-link {{ position:absolute; left:1rem; top:-5rem; padding:.7rem 1rem; background:var(--paper); color:var(--ink); z-index:10; }} .skip-link:focus {{ top:1rem; }}
header {{ background:linear-gradient(125deg,#102f52,#0b6979); color:#fff; }} .hero {{ max-width:1120px; margin:auto; padding:4.5rem 1.25rem 4rem; }}
.eyebrow {{ font-size:.78rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; opacity:.82; }} h1 {{ max-width:900px; margin:.35rem 0 1rem; font-size:clamp(2rem,5vw,4rem); line-height:1.05; letter-spacing:-.035em; }}
.hero p {{ max-width:780px; margin:0; font-size:1.18rem; }} main {{ max-width:1120px; margin:auto; padding:2rem 1.25rem 5rem; }}
section {{ margin:0 0 3.2rem; scroll-margin-top:1rem; }} h2 {{ margin:0 0 .8rem; color:var(--navy); font-size:clamp(1.55rem,3vw,2.25rem); line-height:1.15; }} h3 {{ margin-top:1.7rem; }}
.summary {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; margin:1.2rem 0; }} .card {{ background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:1.15rem; box-shadow:0 8px 24px rgb(15 40 65 / .07); }}
.metric {{ display:block; color:var(--navy); font-size:2rem; font-weight:850; line-height:1.05; }} .label {{ color:var(--muted); font-size:.9rem; }}
.callout {{ border-left:5px solid var(--cyan); background:var(--paper); padding:1rem 1.2rem; border-radius:0 12px 12px 0; }} .warning {{ border-left-color:var(--warn); }}
.table-wrap {{ overflow-x:auto; margin:1rem 0; border:1px solid var(--line); border-radius:12px; background:var(--paper); }} table {{ border-collapse:collapse; width:100%; min-width:720px; }} caption {{ padding:1rem; text-align:left; color:var(--muted); font-size:.92rem; }} th,td {{ padding:.72rem .85rem; border-top:1px solid var(--line); text-align:right; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }} thead th {{ color:var(--navy); background:color-mix(in srgb,var(--paper) 86%,var(--blue)); border-top:0; }} tbody tr:hover {{ background:color-mix(in srgb,var(--paper) 94%,var(--cyan)); }}
.hashes table {{ min-width:900px; }} .hashes td:last-child {{ font:12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace; }}
.tag {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:.22rem .65rem; margin:.2rem .3rem .2rem 0; color:var(--muted); font-size:.82rem; }}
footer {{ border-top:1px solid var(--line); color:var(--muted); }} .footer-inner {{ max-width:1120px; margin:auto; padding:1.5rem 1.25rem; }}
@media (max-width:760px) {{ .hero {{ padding:3.2rem 1rem 2.8rem; }} main {{ padding:1.5rem 1rem 4rem; }} .summary {{ grid-template-columns:1fr; }} section {{ margin-bottom:2.5rem; }} .table-wrap {{ margin-inline:-1rem; border-radius:0; border-inline:0; }} th,td {{ padding:.65rem .7rem; }} }}
@media (prefers-reduced-motion:reduce) {{ html {{ scroll-behavior:auto; }} *,*::before,*::after {{ animation-duration:.01ms!important; transition-duration:.01ms!important; }} }}
@media print {{ body {{ background:#fff; color:#111; }} header {{ background:#fff; color:#111; }} .card,.table-wrap {{ box-shadow:none; break-inside:avoid; }} }}
</style>
</head>
<body>
<a class="skip-link" href="#main">Skip to main content</a>
<header><div class="hero"><div class="eyebrow">TREC RAG 2026 · post-qrels diagnostic · v3</div><h1>{title}</h1><p>Protecting RRF’s first 100 results and using tethered DUAL below that boundary moved more binary-relevant documents (qrels grade &gt;= 2) into the first 1,000 without changing the candidate set.</p></div></header>
<main id="main" tabindex="-1">
<section aria-labelledby="summary-title"><h2 id="summary-title">Technical summary</h2>
<div class="summary"><div class="card"><span class="metric">712 → 764</span><span class="label">pooled binary-relevant documents at 1,000 (+52)</span></div><div class="card"><span class="metric">95 → 160</span><span class="label">facet-only binary-relevant documents retained at 1,000</span></div><div class="card"><span class="metric">31.0614%</span><span class="label">exhaustive binary-relevant full-union recall</span></div></div>
<p><strong>Finding.</strong> RRF100-TETHERED-DUAL is the best supported next ranking configuration for this fixed candidate union: its protected RRF head preserved both 299 binary-relevant documents and macro nDCG 0.4048 at depth 100, while soft tethering increased the pooled binary-relevant count at 1,000 by 52.</p>
<p><strong>Next action.</strong> Improve candidate generation on fresh preregistered topics, then evaluate answer generation and nugget support in the separate answer-generation worktree. Reordering alone cannot add any of the 1,942 binary-relevant documents missing from this union.</p></section>

<section aria-labelledby="depth-title"><h2 id="depth-title">The protected head trades small mid-depth regressions for the best result at 1,000</h2><p>Read the tables across each arm. Counts are pooled across four topics; percentages use the matching pooled denominator.</p><div aria-label="Recall-depth comparison">{recall_table}</div>{graded_table}{retention_table}<p>RRF100-TETHERED-DUAL exactly matches RRF at 100, trails by 9 at 250, leads by 10 at 500, leads by 52 at 1,000, and leads by 31 at 1,500. Every arm reaches the same 875 binary-relevant documents at the full union because these arms only reorder the same candidates.</p></section>

<section aria-labelledby="quality-title"><h2 id="quality-title">Ranking quality and pooled recall answer different questions</h2><p>nDCG is a macro mean: each topic contributes equally, and higher grades near the top matter more. Recall AUC is also a macro topic mean. Neither should be added to, averaged with, or described as pooled document recall.</p>{quality_table}<div class="callout"><strong>Interpretation boundary.</strong> The protected arm preserves RRF at @100 by construction. Its higher macro nDCG@1,000 (0.3103 versus 0.2908) and recall AUC (0.2398 versus 0.2310) support the deep-ranking result, but do not establish production generalization.</div></section>

<section aria-labelledby="overlap-title"><h2 id="overlap-title">Facet retrieval adds 177 binary-relevant candidates, but the union still misses 1,942</h2><p>The original and facet columns are nonadditive because documents found by both appear in the overlap. The exact decomposition is shown as a semantic table: original only + overlap + facet only equals the binary-relevant union. “Incremental” and “facet only” are the same set here.</p>{overlap_table}<p>The four-topic union contains 875 / 2,817 binary-relevant documents (qrels grade &gt;= 2). That is exhaustive binary-relevant document recall of 31.0614%, leaving 1,942 outside the candidate set. The qrels projection also contains 1,456 grade-1 documents; they are excluded from this binary-relevance denominator. No scoring rule or reordering can raise the binary-relevant ceiling.</p></section>

<section aria-labelledby="topics-title"><h2 id="topics-title">Deep gains are real but uneven by topic</h2><p>These deltas compare the protected arm directly with RRF. The preserved @100 values are all zero by construction; later gains and regressions reveal where the tethered tail helps or displaces relevant material.</p>{topic_table}</section>

<section aria-labelledby="proxy-title"><h2 id="proxy-title">Facet exposure is diagnostic evidence, not nugget coverage</h2><p><strong>binary-relevant facet exposure is only a proxy.</strong> It counts a binary-relevant document (qrels grade &gt;= 2) whose marginal admission is attributed to a facet stream. It does not inspect whether the document actually supports that facet, whether a passage contains the needed nugget, or whether a generated answer uses it correctly.</p>{proxy_table}{sensitivity_table}<p>Topic 72 shows the strongest proxy behavior (32 binary-relevant attributions across seven represented facets); topic 300 is weakest (one binary-relevant attribution across one facet). The no-redundancy comparison shows whether that attribution is sensitive to DUAL's redundancy term. This points to a remaining candidate-generation gap rather than a ranking-only problem.</p><div class="callout warning"><strong>This is not answer-generation evaluation.</strong> The 31.1% exhaustive binary-relevant document recall is low, but it is not equivalent to 31.1% answer coverage. One document may support multiple answer nuggets, several documents may repeat the same nugget, and qrels judge topical relevance rather than final answer completeness or faithfulness.</div></section>

<section aria-labelledby="scope-title"><h2 id="scope-title">Scope and metric definitions</h2><p><span class="tag">4 historical diagnostic topics</span><span class="tag">post-qrels</span><span class="tag">binary-relevant (qrels grade &gt;= 2)</span><span class="tag">fixed 8,114-document union</span><span class="tag">no new retrieval</span><span class="tag">not production validation</span></p>
<h3>Pooled binary recall</h3><p>Unique binary-relevant documents retrieved across all four topics divided by all 2,817 binary-relevant documents. Binary-relevant means qrels grade &gt;= 2. The complete qrels distribution is 360 grade-0, 1,456 grade-1 excluded, and 2,817 grade &gt;= 2 binary-relevant documents. Counts at a depth are pooled micro totals.</p><h3>Macro nDCG and recall AUC</h3><p>nDCG is calculated per topic and then averaged across four topics. Recall AUC is calculated for each topic as the mean binary recall at every rank from 1 through that topic's full candidate depth, then those four topic values are averaged. nDCG rewards graded relevance near the top.</p><h3>True RAG evaluation</h3><p>Answer completeness, supported nuggets, citation correctness, faithfulness, and end-to-end answer quality are out of scope here and belong to the separate answer-generation worktree.</p></section>

<section class="hashes" aria-labelledby="sources-title"><h2 id="sources-title">Sources and reproducibility</h2><p>The report authenticates the independently approved freeze and evaluation, plus the prior v2 summary. Labels and hashes are included; machine-local paths, raw documents, document identifiers, credentials, and request material are excluded.</p>{source_table}{call_table}<p>The ranking freeze and evaluation receipts each record zero retrieval, inference, model loads, hosted inference, network, paid calls, and external cost. The report build itself is offline and deterministic apart from SQLite container bytes.</p></section>
</main>
<footer><div class="footer-inner">Tethered facet soft-coverage proxy · canonical v3 report · source-bound and offline</div></footer>
</body></html>"""


def _write_sqlite(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE arm_metrics (
              arm TEXT PRIMARY KEY, binary_relevant_100 INTEGER,
              binary_relevant_250 INTEGER, binary_relevant_500 INTEGER,
              binary_relevant_1000 INTEGER, binary_relevant_1500 INTEGER,
              binary_relevant_full INTEGER, ndcg_10 REAL, ndcg_100 REAL,
              ndcg_1000 REAL, recall_auc REAL
            );
            CREATE TABLE arm_depth_metrics (
              arm TEXT, depth TEXT, binary_relevant_count INTEGER,
              binary_recall REAL, graded_recall REAL,
              facet_only_relevant_retained INTEGER, facet_only_retention REAL,
              PRIMARY KEY (arm, depth)
            );
            CREATE TABLE arm_quality_metrics (
              arm TEXT PRIMARY KEY, ndcg_10 REAL, ndcg_100 REAL,
              ndcg_500 REAL, ndcg_1000 REAL, ndcg_1500 REAL, recall_auc REAL
            );
            CREATE TABLE topic_deltas (
              topic_id TEXT PRIMARY KEY, binary_relevant_delta_100 INTEGER,
              binary_relevant_delta_250 INTEGER, binary_relevant_delta_500 INTEGER,
              binary_relevant_delta_1000 INTEGER, binary_relevant_delta_1500 INTEGER,
              ndcg_delta_100 REAL, ndcg_delta_1000 REAL, recall_auc_delta REAL
            );
            CREATE TABLE overlap_decomposition (
              topic_id TEXT PRIMARY KEY, original_binary_relevant INTEGER,
              facet_binary_relevant INTEGER, overlap_binary_relevant INTEGER,
              original_only_binary_relevant INTEGER, facet_only_binary_relevant INTEGER,
              incremental_binary_relevant INTEGER, binary_relevant_union INTEGER,
              missed_binary_relevant INTEGER, total_binary_relevant INTEGER
            );
            CREATE TABLE facet_proxy (
              topic_id TEXT PRIMARY KEY, binary_relevant_attributions INTEGER,
              facets_represented INTEGER
            );
            CREATE TABLE facet_proxy_sensitivity (
              topic_id TEXT PRIMARY KEY, tethered_dual_attributions INTEGER,
              tethered_dual_nr_attributions INTEGER, delta_without_redundancy INTEGER
            );
            CREATE TABLE report_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE sources (source_id TEXT PRIMARY KEY, label TEXT, bytes INTEGER, sha256 TEXT);
            CREATE TABLE receipts (operation TEXT PRIMARY KEY, value REAL);
            """
        )
        connection.executemany(
            "INSERT INTO arm_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    row["arm"],
                    *[row[f"relevant_count@{depth}"] for depth in DEPTHS],
                    row["relevant_count_full"],
                    row["ndcg@10"],
                    row["ndcg@100"],
                    row["ndcg@1000"],
                    row["recall_auc"],
                )
                for row in payload["arm_metrics"]
            ],
        )
        depth_labels: tuple[int | str, ...] = (*DEPTHS, "full")
        connection.executemany(
            "INSERT INTO arm_depth_metrics VALUES (?,?,?,?,?,?,?)",
            [
                (
                    row["arm"], str(depth), row[f"relevant_count{'_' if depth == 'full' else '@'}{depth}"],
                    row[f"binary_recall{'_' if depth == 'full' else '@'}{depth}"],
                    row[f"graded_recall{'_' if depth == 'full' else '@'}{depth}"],
                    row[f"facet_only_relevant_retained{'_' if depth == 'full' else '@'}{depth}"],
                    row[f"facet_only_relevant_retention{'_' if depth == 'full' else '@'}{depth}"],
                )
                for row in payload["arm_metrics"] for depth in depth_labels
            ],
        )
        connection.executemany(
            "INSERT INTO arm_quality_metrics VALUES (?,?,?,?,?,?,?)",
            [(row["arm"], row["ndcg@10"], row["ndcg@100"], row["ndcg@500"], row["ndcg@1000"], row["ndcg@1500"], row["recall_auc"]) for row in payload["arm_metrics"]],
        )
        connection.executemany(
            "INSERT INTO topic_deltas VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    row["topic_id"],
                    *[row[f"relevant_delta@{depth}"] for depth in DEPTHS],
                    row["ndcg_delta@100"],
                    row["ndcg_delta@1000"],
                    row["recall_auc_delta"],
                )
                for row in payload["topic_deltas"]
            ],
        )
        connection.executemany(
            "INSERT INTO overlap_decomposition VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    row["topic_id"],
                    row["original_relevant"],
                    row["facet_relevant"],
                    row["overlap_relevant"],
                    row["original_only_relevant"],
                    row["facet_only_relevant"],
                    row["incremental_relevant"],
                    row["union_relevant"],
                    row["missed_relevant"],
                    row["total_relevant"],
                )
                for row in payload["overlap_decomposition"]
            ],
        )
        connection.executemany(
            "INSERT INTO facet_proxy VALUES (?,?,?)",
            [
                (
                    row["topic_id"],
                    row["binary_relevant_attributions"],
                    row["facets_with_positive_attribution"],
                )
                for row in payload["facet_proxy_by_topic"]
            ],
        )
        connection.executemany(
            "INSERT INTO facet_proxy_sensitivity VALUES (?,?,?,?)",
            [(row["topic_id"], row["tethered_dual_attributions"], row["tethered_dual_nr_attributions"], row["delta_without_redundancy"]) for row in payload["facet_proxy_sensitivity"]],
        )
        metadata = {
            "relevance_threshold": payload["relevance_threshold"],
            **payload["qrels_grade_counts"],
            **{
                f"freeze_{key}": value
                for key, value in payload["freeze_verification"].items()
            },
            **{f"evaluation_{key}": value for key, value in payload["evaluation_verification"].items()},
        }
        connection.executemany(
            "INSERT INTO report_metadata VALUES (?,?)",
            [(key, str(value)) for key, value in metadata.items()],
        )
        connection.executemany(
            "INSERT INTO sources VALUES (?,?,?,?)",
            [
                (source_id, source["label"], source["bytes"], source["sha256"])
                for source_id, source in payload["source_hashes"].items()
            ],
        )
        connection.executemany(
            "INSERT INTO receipts VALUES (?,?)", payload["external_calls"].items()
        )


def write_report(
    freeze: Path | str,
    evaluation: Path | str,
    prior_summary: Path | str,
    output: Path | str,
) -> dict[str, object]:
    """Build and atomically write the four canonical report artifacts."""

    payload = build_report_payload(freeze, evaluation, prior_summary)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        **payload,
        "surface": "report",
        "chart_omissions": [
            {
                "name": "four-topic overlap decomposition",
                "replacement": "semantic_table",
                "reason": "Exact overlap counts are more audit-friendly; a stacked chart would imply false additivity between original and facet totals.",
            }
        ],
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "topic_ids": list(TOPIC_IDS),
        "relevance_threshold": payload["relevance_threshold"],
        "qrels_grade_counts": payload["qrels_grade_counts"],
        "headline": payload["findings"],
        "source_hashes": payload["source_hashes"],
        "external_calls": payload["external_calls"],
        "freeze_verification": payload["freeze_verification"],
        "evaluation_verification": payload["evaluation_verification"],
        "metric_definitions": payload["metric_definitions"],
        "report_boundaries": payload["report_boundaries"],
    }
    html_bytes = render_report(payload).encode("utf-8")
    writes = {
        "artifact.json": _json_bytes(artifact),
        "summary.json": _json_bytes(summary),
        "report.html": html_bytes,
    }
    for name, content in writes.items():
        temporary = output_dir / f".{name}.tmp"
        temporary.write_bytes(content)
        os.replace(temporary, output_dir / name)
    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as handle:
        sqlite_tmp = Path(handle.name)
    try:
        _write_sqlite(sqlite_tmp, payload)
        os.replace(sqlite_tmp, output_dir / "report_data.sqlite")
    finally:
        sqlite_tmp.unlink(missing_ok=True)
    files = {}
    for name in ("artifact.json", "summary.json", "report_data.sqlite", "report.html"):
        content = (output_dir / name).read_bytes()
        files[name] = {"bytes": len(content), "sha256": _sha256(content)}
    return {"status": "complete", "schema_version": SCHEMA_VERSION, "files": files}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--prior-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = write_report(args.freeze, args.evaluation, args.prior_summary, args.output)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
