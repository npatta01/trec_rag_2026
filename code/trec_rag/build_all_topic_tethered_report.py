"""Build the canonical all-topic tethered-facet validation report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .all_topic_tethered_evaluate import _load_qrels
from .topic_failure_postmortem import compute_capture_diagnostic


CANONICAL_RANKING_ROOT_SHA256 = "6f4c35899f90c1d60324caf24bf8834d3482c5e8b9e785f8316ab0eec55fc305"
CANONICAL_EVALUATION_ROOT_SHA256 = "e49ce3f7f0cfeedf1863670ae4cfe5781d40163449a172a5822d8eb11eeffe84"
CANONICAL_PLANNING_ROOT_SHA256 = "bc1351cca8aa05dd0979a342c8dd5f72395f2b7a9207ae20668aba05f1ab85a2"
CANONICAL_RETRIEVAL_ROOT_SHA256 = "f2191c295f600d0243f4dcd2b1dc23a05b9fa8a7d9a4d6258983a0d3c9433aeb"
CANONICAL_SCORE_PLAN_ROOT_SHA256 = "48c8b9be21adece21c1f17cdeec4de8694a83f5ee7fccaa9fc07ff8ca36944c3"
CANONICAL_SCORING_ROOT_SHA256 = "4f67107770b1cd35598adc75beeeafe205ba984f201cedf4b505c4d20515e990"
SOURCE_SEAL_SHA256 = {
    "planning/SEALED.json": "dda97e1872c864f35af16f1a0b8d57d7218778a1274700197c773cb8ae75aa85",
    "retrieval/RETRIEVAL_SEALED.json": "3175f303a63b561b5fe4d3b6c0185f262fcc1032dd97a1ac7b12573db73a6288",
    "scoring/SCORE_PLAN_SEALED.json": "f224599eaadab08453bc3a6b7c28c00e1462e5b6b033e7c91bf22ca63ff67d9f",
    "scoring/SCORING_SEALED.json": "fcbdc59bc9dd27aaf42df7df8b5508a3f4af93d2e5a96217b350b0b728b391fb",
}
RANKING_DIR = "rankings_v3"
EVALUATION_DIR = "evaluation_v3"
DEPTHS = (100, 250, 500, 1000, 1500)
CAPTURE_DEPTHS = (*DEPTHS, "full")
PRIMARY_ARM = "RRF100-STATIC-DUAL"
POSTMORTEM_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports/experiments/all_topic_tethered_facet_validation_v1/postmortem.json"
)
VERIFICATION_BUNDLE_PATH = (
    "cache/experiments/all_topic_tethered_facet_validation_v1_sources_v3.tar.zst"
)
VERIFICATION_BUNDLE_SHA256 = "3b726dcff28f8e67e6d4a5cf330e390c150b0a601091af17ae59b5870bc46e59"


@dataclass(frozen=True)
class BuiltReport:
    summary: dict[str, Any]
    artifact: dict[str, Any]
    html: str
    datasets: dict[str, list[dict[str, Any]]]
    output_dir: Path | None = None


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _verify_seal(directory: Path, expected_root: str) -> dict[str, Any]:
    seal = _json(directory / "SEALED.json")
    if seal.get("root_sha256") != expected_root:
        raise ValueError(f"{directory.name} differs from corrected canonical root")
    files = seal.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{directory.name} seal has no file inventory")
    for name, record in files.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError(f"{directory.name} seal inventory is malformed")
        data = (directory / name).read_bytes()
        if len(data) != record.get("bytes") or _sha256_bytes(data) != record.get("sha256"):
            raise ValueError(f"{directory.name}/{name} differs from its sealed bytes")
    return seal


def _verify_cost_sources(root: Path) -> dict[str, dict[str, Any]]:
    seals = {}
    for relative, digest in SOURCE_SEAL_SHA256.items():
        data = (root / relative).read_bytes()
        if _sha256_bytes(data) != digest:
            raise ValueError(f"{relative} differs from the pinned canonical seal")
        seals[relative] = json.loads(data)
    expected_roots = {
        "planning/SEALED.json": CANONICAL_PLANNING_ROOT_SHA256,
        "retrieval/RETRIEVAL_SEALED.json": CANONICAL_RETRIEVAL_ROOT_SHA256,
        "scoring/SCORE_PLAN_SEALED.json": CANONICAL_SCORE_PLAN_ROOT_SHA256,
        "scoring/SCORING_SEALED.json": CANONICAL_SCORING_ROOT_SHA256,
    }
    for relative, expected in expected_roots.items():
        if seals[relative].get("root_sha256") != expected:
            raise ValueError(f"{relative} root differs")
    if seals["scoring/SCORING_SEALED.json"].get("score_plan_root_sha256") != CANONICAL_SCORE_PLAN_ROOT_SHA256:
        raise ValueError("scoring seal differs from score-plan binding")
    for relative, seal_key, leaf in (
        ("retrieval/retrieval_summary.json", "retrieval/RETRIEVAL_SEALED.json", "retrieval_summary.json"),
        ("scoring/scoring_receipt.json", "scoring/SCORING_SEALED.json", "scoring_receipt.json"),
    ):
        data = (root / relative).read_bytes(); record = seals[seal_key]["files"][leaf]
        if len(data) != record["bytes"] or _sha256_bytes(data) != record["sha256"]:
            raise ValueError(f"{leaf} differs from the pinned sealed inventory")
    return seals


def _load_sources(root: Path, ranking_dir_name: str, evaluation_dir_name: str) -> dict[str, dict[str, Any]]:
    if ranking_dir_name != RANKING_DIR or evaluation_dir_name != EVALUATION_DIR:
        raise ValueError("superseded v1/v2 ranking/evaluation evidence is rejected")
    ranking_dir = root / ranking_dir_name
    evaluation_dir = root / evaluation_dir_name
    ranking_seal = _verify_seal(ranking_dir, CANONICAL_RANKING_ROOT_SHA256)
    evaluation_seal = _verify_seal(evaluation_dir, CANONICAL_EVALUATION_ROOT_SHA256)
    cost_seals = _verify_cost_sources(root)
    bindings = _json(evaluation_dir / "input_bindings.json")
    if bindings.get("ranking_root_sha256") != CANONICAL_RANKING_ROOT_SHA256:
        raise ValueError("evaluation is not bound to corrected canonical ranking")
    if bindings.get("upstream_roots") != {
        "planning_root_sha256": CANONICAL_PLANNING_ROOT_SHA256,
        "retrieval_root_sha256": CANONICAL_RETRIEVAL_ROOT_SHA256,
        "score_plan_root_sha256": CANONICAL_SCORE_PLAN_ROOT_SHA256,
        "scoring_root_sha256": CANONICAL_SCORING_ROOT_SHA256,
    }:
        raise ValueError("evaluation upstream roots differ from canonical cost sources")
    summary = _json(evaluation_dir / "summary.json")
    if summary.get("schema_version") != "all-topic-tethered-evaluation-v3":
        raise ValueError("evaluation schema is not canonical v3")
    return {
        "ranking_seal": ranking_seal,
        "evaluation_seal": evaluation_seal,
        "evaluation_summary": summary,
        "metrics": _json(evaluation_dir / "metrics.json"),
        "diagnostics": _json(evaluation_dir / "diagnostics.json"),
        "bindings": bindings,
        "ranking_parameters": _json(ranking_dir / "parameters.json"),
        "retrieval": _json(root / "retrieval/retrieval_summary.json"),
        "scoring": _json(root / "scoring/scoring_receipt.json"),
        "planning_seal": cost_seals["planning/SEALED.json"],
        "retrieval_seal": cost_seals["retrieval/RETRIEVAL_SEALED.json"],
        "scoring_seal": cost_seals["scoring/SCORING_SEALED.json"],
    }


def _fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _metric_key(name: str, depth: int | str) -> str:
    return f"{name}_full" if depth == "full" else f"{name}@{depth}"


def _depth_label(depth: int | str) -> str:
    return "Full union" if depth == "full" else f"{depth:,}"


def _arm_rows(metrics: Mapping[str, Any], decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ladder = ["RRF", *decision["selection_ladder"]]
    for arm in ladder:
        item = metrics["arms"][arm]
        pooled = item["aggregate"]["pooled"]
        rows.append({
            "arm": arm,
            "known_relevant_at_1000": pooled["known_relevant_count@1000"],
            "binary_recall_at_1000": pooled["binary_recall@1000"],
            "macro_recall_at_1000": item["aggregate"]["macro"]["binary_recall@1000"],
            "judged_rate_at_1000": pooled["judged_rate@1000"],
            "wins": item["wins_at_1000"],
            "ties": item["ties_at_1000"],
            "losses": item["losses_at_1000"],
            "loss_topic_ids": item["loss_topic_ids_at_1000"],
            "worst_topic_id": item["worst_regression_at_1000"]["topic_id"],
            "worst_known_relevant_delta": item["worst_regression_at_1000"]["known_relevant_count_delta"],
            "promoted": arm != "RRF" and decision["arm_decisions"][arm]["promoted"],
            "failed_rules": [] if arm == "RRF" else decision["arm_decisions"][arm]["failed_rules"],
        })
    return rows


def _topic_rows(metrics: Mapping[str, Any], topic_ids: list[str]) -> list[dict[str, Any]]:
    baseline = metrics["arms"]["RRF"]
    rows: list[dict[str, Any]] = []
    for topic_id in topic_ids:
        base = baseline["per_topic"][topic_id]
        primary_delta = metrics["arms"][PRIMARY_ARM]["per_topic_deltas"][topic_id]
        rrf500_delta = metrics["arms"]["RRF500-REINIT-DUAL"]["per_topic_deltas"][topic_id]
        nr_delta = metrics["arms"]["RRF100-STATIC-DUAL-NR"]["per_topic_deltas"][topic_id]
        rows.append({
            "topic_id": topic_id,
            "known_relevant_total": base["known_relevant_total"],
            "rrf_known_relevant_at_1000": base["known_relevant_count@1000"],
            "rrf_recall_at_1000": base["binary_recall@1000"],
            "primary_known_relevant_delta_at_1000": int(primary_delta["known_relevant_count@1000"]),
            "primary_recall_delta_at_1000": primary_delta["binary_recall@1000"],
            "rrf500_known_relevant_delta_at_1000": int(rrf500_delta["known_relevant_count@1000"]),
            "nr_known_relevant_delta_at_1000": int(nr_delta["known_relevant_count@1000"]),
            "primary_delta_at_250": int(primary_delta["known_relevant_count@250"]),
            "primary_delta_at_500": int(primary_delta["known_relevant_count@500"]),
            "primary_delta_at_1500": int(primary_delta["known_relevant_count@1500"]),
        })
    return rows


def _depth_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ("RRF", PRIMARY_ARM, "RRF500-REINIT-DUAL", "RRF100-STATIC-DUAL-NR"):
        pooled = metrics["arms"][arm]["aggregate"]["pooled"]
        for depth in CAPTURE_DEPTHS:
            rows.append({
                "arm": arm,
                "depth": depth,
                "metric_source": "sealed evaluation",
                "binary_recall": pooled[_metric_key("binary_recall", depth)],
                "known_relevant_count": pooled[_metric_key("known_relevant_count", depth)],
                "known_relevant_total": pooled["known_relevant_total"],
                "graded_gain": pooled[_metric_key("graded_gain", depth)],
                "graded_gain_total": pooled["known_relevant_graded_gain_total"],
                "graded_recall": pooled[_metric_key("graded_recall", depth)],
                "facet_only_retention": pooled[
                    _metric_key("facet_only_known_relevant_retention", depth)
                ],
                "judged_rate": pooled[_metric_key("judged_rate", depth)],
            })
    return rows


def _authenticated_depth_20_diagnostic(
    root: Path, bindings: Mapping[str, Any]
) -> dict[str, object]:
    qrels_binding = bindings.get("qrels")
    if not isinstance(qrels_binding, Mapping):
        raise ValueError("evaluation qrels binding is missing")
    logical_path = Path(str(qrels_binding.get("path")))
    if logical_path.is_absolute() or ".." in logical_path.parts:
        raise ValueError("evaluation qrels binding path is not portable")
    qrels_bytes = (Path(__file__).resolve().parents[2] / logical_path).read_bytes()
    if (
        len(qrels_bytes) != qrels_binding.get("bytes")
        or _sha256_bytes(qrels_bytes) != qrels_binding.get("sha256")
    ):
        raise ValueError("depth-20 qrels differ from authenticated evaluation binding")
    rankings: dict[str, list[str]] = {}
    with (root / RANKING_DIR / "rankings.jsonl").open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("arm") == "RRF":
                rankings.setdefault(str(row["topic_id"]), []).append(
                    str(row["document_id"])
                )
    return compute_capture_diagnostic(rankings, _load_qrels(qrels_bytes), depth=20)


def _capture_diagnostic_row(
    postmortem: Mapping[str, Any],
    *,
    authenticated_diagnostic: Mapping[str, object],
    qrels_sha256: str,
    known_relevant_total: int,
    graded_gain_total: int,
) -> dict[str, Any]:
    provenance = postmortem.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("qrels_sha256") != qrels_sha256:
        raise ValueError("postmortem depth-20 diagnostic qrels binding differs")
    raw = postmortem.get("rrf_capture_diagnostic_at_20")
    if not isinstance(raw, Mapping):
        raise ValueError("postmortem depth-20 diagnostic is missing")
    if dict(raw) != dict(authenticated_diagnostic):
        raise ValueError(
            "postmortem depth-20 diagnostic differs from authenticated ranking and qrels"
        )
    required = {
        "arm": "RRF",
        "depth": 20,
        "metric_source": "postmortem diagnostic",
        "known_relevant_total": known_relevant_total,
        "graded_gain_total": graded_gain_total,
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise ValueError("postmortem depth-20 diagnostic does not reconcile to sealed totals")
    known_count = int(raw["known_relevant_count"])
    gain = int(raw["graded_gain"])
    retrieved_count = int(raw["retrieved_count"])
    judged_count = int(raw["judged_count"])
    if (
        float(raw["binary_recall"]) != known_count / known_relevant_total
        or float(raw["graded_recall"]) != gain / graded_gain_total
        or float(raw["judged_rate"]) != judged_count / retrieved_count
    ):
        raise ValueError("postmortem depth-20 diagnostic ratios do not reconcile")
    return {
        "arm": "RRF",
        "depth": 20,
        "metric_source": "postmortem diagnostic",
        "binary_recall": float(raw["binary_recall"]),
        "known_relevant_count": known_count,
        "known_relevant_total": known_relevant_total,
        "graded_gain": gain,
        "graded_gain_total": graded_gain_total,
        "graded_recall": float(raw["graded_recall"]),
        "facet_only_retention": None,
        "judged_rate": float(raw["judged_rate"]),
    }


def _failure_rows(postmortem: Mapping[str, Any]) -> list[dict[str, Any]]:
    if postmortem.get("schema_version") != "topic-failure-postmortem-v1":
        raise ValueError("postmortem schema is not canonical v1")
    if postmortem.get("experiment_id") != "all_topic_tethered_facet_validation_v1":
        raise ValueError("postmortem experiment differs")
    provenance = postmortem.get("provenance", {})
    if provenance.get("ranking_root_sha256") != CANONICAL_RANKING_ROOT_SHA256:
        raise ValueError("postmortem ranking root differs")
    if provenance.get("retrieval_root_sha256") != CANONICAL_RETRIEVAL_ROOT_SHA256:
        raise ValueError("postmortem retrieval root differs")
    if provenance.get("scoring_root_sha256") != CANONICAL_SCORING_ROOT_SHA256:
        raise ValueError("postmortem scoring root differs")

    topics = postmortem.get("topics", {})
    rows: list[dict[str, Any]] = []
    for topic_id in ("31", "300"):
        topic = topics.get(topic_id)
        if not isinstance(topic, dict):
            raise ValueError(f"postmortem is missing Topic {topic_id}")
        row = {
            "topic_id": topic_id,
            "known_relevant_total": topic["known_relevant_total"],
            "primary_at_1000": topic["primary_at_1000"],
            "primary_boundary_attribution": topic["primary_boundary_attribution"],
        }
        if topic_id == "300":
            replay = topic["recovery_replay"]
            row["facet_bucket_yield"] = topic["facet_bucket_yield"]
            row["recovery_replay"] = {
                "facet_rank_cap": replay["facet_rank_cap"],
                "protected_prefix_depth": replay["protected_prefix_depth"],
                **replay["arms"]["facet_rank_cap_100"],
            }
        rows.append(row)
    return rows


def _bucket_rows(diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    order = ("1-50", "51-100", "101-150", "151-200")
    rows: list[dict[str, Any]] = []
    for topic_id, buckets in diagnostics["facet_rank_bucket_yield"].items():
        for bucket in order:
            row = buckets[bucket]
            rows.append({"topic_id": topic_id, "bucket": bucket, **row})
    return rows


def _line_chart(depth_rows: list[dict[str, Any]]) -> str:
    selected = {"RRF": "#375a9e", PRIMARY_ARM: "#c24f3d", "RRF500-REINIT-DUAL": "#26806d"}
    width, height, left, top, plot_w, plot_h = 760, 330, 64, 28, 650, 240
    depths = list(CAPTURE_DEPTHS)
    max_y = max(
        r["binary_recall"]
        for r in depth_rows
        if r["arm"] in selected
    ) * 1.05
    parts = [f'<svg viewBox="0 0 {width} {height}" data-y-max="{max_y:.4f}" role="img" aria-labelledby="recall-title recall-desc"><title id="recall-title">Recall and retained facet evidence by depth</title><desc id="recall-desc">Binary recall rises with depth for RRF and the two strongest DUAL variants. Alternatives improve aggregate recall but still cause topic regressions.</desc>']
    for tick in (0.0, 0.1, 0.2, 0.3):
        y = top + plot_h - tick / max_y * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/><text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{tick:.1f}</text>')
    for i, depth in enumerate(depths):
        x = left + i * plot_w / (len(depths) - 1)
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle">{_depth_label(depth)}</text>')
    for arm, color in selected.items():
        values = [next(r["binary_recall"] for r in depth_rows if r["arm"] == arm and r["depth"] == d) for d in depths]
        points = " ".join(f'{left+i*plot_w/(len(depths)-1):.1f},{top+plot_h-v/max_y*plot_h:.1f}' for i, v in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>')
        for i, value in enumerate(values):
            x = left + i * plot_w / (len(depths) - 1); y = top + plot_h - value / max_y * plot_h
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
    parts.append('</svg><div class="legend"><span><i style="background:#375a9e"></i>RRF</span><span><i style="background:#c24f3d"></i>Primary DUAL</span><span><i style="background:#26806d"></i>RRF500 DUAL</span></div>')
    return "".join(parts)


def _bucket_chart(bucket_rows: list[dict[str, Any]]) -> str:
    order = ("1-50", "51-100", "101-150", "151-200")
    aggregates = []
    for bucket in order:
        chosen = [r for r in bucket_rows if r["bucket"] == bucket]
        relevant = sum(r["known_relevant_count"] for r in chosen)
        candidates = sum(r["unique_candidate_count"] for r in chosen)
        aggregates.append((bucket, relevant / candidates, relevant, candidates))
    width, height, left, top, plot_w, plot_h = 760, 310, 70, 25, 620, 220
    max_y = max(v for _, v, _, _ in aggregates) * 1.15
    bar_w = 90
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="yield-title yield-desc"><title id="yield-title">Known-relevant yield falls with facet depth</title><desc id="yield-desc">Pooled known-relevant yield by facet rank bucket, calculated across all twenty-two topics.</desc>']
    for i, (bucket, value, relevant, candidates) in enumerate(aggregates):
        x = left + 45 + i * plot_w / 4
        h = value / max_y * plot_h; y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="5" fill="#375a9e"/><text x="{x+bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" class="value">{value*100:.1f}%</text><text x="{x+bar_w/2:.1f}" y="{top+plot_h+23}" text-anchor="middle">{bucket}</text>')
    parts.append('</svg>')
    return "".join(parts)


def _movement_counts(movement: Mapping[str, Any]) -> str:
    return (
        f'{movement["total"]:,} total · '
        f'{movement["known_relevant"]:,} known relevant · '
        f'{movement["judged_below_2"]:,} judged below 2 · '
        f'{movement["unjudged"]:,} unjudged (unknown)'
    )


def _mapping_text(values: Mapping[str, Any], *, facet_names: bool = False) -> str:
    labels = []
    for name, count in sorted(values.items()):
        label = name.split("-", 1)[-1] if facet_names else name
        labels.append(f'{label.replace("_", " ").replace("-", " ")} {count:,}')
    return "; ".join(labels)


def _attribution_html(attribution: Mapping[str, Any]) -> str:
    blocks = []
    for direction in ("outgoing", "incoming"):
        values = attribution[direction]
        stream_items = []
        for stream_id, stream in sorted(
            values["facet_retrieval_stream_memberships"].items()
        ):
            ranks = _mapping_text(stream["rank_buckets"])
            stream_items.append(
                f'<li><code>{html.escape(stream_id)}</code> '
                f'({html.escape(stream["facet_label"])}), query '
                f'<q>{html.escape(stream["query_formulation"])}</q>: '
                f'{stream["total"]:,} memberships · '
                f'{stream["known_relevant"]:,} known relevant · '
                f'{stream["judged_below_2"]:,} judged below 2 · '
                f'{stream["unjudged"]:,} unjudged (unknown) · ranks {ranks}.</li>'
            )
        blocks.append(
            f'<div><h4>{direction.title()} attribution</h4>'
            f'<p><strong>Source families:</strong> {_mapping_text(values["source_family"])}.<br>'
            f'<strong>Best facet-rank buckets:</strong> {_mapping_text(values["best_facet_rank_bucket"])}.<br>'
            f'<strong>Facet retrieval-stream memberships (actual accepted-union provenance):</strong></p>'
            f'<ul>{"".join(stream_items) or "<li>none</li>"}</ul>'
            f'<p><strong>DUAL selection-coverage facets (greedy selection audit, not retrieval-stream provenance):</strong> '
            f'{_mapping_text(values["dual_selection_coverage_facets"], facet_names=True)}.'
            f'</p></div>'
        )
    return f'<div class="evidence-grid">{"".join(blocks)}</div>'


def _failure_details(row: Mapping[str, Any]) -> str:
    primary = row["primary_at_1000"]
    return (
        f'<details><summary>Topic {row["topic_id"]} cutoff mechanics: '
        f'{primary["known_relevant_delta"]:+d} known relevant at 1,000</summary>'
        f'<p>The candidate moves {primary["incoming"]["total"]:,} documents across '
        f'the cutoff. <strong>Outgoing:</strong> {_movement_counts(primary["outgoing"])}. '
        f'<strong>Incoming:</strong> {_movement_counts(primary["incoming"])}.</p>'
        f'{_attribution_html(row["primary_boundary_attribution"])}</details>'
    )


def _render_html(summary: Mapping[str, Any], datasets: Mapping[str, list[dict[str, Any]]]) -> str:
    arms = datasets["arm_metrics"]
    topics = datasets["topic_metrics"]
    depth_rows = datasets["depth_metrics"]
    bucket_rows = datasets["facet_bucket_yield"]
    failures = {row["topic_id"]: row for row in datasets["failure_evidence"]}
    primary = next(r for r in arms if r["arm"] == PRIMARY_ARM)
    base = next(r for r in arms if r["arm"] == "RRF")
    ladder_rows = "".join(
        f'<tr><th scope="row">{html.escape(r["arm"])}</th><td>{r["known_relevant_at_1000"]:,}</td><td>{r["known_relevant_at_1000"]-base["known_relevant_at_1000"]:+,}</td><td>{_fmt_pct(r["binary_recall_at_1000"])}</td><td>{r["wins"]}/{r["ties"]}/{r["losses"]}</td><td>{", ".join(r["loss_topic_ids"]) or "none"}</td><td>{"retain baseline" if r["arm"] == "RRF" else "reject"}</td></tr>'
        for r in arms
    )
    topic_rows = "".join(
        f'<tr class="{"loss" if r["primary_known_relevant_delta_at_1000"] < 0 else ""}"><th scope="row">Topic {r["topic_id"]}</th><td>{r["known_relevant_total"]:,}</td><td>{r["rrf_known_relevant_at_1000"]:,}</td><td>{_fmt_pct(r["rrf_recall_at_1000"])}</td><td>{r["primary_known_relevant_delta_at_1000"]:+d}</td><td>{r["rrf500_known_relevant_delta_at_1000"]:+d}</td><td>{r["nr_known_relevant_delta_at_1000"]:+d}</td><td>{r["primary_delta_at_250"]:+d} / {r["primary_delta_at_500"]:+d} / {r["primary_delta_at_1500"]:+d}</td></tr>'
        for r in topics
    )
    diag = "".join(
        f'<details><summary>Topic {r["topic_id"]}: primary DUAL {r["primary_known_relevant_delta_at_1000"]:+d} at 1,000</summary><p>RRF retrieves {r["rrf_known_relevant_at_1000"]:,} of {r["known_relevant_total"]:,} known-relevant documents at depth 1,000 ({_fmt_pct(r["rrf_recall_at_1000"],2)}). Primary DUAL deltas at depths 250, 500, 1,000, and 1,500 are {r["primary_delta_at_250"]:+d}, {r["primary_delta_at_500"]:+d}, {r["primary_known_relevant_delta_at_1000"]:+d}, and {r["primary_delta_at_1500"]:+d}.</p></details>'
        for r in topics if r["primary_known_relevant_delta_at_1000"] < 0 or r["primary_known_relevant_delta_at_1000"] >= 20
    )
    capture_rows = "".join(
        f'<tr><th scope="row">{_depth_label(r["depth"])}</th>'
        f'<td>{r["metric_source"].capitalize()}</td>'
        f'<td>{r["known_relevant_count"]:,} / {r["known_relevant_total"]:,}</td>'
        f'<td>{_fmt_pct(r["binary_recall"], 2)}</td>'
        f'<td>{r["graded_gain"]:,} / {r["graded_gain_total"]:,}</td>'
        f'<td>{_fmt_pct(r["graded_recall"], 2)}</td></tr>'
        for r in depth_rows if r["arm"] == "RRF"
    )
    topic_300_replay = failures["300"]["recovery_replay"]
    replay_yields = " · ".join(
        f'{bucket} {_fmt_pct(value, 2)}'
        for bucket, value in failures["300"]["facet_bucket_yield"].items()
    )
    failure_disclosures = (
        _failure_details(failures["31"])
        + _failure_details(failures["300"])
        + (
            f'<details><summary>Topic 300 facet-tail replay: cap '
            f'{topic_300_replay["facet_rank_cap"]} recovers '
            f'{topic_300_replay["known_relevant_delta"]:+d} at 1,000</summary>'
            f'<p>The offline replay protects the first '
            f'{topic_300_replay["protected_prefix_depth"]} RRF results, keeps '
            f'original-stream candidates eligible, and defers facet-only candidates whose '
            f'best facet rank is deeper than cap {topic_300_replay["facet_rank_cap"]}. '
            f'<strong>Outgoing:</strong> {_movement_counts(topic_300_replay["outgoing"])}. '
            f'<strong>Incoming:</strong> {_movement_counts(topic_300_replay["incoming"])}.</p>'
            f'<p><strong>Facet-tail known-relevant yield:</strong> {replay_yields}. '
            f'The +2 result is retrospective recovery evidence, not a promotion result.</p></details>'
        )
    )
    costs = summary["costs"]
    prov = summary["provenance"]
    stats = summary["statistics"]
    capture = summary["capture"]
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light dark"><title>All-topic tethered-facet validation</title>
<style>
:root{{--bg:#f4f6f8;--card:#fff;--ink:#17202a;--muted:#56616e;--line:#cbd3dc;--accent:#274f91;--warn:#a33d30;--good:#1d745d}}@media(prefers-color-scheme:dark){{:root{{--bg:#111820;--card:#18232e;--ink:#edf3f8;--muted:#b9c4cf;--line:#40505f;--accent:#91b9ff;--warn:#ff9f91;--good:#78d6b7}}}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}}main{{max-width:1120px;margin:auto;padding:24px}}header,section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:clamp(18px,3vw,34px);margin:0 0 20px}}h1{{font-size:clamp(2rem,5vw,4rem);line-height:1.02;max-width:15ch;margin:.2em 0}}h2{{font-size:clamp(1.4rem,3vw,2.1rem);line-height:1.15}}h3{{margin-top:1.8em}}h4{{margin-bottom:.35em}}p{{max-width:78ch}}.eyebrow{{text-transform:uppercase;letter-spacing:.12em;color:var(--accent);font-weight:750}}.verdict{{border-left:7px solid var(--warn)}}.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}}.kpi{{border:1px solid var(--line);border-radius:10px;padding:16px}}.kpi strong{{display:block;font-size:1.8rem}}.muted,figcaption{{color:var(--muted)}}.callout{{background:color-mix(in srgb,var(--warn) 10%,transparent);border-left:4px solid var(--warn);padding:12px 16px}}.table-wrap{{overflow-x:auto;max-width:100%;border:1px solid var(--line);border-radius:9px}}table{{border-collapse:collapse;width:100%;min-width:780px}}th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th:first-child,td:first-child{{text-align:left;position:sticky;left:0;background:var(--card)}}thead th{{text-align:right;background:color-mix(in srgb,var(--accent) 9%,var(--card))}}tr.loss th,tr.loss td{{color:var(--warn);font-weight:700}}figure{{margin:24px 0}}svg{{width:100%;height:auto;background:color-mix(in srgb,var(--accent) 4%,var(--card));border-radius:10px}}svg text{{fill:var(--ink);font:13px system-ui,sans-serif}}svg .grid{{stroke:var(--line);stroke-width:1}}svg .value{{font-weight:750}}.legend{{display:flex;gap:18px;flex-wrap:wrap}}.legend i{{display:inline-block;width:16px;height:4px;vertical-align:middle;margin-right:6px}}.method,.evidence-grid{{display:flex;gap:8px;align-items:stretch;flex-wrap:wrap}}.method div,.evidence-grid div{{flex:1 1 240px;border:1px solid var(--line);padding:12px;border-radius:8px}}.method b{{display:block}}details{{border-top:1px solid var(--line);padding:12px 0}}summary{{cursor:pointer;font-weight:700}}summary:focus-visible,a:focus-visible,.table-wrap:focus-visible{{outline:3px solid var(--accent);outline-offset:3px}}code{{overflow-wrap:anywhere}}@media(max-width:600px){{main{{padding:10px}}header,section{{padding:17px}}.kpi strong{{font-size:1.4rem}}}}
</style></head><body><main>
<header class="verdict" id="top"><p class="eyebrow">Portable canonical v3 evidence · decision report</p><h1>Retain RRF.</h1><p class="lede">The preregistered alternatives improve pooled known-relevant recall, but every alternative loses at least one topic at depth 1,000. That violates the zero-loss promotion rule.</p><div class="kpis"><div class="kpi"><strong>{base["known_relevant_at_1000"]:,}</strong>RRF known relevant @1,000</div><div class="kpi"><strong>+{primary["known_relevant_at_1000"]-base["known_relevant_at_1000"]}</strong>primary DUAL pooled gain</div><div class="kpi"><strong>{primary["wins"]}/{primary["ties"]}/{primary["losses"]}</strong>topic wins / ties / losses</div><div class="kpi"><strong>−7</strong>worst loss, Topic 31</div></div><p class="callout"><strong>Why not promote?</strong> Primary DUAL loses 7 known-relevant documents on Topic 31 and 3 on Topic 300 at 1,000. RRF500-REINIT-DUAL removes the Topic 31 loss but still loses 3 on Topic 300.</p></header>
<section id="capture"><h2>How much graded evidence do we capture?</h2><p><strong>At depth 20, the authenticated postmortem diagnostic finds {capture["rrf_at_20"]["known_relevant"]:,} / {capture["known_relevant_total"]:,} known-relevant documents ({_fmt_pct(capture["rrf_at_20"]["binary_recall"],2)}) and {capture["rrf_at_20"]["graded_gain"]:,} / {capture["graded_gain_total"]:,} graded-gain units ({_fmt_pct(capture["rrf_at_20"]["graded_recall"],2)}).</strong> This diagnostic is bound to the pinned qrels and RRF ranking but does not alter the sealed evaluation depths. At depth 1,000, RRF captures {capture["rrf_at_1000"]["known_relevant"]:,} / {capture["known_relevant_total"]:,} known-relevant documents ({_fmt_pct(capture["rrf_at_1000"]["binary_recall"],2)}) and {capture["rrf_at_1000"]["graded_gain"]:,} / {capture["graded_gain_total"]:,} graded-gain units ({_fmt_pct(capture["rrf_at_1000"]["graded_recall"],2)}). The full accepted union raises those ceilings to {_fmt_pct(capture["full_union"]["binary_recall"],2)} binary recall and {_fmt_pct(capture["full_union"]["graded_recall"],2)} graded recall.</p><p>Known-relevant binary recall gives every judgment at grade 2 or higher equal weight. Graded gain uses <code>2**grade - 1</code>, so higher-grade evidence contributes more. These are separate questions and neither is precision over unjudged documents.</p><div class="table-wrap" tabindex="0" role="region" aria-label="RRF binary and graded capture by diagnostic and sealed depth; scroll horizontally for all columns"><table><thead><tr><th scope="col">Depth</th><th scope="col">Metric source</th><th scope="col">Known relevant</th><th scope="col">Binary recall</th><th scope="col">Graded gain</th><th scope="col">Graded recall</th></tr></thead><tbody>{capture_rows}</tbody></table></div><p class="callout"><strong>Judgment-pool dependent.</strong> The denominators cover known judgments only. Unjudged candidates remain unknown, not nonrelevant, and the full-union values are a ceiling within this accepted candidate pool rather than corpus-wide recall.</p></section>
<section id="failure-analysis"><h2>Why Topics 31 and 300 fail at the cutoff</h2><p>The pooled gain hides two decisive regressions. Open the native disclosures for the incoming/outgoing label mix, source-family and facet-depth attribution, and the bounded replay. No document text or document identifiers are exposed.</p>{failure_disclosures}</section>
<section id="scope"><h2>What this result means—and what it does not</h2><p>This is a <strong>retrospective full-development stress test</strong> over all 22 development topics using already known judgments. It measures retrieval of <strong>known-relevant</strong> documents and is useful for diagnosing headroom and regressions. It is <strong>not evidence of generalization</strong> to new topics, unseen judgments, or production traffic. The downstream RAG answer generation is out of scope; no claim is made about answer accuracy, faithfulness, citation quality, or user utility.</p><p>The v1/v2 ranking/evaluation rejected label is intentional: only portable <code>rankings_v3</code> and <code>evaluation_v3</code> are admissible here. Their ranking bytes and scientific conclusion are unchanged from corrected v2; v3 removes checkout-local paths from sealed identity and enforces the judged-rate promotion guard.</p></section>
<section id="ladder"><h2>Exact preregistered selection ladder</h2><p>The baseline appears first for orientation; alternatives follow the sealed ladder exactly. Aggregate improvements are insufficient when a zero-loss guard fails.</p><div class="table-wrap" tabindex="0" role="region" aria-label="Exact selection ladder; scroll horizontally for all columns"><table><thead><tr><th scope="col">Arm</th><th scope="col">Known rel. @1k</th><th scope="col">Δ vs RRF</th><th scope="col">Pooled recall</th><th scope="col">W/T/L</th><th scope="col">Loss topics</th><th scope="col">Decision</th></tr></thead><tbody>{ladder_rows}</tbody></table></div></section>
<section id="depth"><h2>Recall and retained facet evidence by depth</h2><p>Recall rises as the cutoff expands, and DUAL variants retain more facet-only evidence. The aggregate curves explain the attraction of the alternatives; the per-topic guard below explains the decision.</p><figure>{_line_chart(depth_rows)}<figcaption>Pooled binary recall over the 12,984 known-relevant documents. Depth 100 is protected and identical across arms.</figcaption></figure><p class="muted">Judged-rate caveat: RRF judged rate falls from {_fmt_pct(next(r for r in depth_rows if r["arm"]=="RRF" and r["depth"]==100)["judged_rate"])} at 100 to {_fmt_pct(next(r for r in depth_rows if r["arm"]=="RRF" and r["depth"]==1000)["judged_rate"])} at 1,000. Unjudged documents are not negatives, so deep precision and yield are conservative and judgment-pool dependent.</p></section>
<section id="yield"><h2>Known-relevant yield falls with facet depth</h2><p>The first 50 results of each deduplicated facet stream carry the highest pooled known-relevant yield. Later buckets still add evidence but at lower density, supporting depth discipline rather than blanket expansion.</p><figure>{_bucket_chart(bucket_rows)}<figcaption>Pooled known-relevant count divided by pooled unique candidates in each facet-rank bucket across 148 facet streams.</figcaption></figure></section>
<section id="topics"><h2>Per-topic deltas expose the promotion blockers</h2><p>Every topic is shown. Count deltas are alternative minus RRF; negative values are regressions. The final column gives primary DUAL deltas at depths 250 / 500 / 1,500.</p><div class="table-wrap" tabindex="0" role="region" aria-label="Per-topic metrics; scroll horizontally for all columns"><table><thead><tr><th scope="col">Topic</th><th scope="col">Known rel.</th><th scope="col">RRF @1k</th><th scope="col">RRF recall</th><th scope="col">Primary Δ @1k</th><th scope="col">RRF500 Δ @1k</th><th scope="col">NR Δ @1k</th><th scope="col">Primary Δ 250/500/1500</th></tr></thead><tbody>{topic_rows}</tbody></table></div><h3>Representative diagnostics</h3>{diag}</section>
<section id="method"><h2>Method and evidence boundary</h2><div class="method" role="list" aria-label="Evaluation method"><div role="listitem"><b>1 · Plan</b>22 narratives → 148 tethered facet queries.</div><div role="listitem"><b>2 · Retrieve</b>Top 200 per facet; 45,144-document union.</div><div role="listitem"><b>3 · Score</b>Local MiniLM narrative/facet features.</div><div role="listitem"><b>4 · Freeze blind</b>Six complete arms, qrels unopened.</div><div role="listitem"><b>5 · Evaluate</b>Pinned development qrels opened only after v3 freeze verification.</div><div role="listitem"><b>6 · Decide</b>Apply exact ladder and all promotion guards.</div></div><p>The portable ranking freeze has 22 topics, 6 complete arms, 270,864 ranking rows, and 206,030 audit rows. The evaluation independently binds that freeze before opening the pinned qrels. The paired exact sign-flip test for primary DUAL estimates +3.394 percentage points mean topic recall (95% bootstrap CI +2.118 to +5.052; raw p=0.000002623; Holm-adjusted p=0.000006676). Statistical significance cannot override the preregistered topic-loss rule.</p></section>
<section id="costs"><h2>Costs and execution accounting</h2><div class="kpis"><div class="kpi"><strong>{costs["facet_requests"]}</strong>live facet retrieval requests</div><div class="kpi"><strong>{costs["local_forward_pairs"]:,}</strong>local forward pairs</div><div class="kpi"><strong>{costs["scoring_seconds"]:.3f}s</strong>local scoring wall time</div><div class="kpi"><strong>$0 recorded</strong>hosted / paid inference</div></div><p>Original-query requests: 0. Retrieval failures/retries: 0/0. Shared-score cache reuses: {costs["cache_reuse_pairs"]:,}. Completed score windows: {costs["windows"]:,}. Peak device/host memory: {costs["peak_device_bytes"]:,} / {costs["peak_host_bytes"]:,} bytes. Ranking and evaluation added zero retrieval, inference, model-load, hosted, paid, or network calls.</p></section>
<section id="limits"><h2>Limitations, decision, and next step</h2><ul><li>Retrospective known-judgment evidence can overstate certainty and does not establish held-out generalization.</li><li>Judgment incompleteness grows with depth; unjudged candidates may contain useful evidence.</li><li>The experiment evaluates retrieval and ranking only; answer generation remains untested.</li><li>Topic 31 and Topic 300 losses are small in pooled terms but decisive under the frozen guard.</li></ul><p><strong>Recommendation:</strong> retain RRF, run a bounded recovery lane for Topics 31/300-like failures, and start the RAG lane from a deeper pool with <strong>Source-diverse evidence selection</strong>. Do not pass the protected top 20 through unchanged and call it facet-aware RAG.</p></section>
<section id="provenance"><h2>Authenticated provenance</h2><p>Ranking v3 root: <code>{prov["ranking_root_sha256"]}</code><br>Evaluation v3 root: <code>{prov["evaluation_root_sha256"]}</code><br>Pinned qrels SHA-256: <code>{prov["qrels_sha256"]}</code><br>Planning root: <code>{prov["planning_root_sha256"]}</code><br>Retrieval root: <code>{prov["retrieval_root_sha256"]}</code><br>Scoring root: <code>{prov["scoring_root_sha256"]}</code></p><p class="muted">This sanitized report contains aggregate metrics and hashes only: no credentials, raw qrels, raw documents, document identifiers, request logs, or local filesystem paths.</p><p><a href="#top">Back to top</a></p></section>
</main></body></html>'''


def build_report(root: Path, *, ranking_dir_name: str = RANKING_DIR, evaluation_dir_name: str = EVALUATION_DIR) -> BuiltReport:
    source = _load_sources(Path(root), ranking_dir_name, evaluation_dir_name)
    postmortem_bytes = POSTMORTEM_PATH.read_bytes()
    postmortem = json.loads(postmortem_bytes)
    if not isinstance(postmortem, dict):
        raise ValueError("expected postmortem JSON object")
    evaluation_summary = source["evaluation_summary"]
    topic_ids = evaluation_summary["topic_ids"]
    metrics = source["metrics"]
    decision = source["diagnostics"]["decision"]
    arm_rows = _arm_rows(metrics, decision)
    topic_rows = _topic_rows(metrics, topic_ids)
    sealed_depth_rows = _depth_rows(metrics)
    rrf_1000 = next(
        row
        for row in sealed_depth_rows
        if row["arm"] == "RRF" and row["depth"] == 1000
    )
    depth_rows = [
        _capture_diagnostic_row(
            postmortem,
            authenticated_diagnostic=_authenticated_depth_20_diagnostic(
                Path(root), source["bindings"]
            ),
            qrels_sha256=source["bindings"]["qrels"]["sha256"],
            known_relevant_total=rrf_1000["known_relevant_total"],
            graded_gain_total=rrf_1000["graded_gain_total"],
        ),
        *sealed_depth_rows,
    ]
    datasets = {
        "arm_metrics": arm_rows,
        "topic_metrics": topic_rows,
        "depth_metrics": depth_rows,
        "facet_bucket_yield": _bucket_rows(source["diagnostics"]),
        "failure_evidence": _failure_rows(postmortem),
    }
    provenance = {
        "ranking_root_sha256": source["ranking_seal"]["root_sha256"],
        "evaluation_root_sha256": source["evaluation_seal"]["root_sha256"],
        "planning_root_sha256": source["bindings"]["upstream_roots"]["planning_root_sha256"],
        "retrieval_root_sha256": source["bindings"]["upstream_roots"]["retrieval_root_sha256"],
        "score_plan_root_sha256": source["bindings"]["upstream_roots"]["score_plan_root_sha256"],
        "scoring_root_sha256": source["bindings"]["upstream_roots"]["scoring_root_sha256"],
        "qrels_sha256": source["bindings"]["qrels"]["sha256"],
    }
    primary = next(row for row in arm_rows if row["arm"] == PRIMARY_ARM)
    primary_statistics = source["diagnostics"]["statistics"][PRIMARY_ARM]
    rrf_20 = next(
        row for row in depth_rows if row["arm"] == "RRF" and row["depth"] == 20
    )
    rrf_full = next(
        row for row in depth_rows if row["arm"] == "RRF" and row["depth"] == "full"
    )
    summary: dict[str, Any] = {
        "schema_version": "all-topic-tethered-report-summary-v1",
        "experiment_id": evaluation_summary["experiment_id"],
        "topic_ids": topic_ids,
        "arms": evaluation_summary["arms"],
        "depths": evaluation_summary["depths"],
        "decision": {
            "selected_arm": decision["selected_arm"],
            "promoted": decision["promoted"],
            "selection_ladder": decision["selection_ladder"],
            "primary_arm": PRIMARY_ARM,
            "primary_wins": primary["wins"],
            "primary_ties": primary["ties"],
            "primary_losses": primary["losses"],
            "primary_loss_topic_ids": primary["loss_topic_ids"],
            "recommendation": "retain RRF",
        },
        "capture": {
            "known_relevant_total": rrf_1000["known_relevant_total"],
            "graded_gain_total": rrf_1000["graded_gain_total"],
            "rrf_at_20": {
                "metric_source": rrf_20["metric_source"],
                "known_relevant": rrf_20["known_relevant_count"],
                "binary_recall": rrf_20["binary_recall"],
                "graded_gain": rrf_20["graded_gain"],
                "graded_recall": rrf_20["graded_recall"],
            },
            "rrf_at_1000": {
                "known_relevant": rrf_1000["known_relevant_count"],
                "binary_recall": rrf_1000["binary_recall"],
                "graded_gain": rrf_1000["graded_gain"],
                "graded_recall": rrf_1000["graded_recall"],
            },
            "full_union": {
                "known_relevant": rrf_full["known_relevant_count"],
                "binary_recall": rrf_full["binary_recall"],
                "graded_gain": rrf_full["graded_gain"],
                "graded_recall": rrf_full["graded_recall"],
            },
        },
        "postmortem": {
            "schema_version": postmortem["schema_version"],
            "sha256": _sha256_bytes(postmortem_bytes),
            "recommendation": postmortem["recommendation"],
        },
        "costs": {
            "facet_requests": source["retrieval"]["facet_request_count"],
            "retrieval_failures": source["retrieval"]["failures"],
            "retrieval_retries": source["retrieval"]["retry_count"],
            "original_network_requests": source["retrieval"]["original_network_requests"],
            "local_forward_pairs": source["scoring"]["unique_forward_pair_count"],
            "cache_reuse_pairs": source["scoring"]["cache_reuse_pair_count"],
            "windows": source["scoring"]["completed_window_count"],
            "scoring_seconds": source["scoring"]["elapsed_seconds"],
            "peak_device_bytes": source["scoring"]["peak_device_memory_bytes"],
            "peak_host_bytes": source["scoring"]["peak_host_memory_bytes"],
            "hosted_calls": source["scoring"]["hosted_calls"],
            "paid_calls": source["scoring"]["paid_calls"],
        },
        "statistics": {
            "test": "paired exact sign-flip test",
            "primary_mean_recall_delta": primary_statistics["sign_flip"]["observed_mean_delta"],
            "bootstrap_ci_95": primary_statistics["bootstrap_ci_95"],
            "raw_p_value": primary_statistics["sign_flip"]["p_value"],
            "holm_adjusted_p_value": primary_statistics["holm_adjusted_p"],
        },
        "provenance": provenance,
        "limitations": [
            "retrospective full-development stress test",
            "known-relevant evidence is judgment-pool dependent",
            "not evidence of generalization",
            "downstream RAG answer generation is out of scope",
        ],
    }
    html_text = _render_html(summary, datasets)
    artifact = {
        "schema_version": "all-topic-tethered-report-artifact-v1",
        "summary_sha256": _sha256_bytes(_canonical_json(summary)),
        "html_sha256": _sha256_bytes(html_text.encode()),
        "source_roots": provenance,
        "datasets": {name: {"row_count": len(rows), "sha256": _sha256_bytes(_canonical_json(rows))} for name, rows in datasets.items()},
        "sanitization": {"raw_qrels": False, "raw_documents": False, "document_ids": False, "credentials": False, "external_dependencies": False},
    }
    return BuiltReport(summary=summary, artifact=artifact, html=html_text, datasets=datasets)


def _write_database(path: Path, datasets: Mapping[str, list[dict[str, Any]]], provenance: Mapping[str, str]) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.executemany("INSERT INTO metadata VALUES (?, ?)", sorted(provenance.items()))
        db.execute("CREATE TABLE arm_metrics (arm TEXT PRIMARY KEY, payload_json TEXT NOT NULL)")
        db.executemany("INSERT INTO arm_metrics VALUES (?, ?)", [(r["arm"], json.dumps(r, sort_keys=True)) for r in datasets["arm_metrics"]])
        db.execute("CREATE TABLE topic_metrics (topic_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)")
        db.executemany("INSERT INTO topic_metrics VALUES (?, ?)", [(r["topic_id"], json.dumps(r, sort_keys=True)) for r in datasets["topic_metrics"]])
        db.execute(
            "CREATE TABLE depth_metrics ("
            "arm TEXT NOT NULL, depth TEXT NOT NULL, metric_source TEXT NOT NULL, "
            "binary_recall REAL NOT NULL, "
            "known_relevant_count INTEGER NOT NULL, known_relevant_total INTEGER NOT NULL, "
            "graded_gain INTEGER NOT NULL, graded_gain_total INTEGER NOT NULL, "
            "graded_recall REAL NOT NULL, facet_only_retention REAL, "
            "judged_rate REAL NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (arm, depth))"
        )
        db.executemany(
            "INSERT INTO depth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["arm"],
                    str(r["depth"]),
                    r["metric_source"],
                    r["binary_recall"],
                    r["known_relevant_count"],
                    r["known_relevant_total"],
                    r["graded_gain"],
                    r["graded_gain_total"],
                    r["graded_recall"],
                    r["facet_only_retention"],
                    r["judged_rate"],
                    json.dumps(r, sort_keys=True),
                )
                for r in datasets["depth_metrics"]
            ],
        )
        db.execute("CREATE TABLE facet_bucket_yield (topic_id TEXT NOT NULL, bucket TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (topic_id, bucket))")
        db.executemany("INSERT INTO facet_bucket_yield VALUES (?, ?, ?)", [(r["topic_id"], r["bucket"], json.dumps(r, sort_keys=True)) for r in datasets["facet_bucket_yield"]])
        db.execute(
            "CREATE TABLE failure_evidence ("
            "topic_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)"
        )
        db.executemany(
            "INSERT INTO failure_evidence VALUES (?, ?)",
            [
                (r["topic_id"], json.dumps(r, sort_keys=True))
                for r in datasets["failure_evidence"]
            ],
        )
        db.commit()
        db.execute("VACUUM")


def write_report(root: Path, output: Path) -> BuiltReport:
    built = build_report(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_bytes(_canonical_json(built.summary))
    (output / "report.html").write_text(built.html, encoding="utf-8")
    _write_database(output / "report_data.sqlite", built.datasets, built.summary["provenance"])
    artifact = {**built.artifact, "sqlite_sha256": _sha256_bytes((output / "report_data.sqlite").read_bytes())}
    (output / "artifact.json").write_bytes(_canonical_json(artifact))
    return BuiltReport(summary=built.summary, artifact=artifact, html=built.html, datasets=built.datasets, output_dir=output)


def verify_report(report: Path, root: Path = Path("outputs/all_topic_tethered_facet_validation_v1")) -> dict[str, Any]:
    report = Path(report)
    summary = _json(report / "summary.json")
    artifact = _json(report / "artifact.json")
    html_bytes = (report / "report.html").read_bytes()
    if _sha256_bytes(_canonical_json(summary)) != artifact.get("summary_sha256"):
        raise ValueError("summary differs from artifact")
    if _sha256_bytes(html_bytes) != artifact.get("html_sha256"):
        raise ValueError("HTML differs from artifact")
    if summary.get("provenance") != artifact.get("source_roots"):
        raise ValueError("artifact roots differ from summary")
    if summary["provenance"].get("ranking_root_sha256") != CANONICAL_RANKING_ROOT_SHA256 or summary["provenance"].get("evaluation_root_sha256") != CANONICAL_EVALUATION_ROOT_SHA256:
        raise ValueError("report is not canonical v3")
    sqlite_bytes = (report / "report_data.sqlite").read_bytes()
    if _sha256_bytes(sqlite_bytes) != artifact.get("sqlite_sha256"):
        raise ValueError("SQLite differs from artifact hash")
    with tempfile.TemporaryDirectory() as temporary:
        expected_dir = Path(temporary) / "expected"
        expected = write_report(root, expected_dir)
        if summary != expected.summary or html_bytes != expected.html.encode() or artifact != expected.artifact:
            raise ValueError("report differs from independent canonical rebuild")
        if sqlite_bytes != (expected_dir / "report_data.sqlite").read_bytes():
            raise ValueError("SQLite differs from independent canonical rebuild")
    with sqlite3.connect(report / "report_data.sqlite") as db:
        topics = db.execute("SELECT COUNT(*) FROM topic_metrics").fetchone()[0]
        arms = db.execute("SELECT COUNT(*) FROM arm_metrics").fetchone()[0]
        metadata = dict(db.execute("SELECT key, value FROM metadata"))
        schemas = dict(db.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"))
        payloads = {
            "arm_metrics": dict(db.execute("SELECT arm, payload_json FROM arm_metrics")),
            "topic_metrics": dict(db.execute("SELECT topic_id, payload_json FROM topic_metrics")),
            "depth_metrics": dict(db.execute("SELECT arm || ':' || depth, payload_json FROM depth_metrics")),
            "facet_bucket_yield": dict(db.execute("SELECT topic_id || ':' || bucket, payload_json FROM facet_bucket_yield")),
            "failure_evidence": dict(db.execute("SELECT topic_id, payload_json FROM failure_evidence")),
        }
    if topics != len(summary["topic_ids"]) or arms != len(summary["arms"]) or metadata != summary["provenance"]:
        raise ValueError("SQLite content differs from JSON summary")
    if set(schemas) != {"metadata", "arm_metrics", "topic_metrics", "depth_metrics", "facet_bucket_yield", "failure_evidence"}:
        raise ValueError("SQLite table schemas differ")
    for name, rows in expected.datasets.items():
        if name == "arm_metrics":
            key = lambda row: row["arm"]
        elif name in {"topic_metrics", "failure_evidence"}:
            key = lambda row: row["topic_id"]
        elif name == "depth_metrics":
            key = lambda row: f'{row["arm"]}:{row["depth"]}'
        else:
            key = lambda row: f'{row["topic_id"]}:{row["bucket"]}'
        expected_payloads = {
            str(key(row)): json.dumps(row, sort_keys=True) for row in rows
        }
        if payloads[name] != expected_payloads or artifact["datasets"][name]["sha256"] != _sha256_bytes(_canonical_json(rows)):
            raise ValueError(f"SQLite {name} payloads differ")
    return {"verified": True, "topic_count": topics, "arm_count": arms, "html_sha256": artifact["html_sha256"], "summary_sha256": artifact["summary_sha256"]}


def _readme(summary: Mapping[str, Any]) -> str:
    return f"""# All-topic tethered-facet validation v1

Decision: **{summary['decision']['recommendation']}**.

This directory contains the sanitized report over portable canonical v3 ranking
and evaluation evidence. Superseded v1/v2 evidence remains immutable but is not
admissible for this report. This is a retrospective full-development stress test
using known-relevant judgments; it is not evidence of generalization, and
downstream RAG answer generation is out of scope.

The tracked report is directly viewable without external data. Full source
verification additionally requires a 3 MiB compressed bundle (about 92 MiB
expanded) containing sealed rankings, evaluation tables, and minimal upstream
receipts. It contains candidate document identifiers. It is not stored in Git
or published with the sanitized report.

Expected local bundle:

- `{VERIFICATION_BUNDLE_PATH}`
- SHA-256: `{VERIFICATION_BUNDLE_SHA256}`

Restore and verify from the repository root on an authorized machine:

```bash
sha256sum {VERIFICATION_BUNDLE_PATH}
tar --zstd -xf {VERIFICATION_BUNDLE_PATH} -C .
.venv/bin/python -m trec_rag.build_all_topic_tethered_report verify \\
  --report reports/experiments/all_topic_tethered_facet_validation_v1 \\
  --root outputs/all_topic_tethered_facet_validation_v1
```

The first command must match the pinned SHA-256 above. Obtain the bundle through
the repository's private artifact-transfer process when it is absent; verification
does not download it automatically.

- `summary.json`: decision, exact roots, costs, statistics, and scope limits
- `report_data.sqlite`: arm, depth, topic, and facet-bucket rows
- `artifact.json`: hashes and sanitization declaration
- `report.html`: self-contained accessible report
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    verify = sub.add_parser("verify")
    verify.add_argument("--report", dest="verify_report", type=Path, required=True)
    verify.add_argument(
        "--root",
        dest="verify_root",
        type=Path,
        default=Path("outputs/all_topic_tethered_facet_validation_v1"),
    )
    write = sub.add_parser("write")
    write.add_argument("--root", dest="write_root", type=Path, required=True)
    write.add_argument("--report", dest="write_report", type=Path, required=True)
    parser.add_argument("--root", dest="legacy_root", type=Path)
    parser.add_argument("--output", dest="legacy_output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify":
        print(json.dumps(verify_report(args.verify_report, args.verify_root), sort_keys=True))
        return 0
    if args.command == "write":
        root = args.write_root
        output = args.write_report
    else:
        root = args.legacy_root
        output = args.legacy_output
    if root is None or output is None:
        parser.error("--root and --output are required when building")
    built = write_report(root, output)
    if args.command is None:
        (output / "README.md").write_text(_readme(built.summary), encoding="utf-8")
    print(json.dumps(verify_report(output, root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
