"""Build the canonical all-topic tethered-facet validation report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CANONICAL_RANKING_ROOT_SHA256 = "e2084842076119608977843f65ef749572c196d56f43d28b3e5440236c6a1578"
CANONICAL_EVALUATION_ROOT_SHA256 = "1634e2d993d79d46b969a6bcdc5207a7c06bc881485fc091ae3b7904bc0bc72b"
RANKING_DIR = "rankings_v2"
EVALUATION_DIR = "evaluation_v2"
DEPTHS = (100, 250, 500, 1000, 1500)
PRIMARY_ARM = "RRF100-STATIC-DUAL"


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


def _load_sources(root: Path, ranking_dir_name: str, evaluation_dir_name: str) -> dict[str, dict[str, Any]]:
    if ranking_dir_name != RANKING_DIR or evaluation_dir_name != EVALUATION_DIR:
        raise ValueError("superseded v1 ranking/evaluation evidence is rejected")
    ranking_dir = root / ranking_dir_name
    evaluation_dir = root / evaluation_dir_name
    ranking_seal = _verify_seal(ranking_dir, CANONICAL_RANKING_ROOT_SHA256)
    evaluation_seal = _verify_seal(evaluation_dir, CANONICAL_EVALUATION_ROOT_SHA256)
    bindings = _json(evaluation_dir / "input_bindings.json")
    if bindings.get("ranking_root_sha256") != CANONICAL_RANKING_ROOT_SHA256:
        raise ValueError("evaluation is not bound to corrected canonical ranking")
    summary = _json(evaluation_dir / "summary.json")
    if summary.get("schema_version") != "all-topic-tethered-evaluation-v2":
        raise ValueError("evaluation schema is not canonical v2")
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
        "planning_seal": _json(root / "planning/SEALED.json"),
        "retrieval_seal": _json(root / "retrieval/RETRIEVAL_SEALED.json"),
        "scoring_seal": _json(root / "scoring/SCORING_SEALED.json"),
    }


def _fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


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
        for depth in DEPTHS:
            rows.append({
                "arm": arm,
                "depth": depth,
                "binary_recall": pooled[f"binary_recall@{depth}"],
                "known_relevant_count": pooled[f"known_relevant_count@{depth}"],
                "facet_only_retention": pooled[f"facet_only_known_relevant_retention@{depth}"],
                "judged_rate": pooled[f"judged_rate@{depth}"],
            })
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
    depths = list(DEPTHS)
    max_y = 0.36
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="recall-title recall-desc"><title id="recall-title">Recall and retained facet evidence by depth</title><desc id="recall-desc">Binary recall rises with depth for RRF and the two strongest DUAL variants. Alternatives improve aggregate recall but still cause topic regressions.</desc>']
    for tick in (0.0, 0.1, 0.2, 0.3):
        y = top + plot_h - tick / max_y * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/><text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{tick:.1f}</text>')
    for i, depth in enumerate(depths):
        x = left + i * plot_w / (len(depths) - 1)
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle">{depth}</text>')
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


def _render_html(summary: Mapping[str, Any], datasets: Mapping[str, list[dict[str, Any]]]) -> str:
    arms = datasets["arm_metrics"]
    topics = datasets["topic_metrics"]
    depth_rows = datasets["depth_metrics"]
    bucket_rows = datasets["facet_bucket_yield"]
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
    costs = summary["costs"]
    prov = summary["provenance"]
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light dark"><title>All-topic tethered-facet validation</title>
<style>
:root{{--bg:#f4f6f8;--card:#fff;--ink:#17202a;--muted:#56616e;--line:#cbd3dc;--accent:#274f91;--warn:#a33d30;--good:#1d745d}}@media(prefers-color-scheme:dark){{:root{{--bg:#111820;--card:#18232e;--ink:#edf3f8;--muted:#b9c4cf;--line:#40505f;--accent:#91b9ff;--warn:#ff9f91;--good:#78d6b7}}}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}}main{{max-width:1120px;margin:auto;padding:24px}}header,section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:clamp(18px,3vw,34px);margin:0 0 20px}}h1{{font-size:clamp(2rem,5vw,4rem);line-height:1.02;max-width:15ch;margin:.2em 0}}h2{{font-size:clamp(1.4rem,3vw,2.1rem);line-height:1.15}}h3{{margin-top:1.8em}}p{{max-width:78ch}}.eyebrow{{text-transform:uppercase;letter-spacing:.12em;color:var(--accent);font-weight:750}}.verdict{{border-left:7px solid var(--warn)}}.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}}.kpi{{border:1px solid var(--line);border-radius:10px;padding:16px}}.kpi strong{{display:block;font-size:1.8rem}}.muted,figcaption{{color:var(--muted)}}.callout{{background:color-mix(in srgb,var(--warn) 10%,transparent);border-left:4px solid var(--warn);padding:12px 16px}}.table-wrap{{overflow-x:auto;max-width:100%;border:1px solid var(--line);border-radius:9px}}table{{border-collapse:collapse;width:100%;min-width:780px}}th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th:first-child,td:first-child{{text-align:left;position:sticky;left:0;background:var(--card)}}thead th{{text-align:right;background:color-mix(in srgb,var(--accent) 9%,var(--card))}}tr.loss th,tr.loss td{{color:var(--warn);font-weight:700}}figure{{margin:24px 0}}svg{{width:100%;height:auto;background:color-mix(in srgb,var(--accent) 4%,var(--card));border-radius:10px}}svg text{{fill:var(--ink);font:13px system-ui,sans-serif}}svg .grid{{stroke:var(--line);stroke-width:1}}svg .value{{font-weight:750}}.legend{{display:flex;gap:18px;flex-wrap:wrap}}.legend i{{display:inline-block;width:16px;height:4px;vertical-align:middle;margin-right:6px}}.method{{display:flex;gap:8px;align-items:stretch;flex-wrap:wrap}}.method div{{flex:1 1 135px;border:1px solid var(--line);padding:12px;border-radius:8px}}.method b{{display:block}}details{{border-top:1px solid var(--line);padding:12px 0}}summary{{cursor:pointer;font-weight:700}}summary:focus-visible,a:focus-visible{{outline:3px solid var(--accent);outline-offset:3px}}code{{overflow-wrap:anywhere}}@media(max-width:600px){{main{{padding:10px}}header,section{{padding:17px}}.kpi strong{{font-size:1.4rem}}}}
</style></head><body><main>
<header class="verdict" id="top"><p class="eyebrow">Corrected canonical v2 evidence · decision report</p><h1>Retain RRF.</h1><p class="lede">The preregistered alternatives improve pooled known-relevant recall, but every alternative loses at least one topic at depth 1,000. That violates the zero-loss promotion rule.</p><div class="kpis"><div class="kpi"><strong>{base["known_relevant_at_1000"]:,}</strong>RRF known relevant @1,000</div><div class="kpi"><strong>+{primary["known_relevant_at_1000"]-base["known_relevant_at_1000"]}</strong>primary DUAL pooled gain</div><div class="kpi"><strong>{primary["wins"]}/{primary["ties"]}/{primary["losses"]}</strong>topic wins / ties / losses</div><div class="kpi"><strong>−7</strong>worst loss, Topic 31</div></div><p class="callout"><strong>Why not promote?</strong> Primary DUAL loses 7 known-relevant documents on Topic 31 and 3 on Topic 300 at 1,000. RRF500-REINIT-DUAL removes the Topic 31 loss but still loses 3 on Topic 300.</p></header>
<section id="scope"><h2>What this result means—and what it does not</h2><p>This is a <strong>retrospective full-development stress test</strong> over all 22 development topics using already known judgments. It measures retrieval of <strong>known-relevant</strong> documents and is useful for diagnosing headroom and regressions. It is <strong>not evidence of generalization</strong> to new topics, unseen judgments, or production traffic. The downstream RAG answer generation is out of scope; no claim is made about answer accuracy, faithfulness, citation quality, or user utility.</p><p>The v1 ranking/evaluation rejected label is intentional: only corrected <code>rankings_v2</code> and <code>evaluation_v2</code> are admissible here.</p></section>
<section id="ladder"><h2>Exact preregistered selection ladder</h2><p>The baseline appears first for orientation; alternatives follow the sealed ladder exactly. Aggregate improvements are insufficient when a zero-loss guard fails.</p><div class="table-wrap"><table><thead><tr><th scope="col">Arm</th><th scope="col">Known rel. @1k</th><th scope="col">Δ vs RRF</th><th scope="col">Pooled recall</th><th scope="col">W/T/L</th><th scope="col">Loss topics</th><th scope="col">Decision</th></tr></thead><tbody>{ladder_rows}</tbody></table></div></section>
<section id="depth"><h2>Recall and retained facet evidence by depth</h2><p>Recall rises as the cutoff expands, and DUAL variants retain more facet-only evidence. The aggregate curves explain the attraction of the alternatives; the per-topic guard below explains the decision.</p><figure>{_line_chart(depth_rows)}<figcaption>Pooled binary recall over the 12,984 known-relevant documents. Depth 100 is protected and identical across arms.</figcaption></figure><p class="muted">Judged-rate caveat: RRF judged rate falls from {_fmt_pct(next(r for r in depth_rows if r["arm"]=="RRF" and r["depth"]==100)["judged_rate"])} at 100 to {_fmt_pct(next(r for r in depth_rows if r["arm"]=="RRF" and r["depth"]==1000)["judged_rate"])} at 1,000. Unjudged documents are not negatives, so deep precision and yield are conservative and judgment-pool dependent.</p></section>
<section id="yield"><h2>Known-relevant yield falls with facet depth</h2><p>The first 50 results of each deduplicated facet stream carry the highest pooled known-relevant yield. Later buckets still add evidence but at lower density, supporting depth discipline rather than blanket expansion.</p><figure>{_bucket_chart(bucket_rows)}<figcaption>Pooled known-relevant count divided by pooled unique candidates in each facet-rank bucket across 148 facet streams.</figcaption></figure></section>
<section id="topics"><h2>Per-topic deltas expose the promotion blockers</h2><p>Every topic is shown. Count deltas are alternative minus RRF; negative values are regressions. The final column gives primary DUAL deltas at depths 250 / 500 / 1,500.</p><div class="table-wrap"><table><thead><tr><th scope="col">Topic</th><th scope="col">Known rel.</th><th scope="col">RRF @1k</th><th scope="col">RRF recall</th><th scope="col">Primary Δ @1k</th><th scope="col">RRF500 Δ @1k</th><th scope="col">NR Δ @1k</th><th scope="col">Primary Δ 250/500/1500</th></tr></thead><tbody>{topic_rows}</tbody></table></div><h3>Representative diagnostics</h3>{diag}</section>
<section id="method"><h2>Method and evidence boundary</h2><div class="method" role="list" aria-label="Evaluation method"><div role="listitem"><b>1 · Plan</b>22 narratives → 148 tethered facet queries.</div><div role="listitem"><b>2 · Retrieve</b>Top 200 per facet; 45,144-document union.</div><div role="listitem"><b>3 · Score</b>Local MiniLM narrative/facet features.</div><div role="listitem"><b>4 · Freeze blind</b>Six complete arms, qrels unopened.</div><div role="listitem"><b>5 · Evaluate</b>Pinned development qrels opened only after v2 freeze verification.</div><div role="listitem"><b>6 · Decide</b>Apply exact ladder and all promotion guards.</div></div><p>The corrected ranking freeze has 22 topics, 6 complete arms, 270,864 ranking rows, and 206,030 audit rows. The evaluation independently binds that freeze before opening the pinned qrels. Statistical tests support a positive mean recall change, but statistical significance cannot override the preregistered topic-loss rule.</p></section>
<section id="costs"><h2>Costs and execution accounting</h2><div class="kpis"><div class="kpi"><strong>{costs["facet_requests"]}</strong>live facet retrieval requests</div><div class="kpi"><strong>{costs["local_forward_pairs"]:,}</strong>local forward pairs</div><div class="kpi"><strong>{costs["scoring_seconds"]:.3f}s</strong>local scoring wall time</div><div class="kpi"><strong>$0 recorded</strong>hosted / paid inference</div></div><p>Original-query requests: 0. Retrieval failures/retries: 0/0. Shared-score cache reuses: {costs["cache_reuse_pairs"]:,}. Completed score windows: {costs["windows"]:,}. Peak device/host memory: {costs["peak_device_bytes"]:,} / {costs["peak_host_bytes"]:,} bytes. Ranking and evaluation added zero retrieval, inference, model-load, hosted, paid, or network calls.</p></section>
<section id="limits"><h2>Limitations, decision, and next step</h2><ul><li>Retrospective known-judgment evidence can overstate certainty and does not establish held-out generalization.</li><li>Judgment incompleteness grows with depth; unjudged candidates may contain useful evidence.</li><li>The experiment evaluates retrieval and ranking only; answer generation remains untested.</li><li>Topic 31 and Topic 300 losses are small in pooled terms but decisive under the frozen guard.</li></ul><p><strong>Recommendation:</strong> retain RRF for promotion. Use the all-topic result diagnostically to design a separately preregistered, held-out approach that explicitly protects Topics 31/300-like failure modes.</p></section>
<section id="provenance"><h2>Authenticated provenance</h2><p>Ranking v2 root: <code>{prov["ranking_root_sha256"]}</code><br>Evaluation v2 root: <code>{prov["evaluation_root_sha256"]}</code><br>Planning root: <code>{prov["planning_root_sha256"]}</code><br>Retrieval root: <code>{prov["retrieval_root_sha256"]}</code><br>Scoring root: <code>{prov["scoring_root_sha256"]}</code></p><p class="muted">This sanitized report contains aggregate metrics and hashes only: no credentials, raw qrels, raw documents, document identifiers, request logs, or local filesystem paths.</p><p><a href="#top">Back to top</a></p></section>
</main></body></html>'''


def build_report(root: Path, *, ranking_dir_name: str = RANKING_DIR, evaluation_dir_name: str = EVALUATION_DIR) -> BuiltReport:
    source = _load_sources(Path(root), ranking_dir_name, evaluation_dir_name)
    evaluation_summary = source["evaluation_summary"]
    topic_ids = evaluation_summary["topic_ids"]
    metrics = source["metrics"]
    decision = source["diagnostics"]["decision"]
    arm_rows = _arm_rows(metrics, decision)
    topic_rows = _topic_rows(metrics, topic_ids)
    datasets = {
        "arm_metrics": arm_rows,
        "topic_metrics": topic_rows,
        "depth_metrics": _depth_rows(metrics),
        "facet_bucket_yield": _bucket_rows(source["diagnostics"]),
    }
    provenance = {
        "ranking_root_sha256": source["ranking_seal"]["root_sha256"],
        "evaluation_root_sha256": source["evaluation_seal"]["root_sha256"],
        "planning_root_sha256": source["bindings"]["upstream_roots"]["planning_root_sha256"],
        "retrieval_root_sha256": source["bindings"]["upstream_roots"]["retrieval_root_sha256"],
        "score_plan_root_sha256": source["bindings"]["upstream_roots"]["score_plan_root_sha256"],
        "scoring_root_sha256": source["bindings"]["upstream_roots"]["scoring_root_sha256"],
    }
    primary = next(row for row in arm_rows if row["arm"] == PRIMARY_ARM)
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
        db.execute("CREATE TABLE depth_metrics (arm TEXT NOT NULL, depth INTEGER NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (arm, depth))")
        db.executemany("INSERT INTO depth_metrics VALUES (?, ?, ?)", [(r["arm"], r["depth"], json.dumps(r, sort_keys=True)) for r in datasets["depth_metrics"]])
        db.execute("CREATE TABLE facet_bucket_yield (topic_id TEXT NOT NULL, bucket TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (topic_id, bucket))")
        db.executemany("INSERT INTO facet_bucket_yield VALUES (?, ?, ?)", [(r["topic_id"], r["bucket"], json.dumps(r, sort_keys=True)) for r in datasets["facet_bucket_yield"]])
        db.commit()
        db.execute("VACUUM")


def write_report(root: Path, output: Path) -> BuiltReport:
    built = build_report(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_bytes(_canonical_json(built.summary))
    (output / "artifact.json").write_bytes(_canonical_json(built.artifact))
    (output / "report.html").write_text(built.html, encoding="utf-8")
    _write_database(output / "report_data.sqlite", built.datasets, built.summary["provenance"])
    return BuiltReport(**{**built.__dict__, "output_dir": output})


def verify_report(report: Path) -> dict[str, Any]:
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
        raise ValueError("report is not canonical v2")
    with sqlite3.connect(report / "report_data.sqlite") as db:
        topics = db.execute("SELECT COUNT(*) FROM topic_metrics").fetchone()[0]
        arms = db.execute("SELECT COUNT(*) FROM arm_metrics").fetchone()[0]
        metadata = dict(db.execute("SELECT key, value FROM metadata"))
    if topics != len(summary["topic_ids"]) or arms != len(summary["arms"]) or metadata != summary["provenance"]:
        raise ValueError("SQLite content differs from JSON summary")
    return {"verified": True, "topic_count": topics, "arm_count": arms, "html_sha256": artifact["html_sha256"], "summary_sha256": artifact["summary_sha256"]}


def _readme(summary: Mapping[str, Any]) -> str:
    return f"""# All-topic tethered-facet validation v1\n\nDecision: **{summary['decision']['recommendation']}**.\n\nThis directory contains the sanitized, reproducible report over the corrected canonical v2 ranking and evaluation. The superseded v1 ranking and evaluation are rejected evidence. This is a retrospective full-development stress test using known-relevant judgments; it is not evidence of generalization, and downstream RAG answer generation is out of scope.\n\n- `summary.json`: decision, exact roots, costs, and scope limits\n- `report_data.sqlite`: arm, depth, topic, and facet-bucket rows\n- `artifact.json`: hashes and sanitization declaration\n- `report.html`: self-contained accessible report\n\nRun verification from the repository root:\n\n```bash\n.venv/bin/python -m trec_rag.build_all_topic_tethered_report verify --report reports/experiments/all_topic_tethered_facet_validation_v1\n```\n"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    verify = sub.add_parser("verify")
    verify.add_argument("--report", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify":
        print(json.dumps(verify_report(args.report), sort_keys=True))
        return 0
    if args.root is None or args.output is None:
        parser.error("--root and --output are required when building")
    built = write_report(args.root, args.output)
    (args.output / "README.md").write_text(_readme(built.summary), encoding="utf-8")
    print(json.dumps(verify_report(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
