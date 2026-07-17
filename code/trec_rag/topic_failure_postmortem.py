"""Offline postmortem helpers for the sealed all-topic retrieval experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from .all_topic_facet_contract import EXPERIMENT_ID
from .all_topic_tethered_evaluate import (
    FACET_RANK_BUCKETS,
    PINNED_QRELS_SHA256,
    _load_qrels,
)
from .all_topic_tethered_rank import (
    CANONICAL_RANKING_ROOT_SHA256,
    PLANNING_ROOT_SHA256,
    RETRIEVAL_ROOT_SHA256,
    SCORE_PLAN_ROOT_SHA256,
    SCORING_ROOT_SHA256,
    _features_for_authorized_topic,
    verify_rankings,
)


SCHEMA_VERSION = "topic-failure-postmortem-v1"
TOPIC_IDS = ("31", "300")
PRIMARY_ARM = "RRF100-STATIC-DUAL"
RRF500_ARM = "RRF500-REINIT-DUAL"
NO_REDUNDANCY_ARM = "RRF100-STATIC-DUAL-NR"
ANALYSIS_DEPTHS = (250, 500, 1000, 1500)


def _label_counts(document_ids: set[str], qrels: Mapping[str, int]) -> dict[str, int]:
    known_relevant = sum(qrels[document_id] >= 2 for document_id in document_ids if document_id in qrels)
    judged_below_2 = sum(qrels[document_id] < 2 for document_id in document_ids if document_id in qrels)
    return {
        "total": len(document_ids),
        "known_relevant": known_relevant,
        "judged_below_2": judged_below_2,
        "unjudged": len(document_ids) - known_relevant - judged_below_2,
    }


def analyze_boundary_changes(
    baseline: Sequence[str],
    candidate: Sequence[str],
    qrels: Mapping[str, int],
    depth: int,
) -> dict[str, object]:
    """Summarize label changes between two ranking prefixes at ``depth``."""

    if depth < 0:
        raise ValueError("depth must be nonnegative")
    baseline_prefix = set(baseline[:depth])
    candidate_prefix = set(candidate[:depth])
    baseline_counts = _label_counts(baseline_prefix, qrels)
    candidate_counts = _label_counts(candidate_prefix, qrels)
    return {
        "depth": depth,
        "baseline": baseline_counts,
        "candidate": candidate_counts,
        "known_relevant_delta": candidate_counts["known_relevant"]
        - baseline_counts["known_relevant"],
        "outgoing": _label_counts(baseline_prefix - candidate_prefix, qrels),
        "incoming": _label_counts(candidate_prefix - baseline_prefix, qrels),
    }


def eligible_documents_for_facet_cap(
    provenance: Mapping[str, Mapping[str, object]], cap: int
) -> set[str]:
    """Return original-stream documents plus facet-only documents within ``cap``."""

    if cap < 1:
        raise ValueError("facet rank cap must be positive")
    eligible: set[str] = set()
    for document_id, row in provenance.items():
        if row.get("original_rank") is not None:
            eligible.add(document_id)
            continue
        ranks = row.get("facet_ranks", ())
        if (
            isinstance(ranks, Sequence)
            and not isinstance(ranks, (str, bytes, bytearray))
            and any(isinstance(rank, int) and not isinstance(rank, bool) and 1 <= rank <= cap for rank in ranks)
        ):
            eligible.add(document_id)
    return eligible


def _complete_permutation(value: object, population: set[str], label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence")
    result = [str(document_id) for document_id in value]
    if len(result) != len(population) or set(result) != population:
        raise ValueError(f"{label} must be one complete accepted-union permutation")
    return result


def _topic_provenance(topic_input: Mapping[str, object]) -> dict[str, dict[str, object]]:
    docids = topic_input.get("docids")
    original = topic_input.get("original_rank")
    facets = topic_input.get("facets")
    if (
        not isinstance(docids, Sequence)
        or isinstance(docids, (str, bytes, bytearray))
        or not isinstance(original, Mapping)
        or not isinstance(facets, Sequence)
        or isinstance(facets, (str, bytes, bytearray))
    ):
        raise ValueError("topic input is missing accepted-union provenance")
    result = {
        str(document_id): {
            "original_rank": original.get(document_id),
            "facet_ranks": [],
        }
        for document_id in docids
    }
    for raw_facet in facets:
        if not isinstance(raw_facet, Mapping) or not isinstance(raw_facet.get("bm25_rank"), Mapping):
            raise ValueError("topic facet provenance is invalid")
        for document_id, rank in raw_facet["bm25_rank"].items():  # type: ignore[index]
            document_key = str(document_id)
            if document_key not in result:
                raise ValueError("facet provenance is outside the accepted union")
            facet_ranks = result[document_key]["facet_ranks"]
            assert isinstance(facet_ranks, list)
            facet_ranks.append(int(rank))
    return result


def replay_recovery_arms(
    topic_input: Mapping[str, object],
    qrels: Mapping[str, int],
    *,
    facet_rank_cap: int = 100,
) -> dict[str, object]:
    """Replay the bounded facet cap and one diagnostic-only score ablation."""

    raw_docids = topic_input.get("docids")
    raw_rankings = topic_input.get("rankings")
    if (
        not isinstance(raw_docids, Sequence)
        or isinstance(raw_docids, (str, bytes, bytearray))
        or not isinstance(raw_rankings, Mapping)
    ):
        raise ValueError("topic input must contain docids and canonical rankings")
    population = {str(document_id) for document_id in raw_docids}
    if len(population) != len(raw_docids) or not population:
        raise ValueError("docids must be a unique accepted-union population")
    rrf = _complete_permutation(raw_rankings.get("RRF"), population, "RRF")
    primary = _complete_permutation(
        raw_rankings.get("RRF100-STATIC-DUAL"), population, "primary DUAL"
    )
    no_redundancy = _complete_permutation(
        raw_rankings.get(NO_REDUNDANCY_ARM), population, "no-redundancy DUAL"
    )
    protected_depth = topic_input.get("protected_prefix_depth", 100)
    if (
        isinstance(protected_depth, bool)
        or not isinstance(protected_depth, int)
        or not 0 <= protected_depth <= len(rrf)
    ):
        raise ValueError("protected prefix depth is invalid")
    if primary[:protected_depth] != rrf[:protected_depth]:
        raise ValueError("primary DUAL protected prefix differs from canonical RRF")

    eligible = eligible_documents_for_facet_cap(
        _topic_provenance(topic_input), facet_rank_cap
    )
    protected = rrf[:protected_depth]
    protected_set = set(protected)
    promotable = [
        document_id
        for document_id in primary
        if document_id not in protected_set and document_id in eligible
    ]
    cap_order = protected + promotable
    cap_seen = set(cap_order)
    deferred = [document_id for document_id in rrf if document_id not in cap_seen]
    cap_order.extend(deferred)

    features = _features_for_authorized_topic(topic_input)
    narrative = features["N"]
    assert isinstance(narrative, Mapping)
    fixed_objectives = topic_input.get("fixed_objectives")
    if not isinstance(fixed_objectives, Mapping) or not isinstance(
        fixed_objectives.get(NO_REDUNDANCY_ARM), Mapping
    ):
        raise ValueError("no-narrative diagnostic requires frozen no-redundancy objectives")
    raw_objectives = fixed_objectives[NO_REDUNDANCY_ARM]
    assert isinstance(raw_objectives, Mapping)
    expected_objectives = population - protected_set
    if set(map(str, raw_objectives)) != expected_objectives:
        raise ValueError("frozen no-redundancy objective coverage differs")
    objectives = {str(document_id): float(value) for document_id, value in raw_objectives.items()}
    if any(not math.isfinite(value) for value in objectives.values()):
        raise ValueError("frozen no-redundancy objective is not finite")
    no_redundancy_rank = {
        document_id: rank for rank, document_id in enumerate(no_redundancy)
    }
    fixed_no_narrative = sorted(
        expected_objectives,
        key=lambda document_id: (
            -(objectives[document_id] - 0.15 * float(narrative[document_id])),
            no_redundancy_rank[document_id],
        ),
    )
    no_narrative_order = protected + fixed_no_narrative

    rankings = {
        "RRF": rrf,
        "canonical_primary": primary,
        f"facet_rank_cap_{facet_rank_cap}": cap_order,
        "no_narrative_score": no_narrative_order,
    }
    permutation_checks = {
        arm: len(order) == len(population) and set(order) == population
        for arm, order in rankings.items()
    }
    if not all(permutation_checks.values()):
        raise ValueError("recovery replay did not preserve the accepted-union permutation")
    analysis_depth = min(1000, len(rrf))
    return {
        "facet_rank_cap": facet_rank_cap,
        "protected_prefix_depth": protected_depth,
        "protected_prefix_identical": {
            arm: order[:protected_depth] == protected
            for arm, order in rankings.items()
            if arm != "RRF"
        },
        "eligible_document_count": len(eligible),
        "ineligible_document_count": len(population - eligible),
        "deferred_document_count": len(deferred),
        "permutation_checks": permutation_checks,
        "rankings": rankings,
        "at_1000": {
            arm: analyze_boundary_changes(rrf, order, qrels, analysis_depth)
            for arm, order in rankings.items()
            if arm != "RRF"
        },
        "no_narrative_score": {
            "status": "post-hoc diagnostic",
            "promotion_eligible": False,
            "base_arm": NO_REDUNDANCY_ARM,
            "method": "fixed objective minus 0.15*N; no greedy replay",
            "removed_weight": "N",
            "unchanged_weights": ["G", "R", "L", "B"],
            "redundancy_state": "fixed zero inherited from diagnostic base arm",
        },
    }


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _movement_markdown(boundary: Mapping[str, object]) -> str:
    rows = []
    for label in ("outgoing", "incoming"):
        counts = _required_mapping(boundary.get(label), f"{label} boundary counts")
        rows.append(
            f"| {label.title()} | {int(counts['total'])} | "
            f"{int(counts['known_relevant'])} | {int(counts['judged_below_2'])} | "
            f"{int(counts['unjudged'])} |"
        )
    return "\n".join(
        [
            "| Movement | Total | Known relevant (grade ≥2) | Judged below 2 | Unjudged (unknown) |",
            "|---|---:|---:|---:|---:|",
            *rows,
        ]
    )


def render_postmortem(analysis: Mapping[str, object]) -> str:
    """Render a deterministic, source-backed Markdown postmortem."""

    topics = _required_mapping(analysis.get("topics"), "topics")
    topic31 = _required_mapping(topics.get("31"), "Topic 31")
    topic300 = _required_mapping(topics.get("300"), "Topic 300")
    primary31 = _required_mapping(topic31.get("primary_at_1000"), "Topic 31 primary boundary")
    primary300 = _required_mapping(topic300.get("primary_at_1000"), "Topic 300 primary boundary")
    recovery = _required_mapping(topic300.get("recovery_replay"), "Topic 300 recovery replay")
    arms = _required_mapping(recovery.get("arms"), "recovery arms")
    cap_name = next((name for name in arms if name.startswith("facet_rank_cap_")), None)
    if cap_name is None:
        raise ValueError("recovery replay is missing the facet-rank-cap arm")
    cap = _required_mapping(arms[cap_name], "facet-rank-cap arm")
    diagnostic = _required_mapping(arms.get("no_narrative_score"), "no-narrative arm")
    bucket_yield = _required_mapping(topic300.get("facet_bucket_yield"), "facet bucket yield")
    checks = _required_mapping(recovery.get("permutation_checks"), "permutation checks")
    provenance = _required_mapping(analysis.get("provenance"), "provenance")

    lines = [
        "# Topic 31/300 retrieval-failure postmortem",
        "",
        "Decision: retain canonical RRF. The bounded Topic 300 replay is recovery evidence, not a promotion result.",
        "",
        "## Evidence boundary",
        "",
        "Relevance means UMBRELA grade 2 or higher. Every unjudged (unknown) document remains separate from judged-below-2 evidence and is never described as nonrelevant. All replays are offline over frozen accepted-union candidates and features.",
        "",
        "## Topic 31 cutoff mechanics",
        "",
        f"Primary DUAL changes known-relevant capture by {int(primary31['known_relevant_delta']):+d} at depth 1,000.",
        "",
        _movement_markdown(primary31),
        "",
        "## Topic 300 cutoff mechanics",
        "",
        f"Primary DUAL changes known-relevant capture by {int(primary300['known_relevant_delta']):+d} at depth 1,000.",
        "",
        _movement_markdown(primary300),
        "",
        "### Judgment-pool dependent facet-tail yield",
        "",
        "These are known-relevant yields within the existing judgment pool; unjudged candidates remain unknown.",
        "",
        "| Per-facet retrieval rank | Known-relevant yield |",
        "|---|---:|",
    ]
    for bucket in ("1-50", "51-100", "101-150", "151-200"):
        lines.append(f"| {bucket} | {float(bucket_yield[bucket]):.2%} |")
    lines.extend(
        [
            "",
            "### Bounded offline recovery replay",
            "",
            f"Protected RRF prefix: {int(recovery['protected_prefix_depth'])}",
            "",
            f"The `{cap_name}` arm changes known-relevant capture by {int(cap['known_relevant_delta']):+d} at depth 1,000. Original-stream candidates stay eligible; facet-only candidates require a facet retrieval rank within the cap. Deferred candidates return in canonical RRF order.",
            "",
            "Complete accepted-union permutation checks: "
            + ", ".join(f"`{name}`={'pass' if bool(value) else 'FAIL'}" for name, value in checks.items())
            + ".",
            "",
            f"The no-narrative-score arm changes known-relevant capture by {int(diagnostic['known_relevant_delta']):+d}. It is a {diagnostic['status']} and is not promotion-eligible. Method: {diagnostic['method']} from `{diagnostic['base_arm']}` ({diagnostic['redundancy_state']}); every remaining DUAL weight is unchanged.",
            "",
            "## Authenticated provenance",
            "",
            f"- Ranking root: `{provenance['ranking_root_sha256']}`",
            f"- Retrieval root: `{provenance['retrieval_root_sha256']}`",
            f"- Scoring root: `{provenance['scoring_root_sha256']}`",
            "- Retrieval, inference, download, model-load, hosted, and paid calls during replay: 0.",
            "",
        ]
    )
    return "\n".join(lines)


def _compact_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _file_binding(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ValueError(f"authenticated source file is unreadable: {path.name}") from exc
    return {"bytes": size, "sha256": digest.hexdigest()}


def _verify_root_seal(
    path: Path,
    expected_root: str,
    label: str,
    *,
    canonical_trailing_newline: bool = False,
) -> dict[str, object]:
    seal = _read_json(path, label)
    files = seal.get("files")
    canonical = _compact_bytes(files)
    if canonical_trailing_newline:
        canonical += b"\n"
    if (
        not isinstance(files, Mapping)
        or seal.get("root_sha256") != expected_root
        or _sha256(canonical) != expected_root
    ):
        raise ValueError(f"{label} root differs from the canonical seal")
    return seal


def _verify_sources(
    source_root: Path, qrels_path: Path, facet_manifest_path: Path
) -> dict[str, object]:
    ranking = verify_rankings(source_root / "rankings_v3")
    if ranking.get("root_sha256") != CANONICAL_RANKING_ROOT_SHA256:
        raise ValueError("canonical ranking verification differs")

    planning = _verify_root_seal(
        source_root / "planning" / "SEALED.json",
        PLANNING_ROOT_SHA256,
        "planning seal",
        canonical_trailing_newline=True,
    )
    retrieval = _verify_root_seal(
        source_root / "retrieval" / "RETRIEVAL_SEALED.json",
        RETRIEVAL_ROOT_SHA256,
        "retrieval seal",
    )
    score_plan = _verify_root_seal(
        source_root / "scoring" / "SCORE_PLAN_SEALED.json",
        SCORE_PLAN_ROOT_SHA256,
        "score-plan seal",
    )
    scoring = _verify_root_seal(
        source_root / "scoring" / "SCORING_SEALED.json",
        SCORING_ROOT_SHA256,
        "scoring seal",
    )
    if scoring.get("score_plan_root_sha256") != SCORE_PLAN_ROOT_SHA256:
        raise ValueError("scoring seal is not bound to the canonical score plan")

    bindings = _read_json(
        source_root / "rankings_v3" / "input_bindings.json", "ranking input bindings"
    )
    accepted = _required_mapping(bindings.get("accepted_union"), "accepted-union binding")
    features = _required_mapping(bindings.get("features"), "feature binding")
    accepted_actual = _file_binding(source_root / "retrieval" / "accepted_union.jsonl")
    feature_actual = _file_binding(source_root / "scoring" / "features.jsonl")
    expected_accepted = {key: accepted.get(key) for key in ("bytes", "sha256")}
    expected_features = {key: features.get(key) for key in ("bytes", "sha256")}
    if accepted_actual != expected_accepted or feature_actual != expected_features:
        raise ValueError("consumed source files differ from canonical ranking bindings")
    retrieval_files = _required_mapping(retrieval.get("files"), "retrieval seal files")
    scoring_files = _required_mapping(scoring.get("files"), "scoring seal files")
    if retrieval_files.get("accepted_union.jsonl") != accepted_actual:
        raise ValueError("accepted union differs from the retrieval seal")
    if scoring_files.get("features.jsonl") != feature_actual:
        raise ValueError("features differ from the scoring seal")

    qrels_content = qrels_path.read_bytes()
    if _sha256(qrels_content) != PINNED_QRELS_SHA256:
        raise ValueError("qrels differ from the pinned UMBRELA judgments")
    facet_content = facet_manifest_path.read_bytes()
    planning_files = _required_mapping(planning.get("files"), "planning seal files")
    manifest_binding = planning_files.get("manifest.json")
    facet_binding = {"bytes": len(facet_content), "sha256": _sha256(facet_content)}
    if facet_binding != manifest_binding:
        raise ValueError("facet manifest differs from the canonical planning manifest")
    if _file_binding(source_root / "planning" / "manifest.json") != facet_binding:
        raise ValueError("source planning manifest differs from the tracked facet manifest")

    return {
        "ranking_root_sha256": CANONICAL_RANKING_ROOT_SHA256,
        "planning_root_sha256": PLANNING_ROOT_SHA256,
        "retrieval_root_sha256": RETRIEVAL_ROOT_SHA256,
        "score_plan_root_sha256": SCORE_PLAN_ROOT_SHA256,
        "scoring_root_sha256": SCORING_ROOT_SHA256,
        "qrels_sha256": PINNED_QRELS_SHA256,
        "facet_manifest_sha256": facet_binding["sha256"],
        "accepted_union_sha256": accepted_actual["sha256"],
        "features_sha256": feature_actual["sha256"],
    }


def _load_rankings(source_root: Path) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {
        topic_id: defaultdict(list) for topic_id in TOPIC_IDS
    }
    path = source_root / "rankings_v3" / "rankings.jsonl"
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            topic_id = str(row.get("topic_id"))
            if topic_id in result:
                result[topic_id][str(row["arm"])].append(str(row["document_id"]))
    required = {"RRF", PRIMARY_ARM, RRF500_ARM, NO_REDUNDANCY_ARM}
    if any(not required <= set(result[topic_id]) for topic_id in TOPIC_IDS):
        raise ValueError("canonical rankings are missing postmortem arms")
    return {topic_id: dict(arms) for topic_id, arms in result.items()}


def _load_audit(
    source_root: Path,
) -> tuple[dict[str, dict[str, str | None]], dict[str, dict[str, float]]]:
    coverage = {topic_id: {} for topic_id in TOPIC_IDS}
    no_redundancy_objectives = {topic_id: {} for topic_id in TOPIC_IDS}
    path = source_root / "rankings_v3" / "audit.jsonl"
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            topic_id = str(row.get("topic_id"))
            if (
                topic_id in coverage
                and row.get("arm") == PRIMARY_ARM
                and row.get("kind") == "selection"
            ):
                document_id = str(row["document_id"])
                value = row.get("coverage_facet")
                coverage[topic_id][document_id] = str(value) if value is not None else None
            if (
                topic_id in no_redundancy_objectives
                and row.get("arm") == NO_REDUNDANCY_ARM
                and row.get("kind") == "selection"
            ):
                no_redundancy_objectives[topic_id][str(row["document_id"])] = float(
                    row["objective"]
                )
    return coverage, no_redundancy_objectives


def _load_provenance(
    source_root: Path,
) -> tuple[
    dict[str, dict[str, list[dict[str, object]]]],
    dict[str, str],
]:
    provenance = {topic_id: {} for topic_id in TOPIC_IDS}
    topic300_texts: dict[str, str] = {}
    path = source_root / "retrieval" / "accepted_union.jsonl"
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            topic_id = str(row.get("topic_id"))
            if topic_id not in provenance:
                continue
            document_id = str(row["document_id"])
            raw_entries = row.get("stream_provenance")
            if not isinstance(raw_entries, list) or document_id in provenance[topic_id]:
                raise ValueError("accepted-union provenance is invalid")
            entries = []
            for raw in raw_entries:
                if not isinstance(raw, Mapping):
                    raise ValueError("accepted-union stream provenance is invalid")
                entries.append(
                    {
                        "stream_id": str(raw.get("stream_id")),
                        "stream_rank": int(raw["stream_rank"]),
                    }
                )
            provenance[topic_id][document_id] = entries
            if topic_id == "300":
                topic300_texts[document_id] = str(row.get("text"))
    return provenance, topic300_texts


def _facet_rows(facet_manifest: Mapping[str, object], topic_id: str) -> list[dict[str, object]]:
    raw_facets = facet_manifest.get("facets")
    if not isinstance(raw_facets, list):
        raise ValueError("facet manifest is missing facets")
    result = [dict(row) for row in raw_facets if isinstance(row, Mapping) and str(row.get("topic_id")) == topic_id]
    result.sort(key=lambda row: (int(row["manifest_order"]), str(row["facet_id"])))
    if not result or len({str(row["facet_id"]) for row in result}) != len(result):
        raise ValueError(f"Topic {topic_id} facet manifest is invalid")
    return result


def _load_topic300_input(
    source_root: Path,
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    provenance: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    texts: Mapping[str, str],
    facet_manifest: Mapping[str, object],
    no_redundancy_objectives: Mapping[str, float],
) -> dict[str, object]:
    facet_rows = _facet_rows(facet_manifest, "300")
    facet_ids = [str(row["facet_id"]) for row in facet_rows]
    scores: dict[str, dict[str, float]] = {
        query_id: {} for query_id in ("g", "n", *facet_ids)
    }
    path = source_root / "scoring" / "features.jsonl"
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if str(row.get("topic_id")) != "300":
                continue
            query_id = str(row.get("query_id"))
            if query_id not in scores:
                raise ValueError("Topic 300 has an unexpected feature query")
            document_id = str(row["document_id"])
            if document_id in scores[query_id]:
                raise ValueError("Topic 300 feature identity is duplicated")
            scores[query_id][document_id] = float(row["percentile"])

    documents = set(provenance["300"])
    if set(texts) != documents or set(scores["g"]) != documents or set(scores["n"]) != documents:
        raise ValueError("Topic 300 global feature population differs")
    original_rank: dict[str, int] = {}
    facet_rank = {facet_id: {} for facet_id in facet_ids}
    for document_id, entries in provenance["300"].items():
        for entry in entries:
            stream_id = str(entry.get("stream_id"))
            rank = int(entry["stream_rank"])
            if stream_id == "original":
                original_rank[document_id] = rank
            elif stream_id in facet_rank:
                facet_rank[stream_id][document_id] = rank
            else:
                raise ValueError("Topic 300 provenance references an unknown facet")
    facets = []
    for row in facet_rows:
        facet_id = str(row["facet_id"])
        if set(scores[facet_id]) != set(facet_rank[facet_id]):
            raise ValueError(f"Topic 300 feature/retrieval population differs for {facet_id}")
        facets.append(
            {
                "facet_id": facet_id,
                "manifest_order": int(row["manifest_order"]),
                "scores": scores[facet_id],
                "bm25_rank": facet_rank[facet_id],
            }
        )
    return {
        "topic_id": "300",
        "docids": list(provenance["300"]),
        "texts": dict(texts),
        "original_rank": original_rank,
        "facets": facets,
        "common_scores": scores["g"],
        "narrative_scores": scores["n"],
        "rankings": {
            "RRF": list(rankings["300"]["RRF"]),
            PRIMARY_ARM: list(rankings["300"][PRIMARY_ARM]),
            NO_REDUNDANCY_ARM: list(rankings["300"][NO_REDUNDANCY_ARM]),
        },
        "fixed_objectives": {
            NO_REDUNDANCY_ARM: dict(no_redundancy_objectives),
        },
        "protected_prefix_depth": 100,
    }


def _bucket_for_rank(rank: int | None) -> str:
    if rank is None:
        return "none"
    for low, high in FACET_RANK_BUCKETS:
        if low <= rank <= high:
            return f"{low}-{high}"
    return ">200"


def _attribution(
    document_ids: set[str],
    provenance: Mapping[str, Sequence[Mapping[str, object]]],
    audit: Mapping[str, str | None],
) -> dict[str, object]:
    source_family = Counter()
    best_facet_bucket = Counter()
    coverage_facet = Counter()
    for document_id in document_ids:
        entries = provenance[document_id]
        has_original = any(str(entry.get("stream_id")) == "original" for entry in entries)
        facet_ranks = [
            int(entry["stream_rank"])
            for entry in entries
            if str(entry.get("stream_id")) != "original"
        ]
        if has_original and facet_ranks:
            source_family["original_and_facet"] += 1
        elif has_original:
            source_family["original_only"] += 1
        else:
            source_family["facet_only"] += 1
        best_facet_bucket[_bucket_for_rank(min(facet_ranks) if facet_ranks else None)] += 1
        coverage_facet[audit.get(document_id) or "none"] += 1
    return {
        "total": len(document_ids),
        "source_family": dict(sorted(source_family.items())),
        "best_facet_rank_bucket": dict(sorted(best_facet_bucket.items())),
        "primary_selection_coverage_facet": dict(sorted(coverage_facet.items())),
    }


def _facet_bucket_yield(
    provenance: Mapping[str, Sequence[Mapping[str, object]]],
    qrels: Mapping[str, int],
) -> dict[str, float]:
    relevant = {document_id for document_id, grade in qrels.items() if grade >= 2}
    result: dict[str, float] = {}
    for low, high in FACET_RANK_BUCKETS:
        documents = {
            document_id
            for document_id, entries in provenance.items()
            if any(
                str(entry.get("stream_id")) != "original"
                and low <= int(entry["stream_rank"]) <= high
                for entry in entries
            )
        }
        result[f"{low}-{high}"] = len(documents & relevant) / len(documents) if documents else 0.0
    return result


def _topic_analysis(
    topic_id: str,
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, int],
    provenance: Mapping[str, Sequence[Mapping[str, object]]],
    audit: Mapping[str, str | None],
) -> dict[str, object]:
    rrf = rankings["RRF"]
    primary = rankings[PRIMARY_ARM]
    boundaries = {
        str(depth): analyze_boundary_changes(rrf, primary, qrels, depth)
        for depth in ANALYSIS_DEPTHS
    }
    at_1000 = boundaries["1000"]
    baseline_prefix = set(rrf[:1000])
    primary_prefix = set(primary[:1000])
    return {
        "union_size": len(rrf),
        "known_relevant_total": sum(grade >= 2 for grade in qrels.values()),
        "primary_by_depth": boundaries,
        "primary_at_1000": at_1000,
        "rrf500_at_1000": analyze_boundary_changes(
            rrf, rankings[RRF500_ARM], qrels, 1000
        ),
        "no_redundancy_at_1000": analyze_boundary_changes(
            rrf, rankings[NO_REDUNDANCY_ARM], qrels, 1000
        ),
        "primary_boundary_attribution": {
            "outgoing": _attribution(baseline_prefix - primary_prefix, provenance, audit),
            "incoming": _attribution(primary_prefix - baseline_prefix, provenance, audit),
        },
    }


def _sanitize_replay(replay: Mapping[str, object]) -> dict[str, object]:
    at_1000 = _required_mapping(replay.get("at_1000"), "replay boundaries")
    cap_name = f"facet_rank_cap_{int(replay['facet_rank_cap'])}"
    cap = dict(_required_mapping(at_1000.get(cap_name), "facet-cap boundary"))
    diagnostic = dict(
        _required_mapping(at_1000.get("no_narrative_score"), "no-narrative boundary")
    )
    diagnostic.update(
        _required_mapping(replay.get("no_narrative_score"), "no-narrative metadata")
    )
    return {
        "facet_rank_cap": replay["facet_rank_cap"],
        "protected_prefix_depth": replay["protected_prefix_depth"],
        "protected_prefix_identical": replay["protected_prefix_identical"],
        "eligible_document_count": replay["eligible_document_count"],
        "ineligible_document_count": replay["ineligible_document_count"],
        "deferred_document_count": replay["deferred_document_count"],
        "permutation_checks": replay["permutation_checks"],
        "arms": {cap_name: cap, "no_narrative_score": diagnostic},
    }


def _assert_sanitized(analysis: Mapping[str, object]) -> None:
    serialized = _pretty_bytes(analysis).decode("utf-8")
    forbidden = ("/home/", "Bearer ", '"text"', '"document_id"', "shard_")
    if any(value in serialized for value in forbidden):
        raise ValueError("postmortem output contains a forbidden path, credential, text, or identifier")


def _build_analysis(
    source_root: Path, qrels_path: Path, facet_manifest_path: Path
) -> dict[str, object]:
    provenance_roots = _verify_sources(source_root, qrels_path, facet_manifest_path)
    qrels_content = qrels_path.read_bytes()
    qrels = _load_qrels(qrels_content)
    facet_manifest = _read_json(facet_manifest_path, "facet manifest")
    rankings = _load_rankings(source_root)
    audit, no_redundancy_objectives = _load_audit(source_root)
    provenance, topic300_texts = _load_provenance(source_root)
    for topic_id in TOPIC_IDS:
        population = set(rankings[topic_id]["RRF"])
        if set(provenance[topic_id]) != population:
            raise ValueError(f"Topic {topic_id} provenance differs from canonical RRF")

    topics = {
        topic_id: _topic_analysis(
            topic_id,
            rankings[topic_id],
            qrels[topic_id],
            provenance[topic_id],
            audit[topic_id],
        )
        for topic_id in TOPIC_IDS
    }
    topics["300"]["facet_bucket_yield"] = _facet_bucket_yield(
        provenance["300"], qrels["300"]
    )
    topic300_input = _load_topic300_input(
        source_root,
        rankings,
        provenance,
        topic300_texts,
        facet_manifest,
        no_redundancy_objectives["300"],
    )
    replay = replay_recovery_arms(topic300_input, qrels["300"], facet_rank_cap=100)
    topics["300"]["recovery_replay"] = _sanitize_replay(replay)
    analysis: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "scope": {
            "topic_ids": list(TOPIC_IDS),
            "relevance_threshold": 2,
            "unjudged_interpretation": "unknown; never nonrelevant",
            "replay_type": "offline frozen-candidate analysis",
            "document_identifiers_emitted": False,
            "raw_model_scores_emitted": False,
        },
        "execution_counters": {
            "retrieval_calls": 0,
            "network_calls": 0,
            "download_calls": 0,
            "model_loads": 0,
            "inference_calls": 0,
            "hosted_calls": 0,
            "paid_calls": 0,
        },
        "provenance": provenance_roots,
        "topics": topics,
        "recommendation": "Retain canonical RRF; use the cap replay only as bounded recovery evidence while beginning source-diverse RAG evidence selection.",
    }
    assert topics["31"]["primary_at_1000"]["known_relevant_delta"] == -7  # type: ignore[index]
    assert topics["300"]["primary_at_1000"]["known_relevant_delta"] == -3  # type: ignore[index]
    assert topics["300"]["facet_bucket_yield"] == {
        "1-50": 0.36180904522613067,
        "51-100": 0.29292929292929293,
        "101-150": 0.06,
        "151-200": 0.08,
    }
    _assert_sanitized(analysis)
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--facet-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analysis = _build_analysis(args.source_root, args.qrels, args.facet_manifest)
    markdown = render_postmortem(analysis)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "postmortem.json").write_bytes(_pretty_bytes(analysis))
    (args.output_dir / "postmortem.md").write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
