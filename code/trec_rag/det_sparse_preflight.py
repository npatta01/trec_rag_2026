"""Offline-only preflight and immutable query-plan materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from trec_rag.det_sparse_config import (
    ARM_NAMES,
    LOCKED_PLANNER_TOPIC_IDS,
    DetSparseConfig,
    load_det_sparse_config,
)
from trec_rag.det_sparse_arms import projected_unique_request_ceiling
from trec_rag.det_sparse_freeze import create_freeze_manifest, verify_freeze_manifest
from trec_rag.det_sparse_provenance import current_source_provenance
from trec_rag.deterministic_sparse import (
    DeterministicSparsePlan,
    build_deterministic_sparse_plan,
)
from trec_rag.pipeline_models import QueryVariant
from trec_rag.query_analyzer import QueryAnalyzer, RemoteLuceneQueryAnalyzer
from trec_rag.topics import Topic, derive_title


PREFLIGHT_SCHEMA_VERSION = "det_sparse_preflight_v1"


@dataclass(frozen=True)
class SelectedTopic:
    topic: Topic
    source_line_number: int
    source_line_sha256: str


@dataclass(frozen=True)
class PreflightResult:
    output_dir: Path
    plans: tuple[DeterministicSparsePlan, ...]
    queries: tuple[QueryVariant, ...]
    valid: bool
    planned_base_requests: int
    derived_max_total_requests: int
    hard_external_request_ceiling: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _create_only(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def load_selected_topics(config: DetSparseConfig) -> tuple[SelectedTopic, ...]:
    """Parse only the four selected TSV records and reject locked IDs."""

    if config.topics_format != "tsv":
        raise ValueError("det_sparse_v1 supports only the frozen TSV topic source")
    requested = set(config.topic_ids)
    if requested.intersection(LOCKED_PLANNER_TOPIC_IDS):
        raise ValueError("selected topics contain a locked planner topic")
    found: dict[str, SelectedTopic] = {}
    with config.topics_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            raw_qid, separator, raw_text = raw_line.rstrip(b"\r\n").partition(b"\t")
            try:
                qid = raw_qid.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"topic line {line_number} has a non-UTF-8 ID") from exc
            if qid not in requested:
                continue
            if not separator:
                raise ValueError(f"selected topic line {line_number} lacks a tab")
            if qid in found:
                raise ValueError(f"selected topic ID is duplicated: {qid}")
            try:
                narrative = raw_text.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"selected topic {qid} is not UTF-8") from exc
            # Preserve the exact field bytes (decoded as UTF-8); only the TSV
            # record delimiter was removed before partitioning.
            if not narrative.strip():
                raise ValueError(f"selected topic {qid} has an empty narrative")
            found[qid] = SelectedTopic(
                topic=Topic(
                    id=qid,
                    title=derive_title(narrative),
                    narrative=narrative,
                ),
                source_line_number=line_number,
                source_line_sha256=_sha256_bytes(raw_line),
            )
    missing = [topic_id for topic_id in config.topic_ids if topic_id not in found]
    if missing:
        raise ValueError("selected topic IDs are missing: " + ", ".join(missing))
    return tuple(found[topic_id] for topic_id in config.topic_ids)


def build_preflight(
    config: DetSparseConfig,
    *,
    query_analyzer: QueryAnalyzer,
    output_dir: Path | None = None,
    source_provenance: Mapping[str, object] | None = None,
) -> PreflightResult:
    """Build exact plans and a create-only pre-retrieval freeze; no qrels/network."""

    destination = (output_dir or config.output_dir / "preflight").resolve()
    if destination.exists():
        raise FileExistsError(f"preflight output already exists: {destination}")
    selected = load_selected_topics(config)

    fingerprint = query_analyzer.fingerprint
    fingerprint_dict = fingerprint.to_dict()
    fingerprint_sha = _sha256_bytes(
        json.dumps(
            fingerprint_dict,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    if fingerprint_sha != config.analyzer.expected_fingerprint_sha256:
        raise ValueError("query analyzer fingerprint differs from the frozen config")

    plans = tuple(
        build_deterministic_sparse_plan(
            topic_id=row.topic.id,
            narrative=row.topic.narrative,
            query_analyzer=query_analyzer,
        )
        for row in selected
    )
    for source_topic, plan in zip(selected, plans):
        if plan.splitter_version != config.splitter_version:
            raise ValueError("planner splitter version differs from frozen config")
        if plan.tokenizer_version != config.tokenizer_version:
            raise ValueError("planner tokenizer version differs from frozen config")
        if plan.analyzer_fingerprint_sha256 != fingerprint_sha:
            raise ValueError("plan analyzer fingerprint differs from preflight")
        if plan.original_query_text != source_topic.topic.narrative:
            raise ValueError("plan original query differs from the exact loaded narrative")
        if len(plan.facets) > config.max_facets:
            raise ValueError("plan exceeds the frozen facet ceiling")
        if any(
            len(facet.unique_analyzed_tokens) < config.min_unique_terms
            for facet in plan.facets
        ):
            raise ValueError("plan contains a facet below the frozen token floor")
    queries = tuple(query for plan in plans for query in plan.query_variants())
    base_registry: list[dict[str, object]] = []
    for plan in plans:
        by_text: dict[str, list[QueryVariant]] = {}
        text_order: list[str] = []
        for query in plan.query_variants():
            if query.query_text not in by_text:
                by_text[query.query_text] = []
                text_order.append(query.query_text)
            by_text[query.query_text].append(query)
        for ordinal, query_text in enumerate(text_order, start=1):
            aliases = by_text[query_text]
            base_registry.append(
                {
                    "topic_id": plan.topic_id,
                    "canonical_query_id": f"{plan.topic_id}:base:{ordinal:02d}",
                    "canonical_variant_name": aliases[0].variant_name,
                    "query_text": query_text,
                    "query_sha256": _sha256_bytes(query_text.encode("utf-8")),
                    "logical_variant_aliases": [query.variant_name for query in aliases],
                    "source_types": [query.source_type for query in aliases],
                }
            )
    planned_base_requests = len(base_registry)
    if planned_base_requests > len(plans) * (1 + config.max_facets):
        raise ValueError("base request projection exceeds the frozen facet ceiling")
    if config.cost.max_external_requests != 36:
        raise ValueError("external call ceiling drifted from 36")
    request_projections = [
        {
            "topic_id": plan.topic_id,
            "base_unique_requests": len(
                {query.query_text for query in plan.query_variants()}
            ),
            "derived_max_unique_requests": projected_unique_request_ceiling(plan),
        }
        for plan in plans
    ]
    if any(
        int(row["derived_max_unique_requests"])
        > config.cost.max_unique_requests_per_topic
        for row in request_projections
    ):
        raise ValueError("derived per-topic request ceiling exceeds nine")
    derived_max_total_requests = sum(
        int(row["derived_max_unique_requests"])
        for row in request_projections
    )
    if derived_max_total_requests > config.cost.max_external_requests:
        raise ValueError("derived total request ceiling exceeds 36")
    fallback_topic_ids = [plan.topic_id for plan in plans if plan.status != "ok"]
    source = dict(source_provenance or {})
    source_commit = source.get("commit")
    source_tree = source.get("tree")
    source_valid = (
        source.get("source_tree_clean") is True
        and isinstance(source_commit, str)
        and len(source_commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in source_commit)
        and isinstance(source_tree, str)
        and len(source_tree) in {40, 64}
        and all(character in "0123456789abcdef" for character in source_tree)
    )
    mechanical_valid = not fallback_topic_ids and source_valid

    destination.mkdir(parents=True)
    artifact_paths: list[Path] = []
    plan_rows: list[dict[str, object]] = []
    for ordinal, (source_topic, plan) in enumerate(zip(selected, plans), start=1):
        plan_path = destination / "plans" / f"{ordinal:02d}_{plan.topic_id}.json"
        _create_only(plan_path, _canonical_bytes(plan.to_dict(), pretty=True))
        artifact_paths.append(plan_path)
        plan_rows.append(
            {
                "ordinal": ordinal,
                "topic_id": plan.topic_id,
                "status": plan.status,
                "plan_path": plan_path.relative_to(destination).as_posix(),
                "plan_sha256": _sha256_file(plan_path),
                "source_line_number": source_topic.source_line_number,
                "source_line_sha256": source_topic.source_line_sha256,
            }
        )

    query_path = destination / "queries.jsonl"
    _create_only(
        query_path,
        b"".join(_canonical_bytes(query.__dict__) for query in queries),
    )
    artifact_paths.append(query_path)
    registry_path = destination / "base_query_registry.json"
    _create_only(registry_path, _canonical_bytes(base_registry, pretty=True))
    artifact_paths.append(registry_path)
    run_metadata = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "config_path": str(config.config_path),
        "config_sha256": _sha256_file(config.config_path),
        "topic_source_path": str(config.topics_path),
        "topic_ids": list(config.topic_ids),
        "locked_planner_topic_ids_requested": False,
        "plans": plan_rows,
        "analyzer_fingerprint": fingerprint_dict,
        "analyzer_fingerprint_sha256": fingerprint_sha,
        "arms": list(ARM_NAMES),
        "planned_base_unique_requests": planned_base_requests,
        "base_query_registry_path": registry_path.relative_to(destination).as_posix(),
        "base_query_registry_sha256": _sha256_file(registry_path),
        "request_projections": request_projections,
        "derived_max_total_unique_requests": derived_max_total_requests,
        "hard_external_request_ceiling": config.cost.max_external_requests,
        "max_unique_requests_per_topic": config.cost.max_unique_requests_per_topic,
        "mechanical_valid": mechanical_valid,
        "fallback_topic_ids": fallback_topic_ids,
        "external_calls": 0,
        "model_calls": 0,
        "reranker_calls": 0,
        "qrels_opened": False,
        "source": source,
    }
    run_path = destination / "_preflight.json"
    _create_only(run_path, _canonical_bytes(run_metadata, pretty=True))
    artifact_paths.append(run_path)
    create_freeze_manifest(
        destination / "pre_retrieval_freeze.json",
        artifact_root=destination,
        artifacts=artifact_paths,
        qrels_path=config.evaluation.qrels,
        metadata={
            "experiment_id": config.experiment_id,
            "topic_ids": list(config.topic_ids),
            "mechanical_valid": mechanical_valid,
            "fallback_topic_ids": fallback_topic_ids,
            "request_projections": request_projections,
            "derived_max_total_unique_requests": derived_max_total_requests,
            "hard_external_request_ceiling": config.cost.max_external_requests,
            "external_calls": 0,
            "model_calls": 0,
            "reranker_calls": 0,
        },
    )
    return PreflightResult(
        output_dir=destination,
        plans=plans,
        queries=queries,
        valid=mechanical_valid,
        planned_base_requests=planned_base_requests,
        derived_max_total_requests=derived_max_total_requests,
        hard_external_request_ceiling=config.cost.max_external_requests,
    )


def validate_preflight(config: DetSparseConfig, output_dir: Path) -> dict[str, object]:
    manifest = verify_freeze_manifest(
        output_dir / "pre_retrieval_freeze.json",
        artifact_root=output_dir,
        expected_qrels_path=config.evaluation.qrels,
    )
    metadata = json.loads((output_dir / "_preflight.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("preflight metadata schema mismatch")
    if metadata.get("experiment_id") != config.experiment_id:
        raise ValueError("preflight experiment identity mismatch")
    if metadata.get("config_path") != str(config.config_path) or metadata.get(
        "config_sha256"
    ) != _sha256_file(config.config_path):
        raise ValueError("preflight config identity/hash mismatch")
    if metadata.get("topic_ids") != list(config.topic_ids):
        raise ValueError("preflight topic identity/order mismatch")
    if metadata.get("mechanical_valid") is not True:
        raise ValueError("preflight is hash-valid but mechanically invalid")
    if metadata.get("fallback_topic_ids") != []:
        raise ValueError("preflight contains fallback topics")
    plans = metadata.get("plans")
    if not isinstance(plans, list) or len(plans) != len(config.topic_ids):
        raise ValueError("preflight plan count mismatch")
    if [row.get("topic_id") for row in plans if isinstance(row, dict)] != list(
        config.topic_ids
    ) or any(not isinstance(row, dict) or row.get("status") != "ok" for row in plans):
        raise ValueError("preflight plans are not all valid and ordered")
    for row in plans:
        plan_path = output_dir / str(row["plan_path"])
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("status") != "ok" or plan.get("topic_id") != row["topic_id"]:
            raise ValueError("frozen plan content is not mechanically valid")
    source = metadata.get("source")
    if (
        not isinstance(source, dict)
        or source.get("source_tree_clean") is not True
        or not isinstance(source.get("commit"), str)
        or len(source["commit"]) not in {40, 64}
        or not all(character in "0123456789abcdef" for character in source["commit"])
        or not isinstance(source.get("tree"), str)
        or len(source["tree"]) not in {40, 64}
        or not all(character in "0123456789abcdef" for character in source["tree"])
    ):
        raise ValueError("preflight source provenance is not clean")
    for field in ("external_calls", "model_calls", "reranker_calls"):
        if metadata.get(field) != 0:
            raise ValueError(f"preflight {field} must be zero")
    if metadata.get("qrels_opened") is not False:
        raise ValueError("preflight violated the qrels firewall")
    projections = metadata.get("request_projections")
    if not isinstance(projections, list) or len(projections) != len(config.topic_ids):
        raise ValueError("preflight request projections are missing")
    if [
        row.get("topic_id") if isinstance(row, dict) else None
        for row in projections
    ] != list(config.topic_ids):
        raise ValueError("preflight request projection topic order is invalid")
    per_topic = [row.get("derived_max_unique_requests") for row in projections]
    if any(type(value) is not int or not 1 <= value <= 9 for value in per_topic):
        raise ValueError("preflight per-topic request ceiling must be an integer in 1..9")
    if metadata.get("derived_max_total_unique_requests") != sum(per_topic):
        raise ValueError("preflight derived request total is inconsistent")
    if sum(per_topic) > 36 or metadata.get("hard_external_request_ceiling") != 36:
        raise ValueError("preflight external request ceiling exceeds 36")
    registry_path = output_dir / str(metadata.get("base_query_registry_path"))
    if not registry_path.is_file() or metadata.get(
        "base_query_registry_sha256"
    ) != _sha256_file(registry_path):
        raise ValueError("preflight canonical query registry is missing or changed")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, list) or len(registry) != metadata.get(
        "planned_base_unique_requests"
    ):
        raise ValueError("preflight canonical query registry count mismatch")
    registry_by_topic: dict[str, list[dict[str, object]]] = {
        topic_id: [] for topic_id in config.topic_ids
    }
    for row in registry:
        if not isinstance(row, dict) or row.get("topic_id") not in registry_by_topic:
            raise ValueError("preflight canonical query registry identity mismatch")
        text = row.get("query_text")
        aliases = row.get("logical_variant_aliases")
        if (
            not isinstance(text, str)
            or not text
            or row.get("query_sha256") != _sha256_bytes(text.encode("utf-8"))
            or not isinstance(aliases, list)
            or not aliases
            or len(aliases) != len(set(aliases))
        ):
            raise ValueError("preflight canonical query registry row is invalid")
        registry_by_topic[str(row["topic_id"])].append(row)
    for projection in projections:
        topic_id = str(projection["topic_id"])
        rows = registry_by_topic[topic_id]
        if len(rows) != projection.get("base_unique_requests") or len(
            {str(row["query_text"]) for row in rows}
        ) != len(rows):
            raise ValueError("preflight canonical query registry undercounts aliases")
    logical_queries = [
        json.loads(line)
        for line in (output_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_aliases = {
        (str(row["topic_id"]), str(row["variant_name"])): str(row["query_text"])
        for row in logical_queries
    }
    registered_aliases = {
        (str(row["topic_id"]), str(alias)): str(row["query_text"])
        for row in registry
        for alias in row["logical_variant_aliases"]
    }
    if registered_aliases != expected_aliases:
        raise ValueError("preflight canonical registry does not cover logical queries exactly")
    freeze_metadata = manifest.get("metadata")
    if not isinstance(freeze_metadata, dict):
        raise ValueError("preflight freeze lacks metadata")
    for field in (
        "topic_ids",
        "mechanical_valid",
        "fallback_topic_ids",
        "request_projections",
        "derived_max_total_unique_requests",
        "hard_external_request_ceiling",
        "external_calls",
        "model_calls",
        "reranker_calls",
    ):
        if freeze_metadata.get(field) != metadata.get(field):
            raise ValueError(f"preflight freeze metadata mismatch: {field}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline preflight for the deterministic sparse pilot",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_det_sparse_config(args.config)
    output_dir = (args.output_dir or config.output_dir / "preflight").resolve()
    if args.validate_only:
        validate_preflight(config, output_dir)
        print(f"Validated frozen preflight under {output_dir}")
        return 0
    analyzer = RemoteLuceneQueryAnalyzer(config.analyzer.url)
    result = build_preflight(
        config,
        query_analyzer=analyzer,
        output_dir=output_dir,
        source_provenance=current_source_provenance(config.root_dir),
    )
    print(f"Wrote frozen preflight under {result.output_dir}")
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
