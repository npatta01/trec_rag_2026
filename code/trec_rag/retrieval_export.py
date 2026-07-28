"""Fail-closed export of sealed facet-pilot artifacts to TREC retrieval runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import yaml

from trec_rag.facet_pilot_config import FacetPilotConfig
from trec_rag.facet_retrieval import (
    LONG_DOCUMENT_WEIGHT,
    RELATIVE_SPAN_DELTA,
    SPAN_SUPPORT_CAP,
    SPAN_SUPPORT_WEIGHT,
    STRONGEST_PASSAGE_WEIGHT,
    TOP_WINDOW_WEIGHTS,
)
from trec_rag.topics import Topic, load_narrative_topics


_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXPORT_SCHEMA = "retrieval_export_manifest_v2"
_RETRIEVAL_ARTIFACTS = frozenset({"decomposition.json", "retrieval/audit.json"})
_RETRIEVAL_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "topic_id",
        "narrative_sha256",
        "decomposition_source_sha256",
        "code_commit",
        "retriever",
        "artifacts",
    }
)
_SCORING_ARTIFACTS = frozenset(
    {
        "scoring/lane_scores.jsonl",
        "scoring/selected_documents.jsonl",
        "scoring/selection.json",
        "scoring/selected_subnarrative_scores.jsonl",
    }
)
_CANONICAL_NUGGETS = "canonical/canonical-nuggets.jsonl"
_CANONICAL_ARTIFACTS = frozenset(
    {
        "canonical/handoff/candidate-requests.jsonl",
        "canonical/handoff/selection-contexts.jsonl",
        "canonical/handoff/handoff-manifest.json",
        "canonical/candidates.jsonl",
        "canonical/candidate-manifest.json",
        "canonical/subnarrative-selections.jsonl",
        "canonical/selection-manifest.json",
        _CANONICAL_NUGGETS,
        "canonical/canonical-nugget-manifest.json",
    }
)
_SCORING_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "topic_id",
        "narrative_sha256",
        "decomposition_source_sha256",
        "code_commit",
        "selection_schema_version",
        "retriever",
        "retrieval_manifest_sha256",
        "scorer",
        "rerank_depth",
        "selection_k",
        "selection_policy",
        "score_policy",
        "selected_set_sha256",
        "artifacts",
    }
)
_CANONICAL_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "topic_id",
        "official_topics_sha256",
        "narrative_sha256",
        "decomposition_source_sha256",
        "scoring_manifest_sha256",
        "handoff_manifest_sha256",
        "code_commit",
        "input_roles",
        "selection_policy",
        "selected_budget",
        "canonical_claim_cap",
        "canonical_supporting_document_cap",
        "artifacts",
    }
)
_SELECTED_DOCUMENT_FIELDS = frozenset(
    {
        "topic_id",
        "docid",
        "selection_rank",
        "selected_from_lane",
        "selected_from_lane_rank",
        "text_sha256",
        "text",
    }
)
_LANE_SCORE_FIELDS = frozenset(
    {
        "topic_id",
        "lane_name",
        "bm25_query_sha256",
        "semantic_query_sha256",
        "docid",
        "bm25_rank",
        "bm25_score",
        "aggregate_rank",
        "aggregate_score",
        "long_document_raw_logit",
        "weighted_passage_raw_logit",
        "within_document_span_support",
        "winning_passages",
        "score_representation",
        "text_sha256",
    }
)
_MEMBERSHIP_FIELDS = frozenset({"docid", "lanes"})
_MEMBERSHIP_LANE_FIELDS = frozenset(
    {"lane_name", "aggregate_rank", "aggregate_score", "bm25_rank", "bm25_score"}
)
_DOWNSTREAM_SCORE_FIELDS = frozenset(
    {
        "topic_id",
        "lane_name",
        "semantic_query_sha256",
        "docid",
        "bm25_rank",
        "bm25_score",
        "aggregate_rank",
        "aggregate_score",
        "long_document_raw_logit",
        "weighted_passage_raw_logit",
        "within_document_span_support",
        "winning_passages",
        "score_representation",
        "text_sha256",
        "selection_rank",
        "subnarrative_id",
        "bm25_queries",
        "bm25_query_sha256s",
        "downstream_only",
    }
)
_CANONICAL_NUGGET_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "result_schema_version",
        "canonical_response_schema_version",
        "selection_schema_version",
        "selection_manifest_schema_version",
        "selection_file",
        "selection_manifest_file",
        "canonical_nugget_file",
        "selections_sha256",
        "selection_manifest_sha256",
        "canonical_nuggets_sha256",
        "output_sha256",
        "selected_budget",
        "selection_count",
        "result_count",
        "state_counts",
        "max_canonical_claims",
        "max_supporting_documents_per_claim",
        "request_sha256s",
        "model",
        "prompt_version",
        "hosted_llm_calls",
        "validated_cache_hits",
        "raw_cache_writes",
        "validated_cache_writes",
    }
)
_CANONICAL_RESULT_STATES = frozenset({"complete", "empty", "fallback_extractive"})
_PASSAGE_FIELDS = frozenset(
    {"chunk_index", "start_char", "end_char", "raw_logit", "weighted_rank"}
)


@dataclass(frozen=True)
class RetrievalExportReceipt:
    official_run: Path
    candidate_pool_run: Path
    with_text_archive: Path
    provenance: Path
    resolved_config: Path
    manifest: Path


@dataclass(frozen=True)
class _SelectedDocument:
    docid: str
    rank: int
    text: str
    text_sha256: str
    selected_from_lane: str
    selected_from_lane_rank: int


@dataclass(frozen=True)
class _TopicProjection:
    topic: Topic
    selected: tuple[_SelectedDocument, ...]
    supported_docids: frozenset[str]
    original_only_fallback: bool
    memberships: Mapping[str, tuple[dict[str, Any], ...]]
    scores: Mapping[str, tuple[dict[str, Any], ...]]
    nuggets: Mapping[str, tuple[dict[str, Any], ...]]
    scoring_manifest_sha256: str
    canonical_manifest_sha256: str
    source_code_commit: str


def export_retrieval_run(
    config: FacetPilotConfig,
    topics: Sequence[Topic],
    *,
    code_commit: str,
) -> RetrievalExportReceipt:
    """Validate sealed topic checkpoints and publish deterministic run projections."""
    if not isinstance(code_commit, str) or not _COMMIT.fullmatch(code_commit):
        raise ValueError("code commit must be a full lowercase SHA-1")
    selected_topics = tuple(topics)
    _validate_topics(selected_topics)
    _validate_existing_export(
        config.output_dir,
        run_id=config.run_id,
        topic_ids=tuple(topic.id for topic in selected_topics),
    )

    projections = tuple(_load_topic_projection(config, topic) for topic in selected_topics)
    official_rows: list[tuple[str, str, int, int, str]] = []
    candidate_rows: list[tuple[str, str, int, int, str]] = []
    for projection in projections:
        final = _final_documents(projection)
        if not final:
            raise ValueError(f"selected topic {projection.topic.id!r} has no supported document")
        official_rows.extend(
            (
                projection.topic.id,
                row.docid,
                rank,
                len(final) - rank + 1,
                config.run_id,
            )
            for rank, row in enumerate(final, start=1)
        )
        candidate_rows.extend(
            (
                projection.topic.id,
                row.docid,
                rank,
                len(projection.selected) - rank + 1,
                f"{config.run_id}-candidate-pool",
            )
            for rank, row in enumerate(projection.selected, start=1)
        )

    official_bytes = _trec_bytes(official_rows)
    candidate_bytes = _trec_bytes(candidate_rows)
    _validate_trec_bytes(official_bytes, official_rows)
    _validate_trec_bytes(candidate_bytes, candidate_rows)

    with_text_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    topic_depths: dict[str, dict[str, int]] = {}
    source_seals: dict[str, dict[str, str]] = {}
    for projection in projections:
        final = _final_documents(projection)
        stage = (
            "original_only_fallback"
            if projection.original_only_fallback
            else "canonical_supported"
        )
        with_text_rows.append(
            {
                "query": {
                    "qid": projection.topic.id,
                    "text": projection.topic.narrative,
                },
                "candidates": [
                    {
                        "docid": document.docid,
                        "rank": rank,
                        "score": len(final) - rank + 1,
                        "doc": document.text,
                        "index": config.retrieval.index,
                        "stage": stage,
                    }
                    for rank, document in enumerate(final, start=1)
                ],
            }
        )
        for rank, document in enumerate(final, start=1):
            provenance_row = {
                "topic_id": projection.topic.id,
                "docid": document.docid,
                "rank": rank,
                "score": len(final) - rank + 1,
                "selection_rank": document.rank,
                "selected_from_lane": document.selected_from_lane,
                "selected_from_lane_rank": document.selected_from_lane_rank,
                "memberships": list(projection.memberships[document.docid]),
                "subnarrative_scores": list(projection.scores[document.docid]),
                "nuggets": list(projection.nuggets[document.docid]),
                "source_seals": {
                    "scoring_manifest_sha256": projection.scoring_manifest_sha256,
                    "canonical_manifest_sha256": projection.canonical_manifest_sha256,
                },
            }
            if projection.original_only_fallback:
                provenance_row["stage"] = stage
            provenance_rows.append(provenance_row)
        topic_depths[projection.topic.id] = {
            "official": len(final),
            "candidate_pool": len(projection.selected),
        }
        source_seals[projection.topic.id] = {
            "scoring_manifest_sha256": projection.scoring_manifest_sha256,
            "canonical_manifest_sha256": projection.canonical_manifest_sha256,
            "source_code_commit": projection.source_code_commit,
        }

    with_text_jsonl = _jsonl_bytes(with_text_rows)
    provenance_bytes = _jsonl_bytes(provenance_rows)
    archive_bytes = _deterministic_zip(with_text_jsonl)
    resolved_bytes = yaml.safe_dump(
        config.resolved_payload(selected_topics),
        allow_unicode=True,
        sort_keys=True,
    ).encode("utf-8")

    output_dir = config.output_dir
    official_run = output_dir / "r_output_trec_rag_2026.tsv"
    candidate_pool_run = output_dir / "retrieval_candidate_pool.trec"
    with_text_archive = output_dir / "retrieval_with_text.jsonl.zip"
    provenance = output_dir / "retrieval_provenance.jsonl"
    resolved_config = output_dir / "resolved_config.yaml"
    manifest = output_dir / "retrieval_export_manifest.json"
    artifacts = {
        official_run.name: official_bytes,
        candidate_pool_run.name: candidate_bytes,
        with_text_archive.name: archive_bytes,
        provenance.name: provenance_bytes,
        resolved_config.name: resolved_bytes,
    }
    manifest_body = _canonical_json_bytes(
        {
            "schema_version": _EXPORT_SCHEMA,
            "export_code_commit": code_commit,
            "source_code_commits": sorted(
                {projection.source_code_commit for projection in projections}
            ),
            "run_id": config.run_id,
            "selected_topic_ids": [topic.id for topic in selected_topics],
            "score_semantics": "ordinal_selection_order",
            "topic_depths": topic_depths,
            "source_seals": source_seals,
            "resolved_config_sha256": sha256(resolved_bytes).hexdigest(),
            "official_row_count": len(official_rows),
            "candidate_pool_row_count": len(candidate_rows),
            "artifacts": {
                name: {"bytes": len(body), "sha256": sha256(body).hexdigest()}
                for name, body in artifacts.items()
            },
        },
        pretty=True,
    )
    if manifest.exists():
        manifest.unlink()
    for path, body in (
        (official_run, official_bytes),
        (candidate_pool_run, candidate_bytes),
        (with_text_archive, archive_bytes),
        (provenance, provenance_bytes),
        (resolved_config, resolved_bytes),
    ):
        _atomic_write(path, body)
    _atomic_write(manifest, manifest_body)
    return RetrievalExportReceipt(
        official_run=official_run,
        candidate_pool_run=candidate_pool_run,
        with_text_archive=with_text_archive,
        provenance=provenance,
        resolved_config=resolved_config,
        manifest=manifest,
    )


def _final_documents(projection: _TopicProjection) -> list[_SelectedDocument]:
    if projection.original_only_fallback:
        return list(projection.selected)
    return [
        row for row in projection.selected if row.docid in projection.supported_docids
    ]


def read_retrieval_export_receipt(
    config: FacetPilotConfig,
    topics: Sequence[Topic],
) -> RetrievalExportReceipt:
    """Re-read and validate a complete retrieval export before returning it."""
    selected_topics = tuple(topics)
    _validate_topics(selected_topics)
    manifest = config.output_dir / "retrieval_export_manifest.json"
    if not manifest.is_file():
        raise ValueError("retrieval export manifest is missing")
    _validate_existing_export(
        config.output_dir,
        run_id=config.run_id,
        topic_ids=tuple(topic.id for topic in selected_topics),
    )
    return RetrievalExportReceipt(
        official_run=config.output_dir / "r_output_trec_rag_2026.tsv",
        candidate_pool_run=config.output_dir / "retrieval_candidate_pool.trec",
        with_text_archive=config.output_dir / "retrieval_with_text.jsonl.zip",
        provenance=config.output_dir / "retrieval_provenance.jsonl",
        resolved_config=config.output_dir / "resolved_config.yaml",
        manifest=manifest,
    )


def validate_retrieval_topic_checkpoints(
    config: FacetPilotConfig,
    topics: Sequence[Topic],
) -> None:
    """Deeply validate selected sealed topic checkpoints without exporting them."""
    selected_topics = tuple(topics)
    if not selected_topics:
        return
    _validate_topics(selected_topics)
    for topic in selected_topics:
        try:
            _load_topic_projection(config, topic)
        except OSError as exc:
            raise ValueError("sealed topic checkpoint is incomplete") from exc


def _validate_topics(topics: tuple[Topic, ...]) -> None:
    if not topics:
        raise ValueError("at least one selected topic is required")
    seen: set[str] = set()
    for topic in topics:
        if not isinstance(topic, Topic) or not topic.id or not topic.narrative.strip():
            raise ValueError("selected topics must have non-empty identities and narratives")
        if topic.id in seen:
            raise ValueError(f"duplicate selected topic ID: {topic.id}")
        if (
            topic.id in {".", ".."}
            or Path(topic.id).name != topic.id
            or any(character.isspace() for character in topic.id)
        ):
            raise ValueError("topic ID must be a safe path component")
        seen.add(topic.id)


def _validate_existing_export(
    output_dir: Path,
    *,
    run_id: str,
    topic_ids: tuple[str, ...],
) -> None:
    manifest_path = output_dir / "retrieval_export_manifest.json"
    if not manifest_path.exists():
        return
    manifest = _manifest(manifest_path.read_bytes(), "existing export manifest")
    if manifest.get("schema_version") != _EXPORT_SCHEMA:
        raise ValueError("unsupported existing export manifest schema")
    expected_fields = {
        "schema_version",
        "export_code_commit",
        "source_code_commits",
        "run_id",
        "selected_topic_ids",
        "score_semantics",
        "topic_depths",
        "source_seals",
        "resolved_config_sha256",
        "official_row_count",
        "candidate_pool_row_count",
        "artifacts",
    }
    if (
        set(manifest) != expected_fields
        or manifest.get("run_id") != run_id
        or manifest.get("selected_topic_ids") != list(topic_ids)
        or manifest.get("score_semantics") != "ordinal_selection_order"
    ):
        raise ValueError("existing export manifest identity changed")
    export_commit = manifest.get("export_code_commit")
    source_commits = manifest.get("source_code_commits")
    if (
        not isinstance(export_commit, str)
        or not _COMMIT.fullmatch(export_commit)
        or not isinstance(source_commits, list)
        or not source_commits
        or source_commits != sorted(set(source_commits))
        or any(not isinstance(item, str) or not _COMMIT.fullmatch(item) for item in source_commits)
    ):
        raise ValueError("existing export revision identity changed")
    artifacts = manifest.get("artifacts")
    expected_names = {
        "r_output_trec_rag_2026.tsv",
        "retrieval_candidate_pool.trec",
        "retrieval_with_text.jsonl.zip",
        "retrieval_provenance.jsonl",
        "resolved_config.yaml",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise ValueError("existing export artifact receipts changed")
    for name in sorted(expected_names):
        receipt = artifacts[name]
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"bytes", "sha256"}
            or type(receipt.get("bytes")) is not int
            or receipt["bytes"] < 0
            or not isinstance(receipt.get("sha256"), str)
            or not _SHA256.fullmatch(receipt["sha256"])
        ):
            raise ValueError("existing export artifact receipt changed")
        body = (output_dir / name).read_bytes()
        if (
            len(body) != receipt["bytes"]
            or sha256(body).hexdigest() != receipt["sha256"]
        ):
            raise ValueError("existing export artifact hash changed")


def _load_topic_projection(
    config: FacetPilotConfig,
    topic: Topic,
) -> _TopicProjection:
    topic_root = config.output_dir / topic.id
    scoring_path = topic_root / "scoring" / "complete.json"
    canonical_path = topic_root / "canonical" / "complete.json"
    scoring_bytes = scoring_path.read_bytes()
    scoring_manifest = _manifest(scoring_bytes, "scoring checkpoint manifest")
    scoring_commit = _validate_scoring_manifest(scoring_manifest, config, topic)
    scoring_receipts = _validate_receipts(topic_root, scoring_manifest)
    if scoring_receipts != _SCORING_ARTIFACTS:
        raise ValueError("scoring checkpoint artifact set changed")
    (
        original_only_fallback,
        decomposition,
        retrieval_audit,
    ) = _validate_retrieval_source_chain(
        topic_root,
        scoring_manifest,
        config,
        topic,
        scoring_commit,
    )

    canonical_bytes = canonical_path.read_bytes()
    canonical_manifest = _manifest(
        canonical_bytes, "canonical checkpoint manifest"
    )
    canonical_commit = _validate_canonical_manifest(
        canonical_manifest,
        config,
        topic,
        scoring_manifest,
        scoring_bytes,
    )
    if canonical_commit != scoring_commit:
        raise ValueError("scoring and canonical checkpoint code identities differ")
    canonical_receipts = _validate_receipts(topic_root, canonical_manifest)
    if canonical_receipts != _CANONICAL_ARTIFACTS:
        raise ValueError("canonical checkpoint artifact set changed")
    handoff_bytes = (
        topic_root / "canonical" / "handoff" / "handoff-manifest.json"
    ).read_bytes()
    if canonical_manifest.get("handoff_manifest_sha256") != sha256(
        handoff_bytes
    ).hexdigest():
        raise ValueError("canonical checkpoint handoff seal changed")

    selected = _load_selected_documents(
        topic_root / "scoring" / "selected_documents.jsonl", topic.id
    )
    selected_set_sha256 = sha256(
        json.dumps(
            [row.docid for row in selected], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if scoring_manifest.get("selected_set_sha256") != selected_set_sha256:
        raise ValueError("scoring checkpoint selected document set changed")
    lane_scores = _load_lane_scores(
        topic_root / "scoring" / "lane_scores.jsonl", topic.id
    )
    from trec_rag.official_run import validate_scoring_selection

    memberships = validate_scoring_selection(
        (topic_root / "scoring" / "selection.json").read_bytes(),
        topic_id=topic.id,
        selected_documents=tuple(row.__dict__ for row in selected),
        lane_score_rows=tuple(lane_scores.values()),
        audit_lanes=retrieval_audit,
        rerank_depth=config.reranking.rerank_depth_per_query,
        selected_set_sha256=selected_set_sha256,
    )
    cross_scores = _load_cross_scores(
        topic_root / "scoring" / "selected_subnarrative_scores.jsonl",
        topic.id,
        selected,
        expected_subnarratives=decomposition.result.subnarratives,
    )
    _validate_memberships(memberships, selected, lane_scores)
    if original_only_fallback:
        _validate_original_only_scoring(
            selected,
            lane_scores,
            cross_scores,
            expected_query_sha256=sha256(topic.narrative.encode("utf-8")).hexdigest(),
        )
    subnarrative_ids = _validate_cross_score_matrix(
        cross_scores,
        selected,
        expected_subnarrative_ids=frozenset(
            row.subnarrative_id for row in decomposition.result.subnarratives
        ),
        allow_empty=original_only_fallback,
    )
    nugget_manifest = _validate_canonical_nugget_manifest(
        topic_root,
        canonical_manifest,
        config,
        subnarrative_ids,
    )
    if original_only_fallback and any(
        nugget_manifest[field] != 0
        for field in (
            "hosted_llm_calls",
            "validated_cache_hits",
            "raw_cache_writes",
            "validated_cache_writes",
        )
    ):
        raise ValueError("original-only fallback canonical work must be empty")
    allowed_evidence, canonical_requests = _load_allowed_canonical_evidence(
        topic_root,
        topic,
        selected,
        decomposition.result.subnarratives,
        selected_budget=config.nuggets.evidence_budget_per_subnarrative,
        max_canonical_claims=config.nuggets.maximum_claims_per_subnarrative,
        max_supporting_documents_per_claim=(
            config.nuggets.maximum_supporting_documents_per_claim
        ),
    )
    supported_docids, nuggets = _load_supported_docids(
        topic_root / _CANONICAL_NUGGETS,
        topic.id,
        selected,
        expected_subnarrative_ids=subnarrative_ids,
        nugget_manifest=nugget_manifest,
        allowed_evidence=allowed_evidence,
        requests=canonical_requests,
    )
    membership_map = {
        row["docid"]: tuple(
            {
                "lane_name": lane["lane_name"],
                "aggregate_rank": lane["aggregate_rank"],
                "aggregate_score": lane["aggregate_score"],
                "bm25_rank": lane["bm25_rank"],
                "bm25_score": lane["bm25_score"],
            }
            for lane in row["lanes"]
        )
        for row in memberships
    }
    score_map = {
        document.docid: tuple(
            score for score in cross_scores if score["docid"] == document.docid
        )
        for document in selected
    }
    return _TopicProjection(
        topic,
        selected,
        frozenset(supported_docids),
        original_only_fallback,
        membership_map,
        score_map,
        nuggets,
        sha256(scoring_bytes).hexdigest(),
        sha256(canonical_bytes).hexdigest(),
        scoring_commit,
    )


def _manifest(body: bytes, label: str) -> dict[str, Any]:
    value = _strict_json(body, label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_source_manifest_identity(
    manifest: Mapping[str, object],
    *,
    expected_fields: frozenset[str],
    topic: Topic,
    phase: str,
) -> str:
    if set(manifest) != expected_fields or manifest.get("schema_version") != "facet_pilot_v2":
        raise ValueError(f"{phase} checkpoint schema changed")
    if manifest.get("topic_id") != topic.id or manifest.get("phase") != phase:
        raise ValueError(f"{phase} checkpoint topic or phase identity changed")
    commit = manifest.get("code_commit")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError(f"{phase} checkpoint code identity is invalid")
    return commit


def _validate_scoring_manifest(
    manifest: Mapping[str, object],
    config: FacetPilotConfig,
    topic: Topic,
) -> str:
    commit = _validate_source_manifest_identity(
        manifest,
        expected_fields=_SCORING_MANIFEST_FIELDS,
        topic=topic,
        phase="score",
    )
    narrative_sha256 = sha256(topic.narrative.encode("utf-8")).hexdigest()
    retriever = manifest.get("retriever")
    scorer = manifest.get("scorer")
    if manifest.get("narrative_sha256") != narrative_sha256:
        raise ValueError("scoring checkpoint narrative identity is stale")
    if (
        not isinstance(manifest.get("decomposition_source_sha256"), str)
        or not _SHA256.fullmatch(manifest["decomposition_source_sha256"])
        or not isinstance(retriever, Mapping)
        or retriever.get("index") != config.retrieval.index
        or retriever.get("hits") != config.retrieval.candidate_depth_per_query
        or not isinstance(scorer, Mapping)
        or scorer.get("model") != config.reranking.model
        or manifest.get("rerank_depth") != config.reranking.rerank_depth_per_query
        or manifest.get("selection_k") != config.reranking.candidate_pool_depth
        or manifest.get("selection_schema_version") != "facet_pilot_selection_v2"
        or manifest.get("selection_policy")
        != _scoring_selection_policy(config.reranking.selection_policy)
    ):
        raise ValueError("scoring checkpoint configuration is incompatible with export")
    for digest_key in ("retrieval_manifest_sha256", "selected_set_sha256"):
        value = manifest.get(digest_key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError("scoring checkpoint digest identity is invalid")
    if manifest.get("score_policy") != _producer_score_policy():
        raise ValueError("scoring checkpoint score policy is invalid")
    return commit


def _producer_score_policy() -> dict[str, object]:
    return {
        "long_document_weight": LONG_DOCUMENT_WEIGHT,
        "strongest_passage_weight": STRONGEST_PASSAGE_WEIGHT,
        "span_support_weight": SPAN_SUPPORT_WEIGHT,
        "span_support_cap": SPAN_SUPPORT_CAP,
        "relative_span_delta": RELATIVE_SPAN_DELTA,
        "top_window_weights": list(TOP_WINDOW_WEIGHTS),
    }


def _validate_retrieval_source_chain(
    topic_root: Path,
    scoring_manifest: Mapping[str, object],
    config: FacetPilotConfig,
    topic: Topic,
    source_commit: str,
) -> tuple[bool, Any, tuple[Any, ...]]:
    retrieval_path = topic_root / "retrieval" / "complete.json"
    retrieval_bytes = retrieval_path.read_bytes()
    if scoring_manifest.get("retrieval_manifest_sha256") != sha256(
        retrieval_bytes
    ).hexdigest():
        raise ValueError("scoring checkpoint retrieval manifest seal changed")
    retrieval = _manifest(retrieval_bytes, "retrieval checkpoint manifest")
    retrieval_commit = _validate_source_manifest_identity(
        retrieval,
        expected_fields=_RETRIEVAL_MANIFEST_FIELDS,
        topic=topic,
        phase="retrieve",
    )
    retriever = retrieval.get("retriever")
    if (
        retrieval_commit != source_commit
        or retrieval.get("narrative_sha256")
        != sha256(topic.narrative.encode("utf-8")).hexdigest()
        or retrieval.get("decomposition_source_sha256")
        != scoring_manifest.get("decomposition_source_sha256")
        or retriever != scoring_manifest.get("retriever")
        or not isinstance(retriever, Mapping)
        or retriever.get("index") != config.retrieval.index
        or retriever.get("hits") != config.retrieval.candidate_depth_per_query
    ):
        raise ValueError("retrieval checkpoint identity is incompatible with export")
    if _validate_receipts(topic_root, retrieval) != _RETRIEVAL_ARTIFACTS:
        raise ValueError("retrieval checkpoint artifact set changed")

    from trec_rag.official_run import (
        decode_retrieval_audit,
        decode_retrieval_decomposition,
        load_validated_decomposition,
    )

    source_decomposition = load_validated_decomposition(
        topic, topic_root / "decomposition" / "result.json"
    )
    decomposition = decode_retrieval_decomposition(
        topic,
        (topic_root / "decomposition.json").read_bytes(),
        expected_source_sha256=scoring_manifest["decomposition_source_sha256"],
    )
    if (
        source_decomposition.source_sha256 != decomposition.source_sha256
        or source_decomposition.result.used_fallback
        != decomposition.result.used_fallback
        or source_decomposition.result.queries != decomposition.result.queries
        or source_decomposition.result.plan != decomposition.result.plan
        or source_decomposition.result.subnarratives
        != decomposition.result.subnarratives
    ):
        raise ValueError(
            "saved decomposition source differs from receipted canonical rendering"
        )
    audit = decode_retrieval_audit(
        topic,
        decomposition,
        (topic_root / "retrieval" / "audit.json").read_bytes(),
        requested_depth=config.retrieval.candidate_depth_per_query,
    )
    return decomposition.result.used_fallback, decomposition, audit


def _validate_canonical_manifest(
    manifest: Mapping[str, object],
    config: FacetPilotConfig,
    topic: Topic,
    scoring_manifest: Mapping[str, object],
    scoring_bytes: bytes,
) -> str:
    commit = _validate_source_manifest_identity(
        manifest,
        expected_fields=_CANONICAL_MANIFEST_FIELDS,
        topic=topic,
        phase="canonical",
    )
    expected_policy = {
        "budgets": [config.nuggets.evidence_budget_per_subnarrative],
        "precluster_limit": 10 * config.nuggets.evidence_budget_per_subnarrative,
        "semantic_threshold": 0.92,
        "mmr_lambda": 0.7,
    }
    if (
        manifest.get("narrative_sha256")
        != sha256(topic.narrative.encode("utf-8")).hexdigest()
    ):
        raise ValueError("canonical checkpoint narrative identity is stale")
    if (
        manifest.get("decomposition_source_sha256")
        != scoring_manifest.get("decomposition_source_sha256")
        or manifest.get("official_topics_sha256") != _official_topics_sha256(config)
        or manifest.get("scoring_manifest_sha256") != sha256(scoring_bytes).hexdigest()
        or manifest.get("selection_policy") != expected_policy
        or manifest.get("selected_budget")
        != config.nuggets.evidence_budget_per_subnarrative
        or manifest.get("canonical_claim_cap")
        != config.nuggets.maximum_claims_per_subnarrative
        or manifest.get("canonical_supporting_document_cap")
        != config.nuggets.maximum_supporting_documents_per_claim
        or manifest.get("input_roles")
        != [
            "official_topic_narrative",
            "validated_generated_decomposition",
            "sealed_scoring_checkpoint",
        ]
    ):
        raise ValueError("canonical checkpoint configuration is incompatible with export")
    handoff_digest = manifest.get("handoff_manifest_sha256")
    if not isinstance(handoff_digest, str) or not _SHA256.fullmatch(handoff_digest):
        raise ValueError("canonical checkpoint handoff digest is invalid")
    return commit


def _scoring_selection_policy(config_policy: str) -> str:
    if config_policy != "round_robin_subnarrative_coverage":
        raise ValueError("scoring checkpoint selection policy configuration is unsupported")
    return "round_robin_lane_order_no_fusion"


def _official_topics_sha256(config: FacetPilotConfig) -> str:
    topics = load_narrative_topics(config.topics_path)
    body = json.dumps(
        [{"id": topic.id, "narrative": topic.narrative} for topic in topics],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(body).hexdigest()


def _validate_canonical_nugget_manifest(
    topic_root: Path,
    canonical_checkpoint: Mapping[str, object],
    config: FacetPilotConfig,
    expected_subnarrative_ids: frozenset[str],
) -> dict[str, Any]:
    canonical_root = topic_root / "canonical"
    value = _manifest(
        (canonical_root / "canonical-nugget-manifest.json").read_bytes(),
        "canonical nugget manifest",
    )
    fixed_identity = {
        "schema_version": "canonical_nugget_manifest_v2",
        "result_schema_version": "canonical_nugget_result_v1",
        "canonical_response_schema_version": "canonical_nuggets_v1",
        "selection_schema_version": "subnarrative_selection_v1",
        "selection_manifest_schema_version": "subnarrative_selection_manifest_v1",
        "selection_file": "subnarrative-selections.jsonl",
        "selection_manifest_file": "selection-manifest.json",
        "canonical_nugget_file": "canonical-nuggets.jsonl",
        "model": "deepseek/deepseek-v4-flash-20260423",
        "prompt_version": "canonical_nuggetizer_v3",
    }
    if (
        set(value) != _CANONICAL_NUGGET_MANIFEST_FIELDS
        or any(value.get(key) != expected for key, expected in fixed_identity.items())
    ):
        raise ValueError("canonical nugget manifest identity changed")

    nugget_bytes = (canonical_root / "canonical-nuggets.jsonl").read_bytes()
    expected_digests = {
        "selections_sha256": sha256(
            (canonical_root / "subnarrative-selections.jsonl").read_bytes()
        ).hexdigest(),
        "selection_manifest_sha256": sha256(
            (canonical_root / "selection-manifest.json").read_bytes()
        ).hexdigest(),
        "canonical_nuggets_sha256": sha256(nugget_bytes).hexdigest(),
        "output_sha256": sha256(nugget_bytes).hexdigest(),
    }
    if any(value.get(key) != digest for key, digest in expected_digests.items()):
        raise ValueError("canonical nugget manifest artifact seal changed")

    expected_count = len(expected_subnarrative_ids)
    if (
        value.get("selected_budget")
        != config.nuggets.evidence_budget_per_subnarrative
        or value.get("selected_budget") != canonical_checkpoint.get("selected_budget")
        or value.get("max_canonical_claims")
        != config.nuggets.maximum_claims_per_subnarrative
        or value.get("max_canonical_claims")
        != canonical_checkpoint.get("canonical_claim_cap")
        or value.get("max_supporting_documents_per_claim")
        != config.nuggets.maximum_supporting_documents_per_claim
        or value.get("max_supporting_documents_per_claim")
        != canonical_checkpoint.get("canonical_supporting_document_cap")
        or type(value.get("selection_count")) is not int
        or value["selection_count"] != expected_count
        or type(value.get("result_count")) is not int
        or value["result_count"] != expected_count
    ):
        raise ValueError("canonical nugget manifest configuration or counts changed")

    requests = value.get("request_sha256s")
    if (
        not isinstance(requests, list)
        or len(requests) != expected_count
        or any(type(item) is not str or not _SHA256.fullmatch(item) for item in requests)
        or len(set(requests)) != expected_count
    ):
        raise ValueError("canonical nugget manifest request digests changed")
    state_counts = value.get("state_counts")
    if (
        not isinstance(state_counts, dict)
        or not set(state_counts) <= _CANONICAL_RESULT_STATES
        or any(type(count) is not int or count < 0 for count in state_counts.values())
        or sum(state_counts.values()) != expected_count
    ):
        raise ValueError("canonical nugget manifest state counts changed")
    for field in (
        "hosted_llm_calls",
        "validated_cache_hits",
        "raw_cache_writes",
        "validated_cache_writes",
    ):
        if type(value.get(field)) is not int or value[field] < 0:
            raise ValueError("canonical nugget manifest run counters changed")
    return value


def _validate_receipts(topic_root: Path, manifest: Mapping[str, object]) -> set[str]:
    receipts = manifest.get("artifacts")
    if not isinstance(receipts, list):
        raise ValueError("checkpoint artifacts changed")
    seen: set[str] = set()
    root = topic_root.resolve()
    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != {
            "relative_path",
            "bytes",
            "sha256",
        }:
            raise ValueError("checkpoint artifact receipt changed")
        relative = receipt["relative_path"]
        byte_count = receipt["bytes"]
        digest = receipt["sha256"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or Path(relative).is_absolute()
        ):
            raise ValueError("checkpoint artifact path changed")
        path = (topic_root / relative).resolve()
        if path == root or root not in path.parents:
            raise ValueError("checkpoint artifact path escapes topic root")
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError("checkpoint artifact byte count changed")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("checkpoint artifact digest changed")
        body = path.read_bytes()
        if len(body) != byte_count or sha256(body).hexdigest() != digest:
            raise ValueError("checkpoint artifact hash changed")
        seen.add(relative)
    return seen


def _load_selected_documents(path: Path, topic_id: str) -> tuple[_SelectedDocument, ...]:
    records = _jsonl(path, "selected documents")
    rows: list[_SelectedDocument] = []
    seen: set[str] = set()
    for expected_rank, record in enumerate(records, start=1):
        if not isinstance(record, dict) or set(record) != _SELECTED_DOCUMENT_FIELDS:
            raise ValueError("selected document schema changed")
        docid = record.get("docid")
        rank = record.get("selection_rank")
        text = record.get("text")
        text_hash = record.get("text_sha256")
        selected_from_lane = record.get("selected_from_lane")
        selected_from_lane_rank = record.get("selected_from_lane_rank")
        if record.get("topic_id") != topic_id:
            raise ValueError("selected document topic identity changed")
        if (
            not isinstance(docid, str)
            or not docid
            or any(character.isspace() for character in docid)
            or docid in seen
        ):
            raise ValueError("duplicate or invalid selected topic-document pair")
        if type(rank) is not int or rank != expected_rank:
            raise ValueError("selected document ranks must be contiguous")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("selected document text must be non-empty")
        if text_hash != sha256(text.encode("utf-8")).hexdigest():
            raise ValueError("selected document text hash changed")
        if (
            not isinstance(selected_from_lane, str)
            or not selected_from_lane
            or isinstance(selected_from_lane_rank, bool)
            or not isinstance(selected_from_lane_rank, int)
            or selected_from_lane_rank <= 0
        ):
            raise ValueError("selected document lane provenance is invalid")
        seen.add(docid)
        rows.append(
            _SelectedDocument(
                docid,
                rank,
                text,
                text_hash,
                selected_from_lane,
                selected_from_lane_rank,
            )
        )
    if not rows:
        raise ValueError("selected document checkpoint is empty")
    return tuple(rows)


def _load_lane_scores(
    path: Path,
    topic_id: str,
) -> Mapping[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in _jsonl(path, "lane scores"):
        if not isinstance(record, dict) or set(record) != _LANE_SCORE_FIELDS:
            raise ValueError("lane score schema changed")
        docid = record.get("docid")
        lane_name = record.get("lane_name")
        key = (docid, lane_name)
        if (
            record.get("topic_id") != topic_id
            or not isinstance(docid, str)
            or not docid
            or not isinstance(lane_name, str)
            or not lane_name
            or key in result
        ):
            raise ValueError("lane score identity is invalid or duplicated")
        _validate_score_components(record, "lane score")
        result[key] = record
    if not result:
        raise ValueError("lane score artifact is empty")
    return result


def _validate_memberships(
    memberships: tuple[dict[str, Any], ...],
    selected: tuple[_SelectedDocument, ...],
    lane_scores: Mapping[tuple[str, str], dict[str, Any]],
) -> None:
    if [row.get("docid") for row in memberships] != [row.docid for row in selected]:
        raise ValueError("selection memberships do not match selected documents")
    selected_by_id = {row.docid: row for row in selected}
    for membership in memberships:
        if set(membership) != _MEMBERSHIP_FIELDS:
            raise ValueError("selection membership schema changed")
        lanes = membership.get("lanes")
        if not isinstance(lanes, list) or not lanes or any(
            not isinstance(lane, dict)
            or set(lane) != _MEMBERSHIP_LANE_FIELDS
            or not isinstance(lane.get("lane_name"), str)
            or not lane["lane_name"]
            for lane in lanes
        ):
            raise ValueError("selection membership lanes are invalid")
        if len({lane["lane_name"] for lane in lanes}) != len(lanes):
            raise ValueError("selection membership lane identities are duplicated")
        document = selected_by_id[membership["docid"]]
        sealed_lane_names = {
            lane_name
            for docid, lane_name in lane_scores
            if docid == document.docid
        }
        if {lane["lane_name"] for lane in lanes} != sealed_lane_names:
            raise ValueError("selection membership lane set differs from sealed lane scores")
        for lane in lanes:
            source = lane_scores.get((document.docid, lane["lane_name"]))
            if source is None or any(
                lane[field] != source[field]
                for field in (
                    "aggregate_rank",
                    "aggregate_score",
                    "bm25_rank",
                    "bm25_score",
                )
            ):
                raise ValueError("selection membership differs from sealed lane score")
        origin = next(
            (lane for lane in lanes if lane["lane_name"] == document.selected_from_lane),
            None,
        )
        if (
            origin is None
            or origin["aggregate_rank"] != document.selected_from_lane_rank
        ):
            raise ValueError("selected document origin differs from sealed lane score")


def _load_cross_scores(
    path: Path,
    topic_id: str,
    selected: tuple[_SelectedDocument, ...],
    *,
    expected_subnarratives: Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    expected_by_id = {
        row.subnarrative_id: row for row in expected_subnarratives
    }
    selected_by_id = {row.docid: row for row in selected}
    records = _jsonl(path, "selected subnarrative scores")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != _DOWNSTREAM_SCORE_FIELDS:
            raise ValueError("subnarrative score schema changed")
        docid = record.get("docid")
        subnarrative_id = record.get("subnarrative_id")
        document = selected_by_id.get(docid) if isinstance(docid, str) else None
        expected_subnarrative = expected_by_id.get(subnarrative_id)
        pair = (docid, subnarrative_id)
        if (
            record.get("topic_id") != topic_id
            or document is None
            or not isinstance(subnarrative_id, str)
            or expected_subnarrative is None
            or pair in seen
        ):
            raise ValueError("subnarrative score identity is invalid or duplicated")
        if (
            record.get("selection_rank") != document.rank
            or record.get("text_sha256") != document.text_sha256
        ):
            raise ValueError("subnarrative score document identity changed")
        _validate_score_components(record, "subnarrative score")
        if record.get("downstream_only") is not True:
            raise ValueError("subnarrative score must be downstream-only")
        queries = record.get("bm25_queries")
        query_hashes = record.get("bm25_query_sha256s")
        if (
            not isinstance(queries, list)
            or not queries
            or any(not isinstance(item, str) or not item for item in queries)
            or not isinstance(query_hashes, list)
            or len(query_hashes) != len(queries)
            or any(not isinstance(item, str) or not _SHA256.fullmatch(item) for item in query_hashes)
            or record.get("lane_name") != f"subnarrative:{subnarrative_id}"
            or record.get("semantic_query_sha256")
            != expected_subnarrative.semantic_query_sha256
            or queries != list(expected_subnarrative.bm25_queries)
            or query_hashes != list(expected_subnarrative.bm25_query_sha256s)
        ):
            raise ValueError("subnarrative score BM25 query provenance is invalid")
        seen.add(pair)
        rows.append(_downstream_score_provenance(record))
    return tuple(rows)


def _validate_score_components(record: Mapping[str, object], label: str) -> None:
    for key in ("bm25_rank", "aggregate_rank"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} rank is invalid")
    for key in (
        "bm25_score",
        "aggregate_score",
        "long_document_raw_logit",
        "weighted_passage_raw_logit",
    ):
        value = record.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label} value is not finite")
    support = record.get("within_document_span_support")
    if isinstance(support, bool) or not isinstance(support, int) or support < 0:
        raise ValueError(f"{label} span support is invalid")
    for key in ("semantic_query_sha256", "text_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{label} digest is invalid")
    if record.get("score_representation") != "raw_logits":
        raise ValueError(f"{label} representation is invalid")
    passages = record.get("winning_passages")
    if not isinstance(passages, list) or not passages:
        raise ValueError(f"{label} winning passages are invalid")
    for passage in passages:
        if not isinstance(passage, dict) or set(passage) != _PASSAGE_FIELDS:
            raise ValueError(f"{label} passage schema changed")
        integer_values = tuple(
            passage[key]
            for key in ("chunk_index", "start_char", "end_char", "weighted_rank")
        )
        raw_logit = passage["raw_logit"]
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values)
            or passage["chunk_index"] < 0
            or passage["start_char"] < 0
            or passage["end_char"] <= passage["start_char"]
            or passage["weighted_rank"] <= 0
            or isinstance(raw_logit, bool)
            or not isinstance(raw_logit, (int, float))
            or not math.isfinite(raw_logit)
        ):
            raise ValueError(f"{label} passage value is invalid")


def _downstream_score_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "topic_id": record["topic_id"],
        "lane_name": record["lane_name"],
        "semantic_query_sha256": record["semantic_query_sha256"],
        "docid": record["docid"],
        "aggregate_rank": record["aggregate_rank"],
        "aggregate_score": record["aggregate_score"],
        "long_document_raw_logit": record["long_document_raw_logit"],
        "weighted_passage_raw_logit": record["weighted_passage_raw_logit"],
        "within_document_span_support": record["within_document_span_support"],
        "winning_passages": [
            {
                "chunk_index": passage["chunk_index"],
                "start_char": passage["start_char"],
                "end_char": passage["end_char"],
                "raw_logit": passage["raw_logit"],
                "weighted_rank": passage["weighted_rank"],
            }
            for passage in record["winning_passages"]
        ],
        "score_representation": record["score_representation"],
        "text_sha256": record["text_sha256"],
        "selection_rank": record["selection_rank"],
        "subnarrative_id": record["subnarrative_id"],
        "bm25_queries": list(record["bm25_queries"]),
        "bm25_query_sha256s": list(record["bm25_query_sha256s"]),
        "downstream_only": True,
    }


def _validate_cross_score_matrix(
    scores: tuple[dict[str, Any], ...],
    selected: tuple[_SelectedDocument, ...],
    *,
    expected_subnarrative_ids: frozenset[str],
    allow_empty: bool = False,
) -> frozenset[str]:
    subnarratives = frozenset(row["subnarrative_id"] for row in scores)
    if not subnarratives:
        if allow_empty and not expected_subnarrative_ids:
            return frozenset()
        raise ValueError("document-by-subnarrative score matrix is empty")
    if subnarratives != expected_subnarrative_ids:
        raise ValueError("document-by-subnarrative score matrix differs from canonical plan")
    actual = {(row["docid"], row["subnarrative_id"]) for row in scores}
    expected = {
        (document.docid, subnarrative_id)
        for document in selected
        for subnarrative_id in subnarratives
    }
    if actual != expected:
        raise ValueError("document-by-subnarrative score matrix is incomplete")
    return subnarratives


def _validate_original_only_scoring(
    selected: tuple[_SelectedDocument, ...],
    lane_scores: Mapping[tuple[str, str], dict[str, Any]],
    cross_scores: tuple[dict[str, Any], ...],
    *,
    expected_query_sha256: str,
) -> None:
    if any(
        score[hash_field] != expected_query_sha256
        for score in lane_scores.values()
        for hash_field in ("bm25_query_sha256", "semantic_query_sha256")
    ):
        raise ValueError(
            "original-only fallback scores are not bound to original narrative query"
        )
    if (
        cross_scores
        or any(document.selected_from_lane != "original" for document in selected)
        or {lane_name for _docid, lane_name in lane_scores} != {"original"}
    ):
        raise ValueError("original-only fallback scoring contains downstream data")


def _load_supported_docids(
    path: Path,
    topic_id: str,
    selected: tuple[_SelectedDocument, ...],
    *,
    expected_subnarrative_ids: frozenset[str],
    nugget_manifest: Mapping[str, Any],
    allowed_evidence: frozenset[tuple[object, ...]],
    requests: Sequence[Any],
) -> tuple[set[str], dict[str, tuple[dict[str, Any], ...]]]:
    from trec_rag.canonical_nuggets import validate_canonical_nugget_result

    selected_by_id = {row.docid: row for row in selected}
    supported: set[str] = set()
    nuggets_by_doc: dict[str, list[dict[str, Any]]] = {
        row.docid: [] for row in selected
    }
    nugget_ids: set[str] = set()
    seen_subnarratives: set[str] = set()
    states: Counter[str] = Counter()
    records = _jsonl(path, "canonical nuggets")
    request_sha256s = [request.request_sha256 for request in requests]
    if (
        len(records) != nugget_manifest["result_count"]
        or nugget_manifest["request_sha256s"] != request_sha256s
    ):
        raise ValueError("canonical nugget result count changed")
    for record, request in zip(records, requests, strict=True):
        try:
            state = validate_canonical_nugget_result(record, request)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "canonical nugget result or canonical evidence semantics are invalid"
            ) from exc
        subnarrative_id = request.subnarrative_id
        if (
            request.topic_id != topic_id
            or subnarrative_id not in expected_subnarrative_ids
            or subnarrative_id in seen_subnarratives
            or request.selected_budget != nugget_manifest["selected_budget"]
        ):
            raise ValueError("canonical nugget subnarrative identity is invalid")
        seen_subnarratives.add(subnarrative_id)
        states[state] += 1
        nuggets = record["nuggets"]
        for nugget in nuggets:
            nugget_id = nugget["canonical_nugget_id"]
            evidence = nugget["evidence"]
            if nugget_id in nugget_ids:
                raise ValueError("canonical nugget identity or evidence is invalid")
            nugget_ids.add(nugget_id)
            for item in evidence:
                docid = item["docid"]
                document = selected_by_id.get(docid)
                if document is None:
                    raise ValueError("canonical evidence document is absent from selected documents")
                if item.get("document_sha256") != document.text_sha256:
                    raise ValueError("canonical evidence document hash changed")
                evidence_text = item.get("text")
                if (
                    not isinstance(evidence_text, str)
                    or not evidence_text
                    or item.get("text_sha256")
                    != sha256(evidence_text.encode("utf-8")).hexdigest()
                ):
                    raise ValueError("canonical evidence text identity changed")
                evidence_identity = (
                    subnarrative_id,
                    item.get("cluster_id"),
                    item.get("candidate_nugget_id"),
                    item.get("candidate_kind"),
                    evidence_text,
                    docid,
                    item.get("document_sha256"),
                )
                if evidence_identity not in allowed_evidence:
                    raise ValueError(
                        "canonical evidence is absent from sealed selected candidate and cluster"
                    )
                supported.add(document.docid)
                nugget_link = {
                    "canonical_nugget_id": nugget_id,
                    "nugget_kind": nugget.get("nugget_kind"),
                    "claim_text": nugget.get("claim_text"),
                    "subnarrative_id": subnarrative_id,
                    "evidence": {
                        "candidate_nugget_id": item["candidate_nugget_id"],
                        "candidate_kind": item["candidate_kind"],
                        "text": item["text"],
                        "text_sha256": item["text_sha256"],
                        "docid": item["docid"],
                        "document_sha256": item["document_sha256"],
                        "cluster_id": item["cluster_id"],
                    },
                }
                if nugget_link not in nuggets_by_doc[document.docid]:
                    nuggets_by_doc[document.docid].append(nugget_link)
    if (
        seen_subnarratives != set(expected_subnarrative_ids)
        or len(records) != nugget_manifest["result_count"]
        or dict(sorted(states.items())) != nugget_manifest["state_counts"]
    ):
        raise ValueError("canonical nugget result set is incomplete")
    return supported, {
        docid: tuple(rows) for docid, rows in nuggets_by_doc.items()
    }


def _load_allowed_canonical_evidence(
    topic_root: Path,
    topic: Topic,
    selected: tuple[_SelectedDocument, ...],
    subnarratives: Sequence[Any],
    *,
    selected_budget: int,
    max_canonical_claims: int,
    max_supporting_documents_per_claim: int,
) -> tuple[frozenset[tuple[object, ...]], tuple[Any, ...]]:
    from trec_rag.canonical_nuggets import (
        build_canonical_nugget_request,
        load_validated_selection_artifacts,
    )
    from trec_rag.evidence_store import load_validated_candidate_artifacts

    canonical_root = topic_root / "canonical"
    selections_path = canonical_root / "subnarrative-selections.jsonl"
    selection_manifest_path = canonical_root / "selection-manifest.json"
    selections, manifest, _policy = load_validated_selection_artifacts(
        selections_path, selection_manifest_path
    )
    candidates_path = canonical_root / "candidates.jsonl"
    candidate_manifest_path = canonical_root / "candidate-manifest.json"
    contexts_path = canonical_root / "handoff" / "selection-contexts.jsonl"
    fixed_files = {
        "candidates_file": candidates_path.name,
        "candidate_manifest_file": candidate_manifest_path.name,
        "contexts_file": contexts_path.name,
    }
    digests = {
        "candidates_sha256": sha256(candidates_path.read_bytes()).hexdigest(),
        "candidate_manifest_sha256": sha256(
            candidate_manifest_path.read_bytes()
        ).hexdigest(),
        "contexts_sha256": sha256(contexts_path.read_bytes()).hexdigest(),
    }
    if any(manifest.get(key) != value for key, value in fixed_files.items()) or any(
        manifest.get(key) != value for key, value in digests.items()
    ):
        raise ValueError("selection manifest evidence artifact seal changed")
    expected_subnarratives = {
        row.subnarrative_id: row.text for row in subnarratives
    }
    selection_by_id: dict[str, Any] = {}
    for selection in selections:
        context = selection.context
        if (
            context.topic_id != topic.id
            or context.official_narrative != topic.narrative
            or expected_subnarratives.get(context.subnarrative_id)
            != context.subnarrative_text
            or context.subnarrative_id in selection_by_id
        ):
            raise ValueError("canonical selection differs from canonical plan")
        selection_by_id[context.subnarrative_id] = selection
    if set(selection_by_id) != set(expected_subnarratives):
        raise ValueError("canonical selection set is incomplete")
    documents = {row.docid: row.text for row in selected}
    candidates = load_validated_candidate_artifacts(
        candidates_path,
        candidate_manifest_path,
        documents=documents,
        subnarratives=expected_subnarratives,
    )
    allowed: set[tuple[object, ...]] = set()
    for subnarrative_id, selection in selection_by_id.items():
        snapshots = {
            snapshot.budget: snapshot for snapshot in selection.snapshots
        }
        snapshot = snapshots.get(selected_budget)
        if snapshot is None:
            raise ValueError("canonical selection lacks configured budget")
        clusters = {cluster.cluster_id: cluster for cluster in selection.clusters}
        for cluster_id in snapshot.cluster_ids:
            cluster = clusters.get(cluster_id)
            if cluster is None:
                raise ValueError("canonical selection snapshot names an unknown cluster")
            for member in cluster.supports:
                candidate = candidates.get(
                    (subnarrative_id, member.candidate_nugget_id)
                )
                if (
                    candidate is None
                    or member.candidate_kind != candidate.candidate_kind
                    or member.text != candidate.text
                    or member.docid != candidate.docid
                    or member.document_sha256 != candidate.document_sha256
                    or member.raw_logit != candidate.sentence_cross_encoder_score
                ):
                    raise ValueError(
                        "canonical selection member differs from sealed candidate"
                    )
                allowed.add(
                    (
                        subnarrative_id,
                        cluster_id,
                        candidate.candidate_nugget_id,
                        candidate.candidate_kind,
                        candidate.text,
                        candidate.docid,
                        candidate.document_sha256,
                    )
                )
    requests = tuple(
        build_canonical_nugget_request(
            selection,
            selected_budget,
            max_canonical_claims=max_canonical_claims,
            max_supporting_documents_per_claim=(
                max_supporting_documents_per_claim
            ),
        )
        for selection in selections
    )
    return frozenset(allowed), requests


def _jsonl(path: Path, label: str) -> tuple[object, ...]:
    body = path.read_bytes()
    if body and not body.endswith(b"\n"):
        raise ValueError(f"{label} must end with LF")
    return tuple(
        _strict_json(line, f"{label} line {line_number}")
        for line_number, line in enumerate(body.splitlines(), start=1)
        if line
    )


def _strict_json(body: bytes, label: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains non-standard JSON constant {value}")

    try:
        return json.loads(
            body,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc


def _trec_bytes(rows: Sequence[tuple[str, str, int, int, str]]) -> bytes:
    return "".join(
        f"{topic_id} Q0 {docid} {rank} {score} {run_id}\n"
        for topic_id, docid, rank, score, run_id in rows
    ).encode("utf-8")


def _validate_trec_bytes(
    body: bytes, expected: Sequence[tuple[str, str, int, int, str]]
) -> None:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("emitted TREC run is not UTF-8") from exc
    if body and not body.endswith(b"\n"):
        raise ValueError("emitted TREC run lacks final LF")
    reparsed: list[tuple[str, str, int, int, str]] = []
    pairs: set[tuple[str, str]] = set()
    last_rank: dict[str, int] = {}
    last_score: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split(" ")
        if len(fields) != 6 or any(not field for field in fields) or fields[1] != "Q0":
            raise ValueError("emitted TREC row is malformed")
        topic_id, _q0, docid, rank_text, score_text, run_id = fields
        try:
            rank = int(rank_text)
            score = int(score_text)
        except ValueError as exc:
            raise ValueError("emitted TREC rank or score is not an integer") from exc
        pair = (topic_id, docid)
        if pair in pairs:
            raise ValueError("duplicate topic-document pair in emitted TREC run")
        if rank != last_rank.get(topic_id, 0) + 1:
            raise ValueError("emitted TREC ranks are not contiguous")
        previous_score = last_score.get(topic_id)
        if previous_score is not None and score >= previous_score:
            raise ValueError("emitted TREC scores are not strictly decreasing")
        pairs.add(pair)
        last_rank[topic_id] = rank
        last_score[topic_id] = score
        reparsed.append((topic_id, docid, rank, score, run_id))
    if reparsed != list(expected):
        raise ValueError("emitted TREC run does not match its projection")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, object] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _deterministic_zip(member_body: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        member = zipfile.ZipInfo(
            "retrieval_with_text.jsonl", date_time=(1980, 1, 1, 0, 0, 0)
        )
        member.compress_type = zipfile.ZIP_DEFLATED
        member.create_system = 3
        member.external_attr = 0o100600 << 16
        archive.writestr(member, member_body)
    return buffer.getvalue()


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(body)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
