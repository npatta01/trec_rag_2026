"""Deterministic quality gating and rank-only facet-aware fusion.

The heavyweight tokenizer, window-planning, score-cache, and ROCm inference
implementation already lives in the facet-local MiniLM modules.  This module
reuses those exact helpers and owns only the new held-out pilot behavior:
top-five lexical gating, within-facet score-to-rank conversion, and six fusion
arms whose cross-stream inputs are ranks rather than raw scores.
"""

from __future__ import annotations

import argparse
import math
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .facet_local_minilm_preflight import (
    build_preflight as build_scoring_preflight,
)
from .facet_local_minilm_preflight import (
    MODEL_ID,
    MODEL_REVISION,
    load_verified_tokenizer,
    score_cache_context,
)
from .facet_local_minilm_rank import aggregate_top4
from .facet_local_minilm_score import run_full_scoring as run_rocm_scoring
from .rerank_score_cache import GlobalScoreCache


ORIGINAL_DEPTH = 100
FACET_DEPTH = 50
FORCED_FACET_DEPTH = 20
RRF_K = 60
XQUAD_LAMBDA = 0.35
ARM_NAMES = ("O", "RRF", "BI", "XQ", "CXQ", "TUS-C")
PROTECTED_TOPIC_IDS = frozenset({"144", "213", "224", "407", "515"})
PRIOR_PILOT_TOPIC_IDS = frozenset({"200", "225", "707", "897"})
_CONTENT_WARNING_PATTERNS = (
    r"\bessay\b",
    r"\bdictionary\b",
    r"\bscrabble\b",
    r"\bhomework\b",
    r"\btemplate\b",
)
_TASK2_CANDIDATE_SCHEMA = "facet-aware-fusion-candidate-v1"
_DEFAULT_MODEL_RECEIPT = Path(
    "outputs/rag25_facet_local_minilm_v1/model_v1/materialization.json"
)
_DEFAULT_SCORE_CACHE_ROOT = Path(
    "/home/npatta01/data/competitions/trec_rag_2026/cache/reranker"
)


@dataclass(frozen=True)
class GateDecision:
    """Frozen top-five lexical gate diagnostics for one facet stream."""

    facet_id: str
    accepted: bool
    structural_checks_passed: bool
    anchor_top5_count: int
    anchor_relation_top5_count: int
    wrong_domain_top5_count: int
    content_warning_top5_count: int
    failed_checks: tuple[str, ...]
    top5_document_ids: tuple[str, ...]
    diagnostics: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "facet_id": self.facet_id,
            "accepted": self.accepted,
            "structural_checks_passed": self.structural_checks_passed,
            "anchor_top5_count": self.anchor_top5_count,
            "anchor_relation_top5_count": self.anchor_relation_top5_count,
            "wrong_domain_top5_count": self.wrong_domain_top5_count,
            "content_warning_top5_count": self.content_warning_top5_count,
            "failed_checks": list(self.failed_checks),
            "top5_document_ids": list(self.top5_document_ids),
            "diagnostics": [dict(row) for row in self.diagnostics],
        }


@dataclass(frozen=True)
class FacetStream:
    """One MiniLM-ranked facet stream plus its frozen gate disposition."""

    facet_id: str
    manifest_order: int
    ranked_docs: tuple[Mapping[str, object], ...]
    accepted: bool
    topic_id: str = ""
    gate: GateDecision | None = None


def _value(row: Mapping[str, object] | object, field: str, default: object = None) -> object:
    if isinstance(row, Mapping):
        return row.get(field, default)
    return getattr(row, field, default)


def _document_id(row: Mapping[str, object] | object) -> str:
    value = _value(row, "document_id")
    if value is None:
        value = _value(row, "docid")
    if not isinstance(value, str) or not value:
        raise ValueError("document_id must be non-empty text")
    return value


def _positive_rank(row: Mapping[str, object] | object) -> int:
    value = _value(row, "rank")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("rank must be a positive integer")
    return value


def _ordered_docs(
    rows: Sequence[Mapping[str, object]],
    *,
    depth: int,
) -> tuple[Mapping[str, object], ...]:
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    materialized: list[tuple[int, str, Mapping[str, object]]] = []
    seen: set[str] = set()
    for row in rows:
        rank = _positive_rank(row)
        document_id = _document_id(row)
        if document_id in seen:
            raise ValueError("one stream may contain each document_id only once")
        seen.add(document_id)
        if rank <= depth:
            materialized.append((rank, document_id, row))
    return tuple(row for _rank, _document_id, row in sorted(materialized))


def rank_normalized(rank: int | None, depth: int) -> float:
    """Transform an in-stream rank without comparing cross-query raw scores."""

    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if rank is None:
        return 0.0
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer or None")
    if rank > depth:
        return 0.0
    return 1.0 / math.log2(1 + rank)


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _literal_term_matches(text: str, term: str) -> bool:
    normalized_term = _normalized_text(term)
    if not normalized_term:
        raise ValueError("gate terms must be non-empty text")
    return re.search(
        rf"(?<!\w){re.escape(normalized_term)}(?!\w)",
        text,
    ) is not None


def _gate_terms(facet: Mapping[str, object] | object, field: str) -> tuple[str, ...]:
    raw = _value(facet, field)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"facet {field} must be a sequence")
    terms = tuple(raw)
    if field != "wrong_domain_patterns" and not terms:
        raise ValueError(f"facet {field} must not be empty")
    if any(not isinstance(term, str) or not term.strip() for term in terms):
        raise ValueError(f"facet {field} terms must be non-empty text")
    return terms


def _document_text(row: Mapping[str, object] | object) -> str:
    for field in ("text", "passage", "contents"):
        value = _value(row, field)
        if isinstance(value, str):
            return value
    raise ValueError("ranked gate documents require text")


def quality_gate(
    facet: Mapping[str, object] | object,
    ranked_docs: Sequence[Mapping[str, object]],
) -> GateDecision:
    """Apply the frozen structural and MiniLM-top-five lexical gate."""

    facet_id = _value(facet, "facet_id")
    query = _value(facet, "query")
    if not isinstance(facet_id, str) or not facet_id:
        raise ValueError("facet_id must be non-empty text")
    if not isinstance(query, str) or not query:
        raise ValueError("facet query must be non-empty text")
    anchors = _gate_terms(facet, "anchor_terms")
    relations = _gate_terms(facet, "relation_terms")
    wrong_domain = _gate_terms(facet, "wrong_domain_patterns")

    normalized_query = _normalized_text(query)
    structural = any(
        _literal_term_matches(normalized_query, term) for term in anchors
    ) and any(_literal_term_matches(normalized_query, term) for term in relations)
    top_five = _ordered_docs(ranked_docs, depth=FACET_DEPTH)[:5]
    diagnostics: list[dict[str, object]] = []
    for row in top_five:
        text = _normalized_text(_document_text(row))
        anchor = any(_literal_term_matches(text, term) for term in anchors)
        relation = any(_literal_term_matches(text, term) for term in relations)
        try:
            domain = any(re.search(pattern, text, re.IGNORECASE) for pattern in wrong_domain)
        except re.error as exc:
            raise ValueError("wrong_domain_patterns must be valid regular expressions") from exc
        warning = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in _CONTENT_WARNING_PATTERNS
        )
        diagnostics.append(
            {
                "document_id": _document_id(row),
                "rank": _positive_rank(row),
                "anchor": anchor,
                "relation": relation,
                "anchor_relation": anchor and relation,
                "wrong_domain": domain,
                "content_warning": warning,
            }
        )

    anchor_count = sum(bool(row["anchor"]) for row in diagnostics)
    relation_count = sum(bool(row["anchor_relation"]) for row in diagnostics)
    wrong_count = sum(bool(row["wrong_domain"]) for row in diagnostics)
    warning_count = sum(bool(row["content_warning"]) for row in diagnostics)
    failed: list[str] = []
    if not structural:
        failed.append("structural")
    if anchor_count < 3:
        failed.append("anchor")
    if relation_count < 2:
        failed.append("anchor_relation")
    if wrong_count >= 2:
        failed.append("wrong_domain")
    return GateDecision(
        facet_id=facet_id,
        accepted=not failed,
        structural_checks_passed=structural,
        anchor_top5_count=anchor_count,
        anchor_relation_top5_count=relation_count,
        wrong_domain_top5_count=wrong_count,
        content_warning_top5_count=warning_count,
        failed_checks=tuple(failed),
        top5_document_ids=tuple(str(row["document_id"]) for row in diagnostics),
        diagnostics=tuple(diagnostics),
    )


def rank_facet_documents(
    candidates: Sequence[Mapping[str, object]],
    scored_windows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate the existing top-four window policy into one facet-local rank."""

    ordered_candidates = _ordered_docs(candidates, depth=ORIGINAL_DEPTH)
    by_document: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for window in scored_windows:
        by_document[_document_id(window)].append(window)
    candidate_ids = {_document_id(row) for row in ordered_candidates}
    if set(by_document) != candidate_ids:
        raise ValueError("scored windows must cover exactly the facet candidates")

    aggregated: list[dict[str, object]] = []
    for candidate in ordered_candidates:
        document_id = _document_id(candidate)
        retrieval_rank = _positive_rank(candidate)
        row = dict(candidate)
        row.update(
            {
                "document_id": document_id,
                "retrieval_rank": retrieval_rank,
                "score": aggregate_top4(by_document[document_id]),
            }
        )
        aggregated.append(row)
    aggregated.sort(
        key=lambda row: (
            -float(row["score"]),
            int(row["retrieval_rank"]),
            str(row["document_id"]),
        )
    )
    for rank, row in enumerate(aggregated, start=1):
        row["rank"] = rank
    return aggregated


def prepare_scoring_candidates(
    manifest: Mapping[str, object],
    retrieval_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Adapt Task 2 JSONL rows to the existing MiniLM window-plan contract."""

    if manifest.get("qrels_opened") is not False:
        raise ValueError("manifest qrels_opened must remain false")
    raw_facets = manifest.get("facets")
    if not isinstance(raw_facets, Sequence) or isinstance(raw_facets, (str, bytes)):
        raise ValueError("manifest facets must be a sequence")
    facets: dict[tuple[str, str], Mapping[str, object]] = {}
    for facet in raw_facets:
        if not isinstance(facet, Mapping):
            raise ValueError("manifest facet rows must be objects")
        topic_id = facet.get("topic_id")
        facet_id = facet.get("facet_id")
        query = facet.get("query")
        order = facet.get("manifest_order")
        if (
            not isinstance(topic_id, str)
            or not isinstance(facet_id, str)
            or not isinstance(query, str)
            or not query
            or isinstance(order, bool)
            or not isinstance(order, int)
            or order < 0
        ):
            raise ValueError("manifest facet identity/query/order is invalid")
        if topic_id in PROTECTED_TOPIC_IDS or topic_id in PRIOR_PILOT_TOPIC_IDS:
            raise ValueError(f"forbidden topic {topic_id} in manifest facets")
        key = (topic_id, facet_id)
        if key in facets:
            raise ValueError("duplicate manifest facet identity")
        facets[key] = facet

    expected_keys = {
        "schema_version",
        "request_key",
        "topic_id",
        "facet_id",
        "query_sha256",
        "rank",
        "docid",
        "score",
        "text",
    }
    adapted: list[tuple[int, int, str, dict[str, object]]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for row in retrieval_rows:
        if set(row) != expected_keys or row.get("schema_version") != _TASK2_CANDIDATE_SCHEMA:
            raise ValueError("Task 2 candidate row fields or schema differ")
        topic_id = row.get("topic_id")
        facet_id = row.get("facet_id")
        rank = row.get("rank")
        docid = row.get("docid")
        text = row.get("text")
        if not isinstance(topic_id, str) or not isinstance(facet_id, str):
            raise ValueError("Task 2 candidate topic/facet identity is invalid")
        if topic_id in PROTECTED_TOPIC_IDS or topic_id in PRIOR_PILOT_TOPIC_IDS:
            raise ValueError(f"forbidden topic {topic_id} in Task 2 candidates")
        facet = facets.get((topic_id, facet_id))
        if facet is None:
            raise ValueError("Task 2 candidate is not in the frozen facet manifest")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 1 <= rank <= ORIGINAL_DEPTH
            or not isinstance(docid, str)
            or not docid
            or not isinstance(text, str)
        ):
            raise ValueError("Task 2 candidate rank/document/text is invalid")
        query = str(facet["query"])
        if row.get("query_sha256") != hashlib.sha256(query.encode("utf-8")).hexdigest():
            raise ValueError("Task 2 candidate query hash differs from manifest")
        identity = (topic_id, facet_id, rank, docid)
        if identity in seen:
            raise ValueError("duplicate Task 2 candidate identity")
        seen.add(identity)
        materialized = {
            **row,
            "family": "facet",
            "variant": facet_id,
            "document_id": docid,
            "query": query,
        }
        adapted.append(
            (int(facet["manifest_order"]), rank, docid, materialized)
        )
    return tuple(row for _order, _rank, _docid, row in sorted(adapted))


def _load_task2_candidates(
    manifest_path: Path,
    retrieval_dir: Path,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    candidate_path = Path(retrieval_dir) / "candidates.jsonl"
    summary_path = Path(retrieval_dir) / "retrieval_summary.json"
    candidate_source = candidate_path.read_bytes()
    try:
        summary = json.loads(summary_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Task 2 retrieval summary is invalid JSON") from exc
    if (
        not isinstance(summary, dict)
        or summary.get("complete") is not True
        or summary.get("qrels_opened") is not False
        or summary.get("candidates_sha256") != _sha256(candidate_source)
        or summary.get("manifest_sha256") != _sha256(Path(manifest_path).read_bytes())
    ):
        raise ValueError("Task 2 retrieval summary bindings are invalid")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(candidate_source.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Task 2 candidate row {line_number} is invalid") from exc
        if not isinstance(row, dict) or _canonical_json_bytes(row, pretty=False).rstrip(b"\n") != line:
            raise ValueError("Task 2 candidates must be canonical JSONL")
        rows.append(row)
    if summary.get("candidate_rows") != len(rows):
        raise ValueError("Task 2 candidate row count differs from summary")
    _reject_qrels_fields(rows, path="Task 2 candidates")
    return tuple(rows), summary


def _load_rank_inputs(
    *,
    manifest_path: Path,
    retrieval_dir: Path,
    cache_root: Path,
) -> tuple[dict[str, object], tuple[dict[str, object], ...], dict[str, object]]:
    from .facet_aware_fusion_manifest import load_manifest

    manifest = load_manifest(Path(manifest_path), cache_root=Path(cache_root))
    rows, summary = _load_task2_candidates(manifest_path, retrieval_dir)
    return manifest, prepare_scoring_candidates(manifest, rows), summary


def run_rank_preflight(
    *,
    manifest_path: Path,
    retrieval_dir: Path,
    cache_root: Path,
    output_dir: Path,
    model_receipt: Path = _DEFAULT_MODEL_RECEIPT,
    score_cache_root: Path = _DEFAULT_SCORE_CACHE_ROOT,
) -> dict[str, object]:
    """Compute exact local tokenizer/window/cache work without writing or inference."""

    destination = Path(output_dir)
    if destination.exists() or os.path.lexists(destination):
        raise FileExistsError(f"create-only ranking freeze already exists: {destination}")
    manifest, candidates, retrieval_summary = _load_rank_inputs(
        manifest_path=manifest_path,
        retrieval_dir=retrieval_dir,
        cache_root=cache_root,
    )
    tokenizer = load_verified_tokenizer(Path(model_receipt))
    context = score_cache_context()
    cache = GlobalScoreCache(Path(score_cache_root), context)
    plan = build_scoring_preflight(candidates, tokenizer, cache)
    return {
        "schema_version": "facet-aware-fusion-rank-preflight-v1",
        "status": "tokenizer_only_preflight_complete",
        "experiment_id": manifest.get("experiment_id"),
        "topic_ids": list(manifest["topic_ids"]),
        "manifest_sha256": _sha256(Path(manifest_path).read_bytes()),
        "retrieval_candidates_sha256": retrieval_summary["candidates_sha256"],
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "score_cache_context": context.artifact_metadata,
        **plan.summary,
        "inference_count": 0,
        "retrieval_call_count": 0,
        "qrels_access_count": 0,
        "qrels_opened": False,
    }


def _original_rank_rows(
    manifest: Mapping[str, object],
    cache_root: Path,
) -> list[dict[str, object]]:
    topics = manifest.get("topics")
    if not isinstance(topics, Sequence) or isinstance(topics, (str, bytes)):
        raise ValueError("manifest topics must be a sequence")
    rows: list[dict[str, object]] = []
    for topic in topics:
        if not isinstance(topic, Mapping):
            raise ValueError("manifest topic rows must be objects")
        topic_id = topic.get("topic_id")
        filename = topic.get("original_cache_filename")
        if not isinstance(topic_id, str) or not isinstance(filename, str):
            raise ValueError("manifest original-cache binding is invalid")
        try:
            payload = json.loads((Path(cache_root) / filename).read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"original cache for topic {topic_id} is invalid") from exc
        response = payload.get("response") if isinstance(payload, Mapping) else None
        candidates = response.get("candidates") if isinstance(response, Mapping) else None
        if not isinstance(candidates, list) or len(candidates) < ORIGINAL_DEPTH:
            raise ValueError(f"original cache for topic {topic_id} has fewer than 100 candidates")
        observed: set[str] = set()
        for expected_rank, candidate in enumerate(candidates[:ORIGINAL_DEPTH], start=1):
            if not isinstance(candidate, Mapping):
                raise ValueError("original cache candidate must be an object")
            docid = candidate.get("docid")
            rank = candidate.get("rank")
            text = candidate.get("doc")
            if (
                not isinstance(docid, str)
                or not docid
                or docid in observed
                or rank != expected_rank
                or not isinstance(text, str)
            ):
                raise ValueError(f"original cache candidates are invalid for topic {topic_id}")
            observed.add(docid)
            rows.append(
                {
                    "topic_id": topic_id,
                    "family": "original",
                    "document_id": docid,
                    "rank": expected_rank,
                    "score": candidate.get("score"),
                    "text": text,
                    "source_cache_filename": filename,
                }
            )
    return rows


def _cached_facet_rank_rows(
    manifest: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    windows: Sequence[object],
    cache: object,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    raw_facets = manifest.get("facets")
    if not isinstance(raw_facets, Sequence) or isinstance(raw_facets, (str, bytes)):
        raise ValueError("manifest facets must be a sequence")
    facets = {
        (str(facet["topic_id"]), str(facet["facet_id"])): facet
        for facet in raw_facets
        if isinstance(facet, Mapping)
    }
    candidates_by_facet: dict[
        tuple[str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    for candidate in candidates:
        candidates_by_facet[
            (str(candidate["topic_id"]), str(candidate["variant"]))
        ].append(candidate)
    windows_by_facet: dict[
        tuple[str, str], list[dict[str, object]]
    ] = defaultdict(list)
    score_rows: list[dict[str, object]] = []
    scores = getattr(cache, "scores", None)
    if not isinstance(scores, Mapping):
        raise ValueError("score cache does not expose authenticated scores")
    for window in windows:
        to_dict = getattr(window, "to_dict", None)
        row = to_dict() if callable(to_dict) else dict(window)  # type: ignore[arg-type]
        cache_key = row.get("cache_key")
        if not isinstance(cache_key, str) or cache_key not in scores:
            raise ValueError("all planned windows must be present in the global score cache")
        scored = {**row, "score": float(scores[cache_key])}
        key = (str(row["topic_id"]), str(row["variant"]))
        windows_by_facet[key].append(scored)
        score_rows.append(scored)

    ranked_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    for key, facet in sorted(
        facets.items(), key=lambda item: int(item[1]["manifest_order"])
    ):
        facet_candidates = candidates_by_facet.get(key, [])
        if len(facet_candidates) != ORIGINAL_DEPTH:
            raise ValueError(f"facet {key[1]} must have exactly 100 Task 2 candidates")
        ranked = rank_facet_documents(facet_candidates, windows_by_facet.get(key, []))
        decision = quality_gate(facet, ranked)
        gate_rows.append(
            {
                "topic_id": key[0],
                "manifest_order": int(facet["manifest_order"]),
                **decision.to_dict(),
            }
        )
        for row in ranked[:FACET_DEPTH]:
            ranked_rows.append(
                {
                    **row,
                    "topic_id": key[0],
                    "family": "facet",
                    "facet_id": key[1],
                    "manifest_order": int(facet["manifest_order"]),
                    "accepted": decision.accepted,
                }
            )
    if set(candidates_by_facet) != set(facets) or set(windows_by_facet) != set(facets):
        raise ValueError("candidate/window facet identities differ from the manifest")
    return ranked_rows, gate_rows, score_rows


def run_rank_freeze(
    *,
    manifest_path: Path,
    retrieval_dir: Path,
    cache_root: Path,
    output_dir: Path,
    model_receipt: Path = _DEFAULT_MODEL_RECEIPT,
    score_cache_root: Path = _DEFAULT_SCORE_CACHE_ROOT,
) -> dict[str, object]:
    """Freeze rankings only after separate approval-gated scoring filled the cache."""

    destination = Path(output_dir)
    if destination.exists() or os.path.lexists(destination):
        raise FileExistsError(f"create-only ranking freeze already exists: {destination}")
    manifest, candidates, retrieval_summary = _load_rank_inputs(
        manifest_path=manifest_path,
        retrieval_dir=retrieval_dir,
        cache_root=cache_root,
    )
    tokenizer = load_verified_tokenizer(Path(model_receipt))
    context = score_cache_context()
    cache = GlobalScoreCache(Path(score_cache_root), context)
    plan = build_scoring_preflight(candidates, tokenizer, cache)
    misses = int(plan.summary.get("unique_uncached_pair_count", 0))
    if misses:
        raise ValueError(
            f"rank freeze has {misses} cache misses; run the separate approval-gated "
            "ROCm scorer before freezing"
        )
    original_rows = _original_rank_rows(manifest, cache_root)
    facet_rows, gates, score_rows = _cached_facet_rank_rows(
        manifest, candidates, plan.windows, cache
    )
    frozen_rows = [*original_rows, *facet_rows]
    score_bytes = _canonical_jsonl_bytes(_stable_rows(score_rows))
    return create_ranking_freeze(
        destination,
        frozen_rows=frozen_rows,
        gates=gates,
        score_rows=score_rows,
        candidate_provenance=[*original_rows, *candidates],
        input_hashes={
            "manifest_sha256": _sha256(Path(manifest_path).read_bytes()),
            "retrieval_sha256": str(retrieval_summary["candidates_sha256"]),
            "scoring_sha256": _sha256(score_bytes),
            "model_materialization_sha256": _sha256(Path(model_receipt).read_bytes()),
        },
        topic_order=[str(topic_id) for topic_id in manifest["topic_ids"]],
    )


def _coerce_facet(value: FacetStream | Mapping[str, object]) -> FacetStream:
    if isinstance(value, FacetStream):
        facet = value
    else:
        facet_id = value.get("facet_id")
        manifest_order = value.get("manifest_order")
        docs = value.get("ranked_docs", value.get("documents"))
        gate = value.get("gate")
        accepted = value.get("accepted")
        if isinstance(gate, GateDecision):
            accepted = gate.accepted
        elif isinstance(gate, Mapping):
            accepted = gate.get("accepted")
        if not isinstance(facet_id, str) or not facet_id:
            raise ValueError("facet_id must be non-empty text")
        if (
            isinstance(manifest_order, bool)
            or not isinstance(manifest_order, int)
            or manifest_order < 0
        ):
            raise ValueError("facet manifest_order must be a non-negative integer")
        if not isinstance(docs, Sequence) or isinstance(docs, (str, bytes)):
            raise ValueError("facet ranked_docs must be a sequence")
        if not isinstance(accepted, bool):
            raise ValueError("facet accepted disposition must be boolean")
        facet = FacetStream(
            facet_id=facet_id,
            manifest_order=manifest_order,
            ranked_docs=tuple(docs),  # type: ignore[arg-type]
            accepted=accepted,
            topic_id=str(value.get("topic_id", "")),
            gate=gate if isinstance(gate, GateDecision) else None,
        )
    if not facet.facet_id:
        raise ValueError("facet_id must be non-empty text")
    if facet.manifest_order < 0:
        raise ValueError("facet manifest_order must be non-negative")
    return facet


def _ordered_facets(
    facets: Sequence[FacetStream | Mapping[str, object]],
) -> tuple[FacetStream, ...]:
    materialized = tuple(_coerce_facet(facet) for facet in facets)
    identities = [(facet.manifest_order, facet.facet_id) for facet in materialized]
    if len({facet.facet_id for facet in materialized}) != len(materialized):
        raise ValueError("facet_id values must be unique within a topic")
    if len({facet.manifest_order for facet in materialized}) != len(materialized):
        raise ValueError("facet manifest_order values must be unique within a topic")
    return tuple(
        facet
        for _identity, facet in sorted(zip(identities, materialized, strict=True))
    )


@dataclass(frozen=True)
class _RankState:
    original: tuple[Mapping[str, object], ...]
    facets: tuple[FacetStream, ...]
    facet_docs: dict[str, tuple[Mapping[str, object], ...]]
    original_rank: dict[str, int]
    facet_rank: dict[str, dict[str, int]]
    pool: tuple[str, ...]
    relevance: dict[str, float]
    best_rank: dict[str, int]


def _rank_state(
    original: Sequence[Mapping[str, object]],
    facets: Sequence[FacetStream | Mapping[str, object]],
) -> _RankState:
    ordered_original = _ordered_docs(original, depth=ORIGINAL_DEPTH)
    accepted = tuple(facet for facet in _ordered_facets(facets) if facet.accepted)
    facet_docs = {
        facet.facet_id: _ordered_docs(facet.ranked_docs, depth=FACET_DEPTH)
        for facet in accepted
    }
    original_rank = {
        _document_id(row): _positive_rank(row) for row in ordered_original
    }
    facet_rank = {
        facet.facet_id: {
            _document_id(row): _positive_rank(row)
            for row in facet_docs[facet.facet_id]
        }
        for facet in accepted
    }
    pool = tuple(
        sorted(
            set(original_rank).union(
                *(set(ranks) for ranks in facet_rank.values())
            )
        )
    )
    relevance: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for document_id in pool:
        ranks = [
            rank
            for rank in (
                original_rank.get(document_id),
                *(facet_rank[facet.facet_id].get(document_id) for facet in accepted),
            )
            if rank is not None
        ]
        best_rank[document_id] = min(ranks)
        relevance[document_id] = max(
            rank_normalized(original_rank.get(document_id), ORIGINAL_DEPTH),
            *(
                rank_normalized(
                    facet_rank[facet.facet_id].get(document_id), FACET_DEPTH
                )
                for facet in accepted
            ),
        )
    return _RankState(
        original=ordered_original,
        facets=accepted,
        facet_docs=facet_docs,
        original_rank=original_rank,
        facet_rank=facet_rank,
        pool=pool,
        relevance=relevance,
        best_rank=best_rank,
    )


def _original_ids(state: _RankState, limit: int) -> list[str]:
    return [_document_id(row) for row in state.original[:limit]]


def family_rrf(
    original: Sequence[Mapping[str, object]],
    facets: Sequence[FacetStream | Mapping[str, object]],
    *,
    limit: int = ORIGINAL_DEPTH,
) -> list[str]:
    """Return the current family-balanced rank-only RRF control."""

    state = _rank_state(original, facets)
    if not state.facets:
        return _original_ids(state, limit)
    facet_weight = 0.5 / len(state.facets)
    scores = {
        document_id: (
            0.5 / (RRF_K + state.original_rank[document_id])
            if document_id in state.original_rank
            else 0.0
        )
        + math.fsum(
            facet_weight / (RRF_K + ranks[document_id])
            for ranks in state.facet_rank.values()
            if document_id in ranks
        )
        for document_id in state.pool
    }
    return sorted(
        state.pool,
        key=lambda document_id: (
            -scores[document_id],
            state.best_rank[document_id],
            document_id,
        ),
    )[:limit]


def balanced_interleave(
    original: Sequence[Mapping[str, object]],
    facets: Sequence[FacetStream | Mapping[str, object]],
    *,
    limit: int = ORIGINAL_DEPTH,
) -> list[str]:
    """Alternate original opportunities with accepted facets in manifest order."""

    state = _rank_state(original, facets)
    if not state.facets:
        return _original_ids(state, limit)
    streams = [
        [_document_id(row) for row in state.original],
        *(
            [_document_id(row) for row in state.facet_docs[facet.facet_id]]
            for facet in state.facets
        ),
    ]
    cursors = [0] * len(streams)
    selected: list[str] = []
    seen: set[str] = set()
    opportunity = 0
    facet_opportunity = 0
    while len(selected) < limit:
        if opportunity % 2 == 0:
            stream_index = 0
        else:
            stream_index = 1 + (facet_opportunity % len(state.facets))
            facet_opportunity += 1
        opportunity += 1
        stream = streams[stream_index]
        while cursors[stream_index] < len(stream):
            document_id = stream[cursors[stream_index]]
            cursors[stream_index] += 1
            if document_id not in seen:
                seen.add(document_id)
                selected.append(document_id)
                break
        if all(cursor >= len(stream) for cursor, stream in zip(cursors, streams)):
            break
    return selected


def _xquad_objective(
    state: _RankState,
    document_id: str,
    residual: Mapping[str, float],
) -> float:
    if not state.facets:
        return state.relevance[document_id]
    weight = 1.0 / len(state.facets)
    novelty = math.fsum(
        weight
        * rank_normalized(
            state.facet_rank[facet.facet_id].get(document_id), FACET_DEPTH
        )
        * residual[facet.facet_id]
        for facet in state.facets
    )
    return (1.0 - XQUAD_LAMBDA) * state.relevance[document_id] + (
        XQUAD_LAMBDA * novelty
    )


def _best_xquad(
    state: _RankState,
    remaining: set[str],
    residual: Mapping[str, float],
) -> str:
    return min(
        remaining,
        key=lambda document_id: (
            -_xquad_objective(state, document_id, residual),
            -state.relevance[document_id],
            state.best_rank[document_id],
            document_id,
        ),
    )


def _update_residual(
    state: _RankState,
    residual: dict[str, float],
    document_id: str,
) -> None:
    for facet in state.facets:
        residual[facet.facet_id] *= 1.0 - rank_normalized(
            state.facet_rank[facet.facet_id].get(document_id), FACET_DEPTH
        )


def xquad(
    original: Sequence[Mapping[str, object]],
    facets: Sequence[FacetStream | Mapping[str, object]],
    *,
    limit: int = ORIGINAL_DEPTH,
) -> list[str]:
    """Greedily select the frozen rank-normalized xQuAD arm."""

    state = _rank_state(original, facets)
    if not state.facets:
        return _original_ids(state, limit)
    remaining = set(state.pool)
    residual = {facet.facet_id: 1.0 for facet in state.facets}
    selected: list[str] = []
    while remaining and len(selected) < limit:
        document_id = _best_xquad(state, remaining, residual)
        remaining.remove(document_id)
        selected.append(document_id)
        _update_residual(state, residual, document_id)
    return selected


def coverage_deadlines(facet_count: int) -> tuple[int, ...]:
    """Return one-based CXQ deadlines in frozen manifest order."""

    if (
        isinstance(facet_count, bool)
        or not isinstance(facet_count, int)
        or facet_count < 1
    ):
        raise ValueError("facet_count must be a positive integer")
    return tuple(
        10 + math.ceil(40 * index / facet_count)
        for index in range(1, facet_count + 1)
    )


def constrained_xquad(
    original: Sequence[Mapping[str, object]],
    facets: Sequence[FacetStream | Mapping[str, object]],
    *,
    limit: int = ORIGINAL_DEPTH,
) -> tuple[list[str], list[dict[str, object]]]:
    """Run xQuAD with accepted-facet top-20 coverage deadlines."""

    state = _rank_state(original, facets)
    if not state.facets:
        ranking = _original_ids(state, limit)
        return ranking, [
            {
                "position": position,
                "document_id": document_id,
                "selection": "original_fallback",
                "forced_facet": None,
                "deadline": None,
                "source_rank": state.original_rank[document_id],
                "prior_coverage_state": {},
            }
            for position, document_id in enumerate(ranking, start=1)
        ]

    deadlines = dict(
        zip(
            (facet.facet_id for facet in state.facets),
            coverage_deadlines(len(state.facets)),
            strict=True,
        )
    )
    remaining = set(state.pool)
    residual = {facet.facet_id: 1.0 for facet in state.facets}
    covered = {facet.facet_id: False for facet in state.facets}
    selected: list[str] = []
    provenance: list[dict[str, object]] = []
    while remaining and len(selected) < limit:
        position = len(selected) + 1
        prior_state = dict(covered)
        overdue = [
            facet
            for facet in state.facets
            if not covered[facet.facet_id]
            and deadlines[facet.facet_id] <= position
        ]
        forced_facet: FacetStream | None = overdue[0] if overdue else None
        document_id: str | None = None
        source_rank: int | None = None
        if forced_facet is not None:
            forced_candidates = [
                row
                for row in state.facet_docs[forced_facet.facet_id]
                if _positive_rank(row) <= FORCED_FACET_DEPTH
                and _document_id(row) in remaining
            ]
            if forced_candidates:
                selected_row = forced_candidates[0]
                document_id = _document_id(selected_row)
                source_rank = _positive_rank(selected_row)
        if document_id is None:
            forced_facet = None
            document_id = _best_xquad(state, remaining, residual)
        remaining.remove(document_id)
        selected.append(document_id)
        for facet in state.facets:
            facet_source_rank = state.facet_rank[facet.facet_id].get(document_id)
            if facet_source_rank is not None and facet_source_rank <= FORCED_FACET_DEPTH:
                covered[facet.facet_id] = True
        provenance.append(
            {
                "position": position,
                "document_id": document_id,
                "selection": "deadline" if forced_facet is not None else "xquad",
                "forced_facet": (
                    forced_facet.facet_id if forced_facet is not None else None
                ),
                "deadline": (
                    deadlines[forced_facet.facet_id]
                    if forced_facet is not None
                    else None
                ),
                "source_rank": source_rank,
                "prior_coverage_state": prior_state,
            }
        )
        _update_residual(state, residual, document_id)
    return selected, provenance


def tus_consensus(
    original: Sequence[Mapping[str, object]],
    facets: Sequence[FacetStream | Mapping[str, object]],
    *,
    limit: int = ORIGINAL_DEPTH,
) -> list[str]:
    """Return the deterministic rank-only multi-facet consensus diagnostic."""

    state = _rank_state(original, facets)
    if not state.facets:
        return _original_ids(state, limit)
    facet_weight = 1.0 / len(state.facets)
    scores = {
        document_id: 0.5
        * rank_normalized(state.original_rank.get(document_id), ORIGINAL_DEPTH)
        + 0.5
        * math.fsum(
            facet_weight
            * rank_normalized(
                state.facet_rank[facet.facet_id].get(document_id), FACET_DEPTH
            )
            for facet in state.facets
        )
        for document_id in state.pool
    }
    return sorted(
        state.pool,
        key=lambda document_id: (
            -scores[document_id],
            -state.relevance[document_id],
            state.best_rank[document_id],
            document_id,
        ),
    )[:limit]


def _row_bool(row: Mapping[str, object], field: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"facet row {field} must be boolean")
    return value


def build_rankings(
    frozen_rows: Sequence[Mapping[str, object]],
    *,
    limit: int = ORIGINAL_DEPTH,
) -> dict[str, dict[str, list[str]]]:
    """Build all six arms from flat, gate-frozen rank rows."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    by_topic_original: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_topic_facet: dict[
        tuple[str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    facet_metadata: dict[tuple[str, str], tuple[int, bool]] = {}
    for row in frozen_rows:
        topic_id = row.get("topic_id")
        family = row.get("family")
        if not isinstance(topic_id, str) or not topic_id:
            raise ValueError("frozen row topic_id must be non-empty text")
        if topic_id in PROTECTED_TOPIC_IDS:
            raise ValueError(f"protected topic {topic_id} is forbidden")
        if topic_id in PRIOR_PILOT_TOPIC_IDS:
            raise ValueError(f"prior-pilot topic {topic_id} is forbidden")
        if family == "original":
            by_topic_original[topic_id].append(row)
            continue
        if family != "facet":
            raise ValueError("frozen row family must be original or facet")
        facet_id = row.get("facet_id", row.get("variant"))
        manifest_order = row.get("manifest_order")
        if not isinstance(facet_id, str) or not facet_id:
            raise ValueError("facet rows require facet_id")
        if (
            isinstance(manifest_order, bool)
            or not isinstance(manifest_order, int)
            or manifest_order < 0
        ):
            raise ValueError("facet rows require non-negative manifest_order")
        accepted = _row_bool(row, "accepted")
        key = (topic_id, facet_id)
        metadata = (manifest_order, accepted)
        previous = facet_metadata.setdefault(key, metadata)
        if previous != metadata:
            raise ValueError("facet rows disagree on manifest order or gate decision")
        by_topic_facet[key].append(row)

    topic_ids = set(by_topic_original).union(topic for topic, _facet in by_topic_facet)
    rankings: dict[str, dict[str, list[str]]] = {}
    for topic_id in sorted(topic_ids, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
        original = by_topic_original.get(topic_id, [])
        if not original:
            raise ValueError(f"topic {topic_id} has no original stream")
        facets = [
            FacetStream(
                facet_id=facet_id,
                manifest_order=facet_metadata[(facet_topic, facet_id)][0],
                ranked_docs=tuple(rows),
                accepted=facet_metadata[(facet_topic, facet_id)][1],
                topic_id=topic_id,
            )
            for (facet_topic, facet_id), rows in by_topic_facet.items()
            if facet_topic == topic_id
        ]
        state = _rank_state(original, facets)
        cxq, _provenance = constrained_xquad(original, facets, limit=limit)
        rankings[topic_id] = {
            "O": _original_ids(state, limit),
            "RRF": family_rrf(original, facets, limit=limit),
            "BI": balanced_interleave(original, facets, limit=limit),
            "XQ": xquad(original, facets, limit=limit),
            "CXQ": cxq,
            "TUS-C": tus_consensus(original, facets, limit=limit),
        }
    return rankings


def ranking_parameters() -> dict[str, object]:
    """Return the complete frozen, qrels-blind ranking parameter record."""

    return {
        "arms": list(ARM_NAMES),
        "original_depth": ORIGINAL_DEPTH,
        "facet_depth": FACET_DEPTH,
        "forced_facet_depth": FORCED_FACET_DEPTH,
        "rrf_k": RRF_K,
        "original_family_weight": 0.5,
        "facet_family_weight": 0.5,
        "xquad_lambda": XQUAD_LAMBDA,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "facet_score_aggregation": "top-four span-distinct windows",
        "cxq_deadline": "10 + ceil(40 * one_based_manifest_index / accepted_facet_count)",
        "raw_score_cross_stream_comparison": False,
        "tie_break": ["objective", "relevance", "best_stream_rank", "document_id"],
        "qrels_opened": False,
    }


_FREEZE_SCHEMA_VERSION = "facet-aware-fusion-freeze-v1"


def _canonical_json_bytes(value: object, *, pretty: bool) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row), pretty=False) for row in rows)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_qrels_fields(value: object, *, path: str = "freeze") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if "qrel" in str(key).casefold() and not (
                key == "qrels_opened" and nested is False
            ):
                raise ValueError(f"qrels-derived field is forbidden before freeze: {path}.{key}")
            _reject_qrels_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_qrels_fields(nested, path=f"{path}[{index}]")


def _freeze_topic_inputs(
    frozen_rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[list[Mapping[str, object]], list[FacetStream]]]:
    originals: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    facet_rows: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    metadata: dict[tuple[str, str], tuple[int, bool]] = {}
    for row in frozen_rows:
        topic_id = row.get("topic_id")
        family = row.get("family")
        if not isinstance(topic_id, str) or not topic_id:
            raise ValueError("frozen row topic_id must be non-empty text")
        if family == "original":
            originals[topic_id].append(row)
            continue
        if family != "facet":
            raise ValueError("frozen row family must be original or facet")
        facet_id = row.get("facet_id", row.get("variant"))
        manifest_order = row.get("manifest_order")
        if not isinstance(facet_id, str) or not facet_id:
            raise ValueError("facet rows require facet_id")
        if (
            isinstance(manifest_order, bool)
            or not isinstance(manifest_order, int)
            or manifest_order < 0
        ):
            raise ValueError("facet rows require non-negative manifest_order")
        accepted = _row_bool(row, "accepted")
        key = (topic_id, facet_id)
        observed = (manifest_order, accepted)
        if metadata.setdefault(key, observed) != observed:
            raise ValueError("facet rows disagree on manifest order or gate decision")
        facet_rows[key].append(row)
    result: dict[str, tuple[list[Mapping[str, object]], list[FacetStream]]] = {}
    for topic_id in set(originals).union(topic for topic, _facet in facet_rows):
        if topic_id in PROTECTED_TOPIC_IDS or topic_id in PRIOR_PILOT_TOPIC_IDS:
            raise ValueError(f"forbidden topic {topic_id} in frozen rows")
        if not originals.get(topic_id):
            raise ValueError(f"topic {topic_id} has no original stream")
        facets = [
            FacetStream(
                topic_id=topic_id,
                facet_id=facet_id,
                manifest_order=metadata[(facet_topic, facet_id)][0],
                accepted=metadata[(facet_topic, facet_id)][1],
                ranked_docs=tuple(rows),
            )
            for (facet_topic, facet_id), rows in facet_rows.items()
            if facet_topic == topic_id
        ]
        result[topic_id] = (originals[topic_id], facets)
    return result


def _stable_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    materialized = [dict(row) for row in rows]
    return sorted(
        materialized,
        key=lambda row: _canonical_json_bytes(row, pretty=False),
    )


def create_ranking_freeze(
    output_dir: Path,
    *,
    frozen_rows: Sequence[Mapping[str, object]],
    gates: Sequence[Mapping[str, object]],
    score_rows: Sequence[Mapping[str, object]],
    candidate_provenance: Sequence[Mapping[str, object]],
    input_hashes: Mapping[str, str],
    topic_order: Sequence[str] | None = None,
    limit: int = ORIGINAL_DEPTH,
) -> dict[str, object]:
    """Atomically create the qrels-blind ranking freeze and content hashes."""

    destination = Path(output_dir)
    if destination.exists() or os.path.lexists(destination):
        raise FileExistsError(f"create-only ranking freeze already exists: {destination}")
    if not input_hashes or any(not _is_sha256(value) for value in input_hashes.values()):
        raise ValueError("input_hashes must contain SHA-256 values")
    _reject_qrels_fields(frozen_rows)
    _reject_qrels_fields(gates)
    _reject_qrels_fields(score_rows)
    _reject_qrels_fields(candidate_provenance)
    rankings = build_rankings(frozen_rows, limit=limit)
    inputs = _freeze_topic_inputs(frozen_rows)
    ordered_topics = (
        list(topic_order)
        if topic_order is not None
        else sorted(rankings, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    )
    if len(ordered_topics) != len(set(ordered_topics)) or set(ordered_topics) != set(rankings):
        raise ValueError("topic_order must contain every ranked topic exactly once")
    for topic_id in ordered_topics:
        for arm in ARM_NAMES:
            ranking = rankings[topic_id][arm]
            if len(ranking) != limit or len(set(ranking)) != limit:
                raise ValueError(
                    f"topic {topic_id} arm {arm} must contain exactly {limit} unique documents"
                )

    payloads: dict[str, tuple[bytes, int | None]] = {
        "scores.jsonl": (_canonical_jsonl_bytes(_stable_rows(score_rows)), len(score_rows)),
        "gates.json": (
            _canonical_json_bytes(_stable_rows(gates), pretty=True),
            len(gates),
        ),
        "candidate_provenance.jsonl": (
            _canonical_jsonl_bytes(_stable_rows(candidate_provenance)),
            len(candidate_provenance),
        ),
        "parameters.json": (_canonical_json_bytes(ranking_parameters(), pretty=True), None),
    }
    cxq_provenance: list[dict[str, object]] = []
    for topic_id in ordered_topics:
        original, facets = inputs[topic_id]
        _ranking, rows = constrained_xquad(original, facets, limit=limit)
        cxq_provenance.extend({"topic_id": topic_id, **row} for row in rows)
    payloads["cxq_provenance.jsonl"] = (
        _canonical_jsonl_bytes(cxq_provenance),
        len(cxq_provenance),
    )
    for arm in ARM_NAMES:
        rows = [
            {"topic_id": topic_id, "rank": rank, "docid": document_id}
            for topic_id in ordered_topics
            for rank, document_id in enumerate(rankings[topic_id][arm], start=1)
        ]
        payloads[f"rankings/{arm}.jsonl"] = (
            _canonical_jsonl_bytes(rows),
            len(rows),
        )

    artifact_records = {
        relative: {
            "bytes": len(content),
            "sha256": _sha256(content),
            **({"rows": rows} if rows is not None else {}),
        }
        for relative, (content, rows) in sorted(payloads.items())
    }
    candidate_pool_hashes = {}
    for topic_id in ordered_topics:
        original, facets = inputs[topic_id]
        state = _rank_state(original, facets)
        candidate_pool_hashes[topic_id] = _sha256(
            _canonical_json_bytes(list(state.pool), pretty=False)
        )
    freeze = {
        "schema_version": _FREEZE_SCHEMA_VERSION,
        "complete": True,
        "qrels_opened": False,
        "topic_ids": ordered_topics,
        "arms": list(ARM_NAMES),
        "depth": limit,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "facet_score_aggregation": "top-four span-distinct windows",
        "candidate_pool_sha256": candidate_pool_hashes,
        "input_hashes": dict(sorted(input_hashes.items())),
        "ranker_code_sha256": _sha256(Path(__file__).read_bytes()),
        "artifacts": artifact_records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for relative, (content, _rows) in payloads.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (staging / "freeze.json").write_bytes(
            _canonical_json_bytes(freeze, pretty=True)
        )
        try:
            os.rename(staging, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"create-only ranking freeze already exists: {destination}"
            ) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return freeze


def verify_ranking_freeze(output_dir: Path) -> dict[str, object]:
    """Authenticate every declared freeze artifact without opening qrels."""

    root = Path(output_dir)
    source = (root / "freeze.json").read_bytes()
    try:
        freeze = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("ranking freeze is not valid JSON") from exc
    if not isinstance(freeze, dict) or source != _canonical_json_bytes(freeze, pretty=True):
        raise ValueError("ranking freeze must be canonical JSON")
    if (
        freeze.get("schema_version") != _FREEZE_SCHEMA_VERSION
        or freeze.get("complete") is not True
        or freeze.get("qrels_opened") is not False
        or freeze.get("arms") != list(ARM_NAMES)
        or freeze.get("model") != MODEL_ID
        or freeze.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("ranking freeze header is invalid")
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("ranking freeze artifacts are invalid")
    for relative, record in artifacts.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(record, Mapping)
        ):
            raise ValueError("ranking freeze artifact path is unsafe")
        content = (root / relative).read_bytes()
        if record.get("bytes") != len(content) or record.get("sha256") != _sha256(content):
            raise ValueError(f"ranking freeze artifact hash mismatch: {relative}")
    depth = freeze.get("depth")
    topic_ids = freeze.get("topic_ids")
    if (
        isinstance(depth, bool)
        or not isinstance(depth, int)
        or depth < 1
        or not isinstance(topic_ids, list)
        or any(not isinstance(topic_id, str) for topic_id in topic_ids)
    ):
        raise ValueError("ranking freeze topic/depth fields are invalid")
    for arm in ARM_NAMES:
        path = root / "rankings" / f"{arm}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for topic_id in topic_ids:
            topic_rows = [row for row in rows if row.get("topic_id") == topic_id]
            if [row.get("rank") for row in topic_rows] != list(range(1, depth + 1)):
                raise ValueError(f"ranking freeze ranks are incomplete for {topic_id}/{arm}")
            docids = [row.get("docid") for row in topic_rows]
            if len(set(docids)) != depth or any(not isinstance(docid, str) for docid in docids):
                raise ValueError(f"ranking freeze documents are invalid for {topic_id}/{arm}")
    _reject_qrels_fields(freeze)
    return freeze


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "freeze"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--manifest", type=Path, required=True)
        subparser.add_argument("--retrieval", type=Path, required=True)
        subparser.add_argument("--cache-root", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument(
            "--model-receipt", type=Path, default=_DEFAULT_MODEL_RECEIPT
        )
        subparser.add_argument(
            "--score-cache", type=Path, default=_DEFAULT_SCORE_CACHE_ROOT
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    common = {
        "manifest_path": args.manifest,
        "retrieval_dir": args.retrieval,
        "cache_root": args.cache_root,
        "output_dir": args.output,
        "model_receipt": args.model_receipt,
        "score_cache_root": args.score_cache,
    }
    if args.command == "preflight":
        result = run_rank_preflight(**common)
    else:
        result = run_rank_freeze(**common)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
