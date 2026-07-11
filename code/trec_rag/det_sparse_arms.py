"""Four-arm membership and weighted-RRF execution for ``det_sparse_v1``."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, Sequence

from trec_rag.det_sparse_config import (
    ARM_NAMES,
    MAX_EXPANDED_NON_PARENT_FACETS,
    DetSparseConfig,
)
from trec_rag.det_sparse_ledger import (
    RETRIEVER_VERSION,
    RetrievalLedger,
    RetrievalRequest,
    RetrievalResult,
)
from trec_rag.deterministic_sparse import (
    PLANNER_VERSION,
    PRF_VERSION,
    TOKENIZER_VERSION,
    DeterministicSparsePlan,
    PrfExpansion,
    build_prf_expansion,
)
from trec_rag.pipeline_models import QueryVariant, RankedCandidate, RetrievedCandidate
from trec_rag.query_analyzer import QueryAnalyzer
from trec_rag.ranking import reciprocal_rank_fusion


@dataclass(frozen=True)
class StreamMembership:
    role: str
    requested_weight: float
    selected_for_unique_stream: bool


@dataclass(frozen=True)
class ArmStream:
    query: QueryVariant
    weight: float
    memberships: tuple[StreamMembership, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": asdict(self.query),
            "weight": self.weight,
            "memberships": [asdict(row) for row in self.memberships],
        }


@dataclass(frozen=True)
class ArmFailure:
    code: str
    message: str


@dataclass(frozen=True)
class ArmPlan:
    topic_id: str
    arm: Literal["O", "F", "E", "FE"]
    status: Literal["ok", "fallback"]
    streams: tuple[ArmStream, ...]
    failure: ArmFailure | None

    def to_dict(self) -> dict[str, object]:
        return {
            "topic_id": self.topic_id,
            "arm": self.arm,
            "status": self.status,
            "streams": [stream.to_dict() for stream in self.streams],
            "failure": asdict(self.failure) if self.failure is not None else None,
        }

    def stream_weights(self, retriever_name: str) -> dict[tuple[str, str], float]:
        if not retriever_name:
            raise ValueError("retriever_name must be non-empty")
        return {
            (stream.query.variant_name, retriever_name): stream.weight
            for stream in self.streams
        }


def _deduplicate_streams(
    logical: Sequence[tuple[str, QueryVariant, float]],
) -> tuple[ArmStream, ...]:
    """Keep one exact query stream and the largest colliding frozen weight."""

    by_text: dict[str, list[tuple[str, QueryVariant, float]]] = {}
    order: list[str] = []
    for role, query, weight in logical:
        if weight <= 0:
            raise ValueError("logical stream weights must be positive")
        if query.query_text not in by_text:
            by_text[query.query_text] = []
            order.append(query.query_text)
        by_text[query.query_text].append((role, query, weight))

    streams: list[ArmStream] = []
    for query_text in order:
        rows = by_text[query_text]
        selected_index = min(
            range(len(rows)),
            key=lambda index: (-rows[index][2], index),
        )
        selected = rows[selected_index]
        streams.append(
            ArmStream(
                query=selected[1],
                weight=selected[2],
                memberships=tuple(
                    StreamMembership(
                        role=role,
                        requested_weight=weight,
                        selected_for_unique_stream=index == selected_index,
                    )
                    for index, (role, _query, weight) in enumerate(rows)
                ),
            )
        )
    return tuple(streams)


def _original_only(
    original: QueryVariant,
    *,
    arm: Literal["O", "F", "E", "FE"],
    status: Literal["ok", "fallback"],
    failure: ArmFailure | None,
) -> ArmPlan:
    return ArmPlan(
        topic_id=original.topic_id,
        arm=arm,
        status=status,
        streams=_deduplicate_streams((("original", original, 1.0),)),
        failure=failure,
    )


def _expansion_query(
    base: QueryVariant,
    expansion: PrfExpansion | None,
    *,
    expected_analyzer_fingerprint_sha256: str | None,
) -> tuple[QueryVariant | None, ArmFailure | None]:
    if expansion is None:
        return None, ArmFailure(
            code="missing_prf_artifact",
            message=f"missing PRF artifact for {base.variant_name}",
        )
    if expansion.planner_version != PLANNER_VERSION or expansion.prf_version != PRF_VERSION:
        return None, ArmFailure(
            code="prf_version_mismatch",
            message=f"PRF version differs for {base.variant_name}",
        )
    if expansion.tokenizer_version != TOKENIZER_VERSION:
        return None, ArmFailure(
            code="prf_tokenizer_mismatch",
            message=f"PRF tokenizer differs for {base.variant_name}",
        )
    if not expansion.provenance_verified:
        return None, ArmFailure(
            code="unverified_prf_provenance",
            message=f"PRF is not bound to verified retrieval for {base.variant_name}",
        )
    if expansion.topic_id != base.topic_id:
        return None, ArmFailure(
            code="prf_topic_mismatch",
            message=f"PRF topic differs for {base.variant_name}",
        )
    if expansion.base_query_text != base.query_text:
        return None, ArmFailure(
            code="prf_base_query_mismatch",
            message=f"PRF base query differs for {base.variant_name}",
        )
    base_sha = hashlib.sha256(base.query_text.encode("utf-8")).hexdigest()
    if expansion.base_query_sha256 != base_sha or expansion.base_source_sha256 != base_sha:
        return None, ArmFailure(
            code="prf_base_hash_mismatch",
            message=f"PRF base hash differs for {base.variant_name}",
        )
    if (
        expected_analyzer_fingerprint_sha256 is None
        or expansion.analyzer_fingerprint_sha256
        != expected_analyzer_fingerprint_sha256
    ):
        return None, ArmFailure(
            code="prf_analyzer_mismatch",
            message=f"PRF analyzer differs for {base.variant_name}",
        )
    sha_values = (
        expansion.token_tape_sha256,
        expansion.analyzer_token_sha256,
        expansion.raw_response_sha256,
        expansion.retrieval_request_key,
        expansion.retrieval_candidates_sha256,
    )
    if any(not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None for value in sha_values):
        return None, ArmFailure(
            code="prf_provenance_hash_missing",
            message=f"PRF provenance hash is missing for {base.variant_name}",
        )
    if expansion.retriever_version != RETRIEVER_VERSION:
        return None, ArmFailure(
            code="prf_retriever_mismatch",
            message=f"PRF retriever version differs for {base.variant_name}",
        )
    if expansion.status == "failure":
        if expansion.query_text is not None or expansion.selected_terms:
            return None, ArmFailure(
                code="invalid_prf_failure",
                message=f"failed PRF contains a query for {base.variant_name}",
            )
        detail = expansion.failure.message if expansion.failure is not None else "unknown"
        return None, ArmFailure(
            code="prf_failure",
            message=f"PRF failed for {base.variant_name}: {detail}",
        )
    if expansion.status == "no_expansion":
        if expansion.query_text is not None or expansion.selected_terms or expansion.failure is not None:
            return None, ArmFailure(
                code="invalid_prf_noop",
                message=f"no-op PRF contains inconsistent fields for {base.variant_name}",
            )
        return None, None
    if expansion.failure is not None or not 1 <= len(expansion.selected_terms) <= 2:
        return None, ArmFailure(
            code="invalid_prf_selection",
            message=f"PRF selection is inconsistent for {base.variant_name}",
        )
    if len(set(expansion.selected_terms)) != len(expansion.selected_terms) or any(
        term != term.lower()
        or not 3 <= len(term) <= 24
        or not all(character.isalpha() for character in term)
        for term in expansion.selected_terms
    ):
        return None, ArmFailure(
            code="invalid_prf_term",
            message=f"PRF contains an invalid term for {base.variant_name}",
        )
    expected_query = " ".join((base.query_text.strip(), *expansion.selected_terms))
    if expansion.query_text != expected_query or expansion.rendered_query_sha256 != hashlib.sha256(
        expected_query.encode("utf-8")
    ).hexdigest():
        return None, ArmFailure(
            code="prf_render_hash_mismatch",
            message=f"PRF rendered query differs for {base.variant_name}",
        )
    selected_rows = [
        row for row in expansion.term_audit if row.disposition == "selected"
    ]
    if any(not isinstance(row.selection_rank, int) for row in selected_rows):
        return None, ArmFailure(
            code="prf_audit_mismatch",
            message=f"PRF audit lacks selection ranks for {base.variant_name}",
        )
    selected_audit = sorted(
        (int(row.selection_rank), row.surface) for row in selected_rows
    )
    if selected_audit != list(enumerate(expansion.selected_terms, start=1)):
        return None, ArmFailure(
            code="prf_audit_mismatch",
            message=f"PRF audit differs from selected terms for {base.variant_name}",
        )
    query = expansion.query_variant(variant_name=f"{base.variant_name}:prf")
    if query is None or query.query_text == base.query_text:
        return None, ArmFailure(
            code="invalid_prf_query",
            message=f"PRF did not produce a distinct query for {base.variant_name}",
        )
    return query, None


def build_arm_plans(
    plan: DeterministicSparsePlan,
    *,
    expansions: Mapping[str, PrfExpansion] | None = None,
) -> tuple[ArmPlan, ...]:
    """Materialize O/F/E/FE with exact failure and no-op semantics."""

    variants = plan.query_variants()
    if not variants or variants[0].variant_name != f"{PLANNER_VERSION}:original":
        raise ValueError("deterministic sparse plan lacks its permanent original stream")
    original = variants[0]
    facets = variants[1:]

    original_arm = _original_only(
        original,
        arm="O",
        status="ok",
        failure=None,
    )
    if plan.status == "fallback":
        failure = ArmFailure(
            code="facet_plan_fallback",
            message=(
                plan.failure.message
                if plan.failure is not None
                else "deterministic facet plan failed"
            ),
        )
        return (
            original_arm,
            *(
                _original_only(
                    original,
                    arm=arm,
                    status="fallback",
                    failure=failure,
                )
                for arm in ("F", "E", "FE")
            ),
        )
    if not facets:
        raise ValueError("successful deterministic sparse plan has no facets")

    facet_count = len(facets)
    facet_logical = [
        (f"facet:{index:02d}", facet, 0.5 / facet_count)
        for index, facet in enumerate(facets, start=1)
    ]
    facet_arm = ArmPlan(
        topic_id=original.topic_id,
        arm="F",
        status="ok",
        streams=_deduplicate_streams(
            (("original", original, 0.5), *facet_logical)
        ),
        failure=None,
    )

    expansions = expansions or {}
    parent_facet = facets[0]
    if parent_facet.variant_name in expansions:
        raise ValueError("det_sparse_v1 forbids PRF expansion of parent facet f01")
    original_expansion, original_failure = _expansion_query(
        original,
        expansions.get(original.variant_name),
        expected_analyzer_fingerprint_sha256=plan.analyzer_fingerprint_sha256,
    )
    if original_failure is not None:
        expansion_arm = _original_only(
            original,
            arm="E",
            status="fallback",
            failure=original_failure,
        )
    else:
        expansion_logical: list[tuple[str, QueryVariant, float]] = [
            ("original", original, 0.5)
        ]
        if original_expansion is not None:
            expansion_logical.append(("expanded_original", original_expansion, 0.5))
        expansion_arm = ArmPlan(
            topic_id=original.topic_id,
            arm="E",
            status="ok",
            streams=_deduplicate_streams(expansion_logical),
            failure=None,
        )

    expandable_facets = facets[1 : 1 + MAX_EXPANDED_NON_PARENT_FACETS]
    expanded_facets: list[QueryVariant | None] = []
    combined_failure = original_failure
    for facet in expandable_facets:
        expanded, failure = _expansion_query(
            facet,
            expansions.get(facet.variant_name),
            expected_analyzer_fingerprint_sha256=plan.analyzer_fingerprint_sha256,
        )
        expanded_facets.append(expanded)
        if combined_failure is None and failure is not None:
            combined_failure = failure
    if combined_failure is not None:
        combined_arm = _original_only(
            original,
            arm="FE",
            status="fallback",
            failure=combined_failure,
        )
    else:
        combined_logical: list[tuple[str, QueryVariant, float]] = [
            ("original", original, 0.5)
        ]
        combined_logical.extend(
            (f"facet:{index:02d}", facet, 0.25 / facet_count)
            for index, facet in enumerate(facets, start=1)
        )
        if original_expansion is not None:
            combined_logical.append(
                ("expanded_original", original_expansion, 0.125)
            )
        expanded_family_count = len(expandable_facets)
        if expanded_family_count:
            combined_logical.extend(
                (
                    f"expanded_facet:{index + 1:02d}",
                    expanded,
                    0.125 / expanded_family_count,
                )
                for index, expanded in enumerate(expanded_facets, start=1)
                if expanded is not None
            )
        combined_arm = ArmPlan(
            topic_id=original.topic_id,
            arm="FE",
            status="ok",
            streams=_deduplicate_streams(combined_logical),
            failure=None,
        )

    result = (original_arm, facet_arm, expansion_arm, combined_arm)
    if tuple(arm.arm for arm in result) != ARM_NAMES:
        raise AssertionError("internal arm order drift")
    return result


def projected_unique_request_ceiling(plan: DeterministicSparsePlan) -> int:
    """Derive the per-topic ceiling from actual unique base queries and rules."""

    variants = plan.query_variants()
    base_unique = len({query.query_text for query in variants})
    if plan.status == "fallback":
        return base_unique
    facets = variants[1:]
    expanded_non_parent = min(
        max(len(facets) - 1, 0),
        MAX_EXPANDED_NON_PARENT_FACETS,
    )
    # Expanded original plus eligible non-parent facets are each at most one
    # new exact query; no-op/colliding PRF can only reduce this upper bound.
    return base_unique + 1 + expanded_non_parent


def fuse_arm(
    arm: ArmPlan,
    candidates: Sequence[RetrievedCandidate],
    *,
    retriever_name: str,
    k: int = 60,
    limit: int = 100,
) -> list[RankedCandidate]:
    """Filter exact arm streams and execute the frozen weighted RRF."""

    if k != 60:
        raise ValueError("det_sparse_v1 RRF k is frozen at 60")
    if limit != 100:
        raise ValueError("det_sparse_v1 fused depth is frozen at 100")
    expected_names = {stream.query.variant_name for stream in arm.streams}
    selected = [
        candidate
        for candidate in candidates
        if candidate.topic_id == arm.topic_id
        and candidate.retriever_name == retriever_name
        and candidate.variant_name in expected_names
    ]
    observed_names = {candidate.variant_name for candidate in selected}
    missing = sorted(expected_names - observed_names)
    if missing:
        raise ValueError(f"arm {arm.arm} lacks retrieval candidates for: {missing}")
    if any(
        candidate.query_text
        != next(
            stream.query.query_text
            for stream in arm.streams
            if stream.query.variant_name == candidate.variant_name
        )
        for candidate in selected
    ):
        raise ValueError(f"arm {arm.arm} candidate query identity mismatch")
    return reciprocal_rank_fusion(
        list(selected),
        k=k,
        stream_weights=arm.stream_weights(retriever_name),
        limit=limit,
    )


def candidates_from_ledger(
    request: RetrievalRequest,
    result: RetrievalResult,
    *,
    retriever_name: str,
) -> tuple[RetrievedCandidate, ...]:
    """Bridge verified ledger rows into the shared pipeline record contract."""

    if result.request_key != request.identity.request_key:
        raise ValueError("retrieval result/request key mismatch")
    if not retriever_name:
        raise ValueError("retriever_name must be non-empty")
    return tuple(
        RetrievedCandidate(
            topic_id=request.identity.topic_id,
            variant_name=request.identity.variant_name,
            retriever_name=retriever_name,
            query_text=request.query_text,
            docid=candidate.docid,
            rank=candidate.rank,
            score=candidate.score,
            text=candidate.text,
        )
        for candidate in result.candidates
    )


def build_verified_prf_from_ledger(
    ledger: RetrievalLedger,
    request: RetrievalRequest,
    *,
    query_analyzer: QueryAnalyzer,
    existing_query_texts: Sequence[str] = (),
) -> PrfExpansion:
    """Build PRF only from a fully verified raw/candidate ledger result."""

    result = ledger.load_verified_result(request)
    rows = candidates_from_ledger(
        request,
        result,
        retriever_name=request.identity.retriever_version,
    )
    expansion = build_prf_expansion(
        base_query_text=request.query_text,
        candidates=rows,
        query_analyzer=query_analyzer,
        raw_response_sha256=result.response_sha256,
        existing_query_texts=existing_query_texts,
    )
    return replace(
        expansion,
        retrieval_request_key=request.identity.request_key,
        retrieval_candidates_sha256=result.candidates_sha256,
        retriever_version=request.identity.retriever_version,
        provenance_verified=True,
    )


def open_frozen_retrieval_ledger(
    config: DetSparseConfig,
    *,
    run_dir: str | os.PathLike[str],
) -> RetrievalLedger:
    """Bind the generic ledger to every frozen cost/cache admission value."""

    return RetrievalLedger(
        Path(run_dir),
        shared_cache_dir=None,
        max_calls=config.cost.max_external_requests,
        max_calls_per_topic=config.cost.max_unique_requests_per_topic,
        min_results=config.retrieval.min_results,
        required_text_results=50,
    )


def retrieval_request_for_query(
    config: DetSparseConfig,
    query: QueryVariant,
    *,
    index_url: str,
) -> RetrievalRequest:
    """Create an exact ledger identity from the frozen experiment config."""

    return RetrievalRequest.from_query(
        topic_id=query.topic_id,
        variant_name=query.variant_name,
        query_text=query.query_text,
        retriever_version=config.retrieval.type,
        index_url=index_url,
        index_id=config.retrieval.index,
        hits=config.retrieval.hits,
        analyzer_fingerprint_sha256=config.analyzer.expected_fingerprint_sha256,
    )
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
