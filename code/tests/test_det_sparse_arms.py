from dataclasses import replace
import hashlib
import json

import pytest

from trec_rag.det_sparse_arms import (
    build_arm_plans,
    build_verified_prf_from_ledger,
    candidates_from_ledger,
    fuse_arm,
    open_frozen_retrieval_ledger,
    projected_unique_request_ceiling,
    retrieval_request_for_query,
)
from trec_rag.det_sparse_config import load_det_sparse_config
from trec_rag.det_sparse_ledger import (
    NormalizedCandidate,
    RawTransportResponse,
    RetrievalLedger,
    RetrievalRequest,
    RetrievalResult,
)
from trec_rag.deterministic_sparse import (
    PRF_VERSION,
    PLANNER_VERSION,
    PrfExpansion,
    PrfFailure,
    PrfTermAudit,
    build_deterministic_sparse_plan,
)
from trec_rag.pipeline_models import RetrievedCandidate
from trec_rag.query_analyzer import AnalyzerFingerprint, AnalyzedQuery, stable_unique


FINGERPRINT = AnalyzerFingerprint(
    contract_version="test",
    implementation="test",
    lucene_version="test",
    analyzer_class="test",
    tokenizer="test",
    filters=(),
    stopword_sha256="0" * 64,
    unicode_version="test",
    index_id="test",
)


class Analyzer:
    @property
    def fingerprint(self):
        return FINGERPRINT

    def analyze(self, text):
        tokens = tuple(
            token.lower().strip(".,?!;:")
            for token in text.split()
            if token.strip(".,?!;:")
        )
        return AnalyzedQuery(tokens, stable_unique(tokens), FINGERPRINT)


def _plan():
    return build_deterministic_sparse_plan(
        topic_id="synthetic",
        narrative="payment system overview: explain privacy controls. what reduces failures?",
        query_analyzer=Analyzer(),
    )


def _expansion(base, term="evidence", status="ok"):
    query_text = f"{base.query_text} {term}" if status == "ok" else None
    base_sha = hashlib.sha256(base.query_text.encode()).hexdigest()
    rendered_sha = hashlib.sha256(query_text.encode()).hexdigest() if query_text else None
    analyzer_sha = hashlib.sha256(
        json.dumps(
            FINGERPRINT.to_dict(), separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    audit = (
        PrfTermAudit(
            surface=term,
            analyzed_tokens=(term,),
            surface_foreground_document_frequency=2,
            surface_top50_document_frequency=2,
            foreground_document_frequency=2,
            background_document_frequency=0,
            top50_document_frequency=2,
            foreground_occurrence_frequency=2,
            top50_occurrence_frequency=2,
            score=1.0,
            disposition="selected",
            reasons=(),
            selection_rank=1,
        ),
    ) if status == "ok" else ()
    return PrfExpansion(
        planner_version=PLANNER_VERSION,
        prf_version=PRF_VERSION,
        tokenizer_version="narrative_token_tape_v1",
        status=status,
        topic_id=base.topic_id,
        base_query_text=base.query_text,
        query_text=query_text,
        selected_terms=(term,) if status == "ok" else (),
        base_source_sha256=base_sha,
        base_query_sha256=base_sha,
        rendered_query_sha256=rendered_sha,
        token_tape_sha256="d" * 64,
        analyzer_token_sha256="e" * 64,
        analyzer_fingerprint_sha256=analyzer_sha,
        raw_response_sha256="1" * 64,
        retrieval_request_key="2" * 64,
        retrieval_candidates_sha256="3" * 64,
        retriever_version="pyserini_remote_raw_first_v1",
        provenance_verified=True,
        term_audit=audit,
        failure=None,
    )


def test_four_arms_have_frozen_membership_weights_and_original_always_present():
    plan = _plan()
    assert plan.status == "ok"
    variants = plan.query_variants()
    expansions = {variants[0].variant_name: _expansion(variants[0], term="alpha")}
    expansions.update(
        {
            variant.variant_name: _expansion(variant, term=term)
            for variant, term in zip(variants[2:], ("beta", "gamma", "delta"))
        }
    )

    arms = build_arm_plans(plan, expansions=expansions)

    assert [arm.arm for arm in arms] == ["O", "F", "E", "FE"]
    assert all(arm.streams[0].query.variant_name == "det_sparse_v1:original" for arm in arms)
    assert [stream.weight for stream in arms[0].streams] == [1.0]
    facet_count = len(plan.facets)
    assert [stream.weight for stream in arms[1].streams] == [
        0.5,
        *([0.5 / facet_count] * facet_count),
    ]
    assert [stream.weight for stream in arms[2].streams] == [0.5, 0.5]
    assert [stream.weight for stream in arms[3].streams] == [
        0.5,
        *([0.25 / facet_count] * facet_count),
        0.125,
        *([0.125 / (facet_count - 1)] * (facet_count - 1)),
    ]
    assert projected_unique_request_ceiling(plan) <= 9


def test_noop_expansion_emits_no_duplicate_request_and_does_not_reassign_weight():
    plan = _plan()
    variants = plan.query_variants()
    expansions = {
        variants[0].variant_name: _expansion(variants[0], status="no_expansion")
    }
    expansions.update(
        {
            variant.variant_name: _expansion(variant, status="no_expansion")
            for variant in variants[2:]
        }
    )

    arms = build_arm_plans(plan, expansions=expansions)

    expansion_arm = arms[2]
    combined_arm = arms[3]
    assert expansion_arm.status == "ok"
    assert len(expansion_arm.streams) == 1
    assert expansion_arm.streams[0].weight == 0.5
    assert len(combined_arm.streams) == 1 + len(plan.facets)
    assert sum(stream.weight for stream in combined_arm.streams) == 0.75


def test_prf_failure_falls_back_whole_affected_arm_to_original():
    plan = _plan()
    variants = plan.query_variants()
    expansions = {variants[0].variant_name: _expansion(variants[0])}
    expansions.update(
        {variant.variant_name: _expansion(variant) for variant in variants[2:]}
    )
    failed = expansions[variants[2].variant_name]
    expansions[variants[2].variant_name] = replace(
        failed,
        status="failure",
        query_text=None,
        selected_terms=(),
        rendered_query_sha256=None,
        failure=PrfFailure("broken", "synthetic failure"),
    )

    arms = build_arm_plans(plan, expansions=expansions)

    assert arms[1].status == "ok"
    assert arms[2].status == "ok"
    assert arms[3].status == "fallback"
    assert [stream.query.variant_name for stream in arms[3].streams] == [
        "det_sparse_v1:original"
    ]
    assert arms[3].streams[0].weight == 1.0


def test_fallback_plan_makes_every_transformed_arm_exact_original():
    plan = build_deterministic_sparse_plan(
        topic_id="synthetic",
        narrative="tiny: a longer request with enough words.",
        query_analyzer=Analyzer(),
    )
    assert plan.status == "fallback"

    arms = build_arm_plans(plan)

    assert arms[0].status == "ok"
    assert all(arm.status == "fallback" for arm in arms[1:])
    assert all(len(arm.streams) == 1 for arm in arms)
    assert all(arm.streams[0].query.query_text == plan.original_query_text for arm in arms)


def test_fuse_arm_uses_only_members_and_frozen_weights():
    plan = _plan()
    variants = plan.query_variants()
    expansions = {variants[0].variant_name: _expansion(variants[0], term="alpha")}
    expansions.update(
        {
            variant.variant_name: _expansion(variant, term=term)
            for variant, term in zip(variants[2:], ("beta", "gamma", "delta"))
        }
    )
    arm = build_arm_plans(plan, expansions=expansions)[1]
    candidates = []
    for stream in arm.streams:
        candidates.extend(
            [
                RetrievedCandidate(
                    "synthetic",
                    stream.query.variant_name,
                    "bm25",
                    stream.query.query_text,
                    "shared",
                    2,
                    7.0,
                    "shared text",
                ),
                RetrievedCandidate(
                    "synthetic",
                    stream.query.variant_name,
                    "bm25",
                    stream.query.query_text,
                    f"only-{stream.query.variant_name}",
                    1,
                    8.0,
                    "unique text",
                ),
            ]
        )
    candidates.append(
        RetrievedCandidate(
            "other-topic", "unrelated", "bm25", "x", "noise", 1, 99.0, "noise"
        )
    )

    ranked = fuse_arm(arm, candidates, retriever_name="bm25")

    assert ranked[0].docid == "shared"
    assert ranked[0].rank == 1
    assert len(ranked[0].provenance) == len(arm.streams)
    with pytest.raises(ValueError, match="frozen at 60"):
        fuse_arm(arm, candidates, retriever_name="bm25", k=30)


def test_verified_ledger_rows_bridge_without_losing_query_identity():
    request = RetrievalRequest.from_query(
        topic_id="synthetic",
        variant_name="det_sparse_v1:original",
        query_text="exact query",
        index_url="https://example.test/search",
        index_id="index",
        hits=100,
        analyzer_fingerprint_sha256="a" * 64,
    )
    result = RetrievalResult(
        request_key=request.identity.request_key,
        candidates=(NormalizedCandidate("doc", 1, 4.0, "text"),),
        response_sha256="b" * 64,
        candidates_sha256="c" * 64,
        cache_hit=True,
        external_calls=0,
    )

    rows = candidates_from_ledger(request, result, retriever_name="bm25")

    assert rows == (
        RetrievedCandidate(
            "synthetic",
            "det_sparse_v1:original",
            "bm25",
            "exact query",
            "doc",
            1,
            4.0,
            "text",
        ),
    )


def test_config_binding_cannot_raise_cost_or_change_request_identity(tmp_path):
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    config = load_det_sparse_config(repo_root / "configs" / "det_sparse_v1.yaml")
    ledger = open_frozen_retrieval_ledger(config, run_dir=tmp_path / "ledger")
    query = _plan().query_variants()[0]
    request = retrieval_request_for_query(
        config,
        query,
        index_url="https://example.test/v1/climbmix-400b/search",
    )

    assert ledger.max_calls == 36
    assert ledger.min_results == 50
    assert request.identity.hits == 100
    assert request.identity.index_id == "climbmix-400b"
    assert request.identity.retriever_version == "pyserini_remote_raw_first_v1"
    assert request.identity.analyzer_fingerprint_sha256 == (
        config.analyzer.expected_fingerprint_sha256
    )


def test_parent_facet_expansion_is_rejected_before_arm_materialization():
    plan = _plan()
    parent = plan.query_variants()[1]

    with pytest.raises(ValueError, match="forbids.*parent"):
        build_arm_plans(
            plan,
            expansions={parent.variant_name: _expansion(parent)},
        )


def test_prf_provenance_is_bound_to_verified_ledger_hashes(tmp_path):
    plan = _plan()
    query = plan.query_variants()[0]
    request = RetrievalRequest.from_query(
        topic_id=query.topic_id,
        variant_name=query.variant_name,
        query_text=query.query_text,
        index_url="https://example.test/search",
        index_id="index",
        hits=100,
        analyzer_fingerprint_sha256=plan.analyzer_fingerprint_sha256,
    )
    body = json.dumps(
        {
            "candidates": [
                {
                    "docid": f"doc-{rank}",
                    "rank": rank,
                    "score": 101 - rank,
                    "doc": {"contents": "general documents"},
                }
                for rank in range(1, 51)
            ]
        }
    ).encode()
    ledger = RetrievalLedger(tmp_path / "ledger")
    result = ledger.retrieve(
        request,
        lambda _request: RawTransportResponse(200, {}, body, 0.1),
    )

    expansion = build_verified_prf_from_ledger(
        ledger,
        request,
        query_analyzer=Analyzer(),
    )

    assert expansion.status == "no_expansion"
    assert expansion.provenance_verified is True
    assert expansion.retrieval_request_key == request.identity.request_key
    assert expansion.raw_response_sha256 == result.response_sha256
    assert expansion.retrieval_candidates_sha256 == result.candidates_sha256
