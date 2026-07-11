from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "docs" / "superpowers" / "det_sparse_v4_advisor_packet.md"


def test_v4_advisor_packet_preserves_no_inference_gate_and_review_questions():
    text = PACKET.read_text(encoding="utf-8")

    assert "Status: offline review packet" in text
    assert "does not authorize model" in text
    assert "denied_topic_count`: 22" in text
    assert "inference_authorized`: `false`" in text
    assert "external_cost_authorized`: `false`" in text
    assert "next_gate`: `advisor_review_before_live_attestation_or_model_inference`" in text
    assert "`source_audit`: 4 source files, `status=pass`" in text
    assert "create-only runner scaffold writes 24 request-hash reservation records" in text
    assert "`unexpected_trec_rag_modules=[]`" in text
    assert "`vllm_0_24_xgrammar_unsupported_feature_lint`, `status=pass`" in text
    assert "offset health/error/response schemas are bound" in text
    assert "strict assistant-content extractor" in text
    assert "source/import audit, offline schema-compatibility" in text
    assert "`request_identity`: 24 cases, first case `synthetic-case-001`" in text
    assert "executable ledger-prefix validator" in text
    assert "offline replay mutation oracles" in text
    assert "scorer/gold linter validates case order" in text
    assert "classification" in text
    assert "offset health/error schema checks:" in text
    assert "`138 passed`." in text
    assert "Advisor questions" in text
    assert "What is not yet proven" in text


def test_v4_advisor_packet_maps_core_issues_to_evidence():
    text = PACKET.read_text(encoding="utf-8")

    required_issues = (
        "V2.1 must remain immutable",
        "Consumed development topics must stay closed",
        "Known-five topics must not be touched",
        "Source/import closure must be fail-closed",
        "Schema/runtime compatibility must be explicit",
        "Local model inventory must be exact and cost-free",
        "Anchor typing/scope must be unambiguous",
        "Ledger integrity must be raw-first and fail-closed",
        "Runner entrypoint must not dispatch before approval",
        "Runner request identities must be fixed before dispatch",
        "Evaluation protocol must avoid qrels leakage",
        "External cost must remain zero",
    )
    for issue in required_issues:
        assert issue in text
