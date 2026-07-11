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
    assert "`source_audit`: 5 source files, `status=pass`" in text
    assert "`runtime_file_access`: `status=pass`" in text
    assert "runtime file-open tracing validated" in text
    assert "model-cache and safetensors path denial" in text
    assert "`models--openai--gpt-oss-20b`, model" in text
    assert "create-only runner scaffold writes 24 request-hash reservation records" in text
    assert "`unexpected_trec_rag_modules=[]`" in text
    assert "`vllm_0_24_xgrammar_unsupported_feature_lint`, `status=pass`" in text
    assert "offset health/error/response schemas are bound" in text
    assert "captured live schema-compiler evidence must bind vLLM `0.24.0`" in text
    assert "Captured runtime evidence must bind served model `gpt-oss-local`" in text
    assert "Live attestation records must be exact before dispatch" in text
    assert "vLLM `0.24.0`, XGrammar `0.2.3`, backend `xgrammar`" in text
    assert "semantic_anchor_live_attestation_review_v1" in text
    assert "dispatch_authorized=false" in text
    assert "--live-attestation-bundle path/to/live-attestation-bundle.json" in text
    assert "strict assistant-content extractor" in text
    assert "source/import audit, offline schema-compatibility" in text
    assert "`request_identity`: 24 cases, first case `synthetic-case-001`" in text
    assert "executable ledger-prefix validator" in text
    assert "actual zero-dispatch run-directory replay" in text
    assert "offline completed synthetic run-directory replay validated" in text
    assert "actual live post-dispatch run still pending" in text
    assert "pre-dispatch run-directory replay, offline completed synthetic" in text
    assert "offline completed synthetic\nrun-directory replay" in text
    assert "offline replay mutation oracles" in text
    assert "scorer/gold linter validates case order" in text
    assert "semantic_anchor_scorer_review_v1" in text
    assert "`raw_sealed_pending_scorer` 24-response terminal state" in text
    assert "`gold_opened=false`, validates ledger-prefix evidence" in text
    assert "validates ledger-prefix evidence" in text
    assert "verifies parsed-response raw body hashes" in text
    assert "Reviewer receipts require at least two unanimous reviewers" in text
    assert "`completed_synthetic_go` only" in text
    assert "`completed_qualification_no_go`" in text
    assert "code/tests/test_det_sparse_v4_scorer.py" in text
    assert "classification" in text
    assert "runtime file-open tracing, live attestation evidence validators, captured-bundle" in text
    assert "ledger-bound scorer-only" in text
    assert "`157 passed`." in text
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
        "Live attestation records must be exact before dispatch",
        "Anchor typing/scope must be unambiguous",
        "Ledger integrity must be raw-first and fail-closed",
        "Runner entrypoint must not dispatch before approval",
        "Runner request identities must be fixed before dispatch",
        "Evaluation protocol must avoid qrels leakage",
        "External cost must remain zero",
    )
    for issue in required_issues:
        assert issue in text
