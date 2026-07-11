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
    assert "file-backed offset parity review requires the full 48-row" in text
    assert "--offset-parity-fixtures path/to/offset-parity-fixtures.json" in text
    assert "exact fixture text contract for every grid cell" in text
    assert "analyzer-zero rows with no\noccurrences" in text
    assert "stable offset fingerprint across every response" in text
    assert "no inference or dispatch authorization" in text
    assert "captured live schema-compiler evidence must bind vLLM `0.24.0`" in text
    assert "Captured runtime evidence must bind served model `gpt-oss-local`" in text
    assert "Live attestation records must be exact before dispatch" in text
    assert "vLLM `0.24.0`, XGrammar `0.2.3`, backend `xgrammar`" in text
    assert "semantic_anchor_live_attestation_review_v1" in text
    assert "semantic_anchor_offset_parity_review_v1" in text
    assert "offset review SHA-256 and offset fingerprint SHA-256" in text
    assert "canonical bundle file" in text
    assert "`bundle_canonical=true`" in text
    assert "semantic_anchor_advisor_dispatch_go_v1" in text
    assert "semantic_anchor_advisor_dispatch_go_review_v1" in text
    assert "manual_runner_invocation_still_required" in text
    assert "--live-attestation-review path/to/live-attestation-review.json" in text
    assert "--advisor-go-receipt path/to/advisor-go.json" in text
    assert "requires the advisor GO receipt to be canonical JSON bytes" in text
    assert "dispatch_authorized=false" in text
    assert "--live-attestation-bundle path/to/live-attestation-bundle.json" in text
    assert "--offset-parity-review path/to/offset-parity-review.json" in text
    assert "strict assistant-content extractor" in text
    assert "source/import audit, offline schema-compatibility" in text
    assert "`request_identity`: 24 cases, first case `synthetic-case-001`" in text
    assert "all 24 per-case response-schema hashes" in text
    assert "canonical per-case\n  response-schema SHA-256s" in text
    assert "executable ledger-prefix validator" in text
    assert "actual zero-dispatch run-directory replay" in text
    assert "offline completed synthetic run-directory replay validated" in text
    assert "missing raw files, terminal counter drift" in text
    assert "premature `gold_opened=true`, sealed response order drift" in text
    assert "--replay-completed-synthetic path/to/run-directory" in text
    assert "--replay-pre-dispatch-no-go path/to/run-directory" in text
    assert "neither dispatches the model,\nopens scorer-only gold" in text
    assert "actual live post-dispatch run still pending" in text
    assert "canonical live-attestation bundle" in text
    assert "advisor\ndispatch-GO binding" in text
    assert "file-backed advisor-GO review CLI" in text
    assert "semantic_anchor_manual_runner_invocation_v1" in text
    assert "semantic_anchor_manual_runner_invocation_review_v1" in text
    assert "--validate-manual-runner-invocation" in text
    assert "--manual-runner-invocation path/to/manual-runner-invocation.json" in text
    assert "does not create a\nnetwork client, invoke a transport, or dispatch the model" in text
    assert "offline replay mutation oracles" in text
    assert "scorer/gold linter validates case order" in text
    assert "accepts only canonical sealed synthetic responses" in text
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
    assert "runtime file-open tracing, live attestation" in text
    assert "evidence validators" in text
    assert "captured-bundle review" in text
    assert "file-backed offset parity review" in text
    assert "offset-parity-to-live-attestation binding" in text
    assert "advisor\ndispatch-GO binding" in text
    assert "file-backed advisor-GO review CLI" in text
    assert "file-backed manual runner invocation review" in text
    assert "runner replay CLI modes" in text
    assert "completed replay mutation coverage" in text
    assert "runner-visible artifact hashes without opening scorer-only gold" in text
    assert "explicit manual runner invocation receipt" in text
    assert "reviewed fake-transport dispatch implementation" in text
    assert "failed-transport terminal sealing" in text
    assert "actual live model invocation remains closed" in text
    assert "canonical sealed scorer input" in text
    assert "ledger-bound scorer-only" in text
    assert "--qualification-review" in text
    assert "--reviewer-receipt path/to/reviewer-receipt.json" in text
    assert "semantic_anchor_untouched_topic_milestone_advisor_approval_v1" in text
    assert "--reviewer-qualification-review path/to/reviewer-qualification-review.json" in text
    assert "--milestone-approval-receipt path/to/milestone-approval-receipt.json" in text
    assert "does not authorize topic access, retrieval,\nreranking, or external cost" in text
    assert "--model-inventory-snapshot" in text
    assert "--model-inventory-attestation path/to/model-inventory-attestation.json" in text
    assert "--schema-compiler-attestation path/to/schema-compiler-attestation.json" in text
    assert "--model-runtime-attestation path/to/model-runtime-attestation.json" in text
    assert "canonical file-backed model-inventory, schema-compiler, and model-runtime" in text
    assert "schema-compiler, and model-runtime\nattestation SHA-256s" in text
    assert "exactly embed those same compiler\nand runtime records" in text
    assert "all five evidence files" in text
    assert "read-only model inventory\ncapture mechanics" in text
    assert "`199 passed`." in text
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
