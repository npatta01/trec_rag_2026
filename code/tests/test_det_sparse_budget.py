from __future__ import annotations

import pytest

from trec_rag.det_sparse_budget import GlobalExternalBudget
from trec_rag.det_sparse_config import (
    ANALYZER_FINGERPRINT_SHA256,
    RETRIEVAL_ENDPOINT_URL,
)
from trec_rag.det_sparse_ledger import RetrievalRequest


def _request(query: str, *, topic_id: str = "200") -> RetrievalRequest:
    return RetrievalRequest.from_query(
        topic_id=topic_id,
        variant_name=f"variant:{query}",
        query_text=query,
        index_url=RETRIEVAL_ENDPOINT_URL,
        index_id="climbmix-400b",
        hits=100,
        analyzer_fingerprint_sha256=ANALYZER_FINGERPRINT_SHA256,
    )


def test_global_budget_refuses_replay_across_independent_instances(tmp_path):
    first = GlobalExternalBudget(tmp_path / "global")
    request = _request("first query")
    receipt = first.reserve(request)

    second = GlobalExternalBudget(tmp_path / "global")
    second.verify_receipts([receipt], {request.identity.request_key: request})
    with pytest.raises(ValueError, match="replay"):
        second.reserve(request)


def test_global_budget_enforces_nine_per_topic_before_transport(tmp_path):
    budget = GlobalExternalBudget(tmp_path / "global")
    for ordinal in range(1, 10):
        budget.reserve(_request(f"query {ordinal}"))

    with pytest.raises(ValueError, match="per-topic"):
        budget.reserve(_request("query 10"))


def test_global_budget_manifest_drift_fails_closed(tmp_path):
    root = tmp_path / "global"
    GlobalExternalBudget(root)
    (root / "budget.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="policy"):
        GlobalExternalBudget(root)
