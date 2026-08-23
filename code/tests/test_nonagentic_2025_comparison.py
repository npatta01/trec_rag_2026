from trec_rag.nonagentic_2025_comparison import _aggregate_results, _repo_root


def _result(topic_id: str, score: float, *, supported: int, partial: int, missing: int):
    total = supported + partial + missing
    summary = {
        "total": total,
        "supported": supported,
        "partially_supported": partial,
        "missing": missing,
        "strict_coverage": supported / total,
        "partial_credit_coverage": (supported + 0.5 * partial) / total,
    }
    return {
        "topic_id": topic_id,
        "overall_score_0_to_5": score,
        "subtopic_coverage": {"rate": 0.5, "covered": 1, "total": 2, "rows": []},
        "content_subtopic_coverage": {"rate": 1.0, "covered": 2, "total": 2},
        "nugget_extraction": {"all": summary, "vital": summary},
        "nugget_quality": {"generated_claim_precision": summary},
        "inputs": {
            "nonagentic_groups": 2,
            "nonagentic_claim_hints": 3,
            "organizer_nuggets_count": total,
        },
    }


def test_aggregate_results_preserves_topic_rows_and_pools_nuggets():
    aggregate = _aggregate_results(
        [_result("b", 2.0, supported=1, partial=1, missing=0),
         _result("a", 4.0, supported=0, partial=2, missing=0)]
    )

    assert [row["topic_id"] for row in aggregate["topics"]] == ["a", "b"]
    assert aggregate["topic_count"] == 2
    assert aggregate["macro"]["score"] == 3.0
    assert aggregate["pooled"]["nuggets"]["total"] == 4
    assert aggregate["pooled"]["nuggets"]["supported"] == 1
    assert aggregate["pooled"]["nuggets"]["partially_supported"] == 3
    assert aggregate["pooled"]["nuggets"]["partial_credit_coverage"] == 0.625


def test_repo_root_is_found_for_configs_and_local_configs(tmp_path):
    root = tmp_path / "repo"
    (root / "code").mkdir(parents=True)
    (root / "trec-rag-data").mkdir()
    (root / "configs" / "local").mkdir(parents=True)

    assert _repo_root(root / "configs" / "comparison.yaml") == root
    assert _repo_root(root / "configs" / "local" / "comparison.yaml") == root
