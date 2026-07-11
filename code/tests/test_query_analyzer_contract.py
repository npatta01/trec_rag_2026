"""Sentinel tests for the query-planner/BM25 analyzer mismatch.

These tests intentionally document non-equivalence. Replace them with a
conformance test against the frozen Anserini analyzer when that implementation
is available.
"""

from trec_rag.query_planner import ANALYZER_VERSION, analyze_content_terms


def test_planner_analyzer_differs_from_documented_pyserini_default() -> None:
    text = "City buses are running on time."

    # Pyserini's official Analyzer API guide documents the default output as
    # ("citi", "buse", "run", "time") because it uses Porter stemming.
    documented_pyserini_tokens = ("citi", "buse", "run", "time")

    assert ANALYZER_VERSION == "unicode_content_terms_v1"
    assert analyze_content_terms(text) == ("city", "buses", "running", "time")
    assert analyze_content_terms(text) != documented_pyserini_tokens


def test_planner_and_lucene_default_stopword_retention_diverge() -> None:
    # Lucene's English stop set removes "not" but retains "how". The planner
    # does the reverse, so even the number of retained terms can disagree.
    assert analyze_content_terms("not how") == ("not",)
    expected_lucene_default = ("how",)
    assert analyze_content_terms("not how") != expected_lucene_default


def test_planner_unique_budget_does_not_collapse_porter_inflections() -> None:
    assert analyze_content_terms("banks banking") == ("banks", "banking")

    # Anserini's default Porter pipeline maps both inputs to "bank". A budget
    # over unique BM25 terms would therefore count one term, not two.
    expected_unique_lucene_default = ("bank",)
    assert analyze_content_terms("banks banking") != expected_unique_lucene_default
