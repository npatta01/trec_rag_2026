from collections import Counter
import re

from trec_rag.deterministic_sparse_v2 import (
    PLANNER_VERSION,
    RENDERER_VERSION,
    SELECTION_VERSION,
    build_deterministic_sparse_v2_plan,
)
from trec_rag.query_analyzer import AnalyzerFingerprint, AnalyzedQuery, stable_unique


FINGERPRINT = AnalyzerFingerprint(
    contract_version="v2-test",
    implementation="unicode-test-analyzer",
    lucene_version="test",
    analyzer_class="TestAnalyzer",
    tokenizer="unicode-letters",
    filters=("lowercase",),
    stopword_sha256="0" * 64,
    unicode_version="test",
    index_id="synthetic",
)


class Analyzer:
    def __init__(self, aliases=None):
        self.aliases = aliases or {}

    @property
    def fingerprint(self):
        return FINGERPRINT

    def analyze(self, text):
        surfaces = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
        tokens = tuple(
            replacement
            for surface in surfaces
            for replacement in self.aliases.get(surface, (surface,))
        )
        return AnalyzedQuery(tokens, stable_unique(tokens), FINGERPRINT)


def test_two_unit_plan_uses_bounded_prefix_and_cannot_collapse_child_into_original():
    narrative = (
        "alpha beta gamma delta: "
        "explain epsilon zeta eta theta."
    )

    plan = build_deterministic_sparse_v2_plan("synthetic", narrative, Analyzer())

    assert plan.status == "ok"
    assert plan.planner_version == PLANNER_VERSION == "det_sparse_v2"
    assert RENDERER_VERSION == "det_sparse_bounded_parent_renderer_v2"
    assert SELECTION_VERSION == "det_sparse_structural_selection_v2"
    assert plan.context is not None
    assert plan.context.query_text == "alpha beta"
    assert plan.context.source_span.end < plan.lexical_units[0].source_span.end
    assert len(plan.context.unique_analyzed_tokens) == 2
    assert [facet.coverage_unit_ids for facet in plan.facets] == [
        ("u001",),
        ("u002",),
    ]
    assert plan.facets[0].query_text == "alpha beta gamma delta:"
    assert plan.facets[1].query_text == (
        "alpha beta explain epsilon zeta eta theta."
    )
    assert all(facet.query_text != narrative for facet in plan.facets)
    assert all(
        facet.bm25_signature != plan.original_bm25_signature
        for facet in plan.facets
    )
    assert len({facet.bm25_signature for facet in plan.facets}) == 2
    assert [variant.variant_name for variant in plan.query_variants()] == [
        "det_sparse_v2:original",
        "det_sparse_v2:facet:f01",
        "det_sparse_v2:facet:f02",
    ]


def test_context_maximizes_bounded_unique_terms_then_uses_shortest_token_end():
    narrative = (
        "alpha alpha beta gamma delta epsilon zeta eta theta: "
        "explain iota kappa lambda mu."
    )

    plan = build_deterministic_sparse_v2_plan("context", narrative, Analyzer())

    assert plan.status == "ok"
    assert plan.context is not None
    # E=8, so K=min(6, floor(8/2))=4. The shortest prefix reaching four
    # unique terms ends at delta, despite the repeated leading alpha.
    assert plan.context.unique_term_ceiling == 4
    assert plan.context.query_text == "alpha alpha beta gamma delta"
    assert len(plan.context.unique_analyzed_tokens) == 4
    selected = [row for row in plan.context_selection_audit if row.selected]
    assert len(selected) == 1
    assert selected[0].source_span == plan.context.source_span
    assert all(
        row.source_span.text
        == narrative[row.source_span.start : row.source_span.end]
        for row in plan.context_selection_audit
    )
    assert Counter(plan.context.analyzed_tokens) < Counter(
        plan.lexical_units[0].analyzed_tokens
    )


def test_context_allocation_uses_floor_for_odd_parent_unique_term_count():
    plan = build_deterministic_sparse_v2_plan(
        "odd-context",
        "alpha beta gamma delta epsilon: explain iota kappa lambda mu.",
        Analyzer(),
    )

    assert plan.status == "ok"
    assert plan.context is not None
    assert plan.context.parent_unique_term_count == 5
    assert plan.context.unique_term_ceiling == 2
    assert plan.context.query_text == "alpha beta"


def test_parent_is_protected_and_only_adjacent_child_groups_merge_to_four():
    narrative = (
        "parent alpha beta gamma delta epsilon: "
        "one red blue. two green gold. three black white. "
        "four north south. five east west."
    )

    plan = build_deterministic_sparse_v2_plan("merge", narrative, Analyzer())

    assert plan.status == "ok"
    assert len(plan.lexical_units) == 6
    assert len(plan.facets) == 4
    assert plan.facets[0].coverage_unit_ids == ("u001",)
    assert [row.result_unit_ids for row in plan.merge_audit] == [
        ("u002", "u003"),
        ("u004", "u005"),
    ]
    assert all("u001" not in row.result_unit_ids for row in plan.merge_audit)
    covered = [
        unit_id
        for facet in plan.facets
        for unit_id in facet.coverage_unit_ids
    ]
    assert covered == [f"u{index:03d}" for index in range(1, 7)]
    assert len(covered) == len(set(covered))
    for row in plan.merge_audit:
        for span in (
            row.left_source_span,
            row.right_source_span,
            row.combined_source_span,
        ):
            assert span.text == narrative[span.start : span.end]
        assert row.left_source_span.start == row.combined_source_span.start
        assert row.right_source_span.end == row.combined_source_span.end
        assert row.left_source_span.end < row.right_source_span.start
        assert row.combined_source_span.text == narrative[
            row.left_source_span.start : row.right_source_span.end
        ]


def test_success_plan_freezes_all_signature_coverage_and_source_invariants():
    narrative = (
        "system payment privacy security controls: "
        "what reduces fraud attacks? how does recovery improve resilience?"
    )

    first = build_deterministic_sparse_v2_plan("audit", narrative, Analyzer())
    second = build_deterministic_sparse_v2_plan("audit", narrative, Analyzer())

    assert first == second
    assert first.status == "ok"
    assert first.failure is None
    assert 2 <= len(first.facets) <= 4
    original_occurrences = sum(count for _term, count in first.original_bm25_signature)
    signatures = set()
    texts = set()
    context_terms = set(first.context.unique_analyzed_tokens)  # type: ignore[union-attr]
    for facet, audit in zip(first.facets, first.invariant_audit):
        assert facet.query_text not in texts
        assert facet.bm25_signature not in signatures
        texts.add(facet.query_text)
        signatures.add(facet.bm25_signature)
        assert len(facet.unique_analyzed_tokens) >= 3
        assert sum(count for _term, count in facet.bm25_signature) < original_occurrences
        assert audit.exact_text_distinct_from_original
        assert audit.signature_distinct_from_original
        assert audit.strict_occurrence_reduction
        assert audit.strict_original_submultiset
        for unit_id, novel_terms in audit.nonparent_unit_novel_terms:
            assert unit_id != "u001"
            assert novel_terms
            assert set(novel_terms).isdisjoint(context_terms)
        for span in facet.query_source_spans:
            assert span.text == narrative[span.start : span.end]
    assert first.to_dict()["original_query_text"] == narrative


def test_single_unit_and_short_parent_fail_to_exact_original_only():
    single = build_deterministic_sparse_v2_plan(
        "single", "alpha beta gamma delta epsilon", Analyzer()
    )
    short = build_deterministic_sparse_v2_plan(
        "short", "alpha beta gamma: explain delta epsilon zeta.", Analyzer()
    )

    assert single.status == "fallback"
    assert single.failure.code == "insufficient_source_units"  # type: ignore[union-attr]
    assert short.status == "fallback"
    assert short.failure.code == "parent_unique_terms_below_four"  # type: ignore[union-attr]
    for plan in (single, short):
        assert plan.facets == ()
        assert len(plan.query_variants()) == 1
        assert plan.query_variants()[0].query_text == plan.original_query_text


def test_child_without_any_term_absent_context_is_explicit_fallback():
    plan = build_deterministic_sparse_v2_plan(
        "no-novel",
        "alpha beta gamma delta: alpha beta.",
        Analyzer(),
    )

    assert plan.status == "fallback"
    assert plan.failure.code == "child_unit_has_no_term_outside_context"  # type: ignore[union-attr]
    assert plan.facets == ()


def test_unavailable_bounded_context_preserves_rejected_candidate_audit():
    analyzer = Analyzer(
        {
            "alpha": ("one", "two", "three"),
            "beta": ("four",),
        }
    )
    plan = build_deterministic_sparse_v2_plan(
        "no-context",
        "alpha beta: explain epsilon zeta eta.",
        analyzer,
    )

    assert plan.status == "fallback"
    assert plan.failure.code == "bounded_parent_context_unavailable"  # type: ignore[union-attr]
    assert plan.context is None
    assert plan.context_selection_audit
    assert not any(row.selected for row in plan.context_selection_audit)
    assert not any(row.admissible for row in plan.context_selection_audit)


def test_pairwise_bm25_signature_collision_fails_instead_of_silent_dedup():
    analyzer = Analyzer({"epsilon": ("gamma",), "zeta": ("delta",)})
    plan = build_deterministic_sparse_v2_plan(
        "collision",
        "alpha beta gamma delta: epsilon zeta.",
        analyzer,
    )

    assert plan.status == "fallback"
    assert plan.failure.code == "duplicate_facet_signature"  # type: ignore[union-attr]
    assert plan.facets == ()
    assert len(plan.invariant_audit) == 2


def test_analyzer_fingerprint_change_is_failure_not_partial_plan():
    changed = AnalyzerFingerprint(
        contract_version="changed",
        implementation="changed",
        lucene_version="test",
        analyzer_class="TestAnalyzer",
        tokenizer="unicode-letters",
        filters=("lowercase",),
        stopword_sha256="0" * 64,
        unicode_version="test",
        index_id="synthetic",
    )

    class DriftingAnalyzer(Analyzer):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def analyze(self, text):
            result = super().analyze(text)
            self.calls += 1
            if self.calls >= 2:
                return AnalyzedQuery(result.tokens, result.unique_tokens, changed)
            return result

    plan = build_deterministic_sparse_v2_plan(
        "drift",
        "alpha beta gamma delta: explain epsilon zeta eta.",
        DriftingAnalyzer(),
    )

    assert plan.status == "fallback"
    assert plan.failure.code == "analyzer_fingerprint_changed"  # type: ignore[union-attr]
    assert plan.facets == ()
