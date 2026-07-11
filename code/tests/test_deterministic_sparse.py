import hashlib
import json
import re

import pytest

from trec_rag.deterministic_sparse import (
    PLANNER_VERSION,
    PRF_VERSION,
    SPLITTER_VERSION,
    TOKENIZER_VERSION,
    build_deterministic_sparse_plan,
    build_prf_expansion,
    split_deterministic_units,
)
from trec_rag.pipeline_models import RetrievedCandidate
from trec_rag.query_analyzer import (
    AnalyzerFingerprint,
    AnalyzedQuery,
    stable_unique,
)
from trec_rag.query_planner import tokenize_narrative


FINGERPRINT = AnalyzerFingerprint(
    contract_version="test-v1",
    implementation="frozen-test-analyzer",
    lucene_version="test",
    analyzer_class="TestAnalyzer",
    tokenizer="unicode-letters",
    filters=("lowercase",),
    stopword_sha256="0" * 64,
    unicode_version="test",
    index_id="synthetic",
)


class FakeAnalyzer:
    def __init__(self, aliases=None):
        self._aliases = aliases or {}

    @property
    def fingerprint(self):
        return FINGERPRINT

    def analyze(self, text):
        surfaces = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
        tokens = []
        for surface in surfaces:
            replacement = self._aliases.get(surface, (surface,))
            tokens.extend(replacement)
        token_tuple = tuple(tokens)
        return AnalyzedQuery(token_tuple, stable_unique(token_tuple), FINGERPRINT)


def _canonical_sha(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_split_rules_preserve_exact_code_point_spans_and_skip_noun_lists():
    narrative = (
        "Café payment systems overview: Explain privacy, security, and cost, "
        "and how failures differ. What helps recovery?"
    )

    spans = split_deterministic_units(narrative)

    assert [span.text for span in spans] == [
        "Café payment systems overview:",
        "Explain privacy, security, and cost,",
        "and how failures differ.",
        "What helps recovery?",
    ]
    assert all(narrative[span.start : span.end] == span.text for span in spans)
    # Python offsets count Unicode code points, so the first span resolves even
    # though é occupies more than one byte in UTF-8.
    assert spans[0].end == len("Café payment systems overview:")

    coordinator = split_deterministic_units(
        "Explain payment system behavior and how failures differ"
    )
    assert [span.text for span in coordinator] == [
        "Explain payment system behavior",
        "and how failures differ",
    ]


def test_plan_copies_parent_tracks_lexical_coverage_and_is_deterministic():
    narrative = (
        "Café payment systems overview: Explain privacy, security, and cost, "
        "and how failures differ. What helps recovery?"
    )
    analyzer = FakeAnalyzer()

    first = build_deterministic_sparse_plan(
        topic_id="synthetic-1", narrative=narrative, query_analyzer=analyzer
    )
    second = build_deterministic_sparse_plan(
        topic_id="synthetic-1", narrative=narrative, query_analyzer=analyzer
    )

    assert first == second
    assert first.planner_version == PLANNER_VERSION == "det_sparse_v1"
    assert first.splitter_version == SPLITTER_VERSION
    assert first.tokenizer_version == TOKENIZER_VERSION
    assert first.status == "ok"
    assert [facet.facet_id for facet in first.facets] == ["f01", "f02", "f03", "f04"]
    assert first.facets[0].query_component_unit_ids == ("u001",)
    assert all(
        facet.query_component_unit_ids[0] == "u001" for facet in first.facets
    )
    assert first.facets[1].query_text == (
        "Café payment systems overview: Explain privacy, security, and cost,"
    )
    assert sorted(
        unit_id
        for facet in first.facets
        for unit_id in facet.coverage_unit_ids
    ) == ["u001", "u002", "u003", "u004"]
    assert tuple(
        token for unit in first.lexical_units for token in unit.analyzed_tokens
    ) == analyzer.analyze(narrative).tokens
    tape = tokenize_narrative(narrative)
    owned_tape_indices = [
        token_index
        for unit in first.lexical_units
        for token_index in unit.token_tape_indices
    ]
    assert sorted(owned_tape_indices) == list(range(tape.token_count))
    assert len(owned_tape_indices) == len(set(owned_tape_indices))
    assert first.narrative_sha256 == hashlib.sha256(narrative.encode()).hexdigest()
    assert first.token_tape_sha256 == _canonical_sha(
        tape.to_dict()
    )
    assert first.analyzer_token_sha256 == _canonical_sha(
        list(analyzer.analyze(narrative).tokens)
    )
    assert first.analyzer_fingerprint_sha256 == _canonical_sha(FINGERPRINT.to_dict())
    assert [variant.variant_name for variant in first.query_variants()] == [
        "det_sparse_v1:original",
        "det_sparse_v1:facet:f01",
        "det_sparse_v1:facet:f02",
        "det_sparse_v1:facet:f03",
        "det_sparse_v1:facet:f04",
    ]


def test_more_than_four_facets_merge_adjacent_minimum_with_earliest_tie():
    narrative = (
        "alpha beta gamma: delta epsilon zeta. eta theta iota. "
        "kappa lambda mu. nu xi omicron. pi rho sigma."
    )

    plan = build_deterministic_sparse_plan(
        topic_id="synthetic-merge",
        narrative=narrative,
        query_analyzer=FakeAnalyzer(),
    )

    assert plan.status == "ok"
    assert len(plan.lexical_units) == 6
    assert len(plan.facets) == 4
    adjacent_merges = [
        row for row in plan.merge_audit if row.action == "adjacent_min_tokens"
    ]
    assert [row.result_group for row in adjacent_merges] == [
        ("u001", "u002"),
        ("u001", "u002", "u003"),
    ]
    assert sorted(
        unit_id for facet in plan.facets for unit_id in facet.coverage_unit_ids
    ) == [f"u{index:03d}" for index in range(1, 7)]


def test_analyzer_identical_renderings_merge_without_duplicate_query():
    narrative = (
        "one two three: alpha beta gamma. ALPHA BETA GAMMA. delta epsilon zeta."
    )
    plan = build_deterministic_sparse_plan(
        topic_id="synthetic-identical",
        narrative=narrative,
        query_analyzer=FakeAnalyzer(),
    )

    assert plan.status == "ok"
    assert len(plan.facets) == 3
    identical = [
        row for row in plan.merge_audit if row.action == "analyzer_identical"
    ]
    assert len(identical) == 1
    assert identical[0].retained_group == ("u002",)
    assert identical[0].merged_group == ("u003",)
    assert len({facet.analyzed_tokens for facet in plan.facets}) == len(plan.facets)
    merged = next(
        facet for facet in plan.facets if facet.coverage_unit_ids == ("u002", "u003")
    )
    assert merged.query_component_unit_ids == ("u001", "u002")


def test_invalid_short_facet_is_explicit_original_fallback():
    narrative = "  tiny parent: a much longer request unit here.  "
    plan = build_deterministic_sparse_plan(
        topic_id="synthetic-fallback",
        narrative=narrative,
        query_analyzer=FakeAnalyzer(),
    )

    assert plan.status == "fallback"
    assert plan.failure is not None
    assert plan.failure.code == "facet_too_short"
    assert plan.facets == ()
    assert plan.original_query_text == narrative
    variants = plan.query_variants()
    assert len(variants) == 1
    assert variants[0].query_text == narrative
    assert variants[0].variant_name == "det_sparse_v1:original"
    assert variants[0].source_type == "det_sparse_v1_original"


def _retrieval_rows():
    rows = []
    for rank in range(1, 51):
        words = ["general", "documents"]
        if rank in {1, 2, 3, 4, 6, 7}:
            words.append("solar")
        if rank in {1, 2, 5, 6}:
            words.append("battery")
        if rank in {1, 2, 3}:
            words.extend(["running", "running"] if rank == 1 else ["running"])
        if rank in {1, 2}:
            words.extend(["runs", "runs"])
        if rank == 1:
            words.extend(["energy", "an", "123", "compound", "Paris"])
        if rank <= 40:
            words.append("common")
        if rank == 5:
            words.append("rare")
        rows.append(
            RetrievedCandidate(
                topic_id="synthetic-prf",
                variant_name="original",
                retriever_name="bm25",
                query_text="energy systems",
                docid=f"doc-{rank:02d}",
                rank=rank,
                score=float(51 - rank),
                text=" ".join(words),
            )
        )
    return rows


def test_prf_uses_frozen_contrast_filters_surface_tie_and_complete_audit():
    analyzer = FakeAnalyzer(
        aliases={
            "running": ("run",),
            "runs": ("run",),
            "compound": ("com", "pound"),
        }
    )
    raw_hash = hashlib.sha256(b"synthetic raw response").hexdigest()

    result = build_prf_expansion(
        base_query_text="energy systems",
        candidates=_retrieval_rows(),
        query_analyzer=analyzer,
        raw_response_sha256=raw_hash,
    )

    assert result.status == "ok"
    assert result.prf_version == PRF_VERSION
    assert result.selected_terms == ("running", "battery")
    assert result.query_text == "energy systems running battery"
    assert result.raw_response_sha256 == raw_hash
    assert result.base_source_sha256 == hashlib.sha256(b"energy systems").hexdigest()
    assert result.base_query_sha256 == result.base_source_sha256
    assert result.provenance_verified is False
    assert result.rendered_query_sha256 == hashlib.sha256(
        result.query_text.encode()
    ).hexdigest()
    assert result.token_tape_sha256 == _canonical_sha(
        tokenize_narrative("energy systems").to_dict()
    )
    assert result.analyzer_token_sha256 == _canonical_sha(["energy", "systems"])
    assert result.analyzer_fingerprint_sha256 == _canonical_sha(FINGERPRINT.to_dict())
    audit = {row.surface: row for row in result.term_audit}
    assert audit["running"].disposition == "selected"
    assert audit["running"].selection_rank == 1
    assert audit["running"].surface_foreground_document_frequency == 3
    assert audit["runs"].surface_foreground_document_frequency == 2
    assert audit["runs"].foreground_document_frequency == 3
    assert audit["runs"].reasons == ("surface_not_preferred_for_analyzed_form",)
    assert "selection_limit_2" in audit["solar"].reasons
    assert "analyzed_form_present_in_base" in audit["energy"].reasons
    assert "surface_length_outside_3_24" in audit["an"].reasons
    assert "surface_not_unicode_letters" in audit["123"].reasons
    assert "analyzer_token_count_not_one" in audit["compound"].reasons
    assert "no_lowercase_source_occurrence" in audit["paris"].reasons
    assert "top50_df_at_least_40" in audit["common"].reasons
    assert "foreground_df_below_2" in audit["rare"].reasons
    expected_score = pytest.approx(
        __import__("math").log((3 + 0.5) / 6)
        - __import__("math").log((0 + 0.5) / 46)
    )
    assert audit["running"].score == expected_score
    assert len(result.term_audit) == len({row.surface for row in result.term_audit})


def test_prf_scores_union_document_frequency_by_analyzed_form():
    rows = []
    for rank in range(1, 51):
        word = "running" if rank == 1 else "runs" if rank == 2 else "Filler"
        rows.append(
            RetrievedCandidate(
                topic_id="synthetic-union",
                variant_name="original",
                retriever_name="bm25",
                query_text="energy systems",
                docid=f"union-{rank:02d}",
                rank=rank,
                score=float(51 - rank),
                text=word,
            )
        )
    result = build_prf_expansion(
        base_query_text="energy systems",
        candidates=rows,
        query_analyzer=FakeAnalyzer(
            aliases={"running": ("run",), "runs": ("run",)}
        ),
        raw_response_sha256="b" * 64,
    )

    assert result.status == "ok"
    assert result.selected_terms == ("running",)
    audit = {row.surface: row for row in result.term_audit}
    assert audit["running"].surface_foreground_document_frequency == 1
    assert audit["running"].foreground_document_frequency == 2
    assert audit["runs"].foreground_document_frequency == 2
    assert audit["runs"].reasons == (
        "surface_not_preferred_for_analyzed_form",
    )


def test_prf_never_emits_existing_query_and_reports_short_input():
    analyzer = FakeAnalyzer(
        aliases={"running": ("run",), "runs": ("run",)}
    )
    raw_hash = "a" * 64
    result = build_prf_expansion(
        base_query_text="energy systems",
        candidates=_retrieval_rows(),
        query_analyzer=analyzer,
        raw_response_sha256=raw_hash,
        existing_query_texts=("energy systems running",),
    )

    assert result.status == "ok"
    assert result.query_text not in {
        "energy systems",
        "energy systems running",
    }
    assert "duplicate_query" in {
        row.surface: row for row in result.term_audit
    }["running"].reasons

    failure = build_prf_expansion(
        base_query_text="energy systems",
        candidates=_retrieval_rows()[:49],
        query_analyzer=analyzer,
        raw_response_sha256=raw_hash,
    )
    assert failure.status == "failure"
    assert failure.failure is not None
    assert failure.failure.code == "insufficient_top50"
    assert failure.query_text is None


def test_prf_rejects_unverifiable_raw_response_hash():
    with pytest.raises(ValueError, match="raw_response_sha256"):
        build_prf_expansion(
            base_query_text="energy systems",
            candidates=_retrieval_rows(),
            query_analyzer=FakeAnalyzer(),
            raw_response_sha256="not-a-hash",
        )
