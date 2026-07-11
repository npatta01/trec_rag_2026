from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import replace

import pytest

import trec_rag.deterministic_sparse_v3 as planner_module
from trec_rag.deterministic_sparse_v3 import (
    ANCHOR_SELECTOR_VERSION,
    CONVERSATIONAL_SURFACES,
    NORMALIZED_CONVERSATIONAL_SURFACES,
    PLANNER_VERSION,
    RENDERER_VERSION,
    SELECTION_VERSION,
    CandidateEvidenceIdentityV3,
    RecurrenceTermAuditV3,
    build_deterministic_sparse_v3_plan,
    candidate_evidence_identity,
    candidate_tie_key,
    precision_admitted,
    recurrence_mass,
)
from trec_rag.query_analyzer import AnalyzerFingerprint, AnalyzedQuery, stable_unique
from trec_rag.query_planner import tokenize_narrative


FINGERPRINT = AnalyzerFingerprint(
    contract_version="v3-test",
    implementation="additive-token-test-analyzer",
    lucene_version="test",
    analyzer_class="SyntheticAnalyzer",
    tokenizer="narrative-token-tape",
    filters=("NFKC", "casefold", "aliases"),
    stopword_sha256="0" * 64,
    unicode_version="test",
    index_id="synthetic",
)


class Analyzer:
    def __init__(self, aliases=None, *, exact_overrides=None):
        self.aliases = {
            "batteries": ("battery",),
            "reports": ("report",),
            "report": ("report",),
            "running": ("run",),
            "runs": ("run",),
            "the": (),
            **(aliases or {}),
        }
        self.exact_overrides = exact_overrides or {}

    @property
    def fingerprint(self):
        return FINGERPRINT

    def analyze(self, text):
        if text in self.exact_overrides:
            tokens = tuple(self.exact_overrides[text])
            return AnalyzedQuery(tokens, stable_unique(tokens), FINGERPRINT)
        result = []
        for token in tokenize_narrative(text).tokens:
            normalized = unicodedata.normalize("NFKC", token.text).casefold()
            if normalized in self.aliases:
                result.extend(self.aliases[normalized])
                continue
            if not any(character.isalnum() for character in normalized):
                continue
            result.extend(re.findall(r"[^\W\d_]+", normalized, flags=re.UNICODE))
        tokens = tuple(result)
        return AnalyzedQuery(tokens, stable_unique(tokens), FINGERPRINT)


def _plan(narrative, analyzer=None):
    return build_deterministic_sparse_v3_plan(
        "synthetic", narrative, analyzer or Analyzer()
    )


def test_inventory_versions_and_exact_evidence_hash_unicode_codepoint_offsets():
    assert len(CONVERSATIONAL_SURFACES) == 114
    assert len(NORMALIZED_CONVERSATIONAL_SURFACES) == 114
    narrative = "Café battery systems chemistry methods: Café battery durability recycling."
    plan = _plan(narrative)

    assert plan.status == "ok"
    assert plan.planner_version == PLANNER_VERSION == "det_sparse_v3"
    assert RENDERER_VERSION == "det_sparse_recurrent_anchor_renderer_v3"
    assert ANCHOR_SELECTOR_VERSION == "det_sparse_cross_unit_recurrent_anchor_v1"
    assert SELECTION_VERSION == "det_sparse_anchor_critical_quantile_selection_v3"
    assert plan.anchor.query_text == "Café battery"  # type: ignore[union-attr]
    candidate = next(row for row in plan.candidate_audit if row.exact_text == "Café battery")
    expected_payload = {
        "end": candidate.source_span.end,
        "narrative_sha256": hashlib.sha256(narrative.encode()).hexdigest(),
        "start": candidate.source_span.start,
        "text_sha256": hashlib.sha256("Café battery".encode()).hexdigest(),
        "token_ids": list(candidate.token_ids),
    }
    expected_bytes = json.dumps(
        expected_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    assert candidate.evidence.to_dict() == expected_payload
    assert candidate.evidence_sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert candidate.source_span.end == len("Café battery")
    assert candidate.source_span.end != len("Café battery".encode())
    changed = CandidateEvidenceIdentityV3(
        candidate.evidence.narrative_sha256,
        candidate.evidence.start,
        len("Café battery".encode()),
        candidate.evidence.text_sha256,
        candidate.evidence.token_ids,
    )
    assert changed.sha256 != candidate.evidence_sha256


def test_token_alignment_punctuation_zero_output_hyphen_and_one_to_many():
    analyzer = Analyzer({"sodium-ion": ("sodium", "ion")})
    narrative = "the sodium-ion storage chemistry methods: sodium-ion durability recycling."
    plan = _plan(narrative, analyzer)

    assert plan.status == "ok"
    assert plan.anchor.query_text == "sodium-ion"  # type: ignore[union-attr]
    assert plan.anchor.analyzed_tokens == ("sodium", "ion")  # type: ignore[union-attr]
    assert len(plan.anchor.token_ids) == 1  # type: ignore[union-attr]
    the = next(row for row in plan.aligned_tokens if row.exact_surface == "the")
    assert the.analyzer_tokens == ()
    colon = next(row for row in plan.aligned_tokens if row.exact_surface == ":")
    assert colon.analyzer_tokens == ()
    assert any(
        row.rejection_reasons == ("endpoint_has_no_analyzer_occurrence",)
        or "endpoint_has_no_analyzer_occurrence" in row.rejection_reasons
        for row in plan.candidate_audit
    )


def test_curly_apostrophe_and_nfkc_conversational_exclusion_is_occurrence_level():
    analyzer = Analyzer({"i’m": ("i", "m")})
    narrative = (
        "I’m asking about sodium ion batteries methods: "
        "Why do sodium ion batteries degrade safely?"
    )
    plan = _plan(narrative, analyzer)

    assert plan.status == "ok"
    assert plan.anchor.core_terms == ("battery", "ion", "sodium")  # type: ignore[union-attr]
    im_rows = [row for row in plan.aligned_occurrences if row.exact_surface == "I’m"]
    assert [row.analyzer_term for row in im_rows] == ["i", "m"]
    assert all(row.conversational for row in im_rows)
    assert all(
        row.conversational
        for row in plan.aligned_occurrences
        if unicodedata.normalize("NFKC", row.exact_surface).casefold()
        in {"asking", "about", "why", "do"}
    )


def test_token_alignment_mismatch_fails_original_only_before_candidate_scoring():
    u1 = "alpha beta energy storage:"
    analyzer = Analyzer(exact_overrides={u1: ("alpha_beta", "energy", "storage")})
    narrative = f"{u1} alpha beta durability recycling."

    plan = _plan(narrative, analyzer)

    assert plan.status == "fallback"
    assert plan.failure.code == "token_analyzer_alignment_mismatch"  # type: ignore[union-attr]
    assert plan.candidate_audit == ()
    assert plan.query_variants()[0].query_text == narrative


def test_unit_tapes_must_reconstruct_the_exact_whole_narrative_analysis():
    narrative = "alpha beta energy storage: alpha beta durability recycling."
    analyzer = Analyzer(
        exact_overrides={
            narrative: (
                "alpha_beta",
                "energy",
                "storage",
                "alpha",
                "beta",
                "durability",
                "recycling",
            )
        }
    )

    plan = _plan(narrative, analyzer)

    assert plan.status == "fallback"
    assert plan.failure.code == "unit_analyzer_alignment_mismatch"  # type: ignore[union-attr]
    assert plan.candidate_audit == ()
    assert plan.query_variants()[0].query_text == narrative


def test_candidate_window_alignment_mismatch_rejects_only_that_exact_window():
    narrative = (
        "sodium ion battery storage chemistry methods: "
        "sodium ion battery durability recycling."
    )
    plan = _plan(
        narrative,
        Analyzer(exact_overrides={"sodium ion": ("sodiumion",)}),
    )

    assert plan.status == "ok"
    mismatched = next(
        row for row in plan.candidate_audit if row.exact_text == "sodium ion"
    )
    assert not mismatched.admissible
    assert "candidate_token_alignment_mismatch" in mismatched.rejection_reasons
    assert mismatched.minimal_hulls == ()
    assert plan.anchor.query_text != "sodium ion"  # type: ignore[union-attr]


def test_generic_only_one_term_and_no_recurrence_all_abstain():
    generic = _plan("please explain report question: please explain report question.")
    one = _plan("sodium storage chemistry materials: sodium performance durability safety.")
    none = _plan("alpha beta gamma delta: epsilon zeta eta theta.")

    for plan in (generic, one, none):
        assert plan.status == "fallback"
        assert plan.failure.code == "recurrent_anchor_unavailable"  # type: ignore[union-attr]
        assert plan.facets == ()


def test_occurrence_level_stem_collision_and_dual_criticality():
    narrative = (
        "reports sodium systems chemistry methods: "
        "report sodium durability recycling. reports sodium output safety."
    )
    plan = _plan(narrative)

    assert plan.status == "ok"
    assert plan.anchor.query_text == "reports sodium"  # type: ignore[union-attr]
    report = next(row for row in plan.recurrence_audit if row.analyzer_term == "report")
    assert report.child_df == 1
    u2, u3 = plan.raw_criticality
    assert u2.full_core_intersection == ("report", "sodium")
    assert u2.eligible_core_intersection == ("sodium",)
    assert u2.label == "complete"
    assert u3.eligible_core_intersection == ("report", "sodium")


def test_recurrence_audit_freezes_exact_tf_df_occurrences_and_raw_unit_support():
    narrative = (
        "sodium sodium ion battery chemistry methods: "
        "sodium ion sodium durability recycling. "
        "sodium battery performance safety."
    )
    plan = _plan(narrative)

    assert plan.status == "ok"
    sodium = next(
        row for row in plan.recurrence_audit if row.analyzer_term == "sodium"
    )
    assert sodium.parent_tf == 2
    assert sodium.child_df == 2
    assert sodium.child_tf == 3
    assert sodium.supporting_child_unit_ids == ("u002", "u003")
    occurrences = {row.occurrence_id: row for row in plan.aligned_occurrences}
    assert len(sodium.parent_occurrence_ids) == 2
    assert len(sodium.child_occurrence_ids) == 3
    assert {
        occurrences[occurrence_id].unit_id
        for occurrence_id in sodium.parent_occurrence_ids
    } == {"u001"}
    assert {
        occurrences[occurrence_id].unit_id
        for occurrence_id in sodium.child_occurrence_ids
    } == {"u002", "u003"}
    assert all(
        occurrences[occurrence_id].analyzer_term == "sodium"
        and not occurrences[occurrence_id].conversational
        for occurrence_id in (
            *sodium.parent_occurrence_ids,
            *sodium.child_occurrence_ids,
        )
    )


def test_repeated_conversational_terms_lose_to_compact_subject():
    plan = _plan(
        "report report report sodium ion battery details: "
        "please report sodium ion battery performance durability."
    )

    assert plan.status == "ok"
    assert plan.anchor.query_text == "sodium ion battery"  # type: ignore[union-attr]
    assert "report" not in plan.anchor.core_terms  # type: ignore[union-attr]


def test_precision_integer_boundary_and_joint_support():
    assert precision_admitted(2, 3)
    assert not precision_admitted(2, 4)
    assert precision_admitted(3, 4)
    split = _plan(
        "sodium ion storage chemistry: sodium performance durability. ion safety recycling."
    )
    joint = _plan(
        "sodium ion storage chemistry: sodium ion performance durability. safety recycling systems."
    )

    assert split.status == "fallback"
    assert split.failure.code == "recurrent_anchor_unavailable"  # type: ignore[union-attr]
    assert joint.status == "ok"
    assert joint.anchor.core_terms == ("ion", "sodium")  # type: ignore[union-attr]
    assert [row.label for row in joint.raw_criticality] == ["complete", "anchorless"]


def test_half_parent_cap_uses_full_occurrences_and_odd_floor():
    seven = _plan(
        "please please sodium ion battery storage chemistry: "
        "sodium ion battery durability recycling."
    )
    nine = _plan(
        "please please sodium sodium ion battery energy storage chemistry: "
        "sodium ion battery energy durability recycling."
    )

    assert seven.status == "ok"
    assert seven.anchor.query_text == "sodium ion battery"  # type: ignore[union-attr]
    assert len(seven.anchor.analyzed_tokens) == 3  # type: ignore[union-attr]
    assert nine.status == "ok"
    assert nine.anchor.query_text == "sodium ion battery energy"  # type: ignore[union-attr]
    over = next(
        row for row in nine.candidate_audit if row.exact_text == "sodium sodium ion battery energy"
    )
    assert "candidate_exceeds_half_parent_occurrence_cap" in over.rejection_reasons


@pytest.mark.parametrize("extra_geothermal", ["", " geothermal heat pumps maintenance lifespan."])
def test_equal_and_unequal_strength_disjoint_maximal_cores_are_ambiguous(
    extra_geothermal,
):
    narrative = (
        "geothermal heat pumps and rooftop solar panels comparison overview: "
        "geothermal heat pumps efficiency installation. "
        f"rooftop solar panels output durability.{extra_geothermal}"
    )
    plan = _plan(narrative)

    assert plan.status == "fallback"
    assert plan.failure.code == "ambiguous_recurrent_anchor"  # type: ignore[union-attr]
    assert len([row for row in plan.core_audit if row.maximal]) == 2
    assert plan.facets == ()


def test_overlapping_cores_are_nonambiguous_and_later_tie_wins():
    plan = _plan(
        "sodium ion battery and ion battery storage comparison overview: "
        "sodium ion battery performance durability. "
        "ion battery storage safety recycling."
    )

    assert plan.status == "ok"
    assert plan.anchor.query_text == "ion battery storage"  # type: ignore[union-attr]
    assert {row.terms for row in plan.core_audit if row.maximal} == {
        ("battery", "ion", "sodium"),
        ("battery", "ion", "storage"),
    }


def test_nested_core_is_not_maximal_and_multiple_minimal_hulls_survive():
    nested = _plan(
        "sodium ion battery storage chemistry methods: "
        "sodium ion battery performance durability."
    )
    repeated = _plan(
        "sodium ion sodium ion chemistry storage methods systems: "
        "sodium ion performance durability."
    )

    assert nested.status == "ok"
    cores = {row.terms: row for row in nested.core_audit}
    assert not cores[("ion", "sodium")].maximal
    assert cores[("battery", "ion", "sodium")].maximal
    assert repeated.status == "ok"
    assert repeated.anchor.query_text == "sodium ion"  # type: ignore[union-attr]
    assert repeated.anchor.source_span.start == repeated.original_query_text.rfind(  # type: ignore[union-attr]
        "sodium ion", 0, repeated.lexical_units[0].source_span.end
    )
    four = next(
        row for row in repeated.candidate_audit if row.exact_text == "sodium ion sodium ion"
    )
    assert [row.exact_text for row in four.minimal_hulls] == [
        "sodium ion",
        "ion sodium",
        "sodium ion",
    ]
    assert len({row.evidence_sha256 for row in four.minimal_hulls}) == 3


def test_raw_zero_partial_complete_criticality_and_strict_two_unit_rendering():
    critical = _plan(
        "sodium ion battery storage chemistry systems methods: "
        "thermal safety recycling. sodium durability recycling. "
        "sodium ion battery efficiency lifespan."
    )
    two = _plan(
        "sodium ion storage chemistry methods: "
        "sodium ion durability recycling performance."
    )

    assert critical.status == "ok"
    assert [row.label for row in critical.raw_criticality] == [
        "anchorless",
        "partial",
        "complete",
    ]
    assert critical.raw_criticality[1].full_core_intersection == ("sodium",)
    assert two.status == "ok"
    assert [facet.query_text for facet in two.facets] == [
        "sodium ion storage chemistry methods:",
        "sodium ion sodium ion durability recycling performance.",
    ]
    assert all(row.reconstruction_exact for row in two.invariant_audit)
    assert all(row.strict_original_submultiset for row in two.invariant_audit)
    assert all(row.occurrence_count < row.original_occurrence_count for row in two.invariant_audit)
    assert [variant.variant_name for variant in two.query_variants()] == [
        "det_sparse_v3:original",
        "det_sparse_v3:facet:f01",
        "det_sparse_v3:facet:f02",
    ]


def test_protected_parent_and_deterministic_child_merges_preserve_exact_spans():
    narrative = (
        "sodium ion storage chemistry methods: "
        "sodium ion red blue. sodium ion red green. sodium ion black white. "
        "sodium ion north south. sodium ion east west."
    )
    plan = _plan(narrative)

    assert plan.status == "ok"
    assert len(plan.facets) == 4
    assert plan.facets[0].coverage_unit_ids == ("u001",)
    assert [row.result_unit_ids for row in plan.merge_audit] == [
        ("u002", "u003"),
        ("u004", "u005"),
    ]
    assert all("u001" not in row.result_unit_ids for row in plan.merge_audit)
    covered = [unit for facet in plan.facets for unit in facet.coverage_unit_ids]
    assert covered == [f"u{index:03d}" for index in range(1, 7)]
    for row in plan.merge_audit:
        for span in (row.left_source_span, row.right_source_span, row.combined_source_span):
            assert span.text == narrative[span.start : span.end]


def test_child_payload_requires_two_unique_eligible_terms_absent_anchor():
    discourse = _plan(
        "sodium ion storage chemistry methods: "
        "sodium ion durability recycling. please explain why it does this."
    )
    one_unique = _plan(
        "sodium ion storage chemistry methods: "
        "sodium ion durability recycling. running runs."
    )

    for plan in (discourse, one_unique):
        assert plan.status == "fallback"
        assert plan.failure.code == "insufficient_child_payload"  # type: ignore[union-attr]
        assert plan.facets == ()


def test_rendered_reconstruction_and_pairwise_signature_collisions_fail_closed():
    narrative = (
        "sodium ion storage chemistry methods: exchange durability recycling. "
        "sodium ion performance durability."
    )
    rendered = "sodium ion exchange durability recycling."
    reconstruction = _plan(
        narrative,
        Analyzer(exact_overrides={rendered: ("ionexchange", "durability", "recycling")}),
    )
    collision = _plan(
        "sodium ion storage chemistry methods: "
        "sodium ion epsilon zeta. sodium ion gamma delta.",
        Analyzer({"epsilon": ("gamma",), "zeta": ("delta",)}),
    )

    assert reconstruction.status == "fallback"
    assert reconstruction.failure.code == "rendered_query_reconstruction_mismatch"  # type: ignore[union-attr]
    assert collision.status == "fallback"
    assert collision.failure.code == "duplicate_facet_signature"  # type: ignore[union-attr]
    assert collision.facets == ()


def test_punctuation_records_count_toward_six_record_candidate_limit():
    narrative = (
        "alpha,beta gamma delta epsilon zeta: "
        "alpha beta durability recycling."
    )
    plan = _plan(narrative)

    assert plan.status == "ok"
    row = next(item for item in plan.candidate_audit if item.exact_text == "alpha,beta")
    assert len(row.token_ids) == 3
    assert row.admissible
    assert all(len(item.token_ids) <= 6 for item in plan.candidate_audit)
    assert not any(
        item.exact_text == "alpha,beta gamma delta epsilon zeta"
        for item in plan.candidate_audit
    )


def test_raw_anchorless_witness_can_be_masked_or_survive_child_merge():
    masked = _plan(
        "sodium ion storage chemistry methods: heat safety. "
        "sodium ion heat safety. sodium alpha beta gamma. ion delta epsilon zeta."
    )
    unmasked = _plan(
        "sodium ion storage chemistry methods: heat safety. heat wind. "
        "sodium ion alpha beta. sodium gamma delta epsilon."
    )

    assert masked.status == "ok"
    assert any(row.label == "anchorless" for row in masked.raw_criticality)
    assert not any(row.label == "anchorless" for row in masked.final_criticality)
    assert masked.merge_audit[0].result_unit_ids == ("u002", "u003")
    assert unmasked.status == "ok"
    assert any(row.label == "anchorless" for row in unmasked.raw_criticality)
    assert any(row.label == "anchorless" for row in unmasked.final_criticality)


def test_recurrence_mass_cap_and_candidate_tie_directions_are_exact():
    recurrence = {
        f"t{count}": RecurrenceTermAuditV3(
            analyzer_term=f"t{count}",
            parent_tf=count,
            child_df=2,
            child_tf=2,
            parent_occurrence_ids=(),
            child_occurrence_ids=(),
            supporting_child_unit_ids=("u002",),
        )
        for count in (1, 2, 3, 4)
    }
    assert [
        recurrence_mass((f"t{count}",), recurrence) for count in (1, 2, 3, 4)
    ] == [2, 4, 6, 6]

    plan = _plan(
        "sodium ion storage chemistry methods: sodium ion durability recycling."
    )
    base = next(row for row in plan.candidate_audit if row.selected)
    same_score_early = replace(
        base,
        source_span=replace(base.source_span, start=0, end=5),
        exact_text="beta",
    )
    same_score_late = replace(
        base,
        source_span=replace(base.source_span, start=3, end=8),
        exact_text="alpha",
    )
    assert min((same_score_early, same_score_late), key=candidate_tie_key) is same_score_late
    shorter = replace(
        same_score_early,
        source_span=replace(base.source_span, start=0, end=4),
        exact_text="beta",
    )
    assert min((same_score_early, shorter), key=candidate_tie_key) is shorter
    alpha = replace(shorter, exact_text="alpha")
    beta = replace(shorter, exact_text="beta")
    assert min((beta, alpha), key=candidate_tie_key) is alpha


_CORE_SCORE_ARGUMENTS = (
    "joint_child_df",
    "recurrence_mass_value",
    "recurrent_term_count",
    "support_union_count",
    "eligible_nonrecurrent_occurrence_count",
    "conversational_occurrence_count",
    "total_analyzed_occurrence_count",
    "source_token_record_count",
)


@pytest.mark.parametrize("decisive_index", range(len(_CORE_SCORE_ARGUMENTS)))
def test_each_core_score_component_has_exact_direction_and_dominates_later_fields(
    decisive_index,
):
    preferred = {name: 5 for name in _CORE_SCORE_ARGUMENTS}
    competitor = preferred.copy()
    decisive_name = _CORE_SCORE_ARGUMENTS[decisive_index]
    if decisive_index < 4:
        preferred[decisive_name] = 6
        competitor[decisive_name] = 5
    else:
        preferred[decisive_name] = 4
        competitor[decisive_name] = 5
    for later_index, later_name in enumerate(
        _CORE_SCORE_ARGUMENTS[decisive_index + 1 :],
        start=decisive_index + 1,
    ):
        if later_index < 4:
            preferred[later_name] = 0
            competitor[later_name] = 100
        else:
            preferred[later_name] = 100
            competitor[later_name] = 0

    preferred_score = planner_module._core_score(**preferred)
    competitor_score = planner_module._core_score(**competitor)

    assert preferred_score[:decisive_index] == competitor_score[:decisive_index]
    assert preferred_score[decisive_index] > competitor_score[decisive_index]
    assert all(
        preferred_score[index] < competitor_score[index]
        for index in range(decisive_index + 1, len(preferred_score))
    )
    assert preferred_score > competitor_score

    plan = _plan(
        "sodium ion storage chemistry methods: sodium ion durability recycling."
    )
    base = next(row for row in plan.candidate_audit if row.selected)
    first_evidence = candidate_evidence_identity(
        narrative_sha256="a" * 64,
        start=0,
        end=1,
        exact_text="a",
        token_ids=(0,),
    )
    second_evidence = candidate_evidence_identity(
        narrative_sha256="a" * 64,
        start=2,
        end=3,
        exact_text="b",
        token_ids=(1,),
    )
    preferred_candidate = replace(
        base,
        evidence=first_evidence,
        evidence_sha256=first_evidence.sha256,
        source_span=replace(base.source_span, start=0, end=1, text="a"),
        exact_text="a",
        token_ids=(0,),
        core_score=preferred_score,
        selected=False,
    )
    later_tie_favored_competitor = replace(
        base,
        evidence=second_evidence,
        evidence_sha256=second_evidence.sha256,
        source_span=replace(base.source_span, start=2, end=3, text="b"),
        exact_text="b",
        token_ids=(1,),
        core_score=competitor_score,
        selected=False,
    )
    _rows, selected = planner_module._select_anchor_candidate(
        (preferred_candidate, later_tie_favored_competitor),
        tuple(row for row in plan.core_audit if row.maximal),
    )
    assert selected.evidence == first_evidence


def test_evidence_hash_collision_never_collapses_or_co_selects_distinct_identities():
    plan = _plan(
        "sodium ion storage chemistry methods: sodium ion durability recycling."
    )
    base = next(row for row in plan.candidate_audit if row.selected)
    base_hull = base.minimal_hulls[0]
    first_evidence = candidate_evidence_identity(
        narrative_sha256="a" * 64,
        start=0,
        end=10,
        exact_text="sodium ion",
        token_ids=(0, 1),
    )
    second_evidence = candidate_evidence_identity(
        narrative_sha256="a" * 64,
        start=20,
        end=30,
        exact_text="sodium ion",
        token_ids=(4, 5),
    )
    forced_collision = "f" * 64
    first_hull = replace(
        base_hull,
        evidence=first_evidence,
        evidence_sha256=forced_collision,
        source_span=replace(
            base_hull.source_span, start=0, end=10, text="sodium ion"
        ),
        token_ids=(0, 1),
    )
    second_hull = replace(
        base_hull,
        evidence=second_evidence,
        evidence_sha256=forced_collision,
        source_span=replace(
            base_hull.source_span, start=20, end=30, text="sodium ion"
        ),
        token_ids=(4, 5),
    )
    first_candidate = replace(
        base,
        evidence=first_evidence,
        evidence_sha256=forced_collision,
        source_span=first_hull.source_span,
        token_ids=first_hull.token_ids,
        minimal_hulls=(first_hull,),
        selected=False,
    )
    second_candidate = replace(
        base,
        evidence=second_evidence,
        evidence_sha256=forced_collision,
        source_span=second_hull.source_span,
        token_ids=second_hull.token_ids,
        minimal_hulls=(second_hull,),
        selected=False,
    )

    cores = planner_module._build_cores((first_candidate, second_candidate))

    assert len(cores) == 1
    assert {row.evidence for row in cores[0].minimal_hulls} == {
        first_evidence,
        second_evidence,
    }
    selected_rows, selected = planner_module._select_anchor_candidate(
        (first_candidate, second_candidate),
        cores,
    )
    assert selected.evidence == second_evidence
    assert [row.evidence for row in selected_rows if row.selected] == [second_evidence]


def test_analyzer_fingerprint_drift_is_explicit_and_never_returns_partial_facets():
    changed = AnalyzerFingerprint(
        contract_version="changed",
        implementation="changed",
        lucene_version="test",
        analyzer_class="SyntheticAnalyzer",
        tokenizer="narrative-token-tape",
        filters=(),
        stopword_sha256="0" * 64,
        unicode_version="test",
        index_id="synthetic",
    )

    class DriftingAnalyzer(Analyzer):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def analyze(self, text):
            row = super().analyze(text)
            self.calls += 1
            if self.calls >= 2:
                return AnalyzedQuery(row.tokens, row.unique_tokens, changed)
            return row

    plan = _plan(
        "sodium ion storage chemistry methods: sodium ion durability recycling.",
        DriftingAnalyzer(),
    )

    assert plan.status == "fallback"
    assert plan.failure.code == "analyzer_fingerprint_changed"  # type: ignore[union-attr]
    assert plan.facets == ()


def test_final_merged_group_must_replay_exact_combined_substring_analysis():
    combined = "red blue. green gold."
    narrative = (
        "sodium ion storage chemistry methods: red blue. green gold. "
        "sodium ion alpha beta. sodium gamma delta."
    )
    analyzer = Analyzer(exact_overrides={combined: ("boundaryterm",)})

    plan = _plan(narrative, analyzer)

    assert plan.status == "fallback"
    assert plan.failure.code == "final_group_token_alignment_mismatch"  # type: ignore[union-attr]
    assert plan.merge_audit[0].combined_source_span.text == combined
    assert plan.facets == ()


def test_evidence_identity_never_collapses_equal_text_at_distinct_spans():
    narrative = "sodium ion sodium ion chemistry storage methods systems"
    first = candidate_evidence_identity(
        narrative_sha256=hashlib.sha256(narrative.encode()).hexdigest(),
        start=0,
        end=10,
        exact_text="sodium ion",
        token_ids=(0, 1),
    )
    second_start = narrative.rfind("sodium ion")
    second = candidate_evidence_identity(
        narrative_sha256=first.narrative_sha256,
        start=second_start,
        end=second_start + len("sodium ion"),
        exact_text="sodium ion",
        token_ids=(2, 3),
    )

    assert first.text_sha256 == second.text_sha256
    assert first.sha256 != second.sha256
    assert first.to_dict() != second.to_dict()
