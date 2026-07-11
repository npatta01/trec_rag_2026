from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import trec_rag.det_sparse_v3_config as config_module
from trec_rag.det_sparse_v3_config import (
    CANDIDATE_EVIDENCE_KEYS,
    CANDIDATE_EVIDENCE_VERSION,
    CANDIDATE_TOPIC_IDS,
    CONVERSATIONAL_NORMALIZED_SHA256,
    CONVERSATIONAL_PROJECTED_TERM_COUNT,
    CONVERSATIONAL_PROJECTED_TERMS_SHA256,
    CONVERSATIONAL_PROJECTION_SHA256,
    CONVERSATIONAL_SOURCE_SHA256,
    CONVERSATIONAL_SURFACES,
    EXCLUDED_TOPIC_IDS,
    NORMALIZED_CONVERSATIONAL_SURFACES,
    build_conversational_projection,
    load_det_sparse_v3_config,
)
from trec_rag.deterministic_sparse_v3 import (
    ANCHOR_SELECTOR_VERSION as PLANNER_ANCHOR_SELECTOR_VERSION,
    CONVERSATIONAL_SURFACES as PLANNER_CONVERSATIONAL_SURFACES,
    CONVERSATIONAL_SURFACE_VERSION as PLANNER_SURFACE_VERSION,
    PLANNER_VERSION as IMPLEMENTED_PLANNER_VERSION,
    RENDERER_VERSION as IMPLEMENTED_RENDERER_VERSION,
    SELECTION_VERSION as IMPLEMENTED_SELECTION_VERSION,
    candidate_evidence_identity,
)
from trec_rag.query_analyzer import AnalyzerFingerprint, AnalyzedQuery


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_CONFIG = REPO_ROOT / "configs" / "det_sparse_v3.yaml"


def _source() -> dict[str, object]:
    value = yaml.safe_load(TRACKED_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_synthetic_config(tmp_path: Path, value: object) -> Path:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "AGENTS.md").write_text("# Synthetic root\n", encoding="utf-8")
    path = root / "configs" / "det_sparse_v3.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _write_raw_synthetic_config(tmp_path: Path, value: str) -> Path:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "AGENTS.md").write_text("# Synthetic root\n", encoding="utf-8")
    path = root / "configs" / "det_sparse_v3.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_tracked_v3_config_freezes_fresh_boundary_and_cost_contract():
    config = load_det_sparse_v3_config(TRACKED_CONFIG)

    assert config.schema_version == "det_sparse_v3"
    assert config.candidate_topic_ids == CANDIDATE_TOPIC_IDS
    assert config.excluded_topic_ids == EXCLUDED_TOPIC_IDS
    assert not set(CANDIDATE_TOPIC_IDS).intersection(EXCLUDED_TOPIC_IDS)
    assert config.selection.version == "det_sparse_anchor_critical_quantile_selection_v3"
    assert config.selection.selected_count == 4
    assert config.selection.bin_count == 3
    assert config.selection.critical_intersection == "full_aligned_analyzer_terms"
    assert config.selection.merge_masked_policy == "stop_no_replacement"
    assert config.planner_version == "det_sparse_v3"
    assert config.renderer_version == "det_sparse_recurrent_anchor_renderer_v3"
    assert config.anchor_selector_version == "det_sparse_cross_unit_recurrent_anchor_v1"
    assert config.candidate_window.max_token_records == 6
    assert config.candidate_window.min_recurrent_terms == 2
    assert config.candidate_window.recurrence_precision_numerator == 2
    assert config.candidate_window.recurrence_precision_denominator == 3
    assert config.candidate_window.parent_occurrence_divisor == 2
    assert config.criticality.min_child_payload_terms == 2
    assert config.candidate_evidence.version == CANDIDATE_EVIDENCE_VERSION
    assert config.candidate_evidence.keys == CANDIDATE_EVIDENCE_KEYS
    assert config.candidate_evidence.encoding == "compact_sorted_key_utf8_json_no_newline"
    assert config.prf_formula_version == "det_sparse_prf_v1"
    assert config.prf_artifact_version == "det_sparse_prf_artifact_v3"
    assert config.retrieval_run_namespace == "det_sparse_v3_fresh_run_local"
    assert config.global_ticket_namespace == "rag25_det_sparse_structural4_v3"
    assert config.retrieval.required_results == 100
    assert config.retrieval.max_attempts == 1
    assert config.retrieval.retry_policy == "none"
    assert config.retrieval.redirect_policy == "none"
    assert config.cost.max_external_requests == 36
    assert config.cost.model_calls == config.cost.reranker_calls == 0
    assert config.external_gate_status == "blocked"


def test_conversational_source_normalization_and_projection_identities_are_frozen():
    config = load_det_sparse_v3_config(TRACKED_CONFIG)
    surfaces = config.conversational_surfaces

    assert len(CONVERSATIONAL_SURFACES) == len(set(CONVERSATIONAL_SURFACES)) == 114
    assert len(NORMALIZED_CONVERSATIONAL_SURFACES) == 114
    assert surfaces.surfaces == CONVERSATIONAL_SURFACES
    assert surfaces.normalized_surfaces == NORMALIZED_CONVERSATIONAL_SURFACES
    assert surfaces.source_sha256 == CONVERSATIONAL_SOURCE_SHA256
    assert surfaces.normalized_sha256 == CONVERSATIONAL_NORMALIZED_SHA256
    assert surfaces.expected_projection_sha256 == CONVERSATIONAL_PROJECTION_SHA256
    assert surfaces.projected_term_count == CONVERSATIONAL_PROJECTED_TERM_COUNT == 75
    assert surfaces.projected_terms_sha256 == CONVERSATIONAL_PROJECTED_TERMS_SHA256
    assert surfaces.projection_policy == "verify_attested_analyzer_before_candidates"


def test_config_is_cross_bound_to_planner_versions_inventory_and_evidence_bytes():
    config = load_det_sparse_v3_config(TRACKED_CONFIG)

    assert IMPLEMENTED_PLANNER_VERSION == config.planner_version
    assert IMPLEMENTED_RENDERER_VERSION == config.renderer_version
    assert PLANNER_ANCHOR_SELECTOR_VERSION == config.anchor_selector_version
    assert IMPLEMENTED_SELECTION_VERSION == config.selection_version
    assert PLANNER_SURFACE_VERSION == config.conversational_surfaces.version
    assert PLANNER_CONVERSATIONAL_SURFACES == CONVERSATIONAL_SURFACES

    identity = candidate_evidence_identity(
        narrative_sha256="a" * 64,
        start=2,
        end=7,
        exact_text="café",
        token_ids=(3, 4),
    )
    exact_payload = {
        "end": 7,
        "narrative_sha256": "a" * 64,
        "start": 2,
        "text_sha256": hashlib.sha256("café".encode("utf-8")).hexdigest(),
        "token_ids": [3, 4],
    }
    exact_bytes = json.dumps(
        exact_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert tuple(exact_payload) == config.candidate_evidence.keys
    assert identity.to_dict() == exact_payload
    assert identity.sha256 == hashlib.sha256(exact_bytes).hexdigest()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["topics"].update(
                {"candidate_ids": list(reversed(value["topics"]["candidate_ids"]))}
            ),
            "topics.candidate_ids",
        ),
        (
            lambda value: value["topics"].update(
                {"excluded_ids": value["topics"]["excluded_ids"][:-1]}
            ),
            "topics.excluded_ids.*length",
        ),
        (
            lambda value: value["query_planning"]["conversational_surfaces"].update(
                {"source_sha256": "0" * 64}
            ),
            "source_sha256.*frozen",
        ),
        (
            lambda value: value["query_planning"]["conversational_surfaces"].update(
                {"expected_projection_sha256": None}
            ),
            "expected_projection_sha256.*type str",
        ),
        (
            lambda value: value["query_planning"]["candidate_evidence"].update(
                {"keys": ["start", "end"]}
            ),
            "candidate_evidence.keys.*length",
        ),
        (
            lambda value: value["expansion"].update(
                {"artifact_version": "det_sparse_prf_artifact_v2"}
            ),
            "artifact_version.*v3",
        ),
        (
            lambda value: value["retrieval_ledger"].update(
                {"run_namespace": "det_sparse_v2_fresh_run_local"}
            ),
            "run_namespace.*v3",
        ),
        (
            lambda value: value["topics"]["selection"].update(
                {"merge_masked_policy": "replace"}
            ),
            "merge_masked_policy.*stop_no_replacement",
        ),
        (
            lambda value: value["cost"].update({"max_external_requests": 37}),
            "max_external_requests.*36",
        ),
    ],
)
def test_v3_loader_rejects_any_frozen_protocol_drift(
    tmp_path: Path,
    mutate,
    message: str,
):
    source = _source()
    mutate(source)
    path = _write_synthetic_config(tmp_path, source)
    with pytest.raises(ValueError, match=message):
        load_det_sparse_v3_config(path)


def test_v3_loader_rejects_unknown_duplicate_and_alias_keys(tmp_path: Path):
    source = _source()
    source["query_planning"]["candidate_window"]["fallback"] = "model"
    path = _write_synthetic_config(tmp_path, source)
    with pytest.raises(ValueError, match="candidate_window keys.*unknown=fallback"):
        load_det_sparse_v3_config(path)

    raw = TRACKED_CONFIG.read_text(encoding="utf-8")
    duplicate = raw.replace(
        "schema_version: det_sparse_v3",
        "schema_version: det_sparse_v3\nschema_version: det_sparse_v3",
        1,
    )
    path = _write_raw_synthetic_config(tmp_path, duplicate)
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        load_det_sparse_v3_config(path)

    alias = raw.replace(
        "schema_version: det_sparse_v3",
        "schema_version: &schema det_sparse_v3",
        1,
    ).replace(
        "  id: rag25_det_sparse_structural4_v3",
        "  id: *schema",
        1,
    )
    path = _write_raw_synthetic_config(tmp_path, alias)
    with pytest.raises(yaml.constructor.ConstructorError, match="aliases are forbidden"):
        load_det_sparse_v3_config(path)


def test_v3_loader_requires_the_canonical_config_path(tmp_path: Path):
    outside = tmp_path / "repo"
    outside.mkdir()
    (outside / "AGENTS.md").write_text("# Synthetic root\n", encoding="utf-8")
    copied = outside / "copied.yaml"
    copied.write_text(TRACKED_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="config path"):
        load_det_sparse_v3_config(copied)


def test_projection_builder_hashes_exact_ordered_surface_to_term_records(monkeypatch):
    fingerprint = AnalyzerFingerprint(
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
            return fingerprint

        def analyze(self, surface):
            tokens = ("shared",) if len(surface) % 2 else ("shared", "even")
            return AnalyzedQuery(tokens, tuple(dict.fromkeys(tokens)), fingerprint)

    records = [
        {
            "source_surface": surface,
            "normalized_surface": config_module.normalize_conversational_surface(
                surface
            ),
            "analyzer_tokens": (
                ["shared"] if len(surface) % 2 else ["shared", "even"]
            ),
        }
        for surface in CONVERSATIONAL_SURFACES
    ]
    projected_terms = ["even", "shared"]

    def canonical_hash(value):
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    monkeypatch.setattr(
        config_module,
        "ANALYZER_FINGERPRINT_SHA256",
        canonical_hash(fingerprint.to_dict()),
    )
    monkeypatch.setattr(
        config_module,
        "CONVERSATIONAL_PROJECTION_SHA256",
        canonical_hash(records),
    )
    monkeypatch.setattr(config_module, "CONVERSATIONAL_PROJECTED_TERM_COUNT", 2)
    monkeypatch.setattr(
        config_module,
        "CONVERSATIONAL_PROJECTED_TERMS_SHA256",
        canonical_hash(projected_terms),
    )

    projection = build_conversational_projection(Analyzer())

    assert len(projection.records) == 114
    assert projection.records[0].source_surface == "a"
    assert projection.records[-1].source_surface == "yours"
    assert projection.projected_terms == ("even", "shared")
    assert projection.sha256 == canonical_hash(records)


def test_projection_builder_rejects_analyzer_fingerprint_drift():
    fingerprint = AnalyzerFingerprint(
        contract_version="wrong",
        implementation="wrong",
        lucene_version="wrong",
        analyzer_class="wrong",
        tokenizer="wrong",
        filters=(),
        stopword_sha256="0" * 64,
        unicode_version=None,
        index_id=None,
    )

    class Analyzer:
        @property
        def fingerprint(self):
            return fingerprint

        def analyze(self, surface):
            raise AssertionError(f"must fail before analyzing {surface}")

    with pytest.raises(ValueError, match="fingerprint drifted"):
        build_conversational_projection(Analyzer())
