from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

from trec_rag import det_sparse_v3_preflight
from trec_rag.det_sparse_v3_config import (
    ANALYZER_FINGERPRINT_SHA256,
    CANDIDATE_TOPIC_IDS,
    CONVERSATIONAL_PROJECTED_TERMS_SHA256,
    CONVERSATIONAL_PROJECTION_ENCODING,
    CONVERSATIONAL_PROJECTION_SHA256,
    CONVERSATIONAL_SURFACES,
    ConversationalProjection,
    ConversationalProjectionRecord,
    normalize_conversational_surface,
    load_det_sparse_v3_config,
)
from trec_rag.det_sparse_v3_preflight import (
    build_preflight,
    load_candidate_topics,
    validate_preflight,
)
from trec_rag.det_sparse_v3_selection import (
    CandidateScreen,
    CriticalityCounts,
    QuantileBinAudit,
    SelectionFailure,
    SelectionOutcomeV3,
    StructuralSelectionV3,
    semantic_plan_sha256,
)
from trec_rag.pipeline_models import QueryVariant
from trec_rag.query_analyzer import (
    AnalyzerFingerprint,
    AnalyzedQuery,
    RemoteLuceneQueryAnalyzer,
    stable_unique,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT = AnalyzerFingerprint(
    contract_version="lucene_default_english_v1",
    implementation="local_lucene_reference_server_v1",
    lucene_version="10.4.0",
    analyzer_class="io.anserini.analysis.DefaultEnglishAnalyzer chain",
    tokenizer="org.apache.lucene.analysis.standard.StandardTokenizer",
    filters=(
        "EnglishPossessiveFilter",
        "LowerCaseFilter",
        "StopFilter(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET)",
        "PorterStemFilter",
    ),
    stopword_sha256="2f66c0e3dde5d31c7e919e2ed4d9d91390696480be361bfa143ca9ae0cb7ca13",
    unicode_version="Lucene-10.4.0-StandardTokenizer-UAX29",
    index_id="hosted_climbmix_unknown_revision",
)
SOURCE = {
    "schema_version": "det_sparse_v3_source_provenance_v1",
    "commit": "a" * 40,
    "tree": "b" * 40,
    "source_tree_clean": True,
    "python_import_root": "/synthetic/code",
    "source_files_sha256": {"synthetic.py": "c" * 64},
    "required_module_bindings": {"synthetic": "synthetic.py"},
    "lucene_attestation_reuse": {"source_sha256": "e" * 64},
}
RUNTIME = {
    "schema_version": "det_sparse_v3_runtime_provenance_v1",
    "python": "3.12-test",
    "implementation": "CPython",
    "platform": "test",
    "executable": "/synthetic/python",
    "packages": {},
    "environment_file_sha256": {"pyproject.toml": "d" * 64},
    "lucene_analyzer_runtime": {
        "runtime_version": "synthetic",
        "artifact_size": 123,
    },
    "lucene_attestation_reuse": {"source_sha256": "e" * 64},
}


class Analyzer:
    @property
    def fingerprint(self):
        return FINGERPRINT

    def analyze(self, text):
        tokens = tuple(re.findall(r"[^\W\d_]+", text.lower()))
        return AnalyzedQuery(tokens, stable_unique(tokens), FINGERPRINT)


@dataclass(frozen=True)
class FakeUnit:
    unit_id: str


@dataclass(frozen=True)
class FakeFacet:
    facet_id: str
    variant_name: str
    coverage_unit_ids: tuple[str, ...]
    query_text: str
    bm25_signature: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class FakeCriticality:
    owner_id: str
    coverage_unit_ids: tuple[str, ...]
    full_core_intersection: tuple[str, ...]
    eligible_core_intersection: tuple[str, ...]
    missing_full_core_terms: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class FakeAnchor:
    core_terms: tuple[str, ...]


@dataclass(frozen=True)
class FakePlan:
    topic_id: str
    status: str
    narrative_sha256: str
    token_tape_sha256: str
    analyzer_token_sha256: str
    analyzer_fingerprint_sha256: str
    original_query_text: str
    original_bm25_signature: tuple[tuple[str, int], ...]
    aligned_tokens: tuple[dict[str, object], ...]
    aligned_occurrences: tuple[dict[str, object], ...]
    lexical_units: tuple[FakeUnit, ...]
    conversational_inventory: dict[str, object]
    recurrence_audit: tuple[dict[str, object], ...]
    candidate_audit: tuple[dict[str, object], ...]
    core_audit: tuple[dict[str, object], ...]
    anchor: FakeAnchor
    raw_criticality: tuple[FakeCriticality, ...]
    final_criticality: tuple[FakeCriticality, ...]
    merge_audit: tuple[dict[str, object], ...]
    facets: tuple[FakeFacet, ...]
    invariant_audit: tuple[dict[str, object], ...]
    failure: None = None

    def to_dict(self):
        return asdict(self)

    def query_variants(self):
        return (
            QueryVariant(
                topic_id=self.topic_id,
                variant_name="det_sparse_v3:original",
                query_text=self.original_query_text,
                source_type="det_sparse_v3_original",
            ),
            *(
                QueryVariant(
                    topic_id=self.topic_id,
                    variant_name=facet.variant_name,
                    query_text=facet.query_text,
                    source_type="det_sparse_v3_facet",
                )
                for facet in self.facets
            ),
        )


def _fake_projection() -> ConversationalProjection:
    terms = tuple(f"term{index:03d}" for index in range(75))
    records = tuple(
        ConversationalProjectionRecord(
            source_surface=surface,
            normalized_surface=normalize_conversational_surface(surface),
            analyzer_tokens=((terms[index],) if index < len(terms) else ()),
        )
        for index, surface in enumerate(CONVERSATIONAL_SURFACES)
    )
    return ConversationalProjection(
        encoding=CONVERSATIONAL_PROJECTION_ENCODING,
        records=records,
        sha256=CONVERSATIONAL_PROJECTION_SHA256,
        projected_terms=terms,
        projected_terms_sha256=CONVERSATIONAL_PROJECTED_TERMS_SHA256,
        analyzer_fingerprint_sha256=ANALYZER_FINGERPRINT_SHA256,
    )


def _fake_plan(topic_id: str, narrative: str) -> FakePlan:
    digest = hashlib.sha256(narrative.encode()).hexdigest()
    units = (FakeUnit("u01"), FakeUnit("u02"), FakeUnit("u03"))
    facets = (
        FakeFacet(
            "f01",
            "det_sparse_v3:facet:f01",
            ("u01",),
            f"parent topic {topic_id}",
            ((f"parent{topic_id}", 1),),
        ),
        FakeFacet(
            "f02",
            "det_sparse_v3:facet:f02",
            ("u02",),
            f"anchor topic {topic_id} first child",
            ((f"anchor{topic_id}", 1), ("first", 1)),
        ),
        FakeFacet(
            "f03",
            "det_sparse_v3:facet:f03",
            ("u03",),
            f"anchor topic {topic_id} second child",
            ((f"anchor{topic_id}", 1), ("second", 1)),
        ),
    )
    return FakePlan(
        topic_id=topic_id,
        status="ok",
        narrative_sha256=digest,
        token_tape_sha256=hashlib.sha256(f"tape:{topic_id}".encode()).hexdigest(),
        analyzer_token_sha256=hashlib.sha256(f"tokens:{topic_id}".encode()).hexdigest(),
        analyzer_fingerprint_sha256=ANALYZER_FINGERPRINT_SHA256,
        original_query_text=narrative,
        original_bm25_signature=((f"original{topic_id}", 2), ("whole", 1)),
        aligned_tokens=(
            {
                "token_id": 0,
                "exact_surface": "anchor",
                "source_span": {"start": 0, "end": 6, "text": "anchor"},
                "analyzer_tokens": ["anchor"],
            },
        ),
        aligned_occurrences=(
            {
                "occurrence_id": "o0001",
                "analyzer_term": "anchor",
                "source_span": {"start": 0, "end": 6, "text": "anchor"},
                "unit_id": "u01",
            },
        ),
        lexical_units=units,
        conversational_inventory={"projection_sha256": CONVERSATIONAL_PROJECTION_SHA256},
        recurrence_audit=(
            {
                "analyzer_term": "anchor",
                "child_df": 2,
                "child_occurrence_ids": ["o0002", "o0003"],
                "supporting_child_unit_ids": ["u02", "u03"],
            },
        ),
        candidate_audit=(
            {
                "evidence_sha256": hashlib.sha256(topic_id.encode()).hexdigest(),
                "exact_text": "anchor topic",
                "core_score": [2, 4, 2, 0, 0, 0, 0, 0],
                "minimal_hulls": [
                    {
                        "evidence": {
                            "start": 0,
                            "end": 12,
                            "narrative_sha256": digest,
                            "text_sha256": hashlib.sha256(b"anchor topic").hexdigest(),
                            "token_ids": [0, 1],
                        },
                        "source_span": {
                            "start": 0,
                            "end": 12,
                            "text": "anchor topic",
                        },
                    }
                ],
            },
        ),
        core_audit=(
            {
                "terms": ["anchor", f"topic{topic_id}"],
                "minimal_hulls": [
                    {
                        "evidence_sha256": hashlib.sha256(
                            f"hull:{topic_id}".encode()
                        ).hexdigest(),
                        "source_span": {"start": 0, "end": 12},
                    }
                ],
            },
        ),
        anchor=FakeAnchor(("anchor", f"topic{topic_id}")),
        raw_criticality=(
            FakeCriticality(
                "u02",
                ("u02",),
                (),
                ("audit_only",),
                ("anchor", f"topic{topic_id}"),
                "anchorless",
            ),
            FakeCriticality(
                "u03",
                ("u03",),
                ("anchor", f"topic{topic_id}"),
                ("anchor",),
                (),
                "complete",
            ),
        ),
        final_criticality=(
            FakeCriticality(
                "f02",
                ("u02",),
                (),
                ("audit_only",),
                ("anchor", f"topic{topic_id}"),
                "anchorless",
            ),
            FakeCriticality(
                "f03",
                ("u03",),
                ("anchor", f"topic{topic_id}"),
                ("anchor",),
                (),
                "complete",
            ),
        ),
        merge_audit=(),
        facets=facets,
        invariant_audit=({"all": True},),
    )


def _fake_outcome(topics, *, merge_masked: bool = False) -> SelectionOutcomeV3:
    plans = {topic.id: _fake_plan(topic.id, topic.narrative) for topic in topics}
    screens = []
    selected = ("14", "31", "72", "273")
    for topic in topics:
        plan = plans[topic.id]
        critical = topic.id == "14"
        screens.append(
            CandidateScreen(
                topic_id=topic.id,
                narrative_sha256=plan.narrative_sha256,
                token_tape_sha256=plan.token_tape_sha256,
                analyzer_token_sha256=plan.analyzer_token_sha256,
                analyzer_fingerprint_sha256=plan.analyzer_fingerprint_sha256,
                unit_count=3,
                original_unique_term_count=8,
                facet_count=3,
                merge_count=0,
                anchor_core_term_count=2,
                anchor_text_sha256="1" * 64,
                anchor_evidence_sha256="2" * 64,
                anchor_core_sha256="3" * 64,
                raw_criticality=CriticalityCounts(1, 0, 1),
                final_criticality=(
                    CriticalityCounts(0, 1, 1)
                    if merge_masked and critical
                    else CriticalityCounts(1, 0, 1)
                ),
                plan_status="ok",
                plan_semantic_sha256=semantic_plan_sha256(plan),
                eligible=True,
                failure_code=None,
                failure_message=None,
                selection_digest=hashlib.sha256(f"select:{topic.id}".encode()).hexdigest(),
                remaining_sort_key=(3, 8, 3, 2, int(topic.id)),
                quantile_bin=(None if critical else 0),
                selection_role=(
                    "critical"
                    if critical
                    else (f"bin{selected.index(topic.id) - 1}" if topic.id in selected else None)
                ),
            )
        )
    bins = (
        QuantileBinAudit(0, 0, 2, ("31", "58"), "31"),
        QuantileBinAudit(1, 2, 5, ("72", "219", "233"), "72"),
        QuantileBinAudit(2, 5, 8, ("273", "477", "499"), "273"),
    )
    selection = StructuralSelectionV3(
        schema_version="det_sparse_anchor_critical_quantile_selection_v3",
        selection_version="det_sparse_anchor_critical_quantile_selection_v3",
        seed="det_sparse_v3_recurrent_anchor_selection_20260711",
        status="failure" if merge_masked else "ok",
        candidate_topic_ids=CANDIDATE_TOPIC_IDS,
        screens=tuple(screens),
        critical_pool_topic_ids=CANDIDATE_TOPIC_IDS,
        critical_topic_id="14",
        remaining_ordered_topic_ids=CANDIDATE_TOPIC_IDS[1:],
        quantile_bins=bins,
        provisional_selected_topic_ids=selected,
        selected_topic_ids=() if merge_masked else selected,
        failure=(
            SelectionFailure(
                "selected_critical_merge_masked",
                "selected critical topic lacks a final full-set anchorless witness",
            )
            if merge_masked
            else None
        ),
    )
    return SelectionOutcomeV3(selection=selection, plans_by_topic=plans)


def _context(tmp_path: Path):
    root = tmp_path / "synthetic_repo"
    root.mkdir()
    (root / "AGENTS.md").write_text("# Synthetic root\n", encoding="utf-8")
    config_path = root / "configs" / "det_sparse_v3.yaml"
    config_path.parent.mkdir()
    shutil.copyfile(REPO_ROOT / "configs" / "det_sparse_v3.yaml", config_path)
    topics_path = root / (
        "trec-rag-data/trec-rag-2026/development-data/topics/"
        "rag25-topics-dev.tsv"
    )
    topics_path.parent.mkdir(parents=True)
    rows = [b"\xff\t\xff\n", b"144\t\xff\n"]
    rows.extend(
        (
            f"{topic_id}\tWhole request for {topic_id}: First child evidence? "
            f"Second child mechanism?\n"
        ).encode()
        for topic_id in CANDIDATE_TOPIC_IDS
    )
    topics_path.write_bytes(b"".join(rows))
    config = load_det_sparse_v3_config(config_path)
    assert hashlib.sha256(
        json.dumps(
            FINGERPRINT.to_dict(), separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest() == ANALYZER_FINGERPRINT_SHA256
    return config


def _bind_dependencies(monkeypatch, *, merge_masked: bool = False, calls=None):
    def fresh(_config):
        if calls is not None:
            calls.append("fresh_analyzer")
        return Analyzer()

    monkeypatch.setattr(det_sparse_v3_preflight, "_fresh_query_analyzer", fresh)
    monkeypatch.setattr(
        det_sparse_v3_preflight,
        "_require_bound_query_analyzer",
        lambda _config, _analyzer: None,
    )
    monkeypatch.setattr(
        det_sparse_v3_preflight,
        "current_source_provenance",
        lambda _root: dict(SOURCE),
    )
    monkeypatch.setattr(
        det_sparse_v3_preflight,
        "current_runtime_provenance",
        lambda _root: dict(RUNTIME),
    )
    monkeypatch.setattr(
        det_sparse_v3_preflight,
        "build_conversational_projection",
        lambda _analyzer: _fake_projection(),
    )

    def select(topics, **_kwargs):
        return _fake_outcome(topics, merge_masked=merge_masked)

    monkeypatch.setattr(
        det_sparse_v3_preflight,
        "screen_and_select_structural_topics_v3",
        select,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object, *, pretty: bool = False) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _reseal_after_attack(
    output: Path,
    *,
    mirror_metadata: bool = True,
    forge_completion: bool = True,
) -> None:
    metadata = json.loads((output / "_preflight.json").read_text())
    manifest_path = output / "pre_retrieval_freeze.json"
    manifest = json.loads(manifest_path.read_text())
    if mirror_metadata:
        for field in tuple(manifest["metadata"]):
            manifest["metadata"][field] = metadata[field]
    for row in manifest["artifacts"]:
        path = output / row["path"]
        row["size"] = path.stat().st_size
        row["sha256"] = _sha256_file(path)
    _write_json(manifest_path, manifest)
    completion_path = output / "preflight_completion.json"
    if forge_completion and completion_path.exists():
        forged = det_sparse_v3_preflight._completion_record(metadata, output)
        _write_json(completion_path, forged, pretty=True)


def _built_preflight(tmp_path, monkeypatch):
    config = _context(tmp_path)
    calls: list[str] = []
    _bind_dependencies(monkeypatch, calls=calls)
    result = build_preflight(config)
    return config, result, calls


def test_candidate_loader_decodes_only_nine_and_rejects_any_whitespace_alias(
    tmp_path,
):
    config = _context(tmp_path)

    rows = load_candidate_topics(config)

    assert tuple(row.topic.id for row in rows) == CANDIDATE_TOPIC_IDS
    assert all(row.topic.id != "144" for row in rows)
    path = config.topics_path
    path.write_bytes(b" 14\t\xff\n" + path.read_bytes())
    with pytest.raises(ValueError, match="whitespace alias"):
        load_candidate_topics(config)


def test_projection_gate_failure_precedes_reservation_and_candidate_read(
    tmp_path, monkeypatch
):
    config = _context(tmp_path)
    _bind_dependencies(monkeypatch)
    opened = False

    def fail_projection(_analyzer):
        raise ValueError("synthetic projection drift")

    def forbidden_loader(_config):
        nonlocal opened
        opened = True
        raise AssertionError("candidate source must remain closed")

    monkeypatch.setattr(
        det_sparse_v3_preflight, "build_conversational_projection", fail_projection
    )
    monkeypatch.setattr(det_sparse_v3_preflight, "load_candidate_topics", forbidden_loader)

    with pytest.raises(ValueError, match="projection drift"):
        build_preflight(config)

    assert not opened
    assert not (config.output_dir / "preflight").exists()


def test_reservation_and_inventory_are_create_only_before_candidate_read(
    tmp_path, monkeypatch
):
    config = _context(tmp_path)
    _bind_dependencies(monkeypatch)

    def fail_loader(_config):
        output = config.output_dir / "preflight"
        reservation = json.loads((output / "preflight_reservation.json").read_text())
        inventory = json.loads((output / "conversational_inventory.json").read_text())
        assert reservation["state"] == "reserved_before_candidate_read"
        assert inventory["state"] == "frozen_before_candidate_read"
        assert reservation["external_calls"] == 0
        assert inventory["projection_sha256"] == CONVERSATIONAL_PROJECTION_SHA256
        raise RuntimeError("synthetic candidate-read crash")

    monkeypatch.setattr(det_sparse_v3_preflight, "load_candidate_topics", fail_loader)

    with pytest.raises(RuntimeError, match="candidate-read crash"):
        build_preflight(config)

    output = config.output_dir / "preflight"
    assert (output / "preflight_reservation.json").is_file()
    assert (output / "conversational_inventory.json").is_file()
    assert not (output / "selection.json").exists()


def test_success_freezes_all_candidate_evidence_selected_witness_and_zero_cost(
    tmp_path, monkeypatch
):
    config, result, calls = _built_preflight(tmp_path, monkeypatch)

    assert result.valid
    assert len(result.selected_topic_ids) == 4
    assert result.derived_max_total_requests <= 36
    assert len(calls) >= 2
    assert not config.evaluation.qrels.exists()
    metadata = json.loads((result.output_dir / "_preflight.json").read_text())
    assert metadata["external_calls"] == 0
    assert metadata["model_calls"] == 0
    assert metadata["reranker_calls"] == 0
    assert metadata["qrels_opened"] is False
    assert metadata["external_gate_status"] == "blocked"
    assert metadata["completion_required"] is True
    completion = json.loads(
        (result.output_dir / "preflight_completion.json").read_text()
    )
    assert completion["state"] == (
        "completed_after_postseal_attestation_and_fresh_replay"
    )
    assert completion["mechanical_valid"] is True
    for projection in metadata["request_projections"]:
        facet_count = projection["facet_count"]
        assert projection["base_unique_requests"] == 1 + facet_count
        # E expands O and FE expands M-1 children; protected f01 never expands.
        assert projection["eligible_expansion_bases"] == facet_count
        assert projection["derived_max_unique_requests"] == 1 + 2 * facet_count
    assert metadata["derived_max_total_unique_requests"] == sum(
        row["derived_max_unique_requests"]
        for row in metadata["request_projections"]
    ) == result.derived_max_total_requests
    assert len(metadata["candidate_plans"]) == 9
    assert len(metadata["selected_plans"]) == 4
    for row in metadata["candidate_plans"]:
        assert row["plan_semantic_sha256"] != row["plan_artifact_sha256"]
        plan = json.loads((result.output_dir / row["plan_path"]).read_text())
        assert plan["aligned_tokens"]
        assert plan["aligned_occurrences"]
        assert plan["recurrence_audit"]
        assert plan["candidate_audit"]
        assert plan["core_audit"]
    criticality = json.loads(
        (result.output_dir / "criticality_ledger.json").read_text()
    )
    critical = next(row for row in criticality["rows"] if row["topic_id"] == "14")
    assert any(row["label"] == "anchorless" for row in critical["raw"])
    assert any(row["label"] == "anchorless" for row in critical["final"])
    validate_preflight(config, result.output_dir)
    with pytest.raises(FileExistsError):
        build_preflight(config)


def test_merge_masked_critical_archives_no_go_without_replacement(
    tmp_path, monkeypatch
):
    config = _context(tmp_path)
    _bind_dependencies(monkeypatch, merge_masked=True)

    result = build_preflight(config)

    assert not result.valid
    assert result.selected_topic_ids == ()
    selection = json.loads((result.output_dir / "selection.json").read_text())
    assert selection["failure"]["code"] == "selected_critical_merge_masked"
    assert selection["provisional_selected_topic_ids"]
    assert selection["selected_topic_ids"] == []
    assert not (result.output_dir / "selected_plans").exists()
    completion = json.loads(
        (result.output_dir / "preflight_completion.json").read_text()
    )
    assert completion["mechanical_valid"] is False
    with pytest.raises(ValueError, match="mechanically invalid"):
        validate_preflight(config, result.output_dir)


def test_in_memory_config_drift_is_rejected_before_any_output(tmp_path, monkeypatch):
    config = _context(tmp_path)
    _bind_dependencies(monkeypatch)
    drifted = replace(config, max_facets=3)

    with pytest.raises(ValueError, match="fresh canonical reload"):
        build_preflight(drifted)

    assert not (config.output_dir / "preflight").exists()


@pytest.mark.parametrize(
    "drift",
    [
        lambda config: replace(config, max_facets=4.0),
        lambda config: replace(config, expand_parent_facet=0),
        lambda config: replace(
            config,
            cost=replace(config.cost, max_external_requests=36.0),
        ),
        lambda config: replace(
            config,
            cost=replace(config.cost, model_calls=False),
        ),
    ],
)
def test_same_value_different_type_config_aliases_are_rejected(
    tmp_path, monkeypatch, drift
):
    config = _context(tmp_path)
    _bind_dependencies(monkeypatch)

    with pytest.raises(ValueError, match="fresh canonical reload"):
        build_preflight(drift(config))

    assert not (config.output_dir / "preflight").exists()


def test_source_attestation_drift_is_rejected_before_reservation(
    tmp_path, monkeypatch
):
    config = _context(tmp_path)
    _bind_dependencies(monkeypatch)
    calls = 0

    def drifting_source(_root):
        nonlocal calls
        calls += 1
        source = dict(SOURCE)
        if calls > 1:
            source["commit"] = "f" * 40
        return source

    monkeypatch.setattr(
        det_sparse_v3_preflight, "current_source_provenance", drifting_source
    )

    with pytest.raises(ValueError, match="before preflight reservation"):
        build_preflight(config)

    assert not (config.output_dir / "preflight").exists()


def test_build_rejects_symlinked_output_parent_before_reservation(
    tmp_path, monkeypatch
):
    config = _context(tmp_path)
    _bind_dependencies(monkeypatch)
    escape = tmp_path / "escaped_outputs"
    escape.mkdir()
    (config.root_dir / "outputs").symlink_to(escape, target_is_directory=True)

    with pytest.raises(ValueError, match="must not traverse a symlink"):
        build_preflight(config)

    assert not (escape / config.experiment_id / "preflight").exists()


def test_selected_critical_witness_uses_full_core_not_label_or_eligible_audit():
    topics = tuple(
        type("SyntheticTopic", (), {"id": topic_id, "narrative": f"text {topic_id}"})()
        for topic_id in CANDIDATE_TOPIC_IDS
    )
    outcome = _fake_outcome(topics)
    selected = tuple(
        outcome.plans_by_topic[topic_id]
        for topic_id in outcome.selection.selected_topic_ids
    )
    assert det_sparse_v3_preflight._selected_critical_witness(outcome, selected)

    critical = selected[0]
    mislabeled = replace(
        critical.final_criticality[0],
        full_core_intersection=("anchor",),
        eligible_core_intersection=(),
    )
    tampered = replace(
        critical,
        final_criticality=(mislabeled, *critical.final_criticality[1:]),
    )
    tampered_outcome = replace(
        outcome,
        plans_by_topic={**outcome.plans_by_topic, critical.topic_id: tampered},
    )
    tampered_selected = (tampered, *selected[1:])
    assert not det_sparse_v3_preflight._selected_critical_witness(
        tampered_outcome, tampered_selected
    )


def test_metadata_paths_are_fixed_and_reject_traversal_alias(tmp_path, monkeypatch):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["selection_path"] = "candidate_plans/../selection.json"
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="fixed path"):
        validate_preflight(config, result.output_dir)


def test_preflight_rejects_symlink_artifact_substitution(tmp_path, monkeypatch):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    target = result.output_dir / "selection.json"
    target.unlink()
    target.symlink_to(result.output_dir / "candidate_screen.json")

    with pytest.raises(ValueError, match="must not be symlinks"):
        validate_preflight(config, result.output_dir)


def test_numeric_fields_reject_boolean_and_float_coercion(tmp_path, monkeypatch):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["request_projections"][0]["derived_max_unique_requests"] = float(
        metadata["request_projections"][0]["derived_max_unique_requests"]
    )
    metadata["external_calls"] = False
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="integer, not bool/float"):
        validate_preflight(config, result.output_dir)


def test_inventory_mutation_is_recomputed_not_self_attested(tmp_path, monkeypatch):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    inventory_path = result.output_dir / "conversational_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["records"][0]["analyzer_tokens"].append("attacker")
    _write_json(inventory_path, inventory, pretty=True)
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["inventory_artifact_sha256"] = _sha256_file(inventory_path)
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="conversational inventory"):
        validate_preflight(config, result.output_dir)


def test_candidate_semantic_hash_mutation_cannot_be_resealed(
    tmp_path, monkeypatch
):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["candidate_plans"][0]["plan_semantic_sha256"] = "f" * 64
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="candidate plan row"):
        validate_preflight(config, result.output_dir)


def test_complete_candidate_plan_evidence_mutation_fails_semantic_replay(
    tmp_path, monkeypatch
):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    plan_path = result.output_dir / metadata["candidate_plans"][0]["plan_path"]
    plan = json.loads(plan_path.read_text())
    plan["recurrence_audit"][0]["child_df"] = 99
    _write_json(plan_path, plan, pretty=True)
    metadata["candidate_plans"][0]["plan_artifact_sha256"] = _sha256_file(plan_path)
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="candidate plan 14"):
        validate_preflight(config, result.output_dir)


def test_criticality_witness_artifact_mutation_fails_replay(tmp_path, monkeypatch):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    path = result.output_dir / "criticality_ledger.json"
    value = json.loads(path.read_text())
    value["rows"][0]["final"][0]["full_core_intersection"] = ["anchor"]
    _write_json(path, value, pretty=True)
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["criticality_artifact_sha256"] = _sha256_file(path)
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="criticality ledger"):
        validate_preflight(config, result.output_dir)


def test_formatting_only_rewrite_is_not_a_valid_reseal(tmp_path, monkeypatch):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    path = result.output_dir / "selection.json"
    value = json.loads(path.read_text())
    _write_json(path, value, pretty=False)
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["selection_artifact_sha256"] = _sha256_file(path)
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="selection bytes are not canonical"):
        validate_preflight(config, result.output_dir)


@pytest.mark.parametrize("attack", ["mutation", "alias", "extra"])
def test_terminal_completion_receipt_mutation_alias_and_extra_are_rejected(
    tmp_path, monkeypatch, attack
):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    receipt_path = result.output_dir / "preflight_completion.json"
    if attack == "mutation":
        receipt = json.loads(receipt_path.read_text())
        receipt["state"] = "attacker_claimed_complete"
        _write_json(receipt_path, receipt, pretty=True)
        message = "terminal completion receipt"
    elif attack == "alias":
        receipt_path.unlink()
        receipt_path.symlink_to(result.output_dir / "selection.json")
        message = "must not be symlinks"
    else:
        (result.output_dir / "preflight_completion_copy.json").write_bytes(
            receipt_path.read_bytes()
        )
        message = "unsealed or missing files"

    with pytest.raises(ValueError, match=message):
        validate_preflight(config, result.output_dir)


def test_unsealed_extra_file_and_empty_directory_are_rejected(
    tmp_path, monkeypatch
):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    (result.output_dir / "unsealed.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsealed or missing files"):
        validate_preflight(config, result.output_dir)

    (result.output_dir / "unsealed.json").unlink()
    (result.output_dir / "unexpected_empty").mkdir()
    with pytest.raises(ValueError, match="unexpected or missing directories"):
        validate_preflight(config, result.output_dir)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unsupported")
def test_special_fifo_entry_is_rejected_before_inventory_checks(
    tmp_path, monkeypatch
):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    os.mkfifo(result.output_dir / "unexpected_fifo")

    with pytest.raises(ValueError, match="regular files or directories"):
        validate_preflight(config, result.output_dir)


def _set_nested(value: object, path: tuple[object, ...], replacement: object) -> None:
    owner = value
    for component in path[:-1]:
        owner = owner[component]  # type: ignore[index]
    owner[path[-1]] = replacement  # type: ignore[index]


@pytest.mark.parametrize(
    ("category", "path", "replacement"),
    [
        ("surface", ("aligned_tokens", 0, "exact_surface"), "attacker"),
        ("offset", ("aligned_tokens", 0, "source_span", "start"), 1),
        (
            "support_unit",
            ("recurrence_audit", 0, "supporting_child_unit_ids", 0),
            "u99",
        ),
        (
            "support_occurrence",
            ("recurrence_audit", 0, "child_occurrence_ids", 0),
            "o9999",
        ),
        ("score", ("candidate_audit", 0, "core_score", 0), 99),
        (
            "hull_evidence",
            ("candidate_audit", 0, "minimal_hulls", 0, "evidence", "start"),
            1,
        ),
        (
            "hull_span",
            ("candidate_audit", 0, "minimal_hulls", 0, "source_span", "end"),
            11,
        ),
        ("criticality_label", ("raw_criticality", 0, "label"), "complete"),
        ("evidence_hash", ("candidate_audit", 0, "evidence_sha256"), "f" * 64),
    ],
)
def test_fresh_replay_rejects_named_candidate_evidence_mutations_after_reseal(
    tmp_path,
    monkeypatch,
    category,
    path,
    replacement,
):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    plan_row = metadata["candidate_plans"][0]
    plan_path = result.output_dir / plan_row["plan_path"]
    plan = json.loads(plan_path.read_text())
    _set_nested(plan, path, replacement)
    _write_json(plan_path, plan, pretty=True)

    # Update every self-attested artifact hash so failure can only come from
    # fresh semantic replay against the candidate narrative and analyzer.
    artifact_hash = _sha256_file(plan_path)
    plan_row["plan_artifact_sha256"] = artifact_hash
    candidate_screen_path = result.output_dir / "candidate_screen.json"
    candidate_screen = json.loads(candidate_screen_path.read_text())
    candidate_screen[0]["candidate_plan_artifact_sha256"] = artifact_hash
    _write_json(candidate_screen_path, candidate_screen, pretty=True)
    metadata["candidate_screen_artifact_sha256"] = _sha256_file(
        candidate_screen_path
    )
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="candidate plan 14"):
        validate_preflight(config, result.output_dir)


@pytest.mark.parametrize("mutation", ["winner", "selected_order"])
def test_fresh_replay_rejects_selection_winner_and_order_mutations(
    tmp_path, monkeypatch, mutation
):
    config, result, _calls = _built_preflight(tmp_path, monkeypatch)
    selection_path = result.output_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    if mutation == "winner":
        selection["quantile_bins"][0]["winner_topic_id"] = "58"
    else:
        selection["selected_topic_ids"] = list(
            reversed(selection["selected_topic_ids"])
        )
    _write_json(selection_path, selection, pretty=True)
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["selection_artifact_sha256"] = _sha256_file(selection_path)
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="structural selection"):
        validate_preflight(config, result.output_dir)


def test_build_replay_refuses_reuse_of_initial_analyzer_instance(
    tmp_path, monkeypatch
):
    config = _context(tmp_path)
    _bind_dependencies(monkeypatch)
    analyzer = Analyzer()
    monkeypatch.setattr(
        det_sparse_v3_preflight,
        "_fresh_query_analyzer",
        lambda _config: analyzer,
    )

    with pytest.raises(ValueError, match="fresh analyzer client instance"):
        build_preflight(config)

    assert not (
        config.output_dir / "preflight" / "preflight_completion.json"
    ).exists()


def test_public_analyzer_binding_rejects_fake_type_and_wrong_loopback_url(tmp_path):
    config = _context(tmp_path)

    with pytest.raises(ValueError, match="exact RemoteLucene"):
        det_sparse_v3_preflight._require_bound_query_analyzer(config, Analyzer())
    with pytest.raises(ValueError, match="wrong base URL"):
        det_sparse_v3_preflight._require_bound_query_analyzer(
            config,
            RemoteLuceneQueryAnalyzer("http://127.0.0.1:19999"),
        )


@pytest.mark.parametrize("provenance", ["source", "runtime"])
@pytest.mark.parametrize(
    ("drift_call", "message"),
    [
        (4, "between screening and writes"),
        (5, "after preflight writes"),
    ],
)
def test_prewrite_and_postseal_provenance_drift_is_rejected(
    tmp_path,
    monkeypatch,
    provenance,
    drift_call,
    message,
):
    config = _context(tmp_path)
    _bind_dependencies(monkeypatch)
    calls = 0

    if provenance == "source":
        def attestation(_root):
            nonlocal calls
            calls += 1
            row = dict(SOURCE)
            if calls == drift_call:
                row["commit"] = "f" * 40
            return row

        monkeypatch.setattr(
            det_sparse_v3_preflight, "current_source_provenance", attestation
        )
    else:
        def attestation(_root):
            nonlocal calls
            calls += 1
            row = dict(RUNTIME)
            if calls == drift_call:
                lucene = dict(row["lucene_analyzer_runtime"])
                # Ordinary equality treats 123 and 123.0 as equal; the gate
                # must compare nested provenance types exactly.
                lucene["artifact_size"] = 123.0
                row["lucene_analyzer_runtime"] = lucene
            return row

        monkeypatch.setattr(
            det_sparse_v3_preflight, "current_runtime_provenance", attestation
        )

    with pytest.raises(ValueError, match=message):
        build_preflight(config)

    completion = config.output_dir / "preflight" / "preflight_completion.json"
    assert not completion.exists()
    if drift_call == 5:
        # The transient attestation has reverted, but the previously sealed
        # archive can never become publicly valid without the terminal receipt.
        with pytest.raises(ValueError, match="completion_path is missing"):
            validate_preflight(config, config.output_dir / "preflight")
