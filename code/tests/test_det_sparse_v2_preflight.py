from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from trec_rag import det_sparse_v2_preflight
from trec_rag.det_sparse_v2_config import (
    CANDIDATE_TOPIC_IDS,
    load_det_sparse_v2_config,
)
from trec_rag.det_sparse_v2_preflight import (
    _coverage_paths,
    build_preflight,
    load_candidate_topics,
    validate_preflight,
)
from trec_rag.query_analyzer import AnalyzerFingerprint, AnalyzedQuery, stable_unique


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
    "schema_version": "det_sparse_v2_source_provenance_v1",
    "commit": "a" * 40,
    "tree": "b" * 40,
    "source_tree_clean": True,
    "python_import_root": "/synthetic/code",
    "source_files_sha256": {"synthetic.py": "c" * 64},
    "required_module_bindings": {"synthetic": "synthetic.py"},
}
RUNTIME = {
    "schema_version": "det_sparse_v2_runtime_provenance_v1",
    "python": "3.12-test",
    "implementation": "CPython",
    "platform": "test",
    "executable": "/synthetic/python",
    "packages": {},
    "environment_file_sha256": {"pyproject.toml": "d" * 64},
    "lucene_analyzer_runtime": {"runtime_version": "synthetic"},
}


class Analyzer:
    @property
    def fingerprint(self):
        return FINGERPRINT

    def analyze(self, text):
        tokens = tuple(re.findall(r"[^\W\d_]+", text.lower()))
        return AnalyzedQuery(tokens, stable_unique(tokens), FINGERPRINT)


def _bind_public_dependencies(monkeypatch, analyzer):
    monkeypatch.setattr(
        det_sparse_v2_preflight,
        "_fresh_query_analyzer",
        lambda _config: analyzer,
    )
    monkeypatch.setattr(
        det_sparse_v2_preflight,
        "_require_bound_query_analyzer",
        lambda _config, _analyzer: None,
    )
    monkeypatch.setattr(
        det_sparse_v2_preflight,
        "current_source_provenance",
        lambda _root: dict(SOURCE),
    )
    monkeypatch.setattr(
        det_sparse_v2_preflight,
        "current_runtime_provenance",
        lambda _root: dict(RUNTIME),
    )


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value, *, pretty=False):
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


def _reseal_after_attack(output, *, mirror_metadata=True):
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


def _built_preflight(tmp_path, monkeypatch):
    config, analyzer = _context(tmp_path)
    _bind_public_dependencies(monkeypatch, analyzer)
    result = build_preflight(config)
    return config, analyzer, result


def _narrative(topic_id: str, unit_count: int) -> str:
    parent = f"shared alpha beta gamma delta epsilon {topic_id}word:"
    children = [
        f"request{letter} evidence{letter} mechanism{letter} outcome{letter}?"
        for letter in "bcdefgh"[: unit_count - 1]
    ]
    return " ".join((parent, *children))


def _context(tmp_path, *, no_long_topics: bool = False):
    root = tmp_path / "synthetic_repo"
    root.mkdir()
    (root / "AGENTS.md").write_text("# Synthetic test root\n", encoding="utf-8")
    config_path = root / "configs" / "det_sparse_v2.yaml"
    config_path.parent.mkdir()
    shutil.copyfile(REPO_ROOT / "configs" / "det_sparse_v2.yaml", config_path)
    topics_path = root / (
        "trec-rag-data/trec-rag-2026/development-data/topics/"
        "rag25-topics-dev.tsv"
    )
    topics_path.parent.mkdir(parents=True)
    rows = []
    for index, topic_id in enumerate(CANDIDATE_TOPIC_IDS):
        if no_long_topics:
            unit_count = 2 if index < 7 else 3
        elif index < 6:
            unit_count = 2
        elif index < 9:
            unit_count = 3
        else:
            unit_count = 4 + (index % 2)
        rows.append(f"{topic_id}\t{_narrative(topic_id, unit_count)}\n".encode())
    # A locked topic deliberately has invalid narrative UTF-8. The selective
    # loader may decode its ID to skip it, but must never decode its narrative.
    rows.insert(0, b"144\t\xff\n")
    rows.insert(0, b"\xff\t\xff\n")
    topics_path.write_bytes(b"".join(rows))
    config = load_det_sparse_v2_config(config_path)
    expected_fingerprint = hashlib.sha256(
        json.dumps(
            FINGERPRINT.to_dict(), separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    assert expected_fingerprint == config.analyzer.expected_fingerprint_sha256
    return config, Analyzer()


def test_selective_candidate_loader_never_decodes_excluded_or_unrelated_rows(tmp_path):
    config, _analyzer = _context(tmp_path)

    rows = load_candidate_topics(config)

    assert tuple(row.topic.id for row in rows) == CANDIDATE_TOPIC_IDS
    assert all(row.topic.id != "144" for row in rows)
    assert all(len(row.source_line_sha256) == 64 for row in rows)


def test_candidate_loader_rejects_whitespace_alias_for_frozen_id(tmp_path):
    config, _analyzer = _context(tmp_path)
    path = config.topics_path
    payload = path.read_bytes().replace(b"14\t", b" 14\t", 1)
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="candidate topic IDs are missing: 14"):
        load_candidate_topics(config)


def test_v2_preflight_selects_four_strata_freezes_distinct_paths_and_never_reads_qrels(
    tmp_path,
    monkeypatch,
):
    config, analyzer = _context(tmp_path)
    _bind_public_dependencies(monkeypatch, analyzer)
    assert not config.evaluation.qrels.exists()

    result = build_preflight(config)

    assert result.valid
    assert len(result.selected_topic_ids) == 4
    assert result.derived_max_total_requests <= 36
    assert result.hard_external_request_ceiling == 36
    assert not config.evaluation.qrels.exists()
    metadata = json.loads((result.output_dir / "_preflight.json").read_text())
    assert metadata["mechanical_valid"] is True
    assert metadata["fallback_topic_ids"] == []
    assert metadata["external_gate_status"] == "blocked"
    assert metadata["external_calls"] == 0
    assert metadata["model_calls"] == 0
    assert metadata["reranker_calls"] == 0
    assert metadata["qrels_opened"] is False
    selection = json.loads((result.output_dir / "selection.json").read_text())
    screens = {row["topic_id"]: row for row in selection["screens"]}
    assert [screens[topic_id]["stratum"] for topic_id in result.selected_topic_ids] == [
        "A",
        "B",
        "C",
        "D",
    ]
    candidates = json.loads((result.output_dir / "candidate_screen.json").read_text())
    assert len(candidates) == len(CANDIDATE_TOPIC_IDS)
    assert all("narrative" not in row for row in candidates)
    registry = json.loads((result.output_dir / "base_query_registry.json").read_text())
    coverage = json.loads((result.output_dir / "coverage_paths.json").read_text())
    assert all(row["non_original_path"] is True for row in coverage)
    for topic_id in result.selected_topic_ids:
        topic_rows = [row for row in registry if row["topic_id"] == topic_id]
        assert len(topic_rows) == 1 + next(
            len(plan.facets) for plan in result.selected_plans if plan.topic_id == topic_id
        )
        assert len({row["query_text"] for row in topic_rows}) == len(topic_rows)
        assert len(
            {
                tuple(tuple(item) for item in row["bm25_signature"])
                for row in topic_rows
            }
        ) == len(topic_rows)
    validate_preflight(config, result.output_dir)
    with pytest.raises(FileExistsError):
        build_preflight(config)


def test_empty_required_stratum_archives_no_go_without_manual_substitution(
    tmp_path, monkeypatch
):
    config, analyzer = _context(tmp_path, no_long_topics=True)
    _bind_public_dependencies(monkeypatch, analyzer)

    result = build_preflight(config)

    assert not result.valid
    assert result.selected_topic_ids == ()
    selection = json.loads((result.output_dir / "selection.json").read_text())
    assert selection["status"] == "failure"
    assert selection["failure"]["code"] == "empty_stratum_D"
    assert not config.evaluation.qrels.exists()
    with pytest.raises(ValueError, match="mechanically invalid"):
        validate_preflight(config, result.output_dir)


def test_v2_preflight_rejects_in_memory_config_drift_before_output(tmp_path):
    config, analyzer = _context(tmp_path)
    drifted = replace(config, max_facets=3)

    with pytest.raises(ValueError, match="fresh canonical reload"):
        build_preflight(drifted)

    assert not (config.output_dir / "preflight").exists()


def test_preflight_reserves_create_only_archive_before_structural_screening(
    tmp_path, monkeypatch
):
    config, analyzer = _context(tmp_path)
    _bind_public_dependencies(monkeypatch, analyzer)

    def fail_screen(*_args, **_kwargs):
        raise RuntimeError("synthetic screening crash")

    monkeypatch.setattr(
        det_sparse_v2_preflight,
        "screen_and_select_structural_topics",
        fail_screen,
    )

    with pytest.raises(RuntimeError, match="screening crash"):
        build_preflight(config)

    reservation = config.output_dir / "preflight" / "preflight_reservation.json"
    assert reservation.is_file()
    stored = json.loads(reservation.read_text())
    assert stored["state"] == "reserved_before_structural_screening"
    assert stored["external_calls"] == 0
    assert stored["qrels_opened"] is False
    assert not (reservation.parent / "selection.json").exists()


def test_public_analyzer_binding_rejects_fake_client_and_wrong_url(tmp_path):
    config, analyzer = _context(tmp_path)

    with pytest.raises(ValueError, match="exact RemoteLucene"):
        det_sparse_v2_preflight._require_bound_query_analyzer(config, analyzer)

    class PretendRemote:
        base_url = "http://127.0.0.1:9999"

    with pytest.raises(ValueError, match="exact RemoteLucene"):
        det_sparse_v2_preflight._require_bound_query_analyzer(
            config, PretendRemote()
        )

    from trec_rag.query_analyzer import RemoteLuceneQueryAnalyzer

    with pytest.raises(ValueError, match="wrong base URL"):
        det_sparse_v2_preflight._require_bound_query_analyzer(
            config,
            RemoteLuceneQueryAnalyzer("http://127.0.0.1:9999"),
        )


def test_build_rejects_symlinked_output_parent_before_reservation(tmp_path, monkeypatch):
    config, analyzer = _context(tmp_path)
    _bind_public_dependencies(monkeypatch, analyzer)
    escape = tmp_path / "escaped_outputs"
    escape.mkdir()
    (config.root_dir / "outputs").symlink_to(escape, target_is_directory=True)

    with pytest.raises(ValueError, match="must not traverse a symlink"):
        build_preflight(config)

    assert not (escape / config.experiment_id / "preflight").exists()


def test_coverage_ledger_rejects_duplicate_unit_assignment(tmp_path, monkeypatch):
    _config, _analyzer, result = _built_preflight(tmp_path, monkeypatch)
    plan = result.selected_plans[0]
    duplicated_child = replace(
        plan.facets[1],
        coverage_unit_ids=(
            plan.facets[0].coverage_unit_ids[0],
            *plan.facets[1].coverage_unit_ids,
        ),
    )
    tampered_plan = replace(
        plan,
        facets=(plan.facets[0], duplicated_child, *plan.facets[2:]),
    )

    with pytest.raises(ValueError, match="exactly once"):
        _coverage_paths((tampered_plan,))


def test_source_attestation_drift_is_refused_before_reservation(tmp_path, monkeypatch):
    config, analyzer = _context(tmp_path)
    _bind_public_dependencies(monkeypatch, analyzer)
    calls = 0

    def drifting_source(_root):
        nonlocal calls
        calls += 1
        row = dict(SOURCE)
        if calls > 1:
            row["commit"] = "e" * 40
        return row

    monkeypatch.setattr(
        det_sparse_v2_preflight,
        "current_source_provenance",
        drifting_source,
    )

    with pytest.raises(ValueError, match="before preflight reservation"):
        build_preflight(config)

    assert not (config.output_dir / "preflight").exists()


def test_queries_jsonl_must_replay_exact_canonical_records(tmp_path, monkeypatch):
    config, _analyzer, result = _built_preflight(tmp_path, monkeypatch)
    path = result.output_dir / "queries.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["query_text"] += " tampered"
    path.write_bytes(
        b"".join(
            (
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + "\n"
            ).encode()
            for row in rows
        )
    )
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["queries_sha256"] = _sha256_file(path)
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="queries.jsonl does not replay"):
        validate_preflight(config, result.output_dir)


def test_metadata_paths_are_fixed_and_cannot_use_traversal_aliases(
    tmp_path, monkeypatch
):
    config, _analyzer, result = _built_preflight(tmp_path, monkeypatch)
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["selection_path"] = "plans/../selection.json"
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="fixed path"):
        validate_preflight(config, result.output_dir)


def test_preflight_rejects_symlink_artifact_substitution(tmp_path, monkeypatch):
    config, _analyzer, result = _built_preflight(tmp_path, monkeypatch)
    registry = result.output_dir / "base_query_registry.json"
    registry.unlink()
    registry.symlink_to(result.output_dir / "selection.json")

    with pytest.raises(ValueError, match="must not be symlinks"):
        validate_preflight(config, result.output_dir)


def test_projection_numbers_reject_float_and_boolean_coercion(tmp_path, monkeypatch):
    config, _analyzer, result = _built_preflight(tmp_path, monkeypatch)
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


def test_metadata_analyzer_identity_is_recomputed_not_self_attested(
    tmp_path, monkeypatch
):
    config, _analyzer, result = _built_preflight(tmp_path, monkeypatch)
    metadata_path = result.output_dir / "_preflight.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["analyzer_fingerprint"]["implementation"] = "attacker"
    _write_json(metadata_path, metadata, pretty=True)
    _reseal_after_attack(result.output_dir)

    with pytest.raises(ValueError, match="analyzer identity/hash"):
        validate_preflight(config, result.output_dir)


def test_freeze_metadata_must_exactly_mirror_preflight_metadata(
    tmp_path, monkeypatch
):
    config, _analyzer, result = _built_preflight(tmp_path, monkeypatch)
    manifest_path = result.output_dir / "pre_retrieval_freeze.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["experiment_id"] = "attacker"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="exactly mirror"):
        validate_preflight(config, result.output_dir)


def test_unsealed_extra_file_is_rejected(tmp_path, monkeypatch):
    config, _analyzer, result = _built_preflight(tmp_path, monkeypatch)
    (result.output_dir / "unsealed.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsealed or missing"):
        validate_preflight(config, result.output_dir)


def test_plan_rows_bind_semantic_object_and_artifact_bytes(tmp_path, monkeypatch):
    _config, _analyzer, result = _built_preflight(tmp_path, monkeypatch)
    metadata = json.loads((result.output_dir / "_preflight.json").read_text())

    for row in metadata["plans"]:
        assert set(row) == {
            "ordinal",
            "topic_id",
            "status",
            "plan_path",
            "plan_semantic_sha256",
            "plan_artifact_sha256",
            "facet_count",
        }
        assert len(row["plan_semantic_sha256"]) == 64
        assert len(row["plan_artifact_sha256"]) == 64
