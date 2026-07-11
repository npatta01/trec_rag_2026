from pathlib import Path

import pytest
import yaml

from trec_rag.det_sparse_v2_config import (
    ARM_NAMES,
    CANDIDATE_TOPIC_IDS,
    EXCLUDED_TOPIC_IDS,
    EXPERIMENT_ID,
    PILOT_TOPIC_COUNT,
    SELECTION_SEED,
    load_det_sparse_v2_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_CONFIG = REPO_ROOT / "configs" / "det_sparse_v2.yaml"


def _source():
    return yaml.safe_load(TRACKED_CONFIG.read_text(encoding="utf-8"))


def _write_synthetic_config(tmp_path, value):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "AGENTS.md").write_text("# Synthetic test root\n", encoding="utf-8")
    path = root / "configs" / "det_sparse_v2.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _write_raw_synthetic_config(tmp_path, value):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "AGENTS.md").write_text("# Synthetic test root\n", encoding="utf-8")
    path = root / "configs" / "det_sparse_v2.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_tracked_v2_config_is_the_frozen_structural_protocol():
    config = load_det_sparse_v2_config(TRACKED_CONFIG)

    assert config.schema_version == "det_sparse_v2"
    assert config.experiment_id == EXPERIMENT_ID
    assert config.candidate_topic_ids == CANDIDATE_TOPIC_IDS
    assert config.excluded_topic_ids == EXCLUDED_TOPIC_IDS
    assert not set(config.candidate_topic_ids).intersection(config.excluded_topic_ids)
    assert config.selection_seed == SELECTION_SEED
    assert config.selection_strata == ("A", "B", "C", "D")
    assert config.selected_topic_count == PILOT_TOPIC_COUNT == 4
    assert config.planner_version == "det_sparse_v2"
    assert config.renderer_version == "det_sparse_bounded_parent_renderer_v2"
    assert config.splitter_version == "det_sparse_exact_span_splitter_v1"
    assert config.tokenizer_version == "narrative_token_tape_v1"
    assert config.max_facets == 4
    assert config.min_facet_terms == 3
    assert config.parent_min_unique_terms == 4
    assert config.bounded_context_min_unique_terms == 2
    assert config.bounded_context_max_unique_terms == 6
    assert config.bounded_context_allocation == "floor_half_u1_unique_terms"
    assert config.prf_formula_version == "det_sparse_prf_v1"
    assert config.prf_artifact_version == "det_sparse_prf_artifact_v2"
    assert config.expand_parent_facet is False
    assert config.max_expanded_facets == 3
    assert config.arms == ARM_NAMES
    assert config.rrf_k == 60
    assert config.cost.max_unique_requests_per_topic == 9
    assert config.cost.max_external_requests == 36
    assert config.cost.model_calls == config.cost.reranker_calls == 0
    assert config.retrieval.endpoint_url.startswith("https://")
    assert config.retrieval.index_revision == "hosted_climbmix_unknown_revision"
    assert config.retrieval.hits == config.retrieval.required_results == 100
    assert config.retrieval.max_attempts == 1
    assert config.retrieval.retry_policy == "none"
    assert config.retrieval.redirect_policy == "none"
    assert config.retrieval_run_namespace == "det_sparse_v2_fresh_run_local"
    assert config.global_ticket_namespace == EXPERIMENT_ID
    assert config.external_gate_status == "blocked"
    assert config.external_gate_reason == "hosted_index_revision_unknown"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda source: source["topics"].update(
                {"candidate_ids": list(reversed(source["topics"]["candidate_ids"]))}
            ),
            "topics.candidate_ids.*frozen",
        ),
        (
            lambda source: source["topics"]["selection"].update({"selected_count": 5}),
            "topics.selection.selected_count.*4",
        ),
        (
            lambda source: source["query_planning"]["bounded_context"].update(
                {"max_unique_terms": 7}
            ),
            "bounded_context.max_unique_terms.*6",
        ),
        (
            lambda source: source["query_planning"].update(
                {"splitter_version": "changed"}
            ),
            "query_planning.splitter_version.*frozen",
        ),
        (
            lambda source: source["query_planning"].update(
                {"parent_min_unique_terms": 3}
            ),
            "parent_min_unique_terms.*4",
        ),
        (
            lambda source: source["retrieval"].update({"required_results": 50}),
            "required_results.*100",
        ),
        (
            lambda source: source["retrieval"].update({"max_attempts": 2}),
            "max_attempts.*1",
        ),
        (
            lambda source: source["retrieval"].update({"retry_policy": "retry"}),
            "retry_policy.*none",
        ),
        (
            lambda source: source["retrieval_ledger"].update(
                {"global_ticket_namespace": "rag25_det_sparse_difficult4_v1"}
            ),
            "global_ticket_namespace.*rag25_det_sparse_structural4_v2",
        ),
        (
            lambda source: source["expansion"].update({"expand_parent_facet": True}),
            "expand_parent_facet.*False",
        ),
        (
            lambda source: source["cost"].update({"max_external_requests": 37}),
            "max_external_requests.*36",
        ),
        (
            lambda source: source["external_gate"].update({"status": "authorized"}),
            "external_gate.status.*blocked",
        ),
    ],
)
def test_v2_loader_rejects_protocol_drift(tmp_path, mutate, message):
    source = _source()
    mutate(source)
    path = _write_synthetic_config(tmp_path, source)

    with pytest.raises(ValueError, match=message):
        load_det_sparse_v2_config(path)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("experiment", "id", "different", "experiment.id"),
        ("topics", "path", "other-topics.tsv", "topics.path"),
        ("retrieval", "endpoint_url", "http://example.test/search", "endpoint_url"),
        ("retrieval", "index", "another-index", "retrieval.index"),
        ("retrieval", "index_revision", "known-revision", "index_revision"),
        ("evaluation", "qrels", "other-qrels.txt", "evaluation.qrels"),
    ],
)
def test_v2_loader_freezes_source_and_service_identity(
    tmp_path,
    section,
    key,
    value,
    message,
):
    source = _source()
    source[section][key] = value
    path = _write_synthetic_config(tmp_path, source)

    with pytest.raises(ValueError, match=message):
        load_det_sparse_v2_config(path)


def test_v2_loader_rejects_non_loopback_or_credentialed_analyzer(tmp_path):
    source = _source()
    source["analyzer"]["url"] = "https://external.example/analyze"
    path = _write_synthetic_config(tmp_path, source)
    with pytest.raises(ValueError, match="analyzer.url"):
        load_det_sparse_v2_config(path)

    source["analyzer"]["url"] = "http://localhost:@external.example/analyze"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="analyzer.url"):
        load_det_sparse_v2_config(path)


@pytest.mark.parametrize(
    ("owner", "insert", "message"),
    [
        ("root", lambda source: source.update({"retries": 10}), "config keys.*unknown=retries"),
        (
            "retrieval",
            lambda source: source["retrieval"].update({"retries": 10}),
            "retrieval keys.*unknown=retries",
        ),
        (
            "selection",
            lambda source: source["topics"]["selection"].update({"manual_ids": ["14"]}),
            "topics.selection keys.*unknown=manual_ids",
        ),
        (
            "bounded context",
            lambda source: source["query_planning"]["bounded_context"].update(
                {"unbounded": True}
            ),
            "bounded_context keys.*unknown=unbounded",
        ),
    ],
)
def test_v2_loader_fails_closed_on_unknown_keys(tmp_path, owner, insert, message):
    del owner
    source = _source()
    insert(source)
    path = _write_synthetic_config(tmp_path, source)

    with pytest.raises(ValueError, match=message):
        load_det_sparse_v2_config(path)


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        (
            "schema_version: det_sparse_v2",
            "schema_version: det_sparse_v2\nschema_version: det_sparse_v2",
            "duplicate key 'schema_version'",
        ),
        (
            "  hits: 100",
            "  hits: 100\n  hits: 100",
            "duplicate key 'hits'",
        ),
    ],
)
def test_v2_loader_rejects_duplicate_yaml_keys_at_any_depth(
    tmp_path,
    needle,
    replacement,
    message,
):
    raw = TRACKED_CONFIG.read_text(encoding="utf-8")
    assert raw.count(needle) == 1
    path = _write_raw_synthetic_config(tmp_path, raw.replace(needle, replacement))

    with pytest.raises(yaml.constructor.ConstructorError, match=message):
        load_det_sparse_v2_config(path)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("experiment", "id", " rag25_det_sparse_structural4_v2", "leading or trailing"),
        ("retrieval", "retry_policy", "none ", "leading or trailing"),
        ("topics", "format", "TSV", "topics.format.*tsv"),
    ],
)
def test_v2_loader_rejects_whitespace_or_case_normalization_drift(
    tmp_path,
    section,
    key,
    value,
    message,
):
    source = _source()
    source[section][key] = value
    path = _write_synthetic_config(tmp_path, source)

    with pytest.raises(ValueError, match=message):
        load_det_sparse_v2_config(path)


def test_v2_loader_freezes_output_and_config_locations(tmp_path):
    source = _source()
    source["experiment"]["output_dir"] = "outputs/fresh-budget-copy"
    path = _write_synthetic_config(tmp_path, source)
    with pytest.raises(ValueError, match="experiment.output_dir"):
        load_det_sparse_v2_config(path)

    outside = tmp_path / "repo" / "copied.yaml"
    outside.write_text(TRACKED_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="config path"):
        load_det_sparse_v2_config(outside)
