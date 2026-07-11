from pathlib import Path

import pytest
import yaml

from trec_rag.det_sparse_config import (
    ARM_NAMES,
    LOCKED_PLANNER_TOPIC_IDS,
    PILOT_TOPIC_IDS,
    load_det_sparse_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_synthetic_config(tmp_path, value):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "AGENTS.md").write_text("# Synthetic test root\n", encoding="utf-8")
    path = root / "configs" / "det_sparse_v1.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_tracked_det_sparse_config_is_the_frozen_cost_bounded_protocol():
    config = load_det_sparse_config(REPO_ROOT / "configs" / "det_sparse_v1.yaml")

    assert config.topic_ids == PILOT_TOPIC_IDS
    assert not LOCKED_PLANNER_TOPIC_IDS.intersection(config.topic_ids)
    assert config.arms == ARM_NAMES
    assert config.max_facets == 4
    assert config.retrieval.hits == 100
    assert config.retrieval.cache_policy == "fresh_run_local"
    assert config.cost.max_external_requests == 36
    assert config.cost.model_calls == 0
    assert config.cost.reranker_calls == 0
    assert config.analyzer.url.startswith("http://127.0.0.1:")


def test_config_rejects_locked_topics_and_cost_or_protocol_drift(tmp_path):
    source = yaml.safe_load(
        (REPO_ROOT / "configs" / "det_sparse_v1.yaml").read_text(encoding="utf-8")
    )

    locked = {**source, "topics": {**source["topics"], "ids": ["144", "225", "707", "897"]}}
    locked_path = _write_synthetic_config(tmp_path, locked)
    with pytest.raises(ValueError, match="topics.ids.*frozen"):
        load_det_sparse_config(locked_path)

    expensive = {
        **source,
        "cost": {**source["cost"], "max_external_requests": 37},
    }
    expensive_path = _write_synthetic_config(tmp_path, expensive)
    with pytest.raises(ValueError, match="max_external_requests.*36"):
        load_det_sparse_config(expensive_path)

    tuned = {**source, "fusion": {**source["fusion"], "k": 30}}
    tuned_path = _write_synthetic_config(tmp_path, tuned)
    with pytest.raises(ValueError, match="fusion.k.*60"):
        load_det_sparse_config(tuned_path)


def test_config_rejects_non_loopback_analyzer(tmp_path):
    source = yaml.safe_load(
        (REPO_ROOT / "configs" / "det_sparse_v1.yaml").read_text(encoding="utf-8")
    )
    source["analyzer"]["url"] = "https://external.example/analyze"
    path = _write_synthetic_config(tmp_path, source)

    with pytest.raises(ValueError, match="analyzer.url"):
        load_det_sparse_config(path)

    source["analyzer"]["url"] = "http://localhost:@external.example/analyze"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="analyzer.url"):
        load_det_sparse_config(path)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("experiment", "id", "different", "experiment.id"),
        ("topics", "path", "other-topics.tsv", "topics.path"),
        ("evaluation", "qrels", "other-qrels.txt", "evaluation.qrels"),
    ],
)
def test_formal_loader_freezes_experiment_topic_and_qrels_identity(
    tmp_path,
    section,
    key,
    value,
    message,
):
    source = yaml.safe_load(
        (REPO_ROOT / "configs" / "det_sparse_v1.yaml").read_text(encoding="utf-8")
    )
    source[section][key] = value
    path = _write_synthetic_config(tmp_path, source)

    with pytest.raises(ValueError, match=message):
        load_det_sparse_config(path)


def test_formal_loader_freezes_output_and_config_locations(tmp_path):
    source = yaml.safe_load(
        (REPO_ROOT / "configs" / "det_sparse_v1.yaml").read_text(encoding="utf-8")
    )
    source["experiment"]["output_dir"] = "outputs/fresh-budget-copy"
    path = _write_synthetic_config(tmp_path, source)
    with pytest.raises(ValueError, match="experiment.output_dir"):
        load_det_sparse_config(path)

    outside = tmp_path / "repo" / "copied.yaml"
    outside.write_text(
        (REPO_ROOT / "configs" / "det_sparse_v1.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config path"):
        load_det_sparse_config(outside)


def test_formal_loader_rejects_unknown_policy_looking_keys(tmp_path):
    source = yaml.safe_load(
        (REPO_ROOT / "configs" / "det_sparse_v1.yaml").read_text(encoding="utf-8")
    )
    source["retrieval"]["retries"] = 10
    path = _write_synthetic_config(tmp_path, source)
    with pytest.raises(ValueError, match="retrieval keys.*unknown=retries"):
        load_det_sparse_config(path)
