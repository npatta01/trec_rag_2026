import json
import re

import pytest

from trec_rag.det_sparse_config import load_det_sparse_config
from trec_rag.det_sparse_preflight import (
    build_preflight,
    load_selected_topics,
    validate_preflight,
)
from trec_rag.query_analyzer import AnalyzerFingerprint, AnalyzedQuery, stable_unique


class Analyzer:
    def __init__(self, fingerprint):
        self._fingerprint = fingerprint

    @property
    def fingerprint(self):
        return self._fingerprint

    def analyze(self, text):
        tokens = tuple(re.findall(r"[^\W\d_]+", text.lower()))
        return AnalyzedQuery(tokens, stable_unique(tokens), self._fingerprint)


def _config(tmp_path):
    topics = tmp_path / "topics.tsv"
    topics.write_text(
        "144\tLOCKED narrative must not be selected.\n"
        "200\tshared  payment context: what improves privacy controls?\n"
        "225\tshared climate context: how do cities reduce heat?\n"
        "707\tshared health context: which policies improve access?\n"
        "897\tshared transport context: where do delays occur?\n",
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.txt"
    tracked_path = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "configs"
        / "det_sparse_v1.yaml"
    )
    config = load_det_sparse_config(tracked_path)
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
    fingerprint_sha = __import__("hashlib").sha256(
        json.dumps(
            fingerprint.to_dict(), separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    config = __import__("dataclasses").replace(
        config,
        root_dir=tmp_path,
        topics_path=topics,
        analyzer=__import__("dataclasses").replace(
            config.analyzer,
            expected_fingerprint_sha256=fingerprint_sha,
        ),
        evaluation=__import__("dataclasses").replace(
            config.evaluation,
            qrels=qrels,
        ),
    )
    return config, Analyzer(fingerprint), qrels


def test_selective_loader_returns_only_frozen_nonlocked_ids(tmp_path):
    config, _analyzer, _qrels = _config(tmp_path)

    rows = load_selected_topics(config)

    assert [row.topic.id for row in rows] == ["200", "225", "707", "897"]
    assert all("LOCKED" not in row.topic.narrative for row in rows)
    assert rows[0].topic.narrative.startswith("shared  payment")


def test_offline_preflight_is_create_only_hash_sealed_and_never_reads_qrels(tmp_path):
    config, analyzer, qrels = _config(tmp_path)
    output = tmp_path / "preflight"
    assert not qrels.exists()

    result = build_preflight(
        config,
        query_analyzer=analyzer,
        output_dir=output,
        source_provenance={
            "commit": "a" * 40,
            "tree": "b" * 40,
            "source_tree_clean": True,
        },
    )

    assert result.valid
    assert result.planned_base_requests <= 20
    assert result.derived_max_total_requests <= 36
    assert result.hard_external_request_ceiling == 36
    assert not qrels.exists()
    metadata = json.loads((output / "_preflight.json").read_text())
    assert metadata["external_calls"] == 0
    assert metadata["model_calls"] == 0
    assert metadata["reranker_calls"] == 0
    assert metadata["qrels_opened"] is False
    assert metadata["locked_planner_topic_ids_requested"] is False
    assert metadata["mechanical_valid"] is True
    assert metadata["fallback_topic_ids"] == []
    assert all(
        row["derived_max_unique_requests"] <= 9
        for row in metadata["request_projections"]
    )
    assert metadata["derived_max_total_unique_requests"] == sum(
        row["derived_max_unique_requests"]
        for row in metadata["request_projections"]
    )
    registry = json.loads((output / "base_query_registry.json").read_text())
    topic_200_rows = [row for row in registry if row["topic_id"] == "200"]
    assert any(
        set(row["logical_variant_aliases"])
        == {"det_sparse_v1:original", "det_sparse_v1:facet:f02"}
        for row in topic_200_rows
    )
    assert all(
        plan.query_variants()[0].variant_name == "det_sparse_v1:original"
        for plan in result.plans
    )
    validate_preflight(config, output)
    with pytest.raises(FileExistsError):
        build_preflight(
            config,
            query_analyzer=analyzer,
            output_dir=output,
            source_provenance={},
        )


def test_preflight_records_fallback_and_returns_invalid_without_partial_facets(tmp_path):
    config, analyzer, _qrels = _config(tmp_path)
    source = config.topics_path.read_text(encoding="utf-8")
    config.topics_path.write_text(
        source.replace(
            "shared  payment context: what improves privacy controls?",
            "tiny: what improves privacy controls?",
        ),
        encoding="utf-8",
    )

    result = build_preflight(
        config,
        query_analyzer=analyzer,
        output_dir=tmp_path / "invalid",
        source_provenance={},
    )

    assert not result.valid
    failed = result.plans[0]
    assert failed.status == "fallback"
    assert failed.facets == ()
    assert [query.variant_name for query in failed.query_variants()] == [
        "det_sparse_v1:original"
    ]
    with pytest.raises(ValueError, match="mechanically invalid"):
        validate_preflight(config, result.output_dir)
