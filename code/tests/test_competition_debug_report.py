"""Contract tests for the read-only competition debug-report data model."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from trec_rag.competition_debug_report import load_debug_report_data


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _write_debug_run(tmp_path: Path) -> tuple[Path, Path]:
    """Write a deliberately small sealed export with hostile stored source text."""
    (tmp_path / "AGENTS.md").write_text("fixture root\n", encoding="utf-8")
    topics_path = tmp_path / "topics.jsonl"
    narrative = 'Original <narrative> & "quotes"'
    topics_path.write_text(
        "\n".join(
            (
                json.dumps({"id": "rag2026-0", "narrative": narrative}),
                json.dumps({"id": "rag2026-1", "narrative": "Second official narrative"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "retrieval.yaml"
    config_path.write_text(
        """schema_version: facet_pilot_config_v1
experiment:
  id: debug-fixture
topics:
  path: topics.jsonl
retrieval:
  index: fixture-index
  cache_dir: cache/retrieval
  query_sources: [original, subnarrative]
  candidate_depth_per_query: 10
reranking:
  model: mixedbread-ai/mxbai-rerank-base-v2
  score_cache_dir: cache/reranker
  device: cpu
  rerank_depth_per_query: 5
  candidate_pool_depth: 5
  selection_policy: round_robin_subnarrative_coverage
nuggets:
  evidence_budget_per_subnarrative: 2
  maximum_claims_per_subnarrative: 2
  maximum_supporting_documents_per_claim: 1
""",
        encoding="utf-8",
    )
    output = tmp_path / "outputs" / "debug-fixture"
    output.mkdir(parents=True)
    (output / "retrieval_export_manifest.json").write_bytes(
        _json_bytes(
            {
                "schema_version": "retrieval_export_manifest_v2",
                "run_id": "debug-fixture",
                "selected_topic_ids": ["rag2026-0"],
                "artifacts": {},
            }
        )
    )

    topic_root = output / "rag2026-0"
    (topic_root / "scoring").mkdir(parents=True)
    subnarrative = 'Facet <scope> & "query"'
    query = "literal facet query - no rewrite"
    decomposition = {
        "schema_version": "facet_pilot_v2",
        "topic_id": "rag2026-0",
        "narrative": narrative,
        "narrative_sha256": _sha256(narrative),
        "source_sha256": "a" * 64,
        "queries": [
            {
                "topic_id": "rag2026-0",
                "variant_name": "original",
                "query_text": narrative,
                "source_type": "original_topic",
            },
            {
                "topic_id": "rag2026-0",
                "variant_name": "facet-1",
                "query_text": query,
                "source_type": "subnarrative_bm25",
            },
        ],
        "plan": {
            "topic_id": "rag2026-0",
            "subnarratives": [{"subnarrative": subnarrative, "bm25_queries": [query]}],
        },
        "subnarratives": [
            {
                "topic_id": "rag2026-0",
                "subnarrative_id": "subnarrative-1",
                "text": subnarrative,
                "bm25_queries": [query],
                "semantic_query_sha256": _sha256(subnarrative),
                "bm25_query_sha256s": [_sha256(query)],
            }
        ],
    }
    (topic_root / "decomposition.json").write_bytes(_json_bytes(decomposition))
    selection = {
        "schema_version": "facet_pilot_selection_v2",
        "topic_id": "rag2026-0",
        "selected_order": ["doc-original", "doc-facet"],
        "union_pool": [
            {"docid": "doc-original", "first_seen_lane": "original", "memberships": ["original"]},
            {"docid": "doc-both", "first_seen_lane": "original", "memberships": ["original", "facet-1"]},
            {"docid": "doc-facet", "first_seen_lane": "facet-1", "memberships": ["facet-1"]},
            {"docid": "doc-discarded", "first_seen_lane": "facet-1", "memberships": ["facet-1"]},
        ],
        "memberships": [
            {"docid": "doc-original", "lanes": [{"lane_name": "original", "aggregate_rank": 1, "aggregate_score": 9.0, "bm25_rank": 1, "bm25_score": 8.0}]},
            {"docid": "doc-facet", "lanes": [{"lane_name": "facet-1", "aggregate_rank": 1, "aggregate_score": 7.0, "bm25_rank": 2, "bm25_score": 6.0}]},
        ],
        "trace": [
            {"slot": 1, "docid": "doc-original", "lane_name": "original", "lane_rank": 1, "lane_exhausted": False, "action": "selected"},
            {"slot": 2, "docid": "doc-facet", "lane_name": "facet-1", "lane_rank": 1, "lane_exhausted": False, "action": "selected"},
        ],
    }
    (topic_root / "scoring" / "selection.json").write_bytes(_json_bytes(selection))
    selected_rows = [
        {
            "topic_id": "rag2026-0", "docid": "doc-original", "selection_rank": 1,
            "selected_from_lane": "original", "selected_from_lane_rank": 1,
            "text": "Original selected source.",
            "text_sha256": _sha256("Original selected source."),
        },
        {
            "topic_id": "rag2026-0", "docid": "doc-facet", "selection_rank": 2,
            "selected_from_lane": "facet-1", "selected_from_lane_rank": 1,
            "text": "Facet selected source with <unsafe> text.",
            "text_sha256": _sha256("Facet selected source with <unsafe> text."),
        },
    ]
    (topic_root / "scoring" / "selected_documents.jsonl").write_bytes(
        b"".join(_json_bytes(row) for row in selected_rows)
    )
    return config_path, output


def test_topic_order_and_identity_are_loaded_from_official_topics(tmp_path: Path) -> None:
    config_path, _output = _write_debug_run(tmp_path)

    data = load_debug_report_data(config_path)

    assert [topic.topic_id for topic in data.topics] == ["rag2026-0"]
    topic = data.topics[0]
    assert topic.narrative == 'Original <narrative> & "quotes"'
    assert topic.subnarratives[0].text == 'Facet <scope> & "query"'
    assert topic.subnarratives[0].bm25_queries == ("literal facet query - no rewrite",)


def test_identity_rejects_a_decomposition_topic_mismatch(tmp_path: Path) -> None:
    config_path, output = _write_debug_run(tmp_path)
    path = output / "rag2026-0" / "decomposition.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["topic_id"] = "rag2026-1"
    path.write_bytes(_json_bytes(value))

    with pytest.raises(ValueError, match="decomposition.*topic"):
        load_debug_report_data(config_path)


def test_identity_rejects_selected_document_absent_from_stored_order(tmp_path: Path) -> None:
    config_path, output = _write_debug_run(tmp_path)
    path = output / "rag2026-0" / "scoring" / "selected_documents.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["docid"] = "doc-not-stored"
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))

    with pytest.raises(ValueError, match="selected.*order"):
        load_debug_report_data(config_path)


def test_new_document_classification_uses_union_membership_and_selected_excerpts(tmp_path: Path) -> None:
    config_path, _output = _write_debug_run(tmp_path)

    data = load_debug_report_data(config_path)
    documents = {row.docid: row for row in data.topics[0].new_documents}

    assert documents["doc-facet"].is_new is True
    assert documents["doc-discarded"].is_new is True
    assert documents["doc-both"].is_new is False
    assert documents["doc-discarded"].excerpt is None
    assert documents["doc-facet"].excerpt == "Facet selected source with <unsafe> text."
