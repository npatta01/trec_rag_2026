"""Contract tests for the read-only competition debug-report data model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
import html
import json
from pathlib import Path
import re
import shutil
import socket
import zipfile

import pytest

import trec_rag.competition_debug_report as debug_report
from trec_rag.competition_debug_report import (
    load_debug_report_data,
    render_debug_report,
)


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
                "variant_name": "facet:subnarrative-1:q1",
                "query_text": query,
                "source_type": "generated_subnarrative_bm25",
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
            {"docid": "doc-both", "first_seen_lane": "original", "memberships": ["original", "facet:subnarrative-1:text"]},
            {"docid": "doc-facet", "first_seen_lane": "facet:subnarrative-1:text", "memberships": ["facet:subnarrative-1:text"]},
            {"docid": "doc-discarded", "first_seen_lane": "facet:subnarrative-1:text", "memberships": ["facet:subnarrative-1:text"]},
        ],
        "memberships": [
            {"docid": "doc-original", "lanes": [{"lane_name": "original", "aggregate_rank": 1, "aggregate_score": 9.0, "bm25_rank": 1, "bm25_score": 8.0}]},
            {"docid": "doc-facet", "lanes": [{"lane_name": "facet:subnarrative-1:text", "aggregate_rank": 1, "aggregate_score": 7.0, "bm25_rank": 2, "bm25_score": 6.0}]},
        ],
        "trace": [
            {"slot": 1, "docid": "doc-original", "lane_name": "original", "lane_rank": 1, "lane_exhausted": False, "action": "selected"},
            {"slot": 2, "docid": "doc-facet", "lane_name": "facet:subnarrative-1:text", "lane_rank": 1, "lane_exhausted": False, "action": "selected"},
        ],
    }
    (topic_root / "scoring" / "selection.json").write_bytes(_json_bytes(selection))
    selected_rows = [
        {
            "topic_id": "rag2026-0", "docid": "doc-original", "selection_rank": 1,
            "selected_from_lane": "original", "selected_from_lane_rank": 1,
            "text": "Intro. Exact evidence sentence. Tail.",
            "text_sha256": _sha256("Intro. Exact evidence sentence. Tail."),
        },
        {
            "topic_id": "rag2026-0", "docid": "doc-facet", "selection_rank": 2,
            "selected_from_lane": "facet:subnarrative-1:text", "selected_from_lane_rank": 1,
            "text": "Facet selected source with <unsafe> text.",
            "text_sha256": _sha256("Facet selected source with <unsafe> text."),
        },
    ]
    (topic_root / "scoring" / "selected_documents.jsonl").write_bytes(
        b"".join(_json_bytes(row) for row in selected_rows)
    )
    audit = {
        "schema_version": "facet_pilot_v2",
        "topic_id": "rag2026-0",
        "narrative_sha256": _sha256(narrative),
        "decomposition_source_sha256": "a" * 64,
        "requested_depth": 10,
        "lanes": [
            {
                "lane_name": "original", "subnarrative_id": None,
                "bm25_query_sha256": _sha256(narrative),
                "semantic_query_sha256": _sha256(narrative),
                "returned_count": 1, "retained_count": 1,
                "candidates": [{"docid": "doc-original", "bm25_rank": 1, "bm25_score": 8.0, "text_sha256": _sha256("Intro. Exact evidence sentence. Tail.")}],
            },
            {
                "lane_name": "facet:subnarrative-1:text", "subnarrative_id": "subnarrative-1",
                "bm25_query_sha256": _sha256(subnarrative),
                "semantic_query_sha256": _sha256(subnarrative),
                "returned_count": 1, "retained_count": 1,
                "candidates": [{"docid": "doc-facet", "bm25_rank": 1, "bm25_score": 6.0, "text_sha256": _sha256("Facet selected source with <unsafe> text.")}],
            },
        ],
    }
    (topic_root / "retrieval").mkdir()
    (topic_root / "retrieval" / "audit.json").write_bytes(_json_bytes(audit))
    _write_task_2_artifacts(output, topic_root, narrative, subnarrative, query)
    return config_path, output


def _write_rag_run(tmp_path: Path, retrieval_output: Path) -> tuple[Path, Path, str]:
    narrative = 'Original <narrative> & "quotes"'
    queries_path = tmp_path / "rag_queries.tsv"
    queries_path.write_text(f"rag2026-0\t{narrative}\n", encoding="utf-8")
    rag_config = tmp_path / "rag.yaml"
    rag_config.write_text(
        """schema_version: competition_rag_config_v1
experiment:
  id: rag-debug-fixture
  output_dir: outputs/rag-debug-fixture
  mode: create
submission:
  team_id: fixture-team
  run_desc: Fixture answer generation.
inputs:
  queries: rag_queries.tsv
  run: outputs/debug-fixture/r_output_trec_rag_2026.tsv
  documents: outputs/debug-fixture/retrieval_with_text.jsonl.zip
  archive_member: null
  topic_ids: [rag2026-0]
retrieval:
  top_k: null
  max_document_words: 1000
generation:
  type: openrouter
  api_base: https://example.invalid/api/v1
  api_key_env: UNUSED_FIXTURE_KEY
  model: fixture/model
  reasoning_effort: medium
  temperature: null
  max_tokens: 1000
  timeout_seconds: 30
  transport_max_attempts: 1
  concurrency: 1
""",
        encoding="utf-8",
    )
    rag_output = tmp_path / "outputs" / "rag-debug-fixture" / "rag_output_trec_rag_2026.jsonl"
    rag_output.parent.mkdir()
    record = {
        "metadata": {
            "team_id": "fixture-team",
            "narrative_id": "rag2026-0",
            "narrative": narrative,
            "run_id": "rag-debug-fixture",
            "run_desc": "Fixture answer generation.",
        },
        "references": ["doc-original"],
        "answer": [
            {"text": "First supported answer.", "citations": [0]},
            {"text": "Second detail.", "citations": [0]},
        ],
    }
    body = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    rag_output.write_bytes(body)
    assert retrieval_output == tmp_path / "outputs" / "debug-fixture"
    return rag_config, rag_output, sha256(body).hexdigest()


def _add_second_debug_topic(tmp_path: Path, output: Path) -> None:
    old_narrative = 'Original <narrative> & "quotes"'
    new_narrative = "Second official narrative"
    old_hash = _sha256(old_narrative)
    new_hash = _sha256(new_narrative)
    source_root = output / "rag2026-0"
    target_root = output / "rag2026-1"
    shutil.copytree(source_root, target_root)

    def rewrite(value: object) -> object:
        if isinstance(value, str):
            return {
                "rag2026-0": "rag2026-1",
                old_narrative: new_narrative,
                old_hash: new_hash,
            }.get(value, value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    for path in target_root.rglob("*.json"):
        path.write_bytes(_json_bytes(rewrite(json.loads(path.read_text(encoding="utf-8")))))
    for path in target_root.rglob("*.jsonl"):
        rows = [rewrite(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_bytes(b"".join(_json_bytes(row) for row in rows))
    nuggets_path = target_root / "canonical" / "canonical-nuggets.jsonl"
    nugget_manifest_path = nuggets_path.with_name("canonical-nugget-manifest.json")
    nugget_manifest = json.loads(nugget_manifest_path.read_text(encoding="utf-8"))
    nugget_manifest["canonical_nuggets_sha256"] = sha256(nuggets_path.read_bytes()).hexdigest()
    nugget_manifest_path.write_bytes(_json_bytes(nugget_manifest))

    run_path = output / "r_output_trec_rag_2026.tsv"
    run_path.write_bytes(
        run_path.read_bytes() + b"rag2026-1 Q0 doc-original 1 1 debug-fixture\n"
    )
    provenance_path = output / "retrieval_provenance.jsonl"
    first_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    second_provenance = rewrite(first_provenance)
    provenance_path.write_bytes(
        _json_bytes(first_provenance) + _json_bytes(second_provenance)
    )
    archive_path = output / "retrieval_with_text.jsonl.zip"
    with zipfile.ZipFile(archive_path) as archive:
        first_archive = json.loads(archive.read("retrieval_with_text.jsonl"))
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "retrieval_with_text.jsonl",
            _json_bytes(first_archive) + _json_bytes(rewrite(first_archive)),
        )
    manifest_path = output / "retrieval_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_topic_ids"] = ["rag2026-0", "rag2026-1"]
    manifest["topic_depths"]["rag2026-1"] = {"official": 1, "candidate_pool": 2}
    manifest["official_row_count"] = 2
    for artifact_name in (
        "r_output_trec_rag_2026.tsv",
        "retrieval_provenance.jsonl",
        "retrieval_with_text.jsonl.zip",
    ):
        body = (output / artifact_name).read_bytes()
        manifest["artifacts"][artifact_name] = {
            "bytes": len(body),
            "sha256": sha256(body).hexdigest(),
        }
    manifest_path.write_bytes(_json_bytes(manifest))


def _extend_rag_run_with_forged_second_narrative(
    tmp_path: Path, rag_config: Path, rag_output: Path
) -> None:
    (tmp_path / "rag_queries.tsv").write_text(
        'rag2026-0\tOriginal <narrative> & "quotes"\n'
        "rag2026-1\tForged second narrative\n",
        encoding="utf-8",
    )
    rag_config.write_text(
        rag_config.read_text(encoding="utf-8").replace(
            "topic_ids: [rag2026-0]", "topic_ids: [rag2026-0, rag2026-1]"
        ),
        encoding="utf-8",
    )
    first = json.loads(rag_output.read_text(encoding="utf-8"))
    second = json.loads(json.dumps(first))
    second["metadata"]["narrative_id"] = "rag2026-1"
    second["metadata"]["narrative"] = "Forged second narrative"
    rag_output.write_bytes(_json_bytes(first) + _json_bytes(second))


def _write_task_2_artifacts(
    output: Path,
    topic_root: Path,
    narrative: str,
    subnarrative: str,
    query: str,
) -> None:
    source = "Intro. Exact evidence sentence. Tail."
    start = 7
    end = 31
    assert source[start:end] == "Exact evidence sentence."
    scores = [
        {
            "topic_id": "rag2026-0",
            "lane_name": "subnarrative:subnarrative-1",
            "semantic_query_sha256": _sha256(subnarrative),
            "docid": "doc-facet",
            "bm25_rank": 2,
            "bm25_score": 6.0,
            "aggregate_rank": 2,
            "aggregate_score": 2.5,
            "long_document_raw_logit": 1.5,
            "weighted_passage_raw_logit": 2.0,
            "within_document_span_support": 1,
            "winning_passages": [
                {"chunk_index": 0, "start_char": 0, "end_char": 5, "raw_logit": 2.0, "weighted_rank": 1}
            ],
            "score_representation": "raw_logits",
            "text_sha256": _sha256("Facet selected source with <unsafe> text."),
            "selection_rank": 2,
            "subnarrative_id": "subnarrative-1",
            "bm25_queries": [query],
            "bm25_query_sha256s": [_sha256(query)],
            "downstream_only": True,
        },
        {
            "topic_id": "rag2026-0",
            "lane_name": "subnarrative:subnarrative-1",
            "semantic_query_sha256": _sha256(subnarrative),
            "docid": "doc-original",
            "bm25_rank": 1,
            "bm25_score": 8.0,
            "aggregate_rank": 1,
            "aggregate_score": 4.5,
            "long_document_raw_logit": 3.0,
            "weighted_passage_raw_logit": 3.5,
            "within_document_span_support": 1,
            "winning_passages": [
                {"chunk_index": 0, "start_char": start, "end_char": end, "raw_logit": 3.5, "weighted_rank": 1}
            ],
            "score_representation": "raw_logits",
            "text_sha256": _sha256(source),
            "selection_rank": 1,
            "subnarrative_id": "subnarrative-1",
            "bm25_queries": [query],
            "bm25_query_sha256s": [_sha256(query)],
            "downstream_only": True,
        },
    ]
    (topic_root / "scoring" / "selected_subnarrative_scores.jsonl").write_bytes(
        b"".join(_json_bytes(row) for row in scores)
    )

    evidence = {
        "candidate_nugget_id": "candidate-1",
        "candidate_kind": "exact_sentence",
        "text": "Exact evidence sentence.",
        "docid": "doc-original",
        "document_sha256": _sha256(source),
        "raw_logit": 3.5,
    }
    selection = {
        "schema_version": "subnarrative_selection_v1",
        "topic_id": "rag2026-0",
        "official_narrative": narrative,
        "official_narrative_sha256": _sha256(narrative),
        "subnarrative_id": "subnarrative-1",
        "subnarrative_text": subnarrative,
        "subnarrative_sha256": _sha256(subnarrative),
        "policy": {"budgets": [2], "precluster_limit": 20, "semantic_threshold": 0.92, "mmr_lambda": 0.7},
        "embedding_identity": {"model": "fixture-minilm"},
        "candidate_count": 1,
        "exact_group_count": 1,
        "precluster_count": 1,
        "semantic_cluster_count": 1,
        "clusters": [
            {
                "cluster_id": "cluster-1",
                "representative_candidate_nugget_id": "candidate-1",
                "representative_text": "Exact evidence sentence.",
                "representative_raw_logit": 3.5,
                "members": [evidence],
                "supports": [evidence],
                "support_document_count": 1,
            }
        ],
        "snapshots": [{"budget": 2, "cluster_ids": ["cluster-1"], "exhausted": True}],
    }
    canonical = topic_root / "canonical"
    canonical.mkdir()
    (canonical / "subnarrative-selections.jsonl").write_bytes(_json_bytes(selection))
    canonical_result = {
        "schema_version": "canonical_nugget_result_v1",
        "topic_id": "rag2026-0",
        "subnarrative_id": "subnarrative-1",
        "selected_budget": 2,
        "request_sha256": "b" * 64,
        "state": "complete",
        "nuggets": [
            {
                "canonical_nugget_id": "canonical-1",
                "nugget_kind": "model_claim",
                "claim_text": "The exact evidence is supported.",
                "evidence": [
                    {
                        "candidate_nugget_id": "candidate-1",
                        "candidate_kind": "exact_sentence",
                        "text": "Exact evidence sentence.",
                        "text_sha256": _sha256("Exact evidence sentence."),
                        "docid": "doc-original",
                        "document_sha256": _sha256(source),
                        "cluster_id": "cluster-1",
                    }
                ],
            }
        ],
        "metadata": {"fixture": True},
        "error": None,
    }
    nuggets_body = _json_bytes(canonical_result)
    (canonical / "canonical-nuggets.jsonl").write_bytes(nuggets_body)
    (canonical / "canonical-nugget-manifest.json").write_bytes(
        _json_bytes(
            {
                "schema_version": "canonical_nugget_manifest_v2",
                "selected_budget": 2,
                "selection_count": 1,
                "result_count": 1,
                "state_counts": {"complete": 1},
                "max_canonical_claims": 2,
                "max_supporting_documents_per_claim": 1,
                "request_sha256s": ["b" * 64],
                "canonical_nuggets_sha256": sha256(nuggets_body).hexdigest(),
            }
        )
    )

    run_body = b"rag2026-0 Q0 doc-original 1 1 debug-fixture\n"
    provenance_body = _json_bytes(
        {
            "topic_id": "rag2026-0",
            "docid": "doc-original",
            "rank": 1,
            "score": 1,
            "selection_rank": 1,
            "selected_from_lane": "original",
            "selected_from_lane_rank": 1,
            "memberships": [{"lane_name": "original", "aggregate_rank": 1, "aggregate_score": 9.0, "bm25_rank": 1, "bm25_score": 8.0}],
            "subnarrative_scores": [scores[1]],
            "nuggets": [{"canonical_nugget_id": "canonical-1", "subnarrative_id": "subnarrative-1"}],
            "source_seals": {"scoring_manifest_sha256": "c" * 64, "canonical_manifest_sha256": "d" * 64},
        }
    )
    (output / "r_output_trec_rag_2026.tsv").write_bytes(run_body)
    (output / "retrieval_provenance.jsonl").write_bytes(provenance_body)
    archive_path = output / "retrieval_with_text.jsonl.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "retrieval_with_text.jsonl",
            _json_bytes(
                {
                    "query": {"qid": "rag2026-0", "text": narrative},
                    "candidates": [{"docid": "doc-original", "rank": 1, "score": 1, "doc": source, "index": "fixture-index", "stage": "canonical_supported"}],
                }
            ),
        )
    artifacts = {
        "r_output_trec_rag_2026.tsv": run_body,
        "retrieval_provenance.jsonl": provenance_body,
        "retrieval_with_text.jsonl.zip": archive_path.read_bytes(),
    }
    (output / "retrieval_export_manifest.json").write_bytes(
        _json_bytes(
            {
                "schema_version": "retrieval_export_manifest_v2",
                "run_id": "debug-fixture",
                "selected_topic_ids": ["rag2026-0"],
                "score_semantics": "ordinal_selection_order",
                "topic_depths": {"rag2026-0": {"official": 1, "candidate_pool": 2}},
                "official_row_count": 1,
                "artifacts": {
                    name: {"bytes": len(body), "sha256": sha256(body).hexdigest()}
                    for name, body in artifacts.items()
                },
            }
        )
    )


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


@pytest.mark.parametrize(
    ("path_suffix", "mutation"),
    [
        (("subnarratives", 0, "topic_id"), "rag2026-1"),
        (("queries", 1, "variant_name"), "forged-lane"),
        (("queries", 1, "source_type"), "forged-source"),
    ],
)
def test_identity_rejects_cross_topic_subnarratives_and_forged_query_lanes(
    tmp_path: Path, path_suffix: tuple[object, ...], mutation: str
) -> None:
    config_path, output = _write_debug_run(tmp_path)
    path = output / "rag2026-0" / "decomposition.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    target: object = value
    for key in path_suffix[:-1]:
        target = target[key]  # type: ignore[index]
    target[path_suffix[-1]] = mutation  # type: ignore[index]
    path.write_bytes(_json_bytes(value))

    with pytest.raises(ValueError, match="decomposition.*(?:identity|queries)"):
        load_debug_report_data(config_path)


def test_selection_rejects_union_memberships_that_disagree_with_selected_provenance(
    tmp_path: Path,
) -> None:
    config_path, output = _write_debug_run(tmp_path)
    path = output / "rag2026-0" / "scoring" / "selection.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["union_pool"][2]["first_seen_lane"] = "original"
    value["union_pool"][2]["memberships"] = ["original"]
    path.write_bytes(_json_bytes(value))

    with pytest.raises(ValueError, match="selection.*membership"):
        load_debug_report_data(config_path)


def test_selection_rejects_trace_that_disagrees_with_selected_rank_provenance(
    tmp_path: Path,
) -> None:
    config_path, output = _write_debug_run(tmp_path)
    path = output / "rag2026-0" / "scoring" / "selection.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["trace"][1]["lane_rank"] = 2
    path.write_bytes(_json_bytes(value))

    with pytest.raises(ValueError, match="selection trace"):
        load_debug_report_data(config_path)


@pytest.mark.parametrize("field", ("topic_id", "narrative_sha256", "decomposition_source_sha256"))
def test_audit_hashes_require_current_topic_and_decomposition_identity(
    tmp_path: Path, field: str
) -> None:
    config_path, output = _write_debug_run(tmp_path)
    path = output / "rag2026-0" / "retrieval" / "audit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value[field] = "rag2026-1" if field == "topic_id" else "f" * 64
    path.write_bytes(_json_bytes(value))

    with pytest.raises(ValueError, match="retrieval audit identity"):
        load_debug_report_data(config_path)


def test_audit_hashes_reject_non_integer_candidate_rank(tmp_path: Path) -> None:
    config_path, output = _write_debug_run(tmp_path)
    path = output / "rag2026-0" / "retrieval" / "audit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["lanes"][0]["candidates"][0]["bm25_rank"] = True
    path.write_bytes(_json_bytes(value))

    with pytest.raises(ValueError, match="retrieval audit candidate identity"):
        load_debug_report_data(config_path)


def test_oversized_artifact_is_rejected_before_receipt_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, output = _write_debug_run(tmp_path)
    path = output / "rag2026-0" / "scoring" / "selection.json"
    path.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    original_open = Path.open

    def guarded_open(candidate: Path, *args: object, **kwargs: object):
        if candidate.resolve() == path.resolve():
            raise AssertionError("oversized artifact must not be opened for hashing")
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="bounded reader limit"):
        load_debug_report_data(config_path)


def test_passage_rankings_retain_stored_ranks_logits_and_exact_source_spans(
    tmp_path: Path,
) -> None:
    config_path, _output = _write_debug_run(tmp_path)

    topic = load_debug_report_data(config_path).topics[0]

    assert [row.aggregate_rank for row in topic.passage_rankings] == [1, 2]
    assert [row.docid for row in topic.passage_rankings] == ["doc-original", "doc-facet"]
    assert topic.passage_rankings[0].score_representation == "raw_logits"
    assert topic.passage_rankings[0].winning_passages[0].raw_logit == 3.5
    assert topic.passage_rankings[0].winning_passages[0].text == "Exact evidence sentence."


def test_nugget_join_uses_configured_budget_selected_clusters_and_evidence_docids(
    tmp_path: Path,
) -> None:
    config_path, _output = _write_debug_run(tmp_path)

    topic = load_debug_report_data(config_path).topics[0]

    assert [cluster.cluster_id for cluster in topic.evidence_clusters] == ["cluster-1"]
    assert topic.evidence_clusters[0].selected_budget == 2
    assert topic.canonical_nuggets[0].selected_budget == 2
    assert topic.canonical_nuggets[0].state == "complete"
    assert topic.canonical_nuggets[0].evidence[0].docid == "doc-original"
    assert topic.canonical_nuggets[0].evidence[0].cluster_id == "cluster-1"
    assert topic.canonical_nuggets[0].maximum_claims == 2
    assert topic.canonical_nuggets[0].maximum_supporting_documents == 1


def test_retrieval_projection_explains_selected_depth_versus_supported_depth(
    tmp_path: Path,
) -> None:
    config_path, _output = _write_debug_run(tmp_path)

    retrieval = load_debug_report_data(config_path).topics[0].retrieval_output

    assert retrieval.selected_pool_depth == 2
    assert retrieval.final_supported_depth == 1
    assert [row.docid for row in retrieval.documents] == ["doc-original"]
    assert retrieval.documents[0].text == "Intro. Exact evidence sentence. Tail."
    assert retrieval.documents[0].canonical_nugget_ids == ("canonical-1",)


def test_candidate_ledger_and_network_are_outside_the_bounded_report_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, output = _write_debug_run(tmp_path)
    forbidden = output / "rag2026-0" / "canonical" / "candidates.jsonl"
    forbidden.mkdir()
    original_open = Path.open
    original_read_bytes = Path.read_bytes

    def guarded_open(candidate: Path, *args: object, **kwargs: object):
        if candidate.resolve() == forbidden.resolve():
            raise AssertionError("candidate ledger must not be opened")
        return original_open(candidate, *args, **kwargs)

    def guarded_read_bytes(candidate: Path) -> bytes:
        if candidate.resolve() == forbidden.resolve():
            raise AssertionError("candidate ledger must not be read")
        return original_read_bytes(candidate)

    def reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("bounded report loading must not use the network")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(socket.socket, "connect", reject_network)

    assert load_debug_report_data(config_path).topics[0].retrieval_output.final_supported_depth == 1


def test_passage_ranking_rejects_offsets_outside_selected_document(tmp_path: Path) -> None:
    config_path, output = _write_debug_run(tmp_path)
    path = output / "rag2026-0" / "scoring" / "selected_subnarrative_scores.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["winning_passages"][0]["end_char"] = 10_000
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))

    with pytest.raises(ValueError, match="winning passage offsets are outside selected document"):
        load_debug_report_data(config_path)


def _rewrite_canonical_result(
    output: Path, mutate: Callable[[dict[str, object]], None],
) -> None:
    path = output / "rag2026-0" / "canonical" / "canonical-nuggets.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    mutate(row)
    body = _json_bytes(row)
    path.write_bytes(body)
    manifest_path = path.with_name("canonical-nugget-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canonical_nuggets_sha256"] = sha256(body).hexdigest()
    manifest["state_counts"] = {row["state"]: 1}
    manifest_path.write_bytes(_json_bytes(manifest))


@pytest.mark.parametrize("case", ("empty_with_nuggets", "complete_with_error", "wrong_kind"))
def test_canonical_state_rejects_semantically_impossible_results(
    tmp_path: Path, case: str
) -> None:
    config_path, output = _write_debug_run(tmp_path)

    def mutate(row: dict[str, object]) -> None:
        if case == "empty_with_nuggets":
            row["state"] = "empty"
        elif case == "complete_with_error":
            row["error"] = "provider failed"
        else:
            row["nuggets"][0]["nugget_kind"] = "extractive_fallback"  # type: ignore[index]

    _rewrite_canonical_result(output, mutate)

    with pytest.raises(ValueError, match="canonical nugget.*(?:state|kind)"):
        load_debug_report_data(config_path)


def test_canonical_request_identity_must_match_manifest_order(tmp_path: Path) -> None:
    config_path, output = _write_debug_run(tmp_path)

    def mutate(row: dict[str, object]) -> None:
        row["request_sha256"] = "e" * 64

    _rewrite_canonical_result(output, mutate)

    with pytest.raises(ValueError, match="canonical nugget.*request"):
        load_debug_report_data(config_path)


def _rewrite_export_artifact_receipt(output: Path, artifact_name: str) -> None:
    artifact = output / artifact_name
    body = artifact.read_bytes()
    manifest_path = output / "retrieval_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][artifact_name] = {
        "bytes": len(body),
        "sha256": sha256(body).hexdigest(),
    }
    manifest_path.write_bytes(_json_bytes(manifest))


def test_root_run_coverage_rejects_unmanifested_topic_rows(tmp_path: Path) -> None:
    config_path, output = _write_debug_run(tmp_path)
    run_path = output / "r_output_trec_rag_2026.tsv"
    run_path.write_bytes(
        run_path.read_bytes() + b"rag2026-extra Q0 doc-extra 1 1 debug-fixture\n"
    )
    _rewrite_export_artifact_receipt(output, run_path.name)
    manifest_path = output / "retrieval_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["official_row_count"] = 2
    manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises(ValueError, match="organizer run.*(?:coverage|manifest)"):
        load_debug_report_data(config_path)


def test_official_row_count_must_equal_trec_rows(tmp_path: Path) -> None:
    config_path, output = _write_debug_run(tmp_path)
    manifest_path = output / "retrieval_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["official_row_count"] = 2
    manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises(ValueError, match="official row count"):
        load_debug_report_data(config_path)


def test_canonical_fallback_rejects_an_empty_exact_evidence_set(tmp_path: Path) -> None:
    config_path, output = _write_debug_run(tmp_path)

    def mutate(row: dict[str, object]) -> None:
        row["state"] = "fallback_extractive"
        row["nuggets"] = []
        row["error"] = "provider failed"

    _rewrite_canonical_result(output, mutate)

    with pytest.raises(ValueError, match="fallback.*exact evidence"):
        load_debug_report_data(config_path)


def test_html_renderer_is_semantic_self_contained_and_escapes_hostile_source_text(
    tmp_path: Path,
) -> None:
    """A renderer change that omitted an escape or HTML landmark would fail here."""
    config_path, _output = _write_debug_run(tmp_path)
    loaded = load_debug_report_data(config_path)
    topic = loaded.topics[0]
    narrative = '<script>alert("narrative")</script>'
    claim = '</script><img src=x onerror=alert(1)>'
    hostile_nugget = replace(topic.canonical_nuggets[0], claim_text=claim)
    rankings = tuple(
        replace(
            topic.passage_rankings[0],
            docid=f"ranked-doc-{rank}",
            aggregate_rank=rank,
            winning_passages=(
                replace(topic.passage_rankings[0].winning_passages[0], text=f"passage {rank}"),
            ),
        )
        for rank in range(1, 8)
    )
    data = replace(
        loaded,
        topics=(
            replace(
                topic,
                narrative=narrative,
                passage_rankings=rankings,
                canonical_nuggets=(hostile_nugget,),
            ),
        ),
    )

    rendered = render_debug_report(data)

    assert rendered.count("<!doctype html>") == 1
    assert '<html lang="en">' in rendered
    assert 'name="viewport"' in rendered
    assert 'name="color-scheme"' in rendered
    assert "<main" in rendered and "<nav" in rendered
    assert '<section id="topic-literal-rag2026-0"' in rendered
    headings = (
        "Narrative",
        "Subnarratives",
        "New documents",
        "Selected documents",
        "Top passages",
        "Final selected nuggets",
        "Final retrieval",
        "Final RAG",
    )
    positions = [rendered.index(f">{heading}<") for heading in headings]
    assert positions == sorted(positions)
    assert "<caption>" in rendered
    assert 'scope="col"' in rendered
    assert ":focus-visible" in rendered
    assert "prefers-reduced-motion" in rendered
    assert "http://" not in rendered and "https://" not in rendered
    assert "<script" not in rendered
    assert '<link rel="stylesheet"' not in rendered
    assert "<img" not in rendered and "@font-face" not in rendered
    assert narrative not in rendered and claim not in rendered
    assert html.escape(narrative, quote=True) in rendered
    assert html.escape(claim, quote=True) in rendered


def test_html_renderer_discloses_passage_rankings_after_the_first_five(tmp_path: Path) -> None:
    """A renderer change that hides or truncates stored passages would fail here."""
    config_path, _output = _write_debug_run(tmp_path)
    loaded = load_debug_report_data(config_path)
    topic = loaded.topics[0]
    rankings = tuple(
        replace(topic.passage_rankings[0], docid=f"ranked-doc-{rank}", aggregate_rank=rank)
        for rank in range(1, 8)
    )

    rendered = render_debug_report(replace(loaded, topics=(replace(topic, passage_rankings=rankings),)))
    passage_section = rendered.split(
        '<section id="stage-literal-rag2026-0-top-passages"', 1
    )[1].split(
        '<section id="stage-literal-rag2026-0-final-selected-nuggets"', 1
    )[0]
    visible, remainder = passage_section.split('<details class="passage-remainder">', 1)

    for rank in range(1, 6):
        assert f"ranked-doc-{rank}" in visible
    for rank in range(6, 8):
        assert f"ranked-doc-{rank}" not in visible
        assert f"ranked-doc-{rank}" in remainder


def test_html_renderer_uses_disjoint_anchor_namespaces_for_valid_topic_id_collision(
    tmp_path: Path,
) -> None:
    """A hashed unsafe ID must not collide with a literal safe ID."""
    config_path, _output = _write_debug_run(tmp_path)
    loaded = load_debug_report_data(config_path)
    source = loaded.topics[0]
    hashed_topic = replace(source, topic_id="x!")
    literal_topic = replace(source, topic_id="topic-4038e65dabd89221")

    rendered = render_debug_report(replace(loaded, topics=(hashed_topic, literal_topic)))

    assert 'href="#topic-hash-4038e65dabd89221"' in rendered
    assert 'href="#topic-literal-topic-4038e65dabd89221"' in rendered
    assert '<section id="topic-hash-4038e65dabd89221"' in rendered
    assert '<section id="topic-literal-topic-4038e65dabd89221"' in rendered
    section_ids = re.findall(r'<section id="([^"]+)"', rendered)
    assert len(section_ids) == len(set(section_ids))


def test_rag_omission_is_visible_in_the_rendered_report(tmp_path: Path) -> None:
    config_path, _output = _write_debug_run(tmp_path)

    data = load_debug_report_data(config_path)
    rendered = render_debug_report(data)

    assert data.topics[0].rag_output is None
    assert "RAG output not supplied." in rendered


def test_standard_rag_config_loads_validated_answers_and_resolved_citations(
    tmp_path: Path,
) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)
    rag_config, _rag_output, output_sha256 = _write_rag_run(tmp_path, retrieval_output)

    data = load_debug_report_data(config_path, rag_config_path=rag_config)

    assert [topic.topic_id for topic in data.topics] == ["rag2026-0"]
    rag = data.topics[0].rag_output
    assert rag is not None
    assert rag.references == ("doc-original",)
    assert [item.text for item in rag.answer_items] == [
        "First supported answer.",
        "Second detail.",
    ]
    assert [item.citations for item in rag.answer_items] == [(0,), (0,)]
    assert [item.citation_docids for item in rag.answer_items] == [
        ("doc-original",),
        ("doc-original",),
    ]
    assert rag.word_count == 5
    assert rag.output_sha256 == output_sha256
    assert data.source_sha256s["rag/rag_output_trec_rag_2026.jsonl"] == output_sha256
    rendered = render_debug_report(data)
    assert "First supported answer." in rendered
    assert "citation 0 → doc-original" in rendered


@pytest.mark.parametrize("mismatch", ("run_path", "topic_narrative"))
def test_rag_compatibility_errors_before_output_is_written(
    tmp_path: Path, mismatch: str
) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)
    rag_config, _rag_output, _output_sha256 = _write_rag_run(tmp_path, retrieval_output)
    if mismatch == "run_path":
        text = rag_config.read_text(encoding="utf-8").replace(
            "outputs/debug-fixture/r_output_trec_rag_2026.tsv",
            "outputs/debug-fixture/retrieval_provenance.jsonl",
        )
        rag_config.write_text(text, encoding="utf-8")
    else:
        (tmp_path / "rag_queries.tsv").write_text(
            "rag2026-0\tDifferent official narrative\n", encoding="utf-8"
        )
    target = retrieval_output / "must-not-be-written.html"

    with pytest.raises(ValueError, match="RAG.*(?:retrieval|topic|narrative|compatible)"):
        debug_report.build_debug_report(
            config_path,
            rag_config_path=rag_config,
            output_path=target,
        )

    assert not target.exists()


def test_cli_emits_one_compact_stable_json_receipt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)
    rag_config, _rag_output, output_sha256 = _write_rag_run(tmp_path, retrieval_output)
    target = retrieval_output / "custom-debug-report.html"
    argv = [
        "--retrieval-config", str(config_path),
        "--rag-config", str(rag_config),
        "--topic", "rag2026-0",
        "--output", str(target),
    ]

    assert debug_report.main(argv) == 0
    first_stdout = capsys.readouterr().out
    assert debug_report.main(argv) == 0
    second_stdout = capsys.readouterr().out

    assert first_stdout == second_stdout
    assert first_stdout.endswith("\n") and first_stdout.count("\n") == 1
    assert " " not in first_stdout
    receipt = json.loads(first_stdout)
    assert set(receipt) == {
        "schema_version",
        "output_path",
        "topic_ids",
        "rag_included",
        "source_sha256s",
    }
    assert list(receipt) == sorted(receipt)
    assert receipt["output_path"] == str(target.resolve())
    assert receipt["topic_ids"] == ["rag2026-0"]
    assert receipt["rag_included"] is True
    assert list(receipt["source_sha256s"]) == sorted(receipt["source_sha256s"])
    assert receipt["source_sha256s"]["rag/rag_output_trec_rag_2026.jsonl"] == output_sha256
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_atomic_build_preserves_existing_output_when_rendering_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)
    target = retrieval_output / "competition_debug_report.html"
    target.write_text("existing report\n", encoding="utf-8")

    def fail_render(_data: object) -> str:
        raise RuntimeError("forced render failure")

    monkeypatch.setattr(debug_report, "render_debug_report", fail_render)

    with pytest.raises(RuntimeError, match="forced render failure"):
        debug_report.build_debug_report(config_path)

    assert target.read_text(encoding="utf-8") == "existing report\n"
    assert list(retrieval_output.glob(".competition_debug_report.html.*.tmp")) == []


def test_atomic_build_restores_existing_output_when_directory_fsync_fails_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)
    target = retrieval_output / "competition_debug_report.html"
    previous = b"exact previous report bytes\n"
    target.write_bytes(previous)
    real_fsync_directory = debug_report._fsync_directory
    failed = False

    def fail_once_after_replacement(path: Path) -> None:
        nonlocal failed
        if not failed and target.read_bytes() != previous:
            failed = True
            raise OSError("forced directory fsync failure after replacement")
        real_fsync_directory(path)

    monkeypatch.setattr(debug_report, "_fsync_directory", fail_once_after_replacement)

    with pytest.raises(OSError, match="forced directory fsync failure after replacement"):
        debug_report.build_debug_report(config_path)

    assert failed is True
    assert target.read_bytes() == previous
    assert list(retrieval_output.glob(".competition_debug_report.html.*.tmp")) == []
    assert list(retrieval_output.glob(".competition_debug_report.html.*.bak")) == []


def test_atomic_build_preserves_named_recovery_when_rollback_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)
    target = retrieval_output / "competition_debug_report.html"
    previous = b"exact recovery bytes\n"
    target.write_bytes(previous)
    real_fsync_directory = debug_report._fsync_directory
    real_replace = debug_report.os.replace
    triggered = False
    target_replacements = 0

    def fail_once_after_replacement(path: Path) -> None:
        nonlocal triggered
        if not triggered and target.read_bytes() != previous:
            triggered = True
            raise OSError("triggering post-replace directory fsync failure")
        real_fsync_directory(path)

    def fail_rollback_replace(source: Path, destination: Path) -> None:
        nonlocal target_replacements
        if Path(destination) == target:
            target_replacements += 1
            if target_replacements == 2:
                raise OSError("rollback replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(debug_report, "_fsync_directory", fail_once_after_replacement)
    monkeypatch.setattr(debug_report.os, "replace", fail_rollback_replace)

    with pytest.raises(RuntimeError) as raised:
        debug_report.build_debug_report(config_path)

    recovery = list(retrieval_output.glob(".competition_debug_report.html.*.bak"))
    assert triggered is True
    assert target.read_bytes() != previous
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == previous
    assert str(recovery[0]) in str(raised.value)
    assert "triggering post-replace directory fsync failure" in str(raised.value)
    assert "rollback replace failure" in str(raised.value)
    assert list(retrieval_output.glob(".competition_debug_report.html.*.tmp")) == []


def test_atomic_success_directory_fsyncs_after_backup_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)
    target = retrieval_output / "competition_debug_report.html"
    previous = b"previous report\n"
    target.write_bytes(previous)
    real_fsync_directory = debug_report._fsync_directory
    synced_after_cleanup = False

    def observe_cleanup_sync(path: Path) -> None:
        nonlocal synced_after_cleanup
        backups = list(retrieval_output.glob(".competition_debug_report.html.*.bak"))
        if target.read_bytes() != previous and not backups:
            synced_after_cleanup = True
        real_fsync_directory(path)

    monkeypatch.setattr(debug_report, "_fsync_directory", observe_cleanup_sync)

    debug_report.build_debug_report(config_path)

    assert synced_after_cleanup is True
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert list(retrieval_output.glob(".competition_debug_report.html.*.tmp")) == []
    assert list(retrieval_output.glob(".competition_debug_report.html.*.bak")) == []


def test_atomic_cleanup_preserves_recovery_when_first_recovery_copy_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)
    target = retrieval_output / "competition_debug_report.html"
    previous = b"old bytes held through cleanup recovery\n"
    target.write_bytes(previous)
    real_fsync_directory = debug_report._fsync_directory
    real_fsync = debug_report.os.fsync
    cleanup_sync_failed = False
    recovery_copy_failed = False

    def fail_cleanup_directory_sync_once(path: Path) -> None:
        nonlocal cleanup_sync_failed
        backups = list(retrieval_output.glob(".competition_debug_report.html.*.bak"))
        if not cleanup_sync_failed and target.read_bytes() != previous and not backups:
            cleanup_sync_failed = True
            raise OSError("primary backup cleanup directory fsync failure")
        real_fsync_directory(path)

    def fail_first_recovery_copy_fsync(descriptor: int) -> None:
        nonlocal recovery_copy_failed
        descriptor_path = Path(debug_report.os.readlink(f"/proc/self/fd/{descriptor}"))
        if (
            cleanup_sync_failed
            and not recovery_copy_failed
            and descriptor_path.suffix == ".bak"
        ):
            recovery_copy_failed = True
            raise OSError("recovery backup file fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(debug_report, "_fsync_directory", fail_cleanup_directory_sync_once)
    monkeypatch.setattr(debug_report.os, "fsync", fail_first_recovery_copy_fsync)

    with pytest.raises(RuntimeError) as raised:
        debug_report.build_debug_report(config_path)

    recovery = list(retrieval_output.glob(".competition_debug_report.html.*.bak"))
    assert cleanup_sync_failed is True
    assert recovery_copy_failed is True
    assert target.read_bytes() != previous
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == previous
    assert str(recovery[0]) in str(raised.value)
    assert "primary backup cleanup directory fsync failure" in str(raised.value)
    assert "recovery backup file fsync failure" in str(raised.value)
    assert list(retrieval_output.glob(".competition_debug_report.html.*.tmp")) == []


def test_atomic_build_restores_existing_output_when_backup_unlink_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)
    target = retrieval_output / "competition_debug_report.html"
    previous = b"previous report before unlink failure\n"
    target.write_bytes(previous)
    real_unlink = Path.unlink
    failed = False

    def fail_backup_unlink_once(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal failed
        if not failed and path.suffix == ".bak":
            failed = True
            raise OSError("forced backup unlink failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_backup_unlink_once)

    with pytest.raises(OSError, match="forced backup unlink failure"):
        debug_report.build_debug_report(config_path)

    assert failed is True
    assert target.read_bytes() == previous
    assert list(retrieval_output.glob(".competition_debug_report.html.*.tmp")) == []
    assert list(retrieval_output.glob(".competition_debug_report.html.*.bak")) == []


def test_build_defaults_to_retrieval_output_directory(tmp_path: Path) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)

    receipt = debug_report.build_debug_report(config_path)

    assert receipt.output_path == (retrieval_output / "competition_debug_report.html").resolve()
    assert receipt.rag_included is False


def test_output_override_cannot_replace_a_sealed_source_artifact(tmp_path: Path) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)
    run_path = retrieval_output / "r_output_trec_rag_2026.tsv"
    original = run_path.read_bytes()

    with pytest.raises(ValueError, match="report output.*(?:HTML|source|artifact)"):
        debug_report.build_debug_report(config_path, output_path=run_path)

    assert run_path.read_bytes() == original


def test_topic_subset_still_rejects_rag_narrative_drift_in_an_unrendered_topic(
    tmp_path: Path,
) -> None:
    config_path, retrieval_output = _write_debug_run(tmp_path)
    _add_second_debug_topic(tmp_path, retrieval_output)
    rag_config, rag_output, _output_sha256 = _write_rag_run(tmp_path, retrieval_output)
    _extend_rag_run_with_forged_second_narrative(tmp_path, rag_config, rag_output)
    target = retrieval_output / "subset-report.html"

    with pytest.raises(ValueError, match="RAG topic narrative"):
        debug_report.build_debug_report(
            config_path,
            rag_config_path=rag_config,
            topic_ids=["rag2026-0"],
            output_path=target,
        )

    assert not target.exists()
