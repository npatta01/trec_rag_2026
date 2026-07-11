from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from trec_rag.det_sparse_config import load_det_sparse_config
from trec_rag.det_sparse_freeze import (
    assess_pilot,
    derive_frozen_ungraded_diagnostics,
    evaluate_frozen_arms,
    validate_evaluation_freeze,
)
from trec_rag.det_sparse_ledger import RawTransportResponse
from trec_rag.det_sparse_preflight import build_preflight
from trec_rag.det_sparse_run import (
    _require_external_authorization,
    execute_det_sparse_run,
)
from trec_rag.det_sparse_transport import TRANSPORT_VERSION
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
    "commit": "a" * 40,
    "tree": "b" * 40,
    "source_tree_clean": True,
}
RUNTIME_BOUNDARY = {
    "transport_version": TRANSPORT_VERSION,
    "transport_class": "trec_rag.det_sparse_transport.FrozenHttpsRetrievalTransport",
    "endpoint_url": "https://api.castorini.uwaterloo.ca/v1/climbmix-400b/search",
    "one_shot_no_retry": True,
    "redirects_allowed": False,
    "injected_opener": False,
    "analyzer_client_class": "trec_rag.query_analyzer.RemoteLuceneQueryAnalyzer",
    "analyzer_url": "http://127.0.0.1:18081",
    "analyzer_conformance_probe_sha256": {"synthetic": "c" * 64},
}


class Analyzer:
    @property
    def fingerprint(self):
        return FINGERPRINT

    def analyze(self, text):
        tokens = tuple(re.findall(r"[^\W\d_]+", text.lower()))
        return AnalyzedQuery(tokens, stable_unique(tokens), FINGERPRINT)


def test_formal_external_gate_is_closed_while_index_revision_is_unknown():
    config = load_det_sparse_config(REPO_ROOT / "configs" / "det_sparse_v1.yaml")

    with pytest.raises(PermissionError, match="immutable revision"):
        _require_external_authorization(config)


def test_executor_hard_gate_precedes_hostile_analyzer_transport_and_ticket_state(
    tmp_path,
):
    config, _analyzer, preflight = _context(
        tmp_path,
        "shared systems context: what improves privacy controls?",
    )

    class HostileAnalyzer:
        def __init__(self):
            self.calls = []

        @property
        def fingerprint(self):
            self.calls.append("fingerprint")
            raise AssertionError("analyzer must remain untouched")

        def analyze(self, _text):
            self.calls.append("analyze")
            raise AssertionError("analyzer must remain untouched")

    analyzer = HostileAnalyzer()
    transport = Transport()

    with pytest.raises(PermissionError, match="immutable revision"):
        execute_det_sparse_run(
            config,
            preflight_dir=preflight,
            run_dir=config.output_dir / "execution",
            query_analyzer=analyzer,
            transport=transport,
        )

    assert analyzer.calls == []
    assert transport.calls == []
    assert not (config.output_dir / "execution").exists()
    assert not (config.root_dir / ".synthetic_global_budget").exists()


def _context(tmp_path, narrative):
    root = tmp_path / "synthetic_repo"
    root.mkdir()
    (root / "AGENTS.md").write_text("# Synthetic test root\n", encoding="utf-8")
    config_path = root / "configs" / "det_sparse_v1.yaml"
    config_path.parent.mkdir()
    shutil.copyfile(REPO_ROOT / "configs" / "det_sparse_v1.yaml", config_path)
    topics = root / (
        "trec-rag-data/trec-rag-2026/development-data/topics/"
        "rag25-topics-dev.tsv"
    )
    topics.parent.mkdir(parents=True)
    topics.write_text(
        "".join(
            f"{topic_id}\t{narrative}\n"
            for topic_id in ("200", "225", "707", "897")
        ),
        encoding="utf-8",
    )
    config = load_det_sparse_config(config_path)
    output = config.output_dir
    analyzer = Analyzer()
    expected_sha = hashlib.sha256(
        json.dumps(
            FINGERPRINT.to_dict(), separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    assert expected_sha == config.analyzer.expected_fingerprint_sha256
    preflight = output / "preflight"
    result = build_preflight(
        config,
        query_analyzer=analyzer,
        output_dir=preflight,
        source_provenance=SOURCE,
    )
    assert result.valid
    return config, analyzer, preflight


def test_executor_rejects_mutated_in_memory_config_before_transport(
    tmp_path,
    monkeypatch,
):
    config, analyzer, preflight = _context(
        tmp_path,
        "shared systems context: what improves privacy controls?",
    )
    drifted = replace(
        config,
        retrieval=replace(config.retrieval, hits=99),
    )
    transport = Transport()
    monkeypatch.setattr(
        "trec_rag.det_sparse_run._current_source_provenance",
        lambda _root: dict(SOURCE),
    )

    with pytest.raises(ValueError, match="fresh canonical config reload"):
        execute_det_sparse_run(
            drifted,
            preflight_dir=preflight,
            run_dir=config.output_dir / "execution",
            query_analyzer=analyzer,
            transport=transport,
        )

    assert transport.calls == []
    assert not (config.output_dir / "execution").exists()


def _body(count=100):
    return json.dumps(
        {
            "candidates": [
                {
                    "docid": f"doc-{rank:03d}",
                    "rank": rank,
                    "score": 101.0 - rank,
                    "doc": {
                        "contents": (
                            "evidence novelterm general documents"
                            if rank <= 2
                            else "general documents"
                        )
                    },
                }
                for rank in range(1, count + 1)
            ]
        },
        separators=(",", ":"),
    ).encode()


class Transport:
    transport_version = TRANSPORT_VERSION
    endpoint_url = "https://api.castorini.uwaterloo.ca/v1/climbmix-400b/search"
    one_shot_no_retry = True
    redirects_allowed = False

    def __init__(self, *, count=100, fail_at=None):
        self.count = count
        self.fail_at = fail_at
        self.calls = []

    def __call__(self, request):
        self.calls.append(request)
        if self.fail_at == len(self.calls):
            raise TimeoutError("synthetic timeout")
        return RawTransportResponse(200, {"X-Synthetic": "true"}, _body(self.count), 0.01)


def _execute(monkeypatch, config, analyzer, preflight, transport):
    monkeypatch.setattr(
        "trec_rag.det_sparse_run._require_external_authorization",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "trec_rag.det_sparse_run._current_source_provenance",
        lambda _root: dict(SOURCE),
    )
    monkeypatch.setattr(
        "trec_rag.det_sparse_run.global_budget_dir",
        lambda _root: config.root_dir / ".synthetic_global_budget",
    )
    monkeypatch.setattr(
        "trec_rag.det_sparse_budget.global_budget_dir",
        lambda _root: config.root_dir / ".synthetic_global_budget",
    )
    monkeypatch.setattr(
        "trec_rag.det_sparse_provenance.current_source_provenance",
        lambda _root: dict(SOURCE),
    )
    monkeypatch.setattr(
        "trec_rag.det_sparse_freeze._validate_evaluation_analyzer_boundary",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "trec_rag.det_sparse_run._validate_formal_runtime_boundary",
        lambda *_args: dict(RUNTIME_BOUNDARY),
    )
    return execute_det_sparse_run(
        config,
        preflight_dir=preflight,
        run_dir=config.output_dir / "execution",
        query_analyzer=analyzer,
        transport=transport,
    )


def test_executor_retrieves_exact_aliases_once_and_freezes_without_qrels(
    tmp_path,
    monkeypatch,
):
    config, analyzer, preflight = _context(
        tmp_path,
        "shared systems context: what improves privacy controls?",
    )
    transport = Transport()

    result = _execute(monkeypatch, config, analyzer, preflight, transport)

    assert result.mechanical_valid
    assert result.status == "success"
    assert result.external_calls == 12
    assert result.per_topic_external_calls == {
        "200": 3,
        "225": 3,
        "707": 3,
        "897": 3,
    }
    assert len(transport.calls) == 12
    registry = json.loads((result.output_dir / "query_registry.json").read_text())
    assert any(
        set(row["logical_variant_aliases"])
        == {"det_sparse_v1:original", "det_sparse_v1:facet:f02"}
        for row in registry
    )
    assert all(
        len([row for row in result.rankings[arm] if row.topic_id == topic_id]) == 100
        for arm in ("O", "F", "E", "FE")
        for topic_id in config.topic_ids
    )
    assert result.final_freeze_path is not None
    assert result.final_freeze_path.is_file()
    assert not config.evaluation.qrels.exists()
    validate_evaluation_freeze(
        result.final_freeze_path,
        artifact_root=config.output_dir,
        expected_qrels_path=config.evaluation.qrels,
        query_analyzer=analyzer,
    )
    ungraded = derive_frozen_ungraded_diagnostics(
        manifest_path=result.final_freeze_path,
        artifact_root=config.output_dir,
        expected_qrels_path=config.evaluation.qrels,
        query_analyzer=analyzer,
    )
    assert not config.evaluation.qrels.exists()
    assert ungraded["cost"]["external_calls"] == 12
    assert ungraded["cost"]["cache_hits"] == 0
    assert ungraded["arm_overlap"]["O"]["200"] == {
        "unique_vs_original_at_100": 0,
        "jaccard_with_original_at_100": 1.0,
    }
    assert any(
        row["logical_variant_name"] == "det_sparse_v1:original"
        for row in ungraded["logical_stream_overlap"]["200"]
    )
    config.evaluation.qrels.parent.mkdir(parents=True, exist_ok=True)
    config.evaluation.qrels.write_text(
        "".join(f"{topic_id} 0 doc-001 4\n" for topic_id in config.topic_ids),
        encoding="utf-8",
    )
    evaluation = evaluate_frozen_arms(
        manifest_path=result.final_freeze_path,
        artifact_root=config.output_dir,
        qrels_path=config.evaluation.qrels,
        rankings=result.rankings,
        query_analyzer=analyzer,
    )
    assert evaluation.arm_metrics["O"]["per_topic"]["200"]["recall@100"] == 1.0
    assert evaluation.evidence["qrels_sha256"] == hashlib.sha256(
        config.evaluation.qrels.read_bytes()
    ).hexdigest()
    assert evaluation.descriptive_diagnostics[
        "graded_new_relevant_vs_original"
    ]["FE"]["200"]["new_relevant_vs_original_at_100"] == 0
    decision = assess_pilot(
        manifest_path=result.final_freeze_path,
        artifact_root=config.output_dir,
        qrels_path=config.evaluation.qrels,
        rankings=result.rankings,
        query_analyzer=analyzer,
    )
    assert decision.preferred_arm == "O"
    assert decision.evaluation_evidence["qrels_sha256"] == evaluation.evidence[
        "qrels_sha256"
    ]


def test_executor_derives_and_enforces_nine_per_topic_thirty_six_total(
    tmp_path,
    monkeypatch,
):
    narrative = (
        "shared systems context: what improves privacy controls? "
        "how are costs measured? which failures recur? where are repairs documented?"
    )
    config, analyzer, preflight = _context(tmp_path, narrative)
    metadata = json.loads((preflight / "_preflight.json").read_text())
    assert [row["derived_max_unique_requests"] for row in metadata["request_projections"]] == [
        9,
        9,
        9,
        9,
    ]
    transport = Transport()

    result = _execute(monkeypatch, config, analyzer, preflight, transport)

    assert result.mechanical_valid
    assert result.external_calls == 36
    assert result.per_topic_external_calls == {topic_id: 9 for topic_id in config.topic_ids}
    assert len(transport.calls) == 36


def test_short_response_fails_after_one_raw_first_call_and_run_cannot_retry(
    tmp_path,
    monkeypatch,
):
    config, analyzer, preflight = _context(
        tmp_path,
        "shared systems context: what improves privacy controls?",
    )
    transport = Transport(count=99)

    result = _execute(monkeypatch, config, analyzer, preflight, transport)

    assert not result.mechanical_valid
    assert result.status == "failure"
    assert result.external_calls == 1
    assert len(transport.calls) == 1
    assert result.final_freeze_path is None
    assert "exactly 100" in result.failure["error"]
    assert not config.evaluation.qrels.exists()
    with pytest.raises(FileExistsError):
        _execute(monkeypatch, config, analyzer, preflight, Transport())


def test_transport_failure_stops_immediately_and_materializes_original_fallback(
    tmp_path,
    monkeypatch,
):
    config, analyzer, preflight = _context(
        tmp_path,
        "shared systems context: what improves privacy controls?",
    )
    transport = Transport(fail_at=2)

    result = _execute(monkeypatch, config, analyzer, preflight, transport)

    assert not result.mechanical_valid
    assert result.external_calls == 2
    assert len(transport.calls) == 2
    assert result.final_freeze_path is None
    arm_plans = json.loads((result.output_dir / "arm_plans.json").read_text())
    assert [row["status"] for row in arm_plans["200"]] == [
        "ok",
        "fallback",
        "fallback",
        "fallback",
    ]
    assert [row.docid for row in result.rankings["O"] if row.topic_id == "200"] == [
        row.docid for row in result.rankings["FE"] if row.topic_id == "200"
    ]


def test_pretransport_analyzer_identity_failure_makes_zero_calls(
    tmp_path,
    monkeypatch,
):
    config, analyzer, preflight = _context(
        tmp_path,
        "shared systems context: what improves privacy controls?",
    )
    changed = AnalyzerFingerprint(
        **{**FINGERPRINT.__dict__, "lucene_version": "changed"}
    )
    analyzer = type(
        "ChangedAnalyzer",
        (),
        {
            "fingerprint": property(lambda _self: changed),
            "analyze": lambda _self, text: AnalyzedQuery(
                tuple(re.findall(r"[^\W\d_]+", text.lower())),
                stable_unique(tuple(re.findall(r"[^\W\d_]+", text.lower()))),
                changed,
            ),
        },
    )()
    transport = Transport()
    monkeypatch.setattr(
        "trec_rag.det_sparse_run._current_source_provenance",
        lambda _root: dict(SOURCE),
    )
    monkeypatch.setattr(
        "trec_rag.det_sparse_run._require_external_authorization",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "trec_rag.det_sparse_run._validate_formal_runtime_boundary",
        lambda *_args: dict(RUNTIME_BOUNDARY),
    )

    with pytest.raises(ValueError, match="analyzer"):
        execute_det_sparse_run(
            config,
            preflight_dir=preflight,
            run_dir=config.output_dir / "execution",
            query_analyzer=analyzer,
            transport=transport,
        )

    assert transport.calls == []
    assert not (config.output_dir / "execution").exists()


@pytest.mark.parametrize("invalid_projection", [False, -1])
def test_pretransport_rejects_forged_nonpositive_or_boolean_projection(
    tmp_path,
    monkeypatch,
    invalid_projection,
):
    config, analyzer, preflight = _context(
        tmp_path,
        "shared systems context: what improves privacy controls?",
    )
    metadata_path = preflight / "_preflight.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["request_projections"][0][
        "derived_max_unique_requests"
    ] = invalid_projection
    metadata["derived_max_total_unique_requests"] = sum(
        row["derived_max_unique_requests"]
        for row in metadata["request_projections"]
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze_path = preflight / "pre_retrieval_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["metadata"]["request_projections"] = metadata["request_projections"]
    freeze["metadata"]["derived_max_total_unique_requests"] = metadata[
        "derived_max_total_unique_requests"
    ]
    row = next(
        item for item in freeze["artifacts"] if item["path"] == "_preflight.json"
    )
    row["size"] = metadata_path.stat().st_size
    row["sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    freeze_path.write_text(
        json.dumps(freeze, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    transport = Transport()
    monkeypatch.setattr(
        "trec_rag.det_sparse_run._require_external_authorization",
        lambda _config: None,
    )

    with pytest.raises(ValueError, match="integer in 1..9"):
        execute_det_sparse_run(
            config,
            preflight_dir=preflight,
            run_dir=config.output_dir / "execution",
            query_analyzer=analyzer,
            transport=transport,
        )

    assert transport.calls == []
    assert not (config.output_dir / "execution").exists()
