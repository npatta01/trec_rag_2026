from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

import trec_rag.facet_aware_fusion_rank as module
import trec_rag.facet_local_minilm_preflight as preflight_module
from trec_rag.facet_aware_fusion_run import _manifest_sha256 as task2_manifest_sha256
from trec_rag.facet_local_minilm_score import load_scoring_inputs
from trec_rag.facet_aware_fusion_rank import (
    ARM_NAMES,
    FacetStream,
    balanced_interleave,
    build_rankings,
    constrained_xquad,
    coverage_deadlines,
    create_ranking_freeze,
    quality_gate,
    rank_facet_documents,
    rank_normalized,
    prepare_scoring_candidates,
    verify_ranking_freeze,
    xquad,
)


def _doc(
    document_id: str,
    rank: int,
    *,
    text: str = "",
    score: float = 0.0,
) -> dict[str, object]:
    return {
        "document_id": document_id,
        "rank": rank,
        "score": score,
        "text": text,
    }


def _facet(
    facet_id: str,
    manifest_order: int,
    ranked_docs: list[dict[str, object]],
    *,
    accepted: bool,
) -> FacetStream:
    return FacetStream(
        facet_id=facet_id,
        manifest_order=manifest_order,
        ranked_docs=tuple(ranked_docs),
        accepted=accepted,
        topic_id="233",
    )


def test_rank_normalization_uses_only_rank_and_depth() -> None:
    assert rank_normalized(1, 50) == 1.0
    assert rank_normalized(2, 50) == pytest.approx(1.0 / math.log2(3))
    assert rank_normalized(50, 50) == pytest.approx(1.0 / math.log2(51))
    assert rank_normalized(None, 50) == 0.0
    assert rank_normalized(51, 50) == 0.0

    with pytest.raises(ValueError, match="depth"):
        rank_normalized(1, 0)
    with pytest.raises(ValueError, match="rank"):
        rank_normalized(0, 50)


def test_quality_gate_applies_anchor_relation_and_wrong_domain_thresholds() -> None:
    facet = {
        "facet_id": "233-depression",
        "query": "teenager social media depression contribution",
        "anchor_terms": ["social media", "social network"],
        "relation_terms": ["mental health", "depression"],
        "wrong_domain_patterns": [r"marketing agency"],
    }
    ranked_docs = [
        _doc("d1", 1, text="Social media can support mental health."),
        _doc("d2", 2, text="A social network may contribute to depression."),
        _doc("d3", 3, text="Social media use among teenagers."),
        _doc("d4", 4, text="Mental health information for parents."),
        _doc("d5", 5, text="Essay template about social media."),
    ]

    accepted = quality_gate(facet, ranked_docs)
    assert accepted.accepted is True
    assert accepted.anchor_top5_count == 4
    assert accepted.anchor_relation_top5_count == 2
    assert accepted.wrong_domain_top5_count == 0
    assert accepted.content_warning_top5_count == 1

    rejected_docs = copy.deepcopy(ranked_docs)
    rejected_docs[3]["text"] = "Social media marketing agency services."
    rejected_docs[4]["text"] = "Social network marketing agency essay template."
    rejected = quality_gate(facet, rejected_docs)
    assert rejected.accepted is False
    assert rejected.wrong_domain_top5_count == 2
    assert "wrong_domain" in rejected.failed_checks


def test_balanced_interleave_alternates_original_and_accepted_facets() -> None:
    original = [_doc(f"o{rank}", rank) for rank in range(1, 7)]
    accepted_a = _facet(
        "a",
        0,
        [_doc("a1", 1), _doc("o2", 2), _doc("a2", 3)],
        accepted=True,
    )
    accepted_b = _facet(
        "b",
        1,
        [_doc("b1", 1), _doc("b2", 2)],
        accepted=True,
    )
    rejected = _facet("rejected", 2, [_doc("r1", 1)], accepted=False)

    assert balanced_interleave(
        original, [accepted_b, rejected, accepted_a], limit=8
    ) == ["o1", "a1", "o2", "b1", "o3", "a2", "o4", "b2"]


def test_xquad_uses_best_original_or_facet_rank_not_raw_scores() -> None:
    original = [
        _doc("original-best", 1, score=1_000_000.0),
        _doc("original-second", 2, score=999_999.0),
    ]
    facet = _facet(
        "accepted",
        0,
        [_doc("facet-best", 1, score=-1_000_000.0)],
        accepted=True,
    )

    first = xquad(original, [facet], limit=3)
    changed_scores = copy.deepcopy(original)
    changed_scores[0]["score"] = -math.inf
    changed_facet = _facet(
        "accepted",
        0,
        [_doc("facet-best", 1, score=math.inf)],
        accepted=True,
    )

    assert first[0] == "facet-best"
    assert xquad(changed_scores, [changed_facet], limit=3) == first


def test_cxq_deadlines_are_one_based_and_rejected_facet_is_never_forced() -> None:
    assert coverage_deadlines(3) == (24, 37, 50)

    original = [_doc(f"o{rank:03d}", rank) for rank in range(1, 101)]
    accepted = _facet(
        "accepted",
        0,
        [_doc(f"a{rank:03d}", rank) for rank in range(1, 51)],
        accepted=True,
    )
    rejected = _facet(
        "rejected",
        1,
        [_doc(f"r{rank:03d}", rank) for rank in range(1, 51)],
        accepted=False,
    )

    ranking, provenance = constrained_xquad(
        original, [rejected, accepted], limit=100
    )
    assert len(ranking) == len(set(ranking)) == 100
    assert rejected.facet_id not in {
        row.get("forced_facet") for row in provenance
    }
    assert not any(document_id.startswith("r") for document_id in ranking)


def test_cxq_records_forced_facet_deadline_source_rank_and_prior_state() -> None:
    # With more than 40 streams, the first deadline arrives before xQuAD can
    # represent every stream.  This compact synthetic case exercises the force
    # path even though the held-out manifest itself has at most seven facets.
    facets = [
        _facet(
            f"f{index:02d}",
            index,
            [_doc(f"z{40 - index:03d}", 1)],
            accepted=True,
        )
        for index in range(41)
    ]

    ranking, provenance = constrained_xquad([], facets, limit=20)
    forced = [row for row in provenance if row["forced_facet"] is not None]

    assert ranking[10] == "z040"
    assert forced[0]["position"] == 11
    assert forced[0]["forced_facet"] == "f00"
    assert forced[0]["deadline"] == 11
    assert forced[0]["source_rank"] == 1
    assert forced[0]["prior_coverage_state"]["f00"] is False


@pytest.fixture
def frozen_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(
        {
            "topic_id": "233",
            "family": "original",
            "document_id": f"o{rank:03d}",
            "rank": rank,
            "score": float(10_000 - rank),
        }
        for rank in range(1, 101)
    )
    rows.extend(
        {
            "topic_id": "233",
            "family": "facet",
            "facet_id": "accepted",
            "manifest_order": 0,
            "accepted": True,
            "document_id": f"a{rank:03d}",
            "rank": rank,
            "score": float(-rank),
        }
        for rank in range(1, 51)
    )
    rows.extend(
        {
            "topic_id": "233",
            "family": "facet",
            "facet_id": "rejected",
            "manifest_order": 1,
            "accepted": False,
            "document_id": f"r{rank:03d}",
            "rank": rank,
            "score": float(1_000_000 - rank),
        }
        for rank in range(1, 51)
    )
    return rows


def test_build_rankings_is_deterministic_complete_and_score_blind(
    frozen_rows: list[dict[str, object]],
) -> None:
    expected = build_rankings(frozen_rows)
    changed = copy.deepcopy(frozen_rows)
    for index, row in enumerate(changed):
        row["score"] = math.inf if index % 2 else -math.inf

    assert build_rankings(list(reversed(frozen_rows))) == expected
    assert build_rankings(changed) == expected
    assert set(expected["233"]) == set(ARM_NAMES)
    for ranking in expected["233"].values():
        assert len(ranking) == len(set(ranking)) == 100
        assert not any(document_id.startswith("r") for document_id in ranking)
    assert expected["233"]["O"] == [f"o{rank:03d}" for rank in range(1, 101)]


def test_rank_facet_documents_reuses_span_distinct_top4_aggregation() -> None:
    candidates = [_doc("d1", 1), _doc("d2", 2)]
    windows = [
        {
            "document_id": "d1",
            "document_start_token": 0,
            "document_end_token": 256,
            "window_id": "d1-w1",
            "score": 0.1,
        },
        {
            "document_id": "d2",
            "document_start_token": 0,
            "document_end_token": 256,
            "window_id": "d2-w1",
            "score": 0.9,
        },
    ]

    ranked = rank_facet_documents(candidates, windows)
    assert [row["document_id"] for row in ranked] == ["d2", "d1"]
    assert [row["rank"] for row in ranked] == [1, 2]
    assert [row["retrieval_rank"] for row in ranked] == [2, 1]


def test_ranking_freeze_is_create_only_complete_and_hash_authenticated(
    tmp_path: Path,
    frozen_rows: list[dict[str, object]],
) -> None:
    output = tmp_path / "freeze"
    created = create_ranking_freeze(
        output,
        frozen_rows=frozen_rows,
        gates=[
            {"topic_id": "233", "facet_id": "accepted", "accepted": True},
            {"topic_id": "233", "facet_id": "rejected", "accepted": False},
        ],
        score_rows=[{"cache_key": "abc", "score": 0.25}],
        candidate_provenance=frozen_rows,
        input_hashes={
            "manifest_sha256": "0" * 64,
            "retrieval_sha256": "1" * 64,
            "scoring_sha256": "2" * 64,
        },
    )

    verified = verify_ranking_freeze(output)
    assert verified == created
    assert verified["complete"] is True
    assert verified["qrels_opened"] is False
    assert verified["topic_ids"] == ["233"]
    assert verified["model"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert verified["model_revision"] == "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    assert set(verified["candidate_pool_sha256"]) == {"233"}
    assert set(verified["artifacts"]) >= {
        "scores.jsonl",
        "gates.json",
        "candidate_provenance.jsonl",
        "cxq_provenance.jsonl",
        "parameters.json",
    }
    with pytest.raises(FileExistsError, match="create-only"):
        create_ranking_freeze(
            output,
            frozen_rows=frozen_rows,
            gates=[],
            score_rows=[],
            candidate_provenance=[],
            input_hashes={"manifest_sha256": "0" * 64},
        )

    ranking_path = output / "rankings" / "O.jsonl"
    ranking_path.write_text(
        ranking_path.read_text(encoding="utf-8")
        + json.dumps({"topic_id": "233", "rank": 101, "docid": "tampered"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_ranking_freeze(output)


def test_task2_candidates_are_adapted_to_existing_window_preflight_schema() -> None:
    manifest = {
        "qrels_opened": False,
        "topic_ids": ["233"],
        "facets": [
            {
                "topic_id": "233",
                "facet_id": "233-positive",
                "query": "teenager social media positive mental health impacts",
                "manifest_order": 0,
            }
        ],
    }
    retrieval_rows = [
        {
            "schema_version": "facet-aware-fusion-candidate-v1",
            "topic_id": "233",
            "facet_id": "233-positive",
            "query_sha256": hashlib.sha256(
                b"teenager social media positive mental health impacts"
            ).hexdigest(),
            "rank": 1,
            "docid": "doc-1",
            "score": 123.0,
            "text": "candidate text",
            "request_key": "request-1",
        }
    ]

    adapted = prepare_scoring_candidates(manifest, retrieval_rows)
    assert adapted == (
        {
            **retrieval_rows[0],
            "family": "facet",
            "variant": "233-positive",
            "document_id": "doc-1",
            "query": "teenager social media positive mental health impacts",
            "text_sha256": hashlib.sha256(b"candidate text").hexdigest(),
        },
    )


def test_task2_adapter_authenticates_text_in_real_window_preflight(
    tmp_path: Path,
) -> None:
    query = "teenager social media positive mental health impacts"
    text = "candidate café 🚀 text for authenticated window planning"
    manifest = {
        "qrels_opened": False,
        "topic_ids": ["233"],
        "facets": [
            {
                "topic_id": "233",
                "facet_id": "233-positive",
                "query": query,
                "manifest_order": 0,
            }
        ],
    }
    retrieval_row = {
        "schema_version": "facet-aware-fusion-candidate-v1",
        "topic_id": "233",
        "facet_id": "233-positive",
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "rank": 1,
        "docid": "doc-1",
        "score": 123.0,
        "text": text,
        "request_key": "request-1",
    }
    adapted = prepare_scoring_candidates(manifest, [retrieval_row])
    cache = module.GlobalScoreCache(
        tmp_path / "score-cache", module.score_cache_context()
    )

    plan = module.build_scoring_preflight(adapted, _WordTokenizer(), cache)
    assert len(plan.windows) == 1
    assert adapted[0]["text_sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    changed_text = {**adapted[0], "text": text + " changed"}
    with pytest.raises(ValueError, match="candidate document hash mismatch"):
        module.build_scoring_preflight([changed_text], _WordTokenizer(), cache)

    changed_hash = {**adapted[0], "text_sha256": "0" * 64}
    with pytest.raises(ValueError, match="candidate document hash mismatch"):
        module.build_scoring_preflight([changed_hash], _WordTokenizer(), cache)


def test_task2_summary_binds_validated_manifest_by_canonical_content(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema_version": "synthetic-pretty-manifest-v1",
        "topic_ids": ["233"],
        "qrels_opened": False,
    }
    manifest_path = tmp_path / "manifest.json"
    _write_canonical_json(manifest_path, manifest)
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() != (
        task2_manifest_sha256(manifest)
    )

    retrieval = tmp_path / "retrieval"
    retrieval.mkdir()
    candidates_source = b""
    (retrieval / "candidates.jsonl").write_bytes(candidates_source)
    _write_canonical_json(
        retrieval / "retrieval_summary.json",
        {
            "complete": True,
            "qrels_opened": False,
            "candidate_rows": 0,
            "candidates_sha256": hashlib.sha256(candidates_source).hexdigest(),
            "manifest_sha256": task2_manifest_sha256(manifest),
        },
    )

    rows, summary = module._load_task2_candidates(manifest, retrieval)
    assert rows == ()
    assert summary["manifest_sha256"] == task2_manifest_sha256(manifest)

    with pytest.raises(ValueError, match="summary bindings"):
        module._load_task2_candidates({**manifest, "topic_ids": ["273"]}, retrieval)


def test_cli_requires_and_forwards_explicit_original_cache_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    retrieval = tmp_path / "retrieval"
    cache_root = tmp_path / "original-cache"
    output = tmp_path / "freeze"
    observed: dict[str, object] = {}

    def fake_preflight(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"document_count": 1, "qrels_opened": False}

    monkeypatch.setattr(module, "run_rank_preflight", fake_preflight)
    assert module.main(
        [
            "preflight",
            "--manifest",
            str(manifest),
            "--retrieval",
            str(retrieval),
            "--cache-root",
            str(cache_root),
            "--output",
            str(output),
        ]
    ) == 0

    assert observed["manifest_path"] == manifest
    assert observed["retrieval_dir"] == retrieval
    assert observed["cache_root"] == cache_root
    assert observed["output_dir"] == output
    assert json.loads(capsys.readouterr().out)["qrels_opened"] is False

    with pytest.raises(SystemExit):
        module.main(
            [
                "preflight",
                "--manifest",
                str(manifest),
                "--retrieval",
                str(retrieval),
                "--output",
                str(output),
            ]
        )


def test_cli_exposes_approval_gated_scoring_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_scoring(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "complete", "qrels_opened": False}

    monkeypatch.setattr(module, "run_rank_scoring", fake_scoring)
    assert module.main(
        [
            "score",
            "--preflight",
            str(tmp_path / "preflight.json"),
            "--approval",
            str(tmp_path / "approval.json"),
            "--approval-sha256",
            "2" * 64,
            "--score-cache",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "scoring"),
        ]
    ) == 0

    assert observed["preflight_path"] == tmp_path / "preflight.json"
    assert observed["approval_path"] == tmp_path / "approval.json"
    assert observed["approval_sha256"] == "2" * 64
    assert observed["output_dir"] == tmp_path / "scoring"
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_freeze_consumes_authenticated_scoring_handoff_without_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = {"topic_ids": ["233"], "topics": [], "facets": []}
    candidate = {
        "topic_id": "233",
        "family": "facet",
        "variant": "233-positive",
        "document_id": "raw",
        "rank": 1,
    }
    monkeypatch.setattr(
        module,
        "_load_rank_inputs",
        lambda **_kwargs: (
            manifest,
            (candidate,),
            {"candidates_sha256": "1" * 64},
        ),
    )
    handoff = module.RankScoringHandoff(
        scored_windows=({"cache_key": "one", "score": 0.5},),
        score_rows=({"cache_key": "one", "score": 0.5},),
        preflight={
            "manifest_sha256": manifest_sha256,
            "task3_retrieval_candidates_sha256": "1" * 64,
            "task3_topic_ids": ["233"],
            "qrels_opened": False,
            "windows_sha256": "3" * 64,
            "model_materialization_receipt_sha256": "4" * 64,
        },
        preflight_sha256="2" * 64,
        scoring_receipt={},
        scoring_receipt_sha256="5" * 64,
        ledger_sha256="6" * 64,
    )
    monkeypatch.setattr(module, "load_rank_scoring_handoff", lambda *_args: handoff)
    original = {"topic_id": "233", "family": "original", "document_id": "o", "rank": 1}
    facet_rank = {
        "topic_id": "233",
        "family": "facet",
        "facet_id": "233-positive",
        "variant": "233-positive",
        "document_id": "f",
        "rank": 1,
        "accepted": True,
        "manifest_order": 0,
    }
    monkeypatch.setattr(module, "_original_rank_rows", lambda *_args: [original])
    monkeypatch.setattr(
        module,
        "_scored_facet_rank_rows",
        lambda *_args: ([facet_rank], []),
    )
    observed: dict[str, object] = {}

    def fake_freeze(output: Path, **kwargs: object) -> dict[str, object]:
        observed.update({"output": output, **kwargs})
        return {"complete": True}

    monkeypatch.setattr(module, "create_ranking_freeze", fake_freeze)

    result = module.run_rank_freeze(
        manifest_path=manifest_path,
        retrieval_dir=tmp_path / "retrieval",
        cache_root=tmp_path / "original-cache",
        preflight_path=tmp_path / "preflight.json",
        scoring_dir=tmp_path / "scoring",
        output_dir=tmp_path / "freeze",
    )

    assert result == {"complete": True}
    assert observed["score_rows"] == handoff.score_rows
    provenance = observed["candidate_provenance"]
    assert isinstance(provenance, list)
    assert {row["provenance_stage"] for row in provenance} == {
        "original_rank",
        "task2_retrieval",
        "facet_minilm_rank",
    }


class _WordTokenizer:
    def encode(self, text, *, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False
        assert truncation is False
        return text.split()

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return " ".join(token_ids)

    def num_special_tokens_to_add(self, *, pair):
        assert pair is True
        return 3


def _synthetic_materialization_receipt(tmp_path: Path) -> Path:
    approval = {
        "schema_version": "facet-local-minilm-model-download-approval-v1",
        "approval_scope": "facet_local_minilm_model_materialization_v1",
        "approved_by": "test",
        "model_id": preflight_module.MODEL_ID,
        "revision": preflight_module.MODEL_REVISION,
        "allow_patterns": list(preflight_module.ALLOW_PATTERNS),
        "allow_patterns_sha256": preflight_module.approval_allow_patterns_sha256(),
        "acknowledged_network_download": True,
        "acknowledged_safe_files_only": True,
        "acknowledged_no_model_or_tokenizer_construction": True,
        "acknowledged_no_inference_qrels_retrieval_or_paid_calls": True,
    }
    approval_path = tmp_path / "model-approval.json"
    _write_canonical_json(approval_path, approval)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for index, relative in enumerate(preflight_module.ALLOW_PATTERNS, start=1):
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"safe-file-{index}\n".encode())
    output = tmp_path / "model-v1"
    preflight_module.materialize_model(
        model_id=preflight_module.MODEL_ID,
        revision=preflight_module.MODEL_REVISION,
        approval_path=approval_path,
        output_dir=output,
        snapshot_download_fn=lambda **_kwargs: str(snapshot),
    )
    return output / "materialization.json"


def test_cache_miss_preflight_persists_existing_scorer_compatible_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "teenager social media mental health"
    text = " ".join(f"token-{index}" for index in range(600))
    candidate = {
        "topic_id": "233",
        "family": "facet",
        "variant": "233-positive",
        "query": query,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "rank": 1,
        "document_id": "doc-1",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    score_cache_root = tmp_path / "score-cache"
    context = module.score_cache_context()
    cache = module.GlobalScoreCache(score_cache_root, context)
    plan = module.build_scoring_preflight([candidate], _WordTokenizer(), cache)
    assert plan.summary["unique_uncached_pair_count"] > 0
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "preflight"
    materialization_receipt = _synthetic_materialization_receipt(tmp_path)

    payload = module.persist_rank_preflight(
        output,
        plan=plan,
        manifest_path=manifest_path,
        retrieval_candidates_sha256="1" * 64,
        materialization_receipt_path=materialization_receipt,
        score_cache_root=score_cache_root,
        tokenizer_class="_WordTokenizer",
        topic_ids=["233"],
    )

    inputs = load_scoring_inputs(output / "preflight.json")
    assert payload["windows_sha256"] == inputs.windows_sha256
    assert inputs.benchmark["uncached_pair_count"] > 0
    assert len(inputs.windows) == payload["summary"]["window_count"]

    observed: dict[str, object] = {}

    def fake_scorer(preflight, approval, **kwargs):
        observed.update(
            {"preflight": preflight, "approval": approval, **kwargs}
        )
        return tuple(object() for _ in inputs.windows)

    monkeypatch.setattr(module, "run_rocm_scoring", fake_scorer)
    scoring_output = tmp_path / "scoring"
    result = module.run_rank_scoring(
        preflight_path=output / "preflight.json",
        approval_path=tmp_path / "approval.json",
        approval_sha256="2" * 64,
        output_dir=scoring_output,
        score_cache_root=score_cache_root,
    )
    assert result["score_row_count"] == len(inputs.windows)
    assert observed["preflight"] == output / "preflight.json"
    assert observed["score_cache_root"] == score_cache_root

    scoring_output.mkdir()
    planned_rows = []
    reservation_hashes = []
    ledger_rows = []
    output_hashes = []
    unique_outputs = {}
    for index, window in enumerate(inputs.windows):
        reservation_payload = {
            "sequence_index": index,
            "topic_id": window.topic_id,
            "family": window.family,
            "variant": window.variant,
            "rank": window.rank,
            "document_id": window.document_id,
            "window_id": window.window_id,
            "cache_key": window.cache_key,
        }
        reservation_sha256 = hashlib.sha256(
            module._compact_json_bytes(reservation_payload)
        ).hexdigest()
        planned_rows.append(
            {**reservation_payload, "reservation_sha256": reservation_sha256}
        )
        reservation_hashes.append(reservation_sha256)
        raw_output_sha256 = hashlib.sha256(
            module._compact_json_bytes(
                {
                    "cache_key": window.cache_key,
                    "inference_dtype": "float32",
                    "score_representation": "raw_logits",
                    "raw_float32_be_sha256": hashlib.sha256(
                        module.struct.pack(">f", 0.5)
                    ).hexdigest(),
                }
            )
        ).hexdigest()
        output_sha256 = hashlib.sha256(
            module._compact_json_bytes(
                {
                    "reservation_sha256": reservation_sha256,
                    "disposition": "forward_pass",
                    "raw_output_sha256": raw_output_sha256,
                }
            )
        ).hexdigest()
        ledger_rows.append(
            {
            "schema_version": "facet-local-minilm-score-row-v2",
            "topic_id": window.topic_id,
            "family": window.family,
            "variant": window.variant,
            "rank": window.rank,
            "document_id": window.document_id,
            "window_id": window.window_id,
            "query_sha256": window.query_sha256,
            "window_sha256": window.window_sha256,
            "cache_key": window.cache_key,
            "reservation_sha256": reservation_sha256,
            "disposition": "forward_pass",
            "score": 0.5,
            "elapsed_seconds": 0.01,
            "peak_device_memory_bytes": 0,
            "peak_host_memory_bytes": 0,
            "model": module.MODEL_ID,
            "model_revision": module.MODEL_REVISION,
            "score_representation": "raw_logits",
            "inference_dtype": "float32",
            "raw_output_sha256": raw_output_sha256,
            "output_sha256": output_sha256,
            }
        )
        output_hashes.append(output_sha256)
        unique_outputs[window.cache_key] = raw_output_sha256
    planned_bytes = b"".join(
        module._canonical_json_bytes(row, pretty=False) for row in planned_rows
    )
    (scoring_output / "planned_reservations.jsonl").write_bytes(planned_bytes)
    reservation_root = hashlib.sha256(
        b"".join((value + "\n").encode() for value in reservation_hashes)
    ).hexdigest()
    run_id = "synthetic-task3"
    approval_sha256 = "2" * 64
    _write_canonical_json(
        scoring_output / "run_reservation.json",
        {
            "schema_version": "facet-local-minilm-run-reservation-v1",
            "status": "reserved",
            "action": "full_scoring",
            "run_id": run_id,
            "output_path": str(scoring_output.resolve()),
            "approval_sha256": approval_sha256,
            "preflight_sha256": inputs.preflight_sha256,
            "windows_sha256": inputs.windows_sha256,
            "model_materialization_receipt_sha256": inputs.materialization_receipt_sha256,
            "benchmark_sample_sha256": inputs.benchmark["sample_sha256"],
            "planned_row_count": len(planned_rows),
            "planned_reservations_bytes": len(planned_bytes),
            "planned_reservations_sha256": hashlib.sha256(planned_bytes).hexdigest(),
            "reservation_sequence_root_sha256": reservation_root,
        }
    )
    ledger_bytes = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for row in ledger_rows
    )
    (scoring_output / "scoring_ledger.jsonl").write_bytes(ledger_bytes)
    receipt = {
        "schema_version": "facet-local-minilm-scoring-receipt-v2",
        "status": "complete",
        "preflight_sha256": inputs.preflight_sha256,
        "windows_sha256": inputs.windows_sha256,
        "model_materialization_receipt_sha256": inputs.materialization_receipt_sha256,
        "full_inference_approval_sha256": approval_sha256,
        "full_inference_request_sha256": "3" * 64,
        "score_cache_path": str(inputs.score_cache_path),
        "cache_before_sha256": None,
        "cache_after_sha256": "4" * 64,
        "cache_before_bytes": 0,
        "cache_after_bytes": 1,
        "cache_before_count": 0,
        "cache_after_count": len(unique_outputs),
        "planned_window_count": len(inputs.windows),
        "completed_window_count": len(inputs.windows),
        "cache_hit_count": 0,
        "forward_pass_count": len(inputs.windows),
        "same_run_reuse_count": 0,
        "unique_score_count": len(unique_outputs),
        "failed_window_count": 0,
        "pending_window_count": 0,
        "reservation_sequence_root_sha256": reservation_root,
        "ledger_bytes": len(ledger_bytes),
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "ledger_sequence_root_sha256": hashlib.sha256(
            b"".join((value + "\n").encode() for value in output_hashes)
        ).hexdigest(),
        "unique_score_root_sha256": hashlib.sha256(
            b"".join(
                f"{key}:{unique_outputs[key]}\n".encode()
                for key in sorted(unique_outputs)
            )
        ).hexdigest(),
        "score_representation": "raw_logits",
        "inference_dtype": "float32",
        "model": module.MODEL_ID,
        "model_revision": module.MODEL_REVISION,
    }
    receipt_bytes = (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    (scoring_output / "scoring_receipt.json").write_bytes(receipt_bytes)
    _write_canonical_json(
        scoring_output / "run_terminal.json",
        {
            "schema_version": "facet-local-minilm-run-terminal-v1",
            "status": "complete",
            "action": "full_scoring",
            "run_id": run_id,
            "approval_sha256": approval_sha256,
            "output_path": str(scoring_output.resolve()),
            "scoring_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "cache_after_sha256": "4" * 64,
        },
    )

    handoff = module.load_rank_scoring_handoff(
        output / "preflight.json", scoring_output
    )
    assert len(handoff.scored_windows) == len(inputs.windows)
    assert handoff.scored_windows[0]["score"] == 0.5
    assert handoff.scoring_receipt_sha256 == hashlib.sha256(receipt_bytes).hexdigest()

    terminal_path = scoring_output / "run_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["scoring_receipt_sha256"] = "0" * 64
    _write_canonical_json(terminal_path, terminal)
    with pytest.raises(ValueError, match="terminal receipt binding"):
        module.load_rank_scoring_handoff(output / "preflight.json", scoring_output)


def test_ranking_freeze_atomic_publish_preserves_raced_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_rows: list[dict[str, object]],
) -> None:
    output = tmp_path / "freeze"
    real_publish = module._publish_directory_noreplace
    raced: dict[str, int] = {}

    def race(stage: Path, destination: Path) -> None:
        destination.mkdir(mode=0o711)
        raced["inode"] = destination.stat().st_ino
        real_publish(stage, destination)

    monkeypatch.setattr(module, "_publish_directory_noreplace", race)
    with pytest.raises(FileExistsError):
        create_ranking_freeze(
            output,
            frozen_rows=frozen_rows,
            gates=[],
            score_rows=[],
            candidate_provenance=frozen_rows,
            input_hashes={"manifest_sha256": "0" * 64},
        )

    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert output.stat().st_ino == raced["inode"]
    assert list(tmp_path.glob(".freeze.*")) == []


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_freeze_requires_exact_artifact_set_including_hash_manifest(
    tmp_path: Path,
    frozen_rows: list[dict[str, object]],
) -> None:
    output = tmp_path / "freeze"
    create_ranking_freeze(
        output,
        frozen_rows=frozen_rows,
        gates=[],
        score_rows=[],
        candidate_provenance=frozen_rows,
        input_hashes={"manifest_sha256": "0" * 64},
    )
    freeze_path = output / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert "hashes.json" in freeze["artifacts"]

    freeze["artifacts"].pop("scores.jsonl")
    _write_canonical_json(freeze_path, freeze)
    with pytest.raises(ValueError, match="exact artifact set"):
        verify_ranking_freeze(output)


def test_freeze_verifier_recomputes_declared_artifact_row_counts(
    tmp_path: Path,
    frozen_rows: list[dict[str, object]],
) -> None:
    output = tmp_path / "freeze"
    create_ranking_freeze(
        output,
        frozen_rows=frozen_rows,
        gates=[],
        score_rows=[{"cache_key": "one", "score": 0.5}],
        candidate_provenance=frozen_rows,
        input_hashes={"manifest_sha256": "0" * 64},
    )
    freeze_path = output / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["artifacts"]["scores.jsonl"]["rows"] = 2
    _write_canonical_json(freeze_path, freeze)

    with pytest.raises(ValueError, match="row count"):
        verify_ranking_freeze(output)


def test_freeze_verifier_authenticates_hash_manifest_contents(
    tmp_path: Path,
    frozen_rows: list[dict[str, object]],
) -> None:
    output = tmp_path / "freeze"
    create_ranking_freeze(
        output,
        frozen_rows=frozen_rows,
        gates=[],
        score_rows=[],
        candidate_provenance=frozen_rows,
        input_hashes={"manifest_sha256": "0" * 64},
    )
    hashes_path = output / "hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["artifacts"].pop("scores.jsonl")
    _write_canonical_json(hashes_path, hashes)
    hashes_source = hashes_path.read_bytes()

    freeze_path = output / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["artifacts"]["hashes.json"] = {
        "bytes": len(hashes_source),
        "sha256": hashlib.sha256(hashes_source).hexdigest(),
    }
    _write_canonical_json(freeze_path, freeze)

    with pytest.raises(ValueError, match="hash manifest authentication"):
        verify_ranking_freeze(output)
