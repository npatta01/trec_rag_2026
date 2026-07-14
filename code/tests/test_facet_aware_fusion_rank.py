from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

import trec_rag.facet_aware_fusion_rank as module
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
        },
    )


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


def test_freeze_refuses_uncached_work_without_implicit_model_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeCache:
        scores: dict[str, float] = {}

        def __init__(self, _root: Path, _context: object) -> None:
            pass

    class FakePlan:
        windows = (object(),)
        summary = {"unique_uncached_pair_count": 1}

    monkeypatch.setattr(
        module,
        "_load_rank_inputs",
        lambda **_kwargs: (
            {"topic_ids": ["233"], "topics": [], "facets": []},
            tuple(),
            {"candidates_sha256": "1" * 64},
        ),
    )
    monkeypatch.setattr(module, "load_verified_tokenizer", lambda _path: object())
    monkeypatch.setattr(module, "GlobalScoreCache", FakeCache)
    monkeypatch.setattr(
        module,
        "build_scoring_preflight",
        lambda _candidates, _tokenizer, _cache: FakePlan(),
    )

    with pytest.raises(ValueError, match="cache misses.*separate approval-gated"):
        module.run_rank_freeze(
            manifest_path=tmp_path / "manifest.json",
            retrieval_dir=tmp_path / "retrieval",
            cache_root=tmp_path / "original-cache",
            output_dir=tmp_path / "freeze",
        )
