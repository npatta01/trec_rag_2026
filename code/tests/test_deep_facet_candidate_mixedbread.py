from __future__ import annotations

from collections.abc import Callable

import pytest

from trec_rag.deep_facet_candidate_mixedbread import (
    MIXEDBREAD_END_DEPTH,
    PROTECTED_TOPIC_IDS,
    RRF_HEAD_DEPTH,
    aggregate_top4,
    assemble_complete_ranking,
    build_residual_pool,
    build_structured_query,
    _load_queries,
    _authenticate_evaluation_inputs,
    _validate_score_row,
    reject_protected_topics_before_access,
    freeze_rankings,
    verify_ranking_freeze,
    verify_ranking_semantics,
)


def test_query_loader_uses_only_gate_accepted_facets(tmp_path, monkeypatch) -> None:
    import json

    import trec_rag.deep_facet_candidate_mixedbread as module

    monkeypatch.setattr(module, "TOPIC_IDS", ("84",))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "topics": [{"topic_id": "84", "query": "Full narrative"}],
                "facets": [
                    {
                        "topic_id": "84",
                        "facet_id": "84-health",
                        "manifest_order": 0,
                        "obligation": "Human health effects.",
                    },
                    {
                        "topic_id": "84",
                        "facet_id": "84-animals",
                        "manifest_order": 1,
                        "obligation": "Animal vaccination effects.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    queries, obligations, facet_ids = _load_queries(manifest, {"84-health"})

    assert obligations == {"84": ["Human health effects."]}
    assert facet_ids == {"84": ["84-health"]}
    assert "Human health effects." in queries["84"]
    assert "Animal vaccination effects." not in queries["84"]


def test_query_loader_rejects_unknown_or_cross_topic_gate_ids(tmp_path, monkeypatch) -> None:
    import json

    import trec_rag.deep_facet_candidate_mixedbread as module

    monkeypatch.setattr(module, "TOPIC_IDS", ("84",))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "topics": [{"topic_id": "84", "query": "Full narrative"}],
                "facets": [
                    {
                        "topic_id": "84",
                        "facet_id": "84-health",
                        "manifest_order": 0,
                        "obligation": "Human health effects.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="accepted facet IDs differ"):
        _load_queries(manifest, {"84-health", "84-animals"})


def test_score_identity_rejects_same_window_from_another_policy() -> None:
    import trec_rag.deep_facet_candidate_mixedbread as module

    window = {
        "topic_id": "84",
        "document_id": "d1",
        "chunk_index": 0,
        "query_sha256": "q" * 64,
        "window_text_sha256": "w" * 64,
    }
    score = {
        **window,
        **module._score_policy_identity(),
        "score": 1.25,
        "score_source": "inference",
    }
    _validate_score_row(score, window)
    score["model_revision"] = "wrong"
    with pytest.raises(ValueError, match="model/policy identity"):
        _validate_score_row(score, window)


def test_existing_evaluation_inputs_require_the_authenticated_hash_chain(
    tmp_path, monkeypatch
) -> None:
    import json
    import shutil

    import trec_rag.deep_facet_candidate_mixedbread as module

    source = module.Path(
        "outputs/rag25_deep_facet_candidates_v1/evaluation_v1"
    ).resolve()
    for name in (
        "qrels_projection.jsonl",
        "qrels_access_receipt.json",
        "metrics.json",
        "decision.json",
        "summary.json",
    ):
        shutil.copy2(source / name, tmp_path / name)
    _authenticate_evaluation_inputs(
        tmp_path / "qrels_projection.jsonl", tmp_path / "metrics.json"
    )

    rows = (tmp_path / "qrels_projection.jsonl").read_text(encoding="utf-8").splitlines()
    duplicate = json.loads(rows[0])
    rows.append(json.dumps(duplicate, sort_keys=True))
    (tmp_path / "qrels_projection.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt does not authenticate"):
        _authenticate_evaluation_inputs(
            tmp_path / "qrels_projection.jsonl", tmp_path / "metrics.json"
        )


def test_structured_query_contains_unchanged_narrative_and_every_obligation_once() -> None:
    narrative = "Explain causes, effects, and prevention of deforestation."
    obligations = (
        "Main causes of deforestation.",
        "Effects on climate.",
        "Effects on animals and humans.",
        "Actions that prevent deforestation.",
    )

    query = build_structured_query(narrative, obligations)

    assert query.startswith(f"Narrative:\n{narrative}\n\nInformation needs:\n")
    for index, obligation in enumerate(obligations, start=1):
        assert query.count(f"{index}. {obligation}") == 1
    assert query.count("Information needs:") == 1


def test_structured_query_rejects_empty_or_duplicate_obligations() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_structured_query("narrative", ())
    with pytest.raises(ValueError, match="duplicate"):
        build_structured_query("narrative", ("same", "same"))


def test_protected_topics_fail_before_any_source_access() -> None:
    accessed = False

    def source_loader() -> None:
        nonlocal accessed
        accessed = True

    with pytest.raises(ValueError, match="protected topic 144"):
        reject_protected_topics_before_access(("219", PROTECTED_TOPIC_IDS[0]), source_loader)
    assert accessed is False


def test_allowed_topics_can_access_source_once() -> None:
    calls = 0

    def source_loader() -> str:
        nonlocal calls
        calls += 1
        return "loaded"

    assert reject_protected_topics_before_access(("219", "72"), source_loader) == "loaded"
    assert calls == 1


def test_residual_pool_is_bounded_union_minus_exact_rrf_head() -> None:
    population = [f"d{index:04d}" for index in range(1300)]
    rrf = population
    global_order = population[300:] + population[:300]
    dual = list(reversed(population))

    pool = build_residual_pool(rrf, global_order, dual)

    expected = (
        set(rrf[:500]) | set(global_order[:500]) | set(dual[:1000])
    ) - set(rrf[:RRF_HEAD_DEPTH])
    assert set(pool) == expected
    assert pool == tuple(document_id for document_id in dual if document_id in expected)
    assert not set(rrf[:RRF_HEAD_DEPTH]).intersection(pool)


def test_complete_ranking_protects_head_reranks_middle_and_keeps_every_document() -> None:
    population = [f"d{index:04d}" for index in range(700)]
    rrf = population
    dual = list(reversed(population))
    residual = tuple(dual[:600])
    scores = {document_id: float(index) for index, document_id in enumerate(residual)}

    ranking = assemble_complete_ranking(rrf, dual, scores)

    assert ranking[:RRF_HEAD_DEPTH] == tuple(rrf[:RRF_HEAD_DEPTH])
    expected_middle = tuple(
        sorted(
            (document_id for document_id in residual if document_id not in set(rrf[:10])),
            key=lambda document_id: (-scores[document_id], dual.index(document_id)),
        )[: MIXEDBREAD_END_DEPTH - RRF_HEAD_DEPTH]
    )
    assert ranking[RRF_HEAD_DEPTH:MIXEDBREAD_END_DEPTH] == expected_middle
    assert len(ranking) == len(population)
    assert len(set(ranking)) == len(population)
    assert set(ranking) == set(population)
    selected = set(ranking[:MIXEDBREAD_END_DEPTH])
    assert ranking[MIXEDBREAD_END_DEPTH:] == tuple(
        document_id for document_id in dual if document_id not in selected
    )


def test_complete_ranking_uses_dual_rank_as_deterministic_score_tie_break() -> None:
    population = [f"d{index:04d}" for index in range(520)]
    rrf = population
    dual = list(reversed(population))
    protected = set(rrf[:RRF_HEAD_DEPTH])
    scores = {document_id: 1.0 for document_id in dual if document_id not in protected}

    first = assemble_complete_ranking(rrf, dual, scores)
    second = assemble_complete_ranking(tuple(rrf), tuple(dual), dict(reversed(tuple(scores.items()))))

    assert first == second
    assert first[10:500] == tuple(document_id for document_id in dual if document_id in scores)[:490]


def test_complete_ranking_rejects_population_or_score_contract_errors() -> None:
    population = [f"d{index:03d}" for index in range(520)]
    with pytest.raises(ValueError, match="same complete population"):
        assemble_complete_ranking(population, population[:-1], {"d010": 1.0})
    with pytest.raises(ValueError, match="protected RRF head"):
        assemble_complete_ranking(population, population, {"d000": 1.0})
    with pytest.raises(ValueError, match="non-finite"):
        assemble_complete_ranking(population, population, {"d010": float("nan")})


def test_top4_aggregation_matches_frozen_weighting() -> None:
    assert aggregate_top4((10.0, 8.0, 6.0, 4.0)) == pytest.approx(8.56)
    assert aggregate_top4((2.0,)) == 2.0
    with pytest.raises(ValueError, match="at least one"):
        aggregate_top4(())


def test_semantic_verifier_rejects_a_self_consistently_resealed_wrong_ranking(
    tmp_path, monkeypatch
) -> None:
    import hashlib
    import json

    import trec_rag.deep_facet_candidate_mixedbread as module
    import trec_rag.deep_facet_candidate_rank as rank_module
    from trec_rag.deep_facet_candidate_rank import create_seal

    topic = "999"
    monkeypatch.setattr(module, "TOPIC_IDS", (topic,))
    monkeypatch.setattr(rank_module, "TOPIC_IDS", (topic,))
    monkeypatch.setattr(module, "_validate_model_materialization", lambda: {})
    monkeypatch.setattr(module, "_verify_preflight_semantics", lambda *_args: None)
    population = [f"d{index:04d}" for index in range(520)]
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for arm, order in {
        "RRF": population,
        "GLOBAL": population[20:] + population[:20],
        "FACET": population,
        "DUAL": list(reversed(population)),
        "DUAL-NR": population,
    }.items():
        rows.extend(
            {"topic_id": topic, "arm": arm, "rank": rank, "document_id": document_id}
            for rank, document_id in enumerate(order, start=1)
        )
    (source / "rankings.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    create_seal(manifest_path=manifest, source_dirs=[], freeze_dir=source, topic_ids=(topic,))

    residual = build_residual_pool(
        population, population[20:] + population[:20], list(reversed(population))
    )
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    candidates = b""
    windows = b"".join(
        json.dumps(
            {
                "topic_id": topic,
                "document_id": document_id,
                "chunk_index": 0,
                "query_sha256": "q" * 64,
                "window_text_sha256": f"{index:064x}",
                "cache_hit": False,
            },
            sort_keys=True,
        ).encode()
        + b"\n"
        for index, document_id in enumerate(residual)
    )
    (preflight / "candidates.jsonl").write_bytes(candidates)
    (preflight / "windows.jsonl").write_bytes(windows)
    artifacts = {
        "candidates.jsonl": {"bytes": 0, "sha256": hashlib.sha256(candidates).hexdigest()},
        "windows.jsonl": {"bytes": len(windows), "sha256": hashlib.sha256(windows).hexdigest()},
    }
    (preflight / "preflight.json").write_text(
        json.dumps(
            {
                "schema_version": module.SCHEMA_VERSION,
                "status": "preflight_complete",
                "qrels_read": False,
                "topic_ids": [topic],
                "model": {},
                "scoring_policy": {
                    "score_representation": module.SCORE_REPRESENTATION,
                    "inference_dtype": module.MODEL_DTYPE,
                    "max_length": module.MAX_LENGTH,
                    "chunk_max_characters": module.CHUNK_MAX_CHARACTERS,
                    "chunk_overlap_characters": module.CHUNK_OVERLAP_CHARACTERS,
                    "aggregation": "chunk_top4_weighted",
                    "top4_weights": list(module.TOP4_WEIGHTS),
                    "batch_size": module.BATCH_SIZE,
                },
                "summary": {
                    "window_count": len(residual),
                    "cache_hit_window_count": 0,
                    "uncached_window_count": len(residual),
                },
                "artifacts": artifacts,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    score_bytes = b"".join(
        json.dumps(
                {
                    **module._score_policy_identity(),
                    "topic_id": topic,
                "document_id": document_id,
                "chunk_index": 0,
                "query_sha256": "q" * 64,
                    "window_text_sha256": f"{index:064x}",
                    "score": float(index),
                    "score_source": "inference",
            },
            sort_keys=True,
        ).encode()
        + b"\n"
        for index, document_id in enumerate(residual)
    )
    (scoring / "scores.jsonl").write_bytes(score_bytes)
    score_artifact = {
        "bytes": len(score_bytes),
        "sha256": hashlib.sha256(score_bytes).hexdigest(),
    }
    (scoring / "receipt.json").write_text(
        json.dumps(
            {
                **module._score_policy_identity(),
                "status": "complete",
                "post_qrels_diagnostic": True,
                "qrels_read": False,
                "network_access": False,
                "retrieval_calls": 0,
                "paid_cost_usd": 0,
                "window_count": len(residual),
                "cache_reused_window_count": 0,
                "inferred_window_count": len(residual),
                "preflight_sha256": hashlib.sha256(
                    (preflight / "preflight.json").read_bytes()
                ).hexdigest(),
                "windows_sha256": artifacts["windows.jsonl"]["sha256"],
                "scores": score_artifact,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "freeze"
    freeze_rankings(
        preflight_dir=preflight,
        scoring_dir=scoring,
        source_freeze_dir=source,
        output_dir=output,
    )

    ranking_rows = [json.loads(line) for line in (output / "rankings.jsonl").read_text().splitlines()]
    ranking_rows[0]["document_id"], ranking_rows[1]["document_id"] = (
        ranking_rows[1]["document_id"],
        ranking_rows[0]["document_id"],
    )
    ranking_bytes = b"".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        for row in ranking_rows
    )
    (output / "rankings.jsonl").write_bytes(ranking_bytes)
    seal_path = output / "SEALED.json"
    seal = json.loads(seal_path.read_text())
    seal["artifacts"]["rankings.jsonl"] = {
        "bytes": len(ranking_bytes),
        "sha256": hashlib.sha256(ranking_bytes).hexdigest(),
    }
    material = {
        "topic_ids": seal["topic_ids"],
        "source_seal_root_sha256": seal["source_seal_root_sha256"],
        "artifacts": seal["artifacts"],
    }
    seal["root_sha256"] = hashlib.sha256(
        json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")

    verify_ranking_freeze(output)
    with pytest.raises(ValueError, match="semantic recomputation"):
        verify_ranking_semantics(output)
