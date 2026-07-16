from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

import trec_rag.tethered_facet_two_basket as module

from trec_rag.tethered_facet_two_basket import (
    FacetScores,
    TopicInput,
    average_rank_percentiles,
    build_facet_basket,
    build_two_basket_permutation,
    freeze_rankings,
    load_frozen_inputs,
    main,
    topic_quotas,
    verify_freeze,
)


def test_average_rank_percentiles_are_query_local_and_tie_aware() -> None:
    assert average_rank_percentiles({"a": 3.0, "b": 3.0, "c": 1.0}) == {
        "a": pytest.approx(5 / 6),
        "b": pytest.approx(5 / 6),
        "c": pytest.approx(1 / 3),
    }


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (7, [29, 29, 29, 29, 28, 28, 28]),
        (4, [50, 50, 50, 50]),
        (6, [34, 34, 33, 33, 33, 33]),
    ],
)
def test_topic_quotas_total_200(count: int, expected: list[int]) -> None:
    assert topic_quotas(count, 200) == expected
    assert sum(expected) == 200


def _facet(
    facet_number: int,
    document_ids: list[str],
    *,
    score_offset: float = 0.0,
) -> FacetScores:
    return FacetScores(
        facet_id=f"219-facet-{facet_number}",
        manifest_order=facet_number,
        scores={
            document_id: score_offset + float(len(document_ids) - index)
            for index, document_id in enumerate(document_ids)
        },
        bm25_ranks={document_id: index + 1 for index, document_id in enumerate(document_ids)},
        query_sha256="1" * 63 + str(facet_number),
        text_sha256={document_id: f"{index:064x}" for index, document_id in enumerate(document_ids)},
        model="cross-encoder/ms-marco-MiniLM-L6-v2",
        model_revision="revision",
        score_schema_version="tethered-facet-minilm-document-score-v1",
    )


def fixture_topic() -> TopicInput:
    accepted = [f"d{index:03d}" for index in range(650)]
    eligible = accepted[300:550]
    return TopicInput(
        topic_id="219",
        accepted_union=tuple(accepted),
        rrf=tuple(accepted),
        dual=tuple(reversed(accepted)),
        facets=tuple(
            _facet(number, eligible, score_offset=number * 1000.0)
            for number in range(4)
        ),
    )


def shuffled_fixture_topic() -> TopicInput:
    topic = fixture_topic()
    return replace(
        topic,
        accepted_union=tuple(reversed(topic.accepted_union)),
        facets=tuple(
            replace(
                facet,
                scores=dict(reversed(list(facet.scores.items()))),
                bm25_ranks=dict(reversed(list(facet.bm25_ranks.items()))),
                text_sha256=dict(reversed(list(facet.text_sha256.items()))),
            )
            for facet in reversed(topic.facets)
        ),
    )


def fixture_topics() -> list[TopicInput]:
    return [replace(fixture_topic(), topic_id=topic_id) for topic_id in ("219", "72", "300", "84")]


def test_facet_basket_uses_query_local_percentiles_and_manifest_quotas() -> None:
    result = build_facet_basket(fixture_topic())

    assert len(result.document_ids) == 200
    assert len(set(result.document_ids)) == 200
    assert not set(result.document_ids) & set(fixture_topic().rrf[:300])
    counts = {
        facet_id: sum(row.facet_id == facet_id for row in result.selections)
        for facet_id in {row.facet_id for row in result.selections}
    }
    assert counts == {f"219-facet-{number}": 50 for number in range(4)}
    assert result.selections[0].percentile == pytest.approx(1.0)


def test_facet_basket_redistributes_shortages_one_slot_per_manifest_round() -> None:
    topic = fixture_topic()
    scarce = _facet(0, list(topic.accepted_union[300:310]))
    plentiful = tuple(
        _facet(number, list(topic.accepted_union[310:610]), score_offset=number * 1000)
        for number in range(1, 4)
    )
    result = build_facet_basket(replace(topic, facets=(scarce, *plentiful)))

    assert len(result.document_ids) == 200
    scarce_rows = [row for row in result.selections if row.facet_id == scarce.facet_id]
    assert len(scarce_rows) == 10
    assert all(row.nominal_quota == 50 and row.shortage == 40 for row in scarce_rows)
    assert any(row.nominal_quota == 50 and row.shortage == 0 for row in result.selections)
    assert sum(row.duplicate_skip_delta for row in result.selections) > 0


def test_shortage_records_global_duplicate_attribution_not_candidate_count() -> None:
    topic = fixture_topic()
    scarce = _facet(0, list(topic.accepted_union[300:310]))
    plentiful = tuple(
        _facet(number, list(topic.accepted_union[300:600]), score_offset=number * 1000)
        for number in range(1, 4)
    )

    result = build_facet_basket(replace(topic, facets=(scarce, *plentiful)))
    scarce_rows = [row for row in result.selections if row.facet_id == scarce.facet_id]

    assert len(scarce_rows) == 1
    assert scarce_rows[0].shortage == 49
    assert dict(result.duplicate_skip_totals)[scarce.facet_id] > 0


def test_redistribution_counts_each_duplicate_edge_once_and_totals_reconcile() -> None:
    topic = fixture_topic()
    scarce = _facet(0, list(topic.accepted_union[300:310]))
    plentiful = tuple(
        _facet(number, list(topic.accepted_union[310:610]), score_offset=number * 1000)
        for number in range(1, 4)
    )

    result = build_facet_basket(replace(topic, facets=(scarce, *plentiful)))
    totals = dict(result.duplicate_skip_totals)

    assert totals == {
        scarce.facet_id: 0,
        plentiful[0].facet_id: 126,
        plentiful[1].facet_id: 125,
        plentiful[2].facet_id: 126,
    }
    for facet in (scarce, *plentiful):
        rows = [row for row in result.selections if row.facet_id == facet.facet_id]
        assert sum(row.duplicate_skip_delta for row in rows) == totals[facet.facet_id]
        assert totals[facet.facet_id] <= len(facet.scores)
    assert sum(totals.values()) > 0


def test_two_basket_ranking_protects_head_and_is_complete() -> None:
    result = build_two_basket_permutation(fixture_topic())
    assert result.document_ids[:100] == fixture_topic().rrf[:100]
    assert result.sources[100:500:2] == ("rrf_basket",) * 200
    assert result.sources[101:500:2] == ("facet_basket",) * 200
    assert len(result.document_ids) == len(set(result.document_ids))
    assert set(result.document_ids) == set(fixture_topic().accepted_union)
    assert result.document_ids[500:] == tuple(
        document_id
        for document_id in fixture_topic().dual
        if document_id not in set(result.document_ids[:500])
    )


def test_ranking_is_invariant_to_input_row_order() -> None:
    assert build_two_basket_permutation(shuffled_fixture_topic()) == (
        build_two_basket_permutation(fixture_topic())
    )


def test_ranking_rejects_nonpilot_and_incomplete_populations() -> None:
    topic = fixture_topic()
    with pytest.raises(ValueError, match="protected pilot"):
        build_two_basket_permutation(replace(topic, topic_id="144"))
    with pytest.raises(ValueError, match="complete permutation"):
        build_two_basket_permutation(replace(topic, dual=topic.dual[:-1]))


def test_freeze_is_create_only_and_verifier_rejects_mutation(tmp_path: Path) -> None:
    output = tmp_path / "freeze"
    binding = tmp_path / "task2-document-scores.jsonl"
    binding.write_text('{"synthetic":true}\n', encoding="utf-8")
    summary = freeze_rankings(
        facet_topics=fixture_topics(),
        tethered_topics=fixture_topics(),
        input_paths={"task2_document_scores": binding},
        output=output,
    )

    assert summary["arms"] == ["FACET-2B", "TETHERED-2B"]
    assert summary["topic_summary"]["219"]["duplicate_skip_totals"]["FACET-2B"] == {
        "219-facet-0": 0,
        "219-facet-1": 50,
        "219-facet-2": 100,
        "219-facet-3": 150,
    }
    ranking_rows = [
        json.loads(line)
        for line in (output / "rankings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    facet_rows = [row for row in ranking_rows if row["source"] == "facet_basket"]
    assert facet_rows
    assert all("duplicate_skip_delta" in row for row in facet_rows)
    assert all("duplicate_skips" not in row for row in ranking_rows)
    assert all("duplicate_skip_totals" not in row for row in ranking_rows)
    assert verify_freeze(output)["ranking_row_count"] == 5200
    with pytest.raises(FileExistsError, match="create-only"):
        freeze_rankings(
            facet_topics=fixture_topics(),
            tethered_topics=fixture_topics(),
            input_paths={"task2_document_scores": binding},
            output=output,
        )

    rankings = output / "rankings.jsonl"
    rows = rankings.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["document_id"] = "tampered"
    rows[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    rankings.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_freeze(output)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_verifier_rejects_semantic_tamper_after_all_hashes_are_restamped(
    tmp_path: Path,
) -> None:
    output = tmp_path / "freeze"
    binding = tmp_path / "source.jsonl"
    binding.write_text("{}\n", encoding="utf-8")
    freeze_rankings(
        facet_topics=fixture_topics(),
        tethered_topics=fixture_topics(),
        input_paths={"source": binding},
        output=output,
    )
    rankings = output / "rankings.jsonl"
    rows = rankings.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["rrf_rank"] = 650
    rows[0] = _canonical(first).decode()
    rankings.write_text("\n".join(rows) + "\n", encoding="utf-8")

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ranking_bytes = rankings.read_bytes()
    summary["artifacts"]["rankings.jsonl"] = {
        "bytes": len(ranking_bytes),
        "sha256": _sha(ranking_bytes),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    seal_path = output / "SEALED.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for name in ("rankings.jsonl", "summary.json"):
        content = (output / name).read_bytes()
        seal["files"][name] = {"bytes": len(content), "sha256": _sha(content)}
    material = {
        key: seal[key]
        for key in ("schema_version", "status", "qrels_opened", "files")
    }
    seal["root_sha256"] = _sha(_canonical(material))
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="semantic"):
        verify_freeze(output)


def test_freeze_requires_every_protected_pilot_topic(tmp_path: Path) -> None:
    binding = tmp_path / "source.jsonl"
    binding.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact protected pilot topics"):
        freeze_rankings(
            facet_topics=[fixture_topic()],
            tethered_topics=[fixture_topic()],
            input_paths={"source": binding},
            output=tmp_path / "freeze",
        )


def test_freeze_requires_identical_arm_pair_coverage_and_task2_schema(
    tmp_path: Path,
) -> None:
    binding = tmp_path / "source.jsonl"
    binding.write_text("{}\n", encoding="utf-8")
    facet_topics = fixture_topics()
    first = facet_topics[0]
    shortened = replace(
        first.facets[0],
        scores=dict(list(first.facets[0].scores.items())[:-1]),
        bm25_ranks=dict(list(first.facets[0].bm25_ranks.items())[:-1]),
        text_sha256=dict(list(first.facets[0].text_sha256.items())[:-1]),
    )
    mismatched = [replace(first, facets=(shortened, *first.facets[1:])), *fixture_topics()[1:]]
    with pytest.raises(ValueError, match="pair coverage"):
        freeze_rankings(
            facet_topics=facet_topics,
            tethered_topics=mismatched,
            input_paths={"source": binding},
            output=tmp_path / "coverage-freeze",
        )

    wrong_schema = replace(
        first.facets[0], score_schema_version="not-task-2-document-scores"
    )
    bad_schema = [replace(first, facets=(wrong_schema, *first.facets[1:])), *fixture_topics()[1:]]
    with pytest.raises(ValueError, match="Task 2 score schema"):
        freeze_rankings(
            facet_topics=facet_topics,
            tethered_topics=bad_schema,
            input_paths={"source": binding},
            output=tmp_path / "schema-freeze",
        )


def test_freeze_bytes_are_invariant_to_all_input_row_orders(tmp_path: Path) -> None:
    binding = tmp_path / "source.jsonl"
    binding.write_text("{}\n", encoding="utf-8")
    canonical = fixture_topics()
    shuffled = [
        replace(shuffled_fixture_topic(), topic_id=topic_id)
        for topic_id in ("84", "300", "72", "219")
    ]
    first = tmp_path / "first"
    second = tmp_path / "second"

    freeze_rankings(
        facet_topics=canonical,
        tethered_topics=canonical,
        input_paths={"source": binding},
        output=first,
    )
    freeze_rankings(
        facet_topics=shuffled,
        tethered_topics=list(reversed(shuffled)),
        input_paths={"source": binding},
        output=second,
    )

    for name in (
        "parameters.json",
        "input_bindings.json",
        "rankings.jsonl",
        "prefixes.json",
        "summary.json",
        "SEALED.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_verifier_rejects_restamped_false_parameters_and_summary(tmp_path: Path) -> None:
    output = tmp_path / "freeze"
    binding = tmp_path / "source.jsonl"
    binding.write_text("{}\n", encoding="utf-8")
    freeze_rankings(
        facet_topics=fixture_topics(),
        tethered_topics=fixture_topics(),
        input_paths={"source": binding},
        output=output,
    )
    parameters_path = output / "parameters.json"
    parameters = json.loads(parameters_path.read_text())
    parameters["tail_order"] = "not_DUAL"
    parameters["prefix_depths"] = [7]
    parameters_path.write_text(json.dumps(parameters, indent=2, sort_keys=True) + "\n")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["topic_summary"]["219"]["facet_counts"] = {
        "FACET-2B": 1,
        "TETHERED-2B": 1,
    }
    parameter_bytes = parameters_path.read_bytes()
    summary["artifacts"]["parameters.json"] = {
        "bytes": len(parameter_bytes),
        "sha256": _sha(parameter_bytes),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    seal_path = output / "SEALED.json"
    seal = json.loads(seal_path.read_text())
    for name in ("parameters.json", "summary.json"):
        content = (output / name).read_bytes()
        seal["files"][name] = {"bytes": len(content), "sha256": _sha(content)}
    material = {
        key: seal[key]
        for key in ("schema_version", "status", "qrels_opened", "files")
    }
    seal["root_sha256"] = _sha(_canonical(material))
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="semantic contract"):
        verify_freeze(output)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _loader_sources(tmp_path: Path) -> tuple[Path, Path]:
    deep = tmp_path / "deep"
    tethered = tmp_path / "tethered"
    for path in (deep / "freeze_v1", deep / "gate_v1", deep / "phase1_v1", tethered):
        path.mkdir(parents=True, exist_ok=True)
    rankings: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    facet_candidates: list[dict[str, object]] = []
    facet_scores: list[dict[str, object]] = []
    tethered_candidates: list[dict[str, object]] = []
    tethered_scores: list[dict[str, object]] = []
    tethered_documents: list[dict[str, object]] = []
    for topic_index, topic in enumerate(("219", "72", "300", "84")):
        documents = [f"{topic}-d{index:03d}" for index in range(650)]
        facet_id = f"{topic}-facet"
        facet_query = f"facet query {topic}"
        narrative = f"full narrative {topic}"
        gates.append({"topic_id": topic, "facet_id": facet_id, "manifest_order": topic_index, "status": "accepted"})
        accepted.extend({"topic_id": topic, "document_id": docid} for docid in documents)
        for arm, ordered in (("RRF", documents), ("DUAL", list(reversed(documents)))):
            rankings.extend(
                {"topic_id": topic, "arm": arm, "rank": rank, "document_id": docid}
                for rank, docid in enumerate(ordered, 1)
            )
        for local_rank, docid in enumerate(documents[300:550], 1):
            text = f"authenticated text for {docid}"
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            facet_candidates.append({
                "topic_id": topic, "facet_id": facet_id, "manifest_order": topic_index,
                "document_id": docid, "docid": docid, "query": facet_query,
                "query_sha256": hashlib.sha256(facet_query.encode()).hexdigest(),
                "text": text, "text_sha256": text_hash, "rank": local_rank,
            })
            window = {
                "topic_id": topic, "facet_id": facet_id, "document_id": docid,
                "window_id": f"facet-{topic}-{local_rank}", "window_text": text,
                "window_sha256": text_hash, "document_sha256": text_hash,
                "document_start_token": 0, "document_end_token": 8,
                "query_sha256": hashlib.sha256(facet_query.encode()).hexdigest(),
                "score": float(251 - local_rank), "model": "synthetic/minilm",
                "model_revision": "fixture-revision",
            }
            facet_scores.append(window)
            full_query = narrative + "\n\nFocus: " + facet_query
            query_hash = hashlib.sha256(full_query.encode()).hexdigest()
            tethered_candidates.append({
                "schema_version": "tethered-facet-minilm-candidate-v1",
                "topic_id": topic, "facet_id": facet_id, "manifest_order": topic_index,
                "document_id": docid, "docid": docid, "facet_query": facet_query,
                "query": full_query, "query_sha256": query_hash, "text": text,
                "text_sha256": text_hash, "prior_bm25_rank": local_rank,
            })
            tethered_scores.append({
                **window, "schema_version": "tethered-facet-minilm-score-v1",
                "window_id": f"tethered-{topic}-{local_rank}",
                "query_sha256": query_hash, "score": float(local_rank),
            })
            tethered_documents.append({
                "schema_version": "tethered-facet-minilm-document-score-v1",
                "topic_id": topic, "facet_id": facet_id, "document_id": docid,
                "score": float(local_rank), "selected_window_count": 1,
                "query_sha256": query_hash, "text_sha256": text_hash,
                "model": "synthetic/minilm", "model_revision": "fixture-revision",
                "window_hashes": [text_hash],
            })
    _write_jsonl(deep / "freeze_v1" / "rankings.jsonl", rankings)
    _write_jsonl(deep / "gate_v1" / "u_accepted.jsonl", accepted)
    (deep / "gate_v1" / "gates.json").write_text(json.dumps({"gates": gates}), encoding="utf-8")
    _write_jsonl(deep / "phase1_v1" / "candidates.jsonl", facet_candidates)
    _write_jsonl(deep / "phase1_v1" / "scores.jsonl", facet_scores)
    _write_jsonl(tethered / "candidates.jsonl", tethered_candidates)
    _write_jsonl(tethered / "scores.jsonl", tethered_scores)
    _write_jsonl(tethered / "document_scores.jsonl", tethered_documents)
    (tethered / "preflight.json").write_text("{}\n", encoding="utf-8")
    (tethered / "scoring_receipt.json").write_text("{}\n", encoding="utf-8")
    (deep / "freeze_v1" / "SEALED.json").write_text("{}\n", encoding="utf-8")
    return deep, tethered


def test_real_freeze_cli_loads_authenticated_task2_and_prior_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep, tethered = _loader_sources(tmp_path)
    monkeypatch.setattr(module, "verify_prior_seal", lambda _path: {"status": "verified"})
    monkeypatch.setattr(module, "verify_scoring", lambda _path: {"status": "complete"})
    output = tmp_path / "freeze"

    assert main(["freeze", "--deep-root", str(deep), "--tethered", str(tethered), "--output", str(output)]) == 0
    assert main(["verify", "--freeze", str(output)]) == 0
    bindings = json.loads((output / "input_bindings.json").read_text())
    assert {"prior_rankings", "accepted_union", "facet_candidates", "facet_window_scores", "tethered_candidates", "tethered_window_scores", "tethered_document_scores"} <= set(bindings["inputs"])
    rows = [json.loads(line) for line in (output / "rankings.jsonl").read_text().splitlines()]
    assert any(row["arm"] == "FACET-2B" and row["source"] == "facet_basket" for row in rows)
    assert any(row["arm"] == "TETHERED-2B" and row["source"] == "facet_basket" for row in rows)

    with (tethered / "scores.jsonl").open("a", encoding="utf-8") as sink:
        sink.write("{}\n")
    with pytest.raises(ValueError, match="input SHA-256"):
        verify_freeze(output)


def test_loader_rejects_tethered_query_and_rank_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep, tethered = _loader_sources(tmp_path)
    monkeypatch.setattr(module, "verify_prior_seal", lambda _path: {"status": "verified"})
    monkeypatch.setattr(module, "verify_scoring", lambda _path: {"status": "complete"})
    candidates = (tethered / "candidates.jsonl").read_text().splitlines()
    first = json.loads(candidates[0])
    first["query"] = "invented narrative\n\nFocus: " + first["facet_query"] + "\nextra"
    first["query_sha256"] = hashlib.sha256(first["query"].encode()).hexdigest()
    candidates[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    (tethered / "candidates.jsonl").write_text("\n".join(candidates) + "\n")

    with pytest.raises(ValueError, match="exact narrative"):
        load_frozen_inputs(deep, tethered)
