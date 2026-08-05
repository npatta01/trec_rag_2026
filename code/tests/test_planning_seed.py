from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import socket
import urllib.request

import pytest

import trec_rag.planning_seed as planning_seed
import trec_rag.facet_extraction as facet_extraction
from trec_rag.facet_extraction import plan_facet_queries
from trec_rag.pipeline_models import jsonable
from trec_rag.planning_seed import (
    PlanningLaneRecord,
    PlanningSeedProvenance,
    PlanningSeedRequest,
    SeedSource,
    import_planning_seed,
    load_seeded_decompositions,
    planning_seed_source_inventory_sha256,
    verify_planning_seed,
)
from trec_rag.topics import Topic

# No real 22-topic metadata golden is asserted here: stable source fixtures are
# not checked into this repository, and the prior receipt hash binds private
# result bytes, manifests, and the live validator/projector module identities.


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _decomposition_bytes(topic: Topic, bm25_counts: tuple[int, ...]) -> bytes:
    subnarratives = [
        {
            "subnarrative": f"Scope {topic.id} facet {facet_index}.",
            "bm25_queries": [
                f"topic {topic.id} facet {facet_index} query {query_index}"
                for query_index in range(1, bm25_count + 1)
            ],
        }
        for facet_index, bm25_count in enumerate(bm25_counts, start=1)
    ]
    planning = plan_facet_queries(
        topic,
        {
            "schema_version": "subnarrative_queries_v1",
            "topic_id": topic.id,
            "subnarratives": subnarratives,
        },
    )
    assert not planning.used_fallback
    assert planning.plan is not None
    return _json_bytes(
        {
            "schema_version": "facet_pilot_v2",
            "topic": {"id": topic.id, "narrative": topic.narrative},
            "narrative_sha256": sha256(topic.narrative.encode("utf-8")).hexdigest(),
            "used_fallback": False,
            "error": None,
            "queries": jsonable(planning.queries),
            "plan": {
                "schema_version": "subnarrative_queries_v1",
                "topic_id": topic.id,
                "subnarratives": subnarratives,
            },
            "subnarratives": jsonable(planning.subnarratives),
        }
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _hand_built_lane_records(
    topic: Topic,
    *,
    topic_order: int,
    subnarrative_count: int,
) -> tuple[PlanningLaneRecord, ...]:
    """Build expected lane identities without invoking the production projector."""
    records = [
        PlanningLaneRecord(
            topic_id=topic.id,
            topic_order=topic_order,
            lane_order=0,
            lane_name="original",
            subnarrative_id=None,
            retrieval_source_type="original_topic",
            retrieval_query_sha256=sha256(topic.narrative.encode("utf-8")).hexdigest(),
            scoring_source_type="semantic_original",
            scoring_query_sha256=sha256(topic.narrative.encode("utf-8")).hexdigest(),
        )
    ]
    for lane_order in range(1, subnarrative_count + 1):
        subnarrative_id = f"subnarrative-{lane_order}"
        text = f"Scope {topic.id} facet {lane_order}."
        text_sha256 = sha256(text.encode("utf-8")).hexdigest()
        records.append(
            PlanningLaneRecord(
                topic_id=topic.id,
                topic_order=topic_order,
                lane_order=lane_order,
                lane_name=f"facet:{subnarrative_id}:text",
                subnarrative_id=subnarrative_id,
                retrieval_source_type="generated_subnarrative",
                retrieval_query_sha256=text_sha256,
                scoring_source_type="generated_subnarrative",
                scoring_query_sha256=text_sha256,
            )
        )
    return tuple(records)


def _request(
    tmp_path: Path,
    *,
    final_topic_subnarrative_count: int = 5,
) -> tuple[PlanningSeedRequest, dict[str, bytes]]:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    topics = tuple(
        Topic(
            id=f"rag2026-{index}",
            title=f"Synthetic topic {index}",
            narrative=f"Explain synthetic topic {index} without private corpus text.",
        )
        for index in range(22)
    )
    sources: list[SeedSource] = []
    source_bytes: dict[str, bytes] = {}
    required_lanes: list[PlanningLaneRecord] = []
    global_subnarrative_index = 0
    planner_query_count = 0
    for index, topic in enumerate(topics):
        relative = f"{topic.id}/decomposition/result.json"
        subnarrative_count = 6 if index < 6 else 5
        if index == len(topics) - 1:
            subnarrative_count = final_topic_subnarrative_count
        bm25_counts: list[int] = []
        for _subnarrative in range(subnarrative_count):
            bm25_counts.append(3 if global_subnarrative_index < 29 else 2)
            global_subnarrative_index += 1
        body = _decomposition_bytes(topic, tuple(bm25_counts))
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        source_bytes[topic.id] = body
        sources.append(SeedSource(topic, relative, relative))
        loaded = planning_seed._validate_with_production_loader(topic, path)
        planner_query_count += len(loaded.result.queries)
        required_lanes.extend(
            _hand_built_lane_records(
                topic,
                topic_order=index,
                subnarrative_count=subnarrative_count,
            )
        )
    if final_topic_subnarrative_count == 5:
        assert planner_query_count == 283
        assert len(required_lanes) == 138
        assert len({row.retrieval_query_sha256 for row in required_lanes}) == 138
    source_tuple = tuple(sources)
    return (
        PlanningSeedRequest(
            source_root=source_root,
            destination_root=destination_root,
            topics=topics,
            sources=source_tuple,
            required_lane_records=tuple(required_lanes),
            provenance=PlanningSeedProvenance(
                source_run_id="legacy-run-2025-01",
                source_commit="a" * 40,
                source_config_sha256="b" * 64,
                source_inventory_sha256=planning_seed_source_inventory_sha256(
                    source_root, source_tuple
                ),
            ),
        ),
        source_bytes,
    )


def test_import_is_byte_identical_receipted_and_zero_call(tmp_path: Path, monkeypatch) -> None:
    request, source_bytes = _request(tmp_path)
    calls = 0

    def poison_backend(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("planning backend must not be called")

    import trec_rag.competition_retrieval as production

    monkeypatch.setattr(production, "extract_facets", poison_backend)
    receipt = import_planning_seed(request)

    assert receipt.content["mode"] == "validated-decomposition-byte-import-v2"
    assert receipt.content["planner_query_variant_count"] == 283
    assert receipt.content["retrieval_lane_occurrence_count"] == 138
    assert receipt.content["unique_retrieval_query_hash_count"] == 138
    assert receipt.content["planning_backend_invocations"] == 0
    assert receipt.content["hosted_planning_calls"] == 0
    assert receipt.content["poison_backend_result"] == "passed"
    assert receipt.content["historical_provenance_status"] == "declared-only"
    assert calls == 0
    for topic in request.topics:
        destination = request.destination_root / topic.id / "decomposition" / "result.json"
        assert destination.read_bytes() == source_bytes[topic.id]
    assert receipt.path == request.destination_root / "planning-seed-receipt.json"
    assert receipt.path.is_file()
    manifest = json.loads(
        (
            request.destination_root
            / request.topics[0].id
            / "planning-seed-manifest.json"
        ).read_bytes()
    )
    assert manifest["mode"] == "validated-decomposition-byte-import-v2"
    assert manifest["source_sha256"] == manifest["destination_sha256"]
    assert manifest["source_bytes"] == manifest["destination_bytes"]
    assert manifest["canonical_plan_payload_sha256"]
    assert manifest["rendered_planner_query_records_sha256"]
    assert manifest["retrieval_lane_records_sha256"]
    assert manifest["retrieval_lane_records"]
    assert manifest["validator_module_sha256"]
    assert manifest["lane_projector_module_sha256"]
    assert verify_planning_seed(request).content == receipt.content
    assert len(load_seeded_decompositions(request)) == 22

    class PoisonPlanner:
        identity = {
            "backend": "poison-planner",
            "model": "must-not-run",
            "prompt_version": "must-not-run",
        }

        def extract(self, _topic: Topic) -> object:
            raise AssertionError("authenticated planning seed must not call a planner")

    seeded, resumed = production._decompose_topic(
        request.topics[0],
        request.destination_root,
        PoisonPlanner(),
    )
    assert resumed is True
    assert seeded.source_sha256 == sha256(source_bytes[request.topics[0].id]).hexdigest()


def test_runtime_seed_rejects_changed_lane_projector_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _source_bytes = _request(tmp_path)
    import_planning_seed(request)
    changed_projector = tmp_path / "changed_facet_retrieval_lanes.py"
    changed_projector.write_text("# changed projector identity\n", encoding="utf-8")

    import trec_rag.competition_retrieval as production
    import trec_rag.facet_retrieval_lanes as projector

    monkeypatch.setattr(projector, "__file__", str(changed_projector))
    with pytest.raises(ValueError, match="projector identity changed"):
        production._decompose_topic(
            request.topics[0],
            request.destination_root,
            None,
        )


def test_unrelated_seed_inventory_allows_live_topic_creation_and_resume(
    tmp_path: Path,
) -> None:
    request, _source_bytes = _request(tmp_path)
    import_planning_seed(request)
    live_topic = Topic(
        "rag2026-live-extra",
        "Live extra",
        "Explain an extra topic outside the authenticated seed inventory.",
    )

    class LivePlanner:
        identity = {
            "backend": "fixture-live-planner",
            "model": "fixture-live-v1",
            "prompt_version": "fixture-live-v1",
        }

        def __init__(self) -> None:
            self.calls: list[str] = []

        def extract(self, topic: Topic) -> object:
            self.calls.append(topic.id)
            return {
                "schema_version": "subnarrative_queries_v1",
                "topic_id": topic.id,
                "subnarratives": [
                    {
                        "subnarrative": "Extra live-planned scope.",
                        "bm25_queries": ["extra live planned scope"],
                    }
                ],
            }

    import trec_rag.competition_retrieval as production

    planner = LivePlanner()
    created, resumed = production._decompose_topic(
        live_topic,
        request.destination_root,
        planner,
    )
    restored, restored_resumed = production._decompose_topic(
        live_topic,
        request.destination_root,
        planner,
    )

    assert resumed is False
    assert restored_resumed is True
    assert restored == created
    assert planner.calls == [live_topic.id]


def test_production_shaped_lane_slice_is_hand_built_independently() -> None:
    topic = Topic(
        id="rag2026-slice",
        title="Slice",
        narrative="Official narrative for the production-shaped lane slice.",
    )
    records = _hand_built_lane_records(topic, topic_order=7, subnarrative_count=2)

    assert records == (
        PlanningLaneRecord(
            "rag2026-slice",
            7,
            0,
            "original",
            None,
            "original_topic",
            sha256(topic.narrative.encode("utf-8")).hexdigest(),
            "semantic_original",
            sha256(topic.narrative.encode("utf-8")).hexdigest(),
        ),
        PlanningLaneRecord(
            "rag2026-slice",
            7,
            1,
            "facet:subnarrative-1:text",
            "subnarrative-1",
            "generated_subnarrative",
            sha256("Scope rag2026-slice facet 1.".encode("utf-8")).hexdigest(),
            "generated_subnarrative",
            sha256("Scope rag2026-slice facet 1.".encode("utf-8")).hexdigest(),
        ),
        PlanningLaneRecord(
            "rag2026-slice",
            7,
            2,
            "facet:subnarrative-2:text",
            "subnarrative-2",
            "generated_subnarrative",
            sha256("Scope rag2026-slice facet 2.".encode("utf-8")).hexdigest(),
            "generated_subnarrative",
            sha256("Scope rag2026-slice facet 2.".encode("utf-8")).hexdigest(),
        ),
    )


def test_load_does_not_reopen_a_different_valid_same_topic_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _ = _request(tmp_path)
    import_planning_seed(request)
    destination = (
        request.destination_root
        / request.topics[0].id
        / "decomposition"
        / "result.json"
    )
    alternate = _decomposition_bytes(request.topics[0], (1, 1, 1, 1, 1, 1))
    real_loader = planning_seed._validate_with_production_loader
    changed = False

    def validate_then_replace(topic: Topic, path: Path):
        nonlocal changed
        result = real_loader(topic, path)
        if path == destination and not changed:
            changed = True
            destination.write_bytes(alternate)
        return result

    monkeypatch.setattr(planning_seed, "_validate_with_production_loader", validate_then_replace)

    with pytest.raises(ValueError, match="destination|source|changed|digest"):
        load_seeded_decompositions(request)


@pytest.mark.parametrize("mutation", ["source", "destination", "module"])
def test_import_rechecks_closure_after_aggregate_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    request, _ = _request(tmp_path)
    first_source = request.source_root / request.topics[0].id / "decomposition" / "result.json"
    first_destination = request.destination_root / request.topics[0].id / "decomposition" / "result.json"
    real_aggregate = planning_seed._aggregate_bytes

    def aggregate_then_mutate(
        current_request: PlanningSeedRequest,
        prepared: tuple[object, ...],
    ):
        result = real_aggregate(current_request, prepared)
        if mutation == "source":
            first_source.write_bytes(first_source.read_bytes() + b"late-source-mutation")
        elif mutation == "destination":
            first_destination.write_bytes(first_destination.read_bytes() + b"late-destination-mutation")
        else:
            monkeypatch.setattr(
                planning_seed,
                "_validator_identity",
                lambda: ("0" * 64, "trec_rag.competition_retrieval:load_validated_decomposition"),
            )
        return result

    monkeypatch.setattr(planning_seed, "_aggregate_bytes", aggregate_then_mutate)

    with pytest.raises(ValueError, match="source|destination|validator|changed"):
        import_planning_seed(request)
    assert not (request.destination_root / planning_seed.PLANNING_SEED_RECEIPT_FILENAME).exists()


@pytest.mark.parametrize("mutation", ["source", "destination", "module"])
def test_import_rechecks_closure_after_reading_published_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    request, _ = _request(tmp_path)
    first_source = request.source_root / request.topics[0].id / "decomposition" / "result.json"
    first_destination = request.destination_root / request.topics[0].id / "decomposition" / "result.json"
    real_read_receipt = planning_seed._read_receipt

    def read_then_mutate(path: Path, expected: object):
        result = real_read_receipt(path, expected)
        if mutation == "source":
            first_source.write_bytes(first_source.read_bytes() + b"post-receipt-source-mutation")
        elif mutation == "destination":
            first_destination.write_bytes(
                first_destination.read_bytes() + b"post-receipt-destination-mutation"
            )
        else:
            monkeypatch.setattr(
                planning_seed,
                "_validator_identity",
                lambda: (
                    "0" * 64,
                    "trec_rag.competition_retrieval:load_validated_decomposition",
                ),
            )
        return result

    monkeypatch.setattr(planning_seed, "_read_receipt", read_then_mutate)

    with pytest.raises(ValueError, match="source|destination|validator|changed"):
        import_planning_seed(request)


@pytest.mark.parametrize("mutation", ["source", "destination", "module"])
def test_verify_rechecks_closure_after_aggregate_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    request, _ = _request(tmp_path)
    import_planning_seed(request)
    first_source = request.source_root / request.topics[0].id / "decomposition" / "result.json"
    first_destination = request.destination_root / request.topics[0].id / "decomposition" / "result.json"
    real_aggregate = planning_seed._aggregate_bytes

    def aggregate_then_mutate(
        current_request: PlanningSeedRequest,
        prepared: tuple[object, ...],
    ):
        result = real_aggregate(current_request, prepared)
        if mutation == "source":
            first_source.write_bytes(first_source.read_bytes() + b"late-source-mutation")
        elif mutation == "destination":
            first_destination.write_bytes(first_destination.read_bytes() + b"late-destination-mutation")
        else:
            monkeypatch.setattr(
                planning_seed,
                "_validator_identity",
                lambda: ("0" * 64, "trec_rag.competition_retrieval:load_validated_decomposition"),
            )
        return result

    monkeypatch.setattr(planning_seed, "_aggregate_bytes", aggregate_then_mutate)

    with pytest.raises(ValueError, match="source|destination|validator|changed"):
        verify_planning_seed(request)


@pytest.mark.parametrize("artifact", ["result", "manifest", "aggregate"])
def test_v1_result_manifest_and_aggregate_artifacts_are_rejected(
    tmp_path: Path,
    artifact: str,
) -> None:
    request, _ = _request(tmp_path)
    if artifact == "result":
        path = request.source_root / request.topics[0].id / "decomposition" / "result.json"
        value = json.loads(path.read_bytes())
        value["schema_version"] = "facet_pilot_v1"
        path.write_bytes(_canonical_bytes(value))
        with pytest.raises(ValueError, match="schema|decomposition|source"):
            import_planning_seed(request)
        return

    import_planning_seed(request)
    if artifact == "manifest":
        path = request.destination_root / request.topics[0].id / "planning-seed-manifest.json"
    else:
        path = request.destination_root / planning_seed.PLANNING_SEED_RECEIPT_FILENAME
    value = json.loads(path.read_bytes())
    value["schema_version"] = "planning_seed_manifest_v1" if artifact == "manifest" else "planning_seed_receipt_v1"
    path.write_bytes(_canonical_bytes(value))
    with pytest.raises(ValueError, match="schema|manifest|aggregate|changed"):
        verify_planning_seed(request)


def test_import_poison_boundaries_prove_no_model_or_network_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _ = _request(tmp_path)
    import trec_rag.competition_retrieval as production

    def poison(*args: object, **kwargs: object) -> object:
        raise AssertionError("planning seed crossed a model or network boundary")

    monkeypatch.setattr(production, "extract_facets", poison)
    monkeypatch.setattr(facet_extraction, "extract_facets", poison)
    monkeypatch.setattr(production, "OpenRouterDeepSeekFacetBackend", poison)
    monkeypatch.setattr(facet_extraction, "OpenRouterDeepSeekFacetBackend", poison)
    monkeypatch.setattr(urllib.request, "urlopen", poison)
    monkeypatch.setattr(socket, "create_connection", poison)
    monkeypatch.setattr(socket, "getaddrinfo", poison)

    receipt = import_planning_seed(request)
    assert receipt.content["planning_backend_invocations"] == 0
    assert receipt.content["hosted_planning_calls"] == 0


def test_existing_unreceipted_result_and_rewritten_json_fail_closed(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    first = request.destination_root / request.topics[0].id / "decomposition" / "result.json"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"unreceipted")
    with pytest.raises(ValueError, match="unreceipted|manifest"):
        import_planning_seed(request)

    request, _ = _request(tmp_path / "rewritten")
    import_planning_seed(request)
    destination = request.destination_root / request.topics[0].id / "decomposition" / "result.json"
    destination.write_bytes(json.dumps(json.loads(destination.read_bytes())).encode("utf-8"))
    with pytest.raises(ValueError, match="byte|digest|result"):
        verify_planning_seed(request)


@pytest.mark.parametrize("mutation", ["source", "topic", "manifest", "aggregate"])
def test_tampering_and_contradictory_republish_fail_closed(tmp_path: Path, mutation: str) -> None:
    request, _ = _request(tmp_path)
    import_planning_seed(request)
    topic = request.topics[0]
    source = request.source_root / topic.id / "decomposition" / "result.json"
    destination = request.destination_root / topic.id / "decomposition" / "result.json"
    manifest = request.destination_root / topic.id / "planning-seed-manifest.json"
    aggregate = request.destination_root / "planning-seed-receipt.json"
    if mutation == "source":
        source.write_bytes(source.read_bytes() + b"changed")
    elif mutation == "topic":
        changed_topics = list(request.topics)
        changed_topics[0] = Topic(topic.id, topic.title, "changed official narrative")
        request = replace(request, topics=tuple(changed_topics))
    elif mutation == "manifest":
        manifest.write_bytes(manifest.read_bytes().replace(b"facet_pilot_v2", b"changed_schema"))
    else:
        aggregate.write_bytes(aggregate.read_bytes().replace(b"legacy-run-2025-01", b"other-run"))
    with pytest.raises(ValueError, match="changed|digest|manifest|narrative|source|JSON"):
        verify_planning_seed(request)
    with pytest.raises(ValueError):
        import_planning_seed(request)
    assert destination.exists()


def test_request_requires_exactly_22_topics_and_authenticated_138_lanes(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    with pytest.raises(ValueError, match="22"):
        import_planning_seed(replace(request, topics=request.topics[:-1], sources=request.sources[:-1]))
    with pytest.raises(ValueError, match="138|lane"):
        import_planning_seed(
            replace(request, required_lane_records=request.required_lane_records[:-1])
        )


@pytest.mark.parametrize("final_count, expected_lanes", [(4, 137), (6, 139)])
def test_projected_137_or_139_lane_sources_fail_closed(
    tmp_path: Path,
    final_count: int,
    expected_lanes: int,
) -> None:
    request, _ = _request(
        tmp_path,
        final_topic_subnarrative_count=final_count,
    )
    assert len(request.required_lane_records) == expected_lanes

    with pytest.raises(ValueError, match="138|lane"):
        import_planning_seed(request)


@pytest.mark.parametrize(
    "mutation",
    [
        "order",
        "lane_name",
        "retrieval_source",
        "retrieval_hash",
        "scoring_source",
        "scoring_hash",
        "subnarrative",
    ],
)
def test_required_lane_binding_is_ordered_and_exact(
    tmp_path: Path,
    mutation: str,
) -> None:
    request, _ = _request(tmp_path)
    records = list(request.required_lane_records)
    if mutation == "order":
        records[1], records[2] = records[2], records[1]
    elif mutation == "lane_name":
        records[1] = replace(records[1], lane_name="facet:changed:text")
    elif mutation == "retrieval_source":
        records[1] = replace(records[1], retrieval_source_type="changed-source")
    elif mutation == "retrieval_hash":
        records[1] = replace(records[1], retrieval_query_sha256="0" * 64)
    elif mutation == "scoring_source":
        records[1] = replace(records[1], scoring_source_type="changed-source")
    elif mutation == "scoring_hash":
        records[1] = replace(records[1], scoring_query_sha256="0" * 64)
    else:
        records[1] = replace(records[1], subnarrative_id="changed-subnarrative")

    with pytest.raises(ValueError, match="lane|request"):
        import_planning_seed(replace(request, required_lane_records=tuple(records)))


def test_source_inventory_digest_must_match_the_exact_source_files(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    request = replace(
        request,
        provenance=PlanningSeedProvenance(
            request.provenance.source_run_id,
            request.provenance.source_commit,
            request.provenance.source_config_sha256,
            "c" * 64,
        ),
    )

    with pytest.raises(ValueError, match="source inventory"):
        import_planning_seed(request)


def test_unsafe_paths_duplicate_hashes_and_symlinks_fail_closed(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    bad_source = SeedSource(
        request.topics[0],
        "../outside/result.json",
        request.sources[0].destination_relative_path,
    )
    with pytest.raises(ValueError, match="path"):
        import_planning_seed(replace(request, sources=(bad_source,) + request.sources[1:]))

    duplicate_lanes = request.required_lane_records[:-1] + (request.required_lane_records[0],)
    with pytest.raises(ValueError, match="duplicate|request"):
        import_planning_seed(replace(request, required_lane_records=duplicate_lanes))

    symlink_root = tmp_path / "symlink-source"
    symlink_root.symlink_to(request.source_root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|path"):
        import_planning_seed(replace(request, source_root=symlink_root))


def test_topic_paths_must_use_the_canonical_decomposition_layout(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    source = request.sources[0]
    misplaced = SeedSource(
        source.topic,
        source.source_relative_path,
        f"another-topic/decomposition/result.json",
    )

    with pytest.raises(ValueError, match="canonical topic decomposition path"):
        import_planning_seed(replace(request, sources=(misplaced,) + request.sources[1:]))


def test_source_change_before_aggregate_prevents_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = _request(tmp_path)
    first_source = (
        request.source_root
        / request.topics[0].id
        / "decomposition"
        / "result.json"
    )
    real_publish = planning_seed._publish_create_only
    changed = False

    def publish_then_change(path: Path, body: bytes) -> None:
        nonlocal changed
        real_publish(path, body)
        if not changed and path.name == planning_seed.PLANNING_SEED_MANIFEST_FILENAME:
            changed = True
            first_source.write_bytes(first_source.read_bytes() + b"changed")

    monkeypatch.setattr(planning_seed, "_publish_create_only", publish_then_change)

    with pytest.raises(ValueError, match="source changed"):
        import_planning_seed(request)

    assert not (
        request.destination_root / planning_seed.PLANNING_SEED_RECEIPT_FILENAME
    ).exists()


def test_lane_projector_change_before_aggregate_prevents_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _ = _request(tmp_path)
    real_identity = planning_seed._lane_projector_identity
    calls = 0

    def changing_identity() -> tuple[str, str]:
        nonlocal calls
        calls += 1
        digest, identity = real_identity()
        return (digest if calls == 1 else "0" * 64, identity)

    monkeypatch.setattr(planning_seed, "_lane_projector_identity", changing_identity)

    with pytest.raises(ValueError, match="projector changed"):
        import_planning_seed(request)

    assert not (
        request.destination_root / planning_seed.PLANNING_SEED_RECEIPT_FILENAME
    ).exists()


def test_concurrent_identical_imports_converge(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = list(executor.map(lambda _index: import_planning_seed(request), range(2)))

    assert receipts[0].content_sha256 == receipts[1].content_sha256
    assert verify_planning_seed(request).content_sha256 == receipts[0].content_sha256
