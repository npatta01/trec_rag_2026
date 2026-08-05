from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from trec_rag.generation_handoff import (
    SOURCE_CONTRACT,
    ClaimHint,
    EvidenceGroup,
    EvidencePassage,
    EvidenceSourceSpan,
    GenerationHandoff,
    GenerationTopic,
    HandoffProducer,
    HandoffIntegrityError,
    SelectedCluster,
    TopicSourceReceipts,
    deserialize_generation_topic,
    load_generation_handoff,
    prompt_sha256,
    render_generation_evidence,
    select_generation_topics,
    serialize_generation_handoff,
    serialize_generation_topic,
    write_generation_handoff,
)


def _literal_handoff() -> GenerationHandoff:
    evidence = (
        EvidencePassage(
            evidence_id="ev-a",
            group_id="group-1",
            cluster_id="cluster-1",
            cluster_ordinal=1,
            support_ordinal=1,
            candidate_kind="extractive",
            docid="doc-a",
            document_rank=1,
            text="Selected alpha passage.",
            document_sha256="a" * 64,
            source_span=EvidenceSourceSpan(
                start_char=10,
                end_char=33,
                start_byte=10,
                end_byte=33,
            ),
        ),
        EvidencePassage(
            evidence_id="ev-b",
            group_id="group-1",
            cluster_id="cluster-1",
            cluster_ordinal=1,
            support_ordinal=2,
            candidate_kind="extractive",
            docid="doc-b",
            document_rank=2,
            text="Selected beta passage.",
            document_sha256="b" * 64,
            source_span=EvidenceSourceSpan(
                start_char=40,
                end_char=62,
                start_byte=40,
                end_byte=62,
            ),
        ),
    )
    group = EvidenceGroup(
        group_id="group-1",
        kind="generated_subnarrative",
        text="The first answer aspect.",
        selected_clusters=(
            SelectedCluster(
                cluster_id="cluster-1",
                ordinal=1,
                representative_evidence_id="ev-a",
                evidence_ids=("ev-a", "ev-b"),
            ),
        ),
    )
    topic = GenerationTopic(
        topic_id="rag2026-58",
        narrative="Explain the first answer aspect.",
        groups=(group,),
        evidence=evidence,
        claim_hints=(
            ClaimHint(
                claim_id="claim-1",
                group_id="group-1",
                kind="canonical",
                text="Alpha is selected.",
                evidence_ids=("ev-a",),
            ),
        ),
        source_receipts=TopicSourceReceipts(
            official_topics_sha256="1" * 64,
            retrieval_topic_sha256="2" * 64,
        ),
    )
    return GenerationHandoff(
        producer=HandoffProducer(
            source_contract=SOURCE_CONTRACT,
            retrieval_run_id="facet-deepseek-selected-evidence-v1",
            producer_revision="deadbeef",
        ),
        topics=(topic,),
    )


def test_handoff_keeps_all_selected_supports_and_serializes_deterministically() -> None:
    handoff = _literal_handoff()
    topic = handoff.topics[0]

    assert tuple(row.evidence_id for row in topic.evidence) == ("ev-a", "ev-b")
    assert topic.claim_hints[0].evidence_ids == ("ev-a",)
    assert topic.citation_docids == ("doc-a", "doc-b")
    assert len(topic.context_sha256) == 64
    assert len(handoff.manifest_sha256) == 64
    assert serialize_generation_handoff(handoff) == serialize_generation_handoff(handoff)


def test_generation_topic_projection_round_trips_exact_canonical_bytes() -> None:
    topic = _literal_handoff().topics[0]
    body = serialize_generation_topic(topic)

    assert body.endswith(b"\n")
    assert deserialize_generation_topic(body) == topic

    noncanonical = json.dumps(topic.to_payload(), indent=2).encode("utf-8") + b"\n"
    with pytest.raises(HandoffIntegrityError, match="canonical"):
        deserialize_generation_topic(noncanonical)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reseal(payload: dict[str, object]) -> bytes:
    topics = payload["topics"]
    assert isinstance(topics, list)
    for topic in topics:
        assert isinstance(topic, dict)
        topic_without_hash = {
            key: value for key, value in topic.items() if key != "context_sha256"
        }
        topic["context_sha256"] = sha256(_canonical(topic_without_hash)).hexdigest()
    root_without_hash = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    payload["manifest_sha256"] = sha256(_canonical(root_without_hash)).hexdigest()
    return _canonical(payload) + b"\n"


def test_manifest_round_trip_and_topic_selection_keep_manifest_order(
    tmp_path: Path,
) -> None:
    first = _literal_handoff().topics[0]
    second = replace(first, topic_id="rag2026-200", narrative="A second narrative.")
    handoff = GenerationHandoff(
        producer=_literal_handoff().producer,
        topics=(first, second),
    )
    path = tmp_path / "generation_handoff_manifest.json"

    write_generation_handoff(path, handoff)
    loaded = load_generation_handoff(path)

    assert loaded == handoff
    assert tuple(
        topic.topic_id
        for topic in select_generation_topics(
            loaded, ("rag2026-200", "rag2026-58")
        )
    ) == ("rag2026-58", "rag2026-200")
    assert select_generation_topics(loaded, None) == loaded.topics


@pytest.mark.parametrize(
    ("topic_ids", "message"),
    [
        ((), "at least one"),
        (("rag2026-58", "rag2026-58"), "duplicate"),
        (("missing",), "unknown"),
        (("",), "non-empty"),
    ],
)
def test_topic_selection_rejects_invalid_requests(
    topic_ids: tuple[str, ...], message: str
) -> None:
    with pytest.raises(HandoffIntegrityError, match=message):
        select_generation_topics(_literal_handoff(), topic_ids)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"generation_handoff_manifest_v1",'
        '"schema_version":"generation_handoff_manifest_v1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(HandoffIntegrityError, match="duplicate JSON key"):
        load_generation_handoff(path)


def test_loader_rejects_unknown_fields_even_when_outer_hash_is_valid(
    tmp_path: Path,
) -> None:
    payload = _literal_handoff().to_payload()
    payload["gold_nuggets"] = []
    path = tmp_path / "unknown.json"
    path.write_bytes(_reseal(payload))

    with pytest.raises(HandoffIntegrityError, match="root fields"):
        load_generation_handoff(path)


def test_loader_rejects_wrong_evidence_text_hash_even_when_resealed(
    tmp_path: Path,
) -> None:
    payload = _literal_handoff().to_payload()
    topics = payload["topics"]
    assert isinstance(topics, list) and isinstance(topics[0], dict)
    evidence = topics[0]["evidence"]
    assert isinstance(evidence, list) and isinstance(evidence[0], dict)
    evidence[0]["text_sha256"] = "f" * 64
    path = tmp_path / "wrong-text-hash.json"
    path.write_bytes(_reseal(payload))

    with pytest.raises(HandoffIntegrityError, match="evidence payload"):
        load_generation_handoff(path)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda payload: payload.update(topic_count=2), id="topic-count"),
        pytest.param(
            lambda payload: payload.update(schema_version="other"), id="schema"
        ),
        pytest.param(
            lambda payload: payload.update(manifest_sha256="0" * 64),
            id="manifest-hash",
        ),
    ],
)
def test_loader_rejects_wrong_root_identity(
    tmp_path: Path, mutate: object
) -> None:
    payload = _literal_handoff().to_payload()
    assert callable(mutate)
    mutate(payload)
    path = tmp_path / "wrong-root.json"
    path.write_bytes(_canonical(payload) + b"\n")

    with pytest.raises(HandoffIntegrityError):
        load_generation_handoff(path)


def test_loader_rejects_non_utf8_input(tmp_path: Path) -> None:
    path = tmp_path / "non-utf8.json"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(HandoffIntegrityError, match="UTF-8"):
        load_generation_handoff(path)


def test_writer_is_idempotent_but_never_replaces_conflicting_bytes(
    tmp_path: Path,
) -> None:
    handoff = _literal_handoff()
    path = tmp_path / "generation_handoff_manifest.json"
    write_generation_handoff(path, handoff)
    expected = path.read_bytes()

    write_generation_handoff(path, handoff)
    path.write_bytes(b"{}\n")

    with pytest.raises(HandoffIntegrityError, match="different bytes"):
        write_generation_handoff(path, handoff)
    assert path.read_bytes() == b"{}\n"
    assert expected == serialize_generation_handoff(handoff)


def test_topic_rejects_evidence_not_selected_exactly_once() -> None:
    topic = _literal_handoff().topics[0]

    with pytest.raises(HandoffIntegrityError, match="unknown evidence"):
        GenerationTopic(
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            groups=topic.groups,
            evidence=topic.evidence[:1],
            claim_hints=topic.claim_hints,
            source_receipts=topic.source_receipts,
        )


def test_topic_rejects_one_rank_for_two_documents() -> None:
    topic = _literal_handoff().topics[0]
    conflicting = replace(topic.evidence[1], document_rank=1)

    with pytest.raises(HandoffIntegrityError, match="identifies multiple docids"):
        GenerationTopic(
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            groups=topic.groups,
            evidence=(topic.evidence[0], conflicting),
            claim_hints=topic.claim_hints,
            source_receipts=topic.source_receipts,
        )


def test_fallback_group_must_copy_the_official_narrative() -> None:
    topic = _literal_handoff().topics[0]
    fallback = replace(topic.groups[0], kind="official_narrative_fallback")

    with pytest.raises(HandoffIntegrityError, match="copy the narrative"):
        GenerationTopic(
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            groups=(fallback,),
            evidence=topic.evidence,
            claim_hints=topic.claim_hints,
            source_receipts=topic.source_receipts,
        )


def test_evidence_source_geometry_must_match_exact_utf8_text() -> None:
    with pytest.raises(HandoffIntegrityError, match="byte span length"):
        EvidencePassage(
            evidence_id="unicode",
            group_id="group",
            cluster_id="cluster",
            cluster_ordinal=1,
            support_ordinal=1,
            candidate_kind="extractive",
            docid="doc",
            document_rank=1,
            text="Café",
            document_sha256="a" * 64,
            source_span=EvidenceSourceSpan(
                start_char=0,
                end_char=4,
                start_byte=0,
                end_byte=4,
            ),
        )


def test_topic_rejects_duplicate_group_evidence_and_claim_ids() -> None:
    topic = _literal_handoff().topics[0]

    with pytest.raises(HandoffIntegrityError, match="group IDs"):
        GenerationTopic(
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            groups=(topic.groups[0], topic.groups[0]),
            evidence=topic.evidence,
            claim_hints=topic.claim_hints,
            source_receipts=topic.source_receipts,
        )
    with pytest.raises(HandoffIntegrityError, match="evidence IDs"):
        GenerationTopic(
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            groups=topic.groups,
            evidence=(topic.evidence[0], replace(topic.evidence[1], evidence_id="ev-a")),
            claim_hints=topic.claim_hints,
            source_receipts=topic.source_receipts,
        )
    with pytest.raises(HandoffIntegrityError, match="claim IDs"):
        GenerationTopic(
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            groups=topic.groups,
            evidence=topic.evidence,
            claim_hints=(topic.claim_hints[0], topic.claim_hints[0]),
            source_receipts=topic.source_receipts,
        )


def test_topic_rejects_noncontiguous_cluster_and_support_ordinals() -> None:
    topic = _literal_handoff().topics[0]
    bad_cluster = replace(topic.groups[0].selected_clusters[0], ordinal=2)

    with pytest.raises(HandoffIntegrityError, match="cluster ordinals"):
        replace(topic.groups[0], selected_clusters=(bad_cluster,))
    with pytest.raises(HandoffIntegrityError, match="provenance differs"):
        GenerationTopic(
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            groups=topic.groups,
            evidence=(topic.evidence[0], replace(topic.evidence[1], support_ordinal=3)),
            claim_hints=topic.claim_hints,
            source_receipts=topic.source_receipts,
        )


def test_topic_rejects_claim_evidence_outside_its_group() -> None:
    topic = _literal_handoff().topics[0]
    outside = replace(topic.claim_hints[0], group_id="other-group")

    with pytest.raises(HandoffIntegrityError, match="unknown group"):
        GenerationTopic(
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            groups=topic.groups,
            evidence=topic.evidence,
            claim_hints=(outside,),
            source_receipts=topic.source_receipts,
        )


def test_topic_rejects_conflicting_hash_for_one_docid() -> None:
    topic = _literal_handoff().topics[0]
    conflicting = replace(
        topic.evidence[1],
        docid="doc-a",
        document_rank=1,
    )

    with pytest.raises(HandoffIntegrityError, match="conflicting rank or hash"):
        GenerationTopic(
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            groups=topic.groups,
            evidence=(topic.evidence[0], conflicting),
            claim_hints=topic.claim_hints,
            source_receipts=topic.source_receipts,
        )


def test_topic_requires_nonempty_selected_evidence() -> None:
    topic = _literal_handoff().topics[0]

    with pytest.raises(HandoffIntegrityError, match="non-empty"):
        GenerationTopic(
            topic_id=topic.topic_id,
            narrative=topic.narrative,
            groups=topic.groups,
            evidence=(),
            claim_hints=(),
            source_receipts=topic.source_receipts,
        )


def test_loader_rejects_nonfinite_and_trailing_content(tmp_path: Path) -> None:
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"topic_count":NaN}\n', encoding="utf-8")
    trailing = tmp_path / "trailing.json"
    trailing.write_bytes(serialize_generation_handoff(_literal_handoff()) + b"x")

    with pytest.raises(HandoffIntegrityError, match="non-finite"):
        load_generation_handoff(nonfinite)
    with pytest.raises(HandoffIntegrityError, match="JSON"):
        load_generation_handoff(trailing)


def test_loader_rejects_wrong_narrative_hash_with_valid_outer_seals(
    tmp_path: Path,
) -> None:
    payload = _literal_handoff().to_payload()
    topics = payload["topics"]
    assert isinstance(topics, list) and isinstance(topics[0], dict)
    topics[0]["narrative_sha256"] = "e" * 64
    path = tmp_path / "wrong-narrative-hash.json"
    path.write_bytes(_reseal(payload))

    with pytest.raises(HandoffIntegrityError, match="topic payload"):
        load_generation_handoff(path)


def test_loader_rejects_noncanonical_but_semantically_equal_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pretty.json"
    path.write_text(
        json.dumps(_literal_handoff().to_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HandoffIntegrityError, match="canonical"):
        load_generation_handoff(path)


def test_renderer_exposes_selected_passages_and_not_full_document_state() -> None:
    topic = _literal_handoff().topics[0]
    full_document_not_selected = "FULL_DOCUMENT_SENTINEL"

    rendered = render_generation_evidence(topic)

    for expected in (
        "Explain the first answer aspect.",
        "group-1",
        "The first answer aspect.",
        "cluster-1",
        "ev-a",
        "ev-b",
        "doc-a",
        "doc-b",
        "document_rank=1",
        "document_rank=2",
        "Selected alpha passage.",
        "Selected beta passage.",
        "ADVISORY CLAIM HINTS",
        "claim-1",
        "Alpha is selected.",
        "Selected evidence passages are factual authority.",
    ):
        assert expected in rendered
    assert full_document_not_selected not in rendered
    assert topic.evidence[0].document_sha256 not in rendered
    assert "start_char" not in rendered
    assert "text_sha256" not in rendered


def test_prompt_hash_authenticates_exact_rendered_utf8() -> None:
    topic = _literal_handoff().topics[0]
    expected = sha256(render_generation_evidence(topic).encode("utf-8")).hexdigest()

    assert prompt_sha256(topic) == expected
