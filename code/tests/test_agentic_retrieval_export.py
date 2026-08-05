from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import stat
import zipfile

import pytest

from trec_rag import agentic_retrieval_export
from trec_rag.agentic_generation_export import (
    AgenticFullTextCandidate,
    AgenticRetrievalRow,
    AgenticTopicProjection,
    serialize_agentic_retrieval_topic,
)
from trec_rag.agentic_retrieval_export import (
    EXPORT_MANIFEST_FILENAME,
    GENERATION_HANDOFF_FILENAME,
    RETRIEVAL_RUN_FILENAME,
    RETRIEVAL_WITH_TEXT_FILENAME,
    AgenticRetrievalExportError,
    load_agentic_retrieval_export,
    publish_agentic_retrieval_export,
)
from trec_rag.agentic_run_state import (
    SubmoduleRevision,
    allocate_topic_attempt,
    create_run_plan,
    seal_topic_success,
)
from trec_rag.generation_handoff import (
    ClaimHint,
    EvidenceGroup,
    EvidencePassage,
    EvidenceSourceSpan,
    GenerationTopic,
    SelectedCluster,
    TopicSourceReceipts,
    load_generation_handoff,
)
from trec_rag.topics import Topic


RUN_ID = "rag26_agentic_export_test"
OFFICIAL_TOPICS_SHA256 = "a" * 64
TOPICS = (
    Topic("rag2026-0", "", "Explain café causes and evidence."),
    Topic("rag2026-1", "", "Explain Ω consequences and evidence."),
)
ROOT_ARTIFACTS = (
    RETRIEVAL_RUN_FILENAME,
    RETRIEVAL_WITH_TEXT_FILENAME,
    GENERATION_HANDOFF_FILENAME,
    EXPORT_MANIFEST_FILENAME,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _generation_topic(
    *,
    topic: Topic,
    first_docid: str,
    first_text: str,
    retrieval_topic_sha256: str,
) -> GenerationTopic:
    group_id = f"group-{topic.id}"
    cluster_id = f"cluster-{topic.id}"
    evidence_id = f"evidence-{topic.id}"
    return GenerationTopic(
        topic_id=topic.id,
        narrative=topic.narrative,
        groups=(
            EvidenceGroup(
                group_id=group_id,
                kind="generated_subnarrative",
                text=f"Grounded need for {topic.id}",
                selected_clusters=(
                    SelectedCluster(
                        cluster_id=cluster_id,
                        ordinal=1,
                        representative_evidence_id=evidence_id,
                        evidence_ids=(evidence_id,),
                    ),
                ),
            ),
        ),
        evidence=(
            EvidencePassage(
                evidence_id=evidence_id,
                group_id=group_id,
                cluster_id=cluster_id,
                cluster_ordinal=1,
                support_ordinal=1,
                candidate_kind="agentic_grounded_nugget",
                docid=first_docid,
                document_rank=1,
                text=first_text,
                document_sha256=_digest(first_text),
                source_span=EvidenceSourceSpan(
                    start_char=0,
                    end_char=len(first_text),
                    start_byte=0,
                    end_byte=len(first_text.encode("utf-8")),
                ),
            ),
        ),
        claim_hints=(
            ClaimHint(
                claim_id=f"claim-{topic.id}",
                group_id=group_id,
                kind="agentic_grounded_nugget",
                text=f"Grounded claim for {topic.id}.",
                evidence_ids=(evidence_id,),
            ),
        ),
        source_receipts=TopicSourceReceipts(
            official_topics_sha256=OFFICIAL_TOPICS_SHA256,
            retrieval_topic_sha256=retrieval_topic_sha256,
        ),
    )


def _projection(topic: Topic, document_count: int) -> AgenticTopicProjection:
    texts = tuple(
        f"Document {rank} for {topic.id}: café α Ω evidence."
        for rank in range(1, document_count + 1)
    )
    docids = tuple(
        f"doc-{topic.id}-{rank}" for rank in range(1, document_count + 1)
    )
    rows = tuple(
        AgenticRetrievalRow(
            topic_id=topic.id,
            docid=docid,
            rank=rank,
            score=document_count - rank + 1,
        )
        for rank, docid in enumerate(docids, start=1)
    )
    candidates = tuple(
        AgenticFullTextCandidate(
            docid=docid,
            text=text,
            rank=rank,
            score=document_count - rank + 1,
            document_sha256=_digest(text),
        )
        for rank, (docid, text) in enumerate(zip(docids, texts, strict=True), start=1)
    )
    preliminary_generation = _generation_topic(
        topic=topic,
        first_docid=docids[0],
        first_text=texts[0],
        retrieval_topic_sha256="0" * 64,
    )
    preliminary = AgenticTopicProjection(
        topic_id=topic.id,
        narrative=topic.narrative,
        retrieval_rows=rows,
        full_text_candidates=candidates,
        generation_topic=preliminary_generation,
        retrieval_topic_sha256="0" * 64,
    )
    retrieval_digest = sha256(
        serialize_agentic_retrieval_topic(preliminary)
    ).hexdigest()
    return AgenticTopicProjection(
        topic_id=topic.id,
        narrative=topic.narrative,
        retrieval_rows=rows,
        full_text_candidates=candidates,
        generation_topic=_generation_topic(
            topic=topic,
            first_docid=docids[0],
            first_text=texts[0],
            retrieval_topic_sha256=retrieval_digest,
        ),
        retrieval_topic_sha256=retrieval_digest,
    )


def _plan(output_dir: Path):
    work_dir = output_dir / "work"
    return work_dir, create_run_plan(
        work_dir=work_dir,
        run_id=RUN_ID,
        config_bytes=b"schema_version: agentic_retrieval_config_v1\n",
        topics=TOPICS,
        official_topics_sha256=OFFICIAL_TOPICS_SHA256,
        source_revision="1" * 40,
        submodule_revisions=(
            SubmoduleRevision("ragdoll", "2" * 40),
            SubmoduleRevision("trec-rag-skills", "3" * 40),
        ),
    )


def _seal(
    *,
    output_dir: Path,
    work_dir: Path,
    plan,
    topic: Topic,
    document_count: int,
    stopping_reason: str,
    synthesis_outcome: str,
):
    projection = _projection(topic, document_count)
    attempt = allocate_topic_attempt(
        work_dir=work_dir, plan=plan, topic_id=topic.id
    )
    return seal_topic_success(
        work_dir=work_dir,
        plan=plan,
        projection=projection,
        attempt=attempt,
        status="complete",
        stopping_reason=stopping_reason,
        synthesis_outcome=synthesis_outcome,
        records_receipt={
            "topic_id": topic.id,
            "run_id": plan.run_id,
            "database_sha256": _digest(f"database-{topic.id}"),
            "semantic_sha256": _digest(f"semantic-{topic.id}"),
            "row_counts": {"documents": document_count},
        },
    )


def _complete_run(output_dir: Path):
    work_dir, plan = _plan(output_dir)
    # Seal in reverse order: aggregation must still use the original plan order.
    second = _seal(
        output_dir=output_dir,
        work_dir=work_dir,
        plan=plan,
        topic=TOPICS[1],
        document_count=1,
        stopping_reason="budget_exhausted",
        synthesis_outcome="deterministic_grounded_recovery",
    )
    first = _seal(
        output_dir=output_dir,
        work_dir=work_dir,
        plan=plan,
        topic=TOPICS[0],
        document_count=2,
        stopping_reason="coverage_sufficient",
        synthesis_outcome="coordinator_selected",
    )
    return work_dir, plan, (first, second)


def test_export_is_deterministic_ordered_and_rag_handoff_compatible(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    work_dir, plan, seals = _complete_run(output_dir)

    first = publish_agentic_retrieval_export(
        output_dir=output_dir,
        work_dir=work_dir,
        plan=plan,
        producer_revision="f" * 40,
    )
    original_bytes = {
        name: (output_dir / name).read_bytes() for name in ROOT_ARTIFACTS
    }
    second = publish_agentic_retrieval_export(
        output_dir=output_dir,
        work_dir=work_dir,
        plan=plan,
        producer_revision="f" * 40,
    )

    assert first == second == load_agentic_retrieval_export(
        output_dir=output_dir, work_dir=work_dir, plan=plan
    )
    assert {
        name: (output_dir / name).read_bytes() for name in ROOT_ARTIFACTS
    } == original_bytes
    assert all(
        stat.S_IMODE((output_dir / name).stat().st_mode) == 0o600
        for name in ROOT_ARTIFACTS
    )

    lines = (output_dir / RETRIEVAL_RUN_FILENAME).read_text(
        encoding="utf-8"
    ).splitlines()
    assert lines == [
        "rag2026-0 Q0 doc-rag2026-0-1 1 2 rag26_agentic_export_test",
        "rag2026-0 Q0 doc-rag2026-0-2 2 1 rag26_agentic_export_test",
        "rag2026-1 Q0 doc-rag2026-1-1 1 1 rag26_agentic_export_test",
    ]
    assert len({tuple(line.split()[::2]) for line in lines}) == len(lines)

    with zipfile.ZipFile(output_dir / RETRIEVAL_WITH_TEXT_FILENAME) as archive:
        assert archive.namelist() == ["retrieval_with_text.jsonl"]
        info = archive.getinfo("retrieval_with_text.jsonl")
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.external_attr >> 16 == 0o100600
        rows = [
            json.loads(line)
            for line in archive.read(info).decode("utf-8").splitlines()
        ]
    assert [row["query"]["qid"] for row in rows] == [
        "rag2026-0",
        "rag2026-1",
    ]
    assert [[candidate["score"] for candidate in row["candidates"]] for row in rows] == [
        [2, 1],
        [1],
    ]

    handoff = load_generation_handoff(
        output_dir / GENERATION_HANDOFF_FILENAME
    )
    assert [topic.topic_id for topic in handoff.topics] == [
        "rag2026-0",
        "rag2026-1",
    ]
    trec_by_topic = {
        topic.id: {
            fields[2]
            for line in lines
            if (fields := line.split())[0] == topic.id
        }
        for topic in TOPICS
    }
    zip_by_topic = {
        row["query"]["qid"]: {
            candidate["docid"] for candidate in row["candidates"]
        }
        for row in rows
    }
    for topic in handoff.topics:
        assert (
            set(topic.citation_docids)
            <= trec_by_topic[topic.topic_id]
            == zip_by_topic[topic.topic_id]
        )

    manifest_body = (output_dir / EXPORT_MANIFEST_FILENAME).read_bytes()
    manifest = json.loads(manifest_body)
    assert manifest["schema_version"] == "agentic_retrieval_export_manifest_v1"
    assert manifest["run_id"] == RUN_ID
    assert manifest["run_plan_sha256"] == plan.plan_sha256
    assert manifest["planned_topic_ids"] == [topic.id for topic in TOPICS]
    assert [
        (row["topic_id"], row["status"], row["stopping_reason"], row["synthesis_outcome"])
        for row in manifest["topics"]
    ] == [
        (
            "rag2026-0",
            "complete",
            "coverage_sufficient",
            "coordinator_selected",
        ),
        (
            "rag2026-1",
            "complete",
            "budget_exhausted",
            "deterministic_grounded_recovery",
        ),
    ]
    assert [row["topic_seal_sha256"] for row in manifest["topics"]] == [
        seal.seal_sha256 for seal in seals
    ]
    artifacts = {row["name"]: row for row in manifest["artifacts"]}
    assert set(artifacts) == set(ROOT_ARTIFACTS[:-1])
    for name, row in artifacts.items():
        body = (output_dir / name).read_bytes()
        assert row["bytes"] == len(body)
        assert row["sha256"] == sha256(body).hexdigest()
    without_digest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    assert manifest["manifest_sha256"] == sha256(
        _canonical(without_digest)
    ).hexdigest()
    assert manifest_body == _canonical(manifest) + b"\n"


def test_export_refuses_an_incomplete_run_before_writing_root_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    work_dir, plan = _plan(output_dir)
    _seal(
        output_dir=output_dir,
        work_dir=work_dir,
        plan=plan,
        topic=TOPICS[0],
        document_count=1,
        stopping_reason="coverage_sufficient",
        synthesis_outcome="coordinator_selected",
    )

    with pytest.raises(AgenticRetrievalExportError, match="not sealed"):
        publish_agentic_retrieval_export(
            output_dir=output_dir,
            work_dir=work_dir,
            plan=plan,
            producer_revision="f" * 40,
        )

    assert all(not (output_dir / name).exists() for name in ROOT_ARTIFACTS)


def test_export_refuses_corrupt_topic_state_before_writing_root_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    work_dir, plan, _seals = _complete_run(output_dir)
    payload = work_dir / "topics" / "rag2026-0" / "retrieval_topic.json"
    payload.write_bytes(payload.read_bytes() + b" ")

    with pytest.raises(AgenticRetrievalExportError, match="artifact"):
        publish_agentic_retrieval_export(
            output_dir=output_dir,
            work_dir=work_dir,
            plan=plan,
            producer_revision="f" * 40,
        )

    assert all(not (output_dir / name).exists() for name in ROOT_ARTIFACTS)


def test_export_never_overwrites_a_conflicting_root_payload(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    work_dir, plan, _seals = _complete_run(output_dir)
    conflict = b"user-owned conflicting bytes\n"
    (output_dir / RETRIEVAL_RUN_FILENAME).write_bytes(conflict)

    with pytest.raises(AgenticRetrievalExportError, match="conflict"):
        publish_agentic_retrieval_export(
            output_dir=output_dir,
            work_dir=work_dir,
            plan=plan,
            producer_revision="f" * 40,
        )

    assert (output_dir / RETRIEVAL_RUN_FILENAME).read_bytes() == conflict
    assert not (output_dir / RETRIEVAL_WITH_TEXT_FILENAME).exists()
    assert not (output_dir / GENERATION_HANDOFF_FILENAME).exists()
    assert not (output_dir / EXPORT_MANIFEST_FILENAME).exists()


def test_export_resumes_identical_payloads_after_a_pre_manifest_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"
    work_dir, plan, _seals = _complete_run(output_dir)

    def fail_after_generation(name: str) -> None:
        if name == GENERATION_HANDOFF_FILENAME:
            raise RuntimeError("simulated crash before outer receipt")

    monkeypatch.setattr(
        agentic_retrieval_export,
        "_PUBLICATION_TEST_HOOK",
        fail_after_generation,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        publish_agentic_retrieval_export(
            output_dir=output_dir,
            work_dir=work_dir,
            plan=plan,
            producer_revision="f" * 40,
        )
    assert all(
        (output_dir / name).exists() for name in ROOT_ARTIFACTS[:-1]
    )
    assert not (output_dir / EXPORT_MANIFEST_FILENAME).exists()
    payload_bytes = {
        name: (output_dir / name).read_bytes() for name in ROOT_ARTIFACTS[:-1]
    }

    monkeypatch.setattr(
        agentic_retrieval_export, "_PUBLICATION_TEST_HOOK", None
    )
    receipt = publish_agentic_retrieval_export(
        output_dir=output_dir,
        work_dir=work_dir,
        plan=plan,
        producer_revision="f" * 40,
    )

    assert receipt.manifest == output_dir / EXPORT_MANIFEST_FILENAME
    assert {
        name: (output_dir / name).read_bytes() for name in ROOT_ARTIFACTS[:-1]
    } == payload_bytes
    assert (output_dir / EXPORT_MANIFEST_FILENAME).exists()
