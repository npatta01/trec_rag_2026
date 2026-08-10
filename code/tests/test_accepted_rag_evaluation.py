"""Tests for immutable accepted-RAG provenance bindings."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from trec_rag.accepted_rag_evaluation import (
    build_accepted_run_binding,
    write_accepted_run_binding,
)
import trec_rag.ragdoll_io as ragdoll_io
from trec_rag.generation_handoff import (
    SOURCE_CONTRACT,
    EvidenceGroup,
    EvidencePassage,
    EvidenceSourceSpan,
    GenerationHandoff,
    GenerationTopic,
    HandoffProducer,
    SelectedCluster,
    TopicSourceReceipts,
    write_generation_handoff,
)


class AcceptedPaths(NamedTuple):
    submission: Path
    bundle_metadata: Path
    handoff: Path
    topic_context_sha256s: dict[str, str]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_handoff(path: Path, topic_ids: tuple[str, ...] = ("topic-a", "topic-b")) -> Path:
    topics: list[GenerationTopic] = []
    for topic_id in topic_ids:
        text = f"Selected evidence for {topic_id}."
        evidence_id = f"{topic_id}-evidence"
        cluster_id = f"{topic_id}-cluster"
        topics.append(
            GenerationTopic(
                topic_id=topic_id,
                narrative=f"Narrative for {topic_id}.",
                groups=(
                    EvidenceGroup(
                        group_id=f"{topic_id}-group",
                        kind="generated_subnarrative",
                        text=text,
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
                        group_id=f"{topic_id}-group",
                        cluster_id=cluster_id,
                        cluster_ordinal=1,
                        support_ordinal=1,
                        candidate_kind="extractive",
                        docid=f"doc-{topic_id}",
                        document_rank=1,
                        text=text,
                        document_sha256=sha256(text.encode()).hexdigest(),
                        source_span=EvidenceSourceSpan(
                            start_char=0,
                            end_char=len(text),
                            start_byte=0,
                            end_byte=len(text.encode()),
                        ),
                    ),
                ),
                claim_hints=(),
                source_receipts=TopicSourceReceipts(
                    official_topics_sha256="1" * 64,
                    retrieval_topic_sha256=sha256(topic_id.encode()).hexdigest(),
                ),
            )
        )
    write_generation_handoff(
        path,
        GenerationHandoff(
            producer=HandoffProducer(
                source_contract=SOURCE_CONTRACT,
                retrieval_run_id="retrieval-fixture",
                producer_revision="test-revision",
            ),
            topics=tuple(topics),
        ),
    )
    return path


def build_accepted_fixture(
    root: Path,
    *,
    run_id: str = "accepted-single",
    mutation: str | None = None,
    source_identity: dict[str, Any] | None = None,
) -> AcceptedPaths:
    handoff = _write_handoff(root / "handoff.json")
    handoff_payload = json.loads(handoff.read_text(encoding="utf-8"))
    topic_context_sha256s = {
        topic["topic_id"]: topic["context_sha256"]
        for topic in handoff_payload["topics"]
    }
    rows = [
        {
            "metadata": {
                "team_id": "test-team",
                "narrative_id": topic_id,
                "narrative": f"Narrative for {topic_id}.",
                "run_id": run_id,
                "run_desc": "Accepted fixture run",
            },
            "references": [f"doc-{topic_id}"],
            "answer": [{"text": f"Answer for {topic_id}.", "citations": [0]}],
        }
        for topic_id in topic_context_sha256s
    ]
    submission = _write_jsonl(root / "accepted.jsonl", rows)
    submission_hash = sha256(submission.read_bytes()).hexdigest()
    if mutation == "submission_hash":
        submission_hash = "0" * 64
    handoff_hash = handoff_payload["manifest_sha256"]
    if mutation == "handoff_hash":
        handoff_hash = "0" * 64
    metadata_run_id = run_id if mutation != "run_id" else "wrong-run"
    bundle_metadata = root / "metadata.json"
    bundle_metadata.write_text(
        json.dumps(
            {
                "schema_version": "trec-rag-2026-rag-submission-bundle-v1",
                "team_id": "test-team",
                "provider": "test-provider",
                "source": {
                    "generation_handoff_manifest_sha256": handoff_hash,
                    "topic_count": len(rows),
                },
                "runs": [
                    {
                        "name": "fixture",
                        "run_id": metadata_run_id,
                        "run_desc": "Accepted fixture run",
                        "path": "fixture/accepted.jsonl",
                        "bytes": len(submission.read_bytes()),
                        "line_count": len(rows),
                        "sha256": submission_hash,
                        "provider": "test-provider",
                        "models": ["test/model"],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if source_identity is not None:
        source_path = root / "multistage_generation_identity.json"
        source_path.write_text(json.dumps(source_identity, sort_keys=True), encoding="utf-8")
    return AcceptedPaths(submission, bundle_metadata, handoff, topic_context_sha256s)


def test_accepted_binding_records_missing_source_identity(tmp_path: Path) -> None:
    paths = build_accepted_fixture(tmp_path)

    binding = build_accepted_run_binding(paths.submission, paths.bundle_metadata, paths.handoff)

    assert binding.schema_version == "accepted_rag_evaluation_binding_v1"
    assert binding.run_id == "accepted-single"
    assert binding.source_identity_available is False
    assert binding.source_identity_reason == "original generation identity was not preserved"
    assert binding.topic_ids == ("topic-a", "topic-b")
    assert binding.topic_context_sha256s == paths.topic_context_sha256s


@pytest.mark.parametrize("mutation", ["submission_hash", "run_id", "handoff_hash"])
def test_accepted_binding_rejects_receipt_mismatch(tmp_path: Path, mutation: str) -> None:
    paths = build_accepted_fixture(tmp_path, mutation=mutation)

    with pytest.raises(ValueError, match="accepted.*does not match"):
        build_accepted_run_binding(paths.submission, paths.bundle_metadata, paths.handoff)


def test_accepted_binding_rejects_mismatched_preserved_multistage_identity(
    tmp_path: Path,
) -> None:
    paths = build_accepted_fixture(tmp_path)
    handoff = json.loads(paths.handoff.read_text(encoding="utf-8"))
    identity = {
        "identity_version": 1,
        "trial_contract_version": "bounded_narrative_revision_trial_v8_screen_liveness",
        "handoff_schema_version": handoff["schema_version"],
        "handoff_manifest_sha256": handoff["manifest_sha256"],
        "submission_run_id": "accepted-single",
        "topics": [
            {
                "topic_id": topic_id,
                "context_sha256": "0" * 64 if topic_id == "topic-a" else digest,
            }
            for topic_id, digest in paths.topic_context_sha256s.items()
        ],
    }
    identity_path = tmp_path / "multistage_generation_identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="accepted.*does not match"):
        build_accepted_run_binding(
            paths.submission,
            paths.bundle_metadata,
            paths.handoff,
            source_identity_path=identity_path,
        )


def test_accepted_binding_rejects_missing_or_changed_multistage_contract_fields(
    tmp_path: Path,
) -> None:
    paths = build_accepted_fixture(tmp_path)
    handoff = json.loads(paths.handoff.read_text(encoding="utf-8"))
    identity = {
        "identity_version": 1,
        "trial_contract_version": "bounded_narrative_revision_trial_v8_screen_liveness",
        "handoff_schema_version": handoff["schema_version"],
        "handoff_manifest_sha256": handoff["manifest_sha256"],
        "submission_run_id": "accepted-single",
        "topics": [
            {"topic_id": topic_id, "context_sha256": digest}
            for topic_id, digest in paths.topic_context_sha256s.items()
        ],
    }
    for field, value in (
        ("identity_version", None),
        ("identity_version", 2),
        ("trial_contract_version", None),
        ("trial_contract_version", "wrong-contract"),
    ):
        candidate = dict(identity)
        if value is None:
            candidate.pop(field)
        else:
            candidate[field] = value
        source_path = tmp_path / f"identity-{field}-{value}.json"
        source_path.write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises(ValueError, match="source identity"):
            build_accepted_run_binding(
                paths.submission,
                paths.bundle_metadata,
                paths.handoff,
                source_identity_path=source_path,
            )


def test_selected_support_rejects_submission_answer_mutation_after_binding(
    tmp_path: Path,
) -> None:
    paths = build_accepted_fixture(tmp_path)
    binding = build_accepted_run_binding(paths.submission, paths.bundle_metadata, paths.handoff)
    binding_path = write_accepted_run_binding(binding, tmp_path / "binding.json")
    records = [json.loads(line) for line in paths.submission.read_text().splitlines()]
    records[0]["answer"][0]["text"] = "Mutated after binding."
    mutated = _write_jsonl(tmp_path / "mutated.jsonl", records)

    with pytest.raises(ValueError, match="submission.*(?:sha256|bytes)"):
        ragdoll_io.selected_evidence_support_rows(mutated, paths.handoff, binding_path)


def test_selected_support_rejects_submission_topic_reorder_after_binding(
    tmp_path: Path,
) -> None:
    paths = build_accepted_fixture(tmp_path)
    binding = build_accepted_run_binding(paths.submission, paths.bundle_metadata, paths.handoff)
    binding_path = write_accepted_run_binding(binding, tmp_path / "binding.json")
    records = [json.loads(line) for line in paths.submission.read_text().splitlines()]
    reordered = _write_jsonl(tmp_path / "reordered.jsonl", list(reversed(records)))

    with pytest.raises(ValueError, match="topic order|sha256|bytes"):
        ragdoll_io.selected_evidence_support_rows(reordered, paths.handoff, binding_path)


def test_write_accepted_binding_is_canonical_and_private(tmp_path: Path) -> None:
    paths = build_accepted_fixture(tmp_path)
    binding = build_accepted_run_binding(paths.submission, paths.bundle_metadata, paths.handoff)
    output = tmp_path / "private" / "binding.json"

    assert write_accepted_run_binding(binding, output) == output
    assert output.stat().st_mode & 0o777 == 0o600
    assert output.parent.stat().st_mode & 0o777 == 0o700
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert list(json.loads(output.read_text(encoding="utf-8"))) == sorted(
        json.loads(output.read_text(encoding="utf-8"))
    )
