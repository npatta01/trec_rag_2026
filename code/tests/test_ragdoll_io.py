"""Tests for the RAGDoll adapters and the archived development-topic input builder."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

import trec_rag.competition_rag as competition_rag
import trec_rag.dev_rag_inputs as dev_rag_inputs
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

NARRATIVE = "I want to understand nuclear energy tradeoffs."


def _submission_record(qid: str = "58") -> dict[str, Any]:
    return {
        "metadata": {
            "team_id": "local-baseline",
            "narrative_id": qid,
            "narrative": NARRATIVE,
            "run_id": "dev-spike",
            "run_desc": "development spike",
        },
        "references": ["climbmix-a", "climbmix-b"],
        "answer": [
            {"text": "Nuclear power emits little carbon.", "citations": [0]},
            {"text": "Waste storage remains unresolved.", "citations": [1]},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows), encoding="utf-8"
    )
    return path


def _gold_record(qid: str = "58", count: int = 3) -> dict[str, Any]:
    return {
        "qid": qid,
        "nuggets": [
            {
                "text": f"gold nugget {index}",
                # Released values really do carry literal embedded quotes.
                "mapped_sub_narrative": '"Key advantages of nuclear energy"',
                "importance": "vital" if index % 2 == 0 else "okay",
                "source": "post-edit",
            }
            for index in range(count)
        ],
    }


def test_answer_rows_expose_qid_that_ragdoll_can_resolve(tmp_path: Path) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])

    rows = ragdoll_io.answer_rows(submission, narratives={"58": NARRATIVE})

    assert len(rows) == 1
    payload = rows[0].as_dict()
    # RAGDoll reads qid/topic_id/query_id only; metadata.narrative_id is invisible to it.
    assert payload["qid"] == "58"
    assert payload["topic_id"] == "58"
    assert payload["answer_text"] == (
        "Nuclear power emits little carbon. Waste storage remains unresolved."
    )
    assert payload["response_length"] == 9


def test_derived_answers_never_mutate_the_submission_record(tmp_path: Path) -> None:
    record = _submission_record()
    submission = _write_jsonl(tmp_path / "rag.jsonl", [record])
    before = submission.read_bytes()

    ragdoll_io.answer_rows(submission, narratives={"58": NARRATIVE})

    assert submission.read_bytes() == before
    # The organizer contract allows exactly these three root keys.
    reloaded = json.loads(before.decode("utf-8").splitlines()[0])
    assert set(reloaded) == {"metadata", "references", "answer"}
    competition_rag.validate_submission_record(
        reloaded,
        topic_id="58",
        narrative=NARRATIVE,
        allowed_docids=["climbmix-a", "climbmix-b"],
        team_id="local-baseline",
        run_id="dev-spike",
        run_desc="development spike",
    )


def test_answer_rows_reject_submission_narrative_that_differs_from_topics(
    tmp_path: Path,
) -> None:
    record = _submission_record()
    record["metadata"]["narrative"] = "Forged question text."
    submission = _write_jsonl(tmp_path / "rag.jsonl", [record])

    with pytest.raises(ValueError, match="authoritative topic narrative"):
        ragdoll_io.answer_rows(submission, narratives={"58": NARRATIVE})


def test_answer_rows_reject_submission_without_metadata_narrative(
    tmp_path: Path,
) -> None:
    record = _submission_record()
    del record["metadata"]["narrative"]
    submission = _write_jsonl(tmp_path / "rag.jsonl", [record])

    with pytest.raises(ValueError, match="missing metadata.narrative"):
        ragdoll_io.answer_rows(submission, narratives={"58": NARRATIVE})


def test_cli_loads_authoritative_topics_for_answer_queries(tmp_path: Path) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])
    topics = tmp_path / "topics.tsv"
    topics.write_text(f"58\t{NARRATIVE}\n", encoding="utf-8")
    gold = _write_jsonl(tmp_path / "gold.jsonl", [_gold_record()])
    answers_out = tmp_path / "answers.jsonl"
    nuggets_out = tmp_path / "nuggets.jsonl"

    assert ragdoll_io.main(
        [
            "--submission",
            str(submission),
            "--topics",
            str(topics),
            "--gold-nuggets",
            str(gold),
            "--answers-out",
            str(answers_out),
            "--nuggets-out",
            str(nuggets_out),
        ]
    ) == 0

    assert json.loads(answers_out.read_text(encoding="utf-8"))["query"] == NARRATIVE


def _ragdoll_cli_args(tmp_path: Path) -> tuple[list[str], Path]:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])
    topics = tmp_path / "topics.tsv"
    topics.write_text(f"58\t{NARRATIVE}\n", encoding="utf-8")
    gold = _write_jsonl(tmp_path / "gold.jsonl", [_gold_record()])
    support_out = tmp_path / "support.jsonl"
    return (
        [
            "--submission",
            str(submission),
            "--topics",
            str(topics),
            "--gold-nuggets",
            str(gold),
            "--answers-out",
            str(tmp_path / "answers.jsonl"),
            "--nuggets-out",
            str(tmp_path / "nuggets.jsonl"),
            "--support-out",
            str(support_out),
        ],
        support_out,
    )


def test_cli_support_can_resolve_the_selected_evidence_handoff(tmp_path: Path) -> None:
    args, support_out = _ragdoll_cli_args(tmp_path)
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {
            "58": [
                ("climbmix-a", "Generator-visible passage A."),
                ("climbmix-b", "Generator-visible passage B."),
            ]
        },
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json", handoff
    )

    assert ragdoll_io.main(
        [
            *args,
            "--handoff-manifest",
            str(handoff),
            "--generation-identity",
            str(identity),
        ]
    ) == 0

    row = json.loads(support_out.read_text(encoding="utf-8"))
    assert row["segments"] == {
        "climbmix-a": "Generator-visible passage A.",
        "climbmix-b": "Generator-visible passage B.",
    }


def test_cli_can_gate_completed_support_judgments(tmp_path: Path) -> None:
    args, _ = _ragdoll_cli_args(tmp_path)
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {
            "58": [
                ("climbmix-a", "Generator-visible passage A."),
                ("climbmix-b", "Generator-visible passage B."),
            ]
        },
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json", handoff
    )
    judgments = _write_jsonl(
        tmp_path / "support_judgments.jsonl",
        [
            _support_judgment(
                task_id="dev-spike:58:s0:c0",
                statement="Nuclear power emits little carbon.",
                citation="Generator-visible passage A.",
                sentence_index=0,
                citation_index=0,
                docid="climbmix-a",
            ),
            _support_judgment(
                task_id="dev-spike:58:s1:c0",
                statement="Waste storage remains unresolved.",
                citation="Generator-visible passage B.",
                sentence_index=1,
                citation_index=0,
                docid="climbmix-b",
            ),
        ],
    )

    assert ragdoll_io.main(
        [
            *args,
            "--handoff-manifest",
            str(handoff),
            "--generation-identity",
            str(identity),
            "--support-judgments",
            str(judgments),
        ]
    ) == 0


def test_cli_rejects_handoff_support_without_generation_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, _ = _ragdoll_cli_args(tmp_path)
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {"58": [("climbmix-a", "Selected A."), ("climbmix-b", "Selected B.")]},
    )

    with pytest.raises(SystemExit, match="2"):
        ragdoll_io.main(
            [
                *args,
                "--handoff-manifest",
                str(handoff),
            ]
        )

    assert "must be provided together" in capsys.readouterr().err


def test_cli_has_no_retrieval_document_support_route(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, _ = _ragdoll_cli_args(tmp_path)

    with pytest.raises(SystemExit, match="2"):
        ragdoll_io.main(
            [
                *args,
                "--documents",
                str(tmp_path / "retrieval_with_text.jsonl.zip"),
            ]
        )

    assert "unrecognized arguments: --documents" in capsys.readouterr().err


def test_gold_nuggets_are_reshaped_without_losing_importance(tmp_path: Path) -> None:
    nuggets = _write_jsonl(tmp_path / "gold.jsonl", [_gold_record(count=4)])

    rows = ragdoll_io.gold_nugget_rows(nuggets, narratives={"58": NARRATIVE})

    assert len(rows) == 1
    reshaped = rows[0]["nuggets"]
    assert len(reshaped) == 4
    assert [nugget["importance"] for nugget in reshaped] == ["vital", "okay", "vital", "okay"]
    # mapped_sub_narrative and source are intentionally dropped.
    assert all(set(nugget) == {"text", "importance"} for nugget in reshaped)
    assert rows[0]["query"] == NARRATIVE


def test_assert_join_rejects_a_silently_empty_join(tmp_path: Path) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record(qid="58")])
    nuggets = _write_jsonl(tmp_path / "gold.jsonl", [_gold_record(qid="999")])

    answers = ragdoll_io.answer_rows(submission, narratives={"58": NARRATIVE})
    gold = ragdoll_io.gold_nugget_rows(nuggets, narratives={"999": NARRATIVE})

    with pytest.raises(ValueError, match="share no topic ids"):
        ragdoll_io.assert_join(answers, gold)


def test_missing_gold_nuggets_for_a_requested_topic_is_an_error(tmp_path: Path) -> None:
    nuggets = _write_jsonl(tmp_path / "gold.jsonl", [_gold_record(qid="58")])

    with pytest.raises(ValueError, match="no gold nuggets for 213"):
        ragdoll_io.gold_nugget_rows(
            nuggets, narratives={"58": NARRATIVE, "213": NARRATIVE}, topic_ids=["58", "213"]
        )


def _write_selected_evidence_handoff(
    path: Path,
    topics: dict[str, list[tuple[str, str]]],
) -> Path:
    generation_topics: list[GenerationTopic] = []
    for topic_id, rows in topics.items():
        group_id = f"{topic_id}-group"
        ranks: dict[str, int] = {}
        evidence: list[EvidencePassage] = []
        clusters: list[SelectedCluster] = []
        for ordinal, (docid, text) in enumerate(rows, start=1):
            evidence_id = f"{topic_id}-e{ordinal}"
            cluster_id = f"{topic_id}-c{ordinal}"
            document_rank = ranks.setdefault(docid, len(ranks) + 1)
            start = ordinal * 100
            evidence.append(
                EvidencePassage(
                    evidence_id=evidence_id,
                    group_id=group_id,
                    cluster_id=cluster_id,
                    cluster_ordinal=ordinal,
                    support_ordinal=1,
                    candidate_kind="extractive",
                    docid=docid,
                    document_rank=document_rank,
                    text=text,
                    document_sha256=sha256(f"{topic_id}:{docid}".encode()).hexdigest(),
                    source_span=EvidenceSourceSpan(
                        start_char=start,
                        end_char=start + len(text),
                        start_byte=start,
                        end_byte=start + len(text.encode("utf-8")),
                    ),
                )
            )
            clusters.append(
                SelectedCluster(
                    cluster_id=cluster_id,
                    ordinal=ordinal,
                    representative_evidence_id=evidence_id,
                    evidence_ids=(evidence_id,),
                )
            )
        generation_topics.append(
            GenerationTopic(
                topic_id=topic_id,
                narrative=NARRATIVE,
                groups=(
                    EvidenceGroup(
                        group_id=group_id,
                        kind="generated_subnarrative",
                        text=f"Evidence for {topic_id}.",
                        selected_clusters=tuple(clusters),
                    ),
                ),
                evidence=tuple(evidence),
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
                retrieval_run_id="test-selected-evidence",
                producer_revision="test-revision",
            ),
            topics=tuple(generation_topics),
        ),
    )
    return path


def _write_generation_identity(
    path: Path,
    handoff_path: Path,
    *,
    run_id: str = "dev-spike",
) -> Path:
    manifest = json.loads(handoff_path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "identity_version": 6,
                "handoff_schema_version": manifest["schema_version"],
                "handoff_manifest_sha256": manifest["manifest_sha256"],
                "prompt_contract_version": "selected_evidence_one_shot_v1",
                "run_id": run_id,
                "selected_topics": [
                    {
                        "topic_id": topic["topic_id"],
                        "context_sha256": topic["context_sha256"],
                    }
                    for topic in manifest["topics"]
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_handoff_support_rows_reject_a_manifest_not_bound_to_generation_identity(
    tmp_path: Path,
) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])
    generated_handoff = _write_selected_evidence_handoff(
        tmp_path / "generated" / "generation_handoff_manifest.json",
        {
            "58": [
                ("climbmix-a", "Generator-visible A."),
                ("climbmix-b", "Generator-visible B."),
            ]
        },
    )
    wrong_handoff = _write_selected_evidence_handoff(
        tmp_path / "wrong" / "generation_handoff_manifest.json",
        {
            "58": [
                ("climbmix-a", "Different selected A."),
                ("climbmix-b", "Different selected B."),
            ]
        },
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json", generated_handoff
    )

    with pytest.raises(ValueError, match="handoff_manifest_sha256"):
        ragdoll_io.selected_evidence_support_rows(
            submission,
            wrong_handoff,
            identity,
        )


def test_handoff_support_rows_reject_a_generation_identity_for_another_run(
    tmp_path: Path,
) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {
            "58": [
                ("climbmix-a", "Generator-visible A."),
                ("climbmix-b", "Generator-visible B."),
            ]
        },
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json",
        handoff,
        run_id="another-run",
    )

    with pytest.raises(ValueError, match="run_id"):
        ragdoll_io.selected_evidence_support_rows(submission, handoff, identity)


def test_handoff_support_rows_reject_a_non_selected_evidence_identity(
    tmp_path: Path,
) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {
            "58": [
                ("climbmix-a", "Generator-visible A."),
                ("climbmix-b", "Generator-visible B."),
            ]
        },
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json", handoff
    )
    payload = json.loads(identity.read_text(encoding="utf-8"))
    payload["prompt_contract_version"] = "legacy_full_document_v1"
    identity.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt_contract_version"):
        ragdoll_io.selected_evidence_support_rows(submission, handoff, identity)


def test_handoff_support_rows_reject_an_identity_for_another_handoff_schema(
    tmp_path: Path,
) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {
            "58": [
                ("climbmix-a", "Generator-visible A."),
                ("climbmix-b", "Generator-visible B."),
            ]
        },
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json", handoff
    )
    payload = json.loads(identity.read_text(encoding="utf-8"))
    payload["handoff_schema_version"] = "another_handoff_schema"
    identity.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="handoff_schema_version"):
        ragdoll_io.selected_evidence_support_rows(submission, handoff, identity)


def test_handoff_support_rows_reject_a_tampered_topic_context_identity(
    tmp_path: Path,
) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {
            "58": [
                ("climbmix-a", "Generator-visible A."),
                ("climbmix-b", "Generator-visible B."),
            ]
        },
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json", handoff
    )
    payload = json.loads(identity.read_text(encoding="utf-8"))
    payload["selected_topics"][0]["context_sha256"] = "f" * 64
    identity.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="context_sha256"):
        ragdoll_io.selected_evidence_support_rows(submission, handoff, identity)


def test_handoff_support_rows_reject_a_different_topic_narrative(
    tmp_path: Path,
) -> None:
    record = _submission_record()
    record["metadata"]["narrative"] = "A different topic with the same ID."
    submission = _write_jsonl(tmp_path / "rag.jsonl", [record])
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {
            "58": [
                ("climbmix-a", "Generator-visible A."),
                ("climbmix-b", "Generator-visible B."),
            ]
        },
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json", handoff
    )

    with pytest.raises(ValueError, match="narrative"):
        ragdoll_io.selected_evidence_support_rows(submission, handoff, identity)


def test_handoff_support_rows_use_exact_selected_passages_in_evidence_order(
    tmp_path: Path,
) -> None:
    record = _submission_record()
    record["answer"] = [
        {"text": "Selected evidence supports this.", "citations": [0]}
    ]
    submission = _write_jsonl(tmp_path / "rag.jsonl", [record])
    before = submission.read_bytes()
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {
            "58": [
                ("climbmix-a", "Passage beyond the document head."),
                ("climbmix-a", "Passage beyond the document head."),
                ("climbmix-a", "Second selected passage."),
                ("climbmix-b", "Uncited selected passage."),
            ]
        },
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json", handoff
    )

    rows = ragdoll_io.selected_evidence_support_rows(submission, handoff, identity)

    assert rows[0]["segments"] == {
        "climbmix-a": "Passage beyond the document head.\n\nSecond selected passage."
    }
    assert submission.read_bytes() == before


def test_handoff_support_rows_keep_shared_docids_bound_to_each_topic(
    tmp_path: Path,
) -> None:
    def record(topic_id: str) -> dict[str, Any]:
        return {
            "metadata": {
                "narrative_id": topic_id,
                "narrative": NARRATIVE,
                "run_id": "dev-spike",
            },
            "references": ["shared"],
            "answer": [{"text": f"answer for {topic_id}", "citations": [0]}],
        }

    submission = _write_jsonl(
        tmp_path / "rag.jsonl", [record("topic-a"), record("topic-b")]
    )
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {
            "topic-a": [("shared", "topic A selected evidence")],
            "topic-b": [("shared", "topic B selected evidence")],
        },
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json", handoff
    )

    rows = ragdoll_io.selected_evidence_support_rows(submission, handoff, identity)

    assert rows[0]["segments"] == {"shared": "topic A selected evidence"}
    assert rows[1]["segments"] == {"shared": "topic B selected evidence"}


def test_handoff_support_rows_reject_cited_docid_outside_topic_evidence(
    tmp_path: Path,
) -> None:
    submission = _write_jsonl(tmp_path / "rag.jsonl", [_submission_record()])
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {"58": [("climbmix-a", "Selected evidence A.")]},
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json", handoff
    )

    with pytest.raises(ValueError, match=r"58.*climbmix-b"):
        ragdoll_io.selected_evidence_support_rows(submission, handoff, identity)


def test_handoff_support_rows_reject_a_citation_outside_references(
    tmp_path: Path,
) -> None:
    """A foreign docid citation would be silently dropped from RAGDoll's denominator."""
    record = _submission_record()
    record["answer"][0]["citations"] = ["climbmix-elsewhere"]
    submission = _write_jsonl(tmp_path / "rag.jsonl", [record])
    handoff = _write_selected_evidence_handoff(
        tmp_path / "generation_handoff_manifest.json",
        {
            "58": [
                ("climbmix-a", "Generator-visible A."),
                ("climbmix-b", "Generator-visible B."),
            ]
        },
    )
    identity = _write_generation_identity(
        tmp_path / "generation_identity.json", handoff
    )

    with pytest.raises(ValueError, match="not in references"):
        ragdoll_io.selected_evidence_support_rows(submission, handoff, identity)


def _support_judgment(
    *,
    task_id: str,
    statement: str,
    citation: str,
    sentence_index: int,
    citation_index: int,
    docid: str,
    status: str = "completed",
    support_label: str | None = "FS",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "statement": statement,
        "citation": citation,
        "status": status,
        "support_label": support_label,
        "metadata": {
            "topic_id": "58",
            "run_id": "dev-spike",
            "sentence_index": sentence_index,
            "citation_index": citation_index,
            "docid": docid,
        },
    }


def test_validate_support_judgments_requires_every_completed_task(
    tmp_path: Path,
) -> None:
    support_input = _write_jsonl(
        tmp_path / "support_input.jsonl",
        [
            {
                "topic_id": "58",
                "run_id": "dev-spike",
                "metadata": _submission_record()["metadata"],
                "references": ["climbmix-a", "climbmix-b"],
                "segments": {
                    "climbmix-a": "Generator-visible passage A.",
                    "climbmix-b": "Generator-visible passage B.",
                },
                "answer": _submission_record()["answer"],
            }
        ],
    )
    judgments = _write_jsonl(
        tmp_path / "support_judgments.jsonl",
        [
            _support_judgment(
                task_id="dev-spike:58:s0:c0",
                statement="Nuclear power emits little carbon.",
                citation="Generator-visible passage A.",
                sentence_index=0,
                citation_index=0,
                docid="climbmix-a",
            ),
            _support_judgment(
                task_id="dev-spike:58:s1:c0",
                statement="Waste storage remains unresolved.",
                citation="Generator-visible passage B.",
                sentence_index=1,
                citation_index=0,
                docid="climbmix-b",
            ),
        ],
    )

    assert ragdoll_io.validate_support_judgments(support_input, judgments) == 2

    failed_rows = list(ragdoll_io._read_jsonl(judgments))
    failed_rows[1]["status"] = "failed"
    failed_rows[1]["support_label"] = None
    _write_jsonl(judgments, failed_rows)
    with pytest.raises(ValueError, match="missing 1 expected task"):
        ragdoll_io.validate_support_judgments(support_input, judgments)

    retried_rows = list(ragdoll_io._read_jsonl(judgments))
    retried_rows.append(
        _support_judgment(
            task_id="dev-spike:58:s1:c0",
            statement="Waste storage remains unresolved.",
            citation="Generator-visible passage B.",
            sentence_index=1,
            citation_index=0,
            docid="climbmix-b",
        )
    )
    _write_jsonl(judgments, retried_rows)
    assert ragdoll_io.validate_support_judgments(support_input, judgments) == 2

    duplicate_completed = list(ragdoll_io._read_jsonl(judgments))
    duplicate_completed.append(duplicate_completed[-1])
    _write_jsonl(judgments, duplicate_completed)
    with pytest.raises(ValueError, match="duplicate completed task"):
        ragdoll_io.validate_support_judgments(support_input, judgments)

    unknown_status = list(ragdoll_io._read_jsonl(judgments))[:2]
    unknown_status[1]["status"] = "retrying"
    unknown_status[1]["support_label"] = None
    _write_jsonl(judgments, unknown_status)
    with pytest.raises(ValueError, match="invalid judgment status"):
        ragdoll_io.validate_support_judgments(support_input, judgments)


def test_validate_support_judgments_rejects_missing_or_wrong_run_rows(
    tmp_path: Path,
) -> None:
    support_input = _write_jsonl(
        tmp_path / "support_input.jsonl",
        [
            {
                "topic_id": "58",
                "run_id": "dev-spike",
                "metadata": _submission_record()["metadata"],
                "references": ["climbmix-a"],
                "segments": {"climbmix-a": "Generator-visible passage A."},
                "answer": [
                    {"text": "Nuclear power emits little carbon.", "citations": [0]}
                ],
            }
        ],
    )
    judgments = _write_jsonl(tmp_path / "support_judgments.jsonl", [])
    with pytest.raises(ValueError, match="missing 1 expected task"):
        ragdoll_io.validate_support_judgments(support_input, judgments)

    _write_jsonl(
        judgments,
        [
            _support_judgment(
                task_id="another-run:58:s0:c0",
                statement="Nuclear power emits little carbon.",
                citation="Generator-visible passage A.",
                sentence_index=0,
                citation_index=0,
                docid="climbmix-a",
            )
        ],
    )
    with pytest.raises(ValueError, match="unexpected task another-run"):
        ragdoll_io.validate_support_judgments(support_input, judgments)


def test_select_passages_counts_the_joiner_against_the_budget() -> None:
    document = (
        ("alpha beta gamma " * 60)
        + ("nuclear fission uranium reactor " * 30)
        + ("delta epsilon " * 80)
    )
    for budget in (50, 240, 300):
        out = dev_rag_inputs.select_passages(
            document, "nuclear fission uranium", budget_words=budget
        )
        assert len(out.split()) <= budget, budget


def test_select_passages_honours_a_budget_smaller_than_one_window() -> None:
    document = ("filler words here " * 100) + ("nuclear fission uranium " * 20)
    out = dev_rag_inputs.select_passages(document, "nuclear fission", budget_words=40)
    # Previously every window was rejected and the document opening was returned instead.
    assert "fission" in out
    assert len(out.split()) <= 40


def test_select_passages_uses_the_repo_semantic_chunker() -> None:
    """Chunking is delegated, not hand-rolled; an injected chunker must be honoured."""

    class RecordingChunker:
        def __init__(self) -> None:
            self.calls = 0

        def split_text(self, text: str, *, document_id: str):
            self.calls += 1
            from trec_rag.chunking import TextChunk

            parts = text.split(" | ")
            offset = 0
            chunks = []
            for index, part in enumerate(parts):
                chunks.append(
                    TextChunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}:{index}",
                        text=part,
                        start_char=offset,
                        end_char=offset + len(part),
                    )
                )
                offset += len(part) + 3
            return chunks

    chunker = RecordingChunker()
    document = " | ".join(["filler " * 30, "nuclear fission uranium reactor core", "more filler " * 20])
    out = dev_rag_inputs.select_passages(
        document, "nuclear fission uranium", budget_words=40, chunker=chunker
    )

    assert chunker.calls == 1
    assert "fission" in out
    assert len(out.split()) <= 40


def test_select_passages_rejects_a_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="budget_words must be a positive integer"):
        dev_rag_inputs.select_passages("a b c", "a", budget_words=0)


def _archive(topic_id: str, count: int) -> dict[str, Any]:
    return {
        "topic_id": topic_id,
        "hits": count,
        "response": {
            "query": {"text": NARRATIVE},
            "candidates": [
                {
                    "docid": f"shard_{index:05d}",
                    "rank": index + 1,
                    "score": 50.0 - index,
                    "doc": f"document text {index}",
                }
                for index in range(count)
            ],
        },
    }


def test_duplicate_docids_are_collapsed_and_ranks_stay_dense(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    archive = _archive("58", 3)
    # A repeated docid at a worse rank must not produce a rank gap.
    archive["response"]["candidates"].append(
        {"docid": "shard_00000", "rank": 4, "score": 10.0, "doc": "duplicate"}
    )
    (cache_dir / "58__original__climbmix_bm25__abc123.json").write_text(
        json.dumps(archive), encoding="utf-8"
    )
    run_path = tmp_path / "run.tsv"

    dev_rag_inputs.build(
        topic_ids=["58"],
        cache_dir=cache_dir,
        run_path=run_path,
        documents_path=tmp_path / "documents.jsonl",
        run_id="dev-spike",
        depth=10,
    )

    ranks = [int(line.split()[3]) for line in run_path.read_text().splitlines()]
    assert ranks == [1, 2, 3]


def test_select_passages_prefers_relevant_windows_over_the_document_opening() -> None:
    boilerplate = "Menu Home About Contact Subscribe Newsletter " * 40
    evidence = "Nuclear fission splits uranium nuclei and releases heat for steam. " * 6
    filler = "unrelated commentary about gardening " * 80
    document = boilerplate + evidence + filler

    selected = dev_rag_inputs.select_passages(
        document, "nuclear fission uranium", budget_words=150
    )

    assert len(selected.split()) <= 150
    assert "fission" in selected and "uranium" in selected
    # Head truncation would have returned only the navigation block.
    assert "fission" not in " ".join(document.split()[:150])


def test_select_passages_returns_short_documents_untouched() -> None:
    document = "A short document about nuclear power."
    assert (
        dev_rag_inputs.select_passages(document, "nuclear", budget_words=100) == document
    )


def test_select_passages_falls_back_when_no_query_term_matches() -> None:
    document = " ".join(f"word{i}" for i in range(400))
    out = dev_rag_inputs.select_passages(document, "nuclear fission", budget_words=50)
    # No window matches, so the opening is used rather than returning nothing.
    assert out.split() == document.split()[:50]


def test_document_rows_apply_the_passage_budget(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    long_text = "Nuclear fission releases heat. " + ("padding words here " * 500)
    archive = {
        "topic_id": "58",
        "response": {
            "query": {"text": "nuclear fission"},
            "candidates": [
                {"docid": "shard_1", "rank": 1, "score": 9.0, "doc": long_text}
            ],
        },
    }
    (cache_dir / "58__original__climbmix_bm25__abc.json").write_text(
        json.dumps(archive), encoding="utf-8"
    )
    documents_path = tmp_path / "documents.jsonl"

    dev_rag_inputs.build(
        topic_ids=["58"],
        cache_dir=cache_dir,
        run_path=tmp_path / "run.tsv",
        documents_path=documents_path,
        run_id="dev-spike",
        depth=1,
        passage_words=60,
    )

    row = json.loads(documents_path.read_text().splitlines()[0])
    assert len(row["candidates"][0]["doc"].split()) <= 60


def test_missing_archive_names_the_topic(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with pytest.raises(ValueError, match="213: no archived retrieval response"):
        dev_rag_inputs.archive_path(cache_dir, "213")
