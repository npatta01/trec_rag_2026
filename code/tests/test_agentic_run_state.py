from __future__ import annotations

import base64
import json
import stat
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import pytest

from trec_rag.agentic_generation_export import (
    prepare_agentic_topic_projection,
    serialize_agentic_retrieval_topic,
)
from trec_rag.agentic_run_state import (
    RUN_PLAN_FILENAME,
    TOPIC_SEAL_FILENAME,
    AgenticRunStateError,
    SubmoduleRevision,
    allocate_topic_attempt,
    create_run_plan,
    load_run_plan,
    load_sealed_topics,
    load_topic_seal,
    resume_run_plan,
    seal_topic_success,
    select_run_topics,
    serialize_run_plan,
    topic_records_receipt_payload,
)
from trec_rag.deepagent_evidence import (
    EvidenceCoverageReport,
    EvidenceReference,
    NeedReport,
    NuggetReport,
)
from trec_rag.document_store import DocumentStore
from trec_rag.generation_handoff import (
    deserialize_generation_topic,
    serialize_generation_topic,
)
from trec_rag.pipeline_models import RankedCandidate
from trec_rag.topic_passage_search import SourceDocument, SourcePassage
from trec_rag.topic_records import TopicEvidenceSnapshot, TopicRecordsReceipt
from trec_rag.topics import Topic


CONFIG_BYTES = (
    "schema_version: agentic_retrieval_config_v1\n"
    "retrieval_mode: agentic\n"
    "experiment:\n  id: rag26_agentic_v1\n"
).encode("utf-8")
RUN_ID = "rag26_agentic_v1"
OFFICIAL_TOPICS_SHA256 = "a" * 64
SOURCE_REVISION = "1" * 40
SUBMODULES = (
    SubmoduleRevision(path="ragdoll", revision="2" * 40),
    SubmoduleRevision(path="trec-rag-skills", revision="3" * 40),
)
TOPICS = (
    Topic(
        id="rag2026-0",
        title="unused",
        narrative="Explique les causes du café ☕ documenté.",
    ),
    Topic(
        id="rag2026-1",
        title="unused",
        narrative="Ω consequences of the documented event.",
    ),
)


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _create(work_dir: Path, **overrides: object):
    arguments: dict[str, object] = {
        "work_dir": work_dir,
        "run_id": RUN_ID,
        "config_bytes": CONFIG_BYTES,
        "topics": TOPICS,
        "official_topics_sha256": OFFICIAL_TOPICS_SHA256,
        "source_revision": SOURCE_REVISION,
        "submodule_revisions": SUBMODULES,
    }
    arguments.update(overrides)
    return create_run_plan(**arguments)  # type: ignore[arg-type]


def _resume(work_dir: Path, **overrides: object):
    arguments: dict[str, object] = {
        "work_dir": work_dir,
        "run_id": RUN_ID,
        "config_bytes": CONFIG_BYTES,
        "topics": TOPICS,
        "official_topics_sha256": OFFICIAL_TOPICS_SHA256,
        "source_revision": SOURCE_REVISION,
        "submodule_revisions": SUBMODULES,
    }
    arguments.update(overrides)
    return resume_run_plan(**arguments)  # type: ignore[arg-type]


# --- projection fixtures -------------------------------------------------


@dataclass(frozen=True)
class _Search:
    candidates: tuple[RankedCandidate, ...]


def _passage(
    *, docid: str, document: str, text: str, passage_id: str, rank: int
) -> SourcePassage:
    start_char = document.index(text)
    end_char = start_char + len(text)
    return SourcePassage(
        passage_id=passage_id,
        docid=docid,
        content_sha256=_digest(document),
        source_rank=rank,
        source_score=1.0 / rank,
        start_char=start_char,
        end_char=end_char,
        start_byte=len(document[:start_char].encode("utf-8")),
        end_byte=len(document[:end_char].encode("utf-8")),
        text_sha256=_digest(text),
        text=text,
        raw_logit=4.0 - rank,
        rank=rank,
        score_cache_key=f"score-{passage_id}",
        scoring_text_sha256=_digest(text),
        chunker_identity={"fixture": "unicode-v1"},
    )


def _source_document(
    docid: str, text: str, passage: SourcePassage, rank: int
) -> SourceDocument:
    return SourceDocument(
        docid=docid,
        content_sha256=_digest(text),
        source_rank=rank,
        source_score=1.0 / rank,
        best_passage_id=passage.passage_id,
        best_passage_raw_logit=passage.raw_logit,
    )


def _projection(tmp_path: Path, topic: Topic):
    """Build one real grounded projection for ``topic`` with Unicode sources."""

    salt = topic.id
    docids = (f"doc-a-{salt}", f"doc-b-{salt}")
    texts = {
        docids[0]: f"Préface café {salt}.\nGrounded α source passage {salt}.\nTail.",
        docids[1]: f"Opening β {salt}.\nIndependent Ω support {salt}.\nTail.",
    }
    passages = {
        "p-a": _passage(
            docid=docids[0],
            document=texts[docids[0]],
            text=f"Grounded α source passage {salt}.",
            passage_id=f"p-a-{salt}",
            rank=2,
        ),
        "p-b": _passage(
            docid=docids[1],
            document=texts[docids[1]],
            text=f"Independent Ω support {salt}.",
            passage_id=f"p-b-{salt}",
            rank=1,
        ),
    }
    store = DocumentStore(tmp_path / "documents")
    for text in texts.values():
        store.admit_text(text)

    nugget = NuggetReport(
        nugget_id="g1",
        text=f"The {salt} event has two independently documented causes.",
        need_ids=("n1",),
        facet_ids=(),
        evidence=(
            EvidenceReference(
                document_id=docids[0],
                snippet_id=passages["p-a"].passage_id,
                page_index=0,
                quote="Model-visible excerpt.",
            ),
            EvidenceReference(
                document_id=docids[1],
                snippet_id=passages["p-b"].passage_id,
                page_index=0,
                quote="Model-visible excerpt.",
            ),
        ),
        contradicts=(),
        support="multi_document",
        superseded_by=None,
        importance="vital",
        support_ratio=1.0,
    )
    report = EvidenceCoverageReport(
        needs=(
            NeedReport(
                need_id="n1",
                narrative_span="What caused the event?",
                question="What caused the event?",
                status="answerable",
                remaining_gap="",
                facet_ids=(),
                nugget_ids=("g1",),
                draft_answer="Grounded draft",
                draft_nugget_ids=("g1",),
            ),
        ),
        facets=(),
        nuggets=(nugget,),
        actions=(),
        searches=(),
        documents=(),
        unresolved_need_ids=(),
        search_count=1,
        inspected_page_count=1,
        state_version=7,
        state_hash="e" * 64,
        terminal_reason="coverage_sufficient",
    )
    snapshot = TopicEvidenceSnapshot(
        facets=(),
        queries=(),
        documents=(
            _source_document(docids[1], texts[docids[1]], passages["p-b"], 1),
            _source_document(docids[0], texts[docids[0]], passages["p-a"], 2),
        ),
        passages=tuple(passages.values()),
        researcher_handoffs=(),
        researcher_evidence=(),
        candidates=(),
        status="complete",
        stopping_reason="coverage_sufficient",
    )
    ranked = tuple(
        RankedCandidate(
            topic_id=topic.id,
            docid=docid,
            rank=rank,
            score=1.0 / rank,
            text=texts[docid],
            provenance=[],
        )
        for rank, docid in enumerate((docids[1], docids[0]), start=1)
    )
    return prepare_agentic_topic_projection(
        topic=topic,
        report=report,
        snapshot=snapshot,
        fused_candidates=ranked,
        searches=(_Search(ranked),),
        document_store=store,
        official_topics_sha256=OFFICIAL_TOPICS_SHA256,
    )


def _row_counts(*, document_count: int) -> dict[str, int]:
    return {
        "candidate": 0,
        "candidate_passage_link": 0,
        "candidate_span": 0,
        "document_binding": document_count,
        "passage": document_count,
        "query_facet": 0,
        "query_identity": 0,
        "query_passage": 0,
        "researcher_evidence": 0,
        "researcher_facet_update": 0,
        "researcher_handoff": 0,
        "retrieval_candidate": 0,
        "stage_seal": 1,
        "subnarrative_identity": 0,
        "topic_completion": 1,
        "topic_identity": 1,
    }


def _receipt(projection) -> dict[str, object]:
    document_sha256s = sorted(
        candidate.document_sha256
        for candidate in projection.full_text_candidates
    )
    return {
        "database_bytes": 4096,
        "database_sha256": "b" * 64,
        "document_sha256s": document_sha256s,
        "manifest_bytes": 512,
        "manifest_sha256": "f" * 64,
        "row_counts": _row_counts(document_count=len(document_sha256s)),
        "run_id": RUN_ID,
        "schema_version": "topic-records-v4",
        "semantic_sha256": "c" * 64,
        "topic_id": projection.topic_id,
    }


def _seal(
    work_dir: Path,
    plan,
    projection,
    *,
    attempt=None,
    status: str = "complete",
    stopping_reason: str = "coverage_sufficient",
    synthesis_outcome: str = "coordinator_selected",
    records_receipt: dict[str, object] | None = None,
):
    if attempt is None:
        attempt = allocate_topic_attempt(
            work_dir=work_dir, plan=plan, topic_id=projection.topic_id
        )
    return seal_topic_success(
        work_dir=work_dir,
        plan=plan,
        projection=projection,
        attempt=attempt,
        status=status,
        stopping_reason=stopping_reason,
        synthesis_outcome=synthesis_outcome,
        records_receipt=records_receipt
        if records_receipt is not None
        else _receipt(projection),
    )


# --- run plan ------------------------------------------------------------


def test_run_plan_create_publishes_a_private_self_authenticating_plan(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"

    plan = _create(work)

    plan_path = work / RUN_PLAN_FILENAME
    body = plan_path.read_bytes()
    assert body == serialize_run_plan(plan)
    assert stat.S_IMODE(plan_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(work.stat().st_mode) == 0o700
    assert not (work / "topics").exists()

    payload = json.loads(body.decode("utf-8"))
    assert payload["run_id"] == RUN_ID
    assert base64.b64decode(payload["config"]["base64"].encode("ascii")) == CONFIG_BYTES
    assert payload["config"]["sha256"] == sha256(CONFIG_BYTES).hexdigest()
    assert payload["topics"]["source_sha256"] == OFFICIAL_TOPICS_SHA256
    assert payload["topics"]["planned"] == [
        {"topic_id": topic.id, "narrative_sha256": _digest(topic.narrative)}
        for topic in TOPICS
    ]
    assert payload["source"] == {
        "revision": SOURCE_REVISION,
        "submodules": [
            {"path": row.path, "revision": row.revision} for row in SUBMODULES
        ],
    }
    recomputed = {key: value for key, value in payload.items() if key != "plan_sha256"}
    assert payload["plan_sha256"] == sha256(
        json.dumps(
            recomputed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    assert plan.planned_topic_ids == ("rag2026-0", "rag2026-1")
    assert plan.config_bytes == CONFIG_BYTES
    assert load_run_plan(work) == plan


def test_run_plan_create_is_exclusive_but_allows_explicit_identical_idempotence(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    first = _create(work)

    with pytest.raises(AgenticRunStateError, match="already exists"):
        _create(work)

    assert _create(work, allow_existing_identical=True) == first

    with pytest.raises(AgenticRunStateError, match="config bytes"):
        _create(
            work,
            allow_existing_identical=True,
            config_bytes=CONFIG_BYTES + b"# drift\n",
        )


def test_run_plan_create_rejects_unsafe_or_repeated_topic_identity(
    tmp_path: Path,
) -> None:
    unsafe = Topic(id="../escape", title="unused", narrative="Narrative.")
    with pytest.raises(AgenticRunStateError, match="topic id"):
        _create(tmp_path / "unsafe", topics=(unsafe,))

    repeated = (TOPICS[0], TOPICS[0])
    with pytest.raises(AgenticRunStateError, match="repeat"):
        _create(tmp_path / "repeated", topics=repeated)

    with pytest.raises(AgenticRunStateError, match="at least one topic"):
        _create(tmp_path / "empty", topics=())


def test_run_plan_rejects_non_sha1_source_and_submodule_revisions(
    tmp_path: Path,
) -> None:
    with pytest.raises(AgenticRunStateError, match="source revision"):
        _create(tmp_path / "source", source_revision="1" * 64)

    with pytest.raises(AgenticRunStateError, match="submodule revision"):
        SubmoduleRevision("ragdoll", "2" * 64)


def test_run_plan_load_rejects_an_authenticated_non_sha1_revision(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    _create(work)
    path = work / RUN_PLAN_FILENAME
    payload = json.loads(path.read_bytes().decode("utf-8"))
    payload["source"]["revision"] = "1" * 64
    payload["plan_sha256"] = sha256(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "plan_sha256"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path.write_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )

    with pytest.raises(AgenticRunStateError, match="source revision"):
        load_run_plan(work)


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"config_bytes": CONFIG_BYTES + b"# drift\n"}, "config bytes"),
        ({"run_id": "rag26_agentic_v2"}, "run id"),
        (
            {
                "topics": (
                    replace(TOPICS[0], narrative="Edited narrative."),
                    TOPICS[1],
                )
            },
            "narrative",
        ),
        ({"topics": (TOPICS[1], TOPICS[0])}, "cohort"),
        ({"topics": (TOPICS[0],)}, "cohort"),
        ({"official_topics_sha256": "d" * 64}, "topic source"),
        ({"source_revision": "9" * 40}, "source revision"),
        (
            {"submodule_revisions": (SUBMODULES[0], SUBMODULES[1], SUBMODULES[0])},
            "submodule",
        ),
        ({"submodule_revisions": (SUBMODULES[1], SUBMODULES[0])}, "submodule"),
    ),
)
def test_run_plan_resume_refuses_every_recorded_identity_drift(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    work = tmp_path / "work"
    plan = _create(work)

    assert _resume(work) == plan

    with pytest.raises(AgenticRunStateError, match=match):
        _resume(work, **overrides)


def test_run_plan_resume_refuses_tampered_plan_bytes(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _create(work)
    plan_path = work / RUN_PLAN_FILENAME
    payload = json.loads(plan_path.read_bytes().decode("utf-8"))
    payload["run_id"] = "rag26_agentic_v2"
    plan_path.chmod(0o600)
    plan_path.write_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )

    with pytest.raises(AgenticRunStateError, match="plan digest"):
        load_run_plan(work)

    plan_path.write_bytes(b"{not json")
    with pytest.raises(AgenticRunStateError, match="run plan"):
        load_run_plan(work)


def test_run_plan_resume_requires_an_existing_plan(tmp_path: Path) -> None:
    with pytest.raises(AgenticRunStateError, match="missing"):
        load_run_plan(tmp_path / "work")


def test_run_plan_selectors_narrow_execution_without_changing_the_cohort(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    plan = _create(work)

    everything = select_run_topics(work_dir=work, plan=plan)
    assert everything.planned_topic_ids == ("rag2026-0", "rag2026-1")
    assert everything.completed_topic_ids == ()
    assert everything.execute_topic_ids == ("rag2026-0", "rag2026-1")

    targeted = select_run_topics(
        work_dir=work, plan=plan, topic_ids=("rag2026-1",)
    )
    assert targeted.execute_topic_ids == ("rag2026-1",)
    assert targeted.planned_topic_ids == plan.planned_topic_ids

    with pytest.raises(AgenticRunStateError, match="outside the run plan"):
        select_run_topics(work_dir=work, plan=plan, topic_ids=("rag2026-7",))

    with pytest.raises(AgenticRunStateError, match="at least one topic"):
        select_run_topics(work_dir=work, plan=plan, topic_ids=())

    _seal(work, plan, _projection(tmp_path, TOPICS[0]))
    after = select_run_topics(work_dir=work, plan=plan)
    assert after.completed_topic_ids == ("rag2026-0",)
    assert after.execute_topic_ids == ("rag2026-1",)
    assert after.planned_topic_ids == plan.planned_topic_ids
    assert load_run_plan(work).planned_topic_ids == plan.planned_topic_ids

    sealed_target = select_run_topics(
        work_dir=work, plan=plan, topic_ids=("rag2026-0", "rag2026-1")
    )
    assert sealed_target.execute_topic_ids == ("rag2026-1",)


# --- per-topic attempts and seals ---------------------------------------


def test_topic_attempts_are_monotonic_and_retain_earlier_failed_material(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    plan = _create(work)

    first = allocate_topic_attempt(work_dir=work, plan=plan, topic_id="rag2026-0")
    assert first.attempt_number == 1
    first.write_artifact("failure.json", b'{"reason":"zero_grounded_nuggets"}\n')
    assert stat.S_IMODE(first.path.stat().st_mode) == 0o700
    assert stat.S_IMODE((first.path / "failure.json").stat().st_mode) == 0o600

    second = allocate_topic_attempt(work_dir=work, plan=plan, topic_id="rag2026-0")
    assert second.attempt_number == 2
    assert second.path != first.path
    assert (first.path / "failure.json").read_bytes() == (
        b'{"reason":"zero_grounded_nuggets"}\n'
    )
    assert select_run_topics(work_dir=work, plan=plan).completed_topic_ids == ()

    with pytest.raises(AgenticRunStateError, match="outside the run plan"):
        allocate_topic_attempt(work_dir=work, plan=plan, topic_id="rag2026-7")

    _seal(work, plan, _projection(tmp_path, TOPICS[0]), attempt=second)
    with pytest.raises(AgenticRunStateError, match="already sealed"):
        allocate_topic_attempt(work_dir=work, plan=plan, topic_id="rag2026-0")
    assert (first.path / "failure.json").exists()


def test_topic_seal_authenticates_projection_generation_and_ledger_receipt(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    plan = _create(work)
    projection = _projection(tmp_path, TOPICS[0])

    seal = _seal(work, plan, projection, stopping_reason="budget_exhausted")

    topic_dir = work / "topics" / "rag2026-0"
    assert stat.S_IMODE(topic_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((topic_dir / TOPIC_SEAL_FILENAME).stat().st_mode) == 0o600
    assert seal.status == "complete"
    assert seal.stopping_reason == "budget_exhausted"
    assert seal.synthesis_outcome == "coordinator_selected"
    assert seal.attempt_number == 1
    assert seal.topic_id == "rag2026-0"
    assert seal.narrative_sha256 == _digest(TOPICS[0].narrative)
    assert seal.retrieval_topic_bytes == serialize_agentic_retrieval_topic(projection)
    assert seal.generation_topic_bytes == serialize_generation_topic(
        projection.generation_topic
    )
    assert seal.retrieval_topic_sha256 == projection.retrieval_topic_sha256
    assert json.loads(seal.records_receipt_bytes.decode("utf-8")) == _receipt(
        projection
    )

    loaded = load_topic_seal(work_dir=work, plan=plan, topic_id="rag2026-0")
    assert loaded == seal
    reloaded_topic = deserialize_generation_topic(loaded.generation_topic_bytes)
    assert reloaded_topic == projection.generation_topic
    assert reloaded_topic.narrative == TOPICS[0].narrative
    assert "café" in reloaded_topic.narrative
    assert any("Ω" in row.text for row in reloaded_topic.evidence)
    assert json.loads(loaded.retrieval_topic_bytes.decode("utf-8")) == (
        projection.retrieval_payload()
    )
    assert seal.full_text_record_sha256 == sha256(
        json.dumps(
            projection.full_text_record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert {row.name for row in seal.artifacts} == {
        "retrieval_topic.json",
        "generation_topic.json",
        "topic_records_receipt.json",
    }
    for row in seal.artifacts:
        published = topic_dir / row.name
        assert published.stat().st_size == row.bytes
        assert sha256(published.read_bytes()).hexdigest() == row.sha256
    receipt_artifact = next(
        row for row in seal.artifacts
        if row.name == "topic_records_receipt.json"
    )
    assert receipt_artifact.sha256 == sha256(
        seal.records_receipt_bytes
    ).hexdigest()


def test_topic_seal_refuses_non_success_or_off_plan_material(tmp_path: Path) -> None:
    work = tmp_path / "work"
    plan = _create(work)
    projection = _projection(tmp_path, TOPICS[0])

    with pytest.raises(AgenticRunStateError, match="complete"):
        _seal(work, plan, projection, status="incomplete")

    with pytest.raises(AgenticRunStateError, match="stopping_reason"):
        _seal(work, plan, projection, stopping_reason="")

    with pytest.raises(AgenticRunStateError, match="synthesis_outcome"):
        _seal(work, plan, projection, synthesis_outcome="Not An Identifier")

    other = _projection(tmp_path, Topic(id="rag2026-9", title="x", narrative="Other."))
    with pytest.raises(AgenticRunStateError, match="outside the run plan"):
        _seal(work, plan, other)

    edited = _projection(
        tmp_path, replace(TOPICS[1], narrative="A narrative the plan never recorded.")
    )
    with pytest.raises(AgenticRunStateError, match="narrative"):
        _seal(work, plan, edited)

    assert not (work / "topics" / "rag2026-0" / TOPIC_SEAL_FILENAME).exists()


@pytest.mark.parametrize(
    "case",
    (
        "missing_field",
        "unknown_field",
        "wrong_schema",
        "bad_database_bytes",
        "bad_manifest_sha256",
        "bad_document_type",
        "unsorted_documents",
        "missing_row_count",
        "bad_row_count",
    ),
)
def test_topic_seal_rejects_malformed_topic_records_receipts(
    tmp_path: Path, case: str
) -> None:
    work = tmp_path / "work"
    plan = _create(work)
    projection = _projection(tmp_path, TOPICS[0])
    receipt = _receipt(projection)

    if case == "missing_field":
        receipt.pop("manifest_bytes")
    elif case == "unknown_field":
        receipt["extra"] = "not canonical"
    elif case == "wrong_schema":
        receipt["schema_version"] = "topic-records-v3"
    elif case == "bad_database_bytes":
        receipt["database_bytes"] = True
    elif case == "bad_manifest_sha256":
        receipt["manifest_sha256"] = "F" * 64
    elif case == "bad_document_type":
        receipt["document_sha256s"] = [
            receipt["document_sha256s"][0],
            123,
        ]
    elif case == "unsorted_documents":
        receipt["document_sha256s"] = list(
            reversed(receipt["document_sha256s"])
        )
    elif case == "missing_row_count":
        row_counts = dict(receipt["row_counts"])
        row_counts.pop("stage_seal")
        receipt["row_counts"] = row_counts
    elif case == "bad_row_count":
        row_counts = dict(receipt["row_counts"])
        row_counts["document_binding"] = False
        receipt["row_counts"] = row_counts
    else:  # pragma: no cover - protects the case table
        raise AssertionError(case)

    with pytest.raises(AgenticRunStateError, match="topic records receipt"):
        _seal(work, plan, projection, records_receipt=receipt)
    assert not (work / "topics" / "rag2026-0" / TOPIC_SEAL_FILENAME).exists()


def test_topic_seal_requires_every_projected_document_in_receipt_closure(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    plan = _create(work)
    projection = _projection(tmp_path, TOPICS[0])
    receipt = _receipt(projection)
    receipt["document_sha256s"] = receipt["document_sha256s"][:1]

    with pytest.raises(AgenticRunStateError, match="document closure"):
        _seal(work, plan, projection, records_receipt=receipt)
    assert not (work / "topics" / "rag2026-0" / TOPIC_SEAL_FILENAME).exists()


def test_topic_seal_republish_is_idempotent_while_conflicting_bytes_fail(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    plan = _create(work)
    projection = _projection(tmp_path, TOPICS[0])
    attempt = allocate_topic_attempt(work_dir=work, plan=plan, topic_id="rag2026-0")

    first = _seal(work, plan, projection, attempt=attempt)
    body = (work / "topics" / "rag2026-0" / TOPIC_SEAL_FILENAME).read_bytes()

    assert _seal(work, plan, projection, attempt=attempt) == first
    assert (work / "topics" / "rag2026-0" / TOPIC_SEAL_FILENAME).read_bytes() == body

    with pytest.raises(AgenticRunStateError, match="conflict"):
        _seal(
            work,
            plan,
            projection,
            attempt=attempt,
            stopping_reason="coverage_sufficient"
            if first.stopping_reason != "coverage_sufficient"
            else "budget_exhausted",
        )
    with pytest.raises(AgenticRunStateError, match="conflict"):
        _seal(
            work,
            plan,
            projection,
            attempt=attempt,
            records_receipt={**_receipt(projection), "database_sha256": "e" * 64},
        )
    assert (work / "topics" / "rag2026-0" / TOPIC_SEAL_FILENAME).read_bytes() == body
    assert load_topic_seal(work_dir=work, plan=plan, topic_id="rag2026-0") == first


def test_topic_seal_completes_a_crashed_publication_that_never_wrote_its_manifest(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    plan = _create(work)
    projection = _projection(tmp_path, TOPICS[0])
    attempt = allocate_topic_attempt(work_dir=work, plan=plan, topic_id="rag2026-0")
    _seal(work, plan, projection, attempt=attempt)

    manifest = work / "topics" / "rag2026-0" / TOPIC_SEAL_FILENAME
    manifest.unlink()

    assert select_run_topics(work_dir=work, plan=plan).completed_topic_ids == ()
    with pytest.raises(AgenticRunStateError, match="not sealed"):
        load_topic_seal(work_dir=work, plan=plan, topic_id="rag2026-0")

    resealed = _seal(work, plan, projection, attempt=attempt)
    assert manifest.exists()
    assert select_run_topics(work_dir=work, plan=plan).completed_topic_ids == (
        "rag2026-0",
    )
    assert load_topic_seal(work_dir=work, plan=plan, topic_id="rag2026-0") == resealed


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("missing_payload", "missing"),
        ("corrupt_payload", "sha256"),
        ("truncated_payload", "bytes"),
        ("tampered_manifest", "seal digest"),
        ("changed_topic_identity", "topic"),
        ("corrupt_manifest", "topic seal"),
    ),
)
def test_topic_seal_fails_closed_on_corrupt_or_missing_artifacts(
    tmp_path: Path, case: str, match: str
) -> None:
    work = tmp_path / "work"
    plan = _create(work)
    _seal(work, plan, _projection(tmp_path, TOPICS[0]))
    topic_dir = work / "topics" / "rag2026-0"
    manifest = topic_dir / TOPIC_SEAL_FILENAME

    def _rewrite(path: Path, body: bytes) -> None:
        path.chmod(0o600)
        path.write_bytes(body)

    if case == "missing_payload":
        (topic_dir / "generation_topic.json").unlink()
    elif case == "corrupt_payload":
        payload = topic_dir / "retrieval_topic.json"
        body = bytearray(payload.read_bytes())
        body[10] = body[10] ^ 0x20
        _rewrite(payload, bytes(body))
    elif case == "truncated_payload":
        payload = topic_dir / "topic_records_receipt.json"
        _rewrite(payload, payload.read_bytes()[:-3])
    elif case == "tampered_manifest":
        payload = json.loads(manifest.read_bytes().decode("utf-8"))
        payload["stopping_reason"] = "budget_exhausted"
        _rewrite(
            manifest,
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            + b"\n",
        )
    elif case == "changed_topic_identity":
        payload = json.loads(manifest.read_bytes().decode("utf-8"))
        payload["topic"]["topic_id"] = "rag2026-1"
        digest = sha256(
            json.dumps(
                {key: value for key, value in payload.items() if key != "seal_sha256"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload["seal_sha256"] = digest
        _rewrite(
            manifest,
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            + b"\n",
        )
    elif case == "corrupt_manifest":
        _rewrite(manifest, b"{not json")
    else:  # pragma: no cover - protects the case table
        raise AssertionError(case)

    with pytest.raises(AgenticRunStateError, match=match):
        load_topic_seal(work_dir=work, plan=plan, topic_id="rag2026-0")
    with pytest.raises(AgenticRunStateError):
        select_run_topics(work_dir=work, plan=plan)


@pytest.mark.parametrize("case", ("unknown_field", "missing_projected_document"))
def test_topic_resume_revalidates_receipt_after_all_hashes_are_recomputed(
    tmp_path: Path, case: str
) -> None:
    work = tmp_path / "work"
    plan = _create(work)
    projection = _projection(tmp_path, TOPICS[0])
    _seal(work, plan, projection)
    topic_dir = work / "topics" / "rag2026-0"
    receipt_path = topic_dir / "topic_records_receipt.json"
    manifest_path = topic_dir / TOPIC_SEAL_FILENAME

    receipt = json.loads(receipt_path.read_bytes().decode("utf-8"))
    if case == "unknown_field":
        receipt["extra"] = "resealed attacker-controlled value"
    else:
        receipt["document_sha256s"] = receipt["document_sha256s"][:1]
    receipt_body = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    receipt_path.write_bytes(receipt_body)

    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    receipt_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["name"] == "topic_records_receipt.json"
    )
    receipt_artifact["bytes"] = len(receipt_body)
    receipt_artifact["sha256"] = sha256(receipt_body).hexdigest()
    manifest["seal_sha256"] = sha256(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key != "seal_sha256"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_bytes(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )

    with pytest.raises(AgenticRunStateError, match="topic records receipt"):
        load_topic_seal(work_dir=work, plan=plan, topic_id="rag2026-0")


def test_topic_seal_is_bound_to_its_own_run_plan(tmp_path: Path) -> None:
    work = tmp_path / "work"
    plan = _create(work)
    _seal(work, plan, _projection(tmp_path, TOPICS[0]))

    other_work = tmp_path / "other"
    other_plan = _create(other_work, run_id="rag26_agentic_v2")
    with pytest.raises(AgenticRunStateError, match="run plan"):
        load_topic_seal(work_dir=work, plan=other_plan, topic_id="rag2026-0")


def test_sealed_topics_load_in_plan_order_for_aggregation(tmp_path: Path) -> None:
    work = tmp_path / "work"
    plan = _create(work)

    _seal(work, plan, _projection(tmp_path, TOPICS[1]))
    with pytest.raises(AgenticRunStateError, match="not sealed"):
        load_sealed_topics(work_dir=work, plan=plan)

    _seal(work, plan, _projection(tmp_path, TOPICS[0]))
    seals = load_sealed_topics(work_dir=work, plan=plan)
    assert [row.topic_id for row in seals] == ["rag2026-0", "rag2026-1"]
    assert [
        deserialize_generation_topic(row.generation_topic_bytes).narrative
        for row in seals
    ] == [TOPICS[0].narrative, TOPICS[1].narrative]
    assert seals == load_sealed_topics(work_dir=work, plan=plan)


def test_topic_records_receipt_payload_is_canonical_and_hashable() -> None:
    receipt = TopicRecordsReceipt(
        database_sha256="b" * 64,
        database_bytes=4096,
        semantic_sha256="c" * 64,
        topic_id="rag2026-0",
        run_id=RUN_ID,
        document_sha256s=("d" * 64, "e" * 64),
        row_counts=_row_counts(document_count=2),
        schema_version="topic-records-v4",
        manifest_sha256="f" * 64,
        manifest_bytes=512,
    )

    payload = topic_records_receipt_payload(receipt)

    assert payload == {
        "database_sha256": "b" * 64,
        "database_bytes": 4096,
        "semantic_sha256": "c" * 64,
        "topic_id": "rag2026-0",
        "run_id": RUN_ID,
        "document_sha256s": ["d" * 64, "e" * 64],
        "row_counts": _row_counts(document_count=2),
        "schema_version": "topic-records-v4",
        "manifest_sha256": "f" * 64,
        "manifest_bytes": 512,
    }
    assert json.dumps(payload, sort_keys=True)
